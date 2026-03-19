"""
Refinement Adapter — Multi-pass LLM refinement driven by the bus.
=================================================================

Replaces the hand-rolled asyncio.create_task refinement chain in promptly_chat.py
with bus-driven signal flow:

    REFINEMENT_START → refinement handler runs passes → writes to RefinementStore
                     → ConvergenceMonitor watches confidence → emits CONVERGENCE
                     → handler stops early if converged

The endpoint fires REFINEMENT_START and forgets. The adapter does the work.
RefinementStore compatibility is preserved — nanoclaw polls the same way.

Usage:
    from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

    handler = RefinementHandler(bus, call_model_fn, get_system_prompt_fn)
    # Auto-subscribes to REFINEMENT_START
    # Writes results to any RefinementStore-compatible dict
"""

import asyncio
import hashlib
import logging
from collections.abc import Callable
from typing import Any

from ..signal import Signal, SignalKind, SignalPriority, TriggerMode
from ..unified_bus import UnifiedBus

logger = logging.getLogger(__name__)


class RefinementHandler:
    """
    Bus-driven multi-pass refinement.

    Subscribes to REFINEMENT_START signals. Each signal triggers a background
    refinement chain that:
    1. Recalls memories (if HoloLoomLite available)
    2. Runs pass 2+ with decreasing temperature
    3. Checks material difference (convergence)
    4. Chains further passes if warranted
    5. Writes results to the refinement store for polling

    The ConvergenceMonitor can also terminate the chain early via CONVERGENCE.
    """

    def __init__(
        self,
        bus: UnifiedBus,
        call_model: Callable,
        get_system_prompt: Callable[[], str],
        refinement_store: Any = None,
        recall_fn: Callable | None = None,
        store_experience_fn: Callable | None = None,
        max_passes: int = 4,
    ):
        """
        Args:
            bus: UnifiedBus to subscribe to
            call_model: async fn(messages, model_spec, temperature, max_tokens) -> dict
            get_system_prompt: fn() -> str
            refinement_store: RefinementStore instance (for polling compat)
            recall_fn: async fn(query) -> str (memory context)
            store_experience_fn: async fn(query, response, room_id, pass_number)
            max_passes: Maximum refinement passes (including pass 1)
        """
        self.bus = bus
        self.call_model = call_model
        self.get_system_prompt = get_system_prompt
        self.store = refinement_store
        self.recall_fn = recall_fn
        self.store_experience_fn = store_experience_fn
        self.max_passes = max_passes

        # Track convergence per execution (set by CONVERGENCE signals)
        self._converged: dict[str, str] = {}  # correlation_id -> reason

        # Subscribe
        self.bus.on(SignalKind.REFINEMENT_START, self._on_refinement_start)
        self.bus.on(SignalKind.CONVERGENCE, self._on_convergence)

    async def _on_convergence(self, signal: Signal) -> Signal | None:
        """Catch convergence signals and mark the execution."""
        correlation_id = signal.correlation_id
        if correlation_id:
            reason = signal.payload.get("reason", "converged")
            self._converged[correlation_id] = reason
        return None

    async def _on_refinement_start(self, signal: Signal) -> Signal | None:
        """Handle REFINEMENT_START — kick off the background refinement chain."""
        payload = signal.payload
        refinement_id = payload.get("refinement_id", "")
        query = payload.get("query", "")
        prior_response = payload.get("prior_response", "")
        room_id = payload.get("room_id", "")
        history = payload.get("history", [])
        complexity = payload.get("complexity", 0.5)
        model_router_fn = payload.get("_model_router_fn")

        correlation_id = signal.correlation_id or signal.id

        # Run the chain in the background (non-blocking)
        asyncio.ensure_future(
            self._run_chain(
                refinement_id=refinement_id,
                query=query,
                prior_response=prior_response,
                room_id=room_id,
                history=history,
                complexity=complexity,
                correlation_id=correlation_id,
                model_router_fn=model_router_fn,
            )
        )

        return None  # No synchronous response

    async def _run_chain(
        self,
        refinement_id: str,
        query: str,
        prior_response: str,
        room_id: str,
        history: list[dict[str, str]],
        complexity: float,
        correlation_id: str,
        model_router_fn: Callable | None = None,
    ) -> None:
        """
        Run the full refinement chain: pass 2 through max_passes.

        Each pass:
        1. Recall memories
        2. Build refinement prompt (bar rises each pass)
        3. Call LLM with decreasing temperature
        4. Check material difference
        5. Write to store
        6. Chain next pass or stop
        """
        current_response = prior_response
        current_id = refinement_id

        for pass_number in range(2, self.max_passes + 1):
            # Check bus-driven convergence
            if correlation_id in self._converged:
                reason = self._converged.pop(correlation_id)
                if self.store:
                    self.store.converged(current_id, f"bus convergence: {reason}")
                logger.info("Refinement %s: bus convergence at pass %d — %s",
                            current_id, pass_number, reason)
                break

            try:
                result = await self._run_single_pass(
                    refinement_id=current_id,
                    query=query,
                    prior_response=current_response,
                    room_id=room_id,
                    history=history,
                    pass_number=pass_number,
                    complexity=complexity,
                    correlation_id=correlation_id,
                    model_router_fn=model_router_fn,
                )

                if result is None:
                    # Converged or skipped — stop chaining
                    break

                # Decide whether to chain
                current_response = result["refined"]
                next_pass = pass_number + 1
                should_chain = (
                    next_pass <= self.max_passes
                    and complexity >= 0.5
                )

                if should_chain:
                    next_id = hashlib.sha256(
                        f"{room_id}:{query}:{current_id}:{next_pass}".encode()
                    ).hexdigest()[:12]
                    if self.store:
                        self.store.create(
                            next_id, query, current_response, room_id,
                            pass_number=next_pass,
                        )
                    # Complete current with pointer to next
                    if self.store:
                        self.store.complete(
                            current_id, current_response,
                            next_refinement_id=next_id,
                            model=result.get("model"),
                        )
                    current_id = next_id
                else:
                    # Final pass — complete without next pointer
                    if self.store:
                        self.store.complete(
                            current_id, current_response,
                            model=result.get("model"),
                        )
                    break

            except Exception as e:
                if self.store:
                    self.store.skip(current_id, f"error: {e}")
                logger.error("Refinement %s pass %d failed: %s",
                             current_id, pass_number, e)
                break

        # Emit REFINEMENT_COMPLETE
        await self.bus.emit(Signal(
            kind=SignalKind.REFINEMENT_COMPLETE,
            priority=SignalPriority.LOW,
            source="refinement_handler",
            payload={
                "refinement_id": refinement_id,
                "room_id": room_id,
                "final_pass": pass_number if 'pass_number' in dir() else 2,
            },
            correlation_id=correlation_id,
            trigger_mode=TriggerMode.REACTIVE,
        ))

    async def _run_single_pass(
        self,
        refinement_id: str,
        query: str,
        prior_response: str,
        room_id: str,
        history: list[dict[str, str]],
        pass_number: int,
        complexity: float,
        correlation_id: str,
        model_router_fn: Callable | None = None,
    ) -> dict[str, Any] | None:
        """
        Run a single refinement pass. Returns {"refined": str, "model": str}
        or None if converged/skipped.
        """
        # Recall memories
        memory_context = ""
        if self.recall_fn:
            try:
                memory_context = await self.recall_fn(query)
            except Exception:
                pass

        # Store prior as experience
        if self.store_experience_fn:
            try:
                await self.store_experience_fn(
                    query, prior_response, room_id, pass_number - 1,
                )
            except Exception:
                pass

        # Build prompt via MRF stack (graceful degradation to fallback)
        try:
            from hololoom.prompting.unified_mrf import (
                RefinementEngine,
                RefinementStrategyType,
                StrategySelector,
            )
            if not hasattr(self, "_mrf_refinement"):
                self._mrf_refinement = RefinementEngine()
                self._mrf_selector = StrategySelector()

            # Classify query type for strategy selection
            lower_q = query.lower()
            if any(k in lower_q for k in ("```", "def ", "class ", "code", "bug")):
                q_type = "code"
            elif any(k in lower_q for k in ("explain", "what is", "how does", "why")):
                q_type = "factual"
            elif any(k in lower_q for k in ("compare", "difference", "tradeoff")):
                q_type = "analytical"
            else:
                q_type = "general"

            # Strategy progression: pass 2 = auto, pass 3 = elegance, pass 4+ = hofstadter
            if pass_number == 2:
                strategy = self._mrf_selector.select_strategy(
                    q_type,
                    [RefinementStrategyType.VERIFY,
                     RefinementStrategyType.CRITIQUE,
                     RefinementStrategyType.REFINE],
                )
            elif pass_number == 3:
                strategy = RefinementStrategyType.ELEGANCE
            else:
                strategy = RefinementStrategyType.HOFSTADTER

            refinement_prompt = self._mrf_refinement.get_refinement_prompt(
                strategy, query, prior_response, iteration=pass_number - 1,
            )
            if memory_context:
                refinement_prompt += f"\n\n## Recalled context\n{memory_context}"
            refinement_prompt += (
                "\n\nWrite the improved response directly, as a follow-up "
                "message in chat. Be concise. No meta-framing.\n\n"
                "If the response was already good enough, respond with "
                "exactly: PASS"
            )
            logger.info("Refinement %s: MRF strategy=%s pass=%d",
                        refinement_id, strategy.value, pass_number)

        except Exception as mrf_err:
            logger.debug("MRF unavailable in bus adapter: %s", mrf_err)
            # Fallback: original ad-hoc prompt
            if pass_number == 2:
                instruction = (
                    "Look at your initial response critically. If you can "
                    "improve it — add something missing, correct an error, "
                    "sharpen the framing — do so."
                )
            else:
                instruction = (
                    f"This is refinement pass {pass_number}. Only continue "
                    "if there's a genuine correction or sharper framing."
                )

            refinement_prompt = (
                f"You gave this response to a question. Now refine it.\n\n"
                f"## Original question\n{query}\n\n"
                f"## Your most recent response\n{prior_response}\n"
                f"{memory_context}\n\n"
                f"## Instructions\n{instruction}\n\n"
                f"Write the improved response directly. Be concise. "
                f"No meta-framing.\n\n"
                f"If good enough, respond with exactly: PASS"
            )

        messages = [{"role": "system", "content": self.get_system_prompt()}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": refinement_prompt})

        # Decreasing temperature — converge toward precision
        temp = max(0.2, 0.5 - (pass_number - 2) * 0.1)

        # Route to model (later passes favor quality)
        model_spec = None
        model_id = "default"
        if model_router_fn:
            try:
                decision = await model_router_fn(
                    query, speed_weight=max(0.0, 0.6 - pass_number * 0.2),
                )
                model_spec = decision.model
                model_id = model_spec.id if model_spec else "default"
            except Exception:
                pass

        import os
        max_tokens = int(os.environ.get("PROMPTLY_MAX_TOKENS", "8192"))
        data = await self.call_model(
            messages, model_spec=model_spec,
            temperature=temp, max_tokens=max_tokens,
        )
        import re
        raw_content = data.get("message", {}).get("content", "").strip()
        refined = re.sub(r"<think>[\s\S]*?</think>\s*", "", raw_content).strip()

        # Model said PASS?
        if refined == "PASS" or not refined:
            if self.store:
                self.store.converged(
                    refinement_id,
                    f"model said PASS at pass {pass_number}",
                    model=model_id,
                )
            logger.info("Refinement %s: converged (model PASS, pass %d)",
                        refinement_id, pass_number)
            return None

        # Material difference check
        if not self._is_materially_different(prior_response, refined):
            if self.store:
                self.store.converged(
                    refinement_id,
                    f"converged at pass {pass_number}",
                    model=model_id,
                )
            logger.info("Refinement %s: converged (not different, pass %d)",
                        refinement_id, pass_number)

            # Emit step for convergence monitor
            await self.bus.emit(Signal(
                kind=SignalKind.STEP_COMPLETE,
                priority=SignalPriority.LOW,
                source="refinement_handler",
                payload={
                    "execution_id": refinement_id,
                    "step_id": f"pass_{pass_number}",
                    "step_type": "refine",
                    "status": "success",
                    "confidence": 0.95,  # High confidence = converged
                },
                correlation_id=correlation_id,
                trigger_mode=TriggerMode.REACTIVE,
            ))

            return None

        # Store refined experience
        if self.store_experience_fn:
            try:
                await self.store_experience_fn(query, refined, room_id, pass_number)
            except Exception:
                pass

        # Emit step completion for convergence monitor
        await self.bus.emit(Signal(
            kind=SignalKind.STEP_COMPLETE,
            priority=SignalPriority.LOW,
            source="refinement_handler",
            payload={
                "execution_id": refinement_id,
                "step_id": f"pass_{pass_number}",
                "step_type": "refine",
                "status": "success",
                "confidence": complexity * 0.7,  # Not yet converged
                "duration_ms": 0,
            },
            correlation_id=correlation_id,
            trigger_mode=TriggerMode.REACTIVE,
        ))

        logger.info("Refinement %s: pass %d complete (%d chars, model=%s)",
                     refinement_id, pass_number, len(refined), model_id)

        return {"refined": refined, "model": model_id}

    @staticmethod
    def _is_materially_different(pass1: str, pass2: str) -> bool:
        """Check if refinement adds real value. Mirrors promptly_chat logic."""
        if not pass2 or not pass2.strip():
            return False
        p1 = pass1.strip().lower()
        p2 = pass2.strip().lower()
        if p1 == p2:
            return False
        p1_words = set(p1.split())
        p2_words = set(p2.split())
        if p1_words and p2_words:
            intersection = len(p1_words & p2_words)
            union = len(p1_words | p2_words)
            if union > 0 and intersection / union > 0.8:
                return False
        if len(pass2.strip()) < 20:
            return False
        return True


__all__ = ["RefinementHandler"]
