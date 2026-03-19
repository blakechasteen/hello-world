"""
Refinement subsystem — RefinementStore, multi-pass background refinement,
complexity estimation, and bus signal helpers.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from .config import (
    DEFAULT_MAX_TOKENS,
    MAX_PASSES,
    OLLAMA_MODEL,
    REFINEMENT_TTL,
)
from .hololoom_bridge import _get_hololoom, _recall_memories, _store_experience
from .llm import _classify_query_type, _get_mrf, _strip_thinking, call_model
from .soul import get_system_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# Refinement Store
# ============================================================================

class RefinementStore:
    """Holds pending/completed refinements keyed by ID. Supports multi-pass chaining."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def create(self, refinement_id: str, query: str, prior_response: str,
               room_id: str, pass_number: int = 2) -> None:
        self._store[refinement_id] = {
            "status": "pending",
            "query": query,
            "prior_response": prior_response,
            "room_id": room_id,
            "pass": pass_number,
            "refinement": None,
            "next_refinement_id": None,
            "created_at": time.time(),
        }

    def complete(self, refinement_id: str, refinement: str | None,
                 next_refinement_id: str | None = None,
                 model: str | None = None,
                 strategy: str | None = None) -> None:
        if refinement_id in self._store:
            self._store[refinement_id]["status"] = "complete"
            self._store[refinement_id]["refinement"] = refinement
            self._store[refinement_id]["next_refinement_id"] = next_refinement_id
            self._store[refinement_id]["model"] = model
            if strategy:
                self._store[refinement_id]["strategy"] = strategy

    def converged(self, refinement_id: str, reason: str = "",
                  model: str | None = None) -> None:
        """Mark as converged — no further passes needed."""
        if refinement_id in self._store:
            self._store[refinement_id]["status"] = "converged"
            self._store[refinement_id]["reason"] = reason
            self._store[refinement_id]["model"] = model

    def skip(self, refinement_id: str, reason: str = "") -> None:
        if refinement_id in self._store:
            self._store[refinement_id]["status"] = "skipped"
            self._store[refinement_id]["reason"] = reason

    def get(self, refinement_id: str) -> dict | None:
        self._cleanup()
        return self._store.get(refinement_id)

    def _cleanup(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items()
                   if now - v["created_at"] > REFINEMENT_TTL]
        for k in expired:
            del self._store[k]


# Module-level singleton
refinements = RefinementStore()


# ============================================================================
# Complexity Heuristics
# ============================================================================

def _estimate_complexity(text: str) -> float:
    """
    Quick heuristic: 0.0 (trivial) to 1.0 (complex).
    Used to decide whether pass 2 is worth firing.
    """
    score = 0.0
    words = text.split()
    word_count = len(words)

    # Length signals
    if word_count > 50:
        score += 0.3
    elif word_count > 20:
        score += 0.15

    # Question depth signals
    lower = text.lower()
    deep_markers = ["how", "why", "explain", "compare", "difference between",
                    "architecture", "design", "tradeoff", "trade-off", "debug",
                    "optimize", "implement", "refactor", "review"]
    shallow_markers = ["hi", "hello", "hey", "thanks", "ok", "yes", "no",
                       "lol", "haha", "cool", "nice"]

    for marker in deep_markers:
        if marker in lower:
            score += 0.15

    for marker in shallow_markers:
        if lower.strip().startswith(marker):
            score -= 0.3

    # Code signals
    if "```" in text or "def " in text or "class " in text:
        score += 0.2

    # Multi-part questions
    if "?" in text and text.count("?") > 1:
        score += 0.15
    if any(text.strip().startswith(f"{i}.") for i in range(1, 6)):
        score += 0.1

    return max(0.0, min(1.0, score))


def _is_materially_different(pass1: str, pass2: str) -> bool:
    """
    Check if the refinement adds real value over pass 1.
    Not just rephrasing — actual new content or corrections.
    """
    if not pass2 or not pass2.strip():
        return False

    # Trivially similar
    p1_clean = pass1.strip().lower()
    p2_clean = pass2.strip().lower()
    if p1_clean == p2_clean:
        return False

    # Word-level jaccard distance — if >80% overlap, it's just rephrasing
    p1_words = set(p1_clean.split())
    p2_words = set(p2_clean.split())
    if p1_words and p2_words:
        intersection = len(p1_words & p2_words)
        union = len(p1_words | p2_words)
        if union > 0 and intersection / union > 0.8:
            return False

    # Pass 2 should be at least slightly substantive
    if len(pass2.strip()) < 20:
        return False

    return True


# ============================================================================
# Bus-Driven Refinement (lazy init, graceful fallback)
# ============================================================================

_refinement_bus = None
_refinement_bus_checked = False


def _get_refinement_bus():
    """
    Get the bus with refinement handler wired up.

    Lazy init — first call creates the handler and subscribes it.
    Returns None if bus is unavailable (falls back to direct tasks).
    """
    global _refinement_bus, _refinement_bus_checked

    if _refinement_bus_checked:
        return _refinement_bus

    _refinement_bus_checked = True

    try:
        from hololoom.apps.server.bus_setup import get_bus
        bus = get_bus()
        if bus is None:
            return None

        from hololoom.core.bus.adapters.refinement_adapter import RefinementHandler

        RefinementHandler(
            bus=bus,
            call_model=call_model,
            get_system_prompt=get_system_prompt,
            refinement_store=refinements,
            recall_fn=_recall_memories_for_bus,
            store_experience_fn=_store_experience_for_bus,
            max_passes=MAX_PASSES,
        )

        _refinement_bus = bus
        logger.info("Refinement handler wired to bus")
        return bus

    except Exception as e:
        logger.info("Bus refinement unavailable (using direct tasks): %s", e)
        return None


def _emit_pass1_to_bus(
    query: str, response: str, room_id: str, tokens: int, duration_ms: float,
) -> None:
    """Fire-and-forget: notify bus of pass 1 completion."""
    try:
        bus = _get_refinement_bus()
        if bus is None:
            return
        from hololoom.core.bus import Signal, SignalKind, SignalPriority
        asyncio.ensure_future(bus.emit(Signal(
            kind=SignalKind.EXECUTION_COMPLETE,
            priority=SignalPriority.LOW,
            source="promptly",
            payload={
                "text": query[:200],
                "room_id": room_id,
                "tokens": tokens,
                "duration_ms": round(duration_ms, 1),
                "response_length": len(response),
                "strategy": "direct_pass",
                "confidence": 1.0,
                "success": True,
                "pass": 1,
            },
        )))
    except Exception:
        pass


async def _recall_memories_for_bus(query: str) -> str:
    """Adapter for RefinementHandler — wraps _recall_memories."""
    loom = await _get_hololoom()
    return await _recall_memories(loom, query)


async def _store_experience_for_bus(
    query: str, response: str, room_id: str, pass_number: int,
) -> None:
    """Adapter for RefinementHandler — wraps _store_experience."""
    loom = await _get_hololoom()
    await _store_experience(loom, query, response, room_id, pass_number)


# ============================================================================
# Multi-Pass Background Refinement
# ============================================================================

async def _run_refinement_pass(
    refinement_id: str,
    query: str,
    prior_response: str,
    room_id: str,
    history: list[dict[str, str]],
    pass_number: int,
    complexity: float,
) -> None:
    """
    Single refinement pass. Can chain into the next pass if:
    - The refinement was materially different (converging, not yet done)
    - We haven't hit MAX_PASSES
    - Complexity warrants it
    """
    try:
        loom = await _get_hololoom()
        memory_context = await _recall_memories(loom, query)

        if memory_context:
            logger.info("Pass %d (%s): recalled memories", pass_number, refinement_id)

        # Store prior interaction as experience
        await _store_experience(loom, query, prior_response, room_id, pass_number - 1)

        # Build refinement prompt via MRF stack (or fallback)
        mrf = _get_mrf()
        mrf_strategy_used = None

        if mrf:
            # MRF: Thompson Sampling picks the best strategy for this query type
            query_type = _classify_query_type(query)
            strategies_enum = mrf["strategies"]

            # Map pass number to strategy progression:
            # Pass 2: auto-select (VERIFY, CRITIQUE, REFINE)
            # Pass 3: ELEGANCE (polish what previous pass produced)
            # Pass 4+: HOFSTADTER (meta-level reflection)
            if pass_number == 2:
                available = [
                    strategies_enum.VERIFY,
                    strategies_enum.CRITIQUE,
                    strategies_enum.REFINE,
                ]
                mrf_strategy_used = mrf["selector"].select_strategy(
                    query_type, available,
                )
            elif pass_number == 3:
                mrf_strategy_used = strategies_enum.ELEGANCE
            else:
                mrf_strategy_used = strategies_enum.HOFSTADTER

            # iteration maps to sub-pass within multi-pass strategies (VERIFY, ELEGANCE)
            iteration = pass_number - 1
            refinement_prompt = mrf["refinement"].get_refinement_prompt(
                mrf_strategy_used, query, prior_response, iteration=iteration,
            )

            # Append memory context and convergence instruction
            if memory_context:
                refinement_prompt += f"\n\n## Recalled context\n{memory_context}"
            refinement_prompt += (
                "\n\nWrite the improved response directly, as a follow-up message "
                "in chat. Be concise. No meta-framing.\n\n"
                "If the response was already good enough, respond with exactly: PASS"
            )
            logger.info(
                "Pass %d (%s): MRF strategy=%s query_type=%s",
                pass_number, refinement_id,
                mrf_strategy_used.value, query_type,
            )
        else:
            # Fallback: original ad-hoc prompt
            if pass_number == 2:
                instruction = (
                    "Look at your initial response critically. If you can improve it "
                    "— add something missing, correct an error, sharpen the framing, "
                    "or connect it to recalled context — do so."
                )
            else:
                instruction = (
                    f"This is refinement pass {pass_number}. Your previous response "
                    "was already a refinement. Only continue if there's a genuine "
                    "correction, a missing connection, or a sharper framing."
                )

            refinement_prompt = f"""You gave this response to a question. Now refine it.

## Original question
{query}

## Your most recent response
{prior_response}
{memory_context}

## Instructions
{instruction}

Write the improved response directly, as if it's a follow-up message in chat. Be concise. Start with the substance, not "Upon reflection..." or similar meta-framing.

If the response was already good enough, respond with exactly: PASS"""

        messages = [{"role": "system", "content": get_system_prompt()}]
        messages.extend(history[-6:])  # Last 3 turns for context
        messages.append({"role": "user", "content": refinement_prompt})

        # Lower temperature with each pass — converge toward precision
        temp = max(0.2, 0.5 - (pass_number - 2) * 0.1)

        # Route refinement to a (potentially better) model
        # Later passes favor quality over speed
        from ..model_router import get_router
        model_router = get_router()
        refine_decision = None
        try:
            refine_decision = await model_router.route(
                query, speed_weight=max(0.0, 0.6 - pass_number * 0.2),
            )
            model_spec = refine_decision.model
        except Exception:
            model_spec = None  # Fall back to default

        data = await call_model(messages, model_spec=model_spec,
                                temperature=temp, max_tokens=DEFAULT_MAX_TOKENS)
        refined = _strip_thinking(data.get("message", {}).get("content", ""))

        # Feed back to bandit
        if refine_decision:
            model_router.feedback(refine_decision, success=bool(refined and refined != "PASS"))

        model_id = model_spec.id if model_spec else OLLAMA_MODEL

        # Check: did the model say PASS?
        if refined == "PASS" or not refined:
            # MRF feedback: low improvement = strategy wasn't needed
            if mrf and mrf_strategy_used:
                query_type = _classify_query_type(query)
                mrf["selector"].update_from_outcome(
                    query_type, mrf_strategy_used, 0.0,
                )
            refinements.converged(refinement_id, f"model said PASS at pass {pass_number}",
                                  model=model_id)
            logger.info("Pass %d (%s): converged (model PASS, model=%s)",
                        pass_number, refinement_id, model_id)
            return

        # Check: is it materially different?
        if not _is_materially_different(prior_response, refined):
            # MRF feedback: marginal improvement
            if mrf and mrf_strategy_used:
                query_type = _classify_query_type(query)
                mrf["selector"].update_from_outcome(
                    query_type, mrf_strategy_used, 0.03,
                )
            refinements.converged(refinement_id, f"converged at pass {pass_number}",
                                  model=model_id)
            logger.info("Pass %d (%s): converged (not materially different, model=%s)",
                        pass_number, refinement_id, model_id)
            return

        # MRF feedback: material improvement = strategy worked
        if mrf and mrf_strategy_used:
            query_type = _classify_query_type(query)
            mrf["selector"].update_from_outcome(
                query_type, mrf_strategy_used, 0.3,
            )

        # Store the refined experience
        await _store_experience(loom, query, refined, room_id, pass_number)

        # Decide whether to chain another pass
        next_pass = pass_number + 1
        should_chain = (
            next_pass <= MAX_PASSES
            and complexity >= 0.5  # Only chain beyond pass 2 for complex queries
        )

        next_id = None
        if should_chain:
            raw = f"{room_id}:{query}:{refinement_id}:{next_pass}"
            next_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
            refinements.create(next_id, query, refined, room_id, pass_number=next_pass)

        # Complete this pass with a pointer to the next
        refinements.complete(refinement_id, refined, next_refinement_id=next_id,
                             model=model_id,
                             strategy=mrf_strategy_used.value if mrf_strategy_used else "fallback")
        strategy_label = mrf_strategy_used.value if mrf_strategy_used else "fallback"
        logger.info(
            "Pass %d (%s): complete (%d chars, strategy=%s)%s",
            pass_number, refinement_id, len(refined), strategy_label,
            f" -> chaining pass {next_pass}" if next_id else "",
        )

        # Fire the next pass if chained
        if next_id:
            asyncio.create_task(
                _run_refinement_pass(
                    next_id, query, refined, room_id,
                    history, next_pass, complexity,
                )
            )

    except Exception as e:
        refinements.skip(refinement_id, f"error: {e}")
        logger.error("Pass %d (%s) failed: %s", pass_number, refinement_id, e)
