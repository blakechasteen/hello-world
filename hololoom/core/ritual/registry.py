"""
RitualRegistry — central hook dispatcher.

Lives on the Loom Scheduler. Responsibilities:
  - Register HookBindings
  - Validate DAG at boot (produces/consumes chains, no cycles)
  - Dispatch rituals with admission control
  - Handle failure policies
  - Write RitualOutcomes to memory
  - Record journal events at each checkpoint
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .budget import check_admission
from .hooks import HookEvent
from .memory import write_outcome
from .policies import FailurePolicy
from .types import (
    HookBinding,
    Priority,
    Ritual,
    RitualContext,
    RitualOutcome,
    RitualResult,
)

logger = logging.getLogger(__name__)


class RitualDAGError(Exception):
    """Raised at boot when the produces/consumes DAG is invalid."""
    pass


class EscalationRequired(Exception):
    """Raised when a ritual's ESCALATE policy fires."""
    def __init__(self, ritual_id: str, ctx: RitualContext):
        self.ritual_id = ritual_id
        self.ctx = ctx
        super().__init__(f"Escalation required for ritual '{ritual_id}'")


class RitualRegistry:
    """
    Central hook dispatcher. Lives on the Loom Scheduler.

    Register HookBindings, validate DAG at boot, dispatch rituals
    with admission control, handle failure policies, write outcomes.
    """

    def __init__(
        self,
        journal: Any = None,
        deferred_queue: Callable[[HookBinding, RitualContext], None] | None = None,
        snapshot_builder: Any = None,   # SnapshotBuilder (ThreadPrimer)
        stuck_detector: Any = None,     # StuckDetector (SnagWatcher)
        closure_gate: Any = None,       # ClosureGate (BindingObligation)
        rule_engine: Any = None,        # RuleEngine (PatternCard)
    ) -> None:
        self._bindings: dict[HookEvent, list[HookBinding]] = defaultdict(list)
        self._rituals: dict[str, Ritual] = {}
        self._journal = journal
        self._deferred_queue = deferred_queue
        self._snapshot_builder = snapshot_builder
        self._stuck_detector = stuck_detector
        self._closure_gate = closure_gate
        self._rule_engine = rule_engine

    def register(self, binding: HookBinding) -> None:
        """Register a hook binding. Sorts by priority within each event bucket."""
        self._bindings[binding.trigger].append(binding)
        self._rituals[binding.ritual.id] = binding.ritual
        self._bindings[binding.trigger].sort(key=lambda b: b.priority.value)

    def validate_dag(self) -> None:
        """
        Run at boot. Raises RitualDAGError on:
          - Missing consumes (ritual needs a Bobbin no prior ritual produces)
          - Cycles in produces/consumes graph
        """
        all_produced: set[str] = set()
        for ritual in self._rituals.values():
            all_produced.update(ritual.produces)

        for ritual in self._rituals.values():
            for required in ritual.consumes:
                if required not in all_produced:
                    raise RitualDAGError(
                        f"Ritual '{ritual.id}' consumes '{required}' "
                        f"but no registered ritual produces it"
                    )

        adj: dict[str, set[str]] = defaultdict(set)
        type_to_producer: dict[str, str] = {}
        type_to_consumers: dict[str, list[str]] = defaultdict(list)

        for ritual in self._rituals.values():
            for p in ritual.produces:
                type_to_producer[p] = ritual.id
            for c in ritual.consumes:
                type_to_consumers[c].append(ritual.id)

        for bobbin_type, producer_id in type_to_producer.items():
            for consumer_id in type_to_consumers.get(bobbin_type, []):
                if producer_id != consumer_id:
                    adj[producer_id].add(consumer_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = dict.fromkeys(self._rituals, WHITE)

        def _dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in adj.get(node, set()):
                if color[neighbor] == GRAY:
                    raise RitualDAGError(
                        f"Cycle detected: '{node}' → '{neighbor}' "
                        f"(via produces/consumes chain)"
                    )
                if color[neighbor] == WHITE:
                    _dfs(neighbor)
            color[node] = BLACK

        for rid in self._rituals:
            if color[rid] == WHITE:
                _dfs(rid)

    async def fire(
        self,
        event: HookEvent,
        ctx: RitualContext,
    ) -> dict[str, RitualResult]:
        """
        Dispatch all rituals bound to this event.
        """
        bindings = self._bindings.get(event, [])
        results: dict[str, RitualResult] = {}

        blocking = [b for b in bindings if b.priority == Priority.BLOCKING]
        parallel = [b for b in bindings if b.priority == Priority.PARALLEL]
        deferred = [b for b in bindings if b.priority == Priority.DEFERRED]

        for binding in blocking:
            # BindingObligation: gate BLOCKING dispatches while closures pending
            if self._closure_gate is not None and self._closure_gate.is_gated(binding.ritual.id):
                self._journal_record(binding.ritual, ctx, "CLOSURE_GATED")
                continue

            result = await self._dispatch_one(binding, ctx)
            if result is not None:
                results[binding.ritual.id] = result
                ctx.ritual_outputs.update(result.produced)

                # Open closure requirement if ritual declares one
                if binding.ritual.closure_required_by and self._closure_gate is not None:
                    from .closure import OpenBinding
                    self._closure_gate.open(OpenBinding(
                        opener_ritual_id=binding.ritual.id,
                        closer_ritual_id=binding.ritual.closure_required_by,
                        opened_at=datetime.now(timezone.utc).isoformat(),
                        session_id=ctx.session_id,
                    ))

                # Close closure requirement if this ritual is a closer
                if self._closure_gate is not None:
                    if self._closure_gate.close(binding.ritual.id):
                        self._journal_record(binding.ritual, ctx, "CLOSURE_SATISFIED")

        if parallel:
            parallel_results = await asyncio.gather(
                *[self._dispatch_one(b, ctx) for b in parallel],
                return_exceptions=True,
            )
            for binding, result in zip(parallel, parallel_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Parallel ritual '%s' raised: %s",
                        binding.ritual.id, result,
                    )
                    continue
                if result is not None:
                    results[binding.ritual.id] = result
                    ctx.ritual_outputs.update(result.produced)

        for binding in deferred:
            self._schedule_deferred(binding, ctx)

        # SnagWatcher: post-dispatch stuck detection (session-level)
        if self._stuck_detector is not None and self._journal is not None:
            try:
                detection = self._stuck_detector.check(self._journal, ctx)
                if detection is not None:
                    self._journal_record_detection(detection, ctx)
            except Exception as e:
                logger.debug("Stuck detection failed (non-blocking): %s", e)

        return results

    async def _dispatch_one(
        self,
        binding: HookBinding,
        ctx: RitualContext,
    ) -> RitualResult | None:
        """Dispatch a single ritual with all guards and policy handling."""
        ritual = binding.ritual
        start = time.monotonic()

        self._journal_record(ritual, ctx, "DISPATCH")

        # Guard: condition check
        if binding.condition is not None and not binding.condition(ctx):
            self._journal_record(ritual, ctx, "CONDITION_FAIL")
            return None

        self._journal_record(ritual, ctx, "CONDITION_PASS")

        # Guard: skip_condition check — skips still write outcomes
        if ritual.skip_condition is not None and ritual.skip_condition(ctx):
            self._journal_record(ritual, ctx, "SKIP", {
                "reason": "skip_condition evaluated True",
            })
            await write_outcome(
                RitualOutcome(
                    ritual_id=ritual.id,
                    ritual_version=ritual.version,
                    session_id=ctx.session_id,
                    agent_id=ctx.agent_id,
                    task_context=ctx.task_intent,
                    fired_at=datetime.now(timezone.utc),
                    duration_ms=0,
                    tokens_used=0,
                    produced={},
                    downstream_task_success=None,
                    confidence_delta=0.0,
                    was_skipped=True,
                    skip_reason="skip_condition evaluated True",
                    failure_policy_invoked=None,
                ),
                ctx,
            )
            return None

        # Admission control
        if not check_admission(ritual.budget, ctx.token_budget):
            self._journal_record(ritual, ctx, "ADMISSION_FAIL", {
                "budget_remaining": ctx.token_budget.remaining,
                "ritual_needs": ritual.budget.max_tokens,
            })
            return await self._handle_failure(
                ritual, ctx, Exception("Budget insufficient"),
                start, policy_override=FailurePolicy.SKIP,
            )

        self._journal_record(ritual, ctx, "ADMISSION_PASS")

        # PatternCard: JIT rule activation
        if self._rule_engine is not None:
            try:
                ruleset = self._rule_engine.activate(ctx)
                ctx.active_rules = ruleset.rules
            except Exception as e:
                logger.debug("Rule activation failed (non-blocking): %s", e)

        # ThreadPrimer: pre-inlined context snapshot
        if self._snapshot_builder is not None:
            try:
                snapshot = self._snapshot_builder.build(
                    ctx,
                    active_rules=ctx.active_rules,
                    budget_tokens=200,
                )
                ctx.priming_context = {
                    "recent_outcomes": snapshot.recent_outcomes,
                    "session_history": snapshot.session_history,
                    "skill_hint": snapshot.skill_hint,
                    "domain_priors": snapshot.domain_priors,
                }
                self._journal_record(ritual, ctx, "SNAPSHOT_BUILT", {
                    "token_cost": snapshot.token_cost,
                    "rules_active": len(ctx.active_rules),
                    "sources": len(snapshot.session_history),
                })
            except Exception as e:
                self._journal_record(ritual, ctx, "SNAPSHOT_FAILED", {"error": str(e)})
                logger.debug("Snapshot build failed (non-blocking): %s", e)

        # Execute with timeout
        self._journal_record(ritual, ctx, "BODY_START")
        try:
            result = await asyncio.wait_for(
                ritual.body(ctx),
                timeout=ritual.budget.max_seconds,
            )
        except asyncio.TimeoutError as e:
            self._journal_record(ritual, ctx, "BODY_FAILED", {"error": "timeout"})
            return await self._handle_failure(ritual, ctx, e, start)
        except Exception as e:
            self._journal_record(ritual, ctx, "BODY_FAILED", {"error": str(e)})
            return await self._handle_failure(ritual, ctx, e, start)

        # Success
        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._journal_record(ritual, ctx, "BODY_COMPLETE", {
            "duration_ms": elapsed_ms,
            "produced": list(result.produced.keys()),
            "notes": result.notes,
        })
        await write_outcome(
            RitualOutcome(
                ritual_id=ritual.id,
                ritual_version=ritual.version,
                session_id=ctx.session_id,
                agent_id=ctx.agent_id,
                task_context=ctx.task_intent,
                fired_at=datetime.now(timezone.utc),
                duration_ms=elapsed_ms,
                tokens_used=0,
                produced=result.produced,
                downstream_task_success=None,
                confidence_delta=result.confidence_delta,
                was_skipped=False,
                skip_reason=None,
                failure_policy_invoked=None,
            ),
            ctx,
        )
        self._journal_record(ritual, ctx, "OUTCOME_WRITTEN")

        return result

    async def _handle_failure(
        self,
        ritual: Ritual,
        ctx: RitualContext,
        error: Exception,
        start: float,
        policy_override: FailurePolicy | None = None,
    ) -> RitualResult | None:
        """Apply the ritual's failure policy."""
        policy = policy_override or ritual.failure_policy
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self._journal_record(ritual, ctx, "POLICY_INVOKED", {
            "policy": policy.value,
            "error": str(error),
        })

        if policy == FailurePolicy.SKIP:
            await write_outcome(
                _failed_outcome(ritual, ctx, elapsed_ms, "SKIP"), ctx,
            )
            return None

        elif policy == FailurePolicy.RETRY_ONCE:
            await asyncio.sleep(0.5)
            try:
                result = await asyncio.wait_for(
                    ritual.body(ctx),
                    timeout=ritual.budget.max_seconds,
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)
                await write_outcome(
                    RitualOutcome(
                        ritual_id=ritual.id,
                        ritual_version=ritual.version,
                        session_id=ctx.session_id,
                        agent_id=ctx.agent_id,
                        task_context=ctx.task_intent,
                        fired_at=datetime.now(timezone.utc),
                        duration_ms=elapsed_ms,
                        tokens_used=0,
                        produced=result.produced,
                        downstream_task_success=None,
                        confidence_delta=result.confidence_delta,
                        was_skipped=False,
                        skip_reason=None,
                        failure_policy_invoked="RETRY_ONCE",
                    ),
                    ctx,
                )
                return result
            except Exception:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                await write_outcome(
                    _failed_outcome(ritual, ctx, elapsed_ms, "RETRY_ONCE→SKIP"),
                    ctx,
                )
                return None

        elif policy == FailurePolicy.HALT:
            await write_outcome(
                _failed_outcome(ritual, ctx, elapsed_ms, "HALT"), ctx,
            )
            raise RuntimeError(f"Ritual '{ritual.id}' halted pipeline: {error}")

        elif policy == FailurePolicy.ESCALATE:
            await write_outcome(
                _failed_outcome(ritual, ctx, elapsed_ms, "ESCALATE"), ctx,
            )
            raise EscalationRequired(ritual.id, ctx)

        elif policy == FailurePolicy.LOG_AND_CONTINUE:
            await write_outcome(
                _failed_outcome(ritual, ctx, elapsed_ms, "LOG_AND_CONTINUE"),
                ctx,
            )
            return None

        elif policy == FailurePolicy.COMPENSATE:
            if ritual.compensation_ritual_id:
                compensation = self._rituals.get(ritual.compensation_ritual_id)
                if compensation:
                    try:
                        result = await compensation.body(ctx)
                        return result
                    except Exception as comp_err:
                        logger.warning(
                            "Compensation ritual '%s' failed: %s",
                            ritual.compensation_ritual_id, comp_err,
                        )
            await write_outcome(
                _failed_outcome(ritual, ctx, elapsed_ms, "COMPENSATE→SKIP"),
                ctx,
            )
            return None

        return None

    def _schedule_deferred(
        self, binding: HookBinding, ctx: RitualContext,
    ) -> None:
        """Schedule a deferred ritual for later execution."""
        if self._deferred_queue is not None:
            try:
                self._deferred_queue(binding, ctx)
                logger.debug(
                    "Deferred ritual '%s' queued for later execution",
                    binding.ritual.id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to queue deferred ritual '%s': %s",
                    binding.ritual.id, e,
                )
        else:
            logger.debug(
                "Deferred ritual '%s' dropped — no queue available",
                binding.ritual.id,
            )

    def _journal_record(
        self,
        ritual: Ritual,
        ctx: RitualContext,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a journal event. Fire-and-forget — never blocks dispatch."""
        if self._journal is None:
            return
        try:
            self._journal.record(
                ritual_id=ritual.id,
                ritual_version=ritual.version,
                session_id=ctx.session_id,
                correlation_id=None,
                event_type=event_type,
                payload=payload,
            )
        except Exception as e:
            logger.debug("Journal record failed (non-blocking): %s", e)

    def _journal_record_detection(
        self,
        detection: Any,  # StuckDetection
        ctx: RitualContext,
    ) -> None:
        """Record a stuck detection event."""
        if self._journal is None:
            return
        try:
            self._journal.record(
                ritual_id="__snag_watcher__",
                ritual_version="1.0.0",
                session_id=ctx.session_id,
                correlation_id=None,
                event_type="STUCK_DETECTED",
                payload={
                    "signal": detection.signal.value,
                    "severity": detection.severity,
                    "recommendation": detection.recommendation,
                    "details": detection.details,
                },
            )
        except Exception as e:
            logger.debug("Stuck detection journal failed: %s", e)


def _failed_outcome(
    ritual: Ritual,
    ctx: RitualContext,
    elapsed_ms: int,
    policy: str,
) -> RitualOutcome:
    """Build a RitualOutcome for a failed execution."""
    return RitualOutcome(
        ritual_id=ritual.id,
        ritual_version=ritual.version,
        session_id=ctx.session_id,
        agent_id=ctx.agent_id,
        task_context=ctx.task_intent,
        fired_at=datetime.now(timezone.utc),
        duration_ms=elapsed_ms,
        tokens_used=0,
        produced={},
        downstream_task_success=False,
        confidence_delta=-0.05,
        was_skipped=False,
        skip_reason=None,
        failure_policy_invoked=policy,
    )
