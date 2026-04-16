"""
Smoke test v2 — all four new systems firing together.

ThreadPrimer + SnagWatcher + BindingObligation + PatternCard
"""

import asyncio
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_v2")


async def main():
    logger.info("=" * 60)
    logger.info("SMOKE TEST v2: ThreadPrimer + SnagWatcher + BindingObligation + PatternCard")
    logger.info("=" * 60)

    from hololoom.core.ritual.journal import RitualJournal
    from hololoom.core.ritual.registry import RitualRegistry
    from hololoom.core.ritual.hooks import HookEvent
    from hololoom.core.ritual.types import (
        Ritual, RitualBudget, RitualContext, RitualResult,
        HookBinding, Priority, TokenBudget,
    )
    from hololoom.core.ritual.policies import FailurePolicy
    from hololoom.core.ritual.snapshot import JournalSnapshotBuilder
    from hololoom.core.ritual.stuck import SlidingWindowDetector
    from hololoom.core.ritual.closure import ClosureGate
    from hololoom.core.ritual.rule_engine import RuleEngine, Rule

    # --- Setup: all four systems ---
    journal = RitualJournal(":memory:")
    snapshot_builder = JournalSnapshotBuilder(journal=journal)
    stuck_detector = SlidingWindowDetector(window_size=8, max_session_events=100)
    closure_gate = ClosureGate()

    rule_engine = RuleEngine()
    rule_engine.register(Rule(
        id="always_degrade", domain="universal",
        text="Every dependency is optional.", always_on=True, weight=2.0,
    ))
    rule_engine.register(Rule(
        id="coding_test", domain="coding",
        text="Run tests after changes.", keywords=["fix", "refactor"], weight=1.0,
    ))
    rule_engine.register(Rule(
        id="low_conf_caution", domain="universal",
        text="Be cautious when confidence is low.",
        confidence_below=0.5, weight=1.0,
    ))

    registry = RitualRegistry(
        journal=journal,
        snapshot_builder=snapshot_builder,
        stuck_detector=stuck_detector,
        closure_gate=closure_gate,
        rule_engine=rule_engine,
    )

    # --- Register rituals ---
    async def dawn_body(ctx):
        logger.info("  [dawn] Priming context: %s", list(ctx.priming_context.keys()) if ctx.priming_context else "empty")
        logger.info("  [dawn] Active rules: %s", [r.rule.id for r in ctx.active_rules] if ctx.active_rules else "none")
        return RitualResult(success=True, produced={"SessionPrimeBobbin": {"primed": True}}, confidence_delta=0.05)

    async def health_check_body(ctx):
        logger.info("  [health] Priming context: %s", list(ctx.priming_context.keys()) if ctx.priming_context else "empty")
        logger.info("  [health] Active rules: %s", [r.rule.id for r in ctx.active_rules] if ctx.active_rules else "none")
        return RitualResult(success=True, produced={"HealthScoreBobbin": {"score": 9.8}}, confidence_delta=0.03)

    async def dusk_body(ctx):
        logger.info("  [dusk] Consolidating session...")
        return RitualResult(success=True, produced={"ConsolidationBobbin": {}}, confidence_delta=0.0)

    dawn = Ritual(
        id="dawn_prime", version="1.0.0", domain="universal",
        intent="Prime session", author="system", consumes=[], produces=["SessionPrimeBobbin"],
        failure_policy=FailurePolicy.LOG_AND_CONTINUE,
        budget=RitualBudget(max_tokens=500, max_seconds=10.0, priority_class="standard"),
        body=dawn_body,
        closure_required_by="dusk_consolidate",
    )

    health = Ritual(
        id="pre_code_health_check", version="1.0.0", domain="coding",
        intent="Check code health", author="system", consumes=[], produces=["HealthScoreBobbin"],
        failure_policy=FailurePolicy.LOG_AND_CONTINUE,
        budget=RitualBudget(max_tokens=300, max_seconds=10.0, priority_class="standard"),
        body=health_check_body,
    )

    dusk = Ritual(
        id="dusk_consolidate", version="1.0.0", domain="universal",
        intent="Consolidate session", author="system", consumes=[], produces=["ConsolidationBobbin"],
        failure_policy=FailurePolicy.LOG_AND_CONTINUE,
        budget=RitualBudget(max_tokens=500, max_seconds=10.0, priority_class="standard"),
        body=dusk_body,
    )

    registry.register(HookBinding(ritual=dawn, trigger=HookEvent.SESSION_START, condition=None, priority=Priority.BLOCKING))
    registry.register(HookBinding(ritual=health, trigger=HookEvent.TASK_RECEIVED, condition=None, priority=Priority.BLOCKING))
    registry.register(HookBinding(ritual=dusk, trigger=HookEvent.SESSION_END, condition=None, priority=Priority.BLOCKING))

    ctx = RitualContext(
        agent_id="smoke-agent", agent_role="coder", domain="coding",
        task_id="smoke-1", task_type="code_modify", task_intent="refactor the weaving module",
        task_complexity="standard", yarn=None, warp=None,
        confidence=0.8, confidence_history=[0.8],
        ritual_outputs={}, ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0, escalation_path=[], session_id="smoke-v2",
        closure_gate=closure_gate,
    )

    # --- Phase 1: SESSION_START (dawn opens binding) ---
    logger.info("")
    logger.info(">>> SESSION_START (dawn_prime — opens binding for dusk)")
    results = await registry.fire(HookEvent.SESSION_START, ctx)
    logger.info("Results: %s", {k: v.success for k, v in results.items()})
    logger.info("Closure gate open? %s", closure_gate.has_mandatory_open())
    logger.info("Pending: %s", [(b.opener_ritual_id, b.closer_ritual_id) for b in closure_gate.pending()])

    # --- Phase 2: TASK_RECEIVED (health check — should be gated!) ---
    logger.info("")
    logger.info(">>> TASK_RECEIVED (health_check — is it gated by closure?)")
    results = await registry.fire(HookEvent.TASK_RECEIVED, ctx)
    logger.info("Results: %s", {k: v.success for k, v in results.items()})
    if not results:
        logger.info("GATED! health_check was blocked by BindingObligation (dawn→dusk pending)")
    else:
        logger.info("health_check ran (closure may not have gated it)")

    # --- Phase 3: SESSION_END (dusk closes binding) ---
    logger.info("")
    logger.info(">>> SESSION_END (dusk_consolidate — closes binding)")
    results = await registry.fire(HookEvent.SESSION_END, ctx)
    logger.info("Results: %s", {k: v.success for k, v in results.items()})
    logger.info("Closure gate open? %s", closure_gate.has_mandatory_open())

    # --- Phase 4: TASK_RECEIVED again (should work now) ---
    logger.info("")
    logger.info(">>> TASK_RECEIVED again (should fire now that binding is closed)")
    results = await registry.fire(HookEvent.TASK_RECEIVED, ctx)
    logger.info("Results: %s", {k: v.success for k, v in results.items()})

    # --- Journal dump ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("JOURNAL DUMP")
    logger.info("=" * 60)
    events = journal.replay_session("smoke-v2")
    for i, e in enumerate(events, 1):
        logger.info(
            "  %2d. [%-18s] %-30s %s",
            i, e.event_type, e.ritual_id, e.payload or "",
        )
    logger.info("")
    logger.info("Total events: %d", len(events))

    # --- Snag check ---
    logger.info("")
    logger.info("SnagWatcher: %s", stuck_detector.check(journal, ctx) or "all clear")

    journal.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SMOKE TEST v2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
