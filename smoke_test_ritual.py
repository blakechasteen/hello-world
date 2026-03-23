"""
Smoke test — prove the ritual system breathes.

Starts the daemon in-process, fires a signal through the bus,
watches the ritual dispatch, and prints the full event journal.
"""

import asyncio
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("smoke_test")


async def main():
    # --- Phase 1: Import and wire ---
    logger.info("=" * 60)
    logger.info("SMOKE TEST: Ritual Grammar + Runtime")
    logger.info("=" * 60)

    from hololoom.core.bus.unified_bus import UnifiedBus
    from hololoom.core.bus.signal import Signal, SignalKind, SignalPriority, TriggerMode
    from hololoom.core.ritual.journal import RitualJournal
    from hololoom.core.ritual.registry import RitualRegistry
    from hololoom.core.runtime import HoloLoomRuntime
    from hololoom.core.runtime.ritual_layer import RitualLayer
    from hololoom.core.runtime.daemon import HoloLoomDaemon

    # --- Phase 2: Create runtime with bus + journal ---
    bus = UnifiedBus()
    journal = RitualJournal(":memory:")

    runtime = HoloLoomRuntime(bus=bus, journal=journal)
    layer = RitualLayer(journal=journal, register_defaults=True)
    runtime.register_layer(layer)

    # --- Phase 3: Start the runtime ---
    logger.info("")
    logger.info("Starting runtime...")
    await runtime.start()

    status = runtime.status()
    logger.info("Runtime status: %s", json.dumps(status, indent=2))

    # --- Phase 4: Fire a QUERY signal through the bus ---
    logger.info("")
    logger.info("Firing QUERY signal (simulating a code_modify task)...")

    query_signal = Signal(
        kind=SignalKind.QUERY,
        priority=SignalPriority.NORMAL,
        source="smoke_test",
        payload={
            "agent_id": "smoke-agent",
            "agent_role": "coder",
            "domain": "coding",
            "task_type": "code_modify",
            "task_intent": "refactor the weaving module",
            "task_complexity": "standard",
            "session_id": "smoke-session-1",
        },
        trigger_mode=TriggerMode.ACTIVE,
    )

    await bus.emit(query_signal)

    # Give the async dispatch a moment
    await asyncio.sleep(0.1)

    # --- Phase 5: Check the journal ---
    logger.info("")
    logger.info("Replaying journal for session 'smoke-session-1'...")
    events = journal.replay_session("smoke-session-1")

    if events:
        for e in events:
            logger.info(
                "  [%s] %s v%s — %s %s",
                e.event_type,
                e.ritual_id,
                e.ritual_version,
                e.payload if e.payload else "",
                f"({e.created_at})" if e.created_at else "",
            )
        logger.info("")
        logger.info("Total journal events: %d", len(events))
    else:
        logger.warning("No journal events found — ritual trigger may not have fired")
        logger.info("Checking registry directly...")

        # Direct fire as fallback
        from hololoom.core.ritual.hooks import HookEvent
        from hololoom.core.ritual.types import RitualContext, TokenBudget

        ctx = RitualContext(
            agent_id="smoke-agent", agent_role="coder", domain="coding",
            task_id="smoke-task", task_type="code_modify",
            task_intent="refactor the weaving module",
            task_complexity="standard", yarn=None, warp=None,
            confidence=0.8, confidence_history=[0.8],
            ritual_outputs={}, ritual_log=[],
            token_budget=TokenBudget(total=5000, used=0),
            time_budget=60.0, escalation_path=[], session_id="smoke-session-1",
        )

        results = await layer.registry.fire(HookEvent.TASK_RECEIVED, ctx)
        logger.info("Direct fire results: %s", {k: v.success for k, v in results.items()})

        events = journal.replay_session("smoke-session-1")
        for e in events:
            logger.info(
                "  [%s] %s v%s — %s",
                e.event_type, e.ritual_id, e.ritual_version,
                e.payload if e.payload else "",
            )
        logger.info("Total journal events: %d", len(events))

    # --- Phase 6: Fire a SESSION_START signal ---
    logger.info("")
    logger.info("Firing SESSION_START (dawn_prime)...")

    from hololoom.core.ritual.hooks import HookEvent
    from hololoom.core.ritual.types import RitualContext, TokenBudget

    ctx = RitualContext(
        agent_id="smoke-agent", agent_role="coder", domain="coding",
        task_id="smoke-task-2", task_type="session",
        task_intent="start session", task_complexity="standard",
        yarn=None, warp=None, confidence=0.8, confidence_history=[0.8],
        ritual_outputs={}, ritual_log=[],
        token_budget=TokenBudget(total=5000, used=0),
        time_budget=60.0, escalation_path=[], session_id="smoke-session-1",
    )

    results = await layer.registry.fire(HookEvent.SESSION_START, ctx)
    logger.info("SESSION_START results: %s", {k: v.success for k, v in results.items()})

    # --- Phase 7: Daemon status check ---
    logger.info("")
    logger.info("Testing daemon request handler...")
    daemon = HoloLoomDaemon(runtime=runtime)
    from hololoom.core.runtime.daemon_protocol import DaemonRequest

    req = DaemonRequest(type="status")
    resp = await daemon.handle_request(req)
    logger.info("Daemon status response:\n%s", json.dumps(resp.payload, indent=2))

    # --- Phase 8: Final journal dump ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("FULL JOURNAL DUMP")
    logger.info("=" * 60)
    all_events = journal.replay_session("smoke-session-1")
    for i, e in enumerate(all_events, 1):
        logger.info(
            "  %2d. [%-18s] %-30s %s",
            i, e.event_type, e.ritual_id, e.payload or "",
        )
    logger.info("")
    logger.info("Total events: %d", len(all_events))

    # --- Shutdown ---
    await runtime.stop()
    journal.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("SMOKE TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
