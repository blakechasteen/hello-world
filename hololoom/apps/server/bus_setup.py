"""
Bus Lifecycle — Singleton wiring for the UnifiedBus in the API server.
======================================================================

Creates and manages the bus, journal, strategy selector, convergence monitor,
and optional triggers. Lazy singleton — first call creates, subsequent calls
return the same instance.

The bus is the nervous system. Endpoints emit signals, adapters react.

Usage:
    from hololoom.apps.server.bus_setup import get_bus, shutdown_bus

    # In endpoint:
    bus = get_bus()
    response = await bus.request(signal, response_kind=..., timeout=30)

    # On server shutdown:
    await shutdown_bus()
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Singleton State
# ============================================================================

_bus = None
_journal = None
_selector = None
_convergence = None
_heartbeat = None
_initialized = False


def get_bus():
    """
    Get or create the UnifiedBus singleton.

    Lazy initialization — the bus and all adapters are created on first call.
    Gracefully degrades if bus components aren't available.
    """
    global _bus, _journal, _selector, _convergence, _heartbeat, _initialized

    if _initialized:
        return _bus

    _initialized = True

    try:
        from hololoom.core.bus import UnifiedBus
        from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor
        from hololoom.core.bus.journal import SignalJournal
        from hololoom.core.bus.strategy_selector import StrategySelector

        # Journal — SQLite persistence for signal traces
        journal_path = os.environ.get(
            "HOLOLOOM_JOURNAL_DB",
            str(Path(__file__).resolve().parent.parent.parent.parent
                / "data" / "signal_journal.db"),
        )
        Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
        _journal = SignalJournal(journal_path)

        # Bus — the backbone
        _bus = UnifiedBus(journal=_journal)

        # Strategy selector — MCTS-driven chain selection on QUERY signals
        _selector = StrategySelector(_bus)

        # Convergence monitor — detects when chains have converged
        _convergence = ConvergenceMonitor(_bus)

        # Heartbeat — optional background pulse
        if os.environ.get("HOLOLOOM_HEARTBEAT", "").lower() == "true":
            from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger
            interval = float(os.environ.get("HOLOLOOM_HEARTBEAT_INTERVAL", "60"))
            _heartbeat = HeartbeatTrigger(_bus, interval=interval)

        logger.info(
            "UnifiedBus initialized (journal=%s, heartbeat=%s)",
            journal_path, _heartbeat is not None,
        )
        return _bus

    except Exception as e:
        logger.warning("UnifiedBus unavailable — endpoints run without bus: %s", e)
        _bus = None
        return None


def get_journal() -> Optional['SignalJournal']:
    """Get the signal journal (None if bus not initialized)."""
    return _journal


def get_selector() -> Optional['StrategySelector']:
    """Get the strategy selector (None if bus not initialized)."""
    return _selector


async def startup_bus() -> None:
    """
    Async startup — call from FastAPI lifespan or on_event("startup").

    Starts the bus and optional triggers.
    """
    bus = get_bus()
    if bus is not None:
        await bus.start()
        if _heartbeat is not None:
            await _heartbeat.start()
        logger.info("UnifiedBus started")


async def shutdown_bus() -> None:
    """
    Async shutdown — call from FastAPI lifespan or on_event("shutdown").

    Stops triggers, bus, and closes the journal.
    """
    if _heartbeat is not None:
        await _heartbeat.stop()
    if _bus is not None:
        await _bus.stop()
    if _journal is not None:
        _journal.close()
    logger.info("UnifiedBus shutdown complete")
