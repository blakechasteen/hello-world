"""
HoloLoom Unified Bus
====================

Single event backbone composing three existing bus patterns:
- Priority dispatch (RitualMessageBus)
- Topic pub/sub (Multi-Agent MessageBus)
- Memory delegation (LiteMemoryBus)

Full stack:
    Signal → UnifiedBus → Adapters → Journal/Neo4j

    Trigger modes: Active, Heartbeat, Cron, Reactive
    Strategic MCTS: Selects chain patterns / workflows per query
    Convergence: Monitors step confidence, early-terminates chains
    Persistence: SQLite (traces), Neo4j (MCTS trees, pattern-domains)

Usage:
    from hololoom.core.bus import UnifiedBus, Signal, SignalKind
    from hololoom.core.bus.journal import SignalJournal
    from hololoom.core.bus.strategy_selector import StrategySelector
    from hololoom.core.bus.adapters.convergence_adapter import ConvergenceMonitor
    from hololoom.core.bus.adapters.trigger_adapters import HeartbeatTrigger, CronTrigger

    journal = SignalJournal("data/signal_journal.db")
    bus = UnifiedBus(journal=journal)
    selector = StrategySelector(bus)
    monitor = ConvergenceMonitor(bus)
    heartbeat = HeartbeatTrigger(bus, interval=60)

    bus.on(SignalKind.STEP_COMPLETE, my_handler)
    await bus.emit(Signal(kind=SignalKind.QUERY, payload={...}))
"""

from .signal import (
    Signal,
    SignalHandler,
    SignalKind,
    SignalPriority,
    TriggerMode,
)
from .unified_bus import UnifiedBus, create_bus

__all__ = [
    # Signal schema
    "Signal",
    "SignalHandler",
    "SignalKind",
    "SignalPriority",
    "TriggerMode",
    # Bus
    "UnifiedBus",
    "create_bus",
]
