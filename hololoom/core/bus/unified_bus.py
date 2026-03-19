"""
Unified Bus — Single event backbone for all HoloLoom execution.
================================================================

Composes three existing bus patterns into one:

1. RitualMessageBus pattern:
   - Priority queue (heapq) for signal ordering
   - TTL-based expiration
   - Correlation/causation chain propagation on handler responses
   - Non-blocking async dispatch
   - Message history + dedup

2. Multi-Agent MessageBus pattern:
   - Agent-addressed pub/sub (subscribe by topic/id)
   - Async dispatch loop
   - Start/stop lifecycle

3. LiteMemoryBus delegation:
   - Memory store/query operations delegated to existing LiteMemoryBus
   - Emits MEMORY_STORED / MEMORY_QUERIED signals on operations
   - 4-level cascade (EXACT → STRUCTURED → GRAPH → SEMANTIC) preserved

New capabilities:
   - SignalKind-based topic routing (subscribe to signal types)
   - TriggerMode tracking (ACTIVE, HEARTBEAT, CRON, REACTIVE)
   - Optional SignalJournal for SQLite persistence (Phase 2)
   - Futures for request-response patterns (emit-and-await)

Design: The UnifiedBus WRAPS existing buses. It does not replace them.
Existing code that uses LiteMemoryBus directly continues to work.
"""

import asyncio
import heapq
import logging
from typing import (
    Any,
)

from .signal import Signal, SignalHandler, SignalKind, SignalPriority, TriggerMode

logger = logging.getLogger(__name__)


class UnifiedBus:
    """
    Single event backbone for HoloLoom.

    Merges priority dispatch (RitualMessageBus), topic pub/sub (Multi-Agent),
    and memory delegation (LiteMemoryBus) into one composable system.

    Usage:
        bus = UnifiedBus()

        # Subscribe to signal kinds
        bus.on(SignalKind.STEP_COMPLETE, my_handler)

        # Subscribe to targeted signals (by name)
        bus.subscribe("promptly", my_handler)

        # Subscribe to all signals
        bus.subscribe_broadcast(my_handler)

        # Emit a signal
        await bus.emit(Signal(kind=SignalKind.QUERY, payload={...}))

        # Emit and wait for a correlated response
        response = await bus.request(
            Signal(kind=SignalKind.QUERY, payload={...}),
            response_kind=SignalKind.EXECUTION_COMPLETE,
            timeout=30.0,
        )
    """

    def __init__(
        self,
        memory_bus: Any | None = None,
        journal: Any | None = None,
        history_limit: int = 1000,
    ):
        """
        Initialize unified bus.

        Args:
            memory_bus: Optional LiteMemoryBus instance to delegate memory ops to.
                        When provided, store_memory/query_memory are available and
                        emit MEMORY_STORED/MEMORY_QUERIED signals automatically.
            journal: Optional SignalJournal for SQLite persistence (Phase 2).
            history_limit: Max signals to keep in memory history.
        """
        # Priority queue (from RitualMessageBus)
        self._queue: list[Signal] = []
        self._lock = asyncio.Lock()

        # Topic-based handlers (from Multi-Agent MessageBus pattern)
        self._target_handlers: dict[str, list[SignalHandler]] = {}

        # Kind-based handlers (new: subscribe to signal types)
        self._kind_handlers: dict[SignalKind, list[SignalHandler]] = {}

        # Broadcast handlers (from RitualMessageBus)
        self._broadcast_handlers: list[SignalHandler] = []

        # Dedup (from RitualMessageBus)
        self._processed_ids: set[str] = set()
        self._processed_limit = history_limit * 2

        # History (from RitualMessageBus)
        self._history: list[Signal] = []
        self._history_limit = history_limit

        # Request-response futures (new: for emit-and-await pattern)
        self._waiters: dict[str, list[tuple[SignalKind, asyncio.Future]]] = {}

        # Delegates
        self._memory_bus = memory_bus
        self._journal = journal

        # Lifecycle
        self._running = False
        self._dispatch_task: asyncio.Task | None = None

        # Stats
        self._emit_count = 0
        self._dispatch_count = 0

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def start(self) -> None:
        """Start the bus dispatch loop."""
        if self._running:
            return
        self._running = True
        logger.info("UnifiedBus started")

    async def stop(self) -> None:
        """Stop the bus and cancel pending waiters."""
        if not self._running:
            return
        self._running = False

        # Cancel all pending futures
        for correlation_id, waiter_list in self._waiters.items():
            for _, future in waiter_list:
                if not future.done():
                    future.cancel()
        self._waiters.clear()

        logger.info("UnifiedBus stopped")

    async def __aenter__(self) -> 'UnifiedBus':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # ========================================================================
    # Emit
    # ========================================================================

    async def emit(self, signal: Signal) -> str:
        """
        Emit a signal to the bus.

        The signal is:
        1. Added to the priority queue
        2. Added to history
        3. Journaled (if journal is configured)
        4. Dispatched to matching handlers (non-blocking)

        Returns:
            Signal ID
        """
        async with self._lock:
            heapq.heappush(self._queue, signal)
            self._add_to_history(signal)

        # Journal (Phase 2 — no-op if journal is None)
        if self._journal:
            try:
                await self._journal.append(signal)
            except Exception as e:
                logger.warning(f"Journal write failed: {e}")

        # Dispatch (non-blocking, matches RitualMessageBus.send pattern)
        asyncio.ensure_future(self._dispatch(signal))

        self._emit_count += 1
        logger.debug(
            f"Signal emitted: {signal.id} ({signal.kind.value}) "
            f"→ {signal.target or 'broadcast'}"
        )
        return signal.id

    # ========================================================================
    # Request-Response (emit and await correlated response)
    # ========================================================================

    async def request(
        self,
        signal: Signal,
        response_kind: SignalKind,
        timeout: float = 30.0,
    ) -> Signal:
        """
        Emit a signal and wait for a correlated response.

        This is the pattern for thin API endpoints:
            response = await bus.request(
                Signal(kind=SignalKind.QUERY, payload={"text": "..."}),
                response_kind=SignalKind.EXECUTION_COMPLETE,
                timeout=30.0,
            )

        Args:
            signal: Signal to emit
            response_kind: Kind of signal to wait for
            timeout: Max seconds to wait

        Returns:
            Response Signal

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Signal] = loop.create_future()

        # Register waiter on the signal's correlation chain
        correlation_key = signal.correlation_id or signal.id
        signal.correlation_id = correlation_key

        if correlation_key not in self._waiters:
            self._waiters[correlation_key] = []
        self._waiters[correlation_key].append((response_kind, future))

        # Emit the signal
        await self.emit(signal)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            # Clean up waiter
            if correlation_key in self._waiters:
                self._waiters[correlation_key] = [
                    (k, f) for k, f in self._waiters[correlation_key]
                    if f is not future
                ]
                if not self._waiters[correlation_key]:
                    del self._waiters[correlation_key]

    # ========================================================================
    # Subscribe
    # ========================================================================

    def subscribe(self, target: str, handler: SignalHandler) -> str:
        """
        Subscribe to signals addressed to a specific target.

        Matches the Multi-Agent MessageBus pattern (agent-addressed).

        Args:
            target: Target identifier (agent name, endpoint, etc.)
            handler: Async handler function

        Returns:
            Subscription ID
        """
        if target not in self._target_handlers:
            self._target_handlers[target] = []
        self._target_handlers[target].append(handler)
        sub_id = f"target:{target}:{len(self._target_handlers[target]) - 1}"
        logger.debug(f"Subscribed: {sub_id}")
        return sub_id

    def on(self, kind: SignalKind, handler: SignalHandler) -> str:
        """
        Subscribe to signals of a specific kind.

        New: Kind-based routing. Handlers receive all signals of the given type
        regardless of target.

        Args:
            kind: Signal kind to subscribe to
            handler: Async handler function

        Returns:
            Subscription ID
        """
        if kind not in self._kind_handlers:
            self._kind_handlers[kind] = []
        self._kind_handlers[kind].append(handler)
        sub_id = f"kind:{kind.value}:{len(self._kind_handlers[kind]) - 1}"
        logger.debug(f"Subscribed: {sub_id}")
        return sub_id

    def subscribe_broadcast(self, handler: SignalHandler) -> str:
        """
        Subscribe to all signals (broadcast).

        Matches RitualMessageBus.subscribe_broadcast pattern.
        """
        self._broadcast_handlers.append(handler)
        sub_id = f"broadcast:{len(self._broadcast_handlers) - 1}"
        logger.debug(f"Subscribed broadcast: {sub_id}")
        return sub_id

    def unsubscribe(self, target: str, handler: SignalHandler) -> bool:
        """Remove a handler from a target subscription."""
        if target in self._target_handlers:
            try:
                self._target_handlers[target].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def off(self, kind: SignalKind, handler: SignalHandler) -> bool:
        """Remove a handler from a kind subscription."""
        if kind in self._kind_handlers:
            try:
                self._kind_handlers[kind].remove(handler)
                return True
            except ValueError:
                pass
        return False

    # ========================================================================
    # Memory Bus Delegation
    # ========================================================================

    async def store_memory(self, item: Any) -> str:
        """
        Store a memory item via the delegated LiteMemoryBus.

        Emits a MEMORY_STORED signal after successful storage.

        Args:
            item: MemoryItem to store

        Returns:
            Stored item ID
        """
        if not self._memory_bus:
            raise RuntimeError("No memory bus configured")

        item_id = await self._memory_bus.store(item)

        # Emit reactive signal
        await self.emit(Signal(
            kind=SignalKind.MEMORY_STORED,
            priority=SignalPriority.LOW,
            source="memory_bus",
            payload={"item_id": item_id, "memory_type": getattr(item, "memory_type", "unknown")},
            trigger_mode=TriggerMode.REACTIVE,
        ))

        return item_id

    async def query_memory(self, query: Any) -> Any:
        """
        Query memory via the delegated LiteMemoryBus.

        Emits a MEMORY_QUERIED signal after query.

        Args:
            query: MemoryQuery

        Returns:
            MemoryResult
        """
        if not self._memory_bus:
            raise RuntimeError("No memory bus configured")

        result = await self._memory_bus.query(query)

        # Emit reactive signal (LOW priority — observability only)
        await self.emit(Signal(
            kind=SignalKind.MEMORY_QUERIED,
            priority=SignalPriority.LOW,
            source="memory_bus",
            payload={
                "intent": getattr(query, "intent", ""),
                "result_count": len(getattr(result, "items", [])),
                "resolution_path": getattr(result, "resolution_path", ""),
            },
            trigger_mode=TriggerMode.REACTIVE,
        ))

        return result

    # ========================================================================
    # Dispatch (internal)
    # ========================================================================

    async def _dispatch(self, signal: Signal) -> None:
        """
        Dispatch signal to matching handlers.

        Routing (mirrors RitualMessageBus._dispatch with kind-based extension):
        1. Skip expired signals (TTL check)
        2. Skip already-processed signals (dedup)
        3. Route to target handlers (if target is set)
        4. Route to kind handlers (always, if subscribed)
        5. Route to broadcast handlers (if target is empty)
        6. Resolve any waiting futures (request-response)
        7. If handler returns a Signal, re-emit with correlation chain
        """
        # TTL check
        if signal.is_expired():
            logger.debug(f"Skipping expired signal: {signal.id}")
            return

        # Dedup
        if signal.id in self._processed_ids:
            return
        self._processed_ids.add(signal.id)

        # Prune dedup set (from RitualMessageBus pattern)
        if len(self._processed_ids) > self._processed_limit:
            oldest = list(self._processed_ids)[:len(self._processed_ids) // 2]
            for oid in oldest:
                self._processed_ids.discard(oid)

        self._dispatch_count += 1

        # Collect all handlers to invoke
        handlers: list[SignalHandler] = []

        # 1. Target-addressed handlers
        if signal.target and signal.target in self._target_handlers:
            handlers.extend(self._target_handlers[signal.target])

        # 2. Kind-based handlers (always fire if subscribed)
        if signal.kind in self._kind_handlers:
            handlers.extend(self._kind_handlers[signal.kind])

        # 3. Broadcast handlers (only when not targeted, matching RitualMessageBus)
        if not signal.target:
            handlers.extend(self._broadcast_handlers)

        # Execute handlers
        for handler in handlers:
            try:
                response = await handler(signal)
                if response is not None and isinstance(response, Signal):
                    # Propagate correlation chain (from RitualMessageBus pattern)
                    response.correlation_id = signal.correlation_id or signal.id
                    response.causation_id = signal.id
                    response.trigger_mode = TriggerMode.REACTIVE
                    await self.emit(response)
            except Exception as e:
                logger.error(f"Handler error for signal {signal.id}: {e}")

        # 4. Resolve waiting futures (request-response pattern)
        correlation_key = signal.correlation_id
        if correlation_key and correlation_key in self._waiters:
            resolved = []
            for i, (expected_kind, future) in enumerate(self._waiters[correlation_key]):
                if signal.kind == expected_kind and not future.done():
                    future.set_result(signal)
                    resolved.append(i)
            # Remove resolved waiters
            for i in reversed(resolved):
                self._waiters[correlation_key].pop(i)
            if not self._waiters[correlation_key]:
                del self._waiters[correlation_key]

    # ========================================================================
    # History & Stats
    # ========================================================================

    def _add_to_history(self, signal: Signal) -> None:
        """Add signal to in-memory history (from RitualMessageBus)."""
        self._history.append(signal)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

    def get_history(
        self,
        correlation_id: str | None = None,
        kind: SignalKind | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Signal]:
        """
        Get signal history with optional filters.
        Matches RitualMessageBus.get_history pattern.
        """
        result = self._history.copy()

        if correlation_id:
            result = [s for s in result if s.correlation_id == correlation_id]
        if kind:
            result = [s for s in result if s.kind == kind]
        if source:
            result = [s for s in result if s.source == source]

        return list(reversed(result[-limit:]))

    def get_statistics(self) -> dict[str, Any]:
        """Get bus statistics (matches RitualMessageBus.get_statistics)."""
        kind_counts: dict[str, int] = {}
        priority_counts: dict[int, int] = {}
        trigger_counts: dict[str, int] = {}

        for s in self._history:
            kind_counts[s.kind.value] = kind_counts.get(s.kind.value, 0) + 1
            priority_counts[s.priority.value] = priority_counts.get(s.priority.value, 0) + 1
            trigger_counts[s.trigger_mode.value] = trigger_counts.get(s.trigger_mode.value, 0) + 1

        return {
            "total_emitted": self._emit_count,
            "total_dispatched": self._dispatch_count,
            "history_size": len(self._history),
            "pending_queue": len(self._queue),
            "active_waiters": sum(len(w) for w in self._waiters.values()),
            "target_subscriptions": {k: len(v) for k, v in self._target_handlers.items()},
            "kind_subscriptions": {k.value: len(v) for k, v in self._kind_handlers.items()},
            "broadcast_handlers": len(self._broadcast_handlers),
            "by_kind": kind_counts,
            "by_priority": priority_counts,
            "by_trigger": trigger_counts,
        }

    def clear_history(self) -> int:
        """Clear signal history. Returns count of cleared signals."""
        count = len(self._history)
        self._history.clear()
        self._processed_ids.clear()
        return count


# ============================================================================
# Convenience: create_bus factory
# ============================================================================

def create_bus(
    memory_bus: Any | None = None,
    journal: Any | None = None,
    history_limit: int = 1000,
) -> UnifiedBus:
    """
    Create a configured UnifiedBus instance.

    Args:
        memory_bus: Optional LiteMemoryBus for memory delegation
        journal: Optional SignalJournal for persistence (Phase 2)
        history_limit: Max signals in memory history

    Returns:
        Configured UnifiedBus
    """
    return UnifiedBus(
        memory_bus=memory_bus,
        journal=journal,
        history_limit=history_limit,
    )


__all__ = [
    "UnifiedBus",
    "create_bus",
]
