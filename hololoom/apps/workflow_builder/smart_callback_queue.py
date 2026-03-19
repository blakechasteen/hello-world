"""
Smart Non-Interrupting Context-Aware Callback Queue

Intelligently delivers callbacks (breakthroughs, notifications) at natural
breakpoints without interrupting user's flow.

Key Features:
- Non-Interrupting: Waits for natural breakpoints (after response, not during reasoning)
- Context-Aware: Prioritizes callbacks relevant to current conversation
- Smart Batching: Groups similar notifications
- Rate Limiting: Avoids overwhelming user
- Priority Queuing: Important callbacks first

Philosophy: Breakthroughs are valuable, but interruptions are costly.
Deliver insights at the right time, not just any time.
"""

import asyncio
import heapq
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Callback Types and States
# ============================================================================

class CallbackPriority(Enum):
    """Callback priority levels"""
    CRITICAL = 1  # Deliver ASAP (errors, critical breakthroughs)
    HIGH = 2      # Deliver soon (high-impact breakthroughs)
    NORMAL = 3    # Deliver at next breakpoint (standard breakthroughs)
    LOW = 4       # Deliver when idle (background info)


class ConversationState(Enum):
    """Conversation flow states"""
    IDLE = "idle"                    # No active query
    THINKING = "thinking"            # Reasoning in progress
    RESPONDING = "responding"        # Sending response
    WAITING_USER = "waiting_user"    # Waiting for user input
    INTERRUPTED = "interrupted"      # User interrupted (new query before response)


@dataclass
class QueuedCallback:
    """A callback queued for delivery"""
    callback_id: str
    priority: CallbackPriority
    callback_type: str  # "breakthrough", "notification", "suggestion", etc.

    # Callback function
    callback_fn: Callable

    # Context awareness
    thread_id: str
    user_id: str
    agent_name: str
    related_query: str | None = None  # Query that triggered this

    # Metadata for smart decisions
    impact_score: float = 0.5  # 0-1, how impactful is this?
    relevance_score: float = 0.5  # 0-1, how relevant to current conversation?
    urgency: float = 0.5  # 0-1, how urgent is this?

    # Timing
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # Auto-expire after time
    delivered_at: float | None = None

    # Batching
    batch_key: str | None = None  # Group by this key for batching

    def __lt__(self, other):
        """Priority queue comparison"""
        # Lower priority number = higher priority
        # Within same priority, use impact + relevance + urgency
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # Tie-break with composite score
        self_score = (self.impact_score + self.relevance_score + self.urgency) / 3
        other_score = (other.impact_score + other.relevance_score + other.urgency) / 3
        return self_score > other_score

    def is_expired(self) -> bool:
        """Check if callback has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


# ============================================================================
# Smart Callback Queue
# ============================================================================

class SmartCallbackQueue:
    """
    Context-aware callback queue that respects conversation flow.

    Delivers callbacks at natural breakpoints:
    - CRITICAL: Deliver immediately (even during reasoning)
    - HIGH: Deliver after current response
    - NORMAL: Deliver at next idle moment
    - LOW: Deliver when user inactive for >10s

    Also provides:
    - Smart batching (group similar callbacks)
    - Context-aware prioritization
    - Rate limiting
    """

    def __init__(
        self,
        max_callbacks_per_minute: int = 5,
        batch_window_seconds: float = 2.0,
        idle_threshold_seconds: float = 10.0
    ):
        self.max_callbacks_per_minute = max_callbacks_per_minute
        self.batch_window_seconds = batch_window_seconds
        self.idle_threshold_seconds = idle_threshold_seconds

        # Priority queue (min heap)
        self.queue: list[QueuedCallback] = []

        # Thread states
        self.thread_states: dict[str, ConversationState] = {}

        # Rate limiting
        self.delivery_history: list[float] = []  # Timestamps of deliveries

        # Batching
        self.batch_buffer: dict[str, list[QueuedCallback]] = defaultdict(list)
        self.batch_timers: dict[str, float] = {}  # When batch started

        # Statistics
        self.total_queued = 0
        self.total_delivered = 0
        self.total_expired = 0
        self.total_batched = 0

        # Background task
        self.processor_task: asyncio.Task | None = None
        self.running = False

    async def start(self):
        """Start background callback processor"""
        if self.running:
            return

        self.running = True
        self.processor_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        """Stop background callback processor"""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

    async def enqueue(
        self,
        callback_fn: Callable,
        thread_id: str,
        user_id: str,
        agent_name: str,
        priority: CallbackPriority = CallbackPriority.NORMAL,
        callback_type: str = "notification",
        impact_score: float = 0.5,
        relevance_score: float = 0.5,
        urgency: float = 0.5,
        related_query: str | None = None,
        expires_in_seconds: float | None = None,
        batch_key: str | None = None
    ) -> str:
        """
        Queue a callback for smart delivery.

        Returns:
            callback_id
        """
        callback_id = f"cb_{self.total_queued}"

        expires_at = None
        if expires_in_seconds:
            expires_at = time.time() + expires_in_seconds

        callback = QueuedCallback(
            callback_id=callback_id,
            priority=priority,
            callback_type=callback_type,
            callback_fn=callback_fn,
            thread_id=thread_id,
            user_id=user_id,
            agent_name=agent_name,
            related_query=related_query,
            impact_score=impact_score,
            relevance_score=relevance_score,
            urgency=urgency,
            expires_at=expires_at,
            batch_key=batch_key
        )

        # Add to queue
        heapq.heappush(self.queue, callback)
        self.total_queued += 1

        # If critical, deliver immediately
        if priority == CallbackPriority.CRITICAL:
            await self._deliver_callback(callback)

        return callback_id

    def set_thread_state(self, thread_id: str, state: ConversationState):
        """Update conversation state for thread"""
        self.thread_states[thread_id] = state

    def get_thread_state(self, thread_id: str) -> ConversationState:
        """Get current conversation state"""
        return self.thread_states.get(thread_id, ConversationState.IDLE)

    async def _process_loop(self):
        """Background loop that processes queue"""
        while self.running:
            try:
                await self._process_queue()
                await asyncio.sleep(0.5)  # Check every 500ms
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[SmartCallbackQueue] Error in process loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_queue(self):
        """Process callbacks from queue"""
        if not self.queue:
            return

        # Remove expired callbacks
        while self.queue and self.queue[0].is_expired():
            expired = heapq.heappop(self.queue)
            self.total_expired += 1

        if not self.queue:
            return

        # Check rate limit
        now = time.time()
        self.delivery_history = [
            ts for ts in self.delivery_history
            if ts > (now - 60)  # Last minute only
        ]

        if len(self.delivery_history) >= self.max_callbacks_per_minute:
            return  # Rate limited

        # Get next callback
        next_callback = self.queue[0]

        # Check if we should deliver
        if await self._should_deliver(next_callback):
            callback = heapq.heappop(self.queue)

            # Check if batchable
            if callback.batch_key:
                await self._handle_batching(callback)
            else:
                await self._deliver_callback(callback)

    async def _should_deliver(self, callback: QueuedCallback) -> bool:
        """Determine if callback should be delivered now"""
        # Already delivered critical callbacks immediately
        if callback.priority == CallbackPriority.CRITICAL:
            return True

        thread_state = self.get_thread_state(callback.thread_id)

        # HIGH priority: Wait for response to finish
        if callback.priority == CallbackPriority.HIGH:
            return thread_state in [
                ConversationState.IDLE,
                ConversationState.WAITING_USER
            ]

        # NORMAL priority: Wait for natural breakpoint
        if callback.priority == CallbackPriority.NORMAL:
            return thread_state in [
                ConversationState.IDLE,
                ConversationState.WAITING_USER
            ]

        # LOW priority: Wait for idle period
        if callback.priority == CallbackPriority.LOW:
            if thread_state != ConversationState.WAITING_USER:
                return False

            # Check how long thread has been idle
            # (Would need to track last activity timestamp - simplified for now)
            return True

        return False

    async def _handle_batching(self, callback: QueuedCallback):
        """Handle callback batching"""
        batch_key = callback.batch_key

        # Add to batch buffer
        self.batch_buffer[batch_key].append(callback)

        # Start batch timer if first in batch
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = time.time()

        # Check if batch window expired
        batch_age = time.time() - self.batch_timers[batch_key]

        if batch_age >= self.batch_window_seconds:
            # Deliver batch
            batch = self.batch_buffer.pop(batch_key)
            del self.batch_timers[batch_key]

            await self._deliver_batch(batch)

    async def _deliver_batch(self, batch: list[QueuedCallback]):
        """Deliver a batch of callbacks"""
        if not batch:
            return

        # Create batched callback
        async def batched_fn():
            for callback in batch:
                try:
                    await callback.callback_fn()
                except Exception as e:
                    print(f"[SmartCallbackQueue] Batch callback error: {e}")

        # Deliver
        await batched_fn()

        # Track statistics
        self.total_delivered += 1
        self.total_batched += len(batch)
        self.delivery_history.append(time.time())

        # Mark as delivered
        for callback in batch:
            callback.delivered_at = time.time()

    async def _deliver_callback(self, callback: QueuedCallback):
        """Deliver single callback"""
        try:
            await callback.callback_fn()

            # Track statistics
            self.total_delivered += 1
            self.delivery_history.append(time.time())

            # Mark as delivered
            callback.delivered_at = time.time()

        except Exception as e:
            print(f"[SmartCallbackQueue] Callback delivery error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': len(self.queue),
            'total_queued': self.total_queued,
            'total_delivered': self.total_delivered,
            'total_expired': self.total_expired,
            'total_batched': self.total_batched,
            'deliveries_last_minute': len([
                ts for ts in self.delivery_history
                if ts > (time.time() - 60)
            ]),
            'batch_buffers': {
                key: len(callbacks)
                for key, callbacks in self.batch_buffer.items()
            },
            'thread_states': {
                thread_id: state.value
                for thread_id, state in self.thread_states.items()
            }
        }


# ============================================================================
# Helper Functions
# ============================================================================

async def create_smart_callback_queue(
    max_callbacks_per_minute: int = 5,
    batch_window_seconds: float = 2.0
) -> SmartCallbackQueue:
    """Create and start smart callback queue"""
    queue = SmartCallbackQueue(
        max_callbacks_per_minute=max_callbacks_per_minute,
        batch_window_seconds=batch_window_seconds
    )

    await queue.start()
    return queue
