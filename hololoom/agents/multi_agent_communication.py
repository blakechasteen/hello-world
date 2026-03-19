#!/usr/bin/env python3
from __future__ import annotations
"""
Multi-Agent Communication System
================================

Enable persistent agents to talk to each other with budget limits and safety guardrails.

Created: 2025-01-20

Philosophy:
"Agents learn more by talking to each other than by working alone."

Features:
- Inter-agent messaging
- Budget limits (tokens, time, conversations)
- Safety guardrails (depth, loops, relevance)
- Conversation orchestration
- Insight sharing

Architecture:
- MessageBus: Async message passing
- ConversationManager: Orchestrate multi-agent dialogues
- BudgetManager: Track and enforce budgets
- SafetyGuardrails: Prevent infinite loops and ensure productivity
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Safety Integration (Dec 2025) - MRF Safe Integration Phase
# Import alignment framework's adversarial detection for inter-agent messages
try:
    from hololoom.alignment.safety_guardrails import (
        AdversarialDetector as AlignmentAdversarialDetector,
    )
    ALIGNMENT_ADVERSARIAL_AVAILABLE = True
except ImportError:
    ALIGNMENT_ADVERSARIAL_AVAILABLE = False
    AlignmentAdversarialDetector = None

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Type of inter-agent message."""
    QUESTION = "question"           # Ask another agent
    ANSWER = "answer"               # Answer to question
    INSIGHT = "insight"             # Share insight
    REQUEST_HELP = "request_help"   # Request collaboration
    OFFER_HELP = "offer_help"       # Offer to help
    ACKNOWLEDGE = "acknowledge"     # Acknowledge message


@dataclass
class Message:
    """Message between agents."""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: str
    timestamp: datetime
    conversation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_message_id: str | None = None


@dataclass
class Conversation:
    """Multi-agent conversation."""
    id: str
    participants: list[str]
    messages: list[Message]
    started_at: datetime
    topic: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Total message count."""
        return len(self.messages)

    @property
    def duration_seconds(self) -> float:
        """Conversation duration."""
        if not self.messages:
            return 0.0
        return (datetime.now() - self.started_at).total_seconds()


@dataclass
class Budget:
    """Budget limits for conversations."""
    max_messages: int = 10              # Max messages per conversation
    max_duration_seconds: float = 300.0  # 5 minutes
    max_depth: int = 3                  # Max agent-to-agent chain depth
    max_conversations_per_hour: int = 10
    max_token_estimate: int = 10000     # Estimated tokens

    # Current usage
    messages_used: int = 0
    duration_used: float = 0.0
    depth_used: int = 0
    conversations_this_hour: int = 0
    tokens_used: int = 0


class SafetyGuardrails:
    """
    Safety guardrails for multi-agent communication.

    Prevents:
    - Infinite loops
    - Excessive resource usage
    - Off-topic conversations
    - Unproductive dialogues
    """

    def __init__(
        self,
        min_insight_length: int = 20,
        similarity_threshold: float = 0.8,
        relevance_keywords: list[str] | None = None
    ):
        """
        Initialize safety guardrails.

        Args:
            min_insight_length: Minimum characters for valid insight
            similarity_threshold: Max similarity to detect loops
            relevance_keywords: Keywords to ensure relevance
        """
        self.min_insight_length = min_insight_length
        self.similarity_threshold = similarity_threshold
        self.relevance_keywords = relevance_keywords or []

        # Safety Integration (Dec 2025) - MRF Safe Integration Phase
        # Add adversarial detection for inter-agent messages
        self.adversarial_detector = None
        if ALIGNMENT_ADVERSARIAL_AVAILABLE:
            try:
                self.adversarial_detector = AlignmentAdversarialDetector()
                logger.info("Adversarial detection enabled for multi-agent communication")
            except Exception as e:
                logger.warning(f"Could not initialize adversarial detector: {e}")

    def check_budget(self, budget: Budget, conversation: Conversation) -> tuple[bool, str]:
        """
        Check if budget limits are exceeded.

        Returns:
            (allowed, reason)
        """
        # Message limit
        if conversation.message_count >= budget.max_messages:
            return False, f"Message limit reached ({budget.max_messages})"

        # Duration limit
        if conversation.duration_seconds >= budget.max_duration_seconds:
            return False, f"Duration limit reached ({budget.max_duration_seconds}s)"

        # Token estimate (rough: 4 chars per token)
        total_chars = sum(len(m.content) for m in conversation.messages)
        estimated_tokens = total_chars // 4
        if estimated_tokens >= budget.max_token_estimate:
            return False, f"Token limit reached (~{budget.max_token_estimate})"

        return True, "Budget OK"

    def detect_loop(self, conversation: Conversation) -> tuple[bool, str]:
        """
        Detect if conversation is looping.

        Returns:
            (is_loop, reason)
        """
        if len(conversation.messages) < 4:
            return False, "Too few messages"

        # Check last 4 messages for similarity
        recent = conversation.messages[-4:]

        # Simple similarity: count common words
        for i in range(len(recent) - 1):
            msg1 = recent[i].content.lower().split()
            msg2 = recent[i + 1].content.lower().split()

            if not msg1 or not msg2:
                continue

            common = len(set(msg1) & set(msg2))
            total = len(set(msg1) | set(msg2))

            if total > 0:
                similarity = common / total
                if similarity >= self.similarity_threshold:
                    return True, f"Loop detected (similarity: {similarity:.2f})"

        return False, "No loop"

    def check_productivity(self, conversation: Conversation) -> tuple[bool, str]:
        """
        Check if conversation is productive.

        Returns:
            (is_productive, reason)
        """
        # Must have insights or help exchanges
        insights = [m for m in conversation.messages if m.message_type == MessageType.INSIGHT]
        help_exchanges = [m for m in conversation.messages
                         if m.message_type in [MessageType.REQUEST_HELP, MessageType.OFFER_HELP]]

        if len(insights) == 0 and len(help_exchanges) == 0:
            if conversation.message_count >= 5:
                return False, "No insights generated after 5 messages"

        # Check insight quality
        for insight_msg in insights:
            if len(insight_msg.content) < self.min_insight_length:
                return False, f"Insight too short ({len(insight_msg.content)} chars)"

        return True, "Productive"

    def check_relevance(self, message: Message, topic: str) -> tuple[bool, str]:
        """
        Check if message is relevant to topic.

        Returns:
            (is_relevant, reason)
        """
        if not self.relevance_keywords:
            return True, "No keyword check"

        content_lower = message.content.lower()
        topic_lower = topic.lower()

        # Check if topic words appear in message
        topic_words = set(topic_lower.split())
        message_words = set(content_lower.split())

        overlap = len(topic_words & message_words)

        if overlap == 0:
            return False, "No topic overlap"

        return True, "Relevant"

    def check_adversarial(self, message: 'Message') -> tuple[bool, str]:
        """
        Check if message contains adversarial content.

        Safety Integration (Dec 2025) - MRF Safe Integration Phase
        Uses alignment framework's adversarial detector for inter-agent messages.

        Args:
            message: Message to check

        Returns:
            (is_adversarial, reason) - True if adversarial content detected
        """
        if not self.adversarial_detector:
            return False, "Adversarial detection not available"

        try:
            is_adversarial, reason = self.adversarial_detector.detect(message.content)
            if is_adversarial:
                logger.warning(
                    f"[SAFETY] Adversarial content detected in inter-agent message: {reason}"
                )
            return is_adversarial, reason
        except Exception as e:
            logger.error(f"Adversarial detection error: {e}")
            # Fail-closed: treat errors as potential adversarial (CRITIQUE principle)
            return True, f"Detection error (fail-closed): {e}"


class BudgetManager:
    """
    Manage budgets for multi-agent conversations.

    Tracks usage and enforces limits.
    """

    def __init__(self, default_budget: Budget):
        """
        Initialize budget manager.

        Args:
            default_budget: Default budget for conversations
        """
        self.default_budget = default_budget
        self.conversation_budgets: dict[str, Budget] = {}
        self.hourly_reset_time = datetime.now()

    def get_budget(self, conversation_id: str) -> Budget:
        """Get budget for conversation."""
        if conversation_id not in self.conversation_budgets:
            # Create new budget
            self.conversation_budgets[conversation_id] = Budget(
                max_messages=self.default_budget.max_messages,
                max_duration_seconds=self.default_budget.max_duration_seconds,
                max_depth=self.default_budget.max_depth,
                max_conversations_per_hour=self.default_budget.max_conversations_per_hour,
                max_token_estimate=self.default_budget.max_token_estimate
            )

        return self.conversation_budgets[conversation_id]

    def update_usage(self, conversation: Conversation) -> None:
        """Update budget usage from conversation."""
        budget = self.get_budget(conversation.id)

        budget.messages_used = conversation.message_count
        budget.duration_used = conversation.duration_seconds

        # Estimate tokens
        total_chars = sum(len(m.content) for m in conversation.messages)
        budget.tokens_used = total_chars // 4

    def reset_hourly_counters(self) -> None:
        """Reset hourly conversation counters."""
        now = datetime.now()
        if (now - self.hourly_reset_time).total_seconds() >= 3600:
            # Reset all hourly counters
            for budget in self.conversation_budgets.values():
                budget.conversations_this_hour = 0
            self.hourly_reset_time = now

    def can_start_conversation(self) -> tuple[bool, str]:
        """Check if new conversation can be started."""
        self.reset_hourly_counters()

        # Count conversations this hour
        hourly_count = sum(
            budget.conversations_this_hour
            for budget in self.conversation_budgets.values()
        )

        if hourly_count >= self.default_budget.max_conversations_per_hour:
            return False, f"Hourly limit reached ({self.default_budget.max_conversations_per_hour})"

        return True, "OK"


class MessageBus:
    """
    Async message bus for inter-agent communication.

    Routes messages between agents.
    """

    def __init__(self):
        """Initialize message bus."""
        self.subscribers: dict[str, list[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._dispatch_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start message bus."""
        if self._running:
            return

        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("📬 Message bus started")

    async def stop(self) -> None:
        """Stop message bus."""
        if not self._running:
            return

        self._running = False

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        logger.info("📬 Message bus stopped")

    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> None:
        """
        Subscribe agent to messages.

        Args:
            agent_id: Agent ID
            callback: Callback for messages
        """
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    async def publish(self, message: Message) -> None:
        """
        Publish message to bus.

        Args:
            message: Message to publish
        """
        await self.message_queue.put(message)

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop."""
        while self._running:
            try:
                # Get message with timeout
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Deliver to recipient
                if message.to_agent in self.subscribers:
                    for callback in self.subscribers[message.to_agent]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(message)
                            else:
                                callback(message)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}", exc_info=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}", exc_info=True)


class ConversationManager:
    """
    Orchestrate multi-agent conversations.

    Manages conversations with budget limits and safety guardrails.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        budget_manager: BudgetManager,
        safety_guardrails: SafetyGuardrails
    ):
        """
        Initialize conversation manager.

        Args:
            message_bus: Message bus for communication
            budget_manager: Budget manager
            safety_guardrails: Safety guardrails
        """
        self.message_bus = message_bus
        self.budget_manager = budget_manager
        self.safety = safety_guardrails

        self.active_conversations: dict[str, Conversation] = {}
        self.conversation_history: list[Conversation] = []

        # Callbacks
        self.on_conversation_start: Callable | None = None
        self.on_conversation_end: Callable | None = None
        self.on_message: Callable | None = None
        self.on_insight: Callable | None = None
        self.on_budget_exceeded: Callable | None = None
        self.on_safety_violation: Callable | None = None

    async def start_conversation(
        self,
        topic: str,
        initiator: str,
        participants: list[str]
    ) -> str | None:
        """
        Start new conversation.

        Args:
            topic: Conversation topic
            initiator: Agent starting conversation
            participants: All participating agents

        Returns:
            Conversation ID or None if not allowed
        """
        # Check budget
        allowed, reason = self.budget_manager.can_start_conversation()
        if not allowed:
            logger.warning(f"⚠️  Cannot start conversation: {reason}")
            if self.on_budget_exceeded:
                self.on_budget_exceeded(reason)
            return None

        # Create conversation
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            participants=participants,
            messages=[],
            started_at=datetime.now(),
            topic=topic,
            metadata={"initiator": initiator}
        )

        self.active_conversations[conversation_id] = conversation

        # Update budget
        budget = self.budget_manager.get_budget(conversation_id)
        budget.conversations_this_hour += 1

        logger.info(f"💬 Started conversation: {topic} ({len(participants)} agents)")

        if self.on_conversation_start:
            self.on_conversation_start(conversation)

        return conversation_id

    async def send_message(
        self,
        conversation_id: str,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        content: str,
        parent_message_id: str | None = None
    ) -> Message | None:
        """
        Send message in conversation.

        Args:
            conversation_id: Conversation ID
            from_agent: Sender
            to_agent: Recipient
            message_type: Message type
            content: Message content
            parent_message_id: Parent message ID

        Returns:
            Sent message or None if blocked
        """
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            logger.error(f"Conversation {conversation_id} not found")
            return None

        # Create message
        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            parent_message_id=parent_message_id
        )

        # Safety checks

        # 1. Budget check
        budget = self.budget_manager.get_budget(conversation_id)
        allowed, reason = self.safety.check_budget(budget, conversation)
        if not allowed:
            logger.warning(f"⚠️  Budget exceeded: {reason}")
            if self.on_budget_exceeded:
                self.on_budget_exceeded(reason)
            await self.end_conversation(conversation_id, reason=f"Budget: {reason}")
            return None

        # 2. Loop detection
        is_loop, reason = self.safety.detect_loop(conversation)
        if is_loop:
            logger.warning(f"⚠️  Loop detected: {reason}")
            if self.on_safety_violation:
                self.on_safety_violation(f"Loop: {reason}")
            await self.end_conversation(conversation_id, reason=f"Loop: {reason}")
            return None

        # 3. Relevance check
        is_relevant, reason = self.safety.check_relevance(message, conversation.topic)
        if not is_relevant:
            logger.warning(f"⚠️  Message not relevant: {reason}")
            # Don't block, just warn

        # Add message
        conversation.messages.append(message)

        # Update budget
        self.budget_manager.update_usage(conversation)

        # Publish to message bus
        await self.message_bus.publish(message)

        # Callbacks
        if self.on_message:
            self.on_message(message)

        if message.message_type == MessageType.INSIGHT and self.on_insight:
            self.on_insight(message)

        logger.debug(f"📨 {from_agent} → {to_agent}: {message_type.value}")

        # Check productivity periodically
        if len(conversation.messages) % 3 == 0:
            is_productive, reason = self.safety.check_productivity(conversation)
            if not is_productive:
                logger.warning(f"⚠️  Conversation not productive: {reason}")
                await self.end_conversation(conversation_id, reason=f"Unproductive: {reason}")

        return message

    async def end_conversation(
        self,
        conversation_id: str,
        reason: str = "Completed"
    ) -> None:
        """
        End conversation.

        Args:
            conversation_id: Conversation ID
            reason: Reason for ending
        """
        conversation = self.active_conversations.pop(conversation_id, None)
        if not conversation:
            return

        conversation.metadata["end_reason"] = reason
        conversation.metadata["ended_at"] = datetime.now().isoformat()

        # Move to history
        self.conversation_history.append(conversation)

        # Keep only recent history
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

        logger.info(f"💬 Ended conversation: {conversation.topic} ({reason})")

        if self.on_conversation_end:
            self.on_conversation_end(conversation)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get conversation by ID."""
        return self.active_conversations.get(conversation_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get conversation statistics."""
        return {
            "active_conversations": len(self.active_conversations),
            "total_conversations": len(self.conversation_history),
            "total_messages": sum(c.message_count for c in self.conversation_history),
            "avg_messages_per_conversation": (
                sum(c.message_count for c in self.conversation_history) / len(self.conversation_history)
                if self.conversation_history else 0
            ),
            "avg_duration_seconds": (
                sum(c.duration_seconds for c in self.conversation_history) / len(self.conversation_history)
                if self.conversation_history else 0
            )
        }
