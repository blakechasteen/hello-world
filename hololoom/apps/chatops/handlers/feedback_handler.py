#!/usr/bin/env python3
"""
Reaction-Based Feedback Handler - Phase 1.1 of ChatOps Roadmap
================================================================
Converts Matrix emoji reactions into Thompson Sampling learning updates.

Philosophy:
    User reactions are implicit quality signals. A thumbs-up indicates
    the response was helpful; thumbs-down indicates it wasn't. These
    signals update Thompson Sampling priors, creating a reinforcement
    loop where the system learns from real user feedback.

Architecture:
    1. BotMessageTracker: Tracks which messages were sent by the bot
    2. ReactionMapper: Maps emoji reactions to reward values
    3. FeedbackProcessor: Processes reactions into learning updates
    4. ThompsonFeedbackUpdater: Applies updates to bandit priors

Reaction Mappings:
    - +1 / thumbsup: +0.75 reward (helpful)
    - -1 / thumbsdown: -0.5 reward (not helpful)
    - star: +1.0 reward (excellent)
    - confused: -0.25 reward (unclear)
    - thinking_face: 0.0 (neutral/noted)
    - eyes: +0.1 reward (interesting)

Date: December 2025
Author: Claude Code
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json

# Graceful nio import
try:
    from nio import (
        AsyncClient,
        MatrixRoom,
        ReactionEvent,
        RoomMessageText,
        Event
    )
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False
    AsyncClient = None
    MatrixRoom = None
    ReactionEvent = None
    RoomMessageText = None

# Handler registry (graceful import)
try:
    from HoloLoom.apps.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
except ImportError:
    HandlerRegistry = None
    HandlerCategory = None
    chatops_handler = None

# Thompson Sampling convergence engine (graceful import)
try:
    from HoloLoom.convergence.engine import ThompsonBandit, ConvergenceEngine
    CONVERGENCE_AVAILABLE = True
except ImportError:
    CONVERGENCE_AVAILABLE = False
    ThompsonBandit = None
    ConvergenceEngine = None

# Hot pattern tracking (graceful import)
try:
    from HoloLoom.recursive.hot_patterns import HotPatternTracker
    HOT_PATTERNS_AVAILABLE = True
except ImportError:
    HOT_PATTERNS_AVAILABLE = False
    HotPatternTracker = None

# Reflection buffer (graceful import)
try:
    from HoloLoom.reflection.buffer import ReflectionBuffer
    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False
    ReflectionBuffer = None

logger = logging.getLogger(__name__)


# ============================================================================
# Reaction Mappings
# ============================================================================

class ReactionType(Enum):
    """Types of feedback reactions."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    EXCELLENT = "excellent"
    CONFUSED = "confused"


@dataclass
class ReactionMapping:
    """Maps an emoji to feedback type and reward."""
    emoji: str
    reaction_type: ReactionType
    reward: float
    description: str


# Unicode emoji mappings (Matrix uses Unicode)
REACTION_MAPPINGS: Dict[str, ReactionMapping] = {
    # Thumbs up variants
    "\U0001F44D": ReactionMapping("\U0001F44D", ReactionType.POSITIVE, 0.75, "thumbs_up"),
    "+1": ReactionMapping("+1", ReactionType.POSITIVE, 0.75, "plus_one"),
    "thumbsup": ReactionMapping("thumbsup", ReactionType.POSITIVE, 0.75, "thumbsup"),

    # Thumbs down variants
    "\U0001F44E": ReactionMapping("\U0001F44E", ReactionType.NEGATIVE, -0.5, "thumbs_down"),
    "-1": ReactionMapping("-1", ReactionType.NEGATIVE, -0.5, "minus_one"),
    "thumbsdown": ReactionMapping("thumbsdown", ReactionType.NEGATIVE, -0.5, "thumbsdown"),

    # Star/excellent
    "\u2B50": ReactionMapping("\u2B50", ReactionType.EXCELLENT, 1.0, "star"),
    "star": ReactionMapping("star", ReactionType.EXCELLENT, 1.0, "star"),
    "\U0001F31F": ReactionMapping("\U0001F31F", ReactionType.EXCELLENT, 1.0, "glowing_star"),

    # Confused/unclear
    "\U0001F615": ReactionMapping("\U0001F615", ReactionType.CONFUSED, -0.25, "confused"),
    "confused": ReactionMapping("confused", ReactionType.CONFUSED, -0.25, "confused"),

    # Thinking/neutral
    "\U0001F914": ReactionMapping("\U0001F914", ReactionType.NEUTRAL, 0.0, "thinking"),
    "thinking_face": ReactionMapping("thinking_face", ReactionType.NEUTRAL, 0.0, "thinking"),

    # Eyes/interesting
    "\U0001F440": ReactionMapping("\U0001F440", ReactionType.POSITIVE, 0.1, "eyes"),
    "eyes": ReactionMapping("eyes", ReactionType.POSITIVE, 0.1, "eyes"),

    # Heart/love
    "\u2764\uFE0F": ReactionMapping("\u2764\uFE0F", ReactionType.EXCELLENT, 0.9, "heart"),
    "\U0001F499": ReactionMapping("\U0001F499", ReactionType.EXCELLENT, 0.9, "blue_heart"),

    # Fire/great
    "\U0001F525": ReactionMapping("\U0001F525", ReactionType.EXCELLENT, 0.85, "fire"),

    # Party/celebration
    "\U0001F389": ReactionMapping("\U0001F389", ReactionType.EXCELLENT, 0.8, "party"),

    # Rocket/fast
    "\U0001F680": ReactionMapping("\U0001F680", ReactionType.POSITIVE, 0.6, "rocket"),

    # Check mark
    "\u2705": ReactionMapping("\u2705", ReactionType.POSITIVE, 0.7, "check_mark"),

    # X mark
    "\u274C": ReactionMapping("\u274C", ReactionType.NEGATIVE, -0.6, "x_mark"),
}


def get_reaction_reward(emoji: str) -> Tuple[float, ReactionType]:
    """
    Get reward value and type for an emoji.

    Args:
        emoji: The emoji string (Unicode or shortcode)

    Returns:
        Tuple of (reward, reaction_type)
    """
    # Try exact match first
    if emoji in REACTION_MAPPINGS:
        mapping = REACTION_MAPPINGS[emoji]
        return (mapping.reward, mapping.reaction_type)

    # Try lowercase shortcode
    emoji_lower = emoji.lower().strip(":")
    if emoji_lower in REACTION_MAPPINGS:
        mapping = REACTION_MAPPINGS[emoji_lower]
        return (mapping.reward, mapping.reaction_type)

    # Unknown emoji - neutral/ignored
    return (0.0, ReactionType.NEUTRAL)


# ============================================================================
# Bot Message Tracking
# ============================================================================

@dataclass
class TrackedMessage:
    """Tracks a bot message for feedback collection."""
    event_id: str
    room_id: str
    query: str
    tool_used: Optional[str] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BotMessageTracker:
    """
    Tracks bot messages to enable reaction-based feedback.

    When the bot sends a response, it registers the message here.
    When reactions come in, we can link them back to the original query.
    """

    def __init__(self, max_messages: int = 1000, ttl_seconds: float = 86400.0):
        """
        Initialize tracker.

        Args:
            max_messages: Maximum messages to track (LRU eviction)
            ttl_seconds: Time-to-live for messages (24 hours default)
        """
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._messages: Dict[str, TrackedMessage] = {}
        self._event_to_message: Dict[str, str] = {}  # event_id -> message key
        self._room_messages: Dict[str, List[str]] = defaultdict(list)  # room_id -> message keys

        logger.info(f"BotMessageTracker initialized (max={max_messages}, ttl={ttl_seconds}s)")

    def track_message(
        self,
        event_id: str,
        room_id: str,
        query: str,
        tool_used: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrackedMessage:
        """
        Track a bot message for feedback collection.

        Args:
            event_id: Matrix event ID of the bot's response
            room_id: Room where message was sent
            query: Original query that generated this response
            tool_used: Tool/adapter that was used
            confidence: Response confidence (0-1)
            metadata: Additional metadata (job_id, etc.)

        Returns:
            TrackedMessage object
        """
        # Clean up old messages first
        self._cleanup_expired()

        # Evict if at capacity
        if len(self._messages) >= self.max_messages:
            self._evict_oldest()

        # Create tracking record
        message = TrackedMessage(
            event_id=event_id,
            room_id=room_id,
            query=query,
            tool_used=tool_used,
            confidence=confidence,
            metadata=metadata or {}
        )

        key = f"{room_id}:{event_id}"
        self._messages[key] = message
        self._event_to_message[event_id] = key
        self._room_messages[room_id].append(key)

        logger.debug(f"Tracking message: {event_id[:20]}... query='{query[:30]}...'")
        return message

    def get_message(self, event_id: str) -> Optional[TrackedMessage]:
        """Get tracked message by event ID."""
        key = self._event_to_message.get(event_id)
        if key:
            return self._messages.get(key)
        return None

    def record_reaction(
        self,
        event_id: str,
        emoji: str,
        user_id: str,
        reaction_event_id: str
    ) -> Optional[TrackedMessage]:
        """
        Record a reaction on a tracked message.

        Args:
            event_id: Event ID of the message being reacted to
            emoji: The reaction emoji
            user_id: User who reacted
            reaction_event_id: Event ID of the reaction itself

        Returns:
            TrackedMessage if found, None otherwise
        """
        message = self.get_message(event_id)
        if message:
            reward, reaction_type = get_reaction_reward(emoji)
            message.reactions.append({
                "emoji": emoji,
                "user_id": user_id,
                "reaction_event_id": reaction_event_id,
                "reward": reward,
                "reaction_type": reaction_type.value,
                "timestamp": time.time()
            })
            logger.debug(f"Recorded reaction {emoji} on message {event_id[:20]}... (reward={reward})")
            return message
        return None

    def get_pending_feedback(self) -> List[TrackedMessage]:
        """Get messages with reactions that haven't been processed."""
        return [m for m in self._messages.values() if m.reactions]

    def _cleanup_expired(self) -> int:
        """Remove expired messages. Returns count removed."""
        now = time.time()
        expired_keys = [
            k for k, m in self._messages.items()
            if now - m.timestamp > self.ttl_seconds
        ]

        for key in expired_keys:
            msg = self._messages.pop(key, None)
            if msg:
                self._event_to_message.pop(msg.event_id, None)
                if key in self._room_messages.get(msg.room_id, []):
                    self._room_messages[msg.room_id].remove(key)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired messages")

        return len(expired_keys)

    def _evict_oldest(self) -> None:
        """Evict oldest message (LRU)."""
        if not self._messages:
            return

        oldest_key = min(self._messages.keys(), key=lambda k: self._messages[k].timestamp)
        msg = self._messages.pop(oldest_key, None)
        if msg:
            self._event_to_message.pop(msg.event_id, None)
            if oldest_key in self._room_messages.get(msg.room_id, []):
                self._room_messages[msg.room_id].remove(oldest_key)

        logger.debug(f"Evicted oldest message for LRU")

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        total_reactions = sum(len(m.reactions) for m in self._messages.values())
        positive = sum(
            1 for m in self._messages.values()
            for r in m.reactions if r["reward"] > 0
        )
        negative = sum(
            1 for m in self._messages.values()
            for r in m.reactions if r["reward"] < 0
        )

        return {
            "tracked_messages": len(self._messages),
            "total_reactions": total_reactions,
            "positive_reactions": positive,
            "negative_reactions": negative,
            "neutral_reactions": total_reactions - positive - negative,
            "rooms_tracked": len(self._room_messages)
        }


# ============================================================================
# Thompson Sampling Feedback Updater
# ============================================================================

class ThompsonFeedbackUpdater:
    """
    Updates Thompson Sampling priors based on reaction feedback.

    Bridges the gap between user reactions and the convergence engine's
    Thompson Bandit, creating a learning loop from real user feedback.
    """

    def __init__(
        self,
        bandit: Optional['ThompsonBandit'] = None,
        hot_tracker: Optional['HotPatternTracker'] = None,
        tool_names: Optional[List[str]] = None
    ):
        """
        Initialize updater.

        Args:
            bandit: Thompson bandit to update (or creates new one)
            hot_tracker: Hot pattern tracker for usage feedback
            tool_names: List of tool names for mapping
        """
        self.tool_names = tool_names or [
            "answer", "research", "clarify", "memory", "verify"
        ]

        # Create or use provided bandit
        if bandit is not None:
            self.bandit = bandit
        elif CONVERGENCE_AVAILABLE and ThompsonBandit:
            self.bandit = ThompsonBandit(n_tools=len(self.tool_names))
        else:
            self.bandit = None
            logger.warning("Thompson Bandit not available - feedback will be logged only")

        self.hot_tracker = hot_tracker

        # Feedback statistics
        self.total_updates = 0
        self.positive_updates = 0
        self.negative_updates = 0

        logger.info(f"ThompsonFeedbackUpdater initialized (tools={self.tool_names})")

    def get_tool_index(self, tool_name: str) -> int:
        """Get index for a tool name."""
        if tool_name in self.tool_names:
            return self.tool_names.index(tool_name)
        # Default to first tool if unknown
        return 0

    def process_feedback(
        self,
        message: TrackedMessage,
        clear_reactions: bool = True
    ) -> Dict[str, Any]:
        """
        Process feedback from reactions on a message.

        Args:
            message: Tracked message with reactions
            clear_reactions: Whether to clear reactions after processing

        Returns:
            Processing result with statistics
        """
        if not message.reactions:
            return {"processed": 0, "total_reward": 0.0}

        # Aggregate rewards from all reactions
        total_reward = sum(r["reward"] for r in message.reactions)
        avg_reward = total_reward / len(message.reactions) if message.reactions else 0.0

        # Get tool index
        tool_name = message.tool_used or "answer"
        tool_idx = self.get_tool_index(tool_name)

        # Update Thompson Bandit
        if self.bandit is not None:
            self.bandit.update(tool_idx, avg_reward)
            logger.info(f"Updated bandit: tool={tool_name}, reward={avg_reward:.3f}")

        # Update hot patterns if available
        if self.hot_tracker is not None:
            confidence_adjustment = message.confidence + (avg_reward * 0.1)
            # Would call: self.hot_tracker.record_access(message.query, confidence_adjustment)
            logger.debug(f"Would update hot patterns for query: {message.query[:30]}...")

        # Update statistics
        self.total_updates += 1
        if avg_reward > 0:
            self.positive_updates += 1
        elif avg_reward < 0:
            self.negative_updates += 1

        result = {
            "processed": len(message.reactions),
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "tool": tool_name,
            "tool_idx": tool_idx,
            "query": message.query[:50],
            "bandit_updated": self.bandit is not None
        }

        # Clear reactions if requested
        if clear_reactions:
            message.reactions.clear()

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics."""
        stats = {
            "total_updates": self.total_updates,
            "positive_updates": self.positive_updates,
            "negative_updates": self.negative_updates,
            "neutral_updates": self.total_updates - self.positive_updates - self.negative_updates,
            "bandit_available": self.bandit is not None
        }

        if self.bandit is not None:
            stats["bandit_stats"] = self.bandit.get_statistics()

        return stats


# ============================================================================
# Feedback Processor (Main Orchestrator)
# ============================================================================

class FeedbackProcessor:
    """
    Main orchestrator for reaction-based feedback processing.

    Coordinates:
    - Message tracking
    - Reaction collection
    - Thompson Sampling updates
    - Feedback logging
    """

    def __init__(
        self,
        bandit: Optional['ThompsonBandit'] = None,
        hot_tracker: Optional['HotPatternTracker'] = None,
        reflection_buffer: Optional['ReflectionBuffer'] = None,
        auto_process_interval: float = 60.0
    ):
        """
        Initialize processor.

        Args:
            bandit: Thompson bandit for learning updates
            hot_tracker: Hot pattern tracker
            reflection_buffer: Reflection buffer for logging
            auto_process_interval: Interval for auto-processing (0 to disable)
        """
        self.tracker = BotMessageTracker()
        self.updater = ThompsonFeedbackUpdater(bandit=bandit, hot_tracker=hot_tracker)
        self.reflection_buffer = reflection_buffer
        self.auto_process_interval = auto_process_interval

        # Background task for auto-processing
        self._auto_process_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info("FeedbackProcessor initialized")

    def track_response(
        self,
        event_id: str,
        room_id: str,
        query: str,
        tool_used: Optional[str] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrackedMessage:
        """Track a bot response for feedback collection."""
        return self.tracker.track_message(
            event_id=event_id,
            room_id=room_id,
            query=query,
            tool_used=tool_used,
            confidence=confidence,
            metadata=metadata
        )

    async def process_reaction(
        self,
        target_event_id: str,
        emoji: str,
        user_id: str,
        reaction_event_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Process an incoming reaction.

        Args:
            target_event_id: Event being reacted to
            emoji: Reaction emoji
            user_id: User who reacted
            reaction_event_id: Reaction event ID

        Returns:
            Processing result if message was tracked, None otherwise
        """
        # Record the reaction
        message = self.tracker.record_reaction(
            event_id=target_event_id,
            emoji=emoji,
            user_id=user_id,
            reaction_event_id=reaction_event_id
        )

        if message is None:
            return None

        reward, reaction_type = get_reaction_reward(emoji)

        # Immediate processing for strong signals
        if reaction_type in (ReactionType.EXCELLENT, ReactionType.NEGATIVE):
            result = self.updater.process_feedback(message)

            # Log to reflection buffer if available
            if self.reflection_buffer:
                await self._log_feedback(message, result)

            return result

        # Weak signals batch-processed later
        return {
            "queued": True,
            "emoji": emoji,
            "reward": reward,
            "reaction_type": reaction_type.value
        }

    async def process_all_pending(self) -> Dict[str, Any]:
        """Process all pending feedback."""
        pending = self.tracker.get_pending_feedback()

        results = []
        for message in pending:
            result = self.updater.process_feedback(message)
            results.append(result)

            # Log to reflection buffer
            if self.reflection_buffer:
                await self._log_feedback(message, result)

        return {
            "processed_messages": len(results),
            "results": results,
            "updater_stats": self.updater.get_statistics()
        }

    async def _log_feedback(self, message: TrackedMessage, result: Dict[str, Any]) -> None:
        """Log feedback to reflection buffer."""
        if self.reflection_buffer is None:
            return

        try:
            # Create feedback record
            feedback = {
                "query": message.query,
                "tool_used": message.tool_used,
                "original_confidence": message.confidence,
                "feedback_reward": result.get("avg_reward", 0.0),
                "reaction_count": result.get("processed", 0),
                "timestamp": time.time()
            }

            # Log as reflection entry
            # await self.reflection_buffer.add_feedback(feedback)
            logger.debug(f"Logged feedback: {feedback}")
        except Exception as e:
            logger.warning(f"Failed to log feedback: {e}")

    async def start_auto_processing(self) -> None:
        """Start background auto-processing task."""
        if self._running:
            return

        self._running = True
        self._auto_process_task = asyncio.create_task(self._auto_process_loop())
        logger.info("Started feedback auto-processing")

    async def stop_auto_processing(self) -> None:
        """Stop background auto-processing task."""
        self._running = False
        if self._auto_process_task:
            self._auto_process_task.cancel()
            try:
                await self._auto_process_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped feedback auto-processing")

    async def _auto_process_loop(self) -> None:
        """Background loop for auto-processing pending feedback."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_process_interval)
                result = await self.process_all_pending()
                if result["processed_messages"] > 0:
                    logger.info(f"Auto-processed {result['processed_messages']} messages")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-processing: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "tracker": self.tracker.get_statistics(),
            "updater": self.updater.get_statistics(),
            "auto_processing": self._running,
            "reflection_enabled": self.reflection_buffer is not None
        }


# ============================================================================
# Global Feedback Processor Instance
# ============================================================================

_global_processor: Optional[FeedbackProcessor] = None


def get_feedback_processor() -> FeedbackProcessor:
    """Get or create global feedback processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = FeedbackProcessor()
    return _global_processor


def set_feedback_processor(processor: FeedbackProcessor) -> None:
    """Set global feedback processor."""
    global _global_processor
    _global_processor = processor


# ============================================================================
# Matrix Bot Integration - Reaction Event Handler
# ============================================================================

async def handle_reaction_event(
    room: 'MatrixRoom',
    event: 'ReactionEvent',
    processor: Optional[FeedbackProcessor] = None
) -> Optional[Dict[str, Any]]:
    """
    Handle a Matrix reaction event.

    This is called when someone reacts to a message in the room.
    We check if it's a bot message and process feedback accordingly.

    Args:
        room: Matrix room
        event: Reaction event
        processor: Feedback processor (uses global if None)

    Returns:
        Processing result if applicable
    """
    if not NIO_AVAILABLE:
        return None

    processor = processor or get_feedback_processor()

    # Extract reaction details from event
    # Matrix reactions have: event.reacts_to (event_id), event.key (emoji)
    try:
        target_event_id = event.reacts_to
        emoji = event.key
        user_id = event.sender
        reaction_event_id = event.event_id
    except AttributeError:
        logger.warning(f"Invalid reaction event format: {event}")
        return None

    result = await processor.process_reaction(
        target_event_id=target_event_id,
        emoji=emoji,
        user_id=user_id,
        reaction_event_id=reaction_event_id
    )

    if result:
        logger.debug(f"Processed reaction from {user_id}: {emoji} -> {result}")

    return result


# ============================================================================
# ChatOps Commands for Feedback Management
# ============================================================================

class FeedbackHandlers:
    """ChatOps handlers for feedback management."""

    def __init__(self, processor: Optional[FeedbackProcessor] = None):
        self.processor = processor or get_feedback_processor()

    async def handle_feedback_stats(
        self,
        room: Any,
        event: Any,
        args: List[str]
    ) -> str:
        """
        Handle !feedback stats command.

        Shows feedback collection and Thompson Sampling statistics.
        """
        stats = self.processor.get_statistics()

        tracker = stats["tracker"]
        updater = stats["updater"]

        lines = [
            "## Feedback Statistics",
            "",
            "### Message Tracking",
            f"- Tracked messages: {tracker['tracked_messages']}",
            f"- Total reactions: {tracker['total_reactions']}",
            f"- Positive reactions: {tracker['positive_reactions']}",
            f"- Negative reactions: {tracker['negative_reactions']}",
            f"- Rooms tracked: {tracker['rooms_tracked']}",
            "",
            "### Thompson Sampling Updates",
            f"- Total updates: {updater['total_updates']}",
            f"- Positive updates: {updater['positive_updates']}",
            f"- Negative updates: {updater['negative_updates']}",
            f"- Bandit available: {'[OK]' if updater['bandit_available'] else '[X]'}",
        ]

        # Add bandit priors if available
        if updater.get("bandit_stats"):
            bandit = updater["bandit_stats"]
            lines.extend([
                "",
                "### Bandit Priors",
            ])
            priors = bandit.get("tool_priors", [])
            tool_names = self.processor.updater.tool_names
            for i, prior in enumerate(priors):
                name = tool_names[i] if i < len(tool_names) else f"tool_{i}"
                pulls = bandit.get("pulls", [])[i] if i < len(bandit.get("pulls", [])) else 0
                lines.append(f"- {name}: {prior:.3f} (pulls: {int(pulls)})")

        lines.extend([
            "",
            f"Auto-processing: {'[OK] Running' if stats['auto_processing'] else '[X] Stopped'}",
        ])

        return "\n".join(lines)

    async def handle_feedback_process(
        self,
        room: Any,
        event: Any,
        args: List[str]
    ) -> str:
        """
        Handle !feedback process command.

        Manually process all pending feedback.
        """
        result = await self.processor.process_all_pending()

        lines = [
            "## Feedback Processing Complete",
            "",
            f"- Messages processed: {result['processed_messages']}",
        ]

        if result["results"]:
            lines.append("")
            lines.append("### Results")
            for r in result["results"][:5]:  # Show first 5
                lines.append(f"- {r.get('tool', '?')}: reward={r.get('avg_reward', 0):.3f}")

        return "\n".join(lines)

    async def handle_feedback_help(
        self,
        room: Any,
        event: Any,
        args: List[str]
    ) -> str:
        """Handle !feedback help command."""
        return """## Feedback Commands

### Usage

React to bot messages to provide feedback:
- [+1] thumbsup - Helpful response (+0.75 reward)
- [-1] thumbsdown - Not helpful (-0.5 reward)
- [star] - Excellent response (+1.0 reward)
- [confused] - Unclear response (-0.25 reward)

### Commands

- `!feedback stats` - Show feedback statistics and Thompson priors
- `!feedback process` - Manually process pending feedback
- `!feedback help` - Show this help

### How It Works

1. Bot tracks responses it sends
2. You react with emoji to indicate quality
3. Reactions update Thompson Sampling priors
4. System learns which tools work best for different queries
"""


# Convenience handler functions
async def handle_feedback_stats(room: Any, event: Any, args: List[str]) -> str:
    """Handle !feedback stats command."""
    handlers = FeedbackHandlers()
    return await handlers.handle_feedback_stats(room, event, args)


async def handle_feedback_process(room: Any, event: Any, args: List[str]) -> str:
    """Handle !feedback process command."""
    handlers = FeedbackHandlers()
    return await handlers.handle_feedback_process(room, event, args)


async def handle_feedback_help(room: Any, event: Any, args: List[str]) -> str:
    """Handle !feedback help command."""
    handlers = FeedbackHandlers()
    return await handlers.handle_feedback_help(room, event, args)


# ============================================================================
# Handler Registration
# ============================================================================

def register_feedback_handlers(bot: Any, registry: Any = None) -> None:
    """
    Register feedback handlers with a Matrix bot.

    Args:
        bot: Matrix bot instance
        registry: Handler registry (uses global if None)
    """
    handlers = FeedbackHandlers()

    if registry:
        registry.register("feedback stats", handlers.handle_feedback_stats)
        registry.register("feedback process", handlers.handle_feedback_process)
        registry.register("feedback help", handlers.handle_feedback_help)

    # Register reaction callback on bot if supported
    if hasattr(bot, "add_event_callback") and NIO_AVAILABLE and ReactionEvent:
        async def reaction_callback(room, event):
            if isinstance(event, ReactionEvent):
                await handle_reaction_event(room, event)

        bot.add_event_callback(reaction_callback, ReactionEvent)
        logger.info("Registered reaction event callback on bot")

    logger.info("Registered feedback handlers")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Core classes
    "FeedbackProcessor",
    "BotMessageTracker",
    "ThompsonFeedbackUpdater",
    "TrackedMessage",

    # Reaction mappings
    "ReactionType",
    "ReactionMapping",
    "REACTION_MAPPINGS",
    "get_reaction_reward",

    # Global instance
    "get_feedback_processor",
    "set_feedback_processor",

    # Matrix integration
    "handle_reaction_event",

    # ChatOps handlers
    "FeedbackHandlers",
    "handle_feedback_stats",
    "handle_feedback_process",
    "handle_feedback_help",
    "register_feedback_handlers",
]
