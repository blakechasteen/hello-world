"""
SlackSpinner - Slack Channel Ingestion
=====================================

Extracts messages from Slack channels into MemoryShards.

Features:
- Channel message extraction with threading support
- User/author information and reactions
- Time-based filtering (e.g., last N days)
- Thread reply extraction
- Emoji reaction tracking
- Graceful handling of API limits
- Conversation history with timestamps

Author: Claude Code
Date: November 2025
"""

import asyncio
import time
from typing import List, Dict, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import os

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore,
    SpinnerStatus
)
from HoloLoom.spinningWheel.utils import create_source_id


# ============================================================================
# Slack Data Structures
# ============================================================================

@dataclass
class SlackMessage:
    """Parsed Slack message."""
    ts: str  # Message timestamp
    user: Optional[str]  # User ID who sent message
    username: Optional[str]  # User name
    text: str  # Message text
    thread_ts: Optional[str] = None  # Parent thread timestamp
    reply_count: int = 0  # Number of thread replies
    reactions: Dict[str, int] = field(default_factory=dict)  # Emoji -> count
    files_count: int = 0
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None


@dataclass
class SlackThread:
    """Parsed Slack thread with parent and replies."""
    parent: SlackMessage
    replies: List[SlackMessage] = field(default_factory=list)


# ============================================================================
# Slack API Client
# ============================================================================

class SlackClient:
    """Client for Slack API."""

    def __init__(self, token: str):
        """
        Initialize Slack client.

        Args:
            token: Slack Bot User OAuth Token
        """
        self.token = token

        if SLACK_SDK_AVAILABLE:
            self.client = WebClient(token=token)
        elif REQUESTS_AVAILABLE:
            self.client = None
            self.headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        else:
            raise RuntimeError("slack-sdk or requests library required")

    async def get_channel_history(
        self,
        channel_id: str,
        days_back: int = 30,
        max_messages: int = 1000
    ) -> List[SlackMessage]:
        """
        Fetch message history from channel.

        Args:
            channel_id: Channel ID (e.g., "C0123456789")
            days_back: Fetch messages from last N days
            max_messages: Maximum messages to fetch

        Returns:
            List of SlackMessage objects
        """
        if SLACK_SDK_AVAILABLE:
            return await self._get_history_sdk(channel_id, days_back, max_messages)
        elif REQUESTS_AVAILABLE:
            return await self._get_history_requests(channel_id, days_back, max_messages)
        else:
            return []

    async def _get_history_sdk(
        self,
        channel_id: str,
        days_back: int,
        max_messages: int
    ) -> List[SlackMessage]:
        """Get history using slack-sdk."""
        oldest = time.time() - (days_back * 86400)
        messages = []

        try:
            # Fetch messages
            result = self.client.conversations_history(
                channel=channel_id,
                oldest=oldest,
                limit=100
            )

            for msg_data in result.get("messages", []):
                msg = self._parse_message(msg_data, channel_id)
                messages.append(msg)

                if len(messages) >= max_messages:
                    break

        except SlackApiError as e:
            print(f"Slack API error: {e}")

        return messages[:max_messages]

    async def _get_history_requests(
        self,
        channel_id: str,
        days_back: int,
        max_messages: int
    ) -> List[SlackMessage]:
        """Get history using requests (fallback)."""
        oldest = time.time() - (days_back * 86400)
        messages = []

        try:
            response = requests.get(
                "https://slack.com/api/conversations.history",
                headers=self.headers,
                params={
                    "channel": channel_id,
                    "oldest": oldest,
                    "limit": 100
                },
                timeout=10
            )

            data = response.json()
            if not data.get("ok"):
                print(f"Slack API error: {data.get('error')}")
                return []

            for msg_data in data.get("messages", []):
                msg = self._parse_message(msg_data, channel_id)
                messages.append(msg)

                if len(messages) >= max_messages:
                    break

        except Exception as e:
            print(f"Error fetching Slack history: {e}")

        return messages[:max_messages]

    async def get_thread_replies(
        self,
        channel_id: str,
        thread_ts: str
    ) -> List[SlackMessage]:
        """
        Fetch replies in a thread.

        Args:
            channel_id: Channel ID
            thread_ts: Thread timestamp

        Returns:
            List of reply messages
        """
        replies = []

        if SLACK_SDK_AVAILABLE:
            try:
                result = self.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=100
                )

                for msg_data in result.get("messages", [])[1:]:  # Skip parent
                    msg = self._parse_message(msg_data, channel_id, thread_ts)
                    replies.append(msg)

            except SlackApiError as e:
                print(f"Error fetching thread replies: {e}")

        elif REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    "https://slack.com/api/conversations.replies",
                    headers=self.headers,
                    params={
                        "channel": channel_id,
                        "ts": thread_ts,
                        "limit": 100
                    },
                    timeout=10
                )

                data = response.json()
                if data.get("ok"):
                    for msg_data in data.get("messages", [])[1:]:
                        msg = self._parse_message(msg_data, channel_id, thread_ts)
                        replies.append(msg)

            except Exception as e:
                print(f"Error fetching thread replies: {e}")

        return replies

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.

        Args:
            user_id: User ID

        Returns:
            User info dict or None
        """
        if SLACK_SDK_AVAILABLE:
            try:
                result = self.client.users_info(user=user_id)
                return result.get("user", {})
            except Exception:
                return None

        elif REQUESTS_AVAILABLE:
            try:
                response = requests.get(
                    "https://slack.com/api/users.info",
                    headers=self.headers,
                    params={"user": user_id},
                    timeout=10
                )
                data = response.json()
                return data.get("user") if data.get("ok") else None
            except Exception:
                return None

        return None

    def _parse_message(
        self,
        msg_data: Dict[str, Any],
        channel_id: str,
        thread_ts: Optional[str] = None
    ) -> SlackMessage:
        """Parse Slack message data."""
        return SlackMessage(
            ts=msg_data.get("ts", ""),
            user=msg_data.get("user"),
            username=msg_data.get("username"),
            text=msg_data.get("text", ""),
            thread_ts=thread_ts or msg_data.get("thread_ts"),
            reply_count=msg_data.get("reply_count", 0),
            reactions={r["name"]: r["count"] for r in msg_data.get("reactions", [])},
            files_count=len(msg_data.get("files", [])),
            channel_id=channel_id,
            channel_name=None
        )


# ============================================================================
# Slack Spinner
# ============================================================================

class SlackSpinner(BaseSpinner):
    """
    Spin Slack channel messages into MemoryShards.

    Features:
    - Channel history extraction
    - Thread reply extraction
    - Emoji reaction tracking
    - Time-based filtering
    - User information enrichment
    - Importance scoring based on engagement

    Usage:
        >>> spinner = SlackSpinner("xoxb-token")
        >>> result = await spinner.spin("C0123456789")  # Channel ID
        >>> print(f"Created {result.shard_count} shards")
    """

    def __init__(
        self,
        slack_token: str,
        importance_threshold: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
        include_threads: bool = True,
        days_back: int = 30,
        max_messages_per_channel: int = 100
    ):
        """
        Initialize SlackSpinner.

        Args:
            slack_token: Slack Bot User OAuth Token
            importance_threshold: Minimum importance score (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
            include_threads: Include thread replies
            days_back: Fetch messages from last N days
            max_messages_per_channel: Maximum messages per channel
        """
        super().__init__(
            name="slack",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.client = SlackClient(slack_token)
        self.include_threads = include_threads
        self.days_back = days_back
        self.max_messages_per_channel = max_messages_per_channel
        self._user_cache: Dict[str, Dict[str, Any]] = {}

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=True,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            max_input_size=None,
            supported_formats=['slack_channel_id']
        )

    def is_available(self) -> bool:
        """Check if Slack SDK available."""
        return SLACK_SDK_AVAILABLE or REQUESTS_AVAILABLE

    # ========================================================================
    # Core Spinning
    # ========================================================================

    async def _spin_impl(
        self,
        source: str,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Process Slack channel.

        Args:
            source: Channel ID (e.g., "C0123456789")
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        channel_id = source

        # Fetch message history
        print(f"Fetching message history from {channel_id}...")
        messages = await self.client.get_channel_history(
            channel_id,
            days_back=self.days_back,
            max_messages=self.max_messages_per_channel
        )

        shards = []

        for msg in messages:
            # Skip bot messages and empty messages
            if not msg.text or msg.user is None:
                continue

            # Create shard for message
            shard = await self._message_to_shard(msg, channel_id)
            shards.append(shard)

            # Fetch and add thread replies if enabled
            if self.include_threads and msg.thread_ts and msg.reply_count > 0:
                print(f"Fetching {msg.reply_count} thread replies...")
                replies = await self.client.get_thread_replies(channel_id, msg.thread_ts)

                for reply in replies:
                    reply_shard = await self._reply_to_shard(reply, channel_id, msg.thread_ts)
                    shards.append(reply_shard)

        return shards

    # ========================================================================
    # Conversion Methods
    # ========================================================================

    async def _message_to_shard(
        self,
        msg: SlackMessage,
        channel_id: str
    ) -> MemoryShard:
        """Convert Slack message to MemoryShard."""
        # Get user info
        user_info = await self._get_cached_user_info(msg.user)
        username = user_info.get("real_name", msg.username or msg.user)

        # Build text
        text_parts = [
            f"Slack Message from @{username}",
            f"Channel: #{msg.channel_name or channel_id}",
            f"Time: {self._format_timestamp(msg.ts)}",
        ]

        if msg.reactions:
            reaction_str = " ".join([f"{emoji}({count})" for emoji, count in msg.reactions.items()])
            text_parts.append(f"Reactions: {reaction_str}")

        if msg.reply_count > 0:
            text_parts.append(f"Replies: {msg.reply_count}")

        text_parts.extend(["", msg.text])

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            username,
            msg.user or "",
            f"slack_{channel_id}",
            *msg.reactions.keys()  # Emoji reactions
        ]

        # Extract motifs
        motifs = ["slack", "message", "channel"]
        if msg.thread_ts:
            motifs.append("threaded")
        if msg.reactions:
            motifs.append("reacted")

        # Score importance
        importance = self._score_message_importance(msg)

        # Create shard
        return self._create_shard(
            id_suffix=f"slack_{channel_id}_{msg.ts.replace('.', '_')}",
            text=text,
            episode=f"slack_{channel_id}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'slack',
                'type': 'message',
                'channel_id': channel_id,
                'user_id': msg.user,
                'username': username,
                'timestamp': msg.ts,
                'thread_ts': msg.thread_ts,
                'reply_count': msg.reply_count,
                'reactions': msg.reactions,
                'files_count': msg.files_count,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0
            }
        )

    async def _reply_to_shard(
        self,
        msg: SlackMessage,
        channel_id: str,
        thread_ts: str
    ) -> MemoryShard:
        """Convert Slack thread reply to MemoryShard."""
        # Get user info
        user_info = await self._get_cached_user_info(msg.user)
        username = user_info.get("real_name", msg.username or msg.user)

        # Build text
        text_parts = [
            f"Slack Reply from @{username}",
            f"Channel: #{msg.channel_name or channel_id}",
            f"Time: {self._format_timestamp(msg.ts)}",
            f"In thread started at {self._format_timestamp(thread_ts)}",
        ]

        if msg.reactions:
            reaction_str = " ".join([f"{emoji}({count})" for emoji, count in msg.reactions.items()])
            text_parts.append(f"Reactions: {reaction_str}")

        text_parts.extend(["", msg.text])

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            username,
            msg.user or "",
            f"slack_{channel_id}",
            *msg.reactions.keys()
        ]

        # Extract motifs
        motifs = ["slack", "thread_reply", "channel"]
        if msg.reactions:
            motifs.append("reacted")

        # Score importance (slightly lower for replies)
        importance = self._score_message_importance(msg)
        importance.score *= 0.9

        # Create shard
        return self._create_shard(
            id_suffix=f"slack_{channel_id}_{msg.ts.replace('.', '_')}_reply",
            text=text,
            episode=f"slack_{channel_id}_{thread_ts.replace('.', '_')}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'slack',
                'type': 'thread_reply',
                'channel_id': channel_id,
                'user_id': msg.user,
                'username': username,
                'timestamp': msg.ts,
                'thread_ts': thread_ts,
                'reactions': msg.reactions,
                'files_count': msg.files_count,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0
            }
        )

    # ========================================================================
    # Importance Scoring
    # ========================================================================

    def _score_message_importance(self, msg: SlackMessage) -> ImportanceScore:
        """Score Slack message importance."""
        signals = ImportanceSignals()

        # Engagement: reactions + reply count
        total_engagement = len(msg.reactions) + msg.reply_count
        signals.engagement_score = min(1.0, total_engagement / 10)

        # Length: message text length
        signals.length_score = min(1.0, len(msg.text) / 500)

        # Structural: well-formed messages score higher
        has_newlines = '\n' in msg.text
        has_code = '```' in msg.text or '`' in msg.text
        has_links = 'http' in msg.text

        score = 0.3
        if has_newlines:
            score += 0.2
        if has_code:
            score += 0.2
        if has_links:
            score += 0.15

        signals.structural_score = min(1.0, score)

        # Custom signals
        if msg.files_count > 0:
            signals.custom_signals['attachments'] = 0.6

        # Technical content detection
        tech_keywords = ['error', 'bug', 'fix', 'deploy', 'api', 'database', 'server']
        text_lower = msg.text.lower()
        if any(kw in text_lower for kw in tech_keywords):
            signals.technical_score = 0.7

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    async def _get_cached_user_info(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Get user info with caching."""
        if not user_id:
            return {}

        if user_id not in self._user_cache:
            info = await self.client.get_user_info(user_id)
            self._user_cache[user_id] = info or {}

        return self._user_cache[user_id]

    @staticmethod
    def _format_timestamp(ts: str) -> str:
        """Convert Slack timestamp to readable format."""
        try:
            timestamp = float(ts)
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ts


# ============================================================================
# Convenience Functions
# ============================================================================

async def spin_slack_channel(
    channel_id: str,
    slack_token: str,
    days_back: int = 30,
    max_messages: int = 100
) -> SpinResult:
    """
    Quick function to spin a Slack channel.

    Args:
        channel_id: Slack channel ID (e.g., "C0123456789")
        slack_token: Slack Bot User OAuth Token
        days_back: Fetch messages from last N days
        max_messages: Maximum messages to fetch

    Returns:
        SpinResult with shards

    Example:
        >>> result = await spin_slack_channel("C123456", "xoxb-token")
        >>> print(f"Created {result.shard_count} shards")
    """
    spinner = SlackSpinner(
        slack_token=slack_token,
        days_back=days_back,
        max_messages_per_channel=max_messages
    )

    return await spinner.spin(channel_id)
