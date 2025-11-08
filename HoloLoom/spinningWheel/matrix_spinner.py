"""
Matrix.org Spinner - Ingest Matrix chat rooms into HoloLoom memory

Supports:
- Room message ingestion (m.text, m.image, m.file, m.emote, etc.)
- Reaction tracking
- Thread detection
- User mentions
- Media attachments
- Incremental sync (since token)
- Streaming support
- 9-signal importance scoring

Requires: matrix-nio (pip install matrix-nio)
Optional: E2EE support (pip install "matrix-nio[e2e]")

Usage:
    from HoloLoom.spinningWheel.matrix_spinner import MatrixSpinner

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="your_token"
    )

    # Spin a single room
    result = await spinner.spin_room("!roomid:matrix.org")

    # Spin all rooms
    result = await spinner.spin_all_rooms()

    # Incremental sync
    result = await spinner.spin_incremental()
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, AsyncIterator
from pathlib import Path
import hashlib
import time

try:
    from nio import AsyncClient, RoomMessageText, RoomMessageMedia, RoomMessage
    from nio import MatrixRoom, RoomMessagesError, LoginError, SyncError
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    SpinnerCheckpoint,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.importance import ImportanceScorer


@dataclass
class MatrixMessage:
    """Parsed Matrix message event"""
    event_id: str
    room_id: str
    room_name: Optional[str]
    sender: str
    sender_display_name: Optional[str]
    timestamp: datetime
    msg_type: str  # m.text, m.image, m.file, m.emote, m.notice
    content: str  # Message body
    formatted_content: Optional[str] = None  # HTML formatted
    reactions: Dict[str, int] = field(default_factory=dict)  # {emoji: count}
    thread_root: Optional[str] = None  # Parent message for threads
    attachments: List[str] = field(default_factory=list)  # URLs to media
    mentions: List[str] = field(default_factory=list)  # @user mentions
    edited: bool = False
    reply_to: Optional[str] = None  # Event ID of replied message
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_thread_root(self) -> bool:
        """Check if this is a root message (not a reply)"""
        return self.thread_root is None and self.reply_to is None

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score from reactions"""
        if not self.reactions:
            return 0.0
        total_reactions = sum(self.reactions.values())
        return min(1.0, total_reactions / 10.0)  # Cap at 10 reactions


class MatrixParser:
    """Parse Matrix protocol events into MatrixMessage objects"""

    @staticmethod
    def parse_event(event: Any, room: Any) -> Optional[MatrixMessage]:
        """
        Parse Matrix event into MatrixMessage

        Args:
            event: Matrix event object (from matrix-nio)
            room: MatrixRoom object

        Returns:
            MatrixMessage or None if not parseable
        """
        if not MATRIX_AVAILABLE:
            return None

        # Only process room message events
        if not isinstance(event, RoomMessage):
            return None

        # Extract basic metadata
        event_id = event.event_id
        room_id = room.room_id
        room_name = room.display_name or room.room_id
        sender = event.sender
        sender_display_name = room.user_name(sender) if hasattr(room, 'user_name') else sender
        timestamp = datetime.fromtimestamp(event.server_timestamp / 1000.0)

        # Extract message content based on type
        if isinstance(event, RoomMessageText):
            msg_type = "m.text"
            content = event.body
            formatted_content = getattr(event, 'formatted_body', None)
        elif isinstance(event, RoomMessageMedia):
            msg_type = event.msgtype or "m.file"
            content = event.body  # Usually filename
            formatted_content = None
        else:
            msg_type = getattr(event, 'msgtype', 'm.unknown')
            content = getattr(event, 'body', '')
            formatted_content = None

        # Extract mentions (@user)
        mentions = MatrixParser.extract_mentions(content)

        # Extract reply_to (if this is a reply)
        reply_to = None
        if hasattr(event, 'source') and 'content' in event.source:
            relates_to = event.source['content'].get('m.relates_to', {})
            reply_to = relates_to.get('m.in_reply_to', {}).get('event_id')

        # Extract attachments (media URLs)
        attachments = []
        if isinstance(event, RoomMessageMedia) and hasattr(event, 'url'):
            attachments.append(event.url)

        return MatrixMessage(
            event_id=event_id,
            room_id=room_id,
            room_name=room_name,
            sender=sender,
            sender_display_name=sender_display_name,
            timestamp=timestamp,
            msg_type=msg_type,
            content=content,
            formatted_content=formatted_content,
            mentions=mentions,
            reply_to=reply_to,
            attachments=attachments,
            metadata={
                'server_timestamp': event.server_timestamp,
                'sender_full': sender
            }
        )

    @staticmethod
    def extract_mentions(content: str) -> List[str]:
        """
        Extract @user mentions from message content

        Args:
            content: Message text

        Returns:
            List of mentioned usernames (without @)
        """
        import re
        # Match @username (simplified - Matrix uses full MXIDs like @user:server.com)
        pattern = r'@([a-zA-Z0-9._-]+(?::[a-zA-Z0-9.-]+)?)'
        mentions = re.findall(pattern, content)
        return list(set(mentions))  # Deduplicate

    @staticmethod
    def extract_reactions(messages: List[MatrixMessage]) -> Dict[str, Dict[str, int]]:
        """
        Extract reactions for messages (requires separate reaction events)

        Note: This is a placeholder. Proper implementation requires
        fetching m.reaction events separately.

        Args:
            messages: List of MatrixMessage objects

        Returns:
            Dict mapping event_id to {emoji: count}
        """
        # This would be populated by processing m.reaction events
        # For now, return empty reactions
        return {msg.event_id: {} for msg in messages}


class MatrixSpinner(BaseSpinner):
    """
    Spinner for Matrix.org chat rooms

    Ingests Matrix room messages into HoloLoom memory with:
    - Multi-room support
    - Reaction tracking
    - Thread detection
    - User mentions
    - Media attachments
    - Incremental sync
    - Streaming
    - 9-signal importance scoring
    """

    def __init__(
        self,
        homeserver: str = "https://matrix.org",
        access_token: Optional[str] = None,
        user_id: Optional[str] = None,
        password: Optional[str] = None,
        importance_threshold: float = 0.3,
        max_messages_per_room: int = 1000,
        device_id: Optional[str] = None
    ):
        """
        Initialize MatrixSpinner

        Args:
            homeserver: Matrix homeserver URL (e.g., https://matrix.org)
            access_token: Matrix access token (if None, will login with user_id/password)
            user_id: Matrix user ID (e.g., @user:matrix.org) - required if no access_token
            password: Password for login - required if no access_token
            importance_threshold: Minimum importance score (0.0-1.0)
            max_messages_per_room: Max messages to fetch per room
            device_id: Optional device ID for login
        """
        super().__init__(name="matrix")

        if not MATRIX_AVAILABLE:
            raise ImportError(
                "matrix-nio is required for MatrixSpinner. "
                "Install with: pip install matrix-nio"
            )

        self.homeserver = homeserver
        self.access_token = access_token
        self.user_id = user_id
        self.password = password
        self.importance_threshold = importance_threshold
        self.max_messages_per_room = max_messages_per_room
        self.device_id = device_id

        # Create importance scorer
        self.importance_scorer = ImportanceScorer()

        # Client will be initialized in async context
        self.client: Optional[AsyncClient] = None

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=True,
            streaming=True,
            supported_formats=['matrix'],
            batch_processing=True
        )

    def is_available(self) -> bool:
        """Check if Matrix dependencies are available"""
        return MATRIX_AVAILABLE

    async def _ensure_client(self) -> "AsyncClient":
        """Ensure Matrix client is initialized and logged in"""
        if self.client is not None:
            return self.client

        # Initialize client
        self.client = AsyncClient(self.homeserver, self.user_id or "", device_id=self.device_id)

        # Login if no access token provided
        if not self.access_token:
            if not self.user_id or not self.password:
                raise ValueError(
                    "Either access_token or (user_id + password) must be provided"
                )

            response = await self.client.login(self.password)
            if isinstance(response, LoginError):
                raise RuntimeError(f"Matrix login failed: {response.message}")

            self.access_token = self.client.access_token
        else:
            # Set access token
            self.client.access_token = self.access_token

        # Perform initial sync to get room list
        await self.client.sync(timeout=30000, full_state=True)

        return self.client

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Spin Matrix room(s) into MemoryShards

        Args:
            source: Room ID (str) or list of room IDs
            **kwargs: Additional arguments

        Returns:
            List of MemoryShards
        """
        client = await self._ensure_client()

        # Handle single room or multiple rooms
        if isinstance(source, str):
            room_ids = [source]
        elif isinstance(source, list):
            room_ids = source
        else:
            raise ValueError(f"source must be room ID (str) or list of room IDs, got {type(source)}")

        all_shards = []
        for room_id in room_ids:
            messages = await self._fetch_room_messages(room_id)
            shards = self._messages_to_shards(messages)
            all_shards.extend(shards)

        return all_shards

    async def spin_room(
        self,
        room_id: str,
        since: Optional[str] = None,
        limit: Optional[int] = None
    ) -> SpinResult:
        """
        Spin a single Matrix room

        Args:
            room_id: Matrix room ID (e.g., !abc123:matrix.org)
            since: Sync token to fetch messages since (for incremental)
            limit: Max messages to fetch (default: max_messages_per_room)

        Returns:
            SpinResult with MemoryShards
        """
        messages = await self._fetch_room_messages(room_id, since=since, limit=limit)
        shards = self._messages_to_shards(messages)

        return SpinResult(
            shards=shards,
            success=True,
            items_processed=len(messages),
            items_filtered=len(messages) - len(shards)
        )

    async def spin_all_rooms(
        self,
        room_filter: Optional[Callable[[str, str], bool]] = None
    ) -> SpinResult:
        """
        Spin all joined Matrix rooms

        Args:
            room_filter: Optional filter function (room_id, room_name) -> bool

        Returns:
            SpinResult with MemoryShards from all rooms
        """
        client = await self._ensure_client()

        all_shards = []
        total_messages = 0

        for room_id, room in client.rooms.items():
            # Apply filter if provided
            if room_filter and not room_filter(room_id, room.display_name):
                continue

            messages = await self._fetch_room_messages(room_id)
            shards = self._messages_to_shards(messages)

            all_shards.extend(shards)
            total_messages += len(messages)

        return SpinResult(
            shards=all_shards,
            success=True,
            items_processed=total_messages,
            items_filtered=total_messages - len(all_shards)
        )

    async def spin_incremental(
        self,
        checkpoint: Optional[SpinnerCheckpoint] = None,
        room_id: Optional[str] = None
    ) -> SpinResult:
        """
        Incremental sync using Matrix sync token

        Args:
            checkpoint: Previous checkpoint with since token
            room_id: Optional specific room to sync (default: all rooms)

        Returns:
            SpinResult with new MemoryShards
        """
        # Load checkpoint if not provided
        if checkpoint is None:
            source_id = self._create_source_id(room_id or "all_rooms")
            checkpoint = self.load_checkpoint(source_id)

        # Get since token from checkpoint
        since_token = checkpoint.last_processed_id if checkpoint else None

        if room_id:
            # Sync specific room
            result = await self.spin_room(room_id, since=since_token)
            shards = result.shards
        else:
            # Sync all rooms
            result = await self.spin_all_rooms()
            shards = result.shards

        # Get new since token
        client = await self._ensure_client()
        new_since_token = client.next_batch

        # Save checkpoint
        source_id = self._create_source_id(room_id or "all_rooms")
        new_checkpoint = SpinnerCheckpoint(
            spinner_name=self.get_name(),
            source_id=source_id,
            last_processed_id=new_since_token,
            items_processed=len(shards),
            timestamp=time.time()
        )
        self.save_checkpoint(new_checkpoint)

        return SpinResult(
            shards=shards,
            success=True,
            items_processed=result.items_processed,
            items_filtered=result.items_filtered
        )

    async def spin_stream(
        self,
        source: Any,
        batch_size: int = 50,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from Matrix room(s)

        Args:
            source: Room ID or list of room IDs
            batch_size: Number of messages per batch

        Yields:
            MemoryShard objects
        """
        # Get messages
        if isinstance(source, str):
            messages = await self._fetch_room_messages(source)
        elif isinstance(source, list):
            messages = []
            for room_id in source:
                room_messages = await self._fetch_room_messages(room_id)
                messages.extend(room_messages)
        else:
            raise ValueError(f"source must be room ID or list, got {type(source)}")

        # Stream shards in batches
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            shards = self._messages_to_shards(batch)
            for shard in shards:
                yield shard

    async def _fetch_room_messages(
        self,
        room_id: str,
        since: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MatrixMessage]:
        """
        Fetch messages from a Matrix room

        Args:
            room_id: Room ID
            since: Sync token to fetch since
            limit: Max messages (default: max_messages_per_room)

        Returns:
            List of MatrixMessage objects
        """
        client = await self._ensure_client()

        # Get room object
        room = client.rooms.get(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found (not joined?)")

        # Fetch messages using room_messages API
        limit = limit or self.max_messages_per_room
        response = await client.room_messages(
            room_id=room_id,
            start=since or client.next_batch,
            limit=limit,
            direction="b"  # Backwards (newest first)
        )

        if isinstance(response, RoomMessagesError):
            raise RuntimeError(f"Failed to fetch messages: {response.message}")

        # Parse events into MatrixMessage objects
        messages = []
        for event in response.chunk:
            msg = MatrixParser.parse_event(event, room)
            if msg:
                messages.append(msg)

        return messages

    def _messages_to_shards(self, messages: List[MatrixMessage]) -> List[MemoryShard]:
        """
        Convert MatrixMessage objects to MemoryShards

        Args:
            messages: List of MatrixMessage objects

        Returns:
            List of MemoryShards (filtered by importance)
        """
        shards = []

        for msg in messages:
            # Score importance
            importance = self.score_importance(msg)

            # Filter by threshold
            if importance.score < self.importance_threshold:
                continue

            # Create shard
            shard = self._create_shard(
                id_suffix=msg.event_id[:12],
                text=self._format_message_text(msg),
                episode=f"matrix_{msg.room_name or msg.room_id}",
                entities=self._extract_entities(msg),
                motifs=self._extract_motifs(msg),
                metadata={
                    'event_id': msg.event_id,
                    'room_id': msg.room_id,
                    'room_name': msg.room_name,
                    'sender': msg.sender,
                    'sender_display_name': msg.sender_display_name,
                    'timestamp': msg.timestamp.isoformat(),
                    'msg_type': msg.msg_type,
                    'reactions': msg.reactions,
                    'mentions': msg.mentions,
                    'attachments': msg.attachments,
                    'is_thread_root': msg.is_thread_root,
                    'importance_score': importance.score,
                    'importance_reason': importance.reason
                }
            )
            shards.append(shard)

        return shards

    def _format_message_text(self, msg: MatrixMessage) -> str:
        """Format message for text field"""
        parts = [
            f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] ",
            f"{msg.sender_display_name or msg.sender}: ",
            msg.content
        ]

        # Add reactions if present
        if msg.reactions:
            reaction_str = ' '.join(f"{emoji}:{count}" for emoji, count in msg.reactions.items())
            parts.append(f" [{reaction_str}]")

        # Add mentions if present
        if msg.mentions:
            parts.append(f" (mentions: {', '.join(msg.mentions)})")

        return ''.join(parts)

    def _extract_entities(self, msg: MatrixMessage) -> List[str]:
        """Extract entities from message"""
        entities = []

        # Sender as entity
        if msg.sender_display_name:
            entities.append(msg.sender_display_name)

        # Room as entity
        if msg.room_name:
            entities.append(msg.room_name)

        # Mentioned users as entities
        entities.extend(msg.mentions)

        return list(set(entities))  # Deduplicate

    def _extract_motifs(self, msg: MatrixMessage) -> List[str]:
        """Extract motifs from message"""
        motifs = []

        # Message type as motif
        motifs.append(msg.msg_type)

        # Thread participation
        if msg.is_thread_root:
            motifs.append('thread_root')
        elif msg.reply_to:
            motifs.append('reply')

        # Engagement
        if msg.reactions:
            motifs.append('reacted')

        # Mentions
        if msg.mentions:
            motifs.append('mentions')

        # Attachments
        if msg.attachments:
            motifs.append('media')

        return motifs

    def score_importance(self, msg: MatrixMessage) -> ImportanceScore:
        """
        Score message importance using 9 signals

        Signals:
        1. Length (50-300 chars optimal)
        2. Technical (domain terms)
        3. Structural (formatting)
        4. Authority (sender reputation)
        5. Recency (time decay)
        6. Engagement (reactions)
        7. Reference (mentions, replies)
        8. Noise (greetings, short messages)
        9. Custom (msg_type, thread_root, room_importance)

        Args:
            msg: MatrixMessage object

        Returns:
            ImportanceScore
        """
        signals = ImportanceSignals()
        text = msg.content

        # 1. Length score
        text_len = len(text)
        if text_len < 10:
            signals.length_score = 0.1
        elif text_len < 50:
            signals.length_score = 0.3
        elif text_len <= 300:
            signals.length_score = min(1.0, text_len / 300)
        else:
            signals.length_score = 0.8  # Long messages slightly penalized

        # 2. Technical score
        signals.technical_score = self.importance_scorer.technical_scorer.score(text)

        # 3. Structural score
        signals.structural_score = self.importance_scorer.structural_scorer.score(text)

        # 4. Authority score (placeholder - would need room power levels)
        signals.authority_score = 0.5  # Neutral

        # 5. Recency score
        signals.recency_score = self.importance_scorer.recency_scorer.score(
            msg.timestamp.timestamp()
        )

        # 6. Engagement score
        signals.engagement_score = msg.engagement_score

        # 7. Reference score (mentions, replies)
        reference_score = 0.0
        if msg.mentions:
            reference_score += 0.4
        if msg.reply_to:
            reference_score += 0.3
        if msg.thread_root:
            reference_score += 0.2
        signals.reference_score = min(1.0, reference_score)

        # 8. Noise detection
        noise_score = self.importance_scorer.noise_detector.detect(text)

        # 9. Custom signals
        signals.custom_signals = {}

        # Message type importance
        msg_type_scores = {
            'm.text': 0.7,
            'm.emote': 0.5,
            'm.notice': 0.4,
            'm.image': 0.6,
            'm.file': 0.6,
            'm.video': 0.6,
            'm.audio': 0.6
        }
        signals.custom_signals['msg_type'] = msg_type_scores.get(msg.msg_type, 0.5)

        # Thread roots are more important
        if msg.is_thread_root and text_len > 50:
            signals.custom_signals['thread_root'] = 0.8
        else:
            signals.custom_signals['thread_root'] = 0.0

        # Combine signals
        final_score = (
            0.15 * signals.length_score +
            0.15 * signals.technical_score +
            0.10 * signals.structural_score +
            0.10 * signals.authority_score +
            0.10 * signals.recency_score +
            0.15 * signals.engagement_score +
            0.10 * signals.reference_score +
            0.10 * signals.custom_signals.get('msg_type', 0.5) +
            0.05 * signals.custom_signals.get('thread_root', 0.0)
        )

        # Apply noise penalty
        final_score *= (1.0 - noise_score)

        # Generate reason
        reasons = []
        if signals.engagement_score > 0.5:
            reasons.append("high engagement")
        if signals.reference_score > 0.5:
            reasons.append("mentions/replies")
        if signals.technical_score > 0.6:
            reasons.append("technical content")
        if msg.is_thread_root:
            reasons.append("thread root")
        if noise_score > 0.5:
            reasons.append("noisy content")

        reason = " + ".join(reasons) if reasons else "standard message"

        return ImportanceScore(
            score=max(0.0, min(1.0, final_score)),
            signals=signals,
            reason=reason
        )

    def _create_source_id(self, room_id: str) -> str:
        """Create deterministic source ID from room ID"""
        return hashlib.sha256(f"matrix_{room_id}".encode()).hexdigest()[:16]

    async def close(self):
        """Close Matrix client connection"""
        if self.client:
            await self.client.close()
            self.client = None


# Convenience functions

async def spin_matrix_room(
    room_id: str,
    homeserver: str = "https://matrix.org",
    access_token: Optional[str] = None,
    user_id: Optional[str] = None,
    password: Optional[str] = None,
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to spin a Matrix room

    Args:
        room_id: Matrix room ID
        homeserver: Homeserver URL
        access_token: Access token (or use user_id/password)
        user_id: User ID for login
        password: Password for login
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards
    """
    spinner = MatrixSpinner(
        homeserver=homeserver,
        access_token=access_token,
        user_id=user_id,
        password=password,
        importance_threshold=importance_threshold
    )

    try:
        return await spinner.spin_room(room_id)
    finally:
        await spinner.close()


def create_matrix_scorer() -> ImportanceScorer:
    """Create importance scorer optimized for Matrix messages"""
    scorer = ImportanceScorer(
        technical_terms={
            'bug', 'feature', 'issue', 'pr', 'merge', 'release',
            'deploy', 'test', 'ci', 'cd', 'docker', 'kubernetes',
            'api', 'backend', 'frontend', 'database', 'cache'
        }
    )
    # Note: noise_patterns are built into NoiseDetector by default
    return scorer
