#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Memory for ChatOps
================================
Knowledge graph schema and management for conversation storage.

This is component #4: Persistent conversation memory using HoloLoom KG:
- Conversation nodes (rooms/channels)
- Message nodes (individual messages)
- User nodes (participants)
- Relationship tracking (SENT_BY, FOLLOWS, MENTIONS, etc.)
- Temporal indexing for retrieval
- Spectral analysis of conversation structure

Architecture:
    Matrix Message -> ConversationMemory -> HoloLoom KG -> Spectral Features

Enables:
    - Semantic search across conversation history
    - User interaction patterns
    - Topic evolution tracking
    - Context-aware responses
"""

import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

try:
    from holoLoom.memory.graph import KG, KGEdge
    from holoLoom.documentation.types import MemoryShard
    import networkx as nx
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("Warning: HoloLoom not available")


logger = logging.getLogger(__name__)


# ============================================================================
# Entity Types
# ============================================================================

class EntityType:
    """Entity types in conversation knowledge graph."""
    CONVERSATION = "CONVERSATION"  # A room/channel
    MESSAGE = "MESSAGE"            # Individual message
    USER = "USER"                  # Participant
    TOPIC = "TOPIC"                # Discussion topic
    ENTITY = "ENTITY"              # Mentioned entity (person, place, thing)
    ACTION = "ACTION"              # Action taken (decision, task)


class RelationType:
    """Relationship types in conversation knowledge graph."""
    # Message relationships
    SENT_BY = "SENT_BY"           # Message -> User
    PART_OF = "PART_OF"           # Message -> Conversation
    FOLLOWS = "FOLLOWS"           # Message -> Previous Message
    REPLIES_TO = "REPLIES_TO"     # Message -> Parent Message (threads)

    # Content relationships
    MENTIONS = "MENTIONS"         # Message -> Entity/User
    DISCUSSES = "DISCUSSES"       # Message -> Topic
    REFERENCES = "REFERENCES"     # Message -> Previous Message (quote)

    # User relationships
    PARTICIPATES_IN = "PARTICIPATES_IN"  # User -> Conversation
    INTERACTS_WITH = "INTERACTS_WITH"    # User -> User

    # Topic relationships
    RELATED_TO = "RELATED_TO"     # Topic -> Topic
    SUBTOPIC_OF = "SUBTOPIC_OF"   # Topic -> Parent Topic


# ============================================================================
# Conversation Memory Manager
# ============================================================================

@dataclass
class ConversationStats:
    """Statistics for a conversation."""
    total_messages: int = 0
    unique_participants: int = 0
    topics_discussed: int = 0
    date_range: tuple = field(default_factory=lambda: (None, None))
    most_active_user: Optional[str] = None
    message_frequency: float = 0.0  # Messages per hour


class ConversationMemory:
    """
    Manages conversation storage in HoloLoom knowledge graph.

    Features:
    - Stores messages as KG nodes with relationships
    - Tracks conversation structure (threads, replies)
    - Indexes by time, user, topic
    - Enables semantic search and retrieval
    - Computes spectral features for conversation analysis

    Usage:
        memory = ConversationMemory()

        # Store message
        msg_id = memory.add_message(
            conversation_id="room_123",
            sender="@alice:matrix.org",
            text="Let's discuss the architecture",
            timestamp=datetime.now()
        )

        # Retrieve conversation context
        context = memory.get_conversation_context(
            conversation_id="room_123",
            limit=10
        )

        # Search across conversations
        results = memory.search("architecture", limit=5)
    """

    def __init__(self, kg: Optional[Any] = None):
        """
        Initialize conversation memory.

        Args:
            kg: Optional KG instance (creates new if None)
        """
        if not HOLOLOOM_AVAILABLE:
            logger.warning("HoloLoom not available - running in mock mode")
            self.kg = None
        else:
            self.kg = kg or KG()

        # Message counters per conversation
        self.message_counters: Dict[str, int] = {}

        # Entity extraction cache
        self.entity_cache: Dict[str, Set[str]] = {}

        logger.info("ConversationMemory initialized")

    # ========================================================================
    # Message Storage
    # ========================================================================

    def add_message(
        self,
        conversation_id: str,
        sender: str,
        text: str,
        timestamp: Optional[datetime] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add message to conversation memory.

        Args:
            conversation_id: Conversation/room ID
            sender: User ID who sent message
            text: Message text
            timestamp: When message was sent
            reply_to: Optional ID of message being replied to
            metadata: Additional metadata

        Returns:
            Message node ID
        """
        if not self.kg:
            return self._mock_add_message(conversation_id, sender, text)

        timestamp = timestamp or datetime.now()

        # Generate message ID
        msg_counter = self.message_counters.get(conversation_id, 0)
        msg_id = self._generate_message_id(conversation_id, msg_counter)
        self.message_counters[conversation_id] = msg_counter + 1

        # Create message node
        self.kg.add_entity(
            entity_id=msg_id,
            entity_type=EntityType.MESSAGE,
            properties={
                "text": text,
                "timestamp": timestamp.isoformat(),
                "conversation_id": conversation_id,
                "sender": sender,
                "metadata": metadata or {}
            }
        )

        # Ensure conversation exists
        self._ensure_conversation(conversation_id)

        # Ensure user exists
        self._ensure_user(sender)

        # Add relationships
        self._add_message_relationships(
            msg_id, conversation_id, sender, timestamp, reply_to
        )

        # Extract and link entities
        entities = self._extract_entities(text)
        self._link_entities(msg_id, entities)

        # Extract and link topics
        topics = self._extract_topics(text)
        self._link_topics(msg_id, topics)

        logger.debug(f"Added message: {msg_id}")
        return msg_id

    def _generate_message_id(self, conversation_id: str, counter: int) -> str:
        """Generate unique message ID."""
        # Clean conversation ID
        clean_id = conversation_id.replace("!", "").replace(":", "_")
        return f"msg_{clean_id}_{counter}"

    def _add_message_relationships(
        self,
        msg_id: str,
        conversation_id: str,
        sender: str,
        timestamp: datetime,
        reply_to: Optional[str]
    ) -> None:
        """Add relationships for message."""
        # Message -> Conversation
        self.kg.add_relation(
            from_id=msg_id,
            to_id=f"conv_{conversation_id}",
            rel_type=RelationType.PART_OF
        )

        # Message -> User
        user_id = self._user_node_id(sender)
        self.kg.add_relation(
            from_id=msg_id,
            to_id=user_id,
            rel_type=RelationType.SENT_BY
        )

        # Message -> Previous Message (chronological chain)
        prev_msg = self._get_previous_message(conversation_id, timestamp)
        if prev_msg:
            self.kg.add_relation(
                from_id=msg_id,
                to_id=prev_msg,
                rel_type=RelationType.FOLLOWS
            )

        # Message -> Replied Message (thread structure)
        if reply_to:
            self.kg.add_relation(
                from_id=msg_id,
                to_id=reply_to,
                rel_type=RelationType.REPLIES_TO
            )

    def _get_previous_message(
        self,
        conversation_id: str,
        timestamp: datetime
    ) -> Optional[str]:
        """Get previous message in conversation."""
        # Would query KG for most recent message before timestamp
        # For now, simplified
        counter = self.message_counters.get(conversation_id, 0)
        if counter > 0:
            return self._generate_message_id(conversation_id, counter - 1)
        return None

    # ========================================================================
    # Entity & Topic Extraction
    # ========================================================================

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from message text.

        Args:
            text: Message text

        Returns:
            List of entity strings
        """
        # Simple extraction - in production would use NLP
        entities = []

        # Extract @mentions
        import re
        mentions = re.findall(r'@[\w.-]+', text)
        entities.extend(mentions)

        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', text)
        entities.extend(urls)

        # Cache for later use
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.entity_cache[text_hash] = set(entities)

        return entities

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract topics from message text.

        Args:
            text: Message text

        Returns:
            List of topic strings
        """
        # Simple keyword extraction - in production would use TF-IDF or LLM
        keywords = ["architecture", "chatops", "integration", "matrix", "hololoom"]
        text_lower = text.lower()

        topics = [kw for kw in keywords if kw in text_lower]
        return topics

    def _link_entities(self, msg_id: str, entities: List[str]) -> None:
        """Link message to mentioned entities."""
        for entity in entities:
            entity_id = self._entity_node_id(entity)

            # Create entity node if doesn't exist
            if not self.kg.has_entity(entity_id):
                self.kg.add_entity(
                    entity_id=entity_id,
                    entity_type=EntityType.ENTITY,
                    properties={"name": entity}
                )

            # Link message to entity
            self.kg.add_relation(
                from_id=msg_id,
                to_id=entity_id,
                rel_type=RelationType.MENTIONS
            )

    def _link_topics(self, msg_id: str, topics: List[str]) -> None:
        """Link message to discussion topics."""
        for topic in topics:
            topic_id = self._topic_node_id(topic)

            # Create topic node if doesn't exist
            if not self.kg.has_entity(topic_id):
                self.kg.add_entity(
                    entity_id=topic_id,
                    entity_type=EntityType.TOPIC,
                    properties={"name": topic}
                )

            # Link message to topic
            self.kg.add_relation(
                from_id=msg_id,
                to_id=topic_id,
                rel_type=RelationType.DISCUSSES
            )

    # ========================================================================
    # Conversation & User Management
    # ========================================================================

    def _ensure_conversation(self, conversation_id: str) -> None:
        """Ensure conversation node exists."""
        conv_id = f"conv_{conversation_id}"

        if self.kg.has_entity(conv_id):
            return

        self.kg.add_entity(
            entity_id=conv_id,
            entity_type=EntityType.CONVERSATION,
            properties={
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "platform": "matrix"
            }
        )

        logger.debug(f"Created conversation: {conv_id}")

    def _ensure_user(self, user_id: str) -> None:
        """Ensure user node exists."""
        node_id = self._user_node_id(user_id)

        if self.kg.has_entity(node_id):
            return

        self.kg.add_entity(
            entity_id=node_id,
            entity_type=EntityType.USER,
            properties={
                "user_id": user_id,
                "platform": "matrix",
                "first_seen": datetime.now().isoformat()
            }
        )

        logger.debug(f"Created user: {node_id}")

    # ========================================================================
    # Retrieval
    # ========================================================================

    def get_conversation_context(
        self,
        conversation_id: str,
        limit: int = 10,
        before: Optional[datetime] = None
    ) -> List[MemoryShard]:
        """
        Get recent messages from conversation.

        Args:
            conversation_id: Conversation ID
            limit: Max messages to retrieve
            before: Only messages before this time

        Returns:
            List of MemoryShard objects
        """
        if not self.kg:
            return self._mock_get_context(conversation_id, limit)

        # Query KG for messages in conversation
        conv_node_id = f"conv_{conversation_id}"

        # Get all messages in conversation
        # In production, would use proper graph query
        messages = self._query_messages(conv_node_id, limit, before)

        # Convert to MemoryShards
        shards = []
        for msg in messages:
            shard = MemoryShard(
                id=msg["id"],
                text=msg["text"],
                episode=conversation_id,
                entities=msg.get("entities", []),
                motifs=msg.get("topics", []),
                metadata=msg.get("metadata", {})
            )
            shards.append(shard)

        return shards

    def _query_messages(
        self,
        conv_node_id: str,
        limit: int,
        before: Optional[datetime]
    ) -> List[Dict]:
        """Query messages from KG."""
        # Simplified - in production would use proper graph traversal
        return []

    def search(
        self,
        query: str,
        limit: int = 5,
        conversation_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search messages by text.

        Args:
            query: Search query
            limit: Max results
            conversation_id: Optional conversation to search within

        Returns:
            List of message dicts
        """
        # Would use semantic search with embeddings
        return []

    def get_statistics(
        self,
        conversation_id: str
    ) -> ConversationStats:
        """
        Get statistics for conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            ConversationStats
        """
        if not self.kg:
            return ConversationStats()

        # Query KG for stats
        # Simplified implementation
        return ConversationStats(
            total_messages=self.message_counters.get(conversation_id, 0),
            unique_participants=0,
            topics_discussed=0
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def _user_node_id(self, user_id: str) -> str:
        """Generate user node ID."""
        clean_id = user_id.replace("@", "").replace(":", "_")
        return f"user_{clean_id}"

    def _entity_node_id(self, entity: str) -> str:
        """Generate entity node ID."""
        clean = entity.replace("@", "").replace(":", "_").replace("/", "_")
        return f"entity_{clean}"

    def _topic_node_id(self, topic: str) -> str:
        """Generate topic node ID."""
        clean = topic.replace(" ", "_").lower()
        return f"topic_{clean}"

    # ========================================================================
    # Mock Methods (when HoloLoom unavailable)
    # ========================================================================

    def _mock_add_message(
        self,
        conversation_id: str,
        sender: str,
        text: str
    ) -> str:
        """Mock message storage."""
        counter = self.message_counters.get(conversation_id, 0)
        msg_id = f"mock_msg_{counter}"
        self.message_counters[conversation_id] = counter + 1
        return msg_id

    def _mock_get_context(
        self,
        conversation_id: str,
        limit: int
    ) -> List[MemoryShard]:
        """Mock context retrieval."""
        return []


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Conversation Memory Demo")
    print("="*80)
    print()

    if not HOLOLOOM_AVAILABLE:
        print("Warning: Running in mock mode (HoloLoom not available)")
        print()

    # Create memory manager
    memory = ConversationMemory()

    # Add messages to a conversation
    conv_id = "!abc123:matrix.org"

    print(f"Adding messages to conversation: {conv_id}\n")

    msg1 = memory.add_message(
        conversation_id=conv_id,
        sender="@alice:matrix.org",
        text="Let's discuss the chatops architecture",
        timestamp=datetime.now()
    )
    print(f"Message 1: {msg1}")

    msg2 = memory.add_message(
        conversation_id=conv_id,
        sender="@bob:matrix.org",
        text="Great idea! We should integrate Matrix with HoloLoom",
        timestamp=datetime.now() + timedelta(seconds=30)
    )
    print(f"Message 2: {msg2}")

    msg3 = memory.add_message(
        conversation_id=conv_id,
        sender="@alice:matrix.org",
        text="Agreed. And store conversations in the knowledge graph",
        timestamp=datetime.now() + timedelta(seconds=60),
        reply_to=msg2
    )
    print(f"Message 3: {msg3}")

    print()

    # Get conversation context
    print("Retrieving conversation context...\n")
    context = memory.get_conversation_context(conv_id, limit=10)
    print(f"Retrieved {len(context)} messages")

    print()

    # Get statistics
    print("Conversation statistics:\n")
    stats = memory.get_statistics(conv_id)
    print(f"  Total messages: {stats.total_messages}")
    print(f"  Unique participants: {stats.unique_participants}")
    print(f"  Topics discussed: {stats.topics_discussed}")

    print("\n✓ Demo complete!")
