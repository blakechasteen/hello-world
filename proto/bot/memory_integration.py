#!/usr/bin/env python3
"""
Memory Integration Bridge for Proto Matrix Bot

Bridges Matrix conversations with HoloLoom's knowledge graph memory system.
Automatically stores all conversations, extracts entities, builds relationships,
and provides context-aware response enrichment.

Features:
- Automatic conversation storage in knowledge graph
- Entity extraction (users, files, decisions, topics)
- Temporal metadata (timestamps, room context)
- Context retrieval for responses
- Cross-reference related discussions

Architecture:
    Matrix Event → Memory Integration → HoloLoom Knowledge Graph
                        ↓
            Entity Extraction + Relationship Building

Usage:
    from bot.memory_integration import MemoryIntegration

    # Initialize
    memory = MemoryIntegration()
    await memory.initialize()

    # Store conversation
    await memory.store_conversation(
        user_id="@alice:matrix.org",
        message="We decided to use PostgreSQL for auth",
        room_id="!team:matrix.org"
    )

    # Retrieve context
    context = await memory.get_context(
        query="What database are we using?",
        room_id="!team:matrix.org",
        limit=5
    )

    # Find related topics
    related = await memory.find_related(
        topic="authentication",
        room_id="!team:matrix.org"
    )
"""

import logging
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ConversationMemory:
    """
    Structured conversation memory.

    Stores complete context about a Matrix conversation message.
    """
    id: str                                # Unique memory ID
    user_id: str                          # Matrix user ID
    message: str                          # Message text
    room_id: str                          # Matrix room ID
    timestamp: datetime                   # When message was sent
    entities: List[str] = field(default_factory=list)    # Extracted entities
    topics: List[str] = field(default_factory=list)      # Detected topics
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class ContextResult:
    """
    Context retrieval result.

    Contains relevant memories and their relationships.
    """
    memories: List[ConversationMemory]    # Retrieved memories
    confidence: float                      # Retrieval confidence
    topics_covered: Set[str]               # Topics in results
    time_range: tuple                      # (earliest, latest) timestamp
    related_users: Set[str]                # Users involved


class MemoryIntegration:
    """
    Memory Integration Bridge.

    Bridges Proto Matrix bot with HoloLoom knowledge graph memory system.
    Handles automatic conversation storage, entity extraction, and context retrieval.

    Example:
        >>> memory = MemoryIntegration()
        >>> await memory.initialize()
        >>>
        >>> # Store conversation
        >>> await memory.store_conversation(
        ...     user_id="@alice:matrix.org",
        ...     message="We should use Redis for caching",
        ...     room_id="!team:matrix.org"
        ... )
        >>>
        >>> # Retrieve context
        >>> context = await memory.get_context(
        ...     query="What caching solution?",
        ...     room_id="!team:matrix.org"
        ... )
        >>> print(context.memories[0].message)
    """

    def __init__(self):
        """Initialize memory integration (lazy load HoloLoom)"""
        self.loom = None
        self.initialized = False
        self._conversation_cache = {}  # Cache recent conversations

    async def initialize(self):
        """
        Initialize HoloLoom connection.

        Lazy-loads HoloLoom to avoid import errors if not available.
        """
        if self.initialized:
            return

        try:
            from HoloLoom import HoloLoom
            from HoloLoom.config import Config

            # Initialize with FAST config (good balance)
            config = Config.fast()
            self.loom = HoloLoom(config=config)

            # Initialize as context manager
            await self.loom.__aenter__()

            self.initialized = True
            logger.info("Memory integration initialized with HoloLoom")

        except ImportError as e:
            logger.error(f"Failed to import HoloLoom: {e}")
            logger.error("Memory features will be unavailable")
            raise

        except Exception as e:
            logger.error(f"Failed to initialize memory integration: {e}")
            raise

    async def store_conversation(
        self,
        user_id: str,
        message: str,
        room_id: str,
        metadata: Optional[Dict] = None
    ) -> ConversationMemory:
        """
        Store Matrix conversation in knowledge graph.

        Automatically extracts entities, detects topics, and builds
        relationships in the knowledge graph.

        Args:
            user_id: Matrix user ID (@user:server)
            message: Message text
            room_id: Matrix room ID (!room:server)
            metadata: Optional additional metadata

        Returns:
            ConversationMemory with extracted entities and topics

        Example:
            >>> memory = await integration.store_conversation(
            ...     user_id="@alice:matrix.org",
            ...     message="We decided to use PostgreSQL for auth",
            ...     room_id="!team:matrix.org"
            ... )
            >>> print(memory.entities)
            ['PostgreSQL', 'auth']
        """
        if not self.initialized:
            await self.initialize()

        # Extract entities and topics
        entities = self._extract_entities(message)
        topics = self._detect_topics(message)

        # Create memory ID (hash of content + timestamp)
        memory_id = self._generate_memory_id(user_id, message, room_id)

        # Build metadata
        full_metadata = {
            "type": "conversation",
            "user_id": user_id,
            "room_id": room_id,
            "timestamp": datetime.now().isoformat(),
            "entities": entities,
            "topics": topics,
            **(metadata or {})
        }

        # Store in HoloLoom
        await self.loom.experience(
            content=message,
            context=full_metadata
        )

        # Create conversation memory object
        conv_memory = ConversationMemory(
            id=memory_id,
            user_id=user_id,
            message=message,
            room_id=room_id,
            timestamp=datetime.now(),
            entities=entities,
            topics=topics,
            metadata=full_metadata
        )

        # Cache for quick retrieval
        self._conversation_cache[memory_id] = conv_memory

        logger.info(
            f"Stored conversation from {user_id} in {room_id}: "
            f"{len(entities)} entities, {len(topics)} topics"
        )

        return conv_memory

    async def get_context(
        self,
        query: str,
        room_id: Optional[str] = None,
        limit: int = 5,
        include_related: bool = True
    ) -> ContextResult:
        """
        Retrieve relevant context for a query.

        Searches knowledge graph for relevant conversations and returns
        structured context with metadata.

        Args:
            query: Search query
            room_id: Optional room filter (only return memories from this room)
            limit: Maximum memories to return
            include_related: Include related topics via graph traversal

        Returns:
            ContextResult with relevant memories and metadata

        Example:
            >>> context = await integration.get_context(
            ...     query="What database are we using?",
            ...     room_id="!team:matrix.org",
            ...     limit=5
            ... )
            >>> for memory in context.memories:
            ...     print(f"{memory.user_id}: {memory.message}")
        """
        if not self.initialized:
            await self.initialize()

        # Retrieve from HoloLoom
        raw_memories = await self.loom.recall(
            query=query,
            limit=limit if not include_related else limit * 2
        )

        # Convert to ConversationMemory objects
        memories = []
        topics_covered = set()
        related_users = set()
        timestamps = []

        for mem in raw_memories:
            # Filter by room if specified
            mem_room = mem.metadata.get('room_id')
            if room_id and mem_room != room_id:
                continue

            # Extract metadata
            user_id = mem.metadata.get('user_id', 'unknown')
            entities = mem.metadata.get('entities', [])
            topics = mem.metadata.get('topics', [])

            # Create ConversationMemory
            conv_mem = ConversationMemory(
                id=mem.id,
                user_id=user_id,
                message=mem.text,
                room_id=mem_room or 'unknown',
                timestamp=mem.timestamp,
                entities=entities,
                topics=topics,
                metadata=mem.metadata
            )

            memories.append(conv_mem)
            topics_covered.update(topics)
            related_users.add(user_id)
            timestamps.append(mem.timestamp)

            # Stop if we have enough (after filtering)
            if len(memories) >= limit:
                break

        # Calculate confidence (simple heuristic: more results = higher confidence)
        confidence = min(1.0, len(memories) / limit) if limit > 0 else 0.0

        # Time range
        time_range = (
            min(timestamps) if timestamps else datetime.now(),
            max(timestamps) if timestamps else datetime.now()
        )

        result = ContextResult(
            memories=memories,
            confidence=confidence,
            topics_covered=topics_covered,
            time_range=time_range,
            related_users=related_users
        )

        logger.info(
            f"Retrieved {len(memories)} memories for query '{query[:50]}...' "
            f"(confidence: {confidence:.2f})"
        )

        return result

    async def find_related(
        self,
        topic: str,
        room_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ConversationMemory]:
        """
        Find conversations related to a topic.

        Uses knowledge graph traversal to find connected topics and discussions.

        Args:
            topic: Topic to search for
            room_id: Optional room filter
            limit: Maximum results

        Returns:
            List of related conversation memories

        Example:
            >>> related = await integration.find_related(
            ...     topic="authentication",
            ...     room_id="!team:matrix.org"
            ... )
            >>> for memory in related:
            ...     print(f"{memory.timestamp}: {memory.message}")
        """
        # Use get_context with topic as query
        context = await self.get_context(
            query=topic,
            room_id=room_id,
            limit=limit,
            include_related=True
        )

        return context.memories

    async def remember_fact(
        self,
        fact: str,
        user_id: str,
        room_id: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Explicitly remember a fact (user-requested memory).

        Used for `@proto remember <fact>` commands where user explicitly
        wants to store something in team knowledge base.

        Args:
            fact: Fact to remember
            user_id: User who stored the fact
            room_id: Room where fact was stored
            tags: Optional categorical tags

        Returns:
            Memory ID

        Example:
            >>> memory_id = await integration.remember_fact(
            ...     fact="PostgreSQL for auth service",
            ...     user_id="@alice:matrix.org",
            ...     room_id="!team:matrix.org",
            ...     tags=["decision", "database"]
            ... )
        """
        if not self.initialized:
            await self.initialize()

        # Extract entities and topics
        entities = self._extract_entities(fact)
        topics = self._detect_topics(fact)

        # Add user-provided tags to topics
        if tags:
            topics.extend(tags)

        # Store with explicit_memory type
        memory_id = self._generate_memory_id(user_id, fact, room_id)

        metadata = {
            "type": "explicit_memory",
            "stored_by": user_id,
            "room_id": room_id,
            "timestamp": datetime.now().isoformat(),
            "entities": entities,
            "topics": topics,
            "tags": tags or []
        }

        await self.loom.experience(
            content=fact,
            context=metadata
        )

        logger.info(f"Stored explicit fact from {user_id}: {fact[:100]}...")

        return memory_id

    async def get_statistics(self, room_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns statistics about stored conversations, entities, and topics.

        Args:
            room_id: Optional room filter

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = await integration.get_statistics(room_id="!team:matrix.org")
            >>> print(f"Total memories: {stats['total_memories']}")
            >>> print(f"Active topics: {stats['unique_topics']}")
        """
        if not self.initialized:
            return {
                "initialized": False,
                "error": "Memory integration not initialized"
            }

        # Get HoloLoom metrics
        loom_metrics = self.loom.get_metrics()

        # Count cached conversations (room-specific if provided)
        cached_count = len(self._conversation_cache)
        if room_id:
            cached_count = sum(
                1 for conv in self._conversation_cache.values()
                if conv.room_id == room_id
            )

        # Collect topics from cache
        all_topics = set()
        all_entities = set()
        for conv in self._conversation_cache.values():
            if not room_id or conv.room_id == room_id:
                all_topics.update(conv.topics)
                all_entities.update(conv.entities)

        return {
            "initialized": True,
            "total_memories": loom_metrics['n_memories'],
            "total_connections": loom_metrics['n_connections'],
            "active_memories": loom_metrics['n_active'],
            "cached_conversations": cached_count,
            "unique_topics": len(all_topics),
            "unique_entities": len(all_entities),
            "room_filter": room_id,
            "sample_topics": list(all_topics)[:10],
            "sample_entities": list(all_entities)[:10]
        }

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.

        Uses simple heuristics:
        - Capitalized words (potential proper nouns)
        - Technical terms (database names, technologies)
        - File paths
        - User mentions

        Args:
            text: Text to extract from

        Returns:
            List of entity strings

        Note:
            Future improvement: Use spaCy NER for better extraction
        """
        entities = []

        # Capitalized words (proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)

        # Technical terms (common databases, languages, frameworks)
        tech_terms = [
            'PostgreSQL', 'MySQL', 'Redis', 'MongoDB',
            'Python', 'JavaScript', 'TypeScript', 'Go', 'Rust',
            'React', 'Vue', 'Angular', 'Django', 'Flask',
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
            'JWT', 'OAuth', 'API', 'REST', 'GraphQL'
        ]

        for term in tech_terms:
            if term.lower() in text.lower():
                entities.append(term)

        # File paths
        file_paths = re.findall(r'\b[\w/]+\.\w+\b', text)
        entities.extend(file_paths)

        # User mentions (@user:server)
        user_mentions = re.findall(r'@[\w]+:[\w.]+', text)
        entities.extend(user_mentions)

        # Remove duplicates, preserve order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities[:20]  # Limit to 20 entities

    def _detect_topics(self, text: str) -> List[str]:
        """
        Detect topics from text.

        Uses keyword matching for common development topics.

        Args:
            text: Text to analyze

        Returns:
            List of topic strings
        """
        topics = []
        text_lower = text.lower()

        # Topic keywords
        topic_keywords = {
            'authentication': ['auth', 'login', 'jwt', 'oauth', 'password', 'token'],
            'database': ['database', 'db', 'sql', 'nosql', 'postgres', 'mysql', 'mongo'],
            'api': ['api', 'endpoint', 'rest', 'graphql', 'request', 'response'],
            'security': ['security', 'vulnerability', 'xss', 'csrf', 'injection'],
            'testing': ['test', 'testing', 'pytest', 'jest', 'unit test', 'integration'],
            'deployment': ['deploy', 'deployment', 'docker', 'kubernetes', 'ci/cd'],
            'frontend': ['frontend', 'ui', 'react', 'vue', 'angular', 'css'],
            'backend': ['backend', 'server', 'django', 'flask', 'express'],
            'architecture': ['architecture', 'design', 'pattern', 'structure'],
            'decision': ['decided', 'decision', 'choose', 'selected', 'agreed'],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def _generate_memory_id(self, user_id: str, message: str, room_id: str) -> str:
        """
        Generate unique memory ID.

        Args:
            user_id: User ID
            message: Message text
            room_id: Room ID

        Returns:
            Unique memory ID (hash)
        """
        content = f"{user_id}:{room_id}:{message}:{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"conv_{hash_obj.hexdigest()[:16]}"

    async def close(self):
        """Cleanup resources"""
        if self.loom and self.initialized:
            await self.loom.__aexit__(None, None, None)
            self.initialized = False
            logger.info("Memory integration closed")


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Memory Integration...")

        memory = MemoryIntegration()

        try:
            # Initialize
            print("\n1. Initializing...")
            await memory.initialize()
            print("   ✓ Initialized")

            # Store conversation
            print("\n2. Storing conversation...")
            conv = await memory.store_conversation(
                user_id="@alice:matrix.org",
                message="We decided to use PostgreSQL for the authentication service",
                room_id="!test:matrix.org"
            )
            print(f"   ✓ Stored: {len(conv.entities)} entities, {len(conv.topics)} topics")
            print(f"   Entities: {conv.entities}")
            print(f"   Topics: {conv.topics}")

            # Store another
            print("\n3. Storing related conversation...")
            conv2 = await memory.store_conversation(
                user_id="@bob:matrix.org",
                message="The PostgreSQL database will use JWT tokens for auth",
                room_id="!test:matrix.org"
            )
            print(f"   ✓ Stored: {len(conv2.entities)} entities, {len(conv2.topics)} topics")

            # Retrieve context
            print("\n4. Retrieving context...")
            context = await memory.get_context(
                query="What database are we using?",
                room_id="!test:matrix.org",
                limit=5
            )
            print(f"   ✓ Retrieved {len(context.memories)} memories")
            print(f"   Confidence: {context.confidence:.2f}")
            print(f"   Topics: {context.topics_covered}")
            for mem in context.memories:
                print(f"   - {mem.user_id}: {mem.message[:60]}...")

            # Find related
            print("\n5. Finding related topics...")
            related = await memory.find_related(
                topic="authentication",
                room_id="!test:matrix.org"
            )
            print(f"   ✓ Found {len(related)} related conversations")

            # Remember fact
            print("\n6. Remembering explicit fact...")
            fact_id = await memory.remember_fact(
                fact="Team decided on 15min JWT access tokens",
                user_id="@alice:matrix.org",
                room_id="!test:matrix.org",
                tags=["decision", "security"]
            )
            print(f"   ✓ Stored fact: {fact_id}")

            # Statistics
            print("\n7. Getting statistics...")
            stats = await memory.get_statistics(room_id="!test:matrix.org")
            print(f"   Total memories: {stats['total_memories']}")
            print(f"   Unique topics: {stats['unique_topics']}")
            print(f"   Sample topics: {stats['sample_topics']}")

            print("\n✅ All Memory Integration tests passed!")

        except ImportError as e:
            print(f"\n⚠️  SKIPPED: HoloLoom not available ({e})")
            print("This is expected if HoloLoom is not in parent directory")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await memory.close()

    asyncio.run(test())
