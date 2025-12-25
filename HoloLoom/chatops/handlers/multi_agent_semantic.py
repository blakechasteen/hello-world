#!/usr/bin/env python3
"""
Multi-Agent Semantic Coordination
==================================
Phase 7: Semantic state sharing across agents

Enables:
- SharedSemanticRegistry: Global registry for semantic states
- Cross-room topic threading: Link related topics across rooms
- Agent handoff with context preservation: Transfer semantic context

Commands:
- !agents list - Show active agents and their semantic state
- !agents topics - Show cross-room topic threads
- !agents handoff <agent_id> - Hand off current context to another agent
- !agents link <room_id> - Link topic thread to another room
- !agents stats - Show multi-agent coordination stats

Architecture:
```
                    SharedSemanticRegistry
                           ↓
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    v                      v                      v
  Agent A              Agent B              Agent C
  Room 1               Room 2               Room 3
  Context α            Context β            Context γ
                           ↓
                    TopicThreading
                (links related contexts)
```

Created: 2025-12-25
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

# Import semantic components
try:
    from HoloLoom.semantic_calculus import (
        PolicySemanticState as SemanticState,
        SEMANTIC_CATEGORIES,
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticState = None
    SEMANTIC_CATEGORIES = {}

# Handler registry
from HoloLoom.chatops.handlers.handler_registry import (
    HandlerRegistry, HandlerCategory, chatops_handler
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class AgentStatus(Enum):
    """Agent operational status."""
    ACTIVE = "active"
    IDLE = "idle"
    HANDOFF_PENDING = "handoff_pending"
    OFFLINE = "offline"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    room_id: str
    status: AgentStatus = AgentStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    query_count: int = 0
    current_topic: Optional[str] = None
    semantic_summary: Optional[Dict[str, float]] = None


@dataclass
class TopicThread:
    """A thread linking related topics across rooms."""
    thread_id: str
    topic_name: str
    room_ids: Set[str] = field(default_factory=set)
    agent_ids: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    semantic_centroid: Optional[np.ndarray] = None
    message_count: int = 0


@dataclass
class HandoffContext:
    """Context passed during agent handoff."""
    from_agent: str
    to_agent: str
    room_id: str
    timestamp: datetime
    semantic_state: Optional[Any] = None
    queries: List[str] = field(default_factory=list)
    topic_shifts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SharedSemanticRegistry - Global Semantic State Coordination
# ============================================================================

class SharedSemanticRegistry:
    """
    Global registry for sharing semantic states across agents.

    Provides:
    - Agent registration and tracking
    - Cross-room topic threading
    - Semantic state caching
    - Handoff coordination

    Thread-safe via asyncio locks.
    """

    _instance: Optional['SharedSemanticRegistry'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (once)."""
        if self._initialized:
            return

        # Agent registry
        self._agents: Dict[str, AgentInfo] = {}

        # Topic threads (cross-room linking)
        self._topics: Dict[str, TopicThread] = {}

        # Room -> Agent mapping
        self._room_agents: Dict[str, str] = {}

        # Semantic state cache (agent_id -> states)
        self._state_cache: Dict[str, List[Any]] = defaultdict(list)

        # Topic similarity threshold
        self._topic_similarity_threshold = 0.7

        # Maximum cache size per agent
        self._max_cache_size = 100

        # Handoff history
        self._handoff_history: List[HandoffContext] = []

        self._initialized = True
        logger.info("SharedSemanticRegistry initialized")

    async def register_agent(
        self,
        agent_id: str,
        room_id: str,
        **metadata
    ) -> AgentInfo:
        """
        Register a new agent with the registry.

        Args:
            agent_id: Unique agent identifier
            room_id: Room the agent operates in
            **metadata: Additional agent metadata

        Returns:
            AgentInfo for the registered agent
        """
        async with self._lock:
            agent = AgentInfo(
                agent_id=agent_id,
                room_id=room_id,
                status=AgentStatus.ACTIVE,
            )
            self._agents[agent_id] = agent
            self._room_agents[room_id] = agent_id

            logger.info(f"Agent registered: {agent_id} in room {room_id}")
            return agent

    async def update_agent_state(
        self,
        agent_id: str,
        semantic_state: Optional[Any] = None,
        query: Optional[str] = None,
        topic: Optional[str] = None
    ):
        """
        Update an agent's semantic state.

        Args:
            agent_id: Agent to update
            semantic_state: Current semantic state
            query: Query that was processed
            topic: Detected topic
        """
        async with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Unknown agent: {agent_id}")
                return

            agent = self._agents[agent_id]
            agent.last_active = datetime.now()
            agent.query_count += 1

            if topic:
                agent.current_topic = topic

            if semantic_state:
                # Cache state
                cache = self._state_cache[agent_id]
                cache.append({
                    'state': semantic_state,
                    'query': query,
                    'topic': topic,
                    'timestamp': datetime.now()
                })

                # Trim cache if needed
                if len(cache) > self._max_cache_size:
                    self._state_cache[agent_id] = cache[-self._max_cache_size:]

                # Update semantic summary
                if hasattr(semantic_state, 'get_category_scores'):
                    agent.semantic_summary = semantic_state.get_category_scores()

    async def find_related_topics(
        self,
        semantic_state: Any,
        exclude_room: Optional[str] = None
    ) -> List[Tuple[TopicThread, float]]:
        """
        Find topic threads related to the given semantic state.

        Args:
            semantic_state: Current semantic state
            exclude_room: Room to exclude from results

        Returns:
            List of (TopicThread, similarity) tuples, sorted by similarity
        """
        if not hasattr(semantic_state, 'get_embedding'):
            return []

        embedding = semantic_state.get_embedding()
        related = []

        async with self._lock:
            for thread in self._topics.values():
                if exclude_room and exclude_room in thread.room_ids:
                    continue

                if thread.semantic_centroid is not None:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(
                        embedding, thread.semantic_centroid
                    )

                    if similarity >= self._topic_similarity_threshold:
                        related.append((thread, similarity))

        # Sort by similarity descending
        related.sort(key=lambda x: x[1], reverse=True)
        return related

    async def create_topic_thread(
        self,
        topic_name: str,
        room_id: str,
        agent_id: str,
        semantic_state: Optional[Any] = None
    ) -> TopicThread:
        """
        Create a new topic thread.

        Args:
            topic_name: Name for the topic
            room_id: Initial room
            agent_id: Initial agent
            semantic_state: Initial semantic state

        Returns:
            Created TopicThread
        """
        async with self._lock:
            thread_id = str(uuid.uuid4())[:8]

            centroid = None
            if semantic_state and hasattr(semantic_state, 'get_embedding'):
                centroid = semantic_state.get_embedding()

            thread = TopicThread(
                thread_id=thread_id,
                topic_name=topic_name,
                room_ids={room_id},
                agent_ids={agent_id},
                semantic_centroid=centroid
            )

            self._topics[thread_id] = thread
            logger.info(f"Topic thread created: {topic_name} ({thread_id})")

            return thread

    async def link_room_to_topic(
        self,
        thread_id: str,
        room_id: str,
        agent_id: str
    ) -> bool:
        """
        Link a room to an existing topic thread.

        Args:
            thread_id: Topic thread to link to
            room_id: Room to link
            agent_id: Agent in the room

        Returns:
            True if linked successfully
        """
        async with self._lock:
            if thread_id not in self._topics:
                return False

            thread = self._topics[thread_id]
            thread.room_ids.add(room_id)
            thread.agent_ids.add(agent_id)
            thread.last_activity = datetime.now()

            logger.info(f"Room {room_id} linked to topic {thread_id}")
            return True

    async def initiate_handoff(
        self,
        from_agent: str,
        to_agent: str,
        include_history: int = 10
    ) -> Optional[HandoffContext]:
        """
        Initiate a context handoff between agents.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            include_history: Number of recent queries to include

        Returns:
            HandoffContext if successful
        """
        async with self._lock:
            if from_agent not in self._agents:
                logger.error(f"Source agent not found: {from_agent}")
                return None

            source = self._agents[from_agent]
            source.status = AgentStatus.HANDOFF_PENDING

            # Gather context
            cache = self._state_cache.get(from_agent, [])
            recent = cache[-include_history:] if cache else []

            queries = [entry['query'] for entry in recent if entry.get('query')]
            states = [entry['state'] for entry in recent if entry.get('state')]

            context = HandoffContext(
                from_agent=from_agent,
                to_agent=to_agent,
                room_id=source.room_id,
                timestamp=datetime.now(),
                semantic_state=states[-1] if states else None,
                queries=queries,
                metadata={
                    'topic': source.current_topic,
                    'query_count': source.query_count,
                    'semantic_summary': source.semantic_summary
                }
            )

            self._handoff_history.append(context)

            # Mark target agent as pending
            if to_agent in self._agents:
                self._agents[to_agent].status = AgentStatus.HANDOFF_PENDING

            logger.info(f"Handoff initiated: {from_agent} -> {to_agent}")
            return context

    async def complete_handoff(
        self,
        context: HandoffContext
    ) -> bool:
        """
        Complete a handoff, transferring context to target agent.

        Args:
            context: HandoffContext from initiate_handoff

        Returns:
            True if completed successfully
        """
        async with self._lock:
            from_agent = context.from_agent
            to_agent = context.to_agent

            if from_agent in self._agents:
                self._agents[from_agent].status = AgentStatus.IDLE

            if to_agent in self._agents:
                target = self._agents[to_agent]
                target.status = AgentStatus.ACTIVE
                target.current_topic = context.metadata.get('topic')

                # Transfer context to target's cache
                if context.semantic_state:
                    self._state_cache[to_agent].append({
                        'state': context.semantic_state,
                        'query': f"[Handoff from {from_agent}]",
                        'topic': context.metadata.get('topic'),
                        'timestamp': datetime.now()
                    })

            logger.info(f"Handoff completed: {from_agent} -> {to_agent}")
            return True

    async def get_all_agents(self) -> List[AgentInfo]:
        """Get all registered agents."""
        async with self._lock:
            return list(self._agents.values())

    async def get_all_topics(self) -> List[TopicThread]:
        """Get all topic threads."""
        async with self._lock:
            return list(self._topics.values())

    async def get_agent_by_room(self, room_id: str) -> Optional[AgentInfo]:
        """Get agent for a room."""
        async with self._lock:
            agent_id = self._room_agents.get(room_id)
            return self._agents.get(agent_id) if agent_id else None

    async def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        async with self._lock:
            active_agents = sum(
                1 for a in self._agents.values()
                if a.status == AgentStatus.ACTIVE
            )

            total_queries = sum(a.query_count for a in self._agents.values())

            return {
                'total_agents': len(self._agents),
                'active_agents': active_agents,
                'total_topics': len(self._topics),
                'total_queries': total_queries,
                'total_handoffs': len(self._handoff_history),
                'cache_entries': sum(
                    len(v) for v in self._state_cache.values()
                )
            }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# Global Registry Instance
# ============================================================================

def get_semantic_registry() -> SharedSemanticRegistry:
    """Get the global SharedSemanticRegistry instance."""
    return SharedSemanticRegistry()


# ============================================================================
# Multi-Agent Matrix Handlers
# ============================================================================

class MultiAgentHandlers:
    """
    Matrix bot handlers for multi-agent semantic coordination.

    Provides commands for:
    - Agent listing and status
    - Topic thread management
    - Agent handoffs
    - Cross-room linking
    """

    def __init__(self, bot, embedder=None):
        """
        Initialize multi-agent handlers.

        Args:
            bot: MatrixBot instance
            embedder: Optional embedding function
        """
        self.bot = bot
        self.embedder = embedder
        self.registry = get_semantic_registry()

        # Auto-register agent for each room
        self._auto_register = True

        logger.info("MultiAgentHandlers initialized (Phase 7)")

    async def ensure_agent_registered(self, room_id: str) -> AgentInfo:
        """Ensure an agent is registered for this room."""
        agent = await self.registry.get_agent_by_room(room_id)
        if not agent:
            agent_id = f"agent_{room_id[:8]}"
            agent = await self.registry.register_agent(agent_id, room_id)
        return agent

    @chatops_handler(
        command="agents",
        description="Multi-agent semantic coordination (Phase 7)",
        usage="!agents list | topics | handoff | link | stats",
        category=HandlerCategory.LEARNING,
        aliases=["agent", "multiagent"]
    )
    async def handle_agents(self, room: 'MatrixRoom', event: 'RoomMessageText', args: str):
        """
        Main multi-agent command handler.

        Usage:
            !agents list - Show registered agents
            !agents topics - Show topic threads
            !agents handoff <agent_id> - Initiate handoff
            !agents link <thread_id> - Link room to topic
            !agents stats - Show coordination stats
            !agents help - Show help
        """
        # Auto-register this room's agent
        await self.ensure_agent_registered(room.room_id)

        if not args:
            await self._show_help(room)
            return

        parts = args.split(maxsplit=1)
        subcommand = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""

        if subcommand == "help":
            await self._show_help(room)
        elif subcommand == "list":
            await self._list_agents(room)
        elif subcommand == "topics":
            await self._list_topics(room)
        elif subcommand == "handoff":
            await self._initiate_handoff(room, event, subargs)
        elif subcommand == "link":
            await self._link_topic(room, event, subargs)
        elif subcommand == "stats":
            await self._show_stats(room)
        elif subcommand == "discover":
            await self._discover_related(room, event, subargs)
        else:
            await self.bot.send_message(
                room.room_id,
                f"❌ Unknown subcommand: `{subcommand}`\nUse `!agents help` for usage.",
                markdown=True
            )

    async def _show_help(self, room: 'MatrixRoom'):
        """Show multi-agent help."""
        help_text = """
**🤖 Multi-Agent Semantic Coordination (Phase 7)**

`!agents list` - Show registered agents
  → Lists all agents with status and semantic summary

`!agents topics` - Show topic threads
  → Lists cross-room topic threads linking conversations

`!agents handoff <agent_id>` - Initiate handoff
  → Transfer semantic context to another agent

`!agents link <thread_id>` - Link to topic
  → Link this room to an existing topic thread

`!agents discover [query]` - Discover related
  → Find related topic threads based on semantic similarity

`!agents stats` - Show statistics
  → Multi-agent coordination statistics

**Architecture:**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Agent A │────→│ Registry│←────│ Agent B │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────────┴───────────────┘
              Topic Threads
         (Cross-room linking)
```

**Handoff Flow:**
1. Agent A initiates handoff to Agent B
2. Semantic context packaged (queries, states, topic)
3. Agent B receives context and continues conversation
"""
        await self.bot.send_message(room.room_id, help_text, markdown=True)

    async def _list_agents(self, room: 'MatrixRoom'):
        """List all registered agents."""
        agents = await self.registry.get_all_agents()

        if not agents:
            await self.bot.send_message(
                room.room_id,
                "📋 No agents registered yet.",
                markdown=True
            )
            return

        lines = ["**📋 Registered Agents**\n"]

        for agent in sorted(agents, key=lambda a: a.last_active, reverse=True):
            status_emoji = {
                AgentStatus.ACTIVE: "🟢",
                AgentStatus.IDLE: "🟡",
                AgentStatus.HANDOFF_PENDING: "🔄",
                AgentStatus.OFFLINE: "🔴"
            }.get(agent.status, "⚪")

            # Format semantic summary if available
            summary = ""
            if agent.semantic_summary:
                top_cats = sorted(
                    agent.semantic_summary.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:2]
                summary = " | " + ", ".join(
                    f"{cat}: {score:.2f}" for cat, score in top_cats
                )

            lines.append(
                f"{status_emoji} `{agent.agent_id}`\n"
                f"   Room: `{agent.room_id[:20]}...`\n"
                f"   Queries: {agent.query_count}{summary}"
            )

        await self.bot.send_message(room.room_id, "\n".join(lines), markdown=True)

    async def _list_topics(self, room: 'MatrixRoom'):
        """List all topic threads."""
        topics = await self.registry.get_all_topics()

        if not topics:
            await self.bot.send_message(
                room.room_id,
                "🧵 No topic threads yet. Create one with semantic queries!",
                markdown=True
            )
            return

        lines = ["**🧵 Topic Threads**\n"]

        for thread in sorted(topics, key=lambda t: t.last_activity, reverse=True):
            lines.append(
                f"**{thread.topic_name}** (`{thread.thread_id}`)\n"
                f"   Rooms: {len(thread.room_ids)} | "
                f"Agents: {len(thread.agent_ids)} | "
                f"Messages: {thread.message_count}"
            )

        await self.bot.send_message(room.room_id, "\n".join(lines), markdown=True)

    async def _initiate_handoff(
        self,
        room: 'MatrixRoom',
        event: 'RoomMessageText',
        target_agent: str
    ):
        """Initiate a handoff to another agent."""
        if not target_agent:
            await self.bot.send_message(
                room.room_id,
                "❌ Usage: `!agents handoff <agent_id>`\n"
                "Use `!agents list` to see available agents.",
                markdown=True
            )
            return

        # Get current room's agent
        current = await self.registry.get_agent_by_room(room.room_id)
        if not current:
            await self.bot.send_message(
                room.room_id,
                "❌ No agent registered for this room.",
                markdown=True
            )
            return

        # Initiate handoff
        context = await self.registry.initiate_handoff(
            from_agent=current.agent_id,
            to_agent=target_agent,
            include_history=10
        )

        if context:
            # Format handoff summary
            summary = (
                f"**🔄 Handoff Initiated**\n\n"
                f"From: `{context.from_agent}`\n"
                f"To: `{context.to_agent}`\n"
                f"Queries transferred: {len(context.queries)}\n"
                f"Topic: {context.metadata.get('topic', 'None')}\n\n"
                f"Target agent can accept with context preservation."
            )

            await self.bot.send_message(room.room_id, summary, markdown=True)

            # Auto-complete for demo
            await self.registry.complete_handoff(context)
        else:
            await self.bot.send_message(
                room.room_id,
                f"❌ Failed to initiate handoff to `{target_agent}`.\n"
                "Check that the agent exists with `!agents list`.",
                markdown=True
            )

    async def _link_topic(
        self,
        room: 'MatrixRoom',
        event: 'RoomMessageText',
        thread_id: str
    ):
        """Link this room to a topic thread."""
        if not thread_id:
            await self.bot.send_message(
                room.room_id,
                "❌ Usage: `!agents link <thread_id>`\n"
                "Use `!agents topics` to see available threads.",
                markdown=True
            )
            return

        agent = await self.registry.get_agent_by_room(room.room_id)
        if not agent:
            await self.bot.send_message(
                room.room_id,
                "❌ No agent registered for this room.",
                markdown=True
            )
            return

        success = await self.registry.link_room_to_topic(
            thread_id=thread_id,
            room_id=room.room_id,
            agent_id=agent.agent_id
        )

        if success:
            topics = await self.registry.get_all_topics()
            thread = next((t for t in topics if t.thread_id == thread_id), None)
            topic_name = thread.topic_name if thread else thread_id

            await self.bot.send_message(
                room.room_id,
                f"✅ **Room linked to topic thread**\n\n"
                f"Topic: {topic_name}\n"
                f"Thread ID: `{thread_id}`\n"
                f"Now participating in cross-room discussion!",
                markdown=True
            )
        else:
            await self.bot.send_message(
                room.room_id,
                f"❌ Topic thread `{thread_id}` not found.",
                markdown=True
            )

    async def _discover_related(
        self,
        room: 'MatrixRoom',
        event: 'RoomMessageText',
        query: str
    ):
        """Discover related topic threads based on query."""
        if not SEMANTIC_AVAILABLE:
            await self.bot.send_message(
                room.room_id,
                "⚠️ Semantic calculus not available.",
                markdown=True
            )
            return

        if not query:
            await self.bot.send_message(
                room.room_id,
                "❌ Usage: `!agents discover <query>`\n"
                "Enter a query to find related topic threads.",
                markdown=True
            )
            return

        # Create semantic state from query
        # Use mock embedding for demo
        embedding = self._create_mock_embedding(query)

        class MockState:
            def __init__(self, emb):
                self._emb = emb
            def get_embedding(self):
                return self._emb

        state = MockState(embedding)
        related = await self.registry.find_related_topics(
            state, exclude_room=room.room_id
        )

        if not related:
            await self.bot.send_message(
                room.room_id,
                f"🔍 No related topics found for: *{query}*\n\n"
                "Start a new topic thread with `!semantic <query>`",
                markdown=True
            )
            return

        lines = [f"**🔍 Related Topics for:** *{query}*\n"]

        for thread, similarity in related[:5]:
            lines.append(
                f"- **{thread.topic_name}** (`{thread.thread_id}`)\n"
                f"  Similarity: {similarity:.1%} | "
                f"Rooms: {len(thread.room_ids)}"
            )

        lines.append("\n*Use `!agents link <thread_id>` to join a thread.*")

        await self.bot.send_message(room.room_id, "\n".join(lines), markdown=True)

    async def _show_stats(self, room: 'MatrixRoom'):
        """Show multi-agent coordination statistics."""
        stats = await self.registry.get_stats()

        text = f"""
**📊 Multi-Agent Coordination Stats**

**Agents:**
- Total: {stats['total_agents']}
- Active: {stats['active_agents']}

**Topics:**
- Total threads: {stats['total_topics']}

**Activity:**
- Total queries: {stats['total_queries']}
- Total handoffs: {stats['total_handoffs']}
- Cached states: {stats['cache_entries']}
"""
        await self.bot.send_message(room.room_id, text, markdown=True)

    def _create_mock_embedding(self, text: str, dim: int = 244) -> np.ndarray:
        """Create a deterministic mock embedding from text."""
        seed = hash(text) % (2**31)
        np.random.seed(seed)
        embedding = np.random.randn(dim) * 0.5

        text_lower = text.lower()
        category_keywords = {
            "Creative": ["write", "poem", "story", "creative"],
            "Cognitive": ["analyze", "explain", "understand", "think"],
            "Emotional": ["feel", "emotion", "happy", "sad"],
        }

        for cat, keywords in category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                if cat in SEMANTIC_CATEGORIES:
                    for idx in SEMANTIC_CATEGORIES[cat]:
                        if idx < dim:
                            embedding[idx] += 2.0
                break

        return embedding


# ============================================================================
# Integration Functions
# ============================================================================

def register_multi_agent_handlers(bot, embedder=None):
    """
    Register multi-agent handlers with a Matrix bot.

    Args:
        bot: MatrixBot instance
        embedder: Optional embedding function

    Returns:
        MultiAgentHandlers instance
    """
    handlers = MultiAgentHandlers(bot, embedder)
    return handlers


async def notify_semantic_update(
    room_id: str,
    semantic_state: Any,
    query: str,
    topic: Optional[str] = None
):
    """
    Notify registry of a semantic state update.

    Call this from semantic handlers to keep registry in sync.

    Args:
        room_id: Room where update occurred
        semantic_state: Updated semantic state
        query: Query that was processed
        topic: Detected topic
    """
    registry = get_semantic_registry()
    agent = await registry.get_agent_by_room(room_id)

    if agent:
        await registry.update_agent_state(
            agent_id=agent.agent_id,
            semantic_state=semantic_state,
            query=query,
            topic=topic
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core classes
    'SharedSemanticRegistry',
    'AgentInfo',
    'AgentStatus',
    'TopicThread',
    'HandoffContext',

    # Handlers
    'MultiAgentHandlers',

    # Functions
    'get_semantic_registry',
    'register_multi_agent_handlers',
    'notify_semantic_update',
]
