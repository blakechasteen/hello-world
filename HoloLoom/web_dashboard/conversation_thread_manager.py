"""
Multi-Threaded Conversation Manager with Breakthrough Sharing

Manages multiple conversation threads where:
- Each thread is an independent MCTS-powered agent
- Breakthroughs in one thread feed forward to all other threads
- Background learning pools knowledge across all threads
- Real-time WebSocket notifications for breakthroughs

Architecture:
    User A, Thread 1 → Breakthrough → Immediately feeds to:
                                        - User A, Thread 2
                                        - User B, Thread 1
                                        - User C, Thread 1
                                        - Background learner

Key Innovation: Discoveries in one conversation accelerate ALL conversations.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time
import uuid

from HoloLoom.agents.background_learner import AgentPool, LearningQueue, Experience
from HoloLoom.agents.mcts_breakthrough import (
    BreakthroughDetector,
    FeedForwardBroadcaster,
    Breakthrough
)
from HoloLoom.agents.orchestrator_mcts import create_mcts_agent
from HoloLoom.documentation.types import Query
from HoloLoom.memory.graph import KG
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings


# ============================================================================
# Conversation Thread
# ============================================================================

@dataclass
class ConversationThread:
    """
    Single conversation thread with MCTS agent.

    Represents one conversation stream (could be user-specific or topic-specific).
    """
    thread_id: str
    user_id: str
    agent_name: str  # budget, architecture, research, etc.
    agent: Any  # MCTS agent instance

    # Thread metadata
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0

    # Breakthrough tracking
    breakthroughs_contributed: int = 0
    breakthroughs_received: int = 0

    # WebSocket connection (if active)
    websocket: Optional[Any] = None

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_active = time.time()
        self.message_count += 1


@dataclass
class BreakthroughNotification:
    """Notification about a breakthrough for WebSocket broadcast"""
    breakthrough_id: str
    source_thread_id: str
    source_user_id: str
    agent_name: str

    # Breakthrough details
    reward: float
    impact_score: float
    confidence_jump: float

    # Timing
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket JSON"""
        return {
            'type': 'breakthrough',
            'breakthrough_id': self.breakthrough_id,
            'source_thread': self.source_thread_id,
            'source_user': self.source_user_id,
            'agent': self.agent_name,
            'reward': round(self.reward, 3),
            'impact_score': round(self.impact_score, 3),
            'confidence_jump': round(self.confidence_jump, 3),
            'timestamp': self.timestamp,
            'message': f"💡 Breakthrough in {self.agent_name} (impact: {self.impact_score:.2f})"
        }


# ============================================================================
# Conversation Thread Manager
# ============================================================================

class ConversationThreadManager:
    """
    Manages multiple conversation threads with breakthrough sharing.

    Key Features:
    - Each thread has independent MCTS agent
    - Breakthroughs automatically shared across threads
    - Background learning pools knowledge
    - Real-time WebSocket notifications
    - Thread lifecycle management
    """

    def __init__(
        self,
        kg: KG,
        emb: MatryoshkaEmbeddings,
        agent_pool: Optional[AgentPool] = None,
        enable_breakthrough_sharing: bool = True
    ):
        self.kg = kg
        self.emb = emb
        self.agent_pool = agent_pool
        self.enable_breakthrough_sharing = enable_breakthrough_sharing

        # Thread management
        self.threads: Dict[str, ConversationThread] = {}
        self.threads_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.threads_by_agent: Dict[str, Set[str]] = defaultdict(set)

        # Breakthrough system (shared across all threads)
        self.breakthrough_detector = BreakthroughDetector(
            reward_improvement_threshold=1.8,  # Slightly lower for conversation
            confidence_jump_threshold=0.18,
            min_visits_threshold=2,
            max_breakthrough_memory=500  # Large memory for multi-thread
        )

        self.breakthrough_broadcaster = FeedForwardBroadcaster()

        # Statistics
        self.total_threads_created = 0
        self.total_messages = 0
        self.total_breakthroughs = 0
        self.cross_thread_accelerations = 0

    async def create_thread(
        self,
        user_id: str,
        agent_name: str = 'general',
        websocket: Optional[Any] = None
    ) -> ConversationThread:
        """
        Create new conversation thread.

        Args:
            user_id: User identifier
            agent_name: Agent type (budget, architecture, research, etc.)
            websocket: WebSocket connection for real-time updates

        Returns:
            New ConversationThread
        """
        thread_id = str(uuid.uuid4())

        # Create MCTS agent for this thread
        agent = await create_mcts_agent(
            agent_name,
            self.kg,
            self.emb,
            mcts_working_memory=True,
            mcts_wm_simulations=50,  # Moderate for conversation
            breakthrough_detector=self.breakthrough_detector  # Shared detector
        ).__aenter__()

        # Register agent with broadcaster
        if self.enable_breakthrough_sharing and hasattr(agent, 'working_memory'):
            # If agent's working_memory has MCTS engine, register it
            if hasattr(agent.working_memory, 'mcts_engine'):
                self.breakthrough_broadcaster.register_listener(
                    agent.working_memory.mcts_engine
                )

        # Create thread
        thread = ConversationThread(
            thread_id=thread_id,
            user_id=user_id,
            agent_name=agent_name,
            agent=agent,
            websocket=websocket
        )

        # Register thread
        self.threads[thread_id] = thread
        self.threads_by_user[user_id].add(thread_id)
        self.threads_by_agent[agent_name].add(thread_id)

        self.total_threads_created += 1

        return thread

    async def query_thread(
        self,
        thread_id: str,
        query_text: str,
        use_mcts: bool = True
    ) -> Any:
        """
        Process query in specific thread.

        Automatically detects breakthroughs and broadcasts to other threads.

        Args:
            thread_id: Thread identifier
            query_text: User query
            use_mcts: Enable MCTS search

        Returns:
            Query result (Spacetime)
        """
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")

        thread = self.threads[thread_id]
        query = Query(text=query_text)

        # Process query
        result = await thread.agent.query(query, use_mcts=use_mcts)

        # Update thread activity
        thread.update_activity()
        self.total_messages += 1

        # Check for new breakthroughs
        new_breakthroughs = self.breakthrough_detector.current_breakthroughs

        if new_breakthroughs:
            # Track breakthrough contributions
            thread.breakthroughs_contributed += len(new_breakthroughs)
            self.total_breakthroughs += len(new_breakthroughs)

            # Broadcast to other threads
            for breakthrough in new_breakthroughs:
                await self._broadcast_breakthrough(thread, breakthrough)

        # Provide feedback to agent pool if available
        if self.agent_pool:
            await self.agent_pool.feedback(
                thread.agent_name,
                query,
                result,
                {'helpful': True, 'confidence': result.confidence}
            )

        return result

    async def _broadcast_breakthrough(
        self,
        source_thread: ConversationThread,
        breakthrough: Breakthrough
    ):
        """
        Broadcast breakthrough to all other active threads.

        Args:
            source_thread: Thread that discovered breakthrough
            breakthrough: The breakthrough to broadcast
        """
        # Use broadcaster to notify MCTS engines
        self.breakthrough_broadcaster.broadcast_breakthrough(breakthrough)

        # Create notification
        notification = BreakthroughNotification(
            breakthrough_id=breakthrough.id,
            source_thread_id=source_thread.thread_id,
            source_user_id=source_thread.user_id,
            agent_name=source_thread.agent_name,
            reward=breakthrough.reward,
            impact_score=breakthrough.impact_score,
            confidence_jump=breakthrough.confidence_jump
        )

        # Send WebSocket notifications to other active threads
        for thread_id, thread in self.threads.items():
            if thread_id != source_thread.thread_id and thread.websocket:
                try:
                    await thread.websocket.send_json(notification.to_dict())
                    thread.breakthroughs_received += 1
                    self.cross_thread_accelerations += 1
                except Exception as e:
                    print(f"[ConversationThreadManager] WebSocket send failed: {e}")

    async def close_thread(self, thread_id: str):
        """Close and cleanup thread"""
        if thread_id not in self.threads:
            return

        thread = self.threads[thread_id]

        # Cleanup agent
        if hasattr(thread.agent, '__aexit__'):
            await thread.agent.__aexit__(None, None, None)

        # Unregister
        self.threads_by_user[thread.user_id].discard(thread_id)
        self.threads_by_agent[thread.agent_name].discard(thread_id)
        del self.threads[thread_id]

    def get_user_threads(self, user_id: str) -> List[ConversationThread]:
        """Get all threads for a user"""
        thread_ids = self.threads_by_user.get(user_id, set())
        return [self.threads[tid] for tid in thread_ids if tid in self.threads]

    def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        """Get statistics for specific thread"""
        if thread_id not in self.threads:
            return {}

        thread = self.threads[thread_id]

        return {
            'thread_id': thread_id,
            'user_id': thread.user_id,
            'agent_name': thread.agent_name,
            'created_at': thread.created_at,
            'last_active': thread.last_active,
            'age_seconds': time.time() - thread.created_at,
            'idle_seconds': time.time() - thread.last_active,
            'message_count': thread.message_count,
            'breakthroughs_contributed': thread.breakthroughs_contributed,
            'breakthroughs_received': thread.breakthroughs_received,
            'net_contribution': thread.breakthroughs_contributed - thread.breakthroughs_received
        }

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all threads"""
        return {
            'active_threads': len(self.threads),
            'total_threads_created': self.total_threads_created,
            'total_messages': self.total_messages,
            'total_breakthroughs': self.total_breakthroughs,
            'cross_thread_accelerations': self.cross_thread_accelerations,
            'threads_by_agent': {
                agent: len(thread_ids)
                for agent, thread_ids in self.threads_by_agent.items()
            },
            'breakthrough_detector': self.breakthrough_detector.get_stats(),
            'broadcaster': self.breakthrough_broadcaster.get_stats()
        }

    async def cleanup_idle_threads(self, max_idle_seconds: float = 3600):
        """Close threads that have been idle too long"""
        now = time.time()
        idle_threads = [
            thread_id
            for thread_id, thread in self.threads.items()
            if (now - thread.last_active) > max_idle_seconds
        ]

        for thread_id in idle_threads:
            await self.close_thread(thread_id)

        return len(idle_threads)


# ============================================================================
# Factory Functions
# ============================================================================

async def create_conversation_thread_manager(
    kg: KG,
    emb: MatryoshkaEmbeddings,
    agent_pool: Optional[AgentPool] = None,
    enable_breakthrough_sharing: bool = True
) -> ConversationThreadManager:
    """Create conversation thread manager"""
    return ConversationThreadManager(
        kg=kg,
        emb=emb,
        agent_pool=agent_pool,
        enable_breakthrough_sharing=enable_breakthrough_sharing
    )
