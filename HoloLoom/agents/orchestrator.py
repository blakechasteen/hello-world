"""
Agent Orchestrator - Specialized Agent with Working Memory

Integrates:
- Agent profile (domain-specific configuration)
- Working memory (semantic + graph + computational trinity)
- Pattern learner (learning from success)
- Weaving orchestrator (core reasoning)

Creates specialized bots that:
- Have persistent "top of mind" context (pins)
- Learn what works over time (patterns)
- Share knowledge graph but prioritize different regions
- Maintain separate working memory state
"""

from typing import Optional, Dict, List
from pathlib import Path
import time

from HoloLoom.agents.types import AgentProfile, AgentStats
from HoloLoom.agents.working_memory import AgentWorkingMemory
from HoloLoom.agents.learner import WorkingMemoryLearner
from HoloLoom.memory.graph import KG
from HoloLoom.documentation.types import Query, MemoryShard, Context
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.config import Config


class AgentOrchestrator:
    """
    Specialized agent bot with working memory and learning.

    Features:
    - Domain-specific configuration (profile)
    - Trinity working memory (semantic + graph + computational)
    - Pattern learning (learns from success)
    - Persistence (state survives restarts)
    """

    def __init__(
        self,
        profile: AgentProfile,
        shared_knowledge: KG,
        embedding_model,
        config: Optional[Config] = None,
        persist_dir: Path = Path('./agents_memory')
    ):
        self.profile = profile
        self.shared_knowledge = shared_knowledge
        self.emb = embedding_model
        self.config = config or Config.fast()

        # Working memory (trinity substrate)
        self.working_memory = AgentWorkingMemory(
            profile=profile,
            yarn_graph=shared_knowledge,
            embedding_model=embedding_model
        )

        # Pattern learner (learning & persistence)
        self.learner = WorkingMemoryLearner(
            profile=profile,
            persist_dir=persist_dir
        ) if profile.enable_learning else None

        # Statistics
        self.stats = AgentStats()

        # For integration with WeavingOrchestrator (set externally)
        self.weaving_orchestrator: Optional['WeavingOrchestrator'] = None

    async def query(
        self,
        query: Query,
        apply_learned_patterns: bool = True
    ) -> Spacetime:
        """
        Process query through agent's working memory + reasoning.

        Flow:
        1. Attend to query (shift focus, activate nodes, tension threads)
        2. Apply learned patterns (if enabled)
        3. Retrieve context from working memory
        4. Reason with weaving orchestrator (or simple baseline)
        5. Record outcome for learning
        6. Return result

        Args:
            query: Query to process
            apply_learned_patterns: Whether to apply learned patterns

        Returns:
            Spacetime with result and provenance
        """
        start_time = time.time()

        # STEP 1: Attend to query (working memory)
        context = await self.working_memory.attend_to(
            query,
            apply_learned_patterns=apply_learned_patterns and self.learner is not None
        )

        # STEP 2: Reason (use weaving orchestrator if available, else simple response)
        if self.weaving_orchestrator:
            # Full reasoning with weaving orchestrator
            spacetime = await self.weaving_orchestrator.weave(query)
        else:
            # Simple baseline - just return top context
            spacetime = self._simple_response(query, context)

        # STEP 3: Record outcome for learning
        if self.learner:
            snapshot_data = self.working_memory.record_outcome(spacetime)
            if snapshot_data:
                self.learner.record_snapshot(
                    query=snapshot_data['query'],
                    state=snapshot_data['state'],
                    outcome=snapshot_data['outcome']
                )
                self.working_memory.clear_pending_snapshot()

        # STEP 4: Update statistics
        self.stats.total_queries += 1
        if spacetime.confidence >= self.profile.success_threshold:
            self.stats.successful_queries += 1

        # Update average confidence (running average)
        self.stats.avg_confidence = (
            (self.stats.avg_confidence * (self.stats.total_queries - 1) + spacetime.confidence) /
            self.stats.total_queries
        )

        # Cache tracking
        if spacetime.metadata.get('cache_hit'):
            self.stats.cache_hits += 1
        else:
            self.stats.cache_misses += 1

        # Learning events
        if self.learner:
            self.stats.total_learning_events = len(self.learner.snapshots)
            self.stats.patterns_learned = len(self.learner.patterns)

        # Add timing
        elapsed = time.time() - start_time
        spacetime.metadata['agent_query_time_ms'] = elapsed * 1000

        return spacetime

    def _simple_response(self, query: Query, context: List[MemoryShard]) -> Spacetime:
        """Simple baseline response (no weaving orchestrator)"""
        from datetime import datetime
        from HoloLoom.fabric.spacetime import WeavingTrace

        # Concatenate top context
        context_text = "\n\n".join([shard.text for shard in context[:3]])

        response = f"Based on {len(context)} relevant memories:\n\n{context_text}"

        # Create minimal trace
        now = datetime.now()
        trace = WeavingTrace(
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            retrieval_mode='simple_baseline',
            tool_selected='answer',
            tool_confidence=0.7
        )

        return Spacetime(
            query_text=query.text,
            response=response,
            tool_used='answer',
            confidence=0.7,  # Baseline confidence
            trace=trace,
            metadata={
                'agent_id': self.profile.agent_id,
                'context_count': len(context),
                'mode': 'simple_baseline'
            }
        )

    def pin(self, concept: str, weight: float = 0.5):
        """
        Pin a concept to working memory (persists across queries).

        This affects all three substrates:
        - Semantic: Pulls focus toward concept
        - Graph: Keeps node highly activated
        - Computational: Keeps thread tensioned

        Args:
            concept: Concept to pin (text)
            weight: Pin strength (0.0-1.0)
        """
        self.working_memory.pin(concept, weight)

    async def relax(self):
        """
        Relax working memory (decay activations, detension threads).

        Pinned concepts persist.
        """
        await self.working_memory.relax()

    def get_working_memory_summary(self) -> Dict:
        """Get human-readable summary of working memory state"""
        return self.working_memory.get_state_summary()

    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        if not self.learner:
            return {'learning_disabled': True}

        return self.learner.get_stats()

    def get_agent_stats(self) -> AgentStats:
        """Get agent runtime statistics"""
        return self.stats

    async def save_state(self):
        """Persist agent state (working memory + learned patterns)"""
        if self.learner:
            self.learner.save_state()

    async def close(self):
        """Clean up agent (save state, relax memory)"""
        await self.save_state()
        await self.relax()

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


def create_agent(
    profile_name: str,
    shared_knowledge: KG,
    embedding_model,
    config: Optional[Config] = None,
    persist_dir: Path = Path('./agents_memory')
) -> AgentOrchestrator:
    """
    Factory function to create agent from profile name.

    Args:
        profile_name: Name of profile ("budget", "architecture", etc.)
        shared_knowledge: Shared knowledge graph
        embedding_model: Embedding model (Matryoshka)
        config: Optional HoloLoom config
        persist_dir: Where to persist agent state

    Returns:
        AgentOrchestrator instance
    """
    from HoloLoom.agents.profiles import get_profile

    profile = get_profile(profile_name)

    return AgentOrchestrator(
        profile=profile,
        shared_knowledge=shared_knowledge,
        embedding_model=embedding_model,
        config=config,
        persist_dir=persist_dir
    )
