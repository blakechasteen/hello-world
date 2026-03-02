"""
MCTS Agent Orchestrator

Agent powered by MCTS at EVERY decision point:
- Working memory: MCTS explores semantic space
- Pattern selection: MCTS validates before applying
- Tool selection: MCTS simulates outcomes
- Multi-step planning: Hierarchical MCTS

Philosophy: "Monte Carlo all the way down"
"""

from typing import Optional, Dict, List
from pathlib import Path
import time

from hololoom.agents.orchestrator import AgentOrchestrator
from hololoom.agents.working_memory_mcts import MCTSWorkingMemory
from hololoom.agents.learner_mcts import MCTSLearner
from hololoom.agents.types import AgentProfile, AgentStats
from hololoom.memory.graph import KG
from hololoom.protocols.types import Query
from hololoom.fabric.spacetime import Spacetime
from hololoom.config import Config


class MCTSAgentOrchestrator(AgentOrchestrator):
    """
    Agent orchestrator powered by MCTS everywhere.

    MCTS Integration Points:
    1. Working Memory: MCTS explores focus trajectories
    2. Pattern Validation: MCTS simulates pattern outcomes
    3. Tool Selection: MCTS evaluates tool choices (future)
    4. Multi-Query Planning: Hierarchical MCTS (future)
    """

    def __init__(
        self,
        profile: AgentProfile,
        shared_knowledge: KG,
        embedding_model,
        config: Optional[Config] = None,
        persist_dir: Path = Path('./agents_memory'),
        # MCTS configuration
        mcts_working_memory: bool = True,
        mcts_pattern_validation: bool = True,
        mcts_wm_simulations: int = 100,
        mcts_pattern_simulations: int = 50,
        mcts_time_budget: Optional[float] = None
    ):
        # Initialize base orchestrator (but override components)
        self.profile = profile
        self.shared_knowledge = shared_knowledge
        self.emb = embedding_model
        self.config = config or Config.fast()

        # MCTS-POWERED WORKING MEMORY
        if mcts_working_memory:
            self.working_memory = MCTSWorkingMemory(
                profile=profile,
                yarn_graph=shared_knowledge,
                embedding_model=embedding_model,
                mcts_simulations=mcts_wm_simulations,
                mcts_time_budget=mcts_time_budget
            )
        else:
            # Standard working memory
            from hololoom.agents.working_memory import AgentWorkingMemory
            self.working_memory = AgentWorkingMemory(
                profile=profile,
                yarn_graph=shared_knowledge,
                embedding_model=embedding_model
            )

        # MCTS-POWERED PATTERN LEARNER
        if mcts_pattern_validation and profile.enable_learning:
            self.learner = MCTSLearner(
                profile=profile,
                persist_dir=persist_dir,
                mcts_simulations=mcts_pattern_simulations
            )
        elif profile.enable_learning:
            # Standard learner
            from hololoom.agents.learner import WorkingMemoryLearner
            self.learner = WorkingMemoryLearner(
                profile=profile,
                persist_dir=persist_dir
            )
        else:
            self.learner = None

        # Statistics
        self.stats = AgentStats()

        # Weaving orchestrator (set externally if needed)
        self.weaving_orchestrator: Optional['WeavingOrchestrator'] = None

        # MCTS configuration flags
        self.use_mcts_wm = mcts_working_memory
        self.use_mcts_patterns = mcts_pattern_validation

    async def query(
        self,
        query: Query,
        use_mcts: bool = True,
        apply_learned_patterns: bool = True
    ) -> Spacetime:
        """
        Process query with MCTS at every decision point.

        Args:
            query: Query to process
            use_mcts: Whether to use MCTS (False = standard query)
            apply_learned_patterns: Whether to apply learned patterns

        Returns:
            Spacetime with result
        """
        start_time = time.time()

        # STEP 1: WORKING MEMORY (MCTS EXPLORATION)
        if self.use_mcts_wm and use_mcts:
            # MCTS explores semantic space for best focus
            context = await self.working_memory.attend_to(
                query,
                use_mcts=True,
                apply_learned_patterns=False  # Validate separately
            )
        else:
            # Standard attend
            context = await self.working_memory.attend_to(
                query,
                apply_learned_patterns=False
            )

        # STEP 2: PATTERN VALIDATION (MCTS SIMULATION)
        if self.use_mcts_patterns and apply_learned_patterns and self.learner:
            # MCTS validates patterns before applying
            validation_result = await self.learner.validate_and_suggest(
                query=query,
                current_state=self.working_memory.state,
                embedding_model=self.emb
            )

            if validation_result['apply_pattern']:
                # Apply validated pattern with modifications
                pattern = validation_result['pattern']
                mods = validation_result['modifications']

                # Apply semantic focus (with modification)
                focus_scale = mods.get('focus_scale', 1.0)
                semantic_shift = pattern.semantic_region - self.working_memory.state.focus_vector
                self.working_memory.state.focus_vector += focus_scale * 0.3 * semantic_shift

                # Normalize
                import numpy as np
                norm = np.linalg.norm(self.working_memory.state.focus_vector)
                if norm > 0:
                    self.working_memory.state.focus_vector /= norm

                # Apply activations
                activation_threshold = mods.get('activation_threshold', 0.5)
                for node_id, activation in pattern.typical_activation_pattern.items():
                    if activation > activation_threshold:
                        self.working_memory.state.activation_map[node_id] = activation

                # Apply tensions
                tension_top_k = mods.get('tension_top_k')
                if tension_top_k:
                    # Only tension top K threads
                    sorted_threads = sorted(
                        pattern.critical_threads,
                        key=lambda t: pattern.typical_activation_pattern.get(t, 0.0),
                        reverse=True
                    )[:tension_top_k]
                    threads = sorted_threads
                else:
                    threads = pattern.critical_threads

                for thread_id in threads:
                    self.working_memory.state.tensioned_threads.add(thread_id)
                    self.working_memory.state.tension_profile[thread_id] = 0.7

                # Re-retrieve with updated state
                context = await self.working_memory._unified_retrieval(query)

        # STEP 3: REASONING (use weaving orchestrator if available)
        if self.weaving_orchestrator:
            spacetime = await self.weaving_orchestrator.weave(query)
        else:
            spacetime = self._simple_response(query, context)

        # STEP 4: LEARNING (record outcome)
        if self.learner:
            snapshot_data = self.working_memory.record_outcome(spacetime)
            if snapshot_data:
                self.learner.record_snapshot(
                    query=snapshot_data['query'],
                    state=snapshot_data['state'],
                    outcome=snapshot_data['outcome']
                )
                self.working_memory.clear_pending_snapshot()

        # STEP 5: UPDATE STATISTICS
        self.stats.total_queries += 1
        if spacetime.confidence >= self.profile.success_threshold:
            self.stats.successful_queries += 1

        # Update average confidence
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

        # Add timing + MCTS metadata
        elapsed = time.time() - start_time
        spacetime.metadata['agent_query_time_ms'] = elapsed * 1000
        spacetime.metadata['used_mcts_wm'] = self.use_mcts_wm and use_mcts
        spacetime.metadata['used_mcts_patterns'] = self.use_mcts_patterns and apply_learned_patterns

        return spacetime

    def get_mcts_statistics(self) -> Dict:
        """Get comprehensive MCTS statistics"""
        stats = {
            'working_memory': {},
            'pattern_validation': {},
            'overall': {
                'mcts_enabled_wm': self.use_mcts_wm,
                'mcts_enabled_patterns': self.use_mcts_patterns
            }
        }

        # Working memory MCTS stats
        if hasattr(self.working_memory, 'get_mcts_statistics'):
            stats['working_memory'] = self.working_memory.get_mcts_statistics()

        # Pattern validation MCTS stats
        if self.learner and hasattr(self.learner, 'get_mcts_statistics'):
            stats['pattern_validation'] = self.learner.get_mcts_statistics()

        return stats

    def _simple_response(self, query: Query, context: List):
        """Simple baseline response (inherited from parent)"""
        from datetime import datetime
        from hololoom.fabric.spacetime import WeavingTrace

        # Concatenate top context
        context_text = "\n\n".join([shard.text for shard in context[:3]])

        response = f"Based on {len(context)} relevant memories:\n\n{context_text}"

        # Create minimal trace
        now = datetime.now()
        trace = WeavingTrace(
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            retrieval_mode='mcts_baseline',
            tool_selected='answer',
            tool_confidence=0.7
        )

        return Spacetime(
            query_text=query.text,
            response=response,
            tool_used='answer',
            confidence=0.7,
            trace=trace,
            metadata={
                'agent_id': self.profile.agent_id,
                'context_count': len(context),
                'mode': 'mcts_baseline'
            }
        )


def create_mcts_agent(
    profile_name: str,
    shared_knowledge: KG,
    embedding_model,
    config: Optional[Config] = None,
    persist_dir: Path = Path('./agents_memory'),
    mcts_working_memory: bool = True,
    mcts_pattern_validation: bool = True,
    mcts_wm_simulations: int = 100,
    mcts_pattern_simulations: int = 50
) -> MCTSAgentOrchestrator:
    """
    Factory function to create MCTS-powered agent.

    Args:
        profile_name: Agent profile name
        shared_knowledge: Shared KG
        embedding_model: Embedding model
        config: HoloLoom config
        persist_dir: Persistence directory
        mcts_working_memory: Enable MCTS for working memory
        mcts_pattern_validation: Enable MCTS for pattern validation
        mcts_wm_simulations: Simulations for working memory MCTS
        mcts_pattern_simulations: Simulations for pattern validation MCTS

    Returns:
        MCTSAgentOrchestrator instance
    """
    from hololoom.agents.profiles import get_profile

    profile = get_profile(profile_name)

    return MCTSAgentOrchestrator(
        profile=profile,
        shared_knowledge=shared_knowledge,
        embedding_model=embedding_model,
        config=config,
        persist_dir=persist_dir,
        mcts_working_memory=mcts_working_memory,
        mcts_pattern_validation=mcts_pattern_validation,
        mcts_wm_simulations=mcts_wm_simulations,
        mcts_pattern_simulations=mcts_pattern_simulations
    )
