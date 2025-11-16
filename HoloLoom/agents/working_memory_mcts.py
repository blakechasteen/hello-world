"""
MCTS-Powered Working Memory

Uses Monte Carlo Tree Search to explore semantic space and find optimal focus.

Key Innovation: Don't just follow momentum - simulate thousands of possible
focus trajectories and pick the best one.

Actions:
- Move toward activated node
- Move toward query
- Explore random direction
- Activate new node
- Tension thread

Reward: Quality of retrieved context (relevance + diversity)
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import copy

from HoloLoom.agents.working_memory import AgentWorkingMemory
from HoloLoom.agents.mcts_core import MCTSStateSpace, MCTSEngine, MCTSNode
from HoloLoom.agents.types import WorkingMemoryState, AgentProfile
from HoloLoom.memory.graph import KG
from HoloLoom.Documentation.types import Query, MemoryShard


# ============================================================================
# Working Memory Actions
# ============================================================================

@dataclass
class WorkingMemoryAction:
    """Action in working memory state space"""
    type: str  # "move_to_node", "move_to_query", "explore", "activate", "tension"
    target: Optional[str] = None  # Node ID if applicable
    direction: Optional[np.ndarray] = None  # Direction vector if applicable
    magnitude: float = 0.3  # Step size

    def __hash__(self):
        """Hash for set membership checks"""
        direction_tuple = tuple(self.direction.flatten()) if self.direction is not None else None
        return hash((self.type, self.target, direction_tuple, self.magnitude))

    def __eq__(self, other):
        """Equality check"""
        if not isinstance(other, WorkingMemoryAction):
            return False
        if self.type != other.type or self.target != other.target or self.magnitude != other.magnitude:
            return False
        if self.direction is None and other.direction is None:
            return True
        if self.direction is None or other.direction is None:
            return False
        return np.allclose(self.direction, other.direction)


# ============================================================================
# Working Memory State Space for MCTS
# ============================================================================

class WorkingMemoryStateSpace(MCTSStateSpace):
    """
    State space for MCTS working memory exploration.

    State: WorkingMemoryState (focus vector + activation map + tensions)
    Actions: Move focus, activate nodes, tension threads
    Reward: Quality of retrieved context
    """

    def __init__(
        self,
        profile: AgentProfile,
        yarn_graph: KG,
        embedding_model,
        query: Query,
        embedding_dim: int
    ):
        self.profile = profile
        self.yarn = yarn_graph
        self.emb = embedding_model
        self.query = query
        self.embedding_dim = embedding_dim

        # Cache query embedding
        self.query_vector = self.emb.encode_scales([query.text], size=embedding_dim)[0]

    def get_legal_actions(self, state: WorkingMemoryState) -> List[WorkingMemoryAction]:
        """Get legal actions from current state"""
        actions = []

        # ACTION 1: Move toward activated nodes
        highly_activated = [
            node_id for node_id, activation
            in state.activation_map.items()
            if activation > 0.5
        ]

        for node_id in highly_activated[:5]:  # Top 5 to limit branching
            actions.append(WorkingMemoryAction(
                type="move_to_node",
                target=node_id,
                magnitude=0.3
            ))

        # ACTION 2: Move toward query
        actions.append(WorkingMemoryAction(
            type="move_to_query",
            magnitude=0.5
        ))

        # ACTION 3: Explore random directions (3 samples)
        for _ in range(3):
            random_direction = np.random.randn(self.embedding_dim)
            random_direction /= np.linalg.norm(random_direction) + 1e-9
            actions.append(WorkingMemoryAction(
                type="explore",
                direction=random_direction,
                magnitude=0.2
            ))

        # ACTION 4: Activate nearby nodes
        for node_id in list(self.yarn.G.nodes())[:10]:  # Sample nodes
            if node_id not in state.activation_map or state.activation_map[node_id] < 0.5:
                actions.append(WorkingMemoryAction(
                    type="activate",
                    target=node_id
                ))

        # ACTION 5: Tension critical threads
        for node_id in list(state.tensioned_threads)[:5]:
            actions.append(WorkingMemoryAction(
                type="tension",
                target=node_id
            ))

        return actions

    def apply_action(self, state: WorkingMemoryState, action: WorkingMemoryAction) -> WorkingMemoryState:
        """Apply action to state, return new state"""
        new_state = state.copy()

        if action.type == "move_to_node":
            # Move focus toward node
            node_data = self.yarn.G.nodes.get(action.target, {})
            content = node_data.get('content', action.target)
            target_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]

            # Blend
            new_state.focus_vector = (
                (1 - action.magnitude) * new_state.focus_vector +
                action.magnitude * target_vector
            )

            # Normalize
            norm = np.linalg.norm(new_state.focus_vector)
            if norm > 0:
                new_state.focus_vector /= norm

        elif action.type == "move_to_query":
            # Move toward query
            new_state.focus_vector = (
                (1 - action.magnitude) * new_state.focus_vector +
                action.magnitude * self.query_vector
            )

            norm = np.linalg.norm(new_state.focus_vector)
            if norm > 0:
                new_state.focus_vector /= norm

        elif action.type == "explore":
            # Move in random direction
            new_state.focus_vector += action.magnitude * action.direction

            norm = np.linalg.norm(new_state.focus_vector)
            if norm > 0:
                new_state.focus_vector /= norm

        elif action.type == "activate":
            # Activate node
            new_state.activation_map[action.target] = 0.8

        elif action.type == "tension":
            # Tension thread
            new_state.tensioned_threads.add(action.target)
            new_state.tension_profile[action.target] = 0.7

        return new_state

    async def evaluate(self, state: WorkingMemoryState) -> float:
        """
        Evaluate state quality.

        Reward = retrieval quality (relevance + diversity + coverage)
        """
        # Get distances to all nodes from this focus
        distances = {}

        for node_id in list(self.yarn.G.nodes())[:50]:  # Sample for speed
            node_data = self.yarn.G.nodes.get(node_id, {})
            content = node_data.get('content', node_id)

            node_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]

            # Cosine similarity
            norm_product = np.linalg.norm(state.focus_vector) * np.linalg.norm(node_vector)
            if norm_product > 0:
                similarity = np.dot(state.focus_vector, node_vector) / norm_product
            else:
                similarity = 0.0

            distances[node_id] = 1.0 - similarity

        # Get top-k closest nodes
        k = self.profile.context_window_size
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])[:k]

        if not sorted_nodes:
            return 0.0

        # RELEVANCE: Average similarity to query
        relevance_scores = []
        for node_id, _ in sorted_nodes:
            node_data = self.yarn.G.nodes.get(node_id, {})
            content = node_data.get('content', node_id)
            node_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]

            norm_product = np.linalg.norm(self.query_vector) * np.linalg.norm(node_vector)
            if norm_product > 0:
                relevance = np.dot(self.query_vector, node_vector) / norm_product
            else:
                relevance = 0.0

            relevance_scores.append(max(0.0, relevance))

        avg_relevance = np.mean(relevance_scores)

        # DIVERSITY: Pairwise dissimilarity of retrieved nodes
        diversity = 0.0
        if len(sorted_nodes) > 1:
            diversities = []
            for i, (node_i, _) in enumerate(sorted_nodes):
                for node_j, _ in sorted_nodes[i+1:]:
                    content_i = self.yarn.G.nodes.get(node_i, {}).get('content', node_i)
                    content_j = self.yarn.G.nodes.get(node_j, {}).get('content', node_j)

                    vec_i = self.emb.encode_scales([content_i], size=self.embedding_dim)[0]
                    vec_j = self.emb.encode_scales([content_j], size=self.embedding_dim)[0]

                    norm_product = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                    if norm_product > 0:
                        sim = np.dot(vec_i, vec_j) / norm_product
                    else:
                        sim = 0.0

                    diversities.append(1.0 - sim)

            diversity = np.mean(diversities)

        # COVERAGE: Use activation + tension bonuses
        activation_bonus = sum(
            state.activation_map.get(node_id, 0.0)
            for node_id, _ in sorted_nodes
        ) / k

        tension_bonus = sum(
            1.0 if node_id in state.tensioned_threads else 0.0
            for node_id, _ in sorted_nodes
        ) / k

        # Combined reward
        reward = (
            0.5 * avg_relevance +      # Relevance most important
            0.2 * diversity +           # Diversity helps
            0.2 * activation_bonus +    # Activated nodes are good
            0.1 * tension_bonus         # Tensioned threads are ready
        )

        return reward

    def is_terminal(self, state: WorkingMemoryState) -> bool:
        """Working memory exploration has no terminal state (use max_depth instead)"""
        return False

    def copy_state(self, state: WorkingMemoryState) -> WorkingMemoryState:
        """Deep copy state"""
        return state.copy()


# ============================================================================
# MCTS-Powered Working Memory
# ============================================================================

class MCTSWorkingMemory(AgentWorkingMemory):
    """
    Working memory powered by MCTS.

    Uses MCTS to explore semantic space and find optimal focus position.
    """

    def __init__(
        self,
        profile: AgentProfile,
        yarn_graph: KG,
        embedding_model,
        matryoshka_scale: Optional[int] = None,
        mcts_simulations: int = 100,
        mcts_time_budget: Optional[float] = None
    ):
        super().__init__(profile, yarn_graph, embedding_model, matryoshka_scale)

        self.mcts_simulations = mcts_simulations
        self.mcts_time_budget = mcts_time_budget

        # MCTS statistics
        self.mcts_stats = {
            'total_searches': 0,
            'avg_simulations': 0,
            'avg_time': 0,
            'avg_reward_improvement': 0
        }

    async def attend_to(
        self,
        query: Query,
        use_mcts: bool = True,
        apply_learned_patterns: bool = True
    ) -> List[MemoryShard]:
        """
        Attend to query using MCTS exploration.

        Args:
            query: Query to process
            use_mcts: Whether to use MCTS (False = standard attend)
            apply_learned_patterns: Whether to apply learned patterns

        Returns:
            Retrieved memory shards
        """
        if not use_mcts:
            # Fall back to standard attend
            return await super().attend_to(query, apply_learned_patterns)

        # MCTS EXPLORATION
        import time
        start_time = time.time()

        # Create state space
        state_space = WorkingMemoryStateSpace(
            profile=self.profile,
            yarn_graph=self.yarn,
            embedding_model=self.emb,
            query=query,
            embedding_dim=self.embedding_dim
        )

        # Create MCTS engine
        mcts = MCTSEngine(
            state_space=state_space,
            exploration_weight=1.414,
            discount_factor=0.95,
            max_depth=10
        )

        # Run MCTS search
        best_action, root = await mcts.search(
            initial_state=self.state,
            n_simulations=self.mcts_simulations,
            time_budget=self.mcts_time_budget
        )

        # Extract best state from tree
        if root.children:
            best_node = root.most_visited_child()
            if best_node:
                # Update state to best found
                self.state = best_node.state

                # Update statistics
                baseline_value = await state_space.evaluate(root.state)
                best_value = await state_space.evaluate(best_node.state)
                improvement = best_value - baseline_value

                self.mcts_stats['total_searches'] += 1
                self.mcts_stats['avg_simulations'] = (
                    (self.mcts_stats['avg_simulations'] * (self.mcts_stats['total_searches'] - 1) + mcts.total_simulations) /
                    self.mcts_stats['total_searches']
                )
                self.mcts_stats['avg_time'] = (
                    (self.mcts_stats['avg_time'] * (self.mcts_stats['total_searches'] - 1) + (time.time() - start_time)) /
                    self.mcts_stats['total_searches']
                )
                self.mcts_stats['avg_reward_improvement'] = (
                    (self.mcts_stats['avg_reward_improvement'] * (self.mcts_stats['total_searches'] - 1) + improvement) /
                    self.mcts_stats['total_searches']
                )

        # Continue with standard retrieval
        return await self._unified_retrieval(query)

    def get_mcts_statistics(self) -> Dict[str, Any]:
        """Get MCTS statistics"""
        return self.mcts_stats.copy()
