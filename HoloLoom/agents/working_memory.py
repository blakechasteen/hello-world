"""
Agent Working Memory - Trinity Substrate

Working memory as unified semantic-graph-computational substrate:
- LEVEL 1 (Semantic): WHERE in 244D space - continuous geometry
- LEVEL 2 (Graph): WHICH nodes activated - discrete structure
- LEVEL 3 (Computational): WHAT's ready for math - warp space

The three levels interact naturally:
- Semantic focus → activates nearby graph nodes
- Activated nodes → get tensioned into warp space
- Tensioned threads → enable computation
- Results → update semantic focus (loop)
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from pathlib import Path

from HoloLoom.agents.types import WorkingMemoryState, AgentProfile
from HoloLoom.memory.graph import KG
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.semantic_calculus.dimensions import EXTENDED_244_DIMENSIONS


class AgentWorkingMemory:
    """
    Working memory with three complementary substrates:
    1. Semantic geometry (244D focus)
    2. Graph activation (network dynamics)
    3. Computational readiness (warp space)
    """

    def __init__(
        self,
        profile: AgentProfile,
        yarn_graph: KG,
        embedding_model,
        matryoshka_scale: Optional[int] = None
    ):
        self.profile = profile
        self.yarn = yarn_graph  # Persistent symbolic memory
        self.emb = embedding_model  # Semantic geometry

        # Detect embedding dimension from model
        if matryoshka_scale is None:
            # Use max size from embedding model
            if hasattr(embedding_model, 'sizes'):
                matryoshka_scale = max(embedding_model.sizes)
            elif hasattr(embedding_model, 'base_dim'):
                matryoshka_scale = embedding_model.base_dim
            else:
                matryoshka_scale = 768  # Default fallback

        self.embedding_dim = matryoshka_scale

        # Current state (trinity)
        self.state = WorkingMemoryState(
            focus_vector=np.zeros(self.embedding_dim),
            attention_radius=profile.attention_radius,
            momentum=profile.momentum,
            activation_map={},
            propagation_decay=profile.propagation_decay,
            tensioned_threads=set(),
            tension_profile={}
        )

        # Warp space (computational substrate) - created on demand
        self.warp: Optional['WarpSpace'] = None

        # For learning (populated during attend_to)
        self._pending_snapshot: Optional[Dict] = None

    async def attend_to(
        self,
        query: Query,
        apply_learned_patterns: bool = True
    ) -> List[MemoryShard]:
        """
        Process query through all three levels:
        1. Shift semantic focus (geometric)
        2. Activate relevant nodes (graph)
        3. Tension threads for computation (computational)

        Returns memories ready for reasoning.
        """

        # LEVEL 1: SEMANTIC SHIFT
        query_vector = self.emb.encode_scales([query.text], size=self.embedding_dim)[0]
        self.state.focus_vector = (
            self.state.momentum * self.state.focus_vector +
            (1 - self.state.momentum) * query_vector
        )

        # Normalize to unit sphere
        norm = np.linalg.norm(self.state.focus_vector)
        if norm > 0:
            self.state.focus_vector /= norm

        # LEVEL 2: GRAPH ACTIVATION
        # Get semantic distances from all nodes to focus
        semantic_distances = self._compute_distances_to_focus()

        # Activate nodes within attention radius
        for node_id, distance in semantic_distances.items():
            if distance < self.state.attention_radius:
                # Activation inversely proportional to distance
                activation = 1.0 - (distance / self.state.attention_radius)
                self.state.activation_map[node_id] = activation
            else:
                # Outside radius - decay existing activation
                self.state.activation_map[node_id] = (
                    self.state.activation_map.get(node_id, 0.0) * 0.5
                )

        # Fallback: if nothing is inside the attention radius (e.g., when
        # running without the full embedding model and using random vectors),
        # still activate the closest nodes so downstream logic has context.
        if not any(a > 0 for a in self.state.activation_map.values()):
            closest = sorted(semantic_distances.items(), key=lambda x: x[1])[:3]
            for node_id, distance in closest:
                activation = max(0.2, 1.0 - distance)
                self.state.activation_map[node_id] = activation

        # Propagate activation through graph edges
        self._propagate_activation()

        # Clean up low activations
        self.state.activation_map = {
            node_id: activation
            for node_id, activation in self.state.activation_map.items()
            if activation > 0.1
        }

        # LEVEL 3: WARP TENSION
        # Get highly activated nodes
        hot_nodes = {
            node_id for node_id, activation
            in self.state.activation_map.items()
            if activation > 0.7
        }

        # Add persistently tensioned threads (pins)
        threads_to_tension = hot_nodes | set(self.state.tension_profile.keys())

        # Update tensioned threads
        self.state.tensioned_threads = threads_to_tension

        # Record for learning (warp operations tracked later)
        self._pending_snapshot = {
            'query': query,
            'state': self.state.copy(),
            'warp_operations': []  # Filled in by orchestrator if available
        }

        # RETRIEVAL: Get memories from all three views
        return await self._unified_retrieval(query)

    def _compute_distances_to_focus(self) -> Dict[str, float]:
        """Compute semantic distance from each node to focus vector"""
        distances = {}

        for node_id in self.yarn.G.nodes():
            # Get node content
            node_data = self.yarn.G.nodes[node_id]
            content = node_data.get('content', node_id)

            # Encode to 244D
            node_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]

            # Compute cosine distance
            norm_product = np.linalg.norm(self.state.focus_vector) * np.linalg.norm(node_vector)
            if norm_product > 0:
                similarity = np.dot(self.state.focus_vector, node_vector) / norm_product
                distance = 1.0 - similarity  # Convert similarity to distance
            else:
                distance = 1.0

            distances[node_id] = distance

        return distances

    def _propagate_activation(self):
        """Propagate activation through graph edges"""
        # Get all edges
        new_activations = self.state.activation_map.copy()

        for source, target, edge_data in self.yarn.G.edges(data=True):
            source_activation = self.state.activation_map.get(source, 0.0)
            if source_activation > 0.3:
                # Propagate to target
                edge_weight = edge_data.get('weight', 1.0)
                propagated = source_activation * edge_weight * self.state.propagation_decay

                # Add to target (accumulate from multiple sources)
                current_target = new_activations.get(target, 0.0)
                new_activations[target] = min(1.0, current_target + propagated)

        self.state.activation_map = new_activations

    async def _unified_retrieval(self, query: Query) -> List[MemoryShard]:
        """
        Retrieve using all three substrates:
        - Semantic: Near focus vector
        - Graph: Highly activated
        - Computational: Tensioned in warp
        """

        # Get all nodes in yarn graph
        all_nodes = list(self.yarn.G.nodes())

        # Score each node
        scored = []
        for node_id in all_nodes:
            node_data = self.yarn.G.nodes[node_id]
            content = node_data.get('content', node_id)

            # SEMANTIC SCORE (distance in 244D)
            node_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]
            norm_product = np.linalg.norm(self.state.focus_vector) * np.linalg.norm(node_vector)
            if norm_product > 0:
                semantic_score = np.dot(self.state.focus_vector, node_vector) / norm_product
            else:
                semantic_score = 0.0

            # ACTIVATION SCORE (graph dynamics)
            activation_score = self.state.activation_map.get(node_id, 0.0)

            # TENSION SCORE (computational readiness)
            tension_score = 1.0 if node_id in self.state.tensioned_threads else 0.3

            # COMBINED SCORE (weighted)
            combined_score = (
                0.4 * semantic_score +      # WHERE (geometric)
                0.4 * activation_score +     # WHICH (graph)
                0.2 * tension_score          # READY (computational)
            )

            # Create memory shard
            shard = MemoryShard(
                id=node_id,
                text=content,
                metadata={'node_id': node_id, 'score': combined_score}
            )
            scored.append((shard, combined_score))

        # Return top-k by combined score
        k = self.profile.context_window_size
        sorted_shards = sorted(scored, key=lambda x: x[1], reverse=True)
        return [shard for shard, score in sorted_shards[:k]]

    def pin(self, concept: str, weight: float = 0.5):
        """
        Pin a concept across all three substrates:
        - Semantic: Pull focus vector toward concept
        - Graph: Boost baseline activation
        - Computational: Keep thread always tensioned
        """
        # SEMANTIC: Pull focus
        concept_vector = self.emb.encode_scales([concept], size=self.embedding_dim)[0]
        self.state.focus_vector += weight * concept_vector

        # Normalize
        norm = np.linalg.norm(self.state.focus_vector)
        if norm > 0:
            self.state.focus_vector /= norm

        # GRAPH: Boost activation
        concept_node = self._find_node_by_content(concept)
        if concept_node:
            self.state.activation_map[concept_node] = 0.8  # High baseline

        # COMPUTATIONAL: Persistent tension
        if concept_node:
            self.state.tension_profile[concept_node] = weight

    def _find_node_by_content(self, content: str) -> Optional[str]:
        """Find graph node by content similarity"""
        concept_vector = self.emb.encode_scales([content], size=self.embedding_dim)[0]

        best_node = None
        best_similarity = 0.0

        for node_id in self.yarn.G.nodes():
            node_data = self.yarn.G.nodes[node_id]
            node_content = node_data.get('content', node_id)
            node_vector = self.emb.encode_scales([node_content], size=self.embedding_dim)[0]

            norm_product = np.linalg.norm(concept_vector) * np.linalg.norm(node_vector)
            if norm_product > 0:
                similarity = np.dot(concept_vector, node_vector) / norm_product
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_node = node_id

        return best_node if best_similarity > 0.7 else None

    async def relax(self):
        """
        Relax working memory (but keep pins):
        - Semantic: Focus drifts toward pinned concepts
        - Graph: Activation decays naturally
        - Computational: Detension non-pinned threads
        """
        # SEMANTIC: Drift toward pins
        if self.state.tension_profile:
            pin_vectors = []
            for thread_id in self.state.tension_profile.keys():
                node_data = self.yarn.G.nodes.get(thread_id, {})
                content = node_data.get('content', thread_id)
                pin_vectors.append(self.emb.encode_scales([content], size=self.embedding_dim)[0])

            if pin_vectors:
                pin_centroid = np.mean(pin_vectors, axis=0)
                self.state.focus_vector = (
                    0.7 * self.state.focus_vector +
                    0.3 * pin_centroid
                )

                # Normalize
                norm = np.linalg.norm(self.state.focus_vector)
                if norm > 0:
                    self.state.focus_vector /= norm

        # GRAPH: Natural decay
        self.state.activation_map = {
            node_id: activation * 0.9  # 10% decay
            for node_id, activation in self.state.activation_map.items()
            if activation > 0.1
        }

        # COMPUTATIONAL: Detension non-pinned threads
        pinned = set(self.state.tension_profile.keys())
        self.state.tensioned_threads = pinned

    def get_state_summary(self) -> Dict:
        """Human-readable summary of working memory"""
        # Get top semantic dimensions
        top_dims_idx = np.argsort(np.abs(self.state.focus_vector))[-5:]
        top_dims = []
        for i in top_dims_idx:
            if i < len(EXTENDED_244_DIMENSIONS):
                top_dims.append(EXTENDED_244_DIMENSIONS[i])
            else:
                top_dims.append(f"dim_{i}")

        return {
            'semantic_focus': top_dims,
            'focus_magnitude': float(np.linalg.norm(self.state.focus_vector)),
            'activated_nodes': len([a for a in self.state.activation_map.values() if a > 0.5]),
            'highly_activated_nodes': len([a for a in self.state.activation_map.values() if a > 0.7]),
            'tensioned_threads': len(self.state.tensioned_threads),
            'pinned_concepts': len(self.state.tension_profile),
            'attention_radius': self.state.attention_radius
        }

    def record_outcome(self, spacetime: Spacetime):
        """Prepare snapshot for learning (actual learning happens in Learner)"""
        if not self._pending_snapshot:
            return

        # Extract outcome
        self._pending_snapshot['outcome'] = {
            'confidence': spacetime.confidence,
            'tool_used': spacetime.metadata.get('tool_used'),
            'warp_operations': spacetime.metadata.get('warp_operations', [])
        }

        return self._pending_snapshot

    def clear_pending_snapshot(self):
        """Clear pending snapshot after learning"""
        self._pending_snapshot = None
