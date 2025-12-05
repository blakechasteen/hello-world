"""
Spring-Based Relationship Physics

Full integration with HoloLoom's Spring Dynamics system for realistic social relationships.

Relationships are modeled as physical springs following Hooke's Law:
    F = -k × (a_i - a_j) - c × v_i

Where:
- k = stiffness (relationship strength)
- Δa = activation difference (emotional/social tension)
- c = damping (prevents oscillation, models emotional regulation)
- v = velocity (rate of emotional/social change)

This creates emergent behaviors:
- Strong relationships resist change (high stiffness)
- Weak relationships fade naturally (low stiffness + decay)
- Emotional tension propagates through social network
- Communities cluster through spring attraction
"""

from typing import Dict, List, Any, Optional
import networkx as nx

from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig, NodeState
from HoloLoom.memory.graph import KG, KGEdge

from ..config import NeuroHoodConfig
from .. import Resident, Relationship


class RelationshipPhysics:
    """
    Physics-based relationship evolution using HoloLoom's Spring Dynamics.

    Maps social relationships to physical springs:
    - Residents = Nodes
    - Relationships = Springs
    - Activation = Emotional/social engagement
    - Force = Social influence
    """

    def __init__(self, config: NeuroHoodConfig, knowledge_graph: KG):
        """
        Initialize relationship physics engine.

        Args:
            config: NeuroHood configuration
            knowledge_graph: Knowledge graph to store relationships
        """
        self.config = config
        self.kg = knowledge_graph

        # Create spring configuration from NeuroHood config
        self.spring_config = SpringConfig(
            stiffness=config.relationship_stiffness,
            damping=config.relationship_damping,
            decay=config.relationship_decay,
            max_iterations=200,
            convergence_epsilon=5e-4,
            dt=0.2,
            activation_threshold=0.1,
            maintain_seed_activation=True,
            use_advanced_integrator=True,
            integrator_type="verlet",  # Energy-conserving, gold standard
            edge_type_multipliers={
                'FRIEND': 1.2,      # Friendship is strong
                'ROMANTIC': 1.5,    # Romance is very strong
                'FAMILY': 1.3,      # Family bonds are strong
                'NEIGHBOR': 0.8,    # Neighborly relations moderate
                'ENEMY': 0.5,       # Enmity is weak (repulsive)
                'ACQUAINTANCE': 0.6,  # Casual acquaintances weak
            }
        )

        # Spring dynamics engine (initialized when needed)
        self.spring_dynamics: Optional[SpringDynamics] = None

        # Metrics
        self.total_propagations = 0
        self.avg_convergence_iterations = 0.0

    def initialize_from_relationships(
        self,
        residents: Dict[str, Resident],
        relationships: Dict[str, Relationship]
    ):
        """
        Initialize spring dynamics from current relationships.

        Builds knowledge graph with residents as nodes and relationships as edges.

        Args:
            residents: Dict of {resident_id: Resident}
            relationships: Dict of {(id_a, id_b): Relationship}
        """
        # Clear existing graph edges for residents
        # (Keep other entities intact)
        resident_ids = set(residents.keys())

        # Add residents as nodes (if not already present)
        for resident in residents.values():
            # Node already added via memory_graph_id during bootstrap
            pass

        # Add relationships as edges
        for key, relationship in relationships.items():
            resident_a = relationship.resident_a
            resident_b = relationship.resident_b

            # Map relationship type to edge type
            edge_type = self._map_relationship_type(relationship.relationship_type)

            # Weight is relationship strength
            edge = KGEdge(
                head=resident_a,
                tail=resident_b,
                rel_type=edge_type,
                weight=relationship.strength
            )

            self.kg.add_edge(edge)

        # Create spring dynamics engine with graph
        self.spring_dynamics = SpringDynamics(self.kg, self.spring_config)

    def propagate_social_activation(
        self,
        seed_residents: Dict[str, float],
        residents: Dict[str, Resident]
    ) -> Dict[str, float]:
        """
        Propagate social/emotional activation through relationship network.

        Example: Alice gets angry (activation=1.0). This propagates to her
        friends and neighbors based on relationship strength.

        Args:
            seed_residents: {resident_id: activation_level}
            residents: All residents

        Returns:
            {resident_id: final_activation}
        """
        if not self.spring_dynamics:
            # Fallback if not initialized
            return {rid: 0.0 for rid in residents.keys()}

        # Reset previous state
        self.spring_dynamics.reset()

        # Activate seed residents
        self.spring_dynamics.activate_nodes(seed_residents)

        # Propagate activation through springs
        result = self.spring_dynamics.propagate()

        self.total_propagations += 1
        self.avg_convergence_iterations = (
            (self.avg_convergence_iterations * (self.total_propagations - 1) +
             result.iterations) / self.total_propagations
        )

        # Extract activations for all residents
        activations = {}
        for resident_id in residents.keys():
            if resident_id in self.spring_dynamics.node_states:
                activations[resident_id] = self.spring_dynamics.node_states[resident_id].activation
            else:
                activations[resident_id] = 0.0

        return activations

    def update_relationships(
        self,
        relationships: Dict[str, Relationship],
        time_delta: float = 1.0
    ) -> Dict[str, Relationship]:
        """
        Update all relationships using spring physics.

        Applies:
        - Natural decay (relationships weaken over time without interaction)
        - Tension relaxation (conflicts gradually resolve)
        - Stiffness changes (relationship strength evolution)

        Args:
            relationships: Current relationships
            time_delta: Time step in simulation units

        Returns:
            Updated relationships
        """
        for key, relationship in relationships.items():
            # Natural decay (relationships fade without maintenance)
            relationship.strength *= (self.spring_config.decay ** time_delta)

            # Tension gradually returns to zero (conflicts resolve over time)
            relationship.tension *= (self.spring_config.damping ** time_delta)

            # Clamp values
            relationship.strength = max(0.0, min(1.0, relationship.strength))
            relationship.tension = max(-1.0, min(1.0, relationship.tension))

        return relationships

    def apply_social_interaction(
        self,
        relationship: Relationship,
        interaction_type: str,
        residents: Dict[str, Resident],
        intensity: float = 0.1
    ) -> Relationship:
        """
        Apply a social interaction to a relationship.

        Simulates physical spring behavior:
        - Positive interaction: Increases stiffness (strengthens spring)
        - Negative interaction: Decreases stiffness (weakens spring)
        - Intensity: How strong the interaction

        Args:
            relationship: Relationship to modify
            interaction_type: "positive", "negative", "neutral"
            intensity: Strength of interaction (0.0-1.0)
            residents: All residents (for updating)

        Returns:
            Updated relationship
        """
        if interaction_type == "positive":
            # Strengthen relationship
            relationship.strength = min(1.0, relationship.strength + intensity)
            relationship.tension = max(-1.0, relationship.tension - intensity)

            # Record in history
            relationship.history.append({
                "type": "positive_interaction",
                "intensity": intensity,
                "new_strength": relationship.strength,
                "new_tension": relationship.tension,
            })

        elif interaction_type == "negative":
            # Weaken relationship
            relationship.strength = max(0.0, relationship.strength - intensity)
            relationship.tension = min(1.0, relationship.tension + intensity)

            # Record in history
            relationship.history.append({
                "type": "negative_interaction",
                "intensity": intensity,
                "new_strength": relationship.strength,
                "new_tension": relationship.tension,
            })

        elif interaction_type == "neutral":
            # No significant change
            pass

        # Update residents' relationship dicts
        resident_a = residents[relationship.resident_a]
        resident_b = residents[relationship.resident_b]

        resident_a.relationships[relationship.resident_b] = relationship.strength
        resident_b.relationships[relationship.resident_a] = relationship.strength

        # Update knowledge graph edge weight
        self._update_kg_edge(relationship)

        return relationship

    def compute_social_force(
        self,
        resident_a: Resident,
        resident_b: Resident,
        relationship: Relationship
    ) -> float:
        """
        Compute social force between two residents.

        Based on spring physics: F = -k × (a_i - a_j)

        Args:
            resident_a: First resident
            resident_b: Second resident
            relationship: Their relationship

        Returns:
            Social force (positive = attraction, negative = repulsion)
        """
        # Get activation levels (use mood as proxy)
        activation_a = resident_a.internal_state.get("mood", 0.5)
        activation_b = resident_b.internal_state.get("mood", 0.5)

        # Activation difference
        delta_activation = activation_a - activation_b

        # Spring force: F = -k × Δa
        stiffness = relationship.strength
        force = -stiffness * delta_activation

        return force

    def detect_social_clusters(
        self,
        residents: Dict[str, Resident],
        relationships: Dict[str, Relationship]
    ) -> List[List[str]]:
        """
        Detect social clusters (cliques) using spring energy.

        Groups with high internal spring energy (strong connections)
        and low external energy (weak connections) form clusters.

        Args:
            residents: All residents
            relationships: All relationships

        Returns:
            List of clusters (each cluster is list of resident IDs)
        """
        # Build NetworkX graph from relationships
        G = nx.Graph()

        for resident in residents.values():
            G.add_node(resident.resident_id)

        for relationship in relationships.values():
            G.add_edge(
                relationship.resident_a,
                relationship.resident_b,
                weight=relationship.strength
            )

        # Use community detection algorithm
        try:
            import networkx.algorithms.community as nx_community
            communities = nx_community.greedy_modularity_communities(G, weight='weight')
            clusters = [list(community) for community in communities]
        except ImportError:
            # Fallback: connected components
            clusters = [list(component) for component in nx.connected_components(G)]

        return clusters

    def get_relationship_energy(
        self,
        relationships: Dict[str, Relationship]
    ) -> float:
        """
        Calculate total potential energy in relationship network.

        E = Σ (1/2 × k × (a_i - a_j)²)

        High energy = system under stress (conflicts, tensions)
        Low energy = system in equilibrium (harmony)

        Args:
            relationships: All relationships

        Returns:
            Total potential energy
        """
        if not self.spring_dynamics:
            # Fallback calculation
            total_energy = 0.0

            for relationship in relationships.values():
                # E = 1/2 × k × tension²
                stiffness = relationship.strength
                tension = relationship.tension
                energy = 0.5 * stiffness * (tension ** 2)
                total_energy += energy

            return total_energy

        # Use spring dynamics engine
        return self.spring_dynamics._compute_energy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get physics engine metrics."""
        return {
            "total_propagations": self.total_propagations,
            "avg_convergence_iterations": self.avg_convergence_iterations,
            "spring_config": {
                "stiffness": self.spring_config.stiffness,
                "damping": self.spring_config.damping,
                "decay": self.spring_config.decay,
                "integrator": self.spring_config.integrator_type,
            }
        }

    # ========== Internal Helpers ==========

    def _map_relationship_type(self, relationship_type: str) -> str:
        """Map NeuroHood relationship type to KG edge type."""
        mapping = {
            "friend": "FRIEND",
            "romantic": "ROMANTIC",
            "family": "FAMILY",
            "neighbor": "NEIGHBOR",
            "enemy": "ENEMY",
            "acquaintance": "ACQUAINTANCE",
        }

        return mapping.get(relationship_type.lower(), "NEIGHBOR")

    def _update_kg_edge(self, relationship: Relationship):
        """Update knowledge graph edge weight."""
        edge_type = self._map_relationship_type(relationship.relationship_type)

        edge = KGEdge(
            head=relationship.resident_a,
            tail=relationship.resident_b,
            rel_type=edge_type,
            weight=relationship.strength
        )

        self.kg.add_edge(edge)
