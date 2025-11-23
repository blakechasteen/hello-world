"""
Relationship Structural Causal Model

Learns the causal mechanisms governing neighborhood relationships from simulation data.

Causal Structure:
    personality_compatibility -> relationship_strength
    personality_compatibility -> communication_quality
    communication_quality -> relationship_strength
    conflict_intensity -> stress_level
    conflict_intensity -> relationship_strength
    stress_level -> retaliation_likelihood
    relationship_strength -> conflict_intensity (negative feedback)

The Neural SCM learns these relationships from observed data, enabling:
- Predictions: "If Alice apologizes, what will happen to the relationship?"
- Interventions: do(communication_quality = high) -> measure outcome
- Counterfactuals: "What if Alice had been polite from the start?"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

from HoloLoom.causal.neural_scm import NeuralStructuralCausalModel
from HoloLoom.causal.dag import CausalDAG, CausalNode, CausalEdge

from .. import Relationship, Resident, NeuroHoodState

logger = logging.getLogger(__name__)


@dataclass
class RelationshipData:
    """Training data extracted from simulation history."""
    # Causal variables (6 core variables)
    personality_compatibility: float  # [0, 1] - how well personalities match
    communication_quality: float      # [0, 1] - positive vs negative interactions
    conflict_intensity: float          # [0, 1] - severity of conflicts
    stress_level: float                # [0, 1] - resident stress
    relationship_strength: float       # [0, 1] - bond strength
    retaliation_likelihood: float      # [0, 1] - probability of negative response

    # Metadata
    resident_a: str = ""
    resident_b: str = ""
    timestamp: float = 0.0
    event_context: str = ""


def RelationshipCausalDAG() -> CausalDAG:
    """
    Create the causal DAG for neighborhood relationships.

    Variables:
    - personality_compatibility: Stable trait (how well residents match)
    - communication_quality: Mediating variable (tone of interactions)
    - conflict_intensity: Stressor variable (conflict severity)
    - stress_level: Emotional state (resident stress)
    - relationship_strength: Outcome variable (bond strength)
    - retaliation_likelihood: Behavioral outcome (aggression probability)

    Causal Structure:
    ```
    personality_compatibility
          |         |
          v         v
    communication_quality -> relationship_strength
                                    ^         |
                                    |         v (negative feedback)
                              conflict_intensity
                                    |
                                    v
                              stress_level
                                    |
                                    v
                          retaliation_likelihood
    ```

    Returns:
        CausalDAG with relationship structure
    """
    dag = CausalDAG()

    # Add nodes
    nodes = [
        CausalNode("personality_compatibility", node_type="root"),
        CausalNode("communication_quality", node_type="mediator"),
        CausalNode("conflict_intensity", node_type="stressor"),
        CausalNode("stress_level", node_type="state"),
        CausalNode("relationship_strength", node_type="outcome"),
        CausalNode("retaliation_likelihood", node_type="outcome"),
    ]

    for node in nodes:
        dag.add_node(node)

    # Add edges (causal relationships)
    edges = [
        # Personality affects communication and strength
        CausalEdge("personality_compatibility", "communication_quality",
                   edge_type="positive", strength=0.7),
        CausalEdge("personality_compatibility", "relationship_strength",
                   edge_type="positive", strength=0.5),

        # Communication affects strength
        CausalEdge("communication_quality", "relationship_strength",
                   edge_type="positive", strength=0.8),

        # Conflict affects stress and strength
        CausalEdge("conflict_intensity", "stress_level",
                   edge_type="positive", strength=0.9),
        CausalEdge("conflict_intensity", "relationship_strength",
                   edge_type="negative", strength=-0.6),

        # Stress affects retaliation
        CausalEdge("stress_level", "retaliation_likelihood",
                   edge_type="positive", strength=0.8),

        # Negative feedback: strong relationships reduce conflict
        CausalEdge("relationship_strength", "conflict_intensity",
                   edge_type="negative", strength=-0.4),
    ]

    for edge in edges:
        dag.add_edge(edge)

    logger.info(f"Created Relationship Causal DAG with {len(nodes)} nodes, {len(edges)} edges")

    return dag


def extract_relationship_data(
    relationship: Relationship,
    residents: Dict[str, Resident],
    personality_system: Any = None,
    timestamp: Optional[float] = None
) -> RelationshipData:
    """
    Extract causal variables from a relationship snapshot.

    Args:
        relationship: Current relationship state
        residents: Dict of all residents
        personality_system: PersonalitySystem for compatibility calculation
        timestamp: Current simulation time

    Returns:
        RelationshipData with all causal variables
    """
    resident_a = residents[relationship.resident_a]
    resident_b = residents[relationship.resident_b]

    # 1. Personality compatibility (stable trait)
    if personality_system:
        compatibility_distance = personality_system.compute_personality_distance(
            relationship.resident_a,
            relationship.resident_b
        )
        # Convert distance [0, 8] to compatibility [0, 1]
        # Distance 0 = perfect match (1.0), distance 8 = complete mismatch (0.0)
        personality_compatibility = max(0.0, 1.0 - (compatibility_distance / 8.0))
    else:
        # Fallback: use inverse of tension
        personality_compatibility = max(0.0, 1.0 - abs(relationship.tension))

    # 2. Communication quality (from recent interactions)
    # Analyze interaction history for tone
    positive_interactions = sum(
        1 for event in relationship.history
        if event.get("type") == "positive_interaction"
    )
    negative_interactions = sum(
        1 for event in relationship.history
        if event.get("type") == "negative_interaction"
    )
    total_interactions = positive_interactions + negative_interactions

    if total_interactions > 0:
        communication_quality = positive_interactions / total_interactions
    else:
        # No interactions yet - use neutral
        communication_quality = 0.5

    # 3. Conflict intensity (current tension level)
    conflict_intensity = abs(relationship.tension)

    # 4. Stress level (from resident internal state)
    stress_level = resident_a.internal_state.get("stress", 0.0)

    # 5. Relationship strength (current bond)
    relationship_strength = relationship.strength

    # 6. Retaliation likelihood (behavioral outcome)
    # High stress + low relationship = high retaliation
    # Low stress + high relationship = low retaliation
    retaliation_likelihood = (stress_level * 0.7 + (1.0 - relationship_strength) * 0.3)
    retaliation_likelihood = min(1.0, max(0.0, retaliation_likelihood))

    return RelationshipData(
        personality_compatibility=personality_compatibility,
        communication_quality=communication_quality,
        conflict_intensity=conflict_intensity,
        stress_level=stress_level,
        relationship_strength=relationship_strength,
        retaliation_likelihood=retaliation_likelihood,
        resident_a=relationship.resident_a,
        resident_b=relationship.resident_b,
        timestamp=timestamp or 0.0,
        event_context=relationship.history[-1].get("type", "unknown") if relationship.history else "initial"
    )


class RelationshipSCM:
    """
    Neural Structural Causal Model for neighborhood relationships.

    Learns causal mechanisms from simulation data, enabling:
    - Predictions: "What will happen if...?"
    - Interventions: Force variables to specific values
    - Counterfactuals: "What would have happened if...?"

    Usage:
        # Create SCM
        scm = RelationshipSCM()

        # Collect training data from simulation
        data = []
        for relationship in state.relationships.values():
            data.append(extract_relationship_data(relationship, state.residents, personality))

        # Learn mechanisms
        scm.fit(data)

        # Predict outcome
        outcome = scm.predict({
            "personality_compatibility": 0.8,
            "communication_quality": 0.9,
            "conflict_intensity": 0.2
        })
        print(f"Predicted relationship strength: {outcome['relationship_strength']:.2f}")
    """

    def __init__(self):
        """Initialize Relationship SCM."""
        self.dag = RelationshipCausalDAG()
        self.nscm = NeuralStructuralCausalModel(self.dag)
        self.variable_names = [
            "personality_compatibility",
            "communication_quality",
            "conflict_intensity",
            "stress_level",
            "relationship_strength",
            "retaliation_likelihood",
        ]
        self.trained = False

    def fit(self, training_data: List[RelationshipData], epochs: int = 100):
        """
        Learn causal mechanisms from training data.

        Args:
            training_data: List of RelationshipData snapshots from simulation
            epochs: Training epochs for neural networks
        """
        logger.info(f"Training Relationship SCM with {len(training_data)} samples")

        # Convert to numpy array
        data_matrix = self._to_matrix(training_data)

        logger.info(f"Data matrix shape: {data_matrix.shape}")
        logger.info(f"Variables: {self.variable_names}")

        # Learn mechanisms for each variable (topological order)
        # The DAG.topological_order() ensures parents are learned before children
        for variable in self.dag.topological_order():
            logger.info(f"Learning mechanism for: {variable}")
            self.nscm.learn_mechanism(
                variable=variable,
                data=data_matrix,
                variable_names=self.variable_names,
                hidden_dim=32,
                epochs=epochs
            )

        self.trained = True
        logger.info("Relationship SCM training complete!")

    def _to_matrix(self, data: List[RelationshipData]) -> np.ndarray:
        """Convert RelationshipData list to numpy matrix."""
        matrix = np.zeros((len(data), len(self.variable_names)))

        for i, sample in enumerate(data):
            matrix[i, 0] = sample.personality_compatibility
            matrix[i, 1] = sample.communication_quality
            matrix[i, 2] = sample.conflict_intensity
            matrix[i, 3] = sample.stress_level
            matrix[i, 4] = sample.relationship_strength
            matrix[i, 5] = sample.retaliation_likelihood

        return matrix

    def predict(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Predict all variables given inputs (forward pass through SCM).

        Args:
            inputs: Dict of input variable values

        Returns:
            Dict of all variable values (inputs + predictions)

        Example:
            outcome = scm.predict({
                "personality_compatibility": 0.8,
                "communication_quality": 0.9
            })
            # Returns all 6 variables with predictions for descendants
        """
        if not self.trained:
            raise ValueError("SCM not trained yet. Call fit() first.")

        # Start with input values
        values = inputs.copy()

        # Forward pass through topological order
        for variable in self.dag.topological_order():
            if variable in values:
                continue  # Already have value (input or computed)

            # Get parents
            parents = list(self.dag.parents(variable))

            if not parents:
                # Root node - use learned marginal
                mechanism = self.nscm.mechanisms[variable]
                values[variable] = float(mechanism.predict(np.array([[0.0]])))[0]
            else:
                # Get parent values
                parent_values = np.array([[values[p] for p in parents]])

                # Predict from mechanism
                mechanism = self.nscm.mechanisms[variable]
                values[variable] = float(mechanism.predict(parent_values)[0])

        return values

    def get_structure_summary(self) -> str:
        """Get human-readable summary of causal structure."""
        lines = ["Relationship Causal Structure:", ""]

        for variable in self.dag.topological_order():
            parents = list(self.dag.parents(variable))
            children = list(self.dag.children(variable))

            lines.append(f"  {variable}:")
            if parents:
                lines.append(f"    Caused by: {', '.join(parents)}")
            if children:
                lines.append(f"    Causes: {', '.join(children)}")
            lines.append("")

        return "\n".join(lines)


def simulate_relationship_outcome(
    scm: RelationshipSCM,
    personality_compatibility: float,
    communication_quality: float,
    conflict_intensity: float,
    n_steps: int = 10
) -> List[Dict[str, float]]:
    """
    Simulate relationship dynamics over time.

    Uses the learned SCM to predict how a relationship evolves
    given initial conditions.

    Args:
        scm: Trained RelationshipSCM
        personality_compatibility: Initial compatibility [0, 1]
        communication_quality: Initial communication quality [0, 1]
        conflict_intensity: Initial conflict intensity [0, 1]
        n_steps: Number of time steps to simulate

    Returns:
        List of state dicts (one per time step)

    Example:
        trajectory = simulate_relationship_outcome(
            scm,
            personality_compatibility=0.7,
            communication_quality=0.5,
            conflict_intensity=0.6,
            n_steps=10
        )

        # Plot relationship strength over time
        strengths = [t['relationship_strength'] for t in trajectory]
    """
    trajectory = []

    # Initial state
    state = {
        "personality_compatibility": personality_compatibility,
        "communication_quality": communication_quality,
        "conflict_intensity": conflict_intensity,
    }

    for step in range(n_steps):
        # Predict current state
        state = scm.predict(state)
        trajectory.append(state.copy())

        # Update for next step (feedback loops)
        # Strong relationships reduce conflict (negative feedback)
        state["conflict_intensity"] = state["conflict_intensity"] * 0.9 + \
                                       (1.0 - state["relationship_strength"]) * 0.1

        # Communication quality slowly regresses to mean influenced by compatibility
        state["communication_quality"] = state["communication_quality"] * 0.8 + \
                                         state["personality_compatibility"] * 0.2

        # Clamp values
        for key in ["conflict_intensity", "communication_quality"]:
            state[key] = min(1.0, max(0.0, state[key]))

    return trajectory
