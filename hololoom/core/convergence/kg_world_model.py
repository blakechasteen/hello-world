"""
Knowledge Graph World Model — Forward Prediction via Graph Dynamics
====================================================================

Wire 10 of the structured AI integration roadmap.

Transforms the knowledge graph from a passive memory store into a
predictive world model. Given the current state (activated nodes + edges),
predicts the next state: which edges will activate, which nodes will
become relevant, how the activation landscape will evolve.

Based on: AriGraph (IJCAI 2025) — KG as world model with semantic +
episodic memory, outperforms RL baselines in text-based games.

The key insight: HoloLoom's Yarn Graph already IS a world model —
it has entities, relationships, temporal edges, and spring dynamics.
What's missing is the forward-prediction step: using the graph to
anticipate future states, not just retrieve past knowledge.

Usage:
    model = KGWorldModel()
    prediction = model.predict_next_state(
        current_activations={"quantum": 0.9, "computing": 0.7},
        kg=my_knowledge_graph,
        spring_config=SpringConfig(),
    )
    print(prediction.predicted_activations)  # What will activate next
    print(prediction.confidence)             # How confident the prediction is
    print(prediction.novel_connections)      # Predicted new edges

Author: Claude Code (Structured AI Wiring - Wire 10)
Date: 2026-03-18
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class StatePrediction:
    """Predicted next state of the knowledge graph."""
    predicted_activations: dict[str, float]   # {node_id: predicted_activation}
    activation_deltas: dict[str, float]       # {node_id: change from current}
    emerging_nodes: list[str]                 # Nodes predicted to activate (were inactive)
    fading_nodes: list[str]                   # Nodes predicted to deactivate
    confidence: float                         # Overall prediction confidence
    steps_ahead: int                          # How many steps this predicts


@dataclass
class TrajectoryPrediction:
    """Multi-step trajectory prediction."""
    states: list[StatePrediction]             # Sequence of predicted states
    total_steps: int
    final_activations: dict[str, float]       # Predicted final state
    convergence_step: int | None           # Step at which predictions stabilize
    energy_trajectory: list[float]            # Spring energy over predicted steps


# ============================================================================
# KG World Model
# ============================================================================

class KGWorldModel:
    """
    Forward-prediction engine for knowledge graph dynamics.

    Uses spring dynamics simulation to predict how activation will
    propagate through the graph over future timesteps. This turns
    the KG from "what do I know?" into "what will happen next?"

    Three prediction modes:
    1. **Single-step**: Predict the immediate next state (fast, ~1ms)
    2. **Trajectory**: Predict N steps into the future
    3. **Convergence**: Predict the final equilibrium state

    Predictions are validated against actual outcomes to learn the
    accuracy of the model (via Thompson Sampling on prediction quality).
    """

    def __init__(
        self,
        stiffness: float = 0.8,
        damping: float = 0.85,
        decay: float = 0.99,
        dt: float = 0.2,
        activation_threshold: float = 0.1,
    ):
        """
        Args:
            stiffness: Spring stiffness (Hooke's Law k)
            damping: Velocity damping (energy dissipation)
            decay: Activation decay per step (forgetting)
            dt: Simulation timestep
            activation_threshold: Minimum activation to consider "active"
        """
        self.stiffness = stiffness
        self.damping = damping
        self.decay = decay
        self.dt = dt
        self.activation_threshold = activation_threshold

        # Track prediction accuracy (Thompson Sampling)
        self._prediction_successes = 1.0
        self._prediction_failures = 1.0

        logger.info(
            f"KGWorldModel initialized (k={stiffness}, c={damping}, "
            f"decay={decay}, dt={dt})"
        )

    def _simulate_step(
        self,
        activations: dict[str, float],
        velocities: dict[str, float],
        neighbors: dict[str, list[tuple[str, float]]],
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """
        Simulate one physics step using Euler integration.

        F = -k(a_i - a_j) - c*v_i  (Hooke's Law + damping)

        Args:
            activations: {node: activation}
            velocities: {node: velocity}
            neighbors: {node: [(neighbor, edge_weight), ...]}

        Returns:
            (new_activations, new_velocities, total_energy)
        """
        new_activations = {}
        new_velocities = {}
        total_energy = 0.0

        for node, activation in activations.items():
            velocity = velocities.get(node, 0.0)

            # Compute spring force from neighbors
            force = 0.0
            for neighbor, weight in neighbors.get(node, []):
                neighbor_activation = activations.get(neighbor, 0.0)
                k = self.stiffness * weight
                delta = neighbor_activation - activation
                force += k * delta

                # Spring potential energy
                total_energy += 0.5 * k * delta ** 2

            # Damping force
            force -= self.damping * velocity

            # Update velocity and activation (Euler)
            new_velocity = velocity + force * self.dt
            new_activation = activation + new_velocity * self.dt
            new_activation *= self.decay  # Natural decay

            # Clamp
            new_activation = max(0.0, min(1.0, new_activation))

            new_activations[node] = new_activation
            new_velocities[node] = new_velocity

            # Kinetic energy
            total_energy += 0.5 * new_velocity ** 2

        return new_activations, new_velocities, total_energy

    def _extract_neighbors(self, kg) -> dict[str, list[tuple[str, float]]]:
        """Extract neighbor map from KG."""
        neighbors: dict[str, list[tuple[str, float]]] = {}

        # Get the NetworkX graph
        if hasattr(kg, 'graph') and hasattr(kg.graph, 'edges'):
            graph = kg.graph
        elif hasattr(kg, 'edges'):
            graph = kg
        else:
            return neighbors

        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            neighbors.setdefault(u, []).append((v, weight))
            neighbors.setdefault(v, []).append((u, weight))

        return neighbors

    def predict_next_state(
        self,
        current_activations: dict[str, float],
        kg,
        current_velocities: dict[str, float] | None = None,
    ) -> StatePrediction:
        """
        Predict the immediate next state (single step).

        Args:
            current_activations: {node_id: activation} — current state
            kg: Knowledge graph (HoloLoom KG or NetworkX)
            current_velocities: Optional velocities (default: all zero)

        Returns:
            StatePrediction for the next timestep
        """
        velocities = current_velocities or dict.fromkeys(current_activations, 0.0)
        neighbors = self._extract_neighbors(kg)

        # Include neighbor nodes that aren't in current activations
        all_nodes = set(current_activations.keys())
        for node in list(current_activations.keys()):
            for neighbor, _ in neighbors.get(node, []):
                if neighbor not in all_nodes:
                    current_activations[neighbor] = 0.0
                    velocities[neighbor] = 0.0
                    all_nodes.add(neighbor)

        new_activations, _, _ = self._simulate_step(
            current_activations, velocities, neighbors
        )

        # Compute deltas
        deltas = {
            n: new_activations.get(n, 0.0) - current_activations.get(n, 0.0)
            for n in all_nodes
        }

        # Find emerging and fading nodes
        emerging = [
            n for n in all_nodes
            if current_activations.get(n, 0.0) < self.activation_threshold
            and new_activations.get(n, 0.0) >= self.activation_threshold
        ]
        fading = [
            n for n in all_nodes
            if current_activations.get(n, 0.0) >= self.activation_threshold
            and new_activations.get(n, 0.0) < self.activation_threshold
        ]

        # Confidence based on prediction accuracy prior
        confidence = self._prediction_successes / (
            self._prediction_successes + self._prediction_failures
        )

        return StatePrediction(
            predicted_activations=new_activations,
            activation_deltas=deltas,
            emerging_nodes=emerging,
            fading_nodes=fading,
            confidence=confidence,
            steps_ahead=1,
        )

    def predict_trajectory(
        self,
        current_activations: dict[str, float],
        kg,
        steps: int = 10,
        convergence_threshold: float = 1e-3,
    ) -> TrajectoryPrediction:
        """
        Predict multi-step trajectory.

        Args:
            current_activations: Starting state
            kg: Knowledge graph
            steps: Number of steps to simulate
            convergence_threshold: Energy change below which we consider converged

        Returns:
            TrajectoryPrediction with sequence of states
        """
        neighbors = self._extract_neighbors(kg)

        # Include all reachable nodes
        activations = dict(current_activations)
        all_nodes = set(activations.keys())
        for node in list(activations.keys()):
            for neighbor, _ in neighbors.get(node, []):
                if neighbor not in all_nodes:
                    activations[neighbor] = 0.0
                    all_nodes.add(neighbor)

        velocities = dict.fromkeys(activations, 0.0)
        states = []
        energies = []
        convergence_step = None
        prev_energy = float('inf')

        for step in range(steps):
            new_activations, new_velocities, energy = self._simulate_step(
                activations, velocities, neighbors
            )

            deltas = {
                n: new_activations.get(n, 0.0) - activations.get(n, 0.0)
                for n in all_nodes
            }
            emerging = [
                n for n in all_nodes
                if activations.get(n, 0.0) < self.activation_threshold
                and new_activations.get(n, 0.0) >= self.activation_threshold
            ]
            fading = [
                n for n in all_nodes
                if activations.get(n, 0.0) >= self.activation_threshold
                and new_activations.get(n, 0.0) < self.activation_threshold
            ]

            confidence = self._prediction_successes / (
                self._prediction_successes + self._prediction_failures
            )

            states.append(StatePrediction(
                predicted_activations=dict(new_activations),
                activation_deltas=deltas,
                emerging_nodes=emerging,
                fading_nodes=fading,
                confidence=confidence,
                steps_ahead=step + 1,
            ))
            energies.append(energy)

            # Convergence check
            if convergence_step is None and abs(energy - prev_energy) < convergence_threshold:
                convergence_step = step + 1

            prev_energy = energy
            activations = new_activations
            velocities = new_velocities

        return TrajectoryPrediction(
            states=states,
            total_steps=steps,
            final_activations=activations,
            convergence_step=convergence_step,
            energy_trajectory=energies,
        )

    def validate_prediction(
        self,
        predicted: StatePrediction,
        actual_activations: dict[str, float],
        tolerance: float = 0.15,
    ) -> float:
        """
        Validate a prediction against actual observed state.
        Updates Thompson Sampling prior on prediction accuracy.

        Args:
            predicted: What we predicted
            actual_activations: What actually happened
            tolerance: Max activation difference to count as "correct"

        Returns:
            Accuracy score (fraction of correct predictions)
        """
        correct = 0
        total = 0

        for node in predicted.predicted_activations:
            if node in actual_activations:
                pred_val = predicted.predicted_activations[node]
                actual_val = actual_activations[node]
                if abs(pred_val - actual_val) < tolerance:
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)

        # Update Thompson prior
        if accuracy > 0.5:
            self._prediction_successes += accuracy
        else:
            self._prediction_failures += (1.0 - accuracy)

        logger.debug(
            f"Prediction validation: {correct}/{total} correct "
            f"(accuracy={accuracy:.2f}, prior={self.prediction_confidence():.2f})"
        )

        return accuracy

    def prediction_confidence(self) -> float:
        """Current confidence in predictions (Thompson prior mean)."""
        return self._prediction_successes / (
            self._prediction_successes + self._prediction_failures
        )


# ============================================================================
# Factory
# ============================================================================

def create_kg_world_model(
    stiffness: float = 0.8,
    damping: float = 0.85,
    decay: float = 0.99,
) -> KGWorldModel:
    """Create a KG world model with spring dynamics parameters."""
    return KGWorldModel(stiffness=stiffness, damping=damping, decay=decay)
