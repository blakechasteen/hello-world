"""
Neural Structural Causal Models

Combines:
- Symbolic causal structure (DAG) - interpretable, verifiable
- Neural mechanisms (learned from data) - powerful, adaptive

Instead of hand-coding:
    recovery = 0.2 * age + 0.6 * treatment + noise

We learn:
    recovery = neural_network(age, treatment) ← Learns complex relationships

But we keep the causal structure explicit:
    age → recovery, treatment → recovery
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

from .dag import CausalDAG, CausalNode, CausalEdge

# Optional: Use actual neural networks if available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NeuralMechanism:
    """
    Neural network representing a causal mechanism.

    f: Parents → Child

    Examples:
        Recovery = f(Age, Treatment)
        f is a neural network that learns the relationship
    """
    variable: str
    parents: List[str]
    network: Any  # Neural network or simple function
    trained: bool = False

    def predict(self, parent_values: np.ndarray) -> np.ndarray:
        """Predict child value from parent values."""
        if TORCH_AVAILABLE and isinstance(self.network, nn.Module):
            with torch.no_grad():
                X = torch.FloatTensor(parent_values)
                return self.network(X).numpy()
        else:
            # Fallback: simple linear function
            return self.network(parent_values)


class SimpleNeuralNet:
    """
    Simple neural network (if PyTorch not available).

    Just a 2-layer network with basic functionality.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = np.tanh(X @ self.W1 + self.b1)  # Hidden layer
        y = h @ self.W2 + self.b2            # Output layer
        return y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """Simple gradient descent training."""
        for epoch in range(epochs):
            # Forward pass
            h = np.tanh(X @ self.W1 + self.b1)
            y_pred = h @ self.W2 + self.b2

            # Loss
            loss = np.mean((y_pred - y) ** 2)

            # Backward pass (chain rule)
            dy = 2 * (y_pred - y) / len(y)
            dW2 = h.T @ dy
            db2 = np.sum(dy, axis=0)

            dh = dy @ self.W2.T
            dh_raw = dh * (1 - h ** 2)  # tanh derivative
            dW1 = X.T @ dh_raw
            db1 = np.sum(dh_raw, axis=0)

            # Update weights
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss:.4f}")


class NeuralStructuralCausalModel:
    """
    Hybrid causal model:
    - Symbolic structure (DAG) from domain knowledge
    - Neural mechanisms learned from data

    Usage:
        # Define structure
        nscm = NeuralStructuralCausalModel()
        nscm.set_structure(dag)

        # Learn mechanisms from data
        nscm.fit(data)

        # Perform interventions
        outcome = nscm.intervene({"treatment": 1})
    """

    def __init__(self, dag: Optional[CausalDAG] = None):
        """
        Initialize Neural SCM.

        Args:
            dag: Causal DAG (symbolic structure)
        """
        self.dag = dag if dag else CausalDAG()
        self.mechanisms: Dict[str, NeuralMechanism] = {}
        self.exogenous: Dict[str, Callable] = {}  # Noise distributions

    def set_structure(self, dag: CausalDAG):
        """Set causal structure."""
        self.dag = dag

    def learn_mechanism(
        self,
        variable: str,
        data: np.ndarray,
        variable_names: List[str],
        hidden_dim: int = 32,
        epochs: int = 100
    ):
        """
        Learn neural mechanism for variable from data.

        Args:
            variable: Target variable
            data: Dataset (rows=samples, cols=variables)
            variable_names: Names of columns in data
            hidden_dim: Hidden layer size
            epochs: Training epochs
        """
        parents = list(self.dag.parents(variable))

        if not parents:
            # Root node (no parents) - just learn marginal distribution
            logger.info(f"{variable} has no parents, using marginal distribution")

            var_idx = variable_names.index(variable)
            mean = np.mean(data[:, var_idx])
            std = np.std(data[:, var_idx])

            # Simple constant function
            self.mechanisms[variable] = NeuralMechanism(
                variable=variable,
                parents=[],
                network=lambda X: np.full((len(X), 1), mean) if len(X.shape) > 1 else mean,
                trained=True
            )
            return

        # Get parent and child indices
        parent_indices = [variable_names.index(p) for p in parents]
        var_idx = variable_names.index(variable)

        # Extract training data
        X = data[:, parent_indices]
        y = data[:, var_idx].reshape(-1, 1)

        logger.info(f"Learning mechanism for {variable} from {parents}")
        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  Input dim: {X.shape[1]}")

        # Create and train neural network
        if TORCH_AVAILABLE:
            network = self._create_torch_network(len(parents), hidden_dim)
            self._train_torch_network(network, X, y, epochs)
        else:
            network = SimpleNeuralNet(len(parents), hidden_dim)
            network.fit(X, y, epochs)

        # Store mechanism
        self.mechanisms[variable] = NeuralMechanism(
            variable=variable,
            parents=parents,
            network=network,
            trained=True
        )

        logger.info(f"✓ Learned mechanism for {variable}")

    def _create_torch_network(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create PyTorch neural network."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def _train_torch_network(
        self,
        network: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int
    ):
        """Train PyTorch network."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = network(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def fit(
        self,
        data: np.ndarray,
        variable_names: List[str],
        hidden_dim: int = 32,
        epochs: int = 100
    ):
        """
        Learn all mechanisms from data.

        Args:
            data: Dataset (rows=samples, cols=variables)
            variable_names: Names of columns
            hidden_dim: Hidden layer size
            epochs: Training epochs
        """
        logger.info("Learning neural mechanisms from data...")

        # Learn in topological order (parents before children)
        for variable in self.dag.topological_order():
            self.learn_mechanism(variable, data, variable_names, hidden_dim, epochs)

        logger.info(f"✓ Learned {len(self.mechanisms)} mechanisms")

    def sample(self, n_samples: int = 1, interventions: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Sample from the causal model.

        Args:
            n_samples: Number of samples to generate
            interventions: Optional interventions {variable: value}

        Returns:
            Samples (n_samples × n_variables)
        """
        if interventions is None:
            interventions = {}

        samples = {}

        # Sample in topological order
        for variable in self.dag.topological_order():
            if variable in interventions:
                # Intervention: Set to fixed value
                samples[variable] = np.full(n_samples, interventions[variable])
            else:
                # Generate from mechanism
                mechanism = self.mechanisms.get(variable)

                if mechanism is None:
                    # No mechanism learned, use random
                    samples[variable] = np.random.randn(n_samples)
                elif not mechanism.parents:
                    # Root node
                    value = mechanism.network(np.zeros((n_samples, 1)))
                    samples[variable] = value.flatten()
                else:
                    # Get parent values
                    parent_values = np.column_stack([
                        samples[p] for p in mechanism.parents
                    ])

                    # Predict from neural network
                    prediction = mechanism.predict(parent_values)

                    # Add noise
                    noise = np.random.randn(n_samples) * 0.1
                    samples[variable] = prediction.flatten() + noise

        # Convert to array
        variables = self.dag.topological_order()
        result = np.column_stack([samples[v] for v in variables])

        return result

    def intervene(
        self,
        interventions: Dict[str, float],
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Perform intervention and compute expected outcomes.

        Args:
            interventions: Variables to intervene on {var: value}
            n_samples: Number of Monte Carlo samples

        Returns:
            Expected values of all variables after intervention
        """
        # Sample from intervened model
        samples = self.sample(n_samples, interventions)

        # Compute expectations
        variables = self.dag.topological_order()
        expectations = {}

        for i, var in enumerate(variables):
            expectations[var] = np.mean(samples[:, i])

        return expectations

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float = 1.0,
        control_value: float = 0.0,
        n_samples: int = 1000
    ) -> float:
        """
        Estimate Average Treatment Effect.

        ATE = E[Y|do(X=1)] - E[Y|do(X=0)]

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment condition value
            control_value: Control condition value
            n_samples: Monte Carlo samples

        Returns:
            ATE estimate
        """
        # Intervene: Treatment
        treatment_outcome = self.intervene({treatment: treatment_value}, n_samples)

        # Intervene: Control
        control_outcome = self.intervene({treatment: control_value}, n_samples)

        # ATE = difference in expected outcomes
        ate = treatment_outcome[outcome] - control_outcome[outcome]

        return ate

    def counterfactual(
        self,
        intervention: Dict[str, float],
        evidence: Dict[str, float],
        query: str,
        n_samples: int = 1000
    ) -> float:
        """
        Approximate counterfactual inference.

        Note: This is an approximation. True counterfactuals require
        inverting the mechanisms to infer exogenous variables.

        Args:
            intervention: Counterfactual intervention
            evidence: Observed values
            query: Variable to query
            n_samples: Monte Carlo samples

        Returns:
            Approximate counterfactual value
        """
        # Simplified counterfactual (not full twin network)
        # Just sample from intervened distribution
        result = self.intervene(intervention, n_samples)
        return result[query]

    def __repr__(self):
        n_mechanisms = len(self.mechanisms)
        n_trained = sum(1 for m in self.mechanisms.values() if m.trained)
        return f"NeuralSCM({n_trained}/{n_mechanisms} mechanisms trained)"
