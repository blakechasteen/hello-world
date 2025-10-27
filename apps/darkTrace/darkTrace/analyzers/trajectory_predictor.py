"""
Trajectory Predictor
====================
System identification and trajectory prediction using differential equations.

The TrajectoryPredictor learns the dynamics of semantic trajectories and
predicts future semantic states based on observed patterns.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import logging

from darkTrace.observers.semantic_observer import StateSnapshot
from darkTrace.observers.trajectory_recorder import Trajectory


logger = logging.getLogger(__name__)


@dataclass
class PredictedState:
    """A predicted future semantic state."""

    # Prediction metadata
    step: int  # Steps ahead from current state
    confidence: float  # Prediction confidence (0-1)

    # Predicted position
    position: List[float]  # Predicted 36D position

    # Predicted dynamics
    velocity: Optional[List[float]] = None
    acceleration: Optional[List[float]] = None

    # Uncertainty bounds
    position_std: Optional[List[float]] = None  # Standard deviation per dimension

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "confidence": self.confidence,
            "position": self.position,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "position_std": self.position_std,
        }


class TrajectoryPredictor:
    """
    Predict future semantic states using learned dynamics.

    Uses system identification to learn a dynamical model:
        dx/dt = f(x, t)

    where x is the semantic state vector (36D position).

    Methods:
    - Linear: x(t+1) = Ax(t) + b
    - Polynomial: x(t+1) = A₀ + A₁x(t) + A₂x(t)² + ...
    - Neural: x(t+1) = NN(x(t))

    Usage:
        predictor = TrajectoryPredictor(method="linear")

        # Learn from trajectories
        predictor.fit(trajectories)

        # Predict future
        predictions = predictor.predict(current_state, horizon=10)
    """

    def __init__(
        self,
        method: str = "linear",  # "linear", "polynomial", or "neural"
        polynomial_degree: int = 2,
        regularization: float = 0.01,
    ):
        """
        Initialize trajectory predictor.

        Args:
            method: Prediction method ("linear", "polynomial", "neural")
            polynomial_degree: Degree for polynomial method
            regularization: L2 regularization strength
        """
        if method not in ("linear", "polynomial", "neural"):
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.polynomial_degree = polynomial_degree
        self.regularization = regularization

        # Learned model parameters
        self.model_params: Optional[Dict[str, Any]] = None
        self.dimensions: int = 36  # Default, will be updated on fit
        self.is_fitted: bool = False

        logger.info(f"TrajectoryPredictor initialized with {method} method")

    def fit(self, trajectories: List[Trajectory]):
        """
        Learn dynamics from observed trajectories.

        Args:
            trajectories: List of Trajectory objects to learn from
        """
        if not trajectories:
            raise ValueError("No trajectories provided for fitting")

        logger.info(f"Fitting predictor on {len(trajectories)} trajectories")

        # Extract state sequences
        X, Y = self._extract_state_sequences(trajectories)

        if len(X) == 0:
            raise ValueError("No valid state sequences found")

        # Update dimensions
        self.dimensions = len(X[0])

        # Fit model based on method
        if self.method == "linear":
            self.model_params = self._fit_linear(X, Y)
        elif self.method == "polynomial":
            self.model_params = self._fit_polynomial(X, Y)
        else:  # neural
            self.model_params = self._fit_neural(X, Y)

        self.is_fitted = True
        logger.info(f"Predictor fitted successfully ({self.method})")

    def _extract_state_sequences(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Extract (state, next_state) pairs from trajectories.

        Args:
            trajectories: List of Trajectory objects

        Returns:
            Tuple of (X, Y) where X[i] -> Y[i]
        """
        X = []  # Current states
        Y = []  # Next states

        for traj in trajectories:
            snapshots = traj.snapshots

            for i in range(len(snapshots) - 1):
                curr_state = snapshots[i].position
                next_state = snapshots[i + 1].position

                # Skip if dimensions don't match
                if len(curr_state) != len(next_state):
                    continue

                X.append(curr_state)
                Y.append(next_state)

        return X, Y

    def _fit_linear(
        self,
        X: List[List[float]],
        Y: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Fit linear model: x(t+1) = Ax(t) + b

        Args:
            X: Current states
            Y: Next states

        Returns:
            Dictionary with model parameters {A, b}
        """
        X_np = np.array(X)
        Y_np = np.array(Y)

        # Add bias column
        X_aug = np.hstack([X_np, np.ones((len(X_np), 1))])

        # Solve with L2 regularization: (X'X + λI)⁻¹X'Y
        reg_matrix = self.regularization * np.eye(X_aug.shape[1])
        params = np.linalg.solve(
            X_aug.T @ X_aug + reg_matrix,
            X_aug.T @ Y_np
        )

        # Split into A and b
        A = params[:-1, :]  # Weight matrix
        b = params[-1, :]   # Bias vector

        return {"A": A, "b": b}

    def _fit_polynomial(
        self,
        X: List[List[float]],
        Y: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Fit polynomial model: x(t+1) = A₀ + A₁x(t) + A₂x(t)² + ...

        Args:
            X: Current states
            Y: Next states

        Returns:
            Dictionary with polynomial coefficients
        """
        X_np = np.array(X)
        Y_np = np.array(Y)

        # Create polynomial features
        X_poly = [X_np]
        for degree in range(2, self.polynomial_degree + 1):
            X_poly.append(X_np ** degree)

        X_poly = np.hstack(X_poly + [np.ones((len(X_np), 1))])

        # Solve with regularization
        reg_matrix = self.regularization * np.eye(X_poly.shape[1])
        params = np.linalg.solve(
            X_poly.T @ X_poly + reg_matrix,
            X_poly.T @ Y_np
        )

        return {
            "params": params,
            "degree": self.polynomial_degree,
        }

    def _fit_neural(
        self,
        X: List[List[float]],
        Y: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Fit neural network model.

        Args:
            X: Current states
            Y: Next states

        Returns:
            Dictionary with neural network parameters
        """
        # TODO: Implement neural network fitting
        # For now, fall back to linear
        logger.warning("Neural method not yet implemented, falling back to linear")
        return self._fit_linear(X, Y)

    def predict(
        self,
        current_state: StateSnapshot,
        horizon: int = 10,
    ) -> List[PredictedState]:
        """
        Predict future semantic states.

        Args:
            current_state: Current semantic state
            horizon: Number of steps to predict ahead

        Returns:
            List of PredictedState objects

        Raises:
            RuntimeError: If predictor not fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        predictions = []
        current_pos = np.array(current_state.position)

        for step in range(1, horizon + 1):
            # Predict next state
            if self.method == "linear":
                next_pos = self._predict_linear(current_pos)
            elif self.method == "polynomial":
                next_pos = self._predict_polynomial(current_pos)
            else:  # neural
                next_pos = self._predict_neural(current_pos)

            # Compute confidence (decreases with horizon)
            confidence = max(0.0, 1.0 - (step / horizon) * 0.5)

            # Create predicted state
            pred = PredictedState(
                step=step,
                confidence=confidence,
                position=next_pos.tolist(),
            )

            predictions.append(pred)

            # Update current position for next iteration
            current_pos = next_pos

        return predictions

    def _predict_linear(self, x: np.ndarray) -> np.ndarray:
        """Predict using linear model."""
        A = self.model_params["A"]
        b = self.model_params["b"]
        return A.T @ x + b

    def _predict_polynomial(self, x: np.ndarray) -> np.ndarray:
        """Predict using polynomial model."""
        params = self.model_params["params"]
        degree = self.model_params["degree"]

        # Create polynomial features
        x_poly = [x]
        for d in range(2, degree + 1):
            x_poly.append(x ** d)

        x_poly = np.hstack(x_poly + [np.ones(1)])

        return params.T @ x_poly

    def _predict_neural(self, x: np.ndarray) -> np.ndarray:
        """Predict using neural network."""
        # TODO: Implement neural prediction
        return self._predict_linear(x)

    def evaluate(
        self,
        test_trajectories: List[Trajectory],
        horizon: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate predictor on test trajectories.

        Args:
            test_trajectories: Trajectories to evaluate on
            horizon: Prediction horizon for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Predictor not fitted")

        errors = []

        for traj in test_trajectories:
            snapshots = traj.snapshots

            for i in range(len(snapshots) - horizon):
                # Predict from current state
                predictions = self.predict(snapshots[i], horizon=horizon)

                # Compare with actual future states
                for j, pred in enumerate(predictions):
                    if i + j + 1 >= len(snapshots):
                        break

                    actual = np.array(snapshots[i + j + 1].position)
                    predicted = np.array(pred.position)

                    # Compute error
                    error = np.linalg.norm(actual - predicted)
                    errors.append(error)

        if not errors:
            return {"mae": 0.0, "rmse": 0.0, "n_samples": 0}

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "n_samples": len(errors),
        }
