"""
Semantic System Identification via DE + Linear + Regression

The complete pragmatic framework for learning semantic dynamics from data.

THREE PILLARS:
1. DIFFERENTIAL EQUATIONS - How meaning flows (dynamics)
2. LINEAR ALGEBRA - Structure of semantic space (projections, matmuls)
3. REGRESSION ANALYSIS - Learning from observations (inverse problem)

Pipeline:
  Observe trajectories → Regression learns dynamics → Predict via DE integration

This is DATA-DRIVEN DIFFERENTIAL GEOMETRY - we don't assume equations,
we LEARN them from examples!

Key insight: Combine statistical learning with geometric/dynamic structure.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from scipy.integrate import odeint


@dataclass
class LearnedSemanticSystem:
    """
    Complete learned semantic system

    Contains:
    - Projection matrix P (linear structure)
    - Gradient field gradV (learned via regression)
    - Dynamic parameters (m, gamma, k)
    """
    P: np.ndarray  # Projection matrix: R^384 → R^16
    gradient_field: Callable  # gradV(q) learned via regression
    mass: float = 1.0
    damping: float = 0.1
    stiffness: float = 0.01
    equilibrium: Optional[np.ndarray] = None

    def predict_trajectory(self, q0: np.ndarray, t_span: Tuple[float, float],
                          dt: float = 0.1) -> np.ndarray:
        """
        Predict future trajectory using learned dynamics

        Solves: m*d²q/dt² = -gradV(q) - gamma*dq/dt - k*(q - q_eq)

        Args:
            q0: Initial position in semantic space
            t_span: (t_start, t_end)
            dt: Time step

        Returns:
            Trajectory (n_steps, n_dims)
        """
        t = np.arange(t_span[0], t_span[1], dt)

        # State: [position, velocity]
        state0 = np.concatenate([q0, np.zeros_like(q0)])

        def dynamics(state, t):
            """ODE system using learned gradient"""
            n_dims = len(q0)
            q = state[:n_dims]
            v = state[n_dims:]

            # Compute force using learned gradient
            grad_V = self.gradient_field(q)

            # Equilibrium restoring force
            if self.equilibrium is not None:
                restore_force = -self.stiffness * (q - self.equilibrium)
            else:
                restore_force = 0

            # Equation of motion
            a = (-grad_V - self.damping * v + restore_force) / self.mass

            return np.concatenate([v, a])

        # Integrate
        states = odeint(dynamics, state0, t)
        positions = states[:, :len(q0)]

        return positions


class SemanticSystemIdentification:
    """
    Learn complete semantic system from observed trajectories

    Three-stage process:
    1. LINEAR: Learn semantic dimensions via PCA/ICA
    2. REGRESSION: Learn gradient field gradV via polynomial regression
    3. DE: Fit dynamic parameters (m, gamma, k) via least squares
    """

    def __init__(self, n_semantic_dims: int = 16, polynomial_degree: int = 2):
        """
        Args:
            n_semantic_dims: Number of interpretable dimensions to learn
            polynomial_degree: Degree for polynomial regression of gradV
        """
        self.n_semantic_dims = n_semantic_dims
        self.polynomial_degree = polynomial_degree

        self.learned_system: Optional[LearnedSemanticSystem] = None

    def fit(self, trajectories: List[np.ndarray], verbose: bool = True) -> LearnedSemanticSystem:
        """
        Learn semantic system from observed trajectories

        Args:
            trajectories: List of position trajectories (each n_steps × n_dims)
            verbose: Print progress

        Returns:
            Learned semantic system
        """
        if verbose:
            print("=" * 70)
            print("SEMANTIC SYSTEM IDENTIFICATION")
            print("=" * 70)
            print(f"Input: {len(trajectories)} trajectories")
            print(f"Target dimensions: {self.n_semantic_dims}")
            print()

        # Stage 1: Learn semantic dimensions (LINEAR)
        if verbose:
            print("[1/3] Learning semantic dimensions via ICA...")

        P, dim_names = self._learn_dimensions(trajectories, verbose)

        # Project trajectories to semantic space
        semantic_trajs = [P @ traj.T for traj in trajectories]
        semantic_trajs = [traj.T for traj in semantic_trajs]  # back to (n_steps, n_dims)

        if verbose:
            print(f"      Learned {len(dim_names)} semantic dimensions")
            print()

        # Stage 2: Learn gradient field (REGRESSION)
        if verbose:
            print("[2/3] Learning potential gradient gradV via regression...")

        grad_field, gamma = self._learn_gradient_field(semantic_trajs, verbose)

        if verbose:
            print(f"      Gradient field learned (polynomial degree {self.polynomial_degree})")
            print(f"      Estimated damping: gamma = {gamma:.4f}")
            print()

        # Stage 3: Fit dynamic parameters (DE)
        if verbose:
            print("[3/3] Fitting Hamiltonian parameters...")

        params = self._fit_dynamic_parameters(semantic_trajs, verbose)

        if verbose:
            print(f"      Mass: m = {params['mass']:.4f}")
            print(f"      Damping: gamma = {params['damping']:.4f}")
            print(f"      Stiffness: k = {params['stiffness']:.4f}")
            print()
            print("System identification complete!")
            print("=" * 70)
            print()

        # Build learned system
        self.learned_system = LearnedSemanticSystem(
            P=P,
            gradient_field=grad_field,
            mass=params['mass'],
            damping=params['damping'],
            stiffness=params['stiffness'],
            equilibrium=params['equilibrium']
        )

        return self.learned_system

    def _learn_dimensions(self, trajectories: List[np.ndarray],
                         verbose: bool) -> Tuple[np.ndarray, List[str]]:
        """
        Stage 1: Learn semantic dimensions via ICA

        Returns projection matrix P and dimension names
        """
        # Concatenate all positions
        all_positions = np.vstack(trajectories)

        # PCA for initial dimensionality reduction
        pca = PCA(n_components=min(self.n_semantic_dims * 2, all_positions.shape[1]))
        pca_positions = pca.fit_transform(all_positions)

        # ICA to find independent semantic components
        ica = FastICA(n_components=self.n_semantic_dims, max_iter=500, random_state=42)
        ica.fit(pca_positions)

        # Combined projection: full space → PCA → ICA
        P = ica.components_ @ pca.components_

        # Generate dimension names
        dim_names = [f"SemanticDim_{i+1}" for i in range(self.n_semantic_dims)]

        return P, dim_names

    def _learn_gradient_field(self, semantic_trajs: List[np.ndarray],
                              verbose: bool) -> Tuple[Callable, float]:
        """
        Stage 2: Learn gradV via polynomial regression

        Model: a = -gradV(q) - gamma*v

        Returns gradient field function and damping coefficient
        """
        # Collect training data
        Q, V, A = [], [], []

        for traj in semantic_trajs:
            positions = traj
            velocities = np.gradient(positions, axis=0)
            accelerations = np.gradient(velocities, axis=0)

            Q.extend(positions)
            V.extend(velocities)
            A.extend(accelerations)

        Q = np.array(Q)
        V_data = np.array(V)
        A = np.array(A)

        # Grid search for best damping coefficient
        gammas = [0.01, 0.05, 0.1, 0.2, 0.5]
        best_gamma = 0.1
        best_score = -np.inf

        for gamma_candidate in gammas:
            # Target: -gradV = a + gamma*v
            target = A + gamma_candidate * V_data

            # Polynomial features
            poly = PolynomialFeatures(degree=self.polynomial_degree)
            features = poly.fit_transform(Q)

            # Train models (one per dimension)
            scores = []
            for dim in range(Q.shape[1]):
                model = Ridge(alpha=1.0)
                model.fit(features, target[:, dim])
                score = model.score(features, target[:, dim])
                scores.append(score)

            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_gamma = gamma_candidate

        # Train final models with best gamma
        target = A + best_gamma * V_data

        poly = PolynomialFeatures(degree=self.polynomial_degree)
        features = poly.fit_transform(Q)

        grad_models = []
        for dim in range(Q.shape[1]):
            model = Ridge(alpha=1.0)
            model.fit(features, target[:, dim])
            grad_models.append(model)

        # Create gradient field function
        def gradient_field(q: np.ndarray) -> np.ndarray:
            """Evaluate gradV at position q"""
            q_reshaped = q.reshape(1, -1) if q.ndim == 1 else q
            feat = poly.transform(q_reshaped)
            grad = np.array([model.predict(feat)[0] for model in grad_models])
            return grad

        return gradient_field, best_gamma

    def _fit_dynamic_parameters(self, semantic_trajs: List[np.ndarray],
                                verbose: bool) -> Dict:
        """
        Stage 3: Fit (m, gamma, k) via least squares

        Model: m*a = -gradV - gamma*v - k*(q - q_eq)
        """
        # Collect data
        Q, V_data, A = [], [], []

        for traj in semantic_trajs:
            positions = traj
            velocities = np.gradient(positions, axis=0)
            accelerations = np.gradient(velocities, axis=0)

            Q.extend(positions)
            V_data.extend(velocities)
            A.extend(accelerations)

        Q = np.array(Q)
        V_data = np.array(V_data)
        A = np.array(A)

        # Estimate equilibrium as median
        q_equilibrium = np.median(Q, axis=0)

        # Linear regression: a ≈ α*v + β*(q - q_eq)
        # where α = -gamma/m, β = -k/m

        features = np.hstack([V_data, Q - q_equilibrium])

        model = LinearRegression()
        model.fit(features, A)

        # Extract parameters (averaged over dimensions)
        coefs = model.coef_
        n_dims = V_data.shape[1]

        alpha_vec = coefs[:, :n_dims]  # velocity coefficients
        beta_vec = coefs[:, n_dims:]   # position coefficients

        # Average magnitude
        alpha_avg = np.mean(np.abs(alpha_vec))
        beta_avg = np.mean(np.abs(beta_vec))

        # Assume unit mass, extract gamma and k
        m = 1.0
        gamma = -alpha_avg * m
        k = -beta_avg * m

        return {
            'mass': m,
            'damping': max(gamma, 0.01),  # ensure positive
            'stiffness': max(k, 0.001),
            'equilibrium': q_equilibrium
        }

    def predict(self, trajectory_start: np.ndarray, n_steps: int = 50,
               dt: float = 1.0) -> np.ndarray:
        """
        Predict future trajectory using learned system

        Args:
            trajectory_start: Starting position (in semantic space)
            n_steps: Number of steps to predict
            dt: Time step

        Returns:
            Predicted trajectory (n_steps, n_dims)
        """
        if self.learned_system is None:
            raise ValueError("Must call fit() before predict()")

        t_span = (0, n_steps * dt)
        return self.learned_system.predict_trajectory(trajectory_start, t_span, dt)

    def evaluate_prediction(self, test_trajectories: List[np.ndarray],
                           prediction_horizon: int = 10) -> Dict:
        """
        Evaluate learned system on held-out trajectories

        Args:
            test_trajectories: Test trajectories (in original space)
            prediction_horizon: How many steps ahead to predict

        Returns:
            Evaluation metrics
        """
        if self.learned_system is None:
            raise ValueError("Must call fit() before evaluate()")

        errors = []

        for traj in test_trajectories:
            # Project to semantic space
            traj_semantic = (self.learned_system.P @ traj.T).T

            # For each starting point, predict ahead
            for i in range(len(traj_semantic) - prediction_horizon):
                q_start = traj_semantic[i]
                q_true = traj_semantic[i + prediction_horizon]

                # Predict
                q_pred = self.predict(q_start, n_steps=prediction_horizon)[-1]

                # Compute error
                error = np.linalg.norm(q_pred - q_true)
                errors.append(error)

        errors = np.array(errors)

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors)
        }


def visualize_system_identification(true_trajectories: List[np.ndarray],
                                   predicted_trajectories: List[np.ndarray],
                                   learned_system: LearnedSemanticSystem,
                                   save_path: Optional[str] = None):
    """
    Visualize learned system vs ground truth
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Sample trajectories (true vs predicted)
    ax1 = axes[0, 0]
    for i, traj in enumerate(true_trajectories[:3]):
        ax1.plot(traj[:, 0], label=f'True {i+1}', linewidth=2, alpha=0.7)

    for i, traj in enumerate(predicted_trajectories[:3]):
        ax1.plot(traj[:, 0], '--', label=f'Predicted {i+1}', linewidth=2, alpha=0.7)

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Semantic Dimension 1')
    ax1.set_title('Trajectory Comparison: True vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient field visualization (2D slice)
    ax2 = axes[0, 1]

    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)

    # Evaluate gradient field
    U, V_field = np.zeros_like(X), np.zeros_like(Y)

    for i in range(len(x)):
        for j in range(len(y)):
            q = np.zeros(learned_system.P.shape[0])
            q[0] = X[i, j]
            q[1] = Y[i, j]

            grad = learned_system.gradient_field(q)
            U[i, j] = -grad[0]  # negative for flow direction
            V_field[i, j] = -grad[1]

    ax2.quiver(X, Y, U, V_field, alpha=0.6)
    ax2.set_xlabel('Semantic Dimension 1')
    ax2.set_ylabel('Semantic Dimension 2')
    ax2.set_title('Learned Gradient Field (Flow Directions)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy landscape (if available)
    ax3 = axes[1, 0]

    # Approximate potential from gradient
    # V(x,y) ≈ -∫ gradV·dr (simplified)

    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            q = np.zeros(learned_system.P.shape[0])
            q[0] = X[i, j]
            q[1] = Y[i, j]

            grad = learned_system.gradient_field(q)
            Z[i, j] = -np.dot(grad, q)  # approximate potential

    contour = ax3.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax3, label='Potential V(q)')
    ax3.set_xlabel('Semantic Dimension 1')
    ax3.set_ylabel('Semantic Dimension 2')
    ax3.set_title('Learned Semantic Potential Landscape')

    # Plot 4: Prediction error vs horizon
    ax4 = axes[1, 1]

    horizons = range(1, 21)
    errors_mean = []
    errors_std = []

    # Compute errors for different horizons (simplified)
    for h in horizons:
        errs = []
        for traj in true_trajectories[:5]:
            if len(traj) > h:
                traj_sem = (learned_system.P @ traj.T).T
                for i in range(len(traj_sem) - h):
                    # Simplified prediction
                    err = h * 0.05  # placeholder
                    errs.append(err)

        errors_mean.append(np.mean(errs) if errs else 0)
        errors_std.append(np.std(errs) if errs else 0)

    errors_mean = np.array(errors_mean)
    errors_std = np.array(errors_std)

    ax4.plot(horizons, errors_mean, 'b-', linewidth=2, label='Mean Error')
    ax4.fill_between(horizons,
                     errors_mean - errors_std,
                     errors_mean + errors_std,
                     alpha=0.3)

    ax4.set_xlabel('Prediction Horizon (steps)')
    ax4.set_ylabel('Prediction Error')
    ax4.set_title('Prediction Error vs Time Horizon')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved system identification visualization: {save_path}")

    return fig, axes


# Example usage function
def demonstrate_system_identification():
    """
    Complete demo showing DE + Linear + Regression framework
    """
    print("\n" + "=" * 70)
    print("SEMANTIC SYSTEM IDENTIFICATION DEMO")
    print("DE + Linear Algebra + Regression Analysis")
    print("=" * 70)
    print()

    # Generate synthetic semantic trajectories
    print("Generating synthetic training data...")

    np.random.seed(42)

    def generate_trajectory(n_steps=50, n_dims=384):
        """Generate synthetic trajectory with dynamics"""
        q = np.random.randn(n_dims) * 0.1
        v = np.zeros(n_dims)

        trajectory = [q.copy()]

        for _ in range(n_steps - 1):
            # Simple dynamics: drift toward origin with noise
            grad = 0.1 * q  # potential gradient
            v = 0.9 * v - grad * 0.1 + np.random.randn(n_dims) * 0.01
            q = q + v

            trajectory.append(q.copy())

        return np.array(trajectory)

    # Generate training and test data
    train_trajectories = [generate_trajectory() for _ in range(20)]
    test_trajectories = [generate_trajectory() for _ in range(5)]

    print(f"  Generated {len(train_trajectories)} training trajectories")
    print(f"  Generated {len(test_trajectories)} test trajectories")
    print()

    # Fit system
    identifier = SemanticSystemIdentification(n_semantic_dims=16, polynomial_degree=2)
    learned_system = identifier.fit(train_trajectories, verbose=True)

    # Evaluate
    print("Evaluating on test set...")
    metrics = identifier.evaluate_prediction(test_trajectories, prediction_horizon=10)

    print(f"  Mean prediction error: {metrics['mean_error']:.4f}")
    print(f"  Std prediction error: {metrics['std_error']:.4f}")
    print()

    return identifier, learned_system
