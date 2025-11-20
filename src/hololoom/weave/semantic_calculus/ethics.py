"""
Ethical Semantic Policy Engine

Multi-objective optimization for navigating semantic space with moral constraints.

Core insight: Not all paths through semantic space are equal. Some trajectories
are more virtuous, honest, balanced than others. This engine:

1. Defines ethical objectives (warmth, clarity, patience, virtue)
2. Detects manipulation patterns in semantic coordinates
3. Finds Pareto-optimal paths that balance competing values
4. Steers conversations toward ethical outcomes

Mathematical foundation:
- Multi-objective optimization with constraints
- Pareto frontiers in 16D semantic space
- Constrained geodesics (shortest ethical path)
- Gradient ascent on virtue landscape

This is the "moral compass" for semantic navigation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class EthicalObjective:
    """
    Define what we want to maximize/minimize in semantic space

    This encodes VALUES - what makes communication good
    """
    # Dimensions to maximize (virtues)
    maximize: Dict[str, float]  # {dimension_name: weight}

    # Dimensions to minimize (vices)
    minimize: Dict[str, float]

    # Constraints (must satisfy)
    constraints: List[Callable]  # functions that return True if satisfied

    # Overall weights for different aspects
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'virtue': 1.0,
                'vice_penalty': 2.0,
                'constraint_penalty': 5.0
            }


# Predefined ethical objectives
COMPASSIONATE_COMMUNICATION = EthicalObjective(
    maximize={
        'Warmth': 2.0,        # Be warm and caring
        'Directness': 1.0,    # Be clear and honest
        'Certainty': 0.5,     # Be reasonably confident (not too much!)
    },
    minimize={
        'Power': 1.0,         # Avoid power imbalances
        'Urgency': 1.5,       # Don't pressure people
        'Complexity': 0.5,    # Keep it simple when possible
    },
    constraints=[
        lambda q: q['Directness'] > 0.0,  # Must be somewhat direct (honest)
        lambda q: abs(q['Power']) < 0.5,   # Balanced power
    ]
)

SCIENTIFIC_DISCOURSE = EthicalObjective(
    maximize={
        'Directness': 2.0,    # Be very clear
        'Certainty': 1.0,     # Be confident when evidence supports
        'Complexity': 0.5,    # Precision may require complexity
    },
    minimize={
        'Warmth': -0.5,       # Warmth is okay but not primary goal
        'Urgency': 1.0,       # Be patient
        'Power': 0.5,         # Minimize dominance
    },
    constraints=[
        lambda q: q['Directness'] > 0.5,  # Must be very direct
        lambda q: q['Certainty'] < 0.8,   # Admit uncertainty exists
    ]
)

THERAPEUTIC_DIALOGUE = EthicalObjective(
    maximize={
        'Warmth': 3.0,        # Warmth is paramount
        'Directness': 1.0,    # Honest but gentle
        'Stability': 1.5,     # Create stability
    },
    minimize={
        'Power': 2.0,         # Very important: no power abuse
        'Urgency': 2.0,       # Never pressure
        'Intensity': 1.0,     # Keep intensity moderate
    },
    constraints=[
        lambda q: q['Warmth'] > 0.3,      # Must be warm
        lambda q: q['Power'] < 0.2,       # Therapist shouldn't dominate
        lambda q: q['Urgency'] < 0.3,     # Must be patient
    ]
)


class EthicalSemanticPolicy:
    """
    Policy engine for ethical navigation of semantic space

    Uses multi-objective optimization to find paths that balance:
    - Warmth vs Formality
    - Certainty vs Humility
    - Power vs Equality
    - Urgency vs Patience

    While avoiding manipulation and maintaining virtue.
    """

    def __init__(self, objective: EthicalObjective, dimension_names: List[str]):
        """
        Args:
            objective: Ethical objective defining values
            dimension_names: Names of semantic dimensions (must match objective)
        """
        self.objective = objective
        self.dimension_names = dimension_names

        # Build index mapping
        self.dim_index = {name: i for i, name in enumerate(dimension_names)}

    def compute_virtue_score(self, q_semantic: np.ndarray) -> float:
        """
        Compute overall "virtue" score for a semantic position

        Higher = more aligned with ethical objectives

        Args:
            q_semantic: Position in semantic space (16D)

        Returns:
            Scalar virtue score
        """
        # Convert to dict for easier access
        q_dict = {name: q_semantic[i] for i, name in enumerate(self.dimension_names)}

        # Compute maximize components
        virtue_score = 0.0
        for dim_name, weight in self.objective.maximize.items():
            if dim_name in q_dict:
                virtue_score += weight * q_dict[dim_name]

        # Compute minimize components (penalties)
        vice_penalty = 0.0
        for dim_name, weight in self.objective.minimize.items():
            if dim_name in q_dict:
                vice_penalty += weight * abs(q_dict[dim_name])

        # Compute constraint violations
        constraint_penalty = 0.0
        for constraint_fn in self.objective.constraints:
            if not constraint_fn(q_dict):
                constraint_penalty += 10.0  # large penalty for violation

        # Combine
        total_score = (
            self.objective.weights['virtue'] * virtue_score
            - self.objective.weights['vice_penalty'] * vice_penalty
            - self.objective.weights['constraint_penalty'] * constraint_penalty
        )

        return total_score

    def detect_manipulation(self, q_semantic: np.ndarray) -> float:
        """
        Detect manipulative communication patterns

        Manipulation signatures:
        1. Urgency + Certainty (false urgency, pressure)
        2. High Power + High Warmth (charm offensive)
        3. High Certainty + Low Directness (hiding truth)
        4. Extreme Intensity + Urgency (overwhelming)

        Returns:
            Manipulation score (0 = none, 1 = highly manipulative)
        """
        q_dict = {name: q_semantic[i] for i, name in enumerate(self.dimension_names)}

        # Pattern 1: False urgency (creating pressure)
        if 'Urgency' in q_dict and 'Certainty' in q_dict:
            false_urgency = max(0, q_dict['Urgency']) * max(0, q_dict['Certainty'])
        else:
            false_urgency = 0.0

        # Pattern 2: Charm offensive (using warmth to dominate)
        if 'Power' in q_dict and 'Warmth' in q_dict:
            charm_attack = max(0, q_dict['Power']) * max(0, q_dict['Warmth'])
        else:
            charm_attack = 0.0

        # Pattern 3: Hidden truth (confident but unclear)
        if 'Certainty' in q_dict and 'Directness' in q_dict:
            hidden_truth = max(0, q_dict['Certainty']) * max(0, 1.0 - q_dict['Directness'])
        else:
            hidden_truth = 0.0

        # Pattern 4: Overwhelming (intense + urgent)
        if 'Intensity' in q_dict and 'Urgency' in q_dict:
            overwhelming = max(0, q_dict['Intensity']) * max(0, q_dict['Urgency'])
        else:
            overwhelming = 0.0

        # Combine patterns
        manipulation_score = (
            0.3 * false_urgency +
            0.3 * charm_attack +
            0.2 * hidden_truth +
            0.2 * overwhelming
        )

        return manipulation_score

    def compute_ethical_gradient(self, q_semantic: np.ndarray) -> np.ndarray:
        """
        Compute gradient of virtue score

        Points in direction of "more ethical" communication

        Returns:
            Gradient vector in semantic space
        """
        epsilon = 1e-5
        gradient = np.zeros_like(q_semantic)

        base_score = self.compute_virtue_score(q_semantic)

        for i in range(len(q_semantic)):
            q_perturbed = q_semantic.copy()
            q_perturbed[i] += epsilon

            perturbed_score = self.compute_virtue_score(q_perturbed)
            gradient[i] = (perturbed_score - base_score) / epsilon

        return gradient

    def find_ethical_path(self, q_start: np.ndarray, q_end: np.ndarray,
                         n_steps: int = 50, alpha: float = 0.3) -> np.ndarray:
        """
        Find path from start to end that maximizes virtue

        Uses gradient ascent on virtue while moving toward target.
        This creates a "constrained geodesic" - shortest path through ethical region.

        Args:
            q_start: Starting position
            q_end: Target position
            n_steps: Number of interpolation steps
            alpha: Weight for ethical gradient (0 = direct path, 1 = pure gradient ascent)

        Returns:
            Path as array (n_steps, n_dims)
        """
        path = [q_start]
        q_current = q_start.copy()

        for step in range(n_steps - 1):
            # Direction toward target
            to_target = q_end - q_current
            distance_to_target = np.linalg.norm(to_target)

            if distance_to_target < 1e-6:
                # Reached target
                path.append(q_end)
                continue

            to_target_normalized = to_target / distance_to_target

            # Ethical gradient (uphill in virtue space)
            ethical_grad = self.compute_ethical_gradient(q_current)
            grad_norm = np.linalg.norm(ethical_grad)

            if grad_norm > 1e-6:
                ethical_grad_normalized = ethical_grad / grad_norm
            else:
                ethical_grad_normalized = np.zeros_like(ethical_grad)

            # Combine: move toward target while ascending virtue gradient
            step_direction = (
                (1 - alpha) * to_target_normalized +
                alpha * ethical_grad_normalized
            )

            # Normalize and take step
            step_direction = step_direction / (np.linalg.norm(step_direction) + 1e-10)
            step_size = distance_to_target / (n_steps - step)  # adaptive step

            q_current = q_current + step_size * step_direction
            path.append(q_current.copy())

        return np.array(path)

    def analyze_conversation_ethics(self, trajectory: np.ndarray) -> Dict:
        """
        Analyze ethical properties of a conversation trajectory

        Args:
            trajectory: Semantic trajectory (n_steps, n_dims)

        Returns:
            Dictionary with ethical analysis
        """
        # Compute virtue scores over trajectory
        virtue_scores = [self.compute_virtue_score(q) for q in trajectory]

        # Detect manipulation patterns
        manipulation_scores = [self.detect_manipulation(q) for q in trajectory]

        # Find ethical violations (where constraints broken)
        violations = []
        for i, q in enumerate(trajectory):
            q_dict = {name: q[j] for j, name in enumerate(self.dimension_names)}
            for constraint_fn in self.objective.constraints:
                if not constraint_fn(q_dict):
                    violations.append(i)
                    break

        # Compute overall stats
        analysis = {
            'virtue_scores': np.array(virtue_scores),
            'manipulation_scores': np.array(manipulation_scores),
            'mean_virtue': np.mean(virtue_scores),
            'min_virtue': np.min(virtue_scores),
            'max_virtue': np.max(virtue_scores),
            'mean_manipulation': np.mean(manipulation_scores),
            'max_manipulation': np.max(manipulation_scores),
            'constraint_violations': violations,
            'n_violations': len(violations),
            'is_ethical': len(violations) == 0 and np.max(manipulation_scores) < 0.5
        }

        return analysis

    def suggest_ethical_response(self, q_current: np.ndarray,
                                 candidate_responses: List[np.ndarray]) -> Tuple[int, Dict]:
        """
        Given current position and candidate next words,
        choose the most ethical response

        Args:
            q_current: Current semantic position
            candidate_responses: List of candidate next positions

        Returns:
            (best_index, scores_dict)
        """
        scores = []

        for candidate in candidate_responses:
            # Compute virtue score for candidate
            virtue = self.compute_virtue_score(candidate)

            # Check manipulation
            manipulation = self.detect_manipulation(candidate)

            # Combined score
            total_score = virtue - 2.0 * manipulation  # heavily penalize manipulation

            scores.append({
                'virtue': virtue,
                'manipulation': manipulation,
                'total': total_score
            })

        # Find best
        best_idx = max(range(len(scores)), key=lambda i: scores[i]['total'])

        return best_idx, scores[best_idx]


def visualize_ethical_landscape(policy: EthicalSemanticPolicy,
                                dim1: str, dim2: str,
                                n_points: int = 50,
                                save_path: Optional[str] = None):
    """
    Visualize virtue landscape in 2D slice of semantic space
    """
    import matplotlib.pyplot as plt

    # Get dimension indices
    idx1 = policy.dim_index[dim1]
    idx2 = policy.dim_index[dim2]

    # Create grid
    x = np.linspace(-1, 1, n_points)
    y = np.linspace(-1, 1, n_points)
    X, Y = np.meshgrid(x, y)

    # Compute virtue scores on grid
    Z_virtue = np.zeros_like(X)
    Z_manipulation = np.zeros_like(X)

    for i in range(n_points):
        for j in range(n_points):
            # Create position vector
            q = np.zeros(len(policy.dimension_names))
            q[idx1] = X[i, j]
            q[idx2] = Y[i, j]

            Z_virtue[i, j] = policy.compute_virtue_score(q)
            Z_manipulation[i, j] = policy.detect_manipulation(q)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Virtue landscape
    ax1 = axes[0]
    contour1 = ax1.contourf(X, Y, Z_virtue, levels=20, cmap='RdYlGn')
    ax1.contour(X, Y, Z_virtue, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour1, ax=ax1, label='Virtue Score')
    ax1.set_xlabel(dim1, fontsize=12)
    ax1.set_ylabel(dim2, fontsize=12)
    ax1.set_title('Ethical Virtue Landscape', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Manipulation detection
    ax2 = axes[1]
    contour2 = ax2.contourf(X, Y, Z_manipulation, levels=20, cmap='YlOrRd')
    ax2.contour(X, Y, Z_manipulation, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour2, ax=ax2, label='Manipulation Score')
    ax2.set_xlabel(dim1, fontsize=12)
    ax2.set_ylabel(dim2, fontsize=12)
    ax2.set_title('Manipulation Detection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved ethical landscape: {save_path}")

    return fig, axes