"""
Semantic Dimensions: Interpretable axes in embedding space

Defines conjugate pairs of semantic dimensions (warmth/coldness, formality/casualness, etc.)
and provides tools to project trajectories onto these dimensions to understand
WHAT is changing in semantic flow.

This is the key to interpretability: instead of tracking raw 384D vectors,
we track motion along meaningful human-interpretable axes.

Based on the insight that semantic space has natural axes that can be learned
from exemplar words at the poles of each dimension.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SemanticDimension:
    """
    A single interpretable dimension in semantic space

    Defined by exemplar words at positive and negative poles.
    For example:
    - Warmth: positive=["warm", "loving", "kind"], negative=["cold", "harsh", "cruel"]
    - Formality: positive=["formal", "professional"], negative=["casual", "colloquial"]
    """
    name: str
    positive_exemplars: List[str]
    negative_exemplars: List[str]
    axis: Optional[np.ndarray] = None  # learned direction vector

    def learn_axis(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Learn the axis direction from exemplars

        Method: Compute centroids of positive and negative exemplars,
        axis is the normalized difference vector.
        """
        # Embed all exemplars
        pos_embeddings = np.array([embed_fn(word) for word in self.positive_exemplars])
        neg_embeddings = np.array([embed_fn(word) for word in self.negative_exemplars])

        # Compute centroids
        pos_centroid = np.mean(pos_embeddings, axis=0)
        neg_centroid = np.mean(neg_embeddings, axis=0)

        # Axis = normalized difference
        self.axis = pos_centroid - neg_centroid
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-10)

        return self.axis

    def project(self, vector: np.ndarray) -> float:
        """
        Project a vector onto this dimension

        Returns scalar indicating position along dimension:
        - Positive values = toward positive pole
        - Negative values = toward negative pole
        """
        if self.axis is None:
            raise ValueError(f"Dimension '{self.name}' axis not learned yet. Call learn_axis() first.")

        return np.dot(vector, self.axis)


# Predefined semantic dimensions
# These are the "conjugate pairs" that form an interpretable basis
STANDARD_DIMENSIONS = [
    # Affective dimensions
    SemanticDimension(
        name="Warmth",
        positive_exemplars=["warm", "loving", "kind", "affectionate", "caring", "tender"],
        negative_exemplars=["cold", "harsh", "cruel", "hostile", "uncaring", "callous"]
    ),
    SemanticDimension(
        name="Valence",
        positive_exemplars=["positive", "good", "happy", "pleasant", "joyful", "delightful"],
        negative_exemplars=["negative", "bad", "sad", "unpleasant", "miserable", "awful"]
    ),
    SemanticDimension(
        name="Arousal",
        positive_exemplars=["excited", "energetic", "intense", "passionate", "thrilling"],
        negative_exemplars=["calm", "peaceful", "relaxed", "serene", "tranquil"]
    ),
    SemanticDimension(
        name="Intensity",
        positive_exemplars=["intense", "extreme", "powerful", "overwhelming", "fierce"],
        negative_exemplars=["mild", "gentle", "subtle", "moderate", "faint"]
    ),

    # Social/interpersonal dimensions
    SemanticDimension(
        name="Formality",
        positive_exemplars=["formal", "professional", "official", "proper", "ceremonial"],
        negative_exemplars=["casual", "informal", "colloquial", "relaxed", "friendly"]
    ),
    SemanticDimension(
        name="Directness",
        positive_exemplars=["direct", "explicit", "clear", "straightforward", "blunt"],
        negative_exemplars=["indirect", "implicit", "vague", "subtle", "evasive"]
    ),
    SemanticDimension(
        name="Power",
        positive_exemplars=["dominant", "authoritative", "commanding", "powerful", "controlling"],
        negative_exemplars=["submissive", "passive", "powerless", "weak", "yielding"]
    ),
    SemanticDimension(
        name="Generosity",
        positive_exemplars=["generous", "giving", "selfless", "charitable", "magnanimous"],
        negative_exemplars=["selfish", "greedy", "stingy", "miserly", "uncharitable"]
    ),

    # Cognitive dimensions
    SemanticDimension(
        name="Certainty",
        positive_exemplars=["certain", "sure", "definite", "confident", "convinced"],
        negative_exemplars=["uncertain", "unsure", "doubtful", "hesitant", "ambiguous"]
    ),
    SemanticDimension(
        name="Complexity",
        positive_exemplars=["complex", "complicated", "intricate", "sophisticated", "elaborate"],
        negative_exemplars=["simple", "basic", "straightforward", "elementary", "plain"]
    ),
    SemanticDimension(
        name="Concreteness",
        positive_exemplars=["concrete", "tangible", "physical", "specific", "material"],
        negative_exemplars=["abstract", "intangible", "theoretical", "conceptual", "general"]
    ),
    SemanticDimension(
        name="Familiarity",
        positive_exemplars=["familiar", "known", "common", "usual", "ordinary"],
        negative_exemplars=["novel", "unfamiliar", "strange", "unusual", "exotic"]
    ),

    # Temporal/dynamic dimensions
    SemanticDimension(
        name="Agency",
        positive_exemplars=["active", "doing", "acting", "causing", "initiating"],
        negative_exemplars=["passive", "receiving", "experiencing", "affected", "undergoing"]
    ),
    SemanticDimension(
        name="Stability",
        positive_exemplars=["stable", "constant", "steady", "unchanging", "fixed"],
        negative_exemplars=["volatile", "changing", "unstable", "fluctuating", "dynamic"]
    ),
    SemanticDimension(
        name="Urgency",
        positive_exemplars=["urgent", "immediate", "pressing", "critical", "emergency"],
        negative_exemplars=["patient", "gradual", "leisurely", "relaxed", "unhurried"]
    ),
    SemanticDimension(
        name="Completion",
        positive_exemplars=["complete", "finished", "final", "concluded", "ending"],
        negative_exemplars=["incomplete", "starting", "beginning", "initial", "nascent"]
    ),
]


class SemanticSpectrum:
    """
    Projects semantic trajectories onto interpretable dimensions

    This is the key to understanding WHAT is changing in a conversation.
    Instead of 384 opaque numbers, we get "warmth increasing, formality decreasing"
    """

    def __init__(self, dimensions: Optional[List[SemanticDimension]] = None):
        """
        Args:
            dimensions: List of semantic dimensions. If None, uses STANDARD_DIMENSIONS
        """
        self.dimensions = dimensions if dimensions is not None else STANDARD_DIMENSIONS
        self._axes_learned = False

    def learn_axes(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Learn all dimension axes from exemplars

        Args:
            embed_fn: Function that maps word -> embedding vector
        """
        print(f"Learning {len(self.dimensions)} semantic dimension axes...")
        for dim in self.dimensions:
            dim.learn_axis(embed_fn)
        self._axes_learned = True
        print(f"  All axes learned successfully")

    def project_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Project a single vector onto all dimensions

        Returns:
            Dictionary mapping dimension name -> projection value
        """
        if not self._axes_learned:
            raise ValueError("Axes not learned yet. Call learn_axes() first.")

        return {dim.name: dim.project(vector) for dim in self.dimensions}

    def project_trajectory(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Project entire trajectory onto all dimensions

        Args:
            positions: Array of shape (n_steps, embedding_dim)

        Returns:
            Dictionary mapping dimension name -> array of projections over time
        """
        if not self._axes_learned:
            raise ValueError("Axes not learned yet. Call learn_axes() first.")

        projections = {}
        for dim in self.dimensions:
            projections[dim.name] = np.array([dim.project(pos) for pos in positions])

        return projections

    def compute_spectrum_velocity(self, positions: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute velocity along each semantic dimension

        This tells us HOW FAST each dimension is changing
        """
        projections = self.project_trajectory(positions)

        velocities = {}
        for dim_name, proj in projections.items():
            velocities[dim_name] = np.gradient(proj, dt)

        return velocities

    def compute_spectrum_acceleration(self, positions: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute acceleration along each semantic dimension

        This tells us which dimensions have FORCES acting on them
        """
        velocities = self.compute_spectrum_velocity(positions, dt)

        accelerations = {}
        for dim_name, vel in velocities.items():
            accelerations[dim_name] = np.gradient(vel, dt)

        return accelerations

    def get_dominant_dimensions(self, velocity_dict: Dict[str, np.ndarray],
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find which dimensions are changing most rapidly

        Args:
            velocity_dict: Output of compute_spectrum_velocity()
            top_k: How many top dimensions to return

        Returns:
            List of (dimension_name, avg_velocity_magnitude) sorted by magnitude
        """
        avg_magnitudes = {
            name: np.mean(np.abs(vel))
            for name, vel in velocity_dict.items()
        }

        sorted_dims = sorted(avg_magnitudes.items(), key=lambda x: x[1], reverse=True)
        return sorted_dims[:top_k]

    def analyze_semantic_forces(self, positions: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Complete analysis: which dimensions are being pushed/pulled

        Returns rich analysis of semantic forces acting on the trajectory
        """
        projections = self.project_trajectory(positions)
        velocities = self.compute_spectrum_velocity(positions, dt)
        accelerations = self.compute_spectrum_acceleration(positions, dt)

        # Find dominant dimensions
        dominant_by_velocity = self.get_dominant_dimensions(velocities, top_k=5)

        # Find dimensions with strongest forces (acceleration)
        dominant_by_force = self.get_dominant_dimensions(accelerations, top_k=5)

        # Compute total semantic distance along each dimension
        distances = {
            name: np.sum(np.abs(np.diff(proj)))
            for name, proj in projections.items()
        }

        return {
            'projections': projections,
            'velocities': velocities,
            'accelerations': accelerations,
            'dominant_velocity': dominant_by_velocity,
            'dominant_force': dominant_by_force,
            'distances': distances
        }


def visualize_semantic_spectrum(analysis: Dict, words: List[str],
                                top_k: int = 8, save_path: Optional[str] = None):
    """
    Visualize how semantic dimensions change over a trajectory

    Shows the top K most active dimensions and how they evolve
    """
    import matplotlib.pyplot as plt

    # Get top K dimensions by total distance traveled
    distances = analysis['distances']
    top_dims = sorted(distances.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_dim_names = [name for name, _ in top_dims]

    projections = analysis['projections']
    velocities = analysis['velocities']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Projections (position along each dimension)
    ax1 = axes[0]
    for dim_name in top_dim_names:
        ax1.plot(projections[dim_name], label=dim_name, linewidth=2, alpha=0.7)
    ax1.set_ylabel('Projection Value', fontsize=12)
    ax1.set_title('Semantic Dimension Positions', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Velocities (rate of change)
    ax2 = axes[1]
    for dim_name in top_dim_names:
        ax2.plot(velocities[dim_name], label=dim_name, linewidth=2, alpha=0.7)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Semantic Dimension Velocities', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Heatmap of all dimensions
    ax3 = axes[2]
    all_velocities = np.array([velocities[name] for name in top_dim_names])
    im = ax3.imshow(all_velocities, aspect='auto', cmap='RdBu_r',
                    interpolation='nearest', vmin=-0.5, vmax=0.5)
    ax3.set_yticks(range(len(top_dim_names)))
    ax3.set_yticklabels(top_dim_names)
    ax3.set_xlabel('Word Index', fontsize=12)
    ax3.set_title('Semantic Velocity Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Velocity')

    # Add word labels if not too many
    if len(words) <= 20:
        for ax in [ax1, ax2]:
            ax.set_xticks(range(len(words)))
            ax.set_xticklabels(words, rotation=45, ha='right')
        ax3.set_xticks(range(len(words)))
        ax3.set_xticklabels(words, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved spectrum visualization: {save_path}")

    return fig, axes


def print_spectrum_summary(analysis: Dict, words: List[str]):
    """
    Print human-readable summary of semantic forces
    """
    print("\n" + "=" * 70)
    print("SEMANTIC SPECTRUM ANALYSIS")
    print("=" * 70)

    print(f"\nTop 5 dimensions by velocity (how fast they're changing):")
    for i, (name, mag) in enumerate(analysis['dominant_velocity'], 1):
        print(f"  {i}. {name:<15} (avg velocity: {mag:.4f})")

    print(f"\nTop 5 dimensions by force (acceleration):")
    for i, (name, mag) in enumerate(analysis['dominant_force'], 1):
        print(f"  {i}. {name:<15} (avg force: {mag:.4f})")

    print(f"\nTop 5 dimensions by distance traveled:")
    distances = sorted(analysis['distances'].items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (name, dist) in enumerate(distances, 1):
        print(f"  {i}. {name:<15} (total distance: {dist:.4f})")

    # Show start and end values for dominant dimensions
    print(f"\nStart -> End values for top 3 dimensions:")
    projections = analysis['projections']
    for i, (name, _) in enumerate(analysis['dominant_velocity'][:3], 1):
        start_val = projections[name][0]
        end_val = projections[name][-1]
        change = end_val - start_val
        direction = "UP" if change > 0 else "DOWN"
        print(f"  {i}. {name:<15}: {start_val:>7.3f} -> {end_val:>7.3f}  {direction} ({abs(change):.3f})")