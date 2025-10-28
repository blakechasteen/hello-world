"""
Semantic Calculus Configuration
================================
Configuration presets and settings for semantic analysis.
"""

from typing import Optional


class SemanticCalculusConfig:
    """Configuration for semantic calculus integration."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 10000,
        dimensions: int = 16,
        dt: float = 1.0,
        mass: float = 1.0,
        ethical_framework: str = "compassionate",
        compute_trajectory: bool = True,
        compute_ethics: bool = True,
        selection_strategy: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """
        Initialize semantic calculus configuration.

        Args:
            enable_cache: Enable embedding cache for performance
            cache_size: Maximum number of cached embeddings
            dimensions: Number of semantic dimensions (default 16)
            dt: Time step for calculus operations
            mass: Semantic mass parameter for dynamics
            ethical_framework: "compassionate", "scientific", or "therapeutic"
            compute_trajectory: Compute velocity/acceleration/curvature
            compute_ethics: Run ethical analysis
            selection_strategy: Strategy for dimension selection ("balanced", "narrative", "dialogue", "hybrid")
            domain: Domain context for dimension selection ("narrative", "dialogue", "technical", "general")
        """
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.dimensions = dimensions
        self.dt = dt
        self.mass = mass
        self.ethical_framework = ethical_framework
        self.compute_trajectory = compute_trajectory
        self.compute_ethics = compute_ethics
        self.selection_strategy = selection_strategy
        self.domain = domain

    @classmethod
    def from_pattern_spec(cls, pattern_spec) -> 'SemanticCalculusConfig':
        """Create config from pattern card specification."""
        return cls(
            enable_cache=True,  # Always enable cache for performance
            cache_size=10000,
            dimensions=getattr(pattern_spec, 'semantic_dimensions', 16),
            dt=1.0,
            mass=1.0,
            ethical_framework="compassionate",
            compute_trajectory=getattr(pattern_spec, 'semantic_trajectory', True),
            compute_ethics=getattr(pattern_spec, 'semantic_ethics', True),
        )

    @classmethod
    def fast(cls) -> 'SemanticCalculusConfig':
        """Fast configuration (minimal features)."""
        return cls(
            enable_cache=True,
            cache_size=5000,
            dimensions=8,  # Fewer dimensions
            compute_trajectory=True,
            compute_ethics=False,  # Skip ethics for speed
        )

    @classmethod
    def balanced(cls) -> 'SemanticCalculusConfig':
        """Balanced configuration (default)."""
        return cls(
            enable_cache=True,
            cache_size=10000,
            dimensions=16,
            compute_trajectory=True,
            compute_ethics=True,
        )

    @classmethod
    def comprehensive(cls) -> 'SemanticCalculusConfig':
        """Comprehensive configuration (all features)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=32,  # More dimensions for detail
            compute_trajectory=True,
            compute_ethics=True,
        )

    @classmethod
    def research(cls) -> 'SemanticCalculusConfig':
        """Research configuration (244D full narrative analysis)."""
        return cls(
            enable_cache=True,
            cache_size=50000,
            dimensions=244,  # Full 244-dimensional semantic space
            compute_trajectory=True,
            compute_ethics=True,
        )

    @classmethod
    def fused_narrative(cls) -> 'SemanticCalculusConfig':
        """FUSED mode optimized for narrative/literary analysis (36D smart selection)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=36,
            compute_trajectory=True,
            compute_ethics=True,
            selection_strategy='narrative',
            domain='narrative',
        )

    @classmethod
    def fused_dialogue(cls) -> 'SemanticCalculusConfig':
        """FUSED mode optimized for dialogue/conversation analysis (36D smart selection)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=36,
            compute_trajectory=True,
            compute_ethics=True,
            selection_strategy='dialogue',
            domain='dialogue',
        )

    @classmethod
    def fused_technical(cls) -> 'SemanticCalculusConfig':
        """FUSED mode optimized for technical/documentation analysis (36D smart selection)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=36,
            compute_trajectory=True,
            compute_ethics=True,
            selection_strategy='balanced',
            domain='technical',
        )

    @classmethod
    def fused_general(cls) -> 'SemanticCalculusConfig':
        """FUSED mode for general-purpose analysis (36D hybrid selection)."""
        return cls(
            enable_cache=True,
            cache_size=20000,
            dimensions=36,
            compute_trajectory=True,
            compute_ethics=True,
            selection_strategy='hybrid',
            domain='general',
        )

    def __repr__(self) -> str:
        return (
            f"SemanticCalculusConfig("
            f"dimensions={self.dimensions}, "
            f"cache={'enabled' if self.enable_cache else 'disabled'}, "
            f"trajectory={self.compute_trajectory}, "
            f"ethics={self.compute_ethics})"
        )
