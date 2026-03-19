"""
Advanced Spectral Expansion Bundle - Wavelets, Diffusion Maps
=============================================================

Multi-scale spectral analysis: wavelets for frequency decomposition,
diffusion maps for manifold geometry.

Install: pip install hololoom[spectral]

Usage:
    from hololoom.config import Config
    from hololoom.expansions.advanced_spectral import AdvancedSpectralConfig

    config = Config.research()
    config.load_expansion(AdvancedSpectralConfig(use_wavelets=True))

Date: December 2025
"""

from dataclasses import dataclass, field
from typing import Any

try:
    from hololoom.config import ExpansionBundle
except ImportError:
    # Fallback for testing
    class ExpansionBundle:
        def get_settings(self) -> dict[str, Any]:
            raise NotImplementedError


@dataclass
class AdvancedSpectralConfig(ExpansionBundle):
    """
    Advanced Spectral Expansion Bundle: Wavelets + Diffusion Maps.

    Multi-scale frequency analysis and manifold geometry:

    Wavelets:
    - Multi-scale decomposition (coarse to fine)
    - Graph wavelet transform on knowledge graph
    - O(n^3) complexity, use sparingly

    Diffusion Maps:
    - Nonlinear dimensionality reduction
    - Captures manifold geometry
    - Kernel-based diffusion process

    Multi-scale Spectral:
    - Hierarchical spectral analysis
    - Matches Matryoshka embedding scales
    - Good for multi-resolution reasoning
    """

    # =========================================================================
    # WAVELET SETTINGS (3 fields)
    # =========================================================================

    use_wavelets: bool = False
    """Enable multi-scale wavelet features (adds ~10-50ms, O(n^3) complexity)."""

    wavelet_scales: list[float] = field(
        default_factory=lambda: [0.1, 1.0, 10.0]
    )
    """
    Wavelet scales (coarse to fine):
    - 0.1: Fine-grained local patterns
    - 1.0: Medium-scale structures
    - 10.0: Coarse global patterns
    """

    wavelet_type: str = "mexican_hat"
    """Wavelet type: 'mexican_hat', 'shannon', or 'meyer'."""

    # =========================================================================
    # DIFFUSION MAP SETTINGS (3 fields)
    # =========================================================================

    use_diffusion_maps: bool = False
    """Enable diffusion geometry (adds ~20-100ms, cached)."""

    diffusion_map_dims: int = 32
    """Diffusion embedding dimension."""

    diffusion_time: float = 1.0
    """Diffusion time parameter (larger = more global structure)."""

    # =========================================================================
    # MULTI-SCALE SPECTRAL SETTINGS (2 fields)
    # =========================================================================

    use_multiscale_spectral: bool = False
    """Enable hierarchical spectral analysis (experimental)."""

    multiscale_spectral_scales: list[int] = field(
        default_factory=lambda: [96, 192, 384]
    )
    """Spectral scales (matches Matryoshka embedding scales)."""

    def get_settings(self) -> dict[str, Any]:
        """Return dictionary of settings to merge into config."""
        return {
            # Wavelets
            "use_wavelets": self.use_wavelets,
            "wavelet_scales": self.wavelet_scales,
            "wavelet_type": self.wavelet_type,

            # Diffusion Maps
            "use_diffusion_maps": self.use_diffusion_maps,
            "diffusion_map_dims": self.diffusion_map_dims,
            "diffusion_time": self.diffusion_time,

            # Multi-scale Spectral
            "use_multiscale_spectral": self.use_multiscale_spectral,
            "multiscale_spectral_scales": self.multiscale_spectral_scales,
        }


# Convenience presets
def wavelets_only() -> AdvancedSpectralConfig:
    """Wavelet decomposition only."""
    return AdvancedSpectralConfig(
        use_wavelets=True,
        wavelet_scales=[0.1, 1.0, 10.0],
        wavelet_type="mexican_hat",
    )


def diffusion_maps_only() -> AdvancedSpectralConfig:
    """Diffusion geometry only."""
    return AdvancedSpectralConfig(
        use_diffusion_maps=True,
        diffusion_map_dims=32,
        diffusion_time=1.0,
    )


def full_spectral() -> AdvancedSpectralConfig:
    """Full spectral analysis (all features)."""
    return AdvancedSpectralConfig(
        use_wavelets=True,
        wavelet_scales=[0.1, 1.0, 10.0],
        wavelet_type="mexican_hat",
        use_diffusion_maps=True,
        diffusion_map_dims=32,
        diffusion_time=1.0,
        use_multiscale_spectral=True,
        multiscale_spectral_scales=[96, 192, 384],
    )


def fast_spectral() -> AdvancedSpectralConfig:
    """Fast spectral analysis (reduced complexity)."""
    return AdvancedSpectralConfig(
        use_wavelets=True,
        wavelet_scales=[1.0],  # Single scale
        wavelet_type="mexican_hat",
        use_diffusion_maps=True,
        diffusion_map_dims=16,  # Reduced dimensions
        diffusion_time=0.5,
    )


__all__ = [
    "AdvancedSpectralConfig",
    "wavelets_only",
    "diffusion_maps_only",
    "full_spectral",
    "fast_spectral",
]
