"""
HoloLoom Expansion Bundles
==========================

Optional research features for HoloLoom. Install via pip extras:

    pip install hololoom[research]     # All research features
    pip install hololoom[physics]      # GP Bandits + PDE Flow
    pip install hololoom[bayesian]     # Variational inference
    pip install hololoom[geometry]     # Riemannian embeddings

Usage:
    from hololoom.config import Config
    from hololoom.expansions.physics import PhysicsConfig

    # Method 1: Load after construction
    config = Config.research()
    config.load_expansion(PhysicsConfig(use_gp_bandits=True))

    # Method 2: Load multiple expansions
    config = Config.research()
    config.load_expansion(PhysicsConfig())
    config.load_expansion(GeometryConfig())

Date: December 2025
"""

from typing import List, Type

# Lazy imports to avoid loading unused expansions
_EXPANSION_REGISTRY: dict = {}


def register_expansion(name: str, expansion_class: type) -> None:
    """Register an expansion bundle."""
    _EXPANSION_REGISTRY[name] = expansion_class


def get_expansion(name: str):
    """Get an expansion bundle by name."""
    if name not in _EXPANSION_REGISTRY:
        # Try to import the module
        if name == "physics":
            from hololoom.expansions.physics import PhysicsConfig
            _EXPANSION_REGISTRY["physics"] = PhysicsConfig
        elif name == "bayesian":
            from hololoom.expansions.bayesian import BayesianConfig
            _EXPANSION_REGISTRY["bayesian"] = BayesianConfig
        elif name == "geometry":
            from hololoom.expansions.geometry import GeometryConfig
            _EXPANSION_REGISTRY["geometry"] = GeometryConfig
        elif name == "advanced_spectral":
            from hololoom.expansions.advanced_spectral import AdvancedSpectralConfig
            _EXPANSION_REGISTRY["advanced_spectral"] = AdvancedSpectralConfig
        else:
            raise ValueError(f"Unknown expansion: {name}")

    return _EXPANSION_REGISTRY[name]


def list_expansions() -> list[str]:
    """List all available expansion bundles."""
    return ["physics", "bayesian", "geometry", "advanced_spectral"]


def load_all_expansions(config) -> None:
    """Load all available expansion bundles into a config (research mode)."""
    for name in list_expansions():
        try:
            expansion_class = get_expansion(name)
            config.load_expansion(expansion_class())
        except ImportError:
            pass  # Skip if dependencies not installed


__all__ = [
    "register_expansion",
    "get_expansion",
    "list_expansions",
    "load_all_expansions",
]
