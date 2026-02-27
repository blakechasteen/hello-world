# HoloLoom Expansion Bundles

**Date**: December 2025
**Status**: Production Ready

Optional research features extracted from the core config for zero-config architecture.

## Philosophy

> "Good software needs very little config."

HoloLoom's core config provides sensible defaults that work out of the box. Expansion bundles add specialized research features without cluttering the main configuration.

## Quick Start

```python
from HoloLoom.config import Config
from HoloLoom.expansions.physics import PhysicsConfig

# Start with a preset
config = Config.research()

# Load expansion bundle
config.load_expansion(PhysicsConfig(use_gp_bandits=True))

# Or use a preset from the expansion
from HoloLoom.expansions.physics import gp_thompson
config.load_expansion(gp_thompson())
```

## Available Bundles

### 1. Physics (`hololoom[physics]`)

**GP Bandits + PDE Semantic Flow** (19 fields)

Gaussian Process optimization and physics-based semantic evolution.

```python
from HoloLoom.expansions.physics import (
    PhysicsConfig,
    gp_thompson,      # Thompson Sampling acquisition
    gp_ucb,           # UCB acquisition
    semantic_heat_diffusion,  # Heat equation smoothing
    semantic_wave,    # Wave dynamics
    semantic_competition,     # Reaction-diffusion
)

# Enable GP Bandits with Thompson Sampling
config.load_expansion(gp_thompson())

# Enable semantic flow with heat diffusion
config.load_expansion(semantic_heat_diffusion())

# Full customization
config.load_expansion(PhysicsConfig(
    use_gp_bandits=True,
    gp_acquisition="thompson",
    gp_kernel_type="matern",
    gp_matern_nu=2.5,
    use_semantic_flow=True,
    pde_type="reaction_diffusion",
))
```

**GP Bandit Settings** (12 fields):
- `use_gp_bandits`: Enable GP optimization
- `gp_acquisition`: "thompson" or "ucb"
- `gp_kernel_type`: "matern" or "rbf"
- `gp_kernel_length_scale`: Smoothness (default: 0.3)
- `gp_kernel_variance`: Amplitude (default: 1.0)
- `gp_matern_nu`: Matern smoothness (1.5, 2.5, or 5.0)
- `gp_noise_variance`: Observation noise
- `gp_ucb_beta`: UCB exploration parameter
- `gp_ucb_adaptive_beta`: Use adaptive beta
- `gp_n_candidates_per_dim`: Discretization resolution
- `gp_update_interval`: Retrain frequency
- `gp_warmup_samples`: Initial random samples

**PDE Semantic Flow Settings** (7 fields):
- `use_semantic_flow`: Enable PDE evolution
- `pde_type`: "heat", "wave", "reaction_diffusion", "hamilton_jacobi"
- `flow_dt`: Timestep
- `flow_steps`: Evolution steps
- `flow_reaction_type`: "logistic", "competitive", "cubic"
- `flow_diffusion_coef`: Diffusion coefficient
- `flow_wave_speed`: Wave speed

---

### 2. Bayesian (`hololoom[bayesian]`)

**Variational Inference** (4 fields)

Bayesian uncertainty quantification for the neural policy.

```python
from HoloLoom.expansions.bayesian import (
    BayesianConfig,
    bayesian_default,
    bayesian_high_confidence,
    bayesian_regularized,
)

# Enable Bayesian policy with default settings
config.load_expansion(bayesian_default())

# High confidence (50 MC samples)
config.load_expansion(bayesian_high_confidence())

# Full customization
config.load_expansion(BayesianConfig(
    use_bayesian=True,
    bayesian_samples=20,
    bayesian_kl_weight=1.5,
    bayesian_prior_std=0.5,
))
```

**Settings**:
- `use_bayesian`: Enable Bayesian uncertainty quantification
- `bayesian_samples`: MC samples (default: 10, causes 10x inference overhead)
- `bayesian_kl_weight`: KL divergence weight in ELBO
- `bayesian_prior_std`: Prior standard deviation

---

### 3. Geometry (`hololoom[geometry]`)

**Riemannian Embeddings** (6 fields)

Non-Euclidean geometry for embeddings: hyperbolic (hierarchies), spherical (clusters).

```python
from HoloLoom.expansions.geometry import (
    GeometryConfig,
    hyperbolic_only,
    spherical_only,
    product_manifold,
    deep_hierarchy,
    tight_clusters,
)

# Product manifold (all three spaces)
config.load_expansion(product_manifold())

# Optimized for deep hierarchies (taxonomies)
config.load_expansion(deep_hierarchy())

# Full customization
config.load_expansion(GeometryConfig(
    use_riemannian=True,
    riemannian_hyperbolic_dim=512,
    riemannian_spherical_dim=128,
    riemannian_euclidean_dim=128,
    riemannian_hyperbolic_curvature=-2.0,
    riemannian_spherical_curvature=1.0,
))
```

**Settings**:
- `use_riemannian`: Enable Riemannian manifold structure
- `riemannian_hyperbolic_dim`: Dimension for hierarchical concepts (K < 0)
- `riemannian_spherical_dim`: Dimension for clustered concepts (K > 0)
- `riemannian_euclidean_dim`: Dimension for linear features (K = 0)
- `riemannian_hyperbolic_curvature`: Negative curvature (-0.1 to -10.0)
- `riemannian_spherical_curvature`: Positive curvature (0.1 to 10.0)

**When to Use**:
- Hyperbolic: Taxonomies, ontologies, tree structures
- Spherical: Topic clusters, categories, circular patterns
- Product: Mixed data with both hierarchical and clustered aspects

---

### 4. Advanced Spectral (`hololoom[spectral]`)

**Wavelets + Diffusion Maps** (8 fields)

Multi-scale spectral analysis for frequency decomposition and manifold geometry.

```python
from HoloLoom.expansions.advanced_spectral import (
    AdvancedSpectralConfig,
    wavelets_only,
    diffusion_maps_only,
    full_spectral,
    fast_spectral,
)

# Wavelet decomposition only
config.load_expansion(wavelets_only())

# Full spectral analysis
config.load_expansion(full_spectral())

# Fast spectral (reduced complexity)
config.load_expansion(fast_spectral())

# Full customization
config.load_expansion(AdvancedSpectralConfig(
    use_wavelets=True,
    wavelet_scales=[0.1, 1.0, 10.0],
    wavelet_type="mexican_hat",
    use_diffusion_maps=True,
    diffusion_map_dims=32,
    diffusion_time=1.0,
    use_multiscale_spectral=True,
))
```

**Wavelet Settings** (3 fields):
- `use_wavelets`: Enable wavelet features (O(n^3) complexity)
- `wavelet_scales`: Coarse to fine scales [0.1, 1.0, 10.0]
- `wavelet_type`: "mexican_hat", "shannon", or "meyer"

**Diffusion Map Settings** (3 fields):
- `use_diffusion_maps`: Enable diffusion geometry
- `diffusion_map_dims`: Embedding dimension
- `diffusion_time`: Time parameter (larger = more global)

**Multi-scale Spectral Settings** (2 fields):
- `use_multiscale_spectral`: Enable hierarchical analysis
- `multiscale_spectral_scales`: Match Matryoshka scales [96, 192, 384]

---

## Installation

```bash
# Core only (no expansions)
pip install hololoom

# With specific expansions
pip install hololoom[physics]
pip install hololoom[bayesian]
pip install hololoom[geometry]
pip install hololoom[spectral]

# All expansions
pip install hololoom[research]
```

## Loading Multiple Expansions

```python
from HoloLoom.config import Config
from HoloLoom.expansions.physics import gp_thompson
from HoloLoom.expansions.geometry import product_manifold
from HoloLoom.expansions.bayesian import bayesian_default

config = Config.research()
config.load_expansion(gp_thompson())
config.load_expansion(product_manifold())
config.load_expansion(bayesian_default())

# Or chain them
config = (Config.research()
    .load_expansion(gp_thompson())
    .load_expansion(product_manifold())
    .load_expansion(bayesian_default()))
```

## Creating Custom Expansions

```python
from dataclasses import dataclass
from typing import Any, Dict

try:
    from HoloLoom.config import ExpansionBundle
except ImportError:
    # Fallback for testing
    class ExpansionBundle:
        def get_settings(self) -> Dict[str, Any]:
            raise NotImplementedError

@dataclass
class MyCustomConfig(ExpansionBundle):
    """My custom research features."""

    enable_my_feature: bool = False
    my_parameter: float = 1.0

    def get_settings(self) -> Dict[str, Any]:
        return {
            "enable_my_feature": self.enable_my_feature,
            "my_parameter": self.my_parameter,
        }

# Usage
config = Config.research()
config.load_expansion(MyCustomConfig(enable_my_feature=True))
```

## Expansion Registry

Get available expansions programmatically:

```python
from HoloLoom.expansions import list_expansions, get_expansion

# List all registered expansions
expansions = list_expansions()
# ['physics', 'bayesian', 'geometry', 'advanced_spectral']

# Get an expansion class by name
PhysicsConfig = get_expansion('physics')
config.load_expansion(PhysicsConfig(use_gp_bandits=True))
```

## Performance Considerations

| Expansion | Overhead | Notes |
|-----------|----------|-------|
| **Physics (GP Bandits)** | ~10-50ms | Per decision, O(n^3) with inducing points |
| **Physics (PDE Flow)** | ~20-100ms | Per evolution step |
| **Bayesian** | 10x inference | MC sampling multiplies inference time |
| **Geometry** | ~5-20ms | Per embedding operation |
| **Advanced Spectral** | ~10-50ms | Wavelets are O(n^3) |

**Recommendations**:
- Use RESEARCH mode for experiments (no timeout constraints)
- Cache expensive computations where possible
- Profile before deploying to production

## Field Count Summary

| Bundle | Fields | Presets |
|--------|--------|---------|
| Physics | 19 | 5 |
| Bayesian | 4 | 3 |
| Geometry | 6 | 5 |
| Advanced Spectral | 8 | 4 |
| **Total** | **37** | **17** |

These 37 fields were extracted from core config, reducing visible complexity by ~30%.

## Migration from Legacy Config

If you were using research features directly on Config:

```python
# Old way (deprecated but still works)
config = Config.research()
config.use_gp_bandits = True
config.gp_acquisition = "thompson"

# New way (recommended)
from HoloLoom.expansions.physics import gp_thompson
config = Config.research()
config.load_expansion(gp_thompson())
```

Both approaches work for backward compatibility, but the expansion bundle approach is cleaner and more maintainable.

## References

- **GP Bandits**: Srinivas et al. (2010) "Gaussian Process Optimization"
- **Riemannian Embeddings**: Nickel & Kiela (2017) "Poincare Embeddings"
- **Wavelets**: Hammond et al. (2011) "Graph Wavelet Transform"
- **Diffusion Maps**: Coifman & Lafon (2006) "Diffusion Maps"
