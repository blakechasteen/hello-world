# Complete Mathematical Integration Roadmap

**Date**: 2025-11-03
**Status**: 🚧 In Progress
**Critical**: Includes Semantic Calculus Integration

---

## Overview

HoloLoom has **TWO** major mathematical systems that need deep integration:

1. **Mathematical Moonshot** (NEW) - Advanced numerical methods
2. **Semantic Calculus** (EXISTING) - 244D interpretable semantic space

Both need to work together for maximum power.

---

## Phase 0: Understanding What We Have

### Semantic Calculus (Existing System)

**Location**: `HoloLoom/semantic_calculus/`

**What It Is**:
- 244D interpretable semantic embedding space
- Based on conjugate pairs (warmth/coldness, formality/casualness, etc.)
- Uses Störmer-Verlet integration (already has symplectic integrators!)
- Projection from 384D → 244D semantic dimensions

**Key Files**:
- `dimensions.py` - Defines semantic dimensions
- `integrator.py` - **Already has Störmer-Verlet!** (line 59-72)
- `integration.py` - Geometric integration
- `dimensions.py` - EXTENDED_244_DIMENSIONS

**Critical Insight**: Semantic calculus already uses **Störmer-Verlet** (symplectic)!
```python
# From integrator.py:72
class GeometricIntegrator:
    """
    Symplectic integrator for semantic flows
    Uses Störmer-Verlet (leapfrog) method which preserves:
    - Symplectic structure: dq ∧ dp
    - Energy: H = T + V (approximately)
    """
```

### Mathematical Moonshot (New Modules)

**What We Built**:
1. Advanced ODE integrators (RK4, Verlet, RK45)
2. Riemannian geometry (curved manifolds)
3. Gaussian Process bandits
4. Advanced spectral methods (wavelets, diffusion maps)
5. Variational inference
6. PDE-based semantic flow

---

## Integration Strategy

### CRITICAL REALIZATION

Semantic Calculus **already does** some of what we built!

| Feature | Semantic Calculus | Moonshot | Integration Needed |
|---------|------------------|----------|-------------------|
| **Symplectic Integration** | ✅ Störmer-Verlet | ✅ Verlet | **MERGE**: Use moonshot's general integrator |
| **Semantic Manifold** | ✅ 244D space | ✅ Riemannian | **COMBINE**: 244D on Riemannian manifold |
| **Interpretability** | ✅ Named dimensions | ❌ Abstract | **KEEP**: Semantic calculus is better |
| **Multi-scale** | ❌ Single scale | ✅ Wavelets | **ADD**: Wavelets to semantic space |
| **Uncertainty** | ❌ Point estimates | ✅ Variational | **ADD**: Bayesian semantic projections |

---

## Priority 0: Unify Integrators (Immediate)

**Goal**: Replace semantic calculus Störmer-Verlet with moonshot's unified integrator framework

### Current State (Semantic Calculus)

```python
# semantic_calculus/integrator.py
class GeometricIntegrator:
    """Hard-coded Störmer-Verlet only"""
    def step(self, state, force_fn, dt):
        # Leapfrog implementation
        pass
```

### After Integration

```python
# semantic_calculus/integrator.py
from HoloLoom.memory.integrators import IntegratorType, create_integrator

class GeometricIntegrator:
    """Now supports RK4, Verlet, RK45, etc."""

    def __init__(self, projection_matrix, integrator_type="verlet"):
        self.integrator = create_integrator(
            IntegratorType.VERLET,  # or RK4, RK45
            force_fn=self._semantic_force_fn
        )
```

**Files to Modify**:
- `semantic_calculus/integrator.py` (line 59-150)
- `semantic_calculus/integration.py`

**Benefit**:
- Can now use RK4 for higher accuracy
- Can use RK45 for adaptive time stepping
- Unified API across all HoloLoom dynamics

---

## Priority 1: Spring Dynamics (Immediate - 2 hours)

**Status**: 70% complete
**Blocker**: Missing helper methods

### What's Needed

Add to `HoloLoom/memory/spring_dynamics.py`:

```python
def _propagate_advanced(self) -> 'SpringPropagationResult':
    """Propagate using advanced integrators."""
    # Build state from node_states
    node_list = list(self.graph.G.nodes())
    q0 = np.array([self.node_states[n].activation for n in node_list])
    p0 = np.array([self.node_states[n].velocity * self.config.mass for n in node_list])

    state = DynamicalState(q=q0, p=p0, t=0.0)

    # Integration loop
    for step in range(self.config.max_iterations):
        state = self.integrator.step(state, self.config.dt)
        # ... convergence check ...

    return result

def _compute_energy_from_state(self, state, node_list):
    """Compute energy from DynamicalState."""
    # Kinetic + Potential
    pass

class _SpringForceFunction(ForceFunction):
    """Force function for Hamiltonian dynamics."""
    def __call__(self, state):
        # Return (dq_dt, dp_dt)
        pass
```

**Test**:
```bash
python test_spring_integration.py
```

---

## Priority 2: Semantic Calculus + Riemannian Geometry (Short-term - 4 hours)

**Goal**: Put 244D semantic space on Riemannian manifold

### Why This Matters

Currently semantic calculus uses **Euclidean** 244D space:
```python
# Current: Euclidean dot product
projection = np.dot(semantic_axis, embedding)
```

But semantic space is naturally **curved** (hierarchies, clusters)!

### Integration Plan

**Step 1**: Wrap semantic embeddings in manifold
```python
# semantic_calculus/integrator.py
from HoloLoom.warp.riemannian_geometry import create_manifold, ManifoldConfig

class GeometricIntegrator:
    def __init__(self, projection_matrix, manifold_type="product"):
        self.P = projection_matrix

        # NEW: Semantic space is a Riemannian manifold
        self.manifold = create_manifold(ManifoldConfig(
            manifold_type=ManifoldType.PRODUCT,
            hyperbolic_dim=82,   # 1/3 for hierarchies
            spherical_dim=82,    # 1/3 for clusters
            euclidean_dim=80     # 1/3 for linear
        ))

    def step(self, state, force_fn, dt):
        # Use exponential map for integration on manifold
        tangent = force_fn(state)
        state.q = self.manifold.exp_map(state.q, dt * tangent)
        return state
```

**Step 2**: Replace distances
```python
# semantic_calculus/dimensions.py
from HoloLoom.warp.riemannian_geometry import HyperbolicSpace

class SemanticDimension:
    def __init__(self, name, positive_exemplars, negative_exemplars, manifold=None):
        self.manifold = manifold or HyperbolicSpace()

    def project(self, vector):
        if self.manifold:
            # Riemannian distance (geodesic)
            v_manifold = self.manifold.project(vector)
            axis_manifold = self.manifold.project(self.axis)
            return self.manifold.distance(v_manifold, axis_manifold)
        else:
            # Fallback: Euclidean
            return np.dot(vector, self.axis)
```

**Files to Modify**:
- `semantic_calculus/integrator.py`
- `semantic_calculus/dimensions.py`
- `semantic_calculus/integration.py`

**Benefit**:
- Hierarchical concepts (IS_A) → Hyperbolic distance
- Clustered concepts (synonyms) → Spherical distance
- True semantic geometry!

---

## Priority 3: Multi-Scale Semantic Calculus (Short-term - 6 hours)

**Goal**: Add wavelet analysis to 244D semantic space

### Current Limitation

Semantic calculus operates at **single scale**:
- 244D projection is fixed
- No multi-scale structure

### Integration Plan

```python
# semantic_calculus/multi_scale.py (NEW FILE)
from HoloLoom.warp.spectral_methods import SpectralWavelet, GraphLaplacian

class MultiScaleSemanticProjector:
    """
    Multi-scale semantic analysis using wavelets.

    Combines:
    - Semantic calculus: Interpretable 244D dimensions
    - Spectral methods: Multi-scale wavelet decomposition

    Result: Semantic features at multiple scales
    """

    def __init__(self, dimensions, knowledge_graph, n_scales=5):
        self.dimensions = dimensions  # 244 semantic dimensions

        # Build Laplacian from KG
        laplacian = GraphLaplacian(knowledge_graph)

        # Create wavelet transform
        self.wavelet = SpectralWavelet(laplacian, n_scales=n_scales)

    def project_multi_scale(self, embedding):
        """
        Project embedding to semantic space at multiple scales.

        Returns:
            {scale: semantic_coordinates}
        """
        multi_scale = {}

        for scale in self.wavelet.scales:
            # Apply wavelet at this scale
            wavelet_op = self.wavelet.heat_kernel(scale)
            embedding_scale = wavelet_op @ embedding

            # Project to semantic dimensions
            semantic_coords = self._project_to_semantics(embedding_scale)
            multi_scale[scale] = semantic_coords

        return multi_scale
```

**Usage**:
```python
from HoloLoom.semantic_calculus.multi_scale import MultiScaleSemanticProjector

projector = MultiScaleSemanticProjector(
    dimensions=EXTENDED_244_DIMENSIONS,
    knowledge_graph=kg,
    n_scales=5
)

# Multi-scale semantic analysis
semantic_multi_scale = projector.project_multi_scale(query_embedding)

# Scale 0.1: Local semantic features (high-frequency)
# Scale 1.0: Medium semantic features
# Scale 10.0: Global semantic features (low-frequency)
```

**Benefit**:
- Coarse-to-fine semantic analysis
- Detect local vs. global semantic patterns
- "Zoom in/out" on semantic structure

---

## Priority 4: GP-Based Semantic Dimension Learning (Medium-term - 8 hours)

**Goal**: Learn optimal semantic dimensions via Gaussian Processes

### Current Limitation

Semantic dimensions are **hand-crafted**:
```python
# semantic_calculus/dimensions.py
SemanticDimension(
    name="Warmth",
    positive_exemplars=["warm", "loving", "kind"],  # Hand-picked!
    negative_exemplars=["cold", "harsh", "cruel"]
)
```

### Integration Plan

```python
# semantic_calculus/dimension_learning.py (NEW FILE)
from HoloLoom.bandits.gaussian_process_bandits import create_gp_thompson_sampling

class AdaptiveSemanticDimensions:
    """
    Learn optimal semantic dimensions via GP optimization.

    Instead of hand-crafting dimensions, we learn them by:
    1. Start with seed dimensions (current 244)
    2. Use GP-TS to explore dimension space
    3. Optimize for interpretability + discriminative power
    """

    def __init__(self, initial_dimensions, vocabulary):
        self.dimensions = initial_dimensions
        self.vocabulary = vocabulary

        # GP-TS for dimension optimization
        # Action space: Choose exemplar words for new dimension
        candidates = self._generate_candidate_dimensions()
        self.gp_ts = create_gp_thompson_sampling(candidates)

    def optimize_dimensions(self, tasks, n_iterations=100):
        """
        Optimize dimensions for discriminative power on tasks.

        Args:
            tasks: List of classification/retrieval tasks
            n_iterations: Optimization iterations
        """
        for iteration in range(n_iterations):
            # Select dimension candidate via GP-TS
            candidate_dim, metadata = self.gp_ts.select_action()

            # Evaluate: How well does this dimension help on tasks?
            reward = self._evaluate_dimension(candidate_dim, tasks)

            # Update GP
            self.gp_ts.update(candidate_dim, reward)

            # Keep top dimensions
            if reward > threshold:
                self.dimensions.append(candidate_dim)
```

**Benefit**:
- Data-driven dimension discovery
- Task-specific semantic spaces
- Automatic improvement over time

---

## Priority 5: Variational Semantic Embeddings (Medium-term - 10 hours)

**Goal**: Bayesian uncertainty in semantic projections

### Current Limitation

Semantic projections are **deterministic**:
```python
projection = np.dot(embedding, semantic_axis)  # Point estimate
```

No uncertainty quantification!

### Integration Plan

```python
# semantic_calculus/bayesian_projector.py (NEW FILE)
from HoloLoom.warp.variational_inference import GaussianVariational, MeanFieldVI

class BayesianSemanticProjector:
    """
    Semantic projections with uncertainty.

    Instead of: projection = P @ embedding
    We get: projection ~ N(μ, σ²)

    This tells us:
    - μ: Expected semantic coordinates
    - σ: Uncertainty in projection

    High uncertainty → ambiguous semantics
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions

        # Variational distribution over projection matrix
        self.q_P = GaussianVariational(dim=244 * 384)  # P is 244 × 384

    def project_with_uncertainty(self, embedding, n_samples=100):
        """
        Project embedding to semantic space with uncertainty.

        Returns:
            (mean_projection, std_projection)
        """
        projections = []

        for _ in range(n_samples):
            # Sample projection matrix from posterior
            P_sample = self.q_P.sample(1)[0].reshape(244, 384)

            # Project
            proj = P_sample @ embedding
            projections.append(proj)

        projections = np.array(projections)

        return np.mean(projections, axis=0), np.std(projections, axis=0)
```

**Usage**:
```python
projector = BayesianSemanticProjector(EXTENDED_244_DIMENSIONS)

# Project with uncertainty
mean_coords, std_coords = projector.project_with_uncertainty(query_embedding)

# High-confidence dimensions (low std)
confident = np.where(std_coords < 0.1)[0]

# Uncertain dimensions (high std)
ambiguous = np.where(std_coords > 0.5)[0]
```

**Benefit**:
- Know when semantic interpretation is uncertain
- OOD detection (high uncertainty → out of distribution)
- Confidence-weighted decisions

---

## Priority 6: PDE-Based Semantic Flow (Long-term - 15 hours)

**Goal**: Temporal dynamics in semantic space

### Integration Plan

```python
# semantic_calculus/temporal_flow.py (NEW FILE)
from HoloLoom.warp.semantic_pde import create_heat_solver, create_reaction_diffusion_solver

class SemanticFlowDynamics:
    """
    PDE-based semantic evolution.

    Models how semantic meaning evolves over time:
    - Heat equation: Diffusion of meaning
    - Wave equation: Semantic resonance
    - Reaction-diffusion: Competing interpretations
    """

    def __init__(self, semantic_graph, dimensions):
        self.dimensions = dimensions

        # Build Laplacian in semantic space (244D)
        self.semantic_laplacian = self._build_semantic_laplacian(semantic_graph)

        # Heat solver for meaning diffusion
        self.heat_solver = create_heat_solver(self.semantic_laplacian, dt=0.01)

        # Reaction-diffusion for disambiguation
        self.reaction_diffusion = create_reaction_diffusion_solver(
            self.semantic_laplacian,
            reaction_type='competitive'  # Winner-take-all for disambiguation
        )

    def evolve_meaning(self, initial_semantic_coords, t_final=5.0):
        """
        Evolve semantic coordinates over time.

        Args:
            initial_semantic_coords: Starting point in 244D space
            t_final: How long to evolve

        Returns:
            Trajectory in semantic space
        """
        times, trajectory = self.heat_solver.solve(
            u0=initial_semantic_coords,
            t_final=t_final,
            n_snapshots=50
        )

        return times, trajectory
```

**Benefit**:
- Model meaning evolution over conversation
- Predict future semantic states
- Disambiguation via competitive dynamics

---

## Complete Integration Timeline

### Immediate (2-4 hours) ✅ Must Do First
1. **Finish Spring Dynamics**
   - Add `_propagate_advanced()`, `_compute_energy_from_state()`
   - Test Euler vs. Verlet convergence

2. **Unify Integrators**
   - Replace semantic calculus Störmer-Verlet with moonshot integrator
   - Test backward compatibility

### Short-Term (1-2 weeks)
3. **Riemannian Semantic Calculus** (4 hours)
   - Wrap 244D space in Riemannian manifold
   - Replace Euclidean distances with geodesics

4. **Multi-Scale Semantic Analysis** (6 hours)
   - Add wavelet decomposition to semantic projections
   - Coarse-to-fine semantic features

5. **GP Bandits in Policy** (4 hours)
   - Replace discrete Thompson Sampling
   - Continuous hyperparameter optimization

### Medium-Term (3-4 weeks)
6. **Adaptive Semantic Dimensions** (8 hours)
   - GP-based dimension learning
   - Task-specific semantic spaces

7. **Variational Semantic Projections** (10 hours)
   - Bayesian uncertainty in semantics
   - Confidence-weighted reasoning

8. **Advanced Spectral Features** (8 hours)
   - Diffusion maps for embeddings
   - Spectral clustering for concepts

### Long-Term (2-3 months)
9. **PDE Semantic Flow** (15 hours)
   - Temporal meaning evolution
   - Disambiguation dynamics

10. **Full System Integration** (20 hours)
    - End-to-end testing
    - Performance benchmarking
    - Production deployment

---

## Key Integration Points

### File Modifications Needed

**Semantic Calculus** (High Priority):
- `semantic_calculus/integrator.py` - Use moonshot integrators
- `semantic_calculus/dimensions.py` - Add Riemannian distances
- `semantic_calculus/integration.py` - Exponential maps on manifold
- `semantic_calculus/multi_scale.py` (NEW) - Wavelet analysis

**Embeddings**:
- `embedding/spectral.py` - Riemannian distances, wavelets

**Policy**:
- `policy/unified.py` - GP-TS for hyperparameters

**Warp**:
- `warp/space.py` - Riemannian attention

---

## Testing Strategy

### Semantic Calculus Tests

```python
# tests/integration/test_semantic_calculus_advanced.py

def test_riemannian_semantic_projection():
    """Hierarchical concepts should have small hyperbolic distance in 244D."""
    pass

def test_multi_scale_semantics():
    """Wavelets should reveal coarse and fine semantic structure."""
    pass

def test_variational_semantic_uncertainty():
    """Ambiguous queries should have high projection uncertainty."""
    pass

def test_pde_meaning_evolution():
    """Heat equation should model meaning diffusion."""
    pass
```

---

## Success Criteria

### Phase 0: Unify Integrators
- ✅ Semantic calculus uses moonshot integrators
- ✅ Backward compatible (existing code works)
- ✅ Can choose RK4, Verlet, RK45

### Phase 1: Spring Dynamics
- ✅ Verlet converges 2-4× faster than Euler
- ✅ Energy conservation: drift < 1%
- ✅ All existing tests pass

### Phase 2: Riemannian Semantics
- ✅ Hierarchical pairs (dog→mammal) have small hyperbolic distance
- ✅ Synonym clusters have small spherical distance
- ✅ Better than Euclidean on semantic tasks

### Phase 3: Multi-Scale Semantics
- ✅ Wavelet decomposition reveals local + global structure
- ✅ Coarse-to-fine semantic analysis works
- ✅ 5× faster than full 384D analysis

### Phase 4: GP Bandits
- ✅ Finds optimal hyperparameters automatically
- ✅ Sublinear regret (O(√T))
- ✅ Better than grid search

### Phase 5: Variational Semantics
- ✅ OOD detection via high uncertainty
- ✅ Calibrated confidence intervals
- ✅ Improves decision quality

### Phase 6: PDE Flow
- ✅ Models meaning evolution over time
- ✅ Disambiguation via competitive dynamics
- ✅ Predicts future semantic states

---

## Critical Path

```
Priority 0 (Immediate):
├─ Unify integrators (semantic calculus ← moonshot)
└─ Finish spring dynamics

Priority 1 (Week 1):
├─ Riemannian semantic calculus
└─ Multi-scale wavelets

Priority 2 (Week 2):
├─ GP bandits in policy
└─ Variational projections

Priority 3 (Month 1):
├─ Adaptive dimensions
└─ Advanced spectral

Priority 4 (Month 2-3):
└─ PDE semantic flow
```

---

## Estimated Total Time

**Complete Integration**: 90-120 hours
- Immediate: 4 hours
- Short-term: 20 hours
- Medium-term: 40 hours
- Long-term: 35 hours
- Testing/docs: 15 hours

---

## Next Actions

1. **RIGHT NOW**: Complete spring dynamics helper methods (2 hours)
2. **TODAY**: Unify semantic calculus integrators (2 hours)
3. **THIS WEEK**: Riemannian semantic calculus (4 hours)
4. **NEXT WEEK**: Multi-scale wavelets + GP bandits (10 hours)

🚀 **Let's start with spring dynamics completion!**
