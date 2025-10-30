# How the Orchestrator Controls Semantic Calculus Exposure

## Architecture Overview

The orchestrator controls which mathematical operations are exposed through a **3-layer abstraction**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR LAYER                        │
│                  (WeavingShuttle)                            │
│                                                              │
│  Controls: Pattern selection (BARE/FAST/FUSED)              │
│            Config presets (fast/balanced/comprehensive)      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   INTEGRATION LAYER                          │
│         (SemanticAnalyzer via ResonanceShed)                │
│                                                              │
│  Exposes: .extract_features(text) → DotPlasma               │
│           .analyze_text(text) → Full analysis               │
│                                                              │
│  Controls via Config:                                       │
│    • compute_trajectory: bool (velocity/accel/curvature)   │
│    • compute_ethics: bool (ethical analysis)               │
│    • dimensions: int (8/16/32 semantic dimensions)         │
│    • enable_cache: bool (performance optimization)         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PURE MATH LAYER                           │
│              (semantic_calculus/math/)                       │
│                                                              │
│  • SemanticFlowCalculus - trajectory computation            │
│  • SemanticSpectrum - dimension projection                  │
│  • GeometricIntegrator - Hamiltonian dynamics               │
│  • EthicalSemanticPolicy - ethical evaluation               │
└─────────────────────────────────────────────────────────────┘
```

## Control Flow

### 1. Pattern Card Selection (Orchestrator)

**File:** [weaving_shuttle.py:428-440](../HoloLoom/weaving_shuttle.py#L428-L440)

```python
if pattern_spec.enable_semantic_flow:
    # Create config from pattern spec
    sem_config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)

    # Create analyzer with clean interface
    semantic_calculus = create_semantic_analyzer(embed_fn, config=sem_config)
```

**What Controls It:**
- `PatternSpec.enable_semantic_flow` - ON/OFF switch
- `PatternSpec.semantic_dimensions` - Number of dimensions (8/16/32)
- `PatternSpec.semantic_trajectory` - Include velocity/accel/curvature
- `PatternSpec.semantic_ethics` - Include ethical analysis

### 2. Configuration Presets

**File:** [semantic_calculus/config.py:60-91](../HoloLoom/semantic_calculus/config.py#L60-L91)

Three presets control feature exposure:

#### Fast Mode (Minimal)
```python
SemanticCalculusConfig.fast()
├── dimensions: 8              # Fewer dimensions
├── compute_trajectory: True   # Basic flow
└── compute_ethics: False      # SKIP ethics for speed
```

#### Balanced Mode (Default)
```python
SemanticCalculusConfig.balanced()
├── dimensions: 16             # Standard dimensions
├── compute_trajectory: True   # Full trajectory
└── compute_ethics: True       # Include ethics
```

#### Comprehensive Mode (Full)
```python
SemanticCalculusConfig.comprehensive()
├── dimensions: 32             # Maximum detail
├── compute_trajectory: True   # Full trajectory
└── compute_ethics: True       # Include ethics
```

### 3. Feature Extraction (Integration Point)

**File:** [semantic_calculus/analyzer.py:99-155](../HoloLoom/semantic_calculus/analyzer.py#L99-L155)

The `extract_features()` method **conditionally exposes** math based on config:

```python
def extract_features(self, text: str) -> Dict[str, Any]:
    """Extract features for DotPlasma integration."""

    # ALWAYS EXPOSED (core)
    trajectory = self.calculus.compute_trajectory(words)
    features = {
        'n_states': len(trajectory.states),
        'total_distance': float(trajectory.total_distance()),
    }

    # CONDITIONALLY EXPOSED (controlled by config.compute_trajectory)
    if self.config.compute_trajectory:
        features.update({
            'avg_velocity': ...,
            'max_velocity': ...,
            'avg_acceleration': ...,
            'curvature': ...,
        })

    # ALWAYS EXPOSED (spectrum analysis)
    semantic_forces = self.spectrum.analyze_semantic_forces(...)
    features['semantic_forces'] = semantic_forces

    # CONDITIONALLY EXPOSED (controlled by config.compute_ethics)
    if self.config.compute_ethics and self.policy:
        ethical_analysis = self.policy.analyze_conversation_ethics(...)
        features['ethics'] = {
            'virtue_score': ...,
            'manipulation_detected': ...,
        }

    return features
```

### 4. Factory Function Assembly

**File:** [semantic_calculus/analyzer.py:158-222](../HoloLoom/semantic_calculus/analyzer.py#L158-L222)

The factory function **conditionally creates** components:

```python
def create_semantic_analyzer(embed_fn, config):
    # ALWAYS CREATED
    calculus = SemanticFlowCalculus(...)      # Core trajectory math
    spectrum = SemanticSpectrum(...)          # Dimension projection

    # CONDITIONALLY CREATED (based on config.compute_ethics)
    integrator = None
    policy = None
    if config.compute_ethics:
        integrator = GeometricIntegrator(...)  # Hamiltonian dynamics
        policy = EthicalSemanticPolicy(...)    # Ethical evaluation

    return SemanticAnalyzer(
        calculus=calculus,
        spectrum=spectrum,
        integrator=integrator,  # May be None
        policy=policy,          # May be None
        config=config,
    )
```

## What Gets Exposed to ResonanceShed

**File:** [resonance/shed.py:218-226](../HoloLoom/resonance/shed.py#L218-L226)

ResonanceShed only calls **one method**:

```python
if isinstance(self.semantic_calculus, SemanticAnalyzer):
    # New integration layer - clean interface
    semantic_features = self.semantic_calculus.extract_features(text)
else:
    # Legacy interface (backward compatible)
    trajectory = self.semantic_calculus.compute_trajectory(words)
```

The **content of `semantic_features`** is controlled by the config, NOT by ResonanceShed.

## Exposure Control Matrix

| Math Component | Always Exposed? | Controlled By | Used For |
|----------------|----------------|---------------|----------|
| **SemanticFlowCalculus** | ✅ Yes | N/A | Core trajectory computation |
| **SemanticSpectrum** | ✅ Yes | N/A | Semantic dimension projection |
| **Velocity/Acceleration** | ⚠️ Conditional | `config.compute_trajectory` | Flow dynamics |
| **Curvature** | ⚠️ Conditional | `config.compute_trajectory` | Semantic turns |
| **GeometricIntegrator** | ⚠️ Conditional | `config.compute_ethics` | Hamiltonian dynamics |
| **EthicalSemanticPolicy** | ⚠️ Conditional | `config.compute_ethics` | Ethical evaluation |

## Example Configurations

### Scenario 1: Fast Query (BARE mode)
```python
config = SemanticCalculusConfig.fast()
# Exposes:
#   - Basic trajectory (n_states, total_distance)
#   - 8D semantic spectrum
#   - Velocity/acceleration (compute_trajectory=True)
# Skips:
#   - Ethics (compute_ethics=False)
```

### Scenario 2: Production Query (FAST mode)
```python
config = SemanticCalculusConfig.balanced()
# Exposes:
#   - Full trajectory metrics
#   - 16D semantic spectrum
#   - Velocity/acceleration/curvature
#   - Ethical analysis
```

### Scenario 3: Research Query (FUSED mode)
```python
config = SemanticCalculusConfig.comprehensive()
# Exposes:
#   - Full trajectory metrics
#   - 32D semantic spectrum (maximum detail)
#   - All dynamics features
#   - Full ethical analysis
```

## Key Design Principles

1. **Lazy Loading**: Components only created if config enables them
2. **Clean Separation**: Pure math doesn't know about orchestrator
3. **Single Entry Point**: ResonanceShed calls one method
4. **Config-Driven**: Orchestrator controls exposure via config
5. **Backward Compatible**: Old interface still works

## Summary

**The orchestrator doesn't directly "know" which math to expose.**

Instead:
1. Orchestrator creates **config** based on PatternCard
2. Config controls what **factory creates**
3. Factory creates components **conditionally**
4. SemanticAnalyzer **conditionally exposes** features
5. ResonanceShed gets **configured features** transparently

This is **dependency injection with configuration control** - the orchestrator controls *what* gets created, but doesn't need to know *how* the math works.
