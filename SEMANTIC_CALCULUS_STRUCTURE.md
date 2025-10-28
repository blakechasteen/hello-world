# Semantic Calculus - Proposed Reorganization

## Current Structure (Flat)
```
HoloLoom/semantic_calculus/
├── flow_calculus.py          # Math
├── dimensions.py              # Math
├── integrator.py              # Math
├── ethics.py                  # Math
├── hyperbolic.py              # Math
├── integral_geometry.py       # Math
├── system_id.py               # Math
├── performance.py             # Utilities
├── integration.py             # HoloLoom adapter
├── mcp_server.py              # External tool
└── __init__.py
```

**Issue**: Math and integration mixed together in flat structure.

## Proposed Structure (Organized)

### Option A: Separate Math Subdirectory
```
HoloLoom/semantic_calculus/
├── math/                      # Pure mathematics (independent)
│   ├── __init__.py
│   ├── flow.py               # Trajectory, velocity, acceleration
│   ├── dimensions.py         # 16D semantic spectrum
│   ├── dynamics.py           # Hamiltonian integration
│   ├── ethics.py             # Multi-objective optimization
│   ├── hyperbolic.py         # Poincaré ball geometry
│   ├── tomography.py         # Integral geometry
│   └── system_id.py          # System identification
│
├── core/                      # Core integration (HoloLoom-specific)
│   ├── __init__.py
│   ├── config.py             # SemanticCalculusConfig
│   ├── analyzer.py           # SemanticAnalyzer
│   ├── adapter.py            # ResonanceShed adapter
│   └── cache.py              # Performance utilities
│
├── tools/                     # External tools
│   ├── __init__.py
│   └── mcp_server.py         # Claude Desktop MCP server
│
└── __init__.py                # Main exports
```

**Benefits**:
- ✅ Clear separation: `math/` is pure, reusable mathematics
- ✅ `core/` is HoloLoom-specific integration
- ✅ `tools/` for external interfaces
- ✅ Math can be used standalone or in other projects
- ✅ Clean imports: `from semantic_calculus.math import flow`

### Option B: Math at Top Level (Following HoloLoom Pattern)
```
HoloLoom/
├── semantic_calculus/         # Application integration
│   ├── __init__.py
│   ├── config.py             # Configuration
│   ├── analyzer.py           # SemanticAnalyzer
│   ├── adapter.py            # Adapters
│   └── mcp_server.py         # MCP tools
│
├── semantic_math/             # Pure mathematics (NEW!)
│   ├── __init__.py
│   ├── flow.py               # Calculus on manifolds
│   ├── dimensions.py         # Spectral projection
│   ├── dynamics.py           # Hamiltonian mechanics
│   ├── ethics.py             # Convex optimization
│   ├── hyperbolic.py         # Non-Euclidean geometry
│   ├── tomography.py         # Radon transforms
│   ├── system_id.py          # Dynamical systems
│   └── performance.py        # Numerical utilities
│
├── config.py                  # Global config (already has semantic fields)
├── weaving_shuttle.py
└── ...
```

**Benefits**:
- ✅ Follows HoloLoom's top-level module pattern
- ✅ Math is completely independent module
- ✅ Can be published as separate package: `pip install semantic-math`
- ✅ Clean imports: `from HoloLoom.semantic_math import flow`
- ✅ Semantic calculus focuses on integration only

### Option C: Keep Current + Add Math Subdirectory
```
HoloLoom/semantic_calculus/
├── math/                      # Pure mathematics
│   ├── flow.py               # Core calculus (move from flow_calculus.py)
│   ├── spectrum.py           # 16D projection (move from dimensions.py)
│   ├── dynamics.py           # Integration (move from integrator.py)
│   ├── optimization.py       # Ethics (move from ethics.py)
│   ├── hyperbolic.py         # Keep
│   ├── tomography.py         # Rename from integral_geometry.py
│   └── system_id.py          # Keep
│
├── config.py                  # Move from integration.py
├── analyzer.py                # Move from integration.py
├── adapter.py                 # New - ResonanceShed helpers
├── performance.py             # Keep - caching utilities
├── mcp_server.py              # Keep - external tool
└── __init__.py                # Update exports
```

**Benefits**:
- ✅ Minimal disruption to existing code
- ✅ Clear math separation
- ✅ Integration code at top level
- ✅ Easier migration

## Recommended: Option C (Minimal Disruption)

### File Reorganization

1. **Create `math/` subdirectory**:
   ```
   mkdir HoloLoom/semantic_calculus/math
   ```

2. **Move/rename math files**:
   ```
   flow_calculus.py → math/flow.py
   dimensions.py → math/spectrum.py
   integrator.py → math/dynamics.py
   ethics.py → math/optimization.py
   hyperbolic.py → math/hyperbolic.py
   integral_geometry.py → math/tomography.py
   system_id.py → math/system_id.py
   ```

3. **Split `integration.py`**:
   ```
   integration.py:SemanticCalculusConfig → config.py
   integration.py:SemanticAnalyzer → analyzer.py
   integration.py:create_semantic_thread → adapter.py
   ```

4. **Final structure**:
   ```
   HoloLoom/semantic_calculus/
   ├── math/
   │   ├── __init__.py
   │   ├── flow.py
   │   ├── spectrum.py
   │   ├── dynamics.py
   │   ├── optimization.py
   │   ├── hyperbolic.py
   │   ├── tomography.py
   │   └── system_id.py
   ├── config.py              # Configuration
   ├── analyzer.py            # Main analyzer
   ├── adapter.py             # HoloLoom adapters
   ├── performance.py         # Caching
   ├── mcp_server.py          # MCP tools
   └── __init__.py
   ```

### Updated Imports

**Old**:
```python
from HoloLoom.semantic_calculus import SemanticFlowCalculus
from HoloLoom.semantic_calculus import SemanticSpectrum
from HoloLoom.semantic_calculus import GeometricIntegrator
```

**New (Clean)**:
```python
# Pure math
from HoloLoom.semantic_calculus.math import Flow, Spectrum, Dynamics

# Integration
from HoloLoom.semantic_calculus import SemanticAnalyzer, SemanticCalculusConfig

# Or convenience (re-exported in __init__.py)
from HoloLoom.semantic_calculus import create_semantic_analyzer
```

### Benefits of This Approach

1. **Clear Separation**: Math is isolated in `math/` subdirectory
2. **Backward Compatible**: Old imports still work via `__init__.py` re-exports
3. **Cleaner Names**: `flow.py` instead of `flow_calculus.py`
4. **Minimal Disruption**: Existing code continues to work
5. **Future-Proof**: Math can be extracted to separate package later

## Implementation Checklist

- [ ] Create `HoloLoom/semantic_calculus/math/` directory
- [ ] Move core math files to `math/` subdirectory
- [ ] Create `math/__init__.py` with clean exports
- [ ] Split `integration.py` into `config.py`, `analyzer.py`, `adapter.py`
- [ ] Update main `__init__.py` to re-export for backward compatibility
- [ ] Update imports in dependent files (ResonanceShed, WeavingShuttle)
- [ ] Update tests
- [ ] Update documentation

## Example: Math Module Exports

**`HoloLoom/semantic_calculus/math/__init__.py`**:
```python
"""
Pure mathematical operations for semantic calculus.

This module is independent of HoloLoom and can be used standalone.
All operations are on embedding manifolds with geometric structure.
"""

from .flow import (
    SemanticState,
    SemanticTrajectory,
    SemanticFlow,  # Renamed from SemanticFlowCalculus
)

from .spectrum import (
    SemanticDimension,
    SemanticSpectrum,
    STANDARD_DIMENSIONS,
)

from .dynamics import (
    HamiltonianDynamics,  # Renamed from GeometricIntegrator
    MultiScaleFlow,       # Renamed from MultiScaleGeometricFlow
)

from .optimization import (
    EthicalObjective,
    EthicalPolicy,  # Renamed from EthicalSemanticPolicy
    COMPASSIONATE_COMMUNICATION,
    SCIENTIFIC_DISCOURSE,
    THERAPEUTIC_DIALOGUE,
)

__all__ = [
    "SemanticFlow",
    "SemanticSpectrum",
    "HamiltonianDynamics",
    "EthicalPolicy",
    # ... more
]
```

**`HoloLoom/semantic_calculus/__init__.py`** (Main):
```python
"""
Semantic calculus integration for HoloLoom.

Quick Start:
    from HoloLoom.semantic_calculus import create_semantic_analyzer

    analyzer = create_semantic_analyzer(embed_fn)
    result = analyzer.analyze_text("Your text...")
"""

# Re-export math for backward compatibility
from .math import (
    SemanticFlow as SemanticFlowCalculus,  # Alias for backward compat
    SemanticSpectrum,
    HamiltonianDynamics as GeometricIntegrator,  # Alias
    EthicalPolicy as EthicalSemanticPolicy,  # Alias
    # ... more
)

# Export integration layer
from .config import SemanticCalculusConfig
from .analyzer import SemanticAnalyzer, create_semantic_analyzer
from .adapter import create_semantic_thread, quick_analysis
from .performance import EmbeddingCache, get_cache_stats

__all__ = [
    # Integration (primary interface)
    "create_semantic_analyzer",
    "SemanticAnalyzer",
    "SemanticCalculusConfig",

    # Math (backward compatible aliases)
    "SemanticFlowCalculus",
    "SemanticSpectrum",
    "GeometricIntegrator",
    "EthicalSemanticPolicy",

    # Performance
    "EmbeddingCache",
    "get_cache_stats",
]
```

Would you like me to implement this reorganization? It will:
1. Create clean `math/` subdirectory with pure mathematics
2. Split integration code into focused files
3. Maintain complete backward compatibility
4. Make the codebase much cleaner and more maintainable
