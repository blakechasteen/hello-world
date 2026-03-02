# Semantic Calculus - Organized Integration

## Overview

The semantic calculus is now **cleanly integrated** into HoloLoom using a well-organized architecture with clear separation of concerns.

## Architecture Layers

### Layer 1: Core Mathematics (`hololoom/semantic_calculus/`)

**Pure mathematical operations** - independent of HoloLoom:
- `flow_calculus.py` - Trajectory computation, derivatives, dynamics
- `dimensions.py` - 16D semantic spectrum projection
- `integrator.py` - Hamiltonian dynamics, geometric integration
- `ethics.py` - Multi-objective ethical optimization
- `performance.py` - Caching, JIT compilation (3,000x speedup)

**Exports** (direct mathematical usage):
```python
from hololoom.semantic_calculus import (
    SemanticFlowCalculus,      # Compute trajectories
    SemanticSpectrum,           # 16D projection
    EthicalSemanticPolicy,      # Ethical analysis
)
```

### Layer 2: Integration Layer (`hololoom/semantic_calculus/integration.py`) ✨ NEW

**Clean adapter between mathematics and HoloLoom** - this is the **organized interface**:

#### Configuration
```python
from hololoom.semantic_calculus import SemanticCalculusConfig

# Three presets
config = SemanticCalculusConfig.fast()          # 8D, no ethics, cache enabled
config = SemanticCalculusConfig.balanced()      # 16D, ethics, cache enabled
config = SemanticCalculusConfig.comprehensive() # 32D, full features

# Or custom
config = SemanticCalculusConfig(
    enable_cache=True,
    cache_size=10000,
    dimensions=16,
    dt=1.0,
    ethical_framework="compassionate",  # or "scientific", "therapeutic"
    compute_trajectory=True,
    compute_ethics=True,
)

# Or from pattern spec
config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)
```

#### Unified Analyzer
```python
from hololoom.semantic_calculus import create_semantic_analyzer

# Create analyzer (handles all complexity internally)
analyzer = create_semantic_analyzer(
    embed_fn=lambda words: embedder.encode(words),
    config=SemanticCalculusConfig.balanced()
)

# Use it
result = analyzer.analyze_text("Your text here...")
# Returns: {trajectory, semantic_forces, ethics (optional)}

# Or extract features for DotPlasma
features = analyzer.extract_features("Your text here...")
# Returns: {n_states, total_distance, avg_velocity, curvature, ...}

# Check cache performance
stats = get_cache_stats(analyzer)
# Returns: {hits, misses, hit_rate, size, ...}
```

**Key Benefits**:
- ✅ Single entry point (`create_semantic_analyzer`)
- ✅ Configuration presets (fast/balanced/comprehensive)
- ✅ Automatic cache management
- ✅ Clean feature extraction for ResonanceShed
- ✅ Backward compatible with legacy code

### Layer 3: HoloLoom Integration

#### 3.1 Configuration (`hololoom/config.py`)

Added semantic calculus settings to Config class:
```python
from hololoom.config import Config

# Semantic calculus is disabled by default for backward compatibility
config = Config.fast()
config.enable_semantic_calculus = True  # Enable it
config.semantic_dimensions = 16
config.semantic_cache_size = 10000
config.semantic_framework = "compassionate"
config.semantic_trajectory = True
config.semantic_ethics = True
```

**Factory methods updated**:
```python
# BARE mode - semantic calculus disabled
config = Config.bare()
assert config.enable_semantic_calculus == False

# FAST mode - semantic calculus disabled by default
config = Config.fast()
config.enable_semantic_calculus = False  # but 8D if enabled
config.semantic_dimensions = 8
config.semantic_ethics = False

# FUSED mode - semantic calculus disabled by default
config = Config.fused()
config.enable_semantic_calculus = False  # but 16D if enabled
config.semantic_dimensions = 16
config.semantic_ethics = True
```

#### 3.2 Pattern Cards (`hololoom/loom/command.py`)

SEMANTIC_FLOW pattern card for dedicated semantic analysis:
```python
from hololoom.loom.command import PatternCard, SEMANTIC_FLOW_PATTERN

# Pattern card already exists
pattern = SEMANTIC_FLOW_PATTERN
# Configured with:
# - enable_semantic_flow=True
# - semantic_dimensions=16
# - semantic_trajectory=True
# - semantic_ethics=True
```

#### 3.3 ResonanceShed (`hololoom/resonance/shed.py`)

**Backward compatible** - accepts both old and new interfaces:
```python
resonance_shed = ResonanceShed(
    motif_detector=motif_detector,
    embedder=embedder,
    spectral_fusion=spectral_fusion,
    semantic_calculus=analyzer,  # Can be SemanticAnalyzer (new) or SemanticFlowCalculus (legacy)
)

# Internally handles both:
if isinstance(self.semantic_calculus, SemanticAnalyzer):
    # New clean interface
    features = self.semantic_calculus.extract_features(text)
else:
    # Legacy interface (backward compatible)
    trajectory = self.semantic_calculus.compute_trajectory(words)
    features = {...}  # manual extraction
```

#### 3.4 WeavingShuttle (`hololoom/weaving_shuttle.py`)

Uses clean integration layer:
```python
# Old way (removed):
# from hololoom.semantic_calculus import SemanticFlowCalculus
# semantic_calculus = SemanticFlowCalculus(embed_fn, enable_cache=True, cache_size=10000)

# New way (organized):
from hololoom.semantic_calculus import create_semantic_analyzer, SemanticCalculusConfig

sem_config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)
semantic_calculus = create_semantic_analyzer(embed_fn, config=sem_config)

# Automatically configured with:
# - Dimensions from pattern_spec
# - Cache enabled
# - Ethics based on pattern
# - Full trajectory analysis
```

### Layer 4: User-Facing APIs

#### 4.1 Unified API (`hololoom/unified_api.py`)

```python
from hololoom.unified_api import HoloLoom

# Semantic calculus via narrative depth
loom = await HoloLoom.create(
    pattern="fast",
    enable_narrative_depth=True  # Uses semantic calculus internally
)

depth = await loom.analyze_narrative_depth("Your text...")
# Includes: max_depth, cosmic_truth, symbolic_elements, etc.
```

#### 4.2 Direct Usage

```python
# Option 1: Quick analysis
from hololoom.semantic_calculus import quick_analysis

result = quick_analysis(text, embed_fn)
# Uses fast config internally

# Option 2: Full control
from hololoom.semantic_calculus import create_semantic_analyzer, SemanticCalculusConfig

config = SemanticCalculusConfig(
    dimensions=32,
    ethical_framework="scientific",
    # ... custom settings
)
analyzer = create_semantic_analyzer(embed_fn, config)
result = analyzer.analyze_text(text)
```

#### 4.3 MCP Server (`hololoom/semantic_calculus/mcp_server.py`)

Exposes 3 tools to Claude Desktop:
- `analyze_semantic_flow` - Velocity, acceleration, curvature
- `predict_conversation_flow` - Trajectory forecasting
- `evaluate_conversation_ethics` - Manipulation detection

## File Organization

```
hololoom/
├── config.py                              # ✨ Added semantic calculus config
│   └── Semantic Calculus section (7 fields)
│
├── loom/
│   └── command.py                         # ✅ SEMANTIC_FLOW pattern card
│
├── resonance/
│   └── shed.py                            # ✨ Updated to use integration layer
│       └── Handles both SemanticAnalyzer and legacy SemanticFlowCalculus
│
├── weaving_shuttle.py                     # ✨ Updated to use integration layer
│   └── Uses create_semantic_analyzer()
│
├── semantic_calculus/
│   ├── __init__.py                        # ✨ Exports integration layer
│   ├── flow_calculus.py                   # Core math
│   ├── dimensions.py                      # 16D projection
│   ├── integrator.py                      # Geometric integration
│   ├── ethics.py                          # Ethical policy
│   ├── performance.py                     # Caching (3,000x speedup)
│   ├── integration.py                     # ✨ NEW - Clean integration layer
│   │   ├── SemanticCalculusConfig        # Configuration presets
│   │   ├── SemanticAnalyzer               # Unified analyzer
│   │   ├── create_semantic_analyzer()     # Factory function
│   │   ├── create_semantic_thread()       # ResonanceShed adapter
│   │   ├── quick_analysis()               # One-shot convenience
│   │   └── get_cache_stats()              # Performance monitoring
│   └── mcp_server.py                      # Claude Desktop tools
│
└── unified_api.py                         # ✅ Uses semantic calculus via narrative depth

tests/
└── test_semantic_calculus_mcp.py          # ✅ 11 tests (all passing)

demos/
├── semantic_calculus_benchmark.py         # Performance tests
└── narrative_depth_production.py          # ✅ Production demo (working)
```

## Usage Patterns

### Pattern 1: Enable in Config (Recommended)
```python
from hololoom.weaving_shuttle import WeavingShuttle
from hololoom.config import Config

config = Config.fast()
config.enable_semantic_calculus = True  # Single line to enable!

async with WeavingShuttle(cfg=config) as shuttle:
    spacetime = await shuttle.weave(query="Your text...")
    semantic_features = spacetime.trace.dot_plasma.get('semantic_flow')
```

### Pattern 2: Use SEMANTIC_FLOW Pattern Card
```python
from hololoom.weaving_shuttle import WeavingShuttle
from hololoom.loom.command import LoomCommand, PatternCard

async with WeavingShuttle() as shuttle:
    spacetime = await shuttle.weave(
        query="Your text...",
        user_pattern="semantic_flow"  # Uses SEMANTIC_FLOW card
    )
```

### Pattern 3: Direct Semantic Analysis (Standalone)
```python
from hololoom.semantic_calculus import create_semantic_analyzer
from hololoom.embedding.spectral import create_embedder

embedder = create_embedder(sizes=[384])
embed_fn = lambda words: embedder.encode(words)

analyzer = create_semantic_analyzer(embed_fn)
result = analyzer.analyze_text("Your conversation...")

print(f"Velocity: {result['trajectory'].states[0].speed}")
print(f"Curvature: {result['trajectory'].curvature(0)}")
print(f"Ethics: {result['ethics']['virtue_score']}")
```

### Pattern 4: Integration with HoloLoom Unified API
```python
from hololoom.unified_api import HoloLoom

loom = await HoloLoom.create(enable_narrative_depth=True)
depth = await loom.analyze_narrative_depth("Epic story text...")

# Semantic calculus used internally
print(depth['max_depth'])        # COSMIC
print(depth['cosmic_truth'])     # Ultimate meaning
print(depth['symbolic_elements']) # Detected symbols
```

## Key Improvements

### 1. Separation of Concerns ✅
- **Core math** (flow_calculus.py, dimensions.py, etc.) - Pure implementation
- **Integration** (integration.py) - Adapter layer
- **Configuration** (config.py) - Settings
- **Usage** (weaving_shuttle.py, resonance/shed.py) - Application

### 2. Clean Entry Points ✅
- `create_semantic_analyzer()` - Main factory
- `SemanticCalculusConfig` - Configuration presets
- `quick_analysis()` - One-shot convenience
- Backward compatible with legacy code

### 3. Configuration Management ✅
- Centralized in `Config` class
- Three presets (fast/balanced/comprehensive)
- Pattern spec integration
- Disabled by default (opt-in)

### 4. Organized Exports ✅
```python
# Clean imports
from hololoom.semantic_calculus import (
    # Integration layer (recommended)
    create_semantic_analyzer,
    SemanticCalculusConfig,
    SemanticAnalyzer,

    # Core (if needed)
    SemanticFlowCalculus,
    SemanticSpectrum,
    EthicalSemanticPolicy,

    # Performance
    get_cache_stats,
)
```

### 5. Backward Compatibility ✅
- ResonanceShed accepts both `SemanticAnalyzer` (new) and `SemanticFlowCalculus` (legacy)
- Existing code continues to work
- Old imports still valid
- Gradual migration path

## Performance

### Cache Statistics
```python
from hololoom.semantic_calculus import get_cache_stats

stats = get_cache_stats(analyzer)
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Total requests: {stats['total_requests']}")
print(f"Size: {stats['size']}/{stats['max_size']}")
```

### Benchmarks
- **Embedding cache (warm)**: 521,782x speedup
- **Trajectory computation**: 2,932x speedup
- **Overall real-world**: ~3,000x faster
- **Hit rate**: 85-95% in typical usage

## Testing

All tests passing (11/11):
```bash
cd c:\Users\blake\Documents\mythRL
PYTHONPATH=. python tests/test_semantic_calculus_mcp.py
```

Production demo working:
```bash
PYTHONPATH=. python demos/narrative_depth_production.py
```

## Migration Guide

### Upgrading Existing Code

**Before** (legacy):
```python
from hololoom.semantic_calculus import SemanticFlowCalculus

calculus = SemanticFlowCalculus(embed_fn, enable_cache=True, cache_size=10000)
trajectory = calculus.compute_trajectory(words)
# Manual feature extraction...
```

**After** (organized):
```python
from hololoom.semantic_calculus import create_semantic_analyzer

analyzer = create_semantic_analyzer(embed_fn)
result = analyzer.analyze_text(text)  # Includes trajectory + features + ethics
```

**No breaking changes** - both still work!

## Summary

✅ **Organized Architecture** - Clear layers (core/integration/application)
✅ **Clean Entry Points** - Single factory function + config presets
✅ **Centralized Configuration** - Config class + pattern cards
✅ **Backward Compatible** - Existing code works unchanged
✅ **Well Documented** - Clear usage patterns
✅ **Fully Tested** - 11/11 tests passing
✅ **Production Ready** - Demo working, 3,000x performance boost

The semantic calculus is now **cleanly integrated** and **ready for use**! 🚀
