"""
darkTrace - Semantic Reverse Engineering of LLMs
=================================================
Provides tools for semantic observation, trajectory prediction,
and LLM fingerprinting through interpretable 244D semantic analysis.

## Architecture

darkTrace implements a 4-layer architecture for LLM reverse engineering:

### Layer 1: OBSERVATION ✅
Real-time semantic monitoring of LLM outputs
- SemanticObserver: 244D semantic state tracking
- TrajectoryRecorder: Continuous trajectory recording
- Uses HoloLoom semantic calculus for analysis

### Layer 2: ANALYSIS 🟡
Trajectory prediction and pattern recognition
- TrajectoryPredictor: System identification via differential equations
- FingerprintGenerator: Create unique semantic fingerprints
- PatternRecognizer: Detect recurring patterns
- AttractorDetector: Find stable semantic attractors

### Layer 3: CONTROL 🔴
Embedding manipulation and semantic steering (future)
- EmbeddingManipulator: Direct embedding space control
- SemanticSteerer: Guide outputs via semantic gradients
- Requires research into safe control methods

### Layer 4: EXPLOITATION 🔴
Security research and behavior analysis (future)
- JailbreakDetector: Detect adversarial prompts
- BehaviorCloner: Clone LLM decision patterns
- Safety-critical, requires explicit opt-in

## Quick Start

```python
from darkTrace import DarkTraceConfig, create_observer

# Create observer with narrative domain
config = DarkTraceConfig.narrative()
observer = create_observer(config)

# Observe LLM output
state = observer.observe("Once upon a time...")
print(f"Dominant dimensions: {state.dominant_dimensions}")
print(f"Semantic velocity: {state.velocity}")

# Get trajectory
trajectory = observer.get_trajectory()
print(f"Trajectory length: {len(trajectory)}")
```

## Configuration Presets

- `DarkTraceConfig.bare()` - Minimal (16D, fast prototyping)
- `DarkTraceConfig.fast()` - Balanced (36D, general use)
- `DarkTraceConfig.fused()` - Full (36D, deep analysis)
- `DarkTraceConfig.narrative()` - Narrative analysis
- `DarkTraceConfig.dialogue()` - Conversation analysis
- `DarkTraceConfig.technical()` - Technical documentation

## Dependencies

darkTrace depends on HoloLoom core for reusable primitives:
- Semantic calculus (244D → 36D smart selection)
- Reflection buffer (learning from outcomes)
- Policy engine (decision making)

Install both:
```bash
pip install -e ../../HoloLoom
pip install -e .
```

## CLI Usage

```bash
# Analyze text file
darktrace analyze input.txt --config narrative

# Real-time monitoring
darktrace monitor --config dialogue --live

# Generate fingerprint
darktrace fingerprint model_output.jsonl --output fingerprint.json
```

## Dashboard Integration

darkTrace exposes an API for smart dashboard integration:
```python
from darkTrace.api import DarkTraceAPI

api = DarkTraceAPI(config=DarkTraceConfig.fused())
result = await api.analyze_text("Your text here")
```

See `api.py` for full API reference.
"""

__version__ = "0.1.0"

# Configuration
from darkTrace.config import (
    DarkTraceConfig,
    ObserverConfig,
    AnalyzerConfig,
    AnalysisLayer,
    DomainType,
)

# Public API exports (to be implemented)
# These will be added as modules are created:
#
# from darkTrace.observers import SemanticObserver, TrajectoryRecorder
# from darkTrace.analyzers import (
#     TrajectoryPredictor,
#     FingerprintGenerator,
#     PatternRecognizer,
#     AttractorDetector
# )

# Factory functions (placeholders for now)
def create_observer(config: DarkTraceConfig):
    """
    Create a semantic observer.

    Args:
        config: DarkTraceConfig instance

    Returns:
        SemanticObserver instance

    Note:
        Implementation pending. Will use HoloLoom semantic calculus.
    """
    raise NotImplementedError(
        "Observer implementation pending. "
        "Will integrate with HoloLoom semantic calculus."
    )


def create_analyzer(config: DarkTraceConfig):
    """
    Create a trajectory analyzer.

    Args:
        config: DarkTraceConfig instance

    Returns:
        TrajectoryAnalyzer instance

    Note:
        Implementation pending. Will use HoloLoom system identification.
    """
    raise NotImplementedError(
        "Analyzer implementation pending. "
        "Will integrate with HoloLoom system identification stubs."
    )


__all__ = [
    # Version
    "__version__",

    # Configuration
    "DarkTraceConfig",
    "ObserverConfig",
    "AnalyzerConfig",
    "AnalysisLayer",
    "DomainType",

    # Factory functions
    "create_observer",
    "create_analyzer",

    # Core components (to be added):
    # "SemanticObserver",
    # "TrajectoryRecorder",
    # "TrajectoryPredictor",
    # "FingerprintGenerator",
    # "PatternRecognizer",
    # "AttractorDetector",
]
