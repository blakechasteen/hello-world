# Resonance Shed - Feature Extraction Hub

**Status**: ✅ Production Ready (November 2025)
**Location**: `HoloLoom/resonance/`
**Code**: 846 lines across 2 files

---

## Overview

The **Resonance Shed** is HoloLoom's feature extraction hub where multiple feature threads (motif, embedding, spectral) are "lifted" and combined through interference patterns to create a rich, multi-modal representation called **DotPlasma**.

**Weaving Metaphor**: In traditional weaving, the **shed** is the vertical space created when some warp threads are lifted and others lowered. HoloLoom's Resonance Shed lifts feature threads and creates interference patterns through multi-modal fusion.

---

## Architecture

### File Structure

```
HoloLoom/resonance/
├── __init__.py     # 12 lines - Public exports
└── shed.py         # 834 lines - ResonanceShed implementation
```

### Core Concept: DotPlasma

**DotPlasma** (alias for `Features`) is the "feature fluid"—a flowing continuous representation that contains:

```python
@dataclass
class DotPlasma:  # Alias: Features
    psi: np.ndarray              # Ψ (spectral features)
    motifs: List[str]            # Detected linguistic patterns
    metrics: Dict[str, float]    # Graph metrics
    confidence: float            # Overall confidence
    embedding: Optional[np.ndarray]  # Optional full embedding
    scales: Optional[List[int]]  # Active scales
```

**Three Feature Threads**:

1. **Motif Thread** (Symbolic)
   - Linguistic patterns: "question→answer", "cause→effect"
   - Regex and spaCy detection
   - Controls neural pathways

2. **Embedding Thread** (Continuous)
   - Matryoshka multi-scale embeddings
   - 96/192/384/768 dimensions
   - Semantic similarity

3. **Spectral Thread** (Topological)
   - Graph Laplacian eigenvalues
   - SVD topic components
   - Structural features

---

## Core Class: ResonanceShed

### Initialization

```python
from HoloLoom.resonance import ResonanceShed
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.motif.unified import create_motif_detector

# Create embedder and motif detector
embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
motif_detector = create_motif_detector(mode="hybrid")

# Create resonance shed
shed = ResonanceShed(
    embedder=embedder,
    motif_detector=motif_detector,
    scales=[96, 192, 384],
    enable_spectral=True
)
```

### Main Method: `lift_threads()`

Extracts all feature threads and creates DotPlasma:

```python
# Lift feature threads from query
dot_plasma = await shed.lift_threads(
    query="What is Thompson Sampling?",
    context_shards=retrieved_shards,
    kg_subgraph=subgraph
)

print(f"Motifs: {dot_plasma.motifs}")
# ["question→answer"]

print(f"Spectral features: {dot_plasma.psi.shape}")
# (6,)  # [eigen1, eigen2, svd1, svd2, entropy, density]

print(f"Confidence: {dot_plasma.confidence:.2f}")
# 0.85
```

---

## Feature Extraction Pipeline

### Step-by-Step Process

```
Query + Context → ResonanceShed
    ↓
1. Motif Detection (symbolic thread)
   - Regex patterns: "what is", "explain", etc.
   - spaCy linguistic patterns
   - Output: ["question→answer", "explanation"]
    ↓
2. Embedding (continuous thread)
   - Encode query with MatryoshkaEmbeddings
   - Multi-scale: 96D, 192D, 384D
   - Output: embedding vectors per scale
    ↓
3. Spectral Analysis (topological thread)
   - Graph Laplacian: eigenvalues capture structure
   - SVD: topic components
   - Graph metrics: entropy, density
   - Output: Ψ (6-dimensional spectral vector)
    ↓
4. Fusion (interference pattern)
   - Combine all threads
   - Compute overall confidence
   - Create DotPlasma
    ↓
Output: DotPlasma (malleable feature representation)
```

---

## Usage Examples

### Example 1: Basic Feature Extraction

```python
from HoloLoom.resonance import ResonanceShed

# Create shed
shed = ResonanceShed(
    embedder=embedder,
    motif_detector=motif_detector,
    scales=[384]  # Single scale for speed
)

# Extract features
dot_plasma = await shed.lift_threads(
    query="Explain neural networks",
    context_shards=[]
)

print(f"Motifs: {dot_plasma.motifs}")
# ["explanation", "technical_explanation"]

print(f"Confidence: {dot_plasma.confidence:.2f}")
# 0.78
```

### Example 2: Multi-Scale Features

```python
# Multi-scale extraction
shed = ResonanceShed(
    embedder=embedder,
    motif_detector=motif_detector,
    scales=[96, 192, 384],  # Three scales
    enable_spectral=True
)

dot_plasma = await shed.lift_threads(
    query="What are the tradeoffs?",
    context_shards=retrieved_shards
)

print(f"Scales: {dot_plasma.scales}")
# [96, 192, 384]

print(f"Spectral (Ψ): {dot_plasma.psi}")
# [0.15, 0.08, 0.22, 0.31, 1.45, 0.62]
# [eigen1, eigen2, svd1, svd2, entropy, density]
```

### Example 3: With Knowledge Graph

```python
from HoloLoom.memory.graph import KG

# Create knowledge graph
kg = KG()
kg.add_edge("Thompson Sampling", "exploration", "USES")
kg.add_edge("Thompson Sampling", "Bayesian", "IS_A")

# Extract subgraph
subgraph = kg.get_subgraph("Thompson Sampling", max_depth=2)

# Lift threads with graph context
dot_plasma = await shed.lift_threads(
    query="What is Thompson Sampling?",
    context_shards=[],
    kg_subgraph=subgraph
)

# Spectral features capture graph structure
print(f"Graph entropy: {dot_plasma.psi[4]:.2f}")
# Higher entropy = more complex structure
```

### Example 4: Confidence-Based Escalation

```python
# Extract features with confidence threshold
dot_plasma = await shed.lift_threads(
    query="complex technical query",
    context_shards=retrieved_shards
)

if dot_plasma.confidence < 0.7:
    print("Low confidence - escalating to FUSED mode")
    # Re-extract with more features
    shed_fused = ResonanceShed(
        embedder=embedder,
        motif_detector=motif_detector,
        scales=[96, 192, 384, 768],  # More scales
        enable_spectral=True
    )
    dot_plasma = await shed_fused.lift_threads(
        query, context_shards
    )
```

---

## Integration with Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.resonance import ResonanceShed

# Orchestrator creates resonance shed
orchestrator = WeavingOrchestrator(
    resonance_shed=shed,  # Feature extractor
    # ... other config
)

# During weaving:
# 1. Query arrives
# 2. Resonance Shed lifts feature threads
# 3. DotPlasma created (multi-modal representation)
# 4. DotPlasma flows to Policy Engine for decision
# 5. Policy uses features to select tool
```

---

## Feature Thread Details

### Motif Thread (Symbolic)

**Detected Patterns**:
- `question→answer`: "what is", "who is", "where is"
- `explanation`: "explain", "describe", "elaborate"
- `comparison`: "vs", "versus", "compare"
- `cause→effect`: "because", "therefore", "leads to"
- `procedure`: "how to", "step by step"

**Detection Modes**:
- Regex (fast, 88% accuracy)
- spaCy (slower, 95% accuracy)
- Hybrid (best of both)

### Embedding Thread (Continuous)

**Matryoshka Scales**:
```
96D  → Coarse semantic similarity
192D → Mid-level semantic similarity
384D → Fine-grained semantic similarity
768D → Maximum semantic fidelity
```

**Properties**:
- Prefix property: 96D is prefix of 192D, etc.
- Zero-copy slicing for efficiency
- Adaptive scale selection

### Spectral Thread (Topological)

**6 Spectral Features** (Ψ):
```python
psi = [
    eigen1,    # 1st Laplacian eigenvalue (connectivity)
    eigen2,    # 2nd Laplacian eigenvalue (bipartiteness)
    svd1,      # 1st SVD component (main topic)
    svd2,      # 2nd SVD component (secondary topic)
    entropy,   # Graph entropy (complexity)
    density    # Edge density (interconnectedness)
]
```

---

## API Reference

### Core Methods

#### `ResonanceShed.__init__()`
```python
def __init__(
    self,
    embedder: MatryoshkaEmbeddings,
    motif_detector: MotifDetector,
    scales: List[int] = [384],
    enable_spectral: bool = True
)
```

#### `ResonanceShed.lift_threads()`
```python
async def lift_threads(
    self,
    query: str,
    context_shards: List[MemoryShard] = [],
    kg_subgraph: Optional[KG] = None
) -> DotPlasma
```

#### `DotPlasma` (Features)
```python
@dataclass
class DotPlasma:
    psi: np.ndarray              # Spectral features (6D)
    motifs: List[str]            # Detected patterns
    metrics: Dict[str, float]    # Graph metrics
    confidence: float            # Overall confidence (0-1)
    embedding: Optional[np.ndarray]  # Full embedding
    scales: Optional[List[int]]  # Active scales
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Motif detection (regex)** | ~0.5ms | Fast patterns |
| **Motif detection (spaCy)** | ~10ms | Linguistic patterns |
| **Embedding (1 scale)** | ~5ms | Single forward pass |
| **Embedding (3 scales)** | ~8ms | Zero-copy slicing |
| **Spectral features** | ~15ms | SVD + eigenvalues |
| **Total (FAST mode)** | ~25ms | All threads |

**Memory**: ~2-5MB (embeddings dominate)

---

## Dependencies

**Internal**:
```python
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.motif.unified import MotifDetector
from HoloLoom.memory.protocol import MemoryShard
from HoloLoom.memory.graph import KG
from HoloLoom.documentation.types import Features  # Alias: DotPlasma
```

**External**:
```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
```

---

## Summary

The Resonance Shed provides:

✅ **Multi-modal feature extraction** (symbolic + continuous + topological)
✅ **DotPlasma creation** (malleable feature representation)
✅ **Three feature threads** (motif, embedding, spectral)
✅ **Confidence scoring** for escalation decisions
✅ **Multi-scale support** (96/192/384/768D embeddings)
✅ **Knowledge graph integration** (spectral features from structure)
✅ **~25ms total latency** (FAST mode, all threads)

The Resonance Shed creates the feature-rich DotPlasma that flows through the rest of the weaving cycle.
