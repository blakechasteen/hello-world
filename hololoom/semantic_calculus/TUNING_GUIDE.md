# 🎛️ Semantic Detection Tuning Guide

The semantic detection layer is **fully configurable**. You can tune it in three ways:

## 1. Dimension Counts (Simple)

Control how many semantic dimensions are detected at each scale.

```python
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus

embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

# MINIMAL: Fast but coarse (good for real-time)
calculator = MatryoshkaSemanticCalculus(
    matryoshka_embedder=embedder,
    semantic_dims={
        'word': 8,      # Detect 8 dimensions at word-level
        'phrase': 32,   # Detect 32 dimensions at phrase-level
        'sentence': 64, # Detect 64 dimensions at sentence-level
        'paragraph': 128  # Detect 128 dimensions at paragraph-level
    }
)

# DEFAULT: Balanced performance and richness
calculator = MatryoshkaSemanticCalculus(
    matryoshka_embedder=embedder,
    semantic_dims=None  # Uses defaults: 16, 64, 96, 228
)

# MAXIMAL: Rich but slow (good for deep analysis)
calculator = MatryoshkaSemanticCalculus(
    matryoshka_embedder=embedder,
    semantic_dims={
        'word': 20,
        'phrase': 80,
        'sentence': 120,
        'paragraph': 244  # Full 244D semantic space!
    }
)
```

### Available Dimension Types

| Count | Includes | Example Dimensions |
|-------|----------|-------------------|
| 16 | STANDARD | Warmth, Valence, Arousal, Intensity, Formality, Power, Certainty |
| 64 | + NARRATIVE, EMOTIONAL, RELATIONAL | Journey, Transformation, Honor, Grief, Jealousy, Trust |
| 96 | + ARCHETYPAL, PHILOSOPHICAL | Hero, Mentor, Shadow, Beauty, Freedom, Wisdom |
| 228 | + TRANSFORMATION, MORAL, CREATIVE, COGNITIVE | All 244D dimensions available |

## 2. Custom Dimensions (Advanced)

Define your own semantic dimensions with custom exemplars:

```python
from HoloLoom.semantic_calculus.dimensions import SemanticDimension, SemanticSpectrum

# Define a custom dimension
tech_innovation = SemanticDimension(
    name="TechInnovation",
    positive_exemplars=[
        "innovative", "disruptive", "cutting-edge",
        "breakthrough", "revolutionary", "transformative"
    ],
    negative_exemplars=[
        "traditional", "legacy", "outdated",
        "conventional", "established", "conservative"
    ]
)

startup_hustle = SemanticDimension(
    name="StartupHustle",
    positive_exemplars=[
        "hustle", "grind", "bootstrap", "pivot",
        "iterate", "ship", "scale", "growth"
    ],
    negative_exemplars=[
        "comfortable", "stable", "predictable",
        "corporate", "bureaucratic", "slow"
    ]
)

# Create a custom spectrum
from HoloLoom.semantic_calculus.dimensions import STANDARD_DIMENSIONS

custom_dims = STANDARD_DIMENSIONS[:10] + [tech_innovation, startup_hustle]
custom_spectrum = SemanticSpectrum(dimensions=custom_dims)

# Learn axes with your embedder
def embed_fn(words):
    if isinstance(words, str):
        return embedder.encode_base([words])[0][:384]
    else:
        return embedder.encode_base(words)[:, :384]

custom_spectrum.learn_axes(embed_fn)
```

## 3. Embedding Model (Performance vs Quality)

Swap out the underlying embedding model for different quality/speed tradeoffs:

```python
# Option 1: Fast and good (DEFAULT, 2022)
embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    base_model_name="all-MiniLM-L12-v2"  # Current default
)

# Option 2: State-of-the-art (2023, slightly slower)
embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    base_model_name="BAAI/bge-small-en-v1.5"  # Best 384D model
)

# Option 3: Highest quality (768D, requires architecture changes)
embedder = MatryoshkaEmbeddings(
    sizes=[192, 384, 768],
    base_model_name="BAAI/bge-base-en-v1.5"  # 768D, best quality
)

# Option 4: Environment variable (global override)
import os
os.environ['HOLOLOOM_BASE_ENCODER'] = 'BAAI/bge-small-en-v1.5'
embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])  # Uses BGE
```

## Complete Example: Custom Startup Semantic Detector

```python
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.semantic_calculus.matryoshka_streaming import MatryoshkaSemanticCalculus
from HoloLoom.semantic_calculus.dimensions import SemanticDimension, STANDARD_DIMENSIONS

# 1. Define custom dimensions for startup analysis
custom_startup_dims = STANDARD_DIMENSIONS[:8] + [
    SemanticDimension(
        name="ProductMarketFit",
        positive_exemplars=["product-market fit", "traction", "growth", "retention", "viral"],
        negative_exemplars=["churn", "stagnant", "failing", "struggling", "pivoting"]
    ),
    SemanticDimension(
        name="FundingStage",
        positive_exemplars=["Series A", "funded", "capitalized", "investor", "valuation"],
        negative_exemplars=["bootstrapped", "unfunded", "self-funded", "pre-seed", "broke"]
    ),
    SemanticDimension(
        name="ExecutionSpeed",
        positive_exemplars=["ship fast", "iterate", "agile", "sprint", "rapid", "velocity"],
        negative_exemplars=["slow", "delayed", "stalled", "blocked", "bottleneck"]
    ),
]

# 2. Use state-of-the-art embeddings
embedder = MatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    base_model_name="BAAI/bge-small-en-v1.5"
)

# 3. Configure semantic detection
calculator = MatryoshkaSemanticCalculus(
    matryoshka_embedder=embedder,
    semantic_dims={
        'word': 8,      # Core emotions/actions
        'phrase': 20,   # Include startup-specific
        'sentence': 40, # Full startup context
        'paragraph': 60 # Deep analysis
    }
)

# 4. Analyze startup text
text = "We pivoted after realizing we had no product-market fit. Now traction is exploding."

async def analyze_startup_text():
    async def word_stream():
        for word in text.split():
            yield word

    async for snapshot in calculator.stream_analyze(word_stream()):
        print(f"Detected: {snapshot.dominant_dimensions_by_scale}")

import asyncio
asyncio.run(analyze_startup_text())
```

## Performance Impact

| Configuration | Dimensions | Speed | Quality | Use Case |
|--------------|------------|-------|---------|----------|
| MINIMAL | 8/32/64/128 | ⚡⚡⚡⚡ | ⭐⭐ | Real-time chat, streaming |
| DEFAULT | 16/64/96/228 | ⚡⚡⚡ | ⭐⭐⭐ | General purpose |
| MAXIMAL | 20/80/120/244 | ⚡⚡ | ⭐⭐⭐⭐ | Deep analysis, research |
| CUSTOM | Variable | Variable | ⭐⭐⭐⭐⭐ | Domain-specific |

## Testing Your Configuration

```bash
# Run the tunable demo
cd HoloLoom/semantic_calculus
python demo_tunable_semantics.py
```

This will compare MINIMAL, DEFAULT, and MAXIMAL configurations side-by-side.

## Next Steps

After tuning semantic detection, you're ready for **Phase 1: Semantic State → Policy Integration** where these semantic features feed directly into decision-making!

See [PRIORITY_ROADMAP.md](../../PRIORITY_ROADMAP.md) for the full plan.