# Semantic Dimensions - Interpretable Embedding Projections

**Status**: Production Ready (November 2025)
**Location**: `hololoom/semantic_calculus/dimensions.py` (1,720 lines)
**Dimensions**: 244 total (16 standard + 228 extended)
**Use Case**: Project opaque 384D embeddings → interpretable semantic axes

Transform high-dimensional embeddings into human-readable semantic coordinates.

---

## Overview

Semantic Dimensions solve a fundamental problem: 384-dimensional embeddings are mathematically powerful but completely opaque. What does dimension 147 mean? Nobody knows.

**The Solution**: Project embeddings onto **learned interpretable axes** with clear semantics:
- "Warmth: +0.73" (warm, friendly language)
- "Formality: -0.45" (casual, informal tone)
- "Urgency: +0.91" (time-sensitive request)

Each dimension is learned from exemplar word pairs (e.g., "warm, friendly, caring" vs "cold, distant, aloof"), creating meaningful axes in semantic space.

---

## Quick Start

```python
from hololoom.semantic_calculus import SemanticSpectrum, STANDARD_DIMENSIONS
from hololoom.embedding.spectral import create_embedder

# Create embedder
embedder = create_embedder(sizes=[384])
embed_fn = lambda word: embedder.encode([word])[0]

# Create spectrum analyzer with 16 standard dimensions
spectrum = SemanticSpectrum(dimensions=STANDARD_DIMENSIONS)
spectrum.learn_axes(embed_fn)

# Project any embedding
query_embedding = embed_fn("Please help me urgently!")
projection = spectrum.project_vector(query_embedding)

print(projection)
# {
#     'Warmth': 0.42,
#     'Formality': -0.38,
#     'Urgency': 0.85,
#     ...
# }
```

---

## 16 Standard Dimensions

The core dimensions that capture fundamental semantic properties:

| Dimension | Positive Pole | Negative Pole | Use Case |
|-----------|--------------|---------------|----------|
| **Warmth** | warm, friendly, caring | cold, distant, aloof | Emotional tone |
| **Valence** | positive, good, pleasant | negative, bad, unpleasant | Sentiment |
| **Arousal** | excited, energetic, alert | calm, relaxed, sleepy | Energy level |
| **Intensity** | intense, extreme, powerful | mild, gentle, subtle | Strength |
| **Formality** | formal, official, professional | informal, casual, colloquial | Register |
| **Directness** | direct, explicit, straightforward | indirect, implicit, subtle | Communication style |
| **Power** | powerful, dominant, commanding | weak, submissive, meek | Authority |
| **Generosity** | generous, giving, altruistic | selfish, greedy, stingy | Sharing orientation |
| **Certainty** | certain, sure, definite | uncertain, doubtful, ambiguous | Confidence |
| **Complexity** | complex, intricate, sophisticated | simple, plain, basic | Cognitive load |
| **Concreteness** | concrete, tangible, physical | abstract, conceptual, theoretical | Specificity |
| **Familiarity** | familiar, common, everyday | unfamiliar, exotic, strange | Novelty |
| **Agency** | active, initiative, self-directed | passive, reactive, dependent | Control |
| **Stability** | stable, steady, constant | unstable, volatile, changing | Consistency |
| **Urgency** | urgent, pressing, immediate | relaxed, leisurely, unhurried | Time pressure |
| **Completion** | complete, finished, whole | incomplete, partial, ongoing | Status |

---

## Extended Dimensions (228 additional)

For deep semantic analysis, HoloLoom provides 228 extended dimensions across 15 categories:

### Dimension Categories

| Category | Count | Focus |
|----------|-------|-------|
| **Narrative** | 16 | Story structure (suspense, resolution, foreshadowing) |
| **Emotional Depth** | 16 | Nuanced emotions (melancholy, euphoria, bittersweet) |
| **Relational** | 16 | Relationships (intimacy, rivalry, mentorship) |
| **Archetypal** | 16 | Mythic patterns (hero, shadow, trickster) |
| **Philosophical** | 16 | Abstract concepts (free will, determinism, meaning) |
| **Transformation** | 16 | Change and growth (evolution, revolution, decay) |
| **Moral/Ethical** | 16 | Ethics (justice, mercy, integrity) |
| **Creative** | 16 | Creativity (originality, convention, inspiration) |
| **Cognitive Complexity** | 16 | Thinking (analytical, intuitive, systematic) |
| **Temporal Narrative** | 16 | Time in stories (flashback, present, foreshadowing) |
| **Spatial Setting** | 12 | Place (interior, exterior, sacred, profane) |
| **Character** | 12 | Characters (protagonist, antagonist, foil) |
| **Plot** | 12 | Plot elements (conflict, climax, resolution) |
| **Theme** | 12 | Themes (love, death, redemption, betrayal) |
| **Style/Voice** | 4 | Literary style (epic, lyric, tragic, sublime) |

### Using Extended Dimensions

```python
from hololoom.semantic_calculus import SemanticSpectrum, EXTENDED_244_DIMENSIONS

# Full 244-dimension projection
spectrum = SemanticSpectrum(dimensions=EXTENDED_244_DIMENSIONS)
spectrum.learn_axes(embed_fn)

projection = spectrum.project_vector(embedding)
# Returns 244 interpretable values!
```

---

## Key Classes

### SemanticDimension

A single interpretable axis in semantic space:

```python
@dataclass
class SemanticDimension:
    name: str                          # Human-readable name
    positive_exemplars: List[str]      # Words defining positive pole
    negative_exemplars: List[str]      # Words defining negative pole
    axis: Optional[np.ndarray] = None  # Learned axis vector

    def learn_axis(self, embed_fn, use_batch=True):
        """Learn axis from exemplar embeddings"""
        # Computes: axis = mean(positive) - mean(negative)
        # Normalized to unit length

    def project(self, vector, manifold=None) -> float:
        """Project vector onto this dimension"""
        # Returns: dot(vector, axis) ∈ [-1, 1] typically
```

### SemanticSpectrum

Projects trajectories onto all dimensions:

```python
class SemanticSpectrum:
    def __init__(self, dimensions: List[SemanticDimension] = None):
        """Initialize with dimension set (default: STANDARD_DIMENSIONS)"""

    def learn_axes(self, embed_fn: Callable):
        """Learn all dimension axes from exemplars"""

    def project_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """Project single vector to all dimensions"""

    def project_trajectory(self, positions: np.ndarray) -> Dict[str, np.ndarray]:
        """Project trajectory (n_steps, embed_dim) to all dimensions"""

    def compute_spectrum_velocity(self, positions, dt=1.0) -> Dict[str, np.ndarray]:
        """Compute rate of change along each dimension"""

    def compute_spectrum_acceleration(self, positions, dt=1.0) -> Dict[str, np.ndarray]:
        """Compute forces acting on each dimension"""

    def get_dominant_dimensions(self, velocity_dict, top_k=5) -> List[Tuple]:
        """Find dimensions changing most rapidly"""

    def analyze_semantic_forces(self, positions, dt=1.0) -> Dict:
        """Complete force analysis across all dimensions"""
```

---

## Trajectory Analysis

Track how semantic properties evolve over a conversation:

```python
# Conversation as word sequence
words = ["hello", "I", "need", "urgent", "help", "please"]
positions = np.array([embed_fn(w) for w in words])

# Analyze semantic forces
analysis = spectrum.analyze_semantic_forces(positions, dt=1.0)

print("Dominant dimensions by velocity:")
for name, magnitude in analysis['dominant_velocity'][:5]:
    print(f"  {name}: {magnitude:.4f}")

print("\nDominant dimensions by force (acceleration):")
for name, magnitude in analysis['dominant_force'][:5]:
    print(f"  {name}: {magnitude:.4f}")

# Track individual dimension evolution
urgency_trajectory = analysis['projections']['Urgency']
# Shows: [0.1, 0.2, 0.5, 0.9, 0.85, 0.8]
# Urgency spikes at "urgent" then slightly declines at "please"
```

---

## PDE-Based Temporal Evolution

Enable physics-based evolution of semantic projections:

```python
# Enable temporal dynamics
spectrum.enable_temporal_dynamics(
    pde_type="heat",        # "heat", "wave", "reaction_diffusion", "hamilton_jacobi"
    dt=0.01                 # Time step
)

# Project initial state
initial = spectrum.project_vector(embedding)

# Evolve over time (simulates how meaning might "spread")
evolved = spectrum.evolve(initial, steps=10)

print(f"Warmth: {initial['Warmth']:.3f} -> {evolved['Warmth']:.3f}")
```

**PDE Types**:
- **heat**: Diffusion (meanings spread/blur over time)
- **wave**: Oscillation (meanings propagate like waves)
- **reaction_diffusion**: Pattern formation (meanings form structures)
- **hamilton_jacobi**: Conservative evolution (energy preserved)

---

## Visualization

```python
from hololoom.semantic_calculus import visualize_semantic_spectrum, print_spectrum_summary

# Analyze trajectory
analysis = spectrum.analyze_semantic_forces(positions)

# Visual plot (matplotlib)
fig, axes = visualize_semantic_spectrum(
    analysis,
    words=["hello", "I", "need", "urgent", "help", "please"],
    top_k=8,  # Show top 8 dimensions
    save_path="semantic_spectrum.png"
)

# Text summary
print_spectrum_summary(analysis, words)
# Output:
# ========================================================================
# SEMANTIC SPECTRUM ANALYSIS
# ========================================================================
#
# Top 5 dimensions by velocity (how fast they're changing):
#   1. Urgency         (avg velocity: 0.1523)
#   2. Formality       (avg velocity: 0.0892)
#   ...
```

---

## Integration with HoloLoom

Semantic dimensions integrate with the broader semantic calculus system:

```python
from hololoom.semantic_calculus import create_semantic_analyzer

# Create complete analysis pipeline
analyzer = create_semantic_analyzer(embed_fn, dt=1.0)

# Components:
# - analyzer['calculus']: SemanticFlowCalculus (trajectories)
# - analyzer['spectrum']: SemanticSpectrum (16D projection)
# - analyzer['integrator']: GeometricIntegrator (symplectic physics)
# - analyzer['policy']: EthicalSemanticPolicy (ethical constraints)

# Full conversation analysis
result = analyze_conversation(words, embed_fn)

print(result['trajectory'])        # Position, velocity, curvature
print(result['semantic_forces'])   # Dimension-wise analysis
print(result['ethics'])            # Ethical evaluation
```

---

## Defining Custom Dimensions

Create your own interpretable dimensions:

```python
from hololoom.semantic_calculus import SemanticDimension, SemanticSpectrum

# Define custom dimension
technical_dim = SemanticDimension(
    name="Technical",
    positive_exemplars=["algorithm", "implementation", "architecture", "protocol", "API"],
    negative_exemplars=["intuitive", "natural", "simple", "everyday", "casual"]
)

business_dim = SemanticDimension(
    name="Business",
    positive_exemplars=["revenue", "profit", "stakeholder", "ROI", "KPI"],
    negative_exemplars=["hobby", "fun", "personal", "leisure", "play"]
)

# Create custom spectrum
custom_spectrum = SemanticSpectrum(dimensions=[technical_dim, business_dim])
custom_spectrum.learn_axes(embed_fn)

# Use for domain-specific analysis
projection = custom_spectrum.project_vector(query_embedding)
print(f"Technical: {projection['Technical']:.2f}")
print(f"Business: {projection['Business']:.2f}")
```

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **learn_axes()** | ~500ms | One-time per dimension set |
| **project_vector()** | <1ms | 16 dot products (standard) |
| **project_trajectory()** | ~5ms | 100-step trajectory |
| **analyze_semantic_forces()** | ~15ms | Full velocity + acceleration |
| **evolve()** (PDE) | ~50ms | 10 steps |

**Caching**: Dimension axes are cached after learning. Repeated projections are fast.

---

## When to Use

**Use Semantic Dimensions when**:
- Need interpretable explanations ("why did the model respond this way?")
- Tracking emotional or tonal shifts in conversations
- Building UIs that show semantic properties
- Debugging retrieval behavior
- Ethical auditing (tracking manipulation, deception signals)

**Use raw embeddings when**:
- Speed is critical (skip projection overhead)
- Need full representational capacity
- Building learned systems (let model learn its own features)

---

## Related Modules

The semantic calculus package provides additional capabilities:

| Module | Purpose |
|--------|---------|
| `flow_calculus.py` | Differential geometry for semantic trajectories |
| `integrator.py` | Symplectic integration (energy-preserving) |
| `ethics.py` | Ethical policies and constraints |
| `hyperbolic.py` | Poincare ball embeddings for hierarchies |
| `integral_geometry.py` | Tomographic reconstruction |
| `system_id.py` | Learn dynamics from data |

---

## Example: Sentiment Tracking

```python
# Track sentiment evolution in customer support conversation
conversation = [
    "I'm really frustrated with your product",
    "It keeps crashing every time I use it",
    "I've tried everything and nothing works",
    "Oh, the reset button fixed it!",
    "Thanks so much for your help!"
]

positions = np.array([embed_fn(turn) for turn in conversation])
analysis = spectrum.analyze_semantic_forces(positions)

# Visualize valence trajectory
valence = analysis['projections']['Valence']
print("Valence trajectory:")
for i, (turn, v) in enumerate(zip(conversation, valence)):
    print(f"  {i+1}. [{v:+.2f}] {turn[:40]}...")

# Output:
# Valence trajectory:
#   1. [-0.72] I'm really frustrated with your product
#   2. [-0.58] It keeps crashing every time I use it
#   3. [-0.81] I've tried everything and nothing works
#   4. [+0.45] Oh, the reset button fixed it!
#   5. [+0.89] Thanks so much for your help!
```

---

## See Also

- [flow_calculus.py](flow_calculus.py) - Differential geometry for trajectories
- [integrator.py](integrator.py) - Symplectic integration
- [ethics.py](ethics.py) - Ethical policy framework
- [CLAUDE.md](../../CLAUDE.md) - Main project documentation
