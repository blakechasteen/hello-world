# 🌊📐 Multi-Scale Streaming Semantic Calculus

**Real-time word-by-word narrative analysis through 244-dimensional semantic space**

## What Is This?

This system performs **geometric integration on semantic trajectories** at multiple nested temporal scales simultaneously. Think of it as "narrative physics" - tracking the forces, momentum, and resonance patterns as text flows through a 244-dimensional interpretable semantic space.

## Key Innovation: Multi-Scale Resonance

Traditional NLP analyzes words in isolation. This system reveals **fractal narrative structure**:

- **Word-level** (1-5 tokens): Semantic jitter, rhetorical devices, micro-emotions
- **Phrase-level** (5-15 tokens): Local gestures, immediate reactions
- **Sentence-level** (15-50 tokens): Complete thoughts, logical flow
- **Paragraph-level** (50-200 tokens): Narrative beats, emotional arcs
- **Chunk-level** (200-1000 tokens): Act structure, thematic evolution

**The magic happens when you look at scale interactions:**
- When all scales align → Strong narrative momentum ("page-turner" quality)
- When scales diverge → Sophisticated complexity (literary fiction)
- When word-level chaos aligns with paragraph-level direction → Climactic moments
- When scales decouple → Stream of consciousness, experimental structure

## The 244 Dimensions

We don't track raw embeddings (opaque 384D vectors). Instead, we project onto **244 interpretable semantic dimensions**:

### Standard Dimensions (16)
Warmth, Valence, Arousal, Intensity, Formality, Directness, Power, Generosity, Certainty, Complexity, Concreteness, Familiarity, Agency, Stability, Urgency, Completion

### Narrative Dimensions (16)
Heroism, Transformation, Conflict, Mystery, Sacrifice, Wisdom, Courage, Redemption, Destiny, Honor, Loyalty, Quest, Transcendence, Shadow, Initiation, Rebirth

### Emotional Depth (16)
Authenticity, Vulnerability, Trust, Hope, Grief, Shame, Compassion, Rage, Longing, Awe, Jealousy, Guilt, Pride, Disgust, Ecstasy, Dread

### Archetypal (16)
Hero, Mentor, Shadow, Trickster, Mother, Father, Child, Anima-Animus, Self, Threshold-Guardian, Herald, Ally, Shapeshifter, Oracle, Ruler, Lover

### Philosophical (16)
Freedom, Meaning, Authenticity, Being, Essence, Time-Consciousness, Death-Awareness, Absurdity, Choice, Bad-Faith, Thrownness, Care, Dasein, Truth, Anxiety, Responsibility

### Transformation (16)
Emergence, Dissolution, Chrysalis, Crisis, Revolution, Awakening, Descent, Ascent, Integration, Differentiation, Ripening, Decay, Renewal, Breakthrough, Regression, Liminal

### Plus 128 more dimensions across moral-ethical, creative, cognitive, temporal, spatial, character, plot, theme, and style categories

## Mathematical Framework

For each semantic dimension at each temporal scale:

```
Position:      q(t) ∈ ℝ²⁴⁴         (where you are in semantic space)
Velocity:      v(t) = dq/dt        (how fast dimensions are changing)
Acceleration:  a(t) = d²v/dt²      (semantic forces - what's pulling the narrative)
Jerk:          j(t) = d³a/dt³      (how forces themselves evolve)

Integration:   ∫v dt → distance traveled per dimension
Hamiltonian:   H = T + V → energy conservation in narrative space
```

### Cross-Scale Metrics

```
Resonance:     corr(velocity_word, velocity_paragraph)
Coupling:      1 / (1 + distance(state_1, state_2))
Phase Coherence: |⟨q₁|q₂⟩| (alignment of semantic vectors)
Momentum:      avg(resonance across all scale pairs)
Complexity:    avg(distance between scale states)
```

## Usage

### Basic Streaming Analysis

```python
from HoloLoom.semantic_calculus.streaming_multi_scale import StreamingSemanticCalculus
from sentence_transformers import SentenceTransformer

# Initialize embedding function
model = SentenceTransformer('all-MiniLM-L6-v2')
embed_fn = lambda text: model.encode(text)

# Create analyzer
analyzer = StreamingSemanticCalculus(
    embed_fn=embed_fn,
    snapshot_interval=1.0  # Seconds between snapshots
)

# Define callback for snapshots
def on_snapshot(snapshot):
    print(f"Word {snapshot.word_count}:")
    print(f"  Momentum: {snapshot.narrative_momentum:.3f}")
    print(f"  Complexity: {snapshot.complexity_index:.3f}")
    print(f"  Active dimensions: {snapshot.dominant_dimensions[:3]}")

analyzer.on_snapshot(on_snapshot)

# Stream analyze text
async def word_stream():
    for word in text.split():
        yield word
        await asyncio.sleep(0.1)

async for snapshot in analyzer.stream_analyze(word_stream()):
    pass  # Snapshots handled by callback
```

### With Live Visualization

```python
from HoloLoom.semantic_calculus.visualize_streaming import RealtimeSemanticVisualizer

# Create visualizer
visualizer = RealtimeSemanticVisualizer(
    window_size=50,
    update_interval_ms=100
)

# Connect to analyzer
def on_snapshot(snapshot):
    visualizer.add_snapshot(snapshot)
    visualizer.update_all()

analyzer.on_snapshot(on_snapshot)

# Start live visualization
visualizer.start_animation()

# Analyze text (visualization updates in real-time)
async for snapshot in analyzer.stream_analyze(word_stream()):
    await asyncio.sleep(0.01)

# Save final visualization
visualizer.save_snapshot('narrative_analysis.png')
```

### Fractal Signature Analysis

```python
from HoloLoom.semantic_calculus.demo_fractal_analysis import FractalSignature

# Analyze text sample
snapshots = []
analyzer.on_snapshot(lambda s: snapshots.append(s))

async for snapshot in analyzer.stream_analyze(word_stream()):
    pass

# Compute fractal signature
signature = FractalSignature("My Novel Chapter 1")
signature.compute_from_snapshots(snapshots)
signature.print_report()

# Output:
# 📊 FRACTAL SIGNATURE: My Novel Chapter 1
# ================================================================
#    Avg Momentum:    0.687
#    Avg Complexity:  0.432
#    Momentum Variance: 0.0823
#
#    Writing Style:   SOPHISTICATED (Literary fiction)
#    Pacing Quality:  DYNAMIC (varied pacing)
#    Coherence:       HIGH (scales aligned)
```

## Visualization Panels

The live visualization shows 6 synchronized panels:

1. **Position Trajectories**: Top dimensions over time (see which themes emerge/fade)
2. **Velocity Heatmap**: Which dimensions are actively changing (hot spots = focus areas)
3. **Acceleration Spikes**: Narrative force moments (peaks = emotionally significant)
4. **Phase Portrait**: Attractors and cycles (reveals narrative structure)
5. **Scale Resonance Matrix**: Coupling between temporal scales (coherence map)
6. **Momentum Curve**: Overall flow quality over time (readers feel this!)

## Applications

### 1. Real-Time Writing Assistant

```
User types: "The meeting was frustrating and went nowhere."

🤖 AI: Low momentum (0.32) - scales diverging
💡 SUGGESTION: Simplify to maintain flow:
   "The meeting went nowhere."
   (Predicted momentum: 0.58)
```

### 2. Style Fingerprinting

```python
# Analyze writing samples
hemingway_sig = analyze_sample(hemingway_text)
# → High momentum (0.82), low complexity (0.28)

faulkner_sig = analyze_sample(faulkner_text)
# → Low momentum (0.31), high complexity (0.76)

# Match target style
your_sig = analyze_sample(your_text)
distance_to_hemingway = compute_distance(your_sig, hemingway_sig)
# → "75% match to Hemingway style"
```

### 3. Pacing Analysis

```python
# Find slow sections
for snapshot in snapshots:
    if snapshot.narrative_momentum < 0.4:
        print(f"Slow at word {snapshot.word_count}")
        print(f"Dominant: {snapshot.dominant_dimensions}")
        # → Guide revision: "Try increasing Tension or Urgency"
```

### 4. Genre Classification

Each genre has characteristic fractal signatures:
- **Thrillers**: High momentum variance (tension peaks)
- **Literary Fiction**: High complexity, moderate momentum
- **Romance**: High Warmth velocity, moderate Intensity acceleration
- **Technical Writing**: Low variance, moderate momentum, stable scales

### 5. Character Voice Consistency

```python
# Check if character dialogue matches established pattern
character_signature = analyze_dialogue(all_previous_dialogue)

new_dialogue_sig = analyze_dialogue(new_scene)

if distance(character_signature, new_dialogue_sig) > threshold:
    print("⚠️ Character voice inconsistency detected")
    print(f"Expected high Formality, but seeing low Formality")
```

## Demos

### Run Fractal Analysis Demo

```bash
cd HoloLoom/semantic_calculus
python demo_fractal_analysis.py
```

Compares Hemingway, Faulkner, Startup, Stream-of-consciousness, and Technical styles.

### Run Live Visualization Demo

```bash
python visualize_streaming.py
```

Shows real-time 6-panel visualization as text streams.

### Run Basic Streaming Demo

```bash
python streaming_multi_scale.py
```

Word-by-word analysis with console output.

## Performance

**Speed:**
- With numba JIT: ~1000 words/second per scale
- Without numba: ~100 words/second per scale
- 5 scales = ~200 words/second total (real-time for reading speed)

**Memory:**
- ~50MB baseline (244D projection matrices)
- +1MB per 1000 words of history (all scales)
- Configurable window size to limit memory

**Accuracy:**
- Semantic projection quality depends on embedding model
- Recommend: `sentence-transformers/all-mpnet-base-v2` (best quality)
- Or: `all-MiniLM-L6-v2` (fast, good quality)

## Architecture

```
StreamingSemanticCalculus
├── SemanticSpectrum (244D projection)
│   ├── EXTENDED_244_DIMENSIONS
│   └── project_vector() → Dict[dim_name, value]
│
├── ScaleWindow (per temporal scale)
│   ├── buffer: deque of tokens
│   ├── positions: deque of 244D vectors
│   ├── velocities: deque of derivatives
│   └── accelerations: deque of second derivatives
│
├── MultiScaleSnapshot
│   ├── states_by_scale: position/velocity/acceleration per scale
│   ├── resonances: cross-scale coupling metrics
│   ├── dominant_dimensions: top changing dimensions
│   ├── narrative_momentum: overall flow quality
│   └── complexity_index: scale divergence
│
└── Callbacks: async event-driven updates
```

## Theory: Why This Works

Traditional NLP operates on surface statistics. This system reveals **geometric structure of meaning**:

1. **Interpretable Dimensions**: Instead of opaque embeddings, we track motion along human-understandable axes (Warmth, Heroism, Tension, etc.)

2. **Temporal Scales**: Different scales capture different aspects:
   - Words = micro-choices (rhetoric, emotion)
   - Sentences = thoughts (logic, argument)
   - Paragraphs = beats (narrative arcs)

3. **Resonance**: When scales move together → coherent narrative flow (readers feel this as "pacing")

4. **Divergence**: When scales separate → complexity, sophistication (readers feel this as "depth")

5. **Forces**: Acceleration = semantic forces pulling the narrative (dramatic moments have high acceleration)

6. **Fractal Signature**: Each text has a unique pattern of how small-scale motion relates to large-scale motion

## Future Directions

- **Interactive editing**: Real-time feedback as you type (VSCode extension)
- **Style transfer**: "Rewrite this in Hemingway's style" → target fractal signature
- **Automated pacing**: "Insert tension here" → suggest edits that increase specific dimensions
- **Character arc tracking**: Project character embeddings through 244D space over time
- **Genre blending**: Combine fractal signatures (80% thriller + 20% romance)
- **Reading difficulty estimation**: High complexity → harder to read
- **Memorability prediction**: High acceleration spikes → memorable moments

## Citation

```bibtex
@software{hololoom_streaming_calculus,
  title={Multi-Scale Streaming Semantic Calculus},
  author={HoloLoom Contributors},
  year={2025},
  url={https://github.com/yourusername/mythRL}
}
```

## License

MIT - See LICENSE file

---

**🌊 The narrative flows like water through 244-dimensional semantic space. This system lets you see the currents. 📐**
