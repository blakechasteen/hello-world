# Semantic Analysis Visualizations Guide

**Location**: `demos/output/`
**Generated**: 2025-10-27

## Overview

These visualizations demonstrate HoloLoom's semantic calculus system - mathematical analysis of how meaning flows and changes through text.

---

## New Visualizations (Just Generated)

### 1. **semantic_trajectory_3d.png** (248KB)
**What it shows**: 3D visualization of how text moves through semantic space

**Components**:
- **Top**: 3D scatter plot showing the "path" words take through meaning-space
  - Points colored by semantic velocity (speed of meaning change)
  - Words labeled at key points
  - Path shown as connected line
- **Bottom Left**: Semantic velocity over time
  - Shows how fast meaning changes at each word
  - Peaks indicate rapid semantic shifts
- **Bottom Right**: Semantic acceleration over time
  - Shows when meaning changes direction
  - High values = topic shifts or conceptual pivots

**Interpretation**:
- Smooth curves = coherent flow
- Sharp turns = topic shifts
- Color gradient = speed of semantic change

---

### 2. **flow_metrics.png** (189KB)
**What it shows**: Detailed mathematical metrics for each word

**4 Charts**:
1. **Velocity (Top Left)**: Rate of meaning change per word
   - Bars show speed
   - Red line = average
   - High bars = words that shift meaning significantly

2. **Acceleration (Top Right)**: Change in semantic direction
   - Shows when conversation pivots
   - Blue line = average
   - Peaks = moments of conceptual shift

3. **Curvature (Bottom Left)**: How much the semantic path "bends"
   - High curvature = sharp semantic turns
   - Green line = average
   - Indicates complexity of semantic trajectory

4. **Kinetic Energy (Bottom Right)**: Semantic momentum
   - Energy = velocity × mass
   - Shows "force" of meaning change
   - High energy = powerful semantic shifts

**Use Cases**:
- Identify key transition words
- Find topic shift points
- Measure conversation complexity

---

### 3. **semantic_dimensions.png** (146KB)
**What it shows**: How 16 interpretable semantic dimensions change over time

**Two Visualizations**:

**Top Heatmap**:
- Rows = 16 semantic dimensions (Warmth, Formality, Certainty, etc.)
- Columns = Words in sequence
- Colors = Activation level
  - Red = High activation
  - Blue = Low activation
- Shows which semantic "flavors" are active at each word

**Bottom Bar Chart**: Dominant Dimensions
- Top 8 dimensions with highest velocity
- Red bars = Dimension increasing
- Blue bars = Dimension decreasing
- Shows which aspects of meaning are changing most

**Dimensions Tracked**:
- Warmth (warm ↔ cold)
- Formality (formal ↔ informal)
- Certainty (certain ↔ uncertain)
- Intensity (intense ↔ mild)
- Directness (direct ↔ indirect)
- Arousal (excited ↔ calm)
- Dominance (dominant ↔ submissive)
- Complexity (complex ↔ simple)
- ... and 8 more

**Interpretation**:
- Heatmap patterns show semantic texture
- Velocity chart shows dynamic changes
- Red/blue split shows direction of change

---

### 4. **ethical_analysis.png** (187KB)
**What it shows**: Ethical evaluation of the text

**Two Charts**:

**Left**: Ethical Metrics Over Trajectory
- Green line = Virtue score (ethical quality)
- Red line = Manipulation score (manipulative patterns)
- Orange threshold line = Ethical boundary
- Shows how ethics evolve through text

**Right**: Overall Assessment
- 3 bars showing:
  1. Mean Virtue (higher = more ethical)
  2. Max Manipulation (lower = less manipulative)
  3. Is Ethical (1.0 = passes, 0.0 = fails)

**Ethical Frameworks**:
- Compassionate: Empathy, understanding, kindness
- Therapeutic: Safety, healing, non-judgment
- Scientific: Accuracy, clarity, honesty

**Use Cases**:
- Content moderation
- Conversation quality assessment
- Detect manipulation patterns
- Ethical AI evaluation

---

### 5. **pattern_cards_comparison.png** (80KB)
**What it shows**: Configuration differences between BARE/FAST/FUSED modes

**4 Comparison Charts**:

1. **Semantic Dimensions**: How many dimensions each card uses
   - Bare: 0 (disabled)
   - Fast: 16D
   - Fused: 32D

2. **Embedding Scales**: Number of multi-scale embeddings
   - Bare: 1 scale (96D)
   - Fast: 2 scales (96D, 192D)
   - Fused: 3 scales (96D, 192D, 384D)

3. **Tools Enabled**: Number of analysis tools
   - Bare: 2 tools (minimal)
   - Fast: 4 tools (balanced)
   - Fused: 6 tools (maximum)

4. **Target Latency**: Expected response time
   - Bare: 50ms (fastest)
   - Fast: 200ms (balanced)
   - Fused: 1000ms (thorough)

**Interpretation**:
- Shows speed/capability tradeoffs
- Visual guide to pattern card selection
- Helps choose appropriate mode for use case

---

## Existing Visualizations

### semantic_flow_3d.png (282KB)
Original 3D semantic flow demo on philosophical text
- Text: "I think therefore I am a conscious being"
- Shows trajectory through semantic space
- Velocity vectors included

### flow_metrics.png (189KB)
Detailed metrics for conversation analysis
- 4-panel layout showing all semantic derivatives
- Word-by-word breakdown

### conversation_flow_3d.png (451KB)
Analysis of conversation with topic shifts
- Shows how conversations meander through meaning space
- Topic shifts visible as sharp turns

### semantic_dimensions.png (146KB)
Heatmap of semantic dimensions over conversation
- 16 dimensions tracked
- Shows semantic texture of dialogue

### ethical_analysis.png (187KB)
Ethical evaluation charts
- Virtue/manipulation tracking
- Overall ethical assessment

---

## Key Metrics Explained

### Velocity (Speed)
- **Meaning**: Rate at which meaning changes
- **Units**: Embedding distance per word
- **High values**: Rapid semantic shifts, topic changes
- **Low values**: Stable, coherent meaning

### Acceleration
- **Meaning**: Change in semantic direction
- **Units**: Change in velocity per word
- **High values**: Pivots, turns in meaning
- **Low values**: Straight semantic path

### Curvature
- **Meaning**: How much the semantic path bends
- **Formula**: |velocity × acceleration| / speed³
- **High values**: Sharp semantic turns
- **Low values**: Straight semantic trajectory

### Kinetic Energy
- **Meaning**: Semantic "momentum"
- **Formula**: ½ × mass × velocity²
- **Interpretation**: Force of meaning change

---

## How to Use These Visualizations

### For Analysis
1. **Start with 3D trajectory** - Get overall picture
2. **Check flow metrics** - Identify key transition points
3. **Examine dimensions** - Understand semantic texture
4. **Review ethics** - Assess quality/manipulation

### For Debugging
1. **Velocity spikes** → Check for inappropriate transitions
2. **High curvature** → Verify topic shifts are intentional
3. **Low virtue scores** → Review ethical concerns
4. **Dimension patterns** → Confirm semantic coherence

### For Research
1. **Compare trajectories** → Different rhetorical styles
2. **Track dimensions** → Semantic features of text types
3. **Measure ethics** → Conversation quality patterns
4. **Pattern cards** → Configuration impact on analysis

---

## Technical Details

### Analysis Configuration
```python
# Embedding
Model: all-MiniLM-L6-v2
Dimensions: 384D → reduced to 3D for visualization

# Semantic Calculus
dt: 1.0 (time step)
Scales: [96, 192, 384] for FUSED mode
Dimensions: 16 semantic axes

# Ethical Framework
Framework: Compassionate Communication
Thresholds: 0.5 for manipulation detection
```

### Generated With
```bash
python demos/semantic_analysis_visualizations.py
```

### Example Text Analyzed
```
"Thompson Sampling balances exploration and exploitation using Bayesian
inference The algorithm maintains posterior distributions over reward
parameters By sampling from these distributions it naturally explores
uncertain options"
```

### Metrics Summary
- **Average velocity**: 0.6572
- **Average acceleration**: 0.5894
- **Total distance**: 30.5337 units
- **Mean virtue**: -15.012
- **Max manipulation**: 0.022

---

## Customization

### Generate Your Own Visualizations

```python
from demos.semantic_analysis_visualizations import main
import asyncio

# Edit the text in main() function
asyncio.run(main())
```

### Analyze Different Text

```python
text = "Your text here..."
words = text.split()

# Create analyzer
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
analyzer = create_semantic_analyzer(embed_fn)

# Analyze
result = analyzer.analyze_text(text)
```

---

## Further Reading

- [Semantic Calculus Docs](../../docs/SEMANTIC_CALCULUS_EXPOSURE.md)
- [Pattern Cards Guide](../../HoloLoom/cards/README.md)
- [Demo Code](../semantic_analysis_visualizations.py)
- [Test Suite](../../tests/test_semantic_calculus_mcp.py)

---

**Generated**: 2025-10-27
**System**: HoloLoom Semantic Calculus v1.0
**All visualizations saved**: `demos/output/`
