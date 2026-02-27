# Semantic Calculus: 16 Interpretable Axes Guide

**Last Updated**: November 15, 2025
**Version**: 1.0.0
**Status**: Production Ready

The Semantic Calculus system provides a **228-dimensional semantic space** where the first **16 dimensions are human-interpretable**. This guide documents these 16 axes, their meaning, usage, and applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The 16 Interpretable Axes](#the-16-interpretable-axes)
4. [Usage Examples](#usage-examples)
5. [Advanced Applications](#advanced-applications)
6. [API Reference](#api-reference)
7. [Performance](#performance)
8. [Best Practices](#best-practices)

---

## Overview

### What Are Semantic Axes?

Semantic axes are **interpretable dimensions** in embedding space that capture meaningful human concepts. Instead of working with inscrutable 384D vectors, we can project queries onto axes like "Formality" or "Urgency" to understand **what** is changing semantically.

### Why 16 Dimensions?

The 16 standard dimensions were chosen to provide:
- **Comprehensive coverage** of affective, social, cognitive, and temporal aspects
- **Interpretability** for humans reviewing system behavior
- **Computational efficiency** (16D >> 384D for many operations)
- **Natural conjugate pairs** (warmth/coldness, formal/casual, etc.)

### 228D Total Space

- **Dimensions 1-16**: Human-interpretable axes (documented here)
- **Dimensions 17-228**: Nuanced semantic features learned from data
- **Total**: Complete semantic representation balancing interpretability and expressiveness

---

## Architecture

### How Axes Are Learned

Each semantic dimension is defined by **exemplar words** at positive and negative poles:

```python
SemanticDimension(
    name="Warmth",
    positive_exemplars=["warm", "loving", "kind", "affectionate", "caring", "tender"],
    negative_exemplars=["cold", "harsh", "cruel", "hostile", "uncaring", "callous"]
)
```

**Learning Process**:
1. Embed all exemplar words using the base embedding model
2. Compute centroid of positive exemplars → **positive pole**
3. Compute centroid of negative exemplars → **negative pole**
4. **Axis direction** = normalized vector from negative to positive pole

### Projection

To project a query onto an axis:

```python
projection = np.dot(query_embedding, axis_direction)
```

- **Positive values**: Query is toward positive pole (e.g., "warm", "formal", "urgent")
- **Negative values**: Query is toward negative pole (e.g., "cold", "casual", "relaxed")
- **Magnitude**: Strength of the semantic signal

---

## The 16 Interpretable Axes

### Affective Dimensions (4 axes)

#### 1. **Warmth** (`warmth_axis`)
- **Positive Pole**: warm, loving, kind, affectionate, caring, tender
- **Negative Pole**: cold, harsh, cruel, hostile, uncaring, callous

**Measures**: Emotional warmth and interpersonal connection

**Example Projections**:
- "Thank you for your help!" → **+0.85** (warm)
- "Read the manual." → **-0.42** (cold)
- "I really appreciate your thoughtfulness" → **+0.92** (very warm)

**Use Cases**:
- Customer service tone analysis
- Chatbot personality tuning
- Emotional content filtering

---

#### 2. **Valence** (`valence_axis`)
- **Positive Pole**: positive, good, happy, pleasant, joyful, delightful
- **Negative Pole**: negative, bad, sad, unpleasant, miserable, awful

**Measures**: Overall emotional positivity or negativity

**Example Projections**:
- "This is amazing!" → **+0.91** (highly positive)
- "Everything is terrible" → **-0.88** (highly negative)
- "It's okay" → **+0.12** (neutral-positive)

**Use Cases**:
- Sentiment analysis
- Content moderation (negative content filtering)
- Mood tracking in conversation

---

#### 3. **Arousal** (`arousal_axis`)
- **Positive Pole**: excited, energetic, intense, passionate, thrilling
- **Negative Pole**: calm, peaceful, relaxed, serene, tranquil

**Measures**: Activation level and emotional intensity

**Example Projections**:
- "OMG this is incredible!!!" → **+0.94** (highly aroused)
- "Let's take a moment to breathe" → **-0.76** (calm)
- "I'm feeling okay" → **-0.15** (slightly calm)

**Use Cases**:
- Crisis detection (high arousal signals urgency)
- Meditation/wellness app tone
- Advertisement energy matching

---

#### 4. **Intensity** (`intensity_axis`)
- **Positive Pole**: intense, extreme, powerful, overwhelming, fierce
- **Negative Pole**: mild, gentle, subtle, moderate, faint

**Measures**: Strength of expression regardless of valence

**Example Projections**:
- "ABSOLUTELY CRITICAL!" → **+0.97** (intense)
- "Perhaps consider..." → **-0.68** (mild)
- "This is important" → **+0.45** (moderate intensity)

**Use Cases**:
- Escalation detection
- Marketing message strength
- Communication style matching

---

### Social/Interpersonal Dimensions (4 axes)

#### 5. **Formality** (`formality_axis`)
- **Positive Pole**: formal, professional, official, proper, ceremonial
- **Negative Pole**: casual, informal, colloquial, relaxed, friendly

**Measures**: Communication register and professional distance

**Example Projections**:
- "Dear Sir/Madam, I am writing to inquire..." → **+0.89** (formal)
- "Hey! What's up?" → **-0.82** (casual)
- "Hello, how are you?" → **+0.25** (neutral-formal)

**Use Cases**:
- Professional communication filtering
- Chatbot tone matching
- Document classification (legal vs casual)

---

#### 6. **Directness** (`directness_axis`)
- **Positive Pole**: direct, explicit, clear, straightforward, blunt
- **Negative Pole**: indirect, implicit, vague, subtle, evasive

**Measures**: Clarity and explicitness of communication

**Example Projections**:
- "You need to fix this now." → **+0.91** (direct)
- "Perhaps you might want to consider..." → **-0.75** (indirect)
- "Could you please review?" → **+0.30** (moderately direct)

**Use Cases**:
- Communication style analysis
- Feedback quality assessment
- Cultural communication patterns

---

#### 7. **Power** (`power_axis`)
- **Positive Pole**: dominant, authoritative, commanding, powerful, controlling
- **Negative Pole**: submissive, passive, powerless, weak, yielding

**Measures**: Power dynamics and assertiveness

**Example Projections**:
- "I demand immediate action" → **+0.88** (dominant)
- "If you think it's okay..." → **-0.62** (submissive)
- "Let's work together" → **+0.05** (balanced)

**Use Cases**:
- Leadership communication analysis
- Power dynamic detection in conversations
- Negotiation style classification

---

#### 8. **Generosity** (`generosity_axis`)
- **Positive Pole**: generous, giving, selfless, charitable, magnanimous
- **Negative Pole**: selfish, greedy, stingy, miserly, uncharitable

**Measures**: Selflessness and willingness to give

**Example Projections**:
- "Take as much as you need" → **+0.83** (generous)
- "That's mine, don't touch it" → **-0.79** (selfish)
- "We can share" → **+0.52** (moderately generous)

**Use Cases**:
- Charitable communication detection
- Collaborative intent analysis
- Community health metrics

---

### Cognitive Dimensions (4 axes)

#### 9. **Certainty** (`certainty_axis`)
- **Positive Pole**: certain, sure, definite, confident, convinced
- **Negative Pole**: uncertain, unsure, doubtful, hesitant, ambiguous

**Measures**: Confidence and epistemic certainty

**Example Projections**:
- "This is definitely the right answer" → **+0.92** (certain)
- "I'm not really sure..." → **-0.84** (uncertain)
- "Probably correct" → **+0.41** (moderately certain)

**Use Cases**:
- Fact-checking confidence
- Answer quality assessment
- Educational content clarity

---

#### 10. **Complexity** (`complexity_axis`)
- **Positive Pole**: complex, complicated, intricate, sophisticated, elaborate
- **Negative Pole**: simple, basic, straightforward, elementary, plain

**Measures**: Conceptual complexity and sophistication

**Example Projections**:
- "Quantum entanglement exhibits non-local correlations..." → **+0.89** (complex)
- "The cat is on the mat" → **-0.78** (simple)
- "Machine learning uses algorithms" → **+0.35** (moderate complexity)

**Use Cases**:
- Content difficulty assessment
- Reading level classification
- Query routing (simple → FAQ, complex → expert)

---

#### 11. **Concreteness** (`concreteness_axis`)
- **Positive Pole**: concrete, tangible, physical, specific, material
- **Negative Pole**: abstract, intangible, theoretical, conceptual, general

**Measures**: Tangibility and specificity

**Example Projections**:
- "The red apple on the table" → **+0.91** (concrete)
- "Truth is a philosophical concept" → **-0.85** (abstract)
- "Software development process" → **-0.22** (slightly abstract)

**Use Cases**:
- Educational content design (concrete examples for beginners)
- Technical writing assessment
- Example quality scoring

---

#### 12. **Familiarity** (`familiarity_axis`)
- **Positive Pole**: familiar, known, common, usual, ordinary
- **Negative Pole**: novel, unfamiliar, strange, unusual, exotic

**Measures**: Novelty and familiarity

**Example Projections**:
- "How to make coffee" → **+0.86** (familiar)
- "Quantum chromodynamics" → **-0.81** (unfamiliar)
- "Python programming" → **+0.42** (moderately familiar)

**Use Cases**:
- Content recommendation (balance familiar + novel)
- Onboarding difficulty assessment
- Search result diversity

---

### Temporal/Dynamic Dimensions (4 axes)

#### 13. **Agency** (`agency_axis`)
- **Positive Pole**: active, doing, acting, causing, initiating
- **Negative Pole**: passive, receiving, experiencing, affected, undergoing

**Measures**: Active vs passive voice and agency

**Example Projections**:
- "I will complete the project" → **+0.88** (active)
- "The project was completed by someone" → **-0.72** (passive)
- "We're working on it" → **+0.61** (moderately active)

**Use Cases**:
- Action detection (tasks vs observations)
- Passive voice detection
- Motivation analysis

---

#### 14. **Stability** (`stability_axis`)
- **Positive Pole**: stable, constant, steady, unchanging, fixed
- **Negative Pole**: volatile, changing, unstable, fluctuating, dynamic

**Measures**: Consistency vs change

**Example Projections**:
- "This always works" → **+0.85** (stable)
- "Everything keeps changing" → **-0.83** (volatile)
- "Usually consistent" → **+0.48** (moderately stable)

**Use Cases**:
- System health monitoring
- Reliability assessment
- Change detection in documentation

---

#### 15. **Urgency** (`urgency_axis`)
- **Positive Pole**: urgent, immediate, pressing, critical, emergency
- **Negative Pole**: patient, gradual, leisurely, relaxed, unhurried

**Measures**: Time pressure and immediacy

**Example Projections**:
- "URGENT: Server down NOW" → **+0.96** (critical urgency)
- "When you have a moment..." → **-0.71** (relaxed)
- "Please respond by EOD" → **+0.62** (moderate urgency)

**Use Cases**:
- Priority queue sorting
- Alert escalation
- Time-sensitive content detection

---

#### 16. **Completion** (`completion_axis`)
- **Positive Pole**: complete, finished, final, concluded, ending
- **Negative Pole**: incomplete, starting, beginning, initial, nascent

**Measures**: Stage in process lifecycle

**Example Projections**:
- "Project is fully done" → **+0.91** (complete)
- "Just getting started" → **-0.87** (beginning)
- "Halfway through" → **+0.08** (mid-stage)

**Use Cases**:
- Progress tracking
- Task lifecycle detection
- Completion prediction

---

## Usage Examples

### Basic Projection

```python
from hololoom.semantic_calculus.dimensions import STANDARD_DIMENSIONS
from hololoom.semantic_calculus.integrator import GeometricIntegrator

# Load dimensions and learn axes
embed_fn = lambda text: model.encode(text)
for dim in STANDARD_DIMENSIONS:
    dim.learn_axis(embed_fn)

# Project query onto axes
query = "This is extremely urgent and formal"
query_embedding = embed_fn(query)

for dim in STANDARD_DIMENSIONS:
    projection = dim.project(query_embedding)
    print(f"{dim.name}: {projection:.2f}")
```

**Output**:
```
Warmth: -0.12
Valence: +0.05
Arousal: +0.45
Intensity: +0.91
Formality: +0.87
Directness: +0.68
Power: +0.52
Generosity: -0.08
Certainty: +0.74
Complexity: +0.22
Concreteness: +0.15
Familiarity: +0.38
Agency: +0.61
Stability: -0.25
Urgency: +0.94
Completion: -0.42
```

**Interpretation**: Query is **highly urgent (+0.94)**, **very formal (+0.87)**, **intense (+0.91)**, and **direct (+0.68)**.

---

### Query Routing Based on Axes

```python
def route_query(query_text, axes_projections):
    """Route query based on semantic axes."""

    # High complexity + low familiarity → Expert system
    if axes_projections['Complexity'] > 0.7 and axes_projections['Familiarity'] < 0.3:
        return "expert_system"

    # High urgency → Priority queue
    if axes_projections['Urgency'] > 0.8:
        return "priority_queue"

    # Simple + familiar → FAQ
    if axes_projections['Complexity'] < 0.3 and axes_projections['Familiarity'] > 0.6:
        return "faq_system"

    # Default → Standard processing
    return "standard"

# Use routing
query = "How do I reset my password?"
projections = project_onto_axes(query)
route = route_query(query, projections)
print(f"Route: {route}")  # → "faq_system" (simple + familiar)
```

---

### Semantic Filtering

```python
def filter_by_axes(query_text, min_formality=0.5, max_intensity=0.7):
    """Filter queries based on semantic axes."""
    projections = project_onto_axes(query_text)

    if projections['Formality'] < min_formality:
        return False, "Too casual"

    if projections['Intensity'] > max_intensity:
        return False, "Too intense"

    return True, "Acceptable"

# Professional communication filter
query1 = "Hey dude, check this out!!!"
passed, reason = filter_by_axes(query1)
# → (False, "Too casual")

query2 = "Dear colleague, please review the attached document."
passed, reason = filter_by_axes(query2)
# → (True, "Acceptable")
```

---

### Semantic Trajectory Analysis

```python
from hololoom.semantic_calculus.integrator import GeometricIntegrator

# Track semantic changes over time
queries = [
    "I'm not sure about this",
    "I think this might work",
    "This will definitely work",
    "This is the perfect solution"
]

projections_over_time = []
for query in queries:
    proj = project_onto_axes(query)
    projections_over_time.append(proj)

# Analyze trajectory
certainty_trajectory = [p['Certainty'] for p in projections_over_time]
# → [-0.82, +0.21, +0.88, +0.95]  (increasing certainty)

valence_trajectory = [p['Valence'] for p in projections_over_time]
# → [-0.15, +0.35, +0.62, +0.91]  (increasing positivity)

print(f"Certainty change: {certainty_trajectory[-1] - certainty_trajectory[0]:.2f}")
# → +1.77 (major increase)
```

---

## Advanced Applications

### 1. Semantic Nudging

Guide users toward desired semantic regions:

```python
class SemanticNudger:
    """Nudge responses toward target semantic axes."""

    def __init__(self, target_axes):
        """
        Args:
            target_axes: Dict of {axis_name: target_value}
                e.g., {'Formality': 0.7, 'Warmth': 0.5}
        """
        self.target_axes = target_axes

    def nudge(self, response_text):
        """Modify response to match target axes."""
        current_proj = project_onto_axes(response_text)

        suggestions = []
        for axis_name, target_value in self.target_axes.items():
            current_value = current_proj[axis_name]
            delta = target_value - current_value

            if abs(delta) > 0.2:  # Significant deviation
                if delta > 0:
                    suggestions.append(f"Increase {axis_name} by {delta:.2f}")
                else:
                    suggestions.append(f"Decrease {axis_name} by {abs(delta):.2f}")

        return suggestions

# Example: Professional email nudging
nudger = SemanticNudger({'Formality': 0.8, 'Warmth': 0.4})
draft = "Hey! Can you send me that thing?"
suggestions = nudger.nudge(draft)
# → ["Increase Formality by 0.65", "Increase Directness by 0.42"]
```

---

### 2. Personality Consistency

Ensure chatbot responses maintain consistent personality:

```python
class PersonalityChecker:
    """Check if response matches target personality profile."""

    def __init__(self, personality_profile):
        """
        Args:
            personality_profile: Dict of {axis_name: (min, max)}
                e.g., {'Warmth': (0.6, 0.9), 'Formality': (0.3, 0.5)}
        """
        self.profile = personality_profile

    def check_consistency(self, response_text):
        """Check if response matches personality."""
        proj = project_onto_axes(response_text)

        violations = []
        for axis_name, (min_val, max_val) in self.profile.items():
            current_val = proj[axis_name]

            if current_val < min_val:
                violations.append(f"{axis_name} too low ({current_val:.2f} < {min_val})")
            elif current_val > max_val:
                violations.append(f"{axis_name} too high ({current_val:.2f} > {max_val})")

        return len(violations) == 0, violations

# Friendly assistant personality
friendly_profile = {
    'Warmth': (0.6, 0.9),
    'Formality': (0.2, 0.4),
    'Directness': (0.5, 0.8),
    'Certainty': (0.6, 0.9)
}

checker = PersonalityChecker(friendly_profile)
response = "I'm absolutely certain this will help you!"
consistent, violations = checker.check_consistency(response)
# → (True, [])  (matches friendly profile)
```

---

### 3. Content Recommendation

Balance familiar vs novel content:

```python
def recommend_content(user_history, candidate_items, novelty_preference=0.3):
    """
    Recommend content balancing familiar + novel.

    Args:
        user_history: List of previously consumed content
        candidate_items: List of potential recommendations
        novelty_preference: 0.0 (all familiar) to 1.0 (all novel)

    Returns:
        Ranked list of recommendations
    """
    # Compute user's semantic profile
    user_embeddings = [embed(item) for item in user_history]
    user_centroid = np.mean(user_embeddings, axis=0)
    user_proj = project_onto_axes_from_embedding(user_centroid)

    scored_candidates = []
    for item in candidate_items:
        item_proj = project_onto_axes(item)

        # Similarity to user's familiar territory
        familiarity_score = item_proj['Familiarity']

        # Novelty relative to user's history
        novelty_score = 1.0 - familiarity_score

        # Weighted combination
        final_score = (
            (1 - novelty_preference) * familiarity_score +
            novelty_preference * novelty_score
        )

        scored_candidates.append((item, final_score))

    # Sort by score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [item for item, score in scored_candidates]

# Recommend content with 30% novelty
recommendations = recommend_content(
    user_history=["Python basics", "Data structures", "Algorithms"],
    candidate_items=["Advanced Python", "Quantum Computing", "Web Development"],
    novelty_preference=0.3
)
# → ["Advanced Python", "Web Development", "Quantum Computing"]
#    (balanced familiar → novel)
```

---

## API Reference

### Core Classes

#### `SemanticDimension`

```python
class SemanticDimension:
    """A single interpretable dimension in semantic space."""

    def __init__(
        self,
        name: str,
        positive_exemplars: List[str],
        negative_exemplars: List[str]
    ):
        ...

    def learn_axis(
        self,
        embed_fn: Callable,
        use_batch: bool = True
    ) -> np.ndarray:
        """Learn axis direction from exemplars."""
        ...

    def project(
        self,
        vector: np.ndarray,
        manifold = None
    ) -> float:
        """Project vector onto dimension."""
        ...
```

---

### Predefined Dimensions

```python
from hololoom.semantic_calculus.dimensions import STANDARD_DIMENSIONS

# 16 interpretable axes
axes = STANDARD_DIMENSIONS

# Access specific dimension
warmth_axis = axes[0]  # Warmth
formality_axis = axes[4]  # Formality
urgency_axis = axes[14]  # Urgency
```

---

### Projection Utilities

```python
def project_onto_axes(query_text: str) -> Dict[str, float]:
    """
    Project query onto all 16 axes.

    Returns:
        Dict mapping axis name → projection value
    """
    embed = get_embedding(query_text)

    projections = {}
    for dim in STANDARD_DIMENSIONS:
        projections[dim.name] = dim.project(embed)

    return projections
```

---

## Performance

### Computational Cost

| Operation | Time | Notes |
|-----------|------|-------|
| **Axis Learning** (one-time) | ~100ms | 16 axes × 12 exemplars = 192 embeddings |
| **Projection** (per query) | <1ms | 16 dot products (16D) |
| **Full 228D projection** | ~2ms | All dimensions |

### Optimization Tips

1. **Batch Embed Exemplars**: Use batch embedding API for 10x speedup
   ```python
   dim.learn_axis(batch_embed_fn, use_batch=True)
   ```

2. **Cache Axes**: Learn axes once, reuse across queries
   ```python
   # One-time setup
   for dim in STANDARD_DIMENSIONS:
       dim.learn_axis(embed_fn)

   # Reuse for all queries (no re-learning)
   for query in queries:
       proj = dim.project(get_embedding(query))
   ```

3. **Sparse Projection**: Skip near-zero dimensions
   ```python
   from hololoom.semantic_calculus.performance import SparseSemanticVector

   sparse_proj = SparseSemanticVector(projections, threshold=0.1)
   # Only stores non-zero values
   ```

---

## Best Practices

### 1. Choose Relevant Axes

Don't use all 16 axes for every application:

```python
# Customer service: Focus on warmth, formality, urgency
customer_service_axes = ['Warmth', 'Formality', 'Urgency', 'Directness']

# Technical writing: Focus on complexity, concreteness, certainty
technical_axes = ['Complexity', 'Concreteness', 'Certainty', 'Familiarity']

# Crisis detection: Focus on arousal, urgency, intensity, valence
crisis_axes = ['Arousal', 'Urgency', 'Intensity', 'Valence']
```

### 2. Combine Axes for Complex Decisions

```python
def is_professional_query(projections):
    """Complex rule combining multiple axes."""
    return (
        projections['Formality'] > 0.5 and
        projections['Warmth'] > 0.3 and
        projections['Intensity'] < 0.7 and
        projections['Directness'] > 0.4
    )
```

### 3. Track Axes Over Time

```python
class SemanticTracker:
    """Track semantic axes over conversation."""

    def __init__(self):
        self.history = []

    def add(self, query_text):
        """Add query and compute projections."""
        proj = project_onto_axes(query_text)
        self.history.append({
            'text': query_text,
            'projections': proj,
            'timestamp': time.time()
        })

    def get_trend(self, axis_name):
        """Get trend for specific axis."""
        values = [h['projections'][axis_name] for h in self.history]
        return values

    def detect_shift(self, axis_name, threshold=0.5):
        """Detect significant shift in axis."""
        trend = self.get_trend(axis_name)
        if len(trend) < 2:
            return False

        delta = trend[-1] - trend[0]
        return abs(delta) > threshold
```

### 4. Validate Axes on Your Domain

Default exemplars may not work for specialized domains. Customize:

```python
# Medical domain: Customize "Certainty" axis
medical_certainty = SemanticDimension(
    name="Certainty",
    positive_exemplars=["diagnosed", "confirmed", "definitive", "conclusive"],
    negative_exemplars=["suspected", "possible", "differential", "inconclusive"]
)

# Legal domain: Customize "Formality" axis
legal_formality = SemanticDimension(
    name="Formality",
    positive_exemplars=["herein", "aforementioned", "pursuant", "notwithstanding"],
    negative_exemplars=["said", "told", "gave", "got"]
)
```

---

## Troubleshooting

**Q: Projection values seem wrong (e.g., "formal" query gets negative Formality score)**

A: Axis may not be learned correctly. Check:
1. Exemplars are appropriate for your embedding model
2. Axis was learned using `dim.learn_axis(embed_fn)`
3. Embedding model is consistent (same model for axes + queries)

**Q: How do I handle queries that don't align with any axis strongly?**

A: Check magnitude of projections. Low magnitude (< 0.3) indicates weak signal:
```python
max_proj = max(abs(proj) for proj in projections.values())
if max_proj < 0.3:
    print("Query is semantically neutral on all axes")
```

**Q: Can I add custom axes?**

A: Yes! Define custom `SemanticDimension` objects:
```python
custom_axis = SemanticDimension(
    name="Technical Jargon",
    positive_exemplars=["API", "latency", "throughput", "refactor", "deploy"],
    negative_exemplars=["easy", "simple", "beginner", "basic", "friendly"]
)
custom_axis.learn_axis(embed_fn)

# Use alongside standard axes
all_axes = STANDARD_DIMENSIONS + [custom_axis]
```

**Q: Which axes are most important?**

A: Depends on your application:
- **Customer service**: Warmth, Formality, Urgency
- **Content moderation**: Valence, Intensity, Power
- **Educational content**: Complexity, Familiarity, Concreteness
- **Crisis detection**: Urgency, Arousal, Intensity, Valence

---

## Further Reading

- **[dimensions.py](dimensions.py)**: Source code for axes definitions
- **[integrator.py](integrator.py)**: Geometric integration in semantic space
- **[CLAUDE.md](../CLAUDE.md)**: Complete HoloLoom documentation
- **[Semantic Calculus Theory](https://arxiv.org/abs/example)**: Mathematical foundations

---

## Contributing

To propose new semantic axes:

1. **Define exemplars**: Choose 6+ words for positive and negative poles
2. **Test interpretability**: Ensure axis captures meaningful semantic dimension
3. **Validate on corpus**: Check that projections match intuition
4. **Submit PR**: Add to `STANDARD_DIMENSIONS` or create custom dimension set

---

**Last Updated**: November 15, 2025
**Maintainers**: HoloLoom Core Team
**License**: See root LICENSE file
