# HoloLoom Writing System

**Universal content generation - memory becomes prose.**

## Philosophy

> "Great writing isn't written, it's refined."

The HoloLoom Writing System transforms memory context into polished prose through multi-pass refinement. Just as HoloLoom's recursive learning improves decision-making over time, the writing system improves content quality through iterative refinement.

## Quick Start

### Simple API

```python
from hololoom.writing import write
from hololoom.documentation.types import MemoryShard

# Create memory context
memories = [
    MemoryShard(
        content="Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff",
        metadata={'relevance': 0.95, 'topic': 'reinforcement_learning'}
    ),
    MemoryShard(
        content="It samples from posterior distributions to balance exploration and exploitation",
        metadata={'relevance': 0.88, 'topic': 'bayesian_methods'}
    )
]

# Generate content (automatic mode detection + refinement)
content = await write(
    query="What is Thompson Sampling?",
    memories=memories,
    refine=True  # Enable 3-pass refinement
)

print(content)
```

**Output:**
```markdown
# What is Thompson Sampling?

Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff.
This fundamental concept in reinforcement learning provides an elegant solution to
a common problem.

The method works by sampling from posterior distributions. This sampling approach
balances exploration of uncertain options with exploitation of known good choices.
Unlike epsilon-greedy strategies, Thompson Sampling adapts naturally to uncertainty.

In essence, this Bayesian framework provides a principled way to handle the
exploration-exploitation tradeoff in decision-making systems.
```

### Advanced Usage

```python
from hololoom.writing import Writer, Composer
from hololoom.writing.core import WritingContext, WritingMode, StyleGuide
from hololoom.writing.modes import NarrativeWriter
from hololoom.writing.refinement import EleganceRefiner

# Configure writer with custom settings
mode_writers = {
    WritingMode.NARRATIVE: NarrativeWriter()
}

refiners = {
    RefinementStrategy.ELEGANCE: EleganceRefiner()
}

composer = Composer(refiners=refiners)
writer = Writer(mode_writers=mode_writers, composer=composer)

# Create context
context = WritingContext(
    query="Explain the ELEGANCE refinement strategy",
    memories=memories,
    mode=WritingMode.NARRATIVE,
    style=StyleGuide.TUFTE  # Clarity-first
)

# Generate with full control
result = await writer.write(context, refine=True)

print(result.summary())
# Output: Generated narrative content with 3 refinement passes
#         Quality: 0.67 → 0.91 (+0.24 improvement)
#         Style: tufte, Confidence: 0.94

print(result.quality_trajectory)
# Output: [0.67, 0.82, 0.91] - improving with each pass
```

## Architecture

### Writing Pipeline

```
Query + Memories
    ↓
1. MODE DETECTION (if AUTO)
    - Analyze query for mode indicators
    - Check memory structure
    - Select: NARRATIVE, TECHNICAL, CREATIVE, etc.
    ↓
2. STYLE DETECTION (if AUTO)
    - Detect audience level
    - Match to mode defaults
    - Select: TUFTE, ACADEMIC, CASUAL, etc.
    ↓
3. INITIAL DRAFT GENERATION
    - Use mode-specific writer
    - Structure: setup → development → conclusion
    - Incorporate memory context
    ↓
4. MULTI-PASS REFINEMENT (if enabled)
    - Pass 1: Clarity (make understandable)
    - Pass 2: Simplicity (remove unnecessary)
    - Pass 3: Beauty (add grace)
    ↓
5. QUALITY SCORING
    - Clarity: sentence structure
    - Completeness: content depth
    - Coherence: logical flow
    - Accuracy: memory alignment
    - Conciseness: signal-to-noise
    - Elegance: stylistic quality
    ↓
WritingResult (content + metadata)
```

## Writing Modes

| Mode | Description | Default Style | Use Case |
|------|-------------|---------------|----------|
| **NARRATIVE** | Story/explanation | TUFTE | "Explain X", "What is Y?" |
| **TECHNICAL** | Documentation | TECHNICAL | "How to implement X" |
| **CREATIVE** | Fiction/poetry | CREATIVE | "Write a story about X" |
| **ANALYSIS** | Report/analysis | ACADEMIC | "Analyze X", "Compare Y and Z" |
| **DIALOGUE** | Conversational | CASUAL | "Discuss X" |
| **CODE_DOC** | Code documentation | TECHNICAL | "Document this function" |

Modes are auto-detected based on query patterns if not specified.

## Refinement Strategies

### ELEGANCE (Default for Narrative/Creative)

Three-pass refinement inspired by good writing principles:

**Pass 1: CLARITY** - Make it understandable
- Break up long sentences (>35 words)
- Replace jargon with plain language
- Add paragraph structure

**Pass 2: SIMPLICITY** - Remove unnecessary
- Cut filler words (very, really, actually, etc.)
- Eliminate redundant phrases ("in order to" → "to")
- Remove excessive modifiers

**Pass 3: BEAUTY** - Add grace
- Convert passive voice to active
- Vary sentence openings
- Strengthen weak verbs ("make use of" → "use")
- Polish transitions

Example trajectory:
```
Initial:  "In order to utilize Thompson Sampling, one must really
           understand that it was designed to facilitate exploration."

Pass 1:   "To use Thompson Sampling, one must understand that it was
           designed to help exploration."

Pass 2:   "To use Thompson Sampling, understand it helps exploration."

Pass 3:   "Thompson Sampling helps exploration through Bayesian inference."
```

Quality: 0.62 → 0.75 → 0.84 → 0.92

### VERIFY (For Technical/Analysis)

Three-pass verification (future implementation):

**Pass 1: ACCURACY** - Verify claims
- Check against memory context
- Flag unsupported statements
- Add citations

**Pass 2: COMPLETENESS** - Fill gaps
- Identify missing information
- Add necessary context
- Balance depth

**Pass 3: CONSISTENCY** - Ensure coherence
- Fix contradictions
- Align terminology
- Verify logic flow

## Style Guides

| Style | Characteristics | Best For |
|-------|----------------|----------|
| **TUFTE** | Clarity-first, minimal, data-dense | Explanations, documentation |
| **ACADEMIC** | Formal, structured, cited | Research, reports |
| **CASUAL** | Conversational, simple, direct | Chat, ELI5 |
| **TECHNICAL** | Precise, jargon-appropriate, detailed | API docs, specs |
| **CREATIVE** | Expressive, varied, stylistic | Stories, essays |

## Quality Scoring

Content quality is scored across six dimensions:

```python
quality_scores = {
    'clarity': 0.85,       # Sentence structure, readability
    'completeness': 0.78,  # Content depth vs. query
    'coherence': 0.92,     # Logical flow, structure
    'accuracy': 0.88,      # Memory alignment
    'conciseness': 0.81,   # Signal-to-noise ratio
    'elegance': 0.74       # Stylistic quality
}

# Overall: weighted average based on mode
overall_quality = 0.83
```

Weights vary by mode:
- **Technical**: Emphasizes accuracy (30%) and completeness (25%)
- **Creative**: Emphasizes elegance (30%) and coherence (25%)
- **Analysis**: Emphasizes accuracy (30%) and clarity (30%)

## Integration with HoloLoom

### With Weaving Orchestrator

```python
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.writing import write
from hololoom.documentation.types import Query

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    # Retrieve memory context
    spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))

    # Generate narrative from retrieved memories
    content = await write(
        query=spacetime.query_text,
        memories=spacetime.trace.retrieved_context,
        mode='narrative',
        style='tufte',
        refine=True
    )

    print(content)
```

### With Recursive Learning

```python
from hololoom.recursive import FullLearningEngine
from hololoom.writing import write

async with FullLearningEngine(cfg=config, shards=shards) as engine:
    # Weave with learning
    spacetime = await engine.weave(query, enable_refinement=True)

    # Generate content from learned context
    content = await write(
        query.text,
        spacetime.trace.retrieved_context,
        refine=True  # Uses same refinement philosophy
    )
```

### With Synthesis (Training Data)

```python
from hololoom.writing import write
from hololoom.synthesis import DataSynthesizer

# Generate high-quality examples
content = await write(query, memories, mode='narrative', refine=True)

# Synthesize into training data
synthesizer = DataSynthesizer()
examples = synthesizer.synthesize_from_text(content, query)
synthesizer.export_jsonl(examples, 'training_data.jsonl')
```

## API Reference

### Primary API

#### `write(query, memories, mode=None, style=None, refine=True, format='markdown')`

Generate content from memory context.

**Args:**
- `query` (str): What to write about
- `memories` (List[MemoryShard]): Memory context
- `mode` (Optional[str]): Writing mode - auto-detected if None
- `style` (Optional[str]): Style guide - auto-detected if None
- `refine` (bool): Enable multi-pass refinement (default: True)
- `format` (str): Output format (default: 'markdown')

**Returns:**
- `str`: Generated content

#### `write_batch(queries, memories, mode=None, style=None, refine=True)`

Batch writing with shared context.

**Args:**
- `queries` (List[str]): List of queries
- `memories` (List[MemoryShard]): Shared memory context
- Other args same as `write()`

**Returns:**
- `List[str]`: Generated content for each query

#### `refine_text(text, query="Refine this text", passes=3, strategy='elegance')`

Refine existing text.

**Args:**
- `text` (str): Text to refine
- `query` (str): Context query
- `passes` (int): Number of refinement passes
- `strategy` (str): Refinement strategy

**Returns:**
- `str`: Refined text

### Advanced API

#### `Writer`

Main writer class.

```python
writer = Writer(
    mode_writers: Dict[WritingMode, ModeWriterProtocol],
    composer: ComposerProtocol
)

result = await writer.write(context, refine=True)
```

**Methods:**
- `write(context, refine=True)` → WritingResult
- `detect_mode(query, memories)` → WritingMode
- `detect_style(query, memories, mode)` → StyleGuide

#### `Composer`

Multi-pass refinement orchestrator.

```python
composer = Composer(refiners: Dict[RefinementStrategy, RefinerProtocol])

result = await composer.compose(
    initial_draft,
    context,
    strategy=RefinementStrategy.ELEGANCE,
    max_passes=3,
    quality_threshold=0.9
)
```

**Methods:**
- `compose(...)` → WritingResult
- `score_quality(text, context)` → (float, Dict[str, float])

#### `WritingResult`

Result of content generation.

```python
@dataclass
class WritingResult:
    content: str
    mode: WritingMode
    style: StyleGuide
    quality_score: float
    confidence: float
    refinement_passes: List[RefinementPass]
    metadata: Dict[str, Any]

    @property
    def quality_trajectory(self) -> List[float]

    @property
    def total_passes(self) -> int

    def summary(self) -> str
```

## Examples

### Example 1: Simple Explanation

```python
from hololoom.writing import write
from hololoom.documentation.types import MemoryShard

memories = [
    MemoryShard(
        content="Multi-armed bandits balance exploration and exploitation",
        metadata={'relevance': 0.9}
    )
]

content = await write("Explain multi-armed bandits", memories)
```

### Example 2: Technical Documentation

```python
content = await write(
    "How to implement Thompson Sampling?",
    memories,
    mode='technical',
    style='technical',
    refine=True
)
```

### Example 3: Creative Writing

```python
content = await write(
    "Write a story about an AI learning to explore",
    memories,
    mode='creative',
    style='creative',
    refine=True
)
```

### Example 4: Refine Existing Text

```python
draft = "Thompson Sampling is very useful for exploration"

refined = await refine_text(
    draft,
    passes=3,
    strategy='elegance'
)

# Output: "Thompson Sampling balances exploration and exploitation through Bayesian inference."
```

## Performance

Typical performance characteristics:

| Operation | Time | Notes |
|-----------|------|-------|
| Mode detection | <1ms | Keyword matching |
| Initial draft | 5-15ms | Depends on memory count |
| Single refinement pass | 10-20ms | Text processing |
| Quality scoring | 2-5ms | Heuristic metrics |
| **Total (with 3-pass refinement)** | **40-80ms** | For typical content |

Memory usage: ~2-5MB for typical content generation.

## Roadmap

### Phase 2: Expansion (Planned)
- Additional modes (technical, creative, analysis)
- VERIFY refinement strategy
- TONE refinement strategy
- Template system
- Export formats (HTML, PDF)

### Phase 3: Intelligence (Future)
- LLM integration for generation
- Neural quality scoring
- Learning which refinements work
- Adaptive strategy selection

### Phase 4: Advanced Features (Future)
- Multi-document synthesis
- Long-form content generation
- Style transfer
- Collaborative writing

## Contributing

The writing system follows HoloLoom's protocol-based architecture:

- **Protocols**: Define interfaces (`WriterProtocol`, `RefinerProtocol`, etc.)
- **Implementations**: Concrete classes implementing protocols
- **Graceful degradation**: Fallback to basic implementations when specialized ones unavailable

To add a new writing mode:

1. Implement `ModeWriterProtocol`
2. Add to `modes/` directory
3. Register in `Writer.mode_writers`

To add a new refinement strategy:

1. Implement `RefinerProtocol`
2. Add to `refinement/` directory
3. Register in `Composer.refiners`

## License

Part of the HoloLoom project. See root LICENSE file.

## Support

For questions or issues, see the main HoloLoom documentation or open an issue in the repository.

---

**Remember:** "Great writing isn't written, it's refined." 📝
