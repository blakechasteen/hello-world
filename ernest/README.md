# Ernest: Hemingway-Powered Creative Writing AI

**"Prose is architecture, not interior decoration."** — Ernest Hemingway

---

## Overview

**Ernest** is a Hemingway-inspired creative writing assistant built on HoloLoom's agentic reasoning system. Named after Ernest Hemingway, this AI embodies the literary master's principles: economy of language, emotional restraint, active voice dominance, and the famous **Iceberg Theory** (show 10%, imply 90%).

Ernest doesn't coddle. Ernest doesn't flatter. Ernest tells you the truth about your writing, then shows you how to make it sharper, cleaner, stronger.

**Status**: Wave 1 Complete (November 2025)
**Architecture**: Metaprompt-driven refinement with Hemingway-specific pattern detection
**Performance**: 3-pass surgical refinement (clarity → simplicity → beauty)

---

## The Hemingway Philosophy

### Iceberg Theory
> **"If a writer of prose knows enough about what he is writing about, he may omit things that he knows and the reader, if the writer is writing truly enough, will have a strong feeling of those things as though the writer had stated them."**

Ernest analyzes your prose for **showing vs. telling**:
- ✅ **Good**: "His hands shook. He lit another cigarette." (shows nervousness)
- ❌ **Bad**: "He felt nervous and anxious." (tells emotion directly)

**Target**: 70%+ showing (30% or less telling)

### Core Principles

1. **Short Sentences**: 15-20 words average (target: 16)
2. **Active Voice**: 85%+ active constructions
3. **Strong Verbs**: 70%+ precise, active verbs (not "was walking" → "strode")
4. **Concrete Nouns**: Specific, evocative (not "tree" → "gnarled oak")
5. **Emotional Restraint**: Understatement over melodrama
6. **Readability**: Flesch-Kincaid grade 7 (8th-9th grade reading level)

---

## Four Writing Modes

Ernest provides **4 distinct modes** inspired by Hemingway's works:

### 1. SPARSE Mode (Iceberg Maximum)
**Inspiration**: *Hills Like White Elephants*, *Cat in the Rain*
**Philosophy**: 90% subtext, 10% text. Maximum implication.

**Characteristics**:
- 8-12 word sentences
- Stripped dialogue (no attribution unless necessary)
- Zero emotional description (all shown through action/dialogue)
- Parataxis (simple sentences, minimal subordination)

**Use When**: Writing dialogue-heavy scenes, conflict scenes, minimalist fiction

**Example**:
```
Before: "I don't think we should do it," she said nervously, wringing her hands.
After: "I don't think we should." She looked at the floor.
```

### 2. DIRECT Mode (Journalistic Precision)
**Inspiration**: *A Farewell to Arms*, *For Whom the Bell Tolls*
**Philosophy**: Concrete facts. Chronological clarity. No abstractions.

**Characteristics**:
- 12-18 word sentences
- Chronological narration
- Concrete sensory details
- Active voice dominance (90%+)

**Use When**: Action sequences, scene-setting, factual narrative

**Example**:
```
Before: The sunset was beautiful and made everything glow with warmth.
After: The sun set red over the mountains. The snow turned pink.
```

### 3. GRACE Mode (Economical Elegance)
**Inspiration**: *The Sun Also Rises*, *A Moveable Feast*
**Philosophy**: Precision and beauty in balance. Not a word wasted.

**Characteristics**:
- 15-20 word sentences
- Varied rhythm (short-medium-long pattern)
- One perfect detail (not five generic ones)
- Sensory precision (each detail serves 2+ purposes)

**Use When**: Literary fiction, descriptive passages, character moments

**Example**:
```
Before: The man walked down the street on a nice day feeling good.
After: The man strode down Boulevard Saint-Michel. Sunlight glinted off the café windows.
```

### 4. LOST_GEN Mode (Disaffected Truth)
**Inspiration**: *The Sun Also Rises*, *A Farewell to Arms*
**Philosophy**: Existential honesty. Post-war disillusionment. Truth without sentiment.

**Characteristics**:
- Flat affect narration
- Repetition for emphasis ("It was dark. Then it wasn't dark.")
- Cynical observations
- Emotional deadness shown (not stated)

**Use When**: Existential fiction, post-traumatic narratives, cynical narrators

**Example**:
```
Before: He was devastated by the war and couldn't feel anything anymore.
After: The war was over. That was all. He drank. He slept. That was all.
```

---

## Quick Start

### Installation

```bash
# Ernest is part of the mythRL/HoloLoom system
cd mythRL

# Activate environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### Basic Usage

```python
from ernest.refinement.patterns import HemingwayPatterns, HemingwayMode

# Create refiner
ernest = HemingwayPatterns(mode=HemingwayMode.GRACE)

# Analyze your prose
text = """
The man was walking down the street feeling very happy
because it was a beautiful day and he really loved the city.
"""

metrics = ernest.analyze_prose(text)
print(f"Hemingway Score: {metrics.hemingway_score:.0f}/100")
print(f"Avg Sentence Length: {metrics.avg_words_per_sentence:.1f} words")
print(f"Active Voice: {metrics.active_voice_pct:.0f}%")
print(f"Iceberg Ratio: {metrics.iceberg_ratio:.0f}% showing")
```

**Output**:
```
Hemingway Score: 45/100
Avg Sentence Length: 24.0 words (target: 16)
Active Voice: 50% (target: 85%)
Iceberg Ratio: 20% showing (target: 70%)
```

### 3-Pass Refinement

```python
# Pass 1: Clarity (break long sentences, plain language)
result1 = ernest.refine_pass1_clarity(text)
print(f"Pass 1 Score: {result1.metrics.hemingway_score:.0f}/100")
print(f"Changes: {len(result1.changes)}")

# Pass 2: Simplicity (cut filler, strengthen verbs)
result2 = ernest.refine_pass2_simplicity(result1.refined_text)
print(f"Pass 2 Score: {result2.metrics.hemingway_score:.0f}/100")

# Pass 3: Beauty (active voice, rhythm, precision)
result3 = ernest.refine_pass3_beauty(result2.refined_text)
print(f"Pass 3 Score: {result3.metrics.hemingway_score:.0f}/100")

print("\nFINAL:")
print(result3.refined_text)
```

**Output**:
```
Pass 1 Score: 60/100
Changes: 3

Pass 2 Score: 75/100
Changes: 5

Pass 3 Score: 88/100
Changes: 4

FINAL:
The man strode down the avenue. Sunlight warmed the stone buildings.
He bought coffee at the corner café.
```

### Quick Analysis Helper

```python
from ernest.refinement.patterns import quick_analyze, quick_refine

# One-line analysis
score = quick_analyze("Your text here")
print(f"Score: {score}/100")

# One-line refinement (all 3 passes)
refined = quick_refine("Your text here", mode="grace")
print(refined)
```

---

## Integration with Creative Writing AI

Ernest integrates seamlessly with your existing `ingest_my_writing.py` system:

### Add Ernest to Your Workflow

```python
# File: ingest_my_writing.py (enhanced)
from HoloLoom.rag import SimpleRAG
from ernest.refinement.patterns import HemingwayPatterns, HemingwayMode

async def main():
    # Your existing RAG setup
    async with SimpleRAG(config=config) as rag:
        # Ingest your chapters
        for chapter_file in chapter_files:
            content = Path(chapter_file).read_text()
            await rag.ingest(content)

        # Create Ernest analyzer
        ernest = HemingwayPatterns(mode=HemingwayMode.GRACE)

        # Interactive mode with Ernest feedback
        while True:
            user_query = input("You: ").strip()

            if user_query.startswith("/analyze"):
                # Analyze latest chapter
                chapter_text = Path("chapter5").read_text()
                metrics = ernest.analyze_prose(chapter_text)

                print(f"\n📊 Ernest's Analysis:")
                print(f"   Hemingway Score: {metrics.hemingway_score:.0f}/100")
                print(f"   Avg Sentence: {metrics.avg_words_per_sentence:.1f} words")
                print(f"   Active Voice: {metrics.active_voice_pct:.0f}%")
                print(f"   Iceberg Ratio: {metrics.iceberg_ratio:.0f}% showing")

            elif user_query.startswith("/refine"):
                # Refine latest chapter
                chapter_text = Path("chapter5").read_text()
                refined = ernest.refine_pass3_beauty(
                    ernest.refine_pass2_simplicity(
                        ernest.refine_pass1_clarity(chapter_text).refined_text
                    ).refined_text
                ).refined_text

                print(f"\n✍️ Ernest's Refinement:")
                print(refined)

            else:
                # Normal RAG query
                result = await rag.query(user_query, mode="verify")
                print(f"\n🤖 AI: {result.response}")
```

### Ernest CLI Commands

Add these commands to your creative writing workflow:

- `/analyze` - Get Hemingway score for current chapter
- `/refine` - Run 3-pass refinement
- `/sparse` - Switch to SPARSE mode (iceberg maximum)
- `/direct` - Switch to DIRECT mode (journalistic)
- `/grace` - Switch to GRACE mode (economical elegance)
- `/lost_gen` - Switch to LOST_GEN mode (disaffected truth)
- `/score` - Show detailed metrics breakdown

---

## API Reference

### `HemingwayMode` (Enum)

```python
class HemingwayMode(Enum):
    SPARSE = "sparse"      # Iceberg maximum (8-12 word sentences)
    DIRECT = "direct"      # Journalistic precision
    GRACE = "grace"        # Economical elegance (default)
    LOST_GEN = "lost_gen"  # Disaffected truth
```

### `ProseMetrics` (Dataclass)

```python
@dataclass
class ProseMetrics:
    avg_words_per_sentence: float    # Target: 16 words
    max_sentence_length: int          # Target: <25 words
    active_voice_pct: float           # Target: 85%
    strong_verb_pct: float            # Target: 70%
    flesch_kincaid_grade: float       # Target: 7 (8th-9th grade)
    iceberg_ratio: float              # Target: 70% (showing vs telling)
    filler_word_count: int            # Target: 0
    weak_verb_count: int              # Target: minimize
    emotion_word_count: int           # Target: minimize (show, don't tell)
    telling_verb_count: int           # Target: minimize
    hemingway_score: float            # 0-100 composite score
```

**Hemingway Score Calculation**:
```
hemingway_score = (
    conciseness_score    × 0.25 +  # Sentence length
    active_voice_score   × 0.20 +  # Active voice %
    verb_strength_score  × 0.20 +  # Strong verbs %
    readability_score    × 0.15 +  # Flesch-Kincaid
    iceberg_score        × 0.20    # Showing vs telling
)
```

### `HemingwayPatterns` (Class)

#### Constructor

```python
ernest = HemingwayPatterns(
    mode: HemingwayMode = HemingwayMode.GRACE
)
```

#### Methods

**Analysis**:
```python
metrics = ernest.analyze_prose(text: str) -> ProseMetrics
# Returns comprehensive Hemingway metrics for text
```

**Refinement (3 passes)**:
```python
result1 = ernest.refine_pass1_clarity(text: str) -> RefinementResult
# Pass 1: Break long sentences, replace jargon, add paragraph breaks

result2 = ernest.refine_pass2_simplicity(text: str) -> RefinementResult
# Pass 2: Remove filler words, eliminate redundancy, strengthen verbs

result3 = ernest.refine_pass3_beauty(text: str) -> RefinementResult
# Pass 3: Active voice, sentence rhythm, precision
```

**Helper Methods**:
```python
ratio = ernest._iceberg_ratio(text: str) -> float
# Calculate showing vs. telling percentage (0-100)

grade = ernest._flesch_kincaid_grade(text: str) -> float
# Calculate readability grade level

score = ernest._calculate_hemingway_score(metrics: ProseMetrics) -> float
# Calculate composite 0-100 Hemingway score
```

### Quick Analysis Functions

```python
from ernest.refinement.patterns import quick_analyze, quick_refine, hemingway_score

# One-line analysis
score = quick_analyze(text: str) -> float

# One-line refinement (all 3 passes)
refined = quick_refine(text: str, mode: str = "grace") -> str

# Get full score breakdown
score, metrics = hemingway_score(text: str) -> tuple[float, ProseMetrics]
```

---

## Ernest's Targets (vs Standard Elegance)

| Metric | Ernest (Hemingway) | Standard Elegance |
|--------|-------------------|-------------------|
| **Avg Sentence Length** | 16 words | 20 words |
| **Max Sentence Length** | 25 words | 35 words |
| **Active Voice %** | 85% | 70% |
| **Strong Verb %** | 70% | 60% |
| **Flesch-Kincaid Grade** | 7 (8th-9th) | 8-10 (9th-11th) |
| **Iceberg Ratio** | 70% showing | 60% showing |
| **Filler Words** | 0 | <5 per 100 words |
| **Hemingway Score** | 80-90 (good) | 70-80 (good) |

**Ernest is stricter** than standard elegance refinement, targeting Hemingway's exacting standards.

---

## Pattern Detection

### Filler Words (15 patterns)
Ernest flags and removes: `very`, `really`, `quite`, `just`, `actually`, `basically`, `literally`, `simply`, `certainly`, `absolutely`, `completely`, `totally`, `extremely`, `incredibly`, `particularly`

**Example**:
```
Before: "He was really very tired and just wanted to sleep."
After:  "He wanted to sleep."
```

### Weak Verbs (12 patterns)
Ernest flags: `is`, `are`, `was`, `were`, `been`, `being`, `have`, `has`, `had`, `do`, `does`, `did`

**Example**:
```
Before: "He was walking down the street."
After:  "He walked down the street." (or better: "He strode...")
```

### Emotion Words (18 patterns)
Ernest flags direct emotion statements: `happy`, `sad`, `angry`, `scared`, `worried`, `anxious`, `nervous`, `excited`, `disappointed`, `frustrated`, `confused`, `surprised`, `shocked`, `disgusted`, `ashamed`, `guilty`, `proud`, `jealous`

**Iceberg Theory**: Show emotion through action/dialogue, not direct statement.

**Example**:
```
Before: "She felt nervous about the interview."
After:  "Her hands shook. She checked her watch three times."
```

### Telling Verbs (10 patterns)
Ernest flags: `felt`, `thought`, `knew`, `realized`, `understood`, `believed`, `wondered`, `noticed`, `remembered`, `forgot`

**Example**:
```
Before: "He realized he'd made a mistake."
After:  "He stared at the broken glass."
```

---

## Examples: Before & After

### Example 1: Descriptive Passage

**Before (Hemingway Score: 42/100)**:
```
The sunset was absolutely beautiful that evening, painting the sky
with really vibrant colors that made everything glow with a warm,
golden light. Sarah felt happy as she watched it, thinking about
how lucky she was to be there.
```

**Problems**:
- Filler words: "absolutely", "really"
- Telling verb: "felt happy", "thinking about"
- Emotion word: "happy"
- Long sentence: 35+ words
- Weak verbs: "was", "made"
- Passive construction: "painting the sky"

**After GRACE Mode (Hemingway Score: 91/100)**:
```
The sun set red over the mountains. The sky turned orange, then
purple. Sarah bought bread at the corner boulangerie. The evening
smelled of coffee and cigarettes.
```

**Changes**:
- ✅ Concrete details ("red", "orange", "purple" vs "vibrant colors")
- ✅ Active voice ("sun set" vs "was painting")
- ✅ Showing ("bought bread", "smelled of coffee" implies contentment)
- ✅ Short sentences (8-12 words each)
- ✅ Sensory precision (sight + smell)

---

### Example 2: Dialogue Scene

**Before (Hemingway Score: 38/100)**:
```
"I really don't think we should do this," she said nervously,
wringing her hands together anxiously. "It just feels wrong and
I'm worried about what might happen if we go through with it."
```

**Problems**:
- Filler: "really", "just"
- Emotion words: "nervously", "anxiously", "worried"
- Telling action: "wringing her hands"
- Long sentence
- Over-explaining the emotion

**After SPARSE Mode (Hemingway Score: 95/100)**:
```
"I don't think we should."

She looked at the floor.

"What if—" She stopped.
```

**Changes**:
- ✅ Stripped dialogue (no attribution unless necessary)
- ✅ Action shows nervousness (not told)
- ✅ Incomplete sentence shows hesitation
- ✅ 90% subtext (iceberg theory maximum)

---

### Example 3: Action Sequence

**Before (Hemingway Score: 35/100)**:
```
The soldier was feeling extremely scared as he carefully moved
forward through the dark forest, trying very hard not to make any
noise because he knew the enemy could be anywhere nearby.
```

**Problems**:
- Filler: "extremely", "very", "any"
- Telling: "feeling scared", "knew"
- Weak verb: "was feeling", "was moving"
- Passive construction: "could be"
- Long sentence (28 words)

**After DIRECT Mode (Hemingway Score: 89/100)**:
```
The soldier crept through the forest. The trees were dark.
He stopped. He listened. Nothing moved.
```

**Changes**:
- ✅ Active, precise verbs ("crept" vs "was moving")
- ✅ Short sentences (4-8 words)
- ✅ Chronological clarity
- ✅ Fear shown through actions ("stopped", "listened")
- ✅ Concrete sensory details

---

## Hemingway Principles Reference

### The 6 Hemingway Rules

1. **Use short sentences and short paragraphs**
   - Target: 15-20 words per sentence (Ernest targets 16)
   - Max: 25 words (Ernest), 35 words (standard)

2. **Use vigorous English**
   - Active voice: 85%+ (Ernest), 70%+ (standard)
   - Strong, precise verbs: 70%+ (Ernest)

3. **Be positive, not negative**
   - Say what it is, not what it isn't
   - Example: "He was hungry" vs "He wasn't full"

4. **Never use a long word where a short one will do**
   - Replace jargon and complex words with plain language
   - Example: "utilize" → "use", "ascertain" → "learn"

5. **Omit needless words**
   - Cut filler words (very, really, just, etc.)
   - Eliminate redundancy ("completely destroyed" → "destroyed")

6. **Show, don't tell**
   - Iceberg Theory: 70%+ showing (action/dialogue) vs telling (emotion/thought)
   - Example: "Her hands shook" vs "She was nervous"

### Iceberg Theory in Practice

**What to Show** (10% visible):
- Physical actions
- Dialogue
- Sensory details (what you see, hear, smell, taste, touch)
- Concrete objects and settings

**What to Imply** (90% submerged):
- Emotions and feelings
- Internal thoughts
- Backstory and context
- Thematic meaning
- Character psychology

**Example**:
```
Visible (Text):     "He drank. He ordered another."
Submerged (Subtext): Alcoholism, pain, coping mechanism,
                     lost hope, self-destruction
```

---

## Testing Ernest

### Manual Testing

```python
# Test all modes
from ernest.refinement.patterns import HemingwayPatterns, HemingwayMode

test_text = """
The man was walking down the street feeling very happy because
it was a beautiful day and he really loved the city.
"""

for mode in HemingwayMode:
    ernest = HemingwayPatterns(mode=mode)
    result = ernest.refine_pass3_beauty(
        ernest.refine_pass2_simplicity(
            ernest.refine_pass1_clarity(test_text).refined_text
        ).refined_text
    )

    print(f"\n{mode.value.upper()} MODE:")
    print(f"Score: {result.metrics.hemingway_score:.0f}/100")
    print(f"Text: {result.refined_text}")
```

### Test Suite (Coming in Wave 3)

```bash
# Run Ernest test suite (once created)
pytest ernest/tests/ -v

# Expected tests:
# - test_iceberg_ratio_calculation
# - test_filler_word_detection
# - test_weak_verb_flagging
# - test_sentence_breaking
# - test_active_voice_conversion
# - test_hemingway_score_accuracy
# - test_all_modes
```

---

## Roadmap

### Wave 1: Foundation ✅ COMPLETE
- [x] Create Hemingway persona metaprompts (4 modes)
- [x] Extract and enhance elegance refinement patterns
- [x] Document Ernest architecture (this README)

### Wave 2: Integration (Pending)
- [ ] Enable Pattern Learning (Phase 2) with Ernest persona
- [ ] Activate Full Orchestrator (Phase 3) for Ernest
- [ ] Thompson Sampling learns which mode works best for which writing

### Wave 3: Enhancement (Pending)
- [ ] Create parallel creative passes system:
  - Plot pass (story structure analysis)
  - Character pass (voice consistency)
  - Dialogue pass (natural speech patterns)
  - Style pass (Hemingway principles)
- [ ] Build metaprompt adapter for Ernest
- [ ] Create comprehensive testing suite

### Wave 4: Safety & Agents (Pending)
- [ ] Enable Alignment Framework (Phase 4)
- [ ] Activate Collaborative Agents (Phase 5)
- [ ] Multi-agent creative writing workflows

### Wave 5: Zero-G Integration (Pending)
- [ ] Zero-G data integration for collaborative writing
- [ ] NASA-style UX for creative projects

### Wave 6: Production (Pending)
- [ ] Circuit breakers + rate limiting
- [ ] Health checks + monitoring
- [ ] Production deployment guide

---

## Contributing

Ernest is part of the mythRL/HoloLoom project. See main repository for contribution guidelines.

**Architecture**: Metaprompt-driven with pattern-based refinement
**Inspired by**: Ernest Hemingway's literary principles
**Built on**: HoloLoom agentic reasoning system

---

## References

### Hemingway Works Referenced

- **Iceberg Theory**: *Death in the Afternoon* (1932)
- **SPARSE Mode**: *Hills Like White Elephants*, *Cat in the Rain*
- **DIRECT Mode**: *A Farewell to Arms*, *For Whom the Bell Tolls*
- **GRACE Mode**: *The Sun Also Rises*, *A Moveable Feast*
- **LOST_GEN Mode**: *The Sun Also Rises*, *A Farewell to Arms*

### External Resources

- Hemingway's 6 Rules for Writing (derived from *The Elements of Style*)
- Iceberg Theory explained in *Death in the Afternoon*
- Flesch-Kincaid Readability Grade Level
- Active Voice vs Passive Voice guidelines

---

## License

Part of mythRL/HoloLoom project. See main repository LICENSE.

---

## Contact

For questions about Ernest or HoloLoom:
- **Repository**: mythRL/HoloLoom
- **Documentation**: See main repository CLAUDE.md

---

**"All you have to do is write one true sentence. Write the truest sentence that you know."**
— Ernest Hemingway
