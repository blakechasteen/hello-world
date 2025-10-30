# Chomsky Linguistic Integration for HoloLoom

**Date:** October 28, 2025
**Status:** Design Document
**Priority:** Research / Enhancement

---

## Executive Summary

This document explores integrating Noam Chomsky's linguistic theories into HoloLoom's feature extraction and semantic representation systems. We focus on three concrete integration points:

1. **Enhanced Motif Detection** - Adding syntactic/grammatical pattern recognition
2. **Linguistic Dimensions in Semantic Calculus** - Expanding the 244D space with linguistic features
3. **Merge Operations in WarpSpace** - Implementing Minimalist Program's Merge for compositional semantics

---

## 1. Enhanced Motif Detection with Syntactic Patterns

### Current State

**File:** `HoloLoom/motif/base.py`

Current motif detection is **keyword-based**:
```python
class RegexMotifDetector(MotifDetector):
    KEYWORDS = [
        "question", "answer", "algorithm", "thompson", "sampling",
        "hive", "inspection", "calculate", "compute"
    ]
```

**Limitations:**
- Purely lexical (word-level matching)
- No structural understanding
- Misses syntactic patterns (e.g., passive voice, subordinate clauses)
- Can't detect transformation patterns (question formation, relativization)

### Proposed Enhancement: Syntactic Motif Detection

#### Key Concepts from Chomsky

**Transformational Grammar** (1957):
- Language has deep structure (underlying meaning) and surface structure (actual utterance)
- Transformations map between them (passivization, question formation, etc.)

**X-bar Theory** (1970):
- All phrases have similar hierarchical structure
- Every phrase has a head (N, V, A, P, D, etc.)
- Constituents: Specifier - Head - Complement

**Minimalist Program** (1995):
- Merge: The fundamental operation combining syntactic objects
- Move: Internal Merge (reordering for emphasis, questions, etc.)
- Feature checking drives syntax

#### Implementation Design

**File:** `HoloLoom/motif/linguistic.py` (new)

```python
"""
Linguistic Motif Detection - Chomsky-inspired syntactic pattern recognition
=============================================================================

Detects structural patterns beyond keywords:
- Constituent structures (NP, VP, PP, CP)
- Grammatical relations (subject, object, adjunct)
- Transformation patterns (passivization, topicalization, wh-movement)
- Dependency chains (long-distance dependencies)

Uses:
- spaCy for dependency parsing
- Custom rules for transformation detection
- Constituency parsing for phrase structure
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re

from HoloLoom.documentation.types import Motif
from HoloLoom.motif.base import MotifDetector

# Import spaCy with graceful fallback
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class SyntacticMotif(Motif):
    """
    Extended motif with syntactic structure.

    Attributes inherited from Motif:
        pattern: str - Pattern name (e.g., "PASSIVE_VOICE", "WH_QUESTION")
        span: Tuple[int, int] - Character span in text
        score: float - Confidence score

    New attributes:
        syntactic_type: str - Type of syntactic structure
        constituents: List[str] - Constituent labels (NP, VP, etc.)
        head: str - Head word of construction
        dependents: List[str] - Dependent words
        transformation: Optional[str] - Name of transformation applied
    """
    syntactic_type: str = "PHRASE"  # PHRASE, CLAUSE, TRANSFORMATION
    constituents: List[str] = None
    head: str = ""
    dependents: List[str] = None
    transformation: Optional[str] = None

    def __post_init__(self):
        if self.constituents is None:
            self.constituents = []
        if self.dependents is None:
            self.dependents = []


class LinguisticMotifDetector(MotifDetector):
    """
    Detects syntactic patterns using Chomskyan linguistic theory.

    Detection Categories:

    1. **Constituent Structures** (X-bar theory)
       - Noun Phrases (NP): "the big red ball"
       - Verb Phrases (VP): "has been running quickly"
       - Prepositional Phrases (PP): "in the morning"
       - Complementizer Phrases (CP): "that he left"

    2. **Grammatical Relations** (theta roles)
       - Subject (Agent): "John hit the ball"
       - Object (Patient): "John hit the ball"
       - Indirect Object (Goal): "John gave Mary the book"
       - Adjuncts (Manner, Time, Place): "quickly", "yesterday", "here"

    3. **Transformations** (movement)
       - Passivization: "The ball was hit by John"
       - Wh-movement: "What did John hit?"
       - Topicalization: "The ball, John hit"
       - Clefting: "It was John who hit the ball"

    4. **Dependency Chains**
       - Long-distance dependencies: "Who do you think that John said that Mary likes?"
       - Binding relationships: "John₁ likes himself₁" (coreference)
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize linguistic motif detector.

        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
        """
        self.nlp = None
        self.spacy_model = spacy_model

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Warning: spaCy model '{spacy_model}' not found. "
                      f"Run: python -m spacy download {spacy_model}")
                self.nlp = None

        # Fallback to regex if no spaCy
        self.regex_fallback = RegexMotifDetector() if self.nlp is None else None

    async def detect(self, text: str) -> List[SyntacticMotif]:
        """
        Detect syntactic motifs in text.

        Returns:
            List of SyntacticMotif objects with structural annotations
        """
        if self.nlp is None:
            # Graceful fallback
            return await self.regex_fallback.detect(text)

        # Parse with spaCy
        doc = self.nlp(text)

        motifs = []

        # 1. Detect constituent structures
        motifs.extend(self._detect_constituents(doc))

        # 2. Detect grammatical relations
        motifs.extend(self._detect_relations(doc))

        # 3. Detect transformations
        motifs.extend(self._detect_transformations(doc))

        # 4. Detect dependency chains
        motifs.extend(self._detect_dependencies(doc))

        return motifs

    def _detect_constituents(self, doc) -> List[SyntacticMotif]:
        """
        Detect phrasal constituents using dependency parse.

        Maps dependency structure → constituency structure:
        - NP: noun + determiners + adjectives + dependents
        - VP: verb + auxiliaries + complements
        - PP: preposition + noun phrase
        """
        motifs = []

        for token in doc:
            # Noun Phrases
            if token.pos_ == "NOUN":
                # Get full NP span (determiners, adjectives, noun)
                np_tokens = [child for child in token.subtree]
                span_start = min(t.idx for t in np_tokens)
                span_end = max(t.idx + len(t.text) for t in np_tokens)

                motifs.append(SyntacticMotif(
                    pattern="NOUN_PHRASE",
                    span=(span_start, span_end),
                    score=0.9,
                    syntactic_type="PHRASE",
                    constituents=["NP"],
                    head=token.text,
                    dependents=[t.text for t in token.children]
                ))

            # Verb Phrases
            elif token.pos_ == "VERB":
                # Get VP span (auxiliaries, verb, complements)
                vp_tokens = [token] + [child for child in token.children
                                       if child.dep_ in ('aux', 'auxpass', 'xcomp', 'ccomp')]
                span_start = min(t.idx for t in vp_tokens)
                span_end = max(t.idx + len(t.text) for t in vp_tokens)

                motifs.append(SyntacticMotif(
                    pattern="VERB_PHRASE",
                    span=(span_start, span_end),
                    score=0.9,
                    syntactic_type="PHRASE",
                    constituents=["VP"],
                    head=token.text,
                    dependents=[t.text for t in token.children]
                ))

            # Prepositional Phrases
            elif token.pos_ == "ADP":
                # Get PP span (preposition + NP)
                pp_tokens = [token] + list(token.children)
                span_start = min(t.idx for t in pp_tokens)
                span_end = max(t.idx + len(t.text) for t in pp_tokens)

                motifs.append(SyntacticMotif(
                    pattern="PREP_PHRASE",
                    span=(span_start, span_end),
                    score=0.85,
                    syntactic_type="PHRASE",
                    constituents=["PP"],
                    head=token.text,
                    dependents=[t.text for t in token.children]
                ))

        return motifs

    def _detect_relations(self, doc) -> List[SyntacticMotif]:
        """
        Detect grammatical relations (theta roles).

        Maps dependency labels → semantic roles:
        - nsubj → Agent
        - nsubjpass → Patient (in passive)
        - dobj → Patient/Theme
        - iobj → Goal/Recipient
        """
        motifs = []

        for token in doc:
            if token.dep_ == "nsubj":
                # Active subject (typically Agent)
                motifs.append(SyntacticMotif(
                    pattern="SUBJECT_AGENT",
                    span=(token.idx, token.idx + len(token.text)),
                    score=0.95,
                    syntactic_type="RELATION",
                    constituents=["SUBJ"],
                    head=token.text,
                    dependents=[]
                ))

            elif token.dep_ == "nsubjpass":
                # Passive subject (Patient)
                motifs.append(SyntacticMotif(
                    pattern="SUBJECT_PATIENT",
                    span=(token.idx, token.idx + len(token.text)),
                    score=0.95,
                    syntactic_type="RELATION",
                    constituents=["SUBJ"],
                    head=token.text,
                    dependents=[],
                    transformation="PASSIVE"
                ))

            elif token.dep_ == "dobj":
                # Direct object (Patient/Theme)
                motifs.append(SyntacticMotif(
                    pattern="OBJECT_PATIENT",
                    span=(token.idx, token.idx + len(token.text)),
                    score=0.9,
                    syntactic_type="RELATION",
                    constituents=["OBJ"],
                    head=token.text,
                    dependents=[]
                ))

        return motifs

    def _detect_transformations(self, doc) -> List[SyntacticMotif]:
        """
        Detect syntactic transformations (movement).

        Chomsky's key insight: Transformations relate surface structure to deep structure.

        Detectable patterns:
        - Passive voice: "was eaten" (auxiliary + past participle)
        - Wh-questions: sentence starts with wh-word
        - Topicalization: fronted object before subject
        """
        motifs = []

        # Passive voice detection
        for token in doc:
            if token.dep_ == "auxpass":
                # Found passive auxiliary ("was", "been", etc.)
                # Get full passive VP
                passive_tokens = [token] + [child for child in token.head.children]
                span_start = min(t.idx for t in passive_tokens)
                span_end = max(t.idx + len(t.text) for t in passive_tokens)

                motifs.append(SyntacticMotif(
                    pattern="PASSIVE_VOICE",
                    span=(span_start, span_end),
                    score=0.95,
                    syntactic_type="TRANSFORMATION",
                    constituents=["VP"],
                    head=token.head.text,
                    dependents=[t.text for t in token.head.children],
                    transformation="PASSIVIZATION"
                ))

        # Wh-question detection
        if doc[0].tag_ in ('WDT', 'WP', 'WP$', 'WRB'):  # Wh-word tags
            # Wh-movement: "What did John see?" → "John saw what?"
            wh_word = doc[0]
            motifs.append(SyntacticMotif(
                pattern="WH_QUESTION",
                span=(0, len(text)),
                score=0.9,
                syntactic_type="TRANSFORMATION",
                constituents=["CP"],
                head=wh_word.text,
                dependents=[],
                transformation="WH_MOVEMENT"
            ))

        return motifs

    def _detect_dependencies(self, doc) -> List[SyntacticMotif]:
        """
        Detect long-distance dependencies.

        Example: "Who₁ do you think [CP that John said [CP that Mary likes t₁]]"
        The "who" at the start is the object of "likes" deep in the structure.
        """
        motifs = []

        # Detect clausal embedding depth
        for sent in doc.sents:
            complement_depth = 0
            for token in sent:
                if token.dep_ in ('ccomp', 'xcomp', 'acl'):
                    complement_depth += 1

            if complement_depth >= 2:
                # Long-distance dependency likely
                motifs.append(SyntacticMotif(
                    pattern="LONG_DISTANCE_DEP",
                    span=(sent.start_char, sent.end_char),
                    score=0.8,
                    syntactic_type="DEPENDENCY",
                    constituents=["CP"] * (complement_depth + 1),
                    head=sent.root.text,
                    dependents=[],
                    transformation="EMBEDDING"
                ))

        return motifs
```

#### Integration with ResonanceShed

**File:** `HoloLoom/resonance/shed.py` (modify)

```python
# In ResonanceShed.__init__():
def __init__(
    self,
    motif_detector=None,
    embedder=None,
    spectral_fusion=None,
    semantic_calculus=None,
    linguistic_motifs: bool = False,  # NEW PARAMETER
    interference_mode: str = "weighted_sum",
    max_feature_density: float = 1.0
):
    """
    Args:
        linguistic_motifs: Use Chomskyan syntactic motif detection (default: False)
    """
    # If linguistic_motifs enabled, use LinguisticMotifDetector
    if linguistic_motifs and motif_detector is None:
        from HoloLoom.motif.linguistic import LinguisticMotifDetector
        motif_detector = LinguisticMotifDetector()

    # ... rest of init ...
```

**Benefits:**
- Richer pattern detection (structure, not just keywords)
- Detects subtle linguistic phenomena (passive voice, embedded clauses)
- Enables syntax-aware semantic analysis
- Backward compatible (fallback to regex if spaCy unavailable)

---

## 2. Linguistic Dimensions in Semantic Calculus

### Current State

**File:** `HoloLoom/semantic_calculus/dimensions.py`

Current 244D semantic space has:
- 16 standard dimensions (Warmth, Valence, Arousal, etc.)
- 228 extended dimensions (Heroism, Transformation, Wisdom, etc.)

**All dimensions are semantic/affective/narrative**, none are **linguistic/grammatical**.

### The Question: Module vs Core?

#### Option A: Linguistic Module (Separate)

**Create:** `HoloLoom/semantic_calculus/linguistic_dimensions.py`

```python
"""
Linguistic Dimensions Module - Grammatical Features as Semantic Axes
======================================================================

Chomsky's insight: Grammar isn't arbitrary - it reflects universal cognitive structures.
These dimensions capture grammatical/syntactic properties as semantic features.

This is a MODULE (not core) because:
- Specialized use case (linguistic analysis)
- Requires spaCy dependency
- Not needed for standard queries
- Can be optionally loaded
"""

LINGUISTIC_DIMENSIONS = [
    # Grammatical Categories (8 dimensions)
    SemanticDimension(
        name="Nominality",
        positive_exemplars=["noun", "entity", "person", "thing", "object", "substance"],
        negative_exemplars=["verb", "action", "process", "event", "change", "motion"]
    ),
    SemanticDimension(
        name="Verbality",
        positive_exemplars=["action", "doing", "acting", "process", "event", "change"],
        negative_exemplars=["state", "being", "property", "quality", "attribute", "static"]
    ),
    SemanticDimension(
        name="Adjectivity",
        positive_exemplars=["quality", "property", "attribute", "characteristic", "feature"],
        negative_exemplars=["action", "entity", "thing", "process", "event"]
    ),
    SemanticDimension(
        name="Relationality",
        positive_exemplars=["relationship", "connection", "link", "between", "among", "with"],
        negative_exemplars=["isolated", "alone", "separate", "independent", "unconnected"]
    ),

    # Theta Roles (8 dimensions)
    SemanticDimension(
        name="Agentivity",
        positive_exemplars=["agent", "actor", "doer", "performer", "initiator", "causer"],
        negative_exemplars=["patient", "recipient", "experiencer", "undergoes", "affected"]
    ),
    SemanticDimension(
        name="Patienthood",
        positive_exemplars=["patient", "theme", "affected", "undergoes", "receives", "experiences"],
        negative_exemplars=["agent", "actor", "initiator", "causes", "controls"]
    ),
    SemanticDimension(
        name="Experiencer",
        positive_exemplars=["experiences", "feels", "perceives", "senses", "aware", "conscious"],
        negative_exemplars=["acts", "causes", "controls", "unaware", "unconscious"]
    ),
    SemanticDimension(
        name="Instrumental",
        positive_exemplars=["instrument", "tool", "means", "method", "with", "using"],
        negative_exemplars=["goal", "purpose", "end", "result", "outcome"]
    ),
    SemanticDimension(
        name="Locative",
        positive_exemplars=["location", "place", "where", "at", "in", "spatial"],
        negative_exemplars=["time", "when", "temporal", "duration", "moment"]
    ),
    SemanticDimension(
        name="Temporal",
        positive_exemplars=["time", "when", "duration", "moment", "period", "temporal"],
        negative_exemplars=["place", "where", "location", "spatial", "position"]
    ),

    # Case/Agreement (6 dimensions)
    SemanticDimension(
        name="Nominative",
        positive_exemplars=["subject", "actor", "I", "he", "she", "who"],
        negative_exemplars=["object", "me", "him", "her", "whom"]
    ),
    SemanticDimension(
        name="Accusative",
        positive_exemplars=["object", "patient", "me", "him", "her", "whom"],
        negative_exemplars=["subject", "actor", "I", "he", "she", "who"]
    ),

    # Voice/Valency (4 dimensions)
    SemanticDimension(
        name="Active_Voice",
        positive_exemplars=["active", "does", "performs", "acts", "causes"],
        negative_exemplars=["passive", "is-done-to", "undergoes", "receives", "affected"]
    ),
    SemanticDimension(
        name="Passive_Voice",
        positive_exemplars=["passive", "is-done-to", "undergoes", "receives", "affected"],
        negative_exemplars=["active", "does", "performs", "acts", "causes"]
    ),

    # Aspect/Modality (6 dimensions)
    SemanticDimension(
        name="Perfectivity",
        positive_exemplars=["completed", "finished", "done", "accomplished", "achieved"],
        negative_exemplars=["ongoing", "incomplete", "in-progress", "continuing", "unfinished"]
    ),
    SemanticDimension(
        name="Imperfectivity",
        positive_exemplars=["ongoing", "in-progress", "continuing", "habitual", "repeated"],
        negative_exemplars=["completed", "finished", "done", "accomplished", "single"]
    ),
    SemanticDimension(
        name="Modality_Necessity",
        positive_exemplars=["must", "should", "necessary", "required", "obligatory"],
        negative_exemplars=["optional", "possible", "unnecessary", "permitted", "allowed"]
    ),
    SemanticDimension(
        name="Modality_Possibility",
        positive_exemplars=["can", "might", "possible", "potential", "permissible"],
        negative_exemplars=["impossible", "cannot", "forbidden", "prohibited"]
    ),

    # Information Structure (6 dimensions)
    SemanticDimension(
        name="Focus",
        positive_exemplars=["emphasized", "highlighted", "stressed", "prominent", "focused"],
        negative_exemplars=["background", "unstressed", "given", "known", "presupposed"]
    ),
    SemanticDimension(
        name="Topic",
        positive_exemplars=["topic", "about", "concerning", "regarding", "as-for"],
        negative_exemplars=["comment", "new-information", "predicate", "assertion"]
    ),
    SemanticDimension(
        name="Given_Information",
        positive_exemplars=["known", "given", "old", "mentioned", "familiar"],
        negative_exemplars=["new", "unknown", "novel", "unmentioned", "unfamiliar"]
    ),

    # Discourse Coherence (4 dimensions)
    SemanticDimension(
        name="Anaphora",
        positive_exemplars=["he", "she", "it", "that", "refers-back", "previously-mentioned"],
        negative_exemplars=["new", "introduces", "first-mention", "novel-entity"]
    ),
    SemanticDimension(
        name="Deixis",
        positive_exemplars=["this", "here", "now", "I", "you", "context-dependent"],
        negative_exemplars=["that", "there", "then", "context-independent", "absolute"]
    ),
]  # Total: 42 linguistic dimensions

# Combined space: 244 (existing) + 42 (linguistic) = 286 dimensions
EXTENDED_286_DIMENSIONS = EXTENDED_244_DIMENSIONS + LINGUISTIC_DIMENSIONS
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Optional (doesn't bloat core)
- ✅ Can be loaded on-demand
- ✅ Specialized for linguistic analysis

**Cons:**
- ❌ Not integrated by default
- ❌ Requires explicit loading
- ❌ Might be underutilized

#### Option B: Linguistic Core (Integrated)

Add linguistic dimensions directly to `EXTENDED_244_DIMENSIONS`:

```python
# In HoloLoom/semantic_calculus/dimensions.py

LINGUISTIC_DIMENSIONS = [
    # ... same 42 dimensions as above ...
]

# Update total count: 244 → 286
EXTENDED_286_DIMENSIONS = (
    STANDARD_DIMENSIONS +           # 16
    NARRATIVE_DIMENSIONS +          # 16
    EMOTIONAL_DEPTH_DIMENSIONS +    # 16
    RELATIONAL_DIMENSIONS +         # 16
    ARCHETYPAL_DIMENSIONS +         # 16
    PHILOSOPHICAL_DIMENSIONS +      # 16
    TRANSFORMATION_DIMENSIONS +     # 16
    MORAL_ETHICAL_DIMENSIONS +      # 16
    CREATIVE_DIMENSIONS +           # 16
    COGNITIVE_COMPLEXITY_DIMENSIONS + # 16
    TEMPORAL_NARRATIVE_DIMENSIONS +  # 16
    SPATIAL_SETTING_DIMENSIONS +    # 12
    CHARACTER_DIMENSIONS +          # 12
    PLOT_DIMENSIONS +               # 12
    THEME_DIMENSIONS +              # 12
    STYLE_VOICE_DIMENSIONS +        # 4
    LINGUISTIC_DIMENSIONS           # 42 NEW
)  # Total = 286
```

**Pros:**
- ✅ Always available
- ✅ Richer default semantic space
- ✅ Language-aware by default

**Cons:**
- ❌ Increases computational cost (286 vs 244 projections)
- ❌ May be unnecessary for non-linguistic queries
- ❌ Bloats semantic space

### Recommendation: Hybrid Approach

**Best of both worlds:**

1. **Create module** (`linguistic_dimensions.py`)
2. **Add config flag** to enable/disable

```python
# HoloLoom/config.py

@dataclass
class Config:
    # ... existing config ...

    # Semantic calculus settings
    semantic_dimensions: str = "standard"  # "standard", "extended", "linguistic"
    use_linguistic_dimensions: bool = False  # Enable Chomskyan linguistic features
```

```python
# HoloLoom/semantic_calculus/__init__.py

def create_semantic_spectrum(mode: str = "extended", use_linguistic: bool = False):
    """
    Create semantic spectrum with specified dimensions.

    Args:
        mode: "standard" (16D), "extended" (244D), or "full" (286D)
        use_linguistic: Add 42 linguistic dimensions (Chomsky-inspired)

    Returns:
        SemanticSpectrum configured with requested dimensions
    """
    from .dimensions import (
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
        LINGUISTIC_DIMENSIONS
    )

    if mode == "standard":
        dims = STANDARD_DIMENSIONS
    elif mode == "full" or use_linguistic:
        dims = EXTENDED_244_DIMENSIONS + LINGUISTIC_DIMENSIONS
    else:
        dims = EXTENDED_244_DIMENSIONS

    return SemanticSpectrum(dimensions=dims)
```

**Usage:**

```python
# Opt-in to linguistic dimensions
config = Config.fused()
config.use_linguistic_dimensions = True

# Creates 286D semantic space
semantic_spectrum = create_semantic_spectrum(
    mode="extended",
    use_linguistic=config.use_linguistic_dimensions
)
```

---

## 3. Merge Operations in WarpSpace

### What is Merge?

**Merge** is the core operation in Chomsky's Minimalist Program (1995+). It's the **fundamental structure-building operation** in human language.

#### The Concept

**Merge takes two syntactic objects and combines them:**

```
Merge(α, β) = {α, β}
```

**Example:**
```
Merge("the", "cat") → {Det: "the", N: "cat"} = NP: "the cat"
Merge(NP: "the cat", V: "sat") → {NP: "the cat", V: "sat"} = VP: "the cat sat"
```

**Two types:**

1. **External Merge**: Combines two separate items
   - "the" + "cat" → "the cat"

2. **Internal Merge**: Moves an item within structure (creates dependencies)
   - "What did John see?" → "John saw what" (wh-word moved to front)

### Why in WarpSpace?

WarpSpace is the **continuous tensor field** where semantic operations occur. It's the perfect place for Merge because:

1. **Compositional semantics**: Merge builds meaning compositionally
2. **Hierarchical structure**: WarpSpace can represent nested structures
3. **Continuous tensors**: Embeddings combine smoothly
4. **Temporary operation**: Merge happens during active computation, then detensions

### Implementation Design

**File:** `HoloLoom/warp/merge.py` (new)

```python
"""
Merge Operations for WarpSpace - Chomskyan Compositional Semantics
===================================================================

Implements Merge from Minimalist Program as tensor operations in WarpSpace.

Key Insight:
In Chomsky's Minimalism, Merge is the atomic structure-building operation.
In HoloLoom's WarpSpace, we implement Merge as compositional tensor fusion:

    Merge(embedding_1, embedding_2) → combined_embedding

This enables:
- Compositional semantics (meaning of "red ball" from "red" + "ball")
- Hierarchical structure (nested merges)
- Head-dependency tracking
- Feature unification

Types of Merge:
1. External Merge: Combine two separate items
2. Internal Merge: Move an item (creates long-distance dependencies)
3. Parallel Merge: Merge multiple items simultaneously
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MergeType(Enum):
    """Type of merge operation."""
    EXTERNAL = "external"  # Combine two separate items
    INTERNAL = "internal"  # Move an item (wh-movement, etc.)
    PARALLEL = "parallel"  # Merge multiple items at once


@dataclass
class MergedObject:
    """
    Result of a Merge operation.

    Represents a syntactic object with compositional semantics.
    """
    embedding: np.ndarray  # Combined embedding
    components: List[str]  # Original components ("the", "cat")
    head: str  # Head word (determines category)
    merge_type: MergeType
    label: str  # Syntactic label (NP, VP, etc.)
    children: List['MergedObject'] = None  # Nested structure
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class MergeOperator:
    """
    Implements Merge operations in WarpSpace.

    Chomsky's Merge in continuous space:
    - External Merge: Vector fusion (weighted sum, concatenation, etc.)
    - Internal Merge: Attention-based reordering
    - Parallel Merge: Multi-head attention

    Usage:
        merger = MergeOperator(fusion_method="weighted_sum")

        # External merge: "the" + "cat" → "the cat"
        the_emb = embedder.encode(["the"])[0]
        cat_emb = embedder.encode(["cat"])[0]
        merged = merger.external_merge(the_emb, cat_emb, head="cat", label="NP")

        # Internal merge: wh-movement
        moved = merger.internal_merge(merged, move_index=0, target_position="front")
    """

    def __init__(
        self,
        fusion_method: str = "weighted_sum",
        head_weight: float = 0.7,
        dependent_weight: float = 0.3
    ):
        """
        Initialize merge operator.

        Args:
            fusion_method: How to combine embeddings ("weighted_sum", "concat", "mlp")
            head_weight: Weight for head in fusion (0-1)
            dependent_weight: Weight for dependent in fusion (0-1)
        """
        self.fusion_method = fusion_method
        self.head_weight = head_weight
        self.dependent_weight = dependent_weight

        # Normalize weights
        total = head_weight + dependent_weight
        self.head_weight /= total
        self.dependent_weight /= total

    def external_merge(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        head: str,
        dependent: str,
        label: str = "PHRASE",
        alpha_is_head: bool = False
    ) -> MergedObject:
        """
        External Merge: Combine two separate syntactic objects.

        Chomsky: Merge(α, β) = {α, β} with label determined by head

        Tensor implementation:
            If α is head: merged_emb = head_weight * α + dependent_weight * β
            If β is head: merged_emb = head_weight * β + dependent_weight * α

        Args:
            alpha: First embedding
            beta: Second embedding
            head: Head word (determines category)
            dependent: Dependent word
            label: Syntactic label (NP, VP, PP, etc.)
            alpha_is_head: True if alpha is the head (default: beta is head)

        Returns:
            MergedObject with compositional embedding

        Example:
            # Merge "the" (determiner) + "cat" (noun) → "the cat" (NP)
            the_emb = embedder.encode(["the"])[0]
            cat_emb = embedder.encode(["cat"])[0]

            # Cat is head (determines that phrase is NP)
            merged = merger.external_merge(
                the_emb, cat_emb,
                head="cat", dependent="the",
                label="NP", alpha_is_head=False
            )
        """
        # Determine head/dependent order
        if alpha_is_head:
            head_emb, dep_emb = alpha, beta
            components = [head, dependent]
        else:
            head_emb, dep_emb = beta, alpha
            components = [dependent, head]

        # Fuse embeddings based on method
        if self.fusion_method == "weighted_sum":
            # Weighted sum (head-prominence bias)
            merged_emb = (self.head_weight * head_emb +
                         self.dependent_weight * dep_emb)

        elif self.fusion_method == "concat":
            # Concatenation (preserves both fully)
            merged_emb = np.concatenate([head_emb, dep_emb])

        elif self.fusion_method == "hadamard":
            # Element-wise product (multiplicative interaction)
            merged_emb = head_emb * dep_emb

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Normalize
        merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-10)

        return MergedObject(
            embedding=merged_emb,
            components=components,
            head=head,
            merge_type=MergeType.EXTERNAL,
            label=label,
            metadata={
                "head_weight": self.head_weight,
                "dependent_weight": self.dependent_weight,
                "fusion_method": self.fusion_method
            }
        )

    def internal_merge(
        self,
        merged_obj: MergedObject,
        move_index: int,
        target_position: str = "front"
    ) -> MergedObject:
        """
        Internal Merge: Move an element within structure.

        Chomsky: Internal Merge creates movement (wh-movement, topicalization, etc.)

        Example: "What did John see?"
            Base structure: "John saw what"
            Internal merge: Move "what" to front → "What did John see?"

        Tensor implementation:
            - Reweight components via attention
            - Emphasize moved element

        Args:
            merged_obj: Existing merged object
            move_index: Index of component to move
            target_position: Where to move ("front", "back")

        Returns:
            New MergedObject with moved component emphasized
        """
        # Compute attention weights (emphasize moved element)
        n_components = len(merged_obj.components)
        attention = np.ones(n_components) * 0.5  # Base attention
        attention[move_index] = 1.5  # Emphasize moved element
        attention = attention / np.sum(attention)  # Normalize

        # If we had component embeddings, we'd reweight them
        # For now, just update metadata

        # Reorder components
        moved_component = merged_obj.components.pop(move_index)
        if target_position == "front":
            new_components = [moved_component] + merged_obj.components
        else:
            new_components = merged_obj.components + [moved_component]

        return MergedObject(
            embedding=merged_obj.embedding,  # Embedding unchanged (simplified)
            components=new_components,
            head=merged_obj.head,
            merge_type=MergeType.INTERNAL,
            label=merged_obj.label,
            metadata={
                **merged_obj.metadata,
                "moved_component": moved_component,
                "move_index": move_index,
                "target_position": target_position,
                "attention_weights": attention.tolist()
            }
        )

    def parallel_merge(
        self,
        embeddings: List[np.ndarray],
        components: List[str],
        head_index: int,
        label: str = "PHRASE"
    ) -> MergedObject:
        """
        Parallel Merge: Merge multiple items simultaneously.

        Useful for complex constructions:
        - "the big red ball" → Merge(Det, Adj, Adj, Noun) all at once
        - Multi-word expressions

        Tensor implementation:
            - Weighted sum with head prominence
            - All dependents contribute equally

        Args:
            embeddings: List of embeddings to merge
            components: List of words
            head_index: Index of head word
            label: Syntactic label

        Returns:
            MergedObject with all components merged
        """
        if len(embeddings) != len(components):
            raise ValueError("Embeddings and components must have same length")

        # Compute weights: head gets head_weight, others share remaining
        weights = np.ones(len(embeddings))
        weights[head_index] = self.head_weight / (1.0 - self.head_weight)  # Boost head
        weights = weights / np.sum(weights)  # Normalize

        # Weighted sum
        merged_emb = np.sum([w * emb for w, emb in zip(weights, embeddings)], axis=0)
        merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-10)

        return MergedObject(
            embedding=merged_emb,
            components=components,
            head=components[head_index],
            merge_type=MergeType.PARALLEL,
            label=label,
            metadata={
                "n_components": len(components),
                "head_index": head_index,
                "weights": weights.tolist()
            }
        )

    def recursive_merge(
        self,
        embeddings: List[np.ndarray],
        words: List[str],
        structure: List[Tuple[int, int, str, str]]
    ) -> MergedObject:
        """
        Recursive Merge: Build hierarchical structure bottom-up.

        Mimics syntactic tree construction:

        Example: "the big cat"
            1. Merge("big", "cat") → Adj+N = N'
            2. Merge("the", N') → Det+N' = NP

        Args:
            embeddings: Leaf embeddings (one per word)
            words: Leaf words
            structure: List of (left_idx, right_idx, head_position, label)
                      head_position: "left" or "right"
                      Merges are applied in order

        Returns:
            Root MergedObject with full tree structure

        Example:
            words = ["the", "big", "cat"]
            embeddings = [emb_the, emb_big, emb_cat]
            structure = [
                (1, 2, "right", "N'"),    # Merge big+cat → N' (cat is head)
                (0, -1, "right", "NP")    # Merge the+N' → NP (N' is head)
            ]
        """
        # Start with leaf nodes
        nodes = [
            MergedObject(
                embedding=emb,
                components=[word],
                head=word,
                merge_type=MergeType.EXTERNAL,
                label=f"LEX({word})"
            )
            for emb, word in zip(embeddings, words)
        ]

        # Apply merges sequentially
        for left_idx, right_idx, head_pos, label in structure:
            # Get nodes to merge
            if right_idx == -1:  # Use last merged node
                right_idx = len(nodes) - 1

            left_node = nodes[left_idx]
            right_node = nodes[right_idx]

            # Determine head
            alpha_is_head = (head_pos == "left")

            # External merge
            merged = self.external_merge(
                left_node.embedding,
                right_node.embedding,
                head=left_node.head if alpha_is_head else right_node.head,
                dependent=right_node.head if alpha_is_head else left_node.head,
                label=label,
                alpha_is_head=alpha_is_head
            )

            # Add children for tree structure
            merged.children = [left_node, right_node]

            # Append to nodes
            nodes.append(merged)

        # Return root (last merged node)
        return nodes[-1]


def visualize_merge_tree(merged_obj: MergedObject, indent: int = 0) -> str:
    """
    Visualize merge tree structure.

    Args:
        merged_obj: Root of merge tree
        indent: Indentation level

    Returns:
        String representation of tree
    """
    prefix = "  " * indent
    result = f"{prefix}{merged_obj.label}: {' '.join(merged_obj.components)}\n"

    for child in merged_obj.children:
        result += visualize_merge_tree(child, indent + 1)

    return result
```

### Integration with WarpSpace

**File:** `HoloLoom/warp/space.py` (modify)

```python
class WarpSpace:
    def __init__(
        self,
        embedder,
        scales: List[int] = [96, 192, 384],
        spectral_fusion=None,
        enable_merge: bool = False,  # NEW
        merge_fusion_method: str = "weighted_sum"  # NEW
    ):
        """
        Args:
            enable_merge: Enable Chomskyan Merge operations (default: False)
            merge_fusion_method: Fusion method for Merge
        """
        # ... existing init ...

        # Merge operator
        if enable_merge:
            from HoloLoom.warp.merge import MergeOperator
            self.merge_operator = MergeOperator(fusion_method=merge_fusion_method)
        else:
            self.merge_operator = None

    def merge_threads(
        self,
        indices: List[int],
        head_index: int,
        label: str = "MERGED_PHRASE"
    ) -> TensionedThread:
        """
        Merge multiple tensioned threads into one.

        Implements Chomskyan Merge in continuous tensor space.

        Args:
            indices: Indices of threads to merge
            head_index: Index of head thread (determines category)
            label: Label for merged thread

        Returns:
            New TensionedThread with merged embedding

        Example:
            # Tension "the", "big", "cat"
            await warp.tension(["the", "big", "cat"])

            # Merge into "the big cat" (cat is head at index 2)
            merged_thread = warp.merge_threads([0, 1, 2], head_index=2, label="NP")
        """
        if not self.merge_operator:
            raise RuntimeError("Merge not enabled. Set enable_merge=True")

        if not self.is_tensioned:
            raise RuntimeError("WarpSpace not tensioned")

        # Get threads
        threads_to_merge = [self.threads[i] for i in indices]
        embeddings = [t.embedding for t in threads_to_merge]
        components = [t.entity for t in threads_to_merge]

        # Parallel merge
        merged_obj = self.merge_operator.parallel_merge(
            embeddings,
            components,
            head_index,
            label
        )

        # Create new tensioned thread
        merged_thread = TensionedThread(
            thread_id=f"merged_{'-'.join(str(i) for i in indices)}",
            entity=' '.join(merged_obj.components),
            embedding=merged_obj.embedding,
            tension=np.mean([t.tension for t in threads_to_merge]),  # Average tension
            metadata={
                "merged_from": indices,
                "head": merged_obj.head,
                "label": merged_obj.label,
                "merge_type": merged_obj.merge_type.value
            }
        )

        return merged_thread
```

**Usage Example:**

```python
# Enable merge in WarpSpace
warp = WarpSpace(
    embedder,
    scales=[96, 192, 384],
    enable_merge=True,
    merge_fusion_method="weighted_sum"
)

# Tension individual words
await warp.tension(["the", "big", "red", "ball"])

# Merge compositionally:
# 1. "big red" → Adj+Adj
adj_phrase = warp.merge_threads([1, 2], head_index=2, label="AdjP")

# 2. "red ball" → Adj+N
noun_phrase = warp.merge_threads([2, 3], head_index=1, label="N'")

# 3. "the [red ball]" → Det+N'
full_np = warp.merge_threads([0, -1], head_index=1, label="NP")

# Result: "the big red ball" with compositional semantics!
```

---

## Summary

### 1. Enhanced Motif Detection ✅
- **Status**: Design complete
- **Implementation**: `HoloLoom/motif/linguistic.py`
- **Impact**: Richer syntactic pattern recognition
- **Dependency**: spaCy (with graceful fallback)
- **Integration**: Optional parameter in ResonanceShed

### 2. Linguistic Dimensions 🔄
- **Status**: Design complete, placement TBD
- **Options**: Module vs Core vs Hybrid
- **Recommendation**: Hybrid (module + config flag)
- **Impact**: Language-aware semantic space (286D)
- **Dependency**: None (uses existing infrastructure)

### 3. Merge Operations ✅
- **Status**: Design complete
- **Implementation**: `HoloLoom/warp/merge.py`
- **Impact**: Compositional semantics in WarpSpace
- **Dependency**: None (uses NumPy)
- **Integration**: Optional parameter in WarpSpace

---

## Next Steps

### Immediate (This Week)
1. ✅ Document design (this file)
2. ⬜ Review with Blake - get feedback on Module vs Core for linguistic dimensions
3. ⬜ Decide on integration priority (Phase 1? Phase 2? Future?)

### Short-term (2-4 weeks)
1. ⬜ Implement `LinguisticMotifDetector`
2. ⬜ Add linguistic dimensions (chosen approach)
3. ⬜ Implement `MergeOperator`
4. ⬜ Integration tests

### Long-term (Research)
1. ⬜ Evaluate impact on semantic understanding
2. ⬜ Benchmark performance (linguistic vs non-linguistic)
3. ⬜ User study: Does linguistic analysis improve query responses?
4. ⬜ Paper: "Chomskyan Compositional Semantics in Neural Weaving Systems"

---

## Open Questions for Discussion

1. **Linguistic Dimensions Placement**: Module, Core, or Hybrid?
   - Blake: Which do you prefer and why?

2. **Performance Impact**: Is 286D semantic space too expensive?
   - Need benchmarks: 244D vs 286D projection speed

3. **Use Cases**: When is linguistic analysis actually helpful?
   - Linguistic queries? Formal language processing? General QA?

4. **Merge Complexity**: Should Merge be simplified for MVP?
   - Start with External Merge only?
   - Skip Internal Merge (movement) for now?

5. **SpaCy Dependency**: Acceptable tradeoff?
   - Graceful fallback ensures system doesn't break
   - But adds complexity

---

**Document Status**: Ready for review
**Next Action**: Discuss with Blake - get decisions on open questions
