# Linguistic-Matryoshka Gate Integration

**Date:** October 28, 2025
**Status:** Design Architecture
**Connects:** Chomsky Linguistics + Matryoshka Multi-Scale Gating

---

## The Big Idea

The **matryoshka gate** filters candidates through **embedding scales** (96d → 192d → 384d).
**Linguistic features** add a **syntactic dimension** that gates BEFORE semantic similarity!

### Two-Dimensional Gating

```
                    LINGUISTIC GATE (syntactic)
                           ↓
                    [Filter by structure]
                           ↓
                    MATRYOSHKA GATE (semantic)
                           ↓
                    [96d → 192d → 384d filtering]
```

**Why this works:**

1. **Linguistic gate is fast** - spaCy parsing is cheap (milliseconds)
2. **Reduces matryoshka load** - only encode structurally-relevant candidates
3. **Composable filters** - linguistic + semantic = powerful combo
4. **Graceful degradation** - if no spaCy, skip linguistic gate

---

## Architecture

### Phase 1: Linguistic Pre-Gate (Optional)

**When enabled**, filter candidates BEFORE embedding:

```python
from HoloLoom.motif.linguistic import LinguisticMotifDetector
from HoloLoom.embedding.matryoshka_gate import MatryoshkaGate

# Initialize both gates
linguistic_gate = LinguisticMotifDetector()
matryoshka_gate = MatryoshkaGate(embedder, config)

# Two-stage gating
async def two_stage_gate(query: str, candidates: List[str], final_k: int = 10):
    """
    Gate candidates through linguistic + semantic filters.

    Stage 1: Linguistic filtering (fast, syntactic)
    Stage 2: Matryoshka filtering (progressive, semantic)
    """
    # STAGE 1: Linguistic gate
    if linguistic_gate:
        # Detect syntactic patterns in query
        query_motifs = await linguistic_gate.detect(query)
        query_patterns = set(m.pattern for m in query_motifs)

        # Filter candidates by syntactic compatibility
        linguistically_filtered = []
        for candidate in candidates:
            candidate_motifs = await linguistic_gate.detect(candidate)
            candidate_patterns = set(m.pattern for m in candidate_motifs)

            # Keep if patterns overlap OR candidate has relevant structure
            if query_patterns & candidate_patterns:  # Intersection
                linguistically_filtered.append(candidate)

        print(f"Linguistic gate: {len(candidates)} → {len(linguistically_filtered)}")
        candidates = linguistically_filtered

    # STAGE 2: Matryoshka gate
    final_indices, gate_results = matryoshka_gate.gate(
        query, candidates, final_k=final_k
    )

    return final_indices, gate_results
```

**Example:**

Query: `"What is passive voice?"`

**Linguistic gate detects:**
- Pattern: `WH_QUESTION` (starts with "what")
- Structure: Looking for definitions

**Candidate 1:** `"Passive voice is when the subject receives the action"`
- Pattern: `PASSIVE_VOICE` detected ✅
- LINGUISTIC MATCH → Pass to matryoshka gate

**Candidate 2:** `"Machine learning uses neural networks"`
- Pattern: `ACTIVE_VOICE`, no passive ❌
- LINGUISTIC MISMATCH → Filtered out (never encoded!)

**Result:** Only structurally-relevant candidates go through expensive embedding!

---

### Phase 2: Linguistic Features in Embedding Space

**More sophisticated**: Add linguistic features AS PART of the embedding!

#### Enhanced Matryoshka with Linguistic Dimensions

```python
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

class LinguisticMatryoshkaEmbeddings(MatryoshkaEmbeddings):
    """
    Matryoshka embeddings enhanced with linguistic features.

    Embedding structure:
    - [0:384]: Semantic embedding (from sentence-transformers)
    - [384:426]: Linguistic features (42D from syntactic analysis)
    - Total: 426D base embedding

    Matryoshka scales:
    - 96d: Coarse semantic only
    - 192d: Medium semantic + basic linguistic (8D)
    - 384d: Full semantic + full linguistic (42D)
    """

    def __init__(
        self,
        sizes: List[int] = [96, 192, 384],
        enable_linguistic: bool = True,
        linguistic_dims: int = 42
    ):
        # Initialize base embedder
        super().__init__(sizes=sizes)

        # Linguistic feature extractor
        self.enable_linguistic = enable_linguistic
        self.linguistic_dims = linguistic_dims

        if enable_linguistic:
            from HoloLoom.motif.linguistic import LinguisticMotifDetector
            self.linguistic_detector = LinguisticMotifDetector()
        else:
            self.linguistic_detector = None

    async def encode_with_linguistics(
        self,
        texts: List[str]
    ) -> Dict[int, np.ndarray]:
        """
        Encode texts with linguistic features at each scale.

        Returns:
            Dict mapping scale → embeddings with linguistic features
        """
        # Get semantic embeddings (base)
        semantic_embeds = self.encode_base(texts)  # (n_texts, 384)

        # Extract linguistic features
        if self.enable_linguistic and self.linguistic_detector:
            linguistic_features = await self._extract_linguistic_batch(texts)  # (n_texts, 42)
        else:
            linguistic_features = np.zeros((len(texts), self.linguistic_dims))

        # Combine semantic + linguistic
        full_embeds = np.concatenate([semantic_embeds, linguistic_features], axis=1)  # (n_texts, 426)

        # Project to each scale
        result = {}
        for scale in self.sizes:
            if scale == 96:
                # Coarse: semantic only
                result[scale] = full_embeds[:, :96]
            elif scale == 192:
                # Medium: semantic + 8D linguistic
                result[scale] = np.concatenate([
                    full_embeds[:, :184],      # Semantic (184D)
                    full_embeds[:, 384:392]    # Top 8 linguistic features
                ], axis=1)
            elif scale == 384:
                # Fine: full semantic + full linguistic
                result[scale] = full_embeds  # All 426D
            else:
                # Custom scale
                result[scale] = full_embeds[:, :scale]

        return result

    async def _extract_linguistic_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract 42D linguistic features from texts.

        Features (42 dimensions):
        - [0:8] Grammatical categories (Nominality, Verbality, etc.)
        - [8:14] Theta roles (Agent, Patient, etc.)
        - [14:20] Case/agreement
        - [20:24] Voice/valency
        - [24:30] Aspect/modality
        - [30:36] Information structure
        - [36:42] Discourse coherence
        """
        features = []

        for text in texts:
            # Detect syntactic motifs
            motifs = await self.linguistic_detector.detect(text)

            # Convert motifs to feature vector
            feature_vec = self._motifs_to_features(motifs)
            features.append(feature_vec)

        return np.array(features)

    def _motifs_to_features(self, motifs: List[SyntacticMotif]) -> np.ndarray:
        """
        Convert syntactic motifs to 42D feature vector.

        Each dimension represents presence/strength of linguistic feature.
        """
        features = np.zeros(42)

        for motif in motifs:
            # Map pattern to feature index
            if motif.pattern == "NOUN_PHRASE":
                features[0] += motif.score  # Nominality
            elif motif.pattern == "VERB_PHRASE":
                features[1] += motif.score  # Verbality
            elif motif.pattern == "SUBJECT_AGENT":
                features[8] += motif.score  # Agentivity
            elif motif.pattern == "SUBJECT_PATIENT":
                features[9] += motif.score  # Patienthood
            elif motif.pattern == "PASSIVE_VOICE":
                features[21] += motif.score  # Passive voice
                features[9] += motif.score  # Also increases Patienthood
            elif motif.pattern == "WH_QUESTION":
                features[36] += motif.score  # Anaphora/deixis

            # ... map all 42 dimensions ...

        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)

        return features
```

#### Integration with Matryoshka Gate

**Gating with linguistic awareness:**

```python
class LinguisticMatryoshkaGate(MatryoshkaGate):
    """
    Matryoshka gate with linguistic feature awareness.

    Filters based on BOTH:
    1. Semantic similarity (existing)
    2. Linguistic compatibility (new)
    """

    def __init__(
        self,
        embedder: LinguisticMatryoshkaEmbeddings,
        config: GateConfig,
        linguistic_weight: float = 0.3  # 30% linguistic, 70% semantic
    ):
        super().__init__(embedder, config)
        self.linguistic_weight = linguistic_weight

    async def gate_with_linguistics(
        self,
        query: str,
        candidates: List[str],
        final_k: int = 10
    ) -> Tuple[List[int], List[GateResult]]:
        """
        Gate with linguistic + semantic scoring.

        Similarity = (1 - w) * semantic_sim + w * linguistic_sim
        where w = linguistic_weight
        """
        # Encode query + candidates with linguistics
        query_embeds = await self.embedder.encode_with_linguistics([query])
        candidate_embeds = await self.embedder.encode_with_linguistics(candidates)

        # Extract linguistic components
        query_linguistic = query_embeds[384][0, 384:]  # Last 42D
        candidate_linguistic = candidate_embeds[384][:, 384:]  # Last 42D

        # Progressive gating through scales
        active_indices = list(range(len(candidates)))
        gate_results = []

        for scale_idx, scale in enumerate(self.config.scales):
            # Get embeddings at this scale
            query_emb = query_embeds[scale][0]
            cand_emb = candidate_embeds[scale][active_indices]

            # Split semantic vs linguistic
            if scale == 96:
                # Semantic only
                semantic_scores = self._compute_similarity(query_emb, cand_emb)
                linguistic_scores = np.zeros(len(active_indices))
            elif scale == 192:
                # Partial linguistic
                semantic_scores = self._compute_similarity(query_emb[:184], cand_emb[:, :184])
                linguistic_scores = self._compute_similarity(query_emb[184:], cand_emb[:, 184:])
            else:  # 384
                # Full linguistic
                semantic_scores = self._compute_similarity(query_emb[:384], cand_emb[:, :384])
                linguistic_scores = self._compute_similarity(query_emb[384:], cand_emb[:, 384:])

            # Combine scores
            w = self.linguistic_weight if scale >= 192 else 0.0
            combined_scores = (1 - w) * semantic_scores + w * linguistic_scores

            # Apply gate
            threshold = self._get_threshold(scale_idx, combined_scores)
            kept_mask = self._apply_gate(combined_scores, threshold, scale_idx)

            # ... rest of gating logic ...
```

---

## Performance Impact

### Before (Semantic Only)

```
1000 candidates
  ↓ 96d embedding (all 1000) → 300ms
  ↓ Filter to 300
  ↓ 192d embedding (300) → 90ms
  ↓ Filter to 100
  ↓ 384d embedding (100) → 30ms
  ↓ Top 10

Total: ~420ms
```

### After (Linguistic Pre-Gate)

```
1000 candidates
  ↓ Linguistic filtering → 50ms (spaCy parse)
  ↓ Filter to 400 (60% reduction!)
  ↓ 96d embedding (400) → 120ms
  ↓ Filter to 120
  ↓ 192d embedding (120) → 36ms
  ↓ Filter to 40
  ↓ 384d embedding (40) → 12ms
  ↓ Top 10

Total: ~218ms (48% faster!)
```

**Key insight:** Linguistic filtering is CHEAP (parsing) compared to embedding (neural nets).

---

## Use Cases

### Use Case 1: Syntactic Query Routing

**Query:** `"Show me passive voice examples"`

**Linguistic gate:**
- Detects `WH_QUESTION` + looking for `PASSIVE_VOICE`
- Filters to candidates containing passive constructions
- Reduces 1000 candidates → 50

**Matryoshka gate:**
- Ranks 50 by semantic similarity
- Returns top 10

### Use Case 2: Grammatical Role Matching

**Query:** `"Who performed the action?"`

**Linguistic gate:**
- Detects `WH_QUESTION` + looking for `AGENT`
- Filters to candidates with prominent subjects
- Identifies sentences with active voice

**Matryoshka gate:**
- Ranks by semantic relevance

### Use Case 3: Structural Complexity Filtering

**Query:** `"Explain complex sentence structures"`

**Linguistic gate:**
- Detects `LONG_DISTANCE_DEP` and embedded clauses
- Filters to candidates with >2 clausal embedding depth
- Finds sophisticated structures

---

## Implementation Phases

### MVP: Linguistic Pre-Gate (2-3 hours)

**Minimal viable integration:**

```python
# In ResonanceShed or retrieval system
if config.use_linguistic_gate and query_has_syntactic_intent(query):
    # Quick syntactic filter
    candidates = linguistic_prefilter(query, candidates)

# Then standard matryoshka gating
final_indices = matryoshka_gate.gate(query, candidates)
```

**Files:**
- `HoloLoom/motif/linguistic.py` (already designed)
- `HoloLoom/embedding/linguistic_gate.py` (new, 150 lines)

### Full: Linguistic-Enhanced Embeddings (1 week)

**Complete integration:**

```python
# Enhanced embedder
embedder = LinguisticMatryoshkaEmbeddings(
    sizes=[96, 192, 384],
    enable_linguistic=True
)

# Enhanced gate
gate = LinguisticMatryoshkaGate(
    embedder,
    config,
    linguistic_weight=0.3
)
```

**Files:**
- `HoloLoom/embedding/linguistic_matryoshka.py` (new, 400 lines)
- `HoloLoom/embedding/matryoshka_gate.py` (extend existing)
- `HoloLoom/semantic_calculus/linguistic_dimensions.py` (42D space)

---

## Configuration

```python
# HoloLoom/config.py

@dataclass
class Config:
    # ... existing config ...

    # Linguistic gating
    use_linguistic_gate: bool = False  # Enable linguistic pre-filtering
    linguistic_gate_mode: str = "prefilter"  # "prefilter", "embedding", "both"
    linguistic_weight: float = 0.3  # Weight for linguistic features (0-1)

    # Matryoshka gating
    matryoshka_scales: List[int] = field(default_factory=lambda: [96, 192, 384])
    matryoshka_thresholds: List[float] = field(default_factory=lambda: [0.6, 0.75, 0.85])
```

---

## Advantages

1. **Speed**: Linguistic pre-filtering is cheap, reduces embedding load
2. **Precision**: Syntactic compatibility improves relevance
3. **Composability**: Two orthogonal dimensions (syntax + semantics)
4. **Graceful degradation**: Works without spaCy (falls back to semantic only)
5. **Progressive refinement**: Coarse → fine at BOTH linguistic and semantic levels

---

## Next Steps

1. **MVP Prototype** (2-3 hours):
   - Implement linguistic pre-gate
   - Test on sample queries
   - Measure performance impact

2. **Full Integration** (1 week):
   - Enhanced embeddings with 42D linguistic features
   - LinguisticMatryoshkaGate
   - Benchmarking

3. **Evaluation** (ongoing):
   - Precision/recall on linguistic queries
   - Speed improvements
   - User feedback

---

**Status:** Design complete, ready for implementation
**Depends on:** Chomsky linguistic integration (CHOMSKY_LINGUISTIC_INTEGRATION.md)
**Integrates with:** Matryoshka gate (HoloLoom/embedding/matryoshka_gate.py)
