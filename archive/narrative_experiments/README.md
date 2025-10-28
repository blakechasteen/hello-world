# Narrative Experiments Archive

**Archived:** 2025-10-27
**Reason:** Consolidated into `hololoom_narrative` framework
**Status:** Reference only

---

## Files Archived

These experimental files were used to develop the narrative intelligence features now in `hololoom_narrative/`.

### Bayesian NLP Experiments
- `bayesian_narrative_nlp.py` (21KB) - Bayesian approach to narrative analysis
- `enhanced_odyssey_bayesian.py` (11KB) - Enhanced Odyssey analysis
- `real_odyssey_bayesian_test.py` (14KB) - Test suite
- `temporal_bayesian_evolution.py` (17KB) - Temporal Bayesian evolution

### Sentiment/Depth Protocols
- `narrative_depth_protocol.py` (17KB) - Depth protocol definition
- `piercing_sentiment_narrative.py` (30KB) - Sentiment piercing experiments
- `piercing_sentiment_final.py` (26KB) - Final sentiment version

**Total:** ~156KB of experimental code

---

## Production Equivalents

These experiments led to production features in `hololoom_narrative/`:

| Experiment | Production Location |
|------------|-------------------|
| `narrative_depth_protocol.py` | `hololoom_narrative/matryoshka_depth.py` |
| `bayesian_narrative_nlp.py` | `hololoom_narrative/intelligence.py` |
| `piercing_sentiment_*.py` | Integrated into narrative intelligence |

---

## Why Archived?

**Before:** Experiments scattered in `dev/`
**After:** Clean production code in `hololoom_narrative/`

**Benefits:**
- Clean `dev/` directory
- Clear framework boundary
- Preserved for reference
- Won't confuse new developers

---

## Usage

These files are **reference only**. Do not import or use in production.

For production narrative features, use:
```python
from hololoom_narrative import NarrativeIntelligence, MatryoshkaNarrativeDepth
```

---

**Status:** Archived, not deprecated
**Maintained:** No
**Reference:** Yes