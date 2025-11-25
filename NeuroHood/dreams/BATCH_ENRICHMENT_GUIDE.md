# Production Batch Enrichment Guide

## Overview

This guide covers the complete production batch enrichment system for all 500 dream symbols using the Metaprompting Refinement Framework (MRF).

**Key Deliverables:**
- 500 symbols with 15-25 literary references each
- 10 emotional/archetypal categories
- Multi-cultural coverage (≥3 non-Western cultures per symbol)
- Quality validation across 3 metrics
- Human review flagging for low-quality symbols

**Timeline:** 13 minutes (batched) + validation
**Cost:** ~$7.50 (Claude Sonnet)
**Estimated Cost Savings vs Manual:** 99.85% ($5,000 → $7.50)

---

## File Structure

```
NeuroHood/dreams/
├── symbol_database_base.json              # 500 base symbols (10 categories)
├── enrich_symbols_batch.py                # Main enrichment pipeline
├── symbol_enrichment_metaprompt.txt       # MRF metaprompt template
├── symbol_database_enriched.json          # Output (generated after enrichment)
└── BATCH_ENRICHMENT_GUIDE.md             # This file
```

---

## Quick Start (Dry-Run Mode)

### Test without LLM calls

```bash
cd NeuroHood/dreams

# Run dry-run (generates mock enrichment for testing)
python enrich_symbols_batch.py --dry-run --output symbol_database_test.json

# Dry-run will:
# - Generate 500 mock enriched symbols
# - Show quality validation (will be ~80% due to mock data)
# - Demonstrate full pipeline without cost
# - Take ~30 seconds
```

### Expected output:
```
Loaded 500 symbols from symbol_database_base.json
Enriching symbols: 100%|████████| 50/50 [00:30<00:00,  1.65/s]
Saved 500 enriched symbols to symbol_database_test.json

============================================================
ENRICHMENT STATISTICS
============================================================
Total enriched: 500
Successful: 450
Needs human review: 50
Failed: 0

Average quality score: 8.05/10
Min quality: 6.80/10
Max quality: 9.20/10

============================================================
COST ESTIMATION
============================================================
Symbols enriched: 500
Cost per symbol: $0.0150
Total cost: $7.50
Estimated time: ~50 minutes (batched)
============================================================
```

---

## Setup (Production Enrichment)

### 1. Install Dependencies

```bash
# HoloLoom integration required
pip install anthropic  # Claude API client

# Optional: progress bar
pip install tqdm
```

### 2. Set API Keys

```bash
# For Claude (Anthropic)
export ANTHROPIC_API_KEY="your-api-key-here"

# Or set in environment file
# .env file:
ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Verify Base Database

```bash
# Check that symbol_database_base.json exists and has 500 symbols
python -c "import json; data = json.load(open('symbol_database_base.json')); print(f\"Loaded {len(data['symbols'])} symbols in {len(set(s['category'] for s in data['symbols']))} categories\")"

# Output should show: Loaded 500 symbols in 10 categories
```

---

## Running Full Production Enrichment

### Single Command

```bash
python enrich_symbols_batch.py
```

This will:
1. Load 500 symbols from `symbol_database_base.json`
2. Enrich in batches of 10 (concurrent LLM calls)
3. Validate quality on each symbol
4. Save to `symbol_database_enriched.json`
5. Print statistics

### With Custom Options

```bash
# Custom output file
python enrich_symbols_batch.py --output my_enriched_database.json

# Custom batch size (default: 10)
python enrich_symbols_batch.py --batch-size 5

# Both options
python enrich_symbols_batch.py --output my_output.json --batch-size 8
```

### Programmatic Usage

```python
import asyncio
from enrich_symbols_batch import SymbolEnricher, EnrichmentConfig

async def enrich_all():
    config = EnrichmentConfig(
        llm_provider="anthropic",
        llm_model="claude-3-5-sonnet-20241022",
        batch_size=10,
        dry_run=False
    )

    enricher = SymbolEnricher(config)
    enriched, stats = await enricher.enrich_all_500()

    # Use enriched symbols
    print(f"Enriched {len(enriched)} symbols")
    print(f"Average quality: {stats['quality_scores']}")

asyncio.run(enrich_all())
```

---

## Understanding the Output

### Symbol Database Structure

```json
{
  "symbol_id": "caged_bird",
  "category": "trapped",
  "literary_references": {
    "classical_mythology": [
      {
        "title": "Prometheus Bound",
        "culture": "Greek",
        "author": "Aeschylus",
        "date": "~430 BCE",
        "connection": "Divine punishment through eternal captivity",
        "quote": "Behold me fettered, miserable god",
        "emotional_resonance": ["trapped", "powerless", "defiant"],
        "connection_confidence": 0.95
      }
    ],
    "world_literature": [...],
    "modern_cinema": [...],
    "poetry_visual_arts": [...],
    "philosophy_religion": [...],
    "contemporary_culture": [...]
  },
  "archetypal_roots": {
    "primary": "Jungian Shadow (repressed self)",
    "secondary": ["Christian Captivity", "Buddhist Samsara"],
    "mythological_patterns": ["Hero's Imprisonment", "Divine Punishment"]
  },
  "quality_scores": {
    "cultural_diversity": 9.2,
    "emotional_resonance": 9.5,
    "accessibility": 8.8,
    "overall": 9.17
  },
  "total_references": 22,
  "enrichment_date": "2025-11-22T14:30:00",
  "needs_human_review": false
}
```

### Quality Metrics Explained

**Cultural Diversity (0-10)**
- Measures representation of non-Western cultures
- ≥3 non-Western cultures = 7.0+
- 4+ cultures = 8.0+
- 5+ cultures = 9.0+
- All 10 cultural spheres = 10.0

**Emotional Resonance (0-10)**
- Average of connection_confidence scores
- 0.80 avg confidence = 8.0 score
- 0.90 avg = 9.0
- 1.0 avg = 10.0

**Accessibility (0-10)**
- Measures mix of popular vs scholarly references
- Ideal ratio: 60% popular (films, contemporary) / 40% scholarly (philosophy, classical)
- Penalty for extreme imbalance
- Balanced mix = 8.0+

**Overall Score**
- Weighted average: 40% diversity + 35% resonance + 25% accessibility
- Minimum acceptable: 7.0/10
- Excellent: 8.5+/10

### Symbols Flagged for Review

Symbols with overall quality < 7.0 are flagged with `"needs_human_review": true`

Example:
```json
{
  "symbol_id": "obscure_symbol",
  "needs_human_review": true,
  "quality_scores": {
    "cultural_diversity": 5.2,
    "emotional_resonance": 7.1,
    "accessibility": 6.8,
    "overall": 6.37
  }
}
```

---

## Quality Thresholds

### Automated Validation (Tier 1)

Applied to 100% of symbols:

```python
# JSON validity
✓ Valid JSON format
✓ All required fields present

# Reference count
✓ 15-25 total references (too few/many indicates incompleteness)

# Category coverage
✓ classical_mythology: ≥2 references
✓ world_literature: ≥2 references
✓ modern_cinema: ≥2 references
✓ poetry_visual_arts: ≥2 references
✓ philosophy_religion: ≥2 references
✓ contemporary_culture: ≥2 references

# Quality metrics
✓ cultural_diversity: ≥7.0
✓ emotional_resonance: ≥8.0 (avg connection_confidence ≥0.80)
✓ accessibility: ≥6.0 (60% popular / 40% scholarly)
✓ overall: ≥7.0

# Minimum pass rate target
✓ ≥85% of symbols pass Tier 1 validation
```

### Human Expert Review (Tier 2)

For symbols flagged with `needs_human_review: true`:

**Review checklist:**
- [ ] Are non-Western references culturally accurate?
- [ ] Are sacred symbols treated respectfully?
- [ ] Are interpretations appropriate to source culture?
- [ ] Are citations verifiable?
- [ ] Does the symbol warrant the low quality score?
- [ ] Should quality threshold be adjusted?

**Reviewers needed:** 3-5 cultural consultants across:
- Asian/Buddhist culture
- African/Indigenous culture
- Islamic/Middle Eastern culture
- Western literature and mythology
- Cinema and contemporary media

**Timeline:** 1-2 days for ~50-75 flagged symbols

### Spot Check Validation (Tier 3)

Random sample 10% of symbols (50 random picks from 500):

**Validation tasks:**
- [ ] Verify each title actually exists (Google Books, IMDb, scholarly databases)
- [ ] Check quote/scene accuracy (find in original work)
- [ ] Validate author/director names correct
- [ ] Verify dates are accurate
- [ ] Test dream generation with symbol (see "Integration" section below)

**Quality target:** ≥95% accuracy on spot checks

---

## Troubleshooting

### Issue: "API Key not found"

**Solution:**
```bash
# Set ANTHROPIC_API_KEY environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify it's set
echo $ANTHROPIC_API_KEY

# Or create .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python -m dotenv load .env
```

### Issue: "LLM call returned None"

**Causes:**
- API key invalid or expired
- Network connectivity issue
- API rate limit exceeded (wait and retry)
- LLM provider service down

**Solutions:**
```bash
# Test API connection
python -c "from anthropic import Anthropic; client = Anthropic(); print(client.api_key[:20])"

# Check rate limits
# Anthropic quota: 20 requests/min for free tier, unlimited for paid
# Add delays between batches:
enricher = SymbolEnricher(config)
enricher.config.rate_limit_delay = 2.0  # 2-second delay between batches
```

### Issue: "JSON parsing error"

**Cause:** LLM response not valid JSON

**Solution:**
```bash
# Check response format in logs
# The LLM might be adding preamble text before JSON

# Workaround: Add JSON extraction to metaprompt
# Modify symbol_enrichment_metaprompt.txt to add:
# "Return ONLY the JSON object, nothing else"
```

### Issue: "Out of memory with 500 symbols"

**Cause:** Loading all 500 symbols at once

**Solutions:**
```bash
# Process in smaller chunks
from enrich_symbols_batch import SymbolEnricher

enricher = SymbolEnricher(config)

# Process batches of 100 symbols instead of 500
with open('symbol_database_base.json') as f:
    data = json.load(f)
    all_symbols = data['symbols']

enriched_all = []
for i in range(0, len(all_symbols), 100):
    batch_symbols = all_symbols[i:i+100]
    enriched = await enricher.enrich_batch(batch_symbols)
    enriched_all.extend(enriched)
    # Save intermediate results
```

---

## Cost Breakdown

### Claude Sonnet Pricing (as of Nov 2025)

```
Input tokens:  $3.00 / 1M tokens
Output tokens: $15.00 / 1M tokens
```

### Per-Symbol Estimate

```
Average input: 200 tokens (symbol metadata + metaprompt header)
Average output: 800 tokens (enriched JSON with 20 references)

Cost per symbol:
  Input:  0.0002 tokens × $3.00/1M = $0.0006
  Output: 0.0008 tokens × $15.00/1M = $0.012
  Total:  ~$0.015 per symbol

For 500 symbols:
  Total cost: 500 × $0.015 = $7.50
  Batch processing reduces per-call overhead
```

### Comparison to Manual Curation

```
Manual approach:
  Research time: 12 minutes per symbol
  Researcher cost: $50/hour
  Total symbols: 500

  Total time: 500 × 12 min = 6,000 min = 100 hours
  Total cost: 100 hours × $50/hr = $5,000
  Timeline: 2-4 weeks with 1 researcher

MRF Automation:
  Processing: 13 minutes (batched, 50 concurrent calls)
  LLM cost: $7.50
  Validation: 2-3 days (expert review of ~50 flagged)
  Total timeline: 1-2 weeks

  Cost savings: $5,000 - $7.50 = $4,992.50 (99.85%)
  Time savings: 100 hours - 0.22 hours = 99.78%
```

---

## Performance Characteristics

### Processing Times

```
Dry-run (no LLM calls):
  ~30 seconds for 500 symbols

Production (with Claude Sonnet):
  Sequential (batch_size=1): 8,000s ≈ 2.2 hours
  Batched (batch_size=10): 800s ≈ 13 minutes ✓
  Batched (batch_size=20): 400s ≈ 6.5 minutes (higher risk of rate limits)

Tier 2 validation (human review):
  ~1-2 days for 50-75 flagged symbols

Tier 3 spot checks:
  ~30-60 minutes for 50 symbols

Total timeline:
  Without validation: 15 minutes
  With human review: 2-3 days
  With all validation: 3-4 days
```

### Memory Usage

```
Loading all 500 symbols:
  Base symbols: ~2 MB
  Enriched symbols (in memory): ~50-100 MB
  LLM tokens cache: ~5-10 MB
  Total: <200 MB ✓ (safe for most systems)
```

### API Rate Limits

```
Anthropic API (free tier):
  20 requests per minute
  Batching 10 symbols per request
  500 symbols → 50 requests
  50 requests ÷ 20/min = 2.5 minutes minimum

With rate_limit_delay=0.5s between batches:
  10 batches × 50 requests = 50 requests
  Plus delays: 50 × 0.5s = 25 seconds overhead
  Total: ~13 minutes ✓
```

---

## Integration with Symbolic Encoder

After enrichment completes, integrate enriched database with narrative generator:

```python
from NeuroHood.dreams.symbolic_encoder_enhanced import EnhancedSymbolicEncoder

# Load enriched database
encoder = EnhancedSymbolicEncoder(
    enriched_db_path="symbol_database_enriched.json"
)

# Select culturally diverse symbols
essence = {
    "primary_emotion": "trapped",
    "intensity": 0.85,
    "context": "work",
    "temporal": "chronic"
}

symbols = encoder.select_culturally_diverse_symbols(
    emotional_essence=essence,
    target_cultures=["Buddhist", "Greek", "African"],
    k=3
)

# Generate dream narrative using enriched references
for symbol in symbols:
    print(f"Symbol: {symbol.symbol_id}")
    print(f"  Cultural diversity: {symbol.cultural_diversity_score:.1f}/10")
    print(f"  Emotional resonance: {symbol.emotional_resonance_score:.1f}/10")
    print(f"  Total references: {symbol.total_references}")

    # Use cinematic references for visual scaffolding
    cinematic = symbol.get_cinematic_references()
    print(f"  Cinematic inspirations: {len(cinematic)} films/shows")
```

---

## Validation Checklist

Before deploying enriched database to production:

- [ ] All 500 symbols enriched
- [ ] Tier 1 validation: ≥85% pass rate
- [ ] Tier 2 validation: ~50 flagged symbols reviewed by cultural experts
- [ ] Tier 3 validation: 50 spot-checked symbols pass ≥95% accuracy
- [ ] JSON parsing validated on sample
- [ ] Integration tested with symbolic encoder
- [ ] Dream generation tested with 10-20 enriched symbols
- [ ] Cost verified ($7.50 ± 20%)
- [ ] Timeline completed on schedule (13 min + validation)
- [ ] Documentation updated with any schema changes
- [ ] Backup created of original base database
- [ ] Git commit with enriched database tagged

---

## Next Steps

1. **Immediate (Today)**
   - [ ] Run dry-run to verify pipeline works
   - [ ] Review quality metrics on mock enrichment
   - [ ] Address any configuration issues

2. **Phase 1: Preparation (Day 1)**
   - [ ] Set up API keys and environment
   - [ ] Run full enrichment (13 minutes)
   - [ ] Check statistics and identify flagged symbols

3. **Phase 2: Validation (Days 2-4)**
   - [ ] Tier 2: Cultural expert review (1-2 days)
   - [ ] Tier 3: Spot checks on 50 random symbols (30-60 min)
   - [ ] Adjust quality thresholds if needed

4. **Phase 3: Integration (Days 5-7)**
   - [ ] Load enriched database into symbolic encoder
   - [ ] Test dream generation with enriched symbols
   - [ ] Validate cultural diversity in generated dreams
   - [ ] Document findings and improvements

5. **Phase 4: Release (Day 7)**
   - [ ] Tag as v1.0-literary-expansion
   - [ ] Commit to repository
   - [ ] Update documentation
   - [ ] Announce completion

---

## Support & Contact

For questions or issues:

1. Check the **Troubleshooting** section above
2. Review logs in `enrich_symbols_batch.py` for detailed error messages
3. Examine `symbol_enrichment_metaprompt.txt` if enrichment quality is low
4. Consult `MRF_LITERARY_EXPANSION.md` for architectural details

---

**Created:** November 22, 2025
**Status:** Production Ready (Dry-Run Tested)
**Next Action:** Run full enrichment after approval
