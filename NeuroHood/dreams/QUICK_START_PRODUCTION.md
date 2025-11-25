# Quick Start: Production Batch Enrichment

**TL;DR**: Run 3 commands to enrich 500 symbols in 20 minutes

---

## Prerequisites

```bash
# 1. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Verify installation
python3 -c "from anthropic import Anthropic; print('OK')"
```

**No API key?** Get one: https://console.anthropic.com/

---

## Command 1: Prepare Database

```bash
cd /c/Users/blake/OneDrive/Documents/mythRL

python3 NeuroHood/dreams/create_base_500_symbols_v2.py
```

**Output**: `symbol_database_base_500.json` (500 symbols)

**Time**: < 1 minute

**Expected**:
```
✓ Saved 500 symbols to symbol_database_base_500.json
  File size: 150 KB
  Categories: 10
  Symbols per category: 50
```

---

## Command 2: Test with 3 Symbols (Optional but Recommended)

```bash
python3 NeuroHood/dreams/test_batch_enrichment_production.py
```

**Tests**: Real LLM enrichment on 3 symbols

**Time**: 1-2 minutes

**Expected**:
```
Enriching symbol: caged_bird
  ✓ caged_bird: 18 refs, quality=8.15, time=30s

Enriching symbol: broken_mirror
  ✓ broken_mirror: 20 refs, quality=8.32, time=28s

Enriching symbol: sunrise
  ✓ sunrise: 19 refs, quality=8.24, time=32s

EXTRAPOLATION TO 500 SYMBOLS
  Estimated total time: 15.0 minutes
  Estimated tokens: 175,000
  Estimated cost: $5.25
```

---

## Command 3: Enrich All 500 Symbols

```bash
python3 NeuroHood/dreams/enrich_symbols_batch.py \
    --input NeuroHood/dreams/symbol_database_base_500.json \
    --output NeuroHood/dreams/symbol_database_enriched_500.json
```

**Configuration** (optional):
```bash
# Faster test (10 symbols only)
python3 NeuroHood/dreams/enrich_symbols_batch.py \
    --input symbol_database_base_500.json \
    --max-symbols 10

# With custom batch size
python3 NeuroHood/dreams/enrich_symbols_batch.py \
    --input symbol_database_base_500.json \
    --batch-size 5  # Slower but more stable

# Dry-run (no API calls, uses mock data)
python3 NeuroHood/dreams/enrich_symbols_batch.py \
    --input symbol_database_base_500.json \
    --dry-run
```

**Time**: 15-20 minutes

**Expected**:
```
Processing batch 1 [========>                    ]  10/500
Processing batch 2 [==================>          ]  50/500
...
Processing batch 50 [================================] 500/500

RESULTS
  Successful: 475/500 (95%)
  Failed: 25/500 (5%)
  Avg quality: 8.15/10
  Total cost: $7.35
  Total time: 17 minutes

✓ Results saved to symbol_database_enriched_500.json
```

---

## What You Get

### Output File: `symbol_database_enriched_500.json`

Each enriched symbol contains:

```json
{
  "symbol_id": "caged_bird",
  "category": "trapped",
  "description": "A bird in a cage, yearning for freedom",
  "emotion_tags": ["trapped", "powerless", "yearning"],
  "literary_references": {
    "classical_mythology": [
      {
        "title": "Prometheus Bound",
        "culture": "Greek",
        "author": "Aeschylus",
        "connection": "Chained god suffering eternal punishment",
        "quote": "My ceaseless agony shall know no term",
        "connection_confidence": 0.95
      },
      // ... more references
    ],
    "world_literature": [ /* ... */ ],
    "modern_cinema": [ /* ... */ ],
    "poetry_visual_arts": [ /* ... */ ],
    "philosophy_religion": [ /* ... */ ],
    "contemporary_culture": [ /* ... */ ]
  },
  "total_references": 19,
  "quality_scores": {
    "cultural_diversity": 9.2,
    "emotional_resonance": 8.8,
    "accessibility": 7.5,
    "overall": 8.32
  },
  "needs_human_review": false
}
```

---

## Validation Checklist

After enrichment, verify:

```bash
# Count symbols
python3 -c "import json; db=json.load(open('symbol_database_enriched_500.json')); print(f'Symbols: {len(db)}')"
# Expected: 500

# Check quality
python3 -c "
import json
db = json.load(open('symbol_database_enriched_500.json'))
qualities = [s['quality_scores']['overall'] for s in db]
print(f'Avg quality: {sum(qualities)/len(qualities):.2f}')
print(f'Min quality: {min(qualities):.2f}')
print(f'Max quality: {max(qualities):.2f}')
print(f'Pass rate: {sum(1 for q in qualities if q >= 7.0)}/{len(qualities)}')
"
# Expected: avg ~8.15, pass rate 95%+

# Check categories
python3 -c "
import json
from collections import Counter
db = json.load(open('symbol_database_enriched_500.json'))
cats = Counter(s['category'] for s in db)
for cat, count in sorted(cats.items()):
    print(f'{cat}: {count}')
"
# Expected: ~50 per category
```

---

## Troubleshooting

### Error: `401 Unauthorized`
```
Solution: Check API key
echo $ANTHROPIC_API_KEY  # Should show "sk-ant-..."
export ANTHROPIC_API_KEY="<correct-key>"
```

### Error: `rate_limit_error`
```
Solution: Reduce batch size
python3 enrich_symbols_batch.py --batch-size 5
```

### Error: `JSON decode error`
```
Solution: Retry automatically happens
If persistent, check that symbol_enrichment_metaprompt.txt exists
ls -la NeuroHood/dreams/symbol_enrichment_metaprompt.txt
```

### Slow performance
```
Solution: Check batch size
Recommended: batch-size 10 (default)
If rate-limited: batch-size 5 (slower but stable)
```

---

## Cost & Time Summary

| Phase | Time | Cost |
|-------|------|------|
| Prepare database | 1 min | $0 |
| Test (optional) | 2 min | ~$0.10 |
| Enrich 500 symbols | 15-20 min | ~$7-8 |
| **Total** | **18-23 min** | **~$7-8** |

---

## Next Steps

1. ✅ Run all 3 commands above
2. ✅ Validate results using checklist
3. ⏳ Human review of ~25 flagged symbols (5%)
4. ⏳ Use enriched database for dream generation!

---

## Files Reference

| File | Purpose | Location |
|------|---------|----------|
| `symbol_database_base_500.json` | Input (500 unenriched symbols) | `NeuroHood/dreams/` |
| `symbol_database_enriched_500.json` | Output (enriched with references) | `NeuroHood/dreams/` |
| `enrichment_batch.log` | Execution log | `NeuroHood/dreams/` |
| `test_enrichment_results.json` | Test results (if using test command) | `NeuroHood/dreams/` |

---

## Complete Docs

- 📖 **Full Setup**: `BATCH_ENRICHMENT_GUIDE.md`
- ✅ **Checklist**: `PRODUCTION_READINESS_CHECKLIST.md`
- 📊 **Summary**: `PRODUCTION_BATCH_SUMMARY.md`
- 📝 **Pilot Results**: `PILOT_ENRICHMENT_REPORT.md`

---

**Ready?** Run: `python3 NeuroHood/dreams/create_base_500_symbols_v2.py` 🚀
