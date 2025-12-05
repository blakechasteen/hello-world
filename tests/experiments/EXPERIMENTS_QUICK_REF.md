# 🧪 Experiments Quick Reference

## One-Line Summary
Automated tests that systematically compare fusion, complexity, budgets, and memory limits across 16 configurations.

---

## Run All Experiments
```powershell
python experiments/run_experiments.py
```
**Time**: ~1 second | **Output**: JSON + Markdown report

---

## What Gets Tested

### Experiment 1: Fusion Impact (2 runs)
```
❓ Question: Does multipass graph crawling help?
🎯 Tests: ON vs OFF
📊 Measures: Depth, quality, time overhead
💡 Answer: Shows if connected knowledge discovery is worth +1-2ms
```

### Experiment 2: Complexity Scaling (4 runs)
```
❓ Question: How does LITE → RESEARCH progression scale?
🎯 Tests: LITE, FAST, FULL, RESEARCH
📊 Measures: Passes, depth, memories, time
💡 Answer: Maps complexity level to performance/quality tradeoffs
```

### Experiment 3: Token Budget (5 runs)
```
❓ Question: What happens with different context sizes?
🎯 Tests: 2000, 3000, 4000, 6000, 8000 tokens
📊 Measures: Compression, usage %, quality
💡 Answer: Finds sweet spot between compression and quality
```

### Experiment 4: Memory Limits (5 runs)
```
❓ Question: Quality vs quantity - what's the balance?
🎯 Tests: 5, 8, 10, 15, 20 max memories
📊 Measures: Avg score, tokens, importance
💡 Answer: Shows precision vs recall tradeoff
```

---

## Results You'll Get

### JSON File (`all_experiments.json`)
```json
{
  "timestamp": "2025-10-30T02:08:57",
  "total_experiments": 16,
  "results": [
    {
      "experiment_name": "Fusion OFF",
      "memories_retrieved": 7,
      "max_depth": 0,
      "avg_composite_score": 0.883,
      "total_tokens": 170,
      "total_time_ms": 0.17
    }
  ]
}
```

### Markdown Report (`experiment_report.md`)
```markdown
## Experiment 1: Fusion Impact

| Metric | Fusion OFF | Fusion ON | Delta | Change % |
|--------|-----------|-----------|-------|----------|
| Memories Retrieved | 7 | 7 | +0 | +0.0% |
| Max Depth | 0 | 2 | +2 | +inf% |
```

---

## Key Metrics Explained

### Memory Metrics
- **Memories Retrieved**: How many items found
- **Max Depth**: Deepest graph hop (0 = direct, 1+ = connected)
- **Avg Composite Score**: Quality (0-1, higher = better)

### Packing Metrics  
- **Total Tokens**: Context size used
- **Token Usage %**: Efficiency (lower = less compression needed)
- **Compressed/Excluded**: Packing decisions

### Performance Metrics
- **Total Time**: End-to-end (target: <300ms)
- **Fusion Time**: Memory retrieval portion
- **Packing Time**: Context optimization portion

---

## Quick Interpretation Guide

### Fusion Impact Results
```
Max Depth: 0 → 2 hops
✅ Fusion discovered connected knowledge
❌ Fusion added overhead but no depth increase
```

### Complexity Scaling Results
```
LITE: 7 memories, depth 0, 0.08ms
RESEARCH: 12 memories, depth 3, 2.5ms
✅ Clear progression observed
❌ No scaling detected (check knowledge base size)
```

### Token Budget Results
```
Budget 2000: 13.1% usage, 2 compressed
Budget 8000: 3.3% usage, 0 compressed
✅ Compression reduces with larger budgets
💡 Sweet spot: 6-8% usage (comfortable headroom)
```

### Memory Limits Results
```
5 memories: Score 0.902 (high precision)
20 memories: Score 0.883 (high recall)
✅ Quality decreases slightly with more items
💡 Use 10-12 for balanced precision/recall
```

---

## Configuration Recommendations

### Speed-Critical (Chat, Real-Time)
```python
complexity = "LITE"       # 1 pass
use_fusion = False        # Skip graph crawling
max_memories = 5          # High precision
token_budget = 2000       # Small context
# Expected: <50ms
```

### Balanced (General Q&A)
```python
complexity = "FULL"       # 3 passes
use_fusion = True         # Enable graph crawling
max_memories = 10         # Balanced
token_budget = 4000       # Medium context
# Expected: <300ms
```

### Research (Deep Analysis)
```python
complexity = "RESEARCH"   # 4 passes
use_fusion = True         # Full graph traversal
max_memories = 20         # High recall
token_budget = 8000       # Large context
# Expected: No limit
```

---

## Typical Results

| Configuration | Memories | Depth | Tokens | Time | Use Case |
|--------------|----------|-------|--------|------|----------|
| LITE + Fusion OFF | 5-7 | 0 | 130 | <50ms | Chat |
| FAST + Fusion ON | 8-10 | 1 | 170 | <150ms | Quick Q&A |
| FULL + Fusion ON | 10-12 | 2 | 250 | <300ms | Standard |
| RESEARCH + Fusion ON | 15-20 | 3 | 350 | <1s | Deep dive |

---

## Troubleshooting

### No differences observed?
- Demo backend only has 7 items (limited)
- Connect to larger knowledge base for dramatic results
- Differences become clear with 100+ items

### Want more experiments?
Edit `run_experiments.py` and add:
```python
async def experiment_5_custom(self):
    # Your custom test
    pass
```

### Need visualizations?
Use the JSON output with:
- Python matplotlib
- Excel/Google Sheets
- Jupyter notebooks
- Web visualization libraries

---

## Files Created

```
experiments/
├── run_experiments.py          # Main experiment runner
├── EXPERIMENTS_GUIDE.md        # Detailed guide
├── EXPERIMENTS_QUICK_REF.md    # This file
└── results/
    ├── all_experiments.json    # Raw data
    └── experiment_report.md    # Formatted report
```

---

## Related Commands

```powershell
# Run experiments
python experiments/run_experiments.py

# View results
cat experiments/results/experiment_report.md

# Parse JSON
python -c "import json; print(json.load(open('experiments/results/all_experiments.json')))"

# Launch UI to test manually
python ui/consciousness_ui_simple.py
```

---

## Summary

**What**: 16 automated experiments testing 4 key parameters
**Why**: Understand tradeoffs and optimize configuration
**How**: One command, generates JSON + Markdown report
**Time**: ~1 second to run all tests
**Output**: Quantitative comparison data

**Status**: 🟢 READY TO RUN

---

**More Info**: See `experiments/EXPERIMENTS_GUIDE.md` for detailed documentation
