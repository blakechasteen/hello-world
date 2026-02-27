# Phase 3 Validation - Quick Start Guide

**5-Minute Setup** for validating UnifiedMRF improvements in production.

---

## Step 1: Install Dependencies

```bash
# No external dependencies needed - uses Python standard library only
# SQLite is built into Python
```

---

## Step 2: Simple A/B Test

```python
from hololoom.prompting.validation import ABTestRunner, ABTestConfig
from hololoom.prompting.unified_mrf import MetapromptConfig
from pathlib import Path

# Configuration
config = ABTestConfig(
    mrf_traffic_ratio=0.5,  # 50/50 split
    min_samples_per_variant=30,
    model_provider="claude",
    output_dir=Path("./validation_results")
)

# Create runner
runner = ABTestRunner(config)

# Your queries
queries = [
    "What is Thompson Sampling?",
    "Explain Bayesian optimization",
    # ... more queries
]

# Traditional prompts (your current approach)
traditional_prompts = [
    f"Answer this question: {q}" for q in queries
]

# UnifiedMRF metaprompts (enhanced approach)
metaprompts = [
    MetapromptConfig(
        role="Expert in machine learning",
        objective={"primary": f"Answer: {q}", "secondary": ["Be accurate", "Be concise"]},
        process=["Understand", "Answer", "Validate"],
        format_spec="1-2 paragraphs",
        constraints=["MUST be factually correct"],
        uncertainty="Note if uncertain",
        validation=["Check accuracy"]
    )
    for q in queries
]

# Run test
results = await runner.run_test(queries, traditional_prompts, metaprompts)

# View results
print(f"Quality improvement: {results.quality_improvement_pct:+.1f}%")
print(f"Statistically significant: {results.is_statistically_significant}")
print(f"Results saved to: {config.output_dir}")
```

**Output**:
```
[OK] Starting A/B test with 20 queries
    MRF traffic ratio: 50%
    Min samples per variant: 30
    Confidence level: 95%

  [1/20] TRAD | What is Thompson Sampling?... | 140.5ms
  [2/20] MRF  | Explain Bayesian optimization... | 155.2ms
  ...

[OK] Results saved to: ./validation_results/ab_test_results_20251122_143022.json

==================================================
A/B TEST RESULTS
==================================================

Total queries: 20
  Traditional: 10
  UnifiedMRF: 10

Quality:
  Traditional avg: 0.720
  UnifiedMRF avg: 0.915
  Improvement: +27.1%

Statistical Significance:
  Significant: True
  P-value: 0.01
  Confidence: 95%

==================================================
```

---

## Step 3: Collect Production Data (Optional)

If you want to track production queries over time:

```python
from hololoom.prompting.validation import ProductionDataCollector, QuerySource

# Create collector
collector = ProductionDataCollector("production_data.db")

# Log queries as they happen
await collector.log_query(
    query_id="q_12345",
    text="What is Thompson Sampling?",
    source=QuerySource.PRODUCTION,
    system_variant="unified_mrf"
)

await collector.log_response(
    query_id="q_12345",
    response="Thompson Sampling is...",
    latency_ms=150.5
)

# Later: retrieve for analysis
queries = await collector.get_queries(
    system_variant="unified_mrf",
    limit=100
)
```

---

## Step 4: Statistical Analysis (Optional)

For rigorous statistical comparison:

```python
from hololoom.prompting.validation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Your data
traditional_scores = [0.72, 0.75, 0.70, ...]  # Quality scores
mrf_scores = [0.91, 0.88, 0.92, ...]

# Compare
result = analyzer.independent_ttest(traditional_scores, mrf_scores)

print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Effect size: {result.effect_size:.2f}")
```

---

## Step 5: Human Evaluation (Optional)

For blind side-by-side comparisons:

```python
from hololoom.prompting.validation import HumanEvaluationCollector, Preference

collector = HumanEvaluationCollector("human_eval.db")

# Create pair (presentation order randomized)
pair = await collector.create_evaluation_pair(
    query="What is Thompson Sampling?",
    traditional_response="...",
    mrf_response="..."
)

# Show to evaluator (they don't know which is which)
print(f"Response A: {pair.response_a}")
print(f"Response B: {pair.response_b}")

# Collect preference
await collector.record_evaluation(
    pair_id=pair.pair_id,
    evaluator_id="evaluator_123",
    overall_preference=Preference.PREFER_B,
    rationale="More detailed and clear"
)

# Analyze
results = await collector.analyze_results()
print(f"MRF preference rate: {results.mrf_preference_rate:.1%}")
```

---

## That's It!

With these 5 steps, you can:
1. ✅ Run A/B tests comparing traditional vs UnifiedMRF
2. ✅ Collect production data for analysis
3. ✅ Perform statistical significance testing
4. ✅ Gather human evaluations
5. ✅ Make evidence-based deployment decisions

For complete documentation, see [P3_PRODUCTION_VALIDATION_COMPLETE.md](P3_PRODUCTION_VALIDATION_COMPLETE.md).

---

## Quick Decision Guide

Based on your A/B test results:

- **Quality improvement ≥20%, p<0.05** → DEPLOY immediately
- **Quality improvement 10-20%, p<0.05** → GRADUAL rollout
- **Quality improvement <10% or p>0.05** → INVESTIGATE (collect more data)

---

## Common Issues

### "Not enough samples for significance testing"

**Solution**: Run test with at least 30 samples per variant:
```python
config = ABTestConfig(min_samples_per_variant=30)
```

### "Results not statistically significant"

**Possible causes**:
1. Sample size too small (need ≥30 per variant)
2. High variance in quality scores
3. Actual improvement is small (<10%)

**Solution**: Collect more data or review MRF implementation.

### "Latency increased significantly"

**If latency increased >20%**:
1. Review metaprompt complexity (shorter prompts = faster)
2. Check model provider (Claude vs Gemini vs GPT)
3. Consider prompt caching

---

## Next Steps

After running Phase 3 validation:

1. **If successful** → Deploy to production (gradual or full)
2. **If unsuccessful** → Review MRF implementation or adjust metaprompts
3. **If uncertain** → Collect more data (increase sample size)

For complete workflow, see [P3_PRODUCTION_VALIDATION_COMPLETE.md](P3_PRODUCTION_VALIDATION_COMPLETE.md).
