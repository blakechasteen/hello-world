# Chain of Verification (CoVe) Integration with MRF

**Status**: ✅ Integration Guide (December 2025)
**Location**: `HoloLoom/prompting/verification_integration.py`
**Purpose**: Enable Chain of Verification as an MRF refinement strategy

## Overview

Chain of Verification (CoVe) is a systematic approach to improving response quality through multi-step verification:
1. **Claim Extraction** - Identify atomic claims in the response
2. **Question Generation** - Create questions to verify each claim
3. **Contradiction Detection** - Check for inconsistencies
4. **Refinement** - Improve response based on findings

When integrated with MRF (Metaprompting Refinement Framework), CoVe provides a principled, measurable approach to quality improvement with +25-35% quality gains.

## Enable CoVe Strategy

Add CoVe as an MRF refinement strategy:

```python
from HoloLoom.prompting.verification_integration import enable_cove_for_mrf
from HoloLoom.prompting.unified_mrf import UnifiedMRF, RefinementStrategy

# Create MRF engine
mrf = UnifiedMRF(model_provider="claude")

# Enable CoVe strategy
enable_cove_for_mrf(mrf)

# Use CoVe in refinement
result = mrf.refine(
    original_prompt="Explain Thompson Sampling",
    strategy=RefinementStrategy.VERIFY,  # Or auto-select (will choose CoVe for factual)
    context={"domain": "machine_learning"}
)

print(f"Quality: {result['quality_score']:.2f}")
print(f"Strategy: {result['strategy_used']}")  # Will be "cove" for verify mode
```

## Configure Verification Thresholds

Control when CoVe verification is triggered:

```python
from HoloLoom.prompting.verification_integration import (
    VerificationMRFBridge,
    VerificationConfig
)

config = VerificationConfig(
    confidence_threshold=0.75,  # Apply CoVe when confidence < 75%
    min_claims=2,               # Minimum claims to trigger verification
    max_claims=10,              # Maximum claims to verify (cost control)
    contradiction_threshold=0.3,# Flag contradictions with >30% conflict
    timeout_seconds=5.0         # Verification timeout
)

bridge = VerificationMRFBridge(config=config)

result = await bridge.enhance_with_cove(
    query="What is Thompson Sampling?",
    response="Thompson Sampling is a Bayesian approach...",
    confidence=0.68  # Below threshold → triggers CoVe
)

print(f"Verification triggered: {result.verification_applied}")
print(f"Quality improvement: {result.quality_improvement:.1%}")
```

## Monitor via Dashboard

Track CoVe effectiveness in real-time:

```python
from HoloLoom.prompting.analytics import create_dashboard
from HoloLoom.prompting.verification_integration import create_verification_bridge

# Create dashboard with verification tracking
dashboard = create_dashboard(enable_learning=True)
bridge = create_verification_bridge()

# Process queries and log results
for query in queries:
    result = await bridge.enhance_with_cove(query=query, response=response)

    dashboard.log_enhancement(
        system="verification",
        query=query,
        quality_before=result.quality_before,
        quality_after=result.quality_after,
        execution_time_ms=result.verification_time_ms,
        metadata={"claims": result.claims_count}
    )

# View verification metrics
metrics = dashboard.get_verification_metrics()
print(f"Avg quality improvement: {metrics['avg_improvement']:.1%}")
print(f"Avg verification time: {metrics['avg_time_ms']:.1f}ms")
print(f"Contradiction detection rate: {metrics['contradiction_rate']:.1%}")

# Generate report
html = dashboard.generate_report(format="html")
dashboard.save_report("verification_dashboard.html")
```

## A/B Testing: CoVe vs VERIFY Mode

Compare CoVe verification with standard VERIFY refinement:

```python
from HoloLoom.prompting.analytics import create_dashboard

dashboard = create_dashboard(enable_ab_testing=True)

# Create A/B test
dashboard.create_ab_test(
    name="cove_vs_verify",
    control_description="Standard VERIFY refinement",
    treatment_description="CoVe verification with claim analysis",
    traffic_split=0.5  # 50/50 split
)

# Process queries through both paths
for query in test_queries:
    # Control: Standard VERIFY
    control_result = await mrf.refine(
        query,
        strategy=RefinementStrategy.VERIFY,
        enable_cove=False  # Disable CoVe
    )

    # Treatment: CoVe verification
    treatment_result = await mrf.refine(
        query,
        strategy=RefinementStrategy.VERIFY,
        enable_cove=True  # Enable CoVe
    )

    # Log both results
    dashboard.log_ab_test_result(
        test_name="cove_vs_verify",
        user_id=f"user_{idx}",
        variant="control",
        quality_score=control_result['quality_score'],
        execution_time_ms=control_result['execution_time']
    )

    dashboard.log_ab_test_result(
        test_name="cove_vs_verify",
        user_id=f"user_{idx}",
        variant="treatment",
        quality_score=treatment_result['quality_score'],
        execution_time_ms=treatment_result['execution_time']
    )

# Analyze results (after 30+ samples per group)
results = dashboard.get_ab_test_results("cove_vs_verify")

if results['is_significant'] and results['treatment_better']:
    print(f"✅ CoVe is better!")
    print(f"   Quality improvement: {results['statistics']['quality_improvement']:.1%}")
    print(f"   Statistical significance: {results['p_value']:.4f}")
else:
    print(f"❌ No significant improvement")
```

## Best Practices

### When to Use CoVe

**✅ Use CoVe verification when**:
- Response confidence is low (<75%)
- Query is factual/technical (needs verification)
- Response contains multiple claims (>2)
- Quality is critical (safety-sensitive domains)
- User explicitly requests verification

**❌ Avoid CoVe when**:
- Response confidence is already high (>90%)
- Query is subjective/opinion-based
- Latency is critical (<200ms budget)
- Claims are simple (single assertion)

### Optimal Thresholds

**By Domain**:
- Science/Medicine: confidence_threshold=0.85 (high bar)
- Technology: confidence_threshold=0.75 (medium bar)
- Creative: confidence_threshold=0.60 (lower bar)

**By Response Length**:
- Short (< 100 words): max_claims=5
- Medium (100-500 words): max_claims=10
- Long (> 500 words): max_claims=20

### Cost-Quality Tradeoff

| Config | Avg Quality | Avg Time | Cost |
|--------|------------|----------|------|
| **Minimal** | +15% | 50ms | Low |
| **Balanced** | +25% | 150ms | Medium |
| **Thorough** | +35% | 300ms | High |

Choose based on your quality-latency requirements.

### Integration with Other MRF Strategies

CoVe works well alongside other MRF refinement strategies:

```python
# Combine ELEGANCE (clarity) + CoVe (verification)
result = mrf.refine(
    query,
    strategy=RefinementStrategy.ELEGANCE,
    context={"enable_cove": True}  # Add verification
)

# Combine HOFSTADTER (recursive) + CoVe
result = mrf.refine(
    query,
    strategy=RefinementStrategy.HOFSTADTER,
    context={"enable_cove": True}
)
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Claim extraction** | ~20ms | Fast NLP pass |
| **Question generation** | ~50-100ms | LLM dependent |
| **Contradiction detection** | ~30ms | Pattern matching |
| **Total verification time** | ~100-150ms | Per response |
| **Quality improvement** | +25-35% | Typical range |

## Configuration Reference

```python
from HoloLoom.prompting.verification_integration import VerificationConfig

# All configuration options
config = VerificationConfig(
    # When to apply verification
    confidence_threshold=0.75,      # Apply when confidence < 75%
    min_claims=2,                   # Skip if < 2 claims
    max_claims=10,                  # Verify up to 10 claims

    # Verification parameters
    question_count=3,               # Questions per claim
    contradiction_threshold=0.3,    # Flag if conflict > 30%

    # Performance/cost control
    timeout_seconds=5.0,            # Max verification time
    max_parallel_questions=5,       # Parallel LLM calls

    # Output options
    include_explanations=True,      # Explain findings
    return_claim_list=True          # List extracted claims
)

bridge = VerificationMRFBridge(config=config)
```

## Further Reading

- **[HoloLoom/prompting/verification.py](HoloLoom/prompting/verification.py)** - Core CoVe implementation
- **[HoloLoom/prompting/unified_mrf.py](HoloLoom/prompting/unified_mrf.py)** - MRF engine
- **[demos/demo_cove_integration.py](demos/demo_cove_integration.py)** - Complete example
