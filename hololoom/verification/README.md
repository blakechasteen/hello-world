# Chain of Verification (CoVe) System

**Status**: Production Ready (2025-12-05)
**Location**: `hololoom/verification/`
**Performance**: <50ms heuristic mode, ~500ms with LLM
**Test Coverage**: 46/46 tests passing (100%)

## Overview

The Chain of Verification system implements Meta AI's 4-step CoVe methodology for reducing hallucinations through systematic verification. Unlike simple confidence scoring, CoVe independently verifies claims to prevent self-confirmation bias.

**Research Foundation**:
- [TACL 2024 Survey](https://aclanthology.org/2024.tacl-1.78/): Self-correction fails without external feedback
- [ACL Findings 2024](https://aclanthology.org/2024.findings-acl.212.pdf): CoVe 4-step methodology significantly reduces hallucinations
- [MiniCheck (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.499/): Token-level uncertainty quantification

## 4-Step CoVe Process

```
┌─────────────────────────────────────────────────────────────┐
│                  Chain of Verification                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Baseline Response                                   │
│  ┌─────────────────────────────────────┐                     │
│  │ Generate initial answer to query    │                     │
│  └─────────────────────────────────────┘                     │
│                      ↓                                        │
│  Step 2: Verification Planning                               │
│  ┌─────────────────────────────────────┐                     │
│  │ Extract verifiable claims           │                     │
│  │ Generate targeted fact-check Qs     │                     │
│  └─────────────────────────────────────┘                     │
│                      ↓                                        │
│  Step 3: Independent Verification  ⚠️ CRITICAL               │
│  ┌─────────────────────────────────────┐                     │
│  │ Answer questions WITHOUT seeing     │                     │
│  │ original response (prevents bias)   │                     │
│  └─────────────────────────────────────┘                     │
│                      ↓                                        │
│  Step 4: Refined Response                                    │
│  ┌─────────────────────────────────────┐                     │
│  │ Synthesize verified answer          │                     │
│  │ Flag contradictions for review      │                     │
│  └─────────────────────────────────────┘                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Step 3 (Independent Verification) answers questions in a **separate LLM context** that cannot see the original response. This prevents self-confirmation bias where the model simply agrees with its own output.

## Quick Start

### Basic Verification

```python
from hololoom.verification import verify_response

result = await verify_response(
    query="What is Thompson Sampling?",
    response="Thompson Sampling was introduced in 1933...",
    confidence=0.7
)

print(result.summary())
# [OK] Verification VERIFIED
#    Claims extracted: 3
#    Questions asked: 9
#    Contradictions: 0
#    Confidence: 0.70 -> 0.85
```

### With Configuration

```python
from hololoom.verification import (
    create_verification_chain,
    VerificationConfig,
    DegradationLevel
)

# Configure verification
config = VerificationConfig(
    confidence_threshold=0.85,
    max_claims=10,
    max_iterations=3,
    preferred_level=DegradationLevel.FULL
)

chain = create_verification_chain(config=config)

result = await chain.verify(
    query="Explain backpropagation",
    response="Backpropagation is an algorithm...",
    confidence=0.6,
    max_iterations=3
)

if result.contradictions_found:
    print("Contradictions detected:")
    for c in result.contradictions_found:
        print(f"  - {c.explanation}")
```

### Component-Level Access

```python
from hololoom.verification import (
    create_claim_extractor,
    create_verification_planner,
    create_independent_verifier,
    create_contradiction_detector,
    VerificationConfig
)

config = VerificationConfig()

# Extract claims
extractor = create_claim_extractor(config)
claims = await extractor.extract(response_text, context={})

# Generate verification questions
planner = create_verification_planner(config)
for claim in claims:
    questions = await planner.plan(claim, context={})

# Answer questions independently
verifier = create_independent_verifier(config)
answers = await verifier.verify(questions, context={})  # No original response!

# Detect contradictions
detector = create_contradiction_detector(config)
contradictions = await detector.detect(claim, answers)
```

## Claim Types

The system extracts and classifies claims:

| Type | Description | Example |
|------|-------------|---------|
| **FACTUAL** | Verifiable facts | "Python was created in 1991" |
| **PROCEDURAL** | Steps or processes | "First initialize, then train" |
| **CAUSAL** | Cause-effect relationships | "Overfitting leads to poor generalization" |
| **DEFINITIONAL** | Definitions | "A neural network is a model..." |
| **COMPARATIVE** | Comparisons | "GPT-4 is larger than GPT-3" |
| **QUANTITATIVE** | Numerical claims | "Training took 3 hours" |
| **TEMPORAL** | Time-based claims | "It was released in 2023" |

## Graceful Degradation

The system operates at 4 levels, automatically falling back when resources unavailable:

| Level | LLM Available | Capabilities | Latency |
|-------|---------------|--------------|---------|
| **FULL** | Anthropic/OpenAI | Complete 4-step CoVe | ~800ms |
| **PARTIAL** | Ollama (local) | CoVe with local LLM | ~1200ms |
| **HEURISTIC** | None | Regex + similarity matching | ~50ms |
| **DISABLED** | N/A | Skip verification | ~0ms |

```python
from hololoom.verification import VerificationConfig, DegradationLevel

# Force heuristic mode (no LLM required)
config = VerificationConfig(
    preferred_level=DegradationLevel.HEURISTIC,
    fallback_on_error=True  # Auto-fallback if LLM fails
)
```

## Contradiction Detection

Detects 7 types of contradictions:

| Type | Description | Example |
|------|-------------|---------|
| **FACTUAL** | Facts disagree | "Released 1933" vs "Released 1950" |
| **LOGICAL** | Logical inconsistency | "Always true" vs "Sometimes false" |
| **TEMPORAL** | Time contradiction | "Before X" vs "After X" |
| **QUANTITATIVE** | Numbers disagree | "100 users" vs "1000 users" |
| **SEMANTIC** | Meaning conflicts | "Bayesian" vs "Frequentist" |
| **NEGATION** | Direct negation | "Is X" vs "Is not X" |

```python
# Check for contradictions in result
if result.contradictions_found:
    for c in result.contradictions_found:
        print(f"Type: {c.contradiction_type.value}")
        print(f"Severity: {c.severity:.2f}")
        print(f"Explanation: {c.explanation}")
```

## Configuration Options

```python
from hololoom.verification import VerificationConfig

config = VerificationConfig(
    # Thresholds
    confidence_threshold=0.85,      # Below this triggers verification
    contradiction_threshold=0.7,    # Above this marks as contradicted
    importance_threshold=0.3,       # Below this skips verification

    # Limits
    max_claims=10,                  # Max claims to extract
    max_questions_per_claim=3,      # Max questions per claim
    max_iterations=3,               # Max refinement iterations

    # Performance
    parallel_verification=True,     # Verify claims in parallel
    enable_caching=True,            # Cache verification results
    cache_ttl_seconds=3600,         # Cache TTL (1 hour)

    # Degradation
    preferred_level=DegradationLevel.FULL,
    fallback_on_error=True,         # Fall back to lower level on error

    # LLM settings
    llm_provider="anthropic",       # "anthropic", "openai", "ollama"
    llm_model="claude-3-5-sonnet",
    llm_temperature=0.1,            # Low temp for factual verification

    # Timeouts
    timeout_ms=5000.0,              # Total verification timeout
    claim_timeout_ms=1000.0         # Per-claim timeout
)
```

### Configuration Presets

```python
from hololoom.verification import (
    create_default_config,  # Balanced defaults
    create_fast_config,     # Quick heuristic verification
    create_thorough_config  # Maximum accuracy
)

# Fast mode for latency-critical applications
fast = create_fast_config()  # 2s timeout, heuristic preferred

# Thorough mode for high-stakes responses
thorough = create_thorough_config()  # 10s timeout, 20 claims, 5 iterations
```

## Iterative Refinement

For low-confidence responses, the system can iterate:

```python
result = await chain.verify(
    query="Complex question",
    response="Initial answer...",
    confidence=0.5,  # Low confidence
    max_iterations=3,
    confidence_threshold=0.85
)

print(f"Iterations: {result.iterations}")
print(f"Confidence: {result.confidence_before:.2f} -> {result.confidence_after:.2f}")
print(f"Refinement applied: {result.refinement_applied}")
```

## Caching

Verification results are cached to avoid redundant work:

```python
# Cache enabled by default
config = VerificationConfig(
    enable_caching=True,
    cache_ttl_seconds=3600  # 1 hour
)

# First verification: computes result
result1 = await chain.verify(query, response, 0.7)

# Second verification: returns cached result
result2 = await chain.verify(query, response, 0.7)  # ~0ms

# Get cache statistics
stats = chain.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## Module Structure

```
hololoom/verification/
├── __init__.py              # Public API exports
├── protocol.py              # Core protocols and data structures
├── chain.py                 # Main CoVe orchestrator
├── claim_extractor.py       # Extract verifiable claims
├── verification_planner.py  # Generate verification questions
├── independent_verifier.py  # Answer questions independently
├── contradiction_detector.py # Detect semantic contradictions
├── result_synthesizer.py    # Synthesize verified response
├── README.md                # This file
└── tests/
    └── test_verification_chain.py  # 46 tests
```

## API Reference

### Main Functions

```python
# Convenience function
verify_response(query, response, confidence, config=None) -> VerificationResult

# Factory functions
create_verification_chain(config=None) -> VerificationChain
create_claim_extractor(config, llm_client=None) -> ClaimExtractor
create_verification_planner(config, llm_client=None) -> VerificationPlanner
create_independent_verifier(config, llm_client=None) -> IndependentVerifier
create_contradiction_detector(config, llm_client=None) -> ContradictionDetector
create_result_synthesizer(config, llm_client=None) -> ResultSynthesizer

# Configuration presets
create_default_config() -> VerificationConfig
create_fast_config() -> VerificationConfig
create_thorough_config() -> VerificationConfig
```

### Data Classes

```python
@dataclass
class VerifiableClaim:
    text: str
    claim_type: ClaimType
    confidence: float
    source_span: Tuple[int, int]
    importance: float
    entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class VerificationQuestion:
    question: str
    claim: VerifiableClaim
    question_type: str
    expected_answer_type: str

@dataclass
class VerificationAnswer:
    question: VerificationQuestion
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str

@dataclass
class Contradiction:
    claim: VerifiableClaim
    evidence: str
    severity: float
    explanation: str
    contradiction_type: ContradictionType

@dataclass
class VerificationResult:
    original_response: str
    verified_response: str
    claims_extracted: List[VerifiableClaim]
    verification_questions: List[VerificationQuestion]
    verification_answers: List[VerificationAnswer]
    contradictions_found: List[Contradiction]
    confidence_before: float
    confidence_after: float
    verification_passed: bool
    refinement_applied: bool
    status: VerificationStatus
    iterations: int
    latency_ms: float
    degradation_level: DegradationLevel
    metadata: Dict[str, Any]
    timestamp: datetime
```

### Enums

```python
class VerificationStatus(Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    NEEDS_HUMAN = "needs_human"
    SKIPPED = "skipped"

class ClaimType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CAUSAL = "causal"
    DEFINITIONAL = "definitional"
    COMPARATIVE = "comparative"
    QUANTITATIVE = "quantitative"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"

class DegradationLevel(Enum):
    FULL = "full"
    PARTIAL = "partial"
    HEURISTIC = "heuristic"
    DISABLED = "disabled"

class ContradictionType(Enum):
    FACTUAL = "factual"
    LOGICAL = "logical"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    SEMANTIC = "semantic"
    NEGATION = "negation"
    UNKNOWN = "unknown"
```

## Running Tests

```bash
# All verification tests
pytest hololoom/verification/tests/ -v

# Result: 46/46 passing (100%)
```

## Running the Demo

```bash
PYTHONPATH=. python demos/demo_chain_of_verification.py
```

Demonstrates:
1. Basic verification chain
2. Contradiction detection
3. Claim extraction
4. Degradation levels
5. Quick verify API
6. Iterative refinement

## Integration with HoloLoom

The verification module integrates with HoloLoom's agentic reasoning:

```python
from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

# VERIFY mode uses CoVe internally
result = await orchestrator.reason(
    Query(text="Explain Thompson Sampling"),
    mode=ReasoningMode.VERIFY  # Uses Chain of Verification
)

print(result.verification.verified)  # True/False
print(result.verification.contradictions)  # Any contradictions found
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Claim extraction (heuristic) | ~5ms | Regex-based |
| Claim extraction (LLM) | ~300ms | With LLM |
| Verification planning | ~10ms | Question generation |
| Independent verification | ~500ms | Per claim (LLM) |
| Contradiction detection | ~20ms | Semantic matching |
| Full CoVe cycle | ~800ms | All 4 steps |
| Cached result | <1ms | Cache hit |

## Best Practices

1. **Use appropriate degradation level**:
   - HEURISTIC for latency-critical paths
   - FULL for high-stakes decisions

2. **Enable caching**: Avoid redundant verification of similar claims

3. **Set reasonable limits**: `max_claims=10` prevents runaway verification

4. **Monitor contradictions**: Log and review contradicted responses

5. **Iterate on low confidence**: Use `max_iterations=3` for complex queries

6. **Independent verification is critical**: Never expose original response during verification

## Troubleshooting

**Q: Verification always returns NEEDS_HUMAN**
A: Check if LLM is available. In heuristic mode, complex claims default to human review.

**Q: Contradictions not detected for obvious errors**
A: Ensure claims are properly extracted. Check `claims_extracted` in result.

**Q: High latency in FULL mode**
A: Consider parallel_verification=True and reasonable timeouts.

**Q: Cache not working**
A: Verify `enable_caching=True` and check `cache_ttl_seconds` hasn't expired.

## References

- [TACL 2024 Survey: Self-Correction in LLMs](https://aclanthology.org/2024.tacl-1.78/)
- [Chain-of-Verification Paper (ACL 2024)](https://aclanthology.org/2024.findings-acl.212.pdf)
- [MiniCheck: Efficient Fact-Checking (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.499/)
