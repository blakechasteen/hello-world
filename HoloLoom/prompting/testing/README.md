# Prompt Testing Framework

**Status**: ✅ Production Ready (November 2025)
**Updated**: December 2025 - LLMJudge Integration
**Location**: `HoloLoom/prompting/testing/`
**Performance**: LLM-based evaluation with heuristic fallback

Comprehensive testing framework for systematic prompt validation and quality assurance across all HoloLoom systems.

## Overview

The Prompt Testing Framework provides three types of tests to ensure prompt quality, robustness, and prevent regressions:

1. **Golden Dataset Tests** - Test against known good outputs
2. **Mutation Tests** - Test prompt robustness to variations
3. **Regression Tests** - Detect quality degradation over time

### Key Features (December 2025)

- **LLM-powered evaluation** using Ollama/Anthropic/OpenAI (via LLMJudge)
- **Multi-criteria scoring** (quality, relevance, coherence, completeness)
- **Automatic fallback** to heuristic scoring if LLM unavailable
- **Parallel execution** for fast test runs
- **Comprehensive metrics** (pass rates, latency, quality scores)
- **Prometheus export** for monitoring integration

## Quick Start

### Basic Usage

```python
from HoloLoom.prompting.testing import create_test_suite, PromptTestConfig

# Create test suite with defaults (LLM-based evaluation)
suite = create_test_suite()

# Run all tests
report = await suite.run_all_tests()

print(f"Pass rate: {report.overall_pass_rate:.1%}")
print(f"Avg quality: {report.avg_quality_score:.2f}")
print(f"Avg latency: {report.avg_latency_ms:.2f}ms")
```

### CLI Usage

```bash
# Run all tests with LLM evaluation (default)
python -m HoloLoom.prompting.testing.test_suite --test-type all

# Run specific test types
python -m HoloLoom.prompting.testing.test_suite --test-type golden
python -m HoloLoom.prompting.testing.test_suite --test-type mutation
python -m HoloLoom.prompting.testing.test_suite --test-type regression

# Configure LLM provider
python -m HoloLoom.prompting.testing.test_suite \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-20241022

# Use Ollama (local, fast)
python -m HoloLoom.prompting.testing.test_suite \
  --llm-provider ollama \
  --llm-model llama3.2:3b

# Disable LLM evaluation (use heuristics)
python -m HoloLoom.prompting.testing.test_suite --no-llm-judge

# Save report to file
python -m HoloLoom.prompting.testing.test_suite \
  --output results/test_report.json
```

## LLMJudge Integration (December 2025)

The framework uses **LLMJudge** for LLM-powered quality evaluation:

### Evaluation Criteria

LLMJudge evaluates responses across 4 criteria (configurable):

1. **Quality** - Overall response quality (grammar, structure, completeness)
2. **Relevance** - How well response addresses the prompt
3. **Coherence** - Logical flow and consistency
4. **Completeness** - Whether response fully answers the question

Each criterion is scored 0.0-1.0, and an overall score is computed.

### Configuration

```python
from HoloLoom.prompting.testing import PromptTestConfig

config = PromptTestConfig(
    # LLM Judge settings
    use_llm_judge=True,                    # Enable LLM evaluation
    llm_provider="ollama",                 # ollama/anthropic/openai
    llm_model="llama3.2:3b",              # Model for evaluation
    llm_criteria=["quality", "relevance", "coherence", "completeness"],
    llm_temperature=0.3,                   # Low temperature for consistency
    llm_timeout_seconds=30.0,
    fallback_to_heuristic=True,            # Fall back if LLM fails

    # Test settings
    quality_threshold=0.8,                 # Minimum quality to pass
    regression_threshold=0.05,             # Max quality drop allowed
    max_mutations=10,
    parallel_execution=True,
    timeout_ms=5000,
)

suite = create_test_suite(config)
```

### Provider Options

| Provider | Models | Latency | Cost |
|----------|--------|---------|------|
| **ollama** | llama3.2:3b, llama3.1:8b | ~200-500ms | Free (local) |
| **anthropic** | claude-3-5-sonnet-20241022 | ~500-1500ms | API cost |
| **openai** | gpt-4, gpt-4-turbo | ~500-2000ms | API cost |

**Recommendation**: Use `ollama` with `llama3.2:3b` for fast, free local evaluation.

### Heuristic Fallback

If LLM is unavailable or `use_llm_judge=False`, the framework falls back to heuristic scoring:

- **Completeness** - Response length validation
- **Format** - Basic structure checks
- **Relevance** - Word overlap with expected output (if available)

Heuristics are faster (~1ms) but less accurate than LLM evaluation.

## Test Types

### 1. Golden Dataset Tests

Test prompts against known good outputs:

```python
# Add golden test cases
from HoloLoom.prompting.testing import GoldenDatasetManager, PromptTestCase, TestType

dataset = GoldenDatasetManager()
dataset.add_golden_pair(
    prompt="What is Thompson Sampling?",
    expected_output="Thompson Sampling is a Bayesian approach..."
)

# Or create custom test case
test_case = PromptTestCase(
    name="thompson_sampling_definition",
    prompt_template="What is Thompson Sampling?",
    test_inputs=[{}],
    expected_qualities={"quality": 0.9, "relevance": 0.95},
    golden_outputs=["Thompson Sampling is..."],
    test_type=TestType.GOLDEN,
)

# Run golden tests
results = await suite.run_golden_tests()
```

### 2. Mutation Tests

Test prompt robustness to variations:

```python
# Test mutations of a prompt
from HoloLoom.prompting.testing import MutationTester

tester = MutationTester(config)

# Generate mutations (synonym substitution, paraphrasing, etc.)
mutations = tester.mutate("What is Thompson Sampling?")
# Returns: [
#   "What is Thompson's Sampling?",
#   "Define Thompson Sampling",
#   "Explain Thompson Sampling",
#   ...
# ]

# Test all mutations
results = await suite.run_mutation_tests(["What is Thompson Sampling?"])

# Check robustness
avg_quality = sum(r.quality_scores.get('overall', 0) for r in results) / len(results)
print(f"Mutation robustness: {avg_quality:.2f}")
```

### 3. Regression Tests

Detect quality degradation over time:

```python
# Run regression detection
from HoloLoom.prompting.testing import RegressionDetector

detector = RegressionDetector(config)

# Compare current results to baseline
current_results = await suite.run_golden_tests()
regressions = await detector.detect_regressions(
    current_results,
    baseline_results  # From previous test run
)

if regressions:
    print(f"⚠️  {len(regressions)} regressions detected!")
    for regression in regressions:
        print(f"  - {regression}")
```

## Test Results

### PromptTestResult

Each test returns a `PromptTestResult`:

```python
@dataclass
class PromptTestResult:
    test_case: PromptTestCase        # Original test case
    passed: bool                      # Did it pass?
    quality_scores: Dict[str, float] # Scores by criterion
    latency_ms: float                # Execution time
    token_count: int                 # Tokens used
    status: TestStatus               # PASSED/FAILED/ERROR
    error_message: Optional[str]     # Error details if failed
    timestamp: datetime              # When executed
```

### PromptTestReport

Aggregated report from test suite:

```python
@dataclass
class PromptTestReport:
    test_results: List[PromptTestResult]
    total_tests: int
    passed_count: int
    failed_count: int
    skipped_count: int
    overall_pass_rate: float         # 0.0-1.0
    avg_latency_ms: float
    avg_quality_score: float
    regressions: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]
```

## Metrics Collection

The framework collects comprehensive metrics:

```python
from HoloLoom.prompting.testing import MetricsCollector, MetricType

collector = suite.metrics_collector

# Get all metrics
metrics = collector.get_metrics()

# Export to Prometheus
prometheus_output = collector.export_prometheus()
# Output:
# prompt_test_quality_score{test="test_name"} 0.92
# prompt_test_latency_ms{test="test_name"} 145.3
# prompt_test_pass_rate 0.85
```

## Configuration Options

Complete configuration reference:

```python
@dataclass
class PromptTestConfig:
    # Quality thresholds
    quality_threshold: float = 0.8           # Min quality to pass (0.0-1.0)
    regression_threshold: float = 0.05       # Max quality drop (0.0-1.0)

    # Test execution
    max_mutations: int = 10                  # Mutations per prompt
    enable_golden_tests: bool = True
    enable_mutation_tests: bool = True
    enable_regression_tests: bool = True
    parallel_execution: bool = True
    timeout_ms: int = 5000

    # LLM Judge (December 2025)
    use_llm_judge: bool = True               # Use LLM evaluation
    llm_provider: str = "ollama"             # ollama/anthropic/openai
    llm_model: str = "llama3.2:3b"          # Model name
    llm_criteria: List[str] = [              # Evaluation criteria
        "quality", "relevance", "coherence", "completeness"
    ]
    llm_temperature: float = 0.3             # Lower = more consistent
    llm_timeout_seconds: float = 30.0
    fallback_to_heuristic: bool = True       # Fallback if LLM fails
```

## Integration with HoloLoom

Test prompts across all HoloLoom systems:

```python
# Test RAG prompts
from HoloLoom.rag import SimpleRAG

async def test_rag_prompt():
    rag = SimpleRAG()

    test_case = PromptTestCase(
        name="rag_query_thompson_sampling",
        prompt_template="What is Thompson Sampling?",
        test_inputs=[{}],
        expected_qualities={"quality": 0.85},
    )

    result = await rag.query(test_case.prompt_template)

    # Evaluate with LLMJudge
    quality = await suite.llm_judge.evaluate(
        task="Evaluate RAG response",
        output=result.response
    )

    return quality.overall_score >= 0.85

# Test Agentic prompts
from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode

async def test_agentic_prompt():
    orchestrator = await create_agentic_orchestrator(config, shards)

    result = await orchestrator.reason(
        Query(text="Compare Thompson Sampling vs UCB"),
        mode=ReasoningMode.RESEARCH
    )

    # Add to test suite
    test_case = PromptTestCase(
        name="agentic_research_comparison",
        prompt_template="Compare Thompson Sampling vs UCB",
        test_inputs=[{}],
        expected_qualities={"quality": 0.8, "completeness": 0.9},
    )

    test_result = await suite.run_single_test(test_case)
    return test_result.passed
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **LLM evaluation (Ollama 3B)** | ~200-500ms | Local, fast |
| **LLM evaluation (Claude)** | ~500-1500ms | API call, high quality |
| **Heuristic evaluation** | <1ms | Fast but less accurate |
| **Mutation generation** | <10ms | Per prompt |
| **Parallel test execution** | 2-5x speedup | Depends on CPU cores |

## Example Output

```
============================================================
PROMPT TEST SUITE REPORT
============================================================
Total Tests: 25
Passed: 23 (92.0%)
Failed: 2
Skipped: 0
Avg Latency: 187.32ms
Avg Quality: 0.87
Regressions: 0
============================================================

Test Details:
  ✓ thompson_sampling_definition: 0.92 (187ms)
  ✓ bayesian_exploration: 0.89 (201ms)
  ✗ edge_case_handling: 0.74 (FAILED - below threshold 0.8)
  ✓ multi_armed_bandit: 0.91 (165ms)
  ...
```

## Key Files

- **test_suite.py** (575 lines) - Main orchestrator
- **protocol.py** (291 lines) - Test protocols and data types
- **golden_dataset.py** - Golden dataset management
- **mutation_testing.py** - Prompt mutation generation
- **regression_testing.py** - Regression detection
- **metrics_collector.py** - Metrics collection and export

## Running Tests

```bash
# Complete test suite
python -m HoloLoom.prompting.testing.test_suite

# Expected output:
# - Golden tests: 10/10 passing
# - Mutation tests: 50/50 passing
# - Regression tests: 5/5 passing
# - Overall: 65/65 passing (100%)
```

## When to Use

**✅ Use Prompt Testing when you need**:
- Systematic quality validation of prompts
- Regression prevention before deployment
- A/B testing of prompt variants
- Continuous monitoring of prompt quality
- Compliance/audit trail for prompts

**🟡 Use LLM evaluation when**:
- Quality matters more than speed
- Local Ollama instance available
- Nuanced quality assessment needed
- Human-like evaluation preferred

**🟡 Use heuristic evaluation when**:
- Speed is critical (<1ms)
- LLM unavailable or too expensive
- Simple quality checks sufficient
- Batch testing thousands of prompts

## Future Enhancements

Planned for Phase 7+:
1. **Multi-model evaluation** - Consensus scoring across multiple LLMs
2. **Custom criteria** - User-defined evaluation dimensions
3. **Batch optimization** - Parallel LLM calls for faster evaluation
4. **Quality trends** - Track quality over time with dashboards
5. **Automated fixes** - Suggest prompt improvements based on failures

See [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md) for complete roadmap.
