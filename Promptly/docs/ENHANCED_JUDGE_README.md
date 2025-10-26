# Enhanced LLM-as-Judge System

## Overview

The Enhanced LLM-as-Judge incorporates best practices from cutting-edge research:

- **Constitutional AI** (Anthropic) - Multi-principle evaluation
- **G-Eval** (Microsoft Research) - Form-filling evaluation paradigm
- **PandaLM** - Multi-aspect evaluation
- **Chain-of-Thought** judging - Reasoning before scoring
- **Pairwise comparison** - Relative quality assessment

## Key Features

### 1. Chain-of-Thought Reasoning (Best Practice)

Instead of direct scoring, the judge reasons step-by-step:

```
1. ANALYSIS: What are the strengths and weaknesses?
2. EVIDENCE: Cite specific examples from the response
3. COMPARISON: How does it compare to an ideal response?
4. SCORE: Rate 0-10 based on your analysis
```

**Why it works:** Reduces bias, increases consistency, provides interpretable reasoning.

### 2. Constitutional AI Evaluation

Evaluate responses against ethical principles:

```python
principles = [
    "The response should be helpful and harmless",
    "The response should respect human autonomy and dignity",
    "The response should be truthful and avoid deception",
    "The response should not exhibit harmful biases",
]
```

**Why it works:** Ensures alignment with human values, safety-first approach.

### 3. G-Eval Methodology

Form-filling paradigm for systematic evaluation:

```
### 1. Key Points Analysis
POINTS: ___________

### 2. Criterion Assessment
ASSESSMENT: ___________

### 3. Strengths
STRENGTHS: ___________

### 4. Weaknesses
WEAKNESSES: ___________

### 5. Score Justification
JUSTIFICATION: ___________
```

**Why it works:** Structured reasoning, reduces hallucination, consistent format.

### 4. Pairwise Comparison

Compare two outputs head-to-head:

```python
winner, confidence, reasoning = judge.compare_pairwise(
    task=task,
    output_a=output_a,
    output_b=output_b,
    criterion=JudgeCriteria.HELPFULNESS
)
```

**Why it works:** Easier for LLMs than absolute scoring, reduces calibration issues.

### 5. Multi-Sample Consistency

Run evaluation multiple times and average:

```python
config = JudgeConfig(
    criteria=[JudgeCriteria.QUALITY],
    num_samples=3  # Average 3 evaluations
)
```

**Why it works:** Reduces variance, improves reliability, detects inconsistencies.

## Evaluation Criteria

### Standard Criteria

1. **QUALITY** - Overall response quality
2. **RELEVANCE** - How well it addresses the task
3. **COHERENCE** - Logical flow and clarity
4. **ACCURACY** - Factual correctness
5. **HELPFULNESS** - How well it solves the problem
6. **SAFETY** - Absence of harmful content
7. **CREATIVITY** - Originality and innovation
8. **CONCISENESS** - Appropriate brevity
9. **COMPLETENESS** - Thoroughness

### Enhanced Criteria

10. **TRUTHFULNESS** - Factual accuracy with hallucination detection
11. **HARMLESSNESS** - Safety through constitutional principles
12. **CUSTOM** - Define your own criteria

## Judging Methods

### 1. SINGLE_SCORE
Direct scoring with rubrics.

**Use when:** Fast evaluation needed, simple tasks.

### 2. CHAIN_OF_THOUGHT (Recommended)
Reasoning before scoring.

**Use when:** Complex evaluation, need explainability.

### 3. PAIRWISE_COMPARISON
Compare two outputs directly.

**Use when:** Relative quality matters more than absolute score.

### 4. REFERENCE_BASED
Compare against gold standard.

**Use when:** You have ideal outputs available.

### 5. CONSTITUTIONAL
Multi-principle ethical evaluation.

**Use when:** Safety and ethics are critical.

### 6. GEVAL
Microsoft's form-filling methodology.

**Use when:** Need structured, systematic evaluation.

## Usage Examples

### Basic Chain-of-Thought Evaluation

```python
from llm_judge_enhanced import EnhancedLLMJudge, JudgeConfig, JudgeCriteria, JudgingMethod

# Setup
judge = EnhancedLLMJudge(executor)

config = JudgeConfig(
    criteria=[JudgeCriteria.HELPFULNESS, JudgeCriteria.COHERENCE],
    method=JudgingMethod.CHAIN_OF_THOUGHT,
    use_cot=True
)

# Evaluate
result = judge.evaluate(task, output, config)

print(f"Overall Score: {result.overall_score:.2f}")
for score in result.scores:
    print(f"{score.criterion}: {score.score:.2f}")
    print(f"Reasoning: {score.reasoning}")
```

### Constitutional AI Evaluation

```python
config = JudgeConfig(
    method=JudgingMethod.CONSTITUTIONAL,
    constitutional_principles=[
        "The response should be age-appropriate",
        "The response should be truthful",
        "The response should be helpful"
    ]
)

result = judge.evaluate(task, output, config)
print(result.summary)
```

### Pairwise Comparison

```python
winner, confidence, reasoning = judge.compare_pairwise(
    task="Explain quantum physics",
    output_a=response_a,
    output_b=response_b,
    criterion=JudgeCriteria.CLARITY
)

print(f"Winner: {winner} (confidence: {confidence:.2f})")
print(f"Why: {reasoning}")
```

### Multi-Sample Consistency

```python
config = JudgeConfig(
    criteria=[JudgeCriteria.QUALITY],
    method=JudgingMethod.CHAIN_OF_THOUGHT,
    num_samples=5  # Run 5 times and average
)

result = judge.evaluate(task, output, config)
# Scores are averaged across samples for consistency
```

### G-Eval Method

```python
config = JudgeConfig(
    criteria=[JudgeCriteria.HELPFULNESS],
    method=JudgingMethod.GEVAL
)

result = judge.evaluate(task, output, config)
# Uses form-filling paradigm for structured evaluation
```

## Best Practices from Research

### 1. Use Chain-of-Thought
**Finding:** LLMs produce more consistent and accurate evaluations when they reason before scoring.

**Source:** Wei et al. (2022) "Chain-of-Thought Prompting"

**Implementation:** All our methods use CoT by default.

### 2. Multiple Samples for Consistency
**Finding:** Single LLM evaluations can be noisy; averaging multiple samples improves reliability.

**Source:** Liu et al. (2023) "G-Eval"

**Implementation:** `num_samples` parameter for averaging.

### 3. Constitutional Principles
**Finding:** Explicit principles improve safety and alignment.

**Source:** Anthropic (2022) "Constitutional AI"

**Implementation:** `CONSTITUTIONAL` judging method.

### 4. Pairwise > Absolute Scoring
**Finding:** LLMs are better at relative comparisons than absolute scoring.

**Source:** Dubois et al. (2023) "AlpacaEval"

**Implementation:** `compare_pairwise()` method.

### 5. Form-Filling Reduces Hallucination
**Finding:** Structured formats reduce judge hallucination and improve consistency.

**Source:** Liu et al. (2023) "G-Eval"

**Implementation:** `GEVAL` method with form templates.

### 6. Lower Temperature for Judging
**Finding:** Lower temperature (0.2-0.4) produces more consistent judgments.

**Source:** Multiple studies on LLM evaluation

**Implementation:** Default `temperature=0.3` in JudgeConfig.

### 7. Explicit Rubrics
**Finding:** Detailed rubrics improve inter-annotator agreement.

**Source:** Standard ML evaluation practices

**Implementation:** Enhanced rubrics with CoT prompts.

### 8. Evidence-Based Scoring
**Finding:** Requiring evidence citations improves accuracy.

**Source:** Multiple evaluation studies

**Implementation:** All rubrics ask for specific evidence.

## Architecture

### Key Classes

**EnhancedLLMJudge** - Main judge class
- `evaluate()` - Main evaluation method
- `compare_pairwise()` - Pairwise comparison
- `_build_cot_judge_prompt()` - CoT prompt builder
- `_build_constitutional_prompt()` - Constitutional prompt
- `_build_geval_prompt()` - G-Eval prompt
- `_parse_cot_response()` - Response parser

**JudgeConfig** - Configuration
- `criteria` - List of criteria to evaluate
- `method` - Judging method (CoT, Constitutional, etc.)
- `num_samples` - Number of samples for consistency
- `temperature` - LLM temperature (default 0.3)
- `constitutional_principles` - Custom principles

**JudgeScore** - Individual criterion score
- `criterion` - What was evaluated
- `score` - 0.0-1.0 normalized score
- `reasoning` - Why this score
- `chain_of_thought` - Step-by-step reasoning
- `confidence` - Judge confidence 0.0-1.0
- `evidence` - Supporting quotes/examples

**JudgeResult** - Complete evaluation
- `scores` - List of JudgeScore objects
- `overall_score` - Averaged score
- `summary` - Human-readable summary
- `method_used` - Which method was used

## Research References

1. **Constitutional AI** - Anthropic (2022)
   - https://arxiv.org/abs/2212.08073
   - Multi-principle evaluation, safety-first

2. **G-Eval** - Microsoft Research (2023)
   - https://arxiv.org/abs/2303.16634
   - Form-filling paradigm, structured evaluation

3. **Chain-of-Thought Prompting** - Wei et al. (2022)
   - https://arxiv.org/abs/2201.11903
   - Reasoning before answering

4. **AlpacaEval** - Dubois et al. (2023)
   - https://arxiv.org/abs/2305.14387
   - Pairwise comparison methods

5. **PandaLM** - Wang et al. (2023)
   - Multi-aspect evaluation

## Performance

Running on **llama3.2:3b** (2GB model):

- **Single evaluation:** ~5-10 seconds
- **Multi-sample (3x):** ~15-30 seconds
- **Pairwise comparison:** ~10-15 seconds
- **Constitutional (6 principles):** ~15-20 seconds

All running locally with no API costs!

## Limitations & Future Work

### Current Limitations

1. **Model Size:** 3B model has limited reasoning depth
2. **Calibration:** Scores may not be perfectly calibrated
3. **Language:** Currently English-only
4. **Context:** Limited context window

### Future Enhancements

1. **Auto-calibration:** Learn score distributions
2. **Meta-evaluation:** Judge the judge
3. **Ensemble judging:** Multiple models
4. **Active learning:** Improve from feedback
5. **Specialized judges:** Domain-specific rubrics

## Integration with Promptly

The Enhanced Judge integrates with:

- **Recursive Loops:** Quality scoring in refinement loops
- **A/B Testing:** Scientific prompt comparison
- **Execution Engine:** Automatic output evaluation
- **Skills System:** Evaluate skill outputs
- **MCP Server:** Available via Claude Desktop

## Summary

The Enhanced LLM-as-Judge provides **research-backed evaluation** using:

✅ Chain-of-thought reasoning
✅ Constitutional AI principles
✅ G-Eval methodology
✅ Pairwise comparison
✅ Multi-sample consistency
✅ Evidence-based scoring
✅ Explicit rubrics

All running on a **2GB local model** with no API costs!

**Result:** Robust, interpretable, research-backed evaluation system for AI outputs.
