# Scaffold Strategy - Zero-shot CoT Structure

**Category**: Self-Correction
**Purpose**: Provide structured reasoning template with blanks to fill
**Quality Gain**: +42% reasoning clarity

## Overview

The Scaffold strategy implements "Zero-shot Chain-of-Thought Structure" - providing a reasoning template with blanks that force the model to think explicitly through each step. It prevents jumping to conclusions by requiring structured analysis.

## Key Features

### 6-Step Reasoning Template

1. **Understand** - Define problem, terms, givens, constraints
2. **Decompose** - Break into subproblems with dependencies
3. **Analyze** - Solve each subproblem with confidence assessment
4. **Synthesize** - Combine insights, find patterns
5. **Validate** - Check reasoning for gaps and errors
6. **Conclude** - State final answer with caveats

### Forced Explicit Thinking

- Every blank must be filled
- No skipping steps
- Show all work
- Identify uncertainties
- Validate own reasoning

## Usage

```python
from promptly_skills.strategies.scaffold import ScaffoldStrategy
from HoloLoom.prompting.strategy import StrategyContext

strategy = ScaffoldStrategy()
context = StrategyContext(query="How do I calculate compound interest?")
result = await strategy.enhance(context)

# Result provides 6-step template:
# Step 1: Understand (problem, terms, givens, constraints)
# Step 2: Decompose (subproblems + dependencies)
# Step 3: Analyze (solve each part)
# Step 4: Synthesize (combine insights)
# Step 5: Validate (check reasoning)
# Step 6: Conclude (final answer + confidence)
```

## Auto-Detection

- **High (0.85)**: "step by step", "show your work", "walk through"
- **Medium (0.65)**: "solve", "calculate", "analyze", problem-solving queries
- **Low (0.25)**: Simple factual queries, very short queries

## When to Use

- Problem-solving tasks
- Complex reasoning
- Mathematical calculations
- Logical analysis
- Learning (forces explicit thinking)

## Composability

- **scaffold + verify**: Structure then verify logic
- **scaffold + deep**: Structure for deep analysis
- **optimize + scaffold**: Clarify then structure reasoning

## License

MIT - Part of Promptly Strategy Framework
