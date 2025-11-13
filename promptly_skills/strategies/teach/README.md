# Teach Strategy - Few-shot Edge Case Learning

**Category**: Meta-Prompting
**Purpose**: Teach through concrete examples (typical, edge, error cases)
**Quality Gain**: +50% clarity through examples

## Overview

Teaches concepts through 3 types of examples: typical cases (happy path), edge cases (boundary conditions), and error cases (failure modes). Learning through concrete demonstration.

## Usage

```python
from promptly_skills.strategies.teach import TeachStrategy

strategy = TeachStrategy()
result = await strategy.enhance(StrategyContext(query="show regex examples"))
```

## Auto-Detection

- **High (0.85)**: "show examples", "demonstrate", "edge cases"
- **Medium (0.55)**: "examples", "instance", "scenario"

## License

MIT
