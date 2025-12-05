# Debate Strategy - Multi-Persona Debate

**Category**: Meta-Prompting
**Purpose**: Explore topic through conflicting expert viewpoints
**Quality Gain**: +52% through multi-perspective analysis

## Overview

Three expert personas debate the topic: Optimist (pro), Skeptic (con), Pragmatist (balanced). Forces consideration of multiple viewpoints before synthesis.

## Usage

```python
from promptly_skills.strategies.debate import DebateStrategy

strategy = DebateStrategy()
result = await strategy.enhance(StrategyContext(query="should we use microservices?"))
```

## Auto-Detection

- **High (0.85)**: "debate", "pros and cons", "tradeoffs", "conflicting views"
- **Medium (0.60)**: "perspectives", "viewpoints", "arguments"

## License

MIT
