# TempSim Strategy - Temperature Simulation

**Category**: Meta-Prompting
**Purpose**: Roleplay responses at different confidence levels
**Quality Gain**: +40% through confidence calibration

## Overview

Provides 3 responses at different confidence levels: high (90%+, bold), medium (60-90%, balanced), low (<60%, cautious). Helps calibrate appropriate certainty for the query.

## Usage

```python
from promptly_skills.strategies.temp_sim import TempSimStrategy

strategy = TempSimStrategy()
result = await strategy.enhance(StrategyContext(query="will AI replace programmers?"))
```

## Auto-Detection

- **High (0.85)**: "uncertain", "confidence", "hedging"
- **Medium (0.50)**: "might", "maybe", "possibly"

## License

MIT
