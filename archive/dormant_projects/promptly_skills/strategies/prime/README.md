# Prime Strategy - Reference Class Priming

**Category**: Meta-Prompting
**Purpose**: Benchmark output against world-class quality exemplars
**Quality Gain**: +48% quality

## Overview

The Prime strategy implements "Reference Class Priming" - setting quality expectations by referencing world-class exemplars. It forces the model to meet standards of domain experts, published works, and production systems.

## Key Features

- **Quality Benchmarks**: World-class experts, academic standards, production quality
- **5 Dimensions**: Clarity, completeness, accuracy, elegance, practicality
- **Quality Checklist**: 20+ specific quality criteria
- **Priming Questions**: 5 questions to check against standards

## Usage

```python
from promptly_skills.strategies.prime import PrimeStrategy

strategy = PrimeStrategy()
result = await strategy.enhance(StrategyContext(query="write production-ready code"))
```

## Auto-Detection

- **High (0.85)**: "best practices", "world class", "expert level"
- **Medium (0.60)**: "high quality", "professional", "top tier"

## License

MIT
