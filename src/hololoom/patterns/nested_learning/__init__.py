#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested Learning Pattern
========================

Multi-timescale learning pattern for HoloLoom.

Quick Start:
```python
from hololoom.patterns.nested_learning import (
    NestedLearningPattern,
    NoOpBanditModule,
    NoOpGraphModule,
    NoOpPolicyModule
)

# Create pattern with no-op stubs
pattern = NestedLearningPattern(
    bandit_module=NoOpBanditModule(),
    graph_module=NoOpGraphModule(),
    policy_module=NoOpPolicyModule()
)

# Use with WeaveRunner
from hololoom.runners import WeaveRunner

runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])
result = runner.run_query("my query", warp_results)
```

Components:
- NestedLearningPattern: Main pattern implementation
- Module protocols: SupportsBanditModule, SupportsGraphModule, SupportsPolicyModule
- No-op stubs: NoOpBanditModule, NoOpGraphModule, NoOpPolicyModule

See docs/patterns/nested_learning.md for full documentation.
"""

from .pattern import (
    NestedLearningPattern,
    SupportsBanditModule,
    SupportsGraphModule,
    SupportsPolicyModule,
)

from .stubs import (
    NoOpBanditModule,
    NoOpGraphModule,
    NoOpPolicyModule,
)


__all__ = [
    # Main pattern
    'NestedLearningPattern',

    # Module protocols
    'SupportsBanditModule',
    'SupportsGraphModule',
    'SupportsPolicyModule',

    # No-op stubs (for scaffolding)
    'NoOpBanditModule',
    'NoOpGraphModule',
    'NoOpPolicyModule',
]


__version__ = "1.0.0"
