#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Runners
================

Orchestrators that wire together core + patterns without coupling.

Runners provide the "agent loop" while keeping core modules clean:
- Core (Shuttle, Warp, Yarn) stays focused on single weaves
- Patterns handle continual learning, logging, adaptation
- Runners orchestrate without modifying core logic

Available Runners:
- WeaveRunner: Standard runner for single-threaded agent loops

Usage:
```python
from hololoom.runners import WeaveRunner
from hololoom.patterns.nested_learning import NestedLearningPattern

runner = WeaveRunner(shuttle=shuttle, patterns=[pattern])
result = runner.run_query("my query", warp_results)
```
"""

from .weave_runner import WeaveRunner, SupportsIntersect


__all__ = ['WeaveRunner', 'SupportsIntersect']
