#!/usr/bin/env python3
"""
Stage Executor Classes - Protocol-Based Implementations
========================================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2: Stage Executor Classes (Days 6-10)

This package provides executor classes that wrap Phase 1 pure functions
with protocol-based design for maximum testability and swappability.

Architecture:
    Each executor class:
    - Inherits from BaseStageExecutor
    - Stores dependencies in __init__
    - Delegates to Phase 1 pure functions in execute()
    - Implements StageExecutorProtocol

Executor Classes:
    Steps 0-3:
    - MetaPromptExecutor: Optional query enhancement
    - PatternSelectionExecutor: Loom Command pattern selection
    - ChronoTriggerExecutor: Temporal window creation
    - ThreadSelectionExecutor: Yarn Graph / Shuttle thread selection

    Steps 4-6:
    - ParallelFeatureExecutor: Resonance + Warp + Retrieval (parallel)

    Steps 7-9:
    - ConvergenceExecutor: Decision collapse via Thompson Sampling
    - ToolExecutionExecutor: Safety-gated tool execution
    - SpacetimeExecutor: Result assembly with provenance

Author: Claude Code (Elegance Pass - Phase 2)
Date: 2025-12-09
"""

# Steps 0-3 executors
from .chrono_executor import ChronoTriggerExecutor

# Steps 7-9 executors
from .convergence_executor import ConvergenceExecutor
from .meta_prompt_executor import MetaPromptExecutor

# Steps 4-6 executor
from .parallel_executor import ParallelFeatureExecutor
from .pattern_executor import PatternSelectionExecutor
from .spacetime_executor import SpacetimeExecutor
from .thread_executor import ThreadSelectionExecutor
from .tool_executor import ToolExecutionExecutor

__all__ = [
    # Steps 0-3
    'MetaPromptExecutor',
    'PatternSelectionExecutor',
    'ChronoTriggerExecutor',
    'ThreadSelectionExecutor',
    # Steps 4-6
    'ParallelFeatureExecutor',
    # Steps 7-9
    'ConvergenceExecutor',
    'ToolExecutionExecutor',
    'SpacetimeExecutor',
]
