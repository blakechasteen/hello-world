#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Orchestrator Protocols Package
========================================

Part of the Elegance Pass architectural refactoring (December 2025).
Phase 2-3: Protocols for Stage Executors and Component DI

This package provides:
- **Stage Protocols** (Phase 2): StageExecutorProtocol, BaseStageExecutor
- **Component Protocols** (Phase 3): 6 capability protocols for dependency injection
- **Default Implementations** (Phase 3): 6 default implementations for DI

Stage Protocols (Phase 2):
    - StageExecutorProtocol: Interface for all weaving stage executors
    - BaseStageExecutor: ABC with timing/logging helpers

Component Protocols (Phase 3):
    - PatternSelectorProtocol: Pattern card selection
    - ThreadSelectorProtocol: Memory thread selection
    - FeatureExtractorProtocol: Feature extraction
    - ConvergenceProtocol: Decision collapse
    - ToolExecutorProtocol: Tool execution
    - SpacetimeAssemblerProtocol: Output assembly

Default Implementations (Phase 3):
    - DefaultPatternSelector: Wraps LoomCommand
    - DefaultThreadSelector: Wraps YarnGraph
    - DefaultFeatureExtractor: Wraps ResonanceShed
    - DefaultConvergence: Wraps ConvergenceEngine
    - DefaultToolExecutor: Wraps ToolExecutor
    - DefaultSpacetimeAssembler: Wraps fabric assembly

Usage:
    >>> from hololoom.orchestrator.protocols import (
    ...     StageExecutorProtocol,
    ...     BaseStageExecutor,
    ...     PatternSelectorProtocol,
    ...     DefaultPatternSelector,
    ... )
    >>>
    >>> # Check protocol compliance
    >>> isinstance(my_executor, StageExecutorProtocol)
    True
    >>> isinstance(DefaultPatternSelector(loom_cmd), PatternSelectorProtocol)
    True

Backward Compatibility:
    All Phase 2 imports continue to work:
    >>> from hololoom.orchestrator.protocols import StageExecutorProtocol

Author: Claude Code (Elegance Pass - Phase 2-3)
Date: 2025-12-09
"""

# Stage Protocols (Phase 2)
from .stage import (
    StageExecutorProtocol,
    BaseStageExecutor,
)

# Component Protocols (Phase 3)
from .components import (
    PatternSelectorProtocol,
    ThreadSelectorProtocol,
    FeatureExtractorProtocol,
    ConvergenceProtocol,
    ToolExecutorProtocol,
    SpacetimeAssemblerProtocol,
)

# Default Implementations (Phase 3)
from .defaults import (
    DefaultPatternSelector,
    DefaultThreadSelector,
    DefaultFeatureExtractor,
    DefaultConvergence,
    DefaultToolExecutor,
    DefaultSpacetimeAssembler,
)

__all__ = [
    # Stage Protocols (Phase 2)
    'StageExecutorProtocol',
    'BaseStageExecutor',

    # Component Protocols (Phase 3)
    'PatternSelectorProtocol',
    'ThreadSelectorProtocol',
    'FeatureExtractorProtocol',
    'ConvergenceProtocol',
    'ToolExecutorProtocol',
    'SpacetimeAssemblerProtocol',

    # Default Implementations (Phase 3)
    'DefaultPatternSelector',
    'DefaultThreadSelector',
    'DefaultFeatureExtractor',
    'DefaultConvergence',
    'DefaultToolExecutor',
    'DefaultSpacetimeAssembler',
]
