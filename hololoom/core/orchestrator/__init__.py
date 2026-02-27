#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Orchestrator Package
==============================

Modular orchestrator components extracted from weaving_orchestrator.py
as part of the Elegance Pass architectural refactoring (December 2025).

This package provides:
- **WeavingContext**: Dataclass accumulating pipeline state
- **Stage Functions**: Pure function implementations of weaving steps
- **Core Utilities**: Complexity detection, metrics, background tasks

Architecture:
    The orchestrator package enables a clean separation of concerns:
    - `context.py`: Pipeline state container (WeavingContext dataclass)
    - `stages/`: Pure function implementations of each weaving step
    - `core/`: Shared utilities and extracted logic
    - `jenny/`: Jenny UI panel detection and context building

Philosophy:
    "Data flows through, orchestrator coordinates"
    Each stage reads from and writes to WeavingContext.
    The main WeavingOrchestrator chains these stages together.

Example:
    >>> from hololoom.core.orchestrator import (
    ...     WeavingContext,
    ...     create_weaving_context,
    ...     execute_step1_pattern_selection,
    ... )
    >>>
    >>> ctx = create_weaving_context(query)
    >>> ctx = await execute_step1_pattern_selection(ctx, loom_command)

Author: Claude Code (Elegance Pass - December 2025)
"""

# Context container
from .context import (
    WeavingContext,
    create_weaving_context,
)

# Stage executor protocols (Phase 2)
from .protocols import (
    StageExecutorProtocol,
    BaseStageExecutor,
)

# Component protocols (Phase 3)
from .protocols import (
    PatternSelectorProtocol,
    ThreadSelectorProtocol,
    FeatureExtractorProtocol,
    ConvergenceProtocol,
    ToolExecutorProtocol,
    SpacetimeAssemblerProtocol,
)

# Default implementations (Phase 3)
from .protocols import (
    DefaultPatternSelector,
    DefaultThreadSelector,
    DefaultFeatureExtractor,
    DefaultConvergence,
    DefaultToolExecutor,
    DefaultSpacetimeAssembler,
)

# Stage functions (Steps 0-3)
from .stages import (
    execute_step0_meta_prompt,
    execute_step1_pattern_selection,
    execute_step2_chrono_trigger,
    execute_step3_thread_selection,
)

# Stage functions (Steps 4-6: Parallel Execution)
from .stages import (
    create_resonance_shed,
    create_warp_space,
    select_pattern_embedder,
    execute_steps_4_6_parallel,
    execute_step5_5_warp_compute,
    execute_step6_5_beta_wave_packing,
)

# Stage functions (Steps 7-9: Convergence, Execution, Output)
from .stages import (
    execute_step7_convergence,
    execute_step8_tool_execution,
    execute_step9_spacetime_fabric,
)

# Core utilities (already extracted)
from .core import (
    assess_complexity_level,
    create_provenance_trace,
)

# Jenny UI integration (already extracted)
from .jenny import (
    detect_jenny_panel_type,
    build_jenny_panel_context,
)

# Pipeline (Phase 2 - Day 10)
from .pipeline import (
    ExecutorPipeline,
    create_default_pipeline,
    create_minimal_pipeline,
)

# Stage Executor Classes (Phase 2)
from .stages.executors import (
    MetaPromptExecutor,
    PatternSelectionExecutor,
    ChronoTriggerExecutor,
    ThreadSelectionExecutor,
    ParallelFeatureExecutor,
    ConvergenceExecutor,
    ToolExecutionExecutor,
    SpacetimeExecutor,
)

# Protocol Factory (Phase 3 - Day 13)
from .protocol_factory import (
    OrchestratorComponents,
    create_component_defaults,
    create_orchestrator_from_protocols,
    create_pipeline_with_protocols,
    ProtocolOrchestrator,
)

# Merged from hololoom.weaving (Wave 2 consolidation, 2026-02-27)
from .weaving_protocols import (
    WeavingStageProtocol,
    ComplexityStrategyProtocol,
    StageResult,
)

# Weaving stage classes (merged from hololoom.weaving.stages)
from .stages import (
    PatternSelectionStage,
    TemporalControlStage,
    FeatureExtractionStage,
    MemoryRetrievalStage,
    DecisionCollapseStage,
)

__all__ = [
    # Context
    'WeavingContext',
    'create_weaving_context',

    # Protocols (Phase 2)
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

    # Steps 0-3
    'execute_step0_meta_prompt',
    'execute_step1_pattern_selection',
    'execute_step2_chrono_trigger',
    'execute_step3_thread_selection',

    # Steps 4-6: Component creation
    'create_resonance_shed',
    'create_warp_space',
    'select_pattern_embedder',

    # Steps 4-6: Parallel execution
    'execute_steps_4_6_parallel',

    # Post-parallel steps
    'execute_step5_5_warp_compute',
    'execute_step6_5_beta_wave_packing',

    # Steps 7-9
    'execute_step7_convergence',
    'execute_step8_tool_execution',
    'execute_step9_spacetime_fabric',

    # Core utilities
    'assess_complexity_level',
    'create_provenance_trace',

    # Jenny UI
    'detect_jenny_panel_type',
    'build_jenny_panel_context',

    # Pipeline (Phase 2)
    'ExecutorPipeline',
    'create_default_pipeline',
    'create_minimal_pipeline',

    # Stage Executor Classes (Phase 2)
    'MetaPromptExecutor',
    'PatternSelectionExecutor',
    'ChronoTriggerExecutor',
    'ThreadSelectionExecutor',
    'ParallelFeatureExecutor',
    'ConvergenceExecutor',
    'ToolExecutionExecutor',
    'SpacetimeExecutor',

    # Protocol Factory (Phase 3)
    'OrchestratorComponents',
    'create_component_defaults',
    'create_orchestrator_from_protocols',
    'create_pipeline_with_protocols',
    'ProtocolOrchestrator',

    # Merged from hololoom.weaving (Wave 2)
    'WeavingStageProtocol',
    'ComplexityStrategyProtocol',
    'StageResult',
    'PatternSelectionStage',
    'TemporalControlStage',
    'FeatureExtractionStage',
    'MemoryRetrievalStage',
    'DecisionCollapseStage',
]
