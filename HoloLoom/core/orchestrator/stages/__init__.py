#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Pipeline Stages - Pure Function Implementations
========================================================

Part of the Elegance Pass architectural refactoring (December 2025).

This module contains pure function implementations of each weaving stage.
Each function follows the signature: step_n(ctx, deps...) -> ctx

Stages are organized into logical groups:
- Steps 0-3: Query setup and thread selection
- Steps 4-6: Parallel feature extraction (resonance, warp, retrieval)
- Steps 7-9: Convergence, execution, and output

Philosophy:
    "Data flows through, orchestrator coordinates"
    Each stage reads from and writes to WeavingContext.
    Stages are pure functions with explicit dependencies.

Example:
    >>> from HoloLoom.core.orchestrator.context import create_weaving_context
    >>> from HoloLoom.core.orchestrator.stages import (
    ...     execute_step0_meta_prompt,
    ...     execute_step1_pattern_selection,
    ...     execute_step2_chrono_trigger,
    ...     execute_step3_thread_selection,
    ... )
    >>>
    >>> ctx = create_weaving_context(query)
    >>> ctx = await execute_step0_meta_prompt(ctx, ...)
    >>> ctx = await execute_step1_pattern_selection(ctx, ...)
    >>> ctx = await execute_step2_chrono_trigger(ctx, ...)
    >>> ctx = await execute_step3_thread_selection(ctx, ...)

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

from .steps_0_3 import (
    # Step 0: Meta-Prompt Enhancement
    execute_step0_meta_prompt,

    # Step 1: Pattern Selection (Loom Command)
    execute_step1_pattern_selection,

    # Step 2: Chrono Trigger (Temporal Window)
    execute_step2_chrono_trigger,

    # Step 3: Thread Selection (Yarn Graph / Shuttle)
    execute_step3_thread_selection,
)

from .steps_4_6 import (
    # Component creation
    create_resonance_shed,
    create_warp_space,
    select_pattern_embedder,

    # Steps 4-6: Parallel execution
    execute_steps_4_6_parallel,

    # Post-parallel steps
    execute_step5_5_warp_compute,
    execute_step6_5_beta_wave_packing,
)

from .steps_7_9 import (
    # Step 7: Convergence Engine (Decision Collapse)
    execute_step7_convergence,

    # Step 8: Tool Execution (Safety Gating)
    execute_step8_tool_execution,

    # Step 9: Spacetime Fabric (Result Assembly)
    execute_step9_spacetime_fabric,
)

# Merged from HoloLoom.weaving.stages (Wave 2 consolidation, 2026-02-27)
from .pattern_selection import PatternSelectionStage
from .temporal_control import TemporalControlStage
from .feature_extraction import FeatureExtractionStage
from .memory_retrieval import MemoryRetrievalStage
from .decision_collapse import DecisionCollapseStage

__all__ = [
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

    # Weaving stage classes (merged from HoloLoom.weaving.stages)
    'PatternSelectionStage',
    'TemporalControlStage',
    'FeatureExtractionStage',
    'MemoryRetrievalStage',
    'DecisionCollapseStage',
]
