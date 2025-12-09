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
    >>> from HoloLoom.orchestrator.context import create_weaving_context
    >>> from HoloLoom.orchestrator.stages import (
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

__all__ = [
    # Steps 0-3
    'execute_step0_meta_prompt',
    'execute_step1_pattern_selection',
    'execute_step2_chrono_trigger',
    'execute_step3_thread_selection',
]
