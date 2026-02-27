#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Pipeline Steps 0-3: Query Setup and Thread Selection
=============================================================

Part of the Elegance Pass architectural refactoring (December 2025).

Pure function implementations of the first 4 weaving steps:
- Step 0: Meta-Prompt Enhancement (optional query rewriting)
- Step 1: Pattern Selection (Loom Command selects BARE/FAST/FUSED)
- Step 2: Chrono Trigger (creates TemporalWindow)
- Step 3: Thread Selection (Shuttle or Yarn Graph)

Each function follows the signature: step_n(ctx, deps...) -> ctx
- Takes WeavingContext as first parameter
- Returns WeavingContext (mutated or new)
- Has no `self` references
- Receives all dependencies as explicit parameters

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.core.orchestrator.context import WeavingContext
    from hololoom.core.protocols.types import Query
    from hololoom.core.loom.command import LoomCommand
    from hololoom.core.chrono.trigger import ChronoTrigger, TemporalWindow
    from hololoom.core.memory.graph import KG

logger = logging.getLogger(__name__)


# ============================================================================
# Step 0: Meta-Prompt Enhancement
# ============================================================================

async def execute_step0_meta_prompt(
    ctx: 'WeavingContext',
    enable_enhancement: bool,
    proto_llm_call: Optional[Callable[[str], Any]],
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Step 0: Meta-Prompt Enhancement (optional).

    Optionally enhances the query through an LLM call before processing.
    This allows for query rewriting, clarification, or expansion.

    Args:
        ctx: Weaving context with query
        enable_enhancement: Whether to attempt enhancement
        proto_llm_call: Async callable for LLM query enhancement
        log: Optional logger (defaults to module logger)

    Returns:
        WeavingContext with enhanced_query set if enhancement applied,
        otherwise returns ctx unchanged.

    Example:
        >>> ctx = await execute_step0_meta_prompt(
        ...     ctx,
        ...     enable_enhancement=True,
        ...     proto_llm_call=orchestrator.proto_llm_call
        ... )
        >>> if ctx.enhancement_applied:
        ...     print(f"Original: {ctx.original_query_text}")
        ...     print(f"Enhanced: {ctx.current_query_text}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Check if enhancement should be applied
    should_enhance = ctx.auto_enhance if ctx.auto_enhance is not None else enable_enhancement

    if not should_enhance:
        log.debug("[STEP 0] Meta-prompt enhancement disabled")
        return ctx

    if proto_llm_call is None:
        log.debug("[STEP 0] No proto_llm_call provided, skipping enhancement")
        return ctx

    try:
        # Store original text
        ctx.original_query_text = ctx.query.text

        # Enhance the query
        enhanced_text = await proto_llm_call(ctx.query.text)

        # Create new Query with enhanced text, preserving metadata
        from hololoom.core.protocols.types import Query
        enhanced_metadata = dict(ctx.query.metadata) if ctx.query.metadata else {}
        enhanced_metadata['original_query'] = ctx.original_query_text
        enhanced_metadata['enhanced'] = True

        ctx.enhanced_query = Query(text=enhanced_text, metadata=enhanced_metadata)
        ctx.enhancement_applied = True

        duration = ctx.record_timing_since('meta_prompt_enhancement', step_start)
        log.info(
            f"[STEP 0] Enhanced query ({len(ctx.original_query_text)} -> "
            f"{len(enhanced_text)} chars) in {duration:.1f}ms"
        )

    except Exception as e:
        log.warning(f"[STEP 0] Auto-enhancement failed: {e}")
        ctx.add_warning(f"Meta-prompt enhancement failed: {e}")
        ctx.enhancement_applied = False

    return ctx


# ============================================================================
# Step 1: Pattern Selection (Loom Command)
# ============================================================================

async def execute_step1_pattern_selection(
    ctx: 'WeavingContext',
    loom_command: 'LoomCommand',
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Step 1: Loom Command selects Pattern Card.

    The Loom Command analyzes the query and selects an appropriate pattern:
    - BARE: Minimal processing (<50ms)
    - FAST: Balanced processing (<150ms)
    - FUSED: Full processing (<300ms)

    Args:
        ctx: Weaving context with query (original or enhanced)
        loom_command: LoomCommand instance for pattern selection
        emit_stage_event: Optional callback for stage events (for UI updates)
        log: Optional logger

    Returns:
        WeavingContext with pattern_spec set.

    Example:
        >>> ctx = await execute_step1_pattern_selection(ctx, loom_command)
        >>> print(f"Selected pattern: {ctx.pattern_spec.name}")
        >>> print(f"Timeout: {ctx.pattern_spec.pipeline_timeout}s")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Emit stage start event
    if emit_stage_event:
        emit_stage_event(1, "Loom Command", None)

    # Get user preference from pattern override
    user_preference = None
    if ctx.pattern_override:
        user_preference = ctx.pattern_override.value

    # Select pattern based on current query (enhanced or original)
    ctx.pattern_spec = loom_command.select_pattern(
        ctx.current_query_text,
        user_preference=user_preference
    )

    # Record timing
    duration = ctx.record_timing_since('pattern_selection', step_start)
    log.info(f"  [1] Pattern selected: {ctx.pattern_spec.name} ({duration:.1f}ms)")

    # Emit stage complete event
    if emit_stage_event:
        emit_stage_event(1, "Loom Command", duration)

    return ctx


# ============================================================================
# Step 2: Chrono Trigger (Temporal Window)
# ============================================================================

async def execute_step2_chrono_trigger(
    ctx: 'WeavingContext',
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
    lookback_days: int = 365,
    recency_bias: float = 0.5,
) -> 'WeavingContext':
    """
    Step 2: Chrono Trigger fires, creates TemporalWindow.

    Creates temporal context for memory retrieval:
    - Defines the time window for relevant memories
    - Sets recency bias for temporal weighting
    - Configures pipeline timeout from pattern spec

    Args:
        ctx: Weaving context with pattern_spec
        emit_stage_event: Optional callback for stage events
        log: Optional logger
        lookback_days: How far back to look for memories (default: 365)
        recency_bias: Weight for recent memories 0-1 (default: 0.5)

    Returns:
        WeavingContext with chrono and temporal_window set.

    Example:
        >>> ctx = await execute_step2_chrono_trigger(ctx)
        >>> print(f"Window: {ctx.temporal_window.start} to {ctx.temporal_window.end}")
        >>> print(f"Recency bias: {ctx.temporal_window.recency_bias}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Emit stage start event
    if emit_stage_event:
        emit_stage_event(2, "Chrono Trigger", None)

    # Import chrono types
    from hololoom.core.chrono.trigger import ChronoTrigger, TemporalWindow

    # Create a minimal config for Chrono using pattern spec timeout
    pipeline_timeout = ctx.pattern_spec.pipeline_timeout if ctx.pattern_spec else 30.0

    class ChronoConfig:
        def __init__(self, timeout: float):
            self.pipeline_timeout = timeout

    ctx.chrono = ChronoTrigger(
        config=ChronoConfig(pipeline_timeout),
        enable_heartbeat=False
    )

    # Create temporal window for memory retrieval
    now = datetime.now()
    ctx.temporal_window = TemporalWindow(
        start=now - timedelta(days=lookback_days),
        end=now,
        max_age=timedelta(days=lookback_days),
        recency_bias=recency_bias
    )

    # Record timing
    duration = ctx.record_timing_since('temporal_setup', step_start)
    log.info(f"  [2] Chrono Trigger fired ({duration:.1f}ms)")

    # Emit stage complete event
    if emit_stage_event:
        emit_stage_event(2, "Chrono Trigger", duration)

    return ctx


# ============================================================================
# Step 3: Thread Selection (Yarn Graph / Shuttle)
# ============================================================================

async def execute_step3_thread_selection(
    ctx: 'WeavingContext',
    yarn_graph: 'KG',
    shuttle_stage: Optional[Any] = None,
    enable_shuttle: bool = False,
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Step 3: Thread Selection from memory.

    Selects relevant memory threads using either:
    - Shuttle: MCTS-powered Warp<->Yarn intersection (advanced)
    - Yarn Graph: Simple temporal thread selection (fallback)

    Args:
        ctx: Weaving context with temporal_window
        yarn_graph: KG instance for thread selection
        shuttle_stage: Optional ShuttleStage for advanced selection
        enable_shuttle: Whether to use Shuttle (if available)
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        WeavingContext with threads, thread_ids, and thread_texts populated.

    Example:
        >>> ctx = await execute_step3_thread_selection(
        ...     ctx,
        ...     yarn_graph=kg,
        ...     enable_shuttle=False
        ... )
        >>> print(f"Selected {len(ctx.threads)} threads")
        >>> for tid in ctx.thread_ids[:3]:
        ...     print(f"  - {tid}")
    """
    log = log or logger
    step_start = ctx.start_timer()

    # Determine selection method
    use_shuttle = enable_shuttle and shuttle_stage is not None

    if use_shuttle:
        # Shuttle: MCTS-powered Warp<->Yarn intersection
        if emit_stage_event:
            emit_stage_event(3, "Shuttle (Warp<->Yarn MCTS)", None)

        ctx.threads = await shuttle_stage.select_threads(
            temporal_window=ctx.temporal_window,
            query=ctx.current_query,
            trajectory_name=None  # Auto-select via Thompson Sampling
        )
        ctx.shuttle_used = True

        duration = ctx.record_timing_since('shuttle_selection', step_start)
        log.info(f"  [3] Shuttle selected {len(ctx.threads)} threads ({duration:.1f}ms)")

        if emit_stage_event:
            emit_stage_event(3, "Shuttle", duration)

    else:
        # Fallback: Simple Yarn Graph thread selection
        if emit_stage_event:
            emit_stage_event(3, "Yarn Graph", None)

        ctx.threads = yarn_graph.select_threads(ctx.temporal_window, ctx.current_query)
        ctx.shuttle_used = False

        duration = ctx.record_timing_since('thread_selection', step_start)
        log.info(f"  [3] Selected {len(ctx.threads)} threads from Yarn Graph ({duration:.1f}ms)")

        if emit_stage_event:
            emit_stage_event(3, "Yarn Graph", duration)

    # Extract thread IDs and texts for downstream use
    ctx.thread_ids = [s.id for s in ctx.threads]
    ctx.thread_texts = [s.text for s in ctx.threads]

    return ctx


__all__ = [
    'execute_step0_meta_prompt',
    'execute_step1_pattern_selection',
    'execute_step2_chrono_trigger',
    'execute_step3_thread_selection',
]
