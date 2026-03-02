#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Context - Pipeline State Container
==========================================

Dataclass that accumulates all intermediate state through the weaving pipeline.
Part of the Elegance Pass architectural refactoring (December 2025).

This context object enables:
- Clear data flow visualization
- Testable isolated stages
- Easy stage addition/removal
- Complete state documentation

Philosophy:
    "Data flows through, orchestrator coordinates"
    Each stage reads from and writes to this context.
    Steps become pure functions: step_n(context, deps) -> context

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from HoloLoom.protocols.types import Query, Features, Context, MemoryShard
    from HoloLoom.protocols import ComplexityLevel, ProvenanceTrace
    from HoloLoom.loom.command import PatternCard, PatternSpec
    from HoloLoom.chrono.trigger import ChronoTrigger, TemporalWindow
    from HoloLoom.resonance.shed import ResonanceShed
    from HoloLoom.warp.space import WarpSpace
    from HoloLoom.convergence.engine import ConvergenceEngine, CollapseResult
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
    from HoloLoom.alignment.safety_guardrails import SafetyDecision, ActionRequest


@dataclass
class WeavingContext:
    """
    Accumulated state through the weaving pipeline.

    This dataclass flows through all 9 stages of the weaving cycle,
    accumulating intermediate results at each step. Each stage reads
    what it needs and writes its outputs.

    Design Principles:
    - All fields are Optional (stages may be skipped)
    - Input fields are immutable after creation
    - Helper methods for common operations
    - Serializable for debugging/persistence

    Stages:
        Step 0: Meta-Prompt Enhancement (optional)
        Step 1: Pattern Selection (Loom Command)
        Step 2: Chrono Trigger (Temporal Window)
        Step 3: Thread Selection (Yarn Graph)
        Step 4: Resonance Shed (Feature Extraction)
        Step 5: Warp Space (Tensor Manifold)
        Step 6: Memory Retrieval (Context Assembly)
        Step 7: Convergence Engine (Tool Selection)
        Step 8: Tool Execution (Safety Gating)
        Step 9: Spacetime Fabric (Result Assembly)

    Example:
        >>> ctx = WeavingContext(query=Query(text="What is Thompson Sampling?"))
        >>> ctx = await execute_step1_pattern_selection(ctx, loom_command, ...)
        >>> ctx = await execute_step2_chrono_trigger(ctx, ...)
        >>> # ... continue through all steps
        >>> return ctx.spacetime
    """

    # ========================================================================
    # Input (immutable after creation)
    # ========================================================================
    query: 'Query'
    pattern_override: Optional['PatternCard'] = None
    complexity_override: Optional['ComplexityLevel'] = None
    auto_enhance: Optional[bool] = None

    # ========================================================================
    # Step 0: Meta-Prompt Enhancement (optional)
    # ========================================================================
    enhanced_query: Optional['Query'] = None
    original_query_text: Optional[str] = None
    enhancement_applied: bool = False

    # ========================================================================
    # Step 1: Pattern Selection (Loom Command)
    # ========================================================================
    pattern_spec: Optional['PatternSpec'] = None
    complexity: Optional['ComplexityLevel'] = None

    # ========================================================================
    # Step 2: Chrono Trigger (Temporal Window)
    # ========================================================================
    chrono: Optional['ChronoTrigger'] = None
    temporal_window: Optional['TemporalWindow'] = None

    # ========================================================================
    # Step 3: Thread Selection (Yarn Graph / Shuttle)
    # ========================================================================
    threads: List['MemoryShard'] = field(default_factory=list)
    thread_ids: List[str] = field(default_factory=list)
    thread_texts: List[str] = field(default_factory=list)
    shuttle_used: bool = False

    # ========================================================================
    # Step 4: Resonance Shed (Feature Extraction)
    # ========================================================================
    dot_plasma: Optional[Dict[str, Any]] = None
    resonance_shed: Optional['ResonanceShed'] = None
    features: Optional['Features'] = None
    pattern_embedder: Optional[Any] = None  # MatryoshkaEmbeddings or LinguisticGate
    semantic_calculus: Optional[Any] = None

    # ========================================================================
    # Step 5: Warp Space (Tensor Manifold)
    # ========================================================================
    warp_space: Optional['WarpSpace'] = None
    warp_operations: List[Tuple[str, str, Any]] = field(default_factory=list)
    warp_compute_results: Optional[Dict[str, Any]] = None

    # ========================================================================
    # Step 6: Memory Retrieval (Context Assembly)
    # ========================================================================
    shards: List['MemoryShard'] = field(default_factory=list)
    shard_texts: List[str] = field(default_factory=list)
    hits: List[Tuple[Any, float]] = field(default_factory=list)
    retrieval_context: Optional['Context'] = None
    packed_context: Optional[Any] = None  # BetaWaveContextPacker result

    # ========================================================================
    # Step 7: Convergence Engine (Tool Selection)
    # ========================================================================
    policy: Optional[Any] = None
    action_plan: Optional[Any] = None  # ActionPlan
    convergence_engine: Optional['ConvergenceEngine'] = None
    collapse_result: Optional['CollapseResult'] = None
    neural_probs: Optional[np.ndarray] = None
    gradient_decision: Optional[Any] = None  # GradientDecision

    # ========================================================================
    # Step 8: Tool Execution (Safety Gating)
    # ========================================================================
    action_request: Optional['ActionRequest'] = None
    safety_decision: Optional['SafetyDecision'] = None
    tool_result: Optional[Dict[str, Any]] = None
    safety_blocked: bool = False

    # ========================================================================
    # Step 9: Spacetime Fabric (Result Assembly)
    # ========================================================================
    spacetime: Optional['Spacetime'] = None
    trace: Optional['WeavingTrace'] = None

    # ========================================================================
    # Metadata & Timing
    # ========================================================================
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Awareness/Consciousness Integration (Phase 1 - November 2025)
    awareness_context: Optional[Dict[str, Any]] = None

    # Conscience Integration (Phase 2C - December 2025)
    conscience_decision: Optional[Any] = None
    conscience_blocked: bool = False

    # Provenance & Tracing
    provenance: Optional['ProvenanceTrace'] = None

    # Timing
    start_time: Optional[datetime] = None

    # Query routing (for fast path detection)
    query_classification: Optional[Any] = None
    fast_path_used: bool = False

    # Cache
    cache_hit: bool = False

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def record_timing(self, stage: str, duration_ms: float) -> None:
        """
        Record timing for a stage.

        Args:
            stage: Stage name (e.g., 'pattern_selection', 'feature_extraction')
            duration_ms: Duration in milliseconds
        """
        self.stage_timings[stage] = duration_ms

    def start_timer(self) -> float:
        """
        Start a timer for a stage.

        Returns:
            Start time in seconds (use with record_timing_since)
        """
        return time.time()

    def record_timing_since(self, stage: str, start: float) -> float:
        """
        Record timing since a start time.

        Args:
            stage: Stage name
            start: Start time from start_timer()

        Returns:
            Duration in milliseconds
        """
        duration_ms = (time.time() - start) * 1000
        self.stage_timings[stage] = duration_ms
        return duration_ms

    def add_error(self, error: str) -> None:
        """Add error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)

    def add_warp_operation(
        self,
        operation: str,
        details: Any = None,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Record a warp space operation.

        Args:
            operation: Operation name ('tension', 'compute', 'detension')
            details: Operation details (count, entropy, etc.)
            timestamp: ISO timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.warp_operations.append((timestamp, operation, details))

    @property
    def total_duration_ms(self) -> float:
        """Calculate total duration from start time to now."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds() * 1000

    @property
    def current_query(self) -> 'Query':
        """Get the current query (enhanced or original)."""
        return self.enhanced_query if self.enhanced_query else self.query

    @property
    def current_query_text(self) -> str:
        """Get the current query text."""
        return self.current_query.text

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings occurred."""
        return len(self.warnings) > 0

    @property
    def is_blocked(self) -> bool:
        """Check if processing was blocked (safety or conscience)."""
        return self.safety_blocked or self.conscience_blocked

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for debugging/serialization.

        Returns:
            Dictionary representation (non-recursive, string representations)
        """
        return {
            # Input
            'query_text': self.query.text if self.query else None,
            'pattern_override': self.pattern_override.value if self.pattern_override else None,
            'complexity_override': self.complexity_override.name if self.complexity_override else None,

            # Processing state
            'enhancement_applied': self.enhancement_applied,
            'pattern_spec_name': self.pattern_spec.name if self.pattern_spec else None,
            'complexity': self.complexity.name if self.complexity else None,

            # Counts
            'thread_count': len(self.threads),
            'shard_count': len(self.shards),
            'hit_count': len(self.hits),

            # Tool selection
            'tool_selected': self.collapse_result.tool if self.collapse_result else None,
            'tool_confidence': self.collapse_result.confidence if self.collapse_result else None,

            # Status
            'cache_hit': self.cache_hit,
            'fast_path_used': self.fast_path_used,
            'safety_blocked': self.safety_blocked,
            'conscience_blocked': self.conscience_blocked,
            'has_errors': self.has_errors,
            'has_warnings': self.has_warnings,

            # Timings
            'stage_timings': self.stage_timings.copy(),
            'total_duration_ms': self.total_duration_ms,

            # Issues
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
        }

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        tool = self.collapse_result.tool if self.collapse_result else "pending"
        conf = f"{self.collapse_result.confidence:.2f}" if self.collapse_result else "?"
        return (
            f"WeavingContext("
            f"query='{self.current_query_text[:30]}...', "
            f"threads={len(self.threads)}, "
            f"shards={len(self.shards)}, "
            f"tool={tool}, "
            f"conf={conf}, "
            f"duration={self.total_duration_ms:.1f}ms"
            f")"
        )


def create_weaving_context(
    query: 'Query',
    pattern_override: Optional['PatternCard'] = None,
    complexity: Optional['ComplexityLevel'] = None,
    auto_enhance: Optional[bool] = None
) -> WeavingContext:
    """
    Factory function to create a WeavingContext with proper initialization.

    Args:
        query: User query
        pattern_override: Optional pattern card override
        complexity: Optional complexity level override
        auto_enhance: Optional auto-enhancement override

    Returns:
        Initialized WeavingContext ready for pipeline processing

    Example:
        >>> ctx = create_weaving_context(Query(text="What is Thompson Sampling?"))
        >>> ctx = await execute_step1(ctx, ...)
    """
    return WeavingContext(
        query=query,
        pattern_override=pattern_override,
        complexity_override=complexity,
        auto_enhance=auto_enhance,
        start_time=datetime.now()
    )


__all__ = [
    'WeavingContext',
    'create_weaving_context',
]
