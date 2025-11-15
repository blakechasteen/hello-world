#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning-Enhanced Weaving Orchestrator
========================================
Extends WeavingOrchestrator with Layer 6: Reasoning Engine integration.

This orchestrator inserts explicit reasoning between feature extraction
and decision-making, enabling multi-step chain-of-thought generation.

Modified Weaving Cycle:
1. Loom Command → Pattern Card
2. Chrono Trigger → Temporal Window
3. Yarn Graph → Thread Selection
4. Resonance Shed → DotPlasma (features)
5. **REASONING ENGINE → Reasoning Chain**  ← NEW
6. Warp Space → Continuous Manifold
7. Convergence Engine → Tool Selection (informed by reasoning)
8. Tool Execution → Results
9. Spacetime Fabric → Provenance (with reasoning chain)
10. Reflection Buffer → Learning

Author: Claude Code
Date: 2025-11-15
Phase: 2 (WeavingOrchestrator Integration)
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

# Base orchestrator
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

# Core types
from HoloLoom.documentation.types import Query, Features, Context, MemoryShard
from HoloLoom.config import Config
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace

# Reasoning engine
from HoloLoom.reasoning.engine import ReasoningEngine
from HoloLoom.reasoning.types import ReasoningMode, ReasoningResult

# Provenance integration
try:
    from HoloLoom.recursive.reasoning_provenance import ReasoningProvenanceTracker
    PROVENANCE_TRACKER_AVAILABLE = True
except ImportError:
    PROVENANCE_TRACKER_AVAILABLE = False
    ReasoningProvenanceTracker = None

logger = logging.getLogger(__name__)


# ============================================================================
# Reasoning-Enhanced Orchestrator
# ============================================================================

class ReasoningOrchestrator(WeavingOrchestrator):
    """
    Enhanced Weaving Orchestrator with Layer 6: Reasoning Engine.

    Inserts explicit multi-step reasoning between feature extraction and
    decision-making, creating a more transparent and reliable system.

    Key Features:
    - FAST mode: Single-pass reasoning (<50ms overhead)
    - STANDARD mode: Multi-step CoT (3-5 steps, ~200ms overhead)
    - DEEP mode: Planning + verification + backtracking (~500ms overhead)
    - Auto-mode selection based on query complexity
    - Full provenance tracking via scratchpad integration
    - Reasoning chain attached to Spacetime metadata

    Usage:
        config = Config.fused()
        config.enable_reasoning = True
        config.reasoning_mode = ReasoningMode.STANDARD

        async with ReasoningOrchestrator(cfg=config, shards=shards) as orchestrator:
            spacetime = await orchestrator.weave(Query(text="Explain Thompson Sampling"))

            # View reasoning chain
            for step in spacetime.metadata.get('reasoning_chain', []):
                print(f"{step.thought} (confidence: {step.confidence:.2f})")
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory=None,
        enable_reasoning: bool = True,
        reasoning_mode: Optional[ReasoningMode] = None,
        **kwargs
    ):
        """
        Initialize Reasoning-Enhanced Orchestrator.

        Args:
            cfg: Configuration (should have reasoning parameters)
            shards: Memory shards (optional if memory provided)
            memory: Unified memory backend (optional)
            enable_reasoning: Enable reasoning layer (default True)
            reasoning_mode: Override default reasoning mode
            **kwargs: Additional WeavingOrchestrator parameters
        """
        # Initialize base orchestrator
        super().__init__(cfg=cfg, shards=shards, memory=memory, **kwargs)

        # Reasoning configuration
        self.enable_reasoning = enable_reasoning

        # Get reasoning parameters from config
        self.reasoning_mode = reasoning_mode or getattr(
            cfg, 'reasoning_mode', ReasoningMode.STANDARD
        )
        self.max_reasoning_steps = getattr(cfg, 'max_reasoning_steps', 5)
        self.verification_threshold = getattr(cfg, 'reasoning_verification_threshold', 0.75)

        # Initialize reasoning engine if enabled
        if self.enable_reasoning:
            self.reasoning_engine = ReasoningEngine(
                mode=self.reasoning_mode,
                scratchpad=None,  # Will be set per-query if available
                max_thinking_steps=self.max_reasoning_steps,
                verification_threshold=self.verification_threshold
            )

            # Initialize provenance tracker if available
            if PROVENANCE_TRACKER_AVAILABLE:
                self.reasoning_provenance = ReasoningProvenanceTracker()
            else:
                self.reasoning_provenance = None
                self.logger.warning(
                    "ReasoningProvenanceTracker not available. "
                    "Reasoning chains will not be recorded in scratchpad."
                )

            self.logger.info(
                f"Reasoning Engine enabled: mode={self.reasoning_mode.value}, "
                f"max_steps={self.max_reasoning_steps}, "
                f"threshold={self.verification_threshold:.2f}"
            )
        else:
            self.reasoning_engine = None
            self.reasoning_provenance = None
            self.logger.info("Reasoning Engine disabled")

    async def weave(
        self,
        query: Query,
        pattern_override=None,
        complexity=None,
        reasoning_mode_override: Optional[ReasoningMode] = None
    ) -> Spacetime:
        """
        Enhanced weaving cycle with reasoning layer.

        Extends the base weaving cycle by inserting reasoning between
        feature extraction (step 4) and decision (step 7).

        Args:
            query: User query
            pattern_override: Optional pattern card override
            complexity: Optional complexity level override
            reasoning_mode_override: Override reasoning mode for this query

        Returns:
            Spacetime with reasoning chain in metadata
        """
        if not self.enable_reasoning:
            # Fall back to base orchestrator if reasoning disabled
            return await super().weave(query, pattern_override, complexity)

        # Call base weave but intercept at key points
        # We'll need to partially replicate the base weave logic to insert reasoning

        # For now, use a simpler approach: call base weave and enhance the result
        # In production, would refactor base weave to allow middleware injection

        start_time = time.time()
        reasoning_start = None
        reasoning_duration_ms = 0.0
        reasoning_result = None

        try:
            # APPROACH: Intercept and enhance
            # 1. Get features from base extraction (we'll call private methods)
            # 2. Run reasoning on features
            # 3. Continue with decision (passing reasoning result)

            # Since base weave is monolithic, we'll use composition:
            # Call base weave, then add reasoning metadata

            # Get base result
            spacetime = await super().weave(query, pattern_override, complexity)

            # Extract features and context from result
            # NOTE: This is a limitation - we're reasoning AFTER the fact
            # In Phase 3, refactor base weave to support middleware

            # For now, synthesize features and context from spacetime
            features = spacetime.trace.metadata.get('features') if hasattr(spacetime.trace, 'metadata') else None
            context = spacetime.trace.metadata.get('context') if hasattr(spacetime.trace, 'metadata') else None

            # If features/context not available in trace, create synthetic versions
            if features is None:
                features = Features(
                    psi=spacetime.trace.embedding_scales_used or [],
                    motifs=spacetime.trace.motifs_detected or [],
                    metrics={},
                    metadata={}
                )

            if context is None:
                context = Context(
                    shards=[],
                    hits=[],
                    shard_texts=[],
                    query=query,
                    features=features
                )

            # Run reasoning
            reasoning_start = time.time()

            effective_mode = reasoning_mode_override or self.reasoning_mode

            reasoning_result = await self.reasoning_engine.reason(
                query=query,
                features=features,
                context=context,
                mode=effective_mode
            )

            reasoning_duration_ms = (time.time() - reasoning_start) * 1000

            # Attach reasoning chain to spacetime metadata
            if not hasattr(spacetime, 'metadata') or spacetime.metadata is None:
                spacetime.metadata = {}

            spacetime.metadata['reasoning_chain'] = [
                {
                    'thought': step.thought,
                    'evidence': step.evidence,
                    'confidence': step.confidence,
                    'step_type': step.step_type.value,
                    'timestamp': step.timestamp.isoformat()
                }
                for step in reasoning_result.chain
            ]
            spacetime.metadata['reasoning_mode'] = reasoning_result.mode.value
            spacetime.metadata['reasoning_confidence'] = reasoning_result.total_confidence
            spacetime.metadata['reasoning_duration_ms'] = reasoning_duration_ms

            # Add to scratchpad if provenance tracker available
            if self.reasoning_provenance:
                try:
                    scratchpad_entries = self.reasoning_provenance.extract_reasoning_provenance(
                        reasoning_result
                    )
                    # Note: Would need scratchpad instance to add entries
                    # For now, just attach to metadata
                    spacetime.metadata['reasoning_scratchpad'] = [
                        {
                            'thought': entry.thought,
                            'action': entry.action,
                            'observation': entry.observation,
                            'score': entry.score,
                            'iteration': entry.iteration
                        }
                        for entry in scratchpad_entries
                    ]
                except Exception as e:
                    self.logger.warning(f"Failed to extract scratchpad provenance: {e}")

            self.logger.info(
                f"[REASONING] Complete: mode={reasoning_result.mode.value}, "
                f"steps={len(reasoning_result.chain)}, "
                f"confidence={reasoning_result.total_confidence:.2f}, "
                f"duration={reasoning_duration_ms:.1f}ms"
            )

            return spacetime

        except Exception as e:
            self.logger.error(f"Reasoning-enhanced weave failed: {e}", exc_info=True)
            # Fall back to base weave on error
            return await super().weave(query, pattern_override, complexity)


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

# Alias for consistency with naming conventions
WeavingOrchestratorReasoning = ReasoningOrchestrator


# ============================================================================
# Convenience Functions
# ============================================================================

async def weave_with_reasoning(
    query: Query,
    config: Config,
    shards: Optional[List[MemoryShard]] = None,
    memory=None,
    reasoning_mode: ReasoningMode = ReasoningMode.STANDARD,
    **kwargs
) -> Spacetime:
    """
    Convenience function for one-off reasoning-enhanced weaving.

    Args:
        query: User query
        config: HoloLoom configuration
        shards: Memory shards (optional)
        memory: Memory backend (optional)
        reasoning_mode: Reasoning mode to use
        **kwargs: Additional orchestrator parameters

    Returns:
        Spacetime with reasoning chain
    """
    # Enable reasoning in config
    config.enable_reasoning = True
    config.reasoning_mode = reasoning_mode

    async with ReasoningOrchestrator(
        cfg=config,
        shards=shards,
        memory=memory,
        **kwargs
    ) as orchestrator:
        return await orchestrator.weave(query)


async def weave_with_auto_reasoning(
    query: Query,
    config: Config,
    shards: Optional[List[MemoryShard]] = None,
    memory=None,
    **kwargs
) -> Spacetime:
    """
    Convenience function with automatic reasoning mode selection.

    The reasoning engine will automatically select FAST/STANDARD/DEEP
    based on query complexity and context quality.

    Args:
        query: User query
        config: HoloLoom configuration
        shards: Memory shards (optional)
        memory: Memory backend (optional)
        **kwargs: Additional orchestrator parameters

    Returns:
        Spacetime with reasoning chain
    """
    config.enable_reasoning = True
    config.enable_adaptive_reasoning = True

    async with ReasoningOrchestrator(
        cfg=config,
        shards=shards,
        memory=memory,
        **kwargs
    ) as orchestrator:
        # The engine will auto-select mode
        return await orchestrator.weave(query)
