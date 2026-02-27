#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weaving Pipeline Steps 4-6: Parallel Feature Extraction
========================================================

Part of the Elegance Pass architectural refactoring (December 2025).

Pure function implementations of the parallelized feature extraction steps:
- Step 4: Resonance Shed (DotPlasma creation via feature threads)
- Step 5: Warp Space (tension threads into continuous manifold)
- Step 6: Memory Retrieval (multipass crawl or legacy retrieval)
- Step 5.5: Warp Compute (tensor operations in continuous space)
- Step 6.5: Beta Wave Context Packing (physics-based optimization)

These steps run in parallel using asyncio.gather for 40-120ms speedup.

Author: Claude Code (Elegance Pass - Phase 1)
Date: 2025-12-09
"""

from __future__ import annotations

import asyncio
import time
import logging
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from HoloLoom.core.orchestrator.context import WeavingContext
    from HoloLoom.config import Config
    from HoloLoom.core.loom.command import PatternSpec
    from HoloLoom.core.embedding.spectral import MatryoshkaEmbeddings
    from HoloLoom.core.resonance.shed import ResonanceShed
    from HoloLoom.core.warp.space import WarpSpace
    from HoloLoom.alignment.safety_guardrails import SafetyGuardrails

logger = logging.getLogger(__name__)


# ============================================================================
# Component Creation Helpers
# ============================================================================

def create_resonance_shed(
    ctx: 'WeavingContext',
    cfg: 'Config',
    pattern_embedder: Any,
    linguistic_gate: Optional[Any] = None,
    guardrails: Optional['SafetyGuardrails'] = None,
    log: Optional[logging.Logger] = None,
) -> 'ResonanceShed':
    """
    Create Resonance Shed based on pattern spec.

    The Resonance Shed is where feature extraction threads combine:
    - Motif Thread: Symbolic pattern detection
    - Embedding Thread: Multi-scale vector representations
    - Spectral Thread: Graph topology features

    Args:
        ctx: Weaving context with pattern_spec
        cfg: Configuration object
        pattern_embedder: Embedder to use (LinguisticGate, ZeroCopy, or standard)
        linguistic_gate: Optional linguistic gate for Phase 5 integration
        guardrails: Optional safety guardrails

    Returns:
        Configured ResonanceShed instance
    """
    log = log or logger
    pattern_spec = ctx.pattern_spec

    # Import required components
    from HoloLoom.core.resonance.motif_detector import create_motif_detector
    from HoloLoom.core.resonance.spectral_fusion import SpectralFusion
    from HoloLoom.core.resonance.shed import ResonanceShed

    # Create components based on pattern spec
    motif_detector = create_motif_detector(mode=pattern_spec.motif_mode)
    spectral_fusion = SpectralFusion() if pattern_spec.enable_spectral else None

    # Create semantic analyzer if enabled by pattern
    semantic_calculus = None
    if pattern_spec.enable_semantic_flow:
        try:
            from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
            from HoloLoom.semantic_calculus.config import SemanticCalculusConfig

            sem_config = SemanticCalculusConfig.from_pattern_spec(pattern_spec)
            embed_fn = lambda words: pattern_embedder.encode(words)
            semantic_calculus = create_semantic_analyzer(embed_fn, config=sem_config)

            log.info(
                f"  [4a] Semantic analyzer enabled ({sem_config.dimensions}D, "
                f"cache={sem_config.enable_cache}, ethics={sem_config.compute_ethics})"
            )
        except Exception as e:
            log.warning(f"  [4a] Semantic analyzer creation failed: {e}")

    # Create Resonance Shed
    resonance_shed = ResonanceShed(
        motif_detector=motif_detector,
        embedder=pattern_embedder,
        spectral_fusion=spectral_fusion,
        semantic_calculus=semantic_calculus,
        interference_mode="weighted_sum",
        target_scale=max(pattern_spec.scales),
        guardrails=guardrails,
    )

    return resonance_shed


def create_warp_space(
    ctx: 'WeavingContext',
    embedder: 'MatryoshkaEmbeddings',
    guardrails: Optional['SafetyGuardrails'] = None,
) -> 'WarpSpace':
    """
    Create Warp Space based on pattern spec.

    The Warp Space is a tensioned tensor field where:
    - Discrete yarn threads are tensioned into continuous manifold
    - Tensor operations can be performed
    - Results collapse back to discrete space

    Args:
        ctx: Weaving context with pattern_spec
        embedder: Matryoshka embeddings for vector operations
        guardrails: Optional safety guardrails

    Returns:
        Configured WarpSpace instance
    """
    from HoloLoom.core.resonance.spectral_fusion import SpectralFusion
    from HoloLoom.core.warp.space import WarpSpace

    pattern_spec = ctx.pattern_spec
    spectral_fusion = SpectralFusion() if pattern_spec.enable_spectral else None

    return WarpSpace(
        embedder=embedder,
        scales=pattern_spec.scales,
        spectral_fusion=spectral_fusion,
        guardrails=guardrails,
    )


def select_pattern_embedder(
    cfg: 'Config',
    pattern_spec: 'PatternSpec',
    linguistic_gate: Optional[Any] = None,
    log: Optional[logging.Logger] = None,
) -> Any:
    """
    Select the appropriate embedder based on configuration.

    Embedder selection priority:
    1. Linguistic Gate (Phase 5 with compositional cache)
    2. Zero-Copy Embeddings (1.4x speedup, 50% memory savings)
    3. Standard Matryoshka Embeddings

    Args:
        cfg: Configuration object
        pattern_spec: Pattern specification with scales
        linguistic_gate: Optional linguistic gate instance

    Returns:
        Selected embedder instance
    """
    log = log or logger

    # Phase 5 Integration: Use linguistic gate if enabled
    if linguistic_gate and cfg.enable_linguistic_gate:
        log.info(
            f"  [4a] Phase 5 Linguistic Gate enabled "
            f"(mode={cfg.linguistic_mode}, cache={cfg.use_compositional_cache})"
        )
        return linguistic_gate

    # Zero-copy embeddings
    if cfg.enable_zero_copy_embeddings:
        from HoloLoom.core.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=pattern_spec.scales,
            base_model_name=cfg.base_model_name,
            store_path=cfg.zero_copy_cache_path,
            max_cache_size=cfg.zero_copy_cache_size
        )
        log.info(
            f"  [4a] Zero-copy embeddings enabled "
            f"(cache={cfg.zero_copy_cache_path}, size={cfg.zero_copy_cache_size})"
        )
        return embedder

    # Standard matryoshka embeddings
    from HoloLoom.core.embedding.spectral import MatryoshkaEmbeddings
    return MatryoshkaEmbeddings(
        sizes=pattern_spec.scales,
        base_model_name=cfg.base_model_name
    )


# ============================================================================
# Individual Step Functions (for parallel execution)
# ============================================================================

async def _step4_feature_extraction(
    ctx: 'WeavingContext',
    resonance_shed: 'ResonanceShed',
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], float]:
    """
    Step 4: Extract features through Resonance Shed.

    Creates DotPlasma - the flowing continuous representation containing:
    - Motif features (symbolic patterns)
    - Embeddings (multi-scale vectors)
    - Spectral features (graph topology)

    Args:
        ctx: Weaving context with current query
        resonance_shed: Configured ResonanceShed instance
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        Tuple of (dot_plasma dict, duration_ms)
    """
    log = log or logger
    start = time.time()

    if emit_stage_event:
        emit_stage_event(4, "Resonance Shed", None)

    # Weave features through resonance shed
    dot_plasma = await resonance_shed.weave(
        text=ctx.current_query_text,
        context_graph=None  # Could add KG here for graph-aware features
    )

    duration = (time.time() - start) * 1000
    thread_count = len(dot_plasma.get('threads', []))
    log.info(f"  [4] DotPlasma created with {thread_count} feature threads ({duration:.1f}ms)")

    if emit_stage_event:
        emit_stage_event(4, "Resonance Shed", duration)

    return dot_plasma, duration


async def _step5_warp_tensioning(
    ctx: 'WeavingContext',
    warp_space: 'WarpSpace',
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[str, str, Any]], float]:
    """
    Step 5: Tension threads into continuous manifold.

    Transforms discrete yarn threads into continuous tensor space
    for mathematical operations.

    Args:
        ctx: Weaving context with thread_texts and thread_ids
        warp_space: Configured WarpSpace instance
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        Tuple of (warp_operations list, duration_ms)
    """
    log = log or logger
    start = time.time()

    if emit_stage_event:
        emit_stage_event(5, "Warp Space", None)

    # Tension threads into continuous manifold
    await warp_space.tension(ctx.thread_texts, thread_ids=ctx.thread_ids)

    duration = (time.time() - start) * 1000
    warp_operations = [(datetime.now().isoformat(), "tension", len(ctx.thread_ids))]

    log.info(f"  [5] Warp Space tensioned with {len(ctx.thread_ids)} threads ({duration:.1f}ms)")

    if emit_stage_event:
        emit_stage_event(5, "Warp Space", duration)

    return warp_operations, duration


async def _step6_memory_retrieval(
    ctx: 'WeavingContext',
    memory: Optional[Any],
    retriever: Optional[Any],
    complexity: Any,
    provenance: Any,
    multipass_memory_crawl: Optional[Callable] = None,
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Any], List[str], List[Tuple[Any, float]], float]:
    """
    Step 6: Retrieve context with multipass memory crawling.

    Uses intelligent retrieval strategy:
    - Multipass crawl: Graph traversal with gated retrieval
    - Legacy retriever: Traditional static shard retrieval

    Args:
        ctx: Weaving context with current query and pattern_spec
        memory: Dynamic memory backend (for multipass crawl)
        retriever: Legacy retriever (fallback)
        complexity: Complexity level for retrieval tuning
        provenance: Provenance trace for logging
        multipass_memory_crawl: Async function for multipass crawling
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        Tuple of (shards, shard_texts, hits, duration_ms)
    """
    log = log or logger
    start = time.time()

    if emit_stage_event:
        emit_stage_event(6, "Memory Retrieval", None)

    shards = []
    shard_texts = []
    hits = []

    if memory and multipass_memory_crawl:
        # Multipass crawling with gated retrieval and graph traversal
        shards = await multipass_memory_crawl(memory, ctx.current_query, complexity, provenance)
        shard_texts = [shard.text for shard in shards]
        hits = [(shard, 1.0) for shard in shards]
        log.info(f"  [6] Multipass crawl retrieved {len(shards)} shards")

    elif retriever:
        # Legacy: Traditional static shard retrieval
        hits = await retriever.search(
            query=ctx.current_query_text,
            k=ctx.pattern_spec.retrieval_k if ctx.pattern_spec else 10,
            fast=(ctx.pattern_spec.retrieval_mode == "fast") if ctx.pattern_spec else True
        )
        shards = [shard for shard, _ in hits]
        shard_texts = [shard.text for shard in shards]
        log.info(f"  [6] Legacy retriever fetched {len(shards)} shards")

    else:
        log.warning("No memory source available (no shards or memory backend)")

    duration = (time.time() - start) * 1000
    log.info(f"  [6] Retrieved {len(hits)} context shards ({duration:.1f}ms)")

    if emit_stage_event:
        emit_stage_event(6, "Memory Retrieval", duration)

    return shards, shard_texts, hits, duration


# ============================================================================
# Main Parallel Execution Function
# ============================================================================

async def execute_steps_4_6_parallel(
    ctx: 'WeavingContext',
    cfg: 'Config',
    embedder: 'MatryoshkaEmbeddings',
    memory: Optional[Any] = None,
    retriever: Optional[Any] = None,
    complexity: Optional[Any] = None,
    provenance: Optional[Any] = None,
    linguistic_gate: Optional[Any] = None,
    guardrails: Optional['SafetyGuardrails'] = None,
    multipass_memory_crawl: Optional[Callable] = None,
    emit_stage_event: Optional[Callable[[int, str, Optional[float]], None]] = None,
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Steps 4-6: Parallelized feature extraction, warp tensioning, retrieval.

    Executes three independent steps concurrently for 40-120ms speedup:
    - Step 4: Feature extraction through Resonance Shed
    - Step 5: Warp Space tensioning
    - Step 6: Memory retrieval

    Args:
        ctx: Weaving context with pattern_spec and threads
        cfg: Configuration object
        embedder: Base embedder for fallback
        memory: Dynamic memory backend (for multipass crawl)
        retriever: Legacy retriever (fallback)
        complexity: Complexity level for retrieval tuning
        provenance: Provenance trace for logging
        linguistic_gate: Optional linguistic gate for Phase 5
        guardrails: Optional safety guardrails
        multipass_memory_crawl: Async function for multipass crawling
        emit_stage_event: Optional callback for stage events
        log: Optional logger

    Returns:
        WeavingContext with dot_plasma, warp_space, shards, and related fields populated.

    Example:
        >>> ctx = await execute_steps_4_6_parallel(
        ...     ctx, cfg, embedder,
        ...     memory=memory_backend,
        ...     emit_stage_event=emit_fn
        ... )
        >>> print(f"Features: {len(ctx.dot_plasma.get('threads', []))} threads")
        >>> print(f"Shards: {len(ctx.shards)}")
    """
    log = log or logger
    parallel_start = time.time()

    # Select embedder based on configuration
    pattern_embedder = select_pattern_embedder(
        cfg, ctx.pattern_spec, linguistic_gate, log
    )
    ctx.pattern_embedder = pattern_embedder

    # Create components
    resonance_shed = create_resonance_shed(
        ctx, cfg, pattern_embedder, linguistic_gate, guardrails, log
    )
    ctx.resonance_shed = resonance_shed

    warp_space = create_warp_space(ctx, embedder, guardrails)
    ctx.warp_space = warp_space

    # Execute parallel steps
    log.info("  [PARALLEL] Executing Steps 4-6 concurrently...")

    try:
        (dot_plasma, t4), (warp_operations, t5), (shards, shard_texts, hits, t6) = await asyncio.gather(
            _step4_feature_extraction(ctx, resonance_shed, emit_stage_event, log),
            _step5_warp_tensioning(ctx, warp_space, emit_stage_event, log),
            _step6_memory_retrieval(
                ctx, memory, retriever, complexity, provenance,
                multipass_memory_crawl, emit_stage_event, log
            ),
            return_exceptions=False
        )

        # Store results in context
        ctx.dot_plasma = dot_plasma
        ctx.warp_operations = warp_operations
        ctx.shards = shards
        ctx.shard_texts = shard_texts
        ctx.hits = hits

        # Record timings
        ctx.record_timing('feature_extraction', t4)
        ctx.record_timing('warp_tensioning', t5)
        ctx.record_timing('retrieval', t6)

        # Calculate parallel execution metrics
        parallel_duration = (time.time() - parallel_start) * 1000
        sequential_duration = t4 + t5 + t6
        speedup = sequential_duration / parallel_duration if parallel_duration > 0 else 1.0

        ctx.record_timing('parallel_execution_wall_time', parallel_duration)
        ctx.stage_timings['parallel_speedup'] = speedup

        log.info(
            f"  [PARALLEL] Steps 4-6 completed in {parallel_duration:.1f}ms "
            f"(sequential would be {sequential_duration:.1f}ms, speedup: {speedup:.2f}x)"
        )

    except Exception as e:
        log.error(f"  [PARALLEL] Parallel execution failed: {e}", exc_info=True)
        ctx.add_error(f"Parallel execution failed: {e}")
        raise

    return ctx


# ============================================================================
# Step 5.5: Warp Space Compute
# ============================================================================

async def execute_step5_5_warp_compute(
    ctx: 'WeavingContext',
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Step 5.5: WarpSpace Compute Operations.

    Performs tensor operations in the continuous manifold:
    - Spectral features computation
    - Attention entropy calculation
    - Context vector generation

    Args:
        ctx: Weaving context with warp_space and dot_plasma
        log: Optional logger

    Returns:
        WeavingContext with warp_compute_results populated.
    """
    log = log or logger
    step_start = ctx.start_timer()

    if ctx.warp_space is None:
        log.warning("  [5.5] Warp Space not initialized, skipping compute")
        return ctx

    try:
        # Get query embedding from DotPlasma
        psi_raw = ctx.dot_plasma.get('psi', []) if ctx.dot_plasma else []
        if isinstance(psi_raw, dict):
            query_embedding = psi_raw[max(psi_raw.keys())]
        else:
            query_embedding = psi_raw

        # Convert to numpy
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Perform continuous tensor operations
        warp_compute_results = ctx.warp_space.compute(
            query_embedding=query_embedding,
            compute_spectral=True
        )
        ctx.warp_compute_results = warp_compute_results

        # Record warp operation
        ctx.add_warp_operation(
            operation="compute",
            details={
                'attention_entropy': warp_compute_results.get('attention_entropy', 0.0),
                'spectral_computed': warp_compute_results['metadata']['spectral_computed']
            }
        )

        duration = ctx.record_timing_since('warp_compute', step_start)

        log.info(
            f"  [5.5] Warp Space compute: "
            f"attention_entropy={warp_compute_results.get('attention_entropy', 0.0):.3f}, "
            f"spectral={warp_compute_results['metadata']['spectral_computed']} "
            f"({duration:.1f}ms)"
        )

    except Exception as e:
        log.warning(f"  [5.5] Warp Space compute failed: {e}. Continuing without warp features.")
        ctx.warp_compute_results = None
        ctx.add_warning(f"Warp compute failed: {e}")
        ctx.record_timing('warp_compute', 0.0)

    return ctx


# ============================================================================
# Step 6.5: Beta Wave Context Packing
# ============================================================================

async def execute_step6_5_beta_wave_packing(
    ctx: 'WeavingContext',
    cfg: 'Config',
    memory: Optional[Any] = None,
    log: Optional[logging.Logger] = None,
) -> 'WeavingContext':
    """
    Step 6.5: Beta Wave Context Packing (optional).

    Physics-based context optimization using activation spreading:
    - Uses spring dynamics for activation propagation
    - Achieves 50% token reduction with <1ms overhead
    - Requires MultiWaveMemoryEngine with spring_engine

    Args:
        ctx: Weaving context with dot_plasma and current query
        cfg: Configuration with packing settings
        memory: Memory backend with spring_engine
        log: Optional logger

    Returns:
        WeavingContext with packed_context in metadata.
    """
    log = log or logger

    # Check if beta wave packing is enabled and available
    if not cfg.enable_beta_wave_packing:
        log.debug("  [6.5] Beta wave packing: DISABLED (config flag off)")
        ctx.packed_context = None
        return ctx

    if not memory or not hasattr(memory, 'spring_engine'):
        log.info("  [6.5] Beta wave packing: DISABLED (memory backend lacks spring_engine)")
        ctx.packed_context = None
        return ctx

    step_start = ctx.start_timer()

    try:
        from HoloLoom.awareness.beta_wave_packer import (
            BetaWaveContextPacker, TokenBudget
        )

        # Create token budget from config
        packing_budget = TokenBudget(
            total=cfg.packing_token_budget,
            reserved_for_query=cfg.packing_query_reserve,
            reserved_for_response=cfg.packing_response_reserve
        )

        # Create beta wave context packer
        packer = BetaWaveContextPacker(
            spring_engine=memory.spring_engine,
            token_budget=packing_budget,
            activation_threshold=cfg.packing_activation_threshold,
            compression_threshold=cfg.packing_compression_threshold
        )

        # Get query embedding from DotPlasma
        psi_raw = ctx.dot_plasma.get('psi', []) if ctx.dot_plasma else []
        if isinstance(psi_raw, dict):
            query_embedding = psi_raw[max(psi_raw.keys())]
        else:
            query_embedding = psi_raw

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype=np.float32)

        # Pack context using physics-based activation spreading
        packed = await packer.pack_context(
            query_text=ctx.current_query_text,
            query_embedding=query_embedding,
            awareness_context=None,
            top_k=len(ctx.shards)
        )

        ctx.packed_context = packed

        duration = ctx.record_timing_since('context_packing', step_start)

        log.info(
            f"  [6.5] Beta wave packing: {packed.elements_included} included, "
            f"{packed.elements_compressed} compressed, "
            f"{packed.elements_excluded} excluded "
            f"({packed.total_tokens}/{packing_budget.available_for_context} tokens, "
            f"avg_activation={packed.avg_activation:.3f}, {duration:.1f}ms)"
        )

    except Exception as e:
        log.warning(f"  [6.5] Beta wave packing failed: {e}. Falling back to raw shards.")
        ctx.packed_context = None
        ctx.add_warning(f"Beta wave packing failed: {e}")

    return ctx


__all__ = [
    # Component creation
    'create_resonance_shed',
    'create_warp_space',
    'select_pattern_embedder',

    # Main parallel execution
    'execute_steps_4_6_parallel',

    # Post-parallel steps
    'execute_step5_5_warp_compute',
    'execute_step6_5_beta_wave_packing',
]
