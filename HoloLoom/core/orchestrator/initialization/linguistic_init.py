#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linguistic Gate Initialization
===============================

Initialize Phase 5 Linguistic Matryoshka Gate.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 948-1003 (~55 lines)

This module handles:
- Linguistic gate configuration from config
- Mode mapping (disabled/prefilter/embedding/both)
- Compositional cache setup
- X-bar theory phrase chunking

This provides 10-300× speedup through:
1. Universal Grammar phrase chunking (X-bar theory)
2. 3-tier compositional cache (parse/merge/semantic)
3. Progressive linguistic filtering

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


def initialize_linguistic_gate(orchestrator: 'WeavingOrchestrator') -> None:
    """
    Initialize Phase 5 Linguistic Matryoshka Gate.

    This provides 10-300× speedup through:
    1. Universal Grammar phrase chunking (X-bar theory)
    2. 3-tier compositional cache (parse/merge/semantic)
    3. Progressive linguistic filtering

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize

    Example:
        >>> initialize_linguistic_gate(orchestrator)

    Side Effects:
        Sets orchestrator.linguistic_gate to LinguisticMatryoshkaGate instance
        (or None on failure/unavailable).

        The gate is configured based on orchestrator.cfg:
        - linguistic_mode: Filter mode (disabled/prefilter/embedding/both)
        - use_compositional_cache: Enable compositional caching
        - parse_cache_size: Parse cache size
        - merge_cache_size: Merge cache size
        - linguistic_weight: Weight for linguistic features
        - prefilter_similarity_threshold: Threshold for pre-filtering
        - prefilter_keep_ratio: Ratio of candidates to keep

    Note:
        Falls back gracefully to standard matryoshka gate if initialization fails.
        This is expected behavior when linguistic modules are unavailable.
    """
    try:
        from HoloLoom.core.embedding.linguistic_matryoshka_gate import (
            LinguisticMatryoshkaGate,
            LinguisticGateConfig,
            LinguisticFilterMode
        )

        logger.info("Initializing Phase 5 Linguistic Matryoshka Gate...")

        # Map config string to enum
        mode_map = {
            'disabled': LinguisticFilterMode.DISABLED,
            'prefilter': LinguisticFilterMode.PREFILTER,
            'embedding': LinguisticFilterMode.EMBEDDING,
            'both': LinguisticFilterMode.BOTH
        }
        linguistic_mode = mode_map.get(orchestrator.cfg.linguistic_mode, LinguisticFilterMode.DISABLED)

        # Create configuration
        gate_config = LinguisticGateConfig(
            linguistic_mode=linguistic_mode,
            use_compositional_cache=orchestrator.cfg.use_compositional_cache,
            parse_cache_size=orchestrator.cfg.parse_cache_size,
            merge_cache_size=orchestrator.cfg.merge_cache_size,
            linguistic_weight=orchestrator.cfg.linguistic_weight,
            prefilter_similarity_threshold=orchestrator.cfg.prefilter_similarity_threshold,
            prefilter_keep_ratio=orchestrator.cfg.prefilter_keep_ratio
        )

        # Create linguistic gate
        orchestrator.linguistic_gate = LinguisticMatryoshkaGate(
            embedder=orchestrator.embedder,
            config=gate_config
        )

        logger.info(
            f"  Phase 5 enabled: mode={linguistic_mode.value}, "
            f"cache={orchestrator.cfg.use_compositional_cache}, "
            f"parse_cache={orchestrator.cfg.parse_cache_size}, "
            f"merge_cache={orchestrator.cfg.merge_cache_size}"
        )

    except Exception as e:
        logger.warning(f"Failed to initialize linguistic gate: {e}")
        logger.warning("Falling back to standard matryoshka gate")
        orchestrator.linguistic_gate = None
