#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config and Memory Initialization
=================================

Initialize memory configuration, pattern card, and protocol architecture.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 401-513 (~113 lines)

This module handles:
- Memory source validation (memory/yarn_graph/shards)
- mythRL protocol architecture setup
- Complexity detection thresholds configuration
- Multipass memory crawling configuration
- Safety guardrails initialization
- Pattern card determination
- Lifecycle management setup

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
import asyncio
import os
import warnings
from typing import Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.weaving_orchestrator import WeavingOrchestrator

from hololoom.config import ExecutionMode
from hololoom.loom.command import PatternCard
from hololoom.protocols import ComplexityLevel
from hololoom.alignment import create_guardrails
from hololoom.alignment.audit_trail import AuditTrail


logger = logging.getLogger(__name__)


def initialize_config_and_memory(
    orchestrator: 'WeavingOrchestrator',
    memory,
    yarn_graph,
    shards,
    pattern_preference,
    enable_complexity_auto_detect: bool
) -> None:
    """
    Initialize memory configuration, pattern card, and protocol architecture.

    Validates memory sources, sets up mythRL protocols, and configures
    complexity detection thresholds.

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize
        memory: Backend memory store (persistent)
        yarn_graph: User-provided Yarn Graph (KG instance)
        shards: Static memory shards (backward compatibility)
        pattern_preference: User's preferred pattern card
        enable_complexity_auto_detect: Enable automatic complexity detection

    Raises:
        ValueError: If no memory source is provided

    Example:
        >>> initialize_config_and_memory(
        ...     orchestrator=shuttle,
        ...     memory=None,
        ...     yarn_graph=kg_instance,
        ...     shards=None,
        ...     pattern_preference=PatternCard.FAST,
        ...     enable_complexity_auto_detect=True
        ... )

    Side Effects:
        Sets the following attributes on orchestrator:
        - memory: Backend memory store
        - yarn_graph_param: User-provided Yarn Graph
        - shards: Static shards list
        - enable_complexity_auto_detect: Auto-detection flag
        - _protocols: Protocol implementations dict
        - _complexity_thresholds: Thresholds for complexity detection
        - _crawl_config: Multipass memory crawling configuration
        - guardrails: SafetyGuardrails instance
        - audit_trail: AuditTrail instance for decision logging
        - default_pattern: Default pattern card
        - _background_tasks: Background task list
        - _bg_lock: Background task lock
        - _closed: Lifecycle flag
    """
    # Validate memory configuration
    if memory is None and yarn_graph is None and shards is None:
        raise ValueError("Either 'memory', 'yarn_graph', or 'shards' must be provided")

    # Deprecation warning for shards parameter
    if shards is not None and yarn_graph is None and memory is None:
        warnings.warn(
            "Using 'shards' parameter is deprecated. "
            "Please use 'yarn_graph' (KG instance) for full metaphor support. "
            "Example: yarn_graph=KG() with memories added via yarn_graph.store().",
            DeprecationWarning,
            stacklevel=2
        )

    orchestrator.memory = memory  # Backend memory store
    orchestrator.yarn_graph_param = yarn_graph  # User-provided Yarn Graph (KG)
    orchestrator.shards = shards or []  # Static shards (backward compatibility)

    # mythRL Protocol-based architecture
    orchestrator.enable_complexity_auto_detect = enable_complexity_auto_detect
    orchestrator._protocols: Dict[str, Any] = {}  # Registered protocol implementations
    orchestrator._complexity_thresholds = {
        # Word count thresholds
        'lite_max_words': 2,      # Up to 2 words = LITE (greetings, simple commands)
        'fast_max_words': 20,     # 3-20 words = FAST (standard questions)
        'full_max_words': 50,     # 21-50 words = FULL (detailed queries)

        # Intent patterns for sophisticated detection
        'greeting_patterns': ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye', 'goodbye'],
        'simple_commands': ['show', 'list', 'get', 'find', 'open'],
        'question_words': ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom'],
        'knowledge_verbs': ['explain', 'describe', 'tell', 'define', 'clarify', 'elaborate'],
        'analysis_verbs': ['analyze', 'compare', 'evaluate', 'assess', 'investigate', 'examine'],
        'research_keywords': ['research', 'deep', 'comprehensive', 'detailed', 'thorough', 'in-depth', 'extensive']
    }

    # Multipass Memory Crawling Configuration
    # Recursive gated retrieval with Matryoshka importance gating
    orchestrator._crawl_config = {
        ComplexityLevel.LITE: {
            'passes': 1,
            'thresholds': [0.7],           # Single pass, high threshold
            'limits': [5],                  # 5 items max
            'graph_expansion': False        # No graph traversal
        },
        ComplexityLevel.FAST: {
            'passes': 2,
            'thresholds': [0.6, 0.75],     # Broad → focused
            'limits': [8, 12],              # 8 initial, 12 expansion
            'graph_expansion': True         # Light graph traversal
        },
        ComplexityLevel.FULL: {
            'passes': 3,
            'thresholds': [0.6, 0.75, 0.85],  # Progressive gating
            'limits': [12, 20, 15],           # Expanding search space
            'graph_expansion': True           # Full graph traversal
        },
        ComplexityLevel.RESEARCH: {
            'passes': 4,
            'thresholds': [0.5, 0.65, 0.8, 0.9],  # Maximum depth
            'limits': [20, 30, 25, 15],            # Deep exploration
            'graph_expansion': True                 # Aggressive graph traversal
        }
    }
    logger.info(f"mythRL protocol system enabled (auto_detect={enable_complexity_auto_detect})")

    # Create a single shared SafetyGuardrails instance used across components
    try:
        # Check if guardrails should be disabled via config or environment variable
        testing_mode = (not getattr(orchestrator.cfg, 'enable_safety_guardrails', True) or
                       os.getenv("DISABLE_GUARDRAILS"))
        orchestrator.guardrails = create_guardrails(testing_mode=testing_mode)
        if testing_mode:
            logger.info("Shared SafetyGuardrails created in testing mode (approval bypassed)")
        else:
            logger.info("Shared SafetyGuardrails instance created and will be threaded to components")
    except Exception as e:
        logger.warning(f"Failed to initialize shared guardrails: {e}. Falling back to local guardrails where needed.")
        orchestrator.guardrails = None

    # Create AuditTrail for decision logging and provenance tracking (December 2025)
    try:
        persist_path = getattr(orchestrator.cfg, 'audit_trail_path', None)
        orchestrator.audit_trail = AuditTrail(
            persist_path=persist_path,
            auto_flush=True  # Flush logs immediately for durability
        )
        logger.info(f"AuditTrail initialized (persist_path={persist_path})")
    except Exception as e:
        logger.warning(f"Failed to initialize AuditTrail: {e}. Decision logging will be disabled.")
        orchestrator.audit_trail = None

    # Determine pattern card from config or preference
    if pattern_preference:
        orchestrator.default_pattern = pattern_preference
    elif orchestrator.cfg.mode == ExecutionMode.BARE:
        orchestrator.default_pattern = PatternCard.BARE
    elif orchestrator.cfg.mode == ExecutionMode.FAST:
        orchestrator.default_pattern = PatternCard.FAST
    else:
        orchestrator.default_pattern = PatternCard.FUSED

    logger.info(f"Initializing WeavingOrchestrator with pattern: {orchestrator.default_pattern.value}")

    # Lifecycle management
    orchestrator._background_tasks: List[asyncio.Task] = []
    orchestrator._bg_lock = asyncio.Lock()  # Protect concurrent access to _background_tasks
    orchestrator._closed = False
