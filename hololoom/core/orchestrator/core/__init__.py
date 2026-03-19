#!/usr/bin/env python3
"""
HoloLoom Orchestrator Core Logic Module
========================================

Core logic functions for the WeavingOrchestrator.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: Various methods throughout file (~351 lines total)

This module provides clean, single-responsibility functions for:
- Complexity detection and assessment
- Provenance trace creation
- Metrics collection (reflection, recursive learning, system-wide)
- Background task management
- Statistical mechanics consolidation

Public API:
    assess_complexity_level: Assess query complexity (LITE/FAST/FULL/RESEARCH)
    create_provenance_trace: Create new provenance trace
    get_reflection_metrics: Get reflection buffer metrics
    get_recursive_learning_stats: Get recursive learning statistics
    get_metrics: Get comprehensive system metrics
    start_background_consolidation: Start statistical mechanics consolidation
    spawn_background_task: Spawn managed background task

Created: 2025-11-22 (Elegance Pass Refactoring - Phase 3)
"""

from .background_tasks import (
    spawn_background_task,
    start_background_consolidation,
)
from .complexity_detection import (
    assess_complexity_level,
    create_provenance_trace,
)
from .metrics_collection import (
    get_metrics,
    get_recursive_learning_stats,
    get_reflection_metrics,
)
from .stat_mech_integration import (
    background_consolidation_loop,
    macrostates_to_shards,
    shards_to_microstates,
)

__all__ = [
    'assess_complexity_level',
    'create_provenance_trace',
    'get_reflection_metrics',
    'get_recursive_learning_stats',
    'get_metrics',
    'start_background_consolidation',
    'spawn_background_task',
    'shards_to_microstates',
    'macrostates_to_shards',
    'background_consolidation_loop',
]
