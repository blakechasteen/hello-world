"""
Jenny Orchestrator Integration - Elegance Pass (December 2025)

Extracted from WeavingOrchestrator for clean separation of concerns.
Handles Jenny Generative UI panel detection and context building.
"""

from .panel_detection import (
    # Constants
    JENNY_CONFIDENCE_THRESHOLD,
    JENNY_DURATION_THRESHOLD_MS,
    JENNY_STAGES_THRESHOLD,
    JENNY_THREADS_THRESHOLD,
    build_jenny_panel_context,
    # Functions
    classify_query_type,
    detect_jenny_panel_type,
    detect_panel_type_heuristic,
    get_panel_type_candidates,
)

__all__ = [
    # Constants
    'JENNY_CONFIDENCE_THRESHOLD',
    'JENNY_THREADS_THRESHOLD',
    'JENNY_STAGES_THRESHOLD',
    'JENNY_DURATION_THRESHOLD_MS',
    # Functions
    'classify_query_type',
    'get_panel_type_candidates',
    'detect_panel_type_heuristic',
    'detect_jenny_panel_type',
    'build_jenny_panel_context',
]
