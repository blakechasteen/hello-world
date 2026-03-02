"""
Jenny Panel Detection - Extracted from WeavingOrchestrator

Handles panel type detection and context building for Jenny Generative UI.
Supports both MRF-learned Thompson Sampling selection and heuristic fallback.

Status: Production Ready (December 2025 - Elegance Pass)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hololoom.protocols.types import Query
    from hololoom.fabric.spacetime import Spacetime
    from hololoom.loom.command import PatternSpec
    from hololoom.protocols import ComplexityLevel

logger = logging.getLogger(__name__)

# ============================================================================
# Jenny Panel Type Detection Thresholds
# ============================================================================
JENNY_CONFIDENCE_THRESHOLD = 0.7   # Below this -> CONFIDENCE panel
JENNY_THREADS_THRESHOLD = 2        # Above this -> GRAPH panel
JENNY_STAGES_THRESHOLD = 3         # Above this -> TIMELINE panel
JENNY_DURATION_THRESHOLD_MS = 100  # Above this -> METRIC panel


def classify_query_type(spacetime: 'Spacetime') -> str:
    """
    Classify query type for Thompson Sampling panel selection.

    Simple heuristic classification:
    - factual: Direct questions, lookups (what, who, when, where)
    - procedural: How-to questions, step-by-step (how, steps, process)
    - analytical: Analysis, comparison, evaluation (why, compare, analyze)
    - exploratory: Open-ended research, discovery

    Args:
        spacetime: Spacetime result from weaving

    Returns:
        Query type string for Thompson Sampling lookup
    """
    query_text = (getattr(spacetime, 'query_text', '') or '').lower()

    # Procedural indicators
    if any(kw in query_text for kw in ['how to', 'steps', 'process', 'implement', 'create']):
        return 'procedural'

    # Analytical indicators
    if any(kw in query_text for kw in ['why', 'compare', 'analyze', 'evaluate', 'tradeoff']):
        return 'analytical'

    # Exploratory indicators
    if any(kw in query_text for kw in ['explore', 'research', 'discover', 'comprehensive']):
        return 'exploratory'

    # Factual (default) - direct questions
    return 'factual'


def get_panel_type_candidates(spacetime: 'Spacetime') -> List:
    """
    Get candidate panel types based on response content analysis.

    Filters panel types to those that make sense for the content,
    reducing the search space for Thompson Sampling.

    Args:
        spacetime: Spacetime result from weaving

    Returns:
        List of appropriate PanelTypeJenny candidates
    """
    from hololoom.visualization.jenny_spec import PanelTypeJenny

    response = spacetime.response or ""
    trace = spacetime.trace
    candidates = []

    # Always include TEXT as a baseline
    candidates.append(PanelTypeJenny.TEXT)

    # Code panel if response has code blocks
    if "```" in response:
        candidates.append(PanelTypeJenny.CODE)

    # Graph panel if multiple threads/entities
    if trace and len(getattr(trace, 'threads_activated', [])) > 1:
        candidates.append(PanelTypeJenny.GRAPH)

    # Confidence panel if low confidence
    if spacetime.confidence < 0.8:
        candidates.append(PanelTypeJenny.CONFIDENCE)

    # Timeline if has timing data
    if trace and len(getattr(trace, 'stage_durations', {})) > 1:
        candidates.append(PanelTypeJenny.TIMELINE)

    # Metric for performance data
    if trace and getattr(trace, 'duration_ms', 0) > 50:
        candidates.append(PanelTypeJenny.METRIC)

    # Reasoning for complex multi-step
    if spacetime.confidence > 0.6 and len(response) > 500:
        candidates.append(PanelTypeJenny.REASONING)

    return candidates


def detect_panel_type_heuristic(spacetime: 'Spacetime'):
    """
    Heuristic-based panel type detection (fallback for MRF).

    Original detection logic for graceful degradation.

    Args:
        spacetime: Spacetime result from weaving

    Returns:
        PanelTypeJenny enum value
    """
    from hololoom.visualization.jenny_spec import PanelTypeJenny

    response = spacetime.response or ""
    trace = spacetime.trace

    # Code detection (has code blocks)
    if "```" in response:
        return PanelTypeJenny.CODE

    # Graph panels for rich context (multiple threads activated)
    if trace and len(getattr(trace, 'threads_activated', [])) > JENNY_THREADS_THRESHOLD:
        return PanelTypeJenny.GRAPH

    # Confidence panels for low-confidence results
    if spacetime.confidence < JENNY_CONFIDENCE_THRESHOLD:
        return PanelTypeJenny.CONFIDENCE

    # Timeline for stage timing data
    if trace and len(getattr(trace, 'stage_durations', {})) > JENNY_STAGES_THRESHOLD:
        return PanelTypeJenny.TIMELINE

    # Metric for performance-critical or numerical results
    if trace and getattr(trace, 'duration_ms', 0) > JENNY_DURATION_THRESHOLD_MS:
        return PanelTypeJenny.METRIC

    # Default to TEXT
    return PanelTypeJenny.TEXT


def detect_jenny_panel_type(
    spacetime: 'Spacetime',
    jenny_mrf_compiler=None,
    jenny_learner=None,
):
    """
    Detect optimal Jenny panel type from Spacetime content.

    Phase 2.1-2.2: Uses MRF-learned Thompson Sampling selection when available,
    with heuristic fallback for graceful degradation.

    Thresholds (for heuristic fallback):
        JENNY_CONFIDENCE_THRESHOLD (0.7): Below triggers CONFIDENCE panel
        JENNY_THREADS_THRESHOLD (2): Above triggers GRAPH panel
        JENNY_STAGES_THRESHOLD (3): Above triggers TIMELINE panel
        JENNY_DURATION_THRESHOLD_MS (100): Above triggers METRIC panel

    Args:
        spacetime: Spacetime result from weaving
        jenny_mrf_compiler: Optional MRF compiler for learned selection
        jenny_learner: Optional Thompson Sampling learner

    Returns:
        PanelTypeJenny enum value
    """
    from hololoom.visualization.jenny_spec import PanelTypeJenny

    # Phase 2.1-2.2: Try MRF-learned selection first
    if jenny_mrf_compiler and jenny_learner:
        try:
            query_type = classify_query_type(spacetime)

            # Get candidate panel types based on content analysis
            candidates = get_panel_type_candidates(spacetime)

            # Use Thompson Sampling to select from candidates
            learned_selection = jenny_learner.select(
                query_type=query_type,
                candidates=candidates,
                exploration_bonus=0.1  # Small exploration bonus
            )

            logger.debug(
                f"MRF panel selection: query_type={query_type}, "
                f"candidates={[c.value for c in candidates]}, selected={learned_selection.value}"
            )
            return learned_selection

        except Exception as e:
            logger.warning(f"MRF panel selection failed, falling back to heuristics: {e}")

    # Fallback to heuristic detection
    return detect_panel_type_heuristic(spacetime)


def build_jenny_panel_context(
    query: 'Query',
    spacetime: 'Spacetime',
    pattern_spec: Optional['PatternSpec'],
    complexity: Optional['ComplexityLevel'],
) -> Dict[str, Any]:
    """
    Build context dict for Jenny panel generation.

    Args:
        query: Original query
        spacetime: Weaving result
        pattern_spec: Pattern used (BARE/FAST/FUSED)
        complexity: Complexity level

    Returns:
        Context dict for JennyRuntime.ask()
    """
    return {
        'session_id': spacetime.metadata.get('session_id', 'default'),
        'spacetime_id': spacetime.metadata.get('spacetime_id', f"st_{int(time.time() * 1000)}"),
        'pattern': pattern_spec.name if pattern_spec else 'unknown',
        'complexity': complexity.name if complexity else 'unknown',
        'response': spacetime.response,
        'confidence': spacetime.confidence,
        'tool_used': spacetime.tool_used,
        'trace': spacetime.trace.to_dict() if hasattr(spacetime.trace, 'to_dict') else {},
        'sources': spacetime.sources_used if hasattr(spacetime, 'sources_used') else [],
    }
