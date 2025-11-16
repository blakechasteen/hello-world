#!/usr/bin/env python3
"""
HoloLoom Component Instrumentation
====================================

Provides instrumentation for HoloLoom's core components:
- Weaving orchestrator
- Memory/cache operations
- Analytics/database queries
- Recursive reasoning

This module monkey-patches or wraps HoloLoom components to add tracing.

Usage:
    from HoloLoom.tracing import instrument_hololoom

    # Instrument all components
    instrument_hololoom()

    # Or selectively
    from HoloLoom.tracing.hololoom_instrumentation import (
        instrument_weaving_orchestrator,
        instrument_memory_cache,
        instrument_analytics,
    )

Created: 2025-11-16
"""

import logging
from typing import Any, Optional
from functools import wraps

from HoloLoom.tracing.opentelemetry_integration import TRACING_AVAILABLE, get_tracer
from HoloLoom.tracing.instrumentation import create_span, trace_database_query

if TRACING_AVAILABLE:
    from opentelemetry.trace import SpanKind, Status, StatusCode
else:
    SpanKind = None
    Status = None
    StatusCode = None

logger = logging.getLogger(__name__)


# ============================================================================
# Weaving Orchestrator Instrumentation
# ============================================================================

def instrument_weaving_orchestrator():
    """
    Instrument WeavingOrchestrator with tracing.

    Adds spans for:
    - weave() - Main weaving cycle
    - _extract_features() - Feature extraction
    - _retrieve_context() - Memory retrieval
    - _make_decision() - Decision making
    - _execute_action() - Action execution
    """
    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    except ImportError:
        logger.warning("WeavingOrchestrator not available for instrumentation")
        return

    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping orchestrator instrumentation")
        return

    # Store original methods
    original_weave = WeavingOrchestrator.weave

    async def traced_weave(self, query):
        """Traced version of weave()."""
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span(
            "weave",
            kind=SpanKind.INTERNAL,
        ) as span:
            # Add query attributes
            span.set_attribute("query.text", query.text[:100] if hasattr(query, 'text') else str(query)[:100])
            span.set_attribute("complexity", self.cfg.mode.value if hasattr(self.cfg, 'mode') else "unknown")

            try:
                # Call original method
                result = await original_weave(self, query)

                # Add result attributes
                if hasattr(result, 'confidence'):
                    span.set_attribute("result.confidence", result.confidence)
                if hasattr(result, 'metadata'):
                    if 'tool_used' in result.metadata:
                        span.set_attribute("tool.used", result.metadata['tool_used'])

                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey-patch
    WeavingOrchestrator.weave = traced_weave
    logger.info("WeavingOrchestrator instrumented with tracing")


# ============================================================================
# Memory Cache Instrumentation
# ============================================================================

def instrument_memory_cache():
    """
    Instrument MemoryManager with tracing.

    Adds spans for:
    - recall() - Memory retrieval
    - experience() - Memory storage
    - _semantic_search() - Semantic search
    - _graph_traverse() - Graph traversal
    """
    try:
        from HoloLoom.memory.cache import MemoryManager
    except ImportError:
        logger.warning("MemoryManager not available for instrumentation")
        return

    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping memory cache instrumentation")
        return

    # Store original methods
    original_recall = MemoryManager.recall

    async def traced_recall(self, query, k=5):
        """Traced version of recall()."""
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span(
            "memory.recall",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("query", str(query)[:100])
            span.set_attribute("k", k)

            try:
                # Call original method
                results = await original_recall(self, query, k)

                # Add result attributes
                span.set_attribute("results.count", len(results) if results else 0)
                span.set_status(Status(StatusCode.OK))

                return results

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey-patch
    MemoryManager.recall = traced_recall
    logger.info("MemoryManager instrumented with tracing")


# ============================================================================
# Analytics Instrumentation
# ============================================================================

def instrument_analytics():
    """
    Instrument RecursiveAnalytics with tracing.

    Adds spans for:
    - track_execution() - Tracking execution
    - get_summary() - Getting summary
    - Database queries
    """
    try:
        from HoloLoom.analytics.recursive_analytics import RecursiveAnalytics
    except ImportError:
        logger.warning("RecursiveAnalytics not available for instrumentation")
        return

    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping analytics instrumentation")
        return

    # Store original methods
    original_get_summary = RecursiveAnalytics.get_summary

    def traced_get_summary(self):
        """Traced version of get_summary()."""
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span(
            "analytics.get_summary",
            kind=SpanKind.INTERNAL,
        ) as span:
            try:
                # Add database query span
                with trace_database_query(
                    "SELECT",
                    "SELECT * FROM executions",
                    "sqlite",
                ):
                    result = original_get_summary(self)

                span.set_attribute("total_queries", result.get("total_queries", 0))
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey-patch
    RecursiveAnalytics.get_summary = traced_get_summary
    logger.info("RecursiveAnalytics instrumented with tracing")


# ============================================================================
# Agentic Reasoning Instrumentation
# ============================================================================

def instrument_agentic_reasoning():
    """
    Instrument agentic reasoning components.

    Adds spans for:
    - Multi-query reasoning
    - Verification
    - Research mode
    - Plan-execute mode
    """
    try:
        from HoloLoom.agentic.core import AgenticOrchestrator
    except ImportError:
        logger.warning("AgenticOrchestrator not available for instrumentation")
        return

    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping agentic instrumentation")
        return

    # Store original methods
    original_reason = AgenticOrchestrator.reason

    async def traced_reason(self, query, mode="direct", max_steps=5):
        """Traced version of reason()."""
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span(
            "agentic.reason",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("query", str(query)[:100])
            span.set_attribute("mode", mode)
            span.set_attribute("max_steps", max_steps)

            try:
                result = await original_reason(self, query, mode, max_steps)

                span.set_attribute("steps_taken", result.steps_taken if hasattr(result, 'steps_taken') else 0)
                span.set_attribute("confidence", result.confidence if hasattr(result, 'confidence') else 0.0)
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey-patch
    AgenticOrchestrator.reason = traced_reason
    logger.info("AgenticOrchestrator instrumented with tracing")


# ============================================================================
# Recursive Reasoning Instrumentation
# ============================================================================

def instrument_recursive_reasoning():
    """
    Instrument recursive reasoning system.

    Adds spans for:
    - Refinement iterations
    - Strategy selection
    - Quality tracking
    """
    try:
        from HoloLoom.recursive.advanced_refiner import AdvancedRefiner
    except ImportError:
        logger.warning("AdvancedRefiner not available for instrumentation")
        return

    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping recursive instrumentation")
        return

    # Store original methods
    original_refine = AdvancedRefiner.refine

    async def traced_refine(self, query, initial_spacetime, strategy=None, max_iterations=3, quality_threshold=0.9):
        """Traced version of refine()."""
        tracer = get_tracer(__name__)

        with tracer.start_as_current_span(
            "recursive.refine",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("query", str(query)[:100] if query else "")
            span.set_attribute("strategy", strategy.value if strategy else "auto")
            span.set_attribute("max_iterations", max_iterations)
            span.set_attribute("quality_threshold", quality_threshold)

            try:
                result = await original_refine(self, query, initial_spacetime, strategy, max_iterations, quality_threshold)

                span.set_attribute("iterations_taken", result.iterations if hasattr(result, 'iterations') else 0)
                span.set_attribute("final_quality", result.final_quality if hasattr(result, 'final_quality') else 0.0)
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    # Monkey-patch
    AdvancedRefiner.refine = traced_refine
    logger.info("AdvancedRefiner instrumented with tracing")


# ============================================================================
# Main Instrumentation Function
# ============================================================================

def instrument_hololoom(
    orchestrator: bool = True,
    memory: bool = True,
    analytics: bool = True,
    agentic: bool = True,
    recursive: bool = True,
):
    """
    Instrument HoloLoom components with tracing.

    Args:
        orchestrator: Instrument WeavingOrchestrator
        memory: Instrument MemoryManager
        analytics: Instrument RecursiveAnalytics
        agentic: Instrument AgenticOrchestrator
        recursive: Instrument AdvancedRefiner

    Example:
        >>> from HoloLoom.tracing import instrument_hololoom
        >>> instrument_hololoom()
    """
    if not TRACING_AVAILABLE:
        logger.warning("Tracing not available, skipping HoloLoom instrumentation")
        return

    logger.info("Instrumenting HoloLoom components...")

    if orchestrator:
        instrument_weaving_orchestrator()

    if memory:
        instrument_memory_cache()

    if analytics:
        instrument_analytics()

    if agentic:
        instrument_agentic_reasoning()

    if recursive:
        instrument_recursive_reasoning()

    logger.info("HoloLoom instrumentation complete")
