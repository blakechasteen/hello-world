"""
Jenny Compiler - Spacetime → JennySpec[] Compilation
=====================================================
Compiles HoloLoom Spacetime output into renderable UI specifications.

Philosophy:
> "The compiler is the brain that decides what to show."
>
> Query intent determines panel types.
> Response content determines panel data.
> Trace metadata informs debugging panels.

The compiler analyzes Spacetime and generates appropriate JennySpec panels
based on query characteristics, response content, and trace data.

Compilation Strategies:
- MINIMAL: Essential panels only (1-2 panels)
- BALANCED: Standard panel suite (2-4 panels)
- COMPREHENSIVE: Full panel suite (4-6 panels)
- AUTO: Auto-detect based on query complexity

This module provides a rule-based compiler implementation.
Future versions will support LLM-based compilation.

References:
- jenny_spec.py (JennySpec dataclass)
- protocols/jenny.py (JennyCompilerProtocol)
- fabric/spacetime.py (Spacetime input)

Author: HoloLoom Team
Date: 2025-12-01 (Jenny MVP Week 1)
"""

import logging
import re
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime

from .jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LifecycleStage,
    BindingMode,
    LayoutHint,
    create_action,
    get_default_actions,
)
from HoloLoom.protocols.jenny import CompilationStrategy, RenderTarget

# Try to import Spacetime
try:
    from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
except ImportError:
    Spacetime = Any  # type: ignore
    WeavingTrace = Any  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================================
# Query Analysis
# ============================================================================

@dataclass
class QueryAnalysis:
    """
    Analysis of a query for determining panel strategy.
    """
    query_type: str  # factual, procedural, analytical, creative, debug
    complexity: str  # simple, moderate, complex
    has_graph_context: bool
    has_sources: bool
    has_reasoning: bool
    has_errors: bool
    has_metrics: bool
    confidence_level: str  # low, medium, high
    keywords: List[str] = field(default_factory=list)


def analyze_query(spacetime: Spacetime) -> QueryAnalysis:
    """
    Analyze a Spacetime to determine query characteristics.

    Args:
        spacetime: The woven output to analyze

    Returns:
        QueryAnalysis with detected characteristics
    """
    query = spacetime.query_text.lower()
    trace = spacetime.trace

    # Detect query type
    query_type = "factual"  # default
    if any(word in query for word in ["how to", "steps", "guide", "tutorial", "implement"]):
        query_type = "procedural"
    elif any(word in query for word in ["compare", "versus", "vs", "difference", "tradeoff", "analyze"]):
        query_type = "analytical"
    elif any(word in query for word in ["create", "write", "generate", "design", "build"]):
        query_type = "creative"
    elif any(word in query for word in ["debug", "error", "trace", "why did", "what happened"]):
        query_type = "debug"

    # Detect complexity
    word_count = len(query.split())
    complexity = "simple"
    if word_count > 15 or query_type in ["analytical", "debug"]:
        complexity = "complex"
    elif word_count > 8:
        complexity = "moderate"

    # Detect content characteristics
    has_graph_context = len(trace.threads_activated) > 2 if trace else False
    has_sources = len(spacetime.sources_used) > 0
    has_reasoning = len(trace.motifs_detected) > 2 if trace else False
    has_errors = len(trace.errors) > 0 if trace else False
    has_metrics = trace.duration_ms is not None if trace else False

    # Confidence level
    confidence_level = "low"
    if spacetime.confidence >= 0.8:
        confidence_level = "high"
    elif spacetime.confidence >= 0.6:
        confidence_level = "medium"

    # Extract keywords (simple regex for now)
    keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', spacetime.query_text)

    return QueryAnalysis(
        query_type=query_type,
        complexity=complexity,
        has_graph_context=has_graph_context,
        has_sources=has_sources,
        has_reasoning=has_reasoning,
        has_errors=has_errors,
        has_metrics=has_metrics,
        confidence_level=confidence_level,
        keywords=keywords[:5],  # Top 5 keywords
    )


# ============================================================================
# Panel Generators
# ============================================================================

def generate_text_panel(
    spacetime: Spacetime,
    priority: int = 0,
    size: PanelSizeJenny = PanelSizeJenny.LARGE
) -> JennySpec:
    """Generate a text panel for the main response."""
    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.TEXT,
        title="Response",
        subtitle=spacetime.query_text[:50] + "..." if len(spacetime.query_text) > 50 else spacetime.query_text,
        content={
            "text": spacetime.response,
            "format": "markdown",
        },
        size=size,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.TEXT),
    )


def generate_confidence_panel(
    spacetime: Spacetime,
    priority: int = 1,
) -> JennySpec:
    """Generate a confidence gauge panel."""
    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.CONFIDENCE,
        title="Confidence",
        content={
            "value": spacetime.confidence,
            "threshold_low": 0.6,
            "threshold_high": 0.8,
            "tool_used": spacetime.tool_used,
        },
        size=PanelSizeJenny.SMALL,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.CONFIDENCE),
    )


def generate_sources_panel(
    spacetime: Spacetime,
    priority: int = 2,
) -> JennySpec:
    """Generate a sources attribution panel."""
    if not spacetime.sources_used:
        return None

    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.SOURCES,
        title="Sources",
        subtitle=f"{len(spacetime.sources_used)} source(s) used",
        content={
            "sources": [
                {"path": src, "relevance": 1.0}  # Would compute actual relevance
                for src in spacetime.sources_used
            ],
        },
        size=PanelSizeJenny.MEDIUM,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.SOURCES),
    )


def generate_graph_panel(
    spacetime: Spacetime,
    priority: int = 3,
) -> JennySpec:
    """Generate a knowledge graph panel showing activated threads."""
    trace = spacetime.trace
    if not trace or len(trace.threads_activated) < 2:
        return None

    # Build nodes and edges from activated threads
    nodes = [
        {"id": thread, "label": thread.split("_")[-1], "type": "concept"}
        for thread in trace.threads_activated[:10]  # Limit to 10
    ]

    # Simple linear connections for now
    edges = [
        {"source": trace.threads_activated[i], "target": trace.threads_activated[i+1], "type": "related"}
        for i in range(min(len(trace.threads_activated) - 1, 9))
    ]

    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.GRAPH,
        title="Knowledge Graph",
        subtitle=f"{len(trace.threads_activated)} threads activated",
        content={
            "nodes": nodes,
            "edges": edges,
            "layout": "force",
        },
        size=PanelSizeJenny.MEDIUM,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.GRAPH),
    )


def generate_timeline_panel(
    spacetime: Spacetime,
    priority: int = 4,
) -> JennySpec:
    """Generate a stage timeline panel showing execution phases."""
    trace = spacetime.trace
    if not trace or not trace.stage_durations:
        return None

    stages = [
        {
            "name": stage,
            "duration_ms": duration,
            "status": "success",
        }
        for stage, duration in trace.stage_durations.items()
    ]

    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.TIMELINE,
        title="Execution Timeline",
        subtitle=f"Total: {trace.duration_ms:.1f}ms",
        content={
            "stages": stages,
            "total_duration_ms": trace.duration_ms,
        },
        size=PanelSizeJenny.MEDIUM,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.TIMELINE),
    )


def generate_reasoning_panel(
    spacetime: Spacetime,
    priority: int = 5,
) -> JennySpec:
    """Generate a reasoning chain panel showing detected motifs."""
    trace = spacetime.trace
    if not trace or not trace.motifs_detected:
        return None

    steps = [
        {"step": i + 1, "motif": motif, "description": f"Detected pattern: {motif}"}
        for i, motif in enumerate(trace.motifs_detected[:5])
    ]

    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.REASONING,
        title="Reasoning Chain",
        subtitle=f"{len(trace.motifs_detected)} patterns detected",
        content={
            "steps": steps,
            "policy_adapter": trace.policy_adapter,
        },
        size=PanelSizeJenny.MEDIUM,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.REASONING),
    )


def generate_metric_panel(
    spacetime: Spacetime,
    metric_name: str,
    metric_value: Any,
    priority: int = 6,
) -> JennySpec:
    """Generate a single metric panel."""
    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.METRIC,
        title=metric_name,
        content={
            "value": metric_value,
            "format": "number" if isinstance(metric_value, (int, float)) else "text",
        },
        size=PanelSizeJenny.SMALL,
        priority=priority,
        actions=get_default_actions(PanelTypeJenny.METRIC),
    )


def generate_why_panel(
    spacetime: Spacetime,
    specs_generated: List[JennySpec],
    analysis: QueryAnalysis,
    priority: int = 99,  # Always lowest priority (shown last)
) -> JennySpec:
    """Generate a 'Why this UI?' meta-panel (SYSTEM stage to break infinite loop)."""
    return JennySpec(
        spacetime_id=f"st_{id(spacetime)}",
        panel_type=PanelTypeJenny.WHY,
        title="Why This UI?",
        content={
            "query_type": analysis.query_type,
            "complexity": analysis.complexity,
            "panels_generated": len(specs_generated),
            "panel_types": [s.panel_type.value for s in specs_generated],
            "reasoning": f"Query classified as {analysis.query_type} ({analysis.complexity}). "
                        f"Generated {len(specs_generated)} panels based on content analysis.",
        },
        size=PanelSizeJenny.SMALL,
        priority=priority,
        lifecycle=LifecycleStage.SYSTEM,  # SYSTEM stage - not logged (breaks infinite loop)
        actions=get_default_actions(PanelTypeJenny.WHY),
    )


# ============================================================================
# JennyCompiler - Main Implementation
# ============================================================================

class JennyCompiler:
    """
    Rule-based compiler that transforms Spacetime into JennySpec panels.

    Implements JennyCompilerProtocol using heuristic rules.
    Future versions will support LLM-based compilation.

    Strategy determines panel count:
    - MINIMAL: 1-2 panels (text + confidence)
    - BALANCED: 2-4 panels (text + confidence + sources/graph)
    - COMPREHENSIVE: 4-6 panels (all relevant panels)
    - AUTO: Automatically select based on query complexity

    Usage:
        compiler = JennyCompiler()
        specs = await compiler.compile(spacetime, strategy=CompilationStrategy.AUTO)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        default_strategy: CompilationStrategy = CompilationStrategy.AUTO,
        include_why_panel: bool = True,
    ):
        """
        Initialize compiler.

        Args:
            default_strategy: Default compilation strategy
            include_why_panel: Whether to include "Why this UI?" panel
        """
        self.default_strategy = default_strategy
        self.include_why_panel = include_why_panel

        logger.info(f"JennyCompiler v{self.VERSION} initialized (strategy={default_strategy.value})")

    @property
    def compiler_version(self) -> str:
        """Return compiler version for provenance."""
        return self.VERSION

    async def compile(
        self,
        spacetime: Spacetime,
        strategy: CompilationStrategy = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[JennySpec]:
        """
        Compile Spacetime into UI specifications.

        Args:
            spacetime: Woven output from HoloLoom weaving cycle
            strategy: Compilation strategy (None = use default)
            context: Optional user/session context

        Returns:
            List of JennySpec panels to render
        """
        strategy = strategy or self.default_strategy

        # Analyze query
        analysis = analyze_query(spacetime)

        # Determine effective strategy
        if strategy == CompilationStrategy.AUTO:
            strategy = self._auto_select_strategy(analysis)

        # Generate panels based on strategy
        specs = []

        # Always include text panel
        specs.append(generate_text_panel(spacetime, priority=0))

        # Add panels based on strategy
        if strategy in [CompilationStrategy.MINIMAL, CompilationStrategy.BALANCED, CompilationStrategy.COMPREHENSIVE]:
            specs.append(generate_confidence_panel(spacetime, priority=1))

        if strategy in [CompilationStrategy.BALANCED, CompilationStrategy.COMPREHENSIVE]:
            # Add sources if available
            sources_panel = generate_sources_panel(spacetime, priority=2)
            if sources_panel:
                specs.append(sources_panel)

            # Add graph if complex enough
            if analysis.has_graph_context:
                graph_panel = generate_graph_panel(spacetime, priority=3)
                if graph_panel:
                    specs.append(graph_panel)

        if strategy == CompilationStrategy.COMPREHENSIVE:
            # Add timeline
            timeline_panel = generate_timeline_panel(spacetime, priority=4)
            if timeline_panel:
                specs.append(timeline_panel)

            # Add reasoning chain
            if analysis.has_reasoning:
                reasoning_panel = generate_reasoning_panel(spacetime, priority=5)
                if reasoning_panel:
                    specs.append(reasoning_panel)

            # Add key metrics
            if analysis.has_metrics and spacetime.trace:
                specs.append(generate_metric_panel(
                    spacetime,
                    "Duration",
                    f"{spacetime.trace.duration_ms:.1f}ms",
                    priority=6
                ))
                specs.append(generate_metric_panel(
                    spacetime,
                    "Threads",
                    len(spacetime.trace.threads_activated),
                    priority=7
                ))

        # Add "Why this UI?" panel if enabled
        if self.include_why_panel:
            why_panel = generate_why_panel(spacetime, specs, analysis, priority=99)
            specs.append(why_panel)

        # Sort by priority
        specs.sort(key=lambda s: s.priority)

        logger.debug(
            f"Compiled {len(specs)} panels for query (strategy={strategy.value}, "
            f"type={analysis.query_type}, complexity={analysis.complexity})"
        )

        return specs

    async def compile_stream(
        self,
        spacetime: Spacetime,
        strategy: CompilationStrategy = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[JennySpec]:
        """
        Stream-compile Spacetime for progressive rendering.

        Yields panels in priority order for progressive display.
        """
        specs = await self.compile(spacetime, strategy, context)

        for spec in specs:
            yield spec

    def get_panel_strategy(self, spacetime: Spacetime) -> Dict[str, Any]:
        """
        Analyze Spacetime and recommend panel strategy (dry run).

        Does not generate specs, just returns strategy recommendation.
        """
        analysis = analyze_query(spacetime)
        strategy = self._auto_select_strategy(analysis)

        # Estimate panel types
        panel_types = ["text", "confidence"]
        if analysis.has_sources:
            panel_types.append("sources")
        if analysis.has_graph_context:
            panel_types.append("graph")
        if strategy == CompilationStrategy.COMPREHENSIVE:
            if analysis.has_metrics:
                panel_types.append("timeline")
            if analysis.has_reasoning:
                panel_types.append("reasoning")

        return {
            "recommended_strategy": strategy.value,
            "estimated_panels": len(panel_types),
            "panel_types": panel_types,
            "layout_hint": self._suggest_layout(analysis, len(panel_types)),
            "reasoning": f"Query type: {analysis.query_type}, complexity: {analysis.complexity}",
            "query_analysis": {
                "type": analysis.query_type,
                "complexity": analysis.complexity,
                "confidence_level": analysis.confidence_level,
                "has_graph": analysis.has_graph_context,
                "has_sources": analysis.has_sources,
            }
        }

    def _auto_select_strategy(self, analysis: QueryAnalysis) -> CompilationStrategy:
        """Select compilation strategy based on query analysis."""
        if analysis.complexity == "simple" and analysis.query_type == "factual":
            return CompilationStrategy.MINIMAL

        if analysis.complexity == "complex" or analysis.query_type in ["analytical", "debug"]:
            return CompilationStrategy.COMPREHENSIVE

        return CompilationStrategy.BALANCED

    def _suggest_layout(self, analysis: QueryAnalysis, panel_count: int) -> str:
        """Suggest layout hint based on analysis."""
        if panel_count <= 2:
            return LayoutHint.STACK.value
        if panel_count <= 4:
            return LayoutHint.GRID.value
        return LayoutHint.FLOW.value


# ============================================================================
# Factory Function
# ============================================================================

def create_jenny_compiler(
    default_strategy: CompilationStrategy = CompilationStrategy.AUTO,
    include_why_panel: bool = True,
) -> JennyCompiler:
    """
    Create a JennyCompiler instance.

    Args:
        default_strategy: Default compilation strategy
        include_why_panel: Whether to include "Why this UI?" panel

    Returns:
        Configured JennyCompiler instance
    """
    return JennyCompiler(
        default_strategy=default_strategy,
        include_why_panel=include_why_panel,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'QueryAnalysis',
    'analyze_query',
    'JennyCompiler',
    'create_jenny_compiler',
    # Panel generators (for custom compilation)
    'generate_text_panel',
    'generate_confidence_panel',
    'generate_sources_panel',
    'generate_graph_panel',
    'generate_timeline_panel',
    'generate_reasoning_panel',
    'generate_metric_panel',
    'generate_why_panel',
]
