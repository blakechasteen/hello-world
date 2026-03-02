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
from hololoom.protocols.jenny import CompilationStrategy, RenderTarget

# Try to import Spacetime
try:
    from hololoom.fabric.spacetime import Spacetime, WeavingTrace
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
# Panel Generator Strategy Pattern
# ============================================================================

from abc import ABC, abstractmethod


class PanelGenerator(ABC):
    """
    Abstract base class for panel generators (Strategy Pattern).

    Each panel type has a dedicated generator class that encapsulates
    the logic for creating that panel type from a Spacetime.

    Usage:
        generator = TextPanelGenerator()
        if generator.should_generate(spacetime, analysis):
            spec = generator.generate(spacetime)
    """

    # Class attributes - override in subclasses
    panel_type: PanelTypeJenny = None
    default_priority: int = 0
    default_size: PanelSizeJenny = PanelSizeJenny.MEDIUM

    @abstractmethod
    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        **kwargs
    ) -> Optional[JennySpec]:
        """
        Generate a JennySpec panel from spacetime.

        Args:
            spacetime: The woven output to create panel from
            priority: Panel priority (lower = higher priority). If None, uses default_priority.
            **kwargs: Additional generator-specific arguments

        Returns:
            JennySpec or None if panel cannot be generated
        """
        pass

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        """
        Check if this panel should be generated for the given spacetime.

        Override in subclasses for conditional panel generation.
        Default implementation returns True.
        """
        return True

    def _make_spacetime_id(self, spacetime: Spacetime) -> str:
        """Generate consistent spacetime_id."""
        return f"st_{id(spacetime)}"

    def _get_priority(self, priority: Optional[int]) -> int:
        """Get priority, using default if not specified."""
        return priority if priority is not None else self.default_priority

    def _get_actions(self) -> tuple:
        """Get default actions for this panel type."""
        return get_default_actions(self.panel_type)


# ============================================================================
# Panel Generator Implementations
# ============================================================================

class TextPanelGenerator(PanelGenerator):
    """Generator for text response panels."""

    panel_type = PanelTypeJenny.TEXT
    default_priority = 0
    default_size = PanelSizeJenny.LARGE

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        size: PanelSizeJenny = None,
        **kwargs
    ) -> JennySpec:
        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Response",
            subtitle=spacetime.query_text[:50] + "..." if len(spacetime.query_text) > 50 else spacetime.query_text,
            content={"text": spacetime.response, "format": "markdown"},
            size=size or self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class ConfidencePanelGenerator(PanelGenerator):
    """Generator for confidence gauge panels."""

    panel_type = PanelTypeJenny.CONFIDENCE
    default_priority = 1
    default_size = PanelSizeJenny.SMALL

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        **kwargs
    ) -> JennySpec:
        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Confidence",
            content={
                "value": spacetime.confidence,
                "threshold_low": 0.6,
                "threshold_high": 0.8,
                "tool_used": spacetime.tool_used,
            },
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class SourcesPanelGenerator(PanelGenerator):
    """Generator for sources attribution panels."""

    panel_type = PanelTypeJenny.SOURCES
    default_priority = 2
    default_size = PanelSizeJenny.MEDIUM

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        return bool(spacetime.sources_used)

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        **kwargs
    ) -> Optional[JennySpec]:
        if not spacetime.sources_used:
            return None
        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Sources",
            subtitle=f"{len(spacetime.sources_used)} source(s) used",
            content={
                "sources": [
                    {"path": src, "relevance": 1.0}
                    for src in spacetime.sources_used
                ],
            },
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class GraphPanelGenerator(PanelGenerator):
    """Generator for knowledge graph panels."""

    panel_type = PanelTypeJenny.GRAPH
    default_priority = 3
    default_size = PanelSizeJenny.MEDIUM

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        return analysis.has_graph_context

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        **kwargs
    ) -> Optional[JennySpec]:
        trace = spacetime.trace
        if not trace or len(trace.threads_activated) < 2:
            return None

        nodes = [
            {"id": thread, "label": thread.split("_")[-1], "type": "concept"}
            for thread in trace.threads_activated[:10]
        ]
        edges = [
            {"source": trace.threads_activated[i], "target": trace.threads_activated[i+1], "type": "related"}
            for i in range(min(len(trace.threads_activated) - 1, 9))
        ]

        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Knowledge Graph",
            subtitle=f"{len(trace.threads_activated)} threads activated",
            content={"nodes": nodes, "edges": edges, "layout": "force"},
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class TimelinePanelGenerator(PanelGenerator):
    """Generator for execution timeline panels."""

    panel_type = PanelTypeJenny.TIMELINE
    default_priority = 4
    default_size = PanelSizeJenny.MEDIUM

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        return analysis.has_metrics

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        **kwargs
    ) -> Optional[JennySpec]:
        trace = spacetime.trace
        if not trace or not trace.stage_durations:
            return None

        stages = [
            {"name": stage, "duration_ms": duration, "status": "success"}
            for stage, duration in trace.stage_durations.items()
        ]

        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Execution Timeline",
            subtitle=f"Total: {trace.duration_ms:.1f}ms",
            content={"stages": stages, "total_duration_ms": trace.duration_ms},
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class ReasoningPanelGenerator(PanelGenerator):
    """
    Generator for step-by-step reasoning chain panels.

    Enhanced (December 2025 - Task 3.2) to provide:
    - Numbered reasoning steps with confidence scores
    - Visual flow diagram (mermaid and simple text formats)
    - Expandable step details
    - Verification status per step
    """

    panel_type = PanelTypeJenny.REASONING
    default_priority = 5
    default_size = PanelSizeJenny.LARGE  # Upgraded from MEDIUM for richer content

    # Step type classifications and their visual markers
    STEP_TYPES = {
        "query_analysis": {"icon": "🔍", "label": "Query Analysis", "color": "#3498db"},
        "memory_retrieval": {"icon": "📚", "label": "Memory Retrieval", "color": "#9b59b6"},
        "pattern_detection": {"icon": "🧩", "label": "Pattern Detection", "color": "#e74c3c"},
        "context_expansion": {"icon": "🌐", "label": "Context Expansion", "color": "#2ecc71"},
        "decision_making": {"icon": "⚖️", "label": "Decision Making", "color": "#f39c12"},
        "verification": {"icon": "✓", "label": "Verification", "color": "#1abc9c"},
        "synthesis": {"icon": "🔗", "label": "Synthesis", "color": "#34495e"},
    }

    # Verification statuses
    VERIFICATION_STATUSES = {
        "complete": {"icon": "✅", "label": "Verified", "description": "Step completed and verified"},
        "in_progress": {"icon": "⏳", "label": "In Progress", "description": "Step currently executing"},
        "pending": {"icon": "⏸️", "label": "Pending", "description": "Step not yet started"},
        "skipped": {"icon": "⏭️", "label": "Skipped", "description": "Step skipped (not needed)"},
        "failed": {"icon": "❌", "label": "Failed", "description": "Step failed verification"},
    }

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        return analysis.has_reasoning

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        verification_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[JennySpec]:
        """
        Generate an enhanced step-by-step reasoning panel.

        Args:
            spacetime: The woven output
            priority: Panel priority
            verification_results: Optional verification results from agentic reasoning

        Returns:
            JennySpec with rich reasoning visualization
        """
        trace = spacetime.trace
        if not trace or not trace.motifs_detected:
            return None

        # Build detailed reasoning steps
        steps = self._build_reasoning_steps(trace, spacetime.confidence, verification_results)

        # Build flow diagram (both mermaid and text formats)
        flow_diagram = self._build_flow_diagram(steps)

        # Calculate overall verification status
        verification_summary = self._build_verification_summary(steps, verification_results)

        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Reasoning Chain",
            subtitle=f"{len(steps)} steps • {verification_summary['verified_count']}/{len(steps)} verified",
            content={
                # Detailed steps with full information
                "steps": steps,
                # Flow diagram in multiple formats
                "flow_diagram": flow_diagram,
                # Verification summary
                "verification": verification_summary,
                # Legacy fields for backward compatibility
                "policy_adapter": trace.policy_adapter,
                "motifs_detected": trace.motifs_detected[:10],
                # Metadata for rendering
                "render_hints": {
                    "expandable": True,
                    "show_confidence_bars": True,
                    "animate_flow": True,
                },
            },
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
            metadata={
                "enriched_version": "3.2",  # Task 3.2 enhanced version
                "has_flow_diagram": True,
                "has_verification": verification_results is not None,
            },
        )

    def _build_reasoning_steps(
        self,
        trace: Any,
        overall_confidence: float,
        verification_results: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build detailed reasoning steps with confidence and verification."""
        steps = []

        # Step 1: Query Analysis (always present)
        steps.append({
            "num": 1,
            "type": "query_analysis",
            "action": "Query Analysis",
            "status": "complete",
            "confidence": min(0.95, overall_confidence + 0.15),  # Usually high confidence
            "icon": self.STEP_TYPES["query_analysis"]["icon"],
            "details": {
                "description": "Analyzed query intent and extracted keywords",
                "duration_ms": trace.stage_durations.get("query_analysis", 5.0) if trace.stage_durations else 5.0,
                "expandable_content": f"Detected motifs: {', '.join(trace.motifs_detected[:3]) if trace.motifs_detected else 'none'}",
            },
            "verification": self._get_step_verification("query_analysis", verification_results),
        })

        # Step 2: Memory Retrieval
        if trace.threads_activated:
            steps.append({
                "num": 2,
                "type": "memory_retrieval",
                "action": "Memory Retrieval",
                "status": "complete",
                "confidence": min(0.9, overall_confidence + 0.1),
                "icon": self.STEP_TYPES["memory_retrieval"]["icon"],
                "details": {
                    "description": f"Retrieved {len(trace.threads_activated)} relevant memory threads",
                    "duration_ms": trace.stage_durations.get("retrieval", 20.0) if trace.stage_durations else 20.0,
                    "expandable_content": f"Threads: {', '.join(trace.threads_activated[:5])}{'...' if len(trace.threads_activated) > 5 else ''}",
                },
                "verification": self._get_step_verification("memory_retrieval", verification_results),
            })

        # Step 3: Pattern Detection (based on motifs)
        for i, motif in enumerate(trace.motifs_detected[:3]):
            step_confidence = max(0.5, overall_confidence - (i * 0.1))  # Decreasing confidence for later patterns
            steps.append({
                "num": len(steps) + 1,
                "type": "pattern_detection",
                "action": f"Pattern: {motif}",
                "status": "complete",
                "confidence": step_confidence,
                "icon": self.STEP_TYPES["pattern_detection"]["icon"],
                "details": {
                    "description": f"Detected reasoning pattern: {motif}",
                    "duration_ms": 2.0 + i,  # Small duration per pattern
                    "expandable_content": self._get_motif_explanation(motif),
                },
                "verification": self._get_step_verification(f"pattern_{i}", verification_results),
            })

        # Step 4: Decision Making
        steps.append({
            "num": len(steps) + 1,
            "type": "decision_making",
            "action": "Decision Making",
            "status": "complete",
            "confidence": overall_confidence,
            "icon": self.STEP_TYPES["decision_making"]["icon"],
            "details": {
                "description": f"Selected response strategy using {trace.policy_adapter or 'default'} adapter",
                "duration_ms": trace.stage_durations.get("decision", 10.0) if trace.stage_durations else 10.0,
                "expandable_content": f"Policy adapter: {trace.policy_adapter}, Tool selected based on confidence threshold",
            },
            "verification": self._get_step_verification("decision_making", verification_results),
        })

        # Step 5: Verification (if verification results provided)
        if verification_results:
            verified = verification_results.get("verified", False)
            claims_checked = verification_results.get("claims_checked", 0)
            claims_verified = verification_results.get("claims_verified", 0)

            steps.append({
                "num": len(steps) + 1,
                "type": "verification",
                "action": "Verification",
                "status": "complete" if verified else "failed",
                "confidence": claims_verified / max(claims_checked, 1),
                "icon": self.STEP_TYPES["verification"]["icon"],
                "details": {
                    "description": f"Verified {claims_verified}/{claims_checked} claims",
                    "duration_ms": verification_results.get("duration_ms", 50.0),
                    "expandable_content": verification_results.get("details", "Verification complete"),
                },
                "verification": {
                    "status": "complete" if verified else "failed",
                    "checked": True,
                    "passed": verified,
                },
            })

        return steps

    def _build_flow_diagram(self, steps: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build flow diagram in multiple formats."""
        # Simple text flow (arrows)
        text_flow_parts = []
        for i, step in enumerate(steps):
            status_icon = "✓" if step["status"] == "complete" else "⏳" if step["status"] == "in_progress" else "○"
            text_flow_parts.append(f"{status_icon} {step['action']}")

        text_flow = " → ".join(text_flow_parts)

        # Mermaid format for rich rendering
        mermaid_lines = ["graph LR"]
        for i, step in enumerate(steps):
            node_id = f"S{i}"
            label = step["action"].replace('"', "'")
            status_class = "complete" if step["status"] == "complete" else "pending"

            # Node definition
            mermaid_lines.append(f'    {node_id}["{step["num"]}. {label}"]')

            # Add class for styling
            mermaid_lines.append(f'    class {node_id} {status_class}')

            # Connection to next step
            if i < len(steps) - 1:
                next_id = f"S{i+1}"
                conf_label = f"{step['confidence']:.0%}"
                mermaid_lines.append(f'    {node_id} -->|{conf_label}| {next_id}')

        # Add class definitions
        mermaid_lines.append('    classDef complete fill:#2ecc71,stroke:#27ae60,color:#fff')
        mermaid_lines.append('    classDef pending fill:#f39c12,stroke:#e67e22,color:#fff')
        mermaid_lines.append('    classDef failed fill:#e74c3c,stroke:#c0392b,color:#fff')

        mermaid_diagram = "\n".join(mermaid_lines)

        return {
            "text": text_flow,
            "mermaid": mermaid_diagram,
            "format": "both",  # Renderer can choose
        }

    def _build_verification_summary(
        self,
        steps: List[Dict[str, Any]],
        verification_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build verification summary across all steps."""
        verified_count = sum(1 for s in steps if s.get("verification", {}).get("passed", s["status"] == "complete"))
        pending_count = sum(1 for s in steps if s["status"] == "pending" or s["status"] == "in_progress")
        failed_count = sum(1 for s in steps if s["status"] == "failed" or (s.get("verification", {}).get("checked") and not s.get("verification", {}).get("passed")))

        return {
            "verified_count": verified_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
            "total_steps": len(steps),
            "overall_status": "complete" if failed_count == 0 and pending_count == 0 else "partial" if verified_count > 0 else "pending",
            "claims_checked": verification_results.get("claims_checked", 0) if verification_results else 0,
            "claims_verified": verification_results.get("claims_verified", 0) if verification_results else 0,
        }

    def _get_step_verification(
        self,
        step_key: str,
        verification_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get verification status for a specific step."""
        if not verification_results:
            return {"status": "not_checked", "checked": False, "passed": None}

        step_verifications = verification_results.get("step_verifications", {})
        if step_key in step_verifications:
            return step_verifications[step_key]

        # Default: assume passed if overall verification passed
        return {
            "status": "inferred",
            "checked": False,
            "passed": verification_results.get("verified", True),
        }

    def _get_motif_explanation(self, motif: str) -> str:
        """Get human-readable explanation for a detected motif."""
        explanations = {
            "question→answer": "Direct question-answer pattern detected. Response will provide a direct answer.",
            "step-by-step": "Procedural pattern detected. Response will be structured as steps.",
            "compare": "Comparison pattern detected. Response will contrast multiple items.",
            "explain": "Explanation pattern detected. Response will provide detailed explanation.",
            "debug": "Debugging pattern detected. Response will analyze execution or errors.",
            "creative": "Creative pattern detected. Response will generate novel content.",
            "code": "Code pattern detected. Response may include code examples.",
        }
        return explanations.get(motif, f"Pattern '{motif}' influences response generation strategy.")


class MetricPanelGenerator(PanelGenerator):
    """Generator for single metric panels."""

    panel_type = PanelTypeJenny.METRIC
    default_priority = 6
    default_size = PanelSizeJenny.SMALL

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        metric_name: str = "Metric",
        metric_value: Any = None,
        **kwargs
    ) -> JennySpec:
        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title=metric_name,
            content={
                "value": metric_value,
                "format": "number" if isinstance(metric_value, (int, float)) else "text",
            },
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
        )


class ComparisonPanelGenerator(PanelGenerator):
    """
    Generator for side-by-side comparison panels.

    Added (December 2025 - Task 3.3) to provide:
    - Before/after code changes
    - Two memory snapshots
    - Multiple query responses
    - Side-by-side analysis with diff highlighting

    Use Cases:
    - Comparing different exploration strategies
    - Before/after code refactoring
    - Comparing multiple answer alternatives
    - Memory state changes over time
    """

    panel_type = PanelTypeJenny.COMPARISON
    default_priority = 3  # Higher priority when comparison is relevant
    default_size = PanelSizeJenny.XLARGE  # Full width for side-by-side

    # Diff type configurations
    DIFF_TYPES = {
        "unified": {
            "label": "Unified Diff",
            "description": "Standard unified diff format (like git diff)",
            "icon": "📝",
        },
        "side_by_side": {
            "label": "Side by Side",
            "description": "Two columns for direct comparison",
            "icon": "⬅️➡️",
        },
        "inline": {
            "label": "Inline Changes",
            "description": "Changes highlighted within the text",
            "icon": "🔀",
        },
        "semantic": {
            "label": "Semantic Diff",
            "description": "Compares meaning rather than text",
            "icon": "🧠",
        },
    }

    # Comparison category configurations
    COMPARISON_CATEGORIES = {
        "code": {
            "icon": "💻",
            "label": "Code Comparison",
            "default_diff_type": "unified",
            "supports_syntax_highlighting": True,
        },
        "text": {
            "icon": "📄",
            "label": "Text Comparison",
            "default_diff_type": "side_by_side",
            "supports_syntax_highlighting": False,
        },
        "memory": {
            "icon": "🧠",
            "label": "Memory Snapshot",
            "default_diff_type": "semantic",
            "supports_syntax_highlighting": False,
        },
        "response": {
            "icon": "💬",
            "label": "Response Comparison",
            "default_diff_type": "side_by_side",
            "supports_syntax_highlighting": False,
        },
        "config": {
            "icon": "⚙️",
            "label": "Configuration Comparison",
            "default_diff_type": "unified",
            "supports_syntax_highlighting": True,
        },
    }

    def should_generate(self, spacetime: Spacetime, analysis: QueryAnalysis) -> bool:
        """Check if comparison panel should be generated."""
        # Generate for analytical queries
        return analysis.query_type == "analytical"

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        left: Optional[Dict[str, Any]] = None,
        right: Optional[Dict[str, Any]] = None,
        diff_type: str = "side_by_side",
        category: str = "text",
        **kwargs
    ) -> Optional[JennySpec]:
        """
        Generate a side-by-side comparison panel.

        Args:
            spacetime: The woven output
            priority: Panel priority
            left: Left side content {"title": str, "content": str, "metadata": dict}
            right: Right side content {"title": str, "content": str, "metadata": dict}
            diff_type: Type of diff ("unified", "side_by_side", "inline", "semantic")
            category: Comparison category ("code", "text", "memory", "response", "config")

        Returns:
            JennySpec with comparison visualization
        """
        # If no left/right provided, try to extract from spacetime
        if left is None or right is None:
            left, right = self._extract_comparison_from_spacetime(spacetime)

        if left is None or right is None:
            return None

        # Validate and get category config
        category_config = self.COMPARISON_CATEGORIES.get(category, self.COMPARISON_CATEGORIES["text"])
        diff_config = self.DIFF_TYPES.get(diff_type, self.DIFF_TYPES["side_by_side"])

        # Calculate changes summary
        changes = self._calculate_changes(left, right, diff_type)

        # Build comparison content
        content = {
            # Main comparison data
            "left": {
                "title": left.get("title", "Before"),
                "content": left.get("content", ""),
                "metadata": left.get("metadata", {}),
            },
            "right": {
                "title": right.get("title", "After"),
                "content": right.get("content", ""),
                "metadata": right.get("metadata", {}),
            },
            # Diff configuration
            "diff_type": diff_type,
            "diff_config": diff_config,
            # Changes summary
            "changes": changes,
            # Category configuration
            "category": category,
            "category_config": category_config,
            # Rendering hints
            "render_hints": {
                "syntax_highlighting": category_config.get("supports_syntax_highlighting", False),
                "line_numbers": category in ["code", "config"],
                "word_wrap": category not in ["code"],
                "show_diff_stats": True,
                "collapsible_unchanged": True,
            },
        }

        # Generate subtitle based on changes
        subtitle = self._build_subtitle(changes, category_config)

        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title=f"{category_config['icon']} {category_config['label']}",
            subtitle=subtitle,
            content=content,
            size=self.default_size,
            priority=self._get_priority(priority),
            actions=self._get_actions(),
            metadata={
                "enriched_version": "3.3",  # Task 3.3 version
                "comparison_category": category,
                "diff_type": diff_type,
                "has_changes": changes["total_changes"] > 0,
            },
        )

    def _extract_comparison_from_spacetime(
        self,
        spacetime: Spacetime
    ) -> tuple:
        """Extract comparison data from spacetime if available."""
        trace = spacetime.trace
        metadata = spacetime.metadata or {}

        # Check if comparison data is in metadata
        if "comparison" in metadata:
            comp = metadata["comparison"]
            return comp.get("left"), comp.get("right")

        # Check if there are multiple responses to compare
        if "alternatives" in metadata and len(metadata["alternatives"]) >= 2:
            alts = metadata["alternatives"]
            return (
                {"title": "Option A", "content": alts[0], "metadata": {"index": 0}},
                {"title": "Option B", "content": alts[1], "metadata": {"index": 1}},
            )

        # Check if there's before/after in trace
        if trace:
            stage_durations = trace.stage_durations or {}
            if "before_state" in metadata and "after_state" in metadata:
                return (
                    {"title": "Before", "content": str(metadata["before_state"]), "metadata": {}},
                    {"title": "After", "content": str(metadata["after_state"]), "metadata": {}},
                )

        return None, None

    def _calculate_changes(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        diff_type: str
    ) -> Dict[str, Any]:
        """Calculate changes between left and right content."""
        left_content = left.get("content", "")
        right_content = right.get("content", "")

        # Simple line-based diff for now
        left_lines = left_content.split("\n") if left_content else []
        right_lines = right_content.split("\n") if right_content else []

        # Count additions, deletions, and modifications
        left_set = set(left_lines)
        right_set = set(right_lines)

        added_lines = len(right_set - left_set)
        removed_lines = len(left_set - right_set)
        unchanged_lines = len(left_set & right_set)

        # Calculate semantic similarity if semantic diff
        similarity = 1.0 - (added_lines + removed_lines) / max(len(left_lines) + len(right_lines), 1)

        return {
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "unchanged_lines": unchanged_lines,
            "total_changes": added_lines + removed_lines,
            "similarity": round(similarity, 3),
            "left_total_lines": len(left_lines),
            "right_total_lines": len(right_lines),
            # Formatted stats
            "stats_text": f"+{added_lines} -{removed_lines}",
            "change_summary": self._format_change_summary(added_lines, removed_lines),
        }

    def _format_change_summary(self, added: int, removed: int) -> str:
        """Format a human-readable change summary."""
        parts = []
        if added > 0:
            parts.append(f"{added} addition{'s' if added != 1 else ''}")
        if removed > 0:
            parts.append(f"{removed} deletion{'s' if removed != 1 else ''}")
        if not parts:
            return "No changes detected"
        return ", ".join(parts)

    def _build_subtitle(
        self,
        changes: Dict[str, Any],
        category_config: Dict[str, Any]
    ) -> str:
        """Build subtitle from changes summary."""
        similarity_pct = int(changes["similarity"] * 100)
        return f"{changes['stats_text']} • {similarity_pct}% similar"


class WhyPanelGenerator(PanelGenerator):
    """
    Generator for 'Why this UI?' meta-panels.

    Enhanced (December 2025) to provide:
    - Decision confidence and reasoning
    - Alternatives considered with scores
    - Customization hints
    - Thompson Sampling statistics (when available)
    """

    panel_type = PanelTypeJenny.WHY
    default_priority = 99  # Always lowest priority
    default_size = PanelSizeJenny.MEDIUM  # Upgraded from SMALL for richer content

    # Alternative panel configurations by query type
    PANEL_ALTERNATIVES = {
        "factual": [
            {"type": "TEXT", "score": 0.9, "reason": "Direct answer format"},
            {"type": "TABLE", "score": 0.4, "reason": "Structured data would be overkill"},
            {"type": "GRAPH", "score": 0.3, "reason": "No relationship visualization needed"},
        ],
        "procedural": [
            {"type": "TEXT", "score": 0.85, "reason": "Step-by-step instructions"},
            {"type": "CODE", "score": 0.7, "reason": "May include code examples"},
            {"type": "TIMELINE", "score": 0.5, "reason": "Sequential process visualization"},
        ],
        "analytical": [
            {"type": "COMPARISON", "score": 0.85, "reason": "Side-by-side analysis optimal"},
            {"type": "GRAPH", "score": 0.75, "reason": "Relationship visualization helpful"},
            {"type": "TABLE", "score": 0.6, "reason": "Data comparison possible"},
            {"type": "TEXT", "score": 0.5, "reason": "Narrative less structured"},
        ],
        "creative": [
            {"type": "TEXT", "score": 0.9, "reason": "Creative output needs prose"},
            {"type": "CODE", "score": 0.6, "reason": "If code generation requested"},
        ],
        "debug": [
            {"type": "TIMELINE", "score": 0.9, "reason": "Execution trace visualization"},
            {"type": "REASONING", "score": 0.85, "reason": "Step-by-step debugging"},
            {"type": "GRAPH", "score": 0.7, "reason": "Dependency visualization"},
            {"type": "TEXT", "score": 0.5, "reason": "Less structured for debugging"},
        ],
    }

    # Customization hints by panel type
    CUSTOMIZATION_HINTS = {
        "text": "Pin to keep visible. Dismiss to train for shorter responses.",
        "confidence": "Pin if confidence display is useful. Dismiss to reduce UI clutter.",
        "sources": "Pin to always show sources. Dismiss if you trust the response.",
        "graph": "Expand for larger view. Pin for persistent knowledge visualization.",
        "timeline": "Useful for performance debugging. Dismiss if not analyzing latency.",
        "reasoning": "Expand to see full reasoning chain. Pin for transparency.",
        "comparison": "Pin for persistent comparison. Dismiss after decision made.",
    }

    def generate(
        self,
        spacetime: Spacetime,
        priority: int = None,
        specs_generated: List[JennySpec] = None,
        analysis: QueryAnalysis = None,
        thompson_stats: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> JennySpec:
        """
        Generate an enhanced 'Why this UI?' meta-panel.

        Args:
            spacetime: The woven output
            priority: Panel priority
            specs_generated: List of panels that were generated
            analysis: Query analysis results
            thompson_stats: Optional Thompson Sampling statistics from panel learner

        Returns:
            JennySpec with rich explanatory content
        """
        specs_generated = specs_generated or []
        query_type = analysis.query_type if analysis else "unknown"
        complexity = analysis.complexity if analysis else "unknown"

        # Build decision explanation
        decision = self._build_decision_explanation(
            query_type, complexity, spacetime.confidence, specs_generated
        )

        # Get alternatives considered
        alternatives = self._get_alternatives_considered(query_type, specs_generated)

        # Build customization hints
        customize = self._build_customization_hints(specs_generated)

        # Build learning statistics (Thompson Sampling)
        learning = self._build_learning_stats(thompson_stats)

        return JennySpec(
            spacetime_id=self._make_spacetime_id(spacetime),
            panel_type=self.panel_type,
            title="Why This UI?",
            subtitle=f"Confidence: {spacetime.confidence:.0%}",
            content={
                # Core decision info
                "decision": decision,
                # Alternatives with scores
                "alternatives": alternatives,
                # Customization guidance
                "customize": customize,
                # Learning statistics
                "learning": learning,
                # Legacy fields for backward compatibility
                "query_type": query_type,
                "complexity": complexity,
                "panels_generated": len(specs_generated),
                "panel_types": [s.panel_type.value for s in specs_generated],
            },
            size=self.default_size,
            priority=self._get_priority(priority),
            lifecycle=LifecycleStage.SYSTEM,  # SYSTEM stage - not logged
            actions=self._get_actions(),
            metadata={
                "enriched_version": "3.1",  # Task 3.1 enhanced version
                "has_alternatives": len(alternatives) > 0,
                "has_learning_stats": learning is not None,
            },
        )

    def _build_decision_explanation(
        self,
        query_type: str,
        complexity: str,
        confidence: float,
        specs_generated: List[JennySpec]
    ) -> str:
        """Build human-readable decision explanation."""
        panel_names = [s.panel_type.value.upper() for s in specs_generated if s.panel_type != PanelTypeJenny.WHY]

        # Confidence descriptor
        conf_desc = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.6 else "low"

        explanation = (
            f"Selected {', '.join(panel_names)} panels "
            f"(confidence: {confidence:.0%}, {conf_desc}). "
            f"Query classified as '{query_type}' with '{complexity}' complexity."
        )

        # Add query-type-specific reasoning
        type_reasons = {
            "factual": "Direct answer format chosen for factual queries.",
            "procedural": "Step-by-step layout optimizes procedural content.",
            "analytical": "Comparison/graph panels enhance analytical understanding.",
            "creative": "Prose-focused layout supports creative output.",
            "debug": "Timeline/reasoning panels reveal execution details.",
        }
        if query_type in type_reasons:
            explanation += f" {type_reasons[query_type]}"

        return explanation

    def _get_alternatives_considered(
        self,
        query_type: str,
        specs_generated: List[JennySpec]
    ) -> List[Dict[str, Any]]:
        """Get alternatives that were considered but not chosen."""
        generated_types = {s.panel_type.value.upper() for s in specs_generated}

        alternatives = []
        for alt in self.PANEL_ALTERNATIVES.get(query_type, []):
            if alt["type"] not in generated_types:
                alternatives.append({
                    "type": alt["type"],
                    "score": alt["score"],
                    "reason": f"Not selected: {alt['reason']}",
                    "would_add": f"Add by pinning a {alt['type']} panel next time",
                })

        # Sort by score descending
        alternatives.sort(key=lambda x: x["score"], reverse=True)
        return alternatives[:3]  # Top 3 alternatives

    def _build_customization_hints(
        self,
        specs_generated: List[JennySpec]
    ) -> Dict[str, str]:
        """Build customization hints for each generated panel."""
        hints = {}
        for spec in specs_generated:
            panel_type = spec.panel_type.value.lower()
            if panel_type in self.CUSTOMIZATION_HINTS:
                hints[panel_type] = self.CUSTOMIZATION_HINTS[panel_type]
        return hints

    def _build_learning_stats(
        self,
        thompson_stats: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Build learning statistics from Thompson Sampling data."""
        if not thompson_stats:
            return None

        return {
            "total_queries": thompson_stats.get("total_queries", 0),
            "query_type_queries": thompson_stats.get("query_type_queries", 0),
            "panel_success_rate": thompson_stats.get("success_rate", 0.0),
            "exploration_rate": thompson_stats.get("exploration_rate", 0.1),
            "confidence_trend": thompson_stats.get("confidence_trend", "stable"),
            "recommendation": thompson_stats.get("recommendation",
                "Pin panels you find useful to train the model"),
        }


# ============================================================================
# Panel Generator Registry
# ============================================================================

PANEL_GENERATORS: Dict[PanelTypeJenny, PanelGenerator] = {
    PanelTypeJenny.TEXT: TextPanelGenerator(),
    PanelTypeJenny.CONFIDENCE: ConfidencePanelGenerator(),
    PanelTypeJenny.SOURCES: SourcesPanelGenerator(),
    PanelTypeJenny.GRAPH: GraphPanelGenerator(),
    PanelTypeJenny.TIMELINE: TimelinePanelGenerator(),
    PanelTypeJenny.REASONING: ReasoningPanelGenerator(),
    PanelTypeJenny.METRIC: MetricPanelGenerator(),
    PanelTypeJenny.WHY: WhyPanelGenerator(),
    PanelTypeJenny.COMPARISON: ComparisonPanelGenerator(),
}


def get_panel_generator(panel_type: PanelTypeJenny) -> Optional[PanelGenerator]:
    """Get the generator for a panel type."""
    return PANEL_GENERATORS.get(panel_type)


def generate_panel(
    panel_type: PanelTypeJenny,
    spacetime: Spacetime,
    priority: int = None,
    **kwargs
) -> Optional[JennySpec]:
    """
    Generate a panel using the registered generator.

    Args:
        panel_type: Type of panel to generate
        spacetime: The woven output
        priority: Panel priority (optional)
        **kwargs: Additional arguments passed to the generator

    Returns:
        JennySpec or None if generator not found or panel cannot be generated
    """
    generator = get_panel_generator(panel_type)
    if generator:
        return generator.generate(spacetime, priority, **kwargs)
    return None


# ============================================================================
# Backward-Compatible Panel Generator Functions (Thin Wrappers)
# ============================================================================
# These functions delegate to the strategy pattern generators for backward
# compatibility. New code should use generate_panel() or the generator classes.

def generate_text_panel(
    spacetime: Spacetime,
    priority: int = 0,
    size: PanelSizeJenny = PanelSizeJenny.LARGE
) -> JennySpec:
    """Generate a text panel for the main response."""
    return PANEL_GENERATORS[PanelTypeJenny.TEXT].generate(spacetime, priority, size=size)


def generate_confidence_panel(
    spacetime: Spacetime,
    priority: int = 1,
) -> JennySpec:
    """Generate a confidence gauge panel."""
    return PANEL_GENERATORS[PanelTypeJenny.CONFIDENCE].generate(spacetime, priority)


def generate_sources_panel(
    spacetime: Spacetime,
    priority: int = 2,
) -> Optional[JennySpec]:
    """Generate a sources attribution panel."""
    return PANEL_GENERATORS[PanelTypeJenny.SOURCES].generate(spacetime, priority)


def generate_graph_panel(
    spacetime: Spacetime,
    priority: int = 3,
) -> Optional[JennySpec]:
    """Generate a knowledge graph panel showing activated threads."""
    return PANEL_GENERATORS[PanelTypeJenny.GRAPH].generate(spacetime, priority)


def generate_timeline_panel(
    spacetime: Spacetime,
    priority: int = 4,
) -> Optional[JennySpec]:
    """Generate a stage timeline panel showing execution phases."""
    return PANEL_GENERATORS[PanelTypeJenny.TIMELINE].generate(spacetime, priority)


def generate_reasoning_panel(
    spacetime: Spacetime,
    priority: int = 5,
    verification_results: Optional[Dict[str, Any]] = None,
) -> Optional[JennySpec]:
    """
    Generate a step-by-step reasoning chain panel.

    Enhanced (December 2025 - Task 3.2) to include:
    - Numbered reasoning steps with confidence scores
    - Visual flow diagram (mermaid and simple text formats)
    - Expandable step details per step
    - Verification status per step

    Args:
        spacetime: The woven output
        priority: Panel priority (default: 5)
        verification_results: Optional verification results from agentic reasoning
            Expected format:
            {
                "verified": bool,
                "claims_checked": int,
                "claims_verified": int,
                "duration_ms": float,
                "details": str,
                "step_verifications": {
                    "query_analysis": {"status": ..., "checked": ..., "passed": ...},
                    ...
                }
            }

    Returns:
        JennySpec with rich reasoning visualization, or None if no reasoning available
    """
    return PANEL_GENERATORS[PanelTypeJenny.REASONING].generate(
        spacetime, priority, verification_results=verification_results
    )


def generate_metric_panel(
    spacetime: Spacetime,
    metric_name: str,
    metric_value: Any,
    priority: int = 6,
) -> JennySpec:
    """Generate a single metric panel."""
    return PANEL_GENERATORS[PanelTypeJenny.METRIC].generate(
        spacetime, priority, metric_name=metric_name, metric_value=metric_value
    )


def generate_why_panel(
    spacetime: Spacetime,
    specs_generated: List[JennySpec],
    analysis: QueryAnalysis,
    priority: int = 99,
    thompson_stats: Optional[Dict[str, Any]] = None,
) -> JennySpec:
    """
    Generate a 'Why this UI?' meta-panel (SYSTEM stage to break infinite loop).

    Enhanced (December 2025) to include:
    - Decision explanation with confidence
    - Alternatives considered with scores
    - Customization hints per panel
    - Thompson Sampling statistics (when available)

    Args:
        spacetime: The woven output
        specs_generated: List of panels that were generated
        analysis: Query analysis results
        priority: Panel priority (default: 99, lowest)
        thompson_stats: Optional Thompson Sampling statistics from panel learner

    Returns:
        JennySpec with rich explanatory content
    """
    return PANEL_GENERATORS[PanelTypeJenny.WHY].generate(
        spacetime, priority,
        specs_generated=specs_generated,
        analysis=analysis,
        thompson_stats=thompson_stats
    )


def generate_comparison_panel(
    spacetime: Spacetime,
    priority: int = 3,
    left: Optional[Dict[str, Any]] = None,
    right: Optional[Dict[str, Any]] = None,
    diff_type: str = "side_by_side",
    category: str = "text",
) -> Optional[JennySpec]:
    """
    Generate a side-by-side comparison panel.

    Added (December 2025 - Task 3.3) for comparing:
    - Before/after code changes
    - Two memory snapshots
    - Multiple query responses
    - Configuration differences

    Args:
        spacetime: The woven output
        priority: Panel priority (default: 3)
        left: Left side content with 'title' and 'content' keys
        right: Right side content with 'title' and 'content' keys
        diff_type: Type of diff visualization:
            - "unified": Standard unified diff format
            - "side_by_side": Two columns for direct comparison
            - "inline": Changes highlighted within the text
            - "semantic": Compares meaning rather than text
        category: Category of comparison:
            - "code": Code with syntax highlighting
            - "text": Plain text comparison
            - "memory": Memory snapshot comparison
            - "response": Query response comparison
            - "config": Configuration file comparison

    Returns:
        JennySpec with comparison visualization, or None if insufficient data
    """
    return PANEL_GENERATORS[PanelTypeJenny.COMPARISON].generate(
        spacetime, priority,
        left=left,
        right=right,
        diff_type=diff_type,
        category=category
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
            spacetime: Woven output from hololoom weaving cycle
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
    # Query Analysis
    'QueryAnalysis',
    'analyze_query',
    # Compiler
    'JennyCompiler',
    'create_jenny_compiler',
    # Strategy Pattern - Base Class & Registry
    'PanelGenerator',
    'PANEL_GENERATORS',
    'get_panel_generator',
    'generate_panel',
    # Strategy Pattern - Generator Classes
    'TextPanelGenerator',
    'ConfidencePanelGenerator',
    'SourcesPanelGenerator',
    'GraphPanelGenerator',
    'TimelinePanelGenerator',
    'ReasoningPanelGenerator',
    'MetricPanelGenerator',
    'WhyPanelGenerator',
    'ComparisonPanelGenerator',
    # Backward-Compatible Functions (thin wrappers)
    'generate_text_panel',
    'generate_confidence_panel',
    'generate_sources_panel',
    'generate_graph_panel',
    'generate_timeline_panel',
    'generate_reasoning_panel',
    'generate_metric_panel',
    'generate_why_panel',
    'generate_comparison_panel',
]
