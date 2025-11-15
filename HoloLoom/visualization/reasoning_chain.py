#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Chain Visualization - Tufte-Style Sequential Flow
============================================================

Visualize multi-step reasoning chains with evidence, confidence, and backtracking.
Follows Edward Tufte's principles: "Above all else show the data."

Author: Claude Code
Date: 2025-11-15
Phase: 4 (Visualization)
Principles: Edward Tufte - Maximize data-ink ratio, show meaning first
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

# Import reasoning types
try:
    from HoloLoom.reasoning.types import (
        ReasoningStep,
        ReasoningResult,
        ReasoningMode,
        StepType,
        STEP_TYPE_ICONS,
    )
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    # Fallback types for standalone usage
    class ReasoningMode(Enum):
        FAST = "fast"
        STANDARD = "standard"
        DEEP = "deep"

    class StepType(Enum):
        UNDERSTANDING = "understanding"
        EVIDENCE = "evidence"
        SYNTHESIS = "synthesis"
        VERIFICATION = "verification"
        BACKTRACK = "backtrack"
        PLANNING = "planning"
        CORRECTION = "correction"

    STEP_TYPE_ICONS = {
        StepType.UNDERSTANDING: "🧠",
        StepType.EVIDENCE: "🔍",
        StepType.SYNTHESIS: "🔗",
        StepType.VERIFICATION: "✓",
        StepType.BACKTRACK: "↩",
        StepType.PLANNING: "📋",
        StepType.CORRECTION: "🔧",
    }


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ChainMetrics:
    """
    Metrics for reasoning chain visualization.

    Attributes:
        total_steps: Total number of reasoning steps
        avg_confidence: Average confidence across all steps
        min_confidence: Minimum confidence in chain
        max_confidence: Maximum confidence in chain
        critical_steps: Number of steps with confidence < 0.5
        backtrack_steps: Number of backtracking steps
        verification_passed: Whether verification passed overall
    """
    total_steps: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    critical_steps: int
    backtrack_steps: int
    verification_passed: bool = True


# ============================================================================
# Reasoning Chain Renderer
# ============================================================================

class ReasoningChainRenderer:
    """
    Tufte-style renderer for reasoning chains.

    Features:
    - Sequential step flow (top to bottom)
    - Confidence sparklines per step
    - Evidence tooltips (collapsible)
    - Backtracking indicators
    - Critical step highlighting (confidence < 0.5)
    - Step type icons
    - Zero external dependencies (pure HTML/CSS/SVG)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        show_evidence: bool = True,
        show_sparklines: bool = True,
        compact_mode: bool = False
    ):
        """
        Initialize renderer.

        Args:
            confidence_threshold: Threshold for critical step highlighting
            show_evidence: Whether to show evidence sections
            show_sparklines: Whether to show confidence sparklines
            compact_mode: Use compact layout for many steps
        """
        self.confidence_threshold = confidence_threshold
        self.show_evidence = show_evidence
        self.show_sparklines = show_sparklines
        self.compact_mode = compact_mode

    def render(
        self,
        chain: List[Any],  # List[ReasoningStep] or compatible
        mode: Any,  # ReasoningMode or str
        title: str = "Reasoning Chain",
        subtitle: Optional[str] = None,
        show_metrics: bool = True
    ) -> str:
        """
        Render reasoning chain as HTML.

        Args:
            chain: List of reasoning steps
            mode: Reasoning mode used
            title: Visualization title
            subtitle: Optional subtitle
            show_metrics: Whether to show summary metrics

        Returns:
            HTML string
        """
        # Calculate metrics
        metrics = self._calculate_metrics(chain)

        # Start HTML
        html = ['<div class="reasoning-chain-container">']

        # Add styles
        html.append(self._get_styles())

        # Header
        html.append(self._render_header(title, subtitle, mode, metrics, show_metrics))

        # Steps
        html.append('<div class="reasoning-steps">')

        for i, step in enumerate(chain):
            html.append(self._render_step(step, i, len(chain), metrics))

        html.append('</div>')  # Close steps

        # Footer with timeline
        if self.show_sparklines:
            html.append(self._render_timeline(chain, metrics))

        html.append('</div>')  # Close container

        return '\n'.join(html)

    def _calculate_metrics(self, chain: List[Any]) -> ChainMetrics:
        """Calculate chain metrics."""
        if not chain:
            return ChainMetrics(
                total_steps=0,
                avg_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                critical_steps=0,
                backtrack_steps=0
            )

        confidences = [step.confidence for step in chain]

        # Count special steps
        critical_steps = sum(1 for c in confidences if c < self.confidence_threshold)
        backtrack_steps = sum(
            1 for step in chain
            if hasattr(step, 'step_type') and
               getattr(step.step_type, 'value', step.step_type) in ['backtrack', StepType.BACKTRACK]
        )

        return ChainMetrics(
            total_steps=len(chain),
            avg_confidence=sum(confidences) / len(confidences),
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            critical_steps=critical_steps,
            backtrack_steps=backtrack_steps,
            verification_passed=critical_steps == 0
        )

    def _render_header(
        self,
        title: str,
        subtitle: Optional[str],
        mode: Any,
        metrics: ChainMetrics,
        show_metrics: bool
    ) -> str:
        """Render header section."""
        mode_str = getattr(mode, 'value', str(mode))

        html = ['<div class="chain-header">']
        html.append(f'<h3>{title}</h3>')

        if subtitle:
            html.append(f'<p class="subtitle">{subtitle}</p>')

        # Mode badge
        mode_class = f'mode-{mode_str.lower()}'
        html.append(f'<span class="mode-badge {mode_class}">{mode_str.upper()}</span>')

        # Metrics summary
        if show_metrics:
            html.append('<div class="metrics-summary">')
            html.append(f'<span class="metric">Steps: {metrics.total_steps}</span>')
            html.append(f'<span class="metric">Confidence: {metrics.avg_confidence:.2f}</span>')

            if metrics.critical_steps > 0:
                html.append(f'<span class="metric critical">⚠ {metrics.critical_steps} critical</span>')

            if metrics.backtrack_steps > 0:
                html.append(f'<span class="metric backtrack">↩ {metrics.backtrack_steps} backtrack</span>')

            html.append('</div>')

        html.append('</div>')
        return '\n'.join(html)

    def _render_step(
        self,
        step: Any,
        index: int,
        total: int,
        metrics: ChainMetrics
    ) -> str:
        """Render single reasoning step."""
        # Determine step class
        step_classes = ['reasoning-step']

        if step.confidence < self.confidence_threshold:
            step_classes.append('critical')
        elif step.confidence >= 0.9:
            step_classes.append('excellent')

        # Check for backtrack
        if hasattr(step, 'step_type'):
            step_type_val = getattr(step.step_type, 'value', step.step_type)
            if step_type_val in ['backtrack', StepType.BACKTRACK]:
                step_classes.append('backtrack')
            elif step_type_val in ['correction', StepType.CORRECTION]:
                step_classes.append('correction')

        if self.compact_mode:
            step_classes.append('compact')

        # Start step
        html = [f'<div class="{" ".join(step_classes)}">']

        # Step header
        html.append(self._render_step_header(step, index, total))

        # Thought content
        html.append(f'<div class="thought">{step.thought}</div>')

        # Evidence (collapsible)
        if self.show_evidence and hasattr(step, 'evidence') and step.evidence:
            html.append(self._render_evidence(step.evidence))

        # Confidence indicator
        html.append(self._render_confidence_indicator(step.confidence))

        html.append('</div>')  # Close step

        return '\n'.join(html)

    def _render_step_header(self, step: Any, index: int, total: int) -> str:
        """Render step header with number, icon, and confidence."""
        # Get icon
        icon = "•"
        if hasattr(step, 'step_type'):
            step_type = step.step_type
            if hasattr(step_type, 'value'):
                # Find matching StepType
                for st in StepType:
                    if st.value == step_type.value:
                        icon = STEP_TYPE_ICONS.get(st, "•")
                        break
            else:
                icon = STEP_TYPE_ICONS.get(step_type, "•")

        html = '<div class="step-header">'
        html += f'<span class="step-num">{index + 1}/{total}</span>'
        html += f'<span class="step-icon">{icon}</span>'
        html += f'<span class="step-confidence">{step.confidence:.2f}</span>'
        html += '</div>'

        return html

    def _render_evidence(self, evidence: str) -> str:
        """Render evidence section (collapsible)."""
        # Truncate long evidence
        display_evidence = evidence
        if len(evidence) > 200:
            display_evidence = evidence[:197] + "..."

        html = '<details class="evidence">'
        html += '<summary>Evidence</summary>'
        html += f'<p>{display_evidence}</p>'
        html += '</details>'

        return html

    def _render_confidence_indicator(self, confidence: float) -> str:
        """Render confidence indicator bar."""
        # Color based on confidence
        if confidence >= 0.9:
            color = "#2ecc71"  # Green
        elif confidence >= 0.7:
            color = "#3498db"  # Blue
        elif confidence >= 0.5:
            color = "#f39c12"  # Amber
        else:
            color = "#e74c3c"  # Red

        width_pct = confidence * 100

        html = '<div class="confidence-bar">'
        html += f'<div class="confidence-fill" style="width: {width_pct}%; background: {color};"></div>'
        html += '</div>'

        return html

    def _render_timeline(self, chain: List[Any], metrics: ChainMetrics) -> str:
        """Render confidence timeline sparkline."""
        if len(chain) < 2:
            return ""

        confidences = [step.confidence for step in chain]

        # Generate sparkline SVG
        width = 400
        height = 40
        padding = 5

        points = []
        for i, conf in enumerate(confidences):
            x = padding + (i / (len(confidences) - 1)) * (width - 2 * padding)
            y = padding + (1 - conf) * (height - 2 * padding)
            points.append(f"{x},{y}")

        polyline = " ".join(points)

        html = '<div class="chain-timeline">'
        html += '<h4>Confidence Trajectory</h4>'
        html += f'<svg width="{width}" height="{height}" class="sparkline">'

        # Mean line
        mean_y = padding + (1 - metrics.avg_confidence) * (height - 2 * padding)
        html += f'<line x1="{padding}" y1="{mean_y}" x2="{width - padding}" y2="{mean_y}" '
        html += 'stroke="#ccc" stroke-width="1" stroke-dasharray="2,2"/>'

        # Confidence line
        html += f'<polyline points="{polyline}" fill="none" stroke="#3498db" stroke-width="2"/>'

        # Points
        for i, conf in enumerate(confidences):
            x = padding + (i / (len(confidences) - 1)) * (width - 2 * padding)
            y = padding + (1 - conf) * (height - 2 * padding)

            color = "#e74c3c" if conf < self.confidence_threshold else "#3498db"
            html += f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>'

        html += '</svg>'
        html += '</div>'

        return html

    def _get_styles(self) -> str:
        """Get CSS styles for reasoning chain."""
        return """
<style>
.reasoning-chain-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica', Arial, sans-serif;
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.chain-header {
    border-bottom: 2px solid #333;
    padding-bottom: 15px;
    margin-bottom: 20px;
}

.chain-header h3 {
    margin: 0 0 5px 0;
    font-size: 24px;
    font-weight: 600;
    color: #333;
}

.chain-header .subtitle {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #666;
}

.mode-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 10px;
}

.mode-fast {
    background: #2ecc71;
    color: white;
}

.mode-standard {
    background: #3498db;
    color: white;
}

.mode-deep {
    background: #9b59b6;
    color: white;
}

.metrics-summary {
    margin-top: 10px;
    font-size: 13px;
    color: #555;
}

.metrics-summary .metric {
    margin-right: 15px;
    font-family: 'SF Mono', Monaco, monospace;
}

.metrics-summary .critical {
    color: #e74c3c;
    font-weight: 600;
}

.metrics-summary .backtrack {
    color: #f39c12;
    font-weight: 600;
}

.reasoning-steps {
    margin: 20px 0;
}

.reasoning-step {
    margin-bottom: 20px;
    padding: 15px;
    border-left: 3px solid #3498db;
    background: #f8f9fa;
    border-radius: 0 4px 4px 0;
    transition: all 0.2s;
}

.reasoning-step:hover {
    background: #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.reasoning-step.critical {
    border-left-color: #e74c3c;
    background: #fee;
}

.reasoning-step.excellent {
    border-left-color: #2ecc71;
    background: #efe;
}

.reasoning-step.backtrack {
    border-left-color: #f39c12;
    background: #fef7e6;
}

.reasoning-step.correction {
    border-left-color: #e67e22;
    background: #fef0e6;
}

.reasoning-step.compact {
    padding: 10px;
    margin-bottom: 10px;
}

.step-header {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    font-size: 13px;
    color: #666;
}

.step-num {
    font-family: 'SF Mono', Monaco, monospace;
    font-weight: 600;
    margin-right: 8px;
    color: #999;
}

.step-icon {
    font-size: 16px;
    margin-right: 8px;
}

.step-confidence {
    margin-left: auto;
    font-family: 'SF Mono', Monaco, monospace;
    font-weight: 600;
    color: #333;
}

.thought {
    font-size: 15px;
    line-height: 1.6;
    color: #333;
    margin-bottom: 10px;
}

.evidence {
    margin-top: 10px;
    font-size: 13px;
}

.evidence summary {
    cursor: pointer;
    color: #3498db;
    font-weight: 500;
    user-select: none;
}

.evidence summary:hover {
    color: #2980b9;
}

.evidence p {
    margin: 8px 0 0 20px;
    padding: 8px;
    background: white;
    border-radius: 3px;
    color: #555;
    line-height: 1.5;
    font-size: 12px;
}

.confidence-bar {
    height: 4px;
    background: #e0e0e0;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
}

.confidence-fill {
    height: 100%;
    transition: width 0.3s;
}

.chain-timeline {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
}

.chain-timeline h4 {
    margin: 0 0 10px 0;
    font-size: 14px;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.sparkline {
    display: block;
    margin: 0 auto;
}
</style>
"""


# ============================================================================
# Convenience Function
# ============================================================================

def render_reasoning_chain(
    chain: List[Any],
    mode: Any = "standard",
    title: str = "Reasoning Chain",
    subtitle: Optional[str] = None,
    show_metrics: bool = True,
    show_evidence: bool = True,
    show_sparklines: bool = True,
    compact_mode: bool = False,
    confidence_threshold: float = 0.5
) -> str:
    """
    Render reasoning chain visualization.

    Args:
        chain: List of reasoning steps (ReasoningStep or compatible objects)
        mode: Reasoning mode (ReasoningMode enum or string)
        title: Visualization title
        subtitle: Optional subtitle
        show_metrics: Show summary metrics
        show_evidence: Show evidence sections
        show_sparklines: Show confidence timeline
        compact_mode: Use compact layout
        confidence_threshold: Threshold for critical highlighting

    Returns:
        HTML string

    Example:
        >>> from HoloLoom.reasoning.engine import ReasoningEngine
        >>> from HoloLoom.visualization.reasoning_chain import render_reasoning_chain
        >>>
        >>> # After running reasoning
        >>> result = await engine.reason(query, features, context)
        >>>
        >>> # Render visualization
        >>> html = render_reasoning_chain(
        ...     chain=result.chain,
        ...     mode=result.mode,
        ...     title="Query: How does Thompson Sampling work?",
        ...     subtitle=f"Duration: {result.duration_ms:.1f}ms"
        ... )
        >>>
        >>> # Save or display
        >>> with open('reasoning_chain.html', 'w') as f:
        ...     f.write(html)
    """
    renderer = ReasoningChainRenderer(
        confidence_threshold=confidence_threshold,
        show_evidence=show_evidence,
        show_sparklines=show_sparklines,
        compact_mode=compact_mode
    )

    return renderer.render(
        chain=chain,
        mode=mode,
        title=title,
        subtitle=subtitle,
        show_metrics=show_metrics
    )


def render_from_reasoning_result(
    result: Any,  # ReasoningResult
    title: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convenience function to render from ReasoningResult object.

    Args:
        result: ReasoningResult from reasoning engine
        title: Optional title override
        **kwargs: Additional render_reasoning_chain arguments

    Returns:
        HTML string

    Example:
        >>> result = await engine.reason(query, features, context)
        >>> html = render_from_reasoning_result(result)
    """
    if title is None and hasattr(result, 'metadata'):
        # Try to extract query from metadata
        if 'query_text' in result.metadata:
            title = f"Query: {result.metadata['query_text'][:50]}"

    subtitle = None
    if hasattr(result, 'duration_ms'):
        subtitle = f"Duration: {result.duration_ms:.1f}ms"
        if hasattr(result, 'total_confidence'):
            subtitle += f" | Confidence: {result.total_confidence:.2f}"

    return render_reasoning_chain(
        chain=result.chain,
        mode=result.mode,
        title=title or "Reasoning Chain",
        subtitle=subtitle,
        **kwargs
    )
