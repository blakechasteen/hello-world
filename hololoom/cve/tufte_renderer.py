"""
TufteRenderer - 2D SVG/CSS Cognitive Visualization Renderer
==========================================================

Implements CognitiveRenderer protocol using HoloLoom's existing
Tufte-style visualizations (pure HTML/CSS/SVG, zero dependencies).

This is the "test in 2D" renderer - renders cognitive events to
static HTML that can be viewed in any browser.

Wraps existing visualizations:
- KnowledgeGraphRenderer → KNOWLEDGE_GRAPH
- StageWaterfallRenderer → WEAVING_CYCLE
- ConfidenceTrajectory → CONFIDENCE_FLOW
- SemanticSpaceRenderer → SEMANTIC_POSITION, SEMANTIC_TRAJECTORY
- CacheGauge → MEMORY_RETRIEVAL (cache effectiveness)
- Small Multiples → Multiple events comparison

Created: 2025-11-30
Author: HoloLoom Team
"""

from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
import json

from hololoom.cve.cognitive_protocol import (
    CognitiveEvent,
    CognitiveFrame,
    CognitiveStream,
    CognitiveVizType,
    CognitiveRenderer,
)


# ============================================================================
# TufteRenderer Implementation
# ============================================================================

class TufteRenderer:
    """
    2D Tufte-style renderer implementing CognitiveRenderer protocol.

    Design Principles (Edward Tufte):
    - Maximize data-ink ratio (~65-75%)
    - Direct labeling (no legend lookup)
    - High data density (16-24x more visible data)
    - Meaning first (critical info highlighted)
    - Zero external dependencies (pure HTML/CSS/SVG)
    """

    # Color palette for cognitive visualizations (semantic colors)
    COLORS = {
        "activation": "#3b82f6",      # Blue - spreading activation
        "probability": "#8b5cf6",     # Purple - probability distributions
        "semantic": "#10b981",        # Green - semantic space
        "decision": "#f59e0b",        # Orange - decisions/choices
        "memory": "#06b6d4",          # Cyan - memory operations
        "confidence": "#6366f1",      # Indigo - confidence levels
        "warning": "#ef4444",         # Red - warnings/errors
        "success": "#22c55e",         # Green - success
        "neutral": "#6b7280",         # Gray - neutral
        "background": "#f9fafb",      # Light gray - backgrounds
    }

    # Viz type to renderer method mapping
    VIZ_RENDERERS = {
        CognitiveVizType.ACTIVATION_SPREAD: "_render_activation_spread",
        CognitiveVizType.PROBABILITY_MANIFOLD: "_render_probability_manifold",
        CognitiveVizType.SEMANTIC_POSITION: "_render_semantic_position",
        CognitiveVizType.SEMANTIC_TRAJECTORY: "_render_semantic_trajectory",
        CognitiveVizType.KNOWLEDGE_GRAPH: "_render_knowledge_graph",
        CognitiveVizType.TOKEN_STREAM: "_render_token_stream",
        CognitiveVizType.DECISION_COLLAPSE: "_render_decision_collapse",
        CognitiveVizType.ATTENTION_FLOW: "_render_attention_flow",
        CognitiveVizType.REASONING_CHAIN: "_render_reasoning_chain",
        CognitiveVizType.MEMORY_RETRIEVAL: "_render_memory_retrieval",
        CognitiveVizType.CONFIDENCE_FLOW: "_render_confidence_flow",
        CognitiveVizType.TOOL_SELECTION: "_render_tool_selection",
        CognitiveVizType.CONTEXT_EXPANSION: "_render_context_expansion",
        CognitiveVizType.WEAVING_CYCLE: "_render_weaving_cycle",
        CognitiveVizType.REFLECTION_LOOP: "_render_reflection_loop",
    }

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
        show_timestamps: bool = True,
        dark_mode: bool = False,
    ):
        """
        Initialize TufteRenderer.

        Args:
            width: Default visualization width (px)
            height: Default visualization height (px)
            show_timestamps: Include timestamp in output
            dark_mode: Use dark color scheme
        """
        self.width = width
        self.height = height
        self.show_timestamps = show_timestamps
        self.dark_mode = dark_mode

        # Adjust colors for dark mode
        if dark_mode:
            self.COLORS = {
                **self.COLORS,
                "background": "#1f2937",
                "neutral": "#9ca3af",
            }

    # =========================================================================
    # CognitiveRenderer Protocol Implementation
    # =========================================================================

    def render_event(self, event: CognitiveEvent) -> str:
        """
        Render single cognitive event to HTML.

        Args:
            event: CognitiveEvent to render

        Returns:
            HTML string
        """
        renderer_method = self.VIZ_RENDERERS.get(event.viz_type)
        if renderer_method:
            method = getattr(self, renderer_method)
            return method(event)
        else:
            return self._render_unknown(event)

    def render_frame(self, frame: CognitiveFrame) -> str:
        """
        Render complete cognitive frame to HTML.

        Args:
            frame: CognitiveFrame containing multiple events

        Returns:
            Complete HTML document
        """
        # Render each event
        event_htmls = [self.render_event(e) for e in frame.events]

        # Frame header
        header = self._render_frame_header(frame)

        # Combine into document
        bg_color = self.COLORS["background"]
        text_color = "#ffffff" if self.dark_mode else "#1f2937"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cognitive Frame - {frame.weaving_stage or 'Unknown Stage'}</title>
    {self._render_base_styles()}
</head>
<body style="background: {bg_color}; color: {text_color}; margin: 0; padding: 20px;">
    <div class="cognitive-frame" style="max-width: 1200px; margin: 0 auto;">
        {header}
        <div class="events-container" style="display: flex; flex-direction: column; gap: 20px;">
            {''.join(event_htmls)}
        </div>
    </div>
</body>
</html>"""

    async def stream_render(self, stream: CognitiveStream) -> AsyncIterator[str]:
        """
        Stream render frames as they arrive.

        Args:
            stream: CognitiveStream of frames

        Yields:
            HTML strings for each frame
        """
        async for frame in stream:
            yield self.render_frame(frame)

    def supports_viz_type(self, viz_type: CognitiveVizType) -> bool:
        """Check if renderer supports a visualization type."""
        return viz_type in self.VIZ_RENDERERS

    def get_supported_types(self) -> List[CognitiveVizType]:
        """Get all supported visualization types."""
        return list(self.VIZ_RENDERERS.keys())

    # =========================================================================
    # Base Styles & Headers
    # =========================================================================

    def _render_base_styles(self) -> str:
        """Render base CSS styles."""
        return """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

    .event-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .event-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
    }

    .event-title {
        font-size: 14px;
        font-weight: 600;
        color: #374151;
    }

    .event-type {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 2px 8px;
        border-radius: 4px;
        background: #f3f4f6;
        color: #6b7280;
    }

    .metric-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }

    .metric {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .metric-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 18px;
        font-weight: 600;
        color: #111827;
        font-variant-numeric: tabular-nums;
    }

    .progress-bar {
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .sparkline {
        display: inline-block;
        vertical-align: middle;
    }

    .node-list {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-top: 8px;
    }

    .node-tag {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 12px;
        background: #dbeafe;
        color: #1d4ed8;
    }

    .node-tag.active {
        background: #3b82f6;
        color: white;
    }
</style>"""

    def _render_frame_header(self, frame: CognitiveFrame) -> str:
        """Render frame header with metadata."""
        timestamp = frame.timestamp.strftime("%H:%M:%S.%f")[:-3]
        stage = frame.weaving_stage or "Unknown"
        confidence = f"{frame.confidence:.1%}" if frame.confidence else "-"

        return f"""
        <div class="frame-header" style="margin-bottom: 24px; padding-bottom: 16px; border-bottom: 2px solid #e5e7eb;">
            <h1 style="font-size: 20px; font-weight: 600; color: #111827; margin-bottom: 8px;">
                Cognitive Frame
            </h1>
            <div style="display: flex; gap: 24px; font-size: 13px; color: #6b7280;">
                <span><strong>Stage:</strong> {stage}</span>
                <span><strong>Events:</strong> {len(frame.events)}</span>
                <span><strong>Confidence:</strong> {confidence}</span>
                <span><strong>Time:</strong> {timestamp}</span>
            </div>
        </div>
        """

    # =========================================================================
    # Individual Event Renderers (15 types)
    # =========================================================================

    def _render_activation_spread(self, event: CognitiveEvent) -> str:
        """Render spreading activation visualization."""
        data = event.data
        source_nodes = data.get("source_nodes", [])
        activation_levels = data.get("activation_levels", {})
        wave_front = data.get("wave_front", [])
        step = data.get("propagation_step", 0)

        # Sort by activation level
        sorted_nodes = sorted(
            activation_levels.items(),
            key=lambda x: x[1],
            reverse=True
        )[:12]  # Top 12 nodes

        # Build activation bars
        bars_html = ""
        for node_id, level in sorted_nodes:
            is_source = node_id in source_nodes
            is_front = node_id in wave_front
            color = "#22c55e" if is_source else ("#3b82f6" if is_front else "#6b7280")
            width_pct = level * 100

            bars_html += f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                <div style="width: 100px; font-size: 11px; color: #4b5563; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {node_id[:15]}
                </div>
                <div style="flex: 1; max-width: 200px;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {width_pct:.1f}%; background: {color};"></div>
                    </div>
                </div>
                <div style="width: 40px; font-size: 11px; font-weight: 600; color: {color};">
                    {level:.2f}
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Spreading Activation</span>
                <span class="event-type" style="background: #dbeafe; color: #1d4ed8;">ACTIVATION</span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Step</span>
                    <span class="metric-value">{step}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sources</span>
                    <span class="metric-value">{len(source_nodes)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Wave Front</span>
                    <span class="metric-value">{len(wave_front)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Nodes</span>
                    <span class="metric-value">{len(activation_levels)}</span>
                </div>
            </div>
            <div style="font-size: 12px; color: #6b7280; margin-bottom: 8px;">
                Activation Levels (top 12):
            </div>
            {bars_html}
        </div>
        """

    def _render_probability_manifold(self, event: CognitiveEvent) -> str:
        """Render Thompson Sampling probability visualization."""
        data = event.data
        tools = data.get("tools", [])
        expected = data.get("expected", [])
        variances = data.get("variances", [])
        selected = data.get("selected")
        alphas = data.get("alphas", [])
        betas = data.get("betas", [])

        # Build probability bars
        bars_html = ""
        max_exp = max(expected) if expected else 1.0

        for i, tool in enumerate(tools):
            exp_val = expected[i] if i < len(expected) else 0.0
            var_val = variances[i] if i < len(variances) else 0.0
            alpha = alphas[i] if i < len(alphas) else 1.0
            beta = betas[i] if i < len(betas) else 1.0

            is_selected = tool == selected
            color = "#22c55e" if is_selected else "#8b5cf6"
            width_pct = (exp_val / max_exp) * 100 if max_exp > 0 else 0

            # Uncertainty indicator (wider = more uncertainty)
            uncertainty_width = min(var_val * 400, 20)

            bars_html += f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 6px 0;
                        {'background: #f0fdf4; border-radius: 4px; padding: 4px;' if is_selected else ''}">
                <div style="width: 80px; font-size: 12px; font-weight: {'600' if is_selected else '400'};
                            color: {'#15803d' if is_selected else '#4b5563'};">
                    {tool}
                </div>
                <div style="flex: 1; max-width: 200px; position: relative;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {width_pct:.1f}%; background: {color};"></div>
                    </div>
                    <div style="position: absolute; top: 50%; left: {width_pct:.1f}%;
                                transform: translate(-50%, -50%);
                                width: {uncertainty_width}px; height: 16px;
                                background: rgba(139,92,246,0.2); border-radius: 2px;"></div>
                </div>
                <div style="width: 50px; font-size: 11px; color: #6b7280;">
                    E={exp_val:.2f}
                </div>
                <div style="width: 80px; font-size: 10px; color: #9ca3af;">
                    a={alpha:.0f} b={beta:.0f}
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Thompson Sampling Manifold</span>
                <span class="event-type" style="background: #ede9fe; color: #6d28d9;">PROBABILITY</span>
            </div>
            <div style="margin-bottom: 12px;">
                <span style="font-size: 12px; color: #6b7280;">
                    Beta distributions for tool selection.
                    Width = uncertainty (variance).
                </span>
            </div>
            {bars_html}
            <div style="margin-top: 12px; font-size: 12px;">
                <strong style="color: #15803d;">Selected:</strong>
                <span style="background: #dcfce7; padding: 2px 8px; border-radius: 4px; font-weight: 500;">
                    {selected or 'None'}
                </span>
            </div>
        </div>
        """

    def _render_semantic_position(self, event: CognitiveEvent) -> str:
        """Render semantic space position visualization."""
        data = event.data
        interpretable = data.get("interpretable", {})
        dominant = data.get("dominant_axes", [])
        query = data.get("query_text", "")

        # Build 16 axes mini-charts
        axes_html = ""
        for axis, value in list(interpretable.items())[:16]:
            # Normalize to 0-1 range (assuming values are roughly -1 to 1)
            normalized = (value + 1) / 2
            width_pct = max(0, min(100, normalized * 100))

            # Color based on value
            if normalized > 0.6:
                color = "#22c55e"
            elif normalized < 0.4:
                color = "#ef4444"
            else:
                color = "#6b7280"

            axes_html += f"""
            <div style="display: flex; align-items: center; gap: 4px;">
                <div style="width: 70px; font-size: 9px; color: #6b7280; text-align: right;">
                    {axis[:10]}
                </div>
                <div style="width: 60px; height: 6px; background: #e5e7eb; border-radius: 3px;">
                    <div style="width: {width_pct:.0f}%; height: 100%; background: {color}; border-radius: 3px;"></div>
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Semantic Position (228D)</span>
                <span class="event-type" style="background: #d1fae5; color: #047857;">SEMANTIC</span>
            </div>
            {f'<div style="font-size: 12px; color: #4b5563; margin-bottom: 12px; padding: 8px; background: #f9fafb; border-radius: 4px;">Query: "{query[:100]}"</div>' if query else ''}
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 8px;">
                16 Interpretable Axes (of 228 total):
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;">
                {axes_html}
            </div>
            {self._render_dominant_axes(dominant)}
        </div>
        """

    def _render_dominant_axes(self, dominant: List) -> str:
        """Render dominant semantic axes tags."""
        if not dominant:
            return ""

        tags = "".join([
            f'<span class="node-tag" style="background: #dcfce7; color: #15803d;">{axis}: {val:.2f}</span>'
            for axis, val in dominant[:5]
        ])

        return f"""
        <div style="margin-top: 12px;">
            <span style="font-size: 11px; color: #6b7280;">Dominant:</span>
            <div class="node-list">{tags}</div>
        </div>
        """

    def _render_semantic_trajectory(self, event: CognitiveEvent) -> str:
        """Render semantic space trajectory visualization."""
        data = event.data
        length = data.get("length", 0)
        velocities = data.get("velocities", [])
        shift_points = data.get("shift_points", [])

        avg_velocity = sum(velocities) / len(velocities) if velocities else 0

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Semantic Trajectory</span>
                <span class="event-type" style="background: #d1fae5; color: #047857;">TRAJECTORY</span>
            </div>
            <div class="metric-row">
                <div class="metric">
                    <span class="metric-label">Path Length</span>
                    <span class="metric-value">{length}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Velocity</span>
                    <span class="metric-value">{avg_velocity:.3f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Shift Points</span>
                    <span class="metric-value">{len(shift_points)}</span>
                </div>
            </div>
        </div>
        """

    def _render_knowledge_graph(self, event: CognitiveEvent) -> str:
        """Render knowledge graph summary (full graph uses dedicated renderer)."""
        data = event.data
        node_count = data.get("node_count", 0)
        edge_count = data.get("edge_count", 0)
        layout = data.get("layout", "force_directed")
        highlighted = data.get("highlighted_path", [])

        nodes = data.get("nodes", [])[:10]  # Show first 10
        node_tags = "".join([
            f'<span class="node-tag">{n.get("label", n.get("id", "?"))[:15]}</span>'
            for n in nodes
        ])

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Knowledge Graph</span>
                <span class="event-type" style="background: #dbeafe; color: #1d4ed8;">GRAPH</span>
            </div>
            <div class="metric-row">
                <div class="metric">
                    <span class="metric-label">Nodes</span>
                    <span class="metric-value">{node_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Edges</span>
                    <span class="metric-value">{edge_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Layout</span>
                    <span class="metric-value" style="font-size: 12px;">{layout}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Path</span>
                    <span class="metric-value">{len(highlighted)}</span>
                </div>
            </div>
            <div style="margin-top: 12px;">
                <span style="font-size: 11px; color: #6b7280;">Sample nodes:</span>
                <div class="node-list">{node_tags}</div>
            </div>
        </div>
        """

    def _render_token_stream(self, event: CognitiveEvent) -> str:
        """Render token stream visualization."""
        data = event.data
        tokens = data.get("tokens", [])
        cumulative = data.get("cumulative", "")
        is_final = data.get("is_final", False)
        speed = data.get("speed", 0)

        # Show last 20 tokens with animation hint
        recent_tokens = tokens[-20:] if len(tokens) > 20 else tokens
        tokens_html = "".join([
            f'<span style="display: inline-block; padding: 2px 4px; margin: 1px; '
            f'background: #dbeafe; color: #1d4ed8; border-radius: 2px; font-size: 11px; '
            f'font-family: monospace;">{t}</span>'
            for t in recent_tokens
        ])

        status_color = "#22c55e" if is_final else "#f59e0b"
        status_text = "Complete" if is_final else "Streaming..."

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Token Stream</span>
                <span class="event-type" style="background: {status_color}20; color: {status_color};">
                    {status_text}
                </span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Tokens</span>
                    <span class="metric-value">{len(tokens)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Speed</span>
                    <span class="metric-value">{speed:.1f}/s</span>
                </div>
            </div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">
                Recent tokens:
            </div>
            <div style="padding: 8px; background: #f9fafb; border-radius: 4px; max-height: 100px; overflow-y: auto;">
                {tokens_html}
            </div>
        </div>
        """

    def _render_decision_collapse(self, event: CognitiveEvent) -> str:
        """Render decision collapse visualization."""
        data = event.data
        options = data.get("options", [])
        probabilities = data.get("probabilities", [])
        selected = data.get("selected", "")
        strategy = data.get("strategy", "unknown")
        confidence = data.get("confidence", 0)

        # Build collapse bars
        bars_html = ""
        for i, opt in enumerate(options):
            prob = probabilities[i] if i < len(probabilities) else 0
            is_selected = opt == selected
            color = "#22c55e" if is_selected else "#e5e7eb"
            text_color = "#15803d" if is_selected else "#6b7280"
            width_pct = prob * 100

            bars_html += f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                <div style="width: 80px; font-size: 12px; font-weight: {'600' if is_selected else '400'}; color: {text_color};">
                    {opt}
                </div>
                <div style="flex: 1; max-width: 200px;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {width_pct:.1f}%; background: {color};"></div>
                    </div>
                </div>
                <div style="width: 50px; font-size: 11px; color: {text_color};">
                    {prob:.1%}
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Decision Collapse</span>
                <span class="event-type" style="background: #fef3c7; color: #b45309;">DECISION</span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Strategy</span>
                    <span class="metric-value" style="font-size: 12px;">{strategy}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value">{confidence:.1%}</span>
                </div>
            </div>
            {bars_html}
            <div style="margin-top: 12px; padding: 8px; background: #f0fdf4; border-radius: 4px; text-align: center;">
                <span style="font-size: 11px; color: #6b7280;">Collapsed to: </span>
                <strong style="color: #15803d;">{selected}</strong>
            </div>
        </div>
        """

    def _render_attention_flow(self, event: CognitiveEvent) -> str:
        """Render attention flow visualization."""
        data = event.data
        query_tokens = data.get("query_tokens", [])
        memory_keys = data.get("memory_keys", [])
        head_index = data.get("head_index", 0)

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Attention Flow (Head {head_index})</span>
                <span class="event-type" style="background: #fce7f3; color: #be185d;">ATTENTION</span>
            </div>
            <div class="metric-row">
                <div class="metric">
                    <span class="metric-label">Query Tokens</span>
                    <span class="metric-value">{len(query_tokens)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Keys</span>
                    <span class="metric-value">{len(memory_keys)}</span>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 11px; color: #6b7280;">
                (Full attention matrix visualization available in WebGL renderer)
            </div>
        </div>
        """

    def _render_reasoning_chain(self, event: CognitiveEvent) -> str:
        """Render reasoning chain visualization."""
        data = event.data
        steps = data.get("steps", [])
        current = data.get("current", 0)
        mode = data.get("mode", "direct")
        confidence = data.get("confidence", 0)

        # Build steps timeline
        steps_html = ""
        for i, step in enumerate(steps[:5]):  # Show first 5 steps
            is_current = i == current
            is_complete = i < current
            color = "#22c55e" if is_complete else ("#3b82f6" if is_current else "#e5e7eb")
            text_color = "#15803d" if is_complete else ("#1d4ed8" if is_current else "#9ca3af")

            query = step.get("query", "")[:50]
            step_conf = step.get("confidence", 0)

            steps_html += f"""
            <div style="display: flex; align-items: flex-start; gap: 8px; margin: 8px 0;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: {color};
                            display: flex; align-items: center; justify-content: center;
                            font-size: 11px; font-weight: 600; color: white;">
                    {i+1}
                </div>
                <div style="flex: 1;">
                    <div style="font-size: 12px; color: {text_color};">
                        {query}...
                    </div>
                    <div style="font-size: 10px; color: #9ca3af;">
                        confidence: {step_conf:.1%}
                    </div>
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Reasoning Chain</span>
                <span class="event-type" style="background: #fee2e2; color: #b91c1c;">REASONING</span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Mode</span>
                    <span class="metric-value" style="font-size: 12px;">{mode}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Steps</span>
                    <span class="metric-value">{len(steps)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current</span>
                    <span class="metric-value">{current + 1}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value">{confidence:.1%}</span>
                </div>
            </div>
            {steps_html}
        </div>
        """

    def _render_memory_retrieval(self, event: CognitiveEvent) -> str:
        """Render memory retrieval visualization."""
        data = event.data
        query = data.get("query", "")
        retrieved = data.get("retrieved", [])
        scores = data.get("scores", [])
        method = data.get("method", "hybrid")
        total = data.get("total_candidates", 0)

        # Build retrieval list
        items_html = ""
        for i, (mem_id, score) in enumerate(zip(retrieved[:8], scores[:8])):
            width_pct = score * 100
            items_html += f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                <div style="width: 120px; font-size: 11px; color: #4b5563; overflow: hidden;
                            text-overflow: ellipsis; white-space: nowrap;">
                    {mem_id[:20]}
                </div>
                <div style="flex: 1; max-width: 150px;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {width_pct:.1f}%; background: #06b6d4;"></div>
                    </div>
                </div>
                <div style="width: 40px; font-size: 11px; color: #06b6d4;">
                    {score:.2f}
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Memory Retrieval</span>
                <span class="event-type" style="background: #cffafe; color: #0891b2;">MEMORY</span>
            </div>
            <div style="font-size: 12px; color: #4b5563; margin-bottom: 12px; padding: 8px;
                        background: #f9fafb; border-radius: 4px;">
                Query: "{query[:80]}"
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Method</span>
                    <span class="metric-value" style="font-size: 12px;">{method}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Retrieved</span>
                    <span class="metric-value">{len(retrieved)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Candidates</span>
                    <span class="metric-value">{total}</span>
                </div>
            </div>
            {items_html}
        </div>
        """

    def _render_confidence_flow(self, event: CognitiveEvent) -> str:
        """Render confidence trajectory visualization."""
        data = event.data
        confidences = data.get("confidences", [])
        cache_hits = data.get("cache_hits", [])
        anomalies = data.get("anomalies", [])

        if not confidences:
            return self._render_empty_card("Confidence Flow", "No data")

        # Generate sparkline SVG
        sparkline = self._generate_sparkline(confidences)

        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        cache_hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Confidence Flow</span>
                <span class="event-type" style="background: #e0e7ff; color: #4338ca;">CONFIDENCE</span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Avg</span>
                    <span class="metric-value">{avg_conf:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Min</span>
                    <span class="metric-value">{min_conf:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max</span>
                    <span class="metric-value">{max_conf:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Cache Hit</span>
                    <span class="metric-value">{cache_hit_rate:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Anomalies</span>
                    <span class="metric-value" style="color: {'#ef4444' if anomalies else '#22c55e'};">
                        {len(anomalies)}
                    </span>
                </div>
            </div>
            <div style="padding: 12px; background: #f9fafb; border-radius: 4px;">
                {sparkline}
            </div>
        </div>
        """

    def _render_tool_selection(self, event: CognitiveEvent) -> str:
        """Render tool selection visualization."""
        data = event.data
        tools = data.get("tools", [])
        probs = data.get("probabilities", [])
        selected = data.get("selected", "")

        # Build horizontal bar chart
        bars_html = ""
        max_prob = max(probs) if probs else 1.0

        for i, tool in enumerate(tools):
            prob = probs[i] if i < len(probs) else 0
            is_selected = tool == selected
            color = "#22c55e" if is_selected else "#6366f1"
            width_pct = (prob / max_prob) * 100 if max_prob > 0 else 0

            bars_html += f"""
            <div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
                <div style="width: 70px; font-size: 12px; font-weight: {'600' if is_selected else '400'};
                            color: {'#15803d' if is_selected else '#4b5563'};">
                    {tool}
                </div>
                <div style="flex: 1; max-width: 200px;">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {width_pct:.1f}%; background: {color};"></div>
                    </div>
                </div>
                <div style="width: 50px; font-size: 11px; color: #6b7280;">
                    {prob:.1%}
                </div>
            </div>
            """

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Tool Selection</span>
                <span class="event-type" style="background: #fef3c7; color: #b45309;">TOOL</span>
            </div>
            {bars_html}
        </div>
        """

    def _render_context_expansion(self, event: CognitiveEvent) -> str:
        """Render context expansion visualization."""
        data = event.data
        seeds = data.get("seeds", [])
        expanded = data.get("expanded", [])
        budget = data.get("budget", 0)
        used = data.get("used", 0)
        strategy = data.get("strategy", "adaptive")

        usage_pct = (used / budget) * 100 if budget > 0 else 0
        color = "#22c55e" if usage_pct < 70 else ("#f59e0b" if usage_pct < 90 else "#ef4444")

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Context Expansion</span>
                <span class="event-type" style="background: #fae8ff; color: #a21caf;">CONTEXT</span>
            </div>
            <div class="metric-row" style="margin-bottom: 12px;">
                <div class="metric">
                    <span class="metric-label">Strategy</span>
                    <span class="metric-value" style="font-size: 12px;">{strategy}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Seeds</span>
                    <span class="metric-value">{len(seeds)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Expanded</span>
                    <span class="metric-value">{len(expanded)}</span>
                </div>
            </div>
            <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">
                Token Budget: {used} / {budget} ({usage_pct:.0f}%)
            </div>
            <div class="progress-bar" style="height: 12px;">
                <div class="progress-fill" style="width: {usage_pct:.1f}%; background: {color};"></div>
            </div>
        </div>
        """

    def _render_weaving_cycle(self, event: CognitiveEvent) -> str:
        """Render 9-stage weaving cycle visualization."""
        data = event.data
        current = data.get("current_stage", 1)
        names = data.get("stage_names", [])
        statuses = data.get("statuses", [])
        durations = data.get("stage_durations", [])
        pattern = data.get("pattern_card", "FAST")

        # Build circular stage indicators
        stages_html = ""
        for i, name in enumerate(names):
            status = statuses[i] if i < len(statuses) else "pending"
            duration = durations[i] if i < len(durations) else 0

            if status == "complete":
                color = "#22c55e"
                bg = "#dcfce7"
            elif status == "active":
                color = "#3b82f6"
                bg = "#dbeafe"
            else:
                color = "#9ca3af"
                bg = "#f3f4f6"

            stages_html += f"""
            <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: {bg};
                            border: 2px solid {color}; display: flex; align-items: center;
                            justify-content: center; font-size: 12px; font-weight: 600; color: {color};">
                    {i + 1}
                </div>
                <div style="font-size: 9px; color: {color}; text-align: center; width: 60px;">
                    {name.split()[0]}
                </div>
                <div style="font-size: 8px; color: #9ca3af;">
                    {duration:.0f}ms
                </div>
            </div>
            """

        total_duration = sum(durations)

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Weaving Cycle</span>
                <span class="event-type" style="background: #fef3c7; color: #b45309;">{pattern}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;
                        flex-wrap: wrap; gap: 8px; margin-top: 12px;">
                {stages_html}
            </div>
            <div style="margin-top: 16px; text-align: center; font-size: 12px; color: #6b7280;">
                Current Stage: <strong style="color: #3b82f6;">{current}</strong> / 9
                &nbsp;|&nbsp;
                Total: <strong>{total_duration:.1f}ms</strong>
            </div>
        </div>
        """

    def _render_reflection_loop(self, event: CognitiveEvent) -> str:
        """Render reflection loop visualization."""
        data = event.data
        query = data.get("query", "")[:60]
        outcome = data.get("outcome", "unknown")
        confidence = data.get("confidence", 0)
        pattern = data.get("pattern_learned")
        policy_updated = data.get("policy_updated", False)

        outcome_color = {
            "success": "#22c55e",
            "failure": "#ef4444",
            "partial": "#f59e0b",
        }.get(outcome, "#6b7280")

        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Reflection Loop</span>
                <span class="event-type" style="background: {outcome_color}20; color: {outcome_color};">
                    {outcome.upper()}
                </span>
            </div>
            <div style="font-size: 12px; color: #4b5563; margin-bottom: 12px;">
                "{query}..."
            </div>
            <div class="metric-row">
                <div class="metric">
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value">{confidence:.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Pattern</span>
                    <span class="metric-value" style="font-size: 12px;">
                        {pattern[:15] if pattern else 'None'}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Policy Updated</span>
                    <span class="metric-value" style="color: {'#22c55e' if policy_updated else '#ef4444'};">
                        {'Yes' if policy_updated else 'No'}
                    </span>
                </div>
            </div>
        </div>
        """

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _render_unknown(self, event: CognitiveEvent) -> str:
        """Render unknown event type."""
        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">Unknown Event</span>
                <span class="event-type">{event.viz_type.value}</span>
            </div>
            <pre style="font-size: 11px; background: #f9fafb; padding: 8px; border-radius: 4px;
                        overflow-x: auto;">{json.dumps(event.data, indent=2, default=str)[:500]}</pre>
        </div>
        """

    def _render_empty_card(self, title: str, message: str) -> str:
        """Render empty state card."""
        return f"""
        <div class="event-card">
            <div class="event-header">
                <span class="event-title">{title}</span>
            </div>
            <div style="text-align: center; padding: 20px; color: #9ca3af;">
                {message}
            </div>
        </div>
        """

    def _generate_sparkline(
        self,
        values: List[float],
        width: int = 200,
        height: int = 40,
        color: str = "#6366f1"
    ) -> str:
        """Generate inline SVG sparkline."""
        if not values or len(values) < 2:
            return ""

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val or 1

        # Generate points
        points = []
        for i, v in enumerate(values):
            x = (i / (len(values) - 1)) * width
            y = height - ((v - min_val) / val_range) * (height - 4) - 2
            points.append(f"{x:.1f},{y:.1f}")

        polyline = " ".join(points)

        # First and last point markers
        first_y = height - ((values[0] - min_val) / val_range) * (height - 4) - 2
        last_y = height - ((values[-1] - min_val) / val_range) * (height - 4) - 2

        return f"""
        <svg class="sparkline" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <polyline fill="none" stroke="{color}" stroke-width="1.5"
                      points="{polyline}" />
            <circle cx="0" cy="{first_y:.1f}" r="2" fill="{color}" />
            <circle cx="{width}" cy="{last_y:.1f}" r="3" fill="{color}" />
        </svg>
        """


# ============================================================================
# Convenience Functions
# ============================================================================

def render_cognitive_event(event: CognitiveEvent, **kwargs) -> str:
    """
    Render a single cognitive event to HTML.

    Args:
        event: CognitiveEvent to render
        **kwargs: Passed to TufteRenderer constructor

    Returns:
        HTML string
    """
    renderer = TufteRenderer(**kwargs)
    return renderer.render_event(event)


def render_cognitive_frame(frame: CognitiveFrame, **kwargs) -> str:
    """
    Render a cognitive frame to complete HTML document.

    Args:
        frame: CognitiveFrame to render
        **kwargs: Passed to TufteRenderer constructor

    Returns:
        Complete HTML document
    """
    renderer = TufteRenderer(**kwargs)
    return renderer.render_frame(frame)


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    from hololoom.cve.cognitive_protocol import (
        ActivationField,
        ProbabilityManifold,
        SemanticPosition,
        WeavingCycle,
        create_frame,
    )

    print("=== TufteRenderer Demo ===\n")

    # Create sample events
    activation = ActivationField(
        source_nodes=["thompson_sampling"],
        activation_levels={
            "thompson_sampling": 1.0,
            "exploration": 0.85,
            "exploitation": 0.80,
            "bayesian": 0.65,
            "bandit": 0.50,
        },
        wave_front=["exploration", "exploitation"],
        propagation_step=2,
    )

    manifold = ProbabilityManifold(
        tool_names=["answer", "research", "explore", "clarify"],
        alpha_values=[12.0, 6.0, 4.0, 2.0],
        beta_values=[3.0, 4.0, 6.0, 8.0],
        selected_tool="answer",
    )

    position = SemanticPosition(
        position=[0.8, 0.2, 0.9, 0.7, 0.3, 0.5, 0.8, 0.4,
                  0.6, 0.7, 0.5, 0.3, 0.2, 0.8, 0.4, 0.6] + [0.0] * 212,
        dominant_axes=[("technicality", 0.9), ("certainty", 0.7), ("actionability", 0.8)],
        query_text="What is Thompson Sampling?",
    )

    cycle = WeavingCycle(
        current_stage=5,
        stage_statuses=["complete", "complete", "complete", "complete", "active",
                        "pending", "pending", "pending", "pending"],
        stage_durations=[12.5, 8.3, 25.1, 45.2, 30.0, 0, 0, 0, 0],
        pattern_card="FAST",
    )

    # Create frame
    frame = create_frame(
        events=[
            activation.to_event(),
            manifold.to_event(),
            position.to_event(),
            cycle.to_event(),
        ],
        query_id="demo_001",
        weaving_stage="decision",
        confidence=0.85,
    )

    # Render
    renderer = TufteRenderer()
    html = renderer.render_frame(frame)

    # Save
    output_path = "demo_tufte_renderer.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_path}")
    print(f"Events rendered: {len(frame.events)}")
    print(f"Supported viz types: {len(renderer.get_supported_types())}")
    print("\nOpen in browser to view!")
