from __future__ import annotations
"""
Jenny Renderer Module
======================
Renders JennySpec panels into various output formats.

Philosophy:
> "Every panel is a view, every view is disposable."

Renderers are stateless transformations: JennySpec → Output String.
They handle visual presentation, animations, and accessibility.

Implementations:
- HTMLRenderer: Static HTML + CSS for dashboards
- TerminalRenderer: ASCII art for CLI/debugging
- (Future) ReactRenderer: React component props
- (Future) ARRenderer: Spatial computing overlays

Author: HoloLoom Team
Date: 2025-12-01 (Jenny MVP Week 2)
"""

import html
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from hololoom.protocols.jenny import RenderTarget

from .jenny_spec import (
    JennySpec,
    LifecycleStage,
    PanelSizeJenny,
    PanelTypeJenny,
)

# Try to import accessibility layer (Phase M3)
try:
    from .jenny_accessibility import (
        AriaAttributes,
        AriaRole,
        JennyAccessibilityLayer,
        create_accessibility_layer,
    )
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False
    JennyAccessibilityLayer = None
    AriaRole = None
    AriaAttributes = None

# Registry import (lazy to avoid circular imports)
_registry = None


def _get_registry():
    """Lazy registry import to avoid circular imports."""
    global _registry
    if _registry is None:
        from .jenny_renderer_registry import get_registry
        _registry = get_registry()
    return _registry


logger = logging.getLogger(__name__)


# ============================================================================
# Render Errors
# ============================================================================

class RenderError(Exception):
    """Raised when rendering fails."""
    pass


class UnsupportedTargetError(RenderError):
    """Raised when renderer doesn't support requested target."""
    pass


# ============================================================================
# Base Renderer Class
# ============================================================================

class JennyRendererBase(ABC):
    """
    Abstract base class for Jenny renderers.

    Provides common infrastructure and ensures protocol compliance.
    Subclasses implement format-specific rendering logic.
    """

    VERSION = "1.0.0"

    def __init__(self):
        """Initialize renderer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def supported_targets(self) -> list[RenderTarget]:
        """Return list of supported render targets."""
        ...

    def supports_target(self, target: RenderTarget) -> bool:
        """Check if renderer supports a target format."""
        return target in self.supported_targets

    @property
    def name(self) -> str:
        """
        Unique renderer name for registry identification.

        Default: lowercase class name without 'Renderer' suffix.
        Subclasses can override for custom names.
        """
        class_name = self.__class__.__name__
        if class_name.endswith('Renderer'):
            return class_name[:-8].lower()
        return class_name.lower()

    def supports_concurrent(self) -> bool:
        """
        Whether this renderer supports concurrent/parallel rendering.

        Default: True (stateless renderers are safe for concurrency).
        Override if renderer has mutable state.
        """
        return True

    async def render(
        self,
        specs: list[JennySpec],
        target: RenderTarget = RenderTarget.HTML,
        options: dict[str, Any] | None = None
    ) -> str:
        """
        Render multiple JennySpecs to output format.

        Args:
            specs: List of JennySpec panels to render
            target: Output format
            options: Renderer-specific options

        Returns:
            Rendered output as string

        Raises:
            UnsupportedTargetError: If target not supported
            RenderError: If rendering fails
        """
        if not self.supports_target(target):
            raise UnsupportedTargetError(
                f"{self.__class__.__name__} does not support {target.value}. "
                f"Supported: {[t.value for t in self.supported_targets]}"
            )

        options = options or {}

        try:
            return await self._render_multiple(specs, target, options)
        except Exception as e:
            self.logger.error(f"Render failed: {e}")
            raise RenderError(f"Failed to render {len(specs)} specs: {e}") from e

    async def render_single(
        self,
        spec: JennySpec,
        target: RenderTarget = RenderTarget.HTML,
        options: dict[str, Any] | None = None
    ) -> str:
        """
        Render a single JennySpec panel.

        Args:
            spec: Single JennySpec to render
            target: Output format
            options: Renderer-specific options

        Returns:
            Rendered panel as string
        """
        if not self.supports_target(target):
            raise UnsupportedTargetError(
                f"{self.__class__.__name__} does not support {target.value}"
            )

        options = options or {}

        try:
            return await self._render_single(spec, target, options)
        except Exception as e:
            self.logger.error(f"Render single failed: {e}")
            raise RenderError(f"Failed to render spec {spec.spec_id}: {e}") from e

    async def render_update(
        self,
        old_spec: JennySpec,
        new_spec: JennySpec,
        target: RenderTarget = RenderTarget.HTML
    ) -> str:
        """
        Render differential update between specs.

        Default implementation re-renders the new spec.
        Subclasses can override for efficient diffing.

        Args:
            old_spec: Previous panel state
            new_spec: New panel state
            target: Output format

        Returns:
            Differential update or full re-render
        """
        # Default: just render the new spec
        # Subclasses can implement actual diffing
        return await self.render_single(new_spec, target)

    @abstractmethod
    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Internal: Render multiple specs."""
        ...

    @abstractmethod
    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Internal: Render single spec."""
        ...


# ============================================================================
# HTML Renderer
# ============================================================================

class HTMLRenderer(JennyRendererBase):
    """
    Renders JennySpec panels to semantic HTML + CSS.

    Features:
    - Semantic HTML5 structure
    - CSS animations for lifecycle states
    - Accessibility (ARIA labels, keyboard nav) - Phase M3
    - Dark/light theme support
    - Responsive grid layout

    Phase M3 Accessibility Features:
    - WCAG 2.1 AA compliance
    - Keyboard navigation (Arrow keys, Tab, Enter, Escape)
    - Focus management with visible focus indicators
    - Screen reader announcements via live regions
    - Reduced motion support

    Usage:
        renderer = HTMLRenderer()
        html = await renderer.render(specs)
        # Embed in page or save to file
    """

    # CSS class prefixes
    PANEL_CLASS = "jenny-panel"
    CONTAINER_CLASS = "jenny-container"

    def __init__(self):
        """Initialize HTML renderer with accessibility layer."""
        super().__init__()

        # Initialize accessibility layer (Phase M3)
        self._a11y: JennyAccessibilityLayer | None = None
        if ACCESSIBILITY_AVAILABLE:
            self._a11y = create_accessibility_layer()
            self.logger.info("Accessibility layer enabled (WCAG 2.1 AA)")

    @property
    def supported_targets(self) -> list[RenderTarget]:
        return [RenderTarget.HTML]

    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render multiple specs to HTML dashboard."""
        include_styles = options.get('include_styles', True)
        include_scripts = options.get('include_scripts', True)
        include_a11y = options.get('include_accessibility', True)
        title = options.get('title', 'Jenny Dashboard')

        # Sort by priority (lower = first)
        sorted_specs = sorted(specs, key=lambda s: s.priority)

        # Collect panel IDs for keyboard navigation
        panel_ids = [spec.spec_id for spec in sorted_specs]

        # Render individual panels with position info for accessibility
        panels_html = []
        for i, spec in enumerate(sorted_specs, 1):
            panel = await self._render_single(
                spec, target, options,
                position=i, total=len(sorted_specs)
            )
            panels_html.append(panel)

        # Build container with accessibility landmark
        container = f'''<div id="jenny-main-content" class="{self.CONTAINER_CLASS}" role="main" aria-label="{html.escape(title)}">
    {''.join(panels_html)}
</div>'''

        # Add styles and scripts if requested
        if include_styles or include_scripts:
            head_content = []
            body_prefix = []
            body_suffix = []

            if include_styles:
                head_content.append(self._get_styles())
            if include_scripts:
                head_content.append(self._get_scripts())

            # Add accessibility infrastructure (Phase M3)
            if include_a11y and self._a11y:
                head_content.append(self._get_a11y_styles())
                body_prefix.append(self._get_a11y_skip_link())
                body_suffix.append(self._get_a11y_live_region())
                body_suffix.append(self._get_a11y_scripts(panel_ids))

            return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    {''.join(head_content)}
</head>
<body>
    {''.join(body_prefix)}
    {container}
    {''.join(body_suffix)}
</body>
</html>'''

        return container

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any],
        position: int | None = None,
        total: int | None = None
    ) -> str:
        """
        Render single spec to HTML panel.

        Args:
            spec: JennySpec to render
            target: Output format
            options: Renderer options
            position: Position in panel set (1-indexed, for accessibility)
            total: Total panels in set (for accessibility)

        Returns:
            Panel HTML string
        """
        # Build CSS classes
        classes = [
            self.PANEL_CLASS,
            f"{self.PANEL_CLASS}--{spec.panel_type.value}",
            f"{self.PANEL_CLASS}--{spec.size.value}",
            f"{self.PANEL_CLASS}--{spec.lifecycle.value}",
        ]

        # Build data attributes for JS
        data_attrs = {
            'data-spec-id': spec.spec_id,
            'data-panel-type': spec.panel_type.value,
            'data-lifecycle': spec.lifecycle.value,
            'data-binding': spec.binding_mode.value,
            'data-priority': str(spec.priority),
        }
        if spec.spacetime_id:
            data_attrs['data-spacetime-id'] = spec.spacetime_id

        data_str = ' '.join(f'{k}="{html.escape(str(v))}"' for k, v in data_attrs.items())

        # Render content based on panel type
        content_html = self._render_panel_content(spec)

        # Render actions
        actions_html = self._render_actions(spec)

        # Build ARIA position attributes (Phase M3)
        aria_position_attrs = []
        if position is not None:
            aria_position_attrs.append(f'aria-posinset="{position}"')
        if total is not None:
            aria_position_attrs.append(f'aria-setsize="{total}"')
        aria_position_str = ' '.join(aria_position_attrs)

        # Determine tabindex (first panel is focusable, others on navigation)
        tabindex = '0' if position == 1 else '-1'

        # Build panel HTML
        return f'''<article class="{' '.join(classes)}" {data_str} role="article" aria-label="{html.escape(spec.title or spec.panel_type.value)}" tabindex="{tabindex}" {aria_position_str}>
    <header class="{self.PANEL_CLASS}__header">
        {f'<h3 class="{self.PANEL_CLASS}__title">{html.escape(spec.title)}</h3>' if spec.title else ''}
        {f'<p class="{self.PANEL_CLASS}__subtitle">{html.escape(spec.subtitle)}</p>' if spec.subtitle else ''}
    </header>
    <div class="{self.PANEL_CLASS}__content">
        {content_html}
    </div>
    {f'<footer class="{self.PANEL_CLASS}__actions">{actions_html}</footer>' if spec.actions else ''}
</article>
'''

    def _render_panel_content(self, spec: JennySpec) -> str:
        """Render content based on panel type."""
        content = spec.content

        if spec.panel_type == PanelTypeJenny.TEXT:
            text = content.get('text', '')
            fmt = content.get('format', 'text')
            if fmt == 'markdown':
                # Simple markdown escape (real impl would use markdown lib)
                return f'<div class="jenny-text jenny-text--markdown">{html.escape(text)}</div>'
            return f'<div class="jenny-text">{html.escape(text)}</div>'

        elif spec.panel_type == PanelTypeJenny.CONFIDENCE:
            value = content.get('value', 0)
            threshold_low = content.get('threshold_low', 0.6)
            threshold_high = content.get('threshold_high', 0.8)
            pct = int(value * 100)

            # Determine color class
            if value >= threshold_high:
                color_class = 'jenny-confidence--high'
            elif value >= threshold_low:
                color_class = 'jenny-confidence--medium'
            else:
                color_class = 'jenny-confidence--low'

            return f'''<div class="jenny-confidence {color_class}">
    <div class="jenny-confidence__value">{pct}%</div>
    <div class="jenny-confidence__bar" style="width: {pct}%"></div>
    <div class="jenny-confidence__label">Confidence</div>
</div>'''

        elif spec.panel_type == PanelTypeJenny.SOURCES:
            sources = content.get('sources', [])
            if not sources:
                return '<div class="jenny-sources jenny-sources--empty">No sources</div>'

            items = []
            for i, source in enumerate(sources, 1):
                if isinstance(source, dict):
                    title = source.get('title', f'Source {i}')
                    url = source.get('url', '#')
                    items.append(f'<li><a href="{html.escape(url)}">{html.escape(title)}</a></li>')
                else:
                    items.append(f'<li>{html.escape(str(source))}</li>')

            return f'<ul class="jenny-sources">{" ".join(items)}</ul>'

        elif spec.panel_type == PanelTypeJenny.GRAPH:
            # Placeholder for knowledge graph
            nodes = content.get('nodes', [])
            edges = content.get('edges', [])
            return f'''<div class="jenny-graph" data-nodes="{len(nodes)}" data-edges="{len(edges)}">
    <div class="jenny-graph__placeholder">Knowledge Graph ({len(nodes)} nodes, {len(edges)} edges)</div>
</div>'''

        elif spec.panel_type == PanelTypeJenny.REASONING:
            steps = content.get('steps', [])
            if not steps:
                return '<div class="jenny-reasoning jenny-reasoning--empty">No reasoning steps</div>'

            items = []
            for i, step in enumerate(steps, 1):
                items.append(f'<li class="jenny-reasoning__step"><span class="jenny-reasoning__num">{i}</span>{html.escape(str(step))}</li>')

            return f'<ol class="jenny-reasoning">{" ".join(items)}</ol>'

        elif spec.panel_type == PanelTypeJenny.WHY:
            query_type = content.get('query_type', 'unknown')
            complexity = content.get('complexity', 'unknown')
            panels_generated = content.get('panels_generated', 0)
            reasoning = content.get('reasoning', '')

            return f'''<div class="jenny-why">
    <dl class="jenny-why__meta">
        <dt>Query Type</dt><dd>{html.escape(query_type)}</dd>
        <dt>Complexity</dt><dd>{html.escape(complexity)}</dd>
        <dt>Panels Generated</dt><dd>{panels_generated}</dd>
    </dl>
    <p class="jenny-why__reasoning">{html.escape(reasoning)}</p>
</div>'''

        elif spec.panel_type == PanelTypeJenny.CODE:
            code = content.get('code', '')
            language = content.get('language', 'text')
            return f'<pre class="jenny-code" data-language="{html.escape(language)}"><code>{html.escape(code)}</code></pre>'

        elif spec.panel_type == PanelTypeJenny.TABLE:
            headers = content.get('headers', [])
            rows = content.get('rows', [])

            header_html = ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
            rows_html = ''.join(
                '<tr>' + ''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row) + '</tr>'
                for row in rows
            )

            return f'''<table class="jenny-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
</table>'''

        elif spec.panel_type == PanelTypeJenny.METRIC:
            value = content.get('value', 0)
            unit = content.get('unit', '')
            label = content.get('label', 'Metric')
            return f'''<div class="jenny-metric">
    <div class="jenny-metric__value">{html.escape(str(value))}{html.escape(unit)}</div>
    <div class="jenny-metric__label">{html.escape(label)}</div>
</div>'''

        elif spec.panel_type == PanelTypeJenny.PROMOTION:
            return self._render_promotion(content)

        elif spec.panel_type == PanelTypeJenny.DEPARTMENT:
            return self._render_department(content)

        elif spec.panel_type == PanelTypeJenny.CROSS_DEPT:
            return self._render_cross_dept(content)

        elif spec.panel_type == PanelTypeJenny.FEEDBACK_STATE:
            return self._render_feedback_state(content)

        else:
            # Fallback: JSON dump
            return f'<pre class="jenny-raw">{html.escape(json.dumps(content, indent=2, default=str))}</pre>'

    # ====================================================================
    # Visualization Primitive Renderers (Five Primitives Grammar)
    # ====================================================================

    def _render_promotion(self, content: dict) -> str:
        """Render promotion flow — Trajectory + Distribution."""
        trajectory = content.get("trajectory", {})
        distribution = content.get("distribution", {})
        stages = trajectory.get("stages", [])
        flow = trajectory.get("flow", {})
        rubric_weights = distribution.get("rubric_weights", {})

        # Trajectory: Sankey-style flow stages
        stage_bars = []
        max_count = max((s.get("count", 0) for s in stages), default=1) or 1
        for stage in stages:
            count = stage.get("count", 0)
            pct = int(count / max_count * 100) if max_count > 0 else 0
            name = html.escape(stage.get("name", ""))
            stage_bars.append(
                f'<div class="jenny-prom__stage">'
                f'<span class="jenny-prom__label">{name}</span>'
                f'<div class="jenny-prom__bar" style="width:{pct}%"></div>'
                f'<span class="jenny-prom__count">{count}</span>'
                f'</div>'
            )

        # Distribution: rubric weight bars
        weight_bars = []
        total_weight = sum(rubric_weights.values()) or 1
        for term, weight in rubric_weights.items():
            pct = int(weight / total_weight * 100)
            weight_bars.append(
                f'<div class="jenny-prom__weight">'
                f'<span class="jenny-prom__term">{html.escape(term)}</span>'
                f'<div class="jenny-prom__wbar" style="width:{pct}%"></div>'
                f'<span class="jenny-prom__wpct">{weight}</span>'
                f'</div>'
            )

        total = flow.get("total_processed", 0)

        return f'''<div class="jenny-promotion" data-feedback-subject="{html.escape(content.get('department', ''))}" data-feedback-dimension="promotion_interest">
    <div class="jenny-prom__trajectory">
        <h4>Promotion Pipeline</h4>
        {''.join(stage_bars)}
        <div class="jenny-prom__total">{total} total processed</div>
    </div>
    <div class="jenny-prom__distribution">
        <h4>Rubric Weights</h4>
        {''.join(weight_bars)}
    </div>
</div>'''

    def _render_department(self, content: dict) -> str:
        """Render department state — Topology + Distribution + Tension."""
        topology = content.get("topology", {})
        distribution = content.get("distribution", {})
        tension = content.get("tension", {})
        dept = content.get("department", "")

        # Topology: community summary
        n_comm = topology.get("n_communities", 0)
        connectivity = topology.get("algebraic_connectivity", 0)
        inter = topology.get("inter_links", 0)
        intra = topology.get("intra_links", 0)

        # Distribution: bandit means
        bandit_means = distribution.get("bandit_means", {})
        mean_bars = []
        for term, mean in bandit_means.items():
            pct = int(mean * 100)
            color = "var(--jenny-success)" if mean >= 0.6 else "var(--jenny-warning)" if mean >= 0.4 else "var(--jenny-error)"
            mean_bars.append(
                f'<div class="jenny-dept__mean">'
                f'<span class="jenny-dept__term">{html.escape(term)}</span>'
                f'<div class="jenny-dept__mbar" style="width:{pct}%;background:{color}"></div>'
                f'<span class="jenny-dept__mpct">{mean:.2f}</span>'
                f'</div>'
            )

        # Tension: drift gauge
        drift = tension.get("drift")
        phase = tension.get("phase", "unknown")
        drift_html = ""
        if drift is not None:
            drift_val = drift if isinstance(drift, (int, float)) else 0
            drift_pct = min(int(drift_val * 100), 100)
            drift_color = "var(--jenny-success)" if drift_val < 0.3 else "var(--jenny-warning)" if drift_val < 0.6 else "var(--jenny-error)"
            drift_html = f'''<div class="jenny-dept__drift">
    <span>KL Drift</span>
    <div class="jenny-dept__gauge" style="width:{drift_pct}%;background:{drift_color}"></div>
    <span>{drift_val:.3f}</span>
</div>'''

        return f'''<div class="jenny-department" data-feedback-subject="{html.escape(dept)}" data-feedback-dimension="dept_interest">
    <div class="jenny-dept__topology">
        <h4>Belief Communities</h4>
        <div class="jenny-dept__stats">
            <span>{n_comm} communities</span>
            <span>connectivity: {connectivity:.3f}</span>
            <span>{inter} cross / {intra} intra</span>
        </div>
    </div>
    <div class="jenny-dept__distribution">
        <h4>Bandit Learning</h4>
        {''.join(mean_bars)}
    </div>
    <div class="jenny-dept__tension">
        <h4>Phase: {html.escape(phase)}</h4>
        {drift_html}
    </div>
</div>'''

    def _render_cross_dept(self, content: dict) -> str:
        """Render cross-department relationships — Topology + Tension."""
        topology = content.get("topology", {})
        tension = content.get("tension", {})
        departments = topology.get("departments", [])
        comparisons = topology.get("comparisons", [])

        # Comparison table
        rows = []
        for comp in comparisons[:10]:
            dept_a = html.escape(comp.get("dept_a", ""))
            dept_b = html.escape(comp.get("dept_b", ""))
            dist = comp.get("distance", 0)
            rel = html.escape(comp.get("relationship", "unknown"))
            bridges = ", ".join(comp.get("bridges", [])[:3])

            dist_color = "var(--jenny-success)" if dist < 0.3 else "var(--jenny-warning)" if dist < 0.6 else "var(--jenny-error)"
            rows.append(
                f'<tr>'
                f'<td>{dept_a}</td><td>{dept_b}</td>'
                f'<td style="color:{dist_color}">{dist:.3f}</td>'
                f'<td>{rel}</td>'
                f'<td>{html.escape(bridges)}</td>'
                f'</tr>'
            )

        most_connected = html.escape(tension.get("most_connected", "none"))
        most_isolated = html.escape(tension.get("most_isolated", "none"))

        return f'''<div class="jenny-crossdept" data-feedback-subject="cross_department" data-feedback-dimension="synthesis_interest">
    <div class="jenny-crossdept__topology">
        <h4>{len(departments)} Departments, {len(comparisons)} Comparisons</h4>
        <table class="jenny-table">
            <thead><tr><th>A</th><th>B</th><th>Distance</th><th>Relationship</th><th>Bridges</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    <div class="jenny-crossdept__tension">
        <div class="jenny-crossdept__pole">
            <span class="jenny-crossdept__label">Most Connected</span>
            <span class="jenny-crossdept__value">{most_connected}</span>
        </div>
        <div class="jenny-crossdept__pole">
            <span class="jenny-crossdept__label">Most Isolated</span>
            <span class="jenny-crossdept__value">{most_isolated}</span>
        </div>
    </div>
</div>'''

    def _render_feedback_state(self, content: dict) -> str:
        """Render learning state — Trajectory + Distribution."""
        trajectory = content.get("trajectory", {})
        distribution = content.get("distribution", {})
        temporal = content.get("temporal_layers", {})

        n_arms = trajectory.get("n_loom_arms", 0)

        # Temporal layers as stacked visualization
        layer_bars = []
        for layer, weight in temporal.items():
            pct = int(float(weight) * 100)
            layer_bars.append(
                f'<div class="jenny-fb__layer">'
                f'<span class="jenny-fb__lname">{html.escape(str(layer))}</span>'
                f'<div class="jenny-fb__lbar" style="width:{pct}%"></div>'
                f'<span class="jenny-fb__lwt">{weight}</span>'
                f'</div>'
            )

        # Bandit arms summary per department
        bandit_arms = distribution.get("bandit_arms", {})
        dept_summaries = []
        for dept, stats in bandit_arms.items():
            n_terms = len(stats) if isinstance(stats, dict) else 0
            dept_summaries.append(
                f'<div class="jenny-fb__dept">'
                f'<span>{html.escape(dept)}</span>'
                f'<span>{n_terms} arms</span>'
                f'</div>'
            )

        return f'''<div class="jenny-feedback" data-feedback-subject="learning_state" data-feedback-dimension="meta_interest">
    <div class="jenny-fb__trajectory">
        <h4>Active Learning</h4>
        <div class="jenny-fb__metric">{n_arms} loom arms</div>
    </div>
    <div class="jenny-fb__temporal">
        <h4>Temporal Layers</h4>
        {''.join(layer_bars)}
    </div>
    <div class="jenny-fb__bandits">
        <h4>Department Bandits</h4>
        {''.join(dept_summaries)}
    </div>
</div>'''

    def _render_actions(self, spec: JennySpec) -> str:
        """Render action buttons."""
        if not spec.actions:
            return ''

        buttons = []
        for action in spec.actions:
            action_id = action.get('action_id', '')
            label = action.get('label', 'Action')
            handler = action.get('handler', '')
            action_type = action.get('type', 'button')
            requires_confirm = action.get('requires_confirmation', False)

            btn_class = 'jenny-action'
            if action_type == 'toggle':
                btn_class += ' jenny-action--toggle'
            if requires_confirm:
                btn_class += ' jenny-action--confirm'

            buttons.append(
                f'<button class="{btn_class}" '
                f'data-action-id="{html.escape(action_id)}" '
                f'data-handler="{html.escape(handler)}" '
                f'data-spec-id="{html.escape(spec.spec_id)}">'
                f'{html.escape(label)}</button>'
            )

        return ' '.join(buttons)

    def _get_styles(self) -> str:
        """Get CSS styles for Jenny panels."""
        return '''<style>
/* Jenny Panel Styles - MVP Week 2 */
:root {
    --jenny-bg: #1a1a2e;
    --jenny-surface: #16213e;
    --jenny-border: #0f3460;
    --jenny-text: #e8e8e8;
    --jenny-text-muted: #a0a0a0;
    --jenny-accent: #00d9ff;
    --jenny-success: #00ff88;
    --jenny-warning: #ffcc00;
    --jenny-error: #ff4444;
    --jenny-radius: 8px;
    --jenny-transition: 0.3s ease;
}

.jenny-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
    padding: 16px;
    background: var(--jenny-bg);
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--jenny-text);
}

.jenny-panel {
    background: var(--jenny-surface);
    border: 1px solid var(--jenny-border);
    border-radius: var(--jenny-radius);
    padding: 16px;
    transition: transform var(--jenny-transition), opacity var(--jenny-transition);
}

/* Lifecycle animations */
.jenny-panel--nascent {
    animation: jenny-spawn 0.3s ease-out;
}

.jenny-panel--dissolving {
    animation: jenny-dissolve 0.3s ease-in forwards;
}

.jenny-panel--system {
    border-color: var(--jenny-accent);
    border-style: dashed;
}

@keyframes jenny-spawn {
    from { opacity: 0; transform: scale(0.95) translateY(-10px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes jenny-dissolve {
    from { opacity: 1; transform: scale(1); }
    to { opacity: 0; transform: scale(0.95); }
}

/* Size variants */
.jenny-panel--small { grid-column: span 1; }
.jenny-panel--medium { grid-column: span 1; }
.jenny-panel--large { grid-column: span 2; }
.jenny-panel--xlarge { grid-column: 1 / -1; }

/* Header */
.jenny-panel__header { margin-bottom: 12px; }
.jenny-panel__title { margin: 0 0 4px 0; font-size: 1.1rem; font-weight: 600; }
.jenny-panel__subtitle { margin: 0; font-size: 0.85rem; color: var(--jenny-text-muted); }

/* Content */
.jenny-panel__content { margin-bottom: 12px; }

/* Actions */
.jenny-panel__actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.jenny-action {
    padding: 6px 12px;
    background: var(--jenny-border);
    border: 1px solid var(--jenny-accent);
    border-radius: 4px;
    color: var(--jenny-accent);
    cursor: pointer;
    font-size: 0.85rem;
    transition: background var(--jenny-transition);
}

.jenny-action:hover {
    background: var(--jenny-accent);
    color: var(--jenny-bg);
}

/* Confidence gauge */
.jenny-confidence {
    text-align: center;
}

.jenny-confidence__value {
    font-size: 2rem;
    font-weight: bold;
}

.jenny-confidence__bar {
    height: 8px;
    background: var(--jenny-accent);
    border-radius: 4px;
    margin: 8px 0;
    transition: width var(--jenny-transition);
}

.jenny-confidence--high .jenny-confidence__bar { background: var(--jenny-success); }
.jenny-confidence--medium .jenny-confidence__bar { background: var(--jenny-warning); }
.jenny-confidence--low .jenny-confidence__bar { background: var(--jenny-error); }

/* Sources list */
.jenny-sources {
    list-style: none;
    padding: 0;
    margin: 0;
}

.jenny-sources li {
    padding: 4px 0;
    border-bottom: 1px solid var(--jenny-border);
}

.jenny-sources a {
    color: var(--jenny-accent);
    text-decoration: none;
}

.jenny-sources a:hover { text-decoration: underline; }

/* Reasoning steps */
.jenny-reasoning {
    padding-left: 0;
    counter-reset: step;
}

.jenny-reasoning__step {
    display: flex;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid var(--jenny-border);
}

.jenny-reasoning__num {
    background: var(--jenny-accent);
    color: var(--jenny-bg);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: bold;
    flex-shrink: 0;
}

/* Why panel */
.jenny-why__meta {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin-bottom: 12px;
}

.jenny-why__meta dt { color: var(--jenny-text-muted); }
.jenny-why__meta dd { margin: 0; }

/* Code block */
.jenny-code {
    background: var(--jenny-bg);
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 0.9rem;
}

/* Table */
.jenny-table {
    width: 100%;
    border-collapse: collapse;
}

.jenny-table th, .jenny-table td {
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid var(--jenny-border);
}

.jenny-table th { color: var(--jenny-text-muted); font-weight: 600; }

/* Metric */
.jenny-metric {
    text-align: center;
}

.jenny-metric__value {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--jenny-accent);
}

.jenny-metric__label {
    color: var(--jenny-text-muted);
    font-size: 0.9rem;
}

/* ===== Visualization Primitive Styles ===== */

/* Shared bar pattern */
.jenny-prom__bar, .jenny-prom__wbar,
.jenny-dept__mbar, .jenny-dept__gauge,
.jenny-fb__lbar {
    height: 8px;
    border-radius: 4px;
    background: var(--jenny-accent);
    transition: width var(--jenny-transition);
    min-width: 2px;
}

/* Promotion panel */
.jenny-promotion { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.jenny-prom__stage, .jenny-prom__weight {
    display: grid; grid-template-columns: 100px 1fr 40px; align-items: center; gap: 8px; margin: 4px 0;
}
.jenny-prom__label, .jenny-prom__term { font-size: 0.85rem; color: var(--jenny-text-muted); }
.jenny-prom__count, .jenny-prom__wpct { font-size: 0.85rem; text-align: right; }
.jenny-prom__total { margin-top: 8px; font-size: 0.85rem; color: var(--jenny-text-muted); }
.jenny-prom__wbar { background: var(--jenny-warning); }

/* Department panel */
.jenny-department { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.jenny-dept__stats { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.85rem; color: var(--jenny-text-muted); }
.jenny-dept__mean {
    display: grid; grid-template-columns: 100px 1fr 50px; align-items: center; gap: 8px; margin: 4px 0;
}
.jenny-dept__term { font-size: 0.85rem; color: var(--jenny-text-muted); }
.jenny-dept__mpct { font-size: 0.85rem; text-align: right; }
.jenny-dept__drift {
    display: grid; grid-template-columns: 60px 1fr 50px; align-items: center; gap: 8px; margin: 8px 0;
}
.jenny-dept__gauge { height: 12px; border-radius: 6px; }

/* Cross-department panel */
.jenny-crossdept { display: grid; gap: 16px; }
.jenny-crossdept__tension { display: flex; gap: 24px; justify-content: center; }
.jenny-crossdept__pole { text-align: center; }
.jenny-crossdept__label { display: block; font-size: 0.8rem; color: var(--jenny-text-muted); }
.jenny-crossdept__value { display: block; font-size: 1.2rem; font-weight: 600; color: var(--jenny-accent); }

/* Feedback state panel */
.jenny-feedback { display: grid; grid-template-columns: auto 1fr 1fr; gap: 16px; }
.jenny-fb__metric { font-size: 1.8rem; font-weight: bold; color: var(--jenny-accent); }
.jenny-fb__layer, .jenny-fb__dept {
    display: grid; grid-template-columns: 100px 1fr 40px; align-items: center; gap: 8px; margin: 4px 0;
}
.jenny-fb__lname { font-size: 0.85rem; color: var(--jenny-text-muted); }
.jenny-fb__lwt { font-size: 0.85rem; text-align: right; }

/* All viz primitives: section headings */
.jenny-promotion h4, .jenny-department h4, .jenny-crossdept h4, .jenny-feedback h4 {
    margin: 0 0 8px 0; font-size: 0.9rem; font-weight: 600; color: var(--jenny-text-muted);
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* Responsive: stack on narrow screens */
@media (max-width: 768px) {
    .jenny-promotion, .jenny-department, .jenny-feedback { grid-template-columns: 1fr; }
}
</style>'''

    def _get_scripts(self) -> str:
        """Get JavaScript for panel interactions."""
        return '''<script>
// Jenny Panel Interactions - MVP Week 2
document.addEventListener('DOMContentLoaded', () => {
    // Action button handlers
    document.querySelectorAll('.jenny-action').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const handler = btn.dataset.handler;
            const specId = btn.dataset.specId;
            const actionId = btn.dataset.actionId;

            console.log(`[Jenny] Action: ${handler} on ${specId}`);

            // Emit custom event for application to handle
            document.dispatchEvent(new CustomEvent('jenny:action', {
                detail: { handler, specId, actionId, button: btn }
            }));
        });
    });

    // ================================================================
    // Feedback Signal Emission
    // Interaction IS the feedback loop. Every action emits a signal.
    // ================================================================
    const TIMESCALE_WEIGHTS = {
        SUB_SECOND: 0.05, SECONDS: 0.15, MINUTES: 0.30,
        HOURS: 0.60, DAYS: 0.85, WEEKS: 1.00
    };

    function emitFeedback(subject, dimension, timescale, value) {
        const signal = {
            subject, dimension, timescale, value,
            weight: TIMESCALE_WEIGHTS[timescale] || 0.15,
            timestamp: new Date().toISOString()
        };
        console.log('[Jenny Feedback]', signal);
        // Fire-and-forget POST to feedback endpoint
        fetch('/viz/feedback/signal', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(signal)
        }).catch(() => {}); // Non-blocking
        document.dispatchEvent(new CustomEvent('jenny:feedback', { detail: signal }));
    }

    // Map action handlers to feedback signals
    document.addEventListener('jenny:action', (e) => {
        const { handler, specId } = e.detail;
        const panel = document.querySelector(`[data-spec-id="${specId}"]`);
        const subject = panel?.querySelector('[data-feedback-subject]')?.dataset.feedbackSubject || specId;
        const dimension = panel?.querySelector('[data-feedback-dimension]')?.dataset.feedbackDimension || 'interest';

        switch (handler) {
            case 'pin_panel':
                emitFeedback(subject, dimension, 'MINUTES', 1.0);
                break;
            case 'dismiss_panel':
                emitFeedback(subject, dimension, 'SECONDS', -0.5);
                break;
            case 'expand_panel':
                emitFeedback(subject, dimension, 'SECONDS', 0.5);
                break;
            case 'refresh_panel':
                emitFeedback(subject, dimension, 'SECONDS', 0.3);
                break;
            case 'drill_panel':
                emitFeedback(subject, dimension, 'SECONDS', 0.7);
                break;
        }
    });

    // Hover tracking: sub-second hypothesis (debounced)
    let hoverTimer = null;
    document.querySelectorAll('[data-feedback-subject]').forEach(el => {
        el.addEventListener('mouseenter', () => {
            hoverTimer = setTimeout(() => {
                const subject = el.dataset.feedbackSubject;
                const dimension = el.dataset.feedbackDimension || 'interest';
                emitFeedback(subject, dimension, 'SUB_SECOND', 0.1);
            }, 500); // Only emit after 500ms hover (not accidental)
        });
        el.addEventListener('mouseleave', () => clearTimeout(hoverTimer));
    });

    // Lifecycle transition handler
    window.jennyTransition = (specId, newLifecycle) => {
        const panel = document.querySelector(`[data-spec-id="${specId}"]`);
        if (panel) {
            panel.classList.remove('jenny-panel--nascent', 'jenny-panel--stable', 'jenny-panel--dissolving');
            panel.classList.add(`jenny-panel--${newLifecycle}`);
            panel.dataset.lifecycle = newLifecycle;

            if (newLifecycle === 'dissolving') {
                setTimeout(() => panel.remove(), 300);
            }
        }
    };
});
</script>'''

    # ========================================================================
    # Accessibility Helper Methods (Phase M3)
    # ========================================================================

    def _get_a11y_styles(self) -> str:
        """
        Get accessibility-specific CSS styles.

        Includes:
        - Focus indicators (visible focus rings)
        - Reduced motion support
        - Skip link styling
        """
        if not self._a11y:
            return ''

        return f'''
{self._a11y.focus_manager.generate_focus_styles()}
{self._a11y.motion_prefs.generate_reduced_motion_styles()}
'''

    def _get_a11y_skip_link(self) -> str:
        """
        Get skip-to-content link for keyboard navigation.

        Allows keyboard users to skip navigation and go directly to main content.
        """
        if not self._a11y:
            return ''

        return self._a11y.focus_manager.generate_skip_link("jenny-main-content")

    def _get_a11y_live_region(self) -> str:
        """
        Get live region HTML for screen reader announcements.

        Includes:
        - Hidden live region element
        - JavaScript announcement helper functions
        """
        if not self._a11y:
            return ''

        return f'''
{self._a11y.live_announcer.generate_live_region_html()}
{self._a11y.live_announcer.generate_announce_script()}
'''

    def _get_a11y_scripts(self, panel_ids: list[str]) -> str:
        """
        Get accessibility JavaScript for keyboard navigation.

        Args:
            panel_ids: List of panel IDs for navigation order

        Returns:
            JavaScript for:
            - Arrow key navigation between panels
            - Enter/Space to activate primary action
            - Escape to dismiss panels
            - Motion preference detection
        """
        if not self._a11y:
            return ''

        return f'''
{self._a11y.keyboard_handler.generate_panel_navigation_script(panel_ids)}
{self._a11y.motion_prefs.generate_motion_query_script()}
'''


# ============================================================================
# Terminal Renderer
# ============================================================================

class TerminalRenderer(JennyRendererBase):
    """
    Renders JennySpec panels to ASCII art for CLI/debugging.

    Features:
    - Box-drawing characters for panel borders
    - Color codes (ANSI) for lifecycle states
    - Compact single-line mode for logging
    - Full-width mode for terminal display

    Usage:
        renderer = TerminalRenderer()
        output = await renderer.render(specs)
        print(output)
    """

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'cyan': '\033[36m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'red': '\033[31m',
        'magenta': '\033[35m',
    }

    # Lifecycle colors
    LIFECYCLE_COLORS = {
        LifecycleStage.NASCENT: 'cyan',
        LifecycleStage.STABLE: 'green',
        LifecycleStage.DISSOLVING: 'yellow',
        LifecycleStage.ARCHIVED: 'dim',
        LifecycleStage.SYSTEM: 'magenta',
    }

    @property
    def supported_targets(self) -> list[RenderTarget]:
        return [RenderTarget.TERMINAL]

    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render multiple specs to terminal output."""
        compact = options.get('compact', False)
        no_color = options.get('no_color', False)
        width = options.get('width', 80)

        if compact:
            return self._render_compact(specs, no_color)

        # Sort by priority
        sorted_specs = sorted(specs, key=lambda s: s.priority)

        lines = []
        lines.append(self._color('bold', f"┌{'─' * (width - 2)}┐", no_color))
        lines.append(self._color('bold', f"│ Jenny Dashboard ({len(specs)} panels){' ' * (width - 28 - len(str(len(specs))))}│", no_color))
        lines.append(self._color('bold', f"└{'─' * (width - 2)}┘", no_color))
        lines.append("")

        for spec in sorted_specs:
            panel = await self._render_single(spec, target, {'width': width, 'no_color': no_color})
            lines.append(panel)
            lines.append("")

        return '\n'.join(lines)

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render single spec to terminal box."""
        width = options.get('width', 60)
        no_color = options.get('no_color', False)

        lifecycle_color = self.LIFECYCLE_COLORS.get(spec.lifecycle, 'reset')

        # Build header line
        title = spec.title or spec.panel_type.value
        lifecycle_badge = f"[{spec.lifecycle.value}]"
        header = f" {title} {lifecycle_badge}"
        padding = width - len(header) - 4
        header_line = f"│{header}{' ' * max(0, padding)}│"

        lines = []
        lines.append(self._color(lifecycle_color, f"┌{'─' * (width - 2)}┐", no_color))
        lines.append(self._color(lifecycle_color, header_line, no_color))
        lines.append(self._color(lifecycle_color, f"├{'─' * (width - 2)}┤", no_color))

        # Render content
        content_lines = self._render_terminal_content(spec, width - 4)
        for line in content_lines:
            padded = f" {line}"
            padded = padded[:width - 3] + ' ' * max(0, width - 3 - len(padded))
            lines.append(self._color(lifecycle_color, f"│{padded}│", no_color))

        # Render actions if present
        if spec.actions:
            lines.append(self._color(lifecycle_color, f"├{'─' * (width - 2)}┤", no_color))
            actions_str = " ".join(f"[{a.get('label', '?')}]" for a in spec.actions)
            actions_line = f" {actions_str}"[:width - 4]
            actions_line = actions_line + ' ' * max(0, width - 3 - len(actions_line))
            lines.append(self._color('dim', f"│{actions_line}│", no_color))

        lines.append(self._color(lifecycle_color, f"└{'─' * (width - 2)}┘", no_color))

        return '\n'.join(lines)

    def _render_terminal_content(self, spec: JennySpec, max_width: int) -> list[str]:
        """Render content for terminal display."""
        content = spec.content
        lines = []

        if spec.panel_type == PanelTypeJenny.TEXT:
            text = content.get('text', '')
            # Word wrap
            words = text.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_width:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

        elif spec.panel_type == PanelTypeJenny.CONFIDENCE:
            value = content.get('value', 0)
            pct = int(value * 100)
            bar_width = min(max_width - 10, 30)
            filled = int(bar_width * value)
            bar = '█' * filled + '░' * (bar_width - filled)
            lines.append(f"Confidence: {pct}%")
            lines.append(f"[{bar}]")

        elif spec.panel_type == PanelTypeJenny.SOURCES:
            sources = content.get('sources', [])
            for i, source in enumerate(sources[:5], 1):
                if isinstance(source, dict):
                    title = source.get('title', f'Source {i}')[:max_width - 4]
                else:
                    title = str(source)[:max_width - 4]
                lines.append(f" {i}. {title}")
            if len(sources) > 5:
                lines.append(f" ... and {len(sources) - 5} more")

        elif spec.panel_type == PanelTypeJenny.WHY:
            lines.append(f"Type: {content.get('query_type', '?')}")
            lines.append(f"Complexity: {content.get('complexity', '?')}")
            lines.append(f"Panels: {content.get('panels_generated', 0)}")
            reasoning = content.get('reasoning', '')
            if reasoning:
                lines.append("")
                lines.append(reasoning[:max_width])

        elif spec.panel_type == PanelTypeJenny.REASONING:
            steps = content.get('steps', [])
            for i, step in enumerate(steps[:5], 1):
                step_text = str(step)[:max_width - 4]
                lines.append(f" {i}. {step_text}")

        else:
            # Fallback: show JSON preview
            json_str = json.dumps(content, default=str)[:max_width * 3]
            for i in range(0, len(json_str), max_width):
                lines.append(json_str[i:i + max_width])

        return lines or ["(empty)"]

    def _render_compact(self, specs: list[JennySpec], no_color: bool) -> str:
        """Render specs in compact single-line format."""
        lines = []
        for spec in specs:
            lifecycle_color = self.LIFECYCLE_COLORS.get(spec.lifecycle, 'reset')
            badge = self._color(lifecycle_color, f"[{spec.lifecycle.value[:3].upper()}]", no_color)
            title = spec.title or spec.panel_type.value
            lines.append(f"{badge} {spec.panel_type.value}: {title} (id={spec.spec_id[:8]}...)")
        return '\n'.join(lines)

    def _color(self, color: str, text: str, no_color: bool = False) -> str:
        """Apply ANSI color to text."""
        if no_color:
            return text
        code = self.COLORS.get(color, '')
        reset = self.COLORS['reset']
        return f"{code}{text}{reset}"


# ============================================================================
# JSON Renderer
# ============================================================================

class JSONRenderer(JennyRendererBase):
    """
    Renders JennySpec panels to JSON format.

    Useful for:
    - API responses
    - Client-side rendering
    - Debugging

    Usage:
        renderer = JSONRenderer()
        json_str = await renderer.render(specs)
    """

    @property
    def supported_targets(self) -> list[RenderTarget]:
        return [RenderTarget.JSON]

    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render specs to JSON array."""
        indent = options.get('indent', 2)
        sorted_specs = sorted(specs, key=lambda s: s.priority)

        data = {
            'panels': [spec.to_dict() for spec in sorted_specs],
            'count': len(specs),
            'rendered_at': datetime.now().isoformat(),
            'renderer_version': self.VERSION,
        }

        return json.dumps(data, indent=indent, default=str)

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render single spec to JSON."""
        indent = options.get('indent', 2)
        return json.dumps(spec.to_dict(), indent=indent, default=str)


# ============================================================================
# React Renderer (Phase M4)
# ============================================================================

class ReactRenderer(JennyRendererBase):
    """
    Renders JennySpec panels to React component props.

    Outputs JSON props that can be consumed by React components.
    Designed for server-side rendering and client hydration.

    Features (Phase M4):
    - React-ready component props
    - Accessibility props (ARIA attributes)
    - Event handler prop stubs
    - CSS class generation
    - TypeScript-friendly output

    Usage:
        renderer = ReactRenderer()
        props_json = await renderer.render(specs)

        # In React:
        const props = JSON.parse(props_json);
        return <JennyDashboard {...props} />;

    Output format:
        {
            "component": "JennyDashboard",
            "props": {
                "panels": [...],
                "accessibility": {...},
                "theme": {...}
            }
        }
    """

    # Component type mapping
    PANEL_COMPONENTS = {
        PanelTypeJenny.TEXT: 'JennyTextPanel',
        PanelTypeJenny.CONFIDENCE: 'JennyConfidenceGauge',
        PanelTypeJenny.SOURCES: 'JennySourcesList',
        PanelTypeJenny.GRAPH: 'JennyKnowledgeGraph',
        PanelTypeJenny.REASONING: 'JennyReasoningSteps',
        PanelTypeJenny.WHY: 'JennyWhyPanel',
        PanelTypeJenny.CODE: 'JennyCodeBlock',
        PanelTypeJenny.TABLE: 'JennyDataTable',
        PanelTypeJenny.METRIC: 'JennyMetricCard',
    }

    @property
    def supported_targets(self) -> list[RenderTarget]:
        return [RenderTarget.REACT]

    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render specs to React props JSON."""
        indent = options.get('indent', 2)
        include_handlers = options.get('include_handlers', True)
        include_accessibility = options.get('include_accessibility', True)
        theme = options.get('theme', 'dark')

        # Sort by priority
        sorted_specs = sorted(specs, key=lambda s: s.priority)

        # Build panel props
        panels = []
        for i, spec in enumerate(sorted_specs, 1):
            panel_props = await self._spec_to_props(
                spec,
                position=i,
                total=len(sorted_specs),
                include_handlers=include_handlers,
                include_accessibility=include_accessibility
            )
            panels.append(panel_props)

        # Build dashboard props
        dashboard_props = {
            'component': 'JennyDashboard',
            'props': {
                'panels': panels,
                'panelCount': len(panels),
                'theme': theme,
                'className': f'jenny-dashboard jenny-dashboard--{theme}',
                'renderedAt': datetime.now().isoformat(),
                'rendererVersion': self.VERSION,
            }
        }

        # Add global accessibility props
        if include_accessibility:
            dashboard_props['props']['accessibility'] = {
                'role': 'main',
                'ariaLabel': 'Jenny Dashboard',
                'ariaLive': 'polite',
                'tabIndex': 0,
                'onKeyDown': 'handleDashboardKeyDown',
            }

        # Add global event handlers
        if include_handlers:
            dashboard_props['props']['handlers'] = {
                'onPanelFocus': 'handlePanelFocus',
                'onPanelAction': 'handlePanelAction',
                'onPanelDismiss': 'handlePanelDismiss',
                'onKeyboardNavigation': 'handleKeyboardNavigation',
            }

        return json.dumps(dashboard_props, indent=indent, default=str)

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any],
        position: int | None = None,
        total: int | None = None
    ) -> str:
        """Render single spec to React props JSON."""
        indent = options.get('indent', 2)
        include_handlers = options.get('include_handlers', True)
        include_accessibility = options.get('include_accessibility', True)

        props = await self._spec_to_props(
            spec,
            position=position or 1,
            total=total or 1,
            include_handlers=include_handlers,
            include_accessibility=include_accessibility
        )

        return json.dumps(props, indent=indent, default=str)

    async def _spec_to_props(
        self,
        spec: JennySpec,
        position: int,
        total: int,
        include_handlers: bool = True,
        include_accessibility: bool = True
    ) -> dict[str, Any]:
        """Convert JennySpec to React component props."""
        # Get component type
        component = self.PANEL_COMPONENTS.get(
            spec.panel_type,
            'JennyGenericPanel'
        )

        # Build CSS classes
        class_names = [
            'jenny-panel',
            f'jenny-panel--{spec.panel_type.value}',
            f'jenny-panel--{spec.size.value}',
            f'jenny-panel--{spec.lifecycle.value}',
        ]

        # Build base props
        props = {
            'component': component,
            'props': {
                # Identity
                'id': spec.spec_id,
                'key': spec.spec_id,

                # Content
                'title': spec.title,
                'subtitle': spec.subtitle,
                'content': spec.content,

                # Metadata
                'panelType': spec.panel_type.value,
                'size': spec.size.value,
                'priority': spec.priority,
                'lifecycle': spec.lifecycle.value,
                'bindingMode': spec.binding_mode.value,

                # Styling
                'className': ' '.join(class_names),

                # References
                'spacetimeId': spec.spacetime_id,
            }
        }

        # Add type-specific content props
        props['props']['contentProps'] = self._get_content_props(spec)

        # Add action props
        if spec.actions:
            props['props']['actions'] = [
                {
                    'actionId': action.get('action_id', ''),
                    'label': action.get('label', 'Action'),
                    'handler': action.get('handler', ''),
                    'type': action.get('type', 'button'),
                    'requiresConfirmation': action.get('requires_confirmation', False),
                    'disabled': action.get('disabled', False),
                }
                for action in spec.actions
            ]

        # Add dissolution props
        if spec.dissolution_trigger:
            props['props']['dissolution'] = {
                'trigger': spec.dissolution_trigger.value,
                'delay': spec.metadata.get('dissolution_delay_ms'),
            }

        # Add accessibility props
        if include_accessibility:
            props['props']['accessibility'] = {
                'role': 'article',
                'ariaLabel': spec.title or spec.panel_type.value,
                'ariaPosinset': position,
                'ariaSetsize': total,
                'tabIndex': 0 if position == 1 else -1,
            }

        # Add event handler props
        if include_handlers:
            props['props']['handlers'] = {
                'onClick': f'handlePanelClick_{spec.spec_id}',
                'onFocus': f'handlePanelFocus_{spec.spec_id}',
                'onKeyDown': f'handlePanelKeyDown_{spec.spec_id}',
                'onDismiss': f'handlePanelDismiss_{spec.spec_id}',
            }

        return props

    def _get_content_props(self, spec: JennySpec) -> dict[str, Any]:
        """Get type-specific content props for React components."""
        content = spec.content
        panel_type = spec.panel_type

        if panel_type == PanelTypeJenny.TEXT:
            return {
                'text': content.get('text', ''),
                'format': content.get('format', 'text'),
                'truncate': content.get('truncate', False),
                'maxLines': content.get('max_lines'),
            }

        elif panel_type == PanelTypeJenny.CONFIDENCE:
            value = content.get('value', 0)
            return {
                'value': value,
                'percentage': int(value * 100),
                'thresholdLow': content.get('threshold_low', 0.6),
                'thresholdHigh': content.get('threshold_high', 0.8),
                'showBar': True,
                'showLabel': True,
                'colorScheme': 'auto',  # high/medium/low based on value
            }

        elif panel_type == PanelTypeJenny.SOURCES:
            sources = content.get('sources', [])
            return {
                'sources': [
                    {
                        'title': s.get('title', f'Source {i+1}') if isinstance(s, dict) else str(s),
                        'url': s.get('url', '#') if isinstance(s, dict) else '#',
                        'relevance': s.get('relevance') if isinstance(s, dict) else None,
                    }
                    for i, s in enumerate(sources)
                ],
                'showRelevance': content.get('show_relevance', False),
                'maxVisible': content.get('max_visible', 10),
            }

        elif panel_type == PanelTypeJenny.GRAPH:
            return {
                'nodes': content.get('nodes', []),
                'edges': content.get('edges', []),
                'layout': content.get('layout', 'force'),
                'interactive': content.get('interactive', True),
                'highlightPath': content.get('highlight_path', []),
            }

        elif panel_type == PanelTypeJenny.REASONING:
            return {
                'steps': content.get('steps', []),
                'currentStep': content.get('current_step'),
                'showNumbers': True,
                'collapsible': content.get('collapsible', False),
            }

        elif panel_type == PanelTypeJenny.WHY:
            return {
                'queryType': content.get('query_type', 'unknown'),
                'complexity': content.get('complexity', 'unknown'),
                'panelsGenerated': content.get('panels_generated', 0),
                'reasoning': content.get('reasoning', ''),
            }

        elif panel_type == PanelTypeJenny.CODE:
            return {
                'code': content.get('code', ''),
                'language': content.get('language', 'text'),
                'showLineNumbers': content.get('show_line_numbers', True),
                'highlight': content.get('highlight', []),
                'copyable': content.get('copyable', True),
            }

        elif panel_type == PanelTypeJenny.TABLE:
            return {
                'headers': content.get('headers', []),
                'rows': content.get('rows', []),
                'sortable': content.get('sortable', False),
                'searchable': content.get('searchable', False),
                'pagination': content.get('pagination'),
            }

        elif panel_type == PanelTypeJenny.METRIC:
            return {
                'value': content.get('value', 0),
                'unit': content.get('unit', ''),
                'label': content.get('label', 'Metric'),
                'trend': content.get('trend'),  # 'up', 'down', 'stable'
                'comparison': content.get('comparison'),
            }

        else:
            # Generic fallback
            return {'raw': content}


# ============================================================================
# AR Renderer (Phase M6)
# ============================================================================

class ARRenderer(JennyRendererBase):
    """
    AR (Augmented Reality) renderer for WebXR-compatible spatial output.

    Outputs 3D spatial panel specifications for AR/VR environments:
    - Position: x, y, z coordinates in meters
    - Orientation: Quaternion or Euler angles
    - Scale: Panel dimensions in physical space
    - Anchors: World/device anchor preferences
    - Gaze targeting: Look-at behavior

    Output Format:
        {
            "ar_version": "1.0",
            "coordinate_system": "world",  # or "device", "anchor"
            "panels": [
                {
                    "spec_id": "...",
                    "panel_type": "text",
                    "transform": {
                        "position": {"x": 0, "y": 1.5, "z": -2},
                        "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
                        "scale": {"x": 0.5, "y": 0.3, "z": 0.01}
                    },
                    "content": {...},
                    "behavior": {
                        "billboard": true,
                        "gaze_target": false,
                        "anchor_type": "world"
                    },
                    "accessibility": {...}
                }
            ],
            "scene_metadata": {...}
        }

    References:
    - WebXR Device API: https://immersive-web.github.io/webxr/
    - A-Frame: https://aframe.io/docs/
    """

    name = "ar"
    supported_targets = [RenderTarget.AR]

    # Size mapping to physical meters
    SIZE_TO_METERS = {
        PanelSizeJenny.SMALL: {"x": 0.3, "y": 0.2, "z": 0.01},
        PanelSizeJenny.MEDIUM: {"x": 0.5, "y": 0.3, "z": 0.01},
        PanelSizeJenny.LARGE: {"x": 0.8, "y": 0.5, "z": 0.01},
        PanelSizeJenny.XLARGE: {"x": 1.0, "y": 0.6, "z": 0.01},
    }

    # Default position (1.5m high, 2m in front of user)
    DEFAULT_POSITION = {"x": 0.0, "y": 1.5, "z": -2.0}

    async def _render_multiple(
        self,
        specs: list[JennySpec],
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render multiple specs to AR scene JSON."""

        panels = []
        for spec in specs:
            panel = self._spec_to_ar_panel(spec, options)
            panels.append(panel)

        # Build scene output
        scene = {
            "ar_version": "1.0",
            "coordinate_system": options.get("coordinate_system", "world"),
            "panels": panels,
            "scene_metadata": {
                "panel_count": len(panels),
                "generated_at": datetime.now().isoformat(),
                "layout_hint": options.get("layout_hint", "radial"),
            }
        }

        return json.dumps(scene, indent=2, default=str)

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: dict[str, Any]
    ) -> str:
        """Render a single spec to AR panel JSON."""
        panel = self._spec_to_ar_panel(spec, options)

        return json.dumps({
            "ar_version": "1.0",
            "coordinate_system": options.get("coordinate_system", "world"),
            "panels": [panel],
            "scene_metadata": {
                "panel_count": 1,
                "generated_at": datetime.now().isoformat(),
            }
        }, indent=2, default=str)

    def _spec_to_ar_panel(
        self,
        spec: JennySpec,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert JennySpec to AR panel specification."""

        # Calculate transform
        transform = self._calculate_transform(spec, options)

        # Build content based on panel type
        content = self._get_ar_content(spec)

        # Calculate behavior settings
        behavior = self._get_behavior(spec, options)

        # Build accessibility
        accessibility = self._get_accessibility(spec)

        return {
            "spec_id": spec.spec_id,
            "spacetime_id": spec.spacetime_id,
            "panel_type": spec.panel_type.value,
            "title": spec.title,
            "subtitle": spec.subtitle,
            "lifecycle": spec.lifecycle.value,
            "transform": transform,
            "content": content,
            "behavior": behavior,
            "accessibility": accessibility,
            "actions": list(spec.actions),
            "metadata": spec.metadata,
        }

    def _calculate_transform(
        self,
        spec: JennySpec,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate 3D transform from spec position and size."""

        # Position from spec (x, y, z tuple)
        pos = spec.position
        position = {
            "x": pos[0] if pos[0] != 0 else self.DEFAULT_POSITION["x"],
            "y": pos[1] if pos[1] != 0 else self.DEFAULT_POSITION["y"],
            "z": pos[2] if pos[2] != 0 else self.DEFAULT_POSITION["z"],
        }

        # Adjust position based on priority (higher priority = closer to user)
        if spec.priority > 0:
            # Move 0.1m closer per priority level
            position["z"] = position["z"] + (spec.priority * 0.1)

        # Rotation (identity quaternion by default)
        rotation = options.get("rotation", {"x": 0, "y": 0, "z": 0, "w": 1})

        # Scale from size
        scale = self.SIZE_TO_METERS.get(spec.size, self.SIZE_TO_METERS[PanelSizeJenny.MEDIUM])

        return {
            "position": position,
            "rotation": rotation,
            "scale": dict(scale),  # Copy to avoid mutation
        }

    def _get_ar_content(self, spec: JennySpec) -> dict[str, Any]:
        """Get AR-optimized content from spec."""
        content = dict(spec.content)
        panel_type = spec.panel_type

        # Add AR-specific content hints
        ar_hints = {}

        if panel_type == PanelTypeJenny.TEXT:
            ar_hints["render_mode"] = "billboard"  # Text faces user
            ar_hints["max_chars"] = 500  # Limit for readability

        elif panel_type == PanelTypeJenny.GRAPH:
            ar_hints["render_mode"] = "3d"  # Graphs can be 3D
            ar_hints["interactive"] = True
            ar_hints["node_scale"] = 0.05  # Node size in meters

        elif panel_type == PanelTypeJenny.CONFIDENCE:
            ar_hints["render_mode"] = "gauge_3d"
            ar_hints["glow_enabled"] = True  # Visual emphasis

        elif panel_type == PanelTypeJenny.TIMELINE:
            ar_hints["render_mode"] = "ribbon"
            ar_hints["depth_enabled"] = True

        elif panel_type == PanelTypeJenny.CODE:
            ar_hints["render_mode"] = "billboard"
            ar_hints["syntax_highlight"] = True
            ar_hints["font_size_m"] = 0.015  # Font size in meters

        elif panel_type == PanelTypeJenny.TABLE:
            ar_hints["render_mode"] = "flat"
            ar_hints["scrollable"] = True

        elif panel_type == PanelTypeJenny.REASONING:
            ar_hints["render_mode"] = "chain_3d"  # 3D reasoning chain
            ar_hints["step_spacing_m"] = 0.2  # Space between steps

        content["ar_hints"] = ar_hints
        return content

    def _get_behavior(
        self,
        spec: JennySpec,
        options: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate AR behavior settings."""

        # Determine billboard mode (panel faces user)
        billboard = spec.panel_type in [
            PanelTypeJenny.TEXT,
            PanelTypeJenny.CONFIDENCE,
            PanelTypeJenny.CODE,
            PanelTypeJenny.TABLE,
            PanelTypeJenny.METRIC,
        ]

        # Gaze targeting for interactive panels
        gaze_target = spec.panel_type in [
            PanelTypeJenny.ACTIONS,
            PanelTypeJenny.GRAPH,
        ]

        # Anchor type
        anchor_type = options.get("anchor_type", "world")
        if spec.metadata.get("follow_device"):
            anchor_type = "device"

        # Dissolution behavior
        dissolution = None
        if spec.dissolution_trigger:
            dissolution = {
                "trigger": spec.dissolution_trigger.value,
                "animation": "fade_dissolve",
                "duration_ms": 300,
            }

        return {
            "billboard": billboard,
            "gaze_target": gaze_target,
            "anchor_type": anchor_type,
            "grabbable": options.get("grabbable", True),
            "resizable": options.get("resizable", False),
            "occlusion": options.get("occlusion", True),
            "cast_shadow": options.get("cast_shadow", False),
            "dissolution": dissolution,
        }

    def _get_accessibility(self, spec: JennySpec) -> dict[str, Any]:
        """Generate AR accessibility features."""

        # Spatial audio description
        audio_description = None
        if spec.title:
            audio_description = f"{spec.panel_type.value} panel: {spec.title}"

        # Haptic feedback settings
        haptics = {
            "on_focus": "light_pulse",
            "on_select": "confirm_tap",
            "on_dismiss": "release",
        }

        return {
            "label": spec.title or f"{spec.panel_type.value} panel",
            "description": spec.subtitle,
            "audio_description": audio_description,
            "haptics": haptics,
            "high_contrast_available": True,
            "magnification_support": True,
            "voice_commands": [
                f"select {spec.title}" if spec.title else None,
                "dismiss",
                "pin",
                "expand",
            ],
        }


# ============================================================================
# Factory Functions (Registry-Backed)
# ============================================================================

def create_renderer(target: RenderTarget = RenderTarget.HTML) -> JennyRendererBase:
    """
    Create a renderer for the specified target.

    Uses the RendererRegistry for plugin-based renderer selection.
    Falls back to hardcoded renderers if registry not initialized.

    Args:
        target: Output format

    Returns:
        Appropriate renderer instance

    Raises:
        UnsupportedTargetError: If no renderer for target
    """
    # Try registry first
    try:
        registry = _get_registry()
        renderer = registry.get_for_target(target)
        if renderer:
            return renderer
    except Exception as e:
        logger.debug(f"Registry lookup failed, using fallback: {e}")

    # Fallback to hardcoded (ensures backward compatibility)
    if target == RenderTarget.HTML:
        return HTMLRenderer()
    elif target == RenderTarget.TERMINAL:
        return TerminalRenderer()
    elif target == RenderTarget.JSON:
        return JSONRenderer()
    elif target == RenderTarget.REACT:
        return ReactRenderer()
    elif target == RenderTarget.AR:
        return ARRenderer()
    else:
        raise UnsupportedTargetError(f"No renderer available for {target.value}")


def get_default_renderer() -> JennyRendererBase:
    """Get the default HTML renderer."""
    return create_renderer(RenderTarget.HTML)


def get_renderer_by_name(name: str) -> JennyRendererBase | None:
    """
    Get a renderer by its unique name.

    Args:
        name: Renderer name (e.g., "html", "terminal", "json")

    Returns:
        Renderer instance or None if not found
    """
    try:
        registry = _get_registry()
        return registry.get_by_name(name)
    except Exception:
        return None


def list_available_renderers() -> list[dict[str, Any]]:
    """
    List all registered renderers.

    Returns:
        List of renderer info dicts with name, priority, targets
    """
    try:
        registry = _get_registry()
        return registry.list_renderers()
    except Exception:
        # Return hardcoded list as fallback
        return [
            {"name": "html", "priority": 10, "targets": ["html"], "concurrent": True},
            {"name": "terminal", "priority": 5, "targets": ["terminal"], "concurrent": True},
            {"name": "json", "priority": 5, "targets": ["json"], "concurrent": True},
            {"name": "react", "priority": 8, "targets": ["react"], "concurrent": True},
            {"name": "ar", "priority": 7, "targets": ["ar"], "concurrent": True},
        ]


# ============================================================================
# Auto-Registration (on import)
# ============================================================================

def _auto_register_renderers():
    """Register built-in renderers with the registry."""
    try:
        registry = _get_registry()
        # Register with priorities (HTML is default)
        registry.register(HTMLRenderer, priority=10, name="html")
        registry.register(TerminalRenderer, priority=5, name="terminal")
        registry.register(JSONRenderer, priority=5, name="json")
        registry.register(ReactRenderer, priority=8, name="react")  # Phase M4
        registry.register(ARRenderer, priority=7, name="ar")  # Phase M6
        logger.debug("Built-in Jenny renderers registered")
    except Exception as e:
        logger.debug(f"Auto-registration skipped: {e}")


# Defer registration to avoid import-time issues
# Will be triggered on first registry access
import atexit

atexit.register(lambda: None)  # Ensure clean shutdown


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Errors
    'RenderError',
    'UnsupportedTargetError',
    # Base
    'JennyRendererBase',
    # Implementations
    'HTMLRenderer',
    'TerminalRenderer',
    'JSONRenderer',
    'ReactRenderer',  # Phase M4
    'ARRenderer',  # Phase M6
    # Factory
    'create_renderer',
    'get_default_renderer',
    'get_renderer_by_name',
    'list_available_renderers',
]


# ============================================================================
# Initialize (trigger auto-registration)
# ============================================================================

# Register built-in renderers when this module is imported
_auto_register_renderers()
