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

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
import html

from .jenny_spec import (
    JennySpec,
    LifecycleStage,
    PanelTypeJenny,
    PanelSizeJenny,
    BindingMode,
    DissolutionTrigger,
)
from HoloLoom.protocols.jenny import RenderTarget, JennyRendererProtocol


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
    def supported_targets(self) -> List[RenderTarget]:
        """Return list of supported render targets."""
        ...

    def supports_target(self, target: RenderTarget) -> bool:
        """Check if renderer supports a target format."""
        return target in self.supported_targets

    async def render(
        self,
        specs: List[JennySpec],
        target: RenderTarget = RenderTarget.HTML,
        options: Optional[Dict[str, Any]] = None
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
        options: Optional[Dict[str, Any]] = None
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
        specs: List[JennySpec],
        target: RenderTarget,
        options: Dict[str, Any]
    ) -> str:
        """Internal: Render multiple specs."""
        ...

    @abstractmethod
    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: Dict[str, Any]
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
    - Accessibility (ARIA labels, keyboard nav)
    - Dark/light theme support
    - Responsive grid layout

    Usage:
        renderer = HTMLRenderer()
        html = await renderer.render(specs)
        # Embed in page or save to file
    """

    # CSS class prefixes
    PANEL_CLASS = "jenny-panel"
    CONTAINER_CLASS = "jenny-container"

    @property
    def supported_targets(self) -> List[RenderTarget]:
        return [RenderTarget.HTML]

    async def _render_multiple(
        self,
        specs: List[JennySpec],
        target: RenderTarget,
        options: Dict[str, Any]
    ) -> str:
        """Render multiple specs to HTML dashboard."""
        include_styles = options.get('include_styles', True)
        include_scripts = options.get('include_scripts', True)
        title = options.get('title', 'Jenny Dashboard')

        # Sort by priority (lower = first)
        sorted_specs = sorted(specs, key=lambda s: s.priority)

        # Render individual panels
        panels_html = []
        for spec in sorted_specs:
            panel = await self._render_single(spec, target, options)
            panels_html.append(panel)

        # Build container
        container = f'''<div class="{self.CONTAINER_CLASS}" role="main" aria-label="{title}">
    {''.join(panels_html)}
</div>'''

        # Add styles and scripts if requested
        if include_styles or include_scripts:
            head_content = []
            if include_styles:
                head_content.append(self._get_styles())
            if include_scripts:
                head_content.append(self._get_scripts())

            return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    {''.join(head_content)}
</head>
<body>
    {container}
</body>
</html>'''

        return container

    async def _render_single(
        self,
        spec: JennySpec,
        target: RenderTarget,
        options: Dict[str, Any]
    ) -> str:
        """Render single spec to HTML panel."""
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

        # Build panel HTML
        return f'''<article class="{' '.join(classes)}" {data_str} role="region" aria-label="{html.escape(spec.title or spec.panel_type.value)}">
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

        else:
            # Fallback: JSON dump
            return f'<pre class="jenny-raw">{html.escape(json.dumps(content, indent=2, default=str))}</pre>'

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
    def supported_targets(self) -> List[RenderTarget]:
        return [RenderTarget.TERMINAL]

    async def _render_multiple(
        self,
        specs: List[JennySpec],
        target: RenderTarget,
        options: Dict[str, Any]
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
        options: Dict[str, Any]
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

    def _render_terminal_content(self, spec: JennySpec, max_width: int) -> List[str]:
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

    def _render_compact(self, specs: List[JennySpec], no_color: bool) -> str:
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
    def supported_targets(self) -> List[RenderTarget]:
        return [RenderTarget.JSON]

    async def _render_multiple(
        self,
        specs: List[JennySpec],
        target: RenderTarget,
        options: Dict[str, Any]
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
        options: Dict[str, Any]
    ) -> str:
        """Render single spec to JSON."""
        indent = options.get('indent', 2)
        return json.dumps(spec.to_dict(), indent=indent, default=str)


# ============================================================================
# Factory Functions
# ============================================================================

def create_renderer(target: RenderTarget = RenderTarget.HTML) -> JennyRendererBase:
    """
    Create a renderer for the specified target.

    Args:
        target: Output format

    Returns:
        Appropriate renderer instance

    Raises:
        UnsupportedTargetError: If no renderer for target
    """
    if target == RenderTarget.HTML:
        return HTMLRenderer()
    elif target == RenderTarget.TERMINAL:
        return TerminalRenderer()
    elif target == RenderTarget.JSON:
        return JSONRenderer()
    else:
        raise UnsupportedTargetError(f"No renderer available for {target.value}")


def get_default_renderer() -> JennyRendererBase:
    """Get the default HTML renderer."""
    return HTMLRenderer()


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
    # Factory
    'create_renderer',
    'get_default_renderer',
]
