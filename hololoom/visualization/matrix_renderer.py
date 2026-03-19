"""
Matrix-Safe HTML Renderer for Jenny Panels
============================================
Renders JennySpec panels into HTML safe for Matrix's `formatted_body`.

Matrix supports a restricted HTML subset (org.matrix.custom.html):
- Allowed: <b>, <i>, <em>, <strong>, <code>, <pre>, <br>, <p>, <ul>, <ol>,
           <li>, <table>, <thead>, <tbody>, <tr>, <th>, <td>, <blockquote>,
           <h1>-<h6>, <hr>, <a href>, <span>, inline style attributes
- NOT allowed: <script>, <style> blocks, CSS classes, data-* attributes,
               <div>, <article>, <section>, <footer>, <header>

Design: Renders only panels that ADD VALUE over raw text. A text re-wrap is
noise; a table, confidence bar, or source list is signal. The caller decides
whether to send the result — if render_if_useful() returns None, skip it.

Author: HoloLoom Team
Date: 2026-03-11
"""

import html as html_mod
import json
import logging
from typing import Any

from .jenny_spec import JennySpec, PanelTypeJenny

logger = logging.getLogger(__name__)

# Panel types that add visual value in Matrix beyond raw text
_USEFUL_PANEL_TYPES = frozenset({
    PanelTypeJenny.TABLE,
    PanelTypeJenny.CONFIDENCE,
    PanelTypeJenny.SOURCES,
    PanelTypeJenny.CODE,
    PanelTypeJenny.METRIC,
    PanelTypeJenny.REASONING,
    PanelTypeJenny.COMPARISON,
    PanelTypeJenny.GRAPH,
})


def render_if_useful(spec: JennySpec) -> str | None:
    """
    Render a JennySpec to Matrix-safe HTML, but ONLY if it adds value.

    Returns None for panel types that would just re-wrap plain text
    (TEXT, WHY, ACTIONS). The caller should skip sending those.

    Returns:
        Matrix-safe HTML string, or None if the panel isn't worth sending.
    """
    if spec.panel_type not in _USEFUL_PANEL_TYPES:
        return None

    rendered = _render_panel(spec)
    if not rendered or len(rendered.strip()) < 10:
        return None

    return rendered


def render_panel(spec: JennySpec) -> str:
    """
    Render a JennySpec to Matrix-safe HTML unconditionally.

    Use render_if_useful() in production — this is for testing/debugging.
    """
    return _render_panel(spec)


def render_panels_block(specs: list[JennySpec]) -> str | None:
    """
    Render multiple specs into a single Matrix HTML block.

    Only includes panels that add value. Returns None if nothing is useful.
    """
    parts = []
    for spec in specs:
        rendered = render_if_useful(spec)
        if rendered:
            parts.append(rendered)

    if not parts:
        return None

    return "<hr>".join(parts)


def _render_panel(spec: JennySpec) -> str:
    """Dispatch to type-specific renderer."""
    content = spec.content
    title = spec.title

    # Title header (if present)
    header = f"<h4>{_esc(title)}</h4>" if title else ""

    body = _RENDERERS.get(spec.panel_type, _render_fallback)(content)

    return f"{header}{body}"


# ============================================================================
# Type-specific renderers (Matrix-safe HTML only)
# ============================================================================

def _render_table(content: dict[str, Any]) -> str:
    headers = content.get("headers", [])
    rows = content.get("rows", [])

    if not headers and not rows:
        return ""

    parts = ["<table>"]
    if headers:
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(f"<th><strong>{_esc(str(h))}</strong></th>")
        parts.append("</tr></thead>")

    if rows:
        parts.append("<tbody>")
        for row in rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{_esc(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")

    parts.append("</table>")
    return "".join(parts)


def _render_confidence(content: dict[str, Any]) -> str:
    value = content.get("value", 0)
    pct = int(value * 100)

    # Visual bar using Unicode blocks (works in all Matrix clients)
    bar_len = 20
    filled = int(bar_len * value)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    # Color hint via inline style
    if pct >= 80:
        color = "#00cc66"
        label = "High"
    elif pct >= 60:
        color = "#ccaa00"
        label = "Medium"
    else:
        color = "#cc4444"
        label = "Low"

    return (
        f'<p><strong>Confidence:</strong> '
        f'<span style="color:{color}"><strong>{pct}%</strong></span> ({label})</p>'
        f'<pre><code>{bar}</code></pre>'
    )


def _render_sources(content: dict[str, Any]) -> str:
    sources = content.get("sources", [])
    if not sources:
        return "<p><em>No sources</em></p>"

    parts = ["<ol>"]
    for source in sources[:10]:
        if isinstance(source, dict):
            title = source.get("title", "Source")
            url = source.get("url")
            if url:
                parts.append(f'<li><a href="{_esc(url)}">{_esc(title)}</a></li>')
            else:
                parts.append(f"<li>{_esc(title)}</li>")
        else:
            parts.append(f"<li>{_esc(str(source))}</li>")
    parts.append("</ol>")

    if len(sources) > 10:
        parts.append(f"<p><em>...and {len(sources) - 10} more</em></p>")

    return "".join(parts)


def _render_code(content: dict[str, Any]) -> str:
    code = content.get("code", "")
    content.get("language", "")

    if not code:
        return ""

    # Matrix supports <pre><code> but not language-specific highlighting
    return f"<pre><code>{_esc(code)}</code></pre>"


def _render_metric(content: dict[str, Any]) -> str:
    value = content.get("value", 0)
    unit = content.get("unit", "")
    label = content.get("label", "Metric")
    fmt = content.get("format", "")

    # Format the value
    if fmt == "number" and isinstance(value, (int, float)):
        display = f"{value:,.1f}" if isinstance(value, float) else f"{value:,}"
    else:
        display = str(value)

    return (
        f'<p><strong>{_esc(label)}:</strong> '
        f'<span style="color:#00d9ff"><strong>{_esc(display)}{_esc(unit)}</strong></span></p>'
    )


def _render_reasoning(content: dict[str, Any]) -> str:
    steps = content.get("steps", [])
    if not steps:
        return ""

    parts = ["<ol>"]
    for step in steps:
        parts.append(f"<li>{_esc(str(step))}</li>")
    parts.append("</ol>")
    return "".join(parts)


def _render_comparison(content: dict[str, Any]) -> str:
    """Render comparison as a two-column table."""
    items = content.get("items", [])
    headers = content.get("headers", ["Option A", "Option B"])

    if not items:
        return ""

    parts = ["<table><thead><tr>"]
    parts.append("<th><strong>Aspect</strong></th>")
    for h in headers:
        parts.append(f"<th><strong>{_esc(str(h))}</strong></th>")
    parts.append("</tr></thead><tbody>")

    for item in items:
        parts.append("<tr>")
        if isinstance(item, dict):
            parts.append(f"<td><strong>{_esc(item.get('aspect', ''))}</strong></td>")
            for val in item.get("values", []):
                parts.append(f"<td>{_esc(str(val))}</td>")
        elif isinstance(item, (list, tuple)):
            for val in item:
                parts.append(f"<td>{_esc(str(val))}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    return "".join(parts)


def _render_graph(content: dict[str, Any]) -> str:
    """Render graph as a text summary (graphs can't render in Matrix)."""
    nodes = content.get("nodes", [])
    edges = content.get("edges", [])
    is_conversation = content.get("conversation_graph", False)

    if not nodes:
        return "<p><em>Empty graph</em></p>"

    heading = "Conversation Topics" if is_conversation else "Knowledge Graph"
    parts = [f"<p><strong>{heading}</strong> ({len(nodes)} nodes, {len(edges)} edges)</p>"]

    # Show up to 8 nodes as a list
    if nodes:
        parts.append("<ul>")
        for node in nodes[:8]:
            if isinstance(node, dict):
                label = node.get("label", node.get("id", "?"))
                weight = node.get("weight")
                if weight and is_conversation:
                    label = f"{label} ({int(weight)} mentions)"
            else:
                label = str(node)
            parts.append(f"<li>{_esc(label)}</li>")
        if len(nodes) > 8:
            parts.append(f"<li><em>...and {len(nodes) - 8} more</em></li>")
        parts.append("</ul>")

    return "".join(parts)


def _render_fallback(content: dict[str, Any]) -> str:
    """Fallback: JSON in a code block."""
    return f"<pre><code>{_esc(json.dumps(content, indent=2, default=str))}</code></pre>"


# Renderer dispatch table
_RENDERERS = {
    PanelTypeJenny.TABLE: _render_table,
    PanelTypeJenny.CONFIDENCE: _render_confidence,
    PanelTypeJenny.SOURCES: _render_sources,
    PanelTypeJenny.CODE: _render_code,
    PanelTypeJenny.METRIC: _render_metric,
    PanelTypeJenny.REASONING: _render_reasoning,
    PanelTypeJenny.COMPARISON: _render_comparison,
    PanelTypeJenny.GRAPH: _render_graph,
}


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html_mod.escape(text, quote=True)
