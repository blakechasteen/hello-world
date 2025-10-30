#!/usr/bin/env python3
"""
HTMLRenderer - Edward Tufte Dashboards as Standalone HTML
==========================================================
Renders Dashboard objects as beautiful, interactive HTML using:
- Plotly.js for charts (timelines, networks, heatmaps)
- Tailwind CSS for styling (CDN, no build step)
- Edward Tufte principles (minimal chrome, maximize data-ink ratio)

Author: Claude Code with HoloLoom architecture
Date: October 28, 2025
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .dashboard import (
    Dashboard, Panel, PanelType, LayoutType, PanelSize,
    LAYOUT_CONFIGS, PANEL_SIZE_CLASSES, STAGE_COLORS
)


class HTMLRenderer:
    """
    Renders Dashboard objects as standalone HTML files.

    Edward Tufte Principles Applied:
    1. Maximize data-ink ratio - minimal decorative elements
    2. Show data variation, not design variation - consistent styling
    3. Reveal data at several levels - overview + detail on hover
    4. Serve a clear purpose - every visual element has meaning
    5. Closely integrate graphics with text - panels flow naturally

    Usage:
        renderer = HTMLRenderer()
        html = renderer.render(dashboard)
        with open('output.html', 'w') as f:
            f.write(html)
    """

    def __init__(self, theme: str = 'light'):
        """
        Initialize HTML renderer.

        Args:
            theme: Color theme ('light' or 'dark')
        """
        self.theme = theme

    def render(self, dashboard: Dashboard) -> str:
        """
        Render complete dashboard as standalone HTML.

        Args:
            dashboard: Dashboard object to render

        Returns:
            Complete HTML string ready to save to file
        """
        # Generate panel HTML
        panels_html = self._render_panels(dashboard)

        # Generate metadata footer
        metadata_html = self._render_metadata(dashboard)

        # Load interactivity JavaScript
        interactivity_js = self._load_interactivity_js()

        # Assemble complete HTML
        return self._assemble_html(
            title=dashboard.title,
            layout_class=LAYOUT_CONFIGS.get(dashboard.layout, LAYOUT_CONFIGS[LayoutType.FLOW]),
            panels_html=panels_html,
            metadata_html=metadata_html,
            interactivity_js=interactivity_js
        )

    def _render_panels(self, dashboard: Dashboard) -> str:
        """
        Render all panels in dashboard.

        Args:
            dashboard: Dashboard object

        Returns:
            HTML string for all panels
        """
        panel_htmls = []

        for panel in dashboard.panels:
            # Render based on panel type
            renderer = self._get_panel_renderer(panel.type)
            if renderer:
                panel_html = renderer(panel)
                panel_htmls.append(panel_html)

        return '\n'.join(panel_htmls)

    def _get_panel_renderer(self, panel_type: PanelType):
        """Get renderer function for panel type."""
        renderers = {
            PanelType.METRIC: self._render_metric,
            PanelType.TIMELINE: self._render_timeline,
            PanelType.NETWORK: self._render_network,
            PanelType.HEATMAP: self._render_heatmap,
            PanelType.TEXT: self._render_text,
            PanelType.DISTRIBUTION: self._render_distribution,
            PanelType.SCATTER: self._render_scatter,
            PanelType.LINE: self._render_line,
            PanelType.BAR: self._render_bar,
            PanelType.INSIGHT: self._render_insight,
        }
        return renderers.get(panel_type)

    # ========================================================================
    # Panel Renderers (one per PanelType)
    # ========================================================================

    def _render_metric(self, panel: Panel) -> str:
        """
        Render METRIC panel (big number with semantic color).

        Tufte: Maximize data-ink ratio - just the number, no decoration.
        Enhanced with Tufte-style sparklines when trend data available.
        Modern: Semantic HTML with data attributes, ARIA labels.
        """
        data = panel.data
        value = data.get('value', 0)
        label = data.get('label', panel.title)
        unit = data.get('unit', '')
        color = data.get('color', 'blue')
        formatted = data.get('formatted', f"{value:.2f}{unit}")

        # Tufte sparkline: optional trend data
        trend = data.get('trend', [])  # List of recent values
        trend_direction = data.get('trend_direction', '')  # 'up', 'down', 'flat'

        # Generate sparkline SVG if trend data available
        sparkline_html = ""
        if trend and len(trend) >= 2:
            sparkline_html = f'<div class="sparkline-container">{self._generate_sparkline(trend, color)}</div>'

        # Trend indicator with semantic class
        trend_indicator = ""
        if trend_direction == 'up':
            trend_indicator = '<span class="trend-indicator trend-up" aria-label="Trending up">&#9650;</span>'
        elif trend_direction == 'down':
            trend_indicator = '<span class="trend-indicator trend-down" aria-label="Trending down">&#9660;</span>'
        elif trend_direction == 'flat':
            trend_indicator = '<span class="trend-indicator trend-flat" aria-label="Stable">&#9644;</span>'

        # Semantic HTML with modern attributes
        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="metric"
                 data-size="{panel.size.value}"
                 data-color="{color}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <div class="metric-label" id="panel-{panel.id}-title">{label}</div>
                <div style="display: flex; align-items: baseline; gap: var(--space-2);">
                    <div class="metric-value numeric" data-color="{color}" aria-live="polite">
                        {formatted}
                    </div>
                    {trend_indicator}
                </div>
                {sparkline_html}
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
            </div>
        </article>
        """

    def _generate_sparkline(self, values: list, color: str) -> str:
        """
        Generate Tufte-style sparkline SVG.

        Sparkline principles (from Tufte):
        - Intense, simple, word-sized graphics
        - High resolution
        - Typically 1-2 inches wide
        - No axes or labels (context provided by surrounding text)
        """
        if not values or len(values) < 2:
            return ""

        # Sparkline dimensions (compact)
        width = 100
        height = 30
        padding = 2

        # Normalize values to fit in sparkline
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1

        # Generate SVG path points
        points = []
        for i, val in enumerate(values):
            x = padding + (i / (len(values) - 1)) * (width - 2 * padding)
            y = height - padding - ((val - min_val) / value_range) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        path_data = "M " + " L ".join(points)

        # Color mapping for sparkline (OKLCH-based)
        stroke_colors = {
            'green': 'oklch(70% 0.16 145)',
            'red': 'oklch(65% 0.18 25)',
            'blue': 'oklch(65% 0.15 250)',
            'yellow': 'oklch(75% 0.15 90)',
            'orange': 'oklch(70% 0.16 50)',
            'purple': 'oklch(65% 0.14 300)',
        }
        stroke_color = stroke_colors.get(color, 'oklch(58% 0.01 270)')

        return f"""
            <svg class="sparkline-svg"
                 width="{width}"
                 height="{height}"
                 role="img"
                 aria-label="Trend: {len(values)} data points">
                <path d="{path_data}"
                      fill="none"
                      stroke="{stroke_color}"
                      stroke-width="1.5"
                      opacity="0.7"/>
                <!-- Last point indicator -->
                <circle cx="{points[-1].split(',')[0]}"
                        cy="{points[-1].split(',')[1]}"
                        r="2"
                        fill="{stroke_color}"/>
            </svg>
            <span style="font-size: var(--font-size-xs); color: var(--color-text-tertiary); margin-left: var(--space-2);">
                Last {len(values)} queries
            </span>
        """

    def _render_timeline(self, panel: Panel) -> str:
        """
        Render TIMELINE panel (waterfall chart of execution stages).

        Tufte: Show mechanism and causality through time.
        Enhanced with automatic bottleneck detection and optimization suggestions.
        """
        data = panel.data
        stages = data.get('stages', [])
        durations = data.get('durations', [])
        percentages = data.get('percentages', [])

        # Get bottleneck info (from Phase 2.3 enhancement)
        bottleneck = data.get('bottleneck', {})
        bottleneck_detected = bottleneck.get('detected', False)
        bottleneck_stage = bottleneck.get('stage', '')
        bottleneck_percentage = bottleneck.get('percentage', 0)
        optimization = bottleneck.get('optimization', '')

        # Create Plotly waterfall chart
        plot_id = f"plot_{panel.id}"
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.LARGE])

        # Use colors from data (from bottleneck detection) or fallback to defaults
        colors = data.get('colors', [STAGE_COLORS.get(stage.lower(), STAGE_COLORS['default']) for stage in stages])

        # Bottleneck warning banner (if detected)
        warning_html = ""
        if bottleneck_detected:
            # Icon: ⚠️ for moderate (40-50%), 🔴 for severe (>50%)
            icon = "🔴" if bottleneck_percentage > 50 else "⚠️"
            # Color: red for severe, orange for moderate
            bg_color = "bg-red-50 border-red-200" if bottleneck_percentage > 50 else "bg-orange-50 border-orange-200"
            text_color = "text-red-800" if bottleneck_percentage > 50 else "text-orange-800"

            warning_html = f"""
            <div class="{bg_color} border-l-4 p-4 mb-4 rounded-r">
                <div class="flex items-start">
                    <div class="flex-shrink-0 text-2xl mr-3">{icon}</div>
                    <div class="flex-1">
                        <div class="text-sm font-semibold {text_color} mb-1">
                            Bottleneck Detected: {bottleneck_stage} ({bottleneck_percentage:.0f}% of total time)
                        </div>
                        <div class="text-xs {text_color} opacity-90">
                            {optimization}
                        </div>
                    </div>
                </div>
            </div>
            """

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="timeline"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                {warning_html}
                <div id="{plot_id}" style="height: 450px;"></div>
            </div>
        </article>
        <script>
        (function() {{
            var data = [{{
                type: 'bar',
                x: {durations},
                y: {stages},
                orientation: 'h',
                marker: {{
                    color: {colors}
                }},
                text: {[f'{d:.1f}ms ({p:.0f}%)' for d, p in zip(durations, percentages)]},
                textposition: 'auto',
                hovertemplate: '%{{y}}<br>%{{x:.1f}}ms (%{{text}})<extra></extra>'
            }}];

            // Get theme-aware colors (Plotly-compatible RGB/hex)
            const getThemeColors = () => {{
                const theme = document.documentElement.getAttribute('data-theme') || 'light';
                if (theme === 'dark') {{
                    return {{
                        bg: '#1f2937',        // Dark elevated background
                        bgSecondary: '#111827', // Dark secondary
                        text: '#f3f4f6',      // Light text
                        grid: '#374151'       // Dark grid
                    }};
                }} else {{
                    return {{
                        bg: '#ffffff',        // White background
                        bgSecondary: '#f9fafb', // Light gray
                        text: '#1f2937',      // Dark text
                        grid: '#e5e7eb'       // Light grid
                    }};
                }}
            }};

            const colors = getThemeColors();

            var layout = {{
                margin: {{ l: 120, r: 20, t: 20, b: 40 }},
                xaxis: {{
                    title: 'Duration (ms)',
                    showgrid: true,
                    gridcolor: colors.grid,
                    color: colors.text
                }},
                yaxis: {{
                    autorange: 'reversed',
                    showgrid: false,
                    color: colors.text
                }},
                paper_bgcolor: colors.bg,
                plot_bgcolor: colors.bg,
                font: {{ family: 'system-ui, -apple-system, sans-serif', size: 12, color: colors.text }},
                showlegend: false
            }};

            var config = {{ responsive: true, displayModeBar: false }};
            Plotly.newPlot('{plot_id}', data, layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'paper_bgcolor': newColors.bg,
                    'plot_bgcolor': newColors.bg,
                    'xaxis.gridcolor': newColors.grid,
                    'xaxis.color': newColors.text,
                    'yaxis.color': newColors.text,
                    'font.color': newColors.text
                }});
            }});
        }})();
        </script>
        """

    def _render_network(self, panel: Panel) -> str:
        """
        Render NETWORK panel (graph of knowledge threads).

        Tufte: Show relationships and structure.

        Uses D3.js force-directed graph for interactive visualization.
        """
        data = panel.data
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        node_count = data.get('node_count', len(nodes))

        plot_id = f"plot_{panel.id}"
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.MEDIUM])

        # If no proper nodes/edges structure, fall back to simple list
        if not nodes or not isinstance(nodes, list):
            threads = data.get('threads', [])
            if isinstance(threads, list):
                nodes_html = ', '.join([
                    f"<span style='display: inline-block; padding: 0.25rem 0.5rem; background: rgba(99, 102, 241, 0.1); color: rgb(79, 70, 229); border-radius: 0.375rem; font-size: 0.75rem; margin: 0.25rem;'>{t}</span>"
                    for t in threads[:10]
                ])
                return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="network"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">
                    {nodes_html}
                </div>
                <div style="font-size: 0.75rem; color: var(--color-text-secondary);">{len(threads)} thread(s) activated</div>
            </div>
        </article>
                """

        # D3.js force-directed graph with enhanced interactivity
        import json
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="network"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                <div style="font-size: 0.75rem; color: var(--color-text-secondary); margin-bottom: 0.5rem;">
                    <span style="margin-right: 1rem;">{node_count} node(s), {len(edges)} edge(s)</span>
                    <span style="opacity: 0.7;">💡 Drag nodes • Scroll to zoom • Double-click to reset</span>
                </div>
                <div id="{plot_id}" style="height: 600px; border: 1px solid var(--color-border-subtle); border-radius: var(--radius-lg); background: var(--color-bg-secondary); position: relative;"></div>
            </div>
        </article>
        <script>
        (function() {{
            // D3.js Enhanced Force-Directed Graph
            const width = document.getElementById('{plot_id}').clientWidth;
            const height = 450;

            const svg = d3.select('#{plot_id}')
                .append('svg')
                .attr('width', width)
                .attr('height', height);

            // Add zoom behavior
            const g = svg.append('g');
            const zoom = d3.zoom()
                .scaleExtent([0.3, 3])
                .on('zoom', (event) => {{
                    g.attr('transform', event.transform);
                }});

            svg.call(zoom);

            // Reset zoom on double-click
            svg.on('dblclick.zoom', () => {{
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity
                );
            }});

            const nodes = {nodes_json};
            const links = {edges_json};

            // Create force simulation with better physics
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(120))
                .force('charge', d3.forceManyBody().strength(-400))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide().radius(d => (d.size || 12) + 15));

            // Draw edges with gradient
            const link = g.append('g')
                .attr('class', 'links')
                .selectAll('line')
                .data(links)
                .enter().append('line')
                .attr('stroke', '#cbd5e1')
                .attr('stroke-width', 1.5)
                .attr('stroke-opacity', 0.6);

            // Draw nodes
            const node = g.append('g')
                .attr('class', 'nodes')
                .selectAll('g')
                .data(nodes)
                .enter().append('g')
                .style('cursor', 'pointer')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended))
                .on('mouseover', function(event, d) {{
                    // Highlight on hover
                    d3.select(this).select('circle')
                        .transition().duration(200)
                        .attr('r', (d.size || 12) * 1.3)
                        .attr('stroke-width', 3);
                }})
                .on('mouseout', function(event, d) {{
                    // Reset on mouseout
                    d3.select(this).select('circle')
                        .transition().duration(200)
                        .attr('r', d.size || 12)
                        .attr('stroke-width', 2);
                }});

            // Node circles with drop shadow
            node.append('circle')
                .attr('r', d => d.size || 12)
                .attr('fill', d => d.color || '#6366f1')
                .attr('stroke', '#fff')
                .attr('stroke-width', 2)
                .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

            // Node labels
            node.append('text')
                .text(d => d.label)
                .attr('x', 0)
                .attr('y', d => (d.size || 12) + 18)
                .attr('text-anchor', 'middle')
                .style('font-size', d => d.type === 'query' ? '12px' : '10px')
                .style('font-weight', d => d.type === 'query' ? 600 : 400)
                .style('fill', '#374151')
                .style('pointer-events', 'none');

            // Enhanced tooltip with full label
            node.append('title')
                .text(d => {{
                    const fullLabel = d.fullLabel || d.label;
                    const type = d.type === 'query' ? 'Query Node' : 'Knowledge Thread';
                    return `${{type}}\\n${{fullLabel}}`;
                }});

            // Update positions on simulation tick
            simulation.on('tick', () => {{
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('transform', d => `translate(${{d.x}},${{d.y}})`);
            }});

            // Enhanced drag functions
            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;

                // Highlight connected nodes
                highlightConnected(event.subject);
            }}

            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}

            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;

                // Reset highlight
                resetHighlight();
            }}

            function highlightConnected(d) {{
                const connectedNodes = new Set();
                connectedNodes.add(d.id);

                links.forEach(link => {{
                    if (link.source.id === d.id) connectedNodes.add(link.target.id);
                    if (link.target.id === d.id) connectedNodes.add(link.source.id);
                }});

                node.style('opacity', n => connectedNodes.has(n.id) ? 1 : 0.2);
                link.style('opacity', l =>
                    l.source.id === d.id || l.target.id === d.id ? 0.8 : 0.1
                );
            }}

            function resetHighlight() {{
                node.style('opacity', 1);
                link.style('opacity', 0.6);
            }}
        }})();
        </script>
        """

    def _render_heatmap(self, panel: Panel) -> str:
        """
        Render HEATMAP panel (semantic dimensions, comparison matrix).

        Tufte: Small multiples - compare many dimensions at once.
        Enhanced with true semantic dimension visualization.

        Supports two data formats:
        1. Dimension format: dimension_names, dimension_scores (1D heatmap)
        2. Matrix format: x_labels, y_labels, values (2D heatmap)
        """
        data = panel.data
        cache_enabled = data.get('cache_enabled', False)
        hit_rate = data.get('hit_rate', 0)

        plot_id = f"plot_{panel.id}"

        # Check for matrix format (x_labels, y_labels, values)
        x_labels = data.get('x_labels', [])
        y_labels = data.get('y_labels', [])
        values = data.get('values', [])

        # Check for dimension format (dimension_names, dimension_scores)
        dim_names = data.get('dimension_names', [])
        dim_scores = data.get('dimension_scores', [])
        total_dims = data.get('total_dimensions', 0)
        showing_top = data.get('showing_top', len(dim_names))

        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.MEDIUM])

        # Matrix format (2D heatmap)
        if x_labels and y_labels and values:
            return self._render_matrix_heatmap(panel, x_labels, y_labels, values, plot_id)

        # Dimension format (1D heatmap)
        elif dim_names and dim_scores:
            return self._render_dimension_heatmap(panel, dim_names, dim_scores, total_dims, showing_top, hit_rate, plot_id)

        # No data - fallback
        else:
            return f"""
            <article class="panel"
                     data-panel-id="{panel.id}"
                     data-panel-type="heatmap"
                     data-size="{panel.size.value}"
                     role="article"
                     aria-labelledby="panel-{panel.id}-title"
                     tabindex="0">
                <div class="panel-content">
                    <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                    {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                    <div style="font-size: 0.875rem; color: var(--color-text-secondary);">
                        Cache: {'Enabled' if cache_enabled else 'Disabled'}<br>
                        Hit Rate: {hit_rate:.1%}<br>
                        <span style="font-style: italic; opacity: 0.7;">No dimension data available</span>
                    </div>
                </div>
            </article>
            """

    def _render_matrix_heatmap(self, panel: Panel, x_labels, y_labels, values, plot_id):
        """Render 2D matrix heatmap (e.g., time vs complexity)"""
        import json

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="heatmap"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                <div id="{plot_id}" style="height: 550px;"></div>
            </div>
        </article>
        <script>
        (function() {{
            const themeColors = getThemeColors();

            var data = [{{
                type: 'heatmap',
                z: {json.dumps(values)},
                x: {json.dumps(x_labels)},
                y: {json.dumps(y_labels)},
                colorscale: [
                    [0, 'rgb(34, 197, 94)'],      // green-500 (fast)
                    [0.5, 'rgb(251, 191, 36)'],   // amber-400 (medium)
                    [1, 'rgb(239, 68, 68)']       // red-500 (slow)
                ],
                hoverongaps: false,
                hovertemplate: '<b>%{{y}}</b><br>%{{x}}: %{{z}}ms<extra></extra>',
                colorbar: {{
                    title: 'Latency (ms)',
                    titleside: 'right',
                    tickmode: 'linear',
                    tick0: 0,
                    dtick: 100,
                    thickness: 15,
                    len: 0.9,
                    bgcolor: 'rgba(0,0,0,0)',
                    tickfont: {{ color: themeColors.text }}
                }}
            }}];

            var layout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: themeColors.text, family: 'Inter, sans-serif' }},
                margin: {{ t: 30, r: 80, b: 60, l: 80 }},
                xaxis: {{
                    title: '',
                    gridcolor: themeColors.grid,
                    tickfont: {{ size: 11 }},
                    showline: true,
                    linecolor: themeColors.grid
                }},
                yaxis: {{
                    title: 'Time of Day',
                    gridcolor: themeColors.grid,
                    tickfont: {{ size: 11 }},
                    showline: true,
                    linecolor: themeColors.grid
                }},
                hoverlabel: {{
                    bgcolor: themeColors.tooltipBg,
                    bordercolor: themeColors.border,
                    font: {{ color: themeColors.text }}
                }}
            }};

            var config = {{
                responsive: true,
                displayModeBar: false
            }};

            Plotly.newPlot('{plot_id}', data, layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'font.color': newColors.text,
                    'xaxis.gridcolor': newColors.grid,
                    'yaxis.gridcolor': newColors.grid,
                    'xaxis.linecolor': newColors.grid,
                    'yaxis.linecolor': newColors.grid
                }});
            }});
        }})();
        </script>
        """

    def _render_dimension_heatmap(self, panel: Panel, dim_names, dim_scores, total_dims, showing_top, hit_rate, plot_id):
        """Render 1D dimension heatmap"""

        # Info banner
        info_html = f"""
        <div class="flex items-center justify-between mb-3 text-xs text-gray-600">
            <div>
                <span class="font-semibold">{showing_top} / {total_dims}</span> dimensions shown
                (sorted by activation strength)
            </div>
            <div>
                Cache Hit Rate: <span class="font-semibold">{hit_rate:.1%}</span>
            </div>
        </div>
        """

        # Create Plotly heatmap
        # Convert scores to 2D array (single row heatmap)
        z_values = [dim_scores]  # Wrap in list to make 2D

        # Color scale: RdBu (red for negative, blue for positive)
        colorscale = 'RdBu_r'  # reversed so positive=blue, negative=red

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="heatmap"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                {info_html}
                <div id="{plot_id}" style="height: 550px;"></div>
            </div>
        </article>
        <script>
        (function() {{
            var data = [{{
                type: 'heatmap',
                z: {z_values},
                x: {dim_names},
                y: ['Query'],
                colorscale: '{colorscale}',
                zmid: 0,  // Center colorscale at zero
                colorbar: {{
                    title: 'Activation',
                    titleside: 'right',
                    tickmode: 'linear',
                    tick0: -1,
                    dtick: 0.5
                }},
                hovertemplate: '%{{x}}<br>Score: %{{z:.3f}}<extra></extra>'
            }}];

            // Get theme-aware colors (Plotly-compatible RGB/hex)
            const getThemeColors = () => {{
                const theme = document.documentElement.getAttribute('data-theme') || 'light';
                if (theme === 'dark') {{
                    return {{
                        bg: '#1f2937',
                        text: '#f3f4f6',
                        grid: '#374151'
                    }};
                }} else {{
                    return {{
                        bg: '#ffffff',
                        text: '#1f2937',
                        grid: '#e5e7eb'
                    }};
                }}
            }};

            const colors = getThemeColors();

            var layout = {{
                margin: {{ l: 80, r: 80, t: 10, b: 120 }},
                xaxis: {{
                    tickangle: -45,
                    tickfont: {{ size: 10 }},
                    showgrid: false,
                    color: colors.text
                }},
                yaxis: {{
                    tickfont: {{ size: 12 }},
                    showgrid: false,
                    color: colors.text
                }},
                paper_bgcolor: colors.bg,
                plot_bgcolor: colors.bg,
                font: {{ family: 'system-ui, -apple-system, sans-serif', color: colors.text }},
                showlegend: false
            }};

            var config = {{ responsive: true, displayModeBar: false }};
            Plotly.newPlot('{plot_id}', data, layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'paper_bgcolor': newColors.bg,
                    'plot_bgcolor': newColors.bg,
                    'xaxis.color': newColors.text,
                    'yaxis.color': newColors.text,
                    'font.color': newColors.text
                }});
            }});
        }})();
        </script>
        """

    def _render_text(self, panel: Panel) -> str:
        """
        Render TEXT panel (query, response, errors).

        Tufte: Context and detail - show the actual content.
        """
        data = panel.data
        content = data.get('content', '')
        label = data.get('label', panel.title)
        length = data.get('length', len(content))

        # Truncate long text with expansion option (unless explicitly disabled)
        truncate = data.get('truncate', True)  # Default to truncating
        if truncate and len(content) > 500:
            display_content = content[:500] + '...'
        else:
            display_content = content

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="text"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                <div class="text-sm" style="color: var(--color-text-primary); line-height: 1.6;">
{display_content}
                </div>
            </div>
        </article>
        """

    def _render_distribution(self, panel: Panel) -> str:
        """
        Render DISTRIBUTION panel (histograms, probability distributions).

        Tufte: Show data variation and uncertainty.
        """
        data = panel.data
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.MEDIUM])

        return f"""
        <div class="{size_class} p-6 rounded-lg shadow-sm border border-gray-200 bg-white">
            <div class="text-lg font-semibold text-gray-800 mb-1">{panel.title}</div>
            {f'<div class="text-sm text-gray-600 mb-4">{panel.subtitle}</div>' if panel.subtitle else ''}
            <div class="text-sm text-gray-500">Distribution visualization (placeholder)</div>
        </div>
        """

    def _render_scatter(self, panel: Panel) -> str:
        """
        Render SCATTER plot panel (correlation, clustering analysis).

        Tufte: Show relationships between variables with minimal decoration.
        """
        data = panel.data
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        labels = data.get('labels', [f'Point {i+1}' for i in range(len(x_values))])
        x_label = data.get('x_label', 'X Axis')
        y_label = data.get('y_label', 'Y Axis')
        colors = data.get('colors', ['#6366f1'] * len(x_values))
        sizes = data.get('sizes', [8] * len(x_values))

        # Calculate correlation if both axes are numeric
        correlation = data.get('correlation', None)
        correlation_text = ""
        if correlation is not None:
            strength = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"
            direction = "positive" if correlation > 0 else "negative"
            correlation_text = f'<div class="text-sm text-gray-600 mb-2">Correlation: {correlation:.3f} ({strength} {direction})</div>'

        plot_id = f"plot_{panel.id}"
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.LARGE])

        import json
        x_json = json.dumps(x_values)
        y_json = json.dumps(y_values)
        labels_json = json.dumps(labels)
        colors_json = json.dumps(colors)
        sizes_json = json.dumps(sizes)

        return f"""
        <div class="{size_class} p-6 rounded-lg shadow-sm border border-gray-200 bg-white" data-panel-id="{panel.id}" data-panel-type="scatter">
            <div class="text-lg font-semibold text-gray-800 mb-1">{panel.title}</div>
            {f'<div class="text-sm text-gray-600 mb-4">{panel.subtitle}</div>' if panel.subtitle else ''}
            {correlation_text}
            <div id="{plot_id}" style="height: 550px;"></div>
        </div>
        <script>
        (function() {{
            var trace = {{
                x: {x_json},
                y: {y_json},
                mode: 'markers',
                type: 'scatter',
                text: {labels_json},
                marker: {{
                    size: {sizes_json},
                    color: {colors_json},
                    opacity: 0.7,
                    line: {{
                        color: 'white',
                        width: 2
                    }}
                }},
                hovertemplate: '<b>%{{text}}</b><br>{x_label}: %{{x}}<br>{y_label}: %{{y}}<extra></extra>'
            }};

            // Get theme-aware colors (Plotly-compatible RGB/hex)
            const getThemeColors = () => {{
                const theme = document.documentElement.getAttribute('data-theme') || 'light';
                if (theme === 'dark') {{
                    return {{
                        bg: '#1f2937',
                        bgSecondary: '#111827',
                        text: '#f3f4f6',
                        grid: '#374151'
                    }};
                }} else {{
                    return {{
                        bg: '#ffffff',
                        bgSecondary: '#f9fafb',
                        text: '#1f2937',
                        grid: '#e5e7eb'
                    }};
                }}
            }};

            const colors = getThemeColors();

            var layout = {{
                xaxis: {{ title: '{x_label}', gridcolor: colors.grid, color: colors.text }},
                yaxis: {{ title: '{y_label}', gridcolor: colors.grid, color: colors.text }},
                plot_bgcolor: colors.bgSecondary,
                paper_bgcolor: colors.bg,
                margin: {{ l: 60, r: 30, t: 30, b: 60 }},
                hovermode: 'closest',
                font: {{ color: colors.text }}
            }};

            var config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }};

            Plotly.newPlot('{plot_id}', [trace], layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'paper_bgcolor': newColors.bg,
                    'plot_bgcolor': newColors.bgSecondary,
                    'xaxis.gridcolor': newColors.grid,
                    'yaxis.gridcolor': newColors.grid,
                    'xaxis.color': newColors.text,
                    'yaxis.color': newColors.text,
                    'font.color': newColors.text
                }});
            }});
        }})();
        </script>
        """

    def _render_line(self, panel: Panel) -> str:
        """
        Render LINE chart panel (time-series trends).

        Tufte: Emphasize data over decoration, use small multiples where appropriate.
        """
        data = panel.data
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        x_label = data.get('x_label', 'Time')
        y_label = data.get('y_label', 'Value')
        line_color = data.get('color', '#6366f1')
        show_points = data.get('show_points', True)

        # Support multiple lines
        lines_data = data.get('lines', None)  # [{'name': 'Line1', 'y': [], 'color': ''}]
        traces_data = data.get('traces', None)  # Legacy support

        plot_id = f"plot_{panel.id}"
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.LARGE])

        # Hero panels get much taller
        chart_height = '700px' if panel.size == PanelSize.HERO else '550px'

        import json

        # Build traces - support both 'lines' (new) and 'traces' (legacy)
        if lines_data:
            # New format: shares x-axis
            traces_json = json.dumps([{
                'x': x_values,
                'y': line['y'],
                'type': 'scatter',
                'mode': 'lines+markers' if show_points else 'lines',
                'name': line.get('name', f'Series {i+1}'),
                'line': {'color': line.get('color', f'hsl({i*60}, 70%, 50%)'), 'width': 3},
                'marker': {'size': 6} if show_points else {}
            } for i, line in enumerate(lines_data)])
        elif traces_data:
            # Legacy format: each trace has own x
            traces_json = json.dumps([{
                'x': trace['x'],
                'y': trace['y'],
                'type': 'scatter',
                'mode': 'lines+markers' if show_points else 'lines',
                'name': trace.get('name', f'Series {i+1}'),
                'line': {'color': trace.get('color', f'hsl({i*60}, 70%, 50%)'), 'width': 3},
                'marker': {'size': 6} if show_points else {}
            } for i, trace in enumerate(traces_data)])
        else:
            # Single line
            x_json = json.dumps(x_values)
            y_json = json.dumps(y_values)
            traces_json = json.dumps([{
                'x': x_values,
                'y': y_values,
                'type': 'scatter',
                'mode': 'lines+markers' if show_points else 'lines',
                'line': {'color': line_color, 'width': 3},
                'marker': {'size': 6} if show_points else {}
            }])

        return f"""
        <article class="panel"
                 data-panel-id="{panel.id}"
                 data-panel-type="line"
                 data-size="{panel.size.value}"
                 role="article"
                 aria-labelledby="panel-{panel.id}-title"
                 tabindex="0">
            <div class="panel-content">
                <h3 class="panel-title" id="panel-{panel.id}-title">{panel.title}</h3>
                {f'<div class="panel-subtitle">{panel.subtitle}</div>' if panel.subtitle else ''}
                <div id="{plot_id}" style="height: {chart_height};"></div>
            </div>
        </article>
        <script>
        (function() {{
            var traces = {traces_json};

            // Get theme-aware colors (Plotly-compatible RGB/hex)
            const getThemeColors = () => {{
                const theme = document.documentElement.getAttribute('data-theme') || 'light';
                if (theme === 'dark') {{
                    return {{
                        bg: '#1f2937',
                        bgSecondary: '#111827',
                        text: '#f3f4f6',
                        grid: '#374151'
                    }};
                }} else {{
                    return {{
                        bg: '#ffffff',
                        bgSecondary: '#f9fafb',
                        text: '#1f2937',
                        grid: '#e5e7eb'
                    }};
                }}
            }};

            const colors = getThemeColors();

            var layout = {{
                xaxis: {{ title: '{x_label}', gridcolor: colors.grid, color: colors.text }},
                yaxis: {{ title: '{y_label}', gridcolor: colors.grid, color: colors.text }},
                plot_bgcolor: colors.bgSecondary,
                paper_bgcolor: colors.bg,
                margin: {{ l: 60, r: 30, t: 30, b: 60 }},
                showlegend: {str(bool(traces_data)).lower()},
                hovermode: 'x unified',
                font: {{ color: colors.text }}
            }};

            var config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            }};

            Plotly.newPlot('{plot_id}', traces, layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'paper_bgcolor': newColors.bg,
                    'plot_bgcolor': newColors.bgSecondary,
                    'xaxis.gridcolor': newColors.grid,
                    'yaxis.gridcolor': newColors.grid,
                    'xaxis.color': newColors.text,
                    'yaxis.color': newColors.text,
                    'font.color': newColors.text
                }});
            }});
        }})();
        </script>
        """

    def _render_bar(self, panel: Panel) -> str:
        """
        Render BAR chart panel (categorical comparisons).

        Tufte: Use horizontal bars for long labels, minimize chart junk.
        """
        data = panel.data
        categories = data.get('categories', [])
        values = data.get('values', [])
        orientation = data.get('orientation', 'v')  # 'v' vertical, 'h' horizontal
        x_label = data.get('x_label', 'Category' if orientation == 'v' else 'Value')
        y_label = data.get('y_label', 'Value' if orientation == 'v' else 'Category')
        colors = data.get('colors', ['#6366f1'] * len(categories))

        plot_id = f"plot_{panel.id}"
        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.LARGE])

        import json
        categories_json = json.dumps(categories)
        values_json = json.dumps(values)
        colors_json = json.dumps(colors)

        # For horizontal bars, swap x and y
        if orientation == 'h':
            x_data = values_json
            y_data = categories_json
        else:
            x_data = categories_json
            y_data = values_json

        return f"""
        <div class="{size_class} p-6 rounded-lg shadow-sm border border-gray-200 bg-white" data-panel-id="{panel.id}" data-panel-type="bar">
            <div class="text-lg font-semibold text-gray-800 mb-1">{panel.title}</div>
            {f'<div class="text-sm text-gray-600 mb-4">{panel.subtitle}</div>' if panel.subtitle else ''}
            <div id="{plot_id}" style="height: 550px;"></div>
        </div>
        <script>
        (function() {{
            var trace = {{
                x: {x_data},
                y: {y_data},
                type: 'bar',
                orientation: '{orientation}',
                marker: {{
                    color: {colors_json},
                    opacity: 0.8
                }},
                text: {values_json},
                textposition: 'auto',
                hovertemplate: '<b>%{{{"x" if orientation == "v" else "y"}}}</b><br>Value: %{{{"y" if orientation == "v" else "x"}}}<extra></extra>'
            }};

            // Get theme-aware colors (Plotly-compatible RGB/hex)
            const getThemeColors = () => {{
                const theme = document.documentElement.getAttribute('data-theme') || 'light';
                if (theme === 'dark') {{
                    return {{
                        bg: '#1f2937',
                        bgSecondary: '#111827',
                        text: '#f3f4f6',
                        grid: '#374151'
                    }};
                }} else {{
                    return {{
                        bg: '#ffffff',
                        bgSecondary: '#f9fafb',
                        text: '#1f2937',
                        grid: '#e5e7eb'
                    }};
                }}
            }};

            const colors = getThemeColors();

            var layout = {{
                xaxis: {{ title: '{x_label}', gridcolor: colors.grid, color: colors.text }},
                yaxis: {{ title: '{y_label}', gridcolor: colors.grid, color: colors.text }},
                plot_bgcolor: colors.bgSecondary,
                paper_bgcolor: colors.bg,
                margin: {{ l: {"150" if orientation == "h" else "60"}, r: 30, t: 30, b: {"60" if orientation == "v" else "40"} }},
                showlegend: false,
                font: {{ color: colors.text }}
            }};

            var config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            }};

            Plotly.newPlot('{plot_id}', [trace], layout, config);

            // Redraw on theme change
            window.addEventListener('themechange', () => {{
                const newColors = getThemeColors();
                Plotly.relayout('{plot_id}', {{
                    'paper_bgcolor': newColors.bg,
                    'plot_bgcolor': newColors.bgSecondary,
                    'xaxis.gridcolor': newColors.grid,
                    'yaxis.gridcolor': newColors.grid,
                    'xaxis.color': newColors.text,
                    'yaxis.color': newColors.text,
                    'font.color': newColors.text
                }});
            }});
        }})();
        </script>
        """

    def _render_insight(self, panel: Panel) -> str:
        """
        Render INSIGHT card panel (auto-detected intelligence).

        Tufte: Communicate insights clearly with minimal decoration.
        Displays patterns, outliers, trends, correlations discovered automatically.
        """
        data = panel.data
        insight_type = data.get('type', 'pattern')  # 'pattern', 'outlier', 'trend', 'correlation'
        message = data.get('message', 'No insight available')
        confidence = data.get('confidence', 0.0)
        details = data.get('details', {})

        # Icon and color based on insight type
        type_config = {
            'pattern': {'icon': '🔍', 'color': 'blue', 'label': 'Pattern'},
            'outlier': {'icon': '⚠️', 'color': 'yellow', 'label': 'Outlier'},
            'trend': {'icon': '📈', 'color': 'green', 'label': 'Trend'},
            'correlation': {'icon': '🔗', 'color': 'purple', 'label': 'Correlation'},
            'recommendation': {'icon': '💡', 'color': 'indigo', 'label': 'Recommendation'}
        }

        config = type_config.get(insight_type, type_config['pattern'])
        icon = config['icon']
        color = config['color']
        label = config['label']

        # Confidence indicator
        confidence_pct = int(confidence * 100)
        confidence_color = 'green' if confidence > 0.8 else 'yellow' if confidence > 0.6 else 'gray'
        confidence_bar_width = confidence_pct

        # Details list
        details_html = ""
        if details:
            details_items = [f'<li class="text-sm text-gray-600">• {k}: {v}</li>' for k, v in details.items()]
            details_html = f'<ul class="mt-3 space-y-1">{"".join(details_items)}</ul>'

        size_class = PANEL_SIZE_CLASSES.get(panel.size, PANEL_SIZE_CLASSES[PanelSize.MEDIUM])

        # Color classes for border
        border_colors = {
            'blue': 'border-l-blue-500',
            'yellow': 'border-l-yellow-500',
            'green': 'border-l-green-500',
            'purple': 'border-l-purple-500',
            'indigo': 'border-l-indigo-500',
            'gray': 'border-l-gray-500'
        }
        border_class = border_colors.get(color, 'border-l-blue-500')

        return f"""
        <div class="{size_class} p-6 rounded-lg shadow-sm border border-gray-200 border-l-4 {border_class} bg-white" data-panel-id="{panel.id}" data-panel-type="insight">
            <div class="flex items-start mb-3">
                <div class="text-3xl mr-3">{icon}</div>
                <div class="flex-1">
                    <div class="flex items-center justify-between mb-1">
                        <div class="text-sm font-medium text-gray-500">{label}</div>
                        <div class="text-xs text-{confidence_color}-600 font-medium">{confidence_pct}% confident</div>
                    </div>
                    <div class="text-base font-semibold text-gray-800">{panel.title}</div>
                </div>
            </div>

            <div class="text-sm text-gray-700 leading-relaxed">{message}</div>

            {details_html}

            <!-- Confidence bar -->
            <div class="mt-4 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div class="h-full bg-{confidence_color}-500" style="width: {confidence_bar_width}%;"></div>
            </div>
        </div>
        """

    def _render_metadata(self, dashboard: Dashboard) -> str:
        """
        Render dashboard metadata footer.

        Args:
            dashboard: Dashboard object

        Returns:
            HTML for metadata footer
        """
        metadata = dashboard.metadata
        complexity = metadata.get('complexity', 'UNKNOWN')
        panel_count = metadata.get('panel_count', len(dashboard.panels))
        generated_at = metadata.get('generated_at', dashboard.created_at.isoformat())

        # Spacetime metadata (if available)
        spacetime_meta = ''
        if hasattr(dashboard.spacetime, 'metadata') and dashboard.spacetime.metadata:
            cache_stats = dashboard.spacetime.metadata.get('semantic_cache', {})
            if cache_stats:
                hits = cache_stats.get('hits', 0)
                misses = cache_stats.get('misses', 0)
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                spacetime_meta = f" | Cache: {hit_rate:.0f}% hits ({hits}/{total})"

        return f"""
        <div class="mt-8 pt-4 border-t border-gray-200 text-xs text-gray-500 text-center">
            Complexity: {complexity} | Panels: {panel_count}{spacetime_meta}
            <br>
            Generated: {generated_at[:19]}
        </div>
        """

    def _load_interactivity_js(self) -> str:
        """
        Load the dashboard interactivity JavaScript.

        Returns:
            JavaScript code as string
        """
        from pathlib import Path

        # Try to load from file
        js_path = Path(__file__).parent / 'dashboard_interactivity.js'
        if js_path.exists():
            return js_path.read_text(encoding='utf-8')

        # Fallback: basic interactivity inline
        return """
        // Basic interactivity fallback
        console.log('[HoloLoom] Using fallback interactivity');

        document.querySelectorAll('[data-panel-id]').forEach(panel => {
            panel.style.cursor = 'pointer';
            panel.addEventListener('click', () => {
                panel.classList.toggle('expanded');
            });
        });
        """

    def _assemble_html(
        self,
        title: str,
        layout_class: str,
        panels_html: str,
        metadata_html: str,
        interactivity_js: str = ""
    ) -> str:
        """
        Assemble complete HTML document with modern CSS/HTML5.

        Args:
            title: Dashboard title
            layout_class: Layout type (metric/flow/research/adaptive)
            panels_html: Rendered panels HTML
            metadata_html: Rendered metadata footer HTML
            interactivity_js: Legacy interactivity (optional)

        Returns:
            Complete HTML document with modern features
        """
        # Load modern CSS and JS
        modern_css = self._load_modern_css()
        modern_js = self._load_modern_js()

        # Map old layout_class to new data-layout attribute
        layout_map = {
            "grid grid-cols-1 gap-6": "metric",
            "grid grid-cols-1 md:grid-cols-2 gap-6": "flow",
            "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6": "research"
        }
        layout_type = layout_map.get(layout_class, "flow")

        return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="HoloLoom Performance Dashboard - {title}">
    <title>{title} - HoloLoom Dashboard</title>

    <!-- Plotly.js (interactive charts) -->
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>

    <!-- D3.js (force-directed graphs) -->
    <script src="https://d3js.org/d3.v7.min.js"></script>

    <!-- Modern CSS System (Phases 1-7) -->
    <style>
{modern_css}
    </style>
</head>
<body>
    <!-- Skip to main content (Accessibility) -->
    <a href="#main-content" class="skip-link">Skip to main content</a>

    <div class="dashboard-container">
        <!-- Dashboard Header -->
        <header class="dashboard-header">
            <h1 class="dashboard-title">{title}</h1>
            <div class="dashboard-title-accent"></div>
        </header>

        <!-- Main Dashboard Grid -->
        <main id="main-content" role="main">
            <div class="dashboard-grid" data-layout="{layout_type}">
{panels_html}
            </div>
        </main>

        <!-- Metadata Footer -->
        <footer class="dashboard-footer" role="contentinfo">
{metadata_html}
        </footer>
    </div>

    <!-- Modern JavaScript (Phases 3 & 5) -->
    <script>
{modern_js}
    </script>

    <!-- Legacy Interactivity (backward compatibility) -->
    <script>
{interactivity_js}
    </script>
</body>
</html>
"""

    def _load_modern_css(self) -> str:
        """Load modern CSS from external file."""
        from pathlib import Path

        css_path = Path(__file__).parent / 'modern_styles.css'
        if css_path.exists():
            return css_path.read_text(encoding='utf-8')

        # Fallback: minimal inline CSS
        return """
        /* Modern CSS not found - using fallback */
        :root {
            --color-bg-primary: #f9fafb;
            --color-text-primary: #1f2937;
            --space-6: 1.5rem;
            --radius-lg: 0.75rem;
        }
        body {
            font-family: system-ui, sans-serif;
            background: var(--color-bg-primary);
            color: var(--color-text-primary);
        }
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .dashboard-grid {
            display: grid;
            gap: var(--space-6);
        }
        """

    def _load_modern_js(self) -> str:
        """Load modern JavaScript from external file."""
        from pathlib import Path

        js_path = Path(__file__).parent / 'modern_interactivity.js'
        if js_path.exists():
            return js_path.read_text(encoding='utf-8')

        # Fallback: minimal inline JS
        return """
        console.log('[HoloLoom] Modern JS not found - using fallback');
        """


# ============================================================================
# Convenience Functions
# ============================================================================

def render_dashboard(dashboard: Dashboard, theme: str = 'light') -> str:
    """
    Render dashboard to HTML string.

    Args:
        dashboard: Dashboard object to render
        theme: Color theme ('light' or 'dark')

    Returns:
        Complete HTML string
    """
    renderer = HTMLRenderer(theme=theme)
    return renderer.render(dashboard)


def save_dashboard(
    dashboard: Dashboard,
    output_path: str,
    theme: str = 'light'
) -> None:
    """
    Render and save dashboard to HTML file.

    Args:
        dashboard: Dashboard object to render
        output_path: Path to save HTML file
        theme: Color theme ('light' or 'dark')
    """
    html = render_dashboard(dashboard, theme=theme)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard saved to: {output_path}")
