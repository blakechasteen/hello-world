"""
Dashboard Constructor - Self-Constructing Dashboard System
===========================================================
The orchestrator that wires StrategySelector intelligence to HTML rendering.

This is the "Wolfram Alpha" layer - it automatically generates optimal
dashboards based on query analysis.
"""

from typing import Dict, List, Any
from datetime import datetime
import json

from .strategy_selector import StrategySelector, QueryIntent
from .dashboard import (
    Dashboard, Panel, PanelType, PanelSize, LayoutType,
    SpacetimeLike, DashboardStrategy
)


class DashboardConstructor:
    """
    Self-constructing dashboard system.

    Usage:
        constructor = DashboardConstructor()
        dashboard = constructor.construct(spacetime)
        html = dashboard.to_html()  # or use separate renderer
    """

    def __init__(self):
        self.selector = StrategySelector()
        self.panel_id_counter = 0

    def construct(self, spacetime: SpacetimeLike) -> Dashboard:
        """
        Auto-construct optimal dashboard from Spacetime.

        Steps:
        1. Analyze query + data (StrategySelector)
        2. Get optimal strategy (layout, panels, title)
        3. Materialize panels from PanelSpecs
        4. Create Dashboard object

        Args:
            spacetime: Spacetime object from WeavingOrchestrator

        Returns:
            Dashboard object ready for rendering
        """
        # Step 1+2: Get optimal strategy
        strategy = self.selector.select(spacetime)

        # Step 3: Materialize panels
        panels = []
        for spec in strategy.panels:
            panel = self._materialize_panel(spec, spacetime)
            panels.append(panel)

        # Step 4: Create dashboard
        dashboard = Dashboard(
            title=strategy.title,
            layout=strategy.layout_type,
            panels=panels,
            spacetime=spacetime,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "query": spacetime.query_text,
                "complexity": strategy.complexity_level.value,
                "intent": self.selector.analyzer.analyze(spacetime).intent.value
            }
        )

        return dashboard

    def _materialize_panel(self, spec, spacetime: SpacetimeLike) -> Panel:
        """
        Convert PanelSpec to materialized Panel with actual data.

        This is where we extract the actual values from Spacetime
        based on the data_source path in the spec.
        """
        self.panel_id_counter += 1
        panel_id = f"panel_{self.panel_id_counter}"

        # Extract data from spacetime based on spec.data_source
        data = self._extract_data(spec.data_source, spacetime)

        # Create panel
        panel = Panel(
            id=panel_id,
            type=spec.type,
            title=spec.title or spec.type.value.title(),
            data=data,
            size=spec.size,
            metadata={}
        )

        return panel

    def _extract_data(self, data_source: str, spacetime: SpacetimeLike) -> Any:
        """
        Extract data from Spacetime using data_source path.

        Examples:
            "confidence" -> spacetime.confidence
            "trace.stage_durations" -> spacetime.trace.stage_durations
            "trace.errors" -> spacetime.trace.errors
        """
        # Handle simple attribute access
        if "." not in data_source:
            return getattr(spacetime, data_source, None)

        # Handle nested attribute access (e.g., "trace.stage_durations")
        parts = data_source.split(".")
        obj = spacetime
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj


class DashboardRenderer:
    """
    Renders Dashboard objects as standalone HTML files.

    This handles the visual presentation layer.
    """

    def __init__(self):
        self.stage_colors = {
            'features': '#6366f1',      # Indigo
            'retrieval': '#10b981',     # Green
            'decision': '#f59e0b',      # Amber
            'execution': '#ef4444',     # Red
            'synthesis': '#8b5cf6'      # Purple
        }

        self.metric_colors = {
            'confidence': 'green',
            'duration': 'blue',
            'tool': 'purple',
            'threads': 'indigo'
        }

    def render(self, dashboard: Dashboard) -> str:
        """Render Dashboard to standalone HTML."""
        layout_class = self._get_layout_class(dashboard.layout)
        panels_html = self._render_panels(dashboard.panels, layout_class)

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard.title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body class="bg-gray-50 p-8">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900">{dashboard.title}</h1>
            <p class="text-sm text-gray-500 mt-2">
                Generated: {dashboard.metadata.get("generated_at", "Unknown")}
                | Complexity: {dashboard.metadata.get("complexity", "N/A")}
                | Intent: {dashboard.metadata.get("intent", "N/A")}
            </p>
        </div>

        <!-- Panels -->
        <div class="{layout_class}">
            {panels_html}
        </div>
    </div>
</body>
</html>'''
        return html

    def _get_layout_class(self, layout: LayoutType) -> str:
        """Get Tailwind CSS class for layout."""
        layouts = {
            LayoutType.METRIC: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6",
            LayoutType.FLOW: "grid grid-cols-1 lg:grid-cols-2 gap-6",
            LayoutType.RESEARCH: "flex flex-col gap-6"
        }
        return layouts.get(layout, layouts[LayoutType.FLOW])

    def _render_panels(self, panels: List[Panel], layout_class: str) -> str:
        """Render all panels."""
        html_parts = []
        for panel in panels:
            panel_html = self._render_panel(panel)
            html_parts.append(panel_html)
        return '\n'.join(html_parts)

    def _render_panel(self, panel: Panel) -> str:
        """Render single panel based on type."""
        renderers = {
            PanelType.METRIC: self._render_metric_panel,
            PanelType.TEXT: self._render_text_panel,
            PanelType.TIMELINE: self._render_timeline_panel,
            PanelType.NETWORK: self._render_network_panel,
            PanelType.DISTRIBUTION: self._render_distribution_panel,
        }

        renderer = renderers.get(panel.type, self._render_fallback_panel)
        return renderer(panel)

    def _render_metric_panel(self, panel: Panel) -> str:
        """Render metric card."""
        value = panel.data
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, int):
            display_value = str(value)
        else:
            display_value = str(value)

        # Determine color based on panel title
        title_lower = panel.title.lower()
        if 'confidence' in title_lower:
            color = 'green'
        elif 'duration' in title_lower or 'time' in title_lower:
            color = 'blue'
        elif 'tool' in title_lower:
            color = 'purple'
        else:
            color = 'indigo'

        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <p class="text-sm text-gray-500">{panel.title}</p>
            <p class="text-3xl font-bold text-{color}-600 mt-2">{display_value}</p>
        </div>'''

    def _render_text_panel(self, panel: Panel) -> str:
        """Render text content panel."""
        content = panel.data
        if isinstance(content, list):
            # Error list or similar
            items = ''.join([f'<li class="text-red-600">{item}</li>' for item in content])
            content_html = f'<ul class="list-disc list-inside space-y-1">{items}</ul>'
        else:
            # Regular text
            content_html = f'<p class="text-gray-700">{content}</p>'

        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">{panel.title}</h3>
            {content_html}
        </div>'''

    def _render_timeline_panel(self, panel: Panel) -> str:
        """Render timeline/waterfall chart."""
        stage_durations = panel.data
        if not isinstance(stage_durations, dict):
            return self._render_fallback_panel(panel)

        stages = list(stage_durations.keys())
        durations = list(stage_durations.values())

        # Create Plotly waterfall chart
        chart_id = f"chart_{panel.id}"

        # Build measure array (all 'relative' except last which is 'total')
        measures = ['relative'] * len(stages)

        # Get colors for stages
        colors = [self.stage_colors.get(stage, '#gray') for stage in stages]

        plotly_config = {
            'x': stages,
            'y': durations,
            'type': 'bar',
            'marker': {'color': colors},
            'name': 'Duration (ms)'
        }

        layout_config = {
            'title': panel.title,
            'xaxis': {'title': 'Stage'},
            'yaxis': {'title': 'Duration (ms)'},
            'showlegend': False
        }

        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200 col-span-full">
            <div id="{chart_id}"></div>
            <script>
                Plotly.newPlot('{chart_id}',
                    [{json.dumps(plotly_config)}],
                    {json.dumps(layout_config)},
                    {{responsive: true}}
                );
            </script>
        </div>'''

    def _render_network_panel(self, panel: Panel) -> str:
        """Render network graph (placeholder for now)."""
        threads = panel.data
        if isinstance(threads, list):
            thread_list = ', '.join(threads)
        else:
            thread_list = str(threads)

        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">{panel.title}</h3>
            <p class="text-gray-700">Activated threads: {thread_list}</p>
            <p class="text-sm text-gray-500 mt-2">
                (Network visualization coming soon)
            </p>
        </div>'''

    def _render_distribution_panel(self, panel: Panel) -> str:
        """Render distribution chart (placeholder)."""
        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">{panel.title}</h3>
            <p class="text-gray-700">Distribution data: {panel.data}</p>
            <p class="text-sm text-gray-500 mt-2">
                (Distribution chart coming soon)
            </p>
        </div>'''

    def _render_fallback_panel(self, panel: Panel) -> str:
        """Fallback renderer for unknown panel types."""
        return f'''
        <div class="bg-white p-6 rounded-lg shadow border border-gray-200">
            <h3 class="text-lg font-semibold text-gray-900 mb-4">{panel.title}</h3>
            <p class="text-sm text-gray-500">Type: {panel.type.value}</p>
            <pre class="text-xs text-gray-600 mt-2 overflow-auto">{json.dumps(panel.data, indent=2, default=str)}</pre>
        </div>'''