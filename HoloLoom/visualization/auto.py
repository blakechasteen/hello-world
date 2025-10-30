#!/usr/bin/env python3
"""
Auto-Visualization: The Ruthlessly Elegant Path
================================================
One function. Zero configuration. Perfect dashboard.

Philosophy: "If you need to configure it, we failed."

Usage:
    from HoloLoom.visualization import auto

    # From Spacetime (HoloLoom query result)
    dashboard = auto(spacetime)

    # From raw data
    dashboard = auto({'month': [...], 'value': [...]})

    # From memory graph
    dashboard = auto(memory_backend)

That's it. Everything else is automatic.

Author: Claude Code
Date: October 29, 2025
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# Spacetime Data Extraction
# ============================================================================

class SpacetimeExtractor:
    """
    Extracts visualization data from Spacetime objects.

    Ruthlessly simple: finds any structured data in Spacetime and converts
    to widget-builder-ready format.
    """

    @staticmethod
    def extract(spacetime: Any) -> Optional[Dict[str, List[Any]]]:
        """
        Extract data from Spacetime for visualization.

        Looks for data in:
        1. spacetime.metadata['visualization_data']
        2. spacetime.metadata['analysis_data']
        3. spacetime.trace (execution timeline)
        4. spacetime itself (if dict-like)

        Returns:
            Dict of column_name -> values, or None if no data found
        """
        # Try metadata first
        if hasattr(spacetime, 'metadata') and spacetime.metadata:
            # Explicit viz data
            if 'visualization_data' in spacetime.metadata:
                return spacetime.metadata['visualization_data']

            # Analysis results
            if 'analysis_data' in spacetime.metadata:
                return spacetime.metadata['analysis_data']

            # Query cache stats (for performance dashboards)
            if 'query_cache' in spacetime.metadata:
                return SpacetimeExtractor._extract_cache_stats(
                    spacetime.metadata['query_cache']
                )

        # Try trace (execution timeline)
        if hasattr(spacetime, 'trace') and spacetime.trace:
            timeline_data = SpacetimeExtractor._extract_timeline(spacetime.trace)
            if timeline_data:
                return timeline_data

        # If spacetime is dict-like, use it directly
        if isinstance(spacetime, dict):
            return spacetime

        return None

    @staticmethod
    def _extract_cache_stats(cache_data: Dict) -> Dict[str, List]:
        """Extract cache performance data."""
        return {
            'metric': ['Hits', 'Misses', 'Total', 'Hit Rate'],
            'value': [
                cache_data.get('hits', 0),
                cache_data.get('misses', 0),
                cache_data.get('total', 0),
                cache_data.get('hit_rate', 0.0) * 100
            ]
        }

    @staticmethod
    def _extract_timeline(trace: Any) -> Optional[Dict[str, List]]:
        """Extract execution timeline from trace."""
        if not hasattr(trace, 'stages'):
            return None

        # Extract stage timings
        stages = getattr(trace, 'stages', [])
        if not stages:
            return None

        return {
            'stage': [s.get('name', f'Stage {i}') for i, s in enumerate(stages)],
            'duration_ms': [s.get('duration_ms', 0) for s in stages]
        }


# ============================================================================
# Memory Graph Extraction
# ============================================================================

class MemoryExtractor:
    """
    Extracts network visualization data from memory graphs.

    Converts KG/NetworkX graphs → network panel data automatically.
    """

    @staticmethod
    def extract(memory_backend: Any) -> Optional[Dict[str, Any]]:
        """
        Extract network data from memory backend.

        Returns:
            Dict with 'nodes' and 'edges' for network visualization
        """
        # Try NetworkX graph (holoLoom.memory.graph)
        if hasattr(memory_backend, 'graph'):
            import networkx as nx
            graph = memory_backend.graph

            if isinstance(graph, nx.Graph):
                return MemoryExtractor._extract_networkx(graph)

        # Try Neo4j backend
        if hasattr(memory_backend, 'get_all_nodes'):
            return MemoryExtractor._extract_neo4j(memory_backend)

        return None

    @staticmethod
    def _extract_networkx(graph) -> Dict[str, Any]:
        """Extract from NetworkX graph."""
        nodes = []
        edges = []

        # Get nodes with metadata
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            nodes.append({
                'id': str(node_id),
                'label': node_data.get('label', str(node_id)),
                'type': node_data.get('type', 'entity'),
                'size': node_data.get('importance', 12)
            })

        # Get edges
        for source, target, edge_data in graph.edges(data=True):
            edges.append({
                'source': str(source),
                'target': str(target),
                'label': edge_data.get('relation', 'related')
            })

        return {'network': {'nodes': nodes, 'edges': edges}}

    @staticmethod
    def _extract_neo4j(backend) -> Dict[str, Any]:
        """Extract from Neo4j backend."""
        # Placeholder - would query Neo4j
        nodes = backend.get_all_nodes()[:50]  # Limit for viz
        edges = backend.get_all_edges()[:100]

        return {'network': {'nodes': nodes, 'edges': edges}}


# ============================================================================
# The Auto Function (Ruthlessly Elegant Entry Point)
# ============================================================================

def auto(
    source: Union[Any, Dict, str, Path],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    open_browser: bool = False
) -> 'Dashboard':
    """
    Automatically generate optimal dashboard from any source.

    The ruthlessly elegant API: one function, zero configuration.

    Args:
        source: Anything you want to visualize:
            - Spacetime object (from HoloLoom query)
            - Dict of data (column_name -> values)
            - Memory backend (KG, Neo4j, NetworkX)
            - File path (CSV, JSON, etc.) - future
        title: Dashboard title (auto-detected if None)
        save_path: Path to save HTML (optional)
        open_browser: Open in browser after generation

    Returns:
        Complete Dashboard object

    Examples:
        >>> from HoloLoom.visualization import auto

        # From query result
        >>> spacetime = await orchestrator.weave(query)
        >>> dashboard = auto(spacetime)

        # From data
        >>> data = {'month': [...], 'sales': [...]}
        >>> dashboard = auto(data, title="Sales Report")

        # From memory
        >>> dashboard = auto(memory_backend, title="Knowledge Graph")

        # Save and open
        >>> dashboard = auto(data, save_path='report.html', open_browser=True)
    """
    from .widget_builder import WidgetBuilder
    from .dashboard import Dashboard
    from .html_renderer import HTMLRenderer
    import webbrowser

    # Step 1: Extract data from source
    data = None
    detected_type = "unknown"

    # Try Spacetime
    if hasattr(source, 'query_text') or hasattr(source, 'trace'):
        data = SpacetimeExtractor.extract(source)
        detected_type = "spacetime"
        if title is None and hasattr(source, 'query_text'):
            title = f"Analysis: {source.query_text[:50]}"

    # Try memory backend
    elif hasattr(source, 'graph') or hasattr(source, 'get_all_nodes'):
        network_data = MemoryExtractor.extract(source)
        if network_data:
            # Build network visualization directly
            return _build_network_dashboard(network_data, title)

    # Try dict
    elif isinstance(source, dict):
        data = source
        detected_type = "dict"

    # Try file path (future)
    elif isinstance(source, (str, Path)):
        raise NotImplementedError("File loading not yet implemented")

    if data is None:
        raise ValueError(
            f"Could not extract data from source (type: {type(source)}). "
            "Source must be Spacetime, dict, or memory backend."
        )

    # Step 2: Build dashboard automatically
    builder = WidgetBuilder()

    if title is None:
        title = f"Auto-Generated {detected_type.title()} Dashboard"

    # Pass spacetime if available for metadata
    spacetime_arg = source if detected_type == "spacetime" else None

    dashboard = builder.build_from_data(
        data=data,
        title=title,
        spacetime=spacetime_arg
    )

    # Step 3: Save if requested
    if save_path:
        renderer = HTMLRenderer()
        html = renderer.render(dashboard)

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding='utf-8')

        print(f"[auto] Dashboard saved: {path}")

        # Open in browser
        if open_browser:
            webbrowser.open(f'file://{path.absolute()}')

    return dashboard


def _build_network_dashboard(network_data: Dict, title: Optional[str]) -> 'Dashboard':
    """Build dashboard with network visualization."""
    from .dashboard import Panel, Dashboard, PanelType, PanelSize, LayoutType
    from dataclasses import dataclass, field
    from typing import Dict, Any
    from datetime import datetime

    # Mock spacetime
    @dataclass
    class MockSpacetime:
        query_text: str = "Knowledge Graph Visualization"
        response: str = "Network visualization"
        tool_used: str = "memory_graph"
        confidence: float = 1.0
        trace: Any = None
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self):
            return {'query': self.query_text}

    panels = []

    # Network panel
    if 'network' in network_data:
        panels.append(Panel(
            id="network_main",
            type=PanelType.NETWORK,
            title="Knowledge Graph Network",
            subtitle=f"{len(network_data['network']['nodes'])} nodes, {len(network_data['network']['edges'])} edges",
            data=network_data['network'],
            size=PanelSize.FULL_WIDTH
        ))

    # Metrics
    if 'network' in network_data:
        nodes = network_data['network']['nodes']
        edges = network_data['network']['edges']

        panels.append(Panel(
            id="metric_nodes",
            type=PanelType.METRIC,
            title="Total Nodes",
            data={
                'value': len(nodes),
                'formatted': str(len(nodes)),
                'label': 'Entities',
                'color': 'blue'
            },
            size=PanelSize.SMALL
        ))

        panels.append(Panel(
            id="metric_edges",
            type=PanelType.METRIC,
            title="Total Edges",
            data={
                'value': len(edges),
                'formatted': str(len(edges)),
                'label': 'Relationships',
                'color': 'green'
            },
            size=PanelSize.SMALL
        ))

    dashboard = Dashboard(
        title=title or "Knowledge Graph Visualization",
        layout=LayoutType.FLOW,
        panels=panels,
        spacetime=MockSpacetime(),
        metadata={
            'complexity': 'FULL',
            'panel_count': len(panels),
            'generated_at': datetime.now().isoformat(),
            'auto_generated': True
        }
    )

    return dashboard


# ============================================================================
# Convenience: Save & Render
# ============================================================================

def render(dashboard: 'Dashboard', theme: str = 'light') -> str:
    """
    Render dashboard to HTML.

    Args:
        dashboard: Dashboard object
        theme: 'light' or 'dark'

    Returns:
        HTML string
    """
    from .html_renderer import HTMLRenderer
    return HTMLRenderer(theme=theme).render(dashboard)


def save(
    dashboard: 'Dashboard',
    path: str,
    theme: str = 'light',
    open_browser: bool = False
):
    """
    Save dashboard to HTML file.

    Args:
        dashboard: Dashboard object
        path: Output file path
        theme: 'light' or 'dark'
        open_browser: Open in browser after saving
    """
    import webbrowser
    from pathlib import Path

    html = render(dashboard, theme=theme)

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(html, encoding='utf-8')

    print(f"[save] Dashboard saved: {file_path} ({len(html)/1024:.1f} KB)")

    if open_browser:
        webbrowser.open(f'file://{file_path.absolute()}')
