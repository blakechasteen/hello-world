"""
HoloLoom Visualization
======================
Self-constructing dashboards from Spacetime fabrics.

Like Wolfram Alpha: Every query auto-generates its optimal visualization.

Ruthlessly Elegant API:
    from HoloLoom.visualization import auto

    # One function. Zero configuration. Perfect dashboard.
    dashboard = auto(spacetime)
    dashboard = auto({'month': [...], 'value': [...]})
    dashboard = auto(memory_backend)
"""

# Ruthlessly Elegant Primary API
from .auto import auto, render, save

# Core Types (for advanced usage)
from .dashboard import (
    Panel, Dashboard, PanelType, PanelSize, LayoutType, ComplexityLevel
)

# Intelligent Builder (auto uses this internally)
from .widget_builder import WidgetBuilder, DataAnalyzer, InsightGenerator

# Legacy/Advanced APIs (prefer auto() instead)
from .html_renderer import HTMLRenderer
from .orchestrator_integration import DashboardOrchestrator, weave_and_visualize

__all__ = [
    # Primary API (ruthlessly simple)
    'auto',           # THE function - visualize anything
    'render',         # Dashboard -> HTML
    'save',           # Dashboard -> file

    # Core types
    'Panel',
    'Dashboard',
    'PanelType',
    'PanelSize',
    'LayoutType',
    'ComplexityLevel',

    # Intelligent engine
    'WidgetBuilder',
    'DataAnalyzer',
    'InsightGenerator',

    # Advanced/Legacy
    'HTMLRenderer',
    'DashboardOrchestrator',
    'weave_and_visualize',
]
