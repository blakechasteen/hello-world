"""
HoloLoom Visualization
======================
Self-constructing dashboards from Spacetime fabrics.

Like Wolfram Alpha: Every query auto-generates its optimal visualization.
"""

from .dashboard import Panel, Dashboard, PanelSpec, DashboardStrategy
from .html_renderer import HTMLRenderer

__all__ = [
    'Panel',
    'Dashboard',
    'PanelSpec',
    'DashboardStrategy',
    'HTMLRenderer'
]
