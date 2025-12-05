"""
COZ (Cozbi's Organix Zentrum) Satellite App

Production management system for beekeeping and organic products business.

Quick Start:
    from apps.coz import create_coz_satellite
    from apps.satellite_protocol import OrbitManager

    coz = create_coz_satellite()
    orbit = OrbitManager(loom_core)
    await orbit.dock_app(coz)

    # COZ is now docked and tools are registered
    # Can invoke via Rift or query via WarpSpace

Features:
- 10 COZ data parsers (time, cost, orders, production, SOPs, etc.)
- 8 intelligence modules (profit, efficiency, waste, customers, etc.)
- Complete integration with Zero-G Loom (WarpSpace, YarnGraph, Rift)
- Semantic search for orders, inventory, SOPs
- Knowledge graph: customer → order → product → SOP

Created: 2025-11-22
"""

from .satellite import COZSatellite, create_coz_satellite

__all__ = [
    'COZSatellite',
    'create_coz_satellite'
]

__version__ = "1.0.0"
