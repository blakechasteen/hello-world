"""
HoloLoom Mobile Integration
============================
Mobile API support for iOS and Android clients.

Components:
- REST API endpoints (api.py)
- OpenAPI specification (openapi.yaml)
- WebSocket protocol documentation
- Example client implementations (Swift, Kotlin)
- Mock server for development

Usage:
    from HoloLoom.mobile.api import add_mobile_routes
    from fastapi import FastAPI

    app = FastAPI()
    add_mobile_routes(app)
"""

__version__ = "1.0.0"

# Export main components
try:
    from .api import add_mobile_routes, MobileConfig
    __all__ = ["add_mobile_routes", "MobileConfig"]
except ImportError:
    # FastAPI not available
    __all__ = []
