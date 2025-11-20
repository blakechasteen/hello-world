"""
HoloLoom HTTP Server
====================
FastAPI server exposing agentic intelligence to external clients (VS Code extension, web apps, etc).
"""

from .agentic_api import app

__all__ = ["app"]
