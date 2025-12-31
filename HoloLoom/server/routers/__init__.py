"""
HoloLoom Server Routers
========================

Modular FastAPI routers extracted from agentic_api.py for better maintainability.
Part of W2 (Monolithic Files) SWOT remediation - December 2025.

Router Organization:
- health.py: Health checks, stats, safety stats, audit trail
- memory.py: Remember, recall, todos, memory add
- detection.py: AI slop, logic errors, hallucination detection
- ingestion.py: Workspace and file ingestion
- codebase.py: Codebase stats, code verification, search
- graph.py: Knowledge graph visualization
- monitor.py: Agent monitoring, sessions, metrics, WebSocket
"""

from .health import router as health_router
from .memory import router as memory_router
from .detection import router as detection_router
from .monitor import router as monitor_router

__all__ = [
    "health_router",
    "memory_router",
    "detection_router",
    "monitor_router",
]
