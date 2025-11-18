"""
HoloLoom Production API
========================

FastAPI-based production API for HoloLoom's advanced memory systems.

Provides REST endpoints for:
- Memory consolidation (Week 5)
- Semantic transition (Week 6)
- Temporal evolution (Week 7)
- Curiosity engine
- Multi-hop graph reasoning

Author: HoloLoom Team
Date: 2025-11-18 (Week 8B: Docker Deployment)
"""

from HoloLoom.api.server import app
from HoloLoom.api.models import *

__all__ = ["app"]
