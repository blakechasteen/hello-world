#!/usr/bin/env python3
"""
HoloLoom SaaS API Routes

Exports all route routers for mounting in FastAPI application.
"""

from .customers import router as customers_router
from .api_keys import router as api_keys_router
from .health import router as health_router

# Future imports:
# from .billing import router as billing_router
# from .dashboard import router as dashboard_router
# from .webhooks import router as webhooks_router

__all__ = [
    "customers_router",
    "api_keys_router",
    "health_router",
    # "billing_router",
    # "dashboard_router",
    # "webhooks_router",
]
