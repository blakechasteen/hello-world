"""
Capabilities Router
====================

System-wide discovery endpoint: what agents, features, and integrations
are available on this HoloLoom instance?

Consumers hit GET /api/capabilities once to understand the full surface area.
"""

import logging
import os

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Discovery"])


def _check_import(module_path: str) -> bool:
    """Check if a module is importable without side effects."""
    try:
        __import__(module_path)
        return True
    except ImportError:
        return False


@router.get("/capabilities")
async def capabilities():
    """
    Discover available agents, features, and integrations.

    Returns a structured manifest so consumers can adapt their UX
    without probing individual endpoints.
    """
    # Agents — which chat agents are mounted?
    agents = []
    try:
        from hololoom.apps.server.promptly.config import (
            JENNY_ENABLED,
            REFINEMENT_ENABLED,
        )
        agents.append({
            "name": "Promptly",
            "prefix": "/promptly",
            "features": {
                "refinement": REFINEMENT_ENABLED,
                "jenny_visualization": JENNY_ENABLED,
                "vault_search": _check_import("hololoom.apps.server.vault_bridge"),
                "tts": _check_import("hololoom.apps.server.tts_client"),
            },
        })
    except ImportError:
        pass

    try:
        from hololoom.apps.server import elle_chat  # noqa: F401
        agents.append({
            "name": "Elle",
            "prefix": "/elle",
            "features": {},
        })
    except ImportError:
        pass

    # Models — registered model specs
    models = []
    try:
        from hololoom.apps.server.model_router import get_router
        model_rtr = get_router()
        models = model_rtr.list_models()
    except Exception:
        pass

    # Weaving stages
    stages = []
    try:
        from hololoom.core.orchestrator.stages import WEAVING_STAGES
        stages = [
            {"id": s["id"], "name": s["name"], "description": s["description"]}
            for s in WEAVING_STAGES
        ]
    except ImportError:
        pass

    # Optional integrations
    integrations = {
        "neo4j": _check_import("neo4j"),
        "qdrant": _check_import("qdrant_client"),
        "redis": _check_import("redis"),
        "spacy": _check_import("spacy"),
        "sentence_transformers": _check_import("sentence_transformers"),
    }

    # Memory backend
    memory_backend = os.environ.get("HOLOLOOM_MEMORY", "hybrid").lower()

    return {
        "service": "HoloLoom Agentic API",
        "version": "1.0.0-alpha.1",
        "agents": agents,
        "models": models,
        "stages": stages,
        "memory_backend": memory_backend,
        "integrations": integrations,
    }
