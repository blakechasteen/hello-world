"""
Elle Core - AR Companion Intelligence

A calm, grounded AR companion who exists in physical space alongside you.
"""
from .engine import ElleEngine, ElleRequest, create_engine
from .core import create_llm_client
from .models import (
    SceneSnapshot,
    SceneObject,
    UserState,
    UserIntent,
    ElleAction,
)

__version__ = "0.1.0"

__all__ = [
    "ElleEngine",
    "ElleRequest",
    "create_engine",
    "create_llm_client",
    "SceneSnapshot",
    "SceneObject",
    "UserState",
    "UserIntent",
    "ElleAction",
]
