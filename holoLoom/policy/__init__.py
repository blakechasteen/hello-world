"""Lazy exports for the policy subpackage."""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "PolicyEngine",
    "NeuralCore",
    "UnifiedPolicy",
    "TSBandit",
    "create_policy",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(f"{__name__}.unified")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
