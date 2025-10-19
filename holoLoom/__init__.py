"""Top-level package initialisation for ``holoLoom``.

This module keeps imports lightweight so that optional heavy dependencies
(e.g. NumPy, PyTorch, networkx) are only required when the corresponding
submodule is actually used.  Subpackages such as ``policy`` and
``embedding`` are therefore imported lazily via ``__getattr__``.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

# Compatibility: if a sibling directory named ``holoLoom`` exists alongside
# the package (common in editable installs), ensure it is discoverable.
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_alt = os.path.join(_base_dir, "holoLoom")
if os.path.isdir(_alt) and os.path.abspath(_alt) not in __path__:
    __path__.insert(0, os.path.abspath(_alt))

__all__ = ["policy", "embedding", "Documentation"]


def _lazy_import(name: str) -> ModuleType:
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name in {"policy", "embedding"}:
        try:
            return _lazy_import(name)
        except ImportError as exc:  # pragma: no cover - exercised in minimal envs
            raise AttributeError(name) from exc
    if name == "Documentation":
        try:
            module = _lazy_import("documentation")
        except ImportError:
            module = importlib.import_module(f"{__name__}.Documentation")
        globals()[name] = module
        return module
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
