"""
HoloLoom nested — MOVED to HoloLoom.memory
===========================================

This shim redirects imports of ``HoloLoom.nested.*`` to their new
locations inside ``HoloLoom.memory``.  Files were renamed with a
``nested_`` prefix during the move:

    HoloLoom.nested.ultra_fast  →  HoloLoom.memory.nested_ultra_fast
    HoloLoom.nested.hierarchy   →  HoloLoom.memory.nested_hierarchy
"""

import importlib
import warnings
import sys


_RENAME_MAP = {
    "HoloLoom.nested.ultra_fast": "HoloLoom.memory.nested_ultra_fast",
    "HoloLoom.nested.hierarchy": "HoloLoom.memory.nested_hierarchy",
}


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.nested.* imports."""

    _PREFIX = "HoloLoom.nested."

    def find_module(self, fullname, path=None):
        if fullname == "HoloLoom.nested" or fullname.startswith(
            self._PREFIX
        ):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        new_name = _RENAME_MAP.get(fullname)
        if new_name is None:
            # For the package itself, re-export from memory
            new_name = "HoloLoom.memory"
        warnings.warn(
            f"Importing from {fullname} is deprecated. "
            f"Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        real = importlib.import_module(new_name)
        sys.modules[fullname] = real
        return real


if not any(isinstance(f, _DeprecatedFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeprecatedFinder())
