"""
HoloLoom ts_core — MOVED to HoloLoom.bandits
=============================================

This shim installs a ``sys.meta_path`` finder so that **any** import of
``HoloLoom.ts_core`` or ``HoloLoom.ts_core.*`` is transparently
redirected to ``HoloLoom.bandits``.  A one-time
``DeprecationWarning`` is emitted to encourage migration.
"""

import importlib
import warnings
import sys


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.ts_core.* imports."""

    _PREFIX = "HoloLoom.ts_core."

    def find_module(self, fullname, path=None):
        if fullname == "HoloLoom.ts_core" or fullname.startswith(
            self._PREFIX
        ):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        new_name = fullname.replace(
            "HoloLoom.ts_core", "HoloLoom.bandits", 1
        )
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
