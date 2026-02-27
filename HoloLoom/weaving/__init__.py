"""
HoloLoom weaving — MOVED to HoloLoom.orchestrator
===================================================

This shim installs a ``sys.meta_path`` finder so that **any** import of
``HoloLoom.weaving`` or ``HoloLoom.weaving.*`` is transparently
redirected to ``HoloLoom.orchestrator``.  A one-time
``DeprecationWarning`` is emitted to encourage migration.
"""

import importlib
import warnings
import sys


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.weaving.* imports."""

    _PREFIX = "HoloLoom.weaving."

    def find_module(self, fullname, path=None):
        if fullname == "HoloLoom.weaving" or fullname.startswith(
            self._PREFIX
        ):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        new_name = fullname.replace(
            "HoloLoom.weaving", "HoloLoom.orchestrator", 1
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
