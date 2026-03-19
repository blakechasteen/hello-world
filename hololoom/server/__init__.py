"""
HoloLoom server — MOVED to HoloLoom.apps.server

This package was relocated on 2026-02-26.
Import from ``HoloLoom.apps.server`` instead.
"""

import importlib
import sys
import warnings


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.server.* imports."""

    _PREFIX = "hololoom.server."

    def find_module(self, fullname, path=None):
        if fullname == "hololoom.server" or fullname.startswith(self._PREFIX):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        new_name = fullname.replace(
            "hololoom.server", "hololoom.apps.server", 1
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

from hololoom.apps.server import *  # noqa: E402,F401,F403
