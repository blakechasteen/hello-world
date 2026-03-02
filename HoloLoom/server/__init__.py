"""
HoloLoom server — MOVED to HoloLoom.apps.server

This package was relocated on 2026-02-26.
Import from ``HoloLoom.apps.server`` instead.
"""

import importlib
import warnings
import sys


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.server.* imports."""

    _PREFIX = "HoloLoom.server."

    def find_module(self, fullname, path=None):
        if fullname == "HoloLoom.server" or fullname.startswith(self._PREFIX):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        new_name = fullname.replace(
            "HoloLoom.server", "HoloLoom.apps.server", 1
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

from HoloLoom.apps.server import *  # noqa: E402,F401,F403
