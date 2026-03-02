"""
HoloLoom web_dashboard — MOVED to HoloLoom.apps.workflow_builder

This package was relocated on 2026-02-26.
Import from ``HoloLoom.apps.workflow_builder`` instead.

This shim exists so that ``from HoloLoom.web_dashboard.X import Y``
emits a helpful deprecation warning rather than a confusing ImportError.
"""

import importlib
import warnings
import sys
import types


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.web_dashboard.* imports."""

    _PREFIX = "HoloLoom.web_dashboard."

    def find_module(self, fullname, path=None):
        if fullname == "HoloLoom.web_dashboard" or fullname.startswith(self._PREFIX):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Map old name → new name
        new_name = fullname.replace(
            "HoloLoom.web_dashboard", "HoloLoom.apps.workflow_builder", 1
        )

        warnings.warn(
            f"Importing from {fullname} is deprecated. "
            f"Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import the real module
        real = importlib.import_module(new_name)
        sys.modules[fullname] = real
        return real


# Install the finder once
if not any(isinstance(f, _DeprecatedFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _DeprecatedFinder())

from HoloLoom.apps.workflow_builder import *  # noqa: E402,F401,F403
