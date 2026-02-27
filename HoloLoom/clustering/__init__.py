"""
HoloLoom clustering — MOVED to HoloLoom.semantic_calculus
=========================================================

This shim installs a ``sys.meta_path`` finder so that **any** import of
``HoloLoom.clustering`` or ``HoloLoom.clustering.*`` is transparently
redirected to ``HoloLoom.semantic_calculus``.

Files were renamed with a ``clustering_`` prefix during the move:

    HoloLoom.clustering.core      →  HoloLoom.semantic_calculus.clustering_core
    HoloLoom.clustering.labeler   →  HoloLoom.semantic_calculus.clustering_labeler
    HoloLoom.clustering.thompson  →  HoloLoom.semantic_calculus.clustering_thompson

Relocated: 2026-02-27
"""

import importlib
import sys
import warnings

_RENAME_MAP = {
    "HoloLoom.clustering.core": "HoloLoom.semantic_calculus.clustering_core",
    "HoloLoom.clustering.labeler": "HoloLoom.semantic_calculus.clustering_labeler",
    "HoloLoom.clustering.thompson": "HoloLoom.semantic_calculus.clustering_thompson",
}


class _DeprecatedFinder:
    _OLD = "HoloLoom.clustering"
    _NEW = "HoloLoom.semantic_calculus"

    def find_module(self, fullname, path=None):
        if fullname == self._OLD or fullname.startswith(self._OLD + "."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        new_name = _RENAME_MAP.get(fullname)
        if new_name is None:
            new_name = fullname.replace(self._OLD, self._NEW, 1)

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
