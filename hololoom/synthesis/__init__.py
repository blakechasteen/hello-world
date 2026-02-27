"""
HoloLoom synthesis -- MOVED to HoloLoom.fabric
================================================

All synthesis modules now live under ``HoloLoom.fabric``:

    HoloLoom.synthesis.enriched_memory    ->  HoloLoom.fabric.enriched_memory
    HoloLoom.synthesis.pattern_extractor  ->  HoloLoom.fabric.pattern_extractor
    HoloLoom.synthesis.data_synthesizer   ->  HoloLoom.fabric.data_synthesizer
    HoloLoom.synthesis.synthesis_bridge   ->  HoloLoom.fabric.synthesis_bridge

Relocated: 2026-02-27
"""

import importlib
import sys
import warnings


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.synthesis -> HoloLoom.fabric."""

    _OLD = "hololoom.synthesis"
    _NEW = "hololoom.fabric"

    def find_module(self, fullname, path=None):
        if fullname == self._OLD or fullname.startswith(self._OLD + "."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
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
