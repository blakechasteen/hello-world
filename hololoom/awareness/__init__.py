"""
HoloLoom awareness — MOVED to HoloLoom.memory.awareness
========================================================

This shim installs a ``sys.meta_path`` finder so that **any** import of
``HoloLoom.awareness`` or ``HoloLoom.awareness.*`` is transparently
redirected to ``HoloLoom.memory.awareness``.

The awareness modules were consolidated into ``hololoom/memory/awareness/``
in December 2025.  This redirect ensures backward compatibility for the
40+ call-sites that still use the old path.

Migration (one-line sed)::

    find . -name '*.py' -exec \\
        sed -i 's/from hololoom\\.awareness/from hololoom.memory.awareness/g' {} +

After all call-sites are updated this shim can be removed.

Relocated: 2025-12-XX (original move)
Shim added: 2026-02-27
"""

import importlib
import sys
import warnings


class _DeprecatedFinder:
    """Meta-path finder that redirects HoloLoom.awareness → HoloLoom.memory.awareness."""

    _OLD = "hololoom.awareness"
    _NEW = "hololoom.memory.awareness"

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
