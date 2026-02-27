"""
HoloLoom - Unified Memory System

The 10/10 Layer: Everything is a memory operation.

Perfect API Surface:
    from HoloLoom import HoloLoom, Memory

    # That's it. Two imports.

    loom = HoloLoom()
    memory = await loom.experience("content")
    memories = await loom.recall("query")
    await loom.reflect(memories, feedback={...})

Advanced users can still import internal components:
    from HoloLoom.core.memory.awareness_graph import AwarenessGraph
    from HoloLoom.input.router import InputRouter
    # Full control when needed
"""

# ============================================================================
# LAZY LOADING: Fix circular import issues
# ============================================================================
#
# This module uses __getattr__ to lazy-load exports, avoiding import-time
# circular dependency issues. All imports happen on first use, not at import time.
#
# Benefits:
# - No circular imports
# - Faster initial import
# - Only loads what's needed
# - Backward compatible
#
# Requires Python 3.7+ for module-level __getattr__
# ============================================================================

import importlib as _importlib
import sys as _sys
import warnings as _warnings

from .__version__ import __version__

__all__ = [
    # Core API (99% of users)
    'HoloLoom',          # The system
    'HoloLoomLite',      # Simplified lite version
    'SimpleLoom',        # Alias for HoloLoomLite
    'Memory',            # The data
    'ActivationStrategy', # Recall strategies
    'Config',            # Configuration

    # Legacy/Advanced (for backward compatibility)
    'policy',
    'embedding',
]

# ==========================================================================
# Backward Compatibility: HoloLoom.X → HoloLoom.core.X  (Wave 3, 2026-02-27)
# ==========================================================================
# Core modules moved to HoloLoom/core/.  Old import paths still work via this
# meta-path finder but emit a DeprecationWarning.
#
# Covers: protocols, memory, embedding, policy, convergence, orchestrator,
#         warp, fabric, chrono, resonance, loom, recursive, reflection.
# ==========================================================================

_CORE_MODULES = frozenset({
    "protocols", "memory", "embedding", "policy", "convergence",
    "orchestrator", "warp", "fabric", "chrono", "resonance",
    "loom", "recursive", "reflection",
})


class _CoreRedirector:
    """Meta-path finder that redirects HoloLoom.<core> → HoloLoom.core.<core>."""

    _PREFIX = "HoloLoom."
    _NEW_PREFIX = "HoloLoom.core."

    def find_module(self, fullname, path=None):
        if not fullname.startswith(self._PREFIX):
            return None
        # Extract the second component: HoloLoom.<X>.rest
        rest = fullname[len(self._PREFIX):]
        top = rest.split(".", 1)[0]
        if top in _CORE_MODULES:
            return self
        return None

    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        new_name = self._PREFIX.join(
            fullname.split(self._PREFIX, 1)
        )  # identity — rebuild below
        rest = fullname[len(self._PREFIX):]
        top = rest.split(".", 1)[0]
        new_name = self._NEW_PREFIX + rest
        _warnings.warn(
            f"Importing from {fullname} is deprecated. "
            f"Use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        real = _importlib.import_module(new_name)
        _sys.modules[fullname] = real
        return real


if not any(isinstance(f, _CoreRedirector) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _CoreRedirector())

# Track what's been loaded to avoid reimport
_lazy_imports = {}

def __getattr__(name):
    """
    Lazy import handler - loads modules on first access.

    This breaks circular dependencies by deferring all imports until actually needed.
    """
    # Return cached import if available
    if name in _lazy_imports:
        return _lazy_imports[name]

    # Core API
    if name == 'HoloLoom':
        from .unified_api import HoloLoom
        _lazy_imports[name] = HoloLoom
        return HoloLoom

    elif name == 'Memory':
        from .core.memory.protocol import Memory
        _lazy_imports[name] = Memory
        return Memory

    elif name == 'ActivationStrategy':
        from .core.memory.awareness_types import ActivationStrategy
        _lazy_imports[name] = ActivationStrategy
        return ActivationStrategy

    elif name == 'Config':
        from .config import Config
        _lazy_imports[name] = Config
        return Config

    # HoloLoom Lite - Simplified API
    elif name == 'HoloLoomLite':
        from .lite import HoloLoomLite
        _lazy_imports[name] = HoloLoomLite
        return HoloLoomLite

    elif name == 'SimpleLoom':
        # Alias for HoloLoomLite
        from .lite import HoloLoomLite
        _lazy_imports['SimpleLoom'] = HoloLoomLite
        _lazy_imports['HoloLoomLite'] = HoloLoomLite
        return HoloLoomLite

    # Legacy/Advanced — redirect to core subpackages
    elif name == 'policy':
        from .core import policy
        _lazy_imports[name] = policy
        return policy

    elif name == 'embedding':
        from .core import embedding
        _lazy_imports[name] = embedding
        return embedding

    # Documentation compatibility
    elif name == 'Documentation' or name == 'documentation':
        try:
            from . import documentation as Documentation
            _lazy_imports['Documentation'] = Documentation
            _lazy_imports['documentation'] = Documentation
            _sys.modules.setdefault(__name__ + '.documentation', Documentation)
            _sys.modules.setdefault(__name__ + '.Documentation', Documentation)
            try:
                _types = Documentation.types  # type: ignore[attr-defined]
                _sys.modules.setdefault(__name__ + '.Documentation.types', _types)
                _sys.modules.setdefault(__name__ + '.documentation.types', _types)
            except Exception:
                pass
            return Documentation
        except ImportError:
            return None

    # Old unified_api compatibility
    elif name == 'create_hololoom':
        try:
            from .unified_api import create_hololoom
            _lazy_imports[name] = create_hololoom
            return create_hololoom
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # ==========================================================================
    # Backward Compatibility Shims (December 2025 Consolidation)
    # ==========================================================================
    # These directories were consolidated into memory/:
    #   - HoloLoom.awareness → HoloLoom.core.memory.awareness
    #   - HoloLoom.memory_symphony → HoloLoom.core.memory.symphony
    #   - HoloLoom.yarn → HoloLoom.core.memory.yarn
    # Old import paths redirect to new locations.
    # ==========================================================================

    elif name == 'awareness':
        from .core.memory import awareness
        _lazy_imports[name] = awareness
        _sys.modules.setdefault(__name__ + '.awareness', awareness)
        return awareness

    elif name == 'memory_symphony':
        from .core.memory import symphony
        _lazy_imports[name] = symphony
        _sys.modules.setdefault(__name__ + '.memory_symphony', symphony)
        return symphony

    elif name == 'yarn':
        from .core.memory import yarn
        _lazy_imports[name] = yarn
        _sys.modules.setdefault(__name__ + '.yarn', yarn)
        return yarn

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
