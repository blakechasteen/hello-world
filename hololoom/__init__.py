"""
HoloLoom - Unified Memory System

The 10/10 Layer: Everything is a memory operation.

Perfect API Surface:
    from hololoom import HoloLoom, Memory

    # That's it. Two imports.

    loom = HoloLoom()
    memory = await loom.experience("content")
    memories = await loom.recall("query")
    await loom.reflect(memories, feedback={...})

Advanced users can still import internal components:
    from hololoom.memory.awareness_graph import AwarenessGraph
    from hololoom.input.router import InputRouter
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

# ============================================================================
# CORE MODULE REDIRECT: HoloLoom.X → HoloLoom.core.X
# ============================================================================
# Core modules now live under HoloLoom/core/.  This meta-path finder
# transparently redirects old import paths so that
#   ``from hololoom.memory.graph import KG``
# resolves to ``HoloLoom.core.memory.graph``.
#
# Installed BEFORE any other imports to avoid circular dependency issues.
# ============================================================================

_CORE_MODULES = frozenset({
    "protocols", "memory", "embedding", "policy", "convergence",
    "warp", "fabric", "chrono", "resonance", "loom",
    "recursive", "reflection", "orchestrator",
})


class _CoreRedirectFinder:
    """Redirect HoloLoom.{core_module} → HoloLoom.core.{core_module}."""

    _PREFIX = "hololoom."

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        rest = fullname[len(self._PREFIX):]
        top = rest.split(".")[0]
        if top not in _CORE_MODULES:
            return None
        new_name = fullname.replace("hololoom.", "hololoom.core.", 1)
        real_spec = _importlib.util.find_spec(new_name)
        if real_spec is None:
            return None
        real_spec.loader = _CoreRedirectLoader(new_name)
        return _importlib.machinery.ModuleSpec(
            fullname,
            _CoreRedirectLoader(new_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )


class _CoreRedirectLoader:
    """Loader that imports the real core module and aliases it."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return None  # use default

    def exec_module(self, module):
        real = _importlib.import_module(self._real_name)
        module.__dict__.update(real.__dict__)
        module.__path__ = getattr(real, "__path__", [])
        module.__loader__ = self
        _sys.modules[module.__name__] = real
        _sys.modules[self._real_name] = real


if not any(isinstance(f, _CoreRedirectFinder) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _CoreRedirectFinder())

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
        from .hololoom import HoloLoom
        _lazy_imports[name] = HoloLoom
        return HoloLoom

    elif name == 'Memory':
        from .memory.protocol import Memory
        _lazy_imports[name] = Memory
        return Memory

    elif name == 'ActivationStrategy':
        from .memory.awareness_types import ActivationStrategy
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

    # Legacy/Advanced
    elif name == 'policy':
        from . import policy
        _lazy_imports[name] = policy
        return policy

    elif name == 'embedding':
        from . import embedding
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
    #   - HoloLoom.awareness → HoloLoom.memory.awareness
    #   - HoloLoom.memory_symphony → HoloLoom.memory.symphony
    #   - HoloLoom.yarn → HoloLoom.memory.yarn
    # Old import paths redirect to new locations.
    # ==========================================================================

    elif name == 'awareness':
        from .memory import awareness
        _lazy_imports[name] = awareness
        _sys.modules.setdefault(__name__ + '.awareness', awareness)
        return awareness

    elif name == 'memory_symphony':
        from .memory import symphony
        _lazy_imports[name] = symphony
        _sys.modules.setdefault(__name__ + '.memory_symphony', symphony)
        return symphony

    elif name == 'yarn':
        from .memory import yarn
        _lazy_imports[name] = yarn
        _sys.modules.setdefault(__name__ + '.yarn', yarn)
        return yarn

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
