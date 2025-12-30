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
    from HoloLoom.memory.awareness_graph import AwarenessGraph
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

import sys as _sys
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
        from .unified_api import HoloLoom
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
