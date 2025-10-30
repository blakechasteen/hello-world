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
import os

# Compatibility: if a sibling directory named `holoLoom` exists, prefer it
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alt = os.path.join(base_dir, 'holoLoom')
if os.path.isdir(alt) and os.path.abspath(alt) not in __path__:
	__path__.insert(0, os.path.abspath(alt))

# ============================================================================
# 10/10 API: Minimal, Perfect, Inevitable
# ============================================================================

from .hololoom import HoloLoom
from .memory.protocol import Memory
from .memory.awareness_types import ActivationStrategy
from .config import Config

__all__ = [
    # Core API (99% of users)
    'HoloLoom',          # The system
    'Memory',            # The data
    'ActivationStrategy', # Recall strategies
    'Config',            # Configuration

    # Legacy/Advanced (for backward compatibility)
    'policy',
    'embedding',
]

# ============================================================================
# Backward Compatibility: Keep existing exports
# ============================================================================

# Re-export common subpackages for existing code
from . import policy
from . import embedding

# Documentation compatibility
try:
	from . import documentation as Documentation
except Exception:
	try:
		from . import Documentation
	except Exception:
		Documentation = None

# Old unified_api compatibility (if it exists)
try:
	from .unified_api import create_hololoom
	__all__.append('create_hololoom')
except ImportError:
	pass
