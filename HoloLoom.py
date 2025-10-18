"""Compatibility shim: make the on-disk `holoLoom` package available as
`HoloLoom` (capitalization-insensitive imports may be used across files).

This module is intentionally small: it imports the real package and
registers it under the alternate name in sys.modules so code that does
`import HoloLoom.something` will work.
"""
import importlib
import sys

try:
    holo = importlib.import_module('holoLoom')
except Exception:
    # If that fails, re-raise with a clearer message
    raise

# Register the loaded module under the alternate name
sys.modules.setdefault('HoloLoom', holo)

# Expose the package's attributes at module level
for attr in ('__path__', '__file__', '__name__'):
    if hasattr(holo, attr):
        try:
            setattr(sys.modules['HoloLoom'], attr, getattr(holo, attr))
        except Exception:
            pass

# Re-export common names for convenience
from holoLoom import *  # noqa: F401,F403
