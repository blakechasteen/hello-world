"""HoloLoom package initializer.

Expose convenient imports for top-level modules used across the codebase.
This file also includes a small compatibility shim: some parts of the
repository use a lowercase `holoLoom` directory on disk. To make
`import HoloLoom.*` work regardless of the on-disk capitalization we
prepend the sibling `holoLoom` directory to the package search path if
it exists.
"""
import os

# Compatibility: if a sibling directory named `holoLoom` exists, prefer it
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alt = os.path.join(base_dir, 'holoLoom')
if os.path.isdir(alt) and os.path.abspath(alt) not in __path__:
	__path__.insert(0, os.path.abspath(alt))

# Re-export common subpackages for simpler imports
from . import policy
from . import embedding
from . import Documentation

__all__ = ["policy", "embedding", "Documentation"]
