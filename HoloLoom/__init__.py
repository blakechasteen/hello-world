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
try:
	# filesystem has `documentation` (lowercase); expose it as `Documentation`
	from . import documentation as Documentation
except Exception:
	# Fall back to a direct import if the package is named differently
	try:
		from . import Documentation
	except Exception:
		Documentation = None

# Import unified API as main export
try:
	from .unified_api import HoloLoom, create_hololoom
	__all__ = ["HoloLoom", "create_hololoom", "policy", "embedding", "Documentation"]
except ImportError as e:
	# Fallback if unified_api not available
	HoloLoom = None
	create_hololoom = None
	__all__ = ["policy", "embedding", "Documentation"]
