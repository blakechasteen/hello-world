"""
Lowercase documentation package aliasing the canonical `HoloLoom.Documentation`.
Provides backward compatibility for imports like `HoloLoom.documentation`.
"""
from importlib import import_module

# Import the canonical package and expose its attributes
_doc_pkg = import_module('HoloLoom.Documentation')
globals().update({k: getattr(_doc_pkg, k) for k in dir(_doc_pkg) if not k.startswith('_')})
