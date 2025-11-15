"""
Promptly Integrations
=====================
External integrations for Promptly framework.
"""

__all__ = []

try:
    from .hololoom import HoloLoomEvaluator, HoloLoomSpinner
    __all__.extend(['HoloLoomEvaluator', 'HoloLoomSpinner'])
except ImportError:
    # HoloLoom not available, gracefully degrade
    pass
