"""
HoloLoom Integrations
=====================
External integrations for HoloLoom neural decision system.
"""

__all__ = []

try:
    from .promptly import PromptlyLoader, PromptlyMemoryAdapter
    __all__.extend(['PromptlyLoader', 'PromptlyMemoryAdapter'])
except ImportError:
    # Promptly not available, gracefully degrade
    pass
