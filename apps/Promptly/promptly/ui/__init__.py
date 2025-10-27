"""
Promptly UI Module
==================
Interactive terminal and web interfaces for Promptly.
"""

# Graceful imports
__all__ = []

try:
    from .terminal_app import PromptlyApp
    __all__.append('PromptlyApp')
except ImportError:
    pass

try:
    from .terminal_app_wired import HoloLoomTerminalApp
    __all__.append('HoloLoomTerminalApp')
except ImportError:
    pass
