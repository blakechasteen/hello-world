"""
Promptly TUI - Terminal User Interface
Rich terminal UI for managing prompts
"""

try:
    from .app import PromptlyTUI, main as tui_main
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    PromptlyTUI = None
    tui_main = None

__all__ = [
    'PromptlyTUI',
    'tui_main',
    'TEXTUAL_AVAILABLE',
]
