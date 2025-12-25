"""
Promptly Core - Database and API layer

This module provides:
- PromptlyDB: SQLite database management with dual storage support
- Promptly: Main API for prompt management with git-like versioning
"""

from promptly.core.database import PromptlyDB
from promptly.core.promptly import Promptly

__all__ = ["PromptlyDB", "Promptly"]
