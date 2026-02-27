"""Proto CLI adapter.

Exports:
    - cli: Main entry point for CLI
    - ProtoCLI: CLI adapter class
    - start_repl: Start interactive REPL

Implementation: 2025-12-02
Status: Production ready
"""

from .main import cli, ProtoCLI, start_repl

__all__ = [
    "cli",
    "ProtoCLI",
    "start_repl",
]
