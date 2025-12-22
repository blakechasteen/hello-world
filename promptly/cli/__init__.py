"""
Promptly CLI - Command Line Interface

Usage:
    promptly add <name> <content>       # Add/update prompt
    promptly get <name> [version]       # Retrieve prompt
    promptly list                       # List all prompts
    promptly branch <name>              # Create branch
    promptly checkout <branch>          # Switch branches
    promptly log <name>                 # View history
    promptly eval <name>                # Run LLM Judge
    promptly chain create <name>        # Create chain
    promptly chain run <name>           # Execute chain
    promptly skill add <name>           # Add skill
    promptly skill run <name>           # Execute skill
"""

from promptly.cli.main import cli, main

__all__ = ["cli", "main"]
