"""
Promptly - Personal Prompt Management Suite

A power-user tool for managing prompts, chains, refinement passes, and microprompts
across all interfaces with slash-command invocation.

Features:
- Git-like versioning (branches, commits, history)
- Dual storage (global ~/.promptly/ + project-local .promptly/)
- LLM Judge for prompt evaluation
- MCP Server for Claude Desktop integration
- VS Code extension for Command Palette integration
- HoloLoom memory integration (optional)

Quick Start:
    from promptly import Promptly

    # Initialize (auto-detects project root)
    p = Promptly()

    # Add a prompt
    p.add("greeting", "Hello, {{name}}! Welcome to {{project}}.")

    # Get a prompt
    prompt = p.get("greeting")

    # Execute with variables
    result = prompt.render(name="World", project="Promptly")

Storage Architecture:
    ~/.promptly/           # Global (user home)
    ├── prompts.db         # Global prompts database
    ├── config.yaml        # User preferences
    └── prompts/           # Global prompt YAML files

    .promptly/             # Project-local (inherits from global)
    ├── prompts.db         # Project-specific prompts
    └── prompts/           # Project prompt overrides

Author: Blake (via Claude Code)
License: Open Core (OSS with potential paid features)
"""

__version__ = "1.0.0"
__author__ = "Blake"

from promptly.core.promptly import Promptly
from promptly.core.database import PromptlyDB

__all__ = [
    "Promptly",
    "PromptlyDB",
    "__version__",
]
