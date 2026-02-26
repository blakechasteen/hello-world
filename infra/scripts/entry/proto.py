#!/usr/bin/env python
"""
Proto CLI Entry Point
=====================
This is the main entry point for Proto from the repository root.

Proto is a general-purpose code agent that integrates deeply with
HoloLoom's infrastructure. Use it for code analysis, review, generation,
and explanation.

Installation:
    # Clone HoloLoom repository
    git clone https://github.com/your-org/mythRL.git
    cd mythRL

    # Install dependencies
    pip install -e .

Usage Examples:
    # Single query
    proto ask "explain this code"
    proto review path/to/file.py
    proto suggest --file src/module.py --type refactor

    # Interactive REPL
    proto repl

    # With context
    proto ask "what does this function do?" --context src/

    # Get help
    proto --help
    proto ask --help

Environment Variables:
    PROTO_MODEL        - LLM provider (default: ollama)
    PROTO_VERIFY       - Enable verification (default: true)
    PROTO_SANDBOX      - Enable ability sandbox (default: false)
    HOLOLOOM_CONFIG    - HoloLoom config file path

Author: HoloLoom Team
License: Apache 2.0
Repository: https://github.com/your-org/mythRL
Documentation: https://docs.hololoom.ai/proto
"""

import sys
from pathlib import Path

# Add repository root to path so imports work
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Import and run CLI
from HoloLoom.apps.departments.proto.adapters.cli.main import cli


if __name__ == "__main__":
    cli()
