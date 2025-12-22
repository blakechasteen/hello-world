"""
Promptly Interactive Demo Suite

5 interactive demos showcasing all Promptly capabilities:
1. Basic Prompt Management - CRUD operations, variables, history
2. Git-like Versioning - Branches, checkouts, comparisons
3. Chains & Skills - Multi-step workflows, reusable templates
4. LLM Judge Evaluation - 12 criteria, 6 judging methods
5. HoloLoom Integration - Thompson Sampling, learning insights

Usage:
    # Run interactive menu
    python -m promptly.demos

    # Run specific demo
    python -m promptly.demos.interactive_demo --demo basic

    # CLI integration
    promptly demo [basic|versioning|chains|judge|hololoom]
"""

from .interactive_demo import (
    DemoRunner,
    run_interactive_demo,
    run_demo,
    setup_sample_data,
)

__all__ = [
    'DemoRunner',
    'run_interactive_demo',
    'run_demo',
    'setup_sample_data',
]
