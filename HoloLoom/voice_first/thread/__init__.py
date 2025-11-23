"""
Thread Management Components

Thread branching, merging, and summarization for Voice-First UX.
"""

from .thread_branching import ThreadBrancher, ThreadBranch, BranchContext

__all__ = [
    'ThreadBrancher',
    'ThreadBranch',
    'BranchContext'
]

__version__ = "0.1.0"
