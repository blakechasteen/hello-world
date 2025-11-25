"""
Thread Management Components

Thread branching, merging, and summarization for Voice-First UX.
"""

from .thread_branching import ThreadBrancher, ThreadBranch, BranchContext
from .thread_merging import ThreadMerger, MergeStrategy, MergeResult

__all__ = [
    'ThreadBrancher',
    'ThreadBranch',
    'BranchContext',
    'ThreadMerger',
    'MergeStrategy',
    'MergeResult'
]

__version__ = "0.2.0"
