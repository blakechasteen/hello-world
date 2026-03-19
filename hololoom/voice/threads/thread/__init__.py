"""
Thread Management Components

Thread branching, merging, and summarization for Voice-First UX.
"""

from .thread_branching import BranchContext, ThreadBranch, ThreadBrancher
from .thread_merging import MergeResult, MergeStrategy, ThreadMerger

__all__ = [
    'ThreadBrancher',
    'ThreadBranch',
    'BranchContext',
    'ThreadMerger',
    'MergeStrategy',
    'MergeResult'
]

__version__ = "0.2.0"
