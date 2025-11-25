"""
Elle Tasks Package

Auto-registers all available tasks on import.

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

# Import tasks to trigger @register_task decorators
from .analyze_task import AnalyzeTask, AnalyzeSentimentTask
from .search_task import SearchTask, SearchKnowledgeTask

# Export task classes for direct use
__all__ = [
    "AnalyzeTask",
    "AnalyzeSentimentTask",
    "SearchTask",
    "SearchKnowledgeTask",
]
