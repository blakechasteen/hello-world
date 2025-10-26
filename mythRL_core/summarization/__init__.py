"""
Summarization Module
====================

Text summarization for ExpertLoom notes.

Generates concise summaries while preserving key entities and measurements.
"""

from .summarizer import (
    TextSummarizer,
    SummarizerConfig,
    SummarizationStrategy,
    summarize_text
)

__all__ = [
    'TextSummarizer',
    'SummarizerConfig',
    'SummarizationStrategy',
    'summarize_text'
]
