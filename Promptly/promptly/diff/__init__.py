#!/usr/bin/env python3
"""
Promptly Diff Module - Advanced diffing capabilities
"""

from .engine import (
    DiffEngine,
    DiffResult,
    DiffChunk,
    DiffStats,
    DiffType,
    DiffLevel,
    MyersDiff,
    create_diff_engine,
    quick_diff,
    diff_stats_only
)

from .compare import (
    ComparisonEngine,
    VersionComparison,
    BranchComparison,
    EvaluationComparison,
    create_comparison_engine
)

from .visual import (
    TerminalDiff,
    HTMLDiff,
    Colors,
    render_terminal,
    render_html
)

__all__ = [
    # Engine
    'DiffEngine',
    'DiffResult',
    'DiffChunk',
    'DiffStats',
    'DiffType',
    'DiffLevel',
    'MyersDiff',
    'create_diff_engine',
    'quick_diff',
    'diff_stats_only',

    # Compare
    'ComparisonEngine',
    'VersionComparison',
    'BranchComparison',
    'EvaluationComparison',
    'create_comparison_engine',

    # Visual
    'TerminalDiff',
    'HTMLDiff',
    'Colors',
    'render_terminal',
    'render_html',
]
