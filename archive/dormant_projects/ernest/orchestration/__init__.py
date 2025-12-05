"""
Ernest Orchestration Module
============================
Full HoloLoom integration for creative writing.

Main exports:
- ErnestOrchestrator: Complete 9-step weaving cycle with Hemingway refinement
- CreativeContext: Context information for mode selection
- quick_ernest_query: One-liner for creative writing queries
- quick_ernest_refine: One-liner for prose refinement
"""

from .ernest_orchestrator import (
    ErnestOrchestrator,
    CreativeContext,
    quick_ernest_query,
    quick_ernest_refine
)

__all__ = [
    "ErnestOrchestrator",
    "CreativeContext",
    "quick_ernest_query",
    "quick_ernest_refine"
]
