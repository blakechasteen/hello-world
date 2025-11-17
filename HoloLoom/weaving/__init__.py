"""
Weaving Module - Refactored Architecture Components
=====================================================

This module contains the refactored components of the weaving orchestrator,
organized into focused, maintainable sub-modules.

**Refactoring Date: November 2025**
**Original File: weaving_orchestrator.py** (3,476 lines)
**Refactored Structure:**

- executors/ - Tool execution layer (ToolExecutor)
- strategies/ - Complexity mode strategies (LITE/FAST/FULL/RESEARCH)
- yarn_graph.py - Memory thread selection
- initialization.py - Component initialization helpers

**Design Philosophy:**
- Single Responsibility: Each class has one clear purpose
- <1,000 lines per file: Maintainable file sizes
- Protocol-based: Clear interfaces between components
- Backward compatible: No breaking changes to public API

Author: Claude Code (Elegance Pass refactoring)
"""

from .executors import ToolExecutor
from .yarn_graph import YarnGraph

__all__ = [
    'ToolExecutor',
    'YarnGraph',
]
