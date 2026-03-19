"""
Weaving Strategies
==================
Complexity-based execution strategies for the weaving pipeline.

Each strategy implements a different complexity level:
- LITE: 3 steps for simple queries (<50ms)
- FAST: 5 steps for standard queries (<150ms)
- FULL: 7 steps for complex queries (<300ms)
- RESEARCH: 9 steps for research mode (no limit)
"""

from .base import BaseStrategy
from .factory import create_strategy
from .fast_strategy import FastStrategy
from .full_strategy import FullStrategy
from .lite_strategy import LiteStrategy
from .research_strategy import ResearchStrategy

__all__ = [
    "BaseStrategy",
    "LiteStrategy",
    "FastStrategy",
    "FullStrategy",
    "ResearchStrategy",
    "create_strategy",
]
