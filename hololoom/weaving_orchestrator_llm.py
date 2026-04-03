"""
LLM Orchestrator - Compatibility Shim
=======================================

DEPRECATED: Moved to hololoom.core.orchestrator.variants.llm

New code should import from:
    from hololoom.core.orchestrator.variants.llm import LLMToolExecutor
"""

from hololoom.core.orchestrator.variants.llm import *  # noqa: F401, F403
from hololoom.core.orchestrator.variants.llm import (  # noqa: F401
    LLMToolExecutor,
)
