"""
Recursive Orchestrator - Compatibility Shim
=============================================

DEPRECATED: Moved to hololoom.core.orchestrator.variants.recursive

New code should import from:
    from hololoom.core.orchestrator.variants.recursive import RecursiveWeavingOrchestrator
"""

from hololoom.core.orchestrator.variants.recursive import *  # noqa: F401, F403
from hololoom.core.orchestrator.variants.recursive import (  # noqa: F401
    RecursiveSpacetime,
    RecursiveWeavingOrchestrator,
    create_recursive_orchestrator,
)
