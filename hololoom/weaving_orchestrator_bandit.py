"""
Bandit Orchestrator - Compatibility Shim
=========================================

DEPRECATED: Moved to hololoom.core.orchestrator.variants.bandit

New code should import from:
    from hololoom.core.orchestrator.variants.bandit import BanditOrchestrator
"""

from hololoom.core.orchestrator.variants.bandit import *  # noqa: F401, F403
from hololoom.core.orchestrator.variants.bandit import (  # noqa: F401
    BanditConfig,
    BanditOrchestrator,
    create_bandit_orchestrator,
)
