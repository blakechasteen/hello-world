"""
O2 Bot - Platform Anarchism meets Agentic Intelligence

Main bot package integrating Matrix.org + HoloLoom + Democratic Governance.
"""

from bot.o2_bot import O2Bot
from bot.governance import GovernanceEngine
from bot.federated_memory import FederatedMemoryManager
from bot.swarm_coordinator import SwarmCoordinator, AgentType

__all__ = [
    'O2Bot',
    'GovernanceEngine',
    'FederatedMemoryManager',
    'SwarmCoordinator',
    'AgentType'
]

__version__ = '1.0.0'
__author__ = 'O2 Community'
__license__ = 'MIT'
