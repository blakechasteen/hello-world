"""
Deep Thinking — The Deliberation Engine
========================================

System 2 reasoning for HoloLoom. Three AirLLM GPU servers run
a propose→challenge→resolve deliberation cycle with controlled
bleed rates (modeled on dreaming.py). Gracefully degrades through
four tiers: FULL → PARTIAL → DEGRADED → DARK.

Components:
    - DeliberationEngine: Three-phase slow reasoning cycle
    - MilestoneGate: Agent checkpoint reviewer with Thompson-learned thresholds
    - TierState: Spectrum-based infrastructure health tracking
    - DeferredQueue: Activation-weighted queue for rig-down periods
    - SleeptimeScheduler: Idle-time dream cycles
    - DeepClient: Async OpenAI-compatible inference client
"""

from hololoom.core.deep_thinking.client import DeepClient
from hololoom.core.deep_thinking.config import ROLE_GPU_MAP, DeepThinkingConfig
from hololoom.core.deep_thinking.deferred import DeferredJob, DeferredQueue
from hololoom.core.deep_thinking.deliberation import (
    Consensus,
    Critique,
    DeliberationEngine,
    DeliberationResult,
    Proposal,
)
from hololoom.core.deep_thinking.gate import GateVerdict, MilestoneContext, MilestoneGate
from hololoom.core.deep_thinking.sleeptime import SleeptimeScheduler
from hololoom.core.deep_thinking.tier import Tier, TierState

__all__ = [
    "DeepThinkingConfig",
    "ROLE_GPU_MAP",
    "Tier",
    "TierState",
    "DeepClient",
    "DeliberationEngine",
    "DeliberationResult",
    "Proposal",
    "Critique",
    "Consensus",
    "MilestoneGate",
    "MilestoneContext",
    "GateVerdict",
    "DeferredQueue",
    "DeferredJob",
    "SleeptimeScheduler",
]
