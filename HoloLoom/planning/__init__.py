"""
HoloLoom Hierarchical Planning

Layer 2 of Cognitive Architecture - Goal-directed planning with causal reasoning.

Neurosymbolic Architecture:
- Symbolic: HTN rules, causal DAG structure
- Neural: Learned action models, value functions (future)

Public API:
    HierarchicalPlanner: Main planning engine
    Goal: Desired state
    Plan: Sequence of actions
    Action: Executable operation
"""

from .planner import HierarchicalPlanner, Goal, Plan, Action, ActionType
from .causal_chain import CausalChainFinder
from .multi_agent import (
    Agent, AgentType, MultiAgentCoordinator, NegotiationProtocol,
    Task, Capability, Proposal, Agreement, Coalition, create_agent
)
from .resources import (
    Resource, ResourceType, ResourceRequirement, ResourceState,
    ResourceViolation, ViolationType, ResourceTracker,
    ResourceAwarePlanner, ResourceAllocator
)
from .replanning import (
    ExecutionStatus, ReplanTrigger, ReplanStrategy,
    ExecutionResult, ExecutionTrace, ExecutionMonitor,
    ReplanningEngine, AdaptivePlanner
)
from .pomdp import (
    BeliefState, ObservationAction, ContingentPlan,
    ObservationModel, BeliefUpdater, POMDPPlanner
)

__all__ = [
    # Core planning
    'HierarchicalPlanner',
    'Goal',
    'Plan',
    'Action',
    'ActionType',
    'CausalChainFinder',

    # Multi-agent
    'Agent',
    'AgentType',
    'MultiAgentCoordinator',
    'NegotiationProtocol',
    'Task',
    'Capability',
    'Proposal',
    'Agreement',
    'Coalition',
    'create_agent',

    # Resources
    'Resource',
    'ResourceType',
    'ResourceRequirement',
    'ResourceState',
    'ResourceViolation',
    'ViolationType',
    'ResourceTracker',
    'ResourceAwarePlanner',
    'ResourceAllocator',

    # Replanning
    'ExecutionStatus',
    'ReplanTrigger',
    'ReplanStrategy',
    'ExecutionResult',
    'ExecutionTrace',
    'ExecutionMonitor',
    'ReplanningEngine',
    'AdaptivePlanner',

    # POMDP
    'BeliefState',
    'ObservationAction',
    'ContingentPlan',
    'ObservationModel',
    'BeliefUpdater',
    'POMDPPlanner',
]
