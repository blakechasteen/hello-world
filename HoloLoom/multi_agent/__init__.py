"""
HoloLoom Multi-Agent Collaboration System
==========================================

Enables multiple HoloLoom instances to collaborate as a distributed intelligence network.

**Date**: November 17, 2025
**Phase**: 7 - Multi-Agent Intelligence

Core Modules:
- communication: Agent-to-agent messaging
- consensus: Shared Yarn Graph with CRDT merging
- delegation: Task routing and load balancing
- specialized_agents: Domain-specific agent configurations

Usage:
    from HoloLoom.multi_agent import AgentCommunicator, TaskRouter, MathAgent

    # Create specialized agents
    math_agent = MathAgent(agent_id="math-01")
    code_agent = CodeAgent(agent_id="code-01")

    # Route query to best agent
    router = TaskRouter(communicator)
    result = await router.route_query(query)
"""

from HoloLoom.multi_agent.communication import (
    AgentMessage,
    AgentCommunicator,
    MessageType
)

from HoloLoom.multi_agent.consensus import (
    GraphOperation,
    YarnGraphConsensus,
    ConflictResolver,
    MergeResult
)

from HoloLoom.multi_agent.delegation import (
    AgentCapability,
    TaskRouter,
    SubTask
)

from HoloLoom.multi_agent.specialized_agents import (
    BaseAgent,
    MathAgent,
    CodeAgent,
    WritingAgent,
    ResearchAgent
)

__all__ = [
    # Communication
    "AgentMessage",
    "AgentCommunicator",
    "MessageType",

    # Consensus
    "GraphOperation",
    "YarnGraphConsensus",
    "ConflictResolver",
    "MergeResult",

    # Delegation
    "AgentCapability",
    "TaskRouter",
    "SubTask",

    # Specialized Agents
    "BaseAgent",
    "MathAgent",
    "CodeAgent",
    "WritingAgent",
    "ResearchAgent",
]
