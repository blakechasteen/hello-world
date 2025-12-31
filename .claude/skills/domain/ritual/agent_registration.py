"""
Ritual Agent Registration
=========================

Created: December 30, 2025
Purpose: Register ritual phases as agent capabilities for autonomous orchestration.

Implements:
- AgentCapability enum for capability-based routing
- RitualAgentInfo for agent metadata and performance tracking
- RitualAgentRegistry with O(1) capability lookup via reverse index
- DEFAULT_RITUAL_AGENTS for pre-registered phase agents

Follows HoloLoom/agentic/multi_agent.py:AgentRegistry pattern.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Capability Enum
# ============================================================================

class AgentCapability(str, Enum):
    """
    Capabilities that ritual phase agents can provide.

    Enables O(1) lookup in AgentRegistry for capability-based routing.
    Using str mixin for JSON serialization compatibility.
    """
    # Ritual-specific capabilities (primary)
    CONTEXT_RESTORATION = "context_restoration"       # AWAKEN
    PLANNING = "planning"                             # PLAN
    CODE_ASSISTANCE = "code_assistance"               # IMPLEMENT
    QUALITY_ASSURANCE = "quality_assurance"           # REVIEW
    KNOWLEDGE_CONSOLIDATION = "knowledge_consolidation"  # REFLECT

    # Cross-cutting capabilities (secondary)
    MEMORY_RETRIEVAL = "memory_retrieval"
    MEMORY_STORAGE = "memory_storage"
    DECISION_SUPPORT = "decision_support"

    # Extended capabilities for future phases
    CONTEXT_EXPANSION = "context_expansion"
    PATTERN_RECOGNITION = "pattern_recognition"
    SYNTHESIS = "synthesis"


# ============================================================================
# Phase to Capability Mapping
# ============================================================================

# Maps ritual phases to their primary capabilities
PHASE_CAPABILITY_MAP: Dict[str, AgentCapability] = {
    "awaken": AgentCapability.CONTEXT_RESTORATION,
    "plan": AgentCapability.PLANNING,
    "implement": AgentCapability.CODE_ASSISTANCE,
    "review": AgentCapability.QUALITY_ASSURANCE,
    "reflect": AgentCapability.KNOWLEDGE_CONSOLIDATION,
}


# ============================================================================
# Agent Info Dataclass
# ============================================================================

@dataclass
class RitualAgentInfo:
    """
    Agent info for ritual phase registration.

    Mirrors HoloLoom/agentic/multi_agent.py:AgentInfo pattern.
    Includes performance tracking for Thompson Sampling integration.
    """
    # Identity
    id: str                                    # e.g., "ritual-awaken"
    name: str                                  # e.g., "AWAKEN Phase Agent"

    # Capabilities (what this agent can do)
    capabilities: Set[AgentCapability]         # Primary + secondary capabilities

    # Routing metadata
    priority: int = 50                         # Higher = preferred when routing (0-100)
    is_active: bool = True                     # Can be disabled without unregistering

    # Phase-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking (for Thompson Sampling)
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.5

    # Thompson Sampling priors (updated by router)
    thompson_alpha: float = 1.0                # Successes + 1 (Beta prior)
    thompson_beta: float = 1.0                 # Failures + 1 (Beta prior)

    @property
    def success_rate(self) -> float:
        """Calculate success rate from invocation counts."""
        if self.total_invocations == 0:
            return 0.5  # Neutral prior
        return self.successful_invocations / self.total_invocations

    @property
    def expected_reward(self) -> float:
        """
        Expected reward from Thompson Sampling.
        E[X] = α / (α + β)
        """
        return self.thompson_alpha / (self.thompson_alpha + self.thompson_beta)

    def record_invocation(
        self,
        success: bool,
        latency_ms: float,
        confidence: float
    ) -> None:
        """
        Record an invocation for performance tracking.

        Updates running averages and Thompson priors.
        """
        self.total_invocations += 1

        if success:
            self.successful_invocations += 1
            # Thompson update for success: α ← α + confidence
            self.thompson_alpha += confidence
        else:
            self.failed_invocations += 1
            # Thompson update for failure: β ← β + (1 - confidence)
            self.thompson_beta += (1.0 - confidence)

        # Update running average latency
        n = self.total_invocations
        self.avg_latency_ms = (
            (self.avg_latency_ms * (n - 1) + latency_ms) / n
        )

        # Update running average confidence
        self.avg_confidence = (
            (self.avg_confidence * (n - 1) + confidence) / n
        )

    def reset_performance(self) -> None:
        """Reset performance tracking to initial state."""
        self.total_invocations = 0
        self.successful_invocations = 0
        self.failed_invocations = 0
        self.avg_latency_ms = 0.0
        self.avg_confidence = 0.5
        self.thompson_alpha = 1.0
        self.thompson_beta = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "id": self.id,
            "name": self.name,
            "capabilities": [c.value for c in self.capabilities],
            "priority": self.priority,
            "is_active": self.is_active,
            "metadata": self.metadata,
            "performance": {
                "total_invocations": self.total_invocations,
                "successful_invocations": self.successful_invocations,
                "failed_invocations": self.failed_invocations,
                "avg_latency_ms": self.avg_latency_ms,
                "avg_confidence": self.avg_confidence,
                "thompson_alpha": self.thompson_alpha,
                "thompson_beta": self.thompson_beta,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RitualAgentInfo":
        """Deserialize from dictionary."""
        agent = cls(
            id=data["id"],
            name=data["name"],
            capabilities={AgentCapability(c) for c in data["capabilities"]},
            priority=data.get("priority", 50),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )

        # Restore performance if present
        if "performance" in data:
            perf = data["performance"]
            agent.total_invocations = perf.get("total_invocations", 0)
            agent.successful_invocations = perf.get("successful_invocations", 0)
            agent.failed_invocations = perf.get("failed_invocations", 0)
            agent.avg_latency_ms = perf.get("avg_latency_ms", 0.0)
            agent.avg_confidence = perf.get("avg_confidence", 0.5)
            agent.thompson_alpha = perf.get("thompson_alpha", 1.0)
            agent.thompson_beta = perf.get("thompson_beta", 1.0)

        return agent


# ============================================================================
# Agent Registry
# ============================================================================

class RitualAgentRegistry:
    """
    Registry for ritual phase agents.

    Mirrors HoloLoom/agentic/multi_agent.py:AgentRegistry pattern.
    O(1) lookup via capability → agent_ids reverse index.

    Features:
    - Register/unregister agents dynamically
    - Find agents by capability (O(1) lookup)
    - Priority-based ordering when multiple agents match
    - Performance tracking integration
    - State persistence/restoration
    """

    def __init__(self):
        # Primary storage: agent_id → agent_info
        self._agents: Dict[str, RitualAgentInfo] = {}

        # Reverse index: capability → set of agent IDs (O(1) lookup)
        self._capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)

        # Track registration order for deterministic iteration
        self._registration_order: List[str] = []

    def register(self, agent: RitualAgentInfo, replace: bool = False) -> bool:
        """
        Register a ritual phase agent with its capabilities.

        Args:
            agent: Agent info to register
            replace: If True, replace existing agent with same ID

        Returns:
            True if registered, False if already exists (and replace=False)
        """
        if agent.id in self._agents:
            if not replace:
                logger.warning(f"Agent '{agent.id}' already registered. Use replace=True to override.")
                return False
            # Unregister existing before replacing
            self.unregister(agent.id)

        # Store agent
        self._agents[agent.id] = agent
        self._registration_order.append(agent.id)

        # Build reverse index for O(1) capability lookup
        for capability in agent.capabilities:
            self._capability_index[capability].add(agent.id)

        logger.info(f"Registered agent '{agent.id}' with capabilities: {[c.value for c in agent.capabilities]}")
        return True

    def unregister(self, agent_id: str) -> bool:
        """
        Unregister an agent by ID.

        Returns:
            True if unregistered, False if not found
        """
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]

        # Remove from capability index
        for capability in agent.capabilities:
            self._capability_index[capability].discard(agent_id)

        # Remove from primary storage
        del self._agents[agent_id]

        # Remove from registration order
        if agent_id in self._registration_order:
            self._registration_order.remove(agent_id)

        logger.info(f"Unregistered agent '{agent_id}'")
        return True

    def get_agent(self, agent_id: str) -> Optional[RitualAgentInfo]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def find_by_capability(
        self,
        capability: AgentCapability,
        active_only: bool = True
    ) -> List[RitualAgentInfo]:
        """
        Find all agents that provide a capability. O(1) lookup.

        Args:
            capability: Capability to search for
            active_only: If True, exclude inactive agents

        Returns:
            List of agents sorted by priority (highest first)
        """
        agent_ids = self._capability_index.get(capability, set())
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]

        if active_only:
            agents = [a for a in agents if a.is_active]

        # Sort by priority (descending) then by expected reward
        agents.sort(key=lambda a: (a.priority, a.expected_reward), reverse=True)

        return agents

    def find_by_phase(self, phase_name: str) -> Optional[RitualAgentInfo]:
        """
        Find the primary agent for a ritual phase.

        Args:
            phase_name: Phase name (awaken, plan, implement, review, reflect)

        Returns:
            Best matching agent or None
        """
        capability = PHASE_CAPABILITY_MAP.get(phase_name.lower())
        if not capability:
            return None

        agents = self.find_by_capability(capability)
        return agents[0] if agents else None

    def list_agents(self, active_only: bool = False) -> List[RitualAgentInfo]:
        """List all registered agents in registration order."""
        agents = [self._agents[aid] for aid in self._registration_order if aid in self._agents]
        if active_only:
            agents = [a for a in agents if a.is_active]
        return agents

    def list_capabilities(self) -> List[AgentCapability]:
        """List all capabilities that have registered agents."""
        return [cap for cap, ids in self._capability_index.items() if ids]

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        agents = self.list_agents()
        active = [a for a in agents if a.is_active]

        total_invocations = sum(a.total_invocations for a in agents)
        total_successes = sum(a.successful_invocations for a in agents)

        return {
            "total_agents": len(agents),
            "active_agents": len(active),
            "capabilities_covered": len(self.list_capabilities()),
            "total_invocations": total_invocations,
            "overall_success_rate": (
                total_successes / total_invocations
                if total_invocations > 0 else 0.5
            ),
            "agents_by_capability": {
                cap.value: len(ids)
                for cap, ids in self._capability_index.items() if ids
            }
        }

    def record_invocation(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
        confidence: float
    ) -> bool:
        """
        Record an invocation for an agent.

        Returns:
            True if recorded, False if agent not found
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.record_invocation(success, latency_ms, confidence)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry state for persistence."""
        return {
            "version": "1.0",
            "agents": [agent.to_dict() for agent in self.list_agents()],
            "registration_order": self._registration_order.copy(),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        """
        Load registry state from persisted data.

        Clears existing state and replaces with loaded data.
        """
        # Clear existing
        self._agents.clear()
        self._capability_index.clear()
        self._registration_order.clear()

        # Load agents
        for agent_data in data.get("agents", []):
            agent = RitualAgentInfo.from_dict(agent_data)
            self.register(agent)

        logger.info(f"Loaded {len(self._agents)} agents from state")


# ============================================================================
# Default Ritual Phase Agents
# ============================================================================

DEFAULT_RITUAL_AGENTS: List[RitualAgentInfo] = [
    RitualAgentInfo(
        id="ritual-awaken",
        name="AWAKEN Phase Agent",
        capabilities={
            AgentCapability.CONTEXT_RESTORATION,
            AgentCapability.MEMORY_RETRIEVAL,
        },
        priority=60,
        metadata={
            "phase": "awaken",
            "skill": "ritual-awaken",
            "description": "Restore context at session start",
            "decision_options": ["Proceed to PLAN", "Explore more context", "Skip to IMPLEMENT"],
        }
    ),
    RitualAgentInfo(
        id="ritual-plan",
        name="PLAN Phase Agent",
        capabilities={
            AgentCapability.PLANNING,
            AgentCapability.MEMORY_RETRIEVAL,
            AgentCapability.MEMORY_STORAGE,
        },
        priority=50,
        metadata={
            "phase": "plan",
            "skill": "ritual-plan",
            "description": "Capture requirements and create implementation plan",
            "decision_options": ["Approve plan", "Refine plan", "Research more", "Skip to IMPLEMENT"],
        }
    ),
    RitualAgentInfo(
        id="ritual-implement",
        name="IMPLEMENT Phase Agent",
        capabilities={
            AgentCapability.CODE_ASSISTANCE,
            AgentCapability.MEMORY_RETRIEVAL,
        },
        priority=40,
        metadata={
            "phase": "implement",
            "skill": "ritual-implement-assist",
            "description": "Optional assistance during manual implementation",
            "decision_options": [],  # No decision point - manual phase
        }
    ),
    RitualAgentInfo(
        id="ritual-review",
        name="REVIEW Phase Agent",
        capabilities={
            AgentCapability.QUALITY_ASSURANCE,
            AgentCapability.MEMORY_STORAGE,
        },
        priority=50,
        metadata={
            "phase": "review",
            "skill": "ritual-review",
            "description": "Quality check and capture learnings",
            "decision_options": ["Approve", "Address issues first", "Skip review"],
        }
    ),
    RitualAgentInfo(
        id="ritual-reflect",
        name="REFLECT Phase Agent",
        capabilities={
            AgentCapability.KNOWLEDGE_CONSOLIDATION,
            AgentCapability.MEMORY_STORAGE,
        },
        priority=60,
        metadata={
            "phase": "reflect",
            "skill": "ritual-reflect",
            "description": "End-of-session consolidation",
            "decision_options": ["Confirm summary", "Add more notes", "Quick close"],
        }
    ),
]


# ============================================================================
# Factory Functions
# ============================================================================

def create_registry(
    register_defaults: bool = True,
    load_from: Optional[Dict[str, Any]] = None
) -> RitualAgentRegistry:
    """
    Create a ritual agent registry.

    Args:
        register_defaults: If True, register DEFAULT_RITUAL_AGENTS
        load_from: Optional state dict to load from

    Returns:
        Configured RitualAgentRegistry
    """
    registry = RitualAgentRegistry()

    if load_from:
        registry.load_state(load_from)
    elif register_defaults:
        for agent in DEFAULT_RITUAL_AGENTS:
            registry.register(agent)

    return registry


def get_capability_for_phase(phase_name: str) -> Optional[AgentCapability]:
    """
    Get the primary capability for a ritual phase.

    Args:
        phase_name: Phase name (awaken, plan, implement, review, reflect)

    Returns:
        AgentCapability or None if invalid phase
    """
    return PHASE_CAPABILITY_MAP.get(phase_name.lower())


def get_phase_for_capability(capability: AgentCapability) -> Optional[str]:
    """
    Get the ritual phase for a capability.

    Args:
        capability: Agent capability

    Returns:
        Phase name or None if not a primary phase capability
    """
    for phase, cap in PHASE_CAPABILITY_MAP.items():
        if cap == capability:
            return phase
    return None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    'AgentCapability',

    # Dataclasses
    'RitualAgentInfo',

    # Registry
    'RitualAgentRegistry',

    # Constants
    'PHASE_CAPABILITY_MAP',
    'DEFAULT_RITUAL_AGENTS',

    # Factory functions
    'create_registry',
    'get_capability_for_phase',
    'get_phase_for_capability',
]
