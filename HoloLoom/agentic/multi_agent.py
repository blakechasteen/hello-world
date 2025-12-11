"""
Multi-Agent Communication Protocol - Phase 7.1
===============================================

Provides infrastructure for multiple HoloLoom agents to communicate,
coordinate, and share context. Builds on Phase 6.3 context_handoff.py.

Key Components:
- AgentMessage: Standard message format for inter-agent communication
- AgentCapability: Enumeration of agent specializations
- AgentRegistry: Discovery and management of available agents
- MessageBus: Async message passing system
- SharedMemory: Context sharing interface between agents

Author: Claude Code
Date: 2025-12-09
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable, Union, TYPE_CHECKING
from datetime import datetime
import asyncio
import uuid
import json
from collections import defaultdict

from .context_handoff import (
    ContextHandoff,
    ContextItem,
    HandoffContext,
    HandoffStrategy,
)

# Import shared enums from protocol (no circular import)
from .protocol import (
    AgentCapability,
    MessageType,
    MessagePriority,
    AgentStatus,
    SelectionStrategy,
    QueryComplexity,
    EnsembleStrategy,
)

# Phase 7.2: Task Delegation imports are lazy to avoid circular imports
# These are imported inside delegate_task_smart() method
if TYPE_CHECKING:
    from .capability_analyzer import QueryCapabilityAnalyzer, CapabilityAnalysis
    from .expert_router import ExpertRouter, RoutingDecision, AgentScore
    from .ensemble_decision import EnsembleAggregator, EnsembleResult, AgentResponse


# ============================================================================
# Message Classes
# ============================================================================

@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.

    Supports both synchronous and asynchronous workflows,
    with built-in context passing and priority handling.
    """
    # Message identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: Optional[str] = None  # Links related messages

    # Routing
    from_agent: str = ""
    to_agent: str = ""  # Empty = broadcast

    # Content
    type: MessageType = MessageType.REQUEST
    content: Any = None
    context: List[ContextItem] = field(default_factory=list)

    # Metadata
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: Optional[int] = None  # Time-to-live

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.timestamp)
        elapsed = (datetime.now() - created).total_seconds()
        return elapsed > self.ttl_seconds

    def create_response(
        self,
        content: Any,
        from_agent: str,
        msg_type: MessageType = MessageType.RESPONSE
    ) -> "AgentMessage":
        """Create a response message to this message."""
        return AgentMessage(
            correlation_id=self.id,
            from_agent=from_agent,
            to_agent=self.from_agent,
            type=msg_type,
            content=content,
            priority=self.priority,
            context=self.context.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type.value,
            "content": self.content,
            "context": [
                {
                    "content": c.content,
                    "source_agent": c.source_agent,
                    "mi_score": c.mi_score,
                }
                for c in self.context
            ],
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Deserialize message from dictionary."""
        context = [
            ContextItem(
                content=c["content"],
                source_agent=c["source_agent"],
                mi_score=c.get("mi_score", 0.0),
            )
            for c in data.get("context", [])
        ]
        return cls(
            id=data["id"],
            correlation_id=data.get("correlation_id"),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            type=MessageType(data["type"]),
            content=data["content"],
            context=context,
            priority=MessagePriority(data["priority"]),
            timestamp=data["timestamp"],
            ttl_seconds=data.get("ttl_seconds"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    id: str
    name: str
    capabilities: Set[AgentCapability]
    status: AgentStatus = AgentStatus.IDLE

    # Performance metrics
    avg_response_time_ms: float = 0.0
    total_requests: int = 0
    success_rate: float = 1.0

    # Configuration
    max_concurrent_tasks: int = 3
    current_tasks: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_available(self) -> bool:
        """Check if agent can accept new tasks."""
        return (
            self.status in (AgentStatus.IDLE, AgentStatus.WAITING) and
            self.current_tasks < self.max_concurrent_tasks
        )

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability."""
        return capability in self.capabilities

    def update_metrics(self, response_time_ms: float, success: bool) -> None:
        """Update performance metrics after task completion."""
        self.total_requests += 1

        # Update running average response time
        self.avg_response_time_ms = (
            (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms)
            / self.total_requests
        )

        # Update success rate
        old_successes = self.success_rate * (self.total_requests - 1)
        self.success_rate = (old_successes + (1.0 if success else 0.0)) / self.total_requests

        self.last_active = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent info to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "capabilities": [c.value for c in self.capabilities],
            "status": self.status.value,
            "avg_response_time_ms": self.avg_response_time_ms,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_tasks": self.current_tasks,
            "metadata": self.metadata,
            "registered_at": self.registered_at,
            "last_active": self.last_active,
        }


# ============================================================================
# Agent Registry
# ============================================================================

class AgentRegistry:
    """
    Registry for discovering and managing available agents.

    Provides:
    - Agent registration and discovery
    - Capability-based routing
    - Load balancing across agents
    - Health monitoring

    Example:
        registry = AgentRegistry()

        # Register agents
        registry.register(AgentInfo(
            id="researcher",
            name="Research Agent",
            capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS}
        ))

        # Find agents by capability
        agents = registry.find_by_capability(AgentCapability.RESEARCH)

        # Get best agent for task
        best = registry.get_best_agent(AgentCapability.RESEARCH)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._agents: Dict[str, AgentInfo] = {}
        self._capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register(self, agent: AgentInfo) -> bool:
        """
        Register an agent with the registry.

        Args:
            agent: Agent information

        Returns:
            True if registered successfully
        """
        async with self._lock:
            if agent.id in self._agents:
                # Update existing registration
                self._agents[agent.id] = agent
            else:
                # New registration
                self._agents[agent.id] = agent

            # Update capability index
            for capability in agent.capabilities:
                self._capability_index[capability].add(agent.id)

            return True

    async def unregister(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if unregistered successfully
        """
        async with self._lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]

            # Remove from capability index
            for capability in agent.capabilities:
                self._capability_index[capability].discard(agent_id)

            # Remove from registry
            del self._agents[agent_id]

            return True

    def get(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def find_by_capability(
        self,
        capability: AgentCapability,
        only_available: bool = True
    ) -> List[AgentInfo]:
        """
        Find agents with specific capability.

        Args:
            capability: Required capability
            only_available: Only return available agents

        Returns:
            List of matching agents
        """
        agent_ids = self._capability_index.get(capability, set())
        agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]

        if only_available:
            agents = [a for a in agents if a.is_available()]

        return agents

    def get_best_agent(
        self,
        capability: AgentCapability,
        strategy: str = "balanced"
    ) -> Optional[AgentInfo]:
        """
        Get the best agent for a capability using specified strategy.

        Strategies:
        - "fastest": Lowest average response time
        - "reliable": Highest success rate
        - "balanced": Balance of speed and reliability
        - "random": Random selection

        Args:
            capability: Required capability
            strategy: Selection strategy

        Returns:
            Best matching agent or None
        """
        agents = self.find_by_capability(capability, only_available=True)

        if not agents:
            return None

        if strategy == "fastest":
            return min(agents, key=lambda a: a.avg_response_time_ms)

        elif strategy == "reliable":
            return max(agents, key=lambda a: a.success_rate)

        elif strategy == "balanced":
            # Score = success_rate / (1 + normalized_response_time)
            def score(agent: AgentInfo) -> float:
                # Normalize response time (0-1 range, lower is better)
                max_time = max(a.avg_response_time_ms for a in agents) or 1.0
                norm_time = agent.avg_response_time_ms / max_time
                return agent.success_rate / (1.0 + norm_time)

            return max(agents, key=score)

        elif strategy == "random":
            import random
            return random.choice(agents)

        else:
            # Default to first available
            return agents[0]

    def get_all_agents(self) -> List[AgentInfo]:
        """Get all registered agents."""
        return list(self._agents.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        agents = list(self._agents.values())

        available_count = sum(1 for a in agents if a.is_available())

        # Aggregate metrics
        total_requests = sum(a.total_requests for a in agents)
        avg_success = (
            sum(a.success_rate * a.total_requests for a in agents) / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            "total_agents": len(agents),
            "available_agents": available_count,
            "total_requests": total_requests,
            "avg_success_rate": avg_success,
            "capabilities": {
                cap.value: len(ids)
                for cap, ids in self._capability_index.items()
            },
        }


# ============================================================================
# Message Bus
# ============================================================================

# Type alias for message handlers
MessageHandler = Callable[[AgentMessage], Awaitable[Optional[AgentMessage]]]


class MessageBus:
    """
    Async message passing system for inter-agent communication.

    Provides:
    - Pub/sub messaging
    - Request/response patterns
    - Priority-based delivery
    - Message routing

    Example:
        bus = MessageBus()

        # Subscribe to messages
        async def handler(msg: AgentMessage):
            return msg.create_response("Got it!", "agent_b")

        bus.subscribe("agent_b", handler)

        # Send message
        response = await bus.request(AgentMessage(
            from_agent="agent_a",
            to_agent="agent_b",
            type=MessageType.REQUEST,
            content="Hello!"
        ))
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        """
        Initialize message bus.

        Args:
            registry: Optional agent registry for routing
        """
        self.registry = registry or AgentRegistry()
        self._handlers: Dict[str, MessageHandler] = {}
        self._queues: Dict[str, asyncio.PriorityQueue] = {}
        self._broadcast_handlers: List[MessageHandler] = []
        self._message_history: List[AgentMessage] = []
        self._history_limit: int = 1000
        self._lock = asyncio.Lock()

    def subscribe(
        self,
        agent_id: str,
        handler: MessageHandler,
        queue_size: int = 100
    ) -> None:
        """
        Subscribe an agent to receive messages.

        Args:
            agent_id: Agent ID to subscribe
            handler: Async handler function for messages
            queue_size: Maximum queue size
        """
        self._handlers[agent_id] = handler
        self._queues[agent_id] = asyncio.PriorityQueue(maxsize=queue_size)

    def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from messages."""
        self._handlers.pop(agent_id, None)
        self._queues.pop(agent_id, None)

    def subscribe_broadcast(self, handler: MessageHandler) -> None:
        """Subscribe to all broadcast messages."""
        self._broadcast_handlers.append(handler)

    async def send(self, message: AgentMessage) -> bool:
        """
        Send a message without waiting for response.

        Args:
            message: Message to send

        Returns:
            True if message was delivered
        """
        if message.is_expired():
            return False

        # Track message
        await self._track_message(message)

        # Broadcast message
        if message.type == MessageType.BROADCAST or not message.to_agent:
            for handler in self._broadcast_handlers:
                asyncio.create_task(self._safe_call(handler, message))
            return True

        # Direct message
        if message.to_agent in self._handlers:
            handler = self._handlers[message.to_agent]
            asyncio.create_task(self._safe_call(handler, message))
            return True

        return False

    async def request(
        self,
        message: AgentMessage,
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response.

        Args:
            message: Request message
            timeout: Timeout in seconds

        Returns:
            Response message or None on timeout
        """
        if message.is_expired():
            return None

        # Track message
        await self._track_message(message)

        # Create response future
        response_future: asyncio.Future = asyncio.Future()

        # Wrap handler to capture response
        original_handler = self._handlers.get(message.to_agent)
        if not original_handler:
            return None

        async def capturing_handler(msg: AgentMessage) -> Optional[AgentMessage]:
            response = await original_handler(msg)
            if response and response.correlation_id == message.id:
                if not response_future.done():
                    response_future.set_result(response)
            return response

        # Temporarily replace handler
        self._handlers[message.to_agent] = capturing_handler

        try:
            # Send message
            await self.send(message)

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)

            # Track response
            if response:
                await self._track_message(response)

            return response

        except asyncio.TimeoutError:
            return None
        finally:
            # Restore original handler
            self._handlers[message.to_agent] = original_handler

    async def route_by_capability(
        self,
        message: AgentMessage,
        capability: AgentCapability,
        strategy: str = "balanced"
    ) -> Optional[AgentMessage]:
        """
        Route message to best agent with capability.

        Args:
            message: Message to route
            capability: Required capability
            strategy: Agent selection strategy

        Returns:
            Response from selected agent or None
        """
        agent = self.registry.get_best_agent(capability, strategy)
        if not agent:
            return None

        message.to_agent = agent.id
        return await self.request(message)

    async def _safe_call(
        self,
        handler: MessageHandler,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Safely call handler with error handling."""
        try:
            return await handler(message)
        except Exception as e:
            # Return error message
            return message.create_response(
                content={"error": str(e)},
                from_agent="system",
                msg_type=MessageType.ERROR
            )

    async def _track_message(self, message: AgentMessage) -> None:
        """Track message in history."""
        async with self._lock:
            self._message_history.append(message)

            # Prune old messages
            if len(self._message_history) > self._history_limit:
                self._message_history = self._message_history[-self._history_limit:]

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Get message history.

        Args:
            agent_id: Filter by agent (None = all)
            limit: Maximum messages to return

        Returns:
            List of messages (newest first)
        """
        messages = self._message_history

        if agent_id:
            messages = [
                m for m in messages
                if m.from_agent == agent_id or m.to_agent == agent_id
            ]

        return messages[-limit:][::-1]

    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        messages = self._message_history

        by_type = defaultdict(int)
        for msg in messages:
            by_type[msg.type.value] += 1

        return {
            "total_messages": len(messages),
            "subscribed_agents": len(self._handlers),
            "broadcast_handlers": len(self._broadcast_handlers),
            "by_type": dict(by_type),
        }


# ============================================================================
# Shared Memory
# ============================================================================

@dataclass
class SharedMemoryEntry:
    """Entry in shared memory."""
    key: str
    value: Any
    owner: str  # Agent that created entry
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        created = datetime.fromisoformat(self.created_at)
        elapsed = (datetime.now() - created).total_seconds()
        return elapsed > self.ttl_seconds


class SharedMemory:
    """
    Shared memory interface for context sharing between agents.

    Provides:
    - Key-value storage for shared context
    - Namespace isolation
    - TTL-based expiration
    - Access tracking
    - Integration with context_handoff for MI-based filtering

    Example:
        memory = SharedMemory()

        # Store context
        await memory.put("research_findings", findings, owner="researcher")

        # Retrieve context
        findings = await memory.get("research_findings")

        # Get context for handoff
        context = await memory.get_context_for_handoff(
            target_agent="summarizer",
            target_capability="summarization"
        )
    """

    def __init__(self, context_handoff: Optional[ContextHandoff] = None):
        """
        Initialize shared memory.

        Args:
            context_handoff: Optional ContextHandoff for MI-based filtering
        """
        self._store: Dict[str, SharedMemoryEntry] = {}
        self._namespaces: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._context_handoff = context_handoff or ContextHandoff()

    async def put(
        self,
        key: str,
        value: Any,
        owner: str,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store value in shared memory.

        Args:
            key: Storage key
            value: Value to store
            owner: Agent that owns this entry
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        async with self._lock:
            full_key = f"{namespace}:{key}"

            entry = SharedMemoryEntry(
                key=full_key,
                value=value,
                owner=owner,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
            )

            self._store[full_key] = entry
            self._namespaces[namespace].add(full_key)

            return True

    async def get(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Any]:
        """
        Retrieve value from shared memory.

        Args:
            key: Storage key
            namespace: Namespace

        Returns:
            Stored value or None
        """
        async with self._lock:
            full_key = f"{namespace}:{key}"
            entry = self._store.get(full_key)

            if entry is None:
                return None

            if entry.is_expired():
                del self._store[full_key]
                self._namespaces[namespace].discard(full_key)
                return None

            entry.access_count += 1
            return entry.value

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete entry from shared memory."""
        async with self._lock:
            full_key = f"{namespace}:{key}"
            if full_key in self._store:
                del self._store[full_key]
                self._namespaces[namespace].discard(full_key)
                return True
            return False

    async def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all entries in a namespace."""
        async with self._lock:
            keys = self._namespaces.get(namespace, set())
            result = {}

            expired_keys = []
            for key in keys:
                entry = self._store.get(key)
                if entry:
                    if entry.is_expired():
                        expired_keys.append(key)
                    else:
                        short_key = key.split(":", 1)[1]
                        result[short_key] = entry.value

            # Clean up expired entries
            for key in expired_keys:
                del self._store[key]
                self._namespaces[namespace].discard(key)

            return result

    async def get_context_for_handoff(
        self,
        from_agent: str,
        target_agent: str,
        target_capability: str,
        namespace: str = "default",
        strategy: Optional[HandoffStrategy] = None,
        token_budget: Optional[int] = None
    ) -> HandoffContext:
        """
        Get context from shared memory filtered for agent handoff.

        Uses MI-based filtering from Phase 6.3 context_handoff.

        Args:
            from_agent: Source agent ID
            target_agent: Target agent ID
            target_capability: Target agent's capability
            namespace: Memory namespace to use
            strategy: Handoff strategy (auto-selected if None)
            token_budget: Maximum tokens for context

        Returns:
            HandoffContext with filtered context
        """
        # Get all entries in namespace
        entries = await self.get_namespace(namespace)

        # Convert to context strings
        context_strings = []
        for key, value in entries.items():
            if isinstance(value, str):
                context_strings.append(value)
            elif isinstance(value, dict) and "content" in value:
                context_strings.append(value["content"])
            else:
                context_strings.append(str(value))

        # Use context_handoff for MI-based filtering
        return self._context_handoff.prepare_handoff(
            from_agent=from_agent,
            to_agent=target_agent,
            full_context=context_strings,
            target_agent_capability=target_capability,
            strategy=strategy,
            token_budget=token_budget,
        )

    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        async with self._lock:
            expired_keys = []

            for key, entry in self._store.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                namespace = key.split(":", 1)[0]
                del self._store[key]
                self._namespaces[namespace].discard(key)

            return len(expired_keys)

    def get_statistics(self) -> Dict[str, Any]:
        """Get shared memory statistics."""
        entries = list(self._store.values())

        total_access = sum(e.access_count for e in entries)

        by_owner = defaultdict(int)
        for entry in entries:
            by_owner[entry.owner] += 1

        return {
            "total_entries": len(entries),
            "namespaces": len(self._namespaces),
            "total_access_count": total_access,
            "entries_by_owner": dict(by_owner),
        }


# ============================================================================
# Multi-Agent Coordinator
# ============================================================================

class MultiAgentCoordinator:
    """
    High-level coordinator for multi-agent workflows.

    Combines registry, message bus, and shared memory into
    a unified interface for agent coordination.

    Example:
        coordinator = MultiAgentCoordinator()

        # Register agents
        await coordinator.register_agent(AgentInfo(...))

        # Send task to best agent
        result = await coordinator.delegate_task(
            task="Summarize these findings",
            capability=AgentCapability.SUMMARIZATION,
            context={"findings": research_data}
        )
    """

    def __init__(self):
        """Initialize coordinator with all components."""
        self.registry = AgentRegistry()
        self.memory = SharedMemory()
        self.bus = MessageBus(self.registry)

    async def register_agent(
        self,
        agent: AgentInfo,
        handler: MessageHandler
    ) -> bool:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent information
            handler: Message handler for the agent

        Returns:
            True if registered successfully
        """
        await self.registry.register(agent)
        self.bus.subscribe(agent.id, handler)
        return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        await self.registry.unregister(agent_id)
        self.bus.unsubscribe(agent_id)
        return True

    async def delegate_task(
        self,
        task: Any,
        capability: AgentCapability,
        from_agent: str = "coordinator",
        context: Optional[Dict[str, Any]] = None,
        strategy: str = "balanced",
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Delegate a task to the best available agent.

        Args:
            task: Task content
            capability: Required capability
            from_agent: Requesting agent ID
            context: Additional context
            strategy: Agent selection strategy
            timeout: Timeout in seconds

        Returns:
            Response message or None
        """
        # Create request message
        message = AgentMessage(
            from_agent=from_agent,
            type=MessageType.REQUEST,
            content=task,
            metadata=context or {},
        )

        # Route to best agent
        return await self.bus.route_by_capability(
            message=message,
            capability=capability,
            strategy=strategy,
        )

    async def delegate_task_smart(
        self,
        task: str,
        capability: Optional[AgentCapability] = None,
        from_agent: str = "coordinator",
        context: Optional[Dict[str, Any]] = None,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        timeout: float = 30.0,
        enable_auto_detect: bool = True,
        enable_ensemble: bool = False,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.CONFIDENCE_WEIGHTED,
        ensemble_top_k: int = 3,
    ) -> Union[Optional[AgentMessage], EnsembleResult]:
        """
        Smart task delegation with automatic capability detection and ensemble support.

        This enhanced version of delegate_task provides:
        - Automatic query→capability detection (if capability=None)
        - Thompson Sampling-based agent selection via ExpertRouter
        - Ensemble decisions when multiple agents should be consulted

        Args:
            task: Task description (used for capability detection)
            capability: Explicit capability (auto-detected if None)
            from_agent: Requesting agent ID
            context: Additional context
            strategy: Agent selection strategy (FASTEST, RELIABLE, BALANCED, THOMPSON)
            timeout: Timeout in seconds
            enable_auto_detect: Enable automatic capability detection
            enable_ensemble: Enable multi-agent ensemble decisions
            ensemble_strategy: Strategy for aggregating ensemble responses
            ensemble_top_k: Number of agents to query for ensemble

        Returns:
            AgentMessage if single-agent, EnsembleResult if ensemble

        Example:
            # Auto-detect capability
            result = await coordinator.delegate_task_smart(
                task="Review this Python code for bugs",
                # capability auto-detected as CODE_REVIEW
            )

            # Ensemble decision
            result = await coordinator.delegate_task_smart(
                task="Analyze the tradeoffs of this architecture",
                enable_ensemble=True,
                ensemble_top_k=3,
            )
        """
        # Initialize components lazily
        if not hasattr(self, '_capability_analyzer'):
            self._capability_analyzer = QueryCapabilityAnalyzer()
        if not hasattr(self, '_expert_router'):
            self._expert_router = ExpertRouter(self.registry)
        if not hasattr(self, '_ensemble_aggregator'):
            self._ensemble_aggregator = EnsembleAggregator()

        # Step 1: Capability Detection
        detected_capability = capability
        capability_analysis = None

        if capability is None and enable_auto_detect:
            capability_analysis = self._capability_analyzer.analyze(task)
            detected_capability = capability_analysis.primary_capability

            # Auto-enable ensemble for complex multi-capability queries
            if capability_analysis.requires_ensemble and not enable_ensemble:
                enable_ensemble = True

        if detected_capability is None:
            detected_capability = AgentCapability.GENERAL

        # Step 2: Agent Selection via ExpertRouter
        routing_decision = self._expert_router.route(
            required_capability=detected_capability,
            strategy=strategy,
            top_k=ensemble_top_k if enable_ensemble else 1,
        )

        if not routing_decision.selected_agent:
            # No available agent for this capability
            return None

        # Step 3: Single Agent or Ensemble Execution
        if enable_ensemble and len(routing_decision.alternatives) > 0:
            return await self._execute_ensemble(
                task=task,
                routing_decision=routing_decision,
                from_agent=from_agent,
                context=context,
                timeout=timeout,
                ensemble_strategy=ensemble_strategy,
                capability_analysis=capability_analysis,
            )
        else:
            # Single agent delegation
            return await self._execute_single_agent(
                task=task,
                agent_id=routing_decision.selected_agent,
                from_agent=from_agent,
                context=context,
                timeout=timeout,
                routing_decision=routing_decision,
                capability_analysis=capability_analysis,
            )

    async def _execute_single_agent(
        self,
        task: str,
        agent_id: str,
        from_agent: str,
        context: Optional[Dict[str, Any]],
        timeout: float,
        routing_decision: RoutingDecision,
        capability_analysis: Optional[CapabilityAnalysis],
    ) -> Optional[AgentMessage]:
        """Execute task with a single agent."""
        start_time = datetime.now()

        # Create request message
        metadata = context.copy() if context else {}
        metadata["routing"] = {
            "strategy": routing_decision.score.agent_id if routing_decision else "direct",
            "used_thompson": routing_decision.used_thompson if routing_decision else False,
            "routing_time_ms": routing_decision.routing_time_ms if routing_decision else 0.0,
        }
        if capability_analysis:
            metadata["capability_detection"] = {
                "detected": capability_analysis.primary_capability.value,
                "confidence": capability_analysis.confidence,
                "complexity": capability_analysis.complexity.value,
            }

        message = AgentMessage(
            from_agent=from_agent,
            to_agent=agent_id,
            type=MessageType.REQUEST,
            content=task,
            metadata=metadata,
        )

        # Execute
        try:
            response = await asyncio.wait_for(
                self.bus.request(message),
                timeout=timeout
            )

            # Update expert router performance
            if response and hasattr(self, '_expert_router'):
                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                success = response.type != MessageType.ERROR
                confidence = response.metadata.get('confidence', 0.5) if response.metadata else 0.5
                self._expert_router.update_performance(
                    agent_id=agent_id,
                    success=success,
                    confidence=confidence,
                    latency_ms=elapsed_ms
                )

            return response

        except asyncio.TimeoutError:
            # Update router with failure
            if hasattr(self, '_expert_router'):
                self._expert_router.update_performance(
                    agent_id=agent_id,
                    success=False,
                    confidence=0.0,
                    latency_ms=timeout * 1000
                )
            return None

    async def _execute_ensemble(
        self,
        task: str,
        routing_decision: RoutingDecision,
        from_agent: str,
        context: Optional[Dict[str, Any]],
        timeout: float,
        ensemble_strategy: EnsembleStrategy,
        capability_analysis: Optional[CapabilityAnalysis],
    ) -> EnsembleResult:
        """Execute task with multiple agents and aggregate results."""
        start_time = datetime.now()

        # Collect all agents to query
        agent_ids = [routing_decision.selected_agent]
        agent_ids.extend([alt.agent_id for alt in routing_decision.alternatives])

        # Execute in parallel
        tasks = []
        for agent_id in agent_ids:
            task_coro = self._execute_single_agent(
                task=task,
                agent_id=agent_id,
                from_agent=from_agent,
                context=context,
                timeout=timeout,
                routing_decision=routing_decision,
                capability_analysis=capability_analysis,
            )
            tasks.append(asyncio.create_task(task_coro))

        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert to AgentResponse objects
        agent_responses: List[AgentResponse] = []
        for agent_id, response in zip(agent_ids, responses):
            if isinstance(response, Exception):
                continue
            if response is None:
                continue

            # Extract response content
            content = response.content if hasattr(response, 'content') else str(response)
            confidence = response.metadata.get('confidence', 0.5) if hasattr(response, 'metadata') and response.metadata else 0.5
            latency = response.metadata.get('latency_ms', 0.0) if hasattr(response, 'metadata') and response.metadata else 0.0

            agent_responses.append(AgentResponse(
                agent_id=agent_id,
                response=content,
                confidence=confidence,
                latency_ms=latency,
                metadata={
                    "message_id": response.id if hasattr(response, 'id') else None,
                    "message_type": response.type.value if hasattr(response, 'type') else None,
                }
            ))

        # Aggregate results
        if not agent_responses:
            return EnsembleResult(
                final_response=None,
                strategy_used=ensemble_strategy,
                agreement_score=0.0,
                individual_responses=[],
                aggregation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": "No valid responses from agents"}
            )

        result = self._ensemble_aggregator.aggregate(agent_responses, ensemble_strategy)

        # Add ensemble metadata
        result.metadata["routing_decision"] = {
            "primary_agent": routing_decision.selected_agent,
            "total_score": routing_decision.score.total_score,
            "used_thompson": routing_decision.used_thompson,
        }
        if capability_analysis:
            result.metadata["capability_detection"] = {
                "detected": capability_analysis.primary_capability.value,
                "confidence": capability_analysis.confidence,
                "requires_ensemble": capability_analysis.requires_ensemble,
            }

        return result

    def get_expert_router_statistics(self) -> Dict[str, Any]:
        """Get statistics from the expert router."""
        if hasattr(self, '_expert_router'):
            return self._expert_router.get_routing_statistics()
        return {}

    def get_capability_keywords(self) -> Dict[str, List[str]]:
        """Get capability keyword mappings."""
        if hasattr(self, '_capability_analyzer'):
            return {
                k.value: v
                for k, v in self._capability_analyzer.get_capability_keywords().items()
            }
        return {}

    async def broadcast_context(
        self,
        key: str,
        value: Any,
        owner: str,
        namespace: str = "shared"
    ) -> bool:
        """
        Store context and notify all agents.

        Args:
            key: Storage key
            value: Context value
            owner: Agent that created context
            namespace: Memory namespace

        Returns:
            True if stored and broadcasted
        """
        # Store in shared memory
        await self.memory.put(key, value, owner, namespace)

        # Broadcast notification
        notification = AgentMessage(
            from_agent=owner,
            type=MessageType.CONTEXT,
            content={"key": key, "namespace": namespace},
        )

        return await self.bus.send(notification)

    async def handoff_to_agent(
        self,
        from_agent: str,
        to_agent: str,
        task: Any,
        context_namespace: str = "shared",
        strategy: Optional[HandoffStrategy] = None
    ) -> Optional[AgentMessage]:
        """
        Handoff a task to another agent with filtered context.

        Uses MI-based context filtering for efficient handoff.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            task: Task to handoff
            context_namespace: Namespace to get context from
            strategy: Handoff strategy

        Returns:
            Response from target agent
        """
        target = self.registry.get(to_agent)
        if not target:
            return None

        # Determine target capability
        target_capability = (
            list(target.capabilities)[0].value
            if target.capabilities else "general"
        )

        # Get filtered context
        handoff_context = await self.memory.get_context_for_handoff(
            from_agent=from_agent,
            target_agent=to_agent,
            target_capability=target_capability,
            namespace=context_namespace,
            strategy=strategy,
        )

        # Create handoff message with context
        message = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            type=MessageType.HANDOFF,
            content=task,
            context=handoff_context.selected_items,
            metadata={
                "handoff_strategy": handoff_context.strategy.value,
                "context_compression": f"{handoff_context.selected_count}/{handoff_context.original_count}",
                "avg_mi": handoff_context.avg_mi,
            },
        )

        return await self.bus.request(message)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        return {
            "registry": self.registry.get_statistics(),
            "memory": self.memory.get_statistics(),
            "bus": self.bus.get_statistics(),
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_coordinator() -> MultiAgentCoordinator:
    """Create a new multi-agent coordinator."""
    return MultiAgentCoordinator()


def create_agent_message(
    from_agent: str,
    to_agent: str,
    content: Any,
    msg_type: MessageType = MessageType.REQUEST,
    priority: MessagePriority = MessagePriority.NORMAL,
) -> AgentMessage:
    """Convenience function to create agent messages."""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        type=msg_type,
        content=content,
        priority=priority,
    )


def create_research_agent() -> AgentInfo:
    """Create a research agent with standard capabilities."""
    return AgentInfo(
        id=f"research_{uuid.uuid4().hex[:6]}",
        name="Research Agent",
        capabilities={
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.VERIFICATION,
        },
    )


def create_synthesis_agent() -> AgentInfo:
    """Create a synthesis agent with standard capabilities."""
    return AgentInfo(
        id=f"synthesis_{uuid.uuid4().hex[:6]}",
        name="Synthesis Agent",
        capabilities={
            AgentCapability.SUMMARIZATION,
            AgentCapability.SYNTHESIS,
            AgentCapability.GENERATION,
        },
    )


def create_code_agent() -> AgentInfo:
    """Create a code agent with standard capabilities."""
    return AgentInfo(
        id=f"code_{uuid.uuid4().hex[:6]}",
        name="Code Agent",
        capabilities={
            AgentCapability.CODE_REVIEW,
            AgentCapability.CODE_GENERATION,
            AgentCapability.DEBUGGING,
        },
    )
