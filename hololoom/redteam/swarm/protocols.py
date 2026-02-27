"""Protocol definitions and data classes for the multi-agent swarm.

CARTS Phase 4: Swarm Protocols
Status: Foundation Implementation (November 2025)

Defines all protocol interfaces and data classes used by the swarm system:
- AgentRole: Specialized roles (scout, attacker, exploiter, coordinator)
- AgentState: Lifecycle states (idle, active, executing, etc.)
- MessagePriority: Message priority levels (low to critical)
- AgentMessage: Message container with routing and acknowledgment
- AgentTask: Task definition with parameters and timeout
- AgentResult: Execution result with discoveries and metrics
- AgentProtocol: Interface for swarm agents
- CoordinatorProtocol: Interface for swarm coordinator
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set
from datetime import datetime


# ============================================================================
# Enums
# ============================================================================


class AgentRole(Enum):
    """Specialized agent roles in the swarm."""

    SCOUT = "scout"  # Surface probing, vulnerability discovery
    ATTACKER = "attacker"  # Attack execution with Thompson Sampling
    EXPLOITER = "exploiter"  # Vulnerability exploitation and escalation
    COORDINATOR = "coordinator"  # Swarm coordination and task distribution


class AgentState(Enum):
    """Agent lifecycle states."""

    IDLE = "idle"  # Waiting for tasks
    ACTIVE = "active"  # Ready to work
    EXECUTING = "executing"  # Currently executing a task
    WAITING = "waiting"  # Waiting for dependencies or resources
    FAILED = "failed"  # Task execution failed
    COMPLETED = "completed"  # Task execution succeeded
    SHUTDOWN = "shutdown"  # Agent shutting down


class MessagePriority(Enum):
    """Message priority levels for queue ordering.

    Used to ensure critical messages (failures, discoveries) are processed
    before routine status updates.
    """

    LOW = 1  # Background status updates
    NORMAL = 2  # Regular tasks and results
    HIGH = 3  # Important discoveries, phase transitions
    CRITICAL = 4  # Failures, security events, coordinator commands


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AgentMessage:
    """Message container for inter-agent communication.

    Features:
    - Unique message ID and correlation for request-response patterns
    - Priority-based queuing (CRITICAL/HIGH/NORMAL/LOW)
    - Optional acknowledgment requirement
    - Type-based routing (task, result, status, discovery)
    - Complete timestamp tracking for latency metrics

    Attributes:
        id: Unique message identifier (UUID)
        sender: Sending agent ID
        recipient: Receiving agent ID or "*" for broadcast
        message_type: Type of message (task, result, status, discovery, command)
        payload: Message content (dict for flexibility)
        priority: MessagePriority for queue ordering
        timestamp: When message was created (Unix timestamp)
        requires_ack: If True, sender expects acknowledgment
        correlation_id: For request-response pattern matching
    """

    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    requires_ack: bool = False
    correlation_id: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        """Time elapsed since message creation."""
        return time.time() - self.timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to dict."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "payload": self.payload,
            "priority": self.priority.name,
            "timestamp": self.timestamp,
            "requires_ack": self.requires_ack,
            "correlation_id": self.correlation_id,
        }


@dataclass
class AgentTask:
    """Task definition for swarm agents.

    Features:
    - Unique task ID for tracking
    - Type-based task routing (probe_surface, execute_attack, exploit_vulnerability)
    - Priority-based execution ordering
    - Timeout for preventing resource exhaustion
    - Optional assignment to specific agent

    Attributes:
        task_id: Unique task identifier (UUID)
        task_type: Type of task (probe_surface, execute_attack, exploit_vulnerability)
        target: Target host/service/endpoint
        parameters: Task-specific parameters (dict)
        priority: MessagePriority for scheduling
        timeout_seconds: Maximum execution time
        assigned_agent: If set, task assigned to specific agent
    """

    task_type: str
    target: str
    parameters: Dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_seconds: float = 30.0
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        """Time elapsed since task creation."""
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dict."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "target": self.target,
            "parameters": self.parameters,
            "priority": self.priority.name,
            "timeout_seconds": self.timeout_seconds,
            "assigned_agent": self.assigned_agent,
            "created_at": self.created_at,
        }


@dataclass
class AgentResult:
    """Execution result from task completion.

    Features:
    - Complete execution context (task ID, agent ID)
    - Success/failure tracking with error messages
    - Execution timing for performance analysis
    - Discoveries list for vulnerability findings
    - Standardized format for result aggregation

    Attributes:
        task_id: ID of completed task
        agent_id: ID of executing agent
        success: Whether task succeeded
        result: Task-specific result data
        error: Error message if failed
        execution_time_ms: Wall-clock execution time
        discoveries: List of discovered vulnerabilities/info
    """

    task_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    completed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dict."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "discoveries": self.discoveries,
            "completed_at": self.completed_at,
        }


# ============================================================================
# Protocols
# ============================================================================


class AgentProtocol(Protocol):
    """Protocol for swarm agents.

    All swarm agents must implement this protocol. This enables:
    - Type-safe agent implementation
    - Flexible agent creation (any class can be an agent)
    - Clean separation of concerns (agents don't depend on coordinator)

    Methods:
        start: Initialize agent and connect to message bus
        stop: Clean shutdown with pending task handling
        handle_message: Process incoming message from bus
        execute_task: Execute a task and return result

    Properties:
        agent_id: Unique agent identifier
        role: AgentRole (scout, attacker, exploiter, coordinator)
        state: Current AgentState
    """

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def role(self) -> AgentRole:
        """Agent's specialized role in the swarm."""
        ...

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        ...

    async def start(self) -> None:
        """Initialize agent and connect to message bus.

        Should:
        - Initialize internal state
        - Connect to message bus
        - Enter ACTIVE state
        - Start listening for messages (if needed)
        """
        ...

    async def stop(self) -> None:
        """Clean shutdown with pending task handling.

        Should:
        - Complete or abandon in-flight tasks
        - Disconnect from message bus
        - Clean up resources
        - Enter SHUTDOWN state
        """
        ...

    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message from bus.

        Called by message bus when a message arrives. Should:
        - Validate message is for this agent
        - Route based on message_type
        - Return acknowledgment if message.requires_ack=True
        - Return None if no response needed

        Args:
            message: Incoming message

        Returns:
            Optional response message (for acknowledgment or error)
        """
        ...

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a task and return result.

        Should:
        - Validate task parameters
        - Enter EXECUTING state
        - Execute task logic
        - Return AgentResult with success/failure

        Args:
            task: Task to execute

        Returns:
            AgentResult with execution outcome
        """
        ...


class CoordinatorProtocol(Protocol):
    """Protocol for swarm coordinator.

    The coordinator is responsible for:
    - Task distribution and scheduling
    - Result aggregation
    - Agent health monitoring
    - Inter-agent communication facilitation

    Methods:
        distribute_task: Assign task to appropriate agent
        aggregate_results: Collect results from multiple agents
        broadcast: Send message to all agents
        get_agent_states: Query current agent states
    """

    async def distribute_task(self, task: AgentTask) -> str:
        """Distribute task to appropriate agent.

        Should:
        - Select agent based on task type and current load
        - Update task.assigned_agent
        - Return assigned agent_id
        - Use Thompson Sampling for strategy selection

        Args:
            task: Task to distribute

        Returns:
            ID of agent assigned to task
        """
        ...

    async def aggregate_results(self, task_id: str) -> List[AgentResult]:
        """Collect results from agents for a task.

        For tasks that may be executed by multiple agents (reconnaissance),
        aggregates all results into a single list.

        Args:
            task_id: Task ID to aggregate results for

        Returns:
            List of all results for this task
        """
        ...

    async def broadcast(self, message: AgentMessage) -> int:
        """Send message to all agents.

        Used for coordination messages (phase changes, alerts, etc).

        Args:
            message: Message to broadcast

        Returns:
            Number of agents that received the message
        """
        ...

    def get_agent_states(self) -> Dict[str, AgentState]:
        """Get current state of all agents.

        Returns:
            Dict mapping agent_id -> AgentState
        """
        ...
