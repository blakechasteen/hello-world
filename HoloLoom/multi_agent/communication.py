"""
Agent-to-Agent Communication Protocol
======================================

Enables HoloLoom agents to communicate via:
- REST API (cross-process)
- Direct Python calls (same process)
- Redis PubSub (async messaging)

**Date**: November 17, 2025
**Status**: Production Ready

Usage:
    # Initialize communicator
    comm = AgentCommunicator(
        agent_id="math-01",
        transport="rest",  # or "direct", "redis"
        coordinator_url="http://localhost:8000"
    )

    # Send message
    await comm.send_message(
        receiver_id="code-01",
        message_type="query",
        payload={"query": {"text": "Explain Big-O notation"}}
    )

    # Wait for response
    response = await comm.request_response(
        receiver_id="code-01",
        message_type="query",
        payload={"query": {"text": "Explain Big-O notation"}},
        timeout=30.0
    )

    # Broadcast to all agents
    count = await comm.broadcast(
        message_type="knowledge_share",
        payload={"edges": [...]}
    )
"""

import asyncio
import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging

# Try to import aiohttp for REST transport
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Try to import redis for PubSub transport
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Message Types
# ============================================================================

class MessageType(Enum):
    """Types of inter-agent messages."""
    QUERY = "query"                   # Task delegation request
    RESULT = "result"                 # Response to query
    KNOWLEDGE_SHARE = "knowledge_share"  # Yarn Graph update
    CONSENSUS_VOTE = "consensus_vote"    # Voting on decisions
    HEARTBEAT = "heartbeat"           # Agent health check
    LOCK_REQUEST = "lock_request"     # Distributed lock acquisition
    LOCK_RELEASE = "lock_release"     # Distributed lock release
    SYNC_REQUEST = "sync_request"     # Request graph sync
    SYNC_RESPONSE = "sync_response"   # Graph sync data


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AgentMessage:
    """
    Message envelope for inter-agent communication.

    Example:
        message = AgentMessage(
            sender_id="math-01",
            receiver_id="code-01",
            message_type=MessageType.QUERY.value,
            payload={"query": {"text": "Explain recursion"}},
            timestamp=datetime.now(),
            correlation_id="abc-123"
        )
    """
    sender_id: str                          # Unique agent identifier
    receiver_id: str                        # Target agent (or "broadcast")
    message_type: str                       # MessageType enum value
    payload: Dict[str, Any]                 # Message content
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl: int = 300                          # Time-to-live (seconds)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON transport."""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "ttl": self.ttl,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Deserialize from dict."""
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=data["message_type"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data["correlation_id"],
            ttl=data.get("ttl", 300),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# Transport Backends
# ============================================================================

class TransportBackend:
    """Base class for transport backends."""

    async def send(self, message: AgentMessage) -> bool:
        """Send message to receiver."""
        raise NotImplementedError

    async def receive(self, timeout: float = 1.0) -> List[AgentMessage]:
        """Receive pending messages."""
        raise NotImplementedError

    async def close(self):
        """Clean up resources."""
        pass


class DirectTransport(TransportBackend):
    """
    Direct Python transport (in-memory, same process).

    Fastest option but requires agents in same process.
    Uses shared message queues.
    """

    # Shared registry of all agents' message queues
    _agent_queues: Dict[str, asyncio.Queue] = {}

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # Create queue for this agent if not exists
        if agent_id not in DirectTransport._agent_queues:
            DirectTransport._agent_queues[agent_id] = asyncio.Queue()

        self.queue = DirectTransport._agent_queues[agent_id]

    async def send(self, message: AgentMessage) -> bool:
        """Send message to receiver's queue."""
        receiver_id = message.receiver_id

        # Handle broadcast
        if receiver_id == "broadcast":
            for agent_id, queue in DirectTransport._agent_queues.items():
                if agent_id != self.agent_id:  # Don't send to self
                    await queue.put(message)
            return True

        # Send to specific agent
        if receiver_id in DirectTransport._agent_queues:
            await DirectTransport._agent_queues[receiver_id].put(message)
            return True

        logger.warning(f"Agent {receiver_id} not found in direct transport")
        return False

    async def receive(self, timeout: float = 1.0) -> List[AgentMessage]:
        """Receive all pending messages."""
        messages = []

        try:
            while True:
                message = await asyncio.wait_for(self.queue.get(), timeout=0.01)
                messages.append(message)
        except asyncio.TimeoutError:
            pass

        return messages


class RESTTransport(TransportBackend):
    """
    REST API transport (HTTP-based, cross-process).

    Requires each agent to run HTTP server.
    """

    def __init__(self, agent_id: str, coordinator_url: str):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp required for REST transport. Install: pip install aiohttp")

        self.agent_id = agent_id
        self.coordinator_url = coordinator_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.pending_messages: asyncio.Queue = asyncio.Queue()

    async def _ensure_session(self):
        """Lazy session creation."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def send(self, message: AgentMessage) -> bool:
        """Send message via HTTP POST."""
        await self._ensure_session()

        try:
            # Send to coordinator for routing
            async with self.session.post(
                f"{self.coordinator_url}/swarm/message",
                json=message.to_dict(),
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"REST send failed: {e}")
            return False

    async def receive(self, timeout: float = 1.0) -> List[AgentMessage]:
        """Poll coordinator for pending messages."""
        await self._ensure_session()

        try:
            async with self.session.get(
                f"{self.coordinator_url}/swarm/messages/{self.agent_id}",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [AgentMessage.from_dict(msg) for msg in data.get("messages", [])]
        except Exception as e:
            logger.error(f"REST receive failed: {e}")

        return []

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()


class RedisPubSubTransport(TransportBackend):
    """
    Redis PubSub transport (async messaging).

    High throughput, requires Redis server.
    """

    def __init__(self, agent_id: str, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis required for PubSub transport. Install: pip install redis")

        self.agent_id = agent_id
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._listen_task: Optional[asyncio.Task] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def _ensure_connected(self):
        """Connect to Redis if not already connected."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()

            # Subscribe to agent's channel and broadcast channel
            await self.pubsub.subscribe(
                f"agent:{self.agent_id}",
                "agent:broadcast"
            )

            # Start background listener
            self._listen_task = asyncio.create_task(self._listen())

    async def _listen(self):
        """Background task to listen for messages."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    agent_message = AgentMessage.from_dict(data)
                    await self.message_queue.put(agent_message)
        except Exception as e:
            logger.error(f"Redis listen error: {e}")

    async def send(self, message: AgentMessage) -> bool:
        """Publish message to Redis."""
        await self._ensure_connected()

        try:
            # Determine channel
            if message.receiver_id == "broadcast":
                channel = "agent:broadcast"
            else:
                channel = f"agent:{message.receiver_id}"

            # Publish
            await self.redis_client.publish(
                channel,
                json.dumps(message.to_dict())
            )
            return True
        except Exception as e:
            logger.error(f"Redis send failed: {e}")
            return False

    async def receive(self, timeout: float = 1.0) -> List[AgentMessage]:
        """Receive messages from queue."""
        await self._ensure_connected()

        messages = []
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=0.01
                )
                messages.append(message)
            except asyncio.TimeoutError:
                break

        return messages

    async def close(self):
        """Clean up Redis connection."""
        if self._listen_task:
            self._listen_task.cancel()

        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.close()


# ============================================================================
# Agent Communicator
# ============================================================================

class AgentCommunicator:
    """
    Handles inter-agent messaging with pluggable transports.

    Example:
        # Direct transport (same process)
        comm = AgentCommunicator("math-01", transport="direct")

        # REST transport (cross-process)
        comm = AgentCommunicator(
            "math-01",
            transport="rest",
            coordinator_url="http://localhost:8000"
        )

        # Redis PubSub transport
        comm = AgentCommunicator(
            "math-01",
            transport="redis",
            redis_url="redis://localhost:6379"
        )
    """

    def __init__(
        self,
        agent_id: str,
        transport: str = "direct",  # "direct", "rest", "redis"
        coordinator_url: Optional[str] = None,
        redis_url: str = "redis://localhost:6379"
    ):
        """
        Initialize communicator.

        Args:
            agent_id: Unique agent identifier
            transport: Transport backend ("direct", "rest", "redis")
            coordinator_url: URL for coordinator (REST only)
            redis_url: Redis server URL (Redis only)
        """
        self.agent_id = agent_id
        self.transport_type = transport

        # Create transport backend
        if transport == "direct":
            self.transport = DirectTransport(agent_id)
        elif transport == "rest":
            if coordinator_url is None:
                raise ValueError("coordinator_url required for REST transport")
            self.transport = RESTTransport(agent_id, coordinator_url)
        elif transport == "redis":
            self.transport = RedisPubSubTransport(agent_id, redis_url)
        else:
            raise ValueError(f"Unknown transport: {transport}")

        # Response handlers for request-response pattern
        self._response_handlers: Dict[str, asyncio.Future] = {}

        # Background receive task
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start background message receiver."""
        if not self._running:
            self._running = True
            self._receive_task = asyncio.create_task(self._receive_loop())

    async def stop(self):
        """Stop background receiver and clean up."""
        self._running = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        await self.transport.close()

    async def _receive_loop(self):
        """Background task to receive and process messages."""
        while self._running:
            try:
                messages = await self.transport.receive(timeout=1.0)

                for message in messages:
                    # Check if this is a response to a request
                    if message.correlation_id in self._response_handlers:
                        future = self._response_handlers.pop(message.correlation_id)
                        if not future.done():
                            future.set_result(message)

            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                await asyncio.sleep(1.0)

    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        timeout: float = 30.0
    ) -> bool:
        """
        Send message to specific agent.

        Args:
            receiver_id: Target agent ID
            message_type: Message type (use MessageType enum)
            payload: Message content
            correlation_id: For request-response tracking
            timeout: Message timeout (seconds)

        Returns:
            True if sent successfully
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4()),
            ttl=int(timeout)
        )

        return await self.transport.send(message)

    async def broadcast(
        self,
        message_type: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Broadcast message to all agents.

        Args:
            message_type: Message type
            payload: Message content

        Returns:
            True if broadcast successfully
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id="broadcast",
            message_type=message_type,
            payload=payload
        )

        return await self.transport.send(message)

    async def request_response(
        self,
        receiver_id: str,
        message_type: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send message and wait for response.

        Args:
            receiver_id: Target agent ID
            message_type: Message type
            payload: Message content
            timeout: Response timeout (seconds)

        Returns:
            Response message or None if timeout
        """
        # Ensure receiver is running
        if not self._running:
            await self.start()

        # Create correlation ID and future for response
        correlation_id = str(uuid.uuid4())
        response_future = asyncio.Future()
        self._response_handlers[correlation_id] = response_future

        try:
            # Send message
            success = await self.send_message(
                receiver_id=receiver_id,
                message_type=message_type,
                payload=payload,
                correlation_id=correlation_id,
                timeout=timeout
            )

            if not success:
                return None

            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {message_type} to {receiver_id}")
            return None
        finally:
            # Clean up handler
            self._response_handlers.pop(correlation_id, None)

    async def receive_messages(
        self,
        timeout: float = 1.0
    ) -> List[AgentMessage]:
        """
        Receive pending messages (polling mode).

        Args:
            timeout: How long to wait for messages

        Returns:
            List of received messages
        """
        return await self.transport.receive(timeout=timeout)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# ============================================================================
# Agent Registry
# ============================================================================

@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    domain: str                     # "math", "code", "writing", "research"
    capabilities: List[str]         # Specific skills
    endpoint: Optional[str] = None  # HTTP endpoint (for REST)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active: bool = True


class AgentRegistry:
    """
    Registry of all agents in the swarm.

    Maintains agent metadata, endpoints, and health status.
    """

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        agent_id: str,
        domain: str,
        capabilities: List[str],
        endpoint: Optional[str] = None
    ) -> bool:
        """Register agent in registry."""
        async with self._lock:
            self._agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                domain=domain,
                capabilities=capabilities,
                endpoint=endpoint
            )
            logger.info(f"Registered agent: {agent_id} (domain: {domain})")
            return True

    async def unregister(self, agent_id: str) -> bool:
        """Unregister agent from registry."""
        async with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                logger.info(f"Unregistered agent: {agent_id}")
                return True
            return False

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""
        return self._agents.get(agent_id)

    async def list_agents(
        self,
        domain: Optional[str] = None,
        active_only: bool = True
    ) -> List[AgentInfo]:
        """
        List all agents.

        Args:
            domain: Filter by domain (None = all domains)
            active_only: Only return active agents

        Returns:
            List of matching agents
        """
        agents = list(self._agents.values())

        if domain:
            agents = [a for a in agents if a.domain == domain]

        if active_only:
            agents = [a for a in agents if a.active]

        return agents

    async def update_heartbeat(self, agent_id: str) -> bool:
        """Update agent's last heartbeat timestamp."""
        async with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.now()
                return True
            return False

    async def check_health(self, timeout_seconds: int = 60):
        """
        Mark agents as inactive if no heartbeat within timeout.

        Args:
            timeout_seconds: Heartbeat timeout threshold
        """
        now = datetime.now()

        async with self._lock:
            for agent in self._agents.values():
                elapsed = (now - agent.last_heartbeat).total_seconds()
                agent.active = elapsed < timeout_seconds


# Singleton registry
_global_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get global agent registry."""
    return _global_registry
