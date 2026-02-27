#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Agent Room Abstraction
=============================

Part of Phase 4: Open & Safe Agent Federation (December 2025)
Day 17: Matrix Transport Layer

Room-level abstraction for multi-agent collaboration over Matrix.
Each room represents a collaboration space where agents can:
- Announce their capabilities
- Send/receive JSON-RPC requests
- Query room capabilities

Author: Claude Code (Phase 4 - December 2025)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from hololoom.federation.types import Capability


# ============================================================================
#  Constants
# ============================================================================

# State event type for capabilities announcement
HOLOLOOM_CAPABILITIES_STATE = "m.hololoom.capabilities"

# Custom message type for RPC
HOLOLOOM_RPC_MSGTYPE = "m.hololoom.rpc"


# ============================================================================
#  Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Announced capabilities of an agent in a room."""

    node_id: str
    matrix_user: str
    capabilities: Set[Capability]
    models: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    announced_at: datetime = field(default_factory=datetime.utcnow)

    def to_state_content(self) -> Dict[str, Any]:
        """Convert to Matrix state event content."""
        return {
            "node_id": self.node_id,
            "capabilities": [c.name for c in self.capabilities],
            "models": list(self.models),
            "version": self.version,
            "announced_at": self.announced_at.isoformat(),
        }

    @classmethod
    def from_state_content(
        cls,
        matrix_user: str,
        content: Dict[str, Any],
    ) -> "AgentCapabilities":
        """Parse from Matrix state event content."""
        capabilities = set()
        for cap_name in content.get("capabilities", []):
            try:
                capabilities.add(Capability[cap_name])
            except KeyError:
                pass  # Unknown capability, skip

        return cls(
            node_id=content.get("node_id", ""),
            matrix_user=matrix_user,
            capabilities=capabilities,
            models=set(content.get("models", [])),
            version=content.get("version", "1.0.0"),
        )


@dataclass
class RoomCapabilitySummary:
    """Summary of all agent capabilities in a room."""

    room_id: str
    agents: List[AgentCapabilities]
    total_agents: int
    available_capabilities: Set[Capability]
    available_models: Set[str]

    def has_capability(self, capability: Capability) -> bool:
        """Check if any agent has this capability."""
        return capability in self.available_capabilities

    def agents_with_capability(self, capability: Capability) -> List[str]:
        """Get node_ids of agents with this capability."""
        return [
            agent.node_id
            for agent in self.agents
            if capability in agent.capabilities
        ]

    def agents_with_model(self, model: str) -> List[str]:
        """Get node_ids of agents with this model."""
        return [
            agent.node_id
            for agent in self.agents
            if model in agent.models
        ]


# ============================================================================
#  Matrix Agent Room
# ============================================================================

class MatrixAgentRoom:
    """
    Room abstraction for multi-agent collaboration.

    Provides high-level operations for agent-to-agent communication:
    - Join/leave rooms with capability announcements
    - Send JSON-RPC requests to room
    - Query room capabilities

    Example:
        >>> room = MatrixAgentRoom(
        ...     room_id="!agents:example.org",
        ...     client=matrix_client,
        ...     node_id="abc123",
        ...     our_capabilities={Capability.RAG, Capability.WEAVING}
        ... )
        >>> await room.join()
        >>> summary = await room.get_capabilities()
        >>> await room.send_rpc("hololoom.recall", {"query": "..."})
        >>> await room.leave()
    """

    def __init__(
        self,
        room_id: str,
        client: Any,  # AsyncClient from matrix-nio
        node_id: str,
        our_capabilities: Set[Capability],
        our_models: Optional[Set[str]] = None,
        version: str = "1.0.0",
        on_rpc_request: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]] = None,
    ):
        """
        Initialize room abstraction.

        Args:
            room_id: Matrix room ID (e.g., "!room:server.org")
            client: matrix-nio AsyncClient instance
            node_id: Our federation node ID
            our_capabilities: Our agent's capabilities to announce
            our_models: Our agent's available models (optional)
            version: Our agent version
            on_rpc_request: Callback for incoming RPC requests
        """
        self._room_id = room_id
        self._client = client
        self._node_id = node_id
        self._our_capabilities = our_capabilities
        self._our_models = our_models or set()
        self._version = version
        self._on_rpc_request = on_rpc_request
        self._joined = False

    @property
    def room_id(self) -> str:
        """Get room ID."""
        return self._room_id

    @property
    def joined(self) -> bool:
        """Check if we've joined the room."""
        return self._joined

    async def join(self) -> bool:
        """
        Join room and announce capabilities.

        Returns:
            True if join succeeded and capabilities announced
        """
        try:
            # Join the room
            response = await self._client.join(self._room_id)

            # Check if join succeeded
            if hasattr(response, "room_id"):
                self._joined = True

                # Announce our capabilities via state event
                await self._announce_capabilities()
                return True

            return False

        except Exception:
            return False

    async def leave(self) -> bool:
        """
        Leave room gracefully.

        Returns:
            True if leave succeeded
        """
        if not self._joined:
            return True

        try:
            # Clear our capabilities (optional - they'll be stale anyway)
            await self._client.room_leave(self._room_id)
            self._joined = False
            return True

        except Exception:
            return False

    async def _announce_capabilities(self) -> None:
        """Announce our capabilities via state event."""
        capabilities = AgentCapabilities(
            node_id=self._node_id,
            matrix_user=self._client.user_id,
            capabilities=self._our_capabilities,
            models=self._our_models,
            version=self._version,
        )

        # Send state event with our user_id as state_key
        await self._client.room_put_state(
            room_id=self._room_id,
            event_type=HOLOLOOM_CAPABILITIES_STATE,
            content=capabilities.to_state_content(),
            state_key=self._client.user_id,
        )

    async def send_rpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Send JSON-RPC request to room.

        Args:
            method: RPC method name (e.g., "hololoom.recall")
            params: Method parameters
            request_id: Optional request ID (generated if not provided)

        Returns:
            Request ID for tracking response
        """
        if not self._joined:
            raise RuntimeError("Must join room before sending RPC")

        request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        # Build JSON-RPC 2.0 message
        rpc_message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id,
        }

        # Send as custom message type
        await self._client.room_send(
            room_id=self._room_id,
            message_type="m.room.message",
            content={
                "msgtype": HOLOLOOM_RPC_MSGTYPE,
                "body": json.dumps(rpc_message),
                "rpc": rpc_message,
            },
        )

        return request_id

    async def send_rpc_response(
        self,
        request_id: str,
        result: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send JSON-RPC response.

        Args:
            request_id: Original request ID
            result: Success result (mutually exclusive with error)
            error: Error object (mutually exclusive with result)
        """
        if not self._joined:
            raise RuntimeError("Must join room before sending response")

        # Build JSON-RPC 2.0 response
        rpc_response: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
        }

        if error is not None:
            rpc_response["error"] = error
        else:
            rpc_response["result"] = result

        # Send as custom message type
        await self._client.room_send(
            room_id=self._room_id,
            message_type="m.room.message",
            content={
                "msgtype": HOLOLOOM_RPC_MSGTYPE,
                "body": json.dumps(rpc_response),
                "rpc": rpc_response,
            },
        )

    async def get_capabilities(self) -> RoomCapabilitySummary:
        """
        Query capabilities of all agents in room.

        Returns:
            Summary of room capabilities
        """
        agents: List[AgentCapabilities] = []
        all_capabilities: Set[Capability] = set()
        all_models: Set[str] = set()

        try:
            # Get all capability state events
            response = await self._client.room_get_state(self._room_id)

            if hasattr(response, "events"):
                for event in response.events:
                    if event.get("type") == HOLOLOOM_CAPABILITIES_STATE:
                        state_key = event.get("state_key", "")
                        content = event.get("content", {})

                        agent = AgentCapabilities.from_state_content(
                            matrix_user=state_key,
                            content=content,
                        )

                        agents.append(agent)
                        all_capabilities.update(agent.capabilities)
                        all_models.update(agent.models)

        except Exception:
            pass  # Return empty summary on error

        return RoomCapabilitySummary(
            room_id=self._room_id,
            agents=agents,
            total_agents=len(agents),
            available_capabilities=all_capabilities,
            available_models=all_models,
        )

    async def find_agents_for_task(
        self,
        required_capabilities: Set[Capability],
        preferred_models: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Find agents that can handle a task.

        Args:
            required_capabilities: Capabilities needed for task
            preferred_models: Preferred models (optional filter)

        Returns:
            List of node_ids of suitable agents
        """
        summary = await self.get_capabilities()

        suitable = []
        for agent in summary.agents:
            # Must have all required capabilities
            if not required_capabilities.issubset(agent.capabilities):
                continue

            # If preferred models specified, prefer agents with them
            if preferred_models:
                if agent.models.intersection(preferred_models):
                    suitable.insert(0, agent.node_id)  # Preferred first
                else:
                    suitable.append(agent.node_id)
            else:
                suitable.append(agent.node_id)

        return suitable


# ============================================================================
#  Factory Function
# ============================================================================

def create_agent_room(
    room_id: str,
    client: Any,
    node_id: str,
    capabilities: Set[Capability],
    models: Optional[Set[str]] = None,
    version: str = "1.0.0",
) -> MatrixAgentRoom:
    """
    Create a MatrixAgentRoom instance.

    Args:
        room_id: Matrix room ID
        client: matrix-nio AsyncClient
        node_id: Our federation node ID
        capabilities: Our capabilities to announce
        models: Our available models
        version: Our agent version

    Returns:
        Configured MatrixAgentRoom
    """
    return MatrixAgentRoom(
        room_id=room_id,
        client=client,
        node_id=node_id,
        our_capabilities=capabilities,
        our_models=models,
        version=version,
    )


# ============================================================================
#  Exports
# ============================================================================

__all__ = [
    # Data classes
    "AgentCapabilities",
    "RoomCapabilitySummary",
    # Main class
    "MatrixAgentRoom",
    # Factory
    "create_agent_room",
    # Constants
    "HOLOLOOM_CAPABILITIES_STATE",
    "HOLOLOOM_RPC_MSGTYPE",
]
