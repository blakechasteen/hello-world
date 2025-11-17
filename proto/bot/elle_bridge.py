"""
Proto Elle Bridge: Integrate Elle AR observations into Proto

This module provides the Proto side of the Elle ↔ Proto bridge,
enabling Proto to:
- Receive Elle observation notifications
- Query Elle's observation history
- Display formatted observations in Matrix
- Track deferred tasks and summaries

Architecture:
    Elle (AR) → HoloLoom (Shared Memory) → Proto (Matrix Bot)
    Proto → HoloLoom → Query Elle Observations

Created: 2025-11-17 (Week 4: Elle AR Bridge Integration)
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import logging
import asyncio

try:
    from nio import AsyncClient, MatrixRoom
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Import Elle bridge components
import sys
from pathlib import Path

# Add elle to path
elle_path = Path(__file__).parent.parent.parent / "elle"
sys.path.insert(0, str(elle_path))

from elle.memory.proto_bridge import (
    ElleObservation,
    ElleProtoMemoryBridge,
    create_bridge
)
from elle.adapters.matrix_adapter import MatrixFormatter


class ProtoElleBridge:
    """
    Proto side of the Elle ↔ Proto bridge.

    Responsibilities:
    - Listen for Elle observation notifications
    - Query Elle observation history via HoloLoom
    - Format and display observations in Matrix
    - Provide command handlers for Proto bot

    Usage:
        bridge = ProtoElleBridge(matrix_client)
        await bridge.start()

        # Handle commands
        await bridge.handle_command("@proto workshop-summary", room_id)
    """

    def __init__(
        self,
        matrix_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Proto Elle bridge.

        Args:
            matrix_client: Matrix nio client (optional)
            logger: Logger instance
        """
        self.matrix_client = matrix_client
        self.logger = logger or logging.getLogger(__name__)

        # Memory bridge (initialized on start)
        self.memory_bridge: Optional[ElleProtoMemoryBridge] = None

        # Formatter
        self.formatter = MatrixFormatter()

        # Notification room (where to post Elle observations)
        self.notification_room: Optional[str] = None

    async def start(self, notification_room: Optional[str] = None):
        """
        Start the bridge.

        Args:
            notification_room: Matrix room ID for Elle notifications (optional)
        """
        self.logger.info("Starting Proto Elle bridge")

        # Store notification room
        self.notification_room = notification_room

        # Initialize memory bridge
        self.memory_bridge = ElleProtoMemoryBridge(
            notification_handler=self._handle_notification
        )
        await self.memory_bridge.__aenter__()

        self.logger.info("Proto Elle bridge started")

    async def stop(self):
        """Stop the bridge."""
        self.logger.info("Stopping Proto Elle bridge")

        if self.memory_bridge:
            await self.memory_bridge.__aexit__(None, None, None)

        self.logger.info("Proto Elle bridge stopped")

    # =========================================================================
    # Notification Handler
    # =========================================================================

    async def _handle_notification(self, observation: ElleObservation):
        """
        Handle Elle observation notification.

        This is called when Elle stores a new observation in HoloLoom.

        Args:
            observation: New ElleObservation
        """
        self.logger.info(
            f"Received Elle notification: {observation.observation_id} "
            f"({observation.location})"
        )

        # Post to notification room if configured
        if self.notification_room and self.matrix_client:
            await self._post_observation_to_matrix(
                observation,
                self.notification_room
            )

    async def _post_observation_to_matrix(
        self,
        observation: ElleObservation,
        room_id: str
    ):
        """Post observation to Matrix room."""
        try:
            # Format observation
            message = self.formatter.format_observation(observation)

            # Send to Matrix
            if MATRIX_AVAILABLE and self.matrix_client:
                await self.matrix_client.room_send(
                    room_id=room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": message,
                    }
                )

                self.logger.info(f"Posted observation to {room_id}")
            else:
                # Fallback: just log
                self.logger.info(f"Observation (no Matrix):\n{message}")

        except Exception as e:
            self.logger.error(f"Error posting observation: {e}")

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def handle_command(
        self,
        command: str,
        room_id: str
    ) -> Optional[str]:
        """
        Handle Elle-related command from Proto.

        Supported commands:
        - @proto workshop-summary
        - @proto elle-status
        - @proto ask-elle <question>
        - @proto recent-observations [location]
        - @proto deferred-tasks [location]

        Args:
            command: Command text
            room_id: Matrix room ID

        Returns:
            Response message (or None if command not recognized)

        Example:
            >>> response = await bridge.handle_command(
            ...     "@proto workshop-summary",
            ...     room_id
            ... )
        """
        if not self.memory_bridge:
            return "❌ Elle bridge not initialized"

        # Parse command
        parts = command.lower().split()

        # Remove @proto prefix
        if parts and parts[0] == "@proto":
            parts = parts[1:]

        if not parts:
            return None

        try:
            # Workshop summary
            if "workshop-summary" in parts or (parts[0] == "workshop" and "summary" in parts):
                return await self._handle_workshop_summary(room_id)

            # Elle status
            elif "elle-status" in parts or (parts[0] == "elle" and "status" in parts):
                return await self._handle_elle_status(room_id)

            # Ask Elle (query observations)
            elif parts[0] == "ask-elle":
                question = " ".join(parts[1:])
                return await self._handle_ask_elle(question, room_id)

            # Recent observations
            elif "recent" in parts and "observations" in parts:
                location = self._extract_location(parts)
                return await self._handle_recent_observations(location, room_id)

            # Deferred tasks
            elif "deferred" in parts or "tasks" in parts:
                location = self._extract_location(parts)
                return await self._handle_deferred_tasks(location, room_id)

            # Not an Elle command
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error handling command: {e}")
            return f"❌ Error: {str(e)}"

    async def _handle_workshop_summary(self, room_id: str) -> str:
        """Handle workshop-summary command."""
        summary = await self.memory_bridge.get_location_summary("workshop", 24)
        return self.formatter.format_summary(summary)

    async def _handle_elle_status(self, room_id: str) -> str:
        """Handle elle-status command."""
        health = await self.memory_bridge.health_check()

        lines = [
            "### 🔗 Elle Bridge Status",
            "",
            f"**Status**: {health['status']}",
        ]

        if 'loom' in health and health['loom']:
            lines.append(f"**Memories**: {health['loom'].get('memories', 'N/A')}")
            lines.append(f"**Active**: {health['loom'].get('active', 'N/A')}")

        lines.append(
            f"**Notifications**: {'✅ Enabled' if health.get('notification_handler_registered') else '❌ Disabled'}"
        )

        return "\n".join(lines)

    async def _handle_ask_elle(self, question: str, room_id: str) -> str:
        """Handle ask-elle command."""
        if not question:
            return "ℹ️ Usage: @proto ask-elle <question>\nExample: @proto ask-elle What did we accomplish in the workshop?"

        # Query observations based on question
        observations = await self.memory_bridge.query_observations(
            query_type="recent",
            limit=5
        )

        if not observations:
            return "Elle hasn't observed anything recently."

        # Format observations
        lines = [
            f"### 💭 Elle's Response to: \"{question}\"",
            "",
        ]

        for obs in observations[:3]:  # Top 3 most relevant
            time_str = obs.timestamp.strftime("%I:%M %p")
            lines.append(f"**{obs.location.title()} ({time_str})**:")
            lines.append(f"  {obs.scene_summary}")
            if obs.action_taken:
                lines.append(f"  _{obs.action_taken}_")
            lines.append("")

        return "\n".join(lines)

    async def _handle_recent_observations(
        self,
        location: Optional[str],
        room_id: str
    ) -> str:
        """Handle recent-observations command."""
        observations = await self.memory_bridge.query_observations(
            query_type="location" if location else "recent",
            location=location,
            limit=5
        )

        if not observations:
            loc_str = f" in {location}" if location else ""
            return f"No recent observations{loc_str}."

        # Format observations
        lines = ["### 📋 Recent Observations", ""]

        for obs in observations:
            time_str = obs.timestamp.strftime("%I:%M %p")
            lines.append(f"**{obs.location.title()} ({time_str})**:")
            lines.append(f"  {obs.scene_summary}")

            if obs.action_taken:
                lines.append(f"  _{obs.action_taken}_")

            if obs.deferred_tasks:
                lines.append(f"  ⏸️ {len(obs.deferred_tasks)} deferred task(s)")

            lines.append("")

        return "\n".join(lines)

    async def _handle_deferred_tasks(
        self,
        location: Optional[str],
        room_id: str
    ) -> str:
        """Handle deferred-tasks command."""
        tasks = await self.memory_bridge.get_deferred_tasks(location)

        if not tasks:
            loc_str = f" in {location}" if location else ""
            return f"No deferred tasks{loc_str}."

        # Format tasks
        lines = ["### ⏸️ Deferred Tasks", ""]

        for task in tasks[:10]:  # Limit to 10
            lines.append(f"**{task['description']}**")
            lines.append(f"  • Location: {task['location']}")
            lines.append(f"  • Reason: {task['reason']}")
            lines.append(f"  • Observed: {task['observed_at'].strftime('%I:%M %p')}")

            if 'elle_note' in task:
                lines.append(f"  • 💭 Elle note: {task['elle_note']}")

            lines.append("")

        return "\n".join(lines)

    def _extract_location(self, parts: List[str]) -> Optional[str]:
        """Extract location from command parts."""
        locations = ["workshop", "garden", "shed", "kitchen", "office"]
        for part in parts:
            if part in locations:
                return part
        return None

    # =========================================================================
    # Quick Helpers
    # =========================================================================

    async def get_recent_observations(
        self,
        location: Optional[str] = None,
        limit: int = 5
    ) -> List[ElleObservation]:
        """
        Get recent observations.

        Args:
            location: Filter by location (optional)
            limit: Maximum observations

        Returns:
            List of ElleObservation objects
        """
        if not self.memory_bridge:
            return []

        return await self.memory_bridge.query_observations(
            query_type="location" if location else "recent",
            location=location,
            limit=limit
        )

    async def get_location_summary(
        self,
        location: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary of activity at location.

        Args:
            location: Location to summarize
            hours: Look back this many hours

        Returns:
            Summary dict
        """
        if not self.memory_bridge:
            return {}

        return await self.memory_bridge.get_location_summary(location, hours)

    async def get_deferred_tasks(
        self,
        location: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all deferred tasks.

        Args:
            location: Filter by location (optional)

        Returns:
            List of task dicts
        """
        if not self.memory_bridge:
            return []

        return await self.memory_bridge.get_deferred_tasks(location)


# =============================================================================
# Integration with Proto Bot
# =============================================================================


async def integrate_elle_bridge_with_proto(
    proto_bot: Any,
    notification_room: Optional[str] = None
):
    """
    Integrate Elle bridge into Proto bot.

    Args:
        proto_bot: Proto bot instance
        notification_room: Matrix room ID for Elle notifications

    Example:
        >>> # In Proto bot initialization
        >>> await integrate_elle_bridge_with_proto(
        ...     proto_bot,
        ...     notification_room="!abc:matrix.org"
        ... )
    """
    # Create bridge
    bridge = ProtoElleBridge(
        matrix_client=proto_bot.client if hasattr(proto_bot, 'client') else None
    )

    # Start bridge
    await bridge.start(notification_room=notification_room)

    # Register command handler with Proto
    if hasattr(proto_bot, 'register_handler'):
        proto_bot.register_handler('elle', bridge.handle_command)

    return bridge


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate Proto Elle bridge."""

        # Create bridge
        bridge = ProtoElleBridge()

        try:
            await bridge.start()

            print("🔗 Bridge started")

            # Simulate commands
            commands = [
                "@proto workshop-summary",
                "@proto elle-status",
                "@proto ask-elle What did we accomplish?",
                "@proto recent-observations workshop",
                "@proto deferred-tasks",
            ]

            for cmd in commands:
                print(f"\n💬 Command: {cmd}")
                response = await bridge.handle_command(cmd, "!demo:matrix.org")
                if response:
                    print(response)
                else:
                    print("(Not an Elle command)")

        finally:
            await bridge.stop()

    # Run demo
    asyncio.run(demo())
