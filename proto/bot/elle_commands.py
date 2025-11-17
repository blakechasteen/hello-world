"""
Proto Elle Commands: Command handlers for Elle AR integration

This module provides Proto command handlers for interacting with Elle's
AR observations through the shared HoloLoom bridge.

Commands:
    @proto workshop-summary       - Get recent workshop activity summary
    @proto elle-status           - Check Elle bridge status
    @proto ask-elle <question>   - Query Elle about physical space
    @proto recent-observations [location] - Get recent Elle observations
    @proto deferred-tasks [location]      - Get deferred tasks from Elle

Created: 2025-11-17 (Week 4: Elle AR Bridge Integration)
"""

from typing import Optional, Dict, Any
import logging

from .elle_bridge import ProtoElleBridge


class ElleCommandHandler:
    """
    Command handler for Elle-related Proto commands.

    This provides a clean interface for Proto bot to handle Elle commands.
    Integrates with ProtoElleBridge for actual functionality.
    """

    def __init__(self, bridge: ProtoElleBridge):
        """
        Initialize command handler.

        Args:
            bridge: ProtoElleBridge instance
        """
        self.bridge = bridge
        self.logger = logging.getLogger(__name__)

    async def handle(self, command: str, room_id: str) -> Optional[str]:
        """
        Handle Elle command.

        Args:
            command: Full command text (e.g., "@proto workshop-summary")
            room_id: Matrix room ID

        Returns:
            Response message (or None if not an Elle command)
        """
        return await self.bridge.handle_command(command, room_id)

    # =========================================================================
    # Individual Command Handlers (for direct integration)
    # =========================================================================

    async def workshop_summary(self, room_id: str) -> str:
        """
        Get workshop activity summary.

        Command: @proto workshop-summary

        Returns formatted summary of last 24 hours of workshop activity.
        """
        return await self.bridge.handle_command("@proto workshop-summary", room_id)

    async def elle_status(self, room_id: str) -> str:
        """
        Get Elle bridge status.

        Command: @proto elle-status

        Returns bridge health and connection status.
        """
        return await self.bridge.handle_command("@proto elle-status", room_id)

    async def ask_elle(self, question: str, room_id: str) -> str:
        """
        Ask Elle a question about physical space.

        Command: @proto ask-elle <question>

        Args:
            question: Question to ask Elle
            room_id: Matrix room ID

        Returns:
            Elle's response based on recent observations

        Example:
            >>> response = await handler.ask_elle(
            ...     "What did we accomplish in the workshop?",
            ...     room_id
            ... )
        """
        return await self.bridge.handle_command(
            f"@proto ask-elle {question}",
            room_id
        )

    async def recent_observations(
        self,
        location: Optional[str] = None,
        room_id: str = None
    ) -> str:
        """
        Get recent Elle observations.

        Command: @proto recent-observations [location]

        Args:
            location: Filter by location (optional)
            room_id: Matrix room ID

        Returns:
            Formatted list of recent observations
        """
        cmd = "@proto recent-observations"
        if location:
            cmd += f" {location}"

        return await self.bridge.handle_command(cmd, room_id)

    async def deferred_tasks(
        self,
        location: Optional[str] = None,
        room_id: str = None
    ) -> str:
        """
        Get deferred tasks from Elle observations.

        Command: @proto deferred-tasks [location]

        Args:
            location: Filter by location (optional)
            room_id: Matrix room ID

        Returns:
            Formatted list of deferred tasks
        """
        cmd = "@proto deferred-tasks"
        if location:
            cmd += f" {location}"

        return await self.bridge.handle_command(cmd, room_id)


# =============================================================================
# Command Registration Helpers
# =============================================================================


ELLE_COMMANDS = {
    "workshop-summary": {
        "description": "Get recent workshop activity summary",
        "usage": "@proto workshop-summary",
        "category": "Elle AR",
    },
    "elle-status": {
        "description": "Check Elle bridge status",
        "usage": "@proto elle-status",
        "category": "Elle AR",
    },
    "ask-elle": {
        "description": "Ask Elle about physical space",
        "usage": "@proto ask-elle <question>",
        "category": "Elle AR",
        "examples": [
            "@proto ask-elle What did we accomplish today?",
            "@proto ask-elle What's the status of the workshop?",
        ],
    },
    "recent-observations": {
        "description": "Get recent Elle observations",
        "usage": "@proto recent-observations [location]",
        "category": "Elle AR",
        "examples": [
            "@proto recent-observations",
            "@proto recent-observations workshop",
        ],
    },
    "deferred-tasks": {
        "description": "Get deferred tasks from Elle",
        "usage": "@proto deferred-tasks [location]",
        "category": "Elle AR",
        "examples": [
            "@proto deferred-tasks",
            "@proto deferred-tasks workshop",
        ],
    },
}


def get_elle_help() -> str:
    """
    Get help text for Elle commands.

    Returns:
        Formatted help text
    """
    lines = [
        "### 👁️ Elle AR Commands",
        "",
        "**Elle Integration**: Query Elle's AR observations from physical spaces",
        "",
    ]

    for cmd, info in ELLE_COMMANDS.items():
        lines.append(f"**{info['usage']}**")
        lines.append(f"  {info['description']}")

        if 'examples' in info:
            lines.append("  Examples:")
            for example in info['examples']:
                lines.append(f"    • `{example}`")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Integration with Proto Command Parser
# =============================================================================


def register_elle_commands(proto_bot: Any, bridge: ProtoElleBridge):
    """
    Register Elle commands with Proto bot.

    Args:
        proto_bot: Proto bot instance
        bridge: ProtoElleBridge instance

    Example:
        >>> bridge = ProtoElleBridge()
        >>> await bridge.start()
        >>> register_elle_commands(proto_bot, bridge)
    """
    handler = ElleCommandHandler(bridge)

    # Register with Proto's command parser
    # (Assuming Proto has a command registry)
    if hasattr(proto_bot, 'register_command'):
        for cmd_name, cmd_info in ELLE_COMMANDS.items():
            proto_bot.register_command(
                name=cmd_name,
                handler=handler.handle,
                description=cmd_info['description'],
                usage=cmd_info['usage'],
                category=cmd_info.get('category', 'Elle AR'),
            )

    # Or register as a group handler
    elif hasattr(proto_bot, 'register_handler'):
        proto_bot.register_handler('elle', handler.handle)


# =============================================================================
# Example: Full Integration
# =============================================================================


async def setup_elle_integration(proto_bot: Any, notification_room: Optional[str] = None):
    """
    Complete setup of Elle integration for Proto.

    This is the one-line setup for Proto to integrate Elle AR observations.

    Args:
        proto_bot: Proto bot instance
        notification_room: Matrix room ID for Elle notifications (optional)

    Returns:
        Tuple of (bridge, handler) for further customization

    Example:
        >>> # In Proto bot initialization
        >>> bridge, handler = await setup_elle_integration(
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

    # Create handler
    handler = ElleCommandHandler(bridge)

    # Register commands
    register_elle_commands(proto_bot, bridge)

    return bridge, handler


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate Elle command handlers."""

        # Create bridge
        bridge = ProtoElleBridge()
        await bridge.start()

        # Create handler
        handler = ElleCommandHandler(bridge)

        # Test commands
        print("📋 Testing Elle commands...\n")

        # Workshop summary
        print("1️⃣ Workshop Summary:")
        response = await handler.workshop_summary("!demo:matrix.org")
        print(response)
        print()

        # Elle status
        print("2️⃣ Elle Status:")
        response = await handler.elle_status("!demo:matrix.org")
        print(response)
        print()

        # Ask Elle
        print("3️⃣ Ask Elle:")
        response = await handler.ask_elle(
            "What did we accomplish in the workshop?",
            "!demo:matrix.org"
        )
        print(response)
        print()

        # Help
        print("4️⃣ Help Text:")
        print(get_elle_help())

        await bridge.stop()

    # Run demo
    asyncio.run(demo())
