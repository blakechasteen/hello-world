"""
Elle Matrix Adapter: Post AR observations to Matrix rooms

This adapter enables Elle to communicate through Matrix chat by:
- Formatting AR observations for text display
- Posting observations to Matrix rooms
- Handling bidirectional communication (queries from users)
- Integrating with Proto bridge for shared memory

Architecture:
    Elle Engine → Matrix Adapter → Matrix Client → Matrix Room
    Matrix Room → Matrix Adapter → Elle Engine (for queries)

Created: 2025-11-17 (Week 4: Elle AR Bridge Integration)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

try:
    from nio import AsyncClient, MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    AsyncClient = None
    MatrixRoom = None
    RoomMessageText = None

from elle.domain import ElleAction, ElleMode, ElleRequest
from elle.memory.proto_bridge import ElleObservation, ElleProtoMemoryBridge


# =============================================================================
# Matrix Message Formatting
# =============================================================================


class MatrixFormatter:
    """
    Formats Elle observations for Matrix display.

    Converts AR observations into readable Matrix messages with:
    - Emoji indicators for different modes
    - Structured formatting for readability
    - Links to related content
    """

    # Mode emoji mapping
    MODE_EMOJI = {
        ElleMode.AMBIENT: "🌅",  # Gentle presence
        ElleMode.CONSULTING: "💡",  # Advisory
        ElleMode.DIRECTIVE: "🎯",  # Clear guidance
        ElleMode.SILENT: "🤫",  # Silent observation
    }

    @staticmethod
    def format_observation(observation: ElleObservation) -> str:
        """
        Format Elle observation for Matrix.

        Args:
            observation: ElleObservation to format

        Returns:
            Formatted Matrix message (Markdown)

        Example:
            >>> obs = ElleObservation(...)
            >>> msg = MatrixFormatter.format_observation(obs)
        """
        lines = []

        # Header with location and time
        time_str = observation.timestamp.strftime("%I:%M %p")
        lines.append(f"### 🏗️ {observation.location.title()} ({time_str})")
        lines.append("")

        # Scene summary
        lines.append(f"**Observation**: {observation.scene_summary}")
        lines.append("")

        # Action taken (if any)
        if observation.action_taken:
            lines.append(f"**Elle's Action**: {observation.action_taken}")

            if observation.user_response:
                lines.append(f"**Your Response**: {observation.user_response}")
            lines.append("")

        # Objects observed
        if observation.objects_observed:
            lines.append("**Objects**:")
            for obj in observation.objects_observed[:5]:  # Limit to 5
                lines.append(f"  • {obj}")
            if len(observation.objects_observed) > 5:
                lines.append(f"  • ... and {len(observation.objects_observed) - 5} more")
            lines.append("")

        # Deferred tasks
        if observation.deferred_tasks:
            lines.append("**⏸️ Deferred Tasks**:")
            for task in observation.deferred_tasks:
                lines.append(f"  • {task['description']}")
                lines.append(f"    _{task['reason']}_")
                if 'elle_note' in task:
                    lines.append(f"    💭 Elle note: {task['elle_note']}")
            lines.append("")

        # Photos
        if observation.photo_ids:
            count = len(observation.photo_ids)
            lines.append(f"📸 {count} photo{'s' if count > 1 else ''} stored")
            lines.append("")

        # Tags
        if observation.tags:
            tags_str = " ".join(f"`{tag}`" for tag in observation.tags)
            lines.append(f"**Tags**: {tags_str}")

        return "\n".join(lines)

    @staticmethod
    def format_action(action: ElleAction, location: str = "Unknown") -> str:
        """
        Format Elle action for Matrix.

        Args:
            action: ElleAction to format
            location: Location context

        Returns:
            Formatted Matrix message

        Example:
            >>> action = ElleAction(...)
            >>> msg = MatrixFormatter.format_action(action)
        """
        lines = []

        # Mode indicator
        emoji = MatrixFormatter.MODE_EMOJI.get(action.mode, "🤖")
        lines.append(f"{emoji} **Elle** ({location})")
        lines.append("")

        # Utterance (if any)
        if action.utterance:
            lines.append(f"_{action.utterance}_")
            lines.append("")

        # Task suggestion
        if action.has_task:
            priority_emoji = {
                'LOW': '🔵',
                'MEDIUM': '🟡',
                'HIGH': '🟠',
                'CRITICAL': '🔴',
            }.get(str(action.task_priority).split('.')[-1], '⚪')

            lines.append("**Suggested Task**:")
            lines.append(f"{priority_emoji} {action.suggested_task_summary}")
            if action.task_estimate_minutes:
                lines.append(f"⏱️ Estimate: ~{action.task_estimate_minutes} minutes")
            lines.append("")

        # Reasoning (debug info)
        if action.reasoning and action.metadata.get('include_reasoning', False):
            lines.append("**Reasoning**:")
            lines.append(f"_{action.reasoning}_")

        return "\n".join(lines)

    @staticmethod
    def format_summary(summary: Dict[str, Any]) -> str:
        """
        Format location summary for Matrix.

        Args:
            summary: Summary dict from ElleProtoMemoryBridge

        Returns:
            Formatted summary message

        Example:
            >>> summary = await bridge.get_location_summary("workshop")
            >>> msg = MatrixFormatter.format_summary(summary)
        """
        lines = []

        # Header
        location = summary['location'].title()
        hours = summary['time_range_hours']
        lines.append(f"### 📊 {location} Summary (Last {hours}h)")
        lines.append("")

        # Stats
        lines.append("**Statistics**:")
        lines.append(f"  • Observations: {summary['observation_count']}")
        lines.append(f"  • Actions taken: {summary['actions_taken']}")
        lines.append(f"  • Deferred tasks: {summary['deferred_tasks_count']}")
        lines.append(f"  • Photos: {summary['photos_count']}")
        lines.append("")

        # Top tags
        if summary['top_tags']:
            lines.append("**Top Tags**:")
            for tag, count in summary['top_tags']:
                lines.append(f"  • `{tag}` ({count})")
            lines.append("")

        # Recent observations
        if summary['recent_observations']:
            lines.append("**Recent Activity**:")
            for obs in summary['recent_observations']:
                time_str = obs['timestamp'].strftime("%I:%M %p")
                lines.append(f"  • **{time_str}**: {obs['summary']}")
                if obs['action']:
                    lines.append(f"    _{obs['action']}_")
                if obs['deferred'] > 0:
                    lines.append(f"    ⏸️ {obs['deferred']} deferred")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Matrix Adapter
# =============================================================================


class ElleMatrixAdapter:
    """
    Bidirectional adapter between Elle and Matrix.

    Responsibilities:
    - Post Elle observations to Matrix rooms
    - Handle user queries from Matrix
    - Format messages appropriately
    - Integrate with Proto bridge

    Usage:
        adapter = ElleMatrixAdapter(homeserver, user_id, access_token)
        await adapter.start()

        # Post observation
        await adapter.post_observation(observation, room_id)

        # Query observations (from Matrix command)
        await adapter.handle_query(query, room_id)
    """

    def __init__(
        self,
        homeserver: str,
        user_id: str,
        access_token: str,
        bridge: Optional[ElleProtoMemoryBridge] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Matrix adapter.

        Args:
            homeserver: Matrix homeserver URL
            user_id: Bot user ID
            access_token: Access token for authentication
            bridge: Optional ElleProtoMemoryBridge instance
            logger: Optional logger
        """
        if not MATRIX_AVAILABLE:
            raise ImportError(
                "Matrix-nio not available. Install with: pip install matrix-nio"
            )

        self.homeserver = homeserver
        self.user_id = user_id
        self.access_token = access_token
        self.bridge = bridge
        self.logger = logger or logging.getLogger(__name__)

        # Matrix client
        self.client = AsyncClient(homeserver, user_id)

        # Formatter
        self.formatter = MatrixFormatter()

    async def start(self):
        """Start Matrix client and login."""
        self.logger.info(f"Starting Matrix adapter for {self.user_id}")

        # Login (if using access token, just set it)
        self.client.access_token = self.access_token

        self.logger.info("Matrix adapter started")

    async def stop(self):
        """Stop Matrix client."""
        self.logger.info("Stopping Matrix adapter")
        await self.client.close()

    # =========================================================================
    # Post Observations to Matrix
    # =========================================================================

    async def post_observation(
        self,
        observation: ElleObservation,
        room_id: str
    ) -> None:
        """
        Post Elle observation to Matrix room.

        Args:
            observation: ElleObservation to post
            room_id: Matrix room ID

        Example:
            >>> observation = ElleObservation(...)
            >>> await adapter.post_observation(observation, "!abc:matrix.org")
        """
        # Format observation
        message = self.formatter.format_observation(observation)

        # Send to Matrix
        try:
            await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message,
                    "format": "org.matrix.custom.html",
                    "formatted_body": self._markdown_to_html(message),
                }
            )

            self.logger.info(
                f"Posted observation {observation.observation_id} to {room_id}"
            )

        except Exception as e:
            self.logger.error(f"Error posting observation: {e}")

    async def post_action(
        self,
        action: ElleAction,
        room_id: str,
        location: str = "Unknown"
    ) -> None:
        """
        Post Elle action to Matrix room.

        Args:
            action: ElleAction to post
            room_id: Matrix room ID
            location: Location context

        Example:
            >>> action = ElleAction(...)
            >>> await adapter.post_action(action, "!abc:matrix.org", "workshop")
        """
        # Format action
        message = self.formatter.format_action(action, location)

        # Send to Matrix
        try:
            await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message,
                    "format": "org.matrix.custom.html",
                    "formatted_body": self._markdown_to_html(message),
                }
            )

            self.logger.info(f"Posted action to {room_id}")

        except Exception as e:
            self.logger.error(f"Error posting action: {e}")

    async def post_summary(
        self,
        summary: Dict[str, Any],
        room_id: str
    ) -> None:
        """
        Post location summary to Matrix room.

        Args:
            summary: Summary dict from bridge
            room_id: Matrix room ID

        Example:
            >>> summary = await bridge.get_location_summary("workshop")
            >>> await adapter.post_summary(summary, "!abc:matrix.org")
        """
        # Format summary
        message = self.formatter.format_summary(summary)

        # Send to Matrix
        try:
            await self.client.room_send(
                room_id=room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message,
                    "format": "org.matrix.custom.html",
                    "formatted_body": self._markdown_to_html(message),
                }
            )

            self.logger.info(f"Posted summary to {room_id}")

        except Exception as e:
            self.logger.error(f"Error posting summary: {e}")

    # =========================================================================
    # Handle Queries from Matrix
    # =========================================================================

    async def handle_query(
        self,
        query_text: str,
        room_id: str
    ) -> None:
        """
        Handle query from Matrix user.

        Supports queries like:
        - "@elle workshop-summary"
        - "@elle status"
        - "@elle recent workshop"

        Args:
            query_text: Query text from user
            room_id: Matrix room ID

        Example:
            >>> await adapter.handle_query("@elle workshop-summary", room_id)
        """
        if not self.bridge:
            await self._send_error(
                room_id,
                "Bridge not configured. Cannot query observations."
            )
            return

        # Parse query
        parts = query_text.lower().split()

        try:
            if "summary" in parts:
                # Get location summary
                location = self._extract_location(parts) or "workshop"
                summary = await self.bridge.get_location_summary(location, 24)
                await self.post_summary(summary, room_id)

            elif "status" in parts:
                # Get overall status
                health = await self.bridge.health_check()
                await self._send_message(
                    room_id,
                    f"🟢 Elle Bridge Status: {health['status']}\n"
                    f"Memories: {health.get('loom', {}).get('memories', 'N/A')}"
                )

            elif "recent" in parts:
                # Get recent observations
                location = self._extract_location(parts)
                observations = await self.bridge.query_observations(
                    query_type="location" if location else "recent",
                    location=location,
                    limit=5
                )

                if observations:
                    for obs in observations:
                        await self.post_observation(obs, room_id)
                else:
                    await self._send_message(room_id, "No recent observations found.")

            elif "deferred" in parts or "tasks" in parts:
                # Get deferred tasks
                location = self._extract_location(parts)
                tasks = await self.bridge.get_deferred_tasks(location)

                if tasks:
                    lines = ["### ⏸️ Deferred Tasks", ""]
                    for task in tasks[:10]:
                        lines.append(f"**{task['description']}**")
                        lines.append(f"  • Location: {task['location']}")
                        lines.append(f"  • Reason: {task['reason']}")
                        if 'elle_note' in task:
                            lines.append(f"  • Note: {task['elle_note']}")
                        lines.append("")

                    await self._send_message(room_id, "\n".join(lines))
                else:
                    await self._send_message(room_id, "No deferred tasks found.")

            else:
                # Unknown query
                await self._send_help(room_id)

        except Exception as e:
            self.logger.error(f"Error handling query: {e}")
            await self._send_error(room_id, f"Error: {str(e)}")

    async def _send_message(self, room_id: str, message: str):
        """Send simple text message."""
        await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={
                "msgtype": "m.text",
                "body": message,
            }
        )

    async def _send_error(self, room_id: str, error: str):
        """Send error message."""
        await self._send_message(room_id, f"❌ {error}")

    async def _send_help(self, room_id: str):
        """Send help message."""
        help_text = """
**Elle Commands**:
  • `@elle workshop-summary` - Get workshop activity summary
  • `@elle status` - Check Elle bridge status
  • `@elle recent [location]` - Get recent observations
  • `@elle deferred [location]` - Get deferred tasks
  • `@elle help` - Show this help
        """.strip()
        await self._send_message(room_id, help_text)

    def _extract_location(self, parts: List[str]) -> Optional[str]:
        """Extract location from query parts."""
        locations = ["workshop", "garden", "shed", "kitchen"]
        for part in parts:
            if part in locations:
                return part
        return None

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert simple Markdown to HTML for Matrix.

        This is a very basic converter. For production, use a proper library.
        """
        import re

        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^\*\*(.+)\*\*:', r'<strong>\1</strong>:', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Line breaks
        html = html.replace('\n', '<br/>')

        return html


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    import asyncio

    async def demo():
        """Demonstrate Elle Matrix adapter."""

        # Note: This requires valid Matrix credentials
        # Replace with your actual values

        homeserver = "https://matrix.org"
        user_id = "@elle:matrix.org"
        access_token = "YOUR_ACCESS_TOKEN"
        room_id = "!YourRoomID:matrix.org"

        # Create adapter
        adapter = ElleMatrixAdapter(
            homeserver=homeserver,
            user_id=user_id,
            access_token=access_token
        )

        try:
            await adapter.start()

            # Create test observation
            observation = ElleObservation(
                observation_id="obs_demo_001",
                timestamp=datetime.now(),
                location="workshop",
                scene_summary="Workbench cleared and organized",
                objects_observed=["shop_vac", "hand_tools", "workbench"],
                action_taken="Suggested clearing sawdust",
                user_response="Completed task",
                deferred_tasks=[
                    {
                        'description': 'Birdhouse project',
                        'reason': 'Need cedar boards',
                        'elle_note': 'Check for brad nails in toolbox',
                    }
                ],
                tags=['productive', 'decluttering'],
            )

            # Post to Matrix
            await adapter.post_observation(observation, room_id)
            print("✅ Posted observation to Matrix")

        finally:
            await adapter.stop()

    # Run demo
    # asyncio.run(demo())
    print("Demo ready. Update credentials and uncomment asyncio.run(demo()) to test.")
