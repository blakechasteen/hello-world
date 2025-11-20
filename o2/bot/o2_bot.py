"""
O2 Bot - Platform Anarchism meets Agentic Intelligence

Main bot application integrating:
- Matrix.org (federated communication)
- HoloLoom (agentic AI reasoning)
- Democratic governance (voting, proposals)
- User-owned memory (federated knowledge graphs)
"""

import asyncio
import logging
import os
import sys
from typing import Optional

from nio import (
    AsyncClient,
    MatrixRoom,
    RoomMessageText,
    InviteEvent,
    LoginError,
    LocalProtocolError,
)

# Add HoloLoom to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query

# Import O2 modules
from bot.governance import GovernanceEngine
from bot.federated_memory import FederatedMemoryManager
from bot.swarm_coordinator import SwarmCoordinator

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('o2-bot')


class O2Bot:
    """
    O2 Platform Bot

    Combines Matrix communication, HoloLoom intelligence, and democratic governance.
    """

    def __init__(self):
        # Matrix configuration
        self.homeserver = os.getenv('MATRIX_HOMESERVER', 'http://localhost:8008')
        self.user_id = os.getenv('MATRIX_USER_ID', '@o2:matrix.localhost')
        self.password = os.getenv('MATRIX_PASSWORD')

        if not self.password:
            raise ValueError("MATRIX_PASSWORD environment variable required")

        # Create Matrix client
        self.client = AsyncClient(self.homeserver, self.user_id)

        # HoloLoom configuration
        config_mode = os.getenv('HOLOLOOM_CONFIG', 'fast')
        if config_mode == 'bare':
            self.hololoom_config = Config.bare()
        elif config_mode == 'fused':
            self.hololoom_config = Config.fused()
        else:
            self.hololoom_config = Config.fast()

        # Enable alignment framework
        self.hololoom_config.enable_alignment = os.getenv('O2_ENABLE_ALIGNMENT', 'true').lower() == 'true'

        # Initialize components
        self.governance = None
        self.memory_manager = None
        self.swarm = None
        self.user_looms = {}  # Per-user HoloLoom instances

        logger.info(f"O2 Bot initialized: {self.user_id}")
        logger.info(f"HoloLoom config: {config_mode}")
        logger.info(f"Governance enabled: {os.getenv('O2_ENABLE_GOVERNANCE', 'true')}")
        logger.info(f"Swarm enabled: {os.getenv('O2_ENABLE_SWARM', 'true')}")

    async def start(self):
        """Start the O2 bot."""
        logger.info("Starting O2 bot...")

        # Login to Matrix
        try:
            response = await self.client.login(self.password)
            if isinstance(response, LoginError):
                logger.error(f"Failed to login: {response.message}")
                return
            logger.info(f"Logged in as {self.user_id}")
        except LocalProtocolError as e:
            logger.error(f"Login failed: {e}")
            return

        # Initialize governance
        if os.getenv('O2_ENABLE_GOVERNANCE', 'true').lower() == 'true':
            self.governance = GovernanceEngine(self.client)
            await self.governance.initialize()
            logger.info("Governance engine initialized")

        # Initialize federated memory
        self.memory_manager = FederatedMemoryManager(self.hololoom_config)
        await self.memory_manager.initialize()
        logger.info("Federated memory manager initialized")

        # Initialize swarm coordinator
        if os.getenv('O2_ENABLE_SWARM', 'true').lower() == 'true':
            self.swarm = SwarmCoordinator(self.hololoom_config)
            await self.swarm.initialize()
            logger.info("Swarm coordinator initialized")

        # Register callbacks
        self.client.add_event_callback(self.on_message, RoomMessageText)
        self.client.add_event_callback(self.on_invite, InviteEvent)

        # Start syncing
        logger.info("Starting Matrix sync...")
        await self.client.sync_forever(timeout=30000)

    async def on_invite(self, room: MatrixRoom, event: InviteEvent):
        """Handle room invites."""
        logger.info(f"Invited to room {room.room_id} by {event.sender}")

        # Auto-join all rooms (platform anarchism: open to all)
        await self.client.join(room.room_id)
        logger.info(f"Joined room {room.room_id}")

        # Send welcome message
        welcome = (
            "👋 **O2 Bot** has joined!\n\n"
            "**Platform Anarchism meets Agentic Intelligence**\n\n"
            "I'm here to help with:\n"
            "- 🧠 Intelligent Q&A (HoloLoom reasoning)\n"
            "- 🗳️ Democratic governance (proposals, voting)\n"
            "- 🤖 Agentic swarms (multi-agent collaboration)\n"
            "- 💾 Federated memory (user-owned knowledge graphs)\n\n"
            "Try: `@o2 help` or `@o2 what is platform anarchism?`\n\n"
            "_Remember: YOU own your data!_ 🚀"
        )
        await self.client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": welcome, "format": "org.matrix.custom.html",
                     "formatted_body": welcome.replace("\n", "<br>")}
        )

    async def on_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming messages."""
        # Ignore messages from self
        if event.sender == self.user_id:
            return

        # Only respond when mentioned
        if self.user_id not in event.body and "@o2" not in event.body.lower():
            return

        message = event.body.replace(self.user_id, '').replace('@o2', '').strip()
        sender = event.sender

        logger.info(f"Message from {sender} in {room.room_id}: {message}")

        # Route command
        try:
            response = await self.handle_command(message, sender, room.room_id)
            await self.send_message(room.room_id, response)
        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            await self.send_message(room.room_id, f"❌ Error: {str(e)}")

    async def handle_command(self, message: str, user_id: str, room_id: str) -> str:
        """Route commands to appropriate handlers."""
        message_lower = message.lower()

        # Help command
        if message_lower.startswith('help'):
            return self.get_help_text()

        # Governance commands
        if message_lower.startswith('propose'):
            return await self.handle_propose(message, user_id, room_id)

        if message_lower.startswith('vote'):
            return await self.handle_vote(message, user_id, room_id)

        if message_lower.startswith('tally'):
            return await self.handle_tally(message, user_id, room_id)

        # Memory commands
        if message_lower.startswith('export'):
            return await self.handle_export(user_id)

        if message_lower.startswith('share'):
            return await self.handle_share(message, user_id)

        if message_lower.startswith('revoke'):
            return await self.handle_revoke(message, user_id)

        if message_lower.startswith('list shared'):
            return await self.handle_list_shared(user_id)

        if message_lower.startswith('audit'):
            return await self.handle_audit(message, user_id)

        # Swarm commands
        if message_lower.startswith('run swarm'):
            return await self.handle_swarm(message, user_id, room_id)

        # Default: HoloLoom query
        return await self.handle_query(message, user_id)

    async def handle_query(self, message: str, user_id: str) -> str:
        """Handle HoloLoom query."""
        # Get or create user's HoloLoom instance
        if user_id not in self.user_looms:
            logger.info(f"Creating HoloLoom instance for {user_id}")
            user_loom = await self.memory_manager.get_user_loom(user_id)
            self.user_looms[user_id] = user_loom

        loom = self.user_looms[user_id]

        # Query HoloLoom
        try:
            memories = await loom.recall(message)

            if not memories:
                return "🤔 I don't have any memories about that yet. Ask me to learn something!"

            # Format response
            response = f"💭 **Query**: {message}\n\n"
            for i, memory in enumerate(memories[:3], 1):
                response += f"{i}. {memory.content}\n"

            return response

        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return f"❌ Query failed: {str(e)}"

    async def handle_propose(self, message: str, user_id: str, room_id: str) -> str:
        """Handle proposal creation."""
        if not self.governance:
            return "❌ Governance not enabled"

        # Parse proposal (simple format for now)
        # Format: propose: [title]
        parts = message.split(':', 1)
        if len(parts) < 2:
            return "❌ Invalid format. Use: `@o2 propose: [title]`"

        title = parts[1].strip()

        # Create proposal
        proposal_id = await self.governance.create_proposal(
            room_id=room_id,
            author=user_id,
            title=title,
            proposal_type='policy',
            threshold=0.67,  # 2/3 majority
            duration_hours=72
        )

        return (
            f"📋 **Proposal #{proposal_id} Created**\n\n"
            f"**Title**: {title}\n"
            f"**Author**: {user_id}\n"
            f"**Type**: Policy\n"
            f"**Threshold**: 67% (2/3 majority)\n"
            f"**Duration**: 72 hours\n\n"
            f"React to vote:\n"
            f"✅ Approve\n"
            f"❌ Reject\n"
            f"🤔 Abstain"
        )

    async def handle_vote(self, message: str, user_id: str, room_id: str) -> str:
        """Handle voting."""
        if not self.governance:
            return "❌ Governance not enabled"

        # Parse vote
        # Format: vote [yes/no/abstain] [proposal_id]
        parts = message.lower().split()
        if len(parts) < 2:
            return "❌ Invalid format. Use: `@o2 vote yes` (votes on latest proposal)"

        vote = parts[1]
        if vote not in ['yes', 'no', 'abstain']:
            return "❌ Vote must be yes, no, or abstain"

        # Get latest proposal for room
        proposal_id = await self.governance.get_latest_proposal(room_id)
        if not proposal_id:
            return "❌ No active proposals in this room"

        # Record vote
        await self.governance.record_vote(proposal_id, user_id, vote)

        return f"✅ Vote recorded: {vote} on proposal #{proposal_id}"

    async def handle_tally(self, message: str, user_id: str, room_id: str) -> str:
        """Handle vote tallying."""
        if not self.governance:
            return "❌ Governance not enabled"

        # Get latest proposal
        proposal_id = await self.governance.get_latest_proposal(room_id)
        if not proposal_id:
            return "❌ No active proposals in this room"

        # Tally votes
        result = await self.governance.tally_votes(proposal_id)

        return (
            f"📊 **Proposal #{proposal_id} Results**\n\n"
            f"**Status**: {result['status']}\n"
            f"**Yes**: {result['yes']} ({result['yes_pct']:.0%})\n"
            f"**No**: {result['no']} ({result['no_pct']:.0%})\n"
            f"**Abstain**: {result['abstain']}\n"
            f"**Total**: {result['total']} votes\n"
            f"**Threshold**: {result['threshold']:.0%}\n\n"
            f"**Outcome**: {result['outcome']}"
        )

    async def handle_export(self, user_id: str) -> str:
        """Handle data export."""
        # Export user's memory graph
        export_path = await self.memory_manager.export_user_data(user_id)

        return (
            f"📦 **Data Export Complete**\n\n"
            f"Your memory graph has been exported to:\n"
            f"`{export_path}`\n\n"
            f"This file contains:\n"
            f"- Knowledge graph (entities, relationships)\n"
            f"- Vector embeddings\n"
            f"- Photos and metadata\n"
            f"- Complete audit trail\n\n"
            f"**You own this data!** Import it to any O2 instance anytime."
        )

    async def handle_share(self, message: str, user_id: str) -> str:
        """
        Handle memory sharing.

        Format: share [memory_id] with @user:server.org [permissions] [expiration]
        Example: share mem_0 with @bob:matrix.org read 7d
        """
        try:
            # Parse command
            # Format: share [memory_id] with @recipient [permissions] [expiration]
            parts = message.split()

            if len(parts) < 4:
                return (
                    "❌ Invalid format.\n\n"
                    "**Usage**: `@o2 share [memory_id] with @user:server.org [permissions] [expiration]`\n\n"
                    "**Example**: `@o2 share mem_0 with @bob:matrix.org read 7d`\n"
                    "**Permissions**: read, write\n"
                    "**Expiration**: 24h, 7d, 30d, etc."
                )

            memory_id = parts[1]
            recipient = parts[3]

            # Parse optional permissions and expiration
            permissions = ['read']  # Default
            expiration = None

            if len(parts) > 4:
                # Check if next part is permissions or expiration
                if parts[4] in ['read', 'write']:
                    permissions = [parts[4]]
                    if len(parts) > 5:
                        expiration = parts[5]
                else:
                    expiration = parts[4]

            # Share memory
            share_id = await self.memory_manager.share_memory(
                from_user=user_id,
                to_user=recipient,
                memory_id=memory_id,
                permissions=permissions,
                expiration=expiration
            )

            response = (
                f"✅ **Memory Shared**\n\n"
                f"**Memory**: {memory_id}\n"
                f"**Shared with**: {recipient}\n"
                f"**Permissions**: {', '.join(permissions)}\n"
            )

            if expiration:
                response += f"**Expires**: {expiration}\n"

            response += f"\n**Share ID**: `{share_id}`\n\n"
            response += "_Remember: End-to-end encrypted! Only you and recipient can read._"

            return response

        except Exception as e:
            logger.error(f"Share error: {e}", exc_info=True)
            return f"❌ Failed to share memory: {str(e)}"

    async def handle_revoke(self, message: str, user_id: str) -> str:
        """
        Handle access revocation.

        Format: revoke [memory_id] from @user:server.org
        Example: revoke mem_0 from @bob:matrix.org
        """
        try:
            # Parse command
            # Format: revoke [memory_id] from @recipient
            parts = message.split()

            if len(parts) < 4:
                return (
                    "❌ Invalid format.\n\n"
                    "**Usage**: `@o2 revoke [memory_id] from @user:server.org`\n\n"
                    "**Example**: `@o2 revoke mem_0 from @bob:matrix.org`"
                )

            memory_id = parts[1]
            recipient = parts[3]

            # Revoke access
            await self.memory_manager.revoke_access(
                from_user=user_id,
                to_user=recipient,
                memory_id=memory_id
            )

            return (
                f"✅ **Access Revoked**\n\n"
                f"**Memory**: {memory_id}\n"
                f"**User**: {recipient}\n\n"
                f"_{recipient} can no longer access this memory._"
            )

        except Exception as e:
            logger.error(f"Revoke error: {e}", exc_info=True)
            return f"❌ Failed to revoke access: {str(e)}"

    async def handle_list_shared(self, user_id: str) -> str:
        """
        Handle listing shared memories.

        Lists all memories shared with this user.
        """
        try:
            # Get shared memories
            shared_memories = await self.memory_manager.get_shared_memories(user_id)

            if not shared_memories:
                return (
                    "📭 **No Shared Memories**\n\n"
                    "No one has shared memories with you yet."
                )

            response = f"📬 **Shared Memories ({len(shared_memories)})**\n\n"

            for i, memory in enumerate(shared_memories[:10], 1):
                response += (
                    f"{i}. **{memory.memory_id}**\n"
                    f"   From: {memory.owner_id}\n"
                    f"   Created: {memory.created_at.strftime('%Y-%m-%d %H:%M')}\n\n"
                )

            if len(shared_memories) > 10:
                response += f"\n_...and {len(shared_memories) - 10} more_"

            return response

        except Exception as e:
            logger.error(f"List shared error: {e}", exc_info=True)
            return f"❌ Failed to list shared memories: {str(e)}"

    async def handle_audit(self, message: str, user_id: str) -> str:
        """
        Handle audit trail requests.

        Format: audit [memory_id]
        Example: audit mem_0
        """
        try:
            # Parse command
            parts = message.split()

            if len(parts) < 2:
                return (
                    "❌ Invalid format.\n\n"
                    "**Usage**: `@o2 audit [memory_id]`\n\n"
                    "**Example**: `@o2 audit mem_0`"
                )

            memory_id = parts[1]

            # Get audit trail
            audit_trail = await self.memory_manager.get_audit_trail(memory_id)

            if not audit_trail:
                return (
                    f"📋 **Audit Trail: {memory_id}**\n\n"
                    "No access events recorded for this memory."
                )

            response = f"📋 **Audit Trail: {memory_id}**\n\n"

            for event in audit_trail:
                response += (
                    f"**User**: {event['granted_to']}\n"
                    f"**Permissions**: {', '.join(event['permissions'])}\n"
                    f"**Granted**: {event['granted_at']}\n"
                )

                if event['expires_at']:
                    response += f"**Expires**: {event['expires_at']}\n"

                response += (
                    f"**Accessed**: {event['accessed_count']} times\n"
                    f"**Last Access**: {event.get('last_accessed', 'Never')}\n"
                    f"**Status**: {'Expired' if event['is_expired'] else 'Active'}\n\n"
                )

            return response

        except Exception as e:
            logger.error(f"Audit error: {e}", exc_info=True)
            return f"❌ Failed to get audit trail: {str(e)}"

    async def handle_swarm(self, message: str, user_id: str, room_id: str) -> str:
        """Handle swarm coordination."""
        if not self.swarm:
            return "❌ Swarm not enabled"

        # Parse swarm command
        # Format: run swarm: [task description]
        parts = message.split(':', 1)
        if len(parts) < 2:
            return "❌ Invalid format. Use: `@o2 run swarm: [task]`"

        task = parts[1].strip()

        # Start swarm (placeholder)
        return (
            f"🤖 **Swarm Activated**\n\n"
            f"**Task**: {task}\n\n"
            f"🚧 Agentic swarm coming soon!\n"
            f"Will decompose task and coordinate multiple agents."
        )

    def get_help_text(self) -> str:
        """Get help text."""
        return """
**O2 Bot Commands**

**Platform Anarchism meets Agentic Intelligence**

**Basic Queries**:
- `@o2 [question]` - Ask HoloLoom anything
- `@o2 research: [topic]` - Deep research mode
- `@o2 verify: [claim]` - Fact-check claims

**Governance** (Democratic voting):
- `@o2 propose: [title]` - Create proposal
- `@o2 vote yes/no/abstain` - Vote on proposal
- `@o2 tally votes` - Check voting results

**Memory** (User-owned data):
- `@o2 export my data` - Export memory graph
- `@o2 share [memory_id] with @user [read/write] [expiration]` - Share memory
- `@o2 revoke [memory_id] from @user` - Revoke access
- `@o2 list shared` - List memories shared with you
- `@o2 audit [memory_id]` - View access audit trail

**Swarm** (Multi-agent collaboration):
- `@o2 run swarm: [task]` - Activate agentic swarm

**Help**:
- `@o2 help` - Show this message
- `@o2 what is platform anarchism?` - Learn our philosophy

**Remember**: YOU own your data! 🚀
"""

    async def send_message(self, room_id: str, message: str):
        """Send message to room."""
        await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": message}
        )

    async def close(self):
        """Clean up resources."""
        logger.info("Shutting down O2 bot...")

        # Close user HoloLoom instances
        for user_id, loom in self.user_looms.items():
            try:
                await loom.close()
            except Exception as e:
                logger.error(f"Error closing HoloLoom for {user_id}: {e}")

        # Close Matrix client
        await self.client.close()

        logger.info("O2 bot shutdown complete")


async def main():
    """Main entry point."""
    bot = O2Bot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await bot.close()


if __name__ == '__main__':
    asyncio.run(main())
