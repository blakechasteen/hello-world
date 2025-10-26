#!/usr/bin/env python3
"""
HoloLoom Matrix Bot Handlers
=============================
Command handlers that integrate Matrix bot with HoloLoom weaving orchestrator.

Commands:
- !weave <query> - Execute query through full weaving cycle
- !memory <action> - Manage HoloLoom memory
- !skill <name> - Execute a skill
- !analyze <text> - Analyze with MCTS
- !stats - Show system statistics
- !loops - Show active loops
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

# Import HoloLoom components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weaving_orchestrator import WeavingOrchestrator
from config import Config

logger = logging.getLogger(__name__)


class HoloLoomMatrixHandlers:
    """
    Handler class for Matrix bot commands that use HoloLoom.

    Integrates Matrix chatops with:
    - Weaving orchestrator (MCTS + memory + context)
    - Memory management (add, search, stats)
    - Skill execution
    - Analytics and monitoring
    """

    def __init__(self, bot, config_mode: str = "fast"):
        """
        Initialize handlers with HoloLoom orchestrator.

        Args:
            bot: MatrixBot instance
            config_mode: HoloLoom config (bare/fast/fused)
        """
        self.bot = bot

        # Initialize HoloLoom weaving orchestrator
        logger.info(f"Initializing HoloLoom orchestrator (mode={config_mode})...")
        self.orchestrator = WeavingOrchestrator(
            config=Config.from_mode(config_mode) if config_mode else None,
            use_mcts=True,
            mcts_simulations=50
        )

        logger.info("HoloLoom handlers initialized")

    # ========================================================================
    # Core Commands
    # ========================================================================

    async def handle_weave(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Execute query through full weaving cycle.

        Usage: !weave What is Thompson Sampling?
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!weave <your query>`\n\nExample: `!weave Explain MCTS`",
                markdown=True
            )
            return

        try:
            # Show typing indicator
            await self.bot.send_typing(room.room_id, typing=True)

            # Send processing message
            await self.bot.send_message(
                room.room_id,
                f"🔮 **Weaving query...**\n\n`{args[:100]}`",
                markdown=True
            )

            # Execute weaving cycle
            spacetime = await self.orchestrator.weave(args)

            # Format response
            response = f"""
**✨ Weaving Complete**

**Query:** {args}

**Decision:**
• Tool: `{spacetime.tool_used}`
• Confidence: {spacetime.confidence:.1%}
• Duration: {spacetime.trace.duration_ms:.0f}ms

**Context:**
• Shards retrieved: {spacetime.trace.context_shards_count}
• Pattern: {spacetime.trace.pattern_used}
• MCTS simulations: 50

**Result:**
{spacetime.trace.execution_result[:500] if spacetime.trace.execution_result else 'No output'}
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Weave error: {e}", exc_info=True)
            await self.bot.send_message(
                room.room_id,
                f"❌ **Error:** {str(e)}"
            )
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_memory(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Manage HoloLoom memory.

        Usage:
            !memory add <text>
            !memory search <query>
            !memory stats
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                """
**Memory Commands:**

• `!memory add <text>` - Add knowledge to memory
• `!memory search <query>` - Search memory
• `!memory stats` - Show memory statistics

**Example:**
```
!memory add MCTS uses Thompson Sampling
!memory search What is MCTS?
```
""",
                markdown=True
            )
            return

        parts = args.split(maxsplit=1)
        action = parts[0].lower()
        content = parts[1] if len(parts) > 1 else ""

        try:
            if action == "add":
                if not content:
                    await self.bot.send_message(room.room_id, "Usage: `!memory add <text>`")
                    return

                # Add to memory
                await self.orchestrator.add_knowledge(
                    content,
                    {"source": "matrix", "user": event.sender, "room": room.room_id}
                )

                await self.bot.send_message(
                    room.room_id,
                    f"✅ **Added to memory**\n\n`{content[:200]}`",
                    markdown=True
                )

            elif action == "search":
                if not content:
                    await self.bot.send_message(room.room_id, "Usage: `!memory search <query>`")
                    return

                # Search memory
                results = await self.orchestrator._retrieve_context(content, limit=5)

                if not results:
                    await self.bot.send_message(room.room_id, "No results found")
                    return

                response = f"**🔍 Search Results** ({len(results)} found)\n\n"
                for i, shard in enumerate(results[:3], 1):
                    text = shard.text[:150] + "..." if len(shard.text) > 150 else shard.text
                    response += f"{i}. `{text}`\n\n"

                await self.bot.send_message(room.room_id, response, markdown=True)

            elif action == "stats":
                # Get memory stats
                stats = self.orchestrator.get_statistics()

                # Check backend type
                backend_info = "In-memory"
                if hasattr(self.orchestrator.memory_store, 'backends'):
                    backends = [b.name for b in self.orchestrator.memory_store.backends]
                    backend_info = f"Hybrid ({', '.join(backends)})"
                elif hasattr(self.orchestrator.memory_store, 'data_dir'):
                    backend_info = f"File ({self.orchestrator.memory_store.data_dir})"

                response = f"""
**📊 Memory Statistics**

**Backend:** {backend_info}

**Weaving:**
• Total cycles: {stats['total_weavings']}
• Pattern usage: {stats['pattern_usage']}

**System:**
• Uptime: {datetime.now().isoformat()}
"""

                await self.bot.send_message(room.room_id, response, markdown=True)

            else:
                await self.bot.send_message(
                    room.room_id,
                    f"Unknown action: `{action}`. Try `!memory` for help."
                )

        except Exception as e:
            logger.error(f"Memory error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    async def handle_analyze(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Analyze text with MCTS decision-making.

        Usage: !analyze <text>
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!analyze <text to analyze>`"
            )
            return

        try:
            await self.bot.send_typing(room.room_id, typing=True)

            # Weave analysis query
            query = f"Analyze the following: {args}"
            spacetime = await self.orchestrator.weave(query)

            response = f"""
**🔬 Analysis Results**

**Input:** {args[:200]}

**Decision:** {spacetime.tool_used} ({spacetime.confidence:.1%} confidence)

**Pattern:** {spacetime.trace.pattern_used}
**Duration:** {spacetime.trace.duration_ms:.0f}ms
**Context used:** {spacetime.trace.context_shards_count} shards
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_stats(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show HoloLoom system statistics.

        Usage: !stats
        """
        try:
            stats = self.orchestrator.get_statistics()

            # Format MCTS stats
            mcts_info = ""
            if 'mcts_stats' in stats:
                mcts = stats['mcts_stats']
                flux = mcts.get('flux_stats', {})
                mcts_info = f"""
**MCTS Flux Capacitor:**
• Total simulations: {flux.get('total_simulations', 0)}
• Decisions: {mcts.get('decision_count', 0)}
• Tools: {list(flux.get('tool_distribution', {}).keys())}
"""

            response = f"""
**📈 HoloLoom Statistics**

**Weaving:**
• Total cycles: {stats['total_weavings']}
• Patterns: {stats['pattern_usage']}

{mcts_info}

**System:**
• Status: ✅ Operational
• Mode: {self.orchestrator.config.mode if hasattr(self.orchestrator.config, 'mode') else 'fast'}
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    async def handle_help(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show help for HoloLoom commands.

        Usage: !help
        """
        help_text = """
**🔮 HoloLoom Bot - Commands**

**Core:**
• `!weave <query>` - Execute full weaving cycle
• `!analyze <text>` - Analyze with MCTS
• `!stats` - System statistics

**Memory:**
• `!memory add <text>` - Add to knowledge base
• `!memory search <query>` - Search memory
• `!memory stats` - Memory statistics

**Utilities:**
• `!ping` - Check bot status
• `!help` - Show this help

**Examples:**
```
!weave Explain Thompson Sampling
!memory add MCTS balances exploration vs exploitation
!memory search What is MCTS?
!analyze This is a complex algorithm
```

**Powered by:**
• MCTS Flux Capacitor (Thompson Sampling)
• Hybrid Memory (Qdrant + Neo4j + File)
• Multi-scale embeddings
• Context-aware weaving
"""

        await self.bot.send_message(room.room_id, help_text, markdown=True)

    async def handle_ping(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Simple ping command.

        Usage: !ping
        """
        stats = self.orchestrator.get_statistics()

        response = f"""
**🏓 Pong!**

HoloLoom is operational.
• Weavings: {stats['total_weavings']}
• Status: ✅ Ready
"""

        await self.bot.send_message(room.room_id, response, markdown=True)

    # ========================================================================
    # Utilities
    # ========================================================================

    def register_all(self):
        """Register all handlers with the bot."""
        self.bot.register_handler("weave", self.handle_weave)
        self.bot.register_handler("memory", self.handle_memory)
        self.bot.register_handler("analyze", self.handle_analyze)
        self.bot.register_handler("stats", self.handle_stats)
        self.bot.register_handler("help", self.handle_help)
        self.bot.register_handler("ping", self.handle_ping)

        logger.info("All HoloLoom handlers registered")

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down HoloLoom orchestrator...")
        self.orchestrator.stop()
        logger.info("HoloLoom handlers shut down")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
HoloLoom Matrix Bot Handlers
=============================

This module provides command handlers that integrate
Matrix chatops with HoloLoom weaving orchestrator.

Commands available:
- !weave <query> - Full weaving cycle
- !memory add/search/stats - Memory management
- !analyze <text> - MCTS analysis
- !stats - System statistics
- !help - Command help
- !ping - Health check

To use:
    from matrix_bot import MatrixBot, MatrixBotConfig
    from hololoom_handlers import HoloLoomMatrixHandlers

    bot = MatrixBot(config)
    handlers = HoloLoomMatrixHandlers(bot)
    handlers.register_all()
    await bot.start()
""")
