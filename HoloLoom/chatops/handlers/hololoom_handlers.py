#!/usr/bin/env python3
"""
HoloLoom Matrix Bot Handlers
=============================
Command handlers that integrate Matrix bot with HoloLoom weaving shuttle.

Commands:
- !weave <query> - Execute query through full weaving cycle with reflection
- !memory <action> - Manage HoloLoom memory
- !trace <query_id> - Show Spacetime trace for a query
- !learn - Trigger learning analysis
- !stats - Show system statistics including reflection metrics
- !analyze <text> - Analyze with convergence engine
- !help - Show command help
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from nio import MatrixRoom, RoomMessageText
except ImportError:
    pass

# Import HoloLoom components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.loom.command import PatternCard

logger = logging.getLogger(__name__)


class HoloLoomMatrixHandlers:
    """
    Handler class for Matrix bot commands that use HoloLoom.

    Integrates Matrix chatops with:
    - Weaving shuttle (full 9-step weaving cycle)
    - Reflection loop (continuous learning)
    - Memory management (add, search, stats)
    - Analytics and monitoring
    """

    def __init__(self, bot, config_mode: str = "fast", memory_shards: Optional[list] = None):
        """
        Initialize handlers with HoloLoom weaving shuttle.

        Args:
            bot: MatrixBot instance
            config_mode: HoloLoom config (bare/fast/fused)
            memory_shards: Optional initial memory shards
        """
        self.bot = bot

        # Initialize HoloLoom weaving shuttle
        logger.info(f"Initializing HoloLoom WeavingShuttle (mode={config_mode})...")

        # Create config
        if config_mode == "bare":
            config = Config.bare()
        elif config_mode == "fused":
            config = Config.fused()
        else:
            config = Config.fast()

        # Create initial memory shards if not provided
        if memory_shards is None:
            memory_shards = [
                MemoryShard(
                    id="welcome",
                    text="HoloLoom is a neural decision-making system with a complete weaving architecture.",
                    episode="system",
                    entities=["HoloLoom", "weaving", "architecture"],
                    motifs=["SYSTEM", "INFO"]
                )
            ]

        # Initialize shuttle with reflection enabled
        self.shuttle = WeavingShuttle(
            cfg=config,
            shards=memory_shards,
            enable_reflection=True,
            reflection_capacity=1000
        )

        # Track spacetime artifacts by query hash
        self.spacetime_history: Dict[str, Any] = {}

        logger.info("HoloLoom handlers initialized with reflection loop")

    # ========================================================================
    # Core Commands
    # ========================================================================

    async def handle_weave(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Execute query through full weaving cycle with reflection.

        Usage: !weave What is Thompson Sampling?

        Supports reactions for feedback:
        - 👍 = helpful (positive feedback)
        - 👎 = not helpful (negative feedback)
        - ⭐ = excellent (high reward)
        """
        if not args:
            await self.bot.send_message(
                room.room_id,
                "Usage: `!weave <your query>`\n\nExample: `!weave Explain Thompson Sampling`",
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

            # Execute weaving cycle with reflection
            query = Query(text=args)
            spacetime = await self.shuttle.weave_and_reflect(
                query,
                feedback={"source": "matrix", "user": event.sender, "room": room.room_id}
            )

            # Store spacetime for trace command
            query_id = f"{event.sender}_{datetime.now().timestamp()}"
            self.spacetime_history[query_id] = spacetime

            # Format response with rich Spacetime details
            response = f"""
**✨ Weaving Complete**

**Query:** {args}

**Decision:**
• Tool: `{spacetime.tool_used}`
• Confidence: {spacetime.confidence:.1%}
• Duration: {spacetime.trace.duration_ms:.0f}ms

**Context:**
• Shards retrieved: {spacetime.trace.context_shards_count}
• Motifs detected: {len(spacetime.trace.motifs_detected)}
• Embedding scales: {spacetime.trace.embedding_scales_used}
• Threads activated: {len(spacetime.trace.threads_activated)}

**Result:**
{spacetime.response[:500] if spacetime.response else 'No output'}

*React with 👍/👎/⭐ to provide feedback*
*Use `!trace {query_id}` for full Spacetime trace*
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

                # Add to memory (create new shard and add to yarn graph)
                new_shard = MemoryShard(
                    id=f"matrix_{datetime.now().timestamp()}",
                    text=content,
                    episode="matrix_chat",
                    entities=[],  # Could extract entities here
                    motifs=[],
                    metadata={"source": "matrix", "user": event.sender, "room": room.room_id}
                )
                self.shuttle.yarn_graph.shards[new_shard.id] = new_shard

                await self.bot.send_message(
                    room.room_id,
                    f"✅ **Added to memory**\n\n`{content[:200]}`",
                    markdown=True
                )

            elif action == "search":
                if not content:
                    await self.bot.send_message(room.room_id, "Usage: `!memory search <query>`")
                    return

                # Search memory using retriever
                query_obj = Query(text=content)
                results = await self.shuttle.retriever.retrieve(query_obj, limit=5)

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
                total_shards = len(self.shuttle.yarn_graph.shards)
                reflection_metrics = self.shuttle.get_reflection_metrics()

                success_rate = reflection_metrics.get('success_rate', 0) if reflection_metrics else 0
                response = f"""
**📊 Memory Statistics**

**Yarn Graph:**
• Total threads (shards): {total_shards}
• Backend: In-memory (Yarn Graph)

**Reflection:**
• Total cycles: {reflection_metrics.get('total_cycles', 0) if reflection_metrics else 0}
• Success rate: {success_rate:.1%}

**System:**
• Mode: {self.shuttle.cfg.mode.value}
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
        Analyze text with convergence engine.

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
            query = Query(text=f"Analyze the following: {args}")
            spacetime = await self.shuttle.weave(query)

            response = f"""
**🔬 Analysis Results**

**Input:** {args[:200]}

**Decision:** {spacetime.tool_used} ({spacetime.confidence:.1%} confidence)

**Weaving:**
• Duration: {spacetime.trace.duration_ms:.0f}ms
• Context: {spacetime.trace.context_shards_count} shards
• Motifs: {len(spacetime.trace.motifs_detected)}
• Scales: {spacetime.trace.embedding_scales_used}
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Analyze error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_stats(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show HoloLoom system statistics including reflection metrics.

        Usage: !stats
        """
        try:
            # Get reflection metrics
            reflection_metrics = self.shuttle.get_reflection_metrics()

            # Build reflection info
            reflection_info = ""
            if reflection_metrics:
                tool_success = reflection_metrics.get('tool_success_rates', {})
                tool_recs = reflection_metrics.get('tool_recommendations', [])

                reflection_info = f"""
**Reflection Loop:**
• Total cycles: {reflection_metrics.get('total_cycles', 0)}
• Success rate: {reflection_metrics.get('success_rate', 0):.1%}
• Learning status: ✅ Active

**Tool Performance:**
"""
                for tool, rate in list(tool_success.items())[:3]:
                    reflection_info += f"• {tool}: {rate:.1%}\n"

                if tool_recs:
                    reflection_info += f"\n**Recommended:** {', '.join(tool_recs[:3])}"

            response = f"""
**📈 HoloLoom Statistics**

**System:**
• Status: ✅ Operational
• Mode: {self.shuttle.cfg.mode.value}
• Reflection: {'Enabled' if self.shuttle.enable_reflection else 'Disabled'}

{reflection_info}

**Architecture:**
• Full 9-step weaving cycle
• Thompson Sampling exploration
• Multi-scale embeddings
• Spacetime provenance tracking
"""

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Stats error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    async def handle_trace(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show full Spacetime trace for a query.

        Usage: !trace [query_id]
        Shows the most recent trace if no ID provided.
        """
        try:
            # Get trace - either latest or by ID
            if args and args in self.spacetime_history:
                spacetime = self.spacetime_history[args]
            elif self.spacetime_history:
                # Get most recent
                spacetime = list(self.spacetime_history.values())[-1]
            else:
                await self.bot.send_message(
                    room.room_id,
                    "No traces available. Use `!weave` first."
                )
                return

            trace = spacetime.trace

            # Format detailed trace
            response = f"""
**🔍 Spacetime Trace**

**Query:** {spacetime.query_text}

**Weaving Cycle (9 steps):**
1. Pattern: `{trace.pattern_used}` selected
2. Temporal window created
3. Threads: {len(trace.threads_activated)} activated
4. Features extracted:
   • Motifs: {', '.join(trace.motifs_detected[:5])}
   • Scales: {trace.embedding_scales_used}
5. Warp space: Tensioned
6. Context: {trace.context_shards_count} shards retrieved
7. Convergence: `{spacetime.tool_used}` (confidence {spacetime.confidence:.1%})
8. Tool executed
9. Spacetime woven

**Stage Timings:**
"""
            for stage, duration in list(trace.stage_durations.items())[:6]:
                response += f"• {stage}: {duration:.0f}ms\n"

            response += f"\n**Total Duration:** {trace.duration_ms:.0f}ms"

            # Add bandit stats if available
            if hasattr(trace, 'bandit_statistics') and trace.bandit_statistics:
                response += "\n\n**Bandit Statistics:**\n"
                for tool, stats in list(trace.bandit_statistics.items())[:3]:
                    response += f"• {tool}: {stats}\n"

            await self.bot.send_message(room.room_id, response, markdown=True)

        except Exception as e:
            logger.error(f"Trace error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")

    async def handle_learn(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Trigger learning analysis and show insights.

        Usage: !learn [force]
        Use 'force' to analyze even if not enough cycles have passed.
        """
        try:
            if not self.shuttle.enable_reflection:
                await self.bot.send_message(
                    room.room_id,
                    "⚠️ Reflection loop is disabled. Enable it to use learning."
                )
                return

            await self.bot.send_typing(room.room_id, typing=True)

            # Analyze and generate learning signals
            force = args.lower() == "force"
            signals = await self.shuttle.learn(force=force)

            if not signals:
                await self.bot.send_message(
                    room.room_id,
                    "📚 No new learning signals yet. Need more cycles for analysis."
                )
                return

            # Format learning insights
            response = f"""
**🧠 Learning Analysis**

Generated {len(signals)} learning signals:

"""

            for signal in signals[:5]:  # Show top 5
                response += f"**{signal.signal_type}**\n"
                if signal.tool:
                    response += f"• Tool: `{signal.tool}`\n"
                if hasattr(signal, 'reward') and signal.reward is not None:
                    response += f"• Reward: {signal.reward:.2f}\n"
                if signal.pattern:
                    response += f"• Pattern: {signal.pattern}\n"
                if signal.recommendation:
                    response += f"• Action: {signal.recommendation}\n"
                response += "\n"

            response += f"*Applying {len(signals)} learning signals to improve system...*"

            await self.bot.send_message(room.room_id, response, markdown=True)

            # Apply signals
            await self.shuttle.apply_learning_signals(signals)

            await self.bot.send_message(
                room.room_id,
                "✅ **Learning complete!** System has adapted based on past performance.",
                markdown=True
            )

        except Exception as e:
            logger.error(f"Learn error: {e}", exc_info=True)
            await self.bot.send_message(room.room_id, f"❌ **Error:** {str(e)}")
        finally:
            await self.bot.send_typing(room.room_id, typing=False)

    async def handle_help(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Show help for HoloLoom commands.

        Usage: !help
        """
        help_text = """
**🔮 HoloLoom Bot - Commands**

**Core:**
• `!weave <query>` - Execute full weaving cycle with reflection
• `!analyze <text>` - Analyze with convergence engine
• `!trace [id]` - Show Spacetime trace (full provenance)
• `!learn [force]` - Trigger learning analysis
• `!stats` - System statistics with reflection metrics

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
!trace  # Show most recent trace
!learn force  # Force learning analysis
!stats  # View reflection metrics
!memory add MCTS balances exploration vs exploitation
```

**Powered by:**
• Full 9-step weaving architecture
• Reflection loop (continuous learning)
• Thompson Sampling exploration
• Multi-scale embeddings (Matryoshka)
• Spacetime provenance tracking
• Convergence engine decision-making
"""

        await self.bot.send_message(room.room_id, help_text, markdown=True)

    async def handle_ping(self, room: MatrixRoom, event: RoomMessageText, args: str):
        """
        Simple ping command.

        Usage: !ping
        """
        reflection_metrics = self.shuttle.get_reflection_metrics()
        total_cycles = reflection_metrics.get('total_cycles', 0) if reflection_metrics else 0

        response = f"""
**🏓 Pong!**

HoloLoom is operational.
• Weaving cycles: {total_cycles}
• Reflection: {'✅ Active' if self.shuttle.enable_reflection else '⚠️ Disabled'}
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
        self.bot.register_handler("trace", self.handle_trace)
        self.bot.register_handler("learn", self.handle_learn)
        self.bot.register_handler("stats", self.handle_stats)
        self.bot.register_handler("help", self.handle_help)
        self.bot.register_handler("ping", self.handle_ping)

        logger.info("All HoloLoom handlers registered (8 commands)")

    async def shutdown(self):
        """Clean shutdown with proper lifecycle management."""
        logger.info("Shutting down HoloLoom WeavingShuttle...")

        # Close shuttle (cancels tasks, flushes reflection buffer)
        await self.shuttle.close()

        logger.info("HoloLoom handlers shut down successfully")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
HoloLoom Matrix Bot Handlers
=============================

This module provides command handlers that integrate
Matrix chatops with HoloLoom WeavingShuttle.

Commands available:
- !weave <query> - Full weaving cycle with reflection
- !trace [id] - Show Spacetime trace with full provenance
- !learn [force] - Trigger learning analysis
- !memory add/search/stats - Memory management
- !analyze <text> - Convergence engine analysis
- !stats - System statistics with reflection metrics
- !help - Command help
- !ping - Health check

Features:
- Full 9-step weaving architecture
- Reflection loop for continuous learning
- Thompson Sampling exploration
- Spacetime provenance tracking
- Multi-scale embeddings (Matryoshka)

To use:
    from HoloLoom.chatops.core.matrix_bot import MatrixBot
    from HoloLoom.chatops.handlers.hololoom_handlers import HoloLoomMatrixHandlers

    bot = MatrixBot(config)
    handlers = HoloLoomMatrixHandlers(bot, config_mode="fast")
    handlers.register_all()
    await bot.start()
""")
