"""
Memory Symphony Handlers for Matrix ChatOps
=============================================

Matrix command handlers for HoloLoom Memory Symphony - intelligent
multi-system memory coordination.

Commands:
- !memory strategy <name> - Switch memory strategy (FAST/BALANCED/DEEP/RESEARCH/AUTO)
- !memory metrics - Show coordination metrics (latency, cache hits, system usage)
- !memory systems - List available memory systems and status
- !memory history - Show recent recall history with strategies used
- !memory help - Memory symphony command help

Usage:
    from HoloLoom.chatops.handlers.memory_symphony_handlers import register_memory_symphony_handlers

    # In run_chatops.py:
    register_memory_symphony_handlers(bot, conductor)

Created: 2025-12-05
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.memory_symphony.conductor import MemoryConductor
    from HoloLoom.memory_symphony.protocol import MemoryStrategy

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Memory Symphony imports
try:
    from HoloLoom.memory_symphony.conductor import MemoryConductor
    from HoloLoom.memory_symphony.protocol import (
        MemoryStrategy,
        MemorySystem,
        MemoryQuery,
        MemoryPerformanceMetrics
    )
    SYMPHONY_AVAILABLE = True
except ImportError:
    SYMPHONY_AVAILABLE = False
    MemoryConductor = None
    MemoryStrategy = None
    MemorySystem = None
    MemoryQuery = None
    MemoryPerformanceMetrics = None

# Handler registry
try:
    from HoloLoom.chatops.handlers.handler_registry import (
        HandlerRegistry, HandlerCategory, chatops_handler
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_memory_strategy(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    conductor: 'MemoryConductor'
) -> str:
    """
    Handle !memory strategy command - Switch or view memory strategy.

    Args:
        room: Matrix room
        event: Message event
        args: Strategy name (or empty to view current)
        conductor: MemoryConductor instance

    Returns:
        Current strategy or confirmation of change
    """
    if not SYMPHONY_AVAILABLE:
        return "❌ Memory Symphony not available"

    strategy_name = args.strip().upper() if args.strip() else None

    # If no strategy specified, show current
    if not strategy_name:
        current = conductor.default_strategy
        strategies = [s.value for s in MemoryStrategy]

        response = f"🎼 **Current Memory Strategy: {current.value.upper()}**\n\n"
        response += "**Available Strategies:**\n"

        strategy_descriptions = {
            "FAST": "Cache-first, minimal graph traversal (~45ms)",
            "BALANCED": "Hybrid: cache + vector + graph (~125ms)",
            "DEEP": "Full graph traversal + spreading activation (~275ms)",
            "RESEARCH": "Maximum exploration, all systems (~400ms)",
            "AUTO": "Automatic selection based on query"
        }

        for s in MemoryStrategy:
            marker = "→" if s == current else " "
            desc = strategy_descriptions.get(s.value.upper(), "")
            response += f"{marker} `{s.value.upper()}` - {desc}\n"

        response += "\n_Use `!memory strategy <name>` to change_"
        return response

    # Try to set new strategy
    try:
        new_strategy = MemoryStrategy(strategy_name.lower())
        conductor.default_strategy = new_strategy

        response = f"✅ **Strategy Changed: {new_strategy.value.upper()}**\n\n"

        # Show what systems will be used
        system_usage = {
            MemoryStrategy.FAST: ["QUERY_CACHE", "VECTOR_MEMORY", "HOT_PATTERNS"],
            MemoryStrategy.BALANCED: ["QUERY_CACHE", "VECTOR_MEMORY", "KNOWLEDGE_GRAPH", "HOT_PATTERNS"],
            MemoryStrategy.DEEP: ["All systems + spreading activation"],
            MemoryStrategy.RESEARCH: ["All 7 systems, maximum exploration"],
            MemoryStrategy.AUTO: ["Automatic based on query complexity"]
        }

        systems = system_usage.get(new_strategy, [])
        response += f"**Systems accessed:** {', '.join(systems)}"

        return response

    except (ValueError, KeyError):
        valid = [s.value.upper() for s in MemoryStrategy]
        return f"❌ Unknown strategy: `{strategy_name}`\n\n**Valid strategies:** {', '.join(valid)}"


async def handle_memory_metrics(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    conductor: 'MemoryConductor'
) -> str:
    """
    Handle !memory metrics command - Show coordination metrics.

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        conductor: MemoryConductor instance

    Returns:
        Performance metrics message
    """
    if not SYMPHONY_AVAILABLE:
        return "❌ Memory Symphony not available"

    try:
        metrics = conductor.get_performance_metrics()

        response = "📊 **Memory Symphony Metrics**\n\n"

        # Query stats
        response += "**Query Statistics:**\n"
        response += f"  Total queries: {metrics.total_queries}\n"
        response += f"  Cache hits: {metrics.cache_hits}\n"
        response += f"  Cache misses: {metrics.cache_misses}\n"

        if metrics.total_queries > 0:
            hit_rate = metrics.cache_hits / metrics.total_queries
            response += f"  Hit rate: {hit_rate:.1%}\n"

        response += f"  Avg latency: {metrics.avg_latency_ms:.1f}ms\n"
        response += f"  Avg results/query: {metrics.avg_results_per_query:.1f}\n"

        # Strategy usage
        if metrics.strategy_usage:
            response += "\n**Strategy Usage:**\n"
            total = sum(metrics.strategy_usage.values())
            for strategy, count in sorted(metrics.strategy_usage.items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                response += f"  {strategy.value.upper()}: {bar} {count} ({pct:.0f}%)\n"

        # System usage
        if metrics.system_usage:
            response += "\n**System Usage:**\n"
            total = sum(metrics.system_usage.values())
            for system, count in sorted(metrics.system_usage.items(), key=lambda x: -x[1]):
                pct = (count / total * 100) if total > 0 else 0
                response += f"  {system.value}: {count} accesses ({pct:.0f}%)\n"

        return response

    except Exception as e:
        logger.error(f"Error in memory metrics: {e}")
        return f"❌ Failed to get metrics: {str(e)}"


async def handle_memory_systems(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    conductor: 'MemoryConductor'
) -> str:
    """
    Handle !memory systems command - List available memory systems.

    Args:
        room: Matrix room
        event: Message event
        args: (ignored)
        conductor: MemoryConductor instance

    Returns:
        List of memory systems with status
    """
    if not SYMPHONY_AVAILABLE:
        return "❌ Memory Symphony not available"

    try:
        response = "🧠 **Memory Systems**\n\n"

        # System descriptions
        system_info = {
            MemorySystem.QUERY_CACHE: {
                "desc": "100x speedup for repeated queries",
                "icon": "⚡"
            },
            MemorySystem.VECTOR_MEMORY: {
                "desc": "Semantic similarity search",
                "icon": "🔢"
            },
            MemorySystem.KNOWLEDGE_GRAPH: {
                "desc": "Entity relationships (Yarn Graph)",
                "icon": "🕸️"
            },
            MemorySystem.HOT_PATTERNS: {
                "desc": "Usage-based adaptation",
                "icon": "🔥"
            },
            MemorySystem.AWARENESS_GRAPH: {
                "desc": "Activation tracking & spreading",
                "icon": "👁️"
            },
            MemorySystem.SPRING_DYNAMICS: {
                "desc": "Physics-based connectivity",
                "icon": "🌀"
            },
            MemorySystem.MULTI_WAVE: {
                "desc": "Temporal wave propagation",
                "icon": "🌊"
            }
        }

        # Check which systems are enabled
        enabled_cache = getattr(conductor, 'enable_cache', True)
        enabled_hot = getattr(conductor, 'enable_hot_patterns', True)
        enabled_awareness = getattr(conductor, 'enable_awareness', True)

        for system in MemorySystem:
            info = system_info.get(system, {"desc": "Memory system", "icon": "📦"})

            # Determine status
            if system == MemorySystem.QUERY_CACHE:
                status = "✅ Enabled" if enabled_cache else "⏸️ Disabled"
            elif system == MemorySystem.HOT_PATTERNS:
                status = "✅ Enabled" if enabled_hot else "⏸️ Disabled"
            elif system == MemorySystem.AWARENESS_GRAPH:
                status = "✅ Enabled" if enabled_awareness else "⏸️ Disabled"
            else:
                status = "✅ Available"

            response += f"{info['icon']} **{system.value}**\n"
            response += f"   {info['desc']}\n"
            response += f"   Status: {status}\n\n"

        # Current strategy info
        current = conductor.default_strategy
        response += f"_Current strategy: {current.value.upper()}_"

        return response

    except Exception as e:
        logger.error(f"Error in memory systems: {e}")
        return f"❌ Failed to get systems: {str(e)}"


async def handle_memory_history(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    conductor: 'MemoryConductor'
) -> str:
    """
    Handle !memory history command - Show recent recall history.

    Args:
        room: Matrix room
        event: Message event
        args: Optional limit (default 5)
        conductor: MemoryConductor instance

    Returns:
        Recent query history with strategies used
    """
    if not SYMPHONY_AVAILABLE:
        return "❌ Memory Symphony not available"

    # Parse limit
    limit = 5
    if args.strip():
        try:
            limit = int(args.strip())
            limit = min(limit, 20)  # Cap at 20
        except ValueError:
            pass

    try:
        history = conductor.get_query_history(limit=limit)

        if not history:
            return "📜 **Memory History**\n\n_No queries recorded yet._"

        response = f"📜 **Memory History** (last {len(history)})\n\n"

        for i, result in enumerate(history, 1):
            # Query text (truncated)
            query_text = result.query.text if hasattr(result, 'query') else "Unknown"
            if len(query_text) > 50:
                query_text = query_text[:50] + "..."

            # Strategy used
            strategy = result.strategy_used.value.upper() if hasattr(result, 'strategy_used') else "?"

            # Systems accessed
            systems = []
            if hasattr(result, 'systems_accessed'):
                systems = [s.value for s in result.systems_accessed[:3]]
            systems_str = ", ".join(systems) if systems else "N/A"

            # Results count
            count = len(result.results) if hasattr(result, 'results') else 0

            # Latency
            latency = result.total_latency_ms if hasattr(result, 'total_latency_ms') else 0

            response += f"**{i}.** `{query_text}`\n"
            response += f"   Strategy: {strategy} | Results: {count} | {latency:.0f}ms\n"
            response += f"   Systems: {systems_str}\n\n"

        return response

    except Exception as e:
        logger.error(f"Error in memory history: {e}")
        return f"❌ Failed to get history: {str(e)}"


async def handle_memory_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !memory help command.

    Returns:
        Help message
    """
    return """🎼 **Memory Symphony Commands**

**Strategy Control:**
• `!memory strategy` - View current strategy and options
• `!memory strategy <name>` - Switch strategy (FAST/BALANCED/DEEP/RESEARCH/AUTO)

**Monitoring:**
• `!memory metrics` - Show performance metrics (latency, cache hits, usage)
• `!memory systems` - List all 7 memory systems with status
• `!memory history [N]` - Show last N queries with strategies used

**Memory Strategies:**
• `FAST` - Cache-first, ~45ms, 2-3 systems
• `BALANCED` - Hybrid approach, ~125ms, 3-4 systems
• `DEEP` - Full traversal, ~275ms, 5-7 systems
• `RESEARCH` - Maximum exploration, ~400ms, all systems
• `AUTO` - Automatic selection based on query complexity

**7 Memory Systems:**
1. Query Cache - 100x speedup for repeated queries
2. Vector Memory - Semantic similarity search
3. Knowledge Graph - Entity relationships
4. Hot Patterns - Usage-based adaptation
5. Awareness Graph - Activation tracking
6. Spring Dynamics - Physics-based connectivity
7. Multi-Wave - Temporal propagation

**Examples:**
```
!memory strategy
!memory strategy DEEP
!memory metrics
!memory systems
!memory history 10
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_memory_symphony_handlers(
    bot,
    conductor: 'MemoryConductor'
) -> None:
    """
    Register all Memory Symphony command handlers.

    Args:
        bot: Matrix bot instance
        conductor: MemoryConductor instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, memory symphony handlers not registered")
        return

    if not SYMPHONY_AVAILABLE:
        logger.warning("Memory Symphony not available, handlers not registered")
        return

    logger.info("Registering Memory Symphony command handlers")

    # Define command map
    command_map = {
        "strategy": lambda room, event, args: handle_memory_strategy(room, event, args, conductor),
        "metrics": lambda room, event, args: handle_memory_metrics(room, event, args, conductor),
        "systems": lambda room, event, args: handle_memory_systems(room, event, args, conductor),
        "history": lambda room, event, args: handle_memory_history(room, event, args, conductor),
        "help": handle_memory_help
    }

    # Register with bot under !memory prefix
    for cmd, handler in command_map.items():
        bot.register_command(f"memory {cmd}", handler)

    # Also register !memory with no args as help
    bot.register_command("memory", handle_memory_help)

    logger.info(f"Registered {len(command_map)} Memory Symphony commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class MemorySymphonyHandlers:
    """
    Decorator-based ChatOps handlers for Memory Symphony.

    Usage:
        from HoloLoom.chatops.handlers.memory_symphony_handlers import MemorySymphonyHandlers

        handlers = MemorySymphonyHandlers(conductor=memory_conductor)
        registry.register_instance(handlers)
    """

    def __init__(self, conductor: 'MemoryConductor'):
        """
        Initialize with MemoryConductor instance.

        Args:
            conductor: MemoryConductor instance for memory coordination
        """
        self.conductor = conductor

    @HandlerRegistry.register(
        command="memory strategy",
        description="Switch or view memory access strategy",
        usage="!memory strategy [FAST|BALANCED|DEEP|RESEARCH|AUTO]",
        category=HandlerCategory.SYSTEM,
        aliases=["ms"]
    )
    async def handle_strategy(self, room, event, args: str) -> str:
        """Handle !memory strategy command."""
        return await handle_memory_strategy(room, event, args, self.conductor)

    @HandlerRegistry.register(
        command="memory metrics",
        description="Show memory coordination performance metrics",
        usage="!memory metrics",
        category=HandlerCategory.SYSTEM,
        aliases=["mm"]
    )
    async def handle_metrics(self, room, event, args: str) -> str:
        """Handle !memory metrics command."""
        return await handle_memory_metrics(room, event, args, self.conductor)

    @HandlerRegistry.register(
        command="memory systems",
        description="List all memory systems with status",
        usage="!memory systems",
        category=HandlerCategory.SYSTEM,
        aliases=["msys"]
    )
    async def handle_systems(self, room, event, args: str) -> str:
        """Handle !memory systems command."""
        return await handle_memory_systems(room, event, args, self.conductor)

    @HandlerRegistry.register(
        command="memory history",
        description="Show recent recall history with strategies",
        usage="!memory history [limit]",
        category=HandlerCategory.SYSTEM,
        aliases=["mh"]
    )
    async def handle_history(self, room, event, args: str) -> str:
        """Handle !memory history command."""
        return await handle_memory_history(room, event, args, self.conductor)

    @HandlerRegistry.register(
        command="memory help",
        description="Show Memory Symphony command help",
        usage="!memory help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !memory help command."""
        return await handle_memory_help(room, event, args)
