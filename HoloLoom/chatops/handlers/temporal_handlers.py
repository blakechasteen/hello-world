"""
Temporal Handlers for Matrix ChatOps
=====================================

Matrix command handlers for HoloLoom temporal queries - time-travel,
time-range queries, and pattern detection.

Commands:
- !temporal travel <timestamp> - View memory state at specific time
- !temporal between <start> <end> - Query memories in time range
- !temporal patterns [min] [days] - Detect recurring patterns
- !temporal help - Temporal command help

Usage:
    from HoloLoom.chatops.handlers.temporal_handlers import register_temporal_handlers

    # In run_chatops.py:
    register_temporal_handlers(bot, memory)

Created: 2025-12-06
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from HoloLoom.memory.unified import UnifiedMemory

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# UnifiedMemory imports
try:
    from HoloLoom.memory.unified import UnifiedMemory, Memory
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False
    UnifiedMemory = None
    Memory = None

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
# Timestamp Parsing Helpers
# ============================================================================

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse timestamp string to datetime.

    Supports:
    - ISO format: YYYY-MM-DDTHH:MM:SS
    - Date only: YYYY-MM-DD
    - Relative shortcuts: now, today, yesterday, week-ago, month-ago

    Args:
        timestamp_str: Timestamp string to parse

    Returns:
        datetime object or None if invalid
    """
    timestamp_str = timestamp_str.strip().lower()
    now = datetime.now()

    # Relative shortcuts
    relative_shortcuts = {
        "now": now,
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "yesterday": now - timedelta(days=1),
        "week-ago": now - timedelta(days=7),
        "2-weeks-ago": now - timedelta(days=14),
        "month-ago": now - timedelta(days=30),
        "3-months-ago": now - timedelta(days=90),
        "year-ago": now - timedelta(days=365),
    }

    if timestamp_str in relative_shortcuts:
        return relative_shortcuts[timestamp_str]

    # Try ISO format
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        pass

    # Try date-only format
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d")
    except ValueError:
        pass

    return None


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time (e.g., '2 days ago')."""
    now = datetime.now()
    diff = now - dt

    if diff.days == 0:
        hours = diff.seconds // 3600
        if hours == 0:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago" if minutes > 1 else "just now"
        return f"{hours} hours ago" if hours > 1 else "1 hour ago"
    elif diff.days == 1:
        return "yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    elif diff.days < 30:
        weeks = diff.days // 7
        return f"{weeks} weeks ago" if weeks > 1 else "1 week ago"
    elif diff.days < 365:
        months = diff.days // 30
        return f"{months} months ago" if months > 1 else "1 month ago"
    else:
        years = diff.days // 365
        return f"{years} years ago" if years > 1 else "1 year ago"


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_temporal_travel(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: 'UnifiedMemory'
) -> str:
    """
    Handle !temporal travel command - View memory state at specific time.

    Args:
        room: Matrix room
        event: Message event
        args: Timestamp (ISO format or relative)
        memory: UnifiedMemory instance

    Returns:
        Memory snapshot at that time
    """
    if not UNIFIED_MEMORY_AVAILABLE:
        return "❌ UnifiedMemory not available"

    if not args.strip():
        return """❌ **Usage:** `!temporal travel <timestamp>`

**Timestamp formats:**
• ISO: `2025-11-20T10:30:00` or `2025-11-20`
• Relative: `yesterday`, `week-ago`, `month-ago`

**Examples:**
```
!temporal travel yesterday
!temporal travel week-ago
!temporal travel 2025-11-01
!temporal travel 2025-11-20T14:30:00
```"""

    timestamp_str = args.strip()

    # Parse timestamp
    dt = parse_timestamp(timestamp_str)
    if not dt:
        return f"❌ Invalid timestamp: `{timestamp_str}`\n\n_Use ISO format (YYYY-MM-DDTHH:MM:SS) or relative (yesterday, week-ago, month-ago)_"

    try:
        # Call time_travel method
        snapshot = memory.time_travel(dt.isoformat())

        # Check for error
        if 'error' in snapshot:
            return f"❌ {snapshot['error']}"

        # Format response
        response = f"⏰ **Time Travel: {format_timestamp(dt)}**\n\n"
        response += f"_({format_relative_time(dt)})_\n\n"

        # Stats
        response += "**Memory State:**\n"
        response += f"  Total memories: {snapshot['total_memories']}\n"
        response += f"  Graph nodes: {snapshot['graph_stats']['nodes']}\n"
        response += f"  Graph edges: {snapshot['graph_stats']['edges']}\n\n"

        # Sample memories
        memories = snapshot.get('memories', [])
        if memories:
            response += f"**Sample Memories ({min(5, len(memories))} shown):**\n"
            for i, mem in enumerate(memories[:5], 1):
                text = mem.text if hasattr(mem, 'text') else str(mem)
                if len(text) > 80:
                    text = text[:80] + "..."
                response += f"{i}. {text}\n"

            if len(memories) > 5:
                response += f"\n_...and {len(memories) - 5} more_\n"
        else:
            response += "_No memories found at this time._\n"

        return response

    except Exception as e:
        logger.error(f"Error in temporal travel: {e}")
        return f"❌ Time travel failed: {str(e)}"


async def handle_temporal_between(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: 'UnifiedMemory'
) -> str:
    """
    Handle !temporal between command - Query memories in time range.

    Args:
        room: Matrix room
        event: Message event
        args: Start and end timestamps
        memory: UnifiedMemory instance

    Returns:
        List of memories in time range
    """
    if not UNIFIED_MEMORY_AVAILABLE:
        return "❌ UnifiedMemory not available"

    if not args.strip():
        return """❌ **Usage:** `!temporal between <start> <end> [limit]`

**Examples:**
```
!temporal between yesterday now
!temporal between week-ago today
!temporal between 2025-11-01 2025-11-30
!temporal between month-ago today 50
```"""

    # Parse arguments
    parts = args.strip().split()

    if len(parts) < 2:
        return "❌ Need both start and end timestamps.\n\n_Example: `!temporal between yesterday now`_"

    start_str = parts[0]
    end_str = parts[1]
    limit = 20

    if len(parts) >= 3:
        try:
            limit = int(parts[2])
            limit = min(limit, 100)  # Cap at 100
        except ValueError:
            pass

    # Parse timestamps
    start_dt = parse_timestamp(start_str)
    end_dt = parse_timestamp(end_str)

    if not start_dt:
        return f"❌ Invalid start timestamp: `{start_str}`"
    if not end_dt:
        return f"❌ Invalid end timestamp: `{end_str}`"

    # Ensure start < end
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    try:
        # Call what_happened_between method
        memories = memory.what_happened_between(
            start_dt.isoformat(),
            end_dt.isoformat(),
            limit=limit
        )

        # Format response
        response = f"📅 **Memories: {format_timestamp(start_dt)} → {format_timestamp(end_dt)}**\n\n"

        if not memories:
            response += "_No memories found in this time range._"
            return response

        response += f"**Found {len(memories)} memories:**\n\n"

        for i, mem in enumerate(memories[:20], 1):
            # Get memory text
            text = mem.text if hasattr(mem, 'text') else str(mem)
            if len(text) > 100:
                text = text[:100] + "..."

            # Get timestamp
            ts = mem.timestamp if hasattr(mem, 'timestamp') else "?"
            if ts and ts != "?":
                try:
                    ts_dt = datetime.fromisoformat(ts)
                    ts = format_relative_time(ts_dt)
                except:
                    pass

            response += f"**{i}.** {text}\n"
            response += f"   _({ts})_\n\n"

        if len(memories) > 20:
            response += f"_...and {len(memories) - 20} more. Use higher limit to see more._\n"

        return response

    except Exception as e:
        logger.error(f"Error in temporal between: {e}")
        return f"❌ Query failed: {str(e)}"


async def handle_temporal_patterns(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    memory: 'UnifiedMemory'
) -> str:
    """
    Handle !temporal patterns command - Detect recurring patterns.

    Args:
        room: Matrix room
        event: Message event
        args: Optional min_occurrences and time_window_days
        memory: UnifiedMemory instance

    Returns:
        List of detected temporal patterns
    """
    if not UNIFIED_MEMORY_AVAILABLE:
        return "❌ UnifiedMemory not available"

    # Parse arguments
    min_occurrences = 2
    time_window_days = 7

    parts = args.strip().split()

    if len(parts) >= 1:
        try:
            min_occurrences = int(parts[0])
            min_occurrences = max(2, min(min_occurrences, 100))  # Clamp 2-100
        except ValueError:
            pass

    if len(parts) >= 2:
        try:
            time_window_days = int(parts[1])
            time_window_days = max(1, min(time_window_days, 365))  # Clamp 1-365
        except ValueError:
            pass

    try:
        # Call detect_temporal_patterns method
        patterns = memory.detect_temporal_patterns(
            min_occurrences=min_occurrences,
            time_window_days=time_window_days
        )

        # Format response
        response = f"🔍 **Temporal Patterns** (min: {min_occurrences}, window: {time_window_days} days)\n\n"

        if not patterns:
            response += "_No patterns detected. Try lowering min_occurrences or increasing time_window._"
            return response

        # Group patterns by type
        recurring = [p for p in patterns if p['pattern_type'] == 'recurring_topic']
        clusters = [p for p in patterns if p['pattern_type'] == 'temporal_cluster']

        if recurring:
            response += f"**📚 Recurring Topics ({len(recurring)}):**\n"
            for i, pattern in enumerate(recurring[:5], 1):
                desc = pattern['description']
                occurrences = pattern['occurrences']
                time_span = pattern.get('time_span', 'N/A')

                response += f"\n{i}. **{desc}**\n"
                response += f"   Occurrences: {occurrences} | Span: {time_span}\n"

                # Sample memories
                mems = pattern.get('memories', [])
                if mems:
                    sample_text = mems[0].text if hasattr(mems[0], 'text') else str(mems[0])
                    if len(sample_text) > 60:
                        sample_text = sample_text[:60] + "..."
                    response += f"   _Example: \"{sample_text}\"_\n"

            if len(recurring) > 5:
                response += f"\n_...and {len(recurring) - 5} more recurring topics_\n"

        if clusters:
            response += f"\n**⚡ Activity Clusters ({len(clusters)}):**\n"
            for i, pattern in enumerate(clusters[:5], 1):
                desc = pattern['description']
                occurrences = pattern['occurrences']
                time_span = pattern.get('time_span', 'N/A')

                response += f"\n{i}. **{desc}**\n"
                response += f"   Span: {time_span}\n"

            if len(clusters) > 5:
                response += f"\n_...and {len(clusters) - 5} more clusters_\n"

        return response

    except Exception as e:
        logger.error(f"Error in temporal patterns: {e}")
        return f"❌ Pattern detection failed: {str(e)}"


async def handle_temporal_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !temporal help command.

    Returns:
        Help message
    """
    return """⏰ **Temporal Query Commands**

**Time Travel:**
• `!temporal travel <timestamp>` - View memory state at specific time
• Timestamps: ISO format or relative (yesterday, week-ago, month-ago)

**Time Range Queries:**
• `!temporal between <start> <end>` - Query memories in time range
• `!temporal between <start> <end> <limit>` - With result limit

**Pattern Detection:**
• `!temporal patterns` - Detect recurring patterns (default: min 2, window 7 days)
• `!temporal patterns <min>` - Set minimum occurrences
• `!temporal patterns <min> <days>` - Set min occurrences and time window

**Timestamp Formats:**
• ISO: `2025-11-20T14:30:00` or `2025-11-20`
• Relative: `now`, `today`, `yesterday`, `week-ago`, `month-ago`

**Pattern Types:**
• **Recurring Topics** - Same themes appearing multiple times
• **Temporal Clusters** - Bursts of activity in time windows

**Examples:**
```
!temporal travel yesterday
!temporal travel 2025-11-01
!temporal between week-ago now
!temporal between 2025-11-01 2025-11-30 50
!temporal patterns
!temporal patterns 3 14
```
"""


# ============================================================================
# Registration
# ============================================================================

def register_temporal_handlers(
    bot,
    memory: 'UnifiedMemory'
) -> None:
    """
    Register all Temporal command handlers.

    Args:
        bot: Matrix bot instance
        memory: UnifiedMemory instance
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, temporal handlers not registered")
        return

    if not UNIFIED_MEMORY_AVAILABLE:
        logger.warning("UnifiedMemory not available, temporal handlers not registered")
        return

    logger.info("Registering Temporal command handlers")

    # Define command map
    command_map = {
        "travel": lambda room, event, args: handle_temporal_travel(room, event, args, memory),
        "between": lambda room, event, args: handle_temporal_between(room, event, args, memory),
        "patterns": lambda room, event, args: handle_temporal_patterns(room, event, args, memory),
        "help": handle_temporal_help
    }

    # Register with bot under !temporal prefix
    for cmd, handler in command_map.items():
        bot.register_command(f"temporal {cmd}", handler)

    # Also register !temporal with no args as help
    bot.register_command("temporal", handle_temporal_help)

    logger.info(f"Registered {len(command_map)} Temporal commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class TemporalHandlers:
    """
    Decorator-based ChatOps handlers for Temporal queries.

    Usage:
        from HoloLoom.chatops.handlers.temporal_handlers import TemporalHandlers

        handlers = TemporalHandlers(memory=unified_memory)
        registry.register_instance(handlers)
    """

    def __init__(self, memory: 'UnifiedMemory'):
        """
        Initialize with UnifiedMemory instance.

        Args:
            memory: UnifiedMemory instance for temporal queries
        """
        self.memory = memory

    @HandlerRegistry.register(
        command="temporal travel",
        description="View memory state at specific time",
        usage="!temporal travel <timestamp>",
        category=HandlerCategory.QUERY,
        aliases=["tt"]
    )
    async def handle_travel(self, room, event, args: str) -> str:
        """Handle !temporal travel command."""
        return await handle_temporal_travel(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="temporal between",
        description="Query memories in time range",
        usage="!temporal between <start> <end> [limit]",
        category=HandlerCategory.QUERY,
        aliases=["tb"]
    )
    async def handle_between(self, room, event, args: str) -> str:
        """Handle !temporal between command."""
        return await handle_temporal_between(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="temporal patterns",
        description="Detect recurring temporal patterns",
        usage="!temporal patterns [min_occurrences] [time_window_days]",
        category=HandlerCategory.QUERY,
        aliases=["tp"]
    )
    async def handle_patterns(self, room, event, args: str) -> str:
        """Handle !temporal patterns command."""
        return await handle_temporal_patterns(room, event, args, self.memory)

    @HandlerRegistry.register(
        command="temporal help",
        description="Show temporal command help",
        usage="!temporal help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    )
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !temporal help command."""
        return await handle_temporal_help(room, event, args)
