from __future__ import annotations
"""
Learning Handlers for Matrix ChatOps
=====================================

Matrix command handlers for HoloLoom Recursive Learning System.

Commands:
- !learn stats - Show learning system statistics
- !learn tool <name> - Show Thompson priors for a tool
- !learn hot [limit] - Show hot patterns (most accessed)
- !learn cold [age] - Show cold patterns (stale, for pruning)
- !learn refine <query> - Refine a query using advanced strategies
- !learn patterns - Show learned motif→tool patterns
- !learn reflect - Show reflection buffer metrics
- !learn help - Show learning help

Usage:
    from hololoom.apps.chatops.handlers.learning_handlers import register_learning_handlers

    # In run_chatops.py:
    register_learning_handlers(bot)

Created: 2025-12-15
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hololoom.recursive import FullLearningEngine, HotPatternTracker

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = None
    RoomMessageText = None

# Learning system imports with graceful degradation
try:
    from hololoom.recursive import (
        AdvancedRefiner,
        FullLearningEngine,
        HotPatternConfig,
        HotPatternFeedbackEngine,
        HotPatternTracker,
        LearnedPattern,
        PatternLearner,
        PolicyWeights,
        RefinementResult,
        RefinementStrategy,
        ThompsonPriors,
        UsageRecord,
    )
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    FullLearningEngine = None
    ThompsonPriors = None
    PolicyWeights = None
    HotPatternTracker = None
    HotPatternFeedbackEngine = None
    HotPatternConfig = None
    UsageRecord = None
    AdvancedRefiner = None
    RefinementStrategy = None
    RefinementResult = None
    PatternLearner = None
    LearnedPattern = None

try:
    from hololoom.reflection.buffer import ReflectionBuffer, ReflectionMetrics
    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False
    ReflectionBuffer = None
    ReflectionMetrics = None

# Handler registry
try:
    from hololoom.apps.chatops.handlers.handler_registry import (
        HandlerCategory,
        HandlerRegistry,
        chatops_handler,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    HandlerRegistry = None
    HandlerCategory = None

logger = logging.getLogger(__name__)


# ============================================================================
# Global instances (set by register_learning_handlers)
# ============================================================================

_learning_engine: Optional['FullLearningEngine'] = None
_hot_tracker: Optional['HotPatternTracker'] = None
_reflection_buffer: Optional['ReflectionBuffer'] = None


def get_learning_engine() -> Optional['FullLearningEngine']:
    """Get the global FullLearningEngine instance."""
    return _learning_engine


def set_learning_engine(engine: 'FullLearningEngine') -> None:
    """Set the global FullLearningEngine instance."""
    global _learning_engine
    _learning_engine = engine


def get_hot_tracker() -> Optional['HotPatternTracker']:
    """Get the global HotPatternTracker instance."""
    return _hot_tracker


def set_hot_tracker(tracker: 'HotPatternTracker') -> None:
    """Set the global HotPatternTracker instance."""
    global _hot_tracker
    _hot_tracker = tracker


def get_reflection_buffer() -> Optional['ReflectionBuffer']:
    """Get the global ReflectionBuffer instance."""
    return _reflection_buffer


def set_reflection_buffer(buffer: 'ReflectionBuffer') -> None:
    """Set the global ReflectionBuffer instance."""
    global _reflection_buffer
    _reflection_buffer = buffer


# ============================================================================
# Learning Stats Handler
# ============================================================================

async def handle_learn_stats(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    learning_engine: Optional['FullLearningEngine'] = None
) -> str:
    """
    Handle !learn stats command - Show learning system statistics.

    Displays Thompson Sampling priors, policy weights, and learning metrics.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        learning_engine: FullLearningEngine instance (optional, uses global)

    Returns:
        Learning statistics summary
    """
    engine = learning_engine or get_learning_engine()

    if not engine and not LEARNING_AVAILABLE:
        return (
            "**Learning System Unavailable**\n\n"
            "The recursive learning system is not installed or configured.\n"
            "Install with: `pip install HoloLoom[learning]`"
        )

    if not engine:
        return (
            "**Learning Engine Not Initialized**\n\n"
            "No learning engine is currently active.\n"
            "Start the system with learning enabled to use these commands."
        )

    try:
        stats = engine.get_learning_statistics()

        output = ["**📊 Learning System Statistics**\n"]

        # Thompson priors summary
        if 'thompson_priors' in stats:
            priors = stats['thompson_priors']
            output.append("**Thompson Sampling Priors:**")
            for tool, data in list(priors.items())[:5]:
                alpha = data.get('alpha', 1.0)
                beta = data.get('beta', 1.0)
                expected = alpha / (alpha + beta)
                output.append(f"  • `{tool}`: α={alpha:.1f}, β={beta:.1f}, E[X]={expected:.2f}")
            if len(priors) > 5:
                output.append(f"  ... and {len(priors) - 5} more tools")
            output.append("")

        # Policy weights summary
        if 'policy_weights' in stats:
            weights = stats['policy_weights']
            output.append("**Policy Adapter Weights:**")
            for adapter, weight in list(weights.items())[:5]:
                output.append(f"  • `{adapter}`: {weight:.3f}")
            output.append("")

        # Hot patterns count
        if 'hot_patterns_count' in stats:
            output.append(f"**Hot Patterns:** {stats['hot_patterns_count']}")

        # Learning metrics
        if 'total_updates' in stats:
            output.append(f"**Total Updates:** {stats['total_updates']}")
        if 'avg_confidence' in stats:
            output.append(f"**Avg Confidence:** {stats['avg_confidence']:.2f}")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting learning stats: {e}")
        return f"**Error:** Failed to get learning statistics: {str(e)}"


# ============================================================================
# Tool Priors Handler
# ============================================================================

async def handle_learn_tool(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    learning_engine: Optional['FullLearningEngine'] = None
) -> str:
    """
    Handle !learn tool command - Show Thompson priors for a specific tool.

    Args:
        room: Matrix room
        event: Message event
        args: Tool name to query
        learning_engine: FullLearningEngine instance (optional, uses global)

    Returns:
        Tool's Thompson Sampling statistics
    """
    if not args.strip():
        return (
            "**Usage:** `!learn tool <name>`\n\n"
            "**Examples:**\n"
            "- `!learn tool answer`\n"
            "- `!learn tool research`\n"
            "- `!learn tool explore`"
        )

    tool_name = args.strip().lower()
    engine = learning_engine or get_learning_engine()

    if not engine:
        return "**Error:** Learning engine not initialized."

    try:
        stats = engine.get_learning_statistics()
        priors = stats.get('thompson_priors', {})

        if tool_name not in priors:
            available = ", ".join(f"`{t}`" for t in list(priors.keys())[:10])
            return (
                f"**Tool Not Found:** `{tool_name}`\n\n"
                f"Available tools: {available}"
            )

        data = priors[tool_name]
        alpha = data.get('alpha', 1.0)
        beta = data.get('beta', 1.0)
        expected = alpha / (alpha + beta)
        uncertainty = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        pulls = int(alpha + beta - 2)  # Subtract initial priors

        output = [
            f"**🎰 Thompson Priors: `{tool_name}`**\n",
            f"**Alpha (successes):** {alpha:.2f}",
            f"**Beta (failures):** {beta:.2f}",
            f"**Expected Reward:** {expected:.3f}",
            f"**Uncertainty (variance):** {uncertainty:.4f}",
            f"**Total Pulls:** {pulls}",
            "",
            f"*Interpretation:* This tool has a {expected*100:.1f}% expected success rate."
        ]

        if expected > 0.7:
            output.append("Status: ✅ High performer")
        elif expected > 0.4:
            output.append("Status: 🟡 Moderate performer")
        else:
            output.append("Status: ⚠️ Low performer (consider investigating)")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting tool priors: {e}")
        return f"**Error:** Failed to get tool priors: {str(e)}"


# ============================================================================
# Hot Patterns Handler
# ============================================================================

async def handle_learn_hot(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    hot_tracker: Optional['HotPatternTracker'] = None
) -> str:
    """
    Handle !learn hot command - Show hot patterns (most accessed knowledge).

    Args:
        room: Matrix room
        event: Message event
        args: Optional limit (default 10)
        hot_tracker: HotPatternTracker instance (optional, uses global)

    Returns:
        List of hot patterns with heat scores
    """
    limit = 10
    if args.strip():
        try:
            limit = int(args.strip())
            limit = min(max(limit, 1), 50)  # Clamp between 1-50
        except ValueError:
            return "**Usage:** `!learn hot [limit]`\n\nLimit must be a number (1-50)."

    tracker = hot_tracker or get_hot_tracker()

    if not tracker and not LEARNING_AVAILABLE:
        return (
            "**Hot Pattern Tracking Unavailable**\n\n"
            "The hot pattern system is not installed or configured."
        )

    if not tracker:
        return (
            "**Hot Tracker Not Initialized**\n\n"
            "No hot pattern tracker is currently active."
        )

    try:
        hot_patterns = tracker.get_hot_patterns(limit=limit)

        if not hot_patterns:
            return (
                "**🔥 Hot Patterns**\n\n"
                "No hot patterns yet. Use the system more to build usage history!"
            )

        output = [f"**🔥 Top {len(hot_patterns)} Hot Patterns**\n"]

        for i, pattern in enumerate(hot_patterns, 1):
            element_id = pattern.element_id if hasattr(pattern, 'element_id') else str(pattern)
            heat = pattern.heat_score if hasattr(pattern, 'heat_score') else 0.0
            access = pattern.access_count if hasattr(pattern, 'access_count') else 0
            success = pattern.success_rate if hasattr(pattern, 'success_rate') else 0.0

            # Heat indicator
            if heat > 10:
                indicator = "🔥🔥🔥"
            elif heat > 5:
                indicator = "🔥🔥"
            elif heat > 1:
                indicator = "🔥"
            else:
                indicator = "•"

            output.append(
                f"{i}. {indicator} `{element_id[:40]}...`"
                if len(str(element_id)) > 40 else
                f"{i}. {indicator} `{element_id}`"
            )
            output.append(f"   Heat: {heat:.2f} | Accesses: {access} | Success: {success*100:.0f}%")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting hot patterns: {e}")
        return f"**Error:** Failed to get hot patterns: {str(e)}"


# ============================================================================
# Cold Patterns Handler
# ============================================================================

async def handle_learn_cold(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    hot_tracker: Optional['HotPatternTracker'] = None
) -> str:
    """
    Handle !learn cold command - Show cold patterns (stale, candidates for pruning).

    Args:
        room: Matrix room
        event: Message event
        args: Optional age threshold in hours (default 24)
        hot_tracker: HotPatternTracker instance (optional, uses global)

    Returns:
        List of cold patterns that haven't been accessed recently
    """
    age_hours = 24
    if args.strip():
        try:
            age_hours = float(args.strip())
            age_hours = min(max(age_hours, 1), 720)  # 1 hour to 30 days
        except ValueError:
            return "**Usage:** `!learn cold [age_hours]`\n\nAge must be a number (1-720 hours)."

    tracker = hot_tracker or get_hot_tracker()

    if not tracker:
        return "**Error:** Hot pattern tracker not initialized."

    try:
        # Get all patterns and filter by age
        all_patterns = tracker.get_all_patterns() if hasattr(tracker, 'get_all_patterns') else []

        age_seconds = age_hours * 3600
        cold_patterns = [
            p for p in all_patterns
            if hasattr(p, 'age_seconds') and p.age_seconds > age_seconds
        ]

        if not cold_patterns:
            return (
                f"**❄️ Cold Patterns (>{age_hours}h)**\n\n"
                "No patterns older than the threshold. Memory is fresh!"
            )

        # Sort by age (oldest first)
        cold_patterns.sort(key=lambda p: p.age_seconds, reverse=True)
        cold_patterns = cold_patterns[:20]  # Limit to 20

        output = [f"**❄️ Cold Patterns (>{age_hours}h old)**\n"]

        for i, pattern in enumerate(cold_patterns, 1):
            element_id = pattern.element_id if hasattr(pattern, 'element_id') else str(pattern)
            age_hrs = pattern.age_seconds / 3600 if hasattr(pattern, 'age_seconds') else 0
            heat = pattern.heat_score if hasattr(pattern, 'heat_score') else 0.0

            output.append(f"{i}. ❄️ `{element_id[:30]}...`" if len(str(element_id)) > 30 else f"{i}. ❄️ `{element_id}`")
            output.append(f"   Age: {age_hrs:.1f}h | Heat: {heat:.2f}")

        output.append("\n*Consider pruning these stale patterns to free memory.*")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting cold patterns: {e}")
        return f"**Error:** Failed to get cold patterns: {str(e)}"


# ============================================================================
# Refine Handler
# ============================================================================

async def handle_learn_refine(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    learning_engine: Optional['FullLearningEngine'] = None
) -> str:
    """
    Handle !learn refine command - Refine a query using advanced strategies.

    Args:
        room: Matrix room
        event: Message event
        args: Query text to refine
        learning_engine: FullLearningEngine instance (optional, uses global)

    Returns:
        Refinement result with quality trajectory
    """
    if not args.strip():
        return (
            "**Usage:** `!learn refine <query>`\n\n"
            "**Examples:**\n"
            "- `!learn refine What is Thompson Sampling?`\n"
            "- `!learn refine Explain neural networks`"
        )

    query_text = args.strip()
    engine = learning_engine or get_learning_engine()

    if not engine:
        return "**Error:** Learning engine not initialized."

    if not LEARNING_AVAILABLE:
        return "**Error:** Advanced refinement is not available."

    try:
        # Create refiner and run refinement

        # This is a simplified version - actual implementation would use orchestrator
        output = [
            "**🔧 Query Refinement**\n",
            f"**Original:** {query_text}",
            "",
            "**Available Strategies:**",
            "• `REFINE` - Iterative context expansion",
            "• `CRITIQUE` - Self-improvement",
            "• `VERIFY` - Accuracy → Completeness → Consistency",
            "• `ELEGANCE` - Clarity → Simplicity → Beauty",
            "• `HOFSTADTER` - Recursive self-reference",
            "",
            "*Note: Full refinement requires an active weaving session.*",
            "*Use `!loom weave` to run queries with automatic refinement.*"
        ]

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error during refinement: {e}")
        return f"**Error:** Refinement failed: {str(e)}"


# ============================================================================
# Patterns Handler
# ============================================================================

async def handle_learn_patterns(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    learning_engine: Optional['FullLearningEngine'] = None
) -> str:
    """
    Handle !learn patterns command - Show learned motif→tool patterns.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        learning_engine: FullLearningEngine instance (optional, uses global)

    Returns:
        List of learned patterns
    """
    engine = learning_engine or get_learning_engine()

    if not engine:
        return "**Error:** Learning engine not initialized."

    try:
        stats = engine.get_learning_statistics()
        patterns = stats.get('learned_patterns', [])

        if not patterns:
            return (
                "**🧩 Learned Patterns**\n\n"
                "No patterns learned yet. The system learns from successful queries."
            )

        output = ["**🧩 Learned Patterns**\n"]

        for i, pattern in enumerate(patterns[:15], 1):
            if isinstance(pattern, dict):
                motif = pattern.get('motif', 'unknown')
                tool = pattern.get('tool', 'unknown')
                confidence = pattern.get('confidence', 0.0)
                count = pattern.get('count', 0)
            else:
                motif = getattr(pattern, 'motif', 'unknown')
                tool = getattr(pattern, 'tool', 'unknown')
                confidence = getattr(pattern, 'confidence', 0.0)
                count = getattr(pattern, 'count', 0)

            output.append(f"{i}. `{motif}` → `{tool}`")
            output.append(f"   Confidence: {confidence:.2f} | Occurrences: {count}")

        if len(patterns) > 15:
            output.append(f"\n*... and {len(patterns) - 15} more patterns*")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting patterns: {e}")
        return f"**Error:** Failed to get patterns: {str(e)}"


# ============================================================================
# Reflect Handler
# ============================================================================

async def handle_learn_reflect(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    reflection_buffer: Optional['ReflectionBuffer'] = None
) -> str:
    """
    Handle !learn reflect command - Show reflection buffer metrics.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        reflection_buffer: ReflectionBuffer instance (optional, uses global)

    Returns:
        Reflection metrics summary
    """
    buffer = reflection_buffer or get_reflection_buffer()

    if not buffer and not REFLECTION_AVAILABLE:
        return (
            "**Reflection Buffer Unavailable**\n\n"
            "The reflection system is not installed or configured."
        )

    if not buffer:
        return (
            "**Reflection Buffer Not Initialized**\n\n"
            "No reflection buffer is currently active."
        )

    try:
        metrics = buffer.get_metrics() if hasattr(buffer, 'get_metrics') else None

        if not metrics:
            return (
                "**🪞 Reflection Metrics**\n\n"
                "No metrics available yet. Process some queries to build history."
            )

        output = ["**🪞 Reflection Metrics**\n"]

        # Cycle counts
        total = getattr(metrics, 'total_cycles', 0)
        successful = getattr(metrics, 'successful_cycles', 0)
        failed = getattr(metrics, 'failed_cycles', 0)
        success_rate = (successful / total * 100) if total > 0 else 0

        output.append(f"**Cycles:** {total} total ({successful} success, {failed} failed)")
        output.append(f"**Success Rate:** {success_rate:.1f}%\n")

        # Tool performance
        tool_rates = getattr(metrics, 'tool_success_rates', {})
        if tool_rates:
            output.append("**Tool Success Rates:**")
            for tool, rate in list(tool_rates.items())[:5]:
                bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
                output.append(f"  • `{tool}`: [{bar}] {rate*100:.0f}%")
            output.append("")

        # Pattern card performance
        pattern_rates = getattr(metrics, 'pattern_success_rates', {})
        if pattern_rates:
            output.append("**Pattern Card Success Rates:**")
            for pattern, rate in pattern_rates.items():
                output.append(f"  • `{pattern}`: {rate*100:.0f}%")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting reflection metrics: {e}")
        return f"**Error:** Failed to get reflection metrics: {str(e)}"


# ============================================================================
# Help Handler
# ============================================================================

async def handle_learn_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !learn help command - Show learning command help.

    Args:
        room: Matrix room
        event: Message event
        args: Unused

    Returns:
        Learning help text
    """
    return """**📚 Learning Commands**

**Statistics & Monitoring:**
• `!learn stats` - Show learning system statistics
• `!learn tool <name>` - Show Thompson priors for a tool
• `!learn reflect` - Show reflection buffer metrics

**Hot Pattern Tracking:**
• `!learn hot [limit]` - Show hot patterns (most accessed)
• `!learn cold [age_hours]` - Show cold patterns (stale)
• `!learn patterns` - Show learned motif→tool patterns

**Query Refinement:**
• `!learn refine <query>` - Refine a query using advanced strategies

**How Learning Works:**
1. **Thompson Sampling**: Tools have α/β priors that update based on success
2. **Hot Patterns**: Frequently accessed knowledge gets retrieval boost (2x)
3. **Pattern Learning**: System learns motif→tool mappings from successes
4. **Reflection**: Past outcomes analyzed for continuous improvement

*For more details, see: CLAUDE.md → Recursive Learning System*"""


# ============================================================================
# Handler Registration
# ============================================================================

class LearningHandlers:
    """Container for learning handlers with shared instances."""

    def __init__(
        self,
        learning_engine: Optional['FullLearningEngine'] = None,
        hot_tracker: Optional['HotPatternTracker'] = None,
        reflection_buffer: Optional['ReflectionBuffer'] = None
    ):
        self.learning_engine = learning_engine
        self.hot_tracker = hot_tracker
        self.reflection_buffer = reflection_buffer

    async def handle_stats(self, room, event, args: str) -> str:
        return await handle_learn_stats(room, event, args, self.learning_engine)

    async def handle_tool(self, room, event, args: str) -> str:
        return await handle_learn_tool(room, event, args, self.learning_engine)

    async def handle_hot(self, room, event, args: str) -> str:
        return await handle_learn_hot(room, event, args, self.hot_tracker)

    async def handle_cold(self, room, event, args: str) -> str:
        return await handle_learn_cold(room, event, args, self.hot_tracker)

    async def handle_refine(self, room, event, args: str) -> str:
        return await handle_learn_refine(room, event, args, self.learning_engine)

    async def handle_patterns(self, room, event, args: str) -> str:
        return await handle_learn_patterns(room, event, args, self.learning_engine)

    async def handle_reflect(self, room, event, args: str) -> str:
        return await handle_learn_reflect(room, event, args, self.reflection_buffer)

    async def handle_help(self, room, event, args: str) -> str:
        return await handle_learn_help(room, event, args)


def register_learning_handlers(
    bot,
    learning_engine: Optional['FullLearningEngine'] = None,
    hot_tracker: Optional['HotPatternTracker'] = None,
    reflection_buffer: Optional['ReflectionBuffer'] = None
) -> LearningHandlers:
    """
    Register learning command handlers with a Matrix bot.

    Args:
        bot: Matrix bot instance
        learning_engine: FullLearningEngine for stats/tools/patterns
        hot_tracker: HotPatternTracker for hot/cold patterns
        reflection_buffer: ReflectionBuffer for reflection metrics

    Returns:
        LearningHandlers instance
    """
    # Set global instances
    if learning_engine:
        set_learning_engine(learning_engine)
    if hot_tracker:
        set_hot_tracker(hot_tracker)
    if reflection_buffer:
        set_reflection_buffer(reflection_buffer)

    handlers = LearningHandlers(learning_engine, hot_tracker, reflection_buffer)

    # Register with bot if it has the expected interface
    if hasattr(bot, 'register_command'):
        bot.register_command("learn stats", handlers.handle_stats)
        bot.register_command("learn tool", handlers.handle_tool)
        bot.register_command("learn hot", handlers.handle_hot)
        bot.register_command("learn cold", handlers.handle_cold)
        bot.register_command("learn refine", handlers.handle_refine)
        bot.register_command("learn patterns", handlers.handle_patterns)
        bot.register_command("learn reflect", handlers.handle_reflect)
        bot.register_command("learn help", handlers.handle_help)

    logger.info("Registered 8 learning command handlers")
    return handlers
