from __future__ import annotations
"""
Awareness Handlers for Matrix ChatOps
======================================

Matrix command handlers for HoloLoom Awareness & Consciousness System.

Commands:
- !aware metrics - Show awareness graph metrics
- !aware activate <query> - Activate memories by query
- !aware spring <nodes> - Physics-based spring propagation
- !aware wave mode - Show current brain wave mode
- !aware consolidate - Run theta wave consolidation
- !aware confidence - Show meta-confidence (confidence about confidence)
- !aware gaps - Detect knowledge gaps
- !aware help - Show awareness help

Usage:
    from hololoom.apps.chatops.handlers.awareness_handlers import register_awareness_handlers

    # In run_chatops.py:
    register_awareness_handlers(bot, awareness_graph)

Created: 2025-12-15
"""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hololoom.memory.awareness_graph import AwarenessGraph
    from hololoom.memory.multi_wave_engine import MultiWaveMemoryEngine
    from hololoom.memory.spring_dynamics import SpringDynamics

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False
    MatrixRoom = None
    RoomMessageText = None

# Awareness system imports with graceful degradation
try:
    from hololoom.memory.awareness_graph import AwarenessGraph
    from hololoom.memory.awareness_types import (
        ActivationBudget,
        ActivationStrategy,
        AwarenessMetrics,
    )
    AWARENESS_AVAILABLE = True
except ImportError:
    AWARENESS_AVAILABLE = False
    AwarenessGraph = None
    AwarenessMetrics = None
    ActivationStrategy = None
    ActivationBudget = None

try:
    from hololoom.memory.spring_dynamics import SpringConfig, SpringDynamics
    SPRING_AVAILABLE = True
except ImportError:
    SPRING_AVAILABLE = False
    SpringDynamics = None
    SpringConfig = None

try:
    from hololoom.memory.multi_wave_engine import (
        BrainWaveMode,
        MultiWaveMemoryEngine,
        ThetaWaveConsolidator,
    )
    MULTIWAVE_AVAILABLE = True
except ImportError:
    MULTIWAVE_AVAILABLE = False
    MultiWaveMemoryEngine = None
    BrainWaveMode = None
    ThetaWaveConsolidator = None

# Meta-awareness (may not exist yet - graceful degradation)
try:
    from hololoom.awareness.meta_awareness import KnowledgeGapHypothesis, MetaConfidence
    META_AWARENESS_AVAILABLE = True
except ImportError:
    META_AWARENESS_AVAILABLE = False
    MetaConfidence = None
    KnowledgeGapHypothesis = None

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
# Global instances (set by register_awareness_handlers)
# ============================================================================

_awareness_graph: Optional['AwarenessGraph'] = None
_spring_dynamics: Optional['SpringDynamics'] = None
_wave_engine: Optional['MultiWaveMemoryEngine'] = None


def get_awareness_graph() -> Optional['AwarenessGraph']:
    """Get the global AwarenessGraph instance."""
    return _awareness_graph


def set_awareness_graph(graph: 'AwarenessGraph') -> None:
    """Set the global AwarenessGraph instance."""
    global _awareness_graph
    _awareness_graph = graph


def get_spring_dynamics() -> Optional['SpringDynamics']:
    """Get the global SpringDynamics instance."""
    return _spring_dynamics


def set_spring_dynamics(dynamics: 'SpringDynamics') -> None:
    """Set the global SpringDynamics instance."""
    global _spring_dynamics
    _spring_dynamics = dynamics


def get_wave_engine() -> Optional['MultiWaveMemoryEngine']:
    """Get the global MultiWaveMemoryEngine instance."""
    return _wave_engine


def set_wave_engine(engine: 'MultiWaveMemoryEngine') -> None:
    """Set the global MultiWaveMemoryEngine instance."""
    global _wave_engine
    _wave_engine = engine


# ============================================================================
# Metrics Handler
# ============================================================================

async def handle_aware_metrics(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    awareness_graph: Optional['AwarenessGraph'] = None
) -> str:
    """
    Handle !aware metrics command - Show awareness graph metrics.

    Displays activation levels, coherence, shift detection, and trajectory.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        awareness_graph: AwarenessGraph instance (optional, uses global)

    Returns:
        Awareness metrics summary
    """
    graph = awareness_graph or get_awareness_graph()

    if not graph and not AWARENESS_AVAILABLE:
        return (
            "**Awareness System Unavailable**\n\n"
            "The awareness graph is not installed or configured.\n"
            "This is a core memory component - check installation."
        )

    if not graph:
        return (
            "**Awareness Graph Not Initialized**\n\n"
            "No awareness graph is currently active."
        )

    try:
        metrics = graph.get_metrics() if hasattr(graph, 'get_metrics') else None

        if not metrics:
            # Try to compute basic metrics manually
            output = ["**🧠 Awareness Metrics**\n"]

            # Get activation field info
            if hasattr(graph, 'activation_field'):
                af = graph.activation_field
                activations = getattr(af, 'activations', {})
                active_count = len([v for v in activations.values() if v > 0.1])
                mean_activation = sum(activations.values()) / len(activations) if activations else 0

                output.append(f"**Active Nodes:** {active_count}")
                output.append(f"**Mean Activation:** {mean_activation:.3f}")

            # Get trajectory info
            if hasattr(graph, 'trajectory'):
                traj = graph.trajectory
                output.append(f"**Trajectory Length:** {len(traj)}")

            return "\n".join(output)

        output = ["**🧠 Awareness Metrics**\n"]

        # Activation metrics
        if hasattr(metrics, 'activation') or isinstance(metrics, dict):
            activation = metrics.get('activation', {}) if isinstance(metrics, dict) else getattr(metrics, 'activation', {})
            output.append("**Activation:**")
            output.append(f"  • Active Nodes: {activation.get('active_nodes', 0)}")
            output.append(f"  • Mean Activation: {activation.get('mean_activation', 0):.3f}")
            output.append(f"  • Peak Activation: {activation.get('peak_activation', 0):.3f}")
            output.append("")

        # Coherence metrics
        if hasattr(metrics, 'coherence') or isinstance(metrics, dict):
            coherence = metrics.get('coherence', {}) if isinstance(metrics, dict) else getattr(metrics, 'coherence', {})
            global_coh = coherence.get('global_coherence', 0)
            output.append("**Coherence:**")
            output.append(f"  • Global: {global_coh:.3f}")

            # Visual coherence bar
            bar = "█" * int(global_coh * 10) + "░" * (10 - int(global_coh * 10))
            output.append(f"  • [{bar}]")
            output.append("")

        # Shift detection
        if hasattr(metrics, 'shift') or isinstance(metrics, dict):
            shift = metrics.get('shift', {}) if isinstance(metrics, dict) else getattr(metrics, 'shift', {})
            if shift:
                output.append("**Context Shift:**")
                output.append(f"  • Shift Detected: {shift.get('detected', False)}")
                output.append(f"  • Magnitude: {shift.get('magnitude', 0):.3f}")
                output.append("")

        # Trajectory
        if hasattr(metrics, 'temporal') or isinstance(metrics, dict):
            temporal = metrics.get('temporal', {}) if isinstance(metrics, dict) else getattr(metrics, 'temporal', {})
            output.append(f"**Trajectory Points:** {temporal.get('trajectory_length', 0)}")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting awareness metrics: {e}")
        return f"**Error:** Failed to get awareness metrics: {str(e)}"


# ============================================================================
# Activate Handler
# ============================================================================

async def handle_aware_activate(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    awareness_graph: Optional['AwarenessGraph'] = None
) -> str:
    """
    Handle !aware activate command - Activate memories by query.

    Args:
        room: Matrix room
        event: Message event
        args: Query text to activate on
        awareness_graph: AwarenessGraph instance (optional, uses global)

    Returns:
        Activated memories with relevance scores
    """
    if not args.strip():
        return (
            "**Usage:** `!aware activate <query>`\n\n"
            "**Examples:**\n"
            "- `!aware activate Thompson Sampling`\n"
            "- `!aware activate neural networks`"
        )

    query_text = args.strip()
    graph = awareness_graph or get_awareness_graph()

    if not graph:
        return "**Error:** Awareness graph not initialized."

    try:
        # Activate based on query
        if hasattr(graph, 'perceive'):
            perception = graph.perceive(query_text)
            activated = perception.activated_nodes if hasattr(perception, 'activated_nodes') else []
        elif hasattr(graph, 'activate'):
            activated = graph.activate(query_text)
        else:
            return "**Error:** Awareness graph doesn't support activation."

        if not activated:
            return (
                f"**🔍 Activation: `{query_text[:40]}...`**\n\n"
                if len(query_text) > 40 else
                f"**🔍 Activation: `{query_text}`**\n\n"
                "No memories activated. Try a different query."
            )

        output = [f"**🔍 Activated {len(activated)} Memories**\n"]

        # Show top activated memories
        for i, item in enumerate(activated[:10], 1):
            if isinstance(item, tuple):
                node_id, activation = item
            elif hasattr(item, 'node_id'):
                node_id = item.node_id
                activation = getattr(item, 'activation', 0.5)
            else:
                node_id = str(item)
                activation = 0.5

            # Activation bar
            bar = "▓" * int(activation * 10) + "░" * (10 - int(activation * 10))
            output.append(f"{i}. [{bar}] `{node_id[:30]}...`" if len(str(node_id)) > 30 else f"{i}. [{bar}] `{node_id}`")

        if len(activated) > 10:
            output.append(f"\n*... and {len(activated) - 10} more*")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error during activation: {e}")
        return f"**Error:** Activation failed: {str(e)}"


# ============================================================================
# Spring Propagation Handler
# ============================================================================

async def handle_aware_spring(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    spring_dynamics: Optional['SpringDynamics'] = None
) -> str:
    """
    Handle !aware spring command - Physics-based spring propagation.

    Args:
        room: Matrix room
        event: Message event
        args: Comma-separated node IDs to activate
        spring_dynamics: SpringDynamics instance (optional, uses global)

    Returns:
        Propagated activation map
    """
    if not args.strip():
        return (
            "**Usage:** `!aware spring <node1,node2,...>`\n\n"
            "**Examples:**\n"
            "- `!aware spring thompson_sampling`\n"
            "- `!aware spring neural_net,backprop,gradient`"
        )

    node_ids = [n.strip() for n in args.split(',') if n.strip()]
    dynamics = spring_dynamics or get_spring_dynamics()

    if not dynamics and not SPRING_AVAILABLE:
        return (
            "**Spring Dynamics Unavailable**\n\n"
            "The physics-based propagation system is not available."
        )

    if not dynamics:
        return (
            "**Spring Dynamics Not Initialized**\n\n"
            "No spring dynamics engine is currently active."
        )

    try:
        # Set initial activations
        initial = dict.fromkeys(node_ids, 1.0)

        if hasattr(dynamics, 'activate_nodes'):
            dynamics.activate_nodes(initial)
        elif hasattr(dynamics, 'set_activations'):
            dynamics.set_activations(initial)

        # Propagate
        result = dynamics.propagate()

        if not result or not hasattr(result, 'activated_nodes'):
            return "**Error:** Propagation returned no results."

        activated = result.activated_nodes

        output = [
            "**⚡ Spring Propagation**\n",
            f"**Seeds:** {', '.join(f'`{n}`' for n in node_ids)}",
            f"**Iterations:** {getattr(result, 'iterations', 'N/A')}",
            f"**Energy:** {getattr(result, 'final_energy', 'N/A')}",
            "",
            f"**Activated Nodes ({len(activated)}):**"
        ]

        # Sort by activation
        sorted_nodes = sorted(
            activated.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]

        for node_id, activation in sorted_nodes:
            if activation > 0.01:
                bar = "▓" * int(activation * 10) + "░" * (10 - int(activation * 10))
                output.append(f"  [{bar}] {activation:.3f} `{node_id[:25]}...`" if len(str(node_id)) > 25 else f"  [{bar}] {activation:.3f} `{node_id}`")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error during spring propagation: {e}")
        return f"**Error:** Spring propagation failed: {str(e)}"


# ============================================================================
# Wave Mode Handler
# ============================================================================

async def handle_aware_wave_mode(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    wave_engine: Optional['MultiWaveMemoryEngine'] = None
) -> str:
    """
    Handle !aware wave mode command - Show current brain wave mode.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        wave_engine: MultiWaveMemoryEngine instance (optional, uses global)

    Returns:
        Current brain wave mode and description
    """
    engine = wave_engine or get_wave_engine()

    if not engine and not MULTIWAVE_AVAILABLE:
        return (
            "**Multi-Wave Engine Unavailable**\n\n"
            "The brain wave simulation system is not available."
        )

    if not engine:
        return (
            "**Wave Engine Not Initialized**\n\n"
            "No multi-wave engine is currently active."
        )

    try:
        mode = engine.get_current_mode() if hasattr(engine, 'get_current_mode') else None

        if not mode:
            mode_name = "UNKNOWN"
        elif isinstance(mode, str):
            mode_name = mode.upper()
        else:
            mode_name = mode.value.upper() if hasattr(mode, 'value') else str(mode).upper()

        mode_info = {
            "BETA": ("🔴 Beta", "13-30 Hz", "Active retrieval (awake)", "Fast memory access"),
            "ALPHA": ("🟠 Alpha", "8-13 Hz", "Relaxed filtering (resting)", "Noise suppression"),
            "THETA": ("🟡 Theta", "4-8 Hz", "Light sleep consolidation", "Strengthening connections"),
            "DELTA": ("🟢 Delta", "0.5-4 Hz", "Deep sleep reorganization", "Pruning weak connections"),
            "REM": ("🟣 REM", "Mixed", "Dreaming, random replay", "Creative bridging"),
        }

        info = mode_info.get(mode_name, ("⚪ Unknown", "N/A", "Unknown state", "N/A"))

        output = [
            "**🌊 Current Brain Wave Mode**\n",
            f"**Mode:** {info[0]}",
            f"**Frequency:** {info[1]}",
            f"**State:** {info[2]}",
            f"**Purpose:** {info[3]}",
            "",
            "**Mode Transitions:**",
            "• BETA → Active query processing",
            "• ALPHA → 5-30 min idle (noise filtering)",
            "• THETA → 30 min-2 hr idle (consolidation)",
            "• DELTA → >2 hr idle (deep pruning)",
            "• REM → >2 hr idle (creative bridging)"
        ]

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting wave mode: {e}")
        return f"**Error:** Failed to get wave mode: {str(e)}"


# ============================================================================
# Consolidate Handler
# ============================================================================

async def handle_aware_consolidate(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    wave_engine: Optional['MultiWaveMemoryEngine'] = None
) -> str:
    """
    Handle !aware consolidate command - Run theta wave consolidation.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        wave_engine: MultiWaveMemoryEngine instance (optional, uses global)

    Returns:
        Consolidation results
    """
    engine = wave_engine or get_wave_engine()

    if not engine:
        return "**Error:** Wave engine not initialized."

    try:
        # Run theta consolidation
        if hasattr(engine, 'theta_consolidator'):
            consolidator = engine.theta_consolidator
            result = consolidator.consolidate() if hasattr(consolidator, 'consolidate') else None
        elif hasattr(engine, 'run_consolidation'):
            result = engine.run_consolidation()
        else:
            return (
                "**Consolidation Not Available**\n\n"
                "The wave engine doesn't support manual consolidation."
            )

        if not result:
            return (
                "**🌙 Theta Consolidation**\n\n"
                "Consolidation completed, but no specific results returned.\n"
                "Check logs for details."
            )

        output = ["**🌙 Theta Consolidation Complete**\n"]

        if isinstance(result, dict):
            output.append(f"**Pairs Analyzed:** {result.get('pairs_analyzed', 'N/A')}")
            output.append(f"**Connections Strengthened:** {result.get('strengthened', 'N/A')}")
            output.append(f"**New Links Created:** {result.get('new_links', 'N/A')}")
        else:
            output.append(f"Result: {result}")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error during consolidation: {e}")
        return f"**Error:** Consolidation failed: {str(e)}"


# ============================================================================
# Meta-Confidence Handler
# ============================================================================

async def handle_aware_confidence(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    awareness_graph: Optional['AwarenessGraph'] = None
) -> str:
    """
    Handle !aware confidence command - Show meta-confidence (confidence about confidence).

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        awareness_graph: AwarenessGraph instance (optional, uses global)

    Returns:
        Meta-confidence metrics
    """
    graph = awareness_graph or get_awareness_graph()

    if not graph:
        return "**Error:** Awareness graph not initialized."

    try:
        # Try to get meta-confidence from awareness integration
        if META_AWARENESS_AVAILABLE and hasattr(graph, 'get_meta_confidence'):
            meta_conf = graph.get_meta_confidence()
        else:
            # Calculate from activation metrics
            metrics = graph.get_metrics() if hasattr(graph, 'get_metrics') else {}

            if isinstance(metrics, dict):
                coherence = metrics.get('coherence', {}).get('global_coherence', 0.5)
                activation = metrics.get('activation', {}).get('mean_activation', 0.5)
            else:
                coherence = getattr(metrics, 'coherence', {}).get('global_coherence', 0.5) if hasattr(metrics, 'coherence') else 0.5
                activation = getattr(metrics, 'activation', {}).get('mean_activation', 0.5) if hasattr(metrics, 'activation') else 0.5

            # Meta-confidence = coherence * activation_density
            meta_conf = coherence * min(activation * 2, 1.0)

        output = [
            "**🔮 Meta-Confidence**\n",
            "*Confidence about our confidence - epistemic awareness*\n",
        ]

        if isinstance(meta_conf, float):
            bar = "█" * int(meta_conf * 10) + "░" * (10 - int(meta_conf * 10))
            output.append(f"**Level:** [{bar}] {meta_conf*100:.1f}%")

            if meta_conf > 0.7:
                output.append("\n**Status:** ✅ High epistemic confidence")
                output.append("The system has good knowledge coverage for recent queries.")
            elif meta_conf > 0.4:
                output.append("\n**Status:** 🟡 Moderate epistemic confidence")
                output.append("Some knowledge gaps may exist.")
            else:
                output.append("\n**Status:** ⚠️ Low epistemic confidence")
                output.append("The system may lack knowledge in recent query areas.")
        else:
            output.append(f"**Result:** {meta_conf}")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error getting meta-confidence: {e}")
        return f"**Error:** Failed to get meta-confidence: {str(e)}"


# ============================================================================
# Knowledge Gaps Handler
# ============================================================================

async def handle_aware_gaps(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    awareness_graph: Optional['AwarenessGraph'] = None
) -> str:
    """
    Handle !aware gaps command - Detect knowledge gaps.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (unused)
        awareness_graph: AwarenessGraph instance (optional, uses global)

    Returns:
        Detected knowledge gaps
    """
    graph = awareness_graph or get_awareness_graph()

    if not graph:
        return "**Error:** Awareness graph not initialized."

    try:
        gaps = []

        # Try meta-awareness system
        if META_AWARENESS_AVAILABLE and hasattr(graph, 'detect_knowledge_gaps'):
            gaps = graph.detect_knowledge_gaps()
        else:
            # Infer gaps from low-confidence queries in trajectory
            if hasattr(graph, 'trajectory'):
                for item in list(graph.trajectory)[-20:]:
                    if hasattr(item, 'confidence') and item.confidence < 0.4:
                        gaps.append({
                            'query': getattr(item, 'query', 'unknown'),
                            'confidence': item.confidence,
                            'reason': 'Low confidence response'
                        })

        if not gaps:
            return (
                "**🕳️ Knowledge Gap Detection**\n\n"
                "No significant knowledge gaps detected in recent queries.\n"
                "The system appears to have good coverage."
            )

        output = [f"**🕳️ Detected {len(gaps)} Knowledge Gaps**\n"]

        for i, gap in enumerate(gaps[:10], 1):
            if isinstance(gap, dict):
                query = gap.get('query', 'Unknown query')
                conf = gap.get('confidence', 0)
                reason = gap.get('reason', 'Unknown reason')
            else:
                query = getattr(gap, 'query', str(gap))
                conf = getattr(gap, 'confidence', 0)
                reason = getattr(gap, 'reason', 'Low confidence')

            output.append(f"{i}. **Gap:** `{query[:40]}...`" if len(str(query)) > 40 else f"{i}. **Gap:** `{query}`")
            output.append(f"   Confidence: {conf:.2f} | Reason: {reason}")

        output.append("\n*Consider adding knowledge in these areas.*")

        return "\n".join(output)

    except Exception as e:
        logger.exception(f"Error detecting gaps: {e}")
        return f"**Error:** Gap detection failed: {str(e)}"


# ============================================================================
# Help Handler
# ============================================================================

async def handle_aware_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !aware help command - Show awareness command help.

    Args:
        room: Matrix room
        event: Message event
        args: Unused

    Returns:
        Awareness help text
    """
    return """**🧠 Awareness Commands**

**Metrics & Monitoring:**
• `!aware metrics` - Show awareness graph metrics
• `!aware confidence` - Show meta-confidence (epistemic awareness)
• `!aware gaps` - Detect knowledge gaps

**Activation & Propagation:**
• `!aware activate <query>` - Activate memories by query
• `!aware spring <nodes>` - Physics-based spring propagation

**Brain Wave System:**
• `!aware wave mode` - Show current brain wave mode
• `!aware consolidate` - Run theta wave consolidation

**Brain Wave Modes:**
• **Beta** (13-30 Hz): Active retrieval
• **Alpha** (8-13 Hz): Relaxed noise filtering
• **Theta** (4-8 Hz): Light sleep consolidation
• **Delta** (0.5-4 Hz): Deep reorganization
• **REM**: Creative bridging

**How Awareness Works:**
1. **228D Semantic Space**: Memories positioned in interpretable dimensions
2. **Activation Field**: Query activates relevant memories
3. **Spring Dynamics**: Physics-based propagation (Hooke's Law)
4. **Coherence**: Measures how well-connected active memories are

*For more details, see: CLAUDE.md → Consciousness Integration*"""


# ============================================================================
# Handler Registration
# ============================================================================

class AwarenessHandlers:
    """Container for awareness handlers with shared instances."""

    def __init__(
        self,
        awareness_graph: Optional['AwarenessGraph'] = None,
        spring_dynamics: Optional['SpringDynamics'] = None,
        wave_engine: Optional['MultiWaveMemoryEngine'] = None
    ):
        self.awareness_graph = awareness_graph
        self.spring_dynamics = spring_dynamics
        self.wave_engine = wave_engine

    async def handle_metrics(self, room, event, args: str) -> str:
        return await handle_aware_metrics(room, event, args, self.awareness_graph)

    async def handle_activate(self, room, event, args: str) -> str:
        return await handle_aware_activate(room, event, args, self.awareness_graph)

    async def handle_spring(self, room, event, args: str) -> str:
        return await handle_aware_spring(room, event, args, self.spring_dynamics)

    async def handle_wave_mode(self, room, event, args: str) -> str:
        return await handle_aware_wave_mode(room, event, args, self.wave_engine)

    async def handle_consolidate(self, room, event, args: str) -> str:
        return await handle_aware_consolidate(room, event, args, self.wave_engine)

    async def handle_confidence(self, room, event, args: str) -> str:
        return await handle_aware_confidence(room, event, args, self.awareness_graph)

    async def handle_gaps(self, room, event, args: str) -> str:
        return await handle_aware_gaps(room, event, args, self.awareness_graph)

    async def handle_help(self, room, event, args: str) -> str:
        return await handle_aware_help(room, event, args)


def register_awareness_handlers(
    bot,
    awareness_graph: Optional['AwarenessGraph'] = None,
    spring_dynamics: Optional['SpringDynamics'] = None,
    wave_engine: Optional['MultiWaveMemoryEngine'] = None
) -> AwarenessHandlers:
    """
    Register awareness command handlers with a Matrix bot.

    Args:
        bot: Matrix bot instance
        awareness_graph: AwarenessGraph for metrics/activate/confidence/gaps
        spring_dynamics: SpringDynamics for spring propagation
        wave_engine: MultiWaveMemoryEngine for wave mode/consolidate

    Returns:
        AwarenessHandlers instance
    """
    # Set global instances
    if awareness_graph:
        set_awareness_graph(awareness_graph)
    if spring_dynamics:
        set_spring_dynamics(spring_dynamics)
    if wave_engine:
        set_wave_engine(wave_engine)

    handlers = AwarenessHandlers(awareness_graph, spring_dynamics, wave_engine)

    # Register with bot if it has the expected interface
    if hasattr(bot, 'register_command'):
        bot.register_command("aware metrics", handlers.handle_metrics)
        bot.register_command("aware activate", handlers.handle_activate)
        bot.register_command("aware spring", handlers.handle_spring)
        bot.register_command("aware wave", handlers.handle_wave_mode)
        bot.register_command("aware consolidate", handlers.handle_consolidate)
        bot.register_command("aware confidence", handlers.handle_confidence)
        bot.register_command("aware gaps", handlers.handle_gaps)
        bot.register_command("aware help", handlers.handle_help)

    logger.info("Registered 8 awareness command handlers")
    return handlers
