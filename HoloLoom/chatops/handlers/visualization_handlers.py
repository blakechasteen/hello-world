"""
Visualization Handlers for Matrix ChatOps
==========================================

Matrix command handlers for HoloLoom Tufte-style visualizations.

Commands:
- !dashboard confidence - Confidence trajectory chart with anomaly detection
- !dashboard cache - Cache effectiveness gauge
- !dashboard waterfall - Stage timing waterfall chart
- !dashboard knowledge - Knowledge graph network visualization
- !dashboard rag - RAG performance dashboard (5 panels)
- !dashboard help - Show dashboard command help

Usage:
    from HoloLoom.chatops.handlers.visualization_handlers import register_visualization_handlers

    # In run_chatops.py:
    register_visualization_handlers(bot, orchestrator)

Created: 2025-12-05
"""

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

try:
    from nio import MatrixRoom, RoomMessageText
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False

# Visualization imports
try:
    from HoloLoom.visualization.confidence_trajectory import render_confidence_trajectory
    CONFIDENCE_AVAILABLE = True
except ImportError:
    CONFIDENCE_AVAILABLE = False
    render_confidence_trajectory = None

try:
    from HoloLoom.visualization.cache_gauge import render_cache_gauge
    CACHE_GAUGE_AVAILABLE = True
except ImportError:
    CACHE_GAUGE_AVAILABLE = False
    render_cache_gauge = None

try:
    from HoloLoom.visualization.stage_waterfall import render_pipeline_waterfall
    WATERFALL_AVAILABLE = True
except ImportError:
    WATERFALL_AVAILABLE = False
    render_pipeline_waterfall = None

try:
    from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False
    render_knowledge_graph_from_kg = None

try:
    from HoloLoom.visualization.rag_dashboard import RAGDashboard
    RAG_DASHBOARD_AVAILABLE = True
except ImportError:
    RAG_DASHBOARD_AVAILABLE = False
    RAGDashboard = None

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
# Session Tracking (for collecting metrics across queries)
# ============================================================================

class SessionMetrics:
    """
    Tracks metrics across a session for visualization.

    Stores confidence scores, cache hits, stage timings, etc.
    for rendering dashboards.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.confidences: List[float] = []
        self.cached: List[bool] = []
        self.query_texts: List[str] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_queries: int = 0
        self.cached_latencies: List[float] = []
        self.uncached_latencies: List[float] = []
        self.stage_timings: List[Dict[str, float]] = []
        self.rag_results: List[Dict[str, Any]] = []

    def record_query(
        self,
        confidence: float,
        cached: bool,
        query_text: str,
        latency_ms: float,
        stage_timings: Optional[Dict[str, float]] = None,
        rag_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record metrics from a query."""
        # Confidence tracking
        self.confidences.append(confidence)
        self.cached.append(cached)
        self.query_texts.append(query_text)

        # Trim to max history
        if len(self.confidences) > self.max_history:
            self.confidences = self.confidences[-self.max_history:]
            self.cached = self.cached[-self.max_history:]
            self.query_texts = self.query_texts[-self.max_history:]

        # Cache tracking
        self.total_queries += 1
        if cached:
            self.cache_hits += 1
            self.cached_latencies.append(latency_ms)
        else:
            self.cache_misses += 1
            self.uncached_latencies.append(latency_ms)

        # Stage timing tracking
        if stage_timings:
            self.stage_timings.append(stage_timings)
            if len(self.stage_timings) > self.max_history:
                self.stage_timings = self.stage_timings[-self.max_history:]

        # RAG result tracking
        if rag_result:
            self.rag_results.append(rag_result)
            if len(self.rag_results) > self.max_history:
                self.rag_results = self.rag_results[-self.max_history:]

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def get_avg_cached_latency(self) -> float:
        """Get average cached latency."""
        if not self.cached_latencies:
            return 0.0
        return sum(self.cached_latencies) / len(self.cached_latencies)

    def get_avg_uncached_latency(self) -> float:
        """Get average uncached latency."""
        if not self.uncached_latencies:
            return 0.0
        return sum(self.uncached_latencies) / len(self.uncached_latencies)

    def reset(self) -> None:
        """Reset all metrics."""
        self.__init__(self.max_history)


# Global session metrics (shared across handlers)
_session_metrics = SessionMetrics()


def get_session_metrics() -> SessionMetrics:
    """Get global session metrics."""
    return _session_metrics


# ============================================================================
# Command Handlers
# ============================================================================

async def handle_dashboard_confidence(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> str:
    """
    Handle !dashboard confidence command - Confidence trajectory chart.

    Shows confidence scores over time with anomaly detection.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (--detect-anomalies, --title=X)
        orchestrator: WeavingOrchestrator instance (optional)

    Returns:
        HTML dashboard or error message
    """
    if not CONFIDENCE_AVAILABLE:
        return "❌ Confidence trajectory visualization not available"

    metrics = get_session_metrics()

    if not metrics.confidences:
        return "📊 **Confidence Dashboard**\n\nNo queries recorded yet. Run some queries first!"

    # Parse args
    detect_anomalies = "--detect-anomalies" in args or "-a" in args
    title = "Session Confidence Trajectory"

    if "--title=" in args:
        parts = args.split("--title=")
        if len(parts) > 1:
            title = parts[1].split()[0] if parts[1] else title

    try:
        html = render_confidence_trajectory(
            confidences=metrics.confidences,
            cached=metrics.cached,
            query_texts=metrics.query_texts,
            title=title,
            detect_anomalies=detect_anomalies
        )

        # Return summary with link to full dashboard
        summary = f"📈 **Confidence Trajectory**\n\n"
        summary += f"**Queries analyzed:** {len(metrics.confidences)}\n"

        if metrics.confidences:
            avg_conf = sum(metrics.confidences) / len(metrics.confidences)
            min_conf = min(metrics.confidences)
            max_conf = max(metrics.confidences)
            summary += f"**Average confidence:** {avg_conf:.2f}\n"
            summary += f"**Range:** {min_conf:.2f} - {max_conf:.2f}\n"

        cache_rate = sum(metrics.cached) / len(metrics.cached) if metrics.cached else 0
        summary += f"**Cache hit rate:** {cache_rate:.1%}\n"

        if detect_anomalies:
            summary += "\n_Anomaly detection enabled_\n"

        # In a real implementation, we'd save HTML and return a link
        # For now, return summary
        summary += "\n_Dashboard HTML generated (save to file for viewing)_"

        return summary

    except Exception as e:
        logger.error(f"Error rendering confidence trajectory: {e}")
        return f"❌ Rendering failed: {str(e)}"


async def handle_dashboard_cache(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> str:
    """
    Handle !dashboard cache command - Cache effectiveness gauge.

    Shows cache hit rate, latencies, and recommendations.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments
        orchestrator: WeavingOrchestrator instance (optional)

    Returns:
        Cache statistics summary
    """
    if not CACHE_GAUGE_AVAILABLE:
        return "❌ Cache gauge visualization not available"

    metrics = get_session_metrics()

    if metrics.total_queries == 0:
        return "📊 **Cache Dashboard**\n\nNo queries recorded yet. Run some queries first!"

    try:
        hit_rate = metrics.get_cache_hit_rate()
        avg_cached = metrics.get_avg_cached_latency()
        avg_uncached = metrics.get_avg_uncached_latency()

        html = render_cache_gauge(
            hit_rate=hit_rate,
            total_queries=metrics.total_queries,
            cache_hits=metrics.cache_hits,
            avg_cached_latency_ms=avg_cached,
            avg_uncached_latency_ms=avg_uncached,
            title="Session Cache Performance",
            show_details=True,
            show_recommendations=True
        )

        # Build summary
        summary = f"💾 **Cache Performance**\n\n"
        summary += f"**Hit Rate:** {hit_rate:.1%}\n"
        summary += f"**Total Queries:** {metrics.total_queries}\n"
        summary += f"**Cache Hits:** {metrics.cache_hits}\n"
        summary += f"**Cache Misses:** {metrics.cache_misses}\n"

        if avg_cached > 0 and avg_uncached > 0:
            speedup = avg_uncached / avg_cached
            summary += f"\n**Avg Cached Latency:** {avg_cached:.1f}ms\n"
            summary += f"**Avg Uncached Latency:** {avg_uncached:.1f}ms\n"
            summary += f"**Speedup:** {speedup:.1f}x\n"

        # Rating
        if hit_rate >= 0.8:
            summary += "\n✅ **Rating:** Excellent"
        elif hit_rate >= 0.6:
            summary += "\n🟢 **Rating:** Good"
        elif hit_rate >= 0.4:
            summary += "\n🟡 **Rating:** Fair"
        else:
            summary += "\n🔴 **Rating:** Poor - consider cache tuning"

        return summary

    except Exception as e:
        logger.error(f"Error rendering cache gauge: {e}")
        return f"❌ Rendering failed: {str(e)}"


async def handle_dashboard_waterfall(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> str:
    """
    Handle !dashboard waterfall command - Stage timing waterfall.

    Shows pipeline stage timings with bottleneck detection.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments
        orchestrator: WeavingOrchestrator instance (optional)

    Returns:
        Stage timing summary
    """
    if not WATERFALL_AVAILABLE:
        return "❌ Waterfall visualization not available"

    metrics = get_session_metrics()

    if not metrics.stage_timings:
        return "📊 **Waterfall Dashboard**\n\nNo stage timing data recorded yet. Run some queries first!"

    try:
        # Use most recent stage timing (already a Dict[str, float])
        latest_timing = metrics.stage_timings[-1]

        # render_pipeline_waterfall expects Dict[str, float]
        html = render_pipeline_waterfall(
            stage_durations=latest_timing,
            title="Latest Query Pipeline"
        )

        # Calculate stages for summary display
        stages = []
        cumulative = 0.0
        for stage_name, duration in latest_timing.items():
            stages.append({
                'name': stage_name,
                'start_ms': cumulative,
                'duration_ms': duration
            })
            cumulative += duration

        total_duration = cumulative

        # Build summary
        summary = f"📊 **Pipeline Waterfall**\n\n"
        summary += f"**Total Duration:** {total_duration:.1f}ms\n"
        summary += f"**Stages:** {len(stages)}\n\n"

        # Find bottleneck
        if stages:
            bottleneck = max(stages, key=lambda s: s['duration_ms'])
            bottleneck_pct = (bottleneck['duration_ms'] / total_duration * 100) if total_duration > 0 else 0

            for stage in stages:
                pct = (stage['duration_ms'] / total_duration * 100) if total_duration > 0 else 0
                bar_len = int(pct / 5)  # 20 char max
                bar = "█" * bar_len + "░" * (20 - bar_len)
                marker = " ⚠️" if stage['name'] == bottleneck['name'] and bottleneck_pct > 40 else ""
                summary += f"**{stage['name']}:** {stage['duration_ms']:.1f}ms ({pct:.0f}%){marker}\n"
                summary += f"`{bar}`\n"

            if bottleneck_pct > 40:
                summary += f"\n⚠️ **Bottleneck:** {bottleneck['name']} ({bottleneck_pct:.0f}% of total)"

        return summary

    except Exception as e:
        logger.error(f"Error rendering waterfall: {e}")
        return f"❌ Rendering failed: {str(e)}"


async def handle_dashboard_knowledge(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> str:
    """
    Handle !dashboard knowledge command - Knowledge graph visualization.

    Shows knowledge graph with force-directed layout.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments (--limit=N)
        orchestrator: WeavingOrchestrator instance (optional)

    Returns:
        Knowledge graph summary
    """
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        return "❌ Knowledge graph visualization not available"

    # Parse limit
    limit = 50
    if "--limit=" in args:
        try:
            limit = int(args.split("--limit=")[1].split()[0])
        except (ValueError, IndexError):
            pass

    try:
        # Try to get graph from orchestrator
        kg = None
        if orchestrator and hasattr(orchestrator, '_memory'):
            memory = orchestrator._memory
            if hasattr(memory, '_backend') and hasattr(memory._backend, 'graph'):
                kg = memory._backend.graph

        if kg is None:
            return "📊 **Knowledge Graph**\n\nNo knowledge graph available. Memory backend not initialized."

        # Get graph stats
        nodes = list(kg.nodes())[:limit]
        edges = list(kg.edges())[:limit * 2]

        html = render_knowledge_graph_from_kg(
            kg,
            title="Knowledge Graph",
            subtitle=f"Showing up to {limit} nodes"
        )

        # Build summary
        summary = f"🕸️ **Knowledge Graph**\n\n"
        summary += f"**Total Nodes:** {kg.number_of_nodes()}\n"
        summary += f"**Total Edges:** {kg.number_of_edges()}\n"
        summary += f"**Displayed:** {len(nodes)} nodes, {len(edges)} edges\n"

        # Edge type breakdown
        edge_types: Dict[str, int] = {}
        for u, v, data in kg.edges(data=True):
            edge_type = data.get('relation', 'UNKNOWN')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        if edge_types:
            summary += "\n**Edge Types:**\n"
            for etype, count in sorted(edge_types.items(), key=lambda x: -x[1])[:5]:
                summary += f"  • {etype}: {count}\n"

        return summary

    except Exception as e:
        logger.error(f"Error rendering knowledge graph: {e}")
        return f"❌ Rendering failed: {str(e)}"


async def handle_dashboard_rag(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> str:
    """
    Handle !dashboard rag command - RAG performance dashboard.

    Shows 5-panel RAG performance dashboard.

    Args:
        room: Matrix room
        event: Message event
        args: Optional arguments
        orchestrator: WeavingOrchestrator instance (optional)

    Returns:
        RAG performance summary
    """
    if not RAG_DASHBOARD_AVAILABLE:
        return "❌ RAG dashboard visualization not available"

    metrics = get_session_metrics()

    if not metrics.rag_results:
        return "📊 **RAG Dashboard**\n\nNo RAG queries recorded yet. Use `!rag query` first!"

    try:
        # Convert to RAGResult format
        from HoloLoom.visualization.rag_dashboard import RAGResult

        results = []
        for r in metrics.rag_results:
            results.append(RAGResult(
                response=r.get('response', ''),
                sources=r.get('sources', []),
                confidence=r.get('confidence', 0.0),
                reasoning_mode=r.get('reasoning_mode', 'direct'),
                metadata=r.get('metadata', {})
            ))

        dashboard = RAGDashboard.from_query_history(
            results,
            title="RAG Performance",
            detect_anomalies=True
        )

        html = dashboard.render()

        # Build summary
        summary = f"📚 **RAG Performance Dashboard**\n\n"
        summary += f"**Queries Analyzed:** {len(results)}\n"

        if results:
            avg_conf = sum(r.confidence for r in results) / len(results)
            avg_sources = sum(len(r.sources) for r in results) / len(results)

            summary += f"**Avg Confidence:** {avg_conf:.2f}\n"
            summary += f"**Avg Sources:** {avg_sources:.1f}\n"

            # Mode breakdown
            modes: Dict[str, int] = {}
            for r in results:
                modes[r.reasoning_mode] = modes.get(r.reasoning_mode, 0) + 1

            summary += "\n**Reasoning Modes:**\n"
            for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
                pct = count / len(results) * 100
                summary += f"  • {mode}: {count} ({pct:.0f}%)\n"

        summary += "\n**5 Panels Generated:**\n"
        summary += "  1. Retrieval Quality\n"
        summary += "  2. Latency Waterfall\n"
        summary += "  3. Cache Effectiveness\n"
        summary += "  4. Confidence Trajectory\n"
        summary += "  5. Source Attribution\n"

        return summary

    except Exception as e:
        logger.error(f"Error rendering RAG dashboard: {e}")
        return f"❌ Rendering failed: {str(e)}"


async def handle_dashboard_help(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !dashboard help command.

    Returns:
        Help message
    """
    return """📊 **Dashboard Commands**

**Visualization Types:**
• `!dashboard confidence` - Confidence trajectory over time
• `!dashboard cache` - Cache effectiveness gauge
• `!dashboard waterfall` - Pipeline stage timing
• `!dashboard knowledge` - Knowledge graph network
• `!dashboard rag` - RAG performance (5 panels)

**Options:**
• `!dashboard confidence --detect-anomalies` - Enable anomaly detection
• `!dashboard confidence --title=MyChart` - Custom title
• `!dashboard knowledge --limit=100` - Limit nodes shown

**Availability:**
• confidence: """ + ("✓" if CONFIDENCE_AVAILABLE else "✗") + """
• cache: """ + ("✓" if CACHE_GAUGE_AVAILABLE else "✗") + """
• waterfall: """ + ("✓" if WATERFALL_AVAILABLE else "✗") + """
• knowledge: """ + ("✓" if KNOWLEDGE_GRAPH_AVAILABLE else "✗") + """
• rag: """ + ("✓" if RAG_DASHBOARD_AVAILABLE else "✗") + """

**Notes:**
- Dashboards track session metrics automatically
- Use `!dashboard reset` to clear session data
- HTML dashboards can be saved for sharing

**Examples:**
```
!dashboard confidence
!dashboard cache
!dashboard waterfall
!dashboard knowledge --limit=50
!dashboard rag
```
"""


async def handle_dashboard_reset(
    room: 'MatrixRoom',
    event: 'RoomMessageText',
    args: str
) -> str:
    """
    Handle !dashboard reset command - Reset session metrics.

    Returns:
        Confirmation message
    """
    metrics = get_session_metrics()
    old_count = metrics.total_queries
    metrics.reset()

    return f"🔄 **Session Reset**\n\nCleared {old_count} queries from session metrics."


# ============================================================================
# Registration
# ============================================================================

def register_visualization_handlers(
    bot,
    orchestrator: Optional['WeavingOrchestrator'] = None
) -> None:
    """
    Register all visualization command handlers.

    Args:
        bot: Matrix bot instance
        orchestrator: WeavingOrchestrator instance (optional)
    """
    if not MATRIX_AVAILABLE:
        logger.warning("Matrix not available, visualization handlers not registered")
        return

    logger.info("Registering visualization command handlers")

    # Define command map
    command_map = {
        "confidence": lambda room, event, args: handle_dashboard_confidence(room, event, args, orchestrator),
        "cache": lambda room, event, args: handle_dashboard_cache(room, event, args, orchestrator),
        "waterfall": lambda room, event, args: handle_dashboard_waterfall(room, event, args, orchestrator),
        "knowledge": lambda room, event, args: handle_dashboard_knowledge(room, event, args, orchestrator),
        "rag": lambda room, event, args: handle_dashboard_rag(room, event, args, orchestrator),
        "reset": handle_dashboard_reset,
        "help": handle_dashboard_help
    }

    # Register with bot
    for cmd, handler in command_map.items():
        bot.register_command(f"dashboard {cmd}", handler)

    # Also register !dashboard with no args as help
    bot.register_command("dashboard", handle_dashboard_help)

    logger.info(f"Registered {len(command_map)} dashboard commands")


# ============================================================================
# Decorator-Based Handler Class
# ============================================================================

class VisualizationHandlers:
    """
    Decorator-based ChatOps handlers for visualizations.

    Usage:
        from HoloLoom.chatops.handlers.visualization_handlers import VisualizationHandlers

        handlers = VisualizationHandlers(orchestrator=orchestrator)
        registry.register_instance(handlers)
    """

    def __init__(self, orchestrator: Optional['WeavingOrchestrator'] = None):
        """
        Initialize with optional WeavingOrchestrator instance.

        Args:
            orchestrator: WeavingOrchestrator instance for graph access
        """
        self.orchestrator = orchestrator

    @HandlerRegistry.register(
        command="dashboard confidence",
        description="Show confidence trajectory with anomaly detection",
        usage="!dashboard confidence [--detect-anomalies] [--title=X]",
        category=HandlerCategory.SYSTEM,
        aliases=["dc"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_confidence(self, room, event, args: str) -> str:
        """Handle !dashboard confidence command."""
        return await handle_dashboard_confidence(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="dashboard cache",
        description="Show cache effectiveness gauge",
        usage="!dashboard cache",
        category=HandlerCategory.SYSTEM,
        aliases=["dcache"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_cache(self, room, event, args: str) -> str:
        """Handle !dashboard cache command."""
        return await handle_dashboard_cache(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="dashboard waterfall",
        description="Show pipeline stage timing waterfall",
        usage="!dashboard waterfall",
        category=HandlerCategory.SYSTEM,
        aliases=["dw"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_waterfall(self, room, event, args: str) -> str:
        """Handle !dashboard waterfall command."""
        return await handle_dashboard_waterfall(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="dashboard knowledge",
        description="Show knowledge graph network visualization",
        usage="!dashboard knowledge [--limit=N]",
        category=HandlerCategory.SYSTEM,
        aliases=["dk", "graph"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_knowledge(self, room, event, args: str) -> str:
        """Handle !dashboard knowledge command."""
        return await handle_dashboard_knowledge(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="dashboard rag",
        description="Show RAG performance dashboard (5 panels)",
        usage="!dashboard rag",
        category=HandlerCategory.SYSTEM,
        aliases=["drag"]
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_rag(self, room, event, args: str) -> str:
        """Handle !dashboard rag command."""
        return await handle_dashboard_rag(room, event, args, self.orchestrator)

    @HandlerRegistry.register(
        command="dashboard reset",
        description="Reset session metrics",
        usage="!dashboard reset",
        category=HandlerCategory.SYSTEM,
        hidden=True
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_reset(self, room, event, args: str) -> str:
        """Handle !dashboard reset command."""
        return await handle_dashboard_reset(room, event, args)

    @HandlerRegistry.register(
        command="dashboard help",
        description="Show dashboard command help",
        usage="!dashboard help",
        category=HandlerCategory.SYSTEM,
        hidden=True
    ) if REGISTRY_AVAILABLE else lambda f: f
    async def handle_help(self, room, event, args: str) -> str:
        """Handle !dashboard help command."""
        return await handle_dashboard_help(room, event, args)
