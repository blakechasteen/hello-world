"""
Jenny visualization integration — conversation graphs, spatial updates,
and background visualization passes.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .config import (
    GRAPH_IDLE_TIMEOUT,
)
from .refinement import refinements

logger = logging.getLogger(__name__)

# ============================================================================
# Jenny Runtime (lazy singleton)
# ============================================================================

_jenny_runtime = None
_jenny_available: bool | None = None


async def _get_jenny():
    """Lazy-init Jenny runtime. Returns None if unavailable."""
    global _jenny_runtime, _jenny_available
    if _jenny_available is False:
        return None
    if _jenny_runtime is not None:
        return _jenny_runtime

    try:
        from hololoom.visualization import JennyConfig, JennyRuntime
        runtime = JennyRuntime(JennyConfig(
            auto_cleanup=False,  # Matrix IS the dissolution mechanism
            enable_ledger=False,
            enable_streaming=False,
        ))
        await runtime.start()
        _jenny_runtime = runtime
        _jenny_available = True
        logger.info("Jenny runtime initialized for visualization pass")
        return runtime
    except Exception as e:
        _jenny_available = False
        logger.info("Jenny not available (visualization disabled): %s", e)
        return None


# ============================================================================
# Single-turn Jenny pass (Stage 1)
# ============================================================================

async def _run_jenny_pass(
    jenny_id: str,
    query: str,
    response: str,
    room_id: str,
    duration_ms: float,
) -> None:
    """
    Background Jenny visualization pass.

    Takes the query + LLM response, runs it through Jenny to detect if
    structured formatting (table, confidence bar, sources, etc.) would
    add value. If so, renders to Matrix-safe HTML and stores in the
    RefinementStore for nanoclaw to pick up via the same polling mechanism.

    Only sends panels that add visual value — never re-wraps plain text.
    """
    try:
        jenny = await _get_jenny()
        if jenny is None:
            refinements.skip(jenny_id, "jenny not available")
            return

        from hololoom.visualization.matrix_renderer import render_if_useful

        # Ask Jenny to auto-detect the best panel type for this content
        panel = await jenny.ask(
            query=query,
            context={
                "text": response,
                "response": response,
                "room_id": room_id,
                "duration_ms": duration_ms,
            },
        )

        # Check if the panel adds value over raw text in Matrix
        matrix_html = render_if_useful(panel)

        if matrix_html is None:
            refinements.converged(jenny_id, "panel type not useful for Matrix")
            logger.debug("Jenny pass (%s): skipped — %s panel not useful",
                         jenny_id, panel.panel_type.value)
            return

        # Store the Matrix-safe HTML as a "refinement" result
        # Nanoclaw will pick this up and send it as formatted_body
        refinements.complete(
            jenny_id,
            refinement=matrix_html,
            model="jenny",
        )
        logger.info(
            "Jenny pass (%s): %s panel rendered (%d chars HTML)",
            jenny_id, panel.panel_type.value, len(matrix_html),
        )

    except Exception as e:
        refinements.skip(jenny_id, f"jenny error: {e}")
        logger.warning("Jenny pass (%s) failed: %s", jenny_id, e)


# ============================================================================
# Conversation Graph (Stage 2 — conversation-aware visualization)
# ============================================================================

_conversation_graphs: dict[str, ConversationGraph] = {}
_conv_analyzer = None
_conv_strategy = None


def _get_conv_analyzer():
    """Lazy-init conversation analyzer."""
    global _conv_analyzer
    if _conv_analyzer is None:
        try:
            from hololoom.visualization.conversation_analyzer import ConversationAnalyzer
            _conv_analyzer = ConversationAnalyzer()
        except ImportError:
            pass
    return _conv_analyzer


def _get_conv_strategy():
    """Lazy-init conversation strategy."""
    global _conv_strategy
    if _conv_strategy is None:
        try:
            from hololoom.visualization.conversation_strategy import (
                ConversationVisualizationStrategy,
            )
            _conv_strategy = ConversationVisualizationStrategy()
        except ImportError:
            pass
    return _conv_strategy


def _update_conversation_graph(room_id: str, user_msg: str, assistant_msg: str) -> None:
    """
    Update the conversation graph for a room after each turn.

    Extracts topics/entities and adds them to the in-memory graph.
    Detects session boundaries (15min gap) and resets when needed.
    """
    analyzer = _get_conv_analyzer()
    if analyzer is None:
        return

    try:
        from hololoom.visualization.conversation_graph import ConversationGraph

        graph = _conversation_graphs.get(room_id)
        if graph is None:
            graph = ConversationGraph(room_id=room_id)
            _conversation_graphs[room_id] = graph

        # Check session boundary
        if analyzer.detect_session_boundary(graph.last_updated, datetime.now()):
            logger.info("Conversation graph reset for room %s (session boundary)", room_id)
            graph.reset()

        # Extract topics/entities from both sides of the conversation
        combined = f"{user_msg} {assistant_msg}"
        topics = analyzer.extract_topics(combined)
        entities = analyzer.extract_entities(user_msg)

        # Check for explicit comparisons
        comparison = analyzer.detect_comparisons(user_msg, graph)
        if comparison:
            graph.add_comparison_edge(comparison[0], comparison[1])

        graph.add_turn(topics, entities, graph.turn_count + 1)

        logger.debug(
            "Conversation graph: room=%s turn=%d topics=%d entities=%d nodes=%d",
            room_id, graph.turn_count, len(topics), len(entities), len(graph.nodes),
        )
    except Exception as e:
        logger.debug("Conversation graph update failed: %s", e)


def _gc_conversation_graphs() -> None:
    """Remove idle conversation graphs (>30min since last update)."""
    now = datetime.now()
    expired = [
        room_id for room_id, graph in _conversation_graphs.items()
        if (now - graph.last_updated).total_seconds() > GRAPH_IDLE_TIMEOUT
    ]
    for room_id in expired:
        del _conversation_graphs[room_id]
        logger.debug("Conversation graph GC: removed room %s", room_id)


def get_conversation_graphs() -> dict[str, ConversationGraph]:
    """Expose conversation graphs dict for route handlers."""
    return _conversation_graphs


async def _run_jenny_conversation_pass(
    jenny_id: str,
    graph: ConversationGraph,
    room_id: str,
) -> None:
    """
    Background Jenny conversation visualization pass.

    Uses the accumulated conversation graph to produce a trajectory-aware
    panel (topic summary, comparison table, etc.). Falls back to skipping
    if the conversation shape doesn't warrant visualization.
    """
    try:
        jenny = await _get_jenny()
        analyzer = _get_conv_analyzer()
        strategy = _get_conv_strategy()

        if jenny is None or analyzer is None or strategy is None:
            refinements.skip(jenny_id, "jenny conversation components not available")
            return

        from hololoom.visualization.jenny_spec import PanelTypeJenny
        from hololoom.visualization.matrix_renderer import render_panel

        # Detect conversation trajectory
        trajectory = analyzer.detect_trajectory(graph)

        # Ask strategy what to show
        panel_type = strategy.select(trajectory, graph, graph.turn_count)
        if panel_type is None:
            refinements.converged(jenny_id, f"trajectory={trajectory}, no viz needed")
            return

        # Build content from graph based on panel type
        if panel_type == PanelTypeJenny.COMPARISON:
            # Find the two heaviest topics for comparison
            sorted_nodes = sorted(
                graph.nodes.values(), key=lambda n: n.weight, reverse=True,
            )
            if len(sorted_nodes) >= 2:
                content = graph.to_comparison_spec(
                    sorted_nodes[0].name, sorted_nodes[1].name,
                )
            else:
                content = graph.to_table_spec()
                panel_type = PanelTypeJenny.TABLE
        else:
            content = graph.to_table_spec()

        # Generate panel via Jenny
        panel = await jenny.ask(
            query=f"Conversation summary ({graph.turn_count} turns)",
            context=content,
            panel_type=panel_type,
        )

        # Render to Matrix-safe HTML
        matrix_html = render_panel(panel)

        if not matrix_html or len(matrix_html.strip()) < 10:
            refinements.converged(jenny_id, "conversation panel too small")
            return

        # Mark graph as rendered (prevents re-rendering same state)
        graph.mark_rendered()

        refinements.complete(jenny_id, refinement=matrix_html, model="jenny-conv")
        logger.info(
            "Jenny conversation pass (%s): %s panel, trajectory=%s, %d nodes, %d chars HTML",
            jenny_id, panel_type.value, trajectory, len(graph.nodes), len(matrix_html),
        )

    except Exception as e:
        refinements.skip(jenny_id, f"jenny conversation error: {e}")
        logger.warning("Jenny conversation pass (%s) failed: %s", jenny_id, e)


# ============================================================================
# Spatial Scene (Stage 3 — spatial conversation visualization)
# ============================================================================

_spatial_dispatchers: dict[str, SpatialSceneDispatcher] = {}


async def _run_spatial_update(room_id: str, graph: ConversationGraph) -> None:
    """Push spatial scene update to connected AR/XR clients."""
    try:
        from hololoom.apps.server.spatial_websocket import get_spatial_ws_manager
        from hololoom.visualization.spatial_dispatcher import SpatialSceneDispatcher

        dispatcher = _spatial_dispatchers.get(room_id)
        if dispatcher is None:
            dispatcher = SpatialSceneDispatcher(room_id)
            _spatial_dispatchers[room_id] = dispatcher

        analyzer = _get_conv_analyzer()
        from hololoom.visualization.conversation_graph import Trajectory
        trajectory = analyzer.detect_trajectory(graph) if analyzer else Trajectory.EXPLORING

        state = dispatcher.update_from_graph(graph, trajectory)
        if state:
            ws_manager = get_spatial_ws_manager()
            if ws_manager:
                await ws_manager.broadcast_scene_update(room_id, state)
                logger.debug(
                    "Spatial update broadcast: room=%s trajectory=%s",
                    room_id, trajectory,
                )
    except Exception as e:
        logger.debug("Spatial update failed for room %s: %s", room_id, e)


def _gc_spatial_dispatchers() -> None:
    """Remove spatial dispatchers for rooms whose graphs were GC'd."""
    expired = [
        room_id for room_id in _spatial_dispatchers
        if room_id not in _conversation_graphs
    ]
    for room_id in expired:
        del _spatial_dispatchers[room_id]
        logger.debug("Spatial dispatcher GC: removed room %s", room_id)


def get_spatial_dispatchers() -> dict[str, SpatialSceneDispatcher]:
    """Expose spatial dispatchers dict for route handlers."""
    return _spatial_dispatchers
