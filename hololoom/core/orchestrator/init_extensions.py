"""
Orchestrator Initialization Extensions
=======================================
Extended initialization for awareness, conscience, Jenny UI, and semantic state.
Extracted from WeavingOrchestrator.__init__() (March 2026 Refactor).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.config import Config
    from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

logger = logging.getLogger(__name__)


def initialize_awareness(orch: WeavingOrchestrator, awareness_layer: Any | None) -> None:
    """Initialize awareness layer (Consciousness Integration - Phase 1)."""
    orch.awareness_layer = awareness_layer
    if orch.awareness_layer is None and hasattr(orch, 'semantic_spectrum'):
        try:
            from hololoom.memory.awareness_graph import AwarenessGraph
            orch.awareness_layer = AwarenessGraph(
                graph_backend=orch.yarn_graph._graph if hasattr(orch, 'yarn_graph') else None,
                semantic_calculus=orch.semantic_spectrum,
                vector_store=None
            )
            logger.info("Auto-created AwarenessGraph for consciousness integration")
        except Exception as e:
            logger.warning(f"Failed to auto-create AwarenessGraph: {e}")
            orch.awareness_layer = None


def initialize_conscience(
    orch: WeavingOrchestrator,
    enable_conscience: bool,
    conscience_adapter: Any | None,
    conscience_available: bool,
) -> None:
    """Initialize conscience adapter (Phase 2C)."""
    orch.enable_conscience = enable_conscience and conscience_available
    orch.conscience_adapter = conscience_adapter
    if orch.enable_conscience and orch.conscience_adapter is None:
        try:
            from hololoom.agentic.conscience_adapter import AgenticConscienceAdapter
            orch.conscience_adapter = AgenticConscienceAdapter(
                guardrails=orch.guardrails,
                auto_create_guardrails=True,
            )
            logger.info("Auto-created AgenticConscienceAdapter for conscience integration")
        except Exception as e:
            logger.warning(f"Failed to auto-create conscience adapter: {e}")
            orch.conscience_adapter = None
            orch.enable_conscience = False


def initialize_jenny_runtime(
    orch: WeavingOrchestrator,
    enable_jenny: bool,
    jenny_persist_path: str,
    cfg: Config,
    jenny_mrf_available: bool,
) -> None:
    """Initialize Jenny Generative UI Runtime (December 2025 - MVP Week 4)."""
    orch.enable_jenny = enable_jenny
    orch.jenny_runtime = None
    orch._jenny_started = False
    orch.jenny_mrf_compiler = None
    orch.jenny_learner = None

    if not enable_jenny:
        return

    try:
        from hololoom.visualization.jenny_renderer import RenderTarget
        from hololoom.visualization.jenny_runtime import JennyConfig, JennyRuntime

        # Use module-level renderer map
        from hololoom.core.orchestrator.weaving_orchestrator import _get_jenny_renderer_map
        renderer_map = _get_jenny_renderer_map()
        renderer_str = getattr(cfg, 'jenny_default_renderer', 'html')
        default_renderer = renderer_map.get(renderer_str, RenderTarget.HTML)

        jenny_config = JennyConfig(
            default_renderer=default_renderer,
            ledger_persist_path=jenny_persist_path,
            cleanup_interval_seconds=getattr(cfg, 'jenny_cleanup_interval', 60.0),
        )
        orch.jenny_runtime = JennyRuntime(config=jenny_config)

        # Phase 2.1-2.2: MRF compiler with Thompson Sampling learning
        jenny_enable_mrf = getattr(cfg, 'jenny_enable_mrf', True)
        jenny_enable_learning = getattr(cfg, 'jenny_enable_learning', True)
        jenny_learning_path = getattr(cfg, 'jenny_learning_persist_path', jenny_persist_path)

        if jenny_mrf_available and jenny_enable_mrf:
            try:
                from hololoom.visualization.jenny_mrf import create_mrf_compiler
                learning_persist_path = f"{jenny_learning_path}/learning.json"
                orch.jenny_mrf_compiler = create_mrf_compiler(
                    enable_learning=jenny_enable_learning,
                    learning_persist_path=learning_persist_path
                )
                orch.jenny_learner = orch.jenny_mrf_compiler.learner if orch.jenny_mrf_compiler else None
                logger.info(
                    f"Jenny MRF compiler enabled (Phase 2.1-2.2), "
                    f"learning={jenny_enable_learning}, persist={learning_persist_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Jenny MRF compiler: {e}")
                orch.jenny_mrf_compiler = None
                orch.jenny_learner = None

        logger.info(
            f"Jenny UI Runtime enabled (Week 4 - full lifecycle), "
            f"renderer={default_renderer.value}, persist={jenny_persist_path}, "
            f"mrf={'enabled' if orch.jenny_mrf_compiler else 'disabled'}"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Jenny UI Runtime: {e}")
        orch.enable_jenny = False
        orch.jenny_runtime = None
        orch.jenny_mrf_compiler = None
        orch.jenny_learner = None


def initialize_semantic_state(orch: WeavingOrchestrator) -> None:
    """Initialize semantic state tracking (Phase 8 - December 2025)."""
    from hololoom.semantic_calculus import SemanticToolSelector

    orch._previous_semantic_state = None
    orch._query_index = 0
    orch._semantic_tool_selector = None

    if hasattr(orch, 'semantic_spectrum') and orch.semantic_spectrum is not None:
        try:
            orch._semantic_tool_selector = SemanticToolSelector()
            logger.info("SemanticToolSelector initialized for semantic-aware tool selection")
        except Exception as e:
            logger.warning(f"Failed to initialize SemanticToolSelector: {e}")
            orch._semantic_tool_selector = None

    # SemanticAwareBandit
    orch._semantic_bandit = None
    if orch.cfg.enable_semantic_bandit:
        try:
            from hololoom.semantic_calculus.semantic_state import SemanticAwareBandit
            orch._semantic_bandit = SemanticAwareBandit(tools=orch.tool_executor.tools)
            logger.info("SemanticAwareBandit initialized for semantic-aware Thompson Sampling")
        except Exception as e:
            logger.warning(f"Failed to initialize SemanticAwareBandit: {e}")
            orch._semantic_bandit = None
