"""
Post-weave hooks for the weaving orchestrator.

Handles: bandit learning, metrics, dashboard generation, Jenny UI compilation,
production metrics. These are fire-and-forget operations run after the pipeline
completes — they don't affect the Spacetime result.

Note: Spacetime assembly (formerly assemble_spacetime) is now handled by
execute_step9_spacetime_fabric in stages/steps_7_9.py as part of Architecture B.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_post_weave_hooks(
    orchestrator: Any,
    spacetime: Any,
    query: Any,
    collapse_result: Any,
    complexity: Any,
    pattern_spec: Any,
    features: Any,
    thread_ids: list[str],
    stage_timings: dict[str, float],
    duration_ms: float,
    prod_start_time: float | None,
    METRICS_ENABLED: bool,
    metrics: Any,
) -> None:
    """
    Run all post-weave hooks: bandit learning, metrics, dashboards.

    These are fire-and-forget operations that don't affect the Spacetime result.
    """
    # Semantic Bandit Learning Update
    if orchestrator._semantic_bandit:
        try:
            success = collapse_result.confidence >= orchestrator.cfg.semantic_bandit_success_threshold
            orchestrator._semantic_bandit.update(
                tool=collapse_result.tool,
                success=success,
                confidence=collapse_result.confidence
            )
        except Exception as e:
            logger.warning(f"Semantic bandit update failed: {e}")

    # Track metrics
    if METRICS_ENABLED:
        metrics.track_query(
            pattern=pattern_spec.name,
            complexity=complexity.name if complexity else 'unknown',
            duration=duration_ms / 1000.0,
            tool_used=collapse_result.tool
        )
        metrics.track_stage_batch(stage_timings)
        metrics.track_tool_execution(
            tool_name=collapse_result.tool,
            duration=(stage_timings.get('tool_execution', 0)) / 1000.0
        )
        if 'parallel_speedup' in stage_timings and 'parallel_execution_wall_time' in stage_timings:
            metrics.track_parallel_execution(
                stage_group='steps_4_6',
                wall_time=stage_timings.get('parallel_execution_wall_time', 0) / 1000.0,
                speedup=stage_timings.get('parallel_speedup', 1.0)
            )
        metrics.set_confidence(collapse_result.tool, collapse_result.confidence)
        metrics.set_active_threads(pattern_spec.name, len(thread_ids))
        metrics.set_retrieval_context_size(len(getattr(features, '_context_memories_count', 0)))
        if features.motifs:
            metrics.track_motifs(len(features.motifs))

    # Generate dashboard
    if orchestrator.dashboard_constructor:
        try:
            dashboard = orchestrator.dashboard_constructor.construct(spacetime)
            spacetime.metadata['dashboard'] = dashboard
            logger.info(f"[DASHBOARD] Generated {len(dashboard.panels)} panels ({dashboard.layout.value} layout)")
        except Exception as e:
            logger.warning(f"[DASHBOARD] Failed to generate dashboard: {e}")

    # Production Hardening: Record metrics
    if orchestrator.enable_production_hardening and orchestrator.monitor and prod_start_time:
        try:
            prod_latency = (time.time() - prod_start_time) * 1000
            orchestrator.monitor.performance.record_query(
                latency_ms=prod_latency,
                cache_hit=False,
                error=None if collapse_result.confidence >= 0.5 else "LowConfidence"
            )
            if hasattr(spacetime, 'confidence'):
                orchestrator.monitor.learning.record_calibration(
                    ece=abs(spacetime.confidence - 1.0)
                )
        except Exception as e:
            logger.warning(f"[PRODUCTION] Failed to record metrics: {e}")


async def run_jenny_compilation(
    orchestrator: Any,
    spacetime: Any,
    query: Any,
    pattern_spec: Any,
    complexity: Any,
    stage_timings: dict[str, float],
) -> None:
    """Run Jenny generative UI compilation if enabled."""
    if not (orchestrator.enable_jenny and orchestrator.jenny_runtime):
        return

    try:
        jenny_start = time.time()

        if not orchestrator._jenny_started:
            await orchestrator.jenny_runtime.start()
            orchestrator._jenny_started = True

        # Import extraction helpers
        from hololoom.core.orchestrator.jenny.context_builder import build_jenny_panel_context
        from hololoom.core.orchestrator.jenny.panel_detection import detect_jenny_panel_type

        panel_type = detect_jenny_panel_type(
            spacetime, orchestrator.jenny_mrf_compiler, orchestrator.jenny_learner
        )
        panel_context = build_jenny_panel_context(
            query, spacetime, pattern_spec, complexity
        )
        panel = await orchestrator.jenny_runtime.ask(
            query=query.text,
            context=panel_context,
            panel_type=panel_type,
        )

        spacetime.metadata['jenny_panel'] = {
            'spec_id': panel.id,
            'title': panel.title,
            'panel_type': panel.panel_type.value,
            'lifecycle': panel.lifecycle.value,
            'html': panel.html,
            'terminal': panel.terminal,
            'json_data': panel.json_data,
            'actions': panel.actions,
        }
        spacetime.metadata['jenny_panel_id'] = panel.id
        spacetime.metadata['jenny_panel_count'] = 1

        jenny_duration = (time.time() - jenny_start) * 1000
        stage_timings['jenny_compilation'] = jenny_duration

        logger.info(
            f"[JENNY] Generated {panel.panel_type.value} panel "
            f"(id={panel.id[:8]}..., lifecycle={panel.lifecycle.value}, "
            f"{jenny_duration:.1f}ms)"
        )
    except Exception as e:
        logger.warning(f"[JENNY] Failed to generate panel: {e}")
