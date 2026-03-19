"""
Fabric Weaving Stage
====================
Step 9 of the weaving cycle: Weave results into Spacetime fabric.

This stage:
- Detensions warp space
- Creates WeavingTrace with full pipeline metadata
- Assembles final Spacetime artifact with all results
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


class FabricWeavingStage:
    """
    Fabric Weaving Stage (Step 9: Spacetime Assembly).

    Collapses the warp space, creates a WeavingTrace capturing full
    pipeline provenance, and assembles the final Spacetime artifact.
    """

    # Note: also accesses infrastructure fields (errors, warnings, stage_timings,
    # awareness_context) that are initialized on context creation — not declared
    # in reads since they don't affect stage ordering.
    reads: ClassVar[frozenset[str]] = frozenset({
        'query', 'tool_result', 'collapse_result', 'dot_plasma',
        'shards', 'features', 'pattern_spec', 'thread_ids',
        'warp_operations', 'action_plan', 'warp_compute_results',
    })
    writes: ClassVar[frozenset[str]] = frozenset({'spacetime', 'trace'})
    cacheable: ClassVar[bool] = False  # Final assembly, always fresh

    def __init__(
        self,
        warp_space: Any | None = None,
        semantic_cache: Any | None = None,
        logger: logging.Logger | None = None,
    ):
        self.warp_space = warp_space
        self.semantic_cache = semantic_cache
        self.logger = logger or logging.getLogger(__name__)

    async def execute(self, ctx: 'WeavingContext') -> None:
        """Weave all results into Spacetime, writing spacetime and trace to ctx."""
        from hololoom.fabric.spacetime import Artifact, Spacetime, WeavingTrace

        # Detension warp space
        warp_operations = list(ctx.warp_operations) if ctx.warp_operations else []
        if self.warp_space:
            warp_updates = self.warp_space.collapse()
            warp_operations.append(
                (datetime.now().isoformat(), "detension", len(warp_updates))
            )

        # Create WeavingTrace
        end_time = datetime.now()
        start_time = ctx.start_time or end_time
        duration_ms = (end_time - start_time).total_seconds() * 1000

        features = ctx.features
        pattern_spec = ctx.pattern_spec
        collapse_result = ctx.collapse_result

        ctx.trace = WeavingTrace(
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            stage_durations=dict(ctx.stage_timings) if ctx.stage_timings else {},
            motifs_detected=self._extract_motifs(features),
            embedding_scales_used=pattern_spec.scales if pattern_spec else [],
            spectral_features=features.spectral if features else {},
            threads_activated=list(ctx.thread_ids) if ctx.thread_ids else [],
            context_shards_count=len(ctx.shards) if ctx.shards else 0,
            retrieval_mode=pattern_spec.retrieval_mode if pattern_spec else "fast",
            policy_adapter=ctx.action_plan.adapter if ctx.action_plan else None,
            tool_selected=collapse_result.tool if collapse_result else None,
            tool_confidence=collapse_result.confidence if collapse_result else 0.0,
            bandit_statistics=collapse_result.bandit_stats if collapse_result else None,
            warp_operations=warp_operations,
            tensor_field_stats={"threads_tensioned": len(ctx.thread_ids) if ctx.thread_ids else 0},
            errors=list(ctx.errors) if ctx.errors else [],
            warnings=list(ctx.warnings) if ctx.warnings else [],
        )

        # Build metadata
        metadata = self._build_metadata(ctx, pattern_spec)

        # Parse artifacts from tool result
        tool_result = ctx.tool_result or {}
        raw_artifacts = tool_result.get('artifacts', [])
        artifacts = [Artifact.from_dict(a) for a in raw_artifacts]

        # Create Spacetime
        ctx.spacetime = Spacetime(
            query_text=ctx.query.text,
            response=tool_result.get('result', 'No response'),
            tool_used=collapse_result.tool if collapse_result else "none",
            confidence=collapse_result.confidence if collapse_result else 0.0,
            artifacts=artifacts,
            trace=ctx.trace,
            metadata=metadata,
            context_summary=f"{len(ctx.shards) if ctx.shards else 0} shards",
            sources_used=[s.id for s in (ctx.shards or [])[:3]],
        )

        self.logger.info("  [9] Spacetime fabric woven!")

    @staticmethod
    def _extract_motifs(features: Any) -> list[str]:
        """Extract motif patterns from Features object."""
        if not features or not features.motifs:
            return []
        return [
            m.pattern if hasattr(m, 'pattern') else str(m)
            for m in features.motifs
        ]

    def _build_metadata(self, ctx: 'WeavingContext', pattern_spec: Any) -> dict[str, Any]:
        """Build metadata dict for Spacetime."""
        metadata: dict[str, Any] = {}

        if pattern_spec:
            metadata['pattern_card'] = pattern_spec.name
            metadata['execution_mode'] = pattern_spec.card.value if hasattr(pattern_spec.card, 'value') else str(pattern_spec.card)
            metadata['loom_command'] = 'auto'
            metadata['chrono_timeout'] = pattern_spec.pipeline_timeout

        # Semantic cache stats
        if self.semantic_cache:
            cache_stats = self.semantic_cache.get_stats()
            metadata['semantic_cache'] = {
                'enabled': True,
                'hit_rate': cache_stats['cache_hit_rate'],
                'hot_hits': cache_stats['hot_hits'],
                'warm_hits': cache_stats['warm_hits'],
                'cold_misses': cache_stats['cold_misses'],
                'estimated_speedup': cache_stats['estimated_speedup'],
            }
        else:
            metadata['semantic_cache'] = {'enabled': False}

        # Awareness context
        awareness_context = ctx.awareness_context
        if awareness_context:
            metadata['awareness'] = {
                'activation_level': awareness_context.get('activation_level', 0.0),
                'coherence': awareness_context.get('coherence', 0.0),
                'active_nodes': awareness_context.get('active_nodes', 0),
                'shift_detected': awareness_context.get('shift_detected', False),
                'semantic_position': awareness_context.get('semantic_position'),
                'perception_time_ms': awareness_context.get('perception_time_ms', 0.0),
            }

        # Semantic state from warp compute
        warp_results = ctx.warp_compute_results
        if warp_results and warp_results.get('semantic_state'):
            metadata['semantic_state'] = warp_results['semantic_state']

        return metadata

    def get_stage_name(self) -> str:
        return "fabric_weaving"

    def get_required_context_keys(self) -> list[str]:
        return ["tool_result", "collapse_result"]

    def get_output_context_keys(self) -> list[str]:
        return ["spacetime", "trace"]
