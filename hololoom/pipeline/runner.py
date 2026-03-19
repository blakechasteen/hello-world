"""
Pipeline Runner — Generic level-aware executor.

Runs stages in dependency-resolved parallel levels.
Works with any typed context. RunPolicy shapes behavior.

One execute loop. No wrapper classes. No inheritance.
"""

import asyncio
import time
from typing import Any, Generic, TypeVar

from hololoom.pipeline.cache import StageCache
from hololoom.pipeline.resolver import resolve_levels
from hololoom.pipeline.types import DefaultPolicy

C = TypeVar('C')


class PipelineRunner(Generic[C]):
    """
    Runs stages in dependency-resolved parallel levels.

    Args:
        stages: name -> stage instance mapping
        pre_satisfied: fields available before any stage runs
        policy: runtime behavior hooks (skip, abort, adapt, observe)
        cache: optional stage-level content-addressed cache
    """

    def __init__(
        self,
        stages: dict[str, Any],
        pre_satisfied: frozenset[str] = frozenset(),
        policy: Any | None = None,
        cache: StageCache | None = None,
    ):
        self.stages = stages
        self.pre_satisfied = pre_satisfied
        self.levels = resolve_levels(stages, pre_satisfied)
        self.policy = policy or DefaultPolicy()
        self.cache = cache

    async def execute(self, ctx: C) -> C:
        """
        Execute all stages in dependency order.

        Stages at the same level run concurrently via asyncio.gather.
        Failures are caught and routed through the policy.
        The pipeline can re-resolve remaining levels mid-flight
        if the policy returns a new stage set from on_level_complete.
        """
        idx = 0
        while idx < len(self.levels):
            level = self.levels[idx]
            runnable = [
                n for n in level
                if not self.policy.should_skip(n, ctx)
            ]
            if not runnable:
                idx += 1
                continue

            # Always use gather — unified failure semantics for 1 or N stages.
            results = await asyncio.gather(
                *(self._run_stage(n, ctx) for n in runnable),
                return_exceptions=True,
            )
            for name, result in zip(runnable, results):
                if isinstance(result, Exception):
                    ctx.add_error(f"Stage '{name}' failed: {result}")
                    if self.policy.should_abort_on_failure(name):
                        ctx.blocked_by = name
                        break

            if ctx.is_blocked:
                break

            override = self.policy.on_level_complete(level, ctx)
            if override is not None:
                # Re-resolve remaining levels with current state as base.
                active = {k: v for k, v in self.stages.items() if k in override}
                completed_writes = frozenset().union(
                    *(self.stages[n].writes for l in self.levels[:idx + 1] for n in l)
                )
                new_levels = resolve_levels(active, self.pre_satisfied | completed_writes)
                self.levels = self.levels[:idx + 1] + new_levels

            idx += 1
        return ctx

    async def _run_stage(self, name: str, ctx: C) -> None:
        """Run a single stage with caching and lifecycle hooks."""
        stage = self.stages[name]
        self.policy.on_stage_start(name, ctx)

        # Cache check (skip for side-effect stages)
        if self.cache is not None and stage.cacheable:
            cache_key = self.cache.hash_inputs(ctx, stage.reads)
            cached = self.cache.get(name, cache_key)
            if cached is not None:
                for field_name, value in cached.items():
                    setattr(ctx, field_name, value)
                self.policy.on_stage_complete(name, ctx, 0.0, None)
                return
        else:
            cache_key = None

        t0 = time.perf_counter()
        error = None
        try:
            await stage.execute(ctx)

            # Cache store
            if cache_key and stage.writes:
                self.cache.set(
                    name, cache_key,
                    {f: getattr(ctx, f) for f in stage.writes},
                )
        except Exception as e:
            error = e
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.policy.on_stage_complete(name, ctx, ms, error)

    def explain(self) -> str:
        """Human-readable execution plan. Like make --dry-run for pipelines."""
        lines = []
        for i, level in enumerate(self.levels):
            names = ', '.join(level)
            par = '  — PARALLEL' if len(level) > 1 else ''
            lines.append(f"  Level {i}: [{names}]{par}")
        return '\n'.join(lines)

    def explain_stage(self, name: str) -> str:
        """Show a single stage's dependencies and position."""
        stage = self.stages[name]
        level_idx = next(i for i, l in enumerate(self.levels) if name in l)
        peers = [n for n in self.levels[level_idx] if n != name]
        writers = {
            f: next(
                (n for n, s in self.stages.items() if f in s.writes),
                'pre_satisfied',
            )
            for f in stage.reads
        }
        lines = [
            f"  {name}:",
            f"    reads:     {dict(writers)}",
            f"    writes:    {stage.writes}",
            f"    level:     {level_idx}"
            + (f" (parallel with {', '.join(peers)})" if peers else ""),
            f"    cacheable: {stage.cacheable}",
        ]
        return '\n'.join(lines)
