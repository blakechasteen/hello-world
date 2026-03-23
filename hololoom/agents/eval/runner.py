"""
Evaluation runner — executes probes against agents under different identity conditions.

The runner is the experiment harness. It:
    1. Loads identity files per condition (BARE/FAST/FULL/RESEARCH or none)
    2. Sends probes to the agent
    3. Collects responses and scores them
    4. Emits signals to the UnifiedBus for continuous tracking
    5. Updates heartbeat.md with rolling metrics

Designed to wire into existing infrastructure:
    - UnifiedBus for signal emission
    - ReflectionBuffer for outcome tracking
    - Thompson Sampling for learning which conditions work
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hololoom.agents.eval.axes import (
    coherence_score,
    differentiation_score,
    groundedness_score,
    composite_score,
    extract_style,
)
from hololoom.agents.eval.identity_loader import IdentityLoader, IdentityMode
from hololoom.agents.eval.probes import Probe, ProbeResult, ProbeSet

logger = logging.getLogger(__name__)


# ============================================================================
# Agent protocol — anything that can generate a response
# ============================================================================

class AgentBackend(Protocol):
    """Protocol for agent backends. Anything with a generate method."""
    async def generate(self, system_prompt: str, user_message: str) -> str: ...


# ============================================================================
# Condition — what identity configuration to test
# ============================================================================

@dataclass
class Condition:
    """An experimental condition — which identity files to load."""
    name: str
    mode: IdentityMode | None  # None = no identity files (baseline)
    description: str = ""


STANDARD_CONDITIONS = [
    Condition("baseline", None, "No identity files — raw model"),
    Condition("bare", IdentityMode.BARE, "agent.md only"),
    Condition("fast", IdentityMode.FAST, "agent.md + soul.md"),
    Condition("full", IdentityMode.FULL, "agent.md + soul.md + tools.md + USER.md"),
    Condition("research", IdentityMode.RESEARCH, "All files including heartbeat + tasks"),
]


# ============================================================================
# Rolling window for continuous measurement
# ============================================================================

@dataclass
class RollingMetrics:
    """Sliding window metrics — no epochs, always current."""
    window_size: int = 100
    results: list[ProbeResult] = field(default_factory=list)

    def add(self, result: ProbeResult) -> None:
        self.results.append(result)
        if len(self.results) > self.window_size:
            self.results = self.results[-self.window_size:]

    def groundedness(self) -> float:
        return groundedness_score(self.results) if self.results else 0.0

    def coherence(self) -> float:
        return coherence_score(self.results) if self.results else 0.0

    def composite(self, differentiation: float = 0.5) -> float:
        """Composite requires cross-agent data for differentiation."""
        g = self.groundedness()
        c = self.coherence()
        return composite_score(c, differentiation, g)

    def by_file(self, filename: str) -> list[ProbeResult]:
        return [r for r in self.results if r.probe.target_file == filename]


# ============================================================================
# Runner
# ============================================================================

@dataclass
class RunSummary:
    """Summary of a complete evaluation run."""
    agent_name: str
    condition: str
    probes_run: int
    groundedness: float
    coherence: float
    results: list[ProbeResult] = field(default_factory=list)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalRunner:
    """
    Runs identity probes against an agent under different conditions.

    Usage:
        runner = EvalRunner(backend=my_ollama_client)
        summary = await runner.run(
            agent_name="promptly",
            probe_set=PROMPTLY_PROBES,
            condition=Condition("full", IdentityMode.FULL),
        )
    """

    def __init__(
        self,
        backend: AgentBackend,
        loader: IdentityLoader | None = None,
        bus: Any | None = None,  # Optional UnifiedBus for signal emission
    ):
        self.backend = backend
        self.loader = loader or IdentityLoader()
        self.bus = bus
        self.metrics: dict[str, RollingMetrics] = {}  # per-agent rolling window

    async def run(
        self,
        agent_name: str,
        probe_set: ProbeSet,
        condition: Condition,
        runs_per_probe: int = 1,
    ) -> RunSummary:
        """Run all probes in a set against an agent under a condition."""
        start = time.monotonic()

        # Build system prompt
        if condition.mode is not None:
            identity = self.loader.load(agent_name, mode=condition.mode)
            system_prompt = identity.prompt
        else:
            system_prompt = f"You are {agent_name}."

        # Ensure rolling metrics exist
        key = f"{agent_name}:{condition.name}"
        if key not in self.metrics:
            self.metrics[key] = RollingMetrics()

        results: list[ProbeResult] = []

        for probe in probe_set.probes:
            for run_idx in range(runs_per_probe):
                result = await self._run_single(
                    system_prompt, probe, condition.name,
                )
                results.append(result)
                self.metrics[key].add(result)

                # Emit to bus if available
                if self.bus is not None:
                    await self._emit(agent_name, condition.name, probe, result)

        elapsed = (time.monotonic() - start) * 1000

        summary = RunSummary(
            agent_name=agent_name,
            condition=condition.name,
            probes_run=len(results),
            groundedness=self.metrics[key].groundedness(),
            coherence=self.metrics[key].coherence(),
            results=results,
            duration_ms=elapsed,
        )

        logger.info(
            "%s [%s]: groundedness=%.3f coherence=%.3f (%d probes, %.0fms)",
            agent_name, condition.name,
            summary.groundedness, summary.coherence,
            summary.probes_run, summary.duration_ms,
        )

        return summary

    async def run_all_conditions(
        self,
        agent_name: str,
        probe_set: ProbeSet,
        conditions: list[Condition] | None = None,
        runs_per_probe: int = 1,
    ) -> list[RunSummary]:
        """Run probes under all conditions for A/B/C/D comparison."""
        conditions = conditions or STANDARD_CONDITIONS
        summaries = []
        for condition in conditions:
            summary = await self.run(agent_name, probe_set, condition, runs_per_probe)
            summaries.append(summary)
        return summaries

    async def _run_single(
        self,
        system_prompt: str,
        probe: Probe,
        condition_name: str,
    ) -> ProbeResult:
        """Execute a single probe."""
        start = time.monotonic()
        try:
            response = await self.backend.generate(system_prompt, probe.query)
        except Exception as e:
            logger.warning("Probe %s failed: %s", probe.id, e)
            response = f"[ERROR: {e}]"

        elapsed = (time.monotonic() - start) * 1000

        result = ProbeResult(
            probe=probe,
            response=response,
            score=0.0,  # Scored below
            latency_ms=elapsed,
            condition=condition_name,
        )

        # Score using groundedness scorer
        from hololoom.agents.eval.axes import _score_single
        result.score = _score_single(result)

        return result

    async def _emit(
        self,
        agent_name: str,
        condition: str,
        probe: Probe,
        result: ProbeResult,
    ) -> None:
        """Emit probe result to UnifiedBus."""
        try:
            from hololoom.core.bus.signal import Signal, SignalKind
            await self.bus.emit(Signal(
                kind=SignalKind.EXECUTION_COMPLETE,
                source=f"eval:{agent_name}",
                data={
                    "type": "identity_probe",
                    "agent": agent_name,
                    "condition": condition,
                    "probe_id": probe.id,
                    "category": probe.category.value,
                    "score": result.score,
                    "latency_ms": result.latency_ms,
                    "target_file": probe.target_file,
                    "target_constraint": probe.target_constraint,
                },
            ))
        except Exception as e:
            logger.debug("Bus emission failed (non-critical): %s", e)
