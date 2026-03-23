"""
IdentityLayer — the autoresearch loop.

Subscribes to HEARTBEAT. On each pulse:
    1. Collect status from ALL runtime layers
    2. Compute identity eval metrics (coherence, groundedness)
    3. Write heartbeat.md — the convergence of all vital signs
    4. Check program.md thresholds
    5. If threshold crossed → propose mutation via MutationProposer

heartbeat.md IS the nervous system's pulse readout. It contains
autonomic ticks, cron jobs, memory stats, AND identity eval metrics.
The identity eval system we built (agents/eval/) feeds into this.

The loop:
    observe → measure → write heartbeat → check thresholds → propose → loop
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .layer import LayerState

logger = logging.getLogger(__name__)


class IdentityLayer:
    """The autoresearch loop. Observes, measures, proposes."""

    name = "identity"

    def __init__(
        self,
        agent_name: str = "promptly",
        identities_root: Path | None = None,
    ):
        self._agent = agent_name
        self._root = identities_root
        self._runtime: Any = None
        self._rolling: Any = None  # RollingMetrics, lazy
        self._proposer: Any = None  # MutationProposer, lazy
        self._state = LayerState.INIT
        self._pulse_count = 0
        self._last_groundedness = 0.0
        self._last_coherence = 0.0
        self._peak_groundedness = 0.0

    async def start(self, runtime: Any) -> None:
        """Subscribe to HEARTBEAT signals."""
        self._runtime = runtime

        # Resolve identities root
        if self._root is None:
            try:
                from hololoom.agents.eval.identity_loader import IdentityLoader
                self._root = IdentityLoader().root
            except ImportError:
                logger.debug("Identity loader unavailable")

        # Initialize rolling metrics
        try:
            from hololoom.agents.eval.runner import RollingMetrics
            self._rolling = RollingMetrics(window_size=100)
        except ImportError:
            logger.debug("Eval runner unavailable — metrics disabled")

        # Initialize mutation proposer
        try:
            from hololoom.agents.eval.mutations import MutationProposer
            self._proposer = MutationProposer(
                agent_name=self._agent,
                root=self._root,
            )
        except ImportError:
            logger.debug("Mutation proposer unavailable")

        # Subscribe to heartbeat
        if runtime.bus is not None:
            try:
                from ..bus.signal import SignalKind
                runtime.bus.on(SignalKind.HEARTBEAT, self._on_heartbeat)
            except Exception as e:
                logger.warning("IdentityLayer bus subscription failed: %s", e)

        self._state = LayerState.RUNNING
        logger.info("IdentityLayer started: agent=%s", self._agent)

    async def stop(self) -> None:
        self._state = LayerState.STOPPED

    async def _on_heartbeat(self, signal: Any) -> None:
        """The pulse. Every 60s, observe everything and write heartbeat.md."""
        self._pulse_count += 1
        tick = signal.payload.get("tick", 0) if hasattr(signal, "payload") else 0

        # 1. Collect layer status from runtime
        layers_status = {}
        if self._runtime:
            try:
                layers_status = self._runtime.status().get("layers", {})
            except Exception:
                pass

        temporal = layers_status.get("temporal", {})
        memory = layers_status.get("memory", {})

        # 2. Compute eval metrics
        g = 0.0
        c = 0.0
        n = 0
        if self._rolling:
            g = self._rolling.groundedness()
            c = self._rolling.coherence()
            n = len(self._rolling.results)
            self._last_groundedness = g
            self._last_coherence = c
            self._peak_groundedness = max(self._peak_groundedness, g)

        # 3. Write heartbeat.md
        self._write_heartbeat(tick, temporal, memory, g, c, n)

        # 4. Check thresholds and propose mutations
        self._check_thresholds(g, c, n)

    def _write_heartbeat(
        self,
        tick: int,
        temporal: dict,
        memory: dict,
        groundedness: float,
        coherence: float,
        n_results: int,
    ) -> None:
        """Write the multi-layered heartbeat.md."""
        if self._root is None:
            return

        agent_dir = self._root / self._agent
        if not agent_dir.exists():
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Format cron jobs table
        cron_jobs = temporal.get("cron_jobs", {})
        if cron_jobs:
            cron_lines = ["| Job | Interval | Runs |", "|-----|----------|------|"]
            for name, info in cron_jobs.items():
                interval = info.get("interval", "?")
                runs = info.get("run_count", 0)
                cron_lines.append(f"| {name} | {interval}s | {runs} |")
            cron_table = "\n".join(cron_lines)
        else:
            cron_table = "(no cron data)"

        # Memory stats
        bus_items = memory.get("bus_items", "-")
        stats = memory.get("stats", {})
        decayed = stats.get("decayed", 0)
        promoted = stats.get("promoted", 0)

        content = f"""# heartbeat.md - {self._agent.capitalize()}
## Last updated: {now}

## Autonomic (60s pulse)
- tick: {tick}
- identity_pulses: {self._pulse_count}

## Memory
- bus_items: {bus_items}
- decayed: {decayed} forgotten
- promoted: {promoted} to vault

## Cron (scheduled work)
{cron_table}

## Identity Eval (rolling {n_results})
| Axis | Value |
|------|-------|
| coherence | {coherence:.3f} |
| groundedness | {groundedness:.3f} |
| peak_groundedness | {self._peak_groundedness:.3f} |

## Observations
(auto-generated from layer status)
"""

        try:
            (agent_dir / "heartbeat.md").write_text(content, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write heartbeat.md: %s", e)

    def _check_thresholds(
        self, groundedness: float, coherence: float, n_results: int,
    ) -> None:
        """Check program.md thresholds, propose mutations if crossed."""
        if self._proposer is None:
            return

        # Groundedness drift from peak
        self._proposer.check_groundedness_drift(
            current=groundedness,
            peak=self._peak_groundedness,
            window_size=n_results,
        )

        # Coherence rigidity
        self._proposer.check_coherence_rigidity(coherence)

        # Write any proposals
        if self._proposer.proposals:
            try:
                self._proposer.write_proposals()
            except Exception as e:
                logger.warning("Failed to write proposals: %s", e)

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "agent": self._agent,
            "pulse_count": self._pulse_count,
            "groundedness": self._last_groundedness,
            "coherence": self._last_coherence,
            "peak_groundedness": self._peak_groundedness,
            "rolling_results": (
                len(self._rolling.results) if self._rolling else 0
            ),
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {"state": self._state.value}

    @property
    def state(self) -> LayerState:
        return self._state

    @property
    def rolling(self) -> Any:
        """Access rolling metrics for external probe results injection."""
        return self._rolling
