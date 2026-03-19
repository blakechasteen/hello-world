"""
Tier State — Infrastructure Health as Spectrum
===============================================

Not binary (up/down) — a four-level spectrum based on how many
GPU servers are reachable. The deliberation cycle adapts its
behavior at each tier.

    FULL (3 GPUs)     → full propose/challenge/resolve
    PARTIAL (1-2 GPUs)→ reduced deliberation
    DEGRADED (0 GPUs) → local 27B, all roles collapsed
    DARK (no inference)→ queue everything

Emits transition events so orchestrator can react:
    FULL→DEGRADED: adjust gate thresholds, log
    DEGRADED→FULL: drain deferred queue
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from hololoom.core.deep_thinking.client import DeepClient
from hololoom.core.deep_thinking.config import DeepThinkingConfig

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    """Infrastructure tier — number of available GPU servers."""
    DARK = 0       # no inference at all
    DEGRADED = 1   # local 27B only (rig unreachable)
    PARTIAL = 2    # 1-2 GPUs alive
    FULL = 3       # all 3 GPUs alive


@dataclass
class TierTransition:
    """Record of a tier change."""
    previous: Tier
    current: Tier
    available_gpus: list[int]
    timestamp: float = 0.0


TransitionCallback = Callable[[TierTransition], None]


class TierState:
    """
    Probes GPU server health, maintains tier, emits transitions.

    Each GPU server is probed independently. Tier = f(healthy count):
        3 healthy → FULL
        1-2 healthy → PARTIAL
        0 healthy, Ollama up → DEGRADED
        nothing → DARK
    """

    def __init__(self, config: DeepThinkingConfig, client: DeepClient):
        self.config = config
        self.client = client

        self.tier: Tier = Tier.DARK
        self.available_gpus: list[int] = []
        self.local_alive: bool = False

        self._callbacks: list[TransitionCallback] = []
        self._health_task: asyncio.Task | None = None

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register a callback for tier changes."""
        self._callbacks.append(callback)

    async def probe(self) -> Tier:
        """
        Probe all endpoints and update tier.

        Returns the current tier after probing.
        """
        # Probe GPU servers in parallel
        gpu_tasks = [
            self.client.is_alive(endpoint)
            for endpoint in self.config.gpu_servers
        ]
        gpu_results = await asyncio.gather(*gpu_tasks, return_exceptions=True)

        alive_gpus = [
            i for i, result in enumerate(gpu_results)
            if result is True
        ]

        # Probe local Ollama
        self.local_alive = await self.client.is_alive(self.config.local_endpoint)

        # Determine tier
        prev = self.tier
        self.available_gpus = alive_gpus

        if len(alive_gpus) == 3:
            self.tier = Tier.FULL
        elif len(alive_gpus) >= 1:
            self.tier = Tier.PARTIAL
        elif self.local_alive:
            self.tier = Tier.DEGRADED
        else:
            self.tier = Tier.DARK

        # Emit transition if changed
        if self.tier != prev:
            transition = TierTransition(
                previous=prev,
                current=self.tier,
                available_gpus=alive_gpus,
            )
            logger.info(
                f"Tier transition: {prev.name} → {self.tier.name} "
                f"(GPUs: {alive_gpus}, local: {self.local_alive})"
            )
            for cb in self._callbacks:
                try:
                    cb(transition)
                except Exception as e:
                    logger.error(f"Tier transition callback error: {e}")

        return self.tier

    async def health_loop(self) -> None:
        """Background loop that probes health periodically."""
        while True:
            try:
                await self.probe()
            except Exception as e:
                logger.error(f"Health probe error: {e}")
            await asyncio.sleep(self.config.health_interval_s)

    def start(self) -> None:
        """Start the background health loop."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self.health_loop())
            logger.info(
                f"Tier health loop started "
                f"(interval: {self.config.health_interval_s}s)"
            )

    def stop(self) -> None:
        """Stop the background health loop."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            logger.info("Tier health loop stopped")

    def endpoint_for_gpu(self, gpu_idx: int) -> str | None:
        """Get the endpoint for a specific GPU, or None if unavailable."""
        if gpu_idx in self.available_gpus:
            return self.config.gpu_servers[gpu_idx]
        return None

    def best_available_endpoint(self) -> str | None:
        """Get any available GPU endpoint, preferring lower indices."""
        for idx in self.available_gpus:
            return self.config.gpu_servers[idx]
        if self.local_alive:
            return self.config.local_endpoint
        return None

    @property
    def gate_threshold(self) -> float:
        """Auto-approve threshold for current tier."""
        if self.tier >= Tier.PARTIAL:
            return self.config.gate_threshold_full
        return self.config.gate_threshold_degraded

    @property
    def model_for_tier(self) -> str:
        """Which model to use at current tier."""
        if self.tier >= Tier.PARTIAL:
            return self.config.rig_model
        return self.config.local_model
