"""
Activation Buffer — collects neural activations + embeddings for SAE training.

Taps multiple points in the hot path:
- NeuralCore pooled_features (384d) — policy decisions
- Matryoshka embeddings at each scale (96d, 192d, 384d) — input representations
- DotPlasma psi vectors — fused feature representations

Stores activations with governance labels (safety score, deception score, RBAC decision)
for supervised feature discovery.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActivationSample:
    """Single activation sample with metadata."""
    timestamp: float
    source: str  # "neural_core", "embedding_96", "embedding_192", "embedding_384", "dot_plasma"
    activation: np.ndarray  # the raw activation vector
    query_text: str = ""
    # Governance labels (filled by governance checkpoints)
    safety_score: float | None = None
    deception_score: float | None = None
    rbac_decision: str | None = None  # "ALLOW", "DENY", "ESCALATE"
    tool_selected: str | None = None  # for neural_core: which tool was chosen
    metadata: dict[str, Any] = field(default_factory=dict)


class ActivationBuffer:
    """Ring buffer that collects activations and flushes to disk."""

    def __init__(
        self,
        buffer_size: int = 4096,
        flush_dir: str = "./data/sae_activations",
        flush_threshold: int = 1024,  # flush when this many samples accumulated
        flush_interval_s: float = 300.0,  # also flush every 5 min
    ):
        self.buffer: deque[ActivationSample] = deque(maxlen=buffer_size)
        self.flush_dir = Path(flush_dir)
        self.flush_threshold = flush_threshold
        self.flush_interval_s = flush_interval_s
        self._lock = asyncio.Lock()
        self._running = False
        self._samples_flushed = 0
        self._flush_count = 0

        # Per-source buffers for scale-specific SAE training
        self._source_counts: dict[str, int] = {}

    async def record(self, sample: ActivationSample) -> None:
        """Record an activation sample. Non-blocking."""
        async with self._lock:
            self.buffer.append(sample)
            self._source_counts[sample.source] = self._source_counts.get(sample.source, 0) + 1

            if len(self.buffer) >= self.flush_threshold:
                await self._flush()

    def record_sync(self, sample: ActivationSample) -> None:
        """Synchronous recording for non-async contexts. Fire-and-forget."""
        self.buffer.append(sample)
        self._source_counts[sample.source] = self._source_counts.get(sample.source, 0) + 1

    async def _flush(self) -> None:
        """Flush buffer to disk as .npz files grouped by source."""
        if not self.buffer:
            return

        self.flush_dir.mkdir(parents=True, exist_ok=True)

        # Group by source
        by_source: dict[str, list[ActivationSample]] = {}
        while self.buffer:
            sample = self.buffer.popleft()
            by_source.setdefault(sample.source, []).append(sample)

        for source, samples in by_source.items():
            source_dir = self.flush_dir / source
            source_dir.mkdir(parents=True, exist_ok=True)

            batch_id = f"{int(time.time())}_{self._flush_count}"

            # Stack activations into matrix
            activations = np.stack([s.activation for s in samples])

            # Save activations as .npz
            np.savez_compressed(
                source_dir / f"batch_{batch_id}.npz",
                activations=activations,
            )

            # Save metadata as JSON sidecar
            metadata = []
            for s in samples:
                metadata.append({
                    "timestamp": s.timestamp,
                    "query_text": s.query_text[:200],  # truncate for storage
                    "safety_score": s.safety_score,
                    "deception_score": s.deception_score,
                    "rbac_decision": s.rbac_decision,
                    "tool_selected": s.tool_selected,
                    "metadata": s.metadata,
                })

            with open(source_dir / f"batch_{batch_id}_meta.json", "w") as f:
                json.dump(metadata, f)

            self._samples_flushed += len(samples)
            logger.info(
                f"Flushed {len(samples)} {source} activations "
                f"(shape={activations.shape}, total={self._samples_flushed})"
            )

        self._flush_count += 1

    async def start_background_flush(self) -> None:
        """Periodic background flush loop."""
        self._running = True
        while self._running:
            await asyncio.sleep(self.flush_interval_s)
            async with self._lock:
                await self._flush()

    def stop(self) -> None:
        self._running = False

    def stats(self) -> dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "samples_flushed": self._samples_flushed,
            "flush_count": self._flush_count,
            "source_counts": dict(self._source_counts),
        }


# ---------------------------------------------------------------------------
# Singleton access — supports both module-global and server-state injection.
#
# Tap points call get_activation_buffer().  By default it returns the
# module-level _global_buffer.  When the server starts it can call
# set_activation_buffer(buf) to inject the instance it owns, so the
# same object is reachable from both server_state and tap-point code.
# ---------------------------------------------------------------------------

_global_buffer: ActivationBuffer | None = None


def get_activation_buffer() -> ActivationBuffer | None:
    """Return the shared ActivationBuffer (or None if not yet initialized)."""
    return _global_buffer


def set_activation_buffer(buf: ActivationBuffer) -> None:
    """Inject a buffer created elsewhere (e.g., server startup)."""
    global _global_buffer
    _global_buffer = buf


def init_activation_buffer(**kwargs) -> ActivationBuffer:
    """Create a new buffer and install it as the global singleton."""
    global _global_buffer
    _global_buffer = ActivationBuffer(**kwargs)
    return _global_buffer
