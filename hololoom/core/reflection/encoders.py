"""Observation and action encoders for reflection-buffer → PPO wiring.

Kept torch-free so the encoders (and their tests) run in lightweight
environments without pulling in the PyTorch dependency that PPOTrainer needs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Canonical tool set — mirrors NeuralCore.tools.
DEFAULT_TOOLS: tuple[str, ...] = ("answer", "search", "notion_write", "calc")

# Retrieval modes seen in pattern cards (BARE/FAST/FUSED + fallbacks).
RETRIEVAL_MODES: tuple[str, ...] = ("bare", "fast", "fused", "hyperspace", "unknown")

# Embedding scales supported by the Matryoshka gate.
EMBEDDING_SCALES: tuple[int, ...] = (96, 192, 384, 768)

# Fixed observation dimensionality produced by encode_observation().
OBSERVATION_DIM: int = (
    len(RETRIEVAL_MODES)  # one-hot retrieval mode
    + len(EMBEDDING_SCALES)  # multi-hot scales used
    + 8  # scalar features
)


def encode_observation(
    observation: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert a reflection-buffer observation dict into a fixed-size float vector.

    Uses the keys produced by ``RewardExtractor.extract_experience`` so the
    features line up with what the weaving cycle actually records.
    """
    info = info or {}

    retrieval_mode = str(observation.get("retrieval_mode", "unknown")).lower()
    mode_onehot = np.zeros(len(RETRIEVAL_MODES), dtype=np.float32)
    if retrieval_mode in RETRIEVAL_MODES:
        mode_onehot[RETRIEVAL_MODES.index(retrieval_mode)] = 1.0
    else:
        mode_onehot[RETRIEVAL_MODES.index("unknown")] = 1.0

    scales_used = observation.get("embedding_scales_used") or []
    scales_multihot = np.zeros(len(EMBEDDING_SCALES), dtype=np.float32)
    for scale in scales_used:
        try:
            scales_multihot[EMBEDDING_SCALES.index(int(scale))] = 1.0
        except (ValueError, TypeError):
            continue

    motif_count = len(observation.get("motifs_detected") or [])
    thread_count = int(observation.get("threads_activated_count") or 0)
    shard_count = int(observation.get("context_shards_count") or 0)

    confidence = float(info.get("confidence", 0.0))
    quality = float(info.get("quality_score", 0.0))
    duration_ms = float(info.get("duration_ms", 0.0))
    n_errors = float(info.get("num_errors", 0))
    n_warnings = float(info.get("num_warnings", 0))

    # Normalize counts/durations into roughly [0, 1]; PPO normalizes advantages
    # downstream so exact scaling doesn't need to be perfect.
    scalars = np.array(
        [
            min(shard_count / 20.0, 1.0),
            min(motif_count / 8.0, 1.0),
            min(thread_count / 16.0, 1.0),
            float(np.clip(confidence, 0.0, 1.0)),
            float(np.clip(quality, 0.0, 1.0)),
            min(duration_ms / 2000.0, 1.0),
            min(n_errors / 5.0, 1.0),
            min(n_warnings / 5.0, 1.0),
        ],
        dtype=np.float32,
    )

    return np.concatenate([mode_onehot, scales_multihot, scalars])


def encode_action(action: str | None, tools: tuple[str, ...] = DEFAULT_TOOLS) -> int:
    """Map a tool name to its index in the canonical tool list.

    Unknown tools fall back to index 0 ("answer") — same convention the
    convergence engine uses when no explicit winner is available.
    """
    if not action:
        return 0
    try:
        return tools.index(action)
    except ValueError:
        return 0
