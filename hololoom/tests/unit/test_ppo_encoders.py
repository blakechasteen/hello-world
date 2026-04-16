"""Unit tests for the PPO observation/action encoders.

These cover the wiring from reflection-buffer experience dicts to the
fixed-size tensors that the PPO trainer consumes. They don't require torch
to be installed — the encoders operate on numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

from hololoom.reflection.encoders import (
    DEFAULT_TOOLS,
    EMBEDDING_SCALES,
    OBSERVATION_DIM,
    RETRIEVAL_MODES,
    encode_action,
    encode_observation,
)


class TestEncodeObservation:
    def test_returns_fixed_dim_vector(self):
        observation = {
            "motifs_detected": [],
            "embedding_scales_used": [],
            "context_shards_count": 0,
            "retrieval_mode": "unknown",
            "threads_activated_count": 0,
        }
        vec = encode_observation(observation)
        assert vec.shape == (OBSERVATION_DIM,)
        assert vec.dtype == np.float32

    def test_known_retrieval_mode_sets_onehot(self):
        vec = encode_observation({"retrieval_mode": "fast"})
        mode_slice = vec[: len(RETRIEVAL_MODES)]
        assert mode_slice[RETRIEVAL_MODES.index("fast")] == pytest.approx(1.0)
        assert mode_slice.sum() == pytest.approx(1.0)

    def test_unknown_retrieval_mode_falls_back(self):
        vec = encode_observation({"retrieval_mode": "some-new-mode"})
        mode_slice = vec[: len(RETRIEVAL_MODES)]
        assert mode_slice[RETRIEVAL_MODES.index("unknown")] == pytest.approx(1.0)

    def test_embedding_scales_multihot(self):
        vec = encode_observation({"embedding_scales_used": [96, 384]})
        start = len(RETRIEVAL_MODES)
        scale_slice = vec[start : start + len(EMBEDDING_SCALES)]
        assert scale_slice[EMBEDDING_SCALES.index(96)] == pytest.approx(1.0)
        assert scale_slice[EMBEDDING_SCALES.index(384)] == pytest.approx(1.0)
        assert scale_slice[EMBEDDING_SCALES.index(192)] == pytest.approx(0.0)

    def test_scalar_normalization(self):
        info = {
            "confidence": 0.8,
            "quality_score": 0.5,
            "duration_ms": 4000.0,
            "num_errors": 2,
            "num_warnings": 1,
        }
        observation = {
            "context_shards_count": 40,
            "motifs_detected": ["a", "b", "c"],
            "threads_activated_count": 8,
            "retrieval_mode": "fused",
        }
        vec = encode_observation(observation, info)
        scalars = vec[-8:]
        # shard_count 40 clamps to 1.0
        assert scalars[0] == pytest.approx(1.0)
        # motif_count 3 / 8
        assert scalars[1] == pytest.approx(3 / 8)
        # thread_count 8 / 16
        assert scalars[2] == pytest.approx(0.5)
        # confidence passed through
        assert scalars[3] == pytest.approx(0.8)
        # quality_score passed through
        assert scalars[4] == pytest.approx(0.5)
        # duration 4000ms clamps to 1.0
        assert scalars[5] == pytest.approx(1.0)

    def test_handles_missing_keys_gracefully(self):
        vec = encode_observation({})
        assert vec.shape == (OBSERVATION_DIM,)
        assert not np.isnan(vec).any()


class TestEncodeAction:
    def test_known_tool_maps_to_index(self):
        for idx, tool in enumerate(DEFAULT_TOOLS):
            assert encode_action(tool) == idx

    def test_unknown_tool_falls_back_to_zero(self):
        assert encode_action("unknown_tool") == 0

    def test_none_falls_back_to_zero(self):
        assert encode_action(None) == 0

    def test_custom_tool_list(self):
        tools = ("foo", "bar", "baz")
        assert encode_action("bar", tools=tools) == 1
        assert encode_action("quux", tools=tools) == 0
