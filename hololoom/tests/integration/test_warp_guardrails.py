from __future__ import annotations

import asyncio
from typing import List, Optional

import numpy as np
import pytest

from hololoom.alignment.safety_guardrails import SafetyGuardrails
from hololoom.warp.space import WarpSpace
from hololoom.warp.optimized import GPUWarpSpace, HAS_TORCH


class _StubEmbedder:
    def __init__(self, scales: List[int]):
        self.scales = scales

    def encode_scales(self, texts: List[str], size: Optional[int] = None):
        import numpy as np

        def _matrix(scale: int) -> np.ndarray:
            base = np.arange(len(texts) * scale, dtype=np.float32).reshape(len(texts), scale)
            return base + scale

        if size is not None:
            return {size: _matrix(size)}

        return {scale: _matrix(scale) for scale in self.scales}


def run_sync(awaitable):
    """Helper to execute async warp operations in sync tests."""
    return asyncio.run(awaitable)


def _encode_query(embedder, text: str, scale: int) -> np.ndarray:
    encoded = embedder.encode_scales([text], size=scale)
    if isinstance(encoded, dict):
        return encoded[scale][0]
    return encoded[0]


def _thread_texts() -> List[str]:
    return [
        "Warp Space guardrail integration thread alpha.",
        "Warp Space guardrail integration thread beta.",
    ]


def test_warp_space_emits_guardrail_metadata():
    guardrails = SafetyGuardrails()
    embedder = _StubEmbedder(scales=[96, 192])
    warp = WarpSpace(embedder, scales=[96, 192], guardrails=guardrails)

    threads = _thread_texts()
    run_sync(warp.tension(threads))
    spectral = warp.compute_spectral_features()
    assert spectral["field_norm"] > 0

    query_vec = _encode_query(embedder, "safety query", scale=192)
    attention = warp.apply_attention(query_vec)
    context_vec = warp.weighted_context(attention)
    assert context_vec.shape[0] == warp.tensor_field.shape[1]

    updates = warp.collapse()
    guardrail_meta = updates.get("guardrails", {})

    expected_stages = {"tension", "spectral", "attention", "weighted_context", "collapse"}
    assert expected_stages.issubset(guardrail_meta.keys())

    for stage in expected_stages:
        decision = guardrail_meta[stage]
        assert decision is not None
        assert decision.get("allowed") is True
        assert decision.get("risk_level")

    recorded_actions = {request.action for request in guardrails.action_history}
    assert "warp_tension" in recorded_actions
    assert "warp_attention" in recorded_actions
    assert "warp_weighted_context" in recorded_actions
    assert "warp_collapse" in recorded_actions


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available for GPU warp guardrail test")
def test_gpu_warp_space_guardrail_trace():
    guardrails = SafetyGuardrails()
    embedder = _StubEmbedder(scales=[96])
    warp = GPUWarpSpace(embedder, scales=[96], guardrails=guardrails, use_gpu=True)

    threads = _thread_texts()
    run_sync(warp.tension(threads))

    query_vec = _encode_query(embedder, "gpu safety query", scale=96)
    attention = warp.compute_attention(query_vec)
    context_vec = warp.weighted_context(attention)
    assert context_vec.shape[0] == warp.tensor_field.shape[1]

    trace = warp.guardrail_trace()
    assert trace["tension"] and trace["tension"]["allowed"] is True
    assert trace["attention"] and trace["attention"]["allowed"] is True
    assert trace["weighted_context"] and trace["weighted_context"]["allowed"] is True

    recorded_actions = {request.action for request in guardrails.action_history}
    assert "gpu_warp_tension" in recorded_actions
    assert "gpu_warp_attention" in recorded_actions
    assert "gpu_warp_weighted_context" in recorded_actions
