#!/usr/bin/env python3
"""Quick test to verify encode_scales returns correct type."""

import sys
sys.path.insert(0, '.')

import numpy as np
from HoloLoom.config import Config
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.embedding.linguistic_matryoshka_gate import (
    LinguisticMatryoshkaGate, LinguisticGateConfig, LinguisticFilterMode
)

def test_encode_scales():
    """Test encode_scales return types."""

    texts = ["test query"]

    print("=" * 80)
    print("ENCODE_SCALES RETURN TYPE TEST")
    print("=" * 80)
    print()

    # Test 1: Standard Matryoshka
    print("[1] Standard MatryoshkaEmbeddings")
    standard = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    result_with_size = standard.encode_scales(texts, size=96)
    print(f"  encode_scales(texts, size=96):")
    print(f"    Type: {type(result_with_size)}")
    print(f"    Value: {result_with_size.shape if isinstance(result_with_size, np.ndarray) else 'NOT ARRAY!'}")

    result_no_size = standard.encode_scales(texts, size=None)
    print(f"  encode_scales(texts, size=None):")
    print(f"    Type: {type(result_no_size)}")
    print(f"    Value: {list(result_no_size.keys()) if isinstance(result_no_size, dict) else 'NOT DICT!'}")
    print()

    # Test 2: Linguistic Gate
    print("[2] LinguisticMatryoshkaGate")
    config = LinguisticGateConfig(
        scales=[96, 192, 384],
        linguistic_mode=LinguisticFilterMode.DISABLED,
        use_compositional_cache=True
    )

    base = MatryoshkaEmbeddings(sizes=[96, 192, 384])
    gate = LinguisticMatryoshkaGate(embedder=base, config=config)

    print(f"  Has compositional cache: {gate.compositional_cache is not None}")

    result_with_size = gate.encode_scales(texts, size=96)
    print(f"  encode_scales(texts, size=96):")
    print(f"    Type: {type(result_with_size)}")
    if isinstance(result_with_size, np.ndarray):
        print(f"    Shape: {result_with_size.shape}")
        print(f"    ✅ CORRECT - returns array")
    elif isinstance(result_with_size, dict):
        print(f"    Keys: {list(result_with_size.keys())}")
        print(f"    ❌ WRONG - returns dict instead of array!")
    else:
        print(f"    ❌ WRONG - unexpected type: {type(result_with_size)}")

    result_no_size = gate.encode_scales(texts, size=None)
    print(f"  encode_scales(texts, size=None):")
    print(f"    Type: {type(result_no_size)}")
    if isinstance(result_no_size, dict):
        print(f"    Keys: {list(result_no_size.keys())}")
        print(f"    ✅ CORRECT - returns dict")
    else:
        print(f"    ❌ WRONG - unexpected type")
    print()

    # Test 3: Policy simulation
    print("[3] Simulating Policy Call")
    policy_mem_dim = 96
    try:
        mem_np = gate.encode_scales(texts, size=policy_mem_dim)
        print(f"  gate.encode_scales(texts, size={policy_mem_dim}):")
        print(f"    Type: {type(mem_np)}")
        print(f"    Shape: {mem_np.shape if hasattr(mem_np, 'shape') else 'NO SHAPE'}")

        # Try torch conversion
        import torch
        mem = torch.tensor(mem_np, dtype=torch.float32)
        print(f"    torch.tensor conversion: ✅ SUCCESS")
        print(f"    Tensor shape: {mem.shape}")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        print(f"    This is the error we're seeing in tests!")

    print()
    print("=" * 80)

if __name__ == "__main__":
    test_encode_scales()
