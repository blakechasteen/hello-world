#!/usr/bin/env python3
"""
Debug Phase 5 Dimension Mismatch
=================================
Traces dimensions through the Phase 5 pipeline to find mismatch.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from HoloLoom.config import Config
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.embedding.linguistic_matryoshka_gate import (
    LinguisticMatryoshkaGate, LinguisticGateConfig, LinguisticFilterMode
)

def debug_dimensions():
    """Debug dimension flow through Phase 5."""

    print("=" * 80)
    print("PHASE 5 DIMENSION DEBUG")
    print("=" * 80)
    print()

    # Test texts
    texts = ["What are mammals?", "Dogs are animals"]

    # ========================================================================
    # 1. Standard MatryoshkaEmbeddings (baseline)
    # ========================================================================

    print("[1] Standard MatryoshkaEmbeddings (baseline)")
    print("-" * 80)

    standard_embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    print(f"Configured sizes: {standard_embedder.sizes}")
    print()

    # Test encode()
    standard_embeds = standard_embedder.encode(texts)
    print(f"encode(texts) shape: {standard_embeds.shape}")
    print(f"  Expected: (2, 384) - largest scale")
    print()

    # Test encode_scales() with size
    standard_embeds_96 = standard_embedder.encode_scales(texts, size=96)
    print(f"encode_scales(texts, size=96) shape: {standard_embeds_96.shape}")
    print(f"  Expected: (2, 96)")
    print()

    # Test encode_scales() without size
    standard_embeds_all = standard_embedder.encode_scales(texts, size=None)
    print(f"encode_scales(texts, size=None):")
    for scale, embeds in standard_embeds_all.items():
        print(f"  scale {scale}: {embeds.shape}")
    print()

    # ========================================================================
    # 2. LinguisticMatryoshkaGate (with compositional cache)
    # ========================================================================

    print("[2] LinguisticMatryoshkaGate (with compositional cache)")
    print("-" * 80)

    # Create config
    config = LinguisticGateConfig(
        scales=[96, 192, 384],
        linguistic_mode=LinguisticFilterMode.DISABLED,
        use_compositional_cache=True
    )

    # Create base embedder
    base_embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

    # Create linguistic gate
    linguistic_gate = LinguisticMatryoshkaGate(
        embedder=base_embedder,
        config=config
    )

    print(f"Configured scales: {config.scales}")
    print(f"Has compositional cache: {linguistic_gate.compositional_cache is not None}")
    print()

    # Test encode()
    try:
        gate_embeds = linguistic_gate.encode(texts)
        print(f"encode(texts) shape: {gate_embeds.shape}")
        print(f"  Expected: (2, 384) - largest scale")
        print()
    except Exception as e:
        print(f"❌ encode() FAILED: {e}")
        print()

    # Test encode_scales() with size
    try:
        gate_embeds_96 = linguistic_gate.encode_scales(texts, size=96)
        print(f"encode_scales(texts, size=96) type: {type(gate_embeds_96)}")
        if isinstance(gate_embeds_96, np.ndarray):
            print(f"  shape: {gate_embeds_96.shape}")
            print(f"  Expected: (2, 96)")
        else:
            print(f"  ❌ WRONG TYPE! Expected np.ndarray, got {type(gate_embeds_96)}")
        print()
    except Exception as e:
        print(f"❌ encode_scales(size=96) FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test encode_scales() without size
    try:
        gate_embeds_all = linguistic_gate.encode_scales(texts, size=None)
        print(f"encode_scales(texts, size=None) type: {type(gate_embeds_all)}")
        if isinstance(gate_embeds_all, dict):
            print(f"  scales returned:")
            for scale, embeds in gate_embeds_all.items():
                print(f"    scale {scale}: {embeds.shape}")
        else:
            print(f"  ❌ WRONG TYPE! Expected dict, got {type(gate_embeds_all)}")
        print()
    except Exception as e:
        print(f"❌ encode_scales(size=None) FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()

    # ========================================================================
    # 3. Compositional Cache Direct Test
    # ========================================================================

    print("[3] Compositional Cache Direct Test")
    print("-" * 80)

    if linguistic_gate.compositional_cache:
        cache = linguistic_gate.compositional_cache

        # Test single text
        text = "What are mammals?"
        emb, trace = cache.get_compositional_embedding(text, return_trace=True)

        print(f"get_compositional_embedding('{text}'):")
        print(f"  embedding shape: {emb.shape}")
        print(f"  Expected: (384,) - base embedder dimension")
        print(f"  Trace: {trace}")
        print()

        # Check if slicing works
        print("Testing dimension slicing:")
        for target_dim in [96, 192, 384]:
            sliced = emb[:target_dim]
            print(f"  emb[:{target_dim}] shape: {sliced.shape}")
        print()
    else:
        print("  No compositional cache available")
        print()

    # ========================================================================
    # 4. Policy Engine Expectations
    # ========================================================================

    print("[4] Policy Engine Expectations")
    print("-" * 80)

    print("Policy engine calls: embedder.encode_scales(texts, size=mem_dim)")
    print(f"  For FAST mode: mem_dim = 192")
    print(f"  Expected return: np.ndarray with shape (n_texts, 192)")
    print()

    print("Testing what policy gets:")
    try:
        policy_embeds = linguistic_gate.encode_scales(texts, size=192)
        print(f"  Type: {type(policy_embeds)}")
        if isinstance(policy_embeds, np.ndarray):
            print(f"  Shape: {policy_embeds.shape}")
            print(f"  ✅ CORRECT! Returns array with shape (2, 192)")
        else:
            print(f"  ❌ WRONG! Returns {type(policy_embeds)} instead of np.ndarray")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    print()

    # ========================================================================
    # Summary
    # ========================================================================

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Checks that should pass:")
    print("  1. encode() returns (n_texts, largest_dim)")
    print("  2. encode_scales(size=X) returns np.ndarray with shape (n_texts, X)")
    print("  3. encode_scales(size=None) returns dict {scale: array}")
    print("  4. Compositional cache embeddings are sliceable")
    print()
    print("❌ If any check fails, that's the source of dimension mismatch!")
    print()

if __name__ == "__main__":
    debug_dimensions()
