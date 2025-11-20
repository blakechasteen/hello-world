"""
Test Multi-Modal Spinners

Quick validation of multi-modal spinner functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.multimodal_spinner import (
    MultiModalSpinner,
    TextSpinner,
    StructuredDataSpinner,
    CrossModalSpinner
)


async def test_text_spinner():
    """Test TextSpinner with simple text."""
    print("\n=== Testing TextSpinner ===")
    
    spinner = TextSpinner()
    text = "Apple Inc. announced record profits today. The company is optimistic about AI."
    
    shards = await spinner.spin(text)
    
    print(f"Input: {text[:60]}...")
    print(f"Shards created: {len(shards)}")
    
    for i, shard in enumerate(shards):
        print(f"\nShard {i+1}:")
        print(f"  ID: {shard.id}")
        print(f"  Text: {shard.text[:80]}...")
        print(f"  Entities: {shard.entities}")
        print(f"  Motifs: {shard.motifs}")
        print(f"  Modality: {shard.metadata.get('modality_type', 'N/A')}")
        print(f"  Confidence: {shard.metadata.get('confidence', 0):.3f}")
        has_embedding = shard.metadata.get('embedding') is not None
        print(f"  Has embedding: {has_embedding}")
    
    print("[OK] TextSpinner test passed")


async def test_structured_spinner():
    """Test StructuredDataSpinner with JSON data."""
    print("\n=== Testing StructuredDataSpinner ===")
    
    spinner = StructuredDataSpinner()
    data = {
        "company": "Apple Inc.",
        "products": ["iPhone", "MacBook", "iPad"],
        "revenue": 394328000000,
        "employees": 164000
    }
    
    shards = await spinner.spin(data)
    
    print(f"Input: {data}")
    print(f"Shards created: {len(shards)}")
    
    for i, shard in enumerate(shards):
        print(f"\nShard {i+1}:")
        print(f"  ID: {shard.id}")
        print(f"  Text: {shard.text[:80]}...")
        print(f"  Modality: {shard.metadata.get('modality_type', 'N/A')}")
        print(f"  Confidence: {shard.metadata.get('confidence', 0):.3f}")
        has_embedding = shard.metadata.get('embedding') is not None
        print(f"  Has embedding: {has_embedding}")
    
    print("[OK] StructuredDataSpinner test passed")


async def test_multimodal_spinner():
    """Test base MultiModalSpinner with auto-detection."""
    print("\n=== Testing MultiModalSpinner (auto-detect) ===")
    
    spinner = MultiModalSpinner()
    
    # Test with various inputs
    test_inputs = [
        ("Text", "The weather is sunny and warm today."),
        ("Structured", {"temperature": 75, "condition": "sunny"}),
        ("List", [1, 2, 3, 4, 5])
    ]
    
    for label, input_data in test_inputs:
        print(f"\n{label} input:")
        shards = await spinner.spin(input_data)
        print(f"  Shards created: {len(shards)}")
        print(f"  Modality: {shards[0].metadata.get('modality_type', 'N/A')}")
        print(f"  Confidence: {shards[0].metadata.get('confidence', 0):.3f}")
    
    print("\n[OK] MultiModalSpinner test passed")


async def test_cross_modal_spinner():
    """Test CrossModalSpinner with multiple modalities."""
    print("\n=== Testing CrossModalSpinner ===")
    
    spinner = CrossModalSpinner()
    
    # Multiple inputs from different modalities
    inputs = [
        "Apple Inc. is a technology company.",
        {"company": "Apple Inc.", "products": ["iPhone"]},
        "The iPhone revolutionized smartphones."
    ]
    
    print("Processing 3 inputs (2 text + 1 structured)...")
    shards = await spinner.spin_multiple(inputs, fusion_strategy="attention")
    
    print(f"\nTotal shards created: {len(shards)}")
    
    # Find the fused shard
    fused_shards = [s for s in shards if s.metadata.get('is_fused')]
    
    if fused_shards:
        fused = fused_shards[0]
        print(f"\nFused shard found:")
        print(f"  ID: {fused.id}")
        print(f"  Component count: {fused.metadata.get('component_count', 0)}")
        print(f"  Component modalities: {fused.metadata.get('component_modalities', [])}")
        print(f"  Fusion strategy: {fused.metadata.get('fusion_strategy', 'N/A')}")
        print(f"  Confidence: {fused.metadata.get('confidence', 0):.3f}")
        has_embedding = fused.metadata.get('embedding') is not None
        print(f"  Has embedding: {has_embedding}")
    
    print("\n[OK] CrossModalSpinner test passed")


async def test_cross_modal_query():
    """Test cross-modal query processing."""
    print("\n=== Testing Cross-Modal Query ===")
    
    spinner = CrossModalSpinner()
    
    query = "Show me text and images about quantum computing"
    
    print(f"Query: '{query}'")
    shards = await spinner.spin_query(query)
    
    print(f"Query shards created: {len(shards)}")
    
    if shards:
        query_shard = shards[0]
        print(f"\nQuery shard:")
        print(f"  ID: {query_shard.id}")
        print(f"  Text: {query_shard.text}")
        print(f"  Is query: {query_shard.metadata.get('is_query', False)}")
        print(f"  Cross-modal: {query_shard.metadata.get('cross_modal', False)}")
        print(f"  Modality filter: {query_shard.metadata.get('modality_filter', 'N/A')}")
        has_embedding = query_shard.metadata.get('embedding') is not None
        print(f"  Has embedding: {has_embedding}")
    
    print("\n[OK] Cross-modal query test passed")


async def run_all_tests():
    """Run all spinner tests."""
    print("\n" + "="*70)
    print("  MULTI-MODAL SPINNER TESTS")
    print("="*70)
    
    tests = [
        test_text_spinner,
        test_structured_spinner,
        test_multimodal_spinner,
        test_cross_modal_spinner,
        test_cross_modal_query
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test_func.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    else:
        print("         ALL TESTS PASSED!")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
