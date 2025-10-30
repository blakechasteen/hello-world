"""
Comprehensive Tests for Multi-Modal Input Processing

Tests all processors, fusion, and routing.
"""

import asyncio
import sys
import time
from pathlib import Path
import numpy as np

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.input import (
    TextProcessor,
    ImageProcessor,
    AudioProcessor,
    StructuredDataProcessor,
    MultiModalFusion,
    InputRouter,
    ModalityType,
    ProcessedInput
)


async def test_text_processor():
    """Test text processor."""
    print("\n=== Test 1: Text Processor ===")
    
    processor = TextProcessor(embedder=None, use_spacy=False, use_textblob=False)
    
    text = "HoloLoom processes multi-modal inputs including text, images, and audio."
    result = await processor.process(text)
    
    print(f"Modality: {result.modality.value}")
    print(f"Content: {result.content[:80]}...")
    print(f"Confidence: {result.confidence:.2f}")
    
    assert result.modality == ModalityType.TEXT
    assert result.confidence > 0.0
    
    print("PASSED")


async def test_structured_processor():
    """Test structured data processor."""
    print("\n=== Test 2: Structured Data Processor ===")
    
    processor = StructuredDataProcessor()
    
    # Test JSON dict
    data = {
        'name': 'Alice',
        'age': 30,
        'email': 'alice@example.com',
        'active': True
    }
    
    result = await processor.process(data)
    
    print(f"Modality: {result.modality.value}")
    print(f"Content: {result.content}")
    print(f"Schema: {result.features['structured']['schema']}")
    print(f"Rows x Cols: {result.features['structured']['row_count']} x {result.features['structured']['column_count']}")
    
    assert result.modality == ModalityType.STRUCTURED
    assert result.features['structured']['column_count'] == 4
    
    print("PASSED")


async def test_multi_modal_fusion_attention():
    """Test multi-modal fusion with attention strategy."""
    print("\n=== Test 3: Multi-Modal Fusion (Attention) ===")
    
    fusion = MultiModalFusion(target_dim=128)
    
    # Create mock inputs
    text_input = ProcessedInput(
        modality=ModalityType.TEXT,
        content="Text content",
        embedding=np.random.randn(256),
        confidence=0.9
    )
    
    structured_input = ProcessedInput(
        modality=ModalityType.STRUCTURED,
        content="Structured data",
        embedding=np.random.randn(128),
        confidence=0.8
    )
    
    # Fuse
    fused = await fusion.fuse([text_input, structured_input], strategy="attention")
    
    print(f"Fused modality: {fused.modality.value}")
    print(f"Fused content: {fused.content}")
    print(f"Fused embedding shape: {fused.embedding.shape}")
    print(f"Fused confidence: {fused.confidence:.2f}")
    
    assert fused.modality == ModalityType.MULTIMODAL
    assert fused.embedding.shape[0] == 128
    assert 'text' in fused.features
    assert 'structured' in fused.features
    
    print("PASSED")


async def test_multi_modal_fusion_strategies():
    """Test all fusion strategies."""
    print("\n=== Test 4: Multi-Modal Fusion Strategies ===")
    
    fusion = MultiModalFusion(target_dim=64)
    
    # Create inputs
    inputs = [
        ProcessedInput(
            modality=ModalityType.TEXT,
            content="Text",
            embedding=np.ones(64) * 0.5,
            confidence=0.9
        ),
        ProcessedInput(
            modality=ModalityType.STRUCTURED,
            content="Data",
            embedding=np.ones(64) * 0.3,
            confidence=0.8
        )
    ]
    
    strategies = ["attention", "concat", "average", "max"]
    
    for strategy in strategies:
        fused = await fusion.fuse(inputs, strategy=strategy)
        print(f"  {strategy}: shape={fused.embedding.shape}, conf={fused.confidence:.2f}")
        assert fused.modality == ModalityType.MULTIMODAL
    
    print("PASSED")


async def test_input_router_text():
    """Test input router with text."""
    print("\n=== Test 5: Input Router (Text) ===")
    
    router = InputRouter()
    
    # Test text string
    result = await router.process("This is a simple text input.")
    
    print(f"Detected modality: {result.modality.value}")
    print(f"Content: {result.content[:50]}...")
    
    assert result.modality == ModalityType.TEXT
    
    print("PASSED")


async def test_input_router_structured():
    """Test input router with structured data."""
    print("\n=== Test 6: Input Router (Structured) ===")
    
    router = InputRouter()
    
    # Test dict
    data = {'key': 'value', 'number': 42}
    result = await router.process(data)
    
    print(f"Detected modality: {result.modality.value}")
    print(f"Content: {result.content}")
    
    assert result.modality == ModalityType.STRUCTURED
    
    print("PASSED")


async def test_input_router_detection():
    """Test modality detection."""
    print("\n=== Test 7: Input Router Detection ===")
    
    router = InputRouter()
    
    test_cases = [
        ("Simple text", ModalityType.TEXT),
        ({'data': 'value'}, ModalityType.STRUCTURED),
        (Path("test.jpg"), ModalityType.IMAGE),
        (Path("test.wav"), ModalityType.AUDIO),
        (Path("test.json"), ModalityType.STRUCTURED),
    ]
    
    for input_data, expected in test_cases:
        detected = router.detect_modality(input_data)
        status = "PASS" if detected == expected else "FAIL"
        print(f"  {str(input_data)[:30]:30s} -> {detected.value:12s} [{status}]")
        assert detected == expected
    
    print("PASSED")


async def test_input_router_multimodal():
    """Test input router with multiple inputs."""
    print("\n=== Test 8: Input Router (Multi-Modal) ===")
    
    router = InputRouter()
    
    # Multiple inputs
    inputs = [
        "This is text content",
        {'data': 'structured'}
    ]
    
    result = await router.process(inputs, fusion_strategy="average")
    
    print(f"Fused modality: {result.modality.value}")
    print(f"Content: {result.content[:80]}...")
    print(f"Features: {list(result.features.keys())}")
    
    assert result.modality == ModalityType.MULTIMODAL
    assert 'text' in result.features
    assert 'structured' in result.features
    
    print("PASSED")


async def test_embedding_alignment():
    """Test embedding alignment."""
    print("\n=== Test 9: Embedding Alignment ===")
    
    fusion = MultiModalFusion(target_dim=128)
    
    # Create inputs with different dimensions
    inputs = [
        ProcessedInput(
            modality=ModalityType.TEXT,
            content="Text",
            embedding=np.random.randn(256),  # 256d
            confidence=0.9
        ),
        ProcessedInput(
            modality=ModalityType.STRUCTURED,
            content="Data",
            embedding=np.random.randn(64),   # 64d
            confidence=0.8
        )
    ]
    
    # Align
    aligned = fusion.align_embeddings(inputs)
    
    for inp in aligned:
        aligned_emb = inp.aligned_embeddings.get(inp.modality)
        print(f"  {inp.modality.value}: original={inp.embedding.shape[0]}d -> aligned={aligned_emb.shape[0]}d")
        assert aligned_emb.shape[0] == 128
    
    print("PASSED")


async def test_cross_modal_similarity():
    """Test cross-modal similarity computation."""
    print("\n=== Test 10: Cross-Modal Similarity ===")
    
    fusion = MultiModalFusion(target_dim=128)
    
    # Create similar inputs
    input1 = ProcessedInput(
        modality=ModalityType.TEXT,
        content="Similar",
        embedding=np.ones(128),
        confidence=0.9
    )
    
    input2 = ProcessedInput(
        modality=ModalityType.STRUCTURED,
        content="Similar",
        embedding=np.ones(128) * 0.9,
        confidence=0.9
    )
    
    # Align first
    aligned = fusion.align_embeddings([input1, input2])
    
    # Compute similarity
    similarity = fusion.compute_cross_modal_similarity(aligned[0], aligned[1])
    
    print(f"  Similarity: {similarity:.3f}")
    assert 0.0 <= similarity <= 1.0
    assert similarity > 0.5  # Should be high for similar vectors
    
    print("PASSED")


async def test_batch_processing():
    """Test batch processing."""
    print("\n=== Test 11: Batch Processing ===")
    
    router = InputRouter()
    
    inputs = [
        "First text input",
        "Second text input",
        {'key': 'value'}
    ]
    
    start_time = time.time()
    results = await router.process_batch(inputs)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  Processed {len(results)} inputs in {elapsed:.1f}ms")
    print(f"  Avg: {elapsed/len(results):.1f}ms per input")
    
    assert len(results) == 3
    assert results[0].modality == ModalityType.TEXT
    assert results[1].modality == ModalityType.TEXT
    assert results[2].modality == ModalityType.STRUCTURED
    
    print("PASSED")


async def test_available_processors():
    """Test processor availability check."""
    print("\n=== Test 12: Available Processors ===")
    
    router = InputRouter()
    available = router.get_available_processors()
    
    print("  Processor availability:")
    for modality, is_available in available.items():
        status = "available" if is_available else "unavailable"
        print(f"    {modality.value}: {status}")
    
    # Text and structured should always be available
    assert available[ModalityType.TEXT]
    assert available[ModalityType.STRUCTURED]
    
    print("PASSED")


async def run_all_tests():
    """Run all comprehensive tests."""
    print("=" * 70)
    print("COMPREHENSIVE MULTI-MODAL INPUT PROCESSING TESTS")
    print("=" * 70)
    
    start_time = time.time()
    
    tests = [
        test_text_processor,
        test_structured_processor,
        test_multi_modal_fusion_attention,
        test_multi_modal_fusion_strategies,
        test_input_router_text,
        test_input_router_structured,
        test_input_router_detection,
        test_input_router_multimodal,
        test_embedding_alignment,
        test_cross_modal_similarity,
        test_batch_processing,
        test_available_processors,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"\nFAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    elapsed = (time.time() - start_time) * 1000
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print(f"Time: {elapsed:.1f}ms")
    print("=" * 70)
    
    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
