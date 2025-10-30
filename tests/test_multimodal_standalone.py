"""
Standalone comprehensive tests for multi-modal input processing.
Run directly without importing full HoloLoom stack.
"""

import sys
import os
import asyncio
import numpy as np
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from input module
from HoloLoom.input.protocol import (
    ModalityType, ProcessedInput, TextFeatures, ImageFeatures, 
    AudioFeatures, StructuredFeatures
)
from HoloLoom.input.text_processor import TextProcessor
from HoloLoom.input.structured_processor import StructuredDataProcessor
from HoloLoom.input.fusion import MultiModalFusion
from HoloLoom.input.router import InputRouter


def test_text_processor():
    """Test basic text processing."""
    print("Testing TextProcessor...")
    processor = TextProcessor()
    
    result = asyncio.run(processor.process(
        "Apple Inc. announced record profits today. The company's CEO was excited about the future."
    ))
    
    assert result.modality == ModalityType.TEXT
    assert result.embedding is not None
    assert len(result.embedding) > 0
    assert result.confidence > 0
    assert isinstance(result.features, TextFeatures)
    assert len(result.features.entities) > 0  # Should find "Apple Inc."
    assert result.features.sentiment is not None
    print("✓ TextProcessor tests passed")


def test_structured_processor():
    """Test structured data processing."""
    print("Testing StructuredDataProcessor...")
    processor = StructuredDataProcessor()
    
    data = {
        "users": [
            {"user_id": 1, "name": "Alice", "age": 30, "score": 95.5},
            {"user_id": 2, "name": "Bob", "age": 25, "score": 87.2},
            {"user_id": 3, "name": "Charlie", "age": 35, "score": 92.8}
        ]
    }
    
    result = asyncio.run(processor.process(data))
    
    assert result.modality == ModalityType.STRUCTURED
    assert result.embedding is not None
    assert len(result.embedding) > 0
    assert result.confidence > 0
    assert isinstance(result.features, StructuredFeatures)
    assert "users" in result.features.schema
    assert len(result.features.summary_stats) > 0
    print("✓ StructuredDataProcessor tests passed")


def test_fusion_attention():
    """Test attention-based fusion."""
    print("Testing MultiModalFusion (attention)...")
    fusion = MultiModalFusion()
    
    # Create mock inputs
    input1 = ProcessedInput(
        modality=ModalityType.TEXT,
        raw_content="test text",
        embedding=np.random.randn(512).astype(np.float32),
        confidence=0.9,
        features=TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input2 = ProcessedInput(
        modality=ModalityType.STRUCTURED,
        raw_content={"key": "value"},
        embedding=np.random.randn(512).astype(np.float32),
        confidence=0.8,
        features=StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    result = asyncio.run(fusion.fuse([input1, input2], strategy="attention"))
    
    assert result.modality == ModalityType.MULTIMODAL
    assert result.embedding is not None
    assert len(result.embedding) == 512
    assert result.confidence > 0
    print("✓ Attention fusion tests passed")


def test_fusion_strategies():
    """Test all fusion strategies."""
    print("Testing all fusion strategies...")
    fusion = MultiModalFusion()
    
    # Create mock inputs with different dimensions
    input1 = ProcessedInput(
        modality=ModalityType.TEXT,
        raw_content="test",
        embedding=np.random.randn(384).astype(np.float32),
        confidence=0.9,
        features=TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input2 = ProcessedInput(
        modality=ModalityType.STRUCTURED,
        raw_content={},
        embedding=np.random.randn(256).astype(np.float32),
        confidence=0.8,
        features=StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    inputs = [input1, input2]
    
    for strategy in ["attention", "concat", "average", "max"]:
        result = asyncio.run(fusion.fuse(inputs, strategy=strategy, target_dim=512))
        assert result.embedding is not None
        assert len(result.embedding) == 512
        print(f"  ✓ {strategy} strategy passed")
    
    print("✓ All fusion strategies tests passed")


def test_router_text():
    """Test router with text input."""
    print("Testing InputRouter (text)...")
    router = InputRouter()
    
    result = asyncio.run(router.process("This is a simple text message."))
    
    assert result.modality == ModalityType.TEXT
    assert result.embedding is not None
    assert result.confidence > 0
    print("✓ Router text tests passed")


def test_router_structured():
    """Test router with structured input."""
    print("Testing InputRouter (structured)...")
    router = InputRouter()
    
    data = {"name": "test", "value": 123, "items": [1, 2, 3]}
    result = asyncio.run(router.process(data))
    
    assert result.modality == ModalityType.STRUCTURED
    assert result.embedding is not None
    assert result.confidence > 0
    print("✓ Router structured tests passed")


def test_router_detection():
    """Test modality detection."""
    print("Testing modality detection...")
    router = InputRouter()
    
    # Text detection
    assert router.detect_modality("Hello world") == ModalityType.TEXT
    assert router.detect_modality({"key": "value"}) == ModalityType.STRUCTURED
    assert router.detect_modality([1, 2, 3]) == ModalityType.STRUCTURED
    
    print("✓ Modality detection tests passed")


def test_router_multimodal():
    """Test router with multiple inputs."""
    print("Testing InputRouter (multimodal)...")
    router = InputRouter()
    
    inputs = [
        "Text content",
        {"structured": "data", "count": 5}
    ]
    
    result = asyncio.run(router.process(inputs))
    
    assert result.modality == ModalityType.MULTIMODAL
    assert result.embedding is not None
    assert result.confidence > 0
    print("✓ Router multimodal tests passed")


def test_embedding_alignment():
    """Test embedding dimension alignment."""
    print("Testing embedding alignment...")
    fusion = MultiModalFusion()
    
    embeddings = [
        np.random.randn(256).astype(np.float32),
        np.random.randn(384).astype(np.float32),
        np.random.randn(512).astype(np.float32)
    ]
    
    aligned = fusion.align_embeddings(embeddings, target_dim=512)
    
    assert len(aligned) == 3
    for emb in aligned:
        assert len(emb) == 512
    
    print("✓ Embedding alignment tests passed")


def test_cross_modal_similarity():
    """Test cross-modal similarity computation."""
    print("Testing cross-modal similarity...")
    fusion = MultiModalFusion()
    
    input1 = ProcessedInput(
        modality=ModalityType.TEXT,
        raw_content="test",
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        features=TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input2 = ProcessedInput(
        modality=ModalityType.STRUCTURED,
        raw_content={},
        embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        confidence=0.8,
        features=StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    similarity = fusion.compute_cross_modal_similarity(input1, input2)
    
    assert 0.0 <= similarity <= 1.0
    print(f"  Similarity: {similarity:.4f}")
    print("✓ Cross-modal similarity tests passed")


def test_batch_processing():
    """Test batch processing."""
    print("Testing batch processing...")
    router = InputRouter()
    
    batch = [
        "First text",
        "Second text",
        {"key": "value"},
        [1, 2, 3]
    ]
    
    results = asyncio.run(router.process_batch(batch))
    
    assert len(results) == 4
    assert results[0].modality == ModalityType.TEXT
    assert results[1].modality == ModalityType.TEXT
    assert results[2].modality == ModalityType.STRUCTURED
    assert results[3].modality == ModalityType.STRUCTURED
    print("✓ Batch processing tests passed")


def test_available_processors():
    """Test processor availability check."""
    print("Testing available processors...")
    router = InputRouter()
    
    available = router.get_available_processors()
    
    assert ModalityType.TEXT in available
    assert ModalityType.STRUCTURED in available
    # IMAGE and AUDIO may not be available without dependencies
    print(f"  Available processors: {[m.name for m in available]}")
    print("✓ Available processors tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MULTI-MODAL INPUT PROCESSING TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Text Processor", test_text_processor),
        ("Structured Processor", test_structured_processor),
        ("Fusion Attention", test_fusion_attention),
        ("Fusion Strategies", test_fusion_strategies),
        ("Router Text", test_router_text),
        ("Router Structured", test_router_structured),
        ("Router Detection", test_router_detection),
        ("Router Multimodal", test_router_multimodal),
        ("Embedding Alignment", test_embedding_alignment),
        ("Cross-Modal Similarity", test_cross_modal_similarity),
        ("Batch Processing", test_batch_processing),
        ("Available Processors", test_available_processors),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
