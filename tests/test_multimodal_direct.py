"""
Direct imports test for multi-modal processing.
Bypasses HoloLoom.__init__.py cascade.
"""

import sys
import os
import asyncio
import numpy as np

# Add paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Direct file imports without package init
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load protocol first
protocol_path = os.path.join(project_root, "HoloLoom", "input", "protocol.py")
protocol = load_module("protocol", protocol_path)

# Load processors
text_path = os.path.join(project_root, "HoloLoom", "input", "text_processor.py")
text_module = load_module("text_processor", text_path)

structured_path = os.path.join(project_root, "HoloLoom", "input", "structured_processor.py")
structured_module = load_module("structured_processor", structured_path)

fusion_path = os.path.join(project_root, "HoloLoom", "input", "fusion.py")
fusion_module = load_module("fusion", fusion_path)

router_path = os.path.join(project_root, "HoloLoom", "input", "router.py")
router_module = load_module("router", router_path)


def test_text_processor():
    """Test basic text processing."""
    print("Testing TextProcessor...")
    processor = text_module.TextProcessor()
    
    result = asyncio.run(processor.process(
        "Apple Inc. announced record profits today. The company's CEO was excited about the future."
    ))
    
    assert result.modality == protocol.ModalityType.TEXT
    assert result.embedding is not None
    assert len(result.embedding) > 0
    assert result.confidence > 0
    print(f"  ✓ Modality: {result.modality.name}")
    print(f"  ✓ Embedding dim: {len(result.embedding)}")
    print(f"  ✓ Confidence: {result.confidence:.3f}")
    print(f"  ✓ Entities: {len(result.features.entities)}")
    print("✓ TextProcessor PASSED")


def test_structured_processor():
    """Test structured data processing."""
    print("\nTesting StructuredDataProcessor...")
    processor = structured_module.StructuredDataProcessor()
    
    data = {
        "users": [
            {"user_id": 1, "name": "Alice", "age": 30, "score": 95.5},
            {"user_id": 2, "name": "Bob", "age": 25, "score": 87.2},
            {"user_id": 3, "name": "Charlie", "age": 35, "score": 92.8}
        ]
    }
    
    result = asyncio.run(processor.process(data))
    
    assert result.modality == protocol.ModalityType.STRUCTURED
    assert result.embedding is not None
    assert len(result.embedding) > 0
    assert result.confidence > 0
    print(f"  ✓ Modality: {result.modality.name}")
    print(f"  ✓ Embedding dim: {len(result.embedding)}")
    print(f"  ✓ Confidence: {result.confidence:.3f}")
    print(f"  ✓ Schema fields: {len(result.features.schema)}")
    print("✓ StructuredDataProcessor PASSED")


def test_fusion():
    """Test multi-modal fusion."""
    print("\nTesting MultiModalFusion...")
    fusion = fusion_module.MultiModalFusion()
    
    # Create mock inputs
    input1 = protocol.ProcessedInput(
        modality=protocol.ModalityType.TEXT,
        raw_content="test text",
        embedding=np.random.randn(512).astype(np.float32),
        confidence=0.9,
        features=protocol.TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input2 = protocol.ProcessedInput(
        modality=protocol.ModalityType.STRUCTURED,
        raw_content={"key": "value"},
        embedding=np.random.randn(512).astype(np.float32),
        confidence=0.8,
        features=protocol.StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    # Test all strategies
    for strategy in ["attention", "concat", "average", "max"]:
        result = asyncio.run(fusion.fuse([input1, input2], strategy=strategy))
        assert result.modality == protocol.ModalityType.MULTIMODAL
        assert result.embedding is not None
        assert result.confidence > 0
        print(f"  ✓ {strategy} strategy: dim={len(result.embedding)}, conf={result.confidence:.3f}")
    
    print("✓ MultiModalFusion PASSED")


def test_router():
    """Test input router."""
    print("\nTesting InputRouter...")
    router = router_module.InputRouter()
    
    # Test text
    result1 = asyncio.run(router.process("This is a test message."))
    assert result1.modality == protocol.ModalityType.TEXT
    print(f"  ✓ Text routing: {result1.modality.name}")
    
    # Test structured
    result2 = asyncio.run(router.process({"key": "value", "count": 123}))
    assert result2.modality == protocol.ModalityType.STRUCTURED
    print(f"  ✓ Structured routing: {result2.modality.name}")
    
    # Test multimodal
    result3 = asyncio.run(router.process(["text", {"data": 1}]))
    assert result3.modality == protocol.ModalityType.MULTIMODAL
    print(f"  ✓ Multimodal routing: {result3.modality.name}")
    
    # Test batch
    batch_results = asyncio.run(router.process_batch([
        "text1", "text2", {"key": "val"}
    ]))
    assert len(batch_results) == 3
    print(f"  ✓ Batch processing: {len(batch_results)} results")
    
    print("✓ InputRouter PASSED")


def test_embedding_alignment():
    """Test embedding alignment."""
    print("\nTesting embedding alignment...")
    fusion = fusion_module.MultiModalFusion()
    
    embeddings = [
        np.random.randn(256).astype(np.float32),
        np.random.randn(384).astype(np.float32),
        np.random.randn(512).astype(np.float32)
    ]
    
    aligned = fusion.align_embeddings(embeddings, target_dim=512)
    
    assert len(aligned) == 3
    for i, emb in enumerate(aligned):
        assert len(emb) == 512
        print(f"  ✓ Aligned {len(embeddings[i])}d → {len(emb)}d")
    
    print("✓ Embedding alignment PASSED")


def test_cross_modal_similarity():
    """Test cross-modal similarity."""
    print("\nTesting cross-modal similarity...")
    fusion = fusion_module.MultiModalFusion()
    
    # Orthogonal vectors (similarity ≈ 0)
    input1 = protocol.ProcessedInput(
        modality=protocol.ModalityType.TEXT,
        raw_content="test",
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
        features=protocol.TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input2 = protocol.ProcessedInput(
        modality=protocol.ModalityType.STRUCTURED,
        raw_content={},
        embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        confidence=0.8,
        features=protocol.StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    similarity = fusion.compute_cross_modal_similarity(input1, input2)
    assert 0.0 <= similarity <= 1.0
    print(f"  ✓ Orthogonal similarity: {similarity:.4f} (expected ≈0)")
    
    # Identical vectors (similarity ≈ 1)
    input3 = protocol.ProcessedInput(
        modality=protocol.ModalityType.TEXT,
        raw_content="test",
        embedding=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        confidence=0.9,
        features=protocol.TextFeatures(entities=[], sentiment=None, topics=[], keyphrases=[]),
        metadata={}
    )
    
    input4 = protocol.ProcessedInput(
        modality=protocol.ModalityType.STRUCTURED,
        raw_content={},
        embedding=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        confidence=0.8,
        features=protocol.StructuredFeatures(schema={}, summary_stats={}, relationships=[]),
        metadata={}
    )
    
    similarity2 = fusion.compute_cross_modal_similarity(input3, input4)
    print(f"  ✓ Identical similarity: {similarity2:.4f} (expected ≈1)")
    
    print("✓ Cross-modal similarity PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE MULTI-MODAL INPUT PROCESSING TESTS")
    print("(Direct module imports - no transformers cascade)")
    print("="*70 + "\n")
    
    tests = [
        ("Text Processor", test_text_processor),
        ("Structured Processor", test_structured_processor),
        ("Multi-Modal Fusion", test_fusion),
        ("Input Router", test_router),
        ("Embedding Alignment", test_embedding_alignment),
        ("Cross-Modal Similarity", test_cross_modal_similarity),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED:")
            print(f"  {type(e).__name__}: {e}")
            errors.append((name, e))
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed:")
        for name, error in errors:
            print(f"         - {name}: {type(error).__name__}")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
