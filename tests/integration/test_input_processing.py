"""
Quick Test for Multi-Modal Input Processing

Tests basic functionality without requiring all optional dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.input import TextProcessor, ModalityType


async def test_text_processor_basic():
    """Test text processor without optional dependencies."""
    print("\n=== Test 1: Basic Text Processing ===")
    
    processor = TextProcessor(
        embedder=None,
        use_spacy=False,  # Disable optional dependencies
        use_textblob=False
    )
    
    # Test simple text
    text = "HoloLoom is a revolutionary neural decision-making system built with Python."
    result = await processor.process(text, extract_entities=False, extract_sentiment=False)
    
    print(f"Modality: {result.modality.value}")
    print(f"Content: {result.content[:100]}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Features: {result.features}")
    
    assert result.modality == ModalityType.TEXT
    assert result.content == text
    assert 0.0 <= result.confidence <= 1.0
    
    print("PASSED\n")


async def test_text_processor_with_features():
    """Test text processor with feature extraction."""
    print("=== Test 2: Text Processing with Features ===")
    
    processor = TextProcessor(
        embedder=None,
        use_spacy=False,
        use_textblob=False
    )
    
    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning uses neural networks with multiple layers.
    """
    
    result = await processor.process(
        text,
        extract_entities=False,
        extract_sentiment=False,
        extract_topics=True,
        extract_keyphrases=True
    )
    
    print(f"Content: {result.content[:80]}...")
    print(f"Topics: {result.features['text'].get('topics', [])}")
    print(f"Keyphrases: {result.features['text'].get('keyphrases', [])[:5]}")
    
    assert result.modality == ModalityType.TEXT
    assert len(result.content) > 0
    
    print("PASSED\n")


async def test_text_processor_confidence():
    """Test confidence scoring."""
    print("=== Test 3: Confidence Scoring ===")
    
    processor = TextProcessor(embedder=None, use_spacy=False, use_textblob=False)
    
    # Very short text (low confidence)
    short_result = await processor.process("Hi")
    print(f"Short text confidence: {short_result.confidence:.2f}")
    
    # Normal text (high confidence)
    normal_result = await processor.process(
        "This is a normal length sentence with proper content."
    )
    print(f"Normal text confidence: {normal_result.confidence:.2f}")
    
    # Assert short text has lower confidence
    assert short_result.confidence < normal_result.confidence
    
    print("PASSED\n")


async def test_modality_types():
    """Test modality type enum."""
    print("=== Test 4: Modality Types ===")
    
    modalities = [
        ModalityType.TEXT,
        ModalityType.IMAGE,
        ModalityType.AUDIO,
        ModalityType.VIDEO,
        ModalityType.STRUCTURED,
        ModalityType.MULTIMODAL
    ]
    
    for modality in modalities:
        print(f"  {modality.name}: {modality.value}")
    
    assert len(modalities) == 6
    print("PASSED\n")


async def test_processed_input_dict():
    """Test ProcessedInput serialization."""
    print("=== Test 5: ProcessedInput Serialization ===")
    
    processor = TextProcessor(embedder=None, use_spacy=False, use_textblob=False)
    result = await processor.process("Test text for serialization")
    
    # Convert to dict
    result_dict = result.to_dict()
    
    print(f"Dict keys: {list(result_dict.keys())}")
    assert 'modality' in result_dict
    assert 'content' in result_dict
    assert 'confidence' in result_dict
    assert result_dict['modality'] == 'text'
    
    print("PASSED\n")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MULTI-MODAL INPUT PROCESSING TESTS")
    print("=" * 60)
    
    tests = [
        test_text_processor_basic,
        test_text_processor_with_features,
        test_text_processor_confidence,
        test_modality_types,
        test_processed_input_dict,
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
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
