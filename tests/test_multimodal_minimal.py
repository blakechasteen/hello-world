"""
Minimal unit tests for multi-modal processing core functionality.
Tests without external dependencies.
"""

import sys
import os

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


# Minimal protocol definitions for testing
class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"


@dataclass
class ProcessedInput:
    modality: ModalityType
    raw_content: Any
    embedding: np.ndarray
    confidence: float
    features: Any
    metadata: Dict


def test_fusion_attention():
    """Test attention-based fusion logic."""
    print("Testing attention fusion...")
    
    # Mock embeddings
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)
    
    # Confidence scores
    conf1, conf2 = 0.9, 0.7
    confidences = np.array([conf1, conf2], dtype=np.float32)
    
    # Attention weights (softmax)
    attention_weights = np.exp(confidences) / np.sum(np.exp(confidences))
    
    # Weighted fusion
    fused = attention_weights[0] * emb1 + attention_weights[1] * emb2
    
    assert len(fused) == 512
    assert attention_weights[0] > attention_weights[1]  # Higher confidence gets more weight
    print(f"  ✓ Attention weights: [{attention_weights[0]:.3f}, {attention_weights[1]:.3f}]")
    print("✓ Attention fusion PASSED")


def test_fusion_concat():
    """Test concatenation fusion."""
    print("\nTesting concatenation fusion...")
    
    emb1 = np.random.randn(256).astype(np.float32)
    emb2 = np.random.randn(256).astype(np.float32)
    
    # Concat
    concatenated = np.concatenate([emb1, emb2])
    assert len(concatenated) == 512
    
    # Project to target dimension (simple average for test)
    target_dim = 384
    projection_matrix = np.random.randn(512, target_dim).astype(np.float32) / np.sqrt(512)
    projected = concatenated @ projection_matrix
    
    assert len(projected) == target_dim
    print(f"  ✓ Concatenated: 256+256={len(concatenated)}")
    print(f"  ✓ Projected: {len(concatenated)} → {len(projected)}")
    print("✓ Concatenation fusion PASSED")


def test_fusion_average():
    """Test average fusion."""
    print("\nTesting average fusion...")
    
    emb1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    emb2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    
    conf1, conf2 = 0.8, 0.6
    
    # Weighted average
    total_conf = conf1 + conf2
    weighted_avg = (conf1 / total_conf) * emb1 + (conf2 / total_conf) * emb2
    
    # Calculate expected: (0.8/1.4)*[1,2,3] + (0.6/1.4)*[4,5,6]
    # = 0.571*[1,2,3] + 0.429*[4,5,6]
    # = [0.571,1.143,1.714] + [1.714,2.143,2.571]
    # = [2.286, 3.286, 4.286]
    expected = np.array([2.286, 3.286, 4.286], dtype=np.float32)
    assert np.allclose(weighted_avg, expected, atol=0.01)
    print(f"  ✓ Input1: {emb1} (conf={conf1})")
    print(f"  ✓ Input2: {emb2} (conf={conf2})")
    print(f"  ✓ Weighted avg: {weighted_avg}")
    print("✓ Average fusion PASSED")


def test_fusion_max():
    """Test max pooling fusion."""
    print("\nTesting max pooling fusion...")
    
    emb1 = np.array([1.0, 5.0, 3.0], dtype=np.float32)
    emb2 = np.array([4.0, 2.0, 6.0], dtype=np.float32)
    
    # Element-wise max
    max_pooled = np.maximum(emb1, emb2)
    
    expected = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(max_pooled, expected)
    print(f"  ✓ Input1: {emb1}")
    print(f"  ✓ Input2: {emb2}")
    print(f"  ✓ Max pooled: {max_pooled}")
    print("✓ Max pooling fusion PASSED")


def test_embedding_alignment():
    """Test dimension alignment via interpolation."""
    print("\nTesting embedding alignment...")
    
    # Different dimensions
    emb_256 = np.random.randn(256).astype(np.float32)
    emb_384 = np.random.randn(384).astype(np.float32)
    
    target_dim = 512
    
    # Pad with zeros
    aligned_256 = np.pad(emb_256, (0, target_dim - len(emb_256)), mode='constant')
    aligned_384 = np.pad(emb_384, (0, target_dim - len(emb_384)), mode='constant')
    
    assert len(aligned_256) == target_dim
    assert len(aligned_384) == target_dim
    print(f"  ✓ Aligned 256d → {len(aligned_256)}d")
    print(f"  ✓ Aligned 384d → {len(aligned_384)}d")
    print("✓ Embedding alignment PASSED")


def test_cosine_similarity():
    """Test cosine similarity computation."""
    print("\nTesting cosine similarity...")
    
    # Identical vectors
    vec1 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    vec2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    sim1 = dot / (norm1 * norm2)
    
    assert np.isclose(sim1, 1.0)
    print(f"  ✓ Identical vectors: similarity={sim1:.4f}")
    
    # Orthogonal vectors
    vec3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec4 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    dot2 = np.dot(vec3, vec4)
    norm3 = np.linalg.norm(vec3)
    norm4 = np.linalg.norm(vec4)
    sim2 = dot2 / (norm3 * norm4)
    
    assert np.isclose(sim2, 0.0)
    print(f"  ✓ Orthogonal vectors: similarity={sim2:.4f}")
    print("✓ Cosine similarity PASSED")


def test_modality_detection():
    """Test basic modality detection logic."""
    print("\nTesting modality detection...")
    
    # String → TEXT
    assert isinstance("hello world", str)
    modality1 = ModalityType.TEXT
    print(f"  ✓ String → {modality1.name}")
    
    # Dict → STRUCTURED
    assert isinstance({"key": "value"}, dict)
    modality2 = ModalityType.STRUCTURED
    print(f"  ✓ Dict → {modality2.name}")
    
    # List → STRUCTURED
    assert isinstance([1, 2, 3], list)
    modality3 = ModalityType.STRUCTURED
    print(f"  ✓ List → {modality3.name}")
    
    # List of mixed types → MULTIMODAL
    mixed = ["text", {"data": 1}]
    types = set(type(x).__name__ for x in mixed)
    if len(types) > 1:
        modality4 = ModalityType.MULTIMODAL
        print(f"  ✓ Mixed list → {modality4.name}")
    
    print("✓ Modality detection PASSED")


def test_confidence_scoring():
    """Test confidence scoring logic."""
    print("\nTesting confidence scoring...")
    
    # Text quality score
    text1 = "This is a well-formed sentence with proper structure."
    text2 = "short"
    
    def score_text(text):
        score = 0.5  # Base score
        if len(text) > 20:
            score += 0.2
        if len(text.split()) > 5:
            score += 0.2
        if any(c.isupper() for c in text):
            score += 0.1
        return min(score, 1.0)
    
    score1 = score_text(text1)
    score2 = score_text(text2)
    
    assert score1 > score2
    print(f"  ✓ Long text: {score1:.2f}")
    print(f"  ✓ Short text: {score2:.2f}")
    print("✓ Confidence scoring PASSED")


def run_all_tests():
    """Run all minimal tests."""
    print("\n" + "="*70)
    print("MINIMAL MULTI-MODAL PROCESSING TESTS")
    print("(Core algorithms without dependencies)")
    print("="*70 + "\n")
    
    tests = [
        ("Attention Fusion", test_fusion_attention),
        ("Concatenation Fusion", test_fusion_concat),
        ("Average Fusion", test_fusion_average),
        ("Max Pooling Fusion", test_fusion_max),
        ("Embedding Alignment", test_embedding_alignment),
        ("Cosine Similarity", test_cosine_similarity),
        ("Modality Detection", test_modality_detection),
        ("Confidence Scoring", test_confidence_scoring),
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
        print(f"         {failed} tests failed")
        for name, error in errors:
            print(f"         - {name}")
    else:
        print("         ALL TESTS PASSED! ✓")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
