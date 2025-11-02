"""
Simple Unit Tests for Beta Wave Context Packer

Simplified tests that use the engine's built-in encode_new_memory method.
Run with: python test_beta_wave_packer_simple.py
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field

from HoloLoom.awareness.beta_wave_packer import (
    BetaWaveContextPacker,
    TokenBudget
)
from HoloLoom.memory.spring_dynamics_engine import (
    SpringDynamicsEngine,
    SpringEngineConfig
)


# Mock Awareness Context
@dataclass
class MockAwarenessContext:
    @dataclass
    class Confidence:
        uncertainty_level: float = 0.3
        query_cache_status: str = "MISS"
        knowledge_gap_detected: bool = False

    @dataclass
    class Patterns:
        domain: str = "quantum_computing"
        subdomain: str = "entanglement"
        seen_count: int = 5
        confidence: float = 0.85

    confidence: Confidence = field(default_factory=Confidence)
    patterns: Patterns = field(default_factory=Patterns)


def create_test_engine():
    """Create a spring engine with test memories"""
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    # Add test memories using the engine's method
    test_memories = [
        ('high_act', 'High activation memory - very relevant', 2.0),
        ('med_act', 'Medium activation memory - somewhat relevant', 1.0),
        ('low_act', 'Low activation memory - barely relevant', 0.3),
    ]

    for mem_id, content, k in test_memories:
        embedding = np.random.randn(128)
        engine.add_node(
            node_id=mem_id,
            content=content,
            embedding=embedding,
            initial_spring_constant=k
        )

    return engine


async def test_1_activation_thresholds():
    """Test that activation thresholds work correctly"""
    print("\n" + "="*70)
    print("TEST 1: Activation Thresholds")
    print("="*70)

    engine = create_test_engine()
    budget = TokenBudget(total=2000, reserved_for_query=200, reserved_for_response=500)
    packer = BetaWaveContextPacker(
        spring_engine=engine,
        token_budget=budget,
        activation_threshold=0.3,
        compression_threshold=0.7
    )

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test query about relevance",
        query_embedding=query_embedding,
        top_k=10
    )

    print(f"✓ Elements included: {packed.elements_included}")
    print(f"✓ Elements compressed: {packed.elements_compressed}")
    print(f"✓ Elements excluded: {packed.elements_excluded}")
    print(f"✓ Avg activation: {packed.avg_activation:.3f}")
    print(f"✓ Min activation: {packed.min_activation:.3f}")

    assert packed.elements_included >= 1, "Should include at least query"
    print("\n[PASS] Activation thresholds working!")


async def test_2_budget_respected():
    """Test that token budget is respected"""
    print("\n" + "="*70)
    print("TEST 2: Budget Constraints")
    print("="*70)

    engine = create_test_engine()

    # Tight budget
    tight_budget = TokenBudget(total=500, reserved_for_query=100, reserved_for_response=200)
    packer = BetaWaveContextPacker(engine, tight_budget)

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test",
        query_embedding=query_embedding,
        top_k=10
    )

    available = tight_budget.available_for_context
    print(f"✓ Budget available: {available} tokens")
    print(f"✓ Actual usage: {packed.total_tokens} tokens")
    print(f"✓ Elements included: {packed.elements_included}")

    assert packed.total_tokens <= available, f"Exceeded budget: {packed.total_tokens} > {available}"
    print("\n[PASS] Budget constraints working!")


async def test_3_awareness_integration():
    """Test awareness context integration"""
    print("\n" + "="*70)
    print("TEST 3: Awareness Integration")
    print("="*70)

    engine = create_test_engine()
    budget = TokenBudget()
    packer = BetaWaveContextPacker(engine, budget)
    awareness_ctx = MockAwarenessContext()

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test",
        query_embedding=query_embedding,
        awareness_context=awareness_ctx,
        top_k=10
    )

    print(f"✓ Query section: {len(packed.query_section)} chars")
    print(f"✓ Awareness section: {len(packed.awareness_section)} chars")
    print(f"✓ Memory section: {len(packed.memory_section)} chars")

    assert packed.awareness_section, "Should have awareness section"
    assert "Confidence" in packed.awareness_section or "Domain" in packed.awareness_section
    print("\n[PASS] Awareness integration working!")


async def test_4_empty_memories():
    """Test graceful handling of empty memories"""
    print("\n" + "="*70)
    print("TEST 4: Empty Memories Edge Case")
    print("="*70)

    # Empty engine
    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    budget = TokenBudget()
    packer = BetaWaveContextPacker(engine, budget)

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test query",
        query_embedding=query_embedding,
        top_k=10
    )

    print(f"✓ Elements included: {packed.elements_included}")
    print(f"✓ Query section exists: {bool(packed.query_section)}")

    assert packed.elements_included >= 1, "Should have at least query"
    assert packed.query_section, "Should have query section"
    print("\n[PASS] Empty memories handled gracefully!")


async def test_5_unicode_content():
    """Test Unicode content handling"""
    print("\n" + "="*70)
    print("TEST 5: Unicode Content")
    print("="*70)

    config = SpringEngineConfig()
    engine = SpringDynamicsEngine(config)

    # Add Unicode memory
    unicode_content = "Quantum entanglement 量子纠缠 🧲 Verschränkung"
    embedding = np.random.randn(128)
    engine.add_node("unicode_mem", embedding, content=unicode_content, initial_spring_constant=2.0)

    budget = TokenBudget()
    packer = BetaWaveContextPacker(engine, budget)

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test 测试 🔬",
        query_embedding=query_embedding,
        top_k=10
    )

    print(f"✓ Total tokens: {packed.total_tokens}")
    print(f"✓ Elements included: {packed.elements_included}")
    print(f"✓ Unicode query in section: {bool(packed.query_section)}")

    assert packed.total_tokens > 0, "Should have token count"
    print("\n[PASS] Unicode content handled!")


async def test_6_compression_logic():
    """Test compression reduces size"""
    print("\n" + "="*70)
    print("TEST 6: Compression Logic")
    print("="*70)

    engine = create_test_engine()
    budget = TokenBudget()
    packer = BetaWaveContextPacker(engine, budget)

    # Test compression method directly
    long_content = "A" * 1000
    compressed = packer._compress_content(long_content, ratio=0.5)

    print(f"✓ Original length: {len(long_content)}")
    print(f"✓ Compressed length: {len(compressed)}")
    print(f"✓ Reduction: {(1 - len(compressed)/len(long_content))*100:.1f}%")

    assert len(compressed) < len(long_content), "Compression should reduce size"
    print("\n[PASS] Compression logic working!")


async def test_7_activation_stats():
    """Test activation statistics tracking"""
    print("\n" + "="*70)
    print("TEST 7: Activation Statistics")
    print("="*70)

    engine = create_test_engine()
    budget = TokenBudget()
    packer = BetaWaveContextPacker(engine, budget)

    query_embedding = np.random.randn(128)
    packed = await packer.pack_context(
        query_text="Test",
        query_embedding=query_embedding,
        top_k=10
    )

    stats = packed.activation_stats
    print(f"✓ Stats keys: {list(stats.keys())}")
    print(f"✓ Beta wave iterations: {stats.get('beta_wave_iterations', 'N/A')}")
    print(f"✓ Total activated: {stats.get('total_activated', 'N/A')}")

    assert 'activation_distribution' in stats or 'beta_wave_iterations' in stats
    print("\n[PASS] Activation statistics tracked!")


async def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪" * 35)
    print("BETA WAVE CONTEXT PACKER - UNIT TESTS".center(70))
    print("Simple Physics-Based Testing".center(70))
    print("🧪" * 35)

    tests = [
        ("Activation Thresholds", test_1_activation_thresholds),
        ("Budget Constraints", test_2_budget_respected),
        ("Awareness Integration", test_3_awareness_integration),
        ("Empty Memories", test_4_empty_memories),
        ("Unicode Content", test_5_unicode_content),
        ("Compression Logic", test_6_compression_logic),
        ("Activation Statistics", test_7_activation_stats),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"TEST SUMMARY".center(70))
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎯 ALL TESTS PASSED! Beta Wave Context Packer is working!")
    else:
        print(f"\n⚠️  {failed} test(s) need attention")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
