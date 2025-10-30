"""
Smart Context Assembly Demo

Shows how awareness + memory combine into optimally-packed LLM prompts:
- Token budget optimization
- Hierarchical compression
- Importance-based selection
- Awareness-guided packing
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.awareness.compositional_awareness import CompositionalAwarenessLayer
from HoloLoom.awareness.context_packer import (
    SmartContextPacker,
    TokenBudget,
    ContextElement,
    ContextImportance,
    CompressionLevel
)


def print_section(title: str, char: str = "="):
    """Print section header"""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


async def demo_1_basic_packing():
    """Demo 1: Basic context packing"""
    print_section("Demo 1: Basic Context Packing")
    
    # Create awareness layer
    awareness = CompositionalAwarenessLayer()
    
    # Create context packer with moderate budget
    packer = SmartContextPacker(
        token_budget=TokenBudget(
            total=4000,
            reserved_for_query=300,
            reserved_for_response=700
        )
    )
    
    # Test query
    query = "What are the key patterns in quantum entanglement?"
    
    # Get awareness context
    awareness_ctx = await awareness.get_unified_context(query)
    
    # Pack context (no memory for now)
    packed = await packer.pack_context(query, awareness_ctx)
    
    print(f"Query: {query}\n")
    print(f"📊 Packing Statistics:")
    print(f"  Total tokens: {packed.total_tokens}")
    print(f"  Elements included: {packed.elements_included}")
    print(f"  Elements compressed: {packed.elements_compressed}")
    print(f"  Elements excluded: {packed.elements_excluded}")
    print(f"  Average importance: {packed.avg_importance:.2f}")
    print(f"  Min importance: {packed.min_importance:.2f}")
    print(f"  Packing time: {packed.packing_time_ms:.2f}ms")
    
    print(f"\n📝 Formatted for LLM:")
    print("-" * 80)
    print(packed.format_for_llm(include_metadata=True))


async def demo_2_with_memories():
    """Demo 2: Context packing with memory integration"""
    print_section("Demo 2: Context Packing with Memories")
    
    # Create components
    awareness = CompositionalAwarenessLayer()
    packer = SmartContextPacker()
    
    # Test query
    query = "How does quantum tunneling work?"
    
    # Get awareness context
    awareness_ctx = await awareness.get_unified_context(query)
    
    # Simulate memory retrieval results
    mock_memories = [
        {
            'text': 'Quantum tunneling is a quantum mechanical phenomenon where particles pass through potential barriers.',
            'score': 0.95,
            'timestamp': '2024-01-15'
        },
        {
            'text': 'In classical physics, a particle needs sufficient energy to overcome a barrier. Quantum mechanics allows probabilistic tunneling.',
            'score': 0.88,
            'timestamp': '2024-01-14'
        },
        {
            'text': 'Scanning tunneling microscopes use quantum tunneling to image surfaces at atomic resolution.',
            'score': 0.82,
            'timestamp': '2024-01-10'
        },
        {
            'text': 'Alpha decay in radioactive nuclei occurs through quantum tunneling of alpha particles.',
            'score': 0.75,
            'timestamp': '2024-01-08'
        },
        {
            'text': 'Quantum tunneling has applications in flash memory and tunnel diodes.',
            'score': 0.68,
            'timestamp': '2024-01-05'
        }
    ]
    
    # Pack with memories
    packed = await packer.pack_context(
        query,
        awareness_ctx,
        memory_results=mock_memories,
        max_memories=5
    )
    
    print(f"Query: {query}\n")
    print(f"📊 Packing Statistics:")
    print(f"  Total tokens: {packed.total_tokens}")
    print(f"  Elements included: {packed.elements_included}")
    print(f"  Compression stats: {packed.compression_stats}")
    print(f"  Packing time: {packed.packing_time_ms:.2f}ms")
    
    print(f"\n📝 Formatted Context:")
    print("-" * 80)
    print(packed.format_for_llm())


async def demo_3_budget_constraints():
    """Demo 3: Token budget constraints and compression"""
    print_section("Demo 3: Token Budget Constraints")
    
    awareness = CompositionalAwarenessLayer()
    query = "Explain quantum entanglement, superposition, and decoherence in detail"
    
    # Test with different budgets
    budgets = [
        ("Tight", TokenBudget(total=2000, reserved_for_query=200, reserved_for_response=300)),
        ("Moderate", TokenBudget(total=4000, reserved_for_query=300, reserved_for_response=700)),
        ("Generous", TokenBudget(total=8000, reserved_for_query=500, reserved_for_response=1000))
    ]
    
    # Rich memory set
    rich_memories = [
        {'text': f'Memory {i}: ' + 'Quantum mechanics is fascinating. ' * 20, 'score': 0.9 - i*0.1}
        for i in range(15)
    ]
    
    awareness_ctx = await awareness.get_unified_context(query)
    
    for budget_name, budget_config in budgets:
        packer = SmartContextPacker(token_budget=budget_config)
        packed = await packer.pack_context(query, awareness_ctx, rich_memories, max_memories=15)
        
        print(f"\n🎯 {budget_name} Budget ({budget_config.total} tokens):")
        print(f"  Available for context: {budget_config.available_for_context}")
        print(f"  Actual usage: {packed.total_tokens}")
        print(f"  Elements: {packed.elements_included} included, "
              f"{packed.elements_compressed} compressed, "
              f"{packed.elements_excluded} excluded")
        print(f"  Compression breakdown: {packed.compression_stats}")


async def demo_4_importance_scoring():
    """Demo 4: Importance-based element selection"""
    print_section("Demo 4: Importance-Based Selection")
    
    from HoloLoom.awareness.context_packer import ContextElement, ContextImportance
    
    # Create mock elements at different importance levels
    elements = [
        ContextElement(
            content="User query: What is quantum computing?",
            importance=ContextImportance.CRITICAL.value,
            token_count=10,
            source="query",
            metadata={"type": "query"}
        ),
        ContextElement(
            content="High confidence pattern: quantum mechanics domain",
            importance=ContextImportance.HIGH.value,
            token_count=15,
            source="awareness",
            metadata={"type": "pattern"},
            summary="Pattern: quantum mechanics"
        ),
        ContextElement(
            content="Recent memory about quantum gates and qubits...",
            importance=ContextImportance.HIGH.value,
            token_count=20,
            source="memory",
            metadata={"type": "memory"},
            summary="Memory: quantum gates"
        ),
        ContextElement(
            content="Related concept: superposition allows multiple states",
            importance=ContextImportance.MEDIUM.value,
            token_count=25,
            source="memory",
            metadata={"type": "memory"},
            summary="Concept: superposition"
        ),
        ContextElement(
            content="Background: quantum computing history dates to 1980s...",
            importance=ContextImportance.LOW.value,
            token_count=30,
            source="memory",
            metadata={"type": "memory"},
            summary="History: 1980s origin"
        )
    ]
    
    print("📋 Elements by importance:\n")
    for i, elem in enumerate(sorted(elements, key=lambda e: e.importance, reverse=True), 1):
        importance_name = {
            1.0: "CRITICAL",
            0.8: "HIGH",
            0.5: "MEDIUM",
            0.2: "LOW"
        }.get(elem.importance, "CUSTOM")
        
        print(f"{i}. [{importance_name:8}] {elem.content[:60]}...")
        print(f"   Source: {elem.source}, Tokens: {elem.token_count}")
        if elem.summary:
            print(f"   Summary: {elem.summary}")
        print()
    
    # Show compression strategies
    print("\n🗜️  Compression Strategies:\n")
    
    for level in CompressionLevel:
        print(f"{level.value.upper()}:")
        for elem in elements[:3]:  # Show first 3
            compressed = elem.compress(level)
            original_len = len(elem.content)
            compressed_len = len(compressed)
            ratio = (1 - compressed_len/original_len) * 100 if original_len > 0 else 0
            print(f"  - {compressed[:50]}... ({compressed_len} chars, {ratio:.0f}% reduction)")
        print()


async def demo_5_awareness_guided_packing():
    """Demo 5: How awareness signals guide packing decisions"""
    print_section("Demo 5: Awareness-Guided Packing")
    
    awareness = CompositionalAwarenessLayer()
    packer = SmartContextPacker(
        token_budget=TokenBudget(total=3000, reserved_for_query=200, reserved_for_response=500)
    )
    
    # Test queries with different awareness profiles
    test_cases = [
        ("What is quantum entanglement?", "Familiar domain"),
        ("Explain zxcvbnm qwerty asdfgh", "High uncertainty"),
        ("How do neural networks learn?", "Medium confidence")
    ]
    
    for query, expected_profile in test_cases:
        print(f"\n🔍 Query: {query}")
        print(f"Expected: {expected_profile}\n")
        
        # Get awareness context
        awareness_ctx = await awareness.get_unified_context(query)
        
        # Mock memories
        memories = [
            {'text': f'Relevant memory {i} about the topic', 'score': 0.9 - i*0.1}
            for i in range(5)
        ]
        
        # Pack context
        packed = await packer.pack_context(query, awareness_ctx, memories, max_memories=5)
        
        # Show awareness signals
        conf = awareness_ctx.confidence
        patterns = awareness_ctx.patterns
        
        print(f"📊 Awareness Signals:")
        print(f"  Confidence: {1.0 - conf.uncertainty_level:.2f}")
        print(f"  Uncertainty: {conf.uncertainty_level:.2f}")
        print(f"  Domain: {patterns.domain}/{patterns.subdomain}")
        print(f"  Familiarity: {patterns.seen_count}× seen")
        print(f"  Cache status: {conf.query_cache_status}")
        
        print(f"\n📦 Packing Results:")
        print(f"  Elements included: {packed.elements_included}")
        print(f"  Average importance: {packed.avg_importance:.2f}")
        print(f"  Compression used: {packed.elements_compressed}/{packed.elements_included}")
        print(f"  Token usage: {packed.total_tokens}/{packer.budget.available_for_context}")


async def demo_6_full_pipeline():
    """Demo 6: Complete pipeline - awareness → memory → packing → LLM"""
    print_section("Demo 6: Full Pipeline Integration")
    
    print("🔄 Complete flow: Query → Awareness → Memory → Packing → LLM\n")
    
    # 1. Query arrives
    query = "What are the applications of quantum tunneling in modern technology?"
    print(f"1️⃣  Query: {query}\n")
    
    # 2. Awareness analysis
    print("2️⃣  Analyzing with awareness layer...")
    awareness = CompositionalAwarenessLayer()
    awareness_ctx = await awareness.get_unified_context(query)
    
    conf = awareness_ctx.confidence
    patterns = awareness_ctx.patterns
    print(f"   ✓ Confidence: {1.0 - conf.uncertainty_level:.2f}")
    print(f"   ✓ Domain: {patterns.domain}/{patterns.subdomain}")
    print(f"   ✓ Structure: {awareness_ctx.structural.suggested_response_type}\n")
    
    # 3. Memory retrieval (simulated)
    print("3️⃣  Retrieving from memory...")
    memories = [
        {'text': 'Quantum tunneling enables flash memory by allowing electrons to pass through insulating barriers.',
         'score': 0.92},
        {'text': 'Scanning tunneling microscopes (STM) use quantum tunneling for atomic-scale imaging.',
         'score': 0.89},
        {'text': 'Tunnel diodes exploit quantum tunneling for ultrafast switching in electronics.',
         'score': 0.85},
        {'text': 'Quantum tunneling is crucial for nuclear fusion in stars like our Sun.',
         'score': 0.78}
    ]
    print(f"   ✓ Retrieved {len(memories)} relevant memories\n")
    
    # 4. Smart packing
    print("4️⃣  Packing context intelligently...")
    packer = SmartContextPacker(
        token_budget=TokenBudget(total=4000, reserved_for_query=300, reserved_for_response=800)
    )
    packed = await packer.pack_context(query, awareness_ctx, memories, max_memories=10)
    
    print(f"   ✓ Packed {packed.elements_included} elements")
    print(f"   ✓ Used {packed.total_tokens}/{packer.budget.available_for_context} tokens")
    print(f"   ✓ Compressed {packed.elements_compressed} elements")
    print(f"   ✓ Average importance: {packed.avg_importance:.2f}\n")
    
    # 5. Format for LLM
    print("5️⃣  Formatted prompt for LLM:")
    print("=" * 80)
    llm_prompt = packed.format_for_llm()
    print(llm_prompt)
    print("=" * 80)
    
    print(f"\n✅ Ready for LLM generation!")
    print(f"   Total context: {packed.total_tokens} tokens")
    print(f"   Remaining for response: {packer.budget.reserved_for_response} tokens")


async def main():
    """Run all demos"""
    print("\n" + "🧠" * 40)
    print("SMART CONTEXT ASSEMBLY DEMO".center(80))
    print("The Bridge Between Consciousness and Generation".center(80))
    print("🧠" * 40)
    
    demos = [
        ("Basic Packing", demo_1_basic_packing),
        ("With Memories", demo_2_with_memories),
        ("Budget Constraints", demo_3_budget_constraints),
        ("Importance Scoring", demo_4_importance_scoring),
        ("Awareness Guided", demo_5_awareness_guided_packing),
        ("Full Pipeline", demo_6_full_pipeline)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            print(f"\n✅ {name} completed successfully")
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎯 Context Assembly Demo Complete!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
