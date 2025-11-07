"""
Demo: Writing System Integration with Weaving Orchestrator

Shows the complete pipeline:
1. Query → Orchestrator → Context Retrieval
2. Context → Writing System → Polished Output
3. Output → Export → Beautiful HTML
"""

import asyncio
import sys
from typing import List

# Fix Windows UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from HoloLoom.config import Config
from HoloLoom.documentation.types import MemoryShard, Query
from HoloLoom.writing import write
from HoloLoom.writing.export import HTMLExporter, MarkdownExporter
from HoloLoom.writing.core.protocol import WritingResult, WritingMode, StyleGuide


def create_test_shards() -> List[MemoryShard]:
    """Create test memory shards about Thompson Sampling."""
    return [
        MemoryShard(
            id='ts1',
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            metadata={'relevance': 0.95, 'topic': 'thompson_sampling', 'source': 'textbook'}
        ),
        MemoryShard(
            id='ts2',
            text="It balances exploration and exploitation by sampling from posterior distributions.",
            metadata={'relevance': 0.92, 'topic': 'exploration', 'source': 'textbook'}
        ),
        MemoryShard(
            id='ts3',
            text="Thompson Sampling maintains a probability distribution over the expected reward of each action.",
            metadata={'relevance': 0.90, 'topic': 'probability', 'source': 'research_paper'}
        ),
        MemoryShard(
            id='ts4',
            text="At each timestep, it samples a reward estimate from each action's posterior distribution.",
            metadata={'relevance': 0.88, 'topic': 'algorithm', 'source': 'research_paper'}
        ),
        MemoryShard(
            id='ts5',
            text="The action with the highest sampled reward is selected and executed.",
            metadata={'relevance': 0.87, 'topic': 'algorithm', 'source': 'research_paper'}
        ),
        MemoryShard(
            id='ts6',
            text="After observing the reward, the posterior distribution is updated using Bayes' theorem.",
            metadata={'relevance': 0.85, 'topic': 'bayesian', 'source': 'textbook'}
        ),
        MemoryShard(
            id='ts7',
            text="Thompson Sampling naturally handles the exploration-exploitation tradeoff without hyperparameters.",
            metadata={'relevance': 0.83, 'topic': 'advantages', 'source': 'blog'}
        ),
        MemoryShard(
            id='ts8',
            text="It has strong theoretical guarantees and empirical performance on many benchmark problems.",
            metadata={'relevance': 0.80, 'topic': 'performance', 'source': 'research_paper'}
        )
    ]


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


async def demo_simple_integration():
    """Demo 1: Simple integration with writing system."""
    print_section("Demo 1: Simple Write API")

    memories = create_test_shards()

    # Simple write (auto-detects mode, applies refinement)
    content = await write(
        "What is Thompson Sampling?",
        memories,
        refine=True
    )

    print("\n📝 Generated Content:")
    print(content)


async def demo_mode_selection():
    """Demo 2: Different modes for different use cases."""
    print_section("Demo 2: Mode Selection")

    memories = create_test_shards()

    # Test different modes
    modes = [
        ('narrative', "Explain Thompson Sampling to a beginner"),
        ('technical', "How to implement Thompson Sampling?"),
        ('analysis', "Compare Thompson Sampling and epsilon-greedy"),
    ]

    for mode, query in modes:
        print(f"\n🎯 Mode: {mode.upper()}")
        print(f"Query: {query}")
        print("-" * 60)

        content = await write(query, memories, mode=mode, refine=True)
        # Show first 200 chars
        preview = content[:200].replace('\n', ' ')
        print(f"{preview}...")


async def demo_refinement_quality():
    """Demo 3: Show refinement quality improvement."""
    print_section("Demo 3: Refinement Quality")

    memories = create_test_shards()
    query = "Explain Thompson Sampling"

    # Without refinement
    print("\n📄 Without Refinement:")
    draft = await write(query, memories, refine=False)
    print(draft[:300])

    # With refinement
    print("\n✨ With ELEGANCE Refinement (3 passes):")
    refined = await write(query, memories, refine=True)
    print(refined[:300])

    # Show improvement
    print(f"\n📊 Improvement:")
    print(f"Draft length: {len(draft)} chars")
    print(f"Refined length: {len(refined)} chars")


async def demo_export_formats():
    """Demo 4: Export to different formats."""
    print_section("Demo 4: Export Formats")

    memories = create_test_shards()

    # Generate content
    content = await write(
        "What is Thompson Sampling and how does it work?",
        memories,
        mode='narrative',
        refine=True
    )

    # Create WritingResult for export
    result = WritingResult(
        content=content,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE,
        quality_score=0.92,
        confidence=0.89,
        refinement_passes=[],
        metadata={'query': 'Thompson Sampling explanation'}
    )

    # Export to Markdown
    print("\n📄 Markdown Export:")
    md_exporter = MarkdownExporter(include_metadata=True, include_quality=True)
    markdown = md_exporter.export(result)
    print(markdown[:400])

    # Export to HTML
    print("\n🌐 HTML Export (with Tufte CSS):")
    html_exporter = HTMLExporter(include_quality_sidebar=True, include_tufte_css=True)
    html = html_exporter.export(result)
    print(f"Generated HTML: {len(html)} chars")
    print("Preview: <!DOCTYPE html>...")
    print(f"Includes: Tufte CSS, Quality sidebar, Responsive design")

    # Save files
    with open('demos/output/writing_demo.md', 'w', encoding='utf-8') as f:
        f.write(markdown)
    with open('demos/output/writing_demo.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("\n✓ Saved to demos/output/writing_demo.{md,html}")


async def demo_orchestrator_integration():
    """Demo 5: Full integration with orchestrator (simulated)."""
    print_section("Demo 5: Orchestrator Integration Pattern")

    print("\n📋 Standard Pattern:")
    print("""
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Step 1: Retrieve context
        spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))

        # Step 2: Generate polished output
        content = await write(
            spacetime.query.text,
            spacetime.trace.retrieved_context,
            refine=True
        )

        print(content)
    """)

    print("\n🎯 What This Does:")
    print("  1. Orchestrator retrieves relevant memories from knowledge graph")
    print("  2. Writing system transforms memories into polished prose")
    print("  3. Automatic mode detection (narrative for 'What is X?')")
    print("  4. Multi-pass ELEGANCE refinement (clarity → simplicity → beauty)")

    # Simulate the pattern
    memories = create_test_shards()
    content = await write(
        "What is Thompson Sampling?",
        memories,
        refine=True
    )

    print("\n✨ Generated Output:")
    print(content)


async def demo_multi_query_synthesis():
    """Demo 6: Multi-query research synthesis."""
    print_section("Demo 6: Multi-Query Research Synthesis")

    memories = create_test_shards()

    # Research queries
    queries = [
        "What is Thompson Sampling?",
        "How does Thompson Sampling work?",
        "What are the advantages of Thompson Sampling?",
        "What are the limitations of Thompson Sampling?"
    ]

    print("\n🔍 Research Queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    # Gather all contexts (simulated - in reality each query would retrieve different memories)
    all_memories = memories  # In real scenario, this would be union of all retrieved contexts

    # Synthesize into comprehensive report
    print("\n📊 Synthesizing comprehensive report...")
    report = await write(
        "Comprehensive analysis of Thompson Sampling",
        all_memories,
        mode='analysis',
        refine=True
    )

    print("\n📄 Synthesized Report:")
    print(report[:500])
    print("...")


async def demo_performance():
    """Demo 7: Performance characteristics."""
    print_section("Demo 7: Performance Characteristics")

    import time
    memories = create_test_shards()
    query = "Explain Thompson Sampling"

    # No refinement
    print("\n⚡ No Refinement:")
    start = time.perf_counter()
    content = await write(query, memories, refine=False)
    duration = (time.perf_counter() - start) * 1000
    print(f"Time: {duration:.1f}ms")
    print(f"Length: {len(content)} chars")

    # With refinement
    print("\n✨ With ELEGANCE Refinement:")
    start = time.perf_counter()
    content = await write(query, memories, refine=True)
    duration = (time.perf_counter() - start) * 1000
    print(f"Time: {duration:.1f}ms")
    print(f"Length: {len(content)} chars")

    print("\n📊 Expected Performance:")
    print("  • Simple generation: 10-20ms")
    print("  • With ELEGANCE: +40-60ms (3 passes)")
    print("  • With VERIFY: +60-80ms (3 passes)")
    print("  • Complete pipeline: 100-150ms")


async def main():
    """Run all demos."""
    print("🎨 HoloLoom Writing System - Integration Demos")
    print("=" * 80)

    demos = [
        ("Simple Write API", demo_simple_integration),
        ("Mode Selection", demo_mode_selection),
        ("Refinement Quality", demo_refinement_quality),
        ("Export Formats", demo_export_formats),
        ("Orchestrator Integration", demo_orchestrator_integration),
        ("Multi-Query Synthesis", demo_multi_query_synthesis),
        ("Performance", demo_performance),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# Demo {i}/{len(demos)}: {name}")
        print('#' * 80)

        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "=" * 80)
    print("✓ All demos complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
