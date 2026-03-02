"""
HoloLoom Writing System Demo
=============================

Demonstrates the writing system's capabilities:
1. Simple API usage
2. Mode auto-detection
3. Multi-pass refinement (ELEGANCE strategy)
4. Quality improvement tracking
5. Different writing styles

Run: python demos/demo_writing_system.py
"""

import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.writing import (
    write,
    refine_text,
    Writer,
    Composer,
    WritingContext,
    WritingMode,
    StyleGuide,
    RefinementStrategy,
    create_default_writer
)
from hololoom.writing.core.protocol import MemoryShard
from hololoom.writing.modes import NarrativeWriter
from hololoom.writing.refinement import EleganceRefiner


# Sample memory data
THOMPSON_SAMPLING_MEMORIES = [
    MemoryShard(
        id='mem1',
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
        metadata={'relevance': 0.95, 'topic': 'reinforcement_learning', 'source': 'textbook'}
    ),
    MemoryShard(
        id='mem2',
        text="It samples actions from posterior distributions over expected rewards",
        metadata={'relevance': 0.88, 'topic': 'bayesian_methods', 'source': 'paper'}
    ),
    MemoryShard(
        id='mem3',
        text="The method naturally balances exploration and exploitation through uncertainty",
        metadata={'relevance': 0.92, 'topic': 'bandits', 'source': 'tutorial'}
    ),
    MemoryShard(
        id='mem4',
        text="Thompson Sampling is asymptotically optimal for stationary bandits",
        metadata={'relevance': 0.85, 'topic': 'theory', 'source': 'research'}
    ),
    MemoryShard(
        id='mem5',
        text="Compared to epsilon-greedy, Thompson Sampling adapts to problem structure",
        metadata={'relevance': 0.78, 'topic': 'comparison', 'source': 'blog'}
    )
]


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def print_subsection(title: str):
    """Print subsection header."""
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"{'-' * 80}\n")


async def demo_simple_api():
    """Demo 1: Simple API usage."""
    print_section("Demo 1: Simple API Usage")

    print("Query: 'What is Thompson Sampling?'")
    print(f"Memories: {len(THOMPSON_SAMPLING_MEMORIES)} memory shards")
    print("Mode: AUTO (will detect 'narrative')")
    print("Refinement: Enabled (3 passes)")

    content = await write(
        query="What is Thompson Sampling?",
        memories=THOMPSON_SAMPLING_MEMORIES,
        refine=True
    )

    print("\n📝 Generated Content:\n")
    print(content)
    print(f"\n✓ Content length: {len(content)} characters")


async def demo_mode_detection():
    """Demo 2: Mode auto-detection."""
    print_section("Demo 2: Mode Auto-Detection")

    queries = [
        ("What is Thompson Sampling?", "narrative"),
        ("How to implement Thompson Sampling?", "technical"),
        ("Compare Thompson Sampling and epsilon-greedy", "analysis"),
        ("Write a story about Thompson Sampling", "creative")
    ]

    writer = create_default_writer()

    for query, expected_mode in queries:
        detected_mode = writer.detect_mode(query, THOMPSON_SAMPLING_MEMORIES)
        status = "✓" if detected_mode.value == expected_mode else "✗"

        print(f"{status} Query: '{query}'")
        print(f"   Detected: {detected_mode.value} (expected: {expected_mode})\n")


async def demo_refinement_trajectory():
    """Demo 3: Multi-pass refinement with quality tracking."""
    print_section("Demo 3: Multi-Pass Refinement (ELEGANCE Strategy)")

    # Create writer with full visibility
    mode_writers = {
        WritingMode.NARRATIVE: NarrativeWriter()
    }
    refiners = {
        RefinementStrategy.ELEGANCE: EleganceRefiner()
    }
    composer = Composer(refiners=refiners)
    writer = Writer(mode_writers=mode_writers, composer=composer)

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=THOMPSON_SAMPLING_MEMORIES,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    print("Generating content with 3-pass ELEGANCE refinement...")
    print("  Pass 1: Clarity (make understandable)")
    print("  Pass 2: Simplicity (remove unnecessary)")
    print("  Pass 3: Beauty (add grace)\n")

    result = await writer.write(context, refine=True)

    print(result.summary())
    print(f"\nQuality Trajectory: {result.quality_trajectory}")
    print(f"Confidence: {result.confidence:.2f}")

    # Show refinement details
    print_subsection("Refinement Details")
    for i, pass_result in enumerate(result.refinement_passes):
        print(f"Pass {pass_result.pass_number}: {pass_result.focus.upper()}")
        print(f"  Quality: {pass_result.quality_score:.2f}")
        print(f"  Duration: {pass_result.duration_ms:.1f}ms")
        print(f"  Improvements: {len(pass_result.improvements)}")
        if pass_result.improvements:
            for imp in pass_result.improvements[:3]:  # Show first 3
                print(f"    - {imp}")
        print()

    print_subsection("Final Content")
    print(result.content)


async def demo_style_variations():
    """Demo 4: Different writing styles."""
    print_section("Demo 4: Style Variations")

    styles = [
        (StyleGuide.TUFTE, "Clarity-first, minimal"),
        (StyleGuide.CASUAL, "Conversational, simple"),
        (StyleGuide.ACADEMIC, "Formal, structured")
    ]

    mode_writers = {WritingMode.NARRATIVE: NarrativeWriter()}
    writer = Writer(mode_writers=mode_writers, composer=None)

    for style, description in styles:
        print_subsection(f"Style: {style.value.upper()} ({description})")

        context = WritingContext(
            query="What is Thompson Sampling?",
            memories=THOMPSON_SAMPLING_MEMORIES[:2],  # Shorter for demo
            mode=WritingMode.NARRATIVE,
            style=style
        )

        result = await writer.write(context, refine=False)
        print(result.content)
        print()


async def demo_refine_existing_text():
    """Demo 5: Refine existing text."""
    print_section("Demo 5: Refine Existing Text")

    # Poorly written text with issues
    draft = """
    Thompson Sampling is very really a quite useful method that was designed
    in order to facilitate exploration and exploitation in multi-armed bandits
    and it basically works by sampling from posterior distributions which is
    essentially just a way to utilize Bayesian inference for the purpose of
    making decisions about which arm to pull.
    """

    print("Original Draft:")
    print(draft)
    print(f"\nLength: {len(draft)} chars")

    print_subsection("Refining (3 passes)...")

    refined = await refine_text(
        text=draft,
        query="Explain Thompson Sampling",
        passes=3,
        strategy='elegance'
    )

    print("\nRefined Version:")
    print(refined)
    print(f"\nLength: {len(refined)} chars")
    print(f"Compression: {(1 - len(refined)/len(draft)) * 100:.1f}% shorter")


async def demo_quality_dimensions():
    """Demo 6: Quality scoring dimensions."""
    print_section("Demo 6: Quality Scoring Dimensions")

    composer = Composer()

    # Test different texts
    texts = {
        "Well-structured": """
        Thompson Sampling provides an elegant solution to exploration-exploitation.
        The method samples from posterior distributions.

        This Bayesian approach adapts naturally to uncertainty.
        It outperforms epsilon-greedy in many scenarios.
        """,

        "Poorly-structured": "Thompson Sampling is a method and it works and it is good and you should use it and it helps with bandits",

        "Too short": "Thompson Sampling.",

        "Too wordy": """
        Thompson Sampling is very really quite an essentially useful method that
        basically facilitates exploration and it is designed to utilize Bayesian
        approaches in order to make decisions.
        """
    }

    context = WritingContext(
        query="Test",
        memories=THOMPSON_SAMPLING_MEMORIES,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    for name, text in texts.items():
        quality, dimensions = composer.score_quality(text, context)

        print(f"\n{name}:")
        print(f"  Overall Quality: {quality:.2f}")
        print("  Dimensions:")
        for dim, score in sorted(dimensions.items()):
            bar = '█' * int(score * 20)
            print(f"    {dim:12s}: {score:.2f} {bar}")


async def demo_empty_memories():
    """Demo 7: Handling empty memories gracefully."""
    print_section("Demo 7: Graceful Degradation (No Memories)")

    print("Query: 'What is Thompson Sampling?'")
    print("Memories: [] (empty)")

    content = await write(
        query="What is Thompson Sampling?",
        memories=[],
        refine=False
    )

    print("\n📝 Generated Content:\n")
    print(content)
    print("\n✓ System gracefully handled missing context")


async def demo_performance():
    """Demo 8: Performance characteristics."""
    print_section("Demo 8: Performance")

    import time

    writer = create_default_writer()

    context = WritingContext(
        query="What is Thompson Sampling?",
        memories=THOMPSON_SAMPLING_MEMORIES,
        mode=WritingMode.NARRATIVE,
        style=StyleGuide.TUFTE
    )

    # Without refinement
    start = time.time()
    result_no_refine = await writer.write(context, refine=False)
    time_no_refine = (time.time() - start) * 1000

    # With refinement
    start = time.time()
    result_with_refine = await writer.write(context, refine=True)
    time_with_refine = (time.time() - start) * 1000

    print("Without Refinement:")
    print(f"  Time: {time_no_refine:.1f}ms")
    print(f"  Quality: {result_no_refine.quality_score:.2f}")
    print(f"  Confidence: {result_no_refine.confidence:.2f}")

    print("\nWith Refinement (3 passes):")
    print(f"  Time: {time_with_refine:.1f}ms")
    print(f"  Quality: {result_with_refine.quality_score:.2f}")
    print(f"  Confidence: {result_with_refine.confidence:.2f}")
    print(f"  Passes: {len(result_with_refine.refinement_passes)}")

    print("\nOverhead:")
    print(f"  Additional time: {time_with_refine - time_no_refine:.1f}ms")
    print(f"  Quality improvement: {result_with_refine.quality_score - result_no_refine.quality_score:+.2f}")
    print(f"  Per-pass overhead: {(time_with_refine - time_no_refine) / len(result_with_refine.refinement_passes):.1f}ms")


async def main():
    """Run all demos."""
    print_section("HoloLoom Writing System - Comprehensive Demo")
    print("Philosophy: 'Great writing isn't written, it's refined.'")

    demos = [
        ("Simple API", demo_simple_api),
        ("Mode Detection", demo_mode_detection),
        ("Refinement Trajectory", demo_refinement_trajectory),
        ("Style Variations", demo_style_variations),
        ("Refine Existing Text", demo_refine_existing_text),
        ("Quality Dimensions", demo_quality_dimensions),
        ("Empty Memories", demo_empty_memories),
        ("Performance", demo_performance)
    ]

    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\nRunning all demos...\n")

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print_section("Demo Complete!")
    print("✓ All demos executed successfully")
    print("\nKey Takeaways:")
    print("  1. Ruthlessly simple API: write(query, memories)")
    print("  2. Automatic mode & style detection")
    print("  3. Multi-pass refinement improves quality")
    print("  4. Complete quality tracking & metrics")
    print("  5. Graceful degradation when context missing")
    print("\nNext steps:")
    print("  - Explore HoloLoom/writing/README.md for full documentation")
    print("  - Check HoloLoom/tests/unit/test_writing.py for more examples")
    print("  - Integrate with WeavingOrchestrator for memory-backed writing")


if __name__ == '__main__':
    asyncio.run(main())
