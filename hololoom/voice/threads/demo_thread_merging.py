"""
Thread Merging Demo

Demonstrates 3 merge strategies for combining conversation threads:
- APPEND: Simple chronological concatenation
- SYNTHESIZE: LLM-powered synthesis of insights
- PRESERVE_ALL: Keep everything with thread markers

Run:
    PYTHONPATH=. python HoloLoom/voice_first/demo_thread_merging.py
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# ============================================================================
# Mock Components
# ============================================================================

@dataclass
class Message:
    """Message in a thread"""
    role: str
    content: str
    timestamp: float
    metadata: dict[str, Any]


@dataclass
class Thread:
    """Conversation thread"""
    thread_id: str
    name: str
    messages: list[Message]
    metadata: dict[str, Any]


class SimpleThreadManager:
    """Simplified ThreadManager for demo"""

    def __init__(self):
        self.threads: dict[str, Thread] = {}
        self.active_thread_id: str = None
        self._next_id = 1

    def create_thread(self, name: str) -> str:
        """Create new thread"""
        thread_id = f"thread_{self._next_id}"
        self._next_id += 1

        self.threads[thread_id] = Thread(
            thread_id=thread_id,
            name=name,
            messages=[],
            metadata={}
        )
        self.active_thread_id = thread_id
        return thread_id

    def get_thread(self, thread_id: str) -> Thread:
        """Get thread by ID"""
        return self.threads.get(thread_id)

    def add_message(self, thread_id: str, role: str, content: str):
        """Add message to thread"""
        thread = self.get_thread(thread_id)
        if thread:
            msg = Message(
                role=role,
                content=content,
                timestamp=datetime.now().timestamp(),
                metadata={}
            )
            thread.messages.append(msg)


class SimpleYarnGraph:
    """Simplified YarnGraph for demo"""

    def __init__(self):
        self.edges: list[dict[str, Any]] = []

    def add_edge(self, edge):
        """Add edge to graph"""
        self.edges.append({
            'source': edge.src,
            'target': edge.dst,
            'relation': edge.type,
            'metadata': edge.metadata
        })

    def show_merges(self):
        """Show all merge relationships"""
        print("\n🔗 Thread Merge Graph:")
        print("=" * 60)

        if not self.edges:
            print("  No merges yet")
            return

        for edge in self.edges:
            print(f"  '{edge['source']}' --[{edge['relation']}]--> '{edge['target']}'")


class SimpleLLMClient:
    """Simplified LLM client for demo (mock synthesis)"""

    async def generate(self, prompt: str) -> str:
        """Mock LLM generation"""
        await asyncio.sleep(0.1)  # Simulate API call

        return """The three farming threads reveal complementary approaches to sustainable agriculture:

**Orchard Planning** focuses on spatial design and tree selection, emphasizing proper spacing (15-20 feet) and soil considerations.

**Biochar Production** introduces a soil amendment strategy using pyrolysis at 400-700°C to create stable carbon, which can significantly improve soil structure and water retention in orchards.

**Composting Methods** provides organic matter management through hot composting, which produces nutrients for both orchard establishment and ongoing maintenance.

Together, these approaches form an integrated system: orchards benefit from biochar-enriched soil and compost nutrients, while orchard waste (prunings) can be converted back into biochar or compost, creating a closed-loop sustainable system."""


# ============================================================================
# Demo Scenario
# ============================================================================

async def run_demo():
    """
    Run complete thread merging demo.

    Scenario:
    1. Create 3 separate threads (orchard, biochar, composting)
    2. Discuss different topics in each thread
    3. Merge using 3 different strategies
    4. Show results and YarnGraph relationships
    """
    # Set UTF-8 encoding for Windows console
    import sys

    from hololoom.voice.threads.thread import MergeStrategy, ThreadMerger
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    print("🔀 Voice-First Thread Merging Demo")
    print("=" * 60)

    # Initialize components
    thread_manager = SimpleThreadManager()
    yarn_graph = SimpleYarnGraph()
    llm_client = SimpleLLMClient()

    # ========================================================================
    # Setup: Create 3 threads with different discussions
    # ========================================================================

    print("\n📝 Creating 3 separate conversation threads...\n")

    # Thread 1: Orchard Planning
    orchard_id = thread_manager.create_thread("orchard planning")
    thread_manager.add_message(orchard_id, "user", "How should I space apple trees?")
    thread_manager.add_message(orchard_id, "assistant", "Standard spacing for apple trees is 15-20 feet apart.")
    thread_manager.add_message(orchard_id, "user", "What about dwarf varieties?")
    thread_manager.add_message(orchard_id, "assistant", "Dwarf varieties can be 8-10 feet apart.")
    print(f"✅ Created '{thread_manager.get_thread(orchard_id).name}' (4 messages)")

    await asyncio.sleep(0.3)

    # Thread 2: Biochar Production
    biochar_id = thread_manager.create_thread("biochar production")
    thread_manager.add_message(biochar_id, "user", "What temperature for biochar?")
    thread_manager.add_message(biochar_id, "assistant", "400-700°C works best for stable carbon.")
    thread_manager.add_message(biochar_id, "user", "Can I use orchard prunings?")
    thread_manager.add_message(biochar_id, "assistant", "Yes! Orchard prunings make excellent feedstock.")
    print(f"✅ Created '{thread_manager.get_thread(biochar_id).name}' (4 messages)")

    await asyncio.sleep(0.3)

    # Thread 3: Composting Methods
    compost_id = thread_manager.create_thread("composting methods")
    thread_manager.add_message(compost_id, "user", "Hot vs cold composting?")
    thread_manager.add_message(compost_id, "assistant", "Hot composting is faster (3-4 months vs 6-12 months).")
    thread_manager.add_message(compost_id, "user", "What temperature for hot composting?")
    thread_manager.add_message(compost_id, "assistant", "Maintain 130-150°F for optimal decomposition.")
    print(f"✅ Created '{thread_manager.get_thread(compost_id).name}' (4 messages)")

    # ========================================================================
    # Demo 1: APPEND Strategy (Simple Concatenation)
    # ========================================================================

    print("\n\n🔀 Demo 1: APPEND Strategy (Chronological Concatenation)")
    print("=" * 60)

    merger = ThreadMerger(
        thread_manager=thread_manager,
        yarn_graph=yarn_graph,
        llm_client=llm_client
    )

    # Create target for APPEND demo
    append_target_id = thread_manager.create_thread("farming knowledge (APPEND)")
    print("\n👤 User: 'merge orchard and biochar threads (APPEND strategy)'")

    result_append = await merger.merge_threads(
        target_thread_id=append_target_id,
        source_thread_ids=[orchard_id, biochar_id],
        strategy=MergeStrategy.APPEND
    )

    print(f"\n✨ Merged {len(result_append.source_thread_ids)} threads")
    print(f"   Strategy: {result_append.strategy.name}")
    print(f"   Messages merged: {result_append.messages_merged}")
    print("   Result: All messages in chronological order")

    # Show sample
    target = thread_manager.get_thread(append_target_id)
    print(f"\n📊 Result Preview ({len(target.messages)} total messages):")
    for i, msg in enumerate(target.messages[:3]):
        role_icon = "👤" if msg.role == "user" else "🤖"
        print(f"   {i+1}. {role_icon} {msg.content[:55]}...")

    # ========================================================================
    # Demo 2: SYNTHESIZE Strategy (LLM-Powered Synthesis)
    # ========================================================================

    print("\n\n🔀 Demo 2: SYNTHESIZE Strategy (LLM-Generated Synthesis)")
    print("=" * 60)

    # Create target for SYNTHESIZE demo
    synth_target_id = thread_manager.create_thread("farming knowledge (SYNTHESIZE)")
    print("\n👤 User: 'merge all 3 farming threads with synthesis'")
    print("🤖 Generating LLM synthesis...")

    result_synth = await merger.merge_threads(
        target_thread_id=synth_target_id,
        source_thread_ids=[orchard_id, biochar_id, compost_id],
        strategy=MergeStrategy.SYNTHESIZE
    )

    print(f"\n✨ Merged {len(result_synth.source_thread_ids)} threads")
    print(f"   Strategy: {result_synth.strategy.name}")
    print(f"   Messages merged: {result_synth.messages_merged}")
    print("   Result: LLM synthesis of all insights")

    # Show synthesis
    print("\n📝 LLM Synthesis:")
    print("-" * 60)
    if result_synth.synthesis:
        # Show first 500 chars of synthesis
        synthesis_preview = result_synth.synthesis[:500]
        if len(result_synth.synthesis) > 500:
            synthesis_preview += "..."
        print(synthesis_preview)

    # ========================================================================
    # Demo 3: PRESERVE_ALL Strategy (Thread Markers)
    # ========================================================================

    print("\n\n🔀 Demo 3: PRESERVE_ALL Strategy (Thread Markers)")
    print("=" * 60)

    # Create target for PRESERVE_ALL demo
    preserve_target_id = thread_manager.create_thread("farming knowledge (PRESERVE_ALL)")
    print("\n👤 User: 'merge threads but keep clear markers'")

    result_preserve = await merger.merge_threads(
        target_thread_id=preserve_target_id,
        source_thread_ids=[orchard_id, biochar_id],
        strategy=MergeStrategy.PRESERVE_ALL
    )

    print(f"\n✨ Merged {len(result_preserve.source_thread_ids)} threads")
    print(f"   Strategy: {result_preserve.strategy.name}")
    print(f"   Messages merged: {result_preserve.messages_merged}")
    print("   Result: All messages with [thread] markers")

    # Show sample with markers
    target = thread_manager.get_thread(preserve_target_id)
    print("\n📊 Result Preview (with thread markers):")
    for i, msg in enumerate(target.messages[:4]):
        role_icon = "👤" if msg.role == "user" else "🤖"
        # Show the thread marker
        content_preview = msg.content[:60]
        print(f"   {i+1}. {role_icon} {content_preview}...")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n\n📊 Merge Summary")
    print("=" * 60)

    all_results = [
        ("APPEND", result_append),
        ("SYNTHESIZE", result_synth),
        ("PRESERVE_ALL", result_preserve)
    ]

    for strategy_name, result in all_results:
        print(f"\n🔀 {strategy_name} Strategy:")
        print(f"   Sources: {len(result.source_thread_ids)} threads")
        print(f"   Messages: {result.messages_merged}")
        if result.synthesis:
            print("   Synthesis: ✅ Generated")

    # Show YarnGraph relationships
    yarn_graph.show_merges()

    # Strategy comparison
    print("\n\n💡 Strategy Comparison")
    print("=" * 60)

    comparisons = [
        ("APPEND", "Chronological order", "Simple, preserves timeline", "Basic concatenation"),
        ("SYNTHESIZE", "LLM synthesis", "Extracts insights", "Requires LLM, slower"),
        ("PRESERVE_ALL", "Thread markers", "Complete context", "More verbose")
    ]

    for strategy, result_type, pros, cons in comparisons:
        print(f"\n📌 {strategy}:")
        print(f"   Result: {result_type}")
        print(f"   Pros: {pros}")
        print(f"   Cons: {cons}")

    print("\n\n✅ Demo Complete!")
    print("=" * 60)
    print("\n🎯 Key Features Demonstrated:")
    print("  ✓ 3 merge strategies (APPEND, SYNTHESIZE, PRESERVE_ALL)")
    print("  ✓ LLM-powered synthesis of insights")
    print("  ✓ Thread markers for context preservation")
    print("  ✓ YarnGraph MERGED_INTO edges")
    print("  ✓ Chronological message ordering")
    print("\n💬 Natural commands:")
    print("  'merge orchard and biochar threads'")
    print("  'combine all farming threads with synthesis'")
    print("  'merge threads but keep clear markers'")


if __name__ == "__main__":
    asyncio.run(run_demo())
