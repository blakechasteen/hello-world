"""
Thread Branching Demo

Demonstrates natural conversation flow with thread branching:
- Create main thread for orchard planning
- Discuss apple trees and spacing
- Branch into pollination topic mid-conversation
- Return to original thread
- Show YarnGraph relationships

Run:
    PYTHONPATH=. python HoloLoom/voice_first/demo_thread_branching.py
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any


# ============================================================================
# Mock Components (Simplified)
# ============================================================================

@dataclass
class Message:
    """Message in a thread"""
    role: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class Thread:
    """Conversation thread"""
    thread_id: str
    name: str
    messages: List[Message]
    metadata: Dict[str, Any]


class SimpleThreadManager:
    """Simplified ThreadManager for demo"""

    def __init__(self):
        self.threads: Dict[str, Thread] = {}
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

    def get_active_thread(self) -> Thread:
        """Get active thread"""
        if self.active_thread_id:
            return self.threads.get(self.active_thread_id)
        return None

    def switch_thread(self, thread_id: str):
        """Switch active thread"""
        if thread_id in self.threads:
            self.active_thread_id = thread_id

    def add_message(self, role: str, content: str):
        """Add message to active thread"""
        thread = self.get_active_thread()
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
        self.edges: List[Dict[str, Any]] = []

    def add_edge(self, edge):
        """Add edge to graph"""
        self.edges.append({
            'source': edge.src,
            'target': edge.dst,
            'relation': edge.type,
            'branch_name': edge.metadata.get('branch_name', '')
        })

    def show_branches(self):
        """Show all branch relationships"""
        print("\n🌳 Thread Relationship Graph:")
        print("=" * 60)

        if not self.edges:
            print("  No branches yet")
            return

        for edge in self.edges:
            print(f"  '{edge['source']}' --[{edge['relation']}]--> '{edge['target']}'")
            if edge['branch_name']:
                print(f"    Branch: {edge['branch_name']}")


# ============================================================================
# Demo Scenario
# ============================================================================

async def run_demo():
    """
    Run complete thread branching demo.

    Scenario:
    1. User discusses apple tree spacing in orchard planning thread
    2. Biochar topic mentioned - important!
    3. User forks into biochar thread (context inherited)
    4. Discuss biochar production
    5. Return to orchard thread
    6. Show YarnGraph relationships
    """
    from HoloLoom.voice_first.thread import ThreadBrancher

    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

    print("🎙️ Voice-First Thread Branching Demo")
    print("=" * 60)

    # Initialize components
    thread_manager = SimpleThreadManager()
    yarn_graph = SimpleYarnGraph()
    brancher = ThreadBrancher(
        thread_manager=thread_manager,
        yarn_graph=yarn_graph,
        default_lookback_seconds=30.0
    )

    # Step 1: Create initial thread for orchard planning
    print("\n👤 User: 'start a new thread for orchard planning'")
    orchard_id = thread_manager.create_thread("orchard planning")
    print(f"✅ Created thread '{orchard_id}' (orchard planning)")

    await asyncio.sleep(0.5)

    # Step 2: Discuss apple trees
    print("\n👤 User: 'How should I space apple trees?'")
    thread_manager.add_message("user", "How should I space apple trees?")
    thread_manager.add_message(
        "assistant",
        "Standard spacing for apple trees is 15-20 feet apart. Consider soil type and rootstock."
    )
    print("🤖 Elle: [provides spacing guidance]")

    await asyncio.sleep(1)

    # Step 3: Biochar mentioned - user realizes this is important
    print("\n👤 User: 'What about biochar for soil amendment?'")
    thread_manager.add_message("user", "What about biochar for soil amendment?")
    thread_manager.add_message(
        "assistant",
        "Biochar can significantly improve soil structure and water retention. It's produced through pyrolysis."
    )
    print("🤖 Elle: [explains biochar benefits]")

    await asyncio.sleep(1)

    # Step 4: User branches - this is important enough for separate exploration
    print("\n👤 User: 'this is important, fork this into biochar production'")
    print("🔀 Branching thread...")

    branch = await brancher.fork_thread(
        parent_thread_id=orchard_id,
        branch_name="biochar production",
        lookback_seconds=30.0  # Inherit last 30 seconds
    )

    print(f"\n✨ Created branch thread '{branch.branch_id}'")
    print(f"   Parent: {branch.parent_thread_name}")
    print(f"   Context inherited: {len(branch.context.messages)} messages")
    print(f"   Entities extracted: {', '.join(branch.context.entities[:5])}")

    # Switch to branch thread
    thread_manager.switch_thread(branch.branch_id)

    await asyncio.sleep(1)

    # Step 5: Continue in biochar branch
    print("\n👤 User: 'What temperature range works best?'")
    thread_manager.add_message("user", "What temperature range works best?")
    thread_manager.add_message(
        "assistant",
        "Biochar production works best at 400-700°C. Higher temps create more stable carbon."
    )
    print("🤖 Elle: [discusses pyrolysis temperature]")

    await asyncio.sleep(1)

    print("\n👤 User: 'Can I use agricultural waste?'")
    thread_manager.add_message("user", "Can I use agricultural waste?")
    thread_manager.add_message(
        "assistant",
        "Yes! Orchard prunings, nut shells, and woody debris are excellent feedstock."
    )
    print("🤖 Elle: [explains feedstock options]")

    await asyncio.sleep(1)

    # Step 6: Return to original thread
    print("\n👤 User: 'go back to orchard planning thread'")
    thread_manager.switch_thread(orchard_id)
    print(f"✅ Switched back to '{thread_manager.get_active_thread().name}'")

    await asyncio.sleep(0.5)

    # Step 7: Continue orchard discussion
    print("\n👤 User: 'What about pest management?'")
    thread_manager.add_message("user", "What about pest management?")
    thread_manager.add_message(
        "assistant",
        "Integrated pest management combines beneficial insects, monitoring, and selective treatments."
    )
    print("🤖 Elle: [discusses IPM]")

    # Step 8: Show thread summary
    print("\n\n📊 Thread Summary")
    print("=" * 60)

    for thread_id, thread in thread_manager.threads.items():
        print(f"\n🧵 {thread.name} ({thread_id})")
        print(f"   Messages: {len(thread.messages)}")
        print(f"   Status: {'[ACTIVE]' if thread_id == thread_manager.active_thread_id else ''}")

        # Show branch metadata if it exists
        if 'branched_from' in thread.metadata:
            parent_id = thread.metadata['branched_from']
            parent = thread_manager.get_thread(parent_id)
            print(f"   Branched from: {parent.name}")
            print(f"   Inherited entities: {', '.join(thread.metadata.get('inherited_entities', [])[:3])}...")

    # Step 9: Show YarnGraph relationships
    yarn_graph.show_branches()

    # Step 10: Show conversation flow
    print("\n\n💬 Conversation Flow")
    print("=" * 60)

    print("\n🧵 Orchard Planning Thread:")
    orchard = thread_manager.get_thread(orchard_id)
    for msg in orchard.messages[:3]:  # Show first 3
        role_icon = "👤" if msg.role == "user" else "🤖"
        print(f"  {role_icon} {msg.content[:60]}...")

    print("\n🧵 Biochar Production Thread (branched):")
    biochar = thread_manager.get_thread(branch.branch_id)
    for msg in biochar.messages[:2]:  # Show first 2
        role_icon = "👤" if msg.role == "user" else "🤖"
        print(f"  {role_icon} {msg.content[:60]}...")

    print("\n\n✅ Demo Complete!")
    print("=" * 60)
    print("\n💡 Key Features Demonstrated:")
    print("  ✓ Natural conversation flow across threads")
    print("  ✓ Mid-conversation branching (fork this...)")
    print("  ✓ Context inheritance (last 30 seconds)")
    print("  ✓ Entity extraction (Biochar, Soil, etc.)")
    print("  ✓ YarnGraph relationship tracking")
    print("  ✓ Seamless thread switching")
    print("\n🎯 No more tab switching - thoughts flow naturally!")


if __name__ == "__main__":
    asyncio.run(run_demo())
