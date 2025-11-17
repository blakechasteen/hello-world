#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Org-Mode Live File Monitoring
====================================

Demonstrates real-time file watching with incremental YarnGraph updates
and temporal evolution tracking.

This demo:
1. Creates a test org file
2. Starts live monitoring
3. Makes changes to the file (simulates editing in Emacs)
4. Shows incremental graph updates
5. Displays temporal evolution history

Usage:
    python demo_orgmode_live.py
    PYTHONPATH=. python demo_orgmode_live.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor, ChangeEvent
from HoloLoom.memory.graph import KG
from HoloLoom.spinningWheel.base import SpinnerConfig


async def demo_live_monitoring():
    """
    Demo: Live file monitoring with incremental updates.
    """
    print("="*80)
    print("DEMO: Org-Mode Live File Monitoring")
    print("="*80)
    print("\nThis demo shows real-time org-mode file monitoring with:")
    print("  • Watchdog-based file watching (inotify/FSEvents)")
    print("  • Incremental YarnGraph updates")
    print("  • Temporal evolution tracking")
    print("  • Change callbacks")

    # Create test org file
    test_file = Path('test_live.org')
    initial_content = """* Project Tasks                                       :work:

** TODO Write documentation                           :priority:
   DEADLINE: <2025-11-20>

   Need to document the new feature.

** IN-PROGRESS Code review

   Reviewing PR #123.

* Personal Notes                                      :personal:

** Journal Entry
   Today's accomplishments.
"""

    print(f"\n1. Creating test file: {test_file}")
    test_file.write_text(initial_content, encoding='utf-8')

    # Initialize KnowledgeGraph
    kg = KG()

    # Set up change callback
    change_count = [0]  # Use list for closure

    def on_change(event: ChangeEvent):
        """Callback when file changes."""
        change_count[0] += 1
        print(f"\n   🔔 Change #{change_count[0]} detected:")
        print(f"      Type: {event.change_type}")
        print(f"      Heading: {event.heading_id}")
        if event.change_type == 'todo_state_change':
            title = event.metadata.get('title', 'Unknown')
            print(f"      {title}: {event.old_value} → {event.new_value}")
        elif event.change_type == 'created':
            print(f"      New heading: {event.new_value}")
        elif event.change_type == 'deleted':
            print(f"      Deleted: {event.old_value}")

    # Create monitor
    config = SpinnerConfig(enable_enrichment=False)
    monitor = OrgLiveMonitor(
        yarn_graph=kg,
        watch_files=[str(test_file)],
        config=config,
        enable_temporal_evolution=True
    )

    monitor.on_change(on_change)

    print("\n2. Starting live monitor...")
    await monitor.start()

    # Show initial graph state
    stats = kg.stats()
    print(f"\n   Initial graph state:")
    print(f"   • Nodes: {stats['num_nodes']}")
    print(f"   • Edges: {stats['num_edges']}")

    print("\n3. Simulating file edits (waiting for watchdog to detect)...\n")

    # Wait for initial load to complete
    await asyncio.sleep(1)

    # === Edit 1: Change TODO state ===
    print("   📝 EDIT 1: Changing TODO state (TODO → DONE)")
    modified_content_1 = initial_content.replace('TODO Write documentation', 'DONE Write documentation')
    modified_content_1 = modified_content_1.replace('DEADLINE: <2025-11-20>', 'CLOSED: [2025-11-17 Sun 15:30]')
    test_file.write_text(modified_content_1, encoding='utf-8')
    await asyncio.sleep(1.5)  # Wait for watchdog

    # === Edit 2: Add new heading ===
    print("\n   📝 EDIT 2: Adding new heading")
    modified_content_2 = modified_content_1 + "\n** TODO Prepare presentation                      :urgent:\n   For team meeting.\n"
    test_file.write_text(modified_content_2, encoding='utf-8')
    await asyncio.sleep(1.5)

    # === Edit 3: Modify content ===
    print("\n   📝 EDIT 3: Modifying heading content")
    modified_content_3 = modified_content_2.replace(
        'Reviewing PR #123.',
        'Reviewing PR #123. Found 3 issues that need fixing.'
    )
    test_file.write_text(modified_content_3, encoding='utf-8')
    await asyncio.sleep(1.5)

    # === Edit 4: Add link ===
    print("\n   📝 EDIT 4: Adding link between headings")
    modified_content_4 = modified_content_3.replace(
        'For team meeting.',
        'For team meeting. See [[Write documentation][related docs]].'
    )
    test_file.write_text(modified_content_4, encoding='utf-8')
    await asyncio.sleep(1.5)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Show final graph state
    stats = kg.stats()
    print(f"\nFinal YarnGraph state:")
    print(f"  • Nodes: {stats['num_nodes']}")
    print(f"  • Edges: {stats['num_edges']}")
    print(f"  • Avg degree: {stats['avg_degree']:.2f}")

    # Show change history
    print(f"\nChange History ({len(monitor.change_history)} events):")
    for i, change in enumerate(monitor.change_history, 1):
        print(f"  {i}. [{change.timestamp.strftime('%H:%M:%S')}] {change.change_type}")
        if change.change_type == 'todo_state_change':
            title = change.metadata.get('title', 'Unknown')
            print(f"     {title}: {change.old_value} → {change.new_value}")

    # Show TODO transitions
    print(f"\nTODO State Transitions:")
    transitions = monitor.get_todo_transitions()
    for heading_id, trans_list in transitions.items():
        print(f"  {heading_id}:")
        for timestamp, old, new in trans_list:
            print(f"    [{timestamp.strftime('%H:%M:%S')}] {old} → {new}")

    # Show temporal evolution in graph
    print(f"\nTemporal Evolution Edges in Graph:")
    change_edges = [(u, v, d) for u, v, d in kg.G.edges(data=True) if d.get('type', '').startswith('CHANGE_')]
    print(f"  Found {len(change_edges)} temporal edges")
    for src, dst, data in change_edges[:5]:  # Show first 5
        print(f"    {src} --{data['type']}--> {dst}")
        if data.get('old_value') and data.get('new_value'):
            print(f"      {data['old_value']} → {data['new_value']}")

    # Show hierarchical structure
    print(f"\nHierarchical Structure (CHILD_OF edges):")
    child_edges = [(u, v, d) for u, v, d in kg.G.edges(data=True) if d.get('type') == 'CHILD_OF']
    for src, dst, data in child_edges:
        level = data.get('level', '?')
        print(f"    {src} --[CHILD_OF, level={level}]--> {dst}")

    # Demonstrate temporal queries
    print(f"\n" + "="*80)
    print("TEMPORAL QUERIES")
    print("="*80)

    # Query: When did this TODO change?
    print("\nQuery: Show all TODO state changes")
    todo_changes = monitor.get_change_history(change_type='todo_state_change')
    for change in todo_changes:
        title = change.metadata.get('title', change.heading_id)
        print(f"  • {title}")
        print(f"    {change.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {change.old_value} → {change.new_value}")

    # Query: What changed in the last 5 seconds?
    five_secs_ago = datetime.now().replace(microsecond=0) - __import__('datetime').timedelta(seconds=5)
    recent_changes = monitor.get_change_history(since=five_secs_ago)
    print(f"\nQuery: Changes in last 5 seconds ({len(recent_changes)} events)")
    for change in recent_changes:
        print(f"  • {change.change_type} @ {change.timestamp.strftime('%H:%M:%S')}")

    # Stop monitoring
    print("\n" + "="*80)
    await monitor.stop()

    # Cleanup
    print(f"\nCleaning up test file...")
    test_file.unlink()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
The OrgLiveMonitor enables:

✓ Real-time file watching using watchdog (inotify/FSEvents)
✓ Incremental YarnGraph updates (only changed headings)
✓ Temporal evolution tracking (preserves change history)
✓ Change callbacks for custom notifications
✓ Temporal queries ("when did this TODO complete?")
✓ Zero-copy synchronization (org-mode → graph)

This means you can:
1. Edit org files in Emacs as usual
2. HoloLoom automatically mirrors changes to the knowledge graph
3. Query not just current state, but historical evolution
4. Get real-time notifications of changes
5. Build time-aware AI agents that understand task progression

Org-mode remains the source of truth, HoloLoom provides the neural layer!
""")


async def demo_simple():
    """Simpler demo without file system events (for environments without watchdog)."""
    print("="*80)
    print("DEMO: Simple Incremental Update (No Watchdog)")
    print("="*80)

    print("\nThis demo shows incremental updates without file system watching.")
    print("(Use this if watchdog is not available)\n")

    from HoloLoom.spinningWheel.orgmode import OrgModeSpinner
    from HoloLoom.memory.graph import KG

    kg = KG()
    spinner = OrgModeSpinner(SpinnerConfig())

    # Initial state
    content_v1 = """
* Task List
** TODO Task 1
** TODO Task 2
"""

    print("1. Parsing initial content...")
    shards_v1 = await spinner.spin({'content': content_v1, 'source': 'test', 'episode': 'demo'})
    print(f"   Parsed {len(shards_v1)} headings")

    # Add to graph
    for shard in shards_v1:
        if shard.metadata.get('parent_id'):
            kg.add_edge(__import__('HoloLoom.memory.graph', fromlist=['KGEdge']).KGEdge(
                src=shard.id,
                dst=shard.metadata['parent_id'],
                type='CHILD_OF'
            ))

    print(f"   Graph: {kg.stats()['num_nodes']} nodes, {kg.stats()['num_edges']} edges")

    # Updated state
    content_v2 = """
* Task List
** DONE Task 1
** TODO Task 2
** TODO Task 3
"""

    print("\n2. Parsing updated content...")
    shards_v2 = await spinner.spin({'content': content_v2, 'source': 'test', 'episode': 'demo'})
    print(f"   Parsed {len(shards_v2)} headings")

    # Detect changes manually
    old_ids = {s.id for s in shards_v1}
    new_ids = {s.id for s in shards_v2}

    created = new_ids - old_ids
    deleted = old_ids - new_ids

    print(f"\n3. Changes detected:")
    print(f"   • Created: {len(created)} headings")
    print(f"   • Deleted: {len(deleted)} headings")
    print(f"   • Modified: {len(old_ids & new_ids)} headings (content may have changed)")

    print("\n✓ Incremental update complete!")


async def main():
    """Run appropriate demo based on available dependencies."""
    try:
        from watchdog.observers import Observer
        # Watchdog available
        await demo_live_monitoring()
    except ImportError:
        print("⚠ Watchdog not available - running simple demo")
        print("  Install with: pip install watchdog")
        print()
        await demo_simple()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
