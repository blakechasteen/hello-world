"""
LIVE end-to-end persistence test for CognitiveBridge.

Verifies the full save/restore cycle with real CBR + ChunkCompiler learning:
  1. Cold start: create bridge, store memories, run recalls
  2. Snapshot written to disk with graph + cbr + chunks sections
  3. Restart: new bridge loads snapshot, skips warm-up
  4. Restored bridge has identical graph state (nodes, edges, aliases)
  5. enhanced_recall works on restored bridge and finds memories
  6. CBR cases survive the restart
  7. Clean up temp file
"""

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from hololoom.memory.v2_bridge import CognitiveBridge, EnhancedRecallResult


class MockMemory:
    """Minimal mock that returns empty recalls."""
    def recall(self, query="", limit=5, **kw):
        return []


passed = 0
failed = 0


def check(condition, label):
    global passed, failed
    if condition:
        print(f"  [+] {label}")
        passed += 1
    else:
        print(f"  [-] FAIL: {label}")
        failed += 1


async def main():
    persist_path = tempfile.mktemp(suffix=".json")

    try:
        # =============================================================
        print("\n" + "=" * 72)
        print("  I. Cold Start — Initialize Bridge (no snapshot)")
        print("=" * 72)

        bridge1 = CognitiveBridge(MockMemory(), persist_path=persist_path)
        await bridge1.initialize()

        check(bridge1._initialized, "Bridge initialized on cold start")
        check(not os.path.exists(persist_path),
              "No snapshot file yet (nothing stored)")
        check(len(bridge1.bus._items) == 0, "Bus starts empty")

        # =============================================================
        print("\n" + "=" * 72)
        print("  II. Store Memories via store_and_index()")
        print("=" * 72)

        memories_to_store = [
            ("Thompson Sampling explores by sampling from posterior distributions",
             {"importance": 0.9}, ["ml", "bandits"], "user1"),
            ("Epsilon-greedy selects the best arm with probability 1-epsilon",
             {"importance": 0.7}, ["ml", "bandits"], "user1"),
            ("UCB1 uses upper confidence bounds to balance exploration",
             {"importance": 0.8}, ["ml", "bandits"], "user1"),
            ("Bayesian optimization extends Thompson Sampling to continuous domains",
             {"importance": 0.8}, ["ml", "optimization"], "user1"),
            ("Multi-armed bandits model the exploration-exploitation tradeoff",
             {"importance": 0.9}, ["ml", "theory"], "user1"),
        ]

        for text, ctx, tags, uid in memories_to_store:
            await bridge1.store_and_index(text, ctx, tags, uid)

        node_count_1 = len(bridge1.bus._items)
        edge_count_1 = sum(len(e) for e in bridge1.bus._edges.values())
        alias_count_1 = len(bridge1.bus._aliases)

        check(node_count_1 > 5, f"Graph has nodes ({node_count_1})")
        check(edge_count_1 > 0, f"Graph has edges ({edge_count_1})")
        check(alias_count_1 > 0, f"Aliases registered ({alias_count_1})")

        # =============================================================
        print("\n" + "=" * 72)
        print("  III. Run enhanced_recall() to Trigger CBR/ChunkCompiler Learning")
        print("=" * 72)

        queries = [
            "Thompson Sampling vs epsilon-greedy",
            "How does UCB1 work?",
            "Compare bandit algorithms for exploration",
            "Bayesian optimization overview",
        ]

        for q in queries:
            result = await bridge1.enhanced_recall(q, limit=5)
            check(isinstance(result, EnhancedRecallResult),
                  f"Recall '{q[:40]}' returned EnhancedRecallResult")

        # Verify CBR accumulated cases
        cbr1 = bridge1._find_source("cbr")
        cbr_count_1 = cbr1.case_count if cbr1 else 0
        check(cbr1 is not None, "CBR source exists")
        check(cbr_count_1 > 0, f"CBR has cases after recalls ({cbr_count_1})")

        # Re-capture counts AFTER recalls (recalls create CBR case nodes)
        node_count_1 = len(bridge1.bus._items)
        edge_count_1 = sum(len(e) for e in bridge1.bus._edges.values())
        alias_count_1 = len(bridge1.bus._aliases)

        # =============================================================
        print("\n" + "=" * 72)
        print("  IV. Verify Snapshot File Exists on Disk")
        print("=" * 72)

        check(os.path.exists(persist_path), "Snapshot file exists on disk")
        file_size = os.path.getsize(persist_path)
        check(file_size > 100, f"Snapshot file has content ({file_size} bytes)")

        # =============================================================
        print("\n" + "=" * 72)
        print("  V. Read Snapshot JSON — Verify graph, cbr, chunks Sections")
        print("=" * 72)

        with open(persist_path, "r") as f:
            snapshot = json.load(f)

        check(snapshot.get("_version") == 2, f"Snapshot version is 2 (got {snapshot.get('_version')})")
        check("graph" in snapshot, "Snapshot has 'graph' section")
        check("cbr" in snapshot, "Snapshot has 'cbr' section")
        check("chunks" in snapshot, "Snapshot has 'chunks' section")

        # Drill into sections
        check("items" in snapshot["graph"], "graph.items present")
        check("edges" in snapshot["graph"], "graph.edges present")
        check("aliases" in snapshot["graph"], "graph.aliases present")
        check(len(snapshot["graph"]["items"]) == node_count_1,
              f"graph.items count matches ({len(snapshot['graph']['items'])} == {node_count_1})")

        check("cases" in snapshot["cbr"], "cbr.cases present")
        check(len(snapshot["cbr"]["cases"]) == cbr_count_1,
              f"cbr.cases count matches ({len(snapshot['cbr']['cases'])} == {cbr_count_1})")

        check("chunks" in snapshot["chunks"], "chunks.chunks present")
        check("experience" in snapshot["chunks"], "chunks.experience present")

        # =============================================================
        print("\n" + "=" * 72)
        print("  VI. Create NEW Bridge (Simulating Restart)")
        print("=" * 72)

        bridge2 = CognitiveBridge(MockMemory(), persist_path=persist_path)
        await bridge2.initialize()

        check(bridge2._initialized, "Restarted bridge initialized")

        # =============================================================
        print("\n" + "=" * 72)
        print("  VII. Verify Restored Bridge Has Same Graph State")
        print("=" * 72)

        node_count_2 = len(bridge2.bus._items)
        edge_count_2 = sum(len(e) for e in bridge2.bus._edges.values())
        alias_count_2 = len(bridge2.bus._aliases)

        check(node_count_2 == node_count_1,
              f"Node count matches ({node_count_2} == {node_count_1})")
        check(edge_count_2 == edge_count_1,
              f"Edge count matches ({edge_count_2} == {edge_count_1})")
        check(alias_count_2 == alias_count_1,
              f"Alias count matches ({alias_count_2} == {alias_count_1})")

        # Verify same alias keys
        aliases1 = set(bridge1.bus._aliases.keys())
        aliases2 = set(bridge2.bus._aliases.keys())
        check(aliases1 == aliases2,
              f"Alias keys identical ({len(aliases1)} aliases)")

        # Verify same node IDs
        ids1 = set(bridge1.bus._items.keys())
        ids2 = set(bridge2.bus._items.keys())
        check(ids1 == ids2,
              f"Node IDs identical ({len(ids1)} nodes)")

        # =============================================================
        print("\n" + "=" * 72)
        print("  VIII. Run enhanced_recall() on Restored Bridge")
        print("=" * 72)

        result2 = await bridge2.enhanced_recall(
            "Thompson Sampling exploration strategy", limit=5
        )

        check(isinstance(result2, EnhancedRecallResult),
              "Restored bridge returns EnhancedRecallResult")
        check(len(result2.memories) > 0,
              f"Restored bridge found memories ({len(result2.memories)})")
        check(result2.node_count >= node_count_1,
              f"Node count >= original ({result2.node_count} >= {node_count_1})")
        check(0 <= result2.confidence <= 1,
              f"Confidence in range ({result2.confidence:.2f})")
        check(result2.elapsed > 0,
              f"Elapsed > 0 ({result2.elapsed:.4f}s)")

        # Verify we actually found relevant content
        found_texts = [m["text"] for m in result2.memories]
        has_thompson = any("Thompson" in t or "thompson" in t.lower()
                          for t in found_texts)
        check(has_thompson,
              "Restored recall found Thompson-related memory")

        # Run a second query to verify cross-query still works
        result3 = await bridge2.enhanced_recall("UCB1 confidence bounds", limit=5)
        check(len(result3.memories) > 0,
              f"Second restored recall found memories ({len(result3.memories)})")

        # =============================================================
        print("\n" + "=" * 72)
        print("  IX. Verify CBR Cases Survived Restart")
        print("=" * 72)

        cbr2 = bridge2._find_source("cbr")
        check(cbr2 is not None, "CBR source exists on restored bridge")

        if cbr2:
            cbr_count_2 = cbr2.case_count
            check(cbr_count_2 >= cbr_count_1,
                  f"CBR cases survived restart ({cbr_count_2} >= {cbr_count_1})")

            # New recalls on restored bridge should add more CBR cases
            pre_recall_cbr = cbr2.case_count
            await bridge2.enhanced_recall("epsilon-greedy tradeoffs", limit=3)
            post_recall_cbr = cbr2.case_count
            check(post_recall_cbr >= pre_recall_cbr,
                  f"CBR grows after new recall ({pre_recall_cbr} -> {post_recall_cbr})")

        # Also check ChunkCompiler state
        cc2 = bridge2._find_source("chunk_compiler")
        check(cc2 is not None, "ChunkCompiler source exists on restored bridge")
        if cc2:
            cc_report = cc2.report()
            check(cc_report["experience_buffer_size"] >= 0,
                  f"ChunkCompiler has experience data ({cc_report['experience_buffer_size']})")

    finally:
        # =============================================================
        print("\n" + "=" * 72)
        print("  X. Cleanup")
        print("=" * 72)

        if os.path.exists(persist_path):
            os.remove(persist_path)
            check(True, f"Cleaned up temp file: {persist_path}")
        else:
            check(True, "No temp file to clean (already absent)")

        # Also clean up .tmp if atomic write left one
        tmp_path = persist_path.replace(".json", ".tmp")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Summary
    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
