"""
Graph Persistence Tests for LiteMemoryBus + CognitiveBridge + CBR + ChunkCompiler

16 test groups covering:
  1. Save/load round-trip (items, edges, aliases match)
  2. Load missing file returns False
  3. Load corrupt JSON raises JSONDecodeError
  4. Load future version raises ValueError
  5. Secondary indices rebuilt on load
  6. Bridge skips warm-up when snapshot loaded
  7. Bridge falls back to warm-up on corrupt file
  8. persist_path=None preserves existing behavior
  9. Incremental save after each store_and_index
  10. Atomic write leaves no .tmp file
  11. StoredItem.to_snapshot preserves all fields
  12. Edges survive JSON round-trip (tuple <-> list)
  13. CBR case library survives round-trip
  14. ChunkCompiler chunks + experience survive round-trip
  15. Combined v2 snapshot contains all sections
  16. v1 snapshot backward compatibility
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(__file__))

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
    from hololoom.memory.lite_bus import LiteMemoryBus, StoredItem
    from hololoom.memory.bus import MemoryItem
    from hololoom.memory.v2_bridge import CognitiveBridge
    from hololoom.memory.unified import UnifiedMemory
    from hololoom.memory.v2.cbr import CBRSystem, Case
    from hololoom.memory.v2.chunk_compiler import (
        ChunkCompiler, CompiledChunk, ChunkAction, ExperienceRecord,
    )

    # Use a temp directory for all snapshot files
    tmpdir = tempfile.mkdtemp(prefix="holo_persist_")

    try:
        # =================================================================
        print("\n" + "=" * 72)
        print("  1. Save/Load Round-Trip")
        print("=" * 72)

        bus = LiteMemoryBus()
        await bus.initialize()

        id1 = await bus.store(MemoryItem(
            content="Thompson Sampling explores via posterior distributions",
            memory_type="factual", importance=0.8,
        ))
        id2 = await bus.store(MemoryItem(
            content="Thompson",
            memory_type="entity", importance=0.6,
        ))
        await bus.store_edge(id1, id2, "MENTIONS")
        check("thompson" in bus._aliases, "Alias registered for entity")

        snap_path = os.path.join(tmpdir, "round_trip.json")
        bus.save_snapshot(snap_path)

        bus2 = LiteMemoryBus()
        await bus2.initialize()
        loaded = bus2.load_snapshot(snap_path)

        check(loaded is True, "load_snapshot returns True")
        check(len(bus2._items) == len(bus._items),
              f"Items match ({len(bus2._items)} == {len(bus._items)})")
        check(bus2._aliases == bus._aliases,
              f"Aliases match ({len(bus2._aliases)})")

        orig_edge_count = sum(len(e) for e in bus._edges.values())
        loaded_edge_count = sum(len(e) for e in bus2._edges.values())
        check(loaded_edge_count == orig_edge_count,
              f"Edges match ({loaded_edge_count} == {orig_edge_count})")

        # =================================================================
        print("\n" + "=" * 72)
        print("  2. Load Missing File Returns False")
        print("=" * 72)

        bus3 = LiteMemoryBus()
        result = bus3.load_snapshot(os.path.join(tmpdir, "nonexistent.json"))
        check(result is False, "Missing file returns False")

        # =================================================================
        print("\n" + "=" * 72)
        print("  3. Load Corrupt JSON Raises JSONDecodeError")
        print("=" * 72)

        corrupt_path = os.path.join(tmpdir, "corrupt.json")
        with open(corrupt_path, "w") as f:
            f.write("{not valid json!!!")

        bus4 = LiteMemoryBus()
        try:
            bus4.load_snapshot(corrupt_path)
            check(False, "Should have raised JSONDecodeError")
        except json.JSONDecodeError:
            check(True, "Corrupt file raises JSONDecodeError")

        # =================================================================
        print("\n" + "=" * 72)
        print("  4. Load Future Version Raises ValueError")
        print("=" * 72)

        future_path = os.path.join(tmpdir, "future.json")
        with open(future_path, "w") as f:
            json.dump({
                "_version": 999,
                "items": {},
                "edges": {},
                "aliases": {},
            }, f)

        bus5 = LiteMemoryBus()
        try:
            bus5.load_snapshot(future_path)
            check(False, "Should have raised ValueError")
        except ValueError as e:
            check("999" in str(e), f"Future version raises ValueError: {e}")

        # =================================================================
        print("\n" + "=" * 72)
        print("  5. Secondary Indices Rebuilt on Load")
        print("=" * 72)

        bus6 = LiteMemoryBus()
        await bus6.initialize()
        await bus6.store(MemoryItem(
            content="Fact A", memory_type="factual", importance=0.5,
        ))
        await bus6.store(MemoryItem(
            content="Episode B", memory_type="episodic", importance=0.5,
        ))
        entity_item = MemoryItem(
            content="EntityC", memory_type="entity", importance=0.5,
            properties={"type": "concept"},
        )
        await bus6.store(entity_item)

        idx_path = os.path.join(tmpdir, "indices.json")
        bus6.save_snapshot(idx_path)

        bus7 = LiteMemoryBus()
        bus7.load_snapshot(idx_path)

        check(len(bus7._by_type["factual"]) >= 1, "factual index rebuilt")
        check(len(bus7._by_type["episodic"]) >= 1, "episodic index rebuilt")
        check(len(bus7._by_type["entity"]) >= 1, "entity index rebuilt")
        check(len(bus7._by_entity_type.get("concept", [])) >= 1,
              "entity_type index rebuilt")

        # =================================================================
        print("\n" + "=" * 72)
        print("  6. Bridge Skips Warm-Up When Snapshot Loaded")
        print("=" * 72)

        bridge_path = os.path.join(tmpdir, "bridge_snap.json")
        mem = UnifiedMemory(
            user_id="test",
            enable_mem0=False, enable_neo4j=False, enable_qdrant=False,
        )
        b1 = CognitiveBridge(mem, persist_path=bridge_path)
        await b1.initialize()
        await b1.store_and_index("PPR uses personalized PageRank for navigation")
        await b1.store_and_index("TMS tracks belief justifications")

        node_count_after_store = len(b1.bus._items)
        check(node_count_after_store > 0, f"Bridge has {node_count_after_store} nodes")
        check(os.path.exists(bridge_path), "Snapshot file created")

        b2 = CognitiveBridge(mem, persist_path=bridge_path)
        await b2.initialize()
        check(b2._initialized, "Second bridge initialized")
        check(len(b2.bus._items) == node_count_after_store,
              f"Loaded {len(b2.bus._items)} nodes (expected {node_count_after_store})")

        # =================================================================
        print("\n" + "=" * 72)
        print("  7. Bridge Falls Back to Warm-Up on Corrupt File")
        print("=" * 72)

        corrupt_bridge_path = os.path.join(tmpdir, "corrupt_bridge.json")
        with open(corrupt_bridge_path, "w") as f:
            f.write("CORRUPT DATA")

        b3 = CognitiveBridge(mem, persist_path=corrupt_bridge_path)
        await b3.initialize()
        check(b3._initialized, "Bridge initialized despite corrupt snapshot")
        check(True, "Warm-up fallback succeeded (no crash)")

        # =================================================================
        print("\n" + "=" * 72)
        print("  8. persist_path=None Preserves Existing Behavior")
        print("=" * 72)

        b4 = CognitiveBridge(mem, persist_path=None)
        await b4.initialize()
        check(b4._initialized, "Bridge without persist_path initializes")
        check(b4.persist_path is None, "persist_path is None")
        await b4.store_and_index("Test memory without persistence")
        check(True, "store_and_index without persist_path works")

        # =================================================================
        print("\n" + "=" * 72)
        print("  9. Incremental Save After Each store_and_index")
        print("=" * 72)

        incr_path = os.path.join(tmpdir, "incremental.json")
        b5 = CognitiveBridge(mem, persist_path=incr_path)
        await b5.initialize()

        await b5.store_and_index("First memory")
        with open(incr_path, "r") as f:
            snap1 = json.load(f)
        count1 = len(snap1["graph"]["items"])

        await b5.store_and_index("Second memory")
        with open(incr_path, "r") as f:
            snap2 = json.load(f)
        count2 = len(snap2["graph"]["items"])

        check(count2 > count1,
              f"Incremental save grows: {count1} -> {count2}")

        # =================================================================
        print("\n" + "=" * 72)
        print("  10. Atomic Write Leaves No .tmp File")
        print("=" * 72)

        atomic_path = os.path.join(tmpdir, "atomic.json")
        bus_a = LiteMemoryBus()
        await bus_a.initialize()
        await bus_a.store(MemoryItem(
            content="Atomic test", memory_type="factual", importance=0.5,
        ))
        bus_a.save_snapshot(atomic_path)

        check(os.path.exists(atomic_path), "Snapshot file exists")
        tmp_path = atomic_path.replace(".json", ".tmp")
        check(not os.path.exists(tmp_path), "No .tmp file left behind")

        # =================================================================
        print("\n" + "=" * 72)
        print("  11. StoredItem.to_snapshot Preserves All Fields")
        print("=" * 72)

        item = StoredItem(
            id="test-id-123",
            content="Test content",
            memory_type="factual",
            entity_type="concept",
            entity_ids=["ent1", "ent2"],
            properties={"key": "value", "nested": {"a": 1}},
            importance=0.75,
            timestamp="2026-02-14T12:00:00",
            access_count=5,
            last_accessed="2026-02-14T13:00:00",
            ttl_seconds=3600.0,
        )
        snap = item.to_snapshot()
        restored = StoredItem.from_snapshot(snap)

        check(restored.id == item.id, "id preserved")
        check(restored.content == item.content, "content preserved")
        check(restored.memory_type == item.memory_type, "memory_type preserved")
        check(restored.entity_type == item.entity_type, "entity_type preserved")
        check(restored.entity_ids == item.entity_ids, "entity_ids preserved")
        check(restored.properties == item.properties, "properties preserved")
        check(restored.importance == item.importance, "importance preserved")
        check(restored.timestamp == item.timestamp, "timestamp preserved")
        check(restored.access_count == item.access_count, "access_count preserved")
        check(restored.last_accessed == item.last_accessed, "last_accessed preserved")
        check(restored.ttl_seconds == item.ttl_seconds, "ttl_seconds preserved")

        # =================================================================
        print("\n" + "=" * 72)
        print("  12. Edges Survive JSON Round-Trip (tuple <-> list)")
        print("=" * 72)

        bus_e = LiteMemoryBus()
        await bus_e.initialize()
        e1 = await bus_e.store(MemoryItem(content="Node A", memory_type="factual"))
        e2 = await bus_e.store(MemoryItem(content="Node B", memory_type="factual"))
        e3 = await bus_e.store(MemoryItem(content="Node C", memory_type="factual"))
        await bus_e.store_edge(e1, e2, "RELATED")
        await bus_e.store_edge(e1, e3, "MENTIONS")

        edge_path = os.path.join(tmpdir, "edges.json")
        bus_e.save_snapshot(edge_path)

        bus_e2 = LiteMemoryBus()
        bus_e2.load_snapshot(edge_path)

        edges_e1 = bus_e2._edges.get(e1, [])
        check(len(edges_e1) == 2, f"Node A has 2 outgoing edges ({len(edges_e1)})")
        if edges_e1:
            check(isinstance(edges_e1[0], tuple),
                  f"Edge is tuple after load (got {type(edges_e1[0]).__name__})")
            rels = {rel for _, rel in edges_e1}
            check("RELATED" in rels, "RELATED edge survived")
            check("MENTIONS" in rels, "MENTIONS edge survived")

        # =================================================================
        print("\n" + "=" * 72)
        print("  13. CBR Case Library Survives Round-Trip")
        print("=" * 72)

        cbr = CBRSystem()
        # Manually inject cases
        case1 = Case(
            case_id="abc123",
            query="Thompson Sampling vs epsilon-greedy",
            features={"query_length": 0.4, "seed_count": 0.6, "ppr_entropy": 0.3},
            preset_id="balanced",
            reward=0.85,
            response_summary="Thompson Sampling uses posterior distributions...",
            shell_count=2,
            node_id="case_abc123",
            retrieval_count=3,
            avg_reuse_reward=0.72,
        )
        case2 = Case(
            case_id="def456",
            query="What is PPR?",
            features={"query_length": 0.2, "seed_count": 0.3},
            preset_id="focused",
            reward=0.92,
            response_summary="Personalized PageRank...",
            shell_count=1,
        )
        cbr._cases = [case1, case2]

        snap = cbr.to_snapshot()
        check(len(snap["cases"]) == 2, "Snapshot has 2 cases")

        cbr2 = CBRSystem()
        cbr2.from_snapshot(snap)
        check(cbr2.case_count == 2, "Restored 2 cases")

        rc1 = cbr2._cases[0]
        check(rc1.case_id == "abc123", "case_id preserved")
        check(rc1.query == "Thompson Sampling vs epsilon-greedy", "full query preserved")
        check(rc1.features == case1.features, "features preserved")
        check(rc1.reward == 0.85, "reward preserved")
        check(rc1.response_summary == "Thompson Sampling uses posterior distributions...",
              "response_summary preserved")
        check(rc1.node_id == "case_abc123", "node_id preserved")
        check(rc1.retrieval_count == 3, "retrieval_count preserved")
        check(rc1.avg_reuse_reward == 0.72, "avg_reuse_reward preserved")

        # =================================================================
        print("\n" + "=" * 72)
        print("  14. ChunkCompiler Chunks + Experience Survive Round-Trip")
        print("=" * 72)

        cc = ChunkCompiler()
        chunk = CompiledChunk(
            chunk_id="chunk001",
            trigger={"query_length": (0.1, 0.5), "seed_count": (0.2, 0.8)},
            action=ChunkAction(preset_id="focused", max_shells=2, skip_verify=True),
            node_id="chunk_chunk001",
            successes=10,
            failures=2,
            consecutive_failures=0,
        )
        cc._chunks = [chunk]
        cc._experience = [
            ExperienceRecord(
                features={"query_length": 0.3, "seed_count": 0.5},
                preset_id="focused",
                reward=0.8,
                shell_count=1,
                query_id="q_001",
            ),
            ExperienceRecord(
                features={"query_length": 0.4, "seed_count": 0.6},
                preset_id="wide",
                reward=0.65,
                shell_count=2,
                query_id="q_002",
            ),
        ]

        snap = cc.to_snapshot()
        check(len(snap["chunks"]) == 1, "Snapshot has 1 chunk")
        check(len(snap["experience"]) == 2, "Snapshot has 2 experiences")

        cc2 = ChunkCompiler()
        cc2.from_snapshot(snap)
        check(len(cc2._chunks) == 1, "Restored 1 chunk")
        check(len(cc2._experience) == 2, "Restored 2 experiences")

        rc = cc2._chunks[0]
        check(rc.chunk_id == "chunk001", "chunk_id preserved")
        check(rc.trigger["query_length"] == (0.1, 0.5), "trigger ranges preserved as tuples")
        check(rc.action.preset_id == "focused", "action.preset_id preserved")
        check(rc.action.max_shells == 2, "action.max_shells preserved")
        check(rc.action.skip_verify is True, "action.skip_verify preserved")
        check(rc.successes == 10, "successes preserved")
        check(rc.failures == 2, "failures preserved")
        check(rc.consecutive_failures == 0, "consecutive_failures preserved")
        check(rc.node_id == "chunk_chunk001", "node_id preserved")

        re = cc2._experience[0]
        check(re.preset_id == "focused", "experience preset_id preserved")
        check(re.reward == 0.8, "experience reward preserved")
        check(re.query_id == "q_001", "experience query_id preserved")

        # =================================================================
        print("\n" + "=" * 72)
        print("  15. Combined v2 Snapshot Contains All Sections")
        print("=" * 72)

        combined_path = os.path.join(tmpdir, "combined.json")
        b6 = CognitiveBridge(mem, persist_path=combined_path)
        await b6.initialize()
        await b6.store_and_index("Memory for combined test")

        with open(combined_path, "r") as f:
            combined = json.load(f)

        check(combined.get("_version") == 2, "v2 snapshot version")
        check("graph" in combined, "graph section present")
        check("cbr" in combined, "cbr section present")
        check("chunks" in combined, "chunks section present")
        check("items" in combined["graph"], "graph has items")
        check("edges" in combined["graph"], "graph has edges")
        check("cases" in combined["cbr"], "cbr has cases list")
        check("chunks" in combined["chunks"], "chunks has chunks list")
        check("experience" in combined["chunks"], "chunks has experience list")

        # Load into fresh bridge and verify all sections restored
        b7 = CognitiveBridge(mem, persist_path=combined_path)
        await b7.initialize()
        check(len(b7.bus._items) == len(b6.bus._items),
              f"Graph nodes restored ({len(b7.bus._items)})")

        # =================================================================
        print("\n" + "=" * 72)
        print("  16. v1 Snapshot Backward Compatibility")
        print("=" * 72)

        # Create a v1-style snapshot (graph-only, no "graph" wrapper)
        v1_path = os.path.join(tmpdir, "v1_compat.json")
        bus_v1 = LiteMemoryBus()
        await bus_v1.initialize()
        await bus_v1.store(MemoryItem(
            content="v1 memory", memory_type="factual", importance=0.7,
        ))
        bus_v1.save_snapshot(v1_path)

        # Bridge should load v1 snapshots gracefully
        b8 = CognitiveBridge(mem, persist_path=v1_path)
        await b8.initialize()
        check(b8._initialized, "Bridge loads v1 snapshot")
        check(len(b8.bus._items) == 1, f"v1 graph nodes loaded ({len(b8.bus._items)})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
