"""
Graph Visualization Export Tests for LiteMemoryBus.

12 pytest tests covering:
  1. to_dot produces valid DOT syntax (starts with "digraph", ends with "}")
  2. Node labels are truncated at max_label_len
  3. Entity nodes get diamond shape
  4. Chunk nodes get hexagon shape
  5. Edge labels include relationship type
  6. Highlighted nodes get special styling
  7. Edges are deduplicated (each pair appears once)
  8. to_graph_dict has nodes, edges, aliases keys
  9. to_graph_dict edges are deduplicated
  10. subgraph returns only nearby nodes (BFS depth)
  11. subgraph respects max_nodes limit
  12. Empty graph produces valid DOT
"""

import asyncio
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from hololoom.memory.lite_bus import LiteMemoryBus, StoredItem
from hololoom.memory.bus import MemoryItem


@pytest.fixture
def bus():
    return LiteMemoryBus()


@pytest.fixture
def populated_bus():
    """Bus with a small graph: 4 nodes, 3 edges, 1 alias."""
    bus = LiteMemoryBus()
    loop = asyncio.new_event_loop()

    async def _populate():
        await bus.initialize()
        id_fact = await bus.store(MemoryItem(
            content="Thompson Sampling uses posterior distributions for exploration",
            memory_type="factual", importance=0.8,
        ))
        id_entity = await bus.store(MemoryItem(
            content="Thompson",
            memory_type="entity", importance=0.6,
        ))
        id_chunk = await bus.store(MemoryItem(
            content="Compiled knowledge chunk about bandits",
            memory_type="chunk", importance=0.7,
        ))
        id_case = await bus.store(MemoryItem(
            content="Case: bandit selection under uncertainty",
            memory_type="case", importance=0.9,
        ))
        await bus.store_edge(id_fact, id_entity, "MENTIONS")
        await bus.store_edge(id_fact, id_chunk, "COMPILED_FROM")
        await bus.store_edge(id_case, id_entity, "SIMILAR_CASE")
        return id_fact, id_entity, id_chunk, id_case

    ids = loop.run_until_complete(_populate())
    loop.close()
    bus._test_ids = ids
    return bus


def test_dot_valid_syntax(populated_bus):
    """1. to_dot produces valid DOT syntax (starts with 'digraph', ends with '}')."""
    dot = populated_bus.to_dot()
    assert dot.startswith("digraph"), "DOT should start with 'digraph'"
    assert dot.rstrip().endswith("}"), "DOT should end with '}'"
    assert "rankdir=LR" in dot
    assert 'fontname="Helvetica"' in dot


def test_node_labels_truncated(populated_bus):
    """2. Node labels are truncated at max_label_len."""
    dot = populated_bus.to_dot(max_label_len=20)
    assert "..." in dot, "Long labels should be truncated with '...'"
    id_fact = populated_bus._test_ids[0]
    item = populated_bus._items[id_fact]
    full_content = item.content
    assert full_content not in dot, "Full content should not appear when truncated"


def test_entity_diamond_shape(populated_bus):
    """3. Entity nodes get diamond shape."""
    dot = populated_bus.to_dot()
    id_entity = populated_bus._test_ids[1]
    pattern = re.compile(rf'"{re.escape(id_entity)}"[^;]*shape=diamond', re.DOTALL)
    assert pattern.search(dot), "Entity node should have shape=diamond"


def test_chunk_hexagon_shape(populated_bus):
    """4. Chunk nodes get hexagon shape."""
    dot = populated_bus.to_dot()
    id_chunk = populated_bus._test_ids[2]
    pattern = re.compile(rf'"{re.escape(id_chunk)}"[^;]*shape=hexagon', re.DOTALL)
    assert pattern.search(dot), "Chunk node should have shape=hexagon"


def test_edge_labels_include_rel_type(populated_bus):
    """5. Edge labels include relationship type."""
    dot = populated_bus.to_dot()
    assert 'label="MENTIONS"' in dot, "MENTIONS edge label should appear"
    assert 'label="COMPILED_FROM"' in dot, "COMPILED_FROM edge label should appear"
    assert 'label="SIMILAR_CASE"' in dot, "SIMILAR_CASE edge label should appear"


def test_highlighted_nodes_styling(populated_bus):
    """6. Highlighted nodes get bold red border."""
    id_fact, id_entity = populated_bus._test_ids[0], populated_bus._test_ids[1]
    dot = populated_bus.to_dot(highlight_ids={id_fact, id_entity})
    for nid in [id_fact, id_entity]:
        pattern = re.compile(rf'"{re.escape(nid)}"[^;]*penwidth=3[^;]*color=red', re.DOTALL)
        assert pattern.search(dot), f"Highlighted node {nid} should have penwidth=3, color=red"


def test_edges_deduplicated(populated_bus):
    """7. Edges are deduplicated (each pair appears once in DOT)."""
    dot = populated_bus.to_dot()
    arrow_lines = [line for line in dot.split("\n") if "->" in line]
    assert len(arrow_lines) == 3, (
        f"Expected 3 deduplicated edges, got {len(arrow_lines)}: {arrow_lines}"
    )


def test_graph_dict_keys(populated_bus):
    """8. to_graph_dict has nodes, edges, aliases keys."""
    d = populated_bus.to_graph_dict()
    assert "nodes" in d
    assert "edges" in d
    assert "aliases" in d
    assert isinstance(d["nodes"], list)
    assert isinstance(d["edges"], list)
    assert isinstance(d["aliases"], dict)
    assert len(d["nodes"]) == 4
    node_keys = set(d["nodes"][0].keys())
    assert {"id", "content", "memory_type", "importance", "entity_type", "access_count"} <= node_keys


def test_graph_dict_edges_deduplicated(populated_bus):
    """9. to_graph_dict edges are deduplicated."""
    d = populated_bus.to_graph_dict()
    assert len(d["edges"]) == 3, (
        f"Expected 3 deduplicated edges, got {len(d['edges'])}"
    )
    for edge in d["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge


def test_subgraph_bfs_depth(populated_bus):
    """10. subgraph returns only nearby nodes (BFS depth)."""
    id_fact = populated_bus._test_ids[0]
    sub = populated_bus.subgraph(id_fact, depth=1)
    assert id_fact in sub._items, "Center node must be in subgraph"
    assert len(sub._items) <= 4, "Depth-1 should not exceed total nodes"
    assert len(sub._items) >= 2, "Depth-1 from fact should include at least fact + one neighbor"

    id_entity = populated_bus._test_ids[1]
    sub_deep = populated_bus.subgraph(id_entity, depth=3)
    assert len(sub_deep._items) == 4, "Depth-3 from entity should reach all 4 nodes"


def test_subgraph_max_nodes(populated_bus):
    """11. subgraph respects max_nodes limit."""
    id_fact = populated_bus._test_ids[0]
    sub = populated_bus.subgraph(id_fact, depth=10, max_nodes=2)
    assert len(sub._items) <= 2, f"max_nodes=2 but got {len(sub._items)} nodes"
    assert id_fact in sub._items, "Center node must always be included"


def test_empty_graph_valid_dot(bus):
    """12. Empty graph produces valid DOT."""
    dot = bus.to_dot()
    assert dot.startswith("digraph")
    assert dot.rstrip().endswith("}")
    arrow_lines = [line for line in dot.split("\n") if "->" in line]
    assert len(arrow_lines) == 0, "Empty graph should have no edges"
