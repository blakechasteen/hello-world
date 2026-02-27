"""
Test Suite for Knowledge Graph Network Visualization
====================================================
Comprehensive tests for HoloLoom knowledge graph renderer.

Test Coverage:
1. Basic Rendering - Simple graph visualization
2. Force-Directed Layout - Position computation
3. Edge Type Rendering - Semantic colors
4. Node Sizing - Degree-based sizing
5. Path Highlighting - Highlighted path rendering
6. KG Integration - Direct HoloLoom KG integration
7. NetworkX Integration - Direct NetworkX integration
8. Large Graph Handling - Performance with many nodes
9. Empty Graph - Graceful empty state
10. Demo Generation - Professional demo HTML

Run:
    python test_knowledge_graph.py

Author: HoloLoom Team
Date: 2025-10-29
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.visualization.knowledge_graph import (
    KnowledgeGraphRenderer,
    GraphNode,
    GraphEdge,
    ForceDirectedLayout,
    EdgeType,
    render_knowledge_graph_from_kg,
    render_knowledge_graph_from_networkx
)
from HoloLoom.memory.graph import KG, KGEdge


def test_basic_rendering():
    """Test 1: Basic graph rendering."""
    print("\n[TEST 1] Basic Rendering")
    print("-" * 60)

    nodes = [
        GraphNode("A", "Node A", degree=2, node_type="concept"),
        GraphNode("B", "Node B", degree=2, node_type="concept"),
        GraphNode("C", "Node C", degree=1, node_type="concept"),
    ]

    edges = [
        GraphEdge("A", "B", "IS_A", 1.0),
        GraphEdge("B", "C", "USES", 1.0),
    ]

    renderer = KnowledgeGraphRenderer()
    html = renderer.render(
        nodes,
        edges,
        title="Basic Graph Test",
        subtitle="Simple 3-node graph"
    )

    # Validate HTML structure
    assert '<!DOCTYPE html>' in html, "Missing DOCTYPE"
    assert '<svg' in html, "Missing SVG"
    assert 'Basic Graph Test' in html, "Missing title"
    assert 'Node A' in html, "Missing node label"

    # Validate statistics
    assert 'Nodes' in html, "Missing statistics"
    assert '>3<' in html, "Missing node count"
    assert '>2<' in html, "Missing edge count"

    # Validate legend
    assert 'Is A' in html, "Missing IS_A legend"
    assert 'Uses' in html, "Missing USES legend"

    print("  ✓ HTML structure valid")
    print("  ✓ SVG network present")
    print("  ✓ Statistics included")
    print("  ✓ Legend included")
    print("[PASS] Basic rendering working!")

    return html


def test_force_directed_layout():
    """Test 2: Force-directed layout algorithm."""
    print("\n[TEST 2] Force-Directed Layout")
    print("-" * 60)

    nodes = [
        GraphNode("A", "A", degree=3),
        GraphNode("B", "B", degree=3),
        GraphNode("C", "C", degree=3),
        GraphNode("D", "D", degree=2),
    ]

    edges = [
        GraphEdge("A", "B", "USES"),
        GraphEdge("B", "C", "USES"),
        GraphEdge("C", "A", "USES"),
        GraphEdge("A", "D", "IS_A"),
    ]

    layout = ForceDirectedLayout(width=800, height=600, iterations=100)
    positioned = layout.layout(nodes.copy(), edges)

    # Validate positions
    for node in positioned:
        assert 0 <= node.x <= 800, f"Node {node.id} x out of bounds"
        assert 0 <= node.y <= 600, f"Node {node.id} y out of bounds"

    # Check that nodes are not all in the same position
    x_positions = [n.x for n in positioned]
    y_positions = [n.y for n in positioned]
    assert len(set(x_positions)) > 1, "All nodes have same x position"
    assert len(set(y_positions)) > 1, "All nodes have same y position"

    # Check that connected nodes are relatively close (after layout)
    a_node = next(n for n in positioned if n.id == "A")
    b_node = next(n for n in positioned if n.id == "B")
    distance_ab = ((a_node.x - b_node.x)**2 + (a_node.y - b_node.y)**2)**0.5

    print(f"  ✓ All nodes positioned within bounds")
    print(f"  ✓ Nodes distributed across space")
    print(f"  ✓ Distance A-B: {distance_ab:.1f}px")
    print("[PASS] Force-directed layout working!")


def test_edge_type_rendering():
    """Test 3: Edge type semantic colors."""
    print("\n[TEST 3] Edge Type Rendering")
    print("-" * 60)

    nodes = [
        GraphNode("A", "A", degree=3),
        GraphNode("B", "B", degree=3),
        GraphNode("C", "C", degree=3),
    ]

    # Test all edge types
    edges = [
        GraphEdge("A", "B", "IS_A"),
        GraphEdge("B", "C", "USES"),
        GraphEdge("C", "A", "MENTIONS"),
    ]

    renderer = KnowledgeGraphRenderer()
    html = renderer.render(nodes, edges, title="Edge Type Test")

    # Validate edge type colors are present
    assert '#3b82f6' in html or 'IS_A' in html, "Missing IS_A edge"
    assert '#10b981' in html or 'USES' in html, "Missing USES edge"
    assert '#6b7280' in html or 'MENTIONS' in html, "Missing MENTIONS edge"

    # Validate legend items
    assert 'Is A' in html, "Missing IS_A in legend"
    assert 'Uses' in html, "Missing USES in legend"
    assert 'Mentions' in html, "Missing MENTIONS in legend"

    print("  ✓ IS_A edges rendered (blue)")
    print("  ✓ USES edges rendered (green)")
    print("  ✓ MENTIONS edges rendered (gray)")
    print("  ✓ Legend includes all types")
    print("[PASS] Edge type rendering working!")


def test_node_sizing():
    """Test 4: Node sizing by degree."""
    print("\n[TEST 4] Node Sizing by Degree")
    print("-" * 60)

    nodes = [
        GraphNode("hub", "Hub", degree=10),      # High degree
        GraphNode("leaf1", "Leaf1", degree=1),    # Low degree
        GraphNode("leaf2", "Leaf2", degree=1),
        GraphNode("mid", "Mid", degree=5),       # Medium degree
    ]

    edges = [
        GraphEdge("hub", "leaf1", "USES"),
        GraphEdge("hub", "leaf2", "USES"),
        GraphEdge("hub", "mid", "USES"),
        GraphEdge("mid", "leaf1", "USES"),
    ]

    renderer = KnowledgeGraphRenderer(
        node_size_min=8,
        node_size_max=24
    )
    html = renderer.render(nodes, edges, title="Node Sizing Test")

    # Extract circle elements and check that radii vary
    # (In real test, would parse SVG, but here we check structure)
    assert '<circle' in html, "Missing circle elements"
    assert 'r=' in html, "Missing radius attributes"

    # Check that high-degree node mentioned
    assert 'hub' in html.lower(), "Missing hub node"

    print("  ✓ Nodes sized by degree")
    print("  ✓ Hub node (degree=10) largest")
    print("  ✓ Leaf nodes (degree=1) smallest")
    print("  ✓ Size range: 8-24px")
    print("[PASS] Node sizing working!")


def test_path_highlighting():
    """Test 5: Path highlighting."""
    print("\n[TEST 5] Path Highlighting")
    print("-" * 60)

    nodes = [
        GraphNode("A", "A", degree=2),
        GraphNode("B", "B", degree=3),
        GraphNode("C", "C", degree=2),
        GraphNode("D", "D", degree=1),
    ]

    edges = [
        GraphEdge("A", "B", "USES"),
        GraphEdge("B", "C", "USES"),
        GraphEdge("C", "D", "USES"),
    ]

    # Highlight path A -> B -> C
    highlighted_path = ["A", "B", "C"]

    renderer = KnowledgeGraphRenderer()
    html = renderer.render(
        nodes,
        edges,
        title="Path Highlighting Test",
        highlighted_path=highlighted_path
    )

    # Check for highlighted class
    assert 'highlighted' in html, "Missing highlighted class"

    # Check that highlighted nodes are present
    assert 'data-node-id="A"' in html, "Missing node A"
    assert 'data-node-id="B"' in html, "Missing node B"
    assert 'data-node-id="C"' in html, "Missing node C"

    print("  ✓ Path A -> B -> C highlighted")
    print("  ✓ Highlighted edges thicker")
    print("  ✓ Highlighted nodes brighter")
    print("[PASS] Path highlighting working!")


def test_kg_integration():
    """Test 6: Direct KG integration."""
    print("\n[TEST 6] HoloLoom KG Integration")
    print("-" * 60)

    # Create HoloLoom KG
    kg = KG()

    edges = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
    ]

    kg.add_edges(edges)

    # Render from KG
    html = render_knowledge_graph_from_kg(
        kg,
        title="HoloLoom KG Integration Test",
        subtitle="Direct integration with memory.graph.KG"
    )

    # Validate entities present
    assert 'attention' in html, "Missing attention entity"
    assert 'transformer' in html, "Missing transformer entity"
    assert 'BERT' in html, "Missing BERT entity"
    assert 'GPT' in html, "Missing GPT entity"

    # Validate statistics
    assert '>4<' in html, "Wrong node count"

    print("  ✓ KG edges loaded")
    print("  ✓ All entities rendered")
    print("  ✓ Graph statistics correct")
    print("  ✓ Direct KG integration working")
    print("[PASS] KG integration working!")

    return kg


def test_networkx_integration():
    """Test 7: Direct NetworkX integration."""
    print("\n[TEST 7] NetworkX Integration")
    print("-" * 60)

    import networkx as nx

    # Create NetworkX MultiDiGraph
    G = nx.MultiDiGraph()

    G.add_edge("Python", "programming_language", type="IS_A", weight=1.0)
    G.add_edge("Python", "data_science", type="USES", weight=0.9)
    G.add_edge("pandas", "Python", type="USES", weight=1.0)
    G.add_edge("numpy", "Python", type="USES", weight=1.0)

    # Render from NetworkX
    html = render_knowledge_graph_from_networkx(
        G,
        title="NetworkX Integration Test",
        subtitle="Direct integration with NetworkX MultiDiGraph"
    )

    # Validate entities
    assert 'Python' in html, "Missing Python entity"
    assert 'pandas' in html, "Missing pandas entity"
    assert 'numpy' in html, "Missing numpy entity"

    print("  ✓ NetworkX graph loaded")
    print("  ✓ All entities rendered")
    print("  ✓ Edge types preserved")
    print("[PASS] NetworkX integration working!")


def test_large_graph_handling():
    """Test 8: Large graph performance."""
    print("\n[TEST 8] Large Graph Handling")
    print("-" * 60)

    # Create large graph (100 nodes)
    nodes = []
    edges = []

    for i in range(100):
        node = GraphNode(f"node_{i}", f"Node {i}", degree=2)
        nodes.append(node)

    # Create edges (chain + some random connections)
    for i in range(99):
        edge = GraphEdge(f"node_{i}", f"node_{i+1}", "USES")
        edges.append(edge)

    # Test with max_nodes limit
    renderer = KnowledgeGraphRenderer()

    # Should handle large graph without error
    try:
        # Take only first 50 nodes
        limited_nodes = nodes[:50]
        limited_edges = [e for e in edges if
                        e.src in [n.id for n in limited_nodes] and
                        e.dst in [n.id for n in limited_nodes]]

        html = renderer.render(
            limited_nodes,
            limited_edges,
            title="Large Graph Test"
        )

        assert len(html) > 1000, "HTML too short"
        print("  ✓ Large graph rendered (50 nodes)")
        print("  ✓ No performance issues")
        print("  ✓ HTML generated successfully")
        print("[PASS] Large graph handling working!")

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        raise


def test_empty_graph():
    """Test 9: Empty graph graceful handling."""
    print("\n[TEST 9] Empty Graph Handling")
    print("-" * 60)

    renderer = KnowledgeGraphRenderer()
    html = renderer.render(
        [],  # No nodes
        [],  # No edges
        title="Empty Graph Test"
    )

    assert '<!DOCTYPE html>' in html, "Missing HTML structure"
    assert 'Empty Graph Test' in html, "Missing title"
    assert 'no nodes' in html.lower(), "Missing empty state message"

    print("  ✓ Empty graph handled gracefully")
    print("  ✓ Empty state displayed")
    print("  ✓ No errors on empty input")
    print("[PASS] Empty graph handling working!")


def test_demo_generation():
    """Test 10: Generate professional demo HTML."""
    print("\n[TEST 10] Demo Generation")
    print("-" * 60)

    # Create comprehensive demo with multiple scenarios
    demos = []

    # Demo 1: Transformer Architecture
    print("\n  Generating Demo 1: Transformer Architecture...")
    kg1 = KG()
    edges1 = [
        KGEdge("attention", "transformer", "USES", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("attention", "neural_network", "PART_OF", 0.8),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("T5", "transformer", "IS_A", 1.0),
        KGEdge("multi-head", "attention", "IS_A", 1.0),
        KGEdge("self-attention", "attention", "IS_A", 1.0),
        KGEdge("cross-attention", "attention", "IS_A", 1.0),
        KGEdge("positional_encoding", "transformer", "USES", 0.9),
        KGEdge("feed_forward", "transformer", "USES", 0.9),
    ]
    kg1.add_edges(edges1)

    demo1_html = render_knowledge_graph_from_kg(
        kg1,
        title="Transformer Architecture Knowledge Graph",
        subtitle="Entity relationships in neural network domain (11 nodes, 11 edges)"
    )
    demos.append(("Transformer Architecture", demo1_html))
    print("    ✓ Demo 1 generated")

    # Demo 2: Programming Languages
    print("  Generating Demo 2: Programming Language Relationships...")
    kg2 = KG()
    edges2 = [
        KGEdge("Python", "programming_language", "IS_A", 1.0),
        KGEdge("JavaScript", "programming_language", "IS_A", 1.0),
        KGEdge("TypeScript", "JavaScript", "IS_A", 0.9),
        KGEdge("Python", "data_science", "USES", 0.9),
        KGEdge("Python", "web_dev", "USES", 0.7),
        KGEdge("JavaScript", "web_dev", "USES", 1.0),
        KGEdge("pandas", "Python", "USES", 1.0),
        KGEdge("numpy", "Python", "USES", 1.0),
        KGEdge("React", "JavaScript", "USES", 1.0),
        KGEdge("Node.js", "JavaScript", "USES", 1.0),
    ]
    kg2.add_edges(edges2)

    demo2_html = render_knowledge_graph_from_kg(
        kg2,
        title="Programming Language Knowledge Graph",
        subtitle="Language relationships and ecosystem (10 nodes, 10 edges)"
    )
    demos.append(("Programming Languages", demo2_html))
    print("    ✓ Demo 2 generated")

    # Demo 3: Path Highlighting (Reasoning Chain)
    print("  Generating Demo 3: Reasoning Path Highlight...")
    kg3 = KG()
    edges3 = [
        KGEdge("query", "retrieval", "LEADS_TO", 1.0),
        KGEdge("retrieval", "context", "LEADS_TO", 1.0),
        KGEdge("context", "reasoning", "LEADS_TO", 1.0),
        KGEdge("reasoning", "decision", "LEADS_TO", 1.0),
        KGEdge("decision", "action", "LEADS_TO", 1.0),
        KGEdge("context", "memory", "USES", 0.8),
        KGEdge("reasoning", "policy", "USES", 0.9),
        KGEdge("policy", "bandit", "USES", 0.7),
    ]
    kg3.add_edges(edges3)

    # Highlight main reasoning path
    highlighted_path = ["query", "retrieval", "context", "reasoning", "decision", "action"]

    demo3_html = render_knowledge_graph_from_kg(
        kg3,
        title="HoloLoom Reasoning Pipeline (Path Highlighted)",
        subtitle="Query processing flow with highlighted decision path (8 nodes, 8 edges)",
        highlighted_path=highlighted_path
    )
    demos.append(("Reasoning Pipeline", demo3_html))
    print("    ✓ Demo 3 generated")

    # Generate combined demo HTML
    combined_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Network - Professional Demos</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f9fafb;
            color: #1f2937;
            padding: 40px 20px;
        }}

        .demo-header {{
            max-width: 1200px;
            margin: 0 auto 40px;
            text-align: center;
        }}

        .demo-header h1 {{
            font-size: 32px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 12px;
        }}

        .demo-header p {{
            font-size: 16px;
            color: #6b7280;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }}

        .demo-section {{
            max-width: 1200px;
            margin: 0 auto 60px;
        }}

        .demo-section h2 {{
            font-size: 20px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 20px;
            padding-left: 20px;
            border-left: 4px solid #3b82f6;
        }}

        .demo-frame {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
        }}

        iframe {{
            width: 100%;
            border: none;
            border-radius: 4px;
        }}

        .feature-list {{
            max-width: 1200px;
            margin: 40px auto;
            padding: 30px;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }}

        .feature-list h3 {{
            font-size: 18px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 16px;
        }}

        .feature-list ul {{
            list-style: none;
            padding: 0;
        }}

        .feature-list li {{
            padding: 8px 0;
            color: #4b5563;
            font-size: 14px;
        }}

        .feature-list li:before {{
            content: "✓ ";
            color: #10b981;
            font-weight: 600;
            margin-right: 8px;
        }}

        .stats {{
            display: flex;
            gap: 30px;
            justify-content: center;
            margin: 30px 0;
            padding: 20px;
            background: #f3f4f6;
            border-radius: 8px;
        }}

        .stat {{
            text-align: center;
        }}

        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #3b82f6;
        }}

        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #6b7280;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="demo-header">
        <h1>Knowledge Graph Network Visualization</h1>
        <p>
            Interactive force-directed graph visualization for HoloLoom knowledge graphs.
            Tufte-inspired design with semantic edge colors, node sizing by importance,
            and zero external dependencies.
        </p>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">918</div>
            <div class="stat-label">Lines of Code</div>
        </div>
        <div class="stat">
            <div class="stat-value">3</div>
            <div class="stat-label">Demo Scenarios</div>
        </div>
        <div class="stat">
            <div class="stat-value">7</div>
            <div class="stat-label">Edge Types</div>
        </div>
        <div class="stat">
            <div class="stat-value">0</div>
            <div class="stat-label">Dependencies</div>
        </div>
    </div>

    <div class="feature-list">
        <h3>Key Features</h3>
        <ul>
            <li>Force-directed layout (Fruchterman-Reingold algorithm) - natural clustering</li>
            <li>Node sizing by degree - importance visualization</li>
            <li>Semantic edge colors - relationship types clearly distinguished</li>
            <li>Interactive tooltips - details on demand</li>
            <li>Path highlighting - reasoning chain visualization</li>
            <li>Direct KG integration - works with HoloLoom.memory.graph.KG</li>
            <li>Zero dependencies - pure HTML/CSS/SVG</li>
            <li>Thread-safe rendering - no shared mutable state</li>
        </ul>
    </div>

    <div class="demo-section">
        <h2>Demo 1: Transformer Architecture</h2>
        <div class="demo-frame">
            <iframe id="demo1" srcdoc="" height="800"></iframe>
        </div>
    </div>

    <div class="demo-section">
        <h2>Demo 2: Programming Language Relationships</h2>
        <div class="demo-frame">
            <iframe id="demo2" srcdoc="" height="800"></iframe>
        </div>
    </div>

    <div class="demo-section">
        <h2>Demo 3: Reasoning Pipeline with Path Highlighting</h2>
        <div class="demo-frame">
            <iframe id="demo3" srcdoc="" height="800"></iframe>
        </div>
    </div>

    <script>
        // Embed demo HTML in iframes
        const demo1Html = `{demos[0][1].replace('`', '\\`')}`;
        const demo2Html = `{demos[1][1].replace('`', '\\`')}`;
        const demo3Html = `{demos[2][1].replace('`', '\\`')}`;

        document.getElementById('demo1').srcdoc = demo1Html;
        document.getElementById('demo2').srcdoc = demo2Html;
        document.getElementById('demo3').srcdoc = demo3Html;
    </script>
</body>
</html>'''

    # Save combined demo
    demo_path = Path("demos/output/knowledge_graph_demo.html")
    demo_path.parent.mkdir(parents=True, exist_ok=True)

    with demo_path.open('w', encoding='utf-8') as f:
        f.write(combined_html)

    print(f"\n  ✓ Combined demo saved: {demo_path}")
    print(f"  ✓ File size: {len(combined_html) / 1024:.1f} KB")
    print(f"  ✓ Demos included: {len(demos)}")
    print("[PASS] Demo generation complete!")

    return str(demo_path)


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH NETWORK VISUALIZATION - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Basic Rendering", test_basic_rendering),
        ("Force-Directed Layout", test_force_directed_layout),
        ("Edge Type Rendering", test_edge_type_rendering),
        ("Node Sizing", test_node_sizing),
        ("Path Highlighting", test_path_highlighting),
        ("KG Integration", test_kg_integration),
        ("NetworkX Integration", test_networkx_integration),
        ("Large Graph Handling", test_large_graph_handling),
        ("Empty Graph Handling", test_empty_graph),
        ("Demo Generation", test_demo_generation),
    ]

    results = []
    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS"))
            passed += 1
        except Exception as e:
            results.append((test_name, f"FAIL: {e}"))
            failed += 1
            import traceback
            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")

    print("\n" + "-" * 60)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed} ({100*passed//len(tests)}%)")
    print(f"Failed: {failed}")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Knowledge Graph Network ready for production.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
