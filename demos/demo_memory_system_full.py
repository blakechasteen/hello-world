"""
HoloLoom Memory System - Full Demonstration
============================================

This demo exercises ALL memory system components:

1. AwarenessGraph - Living memory with semantic topology
2. Knowledge Graph (KG) - NetworkX-based graph storage
3. Memory Conductor - Multi-system coordination
4. Activation Field - Spreading activation
5. UnifiedMemory - User-facing API

Run with: PYTHONPATH=. python demos/demo_memory_system_full.py
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_status(success: bool, message: str):
    """Print status with checkmark or X."""
    mark = "[PASS]" if success else "[FAIL]"
    print(f"  {mark} {message}")


async def test_knowledge_graph():
    """Test Knowledge Graph (KG) - Core graph storage."""
    print_section("1. KNOWLEDGE GRAPH (KG)")

    try:
        from hololoom.memory.graph import KG, KGEdge

        # Create knowledge graph
        kg = KG()
        print_status(True, "KG instance created")

        # Add edges with different types
        edges = [
            KGEdge("Thompson Sampling", "Bayesian Methods", "IS_A", 0.9),
            KGEdge("Thompson Sampling", "Exploration", "USES", 0.85),
            KGEdge("Thompson Sampling", "Multi-Armed Bandits", "MENTIONS", 0.8),
            KGEdge("Exploration", "Exploitation", "LEADS_TO", 0.75),
            KGEdge("Bayesian Methods", "Probability Theory", "PART_OF", 0.9),
            KGEdge("UCB Algorithm", "Multi-Armed Bandits", "USES", 0.85),
            KGEdge("UCB Algorithm", "Exploration", "USES", 0.8),
        ]

        for edge in edges:
            kg.add_edge(edge)

        print_status(True, f"Added {len(edges)} edges to graph")

        # Check graph structure
        n_nodes = len(kg.G.nodes())
        n_edges = len(kg.G.edges())
        print_status(True, f"Graph has {n_nodes} nodes, {n_edges} edges")

        # Get neighbors (using direction parameter - no edge_types filter in this API)
        neighbors = kg.get_neighbors("Thompson Sampling", direction="out")
        print_status(True, f"Thompson Sampling out-neighbors: {neighbors}")

        # Extract subgraph
        subgraph = kg.subgraph_for_entities(["Thompson Sampling", "UCB Algorithm"])
        print_status(True, f"Subgraph extracted with {len(subgraph.nodes())} nodes")

        # Path finding
        paths = kg.get_paths("Thompson Sampling", "Probability Theory", max_length=3)
        print_status(True, f"Found {len(paths)} paths from Thompson Sampling to Probability Theory")

        return True, kg

    except Exception as e:
        print_status(False, f"Knowledge Graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_activation_field():
    """Test ActivationField - Spreading activation."""
    print_section("2. ACTIVATION FIELD")

    try:
        from hololoom.memory.activation_field import ActivationField
        import numpy as np

        # Create activation field
        field = ActivationField()
        print_status(True, "ActivationField instance created")

        # Add nodes to spatial index (simulating 228D semantic space)
        nodes = {
            "node_1": np.random.randn(228) * 0.1,
            "node_2": np.random.randn(228) * 0.1,
            "node_3": np.random.randn(228) * 0.1,
            "node_4": np.random.randn(228) * 0.1 + 0.5,  # Further away
            "node_5": np.random.randn(228) * 0.1 + 0.5,
        }

        for node_id, pos in nodes.items():
            field.update_spatial_index(node_id, pos)

        print_status(True, f"Added {len(nodes)} nodes to spatial index")

        # Activate region around query
        query_pos = np.zeros(228)  # Center of space
        activated = field.activate_region(
            center=query_pos,
            radius=2.0,  # Large radius to catch nodes
            node_ids=list(nodes.keys())
        )
        print_status(True, f"Activated {len(activated)} nodes")

        # Check activation levels
        for node_id in activated:
            level = field.levels.get(node_id, 0.0)
            print(f"      {node_id}: activation={level:.3f}")

        # Get stats
        stats = field.get_stats()
        print_status(True, f"Active nodes: {stats['n_active']}, density: {stats['density']:.2f}")

        return True

    except Exception as e:
        print_status(False, f"ActivationField test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_awareness_graph():
    """Test AwarenessGraph - Living memory with semantic topology."""
    print_section("3. AWARENESS GRAPH")

    try:
        import networkx as nx
        import numpy as np
        from hololoom.memory.awareness_graph import AwarenessGraph

        # Create a mock semantic calculus (simplified for demo)
        # Import real MatryoshkaScale for proper dict key matching
        from hololoom.semantic_calculus.matryoshka_streaming import MatryoshkaScale

        class MockSemanticCalculus:
            async def stream_analyze(self, word_stream):
                """Mock stream analysis returning snapshots."""
                words = []
                async for word in word_stream:
                    words.append(word)

                # Create a mock snapshot with proper attributes matching real interface
                class Snapshot:
                    # states_by_scale - Dict mapping scale to state dict
                    states_by_scale = {
                        MatryoshkaScale.PARAGRAPH: {
                            'position': np.random.randn(228) * 0.1,
                            'velocity': np.random.randn(228) * 0.01
                        }
                    }
                    # dominant dimensions by scale
                    dominant_dimensions_by_scale = {
                        MatryoshkaScale.PARAGRAPH: [0, 1, 2]
                    }
                    narrative_momentum = 0.8
                    complexity_index = 0.6

                yield Snapshot()

        # Create awareness graph
        graph_backend = nx.MultiDiGraph()
        semantic_calculus = MockSemanticCalculus()

        awareness = AwarenessGraph(
            graph_backend=graph_backend,
            semantic_calculus=semantic_calculus,
            vector_store=None
        )
        print_status(True, "AwarenessGraph instance created")

        # Perceive content
        perception = await awareness.perceive("Thompson Sampling balances exploration and exploitation")
        print_status(True, f"Perception: position shape={perception.position.shape}, momentum={perception.momentum}")

        # Remember content
        memory_id = await awareness.remember(
            "Thompson Sampling balances exploration and exploitation",
            perception,
            context={"topic": "machine_learning"}
        )
        print_status(True, f"Memory stored with ID: {memory_id[:20]}...")

        # Perceive another memory
        perception2 = await awareness.perceive("UCB provides optimism in the face of uncertainty")
        memory_id2 = await awareness.remember(
            "UCB provides optimism in the face of uncertainty",
            perception2,
            context={"topic": "machine_learning"}
        )
        print_status(True, f"Second memory stored: {memory_id2[:20]}...")

        # Get metrics
        metrics = awareness.get_metrics()
        print_status(True, f"Metrics: {metrics.n_memories} memories, {metrics.n_connections} connections")

        return True

    except Exception as e:
        print_status(False, f"AwarenessGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_conductor():
    """Test Memory Conductor - Multi-system coordination."""
    print_section("4. MEMORY CONDUCTOR")

    try:
        from hololoom.memory.symphony import (
            MemoryConductor,
            MemoryQuery,
            MemoryStrategy,
            create_memory_conductor
        )
        import networkx as nx

        # Create a mock memory backend with a graph
        class MockMemoryBackend:
            def __init__(self):
                self.graph = nx.MultiDiGraph()
                # Add some nodes
                self.graph.add_node("thompson_sampling", content="Thompson Sampling algorithm")
                self.graph.add_node("ucb", content="UCB algorithm")
                self.graph.add_node("exploration", content="Exploration strategy")
                self.graph.add_edge("thompson_sampling", "exploration")
                self.graph.add_edge("ucb", "exploration")

            async def recall(self, query_text: str = None, limit: int = 5, k: int = None, **kwargs):
                """Mock recall - accepts both limit and k for compatibility."""
                # Handle both parameter names
                actual_limit = k if k is not None else limit

                # Return mock memories
                class MockMemory:
                    def __init__(self, id, text, score):
                        self.id = id
                        self.text = text
                        self.content = text  # Alias for compatibility
                        self.relevance = score
                        self.score = score  # Alias for compatibility
                        self.context = {}
                        self.metadata = {}

                return [
                    MockMemory("mem_1", "Thompson Sampling balances exploration", 0.9),
                    MockMemory("mem_2", "UCB provides optimism", 0.85),
                ][:actual_limit]

        # Create conductor
        backend = MockMemoryBackend()
        conductor = create_memory_conductor(
            memory_backend=backend,
            strategy=MemoryStrategy.AUTO
        )
        print_status(True, "MemoryConductor created")

        # Create query
        query = MemoryQuery(
            text="What is Thompson Sampling?",
            k=5,
            strategy=MemoryStrategy.BALANCED
        )
        print_status(True, f"Query created: '{query.text[:40]}...'")

        # Execute query
        result = await conductor.recall(query)
        print_status(True, f"Recall returned {len(result.results)} results")
        print_status(True, f"Strategy used: {result.strategy_used.value}")
        print_status(True, f"Systems accessed: {[s.value for s in result.systems_accessed]}")
        print_status(True, f"Latency: {result.total_latency_ms:.2f}ms")

        # Show results
        for r in result.results[:3]:
            print(f"      {r.node_id}: relevance={r.relevance:.2f}")

        # Get metrics
        metrics = conductor.get_performance_metrics()
        print_status(True, f"Total queries: {metrics.total_queries}, Cache hits: {metrics.cache_hits}")

        return True

    except Exception as e:
        print_status(False, f"Memory Conductor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_unified_memory():
    """Test UnifiedMemory - User-facing API."""
    print_section("5. UNIFIED MEMORY API")

    try:
        from hololoom.memory.unified import (
            UnifiedMemory,
            Memory,
            RecallStrategy,
            NavigationDirection
        )

        # Create unified memory
        memory = UnifiedMemory(user_id="demo_user", enable_conductor=False)
        print_status(True, "UnifiedMemory instance created")

        # Store memories
        mem_ids = []
        memories_to_store = [
            ("Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
             {"topic": "algorithms", "category": "reinforcement_learning"}),
            ("It balances exploration and exploitation using Beta distributions",
             {"topic": "algorithms", "category": "probability"}),
            ("UCB provides an upper confidence bound for action selection",
             {"topic": "algorithms", "category": "reinforcement_learning"}),
            ("Both algorithms are used in A/B testing and recommendation systems",
             {"topic": "applications", "category": "industry"}),
        ]

        for text, context in memories_to_store:
            mem_id = await memory.store(text=text, context=context, importance=0.8)
            mem_ids.append(mem_id)
            print(f"      Stored: {mem_id}")

        print_status(True, f"Stored {len(mem_ids)} memories")

        # Recall with different strategies
        print("\n  Testing recall strategies:")

        # SIMILAR
        results = await memory.recall(
            "multi-armed bandit",
            strategy=RecallStrategy.SIMILAR,
            limit=3
        )
        print_status(True, f"SIMILAR: Found {len(results)} memories")

        # BALANCED
        results = await memory.recall(
            "exploration vs exploitation",
            strategy=RecallStrategy.BALANCED,
            limit=3
        )
        print_status(True, f"BALANCED: Found {len(results)} memories")

        # Discover patterns (will be limited without graph backend)
        patterns = memory.discover_patterns(min_strength=0.3)
        print_status(True, f"Discovered {len(patterns)} patterns")

        # Health check
        health = await memory.health_check()
        print_status(True, f"Health status: {health['status']}")
        print(f"      Backend available: {health['components']['backend']['available']}")
        print(f"      Total memories: {health['metrics']['total_memories']}")

        return True

    except Exception as e:
        print_status(False, f"UnifiedMemory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hololoom_main_api():
    """Test HoloLoom main API - The 10/10 layer."""
    print_section("6. HOLOLOOM MAIN API (10/10 Layer)")

    try:
        # Import from hololoom.py (memory-focused) not unified_api.py
        from hololoom.hololoom import hololoom

        print("  Creating HoloLoom instance (may take a moment)...")

        # Use context manager for proper lifecycle
        async with HoloLoom() as loom:
            print_status(True, "HoloLoom instance created")

            # Experience (form memories)
            print("\n  Testing experience() - Form memories:")
            memories_formed = []

            experience_data = [
                "Thompson Sampling is a Bayesian algorithm for multi-armed bandits",
                "It maintains Beta distributions for each arm's success probability",
                "UCB (Upper Confidence Bound) is an alternative exploration strategy",
                "Both are used in recommendation systems and A/B testing",
            ]

            for content in experience_data:
                mem = await loom.experience(content)
                memories_formed.append(mem)
                print(f"      Formed memory: {mem.id if hasattr(mem, 'id') else str(mem)[:30]}")

            print_status(True, f"Formed {len(memories_formed)} memories")

            # Recall (retrieve memories)
            print("\n  Testing recall() - Retrieve memories:")
            recalled = await loom.recall("What is Thompson Sampling?")
            print_status(True, f"Recalled {len(recalled)} memories")

            for mem in recalled[:3]:
                text = mem.text if hasattr(mem, 'text') else str(mem)
                print(f"      - {text[:60]}...")

            # Get metrics
            print("\n  Testing get_metrics() - System metrics:")
            metrics = loom.get_metrics()
            print_status(True, f"Metrics retrieved")

            if 'activation' in metrics:
                print(f"      Activation - Active nodes: {metrics['activation'].get('active_nodes', 'N/A')}")
            if 'coherence' in metrics:
                print(f"      Coherence: {metrics['coherence'].get('global_coherence', 'N/A')}")

            # Summary
            print("\n  Testing summary() - System summary:")
            summary = loom.summary()
            print_status(True, "Summary generated")
            print(f"      {summary[:200]}..." if len(summary) > 200 else f"      {summary}")

        print_status(True, "HoloLoom context manager closed cleanly")
        return True

    except Exception as e:
        print_status(False, f"HoloLoom main API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all memory system tests."""
    print("\n" + "="*60)
    print("    HOLOLOOM MEMORY SYSTEM - FULL DEMONSTRATION")
    print("="*60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    print("This demo exercises all core memory system components.\n")

    results = {}
    start_time = time.time()

    # Run tests in order
    tests = [
        ("Knowledge Graph", test_knowledge_graph),
        ("Activation Field", test_activation_field),
        ("Awareness Graph", test_awareness_graph),
        ("Memory Conductor", test_memory_conductor),
        ("Unified Memory", test_unified_memory),
        ("HoloLoom Main API", test_hololoom_main_api),
    ]

    for name, test_func in tests:
        try:
            result = await test_func()
            # Handle both tuple and bool returns
            if isinstance(result, tuple):
                results[name] = result[0]
            else:
                results[name] = result
        except Exception as e:
            print_status(False, f"Test {name} crashed: {e}")
            results[name] = False

    # Summary
    elapsed = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print_section("FINAL SUMMARY")
    print(f"\n  Tests passed: {passed}/{total}")
    print(f"  Total time: {elapsed:.2f}s")
    print()

    for name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")

    if passed == total:
        print(f"\n  {'='*50}")
        print("  THE MEMORY SYSTEM IS WORKING!")
        print(f"  {'='*50}")
    else:
        print(f"\n  {'='*50}")
        print(f"  {total - passed} test(s) failed - see output above")
        print(f"  {'='*50}")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
