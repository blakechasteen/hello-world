#!/usr/bin/env python3
"""
Test Real HoloLoom Integration

Tests that EnhancedHoloLoomBot:
1. Initializes with production features
2. Uses dynamic Yarn Graph (KG) memory
3. Executes full weaving cycle
4. Provides real-time WebSocket updates
5. Returns complete response with stages

Usage:
    PYTHONPATH=. python test_hololoom_integration.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

# Add HoloLoom to path
HOLOLOOM_PATH = Path(__file__).parent.parent / "HoloLoom"
if HOLOLOOM_PATH.exists():
    sys.path.insert(0, str(HOLOLOOM_PATH.parent))

from bot.hololoom_integration_enhanced import EnhancedHoloLoomBot


async def test_initialization():
    """Test bot initialization"""
    print("\n=== Test 1: Bot Initialization ===")

    # Create bot with production features
    bot = EnhancedHoloLoomBot(
        config_mode="FAST",
        enable_production_hardening=True,
        enable_neo4j=False,  # Use NetworkX for test
        websocket_callback=None  # No WebSocket in test
    )

    # Initialize
    await bot.initialize()

    # Verify initialization
    assert bot.initialized, "Bot should be initialized"
    assert bot.orchestrator is not None, "Orchestrator should exist"
    assert bot.yarn_graph is not None, "Yarn Graph should exist"
    assert bot.config is not None, "Config should exist"

    # Verify Yarn Graph has knowledge
    node_count = bot.yarn_graph.num_nodes()
    edge_count = bot.yarn_graph.num_edges()

    print(f"✓ Bot initialized successfully")
    print(f"  - Orchestrator: {type(bot.orchestrator).__name__}")
    print(f"  - Yarn Graph nodes: {node_count}")
    print(f"  - Yarn Graph edges: {edge_count}")
    print(f"  - Config mode: {bot.config_mode}")
    print(f"  - Production hardening: {bot.enable_production_hardening}")

    # Cleanup
    await bot.cleanup()
    print(f"✓ Bot cleaned up successfully")

    return True


async def test_weaving():
    """Test weaving cycle"""
    print("\n=== Test 2: Weaving Cycle ===")

    # Track WebSocket updates
    websocket_updates = []

    async def websocket_callback(data):
        """Track WebSocket updates"""
        websocket_updates.append(data)
        print(f"  📡 WebSocket: {data.get('type')}")

    # Create bot with WebSocket callback
    bot = EnhancedHoloLoomBot(
        config_mode="FAST",
        enable_production_hardening=True,
        enable_neo4j=False,
        websocket_callback=websocket_callback
    )

    try:
        # Initialize
        await bot.initialize()

        # Execute weaving
        print("\n🔄 Executing weaving...")
        response = await bot.weave(
            query="What is Thompson Sampling?",
            user_id="@test:matrix.org",
            room_id="!test:matrix.org",
            complexity="FAST"
        )

        # Verify response
        assert response.text, "Response should have text"
        assert response.confidence >= 0.0 and response.confidence <= 1.0, "Confidence should be [0-1]"
        assert response.tool_used, "Tool should be selected"
        assert response.latency_ms > 0, "Latency should be positive"

        print(f"\n✓ Weaving completed successfully")
        print(f"  - Response length: {len(response.text)} chars")
        print(f"  - Confidence: {response.confidence:.2%}")
        print(f"  - Tool used: {response.tool_used}")
        print(f"  - Latency: {response.latency_ms:.0f}ms")
        print(f"  - Context depth: {response.context_depth}")
        print(f"  - Memories retrieved: {response.memories_retrieved}")
        print(f"  - Cache hit: {response.cache_hit}")
        print(f"  - Stages: {len(response.stages)}")

        # Verify WebSocket updates
        print(f"\n✓ WebSocket updates: {len(websocket_updates)}")

        update_types = [update.get('type') for update in websocket_updates]
        print(f"  - Update types: {set(update_types)}")

        # Should have: weaving_start, weaving_stage (multiple), weaving_complete
        assert 'weaving_start' in update_types, "Should have weaving_start"
        assert 'weaving_complete' in update_types, "Should have weaving_complete"

        # Print sample stage data
        if response.stages:
            print(f"\n📊 Sample stage data:")
            for stage in response.stages[:3]:  # First 3 stages
                print(f"  - {stage['name']}: {stage['status']} ({stage.get('duration_ms', 0):.1f}ms)")

        return True

    finally:
        await bot.cleanup()


async def test_statistics():
    """Test statistics retrieval"""
    print("\n=== Test 3: Statistics ===")

    bot = EnhancedHoloLoomBot(
        config_mode="FAST",
        enable_production_hardening=True
    )

    try:
        await bot.initialize()

        # Get statistics
        stats = await bot.get_statistics()

        print(f"✓ Statistics retrieved")
        print(f"  - Initialized: {stats['initialized']}")
        print(f"  - Config mode: {stats['config_mode']}")
        print(f"  - Production hardening: {stats['production_hardening']}")
        print(f"  - Memory backend: {stats['memory_backend']}")

        if 'memory' in stats:
            print(f"  - Memory nodes: {stats['memory']['nodes']}")
            print(f"  - Memory edges: {stats['memory']['edges']}")

        return True

    finally:
        await bot.cleanup()


async def test_memory_operations():
    """Test adding memories"""
    print("\n=== Test 4: Memory Operations ===")

    bot = EnhancedHoloLoomBot(config_mode="FAST")

    try:
        await bot.initialize()

        initial_nodes = bot.yarn_graph.num_nodes()
        initial_edges = bot.yarn_graph.num_edges()

        print(f"Initial: {initial_nodes} nodes, {initial_edges} edges")

        # Add a memory
        await bot.add_memory(
            text="Python Thompson Sampling implementation uses numpy",
            entities=["Python", "Thompson Sampling", "numpy"],
            user_id="@test:matrix.org"
        )

        final_nodes = bot.yarn_graph.num_nodes()
        final_edges = bot.yarn_graph.num_edges()

        print(f"After add: {final_nodes} nodes, {final_edges} edges")

        assert final_nodes >= initial_nodes, "Nodes should increase or stay same"
        assert final_edges > initial_edges, "Edges should increase"

        print(f"✓ Memory added successfully (+{final_edges - initial_edges} edges)")

        return True

    finally:
        await bot.cleanup()


async def run_all_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("Real HoloLoom Integration Test Suite")
    print("=" * 70)

    tests = [
        ("Initialization", test_initialization),
        ("Weaving Cycle", test_weaving),
        ("Statistics", test_statistics),
        ("Memory Operations", test_memory_operations),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("\n🎉 All tests passed! Real HoloLoom integration is working.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check output above.")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
