#!/usr/bin/env python3
"""
Demo: HoloLoom MCP Tools
=========================

**Created**: November 21, 2025
**Purpose**: Demonstrate HoloLoom MCP server capabilities

This demo shows how to use HoloLoom's MCP tools for Claude Desktop integration.
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hololoom.mcp_tools import server
    MCP_AVAILABLE = True
except ImportError:
    print("ERROR: Could not import HoloLoom MCP tools")
    MCP_AVAILABLE = False
    sys.exit(1)


async def demo_memory_tools():
    """Demonstrate memory storage and retrieval."""
    print("\n" + "="*70)
    print("Demo 1: Memory Tools (experience + recall)")
    print("="*70)

    if not server.HOLOLOOM_AVAILABLE:
        print("⚠ HoloLoom not available, skipping")
        return

    # Store some knowledge
    print("\n1. Storing memories...")
    memories = [
        "Thompson Sampling is a Bayesian exploration algorithm that balances exploitation and exploration.",
        "HoloLoom uses a 9-step weaving cycle for processing queries.",
        "The Policy Engine combines neural predictions with Thompson Sampling priors.",
    ]

    for memory in memories:
        result = await server.tool_experience({"content": memory})
        print(f"✓ Stored: {memory[:60]}...")

    # Recall knowledge
    print("\n2. Recalling memories...")
    queries = [
        "What is Thompson Sampling?",
        "How does HoloLoom process queries?",
        "What does the Policy Engine do?"
    ]

    for query in queries:
        result = await server.tool_recall({"query": query, "limit": 2})
        import json
        data = json.loads(result[0].text)
        print(f"\nQuery: {query}")
        print(f"Found: {data['memories_found']} memories")
        if data['memories']:
            print(f"Top result: {data['memories'][0]['content'][:100]}...")


async def demo_metrics():
    """Demonstrate metrics retrieval."""
    print("\n" + "="*70)
    print("Demo 2: System Metrics")
    print("="*70)

    if not server.HOLOLOOM_AVAILABLE:
        print("⚠ HoloLoom not available, skipping")
        return

    result = await server.tool_metrics({
        "include_learning": True,
        "include_performance": True
    })

    import json
    data = json.loads(result[0].text)

    print(f"\n✓ Retrieved metrics at {data['timestamp']}")
    print(f"  Metrics: {json.dumps(data['metrics'], indent=2)[:200]}...")


async def demo_weaving_cycle():
    """Demonstrate weaving cycle."""
    print("\n" + "="*70)
    print("Demo 3: Weaving Cycle (9-step processing)")
    print("="*70)

    if not server.HOLOLOOM_AVAILABLE:
        print("⚠ HoloLoom not available, skipping")
        return

    queries = [
        ("What is 2+2?", "FAST"),
        ("Explain the tradeoffs of Thompson Sampling vs epsilon-greedy", "FUSED"),
    ]

    for query, mode in queries:
        print(f"\nQuery: {query}")
        print(f"Mode: {mode}")

        result = await server.tool_weave({
            "query": query,
            "mode": mode,
            "enable_reflection": False
        })

        import json
        data = json.loads(result[0].text)
        print(f"Confidence: {data['confidence']:.2f}")
        print(f"Response: {data['response'][:150]}...")


async def demo_agentic_reasoning():
    """Demonstrate agentic reasoning."""
    print("\n" + "="*70)
    print("Demo 4: Agentic Reasoning (4 modes)")
    print("="*70)

    if not server.AGENTIC_AVAILABLE:
        print("⚠ Agentic reasoning not available, skipping")
        return

    modes = [
        ("What is Thompson Sampling?", "direct"),
        ("Verify that Thompson Sampling is Bayesian", "verify"),
        ("Research all tradeoffs of Thompson Sampling", "research"),
    ]

    for query, mode in modes:
        print(f"\nQuery: {query}")
        print(f"Mode: {mode}")

        result = await server.tool_reason({
            "query": query,
            "mode": mode,
            "max_steps": 5
        })

        import json
        data = json.loads(result[0].text)
        print(f"Confidence: {data['confidence']:.2f}")
        print(f"Steps: {data.get('steps_taken', 'N/A')}")
        print(f"Response: {data['response'][:150]}...")


async def demo_summary():
    """Demonstrate system summary."""
    print("\n" + "="*70)
    print("Demo 5: System Summary")
    print("="*70)

    if not server.HOLOLOOM_AVAILABLE:
        print("⚠ HoloLoom not available, skipping")
        return

    result = await server.tool_summary({})
    print(f"\n{result[0].text[:300]}...")


async def demo_tool_listing():
    """Demonstrate tool listing."""
    print("\n" + "="*70)
    print("Demo 6: List All Available Tools")
    print("="*70)

    if not server.MCP_AVAILABLE:
        print("⚠ MCP SDK not available, skipping")
        return

    tools = await server.list_tools()

    print(f"\nTotal tools available: {len(tools)}")
    print("\nTool List:")
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool.name}")
        print(f"   {tool.description[:100]}...")


async def main():
    """Run all demos."""
    print("="*70)
    print(" HoloLoom MCP Tools Demo")
    print("="*70)
    print(f"\nMCP SDK: {'✓ Available' if server.MCP_AVAILABLE else '✗ Not installed'}")
    print(f"HoloLoom: {'✓ Available' if server.HOLOLOOM_AVAILABLE else '✗ Not available'}")
    print(f"Agentic Reasoning: {'✓ Available' if server.AGENTIC_AVAILABLE else '✗ Not available'}")
    print(f"Recursive Learning: {'✓ Available' if server.RECURSIVE_AVAILABLE else '✗ Not available'}")

    # Run demos
    await demo_tool_listing()
    await demo_memory_tools()
    await demo_metrics()
    await demo_weaving_cycle()
    await demo_agentic_reasoning()
    await demo_summary()

    print("\n" + "="*70)
    print(" Demo Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Configure Claude Desktop with HoloLoom MCP server")
    print("2. Restart Claude Desktop")
    print("3. Use tools directly in Claude chat:")
    print("   - 'Store this in memory: <fact>'")
    print("   - 'Recall what I said about <topic>'")
    print("   - 'Show me system metrics'")
    print("\nSee HoloLoom/mcp_tools/README.md for configuration details.")


if __name__ == "__main__":
    if not MCP_AVAILABLE:
        print("ERROR: MCP tools not available")
        sys.exit(1)

    asyncio.run(main())
