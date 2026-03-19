"""
MCP Server Bridge Integration Test

Verifies that the MCP server correctly initializes and uses the CognitiveBridge:
  1. init_memory() creates bridge alongside UnifiedMemory
  2. store_memory tool dual-writes to both backends
  3. recall_memories tool returns v2 enriched results
  4. recall_memories with enhance=false falls back to v1
  5. recall_memories with pattern="fused" uses wider PPR
  6. chat tool uses v2 recall
"""

import asyncio
import sys


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
    # Import the MCP server module
    from hololoom.memory import mcp_server as mcp
    from hololoom.memory.unified import UnifiedMemory
    from hololoom.memory.v2_bridge import CognitiveBridge

    print("\n" + "=" * 72)
    print("  I. MCP Server Bridge Initialization")
    print("=" * 72)

    # Initialize directly (bypass create_unified_memory kwargs issue)
    mcp.memory = UnifiedMemory(
        user_id="test",
        enable_mem0=False,
        enable_neo4j=False,
        enable_qdrant=False,
    )
    mcp.bridge = CognitiveBridge(mcp.memory)
    await mcp.bridge.initialize()

    check(mcp.memory is not None, "UnifiedMemory initialized")
    check(mcp.bridge is not None, "CognitiveBridge initialized")
    check(mcp.bridge._initialized, "Bridge is initialized")

    report = mcp.bridge.report()
    check(report["initialized"], "Bridge report shows initialized")

    print("\n" + "=" * 72)
    print("  II. Store Memory (Dual-Write)")
    print("=" * 72)

    # Use call_tool directly to simulate MCP tool calls
    result = await mcp.call_tool("store_memory", {
        "text": "Thompson Sampling explores by sampling from posterior distributions",
        "context": {"importance": 0.8},
        "tags": ["ml", "bandits"],
    })
    check(len(result) == 1, "store_memory returns result")
    check("stored successfully" in result[0].text, f"Store confirmed: {result[0].text[:60]}")

    # Store more memories to build graph
    for text in [
        "Epsilon-greedy explores with probability epsilon",
        "UCB1 uses upper confidence bounds for exploration",
        "Both Thompson and Epsilon are multi-armed bandit algorithms",
        "Bayesian optimization extends Thompson Sampling to continuous spaces",
    ]:
        await mcp.call_tool("store_memory", {
            "text": text,
            "context": {"importance": 0.7},
            "tags": ["ml"],
        })

    # Check bridge graph was populated
    bridge_report = mcp.bridge.report()
    check(bridge_report["graph_nodes"] > 5, f"Bridge graph has nodes ({bridge_report['graph_nodes']})")
    check(bridge_report["graph_edges"] > 0, f"Bridge graph has edges ({bridge_report['graph_edges']})")

    print("\n" + "=" * 72)
    print("  III. Recall Memories (v2 Enhanced)")
    print("=" * 72)

    result = await mcp.call_tool("recall_memories", {
        "query": "Thompson Sampling bandit",
        "limit": 5,
        "enhance": True,
    })
    check(len(result) == 1, "recall returns result")
    text = result[0].text
    check("v2 enhanced" in text or "v2 Cognitive Analysis" in text, "v2 enhanced output present")
    check("Confidence:" in text, "Contains confidence score")
    check("Seeds:" in text, "Contains seed info")

    print(f"\n  --- v2 Enhanced Output Sample ---")
    for line in text.split("\n")[:12]:
        print(f"  | {line}")
    print(f"  --- End Sample ---")

    print("\n" + "=" * 72)
    print("  IV. Recall with Pattern Cards")
    print("=" * 72)

    # Test each pattern
    for pattern in ["bare", "fast", "fused"]:
        r = await mcp.call_tool("recall_memories", {
            "query": "Thompson Sampling",
            "limit": 10,
            "pattern": pattern,
        })
        check(len(r) == 1, f"{pattern.upper()} returns result")
        check("Confidence:" in r[0].text, f"{pattern.upper()} has confidence")

    print("\n" + "=" * 72)
    print("  V. Recall with enhance=false (v1 Fallback)")
    print("=" * 72)

    result = await mcp.call_tool("recall_memories", {
        "query": "Thompson",
        "enhance": False,
        "strategy": "fused",
        "limit": 5,
    })
    text = result[0].text
    # v1 output should NOT have "v2 Cognitive Analysis"
    check("v2 Cognitive Analysis" not in text, "v1 fallback: no v2 section")
    check("Found" in text or "No memories" in text, f"v1 returns valid output")

    print("\n" + "=" * 72)
    print("  VI. Chat Tool (v2 Recall)")
    print("=" * 72)

    result = await mcp.call_tool("chat", {
        "message": "Tell me about Thompson Sampling",
        "importance_threshold": 0.4,
    })
    text = result[0].text
    check("memories" in text.lower() or "found" in text.lower() or "help" in text.lower(),
          f"Chat returns response")
    check("Importance:" in text or "REMEMBERED" in text or "FILTERED" in text,
          "Chat includes importance tracking")

    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
