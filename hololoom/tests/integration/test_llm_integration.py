#!/usr/bin/env python3
"""
Quick test to verify LLM integration with WeavingOrchestrator.

Tests:
1. LLM initialization (with fallback)
2. Answer generation with actual LLM (if Ollama available)
3. Fallback behavior when LLM unavailable

Usage:
    PYTHONPATH=. python test_llm_integration.py
"""

import asyncio
import logging
from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_llm_integration():
    """Test LLM integration with orchestrator."""
    print("="*70)
    print("Testing LLM Integration with WeavingOrchestrator")
    print("="*70)

    # Create test shards
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            metadata={"topic": "algorithms", "confidence": 0.9}
        ),
        MemoryShard(
            id="shard_2",
            text="It balances exploration and exploitation using probability matching.",
            metadata={"topic": "algorithms", "confidence": 0.85}
        )
    ]

    # Create config and orchestrator
    config = Config.fast()
    config.enable_agentic_reasoning = False  # Use direct mode for simple test

    print(f"\n[1/3] Initializing orchestrator...")
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        print(f"✅ Orchestrator initialized")

        # Check if LLM is available
        llm_status = "available" if orchestrator.tool_executor.llm else "unavailable (fallback mode)"
        print(f"   LLM status: {llm_status}")

        # Test query
        query = Query(text="What is Thompson Sampling?")
        print(f"\n[2/3] Processing query: {query.text}")

        spacetime = await orchestrator.weave(query)

        print(f"✅ Query processed")
        print(f"   Tool used: {spacetime.tool_used}")
        print(f"   Confidence: {spacetime.confidence:.2f}")

        # Display result
        print(f"\n[3/3] Result:")
        print("="*70)
        result_text = spacetime.tool_output.get("result", "No result")
        print(result_text)
        print("="*70)

        # Show metadata
        print(f"\nMetadata:")
        if "llm_provider" in spacetime.tool_output:
            print(f"  LLM Provider: {spacetime.tool_output['llm_provider']}")
            print(f"  LLM Model: {spacetime.tool_output['llm_model']}")
            print(f"  Usage: {spacetime.tool_output.get('usage', 'N/A')}")
        else:
            print(f"  (Fallback mode - no LLM used)")

        print(f"  Sources: {spacetime.tool_output.get('sources', 0)}")
        print(f"  Context tokens: {spacetime.tool_output.get('context_tokens', 'N/A')}")

        print(f"\n✅ Test complete!")

        if "llm_provider" in spacetime.tool_output:
            print("\n🎉 SUCCESS: LLM integration working! Actual LLM response generated.")
        else:
            print("\n⚠️  FALLBACK: LLM unavailable, using fallback mode.")
            print("   To enable LLM: Install Ollama and run 'ollama pull llama3.2:3b'")


if __name__ == "__main__":
    asyncio.run(test_llm_integration())
