#!/usr/bin/env python3
"""
Test LLM-Activated Agentic Search for Memory Recall

Verifies that RESEARCH mode uses LLM to generate intelligent follow-up queries
instead of hardcoded templates.

Usage:
    PYTHONPATH=. python test_llm_agentic_search.py
"""

import pytest

pytest.importorskip("gradio")

import asyncio
import logging
from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_llm_agentic_search():
    """Test LLM-activated agentic search with RESEARCH mode."""
    print("="*80)
    print("Testing LLM-Activated Agentic Search for Memory Recall")
    print("="*80)

    # Create test shards with rich content
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that uses probability matching.",
            metadata={"topic": "algorithms", "confidence": 0.9}
        ),
        MemoryShard(
            id="shard_2",
            text="Unlike epsilon-greedy which explores uniformly, Thompson Sampling explores proportionally to the probability of being optimal.",
            metadata={"topic": "algorithms", "confidence": 0.85}
        ),
        MemoryShard(
            id="shard_3",
            text="Thompson Sampling is used in A/B testing, recommendation systems, and clinical trials for adaptive experimentation.",
            metadata={"topic": "applications", "confidence": 0.8}
        ),
        MemoryShard(
            id="shard_4",
            text="The main tradeoff is that Thompson Sampling requires maintaining probability distributions, which adds computational overhead.",
            metadata={"topic": "tradeoffs", "confidence": 0.75}
        )
    ]

    # Create config
    config = Config.fast()
    config.enable_agentic_reasoning = True

    print(f"\n[1/5] Creating agentic orchestrator...")
    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=shards,
        enable_verification=True,
        enable_goal_tracking=True
    )
    print(f"✅ Orchestrator created")

    # Check if LLM is available
    llm_available = orchestrator.llm is not None
    print(f"   LLM status: {'available' if llm_available else 'unavailable (fallback mode)'}")

    # Test query
    query = Query(text="How does Thompson Sampling work and when should I use it?")
    print(f"\n[2/5] Running RESEARCH mode with query:")
    print(f"   '{query.text}'")

    try:
        result = await orchestrator.reason(
            query=query,
            mode=ReasoningMode.RESEARCH,
            max_steps=3  # Generate 3 research queries
        )

        print(f"\n✅ RESEARCH mode complete!")
        print(f"   Reasoning mode: {result.reasoning_mode.value}")
        print(f"   Total queries: {result.total_queries}")
        print(f"   Duration: {result.total_duration_ms:.1f}ms")
        print(f"   Final confidence: {result.spacetime.confidence:.2f}")

        # Display research steps
        print(f"\n[3/5] Research steps taken:")
        print("="*80)
        for i, step in enumerate(result.steps_taken):
            step_type = step.get("type", "unknown")
            print(f"\nStep {i+1}: {step_type.upper()}")

            if step_type == "research_query":
                query_text = step.get("query", "")
                confidence = step.get("confidence", 0.0)
                findings = step.get("findings", "")

                print(f"  Query: {query_text}")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Findings: {findings}")

                # Check if this looks like LLM-generated (more specific than template)
                is_template = any([
                    "What are the key concepts" in query_text,
                    "What are the tradeoffs" in query_text,
                    "What are practical applications" in query_text,
                    "What are common misconceptions" in query_text,
                    "What are recent developments" in query_text
                ])

                marker = "📋 [Template]" if is_template else "🤖 [LLM-Generated]"
                print(f"  {marker}")

            elif step_type == "synthesis":
                print(f"  Synthesizing findings from {step.get('sources', 0)} sources")
                print(f"  Final confidence: {step.get('confidence', 0.0):.2f}")

        # Display final result
        print(f"\n[4/5] Final Result:")
        print("="*80)
        result_text = result.spacetime.response or "No result"
        print(result_text[:500])  # First 500 chars
        if len(result_text) > 500:
            print("...")
        print("="*80)

        # Analyze results
        print(f"\n[5/5] Analysis:")
        research_steps = [s for s in result.steps_taken if s.get("type") == "research_query"]
        template_count = sum(1 for s in research_steps if any([
            "What are the key concepts" in s.get("query", ""),
            "What are the tradeoffs" in s.get("query", ""),
            "What are practical applications" in s.get("query", "")
        ]))

        if llm_available:
            if template_count == 0:
                print("✅ SUCCESS: All research queries are LLM-generated!")
                print("   Agentic search is using intelligent query generation.")
            elif template_count < len(research_steps):
                print(f"⚠️  PARTIAL: {len(research_steps) - template_count}/{len(research_steps)} queries are LLM-generated")
                print("   LLM generation working but some fallbacks occurred.")
            else:
                print(f"❌ FAILURE: All queries are templates (LLM not being used)")
                print("   LLM is available but query generation is not working.")
        else:
            print(f"⚠️  FALLBACK MODE: LLM unavailable, using template queries")
            print(f"   To enable LLM: Install Ollama and run 'ollama pull llama3.2:3b'")
            print(f"   Template queries: {template_count}/{len(research_steps)}")

        print(f"\n✅ Test complete!")

        # Cleanup
        if hasattr(orchestrator.learning_engine, '__aexit__'):
            await orchestrator.learning_engine.__aexit__(None, None, None)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(test_llm_agentic_search())
