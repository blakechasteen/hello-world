#!/usr/bin/env python3
"""
Live HoloLoom test with real Ollama LLM calls.
Tests: direct answer, multipass research, verification, agentic search.
"""
import asyncio
import logging
import sys
import time

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')

from hololoom.awareness.llm_integration import OllamaLLM, create_llm
from hololoom.documentation.types import Query, MemoryShard
from hololoom.config import Config


# Rich test shards
SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that uses probability matching.", metadata={"topic": "algorithms", "confidence": 0.9}),
    MemoryShard(id="s2", text="Unlike epsilon-greedy which explores uniformly, Thompson Sampling explores proportionally to the probability of being optimal.", metadata={"topic": "algorithms", "confidence": 0.85}),
    MemoryShard(id="s3", text="Thompson Sampling is used in A/B testing, recommendation systems, and clinical trials for adaptive experimentation.", metadata={"topic": "applications", "confidence": 0.8}),
    MemoryShard(id="s4", text="The main tradeoff is that Thompson Sampling requires maintaining probability distributions, which adds computational overhead.", metadata={"topic": "tradeoffs", "confidence": 0.75}),
    MemoryShard(id="s5", text="HoloLoom uses a 9-step weaving cycle: Loom Command, Chrono Trigger, Yarn Graph, Resonance Shed, Warp Space, Convergence Engine, Tool Execution, Spacetime Fabric, Reflection Buffer.", metadata={"topic": "architecture", "confidence": 0.95}),
    MemoryShard(id="s6", text="The multipass crawl system retrieves memories in progressive passes with increasing thresholds, from broad exploration to focused retrieval.", metadata={"topic": "retrieval", "confidence": 0.88}),
]


def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


async def test_1_raw_ollama():
    """Test 1: Raw Ollama LLM call (no HoloLoom)."""
    banner("TEST 1: Raw Ollama LLM Call")

    llm = OllamaLLM(model="llama3.2:3b")
    print(f"  Available: {llm.is_available()}")

    t0 = time.perf_counter()
    response = await llm.generate(
        prompt="In 2-3 sentences, what is Thompson Sampling?",
        system_prompt="You are a concise AI assistant.",
        max_tokens=150,
        temperature=0.7
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Model: {response.model}")
    print(f"  Tokens: {response.usage}")
    print(f"  Time: {elapsed:.0f}ms")
    print(f"\n  Response:\n  {response.content}")
    return True


async def test_2_weaving_orchestrator_llm():
    """Test 2: Full WeavingOrchestrator with LLM tool executor."""
    banner("TEST 2: WeavingOrchestrator + LLM (Direct Weave)")

    from hololoom.weaving_orchestrator_llm import WeavingOrchestrator

    config = Config.fast()
    config.enable_agentic_reasoning = False
    config.enable_safety_guardrails = False  # testing mode - bypass approval requirements

    t0 = time.perf_counter()
    async with WeavingOrchestrator(cfg=config, shards=SHARDS, llm_provider="ollama") as orchestrator:
        init_time = (time.perf_counter() - t0) * 1000
        print(f"  Orchestrator init: {init_time:.0f}ms")
        print(f"  LLM active: {orchestrator.tool_executor.llm is not None}")

        query = Query(text="What is Thompson Sampling and how does it work?")
        print(f"  Query: {query.text}")

        t1 = time.perf_counter()
        spacetime = await orchestrator.weave(query)
        weave_time = (time.perf_counter() - t1) * 1000

        print(f"  Weave time: {weave_time:.0f}ms")
        print(f"  Confidence: {spacetime.confidence:.2f}")
        print(f"  Tool: {spacetime.tool_used}")

        # Spacetime stores results in response + metadata
        response_text = spacetime.response or "No response"
        meta = spacetime.metadata or {}
        llm_provider = meta.get("llm_provider", "none")
        llm_model = meta.get("llm_model", "none")
        usage = meta.get("usage", {})

        print(f"  LLM Provider: {llm_provider}")
        print(f"  LLM Model: {llm_model}")
        print(f"  Usage: {usage}")
        print(f"  Metadata keys: {list(meta.keys())}")
        print(f"\n  Response:\n  {response_text[:500]}")

        return len(response_text) > 20  # Real response, not just error stub


async def test_3_agentic_research():
    """Test 3: Agentic RESEARCH mode with LLM-generated queries."""
    banner("TEST 3: Agentic RESEARCH Mode (Multipass + LLM Query Gen)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False  # testing mode

    t0 = time.perf_counter()
    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=SHARDS,
        enable_verification=True,
        enable_goal_tracking=True
    )
    init_time = (time.perf_counter() - t0) * 1000
    print(f"  Init: {init_time:.0f}ms")
    print(f"  LLM active: {orchestrator.llm is not None}")

    query = Query(text="How does Thompson Sampling work and when should I use it?")
    print(f"  Query: {query.text}")
    print(f"  Mode: RESEARCH (max_steps=3)")

    t1 = time.perf_counter()
    result = await orchestrator.reason(
        query=query,
        mode=ReasoningMode.RESEARCH,
        max_steps=3
    )
    research_time = (time.perf_counter() - t1) * 1000

    print(f"\n  Total time: {research_time:.0f}ms")
    print(f"  Total queries: {result.total_queries}")
    print(f"  Final confidence: {result.spacetime.confidence:.2f}")

    # Show research steps
    llm_generated = 0
    template_count = 0
    for i, step in enumerate(result.steps_taken):
        step_type = step.get("type", "unknown")
        if step_type == "research_query":
            q = step.get("query", "")
            conf = step.get("confidence", 0)
            is_template = any(t in q for t in [
                "What are the key concepts",
                "What are the tradeoffs",
                "What are practical applications",
                "What are common misconceptions",
                "What are recent developments"
            ])
            marker = "[TEMPLATE]" if is_template else "[LLM-GEN]"
            if is_template:
                template_count += 1
            else:
                llm_generated += 1
            print(f"  Step {i+1}: {marker} {q[:80]}...")
            print(f"          conf={conf:.2f}")
        elif step_type == "synthesis":
            print(f"  Step {i+1}: [SYNTHESIS] sources={step.get('sources', 0)} conf={step.get('confidence', 0):.2f}")

    # Final result
    response = result.spacetime.response or "No response"
    print(f"\n  Final Response:\n  {response[:500]}")

    print(f"\n  Summary: {llm_generated} LLM-generated, {template_count} template queries")

    return llm_generated > 0


async def test_4_agentic_verify():
    """Test 4: Agentic VERIFY mode - answer + verification loops."""
    banner("TEST 4: Agentic VERIFY Mode")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False  # testing mode

    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=SHARDS,
        enable_verification=True,
        enable_goal_tracking=True
    )
    print(f"  LLM active: {orchestrator.llm is not None}")

    query = Query(text="Is Thompson Sampling always the best approach for bandit problems?")
    print(f"  Query: {query.text}")
    print(f"  Mode: VERIFY (max_steps=3)")

    t0 = time.perf_counter()
    result = await orchestrator.reason(
        query=query,
        mode=ReasoningMode.VERIFY,
        max_steps=3
    )
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Time: {elapsed:.0f}ms")
    print(f"  Total queries: {result.total_queries}")

    if result.verification:
        v = result.verification
        print(f"  Verified: {v.verified}")
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Contradictions: {len(v.contradictions)}")
        print(f"  Supporting: {len(v.supporting_evidence)}")
        if v.suggested_refinements:
            print(f"  Refinements: {v.suggested_refinements[0][:80]}...")

    for i, step in enumerate(result.steps_taken):
        print(f"  Step {i+1}: {step.get('type', '?')} conf={step.get('confidence', 0):.2f}")

    return True


async def test_5_streaming():
    """Test 5: Streaming LLM generation."""
    banner("TEST 5: Streaming Ollama Generation")

    llm = OllamaLLM(model="llama3.2:3b")

    print("  Streaming response: ", end="", flush=True)
    t0 = time.perf_counter()
    tokens = 0
    async for token in llm.stream_generate(
        prompt="What is multipass retrieval in 1 sentence?",
        system_prompt="Be very concise.",
        max_tokens=80,
        temperature=0.5
    ):
        print(token, end="", flush=True)
        tokens += 1
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  [{tokens} chunks, {elapsed:.0f}ms]")
    return True


async def main():
    banner("HoloLoom Live LLM Test Suite")
    print(f"  Testing with Ollama llama3.2:3b on localhost")

    results = {}

    # Test 1: Raw Ollama
    try:
        results["raw_ollama"] = await test_1_raw_ollama()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["raw_ollama"] = False

    # Test 2: WeavingOrchestrator + LLM
    try:
        results["weaving_llm"] = await test_2_weaving_orchestrator_llm()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["weaving_llm"] = False

    # Test 3: Agentic RESEARCH
    try:
        results["agentic_research"] = await test_3_agentic_research()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["agentic_research"] = False

    # Test 4: Agentic VERIFY
    try:
        results["agentic_verify"] = await test_4_agentic_verify()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        results["agentic_verify"] = False

    # Test 5: Streaming
    try:
        results["streaming"] = await test_5_streaming()
    except Exception as e:
        print(f"  FAILED: {e}")
        results["streaming"] = False

    # Summary
    banner("RESULTS")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
