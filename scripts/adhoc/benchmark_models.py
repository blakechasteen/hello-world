#!/usr/bin/env python3
"""
Benchmark: llama3.2:3b vs qwen2.5:14b on HoloLoom tasks.
Compares speed, quality, and token counts across identical prompts.
"""
import asyncio
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
logging.basicConfig(level=logging.WARNING)

from hololoom.awareness.llm_integration import OllamaLLM
from hololoom.documentation.types import Query, MemoryShard
from hololoom.config import Config


SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that uses probability matching.", metadata={"topic": "algorithms", "confidence": 0.9}),
    MemoryShard(id="s2", text="Unlike epsilon-greedy which explores uniformly, Thompson Sampling explores proportionally to the probability of being optimal.", metadata={"topic": "algorithms", "confidence": 0.85}),
    MemoryShard(id="s3", text="Thompson Sampling is used in A/B testing, recommendation systems, and clinical trials for adaptive experimentation.", metadata={"topic": "applications", "confidence": 0.8}),
    MemoryShard(id="s4", text="The main tradeoff is that Thompson Sampling requires maintaining probability distributions, which adds computational overhead.", metadata={"topic": "tradeoffs", "confidence": 0.75}),
    MemoryShard(id="s5", text="HoloLoom uses a 9-step weaving cycle: Loom Command, Chrono Trigger, Yarn Graph, Resonance Shed, Warp Space, Convergence Engine, Tool Execution, Spacetime Fabric, Reflection Buffer.", metadata={"topic": "architecture", "confidence": 0.95}),
    MemoryShard(id="s6", text="The multipass crawl system retrieves memories in progressive passes with increasing thresholds, from broad exploration to focused retrieval.", metadata={"topic": "retrieval", "confidence": 0.88}),
]

BENCHMARKS = [
    {
        "name": "Simple Q&A",
        "prompt": "In 2-3 sentences, what is Thompson Sampling?",
        "system": "You are a concise AI assistant.",
        "max_tokens": 150,
    },
    {
        "name": "Research Query Gen",
        "prompt": (
            "Query: How does Thompson Sampling work and when should I use it?\n\n"
            "Generate 3 research questions to explore this topic thoroughly, one per line.\n"
            "Focus on: key concepts, practical applications, tradeoffs.\n\nQuestions:"
        ),
        "system": "You are a research assistant. Generate specific follow-up questions.",
        "max_tokens": 200,
    },
    {
        "name": "RAG Synthesis",
        "prompt": (
            "Context:\n"
            "[1] Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.\n"
            "[2] It explores proportionally to the probability of being optimal.\n"
            "[3] It is used in A/B testing, recommendation systems, and clinical trials.\n"
            "[4] The main tradeoff is computational overhead from maintaining distributions.\n\n"
            "Question: Explain Thompson Sampling, its mechanism, applications, and limitations.\n\n"
            "Answer (be thorough but concise):"
        ),
        "system": "You are a helpful AI assistant integrated with a knowledge system. Answer based on the provided context.",
        "max_tokens": 400,
    },
    {
        "name": "Verification Challenge",
        "prompt": (
            "A system claims: 'Thompson Sampling is always the optimal solution for multi-armed bandit problems.'\n\n"
            "Is this claim accurate? Identify any weaknesses, counterexamples, or nuances."
        ),
        "system": "You are a critical reasoning assistant. Evaluate claims carefully.",
        "max_tokens": 300,
    },
]

MODELS = ["llama3.2:3b", "qwen2.5:14b"]


def banner(text):
    print(f"\n{'='*72}")
    print(f"  {text}")
    print(f"{'='*72}")


async def run_benchmark(model_name, bench):
    """Run a single benchmark and return metrics."""
    llm = OllamaLLM(model=model_name)

    t0 = time.perf_counter()
    response = await llm.generate(
        prompt=bench["prompt"],
        system_prompt=bench["system"],
        max_tokens=bench["max_tokens"],
        temperature=0.7,
    )
    elapsed = time.perf_counter() - t0

    prompt_tokens = response.usage.get("prompt_tokens", 0) if response.usage else 0
    completion_tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
    total_tokens = prompt_tokens + completion_tokens
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    return {
        "model": model_name,
        "bench": bench["name"],
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_per_sec": tok_per_sec,
        "response": response.content,
        "response_len": len(response.content),
    }


async def run_agentic_research(model_name):
    """Run full agentic RESEARCH pipeline with given model."""
    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    # Patch the LLM model in the orchestrator
    orchestrator = await create_agentic_orchestrator(
        config=config,
        shards=SHARDS,
        enable_verification=True,
        enable_goal_tracking=True,
    )

    # Swap the LLM to the target model
    if orchestrator.llm:
        orchestrator.llm = OllamaLLM(model=model_name)
    # Also swap in the underlying weaving orchestrator
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = OllamaLLM(model=model_name)

    query = Query(text="How does Thompson Sampling work and when should I use it?")

    t0 = time.perf_counter()
    result = await orchestrator.reason(
        query=query,
        mode=ReasoningMode.RESEARCH,
        max_steps=3,
    )
    elapsed = time.perf_counter() - t0

    steps = result.steps_taken
    research_qs = [s.get("query", "") for s in steps if s.get("type") == "research_query"]
    response_text = result.spacetime.response or ""

    return {
        "model": model_name,
        "elapsed_s": elapsed,
        "total_queries": result.total_queries,
        "research_questions": research_qs,
        "response_preview": response_text[:300],
        "confidence": result.spacetime.confidence,
    }


async def main():
    banner("Model Benchmark: llama3.2:3b vs qwen2.5:14b")

    # Verify both models available
    for m in MODELS:
        llm = OllamaLLM(model=m)
        if not llm.is_available():
            print(f"  WARNING: {m} not available, skipping")
            MODELS.remove(m)
    print(f"  Models: {MODELS}")

    # ── Part 1: Raw LLM benchmarks ──
    banner("Part 1: Raw LLM Benchmarks")

    all_results = []
    for bench in BENCHMARKS:
        print(f"\n  --- {bench['name']} ---")
        for model in MODELS:
            r = await run_benchmark(model, bench)
            all_results.append(r)
            print(f"  [{model:16s}] {r['elapsed_s']:5.1f}s | {r['tok_per_sec']:5.1f} tok/s | {r['completion_tokens']:3d} tokens | {r['response_len']:4d} chars")

    # Show responses side by side
    banner("Part 2: Response Comparison")
    for bench in BENCHMARKS:
        print(f"\n  --- {bench['name']} ---")
        for model in MODELS:
            r = next(x for x in all_results if x["model"] == model and x["bench"] == bench["name"])
            print(f"\n  [{model}]:")
            # Indent response
            for line in r["response"][:400].split("\n"):
                print(f"    {line}")
            if len(r["response"]) > 400:
                print(f"    ...")

    # ── Part 3: Agentic RESEARCH pipeline ──
    banner("Part 3: Agentic RESEARCH Pipeline")
    for model in MODELS:
        print(f"\n  [{model}] Running RESEARCH mode...")
        try:
            r = await run_agentic_research(model)
            print(f"    Time: {r['elapsed_s']:.1f}s")
            print(f"    Queries: {r['total_queries']}")
            print(f"    Confidence: {r['confidence']:.2f}")
            print(f"    Research Qs:")
            for i, q in enumerate(r["research_questions"]):
                print(f"      {i+1}. {q[:90]}")
            print(f"    Response: {r['response_preview'][:200]}...")
        except Exception as e:
            print(f"    FAILED: {e}")

    # ── Summary table ──
    banner("Summary")
    print(f"  {'Benchmark':<22s} | {'Model':<16s} | {'Time':>6s} | {'tok/s':>6s} | {'Tokens':>6s}")
    print(f"  {'-'*22}-+-{'-'*16}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for r in all_results:
        print(f"  {r['bench']:<22s} | {r['model']:<16s} | {r['elapsed_s']:5.1f}s | {r['tok_per_sec']:5.1f} | {r['completion_tokens']:>6d}")


if __name__ == "__main__":
    asyncio.run(main())
