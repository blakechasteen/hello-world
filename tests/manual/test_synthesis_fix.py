#!/usr/bin/env python3
"""Quick test: verify LLM synthesis fix for VERIFY/RESEARCH/PLAN_EXECUTE modes."""
import asyncio
import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.WARNING)

from hololoom.awareness.llm_integration import OllamaLLM
from hololoom.documentation.types import Query, MemoryShard
from hololoom.config import Config

MODEL = "qwen2.5:14b"
STUB_PHRASES = {"Calculation completed", "Search results based on query",
                "Successfully wrote to Notion database", "No response"}

SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.", metadata={"topic": "algorithms"}),
    MemoryShard(id="s2", text="It explores proportionally to the probability of being optimal.", metadata={"topic": "algorithms"}),
    MemoryShard(id="s3", text="Applications include A/B testing, recommendation systems, and clinical trials.", metadata={"topic": "applications"}),
    MemoryShard(id="s4", text="The main tradeoff is computational overhead from maintaining distributions.", metadata={"topic": "tradeoffs"}),
]


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


def patch_llm(orchestrator):
    """Swap LLM to Qwen on all relevant paths."""
    llm = OllamaLLM(model=MODEL)
    if orchestrator.llm is not None or True:
        orchestrator.llm = llm
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = llm


async def main():
    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    banner(f"Synthesis Fix Test ({MODEL})")

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    query = Query(text="When should I use Thompson Sampling vs epsilon-greedy for optimizing meal prep rotation?")

    modes = [
        ("DIRECT", ReasoningMode.DIRECT, 1),
        ("VERIFY", ReasoningMode.VERIFY, 3),
        ("RESEARCH", ReasoningMode.RESEARCH, 3),
        ("PLAN_EXECUTE", ReasoningMode.PLAN_EXECUTE, 4),
    ]

    results = {}
    for label, mode, steps in modes:
        print(f"\n  --- {label} ---")
        orchestrator = await create_agentic_orchestrator(
            config=config, shards=SHARDS,
            enable_verification=True, enable_goal_tracking=True
        )
        patch_llm(orchestrator)

        t0 = time.perf_counter()
        result = await orchestrator.reason(query=query, mode=mode, max_steps=steps)
        elapsed = time.perf_counter() - t0

        resp = result.spacetime.response or ""
        is_stub = resp.strip() in STUB_PHRASES or len(resp) < 50
        status = "STUB" if is_stub else "REAL"

        print(f"    Time: {elapsed:.1f}s | Resp: {len(resp)} chars | [{status}]")
        print(f"    Preview: {resp[:200]}...")
        results[label] = {"resp_len": len(resp), "is_stub": is_stub, "time": elapsed}

    banner("Results")
    print(f"  {'Mode':<14s} | {'Time':>6s} | {'Chars':>6s} | Status")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*6}-+--------")
    for label, r in results.items():
        status = "STUB" if r["is_stub"] else "REAL"
        print(f"  {label:<14s} | {r['time']:5.1f}s | {r['resp_len']:>6d} | {status}")

    stubs = sum(1 for r in results.values() if r["is_stub"])
    print(f"\n  {4 - stubs}/4 modes producing real LLM responses")
    if stubs == 0:
        print("  ALL MODES FIXED!")
    else:
        print(f"  {stubs} modes still returning stubs")


if __name__ == "__main__":
    asyncio.run(main())
