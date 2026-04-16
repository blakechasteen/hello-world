#!/usr/bin/env python3
"""
Benchmark: Qwen3 30B MoE vs Qwen 2.5 14B Dense

Compares:
  1. Tokens/sec (MoE should be ~2-4x faster with only 3B active params)
  2. Response quality (30B knowledge base should produce richer answers)
  3. Contextual bandit behavior (does MoE quality shift tool preferences?)

MoE architecture: 30B total params, ~3B active per token.
On CPU (i7-8700K, 32GB RAM):
  - Qwen 2.5 14B dense: ~15 tok/s (all 14B active)
  - Qwen3 30B MoE: expected ~30-40 tok/s (only 3B active)
"""
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
from hololoom.weaving_orchestrator import WeavingOrchestrator

SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.", metadata={"topic": "algo"}),
    MemoryShard(id="s2", text="It explores proportionally to the probability of being optimal.", metadata={"topic": "algo"}),
    MemoryShard(id="s3", text="Applications include A/B testing, recommendation systems, and clinical trials.", metadata={"topic": "apps"}),
    MemoryShard(id="s4", text="The main tradeoff is computational overhead from maintaining distributions.", metadata={"topic": "tradeoffs"}),
    MemoryShard(id="s5", text="SOUS uses GPS scoring: gap_score * w_gap + emergency * w_emergency + connector * w_connector.", metadata={"topic": "kitchen"}),
    MemoryShard(id="s6", text="Connectors are ingredients that bridge multiple recipes, unlocking more meal options.", metadata={"topic": "kitchen"}),
]

# Mix of categories to exercise contextual bandits
QUERIES = [
    ("What is Thompson Sampling?", "factual"),
    ("Compare Thompson Sampling vs epsilon-greedy for grocery optimization.", "analytical"),
    ("What are practical applications of bandit algorithms beyond A/B testing?", "exploratory"),
    ("Explain the GPS scoring formula step by step.", "structured"),
    ("How do connectors bridge recipes in kitchen management?", "factual"),
    ("Which is better for meal planning: UCB or Thompson Sampling?", "analytical"),
]


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def benchmark_model(model_name: str, label: str):
    """Run benchmark queries through a model and collect metrics."""
    banner(f"{label} ({model_name})")

    config = Config.fast()
    config.enable_agentic_reasoning = False
    config.enable_safety_guardrails = False

    results = []

    async with WeavingOrchestrator(cfg=config, shards=SHARDS) as orch:
        orch.tool_executor.llm = OllamaLLM(model=model_name)

        # Warmup (also inits convergence engine)
        print("  Warming up...")
        _ = await orch.weave(Query(text="hello"))

        for i, (q_text, expected_ctx) in enumerate(QUERIES):
            t0 = time.perf_counter()
            spacetime = await orch.weave(Query(text=q_text))
            elapsed = time.perf_counter() - t0

            resp = spacetime.response or ""
            tool = spacetime.tool_used
            quality = spacetime.quality_score or 0.0

            # Estimate tokens (rough: 1 token ~= 4 chars)
            est_tokens = len(resp) / 4
            tps = est_tokens / elapsed if elapsed > 0 else 0

            results.append({
                "query": q_text,
                "ctx": expected_ctx,
                "tool": tool,
                "quality": quality,
                "chars": len(resp),
                "tokens_est": int(est_tokens),
                "elapsed": elapsed,
                "tps": tps,
            })

            print(f"  Q{i+1}: {q_text[:50]}...")
            print(f"    {tool} | {quality:.3f} | {len(resp)} chars | {elapsed:.1f}s | ~{tps:.0f} tok/s")

    return results


async def main():
    models = [
        ("qwen2.5:14b", "Dense 14B (baseline)"),
        ("qwen3:30b", "MoE 30B-A3B"),
    ]

    all_results = {}
    for model, label in models:
        try:
            all_results[label] = await benchmark_model(model, label)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            all_results[label] = None

    # ================================================================
    # Head-to-head comparison
    # ================================================================
    banner("Head-to-Head Comparison")

    labels = [l for l, r in all_results.items() if r]
    if len(labels) < 2:
        print("  Not enough models to compare. Done.")
        return

    print(f"\n  {'Query':<40s}", end="")
    for label in labels:
        print(f" | {label:>20s}", end="")
    print()
    print(f"  {'-'*40}", end="")
    for _ in labels:
        print(f"-+-{'-'*20}", end="")
    print()

    for i, (q_text, _) in enumerate(QUERIES):
        print(f"  {q_text[:40]:<40s}", end="")
        for label in labels:
            r = all_results[label][i]
            print(f" | {r['tps']:5.0f} t/s {r['chars']:>5d}ch", end="")
        print()

    # Summary stats
    print()
    for label in labels:
        results = all_results[label]
        avg_tps = sum(r["tps"] for r in results) / len(results)
        avg_quality = sum(r["quality"] for r in results) / len(results)
        avg_chars = sum(r["chars"] for r in results) / len(results)
        total_time = sum(r["elapsed"] for r in results)

        print(f"  {label}:")
        print(f"    Avg tok/s: {avg_tps:.0f}")
        print(f"    Avg quality: {avg_quality:.3f}")
        print(f"    Avg response: {avg_chars:.0f} chars")
        print(f"    Total time: {total_time:.1f}s")

    # Speed ratio
    if len(labels) == 2:
        tps_0 = sum(r["tps"] for r in all_results[labels[0]]) / len(all_results[labels[0]])
        tps_1 = sum(r["tps"] for r in all_results[labels[1]]) / len(all_results[labels[1]])
        if tps_0 > 0:
            ratio = tps_1 / tps_0
            faster = labels[1] if ratio > 1 else labels[0]
            print(f"\n  Speed ratio: {labels[1]} is {ratio:.1f}x {'faster' if ratio > 1 else 'slower'} than {labels[0]}")


if __name__ == "__main__":
    asyncio.run(main())
