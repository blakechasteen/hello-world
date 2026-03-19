#!/usr/bin/env python3
"""
Test: Contextual Thompson Sampling — do different query types learn different tool preferences?

Runs 16 queries (4 per category) through the same orchestrator and checks:
  1. Each category develops its OWN priors (not one global winner)
  2. Tools are selected context-sensitively
  3. Per-category alpha/beta evolve independently

Categories:
  factual     — "What is X?" → answer should win
  analytical  — "Compare X vs Y" → calc should win
  exploratory — "What are applications..." → search should win
  structured  — "Explain/summarize..." → notion_write should win
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

MODEL = "qwen2.5:14b"

SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.", metadata={"topic": "algo"}),
    MemoryShard(id="s2", text="It explores proportionally to the probability of being optimal.", metadata={"topic": "algo"}),
    MemoryShard(id="s3", text="Applications include A/B testing, recommendation systems, and clinical trials.", metadata={"topic": "apps"}),
    MemoryShard(id="s4", text="The main tradeoff is computational overhead from maintaining distributions.", metadata={"topic": "tradeoffs"}),
    MemoryShard(id="s5", text="SOUS uses GPS scoring: gap_score * w_gap + emergency * w_emergency + connector * w_connector.", metadata={"topic": "kitchen"}),
    MemoryShard(id="s6", text="Connectors are ingredients that bridge multiple recipes, unlocking more meal options.", metadata={"topic": "kitchen"}),
    MemoryShard(id="s7", text="Epsilon-greedy explores uniformly at random, while Thompson Sampling explores proportionally to optimality probability.", metadata={"topic": "algo"}),
    MemoryShard(id="s8", text="UCB algorithms use confidence bounds rather than sampling from posterior distributions.", metadata={"topic": "algo"}),
]

# 4 queries per category, interleaved so the bandit sees variety
QUERIES = [
    # factual — direct knowledge
    ("What is Thompson Sampling?", "factual"),
    ("What is the GPS scoring formula?", "factual"),
    ("What is a connector ingredient?", "factual"),
    ("What is a Beta distribution?", "factual"),

    # analytical — comparisons/tradeoffs
    ("How does epsilon-greedy compare to Thompson Sampling?", "analytical"),
    ("Compare UCB versus Thompson Sampling for meal planning.", "analytical"),
    ("What are the tradeoffs of maintaining posterior distributions vs point estimates?", "analytical"),
    ("Which is better for A/B testing: Thompson Sampling or frequentist methods?", "analytical"),

    # exploratory — broad investigation
    ("What are practical applications of bandit algorithms?", "exploratory"),
    ("How could Thompson Sampling improve grocery shopping decisions?", "exploratory"),
    ("What role do connectors play in kitchen management?", "exploratory"),
    ("When should you use multi-armed bandits in production systems?", "exploratory"),

    # structured — organized output
    ("Explain how Thompson Sampling works step by step.", "structured"),
    ("Summarize the relationship between GPS scoring and connectors.", "structured"),
    ("Describe the complete workflow of a Bayesian bandit system.", "structured"),
    ("Explain the multi-armed bandit problem and its solutions.", "structured"),
]


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def main():
    banner(f"Contextual Bandits Test ({MODEL})")

    config = Config.fast()
    config.enable_agentic_reasoning = False
    config.enable_safety_guardrails = False

    async with WeavingOrchestrator(cfg=config, shards=SHARDS) as orch:
        # Swap to Qwen
        orch.tool_executor.llm = OllamaLLM(model=MODEL)

        # Force init
        print("  (Initializing convergence engine...)")
        _ = await orch.weave(Query(text="hello"))
        convergence = orch._convergence

        tools = convergence.tools
        print(f"  Tools: {tools}")
        print(f"  Contexts: {convergence.contexts}")

        # Verify classifier works
        print("\n  --- Query Classification Check ---")
        for q_text, expected_ctx in QUERIES[:4] + QUERIES[4:5] + QUERIES[8:9] + QUERIES[12:13]:
            actual = orch._classify_query_intent(q_text)
            match = "OK" if actual == expected_ctx else f"MISMATCH (expected {expected_ctx})"
            print(f"    [{actual:<12s}] {q_text[:50]}... {match}")

        # Run all queries
        results = []
        for i, (q_text, expected_ctx) in enumerate(QUERIES):
            t0 = time.perf_counter()
            spacetime = await orch.weave(Query(text=q_text))
            elapsed = time.perf_counter() - t0

            resp = spacetime.response or ""
            tool = spacetime.tool_used
            conf = spacetime.confidence
            quality = spacetime.quality_score or 0.0
            actual_ctx = orch._classify_query_intent(q_text)

            results.append({
                "query": q_text,
                "expected_ctx": expected_ctx,
                "actual_ctx": actual_ctx,
                "tool": tool,
                "conf": conf,
                "quality": quality,
                "resp_len": len(resp),
                "elapsed": elapsed,
            })

            print(f"\n  Q{i+1}: {q_text[:55]}...")
            print(f"    Ctx: {actual_ctx} | Tool: {tool} | Quality: {quality:.3f} | {len(resp)} chars | {elapsed:.1f}s")

        # ================================================================
        # Results
        # ================================================================
        banner("Contextual Prior Evolution")

        # Show per-context priors
        all_stats = convergence.bandit.get_all_statistics()
        for ctx_name in convergence.contexts:
            stats = all_stats.get(ctx_name, {})
            priors = stats.get("tool_priors", [0.5] * len(tools))
            successes = stats.get("success_counts", [1.0] * len(tools))
            failures = stats.get("failure_counts", [1.0] * len(tools))

            print(f"\n  --- {ctx_name.upper()} ---")
            print(f"  {'Tool':<16s} | {'Prior':>7s} | {'alpha':>7s} | {'beta':>7s}")
            print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
            for j, tool in enumerate(tools):
                print(f"  {tool:<16s} | {priors[j]:7.4f} | {successes[j]:7.2f} | {failures[j]:7.2f}")

        # Global (fallback) priors
        g_stats = all_stats.get("global", {})
        g_priors = g_stats.get("tool_priors", [])
        if g_priors:
            print(f"\n  --- GLOBAL (fallback) ---")
            print(f"  {'Tool':<16s} | {'Prior':>7s}")
            print(f"  {'-'*16}-+-{'-'*7}")
            for j, tool in enumerate(tools):
                print(f"  {tool:<16s} | {g_priors[j]:7.4f}")

        # Per-category tool selection summary
        banner("Tool Selection by Category")
        from collections import Counter
        for ctx in convergence.contexts:
            ctx_results = [r for r in results if r["actual_ctx"] == ctx]
            if not ctx_results:
                continue
            tool_counts = Counter(r["tool"] for r in ctx_results)
            avg_quality = sum(r["quality"] for r in ctx_results) / len(ctx_results)
            print(f"\n  {ctx.upper()} ({len(ctx_results)} queries, avg quality: {avg_quality:.3f})")
            for tool, count in tool_counts.most_common():
                bar = "#" * (count * 4)
                print(f"    {tool:<14s} {bar} ({count})")

        # ================================================================
        # Verdict
        # ================================================================
        banner("Verdict")

        # Check 1: Classification accuracy
        classify_correct = sum(1 for r in results if r["actual_ctx"] == r["expected_ctx"])
        classify_pct = classify_correct / len(results)
        print(f"  [{'PASS' if classify_pct >= 0.75 else 'FAIL'}] Query classification: {classify_correct}/{len(results)} correct ({classify_pct:.0%})")

        # Check 2: Priors diverge across contexts (not all identical)
        ctx_top_tools = {}
        for ctx in convergence.contexts:
            stats = all_stats.get(ctx, {})
            priors = stats.get("tool_priors", [0.5] * len(tools))
            top_idx = max(range(len(priors)), key=lambda i: priors[i])
            ctx_top_tools[ctx] = tools[top_idx]

        unique_tops = len(set(ctx_top_tools.values()))
        print(f"  [{'PASS' if unique_tops >= 2 else 'FAIL'}] Contextual divergence: {unique_tops} different top tools across {len(convergence.contexts)} contexts")
        for ctx, top in ctx_top_tools.items():
            print(f"    {ctx}: {top}")

        # Check 3: All queries produce real responses
        all_real = all(r["resp_len"] > 50 for r in results)
        print(f"  [{'PASS' if all_real else 'FAIL'}] All responses are real (>50 chars)")

        # Check 4: Quality scores are meaningful
        avg_quality = sum(r["quality"] for r in results) / len(results)
        print(f"  [{'PASS' if avg_quality > 0.5 else 'FAIL'}] Average quality: {avg_quality:.3f}")

        # Check 5: Priors actually shifted (learning happened)
        any_learning = False
        for ctx in convergence.contexts:
            stats = all_stats.get(ctx, {})
            priors = stats.get("tool_priors", [0.5] * len(tools))
            if any(abs(p - 0.5) > 0.01 for p in priors):
                any_learning = True
                break
        print(f"  [{'PASS' if any_learning else 'FAIL'}] Contextual priors shifted from uniform")

        checks_passed = sum([
            classify_pct >= 0.75,
            unique_tops >= 2,
            all_real,
            avg_quality > 0.5,
            any_learning
        ])
        print(f"\n  {checks_passed}/5 checks passed")

        if checks_passed == 5:
            print("  CONTEXTUAL BANDITS ARE LEARNING.")
        elif checks_passed >= 3:
            print("  Partial learning — context helps but not fully differentiated yet.")
        else:
            print("  Not learning contextually.")


if __name__ == "__main__":
    asyncio.run(main())
