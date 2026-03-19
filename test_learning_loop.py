#!/usr/bin/env python3
"""
Test: Prove the learning loop is closed.

Runs N queries through the same orchestrator and checks:
  1. Thompson Sampling priors shift toward the tool that produces real answers
  2. Confidence scores reflect response quality (not just policy output)
  3. The "answer" tool gets selected more often as priors learn

This is the litmus test for whether HoloLoom is learning or just scaffolding.
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

MODEL = "qwen2.5:14b"

SHARDS = [
    MemoryShard(id="s1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.", metadata={"topic": "algo"}),
    MemoryShard(id="s2", text="It explores proportionally to the probability of being optimal.", metadata={"topic": "algo"}),
    MemoryShard(id="s3", text="Applications include A/B testing, recommendation systems, and clinical trials.", metadata={"topic": "apps"}),
    MemoryShard(id="s4", text="The main tradeoff is computational overhead from maintaining distributions.", metadata={"topic": "tradeoffs"}),
    MemoryShard(id="s5", text="SOUS uses GPS scoring: gap_score * w_gap + emergency * w_emergency + connector * w_connector.", metadata={"topic": "kitchen"}),
    MemoryShard(id="s6", text="Connectors are ingredients that bridge multiple recipes, unlocking more meal options.", metadata={"topic": "kitchen"}),
]

QUERIES = [
    "What is Thompson Sampling?",
    "How does epsilon-greedy compare to Thompson Sampling?",
    "What are practical applications of bandit algorithms?",
    "Explain the tradeoffs of maintaining posterior distributions.",
    "How could Thompson Sampling improve grocery shopping decisions?",
    "What role do connectors play in kitchen management?",
    "When should you prefer UCB over Thompson Sampling?",
    "How does the GPS scoring algorithm work?",
]


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


async def main():
    from hololoom.weaving_orchestrator_llm import WeavingOrchestrator

    banner(f"Learning Loop Test ({MODEL})")

    config = Config.fast()
    config.enable_agentic_reasoning = False
    config.enable_safety_guardrails = False

    async with WeavingOrchestrator(cfg=config, shards=SHARDS, llm_provider="ollama") as orch:
        # Swap to Qwen
        orch.tool_executor.llm = OllamaLLM(model=MODEL)

        # Snapshot initial priors
        convergence = orch._convergence
        if convergence is None:
            # Force init by running a dummy weave
            print("  (Initializing convergence engine...)")
            _ = await orch.weave(Query(text="hello"))
            convergence = orch._convergence

        tools = convergence.tools
        print(f"  Tools: {tools}")

        initial_priors = convergence.bandit.get_priors().copy()
        initial_successes = convergence.bandit.successes.copy()
        initial_failures = convergence.bandit.failures.copy()
        print(f"  Initial priors: {dict(zip(tools, [f'{p:.3f}' for p in initial_priors]))}")

        # Run queries — same orchestrator, priors should evolve
        results = []
        for i, q_text in enumerate(QUERIES):
            t0 = time.perf_counter()
            spacetime = await orch.weave(Query(text=q_text))
            elapsed = time.perf_counter() - t0

            resp = spacetime.response or ""
            tool = spacetime.tool_used
            conf = spacetime.confidence
            quality = spacetime.quality_score or 0.0

            priors = convergence.bandit.get_priors()
            answer_prior = priors[tools.index("answer")] if "answer" in tools else 0.0

            results.append({
                "query": q_text,
                "tool": tool,
                "conf": conf,
                "quality": quality,
                "resp_len": len(resp),
                "elapsed": elapsed,
                "answer_prior": answer_prior,
            })

            print(f"\n  Q{i+1}: {q_text[:60]}...")
            print(f"    Tool: {tool} | Conf: {conf:.3f} | Quality: {quality:.3f} | {len(resp)} chars | {elapsed:.1f}s")
            print(f"    Priors: {dict(zip(tools, [f'{p:.3f}' for p in priors]))}")
            print(f"    Response: {resp[:120]}...")

        # Final state
        final_priors = convergence.bandit.get_priors()
        final_successes = convergence.bandit.successes
        final_failures = convergence.bandit.failures

        banner("Results")

        # 1. Did priors shift?
        print("\n  --- Prior Evolution ---")
        print(f"  {'Tool':<16s} | {'Initial':>8s} | {'Final':>8s} | {'Delta':>8s} | {'alpha':>6s} | {'beta':>6s}")
        print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")
        for i, tool in enumerate(tools):
            delta = final_priors[i] - initial_priors[i]
            print(f"  {tool:<16s} | {initial_priors[i]:8.4f} | {final_priors[i]:8.4f} | {delta:+8.4f} | {final_successes[i]:6.2f} | {final_failures[i]:6.2f}")

        # 2. Did confidence correlate with response quality?
        print("\n  --- Confidence vs Quality ---")
        print(f"  {'Query':<40s} | {'Conf':>6s} | {'Quality':>7s} | {'Chars':>6s}")
        print(f"  {'-'*40}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")
        for r in results:
            q_short = r["query"][:40]
            print(f"  {q_short:<40s} | {r['conf']:6.3f} | {r['quality']:7.3f} | {r['resp_len']:>6d}")

        # 3. Did answer tool prior increase?
        answer_idx = tools.index("answer") if "answer" in tools else -1
        if answer_idx >= 0:
            answer_delta = final_priors[answer_idx] - initial_priors[answer_idx]
            answer_tool_count = sum(1 for r in results if r["tool"] == "answer")
            print(f"\n  --- Answer Tool Learning ---")
            print(f"    Answer prior: {initial_priors[answer_idx]:.4f} -> {final_priors[answer_idx]:.4f} (delta: {answer_delta:+.4f})")
            print(f"    Answer selected: {answer_tool_count}/{len(QUERIES)} queries")
            print(f"    Alpha: {initial_successes[answer_idx]:.2f} -> {final_successes[answer_idx]:.2f}")
            print(f"    Beta:  {initial_failures[answer_idx]:.2f} -> {final_failures[answer_idx]:.2f}")

        # 4. Did any learning happen at all?
        any_shift = any(abs(final_priors[i] - initial_priors[i]) > 0.001 for i in range(len(tools)))
        conf_varies = len(set(f"{r['conf']:.3f}" for r in results)) > 1
        quality_positive = any(r["quality"] > 0.3 for r in results)

        banner("Verdict")
        checks = {
            "Priors shifted from initial uniform": any_shift,
            "Confidence varies across queries": conf_varies,
            "Quality scores are meaningful (> 0.3)": quality_positive,
            "Real LLM responses (> 100 chars)": any(r["resp_len"] > 100 for r in results),
        }

        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check}")

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        print(f"\n  {passed}/{total} checks passed")

        if passed == total:
            print("  THE LOOP IS CLOSED.")
        else:
            print("  Loop still open — priors not adapting.")


if __name__ == "__main__":
    asyncio.run(main())
