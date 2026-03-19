#!/usr/bin/env python3
"""
Live integration test: v2 Full Feedback Loop

Runs the complete self-optimization cycle:
  Phase A: Weave 5 queries → collect audit data
  Phase B: Run training pipeline → extract optimized configs
  Phase C: Weave same queries with optimized configs → compare

This test requires Ollama running with a model available.

Run: python test_v2_live.py [--model MODEL]
"""
import asyncio
import sys
import time
import logging

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
for name in ['sentence_transformers', 'httpx', 'httpcore', 'urllib3',
             'hololoom.alignment', 'hololoom.loom', 'hololoom.chrono',
             'hololoom.resonance', 'hololoom.warp', 'hololoom.convergence',
             'hololoom.reflection', 'hololoom.embedding', 'hololoom.motif',
             'hololoom.performance', 'hololoom.fabric']:
    logging.getLogger(name).setLevel(logging.WARNING)

from hololoom.memory.lite_bus import LiteMemoryBus
from hololoom.memory.bus import MemoryItem
from hololoom.memory.v2 import (
    Navigator, PPRConfig, MatryoshkaOrchestrator,
    ShellType, DEFAULT_SHELLS,
    TrainingPipeline, AuditCollector, TrainingExample,
    RewardJudge,
)


# Parse CLI
MODEL = "qwen2.5:14b"
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]


def banner(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


def table(headers, rows, widths):
    fmt = "  " + "".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "-" * sum(widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


async def build_bus() -> LiteMemoryBus:
    """Build a bus with enough data for meaningful PPR traversal."""
    bus = LiteMemoryBus()
    await bus.initialize()

    ids = {}
    entities = [
        ("Thompson Sampling", "algorithm", 0.95),
        ("Epsilon Greedy", "algorithm", 0.85),
        ("UCB1", "algorithm", 0.88),
        ("Contextual Bandits", "framework", 0.92),
        ("Multi-Armed Bandit", "concept", 0.97),
        ("Bayesian Optimization", "technique", 0.90),
        ("LinUCB", "algorithm", 0.87),
        ("Regret Analysis", "concept", 0.82),
    ]
    for name, etype, imp in entities:
        ids[name] = await bus.store(MemoryItem(
            content=name, memory_type="entity",
            properties={"type": etype}, importance=imp,
        ))
        await bus.add_alias(name.lower(), ids[name])

    await bus.add_alias("ts", ids["Thompson Sampling"])
    await bus.add_alias("mab", ids["Multi-Armed Bandit"])
    await bus.add_alias("ucb", ids["UCB1"])
    await bus.add_alias("bandit", ids["Multi-Armed Bandit"])

    facts = [
        ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation.", ["Thompson Sampling"]),
        ("Thompson Sampling samples from the posterior distribution of each arm's reward.", ["Thompson Sampling"]),
        ("Thompson Sampling converges to the optimal arm with probability 1.", ["Thompson Sampling"]),
        ("Epsilon Greedy explores with probability epsilon and exploits otherwise.", ["Epsilon Greedy"]),
        ("Epsilon Greedy is simple to implement but has linear regret.", ["Epsilon Greedy"]),
        ("UCB1 computes upper confidence bounds using the Hoeffding inequality.", ["UCB1"]),
        ("UCB1 achieves logarithmic regret without knowing the time horizon.", ["UCB1"]),
        ("Thompson Sampling outperforms Epsilon Greedy when the action space is large.", ["Thompson Sampling", "Epsilon Greedy"]),
        ("Contextual bandits use feature vectors to make per-decision adaptations.", ["Contextual Bandits", "Multi-Armed Bandit"]),
        ("LinUCB and Thompson Sampling are popular contextual bandit algorithms.", ["Contextual Bandits", "Thompson Sampling", "LinUCB"]),
        ("The multi-armed bandit problem models the exploration-exploitation tradeoff.", ["Multi-Armed Bandit"]),
        ("Applications include A/B testing, recommendation systems, and clinical trials.", ["Multi-Armed Bandit"]),
        ("Bayesian Optimization uses a surrogate model to guide expensive function evaluation.", ["Bayesian Optimization"]),
        ("Thompson Sampling can be seen as a special case of Bayesian Optimization for bandits.", ["Thompson Sampling", "Bayesian Optimization"]),
        ("Regret measures the cumulative loss from not always choosing the best arm.", ["Regret Analysis", "Multi-Armed Bandit"]),
        ("UCB1's regret bound is O(sqrt(KT log T)) where K is the number of arms.", ["UCB1", "Regret Analysis"]),
        ("LinUCB extends UCB to the contextual setting with linear reward models.", ["LinUCB", "UCB1", "Contextual Bandits"]),
    ]
    for text, entity_names in facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=[ids[n] for n in entity_names],
            importance=0.85,
        ))

    # Sparse/isolated facts — no entity links, low importance
    # These should pull down confidence for queries that retrieve them
    sparse_facts = [
        "Some researchers prefer frequentist approaches over Bayesian methods.",
        "The term 'bandit' comes from one-armed bandit slot machines.",
        "Online learning is related to but distinct from bandit problems.",
    ]
    for text in sparse_facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            importance=0.3,
        ))

    edges = [
        ("Thompson Sampling", "Epsilon Greedy", "COMPARED_TO"),
        ("Thompson Sampling", "UCB1", "COMPARED_TO"),
        ("Epsilon Greedy", "UCB1", "COMPARED_TO"),
        ("Thompson Sampling", "Multi-Armed Bandit", "SOLVES"),
        ("Epsilon Greedy", "Multi-Armed Bandit", "SOLVES"),
        ("UCB1", "Multi-Armed Bandit", "SOLVES"),
        ("Contextual Bandits", "Multi-Armed Bandit", "EXTENDS"),
        ("Thompson Sampling", "Contextual Bandits", "USED_IN"),
        ("Thompson Sampling", "Bayesian Optimization", "RELATED_TO"),
        ("LinUCB", "Contextual Bandits", "USED_IN"),
        ("LinUCB", "UCB1", "EXTENDS"),
        ("Regret Analysis", "Multi-Armed Bandit", "ANALYZES"),
    ]
    for src, dst, rel in edges:
        await bus.store_edge(ids[src], ids[dst], rel)

    status = bus.status()
    print(f"  Bus: {status['item_count']} items, {status['edge_count']} edges, {status['alias_count']} aliases")
    return bus


QUERIES = [
    "Compare Thompson Sampling vs epsilon-greedy for exploration",
    "How does UCB1 achieve logarithmic regret?",
    "What are contextual bandits and when should I use them?",
    "Explain the connection between Thompson Sampling and Bayesian Optimization",
    "What is regret in multi-armed bandits?",
]


async def run_queries(bus, llm_fn, shell_configs=None, label=""):
    """Run all queries through the Matryoshka loop, return metrics."""
    nav = Navigator.from_lite_bus(bus, config=PPRConfig(max_results=30))
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=llm_fn,
        shell_configs=shell_configs,
    )
    judge = RewardJudge(llm_fn=llm_fn)

    results = []
    for i, q in enumerate(QUERIES):
        t0 = time.perf_counter()
        try:
            r = await orch.run(q)
            elapsed = time.perf_counter() - t0

            # Real composite reward via RewardJudge
            final = r.shells_executed[-1]
            reward = await judge.score(
                query=q,
                response=r.response,
                confidence=final.confidence.combined,
                context_items=final.context_block.items_packed,
            )

            results.append({
                "query": q[:50],
                "shells": r.shell_count,
                "confidence": r.final_confidence.combined,
                "decision": r.final_confidence.decision,
                "tokens": r.total_tokens,
                "resp_len": len(r.response),
                "time": elapsed,
                "response": r.response,
                "shell_path": " > ".join(s.shell_type.value for s in r.shells_executed),
                "reward": reward,
            })

            # Emit audit entry to bus for training pipeline consumption
            bus._audit.append({
                "type": "v2_weave_cycle",
                "query": q,
                "tool_used": f"matryoshka:{r.final_shell.value}",
                "response_length": len(r.response),
                "v2_confidence": final.confidence.combined,
                "v2_decision": final.confidence.decision,
                "v2_structural": final.confidence.structural.score,
                "v2_narrative": final.confidence.narrative.score,
                "context_tokens": final.context_block.token_count,
                "context_items": final.context_block.items_packed,
                "reward": reward,
                "duration_ms": elapsed * 1000,
            })

            status = f"OK ({r.final_shell.value}) reward={reward:.3f}"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append({
                "query": q[:50],
                "shells": 0,
                "confidence": 0,
                "decision": "error",
                "tokens": 0,
                "resp_len": 0,
                "time": elapsed,
                "response": "",
                "shell_path": "error",
                "reward": 0.0,
            })
            status = f"ERR: {str(e)[:40]}"

        print(f"  [{label} {i+1}/{len(QUERIES)}] {status} ({elapsed:.1f}s)")

    return results


async def main():
    banner("Live Integration: Self-Optimizing Feedback Loop")
    print(f"  Model: {MODEL}")
    print(f"  Queries: {len(QUERIES)}")

    bus = await build_bus()

    # ── LLM adapter ──────────────────────────────────────────
    from hololoom.awareness.llm_integration import OllamaLLM
    llm = OllamaLLM(model=MODEL)

    async def llm_fn(prompt: str, max_tokens: int = 3000) -> str:
        resp = await llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        return resp.content

    # ================================================================
    # PHASE A: Baseline — weave with default configs
    # ================================================================
    banner("Phase A: Baseline (default configs)")
    audit_before = len(bus._audit)

    t0 = time.perf_counter()
    baseline = await run_queries(bus, llm_fn, shell_configs=None, label="A")
    phase_a_time = time.perf_counter() - t0

    audit_after = len(bus._audit)
    print(f"\n  Audit entries: {audit_before} -> {audit_after} (+{audit_after - audit_before})")
    print(f"  Phase A time: {phase_a_time:.1f}s")

    # Show baseline table
    print()
    table(
        ["Query", "Path", "Conf", "Reward", "Tokens", "Resp", "Time"],
        [
            [r["query"][:35], r["shell_path"], f"{r['confidence']:.2f}",
             f"{r['reward']:.3f}", r["tokens"], r["resp_len"], f"{r['time']:.1f}s"]
            for r in baseline
        ],
        [37, 18, 7, 8, 8, 7, 8],
    )

    # ================================================================
    # PHASE B: Train — run training pipeline on collected audit
    # ================================================================
    banner("Phase B: Training Pipeline")

    # Collect audit summary
    collector = AuditCollector(bus)
    summary = collector.collect()
    print(f"  Audit entries: {summary.total}")
    print(f"  Avg confidence: {summary.avg_confidence:.3f}")
    print(f"  Avg reward: {summary.avg_reward:.3f}")
    print(f"  Good rate: {summary.good_rate:.0%}")
    print(f"  Poor rate: {summary.poor_rate:.0%}")
    print(f"  By shell: {dict(summary.by_shell)}")

    # Run training (bandits only — fast, no LLM calls for prompt refinement)
    pipeline = TrainingPipeline(
        llm_fn=None,  # Bandits only for speed
        enable_prompt_refinement=False,
        enable_signature_optimization=False,
    )

    # Pre-train from collected audit
    pipeline._train_bandits(summary.examples)
    trained_configs = pipeline.get_configs()

    print(f"\n  Optimized configs vs defaults:")
    for shell_type in [ShellType.PRIME, ShellType.VERIFY, ShellType.FLAG]:
        default = DEFAULT_SHELLS[shell_type]
        trained = trained_configs[shell_type]
        changes = []
        if abs(trained.ppr_alpha - default.ppr_alpha) > 0.001:
            changes.append(f"alpha {default.ppr_alpha:.2f} -> {trained.ppr_alpha:.2f}")
        if trained.token_budget != default.token_budget:
            changes.append(f"budget {default.token_budget} -> {trained.token_budget}")
        change_str = ", ".join(changes) if changes else "(unchanged)"
        print(f"    {shell_type.value:8s}: {change_str}")

    # Show bandit posteriors
    print(f"\n  Bandit posteriors:")
    for name, bandit in pipeline.bandits.items():
        best_idx, best_val = bandit.best()
        total_pulls = sum(a.pulls for a in bandit.arms)
        if total_pulls > 0:
            print(f"    {name:30s}: best={best_val:.3f} (mean={bandit.arms[best_idx].mean:.3f}, pulls={total_pulls})")

    # ================================================================
    # PHASE C: Optimized — weave with trained configs
    # ================================================================
    banner("Phase C: Optimized (trained configs)")

    t0 = time.perf_counter()
    optimized = await run_queries(bus, llm_fn, shell_configs=trained_configs, label="C")
    phase_c_time = time.perf_counter() - t0

    print(f"\n  Phase C time: {phase_c_time:.1f}s")

    # Show optimized table
    print()
    table(
        ["Query", "Path", "Conf", "Reward", "Tokens", "Resp", "Time"],
        [
            [r["query"][:35], r["shell_path"], f"{r['confidence']:.2f}",
             f"{r['reward']:.3f}", r["tokens"], r["resp_len"], f"{r['time']:.1f}s"]
            for r in optimized
        ],
        [37, 18, 7, 8, 8, 7, 8],
    )

    # ================================================================
    # COMPARISON
    # ================================================================
    banner("Comparison: Baseline vs Optimized")

    def avg(results, key):
        vals = [r[key] for r in results if r[key] > 0]
        return sum(vals) / len(vals) if vals else 0

    metrics = [
        ("Avg confidence", f"{avg(baseline, 'confidence'):.3f}", f"{avg(optimized, 'confidence'):.3f}"),
        ("Avg reward", f"{avg(baseline, 'reward'):.3f}", f"{avg(optimized, 'reward'):.3f}"),
        ("Avg tokens", f"{avg(baseline, 'tokens'):.0f}", f"{avg(optimized, 'tokens'):.0f}"),
        ("Avg response len", f"{avg(baseline, 'resp_len'):.0f}", f"{avg(optimized, 'resp_len'):.0f}"),
        ("Avg time (s)", f"{avg(baseline, 'time'):.1f}", f"{avg(optimized, 'time'):.1f}"),
        ("Total time (s)", f"{phase_a_time:.1f}", f"{phase_c_time:.1f}"),
        ("Successful", f"{sum(1 for r in baseline if r['resp_len'] > 0)}/{len(baseline)}",
                       f"{sum(1 for r in optimized if r['resp_len'] > 0)}/{len(optimized)}"),
    ]
    table(["Metric", "Baseline", "Optimized"], metrics, [20, 15, 15])

    # Per-query comparison
    print(f"\n  Per-query delta:")
    for i, (b, o) in enumerate(zip(baseline, optimized)):
        conf_delta = o["confidence"] - b["confidence"]
        time_delta = o["time"] - b["time"]
        tokens_delta = o["tokens"] - b["tokens"]
        print(f"    Q{i+1}: conf {conf_delta:+.3f}, tokens {tokens_delta:+d}, time {time_delta:+.1f}s")

    # Final audit state
    final_summary = AuditCollector(bus).collect()
    print(f"\n  Final audit: {final_summary.total} entries")
    print(f"  Final avg reward: {final_summary.avg_reward:.3f}")

    # Success criteria
    b_success = sum(1 for r in baseline if r["resp_len"] > 50)
    o_success = sum(1 for r in optimized if r["resp_len"] > 50)
    print(f"\n  Baseline success:  {b_success}/{len(baseline)}")
    print(f"  Optimized success: {o_success}/{len(optimized)}")

    if b_success > 0 and o_success > 0:
        print(f"\n  FEEDBACK LOOP COMPLETE: Both phases produced real responses")
        print(f"  Training pipeline consumed audit data and produced optimized configs")
    elif o_success > 0:
        print(f"\n  OPTIMIZED PHASE SUCCEEDED (baseline had issues)")
    else:
        print(f"\n  LLM ISSUES: Check Ollama model availability")

    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
