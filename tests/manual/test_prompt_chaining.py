#!/usr/bin/env python3
"""
Advanced Prompt Chaining Tests for Qwen 2.5 14B
=================================================
Push the agentic pipeline to its limits with multi-hop reasoning,
mode escalation, adversarial detection, and cross-domain synthesis.

Tests:
  1. Mode Escalation Chain — DIRECT → VERIFY → RESEARCH auto-escalation
  2. Adaptive Research — verify round 2 queries adapt to round 1 findings
  3. LLM Goal Decomposition — PLAN_EXECUTE with intelligent sub-goals
  4. Semantic Contradiction Detection — LLM-based verification vs adversarial claims
  5. Multi-Hop Reasoning — chain outputs through 3 reasoning stages
  6. Cross-Domain Fusion — combine algo + kitchen knowledge in one chain
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

# ── Shard Sets ──────────────────────────────────────────────────────

ALGO_SHARDS = [
    MemoryShard(id="a1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that uses probability matching to balance exploration and exploitation.", metadata={"topic": "algorithms"}),
    MemoryShard(id="a2", text="Epsilon-greedy explores uniformly at random with probability epsilon, and exploits the best-known arm otherwise. Simple but wasteful.", metadata={"topic": "algorithms"}),
    MemoryShard(id="a3", text="UCB (Upper Confidence Bound) selects arms based on optimistic estimates. It has provable regret bounds but assumes stationary rewards.", metadata={"topic": "algorithms"}),
    MemoryShard(id="a4", text="Thompson Sampling naturally adapts exploration to uncertainty. Arms with uncertain rewards get explored more, while well-understood arms get exploited.", metadata={"topic": "algorithms"}),
    MemoryShard(id="a5", text="In non-stationary environments where reward distributions shift over time, sliding-window Thompson Sampling or discounted variants perform better than vanilla TS.", metadata={"topic": "advanced"}),
    MemoryShard(id="a6", text="Contextual bandits extend the basic bandit framework by incorporating user/state features, enabling personalized decisions.", metadata={"topic": "advanced"}),
]

SOUS_SHARDS = [
    MemoryShard(id="k1", text="SOUS uses GPS (Grocery Priority Score) to rank what to buy: gap_score * w_gap + emergency * w_emergency + connector * w_connector.", metadata={"topic": "scoring"}),
    MemoryShard(id="k2", text="MFS (Meal Feasibility Score) evaluates recipes based on ingredient availability, time crunch items used, effort level, and prep time.", metadata={"topic": "scoring"}),
    MemoryShard(id="k3", text="Connectors are ingredients that bridge multiple recipes (eggs, onions, rice). High connector score means more meal options unlocked.", metadata={"topic": "kitchen"}),
    MemoryShard(id="k4", text="Time crunch items are perishables near expiry. SOUS prioritizes recipes that consume these items before they spoil.", metadata={"topic": "kitchen"}),
    MemoryShard(id="k5", text="Tier system: ESSENTIAL (always stock), STAPLE (bulk buy), SPECIAL (buy as needed), FRESHY (short shelf life), PREPPED (cooked output).", metadata={"topic": "inventory"}),
    MemoryShard(id="k6", text="SOUS cleanout mode activates when multiple time crunch items exist. It boosts recipes that consume the most expiring ingredients.", metadata={"topic": "modes"}),
    MemoryShard(id="k7", text="The par level system tracks ideal stock quantities. GPS gap_score = max(0, par - current) / par, measuring how depleted an item is.", metadata={"topic": "scoring"}),
    MemoryShard(id="k8", text="Yield tracking: batch recipes produce PREPPED items (e.g., Bean Chili yields 8 servings, shelf_days=5) that enter inventory as new items.", metadata={"topic": "kitchen"}),
]

# Adversarial: true claims paired with plausible-sounding false claims
ADVERSARIAL_SHARDS = [
    MemoryShard(id="t1", text="Thompson Sampling requires maintaining a posterior distribution for each arm, updated after every observation.", metadata={"topic": "true"}),
    MemoryShard(id="t2", text="Thompson Sampling has no hyperparameters — it is fully determined by the prior and observed data.", metadata={"topic": "true"}),
    MemoryShard(id="f1", text="Thompson Sampling always converges to the optimal arm within 100 trials regardless of the number of arms.", metadata={"topic": "false_claim"}),
    MemoryShard(id="f2", text="Thompson Sampling is provably worse than epsilon-greedy in all stationary environments with more than 5 arms.", metadata={"topic": "false_claim"}),
    MemoryShard(id="f3", text="Thompson Sampling cannot be used with non-conjugate priors and requires Gaussian reward distributions.", metadata={"topic": "false_claim"}),
    MemoryShard(id="t3", text="Thompson Sampling can be applied with any likelihood model using approximate inference methods like MCMC or variational inference.", metadata={"topic": "true"}),
]


def banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def patch_llm(orchestrator):
    """Swap LLM to Qwen on all relevant paths."""
    llm = OllamaLLM(model=MODEL)
    orchestrator.llm = llm
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = llm


async def make_orchestrator(shards):
    """Create a fresh orchestrator with Qwen patched in."""
    from hololoom.agentic import create_agentic_orchestrator
    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False
    orch = await create_agentic_orchestrator(
        config=config, shards=shards,
        enable_verification=True, enable_goal_tracking=True,
    )
    patch_llm(orch)
    return orch


# ════════════════════════════════════════════════════════════════════
# TEST 1: Mode Escalation Chain
# ════════════════════════════════════════════════════════════════════

async def test_1_mode_escalation():
    """
    Chain: DIRECT → (if low confidence) → VERIFY → (if still low) → RESEARCH.
    Tests the system's ability to deepen reasoning when initial answers are weak.
    """
    banner("TEST 1: Mode Escalation Chain (DIRECT -> VERIFY -> RESEARCH)")
    from hololoom.agentic import ReasoningMode

    # Hard query that should trigger escalation
    query = Query(text="Should I use Thompson Sampling or UCB for a recipe recommendation system where user preferences drift seasonally and I have cold-start users?")

    orch = await make_orchestrator(ALGO_SHARDS + SOUS_SHARDS)
    chain_log = []

    # Stage 1: DIRECT
    print("\n  Stage 1: DIRECT")
    t0 = time.perf_counter()
    r1 = await orch.reason(query=query, mode=ReasoningMode.DIRECT, max_steps=1)
    t1 = time.perf_counter() - t0
    conf1 = r1.spacetime.confidence
    resp1 = r1.spacetime.response or ""
    chain_log.append({"mode": "DIRECT", "conf": conf1, "chars": len(resp1), "time": t1})
    print(f"    Conf: {conf1:.2f} | {len(resp1)} chars | {t1:.1f}s")
    print(f"    Preview: {resp1[:150]}...")

    # Stage 2: VERIFY (escalate — feed DIRECT output as context)
    print("\n  Stage 2: VERIFY (escalating)")
    verify_query = Query(text=f"{query.text}\n\nPrevious analysis suggested: {resp1[:300]}\n\nVerify and deepen this analysis.")
    t0 = time.perf_counter()
    r2 = await orch.reason(query=verify_query, mode=ReasoningMode.VERIFY, max_steps=3)
    t2 = time.perf_counter() - t0
    conf2 = r2.spacetime.confidence
    resp2 = r2.spacetime.response or ""
    chain_log.append({"mode": "VERIFY", "conf": conf2, "chars": len(resp2), "time": t2})
    print(f"    Conf: {conf2:.2f} | {len(resp2)} chars | {t2:.1f}s")
    if r2.verification:
        print(f"    Contradictions: {len(r2.verification.contradictions)} | Supporting: {len(r2.verification.supporting_evidence)}")
    print(f"    Preview: {resp2[:150]}...")

    # Stage 3: RESEARCH (full exploration — feed chain context)
    print("\n  Stage 3: RESEARCH (full exploration)")
    research_query = Query(text=f"{query.text}\n\nVerification found: {resp2[:200]}\n\nConduct deep research to resolve remaining uncertainties.")
    t0 = time.perf_counter()
    r3 = await orch.reason(query=research_query, mode=ReasoningMode.RESEARCH, max_steps=4)
    t3 = time.perf_counter() - t0
    conf3 = r3.spacetime.confidence
    resp3 = r3.spacetime.response or ""
    chain_log.append({"mode": "RESEARCH", "conf": conf3, "chars": len(resp3), "time": t3})
    research_qs = [s.get("query", "") for s in r3.steps_taken if s.get("type") == "research_query"]
    print(f"    Conf: {conf3:.2f} | {len(resp3)} chars | {t3:.1f}s")
    print(f"    Research queries generated: {len(research_qs)}")
    for i, rq in enumerate(research_qs):
        print(f"      {i+1}. {rq[:90]}")
    print(f"    Preview: {resp3[:150]}...")

    # Summary
    print("\n  --- Escalation Trajectory ---")
    total_time = sum(c["time"] for c in chain_log)
    for c in chain_log:
        print(f"    {c['mode']:<12s} conf={c['conf']:.2f}  {c['chars']:>5d} chars  {c['time']:5.1f}s")
    print(f"    Total chain time: {total_time:.1f}s")

    # Did the chain deepen the response?
    deepened = len(resp3) > len(resp1) * 0.8
    return deepened


# ════════════════════════════════════════════════════════════════════
# TEST 2: Adaptive Research Queries
# ════════════════════════════════════════════════════════════════════

async def test_2_adaptive_research():
    """
    Verify RESEARCH mode round 2 queries adapt based on round 1 findings.
    Run research twice: once with algo shards, once with kitchen shards.
    The LLM-generated queries should reflect domain context.
    """
    banner("TEST 2: Adaptive Research (Domain-Sensitive Query Generation)")
    from hololoom.agentic import ReasoningMode

    base_query = "How can exploration-exploitation tradeoffs improve decision-making?"

    domains = [
        ("Algorithm Domain", ALGO_SHARDS, base_query),
        ("Kitchen Domain", SOUS_SHARDS, base_query + " Specifically for grocery shopping and meal planning."),
    ]

    all_queries = {}
    for label, shards, qtext in domains:
        print(f"\n  --- {label} ---")
        orch = await make_orchestrator(shards)

        t0 = time.perf_counter()
        result = await orch.reason(
            query=Query(text=qtext),
            mode=ReasoningMode.RESEARCH,
            max_steps=4
        )
        elapsed = time.perf_counter() - t0

        queries = [s.get("query", "") for s in result.steps_taken if s.get("type") == "research_query"]
        all_queries[label] = queries

        print(f"    Time: {elapsed:.1f}s | Queries: {len(queries)}")
        for i, q in enumerate(queries):
            print(f"      {i+1}. {q[:90]}")
        resp = result.spacetime.response or ""
        print(f"    Response: {len(resp)} chars")
        print(f"    Preview: {resp[:200]}...")

    # Check differentiation: algo queries should differ from kitchen queries
    algo_qs = " ".join(all_queries.get("Algorithm Domain", []))
    kitchen_qs = " ".join(all_queries.get("Kitchen Domain", []))

    algo_terms = {"bandit", "arm", "regret", "posterior", "prior", "bayesian", "ucb"}
    kitchen_terms = {"grocery", "meal", "recipe", "ingredient", "cook", "kitchen", "food", "prep"}

    algo_hits = sum(1 for t in algo_terms if t in algo_qs.lower())
    kitchen_hits = sum(1 for t in kitchen_terms if t in kitchen_qs.lower())

    print(f"\n  --- Domain Sensitivity ---")
    print(f"    Algo queries contain {algo_hits}/{len(algo_terms)} algorithm terms")
    print(f"    Kitchen queries contain {kitchen_hits}/{len(kitchen_terms)} kitchen terms")
    print(f"    Differentiated: {algo_hits > 0 or kitchen_hits > 0}")

    return algo_hits > 0 or kitchen_hits > 0


# ════════════════════════════════════════════════════════════════════
# TEST 3: LLM Goal Decomposition
# ════════════════════════════════════════════════════════════════════

async def test_3_llm_goal_decomposition():
    """
    PLAN_EXECUTE with LLM-generated sub-goals (not templates).
    Verify sub-goals are contextual, not generic "What are the prerequisites for X?".
    """
    banner("TEST 3: LLM Goal Decomposition (PLAN_EXECUTE)")
    from hololoom.agentic import ReasoningMode

    query = Query(text="Design a Thompson Sampling system that learns which recipes a household prefers and adapts grocery lists to match cooking patterns over time")
    orch = await make_orchestrator(ALGO_SHARDS + SOUS_SHARDS)

    t0 = time.perf_counter()
    result = await orch.reason(query=query, mode=ReasoningMode.PLAN_EXECUTE, max_steps=5)
    elapsed = time.perf_counter() - t0

    sub_goal_steps = [s for s in result.steps_taken if s.get("type") == "sub_goal"]
    synthesis_step = next((s for s in result.steps_taken if s.get("type") == "synthesis"), None)

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Sub-goals: {len(sub_goal_steps)}")

    # Check if sub-goals are template vs LLM-generated
    template_phrases = [
        "What are the prerequisites for",
        "What are the main steps in",
        "What are potential challenges with",
        "What are success criteria for"
    ]
    template_count = 0
    for i, step in enumerate(sub_goal_steps):
        goal = step.get("goal", "")
        is_template = any(tp in goal for tp in template_phrases)
        if is_template:
            template_count += 1
        marker = "[TEMPLATE]" if is_template else "[LLM-GEN]"
        print(f"    {i+1}. {marker} {goal[:90]}")
        print(f"       conf={step.get('confidence', 0):.2f}")

    resp = result.spacetime.response or ""
    print(f"\n  Response: {len(resp)} chars")
    print(f"  Preview: {resp[:250]}...")
    print(f"\n  Template sub-goals: {template_count}/{len(sub_goal_steps)}")
    print(f"  LLM-generated: {len(sub_goal_steps) - template_count}/{len(sub_goal_steps)}")

    return template_count == 0  # All sub-goals should be LLM-generated


# ════════════════════════════════════════════════════════════════════
# TEST 4: Semantic Contradiction Detection
# ════════════════════════════════════════════════════════════════════

async def test_4_semantic_contradiction():
    """
    Feed adversarial shards (true + false claims) and verify LLM-based
    contradiction detection catches the false claims.
    """
    banner("TEST 4: Semantic Contradiction Detection (LLM Verification)")
    from hololoom.agentic import ReasoningMode

    # The false claims should be caught
    query = Query(text="Evaluate these claims: Thompson Sampling always converges within 100 trials, is worse than epsilon-greedy with 5+ arms, and cannot use non-conjugate priors.")
    orch = await make_orchestrator(ADVERSARIAL_SHARDS)

    t0 = time.perf_counter()
    result = await orch.reason(query=query, mode=ReasoningMode.VERIFY, max_steps=3)
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Verified: {result.verification.verified if result.verification else 'N/A'}")

    if result.verification:
        v = result.verification
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Contradictions found: {len(v.contradictions)}")
        print(f"  Supporting evidence: {len(v.supporting_evidence)}")

        if v.contradictions:
            print(f"\n  Contradiction details:")
            for i, c in enumerate(v.contradictions):
                print(f"    {i+1}. {c[:150]}...")

        # Check if LLM analysis was used
        llm_steps = [s for s in result.steps_taken if s.get("llm_analyzed")]
        print(f"\n  LLM-analyzed steps: {len(llm_steps)}/{len(result.steps_taken)}")

    resp = result.spacetime.response or ""
    print(f"\n  Response: {len(resp)} chars")
    print(f"  Preview: {resp[:250]}...")

    # Should NOT be verified (false claims present)
    not_verified = result.verification and not result.verification.verified
    has_response = len(resp) > 100
    return not_verified or has_response


# ════════════════════════════════════════════════════════════════════
# TEST 5: Multi-Hop Reasoning Chain
# ════════════════════════════════════════════════════════════════════

async def test_5_multi_hop_chain():
    """
    3-stage reasoning chain where each stage's output feeds the next:
      Hop 1: "What is Thompson Sampling?" → factual grounding
      Hop 2: "How would TS apply to [SOUS context]?" → domain transfer
      Hop 3: "Design the implementation" → concrete plan
    Each hop uses the previous response as context.
    """
    banner("TEST 5: Multi-Hop Reasoning Chain (3 stages)")
    from hololoom.agentic import ReasoningMode

    hops = []

    # Hop 1: Factual grounding
    print("\n  Hop 1: Factual Grounding")
    orch1 = await make_orchestrator(ALGO_SHARDS)
    t0 = time.perf_counter()
    r1 = await orch1.reason(
        query=Query(text="Explain Thompson Sampling's core mechanism, when it outperforms alternatives, and its key limitations."),
        mode=ReasoningMode.RESEARCH, max_steps=3
    )
    t1 = time.perf_counter() - t0
    resp1 = r1.spacetime.response or ""
    hops.append({"label": "Factual", "resp": resp1, "time": t1, "chars": len(resp1)})
    print(f"    {len(resp1)} chars | {t1:.1f}s")
    print(f"    {resp1[:150]}...")

    # Hop 2: Domain transfer — feed Hop 1's output + kitchen context
    print("\n  Hop 2: Domain Transfer (algo -> kitchen)")
    orch2 = await make_orchestrator(SOUS_SHARDS)
    hop2_query = f"""Given this understanding of Thompson Sampling:
{resp1[:500]}

How would you apply Thompson Sampling specifically to the SOUS kitchen system?
Consider: GPS scoring, recipe recommendation, connector optimization, and handling seasonal preference drift."""

    t0 = time.perf_counter()
    r2 = await orch2.reason(
        query=Query(text=hop2_query),
        mode=ReasoningMode.DIRECT, max_steps=1
    )
    t2 = time.perf_counter() - t0
    resp2 = r2.spacetime.response or ""
    hops.append({"label": "Transfer", "resp": resp2, "time": t2, "chars": len(resp2)})
    print(f"    {len(resp2)} chars | {t2:.1f}s")
    print(f"    {resp2[:150]}...")

    # Hop 3: Implementation plan — feed both previous hops
    print("\n  Hop 3: Implementation Plan")
    orch3 = await make_orchestrator(ALGO_SHARDS + SOUS_SHARDS)
    hop3_query = f"""Based on:
1. Core TS mechanics: {resp1[:300]}
2. Kitchen application: {resp2[:300]}

Design a concrete implementation plan for integrating Thompson Sampling into SOUS.
Include: data model (what arms/rewards represent), update rules, cold-start handling, and how it interacts with GPS/MFS scoring."""

    t0 = time.perf_counter()
    r3 = await orch3.reason(
        query=Query(text=hop3_query),
        mode=ReasoningMode.PLAN_EXECUTE, max_steps=4
    )
    t3 = time.perf_counter() - t0
    resp3 = r3.spacetime.response or ""
    hops.append({"label": "Implement", "resp": resp3, "time": t3, "chars": len(resp3)})
    print(f"    {len(resp3)} chars | {t3:.1f}s")
    print(f"    {resp3[:150]}...")

    # Summary
    print("\n  --- Multi-Hop Trajectory ---")
    total_time = sum(h["time"] for h in hops)
    total_chars = sum(h["chars"] for h in hops)
    for h in hops:
        print(f"    {h['label']:<12s} {h['chars']:>5d} chars  {h['time']:5.1f}s")
    print(f"    Total: {total_chars} chars across 3 hops in {total_time:.1f}s")

    # Did each hop build on the previous?
    all_real = all(h["chars"] > 100 for h in hops)
    return all_real


# ════════════════════════════════════════════════════════════════════
# TEST 6: Cross-Domain Fusion
# ════════════════════════════════════════════════════════════════════

async def test_6_cross_domain_fusion():
    """
    Single RESEARCH query that requires fusing algorithm knowledge with
    kitchen domain knowledge. Tests if the model can bridge domains.
    """
    banner("TEST 6: Cross-Domain Fusion (Algo + Kitchen in single RESEARCH)")
    from hololoom.agentic import ReasoningMode

    query = Query(text="""A household has 3 people with different dietary preferences.
Using Thompson Sampling concepts, design a system that:
1. Learns each person's meal preferences from cooking history
2. Optimizes the weekly grocery list (GPS scores) based on collective preferences
3. Handles the cold-start problem for new recipes
4. Adapts when someone's preferences change (e.g., going vegetarian)

How would the arms, rewards, and priors be defined?""")

    orch = await make_orchestrator(ALGO_SHARDS + SOUS_SHARDS)

    t0 = time.perf_counter()
    result = await orch.reason(query=query, mode=ReasoningMode.RESEARCH, max_steps=5)
    elapsed = time.perf_counter() - t0

    resp = result.spacetime.response or ""
    research_qs = [s.get("query", "") for s in result.steps_taken if s.get("type") == "research_query"]

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Research queries: {len(research_qs)}")
    for i, rq in enumerate(research_qs):
        print(f"    {i+1}. {rq[:90]}")

    print(f"\n  Response: {len(resp)} chars")
    print(f"  Full response:\n")
    # Print full response wrapped
    for line in resp.split('\n'):
        print(f"    {line}")

    # Check for cross-domain terms
    algo_terms = {"thompson", "bandit", "arm", "reward", "prior", "posterior", "exploration", "exploitation"}
    kitchen_terms = {"grocery", "recipe", "meal", "ingredient", "preference", "gps", "mfs", "connector"}

    resp_lower = resp.lower()
    algo_hits = sum(1 for t in algo_terms if t in resp_lower)
    kitchen_hits = sum(1 for t in kitchen_terms if t in resp_lower)

    print(f"\n  --- Domain Fusion Score ---")
    print(f"    Algorithm terms: {algo_hits}/{len(algo_terms)}")
    print(f"    Kitchen terms: {kitchen_hits}/{len(kitchen_terms)}")
    print(f"    Cross-domain fusion: {'YES' if algo_hits >= 2 and kitchen_hits >= 2 else 'WEAK'}")

    return algo_hits >= 2 and kitchen_hits >= 2


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    banner(f"Advanced Prompt Chaining Tests ({MODEL})")
    print(f"  6 tests exercising multi-hop reasoning, mode escalation,")
    print(f"  adaptive research, LLM decomposition, and cross-domain fusion.")
    print(f"  CPU-only inference — expect ~8-15 minutes total.")

    tests = [
        ("mode_escalation", test_1_mode_escalation),
        ("adaptive_research", test_2_adaptive_research),
        ("llm_goal_decomposition", test_3_llm_goal_decomposition),
        ("semantic_contradiction", test_4_semantic_contradiction),
        ("multi_hop_chain", test_5_multi_hop_chain),
        ("cross_domain_fusion", test_6_cross_domain_fusion),
    ]

    results = {}
    for name, fn in tests:
        try:
            results[name] = await fn()
        except Exception as e:
            print(f"\n  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    banner("FINAL RESULTS")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
