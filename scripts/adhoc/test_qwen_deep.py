#!/usr/bin/env python3
"""
Deep HoloLoom tests with Qwen 2.5 14B.

Tests:
  1. PLAN_EXECUTE mode — goal decomposition into sub-goals
  2. Adversarial VERIFY — contradictory shards to test reasoning
  3. SOUS domain RAG — kitchen knowledge retrieval + synthesis
  4. Multi-turn RESEARCH — iterative refinement across rounds
  5. Comparative reasoning — same query, all 4 modes side-by-side
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


def banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def sub(title):
    print(f"\n  --- {title} ---")


# ============================================================================
# SHARD SETS
# ============================================================================

# Rich domain shards about Thompson Sampling + bandits
ALGO_SHARDS = [
    MemoryShard(id="ts1", text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It maintains a posterior distribution over the reward probability of each arm, and at each step samples from these posteriors to decide which arm to pull.", metadata={"topic": "algorithms", "confidence": 0.95}),
    MemoryShard(id="ts2", text="The regret bound for Thompson Sampling in Bernoulli bandits is O(sqrt(KT log T)), where K is the number of arms and T is the time horizon. This is near-optimal compared to the information-theoretic lower bound.", metadata={"topic": "theory", "confidence": 0.9}),
    MemoryShard(id="ts3", text="Thompson Sampling outperforms epsilon-greedy in non-stationary environments because the posterior naturally adapts to changing reward distributions, while epsilon-greedy maintains fixed exploration rates.", metadata={"topic": "comparison", "confidence": 0.88}),
    MemoryShard(id="ts4", text="UCB (Upper Confidence Bound) algorithms provide deterministic exploration guarantees, while Thompson Sampling provides probabilistic exploration. In practice, Thompson Sampling often achieves lower empirical regret than UCB despite weaker theoretical guarantees.", metadata={"topic": "comparison", "confidence": 0.85}),
    MemoryShard(id="ts5", text="Thompson Sampling has been successfully deployed at scale by Netflix for content recommendation, by Google for ad placement optimization, and by Microsoft for news article selection in MSN.", metadata={"topic": "applications", "confidence": 0.82}),
    MemoryShard(id="ts6", text="A key limitation of Thompson Sampling is the computational cost of maintaining and sampling from posterior distributions, especially for complex reward models. Approximate inference methods like variational Bayes are often needed at scale.", metadata={"topic": "limitations", "confidence": 0.8}),
    MemoryShard(id="ts7", text="In contextual bandits, Thompson Sampling extends naturally by conditioning the posterior on context features. This is the basis of systems like LinTS (Linear Thompson Sampling) and Neural Thompson Sampling.", metadata={"topic": "extensions", "confidence": 0.78}),
    MemoryShard(id="ts8", text="For delayed feedback settings, Thompson Sampling remains effective because the Bayesian update can incorporate feedback asynchronously. This makes it suitable for clinical trials and ad campaigns where outcomes are delayed.", metadata={"topic": "robustness", "confidence": 0.75}),
]

# Contradictory shards for adversarial verification
CONTRADICTORY_SHARDS = [
    MemoryShard(id="c1", text="Thompson Sampling always converges to the optimal arm in finite time for any bandit problem.", metadata={"topic": "claim", "confidence": 0.9}),
    MemoryShard(id="c2", text="Thompson Sampling can fail to converge in adversarial bandit settings where the reward distribution changes based on the learner's choices.", metadata={"topic": "counter", "confidence": 0.85}),
    MemoryShard(id="c3", text="Thompson Sampling requires no hyperparameter tuning, making it universally applicable.", metadata={"topic": "claim", "confidence": 0.8}),
    MemoryShard(id="c4", text="The choice of prior distribution in Thompson Sampling significantly affects performance. A poorly chosen prior can lead to excessive exploration or premature exploitation, effectively making prior selection a critical hyperparameter.", metadata={"topic": "counter", "confidence": 0.88}),
    MemoryShard(id="c5", text="Thompson Sampling is computationally cheaper than UCB because it only requires sampling, not confidence interval computation.", metadata={"topic": "claim", "confidence": 0.7}),
    MemoryShard(id="c6", text="For complex models like neural networks, Thompson Sampling requires approximate posterior inference (MCMC or variational methods), which is significantly more expensive than the simple confidence bounds used by UCB.", metadata={"topic": "counter", "confidence": 0.92}),
]

# SOUS kitchen domain shards
SOUS_SHARDS = [
    MemoryShard(id="k1", text="The GPS (Grocery Priority Score) algorithm ranks shopping items by combining gap-to-par ratio, tier multiplier, emergency flags, connector bonuses, and time-crunch support. Items scoring above T0=80 are P0 (emergency), above T1=35 are P1 (restore PAR).", metadata={"topic": "sous_scoring", "confidence": 0.95}),
    MemoryShard(id="k2", text="MFS (Meal Feasibility Score) ranks recipes by time-crunch ingredient usage, urgency weighting, missing ingredients penalty, effort cost, and batch bonuses. Meals are bucketed into Rescue (uses TC items), Autopilot (easy, no missing), and Project (needs shopping).", metadata={"topic": "sous_scoring", "confidence": 0.93}),
    MemoryShard(id="k3", text="The PAR level system defines target stock quantities per item. When inventory drops below PAR, the item appears on the grocery list. The buffer zone between PAR and 2x PAR represents comfortable stock. Items above 2x PAR are overstocked.", metadata={"topic": "sous_inventory", "confidence": 0.9}),
    MemoryShard(id="k4", text="Time-Crunch Events (TCE) represent perishable items approaching expiration. They have deadlines in days, urgency scores (3 for 1 day, 2 for 2 days, 1 for 3-4 days), and drive Rescue meal suggestions that consume them before spoilage.", metadata={"topic": "sous_time_crunch", "confidence": 0.92}),
    MemoryShard(id="k5", text="Connector ingredients are items that bridge multiple recipes. Onions, garlic, eggs, and soy sauce are common connectors. The GPS gives connectors a bonus because restocking them unlocks the most meal options.", metadata={"topic": "sous_connectors", "confidence": 0.88}),
    MemoryShard(id="k6", text="SOUS operates in three modes: QUICK (penalizes long cook times), PREP (bonuses for batch cooking), and CLEANOUT (maximizes time-crunch rescue). Mode selection adjusts MFS weights dynamically.", metadata={"topic": "sous_modes", "confidence": 0.9}),
    MemoryShard(id="k7", text="The cook action consumes ingredients from inventory, removes used time-crunch items, and creates prepped items with their own shelf life. Prepped items appear in inventory as a separate PREPPED tier.", metadata={"topic": "sous_cooking", "confidence": 0.87}),
    MemoryShard(id="k8", text="SOUS uses a control loop: inventory state + time-crunch events -> scoring engine -> prioritized grocery list + ranked meal suggestions -> user actions (cook/shop) -> updated inventory state. The loop runs continuously.", metadata={"topic": "sous_architecture", "confidence": 0.95}),
]


# ============================================================================
# TEST 1: PLAN_EXECUTE Mode
# ============================================================================

async def test_plan_execute():
    """Goal decomposition into sub-goals, executed sequentially."""
    banner("TEST 1: PLAN_EXECUTE Mode (Goal Decomposition)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    orchestrator = await create_agentic_orchestrator(
        config=config, shards=ALGO_SHARDS,
        enable_verification=True, enable_goal_tracking=True,
    )
    # Swap LLM to Qwen
    orchestrator.llm = OllamaLLM(model=MODEL)
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = OllamaLLM(model=MODEL)

    query = Query(text="Design a Thompson Sampling system for a recipe recommendation engine that adapts to user taste over time")

    print(f"  Model: {MODEL}")
    print(f"  Query: {query.text}")
    print(f"  Mode: PLAN_EXECUTE (max_steps=4)")

    t0 = time.perf_counter()
    result = await orchestrator.reason(
        query=query, mode=ReasoningMode.PLAN_EXECUTE, max_steps=4
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Total queries: {result.total_queries}")
    print(f"  Confidence: {result.spacetime.confidence:.2f}")

    print(f"\n  Sub-goals decomposed:")
    for i, step in enumerate(result.steps_taken):
        stype = step.get("type", "?")
        if stype == "sub_goal":
            goal = step.get("goal", "")
            conf = step.get("confidence", 0)
            done = step.get("completed", False)
            print(f"    {i}. [{'DONE' if done else 'OPEN'}] {goal[:90]}")
            print(f"       conf={conf:.2f}")
        elif stype == "synthesis":
            print(f"    {i}. [SYNTHESIS] sub_goals_completed={step.get('sub_goals_completed', 0)} conf={step.get('confidence', 0):.2f}")

    resp = result.spacetime.response or ""
    if resp:
        print(f"\n  Response ({len(resp)} chars):")
        for line in resp[:600].split("\n"):
            print(f"    {line}")
        if len(resp) > 600:
            print(f"    ... ({len(resp) - 600} more chars)")

    return True


# ============================================================================
# TEST 2: Adversarial VERIFY with Contradictory Evidence
# ============================================================================

async def test_adversarial_verify():
    """Verification with intentionally contradictory shards."""
    banner("TEST 2: Adversarial VERIFY (Contradictory Evidence)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    orchestrator = await create_agentic_orchestrator(
        config=config, shards=CONTRADICTORY_SHARDS,
        enable_verification=True, enable_goal_tracking=True,
    )
    orchestrator.llm = OllamaLLM(model=MODEL)
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = OllamaLLM(model=MODEL)

    query = Query(text="Is Thompson Sampling universally superior and free of hyperparameters?")

    print(f"  Model: {MODEL}")
    print(f"  Query: {query.text}")
    print(f"  Shards: 3 claims + 3 contradictions")
    print(f"  Mode: VERIFY (max_steps=3)")

    t0 = time.perf_counter()
    result = await orchestrator.reason(
        query=query, mode=ReasoningMode.VERIFY, max_steps=3
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.1f}s")

    if result.verification:
        v = result.verification
        print(f"  Verified: {v.verified}")
        print(f"  Confidence: {v.confidence:.2f}")
        print(f"  Contradictions found: {len(v.contradictions)}")
        print(f"  Supporting evidence: {len(v.supporting_evidence)}")
        if v.contradictions:
            print(f"\n  Contradictions:")
            for c in v.contradictions[:3]:
                print(f"    - {c[:120]}...")
        if v.suggested_refinements:
            print(f"\n  Refinements suggested:")
            for r in v.suggested_refinements[:2]:
                print(f"    - {r[:120]}...")

    for i, step in enumerate(result.steps_taken):
        print(f"  Step {i+1}: {step.get('type', '?')} conf={step.get('confidence', 0):.2f}")

    return True


# ============================================================================
# TEST 3: SOUS Domain RAG
# ============================================================================

async def test_sous_domain_rag():
    """RAG over SOUS kitchen knowledge — tests domain-specific retrieval."""
    banner("TEST 3: SOUS Domain RAG (Kitchen Knowledge)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    orchestrator = await create_agentic_orchestrator(
        config=config, shards=SOUS_SHARDS,
        enable_verification=True, enable_goal_tracking=True,
    )
    orchestrator.llm = OllamaLLM(model=MODEL)
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = OllamaLLM(model=MODEL)

    queries = [
        "I have spinach expiring tomorrow, 6 eggs, and rice. What should I cook?",
        "Explain how the GPS scoring algorithm decides what goes on my grocery list",
        "How do connectors improve the SOUS meal suggestion system?",
    ]

    for q_text in queries:
        query = Query(text=q_text)
        sub(q_text[:60] + "...")

        t0 = time.perf_counter()
        result = await orchestrator.reason(
            query=query, mode=ReasoningMode.DIRECT, max_steps=1
        )
        elapsed = time.perf_counter() - t0

        resp = result.spacetime.response or "(no response)"
        print(f"    Time: {elapsed:.1f}s | Conf: {result.spacetime.confidence:.2f}")
        print(f"    LLM used: {result.steps_taken[0].get('llm_generated', False) if result.steps_taken else '?'}")
        # Print response, wrapped
        for line in resp[:400].split("\n"):
            print(f"    {line}")
        if len(resp) > 400:
            print(f"    ... ({len(resp) - 400} more)")

    return True


# ============================================================================
# TEST 4: Iterative RESEARCH with Adaptive Follow-ups
# ============================================================================

async def test_iterative_research():
    """RESEARCH mode with max_steps=5 for deeper exploration."""
    banner("TEST 4: Deep RESEARCH (5-step Iterative Exploration)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    # Combine algorithm + SOUS shards for cross-domain reasoning
    all_shards = ALGO_SHARDS + SOUS_SHARDS

    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_safety_guardrails = False

    orchestrator = await create_agentic_orchestrator(
        config=config, shards=all_shards,
        enable_verification=True, enable_goal_tracking=True,
    )
    orchestrator.llm = OllamaLLM(model=MODEL)
    if hasattr(orchestrator.learning_engine, 'orchestrator'):
        wo = orchestrator.learning_engine.orchestrator
        if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
            wo.tool_executor.llm = OllamaLLM(model=MODEL)

    query = Query(text="How could Thompson Sampling improve the SOUS kitchen control loop for smarter grocery and meal decisions?")

    print(f"  Model: {MODEL}")
    print(f"  Query: {query.text}")
    print(f"  Shards: {len(all_shards)} (algo + SOUS cross-domain)")
    print(f"  Mode: RESEARCH (max_steps=5)")

    t0 = time.perf_counter()
    result = await orchestrator.reason(
        query=query, mode=ReasoningMode.RESEARCH, max_steps=5
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Total queries: {result.total_queries}")
    print(f"  Confidence: {result.spacetime.confidence:.2f}")

    print(f"\n  Research trajectory:")
    for i, step in enumerate(result.steps_taken):
        stype = step.get("type", "?")
        if stype == "research_query":
            q = step.get("query", "")
            conf = step.get("confidence", 0)
            # Check if LLM-generated
            templates = ["What are the key concepts", "What are the tradeoffs", "What are practical applications"]
            is_template = any(t in q for t in templates)
            marker = "TPL" if is_template else "LLM"
            print(f"    {i+1}. [{marker}] {q[:85]}")
            print(f"       conf={conf:.2f}")
        elif stype == "synthesis":
            print(f"    {i+1}. [SYN] sources={step.get('sources', 0)} conf={step.get('confidence', 0):.2f}")

    resp = result.spacetime.response or ""
    if resp:
        print(f"\n  Synthesized Response ({len(resp)} chars):")
        for line in resp[:800].split("\n"):
            print(f"    {line}")
        if len(resp) > 800:
            print(f"    ... ({len(resp) - 800} more)")

    return True


# ============================================================================
# TEST 5: All 4 Modes Compared on Same Query
# ============================================================================

async def test_mode_comparison():
    """Same query through DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE."""
    banner("TEST 5: Mode Comparison (Same Query, 4 Modes)")

    from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

    query_text = "Should I use Thompson Sampling or epsilon-greedy for optimizing my weekly meal prep rotation?"

    modes = [
        (ReasoningMode.DIRECT, 1),
        (ReasoningMode.VERIFY, 3),
        (ReasoningMode.RESEARCH, 3),
        (ReasoningMode.PLAN_EXECUTE, 4),
    ]

    all_shards = ALGO_SHARDS + SOUS_SHARDS
    results_table = []

    for mode, max_steps in modes:
        config = Config.fast()
        config.enable_agentic_reasoning = True
        config.enable_safety_guardrails = False

        orchestrator = await create_agentic_orchestrator(
            config=config, shards=all_shards,
            enable_verification=True, enable_goal_tracking=True,
        )
        orchestrator.llm = OllamaLLM(model=MODEL)
        if hasattr(orchestrator.learning_engine, 'orchestrator'):
            wo = orchestrator.learning_engine.orchestrator
            if hasattr(wo, 'tool_executor') and hasattr(wo.tool_executor, 'llm'):
                wo.tool_executor.llm = OllamaLLM(model=MODEL)

        sub(f"{mode.value.upper()} (max_steps={max_steps})")

        t0 = time.perf_counter()
        result = await orchestrator.reason(
            query=Query(text=query_text), mode=mode, max_steps=max_steps
        )
        elapsed = time.perf_counter() - t0

        resp = result.spacetime.response or ""
        n_steps = len(result.steps_taken)

        print(f"    Time: {elapsed:.1f}s | Steps: {n_steps} | Conf: {result.spacetime.confidence:.2f}")
        for line in resp[:300].split("\n"):
            print(f"    {line}")
        if len(resp) > 300:
            print(f"    ...")

        results_table.append({
            "mode": mode.value, "time": elapsed, "steps": n_steps,
            "conf": result.spacetime.confidence, "resp_len": len(resp)
        })

    sub("Comparison Table")
    print(f"    {'Mode':<14s} | {'Time':>6s} | {'Steps':>5s} | {'Conf':>5s} | {'Resp':>5s}")
    print(f"    {'-'*14}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    for r in results_table:
        print(f"    {r['mode']:<14s} | {r['time']:5.1f}s | {r['steps']:>5d} | {r['conf']:5.2f} | {r['resp_len']:>5d}")

    return True


# ============================================================================
# MAIN
# ============================================================================

async def main():
    banner(f"Deep HoloLoom Tests with {MODEL}")

    # Verify model
    llm = OllamaLLM(model=MODEL)
    if not llm.is_available():
        print(f"  ERROR: {MODEL} not available. Run: ollama pull {MODEL}")
        return

    print(f"  Model: {MODEL}")
    print(f"  Tests: PLAN_EXECUTE, Adversarial VERIFY, SOUS RAG, Deep RESEARCH, Mode Comparison")
    print(f"  Note: CPU-only inference, expect ~15 tok/s. Full suite ~5-8 min.")

    results = {}
    tests = [
        ("plan_execute", test_plan_execute),
        ("adversarial_verify", test_adversarial_verify),
        ("sous_rag", test_sous_domain_rag),
        ("deep_research", test_iterative_research),
        ("mode_comparison", test_mode_comparison),
    ]

    for name, fn in tests:
        try:
            results[name] = await fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    banner("FINAL RESULTS")
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")


if __name__ == "__main__":
    asyncio.run(main())
