#!/usr/bin/env python3
"""
Tests: v2 Phase 3 — Training Pipeline

Fast structural tests (no LLM required) + optional LLM tests with --llm flag.

Sections:
  I.   AuditCollector — reads and aggregates v2_weave_cycle entries
  II.  ParameterBandit — Thompson Sampling mechanics (posteriors, convergence, selection)
  III. PromptRefiner — structural tests with mock LLM
  IV.  SignatureOptimizer — structural tests with mock LLM
  V.   TrainingPipeline — end-to-end with mock audit data
  VI.  Meta-Properties — self-referential optimization invariants

Run: python test_v2_training.py [--llm]
"""
import asyncio
import math
import random
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from hololoom.memory.v2.training import (
    AuditCollector,
    AuditSummary,
    BanditArm,
    OptimizationResult,
    ParameterBandit,
    PromptRefiner,
    RefinementResult,
    Signature,
    SignatureOptimizer,
    TrainingExample,
    TrainingPipeline,
)
from hololoom.memory.v2.orchestrator import ShellType, ShellConfig, DEFAULT_SHELLS


# =============================================================================
# Helpers
# =============================================================================

USE_LLM = "--llm" in sys.argv

passed = 0
failed = 0
skipped = 0


def test(name):
    def decorator(fn):
        async def wrapper():
            global passed, failed, skipped
            try:
                result = fn() if not asyncio.iscoroutinefunction(fn) else await fn()
                if result == "SKIP":
                    skipped += 1
                    print(f"  [SKIP] {name}")
                else:
                    passed += 1
                    print(f"  [PASS] {name}")
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
        wrapper._test_name = name
        return wrapper
    return decorator


def make_examples(n=20, shell="prime", reward_range=(0.3, 0.9)):
    """Generate synthetic training examples."""
    examples = []
    for i in range(n):
        reward = reward_range[0] + (reward_range[1] - reward_range[0]) * (i / max(1, n - 1))
        conf = 0.5 + reward * 0.3
        examples.append(TrainingExample(
            query=f"Test query {i} about topic {i % 5}",
            shell_type=shell,
            confidence=conf,
            structural=conf * 0.9,
            narrative=conf * 1.1,
            reward=reward,
            context_tokens=512 + i * 50,
            context_items=5 + i,
            duration_ms=1000 + i * 100,
            tool_used=f"matryoshka:{shell}",
            response_length=200 + i * 30,
            timestamp=f"2025-01-{i+1:02d}T12:00:00",
        ))
    return examples


def make_bus_with_audit(examples):
    """Create a mock bus object with audit entries."""
    class MockBus:
        def __init__(self):
            self._audit = []

    bus = MockBus()
    for ex in examples:
        bus._audit.append({
            "type": "v2_weave_cycle",
            "query": ex.query,
            "tool_used": ex.tool_used,
            "response_length": ex.response_length,
            "v2_confidence": ex.confidence,
            "v2_decision": "sufficient" if ex.confidence >= 0.75 else "verify",
            "v2_structural": ex.structural,
            "v2_narrative": ex.narrative,
            "context_tokens": ex.context_tokens,
            "context_items": ex.context_items,
            "reward": ex.reward,
            "duration_ms": ex.duration_ms,
            "timestamp": ex.timestamp,
        })
    return bus


async def mock_llm(prompt: str, max_tokens: int = 512) -> str:
    """Deterministic mock LLM for structural tests."""
    p = prompt.lower()
    if "rate" in p and "0.0" in p and "1.0" in p:
        return "0.72"
    if "improve" in p or "improved" in p:
        return (
            "Based on the retrieved context, synthesize a comprehensive answer.\n\n"
            "{context}\n\n"
            "Question: {query}\n\n"
            "Previous response: {previous_response}\n\n"
            "Provide a thorough, well-structured response:"
        )
    if "generate" in p or "alternative" in p:
        return (
            "1. Synthesize information from context to answer the query comprehensively and accurately.\n"
            "2. Use the retrieved context to ground your response with specific evidence and reasoning.\n"
            "3. Analyze the context carefully and provide a well-structured, evidence-based response."
        )
    if "critique" in p or "weakness" in p:
        return "The template is too generic. It doesn't instruct the model to cite sources or reason step-by-step."
    return "Mock LLM response."


# =============================================================================
# I. AuditCollector
# =============================================================================

@test("AuditCollector: collects v2_weave_cycle entries")
def test_collector_basic():
    examples = make_examples(10)
    bus = make_bus_with_audit(examples)
    collector = AuditCollector(bus)
    summary = collector.collect()

    assert summary.total == 10, f"Expected 10, got {summary.total}"
    assert len(summary.examples) == 10
    assert summary.avg_reward > 0
    assert summary.avg_confidence > 0


@test("AuditCollector: filters non-v2 entries")
def test_collector_filters():
    examples = make_examples(5)
    bus = make_bus_with_audit(examples)
    # Add non-v2 entries
    bus._audit.append({"type": "other_entry", "data": "noise"})
    bus._audit.append({"type": "legacy_weave", "data": "old"})

    collector = AuditCollector(bus)
    summary = collector.collect()
    assert summary.total == 5, f"Expected 5, got {summary.total}"


@test("AuditCollector: empty bus returns zero summary")
def test_collector_empty():
    class EmptyBus:
        _audit = []
    summary = AuditCollector(EmptyBus()).collect()
    assert summary.total == 0
    assert summary.avg_reward == 0
    assert summary.good_rate == 0


@test("AuditCollector: respects limit parameter")
def test_collector_limit():
    examples = make_examples(50)
    bus = make_bus_with_audit(examples)
    collector = AuditCollector(bus)
    summary = collector.collect(limit=10)
    # Should only take last 10 entries from _audit
    assert summary.total == 10


@test("AuditCollector: shell type extraction from tool_used")
def test_collector_shell_type():
    examples = make_examples(5, shell="prime") + make_examples(3, shell="verify")
    bus = make_bus_with_audit(examples)
    summary = AuditCollector(bus).collect()
    assert summary.by_shell.get("prime", 0) == 5
    assert summary.by_shell.get("verify", 0) == 3


@test("AuditCollector: good/poor rate calculation")
def test_collector_rates():
    # All good rewards
    good = make_examples(10, reward_range=(0.7, 0.95))
    bus = make_bus_with_audit(good)
    summary = AuditCollector(bus).collect()
    assert summary.good_rate == 1.0, f"Expected 1.0, got {summary.good_rate}"
    assert summary.poor_rate == 0.0

    # All poor rewards
    poor = make_examples(10, reward_range=(0.1, 0.35))
    bus2 = make_bus_with_audit(poor)
    summary2 = AuditCollector(bus2).collect()
    assert summary2.poor_rate == 1.0, f"Expected 1.0, got {summary2.poor_rate}"


# =============================================================================
# II. ParameterBandit
# =============================================================================

@test("BanditArm: Beta posterior update")
def test_arm_posterior():
    arm = BanditArm(value=0.15, alpha=1.0, beta=1.0)
    assert arm.mean == 0.5  # Uniform prior

    # High reward → alpha increases
    arm.update(0.8)
    assert arm.alpha > 1.0
    assert arm.pulls == 1
    assert arm.mean > 0.5  # Shifted toward success

    # Low reward → beta increases
    arm2 = BanditArm(value=0.15, alpha=1.0, beta=1.0)
    arm2.update(0.2)
    assert arm2.beta > 1.0
    assert arm2.mean < 0.5


@test("BanditArm: sample is in [0, 1]")
def test_arm_sample_range():
    arm = BanditArm(value=0.5, alpha=2.0, beta=3.0)
    for _ in range(100):
        s = arm.sample()
        assert 0.0 <= s <= 1.0, f"Sample out of range: {s}"


@test("ParameterBandit: select returns valid index and value")
def test_bandit_select():
    bandit = ParameterBandit("test", [0.1, 0.2, 0.3, 0.4, 0.5])
    for _ in range(50):
        idx, val = bandit.select()
        assert 0 <= idx < 5
        assert val in [0.1, 0.2, 0.3, 0.4, 0.5]


@test("ParameterBandit: best converges to high-reward arm")
def test_bandit_convergence():
    bandit = ParameterBandit("test", [0.1, 0.2, 0.3])
    # Arm 2 (value=0.3) always gets high reward
    for _ in range(100):
        bandit.update(0, 0.2)  # Low reward for arm 0
        bandit.update(1, 0.3)  # Low reward for arm 1
        bandit.update(2, 0.9)  # High reward for arm 2

    best_idx, best_val = bandit.best()
    assert best_idx == 2, f"Expected arm 2 to win, got arm {best_idx}"
    assert best_val == 0.3


@test("ParameterBandit: Thompson Sampling explores early, exploits late")
def test_bandit_exploration():
    bandit = ParameterBandit("test", [0.1, 0.2, 0.3])

    # Early: uniform prior → high variance → all arms get pulled
    early_selections = set()
    for _ in range(100):
        idx, _ = bandit.select()
        early_selections.add(idx)
    assert len(early_selections) == 3, "All arms should be explored with uniform prior"

    # Train heavily on arm 1
    for _ in range(200):
        bandit.update(1, 0.95)
        bandit.update(0, 0.1)
        bandit.update(2, 0.1)

    # Late: should overwhelmingly pick arm 1
    late_counts = {0: 0, 1: 0, 2: 0}
    for _ in range(100):
        idx, _ = bandit.select()
        late_counts[idx] += 1

    assert late_counts[1] > 80, f"Arm 1 should dominate, got {late_counts}"


@test("ParameterBandit: train_from_examples assigns to closest arm")
def test_bandit_train_from_examples():
    bandit = ParameterBandit("alpha", [0.10, 0.15, 0.20, 0.25, 0.30])

    examples = [
        TrainingExample(
            query="test", shell_type="prime", confidence=0.8,
            structural=0.7, narrative=0.9, reward=0.85,
            context_tokens=500, context_items=5, duration_ms=1000,
        ),
    ]

    # Value function returns 0.15 → should update arm at index 1
    bandit.train_from_examples(examples, lambda e: 0.15)
    assert bandit.arms[1].pulls == 1
    assert bandit.arms[0].pulls == 0


@test("ParameterBandit: report returns correct structure")
def test_bandit_report():
    bandit = ParameterBandit("test", [0.1, 0.2])
    bandit.update(0, 0.8)
    report = bandit.report()

    assert len(report) == 2
    assert "value" in report[0]
    assert "mean" in report[0]
    assert "pulls" in report[0]
    assert report[0]["pulls"] == 1
    assert report[1]["pulls"] == 0


# =============================================================================
# III. PromptRefiner (mock LLM)
# =============================================================================

@test("PromptRefiner: returns unchanged template when no poor examples")
async def test_refiner_no_poor():
    refiner = PromptRefiner(llm_fn=mock_llm)
    result = await refiner.refine(
        shell_type=ShellType.PRIME,
        current_template="Original {context} {query}",
        examples=make_examples(5, reward_range=(0.7, 0.9)),  # All good
    )
    assert result.refined_template == "Original {context} {query}"
    assert result.improvement_score == 0.0
    assert result.iterations == 0


@test("PromptRefiner: refines template from poor examples")
async def test_refiner_with_poor():
    refiner = PromptRefiner(llm_fn=mock_llm, max_iterations=1)
    poor_examples = make_examples(5, reward_range=(0.1, 0.3))

    result = await refiner.refine(
        shell_type=ShellType.PRIME,
        current_template=DEFAULT_SHELLS[ShellType.PRIME].prompt_template,
        examples=poor_examples,
    )

    assert result.iterations >= 1
    assert result.examples_used > 0
    assert result.shell_type == ShellType.PRIME
    assert len(result.critique) > 0


@test("PromptRefiner: preserves placeholders in refined template")
async def test_refiner_placeholder_safety():
    refiner = PromptRefiner(llm_fn=mock_llm, max_iterations=1)
    poor = make_examples(5, reward_range=(0.1, 0.3))

    result = await refiner.refine(
        shell_type=ShellType.VERIFY,
        current_template=DEFAULT_SHELLS[ShellType.VERIFY].prompt_template,
        examples=poor,
    )

    # The mock LLM returns a template with {context} and {query}
    assert "{context}" in result.refined_template
    assert "{query}" in result.refined_template


@test("PromptRefiner: rejects template missing placeholders")
async def test_refiner_rejects_bad_template():
    async def bad_llm(prompt, max_tokens=512):
        p = prompt.lower()
        if "improve" in p:
            return "Just answer the question clearly."  # Missing placeholders!
        if "rate" in p:
            return "0.8"
        return "Critique here."

    refiner = PromptRefiner(llm_fn=bad_llm, max_iterations=1)
    poor = make_examples(3, reward_range=(0.1, 0.3))

    original = DEFAULT_SHELLS[ShellType.PRIME].prompt_template
    result = await refiner.refine(
        shell_type=ShellType.PRIME,
        current_template=original,
        examples=poor,
    )
    # Should keep original since improved version lacks placeholders
    assert result.refined_template == original


# =============================================================================
# IV. SignatureOptimizer (mock LLM)
# =============================================================================

@test("SignatureOptimizer: returns signature unchanged with no examples")
async def test_sig_no_examples():
    optimizer = SignatureOptimizer(llm_fn=mock_llm)
    sig = Signature(
        name="prime",
        input_fields=["context", "query"],
        output_fields=["response"],
        instruction="Answer the question.",
    )
    result = await optimizer.optimize(sig, [])
    assert result.instruction == "Answer the question."


@test("SignatureOptimizer: generates candidate instructions")
async def test_sig_candidates():
    optimizer = SignatureOptimizer(llm_fn=mock_llm, n_candidates=3)
    sig = Signature(
        name="prime",
        input_fields=["context", "query"],
        output_fields=["response"],
        instruction="Answer the question.",
    )
    mixed = make_examples(10, reward_range=(0.2, 0.9))
    result = await optimizer.optimize(sig, mixed)

    # Should have an instruction (either original or improved)
    assert len(result.instruction) > 10
    assert result.name == "prime"


@test("SignatureOptimizer: selects demonstrations from good examples")
async def test_sig_demonstrations():
    optimizer = SignatureOptimizer(llm_fn=mock_llm)
    sig = Signature(
        name="prime",
        input_fields=["context", "query"],
        output_fields=["response"],
        instruction="Answer the question.",
    )
    examples = make_examples(10, reward_range=(0.6, 0.95))
    result = await optimizer.optimize(sig, examples)

    # Should have bootstrapped demonstrations
    assert len(result.demonstrations) <= 3
    if result.demonstrations:
        assert "query" in result.demonstrations[0]


# =============================================================================
# V. TrainingPipeline — end-to-end
# =============================================================================

@test("TrainingPipeline: runs full pipeline with mock audit")
async def test_pipeline_full():
    examples = (
        make_examples(10, shell="prime", reward_range=(0.3, 0.9))
        + make_examples(5, shell="verify", reward_range=(0.4, 0.8))
    )
    bus = make_bus_with_audit(examples)

    pipeline = TrainingPipeline(
        llm_fn=mock_llm,
        enable_prompt_refinement=True,
        enable_signature_optimization=True,
    )
    result = await pipeline.run(bus)

    assert isinstance(result, OptimizationResult)
    assert result.summary.total == 15
    assert result.elapsed_seconds > 0
    assert ShellType.PRIME in result.shell_configs
    assert ShellType.VERIFY in result.shell_configs


@test("TrainingPipeline: works without LLM (bandits only)")
async def test_pipeline_no_llm():
    examples = make_examples(20, shell="prime")
    bus = make_bus_with_audit(examples)

    pipeline = TrainingPipeline(
        llm_fn=None,  # No LLM
        enable_prompt_refinement=True,  # Will be disabled since no llm_fn
        enable_signature_optimization=True,
    )
    result = await pipeline.run(bus)

    assert result.summary.total == 20
    # Refinements should be empty since no LLM
    assert len(result.refinements) == 0
    assert len(result.signature_updates) == 0
    # But bandit reports should exist
    assert len(result.bandit_reports) > 0


@test("TrainingPipeline: bandit configs reflect training data")
async def test_pipeline_configs():
    examples = make_examples(50, shell="prime", reward_range=(0.5, 0.95))
    bus = make_bus_with_audit(examples)

    pipeline = TrainingPipeline(llm_fn=None)
    result = await pipeline.run(bus)

    # Should have configs for all shell types
    configs = result.shell_configs
    assert ShellType.PRIME in configs
    prime = configs[ShellType.PRIME]
    assert isinstance(prime.token_budget, int)
    assert prime.ppr_alpha > 0


@test("TrainingPipeline: online update updates bandits")
def test_pipeline_online():
    pipeline = TrainingPipeline(llm_fn=None)

    # Check initial pulls
    initial_pulls = sum(
        arm.pulls for bandit in pipeline.bandits.values() for arm in bandit.arms
    )
    assert initial_pulls == 0

    # Online update
    ex = TrainingExample(
        query="test", shell_type="prime", confidence=0.8,
        structural=0.7, narrative=0.9, reward=0.85,
        context_tokens=1500, context_items=10, duration_ms=2000,
    )
    pipeline.update_online(ex)

    # Should have updated some arms
    updated_pulls = sum(
        arm.pulls for bandit in pipeline.bandits.values() for arm in bandit.arms
    )
    assert updated_pulls > 0, "Online update should update bandit arms"


@test("TrainingPipeline: changes() reports meaningful output")
async def test_pipeline_changes():
    examples = make_examples(20, shell="prime", reward_range=(0.3, 0.8))
    bus = make_bus_with_audit(examples)

    pipeline = TrainingPipeline(llm_fn=mock_llm)
    result = await pipeline.run(bus)

    changes = result.changes()
    assert isinstance(changes, list)
    # Should have at least bandit reports
    assert len(changes) > 0
    for change in changes:
        assert isinstance(change, str)


@test("TrainingPipeline: get_configs returns valid ShellConfigs")
def test_pipeline_get_configs():
    pipeline = TrainingPipeline(llm_fn=None)
    configs = pipeline.get_configs()

    for shell_type, config in configs.items():
        assert isinstance(config, ShellConfig)
        assert config.shell_type == shell_type
        assert config.token_budget > 0
        assert config.ppr_alpha > 0
        assert len(config.prompt_template) > 0


# =============================================================================
# VI. Meta-Properties — self-referential invariants
# =============================================================================

@test("Meta: Thompson Sampling posterior is proper distribution")
def test_meta_proper_posterior():
    """Every BanditArm has valid Beta parameters (alpha > 0, beta > 0)."""
    bandit = ParameterBandit("test", [0.1, 0.2, 0.3, 0.4, 0.5])

    # Random updates
    for _ in range(200):
        idx = random.randint(0, 4)
        reward = random.random()
        bandit.update(idx, reward)

    for arm in bandit.arms:
        assert arm.alpha > 0, f"Alpha must be > 0, got {arm.alpha}"
        assert arm.beta > 0, f"Beta must be > 0, got {arm.beta}"
        assert 0 <= arm.mean <= 1, f"Mean must be in [0,1], got {arm.mean}"


@test("Meta: bandit posterior concentrates with more data")
def test_meta_posterior_concentration():
    """Variance of Beta(a,b) = ab/((a+b)^2(a+b+1)). More data → less variance."""
    arm = BanditArm(value=0.5)

    # Variance before
    a0, b0 = arm.alpha, arm.beta
    var_before = (a0 * b0) / ((a0 + b0) ** 2 * (a0 + b0 + 1))

    # 100 updates with consistent high reward
    for _ in range(100):
        arm.update(0.85)

    a1, b1 = arm.alpha, arm.beta
    var_after = (a1 * b1) / ((a1 + b1) ** 2 * (a1 + b1 + 1))

    assert var_after < var_before, f"Variance should decrease: {var_before:.6f} → {var_after:.6f}"
    ratio = var_after / var_before
    assert ratio < 0.1, f"Variance should drop significantly, ratio={ratio:.4f}"


@test("Meta: pipeline is idempotent on empty audit")
async def test_meta_idempotent_empty():
    """Running on empty audit should produce unchanged configs."""
    class EmptyBus:
        _audit = []

    pipeline = TrainingPipeline(llm_fn=None)
    configs_before = pipeline.get_configs()
    result = await pipeline.run(EmptyBus())
    configs_after = result.shell_configs

    for shell_type in [ShellType.PRIME, ShellType.VERIFY, ShellType.FLAG]:
        before = configs_before[shell_type]
        after = configs_after[shell_type]
        assert before.ppr_alpha == after.ppr_alpha, f"{shell_type}: alpha changed on empty audit"
        assert before.token_budget == after.token_budget, f"{shell_type}: budget changed on empty audit"


@test("Meta: high-reward training improves best-arm mean")
def test_meta_reward_improves_mean():
    """Consistently high rewards should push the best arm's mean higher."""
    bandit = ParameterBandit("test", [0.1, 0.2, 0.3])
    _, initial_best = bandit.best()
    initial_mean = max(arm.mean for arm in bandit.arms)

    # Train arm 1 with high rewards
    for _ in range(50):
        bandit.update(1, 0.9)

    final_mean = bandit.arms[1].mean
    assert final_mean > initial_mean, f"Mean should improve: {initial_mean:.3f} → {final_mean:.3f}"
    assert final_mean > 0.7, f"Should be well above 0.5 with consistent high reward, got {final_mean:.3f}"


@test("Meta: TrainingExample.is_good and is_poor are mutually exclusive")
def test_meta_example_classification():
    """No example should be both good and poor."""
    for reward in [0.0, 0.1, 0.39, 0.4, 0.5, 0.6, 0.61, 0.8, 1.0]:
        ex = TrainingExample(
            query="test", shell_type="prime", confidence=0.5,
            structural=0.5, narrative=0.5, reward=reward,
            context_tokens=100, context_items=5, duration_ms=500,
        )
        assert not (ex.is_good and ex.is_poor), f"reward={reward} classified as both good and poor"


@test("Meta: audit roundtrip preserves data fidelity")
def test_meta_audit_roundtrip():
    """Data stored in audit should be recoverable by AuditCollector."""
    original = TrainingExample(
        query="How does Thompson Sampling work?",
        shell_type="prime",
        confidence=0.823,
        structural=0.756,
        narrative=0.891,
        reward=0.789,
        context_tokens=1847,
        context_items=12,
        duration_ms=3456.7,
        tool_used="matryoshka:prime",
        response_length=1500,
        timestamp="2025-06-15T12:00:00",
    )
    bus = make_bus_with_audit([original])
    summary = AuditCollector(bus).collect()
    recovered = summary.examples[0]

    assert recovered.query == original.query
    assert abs(recovered.confidence - original.confidence) < 1e-9
    assert abs(recovered.structural - original.structural) < 1e-9
    assert abs(recovered.narrative - original.narrative) < 1e-9
    assert abs(recovered.reward - original.reward) < 1e-9
    assert recovered.context_tokens == original.context_tokens
    assert recovered.context_items == original.context_items
    assert recovered.shell_type == original.shell_type


# =============================================================================
# VII. RewardJudge
# =============================================================================

from hololoom.memory.v2.training import RewardJudge


@test("RewardJudge: structural-only when no LLM")
async def test_reward_no_llm():
    judge = RewardJudge(llm_fn=None)
    reward = await judge.score(
        query="How does Thompson Sampling work?",
        response="Thompson Sampling uses Bayesian posteriors..." * 20,
        confidence=0.8,
    )
    assert 0.0 <= reward <= 1.0
    assert reward > 0.5, f"High confidence + long response should score well, got {reward:.3f}"


@test("RewardJudge: empty response scores low")
async def test_reward_empty():
    judge = RewardJudge(llm_fn=None)
    reward = await judge.score(query="test", response="", confidence=0.8)
    assert reward < 0.7, f"Empty response should score lower, got {reward:.3f}"


@test("RewardJudge: LLM judge called for substantial response")
async def test_reward_with_llm():
    judge = RewardJudge(llm_fn=mock_llm)
    reward = await judge.score(
        query="Compare algorithms",
        response="Thompson Sampling outperforms epsilon-greedy..." * 10,
        confidence=0.7,
    )
    assert 0.0 <= reward <= 1.0


@test("RewardJudge: handles LLM failure gracefully")
async def test_reward_llm_failure():
    async def failing_llm(prompt, max_tokens=32):
        raise RuntimeError("LLM unavailable")

    judge = RewardJudge(llm_fn=failing_llm)
    reward = await judge.score(
        query="test", response="Some response here with enough text.",
        confidence=0.6,
    )
    # Should fallback to 0.5 for LLM component
    assert 0.0 <= reward <= 1.0


@test("RewardJudge: confidence dominates when response is short")
async def test_reward_confidence_dominates():
    judge = RewardJudge(llm_fn=None)
    high = await judge.score(query="test", response="Short.", confidence=0.95)
    low = await judge.score(query="test", response="Short.", confidence=0.2)
    assert high > low, f"Higher confidence should score better: {high:.3f} vs {low:.3f}"


# =============================================================================
# VIII. Persistence
# =============================================================================

import tempfile
import os


@test("Persistence: save and load roundtrip")
def test_persistence_roundtrip():
    pipeline = TrainingPipeline(llm_fn=None)
    # Train some data
    for _ in range(50):
        pipeline.bandits["prime.alpha"].update(2, 0.9)
        pipeline.bandits["verify.alpha"].update(1, 0.3)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    try:
        pipeline.save(path)
        assert os.path.exists(path)

        # Load into fresh pipeline
        pipeline2 = TrainingPipeline(llm_fn=None)
        loaded = pipeline2.load(path)
        assert loaded, "Load should return True"

        # Posteriors should match
        for name in pipeline.bandits:
            for i, arm in enumerate(pipeline.bandits[name].arms):
                arm2 = pipeline2.bandits[name].arms[i]
                assert abs(arm.alpha - arm2.alpha) < 1e-9, f"{name}[{i}] alpha mismatch"
                assert abs(arm.beta - arm2.beta) < 1e-9, f"{name}[{i}] beta mismatch"
                assert arm.pulls == arm2.pulls, f"{name}[{i}] pulls mismatch"
    finally:
        os.unlink(path)


@test("Persistence: load from nonexistent file returns False")
def test_persistence_missing():
    pipeline = TrainingPipeline(llm_fn=None)
    loaded = pipeline.load("/nonexistent/path/state.json")
    assert not loaded


@test("Persistence: load from corrupt file returns False")
def test_persistence_corrupt():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        f.write("not valid json{{{")
        path = f.name

    try:
        pipeline = TrainingPipeline(llm_fn=None)
        loaded = pipeline.load(path)
        assert not loaded
    finally:
        os.unlink(path)


# =============================================================================
# Collect and run
# =============================================================================

all_tests = [v for v in globals().values() if callable(v) and hasattr(v, '_test_name')]


async def main():
    print(f"\n{'='*72}")
    print(f"  Phase 3: Training Pipeline Tests")
    print(f"  ({len(all_tests)} tests, LLM={'ON' if USE_LLM else 'OFF'})")
    print(f"{'='*72}")

    sections = {
        "I. AuditCollector": [t for t in all_tests if "collector" in t._test_name.lower() or "audit" in t._test_name.lower().split(":")[0]],
        "II. ParameterBandit": [t for t in all_tests if "bandit" in t._test_name.lower() or "arm" in t._test_name.lower()],
        "III. PromptRefiner": [t for t in all_tests if "refiner" in t._test_name.lower()],
        "IV. SignatureOptimizer": [t for t in all_tests if "signature" in t._test_name.lower() or "sig" in t._test_name.lower().split(":")[0]],
        "V. TrainingPipeline": [t for t in all_tests if "pipeline" in t._test_name.lower() and "meta" not in t._test_name.lower() and "persist" not in t._test_name.lower()],
        "VI. Meta-Properties": [t for t in all_tests if "meta" in t._test_name.lower()],
        "VII. RewardJudge": [t for t in all_tests if "reward" in t._test_name.lower()],
        "VIII. Persistence": [t for t in all_tests if "persist" in t._test_name.lower()],
    }

    # Find uncategorized tests
    categorized = set()
    for tests in sections.values():
        for t in tests:
            categorized.add(id(t))
    uncategorized = [t for t in all_tests if id(t) not in categorized]
    if uncategorized:
        sections["VII. Other"] = uncategorized

    for section_name, tests in sections.items():
        if not tests:
            continue
        print(f"\n  {section_name}")
        print(f"  {'─'*60}")
        for t in tests:
            await t()

    print(f"\n{'='*72}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*72}")

    if failed == 0:
        print(f"  PHASE 3: ALL {passed} TESTS PASSED")
    else:
        print(f"  {failed} FAILURES — see output above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
