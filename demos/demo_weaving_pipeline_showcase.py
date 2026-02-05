#!/usr/bin/env python3
"""
HoloLoom Weaving Pipeline Showcase
====================================
Simulates the full 9-step weaving cycle — the beating heart of HoloLoom.

Every query travels through these stages:

  Step 0: Meta-Prompt Enhancement    (optional query rewriting)
  Step 1: Pattern Selection          (BARE / FAST / FUSED)
  Step 2: Chrono Trigger             (temporal window)
  Step 3: Thread Selection           (memory retrieval)
  ── parallel ──────────────────────────────────────────
  Step 4: Resonance Shed             (feature extraction -> DotPlasma)
  Step 5: Warp Space                 (continuous tensor manifold)
  Step 6: Memory Retrieval           (context assembly)
  ── sequential ────────────────────────────────────────
  Step 7: Convergence Engine         (tool selection via Thompson Sampling)
  Step 8: Tool Execution             (safety-gated action)
  Step 9: Spacetime Fabric           (result + provenance assembly)

Run:
    python demos/demo_weaving_pipeline_showcase.py

No heavy dependencies required (numpy only).
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

class PatternCard(Enum):
    BARE = "bare"     # Minimal: <50ms
    FAST = "fast"     # Balanced: <150ms
    FUSED = "fused"   # Full: <300ms


class ComplexityLevel(Enum):
    LITE = "lite"           # 3 steps
    FAST = "fast"           # 5 steps
    FULL = "full"           # 7 steps
    RESEARCH = "research"   # 9 steps


class CollapseStrategy(Enum):
    ARGMAX = "argmax"
    EPSILON_GREEDY = "epsilon_greedy"
    BAYESIAN_BLEND = "bayesian_blend"
    PURE_THOMPSON = "pure_thompson"


@dataclass
class MemoryShard:
    id: str
    content: str
    relevance: float = 0.0
    timestamp: float = 0.0


@dataclass
class DotPlasma:
    """Feature fluid — the fused representation flowing between stages."""
    motifs: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    spectral: Optional[np.ndarray] = None


@dataclass
class WeavingTrace:
    """Complete provenance for debugging and reflection."""
    stage_timings: Dict[str, float] = field(default_factory=dict)
    decisions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_timings.values())


@dataclass
class Spacetime:
    """The final woven fabric: response + metadata + lineage."""
    response: str
    confidence: float
    tool_used: str
    trace: WeavingTrace
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline simulation
# ---------------------------------------------------------------------------

class WeavingPipeline:
    """Simulates the 9-step weaving cycle with realistic timing and data flow."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

        # Simulated memory store
        self.memory_shards = [
            MemoryShard("ts-001", "Thompson Sampling balances exploration and exploitation using posterior sampling", 0.0, time.time() - 3600),
            MemoryShard("ts-002", "Beta(alpha, beta) distribution models Bernoulli success rates", 0.0, time.time() - 7200),
            MemoryShard("ts-003", "Multi-armed bandits face the explore-exploit dilemma", 0.0, time.time() - 1800),
            MemoryShard("rl-001", "Reinforcement learning optimises cumulative reward over episodes", 0.0, time.time() - 86400),
            MemoryShard("rl-002", "PPO uses clipped surrogate objectives for stable policy updates", 0.0, time.time() - 172800),
            MemoryShard("ml-001", "Neural networks learn hierarchical feature representations", 0.0, time.time() - 259200),
            MemoryShard("ml-002", "Gradient descent minimises loss by following the negative gradient", 0.0, time.time() - 345600),
            MemoryShard("stat-001", "Bayesian inference combines prior beliefs with observed likelihood", 0.0, time.time() - 432000),
        ]

        # Tool definitions
        self.tools = ["answer", "research", "clarify", "code"]

    def _sim_latency(self, base_ms: float, variance: float = 0.2) -> float:
        """Simulate realistic latency with jitter."""
        jitter = self.rng.normal(0, base_ms * variance)
        return max(0.1, base_ms + jitter)

    # --- Step 0: Meta-Prompt Enhancement ---

    def step0_meta_prompt(self, query: str, enable: bool = True) -> Tuple[str, Dict]:
        t0 = time.perf_counter()
        enhanced = query
        applied = False

        if enable and len(query.split()) < 8:
            # Expand short queries with context
            enhanced = f"{query} — explain the concept, its mechanisms, and practical applications"
            applied = True

        latency = self._sim_latency(2.0) if enable else 0.0
        return enhanced, {
            "stage": "meta_prompt",
            "latency_ms": latency,
            "enhanced": applied,
            "original": query,
            "result": enhanced,
        }

    # --- Step 1: Pattern Selection ---

    def step1_pattern_selection(self, query: str) -> Tuple[PatternCard, ComplexityLevel, Dict]:
        word_count = len(query.split())
        research_keywords = {"analyze", "compare", "evaluate", "tradeoffs", "comprehensive"}
        has_research = any(kw in query.lower() for kw in research_keywords)

        if has_research or word_count > 20:
            pattern = PatternCard.FUSED
            complexity = ComplexityLevel.RESEARCH
        elif word_count > 10:
            pattern = PatternCard.FAST
            complexity = ComplexityLevel.FULL
        elif word_count > 5:
            pattern = PatternCard.FAST
            complexity = ComplexityLevel.FAST
        else:
            pattern = PatternCard.BARE
            complexity = ComplexityLevel.LITE

        latency = self._sim_latency(1.5)
        return pattern, complexity, {
            "stage": "pattern_selection",
            "latency_ms": latency,
            "pattern": pattern.value,
            "complexity": complexity.value,
            "word_count": word_count,
        }

    # --- Step 2: Chrono Trigger ---

    def step2_chrono_trigger(self, complexity: ComplexityLevel) -> Tuple[float, float, Dict]:
        lookback_days = {
            ComplexityLevel.LITE: 7,
            ComplexityLevel.FAST: 30,
            ComplexityLevel.FULL: 90,
            ComplexityLevel.RESEARCH: 365,
        }[complexity]

        timeout_ms = {
            ComplexityLevel.LITE: 50,
            ComplexityLevel.FAST: 150,
            ComplexityLevel.FULL: 300,
            ComplexityLevel.RESEARCH: 5000,
        }[complexity]

        recency_bias = 0.3 + 0.1 * (365 - lookback_days) / 365

        latency = self._sim_latency(1.0)
        return lookback_days, recency_bias, {
            "stage": "chrono_trigger",
            "latency_ms": latency,
            "lookback_days": lookback_days,
            "timeout_ms": timeout_ms,
            "recency_bias": round(recency_bias, 3),
        }

    # --- Step 3: Thread Selection ---

    def step3_thread_selection(
        self, query: str, lookback_days: float, recency_bias: float
    ) -> Tuple[List[MemoryShard], Dict]:
        query_words = set(query.lower().split())
        now = time.time()
        cutoff = now - lookback_days * 86400

        scored = []
        for shard in self.memory_shards:
            if shard.timestamp < cutoff:
                continue
            shard_words = set(shard.content.lower().split())
            relevance = len(query_words & shard_words) / max(len(query_words), 1)
            age_days = (now - shard.timestamp) / 86400
            recency_score = np.exp(-0.05 * age_days) * recency_bias
            combined = 0.6 * relevance + 0.4 * recency_score
            shard.relevance = combined
            scored.append((shard, combined))

        scored.sort(key=lambda x: -x[1])
        selected = [s for s, _ in scored[:5]]

        latency = self._sim_latency(8.0)
        return selected, {
            "stage": "thread_selection",
            "latency_ms": latency,
            "candidates": len(self.memory_shards),
            "within_window": len(scored),
            "selected": len(selected),
            "top_shard": selected[0].id if selected else "none",
        }

    # --- Steps 4-6: Parallel Feature Extraction ---

    def step4_resonance_shed(self, threads: List[MemoryShard]) -> Tuple[DotPlasma, Dict]:
        """Extract features: motif + embedding + spectral -> DotPlasma."""
        motifs = {}
        for shard in threads:
            words = shard.content.lower().split()
            for word in words:
                if len(word) > 5:
                    motifs[word] = motifs.get(word, 0) + shard.relevance

        # Top motifs
        top_motifs = dict(sorted(motifs.items(), key=lambda x: -x[1])[:8])

        # Simulated embedding (384D -> just use small array for demo)
        embedding = self.rng.standard_normal(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        # Spectral features (6D graph topology)
        spectral = self.rng.standard_normal(6).astype(np.float32)

        plasma = DotPlasma(motifs=top_motifs, embedding=embedding, spectral=spectral)
        latency = self._sim_latency(15.0)
        return plasma, {
            "stage": "resonance_shed",
            "latency_ms": latency,
            "motifs_extracted": len(top_motifs),
            "embedding_dims": 384,
            "spectral_dims": 6,
        }

    def step5_warp_space(self, plasma: DotPlasma) -> Tuple[np.ndarray, Dict]:
        """Tension threads into continuous manifold, compute, collapse."""
        # Simulate tensor operations in warp space
        tensioned = np.concatenate([
            plasma.embedding[:128],  # Use first 128 dims
            plasma.spectral,
        ])
        # "Compute" — attention-like operation
        result = tensioned * np.tanh(tensioned)

        latency = self._sim_latency(10.0)
        return result, {
            "stage": "warp_space",
            "latency_ms": latency,
            "tensor_shape": result.shape,
            "operations": ["tension", "tanh_gate", "collapse"],
        }

    def step6_memory_retrieval(
        self, threads: List[MemoryShard], query: str
    ) -> Tuple[str, Dict]:
        """Assemble context from retrieved threads."""
        context_parts = []
        for shard in threads[:3]:
            context_parts.append(shard.content)
        context = " | ".join(context_parts)

        latency = self._sim_latency(20.0)
        return context, {
            "stage": "memory_retrieval",
            "latency_ms": latency,
            "context_length": len(context),
            "shards_used": min(3, len(threads)),
        }

    # --- Step 7: Convergence Engine ---

    def step7_convergence(
        self,
        plasma: DotPlasma,
        strategy: CollapseStrategy = CollapseStrategy.BAYESIAN_BLEND,
    ) -> Tuple[str, float, Dict[str, float], Dict]:
        """Collapse probability distribution to discrete tool selection."""

        # Neural network prediction (simulated)
        neural_probs = self.rng.dirichlet(np.array([3.0, 1.0, 0.5, 0.3]))

        # Thompson Sampling priors
        alphas = np.array([5.0, 2.0, 1.5, 1.0])
        betas = np.array([2.0, 3.0, 4.0, 5.0])
        thompson_samples = self.rng.beta(alphas, betas)
        thompson_probs = thompson_samples / thompson_samples.sum()

        # Blend based on strategy
        if strategy == CollapseStrategy.ARGMAX:
            final_probs = neural_probs
        elif strategy == CollapseStrategy.EPSILON_GREEDY:
            if self.rng.random() < 0.1:  # 10% explore
                final_probs = np.ones(4) / 4
            else:
                final_probs = neural_probs
        elif strategy == CollapseStrategy.BAYESIAN_BLEND:
            final_probs = 0.7 * neural_probs + 0.3 * thompson_probs
        elif strategy == CollapseStrategy.PURE_THOMPSON:
            final_probs = thompson_probs
        else:
            final_probs = neural_probs

        final_probs = final_probs / final_probs.sum()
        selected_idx = int(np.argmax(final_probs))
        selected_tool = self.tools[selected_idx]
        confidence = float(final_probs[selected_idx])

        tool_probs = {t: float(p) for t, p in zip(self.tools, final_probs)}

        latency = self._sim_latency(5.0)
        return selected_tool, confidence, tool_probs, {
            "stage": "convergence",
            "latency_ms": latency,
            "strategy": strategy.value,
            "selected_tool": selected_tool,
            "confidence": round(confidence, 4),
            "neural_top": self.tools[int(np.argmax(neural_probs))],
            "thompson_top": self.tools[int(np.argmax(thompson_probs))],
        }

    # --- Step 8: Tool Execution ---

    def step8_tool_execution(
        self, tool: str, context: str, query: str
    ) -> Tuple[str, bool, Dict]:
        """Execute selected tool with safety gating."""

        # Safety check (simulate)
        risk_score = self.rng.random() * 0.3  # Low risk for demo queries
        safety_passed = risk_score < 0.5

        # Generate response based on tool
        responses = {
            "answer": f"Thompson Sampling uses Bayesian posteriors (Beta distributions) to balance exploration of uncertain options with exploitation of known-good ones. Each arm maintains Beta(alpha, beta) parameters updated on success/failure.",
            "research": f"Research mode activated. Exploring: {query}. Found {len(context.split('|'))} relevant sources spanning bandits, Bayesian inference, and decision theory.",
            "clarify": f"To clarify: Thompson Sampling specifically refers to the technique of sampling from posterior distributions to make decisions, as opposed to deterministic approaches like UCB.",
            "code": "```python\nimport numpy as np\n\ndef thompson_sample(alphas, betas):\n    samples = np.random.beta(alphas, betas)\n    return np.argmax(samples)\n```",
        }

        response = responses.get(tool, "Processing...")

        latency = self._sim_latency(25.0)
        return response, safety_passed, {
            "stage": "tool_execution",
            "latency_ms": latency,
            "tool": tool,
            "risk_score": round(risk_score, 4),
            "safety_passed": safety_passed,
            "response_length": len(response),
        }

    # --- Step 9: Spacetime Fabric ---

    def step9_spacetime_fabric(
        self, response: str, confidence: float, tool: str, trace: WeavingTrace
    ) -> Tuple[Spacetime, Dict]:
        """Assemble final Spacetime artifact with provenance."""

        spacetime = Spacetime(
            response=response,
            confidence=confidence,
            tool_used=tool,
            trace=trace,
            metadata={
                "total_duration_ms": trace.total_ms,
                "stages_completed": len(trace.stage_timings),
            },
        )

        latency = self._sim_latency(2.0)
        return spacetime, {
            "stage": "spacetime_fabric",
            "latency_ms": latency,
            "total_pipeline_ms": trace.total_ms + latency,
        }


# ---------------------------------------------------------------------------
# Demo 1: Full 9-Step Pipeline Walk-Through
# ---------------------------------------------------------------------------

def demo_full_pipeline(pipeline: WeavingPipeline):
    """Walk through every stage of the weaving cycle with a sample query."""
    print("=" * 64)
    print("Demo 1: Full 9-Step Weaving Cycle")
    print("=" * 64)
    print()

    query = "Explain Thompson Sampling"
    trace = WeavingTrace()

    print(f"  Query: \"{query}\"")
    print()

    # Step 0
    enhanced, info = pipeline.step0_meta_prompt(query)
    trace.stage_timings["0_meta_prompt"] = info["latency_ms"]
    trace.decisions.append(f"Enhanced: {info['enhanced']}")
    _print_stage(0, "Meta-Prompt Enhancement", info)
    if info["enhanced"]:
        print(f"         Expanded: \"{enhanced[:70]}...\"")
        print()

    # Step 1
    pattern, complexity, info = pipeline.step1_pattern_selection(enhanced)
    trace.stage_timings["1_pattern"] = info["latency_ms"]
    trace.decisions.append(f"Pattern: {pattern.value}, Complexity: {complexity.value}")
    _print_stage(1, "Pattern Selection", info)

    # Step 2
    lookback, recency, info = pipeline.step2_chrono_trigger(complexity)
    trace.stage_timings["2_chrono"] = info["latency_ms"]
    _print_stage(2, "Chrono Trigger", info)

    # Step 3
    threads, info = pipeline.step3_thread_selection(enhanced, lookback, recency)
    trace.stage_timings["3_threads"] = info["latency_ms"]
    _print_stage(3, "Thread Selection", info)
    for shard in threads[:3]:
        print(f"         {shard.id}: \"{shard.content[:55]}...\" (rel={shard.relevance:.3f})")
    print()

    # Steps 4-6 (parallel)
    print("  ┌─ Parallel Execution ─────────────────────────────────────┐")
    print("  │                                                          │")

    plasma, info4 = pipeline.step4_resonance_shed(threads)
    trace.stage_timings["4_resonance"] = info4["latency_ms"]
    _print_stage(4, "Resonance Shed  (features)", info4, indent="  │ ")
    top_motifs = list(plasma.motifs.items())[:4]
    print(f"  │          Motifs: {', '.join(f'{m}({s:.2f})' for m, s in top_motifs)}")
    print("  │")

    warp_result, info5 = pipeline.step5_warp_space(plasma)
    trace.stage_timings["5_warp"] = info5["latency_ms"]
    _print_stage(5, "Warp Space      (tensors)", info5, indent="  │ ")

    context, info6 = pipeline.step6_memory_retrieval(threads, enhanced)
    trace.stage_timings["6_memory"] = info6["latency_ms"]
    _print_stage(6, "Memory Retrieval (context)", info6, indent="  │ ")

    parallel_time = max(info4["latency_ms"], info5["latency_ms"], info6["latency_ms"])
    sequential_time = info4["latency_ms"] + info5["latency_ms"] + info6["latency_ms"]
    print(f"  │                                                          │")
    print(f"  │  Parallel: {parallel_time:.1f}ms (sequential would be {sequential_time:.1f}ms)      │")
    print(f"  └──────────────────────────────────────────────────────────┘")
    print()

    # Step 7
    tool, confidence, tool_probs, info = pipeline.step7_convergence(plasma)
    trace.stage_timings["7_convergence"] = info["latency_ms"]
    trace.decisions.append(f"Tool: {tool} ({confidence:.3f})")
    _print_stage(7, "Convergence Engine", info)
    for t, p in sorted(tool_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        sel = " <-" if t == tool else ""
        print(f"         {t:<10} {p:.3f} {bar}{sel}")
    print()

    # Step 8
    response, safe, info = pipeline.step8_tool_execution(tool, context, query)
    trace.stage_timings["8_execution"] = info["latency_ms"]
    _print_stage(8, "Tool Execution", info)
    print(f"         Response: \"{response[:70]}...\"")
    print()

    # Step 9
    spacetime, info = pipeline.step9_spacetime_fabric(response, confidence, tool, trace)
    trace.stage_timings["9_fabric"] = info["latency_ms"]
    _print_stage(9, "Spacetime Fabric", info)

    print()
    return trace


# ---------------------------------------------------------------------------
# Demo 2: Stage Timing Waterfall
# ---------------------------------------------------------------------------

def demo_timing_waterfall(trace: WeavingTrace):
    """Visualise pipeline stage timings as a waterfall chart."""
    print("=" * 64)
    print("Demo 2: Stage Timing Waterfall")
    print("=" * 64)
    print()

    total = trace.total_ms
    cumulative = 0.0

    stage_labels = {
        "0_meta_prompt": "Meta-Prompt",
        "1_pattern": "Pattern Select",
        "2_chrono": "Chrono Trigger",
        "3_threads": "Thread Select",
        "4_resonance": "Resonance Shed",
        "5_warp": "Warp Space",
        "6_memory": "Memory Retrieve",
        "7_convergence": "Convergence",
        "8_execution": "Tool Execute",
        "9_fabric": "Spacetime",
    }

    print(f"  Total pipeline: {total:.1f}ms")
    print()
    print(f"  {'Stage':<17} {'Time':>7}  {'%':>5}  Waterfall")
    print(f"  {'─' * 17} {'─' * 7}  {'─' * 5}  {'─' * 35}")

    for stage_key, latency in trace.stage_timings.items():
        label = stage_labels.get(stage_key, stage_key)
        pct = (latency / total) * 100 if total > 0 else 0
        offset = int((cumulative / total) * 35) if total > 0 else 0
        bar_len = max(1, int((latency / total) * 35)) if total > 0 else 1

        bar = " " * offset + "█" * bar_len
        bottleneck = " !" if pct > 25 else ""
        print(f"  {label:<17} {latency:>6.1f}ms {pct:>5.1f}% {bar}{bottleneck}")
        cumulative += latency

    print()
    print("  ! = bottleneck (>25% of total time)")
    print()


# ---------------------------------------------------------------------------
# Demo 3: Pattern Card Comparison
# ---------------------------------------------------------------------------

def demo_pattern_cards(pipeline: WeavingPipeline):
    """Compare BARE, FAST, and FUSED execution profiles."""
    print("=" * 64)
    print("Demo 3: Pattern Card Comparison (BARE / FAST / FUSED)")
    print("=" * 64)
    print()

    queries = {
        PatternCard.BARE: "Hello",
        PatternCard.FAST: "Explain Thompson Sampling",
        PatternCard.FUSED: "Analyze the tradeoffs between Thompson Sampling and UCB algorithms for exploration in non-stationary environments",
    }

    profiles = {
        PatternCard.BARE: {
            "steps": "1-3 only",
            "target": "<50ms",
            "features": "Regex motifs, single scale, simple policy",
            "scales": 1,
        },
        PatternCard.FAST: {
            "steps": "1-7",
            "target": "<150ms",
            "features": "Hybrid motifs, 2 scales, neural policy",
            "scales": 2,
        },
        PatternCard.FUSED: {
            "steps": "0-9 (all)",
            "target": "<300ms",
            "features": "All features, 3 scales, multi-scale retrieval",
            "scales": 3,
        },
    }

    for pattern in [PatternCard.BARE, PatternCard.FAST, PatternCard.FUSED]:
        profile = profiles[pattern]
        query = queries[pattern]

        print(f"  ┌─ {pattern.value.upper()} ─────────────────────────────────────────────────┐")
        print(f"  │  Query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")
        print(f"  │  Steps: {profile['steps']:<10}  Target: {profile['target']:<8}  Scales: {profile['scales']}")
        print(f"  │  {profile['features']}")

        # Simulate timing
        base_latency = {"bare": 15, "fast": 60, "fused": 120}[pattern.value]
        stages_active = {"bare": 3, "fast": 7, "fused": 10}[pattern.value]
        timings = []
        for i in range(stages_active):
            t = pipeline._sim_latency(base_latency / stages_active)
            timings.append(t)

        total = sum(timings)
        bar = "█" * int(total / 5) + "░" * max(0, 60 - int(total / 5))
        print(f"  │  Simulated: {total:.1f}ms  {bar[:50]}")
        print(f"  └──────────────────────────────────────────────────────────┘")
        print()


# ---------------------------------------------------------------------------
# Demo 4: Convergence Strategies
# ---------------------------------------------------------------------------

def demo_convergence_strategies(pipeline: WeavingPipeline):
    """Compare tool selection across 4 convergence strategies."""
    print("=" * 64)
    print("Demo 4: Convergence Strategy Comparison")
    print("=" * 64)
    print()
    print("  Four strategies for collapsing probability -> action:")
    print("    ARGMAX:         Always pick highest neural probability")
    print("    EPSILON_GREEDY: 90% exploit neural, 10% random explore")
    print("    BAYESIAN_BLEND: 70% neural + 30% Thompson Sampling")
    print("    PURE_THOMPSON:  100% Thompson Sampling (ignore neural)")
    print()

    plasma = DotPlasma(
        motifs={"sampling": 0.8, "bayesian": 0.7},
        embedding=pipeline.rng.standard_normal(384).astype(np.float32),
        spectral=pipeline.rng.standard_normal(6).astype(np.float32),
    )

    strategies = [
        CollapseStrategy.ARGMAX,
        CollapseStrategy.EPSILON_GREEDY,
        CollapseStrategy.BAYESIAN_BLEND,
        CollapseStrategy.PURE_THOMPSON,
    ]

    # Run multiple trials per strategy
    n_trials = 50
    print(f"  Selection frequency over {n_trials} trials:")
    print()
    print(f"  {'Strategy':<18}", end="")
    for tool in pipeline.tools:
        print(f" {tool:>10}", end="")
    print("    Most Selected")
    print(f"  {'─' * 18}", end="")
    for _ in pipeline.tools:
        print(f" {'─' * 10}", end="")
    print(f"  {'─' * 15}")

    for strategy in strategies:
        counts = {t: 0 for t in pipeline.tools}
        for _ in range(n_trials):
            tool, conf, probs, _ = pipeline.step7_convergence(plasma, strategy)
            counts[tool] += 1

        print(f"  {strategy.value:<18}", end="")
        for tool in pipeline.tools:
            freq = counts[tool] / n_trials
            print(f" {freq:>9.0%}", end="")

        most = max(counts, key=lambda t: counts[t])
        print(f"    {most}")

    print()
    print("  Key insight: ARGMAX is deterministic, Thompson Sampling explores.")
    print("  BAYESIAN_BLEND balances neural certainty with Bayesian uncertainty.")
    print()


# ---------------------------------------------------------------------------
# Demo 5: Pipeline Decision Trace
# ---------------------------------------------------------------------------

def demo_decision_trace(pipeline: WeavingPipeline):
    """Show the complete decision trail for a query — full provenance."""
    print("=" * 64)
    print("Demo 5: Decision Trace (Full Provenance)")
    print("=" * 64)
    print()

    query = "Compare Thompson Sampling with UCB"
    print(f"  Query: \"{query}\"")
    print()

    # Run pipeline, collect all stage info
    enhanced, i0 = pipeline.step0_meta_prompt(query)
    pattern, complexity, i1 = pipeline.step1_pattern_selection(enhanced)
    lookback, recency, i2 = pipeline.step2_chrono_trigger(complexity)
    threads, i3 = pipeline.step3_thread_selection(enhanced, lookback, recency)
    plasma, i4 = pipeline.step4_resonance_shed(threads)
    warp, i5 = pipeline.step5_warp_space(plasma)
    context, i6 = pipeline.step6_memory_retrieval(threads, enhanced)
    tool, conf, probs, i7 = pipeline.step7_convergence(plasma)
    response, safe, i8 = pipeline.step8_tool_execution(tool, context, query)

    all_info = [i0, i1, i2, i3, i4, i5, i6, i7, i8]

    print("  Decision Chain:")
    print()
    for i, info in enumerate(all_info):
        stage = info["stage"]
        latency = info["latency_ms"]
        prefix = "├──" if i < len(all_info) - 1 else "└──"

        # Extract key decision
        decisions = {
            "meta_prompt": f"enhanced={info.get('enhanced', False)}",
            "pattern_selection": f"pattern={info.get('pattern', '?')}, complexity={info.get('complexity', '?')}",
            "chrono_trigger": f"lookback={info.get('lookback_days', '?')}d, bias={info.get('recency_bias', '?')}",
            "thread_selection": f"selected={info.get('selected', '?')}/{info.get('candidates', '?')} shards",
            "resonance_shed": f"motifs={info.get('motifs_extracted', '?')}, embedding={info.get('embedding_dims', '?')}D",
            "warp_space": f"ops={info.get('operations', [])}",
            "memory_retrieval": f"context={info.get('context_length', '?')} chars",
            "convergence": f"tool={info.get('selected_tool', '?')} ({info.get('confidence', '?')}), strategy={info.get('strategy', '?')}",
            "tool_execution": f"risk={info.get('risk_score', '?')}, safe={info.get('safety_passed', '?')}",
        }

        decision = decisions.get(stage, "")
        print(f"  {prefix} Step {i}: {stage}")
        print(f"  {'│' if i < len(all_info) - 1 else ' '}     {latency:.1f}ms | {decision}")

    total_ms = sum(info["latency_ms"] for info in all_info)
    print()
    print(f"  Total: {total_ms:.1f}ms | Tool: {tool} | Confidence: {conf:.3f}")
    print()
    print("  This trace provides complete provenance for every decision,")
    print("  enabling debugging, auditing, and reflection learning.")
    print()


# ---------------------------------------------------------------------------
# Demo 6: Multi-Query Performance Profile
# ---------------------------------------------------------------------------

def demo_performance_profile(pipeline: WeavingPipeline):
    """Run multiple queries and show aggregate performance statistics."""
    print("=" * 64)
    print("Demo 6: Multi-Query Performance Profile")
    print("=" * 64)
    print()

    queries = [
        "Hi",
        "What is Thompson Sampling?",
        "Explain how bandits work",
        "Compare Thompson Sampling with UCB algorithms",
        "Analyze the theoretical convergence properties of Bayesian exploration strategies",
    ]

    all_timings = []
    all_patterns = []

    print(f"  Running {len(queries)} queries...")
    print()
    print(f"  {'Query':<55} {'Pattern':<7} {'Time':>7} {'Tool':>10}")
    print(f"  {'─' * 55} {'─' * 7} {'─' * 7} {'─' * 10}")

    for query in queries:
        enhanced, _ = pipeline.step0_meta_prompt(query, enable=False)
        pattern, complexity, i1 = pipeline.step1_pattern_selection(query)
        lookback, recency, i2 = pipeline.step2_chrono_trigger(complexity)
        threads, i3 = pipeline.step3_thread_selection(query, lookback, recency)
        plasma, i4 = pipeline.step4_resonance_shed(threads)
        warp, i5 = pipeline.step5_warp_space(plasma)
        context, i6 = pipeline.step6_memory_retrieval(threads, query)
        tool, conf, probs, i7 = pipeline.step7_convergence(plasma)

        total = sum(info["latency_ms"] for info in [i1, i2, i3, i4, i5, i6, i7])
        all_timings.append(total)
        all_patterns.append(pattern.value)

        q_display = query[:52] + "..." if len(query) > 52 else query
        print(f"  {q_display:<55} {pattern.value:<7} {total:>6.1f}ms {tool:>10}")

    print()
    print(f"  Aggregate Statistics:")
    print(f"    Mean latency:   {np.mean(all_timings):>7.1f}ms")
    print(f"    Median latency: {np.median(all_timings):>7.1f}ms")
    print(f"    P95 latency:    {np.percentile(all_timings, 95):>7.1f}ms")
    print(f"    Min / Max:      {min(all_timings):>6.1f} / {max(all_timings):.1f}ms")
    print()

    # Pattern distribution
    from collections import Counter
    pattern_counts = Counter(all_patterns)
    print(f"  Pattern Distribution:")
    for p in ["bare", "fast", "fused"]:
        count = pattern_counts.get(p, 0)
        bar = "█" * (count * 4) + "░" * ((5 - count) * 4)
        print(f"    {p:<7} {count}/{len(queries)}  {bar}")
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_stage(step: int, name: str, info: Dict, indent: str = "  "):
    """Print a single stage summary."""
    latency = info["latency_ms"]
    keys = {k: v for k, v in info.items() if k not in ("stage", "latency_ms")}
    detail = ", ".join(f"{k}={v}" for k, v in list(keys.items())[:3])
    print(f"{indent}Step {step}: {name}")
    print(f"{indent}       {latency:.1f}ms | {detail}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║       HoloLoom Weaving Pipeline Showcase                  ║")
    print("  ║       The 9-Step Orchestration Cycle                      ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()

    pipeline = WeavingPipeline(seed=42)

    trace = demo_full_pipeline(pipeline)
    demo_timing_waterfall(trace)
    demo_pattern_cards(pipeline)
    demo_convergence_strategies(pipeline)
    demo_decision_trace(pipeline)
    demo_performance_profile(pipeline)

    print("  " + "═" * 62)
    print("  All demonstrations complete.")
    print()


if __name__ == "__main__":
    main()
