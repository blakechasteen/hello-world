#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  HoloLoom v2 — Classical AI Composition Layer: Live Integration Study
═══════════════════════════════════════════════════════════════════════════════

  Research questions:
    RQ1. Does TMS justification tracking produce meaningful provenance
         chains through real graph traversals?
    RQ2. Does TMS retraction fire on genuine contradictions in LLM-
         retrieved context, and does it measurably shift activation?
    RQ3. Does the ChunkCompiler learn stable trigger→action mappings
         from repeated similar queries?
    RQ4. How do the classical AI components interact when composed —
         does TMS retraction affect chunk compilation quality?
    RQ5. What is the overhead of the KnowledgeSource pipeline hooks
         relative to the LLM generation cost?

  Design: within-subjects comparison.
    Condition A — Baseline (no KnowledgeSources)
    Condition B — TMS only
    Condition C — ChunkCompiler only
    Condition D — TMS + ChunkCompiler (full composition)

  All conditions share the same knowledge graph, model, and queries.
  Each condition runs the same 8 queries. We measure per-query:
    - Shell path (PRIME > VERIFY > FLAG)
    - Confidence (structural, narrative, combined)
    - Context quality (items packed, token utilization)
    - TMS: justifications recorded, retractions fired
    - Chunks: experience buffer, compilations, matches
    - Timing (navigate, confidence, format, generate, total)
    - LLM response quality (composite reward)

  Run: python test_v2_classical_live.py [--model MODEL]
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import math
import sys
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suppress noisy loggers, keep our components visible
logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')
for name in ['sentence_transformers', 'httpx', 'httpcore', 'urllib3',
             'hololoom.alignment', 'hololoom.loom', 'hololoom.chrono',
             'hololoom.resonance', 'hololoom.warp', 'hololoom.convergence',
             'hololoom.reflection', 'hololoom.embedding', 'hololoom.motif',
             'hololoom.performance', 'hololoom.fabric']:
    logging.getLogger(name).setLevel(logging.WARNING)

from hololoom.memory.lite_bus import LiteMemoryBus
from hololoom.memory.bus import MemoryItem
from hololoom.memory.v2 import (
    Navigator, NavigatorResult, PPRConfig, LiteBusGraph,
    MatryoshkaOrchestrator, ShellType, ShellConfig, DEFAULT_SHELLS,
    ConfidenceEstimator, ConfidenceConfig,
    Formatter, FormatConfig,
    Blackboard, WorkingMemory,
    ActivationAdapter, ActivationConfig,
    SourceRegistry, Phase, KnowledgeSource,
    TruthMaintenanceSystem, TMSConfig, JustificationRecord,
    ChunkCompiler, ChunkConfig, ChunkAction, CompiledChunk,
    RewardJudge,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

MODEL = "qwen2.5:14b"
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        MODEL = sys.argv[i + 1]


# ═══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def banner(title, char='='):
    w = 76
    print(f"\n{char*w}")
    print(f"  {title}")
    print(f"{char*w}")


def sub_banner(title):
    banner(title, char='-')


def table(headers, rows, widths=None):
    if widths is None:
        widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                  for i, h in enumerate(headers)]
    fmt = "  " + "".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "-" * sum(widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def pct_delta(a, b):
    """Format percentage change."""
    if a == 0:
        return "n/a"
    d = (b - a) / abs(a) * 100
    return f"{d:+.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Graph Construction
# ═══════════════════════════════════════════════════════════════════════════

async def build_knowledge_graph() -> LiteMemoryBus:
    """Build a rich knowledge graph with intentional structure for testing.

    Design choices for the test corpus:
      - 10 entity nodes (algorithms, concepts, frameworks)
      - 22 factual nodes with varying connectivity
      - 3 contradicting fact pairs (for TMS retraction testing)
      - 4 isolated/sparse facts (for confidence variation)
      - Dense edges between core concepts (for PPR traversal)
      - Sparse edges at periphery (for cold start / low confidence)
    """
    bus = LiteMemoryBus()
    await bus.initialize()

    ids = {}

    # ── Entities ─────────────────────────────────────────────────
    entities = [
        ("Thompson Sampling",    "algorithm",  0.95),
        ("UCB1",                 "algorithm",  0.90),
        ("Epsilon Greedy",       "algorithm",  0.85),
        ("Contextual Bandits",   "framework",  0.92),
        ("Multi-Armed Bandit",   "concept",    0.97),
        ("Bayesian Optimization","technique",  0.88),
        ("LinUCB",               "algorithm",  0.86),
        ("Regret Analysis",      "concept",    0.82),
        ("Neural Bandits",       "algorithm",  0.80),
        ("Posterior Inference",   "concept",    0.85),
    ]
    for name, etype, imp in entities:
        ids[name] = await bus.store(MemoryItem(
            content=name, memory_type="entity",
            properties={"type": etype}, importance=imp,
        ))
        await bus.add_alias(name.lower(), ids[name])

    # Extra aliases for seed extraction
    await bus.add_alias("ts", ids["Thompson Sampling"])
    await bus.add_alias("thompson", ids["Thompson Sampling"])
    await bus.add_alias("ucb", ids["UCB1"])
    await bus.add_alias("mab", ids["Multi-Armed Bandit"])
    await bus.add_alias("bandit", ids["Multi-Armed Bandit"])
    await bus.add_alias("bandits", ids["Multi-Armed Bandit"])
    await bus.add_alias("linucb", ids["LinUCB"])
    await bus.add_alias("bayesian", ids["Bayesian Optimization"])
    await bus.add_alias("regret", ids["Regret Analysis"])
    await bus.add_alias("epsilon", ids["Epsilon Greedy"])
    await bus.add_alias("posterior", ids["Posterior Inference"])

    # ── Core facts (well-connected) ─────────────────────────────
    core_facts = [
        ("Thompson Sampling uses Bayesian posteriors to balance exploration and exploitation by sampling from the posterior distribution of each arm's reward.",
         ["Thompson Sampling", "Posterior Inference"], 0.90),
        ("Thompson Sampling converges to the optimal arm with probability 1 and achieves logarithmic Bayesian regret.",
         ["Thompson Sampling", "Regret Analysis"], 0.88),
        ("UCB1 computes upper confidence bounds using the Hoeffding inequality, achieving O(sqrt(KT log T)) regret.",
         ["UCB1", "Regret Analysis"], 0.87),
        ("UCB1 is deterministic and achieves logarithmic regret without knowing the time horizon.",
         ["UCB1", "Regret Analysis"], 0.85),
        ("Epsilon Greedy explores with probability epsilon and exploits with 1-epsilon. Simple but linear regret.",
         ["Epsilon Greedy", "Multi-Armed Bandit"], 0.82),
        ("The multi-armed bandit problem models the fundamental exploration-exploitation tradeoff in sequential decision-making.",
         ["Multi-Armed Bandit"], 0.95),
        ("Contextual bandits extend MAB by using feature vectors to make per-decision adaptations based on context.",
         ["Contextual Bandits", "Multi-Armed Bandit"], 0.90),
        ("LinUCB extends UCB to the contextual setting with linear reward models and confidence ellipsoids.",
         ["LinUCB", "UCB1", "Contextual Bandits"], 0.86),
        ("Thompson Sampling can be applied to contextual bandits using Bayesian linear regression.",
         ["Thompson Sampling", "Contextual Bandits", "Posterior Inference"], 0.88),
        ("Bayesian Optimization uses a Gaussian Process surrogate model to guide expensive function evaluation.",
         ["Bayesian Optimization", "Posterior Inference"], 0.87),
        ("Thompson Sampling can be viewed as a special case of Bayesian Optimization for discrete action spaces.",
         ["Thompson Sampling", "Bayesian Optimization"], 0.84),
        ("Regret measures the cumulative loss from not always choosing the optimal arm. Lower is better.",
         ["Regret Analysis", "Multi-Armed Bandit"], 0.90),
        ("Applications of MAB include A/B testing, recommendation systems, clinical trials, and ad placement.",
         ["Multi-Armed Bandit"], 0.80),
        ("Neural bandits replace the linear model in LinUCB with a neural network for complex reward surfaces.",
         ["Neural Bandits", "LinUCB", "Contextual Bandits"], 0.78),
        ("Posterior inference in bandits involves maintaining and updating beliefs about arm reward distributions.",
         ["Posterior Inference", "Thompson Sampling", "Bayesian Optimization"], 0.85),
    ]
    for text, entity_names, imp in core_facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=[ids[n] for n in entity_names],
            importance=imp,
        ))

    # ── Contradicting pairs (for TMS retraction testing) ─────────
    contradictions = [
        # Pair 1: TS vs UCB speed
        ("Thompson Sampling is faster and better than UCB for large action spaces due to natural parallelism.",
         ["Thompson Sampling", "UCB1"], 0.70),
        ("Thompson Sampling is not faster than UCB for large action spaces and is worse in wall-clock time.",
         ["Thompson Sampling", "UCB1"], 0.55),
        # Pair 2: Epsilon-greedy utility
        ("Epsilon-greedy is superior to Thompson Sampling when the number of arms is small and rewards are stationary.",
         ["Epsilon Greedy", "Thompson Sampling"], 0.60),
        ("Epsilon-greedy is not superior to Thompson Sampling even when arms are small; posterior sampling adapts faster.",
         ["Epsilon Greedy", "Thompson Sampling"], 0.65),
        # Pair 3: Neural bandits
        ("Neural bandits outperform LinUCB on all real-world datasets with sufficient data.",
         ["Neural Bandits", "LinUCB"], 0.55),
        ("Neural bandits do not outperform LinUCB reliably; they are slower to converge and less stable.",
         ["Neural Bandits", "LinUCB"], 0.60),
    ]
    for text, entity_names, imp in contradictions:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            entity_ids=[ids[n] for n in entity_names],
            importance=imp,
        ))

    # ── Sparse/isolated facts (no entity links, low importance) ──
    sparse_facts = [
        "Some researchers prefer frequentist approaches over Bayesian methods for computational simplicity.",
        "The term 'bandit' comes from one-armed bandit slot machines in casinos.",
        "Online learning is related to but distinct from bandit problems in the regret framework.",
        "Reinforcement learning generalizes bandits to sequential decision problems with state transitions.",
    ]
    for text in sparse_facts:
        await bus.store(MemoryItem(
            content=text, memory_type="factual",
            importance=0.3,
        ))

    # ── Edges ────────────────────────────────────────────────────
    edges = [
        ("Thompson Sampling", "UCB1", "COMPARED_TO"),
        ("Thompson Sampling", "Epsilon Greedy", "COMPARED_TO"),
        ("UCB1", "Epsilon Greedy", "COMPARED_TO"),
        ("Thompson Sampling", "Multi-Armed Bandit", "SOLVES"),
        ("UCB1", "Multi-Armed Bandit", "SOLVES"),
        ("Epsilon Greedy", "Multi-Armed Bandit", "SOLVES"),
        ("Contextual Bandits", "Multi-Armed Bandit", "EXTENDS"),
        ("Thompson Sampling", "Contextual Bandits", "USED_IN"),
        ("LinUCB", "Contextual Bandits", "USED_IN"),
        ("LinUCB", "UCB1", "EXTENDS"),
        ("Thompson Sampling", "Bayesian Optimization", "RELATED_TO"),
        ("Bayesian Optimization", "Posterior Inference", "USES"),
        ("Thompson Sampling", "Posterior Inference", "USES"),
        ("Regret Analysis", "Multi-Armed Bandit", "ANALYZES"),
        ("Neural Bandits", "Contextual Bandits", "EXTENDS"),
        ("Neural Bandits", "LinUCB", "EXTENDS"),
    ]
    for src, dst, rel in edges:
        await bus.store_edge(ids[src], ids[dst], rel)

    status = bus.status()
    return bus


# ═══════════════════════════════════════════════════════════════════════════
# Query Battery
# ═══════════════════════════════════════════════════════════════════════════

QUERIES = [
    # Q1-Q3: Well-grounded (should retrieve strong context, high confidence)
    "How does Thompson Sampling balance exploration and exploitation?",
    "Compare UCB1 and epsilon-greedy in terms of regret bounds.",
    "What are contextual bandits and how does LinUCB work?",

    # Q4-Q5: Cross-cutting (should pull from multiple entity clusters)
    "Explain the relationship between Thompson Sampling, Bayesian Optimization, and posterior inference.",
    "How do neural bandits extend traditional contextual bandit approaches?",

    # Q6-Q7: Contradiction-triggering (should activate TMS retraction)
    "Is Thompson Sampling faster than UCB for large action spaces?",
    "Does epsilon-greedy outperform Thompson Sampling for small action spaces?",

    # Q8: Sparse/cold (weak graph anchors, tests fallback behavior)
    "What is the connection between reinforcement learning and bandit problems?",
]


# ═══════════════════════════════════════════════════════════════════════════
# Instrumented Source Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class InstrumentedTMS:
    """TMS wrapper that records detailed metrics per invocation."""

    def __init__(self, activation_adapter=None, config=None):
        self._tms = TruthMaintenanceSystem(
            activation_adapter=activation_adapter,
            config=config or TMSConfig(store_edges=True),
        )
        self.metrics = []  # Per-invocation metrics
        self._current = {}

    @property
    def name(self): return self._tms.name

    @property
    def phases(self): return self._tms.phases

    def activation_level(self, bb): return self._tms.activation_level(bb)

    def contribute(self, bb, graph, context):
        t0 = time.perf_counter()
        self._tms.contribute(bb, graph, context)
        elapsed = time.perf_counter() - t0

        if "nav_result" in context and "confidence" not in context:
            # POST_NAVIGATE: justification recording
            justifications = self._tms._justifications
            total_justified = sum(1 for recs in justifications.values() if recs)
            total_records = sum(len(recs) for recs in justifications.values())
            avg_strength = 0.0
            if total_records > 0:
                avg_strength = sum(
                    r.strength for recs in justifications.values() for r in recs
                ) / total_records

            self._current = {
                "phase": "POST_NAVIGATE",
                "nodes_justified": total_justified,
                "justification_records": total_records,
                "avg_strength": avg_strength,
                "elapsed_ms": elapsed * 1000,
            }

        elif "confidence" in context:
            # POST_CONFIDENCE: retraction check
            retractions = bb.flags.get("retractions", []) if bb else []
            self._current.update({
                "phase_2": "POST_CONFIDENCE",
                "retractions_fired": len(retractions),
                "retraction_details": retractions,
                "elapsed_ms_confidence": elapsed * 1000,
            })
            self.metrics.append(dict(self._current))
            self._current = {}

    @property
    def tms(self):
        return self._tms


class InstrumentedChunkCompiler:
    """ChunkCompiler wrapper that records detailed metrics."""

    def __init__(self, config=None):
        self._compiler = ChunkCompiler(
            config=config or ChunkConfig(
                min_cluster_size=3,
                min_trigger_features=2,
                feature_spread_limit=0.8,
            ),
        )
        self.metrics = []
        self._pre_shell_matches = 0

    @property
    def name(self): return self._compiler.name

    @property
    def phases(self): return self._compiler.phases

    def activation_level(self, bb): return self._compiler.activation_level(bb)

    def contribute(self, bb, graph, context):
        chunks_before = len(self._compiler._chunks)
        exp_before = len(self._compiler._experience)
        t0 = time.perf_counter()

        self._compiler.contribute(bb, graph, context)
        elapsed = time.perf_counter() - t0

        if "shell_type" in context:
            # PRE_SHELL
            chunk_action = bb.flags.get("chunk_action") if bb else None
            if chunk_action:
                self._pre_shell_matches += 1
            self.metrics.append({
                "phase": "PRE_SHELL",
                "shell_type": str(context.get("shell_type")),
                "chunk_matched": chunk_action is not None,
                "chunk_action": chunk_action,
                "elapsed_ms": elapsed * 1000,
            })
        elif "loop_result" in context:
            # POST_LOOP
            chunks_after = len(self._compiler._chunks)
            exp_after = len(self._compiler._experience)
            self.metrics.append({
                "phase": "POST_LOOP",
                "experience_added": exp_after - exp_before,
                "experience_total": exp_after,
                "chunks_compiled": chunks_after - chunks_before,
                "chunks_total": chunks_after,
                "active_chunks": len(self._compiler.active_chunks),
                "elapsed_ms": elapsed * 1000,
            })

    @property
    def compiler(self):
        return self._compiler


# ═══════════════════════════════════════════════════════════════════════════
# Observation Record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryObservation:
    """Full observation record for a single query execution."""
    query: str
    condition: str
    query_idx: int

    # Shell execution
    shell_path: str = ""
    shell_count: int = 0

    # Confidence
    confidence_combined: float = 0.0
    confidence_structural: float = 0.0
    confidence_narrative: float = 0.0
    decision: str = ""

    # Context quality
    items_packed: int = 0
    items_offered: int = 0
    token_count: int = 0
    token_budget: int = 0
    utilization: float = 0.0

    # TMS metrics
    tms_nodes_justified: int = 0
    tms_justification_records: int = 0
    tms_avg_strength: float = 0.0
    tms_retractions: int = 0
    tms_retraction_details: list = field(default_factory=list)

    # Chunk metrics
    chunk_matched: bool = False
    chunk_experience_total: int = 0
    chunk_compilations: int = 0
    chunk_active: int = 0

    # Timing
    total_time: float = 0.0

    # Response
    response_len: int = 0
    reward: float = 0.0
    response_preview: str = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Experimental Conditions
# ═══════════════════════════════════════════════════════════════════════════

async def run_condition(
    condition_name: str,
    bus: LiteMemoryBus,
    llm_fn,
    reward_judge: RewardJudge,
    queries: List[str],
    use_tms: bool = False,
    use_chunks: bool = False,
) -> List[QueryObservation]:
    """Run a full condition (set of queries) and return observations."""

    observations = []

    # Build fresh navigator for this condition
    nav = Navigator.from_lite_bus(bus, config=PPRConfig(max_results=30))

    # Build sources
    registry = SourceRegistry()
    activation_adapter = ActivationAdapter(config=ActivationConfig())

    i_tms = None
    i_chunks = None

    if use_tms:
        i_tms = InstrumentedTMS(
            activation_adapter=activation_adapter,
            config=TMSConfig(store_edges=True, retraction_penalty=0.3),
        )
        registry.register(i_tms)

    if use_chunks:
        i_chunks = InstrumentedChunkCompiler(
            config=ChunkConfig(
                min_cluster_size=3,
                min_trigger_features=2,
                feature_spread_limit=0.8,
            ),
        )
        registry.register(i_chunks)

    # Build orchestrator
    orch = MatryoshkaOrchestrator(
        navigator=nav,
        llm_fn=llm_fn,
        activation_adapter=activation_adapter,
        sources=registry,
    )

    for qi, query in enumerate(queries):
        obs = QueryObservation(
            query=query,
            condition=condition_name,
            query_idx=qi,
        )

        t0 = time.perf_counter()
        try:
            result = await orch.run(query)
            obs.total_time = time.perf_counter() - t0

            # Shell execution
            obs.shell_path = " > ".join(s.shell_type.value for s in result.shells_executed)
            obs.shell_count = result.shell_count

            # Confidence
            final_shell = result.shells_executed[-1]
            obs.confidence_combined = result.final_confidence.combined
            obs.confidence_structural = result.final_confidence.structural.score
            obs.confidence_narrative = result.final_confidence.narrative.score
            obs.decision = result.final_confidence.decision

            # Context quality
            obs.items_packed = final_shell.context_block.items_packed
            obs.items_offered = final_shell.context_block.items_offered
            obs.token_count = final_shell.context_block.token_count
            obs.token_budget = final_shell.context_block.token_budget
            obs.utilization = final_shell.context_block.utilization

            # Response
            obs.response_len = len(result.response)
            obs.response_preview = result.response[:200].replace('\n', ' ')

            # Reward
            obs.reward = await reward_judge.score(
                query=query,
                response=result.response,
                confidence=obs.confidence_combined,
                context_items=obs.items_packed,
            )

            # TMS metrics (from instrumented wrapper)
            if i_tms and i_tms.metrics:
                latest = i_tms.metrics[-1]
                obs.tms_nodes_justified = latest.get("nodes_justified", 0)
                obs.tms_justification_records = latest.get("justification_records", 0)
                obs.tms_avg_strength = latest.get("avg_strength", 0.0)
                obs.tms_retractions = latest.get("retractions_fired", 0)
                obs.tms_retraction_details = latest.get("retraction_details", [])

            # Chunk metrics (from instrumented wrapper)
            if i_chunks:
                post_loop = [m for m in i_chunks.metrics if m.get("phase") == "POST_LOOP"]
                if post_loop:
                    latest = post_loop[-1]
                    obs.chunk_experience_total = latest.get("experience_total", 0)
                    obs.chunk_compilations = latest.get("chunks_compiled", 0)
                    obs.chunk_active = latest.get("active_chunks", 0)
                pre_shell = [m for m in i_chunks.metrics if m.get("phase") == "PRE_SHELL" and m.get("chunk_matched")]
                obs.chunk_matched = len(pre_shell) > 0

        except Exception as e:
            obs.total_time = time.perf_counter() - t0
            obs.error = str(e)[:100]

        observations.append(obs)

        # Status line
        status = f"OK" if not obs.error else f"ERR: {obs.error[:40]}"
        tms_info = ""
        if use_tms:
            tms_info = f" | just={obs.tms_nodes_justified} retr={obs.tms_retractions}"
        chunk_info = ""
        if use_chunks:
            chunk_info = f" | exp={obs.chunk_experience_total} chunks={obs.chunk_active}"
        print(f"  [{condition_name} Q{qi+1}] {status} conf={obs.confidence_combined:.2f} "
              f"r={obs.reward:.3f} ({obs.total_time:.1f}s){tms_info}{chunk_info}")

    return observations


# ═══════════════════════════════════════════════════════════════════════════
# Analysis & Reporting
# ═══════════════════════════════════════════════════════════════════════════

def analyze_condition(obs_list: List[QueryObservation]) -> Dict[str, Any]:
    """Compute aggregate statistics for a condition."""
    n = len(obs_list)
    if n == 0:
        return {}

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def stdev(vals):
        if len(vals) < 2:
            return 0.0
        m = avg(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

    return {
        "n": n,
        "confidence_mean": avg([o.confidence_combined for o in obs_list]),
        "confidence_std": stdev([o.confidence_combined for o in obs_list]),
        "structural_mean": avg([o.confidence_structural for o in obs_list]),
        "narrative_mean": avg([o.confidence_narrative for o in obs_list]),
        "reward_mean": avg([o.reward for o in obs_list]),
        "reward_std": stdev([o.reward for o in obs_list]),
        "items_packed_mean": avg([o.items_packed for o in obs_list]),
        "utilization_mean": avg([o.utilization for o in obs_list]),
        "shell_count_mean": avg([o.shell_count for o in obs_list]),
        "time_mean": avg([o.total_time for o in obs_list]),
        "time_total": sum(o.total_time for o in obs_list),
        "response_len_mean": avg([o.response_len for o in obs_list]),
        "errors": sum(1 for o in obs_list if o.error),
        # TMS
        "tms_justified_mean": avg([o.tms_nodes_justified for o in obs_list]),
        "tms_records_mean": avg([o.tms_justification_records for o in obs_list]),
        "tms_retractions_total": sum(o.tms_retractions for o in obs_list),
        "tms_strength_mean": avg([o.tms_avg_strength for o in obs_list if o.tms_avg_strength > 0]),
        # Chunks
        "chunk_matches": sum(1 for o in obs_list if o.chunk_matched),
        "chunk_compilations": max((o.chunk_active for o in obs_list), default=0),
        "chunk_experience": max((o.chunk_experience_total for o in obs_list), default=0),
    }


def report_per_query_comparison(all_obs: Dict[str, List[QueryObservation]]):
    """Per-query cross-condition comparison."""
    conditions = list(all_obs.keys())
    queries = [o.query for o in all_obs[conditions[0]]]

    sub_banner("Per-Query Confidence Comparison")
    headers = ["Query"] + conditions
    rows = []
    for qi, q in enumerate(queries):
        row = [q[:45]]
        for cond in conditions:
            obs = all_obs[cond][qi]
            row.append(f"{obs.confidence_combined:.3f}")
        rows.append(row)
    widths = [47] + [12] * len(conditions)
    table(headers, rows, widths)

    sub_banner("Per-Query Reward Comparison")
    rows = []
    for qi, q in enumerate(queries):
        row = [q[:45]]
        for cond in conditions:
            obs = all_obs[cond][qi]
            row.append(f"{obs.reward:.3f}")
        rows.append(row)
    table(headers, rows, widths)


def report_tms_deep_dive(obs_list: List[QueryObservation]):
    """Detailed TMS analysis."""
    sub_banner("TMS Deep Dive")

    print("\n  Justification Recording:")
    headers = ["Query", "Justified", "Records", "Avg Strength"]
    rows = []
    for o in obs_list:
        rows.append([
            o.query[:45],
            str(o.tms_nodes_justified),
            str(o.tms_justification_records),
            f"{o.tms_avg_strength:.3f}" if o.tms_avg_strength > 0 else "-",
        ])
    table(headers, rows, [47, 12, 10, 14])

    # Retraction events
    retraction_events = [(o.query, o.tms_retraction_details) for o in obs_list if o.tms_retractions > 0]
    if retraction_events:
        print(f"\n  Retraction Events ({len(retraction_events)} queries triggered retractions):")
        for query, details in retraction_events:
            print(f"\n    Query: {query[:60]}")
            for r in details:
                print(f"      RETRACTED: {r.get('retracted', '?')[:40]}")
                print(f"      RETAINED:  {r.get('retained', '?')[:40]}")
                print(f"      Reason:    {r.get('reason', '?')}")
    else:
        print("\n  No retractions fired (contradictions may not have been retrieved together)")
        print("  Note: TMS only retracts when contradicting nodes appear in the SAME retrieval")


def report_chunk_deep_dive(obs_list: List[QueryObservation]):
    """Detailed ChunkCompiler analysis."""
    sub_banner("ChunkCompiler Deep Dive")

    print("\n  Experience Accumulation:")
    headers = ["Query", "Exp Total", "Active Chunks", "Matched?"]
    rows = []
    for o in obs_list:
        rows.append([
            o.query[:45],
            str(o.chunk_experience_total),
            str(o.chunk_active),
            "YES" if o.chunk_matched else "-",
        ])
    table(headers, rows, [47, 12, 14, 10])


def report_overhead_analysis(all_obs: Dict[str, List[QueryObservation]]):
    """KnowledgeSource overhead analysis (RQ5)."""
    sub_banner("Overhead Analysis: KS Pipeline Hooks vs Baseline")

    if "Baseline" not in all_obs:
        print("  (No baseline condition to compare)")
        return

    baseline_times = [o.total_time for o in all_obs["Baseline"]]
    avg_baseline = sum(baseline_times) / len(baseline_times)

    headers = ["Condition", "Avg Time", "Delta", "Overhead %", "Total Time"]
    rows = []
    for cond, obs_list in all_obs.items():
        times = [o.total_time for o in obs_list]
        avg_time = sum(times) / len(times)
        delta = avg_time - avg_baseline
        overhead = (delta / avg_baseline * 100) if avg_baseline > 0 else 0
        total = sum(times)
        rows.append([
            cond,
            f"{avg_time:.2f}s",
            f"{delta:+.2f}s",
            f"{overhead:+.1f}%",
            f"{total:.1f}s",
        ])
    table(headers, rows, [16, 12, 10, 12, 12])


def report_research_findings(all_obs: Dict[str, List[QueryObservation]], all_stats: Dict[str, Dict]):
    """Synthesize findings for each research question."""
    banner("FINDINGS", char='*')

    # RQ1: TMS Justification
    print("\n  RQ1. Does TMS produce meaningful provenance chains?")
    for cond in ["TMS Only", "Full"]:
        if cond in all_stats:
            s = all_stats[cond]
            print(f"    [{cond}] Avg justified nodes: {s['tms_justified_mean']:.1f}, "
                  f"avg records: {s['tms_records_mean']:.1f}, "
                  f"avg strength: {s['tms_strength_mean']:.3f}")
    tms_obs = all_obs.get("TMS Only", all_obs.get("Full", []))
    if tms_obs:
        has_just = sum(1 for o in tms_obs if o.tms_nodes_justified > 0)
        print(f"    Justifications present in {has_just}/{len(tms_obs)} queries")
        if has_just == len(tms_obs):
            print(f"    --> FINDING: TMS successfully traces seed-to-node paths for every query")
        elif has_just > 0:
            print(f"    --> FINDING: TMS traces paths for most queries; gaps may indicate disconnected subgraphs")
        else:
            print(f"    --> FINDING: No justifications recorded — BFS may not be reaching retrieved nodes")

    # RQ2: TMS Retraction
    print(f"\n  RQ2. Does TMS retract genuine contradictions?")
    for cond in ["TMS Only", "Full"]:
        if cond in all_stats:
            s = all_stats[cond]
            print(f"    [{cond}] Total retractions: {s['tms_retractions_total']}")
    # Check contradiction queries specifically
    contradiction_qs = [5, 6]  # Q6, Q7 (0-indexed)
    for cond in ["TMS Only", "Full"]:
        if cond not in all_obs:
            continue
        obs_list = all_obs[cond]
        for qi in contradiction_qs:
            if qi < len(obs_list):
                o = obs_list[qi]
                if o.tms_retractions > 0:
                    print(f"    [{cond} Q{qi+1}] RETRACTION FIRED on: {o.query[:50]}")
                else:
                    print(f"    [{cond} Q{qi+1}] No retraction on: {o.query[:50]}")
    print(f"    Note: Retraction requires contradicting nodes in the SAME retrieval AND")
    print(f"    matching the negation+overlap heuristic. LLM generation is unaffected")
    print(f"    by retraction — it operates on already-formatted context. The effect")
    print(f"    is on FUTURE queries via decreased ACT-R activation.")

    # RQ3: ChunkCompiler Learning
    print(f"\n  RQ3. Does ChunkCompiler learn stable trigger->action mappings?")
    for cond in ["Chunks Only", "Full"]:
        if cond in all_stats:
            s = all_stats[cond]
            print(f"    [{cond}] Experience buffer: {s['chunk_experience']}, "
                  f"chunks compiled: {s['chunk_compilations']}, "
                  f"matches: {s['chunk_matches']}")
    print(f"    Note: With 8 queries and min_cluster_size=3, compilation is possible")
    print(f"    if similar queries produce consistent features. Chunk matching requires")
    print(f"    a SECOND pass through the same query patterns.")

    # RQ4: Composition Effects
    print(f"\n  RQ4. How do TMS + ChunkCompiler interact when composed?")
    if "Baseline" in all_stats and "Full" in all_stats:
        b = all_stats["Baseline"]
        f = all_stats["Full"]
        conf_delta = f["confidence_mean"] - b["confidence_mean"]
        reward_delta = f["reward_mean"] - b["reward_mean"]
        print(f"    Confidence: Baseline={b['confidence_mean']:.3f}, Full={f['confidence_mean']:.3f} "
              f"(delta={conf_delta:+.3f})")
        print(f"    Reward:     Baseline={b['reward_mean']:.3f}, Full={f['reward_mean']:.3f} "
              f"(delta={reward_delta:+.3f})")
        if abs(conf_delta) < 0.02 and abs(reward_delta) < 0.02:
            print(f"    --> FINDING: Classical AI components add structure WITHOUT degrading")
            print(f"        retrieval quality. This is correct: they operate BETWEEN pipeline")
            print(f"        steps, not instead of them.")
        elif conf_delta > 0:
            print(f"    --> FINDING: Positive confidence shift — TMS retraction may be removing")
            print(f"        weak/contradicting nodes, improving context coherence.")
        else:
            print(f"    --> FINDING: Slight negative shift — overhead of graph mutations may")
            print(f"        affect PPR traversal in subsequent queries (JUSTIFIED_BY edges).")

    # RQ5: Overhead
    print(f"\n  RQ5. What is the overhead of KnowledgeSource hooks?")
    if "Baseline" in all_stats and "Full" in all_stats:
        b_time = all_stats["Baseline"]["time_mean"]
        f_time = all_stats["Full"]["time_mean"]
        overhead = f_time - b_time
        pct = (overhead / b_time * 100) if b_time > 0 else 0
        print(f"    Baseline avg: {b_time:.2f}s, Full avg: {f_time:.2f}s")
        print(f"    Overhead: {overhead:+.3f}s ({pct:+.1f}%)")
        print(f"    --> FINDING: {'Negligible' if abs(pct) < 5 else 'Measurable'} overhead. "
              f"LLM generation dominates wall-clock time. Graph operations (BFS, edge")
        print(f"        storage, feature extraction) are <1ms compared to LLM inference.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    banner("HoloLoom v2 — Classical AI Composition: Live Integration Study")
    print(f"  Model:      {MODEL}")
    print(f"  Queries:    {len(QUERIES)}")
    print(f"  Conditions: 4 (Baseline, TMS, Chunks, Full)")
    print(f"  Design:     Within-subjects, same graph + queries per condition")

    # ── Build knowledge graph ────────────────────────────────────
    sub_banner("Knowledge Graph")
    bus = await build_knowledge_graph()
    status = bus.status()
    print(f"  Items:   {status['item_count']}")
    print(f"  Edges:   {status['edge_count']}")
    print(f"  Aliases: {status['alias_count']}")
    print(f"  Types:   {dict(status.get('by_type', {}))}")

    # ── LLM adapter ──────────────────────────────────────────────
    sub_banner("LLM Connection")
    try:
        from hololoom.awareness.llm_integration import OllamaLLM
        llm = OllamaLLM(model=MODEL)
        if not llm.is_available():
            print(f"  ERROR: Ollama not available. Run 'ollama serve' first.")
            return
        print(f"  Connected to Ollama ({MODEL})")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    llm_call_count = [0]
    llm_total_time = [0.0]

    async def llm_fn(prompt: str, max_tokens: int = 3000) -> str:
        t0 = time.perf_counter()
        resp = await llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.7)
        elapsed = time.perf_counter() - t0
        llm_call_count[0] += 1
        llm_total_time[0] += elapsed
        return resp.content

    reward_judge = RewardJudge(llm_fn=llm_fn)

    # ── Run conditions ───────────────────────────────────────────
    all_obs = {}
    all_stats = {}
    conditions = [
        ("Baseline",    False, False),
        ("TMS Only",    True,  False),
        ("Chunks Only", False, True),
        ("Full",        True,  True),
    ]

    for cond_name, use_tms, use_chunks in conditions:
        banner(f"Condition: {cond_name}")
        components = []
        if use_tms: components.append("TruthMaintenanceSystem")
        if use_chunks: components.append("ChunkCompiler")
        if not components: components.append("(none)")
        print(f"  KnowledgeSources: {', '.join(components)}")
        print()

        obs = await run_condition(
            condition_name=cond_name,
            bus=bus,
            llm_fn=llm_fn,
            reward_judge=reward_judge,
            queries=QUERIES,
            use_tms=use_tms,
            use_chunks=use_chunks,
        )
        all_obs[cond_name] = obs
        all_stats[cond_name] = analyze_condition(obs)

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    banner("AGGREGATE RESULTS")

    # Summary table
    headers = ["Metric", "Baseline", "TMS Only", "Chunks Only", "Full"]
    metrics = [
        "confidence_mean", "confidence_std", "structural_mean", "narrative_mean",
        "reward_mean", "reward_std", "items_packed_mean", "utilization_mean",
        "shell_count_mean", "time_mean", "response_len_mean",
    ]
    labels = [
        "Confidence (mean)", "Confidence (std)", "Structural", "Narrative",
        "Reward (mean)", "Reward (std)", "Items Packed", "Utilization",
        "Shells (mean)", "Time (mean, s)", "Response Len",
    ]
    rows = []
    for label, metric in zip(labels, metrics):
        row = [label]
        for cond in ["Baseline", "TMS Only", "Chunks Only", "Full"]:
            val = all_stats.get(cond, {}).get(metric, 0)
            if isinstance(val, float):
                row.append(f"{val:.3f}")
            else:
                row.append(str(val))
        rows.append(row)
    table(headers, rows, [20, 12, 12, 12, 12])

    # TMS-specific
    print(f"\n  TMS Statistics:")
    for cond in ["TMS Only", "Full"]:
        if cond in all_stats:
            s = all_stats[cond]
            print(f"    [{cond}] Justified={s['tms_justified_mean']:.1f}/query, "
                  f"Records={s['tms_records_mean']:.1f}/query, "
                  f"Retractions={s['tms_retractions_total']} total, "
                  f"Strength={s['tms_strength_mean']:.3f}")

    # Chunk-specific
    print(f"\n  Chunk Statistics:")
    for cond in ["Chunks Only", "Full"]:
        if cond in all_stats:
            s = all_stats[cond]
            print(f"    [{cond}] Experience={s['chunk_experience']}, "
                  f"Compiled={s['chunk_compilations']}, "
                  f"Matches={s['chunk_matches']}")

    # Per-query comparison
    report_per_query_comparison(all_obs)

    # Deep dives
    if "TMS Only" in all_obs:
        report_tms_deep_dive(all_obs["TMS Only"])
    if "Chunks Only" in all_obs:
        report_chunk_deep_dive(all_obs["Chunks Only"])

    # Overhead
    report_overhead_analysis(all_obs)

    # LLM usage
    sub_banner("LLM Usage")
    print(f"  Total LLM calls:  {llm_call_count[0]}")
    print(f"  Total LLM time:   {llm_total_time[0]:.1f}s")
    print(f"  Avg per call:     {llm_total_time[0] / max(1, llm_call_count[0]):.2f}s")

    # Research findings
    report_research_findings(all_obs, all_stats)

    # ── Save raw data ────────────────────────────────────────────
    sub_banner("Raw Data Export")
    raw_data = {
        "model": MODEL,
        "queries": QUERIES,
        "conditions": {},
    }
    for cond, obs_list in all_obs.items():
        raw_data["conditions"][cond] = {
            "observations": [
                {
                    "query": o.query,
                    "confidence": o.confidence_combined,
                    "structural": o.confidence_structural,
                    "narrative": o.confidence_narrative,
                    "reward": o.reward,
                    "shell_path": o.shell_path,
                    "shell_count": o.shell_count,
                    "items_packed": o.items_packed,
                    "token_count": o.token_count,
                    "utilization": o.utilization,
                    "tms_justified": o.tms_nodes_justified,
                    "tms_records": o.tms_justification_records,
                    "tms_retractions": o.tms_retractions,
                    "tms_strength": o.tms_avg_strength,
                    "chunk_matched": o.chunk_matched,
                    "chunk_experience": o.chunk_experience_total,
                    "chunk_active": o.chunk_active,
                    "total_time": o.total_time,
                    "response_len": o.response_len,
                    "response_preview": o.response_preview,
                    "error": o.error,
                }
                for o in obs_list
            ],
            "aggregate": all_stats[cond],
        }

    out_path = "classical_ai_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"  Saved to {out_path}")

    await bus.close()
    banner("STUDY COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
