"""
Deep Thinking Configuration
===========================

Dataclasses for the Deliberation Engine — three-phase slow reasoning
on AirLLM GPU servers with graceful tier degradation.

Bleed rates modeled on dreaming.py's collective consolidation:
- Plans flow fully to Critic (100%)
- Critic reasoning bleeds conservatively to Synthesizer (30%)
- Consensus propagates fully (100%)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DeepThinkingConfig:
    """
    Configuration for the Deliberation Engine.

    Covers GPU server endpoints, local fallback, deliberation
    bleed rates, tier thresholds, and sleeptime behavior.
    """

    # ── GPU servers (one per 3080, OpenAI-compatible) ──────────
    gpu_servers: list[str] = field(default_factory=lambda: [
        f"http://{os.environ.get('MINING_RIG_IP', '127.0.0.1')}:{port}"
        for port in (8001, 8002, 8003)
    ])
    rig_model: str = "qwen3.5:108b"

    # ── Local fallback (Ollama) ────────────────────────────────
    local_endpoint: str = "http://localhost:11434"
    local_model: str = "qwen3.5:27b"

    # ── Deliberation bleed rates (from dreaming.py) ────────────
    plan_to_critic_bleed: float = 1.0    # plans always flow fully
    plan_reasoning_bleed: float = 0.3    # how much plan reasoning the critic sees
    consensus_bleed: float = 1.0         # resolved consensus propagates fully

    # ── Tier detection ─────────────────────────────────────────
    health_interval_s: float = 30.0
    health_timeout_s: float = 5.0

    # ── Gate thresholds ────────────────────────────────────────
    gate_threshold_full: float = 0.9     # auto-approve (matches heartbeat > 0.9)
    gate_threshold_degraded: float = 0.7 # lower bar for weaker model
    defer_below: float = 0.5            # queue for retroactive 108B review

    # ── Tier 2 behavior ───────────────────────────────────────
    tier2_milestone_multiplier: int = 2  # check 2x more often in degraded

    # ── Sleeptime ──────────────────────────────────────────────
    idle_timeout_s: float = 300.0        # 5 min no activity → idle
    sleeptime_enabled: bool = True

    # ── Deferred queue ─────────────────────────────────────────
    deferred_path: str = "./deep_thinking_deferred.json"
    staleness_boost_per_hour: float = 0.1  # priority bump per hour in queue

    # ── Inference ──────────────────────────────────────────────
    inference_timeout_s: float = 900.0   # 15 min — AirLLM is slow
    max_context_tokens: int = 131072
    max_output_tokens: int = 8192


# ── Worker role GPU assignments ────────────────────────────────

ROLE_GPU_MAP: dict[str, int] = {
    "planner": 0,      # gpu_servers[0]
    "critic": 1,       # gpu_servers[1]
    "synthesizer": 2,  # gpu_servers[2]
}

# ── System prompts per role ────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {
    "planner": (
        "You are a strategic planner. Given a goal and context, decompose it "
        "into a detailed task graph with dependencies, prerequisites, and "
        "expected outputs. Think step by step. Identify risks and unknowns. "
        "Output structured JSON with nodes (tasks) and edges (dependencies)."
    ),
    "critic": (
        "You are an adversarial critic. Given a plan, find logical flaws, "
        "missing steps, hidden assumptions, and risks. Be thorough and "
        "skeptical. Score the plan 0-1 for soundness. Suggest specific "
        "corrections. Do not be charitable — surface every weakness."
    ),
    "synthesizer": (
        "You are a consensus synthesizer. Given a proposal and its critique, "
        "produce a resolved plan that incorporates valid criticisms while "
        "preserving the proposal's strengths. Explicitly list which critiques "
        "were incorporated and which were rejected (with reasons). Surface "
        "any unresolved tensions as dissent — do not average them away."
    ),
}

# ── Gate prompt ────────────────────────────────────────────────

GATE_PROMPT = (
    "You are a milestone reviewer. Given an agent's goal, work so far, and "
    "current milestone, evaluate whether the agent should:\n"
    "- PROCEED: work is sound, continue\n"
    "- REVISE: work has issues, provide specific corrections\n"
    "- ABORT: work is fundamentally misaligned with goal\n\n"
    "Output JSON: {\"verdict\": \"proceed\"|\"revise\"|\"abort\", "
    "\"confidence\": 0.0-1.0, \"reasoning\": \"...\", "
    "\"corrections\": \"...\" | null}"
)
