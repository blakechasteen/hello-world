"""
Training Pipeline — Audit-driven optimization for the Matryoshka shell loop.

Three optimization strategies working from the same audit stream:

  1. ParameterBandit   — Thompson Sampling over shell configs (alpha, budget, thresholds)
  2. PromptRefiner     — LLM-driven iterative improvement of shell prompt templates
  3. SignatureOptimizer — DSPy-style instruction tuning from (input, output, reward) triples

The pipeline is:
  - Offline: run on collected audit data after a session
  - Online: update parameters during a session (warm-start from priors)
  - Self-referential: uses Thompson Sampling to optimize Thompson Sampling

Design borrowed from Promptly's loop composition pattern:
  each optimizer is a composable step, the TrainingPipeline threads
  audit data through them and produces updated ShellConfigs.
"""

import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from .orchestrator import ShellConfig, ShellType, DEFAULT_SHELLS

logger = logging.getLogger(__name__)


# =============================================================================
# Training Examples (from audit stream)
# =============================================================================

@dataclass
class TrainingExample:
    """A single observation from the audit stream."""
    query: str
    shell_type: str               # "prime", "verify", "flag"
    confidence: float             # v2 combined confidence
    structural: float             # v2 structural score
    narrative: float              # v2 narrative score
    reward: float                 # blended reward signal
    context_tokens: int
    context_items: int
    duration_ms: float
    tool_used: str = ""
    response_length: int = 0
    timestamp: str = ""

    @property
    def is_good(self) -> bool:
        return self.reward > 0.6

    @property
    def is_poor(self) -> bool:
        return self.reward < 0.4


@dataclass
class AuditSummary:
    """Aggregated statistics from collected audit entries."""
    total: int
    by_shell: Dict[str, int]
    by_decision: Dict[str, int]
    avg_confidence: float
    avg_reward: float
    avg_duration_ms: float
    good_rate: float              # fraction with reward > 0.6
    poor_rate: float              # fraction with reward < 0.4
    examples: List[TrainingExample]


# =============================================================================
# Audit Collector
# =============================================================================

class AuditCollector:
    """Reads v2_weave_cycle entries from LiteMemoryBus audit log.

    Usage:
        collector = AuditCollector(bus)
        summary = collector.collect()
        print(f"{summary.total} entries, avg_reward={summary.avg_reward:.3f}")
    """

    def __init__(self, bus: Any):
        self.bus = bus

    def collect(self, limit: int = 500) -> AuditSummary:
        """Collect and aggregate training examples from the audit log."""
        raw = self.bus._audit[-limit:] if hasattr(self.bus, '_audit') else []

        examples = []
        for entry in raw:
            if entry.get("type") != "v2_weave_cycle":
                continue

            tool = entry.get("tool_used", "")
            shell = tool.split(":")[-1] if ":" in tool else "prime"

            examples.append(TrainingExample(
                query=entry.get("query", ""),
                shell_type=shell,
                confidence=entry.get("v2_confidence", 0.0),
                structural=entry.get("v2_structural", 0.0),
                narrative=entry.get("v2_narrative", 0.0),
                reward=entry.get("reward", 0.0),
                context_tokens=entry.get("context_tokens", 0),
                context_items=entry.get("context_items", 0),
                duration_ms=entry.get("duration_ms", 0.0),
                tool_used=tool,
                response_length=entry.get("response_length", 0),
                timestamp=entry.get("timestamp", ""),
            ))

        if not examples:
            return AuditSummary(
                total=0, by_shell={}, by_decision={},
                avg_confidence=0, avg_reward=0, avg_duration_ms=0,
                good_rate=0, poor_rate=0, examples=[],
            )

        by_shell = defaultdict(int)
        by_decision = defaultdict(int)
        for ex in examples:
            by_shell[ex.shell_type] += 1
            decision = "sufficient" if ex.confidence >= 0.75 else ("flag" if ex.confidence <= 0.35 else "verify")
            by_decision[decision] += 1

        return AuditSummary(
            total=len(examples),
            by_shell=dict(by_shell),
            by_decision=dict(by_decision),
            avg_confidence=sum(e.confidence for e in examples) / len(examples),
            avg_reward=sum(e.reward for e in examples) / len(examples),
            avg_duration_ms=sum(e.duration_ms for e in examples) / len(examples),
            good_rate=sum(1 for e in examples if e.is_good) / len(examples),
            poor_rate=sum(1 for e in examples if e.is_poor) / len(examples),
            examples=examples,
        )


# =============================================================================
# Parameter Bandit — Thompson Sampling over shell configurations
# =============================================================================

@dataclass
class BanditArm:
    """A candidate parameter value with Beta prior."""
    value: float
    alpha: float = 1.0   # successes + 1
    beta: float = 1.0    # failures + 1
    pulls: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Sample from Beta posterior."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float):
        """Update posterior from observed reward."""
        self.pulls += 1
        if reward > 0.5:
            self.alpha += reward
        else:
            self.beta += (1.0 - reward)


class ParameterBandit:
    """Thompson Sampling over parameter configurations.

    Each parameter (alpha, token_budget, threshold) has a set of candidate
    values. The bandit learns which values produce higher rewards.

    Self-referential: the system uses Thompson Sampling for tool selection,
    and now uses Thompson Sampling to optimize its own Thompson Sampling.

    Usage:
        bandit = ParameterBandit("prime.alpha", [0.10, 0.15, 0.20, 0.25, 0.30])
        idx, value = bandit.select()
        # ... run with this alpha, observe reward ...
        bandit.update(idx, reward=0.85)
    """

    def __init__(self, name: str, candidates: List[float]):
        self.name = name
        self.arms = [BanditArm(value=v) for v in candidates]

    def select(self) -> Tuple[int, float]:
        """Thompson sample: pick the arm with highest posterior sample."""
        samples = [arm.sample() for arm in self.arms]
        idx = max(range(len(samples)), key=lambda i: samples[i])
        return idx, self.arms[idx].value

    def update(self, idx: int, reward: float):
        """Update arm posterior from observed reward."""
        self.arms[idx].update(reward)

    def best(self) -> Tuple[int, float]:
        """Return arm with highest posterior mean (exploitation)."""
        idx = max(range(len(self.arms)), key=lambda i: self.arms[i].mean)
        return idx, self.arms[idx].value

    def train_from_examples(self, examples: List[TrainingExample], value_fn: Callable[[TrainingExample], float]):
        """Batch training: assign each example to the closest arm, update."""
        for ex in examples:
            current_value = value_fn(ex)
            # Find closest arm
            closest_idx = min(range(len(self.arms)),
                              key=lambda i: abs(self.arms[i].value - current_value))
            self.arms[closest_idx].update(ex.reward)

    def report(self) -> List[Dict[str, Any]]:
        """Report arm statistics."""
        return [
            {
                "value": arm.value,
                "mean": arm.mean,
                "pulls": arm.pulls,
                "alpha": arm.alpha,
                "beta": arm.beta,
            }
            for arm in self.arms
        ]


# =============================================================================
# Prompt Refiner — LLM-driven iterative prompt improvement
# =============================================================================

@dataclass
class RefinementResult:
    """Result of a prompt refinement cycle."""
    shell_type: ShellType
    original_template: str
    refined_template: str
    critique: str
    improvement_score: float      # Estimated improvement from LLM self-critique
    examples_used: int
    iterations: int


class PromptRefiner:
    """Iteratively refine shell prompt templates from audit data.

    Carries forward Promptly's refine_iteratively pattern:
      1. Collect low-reward examples for a shell
      2. Ask LLM to critique the current template against these examples
      3. Ask LLM to propose an improved template
      4. Score the improvement via self-critique
      5. Accept if improvement exceeds threshold

    Usage:
        refiner = PromptRefiner(llm_fn=ollama_generate)
        result = await refiner.refine(
            shell_type=ShellType.PRIME,
            current_template=DEFAULT_SHELLS[ShellType.PRIME].prompt_template,
            examples=poor_examples,
        )
        if result.improvement_score > 0.3:
            shell_config.prompt_template = result.refined_template
    """

    def __init__(
        self,
        llm_fn: Callable[[str, int], Coroutine[Any, Any, str]],
        max_iterations: int = 2,
        quality_threshold: float = 0.3,
    ):
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    async def refine(
        self,
        shell_type: ShellType,
        current_template: str,
        examples: List[TrainingExample],
        max_examples: int = 5,
    ) -> RefinementResult:
        """Run the refinement loop for a shell's prompt template."""
        # Select the most informative poor examples
        poor = sorted(
            [e for e in examples if e.is_poor],
            key=lambda e: e.reward,
        )[:max_examples]

        if not poor:
            return RefinementResult(
                shell_type=shell_type,
                original_template=current_template,
                refined_template=current_template,
                critique="No poor examples to learn from.",
                improvement_score=0.0,
                examples_used=0,
                iterations=0,
            )

        working_template = current_template
        best_critique = ""
        best_score = 0.0

        for iteration in range(self.max_iterations):
            # Step 1: Critique current template
            critique = await self._critique(shell_type, working_template, poor)

            # Step 2: Generate improved template
            improved = await self._improve(shell_type, working_template, critique, poor)

            # Step 3: Self-score the improvement
            score = await self._score(shell_type, working_template, improved, poor)

            logger.info(
                "PromptRefiner: %s iter=%d score=%.2f",
                shell_type.value, iteration + 1, score,
            )

            if score > best_score:
                best_score = score
                best_critique = critique
                working_template = improved

            # Convergence: stop if improvement is marginal
            if score < self.quality_threshold:
                break

        return RefinementResult(
            shell_type=shell_type,
            original_template=current_template,
            refined_template=working_template,
            critique=best_critique,
            improvement_score=best_score,
            examples_used=len(poor),
            iterations=iteration + 1,
        )

    async def _critique(
        self,
        shell_type: ShellType,
        template: str,
        examples: List[TrainingExample],
    ) -> str:
        examples_text = "\n".join(
            f"  Query: {e.query[:80]}\n  Reward: {e.reward:.2f}, Confidence: {e.confidence:.2f}"
            for e in examples[:3]
        )
        prompt = (
            f"You are analyzing the prompt template for a {shell_type.value} shell in a retrieval system.\n\n"
            f"Current template:\n```\n{template}\n```\n\n"
            f"This template produced low-quality responses for these queries:\n{examples_text}\n\n"
            f"What specific weaknesses does this template have? "
            f"Focus on structure, clarity, and how it uses the context block. Be concise."
        )
        return await self.llm_fn(prompt, 512)

    async def _improve(
        self,
        shell_type: ShellType,
        template: str,
        critique: str,
        examples: List[TrainingExample],
    ) -> str:
        prompt = (
            f"Improve this {shell_type.value} shell prompt template.\n\n"
            f"Current template:\n```\n{template}\n```\n\n"
            f"Critique:\n{critique}\n\n"
            f"Write an improved version that addresses the critique. "
            f"Keep the {{context}}, {{query}}, and {{previous_response}} placeholders. "
            f"Return ONLY the improved template, no explanation."
        )
        result = await self.llm_fn(prompt, 1024)
        # Validate placeholders are preserved
        if "{context}" not in result or "{query}" not in result:
            logger.warning("PromptRefiner: improved template missing placeholders, keeping original")
            return template
        return result.strip()

    async def _score(
        self,
        shell_type: ShellType,
        original: str,
        improved: str,
        examples: List[TrainingExample],
    ) -> float:
        prompt = (
            f"Rate the improvement of this prompt template on a scale of 0.0 to 1.0.\n\n"
            f"Original:\n```\n{original}\n```\n\n"
            f"Improved:\n```\n{improved}\n```\n\n"
            f"Consider: clarity, specificity, use of context block, instruction quality.\n"
            f"Return ONLY a number between 0.0 and 1.0."
        )
        result = await self.llm_fn(prompt, 32)
        try:
            score = float(result.strip().split()[0])
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return 0.0


# =============================================================================
# Signature Optimizer — DSPy-style instruction tuning
# =============================================================================

@dataclass
class Signature:
    """DSPy-style signature: defines what the shell should do.

    A signature is a declarative specification:
      input_fields → output_fields
    with an instruction that we optimize.
    """
    name: str
    input_fields: List[str]       # e.g., ["context", "query"]
    output_fields: List[str]      # e.g., ["response"]
    instruction: str              # The part we optimize
    demonstrations: List[Dict[str, str]] = field(default_factory=list)


class SignatureOptimizer:
    """DSPy-style: define what you want, optimize how to get it.

    The core DSPy insight: instead of hand-writing prompts, you define
    a signature (input → output) and let the optimizer find the best
    instruction + demonstrations to achieve it.

    The optimizer:
      1. Collects (input, output, reward) triples from audit data
      2. Generates candidate instructions via LLM bootstrapping
      3. Scores candidates against the training data
      4. Selects the best instruction

    Usage:
        sig = Signature(
            name="prime",
            input_fields=["context", "query"],
            output_fields=["response"],
            instruction="Answer the question using the context.",
        )
        optimizer = SignatureOptimizer(llm_fn=ollama)
        optimized = await optimizer.optimize(sig, training_examples)
    """

    def __init__(
        self,
        llm_fn: Callable[[str, int], Coroutine[Any, Any, str]],
        n_candidates: int = 3,
    ):
        self.llm_fn = llm_fn
        self.n_candidates = n_candidates

    async def optimize(
        self,
        signature: Signature,
        examples: List[TrainingExample],
    ) -> Signature:
        """Generate and evaluate instruction candidates."""
        if not examples:
            return signature

        # Partition into good and poor examples
        good = [e for e in examples if e.is_good]
        poor = [e for e in examples if e.is_poor]

        # Bootstrap: generate candidate instructions from good examples
        candidates = await self._bootstrap_instructions(signature, good, poor)
        candidates.append(signature.instruction)  # Include current as baseline

        # Score each candidate
        scores = []
        for candidate in candidates:
            score = await self._score_instruction(signature, candidate, examples)
            scores.append(score)

        # Select best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_instruction = candidates[best_idx]
        best_score = scores[best_idx]

        logger.info(
            "SignatureOptimizer: %s best=%d/%d score=%.2f %s",
            signature.name, best_idx, len(candidates), best_score,
            "(kept original)" if best_idx == len(candidates) - 1 else "(new instruction)",
        )

        # Bootstrap demonstrations from high-reward examples
        demos = self._select_demonstrations(good, max_demos=3)

        return Signature(
            name=signature.name,
            input_fields=signature.input_fields,
            output_fields=signature.output_fields,
            instruction=best_instruction,
            demonstrations=demos,
        )

    async def _bootstrap_instructions(
        self,
        signature: Signature,
        good: List[TrainingExample],
        poor: List[TrainingExample],
    ) -> List[str]:
        """Generate candidate instructions by asking LLM to generalize from examples."""
        good_text = "\n".join(
            f"  Query: {e.query[:60]}  Reward: {e.reward:.2f}"
            for e in good[:3]
        )
        poor_text = "\n".join(
            f"  Query: {e.query[:60]}  Reward: {e.reward:.2f}"
            for e in poor[:3]
        )

        prompt = (
            f"You are optimizing the instruction for a '{signature.name}' module.\n\n"
            f"The module takes: {', '.join(signature.input_fields)}\n"
            f"And produces: {', '.join(signature.output_fields)}\n\n"
            f"Current instruction: \"{signature.instruction}\"\n\n"
            f"High-reward queries:\n{good_text}\n\n"
            f"Low-reward queries:\n{poor_text}\n\n"
            f"Generate {self.n_candidates} alternative instructions that would perform "
            f"better on the low-reward queries while maintaining quality on high-reward ones.\n\n"
            f"Format: one instruction per line, numbered 1-{self.n_candidates}."
        )

        result = await self.llm_fn(prompt, 512)
        candidates = []
        for line in result.strip().split("\n"):
            line = line.strip()
            # Strip numbering
            if line and line[0].isdigit():
                line = line.lstrip("0123456789.)-: ")
            if line and len(line) > 20:
                candidates.append(line)

        return candidates[:self.n_candidates]

    async def _score_instruction(
        self,
        signature: Signature,
        instruction: str,
        examples: List[TrainingExample],
    ) -> float:
        """Score an instruction against training examples.

        For efficiency, we use LLM self-assessment rather than
        running the full pipeline for each candidate.
        """
        sample = examples[:5]
        queries_text = "\n".join(f"  - {e.query[:60]} (reward={e.reward:.2f})" for e in sample)

        prompt = (
            f"Rate this instruction for a retrieval-augmented generation system.\n\n"
            f"Instruction: \"{instruction}\"\n\n"
            f"The system handles queries like:\n{queries_text}\n\n"
            f"Rate 0.0-1.0 on: clarity, specificity, likelihood of producing "
            f"high-quality responses. Return ONLY a number."
        )
        result = await self.llm_fn(prompt, 32)
        try:
            return max(0.0, min(1.0, float(result.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.5

    def _select_demonstrations(
        self,
        good_examples: List[TrainingExample],
        max_demos: int = 3,
    ) -> List[Dict[str, str]]:
        """Select diverse high-reward examples as few-shot demonstrations."""
        if not good_examples:
            return []

        # Sort by reward descending, take top
        sorted_examples = sorted(good_examples, key=lambda e: e.reward, reverse=True)
        demos = []
        for ex in sorted_examples[:max_demos]:
            demos.append({
                "query": ex.query,
                "reward": f"{ex.reward:.2f}",
                "confidence": f"{ex.confidence:.2f}",
            })
        return demos


# =============================================================================
# Optimization Result
# =============================================================================

@dataclass
class OptimizationResult:
    """Output from a full training pipeline run."""
    shell_configs: Dict[ShellType, ShellConfig]
    refinements: List[RefinementResult]
    bandit_reports: Dict[str, List[Dict[str, Any]]]
    signature_updates: Dict[str, Signature]
    summary: AuditSummary
    elapsed_seconds: float

    def changes(self) -> List[str]:
        """Summarize what changed."""
        changes = []
        for name, report in self.bandit_reports.items():
            best = max(report, key=lambda r: r["mean"])
            changes.append(f"{name}: best={best['value']:.3f} (mean={best['mean']:.3f}, pulls={best['pulls']})")
        for r in self.refinements:
            if r.improvement_score > 0.3:
                changes.append(f"{r.shell_type.value} prompt: refined (score={r.improvement_score:.2f})")
        for name, sig in self.signature_updates.items():
            if sig.demonstrations:
                changes.append(f"{name} signature: {len(sig.demonstrations)} demonstrations bootstrapped")
        return changes


# =============================================================================
# Training Pipeline — Orchestrates all three strategies
# =============================================================================

class TrainingPipeline:
    """Full training pipeline: audit → optimize → updated configs.

    Composes three optimization strategies:
      1. ParameterBandit  — tunes numeric parameters (alpha, thresholds)
      2. PromptRefiner    — improves prompt templates (requires LLM)
      3. SignatureOptimizer — tunes instructions DSPy-style (requires LLM)

    The pipeline can run:
      - Offline: process collected audit data
      - Online: update after each weave cycle

    Usage:
        pipeline = TrainingPipeline(llm_fn=ollama)
        result = await pipeline.run(bus)
        for change in result.changes():
            print(change)
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, int], Coroutine[Any, Any, str]]] = None,
        base_configs: Optional[Dict[ShellType, ShellConfig]] = None,
        enable_prompt_refinement: bool = True,
        enable_signature_optimization: bool = True,
    ):
        self.llm_fn = llm_fn
        self.base_configs = base_configs or dict(DEFAULT_SHELLS)
        self.enable_prompt_refinement = enable_prompt_refinement and llm_fn is not None
        self.enable_signature_optimization = enable_signature_optimization and llm_fn is not None

        # Initialize parameter bandits
        self.bandits = self._init_bandits()

    def _init_bandits(self) -> Dict[str, ParameterBandit]:
        """Create Thompson Sampling bandits for key parameters."""
        return {
            "prime.alpha": ParameterBandit("prime.alpha",
                [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
            "verify.alpha": ParameterBandit("verify.alpha",
                [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]),
            "prime.token_budget": ParameterBandit("prime.token_budget",
                [1024, 1536, 2048, 3072, 4096]),
            "verify.token_budget": ParameterBandit("verify.token_budget",
                [512, 768, 1024, 1536, 2048]),
            "confidence.high_threshold": ParameterBandit("confidence.high_threshold",
                [0.65, 0.70, 0.75, 0.80, 0.85]),
            "confidence.low_threshold": ParameterBandit("confidence.low_threshold",
                [0.25, 0.30, 0.35, 0.40, 0.45]),
        }

    async def run(
        self,
        bus: Any,
        limit: int = 500,
    ) -> OptimizationResult:
        """Run the full training pipeline on audit data.

        Args:
            bus: LiteMemoryBus with audit log
            limit: Max audit entries to process

        Returns:
            OptimizationResult with updated configs and reports
        """
        t0 = time.perf_counter()

        # 1. Collect audit data
        collector = AuditCollector(bus)
        summary = collector.collect(limit=limit)

        logger.info(
            "TrainingPipeline: %d examples (good=%.0f%%, poor=%.0f%%, avg_reward=%.2f)",
            summary.total, summary.good_rate * 100, summary.poor_rate * 100, summary.avg_reward,
        )

        # 2. Train parameter bandits
        self._train_bandits(summary.examples)

        # 3. Build updated shell configs from bandit selections
        updated_configs = self._build_configs()

        # 4. Prompt refinement (if LLM available)
        refinements = []
        if self.enable_prompt_refinement:
            for shell_type in [ShellType.PRIME, ShellType.VERIFY, ShellType.FLAG]:
                shell_examples = [e for e in summary.examples if e.shell_type == shell_type.value]
                if shell_examples:
                    refiner = PromptRefiner(self.llm_fn)
                    result = await refiner.refine(
                        shell_type=shell_type,
                        current_template=self.base_configs[shell_type].prompt_template,
                        examples=shell_examples,
                    )
                    refinements.append(result)
                    if result.improvement_score > 0.3:
                        updated_configs[shell_type].prompt_template = result.refined_template

        # 5. Signature optimization (if LLM available)
        signature_updates = {}
        if self.enable_signature_optimization:
            optimizer = SignatureOptimizer(self.llm_fn)

            for shell_type in [ShellType.PRIME, ShellType.VERIFY]:
                sig = Signature(
                    name=shell_type.value,
                    input_fields=["context", "query"],
                    output_fields=["response"],
                    instruction=self._extract_instruction(updated_configs[shell_type]),
                )
                shell_examples = [e for e in summary.examples if e.shell_type == shell_type.value]
                if shell_examples:
                    optimized = await optimizer.optimize(sig, shell_examples)
                    signature_updates[shell_type.value] = optimized

        elapsed = time.perf_counter() - t0

        return OptimizationResult(
            shell_configs=updated_configs,
            refinements=refinements,
            bandit_reports={name: b.report() for name, b in self.bandits.items()},
            signature_updates=signature_updates,
            summary=summary,
            elapsed_seconds=elapsed,
        )

    def update_online(self, example: TrainingExample):
        """Online update: process a single new audit entry.

        Call this after each weave cycle for real-time adaptation.
        """
        # Update relevant bandits
        for name, bandit in self.bandits.items():
            if name.startswith("prime.") and example.shell_type == "prime":
                bandit.train_from_examples([example], self._value_fn_for(name))
            elif name.startswith("verify.") and example.shell_type == "verify":
                bandit.train_from_examples([example], self._value_fn_for(name))
            elif name.startswith("confidence."):
                bandit.train_from_examples([example], self._value_fn_for(name))

    def get_configs(self) -> Dict[ShellType, ShellConfig]:
        """Get current best configs from bandit posteriors."""
        return self._build_configs()

    # =========================================================================
    # Internal
    # =========================================================================

    def _train_bandits(self, examples: List[TrainingExample]):
        """Batch-train all bandits from examples."""
        for name, bandit in self.bandits.items():
            relevant = self._filter_examples(name, examples)
            if relevant:
                bandit.train_from_examples(relevant, self._value_fn_for(name))

    def _filter_examples(self, bandit_name: str, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Filter examples relevant to a specific bandit."""
        prefix = bandit_name.split(".")[0]
        if prefix in ("prime", "verify", "flag"):
            return [e for e in examples if e.shell_type == prefix]
        return examples  # confidence bandits use all examples

    def _value_fn_for(self, bandit_name: str) -> Callable[[TrainingExample], float]:
        """Return a function that extracts the relevant value from an example."""
        mapping = {
            "prime.alpha": lambda e: 0.15,       # Default; we learn from reward
            "verify.alpha": lambda e: 0.25,
            "prime.token_budget": lambda e: float(e.context_tokens),
            "verify.token_budget": lambda e: float(e.context_tokens),
            "confidence.high_threshold": lambda e: e.confidence,
            "confidence.low_threshold": lambda e: e.confidence,
        }
        return mapping.get(bandit_name, lambda e: 0.5)

    def _build_configs(self) -> Dict[ShellType, ShellConfig]:
        """Build shell configs from current bandit posteriors."""
        configs = {}
        for shell_type, base in self.base_configs.items():
            config = ShellConfig(
                shell_type=shell_type,
                token_budget=base.token_budget,
                ppr_alpha=base.ppr_alpha,
                ppr_max_results=base.ppr_max_results,
                max_seeds=base.max_seeds,
                prompt_template=base.prompt_template,
            )

            # Apply bandit selections
            prefix = shell_type.value
            alpha_name = f"{prefix}.alpha"
            budget_name = f"{prefix}.token_budget"

            if alpha_name in self.bandits:
                _, best_alpha = self.bandits[alpha_name].best()
                config.ppr_alpha = best_alpha

            if budget_name in self.bandits:
                _, best_budget = self.bandits[budget_name].best()
                config.token_budget = int(best_budget)

            configs[shell_type] = config

        return configs

    def _extract_instruction(self, config: ShellConfig) -> str:
        """Extract the core instruction from a prompt template."""
        # Take the first line (before {context}) as the instruction
        lines = config.prompt_template.strip().split("\n")
        return lines[0] if lines else ""

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str):
        """Save training state (bandit posteriors) to JSON."""
        state = {}
        for name, bandit in self.bandits.items():
            state[name] = [
                {
                    "value": arm.value,
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "pulls": arm.pulls,
                }
                for arm in bandit.arms
            ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("TrainingPipeline: saved state to %s (%d bandits)", path, len(state))

    def load(self, path: str) -> bool:
        """Load training state from JSON. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            for name, arms_data in state.items():
                if name in self.bandits:
                    bandit = self.bandits[name]
                    for i, arm_data in enumerate(arms_data):
                        if i < len(bandit.arms) and bandit.arms[i].value == arm_data["value"]:
                            bandit.arms[i].alpha = arm_data["alpha"]
                            bandit.arms[i].beta = arm_data["beta"]
                            bandit.arms[i].pulls = arm_data["pulls"]
            total_pulls = sum(a.pulls for b in self.bandits.values() for a in b.arms)
            logger.info("TrainingPipeline: loaded state from %s (%d total pulls)", path, total_pulls)
            return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("TrainingPipeline: failed to load state: %s", e)
            return False


# =============================================================================
# LLM-as-Judge Composite Reward
# =============================================================================

class RewardJudge:
    """Composite reward signal combining structural metrics + LLM evaluation.

    The reward function is the single most important signal in the system —
    everything downstream (bandits, refinement, signatures) learns from it.

    Components:
      1. Structural score (0.3) — confidence from the estimator, no LLM needed
      2. Substance score (0.2)  — response length normalized (diminishing returns)
      3. LLM judge score (0.5)  — relevance, accuracy, completeness rated by LLM

    When no LLM is available, falls back to structural + substance (reweighted).

    Usage:
        judge = RewardJudge(llm_fn=ollama_generate)
        reward = await judge.score(query="...", response="...", confidence=0.72)
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, int], Coroutine[Any, Any, str]]] = None,
        structural_weight: float = 0.3,
        substance_weight: float = 0.2,
        llm_weight: float = 0.5,
    ):
        self.llm_fn = llm_fn
        self.structural_weight = structural_weight
        self.substance_weight = substance_weight
        self.llm_weight = llm_weight

    async def score(
        self,
        query: str,
        response: str,
        confidence: float,
        context_items: int = 0,
    ) -> float:
        """Compute composite reward.

        Returns:
            Float in [0, 1] — the training reward signal.
        """
        # 1. Structural: direct from confidence estimator
        structural = min(1.0, confidence)

        # 2. Substance: log-scaled response length (diminishing returns)
        if len(response) > 0:
            # 200 chars = 0.5, 800 chars = 0.8, 2000+ chars = ~1.0
            substance = min(1.0, math.log(1 + len(response) / 100) / math.log(21))
        else:
            substance = 0.0

        # 3. LLM judge
        if self.llm_fn and len(response) > 50:
            llm_score = await self._llm_judge(query, response)
            total = (
                self.structural_weight * structural +
                self.substance_weight * substance +
                self.llm_weight * llm_score
            )
        else:
            # Fallback: reweight structural + substance
            fallback_total = self.structural_weight + self.substance_weight
            if fallback_total > 0:
                total = (
                    (self.structural_weight / fallback_total) * structural +
                    (self.substance_weight / fallback_total) * substance
                )
            else:
                total = 0.5

        return max(0.0, min(1.0, total))

    async def _llm_judge(self, query: str, response: str) -> float:
        """Ask LLM to rate the response on three axes."""
        # Truncate long responses for the judge prompt
        resp_preview = response[:1500] if len(response) > 1500 else response

        prompt = (
            "Rate this AI response on a scale of 0.0 to 1.0.\n\n"
            f"Question: {query}\n\n"
            f"Response:\n{resp_preview}\n\n"
            "Score based on:\n"
            "- Relevance: Does it answer the actual question?\n"
            "- Accuracy: Are the claims factually correct?\n"
            "- Completeness: Does it cover the key aspects?\n\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )
        try:
            result = await self.llm_fn(prompt, 32)
            # Parse the score — handle common LLM output formats
            text = result.strip()
            # Strip any surrounding text, find the number
            for token in text.split():
                try:
                    score = float(token.rstrip(".,;:"))
                    if 0.0 <= score <= 1.0:
                        return score
                except ValueError:
                    continue
            return 0.5  # Couldn't parse
        except Exception as e:
            logger.warning("RewardJudge: LLM judge failed: %s", e)
            return 0.5


# =============================================================================
# Contextual Bandit Adapter — Upgrades context-free bandits to contextual
# =============================================================================

# Shell config presets: combinations of (alpha, token_budget, max_seeds)
# Each preset is an Action the contextual bandit can select
CONFIG_PRESETS = [
    {"id": "conservative",  "alpha": 0.10, "budget": 2048, "seeds": 10},
    {"id": "balanced",      "alpha": 0.15, "budget": 2048, "seeds": 10},
    {"id": "focused",       "alpha": 0.25, "budget": 1536, "seeds": 8},
    {"id": "aggressive",    "alpha": 0.30, "budget": 1024, "seeds": 5},
    {"id": "wide",          "alpha": 0.10, "budget": 3072, "seeds": 15},
    {"id": "precision",     "alpha": 0.35, "budget": 768,  "seeds": 3},
]


class ContextualBanditAdapter:
    """Wraps NeuralThompsonPolicy for contextual shell config optimization.

    Upgrades from simple Beta Thompson Sampling (ParameterBandit) to
    contextual bandits that choose configs based on query features.

    The key insight: different queries need different configs. A specific
    factual query benefits from focused retrieval (high alpha, low budget),
    while an exploratory query needs wide retrieval (low alpha, high budget).

    Context vector (6-dim, from Blackboard.context_features()):
      1. query_length: normalized word count
      2. seed_count: normalized seed count
      3. ppr_entropy: PPR score distribution entropy
      4. entity_ratio: fraction of seeds from entity matches
      5. wm_overlap: working memory overlap with current query
      6. confidence: latest confidence score

    Actions: shell config presets (combinations of alpha, budget, max_seeds)

    Falls back to ParameterBandit when NeuralThompsonPolicy is unavailable
    (torch not installed or not enough training data).

    Usage:
        adapter = ContextualBanditAdapter()
        configs = adapter.select_config(blackboard)
        # ... run orchestrator with these configs ...
        adapter.update(blackboard, reward=0.85)
    """

    CONTEXT_DIM = 6  # Must match Blackboard.context_features()

    def __init__(
        self,
        policy=None,  # Optional NeuralThompsonPolicy
        fallback_bandits: Optional[Dict[str, "ParameterBandit"]] = None,
        presets: Optional[List[Dict]] = None,
    ):
        self._policy = policy  # NeuralThompsonPolicy or None
        self._fallback = fallback_bandits  # Dict[str, ParameterBandit]
        self._presets = presets or CONFIG_PRESETS
        self._last_action_id: Optional[str] = None
        self._last_context_id: Optional[str] = None
        self._numpy = None

        # Try to import numpy for context vector
        try:
            import numpy as np
            self._numpy = np
        except ImportError:
            pass

    @property
    def has_neural_policy(self) -> bool:
        """Whether the neural contextual bandit is available."""
        return self._policy is not None and self._numpy is not None

    def select_config(self, blackboard) -> Dict[ShellType, ShellConfig]:
        """Select shell configs based on query context.

        Uses NeuralThompsonPolicy if available, else falls back to
        simple ParameterBandit selections.
        """
        if self.has_neural_policy:
            return self._select_contextual(blackboard)
        elif self._fallback:
            return self._select_fallback()
        else:
            return dict(DEFAULT_SHELLS)

    def update(self, blackboard, reward: float):
        """Update policy from observed reward."""
        if self.has_neural_policy and self._last_context_id is not None:
            try:
                from hololoom.bandits.neural_ts.types import Observation
                obs = Observation(
                    context_id=self._last_context_id,
                    action_id=self._last_action_id or "",
                    reward=reward,
                )
                self._policy.update(obs)
            except Exception as e:
                logger.debug("ContextualBandit update failed: %s", e)

        # Also update fallback bandits if available
        if self._fallback:
            for bandit in self._fallback.values():
                # Simple: update the arm closest to what we used
                if self._last_action_id:
                    preset = self._get_preset(self._last_action_id)
                    if preset:
                        for name, b in self._fallback.items():
                            if "alpha" in name:
                                closest = min(range(len(b.arms)),
                                              key=lambda i: abs(b.arms[i].value - preset["alpha"]))
                                b.update(closest, reward)
                            elif "budget" in name:
                                closest = min(range(len(b.arms)),
                                              key=lambda i: abs(b.arms[i].value - preset["budget"]))
                                b.update(closest, reward)

    def _select_contextual(self, blackboard) -> Dict[ShellType, ShellConfig]:
        """Select config using NeuralThompsonPolicy."""
        np = self._numpy
        from hololoom.bandits.neural_ts.types import Context, Action

        # Extract context features from blackboard
        features = blackboard.context_features()
        ctx_vector = np.array([
            features.get("query_length", 0.0),
            features.get("seed_count", 0.0),
            features.get("ppr_entropy", 0.0),
            features.get("entity_ratio", 0.0),
            features.get("wm_overlap", 0.0),
            features.get("confidence", 0.0),
        ], dtype=np.float32)

        ctx = Context(id=blackboard.query_id, x=ctx_vector)

        # Build actions from presets
        actions = []
        for preset in self._presets:
            action_features = np.array([
                preset["alpha"],
                preset["budget"] / 4096.0,  # Normalize
                preset["seeds"] / 15.0,      # Normalize
            ], dtype=np.float32)
            actions.append(Action(id=preset["id"], a=action_features))

        # Select
        chosen = self._policy.select(ctx, actions)
        self._last_action_id = chosen.id
        self._last_context_id = ctx.id

        # Convert to ShellConfigs
        return self._preset_to_configs(chosen.id)

    def _select_fallback(self) -> Dict[ShellType, ShellConfig]:
        """Select config using simple ParameterBandit (no context)."""
        configs = {}
        for shell_type, base in DEFAULT_SHELLS.items():
            config = ShellConfig(
                shell_type=shell_type,
                token_budget=base.token_budget,
                ppr_alpha=base.ppr_alpha,
                ppr_max_results=base.ppr_max_results,
                max_seeds=base.max_seeds,
                prompt_template=base.prompt_template,
            )

            prefix = shell_type.value
            alpha_name = f"{prefix}.alpha"
            budget_name = f"{prefix}.token_budget"

            if alpha_name in self._fallback:
                _, best_alpha = self._fallback[alpha_name].best()
                config.ppr_alpha = best_alpha
            if budget_name in self._fallback:
                _, best_budget = self._fallback[budget_name].best()
                config.token_budget = int(best_budget)

            configs[shell_type] = config

        return configs

    def _preset_to_configs(self, preset_id: str) -> Dict[ShellType, ShellConfig]:
        """Convert a preset ID to shell configs."""
        preset = self._get_preset(preset_id)
        if not preset:
            return dict(DEFAULT_SHELLS)

        configs = {}
        for shell_type, base in DEFAULT_SHELLS.items():
            config = ShellConfig(
                shell_type=shell_type,
                token_budget=preset["budget"] if shell_type == ShellType.PRIME else base.token_budget,
                ppr_alpha=preset["alpha"] if shell_type == ShellType.PRIME else base.ppr_alpha,
                ppr_max_results=base.ppr_max_results,
                max_seeds=preset["seeds"] if shell_type == ShellType.PRIME else base.max_seeds,
                prompt_template=base.prompt_template,
            )
            configs[shell_type] = config

        return configs

    def _get_preset(self, preset_id: str) -> Optional[Dict]:
        """Look up a preset by ID."""
        for p in self._presets:
            if p["id"] == preset_id:
                return p
        return None

    def report(self) -> Dict[str, Any]:
        """Get diagnostic info about the contextual bandit state."""
        info = {
            "has_neural_policy": self.has_neural_policy,
            "has_fallback": self._fallback is not None,
            "last_action": self._last_action_id,
            "presets": [p["id"] for p in self._presets],
        }
        if self.has_neural_policy:
            info["policy_stats"] = self._policy.get_statistics()
        return info
