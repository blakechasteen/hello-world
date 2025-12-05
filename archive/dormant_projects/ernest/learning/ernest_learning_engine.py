"""
Ernest Learning Engine - Phase 2
=================================
Adaptive learning for creative writing refinement.

Ernest learns from every refinement:
- Which writing modes (SPARSE/DIRECT/GRACE/LOST_GEN) work best
- Which refinement passes are most effective
- Patterns in successful creative writing
- Thompson Sampling adaptation for mode selection

This enables Ernest to get better at understanding YOUR writing style
and recommending the right approach for each piece.

Integration with HoloLoom's recursive learning system for self-improvement.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import json

from HoloLoom.recursive.full_learning_loop import (
    ThompsonPriors,
    PolicyWeights,
    LearningMetrics
)
from ernest.refinement.patterns import (
    HemingwayPatterns,
    HemingwayMode,
    ProseMetrics,
    RefinementResult
)


@dataclass
class WritingModePrefs:
    """
    Learned preferences for writing modes based on context.

    Ernest learns which modes work best for different types of writing:
    - Dialogue scenes → SPARSE mode
    - Action sequences → DIRECT mode
    - Descriptive passages → GRACE mode
    - Existential narration → LOST_GEN mode
    """
    mode_successes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mode_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mode_avg_scores: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # Context-specific mode preferences
    context_modes: Dict[str, str] = field(default_factory=dict)

    def update(self, mode: HemingwayMode, score: float, context: str = "general"):
        """
        Update mode preferences based on Hemingway score.

        Args:
            mode: Writing mode used
            score: Hemingway score achieved (0-100)
            context: Writing context (dialogue/action/description/etc)
        """
        mode_str = mode.value

        # Update totals
        self.mode_totals[mode_str] += 1

        # Success if score >= 80
        if score >= 80:
            self.mode_successes[mode_str] += 1

        # Update rolling average score
        prev_avg = self.mode_avg_scores[mode_str]
        total = self.mode_totals[mode_str]
        self.mode_avg_scores[mode_str] = (
            (prev_avg * (total - 1) + score) / total
        )

        # Learn context-specific preferences
        if context != "general":
            # If this mode works well for this context, remember it
            if score >= 85:
                self.context_modes[context] = mode_str

    def get_best_mode(self, context: str = "general") -> HemingwayMode:
        """
        Get best mode for given context.

        Returns mode with highest average score, or context-specific
        learned preference if available.
        """
        # Check for context-specific learned preference
        if context in self.context_modes:
            return HemingwayMode(self.context_modes[context])

        # Fall back to highest average score
        if not self.mode_avg_scores:
            return HemingwayMode.GRACE  # Default

        best_mode = max(
            self.mode_avg_scores.items(),
            key=lambda x: x[1]
        )[0]
        return HemingwayMode(best_mode)

    def get_success_rate(self, mode: HemingwayMode) -> float:
        """Get success rate for a mode"""
        mode_str = mode.value
        total = self.mode_totals[mode_str]
        if total == 0:
            return 0.5  # Unknown
        return self.mode_successes[mode_str] / total

    def get_avg_score(self, mode: HemingwayMode) -> float:
        """Get average Hemingway score for a mode"""
        return self.mode_avg_scores.get(mode.value, 50.0)


@dataclass
class RefinementPassStats:
    """
    Statistics on refinement pass effectiveness.

    Tracks which passes (clarity/simplicity/beauty) provide the most
    improvement for different types of prose issues.
    """
    pass_improvements: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record_improvement(self, pass_name: str, before_score: float, after_score: float):
        """Record score improvement from a refinement pass"""
        improvement = after_score - before_score
        self.pass_improvements[pass_name].append(improvement)

    def get_avg_improvement(self, pass_name: str) -> float:
        """Get average improvement for a refinement pass"""
        improvements = self.pass_improvements[pass_name]
        if not improvements:
            return 0.0
        return sum(improvements) / len(improvements)

    def get_most_effective_pass(self) -> str:
        """Get most effective refinement pass"""
        if not self.pass_improvements:
            return "clarity"  # Default

        avg_improvements = {
            name: sum(imps) / len(imps)
            for name, imps in self.pass_improvements.items()
            if imps
        }

        if not avg_improvements:
            return "clarity"

        return max(avg_improvements.items(), key=lambda x: x[1])[0]


@dataclass
class CreativePatterns:
    """
    Learned patterns from creative writing.

    Ernest learns what makes YOUR writing stronger:
    - Strong verb patterns (what verbs work well in your style)
    - Sentence rhythm preferences
    - Dialogue patterns
    - Description patterns
    """
    strong_verbs_seen: Set[str] = field(default_factory=set)
    effective_sentence_lengths: List[int] = field(default_factory=list)
    dialogue_patterns: List[str] = field(default_factory=list)
    description_patterns: List[str] = field(default_factory=list)

    def learn_from_text(self, text: str, score: float):
        """
        Learn patterns from high-scoring text.

        Only learns from text with Hemingway score >= 85.
        """
        if score < 85:
            return

        # Learn sentence length distribution
        sentences = text.split('.')
        for sentence in sentences:
            words = sentence.split()
            if 5 <= len(words) <= 25:
                self.effective_sentence_lengths.append(len(words))

        # TODO: More sophisticated pattern extraction
        # - Extract strong verbs using spaCy
        # - Identify dialogue patterns
        # - Extract descriptive phrases


@dataclass
class ErnestLearningMetrics:
    """
    Ernest-specific learning metrics.

    Tracks Ernest's improvement over time.
    """
    refinements_performed: int = 0
    avg_hemingway_score_before: float = 0.0
    avg_hemingway_score_after: float = 0.0
    modes_tried: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_improvement: float = 0.0
    learning_rate: float = 0.0  # Improvement per refinement

    def update(self, before_score: float, after_score: float, mode: HemingwayMode):
        """Update metrics with refinement result"""
        self.refinements_performed += 1

        # Update average scores
        n = self.refinements_performed
        self.avg_hemingway_score_before = (
            (self.avg_hemingway_score_before * (n - 1) + before_score) / n
        )
        self.avg_hemingway_score_after = (
            (self.avg_hemingway_score_after * (n - 1) + after_score) / n
        )

        # Track mode usage
        self.modes_tried[mode.value] += 1

        # Update total improvement
        improvement = after_score - before_score
        self.total_improvement += improvement

        # Calculate learning rate
        self.learning_rate = self.total_improvement / n


class ErnestBackgroundLearner:
    """
    Background thread that learns from Ernest's refinements.

    Similar to HoloLoom's BackgroundLearner, but specialized for
    creative writing refinement patterns.
    """

    def __init__(
        self,
        mode_prefs: WritingModePrefs,
        pass_stats: RefinementPassStats,
        creative_patterns: CreativePatterns,
        thompson_priors: ThompsonPriors,
        update_interval: float = 60.0
    ):
        """
        Initialize Ernest background learner.

        Args:
            mode_prefs: Writing mode preferences
            pass_stats: Refinement pass statistics
            creative_patterns: Creative writing patterns
            thompson_priors: Thompson Sampling priors for mode selection
            update_interval: Seconds between learning updates
        """
        self.mode_prefs = mode_prefs
        self.pass_stats = pass_stats
        self.creative_patterns = creative_patterns
        self.thompson_priors = thompson_priors
        self.update_interval = update_interval
        self.logger = logging.getLogger(f"{__name__}.ErnestBackgroundLearner")

        # Recent refinements
        self.recent_refinements: deque = deque(maxlen=50)
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background learning loop"""
        if self.running:
            self.logger.warning("Ernest background learner already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._learning_loop())
        self.logger.info("Ernest background learner started")

    async def stop(self):
        """Stop background learning loop"""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info("Ernest background learner stopped")

    def record_refinement(
        self,
        mode: HemingwayMode,
        before_score: float,
        after_score: float,
        text: str,
        context: str = "general"
    ):
        """Record refinement for learning"""
        self.recent_refinements.append({
            "mode": mode,
            "before_score": before_score,
            "after_score": after_score,
            "text": text,
            "context": context,
            "timestamp": datetime.now()
        })

    async def _learning_loop(self):
        """Main background learning loop"""
        while self.running:
            try:
                await asyncio.sleep(self.update_interval)

                if not self.recent_refinements:
                    continue

                self.logger.info("Running Ernest background learning update...")

                # 1. Update mode preferences
                await self._update_mode_preferences()

                # 2. Update Thompson priors for modes
                await self._update_thompson_priors()

                # 3. Learn creative patterns from high-scoring text
                await self._learn_creative_patterns()

                self.logger.info("Ernest background learning update complete")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in Ernest learning loop: {e}")

    async def _update_mode_preferences(self):
        """Update mode preferences from recent refinements"""
        for refinement in self.recent_refinements:
            mode = refinement["mode"]
            after_score = refinement["after_score"]
            context = refinement["context"]

            self.mode_prefs.update(mode, after_score, context)

        self.logger.info(
            f"Updated mode preferences from {len(self.recent_refinements)} refinements"
        )

    async def _update_thompson_priors(self):
        """Update Thompson Sampling priors for mode selection"""
        mode_scores = defaultdict(list)

        # Collect scores by mode
        for refinement in self.recent_refinements:
            mode = refinement["mode"].value
            after_score = refinement["after_score"]
            mode_scores[mode].append(after_score)

        # Update Thompson priors
        for mode, scores in mode_scores.items():
            avg_score = sum(scores) / len(scores)

            # Normalize to 0-1 (score is 0-100)
            normalized_score = avg_score / 100.0

            if normalized_score >= 0.80:
                # Success (high Hemingway score)
                self.thompson_priors.update_success(mode, normalized_score)
            else:
                # Failure (low Hemingway score)
                self.thompson_priors.update_failure(mode, normalized_score)

        self.logger.info(
            f"Updated Thompson priors for {len(mode_scores)} modes"
        )

    async def _learn_creative_patterns(self):
        """Learn patterns from high-scoring refinements"""
        high_scoring = [
            r for r in self.recent_refinements
            if r["after_score"] >= 85
        ]

        for refinement in high_scoring:
            text = refinement["text"]
            score = refinement["after_score"]
            self.creative_patterns.learn_from_text(text, score)

        self.logger.info(
            f"Learned patterns from {len(high_scoring)} high-scoring refinements"
        )


class ErnestLearningEngine:
    """
    Complete learning system for Ernest.

    Integrates with HoloLoom's recursive learning while specializing
    for creative writing refinement.

    Usage:
    ```python
    async with ErnestLearningEngine() as ernest:
        # Refine text with learning
        result = await ernest.refine_with_learning(
            text="Your creative writing here",
            context="dialogue"  # or "action", "description", etc.
        )

        print(f"Before: {result.before_score:.0f}/100")
        print(f"After: {result.after_score:.0f}/100")
        print(f"Mode used: {result.mode.value}")

        # Get learning statistics
        stats = ernest.get_learning_statistics()
        print(f"Total refinements: {stats['refinements_performed']}")
        print(f"Avg improvement: {stats['avg_improvement']:.1f} points")
    ```
    """

    def __init__(
        self,
        enable_background_learning: bool = True,
        learning_update_interval: float = 60.0,
        persistence_path: Optional[str] = None
    ):
        """
        Initialize Ernest learning engine.

        Args:
            enable_background_learning: Enable background learning thread
            learning_update_interval: Seconds between learning updates
            persistence_path: Path to save/load learning state
        """
        self.logger = logging.getLogger(f"{__name__}.ErnestLearningEngine")

        # Learning components
        self.mode_prefs = WritingModePrefs()
        self.pass_stats = RefinementPassStats()
        self.creative_patterns = CreativePatterns()
        self.thompson_priors = ThompsonPriors()
        self.metrics = ErnestLearningMetrics()

        # Background learner
        self.background_learner: Optional[ErnestBackgroundLearner] = None
        self.enable_background_learning = enable_background_learning
        self.learning_update_interval = learning_update_interval

        # Persistence
        self.persistence_path = persistence_path
        if persistence_path:
            self._load_learning_state()

    async def __aenter__(self):
        """Async context manager entry"""
        # Start background learner
        if self.enable_background_learning:
            self.background_learner = ErnestBackgroundLearner(
                mode_prefs=self.mode_prefs,
                pass_stats=self.pass_stats,
                creative_patterns=self.creative_patterns,
                thompson_priors=self.thompson_priors,
                update_interval=self.learning_update_interval
            )
            await self.background_learner.start()
            self.logger.info("Ernest background learning enabled")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Stop background learner
        if self.background_learner:
            await self.background_learner.stop()

        # Save learning state
        if self.persistence_path:
            self._save_learning_state()

        return False

    async def refine_with_learning(
        self,
        text: str,
        context: str = "general",
        mode: Optional[HemingwayMode] = None
    ) -> Dict[str, Any]:
        """
        Refine text with learning.

        Ernest automatically:
        1. Selects best mode (unless specified)
        2. Performs 3-pass refinement
        3. Learns from the result
        4. Updates mode preferences

        Args:
            text: Text to refine
            context: Writing context (dialogue/action/description/etc)
            mode: Specific mode to use (None = auto-select)

        Returns:
            Dict with refinement results and metrics
        """
        # Auto-select mode if not specified
        if mode is None:
            mode = self.mode_prefs.get_best_mode(context)
            self.logger.info(f"Auto-selected mode: {mode.value} for context: {context}")

        # Create refiner
        ernest = HemingwayPatterns(mode=mode)

        # Get initial score
        before_metrics = ernest.analyze_prose(text)
        before_score = before_metrics.hemingway_score

        # Perform 3-pass refinement
        result1 = ernest.refine_pass1_clarity(text)
        self.pass_stats.record_improvement("clarity", before_score, result1.metrics.hemingway_score)

        result2 = ernest.refine_pass2_simplicity(result1.refined_text)
        self.pass_stats.record_improvement("simplicity", result1.metrics.hemingway_score, result2.metrics.hemingway_score)

        result3 = ernest.refine_pass3_beauty(result2.refined_text)
        self.pass_stats.record_improvement("beauty", result2.metrics.hemingway_score, result3.metrics.hemingway_score)

        # Get final score
        after_score = result3.metrics.hemingway_score
        refined_text = result3.refined_text

        # Update learning metrics
        self.metrics.update(before_score, after_score, mode)

        # Record for background learning
        if self.background_learner:
            self.background_learner.record_refinement(
                mode=mode,
                before_score=before_score,
                after_score=after_score,
                text=refined_text,
                context=context
            )

        # Update mode preferences immediately
        self.mode_prefs.update(mode, after_score, context)

        return {
            "original_text": text,
            "refined_text": refined_text,
            "before_score": before_score,
            "after_score": after_score,
            "improvement": after_score - before_score,
            "mode": mode,
            "context": context,
            "before_metrics": before_metrics,
            "after_metrics": result3.metrics,
            "pass1_improvement": result1.metrics.hemingway_score - before_score,
            "pass2_improvement": result2.metrics.hemingway_score - result1.metrics.hemingway_score,
            "pass3_improvement": result3.metrics.hemingway_score - result2.metrics.hemingway_score,
            "changes": result1.changes + result2.changes + result3.changes
        }

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics.

        Returns:
            Dict with all learning metrics
        """
        return {
            # Overall metrics
            "refinements_performed": self.metrics.refinements_performed,
            "avg_score_before": self.metrics.avg_hemingway_score_before,
            "avg_score_after": self.metrics.avg_hemingway_score_after,
            "avg_improvement": self.metrics.avg_hemingway_score_after - self.metrics.avg_hemingway_score_before,
            "total_improvement": self.metrics.total_improvement,
            "learning_rate": self.metrics.learning_rate,

            # Mode statistics
            "modes_tried": dict(self.metrics.modes_tried),
            "mode_success_rates": {
                mode.value: self.mode_prefs.get_success_rate(mode)
                for mode in HemingwayMode
            },
            "mode_avg_scores": {
                mode.value: self.mode_prefs.get_avg_score(mode)
                for mode in HemingwayMode
            },
            "best_mode": self.mode_prefs.get_best_mode().value,

            # Pass statistics
            "pass_improvements": {
                pass_name: self.pass_stats.get_avg_improvement(pass_name)
                for pass_name in ["clarity", "simplicity", "beauty"]
            },
            "most_effective_pass": self.pass_stats.get_most_effective_pass(),

            # Thompson Sampling
            "thompson_expected_rewards": {
                mode.value: self.thompson_priors.get_expected_reward(mode.value)
                for mode in HemingwayMode
            },

            # Creative patterns
            "patterns_learned": {
                "strong_verbs": len(self.creative_patterns.strong_verbs_seen),
                "sentence_length_samples": len(self.creative_patterns.effective_sentence_lengths),
                "avg_effective_sentence_length": (
                    sum(self.creative_patterns.effective_sentence_lengths) /
                    len(self.creative_patterns.effective_sentence_lengths)
                    if self.creative_patterns.effective_sentence_lengths else 0
                )
            }
        }

    def _save_learning_state(self):
        """Save learning state to disk"""
        if not self.persistence_path:
            return

        state = {
            "mode_prefs": {
                "mode_successes": dict(self.mode_prefs.mode_successes),
                "mode_totals": dict(self.mode_prefs.mode_totals),
                "mode_avg_scores": dict(self.mode_prefs.mode_avg_scores),
                "context_modes": dict(self.mode_prefs.context_modes)
            },
            "metrics": {
                "refinements_performed": self.metrics.refinements_performed,
                "avg_hemingway_score_before": self.metrics.avg_hemingway_score_before,
                "avg_hemingway_score_after": self.metrics.avg_hemingway_score_after,
                "modes_tried": dict(self.metrics.modes_tried),
                "total_improvement": self.metrics.total_improvement,
                "learning_rate": self.metrics.learning_rate
            },
            "thompson_priors": {
                mode: dict(priors)
                for mode, priors in self.thompson_priors.tool_priors.items()
            }
        }

        path = Path(self.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved Ernest learning state to {path}")

    def _load_learning_state(self):
        """Load learning state from disk"""
        if not self.persistence_path:
            return

        path = Path(self.persistence_path)
        if not path.exists():
            self.logger.info("No existing learning state found")
            return

        try:
            with open(path, 'r') as f:
                state = json.load(f)

            # Restore mode preferences
            self.mode_prefs.mode_successes = defaultdict(int, state["mode_prefs"]["mode_successes"])
            self.mode_prefs.mode_totals = defaultdict(int, state["mode_prefs"]["mode_totals"])
            self.mode_prefs.mode_avg_scores = defaultdict(float, state["mode_prefs"]["mode_avg_scores"])
            self.mode_prefs.context_modes = dict(state["mode_prefs"]["context_modes"])

            # Restore metrics
            self.metrics.refinements_performed = state["metrics"]["refinements_performed"]
            self.metrics.avg_hemingway_score_before = state["metrics"]["avg_hemingway_score_before"]
            self.metrics.avg_hemingway_score_after = state["metrics"]["avg_hemingway_score_after"]
            self.metrics.modes_tried = defaultdict(int, state["metrics"]["modes_tried"])
            self.metrics.total_improvement = state["metrics"]["total_improvement"]
            self.metrics.learning_rate = state["metrics"]["learning_rate"]

            # Restore Thompson priors
            self.thompson_priors.tool_priors = defaultdict(
                lambda: {"alpha": 1.0, "beta": 1.0},
                {mode: dict(priors) for mode, priors in state["thompson_priors"].items()}
            )

            self.logger.info(f"Loaded Ernest learning state from {path}")
            self.logger.info(f"  {self.metrics.refinements_performed} refinements performed")
            self.logger.info(f"  Avg improvement: {self.metrics.avg_hemingway_score_after - self.metrics.avg_hemingway_score_before:.1f} points")

        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")


# Quick helper functions
async def refine_with_ernest(text: str, context: str = "general") -> str:
    """
    Quick refinement with Ernest learning.

    One-liner for getting refined text with automatic learning.
    """
    async with ErnestLearningEngine() as ernest:
        result = await ernest.refine_with_learning(text, context)
        return result["refined_text"]


async def analyze_with_ernest(text: str) -> Dict[str, Any]:
    """
    Quick analysis with Ernest (no refinement).

    Returns Hemingway metrics and recommendations.
    """
    ernest_patterns = HemingwayPatterns()
    metrics = ernest_patterns.analyze_prose(text)

    # Auto-select best mode based on metrics
    if metrics.iceberg_ratio < 50:
        recommended_mode = HemingwayMode.SPARSE  # Need more showing
    elif metrics.avg_words_per_sentence > 25:
        recommended_mode = HemingwayMode.DIRECT  # Need shorter sentences
    elif metrics.active_voice_pct < 70:
        recommended_mode = HemingwayMode.GRACE  # Need more active voice
    else:
        recommended_mode = HemingwayMode.GRACE  # Default

    return {
        "hemingway_score": metrics.hemingway_score,
        "metrics": metrics,
        "recommended_mode": recommended_mode.value,
        "needs_improvement": [
            "Iceberg ratio (showing vs. telling)" if metrics.iceberg_ratio < 70 else None,
            "Sentence length" if metrics.avg_words_per_sentence > 20 else None,
            "Active voice" if metrics.active_voice_pct < 85 else None,
            "Verb strength" if metrics.strong_verb_pct < 70 else None,
            "Readability" if metrics.flesch_kincaid_grade > 8 else None
        ]
    }
