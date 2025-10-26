#!/usr/bin/env python3
"""
Self-Improving ChatOps Bot

Continuous learning system that:
- A/B tests different response styles automatically
- Auto-promotes winning variants
- Adapts to user preferences over time
- Optimizes prompts based on feedback
- Learns from high-quality interactions

Usage:
    from HoloLoom.chatops.self_improving_bot import SelfImprovingBot

    bot = SelfImprovingBot()

    # Automatic improvement cycle
    await bot.start_improvement_cycle()

    # Manual experiment
    await bot.run_experiment("response_style")

    # Get improvement statistics
    stats = bot.get_improvement_stats()
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import random
import statistics

try:
    from promptly_integration import PromptlyEnhancedBot, UltrapromptConfig
    PROMPTLY_AVAILABLE = True
except ImportError:
    PROMPTLY_AVAILABLE = False


@dataclass
class ExperimentVariant:
    """A/B test variant configuration"""
    variant_id: str
    name: str
    description: str
    ultraprompt_config: Dict[str, Any]
    prompt_template: Optional[str] = None

    # Statistics
    exposures: int = 0
    successes: int = 0  # Quality score >= threshold
    total_quality_score: float = 0.0
    user_satisfaction: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class Experiment:
    """A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    variants: List[ExperimentVariant]

    # Configuration
    traffic_allocation: Dict[str, float]  # variant_id -> percentage
    success_metric: str = "quality_score"
    min_sample_size: int = 30
    confidence_level: float = 0.95

    # State
    status: str = "active"  # active, completed, cancelled
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    winner: Optional[str] = None

    # Results
    statistical_significance: float = 0.0
    effect_size: float = 0.0


@dataclass
class LearningPattern:
    """Learned pattern from interactions"""
    pattern_id: str
    query_type: str
    successful_approach: str
    quality_score: float
    frequency: int
    last_seen: datetime

    # Evidence
    example_queries: List[str] = field(default_factory=list)
    example_responses: List[str] = field(default_factory=list)


class SelfImprovingBot:
    """
    ChatOps bot that continuously improves through:
    - Automatic A/B testing
    - Winner auto-promotion
    - Pattern learning
    - Adaptive optimization
    """

    def __init__(
        self,
        base_bot: Optional[PromptlyEnhancedBot] = None,
        improvement_interval: int = 3600,  # 1 hour
        experiment_duration: int = 86400,  # 24 hours
        min_improvement_threshold: float = 0.05,  # 5% improvement
        storage_path: Optional[Path] = None
    ):
        if not PROMPTLY_AVAILABLE:
            raise ImportError("Promptly integration required")

        self.base_bot = base_bot or PromptlyEnhancedBot()
        self.improvement_interval = improvement_interval
        self.experiment_duration = experiment_duration
        self.min_improvement_threshold = min_improvement_threshold
        self.storage_path = storage_path or Path("./chatops_data/learning")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Active experiments
        self.experiments: Dict[str, Experiment] = {}

        # Learned patterns
        self.patterns: Dict[str, LearningPattern] = {}

        # Current active variants (per query type)
        self.active_variants: Dict[str, str] = {}

        # Improvement statistics
        self.stats = {
            "experiments_run": 0,
            "winners_promoted": 0,
            "avg_improvement": 0.0,
            "patterns_learned": 0,
            "total_optimizations": 0
        }

        # Load state
        self._load_state()

        # Improvement loop task
        self.improvement_task: Optional[asyncio.Task] = None

        logging.info("SelfImprovingBot initialized")

    async def start_improvement_cycle(self):
        """Start continuous improvement loop"""
        if self.improvement_task and not self.improvement_task.done():
            logging.warning("Improvement cycle already running")
            return

        self.improvement_task = asyncio.create_task(self._improvement_loop())
        logging.info("Started improvement cycle")

    async def stop_improvement_cycle(self):
        """Stop improvement loop"""
        if self.improvement_task:
            self.improvement_task.cancel()
            try:
                await self.improvement_task
            except asyncio.CancelledError:
                pass
            logging.info("Stopped improvement cycle")

    async def _improvement_loop(self):
        """Main improvement loop"""
        while True:
            try:
                # Check experiments
                await self._check_experiments()

                # Promote winners
                await self._promote_winners()

                # Identify improvement opportunities
                opportunities = await self._identify_opportunities()

                # Launch new experiments
                for opportunity in opportunities:
                    await self._launch_experiment(opportunity)

                # Save state
                self._save_state()

                # Wait for next cycle
                await asyncio.sleep(self.improvement_interval)

            except Exception as e:
                logging.error(f"Improvement loop error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    async def _check_experiments(self):
        """Check status of active experiments"""
        now = datetime.now()

        for exp_id, experiment in list(self.experiments.items()):
            if experiment.status != "active":
                continue

            # Check if experiment should complete
            duration = (now - experiment.started_at).total_seconds()

            if duration >= self.experiment_duration:
                await self._complete_experiment(exp_id)
            elif self._has_statistical_significance(experiment):
                logging.info(f"Experiment {exp_id} reached statistical significance early")
                await self._complete_experiment(exp_id)

    async def _complete_experiment(self, exp_id: str):
        """Complete an experiment and determine winner"""
        experiment = self.experiments[exp_id]

        # Calculate results
        results = self._analyze_experiment(experiment)

        experiment.status = "completed"
        experiment.completed_at = datetime.now()
        experiment.winner = results["winner"]
        experiment.statistical_significance = results["significance"]
        experiment.effect_size = results["effect_size"]

        self.stats["experiments_run"] += 1

        logging.info(
            f"Experiment {exp_id} completed. "
            f"Winner: {results['winner']} "
            f"(+{results['effect_size']:.1%} improvement)"
        )

    async def _promote_winners(self):
        """Promote winning variants from completed experiments"""
        for exp_id, experiment in self.experiments.items():
            if experiment.status != "completed":
                continue

            if experiment.winner and experiment.winner != "control":
                # Get winning variant
                winner_variant = next(
                    v for v in experiment.variants
                    if v.variant_id == experiment.winner
                )

                # Check minimum improvement threshold
                control_variant = next(
                    v for v in experiment.variants
                    if v.variant_id == "control"
                )

                improvement = self._calculate_improvement(winner_variant, control_variant)

                if improvement >= self.min_improvement_threshold:
                    await self._promote_variant(experiment, winner_variant)
                    self.stats["winners_promoted"] += 1
                    self.stats["avg_improvement"] = (
                        (self.stats["avg_improvement"] * (self.stats["winners_promoted"] - 1) + improvement) /
                        self.stats["winners_promoted"]
                    )

    async def _promote_variant(self, experiment: Experiment, variant: ExperimentVariant):
        """Promote winning variant to production"""

        # Update active variant for this query type
        query_type = experiment.name.split("_")[0]  # e.g., "incident_response_style" -> "incident"
        self.active_variants[query_type] = variant.variant_id

        # Update base bot configuration
        if variant.ultraprompt_config:
            self._apply_ultraprompt_config(variant.ultraprompt_config)

        # Store as learned pattern
        pattern = LearningPattern(
            pattern_id=f"{experiment.experiment_id}_{variant.variant_id}",
            query_type=query_type,
            successful_approach=variant.description,
            quality_score=variant.total_quality_score / max(variant.exposures, 1),
            frequency=variant.exposures,
            last_seen=datetime.now()
        )
        self.patterns[pattern.pattern_id] = pattern
        self.stats["patterns_learned"] += 1

        logging.info(
            f"Promoted variant '{variant.name}' for {query_type} queries. "
            f"Quality: {pattern.quality_score:.2f}"
        )

    async def _identify_opportunities(self) -> List[Dict[str, Any]]:
        """Identify improvement opportunities from usage data"""
        opportunities = []

        # Get quality statistics
        quality_stats = self.base_bot.get_quality_statistics()

        # Low quality rate opportunity
        if quality_stats.get("low_quality_rate", 0) > 0.15:
            opportunities.append({
                "type": "quality_improvement",
                "metric": "low_quality_rate",
                "current_value": quality_stats["low_quality_rate"],
                "priority": "high"
            })

        # High retry rate opportunity
        if quality_stats.get("retry_count", 0) / max(quality_stats.get("total_responses", 1), 1) > 0.20:
            opportunities.append({
                "type": "retry_reduction",
                "metric": "retry_rate",
                "current_value": quality_stats["retry_count"] / quality_stats["total_responses"],
                "priority": "medium"
            })

        # Response time opportunity (if available)
        # TODO: Add response time tracking

        return opportunities

    async def _launch_experiment(self, opportunity: Dict[str, Any]):
        """Launch new A/B test experiment based on opportunity"""

        if opportunity["type"] == "quality_improvement":
            experiment = self._create_quality_experiment()
        elif opportunity["type"] == "retry_reduction":
            experiment = self._create_retry_experiment()
        else:
            return

        self.experiments[experiment.experiment_id] = experiment
        logging.info(f"Launched experiment: {experiment.name}")

    def _create_quality_experiment(self) -> Experiment:
        """Create experiment to improve response quality"""

        exp_id = f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Control variant (current config)
        control = ExperimentVariant(
            variant_id="control",
            name="Current Config",
            description="Baseline configuration",
            ultraprompt_config=self._get_current_config()
        )

        # Variant A: More verification
        variant_a = ExperimentVariant(
            variant_id="more_verification",
            name="Enhanced Verification",
            description="Stronger verification step",
            ultraprompt_config={
                **self._get_current_config(),
                "use_verification": True,
                "verification_depth": "enhanced"
            }
        )

        # Variant B: Better planning
        variant_b = ExperimentVariant(
            variant_id="better_planning",
            name="Detailed Planning",
            description="More detailed planning step",
            ultraprompt_config={
                **self._get_current_config(),
                "use_planning": True,
                "planning_depth": "detailed"
            }
        )

        return Experiment(
            experiment_id=exp_id,
            name="quality_improvement",
            description="Improve overall response quality",
            variants=[control, variant_a, variant_b],
            traffic_allocation={
                "control": 0.34,
                "more_verification": 0.33,
                "better_planning": 0.33
            },
            success_metric="quality_score",
            min_sample_size=30
        )

    def _create_retry_experiment(self) -> Experiment:
        """Create experiment to reduce retry rate"""

        exp_id = f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        control = ExperimentVariant(
            variant_id="control",
            name="Current Config",
            description="Baseline configuration",
            ultraprompt_config=self._get_current_config()
        )

        # Variant A: Lower temperature
        variant_a = ExperimentVariant(
            variant_id="lower_temperature",
            name="Lower Temperature",
            description="More deterministic responses",
            ultraprompt_config={
                **self._get_current_config(),
                "temperature": 0.5
            }
        )

        # Variant B: Prompt chaining
        variant_b = ExperimentVariant(
            variant_id="prompt_chaining",
            name="Prompt Chaining",
            description="Multi-stage execution",
            ultraprompt_config={
                **self._get_current_config(),
                "use_prompt_chaining": True
            }
        )

        return Experiment(
            experiment_id=exp_id,
            name="retry_reduction",
            description="Reduce auto-retry rate",
            variants=[control, variant_a, variant_b],
            traffic_allocation={
                "control": 0.34,
                "lower_temperature": 0.33,
                "prompt_chaining": 0.33
            },
            success_metric="retry_rate",
            min_sample_size=30
        )

    async def process_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        query_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query with active experiments and learning.

        Args:
            query: User query
            context: Additional context
            query_type: Type of query (incident, code_review, etc.)

        Returns:
            Response with experiment tracking
        """

        # Determine query type if not provided
        if not query_type:
            query_type = await self._classify_query(query)

        # Select variant for this query
        variant_id = await self._select_variant(query_type)

        # Apply variant config if in experiment
        if variant_id != "control":
            original_config = self._get_current_config()
            variant = self._get_variant(variant_id)
            if variant and variant.ultraprompt_config:
                self._apply_ultraprompt_config(variant.ultraprompt_config)

        # Process with base bot
        start_time = datetime.now()
        result = await self.base_bot.process_with_ultraprompt(query, context)
        response_time = (datetime.now() - start_time).total_seconds()

        # Restore original config
        if variant_id != "control":
            self._apply_ultraprompt_config(original_config)

        # Record metrics
        await self._record_metrics(
            variant_id=variant_id,
            query_type=query_type,
            quality_score=result.get("quality_score", 0.0),
            response_time=response_time,
            retry_count=result.get("retry_count", 0)
        )

        # Add experiment metadata
        result["experiment"] = {
            "variant_id": variant_id,
            "query_type": query_type
        }

        return result

    async def _select_variant(self, query_type: str) -> str:
        """Select variant based on active experiments"""

        # Find active experiment for this query type
        active_exp = next(
            (exp for exp in self.experiments.values()
             if exp.status == "active" and query_type in exp.name),
            None
        )

        if not active_exp:
            # Use promoted variant if available
            return self.active_variants.get(query_type, "control")

        # Select based on traffic allocation
        rand = random.random()
        cumulative = 0.0

        for variant_id, allocation in active_exp.traffic_allocation.items():
            cumulative += allocation
            if rand < cumulative:
                return variant_id

        return "control"

    async def _record_metrics(
        self,
        variant_id: str,
        query_type: str,
        quality_score: float,
        response_time: float,
        retry_count: int
    ):
        """Record metrics for variant"""

        variant = self._get_variant(variant_id)
        if not variant:
            return

        variant.exposures += 1
        variant.total_quality_score += quality_score
        variant.response_times.append(response_time)
        variant.retry_count += retry_count

        # Success if quality above threshold
        if quality_score >= self.base_bot.judge_config.min_score_threshold:
            variant.successes += 1

    def _get_variant(self, variant_id: str) -> Optional[ExperimentVariant]:
        """Get variant by ID from any experiment"""
        for experiment in self.experiments.values():
            for variant in experiment.variants:
                if variant.variant_id == variant_id:
                    return variant
        return None

    def _analyze_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Analyze experiment results and determine winner"""

        results = {}

        for variant in experiment.variants:
            if variant.exposures == 0:
                continue

            avg_quality = variant.total_quality_score / variant.exposures
            success_rate = variant.successes / variant.exposures
            avg_response_time = statistics.mean(variant.response_times) if variant.response_times else 0

            results[variant.variant_id] = {
                "avg_quality": avg_quality,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "exposures": variant.exposures
            }

        # Determine winner based on success metric
        if experiment.success_metric == "quality_score":
            winner = max(results.items(), key=lambda x: x[1]["avg_quality"])[0]
        elif experiment.success_metric == "retry_rate":
            winner = max(results.items(), key=lambda x: x[1]["success_rate"])[0]
        else:
            winner = "control"

        # Calculate effect size
        control_metric = results.get("control", {}).get("avg_quality", 0)
        winner_metric = results.get(winner, {}).get("avg_quality", 0)
        effect_size = (winner_metric - control_metric) / max(control_metric, 0.01)

        # Calculate statistical significance (simplified)
        significance = self._calculate_significance(experiment, winner)

        return {
            "winner": winner,
            "effect_size": effect_size,
            "significance": significance,
            "results": results
        }

    def _calculate_improvement(
        self,
        winner: ExperimentVariant,
        control: ExperimentVariant
    ) -> float:
        """Calculate improvement percentage"""

        winner_quality = winner.total_quality_score / max(winner.exposures, 1)
        control_quality = control.total_quality_score / max(control.exposures, 1)

        return (winner_quality - control_quality) / max(control_quality, 0.01)

    def _has_statistical_significance(self, experiment: Experiment) -> bool:
        """Check if experiment has reached statistical significance"""

        # Simplified check - need minimum samples
        min_exposures = experiment.min_sample_size

        for variant in experiment.variants:
            if variant.exposures < min_exposures:
                return False

        # Calculate significance
        results = self._analyze_experiment(experiment)
        return results["significance"] >= experiment.confidence_level

    def _calculate_significance(self, experiment: Experiment, winner_id: str) -> float:
        """Calculate statistical significance (simplified)"""

        # This is a simplified version - use proper statistical tests in production
        winner = next(v for v in experiment.variants if v.variant_id == winner_id)
        control = next(v for v in experiment.variants if v.variant_id == "control")

        if winner.exposures < experiment.min_sample_size:
            return 0.0

        # Simplified confidence based on sample size and effect size
        effect = abs(self._calculate_improvement(winner, control))
        sample_factor = min(winner.exposures / experiment.min_sample_size, 1.0)

        return min(effect * sample_factor, 0.99)

    async def _classify_query(self, query: str) -> str:
        """Classify query type"""

        query_lower = query.lower()

        if any(word in query_lower for word in ["incident", "error", "down", "500", "failing"]):
            return "incident"
        elif any(word in query_lower for word in ["review", "pr", "pull request", "code"]):
            return "code_review"
        elif any(word in query_lower for word in ["deploy", "release", "workflow"]):
            return "workflow"
        else:
            return "general"

    def _get_current_config(self) -> Dict[str, Any]:
        """Get current ultraprompt configuration"""
        config = self.base_bot.ultraprompt_config
        return {
            "temperature": getattr(config, "temperature", 0.7),
            "use_verification": config.use_verification,
            "use_planning": config.use_planning,
            "use_prompt_chaining": config.use_prompt_chaining,
            "sections": config.sections
        }

    def _apply_ultraprompt_config(self, config: Dict[str, Any]):
        """Apply ultraprompt configuration"""
        for key, value in config.items():
            if hasattr(self.base_bot.ultraprompt_config, key):
                setattr(self.base_bot.ultraprompt_config, key, value)

    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get improvement statistics"""
        return {
            **self.stats,
            "active_experiments": len([e for e in self.experiments.values() if e.status == "active"]),
            "completed_experiments": len([e for e in self.experiments.values() if e.status == "completed"]),
            "learned_patterns": len(self.patterns),
            "active_variants": dict(self.active_variants)
        }

    def get_experiment_results(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed experiment results"""
        experiment = self.experiments.get(exp_id)
        if not experiment:
            return None

        return {
            "experiment_id": exp_id,
            "name": experiment.name,
            "status": experiment.status,
            "winner": experiment.winner,
            "significance": experiment.statistical_significance,
            "effect_size": experiment.effect_size,
            "variants": [
                {
                    "id": v.variant_id,
                    "name": v.name,
                    "exposures": v.exposures,
                    "avg_quality": v.total_quality_score / max(v.exposures, 1),
                    "success_rate": v.successes / max(v.exposures, 1)
                }
                for v in experiment.variants
            ]
        }

    def _save_state(self):
        """Save experiments and patterns to disk"""
        state = {
            "experiments": {
                exp_id: {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "status": exp.status,
                    "winner": exp.winner,
                    "started_at": exp.started_at.isoformat(),
                    "completed_at": exp.completed_at.isoformat() if exp.completed_at else None
                }
                for exp_id, exp in self.experiments.items()
            },
            "patterns": {
                pat_id: {
                    "pattern_id": pat.pattern_id,
                    "query_type": pat.query_type,
                    "successful_approach": pat.successful_approach,
                    "quality_score": pat.quality_score,
                    "frequency": pat.frequency
                }
                for pat_id, pat in self.patterns.items()
            },
            "active_variants": self.active_variants,
            "stats": self.stats
        }

        state_file = self.storage_path / "improvement_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load experiments and patterns from disk"""
        state_file = self.storage_path / "improvement_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load stats
            self.stats = state.get("stats", self.stats)

            # Load active variants
            self.active_variants = state.get("active_variants", {})

            logging.info(f"Loaded state: {len(state.get('experiments', {}))} experiments, {len(state.get('patterns', {}))} patterns")

        except Exception as e:
            logging.error(f"Failed to load state: {e}")


# Demo
async def demo_self_improving():
    """Demonstrate self-improving bot"""

    print("🤖 Self-Improving ChatOps Bot Demo\n")

    bot = SelfImprovingBot()

    # Start improvement cycle
    await bot.start_improvement_cycle()

    # Simulate some queries
    print("Processing queries with experiments...\n")

    for i in range(5):
        result = await bot.process_query(
            f"Sample incident query {i}",
            query_type="incident"
        )
        print(f"Query {i+1}:")
        print(f"  Variant: {result['experiment']['variant_id']}")
        print(f"  Quality: {result.get('quality_score', 0.0):.2f}")
        print()

    # Get stats
    stats = bot.get_improvement_stats()
    print("Improvement Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    await bot.stop_improvement_cycle()


if __name__ == "__main__":
    asyncio.run(demo_self_improving())
