"""A/B Testing Framework for Strategy Changes.

CARTS Phase 4: Statistical Validation System
Status: Foundation Implementation (December 2025)
Location: HoloLoom/redteam/swarm/ab_testing.py

FIRST PRINCIPLE (5.5): "A/B testing for high-stakes decisions"

Before deploying new strategy systemwide:
1. Create A/B test (control: old, treatment: new)
2. Run with statistical rigor
3. Require significance before adoption
4. Measure effect size, not just p-value

This module implements a rigorous A/B testing framework for validating
strategy changes before deployment, ensuring that improvements are
statistically significant and practically meaningful.
"""

import asyncio
import hashlib
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Enums
# =============================================================================


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""

    DRAFT = "draft"  # Not yet started
    RUNNING = "running"  # Actively collecting data
    PAUSED = "paused"  # Temporarily stopped
    COMPLETED = "completed"  # Reached sample size
    STOPPED_EARLY = "stopped_early"  # Early stopping triggered
    DEPLOYED = "deployed"  # Winner deployed to production


class TrafficAllocation(Enum):
    """How traffic is split between variants."""

    EQUAL = "equal"  # 50/50 split
    WEIGHTED = "weighted"  # Custom weights
    MULTI_ARM = "multi_arm"  # Multi-arm bandit allocation


class SignificanceLevel(Enum):
    """Statistical significance thresholds."""

    RELAXED = 0.10  # 90% confidence (exploratory)
    STANDARD = 0.05  # 95% confidence (default)
    STRICT = 0.01  # 99% confidence (high-stakes)
    VERY_STRICT = 0.001  # 99.9% confidence (critical)


class EffectSizeCategory(Enum):
    """Cohen's d effect size categories."""

    NEGLIGIBLE = "negligible"  # |d| < 0.2
    SMALL = "small"  # 0.2 <= |d| < 0.5
    MEDIUM = "medium"  # 0.5 <= |d| < 0.8
    LARGE = "large"  # |d| >= 0.8


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Variant:
    """A variant in an A/B test (control or treatment)."""

    name: str
    strategy_id: str
    description: str = ""
    weight: float = 0.5  # Traffic allocation weight (0.0-1.0)

    # Metrics
    sample_count: int = 0
    success_count: int = 0
    total_reward: float = 0.0

    # Per-sample data for statistical analysis
    samples: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.sample_count == 0:
            return 0.0
        return self.success_count / self.sample_count

    @property
    def mean_reward(self) -> float:
        """Calculate mean reward."""
        if self.sample_count == 0:
            return 0.0
        return self.total_reward / self.sample_count

    @property
    def variance(self) -> float:
        """Calculate sample variance."""
        if len(self.samples) < 2:
            return 0.0
        mean = self.mean_reward
        return sum((x - mean) ** 2 for x in self.samples) / (len(self.samples) - 1)

    @property
    def std_dev(self) -> float:
        """Calculate sample standard deviation."""
        return math.sqrt(self.variance)

    def record_sample(self, reward: float, success: bool = True) -> None:
        """Record a sample for this variant."""
        self.sample_count += 1
        self.total_reward += reward
        self.samples.append(reward)
        if success:
            self.success_count += 1


@dataclass
class StatisticalResult:
    """Results of statistical analysis."""

    # Effect size
    cohens_d: float
    effect_category: EffectSizeCategory

    # Significance testing
    t_statistic: float
    p_value: float
    is_significant: bool

    # Confidence intervals
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Practical significance
    relative_improvement: float  # Percentage improvement
    absolute_improvement: float  # Absolute difference

    # Sample information
    control_n: int
    treatment_n: int
    total_n: int

    # Recommendation
    recommendation: str
    can_deploy: bool


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str
    description: str = ""

    # Variants
    control_strategy: str = ""
    treatment_strategy: str = ""

    # Traffic allocation
    allocation: TrafficAllocation = TrafficAllocation.EQUAL
    control_weight: float = 0.5
    treatment_weight: float = 0.5

    # Statistical parameters
    significance_level: SignificanceLevel = SignificanceLevel.STANDARD
    min_effect_size: float = 0.1  # Minimum practical effect (10% improvement)
    min_samples_per_variant: int = 30  # Statistical minimum
    max_samples: int = 10000  # Upper bound

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_confidence: float = 0.99  # Very high confidence for early stop

    # Deployment gate
    require_effect_size: bool = True
    min_cohens_d: float = 0.2  # At least small effect required

    # Metadata
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class Experiment:
    """An A/B experiment instance."""

    id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Variants
    control: Variant = field(default_factory=lambda: Variant("control", ""))
    treatment: Variant = field(default_factory=lambda: Variant("treatment", ""))

    # Timestamps
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Results
    latest_result: Optional[StatisticalResult] = None
    analysis_history: List[StatisticalResult] = field(default_factory=list)

    def __post_init__(self):
        """Initialize variants from config."""
        if self.config.control_strategy:
            self.control = Variant(
                name="control",
                strategy_id=self.config.control_strategy,
                weight=self.config.control_weight,
            )
        if self.config.treatment_strategy:
            self.treatment = Variant(
                name="treatment",
                strategy_id=self.config.treatment_strategy,
                weight=self.config.treatment_weight,
            )

    @property
    def total_samples(self) -> int:
        """Total samples across all variants."""
        return self.control.sample_count + self.treatment.sample_count

    @property
    def has_minimum_samples(self) -> bool:
        """Check if minimum sample size reached."""
        min_n = self.config.min_samples_per_variant
        return (
            self.control.sample_count >= min_n
            and self.treatment.sample_count >= min_n
        )

    @property
    def at_max_samples(self) -> bool:
        """Check if max sample size reached."""
        return self.total_samples >= self.config.max_samples


# =============================================================================
# Statistical Analysis
# =============================================================================


class StatisticalAnalyzer:
    """Performs rigorous statistical analysis of A/B test results.

    Implements:
    - Two-sample t-test for significance
    - Cohen's d for effect size
    - Confidence intervals
    - Effect size categorization
    - Deployment recommendations
    """

    def __init__(self):
        """Initialize analyzer."""
        # Pre-computed critical t-values for common significance levels
        # (approximations for df > 30)
        self._t_critical = {
            0.10: 1.645,  # 90% confidence
            0.05: 1.96,  # 95% confidence
            0.01: 2.576,  # 99% confidence
            0.001: 3.291,  # 99.9% confidence
        }

    def analyze(
        self,
        control: Variant,
        treatment: Variant,
        significance_level: SignificanceLevel = SignificanceLevel.STANDARD,
        min_cohens_d: float = 0.2,
    ) -> StatisticalResult:
        """Perform complete statistical analysis.

        Args:
            control: Control variant with samples
            treatment: Treatment variant with samples
            significance_level: Significance threshold
            min_cohens_d: Minimum effect size for deployment

        Returns:
            Complete statistical results with recommendation
        """
        alpha = significance_level.value

        # Calculate means and standard deviations
        mu_c = control.mean_reward
        mu_t = treatment.mean_reward
        sd_c = control.std_dev
        sd_t = treatment.std_dev
        n_c = control.sample_count
        n_t = treatment.sample_count

        # Handle edge cases
        if n_c < 2 or n_t < 2:
            return self._insufficient_data_result(n_c, n_t, alpha)

        # Pooled standard deviation (for Cohen's d)
        pooled_var = ((n_c - 1) * sd_c**2 + (n_t - 1) * sd_t**2) / (n_c + n_t - 2)
        pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 0.001

        # Cohen's d effect size
        cohens_d = (mu_t - mu_c) / pooled_sd if pooled_sd > 0 else 0.0
        effect_category = self._categorize_effect(cohens_d)

        # Welch's t-test (doesn't assume equal variances)
        se = math.sqrt(sd_c**2 / n_c + sd_t**2 / n_t) if (sd_c > 0 or sd_t > 0) else 0.001
        t_stat = (mu_t - mu_c) / se

        # p-value approximation (two-tailed)
        # Using normal approximation for large samples
        p_value = self._compute_p_value(abs(t_stat))
        is_significant = p_value < alpha

        # Confidence interval for difference
        t_crit = self._t_critical.get(alpha, 1.96)
        margin = t_crit * se
        ci_lower = (mu_t - mu_c) - margin
        ci_upper = (mu_t - mu_c) + margin

        # Practical significance
        relative_improvement = (
            (mu_t - mu_c) / mu_c * 100 if mu_c != 0 else 0.0
        )
        absolute_improvement = mu_t - mu_c

        # Deployment recommendation
        recommendation, can_deploy = self._make_recommendation(
            is_significant=is_significant,
            p_value=p_value,
            cohens_d=cohens_d,
            min_cohens_d=min_cohens_d,
            relative_improvement=relative_improvement,
            n_c=n_c,
            n_t=n_t,
        )

        return StatisticalResult(
            cohens_d=cohens_d,
            effect_category=effect_category,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=1.0 - alpha,
            relative_improvement=relative_improvement,
            absolute_improvement=absolute_improvement,
            control_n=n_c,
            treatment_n=n_t,
            total_n=n_c + n_t,
            recommendation=recommendation,
            can_deploy=can_deploy,
        )

    def _categorize_effect(self, d: float) -> EffectSizeCategory:
        """Categorize Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return EffectSizeCategory.NEGLIGIBLE
        elif abs_d < 0.5:
            return EffectSizeCategory.SMALL
        elif abs_d < 0.8:
            return EffectSizeCategory.MEDIUM
        else:
            return EffectSizeCategory.LARGE

    def _compute_p_value(self, z: float) -> float:
        """Compute two-tailed p-value from z-score.

        Uses error function approximation for normal CDF.
        """
        # Approximation of cumulative normal distribution
        # Based on Abramowitz and Stegun formula 7.1.26
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if z >= 0 else -1
        z = abs(z) / math.sqrt(2)

        t = 1.0 / (1.0 + p * z)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

        cdf = 0.5 * (1.0 + sign * y)

        # Two-tailed p-value
        return 2 * (1 - cdf)

    def _insufficient_data_result(
        self, n_c: int, n_t: int, alpha: float
    ) -> StatisticalResult:
        """Return result when insufficient data."""
        return StatisticalResult(
            cohens_d=0.0,
            effect_category=EffectSizeCategory.NEGLIGIBLE,
            t_statistic=0.0,
            p_value=1.0,
            is_significant=False,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=1.0 - alpha,
            relative_improvement=0.0,
            absolute_improvement=0.0,
            control_n=n_c,
            treatment_n=n_t,
            total_n=n_c + n_t,
            recommendation="Insufficient data - need at least 2 samples per variant",
            can_deploy=False,
        )

    def _make_recommendation(
        self,
        is_significant: bool,
        p_value: float,
        cohens_d: float,
        min_cohens_d: float,
        relative_improvement: float,
        n_c: int,
        n_t: int,
    ) -> Tuple[str, bool]:
        """Generate deployment recommendation.

        Returns:
            Tuple of (recommendation_text, can_deploy)
        """
        min_samples = 30

        if n_c < min_samples or n_t < min_samples:
            return (
                f"Need more data: {min_samples} samples per variant required "
                f"(control: {n_c}, treatment: {n_t})",
                False,
            )

        if not is_significant:
            return (
                f"No significant difference detected (p={p_value:.4f}). "
                f"Either continue collecting data or keep control.",
                False,
            )

        if abs(cohens_d) < min_cohens_d:
            return (
                f"Statistically significant but effect too small "
                f"(Cohen's d={cohens_d:.3f}, minimum={min_cohens_d}). "
                f"Consider if practical benefit justifies change.",
                False,
            )

        if cohens_d > 0:
            return (
                f"✓ DEPLOY TREATMENT: {relative_improvement:.1f}% improvement, "
                f"Cohen's d={cohens_d:.3f}, p={p_value:.4f}",
                True,
            )
        else:
            return (
                f"✓ KEEP CONTROL: Treatment {abs(relative_improvement):.1f}% worse, "
                f"Cohen's d={cohens_d:.3f}",
                False,
            )


# =============================================================================
# Traffic Splitter
# =============================================================================


class TrafficSplitter:
    """Assigns incoming requests to experiment variants.

    Uses deterministic hashing for consistent assignment
    (same ID always goes to same variant).
    """

    def __init__(self, seed: int = 42):
        """Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def assign(
        self,
        request_id: str,
        experiment: Experiment,
    ) -> Variant:
        """Assign a request to a variant.

        Uses deterministic hashing so same request ID always
        gets the same variant (important for consistency).

        Args:
            request_id: Unique request identifier
            experiment: Experiment to assign to

        Returns:
            Assigned variant (control or treatment)
        """
        # Hash request ID + experiment ID for deterministic assignment
        hash_input = f"{experiment.id}:{request_id}:{self._seed}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

        # Normalize to [0, 1)
        normalized = (hash_value % 10000) / 10000

        # Assign based on weights
        if normalized < experiment.control.weight:
            return experiment.control
        else:
            return experiment.treatment

    def assign_random(self, experiment: Experiment) -> Variant:
        """Assign randomly (for cases without request ID).

        Args:
            experiment: Experiment to assign to

        Returns:
            Randomly assigned variant
        """
        if self._rng.random() < experiment.control.weight:
            return experiment.control
        else:
            return experiment.treatment


# =============================================================================
# Early Stopping
# =============================================================================


class EarlyStoppingChecker:
    """Checks if experiment can be stopped early.

    Implements:
    - Overwhelming evidence (very high confidence)
    - Futility stopping (treatment clearly worse)
    - Maximum sample size reached
    """

    def __init__(
        self,
        analyzer: StatisticalAnalyzer,
        confidence_threshold: float = 0.99,
    ):
        """Initialize checker.

        Args:
            analyzer: Statistical analyzer
            confidence_threshold: Confidence level for early stopping
        """
        self._analyzer = analyzer
        self._confidence_threshold = confidence_threshold

    def should_stop(self, experiment: Experiment) -> Tuple[bool, str]:
        """Check if experiment should stop early.

        Args:
            experiment: Experiment to check

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check max samples
        if experiment.at_max_samples:
            return True, "Maximum sample size reached"

        # Need minimum samples first
        if not experiment.has_minimum_samples:
            return False, "Minimum samples not yet reached"

        # Perform analysis
        result = self._analyzer.analyze(
            experiment.control,
            experiment.treatment,
            significance_level=SignificanceLevel.VERY_STRICT,  # Use strict for early stop
        )

        # Overwhelming evidence for treatment
        if (
            result.is_significant
            and result.cohens_d > 0.5  # Medium+ effect
            and result.p_value < (1 - self._confidence_threshold)
        ):
            return True, f"Overwhelming evidence for treatment (d={result.cohens_d:.3f})"

        # Futility: Treatment clearly worse
        if (
            result.is_significant
            and result.cohens_d < -0.3  # Treatment notably worse
            and result.p_value < 0.01
        ):
            return True, f"Futility: Treatment is worse (d={result.cohens_d:.3f})"

        return False, "Continue collecting data"


# =============================================================================
# A/B Test Manager
# =============================================================================


class ABTestManager:
    """Manages A/B experiments lifecycle.

    Provides:
    - Experiment creation and configuration
    - Traffic assignment
    - Result recording
    - Statistical analysis
    - Early stopping checks
    - Deployment decisions
    """

    def __init__(self, seed: int = 42):
        """Initialize manager.

        Args:
            seed: Random seed for reproducibility
        """
        self._experiments: Dict[str, Experiment] = {}
        self._analyzer = StatisticalAnalyzer()
        self._splitter = TrafficSplitter(seed=seed)
        self._early_stopper = EarlyStoppingChecker(self._analyzer)

    # -------------------------------------------------------------------------
    # Experiment Management
    # -------------------------------------------------------------------------

    def create_experiment(
        self,
        config: ExperimentConfig,
        experiment_id: Optional[str] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            config: Experiment configuration
            experiment_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Created experiment
        """
        exp_id = experiment_id or str(uuid.uuid4())[:8]

        experiment = Experiment(
            id=exp_id,
            config=config,
            status=ExperimentStatus.DRAFT,
        )

        self._experiments[exp_id] = experiment
        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment (begin collecting data).

        Args:
            experiment_id: Experiment to start

        Returns:
            Started experiment
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment {experiment_id} is not in DRAFT status")

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()

        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment.

        Args:
            experiment_id: Experiment to pause

        Returns:
            Paused experiment
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not RUNNING")

        experiment.status = ExperimentStatus.PAUSED
        return experiment

    def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a paused experiment.

        Args:
            experiment_id: Experiment to resume

        Returns:
            Resumed experiment
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Experiment {experiment_id} is not PAUSED")

        experiment.status = ExperimentStatus.RUNNING
        return experiment

    def stop_experiment(
        self,
        experiment_id: str,
        reason: str = "Manual stop",
    ) -> Experiment:
        """Stop an experiment early.

        Args:
            experiment_id: Experiment to stop
            reason: Reason for stopping

        Returns:
            Stopped experiment
        """
        experiment = self._get_experiment(experiment_id)
        experiment.status = ExperimentStatus.STOPPED_EARLY
        experiment.completed_at = time.time()

        # Final analysis
        self._analyze_experiment(experiment)

        return experiment

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Mark experiment as completed.

        Args:
            experiment_id: Experiment to complete

        Returns:
            Completed experiment
        """
        experiment = self._get_experiment(experiment_id)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = time.time()

        # Final analysis
        self._analyze_experiment(experiment)

        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment
        """
        return self._get_experiment(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
    ) -> List[Experiment]:
        """List experiments.

        Args:
            status: Filter by status (None for all)

        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())

        if status is not None:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    # -------------------------------------------------------------------------
    # Traffic Assignment & Recording
    # -------------------------------------------------------------------------

    def assign_variant(
        self,
        experiment_id: str,
        request_id: Optional[str] = None,
    ) -> Variant:
        """Assign a request to a variant.

        Args:
            experiment_id: Experiment ID
            request_id: Optional request ID (for consistent assignment)

        Returns:
            Assigned variant
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not RUNNING")

        if request_id:
            return self._splitter.assign(request_id, experiment)
        else:
            return self._splitter.assign_random(experiment)

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        reward: float,
        success: bool = True,
    ) -> None:
        """Record a result for a variant.

        Args:
            experiment_id: Experiment ID
            variant_name: Variant name ("control" or "treatment")
            reward: Reward value
            success: Whether the trial was successful
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment {experiment_id} is not RUNNING")

        if variant_name == "control":
            experiment.control.record_sample(reward, success)
        elif variant_name == "treatment":
            experiment.treatment.record_sample(reward, success)
        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        # Check early stopping
        if experiment.config.enable_early_stopping:
            should_stop, reason = self._early_stopper.should_stop(experiment)
            if should_stop:
                experiment.status = ExperimentStatus.STOPPED_EARLY
                experiment.completed_at = time.time()
                self._analyze_experiment(experiment)

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def analyze(self, experiment_id: str) -> StatisticalResult:
        """Analyze experiment results.

        Args:
            experiment_id: Experiment ID

        Returns:
            Statistical analysis results
        """
        experiment = self._get_experiment(experiment_id)
        return self._analyze_experiment(experiment)

    def _analyze_experiment(self, experiment: Experiment) -> StatisticalResult:
        """Perform statistical analysis on experiment.

        Args:
            experiment: Experiment to analyze

        Returns:
            Statistical results
        """
        result = self._analyzer.analyze(
            control=experiment.control,
            treatment=experiment.treatment,
            significance_level=experiment.config.significance_level,
            min_cohens_d=experiment.config.min_cohens_d,
        )

        experiment.latest_result = result
        experiment.analysis_history.append(result)

        return result

    # -------------------------------------------------------------------------
    # Deployment
    # -------------------------------------------------------------------------

    def can_deploy(self, experiment_id: str) -> Tuple[bool, str]:
        """Check if treatment can be deployed.

        Args:
            experiment_id: Experiment ID

        Returns:
            Tuple of (can_deploy, reason)
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status == ExperimentStatus.RUNNING:
            return False, "Experiment still running"

        if experiment.latest_result is None:
            self._analyze_experiment(experiment)

        result = experiment.latest_result
        return result.can_deploy, result.recommendation

    def deploy_winner(self, experiment_id: str) -> str:
        """Get the winning strategy to deploy.

        Args:
            experiment_id: Experiment ID

        Returns:
            Strategy ID to deploy

        Raises:
            ValueError: If deployment not allowed
        """
        experiment = self._get_experiment(experiment_id)
        can_deploy, reason = self.can_deploy(experiment_id)

        if not can_deploy:
            raise ValueError(f"Cannot deploy: {reason}")

        result = experiment.latest_result

        if result.cohens_d > 0:
            winner = experiment.treatment.strategy_id
        else:
            winner = experiment.control.strategy_id

        experiment.status = ExperimentStatus.DEPLOYED
        return winner

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID, raising if not found."""
        if experiment_id not in self._experiments:
            raise KeyError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]


# =============================================================================
# Convenience Functions
# =============================================================================


def create_ab_test_manager(seed: int = 42) -> ABTestManager:
    """Create an A/B test manager.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured ABTestManager
    """
    return ABTestManager(seed=seed)


def create_experiment_config(
    name: str,
    control_strategy: str,
    treatment_strategy: str,
    description: str = "",
    significance_level: SignificanceLevel = SignificanceLevel.STANDARD,
    min_samples: int = 30,
    max_samples: int = 10000,
) -> ExperimentConfig:
    """Create an experiment configuration.

    Args:
        name: Experiment name
        control_strategy: Strategy ID for control variant
        treatment_strategy: Strategy ID for treatment variant
        description: Experiment description
        significance_level: Statistical significance threshold
        min_samples: Minimum samples per variant
        max_samples: Maximum total samples

    Returns:
        Configured ExperimentConfig
    """
    return ExperimentConfig(
        name=name,
        description=description,
        control_strategy=control_strategy,
        treatment_strategy=treatment_strategy,
        significance_level=significance_level,
        min_samples_per_variant=min_samples,
        max_samples=max_samples,
    )


async def run_ab_test(
    manager: ABTestManager,
    experiment_id: str,
    get_reward: Callable[[str], float],
    n_samples: int = 1000,
) -> StatisticalResult:
    """Run an A/B test to completion.

    Args:
        manager: A/B test manager
        experiment_id: Experiment to run
        get_reward: Function that returns reward given strategy ID
        n_samples: Number of samples to collect

    Returns:
        Final statistical results
    """
    manager.start_experiment(experiment_id)

    for i in range(n_samples):
        # Get experiment (might have been stopped early)
        try:
            experiment = manager.get_experiment(experiment_id)
        except KeyError:
            break

        if experiment.status != ExperimentStatus.RUNNING:
            break

        # Assign variant
        variant = manager.assign_variant(experiment_id, request_id=str(i))

        # Get reward
        reward = get_reward(variant.strategy_id)

        # Record result
        manager.record_result(
            experiment_id,
            variant.name,
            reward,
            success=reward > 0,
        )

        # Small delay to avoid tight loop
        await asyncio.sleep(0)

    # Ensure experiment is completed
    experiment = manager.get_experiment(experiment_id)
    if experiment.status == ExperimentStatus.RUNNING:
        manager.complete_experiment(experiment_id)

    return experiment.latest_result
