"""
Attack A/B Testing Framework for CARTS
======================================

Statistical A/B testing framework for comparing attack strategies,
payload variants, and mutation approaches.

Implements ABTestProtocol with:
- Multiple concurrent A/B tests
- Statistical significance testing (t-test, Mann-Whitney)
- Effect size calculation (Cohen's d)
- Minimum sample size enforcement
- Auto-conclusion when significance reached

Philosophy: "Measure twice, deploy once."

Author: CARTS Team
Date: 2025-12-03
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


# =============================================================================
# Enums
# =============================================================================

class TestStatus(Enum):
    """Status of an A/B test."""
    ACTIVE = "active"
    CONCLUDED = "concluded"
    INSUFFICIENT_DATA = "insufficient_data"
    PAUSED = "paused"


class SignificanceMethod(Enum):
    """Statistical significance method."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    BAYESIAN = "bayesian"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestVariant:
    """
    Data for one variant (control or treatment) in an A/B test.

    Attributes:
        name: Variant identifier (control/treatment)
        description: Human-readable description
        results: List of (success: bool, severity: float) tuples
        total_trials: Number of trials
        successes: Number of successful attacks
        total_severity: Sum of severity scores
    """
    name: str
    description: str
    results: List[Tuple[bool, float]] = field(default_factory=list)
    total_trials: int = 0
    successes: int = 0
    total_severity: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_trials == 0:
            return 0.0
        return self.successes / self.total_trials

    @property
    def mean_severity(self) -> float:
        """Calculate mean severity (among successes)."""
        if self.successes == 0:
            return 0.0
        return self.total_severity / self.successes

    @property
    def variance(self) -> float:
        """Calculate variance of success rate."""
        if self.total_trials < 2:
            return 0.0
        p = self.success_rate
        return p * (1 - p) / self.total_trials

    def record(self, success: bool, severity: float = 0.0) -> None:
        """Record a trial result."""
        self.results.append((success, severity))
        self.total_trials += 1
        if success:
            self.successes += 1
            self.total_severity += severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'total_trials': self.total_trials,
            'successes': self.successes,
            'success_rate': self.success_rate,
            'total_severity': self.total_severity,
            'mean_severity': self.mean_severity,
        }


@dataclass
class ABTest:
    """
    A single A/B test comparing two variants.

    Attributes:
        name: Unique test identifier
        control: Control variant data
        treatment: Treatment variant data
        min_samples: Minimum samples per variant before analysis
        status: Current test status
        created_at: Test creation timestamp
        concluded_at: Test conclusion timestamp (if concluded)
        conclusion: Final result (if concluded)
    """
    name: str
    control: TestVariant
    treatment: TestVariant
    min_samples: int = 30
    status: TestStatus = TestStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    concluded_at: Optional[str] = None
    conclusion: Optional[Dict[str, Any]] = None

    def can_analyze(self) -> bool:
        """Check if we have enough samples for analysis."""
        return (
            self.control.total_trials >= self.min_samples and
            self.treatment.total_trials >= self.min_samples
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'control': self.control.to_dict(),
            'treatment': self.treatment.to_dict(),
            'min_samples': self.min_samples,
            'status': self.status.value,
            'created_at': self.created_at,
            'concluded_at': self.concluded_at,
            'conclusion': self.conclusion,
        }


@dataclass
class ABTestAnalysis:
    """
    Statistical analysis result for an A/B test.

    Attributes:
        winner: "control", "treatment", or "inconclusive"
        p_value: Statistical significance (p < 0.05 = significant)
        effect_size: Cohen's d effect size
        control_mean: Mean success rate for control
        treatment_mean: Mean success rate for treatment
        sample_size_control: Number of samples in control
        sample_size_treatment: Number of samples in treatment
        is_significant: True if p_value < 0.05
        improvement: Relative improvement of treatment over control
        confidence_interval: 95% CI for effect size
        method: Statistical method used
    """
    winner: str
    p_value: float
    effect_size: float
    control_mean: float
    treatment_mean: float
    sample_size_control: int
    sample_size_treatment: int
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    method: SignificanceMethod = SignificanceMethod.T_TEST

    @property
    def is_significant(self) -> bool:
        """True if p_value < 0.05."""
        return self.p_value < 0.05

    @property
    def improvement(self) -> float:
        """Percentage improvement of treatment over control."""
        if self.control_mean == 0:
            return float('inf') if self.treatment_mean > 0 else 0.0
        return (self.treatment_mean - self.control_mean) / self.control_mean

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'winner': self.winner,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'control_mean': self.control_mean,
            'treatment_mean': self.treatment_mean,
            'sample_size_control': self.sample_size_control,
            'sample_size_treatment': self.sample_size_treatment,
            'is_significant': self.is_significant,
            'improvement': self.improvement,
            'confidence_interval': list(self.confidence_interval),
            'method': self.method.value,
        }


# =============================================================================
# Main Class
# =============================================================================

class AttackABTester:
    """
    A/B testing framework for comparing attack strategies.

    Implements ABTestProtocol with statistical rigor:
    - Creates and manages multiple concurrent tests
    - Tracks results per variant
    - Computes statistical significance
    - Auto-concludes when significance reached

    Example:
        tester = AttackABTester()

        # Create a test
        tester.create_test(
            name="unicode_vs_encoding",
            control="Unicode bypass (standard)",
            treatment="Encoding bypass (new)",
            min_samples=50
        )

        # Record results
        tester.record_result("unicode_vs_encoding", "control", True, 0.7)
        tester.record_result("unicode_vs_encoding", "treatment", True, 0.9)

        # Analyze
        result = tester.analyze("unicode_vs_encoding")
        if result and result.is_significant:
            print(f"Winner: {result.winner}, improvement: {result.improvement:.1%}")
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.2,
        auto_conclude: bool = True,
        method: SignificanceMethod = SignificanceMethod.T_TEST,
    ):
        """
        Initialize A/B tester.

        Args:
            significance_threshold: p-value threshold for significance
            min_effect_size: Minimum Cohen's d to consider meaningful
            auto_conclude: Auto-conclude when significance reached
            method: Statistical method for significance testing
        """
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size
        self.auto_conclude = auto_conclude
        self.method = method

        self._tests: Dict[str, ABTest] = {}

    # -------------------------------------------------------------------------
    # ABTestProtocol Implementation
    # -------------------------------------------------------------------------

    def create_test(
        self,
        name: str,
        control: str,
        treatment: str,
        min_samples: int = 30,
    ) -> None:
        """
        Create a new A/B test.

        Args:
            name: Unique test name
            control: Description of control variant
            treatment: Description of treatment variant
            min_samples: Minimum samples per variant before analysis
        """
        if name in self._tests:
            raise ValueError(f"Test '{name}' already exists")

        self._tests[name] = ABTest(
            name=name,
            control=TestVariant(name="control", description=control),
            treatment=TestVariant(name="treatment", description=treatment),
            min_samples=min_samples,
        )

    def record_result(
        self,
        test_name: str,
        variant: str,
        success: bool,
        severity: float = 0.0,
    ) -> None:
        """
        Record a result for a test variant.

        Args:
            test_name: Name of the test
            variant: "control" or "treatment"
            success: Whether the attack succeeded
            severity: Severity score if successful
        """
        if test_name not in self._tests:
            raise ValueError(f"Test '{test_name}' not found")

        test = self._tests[test_name]

        if test.status != TestStatus.ACTIVE:
            return  # Ignore results for concluded tests

        if variant == "control":
            test.control.record(success, severity)
        elif variant == "treatment":
            test.treatment.record(success, severity)
        else:
            raise ValueError(f"Invalid variant '{variant}', must be 'control' or 'treatment'")

        # Auto-conclude if enabled and significant
        if self.auto_conclude and test.can_analyze():
            analysis = self._compute_analysis(test)
            if analysis and analysis.is_significant:
                self.conclude_test(test_name)

    def analyze(self, test_name: str) -> Optional[ABTestAnalysis]:
        """
        Analyze test results.

        Returns None if insufficient samples.

        Args:
            test_name: Name of the test

        Returns:
            ABTestAnalysis or None if insufficient data
        """
        if test_name not in self._tests:
            raise ValueError(f"Test '{test_name}' not found")

        test = self._tests[test_name]

        if not test.can_analyze():
            return None

        return self._compute_analysis(test)

    def get_active_tests(self) -> List[str]:
        """Get names of all active tests."""
        return [
            name for name, test in self._tests.items()
            if test.status == TestStatus.ACTIVE
        ]

    def conclude_test(self, test_name: str) -> Optional[ABTestAnalysis]:
        """
        Conclude a test and return final results.

        Args:
            test_name: Name of the test

        Returns:
            Final ABTestAnalysis or None if insufficient data
        """
        if test_name not in self._tests:
            raise ValueError(f"Test '{test_name}' not found")

        test = self._tests[test_name]

        if test.status == TestStatus.CONCLUDED:
            # Already concluded, return cached result
            if test.conclusion:
                return ABTestAnalysis(**test.conclusion)
            return None

        analysis = self._compute_analysis(test)

        test.status = TestStatus.CONCLUDED
        test.concluded_at = datetime.now().isoformat()

        if analysis:
            test.conclusion = analysis.to_dict()

        return analysis

    # -------------------------------------------------------------------------
    # Statistical Methods
    # -------------------------------------------------------------------------

    def _compute_analysis(self, test: ABTest) -> Optional[ABTestAnalysis]:
        """Compute statistical analysis for a test."""
        if not test.can_analyze():
            return None

        control = test.control
        treatment = test.treatment

        # Calculate means
        control_mean = control.success_rate
        treatment_mean = treatment.success_rate

        # Calculate pooled standard deviation for Cohen's d
        pooled_std = self._pooled_std(control, treatment)

        # Calculate effect size (Cohen's d)
        if pooled_std > 0:
            effect_size = (treatment_mean - control_mean) / pooled_std
        else:
            effect_size = 0.0

        # Calculate p-value based on method
        p_value = self._calculate_p_value(control, treatment)

        # Determine winner
        if p_value < self.significance_threshold:
            if treatment_mean > control_mean:
                winner = "treatment"
            elif control_mean > treatment_mean:
                winner = "control"
            else:
                winner = "inconclusive"
        else:
            winner = "inconclusive"

        # Calculate confidence interval for effect size
        ci = self._effect_size_ci(effect_size, control.total_trials, treatment.total_trials)

        return ABTestAnalysis(
            winner=winner,
            p_value=p_value,
            effect_size=effect_size,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            sample_size_control=control.total_trials,
            sample_size_treatment=treatment.total_trials,
            confidence_interval=ci,
            method=self.method,
        )

    def _pooled_std(self, control: TestVariant, treatment: TestVariant) -> float:
        """Calculate pooled standard deviation."""
        n1 = control.total_trials
        n2 = treatment.total_trials

        # Variance of binomial proportion
        var1 = control.success_rate * (1 - control.success_rate)
        var2 = treatment.success_rate * (1 - treatment.success_rate)

        if n1 + n2 - 2 <= 0:
            return 0.0

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        return math.sqrt(pooled_var)

    def _calculate_p_value(self, control: TestVariant, treatment: TestVariant) -> float:
        """Calculate p-value using specified method."""
        if self.method == SignificanceMethod.T_TEST:
            return self._t_test_p_value(control, treatment)
        elif self.method == SignificanceMethod.MANN_WHITNEY:
            return self._mann_whitney_p_value(control, treatment)
        elif self.method == SignificanceMethod.BAYESIAN:
            return self._bayesian_p_value(control, treatment)
        else:
            return self._t_test_p_value(control, treatment)

    def _t_test_p_value(self, control: TestVariant, treatment: TestVariant) -> float:
        """Calculate p-value using two-sample t-test for proportions."""
        n1 = control.total_trials
        n2 = treatment.total_trials
        p1 = control.success_rate
        p2 = treatment.success_rate

        # Pooled proportion under null hypothesis
        p_pooled = (control.successes + treatment.successes) / (n1 + n2)

        if p_pooled == 0 or p_pooled == 1:
            return 1.0  # No variation

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        if se == 0:
            return 1.0

        # Z-statistic
        z = (p2 - p1) / se

        # Two-tailed p-value using normal approximation
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    def _mann_whitney_p_value(self, control: TestVariant, treatment: TestVariant) -> float:
        """Calculate p-value using Mann-Whitney U test approximation."""
        # Simplified: use normal approximation for large samples
        n1 = control.total_trials
        n2 = treatment.total_trials

        # Count ranks (simplified for binary outcomes)
        # This is a simplified approximation
        u1 = control.successes * (n2 - treatment.successes) + \
             (control.total_trials - control.successes) * treatment.successes * 0.5

        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

        if sigma == 0:
            return 1.0

        z = (u1 - mu) / sigma
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        return p_value

    def _bayesian_p_value(self, control: TestVariant, treatment: TestVariant) -> float:
        """Calculate Bayesian probability that treatment > control."""
        # Monte Carlo estimation
        n_samples = 10000

        alpha1 = 1 + control.successes
        beta1 = 1 + control.total_trials - control.successes
        alpha2 = 1 + treatment.successes
        beta2 = 1 + treatment.total_trials - treatment.successes

        treatment_wins = 0
        for _ in range(n_samples):
            sample1 = random.betavariate(alpha1, beta1)
            sample2 = random.betavariate(alpha2, beta2)
            if sample2 > sample1:
                treatment_wins += 1

        prob_better = treatment_wins / n_samples

        # Convert to two-tailed p-value
        p_value = 2 * min(prob_better, 1 - prob_better)

        return p_value

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)

    def _effect_size_ci(
        self,
        d: float,
        n1: int,
        n2: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Cohen's d."""
        # Standard error of d
        se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        lower = d - z * se
        upper = d + z * se

        return (lower, upper)

    # -------------------------------------------------------------------------
    # Assignment & Persistence
    # -------------------------------------------------------------------------

    def assign_variant(self, test_name: str) -> str:
        """
        Randomly assign a variant for a new trial.

        Args:
            test_name: Name of the test

        Returns:
            "control" or "treatment"
        """
        if test_name not in self._tests:
            raise ValueError(f"Test '{test_name}' not found")

        return "treatment" if random.random() < 0.5 else "control"

    def save(self, path: Path) -> None:
        """Save all tests to file."""
        data = {
            'tests': {name: test.to_dict() for name, test in self._tests.items()},
            'config': {
                'significance_threshold': self.significance_threshold,
                'min_effect_size': self.min_effect_size,
                'auto_conclude': self.auto_conclude,
                'method': self.method.value,
            }
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load tests from file."""
        path = Path(path)

        if not path.exists():
            return

        with open(path, 'r') as f:
            data = json.load(f)

        # Load config
        config = data.get('config', {})
        self.significance_threshold = config.get('significance_threshold', 0.05)
        self.min_effect_size = config.get('min_effect_size', 0.2)
        self.auto_conclude = config.get('auto_conclude', True)
        self.method = SignificanceMethod(config.get('method', 't_test'))

        # Load tests
        for name, test_data in data.get('tests', {}).items():
            control_data = test_data['control']
            treatment_data = test_data['treatment']

            control = TestVariant(
                name="control",
                description=control_data['description'],
                total_trials=control_data['total_trials'],
                successes=control_data['successes'],
                total_severity=control_data['total_severity'],
            )

            treatment = TestVariant(
                name="treatment",
                description=treatment_data['description'],
                total_trials=treatment_data['total_trials'],
                successes=treatment_data['successes'],
                total_severity=treatment_data['total_severity'],
            )

            self._tests[name] = ABTest(
                name=name,
                control=control,
                treatment=treatment,
                min_samples=test_data['min_samples'],
                status=TestStatus(test_data['status']),
                created_at=test_data['created_at'],
                concluded_at=test_data.get('concluded_at'),
                conclusion=test_data.get('conclusion'),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        active = [t for t in self._tests.values() if t.status == TestStatus.ACTIVE]
        concluded = [t for t in self._tests.values() if t.status == TestStatus.CONCLUDED]

        return {
            'total_tests': len(self._tests),
            'active_tests': len(active),
            'concluded_tests': len(concluded),
            'tests': {name: test.to_dict() for name, test in self._tests.items()},
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ab_tester(
    significance_threshold: float = 0.05,
    min_effect_size: float = 0.2,
    auto_conclude: bool = True,
    method: str = "t_test",
) -> AttackABTester:
    """
    Create an A/B tester with specified configuration.

    Args:
        significance_threshold: p-value threshold for significance
        min_effect_size: Minimum Cohen's d to consider meaningful
        auto_conclude: Auto-conclude when significance reached
        method: Statistical method ("t_test", "mann_whitney", "bayesian")

    Returns:
        Configured AttackABTester
    """
    return AttackABTester(
        significance_threshold=significance_threshold,
        min_effect_size=min_effect_size,
        auto_conclude=auto_conclude,
        method=SignificanceMethod(method),
    )


def run_quick_ab_test(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
) -> ABTestAnalysis:
    """
    Run a quick A/B test analysis on aggregated results.

    Args:
        control_successes: Number of successes in control
        control_trials: Total trials in control
        treatment_successes: Number of successes in treatment
        treatment_trials: Total trials in treatment

    Returns:
        ABTestAnalysis with results
    """
    tester = AttackABTester(auto_conclude=False)

    tester.create_test(
        name="quick_test",
        control="Control variant",
        treatment="Treatment variant",
        min_samples=1,
    )

    # Record control results
    for i in range(control_trials):
        success = i < control_successes
        tester.record_result("quick_test", "control", success)

    # Record treatment results
    for i in range(treatment_trials):
        success = i < treatment_successes
        tester.record_result("quick_test", "treatment", success)

    return tester.analyze("quick_test")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    'TestStatus',
    'SignificanceMethod',

    # Data Classes
    'TestVariant',
    'ABTest',
    'ABTestAnalysis',

    # Main Class
    'AttackABTester',

    # Convenience Functions
    'create_ab_tester',
    'run_quick_ab_test',
]
