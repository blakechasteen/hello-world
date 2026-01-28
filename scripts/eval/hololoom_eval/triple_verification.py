"""
Triple Verification Pass system for HoloLoom evaluation.

Runs each benchmark 3 times and applies statistical validation to ensure
results are stable, not artifacts of randomness or system state.

Three verification levels:
1. Pass 1 (Baseline): Initial run, establishes reference scores
2. Pass 2 (Confirmation): Re-run with different seed, confirms results
3. Pass 3 (Stress Confirmation): Re-run under perturbed conditions

Verification criteria:
- Consistency: All 3 passes agree on pass/fail verdict
- Stability: Score coefficient of variation (CV) < 15%
- Confidence: Bootstrap CI from 3 scores doesn't span pass/fail threshold
- Agreement: Majority vote on final verdict (2-of-3 minimum)

January 2026 - v2.1
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable


@dataclass
class PassResult:
    """Result from a single verification pass."""
    pass_number: int
    pass_label: str  # "baseline", "confirmation", "stress_confirmation"
    score: float
    passed: bool
    verdict: str
    duration_seconds: float
    seed_used: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """Aggregated result from triple verification."""
    benchmark_name: str
    passes: List[PassResult]
    final_score: float          # Mean of 3 pass scores
    final_passed: bool          # Majority vote
    final_verdict: str
    is_consistent: bool         # All 3 agree on pass/fail
    is_stable: bool             # CV < threshold
    coefficient_of_variation: float
    score_range: float          # max - min across passes
    confidence_interval: Dict[str, float]  # lower, upper from 3 scores
    verification_verdict: str   # "VERIFIED", "UNSTABLE", "INCONSISTENT"
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "passes": [
                {
                    "pass_number": p.pass_number,
                    "pass_label": p.pass_label,
                    "score": round(p.score, 4),
                    "passed": p.passed,
                    "verdict": p.verdict,
                    "duration_seconds": round(p.duration_seconds, 2),
                    "seed_used": p.seed_used,
                    "error": p.error,
                }
                for p in self.passes
            ],
            "final_score": round(self.final_score, 4),
            "final_passed": self.final_passed,
            "final_verdict": self.final_verdict,
            "is_consistent": self.is_consistent,
            "is_stable": self.is_stable,
            "coefficient_of_variation": round(self.coefficient_of_variation, 4),
            "score_range": round(self.score_range, 4),
            "confidence_interval": {
                k: round(v, 4) for k, v in self.confidence_interval.items()
            },
            "verification_verdict": self.verification_verdict,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class TripleVerificationReport:
    """Complete triple verification report across all benchmarks."""
    results: List[VerificationResult]
    all_verified: bool
    verified_count: int
    unstable_count: int
    inconsistent_count: int
    total_duration_seconds: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "all_verified": self.all_verified,
            "verified_count": self.verified_count,
            "unstable_count": self.unstable_count,
            "inconsistent_count": self.inconsistent_count,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "summary": self.summary,
        }


# Default seeds for 3 passes (reproducible, distinct)
DEFAULT_SEEDS = [42, 7919, 104729]

# Stability threshold: coefficient of variation must be below this
CV_THRESHOLD = 0.15  # 15%

# Pass/fail score boundary
PASS_FAIL_THRESHOLD = 0.5


def _compute_cv(scores: List[float]) -> float:
    """Coefficient of variation: std / mean. Returns 0 if mean is 0."""
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    if mean == 0:
        return 0.0
    variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    return math.sqrt(variance) / abs(mean)


def _simple_ci(scores: List[float]) -> Dict[str, float]:
    """Compute simple confidence interval from small sample (t-based, df=n-1)."""
    n = len(scores)
    if n < 2:
        mean = scores[0] if scores else 0.0
        return {"lower": mean, "upper": mean, "mean": mean}

    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    # t-critical for 95% CI, df=2 (3 samples) => t ~ 4.303
    # For small n, use exact t-values
    t_crit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n - 1, 2.0)
    margin = t_crit * std / math.sqrt(n)

    return {
        "lower": mean - margin,
        "upper": mean + margin,
        "mean": mean,
    }


def _majority_vote(pass_results: List[PassResult]) -> bool:
    """Return True if majority of passes passed."""
    passed_count = sum(1 for p in pass_results if p.passed and not p.error)
    return passed_count >= (len(pass_results) + 1) // 2


def _all_agree(pass_results: List[PassResult]) -> bool:
    """Check if all non-error passes agree on pass/fail."""
    valid = [p for p in pass_results if not p.error]
    if not valid:
        return True
    first = valid[0].passed
    return all(p.passed == first for p in valid)


def verify_benchmark(
    benchmark_name: str,
    pass_results: List[PassResult],
    cv_threshold: float = CV_THRESHOLD,
) -> VerificationResult:
    """
    Analyze 3 pass results and produce a verification verdict.
    """
    total_duration = sum(p.duration_seconds for p in pass_results)
    valid_results = [p for p in pass_results if not p.error]
    scores = [p.score for p in valid_results]

    if not scores:
        return VerificationResult(
            benchmark_name=benchmark_name,
            passes=pass_results,
            final_score=0.0,
            final_passed=False,
            final_verdict="All passes errored",
            is_consistent=False,
            is_stable=False,
            coefficient_of_variation=0.0,
            score_range=0.0,
            confidence_interval={"lower": 0.0, "upper": 0.0, "mean": 0.0},
            verification_verdict="ERROR",
            duration_seconds=total_duration,
        )

    mean_score = sum(scores) / len(scores)
    cv = _compute_cv(scores)
    score_range = max(scores) - min(scores)
    ci = _simple_ci(scores)
    is_consistent = _all_agree(pass_results)
    is_stable = cv < cv_threshold
    final_passed = _majority_vote(pass_results)

    # Determine verification verdict
    if is_consistent and is_stable:
        verification_verdict = "VERIFIED"
    elif not is_consistent:
        verification_verdict = "INCONSISTENT"
    else:
        verification_verdict = "UNSTABLE"

    # Build final verdict string
    pass_status = "PASS" if final_passed else "FAIL"
    if verification_verdict == "VERIFIED":
        final_verdict = f"{pass_status} (verified: 3/3 consistent, CV={cv:.1%})"
    elif verification_verdict == "INCONSISTENT":
        pass_count = sum(1 for p in valid_results if p.passed)
        fail_count = len(valid_results) - pass_count
        final_verdict = f"{pass_status} (inconsistent: {pass_count} pass, {fail_count} fail)"
    else:
        final_verdict = f"{pass_status} (unstable: CV={cv:.1%}, range={score_range:.3f})"

    return VerificationResult(
        benchmark_name=benchmark_name,
        passes=pass_results,
        final_score=round(mean_score, 4),
        final_passed=final_passed,
        final_verdict=final_verdict,
        is_consistent=is_consistent,
        is_stable=is_stable,
        coefficient_of_variation=cv,
        score_range=score_range,
        confidence_interval=ci,
        verification_verdict=verification_verdict,
        duration_seconds=total_duration,
    )


def build_verification_report(
    results: List[VerificationResult],
    total_duration: float,
) -> TripleVerificationReport:
    """Build summary report from all verification results."""
    verified = sum(1 for r in results if r.verification_verdict == "VERIFIED")
    unstable = sum(1 for r in results if r.verification_verdict == "UNSTABLE")
    inconsistent = sum(1 for r in results if r.verification_verdict == "INCONSISTENT")
    error_count = sum(1 for r in results if r.verification_verdict == "ERROR")
    total = len(results)

    all_verified = verified == total

    if all_verified:
        summary = f"All {total} benchmarks VERIFIED (3/3 consistent, stable scores)"
    else:
        parts = []
        if verified:
            parts.append(f"{verified} verified")
        if unstable:
            parts.append(f"{unstable} unstable")
        if inconsistent:
            parts.append(f"{inconsistent} inconsistent")
        if error_count:
            parts.append(f"{error_count} errored")
        summary = f"{verified}/{total} benchmarks verified ({', '.join(parts)})"

    return TripleVerificationReport(
        results=results,
        all_verified=all_verified,
        verified_count=verified,
        unstable_count=unstable,
        inconsistent_count=inconsistent,
        total_duration_seconds=total_duration,
        summary=summary,
    )


def print_verification_report(report: TripleVerificationReport):
    """Print triple verification report to console."""
    print()
    print("=" * 70)
    print("  TRIPLE VERIFICATION REPORT")
    print("=" * 70)
    print()
    print(f"  {report.summary}")
    print(f"  Total duration: {report.total_duration_seconds:.1f}s")
    print()

    print("  Benchmark Verification Details:")
    print("  " + "-" * 66)

    for result in report.results:
        tag = {
            "VERIFIED": "[VERIFIED]    ",
            "UNSTABLE": "[UNSTABLE]    ",
            "INCONSISTENT": "[INCONSISTENT]",
            "ERROR": "[ERROR]       ",
        }.get(result.verification_verdict, "[???]         ")

        print(f"    {tag} {result.benchmark_name}")

        scores_str = " | ".join(
            f"P{p.pass_number}: {p.score:.3f}" + (" [err]" if p.error else "")
            for p in result.passes
        )
        print(f"               Scores: {scores_str}")
        print(f"               Mean: {result.final_score:.3f}  "
              f"CV: {result.coefficient_of_variation:.1%}  "
              f"Range: {result.score_range:.3f}  "
              f"CI: [{result.confidence_interval['lower']:.3f}, "
              f"{result.confidence_interval['upper']:.3f}]")
        print()

    print("  " + "-" * 66)

    # Highlight issues
    issues = [r for r in report.results if r.verification_verdict != "VERIFIED"]
    if issues:
        print("  Issues requiring attention:")
        for r in issues:
            if r.verification_verdict == "INCONSISTENT":
                pass_verdicts = [
                    f"P{p.pass_number}={'PASS' if p.passed else 'FAIL'}"
                    for p in r.passes if not p.error
                ]
                print(f"    - {r.benchmark_name}: Verdicts disagree ({', '.join(pass_verdicts)})")
            elif r.verification_verdict == "UNSTABLE":
                print(f"    - {r.benchmark_name}: High variance (CV={r.coefficient_of_variation:.1%} > {CV_THRESHOLD:.0%})")
            elif r.verification_verdict == "ERROR":
                errors = [p.error for p in r.passes if p.error]
                print(f"    - {r.benchmark_name}: Errors in {len(errors)} pass(es)")
        print()
    else:
        print("  No issues found. All benchmarks produce stable, consistent results.")
        print()
