"""
Feedback Tracker
================

🐷 Charlotte's Web of Learning! 🐷

Tracks fix outcomes to enable institutional learning and Thompson Sampling.

What Gets Tracked:
- Fix attempts (proposed, applied, validated)
- Outcomes (success, failure, rollback)
- Strategy performance (AST vs Template vs Manual)
- Confidence calibration (predicted vs actual success)

Learning Signals:
- Which fix strategies work best per issue category
- Confidence score accuracy (calibration)
- False positive patterns
- Degradation detection

Moonshot Integration:
- Enables adaptive fix strategy selection (Idea #2)
- Provides data for Thompson Sampling (Phase 4)
- Tracks system confidence over time (Idea #4)
- Detects degradation patterns (Idea #7)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List
from .xterminator_types import RiskLevel, FixStrategy


class FixOutcome(Enum):
    """Outcome of a fix attempt"""
    SUCCESS = "success"              # Fix applied successfully, validated
    PARTIAL = "partial"              # Fix applied but validation warnings
    FAILURE = "failure"              # Fix failed to apply or validate
    ROLLBACK = "rollback"            # Fix rolled back after issues detected
    SKIPPED = "skipped"              # Fix skipped (false positive or low confidence)
    PENDING = "pending"              # Fix applied, awaiting validation


@dataclass
class FixAttempt:
    """
    Record of a single fix attempt.

    Stores all context needed for learning and Thompson Sampling.
    """
    # Identification
    attempt_id: str
    timestamp: float
    file_path: str
    issue_category: str
    line_number: int

    # Classification metadata
    confidence: float
    risk_level: str
    fix_strategy: str
    context_type: str
    has_tests: bool
    scope_depth: int

    # Decision
    decision: str  # AUTO, REVIEW, MANUAL, SKIP
    decision_reason: str

    # Outcome (filled in after validation)
    outcome: str = FixOutcome.PENDING.value
    outcome_reason: str = ""
    validation_passed: bool = False
    validation_duration_ms: float = 0.0

    # Validation details
    syntax_passed: bool = False
    imports_passed: bool = False
    tests_passed: bool = False
    trough_passed: bool = False
    regression_passed: bool = False

    # Performance
    fix_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Git metadata (if applied)
    commit_sha: Optional[str] = None
    rollback_sha: Optional[str] = None

    # Learning signals
    was_correct_strategy: bool = True  # Did the strategy work?
    confidence_accurate: bool = True   # Was confidence score accurate?
    false_positive_detected: bool = False  # Was this a false positive?

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'FixAttempt':
        """Create from dictionary"""
        return cls(**data)

    def mark_success(self, reason: str = "Fix validated successfully"):
        """Mark attempt as successful"""
        self.outcome = FixOutcome.SUCCESS.value
        self.outcome_reason = reason
        self.validation_passed = True
        self.was_correct_strategy = True
        self.confidence_accurate = True

    def mark_failure(self, reason: str):
        """Mark attempt as failed"""
        self.outcome = FixOutcome.FAILURE.value
        self.outcome_reason = reason
        self.validation_passed = False
        self.was_correct_strategy = False

    def mark_rollback(self, reason: str):
        """Mark attempt as rolled back"""
        self.outcome = FixOutcome.ROLLBACK.value
        self.outcome_reason = reason
        self.validation_passed = False
        self.was_correct_strategy = False


class FeedbackTracker:
    """
    Tracks fix outcomes for institutional learning.

    Stores attempts in JSON for persistence and analysis.
    """

    def __init__(self, log_file: str = "./xterminator_feedback.jsonl"):
        self.log_file = Path(log_file)
        self.attempts: List[FixAttempt] = []

        # Load existing attempts
        self._load_attempts()

    def record_attempt(self, attempt: FixAttempt):
        """
        Record a fix attempt.

        Args:
            attempt: FixAttempt object
        """
        self.attempts.append(attempt)
        self._save_attempt(attempt)

    def update_outcome(
        self,
        attempt_id: str,
        outcome: FixOutcome,
        reason: str = "",
        validation_results: Optional[dict] = None
    ):
        """
        Update outcome of a previously recorded attempt.

        Args:
            attempt_id: Attempt ID
            outcome: FixOutcome enum
            reason: Reason for outcome
            validation_results: Optional validation details
        """
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                attempt.outcome = outcome.value
                attempt.outcome_reason = reason

                if validation_results:
                    attempt.validation_passed = validation_results.get('all_passed', False)
                    attempt.syntax_passed = validation_results.get('syntax_passed', False)
                    attempt.imports_passed = validation_results.get('imports_passed', False)
                    attempt.tests_passed = validation_results.get('tests_passed', False)
                    attempt.trough_passed = validation_results.get('trough_passed', False)
                    attempt.regression_passed = validation_results.get('regression_passed', False)
                    attempt.validation_duration_ms = validation_results.get('duration_ms', 0.0)

                # Update learning signals
                if outcome == FixOutcome.SUCCESS:
                    attempt.was_correct_strategy = True
                    attempt.confidence_accurate = True
                elif outcome in [FixOutcome.FAILURE, FixOutcome.ROLLBACK]:
                    attempt.was_correct_strategy = False
                    # Check if confidence was too high
                    if attempt.confidence >= 0.85:
                        attempt.confidence_accurate = False

                self._save_attempt(attempt)
                break

    def get_strategy_performance(self, strategy: FixStrategy) -> dict:
        """
        Get performance metrics for a fix strategy.

        Args:
            strategy: FixStrategy enum

        Returns:
            Dictionary with success rate, count, etc.
        """
        strategy_attempts = [
            a for a in self.attempts
            if a.fix_strategy == strategy.value
        ]

        if not strategy_attempts:
            return {
                'strategy': strategy.value,
                'total_attempts': 0,
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_duration_ms': 0.0
            }

        successes = sum(1 for a in strategy_attempts if a.outcome == FixOutcome.SUCCESS.value)
        success_rate = successes / len(strategy_attempts)

        avg_confidence = sum(a.confidence for a in strategy_attempts) / len(strategy_attempts)
        avg_duration = sum(a.total_duration_ms for a in strategy_attempts) / len(strategy_attempts)

        return {
            'strategy': strategy.value,
            'total_attempts': len(strategy_attempts),
            'successes': successes,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'avg_duration_ms': avg_duration
        }

    def get_confidence_calibration(self) -> dict:
        """
        Analyze confidence score calibration.

        Returns:
            Calibration metrics (accuracy, overconfidence, underconfidence)
        """
        completed_attempts = [
            a for a in self.attempts
            if a.outcome != FixOutcome.PENDING.value
        ]

        if not completed_attempts:
            return {
                'total_attempts': 0,
                'calibration_accuracy': 0.0,
                'overconfident_rate': 0.0,
                'underconfident_rate': 0.0
            }

        # Group by confidence bins
        bins = {
            'very_high': [],  # ≥0.85
            'high': [],       # 0.70-0.85
            'medium': [],     # 0.55-0.70
            'low': []         # <0.55
        }

        for attempt in completed_attempts:
            conf = attempt.confidence
            if conf >= 0.85:
                bins['very_high'].append(attempt)
            elif conf >= 0.70:
                bins['high'].append(attempt)
            elif conf >= 0.55:
                bins['medium'].append(attempt)
            else:
                bins['low'].append(attempt)

        # Calculate success rates per bin
        bin_stats = {}
        for bin_name, bin_attempts in bins.items():
            if bin_attempts:
                successes = sum(1 for a in bin_attempts if a.outcome == FixOutcome.SUCCESS.value)
                bin_stats[bin_name] = {
                    'count': len(bin_attempts),
                    'success_rate': successes / len(bin_attempts),
                    'avg_confidence': sum(a.confidence for a in bin_attempts) / len(bin_attempts)
                }

        # Detect overconfidence/underconfidence
        overconfident = 0
        underconfident = 0

        for attempt in completed_attempts:
            success = attempt.outcome == FixOutcome.SUCCESS.value
            if attempt.confidence >= 0.85 and not success:
                overconfident += 1
            elif attempt.confidence < 0.70 and success:
                underconfident += 1

        return {
            'total_attempts': len(completed_attempts),
            'bin_statistics': bin_stats,
            'overconfident_rate': overconfident / len(completed_attempts),
            'underconfident_rate': underconfident / len(completed_attempts),
            'calibration_accuracy': sum(1 for a in completed_attempts if a.confidence_accurate) / len(completed_attempts)
        }

    def get_false_positive_patterns(self) -> dict:
        """
        Analyze false positive patterns.

        Returns:
            Common patterns in false positives
        """
        false_positives = [
            a for a in self.attempts
            if a.false_positive_detected or a.outcome == FixOutcome.SKIPPED.value
        ]

        if not false_positives:
            return {'total': 0, 'patterns': {}}

        # Group by issue category
        category_counts = {}
        for fp in false_positives:
            cat = fp.issue_category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Group by context type
        context_counts = {}
        for fp in false_positives:
            ctx = fp.context_type
            context_counts[ctx] = context_counts.get(ctx, 0) + 1

        return {
            'total': len(false_positives),
            'rate': len(false_positives) / len(self.attempts) if self.attempts else 0.0,
            'by_category': category_counts,
            'by_context': context_counts
        }

    def get_thompson_sampling_data(self) -> dict:
        """
        Get data formatted for Thompson Sampling (Phase 4).

        Returns:
            Alpha/Beta parameters for each strategy
        """
        strategies = [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]

        thompson_data = {}
        for strategy in strategies:
            perf = self.get_strategy_performance(strategy)

            # Thompson Sampling uses Beta(α, β) distribution
            # α = successes + 1 (Laplace smoothing)
            # β = failures + 1
            successes = perf.get('successes', 0)
            failures = perf['total_attempts'] - successes

            thompson_data[strategy.value] = {
                'alpha': successes + 1,
                'beta': failures + 1,
                'expected_reward': (successes + 1) / (perf['total_attempts'] + 2),
                'total_attempts': perf['total_attempts']
            }

        return thompson_data

    def detect_degradation(self, window_size: int = 20, threshold: float = 0.10) -> dict:
        """
        Detect if fix success rate is degrading over time.

        Args:
            window_size: Number of recent attempts to analyze
            threshold: Minimum drop in success rate to flag degradation

        Returns:
            Degradation analysis
        """
        if len(self.attempts) < window_size * 2:
            return {'degradation_detected': False, 'reason': 'Insufficient data'}

        # Recent window
        recent = self.attempts[-window_size:]
        recent_successes = sum(1 for a in recent if a.outcome == FixOutcome.SUCCESS.value)
        recent_rate = recent_successes / len(recent)

        # Historical window (before recent)
        historical = self.attempts[-(window_size * 2):-window_size]
        historical_successes = sum(1 for a in historical if a.outcome == FixOutcome.SUCCESS.value)
        historical_rate = historical_successes / len(historical)

        # Check for degradation
        drop = historical_rate - recent_rate
        degradation_detected = drop >= threshold

        return {
            'degradation_detected': degradation_detected,
            'recent_success_rate': recent_rate,
            'historical_success_rate': historical_rate,
            'drop': drop,
            'threshold': threshold,
            'window_size': window_size
        }

    def get_summary_statistics(self) -> dict:
        """Get overall summary statistics"""
        if not self.attempts:
            return {'total_attempts': 0}

        total = len(self.attempts)
        successes = sum(1 for a in self.attempts if a.outcome == FixOutcome.SUCCESS.value)
        failures = sum(1 for a in self.attempts if a.outcome == FixOutcome.FAILURE.value)
        rollbacks = sum(1 for a in self.attempts if a.outcome == FixOutcome.ROLLBACK.value)
        skipped = sum(1 for a in self.attempts if a.outcome == FixOutcome.SKIPPED.value)

        return {
            'total_attempts': total,
            'successes': successes,
            'failures': failures,
            'rollbacks': rollbacks,
            'skipped': skipped,
            'success_rate': successes / total if total > 0 else 0.0,
            'avg_confidence': sum(a.confidence for a in self.attempts) / total,
            'avg_duration_ms': sum(a.total_duration_ms for a in self.attempts) / total,
            'confidence_calibration': self.get_confidence_calibration(),
            'strategy_performance': {
                strategy.value: self.get_strategy_performance(strategy)
                for strategy in [FixStrategy.AST, FixStrategy.TEMPLATE, FixStrategy.MANUAL]
            }
        }

    def _save_attempt(self, attempt: FixAttempt):
        """Save attempt to JSONL file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(attempt.to_dict(), f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to save attempt: {e}")

    def _load_attempts(self):
        """Load attempts from JSONL file"""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.attempts.append(FixAttempt.from_dict(data))
        except Exception as e:
            print(f"Warning: Failed to load attempts: {e}")

    def export_for_analysis(self, output_file: str):
        """Export all attempts as JSON for analysis"""
        data = [a.to_dict() for a in self.attempts]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
