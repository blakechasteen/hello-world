"""
Moonshot Integration
===================

🐷 Phase 1: Complete xTerminator Core + Feedback Loop 🐷

Integrates all Phase 1 moonshot features:
1. Configurable auto-fix policies (domain-specific)
2. Feedback tracking (fix outcomes, learning signals)
3. Complete fix pipeline (propose → validate → apply → learn)

This is the orchestration layer that ties together:
- Classification Engine (Phase 1)
- AST/Template Fixers (Phases 2-3)
- Git Applicator (Phase 4)
- Validation Pipeline (Phase 5)
- AutofixPolicy (Phase 1 moonshot)
- FeedbackTracker (Phase 1 moonshot)

Usage:
    # Create moonshot-enabled orchestrator
    orchestrator = MoonshotOrchestrator(
        policy=AutofixPolicy.conservative(),  # Healthcare-grade
        enable_feedback=True
    )

    # Process an issue
    result = await orchestrator.process_issue(
        issue,
        full_code=code,
        file_path=path
    )

    # View learning statistics
    stats = orchestrator.get_learning_statistics()
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict

from .classification_engine import ClassificationEngine
from .ast_fixer import ASTFixer
from .template_fixer import TemplateFixer
from .git_applicator import GitApplicator, GitOperationResult
from .validator import ValidationPipeline, ValidationPipelineReport
from .autofix_policy import AutofixPolicy, FixDecision, PolicyProfile
from .feedback_tracker import FeedbackTracker, FixAttempt, FixOutcome
from .xterminator_types import FixProposal, FixStrategy, RiskLevel
from .thompson_bandit import ThompsonBandit, SelectionMode, create_thompson_bandit
from .confidence_calibration import ConfidenceCalibrator, create_confidence_calibrator


@dataclass
class MoonshotResult:
    """
    Complete result from moonshot-enabled processing.

    Includes classification, fix, validation, and learning signals.
    """
    # Identification
    fix_id: str
    issue_category: str
    file_path: str
    line_number: int

    # Classification
    confidence: float
    risk_level: RiskLevel
    fix_strategy: FixStrategy
    decision: FixDecision
    decision_reason: str

    # Outcome
    success: bool
    outcome: FixOutcome
    outcome_reason: str

    # Git details (if applied)
    commit_hash: Optional[str] = None
    git_message: str = ""

    # Validation report
    validation_report: Optional[ValidationPipelineReport] = None

    # Performance
    classification_duration_ms: float = 0.0
    fix_duration_ms: float = 0.0
    validation_duration_ms: float = 0.0
    total_duration_ms: float = 0.0

    # Learning signals
    feedback_recorded: bool = False
    learning_signals: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"Fix {self.fix_id}:",
            f"  File: {self.file_path}:{self.line_number}",
            f"  Category: {self.issue_category}",
            f"  Confidence: {self.confidence:.2f} ({self.risk_level.value} risk)",
            f"  Strategy: {self.fix_strategy.value}",
            f"  Decision: {self.decision.value} - {self.decision_reason}",
            f"  Outcome: {self.outcome.value} - {self.outcome_reason}",
            f"  Duration: {self.total_duration_ms:.0f}ms",
        ]

        if self.commit_hash:
            lines.append(f"  Commit: {self.commit_hash[:8]}")

        if self.validation_report and self.validation_report.all_passed:
            lines.append(f"  Validation: ✓ All checks passed")
        elif self.validation_report:
            lines.append(f"  Validation: ✗ {self.validation_report.get_failed_stages()}")

        if self.feedback_recorded:
            lines.append(f"  Feedback: ✓ Recorded for learning")

        return "\n".join(lines)


class MoonshotOrchestrator:
    """
    Orchestrates the complete fix pipeline with moonshot features.

    Enables:
    - Domain-specific auto-fix policies
    - Feedback tracking for learning
    - Complete fix pipeline (classify → fix → validate → apply → learn)
    """

    def __init__(
        self,
        policy: Optional[AutofixPolicy] = None,
        enable_feedback: bool = True,
        feedback_log: str = "./xterminator_feedback.jsonl",
        repo_path: str = ".",
        enable_thompson_sampling: bool = False,
        thompson_mode: SelectionMode = SelectionMode.THOMPSON,
        thompson_state_path: Optional[str] = None
    ):
        """
        Initialize MoonshotOrchestrator.

        Args:
            policy: AutofixPolicy (defaults to balanced)
            enable_feedback: Enable feedback tracking?
            feedback_log: Path to feedback log file
            repo_path: Path to git repository
            enable_thompson_sampling: Enable Thompson Sampling for adaptive strategy selection?
            thompson_mode: Thompson Sampling selection mode
            thompson_state_path: Path to save/load Thompson bandit state
        """
        # Policy
        self.policy = policy or AutofixPolicy.balanced()

        # Components
        self.classifier = ClassificationEngine()
        self.ast_fixer = ASTFixer()
        self.template_fixer = TemplateFixer()
        self.git_applicator = GitApplicator(repo_path=repo_path)
        self.validator = ValidationPipeline()

        # Feedback tracking
        self.enable_feedback = enable_feedback
        if enable_feedback:
            self.feedback_tracker = FeedbackTracker(log_file=feedback_log)
        else:
            self.feedback_tracker = None

        # Thompson Sampling (Phase 4)
        self.enable_thompson_sampling = enable_thompson_sampling
        if enable_thompson_sampling:
            self.thompson_bandit = create_thompson_bandit(
                mode=thompson_mode,
                state_path=Path(thompson_state_path) if thompson_state_path else None
            )
            # Confidence calibration (Phase 4)
            calibration_path = Path(thompson_state_path).parent / "calibration.json" if thompson_state_path else None
            self.confidence_calibrator = create_confidence_calibrator(
                num_bins=10,
                state_path=calibration_path
            )
        else:
            self.thompson_bandit = None
            self.confidence_calibrator = None

    async def process_issue(
        self,
        issue,  # SlopIssue from Trough
        full_code: Optional[str] = None,
        file_path: Optional[str] = None,
        apply_fix: bool = True,
        dry_run: bool = False
    ) -> MoonshotResult:
        """
        Process a single issue through complete pipeline.

        Steps:
        1. Classify issue (context, risk, strategy, confidence)
        2. Make fix decision based on policy
        3. Apply fix if decision = AUTO
        4. Validate fix (5-stage validation)
        5. Record feedback for learning

        Args:
            issue: SlopIssue from Trough
            full_code: Full file content
            file_path: Path to file
            apply_fix: Actually apply the fix? (vs. just classify)
            dry_run: If True, don't commit to git

        Returns:
            MoonshotResult with complete details
        """
        start_time = time.time()
        fix_id = str(uuid.uuid4())[:8]

        # Step 1: Classify
        classify_start = time.time()
        proposal = await self.classifier.classify_and_propose(issue, full_code, file_path)
        classification_duration = (time.time() - classify_start) * 1000

        # Step 1.5: Thompson Sampling strategy selection (Phase 4)
        original_strategy = proposal.fix_strategy
        if self.enable_thompson_sampling and self.thompson_bandit:
            # Override strategy with Thompson Sampling selection
            selected_strategy = self.thompson_bandit.select_strategy()
            proposal.fix_strategy = selected_strategy

        # Step 2: Policy decision
        decision, decision_reason = self.policy.decide(
            confidence=proposal.confidence,
            risk_level=proposal.risk_level,
            fix_strategy=proposal.fix_strategy,
            has_tests=proposal.context.has_tests
        )

        # Extract issue details
        issue_category = proposal.issue_category
        file_path_str = proposal.file_path
        line_number = proposal.line_number

        # Initialize result
        result = MoonshotResult(
            fix_id=fix_id,
            issue_category=issue_category,
            file_path=file_path_str,
            line_number=line_number,
            confidence=proposal.confidence,
            risk_level=proposal.risk_level,
            fix_strategy=proposal.fix_strategy,
            decision=decision,
            decision_reason=decision_reason,
            success=False,
            outcome=FixOutcome.PENDING,
            outcome_reason="Not processed yet",
            classification_duration_ms=classification_duration
        )

        # Step 3: Handle decision
        if decision == FixDecision.SKIP:
            result.success = True
            result.outcome = FixOutcome.SKIPPED
            result.outcome_reason = decision_reason
            result.total_duration_ms = (time.time() - start_time) * 1000

            # Record skipped fix
            if self.enable_feedback:
                await self._record_feedback(proposal, result, decision)

            return result

        elif decision in [FixDecision.MANUAL, FixDecision.REVIEW]:
            result.success = True
            result.outcome = FixOutcome.PENDING
            result.outcome_reason = f"{decision.value.upper()}: {decision_reason}"
            result.total_duration_ms = (time.time() - start_time) * 1000

            # Record for manual/review
            if self.enable_feedback:
                await self._record_feedback(proposal, result, decision)

            return result

        # Step 4: AUTO decision - Apply fix
        if not apply_fix:
            result.success = True
            result.outcome = FixOutcome.PENDING
            result.outcome_reason = "Fix approved but not applied (apply_fix=False)"
            result.total_duration_ms = (time.time() - start_time) * 1000
            return result

        # Apply fix based on strategy
        fix_start = time.time()
        fix_result = await self._apply_fix(proposal, full_code or proposal.original_code)
        fix_duration = (time.time() - fix_start) * 1000
        result.fix_duration_ms = fix_duration

        if not fix_result:
            result.success = False
            result.outcome = FixOutcome.FAILURE
            result.outcome_reason = "Fix failed to apply"
            result.total_duration_ms = (time.time() - start_time) * 1000

            if self.enable_feedback:
                await self._record_feedback(proposal, result, decision)

            return result

        fixed_code, diff = fix_result

        # Step 5: Validate
        validation_start = time.time()
        validation_report = await self.validator.validate_fix(
            original_code=full_code or proposal.original_code,
            fixed_code=fixed_code,
            file_path=file_path_str
        )
        validation_duration = (time.time() - validation_start) * 1000
        result.validation_duration_ms = validation_duration
        result.validation_report = validation_report

        # Step 6: Commit to git (if validation passed)
        should_commit = self.validator.should_commit(validation_report)

        if should_commit:
            git_result = await self.git_applicator.apply_fix(
                file_path=file_path_str,
                fixed_code=fixed_code,
                proposal=proposal,
                dry_run=dry_run
            )

            if git_result.success:
                result.success = True
                result.outcome = FixOutcome.SUCCESS
                result.outcome_reason = "Fix applied, validated, and committed"
                result.commit_hash = git_result.commit_hash
                result.git_message = git_result.message
            else:
                result.success = False
                result.outcome = FixOutcome.FAILURE
                result.outcome_reason = f"Git commit failed: {git_result.message}"
        else:
            result.success = False
            result.outcome = FixOutcome.FAILURE
            result.outcome_reason = f"Validation failed: {validation_report.get_failed_stages()}"

        result.total_duration_ms = (time.time() - start_time) * 1000

        # Step 7: Record feedback
        if self.enable_feedback:
            await self._record_feedback(proposal, result, decision)
            result.feedback_recorded = True

        # Step 8: Update Thompson Sampling (Phase 4)
        if self.enable_thompson_sampling and self.thompson_bandit:
            # Determine success
            success = result.outcome == FixOutcome.SUCCESS
            # Reward = confidence score
            reward = result.confidence

            # Update bandit
            self.thompson_bandit.update(
                strategy=proposal.fix_strategy,
                success=success,
                reward=reward,
                metadata={
                    'fix_id': fix_id,
                    'issue_category': issue_category,
                    'risk_level': result.risk_level.value,
                    'outcome': result.outcome.value
                }
            )

            # Update confidence calibration
            if self.confidence_calibrator:
                self.confidence_calibrator.add_sample(
                    confidence=result.confidence,
                    success=success
                )

        return result

    async def process_batch(
        self,
        issues: List,
        full_code: Optional[str] = None,
        file_path: Optional[str] = None,
        apply_fixes: bool = True,
        dry_run: bool = False
    ) -> List[MoonshotResult]:
        """
        Process multiple issues in parallel.

        Args:
            issues: List of SlopIssue objects
            full_code: Full file content (shared)
            file_path: Path to file (shared)
            apply_fixes: Actually apply fixes?
            dry_run: Git dry-run mode?

        Returns:
            List of MoonshotResult objects
        """
        tasks = [
            self.process_issue(issue, full_code, file_path, apply_fixes, dry_run)
            for issue in issues
        ]
        return await asyncio.gather(*tasks)

    async def _apply_fix(
        self,
        proposal: FixProposal,
        original_code: str
    ) -> Optional[tuple[str, str]]:
        """
        Apply fix using appropriate fixer.

        Returns:
            (fixed_code, diff) or None if fix failed
        """
        if proposal.fix_strategy == FixStrategy.AST:
            return await self.ast_fixer.fix_issue(proposal, original_code)
        elif proposal.fix_strategy == FixStrategy.TEMPLATE:
            return await self.template_fixer.fix_issue(proposal, original_code)
        else:
            return None

    async def _record_feedback(
        self,
        proposal: FixProposal,
        result: MoonshotResult,
        decision: FixDecision
    ):
        """Record feedback for learning"""
        if not self.feedback_tracker:
            return

        # Create attempt record
        attempt = FixAttempt(
            attempt_id=result.fix_id,
            timestamp=time.time(),
            file_path=result.file_path,
            issue_category=result.issue_category,
            line_number=result.line_number,
            confidence=result.confidence,
            risk_level=result.risk_level.value,
            fix_strategy=result.fix_strategy.value,
            context_type=proposal.context.context_type.value,
            has_tests=proposal.context.has_tests,
            scope_depth=proposal.context.scope_depth,
            decision=decision.value,
            decision_reason=result.decision_reason,
            outcome=result.outcome.value,
            outcome_reason=result.outcome_reason,
            validation_passed=result.validation_report.all_passed if result.validation_report else False,
            fix_duration_ms=result.fix_duration_ms,
            total_duration_ms=result.total_duration_ms,
            commit_sha=result.commit_hash
        )

        # Fill validation details if available
        if result.validation_report:
            attempt.syntax_passed = result.validation_report.stages['syntax'].passed
            attempt.imports_passed = result.validation_report.stages['imports'].passed
            attempt.tests_passed = result.validation_report.stages['tests'].passed
            attempt.trough_passed = result.validation_report.stages['trough'].passed
            attempt.regression_passed = result.validation_report.stages['regression'].passed
            attempt.validation_duration_ms = result.validation_duration_ms

        # Update learning signals
        if result.outcome == FixOutcome.SUCCESS:
            attempt.was_correct_strategy = True
            attempt.confidence_accurate = True
        elif result.outcome in [FixOutcome.FAILURE, FixOutcome.ROLLBACK]:
            attempt.was_correct_strategy = False
            # Was confidence too high?
            if result.confidence >= 0.85:
                attempt.confidence_accurate = False

        # Record
        self.feedback_tracker.record_attempt(attempt)

    def get_learning_statistics(self) -> dict:
        """Get learning statistics from feedback tracker"""
        if not self.feedback_tracker:
            return {'feedback_tracking': 'disabled'}

        return self.feedback_tracker.get_summary_statistics()

    def get_thompson_sampling_data(self) -> dict:
        """Get Thompson Sampling data (for Phase 4)"""
        if not self.feedback_tracker:
            return {}

        return self.feedback_tracker.get_thompson_sampling_data()

    def detect_degradation(self, window_size: int = 20, threshold: float = 0.10) -> dict:
        """Detect if system is degrading"""
        if not self.feedback_tracker:
            return {'degradation_detection': 'disabled'}

        return self.feedback_tracker.detect_degradation(window_size, threshold)

    def export_feedback(self, output_file: str):
        """Export feedback for analysis"""
        if not self.feedback_tracker:
            return

        self.feedback_tracker.export_for_analysis(output_file)

    # Thompson Sampling methods (Phase 4)

    def get_thompson_performance(self) -> dict:
        """Get Thompson Sampling performance summary"""
        if not self.thompson_bandit:
            return {'thompson_sampling': 'disabled'}

        return self.thompson_bandit.get_performance_summary()

    def get_best_strategy(self) -> tuple:
        """Get strategy with highest expected reward"""
        if not self.thompson_bandit:
            return (None, 0.0)

        return self.thompson_bandit.get_best_strategy()

    def get_learning_curve(self, window_size: int = 50) -> List[Dict]:
        """Get learning curve showing performance improvement"""
        if not self.thompson_bandit:
            return []

        return self.thompson_bandit.get_learning_curve(window_size)

    def detect_thompson_convergence(
        self,
        window_size: int = 100,
        threshold: float = 0.05
    ) -> tuple:
        """Detect if Thompson Sampling has converged"""
        if not self.thompson_bandit:
            return (False, {'reason': 'Thompson Sampling disabled'})

        return self.thompson_bandit.detect_convergence(window_size, threshold)

    def get_confidence_calibration(self) -> dict:
        """Get confidence calibration summary"""
        if not self.confidence_calibrator:
            return {'confidence_calibration': 'disabled'}

        return self.confidence_calibrator.get_calibration_summary()

    def suggest_confidence_adjustment(self, confidence: float) -> tuple:
        """Suggest adjusted confidence based on calibration"""
        if not self.confidence_calibrator:
            return (confidence, 'Calibration disabled')

        return self.confidence_calibrator.suggest_confidence_adjustment(confidence)
