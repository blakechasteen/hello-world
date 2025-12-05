#!/usr/bin/env python3
"""
xTerminator Moonshot Demo - Phase 1
====================================

🐷 "Some Pig Built a Learning System!" 🐷

Demonstrates Phase 1 moonshot features:
1. Domain-specific policies (healthcare strict, beekeeping relaxed)
2. Feedback tracking (fix outcomes, learning signals)
3. Complete fix pipeline (classify → fix → validate → apply → learn)
4. Thompson Sampling data collection (prep for Phase 4)
5. Degradation detection (prep for Phase 7)

The Demo:
- Process same issues with 3 different policies
- Track fix outcomes and learning signals
- Show how policies affect auto-fix decisions
- Display learning statistics
- Export Thompson Sampling data

Run:
    python xterminator/demo_moonshot_phase1.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from xterminator.moonshot_integration import MoonshotOrchestrator, MoonshotResult
from xterminator.autofix_policy import AutofixPolicy, PolicyProfile
from xterminator.feedback_tracker import FeedbackTracker, FixOutcome
from xterminator.xterminator_types import RiskLevel, FixStrategy

# Mock SlopIssue class (normally from Trough)
class MockSlopIssue:
    def __init__(self, category, severity, line_number, code_snippet, file_path="demo.py"):
        self.category = category
        self.severity = severity
        self.line_number = line_number
        self.code_snippet = code_snippet
        self.file_path = file_path
        self.confidence = 0.85


def print_banner(text: str):
    """Print a pretty banner"""
    border = "=" * 70
    print(f"\n{border}")
    print(f"{text:^70}")
    print(f"{border}\n")


def print_result(result: MoonshotResult):
    """Print a result summary"""
    # Decision icon
    decision_icons = {
        'auto': '🤖 AUTO',
        'review': '👀 REVIEW',
        'manual': '🔧 MANUAL',
        'skip': '⏭️  SKIP'
    }
    decision_icon = decision_icons.get(result.decision.value, result.decision.value.upper())

    # Outcome icon
    outcome_icons = {
        'success': '✅',
        'failure': '❌',
        'skipped': '⏭️',
        'pending': '⏳',
        'rollback': '↩️'
    }
    outcome_icon = outcome_icons.get(result.outcome.value, '?')

    print(f"  {decision_icon:12} | {result.issue_category:20} | {result.confidence:.2f} conf | {result.risk_level.value:8} risk")
    print(f"  {outcome_icon} {result.outcome.value:10} | {result.outcome_reason}")
    print(f"  ⏱️  {result.total_duration_ms:.0f}ms\n")


async def demo_scenario_1_policy_comparison():
    """
    Scenario 1: Policy Comparison

    Same issues, different policies → different decisions
    """
    print_banner("🐷 SCENARIO 1: Policy Comparison 🐷")
    print("Same issues processed with 3 different policies:")
    print("  CONSERVATIVE (Healthcare) - Very strict, 95% min confidence")
    print("  BALANCED (Default) - Reasonable, 85% min confidence")
    print("  AGGRESSIVE (Beekeeping) - Relaxed, 70% min confidence\n")

    # Create test issues
    issues = [
        MockSlopIssue("unused_import", "LOW", 5, "import os", "demo.py"),
        MockSlopIssue("error_handling", "MEDIUM", 12, "result = risky_call()", "demo.py"),
        MockSlopIssue("dead_code", "LOW", 20, "return x\nprint('never')", "demo.py"),
    ]

    # Test with each policy
    policies = {
        'CONSERVATIVE (Healthcare)': AutofixPolicy.conservative(domain='healthcare'),
        'BALANCED (Default)': AutofixPolicy.balanced(domain='general'),
        'AGGRESSIVE (Beekeeping)': AutofixPolicy.aggressive(domain='beekeeping')
    }

    for policy_name, policy in policies.items():
        print(f"\n{'─' * 70}")
        print(f"Policy: {policy_name}")
        print(f"{'─' * 70}")

        orchestrator = MoonshotOrchestrator(
            policy=policy,
            enable_feedback=True,
            feedback_log=f"./demo_feedback_{policy.profile.value}.jsonl"
        )

        for issue in issues:
            # Mock classification result
            from xterminator.xterminator_types import FixProposal, CodeContext, ContextType

            context = CodeContext(
                context_type=ContextType.EXECUTABLE,
                has_tests=True,
                scope_depth=1
            )

            proposal = FixProposal(
                fix_id=f"demo_{issue.category}",
                issue_category=issue.category,
                risk_level=RiskLevel.LOW if issue.severity == "LOW" else RiskLevel.MEDIUM,
                confidence=0.87,
                fix_strategy=FixStrategy.AST,
                file_path=issue.file_path,
                line_number=issue.line_number,
                original_code=issue.code_snippet,
                context=context,
                explanation=f"Demo fix for {issue.category}",
                implementation_notes="Demo only",
                metadata={}
            )

            # Make decision
            decision, reason = policy.decide(
                confidence=proposal.confidence,
                risk_level=proposal.risk_level,
                fix_strategy=proposal.fix_strategy,
                has_tests=proposal.context.has_tests
            )

            # Create mock result
            result = MoonshotResult(
                fix_id=proposal.fix_id,
                issue_category=proposal.issue_category,
                file_path=proposal.file_path,
                line_number=proposal.line_number,
                confidence=proposal.confidence,
                risk_level=proposal.risk_level,
                fix_strategy=proposal.fix_strategy,
                decision=decision,
                decision_reason=reason,
                success=(decision.value == 'auto'),
                outcome=FixOutcome.SUCCESS if decision.value == 'auto' else FixOutcome.PENDING,
                outcome_reason=reason,
                total_duration_ms=5.0
            )

            print_result(result)


async def demo_scenario_2_feedback_tracking():
    """
    Scenario 2: Feedback Tracking

    Show how fix outcomes are tracked for learning
    """
    print_banner("🐷 SCENARIO 2: Feedback Tracking 🐷")
    print("Simulating multiple fix attempts and tracking outcomes...\n")

    tracker = FeedbackTracker(log_file="./demo_feedback_tracking.jsonl")

    # Simulate fix attempts with different outcomes
    from xterminator.feedback_tracker import FixAttempt
    import time

    attempts_data = [
        # (category, confidence, risk, strategy, outcome, success)
        ("unused_import", 0.95, "LOW", "AST", FixOutcome.SUCCESS, True),
        ("dead_code", 0.92, "LOW", "AST", FixOutcome.SUCCESS, True),
        ("error_handling", 0.88, "MEDIUM", "TEMPLATE", FixOutcome.SUCCESS, True),
        ("hardcoded_values", 0.85, "LOW", "TEMPLATE", FixOutcome.SUCCESS, True),
        ("poor_naming", 0.75, "LOW", "AST", FixOutcome.FAILURE, False),
        ("resource_management", 0.90, "MEDIUM", "TEMPLATE", FixOutcome.SUCCESS, True),
        ("copy_paste", 0.88, "LOW", "AST", FixOutcome.SUCCESS, True),
        ("error_handling", 0.82, "MEDIUM", "TEMPLATE", FixOutcome.PARTIAL, True),
        ("dead_code", 0.94, "LOW", "AST", FixOutcome.SUCCESS, True),
        ("unused_import", 0.97, "LOW", "AST", FixOutcome.SUCCESS, True),
    ]

    for i, (category, conf, risk, strategy, outcome, success) in enumerate(attempts_data):
        attempt = FixAttempt(
            attempt_id=f"demo_{i}",
            timestamp=time.time(),
            file_path="demo.py",
            issue_category=category,
            line_number=10 + i,
            confidence=conf,
            risk_level=risk,
            fix_strategy=strategy,
            context_type="EXECUTABLE",
            has_tests=True,
            scope_depth=1,
            decision="AUTO",
            decision_reason="Demo attempt",
            outcome=outcome.value,
            outcome_reason="Simulated outcome",
            validation_passed=success,
            fix_duration_ms=50.0,
            total_duration_ms=150.0,
            syntax_passed=success,
            imports_passed=success,
            tests_passed=success,
            trough_passed=success,
            regression_passed=success,
            was_correct_strategy=success,
            confidence_accurate=success
        )
        tracker.record_attempt(attempt)
        print(f"  ✓ Recorded: {category:20} | {conf:.2f} conf | {outcome.value:10}")

    # Show statistics
    print(f"\n{'─' * 70}")
    print("Learning Statistics:")
    print(f"{'─' * 70}\n")

    stats = tracker.get_summary_statistics()
    print(f"Total Attempts: {stats['total_attempts']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Confidence: {stats['avg_confidence']:.2f}")
    print(f"Avg Duration: {stats['avg_duration_ms']:.0f}ms\n")

    # Strategy performance
    print("Strategy Performance:")
    for strategy_name, perf in stats['strategy_performance'].items():
        if perf['total_attempts'] > 0:
            print(f"  {strategy_name:10} | {perf['total_attempts']} attempts | "
                  f"{perf['success_rate']:.1%} success | {perf['avg_confidence']:.2f} avg conf")

    # Confidence calibration
    print(f"\nConfidence Calibration:")
    calib = stats['confidence_calibration']
    print(f"  Accuracy: {calib['calibration_accuracy']:.1%}")
    print(f"  Overconfident: {calib['overconfident_rate']:.1%}")
    print(f"  Underconfident: {calib['underconfident_rate']:.1%}")


async def demo_scenario_3_thompson_sampling():
    """
    Scenario 3: Thompson Sampling Data

    Show data ready for adaptive strategy selection (Phase 4)
    """
    print_banner("🐷 SCENARIO 3: Thompson Sampling Data (Phase 4 Prep) 🐷")
    print("Data collected for adaptive strategy selection...\n")

    # Reuse tracker from scenario 2
    tracker = FeedbackTracker(log_file="./demo_feedback_tracking.jsonl")

    thompson_data = tracker.get_thompson_sampling_data()

    print("Thompson Sampling Parameters (Beta Distribution):")
    print("  Strategy     | α (successes) | β (failures) | Expected Reward | Attempts")
    print("  " + "─" * 75)

    for strategy, params in thompson_data.items():
        alpha = params['alpha']
        beta = params['beta']
        reward = params['expected_reward']
        attempts = params['total_attempts']
        print(f"  {strategy:12} | {alpha:13.0f} | {beta:12.0f} | {reward:15.2%} | {attempts:8}")

    print("\nInterpretation:")
    print("  - Higher Expected Reward → Better strategy")
    print("  - Phase 4 will use Thompson Sampling to adaptively select strategies")
    print("  - System learns which strategies work best per issue category")


async def demo_scenario_4_degradation_detection():
    """
    Scenario 4: Degradation Detection

    Show live monitoring capability (prep for Phase 7)
    """
    print_banner("🐷 SCENARIO 4: Degradation Detection (Phase 7 Prep) 🐷")
    print("Detecting if fix success rate is degrading over time...\n")

    tracker = FeedbackTracker(log_file="./demo_feedback_tracking.jsonl")

    degradation = tracker.detect_degradation(window_size=5, threshold=0.10)

    print(f"Degradation Analysis:")
    print(f"  Historical Success Rate: {degradation['historical_success_rate']:.1%}")
    print(f"  Recent Success Rate: {degradation['recent_success_rate']:.1%}")
    print(f"  Drop: {degradation['drop']:.1%} (threshold: {degradation['threshold']:.1%})")
    print(f"  Degradation Detected: {'⚠️  YES' if degradation['degradation_detected'] else '✓ NO'}")

    print("\nInterpretation:")
    print("  - Phase 7 will monitor departments for degradation")
    print("  - Alert if success rate drops significantly")
    print("  - Enables proactive quality assurance")


async def main():
    """Run all demo scenarios"""
    print("\n")
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)
    print("        xTerminator Moonshot Demo - Phase 1".center(70))
    print("      \"From Pig Trough to Prize-Winning Learning!\"".center(70))
    print("=" * 70)
    print("🐷" * 35)
    print("=" * 70)

    await demo_scenario_1_policy_comparison()
    await demo_scenario_2_feedback_tracking()
    await demo_scenario_3_thompson_sampling()
    await demo_scenario_4_degradation_detection()

    print_banner("🐷 Phase 1 Complete! 🐷")
    print("Next steps:")
    print("  - Phase 2 (Weeks 3-4): Department Protocol")
    print("  - Phase 3 (Weeks 5-7): Orchestration Integration")
    print("  - Phase 4 (Weeks 8-10): Institutional Learning (Thompson Sampling)")
    print("  - Phase 5 (Weeks 11-18): Advanced Features (Marketplace, Monitoring)")
    print("\n(*)<  OINK OINK OINK!")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
