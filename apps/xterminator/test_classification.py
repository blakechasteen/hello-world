#!/usr/bin/env python3
"""
xTerminator Classification Engine Tests
========================================

Tests all components of the classification system.
"""

import asyncio
from dataclasses import dataclass
from xterminator import (
    ClassificationEngine,
    ContextDetector,
    RiskAssessor,
    StrategySelector,
    ConfidenceScorer,
    RiskLevel,
    FixStrategy,
    ContextType,
    CodeContext
)


# Mock SlopIssue for testing
@dataclass
class MockIssue:
    category: str
    severity: str
    message: str
    file_path: str
    line_number: int
    code_snippet: str
    suggestion: str
    confidence: float = 0.8


async def test_context_detector():
    """Test context detection"""
    print("Testing Context Detector...")
    print("-" * 60)

    detector = ContextDetector()

    # Test 1: Comment detection
    code_lines = [
        "import os",
        "# SQL query: SELECT * FROM users",  # False positive!
        "def process():",
        "    pass"
    ]

    context = detector.analyze_context(
        file_path="test.py",
        line_number=2,
        code_lines=code_lines
    )

    assert context.context_type == ContextType.COMMENT, "Should detect comment"
    assert not context.is_safe_location(), "Comment is not safe location"
    print("[OK] Comment detection works")

    # Test 2: Executable code detection
    context = detector.analyze_context(
        file_path="test.py",
        line_number=4,
        code_lines=code_lines
    )

    assert context.context_type == ContextType.EXECUTABLE, "Should detect executable"
    assert context.is_safe_location(), "Executable is safe location"
    print("[OK] Executable code detection works")

    # Test 3: Function definition detection
    context = detector.analyze_context(
        file_path="test.py",
        line_number=3,
        code_lines=code_lines
    )

    assert context.context_type == ContextType.DEFINITION, "Should detect definition"
    print("[OK] Function definition detection works")

    print()


async def test_risk_assessor():
    """Test risk assessment"""
    print("Testing Risk Assessor...")
    print("-" * 60)

    assessor = RiskAssessor()

    # Test 1: Security issue = CRITICAL
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="production/auth.py",
        line_number=100,
        has_tests=False
    )

    code_lines = ["response = requests.get(url)"] * 10

    risk, factors = assessor.assess_risk(
        issue_category='sql_injection',
        issue_severity='critical',
        context=context,
        code_lines=code_lines
    )

    assert risk == RiskLevel.CRITICAL, "Security issue should be CRITICAL"
    print(f"[OK] Security assessment: {risk.value}")
    print(f"  Factors: {factors}")

    # Test 2: Copy-paste with tests = LOW
    context.has_tests = True
    context.file_path = "test_utils.py"

    risk, factors = assessor.assess_risk(
        issue_category='copy_paste',
        issue_severity='medium',
        context=context,
        code_lines=code_lines
    )

    assert risk == RiskLevel.LOW, "Copy-paste with tests should be LOW"
    print(f"[OK] Copy-paste assessment: {risk.value}")

    # Test 3: Error handling without tests in production = CRITICAL (escalated)
    context.has_tests = False
    context.file_path = "production/api.py"

    risk, factors = assessor.assess_risk(
        issue_category='error_handling',
        issue_severity='high',
        context=context,
        code_lines=code_lines
    )

    assert risk == RiskLevel.CRITICAL, "Error handling in production without tests escalates to CRITICAL"
    print(f"[OK] Error handling assessment: {risk.value} (escalated from HIGH)")
    print(f"  Factors: {factors}")

    print()


async def test_strategy_selector():
    """Test strategy selection"""
    print("Testing Strategy Selector...")
    print("-" * 60)

    selector = StrategySelector()

    # Test 1: Copy-paste = AST
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="test.py",
        line_number=10,
        has_tests=True
    )

    strategy, reason, alternatives = selector.select_strategy(
        issue_category='copy_paste',
        risk_level=RiskLevel.LOW,
        context=context,
        confidence=0.85
    )

    assert strategy == FixStrategy.AST, "Copy-paste should use AST"
    print(f"[OK] Copy-paste strategy: {strategy.value}")
    print(f"  Reason: {reason}")

    # Test 2: Security = MANUAL
    strategy, reason, alternatives = selector.select_strategy(
        issue_category='security',
        risk_level=RiskLevel.CRITICAL,
        context=context,
        confidence=0.90
    )

    assert strategy == FixStrategy.MANUAL, "Security should be MANUAL"
    print(f"[OK] Security strategy: {strategy.value}")
    print(f"  Reason: {reason}")

    # Test 3: Comment = SKIP
    context.context_type = ContextType.COMMENT

    strategy, reason, alternatives = selector.select_strategy(
        issue_category='sql_injection',
        risk_level=RiskLevel.CRITICAL,
        context=context,
        confidence=0.95
    )

    assert strategy == FixStrategy.SKIP, "Comment should be SKIP"
    print(f"[OK] Comment strategy: {strategy.value}")
    print(f"  Reason: {reason}")

    # Test 4: Error handling = TEMPLATE
    context.context_type = ContextType.EXECUTABLE

    strategy, reason, alternatives = selector.select_strategy(
        issue_category='error_handling',
        risk_level=RiskLevel.MEDIUM,
        context=context,
        confidence=0.80
    )

    assert strategy == FixStrategy.TEMPLATE, "Error handling should use TEMPLATE"
    print(f"[OK] Error handling strategy: {strategy.value}")

    print()


async def test_confidence_scorer():
    """Test confidence scoring"""
    print("Testing Confidence Scorer...")
    print("-" * 60)

    scorer = ConfidenceScorer()

    # Test 1: High confidence scenario
    context = CodeContext(
        context_type=ContextType.EXECUTABLE,
        file_path="test_module.py",
        line_number=50,
        has_tests=True,
        scope_depth=1
    )

    confidence, breakdown = scorer.calculate_confidence(
        detection_confidence=0.90,
        issue_category='copy_paste',
        context=context,
        risk_level=RiskLevel.LOW,
        selected_strategy=FixStrategy.AST,
        code_complexity=2
    )

    assert confidence >= 0.80, "Should have high confidence"
    print(f"[OK] High confidence scenario: {confidence:.3f}")
    print(scorer.format_breakdown(breakdown))

    # Test 2: Low confidence scenario
    context.has_tests = False
    context.scope_depth = 5

    confidence, breakdown = scorer.calculate_confidence(
        detection_confidence=0.60,
        issue_category='logic_error',
        context=context,
        risk_level=RiskLevel.HIGH,
        selected_strategy=FixStrategy.MANUAL,
        code_complexity=8
    )

    assert confidence < 0.60, "Should have low confidence"
    print(f"\n[OK] Low confidence scenario: {confidence:.3f}")
    print(scorer.format_breakdown(breakdown))

    print()


async def test_classification_engine():
    """Test full classification pipeline"""
    print("Testing Classification Engine...")
    print("-" * 60)

    engine = ClassificationEngine()

    # Test 1: Auto-fixable issue (copy-paste with tests)
    issue = MockIssue(
        category='copy_paste',
        severity='medium',
        message='Duplicated code block',
        file_path='test_utils.py',
        line_number=50,
        code_snippet='def process_a():\n    result = fetch_data()\n    return result',
        suggestion='Extract common function',
        confidence=0.85
    )

    full_code = """
def process_a():
    result = fetch_data()
    return result

def process_b():
    result = fetch_data()
    return result

def test_process_a():
    assert process_a() is not None
"""

    result = await engine.classify(issue, full_code=full_code)

    print(f"Issue: {issue.category} at line {issue.line_number}")
    print(f"  Context: {result.context.context_type.value}")
    print(f"  Risk: {result.risk_level.value}")
    print(f"  Strategy: {result.selected_strategy.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  False Positive: {result.is_false_positive}")
    print(f"  Recommendation: {result.recommendation}")

    assert result.context.context_type == ContextType.EXECUTABLE
    assert result.risk_level == RiskLevel.LOW
    assert result.selected_strategy == FixStrategy.AST
    assert result.confidence >= 0.70
    print("[OK] Auto-fixable issue classified correctly")

    # Test 2: False positive (SQL in comment)
    issue2 = MockIssue(
        category='sql_injection',
        severity='critical',
        message='SQL keyword detected',
        file_path='docs.py',
        line_number=4,  # Line with comment (1-indexed)
        code_snippet='# SELECT * FROM users WHERE id = ?',
        suggestion='Use parameterized queries',
        confidence=0.75
    )

    full_code2 = """import os

# Example query:
# SELECT * FROM users WHERE id = ?

def get_user(user_id):
    # Safe implementation
    return db.query("SELECT * FROM users WHERE id = ?", user_id)
"""

    result2 = await engine.classify(issue2, full_code=full_code2)

    print(f"\nIssue: {issue2.category} at line {issue2.line_number}")
    print(f"  Context: {result2.context.context_type.value}")
    print(f"  Risk: {result2.risk_level.value}")
    print(f"  Strategy: {result2.selected_strategy.value}")
    print(f"  False Positive: {result2.is_false_positive}")
    print(f"  Reason: {result2.false_positive_reason}")

    assert result2.is_false_positive
    assert result2.selected_strategy == FixStrategy.SKIP
    print("[OK] False positive detected correctly")

    # Test 3: Batch classification
    issues = [issue, issue2]
    results = await engine.classify_batch(issues, full_code=full_code)

    stats = engine.get_statistics(results)
    print(f"\nBatch Statistics:")
    print(f"  Total: {stats['total_issues']}")
    print(f"  False positives: {stats['false_positives']} ({stats['false_positive_rate']:.1%})")
    print(f"  Auto-fixable: {stats['auto_fixable']} ({stats['auto_fixable_rate']:.1%})")
    print(f"  Needs review: {stats['needs_review']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}")

    print("[OK] Batch classification works")

    print()


async def test_fix_proposal():
    """Test fix proposal generation"""
    print("Testing Fix Proposal Generation...")
    print("-" * 60)

    engine = ClassificationEngine()

    issue = MockIssue(
        category='error_handling',
        severity='high',
        message='Network call without error handling',
        file_path='api.py',
        line_number=25,
        code_snippet='response = requests.get(url)',
        suggestion='Wrap in try/except',
        confidence=0.80
    )

    full_code = """
import requests

def fetch_data(url):
    response = requests.get(url)
    return response.json()

def test_fetch_data():
    data = fetch_data('http://test.com')
    assert data is not None
"""

    proposal = await engine.classify_and_propose(issue, full_code=full_code)

    print(f"Fix Proposal:")
    print(f"  ID: {proposal.fix_id}")
    print(f"  Category: {proposal.issue_category}")
    print(f"  Risk: {proposal.risk_level.value}")
    print(f"  Strategy: {proposal.fix_strategy.value}")
    print(f"  Confidence: {proposal.confidence:.3f}")
    print(f"  Safe to autofix: {proposal.safe_to_autofix}")
    print(f"  Requires approval: {proposal.requires_approval}")
    print(f"  Validation steps: {proposal.validation_steps}")

    hints = proposal.metadata.get('implementation_hints', [])
    if hints:
        print(f"  Implementation hints:")
        for hint in hints[:3]:
            print(f"    - {hint}")

    print(f"\n{proposal.summary()}")
    print("[OK] Fix proposal generated successfully")

    print()


async def main():
    """Run all tests"""
    print("=" * 60)
    print("xTerminator Classification Engine Test Suite")
    print("=" * 60)
    print()

    await test_context_detector()
    await test_risk_assessor()
    await test_strategy_selector()
    await test_confidence_scorer()
    await test_classification_engine()
    await test_fix_proposal()

    print("=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
