#!/usr/bin/env python3
"""
🐷 COMPLETE TORUGH PIPELINE DEMONSTRATION
==========================================

"From slop to prize-winning code in 5 phases!"

This demo shows the COMPLETE xTerminator pipeline:
1. Detection (Trough) - Find the slop
2. Classification (Phase 1) - Sort the slop
3. Fixing (Phase 2/3) - Clean the slop (AST or Template)
4. Validation (Phase 5) - Check the cleaning
5. Committing (Phase 4) - Save the clean code

The Barnyard Brigade in action!
"""

import sys
import asyncio
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from trough import AISlopDetector, Language, Severity, SlopCategory
from xterminator import (
    # Phase 1: Classification
    ClassificationEngine,
    FixStrategy,

    # Phase 2 & 3: Fixing
    ASTFixer,
    TemplateFixer,

    # Phase 4: Git Integration
    GitApplicator,

    # Phase 5: Validation
    ValidationPipeline,
)


# Test code with various issues
TEST_CODE_WITH_ISSUES = '''
import os
import sys
import json  # Unused import - Templeton will notice!

def process_data(filename):
    """Process data from file"""
    # Missing error handling - Fern will catch this!
    f = open(filename)  # Should use 'with' statement
    data = json.load(f)

    # Hardcoded API key - Security issue!
    API_KEY = "sk-1234567890abcdef"

    return data

def calculate_stats(numbers):
    total = sum(numbers)
    count = len(numbers)

    # Magic number - should be a constant
    if count > 10:
        return total / count
    return 0

# Duplicate code - Wilbur hates this!
def get_user_name():
    data = load_data()
    if data:
        return data.get('name')
    return None

def get_user_email():
    data = load_data()  # Same pattern!
    if data:
        return data.get('email')
    return None
'''


class CompleteTorughDemo:
    """
    Complete TORUGH pipeline demonstration.

    "Some Pig demonstrated a pipeline!" - Wilbur
    """

    def __init__(self):
        # Phase 1: Classification
        self.classifier = ClassificationEngine()

        # Phase 2: AST Fixer (Wilbur)
        self.ast_fixer = ASTFixer()

        # Phase 3: Template Fixer (Charlotte)
        self.template_fixer = TemplateFixer()

        # Phase 4: Git Integration (Templeton) - Dry run for demo
        self.git_applicator = GitApplicator()

        # Phase 5: Validation (Fern)
        self.validator = ValidationPipeline()

        # Trough detector
        self.detector = AISlopDetector()

    async def run_complete_pipeline(self, code: str, file_path: str = "demo.py"):
        """Run the complete TORUGH pipeline on code."""

        print("=" * 70)
        print("COMPLETE TORUGH PIPELINE DEMONSTRATION")
        print("  (*)< From Slop to Prize-Winning Code!")
        print("=" * 70)
        print()

        # ==================================================================
        # PHASE 0: DETECTION (Trough)
        # ==================================================================
        print("[PHASE 0] (*)< TROUGH - Detecting Slop...")
        print("-" * 70)

        issues = await self.detector.detect_all(code, Language.PYTHON, file_path)

        print(f"Found {len(issues)} issues:")
        for i, issue in enumerate(issues[:10], 1):  # Show first 10
            print(f"  {i}. Line {issue.line_number}: {issue.category.value} ({issue.severity.value})")
            print(f"     {issue.message[:60]}...")

        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        print()

        # ==================================================================
        # Process first 3 issues through complete pipeline
        # ==================================================================

        successful_fixes = []
        failed_fixes = []

        for idx, issue in enumerate(issues[:3], 1):
            print("=" * 70)
            print(f"PROCESSING ISSUE #{idx}/{min(3, len(issues))}")
            print("=" * 70)
            print()

            # ============================================================
            # PHASE 1: CLASSIFICATION
            # ============================================================
            print("[PHASE 1] (*)<  CLASSIFICATION - Sorting the Slop...")
            print("-" * 70)
            print(f"Issue: {issue.category.value} at line {issue.line_number}")
            print(f"Message: {issue.message}")
            print()

            try:
                proposal = await self.classifier.classify_and_propose(issue, code, file_path)
            except Exception as e:
                print(f"[ERROR] Classification failed: {e}")
                failed_fixes.append({'issue': issue, 'stage': 'classification', 'error': str(e)})
                continue

            print(f"Risk: {proposal.risk_level.value}")
            print(f"Strategy: {proposal.fix_strategy.value}")
            print(f"Confidence: {proposal.confidence:.3f}")
            print(f"Safe to autofix: {proposal.safe_to_autofix}")
            print()

            if not proposal.safe_to_autofix:
                print("[SKIP] Not safe to autofix - needs manual review")
                print()
                failed_fixes.append({'issue': issue, 'stage': 'safety_check', 'reason': 'not_safe'})
                continue

            if proposal.fix_strategy == FixStrategy.SKIP:
                print("[SKIP] False positive detected")
                print()
                continue

            if proposal.fix_strategy == FixStrategy.MANUAL:
                print("[SKIP] Requires manual fix")
                print()
                failed_fixes.append({'issue': issue, 'stage': 'strategy', 'reason': 'manual_only'})
                continue

            # ============================================================
            # PHASE 2 & 3: FIXING (Wilbur or Charlotte)
            # ============================================================
            if proposal.fix_strategy == FixStrategy.AST:
                print("[PHASE 2] (*)<  WILBUR - AST Auto-Fixing...")
                fixer = self.ast_fixer
                piglet = "Wilbur"
            else:
                print("[PHASE 3] (*)<  CHARLOTTE - Template Fixing...")
                fixer = self.template_fixer
                piglet = "Charlotte"

            print("-" * 70)

            try:
                result = await fixer.fix_issue(proposal, code)
            except Exception as e:
                print(f"[ERROR] {piglet} fix failed: {e}")
                failed_fixes.append({'issue': issue, 'stage': 'fixing', 'error': str(e)})
                continue

            if not result:
                print(f"[SKIP] {piglet} couldn't fix this issue")
                print()
                failed_fixes.append({'issue': issue, 'stage': 'fixing', 'reason': 'no_fix'})
                continue

            fixed_code, diff = result
            print(f"[OK] {piglet} applied fix!")
            print()
            print("Diff preview:")
            print(diff[:500] if len(diff) > 500 else diff)
            if len(diff) > 500:
                print("... (truncated)")
            print()

            # ============================================================
            # PHASE 5: VALIDATION (Fern)
            # ============================================================
            print("[PHASE 5] (*)<  FERN - Validating Fix...")
            print("-" * 70)

            try:
                report = await self.validator.validate_fix(code, fixed_code, file_path)
            except Exception as e:
                print(f"[ERROR] Validation failed: {e}")
                failed_fixes.append({'issue': issue, 'stage': 'validation', 'error': str(e)})
                continue

            # Show validation results
            for result in report.results:
                status = "[OK]" if result.passed else "[FAIL]"
                print(f"{status} {result.stage:15s} {result.message}")

            print()
            print(f"Overall: {'PASS' if report.all_passed else 'FAIL'}")
            print()

            if not self.validator.should_commit(report):
                print("[SKIP] Validation failed - not committing")
                print()
                failed_fixes.append({'issue': issue, 'stage': 'validation', 'reason': 'failed'})
                continue

            # ============================================================
            # PHASE 4: GIT INTEGRATION (Templeton) - DRY RUN
            # ============================================================
            print("[PHASE 4] (*)<  TEMPLETON - Git Integration (DRY RUN)...")
            print("-" * 70)

            print("[DRY RUN] Would commit:")
            print(f"  File: {file_path}")
            print(f"  Category: {issue.category.value}")
            print(f"  Risk: {proposal.risk_level.value}")
            print(f"  Confidence: {proposal.confidence:.3f}")
            print()
            print(f"[DRY RUN] Commit message preview:")
            print(f"  fix({issue.category.value}): {issue.message[:50]}...")
            print()

            successful_fixes.append({
                'issue': issue,
                'proposal': proposal,
                'fixed_code': fixed_code,
                'diff': diff,
                'validation_report': report
            })

            print("[OK] Complete pipeline SUCCESS!")
            print()

        # ==================================================================
        # SUMMARY
        # ==================================================================
        print("=" * 70)
        print("TORUGH PIPELINE SUMMARY")
        print("=" * 70)
        print()
        print(f"Total Issues Detected: {len(issues)}")
        print(f"Issues Processed: {min(3, len(issues))}")
        print(f"Successful Fixes: {len(successful_fixes)}")
        print(f"Failed/Skipped: {len(failed_fixes)}")
        print()

        if successful_fixes:
            print("(*)<  Successful Fixes:")
            for i, fix in enumerate(successful_fixes, 1):
                issue = fix['issue']
                proposal = fix['proposal']
                print(f"  {i}. Line {issue.line_number}: {issue.category.value}")
                print(f"     Strategy: {proposal.fix_strategy.value}")
                print(f"     Confidence: {proposal.confidence:.3f}")
            print()

        if failed_fixes:
            print("[WARN] Failed/Skipped Fixes:")
            for i, fail in enumerate(failed_fixes, 1):
                issue = fail['issue']
                stage = fail.get('stage', 'unknown')
                reason = fail.get('reason', fail.get('error', 'unknown'))
                print(f"  {i}. Line {issue.line_number}: {issue.category.value}")
                print(f"     Failed at: {stage}")
                print(f"     Reason: {reason}")
            print()

        print("=" * 70)
        print("(*)< Pipeline demonstration complete!")
        print("     From trough to treasure in 5 phases!")
        print("=" * 70)

        return {
            'total_issues': len(issues),
            'processed': min(3, len(issues)),
            'successful': len(successful_fixes),
            'failed': len(failed_fixes),
            'successful_fixes': successful_fixes,
            'failed_fixes': failed_fixes
        }


async def main():
    """Run the complete TORUGH demonstration."""

    demo = CompleteTorughDemo()

    results = await demo.run_complete_pipeline(TEST_CODE_WITH_ISSUES, "demo_test.py")

    print()
    print("(*)<  DEMONSTRATION COMPLETE!")
    print()
    print("The Barnyard Brigade has shown you the complete pipeline:")
    print("  1. Trough detection (found the slop)")
    print("  2. Classification (sorted the slop)")
    print("  3. Fixing (cleaned the slop)")
    print("  4. Validation (checked the cleaning)")
    print("  5. Git integration (saved the clean code)")
    print()
    print("May your code be clean, your tests be green,")
    print("and your commits be atomic!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
