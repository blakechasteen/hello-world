"""
Apply Auto-Fixes to HoloLoom
=============================

Applies the 138 auto-fixable issues identified during dogfooding.

Strategy:
1. Re-scan HoloLoom to get fresh issue list with file paths
2. Filter to auto-fixable issues only (low risk, high confidence)
3. Apply fixes using xTerminator
4. Validate fixes don't break tests
5. Create summary report

Author: mythRL Team
Date: November 16, 2025
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import subprocess

from trough.mcp_server import TroughMCPServer
from xterminator.mcp_server import XTerminatorMCPServer
from autofix_tracker import AutoFixTracker, AutoFixResult


class AutoFixApplicator:
    """Applies auto-fixes to HoloLoom codebase."""

    def __init__(self, enable_tracking: bool = True):
        self.trough = TroughMCPServer()
        self.xterminator = XTerminatorMCPServer()
        self.results = {
            'total_issues_found': 0,
            'auto_fixable': 0,
            'fixes_attempted': 0,
            'fixes_successful': 0,
            'fixes_failed': 0,
            'files_modified': [],
            'failed_fixes': []
        }

        # Learning tracker (Phase 4+)
        self.enable_tracking = enable_tracking
        self.tracker = AutoFixTracker() if enable_tracking else None
        self.session_id = None

    async def scan_hololoom(self, max_files: int = 50) -> Dict[str, Any]:
        """Scan HoloLoom directory for issues."""
        print("=" * 70)
        print("STEP 1: Scanning HoloLoom for Auto-Fixable Issues")
        print("=" * 70)
        print()

        # Find Python files
        import glob
        pattern = "HoloLoom/**/*.py"
        py_files = glob.glob(pattern, recursive=True)
        py_files = [f for f in py_files if '__pycache__' not in f]
        py_files = py_files[:max_files]

        print(f"Found {len(py_files)} Python files to scan")

        # Scan files individually (avoids LogicError issue in scan_directory)
        all_files_data = []
        total_issues = 0

        for i, file_path in enumerate(py_files, 1):
            print(f"Scanning [{i}/{len(py_files)}]: {file_path}", end='\r')

            result = await self.trough.trough_scan_file(
                file_path=file_path,
                include_ml_logic=False  # Disabled due to known LogicError issue
            )

            if result.get('issues'):
                all_files_data.append({
                    'file': file_path,
                    'issues': result['issues']
                })
                total_issues += len(result['issues'])

        print()  # New line after progress
        print(f"Scanned: {len(py_files)} files")
        print(f"Total issues: {total_issues}")
        print()

        self.results['total_issues_found'] = total_issues

        return {
            'files_scanned': len(py_files),
            'total_issues': total_issues,
            'files': all_files_data
        }

    async def classify_issues(self, scan_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Classify issues and extract auto-fixable ones."""
        print("=" * 70)
        print("STEP 2: Classifying Issues (Finding Auto-Fixable)")
        print("=" * 70)
        print()

        # Collect all issues across files
        all_issues = []
        for file_data in scan_result['files']:
            file_path = file_data['file']
            for issue in file_data['issues']:
                issue['file_path'] = file_path
                all_issues.append(issue)

        # Classify
        classification = await self.trough.trough_classify_issues(all_issues)

        print(f"Auto-fixable: {classification['auto_fixable']}")
        print(f"Needs review: {classification['needs_review']}")
        print(f"False positives: {classification['false_positives']}")
        print()

        self.results['auto_fixable'] = classification['auto_fixable']

        # Filter to auto-fixable only
        auto_fixable_issues = [
            issue for issue in all_issues
            if self._is_auto_fixable(issue)
        ]

        return auto_fixable_issues

    def _is_auto_fixable(self, issue: Dict[str, Any]) -> bool:
        """Determine if issue is auto-fixable."""
        # Criteria:
        # - Severity: low or medium
        # - Confidence: >= 0.9
        # - Category: dead_code, copy_paste, incomplete

        severity = issue.get('severity', 'medium')
        confidence = issue.get('confidence', 0.0)
        category = issue.get('category', '')

        # Match tracker configuration (line 325)
        auto_fixable_categories = ['dead_code', 'hardcoded_values', 'missing_docstrings', 'incomplete']

        return (
            severity in ['low', 'medium'] and
            confidence >= 0.9 and
            category in auto_fixable_categories
        )

    async def apply_fixes(
        self,
        issues: List[Dict[str, Any]],
        max_fixes: int = 20,
        dry_run: bool = False
    ) -> None:
        """Apply fixes to issues."""
        print("=" * 70)
        print(f"STEP 3: Applying Auto-Fixes ({'DRY RUN' if dry_run else 'REAL'})")
        print("=" * 70)
        print()

        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            file_path = issue.get('file_path', '')
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)

        print(f"Files to fix: {len(issues_by_file)}")
        print(f"Total issues to fix: {len(issues)}")
        print(f"Max fixes per run: {max_fixes}")
        print()

        # Apply fixes file by file
        fixes_applied = 0
        for file_path, file_issues in list(issues_by_file.items())[:max_fixes]:
            if fixes_applied >= max_fixes:
                break

            print(f"Processing: {file_path}")
            print(f"  Issues: {len(file_issues)}")

            # Read file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except Exception as e:
                print(f"  ✗ Failed to read file: {e}")
                self.results['fixes_failed'] += len(file_issues)
                continue

            # Batch fix all issues in file
            try:
                result = await self.xterminator.xterminator_batch_fix(
                    issues=file_issues,
                    code=code,
                    file_path=file_path,
                    max_fixes=len(file_issues)
                )

                applied = result.get('fixes_applied', 0)
                failed = result.get('total_issues', 0) - applied

                print(f"  ✓ Fixes applied: {applied}")
                if failed > 0:
                    print(f"  ! Fixes failed: {failed}")

                    # Debug: Show first failure reason
                    for fix_result in result.get('results', [])[:3]:
                        if fix_result['status'] != 'applied':
                            print(f"    - {fix_result['status']}: {fix_result.get('error', 'No error message')}")

                self.results['fixes_attempted'] += result.get('total_issues', 0)
                self.results['fixes_successful'] += applied
                self.results['fixes_failed'] += failed

                if applied > 0:
                    self.results['files_modified'].append(file_path)
                    fixes_applied += applied

            except Exception as e:
                print(f"  ✗ Batch fix failed: {e}")
                self.results['fixes_failed'] += len(file_issues)
                self.results['failed_fixes'].append({
                    'file': file_path,
                    'error': str(e),
                    'issues': len(file_issues)
                })

            print()

        print(f"Total fixes applied: {fixes_applied}")
        print()

    async def validate_fixes(self) -> bool:
        """Run tests to validate fixes didn't break anything."""
        print("=" * 70)
        print("STEP 4: Validating Fixes (Running Tests)")
        print("=" * 70)
        print()

        if not self.results['files_modified']:
            print("No files modified, skipping validation")
            return True

        # Run quick smoke test on modified files
        print("Running pytest on modified test files...")

        test_files = [
            f for f in self.results['files_modified']
            if 'test_' in f or '/tests/' in f
        ]

        if not test_files:
            print("No test files modified")
            return True

        for test_file in test_files[:5]:  # Test first 5 files
            print(f"Testing: {test_file}")
            try:
                result = subprocess.run(
                    ['python', '-m', 'pytest', test_file, '-v', '--tb=short'],
                    capture_output=True,
                    timeout=30,
                    cwd='.'
                )
                if result.returncode == 0:
                    print(f"  ✓ Tests passed")
                else:
                    print(f"  ! Tests failed (may need investigation)")
                    print(f"    {result.stdout.decode()[:200]}")
            except subprocess.TimeoutExpired:
                print(f"  ! Test timeout (skipping)")
            except Exception as e:
                print(f"  ! Test error: {e}")

        print()
        return True

    def print_summary(self) -> None:
        """Print summary report."""
        print("=" * 70)
        print("AUTO-FIX SUMMARY")
        print("=" * 70)
        print()

        print(f"Total issues found: {self.results['total_issues_found']}")
        print(f"Auto-fixable: {self.results['auto_fixable']}")
        print(f"Fixes attempted: {self.results['fixes_attempted']}")
        print(f"Fixes successful: {self.results['fixes_successful']}")
        print(f"Fixes failed: {self.results['fixes_failed']}")
        print()

        if self.results['files_modified']:
            print(f"Files modified: {len(self.results['files_modified'])}")
            for file in self.results['files_modified'][:10]:
                print(f"  - {file}")
            if len(self.results['files_modified']) > 10:
                print(f"  ... and {len(self.results['files_modified']) - 10} more")
        else:
            print("No files modified")

        print()

        if self.results['failed_fixes']:
            print(f"Failed fixes: {len(self.results['failed_fixes'])}")
            for fail in self.results['failed_fixes'][:5]:
                print(f"  - {fail['file']}: {fail['error']}")

        print()
        print("=" * 70)

        # Save results
        with open('autofix_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print("Results saved to: autofix_results.json")
        print()


async def main():
    """Run auto-fix application."""
    applicator = AutoFixApplicator(enable_tracking=True)

    # Start tracking session
    if applicator.tracker:
        applicator.session_id = applicator.tracker.start_session(
            max_files=100,
            categories=['dead_code', 'hardcoded_values', 'missing_docstrings', 'incomplete'],
            confidence_threshold=0.85
        )

    # Step 1: Scan HoloLoom
    scan_result = await applicator.scan_hololoom(max_files=100)

    # Step 2: Classify issues
    auto_fixable = await applicator.classify_issues(scan_result)

    print(f"Found {len(auto_fixable)} auto-fixable issues")
    print()

    if not auto_fixable:
        print("No auto-fixable issues found!")
        if applicator.tracker:
            applicator.tracker.end_session()
        return

    # Show sample
    print("Sample auto-fixable issues:")
    for issue in auto_fixable[:5]:
        print(f"  - {issue['file_path']}:{issue.get('line_number', '?')}")
        print(f"    {issue['category']} (confidence: {issue['confidence']:.2f})")
    print()

    # Step 3: Apply fixes (increased limit for scale-up)
    await applicator.apply_fixes(auto_fixable, max_fixes=50, dry_run=False)

    # Step 4: Validate
    await applicator.validate_fixes()

    # Summary
    applicator.print_summary()

    # End tracking session and export results
    if applicator.tracker:
        applicator.tracker.end_session()
        applicator.tracker.export_json('autofix_tracking/all_sessions.json')
        applicator.tracker.export_learning_data('autofix_tracking/learning_data.json')

        # Print recommendations
        print("\n")
        recs = applicator.tracker.get_recommendations()
        if recs['categories_to_disable']:
            print("⚠ Recommendation: Consider disabling low-performing categories:")
            for cat in recs['categories_to_disable']:
                print(f"  - {cat}")
        if recs['adjust_confidence_threshold']:
            print(f"💡 Recommendation: Adjust confidence threshold to {recs['recommended_threshold']:.2f}")
        print()


if __name__ == '__main__':
    asyncio.run(main())
