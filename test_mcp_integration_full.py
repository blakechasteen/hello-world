"""
MCP Integration Test Suite - Full Cross-Department Workflow
===========================================================

Tests the complete integration of Trough and xTerminator MCP servers
with HoloLoom's cross-department architecture.

Test Coverage:
1. Trough MCP Server (4 tools)
2. xTerminator MCP Server (5 tools)
3. Cross-department workflow (QA → Verification → QA)
4. Session-aware state management

Author: mythRL Team
Date: November 16, 2025
"""

import asyncio
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import json

# Import MCP servers
from trough.mcp_server import TroughMCPServer
from xterminator.mcp_server import XTerminatorMCPServer


# Test code samples
PYTHON_CODE_WITH_ISSUES = '''
def process_data(data):
    """Process the data."""
    # TODO: implement this
    pass

def calculate_average(numbers):
    """Calculate average."""
    if len(numbers) == 0:
        return 0
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)

# Debug code that should be removed
print("Debug: Starting processing")
x = 42  # Unused variable
'''

TYPESCRIPT_CODE_WITH_ISSUES = '''
function fetchData(url: string) {
    fetch(url).then(response => response.json());  // No .catch()!
}

const API_KEY = "sk-1234567890abcdefghijklmnopqrstuvwxyz";  // Hardcoded!

function processUser(user: any) {  // Using 'any'!
    console.log(user.name);
}

var legacyVar = 42;  // Using 'var'
'''


class MCPIntegrationTester:
    """Comprehensive MCP integration tester."""

    def __init__(self):
        self.trough_server = TroughMCPServer()
        self.xterminator_server = XTerminatorMCPServer()
        self.test_results = []
        self.session_id = "test_session_001"

    async def run_all_tests(self):
        """Run complete test suite."""
        print("=" * 70)
        print("MCP INTEGRATION TEST SUITE")
        print("=" * 70)
        print()

        # Test 1: Trough MCP Server
        await self._test_trough_scan_file()
        await self._test_trough_scan_directory()
        await self._test_trough_classify_issues()
        await self._test_trough_metrics()

        # Test 2: xTerminator MCP Server
        await self._test_xterminator_propose_fix()
        await self._test_xterminator_validate_fix()
        await self._test_xterminator_batch_fix()

        # Test 3: Cross-Department Workflow
        await self._test_qa_verification_loop()

        # Test 4: Session-Aware State
        await self._test_session_state()

        # Summary
        self._print_summary()

    # ===================================================================
    # TROUGH MCP SERVER TESTS
    # ===================================================================

    async def _test_trough_scan_file(self):
        """Test 1.1: Scan single file for issues."""
        print("Test 1.1: Trough - Scan Single File")
        print("-" * 70)

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(PYTHON_CODE_WITH_ISSUES)
                temp_path = f.name

            # Scan file
            result = await self.trough_server.trough_scan_file(
                file_path=temp_path,
                include_ml_logic=False  # Disabled for stability
            )

            # Validate
            assert 'issues' in result, "Missing 'issues' field"
            assert 'file_path' in result, "Missing 'file_path' field"
            assert len(result['issues']) > 0, "Should detect issues"

            print(f"✓ Detected {len(result['issues'])} issues")
            print(f"  Categories: {set(i['category'] for i in result['issues'])}")

            self.test_results.append({
                'test': 'trough_scan_file',
                'status': 'PASS',
                'issues_detected': len(result['issues'])
            })

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'trough_scan_file',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    async def _test_trough_scan_directory(self):
        """Test 1.2: Scan directory for issues."""
        print("Test 1.2: Trough - Scan Directory")
        print("-" * 70)

        try:
            # Create temp directory with multiple files
            with tempfile.TemporaryDirectory() as tmpdir:
                # Python file
                py_file = Path(tmpdir) / "test.py"
                py_file.write_text(PYTHON_CODE_WITH_ISSUES)

                # TypeScript file
                ts_file = Path(tmpdir) / "test.ts"
                ts_file.write_text(TYPESCRIPT_CODE_WITH_ISSUES)

                # Scan directory
                result = await self.trough_server.trough_scan_directory(
                    directory=tmpdir,
                    file_pattern="**/*.py",
                    max_files=10
                )

                # Validate
                assert 'files_scanned' in result, "Missing 'files_scanned'"
                assert 'total_issues' in result, "Missing 'total_issues'"
                assert result['files_scanned'] > 0, "Should scan files"

                print(f"✓ Scanned {result['files_scanned']} files")
                print(f"  Total issues: {result['total_issues']}")
                print(f"  Files with issues: {len(result['files'])}")

                self.test_results.append({
                    'test': 'trough_scan_directory',
                    'status': 'PASS',
                    'files_scanned': result['files_scanned'],
                    'issues_detected': result['total_issues']
                })

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'trough_scan_directory',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    async def _test_trough_classify_issues(self):
        """Test 1.3: Classify issues by fixability."""
        print("Test 1.3: Trough - Classify Issues")
        print("-" * 70)

        try:
            # Create temp file and scan
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(PYTHON_CODE_WITH_ISSUES)
                temp_path = f.name

            # Scan file
            scan_result = await self.trough_server.trough_scan_file(
                file_path=temp_path,
                include_ml_logic=False
            )

            # Classify issues
            classify_result = await self.trough_server.trough_classify_issues(
                issues=scan_result['issues']
            )

            # Validate
            assert 'auto_fixable' in classify_result
            assert 'needs_review' in classify_result
            assert 'false_positives' in classify_result

            total = (
                classify_result['auto_fixable'] +
                classify_result['needs_review'] +
                classify_result['false_positives']
            )
            assert total == len(scan_result['issues'])

            print(f"✓ Classified {total} issues:")
            print(f"  Auto-fixable: {classify_result['auto_fixable']}")
            print(f"  Needs review: {classify_result['needs_review']}")
            print(f"  False positives: {classify_result['false_positives']}")

            self.test_results.append({
                'test': 'trough_classify_issues',
                'status': 'PASS',
                'classification': classify_result
            })

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'trough_classify_issues',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    async def _test_trough_metrics(self):
        """Test 1.4: Get quality metrics."""
        print("Test 1.4: Trough - Get Metrics")
        print("-" * 70)

        try:
            # Get metrics
            metrics = await self.trough_server.trough_get_metrics()

            # Validate
            assert 'total_scans' in metrics
            assert 'total_issues_detected' in metrics
            assert 'session_id' in metrics

            print(f"✓ Retrieved metrics:")
            print(f"  Total scans: {metrics['total_scans']}")
            print(f"  Issues detected: {metrics['total_issues_detected']}")
            print(f"  Session: {metrics['session_id']}")

            self.test_results.append({
                'test': 'trough_get_metrics',
                'status': 'PASS',
                'metrics': metrics
            })

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'trough_get_metrics',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    # ===================================================================
    # xTERMINATOR MCP SERVER TESTS
    # ===================================================================

    async def _test_xterminator_propose_fix(self):
        """Test 2.1: Propose fix for issue."""
        print("Test 2.1: xTerminator - Propose Fix")
        print("-" * 70)

        try:
            # Create issue (use lowercase enum values!)
            issue = {
                'category': 'dead_code',
                'severity': 'medium',
                'message': 'Unused variable',
                'line_number': 13,
                'code_snippet': 'x = 42  # Unused variable',
                'suggestion': 'Remove unused variable',
                'confidence': 0.9,
                'file_path': 'test.py'
            }

            # Propose fix
            result = await self.xterminator_server.xterminator_propose_fix(
                issue=issue,
                code=PYTHON_CODE_WITH_ISSUES,
                file_path='test.py',
                safety_level='balanced'
            )

            # Validate
            assert 'proposal' in result
            assert 'fix_id' in result
            assert result['proposal']['fix_strategy'] in ['ast', 'template', 'manual']

            print(f"✓ Generated fix proposal:")
            print(f"  Fix ID: {result['fix_id']}")
            print(f"  Strategy: {result['proposal']['fix_strategy']}")
            print(f"  Risk: {result['proposal']['risk_level']}")

            self.test_results.append({
                'test': 'xterminator_propose_fix',
                'status': 'PASS',
                'fix_id': result['fix_id']
            })

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'xterminator_propose_fix',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    async def _test_xterminator_validate_fix(self):
        """Test 2.2: Validate fix proposal."""
        print("Test 2.2: xTerminator - Validate Fix")
        print("-" * 70)

        try:
            # Create issue and proposal (use lowercase enum values!)
            issue = {
                'category': 'dead_code',
                'severity': 'medium',
                'message': 'Unused variable',
                'line_number': 13,
                'code_snippet': 'x = 42',
                'suggestion': 'Remove unused variable',
                'confidence': 0.9,
                'file_path': 'test.py'
            }

            proposal_result = await self.xterminator_server.xterminator_propose_fix(
                issue=issue,
                code=PYTHON_CODE_WITH_ISSUES,
                file_path='test.py'
            )

            # Validate fix
            validation_result = await self.xterminator_server.xterminator_validate_fix(
                fix_id=proposal_result['fix_id']
            )

            # Validate
            assert 'validation_passed' in validation_result
            assert 'steps' in validation_result

            print(f"✓ Validation completed:")
            print(f"  Passed: {validation_result['validation_passed']}")
            print(f"  Steps run: {len(validation_result['steps'])}")

            self.test_results.append({
                'test': 'xterminator_validate_fix',
                'status': 'PASS',
                'validation_passed': validation_result['validation_passed']
            })

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'xterminator_validate_fix',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    async def _test_xterminator_batch_fix(self):
        """Test 2.3: Batch fix multiple issues."""
        print("Test 2.3: xTerminator - Batch Fix")
        print("-" * 70)

        try:
            # Create multiple issues (use lowercase enum values!)
            issues = [
                {
                    'category': 'dead_code',
                    'severity': 'medium',
                    'message': 'Unused variable',
                    'line_number': 13,
                    'code_snippet': 'x = 42',
                    'suggestion': 'Remove',
                    'confidence': 0.9,
                    'file_path': 'test.py'
                },
                {
                    'category': 'incomplete',  # Using incomplete for debug code
                    'severity': 'low',
                    'message': 'Debug print',
                    'line_number': 12,
                    'code_snippet': 'print("Debug: Starting")',
                    'suggestion': 'Remove',
                    'confidence': 0.95,
                    'file_path': 'test.py'
                }
            ]

            # Batch fix
            result = await self.xterminator_server.xterminator_batch_fix(
                issues=issues,
                code=PYTHON_CODE_WITH_ISSUES,
                file_path='test.py',
                max_fixes=5
            )

            # Validate
            assert 'total_issues' in result
            assert 'fixes_applied' in result

            print(f"✓ Batch fix completed:")
            print(f"  Total issues: {result['total_issues']}")
            print(f"  Fixes applied: {result['fixes_applied']}")

            self.test_results.append({
                'test': 'xterminator_batch_fix',
                'status': 'PASS',
                'fixes_applied': result['fixes_applied']
            })

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'xterminator_batch_fix',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    # ===================================================================
    # CROSS-DEPARTMENT WORKFLOW TESTS
    # ===================================================================

    async def _test_qa_verification_loop(self):
        """Test 3: QA → Verification → QA feedback loop."""
        print("Test 3: Cross-Department Workflow (QA → Verification → QA)")
        print("-" * 70)

        try:
            # Step 1: QA detects issues (Trough)
            print("  Step 1: QA detects issues...")
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(PYTHON_CODE_WITH_ISSUES)
                temp_path = f.name

            detection_result = await self.trough_server.trough_scan_file(
                file_path=temp_path,
                include_ml_logic=False
            )

            print(f"    ✓ Detected {len(detection_result['issues'])} issues")

            # Step 2: QA sends to Verification (xTerminator proposes fixes)
            print("  Step 2: Verification proposes fixes...")
            fix_proposals = []
            for issue in detection_result['issues'][:3]:  # Test first 3
                proposal = await self.xterminator_server.xterminator_propose_fix(
                    issue=issue,
                    code=PYTHON_CODE_WITH_ISSUES,
                    file_path=temp_path,
                    safety_level='balanced'
                )
                fix_proposals.append(proposal)

            print(f"    ✓ Generated {len(fix_proposals)} fix proposals")

            # Step 3: Verification validates fixes
            print("  Step 3: Verification validates fixes...")
            validations = []
            for proposal in fix_proposals:
                validation = await self.xterminator_server.xterminator_validate_fix(
                    fix_id=proposal['fix_id']
                )
                validations.append(validation)

            passed = sum(1 for v in validations if v['validation_passed'])
            print(f"    ✓ Validated {len(validations)} fixes ({passed} passed)")

            # Step 4: QA learns from feedback (metrics updated)
            print("  Step 4: QA learns from feedback...")
            metrics = await self.trough_server.trough_get_metrics()

            print(f"    ✓ Updated metrics: {metrics['total_scans']} scans")

            print("✓ Full workflow completed successfully!")

            self.test_results.append({
                'test': 'qa_verification_loop',
                'status': 'PASS',
                'issues_detected': len(detection_result['issues']),
                'fixes_proposed': len(fix_proposals),
                'fixes_validated': passed
            })

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'qa_verification_loop',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    # ===================================================================
    # SESSION STATE TESTS
    # ===================================================================

    async def _test_session_state(self):
        """Test 4: Session-aware state management."""
        print("Test 4: Session-Aware State Management")
        print("-" * 70)

        try:
            # Multiple scans in same session
            initial_metrics = await self.trough_server.trough_get_metrics()
            initial_scans = initial_metrics['total_scans']

            # Perform 3 scans
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(PYTHON_CODE_WITH_ISSUES)
                temp_path = f.name

            for i in range(3):
                await self.trough_server.trough_scan_file(
                    file_path=temp_path,
                    include_ml_logic=False
                )

            # Check metrics updated
            final_metrics = await self.trough_server.trough_get_metrics()
            final_scans = final_metrics['total_scans']

            assert final_scans >= initial_scans + 3, "Metrics should increment"

            print(f"✓ Session state maintained:")
            print(f"  Initial scans: {initial_scans}")
            print(f"  Final scans: {final_scans}")
            print(f"  Incremented by: {final_scans - initial_scans}")

            self.test_results.append({
                'test': 'session_state',
                'status': 'PASS',
                'scans_incremented': final_scans - initial_scans
            })

            # Cleanup
            os.unlink(temp_path)

        except Exception as e:
            print(f"✗ Test failed: {e}")
            self.test_results.append({
                'test': 'session_state',
                'status': 'FAIL',
                'error': str(e)
            })

        print()

    # ===================================================================
    # SUMMARY
    # ===================================================================

    def _print_summary(self):
        """Print test summary."""
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print()

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print()

        if failed > 0:
            print("FAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  ✗ {result['test']}: {result.get('error', 'Unknown')}")
        else:
            print("✓ ALL TESTS PASSED!")

        print()
        print("=" * 70)

        # Save results
        with open('mcp_integration_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2)

        print("Results saved to: mcp_integration_test_results.json")
        print()


async def main():
    """Run integration tests."""
    tester = MCPIntegrationTester()
    await tester.run_all_tests()


if __name__ == '__main__':
    asyncio.run(main())
