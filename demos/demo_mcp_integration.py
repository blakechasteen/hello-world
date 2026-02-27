#!/usr/bin/env python3
"""
MCP Integration Demo - Trough + xTerminator Cross-Department Workflow
=====================================================================

Demonstrates the complete cross-department QA workflow:
1. Trough MCP Server detects issues
2. xTerminator MCP Server proposes fixes
3. Validation pipeline ensures quality
4. Git integration applies fixes safely

This shows how other HoloLoom departments (Verification, MasterWeaver,
Infrastructure) can use the QA Department via MCP.

Author: mythRL Team
Date: November 15, 2025
"""

import asyncio
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from trough.mcp_server import TroughMCPServer
from xterminator.mcp_server import XTerminatorMCPServer


async def demo_cross_department_workflow():
    """
    Simulate cross-department QA integration.

    Scenario: MasterWeaver department requests QA scan of entity extractor code.
    """
    print("=" * 70)
    print("MCP INTEGRATION DEMO")
    print("  Trough + xTerminator Cross-Department Workflow")
    print("=" * 70)
    print()

    # Initialize MCP servers
    print("[1/6] Initializing MCP Servers...")
    trough_server = TroughMCPServer()
    xterminator_server = XTerminatorMCPServer()
    print(f"  ✓ Trough MCP Server v{trough_server.info.version}")
    print(f"  ✓ xTerminator MCP Server v{xterminator_server.info.version}")
    print()

    # Scenario: MasterWeaver requests QA scan
    print("[2/6] Cross-Department Request: MasterWeaver → QA")
    print("  Request: 'Scan entity extraction code for quality issues'")
    print()

    # Step 1: Trough detection
    print("[3/6] Trough Detection...")

    # Scan a sample file (use test file for demo)
    test_file = "hololoom/agentic_search/test_citation.py"

    scan_result = await trough_server.trough_scan_file(
        file_path=test_file,
        min_severity="medium",
        include_ml_logic=True
    )

    print(f"  File: {test_file}")
    print(f"  Issues found: {scan_result['total_issues']}")
    print(f"  Severity breakdown: {scan_result['issues_by_severity']}")
    print(f"  Scan time: {scan_result['scan_duration_ms']:.1f}ms")
    print()

    # Step 2: Classify issues
    print("[4/6] Classifying Issues...")

    if scan_result['issues']:
        classification = await trough_server.trough_classify_issues(
            scan_result['issues']
        )

        summary = classification['classification_summary']
        print(f"  Auto-fixable: {summary['auto_fixable_count']}")
        print(f"  Needs review: {summary['needs_review_count']}")
        print(f"  Manual only: {summary['manual_only_count']}")
        print(f"  False positives: {summary['false_positives_count']}")
        print(f"  Auto-fix rate: {summary['auto_fixable_percent']:.1f}%")
        print()

        # Step 3: Propose fixes for auto-fixable issues
        if classification['auto_fixable']:
            print("[5/6] Proposing Fixes (xTerminator)...")

            # Read file content
            with open(test_file, 'r') as f:
                code = f.read()

            # Propose fix for first auto-fixable issue
            issue = classification['auto_fixable'][0]

            proposal = await xterminator_server.xterminator_propose_fix(
                issue=issue,
                code=code,
                file_path=test_file,
                safety_level="conservative"
            )

            if proposal.get('fix_id'):
                print(f"  Fix ID: {proposal['fix_id']}")
                print(f"  Strategy: {proposal['strategy']}")
                print(f"  Confidence: {proposal['confidence']:.1%}")
                print(f"  Risk: {proposal['risk_level']}")
                print(f"  Requires approval: {proposal['requires_approval']}")
                print()

                # Step 4: Validate fix
                print("[6/6] Validating Fix...")

                validation = await xterminator_server.xterminator_validate_fix(
                    fix_id=proposal['fix_id']
                )

                print(f"  Validation passed: {validation['validation_passed']}")
                print(f"  Safe to apply: {validation['safe_to_apply']}")
                print(f"  Validation time: {validation['validation_duration_ms']:.1f}ms")
                print()

                if validation['safe_to_apply']:
                    print("  ✓ Fix is safe and ready to apply")
                    print("  → Would apply fix with Git integration (demo mode: skipping)")
                else:
                    print("  ✗ Fix requires manual review")
                    for step in validation['steps']:
                        status = "✓" if step['passed'] else "✗"
                        print(f"    {status} {step['step']}: {step['message']}")
                print()
            else:
                print("  ✗ Could not generate fix automatically")
                print(f"  Reason: {proposal.get('error')}")
                print()
        else:
            print("[5/6] No auto-fixable issues found")
            print("[6/6] Validation step skipped")
            print()

    else:
        print("[4/6] No issues found - code quality looks good! ✓")
        print("[5/6] Fix proposal step skipped")
        print("[6/6] Validation step skipped")
        print()

    # Get metrics
    print("=" * 70)
    print("QA DEPARTMENT METRICS")
    print("=" * 70)

    trough_metrics = await trough_server.trough_get_metrics()
    print()
    print("Trough Detection:")
    print(f"  Total scans: {trough_metrics['total_scans']}")
    print(f"  Total issues: {trough_metrics['total_issues']}")
    print(f"  Avg issues/file: {trough_metrics['avg_issues_per_file']:.1f}")
    print()

    print("xTerminator Fixing:")
    print(f"  Fixes proposed: {xterminator_server.session_state['fixes_proposed']}")
    print(f"  Fixes validated: {xterminator_server.session_state['fixes_validated']}")
    print(f"  Fixes applied: {xterminator_server.session_state['fixes_applied']}")
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("✓ Trough MCP Server working correctly")
    print("✓ xTerminator MCP Server working correctly")
    print("✓ Cross-department integration validated")
    print()
    print("Next steps:")
    print("  1. Integrate with HoloLoom Orchestration Department")
    print("  2. Enable Verification → QA workflow")
    print("  3. Enable MasterWeaver → QA workflow")
    print("  4. Enable Infrastructure logging")
    print()


async def demo_directory_scan():
    """
    Demonstrate directory scanning capability.
    """
    print()
    print("=" * 70)
    print("BONUS DEMO: Directory Scan")
    print("=" * 70)
    print()

    server = TroughMCPServer()

    # Scan trough directory
    print("Scanning trough/ directory...")
    result = await server.trough_scan_directory(
        directory="trough",
        max_files=10,
        min_severity="medium"
    )

    print(f"  Files scanned: {result['files_scanned']}")
    print(f"  Total issues: {result['total_issues']}")
    print(f"  Severity breakdown: {result['issues_by_severity']}")
    print(f"  Scan time: {result['scan_duration_ms']:.1f}ms")
    print()

    # Show top files by issue count
    files_with_issues = sorted(
        result['files'],
        key=lambda x: x['issues'],
        reverse=True
    )[:5]

    print("Top files by issue count:")
    for i, file_info in enumerate(files_with_issues, 1):
        print(f"  {i}. {file_info['file']}: {file_info['issues']} issues")
    print()


async def demo_batch_fixing():
    """
    Demonstrate batch fixing capability.
    """
    print("=" * 70)
    print("BONUS DEMO: Batch Fixing")
    print("=" * 70)
    print()

    trough = TroughMCPServer()
    xterminator = XTerminatorMCPServer()

    # Scan file
    test_file = "hololoom/agentic_search/test_citation.py"

    print(f"Scanning {test_file}...")
    scan_result = await trough.trough_scan_file(
        file_path=test_file,
        min_severity="low"
    )

    print(f"  Found {scan_result['total_issues']} issues")
    print()

    if scan_result['issues']:
        # Classify
        classification = await trough.trough_classify_issues(scan_result['issues'])

        auto_fixable = classification['auto_fixable']
        print(f"  Auto-fixable: {len(auto_fixable)}")

        if auto_fixable:
            # Read file
            with open(test_file, 'r') as f:
                code = f.read()

            # Batch fix
            print()
            print("Applying batch fixes...")
            batch_result = await xterminator.xterminator_batch_fix(
                issues=auto_fixable[:3],  # Limit to 3 for demo
                code=code,
                file_path=test_file,
                max_fixes=3
            )

            print(f"  Total issues: {batch_result['total_issues']}")
            print(f"  Proposed: {batch_result['fixes_proposed']}")
            print(f"  Validated: {batch_result['fixes_validated']}")
            print(f"  Applied: {batch_result['fixes_applied']}")
            print()

            # Show results
            for i, result in enumerate(batch_result['results'], 1):
                print(f"  {i}. {result['status']}")
                if 'fix_id' in result:
                    print(f"     Fix ID: {result['fix_id']}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
            print()
    else:
        print("  No issues to fix!")
        print()


async def main():
    """Run all demos."""
    try:
        # Main cross-department workflow
        await demo_cross_department_workflow()

        # Bonus demos
        await demo_directory_scan()
        await demo_batch_fixing()

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print()
        print("Note: This demo requires HoloLoom files to be present.")
        print("Run from repository root with test files available.")
        print()
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
