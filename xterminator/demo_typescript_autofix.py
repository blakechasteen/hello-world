"""
TypeScript Autofix Demo
=======================

Demonstrates TypeScript autofix capabilities on real Trough codebase.

Author: mythRL Team
Date: November 16, 2025
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any
from typescript_fixer import TypeScriptFixer
from autofix_tracker import AutoFixTracker, AutoFixResult
from autofix_policy import AutofixPolicy, FixDecision


async def demo_basic_fixes():
    """Demonstrate basic TypeScript fixes"""

    print("=" * 70)
    print("TYPESCRIPT AUTOFIX DEMO - Basic Fixes")
    print("=" * 70)
    print()

    fixer = TypeScriptFixer(use_llm=False)

    # Example 1: Unused import
    print("Example 1: Unused Import Fix")
    print("-" * 70)

    ts_code_1 = """import * as vscode from 'vscode';
import { HoloLoomBridge, CodeContext } from './HoloLoomBridge';

export class Example {
    private bridge: HoloLoomBridge;

    constructor(bridge: HoloLoomBridge) {
        this.bridge = bridge;
    }
}
"""

    issue_1 = {
        'category': 'dead_code',
        'message': "Unused import 'CodeContext'",
        'line_number': 2,
        'code_snippet': "import { HoloLoomBridge, CodeContext } from './HoloLoomBridge';"
    }

    print("Original Code:")
    print(ts_code_1)

    result = await fixer.fix_issue(issue_1, ts_code_1, 'example.ts')
    if result:
        fixed_code, diff = result
        print("\nDiff:")
        print(diff)
    print()

    # Example 2: Any type fix
    print("Example 2: Replace 'any' with 'unknown'")
    print("-" * 70)

    ts_code_2 = """export async function process(data: any) {
    return data.value;
}
"""

    issue_2 = {
        'category': 'type_safety',
        'message': "Parameter 'data' implicitly has an 'any' type",
        'line_number': 1,
        'code_snippet': 'export async function process(data: any) {'
    }

    print("Original Code:")
    print(ts_code_2)

    result = await fixer.fix_issue(issue_2, ts_code_2, 'example.ts')
    if result:
        fixed_code, diff = result
        print("\nDiff:")
        print(diff)
    print()

    # Example 3: Non-null assertion fix
    print("Example 3: Replace Non-Null Assertion with Optional Chaining")
    print("-" * 70)

    ts_code_3 = """export class Service {
    getValue(obj: any) {
        const value = obj!.property;
        const item = arr![0];
        return value;
    }
}
"""

    issue_3 = {
        'category': 'type_safety',
        'message': "Non-null assertion is risky",
        'line_number': 3,
        'code_snippet': "const value = obj!.property;"
    }

    print("Original Code:")
    print(ts_code_3)

    result = await fixer.fix_issue(issue_3, ts_code_3, 'example.ts')
    if result:
        fixed_code, diff = result
        print("\nDiff:")
        print(diff)
    print()


async def demo_trough_scan():
    """Scan actual Trough codebase for issues"""

    print("=" * 70)
    print("TYPESCRIPT AUTOFIX DEMO - Trough Codebase Scan")
    print("=" * 70)
    print()

    trough_src = Path(__file__).parent.parent / 'trough' / 'src'

    if not trough_src.exists():
        print("❌ Trough source directory not found")
        print(f"   Expected: {trough_src}")
        return

    fixer = TypeScriptFixer(use_llm=False)
    all_issues = []

    print(f"Scanning: {trough_src}")
    print()

    # Scan all TypeScript files
    for ts_file in sorted(trough_src.glob('*.ts')):
        print(f"📄 {ts_file.name}")

        try:
            code = ts_file.read_text()
            issues = fixer.detect_unused_imports(code)

            if issues:
                print(f"   Found {len(issues)} unused imports:")
                for issue in issues[:3]:  # Show first 3
                    print(f"   - Line {issue['line_number']}: {issue['message']}")
                if len(issues) > 3:
                    print(f"   ... and {len(issues) - 3} more")

                all_issues.extend([(ts_file, issue) for issue in issues])
            else:
                print("   ✅ No unused imports")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

    print("=" * 70)
    print(f"Total Issues Found: {len(all_issues)}")
    print("=" * 70)
    print()

    return all_issues


async def demo_autofix_with_policy():
    """Demonstrate autofix with policy-based decision making"""

    print("=" * 70)
    print("TYPESCRIPT AUTOFIX DEMO - Policy-Based Auto-Fix")
    print("=" * 70)
    print()

    # Create policy (balanced - default)
    policy = AutofixPolicy.balanced(
        department_name="TypeScript Team",
        domain="typescript"
    )

    print(f"Policy: {policy}")
    print()

    # Create fixer and tracker
    fixer = TypeScriptFixer(use_llm=False)
    tracker = AutoFixTracker(output_dir="./autofix_tracking")

    # Start session
    session_id = tracker.start_session(
        max_files=10,
        categories=['dead_code', 'type_safety', 'code_quality'],
        confidence_threshold=0.85
    )

    # Example issues
    issues = [
        {
            'file_path': 'example.ts',
            'category': 'dead_code',
            'message': "Unused import 'Unused'",
            'line_number': 1,
            'code_snippet': "import { Unused } from './module';",
            'full_code': "import { Unused } from './module';\nexport class Example {}",
            'confidence': 0.95,
            'risk_level': 'LOW',
            'fix_strategy': 'AST',
            'has_tests': True
        },
        {
            'file_path': 'service.ts',
            'category': 'type_safety',
            'message': "Parameter 'data' has type 'any'",
            'line_number': 5,
            'code_snippet': "function process(data: any) {",
            'full_code': "export function process(data: any) {\n    return data.value;\n}",
            'confidence': 0.75,
            'risk_level': 'MEDIUM',
            'fix_strategy': 'AST',
            'has_tests': False
        }
    ]

    print("Processing issues with policy...")
    print()

    for issue_data in issues:
        print(f"Issue: {issue_data['message']}")
        print(f"  File: {issue_data['file_path']}")
        print(f"  Confidence: {issue_data['confidence']:.1%}")
        print(f"  Risk: {issue_data['risk_level']}")

        # Make decision using policy
        from xterminator_types import RiskLevel, FixStrategy
        decision, reason = policy.decide(
            confidence=issue_data['confidence'],
            risk_level=RiskLevel[issue_data['risk_level']],
            fix_strategy=FixStrategy[issue_data['fix_strategy']],
            has_tests=issue_data['has_tests']
        )

        print(f"  Decision: {decision.value}")
        print(f"  Reason: {reason}")

        if decision == FixDecision.AUTO:
            # Apply fix
            issue = {
                'category': issue_data['category'],
                'message': issue_data['message'],
                'line_number': issue_data['line_number'],
                'code_snippet': issue_data['code_snippet']
            }

            result = await fixer.fix_issue(
                issue,
                issue_data['full_code'],
                issue_data['file_path']
            )

            if result:
                fixed_code, diff = result

                # Track result
                fix_result = AutoFixResult(
                    fix_id=f"ts_fix_{len(tracker.sessions) + 1}",
                    category=issue_data['category'],
                    severity='low',
                    message=issue_data['message'],
                    confidence=issue_data['confidence'],
                    file_path=issue_data['file_path'],
                    line_number=issue_data['line_number'],
                    code_snippet=issue_data['code_snippet'],
                    strategy='ast',
                    original_code=issue_data['full_code'],
                    fixed_code=fixed_code,
                    diff=diff,
                    applied=True,
                    validation_passed=True,
                    errors=[],
                    warnings=[],
                    branch='autofix/typescript',
                    duration_ms=5.0
                )

                tracker.track_fix(fix_result)

                print("  ✅ Fix applied successfully")
                print()
                print("  Diff Preview:")
                print("  " + "\n  ".join(diff.split('\n')[:10]))
            else:
                print("  ❌ Fix failed")

        print()
        print("-" * 70)
        print()

    # End session and show statistics
    tracker.end_session()


async def main():
    """Run all demos"""

    # Demo 1: Basic fixes
    await demo_basic_fixes()

    # Demo 2: Scan Trough codebase
    await demo_trough_scan()

    # Demo 3: Policy-based autofix
    await demo_autofix_with_policy()


if __name__ == '__main__':
    asyncio.run(main())
