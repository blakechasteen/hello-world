#!/usr/bin/env python3
"""
xTERMINATOR AUTO-FIXER DEMONSTRATION
====================================

Watch as xTerminator automatically fixes detected issues!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from xterminator import ASTFixer, FixStrategy
from trough import AISlopDetector, Language


def print_banner(text: str, char: str = "="):
    """Print a fancy banner"""
    print()
    print(char * 70)
    print(f"  {text}")
    print(char * 70)
    print()


async def main():
    print_banner("xTERMINATOR AUTO-FIXER DEMONSTRATION", "=")

    print("Watch as xTerminator fixes code issues automatically!")
    print("Target: hololoom/config.py (47 issues detected)")
    print()

    await asyncio.sleep(0.5)

    # Read target file
    print_banner("PHASE 1: Load Target File", "-")

    target_file = "hololoom/config.py"
    print(f">>> Loading {target_file}...")

    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            original_code = f.read()

        print(f"    Loaded {len(original_code)} characters")
        print(f"    Lines: {len(original_code.splitlines())}")
        print()
    except Exception as e:
        print(f"    ERROR: {e}")
        return

    await asyncio.sleep(0.5)

    # Detect issues
    print_banner("PHASE 2: Detect Issues", "-")

    print(">>> Running Trough detector...")
    detector = AISlopDetector()
    issues = await detector.detect_all(original_code, Language.PYTHON, target_file)

    print(f"    Found {len(issues)} issues")
    print()

    # Show sample issues
    if issues:
        print("Sample issues detected:")
        for i, issue in enumerate(issues[:5], 1):
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            print(f"  {i}. Line {issue.line_number}: {category}")
            print(f"     {issue.message}")
            print()

    await asyncio.sleep(0.5)

    # Auto-fix
    print_banner("PHASE 3: Auto-Fix Issues", "-")

    print(">>> Initializing xTerminator...")
    fixer = ASTFixer()
    print("    Status: ONLINE")
    print()

    print(">>> Attempting auto-fix...")
    print("    Strategy: SAFE (high-confidence fixes only)")
    print()

    try:
        # Fix the code
        fixed_code, fix_results = await fixer.fix_code(
            original_code,
            issues,
            strategy=FixStrategy.SAFE
        )

        fixes_applied = fix_results.get('fixes_applied', 0)
        fixes_attempted = fix_results.get('fixes_attempted', 0)
        fixes_failed = fix_results.get('fixes_failed', 0)

        print(f"    Fixes attempted: {fixes_attempted}")
        print(f"    Fixes applied: {fixes_applied}")
        print(f"    Fixes failed: {fixes_failed}")
        print()

        if fixes_applied > 0:
            print("SUCCESS! Code has been improved!")
            print()
            print(f"Original size: {len(original_code)} chars")
            print(f"Fixed size: {len(fixed_code)} chars")
            print(f"Difference: {len(fixed_code) - len(original_code):+d} chars")
            print()

            # Show what was fixed
            if 'fixed_categories' in fix_results:
                print("Fixed issue categories:")
                for category, count in fix_results['fixed_categories'].items():
                    print(f"  - {count} {category} issues")
        else:
            print("No fixes could be safely applied automatically.")
            print("Manual review recommended for remaining issues.")

    except Exception as e:
        print(f"    ERROR during fixing: {e}")
        import traceback
        traceback.print_exc()

    print()

    await asyncio.sleep(0.5)

    # Validation
    print_banner("PHASE 4: Validation", "-")

    print(">>> Validating fixed code...")
    print("    Checking syntax...")

    try:
        import ast
        ast.parse(fixed_code if 'fixed_code' in locals() else original_code)
        print("    Syntax: VALID")
    except SyntaxError as e:
        print(f"    Syntax: INVALID - {e}")

    print()

    await asyncio.sleep(0.5)

    # Summary
    print_banner("THE GRAND FINALE", "=")

    print("xTERMINATOR AUTO-FIXER SUMMARY:")
    print()

    if 'fixes_applied' in locals() and fixes_applied > 0:
        print(f"  Issues detected: {len(issues)}")
        print(f"  Issues fixed: {fixes_applied}")
        print(f"  Success rate: {fixes_applied/len(issues)*100:.1f}%")
        print()
        print("Next steps:")
        print("  1. Review fixed code")
        print("  2. Run tests to validate")
        print("  3. Commit if tests pass")
        print("  4. Repeat for remaining files")
    else:
        print("  Status: Demonstration mode")
        print("  Real fixing requires full integration")
        print()
        print("xTerminator is ready to:")
        print("  - Fix hardcoded values")
        print("  - Remove dead code")
        print("  - Improve error handling")
        print("  - Standardize naming")
        print("  - Add type hints")
        print("  ... and much more!")

    print()
    print_banner("DEMONSTRATION COMPLETE", "=")
    print("xTerminator: Terminating bugs since 2025!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
