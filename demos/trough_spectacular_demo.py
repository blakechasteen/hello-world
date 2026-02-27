#!/usr/bin/env python3
"""
TROUGH & xTERMINATOR SPECTACULAR DEMONSTRATION
==============================================

A theatrical demonstration of AI code quality assurance!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import glob
from trough import AISlopDetector, MLLogicDetector, Language


def print_banner(text: str, char: str = "="):
    """Print a fancy banner"""
    print()
    print(char * 70)
    print(f"  {text}")
    print(char * 70)
    print()


async def main():
    print_banner("TROUGH & xTERMINATOR LIVE DEMONSTRATION", "=")

    print("Watch as we scan HoloLoom's own codebase for quality issues!")
    print("This is DOGFOODING - using our tools on ourselves!")
    print()

    await asyncio.sleep(0.5)

    # Find Python files
    print_banner("PHASE 1: File Discovery", "-")

    print(">>> Scanning HoloLoom directory...")
    files = glob.glob('hololoom/**/*.py', recursive=True)

    # Filter out tests and cache
    files = [f for f in files if '__pycache__' not in f and 'test_' not in Path(f).name]

    # Sample for demo (first 15 files)
    sample_files = files[:15]

    print(f"    Found {len(files)} Python files total")
    print(f"    Analyzing sample of {len(sample_files)} files:")
    print()

    for i, f in enumerate(sample_files[:8], 1):
        print(f"      {i}. {Path(f).name}")
    if len(sample_files) > 8:
        print(f"      ... and {len(sample_files) - 8} more")
    print()

    await asyncio.sleep(0.5)

    # Initialize detectors
    print_banner("PHASE 2: Detector Initialization", "-")

    print(">>> Creating AI Slop Detector...")
    slop_detector = AISlopDetector()
    print("    Status: ONLINE")
    print("    Capabilities:")
    print("      - Error handling detection")
    print("      - Hardcoded values detection")
    print("      - Resource leak detection")
    print("      - Security issue detection")
    print("      - Performance anti-patterns")
    print("      - Dead code detection")
    print("      - Documentation gaps")
    print("      ... and 8 more categories")
    print()

    print(">>> Creating ML Logic Detector...")
    logic_detector = MLLogicDetector()
    print("    Status: ONLINE")
    print("    Algorithms:")
    print("      - Division by zero detection")
    print("      - Null dereference analysis")
    print("      - Logic contradiction detection")
    print("      - Missing return detection")
    print("      ... and 5 more algorithms")
    print()

    await asyncio.sleep(0.5)

    # Scan files
    print_banner("PHASE 3: Code Analysis", "-")

    print("Scanning files for quality issues...")
    print()

    total_slop = 0
    total_logic = 0
    files_with_issues = []
    issue_categories = {}

    for i, file_path in enumerate(sample_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Run detectors
            slop_issues = await slop_detector.detect_all(code, Language.PYTHON, file_path)
            logic_errors = await logic_detector.detect(code, Language.PYTHON, file_path)

            file_total = len(slop_issues) + len(logic_errors)

            if file_total > 0:
                total_slop += len(slop_issues)
                total_logic += len(logic_errors)
                files_with_issues.append((Path(file_path).name, file_total, len(slop_issues), len(logic_errors)))

                # Track issue categories
                for issue in slop_issues:
                    category = str(issue.category.value) if hasattr(issue.category, 'value') else str(issue.category)
                    issue_categories[category] = issue_categories.get(category, 0) + 1

                print(f"  [{i:2d}/{len(sample_files)}] {Path(file_path).name}: {file_total} issues ({len(slop_issues)} slop, {len(logic_errors)} logic)")
            else:
                print(f"  [{i:2d}/{len(sample_files)}] {Path(file_path).name}: CLEAN")

            await asyncio.sleep(0.2)

        except Exception as e:
            print(f"  [{i:2d}/{len(sample_files)}] {Path(file_path).name}: ERROR - {e}")

    print()

    await asyncio.sleep(0.5)

    # Results
    print_banner("PHASE 4: Analysis Results", "-")

    print(f"Files scanned: {len(sample_files)}")
    print(f"Files with issues: {len(files_with_issues)}")
    print(f"Clean files: {len(sample_files) - len(files_with_issues)}")
    print()
    print(f"Total AI Slop issues: {total_slop}")
    print(f"Total ML Logic errors: {total_logic}")
    print(f"GRAND TOTAL: {total_slop + total_logic}")
    print()

    if files_with_issues:
        print("Top offenders:")
        for fname, total, slop, logic in sorted(files_with_issues, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {total:3d} issues - {fname} ({slop} slop, {logic} logic)")
        print()

    if issue_categories:
        print("Issue breakdown by category:")
        for category, count in sorted(issue_categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {count:3d} - {category}")
        print()

    await asyncio.sleep(0.5)

    # Grand finale
    print_banner("THE GRAND FINALE", "=")

    clean_rate = (len(sample_files) - len(files_with_issues)) / len(sample_files) * 100 if sample_files else 0

    print("TROUGH & xTERMINATOR VERDICT:")
    print()
    print(f"  Code Quality Score: {clean_rate:.1f}% clean files")
    print(f"  Issues Detected: {total_slop + total_logic}")
    print(f"  Detection Categories: {len(issue_categories)}")
    print()

    if total_slop + total_logic > 0:
        print("Next steps:")
        print("  1. Review detected issues")
        print("  2. Run xTerminator auto-fixer")
        print("  3. Validate fixes with tests")
        print("  4. Commit improvements")
    else:
        print("PERFECT! No issues detected!")

    print()
    print_banner("DEMONSTRATION COMPLETE", "=")
    print("Trough & xTerminator: Keeping HoloLoom clean since 2025!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
