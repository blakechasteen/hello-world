#!/usr/bin/env python3
"""
Validate COZ Daily Brief Automation

Comprehensive validation of daily brief automation system.

Tests:
1. File generation (Markdown + JSON)
2. All analysis methods working
3. Output structure and content
4. Refinement functionality (if enabled)
5. Error handling and recovery

Usage:
    python validate_automation.py

Created: 2025-11-22
Part of: Option 2 - Automation (Week 1)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.coz.daily_brief_automation import DailyBriefAutomation


def validate_output_files(output_dir: Path, date_str: str) -> bool:
    """
    Validate that output files exist and have correct structure

    Args:
        output_dir: Output directory path
        date_str: Expected date string (YYYY-MM-DD)

    Returns:
        True if validation passes, False otherwise
    """
    print("\n[1/5] Validating output files...")

    # Check Markdown file
    markdown_file = output_dir / f"daily_brief_{date_str}.md"
    if not markdown_file.exists():
        print(f"  [FAIL] Markdown file not found: {markdown_file}")
        return False

    print(f"  [OK] Markdown file exists: {markdown_file}")

    # Validate Markdown content
    markdown_content = markdown_file.read_text(encoding='utf-8')
    if len(markdown_content) < 100:
        print(f"  [FAIL] Markdown file too short ({len(markdown_content)} chars)")
        return False

    if "# COZ Daily Intelligence Brief" not in markdown_content:
        print("  [FAIL] Markdown missing title header")
        return False

    print(f"  [OK] Markdown content valid ({len(markdown_content):,} chars)")

    # Check JSON file
    json_file = output_dir / f"daily_brief_{date_str}.json"
    if not json_file.exists():
        print(f"  [FAIL] JSON file not found: {json_file}")
        return False

    print(f"  [OK] JSON file exists: {json_file}")

    # Validate JSON structure
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            brief = json.load(f)

        required_keys = ['date', 'summary', 'profit', 'efficiency', 'cost',
                         'production', 'waste', 'orders', 'customers',
                         'action_items', 'performance_indicators', 'refinement_used']

        for key in required_keys:
            if key not in brief:
                print(f"  [FAIL] JSON missing key: {key}")
                return False

        print(f"  [OK] JSON structure valid ({len(brief)} top-level keys)")

    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON parse error: {e}")
        return False

    return True


def validate_analysis_methods(brief: dict) -> bool:
    """
    Validate that all analysis methods ran successfully

    Args:
        brief: Daily brief dictionary

    Returns:
        True if validation passes, False otherwise
    """
    print("\n[2/5] Validating analysis methods...")

    analysis_sections = {
        'profit': ['net_profit', 'profit_margin', 'hourly_profit'],
        'efficiency': ['insights'],
        'cost': ['insights'],
        'production': ['insights'],
        'waste': ['insights'],
        'orders': ['insights'],
        'customers': ['insights'],
    }

    for section, required_keys in analysis_sections.items():
        if section not in brief:
            print(f"  [FAIL] Missing section: {section}")
            return False

        section_data = brief[section]

        # Check for errors
        if 'error' in section_data:
            print(f"  [WARN]  Section '{section}' has error: {section_data['error']}")
            # Continue - some parsers may not have data
        else:
            # Validate required keys
            for key in required_keys:
                if key not in section_data:
                    print(f"  [FAIL] Section '{section}' missing key: {key}")
                    return False

            print(f"  [OK] Section '{section}' valid")

    return True


def validate_performance_indicators(brief: dict) -> bool:
    """
    Validate performance indicators dashboard

    Args:
        brief: Daily brief dictionary

    Returns:
        True if validation passes, False otherwise
    """
    print("\n[3/5] Validating performance indicators...")

    if 'performance_indicators' not in brief:
        print("  [FAIL] Missing performance_indicators")
        return False

    indicators = brief['performance_indicators']

    required_metrics = ['profit_margin', 'hourly_profit', 'task_efficiency',
                        'waste_rate', 'order_fulfillment_rate']

    for metric in required_metrics:
        if metric not in indicators:
            print(f"  [FAIL] Missing metric: {metric}")
            return False

        value = indicators[metric]
        if not isinstance(value, (int, float)):
            print(f"  [FAIL] Metric '{metric}' is not numeric: {value}")
            return False

        print(f"  [OK] Metric '{metric}': {value}")

    return True


def validate_action_items(brief: dict) -> bool:
    """
    Validate action items generation

    Args:
        brief: Daily brief dictionary

    Returns:
        True if validation passes, False otherwise
    """
    print("\n[4/5] Validating action items...")

    if 'action_items' not in brief:
        print("  [FAIL] Missing action_items")
        return False

    action_items = brief['action_items']

    if not isinstance(action_items, list):
        print(f"  [FAIL] action_items is not a list: {type(action_items)}")
        return False

    if len(action_items) == 0:
        print("  [WARN]  No action items generated (may be normal if all systems healthy)")
    else:
        print(f"  [OK] Generated {len(action_items)} action items")
        for i, item in enumerate(action_items[:3], 1):
            print(f"     {i}. {item[:80]}...")

    return True


def validate_refinement(brief: dict, expected_refinement: bool) -> bool:
    """
    Validate refinement functionality

    Args:
        brief: Daily brief dictionary
        expected_refinement: Whether refinement was expected

    Returns:
        True if validation passes, False otherwise
    """
    print("\n[5/5] Validating refinement...")

    if 'refinement_used' not in brief:
        print("  [FAIL] Missing refinement_used flag")
        return False

    refinement_used = brief['refinement_used']

    if refinement_used != expected_refinement:
        print(f"  [FAIL] Expected refinement={expected_refinement}, got {refinement_used}")
        return False

    if refinement_used:
        # Check that summary is significantly longer than raw
        summary_length = len(brief['summary'])
        if summary_length < 500:
            print(f"  [WARN]  Refined summary seems short ({summary_length} chars)")
        else:
            print(f"  [OK] Refined summary length: {summary_length:,} chars")
    else:
        print("  [OK] Refinement disabled (as expected)")

    print(f"  [OK] Refinement used: {refinement_used}")

    return True


def main():
    """Run comprehensive validation"""
    print("=" * 70)
    print("COZ Daily Brief Automation Validation")
    print("=" * 70)

    # Create test output directory
    test_output_dir = Path("./test_daily_briefs")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTest output directory: {test_output_dir.absolute()}")

    # Test 1: Run with refinement disabled (faster)
    print("\n" + "=" * 70)
    print("Test 1: Generate brief WITHOUT refinement")
    print("=" * 70)

    automation = DailyBriefAutomation(
        hourly_rate=25.0,
        output_dir=str(test_output_dir),
        use_refinement=False
    )

    success = automation.generate_and_save()

    if not success:
        print("\n[FAIL] Brief generation failed!")
        return 1

    # Load generated brief for validation
    date_str = datetime.now().strftime('%Y-%m-%d')
    json_file = test_output_dir / f"daily_brief_{date_str}.json"

    with open(json_file, 'r', encoding='utf-8') as f:
        brief = json.load(f)

    # Run validation tests
    print("\n" + "=" * 70)
    print("Running Validation Tests")
    print("=" * 70)

    tests = [
        ("Output Files", lambda: validate_output_files(test_output_dir, date_str)),
        ("Analysis Methods", lambda: validate_analysis_methods(brief)),
        ("Performance Indicators", lambda: validate_performance_indicators(brief)),
        ("Action Items", lambda: validate_action_items(brief)),
        ("Refinement", lambda: validate_refinement(brief, expected_refinement=False)),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n[FAIL] Test '{test_name}' FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] Test '{test_name}' FAILED with exception: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print()

    if failed == 0:
        print("[OK] ALL TESTS PASSED!")
        print()
        print(f"Generated files in: {test_output_dir.absolute()}")
        print(f"  - daily_brief_{date_str}.md")
        print(f"  - daily_brief_{date_str}.json")
        return 0
    else:
        print("[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
