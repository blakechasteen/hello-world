"""
Phase 1 Validation Script - Quick Check

Validates Phase 1 implementation without complex imports.
Run from coz directory: python validate_phase1.py
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def validate_time_tracking():
    """Validate time_tracking.csv exists and is parseable"""
    print("[1/4] Validating time_tracking.csv...")

    file_path = Path("time_tracking.csv")
    if not file_path.exists():
        print("  ❌ File not found: time_tracking.csv")
        return False

    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            entries = list(reader)

            if not entries:
                print("  ❌ File is empty")
                return False

            # Validate required columns
            required_cols = ['Date', 'Task ID', 'Task Name', 'Category',
                           'Estimated Hours', 'Actual Hours', 'Notes']
            first_row = entries[0]

            for col in required_cols:
                if col not in first_row:
                    print(f"  ❌ Missing column: {col}")
                    return False

            # Calculate summary
            total_estimated = sum(float(e['Estimated Hours']) for e in entries)
            total_actual = sum(float(e['Actual Hours']) for e in entries)
            variance = total_actual - total_estimated

            print(f"  ✅ Parsed {len(entries)} time entries")
            print(f"     Total estimated: {total_estimated:.1f}h")
            print(f"     Total actual: {total_actual:.1f}h")
            print(f"     Variance: {variance:+.1f}h")

            return True

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def validate_cost_tracking():
    """Validate cost_tracking.csv exists and is parseable"""
    print("\n[2/4] Validating cost_tracking.csv...")

    file_path = Path("cost_tracking.csv")
    if not file_path.exists():
        print("  ❌ File not found: cost_tracking.csv")
        return False

    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            entries = list(reader)

            if not entries:
                print("  ❌ File is empty")
                return False

            # Validate required columns
            required_cols = ['Date', 'Task ID', 'Task Name', 'Material Cost',
                           'Labor Cost', 'Overhead Cost', 'Total Cost', 'Notes']
            first_row = entries[0]

            for col in required_cols:
                if col not in first_row:
                    print(f"  ❌ Missing column: {col}")
                    return False

            # Calculate summary
            total_material = sum(float(e['Material Cost']) for e in entries)
            total_labor = sum(float(e['Labor Cost']) for e in entries)
            total_overhead = sum(float(e['Overhead Cost']) for e in entries)
            total_cost = sum(float(e['Total Cost']) for e in entries)

            print(f"  ✅ Parsed {len(entries)} cost entries")
            print(f"     Total material: ${total_material:.2f}")
            print(f"     Total labor: ${total_labor:.2f}")
            print(f"     Total overhead: ${total_overhead:.2f}")
            print(f"     Total cost: ${total_cost:.2f}")

            return True

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def validate_parsers():
    """Validate parser Python files exist"""
    print("\n[3/4] Validating parser implementations...")

    files = [
        ("time_tracking_parser.py", "TimeTrackingParser"),
        ("cost_tracking_parser.py", "CostTrackingParser"),
        ("intelligence.py", "IntelligenceEngine"),
    ]

    all_exist = True

    for filename, class_name in files:
        path = Path(filename)
        if path.exists():
            # Check file size (basic validation)
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {filename} not found")
            all_exist = False

    return all_exist


def validate_sync_manager():
    """Validate SyncManager has Phase 1 integrations"""
    print("\n[4/4] Validating SyncManager integration...")

    sync_manager_path = Path("sync_manager.py")
    if not sync_manager_path.exists():
        print("  ❌ sync_manager.py not found")
        return False

    try:
        with open(sync_manager_path, 'r') as f:
            content = f.read()

            # Check for Phase 1 imports
            checks = [
                ("TimeTrackingParser import", "from elle.coz.time_tracking_parser import TimeTrackingParser"),
                ("CostTrackingParser import", "from elle.coz.cost_tracking_parser import CostTrackingParser"),
                ("IntelligenceEngine import", "from elle.coz.intelligence import IntelligenceEngine"),
                ("time_tracking parser init", "self.time_tracking = TimeTrackingParser"),
                ("cost_tracking parser init", "self.cost_tracking = CostTrackingParser"),
                ("intelligence engine init", "self.intelligence = IntelligenceEngine"),
                ("profit analysis method", "def get_profit_analysis"),
                ("enhanced daily brief", "# Phase 1 Enhancements"),
            ]

            all_found = True
            for check_name, check_string in checks:
                if check_string in content:
                    print(f"  ✅ {check_name}")
                else:
                    print(f"  ❌ {check_name} not found")
                    all_found = False

            return all_found

    except Exception as e:
        print(f"  ❌ Error validating sync_manager.py: {e}")
        return False


def main():
    print("=" * 70)
    print(" COZ Phase 1: Core Tracking Validation")
    print("=" * 70)
    print()

    results = {
        "time_tracking_csv": validate_time_tracking(),
        "cost_tracking_csv": validate_cost_tracking(),
        "parser_files": validate_parsers(),
        "sync_manager_integration": validate_sync_manager(),
    }

    print("\n" + "=" * 70)
    print(" Validation Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name.replace('_', ' ').title()}")

    print()
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print()
        print("✅ Phase 1 validation successful!")
        print()
        print("Next steps:")
        print("  1. Test with: cd ../.. && PYTHONPATH=. python -c 'from elle.coz.sync_manager import SyncManager; s=SyncManager(); s.parse_all()'")
        print("  2. View profit analysis: s.get_profit_analysis()")
        print("  3. View daily brief: s.get_daily_brief()")
        return 0
    else:
        print()
        print("❌ Phase 1 validation failed")
        print("Please fix the failed checks above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
