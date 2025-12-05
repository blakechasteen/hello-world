"""
Phase 2 Validation Script - Quick Check

Validates Phase 2 implementation without complex imports.
Run from coz directory: python validate_phase2.py
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def validate_customer_orders():
    """Validate customer_orders.csv exists and is parseable"""
    print("[1/5] Validating customer_orders.csv...")

    file_path = Path("customer_orders.csv")
    if not file_path.exists():
        print("  ❌ File not found: customer_orders.csv")
        return False

    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            orders = list(reader)

            if not orders:
                print("  ❌ File is empty")
                return False

            # Validate required columns
            required_cols = ['Order ID', 'Customer Name', 'Order Date', 'Due Date',
                           'Status', 'Products', 'Quantities', 'Total Price', 'Notes']
            first_row = orders[0]

            for col in required_cols:
                if col not in first_row:
                    print(f"  ❌ Missing column: {col}")
                    return False

            # Calculate summary
            total_revenue = sum(float(o['Total Price']) for o in orders)
            statuses = {}
            for order in orders:
                status = order['Status']
                statuses[status] = statuses.get(status, 0) + 1

            print(f"  ✅ Parsed {len(orders)} orders")
            print(f"     Total pipeline: ${total_revenue:.2f}")
            print(f"     By status: {dict(statuses)}")

            return True

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def validate_sops():
    """Validate sops.md exists and is parseable"""
    print("\n[2/5] Validating sops.md...")

    file_path = Path("sops.md")
    if not file_path.exists():
        print("  ❌ File not found: sops.md")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Count SOP sections
            import re
            sop_sections = re.findall(r'## SOP: (.+)', content)

            if not sop_sections:
                print("  ❌ No SOPs found in file")
                return False

            print(f"  ✅ Found {len(sop_sections)} SOPs")
            for sop_name in sop_sections:
                print(f"     - {sop_name}")

            # Check for required sections
            required_sections = ['Steps:', 'Materials:', 'Equipment:']
            for section in required_sections:
                if f"### {section}" not in content:
                    print(f"  ⚠️ Missing section: {section}")

            return True

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def validate_production_log():
    """Validate production_log.csv exists and is parseable"""
    print("\n[3/5] Validating production_log.csv...")

    file_path = Path("production_log.csv")
    if not file_path.exists():
        print("  ❌ File not found: production_log.csv")
        return False

    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            entries = list(reader)

            if not entries:
                print("  ❌ File is empty")
                return False

            # Validate required columns
            required_cols = ['Date', 'Product', 'Quantity Produced', 'Quantity Sold',
                           'Quantity Wasted', 'Waste Reason', 'Notes']
            first_row = entries[0]

            for col in required_cols:
                if col not in first_row:
                    print(f"  ❌ Missing column: {col}")
                    return False

            # Calculate summary
            total_produced = sum(int(e['Quantity Produced']) for e in entries)
            total_sold = sum(int(e['Quantity Sold']) for e in entries)
            total_wasted = sum(int(e['Quantity Wasted']) for e in entries)

            waste_percent = (total_wasted / total_produced * 100) if total_produced > 0 else 0
            sellthrough = (total_sold / total_produced * 100) if total_produced > 0 else 0

            print(f"  ✅ Parsed {len(entries)} production entries")
            print(f"     Produced: {total_produced}, Sold: {total_sold}, Wasted: {total_wasted}")
            print(f"     Waste rate: {waste_percent:.1f}%, Sellthrough: {sellthrough:.1f}%")

            return True

    except Exception as e:
        print(f"  ❌ Error parsing file: {e}")
        return False


def validate_parsers():
    """Validate parser Python files exist"""
    print("\n[4/5] Validating Phase 2 parser implementations...")

    files = [
        ("customer_orders_parser.py", "CustomerOrdersParser"),
        ("sop_parser.py", "SOPParser"),
        ("production_log_parser.py", "ProductionLogParser"),
        ("intelligence.py", "IntelligenceEngine (Phase 2 expanded)"),
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


def validate_sync_manager_integration():
    """Validate SyncManager has Phase 2 integrations"""
    print("\n[5/5] Validating SyncManager Phase 2 integration...")

    sync_manager_path = Path("sync_manager.py")
    if not sync_manager_path.exists():
        print("  ❌ sync_manager.py not found")
        return False

    try:
        with open(sync_manager_path, 'r') as f:
            content = f.read()

            # Check for Phase 2 imports and integrations
            checks = [
                ("CustomerOrdersParser import", "from elle.coz.customer_orders_parser import CustomerOrdersParser"),
                ("SOPParser import", "from elle.coz.sop_parser import SOPParser"),
                ("ProductionLogParser import", "from elle.coz.production_log_parser import ProductionLogParser"),
                ("customer_orders parser init", "self.customer_orders = CustomerOrdersParser"),
                ("sops parser init", "self.sops = SOPParser"),
                ("production_log parser init", "self.production_log = ProductionLogParser"),
                ("revenue_cost_analysis method", "def get_revenue_cost_analysis"),
                ("production_efficiency method", "def get_production_efficiency_analysis"),
                ("waste_reduction method", "def get_waste_reduction_recommendations"),
                ("order_fulfillment method", "def get_order_fulfillment_optimization"),
                ("customer_insights method", "def get_customer_insights"),
                ("Phase 2 daily brief", "# Phase 2 Enhancements"),
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
    print(" COZ Phase 2: Orders & Intelligence Validation")
    print("=" * 70)
    print()

    results = {
        "customer_orders_csv": validate_customer_orders(),
        "sops_md": validate_sops(),
        "production_log_csv": validate_production_log(),
        "parser_files": validate_parsers(),
        "sync_manager_integration": validate_sync_manager_integration(),
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
        print("✅ Phase 2 validation successful!")
        print()
        print("Phase 2 Complete: Orders & Intelligence")
        print("  ✅ 3 new parsers (customer_orders, sops, production_log)")
        print("  ✅ 5 advanced intelligence methods")
        print("  ✅ Enhanced daily brief with Phase 2 insights")
        print("  ✅ Full SyncManager integration")
        print()
        print("Next steps (Phase 3 - Optional):")
        print("  1. Implement expenses_parser.py")
        print("  2. Implement suppliers_parser.py")
        print("  3. Add performance optimizations (caching, indexing)")
        print("  4. Implement predictive analytics")
        return 0
    else:
        print()
        print("❌ Phase 2 validation failed")
        print("Please fix the failed checks above")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
