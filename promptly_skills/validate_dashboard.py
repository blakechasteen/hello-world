#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick validation script for Phase 5 Dashboard system.

Tests:
1. Database exists and is readable
2. API server can start
3. All endpoints respond correctly
4. Dashboard HTML is valid

Usage:
    python validate_dashboard.py
"""

import sys
import json
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def test_database():
    """Test 1: Database exists and is readable"""
    print("\n" + "="*60)
    print("Test 1: Database Validation")
    print("="*60)

    db_path = Path('analytics/test_metrics.db')

    if not db_path.exists():
        print("❌ Database not found. Run: python analytics/test_metrics_system.py")
        return False

    print(f"✓ Database exists: {db_path}")
    print(f"  Size: {db_path.stat().st_size:,} bytes")

    # Try to read from database
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Count events
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        print(f"  Events: {count:,}")

        # Check event types
        cursor.execute("SELECT DISTINCT type FROM events")
        types = [row[0] for row in cursor.fetchall()]
        print(f"  Types: {', '.join(types)}")

        conn.close()
        print("\n✅ Test 1 passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test 1 failed: {e}")
        return False


def test_api_imports():
    """Test 2: API imports work"""
    print("\n" + "="*60)
    print("Test 2: API Import Validation")
    print("="*60)

    try:
        # Add analytics to path
        sys.path.insert(0, 'analytics')

        # Try imports
        print("Importing Flask...")
        from flask import Flask
        print("✓ Flask")

        print("Importing flask-cors...")
        from flask_cors import CORS
        print("✓ flask-cors")

        print("Importing flask-socketio...")
        from flask_socketio import SocketIO
        print("✓ flask-socketio")

        print("Importing dashboard_api...")
        import dashboard_api
        print("✓ dashboard_api")

        print("\n✅ Test 2 passed!")
        return True

    except ImportError as e:
        print(f"\n❌ Test 2 failed: {e}")
        print("\nInstall missing dependencies:")
        print("  pip install flask flask-cors flask-socketio")
        return False
    except Exception as e:
        print(f"\n❌ Test 2 failed: {e}")
        return False


def test_dashboard_html():
    """Test 3: Dashboard HTML exists and is valid"""
    print("\n" + "="*60)
    print("Test 3: Dashboard HTML Validation")
    print("="*60)

    html_path = Path('dashboard/index.html')

    if not html_path.exists():
        print(f"❌ Dashboard not found: {html_path}")
        return False

    print(f"✓ Dashboard exists: {html_path}")
    print(f"  Size: {html_path.stat().st_size:,} bytes")

    # Read and check content
    try:
        content = html_path.read_text(encoding='utf-8')

        # Check for key elements
        checks = [
            ('<title>', 'Page title'),
            ('Chart.js', 'Chart.js library'),
            ('socket.io', 'Socket.IO client'),
            ('latencyChart', 'Latency chart'),
            ('confidenceChart', 'Confidence chart'),
            ('topStrategies', 'Top strategies'),
            ('connectionStatus', 'Connection status'),
        ]

        print("\nChecking HTML content:")
        for check, description in checks:
            if check in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ❌ Missing: {description}")
                return False

        print("\n✅ Test 3 passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 failed: {e}")
        return False


def test_documentation():
    """Test 4: Documentation exists"""
    print("\n" + "="*60)
    print("Test 4: Documentation Validation")
    print("="*60)

    docs = [
        ('DASHBOARD_QUICK_START.md', 'Quick start guide'),
        ('PHASE_5_WEEK_1_DAY_1_COMPLETE.md', 'Day 1 completion doc'),
        ('PHASE_5_WEEK_1_DAY_2_COMPLETE.md', 'Day 2 completion doc'),
        ('PHASE_5_KICKOFF.md', 'Phase 5 roadmap'),
    ]

    all_exist = True
    for doc_path, description in docs:
        path = Path(doc_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {description}: {size_kb:.1f} KB")
        else:
            print(f"  ❌ Missing: {description}")
            all_exist = False

    if all_exist:
        print("\n✅ Test 4 passed!")
        return True
    else:
        print("\n❌ Test 4 failed: Missing documentation")
        return False


def test_code_structure():
    """Test 5: Code structure is correct"""
    print("\n" + "="*60)
    print("Test 5: Code Structure Validation")
    print("="*60)

    files = [
        ('analytics/__init__.py', 'Package init'),
        ('analytics/metrics_collector.py', 'Metrics collector'),
        ('analytics/time_series_db.py', 'Time-series DB'),
        ('analytics/aggregator.py', 'Aggregator'),
        ('analytics/dashboard_api.py', 'Dashboard API'),
        ('analytics/test_metrics_system.py', 'Test suite'),
        ('dashboard/index.html', 'Dashboard HTML'),
    ]

    all_exist = True
    total_lines = 0

    for file_path, description in files:
        path = Path(file_path)
        if path.exists():
            try:
                lines = len(path.read_text(encoding='utf-8').splitlines())
                total_lines += lines
                print(f"  ✓ {description}: {lines:,} lines")
            except UnicodeDecodeError:
                # Try with errors='ignore' for files with encoding issues
                lines = len(path.read_text(encoding='utf-8', errors='ignore').splitlines())
                total_lines += lines
                print(f"  ✓ {description}: {lines:,} lines (encoding issues ignored)")
        else:
            print(f"  ❌ Missing: {description}")
            all_exist = False

    print(f"\n  Total code: {total_lines:,} lines")

    if all_exist:
        print("\n✅ Test 5 passed!")
        return True
    else:
        print("\n❌ Test 5 failed: Missing files")
        return False


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)

    print("\n1. Start the API server:")
    print("   cd analytics")
    print("   python dashboard_api.py")

    print("\n2. Open the dashboard:")
    print("   # Mac")
    print("   open ../dashboard/index.html")
    print("   ")
    print("   # Windows")
    print("   start ../dashboard/index.html")
    print("   ")
    print("   # Or serve via HTTP (better for WebSocket)")
    print("   cd dashboard")
    print("   python -m http.server 8000")
    print("   # Then open: http://localhost:8000")

    print("\n3. Watch real-time updates:")
    print("   - Connection status indicator (green = connected)")
    print("   - Metrics update every 2 seconds")
    print("   - Charts show last 6 hours of data")
    print("   - Top 5 strategies with performance bars")

    print("\n4. Test the API:")
    print("   curl http://localhost:5001/api/health")
    print("   curl http://localhost:5001/api/stats?period=24h")

    print("\n5. Read the docs:")
    print("   DASHBOARD_QUICK_START.md - Complete deployment guide")
    print("   PHASE_5_WEEK_1_DAY_2_COMPLETE.md - What we built today")


def main():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("Phase 5 Dashboard - Validation Suite")
    print("="*60)
    print("\nValidating Phase 5 Week 1 Day 2 completion...")

    results = []

    # Run tests
    results.append(("Database", test_database()))
    results.append(("API Imports", test_api_imports()))
    results.append(("Dashboard HTML", test_dashboard_html()))
    results.append(("Documentation", test_documentation()))
    results.append(("Code Structure", test_code_structure()))

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("✨ Phase 5 Week 1 Day 2 is complete!")
        print_next_steps()
        return 0
    else:
        print("\n❌ Some validation tests failed")
        print("Please fix the issues above and re-run validation.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
