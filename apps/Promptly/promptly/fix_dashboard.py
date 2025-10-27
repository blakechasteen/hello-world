#!/usr/bin/env python3
"""
Fix Dashboard Performance Issues
=================================
Checks database, adds indexes, and generates test data if needed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tools.prompt_analytics import PromptAnalytics
import sqlite3
from pathlib import Path

def check_database():
    """Check if database exists and has data"""
    analytics = PromptAnalytics()

    print("=" * 60)
    print("Dashboard Performance Check")
    print("=" * 60)

    # Check database
    print(f"\n1. Checking database: {analytics.db_path}")
    if Path(analytics.db_path).exists():
        print("   [OK] Database exists")
        size = Path(analytics.db_path).stat().st_size
        print(f"   [OK] Size: {size} bytes")
    else:
        print("   [WARN] Database not found, will be created")

    # Check data
    print("\n2. Checking data...")
    try:
        summary = analytics.get_summary()
        print(f"   [OK] Total executions: {summary['total_executions']}")
        print(f"   [OK] Unique prompts: {summary['unique_prompts']}")

        if summary['total_executions'] == 0:
            print("\n   [WARN] No data found!")
            print("   [INFO] Run: python demo_dashboard_charts.py")
            return False
    except Exception as e:
        print(f"   [ERROR] Could not read data: {e}")
        return False

    # Add indexes for performance
    print("\n3. Adding database indexes...")
    try:
        conn = sqlite3.connect(analytics.db_path)
        cursor = conn.cursor()

        # Add index on prompt_name for faster GROUP BY
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_name
            ON executions(prompt_name)
        """)

        # Add index on timestamp for timeline queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON executions(timestamp)
        """)

        # Add composite index for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prompt_timestamp
            ON executions(prompt_name, timestamp)
        """)

        conn.commit()
        conn.close()
        print("   [OK] Indexes created/verified")
    except Exception as e:
        print(f"   [WARN] Could not create indexes: {e}")

    # Test query performance
    print("\n4. Testing query performance...")
    import time

    try:
        conn = sqlite3.connect(analytics.db_path)
        cursor = conn.cursor()

        # Test GROUP BY query
        start = time.time()
        cursor.execute("""
            SELECT
                prompt_name,
                COUNT(*) as executions,
                AVG(execution_time) as avg_time
            FROM executions
            GROUP BY prompt_name
            LIMIT 50
        """)
        results = cursor.fetchall()
        duration = (time.time() - start) * 1000

        print(f"   [OK] Query took {duration:.2f}ms")
        print(f"   [OK] Found {len(results)} prompts")

        if duration > 1000:
            print("   [WARN] Query is slow (>1s), database might be large")

        conn.close()
    except Exception as e:
        print(f"   [ERROR] Query test failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("Performance Check Complete!")
    print("=" * 60)
    print("\nTo run dashboard:")
    print("  python web_dashboard_realtime.py")
    print("\nThen open: http://localhost:5000")

    return True

if __name__ == "__main__":
    success = check_database()

    if not success:
        print("\n[ACTION REQUIRED] Generate test data:")
        print("  python demo_dashboard_charts.py")
