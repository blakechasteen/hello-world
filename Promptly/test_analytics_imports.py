#!/usr/bin/env python3
"""
Test Analytics Imports - Verify all modules can be imported
"""

import sys
from pathlib import Path
import os

# Add promptly subdirectory to path
promptly_path = Path(__file__).parent / 'promptly'
sys.path.insert(0, str(promptly_path))

print(f"Python path: {sys.path[0]}")

def test_imports():
    """Test that all analytics modules can be imported"""
    print("Testing analytics imports...")

    try:
        # Core imports
        print("  ✓ Importing performance...")
        from analytics.performance import PerformanceMonitor, OperationTimer

        print("  ✓ Importing usage...")
        from analytics.usage import UsageAnalytics, UsageTracker

        print("  ✓ Importing quality...")
        from analytics.quality import QualityMetrics, QualityTracker

        print("  ✓ Importing visualize...")
        from analytics.visualize import Visualizer, ChartType

        print("  ✓ Importing reports...")
        from analytics.reports import (
            ReportGenerator, ReportFormat, ReportPeriod, ReportConfig
        )

        print("  ✓ Importing integrations...")
        from analytics.integrations import (
            GrafanaExporter, AnalyticsHub, StructuredLogger
        )

        print("  ✓ Importing instrumentation...")
        from analytics.instrumentation import (
            enable_analytics, PromptlyAnalytics, track_operation
        )

        print("  ✓ Importing package...")
        from analytics import (
            PerformanceMonitor,
            UsageAnalytics,
            QualityMetrics,
            Visualizer,
            ReportGenerator,
            enable_analytics
        )

        print("\n✓ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality without Promptly instance"""
    print("\nTesting basic functionality...")

    try:
        from analytics import PerformanceMonitor, UsageAnalytics, QualityMetrics
        import tempfile
        import os

        # Create temporary database paths
        temp_dir = tempfile.mkdtemp()

        print("  ✓ Creating PerformanceMonitor...")
        perf = PerformanceMonitor(
            db_path=os.path.join(temp_dir, 'perf.db'),
            retention_days=30,
            enable_sampling=False  # Don't sample in test
        )

        print("  ✓ Creating UsageAnalytics...")
        usage = UsageAnalytics(
            db_path=os.path.join(temp_dir, 'usage.db'),
            retention_days=90
        )

        print("  ✓ Creating QualityMetrics...")
        quality = QualityMetrics(
            db_path=os.path.join(temp_dir, 'quality.db'),
            retention_days=180
        )

        # Test basic operations
        print("  ✓ Testing performance tracking...")
        with perf.time_operation('test_op'):
            pass

        stats = perf.get_operation_stats()
        assert 'test_op' in stats, "Operation not tracked"
        assert stats['test_op']['count'] == 1, "Operation count incorrect"

        print("  ✓ Testing usage tracking...")
        usage.tracker.track_prompt_access('test_prompt', 'main', 'get')
        most_used = usage.get_most_used_prompts(days=1)
        assert len(most_used) == 1, "Usage not tracked"

        print("  ✓ Testing quality tracking...")
        quality.tracker.record_evaluation(
            'test_prompt', 1, 'main', 0.95, 'test', 'test_1'
        )
        stats = quality.get_prompt_quality_stats('test_prompt', days=1)
        assert stats is not None, "Quality not tracked"
        assert stats['measurements'] == 1, "Quality measurement incorrect"

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print("\n✓ All functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI can be imported"""
    print("\nTesting CLI import...")

    try:
        from analytics import cli
        print("  ✓ CLI imported successfully")

        # Check main commands exist
        assert hasattr(cli, 'analytics_cli'), "analytics_cli not found"
        assert hasattr(cli, 'stats'), "stats command not found"
        assert hasattr(cli, 'report'), "report command not found"
        assert hasattr(cli, 'export'), "export command not found"

        print("  ✓ All CLI commands present")
        print("\n✓ CLI test passed!")
        return True

    except Exception as e:
        print(f"\n✗ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("Promptly Analytics - Import and Functionality Tests")
    print("=" * 80)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))

    # Test CLI
    results.append(("CLI", test_cli()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:<30} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
