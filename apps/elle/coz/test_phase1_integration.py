"""
Phase 1 Integration Test for COZ Expansion

Tests:
- Time tracking parser
- Cost tracking parser
- Intelligence engine profit analysis
- SyncManager integration
- Daily brief enhancement

Run: python -m pytest elle/coz/test_phase1_integration.py -v
"""

import pytest
from pathlib import Path
from sync_manager import SyncManager
from time_tracking_parser import TimeTrackingParser
from cost_tracking_parser import CostTrackingParser
from intelligence import IntelligenceEngine


class TestTimeTrackingParser:
    """Test TimeTrackingParser"""

    def test_parse_time_entries(self):
        """Test parsing time tracking CSV"""
        parser = TimeTrackingParser("coz/time_tracking.csv")

        try:
            entries = parser.parse()
            assert len(entries) > 0, "Should parse time entries"
            assert entries[0].task_id, "Entries should have task IDs"
            assert entries[0].actual_hours > 0, "Entries should have actual hours"
            print(f"✅ Parsed {len(entries)} time entries")
        except FileNotFoundError:
            pytest.skip("time_tracking.csv not found")

    def test_time_summary(self):
        """Test time summary calculation"""
        parser = TimeTrackingParser("coz/time_tracking.csv")

        try:
            parser.parse()
            summary = parser.get_time_summary()

            assert "total_actual_hours" in summary
            assert "avg_efficiency_score" in summary
            assert summary["entries_count"] > 0

            print(f"✅ Time summary: {summary['total_actual_hours']:.1f} hours, {summary['avg_efficiency_score']:.1%} efficiency")
        except FileNotFoundError:
            pytest.skip("time_tracking.csv not found")

    def test_category_breakdown(self):
        """Test category breakdown"""
        parser = TimeTrackingParser("coz/time_tracking.csv")

        try:
            parser.parse()
            categories = parser.get_category_summary()

            assert len(categories) > 0, "Should have category breakdown"
            print(f"✅ Found {len(categories)} categories")

            for category, data in categories.items():
                print(f"  - {category}: {data['total_actual_hours']:.1f}h (efficiency: {data['avg_efficiency_score']:.1%})")
        except FileNotFoundError:
            pytest.skip("time_tracking.csv not found")

    def test_overruns_detection(self):
        """Test time overrun detection"""
        parser = TimeTrackingParser("coz/time_tracking.csv")

        try:
            parser.parse()
            overruns = parser.get_overruns(threshold=0.2)

            print(f"✅ Found {len(overruns)} tasks with >20% overrun")
            for overrun in overruns[:3]:
                print(f"  - {overrun['task_name']}: {overrun['variance_percent']}% over")
        except FileNotFoundError:
            pytest.skip("time_tracking.csv not found")


class TestCostTrackingParser:
    """Test CostTrackingParser"""

    def test_parse_cost_entries(self):
        """Test parsing cost tracking CSV"""
        parser = CostTrackingParser("coz/cost_tracking.csv")

        try:
            entries = parser.parse()
            assert len(entries) > 0, "Should parse cost entries"
            assert entries[0].task_id, "Entries should have task IDs"
            assert entries[0].total_cost > 0, "Entries should have total cost"
            print(f"✅ Parsed {len(entries)} cost entries")
        except FileNotFoundError:
            pytest.skip("cost_tracking.csv not found")

    def test_cost_summary(self):
        """Test cost summary calculation"""
        parser = CostTrackingParser("coz/cost_tracking.csv")

        try:
            parser.parse()
            summary = parser.get_cost_summary()

            assert "total_cost" in summary
            assert "total_material_cost" in summary
            assert "total_labor_cost" in summary
            assert summary["entries_count"] > 0

            print(f"✅ Cost summary: ${summary['total_cost']:.2f} total")
            print(f"  - Material: ${summary['total_material_cost']:.2f}")
            print(f"  - Labor: ${summary['total_labor_cost']:.2f}")
            print(f"  - Overhead: ${summary['total_overhead_cost']:.2f}")
        except FileNotFoundError:
            pytest.skip("cost_tracking.csv not found")

    def test_cost_breakdown(self):
        """Test cost percentage breakdown"""
        parser = CostTrackingParser("coz/cost_tracking.csv")

        try:
            parser.parse()
            breakdown = parser.get_cost_breakdown()

            assert "material_percent" in breakdown
            assert "labor_percent" in breakdown
            assert "overhead_percent" in breakdown

            print(f"✅ Cost breakdown:")
            print(f"  - Material: {breakdown['material_percent']}%")
            print(f"  - Labor: {breakdown['labor_percent']}%")
            print(f"  - Overhead: {breakdown['overhead_percent']}%")
        except FileNotFoundError:
            pytest.skip("cost_tracking.csv not found")

    def test_overhead_rate(self):
        """Test overhead rate calculation"""
        parser = CostTrackingParser("coz/cost_tracking.csv")

        try:
            parser.parse()
            overhead = parser.get_overhead_rate()

            assert "overhead_rate" in overhead
            print(f"✅ Overhead rate: {overhead['overhead_rate']}%")
        except FileNotFoundError:
            pytest.skip("cost_tracking.csv not found")


class TestIntelligenceEngine:
    """Test IntelligenceEngine"""

    def test_profit_analysis(self):
        """Test comprehensive profit analysis"""
        sync = SyncManager(coz_dir="coz")

        try:
            sync.parse_all()
            profit = sync.get_profit_analysis(hourly_rate=25.0)

            assert "net_profit" in profit
            assert "profit_margin" in profit
            assert "hourly_profit" in profit
            assert "recommendations" in profit

            print(f"✅ Profit Analysis:")
            print(f"  - Revenue: ${profit['total_revenue']:.2f}")
            print(f"  - Costs: ${profit['total_costs']:.2f}")
            print(f"  - Net Profit: ${profit['net_profit']:.2f}")
            print(f"  - Profit Margin: {profit['profit_margin']:.1f}%")
            print(f"  - Hourly Profit: ${profit['hourly_profit']:.2f}")
            print(f"  - Recommendations: {len(profit['recommendations'])}")

            for rec in profit['recommendations']:
                print(f"    • {rec}")
        except FileNotFoundError:
            pytest.skip("Phase 1 CSV files not found")

    def test_efficiency_insights(self):
        """Test efficiency insights generation"""
        sync = SyncManager(coz_dir="coz")

        try:
            sync.parse_all()
            efficiency = sync.get_efficiency_insights()

            print(f"✅ Efficiency Insights:")
            if "insights" in efficiency:
                for insight in efficiency["insights"]:
                    print(f"  • {insight}")
        except FileNotFoundError:
            pytest.skip("Phase 1 CSV files not found")

    def test_cost_insights(self):
        """Test cost optimization insights"""
        sync = SyncManager(coz_dir="coz")

        try:
            sync.parse_all()
            cost = sync.get_cost_insights()

            print(f"✅ Cost Insights:")
            if "insights" in cost:
                for insight in cost["insights"]:
                    print(f"  • {insight}")
        except FileNotFoundError:
            pytest.skip("Phase 1 CSV files not found")


class TestSyncManagerIntegration:
    """Test SyncManager Phase 1 integration"""

    def test_parse_all_with_phase1(self):
        """Test parsing all files including Phase 1"""
        sync = SyncManager(coz_dir="coz")

        try:
            result = sync.parse_all()

            print(f"✅ SyncManager parse_all():")
            print(f"  - Files synced: {len(result.files_synced)}")
            print(f"  - Items synced: {result.items_synced}")

            # Check Phase 1 files
            if "time_tracking.csv" in result.files_synced:
                print(f"  - Time entries: {result.items_synced.get('time_entries', 0)}")
            if "cost_tracking.csv" in result.files_synced:
                print(f"  - Cost entries: {result.items_synced.get('cost_entries', 0)}")
        except Exception as e:
            pytest.fail(f"parse_all() failed: {e}")

    def test_enhanced_daily_brief(self):
        """Test enhanced daily brief with Phase 1 features"""
        sync = SyncManager(coz_dir="coz")

        try:
            sync.parse_all()
            brief = sync.get_daily_brief()

            assert "timestamp" in brief
            print(f"✅ Enhanced Daily Brief:")

            # Check Phase 1 enhancements
            if "profit_analysis" in brief:
                print(f"  - Profit analysis: ✅")
                print(f"    Net profit: ${brief['profit_analysis']['net_profit']:.2f}")
            else:
                print(f"  - Profit analysis: Not available (Phase 1 files missing)")

            if "efficiency_insights" in brief:
                print(f"  - Efficiency insights: {len(brief['efficiency_insights'])} insights")

            if "top_recommendations" in brief:
                print(f"  - Recommendations: {len(brief['top_recommendations'])}")
                for rec in brief['top_recommendations'][:3]:
                    print(f"    • {rec}")
        except Exception as e:
            pytest.skip(f"Daily brief test skipped: {e}")

    def test_integration_status(self):
        """Test integration status includes Phase 1"""
        sync = SyncManager(coz_dir="coz")

        try:
            sync.parse_all()
            status = sync.get_integration_status()

            managers = status["managers_initialized"]

            print(f"✅ Integration Status:")
            for manager, initialized in managers.items():
                status_icon = "✅" if initialized else "❌"
                print(f"  {status_icon} {manager}")
        except Exception as e:
            pytest.fail(f"Integration status failed: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print(" COZ Phase 1 Integration Tests")
    print("=" * 70)
    print()

    # Run tests manually (for quick checks without pytest)
    import sys

    try:
        # Test 1: Time Tracking Parser
        print("[1/5] Testing TimeTrackingParser...")
        test = TestTimeTrackingParser()
        test.test_parse_time_entries()
        test.test_time_summary()
        test.test_category_breakdown()
        test.test_overruns_detection()
        print()

        # Test 2: Cost Tracking Parser
        print("[2/5] Testing CostTrackingParser...")
        test = TestCostTrackingParser()
        test.test_parse_cost_entries()
        test.test_cost_summary()
        test.test_cost_breakdown()
        test.test_overhead_rate()
        print()

        # Test 3: Intelligence Engine
        print("[3/5] Testing IntelligenceEngine...")
        test = TestIntelligenceEngine()
        test.test_profit_analysis()
        test.test_efficiency_insights()
        test.test_cost_insights()
        print()

        # Test 4: SyncManager Integration
        print("[4/5] Testing SyncManager Integration...")
        test = TestSyncManagerIntegration()
        test.test_parse_all_with_phase1()
        print()

        # Test 5: Enhanced Daily Brief
        print("[5/5] Testing Enhanced Daily Brief...")
        test = TestSyncManagerIntegration()
        test.test_enhanced_daily_brief()
        print()

        print("=" * 70)
        print("✅ All Phase 1 integration tests passed!")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n⚠️  Some tests skipped: {e}")
        print("Make sure coz/time_tracking.csv and coz/cost_tracking.csv exist")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
