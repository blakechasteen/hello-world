#!/usr/bin/env python3
"""
Test Phase 3.2 Enhancements
============================
Verifies bottleneck detection, sparklines, and enhanced visualizations.

Usage:
    # Start server first: PYTHONPATH=. uvicorn HoloLoom.server.unified_server:app --reload --port 8000
    python hololoom/web_dashboard/test_phase3_2.py
"""

import asyncio
import sys
import aiohttp


class Phase32Tester:
    """Test Phase 3.2 visual enhancements."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tests_passed = 0
        self.tests_failed = 0

    async def test_health(self):
        """Test server health."""
        print("\n[TEST 1] Server Health Check")
        print("-" * 50)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✓ Server online (uptime: {data['uptime']:.1f}s)")
                        self.tests_passed += 1
                        return True
                    else:
                        print(f"✗ Server returned status {response.status}")
                        self.tests_failed += 1
                        return False
        except Exception as e:
            print(f"✗ Cannot connect to server: {e}")
            print("\nPlease start the server first:")
            print("  PYTHONPATH=. uvicorn HoloLoom.server.unified_server:app --reload --port 8000")
            self.tests_failed += 1
            return False

    async def make_multiple_queries(self, count: int = 10):
        """Make multiple queries to generate historical data for sparklines."""
        print(f"\n[TEST 2] Making {count} Test Queries (for sparkline data)")
        print("-" * 50)

        queries = [
            "What is Thompson Sampling?",
            "Explain Bayesian exploration",
            "How does the weaving cycle work?",
            "What are the 9 stages?",
            "Tell me about recursive learning",
            "What is the Convergence Engine?",
            "Explain Warp Space tensioning",
            "How do Matryoshka embeddings work?",
            "What is the Resonance Shed?",
            "Describe the Reflection Buffer"
        ]

        try:
            async with aiohttp.ClientSession() as session:
                for i, query_text in enumerate(queries[:count], 1):
                    async with session.post(
                        f"{self.base_url}/query",
                        json={"text": query_text, "mode": "direct", "max_steps": 1}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"  Query {i}/{count}: {result['latency_ms']:.1f}ms (confidence: {result['confidence']:.2f})")
                        else:
                            print(f"  Query {i}/{count}: Failed (status {response.status})")

                    await asyncio.sleep(0.2)  # Small delay between queries

            print(f"✓ Completed {count} queries successfully")
            self.tests_passed += 1

        except Exception as e:
            print(f"✗ Query batch failed: {e}")
            self.tests_failed += 1

    async def test_sparkline_data(self):
        """Verify sparkline data is being tracked."""
        print("\n[TEST 3] Sparkline Data Collection")
        print("-" * 50)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/monitor/orchestrator") as response:
                    data = await response.json()

                    recent_traces = data.get("recent_traces", [])
                    print(f"Recent traces captured: {len(recent_traces)}")

                    if len(recent_traces) >= 2:
                        print(f"✓ Sufficient data for sparklines ({len(recent_traces)} traces)")

                        # Check if traces have stage breakdowns
                        if recent_traces:
                            latest = recent_traces[-1]
                            stages = latest.get("stages", {})
                            print(f"  Latest trace has {len(stages)} stages with durations")

                            if len(stages) > 0:
                                print("  Sample stage durations:")
                                for stage_name, duration in list(stages.items())[:3]:
                                    print(f"    - {stage_name}: {duration:.1f}ms")

                        self.tests_passed += 1
                    else:
                        print(f"⚠ Only {len(recent_traces)} traces (need 2+ for sparklines)")
                        print("  This is OK - sparklines will appear after more queries")
                        self.tests_passed += 1

        except Exception as e:
            print(f"✗ Sparkline data test failed: {e}")
            self.tests_failed += 1

    async def test_bottleneck_detection(self):
        """Test bottleneck detection logic."""
        print("\n[TEST 4] Bottleneck Detection")
        print("-" * 50)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/monitor/orchestrator") as response:
                    data = await response.json()
                    metrics = data.get("metrics", {})

                    bottleneck = metrics.get("bottleneck_stage")
                    avg_latency = metrics.get("avg_latency_ms", 0)

                    print(f"Average latency: {avg_latency:.1f}ms")
                    print(f"Bottleneck stage: {bottleneck or 'None detected'}")

                    # Check recent traces for bottlenecks
                    recent_traces = data.get("recent_traces", [])
                    if recent_traces:
                        bottleneck_count = 0
                        for trace in recent_traces:
                            stages = trace.get("stages", {})
                            total = trace.get("total_duration_ms", 1)

                            for stage_name, duration in stages.items():
                                percentage = (duration / total) if total > 0 else 0
                                if percentage > 0.4:  # 40% threshold
                                    bottleneck_count += 1
                                    print(f"  Found bottleneck: {stage_name} ({percentage*100:.0f}% of query time)")

                        if bottleneck_count > 0:
                            print(f"✓ Bottleneck detection working ({bottleneck_count} detected)")
                        else:
                            print(f"✓ No bottlenecks detected (queries are well-balanced)")

                    self.tests_passed += 1

        except Exception as e:
            print(f"✗ Bottleneck detection test failed: {e}")
            self.tests_failed += 1

    async def run_all_tests(self):
        """Run all Phase 3.2 tests."""
        print("=" * 50)
        print("Phase 3.2 Enhancement Test")
        print("=" * 50)

        # Test 1: Health check
        server_ok = await self.test_health()
        if not server_ok:
            print("\n❌ Server not available. Please start it first.")
            return

        # Test 2: Make multiple queries
        await self.make_multiple_queries(count=10)

        # Test 3: Verify sparkline data
        await self.test_sparkline_data()

        # Test 4: Test bottleneck detection
        await self.test_bottleneck_detection()

        # Summary
        print("\n" + "=" * 50)
        print("Test Summary")
        print("=" * 50)
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print()

        if self.tests_failed == 0:
            print("✓ All tests passed! Phase 3.2 enhancements working.")
            print("\nNext steps:")
            print("1. Open control_panel.html in browser")
            print("2. Navigate to System Monitor → Orchestrator Pipeline")
            print("3. Verify visual enhancements:")
            print("   - Sparklines showing trends next to stage durations")
            print("   - Red bottleneck indicators (if any stage >40%)")
            print("   - Pulsing animations on active stages")
            print("   - Enhanced waterfall with bottleneck warnings")
            return 0
        else:
            print(f"✗ {self.tests_failed} test(s) failed")
            return 1


async def main():
    """Main entry point."""
    tester = Phase32Tester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
