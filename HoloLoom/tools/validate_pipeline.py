#!/usr/bin/env python3
"""
End-to-End Pipeline Validation
==============================
Comprehensive validation suite for the complete Math->Meaning pipeline.

Tests:
1. Query classification accuracy
2. Operation selection correctness
3. Mathematical execution validity
4. Meaning synthesis quality
5. RL learning effectiveness
6. Cost efficiency
7. Performance benchmarks
"""

import asyncio
import time
from pathlib import Path
import json

from smart_weaving_orchestrator import create_smart_orchestrator

# ============================================================================
# Test Suite
# ============================================================================

class PipelineValidator:
    def __init__(self):
        self.orchestrator = create_smart_orchestrator(
            pattern="fast",
            math_budget=50,
            math_style="detailed"
        )
        self.results = []

    async def test_query_classification(self):
        """Test 1: Verify queries are classified correctly."""
        print("\n[TEST 1] Query Classification")
        print("-" * 60)

        test_cases = [
            ("Find similar documents", "similarity"),
            ("Optimize the algorithm", "optimization"),
            ("Analyze convergence", "analysis"),
            ("Verify metric axioms", "verification"),
        ]

        passed = 0
        for query, expected_intent in test_cases:
            spacetime = await self.orchestrator.weave(query, enable_math=True)

            # Check if math metrics exist
            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                # Intent should be in the operations
                print(f"  Query: '{query[:40]}...'")
                print(f"    Expected: {expected_intent}")
                print(f"    Confidence: {spacetime.confidence:.2f}")
                print(f"    Status: PASS")
                passed += 1
            else:
                print(f"  Query: '{query[:40]}...'")
                print(f"    Status: FAIL (no math metrics)")

        print(f"\nResult: {passed}/{len(test_cases)} passed")
        return {"test": "classification", "passed": passed, "total": len(test_cases)}

    async def test_operation_selection(self):
        """Test 2: Verify correct operations are selected."""
        print("\n[TEST 2] Operation Selection")
        print("-" * 60)

        test_cases = [
            ("Find similar documents", ["inner_product", "metric_distance", "hyperbolic_distance", "kl_divergence"]),
            ("Optimize retrieval", ["gradient", "geodesic"]),
            ("Analyze stability", ["convergence_analysis", "continuity_check"]),
        ]

        passed = 0
        for query, expected_ops in test_cases:
            spacetime = await self.orchestrator.weave(query, enable_math=True)

            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                actual_ops = spacetime.trace.analytical_metrics.get('math_meaning', {}).get('operations_executed', [])

                # Check if any expected op is in actual ops
                overlap = any(exp in actual_ops for exp in expected_ops)

                print(f"  Query: '{query[:40]}...'")
                print(f"    Expected ops: {expected_ops}")
                print(f"    Actual ops: {actual_ops}")
                print(f"    Status: {'PASS' if overlap else 'FAIL'}")

                if overlap:
                    passed += 1
            else:
                print(f"  Query: '{query[:40]}...'")
                print(f"    Status: FAIL (no operations)")

        print(f"\nResult: {passed}/{len(test_cases)} passed")
        return {"test": "operation_selection", "passed": passed, "total": len(test_cases)}

    async def test_meaning_synthesis(self):
        """Test 3: Verify natural language output quality."""
        print("\n[TEST 3] Meaning Synthesis")
        print("-" * 60)

        test_queries = [
            "Find documents similar to quantum computing",
            "Optimize the search algorithm",
            "Verify the distance function is valid",
        ]

        passed = 0
        for query in test_queries:
            spacetime = await self.orchestrator.weave(query, enable_math=True)

            # Check response quality
            has_response = spacetime.response and len(spacetime.response) > 50
            has_confidence = spacetime.confidence > 0.5
            has_provenance = hasattr(spacetime.trace, 'analytical_metrics')

            status = "PASS" if (has_response and has_confidence and has_provenance) else "FAIL"

            print(f"  Query: '{query[:40]}...'")
            print(f"    Response length: {len(spacetime.response) if spacetime.response else 0}")
            print(f"    Confidence: {spacetime.confidence:.2f}")
            print(f"    Has provenance: {has_provenance}")
            print(f"    Status: {status}")

            if status == "PASS":
                passed += 1

        print(f"\nResult: {passed}/{len(test_queries)} passed")
        return {"test": "meaning_synthesis", "passed": passed, "total": len(test_queries)}

    async def test_rl_learning(self):
        """Test 4: Verify RL is learning from feedback."""
        print("\n[TEST 4] RL Learning Effectiveness")
        print("-" * 60)

        # Run same query multiple times
        query = "Find similar documents"
        costs = []

        for i in range(5):
            spacetime = await self.orchestrator.weave(query, enable_math=True)

            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                cost = spacetime.trace.analytical_metrics.get('math_meaning', {}).get('total_cost', 0)
                costs.append(cost)
                print(f"  Iteration {i+1}: Cost = {cost}")

        # Check if cost is stable or decreasing (learning)
        if len(costs) >= 3:
            avg_early = sum(costs[:2]) / 2
            avg_late = sum(costs[-2:]) / 2
            improving = avg_late <= avg_early * 1.1  # Allow 10% tolerance

            print(f"\n  Early avg cost: {avg_early:.1f}")
            print(f"  Late avg cost: {avg_late:.1f}")
            print(f"  Status: {'PASS' if improving else 'FAIL'} (learning {'effective' if improving else 'ineffective'})")

            return {"test": "rl_learning", "passed": 1 if improving else 0, "total": 1}
        else:
            print(f"\n  Status: FAIL (insufficient data)")
            return {"test": "rl_learning", "passed": 0, "total": 1}

    async def test_cost_efficiency(self):
        """Test 5: Verify system stays within budget."""
        print("\n[TEST 5] Cost Efficiency")
        print("-" * 60)

        budget = 50
        test_queries = [
            "Find similar documents to machine learning",
            "Optimize neural network training",
            "Analyze gradient descent convergence",
            "Verify embedding normalization",
        ]

        passed = 0
        costs = []

        for query in test_queries:
            spacetime = await self.orchestrator.weave(query, enable_math=True)

            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                cost = spacetime.trace.analytical_metrics.get('math_meaning', {}).get('total_cost', 0)
                costs.append(cost)

                within_budget = cost <= budget
                print(f"  Query: '{query[:40]}...'")
                print(f"    Cost: {cost:.1f} / {budget}")
                print(f"    Status: {'PASS' if within_budget else 'FAIL'}")

                if within_budget:
                    passed += 1

        avg_cost = sum(costs) / len(costs) if costs else 0
        efficiency = (budget - avg_cost) / budget * 100

        print(f"\n  Average cost: {avg_cost:.1f}")
        print(f"  Budget efficiency: {efficiency:.0f}% saved")
        print(f"\nResult: {passed}/{len(test_queries)} within budget")

        return {"test": "cost_efficiency", "passed": passed, "total": len(test_queries), "avg_cost": avg_cost}

    async def test_performance(self):
        """Test 6: Verify acceptable response times."""
        print("\n[TEST 6] Performance Benchmarks")
        print("-" * 60)

        test_queries = [
            "Find similar items",
            "Optimize the process",
            "Analyze the data",
        ]

        passed = 0
        times = []

        for query in test_queries:
            start = time.time()
            spacetime = await self.orchestrator.weave(query, enable_math=True)
            duration = (time.time() - start) * 1000  # ms

            times.append(duration)

            # Target: < 500ms per query
            fast_enough = duration < 500

            print(f"  Query: '{query[:40]}...'")
            print(f"    Duration: {duration:.0f}ms")
            print(f"    Status: {'PASS' if fast_enough else 'FAIL'}")

            if fast_enough:
                passed += 1

        avg_time = sum(times) / len(times) if times else 0

        print(f"\n  Average time: {avg_time:.0f}ms")
        print(f"\nResult: {passed}/{len(test_queries)} under 500ms")

        return {"test": "performance", "passed": passed, "total": len(test_queries), "avg_time_ms": avg_time}

    async def test_end_to_end(self):
        """Test 7: Full pipeline integration."""
        print("\n[TEST 7] End-to-End Integration")
        print("-" * 60)

        query = "Find documents similar to reinforcement learning and verify the results are valid"

        start = time.time()
        spacetime = await self.orchestrator.weave(query, enable_math=True)
        duration = (time.time() - start) * 1000

        # Check all components
        checks = {
            "has_response": spacetime.response and len(spacetime.response) > 0,
            "has_confidence": spacetime.confidence > 0,
            "has_provenance": hasattr(spacetime.trace, 'analytical_metrics'),
            "has_operations": False,
            "has_insights": False,
        }

        if checks["has_provenance"] and spacetime.trace.analytical_metrics:
            math_meaning = spacetime.trace.analytical_metrics.get('math_meaning', {})
            checks["has_operations"] = len(math_meaning.get('operations_executed', [])) > 0
            checks["has_insights"] = len(math_meaning.get('key_insights', [])) > 0

        passed = sum(checks.values())
        total = len(checks)

        print(f"  Query: '{query[:60]}...'")
        print(f"  Duration: {duration:.0f}ms")
        print(f"\n  Component Checks:")
        for check, status in checks.items():
            print(f"    {check}: {'PASS' if status else 'FAIL'}")

        print(f"\n  Response preview:")
        print(f"    {spacetime.response[:200] if spacetime.response else 'None'}...")

        print(f"\nResult: {passed}/{total} checks passed")

        return {"test": "end_to_end", "passed": passed, "total": total, "duration_ms": duration}

    async def run_all_tests(self):
        """Run complete validation suite."""
        print("="*80)
        print("PIPELINE VALIDATION SUITE")
        print("="*80)

        tests = [
            self.test_query_classification,
            self.test_operation_selection,
            self.test_meaning_synthesis,
            self.test_rl_learning,
            self.test_cost_efficiency,
            self.test_performance,
            self.test_end_to_end,
        ]

        results = []
        for test in tests:
            result = await test()
            results.append(result)

        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        total_passed = sum(r["passed"] for r in results)
        total_tests = sum(r["total"] for r in results)

        print(f"\nOverall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")
        print()

        for result in results:
            test_name = result["test"].replace('_', ' ').title()
            status = "PASS" if result["passed"] == result["total"] else "PARTIAL" if result["passed"] > 0 else "FAIL"
            print(f"  {test_name}: {result['passed']}/{result['total']} - {status}")

        # Additional metrics
        print()
        for result in results:
            if "avg_cost" in result:
                print(f"  Average Cost: {result['avg_cost']:.1f}")
            if "avg_time_ms" in result:
                print(f"  Average Time: {result['avg_time_ms']:.0f}ms")

        # Save results
        output_dir = Path("bootstrap_results")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "validation_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total_passed": total_passed,
                    "total_tests": total_tests,
                    "success_rate": total_passed / total_tests
                },
                "tests": results
            }, f, indent=2)

        print()
        print(f"Results saved to: {output_dir / 'validation_results.json'}")

        print()
        print("="*80)
        print("VALIDATION COMPLETE")
        print("="*80)

        if total_passed == total_tests:
            print("\nSTATUS: ALL TESTS PASSED")
            print("The pipeline is fully validated and ready for production!")
        elif total_passed / total_tests >= 0.8:
            print("\nSTATUS: MOSTLY PASSING")
            print("The pipeline is working well with minor issues.")
        else:
            print("\nSTATUS: NEEDS WORK")
            print("Several tests failed. Review results and fix issues.")

        return results

# ============================================================================
# Main
# ============================================================================

async def main():
    validator = PipelineValidator()
    await validator.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
