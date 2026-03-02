#!/usr/bin/env python3
"""
Bootstrap Script - Train RL with 100 Diverse Queries
=====================================================
Runs 100 carefully designed queries across 4 categories to initialize
the RL learning system with good priors.

Categories:
- Similarity (25 queries)
- Optimization (25 queries)
- Analysis (25 queries)
- Verification (25 queries)

Tracks:
- Learning curves (success rate over time)
- Operation usage patterns
- Confidence evolution
- Cost efficiency
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np

# Reduce logging verbosity
logging.basicConfig(level=logging.WARNING)

from smart_weaving_orchestrator import create_smart_orchestrator

# ============================================================================
# Query Generator
# ============================================================================

def generate_diverse_queries():
    """Generate 100 diverse queries across 4 categories."""

    queries = {
        "similarity": [
            # Scientific
            "Find documents similar to quantum entanglement",
            "Find documents similar to neural architecture search",
            "Find documents similar to reinforcement learning from human feedback",
            "Find documents similar to graph neural networks",
            "Find documents similar to attention mechanisms",

            # Technical
            "Find similar code patterns to async/await",
            "Find similar algorithms to quicksort",
            "Find similar data structures to B-trees",
            "Find similar design patterns to observer",
            "Find similar architectures to microservices",

            # Domain-specific
            "Find similar patients to this diagnosis",
            "Find similar market trends to 2020 volatility",
            "Find similar customer segments to millennials",
            "Find similar proteins to hemoglobin",
            "Find similar weather patterns to El Niño",

            # Abstract
            "Find similar concepts to emergence",
            "Find similar theories to relativity",
            "Find similar philosophies to stoicism",
            "Find similar arguments to Pascal's wager",
            "Find similar paradoxes to Zeno's",

            # Varied phrasing
            "What's similar to transformer models?",
            "Show me things like gradient descent",
            "Documents related to CRISPR",
            "Papers about things like diffusion models",
            "Research similar to AlphaFold"
        ],

        "optimization": [
            # Speed
            "Optimize query response time",
            "Optimize database indexing strategy",
            "Optimize network latency",
            "Optimize compilation speed",
            "Optimize rendering performance",

            # Resources
            "Optimize memory usage",
            "Optimize disk I/O",
            "Optimize CPU utilization",
            "Optimize bandwidth consumption",
            "Optimize power efficiency",

            # Quality
            "Optimize model accuracy",
            "Optimize prediction confidence",
            "Optimize retrieval precision",
            "Optimize clustering quality",
            "Optimize embedding quality",

            # Cost
            "Optimize computational cost",
            "Optimize API call budget",
            "Optimize training time",
            "Optimize inference latency",
            "Optimize storage costs",

            # Multi-objective
            "Optimize for both speed and accuracy",
            "Optimize latency while maintaining quality",
            "Optimize throughput under resource constraints",
            "Optimize cost-effectiveness of the pipeline",
            "Optimize learning rate and batch size together"
        ],

        "analysis": [
            # Convergence
            "Analyze convergence of gradient descent",
            "Analyze convergence of the learning process",
            "Analyze convergence of the optimization",
            "Analyze whether training is converging",
            "Analyze convergence rate of the algorithm",

            # Stability
            "Analyze stability of the neural network",
            "Analyze stability of embeddings over time",
            "Analyze numerical stability of computations",
            "Analyze stability of the attention mechanism",
            "Analyze stability under distribution shift",

            # Structure
            "Analyze graph structure of the knowledge base",
            "Analyze clustering structure in embeddings",
            "Analyze hierarchical structure of concepts",
            "Analyze topology of the manifold",
            "Analyze community structure in the network",

            # Distributions
            "Analyze distribution of similarities",
            "Analyze distribution of errors",
            "Analyze distribution of attention weights",
            "Analyze distribution of gradients",
            "Analyze distribution of activations",

            # Patterns
            "Analyze patterns in query logs",
            "Analyze patterns in failure cases",
            "Analyze patterns in user behavior",
            "Analyze patterns in time series data",
            "Analyze patterns in feature correlations"
        ],

        "verification": [
            # Metric properties
            "Verify the distance function is valid",
            "Verify metric space axioms hold",
            "Verify triangle inequality is satisfied",
            "Verify symmetry of the distance metric",
            "Verify positive definiteness",

            # Mathematical properties
            "Verify orthogonality of basis vectors",
            "Verify normalization of embeddings",
            "Verify linear independence",
            "Verify positive semi-definiteness",
            "Verify convexity of the loss function",

            # Numerical properties
            "Verify numerical stability of the algorithm",
            "Verify no NaN or infinity values",
            "Verify bounded outputs",
            "Verify gradient magnitudes are reasonable",
            "Verify weights are in valid ranges",

            # Convergence properties
            "Verify monotonic decrease of loss",
            "Verify convergence to stationary point",
            "Verify satisfaction of KKT conditions",
            "Verify Lipschitz continuity",
            "Verify contraction mapping property",

            # Consistency
            "Verify consistency with previous results",
            "Verify reproducibility of outputs",
            "Verify invariance to random seed",
            "Verify deterministic behavior",
            "Verify idempotency of operations"
        ]
    }

    return queries


# ============================================================================
# Bootstrap Runner
# ============================================================================

async def run_bootstrap():
    """Run bootstrap with progress tracking."""

    print("="*80)
    print("BOOTSTRAP: Training RL with 100 Diverse Queries")
    print("="*80)
    print()

    # Create orchestrator
    print("Initializing SmartWeavingOrchestrator...")
    orchestrator = create_smart_orchestrator(
        pattern="fast",
        math_budget=50,
        math_style="detailed"
    )
    print("OK Initialized")
    print()

    # Generate queries
    query_sets = generate_diverse_queries()
    all_queries = []
    for category, queries in query_sets.items():
        for query in queries:
            all_queries.append({"category": category, "text": query})

    print(f"Generated {len(all_queries)} diverse queries")
    print(f"  Similarity: {len(query_sets['similarity'])}")
    print(f"  Optimization: {len(query_sets['optimization'])}")
    print(f"  Analysis: {len(query_sets['analysis'])}")
    print(f"  Verification: {len(query_sets['verification'])}")
    print()

    # Run queries with tracking
    results = []
    learning_curve = []

    print("Running bootstrap queries...")
    print("-" * 80)

    for i, query_data in enumerate(all_queries):
        query = query_data["text"]
        category = query_data["category"]

        # Progress
        if i % 10 == 0:
            print(f"\n[{i}/{len(all_queries)}] Progress: {i/len(all_queries)*100:.0f}%")

        print(f"  {category[:4]}: {query[:60]}...", end=" ", flush=True)

        try:
            # Execute query
            spacetime = await orchestrator.weave(query, enable_math=True)

            # Extract results
            result = {
                "iteration": i,
                "category": category,
                "query": query,
                "success": spacetime.confidence >= 0.5,
                "confidence": spacetime.confidence,
                "tool": spacetime.tool_used,
                "duration_ms": spacetime.trace.duration_ms,
                "timestamp": datetime.now().isoformat()
            }

            # Extract math metrics if available
            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                math_metrics = spacetime.trace.analytical_metrics.get('math_meaning', {})
                result["math_operations"] = math_metrics.get("operations_executed", [])
                result["math_cost"] = math_metrics.get("total_cost", 0)
                result["math_confidence"] = math_metrics.get("confidence", 0.0)
                result["math_insights"] = len(math_metrics.get("key_insights", []))

            results.append(result)
            print(f"OK (conf={spacetime.confidence:.2f})")

        except Exception as e:
            print(f"FAIL ({e})")
            results.append({
                "iteration": i,
                "category": category,
                "query": query,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

        # Track learning curve every 10 queries
        if (i + 1) % 10 == 0:
            recent = results[-10:]
            learning_curve.append({
                "iteration": i + 1,
                "avg_confidence": np.mean([r.get("confidence", 0) for r in recent if "confidence" in r]),
                "success_rate": np.mean([r.get("success", False) for r in recent]),
                "avg_duration_ms": np.mean([r.get("duration_ms", 0) for r in recent if "duration_ms" in r]),
                "avg_math_cost": np.mean([r.get("math_cost", 0) for r in recent if "math_cost" in r]),
            })

    print()
    print("-" * 80)
    print("Bootstrap complete!")
    print()

    # Compute statistics
    successful = [r for r in results if r.get("success", False)]

    print("="*80)
    print("BOOTSTRAP RESULTS")
    print("="*80)
    print()

    print(f"Overall Performance:")
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Avg confidence: {np.mean([r.get('confidence', 0) for r in successful]):.2f}")
    print(f"  Avg duration: {np.mean([r.get('duration_ms', 0) for r in successful]):.0f}ms")
    print()

    # By category
    print(f"Performance by Category:")
    for category in ["similarity", "optimization", "analysis", "verification"]:
        cat_results = [r for r in results if r.get("category") == category]
        cat_successful = [r for r in cat_results if r.get("success", False)]

        if cat_results:
            print(f"  {category.capitalize()}:")
            print(f"    Success: {len(cat_successful)}/{len(cat_results)} ({len(cat_successful)/len(cat_results)*100:.0f}%)")
            if cat_successful:
                print(f"    Avg confidence: {np.mean([r.get('confidence', 0) for r in cat_successful]):.2f}")
    print()

    # Math pipeline stats
    with_math = [r for r in results if "math_operations" in r]
    if with_math:
        print(f"Math Pipeline:")
        print(f"  Executions: {len(with_math)}")
        print(f"  Avg operations: {np.mean([len(r['math_operations']) for r in with_math]):.1f}")
        print(f"  Avg cost: {np.mean([r.get('math_cost', 0) for r in with_math]):.1f}")
        print(f"  Avg confidence: {np.mean([r.get('math_confidence', 0) for r in with_math]):.2f}")
        print()

        # Operation usage
        all_ops = {}
        for r in with_math:
            for op in r.get("math_operations", []):
                all_ops[op] = all_ops.get(op, 0) + 1

        print(f"  Top operations:")
        for op, count in sorted(all_ops.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {op}: {count}")
        print()

    # Get orchestrator statistics
    stats = orchestrator.get_statistics()

    if "math_pipeline" in stats:
        mp = stats["math_pipeline"]
        print(f"RL Learning Statistics:")
        print(f"  Total executions: {mp['total_executions']}")
        print(f"  Total cost: {mp['total_cost']}")
        print(f"  Avg confidence: {mp['avg_confidence']:.2f}")
        print()

        if "rl_learning" in mp and mp["rl_learning"].get("total_feedback", 0) > 0:
            rl = mp["rl_learning"]
            print(f"  RL Feedback: {rl.get('total_feedback', 0)}")

            leaderboard = rl.get("leaderboard", [])
            if leaderboard:
                print(f"  Top operations by success rate:")
                for i, op_stats in enumerate(leaderboard[:10], 1):
                    total = op_stats["successes"] + op_stats["failures"]
                    rate = op_stats["successes"] / total if total > 0 else 0
                    print(f"    {i}. {op_stats['operation_name']} ({op_stats['intent']}): "
                          f"{op_stats['successes']}/{total} ({rate:.1%})")
        print()

    # Save results
    output_dir = Path("bootstrap_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save learning curve
    with open(output_dir / f"learning_curve_{timestamp}.json", "w") as f:
        json.dump(learning_curve, f, indent=2)

    # Save statistics
    with open(output_dir / f"statistics_{timestamp}.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Results saved to: {output_dir}/")
    print(f"  - results_{timestamp}.json")
    print(f"  - learning_curve_{timestamp}.json")
    print(f"  - statistics_{timestamp}.json")
    print()

    # Save RL state
    if orchestrator.math_pipeline:
        orchestrator.math_pipeline.selector._save_state()
        print(f"RL state saved to: HoloLoom/warp/math/.smart_selector_state.json")

    print()
    print("="*80)
    print("BOOTSTRAP COMPLETE - System is now trained!")
    print("="*80)

    return results, learning_curve, stats


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results, learning_curve, stats = asyncio.run(run_bootstrap())
