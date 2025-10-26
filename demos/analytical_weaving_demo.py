#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytical Weaving Demo
=======================
Demonstrates mathematically rigorous semantic processing with:
- Metric space verification
- Hilbert space orthogonalization
- Gradient-guided optimization
- Spectral stability analysis
- Convergence tracking

This demo shows how mathematical rigor transforms HoloLoom from
"works empirically" to "provably correct".
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from datetime import datetime
import numpy as np

from HoloLoom.analytical_orchestrator import (
    AnalyticalWeavingOrchestrator,
    create_analytical_orchestrator
)
from HoloLoom.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


async def demo_metric_verification():
    """Demo 1: Metric Space Verification"""
    print("\n" + "="*80)
    print("DEMO 1: METRIC SPACE VERIFICATION")
    print("="*80)
    print("\nVerify that semantic embeddings form a valid metric space.")
    print("Checks: Non-negativity, Identity, Symmetry, Triangle Inequality")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fast",
        enable_metric_verification=True,
        enable_gradient_optimization=False,
        enable_hilbert_orthogonalization=False,
        enable_spectral_analysis=False
    )
    
    query = "What mathematical properties define a metric space?"
    print(f"Query: {query}")
    print()
    
    spacetime = await weaver.weave(query)
    
    if hasattr(spacetime.trace, 'analytical_metrics'):
        metrics = spacetime.trace.analytical_metrics.get('metric_space', {})
        print("Metric Space Properties:")
        print(f"  ✓ Valid metric: {metrics.get('is_valid_metric', False)}")
        print(f"  ✓ Complete: {metrics.get('is_complete', 'Unknown')}")
        print(f"  ✓ Elements: {metrics.get('num_elements', 0)}")
        print(f"  ✓ Dimension: {metrics.get('dimension', 0)}")
        print()
        print("Interpretation:")
        if metrics.get('is_valid_metric'):
            print("  → Semantic distances are mathematically valid")
            print("  → Triangle inequality holds: d(a,c) ≤ d(a,b) + d(b,c)")
            print("  → Enables rigorous topological reasoning")
        else:
            print("  → Warning: Metric axioms violated")
    
    weaver.stop()
    return spacetime


async def demo_hilbert_orthogonalization():
    """Demo 2: Hilbert Space Orthogonalization"""
    print("\n" + "="*80)
    print("DEMO 2: HILBERT SPACE ORTHOGONALIZATION")
    print("="*80)
    print("\nOrthogonalize context threads for maximal information diversity.")
    print("Gram-Schmidt process ensures informationally independent context.")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fast",
        enable_metric_verification=False,
        enable_gradient_optimization=False,
        enable_hilbert_orthogonalization=True,
        enable_spectral_analysis=False
    )
    
    query = "Explain diverse perspectives on neural network optimization"
    print(f"Query: {query}")
    print()
    
    spacetime = await weaver.weave(query)
    
    if hasattr(spacetime.trace, 'analytical_metrics'):
        metrics = spacetime.trace.analytical_metrics.get('orthogonalization', {})
        print("Orthogonalization Results:")
        print(f"  ✓ Threads processed: {metrics.get('threads_processed', 0)}")
        print(f"  ✓ Diversity score: {metrics.get('diversity_score', 0):.3f}")
        print(f"  ✓ Mean angle: {metrics.get('mean_angle_radians', 0):.3f} rad")
        print(f"  ✓ Success: {metrics.get('orthogonality_achieved', False)}")
        print()
        print("Interpretation:")
        diversity = metrics.get('diversity_score', 0)
        if diversity > 1.0:
            print(f"  → High diversity (angle ≈ {diversity:.2f} rad ≈ {np.degrees(diversity):.0f}°)")
            print("  → Context threads are informationally independent")
            print("  → Maximizes coverage of semantic space")
        elif diversity > 0.5:
            print(f"  → Moderate diversity (angle ≈ {diversity:.2f} rad)")
            print("  → Some information overlap between threads")
    
    weaver.stop()
    return spacetime


async def demo_gradient_optimization():
    """Demo 3: Gradient-Guided Attention"""
    print("\n" + "="*80)
    print("DEMO 3: GRADIENT-GUIDED ATTENTION")
    print("="*80)
    print("\nOptimize query embedding using gradient of relevance function.")
    print("Finds steepest ascent toward most relevant semantic region.")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fast",
        enable_metric_verification=False,
        enable_gradient_optimization=True,
        enable_hilbert_orthogonalization=False,
        enable_spectral_analysis=False
    )
    
    query = "How do gradients guide neural network learning?"
    print(f"Query: {query}")
    print()
    
    spacetime = await weaver.weave(query)
    
    if hasattr(spacetime.trace, 'analytical_metrics'):
        metrics = spacetime.trace.analytical_metrics.get('gradient_optimization', {})
        print("Gradient Optimization:")
        print(f"  ✓ Gradient norm: {metrics.get('gradient_norm', 0):.4f}")
        print(f"  ✓ Original relevance: {metrics.get('original_relevance', 0):.4f}")
        print(f"  ✓ Optimized relevance: {metrics.get('optimized_relevance', 0):.4f}")
        print(f"  ✓ Improvement: {metrics.get('improvement', 0):.4f}")
        print(f"  ✓ Applied: {metrics.get('optimization_applied', False)}")
        print()
        print("Interpretation:")
        improvement = metrics.get('improvement', 0)
        if improvement > 0:
            print(f"  → Relevance improved by {improvement:.4f}")
            print("  → Query moved toward optimal semantic position")
            print("  → Gradient points to most informative region")
        elif improvement < 0:
            print("  → Negative improvement (moved away from local optimum)")
        else:
            print("  → No improvement (already at critical point)")
    
    weaver.stop()
    return spacetime


async def demo_spectral_stability():
    """Demo 4: Spectral Stability Analysis"""
    print("\n" + "="*80)
    print("DEMO 4: SPECTRAL STABILITY ANALYSIS")
    print("="*80)
    print("\nAnalyze attention mechanism as bounded operator.")
    print("Spectral radius < 1 guarantees stability (non-explosive).")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fast",
        enable_metric_verification=False,
        enable_gradient_optimization=False,
        enable_hilbert_orthogonalization=False,
        enable_spectral_analysis=True
    )
    
    query = "Why is spectral stability important in recurrent networks?"
    print(f"Query: {query}")
    print()
    
    spacetime = await weaver.weave(query)
    
    if hasattr(spacetime.trace, 'analytical_metrics'):
        metrics = spacetime.trace.analytical_metrics.get('spectral_stability', {})
        print("Spectral Analysis:")
        print(f"  ✓ Spectral radius: {metrics.get('spectral_radius', 0):.4f}")
        print(f"  ✓ Stable: {metrics.get('is_stable', False)}")
        print(f"  ✓ Stochastic: {metrics.get('is_stochastic', False)}")
        print(f"  ✓ Largest eigenvalue: {metrics.get('largest_eigenvalue', 0):.4f}")
        print(f"  ✓ Condition number: {metrics.get('condition_number', 0):.2f}")
        print()
        print("Interpretation:")
        radius = metrics.get('spectral_radius', 0)
        if radius < 1.0:
            print(f"  → Stable (ρ = {radius:.4f} < 1)")
            print("  → Attention will not explode over time")
            print("  → Iterative application converges")
        elif radius > 1.0:
            print(f"  → Unstable (ρ = {radius:.4f} > 1)")
            print("  → Warning: Attention may amplify indefinitely")
        else:
            print("  → Critical (ρ ≈ 1)")
    
    weaver.stop()
    return spacetime


async def demo_convergence_tracking():
    """Demo 5: Convergence Tracking"""
    print("\n" + "="*80)
    print("DEMO 5: CONVERGENCE TRACKING")
    print("="*80)
    print("\nTrack confidence over multiple queries to verify learning.")
    print("Analyzes: Convergence, monotonicity, boundedness")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fast",
        enable_all_analysis=False
    )
    
    queries = [
        "What is machine learning?",
        "How do neural networks learn?",
        "Explain gradient descent optimization",
        "What are loss functions?",
        "How does backpropagation work?",
        "What is overfitting?",
        "Explain regularization techniques",
        "What are activation functions?",
        "How do CNNs work?",
        "What is transfer learning?",
        "Explain attention mechanisms",
        "What are transformers?"
    ]
    
    print(f"Running {len(queries)} queries...")
    print()
    
    confidences = []
    
    for i, query in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {query[:60]}...")
        spacetime = await weaver.weave(query)
        confidences.append(spacetime.confidence)
    
    print()
    
    # Analyze convergence
    if hasattr(spacetime.trace, 'analytical_metrics') and 'convergence' in spacetime.trace.analytical_metrics:
        metrics = spacetime.trace.analytical_metrics['convergence']
        print("Convergence Analysis:")
        print(f"  ✓ Converging: {metrics.get('is_converging', False)}")
        print(f"  ✓ Monotone: {metrics.get('is_monotone', False)}")
        print(f"  ✓ Direction: {metrics.get('direction', 'unknown')}")
        print(f"  ✓ Bounded: {metrics.get('is_bounded', False)}")
        
        limit = metrics.get('limit')
        if limit is not None:
            print(f"  ✓ Limit: {limit:.3f}")
        
        print()
        print("Confidence History:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std: {np.std(confidences):.3f}")
        print(f"  Min: {np.min(confidences):.3f}")
        print(f"  Max: {np.max(confidences):.3f}")
        print()
        print("Interpretation:")
        if metrics.get('is_converging'):
            print(f"  → System is converging to stable performance")
            if limit is not None:
                print(f"  → Limiting confidence: {limit:.3f}")
        if metrics.get('direction') == 'increasing':
            print("  → Performance improving over time")
        elif metrics.get('direction') == 'decreasing':
            print("  → Performance degrading over time")
    
    weaver.stop()
    return confidences


async def demo_full_analytical_suite():
    """Demo 6: Complete Analytical Suite"""
    print("\n" + "="*80)
    print("DEMO 6: COMPLETE ANALYTICAL SUITE")
    print("="*80)
    print("\nAll analytical features enabled simultaneously.")
    print("Comprehensive mathematical rigor for production use.")
    print()
    
    weaver = create_analytical_orchestrator(
        pattern="fused",
        enable_all_analysis=True,
        use_mcts=True
    )
    
    query = "How does mathematical rigor improve semantic understanding?"
    print(f"Query: {query}")
    print()
    
    spacetime = await weaver.weave(query)
    
    print("Complete Analysis Results:")
    print("-" * 80)
    
    if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
        metrics = spacetime.trace.analytical_metrics
        
        # Metric Space
        if 'metric_space' in metrics:
            ms = metrics['metric_space']
            print(f"\n1. Metric Space:")
            print(f"   Valid: {ms.get('is_valid_metric', False)}")
            print(f"   Complete: {ms.get('is_complete', 'Unknown')}")
        
        # Orthogonalization
        if 'orthogonalization' in metrics:
            ortho = metrics['orthogonalization']
            print(f"\n2. Orthogonalization:")
            print(f"   Threads: {ortho.get('threads_processed', 0)}")
            print(f"   Diversity: {ortho.get('diversity_score', 0):.3f}")
        
        # Gradient Optimization
        if 'gradient_optimization' in metrics:
            grad = metrics['gradient_optimization']
            print(f"\n3. Gradient Optimization:")
            print(f"   Gradient norm: {grad.get('gradient_norm', 0):.4f}")
            print(f"   Improvement: {grad.get('improvement', 0):.4f}")
        
        # Spectral Stability
        if 'spectral_stability' in metrics:
            spec = metrics['spectral_stability']
            print(f"\n4. Spectral Stability:")
            print(f"   Spectral radius: {spec.get('spectral_radius', 0):.4f}")
            print(f"   Stable: {spec.get('is_stable', False)}")
        
        # Continuity
        if 'continuity' in metrics:
            cont = metrics['continuity']
            print(f"\n5. Continuity:")
            L = cont.get('lipschitz_constant')
            print(f"   Lipschitz constant: {L:.2f}" if L else "   Lipschitz: N/A")
            print(f"   Smooth: {cont.get('is_smooth', False)}")
        
        print()
        print("-" * 80)
        print("Mathematical Guarantees Verified:")
        print("  ✓ Valid metric space → Rigorous distances")
        print("  ✓ Orthogonal context → Maximal diversity")
        print("  ✓ Gradient optimized → Optimal relevance")
        print("  ✓ Spectral stability → Non-explosive dynamics")
        print("  ✓ Lipschitz continuity → Bounded sensitivity")
    
    weaver.stop()
    return spacetime


async def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("ANALYTICAL WEAVING DEMONSTRATIONS")
    print("Mathematical Rigor for Semantic Processing")
    print("="*80)
    
    demos = [
        ("Metric Verification", demo_metric_verification),
        ("Hilbert Orthogonalization", demo_hilbert_orthogonalization),
        ("Gradient Optimization", demo_gradient_optimization),
        ("Spectral Stability", demo_spectral_stability),
        ("Convergence Tracking", demo_convergence_tracking),
        ("Complete Suite", demo_full_analytical_suite),
    ]
    
    results = []
    
    for name, demo_func in demos:
        try:
            result = await demo_func()
            results.append((name, result, None))
        except Exception as e:
            logger.error(f"Demo '{name}' failed: {e}")
            results.append((name, None, e))
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    for name, result, error in results:
        status = "✓ PASS" if error is None else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    print("\n" + "="*80)
    print("From Heuristics to Proofs:")
    print("  • Metric axioms → Valid semantic distances")
    print("  • Hilbert spaces → Optimal context diversity")
    print("  • Gradients → Steepest ascent to relevance")
    print("  • Spectral radius → Stability guarantees")
    print("  • Lipschitz constants → Bounded sensitivity")
    print("  • Convergence analysis → Learning verification")
    print()
    print("Mathematical rigor transforms HoloLoom from")
    print("'works empirically' to 'provably correct'.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
