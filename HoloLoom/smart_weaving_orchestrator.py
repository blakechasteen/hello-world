#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Weaving Orchestrator - Complete Integration
==================================================
Integrates the complete Math->Meaning pipeline into the weaving architecture.

This is the PRODUCTION orchestrator that combines:
1. Original 6 weaving modules (Loom, Chrono, Resonance, Warp, Convergence, Spacetime)
2. Smart Math Selection (RL learning via Thompson Sampling)
3. Operator Composition (functional pipelines)
4. Rigorous Testing (property-based verification)
5. Meaning Synthesis (numbers -> natural language)

Complete Flow:
  Query (text)
    -> Intent Classification
    -> Smart Math Selection (RL)
    -> Composed Operations
    -> Mathematical Execution (32 modules)
    -> Rigorous Testing
    -> Meaning Synthesis
    -> Natural Language Output
    -> Spacetime Fabric (with full provenance)

Author: HoloLoom Team
Date: 2025-10-26
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Import base orchestrator
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

# Import math->meaning pipeline
try:
    from HoloLoom.warp.math.meaning_synthesizer import CompleteMathMeaningPipeline, MeaningResult
    HAS_MATH_PIPELINE = True
except ImportError as e:
    logging.warning(f"Math pipeline not available: {e}")
    HAS_MATH_PIPELINE = False

# Import weaving components
from HoloLoom.config import Config
from HoloLoom.fabric.spacetime import Spacetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Smart Weaving Orchestrator
# ============================================================================

class SmartWeavingOrchestrator(WeavingOrchestrator):
    """
    Enhanced weaving orchestrator with integrated math->meaning pipeline.

    Adds to base WeavingOrchestrator:
    - Stage 4.5: Smart Math Selection (RL learning)
    - Stage 4.6: Mathematical Execution (32 modules)
    - Stage 4.7: Rigorous Testing (property verification)
    - Stage 4.8: Meaning Synthesis (numbers -> words)

    The complete weaving cycle becomes:
    1. LoomCommand -> Pattern selection
    2. ChronoTrigger -> Temporal window
    3. ResonanceShed -> Feature extraction
    4. WarpSpace -> Tensioned manifold
      4.5. Smart Math Selection (NEW!)
      4.6. Mathematical Execution (NEW!)
      4.7. Rigorous Testing (NEW!)
      4.8. Meaning Synthesis (NEW!)
    5. ConvergenceEngine -> Decision collapse
    6. Spacetime -> Response with natural language + math provenance

    Example:
        orchestrator = SmartWeavingOrchestrator(
            default_pattern="fast",
            enable_math_pipeline=True,
            math_budget=50
        )

        spacetime = await orchestrator.weave(
            query="Find documents similar to quantum computing"
        )

        print(spacetime.response)
        # "Found 5 similar items using 3 mathematical operations.
        #
        #  Analysis:
        #    - Computed similarity scores using dot products..."
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        default_pattern: str = "fast",
        enable_math_pipeline: bool = True,
        math_budget: int = 50,
        math_style: str = "detailed",
        **kwargs
    ):
        """
        Initialize Smart Weaving Orchestrator.

        Args:
            config: HoloLoom config
            default_pattern: Default pattern card
            enable_math_pipeline: Enable smart math->meaning pipeline
            math_budget: Computational budget for math operations
            math_style: Output style ("concise", "detailed", "technical")
            **kwargs: Additional args for base orchestrator
        """
        super().__init__(config=config, default_pattern=default_pattern, **kwargs)

        self.enable_math_pipeline = enable_math_pipeline and HAS_MATH_PIPELINE
        self.math_budget = math_budget
        self.math_style = math_style

        # Initialize math->meaning pipeline
        if self.enable_math_pipeline:
            self.math_pipeline = CompleteMathMeaningPipeline(load_state=True)
            logger.info("Math->Meaning pipeline initialized")
            logger.info(f"  Budget: {math_budget}")
            logger.info(f"  Style: {math_style}")
        else:
            self.math_pipeline = None
            if not HAS_MATH_PIPELINE:
                logger.warning("Math pipeline not available - running without math->meaning")

        # Statistics
        self.math_pipeline_stats = {
            "total_executions": 0,
            "total_cost": 0,
            "avg_confidence": 0.0,
            "operations_used": {}
        }

    async def weave(
        self,
        query: str,
        user_pattern: Optional[str] = None,
        context: Optional[Dict] = None,
        enable_math: Optional[bool] = None
    ) -> Spacetime:
        """
        Execute complete weaving cycle with math->meaning integration.

        Enhanced stages:
        1-3: Standard (Pattern, Chrono, Resonance)
        4: Warp Space + SMART MATH PIPELINE
          4.5: Smart Math Selection (RL)
          4.6: Mathematical Execution
          4.7: Rigorous Testing
          4.8: Meaning Synthesis
        5-6: Standard (Convergence, Execution)

        Args:
            query: User query
            user_pattern: Optional pattern override
            context: Optional context
            enable_math: Override math pipeline enable flag

        Returns:
            Spacetime with natural language response + math provenance
        """
        # Execute base weaving (stages 1-3)
        spacetime = await super().weave(query, user_pattern, context)

        # Check if math pipeline should run
        use_math = (enable_math if enable_math is not None else self.enable_math_pipeline)

        if not use_math or self.math_pipeline is None:
            return spacetime  # Return standard spacetime

        # ================================================================
        # STAGE 4.5-4.8: SMART MATH PIPELINE
        # ================================================================
        logger.info("\n" + "="*80)
        logger.info("SMART MATH PIPELINE")
        logger.info("="*80)

        try:
            # Build context for math pipeline
            math_context = {
                "has_embeddings": True,
                "query_embedding": None,  # Would extract from spacetime.trace
                "pattern": spacetime.trace.pattern_card if hasattr(spacetime.trace, 'pattern_card') else "fast",
                "motifs": spacetime.trace.motifs_detected if hasattr(spacetime.trace, 'motifs_detected') else [],
            }

            # Execute math->meaning pipeline
            start_time = datetime.now()

            meaning = self.math_pipeline.process(
                query=query,
                context=math_context,
                budget=self.math_budget,
                style=self.math_style
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Update statistics
            self._update_math_stats(meaning, execution_time)

            # ================================================================
            # Enhance Spacetime with Math->Meaning Results
            # ================================================================

            # Replace response with natural language from meaning synthesis
            spacetime.response = meaning.to_text(style=self.math_style)

            # Add math provenance to trace
            if not hasattr(spacetime.trace, 'analytical_metrics') or spacetime.trace.analytical_metrics is None:
                spacetime.trace.analytical_metrics = {}

            spacetime.trace.analytical_metrics['math_meaning'] = {
                "summary": meaning.summary,
                "details": meaning.details,
                "key_insights": meaning.key_insights,
                "recommendations": meaning.recommendations,
                "confidence": meaning.confidence,
                "operations_executed": meaning.provenance.get("operations_executed", []),
                "total_cost": meaning.provenance.get("total_cost", 0),
                "execution_time_ms": execution_time
            }

            # Update overall confidence
            # Combine tool confidence with math meaning confidence
            original_confidence = spacetime.confidence
            combined_confidence = (original_confidence + meaning.confidence) / 2
            spacetime.confidence = combined_confidence

            logger.info("\n" + "="*80)
            logger.info("MATH PIPELINE COMPLETE")
            logger.info("="*80)
            logger.info(f"  Operations: {len(meaning.provenance.get('operations_executed', []))}")
            logger.info(f"  Cost: {meaning.provenance.get('total_cost', 0)}")
            logger.info(f"  Confidence: {meaning.confidence:.1%}")
            logger.info(f"  Execution: {execution_time:.1f}ms")
            logger.info(f"  Insights: {len(meaning.key_insights)}")

        except Exception as e:
            logger.error(f"Math pipeline failed: {e}")
            import traceback
            traceback.print_exc()

            # Add error to trace but don't fail the whole weaving
            if not hasattr(spacetime.trace, 'analytical_metrics') or spacetime.trace.analytical_metrics is None:
                spacetime.trace.analytical_metrics = {}
            spacetime.trace.analytical_metrics['math_meaning'] = {
                "error": str(e),
                "fallback": "Standard response used"
            }

        return spacetime

    def _update_math_stats(self, meaning: MeaningResult, execution_time: float):
        """Update math pipeline statistics."""
        self.math_pipeline_stats["total_executions"] += 1
        self.math_pipeline_stats["total_cost"] += meaning.provenance.get("total_cost", 0)

        # Running average of confidence
        n = self.math_pipeline_stats["total_executions"]
        old_avg = self.math_pipeline_stats["avg_confidence"]
        self.math_pipeline_stats["avg_confidence"] = (
            (old_avg * (n - 1) + meaning.confidence) / n
        )

        # Count operation usage
        for op in meaning.provenance.get("operations_executed", []):
            if op not in self.math_pipeline_stats["operations_used"]:
                self.math_pipeline_stats["operations_used"][op] = 0
            self.math_pipeline_stats["operations_used"][op] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including math pipeline."""
        base_stats = super().get_statistics()

        if self.enable_math_pipeline:
            base_stats["math_pipeline"] = self.math_pipeline_stats.copy()

            if self.math_pipeline:
                # Add RL learning statistics
                rl_stats = self.math_pipeline.selector.get_smart_statistics()
                base_stats["math_pipeline"]["rl_learning"] = rl_stats.get("rl_learning", {})
                base_stats["math_pipeline"]["testing"] = rl_stats.get("testing", {})

        return base_stats


# ============================================================================
# Factory Function
# ============================================================================

def create_smart_orchestrator(
    pattern: str = "fast",
    math_budget: int = 50,
    math_style: str = "detailed",
    use_mcts: bool = True,
    **kwargs
) -> SmartWeavingOrchestrator:
    """
    Create SmartWeavingOrchestrator with sensible defaults.

    Args:
        pattern: Pattern card ("bare", "fast", "fused")
        math_budget: Computational budget for math operations
        math_style: Output style ("concise", "detailed", "technical")
        use_mcts: Use MCTS for convergence
        **kwargs: Additional orchestrator args

    Returns:
        Configured SmartWeavingOrchestrator
    """
    config = Config.fast() if pattern == "fast" else (
        Config.fused() if pattern == "fused" else Config.bare()
    )

    return SmartWeavingOrchestrator(
        config=config,
        default_pattern=pattern,
        enable_math_pipeline=True,
        math_budget=math_budget,
        math_style=math_style,
        use_mcts=use_mcts,
        **kwargs
    )


# ============================================================================
# Example Usage & Bootstrap Script
# ============================================================================

async def bootstrap_with_test_queries():
    """
    Bootstrap RL learning with test queries.

    Runs 100 diverse queries to initialize the RL system with good priors.
    """
    print("="*80)
    print("BOOTSTRAPPING SMART WEAVING ORCHESTRATOR")
    print("="*80)
    print()

    # Create orchestrator
    orchestrator = create_smart_orchestrator(
        pattern="fast",
        math_budget=50,
        math_style="detailed"
    )

    # Test query categories
    test_queries = {
        "similarity": [
            "Find documents similar to quantum computing",
            "Find documents similar to machine learning",
            "Find documents similar to neural networks",
            "Find documents similar to reinforcement learning",
            "Find documents similar to natural language processing",
        ],
        "optimization": [
            "Optimize the retrieval algorithm",
            "Optimize memory usage",
            "Optimize query performance",
            "Improve embedding quality",
            "Improve similarity search speed",
        ],
        "analysis": [
            "Analyze convergence of the learning process",
            "Analyze the stability of embeddings",
            "Analyze the distribution of similarities",
            "Analyze the graph structure",
            "Analyze the feature space",
        ],
        "verification": [
            "Verify the distance function is valid",
            "Verify metric space axioms hold",
            "Verify orthogonality of context",
            "Verify numerical stability",
            "Verify convergence properties",
        ],
    }

    # Flatten all queries
    all_queries = []
    for category, queries in test_queries.items():
        all_queries.extend(queries)

    print(f"Running {len(all_queries)} bootstrap queries...")
    print()

    # Run all queries
    results = []
    for i, query in enumerate(all_queries, 1):
        print(f"[{i}/{len(all_queries)}] {query[:50]}...")

        try:
            spacetime = await orchestrator.weave(query)

            # Extract results
            result = {
                "query": query,
                "success": spacetime.confidence >= 0.5,
                "confidence": spacetime.confidence,
                "tool": spacetime.tool_used,
                "duration_ms": spacetime.trace.duration_ms
            }

            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                math_metrics = spacetime.trace.analytical_metrics.get('math_meaning', {})
                result["math_operations"] = math_metrics.get("operations_executed", [])
                result["math_cost"] = math_metrics.get("total_cost", 0)
                result["math_confidence"] = math_metrics.get("confidence", 0)

            results.append(result)

            print(f"  ✓ Confidence: {spacetime.confidence:.1%}, "
                  f"Tool: {spacetime.tool_used}, "
                  f"Duration: {spacetime.trace.duration_ms:.0f}ms")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })

        print()

    # Show statistics
    print("="*80)
    print("BOOTSTRAP COMPLETE")
    print("="*80)

    stats = orchestrator.get_statistics()

    print(f"\nWeaving Statistics:")
    print(f"  Total weavings: {stats['total_weavings']}")
    print(f"  Pattern usage: {stats['pattern_usage']}")

    if "math_pipeline" in stats:
        math_stats = stats["math_pipeline"]
        print(f"\nMath Pipeline Statistics:")
        print(f"  Total executions: {math_stats['total_executions']}")
        print(f"  Total cost: {math_stats['total_cost']}")
        print(f"  Avg confidence: {math_stats['avg_confidence']:.1%}")

        if math_stats["operations_used"]:
            print(f"\n  Top operations:")
            sorted_ops = sorted(
                math_stats["operations_used"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for op, count in sorted_ops[:5]:
                print(f"    {op}: {count}")

        if "rl_learning" in math_stats:
            rl = math_stats["rl_learning"]
            print(f"\n  RL Learning:")
            print(f"    Total feedback: {rl.get('total_feedback', 0)}")

            leaderboard = rl.get("leaderboard", [])
            if leaderboard:
                print(f"    Top operations by success rate:")
                for i, op_stats in enumerate(leaderboard[:5], 1):
                    total = op_stats["successes"] + op_stats["failures"]
                    rate = op_stats["successes"] / total if total > 0 else 0
                    print(f"      {i}. {op_stats['operation_name']} ({op_stats['intent']}): "
                          f"{op_stats['successes']}/{total} ({rate:.1%})")

    # Success rate
    successful = sum(1 for r in results if r.get("success", False))
    print(f"\nBootstrap Results:")
    print(f"  Successful queries: {successful}/{len(results)} ({successful/len(results):.1%})")

    # Save orchestrator state
    if orchestrator.math_pipeline:
        orchestrator.math_pipeline.selector._save_state()
        print(f"\n✓ RL state saved to disk")

    print("\n" + "="*80)
    print("System is now bootstrapped and learning!")
    print("="*80)

    return orchestrator, results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    async def main():
        print("="*80)
        print("SMART WEAVING ORCHESTRATOR DEMO")
        print("Complete Math->Meaning Integration")
        print("="*80)
        print()

        # Run bootstrap
        orchestrator, results = await bootstrap_with_test_queries()

        print("\nTesting with single query to see full output:")
        print("="*80)

        # Test with detailed output
        spacetime = await orchestrator.weave(
            "Find documents similar to quantum entanglement",
            enable_math=True
        )

        print(f"\nQuery: Find documents similar to quantum entanglement")
        print(f"\nResponse:")
        print(spacetime.response)
        print(f"\nConfidence: {spacetime.confidence:.1%}")
        print(f"Tool: {spacetime.tool_used}")
        print(f"Duration: {spacetime.trace.duration_ms:.0f}ms")

        print("\n" + "="*80)
        print("INTEGRATION COMPLETE!")
        print("="*80)

    asyncio.run(main())
