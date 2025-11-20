"""
Chain Orchestrator Demonstrations

Shows 6 example chains with the ChainOrchestrator system:
1. Simple query (execute only)
2. Verified query (execute + verify)
3. Auto-refine (execute + verify + conditional refine)
4. Iterative improvement (loop until high confidence)
5. Multi-strategy (fallback strategies)
6. Research pipeline (full cycle with learning)

Usage:
    PYTHONPATH=. python demos/demo_chain_orchestrator.py

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from HoloLoom.chaining import (
    Chain,
    ChainStep,
    StepType,
    ChainOrchestrator,
    ChainPatterns,
    Conditions,
)
from HoloLoom.departments.protocol import DepartmentRequest, DepartmentResponse, ConfidenceMetadata
from unittest.mock import AsyncMock, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Mock Department for Demos
# ============================================================================

class DemoDepartment:
    """Mock department for demonstrations."""

    def __init__(self, confidence_seed: float = 0.8):
        self.department_id = "demo_rag"
        self.confidence_seed = confidence_seed
        self.call_count = 0
        self.refine_count = 0

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute RAG query."""
        self.call_count += 1
        query = request.query.get("query", "")
        mode = request.query.get("mode", "verify")

        # Vary confidence by mode
        confidence_boost = {
            "direct": -0.1,
            "verify": 0.0,
            "research": 0.2,
            "plan_execute": 0.25,
        }.get(mode, 0.0)

        confidence = min(1.0, self.confidence_seed + confidence_boost + (self.call_count - 1) * 0.05)

        logger.info(f"  📊 Execute: mode={mode}, confidence={confidence:.2f}")

        return DepartmentResponse(
            department_id=self.department_id,
            request_id=request.request_id,
            response={
                "answer": f"This is a test answer for: {query[:40]}... (mode={mode}, attempt={self.call_count})",
                "sources": ["source1", "source2", "source3"],
                "reasoning_mode": mode,
            },
            confidence=ConfidenceMetadata.from_score(confidence),
            metadata={
                "original_query": query,
                "execution_mode": mode,
            },
        )

    async def verify(self, response: DepartmentResponse):
        """Verify response using DS-STAR framework."""
        logger.info(f"  ✓ Verify: Running DS-STAR checks...")

        return MagicMock(
            verified=True,
            overall_score=0.88,
            checks=[
                MagicMock(dimension="Domain", passed=True, score=0.95),
                MagicMock(dimension="Sensibility", passed=True, score=0.90),
                MagicMock(dimension="Temporal", passed=True, score=0.85),
                MagicMock(dimension="Argument", passed=True, score=0.85),
                MagicMock(dimension="Reference", passed=True, score=0.85),
            ],
            recommendations=[],
        )

    async def refine(self, response: DepartmentResponse) -> DepartmentResponse:
        """Refine low-confidence response."""
        self.refine_count += 1
        old_conf = response.confidence.score
        new_conf = min(1.0, old_conf + 0.15)

        logger.info(f"  🔧 Refine #{self.refine_count}: {old_conf:.2f} → {new_conf:.2f}")

        response.confidence = ConfidenceMetadata.from_score(new_conf)
        response.metadata["refinement_attempt"] = self.refine_count

        return response

    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """Update learning strategy."""
        logger.info(f"  📚 Learn: Updated strategy with feedback")


# ============================================================================
# Demo 1: Simple Query
# ============================================================================

async def demo_simple_query():
    """Demo 1: Simple query (execute only)."""
    print("\n" + "="*70)
    print("DEMO 1: Simple Query (Execute Only)")
    print("="*70)
    print("Workflow: [execute] → [output]")
    print("Use case: Speed is critical, confidence is sufficient")
    print("Expected latency: ~150ms\n")

    dept = DemoDepartment(confidence_seed=0.85)
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.simple_query()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "What is machine learning?")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Final response: {result.final_response.response['answer'][:50]}...")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 2: Verified Query
# ============================================================================

async def demo_verified_query():
    """Demo 2: Verified query (execute + verify)."""
    print("\n" + "="*70)
    print("DEMO 2: Verified Query (Execute + Verify)")
    print("="*70)
    print("Workflow: [execute] → [verify] → [output]")
    print("Use case: Standard quality checks needed")
    print("Expected latency: ~200-250ms\n")

    dept = DemoDepartment(confidence_seed=0.80)
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.verified_query()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "How does deep learning work?")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Steps executed: {result.stats.completed_steps}")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 3: Auto-Refine (Low Confidence)
# ============================================================================

async def demo_auto_refine():
    """Demo 3: Auto-refine (triggers refinement on low confidence)."""
    print("\n" + "="*70)
    print("DEMO 3: Auto-Refine (Execute + Verify + Conditional Refine)")
    print("="*70)
    print("Workflow: [execute] → [verify] → [if low confidence] → [refine]")
    print("Use case: Automatic improvement on low confidence")
    print("Expected latency: 200-400ms (varies based on confidence)\n")

    dept = DemoDepartment(confidence_seed=0.65)  # Start low
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.auto_refine()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "Explain quantum computing")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Final confidence: {result.confidence:.2f}")
    print(f"  Refines triggered: {dept.refine_count}")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 4: Iterative Improvement
# ============================================================================

async def demo_iterative_improve():
    """Demo 4: Iterative improvement (loop until high confidence)."""
    print("\n" + "="*70)
    print("DEMO 4: Iterative Improvement (Loop Until High Confidence)")
    print("="*70)
    print("Workflow: [execute] → [verify] → [if low] → [loop refine] → [output]")
    print("Use case: Quality is critical, can spend 5-10s per query")
    print("Expected latency: 500ms-2s\n")

    dept = DemoDepartment(confidence_seed=0.60)  # Start lower
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.iterative_improve()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "What are neural networks?")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Final confidence: {result.confidence:.2f}")
    print(f"  Refines performed: {dept.refine_count}")
    print(f"  Total execution time: {result.stats.total_duration_ms:.1f}ms")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 5: Multi-Strategy Fallback
# ============================================================================

async def demo_multi_strategy():
    """Demo 5: Multi-strategy (try multiple approaches)."""
    print("\n" + "="*70)
    print("DEMO 5: Multi-Strategy Fallback")
    print("="*70)
    print("Workflow: [direct] → [verify] → [if failed] → [research] → [verify]")
    print("Use case: First approach might fail, want fallback")
    print("Expected latency: 150-350ms\n")

    dept = DemoDepartment(confidence_seed=0.75)
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.multi_strategy()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "Compare different ML algorithms")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Final confidence: {result.confidence:.2f}")
    print(f"  Strategies attempted: {dept.call_count}")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 6: Research Pipeline
# ============================================================================

async def demo_research_pipeline():
    """Demo 6: Research pipeline (full cycle with learning)."""
    print("\n" + "="*70)
    print("DEMO 6: Research Pipeline (Full Cycle with Learning)")
    print("="*70)
    print("Workflow: [execute] → [verify] → [if needed refine] → [learn]")
    print("Use case: Deep research needed, system should learn")
    print("Expected latency: 300-600ms\n")

    dept = DemoDepartment(confidence_seed=0.70)
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    chain = ChainPatterns.research_pipeline()
    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "What is the future of AI?")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Final confidence: {result.confidence:.2f}")
    print(f"  Refinements: {dept.refine_count}")
    print(f"  Learning signals captured: Yes")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 7: Custom Chain with Conditions
# ============================================================================

async def demo_custom_chain():
    """Demo 7: Custom chain with complex conditions."""
    print("\n" + "="*70)
    print("DEMO 7: Custom Chain with Complex Conditions")
    print("="*70)
    print("Workflow: Custom chain with multiple decision points")
    print("Use case: Domain-specific logic\n")

    dept = DemoDepartment(confidence_seed=0.72)
    orchestrator = ChainOrchestrator(dept, enable_tracing=True)

    # Build custom chain
    chain = Chain(name="custom_quality_first", entry_point="execute")

    chain.add_step("execute", ChainStep(
        step_type=StepType.EXECUTE,
        params={"mode": "research", "max_sources": 15},
        next_step="verify",
    ))

    chain.add_step("verify", ChainStep(
        step_type=StepType.VERIFY,
        params={},
        next_step="check_confidence",
    ))

    # Decision 1: Check confidence
    chain.add_step("check_confidence", ChainStep(
        step_type=StepType.CONDITION,
        params={},
        condition=Conditions.confidence_above(0.8),
        on_success="check_sources",  # If confident, check sources
        on_failure="refine",  # If not confident, refine
    ))

    # Decision 2: Check sources (only if confident)
    chain.add_step("check_sources", ChainStep(
        step_type=StepType.CONDITION,
        params={},
        condition=Conditions.has_sources(min_count=3),
        on_success=None,  # Success: output
        on_failure="add_sources",  # Failure: need more sources
    ))

    # Refinement path
    chain.add_step("refine", ChainStep(
        step_type=StepType.REFINE,
        params={},
        next_step="verify",  # Loop back
    ))

    chain.add_step("add_sources", ChainStep(
        step_type=StepType.EXECUTE,
        params={"mode": "research", "max_sources": 20},
        next_step="verify",  # Re-verify with more sources
    ))

    print(chain.visualize())

    result = await orchestrator.execute_chain(chain, "Complex research question")

    print(f"\n✓ Chain executed successfully!")
    print(f"  Final confidence: {result.confidence:.2f}")
    print(f"  Decision points traversed: 2")
    if result.trace:
        print(result.trace.get_summary())


# ============================================================================
# Demo 8: Performance Comparison
# ============================================================================

async def demo_performance_comparison():
    """Demo 8: Compare latency of different patterns."""
    print("\n" + "="*70)
    print("DEMO 8: Performance Comparison")
    print("="*70)
    print("Comparing latency and quality tradeoffs\n")

    patterns = [
        ("simple_query", ChainPatterns.simple_query()),
        ("verified_query", ChainPatterns.verified_query()),
        ("auto_refine", ChainPatterns.auto_refine()),
        ("balanced", ChainPatterns.balanced()),
    ]

    print(f"{'Pattern':<20} {'Latency (ms)':<15} {'Confidence':<15} {'Steps':<10}")
    print("-" * 60)

    for name, chain in patterns:
        dept = DemoDepartment(confidence_seed=0.75)
        orchestrator = ChainOrchestrator(dept, enable_tracing=False)

        result = await orchestrator.execute_chain(chain, "Test question")

        print(
            f"{name:<20} {result.stats.total_duration_ms:>13.1f} "
            f"{result.confidence:>14.2f}  {result.stats.completed_steps:>9}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  HoloLoom Chain Orchestrator Demonstrations".center(68) + "║")
    print("║" + "  Declarative Prompt Chaining System".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")

    demos = [
        ("1. Simple Query", demo_simple_query),
        ("2. Verified Query", demo_verified_query),
        ("3. Auto-Refine", demo_auto_refine),
        ("4. Iterative Improvement", demo_iterative_improve),
        ("5. Multi-Strategy", demo_multi_strategy),
        ("6. Research Pipeline", demo_research_pipeline),
        ("7. Custom Chain", demo_custom_chain),
        ("8. Performance Comparison", demo_performance_comparison),
    ]

    for i, (name, demo_func) in enumerate(demos):
        try:
            await demo_func()
        except Exception as e:
            logger.error(f"Demo failed: {e}", exc_info=True)
            print(f"\n✗ Demo failed: {e}")

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
