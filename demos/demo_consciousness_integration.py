"""
Demo: Consciousness Integration (Phase 1)
==========================================

Showcases the complete Phase 1 consciousness integration across all HoloLoom systems:
- Weaving Orchestrator with awareness context
- RAG with epistemic confidence
- Alignment with epistemic humility
- Agentic reasoning with multi-query tracking and early stopping

Usage:
    PYTHONPATH=. python demos/demo_consciousness_integration.py

Requirements:
    - HoloLoom installed with awareness layer
    - Optional: Ollama for LLM generation
"""

import asyncio
import logging
from typing import List

from hololoom.config import Config
from hololoom.protocols.types import Query, MemoryShard
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.rag import SimpleRAG
from hololoom.alignment import SafetyGuardrails
from hololoom.agentic import AgenticOrchestrator, ReasoningMode, create_agentic_orchestrator
from hololoom.memory.awareness_graph import AwarenessGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_shards() -> List[MemoryShard]:
    """Create test knowledge base about Thompson Sampling."""
    shards = [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian approach to the exploration-exploitation dilemma in reinforcement learning.",
            entities=["Thompson Sampling", "Bayesian", "exploration-exploitation", "reinforcement learning"],
            motifs=["definition", "bayesian_method"],
            timestamp=1.0,
            metadata={"source": "textbook"}
        ),
        MemoryShard(
            id="shard_2",
            text="Thompson Sampling maintains Beta distributions for each action, updating alpha/beta parameters based on rewards.",
            entities=["Thompson Sampling", "Beta distributions", "alpha", "beta", "rewards"],
            motifs=["algorithm", "parameters"],
            timestamp=2.0,
            metadata={"source": "paper"}
        ),
        MemoryShard(
            id="shard_3",
            text="Compared to epsilon-greedy, Thompson Sampling achieves better regret bounds in multi-armed bandit problems.",
            entities=["Thompson Sampling", "epsilon-greedy", "regret bounds", "multi-armed bandit"],
            motifs=["comparison", "performance"],
            timestamp=3.0,
            metadata={"source": "research"}
        ),
        MemoryShard(
            id="shard_4",
            text="Thompson Sampling is particularly effective when you need to balance exploration and exploitation naturally.",
            entities=["Thompson Sampling", "exploration", "exploitation"],
            motifs=["use_case", "effectiveness"],
            timestamp=4.0,
            metadata={"source": "tutorial"}
        ),
    ]
    return shards


async def demo_1_weaving_orchestrator():
    """Demo 1: Weaving Orchestrator with Awareness Context"""
    print("\n" + "="*80)
    print("DEMO 1: Weaving Orchestrator with Awareness Context")
    print("="*80)

    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # The orchestrator auto-creates awareness_layer if not provided
        query = Query(text="What is Thompson Sampling?")

        logger.info(f"Processing query: {query.text}")
        spacetime = await orchestrator.weave(query)

        # Display results
        print("\n[Results] Weaving Orchestrator:")
        print(f"  Query: {query.text}")
        print(f"  Confidence: {spacetime.confidence:.3f}")

        # Check awareness metadata
        if 'awareness' in spacetime.metadata:
            awareness = spacetime.metadata['awareness']
            print(f"\n[Awareness] Context:")
            print(f"  Activation Level: {awareness.get('activation_level', 0.0):.3f}")
            print(f"  Coherence: {awareness.get('coherence', 0.0):.3f}")
            print(f"  Active Nodes: {awareness.get('active_nodes', 0)}")
            print(f"  Shift Detected: {awareness.get('shift_detected', False)}")
            print(f"  Perception Time: {awareness.get('perception_time_ms', 0.0):.2f}ms")
        else:
            print("\n[Warning] Awareness context not available (awareness_layer may be disabled)")

        return spacetime


async def demo_2_rag_epistemic():
    """Demo 2: RAG with Epistemic Confidence"""
    print("\n" + "="*80)
    print("DEMO 2: RAG with Epistemic Confidence")
    print("="*80)

    # Create RAG system
    async with SimpleRAG() as rag:
        # Ingest knowledge
        print("\n📚 Ingesting knowledge...")
        await rag.ingest("Thompson Sampling is a Bayesian bandit algorithm.")
        await rag.ingest("It balances exploration and exploitation naturally.")

        # Query with epistemic confidence
        query = "What is Thompson Sampling?"
        logger.info(f"Querying: {query}")
        result = await rag.query(query, mode="direct")

        # Display results
        print(f"\n[Results] RAG:")
        print(f"  Query: {query}")
        print(f"  Response: {result.response[:200]}...")
        print(f"  Confidence: {result.confidence:.3f}")

        if result.epistemic_confidence is not None:
            print(f"\n[Epistemic] Confidence: {result.epistemic_confidence:.3f}")
            print(f"  (System's self-awareness of knowledge gaps)")

            # Interpret epistemic confidence
            if result.epistemic_confidence < 0.3:
                print(f"  [Warning] Very uncertain - requires more context")
            elif result.epistemic_confidence < 0.6:
                print(f"  [Caution] Moderate uncertainty")
            else:
                print(f"  [OK] High confidence in answer quality")
        else:
            print(f"\n[Warning] Epistemic confidence not available (awareness layer may be disabled)")

        return result


async def demo_3_alignment_epistemic_humility():
    """Demo 3: Alignment with Epistemic Humility"""
    print("\n" + "="*80)
    print("DEMO 3: Alignment with Epistemic Humility")
    print("="*80)

    from hololoom.alignment.safety_guardrails import ActionRequest, RiskLevel

    guardrails = SafetyGuardrails(enable_human_in_loop=False)

    # Test cases with different epistemic confidence levels
    test_cases = [
        ("execute_code", 0.9, "High confidence code execution"),
        ("execute_code", 0.5, "Moderate confidence code execution"),
        ("execute_code", 0.2, "Low confidence code execution"),
    ]

    print("\n[Safety] Testing Guardrails with Epistemic Humility:")

    for action, epistemic_conf, description in test_cases:
        request = ActionRequest(
            action=action,
            context={"code": "print('hello')"},
            timestamp=1.0
        )

        decision = guardrails.evaluate(
            request,
            epistemic_confidence=epistemic_conf
        )

        print(f"\n  {description}:")
        print(f"    Epistemic Confidence: {epistemic_conf:.3f}")
        print(f"    Risk Level: {decision.risk_level.value}")
        print(f"    Allowed: {decision.allowed}")
        print(f"    Reason: {decision.reason}")

        # Check if epistemic humility adjusted risk
        if 'epistemic_warning' in decision.metadata:
            print(f"    [Warning] Epistemic: {decision.metadata['epistemic_warning']}")


async def demo_4_agentic_multi_query():
    """Demo 4: Agentic Reasoning with Multi-Query Tracking"""
    print("\n" + "="*80)
    print("DEMO 4: Agentic Reasoning with Multi-Query Tracking")
    print("="*80)

    config = Config.fast()
    shards = create_test_shards()

    # Create agentic orchestrator
    agent = await create_agentic_orchestrator(
        config=config,
        shards=shards,
        enable_verification=True
    )

    # Test different reasoning modes
    queries_and_modes = [
        ("What is Thompson Sampling?", ReasoningMode.DIRECT),
        ("Is Thompson Sampling optimal for all problems?", ReasoningMode.VERIFY),
        ("Compare Thompson Sampling to other bandit algorithms", ReasoningMode.RESEARCH),
    ]

    for query_text, mode in queries_and_modes:
        print(f"\n{'─'*80}")
        print(f"Mode: {mode.value.upper()}")
        print(f"Query: {query_text}")
        print(f"{'─'*80}")

        query = Query(text=query_text)
        result = await agent.reason(query, mode=mode, max_steps=3)

        # Display results
        print(f"\n[Results] Agentic:")
        print(f"  Total Steps: {result.total_queries}")
        print(f"  Final Confidence: {result.spacetime.confidence:.3f}")

        if result.aggregated_epistemic_confidence is not None:
            print(f"  Aggregated Epistemic Confidence: {result.aggregated_epistemic_confidence:.3f}")
            print(f"  (Weighted average across all reasoning steps)")
        else:
            print(f"  [Warning] Epistemic confidence not available")

        # Show step-by-step epistemic confidence
        print(f"\n[Analysis] Step-by-Step Epistemic Confidence:")
        for i, step in enumerate(result.steps_taken):
            step_type = step.get('type', 'unknown')
            confidence = step.get('confidence', 0.0)
            epistemic = step.get('epistemic_confidence')

            if epistemic is not None:
                print(f"    Step {i+1} ({step_type}): confidence={confidence:.3f}, epistemic={epistemic:.3f}")
            else:
                print(f"    Step {i+1} ({step_type}): confidence={confidence:.3f}, epistemic=N/A")

        # Show verification results if available
        if result.verification:
            print(f"\n[Verification] Results:")
            print(f"  Verified: {result.verification.verified}")
            print(f"  Contradictions: {len(result.verification.contradictions)}")
            print(f"  Supporting Evidence: {len(result.verification.supporting_evidence)}")

    await agent.close()


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("HoloLoom Consciousness Integration Demo (Phase 1)")
    print("="*80)
    print("\nShowcasing epistemic awareness across all systems:")
    print("  1. Weaving Orchestrator - Awareness context injection")
    print("  2. RAG System - Epistemic confidence in retrieval")
    print("  3. Alignment Framework - Epistemic humility for safety")
    print("  4. Agentic Reasoning - Multi-query epistemic tracking")

    try:
        # Run all demos
        await demo_1_weaving_orchestrator()
        await demo_2_rag_epistemic()
        await demo_3_alignment_epistemic_humility()
        await demo_4_agentic_multi_query()

        print("\n" + "="*80)
        print("[SUCCESS] All demos completed successfully!")
        print("="*80)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n[ERROR] Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
