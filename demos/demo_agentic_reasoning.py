#!/usr/bin/env python3
"""
Demo: Agentic Intelligence Integration
=======================================
Shows all 4 reasoning modes with verification loops.

Usage:
    PYTHONPATH=. python demos/demo_agentic_reasoning.py
"""

import asyncio
from datetime import datetime
from hololoom.documentation.types import Query, MemoryShard
from hololoom.config import Config
from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.alignment.audit_trail import AuditTrail


def create_test_shards():
    """Create knowledge base about Thompson Sampling."""
    return [
        MemoryShard(
            id="ts_def",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                 "It maintains a probability distribution over the reward of each arm and "
                 "samples from these distributions to select actions.",
            episode="thompson_basics",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit"],
            motifs=["definition", "probability"]
        ),
        MemoryShard(
            id="ts_tradeoff",
            text="Thompson Sampling naturally balances exploration and exploitation through "
                 "its probabilistic sampling. However, it can be computationally expensive "
                 "for complex posterior distributions.",
            episode="thompson_tradeoffs",
            entities=["Thompson Sampling", "exploration", "exploitation"],
            motifs=["tradeoff", "performance"]
        ),
        MemoryShard(
            id="ucb_comparison",
            text="UCB (Upper Confidence Bound) provides deterministic guarantees but may "
                 "over-explore. Thompson Sampling is more sample-efficient in practice "
                 "but lacks worst-case guarantees.",
            episode="algorithm_comparison",
            entities=["UCB", "Thompson Sampling"],
            motifs=["comparison", "guarantees"]
        ),
        MemoryShard(
            id="ts_applications",
            text="Thompson Sampling is widely used in recommender systems, clinical trials, "
                 "and online advertising. It's particularly effective when prior knowledge "
                 "is available.",
            episode="real_world",
            entities=["Thompson Sampling", "applications"],
            motifs=["practical", "use_case"]
        ),
    ]


async def demo_direct_mode(agent):
    """Demo: DIRECT mode (single-pass answer)."""
    print("\n" + "="*80)
    print("MODE 1: DIRECT (Single-pass answer)")
    print("="*80)

    query = Query(text="What is Thompson Sampling?")
    print(f"\nQuery: {query.text}")

    start = datetime.now()
    result = await agent.reason(query, mode=ReasoningMode.DIRECT)
    duration = (datetime.now() - start).total_seconds() * 1000

    print(f"\nResult:")
    print(f"  Confidence: {result.spacetime.confidence:.3f}")
    print(f"  Steps: {result.total_queries}")
    print(f"  Duration: {duration:.1f}ms")
    print(f"  Tool used: {result.spacetime.metadata.get('tool_used')}")

    return result


async def demo_verify_mode(agent):
    """Demo: VERIFY mode (answer + verification)."""
    print("\n" + "="*80)
    print("MODE 2: VERIFY (Answer + verification loop)")
    print("="*80)

    query = Query(text="Is Thompson Sampling always optimal?")
    print(f"\nQuery: {query.text}")

    start = datetime.now()
    result = await agent.reason(
        query,
        mode=ReasoningMode.VERIFY,
        confidence_threshold=0.85,
        max_steps=3
    )
    duration = (datetime.now() - start).total_seconds() * 1000

    print(f"\nResult:")
    print(f"  Verified: {result.verification.verified}")
    print(f"  Initial confidence: {result.spacetime.confidence:.3f}")
    print(f"  Verification queries: {len(result.verification.verification_queries)}")

    if result.verification.contradictions:
        print(f"\n  Contradictions found:")
        for i, c in enumerate(result.verification.contradictions[:2], 1):
            print(f"    {i}. {c[:100]}...")

    if result.verification.supporting_evidence:
        print(f"\n  Supporting evidence:")
        for i, s in enumerate(result.verification.supporting_evidence[:2], 1):
            print(f"    {i}. {s[:100]}...")

    print(f"\n  Total steps: {result.total_queries}")
    print(f"  Duration: {duration:.1f}ms")

    return result


async def demo_research_mode(agent):
    """Demo: RESEARCH mode (multi-query exploration)."""
    print("\n" + "="*80)
    print("MODE 3: RESEARCH (Multi-query exploration)")
    print("="*80)

    query = Query(text="Compare Thompson Sampling vs UCB")
    print(f"\nQuery: {query.text}")

    start = datetime.now()
    result = await agent.reason(
        query,
        mode=ReasoningMode.RESEARCH,
        max_steps=5
    )
    duration = (datetime.now() - start).total_seconds() * 1000

    print(f"\nResult:")
    print(f"  Evidence gathered: {len(result.intent.evidence_gathered)} pieces")
    print(f"  Final confidence: {result.spacetime.confidence:.3f}")
    print(f"  Total queries: {result.total_queries}")
    print(f"  Duration: {duration:.1f}ms")

    print(f"\n  Research steps:")
    for i, step in enumerate(result.steps_taken[:-1], 1):  # Exclude synthesis
        print(f"    {i}. {step['query'][:60]}... (conf: {step['confidence']:.3f})")

    return result


async def demo_plan_execute_mode(agent):
    """Demo: PLAN_EXECUTE mode (goal decomposition)."""
    print("\n" + "="*80)
    print("MODE 4: PLAN_EXECUTE (Goal decomposition)")
    print("="*80)

    query = Query(text="Implement a Thompson Sampling bandit")
    print(f"\nQuery: {query.text}")

    start = datetime.now()
    result = await agent.reason(
        query,
        mode=ReasoningMode.PLAN_EXECUTE,
        max_steps=4
    )
    duration = (datetime.now() - start).total_seconds() * 1000

    print(f"\nResult:")
    print(f"  Goal: {result.intent.goal}")
    print(f"  Sub-goals: {len(result.intent.sub_goals)}")

    print(f"\n  Execution plan:")
    for i, sub_goal in enumerate(result.intent.sub_goals[:4], 1):
        print(f"    {i}. {sub_goal}")

    print(f"\n  Execution results:")
    for step in result.steps_taken[:-1]:  # Exclude synthesis
        if step['type'] == 'sub_goal':
            status = "✓" if step.get('completed') else "○"
            print(f"    {status} {step['goal'][:60]}... (conf: {step['confidence']:.3f})")

    print(f"\n  Total queries: {result.total_queries}")
    print(f"  Duration: {duration:.1f}ms")

    return result


async def main():
    """Run all demos."""
    print("="*80)
    print("Agentic Intelligence Demo - 4 Reasoning Modes")
    print("="*80)

    # Setup
    config = Config.fast()
    shards = create_test_shards()
    audit_trail = AuditTrail()

    # Create agentic orchestrator
    async with await create_agentic_orchestrator(
        config,
        shards,
        enable_verification=True,
        enable_goal_tracking=True,
        audit_trail=audit_trail
    ) as agent:

        # Run all modes
        await demo_direct_mode(agent)
        await demo_verify_mode(agent)
        await demo_research_mode(agent)
        await demo_plan_execute_mode(agent)

    # Summary
    print("\n" + "="*80)
    print("Summary: Reasoning Modes")
    print("="*80)
    print(f"""
    DIRECT:       Single-pass answer              (~150ms, confidence ≥ 0.8)
    VERIFY:       Answer + contradiction check    (~600ms, when claims need verification)
    RESEARCH:     Multi-query exploration         (~900ms, open-ended questions)
    PLAN_EXECUTE: Goal decomposition              (~750ms, multi-step tasks)

    Recommendation: Use DIRECT by default, VERIFY when confidence < 0.85
    """)

    # Audit trail summary
    print("\nAudit Trail:")
    print(f"  Total decisions logged: {len(audit_trail.logs)}")
    print(f"  Approved: {sum(1 for log in audit_trail.logs if log.outcome.value == 'approved')}")
    print(f"  Reasoning modes used: {set(log.metadata.get('mode') for log in audit_trail.logs if 'mode' in log.metadata)}")


if __name__ == "__main__":
    asyncio.run(main())