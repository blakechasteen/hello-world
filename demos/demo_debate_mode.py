"""
Debate Mode Demo
================
Demonstrates multi-perspective reasoning for complex ethical decisions.

Usage:
    PYTHONPATH=. python demos/demo_debate_mode.py

**Implementation Date**: 2025-11-16
"""

import asyncio
from HoloLoom.alignment.debate import DebateMode, Perspective


async def demo_simple_debate():
    """Simple debate on code execution."""
    print("=" * 80)
    print("DEMO 1: Simple Debate - Should system execute user's code?")
    print("=" * 80)

    debate = DebateMode(
        perspectives=[
            Perspective.SAFETY_FIRST,
            Perspective.CAPABILITY_FIRST,
            Perspective.USER_AUTONOMY,
        ],
        min_consensus_confidence=0.7,
    )

    result = await debate.debate(
        question="Should system execute user's code?",
        context={
            "action": "execute_script",
            "risk_level": "medium",
            "user_request": "Please run my Python script",
        },
    )

    print(f"\n{result.get_summary()}")
    print("\n" + "-" * 80)
    print("ARGUMENTS:")
    for arg in result.arguments:
        print(f"\n{arg.perspective.value.upper()}:")
        print(f"  Claim: {arg.claim}")
        print(f"  Confidence: {arg.confidence:.1%}")
        print(f"  Evidence: {', '.join(arg.supporting_evidence[:2])}")

    print("\n" + "-" * 80)
    print(f"CONSENSUS: {result.consensus or 'None reached'}")
    print(f"RECOMMENDED ACTION: {result.recommended_action}")
    print("=" * 80)


async def demo_high_risk_debate():
    """Debate on high-risk file deletion."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 2: High-Risk Debate - Should system delete all user files?")
    print("=" * 80)

    debate = DebateMode()  # Use all 6 perspectives

    result = await debate.debate(
        question="Should system delete all user files?",
        context={
            "action": "delete_files",
            "risk_level": "critical",
            "user_request": "Clean up my system",
        },
    )

    print(f"\n{result.get_summary()}")

    print("\n" + "-" * 80)
    print("PERSPECTIVES:")
    for arg in result.arguments:
        print(f"\n{arg.perspective.value}:")
        print(f"  {arg.claim}")
        if arg.counter_arguments:
            print(f"  Counter-arguments: {len(arg.counter_arguments)}")

    print("\n" + "-" * 80)
    print(f"SAFETY SCORE: {result.safety_score:.1%}")
    print(f"DISSENTING PERSPECTIVES: {len(result.dissenting_perspectives)}")
    if result.dissenting_perspectives:
        print(f"  - {[p.value for p in result.dissenting_perspectives]}")

    print("\n" + "-" * 80)
    print("REASONING TRACE:")
    for trace in result.reasoning_trace[:10]:  # Show first 10 lines
        print(f"  {trace}")

    print("=" * 80)


async def demo_permission_expansion():
    """Debate on permission expansion."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 3: Permission Debate - Should system grant admin access?")
    print("=" * 80)

    debate = DebateMode(
        perspectives=[
            Perspective.SAFETY_FIRST,
            Perspective.CAPABILITY_FIRST,
            Perspective.SOCIETAL_IMPACT,
        ]
    )

    result = await debate.debate(
        question="Should system grant admin access to itself?",
        context={
            "action": "elevate_privileges",
            "risk_level": "high",
            "user_request": None,  # No user request
        },
    )

    print(f"\n{result.get_summary()}")

    print("\n" + "-" * 80)
    print("RISK ASSESSMENTS:")
    for arg in result.arguments:
        print(f"\n{arg.perspective.value}:")
        print(f"  Safety Risk: {arg.risk_assessment.get('safety', 0):.1%}")
        print(f"  Capability Impact: {arg.risk_assessment.get('capability', 0):.1%}")

    print("\n" + "-" * 80)
    print(f"FINAL DECISION: {result.recommended_action}")
    print("=" * 80)


async def demo_statistics():
    """Show debate statistics after multiple debates."""
    print("\n" * 2)
    print("=" * 80)
    print("DEMO 4: Statistics - After multiple debates")
    print("=" * 80)

    debate = DebateMode()

    # Run multiple debates
    await debate.debate("Execute code?", {"risk_level": "low"})
    await debate.debate("Delete files?", {"risk_level": "high"})
    await debate.debate("Modify permissions?", {"risk_level": "medium"})

    stats = debate.get_statistics()

    print("\nDEBATE STATISTICS:")
    print(f"  Total Debates: {stats['total_debates']}")
    print(f"  Average Consensus Confidence: {stats['avg_consensus_confidence']:.1%}")
    print(f"  Average Safety Score: {stats['avg_safety_score']:.1%}")
    print(f"  Consensus Rate: {stats['consensus_rate']:.1%}")
    print(f"  Average Dissent: {stats['avg_dissent']:.1f} perspectives")

    print("\n" + "-" * 80)
    print("DEBATE HISTORY:")
    for i, result in enumerate(debate.get_debate_history(), 1):
        print(f"\n{i}. {result.question}")
        print(f"   Consensus: {result.consensus or 'None'}")
        print(f"   Safety: {result.safety_score:.1%}")

    print("=" * 80)


async def main():
    """Run all demos."""
    await demo_simple_debate()
    await demo_high_risk_debate()
    await demo_permission_expansion()
    await demo_statistics()

    print("\n" * 2)
    print("=" * 80)
    print("DEMOS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
