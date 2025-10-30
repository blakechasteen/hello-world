#!/usr/bin/env python3
"""
Test StrategySelector - Intelligent Panel Selection
===================================================
Validates intent detection and panel generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

from HoloLoom.visualization.strategy import StrategySelector, QueryIntent, UserPreferences
from HoloLoom.visualization.dashboard import ComplexityLevel


@dataclass
class MockTrace:
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    duration_ms: float = 150.0
    stage_durations: Dict[str, float] = field(default_factory=lambda: {
        "pattern_selection": 5.0,
        "retrieval": 50.0,
        "convergence": 30.0,
        "tool_execution": 60.0
    })
    threads_activated: List[str] = field(default_factory=lambda: ["thread1", "thread2"])
    errors: List = field(default_factory=list)


@dataclass
class MockSpacetime:
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: MockTrace
    complexity: ComplexityLevel = ComplexityLevel.FAST
    metadata: Dict = field(default_factory=lambda: {
        'semantic_cache': {'enabled': True}
    })


def test_intent_detection():
    """Test query intent detection."""
    print("=" * 70)
    print("TEST 1: Intent Detection")
    print("=" * 70)

    selector = StrategySelector()

    test_cases = [
        ("What is Thompson Sampling?", QueryIntent.FACTUAL),
        ("How does the orchestrator work?", QueryIntent.EXPLORATORY),
        ("Compare FAST vs FUSED mode", QueryIntent.COMPARISON),
        ("Why did the query fail?", QueryIntent.DEBUGGING),
        ("How to improve performance?", QueryIntent.OPTIMIZATION),
    ]

    passed = 0
    for query, expected_intent in test_cases:
        spacetime = MockSpacetime(
            query_text=query,
            response="Mock response",
            tool_used="answer",
            confidence=0.85,
            trace=MockTrace()
        )

        strategy = selector.select(spacetime)
        detected_intent = selector.analyze_query(spacetime).intent

        status = "✅" if detected_intent == expected_intent else "❌"
        print(f"{status} '{query[:40]}...'")
        print(f"   Expected: {expected_intent.value}")
        print(f"   Detected: {detected_intent.value}")

        if detected_intent == expected_intent:
            passed += 1

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_panel_generation():
    """Test panel generation for different intents."""
    print("\n" + "=" * 70)
    print("TEST 2: Panel Generation")
    print("=" * 70)

    selector = StrategySelector()

    # Exploratory query
    spacetime = MockSpacetime(
        query_text="How does the weaving orchestrator work?",
        response="The orchestrator coordinates...",
        tool_used="answer",
        confidence=0.92,
        trace=MockTrace()
    )

    strategy = selector.select(spacetime)

    print(f"\nQuery: '{spacetime.query_text}'")
    print(f"Intent: EXPLORATORY")
    print(f"Layout: {strategy.layout_type.value}")
    print(f"Panels: {len(strategy.panels)}")

    print("\nGenerated Panels:")
    for i, panel in enumerate(strategy.panels, 1):
        print(f"  {i}. {panel.title} ({panel.type.value}, priority={panel.priority})")

    # Should have timeline and network for exploratory
    has_timeline = any(p.title == "Execution Timeline" for p in strategy.panels)
    has_network = any(p.title == "Knowledge Threads" for p in strategy.panels)

    print(f"\n✅ Has Timeline: {has_timeline}")
    print(f"✅ Has Network: {has_network}")

    return has_timeline and has_network


def test_narrative_ordering():
    """Test narrative flow ordering."""
    print("\n" + "=" * 70)
    print("TEST 3: Narrative Ordering")
    print("=" * 70)

    selector = StrategySelector()

    spacetime = MockSpacetime(
        query_text="Debug the slow query performance",
        response="Performance issue found...",
        tool_used="answer",
        confidence=0.75,
        trace=MockTrace()
    )

    strategy = selector.select(spacetime)

    print(f"\nQuery: '{spacetime.query_text}'")
    print(f"Intent: DEBUGGING")
    print("\nNarrative Order (Hook → Context → Mechanism → Conclusion):")

    for i, panel in enumerate(strategy.panels, 1):
        role = "HOOK" if panel.priority >= 9 else "CONTEXT" if panel.priority >= 7 else "MECHANISM" if panel.priority >= 5 else "CONCLUSION"
        print(f"  {i}. [{role:10s}] {panel.title} (priority={panel.priority})")

    # High priority panels should be first (hooks)
    first_panel_priority = strategy.panels[0].priority
    is_hook_first = first_panel_priority >= 9

    print(f"\n✅ Hook first: {is_hook_first} (priority={first_panel_priority})")

    return is_hook_first


def test_user_preferences():
    """Test user preference application."""
    print("\n" + "=" * 70)
    print("TEST 4: User Preferences")
    print("=" * 70)

    # Hide timeline panels
    from HoloLoom.visualization.dashboard import PanelType
    prefs = UserPreferences(
        hidden_panels=[PanelType.TIMELINE],
        preferred_panels=[PanelType.METRIC]
    )

    selector = StrategySelector(user_prefs=prefs)

    spacetime = MockSpacetime(
        query_text="How does this work?",
        response="It works by...",
        tool_used="answer",
        confidence=0.88,
        trace=MockTrace()
    )

    strategy = selector.select(spacetime)

    # Check no timeline panels
    has_timeline = any(p.type == PanelType.TIMELINE for p in strategy.panels)

    print(f"Hidden panels: {[p.value for p in prefs.hidden_panels]}")
    print(f"Generated {len(strategy.panels)} panels")
    print(f"\n✅ Timeline hidden: {not has_timeline}")

    return not has_timeline


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("STRATEGY SELECTOR TEST SUITE")
    print("=" * 70 + "\n")

    results = []

    results.append(("Intent Detection", test_intent_detection()))
    results.append(("Panel Generation", test_panel_generation()))
    results.append(("Narrative Ordering", test_narrative_ordering()))
    results.append(("User Preferences", test_user_preferences()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed")


if __name__ == "__main__":
    main()
