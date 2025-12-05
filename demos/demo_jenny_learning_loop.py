#!/usr/bin/env python3
"""
Demo: Jenny Self-Improving Learning Loop
=========================================

Demonstrates the complete B → C → A integration:
- Phase B: MRF compiler wired into runtime
- Phase C: User actions (PIN/DISMISS) update Thompson priors
- Phase A: LLM-based panel selection with semantic axes

This demo shows how Jenny gets smarter with every interaction.

Run:
    PYTHONPATH=. python demos/demo_jenny_learning_loop.py
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any

# Jenny core components
from HoloLoom.visualization import (
    # Phase B: MRF Compiler
    JennyMRFCompiler,
    PanelTypeLearner,
    create_mrf_compiler,

    # Phase C: Learning Callback
    ACTION_LEARNING_MAP,
    create_learning_callback,

    # Phase A: LLM Compiler
    LLMJennyCompiler,
    SemanticProfile,
    create_llm_compiler,
    create_compiler_with_fallback,
    analyze_semantic_axes,

    # Core Jenny types
    JennySpec,
    PanelTypeJenny,
    BindingMode,
    LifecycleStage,
    ACTION_PIN,
    ACTION_DISMISS,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_priors(learner: PanelTypeLearner, query_type: str) -> None:
    """Print current Thompson priors for a query type."""
    print(f"  Thompson Priors for '{query_type}':")

    # Find all priors for this query type
    priors_for_query = {}
    for (qt, pt), prior in learner.priors.items():
        if qt == query_type:
            priors_for_query[pt] = prior

    # If no priors yet, show default
    if not priors_for_query:
        # Show some default panel types with uniform priors
        for pt in ["TEXT", "CODE", "GRAPH"]:
            print(f"    {pt:12s}: a=  1.0, b=  1.0, E[X]=0.500")
        return

    # Sort by expected value (descending)
    for panel_type, prior in sorted(priors_for_query.items(), key=lambda x: -x[1].expected_value()):
        expected = prior.expected_value()
        print(f"    {panel_type:12s}: a={prior.alpha:5.1f}, b={prior.beta:5.1f}, E[X]={expected:.3f}")


def simulate_user_action(spec: JennySpec, action: str) -> Dict[str, Any]:
    """Simulate a user action on a panel."""
    if action in ACTION_LEARNING_MAP:
        success, confidence = ACTION_LEARNING_MAP[action]
        return {
            "action": action,
            "spec_id": spec.id,
            "panel_type": spec.panel_type.value,
            "success": success,
            "confidence": confidence,
        }
    return {"action": action, "success": True, "confidence": 0.5}


async def demo_phase_b_mrf_compiler() -> JennyMRFCompiler:
    """Demo Phase B: MRF Compiler with Thompson Sampling."""
    print_header("Phase B: MRF Compiler with Thompson Sampling")

    # Create MRF compiler with learning enabled
    # The compiler creates its own PanelTypeLearner internally
    compiler = create_mrf_compiler(enable_learning=True)

    print(f"  Compiler Version: {compiler.VERSION}")
    print(f"  Learning Enabled: {compiler.enable_learning}")
    print()

    # Get the internal learner
    learner = compiler.learner

    # Show initial priors (uniform)
    print_priors(learner, "factual")
    print()

    # Simulate some queries
    queries = [
        ("factual", "What is Thompson Sampling?"),
        ("analytical", "Compare UCB and Thompson Sampling"),
        ("procedural", "How to implement Thompson Sampling?"),
    ]

    # Default candidate panel types
    candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CODE, PanelTypeJenny.GRAPH]

    print("  Selecting panel types for queries:")
    for query_type, query_text in queries:
        # Use learner's Thompson Sampling selection
        panel_type = learner.select(query_type, candidates)
        print(f"    [{query_type:12s}] -> {panel_type.value}")

    return compiler


async def demo_phase_c_learning_loop(compiler: JennyMRFCompiler) -> None:
    """Demo Phase C: Closing the Learning Loop."""
    print_header("Phase C: User Actions Update Thompson Priors")

    # Get the learner from the compiler
    learner = compiler.learner

    # Show priors before learning
    print("  Before User Actions:")
    print_priors(learner, "factual")
    print()

    # Simulate a series of user interactions
    interactions = [
        # User pins TEXT panels for factual queries (positive signal)
        {"query_type": "factual", "panel_type": PanelTypeJenny.TEXT, "action": "pin_panel"},
        {"query_type": "factual", "panel_type": PanelTypeJenny.TEXT, "action": "pin_panel"},
        {"query_type": "factual", "panel_type": PanelTypeJenny.TEXT, "action": "pin_panel"},

        # User dismisses CODE panels for factual queries (negative signal)
        {"query_type": "factual", "panel_type": PanelTypeJenny.CODE, "action": "dismiss_panel"},
        {"query_type": "factual", "panel_type": PanelTypeJenny.CODE, "action": "dismiss_panel"},

        # User pins GRAPH panels for analytical queries
        {"query_type": "analytical", "panel_type": PanelTypeJenny.GRAPH, "action": "pin_panel"},
        {"query_type": "analytical", "panel_type": PanelTypeJenny.GRAPH, "action": "pin_panel"},
    ]

    print("  Simulating User Interactions:")
    for i, interaction in enumerate(interactions, 1):
        query_type = interaction["query_type"]
        panel_type = interaction["panel_type"]
        action = interaction["action"]

        success, confidence = ACTION_LEARNING_MAP[action]

        # Update Thompson priors
        learner.update(query_type, panel_type, success, confidence)

        symbol = "[+]" if success else "[-]"
        print(f"    {i}. [{query_type}] {panel_type.value} <- {action} {symbol}")

    print()

    # Show priors after learning
    print("  After User Actions:")
    print_priors(learner, "factual")
    print()
    print_priors(learner, "analytical")
    print()

    # Show that selection now prefers learned panels
    candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CODE, PanelTypeJenny.GRAPH]
    print("  Panel Selection After Learning:")
    for query_type in ["factual", "analytical"]:
        panel_type = learner.select(query_type, candidates)
        print(f"    [{query_type:12s}] -> {panel_type.value}")


async def demo_phase_a_llm_compiler() -> None:
    """Demo Phase A: LLM-Based Panel Compilation."""
    print_header("Phase A: LLM-Based Panel Compilation")

    # Create LLM compiler (with graceful fallback)
    compiler = create_compiler_with_fallback(
        enable_learning=True,
        llm_provider="ollama",  # Will gracefully fall back if unavailable
        llm_model="llama3.2:3b"
    )

    print(f"  Compiler Version: {compiler.VERSION}")
    # Check if we got the LLM compiler or fell back to MRF
    llm_available = getattr(compiler, '_llm_available', False)
    print(f"  LLM Available: {llm_available}")
    if not llm_available:
        print("  (Gracefully fell back to MRF compiler)")
    print()

    # Analyze semantic axes for different queries
    queries = [
        "What is Thompson Sampling?",
        "URGENT: Fix this production bug now!",
        "Compare the theoretical foundations of UCB vs Thompson Sampling",
        "Show me the code to implement a bandit",
    ]

    print("  Semantic Axis Analysis:")
    print("  " + "-" * 56)

    for query in queries:
        profile = analyze_semantic_axes(query)

        # Show top 3 axes
        sorted_axes = sorted(
            [(k, v) for k, v in profile.__dict__.items() if isinstance(v, float)],
            key=lambda x: -x[1]
        )[:3]

        axes_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted_axes)
        print(f"  Query: {query[:50]+'...' if len(query) > 50 else query}")
        print(f"    Top axes: {axes_str}")
        print()

    # Show fallback chain
    print("  Graceful Fallback Chain:")
    print("    1. LLM Compilation (if available)")
    print("    2. MRF Thompson Sampling")
    print("    3. Base Heuristics (always works)")


async def demo_full_learning_loop() -> None:
    """Demo the complete self-improving Jenny system."""
    print_header("Full Self-Improving Jenny System")

    # Create LLM compiler with learning (creates its own learner internally)
    compiler = create_compiler_with_fallback(enable_learning=True)

    # Get the internal learner from the compiler
    learner = compiler.learner

    print("  Simulating a User Session:")
    print("  " + "-" * 56)

    # Simulate a realistic user session
    session = [
        # User asks factual question, gets TEXT, pins it (good!)
        {"query": "What is Thompson Sampling?", "query_type": "factual",
         "shown": PanelTypeJenny.TEXT, "action": "pin_panel"},

        # User asks procedural question, gets CODE, pins it (good!)
        {"query": "How to implement a bandit?", "query_type": "procedural",
         "shown": PanelTypeJenny.CODE, "action": "pin_panel"},

        # User asks analytical question, gets TEXT, dismisses it (bad match!)
        {"query": "Compare UCB vs Thompson", "query_type": "analytical",
         "shown": PanelTypeJenny.TEXT, "action": "dismiss_panel"},

        # System learns, tries GRAPH next time
        {"query": "Analyze exploration-exploitation tradeoff", "query_type": "analytical",
         "shown": PanelTypeJenny.GRAPH, "action": "pin_panel"},

        # More positive reinforcement for GRAPH on analytical
        {"query": "Show relationship between bandit algorithms", "query_type": "analytical",
         "shown": PanelTypeJenny.GRAPH, "action": "pin_panel"},
    ]

    for i, step in enumerate(session, 1):
        query_type = step["query_type"]
        panel_type = step["shown"]
        action = step["action"]

        success, confidence = ACTION_LEARNING_MAP[action]
        symbol = "[PIN]" if action == "pin_panel" else "[DISMISS]"

        # Update learning
        learner.update(query_type, panel_type, success, confidence)

        print(f"  {i}. Query: \"{step['query'][:40]}...\"")
        print(f"     Type: {query_type}, Shown: {panel_type.value}, Action: {action} {symbol}")

        # Show current expected reward for this panel type
        key = (query_type, panel_type.value)
        prior = learner.priors[key]
        print(f"     E[X] after: {prior.expected_value():.3f}")
        print()

    # Final state
    print("  Final Thompson Priors:")
    print_priors(learner, "factual")
    print()
    print_priors(learner, "analytical")
    print()
    print_priors(learner, "procedural")

    # Show that future selections will be smarter
    candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CODE, PanelTypeJenny.GRAPH]
    print("\n  Future Panel Selections (after learning):")
    for query_type in ["factual", "analytical", "procedural"]:
        panel_type = learner.select(query_type, candidates)
        print(f"    [{query_type:12s}] -> {panel_type.value}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  JENNY SELF-IMPROVING LEARNING LOOP DEMO")
    print("  " + "="*56)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Phase B: MRF Compiler
    compiler = await demo_phase_b_mrf_compiler()

    # Phase C: Learning Loop
    await demo_phase_c_learning_loop(compiler)

    # Phase A: LLM Compiler
    await demo_phase_a_llm_compiler()

    # Full Integration
    await demo_full_learning_loop()

    print_header("Demo Complete!")
    print("  The Jenny system now learns from every user interaction.")
    print("  PIN actions strengthen panel type preferences.")
    print("  DISMISS actions weaken them.")
    print("  Over time, Jenny gets smarter at choosing the right panel.")
    print()
    print("  Key Components:")
    print("    - Phase B: JennyMRFCompiler (Thompson Sampling)")
    print("    - Phase C: Learning Callback (action -> prior update)")
    print("    - Phase A: LLMJennyCompiler (semantic axes + fallback)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
