"""
Curiosity Engine Integration with Research Assistant
=====================================================

Shows how to integrate the Curiosity Engine into the Research Assistant chatbot
to provide proactive exploration suggestions in the sidebar.

Author: HoloLoom Team
Date: 2025-11-17
"""

import asyncio
from datetime import datetime, timedelta
from typing import List

try:
    from HoloLoom.memory.curiosity import CuriosityEngine, CuriosityConfig, ExplorationSuggestion
    from HoloLoom.memory.graph import KG, KGEdge
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This is a conceptual demo showing integration patterns.")
    print("In production, ensure all HoloLoom dependencies are installed.")
    exit(1)


class ResearchAssistantWithCuriosity:
    """
    Research Assistant with integrated Curiosity Engine.

    Provides proactive learning suggestions in addition to reactive Q&A.
    """

    def __init__(self):
        """Initialize Research Assistant with Curiosity Engine."""
        # Knowledge graph (in production, this would be the main HoloLoom KG)
        self.kg = KG()

        # Curiosity engine configuration
        curiosity_config = CuriosityConfig(
            max_suggestions=5,
            gap_importance_threshold=0.5,
            contradiction_score_threshold=0.6,
            trending_window_hours=24,
            enable_serendipity=True,
            serendipity_probability=0.2
        )

        self.curiosity_engine = CuriosityEngine(
            kg=self.kg,
            config=curiosity_config
        )

        # Session state
        self.conversation_history: List[dict] = []

    async def process_query(self, query: str, entities: List[str] = None) -> dict:
        """
        Process a user query (main chatbot function).

        Args:
            query: User question
            entities: Extracted entities (from orchestrator)

        Returns:
            Response dict with answer and suggestions
        """
        # Track query for curiosity engine
        self.curiosity_engine.track_query(
            query_text=query,
            entities=entities or [],
            timestamp=datetime.now()
        )

        # In production: get answer from HoloLoom orchestrator
        # For demo, we'll simulate
        answer = f"[Simulated answer to: {query}]"

        # Get proactive suggestions
        suggestions = await self.curiosity_engine.suggest_exploration(limit=3)

        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'answer': answer,
            'entities': entities,
            'timestamp': datetime.now(),
            'suggestions': suggestions
        })

        return {
            'answer': answer,
            'suggestions': suggestions,
            'entities': entities
        }

    async def get_sidebar_suggestions(self, limit: int = 5) -> List[ExplorationSuggestion]:
        """
        Get proactive exploration suggestions for sidebar display.

        Args:
            limit: Max suggestions to return

        Returns:
            List of suggestions
        """
        return await self.curiosity_engine.suggest_exploration(limit=limit)

    def render_sidebar_suggestions(self, suggestions: List[ExplorationSuggestion]):
        """
        Render suggestions in sidebar format (conceptual Streamlit code).

        This shows how suggestions would be displayed in the Research Assistant UI.
        """
        print("\n" + "="*70)
        print("💡 SUGGESTED EXPLORATION (Sidebar)")
        print("="*70)

        if not suggestions:
            print("\nNo suggestions at this time. Keep exploring!\n")
            return

        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n[Suggestion {i}]")
            print(f"📚 {suggestion.concept}")
            print(f"   Importance: {'⭐' * int(suggestion.importance * 5)}")
            print(f"\n   {suggestion.reason}\n")
            print(f"   💬 Try asking: \"{suggestion.suggested_query}\"")
            print(f"   🎯 Benefit: {suggestion.expected_benefit}")
            print(f"\n   [Explore Now] [Dismiss] [Remind Me Later]")
            print()


async def demo_research_session():
    """Simulate a research session with curiosity-driven exploration."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║    Research Assistant with Curiosity Engine - Session Demo        ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # Initialize assistant
    assistant = ResearchAssistantWithCuriosity()

    # Populate knowledge graph with some initial knowledge
    assistant.kg.add_edges([
        KGEdge("machine learning", "supervised learning", "HAS_TYPE"),
        KGEdge("machine learning", "neural networks", "USES"),
        KGEdge("neural networks", "backpropagation", "USES"),
        KGEdge("neural networks", "deep learning", "IS_A"),
        KGEdge("deep learning", "transformers", "USES"),
        KGEdge("transformers", "attention mechanism", "USES"),

        # Gap: reinforcement learning mentioned but not explored
        KGEdge("machine learning", "reinforcement learning", "HAS_TYPE"),

        # Gap: Python mentioned but not deeply connected
        KGEdge("programming", "python", "HAS_LANGUAGE"),
        KGEdge("python", "data science", "USED_IN"),
    ])

    print("Session Start: User begins research on machine learning\n")

    # ========================================================================
    # Query 1: Introduction to neural networks
    # ========================================================================

    print("\n" + "-"*70)
    print("USER QUERY #1")
    print("-"*70)

    result = await assistant.process_query(
        "What are neural networks?",
        entities=["neural networks"]
    )

    print(f"\n🗣️  User: {assistant.conversation_history[-1]['query']}")
    print(f"🤖 Assistant: {result['answer']}\n")

    # Show sidebar suggestions
    suggestions = await assistant.get_sidebar_suggestions(limit=3)
    assistant.render_sidebar_suggestions(suggestions)

    # ========================================================================
    # Query 2: Deep dive into neural networks (trending topic)
    # ========================================================================

    print("\n" + "-"*70)
    print("USER QUERY #2 (5 minutes later)")
    print("-"*70)

    await asyncio.sleep(0.1)  # Simulate time passing

    result = await assistant.process_query(
        "How does backpropagation work in neural networks?",
        entities=["neural networks", "backpropagation"]
    )

    print(f"\n🗣️  User: {assistant.conversation_history[-1]['query']}")
    print(f"🤖 Assistant: {result['answer']}\n")

    # ========================================================================
    # Query 3: Continue exploring neural networks (now trending)
    # ========================================================================

    print("\n" + "-"*70)
    print("USER QUERY #3 (10 minutes later)")
    print("-"*70)

    await asyncio.sleep(0.1)

    result = await assistant.process_query(
        "What are activation functions in neural networks?",
        entities=["neural networks", "activation functions"]
    )

    print(f"\n🗣️  User: {assistant.conversation_history[-1]['query']}")
    print(f"🤖 Assistant: {result['answer']}\n")

    # Now neural networks is trending - suggestions should adapt
    suggestions = await assistant.get_sidebar_suggestions(limit=5)

    print("\n📊 ADAPTIVE SUGGESTIONS (after trending topic detected)")
    assistant.render_sidebar_suggestions(suggestions)

    # ========================================================================
    # User accepts a suggestion
    # ========================================================================

    print("\n" + "-"*70)
    print("USER ACCEPTS SUGGESTION")
    print("-"*70)

    if suggestions:
        accepted_suggestion = suggestions[0]
        print(f"\n✅ User clicks: \"{accepted_suggestion.suggested_query}\"\n")

        result = await assistant.process_query(
            accepted_suggestion.suggested_query,
            entities=[accepted_suggestion.concept]
        )

        print(f"🗣️  User: {assistant.conversation_history[-1]['query']}")
        print(f"🤖 Assistant: {result['answer']}\n")

        print("✨ Curiosity-driven exploration successful!")
        print(f"   User discovered: {accepted_suggestion.concept}")
        print(f"   Benefit: {accepted_suggestion.expected_benefit}")

    # ========================================================================
    # Contradiction scenario
    # ========================================================================

    print("\n" + "-"*70)
    print("CONTRADICTION SCENARIO (2 weeks later)")
    print("-"*70)

    # Simulate time passing and user changing their mind
    assistant.curiosity_engine.contradiction_detector.add_memory({
        'id': 'm1',
        'content': 'Neural networks are easy to train and require minimal tuning',
        'timestamp': datetime.now() - timedelta(days=14),
        'entities': ['neural networks']
    })

    await asyncio.sleep(0.1)

    assistant.curiosity_engine.contradiction_detector.add_memory({
        'id': 'm2',
        'content': 'Neural networks are extremely difficult to train and require extensive hyperparameter tuning',
        'timestamp': datetime.now(),
        'entities': ['neural networks']
    })

    result = await assistant.process_query(
        "How hard is it to train neural networks?",
        entities=["neural networks"]
    )

    print(f"\n🗣️  User: {assistant.conversation_history[-1]['query']}")
    print(f"🤖 Assistant: {result['answer']}\n")

    # Suggestions should now include contradiction resolution
    suggestions = await assistant.get_sidebar_suggestions(limit=5)

    print("⚠️  CONTRADICTION DETECTED - Updated suggestions:")
    assistant.render_sidebar_suggestions(suggestions)

    # ========================================================================
    # Session Summary
    # ========================================================================

    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70 + "\n")

    stats = assistant.curiosity_engine.get_statistics()

    print(f"Total queries: {len(assistant.conversation_history)}")
    print(f"Unique concepts explored: {stats['unique_entities_accessed']}")
    print(f"\nTop trending topics:")
    for entity, count in stats['top_trending_entities'][:5]:
        print(f"  • {entity}: {count} queries")

    print(f"\nCuriosity suggestions generated: {stats['suggestions_cached']}")

    print("\n" + "="*70)
    print("Integration Benefits:")
    print("="*70)
    print("""
1. PROACTIVE LEARNING
   • Suggests gaps before user realizes them
   • Guides exploration naturally

2. CONTRADICTION AWARENESS
   • Detects when beliefs/facts change
   • Prompts reflection and clarification

3. PERSONALIZED PATHS
   • Adapts to user's trending interests
   • Suggests related concepts based on access patterns

4. SERENDIPITOUS DISCOVERY
   • Occasionally suggests unexplored areas
   • Prevents learning tunnel vision

5. ENGAGEMENT
   • Keeps users exploring beyond single questions
   • Creates continuous learning journey
    """)

    print("="*70)
    print("UI Integration Points:")
    print("="*70)
    print("""
Streamlit Sidebar (Research Assistant):

    st.sidebar.header("💡 Suggested Exploration")

    suggestions = await assistant.get_sidebar_suggestions(limit=3)

    for s in suggestions:
        with st.expander(f"{s.concept} ({s.importance:.0%} important)"):
            st.write(s.reason)
            st.caption(f"Expected benefit: {s.expected_benefit}")

            if st.button(f"Explore: {s.suggested_query}", key=s.concept):
                # Auto-fill chat with suggested query
                st.session_state.next_query = s.suggested_query
                st.rerun()

VS Code Extension:

    // Show suggestions in sidebar tree view
    vscode.window.createTreeView('hololoom.suggestions', {
        treeDataProvider: new SuggestionsProvider(suggestions)
    });

    // Click handler
    vscode.commands.registerCommand('hololoom.exploreSuggestion', (query) => {
        // Open chat and auto-fill query
        vscode.window.showInputBox({ value: query });
    });

Terminal UI:

    # Periodic suggestion display (every 5 queries)
    if query_count % 5 == 0:
        suggestions = await engine.suggest_exploration(limit=3)
        print("\\n💡 Suggested next steps:")
        for s in suggestions[:3]:
            print(f"  • {s.suggested_query}")
    """)


async def main():
    """Run the demo."""
    await demo_research_session()

    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("  1. Integrate CuriosityEngine into Research Assistant (Week 4)")
    print("  2. Add sidebar UI for suggestions")
    print("  3. Track user acceptance rate (metrics)")
    print("  4. Fine-tune importance thresholds based on feedback")
    print()


if __name__ == "__main__":
    asyncio.run(main())
