"""
Demonstration of emotion modeling and quest generation in Elle Game Engine.

Shows:
1. NPC emotional states changing based on player actions
2. Quest generation adapting to NPC emotions
3. Emotion-quest interaction loop

Usage:
    PYTHONPATH=. python demos/demo_emotion_quest.py

Created: 2025-11-16
"""

import asyncio
import time
from apps.elle_game_engine.emotion import EmotionalState, EmotionEngine
from apps.elle_game_engine.quest import QuestGenerator, QuestDifficulty
from apps.elle_game_engine.llm_client import DummyLLMClient


def print_separator(title: str = ""):
    """Print visual separator."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def print_emotional_state(npc_name: str, state: EmotionalState):
    """Print formatted emotional state."""
    emotion = state.get_emotion_label()
    tone = state.get_tone()

    print(f"🧠 {npc_name}'s Emotional State:")
    print(f"   Emotion: {emotion.upper()}")
    print(f"   Valence: {state.valence:+.2f} (negative ← 0 → positive)")
    print(f"   Arousal: {state.arousal:.2f} (calm ← 0.5 → excited)")
    print(f"   Dominance: {state.dominance:.2f} (submissive ← 0.5 → dominant)")
    print(f"   Trust: {state.trust:.2f} (distrust ← 0.5 → trust)")
    print(f"   Tone: {tone}")
    print()


def print_quest(quest):
    """Print formatted quest."""
    print(f"📜 Quest: {quest.title}")
    print(f"   Difficulty: {quest.difficulty.value.upper()}")
    print(f"   Description: {quest.description}")
    print(f"   Objectives:")
    for obj in quest.objectives:
        print(f"      • {obj.description}")
    if quest.reward:
        print(f"   Reward: {quest.reward.description}")
        if quest.reward.xp:
            print(f"      XP: {quest.reward.xp}")
        if quest.reward.gold:
            print(f"      Gold: {quest.reward.gold}")
    print()


async def demo_emotion_changes():
    """Demonstrate NPC emotions changing based on player actions."""
    print_separator("DEMO 1: NPC Emotions Reacting to Player Actions")

    # Create emotion engine and initial state
    engine = EmotionEngine()
    innkeeper_emotion = EmotionalState.from_emotion_label("neutral", trust=0.5)

    print("🎭 Scenario: You meet the innkeeper for the first time\n")
    print_emotional_state("Innkeeper", innkeeper_emotion)

    # Player helps the innkeeper
    print("💭 Player Action: You help the innkeeper clean tables")
    innkeeper_emotion = engine.process_player_action(
        innkeeper_emotion,
        action_type="help",
        npc_id="innkeeper",
        intensity=1.0,
    )
    print_emotional_state("Innkeeper", innkeeper_emotion)

    # Player gives a gift
    print("💭 Player Action: You give the innkeeper a rare ingredient")
    innkeeper_emotion = engine.process_player_action(
        innkeeper_emotion,
        action_type="gift",
        npc_id="innkeeper",
        intensity=1.5,
    )
    print_emotional_state("Innkeeper", innkeeper_emotion)

    # Time passes (decay)
    print("⏰ Time passes (10 seconds)...")
    time.sleep(0.1)  # Simulated time
    innkeeper_emotion.decay(time_elapsed_seconds=10.0, decay_rate=0.1)
    print_emotional_state("Innkeeper", innkeeper_emotion)

    # Player insults the innkeeper
    print("💭 Player Action: You make a rude comment about the inn")
    innkeeper_emotion = engine.process_player_action(
        innkeeper_emotion,
        action_type="insult",
        npc_id="innkeeper",
        intensity=1.0,
    )
    print_emotional_state("Innkeeper", innkeeper_emotion)

    return innkeeper_emotion


async def demo_quest_generation():
    """Demonstrate quest generation based on emotional states."""
    print_separator("DEMO 2: Quest Generation Based on NPC Emotions")

    # Create quest generator with dummy LLM
    llm_client = DummyLLMClient()

    # Override complete to return realistic quest JSON
    async def mock_complete_happy(prompt, **kwargs):
        return """{
            "title": "Gather Fresh Herbs",
            "description": "I'm grateful for your help! Could you gather some fresh herbs from the nearby forest? I need them for tonight's stew.",
            "difficulty": "easy",
            "objectives": [
                {"description": "Collect 5 fresh herbs from the forest", "optional": false, "target": 5},
                {"description": "Return to the innkeeper", "optional": false, "target": 1}
            ],
            "reward": {
                "description": "A warm meal and some gold",
                "xp": 100,
                "gold": 25,
                "items": ["hearty_stew"]
            },
            "time_limit_seconds": null,
            "emotional_rationale": "The innkeeper is happy and grateful, so they offer an easy, helpful quest with generous rewards."
        }"""

    async def mock_complete_angry(prompt, **kwargs):
        return """{
            "title": "Deal with the Bandits",
            "description": "You've caused me enough trouble! Make yourself useful and clear out the bandits that have been harassing my supply deliveries. Don't come back until it's done!",
            "difficulty": "hard",
            "objectives": [
                {"description": "Defeat the bandit leader", "optional": false, "target": 1},
                {"description": "Recover stolen supplies", "optional": false, "target": 10},
                {"description": "Return to the innkeeper with proof", "optional": false, "target": 1}
            ],
            "reward": {
                "description": "Reduced room rates (if you succeed)",
                "xp": 300,
                "gold": 50
            },
            "time_limit_seconds": 3600,
            "emotional_rationale": "The innkeeper is angry and distrustful, so they give a difficult, dangerous quest with strict time limits."
        }"""

    # Happy innkeeper quest
    print("😊 Scenario 1: Happy, Trusting Innkeeper\n")
    happy_state = EmotionalState.from_emotion_label("grateful", trust=0.9)
    print_emotional_state("Innkeeper", happy_state)

    llm_client.complete = mock_complete_happy
    generator = QuestGenerator(llm_client)

    quest = await generator.generate_quest(
        npc_id="innkeeper",
        npc_name="Martha",
        npc_role="innkeeper",
        emotional_state=happy_state,
        player_level=3,
        world_tension="calm",
    )

    print_quest(quest)

    # Angry innkeeper quest
    print_separator()
    print("😠 Scenario 2: Angry, Distrustful Innkeeper\n")
    angry_state = EmotionalState.from_emotion_label("angry", trust=0.2)
    print_emotional_state("Innkeeper", angry_state)

    llm_client.complete = mock_complete_angry
    generator = QuestGenerator(llm_client)

    quest = await generator.generate_quest(
        npc_id="innkeeper",
        npc_name="Martha",
        npc_role="innkeeper",
        emotional_state=angry_state,
        player_level=3,
        world_tension="tense",
    )

    print_quest(quest)


async def demo_emotion_quest_loop():
    """Demonstrate the feedback loop between emotions and quests."""
    print_separator("DEMO 3: Emotion-Quest Feedback Loop")

    print("🔄 Scenario: Complete interaction cycle with NPC\n")

    # Initialize
    engine = EmotionEngine()
    merchant_emotion = EmotionalState.from_emotion_label("neutral", trust=0.5)

    print("🏪 You approach a merchant in the town square\n")
    print_emotional_state("Merchant", merchant_emotion)

    # Step 1: Player greets respectfully
    print("💭 Player Action: You greet the merchant politely")
    merchant_emotion = engine.process_player_action(
        merchant_emotion,
        action_type="compliment",
        npc_id="merchant",
    )
    print_emotional_state("Merchant", merchant_emotion)

    # Step 2: Merchant offers quest (easy because of positive emotion)
    print("📜 The merchant offers you a quest:")
    print("   'Since you seem trustworthy, could you deliver this package?'\n")

    # Get emotion modifiers
    modifiers = engine.get_emotion_modifiers(merchant_emotion)
    print(f"🎲 Emotion Modifiers (affect game mechanics):")
    print(f"   Price Multiplier: {modifiers['price_multiplier']:.2f}x (lower is better)")
    print(f"   Quest Difficulty: {modifiers['quest_difficulty_modifier']:.2f}x (lower is easier)")
    print(f"   Hint Generosity: {modifiers['hint_generosity']:.2f} (higher = more helpful)")
    print()

    # Step 3: Player completes quest successfully
    print("✅ You complete the quest successfully\n")
    print("💭 Player Action: Return and deliver package")
    merchant_emotion = engine.process_player_action(
        merchant_emotion,
        action_type="help",
        npc_id="merchant",
        intensity=1.5,
    )
    print_emotional_state("Merchant", merchant_emotion)

    # Step 4: Merchant now offers harder quest with better rewards
    print("📜 The merchant is impressed and offers a more important quest:")
    print("   'I trust you now. Can you help me negotiate with a difficult supplier?'\n")

    modifiers = engine.get_emotion_modifiers(merchant_emotion)
    print(f"🎲 Updated Emotion Modifiers:")
    print(f"   Price Multiplier: {modifiers['price_multiplier']:.2f}x (lower is better)")
    print(f"   Quest Difficulty: {modifiers['quest_difficulty_modifier']:.2f}x")
    print(f"   Hint Generosity: {modifiers['hint_generosity']:.2f}")
    print()


async def demo_emotion_context_in_prompts():
    """Demonstrate how emotions inject into LLM prompts."""
    print_separator("DEMO 4: Emotion Context in LLM Prompts")

    print("🤖 How NPC emotions affect dialogue generation\n")

    engine = EmotionEngine()

    # Show different emotional contexts
    emotions = [
        ("happy", EmotionalState.from_emotion_label("happy", trust=0.9)),
        ("angry", EmotionalState.from_emotion_label("angry", trust=0.2)),
        ("sad", EmotionalState.from_emotion_label("sad", trust=0.4)),
        ("suspicious", EmotionalState.from_emotion_label("suspicious", trust=0.3)),
    ]

    for label, state in emotions:
        context = engine.generate_emotion_context(state)
        print(f"📝 {label.upper()} NPC:")
        print(f"   LLM Context: '{context}'")
        print()


async def main():
    """Run all demonstrations."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          ELLE GAME ENGINE: EMOTION & QUEST DEMONSTRATION            ║
║                                                                      ║
║  This demo shows how NPC emotions change based on player actions    ║
║  and how those emotions affect quest generation and difficulty      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run demonstrations
    await demo_emotion_changes()
    await asyncio.sleep(1)

    await demo_quest_generation()
    await asyncio.sleep(1)

    await demo_emotion_quest_loop()
    await asyncio.sleep(1)

    await demo_emotion_context_in_prompts()

    print_separator("DEMONSTRATION COMPLETE")

    print("""
📚 Key Takeaways:

1. **Emotion Modeling**: NPCs have rich emotional states (valence, arousal, dominance, trust)
   that change based on player actions and decay over time.

2. **Quest Generation**: Quests adapt to NPC emotions:
   - Happy NPCs give easier, more rewarding quests
   - Angry NPCs give harder, more demanding quests
   - Trust affects quest difficulty and rewards

3. **Feedback Loop**: Completing quests improves NPC emotions, unlocking better quests

4. **Game Mechanics**: Emotions affect prices, quest difficulty, and NPC helpfulness

5. **LLM Integration**: Emotional context is injected into prompts for authentic dialogue

🎮 Try it yourself:
   - Run the FastAPI service: python apps/elle_game_engine/service.py
   - Test the quest endpoint: POST /elle/game/quest/generate
   - See how different emotions generate different quests!

📖 See tests for more examples:
   - apps/elle_game_engine/tests/test_emotion.py (15+ tests)
   - apps/elle_game_engine/tests/test_quest.py (15+ tests)
    """)


if __name__ == "__main__":
    asyncio.run(main())
