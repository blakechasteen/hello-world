#!/usr/bin/env python3
"""
Demo: Elle Game Engine Session Management

Demonstrates stateful conversation with persistent memory across multiple turns.
"""

import asyncio
from apps.elle_game_engine.session import (
    GameSession,
    InMemorySessionStore,
    JSONSessionStore,
)
from apps.elle_game_engine.models import (
    GameStateSnapshot,
    NPCState,
    PlayerState,
    WorldState,
    PlayerIntent,
    PlayerIntentType,
)
from apps.elle_game_engine.policy import GamePolicy
from apps.elle_game_engine.llm_client import create_llm_client


async def demo_session_workflow():
    """Demonstrate full session management workflow."""

    print("=" * 70)
    print("Elle Game Engine - Session Management Demo")
    print("=" * 70)
    print()

    # 1. Create session store
    print("1. Creating session store (in-memory)...")
    store = InMemorySessionStore()
    print(f"   ✓ Store created (type: {type(store).__name__})")
    print()

    # 2. Create session for player
    print("2. Creating session for player...")
    session = store.create_session(player_id="demo_player_123")
    print(f"   ✓ Session ID: {session.session_id}")
    print(f"   ✓ Player ID: {session.player_id}")
    print()

    # 3. Set up policy engine
    print("3. Initializing policy engine...")
    llm_client = create_llm_client("dummy")
    policy = GamePolicy(llm_client)
    print(f"   ✓ Policy ready (using DummyLLMClient for demo)")
    print()

    # 4. Simulate multi-turn conversation
    print("4. Simulating multi-turn conversation...")
    print()

    # Turn 1: Initial greeting
    print("   Turn 1: Player greets innkeeper")
    print("   " + "-" * 66)

    game_state_1 = GameStateSnapshot(
        scene_id="village_inn",
        npcs=[
            NPCState(
                id="innkeeper",
                name="Bob",
                role="innkeeper",
                mood="neutral",
                location="behind_bar",
            )
        ],
        player=PlayerState(name="Hero", location="village_inn"),
        world=WorldState(time_of_day="evening"),
        tags=["cozy", "quiet"],
    )

    player_intent_1 = PlayerIntent(
        type=PlayerIntentType.TALK_TO_NPC,
        target_npc_id="innkeeper",
        raw_input="Hello! Do you have a room available?",
    )

    # Get conversation context (empty for first turn)
    context_1 = session.get_conversation_context()
    print(f"   Context: {context_1}")

    # Get response from policy
    action_1 = await policy.decide(
        game_state_1,
        player_intent_1,
        conversation_context=context_1,
    )

    print(f"   Player: {player_intent_1.raw_input}")
    print(f"   Elle (mode={action_1.mode.value}):", end=" ")
    if action_1.dialogue:
        print(f"{action_1.dialogue[0].text}")
    else:
        print(action_1.debug_notes or "(no response)")

    # Update session
    session.add_exchange(
        player_query=player_intent_1.raw_input,
        elle_response=action_1.dialogue[0].text if action_1.dialogue else "(no response)",
        npc_id="innkeeper",
    )

    # Update NPC relationship
    if action_1.dialogue:
        tone = action_1.dialogue[0].tone or "neutral"
        reputation_delta = 5 if tone in ["warm", "grateful"] else 0
        session.update_npc_relationship(
            npc_id="innkeeper",
            reputation_delta=reputation_delta,
            mood=tone,
        )

    print(f"   ✓ Session updated (history: {len(session.conversation_history)} exchanges)")
    print()

    # Turn 2: Ask about price (Elle remembers previous context)
    print("   Turn 2: Player asks about price")
    print("   " + "-" * 66)

    player_intent_2 = PlayerIntent(
        type=PlayerIntentType.TALK_TO_NPC,
        target_npc_id="innkeeper",
        raw_input="How much for a room?",
    )

    # Get conversation context (now includes turn 1)
    context_2 = session.get_conversation_context()
    print(f"   Context: {context_2[:80]}...")

    action_2 = await policy.decide(
        game_state_1,
        player_intent_2,
        conversation_context=context_2,
    )

    print(f"   Player: {player_intent_2.raw_input}")
    print(f"   Elle (mode={action_2.mode.value}):", end=" ")
    if action_2.dialogue:
        print(f"{action_2.dialogue[0].text}")
    else:
        print(action_2.debug_notes or "(no response)")

    session.add_exchange(
        player_query=player_intent_2.raw_input,
        elle_response=action_2.dialogue[0].text if action_2.dialogue else "(no response)",
        npc_id="innkeeper",
    )

    print(f"   ✓ Session updated (history: {len(session.conversation_history)} exchanges)")
    print()

    # Turn 3: Confirm booking
    print("   Turn 3: Player confirms booking")
    print("   " + "-" * 66)

    player_intent_3 = PlayerIntent(
        type=PlayerIntentType.TALK_TO_NPC,
        target_npc_id="innkeeper",
        raw_input="I'll take it.",
    )

    context_3 = session.get_conversation_context()
    print(f"   Context: {len(context_3.split(chr(10)))} lines of history")

    action_3 = await policy.decide(
        game_state_1,
        player_intent_3,
        conversation_context=context_3,
    )

    print(f"   Player: {player_intent_3.raw_input}")
    print(f"   Elle (mode={action_3.mode.value}):", end=" ")
    if action_3.dialogue:
        print(f"{action_3.dialogue[0].text}")
    else:
        print(action_3.debug_notes or "(no response)")

    session.add_exchange(
        player_query=player_intent_3.raw_input,
        elle_response=action_3.dialogue[0].text if action_3.dialogue else "(no response)",
        npc_id="innkeeper",
    )

    # Simulate world change (room booked)
    if action_3.world_reaction:
        session.update_world_flags(action_3.world_reaction.flag_changes)
        print(f"   ✓ World flags updated: {action_3.world_reaction.flag_changes}")

    print(f"   ✓ Session updated (history: {len(session.conversation_history)} exchanges)")
    print()

    # 5. Save session
    print("5. Saving session to store...")
    store.update_session(session)
    print(f"   ✓ Session saved (store now has {len(store)} sessions)")
    print()

    # 6. Retrieve session (simulating service restart)
    print("6. Retrieving session (simulating service restart)...")
    retrieved = store.get_session(session.session_id)
    if retrieved:
        print(f"   ✓ Session retrieved")
        print(f"   ✓ Conversation history: {len(retrieved.conversation_history)} exchanges")
        print(f"   ✓ World flags: {dict(retrieved.world_flags)}")
        print(f"   ✓ NPC relationships: {list(retrieved.npc_relationships.keys())}")

        # Show NPC relationship details
        if "innkeeper" in retrieved.npc_relationships:
            rel = retrieved.npc_relationships["innkeeper"]
            print()
            print("   NPC Relationship Details:")
            print(f"     - NPC: innkeeper")
            print(f"     - Reputation: {rel.reputation}")
            print(f"     - Interactions: {rel.interactions}")
            print(f"     - Last Mood: {rel.last_mood}")
    print()

    # 7. Demo session serialization
    print("7. Testing session serialization...")
    session_dict = session.to_dict()
    deserialized = GameSession.from_dict(session_dict)
    print(f"   ✓ Serialized to dict ({len(session_dict)} keys)")
    print(f"   ✓ Deserialized successfully")
    print(f"   ✓ Data integrity: {deserialized.session_id == session.session_id}")
    print()

    # 8. Final statistics
    print("8. Session Statistics:")
    print(f"   Total exchanges: {len(session.conversation_history)}")
    print(f"   World flags: {len(session.world_flags)}")
    print(f"   NPC relationships: {len(session.npc_relationships)}")
    print(f"   Max history size: {session.max_history_size}")
    print(f"   Created at: {session.created_at}")
    print(f"   Last accessed: {session.last_accessed}")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


async def demo_file_based_storage():
    """Demonstrate file-based session storage."""

    print()
    print("=" * 70)
    print("File-Based Storage Demo")
    print("=" * 70)
    print()

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"1. Creating JSON session store in: {tmpdir}")
        store = JSONSessionStore(storage_path=tmpdir)
        print(f"   ✓ Store created")
        print()

        print("2. Creating session...")
        session = store.create_session(player_id="file_demo_player")
        print(f"   ✓ Session ID: {session.session_id}")
        print()

        print("3. Adding conversation history...")
        session.add_exchange("Hello", "Greetings!", "npc1")
        session.add_exchange("How are you?", "I am well!", "npc1")
        session.update_world_flags({"flag1": True})
        store.update_session(session)
        print(f"   ✓ Session saved to disk")
        print()

        print("4. Creating new store instance (simulating restart)...")
        store2 = JSONSessionStore(storage_path=tmpdir)
        print(f"   ✓ New store created")
        print(f"   ✓ Auto-loaded {len(store2)} sessions from disk")
        print()

        print("5. Retrieving session from new store...")
        retrieved = store2.get_session(session.session_id)
        if retrieved:
            print(f"   ✓ Session retrieved")
            print(f"   ✓ History: {len(retrieved.conversation_history)} exchanges")
            print(f"   ✓ Flags: {dict(retrieved.world_flags)}")
        print()

        print("=" * 70)
        print("File-Based Demo Complete!")
        print("=" * 70)


if __name__ == "__main__":
    # Run both demos
    asyncio.run(demo_session_workflow())
    asyncio.run(demo_file_based_storage())
