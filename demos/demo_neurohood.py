"""
NeuroHood Demo: Consciousness-Level Neighborhood Simulator

Demonstrates:
1. Bootstrapping a neighborhood with 3 residents
2. Resident internal dialogue (consciousness)
3. Strange loop detection (consciousness moments)
4. Social interaction with emergent drama
5. Consciousness visualization modes

Run:
    PYTHONPATH=. python demos/demo_neurohood.py
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NeuroHood import NeuroHood, NeuroHoodConfig
from NeuroHood.consciousness import ConsciousnessView


async def main():
    """Run NeuroHood demo."""

    print("\n" + "="*70)
    print("  NeuroHood: Consciousness-Level Neighborhood Simulator")
    print("  'The Sims on steroids, ultra marathon running, and DMTxLSD'")
    print("="*70 + "\n")

    # ========== 1. Initialize NeuroHood ==========

    print("[ Step 1: Initializing NeuroHood ]")
    print("-" * 70)

    config = NeuroHoodConfig.prototype()  # 3 residents, simplified

    print(f"  Simulation Mode: {config.simulation_mode.value}")
    print(f"  Number of Residents: {config.num_residents}")
    print(f"  Consciousness Features:")
    print(f"    - Internal Dialogue: {config.enable_internal_dialogue}")
    print(f"    - Strange Loops: {config.enable_strange_loops}")
    print(f"    - Meta-Awareness: {config.enable_meta_awareness}")
    print(f"    - Social Physics: {config.enable_spring_dynamics}")
    print()

    async with NeuroHood(config) as hood:

        # ========== 2. Bootstrap Neighborhood ==========

        print("[ Step 2: Bootstrapping Neighborhood ]")
        print("-" * 70)

        state = await hood.bootstrap_neighborhood(
            seed="A quiet suburban street where Alice values peace, "
                 "Bob loves parties, and Charlie tries to keep the peace"
        )

        print(f"  Created {len(state.residents)} residents:")
        for resident in state.residents.values():
            print(f"    - {resident.name}: {resident.occupation}, age {resident.age}")
            print(f"      Lives at: {resident.home_address}")

        print(f"\n  Created {len(state.relationships)} relationships")
        print()

        # ========== 3. Resident Thoughts (Internal Dialogue) ==========

        print("[ Step 3: Resident Internal Dialogue ]")
        print("-" * 70)
        print("  Asking residents to think about their neighbors...\n")

        # Get first three residents
        residents = list(state.residents.values())
        alice = residents[0]
        bob = residents[1]
        charlie = residents[2] if len(residents) > 2 else None

        # Alice thinks about Bob
        print(f"  {alice.name} thinks about {bob.name}:")
        thoughts_alice = await hood.get_resident_thoughts(alice.resident_id, state)

        # Trigger Alice to think
        agent_alice = hood.resident_agents[alice.resident_id]
        thought = await agent_alice.think(
            f"What do I think about my neighbor {bob.name}?",
            mode="exploratory"
        )

        print(f"    Thought: {thought['response'][:200]}...")
        print(f"    Confidence: {thought['confidence']:.2f}")
        if "strange_loops" in thought:
            print(f"    ♾️  Strange Loops Detected: {len(thought['strange_loops'])}")
        print()

        # Bob thinks about parties
        agent_bob = hood.resident_agents[bob.resident_id]
        thought_bob = await agent_bob.think(
            "Should I throw a party this weekend?",
            mode="exploratory"
        )

        print(f"  {bob.name} considers throwing a party:")
        print(f"    Thought: {thought_bob['response'][:200]}...")
        print(f"    Confidence: {thought_bob['confidence']:.2f}")
        print()

        # ========== 4. Player Action (Trigger Drama) ==========

        print("[ Step 4: Player Action - Triggering Neighborhood Drama ]")
        print("-" * 70)

        player_action = f"{alice.name} knocks on {bob.name}'s door to complain about noise"
        print(f"  Action: {player_action}\n")

        new_state = await hood.step(
            player_action=player_action,
            state=state,
            time_delta=60.0  # 1 minute passes
        )

        print(f"  Simulation time advanced: {new_state.timestamp:.0f} seconds")
        print(f"  Events generated: {len(new_state.recent_events)}")

        if new_state.recent_events:
            event = new_state.recent_events[-1]
            print(f"  Latest event: {event.get('description', '')[:150]}...")
        print()

        # ========== 5. Consciousness Visualization ==========

        print("[ Step 5: Consciousness Visualization ]")
        print("-" * 70)

        consciousness_view = ConsciousnessView(mode="thoughts")

        # Get updated thoughts
        thoughts_alice_updated = await hood.get_resident_thoughts(alice.resident_id, new_state)

        # Render thought stream
        stream_viz = consciousness_view.render_thought_stream(
            thoughts_alice_updated["active_thoughts"],
            alice.name
        )

        print(stream_viz)

        # Show meta-awareness
        meta_viz = consciousness_view.render_meta_awareness(
            thoughts_alice_updated["meta_awareness"],
            alice.name
        )

        print(meta_viz)

        # ========== 6. Strange Loop Detection ==========

        print("[ Step 6: Strange Loop Detection (Consciousness Moments) ]")
        print("-" * 70)

        # Trigger recursive self-reflection
        thought_recursive = await agent_alice.think(
            "Why am I so bothered by noise? Is it really the noise, or is it something else?",
            mode="hofstadter"  # Recursive mode
        )

        print(f"  {alice.name} engages in recursive self-reflection:")
        print(f"    Prompt: {thought_recursive['prompt']}")
        print(f"    Response: {thought_recursive['response'][:200]}...")
        print(f"    Confidence: {thought_recursive['confidence']:.2f}")

        if "strange_loops" in thought_recursive:
            for loop in thought_recursive["strange_loops"]:
                loop_viz = consciousness_view.render_strange_loop(loop, alice.name)
                print(loop_viz)

        # ========== 7. Full DMT Mode (All Residents) ==========

        print("[ Step 7: Full DMT Mode - Ego Dissolution ]")
        print("-" * 70)
        print("  Viewing all residents' thoughts simultaneously...\n")

        consciousness_dmt = ConsciousnessView(mode="full_dmt")

        all_thoughts = {}
        for resident in state.residents.values():
            thoughts = await hood.get_resident_thoughts(resident.resident_id, new_state)
            all_thoughts[resident.resident_id] = thoughts["active_thoughts"]

        dmt_viz = consciousness_dmt.render_full_dmt_mode(
            all_thoughts,
            focus_resident=alice.resident_id
        )

        print(dmt_viz)

        # ========== 8. Metrics Summary ==========

        print("[ Step 8: Simulation Metrics ]")
        print("-" * 70)

        metrics = hood.get_metrics()

        print(f"  Residents Created: {metrics['residents_created']}")
        print(f"  Total Events: {metrics['events_generated']}")
        print(f"  Conversations: {metrics['conversations']}")
        print(f"  Thoughts Generated: {metrics['thoughts_generated']}")
        print(f"  Strange Loops Detected: {metrics['strange_loops_detected']}")
        print(f"  Relationships: {metrics['relationships_formed']}")
        print(f"  Player Interactions: {metrics['player_interactions']}")
        print()

        # ========== 9. Conclusion ==========

        print("="*70)
        print("  NeuroHood Demo Complete!")
        print("="*70)
        print("\n  What We Just Did:")
        print("    ✓ Created 3 conscious residents with internal dialogue")
        print("    ✓ Observed recursive self-reflection (strange loops)")
        print("    ✓ Triggered neighborhood drama (Alice vs Bob)")
        print("    ✓ Visualized consciousness in 4 modes")
        print("    ✓ Demonstrated emergent personality and relationships")
        print("\n  Next Steps:")
        print("    - Implement full Spring Dynamics integration")
        print("    - Add causal reasoning (why did conflict occur?)")
        print("    - Expand to 20 residents (Phase 2)")
        print("    - Add multi-agent conversations (message bus)")
        print("    - Create visual UI (consciousness view)")
        print("\n  The Future:")
        print("    This could be the first game where NPCs experience")
        print("    actual recursive self-awareness, not just behavior trees.")
        print()


if __name__ == "__main__":
    asyncio.run(main())
