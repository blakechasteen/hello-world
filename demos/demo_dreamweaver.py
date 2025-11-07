"""
DreamWeaver Demo: Open-Source World Building

Demonstrates Phase 0 architecture with placeholder implementations.
Shows the complete API surface and integration with HoloLoom.

Run:
    PYTHONPATH=. python demos/demo_dreamweaver.py
"""

import asyncio
from datetime import datetime

# Phase 0: Core architecture imports
from HoloLoom.dreamweaving import (
    DreamWeaver,
    DreamWeaverConfig,
    WorldEntity,
    NarrativeEvent,
    EntityType,
    EventType,
    GenerationMode,
    ConsistencyLevel,
)

from HoloLoom.config import Config
from HoloLoom.documentation.types import MemoryShard


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_entity(entity: WorldEntity):
    """Pretty-print world entity."""
    print(f"📦 {entity.name} ({entity.entity_type.value})")
    print(f"   ID: {entity.entity_id}")
    print(f"   Description: {entity.description}")
    if entity.attributes:
        print(f"   Attributes: {entity.attributes}")
    if entity.relationships:
        print(f"   Relationships: {len(entity.relationships)}")
    print()


def print_event(event: NarrativeEvent):
    """Pretty-print narrative event."""
    print(f"⚡ {event.event_type.value.upper()} @ t={event.timestamp}")
    print(f"   ID: {event.event_id}")
    print(f"   Description: {event.description}")
    print(f"   Participants: {', '.join(event.participants)}")
    if event.location_id:
        print(f"   Location: {event.location_id}")
    print()


async def demo_phase_0_architecture():
    """
    Demonstrate Phase 0: Core architecture and API surface.

    Shows:
    - DreamWeaver initialization
    - World bootstrapping
    - Entity generation
    - Event generation
    - World evolution
    - Consistency checking
    - Persistence
    """

    print_header("DreamWeaver Demo - Phase 0: Architecture")

    print("🌟 Initializing DreamWeaver...")
    print("   This is a demonstration of the Phase 0 architecture.")
    print("   Entity/event generation uses placeholder implementations.")
    print("   Full rule-based generation comes in Phase 1.\n")

    # Configure DreamWeaver
    config = DreamWeaverConfig(
        world_id="demo_fantasy_world",
        generation_mode=GenerationMode.PROCEDURAL,
        consistency_level=ConsistencyLevel.BALANCED,
        auto_save=False,  # Don't save demo world
        creativity_temperature=0.8,
        use_thompson_sampling=True,
    )

    # Configure HoloLoom (use FAST mode for demo)
    hololoom_config = Config.fast()

    print(f"Configuration:")
    print(f"  World ID: {config.world_id}")
    print(f"  Generation Mode: {config.generation_mode.value}")
    print(f"  Consistency Level: {config.consistency_level.value}")
    print(f"  HoloLoom Mode: FAST")
    print()

    # Initialize DreamWeaver
    async with DreamWeaver(config, hololoom_config) as weaver:
        print("✅ DreamWeaver initialized successfully!\n")

        # ========== Bootstrap World ==========
        print_header("1. Bootstrap World")

        seed_prompt = (
            "A high-fantasy world with magic, dragons, and ancient ruins. "
            "The world is divided into five kingdoms, each with distinct cultures. "
            "Magic is powered by connection to elemental forces."
        )

        print(f"Seed Prompt:\n  {seed_prompt}\n")
        print("Generating initial world state...")

        world_state = await weaver.bootstrap_world(seed_prompt)

        print(f"✅ World bootstrapped!")
        print(f"   Timestamp: {world_state.timestamp}")
        print(f"   Entities: {len(world_state.entities)}")
        print(f"   Events: {len(world_state.recent_events)}")
        print(f"   Active Threads: {len(world_state.active_threads)}")
        print()

        # ========== Generate Entities ==========
        print_header("2. Generate Entities")

        print("Generating character entity...")

        character = await weaver.generate_entity(
            entity_type=EntityType.CHARACTER,
            context={
                "world_theme": "high-fantasy",
                "role": "protagonist",
            },
            constraints={
                "personality": "brave but reckless",
                "magical_affinity": "fire",
            }
        )

        print_entity(character)

        print("Generating location entity...")

        location = await weaver.generate_entity(
            entity_type=EntityType.LOCATION,
            context={
                "world_theme": "high-fantasy",
                "climate": "temperate",
            },
            constraints={
                "has_magic": True,
                "danger_level": "moderate",
            }
        )

        print_entity(location)

        # ========== Generate Events ==========
        print_header("3. Generate Events")

        print("Generating narrative event...")

        event = await weaver.generate_event(
            event_type=EventType.DISCOVERY,
            participants=[character.entity_id],
            context={
                "location": location.entity_id,
                "tension_level": "medium",
            },
            constraints={
                "significance": "major",
            }
        )

        print_event(event)

        # ========== World Evolution ==========
        print_header("4. World Evolution")

        print("User Input: 'I explore the ancient temple ruins.'\n")

        new_world_state = await weaver.step(
            user_input="I explore the ancient temple ruins.",
            world_state=world_state,
        )

        print(f"✅ World state updated!")
        print(f"   Timestamp: {world_state.timestamp} → {new_world_state.timestamp}")
        print(f"   Entities: {len(world_state.entities)} → {len(new_world_state.entities)}")
        print(f"   Events: {len(world_state.recent_events)} → {len(new_world_state.recent_events)}")
        print()

        # ========== Query World ==========
        print_header("5. Query World (Read-Only)")

        query = "What kingdoms exist in this world?"
        print(f"Query: {query}\n")

        response = await weaver.query_world(query, new_world_state)

        print(f"Response:\n  {response}\n")

        # ========== Consistency Checking ==========
        print_header("6. Consistency Checking")

        print("Checking world state consistency...")

        violations = await weaver.check_consistency(new_world_state)

        if violations:
            print(f"⚠️  Found {len(violations)} consistency violations:\n")
            for violation in violations:
                print(f"   [{violation['severity']:.1f}] {violation['rule_type']}: {violation['description']}")
        else:
            print("✅ No consistency violations detected!")

        print()

        # ========== Narrative Threads ==========
        print_header("7. Narrative Thread Management")

        print("Creating narrative thread...")

        thread = await weaver.create_thread(
            thread_type="plot",
            title="The Quest for the Lost Artifact",
            participants=[character.entity_id],
            world_state=new_world_state,
        )

        print(f"✅ Thread created!")
        print(f"   ID: {thread.thread_id}")
        print(f"   Type: {thread.thread_type}")
        print(f"   Title: {thread.title}")
        print(f"   Status: {thread.status}")
        print(f"   Tension: {thread.tension}")
        print()

        # ========== Metrics ==========
        print_header("8. Metrics & Statistics")

        metrics = weaver.get_metrics()

        print("DreamWeaver Metrics:")
        for key, value in sorted(metrics.items()):
            print(f"   {key}: {value}")
        print()

        # ========== World State Inspection ==========
        print_header("9. World State Details")

        print(f"Current World State (t={new_world_state.timestamp}):")
        print(f"   World ID: {config.world_id}")
        print(f"   Total Entities: {len(new_world_state.entities)}")
        print(f"   Total Events: {len(new_world_state.recent_events)}")
        print(f"   Active Threads: {len(new_world_state.active_threads)}")
        print(f"   Consistency Violations: {len(new_world_state.consistency_violations)}")
        print()

        if new_world_state.entities:
            print("Sample Entities:")
            for i, entity in enumerate(list(new_world_state.entities.values())[:3]):
                print(f"   {i+1}. {entity.name} ({entity.entity_type.value})")
        print()

        if new_world_state.active_threads:
            print("Active Narrative Threads:")
            for i, thread in enumerate(new_world_state.active_threads):
                print(f"   {i+1}. {thread.title} ({thread.status})")
        print()

        # ========== Architecture Overview ==========
        print_header("10. Architecture Overview")

        print("DreamWeaver Architecture:")
        print()
        print("┌─────────────────────────────────────────────┐")
        print("│          DreamWeaver Orchestrator           │")
        print("│    (Core world building coordination)       │")
        print("└──────────────┬──────────────────────────────┘")
        print("               │")
        print("      ┌────────┴────────┐")
        print("      │                 │")
        print("┌─────▼──────┐    ┌────▼────────┐")
        print("│  World     │    │ Consistency │")
        print("│  Fabric    │    │ Engine      │")
        print("│ (State Mgr)│    │ (Rules)     │")
        print("└─────┬──────┘    └────┬────────┘")
        print("      │                │")
        print("┌─────▼──────┐    ┌────▼────────┐")
        print("│ Narrative  │    │ Generative  │")
        print("│ Memory     │    │ Loom        │")
        print("│ (Story KG) │    │ (Content)   │")
        print("└─────┬──────┘    └────┬────────┘")
        print("      │                │")
        print("      └────────┬───────┘")
        print("               │")
        print("      ┌────────▼──────────┐")
        print("      │ HoloLoom          │")
        print("      │ (Weaving          │")
        print("      │  Orchestrator)    │")
        print("      └───────────────────┘")
        print()

        print("Integration Points:")
        print("  ✓ Yarn Graph: Persistent world memory")
        print("  ✓ Warp Space: Narrative manifold tensioning")
        print("  ✓ Convergence Engine: Story possibility collapse")
        print("  ✓ Thompson Sampling: Narrative branch exploration")
        print("  ✓ Reflection Buffer: Player feedback learning")
        print("  ✓ Chrono Trigger: In-world temporal control")
        print()

        # ========== Next Steps ==========
        print_header("What's Next?")

        print("Phase 0 (Current): ✅ Architecture Complete")
        print("  - Core data structures defined")
        print("  - Protocol-based design")
        print("  - DreamWeaver orchestrator skeleton")
        print("  - HoloLoom integration points")
        print()

        print("Phase 1 (Weeks 1-8): 🚧 Core World Building")
        print("  - Narrative memory (story-aware KG)")
        print("  - World fabric (versioning + persistence)")
        print("  - Consistency engine (rule-based checking)")
        print("  - Generative loom (template-based generation)")
        print()

        print("Phase 2 (Weeks 9-12): 📅 LLM Integration")
        print("  - Natural language generation")
        print("  - Dynamic dialogue")
        print("  - Context-aware descriptions")
        print()

        print("Phase 3 (Weeks 13-18): 📅 Collaborative Authoring")
        print("  - Human-AI co-creation")
        print("  - Authorship tracking")
        print("  - Web dashboard")
        print()

        print("See DREAMWEAVER_ROADMAP.md for complete 6-phase plan!")
        print()

    print_header("Demo Complete!")

    print("✅ All Phase 0 features demonstrated successfully!")
    print()
    print("Key Takeaways:")
    print("  1. DreamWeaver extends HoloLoom's weaving metaphor to world building")
    print("  2. Protocol-based design enables flexible implementations")
    print("  3. Async lifecycle management (async context managers)")
    print("  4. Complete integration with HoloLoom's 9-layer system")
    print("  5. Ready for Phase 1 implementation!")
    print()
    print("Next: Review DREAMWEAVER_ROADMAP.md and begin Phase 1 (Narrative Memory)")
    print()


async def demo_api_surface():
    """
    Quick demo of the complete API surface.
    """

    print_header("DreamWeaver API Surface")

    print("Core API (10 methods):")
    print()
    print("Initialization:")
    print("  - DreamWeaver(config, hololoom_config)")
    print("  - bootstrap_world(seed_prompt)")
    print("  - load_world(world_id)")
    print()
    print("Evolution:")
    print("  - step(user_input, world_state)")
    print("  - query_world(query, world_state)")
    print()
    print("Generation:")
    print("  - generate_entity(entity_type, context)")
    print("  - generate_event(event_type, participants, context)")
    print()
    print("Consistency:")
    print("  - check_consistency(world_state, rules)")
    print("  - resolve_contradiction(world_state, violation)")
    print()
    print("Persistence:")
    print("  - save_world(world_state, path)")
    print()

    print("Extended API (Future):")
    print("  - Thread management (create_thread, resolve_thread)")
    print("  - Relationship operations (add_relationship, remove_relationship)")
    print("  - Temporal operations (rewind, fast_forward, branch_timeline)")
    print("  - Authorship tracking (get_authorship, approve_suggestion)")
    print("  - Analytics (get_metrics, get_statistics, export_graph)")
    print()


if __name__ == "__main__":
    print("\n" + "🌟" * 30)
    print("   DreamWeaver: Open-Source World Building for HoloLoom")
    print("🌟" * 30 + "\n")

    # Run main demo
    asyncio.run(demo_phase_0_architecture())

    # Show API surface
    asyncio.run(demo_api_surface())

    print("Demo finished! Check DREAMWEAVER_ROADMAP.md for the full 6-phase plan.\n")