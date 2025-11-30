"""
Demo: Scaled Neighborhood with Sims Features

Shows how to scale from 6 to 50+ residents while maintaining
rich Sims-style features (moodlets, skills, energy, life goals).

Author: Claude
Date: November 2025
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from resident_generator import (
    ResidentGenerator,
    GenerationConfig,
    Archetype,
    generate_scaled_neighborhood,
)
from sims_inspired import SimsEnhancedEngine, LifeGoal, DEFAULT_LIFE_GOALS
from solo_activities import DEFAULT_PROJECTS


async def demo_scaled_neighborhood():
    """Demonstrate a scaled neighborhood with Sims features."""

    print("=" * 70)
    print("SCALED NEIGHBORHOOD DEMO")
    print("Demonstrating 50 procedurally generated residents with Sims features")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Generate 50 diverse residents
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: GENERATING 50 DIVERSE RESIDENTS")
    print("=" * 50)

    config = GenerationConfig(
        num_residents=50,
        seed=42,  # Reproducible
        cultural_mix={
            "hispanic": 0.30,
            "anglo": 0.25,
            "african_american": 0.20,
            "asian": 0.15,
            "mixed": 0.10,
        },
        ensure_variety=True,
    )

    generator = ResidentGenerator(config)
    residents, projects, archetypes = generator.generate_neighborhood()

    print(f"\nGenerated {len(residents)} residents")
    print(generator.get_generation_summary())

    # =========================================================================
    # STEP 2: Create the Sims-Enhanced Engine
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: CREATING SIMS-ENHANCED ENGINE")
    print("=" * 50)

    # Create engine (uses default 6 residents initially)
    engine = SimsEnhancedEngine()
    await engine.initialize()

    # Replace with generated residents
    # Note: AgentState only has basic fields (agent_id, name, is_user, personality, emotions)
    # Full resident data (backstory, occupation, age) is stored in the `residents` dict
    engine.state.agents.clear()
    from neighborhood_engine import AgentState
    for resident_id, resident in residents.items():
        agent = AgentState(
            agent_id=resident_id,
            name=resident.name,
            is_user=False,
            personality=resident.personality.__dict__,
        )
        engine.state.agents[resident_id] = agent

    # Add generated projects
    engine.projects = {k: list(v) for k, v in projects.items()}

    # Assign life goals based on archetypes
    for resident_id, archetype in archetypes.items():
        # Pick a life goal that fits the archetype
        goal_map = {
            Archetype.ARTIST: "artistic_prodigy",
            Archetype.WRITER: "bestselling_author",
            Archetype.MUSICIAN: "musical_genius",
            Archetype.TEACHER: "beloved_mentor",
            Archetype.ENTREPRENEUR: "self_made",
            Archetype.ELDER: "beloved_mentor",
            Archetype.NEWCOMER: "social_butterfly",
            Archetype.HEALER: "inner_peace",
            Archetype.STUDENT: "self_made",
            Archetype.RETIREE: "beloved_mentor",
            Archetype.HEALTHCARE: "inner_peace",
            Archetype.LOCAL_BUSINESS: "self_made",
            Archetype.CRAFTSPERSON: "artistic_prodigy",
            Archetype.SCIENTIST: "bestselling_author",  # Discovery fame
            Archetype.TECH_WORKER: "self_made",
            Archetype.COMMUNITY_ORGANIZER: "social_butterfly",
            Archetype.YOUNG_PARENT: "inner_peace",
            Archetype.EMPTY_NESTER: "beloved_mentor",
            Archetype.INTROVERT_SAGE: "inner_peace",
            Archetype.SOCIAL_BUTTERFLY: "social_butterfly",
            Archetype.SKEPTIC: "bestselling_author",
            Archetype.OPTIMIST: "social_butterfly",
        }
        goal_id = goal_map.get(archetype, "inner_peace")
        if goal_id in DEFAULT_LIFE_GOALS:
            engine.life_goals[resident_id] = LifeGoal(**DEFAULT_LIFE_GOALS[goal_id])

    print(f"\n[Engine] Loaded {len(engine.state.agents)} agents")
    print(f"[Engine] Loaded {sum(len(p) for p in engine.projects.values())} projects")
    print(f"[Engine] Assigned {len(engine.life_goals)} life goals")

    # =========================================================================
    # STEP 3: Show sample status panels
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: SAMPLE RESIDENT STATUS PANELS")
    print("=" * 50)

    # Show 5 random residents
    import random
    sample_ids = random.sample(list(residents.keys()), min(5, len(residents)))

    for agent_id in sample_ids:
        resident = residents[agent_id]
        archetype = archetypes[agent_id]

        print(f"\n{resident.name} ({resident.age}, {resident.occupation})")
        print(f"  Archetype: {archetype.value}")
        print(f"  Personality: {resident.personality.describe()}")
        print(f"  Backstory: {resident.backstory[:80]}...")

        # Sims features
        energy = engine.energy.get_energy(agent_id)
        print(f"  Energy: {energy.current:.0f}/100 ({energy.status})")

        if agent_id in engine.life_goals:
            goal = engine.life_goals[agent_id]
            print(f"  Life Goal: {goal.name}")

        if agent_id in engine.projects and engine.projects[agent_id]:
            project = engine.projects[agent_id][0]
            print(f"  Current Project: {project.name}")

    # =========================================================================
    # STEP 4: Simulate some activity
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: SIMULATING ACTIVITY")
    print("=" * 50)

    # Add some moodlets to random residents
    emotions = ["inspired", "happy", "confident", "creative"]
    for agent_id in random.sample(list(residents.keys()), 10):
        engine.moodlets.add_moodlet(agent_id, random.choice(["great_conversation", "creative_flow", "well_rested"]), "demo")

    # Add skill XP to random residents
    skills = ["Painting", "Writing", "Cooking", "Gardening", "Charisma", "Programming"]
    for agent_id in random.sample(list(residents.keys()), 15):
        skill = random.choice(skills)
        xp = random.randint(50, 200)
        engine.skills.add_experience(agent_id, skill, xp)

    print("\nAdded moodlets to 10 residents")
    print("Added skill XP to 15 residents")

    # =========================================================================
    # STEP 5: Neighborhood statistics
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: NEIGHBORHOOD STATISTICS")
    print("=" * 50)

    # Archetype distribution
    archetype_counts = {}
    for a in archetypes.values():
        archetype_counts[a.value] = archetype_counts.get(a.value, 0) + 1

    print("\nArchetype Distribution:")
    for archetype, count in sorted(archetype_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "#" * count
        print(f"  {archetype:20} {bar} ({count})")

    # Age distribution
    ages = [r.age for r in residents.values()]
    young = sum(1 for a in ages if a < 30)
    middle = sum(1 for a in ages if 30 <= a < 55)
    senior = sum(1 for a in ages if a >= 55)

    print(f"\nAge Distribution:")
    print(f"  Young (18-29):  {'#' * young} ({young})")
    print(f"  Middle (30-54): {'#' * middle} ({middle})")
    print(f"  Senior (55+):   {'#' * senior} ({senior})")

    # Mood overview
    happy_count = 0
    neutral_count = 0
    sad_count = 0
    for agent_id in residents:
        score = engine.moodlets.get_mood_score(agent_id)
        if score > 0:
            happy_count += 1
        elif score < 0:
            sad_count += 1
        else:
            neutral_count += 1

    print(f"\nMood Overview:")
    print(f"  Happy:   {'#' * happy_count} ({happy_count})")
    print(f"  Neutral: {'#' * neutral_count} ({neutral_count})")
    print(f"  Sad:     {'#' * sad_count} ({sad_count})")

    # Energy overview
    energized = sum(1 for aid in residents if engine.energy.get_energy(aid).current >= 70)
    tired = sum(1 for aid in residents if engine.energy.get_energy(aid).current < 40)
    normal = len(residents) - energized - tired

    print(f"\nEnergy Overview:")
    print(f"  Energized: {'#' * energized} ({energized})")
    print(f"  Normal:    {'#' * normal} ({normal})")
    print(f"  Tired:     {'#' * tired} ({tired})")

    # =========================================================================
    # STEP 6: End of day summaries for a few
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: END OF DAY SUMMARIES")
    print("=" * 50)

    for agent_id in sample_ids[:3]:
        resident = residents[agent_id]
        agent = engine.state.agents[agent_id]
        energy = engine.energy.get_energy(agent_id)
        moodlets = engine.moodlets.get_active_moodlets(agent_id)

        summary = await engine.summary_generator.generate_summary(
            agent_name=resident.name,
            agent_personality=agent.personality,
            memories_today=[],  # No memories for demo
            moodlets=moodlets,
            energy_level=energy.current,
        )
        print(f"\n{resident.name}'s Day:")
        print(f"  \"{summary}\"")

    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCALED NEIGHBORHOOD DEMO COMPLETE")
    print("=" * 70)
    print(f"\nSuccessfully created a neighborhood of {len(residents)} residents with:")
    print("  - Diverse archetypes (21 types)")
    print("  - Rich backstories")
    print("  - Personal projects")
    print("  - Life goals/aspirations")
    print("  - Sims-style moodlets, skills, and energy")
    print("\nThe generator can scale to 100s or 1000s of residents!")


if __name__ == "__main__":
    asyncio.run(demo_scaled_neighborhood())
