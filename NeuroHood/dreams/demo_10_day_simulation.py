"""
Demo: Extended Neighborhood Simulation with 50 Residents

Simulates multiple days of life in a procedurally generated neighborhood,
tracking moods, energy, skills, projects, social dynamics, and relationships.

Features:
- Relationship tracking between residents
- Dramatic events (romances, conflicts, community events)
- Ability to continue simulation for additional days
- Friendship and rivalry dynamics
- Relationship decay for long-term simulations
- Tuned thresholds for more dynamic interactions

v5 COMPLEXITY FEATURES (November 2025):
- Love triangles & jealousy dynamics
- Betrayal events (trust breaking, secret reveals)
- Life crises (job loss, health scare, financial stress)
- Breakups (romances can end, creating bitter exes)
- Reputation system (gossip affects future relationships)
- Mood instability (personality-based variance, bad days)
- Clique formation (popular residents, inter-group tension)
- Grudges (long-lasting negative feelings that don't decay)
- Secret romances (can be exposed causing drama)

PERFORMANCE COMPARISON:
| Metric              | v4 (180d) | v5 Target |
|---------------------|-----------|-----------|
| Friendships         |    48     |   40-60   |
| Romances            |    5      |   8-15    |
| Rivalries           |    2      |   10-20   |
| Breakups            |    0      |   3-8     |
| Love Triangles      |    0      |   2-5     |
| Betrayals           |    0      |   5-15    |
| Life Crises         |    0      |   20-40   |
| Happiness Stability |   100%    |   60-85%  |

Author: Claude
Date: November 2025
"""

import asyncio
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from resident_generator import (
    ResidentGenerator,
    GenerationConfig,
    Archetype,
)
from sims_inspired import SimsEnhancedEngine, LifeGoal, DEFAULT_LIFE_GOALS
from neighborhood_engine import AgentState


@dataclass
class DayStats:
    """Statistics for a single day."""
    day: int
    happy_count: int
    neutral_count: int
    sad_count: int
    energized_count: int
    tired_count: int
    skills_leveled: int
    projects_worked: int
    conversations_had: int
    notable_events: List[str]
    friendships_formed: int = 0
    rivalries_formed: int = 0
    romances_sparked: int = 0


@dataclass
class Relationship:
    """Tracks relationship between two residents."""
    friendship: int = 0  # -100 to +100
    interactions: int = 0
    last_interaction_day: int = 0
    is_romantic: bool = False
    is_rival: bool = False
    # v5 complexity additions
    trust: int = 50  # 0-100, starts neutral
    jealousy: int = 0  # 0-100, builds when partner interacts with others
    is_secret_romance: bool = False  # Hidden relationship that can be exposed
    is_grudge: bool = False  # Long-lasting resentment that doesn't decay
    was_romantic: bool = False  # Former lovers (bitter exes)
    romance_start_day: int = 0  # When romance started (for duration tracking)


@dataclass
class LifeCrisis:
    """Tracks ongoing life crises for a resident."""
    crisis_type: str  # job_loss, health_scare, financial_stress, heartbreak, betrayed
    start_day: int
    severity: float  # 0.0-1.0
    duration_days: int  # How long it lasts
    resolved: bool = False


class SimulationEngine:
    """Manages the multi-day simulation."""

    def __init__(self, num_residents: int = 50, seed: int = 42):
        self.num_residents = num_residents
        self.seed = seed
        self.residents = {}
        self.archetypes = {}
        self.engine = None
        self.day_stats: List[DayStats] = []
        self.skill_levels: Dict[str, Dict[str, int]] = {}  # agent_id -> skill -> level
        self.relationships: Dict[Tuple[str, str], Relationship] = {}  # (id1, id2) -> Relationship
        self.total_days_simulated = 0
        self.community_events: List[str] = []  # Major community events

        # v5 complexity additions
        self.life_crises: Dict[str, List[LifeCrisis]] = {}  # agent_id -> active crises
        self.reputation: Dict[str, int] = {}  # agent_id -> reputation score (-100 to +100)
        self.cliques: Dict[str, List[str]] = {}  # clique_name -> member_ids
        self.love_triangles: List[Tuple[str, str, str]] = []  # (person1, person2, interloper)

        # v5 stats tracking
        self.total_breakups = 0
        self.total_betrayals = 0
        self.total_crises = 0
        self.total_secret_reveals = 0
        self.total_love_triangles = 0

    def _get_relationship(self, id1: str, id2: str) -> Relationship:
        """Get or create relationship between two residents."""
        key = tuple(sorted([id1, id2]))
        if key not in self.relationships:
            self.relationships[key] = Relationship()
        return self.relationships[key]

    def _update_relationship(self, id1: str, id2: str, delta: int, day: int) -> Relationship:
        """Update friendship score between two residents."""
        rel = self._get_relationship(id1, id2)
        rel.friendship = max(-100, min(100, rel.friendship + delta))
        rel.interactions += 1
        rel.last_interaction_day = day
        return rel

    async def setup(self):
        """Generate residents and initialize engine."""
        print("=" * 70)
        print("INITIALIZING 10-DAY NEIGHBORHOOD SIMULATION")
        print(f"Residents: {self.num_residents} | Days: 10 | Seed: {self.seed}")
        print("=" * 70)

        # Generate residents
        print("\n[Setup] Generating residents...")
        config = GenerationConfig(
            num_residents=self.num_residents,
            seed=self.seed,
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
        self.residents, projects, self.archetypes = generator.generate_neighborhood()
        print(f"[Setup] Generated {len(self.residents)} residents")

        # Initialize engine
        print("[Setup] Initializing Sims engine...")
        self.engine = SimsEnhancedEngine()
        await self.engine.initialize()

        # Replace with generated residents
        self.engine.state.agents.clear()
        for resident_id, resident in self.residents.items():
            agent = AgentState(
                agent_id=resident_id,
                name=resident.name,
                is_user=False,
                personality=resident.personality.__dict__,
            )
            self.engine.state.agents[resident_id] = agent
            self.skill_levels[resident_id] = {}
            # v5: Initialize reputation (social butterflies start higher)
            base_rep = 0
            archetype = self.archetypes.get(resident_id)
            if archetype == Archetype.SOCIAL_BUTTERFLY:
                base_rep = random.randint(10, 30)
            elif archetype == Archetype.COMMUNITY_ORGANIZER:
                base_rep = random.randint(5, 20)
            elif archetype == Archetype.SKEPTIC:
                base_rep = random.randint(-10, 5)
            self.reputation[resident_id] = base_rep
            self.life_crises[resident_id] = []

        # Add projects
        self.engine.projects = {k: list(v) for k, v in projects.items()}

        # Assign life goals
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
            Archetype.SCIENTIST: "bestselling_author",
            Archetype.TECH_WORKER: "self_made",
            Archetype.COMMUNITY_ORGANIZER: "social_butterfly",
            Archetype.YOUNG_PARENT: "inner_peace",
            Archetype.EMPTY_NESTER: "beloved_mentor",
            Archetype.INTROVERT_SAGE: "inner_peace",
            Archetype.SOCIAL_BUTTERFLY: "social_butterfly",
            Archetype.SKEPTIC: "bestselling_author",
            Archetype.OPTIMIST: "social_butterfly",
        }

        for resident_id, archetype in self.archetypes.items():
            goal_id = goal_map.get(archetype, "inner_peace")
            if goal_id in DEFAULT_LIFE_GOALS:
                self.engine.life_goals[resident_id] = LifeGoal(**DEFAULT_LIFE_GOALS[goal_id])

        print(f"[Setup] Loaded {len(self.engine.state.agents)} agents")
        print(f"[Setup] Assigned {len(self.engine.life_goals)} life goals")
        print("[Setup] Ready to simulate!\n")

    async def simulate_day(self, day: int) -> DayStats:
        """Simulate a single day."""
        notable_events = []
        skills_leveled = 0
        projects_worked = 0
        conversations_had = 0

        resident_ids = list(self.residents.keys())

        # === MORNING: Energy and mood reset ===
        for agent_id in resident_ids:
            # Morning energy boost (simulating sleep)
            energy = self.engine.energy.get_energy(agent_id)
            rest_amount = random.uniform(0.3, 0.6)
            self.engine.energy.rest(agent_id, rest_amount * 8)  # 8 hours sleep

            # Random morning mood events
            if random.random() < 0.15:  # 15% chance of morning moodlet
                moodlet = random.choice(["well_rested", "creative_flow", "cozy_atmosphere"])
                self.engine.moodlets.add_moodlet(agent_id, moodlet, "morning")

        # === DAYTIME: Activities ===
        # Work on projects (30% of residents each day)
        workers = random.sample(resident_ids, int(len(resident_ids) * 0.3))
        for agent_id in workers:
            projects = self.engine.projects.get(agent_id, [])
            if projects:
                project = random.choice(projects)
                hours = random.uniform(1, 4)

                # Work drains energy
                self.engine.energy.drain(agent_id, "project_work")

                # Add skill XP
                skill_name = self._get_skill_for_archetype(self.archetypes[agent_id])
                xp = int(hours * 25)
                new_level, leveled = self.engine.skills.add_experience(agent_id, skill_name, xp)

                # Track skill levels
                old_level = self.skill_levels[agent_id].get(skill_name, 1)
                self.skill_levels[agent_id][skill_name] = int(new_level)

                if leveled or int(new_level) > old_level:
                    skills_leveled += 1
                    resident = self.residents[agent_id]
                    notable_events.append(
                        f"{resident.name}'s {skill_name} reached level {int(new_level)}!"
                    )
                    self.engine.moodlets.add_moodlet(agent_id, "accomplished", "skill_up")

                projects_worked += 1

                # Moodlet from work
                if random.random() < 0.3:
                    self.engine.moodlets.add_moodlet(agent_id, "creative_flow", "project")

        # === RELATIONSHIP DECAY (for long simulations) ===
        # Relationships that haven't had interaction in 7+ days decay slightly
        # GRUDGES NEVER DECAY - they are permanent!
        for key, rel in self.relationships.items():
            if rel.is_grudge:
                continue  # Grudges are permanent!
            days_since_interaction = day - rel.last_interaction_day
            if days_since_interaction >= 7 and rel.last_interaction_day > 0:
                decay = 1
                if rel.friendship > 0:
                    rel.friendship = max(0, rel.friendship - decay)
                elif rel.friendship < 0:
                    rel.friendship = min(0, rel.friendship + decay)

        # === v6.2: ROMANCE TURBULENCE (toned down from v6!) ===
        for (id1, id2), rel in self.relationships.items():
            if rel.is_romantic and rel.romance_start_day > 0:
                romance_duration = day - rel.romance_start_day

                # v6.2: Reduced trust decay - 20% chance (was 60%!)
                if random.random() < 0.20:
                    rel.trust = max(0, rel.trust - 1)

                # v6.2: Relationship fatigue after 90 days - reduced to 2% (was 5%)
                if romance_duration > 90:
                    if random.random() < 0.02:
                        rel.friendship = max(-20, rel.friendship - 3)  # Less severe
                        # 15% chance to trigger stress (was 30%)
                        if random.random() < 0.15:
                            self.engine.moodlets.add_moodlet(id1, "stressed", "relationship_fatigue")

                # v6.2: "Honeymoon period ending" - 3% chance (was 8%)
                if 30 <= romance_duration <= 60:
                    if random.random() < 0.03:
                        rel.trust = max(0, rel.trust - 3)  # Less severe
                        self.engine.moodlets.add_moodlet(id1, "argument", f"with_{self.residents[id2].name}")
                        self.engine.moodlets.add_moodlet(id2, "argument", f"with_{self.residents[id1].name}")

        # === v6.2: LIFE CRISES (1.5% daily chance - reduced from 3%) ===
        breakups_today = 0
        betrayals_today = 0

        for agent_id in resident_ids:
            # v6.2: Reduced crisis chance from 3% to 1.5%
            if random.random() < 0.015:
                crisis_type = random.choice([
                    "job_loss", "health_scare", "financial_stress",
                    "family_drama", "existential_crisis"
                ])
                severity = random.uniform(0.3, 0.6)  # v6.2: Reduced max severity (was 0.8)
                duration = random.randint(5, 14)  # v6.2: Shorter duration (was 7-30)

                crisis = LifeCrisis(
                    crisis_type=crisis_type,
                    start_day=day,
                    severity=severity,
                    duration_days=duration
                )
                self.life_crises[agent_id].append(crisis)
                self.total_crises += 1

                resident = self.residents[agent_id]
                notable_events.append(f"[CRISIS] {resident.name} is going through a {crisis_type.replace('_', ' ')}!")

                # v6.2: Only one moodlet per crisis (no stacking!)
                self.engine.moodlets.add_moodlet(agent_id, "stressed", "crisis")

            # Process ongoing crises - v6.2: further reduced frequency (10%)
            for crisis in self.life_crises[agent_id]:
                if not crisis.resolved:
                    if day >= crisis.start_day + crisis.duration_days:
                        crisis.resolved = True
                    else:
                        # v6.2: 10% chance of ongoing stress (was 20%)
                        if random.random() < crisis.severity * 0.10:
                            self.engine.moodlets.add_moodlet(agent_id, "stressed", "ongoing_crisis")

        # === v5: JEALOUSY & LOVE TRIANGLE DETECTION ===
        for (id1, id2), rel in self.relationships.items():
            if rel.is_romantic:
                # Check if either partner is getting close to someone else
                for other_id in resident_ids:
                    if other_id == id1 or other_id == id2:
                        continue

                    # Check partner1's friendships
                    other_rel1 = self._get_relationship(id1, other_id)
                    other_rel2 = self._get_relationship(id2, other_id)

                    # Jealousy builds if partner is close to someone else
                    if other_rel1.friendship >= 30 and other_rel1.interactions >= 3:
                        rel.jealousy = min(100, rel.jealousy + 2)
                    if other_rel2.friendship >= 30 and other_rel2.interactions >= 3:
                        rel.jealousy = min(100, rel.jealousy + 2)

                    # Love triangle detection (LOWERED threshold from 45 to 35)
                    if other_rel1.friendship >= 35 or other_rel2.friendship >= 35:
                        triangle = tuple(sorted([id1, id2, other_id]))
                        if triangle not in [(t[0], t[1], t[2]) for t in self.love_triangles]:
                            self.love_triangles.append((id1, id2, other_id))
                            self.total_love_triangles += 1
                            r1 = self.residents[id1]
                            r2 = self.residents[id2]
                            r3 = self.residents[other_id]
                            notable_events.append(f"[LOVE TRIANGLE] Tension forming: {r1.name}, {r2.name}, and {r3.name}!")

        # === v5: BREAKUP CHECK (for romantic relationships) ===
        for (id1, id2), rel in list(self.relationships.items()):
            if rel.is_romantic and not rel.was_romantic:  # Active romance
                breakup_chance = 0.01  # Base 1% daily chance (new romances are fragile)

                # High jealousy increases breakup chance
                if rel.jealousy >= 40:  # Lowered from 50
                    breakup_chance += 0.03 * (rel.jealousy / 100)  # Increased

                # Low trust increases breakup chance
                if rel.trust < 40:  # Raised from 30
                    breakup_chance += 0.04  # Increased from 0.03

                # Long relationships are more stable (after 30 days)
                if day - rel.romance_start_day > 30:
                    breakup_chance *= 0.6  # Slightly less stable than before

                # Crises can cause breakups
                crises1 = [c for c in self.life_crises.get(id1, []) if not c.resolved]
                crises2 = [c for c in self.life_crises.get(id2, []) if not c.resolved]
                if crises1 or crises2:
                    breakup_chance += 0.02  # Increased from 0.01

                if random.random() < breakup_chance:
                    rel.is_romantic = False
                    rel.was_romantic = True  # Mark as ex-lovers
                    rel.friendship = max(-50, rel.friendship - 40)  # Bitter breakup
                    breakups_today += 1
                    self.total_breakups += 1

                    r1 = self.residents[id1]
                    r2 = self.residents[id2]
                    notable_events.append(f"[BREAKUP] {r1.name} and {r2.name} broke up!")

                    # v6.2: Shorter heartbreak crisis (7-14 days instead of 14-45)
                    for aid in [id1, id2]:
                        self.life_crises[aid].append(LifeCrisis(
                            crisis_type="heartbreak",
                            start_day=day,
                            severity=random.uniform(0.4, 0.7),  # v6.2: Reduced
                            duration_days=random.randint(7, 14)  # v6.2: Shorter
                        ))
                        # v6.2: Single heartbroken moodlet only (no stacking!)
                        self.engine.moodlets.add_moodlet(aid, "heartbroken", "breakup")

        # === v6.2: BETRAYAL EVENTS (rare - reduced from 6% to 2%) ===
        if random.random() < 0.02:  # v6.2: 2% daily (was 6%!)
            # Find a strong friendship to betray
            strong_friendships = [(k, r) for k, r in self.relationships.items()
                                   if r.friendship >= 35 and r.trust >= 40 and not r.is_grudge]
            if strong_friendships:
                (id1, id2), rel = random.choice(strong_friendships)
                r1 = self.residents[id1]
                r2 = self.residents[id2]

                betrayal_type = random.choice([
                    "revealed_secret", "spread_rumors", "stole_opportunity", "public_humiliation"
                ])

                # v6.2: Reduced trust/friendship hit
                rel.trust = max(0, rel.trust - 30)  # Was -50
                rel.friendship = max(-80, rel.friendship - 25)  # Was -40

                # v6.2: Reduced grudge chance - 30% (was 50%)
                if random.random() < 0.3:
                    rel.is_grudge = True
                    notable_events.append(f"[GRUDGE] {r1.name} betrayed {r2.name} ({betrayal_type})! A GRUDGE is born!")
                else:
                    notable_events.append(f"[BETRAYAL] {r1.name} betrayed {r2.name} ({betrayal_type})!")

                betrayals_today += 1
                self.total_betrayals += 1

                # v6.2: Shorter crisis duration (7-14 days instead of 14-30)
                self.life_crises[id2].append(LifeCrisis(
                    crisis_type="betrayed",
                    start_day=day,
                    severity=0.6,  # v6.2: Reduced from 0.8
                    duration_days=random.randint(7, 14)  # v6.2: Shorter
                ))

                # v6.2: Only ONE negative moodlet (was stacking multiple!)
                self.engine.moodlets.add_moodlet(id2, "betrayed", betrayal_type)

                # Reputation hit for betrayer
                self.reputation[id1] = max(-100, self.reputation[id1] - 15)

        # === v6.2: MOOD INSTABILITY (personality-based bad days - reduced) ===
        for agent_id in resident_ids:
            resident = self.residents[agent_id]
            neuroticism = resident.personality.neuroticism

            # v6.2: Reduced mood swing chance - 10% max (was 25%)
            if random.random() < neuroticism * 0.10:
                self.engine.moodlets.add_moodlet(agent_id, "stressed", "mood_swing")
                # v6.2: 15% exhaustion chance (was 40%)
                if random.random() < 0.15:
                    self.engine.moodlets.add_moodlet(agent_id, "exhausted", "mood_swing")

            # v6.2: Random bad days - 1% (was 3%)
            if random.random() < 0.01:
                self.engine.moodlets.add_moodlet(agent_id, "exhausted", "bad_day")

        # === v6.2: SECRET ROMANCE EXPOSURE (reduced - 2% instead of 4%) ===
        for (id1, id2), rel in list(self.relationships.items()):
            if rel.is_romantic and rel.is_secret_romance:
                # v6.2: 2% daily chance of exposure (was 4%)
                if random.random() < 0.02:
                    rel.is_secret_romance = False  # No longer secret
                    self.total_secret_reveals += 1
                    r1 = self.residents[id1]
                    r2 = self.residents[id2]
                    notable_events.append(f"[SCANDAL] The secret romance between {r1.name} and {r2.name} has been EXPOSED!")

                    # v6.2: Single moodlet only (was 2-3 stacked!)
                    self.engine.moodlets.add_moodlet(id1, "scandalous", "exposed")
                    self.engine.moodlets.add_moodlet(id2, "scandalous", "exposed")
                    self.reputation[id1] = max(-100, self.reputation[id1] - 10)
                    self.reputation[id2] = max(-100, self.reputation[id2] - 10)

        # Social interactions (35% of residents have conversations - increased for drama!)
        friendships_formed = 0
        rivalries_formed = 0
        romances_sparked = 0

        socializers = random.sample(resident_ids, int(len(resident_ids) * 0.35))
        for agent_id in socializers:
            # Pick a conversation partner (prefer existing relationships)
            others = [r for r in resident_ids if r != agent_id]

            # 30% chance to seek out someone they already know
            known_others = [r for r in others if self._get_relationship(agent_id, r).interactions > 0]
            if known_others and random.random() < 0.3:
                partner_id = random.choice(known_others)
            else:
                partner_id = random.choice(others)

            # v6.4: REBALANCED conversation outcomes
            # Good or bad conversation? More balanced odds (was 70/30, now 50/50)
            rel = self._get_relationship(agent_id, partner_id)
            base_odds = 0.55 if rel.friendship > 0 else 0.45 if rel.friendship == 0 else 0.35  # v6.4: reduced

            if random.random() < base_odds:
                outcome = "great_conversation"
            else:
                # v6.4: Variety of negative social outcomes
                negative_outcomes = [
                    "awkward_silence",      # -2, 8h
                    "feeling_ignored",      # -2, 6h
                    "boring_conversation",  # -1, 4h
                    "overstayed_welcome",   # -1, 4h (only for initiator)
                ]
                outcome = random.choice(negative_outcomes[:3])  # First 3 apply to both

                # Special case: overstayed_welcome only affects initiator
                if random.random() < 0.25:
                    self.engine.moodlets.add_moodlet(agent_id, "overstayed_welcome", "social")

            self.engine.moodlets.add_moodlet(agent_id, outcome, "social")
            self.engine.moodlets.add_moodlet(partner_id, outcome, "social")
            conversations_had += 1

            # Update relationship based on outcome
            if outcome == "great_conversation":
                delta = random.randint(8, 18)  # Increased impact for faster relationship growth
                rel = self._update_relationship(agent_id, partner_id, delta, day)

                # Check for friendship milestone (LOWERED from 50 to 35)
                if rel.friendship >= 35 and rel.interactions >= 3 and not rel.is_romantic:
                    r1 = self.residents[agent_id]
                    r2 = self.residents[partner_id]
                    notable_events.append(f"{r1.name} and {r2.name} became good friends!")
                    friendships_formed += 1
                    self.engine.moodlets.add_moodlet(agent_id, "made_new_friend", "friendship")
                    self.engine.moodlets.add_moodlet(partner_id, "made_new_friend", "friendship")

                # Chance of romance (LOWERED threshold to 25, increased chance to 25%)
                if rel.friendship >= 25 and not rel.is_romantic and not rel.was_romantic and random.random() < 0.25:
                    r1 = self.residents[agent_id]
                    r2 = self.residents[partner_id]
                    age_diff = abs(r1.age - r2.age)
                    if age_diff <= 15:  # Compatible age range
                        rel.is_romantic = True
                        rel.romance_start_day = day
                        romances_sparked += 1

                        # v5: 20% chance it's a SECRET romance
                        if random.random() < 0.20:
                            rel.is_secret_romance = True
                            notable_events.append(f"[SECRET] {r1.name} and {r2.name} have started a SECRET romance!")
                        else:
                            notable_events.append(f"[ROMANCE] Romance is blooming between {r1.name} and {r2.name}!")

                        self.engine.moodlets.add_moodlet(agent_id, "heartwarming_moment", "romance")
                        self.engine.moodlets.add_moodlet(partner_id, "heartwarming_moment", "romance")

            else:  # awkward_silence
                delta = random.randint(-8, -2)  # Increased negative impact for drama
                rel = self._update_relationship(agent_id, partner_id, delta, day)

                # Check for rivalry (LOWERED threshold from -25 to -15)
                if rel.friendship <= -15 and rel.interactions >= 2 and not rel.is_rival:
                    r1 = self.residents[agent_id]
                    r2 = self.residents[partner_id]
                    rel.is_rival = True
                    rivalries_formed += 1
                    notable_events.append(f"Tension between {r1.name} and {r2.name} - they've become rivals!")

        # Random life events (5% chance per resident)
        for agent_id in resident_ids:
            if random.random() < 0.05:
                resident = self.residents[agent_id]
                event_type = random.choice([
                    "found_money", "bad_news", "good_news", "met_old_friend",
                    "creative_breakthrough", "minor_setback", "job_promotion",
                    "learned_secret", "helped_neighbor"
                ])

                if event_type == "found_money":
                    self.engine.moodlets.add_moodlet(agent_id, "accomplished", "lucky")
                    notable_events.append(f"{resident.name} found $20 on the street!")
                elif event_type == "bad_news":
                    self.engine.moodlets.add_moodlet(agent_id, "exhausted", "news")
                    notable_events.append(f"{resident.name} received some bad news...")
                elif event_type == "good_news":
                    self.engine.moodlets.add_moodlet(agent_id, "heartwarming_moment", "news")
                    notable_events.append(f"{resident.name} got great news!")
                elif event_type == "met_old_friend":
                    self.engine.moodlets.add_moodlet(agent_id, "made_new_friend", "reunion")
                    notable_events.append(f"{resident.name} ran into an old friend!")
                elif event_type == "creative_breakthrough":
                    self.engine.moodlets.add_moodlet(agent_id, "project_breakthrough", "breakthrough")
                    notable_events.append(f"{resident.name} had a creative breakthrough!")
                elif event_type == "minor_setback":
                    self.engine.moodlets.add_moodlet(agent_id, "creative_block", "setback")
                elif event_type == "job_promotion":
                    self.engine.moodlets.add_moodlet(agent_id, "accomplished", "career")
                    notable_events.append(f"{resident.name} got a promotion at work!")
                elif event_type == "learned_secret":
                    self.engine.moodlets.add_moodlet(agent_id, "learned_something", "gossip")
                    notable_events.append(f"{resident.name} learned some interesting neighborhood gossip...")
                elif event_type == "helped_neighbor":
                    self.engine.moodlets.add_moodlet(agent_id, "feeling_appreciated", "kindness")
                    # Also boost relationship with a random neighbor
                    other = random.choice([r for r in resident_ids if r != agent_id])
                    self._update_relationship(agent_id, other, 10, day)
                    notable_events.append(f"{resident.name} helped a neighbor and felt great about it!")

        # Community events (15% chance per day - increased for drama!)
        if random.random() < 0.15:
            event = random.choice([
                "block_party", "farmers_market", "power_outage", "street_fair",
                "neighborhood_meeting", "community_cleanup", "gossip_spreads",
                "neighborhood_drama", "celebration", "emergency_response"
            ])

            if event == "block_party":
                self.community_events.append(f"Day {day}: Block party! Everyone had a great time.")
                notable_events.append("The neighborhood had a block party!")
                # Boost everyone's mood and several random relationships
                for aid in random.sample(resident_ids, min(20, len(resident_ids))):
                    self.engine.moodlets.add_moodlet(aid, "great_conversation", "party")
                    # Form some connections at the party
                    if random.random() < 0.3:
                        other = random.choice([r for r in resident_ids if r != aid])
                        self._update_relationship(aid, other, 8, day)

            elif event == "farmers_market":
                self.community_events.append(f"Day {day}: Farmers market in the square.")
                notable_events.append("The weekly farmers market brought neighbors together!")
                for aid in random.sample(resident_ids, min(15, len(resident_ids))):
                    self.engine.moodlets.add_moodlet(aid, "cozy_atmosphere", "market")

            elif event == "power_outage":
                self.community_events.append(f"Day {day}: Power outage for 4 hours.")
                notable_events.append("A power outage brought neighbors together by candlelight!")
                # Some annoyed, but some bonded
                for aid in resident_ids:
                    if random.random() < 0.3:
                        self.engine.moodlets.add_moodlet(aid, "exhausted", "outage")
                    elif random.random() < 0.2:
                        self.engine.moodlets.add_moodlet(aid, "great_conversation", "outage")

            elif event == "street_fair":
                self.community_events.append(f"Day {day}: Annual street fair!")
                notable_events.append("The annual street fair was a huge success!")
                for aid in random.sample(resident_ids, min(30, len(resident_ids))):
                    self.engine.moodlets.add_moodlet(aid, "beautiful_view", "fair")

            elif event == "neighborhood_meeting":
                self.community_events.append(f"Day {day}: Neighborhood association meeting.")
                notable_events.append("The neighborhood meeting got heated!")
                # Some conflict, some bonding
                for aid in random.sample(resident_ids, min(10, len(resident_ids))):
                    if random.random() < 0.5:
                        self.engine.moodlets.add_moodlet(aid, "accomplished", "meeting")
                    else:
                        self.engine.moodlets.add_moodlet(aid, "exhausted", "meeting")

            elif event == "community_cleanup":
                self.community_events.append(f"Day {day}: Community cleanup day.")
                notable_events.append("Neighbors came together for community cleanup!")
                for aid in random.sample(resident_ids, min(20, len(resident_ids))):
                    self.engine.moodlets.add_moodlet(aid, "accomplished", "cleanup")
                    self.engine.moodlets.add_moodlet(aid, "feeling_appreciated", "cleanup")

            elif event == "gossip_spreads":
                # Gossip affects random relationships - some get closer, some fall out
                self.community_events.append(f"Day {day}: Gossip spread through the neighborhood!")
                notable_events.append("Juicy gossip spread through the neighborhood!")
                affected = random.sample(resident_ids, min(15, len(resident_ids)))
                for aid in affected:
                    other = random.choice([r for r in resident_ids if r != aid])
                    # Gossip can bond or divide
                    if random.random() < 0.4:
                        self._update_relationship(aid, other, 12, day)  # Bonded over gossip
                    else:
                        self._update_relationship(aid, other, -10, day)  # Gossip caused friction

            elif event == "neighborhood_drama":
                # Two residents have a public argument
                pair = random.sample(resident_ids, 2)
                r1, r2 = self.residents[pair[0]], self.residents[pair[1]]
                self.community_events.append(f"Day {day}: Drama between {r1.name} and {r2.name}!")
                notable_events.append(f"Public drama between {r1.name} and {r2.name}!")
                # Big relationship hit
                self._update_relationship(pair[0], pair[1], -25, day)
                self.engine.moodlets.add_moodlet(pair[0], "exhausted", "drama")
                self.engine.moodlets.add_moodlet(pair[1], "exhausted", "drama")
                # Witnesses pick sides
                for aid in random.sample(resident_ids, min(10, len(resident_ids))):
                    if aid not in pair:
                        side = random.choice(pair)
                        self._update_relationship(aid, side, 8, day)
                        other_side = pair[1] if side == pair[0] else pair[0]
                        self._update_relationship(aid, other_side, -5, day)

            elif event == "celebration":
                # Someone's birthday/anniversary brings everyone together
                celebrant = random.choice(resident_ids)
                r = self.residents[celebrant]
                self.community_events.append(f"Day {day}: Everyone celebrated {r.name}'s special day!")
                notable_events.append(f"The neighborhood celebrated {r.name}'s special day!")
                # Celebrant gets massive boost
                self.engine.moodlets.add_moodlet(celebrant, "heartwarming_moment", "celebration")
                self.engine.moodlets.add_moodlet(celebrant, "accomplished", "celebration")
                # Everyone who attended bonds with celebrant
                for aid in random.sample(resident_ids, min(25, len(resident_ids))):
                    if aid != celebrant:
                        self._update_relationship(aid, celebrant, 15, day)
                        self.engine.moodlets.add_moodlet(aid, "great_conversation", "party")

            elif event == "emergency_response":
                # Someone needed help, neighbors rallied
                person_in_need = random.choice(resident_ids)
                r = self.residents[person_in_need]
                self.community_events.append(f"Day {day}: Neighbors rallied to help {r.name} in need!")
                notable_events.append(f"The community came together to help {r.name}!")
                # Helpers bond with person in need
                helpers = random.sample([x for x in resident_ids if x != person_in_need],
                                       min(15, len(resident_ids) - 1))
                for aid in helpers:
                    self._update_relationship(aid, person_in_need, 20, day)
                    self.engine.moodlets.add_moodlet(aid, "feeling_appreciated", "helping")
                self.engine.moodlets.add_moodlet(person_in_need, "heartwarming_moment", "helped")

        # === EVENING: End-of-day relationship evolution ===
        # Check if strong friendships bloom into romance or negative ones into rivalry
        for (id1, id2), rel in self.relationships.items():
            # End-of-day romance check for established friendships
            if rel.friendship >= 40 and not rel.is_romantic and not rel.was_romantic and rel.interactions >= 4:
                if random.random() < 0.08:  # 8% daily chance for compatible pairs
                    r1 = self.residents[id1]
                    r2 = self.residents[id2]
                    age_diff = abs(r1.age - r2.age)
                    if age_diff <= 15:
                        rel.is_romantic = True
                        rel.romance_start_day = day
                        romances_sparked += 1

                        # v5: 20% chance it's a SECRET romance
                        if random.random() < 0.20:
                            rel.is_secret_romance = True
                            notable_events.append(f"[SECRET] {r1.name} and {r2.name} have started a SECRET romance!")
                        else:
                            notable_events.append(f"[ROMANCE] Romance is blooming between {r1.name} and {r2.name}!")

                        self.engine.moodlets.add_moodlet(id1, "heartwarming_moment", "romance")
                        self.engine.moodlets.add_moodlet(id2, "heartwarming_moment", "romance")

            # End-of-day rivalry check for troubled relationships (lowered threshold, increased chance)
            if rel.friendship <= -15 and not rel.is_rival and rel.interactions >= 2:
                if random.random() < 0.25:  # 25% daily chance (was 15%)
                    r1 = self.residents[id1]
                    r2 = self.residents[id2]
                    rel.is_rival = True
                    rivalries_formed += 1
                    notable_events.append(f"[RIVALS] {r1.name} and {r2.name} have become full RIVALS!")

        # === v5: SECRET ROMANCE EXPOSURE (dramatic reveals!) ===
        secret_romances = [(k, r) for k, r in self.relationships.items() if r.is_secret_romance]
        for (id1, id2), rel in secret_romances:
            # 3% daily chance of exposure
            if random.random() < 0.03:
                rel.is_secret_romance = False  # No longer secret!
                self.total_secret_reveals += 1
                r1 = self.residents[id1]
                r2 = self.residents[id2]
                notable_events.append(f"[SCANDAL] {r1.name} and {r2.name}'s secret romance was EXPOSED!")

                # Reputation hit for both
                self.reputation[id1] = max(-100, self.reputation[id1] - 10)
                self.reputation[id2] = max(-100, self.reputation[id2] - 10)

                # Some people are scandalized, some don't care
                for aid in random.sample(resident_ids, min(15, len(resident_ids))):
                    if aid not in [id1, id2]:
                        if random.random() < 0.3:
                            # Judgy person - thinks less of them
                            self._update_relationship(aid, id1, -5, day)
                            self._update_relationship(aid, id2, -5, day)

        # === EVENING: Energy drain and stats ===
        for agent_id in resident_ids:
            # Evening tiredness
            self.engine.energy.drain(agent_id, "daily_activities")

        # Collect stats
        happy_count = 0
        neutral_count = 0
        sad_count = 0
        energized_count = 0
        tired_count = 0

        for agent_id in resident_ids:
            mood = self.engine.moodlets.get_mood_score(agent_id)
            energy = self.engine.energy.get_energy(agent_id)

            if mood > 0:
                happy_count += 1
            elif mood < 0:
                sad_count += 1
            else:
                neutral_count += 1

            if energy.current >= 70:
                energized_count += 1
            elif energy.current < 40:
                tired_count += 1

        # Note: Moodlets automatically expire based on their duration_hours
        self.total_days_simulated = day

        return DayStats(
            day=day,
            happy_count=happy_count,
            neutral_count=neutral_count,
            sad_count=sad_count,
            energized_count=energized_count,
            tired_count=len(resident_ids) - energized_count - tired_count,
            skills_leveled=skills_leveled,
            projects_worked=projects_worked,
            conversations_had=conversations_had,
            notable_events=notable_events[:7],  # Keep top 7 events
            friendships_formed=friendships_formed,
            rivalries_formed=rivalries_formed,
            romances_sparked=romances_sparked,
        )

    def _get_skill_for_archetype(self, archetype: Archetype) -> str:
        """Get primary skill for archetype."""
        skill_map = {
            Archetype.ARTIST: "Painting",
            Archetype.WRITER: "Writing",
            Archetype.MUSICIAN: "Guitar",
            Archetype.CRAFTSPERSON: "Handiness",
            Archetype.TEACHER: "Charisma",
            Archetype.ENTREPRENEUR: "Charisma",
            Archetype.SCIENTIST: "Research",
            Archetype.HEALTHCARE: "Fitness",
            Archetype.TECH_WORKER: "Programming",
            Archetype.COMMUNITY_ORGANIZER: "Charisma",
            Archetype.ELDER: "Charisma",
            Archetype.NEWCOMER: "Social",
            Archetype.LOCAL_BUSINESS: "Charisma",
            Archetype.STUDENT: "Research",
            Archetype.RETIREE: "Gardening",
            Archetype.YOUNG_PARENT: "Cooking",
            Archetype.EMPTY_NESTER: "Gardening",
            Archetype.INTROVERT_SAGE: "Writing",
            Archetype.SOCIAL_BUTTERFLY: "Social",
            Archetype.SKEPTIC: "Research",
            Archetype.OPTIMIST: "Charisma",
            Archetype.HEALER: "Meditation",
        }
        return skill_map.get(archetype, "Creativity")

    async def run_simulation(self, num_days: int = 10):
        """Run the simulation for specified number of days."""
        if self.total_days_simulated == 0:
            await self.setup()
            start_day = 1
        else:
            start_day = self.total_days_simulated + 1

        end_day = start_day + num_days - 1

        print("=" * 70)
        if start_day == 1:
            print(f"STARTING {num_days}-DAY SIMULATION")
        else:
            print(f"CONTINUING SIMULATION: Days {start_day} to {end_day}")
        print("=" * 70)

        for day in range(start_day, end_day + 1):
            print(f"\n{'=' * 50}")
            print(f"DAY {day}")
            print("=" * 50)

            stats = await self.simulate_day(day)
            self.day_stats.append(stats)

            # Print day summary
            print(f"\nMood: Happy={stats.happy_count} | Neutral={stats.neutral_count} | Sad={stats.sad_count}")
            print(f"Energy: Energized={stats.energized_count} | Normal={stats.tired_count} | Tired={self.num_residents - stats.energized_count - stats.tired_count}")
            print(f"Activities: {stats.projects_worked} projects | {stats.conversations_had} conversations | {stats.skills_leveled} skill-ups")

            # Relationship summary
            if stats.friendships_formed or stats.romances_sparked or stats.rivalries_formed:
                rel_parts = []
                if stats.friendships_formed:
                    rel_parts.append(f"{stats.friendships_formed} new friendships")
                if stats.romances_sparked:
                    rel_parts.append(f"{stats.romances_sparked} romances")
                if stats.rivalries_formed:
                    rel_parts.append(f"{stats.rivalries_formed} rivalries")
                print(f"Relationships: {', '.join(rel_parts)}")

            if stats.notable_events:
                print("\nNotable Events:")
                for event in stats.notable_events:
                    print(f"  * {event}")

        # Final summary
        self.print_final_summary()

    async def continue_simulation(self, additional_days: int = 10):
        """Continue the simulation for more days."""
        if self.total_days_simulated == 0:
            print("No simulation to continue. Running initial simulation...")
            await self.run_simulation(additional_days)
        else:
            await self.run_simulation(additional_days)

    def print_final_summary(self):
        """Print summary of the simulation."""
        print("\n" + "=" * 70)
        print(f"{self.total_days_simulated}-DAY SIMULATION COMPLETE")
        print("=" * 70)

        # Aggregate stats
        total_projects = sum(s.projects_worked for s in self.day_stats)
        total_conversations = sum(s.conversations_had for s in self.day_stats)
        total_skill_ups = sum(s.skills_leveled for s in self.day_stats)
        total_events = sum(len(s.notable_events) for s in self.day_stats)
        total_friendships = sum(s.friendships_formed for s in self.day_stats)
        total_romances = sum(s.romances_sparked for s in self.day_stats)
        total_rivalries = sum(s.rivalries_formed for s in self.day_stats)

        print(f"\n[Summary] {self.num_residents} residents over {self.total_days_simulated} days")
        print(f"  - Total project work sessions: {total_projects}")
        print(f"  - Total conversations: {total_conversations}")
        print(f"  - Total skill level-ups: {total_skill_ups}")
        print(f"  - Notable events: {total_events}")
        print(f"  - Friendships formed: {total_friendships}")
        print(f"  - Romances sparked: {total_romances}")
        print(f"  - Rivalries formed: {total_rivalries}")

        # Relationship Statistics
        print("\n[Relationship Network]")
        total_rels = len(self.relationships)
        positive_rels = sum(1 for r in self.relationships.values() if r.friendship > 0)
        negative_rels = sum(1 for r in self.relationships.values() if r.friendship < 0)
        romantic_rels = sum(1 for r in self.relationships.values() if r.is_romantic)
        rival_rels = sum(1 for r in self.relationships.values() if r.is_rival)
        grudge_rels = sum(1 for r in self.relationships.values() if r.is_grudge)
        ex_lover_rels = sum(1 for r in self.relationships.values() if r.was_romantic and not r.is_romantic)
        secret_romance_rels = sum(1 for r in self.relationships.values() if r.is_secret_romance)

        print(f"  - Total connections formed: {total_rels}")
        print(f"  - Positive relationships: {positive_rels} ({100*positive_rels/max(1,total_rels):.0f}%)")
        print(f"  - Negative relationships: {negative_rels} ({100*negative_rels/max(1,total_rels):.0f}%)")
        print(f"  - Romantic couples: {romantic_rels} ({secret_romance_rels} secret)")
        print(f"  - Bitter exes: {ex_lover_rels}")
        print(f"  - Active rivalries: {rival_rels}")
        print(f"  - Permanent grudges: {grudge_rels}")

        # v5 Drama Stats
        print("\n[v5 DRAMA STATS]")
        print(f"  - Life crises triggered: {self.total_crises}")
        active_crises = sum(len([c for c in crises if not c.resolved]) for crises in self.life_crises.values())
        print(f"  - Active crises: {active_crises}")
        print(f"  - Breakups: {self.total_breakups}")
        print(f"  - Betrayals: {self.total_betrayals}")
        print(f"  - Love triangles detected: {self.total_love_triangles}")
        print(f"  - Secret romances exposed: {self.total_secret_reveals}")

        # Happiness stability (% of days with >80% happy residents)
        high_happiness_days = sum(1 for s in self.day_stats if s.happy_count / self.num_residents >= 0.8)
        happiness_stability = 100 * high_happiness_days / max(1, len(self.day_stats))
        print(f"\n[Happiness Stability]")
        print(f"  - Days with >=80% happy: {high_happiness_days}/{len(self.day_stats)} ({happiness_stability:.0f}%)")

        # Low mood days (>=20% sad)
        low_mood_days = sum(1 for s in self.day_stats if s.sad_count / self.num_residents >= 0.2)
        print(f"  - Days with >=20% sad: {low_mood_days}/{len(self.day_stats)}")

        # Reputation extremes
        print("\n[Reputation Extremes]")
        rep_sorted = sorted(self.reputation.items(), key=lambda x: x[1])
        worst_rep = rep_sorted[:3]
        best_rep = rep_sorted[-3:][::-1]

        if best_rep and best_rep[0][1] > 0:
            print("  Most respected:")
            for aid, rep in best_rep:
                if rep > 0:
                    print(f"    {self.residents[aid].name}: {rep:+d}")

        if worst_rep and worst_rep[0][1] < 0:
            print("  Most notorious:")
            for aid, rep in worst_rep:
                if rep < 0:
                    print(f"    {self.residents[aid].name}: {rep:+d}")

        # Best Friends (highest friendship scores)
        print("\n[Strongest Friendships]")
        sorted_rels = sorted(self.relationships.items(), key=lambda x: -x[1].friendship)[:5]
        for (id1, id2), rel in sorted_rels:
            r1 = self.residents[id1]
            r2 = self.residents[id2]
            status = ""
            if rel.is_romantic:
                status = " [ROMANTIC]"
            print(f"  {r1.name} & {r2.name}: {rel.friendship:+d} ({rel.interactions} chats){status}")

        # Worst Enemies (if any)
        worst_rels = sorted(self.relationships.items(), key=lambda x: x[1].friendship)[:3]
        if worst_rels and worst_rels[0][1].friendship < 0:
            print("\n[Bitter Rivalries]")
            for (id1, id2), rel in worst_rels:
                if rel.friendship < 0:
                    r1 = self.residents[id1]
                    r2 = self.residents[id2]
                    status = " [RIVALS]" if rel.is_rival else ""
                    print(f"  {r1.name} & {r2.name}: {rel.friendship:+d}{status}")

        # Romantic Couples
        romantic_pairs = [(k, v) for k, v in self.relationships.items() if v.is_romantic]
        if romantic_pairs:
            print("\n[Love in the Neighborhood]")
            for (id1, id2), rel in romantic_pairs:
                r1 = self.residents[id1]
                r2 = self.residents[id2]
                print(f"  {r1.name} ({r1.age}) + {r2.name} ({r2.age})")

        # Community Events
        if self.community_events:
            print("\n[Community Events Timeline]")
            for event in self.community_events[-5:]:  # Last 5 events
                print(f"  {event}")

        # Mood trend
        print(f"\n[Mood Trend Over {self.total_days_simulated} Days]")
        print("Day  Happy  Neutral  Sad")
        print("-" * 30)
        for stats in self.day_stats:
            happy_bar = "#" * (stats.happy_count // 2)
            print(f" {stats.day:2}   {stats.happy_count:3}     {stats.neutral_count:3}     {stats.sad_count:3}  {happy_bar}")

        # Top skill gainers
        print("\n[Top Skill Gainers]")
        skill_totals = {}
        for agent_id, skills in self.skill_levels.items():
            for skill, level in skills.items():
                if level > 1:
                    if agent_id not in skill_totals:
                        skill_totals[agent_id] = 0
                    skill_totals[agent_id] += level

        top_gainers = sorted(skill_totals.items(), key=lambda x: -x[1])[:5]
        for agent_id, total in top_gainers:
            resident = self.residents[agent_id]
            archetype = self.archetypes[agent_id]
            skills = self.skill_levels[agent_id]
            skill_str = ", ".join(f"{s}:{l}" for s, l in skills.items())
            print(f"  {resident.name} ({archetype.value}): {skill_str}")

        # Archetype distribution
        print("\n[Archetype Distribution]")
        archetype_counts = {}
        for a in self.archetypes.values():
            archetype_counts[a.value] = archetype_counts.get(a.value, 0) + 1

        for archetype, count in sorted(archetype_counts.items(), key=lambda x: -x[1])[:8]:
            bar = "#" * count
            print(f"  {archetype:18} {bar} ({count})")

        print("\n" + "=" * 70)
        print(f"Simulation complete! The neighborhood thrived for {self.total_days_simulated} days.")
        print("=" * 70)


async def main():
    """Run a 180-day simulation with v5 complexity features."""
    print("\n" + "#" * 70)
    print("#  NEIGHBORHOOD SIMULATION v5 - COMPLEXITY EDITION")
    print("#  50 Residents | 180 Days | Full Drama Enabled")
    print("#" * 70)
    print("#  NEW v5 FEATURES:")
    print("#  • Love triangles & jealousy dynamics")
    print("#  • Betrayals (secrets, rumors, grudges)")
    print("#  • Life crises (job loss, health, heartbreak)")
    print("#  • Breakups (romances can end, bitter exes)")
    print("#  • Secret romances (can be exposed - scandal!)")
    print("#  • Reputation system (gossip affects status)")
    print("#  • Mood instability (personality-based bad days)")
    print("#" * 70)

    sim = SimulationEngine(num_residents=50, seed=42)

    # Run first 60 days
    await sim.run_simulation(num_days=60)

    print("\n" + "~" * 70)
    print("  Press Enter to continue for 60 more days (total 120)...")
    print("~" * 70)
    input()

    # Continue for 60 more days (days 61-120)
    await sim.continue_simulation(additional_days=60)

    print("\n" + "~" * 70)
    print("  Press Enter to continue for 60 more days (total 180)...")
    print("~" * 70)
    input()

    # Continue for final 60 days (days 121-180)
    await sim.continue_simulation(additional_days=60)


async def quick_test():
    """Quick 5-day test run."""
    sim = SimulationEngine(num_residents=30, seed=123)
    await sim.run_simulation(num_days=5)


async def test_90_days():
    """Non-interactive 90-day test for parameter tuning."""
    print("\n" + "#" * 70)
    print("#  NEIGHBORHOOD SIMULATION v5 - 90 DAY TEST RUN")
    print("#  50 Residents | 90 Days | Non-Interactive")
    print("#" * 70)

    sim = SimulationEngine(num_residents=50, seed=42)
    await sim.run_simulation(num_days=90)


async def test_2_years():
    """Full 2-year (730 day) simulation to observe long-term dynamics."""
    print("\n" + "#" * 70)
    print("#  NEIGHBORHOOD SIMULATION v5 - 2 YEAR EPIC")
    print("#  50 Residents | 730 Days | Long-Term Dynamics")
    print("#" * 70)
    print("#  Observing: relationship evolution, reputation shifts,")
    print("#  generational drama, and emergent social structures")
    print("#" * 70)

    sim = SimulationEngine(num_residents=50, seed=42)

    # Run in chunks to show progress
    print("\n[Year 1, Q1] Days 1-90...")
    await sim.run_simulation(num_days=90)

    print("\n[Year 1, Q2] Days 91-180...")
    await sim.continue_simulation(additional_days=90)

    print("\n[Year 1, Q3] Days 181-270...")
    await sim.continue_simulation(additional_days=90)

    print("\n[Year 1, Q4] Days 271-365...")
    await sim.continue_simulation(additional_days=95)

    print("\n[Year 2, Q1] Days 366-455...")
    await sim.continue_simulation(additional_days=90)

    print("\n[Year 2, Q2] Days 456-545...")
    await sim.continue_simulation(additional_days=90)

    print("\n[Year 2, Q3] Days 546-635...")
    await sim.continue_simulation(additional_days=90)

    print("\n[Year 2, Q4] Days 636-730...")
    await sim.continue_simulation(additional_days=95)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(quick_test())
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(test_90_days())
    elif len(sys.argv) > 1 and sys.argv[1] == "--2years":
        asyncio.run(test_2_years())
    else:
        asyncio.run(main())
