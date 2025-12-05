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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from resident_generator import (
    ResidentGenerator,
    GenerationConfig,
    Archetype,
)
from sims_inspired import SimsEnhancedEngine, LifeGoal, DEFAULT_LIFE_GOALS
from neighborhood_engine import AgentState


# =============================================================================
# v7: SEASONAL SYSTEM (November 2025)
# =============================================================================

class Season(Enum):
    """Four seasons of the year."""
    WINTER = "winter"   # Dec-Feb (Days 335-59)
    SPRING = "spring"   # Mar-May (Days 60-151)
    SUMMER = "summer"   # Jun-Aug (Days 152-243)
    AUTUMN = "autumn"   # Sep-Nov (Days 244-334)


def get_season(day: int) -> Season:
    """Get the season for a given simulation day (1-365 cycle)."""
    day_of_year = ((day - 1) % 365) + 1  # 1-365

    if day_of_year >= 335 or day_of_year < 60:
        return Season.WINTER
    elif day_of_year < 152:
        return Season.SPRING
    elif day_of_year < 244:
        return Season.SUMMER
    else:
        return Season.AUTUMN


def get_season_name(day: int) -> str:
    """Get human-readable season name with month range."""
    season = get_season(day)
    return {
        Season.WINTER: "Winter (Dec-Feb)",
        Season.SPRING: "Spring (Mar-May)",
        Season.SUMMER: "Summer (Jun-Aug)",
        Season.AUTUMN: "Autumn (Sep-Nov)",
    }[season]


# Seasonal moodlet pools - weighted (positive, negative)
SEASONAL_MOODLETS = {
    Season.WINTER: {
        "positive": ["winter_cozy", "holiday_cheer"],
        "negative": ["winter_blues", "cabin_fever"],
        "positive_weight": 0.4,  # 40% chance of positive
        "negative_weight": 0.3,  # 30% chance of negative
    },
    Season.SPRING: {
        "positive": ["spring_renewal", "spring_cleaning"],
        "negative": ["spring_allergies"],
        "positive_weight": 0.5,  # Spring is generally uplifting
        "negative_weight": 0.2,
    },
    Season.SUMMER: {
        "positive": ["summer_vibes", "vacation_mode"],
        "negative": ["heat_exhaustion"],
        "positive_weight": 0.5,
        "negative_weight": 0.25,
    },
    Season.AUTUMN: {
        "positive": ["harvest_gratitude", "cozy_sweater"],
        "negative": ["autumn_melancholy"],
        "positive_weight": 0.45,
        "negative_weight": 0.25,
    },
}


# =============================================================================
# v7: MAJOR LIFE EVENTS (November 2025)
# =============================================================================

@dataclass
class MajorLifeEvent:
    """Tracks a major life event for a resident."""
    event_type: str  # marriage, birth, death, promotion, retirement, divorce
    day: int
    participants: List[str] = field(default_factory=list)  # Other people involved
    details: str = ""


# Neighborhood-wide events
@dataclass
class NeighborhoodEvent:
    """Tracks a neighborhood-wide event."""
    event_type: str  # block_party, festival, crisis, power_outage
    day: int
    affected_residents: List[str] = field(default_factory=list)
    details: str = ""


# =============================================================================
# v8.0: ECONOMIC ECOSYSTEM (November 2025)
# =============================================================================

@dataclass
class FinancialState:
    """Tracks a resident's financial situation."""
    income: float = 50000.0  # Annual income
    savings: float = 10000.0  # Current savings
    debt: float = 0.0  # Outstanding debt
    monthly_expenses: float = 3000.0  # Fixed monthly costs
    financial_stress: float = 0.0  # 0.0-1.0 stress level
    last_paycheck_day: int = 0  # For bi-weekly paycheck tracking
    job_stability: float = 0.8  # 0.0-1.0 (1.0 = very stable)
    is_employed: bool = True
    unemployment_day: int = 0  # Day they lost job (if unemployed)

    def net_worth(self) -> float:
        """Calculate net worth (savings - debt)."""
        return self.savings - self.debt

    def calculate_stress(self) -> float:
        """Calculate financial stress based on situation."""
        stress = 0.0

        # Debt-to-income ratio stress
        if self.income > 0:
            debt_ratio = self.debt / self.income
            if debt_ratio > 0.5:
                stress += 0.3
            elif debt_ratio > 0.3:
                stress += 0.15

        # Low savings stress (< 3 months expenses)
        safety_net = self.savings / max(self.monthly_expenses, 1)
        if safety_net < 1:
            stress += 0.4
        elif safety_net < 3:
            stress += 0.2
        elif safety_net < 6:
            stress += 0.1

        # Unemployment stress
        if not self.is_employed:
            stress += 0.5

        # Job instability stress
        if self.job_stability < 0.5:
            stress += 0.2

        return min(1.0, stress)


# Income brackets by archetype (annual income)
ARCHETYPE_INCOME = {
    "artist": (25000, 60000),
    "writer": (30000, 80000),
    "musician": (20000, 70000),
    "teacher": (40000, 70000),
    "entrepreneur": (30000, 150000),
    "elder": (20000, 50000),  # Fixed income
    "newcomer": (35000, 65000),
    "healer": (45000, 90000),
    "student": (10000, 30000),  # Part-time work
    "retiree": (25000, 60000),  # Pension/savings
    "healthcare": (50000, 120000),
    "local_business": (40000, 100000),
    "craftsperson": (30000, 70000),
    "scientist": (60000, 130000),
    "tech_worker": (70000, 180000),
    "community_organizer": (35000, 60000),
    "young_parent": (40000, 80000),
    "empty_nester": (50000, 100000),
    "introvert_sage": (40000, 80000),
    "social_butterfly": (45000, 85000),
    "skeptic": (45000, 90000),
    "optimist": (40000, 75000),
}

# Economic event types
ECONOMIC_EVENTS = {
    "bonus": {"chance": 0.02, "amount_range": (500, 5000)},  # 2% daily chance
    "unexpected_expense": {"chance": 0.03, "amount_range": (200, 2000)},
    "inheritance": {"chance": 0.001, "amount_range": (5000, 50000)},  # Rare windfall
    "medical_bill": {"chance": 0.01, "amount_range": (500, 10000)},
    "car_repair": {"chance": 0.015, "amount_range": (200, 3000)},
    "raise": {"chance": 0.005, "percentage": (0.03, 0.10)},  # 3-10% raise
    "job_loss": {"chance": 0.002},  # 0.2% daily (rare but impactful)
    "new_job": {"chance": 0.01},  # For unemployed residents
}


# =============================================================================
# v8.1: SOCIAL NETWORKS (November 2025)
# =============================================================================

class GroupType(Enum):
    """Types of social groups that can form."""
    HOBBY_CLUB = "hobby_club"       # Shared interests (book club, gaming group)
    PROFESSIONAL = "professional"   # Work-related networks
    NEIGHBORHOOD = "neighborhood"   # Geographic proximity groups
    SUPPORT = "support"             # Support groups (parents, caregivers)
    ACTIVIST = "activist"           # Cause-oriented groups
    SOCIAL = "social"               # Pure social gatherings
    CULTURAL = "cultural"           # Shared cultural background
    GENERATIONAL = "generational"   # Age-based groups (young parents, retirees)


@dataclass
class SocialGroup:
    """Represents a social group/clique in the neighborhood."""
    group_id: str
    name: str
    group_type: GroupType
    members: List[str] = field(default_factory=list)
    leader_id: Optional[str] = None  # Most influential member
    formation_day: int = 0
    shared_traits: List[str] = field(default_factory=list)  # What unites them
    cohesion: float = 0.5  # 0.0-1.0 how tight-knit the group is
    influence_score: float = 0.0  # Combined influence in neighborhood
    is_active: bool = True
    last_activity_day: int = 0

    def size(self) -> int:
        return len(self.members)


@dataclass
class Gossip:
    """Represents a piece of gossip spreading through the network."""
    gossip_id: str
    subject_id: str  # Who the gossip is about
    gossip_type: str  # romance, scandal, achievement, secret, rumor
    content: str  # Brief description
    origin_day: int
    originator_id: str  # Who started it
    spread_to: List[str] = field(default_factory=list)  # Who knows
    is_true: bool = True  # True gossip vs rumor
    impact_on_reputation: int = 0  # -20 to +20 reputation impact
    expires_day: int = 0  # When it becomes old news


@dataclass
class SocialInfluence:
    """Tracks a resident's social influence metrics."""
    influence_score: float = 50.0  # 0-100 overall influence
    network_centrality: float = 0.0  # How connected they are
    reputation_trend: int = 0  # Recent reputation changes
    groups_led: int = 0
    groups_member: int = 0
    gossip_spread_power: float = 0.5  # How far their gossip travels
    information_access: float = 0.5  # How much they hear


# Group formation triggers based on shared traits
GROUP_FORMATION_TRIGGERS = {
    "hobby_club": {
        "min_members": 3,
        "trait_match": ["creative", "artistic", "intellectual"],
        "archetype_match": ["artist", "writer", "musician", "craftsperson"],
    },
    "professional": {
        "min_members": 2,
        "trait_match": ["ambitious", "hardworking"],
        "archetype_match": ["tech_worker", "scientist", "entrepreneur", "healthcare"],
    },
    "support": {
        "min_members": 2,
        "trait_match": ["caring", "empathetic"],
        "archetype_match": ["young_parent", "healer", "elder"],
    },
    "activist": {
        "min_members": 3,
        "trait_match": ["passionate", "idealistic"],
        "archetype_match": ["community_organizer", "student"],
    },
    "generational": {
        "min_members": 3,
        "age_range": 15,  # Within 15 years of each other
    },
}

# Gossip event types and their properties
GOSSIP_TYPES = {
    "new_romance": {"spread_rate": 0.4, "duration_days": 14, "reputation_impact": 0},
    "secret_romance_exposed": {"spread_rate": 0.7, "duration_days": 21, "reputation_impact": -10},
    "breakup": {"spread_rate": 0.5, "duration_days": 10, "reputation_impact": -5},
    "achievement": {"spread_rate": 0.3, "duration_days": 7, "reputation_impact": 10},
    "scandal": {"spread_rate": 0.8, "duration_days": 30, "reputation_impact": -20},
    "job_news": {"spread_rate": 0.25, "duration_days": 5, "reputation_impact": 0},
    "health_concern": {"spread_rate": 0.2, "duration_days": 14, "reputation_impact": 5},
    "rumor": {"spread_rate": 0.6, "duration_days": 7, "reputation_impact": -5},
    "good_deed": {"spread_rate": 0.35, "duration_days": 10, "reputation_impact": 15},
}


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

        # v7 additions (November 2025)
        self.major_life_events: Dict[str, List[MajorLifeEvent]] = {}  # agent_id -> events
        self.neighborhood_events: List[NeighborhoodEvent] = []
        self.marriages: List[Tuple[str, str, int]] = []  # (id1, id2, day)
        self.births: List[Tuple[str, int]] = []  # (parent_id, day)
        self.deaths: List[Tuple[str, int]] = []  # (agent_id, day)  - "virtual" deaths (moved away, etc)

        # v7 stats tracking
        self.total_marriages = 0
        self.total_births = 0
        self.total_deaths = 0
        self.total_promotions = 0
        self.total_neighborhood_events = 0
        self.seasonal_moodlets_applied = {season: 0 for season in Season}

        # v8.0 additions (November 2025) - Economic Ecosystem
        self.finances: Dict[str, FinancialState] = {}  # agent_id -> FinancialState
        self.economic_events: List[Tuple[str, str, int, float]] = []  # (agent_id, event_type, day, amount)

        # v8.0 stats tracking
        self.total_bonuses = 0
        self.total_raises = 0
        self.total_job_losses = 0
        self.total_new_jobs = 0
        self.total_unexpected_expenses = 0
        self.total_windfalls = 0  # Inheritances, etc.
        self.total_economic_moodlets = 0
        self.neighborhood_wealth = 0.0  # Total net worth of all residents
        self.income_inequality_gini = 0.0  # Gini coefficient tracking

        # v8.1 additions (November 2025) - Social Networks
        self.social_groups: Dict[str, SocialGroup] = {}  # group_id -> SocialGroup
        self.social_influence: Dict[str, SocialInfluence] = {}  # agent_id -> SocialInfluence
        self.active_gossip: List[Gossip] = []  # Currently spreading gossip
        self.gossip_history: List[Gossip] = []  # All gossip ever created
        self.information_known: Dict[str, List[str]] = {}  # agent_id -> list of gossip_ids they know

        # v8.1 stats tracking
        self.total_groups_formed = 0
        self.total_groups_dissolved = 0
        self.total_gossip_created = 0
        self.total_gossip_spread = 0
        self.total_social_moodlets = 0
        self.average_network_centrality = 0.0
        self.most_influential_resident = ""
        self.most_connected_group = ""
        self.next_group_id = 1  # For unique group IDs

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

    # =========================================================================
    # v8.1: SOCIAL NETWORK HELPER METHODS (November 2025)
    # =========================================================================

    def _try_form_social_group(self, day: int, notable_events: List[str]):
        """Try to form a new social group based on shared traits."""
        # Find potential groups based on archetypes
        archetype_clusters = {}
        for agent_id, archetype in self.archetypes.items():
            archetype_name = archetype.value if archetype else "unknown"
            if archetype_name not in archetype_clusters:
                archetype_clusters[archetype_name] = []
            archetype_clusters[archetype_name].append(agent_id)

        # Check each group formation trigger
        for group_type_name, trigger in GROUP_FORMATION_TRIGGERS.items():
            # Skip if already have too many groups of this type
            existing_of_type = sum(1 for g in self.social_groups.values()
                                   if g.is_active and g.group_type.value == group_type_name)
            if existing_of_type >= 3:  # Max 3 active groups of each type
                continue

            candidates = []

            # Match by archetype
            if "archetype_match" in trigger:
                for archetype_name in trigger["archetype_match"]:
                    candidates.extend(archetype_clusters.get(archetype_name, []))

            # Match by age range (generational groups)
            if "age_range" in trigger:
                age_groups = {}
                for agent_id, resident in self.residents.items():
                    age_bucket = resident.age // trigger["age_range"]
                    if age_bucket not in age_groups:
                        age_groups[age_bucket] = []
                    age_groups[age_bucket].append(agent_id)
                # Pick the largest age group
                if age_groups:
                    largest_bucket = max(age_groups.items(), key=lambda x: len(x[1]))
                    candidates = largest_bucket[1]

            # Filter to non-grouped residents (or allow multiple groups)
            min_members = trigger.get("min_members", 3)

            # Filter out residents who are already in too many groups
            candidates = [c for c in candidates if
                         self.social_influence.get(c) and
                         self.social_influence[c].groups_member < 3]

            # Need at least min_members who also have positive relationships
            if len(candidates) >= min_members:
                # Select members who have positive relationships with each other
                selected = self._select_compatible_group_members(candidates, min_members)

                if len(selected) >= min_members:
                    group_id = f"group_{self.next_group_id}"
                    self.next_group_id += 1

                    # Determine group name based on type
                    group_type = GroupType(group_type_name)
                    group_names = {
                        GroupType.HOBBY_CLUB: ["Book Club", "Art Circle", "Music Jam", "Gaming Group", "Crafters Collective"],
                        GroupType.PROFESSIONAL: ["Networking Group", "Industry Meetup", "Career Circle", "Professionals United"],
                        GroupType.SUPPORT: ["Support Circle", "Parents Group", "Caregivers Network", "Wellness Warriors"],
                        GroupType.ACTIVIST: ["Change Makers", "Community Action", "Neighborhood Watch", "Green Team"],
                        GroupType.GENERATIONAL: ["Young Adults", "Midlife Crew", "Senior Circle", "Golden Years Club"],
                        GroupType.NEIGHBORHOOD: ["Block Party Crew", "Neighbors United", "Local Friends"],
                        GroupType.SOCIAL: ["Social Circle", "Weekend Crew", "Fun Friends"],
                        GroupType.CULTURAL: ["Heritage Circle", "Cultural Connection", "Traditions Group"],
                    }
                    name = random.choice(group_names.get(group_type, ["The Group"]))

                    # Pick a leader (highest influence in the group)
                    leader_id = max(selected, key=lambda x: self.social_influence.get(x, SocialInfluence()).influence_score)

                    group = SocialGroup(
                        group_id=group_id,
                        name=name,
                        group_type=group_type,
                        members=selected,
                        leader_id=leader_id,
                        formation_day=day,
                        shared_traits=trigger.get("trait_match", []),
                        cohesion=0.5,
                        influence_score=sum(self.social_influence.get(m, SocialInfluence()).influence_score
                                           for m in selected) / len(selected),
                        is_active=True,
                        last_activity_day=day,
                    )

                    self.social_groups[group_id] = group
                    self.total_groups_formed += 1

                    # Update member influence
                    for member_id in selected:
                        si = self.social_influence.get(member_id)
                        if si:
                            si.groups_member += 1
                            self.engine.moodlets.add_moodlet(member_id, "joined_group", f"group:{name}")
                            self.total_social_moodlets += 1

                    # Leader gets extra boost
                    si = self.social_influence.get(leader_id)
                    if si:
                        si.groups_led += 1
                        si.influence_score = min(100, si.influence_score + 5)

                    leader_name = self.residents[leader_id].name
                    notable_events.append(f"New group formed: '{name}' led by {leader_name}")
                    break  # Only form one group per day

    def _select_compatible_group_members(self, candidates: List[str], min_members: int) -> List[str]:
        """Select candidates who have positive relationships with each other."""
        if len(candidates) < min_members:
            return []

        # Score each candidate by their relationships with other candidates
        scores = {}
        for c in candidates:
            score = 0
            for other in candidates:
                if c != other:
                    rel = self.relationships.get(tuple(sorted([c, other])))
                    if rel and rel.friendship > 0:
                        score += rel.friendship
            scores[c] = score

        # Sort by score and take the top ones
        sorted_candidates = sorted(candidates, key=lambda x: scores.get(x, 0), reverse=True)

        # Take min_members to min_members + 2 (vary group size)
        max_members = min(len(sorted_candidates), min_members + random.randint(0, 2))
        return sorted_candidates[:max_members]

    def _spread_gossip(self, day: int, notable_events: List[str]):
        """Spread active gossip through the network."""
        for gossip in self.active_gossip:
            gossip_info = GOSSIP_TYPES.get(gossip.gossip_type, {"spread_rate": 0.3})
            spread_rate = gossip_info["spread_rate"]

            # Find people who can spread the gossip
            spreaders = gossip.spread_to[:]

            for spreader_id in spreaders:
                si = self.social_influence.get(spreader_id)
                personal_spread_power = si.gossip_spread_power if si else 0.5

                # Find their social connections
                for (id1, id2), rel in self.relationships.items():
                    if rel.friendship < 10:  # Only spread to people you like
                        continue

                    other_id = id2 if id1 == spreader_id else id1 if id2 == spreader_id else None
                    if not other_id or other_id in gossip.spread_to:
                        continue

                    # Calculate spread chance
                    other_si = self.social_influence.get(other_id)
                    other_access = other_si.information_access if other_si else 0.5

                    spread_chance = spread_rate * personal_spread_power * other_access

                    if random.random() < spread_chance:
                        gossip.spread_to.append(other_id)
                        self.information_known.setdefault(other_id, []).append(gossip.gossip_id)
                        self.total_gossip_spread += 1

                        # Apply moodlets based on gossip type
                        if random.random() < 0.3:  # 30% chance to react
                            if gossip.subject_id == other_id:
                                # They learned gossip about themselves
                                if gossip.impact_on_reputation > 0:
                                    self.engine.moodlets.add_moodlet(other_id, "good_gossip_about_me", "gossip")
                                else:
                                    self.engine.moodlets.add_moodlet(other_id, "bad_gossip_about_me", "gossip")
                            else:
                                self.engine.moodlets.add_moodlet(other_id, "juicy_gossip", "gossip")
                            self.total_social_moodlets += 1

                        # Update reputation based on gossip
                        if gossip.impact_on_reputation != 0:
                            self.reputation[gossip.subject_id] = max(-100, min(100,
                                self.reputation.get(gossip.subject_id, 0) + gossip.impact_on_reputation // 2))

    def _generate_gossip_from_events(self, day: int, notable_events: List[str]):
        """Generate gossip from significant events that happened."""
        # Check for romance-related gossip
        for (id1, id2), rel in self.relationships.items():
            # New romance gossip
            if rel.is_romantic and rel.romance_start_day == day:
                gossip_id = f"gossip_{len(self.gossip_history) + len(self.active_gossip) + 1}"
                gossip_type = "secret_romance_exposed" if rel.is_secret_romance else "new_romance"
                info = GOSSIP_TYPES[gossip_type]

                name1 = self.residents[id1].name
                name2 = self.residents[id2].name

                gossip = Gossip(
                    gossip_id=gossip_id,
                    subject_id=id1,  # Primary subject
                    gossip_type=gossip_type,
                    content=f"{name1} and {name2} are together!",
                    origin_day=day,
                    originator_id=random.choice([id1, id2]),  # One of them leaked it
                    spread_to=[id1, id2],  # They know about themselves
                    is_true=True,
                    impact_on_reputation=info["reputation_impact"],
                    expires_day=day + info["duration_days"],
                )
                self.active_gossip.append(gossip)
                self.total_gossip_created += 1

            # Breakup gossip
            if rel.was_romantic and not rel.is_romantic:
                if random.random() < 0.3:  # 30% chance to generate breakup gossip
                    gossip_id = f"gossip_{len(self.gossip_history) + len(self.active_gossip) + 1}"
                    info = GOSSIP_TYPES["breakup"]

                    name1 = self.residents[id1].name
                    name2 = self.residents[id2].name

                    gossip = Gossip(
                        gossip_id=gossip_id,
                        subject_id=id1,
                        gossip_type="breakup",
                        content=f"{name1} and {name2} broke up!",
                        origin_day=day,
                        originator_id=random.choice(list(self.residents.keys())),  # Random witness
                        spread_to=[],
                        is_true=True,
                        impact_on_reputation=info["reputation_impact"],
                        expires_day=day + info["duration_days"],
                    )
                    self.active_gossip.append(gossip)
                    self.total_gossip_created += 1

        # Good deed gossip (from high-reputation individuals)
        for agent_id, rep in self.reputation.items():
            if rep > 50 and random.random() < 0.01:  # 1% chance for good reputation people
                gossip_id = f"gossip_{len(self.gossip_history) + len(self.active_gossip) + 1}"
                info = GOSSIP_TYPES["good_deed"]

                name = self.residents[agent_id].name

                gossip = Gossip(
                    gossip_id=gossip_id,
                    subject_id=agent_id,
                    gossip_type="good_deed",
                    content=f"{name} did something wonderful for the community!",
                    origin_day=day,
                    originator_id=random.choice(list(self.residents.keys())),
                    spread_to=[],
                    is_true=True,
                    impact_on_reputation=info["reputation_impact"],
                    expires_day=day + info["duration_days"],
                )
                self.active_gossip.append(gossip)
                self.total_gossip_created += 1

    def _update_network_centrality(self):
        """Update network centrality for all residents based on relationships."""
        # Calculate simple degree centrality based on positive relationships
        for agent_id in self.residents:
            positive_connections = 0
            total_relationship_strength = 0

            for (id1, id2), rel in self.relationships.items():
                if agent_id in (id1, id2) and rel.friendship > 0:
                    positive_connections += 1
                    total_relationship_strength += rel.friendship

            # Normalize centrality (0-1 scale)
            max_connections = len(self.residents) - 1
            centrality = positive_connections / max_connections if max_connections > 0 else 0

            # Update social influence
            si = self.social_influence.get(agent_id)
            if si:
                si.network_centrality = centrality

                # Influence score is affected by centrality and group membership
                base_influence = 50.0
                centrality_bonus = centrality * 30  # Up to +30 from centrality
                group_bonus = si.groups_member * 5 + si.groups_led * 10  # Groups add influence
                reputation_bonus = self.reputation.get(agent_id, 0) * 0.2  # Reputation affects influence

                si.influence_score = min(100, max(0, base_influence + centrality_bonus + group_bonus + reputation_bonus))

        # Track averages
        if self.social_influence:
            self.average_network_centrality = sum(
                si.network_centrality for si in self.social_influence.values()
            ) / len(self.social_influence)

            # Find most influential
            most_influential = max(self.social_influence.items(),
                                   key=lambda x: x[1].influence_score)
            self.most_influential_resident = most_influential[0]

        # Find most connected group
        if self.social_groups:
            active_groups = [g for g in self.social_groups.values() if g.is_active]
            if active_groups:
                most_connected = max(active_groups, key=lambda g: len(g.members))
                self.most_connected_group = most_connected.name

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
            # v7: Initialize major life events tracking
            self.major_life_events[resident_id] = []

            # v8.0: Initialize financial state based on archetype and age
            archetype_name = archetype.value if archetype else "unknown"
            income_range = ARCHETYPE_INCOME.get(archetype_name, (35000, 55000))
            base_income = random.uniform(income_range[0], income_range[1])

            # Age-based adjustments
            age = resident.age
            if age < 25:
                # Young - lower income, minimal savings
                income_multiplier = 0.6
                savings_months = random.uniform(0.5, 3)
            elif age < 35:
                # Early career - growing income, moderate savings
                income_multiplier = 0.85
                savings_months = random.uniform(2, 6)
            elif age < 50:
                # Peak earning years
                income_multiplier = 1.0
                savings_months = random.uniform(3, 12)
            elif age < 65:
                # Late career - high income, high savings
                income_multiplier = 1.1
                savings_months = random.uniform(6, 24)
            else:
                # Retired - fixed income, variable savings
                income_multiplier = 0.5
                savings_months = random.uniform(12, 60)

            annual_income = base_income * income_multiplier
            monthly_income = annual_income / 12
            monthly_expenses = monthly_income * random.uniform(0.6, 0.85)
            savings = monthly_expenses * savings_months

            # Some people have debt
            debt = 0.0
            if random.random() < 0.4:  # 40% have some debt
                if age < 30:
                    debt = random.uniform(5000, 50000)  # Student loans, etc.
                elif age < 50:
                    debt = random.uniform(0, 30000)  # Car loans, credit cards
                else:
                    debt = random.uniform(0, 10000)  # Minimal debt

            # Job stability based on archetype
            if archetype_name in ["tech_worker", "scientist", "healthcare", "teacher"]:
                job_stability = random.uniform(0.7, 0.95)
            elif archetype_name in ["artist", "musician", "writer", "entrepreneur"]:
                job_stability = random.uniform(0.3, 0.7)
            elif archetype_name in ["retiree", "student"]:
                job_stability = 1.0  # N/A but stable
            else:
                job_stability = random.uniform(0.5, 0.85)

            is_employed = archetype_name not in ["retiree", "student"] or random.random() < 0.3
            if archetype_name == "student":
                is_employed = random.random() < 0.5  # Part-time work

            self.finances[resident_id] = FinancialState(
                income=annual_income,
                savings=savings,
                debt=debt,
                monthly_expenses=monthly_expenses,
                financial_stress=0.0,  # Will be calculated
                last_paycheck_day=random.randint(1, 14),
                job_stability=job_stability,
                is_employed=is_employed,
                unemployment_day=0,
            )
            # Calculate initial stress
            self.finances[resident_id].financial_stress = self.finances[resident_id].calculate_stress()

            # v8.1: Initialize social influence based on archetype and personality
            base_influence = 50.0
            gossip_power = 0.5
            info_access = 0.5

            # Social butterflies and community organizers have higher influence
            if archetype == Archetype.SOCIAL_BUTTERFLY:
                base_influence = random.uniform(65, 85)
                gossip_power = random.uniform(0.7, 0.9)
                info_access = random.uniform(0.7, 0.9)
            elif archetype == Archetype.COMMUNITY_ORGANIZER:
                base_influence = random.uniform(60, 80)
                gossip_power = random.uniform(0.6, 0.8)
                info_access = random.uniform(0.7, 0.85)
            elif archetype == Archetype.INTROVERT_SAGE:
                base_influence = random.uniform(40, 60)
                gossip_power = random.uniform(0.2, 0.4)
                info_access = random.uniform(0.3, 0.5)
            elif archetype in [Archetype.ELDER, Archetype.RETIREE]:
                base_influence = random.uniform(45, 70)
                gossip_power = random.uniform(0.5, 0.7)
                info_access = random.uniform(0.6, 0.8)
            elif archetype == Archetype.NEWCOMER:
                base_influence = random.uniform(30, 50)
                gossip_power = random.uniform(0.3, 0.5)
                info_access = random.uniform(0.2, 0.4)

            # Personality modifiers
            personality = resident.personality
            if hasattr(personality, 'extraversion') and personality.extraversion > 0.7:
                base_influence += 10
                gossip_power += 0.1
                info_access += 0.1
            elif hasattr(personality, 'extraversion') and personality.extraversion < 0.3:
                base_influence -= 10
                gossip_power -= 0.1
                info_access -= 0.1

            self.social_influence[resident_id] = SocialInfluence(
                influence_score=min(100, max(0, base_influence)),
                network_centrality=0.0,  # Will be calculated based on relationships
                reputation_trend=0,
                groups_led=0,
                groups_member=0,
                gossip_spread_power=min(1.0, max(0.1, gossip_power)),
                information_access=min(1.0, max(0.1, info_access)),
            )
            self.information_known[resident_id] = []

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

        # === v7: SEASONAL MOOD EFFECTS ===
        current_season = get_season(day)
        season_config = SEASONAL_MOODLETS[current_season]

        for agent_id in resident_ids:
            resident = self.residents[agent_id]
            personality = resident.personality

            # Base seasonal effect chances (modified by personality)
            positive_chance = season_config["positive_weight"]
            negative_chance = season_config["negative_weight"]

            # Personality modifiers for seasonal effects
            # Introverts handle winter better (cozy indoors), worse in summer (outdoor pressure)
            # Extroverts love summer, struggle with winter isolation
            if current_season == Season.WINTER:
                if personality.openness < 0.4:  # Introverts
                    positive_chance += 0.1  # More likely to enjoy cozy winter
                    negative_chance -= 0.05
                elif personality.extraversion > 0.6:  # Extroverts
                    negative_chance += 0.1  # More cabin fever
                    positive_chance -= 0.05
            elif current_season == Season.SUMMER:
                if personality.extraversion > 0.6:  # Extroverts
                    positive_chance += 0.1  # Love the social summer
                    negative_chance -= 0.05
                elif personality.openness < 0.4:  # Introverts
                    negative_chance += 0.05  # Mild heat exhaustion from obligations
            elif current_season == Season.AUTUMN:
                # Melancholic personalities feel autumn more deeply
                if personality.neuroticism > 0.6:
                    negative_chance += 0.1  # More melancholy
                else:
                    positive_chance += 0.05  # Cozy sweater vibes
            elif current_season == Season.SPRING:
                # Most people feel positive in spring
                positive_chance += 0.05
                # But allergies are random
                if random.random() < 0.1:  # 10% chance of allergies
                    self.engine.moodlets.add_moodlet(agent_id, "spring_allergies", "season")
                    self.seasonal_moodlets_applied[current_season] += 1

            # Apply positive seasonal moodlet
            if random.random() < positive_chance:
                moodlet = random.choice(season_config["positive"])
                self.engine.moodlets.add_moodlet(agent_id, moodlet, "season")
                self.seasonal_moodlets_applied[current_season] += 1

            # Apply negative seasonal moodlet (separate roll)
            if random.random() < negative_chance:
                moodlet = random.choice(season_config["negative"])
                self.engine.moodlets.add_moodlet(agent_id, moodlet, "season")
                self.seasonal_moodlets_applied[current_season] += 1

        # === v8.0: ECONOMIC EVENTS ===
        day_of_month = (day % 30) + 1  # Simulate monthly cycle

        for agent_id in resident_ids:
            finance = self.finances.get(agent_id)
            if not finance:
                continue

            resident = self.residents[agent_id]
            personality = resident.personality
            archetype = self.archetypes.get(agent_id)
            archetype_name = archetype.value if archetype else "unknown"

            # --- BI-WEEKLY PAYCHECK (every 14 days from their start day) ---
            if finance.is_employed and (day - finance.last_paycheck_day) % 14 == 0 and day > 0:
                paycheck = finance.income / 26  # 26 bi-weekly paychecks per year
                finance.savings += paycheck
                # Conscientious people save more
                if personality.conscientiousness > 0.7:
                    bonus_save = paycheck * 0.1
                    finance.savings += bonus_save
                # Random satisfaction from payday
                if random.random() < 0.3:
                    self.engine.moodlets.add_moodlet(agent_id, "financial_security", "payday")
                    self.total_economic_moodlets += 1

            # --- MONTHLY EXPENSES (day 1 of each month) ---
            if day_of_month == 1:
                finance.savings -= finance.monthly_expenses
                # Pay down debt if possible
                if finance.debt > 0 and finance.savings > finance.monthly_expenses * 2:
                    debt_payment = min(finance.debt, finance.monthly_expenses * 0.2)
                    finance.debt -= debt_payment
                    finance.savings -= debt_payment
                    if finance.debt == 0:
                        self.engine.moodlets.add_moodlet(agent_id, "debt_paid_off", "financial")
                        notable_events.append(f"{resident.name} paid off all their debt!")
                        self.total_economic_moodlets += 1

            # --- RANDOM ECONOMIC EVENTS ---
            # Bonus (employed only)
            if finance.is_employed and random.random() < ECONOMIC_EVENTS["bonus"]["chance"]:
                bonus_range = ECONOMIC_EVENTS["bonus"]["amount_range"]
                bonus = random.uniform(bonus_range[0], bonus_range[1])
                # High performers get bigger bonuses
                if personality.conscientiousness > 0.7:
                    bonus *= 1.3
                finance.savings += bonus
                self.economic_events.append((agent_id, "bonus", day, bonus))
                self.engine.moodlets.add_moodlet(agent_id, "bonus_received", "work")
                notable_events.append(f"{resident.name} received a ${bonus:.0f} bonus!")
                self.total_bonuses += 1
                self.total_economic_moodlets += 1

            # Raise (employed only)
            if finance.is_employed and random.random() < ECONOMIC_EVENTS["raise"]["chance"]:
                raise_range = ECONOMIC_EVENTS["raise"]["percentage"]
                raise_pct = random.uniform(raise_range[0], raise_range[1])
                old_income = finance.income
                finance.income *= (1 + raise_pct)
                self.economic_events.append((agent_id, "raise", day, finance.income - old_income))
                self.engine.moodlets.add_moodlet(agent_id, "pay_raise", "work")
                notable_events.append(f"{resident.name} got a {raise_pct*100:.1f}% raise!")
                self.total_raises += 1
                self.total_economic_moodlets += 1

            # Unexpected expense
            if random.random() < ECONOMIC_EVENTS["unexpected_expense"]["chance"]:
                expense_range = ECONOMIC_EVENTS["unexpected_expense"]["amount_range"]
                expense = random.uniform(expense_range[0], expense_range[1])
                finance.savings -= expense
                if finance.savings < 0:
                    finance.debt -= finance.savings  # Convert negative to debt
                    finance.savings = 0
                self.economic_events.append((agent_id, "unexpected_expense", day, expense))
                self.engine.moodlets.add_moodlet(agent_id, "unexpected_bill", "financial")
                self.total_unexpected_expenses += 1
                self.total_economic_moodlets += 1

            # Medical bill
            if random.random() < ECONOMIC_EVENTS["medical_bill"]["chance"]:
                bill_range = ECONOMIC_EVENTS["medical_bill"]["amount_range"]
                bill = random.uniform(bill_range[0], bill_range[1])
                finance.savings -= bill
                if finance.savings < 0:
                    finance.debt -= finance.savings
                    finance.savings = 0
                self.economic_events.append((agent_id, "medical_bill", day, bill))
                self.engine.moodlets.add_moodlet(agent_id, "medical_debt", "health")
                notable_events.append(f"{resident.name} received a ${bill:.0f} medical bill")
                self.total_economic_moodlets += 1

            # Car repair
            if random.random() < ECONOMIC_EVENTS["car_repair"]["chance"]:
                repair_range = ECONOMIC_EVENTS["car_repair"]["amount_range"]
                repair = random.uniform(repair_range[0], repair_range[1])
                finance.savings -= repair
                if finance.savings < 0:
                    finance.debt -= finance.savings
                    finance.savings = 0
                self.economic_events.append((agent_id, "car_repair", day, repair))
                self.engine.moodlets.add_moodlet(agent_id, "car_troubles", "maintenance")
                self.total_economic_moodlets += 1

            # Inheritance (rare windfall)
            if random.random() < ECONOMIC_EVENTS["inheritance"]["chance"]:
                inherit_range = ECONOMIC_EVENTS["inheritance"]["amount_range"]
                inheritance = random.uniform(inherit_range[0], inherit_range[1])
                finance.savings += inheritance
                self.economic_events.append((agent_id, "inheritance", day, inheritance))
                self.engine.moodlets.add_moodlet(agent_id, "inheritance_received", "family")
                notable_events.append(f"{resident.name} received a ${inheritance:.0f} inheritance")
                self.total_windfalls += 1
                self.total_economic_moodlets += 1

            # Job loss (employed only, affected by job stability)
            if finance.is_employed:
                job_loss_chance = ECONOMIC_EVENTS["job_loss"]["chance"]
                # Low stability increases risk
                job_loss_chance *= (1.5 - finance.job_stability)
                # Neurotic personalities more vulnerable (perceived instability)
                if personality.neuroticism > 0.7:
                    job_loss_chance *= 1.2
                if random.random() < job_loss_chance:
                    finance.is_employed = False
                    finance.unemployment_day = day
                    self.economic_events.append((agent_id, "job_loss", day, 0))
                    self.engine.moodlets.add_moodlet(agent_id, "unemployed", "work")
                    notable_events.append(f"{resident.name} lost their job!")
                    self.total_job_losses += 1
                    self.total_economic_moodlets += 1

            # Finding new job (unemployed only, takes time)
            if not finance.is_employed and finance.unemployment_day > 0:
                days_unemployed = day - finance.unemployment_day
                # Base chance increases over time (desperation / more applications)
                find_job_chance = ECONOMIC_EVENTS["new_job"]["chance"]
                find_job_chance += days_unemployed * 0.002  # +0.2% per day
                # Extroverts network better
                if personality.extraversion > 0.6:
                    find_job_chance *= 1.3
                # Conscientious apply more
                if personality.conscientiousness > 0.6:
                    find_job_chance *= 1.2
                if random.random() < find_job_chance:
                    finance.is_employed = True
                    finance.unemployment_day = 0
                    # New job might pay differently
                    income_change = random.uniform(0.8, 1.1)  # Could be -20% to +10%
                    finance.income *= income_change
                    finance.job_stability = random.uniform(0.5, 0.8)  # Start fresh
                    self.economic_events.append((agent_id, "new_job", day, finance.income))
                    self.engine.moodlets.add_moodlet(agent_id, "found_new_job", "work")
                    notable_events.append(f"{resident.name} found a new job!")
                    self.total_new_jobs += 1
                    self.total_economic_moodlets += 1

            # --- UPDATE FINANCIAL STRESS ---
            old_stress = finance.financial_stress
            finance.financial_stress = finance.calculate_stress()

            # --- STRESS-BASED MOODLETS ---
            if finance.financial_stress > 0.7:
                if random.random() < 0.3:  # 30% chance when high stress
                    self.engine.moodlets.add_moodlet(agent_id, "financial_stress", "money")
                    self.total_economic_moodlets += 1
            elif finance.financial_stress > 0.4:
                if random.random() < 0.1:  # 10% chance for moderate stress
                    self.engine.moodlets.add_moodlet(agent_id, "money_worries", "money")
                    self.total_economic_moodlets += 1

            # Broke check
            if finance.savings <= 0 and finance.debt > finance.income * 0.5:
                if random.random() < 0.2:  # 20% chance when broke
                    self.engine.moodlets.add_moodlet(agent_id, "broke", "financial")
                    self.total_economic_moodlets += 1

            # Financial security feeling (low stress, good savings)
            if finance.financial_stress < 0.2 and finance.savings > finance.monthly_expenses * 6:
                if random.random() < 0.05:  # 5% chance
                    self.engine.moodlets.add_moodlet(agent_id, "financial_security", "savings")
                    self.total_economic_moodlets += 1

        # === v8.1: SOCIAL NETWORKS ===
        # --- GROUP FORMATION ---
        # Check for potential new group formation (5% chance per day)
        if random.random() < 0.05:
            self._try_form_social_group(day, notable_events)

        # --- GROUP ACTIVITIES ---
        # Active groups have a chance to do activities
        for group_id, group in list(self.social_groups.items()):
            if not group.is_active:
                continue

            # 20% chance of group activity each day
            if random.random() < 0.20:
                group.last_activity_day = day
                group.cohesion = min(1.0, group.cohesion + 0.05)  # Activities build cohesion

                for member_id in group.members:
                    if random.random() < 0.5:  # 50% chance each member feels the bond
                        self.engine.moodlets.add_moodlet(member_id, "group_activity", f"group:{group.name}")
                        self.total_social_moodlets += 1

                # Update influence for leader
                if group.leader_id and random.random() < 0.3:
                    self.engine.moodlets.add_moodlet(group.leader_id, "group_leader", f"group:{group.name}")
                    self.total_social_moodlets += 1

            # Group dissolution check (inactive or low cohesion)
            days_inactive = day - group.last_activity_day
            if days_inactive > 30 or group.cohesion < 0.2:
                if random.random() < 0.1:  # 10% chance
                    group.is_active = False
                    for member_id in group.members:
                        if random.random() < 0.4:
                            self.engine.moodlets.add_moodlet(member_id, "group_dissolved", f"group:{group.name}")
                            self.total_social_moodlets += 1
                        si = self.social_influence.get(member_id)
                        if si:
                            si.groups_member = max(0, si.groups_member - 1)
                    self.total_groups_dissolved += 1
                    notable_events.append(f"The {group.name} group disbanded!")

        # --- GOSSIP SPREAD ---
        self._spread_gossip(day, notable_events)

        # --- GOSSIP GENERATION ---
        # New gossip can be created from dramatic events
        self._generate_gossip_from_events(day, notable_events)

        # --- EXPIRE OLD GOSSIP ---
        for gossip in self.active_gossip[:]:
            if day >= gossip.expires_day:
                self.active_gossip.remove(gossip)
                self.gossip_history.append(gossip)

        # --- UPDATE NETWORK CENTRALITY ---
        self._update_network_centrality()

        # --- INFLUENCE-BASED MOODLETS ---
        for agent_id in resident_ids:
            si = self.social_influence.get(agent_id)
            if not si:
                continue

            # High influence moodlets
            if si.influence_score >= 80 and random.random() < 0.05:
                self.engine.moodlets.add_moodlet(agent_id, "opinion_leader", "influence")
                self.total_social_moodlets += 1
            elif si.influence_score >= 65 and random.random() < 0.03:
                self.engine.moodlets.add_moodlet(agent_id, "social_influence", "influence")
                self.total_social_moodlets += 1

            # Low influence moodlets
            if si.influence_score < 30 and random.random() < 0.03:
                self.engine.moodlets.add_moodlet(agent_id, "lost_influence", "social")
                self.total_social_moodlets += 1

            # Social butterfly effect for highly connected
            if si.network_centrality > 0.7 and random.random() < 0.05:
                archetype = self.archetypes.get(agent_id)
                if archetype == Archetype.SOCIAL_BUTTERFLY:
                    self.engine.moodlets.add_moodlet(agent_id, "social_butterfly_high", "personality")
                    self.total_social_moodlets += 1
                elif archetype == Archetype.INTROVERT_SAGE:
                    # Introverts can get overwhelmed
                    if random.random() < 0.3:
                        self.engine.moodlets.add_moodlet(agent_id, "social_exhaustion", "personality")
                        self.total_social_moodlets += 1

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
        # v7: Enhanced with NeighborhoodEvent tracking and seasonal influence
        current_season = get_season(day)

        # Season affects event types
        if current_season == Season.SUMMER:
            event_chance = 0.20  # More outdoor events in summer
        elif current_season == Season.WINTER:
            event_chance = 0.10  # Fewer outdoor events in winter
        else:
            event_chance = 0.15

        if random.random() < event_chance:
            # v7: Event pool varies by season
            if current_season == Season.SUMMER:
                event_pool = [
                    "block_party", "street_fair", "farmers_market",
                    "community_cleanup", "celebration", "gossip_spreads",
                    "neighborhood_drama", "emergency_response"
                ]
            elif current_season == Season.WINTER:
                event_pool = [
                    "neighborhood_meeting", "gossip_spreads", "power_outage",
                    "celebration", "emergency_response", "neighborhood_drama"
                ]
            else:
                event_pool = [
                    "block_party", "farmers_market", "power_outage", "street_fair",
                    "neighborhood_meeting", "community_cleanup", "gossip_spreads",
                    "neighborhood_drama", "celebration", "emergency_response"
                ]

            event = random.choice(event_pool)
            self.total_neighborhood_events += 1

            if event == "block_party":
                self.community_events.append(f"Day {day}: Block party! Everyone had a great time.")
                notable_events.append("[NEIGHBORHOOD] The neighborhood had a block party!")

                # v7: Track as NeighborhoodEvent
                affected = random.sample(resident_ids, min(20, len(resident_ids)))
                neighborhood_event = NeighborhoodEvent(
                    event_type="block_party",
                    day=day,
                    affected_residents=affected,
                    details=f"Block party in {get_season_name(day)}"
                )
                self.neighborhood_events.append(neighborhood_event)

                # v7: Use block_party_fun moodlet
                for aid in affected:
                    self.engine.moodlets.add_moodlet(aid, "block_party_fun", "party")
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
                notable_events.append("[NEIGHBORHOOD] A power outage hit the neighborhood!")

                # v7: Track as NeighborhoodEvent
                neighborhood_event = NeighborhoodEvent(
                    event_type="power_outage",
                    day=day,
                    affected_residents=resident_ids,
                    details="Power outage affecting entire neighborhood"
                )
                self.neighborhood_events.append(neighborhood_event)

                # v7: Use power_outage moodlet, some bond over it
                for aid in resident_ids:
                    if random.random() < 0.3:
                        self.engine.moodlets.add_moodlet(aid, "power_outage", "crisis")
                    elif random.random() < 0.2:
                        self.engine.moodlets.add_moodlet(aid, "great_conversation", "outage")

            elif event == "street_fair":
                self.community_events.append(f"Day {day}: Annual street fair!")
                notable_events.append("[NEIGHBORHOOD] The annual street fair was a huge success!")

                # v7: Track as NeighborhoodEvent
                affected = random.sample(resident_ids, min(30, len(resident_ids)))
                neighborhood_event = NeighborhoodEvent(
                    event_type="festival",
                    day=day,
                    affected_residents=affected,
                    details=f"Street fair - {get_season_name(day)}"
                )
                self.neighborhood_events.append(neighborhood_event)

                # v7: Use festival_excitement moodlet
                for aid in affected:
                    self.engine.moodlets.add_moodlet(aid, "festival_excitement", "fair")

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
                notable_events.append("[NEIGHBORHOOD] Neighbors came together for community cleanup!")

                # v7: Track as NeighborhoodEvent
                affected = random.sample(resident_ids, min(20, len(resident_ids)))
                neighborhood_event = NeighborhoodEvent(
                    event_type="community_cleanup",
                    day=day,
                    affected_residents=affected,
                    details="Community cleanup day"
                )
                self.neighborhood_events.append(neighborhood_event)

                # v7: Use community_achievement moodlet
                for aid in affected:
                    self.engine.moodlets.add_moodlet(aid, "community_achievement", "cleanup")

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

        # === v7: MAJOR LIFE EVENTS ===
        # Marriages: Long-term couples (90+ days together) may get married
        for (id1, id2), rel in list(self.relationships.items()):
            if rel.is_romantic and not rel.was_romantic:
                romance_duration = day - rel.romance_start_day
                # Marriage chance for long-term couples
                if romance_duration >= 90 and rel.trust >= 60 and rel.friendship >= 50:
                    # Check if neither is already married
                    already_married_1 = any(m[0] == id1 or m[1] == id1 for m in self.marriages)
                    already_married_2 = any(m[0] == id2 or m[1] == id2 for m in self.marriages)
                    if not already_married_1 and not already_married_2:
                        # 0.5% daily chance of marriage proposal
                        if random.random() < 0.005:
                            self.marriages.append((id1, id2, day))
                            self.total_marriages += 1
                            r1 = self.residents[id1]
                            r2 = self.residents[id2]
                            notable_events.append(f"[WEDDING] {r1.name} and {r2.name} got MARRIED!")

                            # Record major life event
                            event = MajorLifeEvent(
                                event_type="marriage",
                                day=day,
                                participants=[id1, id2],
                                details=f"{r1.name} married {r2.name}"
                            )
                            self.major_life_events[id1].append(event)
                            self.major_life_events[id2].append(event)

                            # Apply moodlets
                            self.engine.moodlets.add_moodlet(id1, "just_married", "wedding")
                            self.engine.moodlets.add_moodlet(id2, "just_married", "wedding")

                            # Boost trust to max
                            rel.trust = 100

        # Births: Married couples may have children
        for (spouse1, spouse2, wedding_day) in self.marriages:
            marriage_duration = day - wedding_day
            # Only after 30 days of marriage, with some probability
            if marriage_duration >= 30:
                r1 = self.residents.get(spouse1)
                r2 = self.residents.get(spouse2)
                if r1 and r2:
                    # Check ages (18-45 for childbearing)
                    if min(r1.age, r2.age) >= 18 and max(r1.age, r2.age) <= 45:
                        # 0.3% daily chance of birth announcement
                        if random.random() < 0.003:
                            self.births.append((spouse1, day))
                            self.total_births += 1
                            notable_events.append(f"[BIRTH] {r1.name} and {r2.name} had a baby!")

                            # Record major life event
                            event = MajorLifeEvent(
                                event_type="birth",
                                day=day,
                                participants=[spouse1, spouse2],
                                details=f"Baby born to {r1.name} and {r2.name}"
                            )
                            self.major_life_events[spouse1].append(event)
                            self.major_life_events[spouse2].append(event)

                            # Apply moodlets
                            self.engine.moodlets.add_moodlet(spouse1, "new_parent", "birth")
                            self.engine.moodlets.add_moodlet(spouse2, "new_parent", "birth")

        # Deaths: Rare, more likely for elderly (70+)
        for agent_id in resident_ids:
            resident = self.residents[agent_id]
            base_death_chance = 0.0001  # 0.01% base daily chance

            # Age modifier
            if resident.age >= 80:
                death_chance = base_death_chance * 5  # 0.05%
            elif resident.age >= 70:
                death_chance = base_death_chance * 2  # 0.02%
            else:
                death_chance = base_death_chance

            # Ongoing severe crises increase risk slightly
            severe_crises = [c for c in self.life_crises.get(agent_id, [])
                           if not c.resolved and c.severity >= 0.7]
            if severe_crises:
                death_chance *= 1.5

            if random.random() < death_chance:
                self.deaths.append((agent_id, day))
                self.total_deaths += 1
                notable_events.append(f"[DEATH] The neighborhood mourns the loss of {resident.name}...")

                # Record major life event
                event = MajorLifeEvent(
                    event_type="death",
                    day=day,
                    participants=[agent_id],
                    details=f"{resident.name} passed away at age {resident.age}"
                )
                self.major_life_events[agent_id].append(event)

                # Apply moodlets to close friends/family
                for (id1, id2), rel in self.relationships.items():
                    if agent_id in (id1, id2):
                        other_id = id2 if id1 == agent_id else id1
                        if rel.friendship >= 40 or rel.is_romantic:
                            self.engine.moodlets.add_moodlet(other_id, "death_in_family", "loss")

        # Promotions: Random career advancement (0.5% daily for working-age adults)
        for agent_id in resident_ids:
            resident = self.residents[agent_id]
            # Working age (25-65) and not in student/retiree archetypes
            if 25 <= resident.age <= 65:
                archetype = self.archetypes.get(agent_id)
                if archetype not in [Archetype.STUDENT, Archetype.RETIREE]:
                    if random.random() < 0.005:  # 0.5% daily
                        self.total_promotions += 1
                        notable_events.append(f"[PROMOTION] {resident.name} got a big promotion!")

                        # Record major life event
                        event = MajorLifeEvent(
                            event_type="promotion",
                            day=day,
                            participants=[agent_id],
                            details=f"{resident.name} was promoted"
                        )
                        self.major_life_events[agent_id].append(event)

                        # Apply moodlets
                        self.engine.moodlets.add_moodlet(agent_id, "got_promoted", "career")

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

        # v7 Life Events
        print("\n[v7 LIFE EVENTS]")
        print(f"  - Marriages: {self.total_marriages}")
        print(f"  - Births: {self.total_births}")
        print(f"  - Deaths: {self.total_deaths}")
        print(f"  - Promotions: {self.total_promotions}")
        print(f"  - Neighborhood events: {self.total_neighborhood_events}")

        # v7 Seasonal Moodlets
        print("\n[v7 SEASONAL MOODLETS]")
        for season in Season:
            count = self.seasonal_moodlets_applied.get(season, 0)
            print(f"  - {season.value.title()}: {count}")

        # v8.0 Economic Stats
        print("\n[v8.0 ECONOMIC STATS]")
        print(f"  - Bonuses awarded: {self.total_bonuses}")
        print(f"  - Raises given: {self.total_raises}")
        print(f"  - Job losses: {self.total_job_losses}")
        print(f"  - New jobs found: {self.total_new_jobs}")
        print(f"  - Unexpected expenses: {self.total_unexpected_expenses}")
        print(f"  - Windfalls (inheritance, etc.): {self.total_windfalls}")
        print(f"  - Economic moodlets applied: {self.total_economic_moodlets}")

        # Calculate neighborhood wealth statistics
        if self.finances:
            total_savings = sum(f.savings for f in self.finances.values())
            total_debt = sum(f.debt for f in self.finances.values())
            total_net_worth = sum(f.net_worth() for f in self.finances.values())
            avg_income = sum(f.income for f in self.finances.values()) / len(self.finances)
            employed_count = sum(1 for f in self.finances.values() if f.is_employed)
            unemployed_count = len(self.finances) - employed_count
            avg_stress = sum(f.financial_stress for f in self.finances.values()) / len(self.finances)

            print(f"\n[v8.0 NEIGHBORHOOD WEALTH]")
            print(f"  - Total savings: ${total_savings:,.0f}")
            print(f"  - Total debt: ${total_debt:,.0f}")
            print(f"  - Net worth: ${total_net_worth:,.0f}")
            print(f"  - Average income: ${avg_income:,.0f}/year")
            print(f"  - Employed: {employed_count} | Unemployed: {unemployed_count}")
            print(f"  - Average financial stress: {avg_stress*100:.1f}%")

            # Calculate income inequality (simple Gini coefficient)
            incomes = sorted([f.income for f in self.finances.values()])
            n = len(incomes)
            if n > 1:
                total = sum(incomes)
                b = sum((n - i) * income for i, income in enumerate(incomes))
                gini = 1 - (2 * b) / (n * total)
                self.income_inequality_gini = gini
                print(f"  - Income inequality (Gini): {gini:.3f}")
                if gini < 0.3:
                    print(f"    -> Low inequality (egalitarian neighborhood)")
                elif gini < 0.4:
                    print(f"    -> Moderate inequality")
                else:
                    print(f"    -> High inequality (economic diversity)")

            # Wealthiest and poorest residents
            by_net_worth = sorted(self.finances.items(), key=lambda x: -x[1].net_worth())
            print(f"\n  Wealthiest residents:")
            for agent_id, finance in by_net_worth[:3]:
                resident = self.residents[agent_id]
                print(f"    {resident.name}: ${finance.net_worth():,.0f} net worth")

            poorest = by_net_worth[-3:][::-1]
            print(f"  Financial struggles:")
            for agent_id, finance in poorest:
                if finance.net_worth() < 5000:  # Only show if actually struggling
                    resident = self.residents[agent_id]
                    print(f"    {resident.name}: ${finance.net_worth():,.0f} net worth")

        # v8.1 Social Networks stats (November 2025)
        if self.social_groups or self.gossip_history:
            print(f"\n[v8.1 SOCIAL NETWORKS]")

            # Group statistics
            active_groups = [g for g in self.social_groups.values() if g.is_active]
            dissolved_groups = [g for g in self.social_groups.values() if not g.is_active]

            print(f"  - Groups formed: {self.total_groups_formed}")
            print(f"  - Groups dissolved: {self.total_groups_dissolved}")
            print(f"  - Currently active groups: {len(active_groups)}")

            if active_groups:
                # Most connected group
                most_connected = max(active_groups, key=lambda g: len(g.members))
                print(f"  - Largest group: {most_connected.name} ({len(most_connected.members)} members)")

                # Group types breakdown
                group_types = {}
                for g in active_groups:
                    group_types[g.group_type.value] = group_types.get(g.group_type.value, 0) + 1
                print(f"  - Group types: {', '.join(f'{k}: {v}' for k, v in group_types.items())}")

            # Gossip statistics
            print(f"\n  [Gossip Network]")
            print(f"  - Total gossip created: {self.total_gossip_created}")
            print(f"  - Total gossip spread events: {self.total_gossip_spread}")
            print(f"  - Active rumors: {len(self.active_gossip)}")

            if self.gossip_history:
                # Most talked about person
                subject_counts = {}
                for g in self.gossip_history:
                    subject_counts[g.subject_id] = subject_counts.get(g.subject_id, 0) + 1
                if subject_counts:
                    most_talked = max(subject_counts.items(), key=lambda x: x[1])
                    resident = self.residents.get(most_talked[0])
                    if resident:
                        print(f"  - Most talked about: {resident.name} ({most_talked[1]} rumors)")

            # Influence statistics
            print(f"\n  [Social Influence]")
            print(f"  - Social moodlets applied: {self.total_social_moodlets}")
            print(f"  - Average network centrality: {self.average_network_centrality:.2f}")

            if self.social_influence:
                # Most influential resident
                by_influence = sorted(self.social_influence.items(), key=lambda x: -x[1].influence_score)
                top_influencer = by_influence[0]
                resident = self.residents.get(top_influencer[0])
                if resident:
                    inf = top_influencer[1]
                    print(f"  - Most influential: {resident.name}")
                    print(f"    -> Influence score: {inf.influence_score:.1f}")
                    print(f"    -> Network centrality: {inf.network_centrality:.2f}")
                    print(f"    -> Groups led: {inf.groups_led} | Member of: {inf.groups_member}")

                # Least influential (if significantly low)
                least_inf = by_influence[-1]
                if least_inf[1].influence_score < 30:
                    resident = self.residents.get(least_inf[0])
                    if resident:
                        print(f"  - Least connected: {resident.name} (influence: {least_inf[1].influence_score:.1f})")

            # Active groups details
            if active_groups:
                print(f"\n  [Active Social Groups]")
                for group in sorted(active_groups, key=lambda g: -len(g.members))[:5]:
                    leader_name = self.residents[group.leader_id].name if group.leader_id else "No leader"
                    print(f"  - {group.name} ({group.group_type.value})")
                    print(f"    Leader: {leader_name} | Members: {len(group.members)} | Cohesion: {group.cohesion:.0%}")

        # v7 Major Life Events by type
        all_life_events = []
        for events in self.major_life_events.values():
            all_life_events.extend(events)
        if all_life_events:
            print("\n[v7 NOTABLE LIFE EVENTS]")
            # Show most recent 5 events
            recent_events = sorted(all_life_events, key=lambda e: e.day, reverse=True)[:5]
            for event in recent_events:
                print(f"  Day {event.day}: [{event.event_type.upper()}] {event.details}")

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
    """Run a 180-day simulation with v8.1 social network features."""
    print("\n" + "#" * 70)
    print("#  NEIGHBORHOOD SIMULATION v8.1 - SOCIAL NETWORKS")
    print("#  50 Residents | 180 Days | Full Social Dynamics")
    print("#" * 70)
    print("#  FEATURES INCLUDED:")
    print("#  v5: Love triangles, betrayals, life crises, breakups")
    print("#  v6: Romance tuning, relationship balance")
    print("#  v7: Seasonal moods, major life events, neighborhood events")
    print("#  v8.0: Economic ecosystem (income, savings, debt, jobs)")
    print("#  v8.1: SOCIAL NETWORKS (NEW!)")
    print("#      • Social groups/cliques (8 types: hobby, professional, etc.)")
    print("#      • Group formation based on shared traits & archetypes")
    print("#      • Group dynamics (activities, cohesion, dissolution)")
    print("#      • Gossip network (rumors spread through relationships)")
    print("#      • Social influence & network centrality tracking")
    print("#      • Information access based on social position")
    print("#      • 21 new social moodlets (group, gossip, influence)")
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
