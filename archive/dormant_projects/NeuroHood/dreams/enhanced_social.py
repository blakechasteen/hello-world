"""
v8.4 Enhanced Social System - Extensible Personality-Behavior System

Creates emergent social patterns from trait combinations, enabling nuanced
characters like "the introvert who is an amazing best friend."

Author: Claude
Date: November 2025
Version: 8.4
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import random


# =============================================================================
# ENUMS
# =============================================================================

class SpecialAbility(Enum):
    """Special social abilities archetypes can have."""
    DEEP_LISTENER = "deep_listener"       # +50% relationship gain from talks
    SECRET_KEEPER = "secret_keeper"       # Never spreads gossip
    CONNECTOR = "connector"               # Can introduce people, creates bonds
    MENTOR = "mentor"                     # Younger agents learn skills faster
    CALM_PRESENCE = "calm_presence"       # Reduces anxiety in nearby agents
    PARTY_STARTER = "party_starter"       # +30% chance to trigger group events
    CONFLICT_RESOLVER = "conflict_resolver"  # Can heal relationship damage
    SOCIAL_CATALYST = "social_catalyst"   # Makes others more social around them
    WISE_COUNSEL = "wise_counsel"         # Sought out for life advice
    ENERGIZER = "energizer"               # Lifts mood of any group they join
    TRUTH_TELLER = "truth_teller"         # Gives honest feedback, respected
    LOYAL_DEFENDER = "loyal_defender"     # Defends friends' reputations


class ArchetypeCategory(Enum):
    """Categories for organizing archetypes."""
    INTROVERSION_BASED = "introversion_based"
    EXTROVERSION_BASED = "extroversion_based"
    EMPATHY_BASED = "empathy_based"
    TRUST_BASED = "trust_based"
    UNIQUE_COMBINATION = "unique_combination"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SocialArchetype:
    """
    Data-driven social behavior template.

    Archetypes modify how agents interact socially based on
    personality trait combinations.
    """
    name: str
    description: str
    category: ArchetypeCategory

    # Trait requirements (min, max values to qualify)
    trait_requirements: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Behavior modifiers (multipliers, 1.0 = neutral)
    initiation_rate: float = 1.0      # How often they start interactions
    response_rate: float = 1.0        # How often they accept interactions
    deepening_rate: float = 1.0       # How fast relationships deepen
    max_close_friends: int = 3        # Relationship capacity
    group_comfort: float = 1.0        # Comfort in groups vs 1-on-1
    loyalty_multiplier: float = 1.0   # How loyal to existing friends
    openness_to_new: float = 1.0      # Willingness to meet new people
    emotional_availability: float = 1.0  # Willingness to share/receive emotions
    social_energy_cost: float = 1.0   # Energy cost of social interactions

    # Special abilities
    special_abilities: List[SpecialAbility] = field(default_factory=list)

    # Moodlet when in social situations
    social_moodlet: Optional[str] = None


@dataclass
class RelationshipModifiers:
    """Modifiers applied to specific relationship."""
    deepening_bonus: float = 0.0
    trust_bonus: float = 0.0
    conflict_resistance: float = 0.0
    special_bond: Optional[str] = None  # e.g., "mentor-mentee", "kindred_spirits"


@dataclass
class SocialState:
    """Tracks agent's social state and evolution."""
    archetype_id: str
    archetype_history: List[Tuple[str, int, str]] = field(default_factory=list)
    # (archetype_id, day_changed, reason)

    # Modifier adjustments from life events
    modifier_adjustments: Dict[str, float] = field(default_factory=dict)

    # Relationship-specific modifiers
    relationship_modifiers: Dict[str, RelationshipModifiers] = field(default_factory=dict)

    # Social metrics
    total_interactions_initiated: int = 0
    total_interactions_received: int = 0
    deep_conversations_count: int = 0
    conflicts_resolved: int = 0
    people_introduced: int = 0

    # Energy tracking
    social_energy: float = 100.0  # 0-100
    last_social_day: int = 0


# =============================================================================
# 20+ SOCIAL ARCHETYPES
# =============================================================================

SOCIAL_ARCHETYPES: Dict[str, SocialArchetype] = {
    # =========================================================================
    # INTROVERSION-BASED (5)
    # =========================================================================
    "introvert_best_friend": SocialArchetype(
        name="Introvert Best Friend",
        description="Few friends, but exceptionally loyal and supportive. "
                   "The rare friend you can count on for anything.",
        category=ArchetypeCategory.INTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.0, 0.4),
            "agreeableness": (0.6, 1.0),
        },
        initiation_rate=0.3,
        response_rate=0.8,
        deepening_rate=1.5,
        max_close_friends=2,
        group_comfort=0.4,
        loyalty_multiplier=2.0,
        openness_to_new=0.3,
        emotional_availability=1.3,
        social_energy_cost=1.5,
        special_abilities=[SpecialAbility.DEEP_LISTENER, SpecialAbility.SECRET_KEEPER, SpecialAbility.LOYAL_DEFENDER],
        social_moodlet="deeply_connected",
    ),

    "quiet_leader": SocialArchetype(
        name="Quiet Leader",
        description="Leads through example and wisdom, not charisma. "
                   "Others naturally look to them for guidance.",
        category=ArchetypeCategory.INTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.2, 0.5),
            "conscientiousness": (0.7, 1.0),
        },
        initiation_rate=0.5,
        response_rate=1.2,
        deepening_rate=1.0,
        max_close_friends=4,
        group_comfort=0.7,
        loyalty_multiplier=1.5,
        openness_to_new=0.6,
        emotional_availability=0.8,
        social_energy_cost=1.2,
        special_abilities=[SpecialAbility.MENTOR, SpecialAbility.WISE_COUNSEL, SpecialAbility.CALM_PRESENCE],
        social_moodlet="leading_quietly",
    ),

    "lone_wolf": SocialArchetype(
        name="Lone Wolf",
        description="Self-sufficient and independent, but occasionally "
                   "adopts 'strays' who need help.",
        category=ArchetypeCategory.INTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.0, 0.3),
            "openness": (0.4, 0.8),
        },
        initiation_rate=0.2,
        response_rate=0.5,
        deepening_rate=0.7,
        max_close_friends=1,
        group_comfort=0.2,
        loyalty_multiplier=1.8,
        openness_to_new=0.2,
        emotional_availability=0.5,
        social_energy_cost=2.0,
        special_abilities=[SpecialAbility.TRUTH_TELLER],
        social_moodlet="solo_content",
    ),

    "deep_thinker": SocialArchetype(
        name="Deep Thinker",
        description="Values intellectual connection over frequency. "
                   "Prefers meaningful discussions to small talk.",
        category=ArchetypeCategory.INTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.2, 0.5),
            "openness": (0.7, 1.0),
        },
        initiation_rate=0.4,
        response_rate=0.9,
        deepening_rate=1.3,
        max_close_friends=3,
        group_comfort=0.5,
        loyalty_multiplier=1.2,
        openness_to_new=0.5,
        emotional_availability=0.9,
        social_energy_cost=1.3,
        special_abilities=[SpecialAbility.WISE_COUNSEL, SpecialAbility.DEEP_LISTENER],
        social_moodlet="intellectually_stimulated",
    ),

    "selective_socializer": SocialArchetype(
        name="Selective Socializer",
        description="Highly picky about friends, but deeply loyal once chosen. "
                   "Quality over quantity, always.",
        category=ArchetypeCategory.INTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.3, 0.5),
            "conscientiousness": (0.6, 1.0),
        },
        initiation_rate=0.4,
        response_rate=0.6,
        deepening_rate=1.4,
        max_close_friends=2,
        group_comfort=0.5,
        loyalty_multiplier=1.7,
        openness_to_new=0.3,
        emotional_availability=1.1,
        social_energy_cost=1.4,
        special_abilities=[SpecialAbility.SECRET_KEEPER, SpecialAbility.LOYAL_DEFENDER],
        social_moodlet="carefully_connected",
    ),

    # =========================================================================
    # EXTROVERSION-BASED (5)
    # =========================================================================
    "social_butterfly": SocialArchetype(
        name="Social Butterfly",
        description="Knows everyone but relationships stay surface-level. "
                   "Life of every party, but rarely deeply intimate.",
        category=ArchetypeCategory.EXTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.7, 1.0),
            "openness": (0.5, 1.0),
        },
        initiation_rate=2.0,
        response_rate=0.9,
        deepening_rate=0.5,
        max_close_friends=1,
        group_comfort=1.5,
        loyalty_multiplier=0.6,
        openness_to_new=2.0,
        emotional_availability=0.6,
        social_energy_cost=0.5,
        special_abilities=[SpecialAbility.CONNECTOR, SpecialAbility.PARTY_STARTER],
        social_moodlet="socially_buzzing",
    ),

    "party_host": SocialArchetype(
        name="Party Host",
        description="Creates social events and brings people together. "
                   "Happiest when hosting and connecting others.",
        category=ArchetypeCategory.EXTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.7, 1.0),
            "agreeableness": (0.6, 1.0),
        },
        initiation_rate=1.8,
        response_rate=1.0,
        deepening_rate=0.8,
        max_close_friends=4,
        group_comfort=1.8,
        loyalty_multiplier=1.0,
        openness_to_new=1.5,
        emotional_availability=0.8,
        social_energy_cost=0.6,
        special_abilities=[SpecialAbility.PARTY_STARTER, SpecialAbility.CONNECTOR, SpecialAbility.ENERGIZER],
        social_moodlet="hosting_joy",
    ),

    "connector": SocialArchetype(
        name="The Connector",
        description="Introduces people to each other, builds networks. "
                   "Sees potential relationships others miss.",
        category=ArchetypeCategory.EXTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.6, 1.0),
            "openness": (0.6, 1.0),
        },
        initiation_rate=1.5,
        response_rate=1.0,
        deepening_rate=0.7,
        max_close_friends=5,
        group_comfort=1.3,
        loyalty_multiplier=0.9,
        openness_to_new=1.8,
        emotional_availability=0.7,
        social_energy_cost=0.7,
        special_abilities=[SpecialAbility.CONNECTOR, SpecialAbility.SOCIAL_CATALYST],
        social_moodlet="connecting_people",
    ),

    "energizer": SocialArchetype(
        name="The Energizer",
        description="Lifts the mood of any group they join. "
                   "Natural optimist who brings positive energy.",
        category=ArchetypeCategory.EXTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.7, 1.0),
            "neuroticism": (0.0, 0.4),
        },
        initiation_rate=1.6,
        response_rate=1.1,
        deepening_rate=0.9,
        max_close_friends=4,
        group_comfort=1.4,
        loyalty_multiplier=1.0,
        openness_to_new=1.4,
        emotional_availability=0.8,
        social_energy_cost=0.5,
        special_abilities=[SpecialAbility.ENERGIZER, SpecialAbility.CALM_PRESENCE],
        social_moodlet="radiating_positivity",
    ),

    "attention_seeker": SocialArchetype(
        name="Attention Seeker",
        description="Needs validation from others, can be drama-prone. "
                   "Charming but exhausting for close relationships.",
        category=ArchetypeCategory.EXTROVERSION_BASED,
        trait_requirements={
            "extraversion": (0.8, 1.0),
            "neuroticism": (0.5, 1.0),
        },
        initiation_rate=2.0,
        response_rate=0.7,
        deepening_rate=0.4,
        max_close_friends=2,
        group_comfort=1.5,
        loyalty_multiplier=0.5,
        openness_to_new=1.8,
        emotional_availability=1.3,
        social_energy_cost=0.4,
        special_abilities=[],  # No positive abilities
        social_moodlet="seeking_attention",
    ),

    # =========================================================================
    # EMPATHY-BASED (4)
    # =========================================================================
    "empath": SocialArchetype(
        name="The Empath",
        description="Absorbs others' emotions, natural counselor. "
                   "Deeply connected but needs time to recharge.",
        category=ArchetypeCategory.EMPATHY_BASED,
        trait_requirements={
            "agreeableness": (0.8, 1.0),
            "openness": (0.6, 1.0),
        },
        initiation_rate=0.8,
        response_rate=1.3,
        deepening_rate=1.4,
        max_close_friends=3,
        group_comfort=0.7,
        loyalty_multiplier=1.4,
        openness_to_new=0.8,
        emotional_availability=1.5,
        social_energy_cost=1.8,
        special_abilities=[SpecialAbility.DEEP_LISTENER, SpecialAbility.CALM_PRESENCE],
        social_moodlet="emotionally_attuned",
    ),

    "fixer": SocialArchetype(
        name="The Fixer",
        description="Tries to solve everyone's problems. "
                   "Helpful but can overstep boundaries.",
        category=ArchetypeCategory.EMPATHY_BASED,
        trait_requirements={
            "agreeableness": (0.6, 1.0),
            "conscientiousness": (0.6, 1.0),
        },
        initiation_rate=1.2,
        response_rate=1.2,
        deepening_rate=1.0,
        max_close_friends=4,
        group_comfort=1.0,
        loyalty_multiplier=1.2,
        openness_to_new=0.9,
        emotional_availability=1.2,
        social_energy_cost=1.3,
        special_abilities=[SpecialAbility.CONFLICT_RESOLVER, SpecialAbility.MENTOR],
        social_moodlet="problem_solving",
    ),

    "nurturer": SocialArchetype(
        name="The Nurturer",
        description="Creates safe spaces for vulnerability. "
                   "People feel comfortable opening up to them.",
        category=ArchetypeCategory.EMPATHY_BASED,
        trait_requirements={
            "agreeableness": (0.7, 1.0),
            "neuroticism": (0.0, 0.5),
        },
        initiation_rate=0.9,
        response_rate=1.3,
        deepening_rate=1.3,
        max_close_friends=5,
        group_comfort=0.9,
        loyalty_multiplier=1.3,
        openness_to_new=0.9,
        emotional_availability=1.4,
        social_energy_cost=1.2,
        special_abilities=[SpecialAbility.DEEP_LISTENER, SpecialAbility.SECRET_KEEPER, SpecialAbility.CALM_PRESENCE],
        social_moodlet="nurturing_others",
    ),

    "boundary_less": SocialArchetype(
        name="Boundary-Less Giver",
        description="Over-gives to others, risks burnout. "
                   "Kind heart but needs to protect their energy.",
        category=ArchetypeCategory.EMPATHY_BASED,
        trait_requirements={
            "agreeableness": (0.8, 1.0),
            "neuroticism": (0.4, 0.8),
        },
        initiation_rate=1.0,
        response_rate=1.5,
        deepening_rate=1.2,
        max_close_friends=6,
        group_comfort=1.0,
        loyalty_multiplier=1.5,
        openness_to_new=1.2,
        emotional_availability=1.8,
        social_energy_cost=2.0,
        special_abilities=[SpecialAbility.DEEP_LISTENER],
        social_moodlet="over_extended",
    ),

    # =========================================================================
    # TRUST-BASED (3)
    # =========================================================================
    "slow_trust_builder": SocialArchetype(
        name="Slow Trust Builder",
        description="Takes months to open up, but worth the wait. "
                   "Once you earn their trust, it's ironclad.",
        category=ArchetypeCategory.TRUST_BASED,
        trait_requirements={
            "neuroticism": (0.4, 0.7),
            "conscientiousness": (0.5, 1.0),
        },
        initiation_rate=0.4,
        response_rate=0.7,
        deepening_rate=0.5,
        max_close_friends=2,
        group_comfort=0.5,
        loyalty_multiplier=2.0,
        openness_to_new=0.3,
        emotional_availability=0.6,
        social_energy_cost=1.4,
        special_abilities=[SpecialAbility.SECRET_KEEPER, SpecialAbility.LOYAL_DEFENDER],
        social_moodlet="cautiously_trusting",
    ),

    "quick_truster": SocialArchetype(
        name="Quick Truster",
        description="Gives everyone a chance, gets hurt sometimes. "
                   "Open and optimistic about people's intentions.",
        category=ArchetypeCategory.TRUST_BASED,
        trait_requirements={
            "agreeableness": (0.6, 1.0),
            "neuroticism": (0.0, 0.4),
        },
        initiation_rate=1.3,
        response_rate=1.3,
        deepening_rate=1.5,
        max_close_friends=5,
        group_comfort=1.2,
        loyalty_multiplier=0.8,
        openness_to_new=1.5,
        emotional_availability=1.3,
        social_energy_cost=0.8,
        special_abilities=[SpecialAbility.CONNECTOR],
        social_moodlet="open_hearted",
    ),

    "betrayal_survivor": SocialArchetype(
        name="Betrayal Survivor",
        description="Walls up after past hurts, but can heal. "
                   "Guarded but yearning for connection.",
        category=ArchetypeCategory.TRUST_BASED,
        trait_requirements={
            "neuroticism": (0.5, 0.9),
            "openness": (0.3, 0.7),
        },
        initiation_rate=0.3,
        response_rate=0.5,
        deepening_rate=0.4,
        max_close_friends=1,
        group_comfort=0.4,
        loyalty_multiplier=1.0,
        openness_to_new=0.2,
        emotional_availability=0.4,
        social_energy_cost=1.8,
        special_abilities=[SpecialAbility.TRUTH_TELLER],
        social_moodlet="guarded_heart",
    ),

    # =========================================================================
    # UNIQUE COMBINATIONS (3)
    # =========================================================================
    "reluctant_mentor": SocialArchetype(
        name="Reluctant Mentor",
        description="Introverted expert who helps despite discomfort. "
                   "Doesn't seek the role but rises to it.",
        category=ArchetypeCategory.UNIQUE_COMBINATION,
        trait_requirements={
            "extraversion": (0.2, 0.5),
            "conscientiousness": (0.7, 1.0),
            "openness": (0.6, 1.0),
        },
        initiation_rate=0.4,
        response_rate=1.0,
        deepening_rate=1.1,
        max_close_friends=3,
        group_comfort=0.5,
        loyalty_multiplier=1.4,
        openness_to_new=0.5,
        emotional_availability=0.7,
        social_energy_cost=1.5,
        special_abilities=[SpecialAbility.MENTOR, SpecialAbility.WISE_COUNSEL],
        social_moodlet="reluctantly_guiding",
    ),

    "social_catalyst": SocialArchetype(
        name="Social Catalyst",
        description="Makes others more social around them. "
                   "Brings out the best in quiet people.",
        category=ArchetypeCategory.UNIQUE_COMBINATION,
        trait_requirements={
            "extraversion": (0.5, 0.8),
            "agreeableness": (0.7, 1.0),
            "openness": (0.6, 1.0),
        },
        initiation_rate=1.3,
        response_rate=1.2,
        deepening_rate=1.0,
        max_close_friends=4,
        group_comfort=1.2,
        loyalty_multiplier=1.1,
        openness_to_new=1.3,
        emotional_availability=1.0,
        social_energy_cost=0.9,
        special_abilities=[SpecialAbility.SOCIAL_CATALYST, SpecialAbility.CONNECTOR, SpecialAbility.ENERGIZER],
        social_moodlet="drawing_out_others",
    ),

    "peacemaker": SocialArchetype(
        name="The Peacemaker",
        description="Resolves conflicts, bridges divided groups. "
                   "Can see all sides and find common ground.",
        category=ArchetypeCategory.UNIQUE_COMBINATION,
        trait_requirements={
            "agreeableness": (0.7, 1.0),
            "neuroticism": (0.0, 0.5),
            "openness": (0.5, 0.8),
        },
        initiation_rate=0.9,
        response_rate=1.2,
        deepening_rate=1.0,
        max_close_friends=5,
        group_comfort=1.1,
        loyalty_multiplier=1.0,
        openness_to_new=1.0,
        emotional_availability=1.1,
        social_energy_cost=1.0,
        special_abilities=[SpecialAbility.CONFLICT_RESOLVER, SpecialAbility.CALM_PRESENCE, SpecialAbility.DEEP_LISTENER],
        social_moodlet="keeping_peace",
    ),
}


# =============================================================================
# SOCIAL MOODLETS
# =============================================================================

SOCIAL_MOODLETS: Dict[str, Dict[str, Any]] = {
    # Archetype-related moodlets
    "deeply_connected": {
        "mood": 15, "energy": -5, "duration": 24,
        "description": "Feeling a deep connection with close friends.",
    },
    "leading_quietly": {
        "mood": 10, "energy": -5, "duration": 12,
        "description": "Guiding others through example.",
    },
    "solo_content": {
        "mood": 10, "energy": 10, "duration": 8,
        "description": "Enjoying peaceful solitude.",
    },
    "intellectually_stimulated": {
        "mood": 15, "energy": 5, "duration": 12,
        "description": "Mind engaged in meaningful discussion.",
    },
    "socially_buzzing": {
        "mood": 20, "energy": 10, "duration": 8,
        "description": "Riding the social energy high.",
    },
    "hosting_joy": {
        "mood": 20, "energy": -10, "duration": 12,
        "description": "Happy from bringing people together.",
    },
    "emotionally_attuned": {
        "mood": 10, "energy": -15, "duration": 12,
        "description": "Deeply feeling others' emotions.",
    },
    "over_extended": {
        "mood": -10, "energy": -20, "duration": 24,
        "description": "Gave too much, feeling depleted.",
    },

    # Interaction moodlets
    "great_conversation": {
        "mood": 15, "energy": 5, "duration": 12,
        "description": "Had a wonderful, meaningful talk.",
    },
    "new_friend_potential": {
        "mood": 10, "energy": 5, "duration": 8,
        "description": "Met someone who could become a friend.",
    },
    "friend_bonded": {
        "mood": 20, "energy": 5, "duration": 24,
        "description": "Friendship deepened to a new level.",
    },
    "social_rejection": {
        "mood": -20, "energy": -10, "duration": 24,
        "description": "Felt rejected in a social situation.",
    },
    "awkward_interaction": {
        "mood": -10, "energy": -5, "duration": 8,
        "description": "Interaction was uncomfortable.",
    },
    "conflict_with_friend": {
        "mood": -25, "energy": -10, "duration": 48,
        "description": "Had a falling out with a friend.",
    },
    "conflict_resolved": {
        "mood": 20, "energy": 5, "duration": 24,
        "description": "Resolved a conflict successfully.",
    },
    "betrayed": {
        "mood": -40, "energy": -20, "duration": 168,
        "description": "Trust was broken by someone close.",
    },
    "defended_by_friend": {
        "mood": 25, "energy": 10, "duration": 48,
        "description": "A friend stood up for you.",
    },

    # Energy moodlets
    "social_battery_full": {
        "mood": 10, "energy": 20, "duration": 8,
        "description": "Feeling socially energized.",
    },
    "social_battery_empty": {
        "mood": -15, "energy": -20, "duration": 12,
        "description": "Socially exhausted, need alone time.",
    },
    "recharging_alone": {
        "mood": 10, "energy": 15, "duration": 8,
        "description": "Restoring energy in solitude.",
    },

    # Special event moodlets
    "made_introduction": {
        "mood": 10, "energy": 5, "duration": 8,
        "description": "Successfully connected two people.",
    },
    "mentored_someone": {
        "mood": 15, "energy": -5, "duration": 12,
        "description": "Shared wisdom with someone younger.",
    },
    "received_wise_counsel": {
        "mood": 15, "energy": 5, "duration": 24,
        "description": "Got valuable advice from a trusted source.",
    },
    "group_harmony": {
        "mood": 15, "energy": 5, "duration": 12,
        "description": "Group is getting along wonderfully.",
    },

    # Archetype evolution moodlets
    "archetype_shifting": {
        "mood": -5, "energy": -5, "duration": 48,
        "description": "Personality patterns are evolving.",
    },
    "found_social_identity": {
        "mood": 20, "energy": 10, "duration": 72,
        "description": "Discovered who you are socially.",
    },
}


# =============================================================================
# ARCHETYPE EVOLUTION TRIGGERS
# =============================================================================

EVOLUTION_TRIGGERS: Dict[str, Dict[str, Any]] = {
    # Negative events that shift archetypes
    "betrayal": {
        "description": "Experienced major betrayal",
        "modifier_changes": {
            "openness_to_new": -0.3,
            "emotional_availability": -0.2,
        },
        "possible_shift_to": ["slow_trust_builder", "betrayal_survivor", "lone_wolf"],
    },
    "repeated_rejection": {
        "description": "Experienced repeated social rejection",
        "modifier_changes": {
            "initiation_rate": -0.2,
            "openness_to_new": -0.2,
        },
        "possible_shift_to": ["lone_wolf", "selective_socializer", "betrayal_survivor"],
    },
    "burnout": {
        "description": "Burned out from over-giving",
        "modifier_changes": {
            "emotional_availability": -0.3,
            "social_energy_cost": 0.3,
        },
        "possible_shift_to": ["selective_socializer", "slow_trust_builder"],
    },

    # Positive events that shift archetypes
    "found_best_friend": {
        "description": "Found a true best friend",
        "modifier_changes": {
            "loyalty_multiplier": 0.2,
            "deepening_rate": 0.1,
        },
        "possible_shift_to": ["introvert_best_friend", "deep_thinker"],
    },
    "leadership_success": {
        "description": "Succeeded in a leadership role",
        "modifier_changes": {
            "initiation_rate": 0.1,
            "group_comfort": 0.1,
        },
        "possible_shift_to": ["quiet_leader", "social_catalyst"],
    },
    "conflict_resolution_success": {
        "description": "Successfully resolved major conflict",
        "modifier_changes": {
            "emotional_availability": 0.1,
        },
        "possible_shift_to": ["peacemaker", "fixer"],
    },
    "mentoring_experience": {
        "description": "Became a mentor to someone",
        "modifier_changes": {
            "loyalty_multiplier": 0.1,
        },
        "possible_shift_to": ["reluctant_mentor", "nurturer"],
    },
    "social_breakthrough": {
        "description": "Overcame social fears",
        "modifier_changes": {
            "initiation_rate": 0.2,
            "openness_to_new": 0.2,
        },
        "possible_shift_to": ["connector", "social_catalyst"],
    },
    "therapy_healing": {
        "description": "Healed through therapy or self-work",
        "modifier_changes": {
            "emotional_availability": 0.2,
            "openness_to_new": 0.1,
        },
        "possible_shift_to": ["quick_truster", "empath"],
    },
}


# =============================================================================
# EXTENSIBLE SOCIAL MANAGER
# =============================================================================

class ExtensibleSocialManager:
    """
    Manages social dynamics with archetype-based behavior modifiers.

    Extensible: archetypes can be added via config without code changes.
    """

    def __init__(
        self,
        archetype_config: Optional[Dict[str, SocialArchetype]] = None,
        moodlet_manager: Optional[Any] = None,
        wellness_manager: Optional[Any] = None,
        seed: int = 42,
    ):
        self.archetypes = archetype_config or SOCIAL_ARCHETYPES
        self.moodlets = moodlet_manager
        self.wellness = wellness_manager
        self.rng = random.Random(seed)

        # Agent states
        self.social_states: Dict[str, SocialState] = {}

        # Relationship tracking (agent_id -> {other_id: depth})
        self.relationships: Dict[str, Dict[str, float]] = {}

        # Statistics
        self.stats = {
            "total_interactions": 0,
            "deep_conversations": 0,
            "conflicts": 0,
            "conflicts_resolved": 0,
            "introductions_made": 0,
            "archetype_evolutions": 0,
        }

    def detect_archetype(
        self,
        agent_id: str,
        personality: Dict[str, float],
    ) -> str:
        """Find best-matching archetype for personality."""
        best_match = "selective_socializer"  # Default
        best_score = 0.0

        for archetype_id, archetype in self.archetypes.items():
            score = self._calculate_archetype_fit(personality, archetype)
            if score > best_score:
                best_score = score
                best_match = archetype_id

        return best_match

    def _calculate_archetype_fit(
        self,
        personality: Dict[str, float],
        archetype: SocialArchetype,
    ) -> float:
        """Calculate how well personality matches archetype requirements."""
        if not archetype.trait_requirements:
            return 0.5  # Neutral fit for archetypes with no requirements

        total_score = 0.0
        num_traits = len(archetype.trait_requirements)

        for trait, (min_val, max_val) in archetype.trait_requirements.items():
            trait_value = personality.get(trait, 0.5)

            if min_val <= trait_value <= max_val:
                # Perfect fit within range
                total_score += 1.0
            elif trait_value < min_val:
                # Below range - partial score based on distance
                distance = min_val - trait_value
                total_score += max(0, 1 - distance * 2)
            else:
                # Above range - partial score based on distance
                distance = trait_value - max_val
                total_score += max(0, 1 - distance * 2)

        return total_score / max(num_traits, 1)

    def initialize_agent(
        self,
        agent_id: str,
        personality: Dict[str, float],
        day: int = 0,
    ) -> SocialState:
        """Initialize social state for an agent."""
        archetype_id = self.detect_archetype(agent_id, personality)

        state = SocialState(
            archetype_id=archetype_id,
            archetype_history=[(archetype_id, day, "initial")],
        )

        self.social_states[agent_id] = state
        self.relationships[agent_id] = {}

        return state

    def get_archetype(self, agent_id: str) -> Optional[SocialArchetype]:
        """Get current archetype for agent."""
        state = self.social_states.get(agent_id)
        if not state:
            return None
        return self.archetypes.get(state.archetype_id)

    # =========================================================================
    # INTERACTION MODIFIERS
    # =========================================================================

    def modify_interaction_probability(
        self,
        agent_id: str,
        base_probability: float,
        interaction_type: str,  # "initiate", "respond", "deepen"
    ) -> float:
        """Apply archetype modifiers to interaction probability."""
        archetype = self.get_archetype(agent_id)
        if not archetype:
            return base_probability

        state = self.social_states[agent_id]

        # Get base modifier from archetype
        modifier = 1.0
        if interaction_type == "initiate":
            modifier = archetype.initiation_rate
        elif interaction_type == "respond":
            modifier = archetype.response_rate
        elif interaction_type == "deepen":
            modifier = archetype.deepening_rate

        # Apply adjustment from life events
        adjustment = state.modifier_adjustments.get(f"{interaction_type}_rate", 0)
        modifier += adjustment

        # Apply social energy modifier
        energy_factor = state.social_energy / 100
        if energy_factor < 0.3:
            modifier *= 0.5  # Very low energy = hard to socialize

        return min(1.0, base_probability * modifier)

    def calculate_social_energy_cost(
        self,
        agent_id: str,
        interaction_type: str,
        duration_minutes: int = 30,
    ) -> float:
        """Calculate energy cost for social interaction."""
        archetype = self.get_archetype(agent_id)
        if not archetype:
            return 10.0  # Default cost

        base_costs = {
            "brief_chat": 5,
            "conversation": 10,
            "deep_talk": 20,
            "group_event": 15,
            "party": 25,
        }

        base_cost = base_costs.get(interaction_type, 10)
        duration_factor = duration_minutes / 30  # Normalize to 30 min

        return base_cost * archetype.social_energy_cost * duration_factor

    def process_interaction(
        self,
        agent_id: str,
        other_id: str,
        interaction_type: str,
        duration_minutes: int = 30,
        was_positive: bool = True,
    ) -> Dict[str, Any]:
        """Process a social interaction between two agents."""
        state = self.social_states.get(agent_id)
        archetype = self.get_archetype(agent_id)
        if not state or not archetype:
            return {"success": False}

        # Calculate energy cost
        energy_cost = self.calculate_social_energy_cost(
            agent_id, interaction_type, duration_minutes
        )
        state.social_energy = max(0, state.social_energy - energy_cost)

        # Update relationship
        if other_id not in self.relationships[agent_id]:
            self.relationships[agent_id][other_id] = 0.0

        relationship_change = 0.0
        if was_positive:
            base_change = 5.0 if interaction_type == "deep_talk" else 2.0
            relationship_change = base_change * archetype.deepening_rate
        else:
            relationship_change = -3.0

        self.relationships[agent_id][other_id] = min(
            100, max(-50, self.relationships[agent_id][other_id] + relationship_change)
        )

        # Update statistics
        state.total_interactions_initiated += 1
        if interaction_type == "deep_talk":
            state.deep_conversations_count += 1
            self.stats["deep_conversations"] += 1

        self.stats["total_interactions"] += 1

        # Add moodlets
        if self.moodlets and was_positive:
            if interaction_type == "deep_talk":
                self.moodlets.add_moodlet(agent_id, "great_conversation", "social")

        # Check if relationship deepened significantly
        if self.relationships[agent_id][other_id] >= 50 and \
           self.relationships[agent_id][other_id] - relationship_change < 50:
            if self.moodlets:
                self.moodlets.add_moodlet(agent_id, "friend_bonded", "social")

        # Update wellness if available
        if self.wellness and interaction_type == "deep_talk":
            # Boost deep_dialogue pillar
            self.wellness.update_pillar(agent_id, "deep_dialogue", 5)

        return {
            "success": True,
            "energy_cost": energy_cost,
            "relationship_change": relationship_change,
            "current_relationship": self.relationships[agent_id][other_id],
            "social_energy_remaining": state.social_energy,
        }

    # =========================================================================
    # SPECIAL ABILITIES
    # =========================================================================

    def trigger_ability(
        self,
        agent_id: str,
        ability: SpecialAbility,
        target_id: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Trigger a special ability."""
        archetype = self.get_archetype(agent_id)
        if not archetype or ability not in archetype.special_abilities:
            return {"success": False, "reason": "Agent doesn't have this ability"}

        result = {"success": True, "ability": ability.value}

        if ability == SpecialAbility.DEEP_LISTENER:
            # +50% relationship gain from conversations
            result["bonus"] = "relationship_gain_x1.5"

        elif ability == SpecialAbility.SECRET_KEEPER:
            # Never spreads gossip
            result["effect"] = "gossip_immunity"

        elif ability == SpecialAbility.CONNECTOR:
            # Introduce two people
            if target_id and context and context.get("introduce_to"):
                other_id = context["introduce_to"]
                # Create initial relationship between introduced people
                if other_id not in self.relationships:
                    self.relationships[other_id] = {}
                if target_id not in self.relationships:
                    self.relationships[target_id] = {}
                self.relationships[target_id][other_id] = 15.0
                self.relationships[other_id][target_id] = 15.0
                self.stats["introductions_made"] += 1
                if self.moodlets:
                    self.moodlets.add_moodlet(agent_id, "made_introduction", "social")
                result["introduced"] = (target_id, other_id)

        elif ability == SpecialAbility.MENTOR:
            # Boost skill learning for target
            if target_id:
                result["target"] = target_id
                result["effect"] = "skill_learning_x1.25"
                if self.moodlets:
                    self.moodlets.add_moodlet(agent_id, "mentored_someone", "social")

        elif ability == SpecialAbility.CALM_PRESENCE:
            # Reduce anxiety in nearby agents
            result["effect"] = "anxiety_reduction"

        elif ability == SpecialAbility.CONFLICT_RESOLVER:
            # Heal relationship damage
            if target_id and context and context.get("other_party"):
                other_id = context["other_party"]
                if target_id in self.relationships and other_id in self.relationships.get(target_id, {}):
                    self.relationships[target_id][other_id] += 20
                    self.relationships[other_id][target_id] += 20
                    self.stats["conflicts_resolved"] += 1
                    if self.moodlets:
                        self.moodlets.add_moodlet(target_id, "conflict_resolved", "social")
                        self.moodlets.add_moodlet(other_id, "conflict_resolved", "social")
                    result["healed"] = (target_id, other_id)

        elif ability == SpecialAbility.ENERGIZER:
            # Boost mood of everyone in group
            result["effect"] = "group_mood_boost"

        return result

    # =========================================================================
    # ARCHETYPE EVOLUTION
    # =========================================================================

    def process_life_event(
        self,
        agent_id: str,
        event_type: str,
        day: int,
    ) -> Dict[str, Any]:
        """Process a life event that might trigger archetype evolution."""
        state = self.social_states.get(agent_id)
        if not state:
            return {"success": False}

        trigger = EVOLUTION_TRIGGERS.get(event_type)
        if not trigger:
            return {"success": True, "evolution": False}

        # Apply modifier changes
        for modifier, change in trigger.get("modifier_changes", {}).items():
            current = state.modifier_adjustments.get(modifier, 0)
            state.modifier_adjustments[modifier] = current + change

        # Check for archetype shift (30% base chance on trigger)
        if self.rng.random() < 0.3:
            possible_shifts = trigger.get("possible_shift_to", [])
            if possible_shifts:
                new_archetype = self.rng.choice(possible_shifts)
                old_archetype = state.archetype_id
                state.archetype_id = new_archetype
                state.archetype_history.append((new_archetype, day, event_type))
                self.stats["archetype_evolutions"] += 1

                if self.moodlets:
                    self.moodlets.add_moodlet(agent_id, "archetype_shifting", "social")

                return {
                    "success": True,
                    "evolution": True,
                    "old_archetype": old_archetype,
                    "new_archetype": new_archetype,
                    "trigger": event_type,
                }

        return {"success": True, "evolution": False, "modifiers_adjusted": True}

    # =========================================================================
    # DAILY UPDATE
    # =========================================================================

    def daily_update(self, day: int):
        """Daily update for all agents."""
        for agent_id, state in self.social_states.items():
            archetype = self.get_archetype(agent_id)
            if not archetype:
                continue

            # Regenerate social energy
            regen_rate = 20 / archetype.social_energy_cost  # Inverse of cost
            state.social_energy = min(100, state.social_energy + regen_rate)

            # Check for social battery depletion moodlet
            if self.moodlets:
                if state.social_energy < 20:
                    self.moodlets.add_moodlet(agent_id, "social_battery_empty", "social")
                elif state.social_energy > 80:
                    self.moodlets.add_moodlet(agent_id, "social_battery_full", "social")

            state.last_social_day = day

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_close_friends(self, agent_id: str, threshold: float = 50.0) -> List[str]:
        """Get list of close friends for agent."""
        relationships = self.relationships.get(agent_id, {})
        return [other_id for other_id, depth in relationships.items() if depth >= threshold]

    def count_close_friends(self, agent_id: str) -> int:
        """Count close friends (for wellness integration)."""
        return len(self.get_close_friends(agent_id))

    def get_relationship_depth(self, agent_id: str, other_id: str) -> float:
        """Get relationship depth between two agents."""
        return self.relationships.get(agent_id, {}).get(other_id, 0.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get social system statistics."""
        archetype_counts = {}
        for state in self.social_states.values():
            archetype_counts[state.archetype_id] = archetype_counts.get(state.archetype_id, 0) + 1

        return {
            "total_agents": len(self.social_states),
            "archetype_distribution": archetype_counts,
            **self.stats,
        }

    def get_social_panel(self, agent_id: str) -> str:
        """Get formatted social panel for display."""
        state = self.social_states.get(agent_id)
        archetype = self.get_archetype(agent_id)
        if not state or not archetype:
            return "No social data"

        lines = []
        lines.append(f"{'='*45}")
        lines.append(f"SOCIAL PROFILE: {archetype.name.upper()}")
        lines.append(f"{'='*45}")
        lines.append(f"Category: {archetype.category.value}")
        lines.append(f"Description: {archetype.description[:60]}...")
        lines.append(f"\nSOCIAL ENERGY: {state.social_energy:.1f}/100")

        # Key modifiers
        lines.append(f"\nBEHAVIOR MODIFIERS:")
        lines.append(f"  Initiation: {archetype.initiation_rate:.1f}x")
        lines.append(f"  Response: {archetype.response_rate:.1f}x")
        lines.append(f"  Deepening: {archetype.deepening_rate:.1f}x")
        lines.append(f"  Max Close Friends: {archetype.max_close_friends}")
        lines.append(f"  Loyalty: {archetype.loyalty_multiplier:.1f}x")

        # Special abilities
        if archetype.special_abilities:
            lines.append(f"\nSPECIAL ABILITIES:")
            for ability in archetype.special_abilities:
                lines.append(f"  - {ability.value}")

        # Relationships
        close_friends = self.get_close_friends(agent_id)
        lines.append(f"\nRELATIONSHIPS:")
        lines.append(f"  Close Friends: {len(close_friends)}")
        lines.append(f"  Total Interactions: {state.total_interactions_initiated}")
        lines.append(f"  Deep Conversations: {state.deep_conversations_count}")

        # Archetype history
        if len(state.archetype_history) > 1:
            lines.append(f"\nARCHETYPE HISTORY:")
            for arch_id, day, reason in state.archetype_history[-3:]:
                lines.append(f"  Day {day}: {arch_id} ({reason})")

        lines.append(f"{'='*45}")
        return "\n".join(lines)


# =============================================================================
# DEMO FUNCTION
# =============================================================================

async def demo_enhanced_social():
    """Demonstrate the enhanced social system."""
    print("=" * 60)
    print("v8.4 ENHANCED SOCIAL SYSTEM DEMO")
    print("=" * 60)

    manager = ExtensibleSocialManager(seed=42)

    # Create agents with different personalities
    personalities = {
        "alice": {
            "extraversion": 0.3, "agreeableness": 0.8,
            "openness": 0.6, "conscientiousness": 0.7, "neuroticism": 0.4
        },
        "bob": {
            "extraversion": 0.8, "agreeableness": 0.6,
            "openness": 0.7, "conscientiousness": 0.5, "neuroticism": 0.3
        },
        "carol": {
            "extraversion": 0.5, "agreeableness": 0.9,
            "openness": 0.7, "conscientiousness": 0.6, "neuroticism": 0.2
        },
        "david": {
            "extraversion": 0.2, "agreeableness": 0.5,
            "openness": 0.4, "conscientiousness": 0.6, "neuroticism": 0.7
        },
    }

    print("\n1. DETECTING ARCHETYPES FROM PERSONALITY")
    print("-" * 40)
    for agent_id, personality in personalities.items():
        state = manager.initialize_agent(agent_id, personality, day=1)
        archetype = manager.get_archetype(agent_id)
        print(f"{agent_id}: {archetype.name}")
        print(f"   {archetype.description[:50]}...")

    print("\n2. ARCHETYPE PANELS")
    print("-" * 40)
    for agent_id in ["alice", "bob"]:
        print(manager.get_social_panel(agent_id))

    print("\n3. SIMULATING INTERACTIONS")
    print("-" * 40)

    # Alice and Carol have a deep conversation
    result = manager.process_interaction(
        "alice", "carol", "deep_talk", duration_minutes=60, was_positive=True
    )
    print(f"Alice & Carol deep talk:")
    print(f"  Energy cost: {result['energy_cost']:.1f}")
    print(f"  Relationship: {result['current_relationship']:.1f}")

    # Bob tries to connect with David (introvert)
    result = manager.process_interaction(
        "bob", "david", "conversation", duration_minutes=20, was_positive=True
    )
    print(f"\nBob & David conversation:")
    print(f"  Relationship: {result['current_relationship']:.1f}")

    # Multiple interactions to build relationships
    for _ in range(5):
        manager.process_interaction("alice", "carol", "conversation", was_positive=True)

    print(f"\nAfter 5 more interactions:")
    print(f"  Alice-Carol depth: {manager.get_relationship_depth('alice', 'carol'):.1f}")
    print(f"  Alice close friends: {manager.count_close_friends('alice')}")

    print("\n4. TESTING SPECIAL ABILITIES")
    print("-" * 40)

    # Carol (likely nurturer/peacemaker) uses conflict resolver
    archetype = manager.get_archetype("carol")
    print(f"Carol's archetype: {archetype.name}")
    print(f"Carol's abilities: {[a.value for a in archetype.special_abilities]}")

    if SpecialAbility.DEEP_LISTENER in archetype.special_abilities:
        result = manager.trigger_ability("carol", SpecialAbility.DEEP_LISTENER)
        print(f"Deep Listener ability: {result}")

    # Bob (likely connector) introduces people
    if SpecialAbility.CONNECTOR in manager.get_archetype("bob").special_abilities:
        result = manager.trigger_ability(
            "bob", SpecialAbility.CONNECTOR,
            target_id="alice",
            context={"introduce_to": "david"}
        )
        print(f"Connector ability: {result}")

    print("\n5. ARCHETYPE EVOLUTION (Life Events)")
    print("-" * 40)

    # David experiences betrayal
    result = manager.process_life_event("david", "betrayal", day=30)
    print(f"David after betrayal: {result}")
    if result.get("evolution"):
        print(f"  Old: {result['old_archetype']} -> New: {result['new_archetype']}")

    # Alice finds best friend
    result = manager.process_life_event("alice", "found_best_friend", day=30)
    print(f"Alice finds best friend: {result}")

    print("\n6. DAILY UPDATE")
    print("-" * 40)
    manager.daily_update(day=31)

    for agent_id in ["alice", "bob", "carol", "david"]:
        state = manager.social_states[agent_id]
        archetype = manager.get_archetype(agent_id)
        print(f"{agent_id}: Energy {state.social_energy:.1f}/100, Archetype: {archetype.name}")

    print("\n7. STATISTICS")
    print("-" * 40)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_enhanced_social())
