"""
NeuroHood v8.5: Life Stages (Branching Life Paths)

Branching life path system where each stage has multiple possible trajectories
based on choices, circumstances, and personality - not just linear age progression.

Key Features:
- 8 Core Life Stages with 3-4 Sub-Branches each (30+ total paths)
- Life events trigger branch changes
- Same age can mean very different life situations
- Generational differences modeled (Silent → Alpha)
- 50+ life milestone events
- Family relationship tracking

Author: Claude
Date: November 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import random


# =============================================================================
# ENUMS
# =============================================================================

class LifeStage(Enum):
    """8 Core Life Stages."""
    CHILDHOOD = "childhood"           # 0-12
    ADOLESCENCE = "adolescence"       # 13-17
    EMERGING_ADULT = "emerging_adult" # 18-25
    YOUNG_ADULT = "young_adult"       # 26-35
    ESTABLISHED = "established"       # 36-50
    MIDLIFE = "midlife"               # 51-65
    ACTIVE_ELDER = "active_elder"     # 66-80
    LATE_ELDER = "late_elder"         # 80+


class Generation(Enum):
    """Generational cohorts with distinct characteristics."""
    SILENT = "silent"           # 1928-1945
    BOOMER = "boomer"           # 1946-1964
    GEN_X = "gen_x"             # 1965-1980
    MILLENNIAL = "millennial"   # 1981-1996
    GEN_Z = "gen_z"             # 1997-2012
    ALPHA = "alpha"             # 2013+


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class LifeBranch:
    """Sub-stage within a life stage."""
    name: str
    stage: LifeStage
    description: str

    # Modifiers (multipliers)
    energy_modifier: float = 1.0
    health_modifier: float = 1.0
    social_modifier: float = 1.0
    career_modifier: float = 1.0
    purpose_modifier: float = 1.0

    # Triggers for entering this branch
    triggers: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    """A significant life event."""
    name: str
    stage: LifeStage
    description: str
    is_optional: bool = False
    typical_age_range: Tuple[int, int] = (0, 100)

    # Effects when completed
    purpose_boost: float = 0.0
    social_boost: float = 0.0
    career_boost: float = 0.0

    # Effects if missed (past typical age without completing)
    regret_if_missed: float = 0.0


@dataclass
class LifeStageState:
    """Current life stage state for an agent."""
    agent_id: str
    current_stage: LifeStage = LifeStage.YOUNG_ADULT
    current_branch: str = "career_focused"
    age: int = 30
    days_until_birthday: int = 180

    # Life history
    previous_branches: List[Tuple[LifeStage, str]] = field(default_factory=list)
    milestones_completed: List[str] = field(default_factory=list)
    milestones_missed: List[str] = field(default_factory=list)

    # Family
    children: List[str] = field(default_factory=list)  # agent_ids
    parents: List[str] = field(default_factory=list)
    spouse: Optional[str] = None

    # Generation
    generation: Generation = Generation.MILLENNIAL


# =============================================================================
# BRANCH DEFINITIONS (30+ Total)
# =============================================================================

LIFE_BRANCHES: Dict[LifeStage, Dict[str, LifeBranch]] = {
    # CHILDHOOD (0-12) - 4 branches
    LifeStage.CHILDHOOD: {
        "sheltered": LifeBranch(
            name="Sheltered",
            stage=LifeStage.CHILDHOOD,
            description="Protected, enriched childhood with stable family",
            social_modifier=1.2,
            health_modifier=1.1,
            triggers=["stable_family", "high_resources", "two_parents"]
        ),
        "latchkey": LifeBranch(
            name="Latchkey",
            stage=LifeStage.CHILDHOOD,
            description="Independent from young age with working parents",
            energy_modifier=1.1,
            social_modifier=0.9,
            triggers=["working_parents", "single_parent", "self_sufficient"]
        ),
        "enriched": LifeBranch(
            name="Enriched",
            stage=LifeStage.CHILDHOOD,
            description="Activities, lessons, and heavy structure",
            career_modifier=1.2,
            purpose_modifier=0.9,
            triggers=["helicopter_parents", "many_activities", "high_achiever"]
        ),
        "struggling": LifeBranch(
            name="Struggling",
            stage=LifeStage.CHILDHOOD,
            description="Adverse childhood experiences building resilience",
            health_modifier=0.9,
            social_modifier=0.8,
            energy_modifier=1.1,
            triggers=["poverty", "instability", "trauma", "neglect"]
        ),
    },

    # ADOLESCENCE (13-17) - 4 branches
    LifeStage.ADOLESCENCE: {
        "rebellious": LifeBranch(
            name="Rebellious",
            stage=LifeStage.ADOLESCENCE,
            description="Pushing boundaries, frequent conflict with authority",
            social_modifier=1.1,
            career_modifier=0.8,
            triggers=["authority_issues", "rule_breaking", "independence_seeking"]
        ),
        "studious": LifeBranch(
            name="Studious",
            stage=LifeStage.ADOLESCENCE,
            description="Academic focus with good grades priority",
            career_modifier=1.3,
            social_modifier=0.8,
            triggers=["academic_success", "college_prep", "high_grades"]
        ),
        "social": LifeBranch(
            name="Social",
            stage=LifeStage.ADOLESCENCE,
            description="Peer-focused with popularity priorities",
            social_modifier=1.4,
            career_modifier=0.9,
            triggers=["popularity", "dating", "sports_team", "many_friends"]
        ),
        "struggling": LifeBranch(
            name="Struggling",
            stage=LifeStage.ADOLESCENCE,
            description="Mental health or identity challenges",
            purpose_modifier=0.7,
            health_modifier=0.8,
            triggers=["depression", "anxiety", "identity_crisis", "isolation"]
        ),
    },

    # EMERGING ADULT (18-25) - 4 branches
    LifeStage.EMERGING_ADULT: {
        "student": LifeBranch(
            name="Student",
            stage=LifeStage.EMERGING_ADULT,
            description="Higher education path with college experience",
            career_modifier=1.2,
            purpose_modifier=1.1,
            triggers=["college_enrollment", "university", "graduate_school"]
        ),
        "worker": LifeBranch(
            name="Worker",
            stage=LifeStage.EMERGING_ADULT,
            description="Direct to workforce gaining experience",
            career_modifier=1.0,
            energy_modifier=0.9,
            triggers=["no_college", "trade_school", "early_job", "military"]
        ),
        "explorer": LifeBranch(
            name="Explorer",
            stage=LifeStage.EMERGING_ADULT,
            description="Travel and self-discovery phase",
            social_modifier=1.2,
            purpose_modifier=1.1,
            career_modifier=0.7,
            triggers=["gap_year", "travel", "adventure_seeking", "backpacking"]
        ),
        "caretaker": LifeBranch(
            name="Caretaker",
            stage=LifeStage.EMERGING_ADULT,
            description="Early family responsibilities",
            purpose_modifier=1.1,
            career_modifier=0.8,
            social_modifier=0.8,
            triggers=["early_children", "sick_parent", "family_obligations"]
        ),
    },

    # YOUNG ADULT (26-35) - 4 branches
    LifeStage.YOUNG_ADULT: {
        "career_focused": LifeBranch(
            name="Career-Focused",
            stage=LifeStage.YOUNG_ADULT,
            description="Ambitious professional climbing the ladder",
            career_modifier=1.4,
            social_modifier=0.8,
            health_modifier=0.9,
            triggers=["promotions", "long_hours", "career_goals", "ambition"]
        ),
        "family_focused": LifeBranch(
            name="Family-Focused",
            stage=LifeStage.YOUNG_ADULT,
            description="Building family as primary focus",
            social_modifier=1.2,
            purpose_modifier=1.2,
            career_modifier=0.8,
            triggers=["marriage", "children", "home_purchase", "nesting"]
        ),
        "adventure_seeking": LifeBranch(
            name="Adventure-Seeking",
            stage=LifeStage.YOUNG_ADULT,
            description="Still exploring and accumulating experiences",
            social_modifier=1.3,
            energy_modifier=1.1,
            career_modifier=0.7,
            triggers=["travel", "hobbies", "dating", "experiences"]
        ),
        "finding_self": LifeBranch(
            name="Finding Self",
            stage=LifeStage.YOUNG_ADULT,
            description="Late bloomer figuring things out",
            purpose_modifier=0.8,
            career_modifier=0.8,
            triggers=["career_change", "therapy", "self_discovery", "uncertainty"]
        ),
    },

    # ESTABLISHED ADULT (36-50) - 4 branches
    LifeStage.ESTABLISHED: {
        "peak_career": LifeBranch(
            name="Peak Career",
            stage=LifeStage.ESTABLISHED,
            description="Leadership roles and career accomplishment",
            career_modifier=1.5,
            energy_modifier=0.9,
            health_modifier=0.9,
            triggers=["leadership_role", "executive", "business_owner", "expertise"]
        ),
        "peak_family": LifeBranch(
            name="Peak Family",
            stage=LifeStage.ESTABLISHED,
            description="Active parenting of school-age children",
            purpose_modifier=1.2,
            social_modifier=1.1,
            career_modifier=0.9,
            triggers=["teen_children", "school_involvement", "family_activities"]
        ),
        "second_wind": LifeBranch(
            name="Second Wind",
            stage=LifeStage.ESTABLISHED,
            description="New chapter after major life change",
            energy_modifier=1.1,
            purpose_modifier=0.9,
            career_modifier=0.8,
            triggers=["divorce", "career_change", "new_relationship", "relocation"]
        ),
        "plateau": LifeBranch(
            name="Plateau",
            stage=LifeStage.ESTABLISHED,
            description="Comfortable routine, stable but stagnant",
            energy_modifier=0.9,
            purpose_modifier=0.7,
            triggers=["comfort", "routine", "no_changes", "settled"]
        ),
    },

    # MIDLIFE (51-65) - 4 branches
    LifeStage.MIDLIFE: {
        "thriving": LifeBranch(
            name="Thriving",
            stage=LifeStage.MIDLIFE,
            description="Peak accomplishment across life domains",
            purpose_modifier=1.3,
            health_modifier=0.95,
            social_modifier=1.1,
            triggers=["career_peak", "healthy_relationships", "financial_security"]
        ),
        "crisis": LifeBranch(
            name="Midlife Crisis",
            stage=LifeStage.MIDLIFE,
            description="Questioning life choices and seeking meaning",
            purpose_modifier=0.5,
            energy_modifier=0.8,
            triggers=["regret", "empty_nest", "career_plateau", "health_scare"]
        ),
        "reinvention": LifeBranch(
            name="Reinvention",
            stage=LifeStage.MIDLIFE,
            description="Starting fresh with new identity or career",
            energy_modifier=1.1,
            career_modifier=0.7,
            purpose_modifier=1.0,
            triggers=["career_change", "divorce", "moved_cities", "new_degree"]
        ),
        "coasting": LifeBranch(
            name="Coasting",
            stage=LifeStage.MIDLIFE,
            description="Comfortable routine waiting for retirement",
            energy_modifier=0.9,
            purpose_modifier=0.7,
            triggers=["comfortable", "risk_averse", "routine", "counting_down"]
        ),
    },

    # ACTIVE ELDER (66-80) - 4 branches
    LifeStage.ACTIVE_ELDER: {
        "engaged_elder": LifeBranch(
            name="Engaged Elder",
            stage=LifeStage.ACTIVE_ELDER,
            description="Active community member with purpose",
            purpose_modifier=1.2,
            social_modifier=1.3,
            triggers=["volunteering", "travel", "community_involvement", "active_lifestyle"]
        ),
        "quiet_retirement": LifeBranch(
            name="Quiet Retirement",
            stage=LifeStage.ACTIVE_ELDER,
            description="Peaceful rest focused on hobbies",
            health_modifier=1.0,
            social_modifier=0.8,
            purpose_modifier=0.9,
            triggers=["hobbies", "relaxation", "gardening", "reading"]
        ),
        "encore_career": LifeBranch(
            name="Encore Career",
            stage=LifeStage.ACTIVE_ELDER,
            description="Second career or consulting work",
            career_modifier=0.8,
            purpose_modifier=1.2,
            energy_modifier=0.9,
            triggers=["consulting", "new_business", "part_time_work", "expertise_sharing"]
        ),
        "caregiver": LifeBranch(
            name="Caregiver",
            stage=LifeStage.ACTIVE_ELDER,
            description="Caring for grandchildren or spouse",
            purpose_modifier=1.1,
            health_modifier=0.85,
            energy_modifier=0.8,
            triggers=["grandchildren", "spouse_health", "family_needs"]
        ),
    },

    # LATE ELDER (80+) - 3 branches
    LifeStage.LATE_ELDER: {
        "vital": LifeBranch(
            name="Vital",
            stage=LifeStage.LATE_ELDER,
            description="Still independent and healthy",
            health_modifier=1.0,
            energy_modifier=0.8,
            purpose_modifier=1.0,
            triggers=["good_health", "independence", "active"]
        ),
        "frail": LifeBranch(
            name="Frail",
            stage=LifeStage.LATE_ELDER,
            description="Declining health requiring support",
            health_modifier=0.6,
            energy_modifier=0.5,
            social_modifier=0.7,
            triggers=["health_decline", "assisted_living", "mobility_issues"]
        ),
        "wisdom_keeper": LifeBranch(
            name="Wisdom Keeper",
            stage=LifeStage.LATE_ELDER,
            description="Respected elder sharing knowledge",
            purpose_modifier=1.3,
            social_modifier=1.2,
            health_modifier=0.8,
            triggers=["mentoring", "storytelling", "community_respect", "legacy"]
        ),
    },
}


# =============================================================================
# MILESTONE EVENTS (50+)
# =============================================================================

LIFE_MILESTONES: Dict[str, Milestone] = {
    # Childhood milestones
    "first_day_school": Milestone(
        name="First Day of School",
        stage=LifeStage.CHILDHOOD,
        description="Starting formal education",
        typical_age_range=(5, 6),
        social_boost=0.1
    ),
    "first_friend": Milestone(
        name="First True Friend",
        stage=LifeStage.CHILDHOOD,
        description="Making a deep childhood friendship",
        typical_age_range=(4, 10),
        social_boost=0.15,
        is_optional=True
    ),
    "learned_to_ride_bike": Milestone(
        name="Learned to Ride Bike",
        stage=LifeStage.CHILDHOOD,
        description="Mastering bicycle riding",
        typical_age_range=(5, 8),
        purpose_boost=0.05,
        is_optional=True
    ),

    # Adolescence milestones
    "puberty": Milestone(
        name="Puberty",
        stage=LifeStage.ADOLESCENCE,
        description="Physical maturation begins",
        typical_age_range=(10, 14)
    ),
    "first_heartbreak": Milestone(
        name="First Heartbreak",
        stage=LifeStage.ADOLESCENCE,
        description="First romantic disappointment",
        typical_age_range=(13, 18),
        is_optional=True,
        regret_if_missed=0.0  # Not regrettable if missed
    ),
    "drivers_license": Milestone(
        name="Driver's License",
        stage=LifeStage.ADOLESCENCE,
        description="Earning the freedom to drive",
        typical_age_range=(16, 18),
        purpose_boost=0.1,
        is_optional=True
    ),
    "prom": Milestone(
        name="Prom",
        stage=LifeStage.ADOLESCENCE,
        description="High school formal dance",
        typical_age_range=(16, 18),
        social_boost=0.05,
        is_optional=True
    ),
    "high_school_graduation": Milestone(
        name="High School Graduation",
        stage=LifeStage.ADOLESCENCE,
        description="Completing secondary education",
        typical_age_range=(17, 19),
        purpose_boost=0.15,
        career_boost=0.1,
        regret_if_missed=0.2
    ),

    # Emerging adult milestones
    "moved_out": Milestone(
        name="Moved Out",
        stage=LifeStage.EMERGING_ADULT,
        description="Living independently from parents",
        typical_age_range=(18, 25),
        purpose_boost=0.1
    ),
    "first_job": Milestone(
        name="First Real Job",
        stage=LifeStage.EMERGING_ADULT,
        description="First full-time employment",
        typical_age_range=(18, 24),
        career_boost=0.15,
        purpose_boost=0.1
    ),
    "first_apartment": Milestone(
        name="First Apartment",
        stage=LifeStage.EMERGING_ADULT,
        description="Signing your own lease",
        typical_age_range=(18, 26),
        purpose_boost=0.1
    ),
    "college_graduation": Milestone(
        name="College Graduation",
        stage=LifeStage.EMERGING_ADULT,
        description="Completing higher education",
        typical_age_range=(21, 26),
        career_boost=0.2,
        purpose_boost=0.15,
        is_optional=True
    ),
    "first_serious_relationship": Milestone(
        name="First Serious Relationship",
        stage=LifeStage.EMERGING_ADULT,
        description="First committed long-term relationship",
        typical_age_range=(18, 28),
        social_boost=0.15,
        is_optional=True
    ),

    # Young adult milestones
    "wedding": Milestone(
        name="Wedding",
        stage=LifeStage.YOUNG_ADULT,
        description="Getting married",
        typical_age_range=(24, 40),
        social_boost=0.2,
        purpose_boost=0.15,
        is_optional=True
    ),
    "first_child": Milestone(
        name="First Child",
        stage=LifeStage.YOUNG_ADULT,
        description="Becoming a parent",
        typical_age_range=(25, 42),
        purpose_boost=0.25,
        social_boost=0.1,
        is_optional=True
    ),
    "bought_home": Milestone(
        name="Bought Home",
        stage=LifeStage.YOUNG_ADULT,
        description="Purchasing first home",
        typical_age_range=(28, 45),
        purpose_boost=0.15,
        is_optional=True
    ),
    "career_breakthrough": Milestone(
        name="Career Breakthrough",
        stage=LifeStage.YOUNG_ADULT,
        description="Major career advancement",
        typical_age_range=(26, 40),
        career_boost=0.25,
        purpose_boost=0.1,
        is_optional=True
    ),

    # Established adult milestones
    "promotion_to_leadership": Milestone(
        name="Promotion to Leadership",
        stage=LifeStage.ESTABLISHED,
        description="Reaching leadership position",
        typical_age_range=(35, 55),
        career_boost=0.2,
        purpose_boost=0.1,
        is_optional=True
    ),
    "kids_graduation": Milestone(
        name="Child's Graduation",
        stage=LifeStage.ESTABLISHED,
        description="Child completing major education",
        typical_age_range=(40, 60),
        purpose_boost=0.15,
        is_optional=True
    ),
    "empty_nest": Milestone(
        name="Empty Nest",
        stage=LifeStage.ESTABLISHED,
        description="Last child leaves home",
        typical_age_range=(45, 65),
        purpose_boost=-0.1,  # Can cause crisis
        is_optional=True
    ),
    "parents_health_decline": Milestone(
        name="Parents' Health Decline",
        stage=LifeStage.ESTABLISHED,
        description="Caring for aging parents",
        typical_age_range=(40, 60),
        purpose_boost=0.05
    ),
    "divorce": Milestone(
        name="Divorce",
        stage=LifeStage.ESTABLISHED,
        description="Marriage ending",
        typical_age_range=(30, 60),
        social_boost=-0.15,
        purpose_boost=-0.1,
        is_optional=True
    ),
    "health_scare": Milestone(
        name="Health Scare",
        stage=LifeStage.ESTABLISHED,
        description="First major health concern",
        typical_age_range=(40, 70),
        purpose_boost=0.1,  # Often creates clarity
        is_optional=True
    ),

    # Midlife milestones
    "career_peak": Milestone(
        name="Career Peak",
        stage=LifeStage.MIDLIFE,
        description="Reaching career zenith",
        typical_age_range=(45, 60),
        career_boost=0.15,
        purpose_boost=0.2,
        is_optional=True
    ),
    "grandchild": Milestone(
        name="First Grandchild",
        stage=LifeStage.MIDLIFE,
        description="Becoming a grandparent",
        typical_age_range=(50, 75),
        purpose_boost=0.2,
        social_boost=0.15,
        is_optional=True
    ),
    "major_career_change": Milestone(
        name="Major Career Change",
        stage=LifeStage.MIDLIFE,
        description="Reinventing professionally",
        typical_age_range=(45, 65),
        career_boost=-0.1,  # Short-term setback
        purpose_boost=0.15,  # Long-term meaning
        is_optional=True
    ),

    # Elder milestones
    "retirement": Milestone(
        name="Retirement",
        stage=LifeStage.ACTIVE_ELDER,
        description="Leaving the workforce",
        typical_age_range=(55, 75),
        career_boost=-0.3,
        purpose_boost=0.05  # Depends on person
    ),
    "50th_anniversary": Milestone(
        name="50th Wedding Anniversary",
        stage=LifeStage.ACTIVE_ELDER,
        description="Half century of marriage",
        typical_age_range=(70, 85),
        social_boost=0.2,
        purpose_boost=0.15,
        is_optional=True
    ),
    "losing_spouse": Milestone(
        name="Losing Spouse",
        stage=LifeStage.ACTIVE_ELDER,
        description="Death of life partner",
        typical_age_range=(65, 95),
        social_boost=-0.3,
        purpose_boost=-0.2,
        is_optional=True
    ),
    "great_grandchild": Milestone(
        name="Great-Grandchild",
        stage=LifeStage.LATE_ELDER,
        description="Meeting next generation",
        typical_age_range=(75, 100),
        purpose_boost=0.15,
        is_optional=True
    ),
    "assisted_living": Milestone(
        name="Moving to Assisted Living",
        stage=LifeStage.LATE_ELDER,
        description="Transitioning to care facility",
        typical_age_range=(75, 95),
        purpose_boost=-0.1,
        is_optional=True
    ),
    "100th_birthday": Milestone(
        name="100th Birthday",
        stage=LifeStage.LATE_ELDER,
        description="Centenarian celebration",
        typical_age_range=(100, 100),
        purpose_boost=0.3,
        social_boost=0.2,
        is_optional=True
    ),
}


# =============================================================================
# GENERATION CHARACTERISTICS
# =============================================================================

GENERATION_TRAITS: Dict[Generation, Dict[str, Any]] = {
    Generation.SILENT: {
        "birth_years": (1928, 1945),
        "description": "Traditional, loyal, disciplined",
        "work_ethic_modifier": 1.3,
        "technology_comfort": 0.3,
        "traditionalism": 0.9,
        "risk_tolerance": 0.4,
        "typical_values": ["duty", "loyalty", "respect", "conformity"]
    },
    Generation.BOOMER: {
        "birth_years": (1946, 1964),
        "description": "Optimistic, workaholic, competitive",
        "work_ethic_modifier": 1.2,
        "technology_comfort": 0.5,
        "traditionalism": 0.7,
        "risk_tolerance": 0.6,
        "typical_values": ["success", "achievement", "ownership", "status"]
    },
    Generation.GEN_X: {
        "birth_years": (1965, 1980),
        "description": "Independent, skeptical, adaptable",
        "work_ethic_modifier": 1.0,
        "technology_comfort": 0.7,
        "traditionalism": 0.5,
        "risk_tolerance": 0.6,
        "typical_values": ["independence", "work-life balance", "skepticism"]
    },
    Generation.MILLENNIAL: {
        "birth_years": (1981, 1996),
        "description": "Tech-savvy, collaborative, purpose-driven",
        "work_ethic_modifier": 0.9,
        "technology_comfort": 0.9,
        "traditionalism": 0.3,
        "risk_tolerance": 0.5,
        "typical_values": ["purpose", "experiences", "authenticity", "collaboration"]
    },
    Generation.GEN_Z: {
        "birth_years": (1997, 2012),
        "description": "Digital native, diverse, pragmatic",
        "work_ethic_modifier": 0.85,
        "technology_comfort": 1.0,
        "traditionalism": 0.2,
        "risk_tolerance": 0.4,
        "typical_values": ["authenticity", "diversity", "mental health", "pragmatism"]
    },
    Generation.ALPHA: {
        "birth_years": (2013, 2030),
        "description": "AI-native, global, TBD",
        "work_ethic_modifier": 1.0,  # Unknown
        "technology_comfort": 1.0,
        "traditionalism": 0.1,
        "risk_tolerance": 0.5,
        "typical_values": ["innovation", "global", "ai-integrated"]
    },
}


# =============================================================================
# LIFE STAGE MOODLETS
# =============================================================================

LIFE_STAGE_MOODLETS: Dict[str, Dict[str, Any]] = {
    # Childhood moodlets
    "innocent_joy": {
        "name": "Innocent Joy",
        "mood_impact": 15,
        "description": "The simple happiness of childhood",
        "stage": LifeStage.CHILDHOOD
    },
    "first_day_anxiety": {
        "name": "First Day Anxiety",
        "mood_impact": -10,
        "description": "Nervous about a new beginning",
        "stage": LifeStage.CHILDHOOD
    },

    # Adolescence moodlets
    "teen_angst": {
        "name": "Teen Angst",
        "mood_impact": -8,
        "description": "The weight of growing up",
        "stage": LifeStage.ADOLESCENCE
    },
    "independence_thrill": {
        "name": "Independence Thrill",
        "mood_impact": 12,
        "description": "Excitement of newfound freedom",
        "stage": LifeStage.ADOLESCENCE
    },

    # Emerging adult moodlets
    "quarter_life_confusion": {
        "name": "Quarter-Life Confusion",
        "mood_impact": -12,
        "description": "Uncertainty about life direction",
        "stage": LifeStage.EMERGING_ADULT
    },
    "world_is_open": {
        "name": "World Is Open",
        "mood_impact": 15,
        "description": "Feeling like anything is possible",
        "stage": LifeStage.EMERGING_ADULT
    },

    # Young adult moodlets
    "career_momentum": {
        "name": "Career Momentum",
        "mood_impact": 10,
        "description": "Building toward something",
        "stage": LifeStage.YOUNG_ADULT
    },
    "settling_down": {
        "name": "Settling Down",
        "mood_impact": 8,
        "description": "Contentment with life foundation",
        "stage": LifeStage.YOUNG_ADULT
    },
    "fomo": {
        "name": "Fear of Missing Out",
        "mood_impact": -8,
        "description": "Worried others are living better",
        "stage": LifeStage.YOUNG_ADULT
    },

    # Established adult moodlets
    "peak_performance": {
        "name": "Peak Performance",
        "mood_impact": 15,
        "description": "Operating at full capacity",
        "stage": LifeStage.ESTABLISHED
    },
    "empty_nest_blues": {
        "name": "Empty Nest Blues",
        "mood_impact": -15,
        "description": "Missing the children at home",
        "stage": LifeStage.ESTABLISHED
    },
    "sandwich_stress": {
        "name": "Sandwich Generation Stress",
        "mood_impact": -12,
        "description": "Caring for parents and children",
        "stage": LifeStage.ESTABLISHED
    },

    # Midlife moodlets
    "midlife_clarity": {
        "name": "Midlife Clarity",
        "mood_impact": 12,
        "description": "Knowing what truly matters",
        "stage": LifeStage.MIDLIFE
    },
    "midlife_crisis": {
        "name": "Midlife Crisis",
        "mood_impact": -20,
        "description": "Questioning everything",
        "stage": LifeStage.MIDLIFE
    },
    "second_wind": {
        "name": "Second Wind",
        "mood_impact": 15,
        "description": "Energy for a new chapter",
        "stage": LifeStage.MIDLIFE
    },

    # Elder moodlets
    "wisdom_of_age": {
        "name": "Wisdom of Age",
        "mood_impact": 10,
        "description": "Peace from understanding life",
        "stage": LifeStage.ACTIVE_ELDER
    },
    "legacy_building": {
        "name": "Legacy Building",
        "mood_impact": 12,
        "description": "Focused on what to leave behind",
        "stage": LifeStage.ACTIVE_ELDER
    },
    "mortality_awareness": {
        "name": "Mortality Awareness",
        "mood_impact": -10,
        "description": "Time is finite",
        "stage": LifeStage.ACTIVE_ELDER
    },

    # Late elder moodlets
    "life_review_peace": {
        "name": "Life Review Peace",
        "mood_impact": 15,
        "description": "Content with the life lived",
        "stage": LifeStage.LATE_ELDER
    },
    "life_review_regret": {
        "name": "Life Review Regret",
        "mood_impact": -20,
        "description": "Wishing things were different",
        "stage": LifeStage.LATE_ELDER
    },
    "gratitude": {
        "name": "Deep Gratitude",
        "mood_impact": 20,
        "description": "Appreciation for every day",
        "stage": LifeStage.LATE_ELDER
    },
}


# =============================================================================
# LIFE STAGE MANAGER
# =============================================================================

class LifeStageManager:
    """Manages life stages, branches, and milestones for all agents."""

    def __init__(
        self,
        moodlet_manager: Optional[Any] = None,
        wellness_manager: Optional[Any] = None,
        career_manager: Optional[Any] = None,
        seed: int = 42
    ):
        self.states: Dict[str, LifeStageState] = {}
        self.branches = LIFE_BRANCHES
        self.milestones = LIFE_MILESTONES
        self.moodlet_manager = moodlet_manager
        self.wellness_manager = wellness_manager
        self.career_manager = career_manager
        self.random = random.Random(seed)
        self.statistics = {
            "branch_transitions": 0,
            "milestones_completed": 0,
            "birthday_celebrations": 0,
        }

    def initialize_agent(
        self,
        agent_id: str,
        age: int,
        birth_year: Optional[int] = None
    ) -> LifeStageState:
        """Initialize life stage state for an agent."""
        stage = self._age_to_stage(age)
        generation = self._birth_year_to_generation(birth_year or (2025 - age))
        default_branch = self._get_default_branch(stage)

        state = LifeStageState(
            agent_id=agent_id,
            current_stage=stage,
            current_branch=default_branch,
            age=age,
            days_until_birthday=self.random.randint(1, 365),
            generation=generation
        )

        self.states[agent_id] = state
        return state

    def _age_to_stage(self, age: int) -> LifeStage:
        """Convert age to appropriate life stage."""
        if age < 13:
            return LifeStage.CHILDHOOD
        elif age < 18:
            return LifeStage.ADOLESCENCE
        elif age < 26:
            return LifeStage.EMERGING_ADULT
        elif age < 36:
            return LifeStage.YOUNG_ADULT
        elif age < 51:
            return LifeStage.ESTABLISHED
        elif age < 66:
            return LifeStage.MIDLIFE
        elif age < 81:
            return LifeStage.ACTIVE_ELDER
        else:
            return LifeStage.LATE_ELDER

    def _birth_year_to_generation(self, birth_year: int) -> Generation:
        """Determine generation from birth year."""
        for gen, traits in GENERATION_TRAITS.items():
            start, end = traits["birth_years"]
            if start <= birth_year <= end:
                return gen
        # Default for future births
        return Generation.ALPHA

    def _get_default_branch(self, stage: LifeStage) -> str:
        """Get default branch for a stage."""
        branches = self.branches.get(stage, {})
        if branches:
            return list(branches.keys())[0]
        return "default"

    def process_life_event(self, agent_id: str, event: str) -> Dict[str, Any]:
        """Process a life event that may trigger branch change or milestone."""
        if agent_id not in self.states:
            return {"changed": False}

        state = self.states[agent_id]
        result = {"changed": False, "branch_changed": False, "milestone": None}

        # Check for branch transition
        current_branches = self.branches.get(state.current_stage, {})
        for branch_name, branch in current_branches.items():
            if event in branch.triggers and branch_name != state.current_branch:
                self._transition_branch(agent_id, branch_name)
                result["branch_changed"] = True
                result["new_branch"] = branch_name
                result["changed"] = True
                break

        # Check if event matches a milestone
        for milestone_id, milestone in self.milestones.items():
            if event.lower() in milestone_id.lower():
                if milestone_id not in state.milestones_completed:
                    self._complete_milestone(agent_id, milestone_id)
                    result["milestone"] = milestone_id
                    result["changed"] = True

        return result

    def _transition_branch(self, agent_id: str, new_branch: str):
        """Transition agent to a new branch."""
        state = self.states[agent_id]

        # Record history
        state.previous_branches.append(
            (state.current_stage, state.current_branch)
        )

        state.current_branch = new_branch
        self.statistics["branch_transitions"] += 1

        # Apply moodlet if available
        if self.moodlet_manager:
            branch_data = self.branches[state.current_stage].get(new_branch)
            if branch_data:
                self.moodlet_manager.add_moodlet(
                    agent_id,
                    f"life_transition_{new_branch}",
                    f"Entering {branch_data.name} phase"
                )

    def _complete_milestone(self, agent_id: str, milestone_id: str):
        """Mark a milestone as completed."""
        state = self.states[agent_id]
        milestone = self.milestones[milestone_id]

        state.milestones_completed.append(milestone_id)
        self.statistics["milestones_completed"] += 1

        # Apply moodlet
        if self.moodlet_manager:
            self.moodlet_manager.add_moodlet(
                agent_id,
                f"milestone_{milestone_id}",
                f"Achieved: {milestone.name}"
            )

        # Apply wellness boosts if available
        if self.wellness_manager:
            if milestone.purpose_boost:
                self._boost_wellness(agent_id, "purpose", milestone.purpose_boost)

    def _boost_wellness(self, agent_id: str, pillar: str, amount: float):
        """Apply wellness boost if manager available."""
        if self.wellness_manager and hasattr(self.wellness_manager, 'wellness_states'):
            if agent_id in self.wellness_manager.wellness_states:
                current = getattr(
                    self.wellness_manager.wellness_states[agent_id],
                    pillar, 50.0
                )
                setattr(
                    self.wellness_manager.wellness_states[agent_id],
                    pillar,
                    min(100, current + amount * 100)
                )

    def daily_update(self, agent_id: str) -> Dict[str, Any]:
        """Process daily updates including birthday."""
        if agent_id not in self.states:
            return {}

        state = self.states[agent_id]
        result = {"events": []}

        state.days_until_birthday -= 1

        if state.days_until_birthday <= 0:
            # Birthday!
            state.age += 1
            state.days_until_birthday = 365
            self.statistics["birthday_celebrations"] += 1
            result["events"].append("birthday")

            # Check for stage transition
            new_stage = self._age_to_stage(state.age)
            if new_stage != state.current_stage:
                state.previous_branches.append(
                    (state.current_stage, state.current_branch)
                )
                state.current_stage = new_stage
                state.current_branch = self._get_default_branch(new_stage)
                result["events"].append("stage_transition")
                result["new_stage"] = new_stage.value

            # Check for missed milestones
            for m_id, milestone in self.milestones.items():
                if m_id not in state.milestones_completed:
                    if state.age > milestone.typical_age_range[1]:
                        if m_id not in state.milestones_missed:
                            state.milestones_missed.append(m_id)
                            if milestone.regret_if_missed > 0:
                                result["events"].append(f"missed_{m_id}")

        return result

    def get_branch_modifiers(self, agent_id: str) -> Dict[str, float]:
        """Get current branch modifiers for an agent."""
        if agent_id not in self.states:
            return {}

        state = self.states[agent_id]
        branches = self.branches.get(state.current_stage, {})
        branch = branches.get(state.current_branch)

        if not branch:
            return {}

        return {
            "energy": branch.energy_modifier,
            "health": branch.health_modifier,
            "social": branch.social_modifier,
            "career": branch.career_modifier,
            "purpose": branch.purpose_modifier,
        }

    def get_generation_traits(self, agent_id: str) -> Dict[str, Any]:
        """Get generational traits for an agent."""
        if agent_id not in self.states:
            return {}

        gen = self.states[agent_id].generation
        return GENERATION_TRAITS.get(gen, {})

    def format_life_panel(self, agent_id: str) -> str:
        """Format life stage info for display."""
        if agent_id not in self.states:
            return "Unknown agent"

        state = self.states[agent_id]
        branch_data = self.branches.get(state.current_stage, {}).get(state.current_branch)
        gen_data = GENERATION_TRAITS.get(state.generation, {})

        lines = [
            f"══════ LIFE STAGE ══════",
            f"Age: {state.age} ({state.generation.value.replace('_', ' ').title()})",
            f"Stage: {state.current_stage.value.replace('_', ' ').title()}",
            f"Branch: {branch_data.name if branch_data else state.current_branch}",
        ]

        if branch_data:
            lines.append(f"  → {branch_data.description}")

        milestones_done = len(state.milestones_completed)
        lines.append(f"Milestones: {milestones_done} completed")

        if state.spouse:
            lines.append(f"Spouse: {state.spouse}")
        if state.children:
            lines.append(f"Children: {len(state.children)}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stage_dist = {}
        generation_dist = {}
        branch_dist = {}

        for state in self.states.values():
            stage_dist[state.current_stage.value] = stage_dist.get(
                state.current_stage.value, 0
            ) + 1
            generation_dist[state.generation.value] = generation_dist.get(
                state.generation.value, 0
            ) + 1
            branch_dist[state.current_branch] = branch_dist.get(
                state.current_branch, 0
            ) + 1

        return {
            "total_agents": len(self.states),
            "stage_distribution": stage_dist,
            "generation_distribution": generation_dist,
            "branch_distribution": branch_dist,
            **self.statistics
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_life_stages():
    """Demonstrate life stages system."""
    print("=" * 60)
    print("NEUROHOOD v8.5: LIFE STAGES DEMO")
    print("=" * 60)

    manager = LifeStageManager()

    # Create agents of different ages
    agents = [
        ("child_001", 8, 2017),
        ("teen_001", 15, 2010),
        ("emerging_001", 22, 2003),
        ("young_001", 32, 1993),
        ("established_001", 45, 1980),
        ("midlife_001", 58, 1967),
        ("elder_001", 72, 1953),
        ("late_elder_001", 85, 1940),
    ]

    print("\n1. INITIALIZING AGENTS ACROSS LIFE STAGES")
    print("-" * 50)

    for agent_id, age, birth_year in agents:
        state = manager.initialize_agent(agent_id, age, birth_year)
        print(f"{agent_id}: {age}yo, {state.current_stage.value}, "
              f"Gen {state.generation.value}")

    print("\n2. LIFE EVENTS AND BRANCH TRANSITIONS")
    print("-" * 50)

    # Trigger life events
    events = [
        ("young_001", "career_goals"),
        ("established_001", "empty_nest"),
        ("midlife_001", "health_scare"),
        ("elder_001", "volunteering"),
    ]

    for agent_id, event in events:
        result = manager.process_life_event(agent_id, event)
        if result["branch_changed"]:
            print(f"{agent_id}: Event '{event}' → Branch: {result['new_branch']}")

    print("\n3. MILESTONE COMPLETION")
    print("-" * 50)

    milestones = [
        ("teen_001", "high_school_graduation"),
        ("emerging_001", "first_job"),
        ("young_001", "wedding"),
    ]

    for agent_id, event in milestones:
        result = manager.process_life_event(agent_id, event)
        if result["milestone"]:
            print(f"{agent_id}: Completed milestone '{result['milestone']}'")

    print("\n4. SAMPLE LIFE PANELS")
    print("-" * 50)

    for agent_id in ["young_001", "midlife_001"]:
        print(f"\n{manager.format_life_panel(agent_id)}")

    print("\n5. GENERATIONAL TRAITS")
    print("-" * 50)

    for gen, traits in list(GENERATION_TRAITS.items())[:3]:
        print(f"{gen.value}: {traits['description']}")
        print(f"  Work ethic: {traits['work_ethic_modifier']:.1f}x, "
              f"Tech: {traits['technology_comfort']:.1f}")

    print("\n6. STATISTICS")
    print("-" * 50)
    stats = manager.get_statistics()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Stage distribution: {stats['stage_distribution']}")
    print(f"Branch transitions: {stats['branch_transitions']}")
    print(f"Milestones completed: {stats['milestones_completed']}")

    print("\n" + "=" * 60)
    print("LIFE STAGES DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_life_stages()
