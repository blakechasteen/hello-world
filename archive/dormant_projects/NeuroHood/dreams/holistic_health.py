"""
v8.2 Holistic Health System for NeuroHood
==========================================

A comprehensive wellness system based on 8 research-backed pillars of health:
1. Exercise - Physical activity and fitness
2. Nutrition - Diet quality and eating habits
3. Gardening - Nature connection and outdoor activity
4. Self-Expression - Creative outlets and personal identity
5. Community - Neighborhood engagement and group participation
6. Close Friends - Deep social bonds (quality over quantity)
7. Deep Dialogue - Meaningful conversations and emotional processing
8. Purpose - Life goals, meaning, and contribution to others

Also includes chronic conditions that can emerge based on lifestyle patterns:
- Diabetes (Type 2)
- Heart Disease
- Cancer (probability-based)
- Arthritis (age-related)
- Depression (linked to isolation)
- Anxiety (linked to stress)

Author: Claude (NeuroHood v8.2)
Date: November 2025
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# WELLNESS PILLARS
# =============================================================================

@dataclass
class WellnessState:
    """
    Tracks the 8 wellness pillars for holistic health.

    Each pillar is scored 0-100:
    - 0-20: Critical (major health risk)
    - 21-40: Poor (health impact)
    - 41-60: Moderate (room for improvement)
    - 61-80: Good (healthy range)
    - 81-100: Excellent (optimal wellness)
    """
    # 8 Wellness Pillars
    exercise: float = 50.0
    nutrition: float = 60.0
    gardening: float = 20.0  # Low default, increases with activity
    self_expression: float = 50.0
    community: float = 40.0
    close_friends: float = 30.0  # Quality of close friendships
    deep_dialogue: float = 40.0
    purpose: float = 50.0

    # Tracking
    last_exercise_day: int = 0
    last_gardening_day: int = 0
    consecutive_sedentary_days: int = 0

    @property
    def physical_health(self) -> float:
        """Composite physical health score (0-100)."""
        return (
            self.exercise * 0.35 +
            self.nutrition * 0.35 +
            self.gardening * 0.15 +
            self.self_expression * 0.15
        )

    @property
    def mental_health(self) -> float:
        """Composite mental health score (0-100)."""
        return (
            self.community * 0.15 +
            self.close_friends * 0.25 +
            self.deep_dialogue * 0.25 +
            self.purpose * 0.25 +
            self.self_expression * 0.10
        )

    @property
    def overall_wellness(self) -> float:
        """Overall wellness score (0-100)."""
        return (self.physical_health + self.mental_health) / 2

    @property
    def wellness_status(self) -> str:
        """Human-readable wellness status."""
        score = self.overall_wellness
        if score >= 80:
            return "Thriving"
        elif score >= 65:
            return "Healthy"
        elif score >= 50:
            return "Moderate"
        elif score >= 35:
            return "Struggling"
        else:
            return "Critical"

    def get_pillar_status(self, pillar: str) -> str:
        """Get status for a specific pillar."""
        value = getattr(self, pillar, 50.0)
        if value >= 80:
            return "Excellent"
        elif value >= 60:
            return "Good"
        elif value >= 40:
            return "Moderate"
        elif value >= 20:
            return "Poor"
        else:
            return "Critical"

    def get_weakest_pillars(self, count: int = 3) -> List[Tuple[str, float]]:
        """Get the weakest wellness pillars."""
        pillars = [
            ("exercise", self.exercise),
            ("nutrition", self.nutrition),
            ("gardening", self.gardening),
            ("self_expression", self.self_expression),
            ("community", self.community),
            ("close_friends", self.close_friends),
            ("deep_dialogue", self.deep_dialogue),
            ("purpose", self.purpose),
        ]
        return sorted(pillars, key=lambda x: x[1])[:count]


# =============================================================================
# CHRONIC CONDITIONS
# =============================================================================

class ConditionType(Enum):
    """Types of chronic conditions."""
    DIABETES = "diabetes"
    HEART_DISEASE = "heart_disease"
    CANCER = "cancer"
    ARTHRITIS = "arthritis"
    DEPRESSION = "depression"
    ANXIETY = "anxiety"


class ConditionSeverity(Enum):
    """Severity levels for chronic conditions."""
    EARLY = "early"          # Just diagnosed, minimal impact
    MANAGED = "managed"      # Under control with treatment
    MODERATE = "moderate"    # Noticeable impact on daily life
    SEVERE = "severe"        # Significant limitations


@dataclass
class ChronicCondition:
    """A chronic health condition with progression."""
    condition_type: ConditionType
    severity: ConditionSeverity = ConditionSeverity.EARLY
    days_since_onset: int = 0
    is_managed: bool = False
    treatment_adherence: float = 0.0  # 0-1, how well they follow treatment

    # Impact on daily life
    energy_modifier: float = 1.0      # Multiplier for energy (0.5-1.0)
    mood_modifier: float = 0.0        # Mood penalty (-3 to 0)
    activity_restriction: float = 0.0  # Activity limitation (0-1)

    @property
    def display_name(self) -> str:
        """Human-readable condition name."""
        names = {
            ConditionType.DIABETES: "Type 2 Diabetes",
            ConditionType.HEART_DISEASE: "Heart Disease",
            ConditionType.CANCER: "Cancer",
            ConditionType.ARTHRITIS: "Arthritis",
            ConditionType.DEPRESSION: "Depression",
            ConditionType.ANXIETY: "Anxiety Disorder",
        }
        return names.get(self.condition_type, self.condition_type.value)

    def progress(self, wellness: WellnessState, days: int = 1):
        """Progress the condition over time."""
        self.days_since_onset += days

        # Update severity based on management and lifestyle
        if self.is_managed and self.treatment_adherence > 0.7:
            # Good management can improve or stabilize
            if self.severity == ConditionSeverity.SEVERE:
                if random.random() < 0.02 * self.treatment_adherence:
                    self.severity = ConditionSeverity.MODERATE
            elif self.severity == ConditionSeverity.MODERATE:
                if random.random() < 0.03 * self.treatment_adherence:
                    self.severity = ConditionSeverity.MANAGED
        else:
            # Poor management can worsen
            if self.severity == ConditionSeverity.EARLY:
                if random.random() < 0.01:
                    self.severity = ConditionSeverity.MODERATE
            elif self.severity == ConditionSeverity.MANAGED:
                if random.random() < 0.02:
                    self.severity = ConditionSeverity.MODERATE

        # Update impact based on severity
        self._update_impacts()

    def _update_impacts(self):
        """Update daily life impacts based on severity."""
        if self.severity == ConditionSeverity.EARLY:
            self.energy_modifier = 0.95
            self.mood_modifier = -0.5
            self.activity_restriction = 0.05
        elif self.severity == ConditionSeverity.MANAGED:
            self.energy_modifier = 0.90
            self.mood_modifier = -1.0
            self.activity_restriction = 0.1
        elif self.severity == ConditionSeverity.MODERATE:
            self.energy_modifier = 0.75
            self.mood_modifier = -2.0
            self.activity_restriction = 0.25
        elif self.severity == ConditionSeverity.SEVERE:
            self.energy_modifier = 0.5
            self.mood_modifier = -3.0
            self.activity_restriction = 0.5


# =============================================================================
# CONDITION RISK FACTORS
# =============================================================================

@dataclass
class ConditionRisk:
    """Risk factors for developing a chronic condition."""
    base_risk: float = 0.001  # Daily probability base

    # Lifestyle modifiers (multiply base risk)
    low_exercise_modifier: float = 2.0
    poor_nutrition_modifier: float = 1.8
    high_stress_modifier: float = 1.5
    isolation_modifier: float = 1.3
    low_purpose_modifier: float = 1.2

    # Age modifiers
    age_threshold: int = 40  # Age when risk increases
    age_multiplier: float = 1.5  # Risk increase per decade after threshold


CONDITION_RISK_PROFILES = {
    ConditionType.DIABETES: ConditionRisk(
        base_risk=0.0005,
        low_exercise_modifier=2.5,
        poor_nutrition_modifier=2.5,
        high_stress_modifier=1.5,
        age_threshold=40,
        age_multiplier=1.8,
    ),
    ConditionType.HEART_DISEASE: ConditionRisk(
        base_risk=0.0003,
        low_exercise_modifier=2.8,
        poor_nutrition_modifier=2.2,
        high_stress_modifier=2.0,
        isolation_modifier=1.5,
        age_threshold=45,
        age_multiplier=2.0,
    ),
    ConditionType.CANCER: ConditionRisk(
        base_risk=0.0002,
        low_exercise_modifier=1.3,
        poor_nutrition_modifier=1.4,
        high_stress_modifier=1.3,  # Immunosuppression
        age_threshold=50,
        age_multiplier=1.8,
    ),
    ConditionType.ARTHRITIS: ConditionRisk(
        base_risk=0.0004,
        low_exercise_modifier=1.5,  # Weak joints
        age_threshold=50,
        age_multiplier=2.5,
    ),
    ConditionType.DEPRESSION: ConditionRisk(
        base_risk=0.0008,
        low_exercise_modifier=1.5,
        isolation_modifier=2.5,
        low_purpose_modifier=2.5,
        high_stress_modifier=2.0,
        age_threshold=0,  # No age factor
        age_multiplier=1.0,
    ),
    ConditionType.ANXIETY: ConditionRisk(
        base_risk=0.001,
        isolation_modifier=2.0,
        low_purpose_modifier=1.8,
        high_stress_modifier=2.5,
        age_threshold=0,
        age_multiplier=1.0,
    ),
}


# =============================================================================
# WELLNESS ACTIVITIES
# =============================================================================

@dataclass
class WellnessActivity:
    """An activity that affects wellness pillars."""
    name: str
    description: str

    # Pillar impacts (positive values = improvement)
    exercise_impact: float = 0.0
    nutrition_impact: float = 0.0
    gardening_impact: float = 0.0
    self_expression_impact: float = 0.0
    community_impact: float = 0.0
    close_friends_impact: float = 0.0
    deep_dialogue_impact: float = 0.0
    purpose_impact: float = 0.0

    # Requirements
    energy_cost: float = 5.0
    min_duration_hours: float = 0.5

    # Moodlet granted (if any)
    moodlet: Optional[str] = None


# Default wellness activities
WELLNESS_ACTIVITIES = {
    # Exercise activities
    "morning_walk": WellnessActivity(
        name="Morning Walk",
        description="A refreshing walk to start the day",
        exercise_impact=2.0,
        gardening_impact=0.5,  # Outdoor time
        purpose_impact=0.5,
        energy_cost=3.0,
        moodlet="nice_walk",
    ),
    "gym_session": WellnessActivity(
        name="Gym Session",
        description="Workout at the gym",
        exercise_impact=5.0,
        self_expression_impact=1.0,
        energy_cost=15.0,
        moodlet="well_rested",  # Paradoxically energizing after rest
    ),
    "sports_game": WellnessActivity(
        name="Sports Game",
        description="Playing sports with others",
        exercise_impact=4.0,
        community_impact=2.0,
        energy_cost=12.0,
        moodlet="great_conversation",
    ),
    "yoga_session": WellnessActivity(
        name="Yoga Session",
        description="Mind-body practice",
        exercise_impact=2.0,
        self_expression_impact=2.0,
        purpose_impact=1.0,
        energy_cost=5.0,
    ),

    # Gardening activities
    "gardening_session": WellnessActivity(
        name="Gardening",
        description="Working in the garden",
        exercise_impact=2.0,
        gardening_impact=5.0,
        nutrition_impact=1.0,  # Home-grown food
        purpose_impact=2.0,
        energy_cost=8.0,
        moodlet="creative_flow",
    ),
    "community_garden": WellnessActivity(
        name="Community Garden",
        description="Working in the neighborhood garden",
        gardening_impact=4.0,
        community_impact=3.0,
        close_friends_impact=1.0,
        energy_cost=10.0,
        moodlet="made_new_friend",
    ),

    # Nutrition activities
    "cooking_meal": WellnessActivity(
        name="Cook a Healthy Meal",
        description="Preparing nutritious food",
        nutrition_impact=3.0,
        self_expression_impact=1.0,
        energy_cost=5.0,
    ),
    "social_meal": WellnessActivity(
        name="Shared Meal with Friends",
        description="Eating together with close friends",
        nutrition_impact=2.0,
        close_friends_impact=3.0,
        deep_dialogue_impact=2.0,
        energy_cost=3.0,
        moodlet="heartwarming_moment",
    ),

    # Self-expression activities
    "art_project": WellnessActivity(
        name="Art Project",
        description="Creative artistic work",
        self_expression_impact=5.0,
        purpose_impact=2.0,
        energy_cost=8.0,
        moodlet="creative_flow",
    ),
    "music_practice": WellnessActivity(
        name="Music Practice",
        description="Playing an instrument or singing",
        self_expression_impact=4.0,
        purpose_impact=1.0,
        energy_cost=6.0,
    ),
    "journaling": WellnessActivity(
        name="Journaling",
        description="Reflective writing",
        self_expression_impact=3.0,
        deep_dialogue_impact=2.0,  # Internal dialogue
        purpose_impact=2.0,
        energy_cost=2.0,
    ),

    # Community activities
    "community_event": WellnessActivity(
        name="Community Event",
        description="Participating in a neighborhood event",
        community_impact=5.0,
        close_friends_impact=2.0,
        energy_cost=8.0,
        moodlet="block_party_fun",
    ),
    "volunteer_work": WellnessActivity(
        name="Volunteer Work",
        description="Helping others in the community",
        community_impact=4.0,
        purpose_impact=4.0,
        energy_cost=10.0,
        moodlet="generous_glow",
    ),
    "group_class": WellnessActivity(
        name="Group Class",
        description="Taking a class with others",
        community_impact=3.0,
        self_expression_impact=2.0,
        energy_cost=6.0,
        moodlet="learned_something",
    ),

    # Close friends activities
    "coffee_with_friend": WellnessActivity(
        name="Coffee with Friend",
        description="One-on-one time with a close friend",
        close_friends_impact=4.0,
        deep_dialogue_impact=3.0,
        energy_cost=3.0,
        moodlet="great_conversation",
    ),
    "help_friend": WellnessActivity(
        name="Help a Friend",
        description="Supporting a friend in need",
        close_friends_impact=5.0,
        purpose_impact=3.0,
        energy_cost=8.0,
        moodlet="feeling_appreciated",
    ),

    # Deep dialogue activities
    "heart_to_heart": WellnessActivity(
        name="Heart-to-Heart Talk",
        description="Deep emotional conversation",
        deep_dialogue_impact=5.0,
        close_friends_impact=3.0,
        energy_cost=6.0,
        moodlet="heartwarming_moment",
    ),
    "therapy_session": WellnessActivity(
        name="Therapy Session",
        description="Professional mental health support",
        deep_dialogue_impact=4.0,
        self_expression_impact=2.0,
        purpose_impact=2.0,
        energy_cost=8.0,
    ),

    # Purpose activities
    "mentoring": WellnessActivity(
        name="Mentoring",
        description="Guiding and teaching others",
        purpose_impact=5.0,
        community_impact=2.0,
        close_friends_impact=2.0,
        energy_cost=8.0,
        moodlet="accomplished",
    ),
    "goal_work": WellnessActivity(
        name="Working on Life Goal",
        description="Making progress on personal aspirations",
        purpose_impact=4.0,
        self_expression_impact=2.0,
        energy_cost=10.0,
        moodlet="project_breakthrough",
    ),
}


# =============================================================================
# HEALTH MOODLETS
# =============================================================================

HEALTH_MOODLETS = {
    # Positive wellness moodlets
    "healthy_glow": {
        "name": "Healthy Glow",
        "description": "Feeling vibrant and energetic",
        "emotion": "energized",
        "intensity": 2,
        "duration_hours": 12,
    },
    "well_balanced": {
        "name": "Well Balanced",
        "description": "Mind and body in harmony",
        "emotion": "content",
        "intensity": 2,
        "duration_hours": 24,
    },
    "exercise_high": {
        "name": "Exercise High",
        "description": "Post-workout endorphin rush",
        "emotion": "energized",
        "intensity": 2,
        "duration_hours": 4,
    },
    "garden_peace": {
        "name": "Garden Peace",
        "description": "Connected to nature",
        "emotion": "peaceful",
        "intensity": 1,
        "duration_hours": 8,
    },
    "purposeful_day": {
        "name": "Purposeful Day",
        "description": "Feeling like I'm making a difference",
        "emotion": "fulfilled",
        "intensity": 2,
        "duration_hours": 12,
    },
    "deep_connection": {
        "name": "Deep Connection",
        "description": "Had a meaningful conversation",
        "emotion": "connected",
        "intensity": 2,
        "duration_hours": 24,
    },

    # Negative wellness moodlets
    "sedentary_slump": {
        "name": "Sedentary Slump",
        "description": "Haven't moved enough lately",
        "emotion": "sluggish",
        "intensity": -1,
        "duration_hours": 12,
    },
    "poor_nutrition": {
        "name": "Eating Poorly",
        "description": "Diet has been unhealthy",
        "emotion": "uncomfortable",
        "intensity": -1,
        "duration_hours": 8,
    },
    "disconnected": {
        "name": "Feeling Disconnected",
        "description": "Missing meaningful connections",
        "emotion": "lonely",
        "intensity": -2,
        "duration_hours": 24,
    },
    "lacking_purpose": {
        "name": "Lacking Purpose",
        "description": "What am I doing with my life?",
        "emotion": "lost",
        "intensity": -2,
        "duration_hours": 48,
    },
    "chronic_fatigue": {
        "name": "Chronic Fatigue",
        "description": "Constantly tired and drained",
        "emotion": "exhausted",
        "intensity": -2,
        "duration_hours": 24,
    },

    # Condition-specific moodlets
    "diabetes_fatigue": {
        "name": "Blood Sugar Crash",
        "description": "Feeling the effects of diabetes",
        "emotion": "weak",
        "intensity": -2,
        "duration_hours": 6,
    },
    "heart_palpitations": {
        "name": "Heart Concerns",
        "description": "Worried about heart health",
        "emotion": "anxious",
        "intensity": -2,
        "duration_hours": 12,
    },
    "joint_pain": {
        "name": "Aching Joints",
        "description": "Arthritis flare-up",
        "emotion": "uncomfortable",
        "intensity": -2,
        "duration_hours": 24,
    },
    "anxiety_spiral": {
        "name": "Anxiety Spiral",
        "description": "Can't stop worrying",
        "emotion": "anxious",
        "intensity": -3,
        "duration_hours": 12,
    },
    "depression_fog": {
        "name": "Depressive Fog",
        "description": "Everything feels grey",
        "emotion": "numb",
        "intensity": -3,
        "duration_hours": 24,
    },

    # Treatment/recovery moodlets
    "medication_relief": {
        "name": "Medication Working",
        "description": "Treatment is helping",
        "emotion": "relieved",
        "intensity": 1,
        "duration_hours": 12,
    },
    "health_checkup_anxiety": {
        "name": "Doctor's Appointment Anxiety",
        "description": "Worried about test results",
        "emotion": "anxious",
        "intensity": -1,
        "duration_hours": 6,
    },
    "health_good_news": {
        "name": "Good Health News",
        "description": "Doctor gave positive news",
        "emotion": "relieved",
        "intensity": 3,
        "duration_hours": 48,
    },
    "condition_managed": {
        "name": "Condition Under Control",
        "description": "Managing my health well",
        "emotion": "empowered",
        "intensity": 2,
        "duration_hours": 24,
    },
}


# =============================================================================
# HOLISTIC HEALTH MANAGER
# =============================================================================

class HolisticHealthManager:
    """
    Manages the holistic health system for all residents.

    Tracks 8 wellness pillars and chronic conditions for each agent,
    with daily updates based on activities and lifestyle patterns.
    """

    # Pillar decay rates per day (if no activity)
    PILLAR_DECAY = {
        "exercise": 2.0,  # Decays faster without activity
        "nutrition": 1.0,
        "gardening": 0.5,
        "self_expression": 1.5,
        "community": 1.0,
        "close_friends": 0.5,  # Slower decay (relationships persist)
        "deep_dialogue": 1.5,
        "purpose": 1.0,
    }

    # Maximum pillar change per day
    MAX_PILLAR_CHANGE = 5.0

    def __init__(self, moodlet_manager=None, economic_manager=None):
        """
        Initialize the health manager.

        Args:
            moodlet_manager: Optional MoodletManager for adding health moodlets
            economic_manager: Optional EconomicManager for linking to financial stress
        """
        self.wellness_states: Dict[str, WellnessState] = {}
        self.conditions: Dict[str, List[ChronicCondition]] = {}
        self.health_events: List[Dict[str, Any]] = []

        self.moodlet_manager = moodlet_manager
        self.economic_manager = economic_manager

        # Track statistics
        self.total_conditions_developed = 0
        self.total_activities_performed = 0

    def initialize_agent(
        self,
        agent_id: str,
        personality: Dict[str, float] = None,
        age: int = 30,
    ) -> WellnessState:
        """
        Initialize wellness state for an agent based on personality.

        Args:
            agent_id: Unique agent identifier
            personality: Big Five personality traits (optional)
            age: Agent's age

        Returns:
            Initialized WellnessState
        """
        personality = personality or {}

        # Base wellness with some variation
        wellness = WellnessState(
            exercise=random.uniform(40, 70),
            nutrition=random.uniform(50, 70),
            gardening=random.uniform(10, 40),
            self_expression=random.uniform(40, 70),
            community=random.uniform(30, 60),
            close_friends=random.uniform(20, 50),
            deep_dialogue=random.uniform(30, 60),
            purpose=random.uniform(40, 70),
        )

        # Modify based on personality (Big Five)
        if personality:
            # Extroverts have higher community/social scores
            extraversion = personality.get("extraversion", 0.5)
            wellness.community += (extraversion - 0.5) * 20
            wellness.close_friends += (extraversion - 0.5) * 10

            # Conscientious people have better exercise/nutrition
            conscientiousness = personality.get("conscientiousness", 0.5)
            wellness.exercise += (conscientiousness - 0.5) * 15
            wellness.nutrition += (conscientiousness - 0.5) * 15

            # Open people have higher self-expression
            openness = personality.get("openness", 0.5)
            wellness.self_expression += (openness - 0.5) * 20
            wellness.gardening += (openness - 0.5) * 10

            # Agreeable people have better relationships
            agreeableness = personality.get("agreeableness", 0.5)
            wellness.close_friends += (agreeableness - 0.5) * 15
            wellness.deep_dialogue += (agreeableness - 0.5) * 10

        # Clamp all values to 0-100
        for pillar in ["exercise", "nutrition", "gardening", "self_expression",
                       "community", "close_friends", "deep_dialogue", "purpose"]:
            current = getattr(wellness, pillar)
            setattr(wellness, pillar, max(0, min(100, current)))

        self.wellness_states[agent_id] = wellness
        self.conditions[agent_id] = []

        return wellness

    def get_wellness(self, agent_id: str) -> WellnessState:
        """Get wellness state for an agent."""
        if agent_id not in self.wellness_states:
            return self.initialize_agent(agent_id)
        return self.wellness_states[agent_id]

    def get_conditions(self, agent_id: str) -> List[ChronicCondition]:
        """Get chronic conditions for an agent."""
        return self.conditions.get(agent_id, [])

    def has_condition(self, agent_id: str, condition_type: ConditionType) -> bool:
        """Check if an agent has a specific condition."""
        for condition in self.get_conditions(agent_id):
            if condition.condition_type == condition_type:
                return True
        return False

    def perform_activity(
        self,
        agent_id: str,
        activity_id: str,
        current_day: int = 0,
    ) -> Dict[str, Any]:
        """
        Perform a wellness activity, updating pillars accordingly.

        Args:
            agent_id: Agent performing the activity
            activity_id: Activity identifier from WELLNESS_ACTIVITIES
            current_day: Current simulation day

        Returns:
            Dict with activity results and pillar changes
        """
        activity = WELLNESS_ACTIVITIES.get(activity_id)
        if not activity:
            return {"success": False, "error": f"Unknown activity: {activity_id}"}

        wellness = self.get_wellness(agent_id)

        # Track changes
        changes = {}

        # Apply pillar impacts
        for pillar, attr in [
            ("exercise", "exercise_impact"),
            ("nutrition", "nutrition_impact"),
            ("gardening", "gardening_impact"),
            ("self_expression", "self_expression_impact"),
            ("community", "community_impact"),
            ("close_friends", "close_friends_impact"),
            ("deep_dialogue", "deep_dialogue_impact"),
            ("purpose", "purpose_impact"),
        ]:
            impact = getattr(activity, attr, 0.0)
            if impact != 0:
                current = getattr(wellness, pillar)
                new_value = max(0, min(100, current + impact))
                setattr(wellness, pillar, new_value)
                changes[pillar] = impact

        # Update tracking
        if activity.exercise_impact > 0:
            wellness.last_exercise_day = current_day
            wellness.consecutive_sedentary_days = 0

        if activity.gardening_impact > 0:
            wellness.last_gardening_day = current_day

        self.total_activities_performed += 1

        # Add moodlet if applicable
        moodlet_added = None
        if activity.moodlet and self.moodlet_manager:
            self.moodlet_manager.add_moodlet(agent_id, activity.moodlet, activity.name)
            moodlet_added = activity.moodlet

        # Log event
        self.health_events.append({
            "type": "activity",
            "agent_id": agent_id,
            "activity": activity_id,
            "day": current_day,
            "changes": changes,
        })

        return {
            "success": True,
            "activity": activity.name,
            "changes": changes,
            "moodlet": moodlet_added,
            "energy_cost": activity.energy_cost,
        }

    def calculate_condition_risk(
        self,
        agent_id: str,
        condition_type: ConditionType,
        age: int = 30,
        stress_level: float = 0.5,
    ) -> float:
        """
        Calculate the daily probability of developing a condition.

        Args:
            agent_id: Agent to check
            condition_type: Type of condition
            age: Agent's age
            stress_level: Current stress level (0-1)

        Returns:
            Daily probability (0-1) of developing the condition
        """
        # Already has this condition?
        if self.has_condition(agent_id, condition_type):
            return 0.0

        risk_profile = CONDITION_RISK_PROFILES.get(condition_type)
        if not risk_profile:
            return 0.0

        wellness = self.get_wellness(agent_id)

        # Start with base risk
        risk = risk_profile.base_risk

        # Apply lifestyle modifiers
        if wellness.exercise < 40:
            risk *= risk_profile.low_exercise_modifier

        if wellness.nutrition < 40:
            risk *= risk_profile.poor_nutrition_modifier

        if stress_level > 0.6:
            risk *= risk_profile.high_stress_modifier

        if wellness.community < 30 and wellness.close_friends < 30:
            risk *= risk_profile.isolation_modifier

        if wellness.purpose < 30:
            risk *= risk_profile.low_purpose_modifier

        # Apply age modifier
        if age > risk_profile.age_threshold:
            decades_over = (age - risk_profile.age_threshold) / 10
            risk *= (risk_profile.age_multiplier ** decades_over)

        # Cap risk at 10% per day to avoid guaranteed conditions
        return min(risk, 0.1)

    def check_for_new_conditions(
        self,
        agent_id: str,
        age: int = 30,
        stress_level: float = 0.5,
        current_day: int = 0,
    ) -> Optional[ChronicCondition]:
        """
        Check if an agent develops a new chronic condition.

        Args:
            agent_id: Agent to check
            age: Agent's age
            stress_level: Current stress level
            current_day: Current simulation day

        Returns:
            New ChronicCondition if developed, None otherwise
        """
        for condition_type in ConditionType:
            risk = self.calculate_condition_risk(
                agent_id, condition_type, age, stress_level
            )

            if random.random() < risk:
                # Developed a new condition!
                condition = ChronicCondition(
                    condition_type=condition_type,
                    severity=ConditionSeverity.EARLY,
                    days_since_onset=0,
                )

                self.conditions[agent_id].append(condition)
                self.total_conditions_developed += 1

                # Log event
                self.health_events.append({
                    "type": "condition_onset",
                    "agent_id": agent_id,
                    "condition": condition_type.value,
                    "day": current_day,
                })

                # Add condition-specific moodlet
                self._add_condition_moodlet(agent_id, condition_type)

                return condition

        return None

    def _add_condition_moodlet(self, agent_id: str, condition_type: ConditionType):
        """Add appropriate moodlet when condition is diagnosed."""
        if not self.moodlet_manager:
            return

        moodlet_map = {
            ConditionType.DIABETES: "diabetes_fatigue",
            ConditionType.HEART_DISEASE: "heart_palpitations",
            ConditionType.ARTHRITIS: "joint_pain",
            ConditionType.DEPRESSION: "depression_fog",
            ConditionType.ANXIETY: "anxiety_spiral",
        }

        moodlet_id = moodlet_map.get(condition_type)
        if moodlet_id:
            self.moodlet_manager.add_moodlet(agent_id, moodlet_id, "health_condition")

    def daily_update(
        self,
        agent_id: str,
        current_day: int = 0,
        age: int = 30,
        stress_level: float = 0.5,
        activities_today: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform daily health update for an agent.

        Args:
            agent_id: Agent to update
            current_day: Current simulation day
            age: Agent's age
            stress_level: Current stress level (0-1)
            activities_today: List of activities performed today

        Returns:
            Dict with update results
        """
        wellness = self.get_wellness(agent_id)
        activities_today = activities_today or []

        results = {
            "pillar_changes": {},
            "conditions_progressed": [],
            "new_condition": None,
            "moodlets_added": [],
        }

        # Apply pillar decay for inactive pillars
        active_pillars = set()
        for activity_id in activities_today:
            activity = WELLNESS_ACTIVITIES.get(activity_id)
            if activity:
                for pillar in ["exercise", "nutrition", "gardening", "self_expression",
                              "community", "close_friends", "deep_dialogue", "purpose"]:
                    if getattr(activity, f"{pillar}_impact", 0) > 0:
                        active_pillars.add(pillar)

        for pillar, decay in self.PILLAR_DECAY.items():
            if pillar not in active_pillars:
                current = getattr(wellness, pillar)
                new_value = max(0, current - decay)
                setattr(wellness, pillar, new_value)
                results["pillar_changes"][pillar] = -decay

        # Check for sedentary days
        if "exercise" not in active_pillars:
            wellness.consecutive_sedentary_days += 1
            if wellness.consecutive_sedentary_days >= 3 and self.moodlet_manager:
                self.moodlet_manager.add_moodlet(agent_id, "sedentary_slump", "lifestyle")
                results["moodlets_added"].append("sedentary_slump")

        # Progress existing conditions
        for condition in self.get_conditions(agent_id):
            condition.progress(wellness)
            results["conditions_progressed"].append(condition.condition_type.value)

        # Check for new conditions (reduced probability)
        if random.random() < 0.1:  # Only check 10% of days
            new_condition = self.check_for_new_conditions(
                agent_id, age, stress_level, current_day
            )
            if new_condition:
                results["new_condition"] = new_condition.display_name

        # Add wellness-based moodlets
        self._add_wellness_moodlets(agent_id, wellness, results)

        return results

    def _add_wellness_moodlets(
        self,
        agent_id: str,
        wellness: WellnessState,
        results: Dict,
    ):
        """Add moodlets based on wellness state."""
        if not self.moodlet_manager:
            return

        # Check overall wellness
        if wellness.overall_wellness >= 80:
            self.moodlet_manager.add_moodlet(agent_id, "healthy_glow", "wellness")
            results["moodlets_added"].append("healthy_glow")

        # Check for poor wellness
        if wellness.deep_dialogue < 25 and wellness.close_friends < 25:
            self.moodlet_manager.add_moodlet(agent_id, "disconnected", "wellness")
            results["moodlets_added"].append("disconnected")

        if wellness.purpose < 25:
            self.moodlet_manager.add_moodlet(agent_id, "lacking_purpose", "wellness")
            results["moodlets_added"].append("lacking_purpose")

    def get_energy_modifier(self, agent_id: str) -> float:
        """
        Get combined energy modifier from all conditions.

        Returns:
            Multiplier for base energy (0.5-1.0)
        """
        modifier = 1.0
        for condition in self.get_conditions(agent_id):
            modifier *= condition.energy_modifier
        return max(0.5, modifier)

    def get_mood_modifier(self, agent_id: str) -> float:
        """
        Get combined mood modifier from all conditions.

        Returns:
            Mood penalty from conditions (-6 to 0)
        """
        modifier = 0.0
        for condition in self.get_conditions(agent_id):
            modifier += condition.mood_modifier
        return max(-6.0, modifier)

    def format_wellness_panel(self, agent_id: str, agent_name: str = None) -> str:
        """Format a wellness panel for display."""
        wellness = self.get_wellness(agent_id)
        conditions = self.get_conditions(agent_id)
        name = agent_name or agent_id

        lines = [
            f"╔══════════════════════════════════════╗",
            f"║  WELLNESS: {name:^24}  ║",
            f"╠══════════════════════════════════════╣",
        ]

        # Overall status
        status = wellness.wellness_status
        physical = wellness.physical_health
        mental = wellness.mental_health
        lines.append(f"║ Status: {status:<14} Overall: {wellness.overall_wellness:.0f}% ║")
        lines.append(f"║ Physical: {physical:.0f}%  Mental: {mental:.0f}%         ║")

        # Pillar bars
        lines.append(f"╟──────────────────────────────────────╢")
        lines.append(f"║ Wellness Pillars:                    ║")

        for pillar, value in [
            ("Exercise", wellness.exercise),
            ("Nutrition", wellness.nutrition),
            ("Gardening", wellness.gardening),
            ("Expression", wellness.self_expression),
            ("Community", wellness.community),
            ("Friends", wellness.close_friends),
            ("Dialogue", wellness.deep_dialogue),
            ("Purpose", wellness.purpose),
        ]:
            bar_len = 15
            filled = int((value / 100) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"║   {pillar:<10} [{bar}] {value:>3.0f}% ║")

        # Conditions
        if conditions:
            lines.append(f"╟──────────────────────────────────────╢")
            lines.append(f"║ Chronic Conditions:                  ║")
            for c in conditions:
                status = "Managed" if c.is_managed else c.severity.value.title()
                lines.append(f"║   • {c.display_name:<18} ({status:<8}) ║")

        lines.append(f"╚══════════════════════════════════════╝")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get health system statistics."""
        total_agents = len(self.wellness_states)

        if total_agents == 0:
            return {"total_agents": 0}

        # Calculate averages
        avg_physical = sum(w.physical_health for w in self.wellness_states.values()) / total_agents
        avg_mental = sum(w.mental_health for w in self.wellness_states.values()) / total_agents

        # Count conditions
        condition_counts = {ct: 0 for ct in ConditionType}
        for agent_conditions in self.conditions.values():
            for c in agent_conditions:
                condition_counts[c.condition_type] += 1

        return {
            "total_agents": total_agents,
            "average_physical_health": round(avg_physical, 1),
            "average_mental_health": round(avg_mental, 1),
            "total_conditions": sum(condition_counts.values()),
            "conditions_by_type": {ct.value: count for ct, count in condition_counts.items()},
            "total_activities_performed": self.total_activities_performed,
            "total_conditions_developed": self.total_conditions_developed,
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_holistic_health():
    """Demonstrate the holistic health system."""

    print("=" * 60)
    print("HOLISTIC HEALTH SYSTEM DEMO (v8.2)")
    print("=" * 60)

    manager = HolisticHealthManager()

    # === Demo 1: Initialize an agent ===
    print("\n" + "=" * 50)
    print("DEMO 1: INITIALIZE AGENT")
    print("=" * 50)

    personality = {
        "openness": 0.8,
        "conscientiousness": 0.6,
        "extraversion": 0.4,
        "agreeableness": 0.7,
        "neuroticism": 0.3,
    }

    wellness = manager.initialize_agent("maya_001", personality, age=35)
    print(manager.format_wellness_panel("maya_001", "Maya"))

    # === Demo 2: Perform activities ===
    print("\n" + "=" * 50)
    print("DEMO 2: PERFORM ACTIVITIES")
    print("=" * 50)

    activities = ["morning_walk", "gardening_session", "coffee_with_friend"]

    for activity_id in activities:
        result = manager.perform_activity("maya_001", activity_id, current_day=1)
        if result["success"]:
            print(f"\n{result['activity']}:")
            for pillar, change in result["changes"].items():
                print(f"  {pillar}: +{change:.1f}")
            if result.get("moodlet"):
                print(f"  Moodlet: {result['moodlet']}")

    print("\n" + manager.format_wellness_panel("maya_001", "Maya (After Activities)"))

    # === Demo 3: Condition risk calculation ===
    print("\n" + "=" * 50)
    print("DEMO 3: CONDITION RISK FACTORS")
    print("=" * 50)

    for condition_type in ConditionType:
        risk = manager.calculate_condition_risk("maya_001", condition_type, age=35, stress_level=0.3)
        print(f"  {condition_type.value}: {risk*100:.4f}% daily risk")

    # === Demo 4: Daily update ===
    print("\n" + "=" * 50)
    print("DEMO 4: DAILY UPDATE")
    print("=" * 50)

    update_result = manager.daily_update(
        "maya_001",
        current_day=2,
        age=35,
        stress_level=0.3,
        activities_today=["yoga_session", "journaling"],
    )

    print("Pillar changes from decay:")
    for pillar, change in update_result["pillar_changes"].items():
        print(f"  {pillar}: {change:.1f}")

    # === Demo 5: Multiple agents with conditions ===
    print("\n" + "=" * 50)
    print("DEMO 5: MULTIPLE AGENTS WITH CONDITIONS")
    print("=" * 50)

    # Create an older agent with poor wellness
    manager.initialize_agent("carlos_004", {"conscientiousness": 0.3}, age=55)
    carlos_wellness = manager.get_wellness("carlos_004")
    carlos_wellness.exercise = 25
    carlos_wellness.nutrition = 35
    carlos_wellness.community = 20

    # Force a condition for demo
    diabetes = ChronicCondition(
        condition_type=ConditionType.DIABETES,
        severity=ConditionSeverity.MODERATE,
        days_since_onset=180,
        is_managed=True,
        treatment_adherence=0.7,
    )
    manager.conditions["carlos_004"].append(diabetes)

    print(manager.format_wellness_panel("carlos_004", "Carlos"))

    # === Demo 6: Statistics ===
    print("\n" + "=" * 50)
    print("DEMO 6: HEALTH STATISTICS")
    print("=" * 50)

    stats = manager.get_statistics()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Average physical health: {stats['average_physical_health']}%")
    print(f"Average mental health: {stats['average_mental_health']}%")
    print(f"Total conditions: {stats['total_conditions']}")
    print(f"Activities performed: {stats['total_activities_performed']}")

    print("\n" + "=" * 60)
    print("HOLISTIC HEALTH DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_holistic_health())
