"""
Sims-Inspired Enhancements for NeuroHood
========================================

Lessons from 20+ years of The Sims franchise applied to our neighborhood simulation:

1. Moodlets - Temporary emotional states with expiration
2. Skills - Visible progression through practice
3. Relationship Milestones - Named relationship stages
4. Energy System - Fatigue affects decisions
5. Life Goals - Long-term aspirations with progress
6. Daily Summaries - "What happened today" narratives
7. Environmental Mood - Locations affect emotions

Author: Claude (NeuroHood Sims Integration)
Date: November 2025
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import uuid4

from solo_activities import SoloActivityEngine, SolitudeActivity


# =============================================================================
# MOODLET SYSTEM
# =============================================================================

class MoodletType(Enum):
    """Categories of moodlets."""
    SOCIAL = "social"
    CREATIVE = "creative"
    ENVIRONMENTAL = "environmental"
    PHYSICAL = "physical"
    ACHIEVEMENT = "achievement"
    RELATIONSHIP = "relationship"
    RANDOM = "random"


@dataclass
class Moodlet:
    """A temporary emotional modifier with expiration.

    Like Sims 4's moodlet system - events create temporary mood effects
    that stack and expire over time.
    """
    moodlet_id: str
    name: str
    description: str
    emotion: str  # happy, sad, tense, confident, inspired, embarrassed, playful, etc.
    intensity: float  # -3 to +3 (negative = bad, positive = good)
    duration_hours: float
    category: MoodletType
    source: str  # What caused this moodlet

    # Timing
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(hours=self.duration_hours)

    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expires_at

    @property
    def time_remaining(self) -> timedelta:
        remaining = self.expires_at - datetime.now()
        return max(remaining, timedelta(0))

    @property
    def icon(self) -> str:
        """Emoji icon based on emotion and intensity."""
        icons = {
            "happy": "😊" if self.intensity > 0 else "😔",
            "sad": "😢",
            "tense": "😰",
            "confident": "😎",
            "inspired": "✨",
            "embarrassed": "😳",
            "playful": "😄",
            "focused": "🎯",
            "energized": "⚡",
            "tired": "😴",
            "social": "💬",
            "romantic": "💕",
            "angry": "😠",
            "bored": "😑",
            "uncomfortable": "😣",
        }
        return icons.get(self.emotion, "💭")

    def __str__(self) -> str:
        sign = "+" if self.intensity > 0 else ""
        hours_left = self.time_remaining.total_seconds() / 3600
        return f"{self.icon} [{sign}{self.intensity:.0f}] {self.name} ({hours_left:.1f}h)"


class MoodletManager:
    """Manages moodlets for all residents."""

    # Predefined moodlet templates
    # v6.2: Positive durations DOUBLED/TRIPLED for better balance with negatives
    TEMPLATES = {
        # Social
        # v6.8: Keep positive at +2, strengthen negative to -3 for balance
        "great_conversation": Moodlet(
            moodlet_id="", name="Great Conversation",
            description="Had a really engaging chat",
            emotion="happy", intensity=2, duration_hours=6,  # v6.8: +2/6h (restored)
            category=MoodletType.SOCIAL, source=""
        ),
        # v6.8: Increased negative intensity for balance
        "awkward_silence": Moodlet(
            moodlet_id="", name="Awkward Silence",
            description="That conversation got weird...",
            emotion="embarrassed", intensity=-3, duration_hours=8,  # v6.8: -3/8h (stronger negative)
            category=MoodletType.SOCIAL, source=""
        ),
        "made_new_friend": Moodlet(
            moodlet_id="", name="New Friend!",
            description="Connected with someone new",
            emotion="happy", intensity=2, duration_hours=12,  # v6.6: +2/12h (middle ground)
            category=MoodletType.SOCIAL, source=""
        ),
        "feeling_lonely": Moodlet(
            moodlet_id="", name="Lonely",
            description="Haven't talked to anyone in a while",
            emotion="sad", intensity=-2, duration_hours=6,
            category=MoodletType.SOCIAL, source=""
        ),
        # v6.4: Additional negative social moodlets for balance
        "feeling_ignored": Moodlet(
            moodlet_id="", name="Feeling Ignored",
            description="No one seemed interested in what I had to say",
            emotion="sad", intensity=-2, duration_hours=6,
            category=MoodletType.SOCIAL, source=""
        ),
        "overstayed_welcome": Moodlet(
            moodlet_id="", name="Overstayed Welcome",
            description="Should have left earlier...",
            emotion="embarrassed", intensity=-1, duration_hours=4,
            category=MoodletType.SOCIAL, source=""
        ),
        "boring_conversation": Moodlet(
            moodlet_id="", name="Bored Stiff",
            description="That talk was mind-numbingly dull",
            emotion="bored", intensity=-1, duration_hours=4,
            category=MoodletType.SOCIAL, source=""
        ),

        # Creative - v6.6: balanced
        "creative_flow": Moodlet(
            moodlet_id="", name="In The Zone",
            description="Creative energy is flowing!",
            emotion="inspired", intensity=2, duration_hours=6,  # v6.6: +2/6h (middle ground)
            category=MoodletType.CREATIVE, source=""
        ),
        "project_breakthrough": Moodlet(
            moodlet_id="", name="Breakthrough!",
            description="Had a major insight on my project",
            emotion="confident", intensity=2, duration_hours=8,  # v6.6: +2/8h (middle ground)
            category=MoodletType.CREATIVE, source=""
        ),
        "creative_block": Moodlet(
            moodlet_id="", name="Blocked",
            description="Can't seem to make progress...",
            emotion="tense", intensity=-2, duration_hours=4,  # v6.3: restored intensity -2
            category=MoodletType.CREATIVE, source=""
        ),

        # Physical - v6.6: balanced
        "well_rested": Moodlet(
            moodlet_id="", name="Well Rested",
            description="Got a good night's sleep",
            emotion="energized", intensity=2, duration_hours=8,  # v6.6: +2/8h (middle ground)
            category=MoodletType.PHYSICAL, source=""
        ),
        "exhausted": Moodlet(
            moodlet_id="", name="Exhausted",
            description="Need to rest...",
            emotion="tired", intensity=-2, duration_hours=4,  # v6.2: intensity -2 (was -3)
            category=MoodletType.PHYSICAL, source=""
        ),
        "nice_walk": Moodlet(
            moodlet_id="", name="Refreshing Walk",
            description="That walk felt great",
            emotion="energized", intensity=1, duration_hours=4,  # v6.6: 4h (middle ground)
            category=MoodletType.PHYSICAL, source=""
        ),

        # Environmental - v6.6: balanced
        "cozy_atmosphere": Moodlet(
            moodlet_id="", name="Cozy Vibes",
            description="This place feels so comfortable",
            emotion="happy", intensity=1, duration_hours=3,  # v6.6: 3h (middle ground)
            category=MoodletType.ENVIRONMENTAL, source=""
        ),
        "beautiful_view": Moodlet(
            moodlet_id="", name="Beautiful View",
            description="What a lovely sight",
            emotion="inspired", intensity=2, duration_hours=3,
            category=MoodletType.ENVIRONMENTAL, source=""
        ),
        "crowded_space": Moodlet(
            moodlet_id="", name="Too Crowded",
            description="Too many people here...",
            emotion="uncomfortable", intensity=-1, duration_hours=1,
            category=MoodletType.ENVIRONMENTAL, source=""
        ),

        # Achievement
        "accomplished": Moodlet(
            moodlet_id="", name="Accomplished",
            description="Finished something important!",
            emotion="confident", intensity=3, duration_hours=8,
            category=MoodletType.ACHIEVEMENT, source=""
        ),
        "learned_something": Moodlet(
            moodlet_id="", name="Learned Something New",
            description="Knowledge gained!",
            emotion="inspired", intensity=2, duration_hours=4,
            category=MoodletType.ACHIEVEMENT, source=""
        ),

        # Relationship
        "heartwarming_moment": Moodlet(
            moodlet_id="", name="Heartwarming",
            description="Shared a special moment",
            emotion="happy", intensity=3, duration_hours=6,
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "argument": Moodlet(
            moodlet_id="", name="Had an Argument",
            description="Things got heated...",
            emotion="angry", intensity=-2, duration_hours=4,
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "feeling_appreciated": Moodlet(
            moodlet_id="", name="Feeling Appreciated",
            description="Someone recognized my efforts",
            emotion="confident", intensity=2, duration_hours=6,
            category=MoodletType.RELATIONSHIP, source=""
        ),

        # v6.3: BALANCED intensities (avg -2.5 to match positive avg +2.5)
        # Durations kept short (12-48h) but intensities restored
        "stressed": Moodlet(
            moodlet_id="", name="Stressed Out",
            description="Life feels overwhelming right now",
            emotion="anxious", intensity=-2, duration_hours=12,
            category=MoodletType.PHYSICAL, source=""
        ),
        "heartbroken": Moodlet(
            moodlet_id="", name="Heartbroken",
            description="The pain of lost love...",
            emotion="sad", intensity=-4, duration_hours=48,  # v6.3: intensity -4 (was -3)
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "betrayed": Moodlet(
            moodlet_id="", name="Betrayed",
            description="Someone I trusted hurt me",
            emotion="angry", intensity=-3, duration_hours=24,  # v6.3: intensity -3 (was -2)
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "jealous": Moodlet(
            moodlet_id="", name="Jealous",
            description="Can't stop thinking about them...",
            emotion="tense", intensity=-2, duration_hours=12,  # v6.3: intensity -2 (was -1)
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "scandalous": Moodlet(
            moodlet_id="", name="Caught in Scandal",
            description="Everyone's talking about me...",
            emotion="embarrassed", intensity=-3, duration_hours=24,  # v6.3: intensity -3 (was -2)
            category=MoodletType.SOCIAL, source=""
        ),

        # v6.3: Long-term negative states with balanced intensities
        "grieving": Moodlet(
            moodlet_id="", name="Grieving",
            description="Processing a deep loss...",
            emotion="sad", intensity=-3, duration_hours=48,  # v6.3: intensity -3 (was -2)
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "burnout": Moodlet(
            moodlet_id="", name="Burned Out",
            description="Completely exhausted by life",
            emotion="tired", intensity=-2, duration_hours=24,  # v6.3: intensity -2 (was -1)
            category=MoodletType.PHYSICAL, source=""
        ),
        "resentful": Moodlet(
            moodlet_id="", name="Deep Resentment",
            description="Can't forgive what happened",
            emotion="angry", intensity=-2, duration_hours=36,  # v6.3: intensity -2 (was -1)
            category=MoodletType.RELATIONSHIP, source=""
        ),
        "humiliated": Moodlet(
            moodlet_id="", name="Humiliated",
            description="Still can't show my face...",
            emotion="embarrassed", intensity=-2, duration_hours=24,  # v6.3: intensity -2 (was -1)
            category=MoodletType.SOCIAL, source=""
        ),
        "anxious_spiral": Moodlet(
            moodlet_id="", name="Anxiety Spiral",
            description="Everything feels like it's falling apart",
            emotion="anxious", intensity=-3, duration_hours=24,  # v6.3: intensity -3 (was -2)
            category=MoodletType.PHYSICAL, source=""
        ),
    }

    def __init__(self):
        self.moodlets: Dict[str, List[Moodlet]] = {}  # agent_id -> moodlets

    # v6.2: EQUAL caps for balanced mood distribution
    MAX_POSITIVE_MOODLETS = 5   # v6.2: Increased from 3 - allow more happiness
    MAX_NEGATIVE_MOODLETS = 5   # v6.2: Decreased from 10 - prevent sadness pile-up

    def add_moodlet(self, agent_id: str, template_name: str, source: str = "") -> Optional[Moodlet]:
        """Add a moodlet from template.

        v6 Enhancement: Caps positive moodlets at 3 to prevent happiness saturation.
        If already at cap, oldest positive moodlet is removed.
        Negative moodlets can stack more freely (up to 10).
        """
        if agent_id not in self.moodlets:
            self.moodlets[agent_id] = []

        template = self.TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"Unknown moodlet template: {template_name}")

        # v6: Check moodlet caps
        active = self.get_active_moodlets(agent_id)
        positive_moodlets = [m for m in active if m.intensity > 0]
        negative_moodlets = [m for m in active if m.intensity < 0]

        # v6: If adding positive and at cap, remove oldest positive
        if template.intensity > 0 and len(positive_moodlets) >= self.MAX_POSITIVE_MOODLETS:
            # Remove oldest positive moodlet
            oldest_positive = min(positive_moodlets, key=lambda m: m.created_at)
            self.moodlets[agent_id].remove(oldest_positive)

        # v6: If adding negative and at cap, remove oldest negative
        if template.intensity < 0 and len(negative_moodlets) >= self.MAX_NEGATIVE_MOODLETS:
            oldest_negative = min(negative_moodlets, key=lambda m: m.created_at)
            self.moodlets[agent_id].remove(oldest_negative)

        # Create new instance from template
        moodlet = Moodlet(
            moodlet_id=str(uuid4())[:8],
            name=template.name,
            description=template.description,
            emotion=template.emotion,
            intensity=template.intensity,
            duration_hours=template.duration_hours,
            category=template.category,
            source=source or template_name
        )

        self.moodlets[agent_id].append(moodlet)
        return moodlet

    def add_custom_moodlet(self, agent_id: str, moodlet: Moodlet) -> Moodlet:
        """Add a custom moodlet."""
        if agent_id not in self.moodlets:
            self.moodlets[agent_id] = []

        moodlet.moodlet_id = str(uuid4())[:8]
        self.moodlets[agent_id].append(moodlet)
        return moodlet

    def get_active_moodlets(self, agent_id: str) -> List[Moodlet]:
        """Get all non-expired moodlets for an agent."""
        if agent_id not in self.moodlets:
            return []

        # Filter out expired and clean up
        active = [m for m in self.moodlets[agent_id] if not m.is_expired]
        self.moodlets[agent_id] = active
        return active

    def get_mood_score(self, agent_id: str) -> float:
        """Get net mood score from all active moodlets."""
        active = self.get_active_moodlets(agent_id)
        return sum(m.intensity for m in active)

    def get_dominant_emotion(self, agent_id: str) -> Optional[str]:
        """Get the strongest current emotion."""
        active = self.get_active_moodlets(agent_id)
        if not active:
            return None

        # Find moodlet with highest absolute intensity
        strongest = max(active, key=lambda m: abs(m.intensity))
        return strongest.emotion

    def format_moodlets(self, agent_id: str) -> str:
        """Format moodlets for display."""
        active = self.get_active_moodlets(agent_id)
        if not active:
            return "  (No active moodlets)"

        return "\n".join(f"  {m}" for m in active)


# =============================================================================
# SKILL SYSTEM
# =============================================================================

class SkillCategory(Enum):
    """Categories of skills."""
    CREATIVE = "creative"
    SOCIAL = "social"
    MENTAL = "mental"
    PHYSICAL = "physical"
    PRACTICAL = "practical"


@dataclass
class Skill:
    """A learnable skill with progression."""
    name: str
    category: SkillCategory
    level: float = 0.0  # 0.0 to 10.0
    experience: float = 0.0  # XP toward next level

    # XP required per level (increases each level)
    XP_PER_LEVEL = [100, 150, 225, 340, 510, 765, 1150, 1725, 2590, 3885]

    @property
    def level_int(self) -> int:
        return int(self.level)

    @property
    def progress_to_next(self) -> float:
        """Progress toward next level (0.0-1.0)."""
        if self.level_int >= 10:
            return 1.0
        required = self.XP_PER_LEVEL[self.level_int]
        return self.experience / required

    def add_experience(self, xp: float) -> Tuple[float, bool]:
        """Add XP, returns (new_level, leveled_up)."""
        if self.level_int >= 10:
            return self.level, False

        self.experience += xp
        required = self.XP_PER_LEVEL[self.level_int]

        leveled_up = False
        while self.experience >= required and self.level_int < 10:
            self.experience -= required
            self.level = min(10.0, self.level + 1.0)
            leveled_up = True
            if self.level_int < 10:
                required = self.XP_PER_LEVEL[self.level_int]

        return self.level, leveled_up

    def __str__(self) -> str:
        bar_filled = int(self.progress_to_next * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        return f"{self.name}: Lv.{self.level_int} [{bar}]"


# Default skills for each resident based on their personality
DEFAULT_SKILLS = {
    "maya_001": {
        "Painting": (SkillCategory.CREATIVE, 6.0),
        "Creativity": (SkillCategory.CREATIVE, 7.0),
        "Gardening": (SkillCategory.PRACTICAL, 4.0),
        "Meditation": (SkillCategory.MENTAL, 3.0),
    },
    "jorge_002": {
        "Cooking": (SkillCategory.PRACTICAL, 7.0),
        "Charisma": (SkillCategory.SOCIAL, 5.0),
        "Fitness": (SkillCategory.PHYSICAL, 4.0),
        "Writing": (SkillCategory.CREATIVE, 3.0),
    },
    "elena_003": {
        "Writing": (SkillCategory.CREATIVE, 8.0),
        "Research": (SkillCategory.MENTAL, 6.0),
        "Logic": (SkillCategory.MENTAL, 5.0),
        "Charisma": (SkillCategory.SOCIAL, 3.0),
    },
    "carlos_004": {
        "Handiness": (SkillCategory.PRACTICAL, 6.0),
        "Fitness": (SkillCategory.PHYSICAL, 5.0),
        "Logic": (SkillCategory.MENTAL, 4.0),
        "Gardening": (SkillCategory.PRACTICAL, 5.0),
    },
    "sofia_005": {
        "Cooking": (SkillCategory.PRACTICAL, 5.0),
        "Charisma": (SkillCategory.SOCIAL, 7.0),
        "Baking": (SkillCategory.PRACTICAL, 6.0),
        "Gardening": (SkillCategory.PRACTICAL, 4.0),
    },
    "diego_006": {
        "Guitar": (SkillCategory.CREATIVE, 6.0),
        "Singing": (SkillCategory.CREATIVE, 5.0),
        "Charisma": (SkillCategory.SOCIAL, 6.0),
        "Writing": (SkillCategory.CREATIVE, 4.0),
    },
}


class SkillManager:
    """Manages skills for all residents."""

    def __init__(self):
        self.skills: Dict[str, Dict[str, Skill]] = {}  # agent_id -> {skill_name -> Skill}

    def initialize_agent(self, agent_id: str):
        """Initialize skills for an agent from defaults."""
        self.skills[agent_id] = {}

        defaults = DEFAULT_SKILLS.get(agent_id, {})
        for skill_name, (category, level) in defaults.items():
            self.skills[agent_id][skill_name] = Skill(
                name=skill_name,
                category=category,
                level=level
            )

    def get_skill(self, agent_id: str, skill_name: str) -> Optional[Skill]:
        """Get a specific skill."""
        if agent_id not in self.skills:
            return None
        return self.skills[agent_id].get(skill_name)

    def add_experience(self, agent_id: str, skill_name: str, xp: float) -> Tuple[float, bool]:
        """Add XP to a skill, returns (new_level, leveled_up)."""
        if agent_id not in self.skills:
            self.initialize_agent(agent_id)

        if skill_name not in self.skills[agent_id]:
            # Create new skill at level 0
            self.skills[agent_id][skill_name] = Skill(
                name=skill_name,
                category=SkillCategory.PRACTICAL,
                level=0.0
            )

        return self.skills[agent_id][skill_name].add_experience(xp)

    def get_all_skills(self, agent_id: str) -> List[Skill]:
        """Get all skills for an agent."""
        if agent_id not in self.skills:
            return []
        return list(self.skills[agent_id].values())

    def format_skills(self, agent_id: str) -> str:
        """Format skills for display."""
        skills = self.get_all_skills(agent_id)
        if not skills:
            return "  (No skills)"

        # Sort by level descending
        skills.sort(key=lambda s: s.level, reverse=True)
        return "\n".join(f"  {s}" for s in skills)


# =============================================================================
# RELATIONSHIP MILESTONES
# =============================================================================

class RelationshipStage(Enum):
    """Named relationship stages (like Sims)."""
    STRANGER = ("Stranger", 0)
    ACQUAINTANCE = ("Acquaintance", 20)
    COLLEAGUE = ("Colleague", 35)
    FRIEND = ("Friend", 50)
    GOOD_FRIEND = ("Good Friend", 65)
    BEST_FRIEND = ("Best Friend", 80)
    SOULMATE = ("Soulmate", 95)

    def __init__(self, display_name: str, threshold: int):
        self.display_name = display_name
        self.threshold = threshold


class RelationshipMilestones:
    """Track relationship progression with milestones."""

    def __init__(self):
        self.milestones: Dict[Tuple[str, str], Set[str]] = {}  # (a, b) -> achieved milestones
        self.previous_stages: Dict[Tuple[str, str], RelationshipStage] = {}

    def get_stage(self, relationship_score: float) -> RelationshipStage:
        """Get relationship stage from score (0-100)."""
        score = relationship_score * 100  # Convert to 0-100

        # Find highest matching stage
        current_stage = RelationshipStage.STRANGER
        for stage in RelationshipStage:
            if score >= stage.threshold:
                current_stage = stage

        return current_stage

    def check_milestone(
        self,
        agent1_id: str,
        agent2_id: str,
        old_score: float,
        new_score: float
    ) -> Optional[str]:
        """Check if a relationship milestone was reached."""
        key = tuple(sorted([agent1_id, agent2_id]))

        old_stage = self.get_stage(old_score)
        new_stage = self.get_stage(new_score)

        if new_stage.threshold > old_stage.threshold:
            # Leveled up!
            milestone = f"became_{new_stage.name.lower()}"

            if key not in self.milestones:
                self.milestones[key] = set()

            if milestone not in self.milestones[key]:
                self.milestones[key].add(milestone)
                self.previous_stages[key] = new_stage
                return new_stage.display_name

        return None


# =============================================================================
# ENERGY SYSTEM
# =============================================================================

@dataclass
class EnergyState:
    """Track energy/fatigue for an agent."""
    current: float = 100.0  # 0-100
    max_energy: float = 100.0

    # Decay rates
    base_decay_per_hour: float = 4.0  # Lose 4 energy per hour
    activity_costs: Dict[str, float] = field(default_factory=lambda: {
        "conversation": 5.0,
        "project_work": 10.0,
        "walking": 3.0,
        "exercise": 15.0,
        "creative_work": 8.0,
        "socializing": 6.0,
    })

    @property
    def percentage(self) -> float:
        return (self.current / self.max_energy) * 100

    @property
    def status(self) -> str:
        if self.current >= 80:
            return "Energized"
        elif self.current >= 60:
            return "Fine"
        elif self.current >= 40:
            return "Tired"
        elif self.current >= 20:
            return "Exhausted"
        else:
            return "Drained"

    @property
    def affects_decisions(self) -> bool:
        """Does low energy affect decision quality?"""
        return self.current < 40

    def drain(self, activity: str) -> float:
        """Drain energy for an activity, return amount drained."""
        cost = self.activity_costs.get(activity, 5.0)
        actual = min(cost, self.current)
        self.current = max(0, self.current - cost)
        return actual

    def rest(self, hours: float) -> float:
        """Restore energy from rest."""
        # Sleeping restores ~30 energy per hour
        restore = hours * 30
        actual = min(restore, self.max_energy - self.current)
        self.current = min(self.max_energy, self.current + restore)
        return actual

    def time_decay(self, hours: float):
        """Apply passive energy decay over time."""
        decay = hours * self.base_decay_per_hour
        self.current = max(0, self.current - decay)


class EnergyManager:
    """Manage energy for all residents."""

    def __init__(self):
        self.energy: Dict[str, EnergyState] = {}

    def initialize_agent(self, agent_id: str, starting_energy: float = 80.0):
        """Initialize energy for an agent."""
        self.energy[agent_id] = EnergyState(current=starting_energy)

    def get_energy(self, agent_id: str) -> EnergyState:
        if agent_id not in self.energy:
            self.initialize_agent(agent_id)
        return self.energy[agent_id]

    def drain(self, agent_id: str, activity: str) -> float:
        return self.get_energy(agent_id).drain(activity)

    def rest(self, agent_id: str, hours: float) -> float:
        return self.get_energy(agent_id).rest(hours)

    def apply_time_decay(self, hours: float):
        """Apply decay to all agents."""
        for state in self.energy.values():
            state.time_decay(hours)


# =============================================================================
# LIFE GOALS / ASPIRATIONS
# =============================================================================

@dataclass
class LifeGoal:
    """A long-term aspiration with milestones."""
    goal_id: str
    name: str
    description: str
    milestones: List[Tuple[str, bool]]  # (milestone_name, completed)
    reward_trait: Optional[str] = None  # Trait gained on completion

    @property
    def progress(self) -> float:
        """Progress as 0.0-1.0."""
        if not self.milestones:
            return 0.0
        completed = sum(1 for _, done in self.milestones if done)
        return completed / len(self.milestones)

    @property
    def is_complete(self) -> bool:
        return all(done for _, done in self.milestones)

    @property
    def current_milestone(self) -> Optional[str]:
        """Get the next incomplete milestone."""
        for name, done in self.milestones:
            if not done:
                return name
        return None

    def complete_milestone(self, milestone_name: str) -> bool:
        """Mark a milestone as complete."""
        for i, (name, done) in enumerate(self.milestones):
            if name == milestone_name and not done:
                self.milestones[i] = (name, True)
                return True
        return False


# Default life goals for each resident
DEFAULT_LIFE_GOALS = {
    "maya_001": LifeGoal(
        goal_id="artistic_prodigy",
        name="Artistic Prodigy",
        description="Become a master painter and inspire others",
        milestones=[
            ("Complete 3 paintings", False),
            ("Reach Painting level 7", False),
            ("Have artwork displayed publicly", False),
            ("Mentor another artist", False),
            ("Reach Painting level 10", False),
        ],
        reward_trait="Virtuoso"
    ),
    "jorge_002": LifeGoal(
        goal_id="master_chef",
        name="Master Chef",
        description="Create the perfect dish and open a restaurant",
        milestones=[
            ("Cook 10 different recipes", False),
            ("Reach Cooking level 7", False),
            ("Create a signature dish", False),
            ("Cater a neighborhood event", False),
            ("Publish a cookbook", False),
        ],
        reward_trait="Culinary Genius"
    ),
    "elena_003": LifeGoal(
        goal_id="bestselling_author",
        name="Bestselling Author",
        description="Write a book that touches hearts",
        milestones=[
            ("Write 5 journal entries", False),
            ("Reach Writing level 7", False),
            ("Complete a short story", False),
            ("Get published", False),
            ("Write a novel", False),
        ],
        reward_trait="Wordsmith"
    ),
    "sofia_005": LifeGoal(
        goal_id="friend_of_the_world",
        name="Friend of the World",
        description="Make deep connections with everyone",
        milestones=[
            ("Become friends with 3 people", False),
            ("Reach Charisma level 5", False),
            ("Become good friends with 3 people", False),
            ("Host a neighborhood gathering", False),
            ("Become best friends with someone", False),
        ],
        reward_trait="Beloved"
    ),
    "diego_006": LifeGoal(
        goal_id="musical_genius",
        name="Musical Genius",
        description="Create music that moves souls",
        milestones=[
            ("Practice guitar for 10 hours", False),
            ("Reach Guitar level 7", False),
            ("Perform for the neighborhood", False),
            ("Write an original song", False),
            ("Form a band", False),
        ],
        reward_trait="Maestro"
    ),
}


# =============================================================================
# DAILY SUMMARY GENERATOR
# =============================================================================

class DailySummaryGenerator:
    """Generate end-of-day narratives for residents."""

    def __init__(self, ollama_client=None):
        self.ollama = ollama_client

    async def generate_summary(
        self,
        agent_name: str,
        agent_personality: Dict,
        memories_today: List[Dict],
        moodlets: List[Moodlet],
        energy_level: float
    ) -> str:
        """Generate a narrative summary of the day."""

        # Build context
        events = [m.get("description", "") for m in memories_today[:10]]
        moods = [f"{m.name} ({m.emotion})" for m in moodlets]

        if self.ollama:
            prompt = f"""Write a brief end-of-day reflection for {agent_name}, a person who is {self._describe_personality(agent_personality)}.

Today's events:
{chr(10).join(f"- {e}" for e in events) if events else "- A quiet day"}

Current moods: {", ".join(moods) if moods else "Neutral"}
Energy level: {energy_level:.0f}%

Write 2-3 sentences in first person, reflecting on the day. Be authentic to their personality."""

            try:
                response = await self.ollama.generate(prompt, max_tokens=100)
                return response.strip()
            except Exception:
                pass

        # Fallback template-based summary
        return self._template_summary(agent_name, events, moodlets, energy_level)

    def _describe_personality(self, p: Dict) -> str:
        traits = []
        if p.get("openness", 0.5) > 0.6:
            traits.append("creative")
        if p.get("extraversion", 0.5) > 0.6:
            traits.append("outgoing")
        elif p.get("extraversion", 0.5) < 0.4:
            traits.append("introverted")
        if p.get("agreeableness", 0.5) > 0.6:
            traits.append("friendly")
        return ", ".join(traits) if traits else "balanced"

    def _template_summary(
        self,
        name: str,
        events: List[str],
        moodlets: List[Moodlet],
        energy: float
    ) -> str:
        """Template-based fallback."""
        if not events:
            return f"{name} had a quiet, uneventful day."

        # Pick dominant mood
        positive = sum(1 for m in moodlets if m.intensity > 0)
        negative = sum(1 for m in moodlets if m.intensity < 0)

        if positive > negative:
            mood = "good"
        elif negative > positive:
            mood = "challenging"
        else:
            mood = "balanced"

        # Energy commentary
        if energy < 30:
            energy_note = f" Feeling exhausted by evening."
        elif energy > 70:
            energy_note = f" Still had energy to spare."
        else:
            energy_note = ""

        return f"{name} had a {mood} day. {events[0]}{energy_note}"


# =============================================================================
# SIMS-ENHANCED ENGINE
# =============================================================================

class SimsEnhancedEngine(SoloActivityEngine):
    """
    Full Sims-inspired neighborhood engine.

    Adds:
    - Moodlet system
    - Skill progression
    - Relationship milestones
    - Energy management
    - Life goals
    - Daily summaries
    """

    def __init__(self):
        super().__init__()

        # New Sims-inspired systems
        self.moodlets = MoodletManager()
        self.skills = SkillManager()
        self.relationships = RelationshipMilestones()
        self.energy = EnergyManager()
        self.life_goals: Dict[str, LifeGoal] = {}
        self.summary_generator = None

        # Track simulation day
        self.current_day: int = 1
        self.day_start_hour: int = 6

    async def initialize(self, load_path=None):
        """Initialize with all Sims-inspired systems."""
        await super().initialize(load_path)

        # Initialize systems for each agent
        for agent_id in self.state.agents:
            self.skills.initialize_agent(agent_id)
            self.energy.initialize_agent(agent_id, random.uniform(60, 90))

            # Assign life goal
            if agent_id in DEFAULT_LIFE_GOALS:
                self.life_goals[agent_id] = DEFAULT_LIFE_GOALS[agent_id]

        # Create summary generator with Ollama if available
        if hasattr(self, 'ollama_client') and self.ollama_client:
            self.summary_generator = DailySummaryGenerator(self.ollama_client)
        else:
            self.summary_generator = DailySummaryGenerator()

        print(f"[Sims] Initialized Sims-inspired systems")
        print(f"  - Moodlets: {len(MoodletManager.TEMPLATES)} templates")
        print(f"  - Skills: {sum(len(s) for s in DEFAULT_SKILLS.values())} default skills")
        print(f"  - Life Goals: {len(DEFAULT_LIFE_GOALS)} aspirations")

    # =========================================================================
    # OVERRIDE CORE METHODS TO ADD SIMS FEATURES
    # =========================================================================

    async def user_talk_to(self, agent_id: str, initial_message: str):
        """Enhanced conversation with moodlets and skill gains."""
        result = await super().user_talk_to(agent_id, initial_message)

        # Drain energy
        self.energy.drain(agent_id, "conversation")

        # Add moodlet based on conversation quality
        if result and len(result.lines) > 4:
            self.moodlets.add_moodlet(agent_id, "great_conversation", f"chat_with_user")
            # Small charisma XP
            new_level, leveled = self.skills.add_experience(agent_id, "Charisma", 15)
            if leveled:
                print(f"  🎯 {self.state.agents[agent_id].name} leveled up Charisma to {int(new_level)}!")

        return result

    async def do_project_work(self, agent_id: str, project_id: str, hours: float):
        """Enhanced project work with skill gains and moodlets."""
        result = await super().do_project_work(agent_id, project_id, hours)

        if not result:
            return result

        agent = self.state.agents[agent_id]

        # Drain energy
        self.energy.drain(agent_id, "project_work")

        # Add skill XP based on project type
        project = next((p for p in self.projects.get(agent_id, []) if p.project_id == project_id), None)
        if project:
            # Map project to skill
            skill_name = self._infer_skill_from_project(project.name, project.description)
            xp = hours * 25  # 25 XP per hour

            new_level, leveled = self.skills.add_experience(agent_id, skill_name, xp)
            if leveled:
                print(f"  🎯 {agent.name}'s {skill_name} reached level {int(new_level)}!")
                self.moodlets.add_moodlet(agent_id, "accomplished", f"leveled_{skill_name}")

        # Moodlet based on result
        if result.get("breakthrough"):
            self.moodlets.add_moodlet(agent_id, "project_breakthrough", project.name)
        elif result.get("obstacle_encountered"):
            self.moodlets.add_moodlet(agent_id, "creative_block", project.name)
        else:
            self.moodlets.add_moodlet(agent_id, "creative_flow", project.name)

        return result

    async def take_solitude_time(self, agent_id: str, activity: SolitudeActivity, location: str):
        """Enhanced solitude with moodlets."""
        result = await super().take_solitude_time(agent_id, activity, location)

        if not result:
            return result

        # Add appropriate moodlet
        if activity == SolitudeActivity.WALKING:
            self.moodlets.add_moodlet(agent_id, "nice_walk", location)
        elif activity == SolitudeActivity.MEDITATION:
            self.moodlets.add_moodlet(agent_id, "well_rested", "meditation")
        elif activity == SolitudeActivity.NAPPING:
            self.energy.rest(agent_id, 1.0)  # 1 hour nap
            self.moodlets.add_moodlet(agent_id, "well_rested", "nap")

        return result

    def _infer_skill_from_project(self, name: str, description: str) -> str:
        """Infer skill from project name and description using keyword matching."""
        text = f"{name} {description}".lower()

        # Keyword to skill mapping
        skill_keywords = {
            "Painting": ["paint", "mural", "art", "canvas", "draw", "sketch", "portrait"],
            "Writing": ["write", "book", "story", "poem", "novel", "journal", "memoir"],
            "Guitar": ["music", "guitar", "song", "melody", "compose", "instrument"],
            "Piano": ["piano", "keyboard", "symphony", "classical"],
            "Cooking": ["cook", "recipe", "food", "bake", "kitchen", "meal", "dish"],
            "Gardening": ["garden", "plant", "flower", "grow", "seed", "herb", "vegetable"],
            "Handiness": ["build", "craft", "repair", "fix", "construct", "woodwork", "tool"],
            "Programming": ["code", "program", "software", "app", "computer", "tech"],
            "Photography": ["photo", "camera", "picture", "shot", "image"],
        }

        for skill, keywords in skill_keywords.items():
            if any(kw in text for kw in keywords):
                return skill

        return "Creativity"  # Default fallback

    # =========================================================================
    # NEW SIMS-SPECIFIC METHODS
    # =========================================================================

    def get_agent_status(self, agent_id: str) -> Dict:
        """Get complete status for an agent (like Sims UI panel)."""
        agent = self.state.agents.get(agent_id)
        if not agent:
            return {}

        return {
            "name": agent.name,
            "energy": self.energy.get_energy(agent_id),
            "mood_score": self.moodlets.get_mood_score(agent_id),
            "dominant_emotion": self.moodlets.get_dominant_emotion(agent_id),
            "active_moodlets": self.moodlets.get_active_moodlets(agent_id),
            "skills": self.skills.get_all_skills(agent_id),
            "life_goal": self.life_goals.get(agent_id),
            "solitude_need": self.solitude_needs.get(agent_id, 0.5),
        }

    def format_agent_panel(self, agent_id: str) -> str:
        """Format a Sims-style status panel for an agent."""
        status = self.get_agent_status(agent_id)
        if not status:
            return "Unknown agent"

        lines = [
            f"╔══════════════════════════════════════╗",
            f"║  {status['name']:^34}  ║",
            f"╠══════════════════════════════════════╣",
        ]

        # Energy bar
        energy = status['energy']
        bar_len = 20
        filled = int((energy.current / 100) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"║ Energy: [{bar}] {energy.status:>10} ║")

        # Mood
        mood_score = status['mood_score']
        mood_icon = "😊" if mood_score > 0 else "😐" if mood_score == 0 else "😔"
        lines.append(f"║ Mood: {mood_icon} {mood_score:+.0f}  ({status['dominant_emotion'] or 'neutral'})  ║")

        # Active moodlets
        moodlets = status['active_moodlets']
        if moodlets:
            lines.append(f"╟──────────────────────────────────────╢")
            lines.append(f"║ Moodlets:                            ║")
            for m in moodlets[:3]:
                lines.append(f"║   {str(m)[:34]:<34} ║")

        # Top skills
        skills = status['skills']
        if skills:
            lines.append(f"╟──────────────────────────────────────╢")
            lines.append(f"║ Skills:                              ║")
            for s in sorted(skills, key=lambda x: x.level, reverse=True)[:3]:
                lines.append(f"║   {str(s)[:34]:<34} ║")

        # Life goal
        goal = status['life_goal']
        if goal:
            lines.append(f"╟──────────────────────────────────────╢")
            lines.append(f"║ Aspiration: {goal.name:<24} ║")
            progress_bar = "●" * int(goal.progress * 5) + "○" * (5 - int(goal.progress * 5))
            lines.append(f"║   Progress: [{progress_bar}] {goal.progress*100:.0f}%          ║")
            if goal.current_milestone:
                lines.append(f"║   Next: {goal.current_milestone[:28]:<28} ║")

        lines.append(f"╚══════════════════════════════════════╝")

        return "\n".join(lines)

    async def end_of_day(self) -> Dict[str, str]:
        """Run end-of-day routines and generate summaries."""
        summaries = {}

        print(f"\n{'='*50}")
        print(f"END OF DAY {self.current_day}")
        print(f"{'='*50}")

        for agent_id, agent in self.state.agents.items():
            if agent.is_user:
                continue

            # Get today's memories
            memories = []
            if hasattr(agent, 'memories'):
                memories = list(agent.memories)[-10:]  # Last 10

            # Get active moodlets
            moodlets = self.moodlets.get_active_moodlets(agent_id)

            # Get energy
            energy = self.energy.get_energy(agent_id).current

            # Generate summary
            summary = await self.summary_generator.generate_summary(
                agent.name,
                agent.personality,
                memories,
                moodlets,
                energy
            )
            summaries[agent_id] = summary

            print(f"\n{agent.name}'s Day:")
            print(f"  \"{summary}\"")

            # Rest overnight (8 hours)
            self.energy.rest(agent_id, 8.0)

        # Advance day
        self.current_day += 1

        return summaries

    def advance_time(self, hours: float):
        """Advance simulation time with energy decay."""
        self.energy.apply_time_decay(hours)


# =============================================================================
# DEMO
# =============================================================================

async def demo_sims_features():
    """Demonstrate all Sims-inspired features."""

    print("=" * 60)
    print("SIMS-INSPIRED NEUROHOOD DEMO")
    print("=" * 60)

    engine = SimsEnhancedEngine()
    await engine.initialize()

    user = engine.add_user("Blake")

    # Get a resident
    maya_id = "maya_001"
    maya = engine.state.agents[maya_id]

    # === Demo 1: Status Panel ===
    print("\n" + "=" * 50)
    print("DEMO 1: SIMS-STYLE STATUS PANEL")
    print("=" * 50)

    print(engine.format_agent_panel(maya_id))

    # === Demo 2: Moodlets ===
    print("\n" + "=" * 50)
    print("DEMO 2: MOODLET SYSTEM")
    print("=" * 50)

    engine.moodlets.add_moodlet(maya_id, "creative_flow", "painting")
    engine.moodlets.add_moodlet(maya_id, "cozy_atmosphere", "studio")

    print(f"\n{maya.name}'s Active Moodlets:")
    print(engine.moodlets.format_moodlets(maya_id))
    print(f"\nNet Mood Score: {engine.moodlets.get_mood_score(maya_id):+.0f}")

    # === Demo 3: Skill Progression ===
    print("\n" + "=" * 50)
    print("DEMO 3: SKILL PROGRESSION")
    print("=" * 50)

    print(f"\n{maya.name}'s Skills (Before):")
    print(engine.skills.format_skills(maya_id))

    # Add XP
    for _ in range(3):
        new_level, leveled = engine.skills.add_experience(maya_id, "Painting", 50)
        if leveled:
            print(f"\n  🎉 LEVEL UP! Painting is now level {int(new_level)}!")

    print(f"\n{maya.name}'s Skills (After):")
    print(engine.skills.format_skills(maya_id))

    # === Demo 4: Project Work with Skills ===
    print("\n" + "=" * 50)
    print("DEMO 4: PROJECT WORK WITH SKILLS")
    print("=" * 50)

    if maya_id in engine.projects and engine.projects[maya_id]:
        project = engine.projects[maya_id][0]
        print(f"\n{maya.name} works on: {project.name}")

        energy_before = engine.energy.get_energy(maya_id).current
        result = await engine.do_project_work(maya_id, project.project_id, hours=2)
        energy_after = engine.energy.get_energy(maya_id).current

        print(f"  Progress: {project.progress:.1%}")
        print(f"  Energy: {energy_before:.0f} → {energy_after:.0f}")

    # === Demo 5: Life Goals ===
    print("\n" + "=" * 50)
    print("DEMO 5: LIFE GOALS / ASPIRATIONS")
    print("=" * 50)

    goal = engine.life_goals.get(maya_id)
    if goal:
        print(f"\n{maya.name}'s Aspiration: {goal.name}")
        print(f"  \"{goal.description}\"")
        print(f"\n  Milestones:")
        for milestone, done in goal.milestones:
            status = "✓" if done else "○"
            print(f"    [{status}] {milestone}")
        print(f"\n  Progress: {goal.progress*100:.0f}%")

        # Complete a milestone
        goal.complete_milestone("Complete 3 paintings")
        print(f"\n  ** Completed: 'Complete 3 paintings' **")
        print(f"  New Progress: {goal.progress*100:.0f}%")

    # === Demo 6: Conversation with All Features ===
    print("\n" + "=" * 50)
    print("DEMO 6: CONVERSATION WITH ALL FEATURES")
    print("=" * 50)

    engine.user_go_to("corner_cafe")

    # Find Sofia at cafe
    agents_at_cafe = engine.get_agents_at("corner_cafe")
    sofia = next((a for a in agents_at_cafe if a.name == "Sofia"), None)

    if sofia:
        print(f"\nYou talk to {sofia.name}...")

        energy_before = engine.energy.get_energy(sofia.agent_id).current
        conv = await engine.user_talk_to(sofia.agent_id, "Hey Sofia! Beautiful day!")
        energy_after = engine.energy.get_energy(sofia.agent_id).current

        if conv:
            for line in conv.lines[:4]:
                print(f"  {line.speaker_name}: \"{line.text}\"")

            print(f"\n  [Energy: {energy_before:.0f} → {energy_after:.0f}]")
            print(f"  [Moodlets:]")
            print(engine.moodlets.format_moodlets(sofia.agent_id))

    # === Demo 7: End of Day Summary ===
    print("\n" + "=" * 50)
    print("DEMO 7: END OF DAY SUMMARY")
    print("=" * 50)

    await engine.end_of_day()

    # Show updated status
    print("\n" + "=" * 50)
    print("FINAL STATUS PANELS")
    print("=" * 50)

    print(engine.format_agent_panel(maya_id))

    await engine.close()

    print("\n" + "=" * 60)
    print("SIMS-INSPIRED DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_sims_features())
