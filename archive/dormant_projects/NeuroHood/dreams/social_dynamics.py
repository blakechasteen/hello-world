"""
Social Dynamics System for NeuroHood

Comprehensive enrichment layer adding:
- Rumors & Gossip (information propagation with distortion)
- Reputation System (trust, favors, social capital)
- Secrets & Confidences (private knowledge, betrayal potential)
- Group Projects (collaborative endeavors)
- Life Events (birthdays, milestones, anniversaries)
- Weather & Seasons (environmental mood modifiers)
- Community Events (festivals, emergencies, newcomers)
- Emergent Storylines (LLM-generated multi-day narrative arcs)

Design Philosophy:
- Protocol-based extensibility
- Event-driven architecture
- Graceful emergent complexity
- Everything connects to everything (but loosely)

Author: Claude (NeuroHood Enrichment)
Date: November 2025
"""

import asyncio
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Set, Callable, Protocol
import json

from solo_activities import SoloActivityEngine, DEEP_BACKSTORIES


# =============================================================================
# CORE PROTOCOLS - Extensibility Through Abstraction
# =============================================================================

class DynamicEvent(Protocol):
    """Protocol for any event that can occur in the neighborhood."""
    event_id: str
    timestamp: datetime
    participants: List[str]

    def describe(self) -> str: ...
    def affects_mood(self) -> Dict[str, Dict[str, float]]: ...


class Propagatable(Protocol):
    """Protocol for things that spread through social networks."""
    def propagate(self, from_agent: str, to_agent: str, relationship_strength: float) -> 'Propagatable': ...
    def decay(self) -> bool: ...  # Returns True if should be removed


# =============================================================================
# WEATHER & SEASONS
# =============================================================================

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class WeatherType(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"
    CLEAR_NIGHT = "clear_night"


@dataclass
class Weather:
    """Current weather state with mood modifiers."""
    weather_type: WeatherType
    temperature: float  # Celsius
    description: str

    # Mood modifiers (applied to all residents)
    mood_modifiers: Dict[str, float] = field(default_factory=dict)

    # Activity modifiers
    indoor_preference: float = 0.5  # 0 = outdoors, 1 = indoors
    social_modifier: float = 1.0    # Multiplier for social interactions

    @classmethod
    def generate(cls, season: Season, time_of_day: str) -> 'Weather':
        """Generate contextual weather."""

        # Season-based probabilities
        weather_probs = {
            Season.SPRING: {
                WeatherType.SUNNY: 0.3, WeatherType.CLOUDY: 0.3,
                WeatherType.RAINY: 0.3, WeatherType.FOGGY: 0.1
            },
            Season.SUMMER: {
                WeatherType.SUNNY: 0.6, WeatherType.CLOUDY: 0.2,
                WeatherType.STORMY: 0.15, WeatherType.CLEAR_NIGHT: 0.05
            },
            Season.AUTUMN: {
                WeatherType.CLOUDY: 0.4, WeatherType.RAINY: 0.3,
                WeatherType.FOGGY: 0.2, WeatherType.SUNNY: 0.1
            },
            Season.WINTER: {
                WeatherType.CLOUDY: 0.3, WeatherType.SNOWY: 0.3,
                WeatherType.FOGGY: 0.2, WeatherType.SUNNY: 0.2
            }
        }

        probs = weather_probs[season]
        weather_type = random.choices(
            list(probs.keys()),
            weights=list(probs.values())
        )[0]

        # Night override
        if time_of_day == "night" and weather_type == WeatherType.SUNNY:
            weather_type = WeatherType.CLEAR_NIGHT

        # Temperature by season
        temp_ranges = {
            Season.SPRING: (10, 20),
            Season.SUMMER: (20, 35),
            Season.AUTUMN: (5, 18),
            Season.WINTER: (-5, 10)
        }
        temp = random.uniform(*temp_ranges[season])

        # Descriptions and modifiers
        weather_data = {
            WeatherType.SUNNY: {
                "description": "Warm sunlight bathes the neighborhood",
                "mood": {"happiness": 0.1, "energy": 0.05},
                "indoor": 0.2, "social": 1.2
            },
            WeatherType.CLOUDY: {
                "description": "Grey clouds hang overhead, diffusing the light",
                "mood": {"happiness": -0.02},
                "indoor": 0.5, "social": 1.0
            },
            WeatherType.RAINY: {
                "description": "Rain patters against windows, creating a cozy atmosphere",
                "mood": {"happiness": -0.05, "anxiety": 0.02},
                "indoor": 0.9, "social": 0.7
            },
            WeatherType.STORMY: {
                "description": "Thunder rumbles as rain lashes the streets",
                "mood": {"anxiety": 0.1, "energy": -0.1},
                "indoor": 1.0, "social": 0.3
            },
            WeatherType.SNOWY: {
                "description": "Soft snow transforms the neighborhood into a winter wonderland",
                "mood": {"happiness": 0.15, "energy": -0.05},
                "indoor": 0.6, "social": 0.8
            },
            WeatherType.FOGGY: {
                "description": "Mist wraps the neighborhood in mystery",
                "mood": {"anxiety": 0.03},
                "indoor": 0.7, "social": 0.6
            },
            WeatherType.CLEAR_NIGHT: {
                "description": "Stars sparkle brilliantly in the clear night sky",
                "mood": {"happiness": 0.05, "peace": 0.1},
                "indoor": 0.4, "social": 0.5
            }
        }

        data = weather_data[weather_type]
        return cls(
            weather_type=weather_type,
            temperature=temp,
            description=data["description"],
            mood_modifiers=data["mood"],
            indoor_preference=data["indoor"],
            social_modifier=data["social"]
        )


@dataclass
class SeasonalCalendar:
    """Tracks seasons, dates, and generates weather."""
    current_day: int = 1
    current_season: Season = Season.SPRING
    current_weather: Optional[Weather] = None
    days_per_season: int = 30

    def advance_day(self, time_of_day: str = "morning") -> Weather:
        """Advance to next day, potentially changing season."""
        self.current_day += 1

        if self.current_day > self.days_per_season:
            self.current_day = 1
            seasons = list(Season)
            current_idx = seasons.index(self.current_season)
            self.current_season = seasons[(current_idx + 1) % len(seasons)]

        self.current_weather = Weather.generate(self.current_season, time_of_day)
        return self.current_weather

    def get_season_mood(self) -> Dict[str, float]:
        """Get baseline mood from current season."""
        moods = {
            Season.SPRING: {"happiness": 0.05, "energy": 0.1, "hope": 0.1},
            Season.SUMMER: {"happiness": 0.1, "energy": 0.05, "relaxation": 0.1},
            Season.AUTUMN: {"melancholy": 0.05, "creativity": 0.1, "nostalgia": 0.1},
            Season.WINTER: {"introspection": 0.1, "coziness": 0.1, "energy": -0.05}
        }
        return moods[self.current_season]


# =============================================================================
# RUMORS & GOSSIP SYSTEM
# =============================================================================

class RumorType(Enum):
    OBSERVATION = "observation"     # "I saw X doing Y"
    SPECULATION = "speculation"     # "I think X might be..."
    SECRET_SHARED = "secret_shared" # "Don't tell anyone, but..."
    OPINION = "opinion"             # "I think X is..."
    NEWS = "news"                   # "Did you hear about..."


@dataclass
class Rumor:
    """A piece of information that spreads and distorts."""
    rumor_id: str
    rumor_type: RumorType
    subject_id: str              # Who it's about
    original_content: str        # The original information
    current_content: str         # May have been distorted

    originator_id: str           # Who started it
    current_holder_id: str       # Who currently "has" it

    # Propagation state
    times_spread: int = 0
    distortion_level: float = 0.0  # 0 = accurate, 1 = completely distorted
    credibility: float = 1.0       # How believable
    juiciness: float = 0.5         # How likely to spread

    # Who knows this rumor
    known_by: Set[str] = field(default_factory=set)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    last_spread: Optional[datetime] = None
    is_verified: bool = False
    is_debunked: bool = False

    def propagate(self, from_agent: str, to_agent: str,
                  relationship_strength: float) -> 'Rumor':
        """Spread rumor, potentially distorting it."""

        # Create propagated version
        new_rumor = Rumor(
            rumor_id=self.rumor_id,
            rumor_type=self.rumor_type,
            subject_id=self.subject_id,
            original_content=self.original_content,
            current_content=self.current_content,  # Will be distorted
            originator_id=self.originator_id,
            current_holder_id=to_agent,
            times_spread=self.times_spread + 1,
            distortion_level=self.distortion_level,
            credibility=self.credibility,
            juiciness=self.juiciness,
            known_by=self.known_by.copy(),
            created_at=self.created_at,
            is_verified=self.is_verified,
            is_debunked=self.is_debunked
        )

        # Distortion based on relationship and spread count
        distortion_chance = 0.2 + (0.1 * self.times_spread) - (relationship_strength * 0.1)
        if random.random() < distortion_chance:
            new_rumor.distortion_level = min(1.0, self.distortion_level + random.uniform(0.05, 0.15))
            # TODO: LLM-based content distortion

        # Credibility decay
        new_rumor.credibility = max(0.1, self.credibility * (0.9 + relationship_strength * 0.1))

        new_rumor.known_by.add(to_agent)
        new_rumor.last_spread = datetime.now()

        return new_rumor

    def decay(self) -> bool:
        """Returns True if rumor should be forgotten."""
        age_days = (datetime.now() - self.created_at).days

        # Juicy rumors last longer
        max_days = int(7 * (1 + self.juiciness))

        # Verified rumors are permanent
        if self.is_verified:
            return False

        return age_days > max_days


class GossipNetwork:
    """Manages rumor propagation across social network."""

    def __init__(self):
        self.active_rumors: Dict[str, Rumor] = {}
        self.agent_rumors: Dict[str, Set[str]] = {}  # agent_id -> rumor_ids they know
        self.rumor_history: List[Dict] = []

    def create_rumor(self, rumor_type: RumorType, subject_id: str,
                     content: str, originator_id: str, juiciness: float = 0.5) -> Rumor:
        """Create a new rumor."""
        rumor = Rumor(
            rumor_id=f"rumor_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            rumor_type=rumor_type,
            subject_id=subject_id,
            original_content=content,
            current_content=content,
            originator_id=originator_id,
            current_holder_id=originator_id,
            juiciness=juiciness,
            known_by={originator_id}
        )

        self.active_rumors[rumor.rumor_id] = rumor

        if originator_id not in self.agent_rumors:
            self.agent_rumors[originator_id] = set()
        self.agent_rumors[originator_id].add(rumor.rumor_id)

        return rumor

    def spread_rumor(self, rumor_id: str, from_agent: str, to_agent: str,
                     relationship_strength: float = 0.5) -> Optional[Rumor]:
        """Attempt to spread a rumor."""
        if rumor_id not in self.active_rumors:
            return None

        rumor = self.active_rumors[rumor_id]

        # Already knows?
        if to_agent in rumor.known_by:
            return None

        # Spread probability based on juiciness and relationship
        spread_prob = rumor.juiciness * (0.5 + relationship_strength * 0.5)
        if random.random() > spread_prob:
            return None

        # Propagate
        new_rumor = rumor.propagate(from_agent, to_agent, relationship_strength)
        self.active_rumors[rumor_id] = new_rumor

        if to_agent not in self.agent_rumors:
            self.agent_rumors[to_agent] = set()
        self.agent_rumors[to_agent].add(rumor_id)

        self.rumor_history.append({
            "rumor_id": rumor_id,
            "from": from_agent,
            "to": to_agent,
            "timestamp": datetime.now().isoformat(),
            "distortion": new_rumor.distortion_level
        })

        return new_rumor

    def get_shareable_rumors(self, agent_id: str, exclude_about: Optional[str] = None) -> List[Rumor]:
        """Get rumors an agent might share."""
        if agent_id not in self.agent_rumors:
            return []

        rumors = []
        for rid in self.agent_rumors[agent_id]:
            if rid in self.active_rumors:
                rumor = self.active_rumors[rid]
                # Don't share rumors about the person you're talking to
                if exclude_about and rumor.subject_id == exclude_about:
                    continue
                rumors.append(rumor)

        # Sort by juiciness
        return sorted(rumors, key=lambda r: r.juiciness, reverse=True)

    def cleanup_old_rumors(self):
        """Remove decayed rumors."""
        to_remove = [rid for rid, r in self.active_rumors.items() if r.decay()]
        for rid in to_remove:
            del self.active_rumors[rid]
            for agent_rumors in self.agent_rumors.values():
                agent_rumors.discard(rid)


# =============================================================================
# REPUTATION SYSTEM
# =============================================================================

@dataclass
class ReputationFactors:
    """Components of reputation."""
    trustworthiness: float = 0.5   # Do they keep secrets?
    reliability: float = 0.5       # Do they follow through?
    generosity: float = 0.5        # Do they help others?
    warmth: float = 0.5            # Are they pleasant?
    competence: float = 0.5        # Are they capable?

    def overall(self) -> float:
        """Calculate overall reputation score."""
        return (self.trustworthiness * 0.25 +
                self.reliability * 0.2 +
                self.generosity * 0.2 +
                self.warmth * 0.2 +
                self.competence * 0.15)


@dataclass
class Favor:
    """A favor owed between agents."""
    favor_id: str
    from_agent: str   # Who did the favor
    to_agent: str     # Who owes
    description: str
    magnitude: float  # 0.1 = small, 1.0 = life-changing
    created_at: datetime
    repaid: bool = False
    repaid_at: Optional[datetime] = None


class ReputationSystem:
    """Tracks reputation and social capital."""

    def __init__(self):
        # Personal reputation factors
        self.reputations: Dict[str, ReputationFactors] = {}

        # Pairwise opinions (what A thinks of B)
        self.opinions: Dict[str, Dict[str, float]] = {}

        # Favors owed
        self.favors: Dict[str, Favor] = {}
        self.agent_favors_owed: Dict[str, Set[str]] = {}  # agent -> favor_ids they owe
        self.agent_favors_given: Dict[str, Set[str]] = {} # agent -> favor_ids owed to them

    def initialize_reputation(self, agent_id: str, personality: Dict[str, float]):
        """Initialize reputation from personality."""
        rep = ReputationFactors(
            trustworthiness=0.5 + (personality.get('conscientiousness', 0.5) - 0.5) * 0.3,
            reliability=0.5 + (personality.get('conscientiousness', 0.5) - 0.5) * 0.4,
            generosity=0.5 + (personality.get('agreeableness', 0.5) - 0.5) * 0.4,
            warmth=0.5 + (personality.get('extraversion', 0.5) - 0.5) * 0.3 +
                   (personality.get('agreeableness', 0.5) - 0.5) * 0.2,
            competence=0.5  # Earned through actions
        )
        self.reputations[agent_id] = rep

    def update_reputation(self, agent_id: str, factor: str, delta: float):
        """Update a reputation factor."""
        if agent_id not in self.reputations:
            self.reputations[agent_id] = ReputationFactors()

        rep = self.reputations[agent_id]
        current = getattr(rep, factor, 0.5)
        setattr(rep, factor, max(0.0, min(1.0, current + delta)))

    def set_opinion(self, from_agent: str, about_agent: str, opinion: float):
        """Set what one agent thinks of another."""
        if from_agent not in self.opinions:
            self.opinions[from_agent] = {}
        self.opinions[from_agent][about_agent] = max(-1.0, min(1.0, opinion))

    def get_opinion(self, from_agent: str, about_agent: str) -> float:
        """Get what one agent thinks of another (-1 to 1)."""
        if from_agent in self.opinions and about_agent in self.opinions[from_agent]:
            return self.opinions[from_agent][about_agent]

        # Default to neutral, slightly influenced by reputation
        if about_agent in self.reputations:
            return (self.reputations[about_agent].overall() - 0.5) * 0.5
        return 0.0

    def record_favor(self, from_agent: str, to_agent: str,
                     description: str, magnitude: float) -> Favor:
        """Record a favor done."""
        favor = Favor(
            favor_id=f"favor_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            from_agent=from_agent,
            to_agent=to_agent,
            description=description,
            magnitude=magnitude,
            created_at=datetime.now()
        )

        self.favors[favor.favor_id] = favor

        if to_agent not in self.agent_favors_owed:
            self.agent_favors_owed[to_agent] = set()
        self.agent_favors_owed[to_agent].add(favor.favor_id)

        if from_agent not in self.agent_favors_given:
            self.agent_favors_given[from_agent] = set()
        self.agent_favors_given[from_agent].add(favor.favor_id)

        # Boost giver's reputation
        self.update_reputation(from_agent, "generosity", magnitude * 0.1)

        return favor

    def get_social_capital(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent's overall social capital."""
        favors_owed_to_them = len(self.agent_favors_given.get(agent_id, set()))
        favors_they_owe = len(self.agent_favors_owed.get(agent_id, set()))

        reputation = self.reputations.get(agent_id, ReputationFactors()).overall()

        # Positive opinions about them
        positive_opinions = sum(
            1 for agent_opinions in self.opinions.values()
            if agent_id in agent_opinions and agent_opinions[agent_id] > 0.3
        )

        return {
            "reputation_score": reputation,
            "favors_owed_to_them": favors_owed_to_them,
            "favors_they_owe": favors_they_owe,
            "positive_relationships": positive_opinions,
            "social_capital": reputation * 0.4 + (favors_owed_to_them * 0.1) + (positive_opinions * 0.05)
        }


# =============================================================================
# SECRETS SYSTEM
# =============================================================================

class SecretType(Enum):
    PERSONAL = "personal"         # About themselves
    RELATIONSHIP = "relationship" # About a relationship
    OBSERVATION = "observation"   # Something they witnessed
    ASPIRATION = "aspiration"     # Hidden dreams
    FEAR = "fear"                 # Hidden fears
    PAST = "past"                 # Something from their past


@dataclass
class Secret:
    """A secret that can be shared or discovered."""
    secret_id: str
    secret_type: SecretType
    owner_id: str                 # Whose secret
    content: str                  # The actual secret

    # Sensitivity
    sensitivity: float = 0.5      # How bad if revealed (0-1)
    shame_level: float = 0.3      # How ashamed they'd be

    # Knowledge state
    known_by: Set[str] = field(default_factory=set)
    confided_to: Set[str] = field(default_factory=set)  # Willingly shared
    discovered_by: Set[str] = field(default_factory=set) # Unwillingly discovered

    # Trust tracking
    trust_required_to_share: float = 0.7  # Minimum trust to confide

    def can_confide_to(self, agent_id: str, trust_level: float) -> bool:
        """Check if owner would confide to this agent."""
        if agent_id in self.known_by:
            return False  # Already knows
        return trust_level >= self.trust_required_to_share

    def confide(self, to_agent: str):
        """Share secret willingly."""
        self.known_by.add(to_agent)
        self.confided_to.add(to_agent)

    def discover(self, by_agent: str):
        """Secret is discovered unwillingly."""
        self.known_by.add(by_agent)
        self.discovered_by.add(by_agent)

    def betray(self, by_agent: str, to_agent: str) -> bool:
        """Agent reveals secret to someone else. Returns True if betrayal."""
        if by_agent not in self.known_by:
            return False

        # It's betrayal if it was confided, not discovered
        is_betrayal = by_agent in self.confided_to

        self.known_by.add(to_agent)
        return is_betrayal


class SecretsManager:
    """Manages all secrets in the neighborhood."""

    def __init__(self):
        self.secrets: Dict[str, Secret] = {}
        self.agent_secrets: Dict[str, Set[str]] = {}  # Secrets owned by agent
        self.betrayals: List[Dict] = []  # History of betrayals

    def create_secret(self, owner_id: str, secret_type: SecretType,
                      content: str, sensitivity: float = 0.5) -> Secret:
        """Create a new secret."""
        secret = Secret(
            secret_id=f"secret_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            secret_type=secret_type,
            owner_id=owner_id,
            content=content,
            sensitivity=sensitivity,
            known_by={owner_id}
        )

        self.secrets[secret.secret_id] = secret

        if owner_id not in self.agent_secrets:
            self.agent_secrets[owner_id] = set()
        self.agent_secrets[owner_id].add(secret.secret_id)

        return secret

    def get_confide_candidates(self, secret_id: str,
                               relationships: Dict[str, float]) -> List[str]:
        """Get agents this secret could be confided to."""
        if secret_id not in self.secrets:
            return []

        secret = self.secrets[secret_id]
        return [
            agent_id for agent_id, trust in relationships.items()
            if secret.can_confide_to(agent_id, trust)
        ]

    def confide_secret(self, secret_id: str, to_agent: str) -> bool:
        """Confide a secret to someone."""
        if secret_id not in self.secrets:
            return False
        self.secrets[secret_id].confide(to_agent)
        return True

    def betray_secret(self, secret_id: str, by_agent: str, to_agent: str,
                      reputation_system: Optional[ReputationSystem] = None) -> bool:
        """Betray a confidence."""
        if secret_id not in self.secrets:
            return False

        secret = self.secrets[secret_id]
        was_betrayal = secret.betray(by_agent, to_agent)

        if was_betrayal:
            self.betrayals.append({
                "secret_id": secret_id,
                "betrayer": by_agent,
                "told_to": to_agent,
                "owner": secret.owner_id,
                "timestamp": datetime.now().isoformat()
            })

            # Devastating reputation hit
            if reputation_system:
                reputation_system.update_reputation(by_agent, "trustworthiness", -0.3)

        return was_betrayal


# =============================================================================
# LIFE EVENTS SYSTEM
# =============================================================================

class LifeEventType(Enum):
    BIRTHDAY = "birthday"
    ANNIVERSARY = "anniversary"
    ACHIEVEMENT = "achievement"
    LOSS = "loss"
    DISCOVERY = "discovery"
    RELATIONSHIP_MILESTONE = "relationship_milestone"
    PERSONAL_GROWTH = "personal_growth"
    SETBACK = "setback"


@dataclass
class LifeEvent:
    """A significant life event."""
    event_id: str
    event_type: LifeEventType
    agent_id: str
    description: str

    # Impact
    mood_impact: Dict[str, float] = field(default_factory=dict)
    relationship_impacts: Dict[str, float] = field(default_factory=dict)

    # Observability
    is_public: bool = True
    witnesses: Set[str] = field(default_factory=set)

    # Timing
    date: datetime = field(default_factory=datetime.now)

    # Can trigger other events
    triggers_rumor: bool = False
    triggers_celebration: bool = False


class LifeEventsCalendar:
    """Manages life events for all residents."""

    def __init__(self):
        self.upcoming_events: List[LifeEvent] = []
        self.past_events: List[LifeEvent] = []
        self.agent_birthdays: Dict[str, int] = {}  # agent_id -> day of year
        self.agent_anniversaries: Dict[str, List[Dict]] = {}  # agent_id -> list of anniversaries

    def set_birthday(self, agent_id: str, day_of_year: int):
        """Set an agent's birthday."""
        self.agent_birthdays[agent_id] = day_of_year

    def check_birthdays(self, current_day: int) -> List[LifeEvent]:
        """Check for birthdays on current day."""
        events = []
        for agent_id, birthday in self.agent_birthdays.items():
            if birthday == current_day:
                event = LifeEvent(
                    event_id=f"birthday_{agent_id}_{current_day}",
                    event_type=LifeEventType.BIRTHDAY,
                    agent_id=agent_id,
                    description=f"It's {agent_id}'s birthday!",
                    mood_impact={"happiness": 0.3, "social": 0.2},
                    is_public=True,
                    triggers_celebration=True
                )
                events.append(event)
        return events

    def create_achievement(self, agent_id: str, description: str,
                           witnesses: Set[str] = None) -> LifeEvent:
        """Create an achievement event."""
        return LifeEvent(
            event_id=f"achieve_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type=LifeEventType.ACHIEVEMENT,
            agent_id=agent_id,
            description=description,
            mood_impact={"happiness": 0.2, "confidence": 0.15, "pride": 0.25},
            witnesses=witnesses or set(),
            is_public=True,
            triggers_rumor=True
        )


# =============================================================================
# GROUP PROJECTS
# =============================================================================

@dataclass
class GroupProject:
    """A collaborative project between multiple residents."""
    project_id: str
    name: str
    description: str

    # Participants
    initiator_id: str
    participants: Set[str]
    roles: Dict[str, str] = field(default_factory=dict)  # agent_id -> role

    # Progress
    status: str = "planning"  # planning, active, completed, abandoned
    progress: float = 0.0
    milestones: List[Dict] = field(default_factory=list)

    # Collaboration quality
    harmony: float = 1.0  # How well they work together
    conflicts: List[Dict] = field(default_factory=list)

    # Timeline
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_participant(self, agent_id: str, role: str):
        """Add a participant with role."""
        self.participants.add(agent_id)
        self.roles[agent_id] = role

    def add_milestone(self, description: str, achieved_by: str):
        """Record a milestone."""
        self.milestones.append({
            "description": description,
            "achieved_by": achieved_by,
            "timestamp": datetime.now().isoformat()
        })
        self.progress = min(1.0, self.progress + 0.1)

    def record_conflict(self, between: List[str], description: str):
        """Record a conflict."""
        self.conflicts.append({
            "between": between,
            "description": description,
            "timestamp": datetime.now().isoformat()
        })
        self.harmony = max(0.0, self.harmony - 0.15)


class GroupProjectManager:
    """Manages collaborative projects."""

    def __init__(self):
        self.projects: Dict[str, GroupProject] = {}
        self.agent_projects: Dict[str, Set[str]] = {}

    def create_project(self, name: str, description: str,
                       initiator_id: str, initial_participants: Set[str] = None) -> GroupProject:
        """Create a new group project."""
        participants = initial_participants or {initiator_id}
        participants.add(initiator_id)

        project = GroupProject(
            project_id=f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            name=name,
            description=description,
            initiator_id=initiator_id,
            participants=participants
        )

        self.projects[project.project_id] = project

        for agent_id in participants:
            if agent_id not in self.agent_projects:
                self.agent_projects[agent_id] = set()
            self.agent_projects[agent_id].add(project.project_id)

        return project

    def suggest_collaborations(self, agent_id: str,
                               all_agents: Dict[str, Any],
                               backstories: Dict) -> List[Dict]:
        """Suggest potential collaborations based on interests."""
        suggestions = []

        my_backstory = backstories.get(agent_id)
        if not my_backstory:
            return []

        for other_id, other_backstory in backstories.items():
            if other_id == agent_id:
                continue

            # Look for complementary skills
            # Maya (artist) + Diego (musician) = art-music collaboration
            my_occ = getattr(my_backstory, 'occupation', '').lower()
            other_occ = getattr(other_backstory, 'occupation', '').lower()

            synergies = [
                ({'art', 'music'}, "An audiovisual art installation"),
                ({'cafe', 'music'}, "Live music nights at the cafe"),
                ({'astronomy', 'art'}, "Constellation illustration project"),
                ({'writing', 'history'}, "Neighborhood history book"),
                ({'cafe', 'community'}, "Community gathering events"),
            ]

            for skills, project_idea in synergies:
                my_match = any(s in my_occ for s in skills)
                other_match = any(s in other_occ for s in skills)

                if my_match and other_match:
                    suggestions.append({
                        "partner": other_id,
                        "idea": project_idea,
                        "synergy": "occupation_match"
                    })

        return suggestions


# =============================================================================
# COMMUNITY EVENTS
# =============================================================================

class CommunityEventType(Enum):
    FESTIVAL = "festival"
    EMERGENCY = "emergency"
    NEWCOMER = "newcomer"
    DEPARTURE = "departure"
    COMMUNITY_MEETING = "community_meeting"
    CELEBRATION = "celebration"
    CRISIS = "crisis"


@dataclass
class CommunityEvent:
    """An event affecting the whole neighborhood."""
    event_id: str
    event_type: CommunityEventType
    title: str
    description: str

    # Scope
    affected_locations: Set[str] = field(default_factory=set)
    affected_agents: Set[str] = field(default_factory=set)

    # Impact
    mood_impact_all: Dict[str, float] = field(default_factory=dict)
    relationship_boost: float = 0.0  # Shared experiences bond people

    # Duration
    start_time: datetime = field(default_factory=datetime.now)
    duration_hours: int = 2
    is_ongoing: bool = True

    # Participation
    attending: Set[str] = field(default_factory=set)
    responses: Dict[str, str] = field(default_factory=dict)  # agent reactions


class CommunityEventGenerator:
    """Generates community events based on context."""

    def __init__(self, ollama_client=None):
        self.client = ollama_client
        self.recent_events: List[CommunityEvent] = []

    def generate_seasonal_festival(self, season: Season) -> CommunityEvent:
        """Generate a seasonal festival."""
        festivals = {
            Season.SPRING: ("Spring Garden Festival",
                           "Residents gather to plant flowers and celebrate renewal"),
            Season.SUMMER: ("Neighborhood Block Party",
                           "Music, food, and laughter fill the streets"),
            Season.AUTUMN: ("Harvest Celebration",
                           "A potluck dinner with the season's bounty"),
            Season.WINTER: ("Winter Lights Festival",
                           "The neighborhood glows with decorations and warmth")
        }

        title, desc = festivals[season]

        return CommunityEvent(
            event_id=f"festival_{season.value}_{random.randint(1000,9999)}",
            event_type=CommunityEventType.FESTIVAL,
            title=title,
            description=desc,
            mood_impact_all={"happiness": 0.2, "community": 0.3, "loneliness": -0.2},
            relationship_boost=0.1,
            duration_hours=6
        )

    def generate_emergency(self, severity: float = 0.5) -> CommunityEvent:
        """Generate an emergency event."""
        emergencies = [
            ("Power Outage", "The neighborhood loses electricity", 0.3),
            ("Water Main Break", "Water service interrupted", 0.4),
            ("Lost Pet Search", "A beloved neighborhood pet goes missing", 0.2),
            ("Minor Fire", "Small fire at the community center", 0.6),
            ("Storm Damage", "Strong winds damage some property", 0.5)
        ]

        # Filter by severity
        valid = [e for e in emergencies if abs(e[2] - severity) < 0.3]
        if not valid:
            valid = emergencies

        title, desc, sev = random.choice(valid)

        return CommunityEvent(
            event_id=f"emergency_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            event_type=CommunityEventType.EMERGENCY,
            title=title,
            description=desc,
            mood_impact_all={"anxiety": sev * 0.4, "community": 0.1},  # Crises bond people
            relationship_boost=0.15,  # Helping each other
            duration_hours=int(sev * 8)
        )


# =============================================================================
# EMERGENT STORYLINES
# =============================================================================

class StorylinePhase(Enum):
    INCITING_INCIDENT = "inciting_incident"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"


@dataclass
class Storyline:
    """An emergent multi-day narrative arc."""
    storyline_id: str
    title: str
    theme: str  # e.g., "reconciliation", "discovery", "conflict"

    # Characters
    protagonist_id: str
    deuteragonist_id: Optional[str] = None  # Second main character
    supporting_characters: Set[str] = field(default_factory=set)

    # Structure
    current_phase: StorylinePhase = StorylinePhase.INCITING_INCIDENT
    beats: List[Dict] = field(default_factory=list)  # Story events

    # Stakes
    stakes_description: str = ""
    emotional_stakes: float = 0.5

    # Lifecycle
    started_at: datetime = field(default_factory=datetime.now)
    estimated_days: int = 3
    is_active: bool = True

    def advance_phase(self):
        """Move to next phase."""
        phases = list(StorylinePhase)
        current_idx = phases.index(self.current_phase)
        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
        else:
            self.is_active = False

    def add_beat(self, description: str, participants: List[str],
                 emotional_impact: Dict[str, float] = None):
        """Add a story beat."""
        self.beats.append({
            "phase": self.current_phase.value,
            "description": description,
            "participants": participants,
            "emotional_impact": emotional_impact or {},
            "timestamp": datetime.now().isoformat()
        })


class StorylineEngine:
    """Generates and manages emergent storylines."""

    def __init__(self, ollama_client=None):
        self.client = ollama_client
        self.active_storylines: Dict[str, Storyline] = {}
        self.completed_storylines: List[Storyline] = []

    async def generate_storyline(self, agents: Dict[str, Any],
                                  backstories: Dict,
                                  relationships: Dict[str, Dict[str, float]],
                                  recent_events: List[str]) -> Optional[Storyline]:
        """Generate a new storyline based on current state."""

        # Find interesting tensions
        tensions = []

        for agent_id, agent in agents.items():
            backstory = backstories.get(agent_id)
            if not backstory:
                continue

            # Unresolved regrets can drive stories
            if hasattr(backstory, 'biggest_regret') and backstory.biggest_regret:
                tensions.append({
                    "agent": agent_id,
                    "type": "regret",
                    "theme": "redemption",
                    "description": backstory.biggest_regret
                })

            # Secret dreams seeking fulfillment
            if hasattr(backstory, 'secret_dream') and backstory.secret_dream:
                tensions.append({
                    "agent": agent_id,
                    "type": "aspiration",
                    "theme": "pursuit",
                    "description": backstory.secret_dream
                })

            # Fears that might manifest
            if hasattr(backstory, 'deepest_fear') and backstory.deepest_fear:
                tensions.append({
                    "agent": agent_id,
                    "type": "fear",
                    "theme": "confrontation",
                    "description": backstory.deepest_fear
                })

        if not tensions:
            return None

        # Pick a tension to explore
        tension = random.choice(tensions)

        # Find a potential deuteragonist (someone connected)
        protagonist = tension["agent"]
        deuteragonist = None

        if protagonist in relationships:
            # Pick someone with moderate relationship (not too close, not too far)
            candidates = [
                aid for aid, strength in relationships[protagonist].items()
                if 0.3 < strength < 0.7
            ]
            if candidates:
                deuteragonist = random.choice(candidates)

        # Generate storyline
        themes = {
            "regret": ["reconciliation", "forgiveness", "closure"],
            "aspiration": ["discovery", "achievement", "revelation"],
            "fear": ["confrontation", "overcoming", "acceptance"]
        }

        theme = random.choice(themes[tension["type"]])

        storyline = Storyline(
            storyline_id=f"story_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=f"{tension['type'].title()} and {theme.title()}",
            theme=theme,
            protagonist_id=protagonist,
            deuteragonist_id=deuteragonist,
            stakes_description=tension["description"],
            emotional_stakes=0.7
        )

        # Generate inciting incident
        if self.client:
            incident = await self._generate_inciting_incident(
                storyline, agents, backstories
            )
            storyline.add_beat(incident, [protagonist])
        else:
            storyline.add_beat(
                f"Something stirs within {agents[protagonist].name} regarding {tension['description']}",
                [protagonist]
            )

        self.active_storylines[storyline.storyline_id] = storyline
        return storyline

    async def _generate_inciting_incident(self, storyline: Storyline,
                                           agents: Dict, backstories: Dict) -> str:
        """Use LLM to generate inciting incident."""
        protagonist = agents.get(storyline.protagonist_id)
        backstory = backstories.get(storyline.protagonist_id)

        if not protagonist or not backstory or not self.client:
            return "Something begins to change..."

        prompt = f"""Generate a brief inciting incident (2-3 sentences) for a story about:

Character: {protagonist.name}
Theme: {storyline.theme}
Stakes: {storyline.stakes_description}
Personality: {getattr(backstory, 'life_philosophy', 'thoughtful')}

The incident should be subtle but meaningful - a small moment that begins change."""

        try:
            response = await self.client.generate(
                system_prompt="You are a narrative designer creating subtle, emotionally resonant story moments.",
                user_prompt=prompt,
                max_tokens=100
            )
            return response.strip()
        except Exception:
            return f"A moment of quiet realization washes over {protagonist.name}..."

    async def advance_storyline(self, storyline_id: str,
                                 context: Dict) -> Optional[Dict]:
        """Advance a storyline to its next beat."""
        if storyline_id not in self.active_storylines:
            return None

        storyline = self.active_storylines[storyline_id]

        # Move to next phase
        storyline.advance_phase()

        if not storyline.is_active:
            # Story complete
            self.completed_storylines.append(storyline)
            del self.active_storylines[storyline_id]
            return {"status": "completed", "storyline": storyline}

        # Generate next beat
        # TODO: LLM-based beat generation
        beat_templates = {
            StorylinePhase.RISING_ACTION: "The situation develops as {protagonist} faces new challenges",
            StorylinePhase.CLIMAX: "Everything comes to a head for {protagonist}",
            StorylinePhase.FALLING_ACTION: "The aftermath begins to settle",
            StorylinePhase.RESOLUTION: "A new normal emerges"
        }

        template = beat_templates.get(storyline.current_phase, "The story continues...")
        beat = template.format(protagonist=storyline.protagonist_id)

        storyline.add_beat(beat, [storyline.protagonist_id])

        return {"status": "advanced", "phase": storyline.current_phase.value, "beat": beat}


# =============================================================================
# UNIFIED SOCIAL DYNAMICS ENGINE
# =============================================================================

class SocialDynamicsEngine(SoloActivityEngine):
    """
    Complete social dynamics engine combining all systems.

    Extends SoloActivityEngine (which extends AutonomousNeighborhoodEngine)
    with social dynamics, creating a rich simulation.
    """

    def __init__(self, ollama_config=None):
        super().__init__(ollama_config)

        # Calendar and weather
        self.calendar = SeasonalCalendar()

        # Social systems
        self.gossip_network = GossipNetwork()
        self.reputation_system = ReputationSystem()
        self.secrets_manager = SecretsManager()

        # Events
        self.life_events = LifeEventsCalendar()
        self.community_events = CommunityEventGenerator()
        self.active_community_events: List[CommunityEvent] = []

        # Collaboration
        self.group_projects = GroupProjectManager()

        # Storylines
        self.storyline_engine: Optional[StorylineEngine] = None

    async def initialize(self, load_path=None):
        """Initialize with all social systems."""
        await super().initialize(load_path)

        # Initialize storyline engine if Ollama available
        use_ollama = getattr(self, '_use_ollama', False) or hasattr(self, 'ollama_client')
        if use_ollama and self.ollama_client:
            self.storyline_engine = StorylineEngine(self.ollama_client)
            self.community_events.client = self.ollama_client

        # Initialize reputations
        for agent_id, agent in self.state.agents.items():
            self.reputation_system.initialize_reputation(agent_id, agent.personality)

        # Generate initial secrets from backstories
        await self._generate_initial_secrets()

        # Set birthdays (random days)
        for agent_id in self.state.agents:
            self.life_events.set_birthday(agent_id, random.randint(1, 120))

        # Initial weather
        self.calendar.advance_day()

    async def _generate_initial_secrets(self):
        """Generate secrets from deep backstories."""
        for agent_id, backstory in DEEP_BACKSTORIES.items():
            # Regrets become secrets
            if backstory.biggest_regret:
                self.secrets_manager.create_secret(
                    agent_id, SecretType.PAST,
                    backstory.biggest_regret,
                    sensitivity=0.7
                )

            # Secret dreams are... secrets
            if backstory.secret_dream:
                self.secrets_manager.create_secret(
                    agent_id, SecretType.ASPIRATION,
                    backstory.secret_dream,
                    sensitivity=0.4
                )

            # Deepest fears
            if backstory.deepest_fear:
                self.secrets_manager.create_secret(
                    agent_id, SecretType.FEAR,
                    backstory.deepest_fear,
                    sensitivity=0.6
                )

    async def simulate_social_step(self) -> Dict[str, Any]:
        """Simulate one step of social dynamics."""
        results = {
            "weather": None,
            "rumors_spread": [],
            "secrets_shared": [],
            "reputation_changes": [],
            "life_events": [],
            "community_events": [],
            "storyline_updates": [],
            "project_progress": []
        }

        # Weather mood effects
        if self.calendar.current_weather:
            for agent_id, agent in self.state.agents.items():
                for emotion, delta in self.calendar.current_weather.mood_modifiers.items():
                    if emotion in agent.emotions:
                        agent.emotions[emotion] = max(-1, min(1,
                            agent.emotions[emotion] + delta * 0.1))
            results["weather"] = {
                "type": self.calendar.current_weather.weather_type.value,
                "description": self.calendar.current_weather.description
            }

        # Rumor spreading (happens during social interactions)
        for agent_id in self.state.agents:
            shareable = self.gossip_network.get_shareable_rumors(agent_id)
            if shareable and random.random() < 0.3:  # 30% chance to spread
                rumor = random.choice(shareable)

                # Pick a target who doesn't know
                possible_targets = [
                    aid for aid in self.state.agents
                    if aid != agent_id and aid not in rumor.known_by
                ]

                if possible_targets:
                    target = random.choice(possible_targets)
                    # Get relationship strength
                    rel_strength = 0.5
                    if hasattr(self.state.agents[agent_id], 'relationships'):
                        rel_strength = self.state.agents[agent_id].relationships.get(
                            target, {}
                        ).get('familiarity', 0.5)

                    new_rumor = self.gossip_network.spread_rumor(
                        rumor.rumor_id, agent_id, target, rel_strength
                    )
                    if new_rumor:
                        results["rumors_spread"].append({
                            "from": agent_id,
                            "to": target,
                            "about": rumor.subject_id,
                            "distortion": new_rumor.distortion_level
                        })

        # Secret sharing (low probability, high trust required)
        for agent_id in self.state.agents:
            if agent_id in self.secrets_manager.agent_secrets:
                for secret_id in self.secrets_manager.agent_secrets[agent_id]:
                    secret = self.secrets_manager.secrets.get(secret_id)
                    if not secret or secret.owner_id != agent_id:
                        continue

                    # Very low chance to confide
                    if random.random() < 0.05:
                        # Find high-trust relationship
                        agent = self.state.agents[agent_id]
                        if hasattr(agent, 'relationships'):
                            for other_id, rel in agent.relationships.items():
                                trust = rel.get('affection', 0) + rel.get('familiarity', 0)
                                if trust > 1.4:  # High trust threshold
                                    if secret.can_confide_to(other_id, trust / 2):
                                        secret.confide(other_id)
                                        results["secrets_shared"].append({
                                            "from": agent_id,
                                            "to": other_id,
                                            "type": secret.secret_type.value
                                        })
                                        break

        # Life events check
        birthday_events = self.life_events.check_birthdays(self.calendar.current_day)
        for event in birthday_events:
            results["life_events"].append({
                "type": "birthday",
                "agent": event.agent_id,
                "description": event.description
            })

        # Storyline advancement (low probability per step)
        if self.storyline_engine:
            # Maybe start new storyline
            if not self.storyline_engine.active_storylines and random.random() < 0.1:
                relationships = {}
                for agent_id, agent in self.state.agents.items():
                    if hasattr(agent, 'relationships'):
                        relationships[agent_id] = {
                            other: rel.get('familiarity', 0.5)
                            for other, rel in agent.relationships.items()
                        }

                storyline = await self.storyline_engine.generate_storyline(
                    self.state.agents, DEEP_BACKSTORIES, relationships, []
                )
                if storyline:
                    results["storyline_updates"].append({
                        "action": "started",
                        "title": storyline.title,
                        "protagonist": storyline.protagonist_id,
                        "theme": storyline.theme
                    })

            # Advance existing storylines
            for storyline_id in list(self.storyline_engine.active_storylines.keys()):
                if random.random() < 0.15:  # 15% chance to advance per step
                    update = await self.storyline_engine.advance_storyline(storyline_id, {})
                    if update:
                        results["storyline_updates"].append(update)

        # Cleanup
        self.gossip_network.cleanup_old_rumors()

        return results

    async def advance_day(self) -> Dict[str, Any]:
        """Advance to the next day with all systems."""
        # New weather
        weather = self.calendar.advance_day()

        # Season change events
        if self.calendar.current_day == 1:
            festival = self.community_events.generate_seasonal_festival(
                self.calendar.current_season
            )
            self.active_community_events.append(festival)

        # Random community event chance
        if random.random() < 0.05:  # 5% daily chance
            emergency = self.community_events.generate_emergency()
            self.active_community_events.append(emergency)

        # Decay solitude needs overnight
        for agent_id in self.solitude_needs:
            self.satisfy_solitude(agent_id, 0.3)  # Sleep recharges

        return {
            "day": self.calendar.current_day,
            "season": self.calendar.current_season.value,
            "weather": weather.weather_type.value,
            "weather_desc": weather.description,
            "community_events": [e.title for e in self.active_community_events if e.is_ongoing]
        }

    def get_social_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get social summary for an agent."""
        return {
            "reputation": self.reputation_system.get_social_capital(agent_id),
            "rumors_known": len(self.gossip_network.agent_rumors.get(agent_id, set())),
            "secrets_kept": len([
                s for sid in self.secrets_manager.agent_secrets.get(agent_id, set())
                for s in [self.secrets_manager.secrets.get(sid)]
                if s and s.owner_id == agent_id
            ]),
            "active_projects": len(self.group_projects.agent_projects.get(agent_id, set())),
            "solitude_need": self.solitude_needs.get(agent_id, 0.5)
        }


# =============================================================================
# DEMO
# =============================================================================

async def demo_social_dynamics():
    """Demonstrate the full social dynamics system."""
    print("=" * 70)
    print("SOCIAL DYNAMICS DEMO")
    print("Complete Neighborhood Enrichment System")
    print("=" * 70)

    engine = SocialDynamicsEngine()
    await engine.initialize()

    # === Demo 1: Weather & Seasons ===
    print("\n" + "=" * 50)
    print("DEMO 1: WEATHER & SEASONS")
    print("=" * 50)

    print(f"\n  Season: {engine.calendar.current_season.value.title()}")
    print(f"  Day: {engine.calendar.current_day}")
    if engine.calendar.current_weather:
        w = engine.calendar.current_weather
        print(f"  Weather: {w.weather_type.value}")
        print(f"  Description: \"{w.description}\"")
        print(f"  Temperature: {w.temperature:.1f}C")
        print(f"  Indoor preference: {w.indoor_preference*100:.0f}%")

    # === Demo 2: Reputation ===
    print("\n" + "=" * 50)
    print("DEMO 2: REPUTATION & SOCIAL CAPITAL")
    print("=" * 50)

    for agent_id in ["maya_001", "sofia_005"]:
        agent = engine.state.agents.get(agent_id)
        capital = engine.reputation_system.get_social_capital(agent_id)
        print(f"\n  {agent.name}:")
        print(f"    Reputation: {capital['reputation_score']:.2f}")
        print(f"    Social Capital: {capital['social_capital']:.2f}")

    # Record a favor
    favor = engine.reputation_system.record_favor(
        "jorge_002", "elena_003",
        "Brought her favorite coffee when she was stressed",
        0.3
    )
    print(f"\n  [Favor Recorded] Jorge helped Elena: {favor.description}")

    # === Demo 3: Rumors ===
    print("\n" + "=" * 50)
    print("DEMO 3: RUMORS & GOSSIP")
    print("=" * 50)

    # Create a rumor
    rumor = engine.gossip_network.create_rumor(
        RumorType.OBSERVATION,
        "diego_006",
        "Diego was seen meeting with a record producer",
        "sofia_005",
        juiciness=0.7
    )
    print(f"\n  [Rumor Created by Sofia]")
    print(f"    About: Diego")
    print(f"    Content: \"{rumor.current_content}\"")
    print(f"    Juiciness: {rumor.juiciness}")

    # Spread it
    engine.gossip_network.spread_rumor(rumor.rumor_id, "sofia_005", "maya_001", 0.6)
    engine.gossip_network.spread_rumor(rumor.rumor_id, "maya_001", "jorge_002", 0.5)

    print(f"\n  [Rumor Spread]")
    print(f"    Known by: {len(rumor.known_by)} people")
    print(f"    Distortion: {rumor.distortion_level*100:.0f}%")

    # === Demo 4: Secrets ===
    print("\n" + "=" * 50)
    print("DEMO 4: SECRETS & CONFIDENCES")
    print("=" * 50)

    # Show secrets from backstories
    for agent_id in ["maya_001", "carlos_004"]:
        agent = engine.state.agents.get(agent_id)
        secrets = [
            engine.secrets_manager.secrets[sid]
            for sid in engine.secrets_manager.agent_secrets.get(agent_id, set())
            if sid in engine.secrets_manager.secrets
        ]
        print(f"\n  {agent.name}'s Secrets:")
        for secret in secrets[:2]:
            print(f"    [{secret.secret_type.value}] \"{secret.content[:60]}...\"")
            print(f"      Sensitivity: {secret.sensitivity:.1f}")

    # === Demo 5: Group Project ===
    print("\n" + "=" * 50)
    print("DEMO 5: GROUP COLLABORATION")
    print("=" * 50)

    # Suggest collaborations
    suggestions = engine.group_projects.suggest_collaborations(
        "maya_001", engine.state.agents, DEEP_BACKSTORIES
    )
    print("\n  Collaboration suggestions for Maya:")
    for s in suggestions[:2]:
        partner = engine.state.agents.get(s['partner'])
        print(f"    With {partner.name if partner else s['partner']}: {s['idea']}")

    # Create a project
    project = engine.group_projects.create_project(
        "Sounds and Colors",
        "An audiovisual installation combining Maya's art with Diego's music",
        "maya_001",
        {"diego_006"}
    )
    project.add_participant("maya_001", "Visual Artist")
    project.add_participant("diego_006", "Composer")

    print(f"\n  [Project Created]")
    print(f"    Name: {project.name}")
    print(f"    Description: {project.description}")
    print(f"    Participants: {[engine.state.agents[p].name for p in project.participants]}")

    # === Demo 6: Storyline ===
    print("\n" + "=" * 50)
    print("DEMO 6: EMERGENT STORYLINE")
    print("=" * 50)

    if engine.storyline_engine:
        # Build relationships dict
        relationships = {}
        for agent_id, agent in engine.state.agents.items():
            if hasattr(agent, 'relationships'):
                relationships[agent_id] = {
                    other: rel.get('familiarity', 0.5)
                    for other, rel in agent.relationships.items()
                }

        storyline = await engine.storyline_engine.generate_storyline(
            engine.state.agents, DEEP_BACKSTORIES, relationships, []
        )

        if storyline:
            protagonist = engine.state.agents.get(storyline.protagonist_id)
            print(f"\n  [Storyline Generated]")
            print(f"    Title: {storyline.title}")
            print(f"    Theme: {storyline.theme}")
            print(f"    Protagonist: {protagonist.name if protagonist else 'Unknown'}")
            print(f"    Stakes: \"{storyline.stakes_description[:60]}...\"")
            if storyline.beats:
                print(f"    Inciting Incident: \"{storyline.beats[0]['description'][:60]}...\"")
    else:
        print("\n  [Storyline engine requires Ollama - showing structure]")
        print("    Would generate multi-day narrative arcs based on:")
        print("    - Character regrets, dreams, and fears")
        print("    - Relationship tensions")
        print("    - Recent events")

    # === Demo 7: Full Simulation Step ===
    print("\n" + "=" * 50)
    print("DEMO 7: COMPLETE SIMULATION STEP")
    print("=" * 50)

    results = await engine.simulate_social_step()

    print(f"\n  Weather: {results['weather']}")
    print(f"  Rumors spread: {len(results['rumors_spread'])}")
    print(f"  Secrets shared: {len(results['secrets_shared'])}")
    print(f"  Life events: {len(results['life_events'])}")
    print(f"  Storyline updates: {len(results['storyline_updates'])}")

    # === Demo 8: Day Advance ===
    print("\n" + "=" * 50)
    print("DEMO 8: ADVANCE TO NEXT DAY")
    print("=" * 50)

    new_day = await engine.advance_day()
    print(f"\n  New Day: {new_day['day']}")
    print(f"  Season: {new_day['season']}")
    print(f"  Weather: {new_day['weather']}")
    print(f"  Description: \"{new_day['weather_desc']}\"")
    if new_day['community_events']:
        print(f"  Community Events: {new_day['community_events']}")

    # === Demo 9: Social Summary ===
    print("\n" + "=" * 50)
    print("DEMO 9: AGENT SOCIAL SUMMARIES")
    print("=" * 50)

    for agent_id in ["maya_001", "diego_006"]:
        agent = engine.state.agents.get(agent_id)
        summary = engine.get_social_summary(agent_id)
        print(f"\n  {agent.name}:")
        print(f"    Reputation: {summary['reputation']['reputation_score']:.2f}")
        print(f"    Rumors Known: {summary['rumors_known']}")
        print(f"    Secrets Kept: {summary['secrets_kept']}")
        print(f"    Active Projects: {summary['active_projects']}")
        print(f"    Solitude Need: {summary['solitude_need']*100:.0f}%")

    await engine.close()

    print("\n" + "=" * 70)
    print("SOCIAL DYNAMICS DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_social_dynamics())
