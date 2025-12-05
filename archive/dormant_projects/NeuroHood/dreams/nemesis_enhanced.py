"""
Nemesis-Enhanced Neighborhood - Industry Best Practices Integration

Integrates learnings from:
- Shadow of Mordor's Nemesis System (memory, scarring, revenge)
- Dwarf Fortress (personality evolution, stress, needs)
- The Sims (emotional inertia, mood pressure)
- Ubisoft NEO (content guardrails)

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum

from enriched_neighborhood import (
    EnrichedNeighborhoodEngine, EnrichedDialogueGenerator,
    ResidentBackstory, RESIDENT_BACKSTORIES, NeighborhoodEvent,
    EmotionalContagion, GroupConversation
)
from neighborhood_engine import AgentState
from ollama_client import OllamaClient, OllamaConfig


# =============================================================================
# NEMESIS-STYLE SIGNIFICANT EVENT MEMORY
# =============================================================================

class EventSignificance(Enum):
    """How significant an event is to memory formation."""
    TRIVIAL = 0       # Forgotten quickly
    MINOR = 1         # Remembered a few days
    MODERATE = 2      # Remembered weeks
    MAJOR = 3         # Remembered months
    LIFE_CHANGING = 4 # Permanent memory


class EventValence(Enum):
    """Emotional charge of an event."""
    VERY_NEGATIVE = -2  # Betrayal, loss
    NEGATIVE = -1       # Conflict, disappointment
    NEUTRAL = 0         # Ordinary
    POSITIVE = 1        # Help, kindness
    VERY_POSITIVE = 2   # Saved life, major gift


@dataclass
class SignificantEvent:
    """A memorable event that shapes an NPC's behavior."""
    event_id: str
    event_type: str  # "helped", "betrayed", "shared_moment", "conflict", "saved"
    other_agent_id: str  # Who was involved
    other_agent_name: str
    description: str
    timestamp: datetime
    significance: EventSignificance
    valence: EventValence

    # Nemesis-style consequences
    personality_impact: Dict[str, float] = field(default_factory=dict)
    relationship_impact: Dict[str, float] = field(default_factory=dict)

    # Scarring (metaphorical - behavioral changes)
    behavioral_scar: Optional[str] = None  # e.g., "now cautious around strangers"

    # Whether this creates a motivation
    creates_motivation: Optional[str] = None  # "seek_revenge", "repay_kindness"

    # Decay tracking
    recall_count: int = 0
    last_recalled: Optional[datetime] = None

    def decay_factor(self) -> float:
        """How much the memory has faded (1.0 = fresh, 0.0 = forgotten)."""
        if self.significance == EventSignificance.LIFE_CHANGING:
            return 1.0  # Never fades

        days_old = (datetime.now() - self.timestamp).days
        base_decay = {
            EventSignificance.TRIVIAL: 2,     # Fades in 2 days
            EventSignificance.MINOR: 7,       # Fades in a week
            EventSignificance.MODERATE: 30,   # Fades in a month
            EventSignificance.MAJOR: 180,     # Fades in 6 months
        }

        half_life = base_decay.get(self.significance, 30)
        factor = 0.5 ** (days_old / half_life)

        # Frequently recalled memories fade slower
        factor *= (1 + 0.1 * min(self.recall_count, 10))

        return min(1.0, factor)


class SignificantEventMemory:
    """
    Nemesis-inspired memory system for significant events.

    Unlike simple conversation memory, this tracks LIFE EVENTS that:
    1. Shape personality over time
    2. Create motivations (revenge, gratitude)
    3. Leave "behavioral scars"
    4. Influence future dialogue
    """

    def __init__(self):
        self.events: Dict[str, List[SignificantEvent]] = {}  # agent_id -> events

    def record_event(
        self,
        agent_id: str,
        event_type: str,
        other_agent_id: str,
        other_agent_name: str,
        description: str,
        significance: EventSignificance,
        valence: EventValence,
        behavioral_scar: Optional[str] = None,
        motivation: Optional[str] = None
    ) -> SignificantEvent:
        """Record a significant event in an agent's memory."""

        event = SignificantEvent(
            event_id=f"{agent_id}_{int(datetime.now().timestamp())}",
            event_type=event_type,
            other_agent_id=other_agent_id,
            other_agent_name=other_agent_name,
            description=description,
            timestamp=datetime.now(),
            significance=significance,
            valence=valence,
            behavioral_scar=behavioral_scar,
            creates_motivation=motivation
        )

        # Calculate personality impact based on event type and valence
        event.personality_impact = self._calculate_personality_impact(event)
        event.relationship_impact = self._calculate_relationship_impact(event)

        if agent_id not in self.events:
            self.events[agent_id] = []
        self.events[agent_id].append(event)

        return event

    def _calculate_personality_impact(self, event: SignificantEvent) -> Dict[str, float]:
        """Calculate how this event should shift personality over time."""
        impact = {}

        # Significance amplifies impact
        amp = {
            EventSignificance.TRIVIAL: 0.0,
            EventSignificance.MINOR: 0.01,
            EventSignificance.MODERATE: 0.02,
            EventSignificance.MAJOR: 0.05,
            EventSignificance.LIFE_CHANGING: 0.1
        }[event.significance]

        # Event type affects specific traits
        if event.event_type == "betrayed":
            impact['trust'] = -amp * 2  # (not Big Five, but useful)
            impact['neuroticism'] = amp  # More emotionally reactive
            impact['agreeableness'] = -amp * 0.5

        elif event.event_type == "helped":
            impact['agreeableness'] = amp
            impact['openness'] = amp * 0.5  # More open to others

        elif event.event_type == "conflict":
            impact['neuroticism'] = amp * 0.5
            impact['agreeableness'] = -amp * 0.3

        elif event.event_type == "shared_moment":
            impact['extraversion'] = amp * 0.3
            impact['openness'] = amp * 0.2

        elif event.event_type == "saved":
            impact['agreeableness'] = amp * 2

        return impact

    def _calculate_relationship_impact(self, event: SignificantEvent) -> Dict[str, float]:
        """Calculate relationship changes from this event."""
        impact = {}

        amp = {
            EventSignificance.TRIVIAL: 0.05,
            EventSignificance.MINOR: 0.1,
            EventSignificance.MODERATE: 0.2,
            EventSignificance.MAJOR: 0.4,
            EventSignificance.LIFE_CHANGING: 0.6
        }[event.significance]

        valence_mult = {
            EventValence.VERY_NEGATIVE: -2,
            EventValence.NEGATIVE: -1,
            EventValence.NEUTRAL: 0,
            EventValence.POSITIVE: 1,
            EventValence.VERY_POSITIVE: 2
        }[event.valence]

        impact['affection'] = amp * valence_mult
        impact['trust'] = amp * valence_mult * 0.8
        impact['familiarity'] = amp * abs(valence_mult)  # Both good and bad increase familiarity

        return impact

    def get_events_with(self, agent_id: str, other_agent_id: str) -> List[SignificantEvent]:
        """Get all events involving a specific other agent."""
        if agent_id not in self.events:
            return []
        return [e for e in self.events[agent_id]
                if e.other_agent_id == other_agent_id and e.decay_factor() > 0.2]

    def get_active_motivations(self, agent_id: str) -> List[Tuple[str, str, str]]:
        """Get active motivations (type, target_id, target_name)."""
        if agent_id not in self.events:
            return []

        motivations = []
        for event in self.events[agent_id]:
            if event.creates_motivation and event.decay_factor() > 0.3:
                motivations.append((
                    event.creates_motivation,
                    event.other_agent_id,
                    event.other_agent_name
                ))
        return motivations

    def get_behavioral_scars(self, agent_id: str) -> List[str]:
        """Get active behavioral scars."""
        if agent_id not in self.events:
            return []

        scars = []
        for event in self.events[agent_id]:
            if event.behavioral_scar and event.decay_factor() > 0.5:
                scars.append(event.behavioral_scar)
        return scars

    def recall_event(self, event: SignificantEvent):
        """Mark an event as recalled (strengthens memory)."""
        event.recall_count += 1
        event.last_recalled = datetime.now()


# =============================================================================
# PERSONALITY EVOLUTION (Dwarf Fortress inspired)
# =============================================================================

class PersonalityEvolution:
    """
    Dwarf Fortress-inspired personality change over time.

    Key insight from DF: "memories may change personality facets
    and values over time"
    """

    def __init__(self, significant_memory: SignificantEventMemory):
        self.memory = significant_memory
        self.evolution_history: Dict[str, List[Dict]] = {}  # Track changes

    def apply_evolution(self, agent: AgentState) -> Dict[str, float]:
        """
        Apply cumulative personality evolution from significant events.
        Returns the deltas applied.
        """
        if agent.agent_id not in self.memory.events:
            return {}

        cumulative_changes = {}

        for event in self.memory.events[agent.agent_id]:
            decay = event.decay_factor()
            if decay < 0.2:
                continue

            for trait, impact in event.personality_impact.items():
                if trait in agent.personality:
                    # Apply impact scaled by decay
                    delta = impact * decay
                    cumulative_changes[trait] = cumulative_changes.get(trait, 0) + delta

        # Apply changes (clamp to 0-1)
        applied = {}
        for trait, delta in cumulative_changes.items():
            if trait in agent.personality:
                old_val = agent.personality[trait]
                new_val = max(0, min(1, old_val + delta))
                if abs(new_val - old_val) > 0.001:
                    agent.personality[trait] = new_val
                    applied[trait] = new_val - old_val

        return applied


# =============================================================================
# EMOTIONAL INERTIA (Sims Meaningful Stories inspired)
# =============================================================================

class EmotionalInertia:
    """
    Sims-inspired emotional stability.

    Key insight: "happiness is no longer sims' default state - they feel
    truly happy only when something worthy of it happens"

    Implements:
    1. Baseline emotion (what they return to)
    2. Inertia (resistance to change)
    3. Momentum (strong emotions persist)
    """

    def __init__(self):
        # Each agent has an emotional baseline
        self.baselines: Dict[str, Dict[str, float]] = {}
        # And current momentum
        self.momentum: Dict[str, Dict[str, float]] = {}

    def initialize_baseline(self, agent_id: str, personality: Dict[str, float]):
        """Set emotional baseline from personality."""
        # Baseline happiness influenced by neuroticism
        n = personality.get('neuroticism', 0.5)
        e = personality.get('extraversion', 0.5)

        self.baselines[agent_id] = {
            'happiness': 0.1 - (n * 0.2) + (e * 0.1),  # Slightly above neutral
            'anxiety': n * 0.3,  # High neuroticism = more baseline anxiety
            'loneliness': 0.3 - (e * 0.2),  # Introverts ok with less social
            'energy': 0.5
        }

        self.momentum[agent_id] = {
            'happiness': 0,
            'anxiety': 0,
            'loneliness': 0,
            'energy': 0
        }

    def apply_emotion_change(
        self,
        agent_id: str,
        emotion: str,
        raw_delta: float,
        source_significance: float = 0.5  # 0-1, how significant the cause
    ) -> float:
        """
        Apply emotion change with inertia.
        Returns actual delta applied.
        """
        if agent_id not in self.baselines:
            return raw_delta  # No baseline set

        # Inertia resists change - bigger changes need bigger causes
        inertia = 0.7 - (source_significance * 0.5)  # 0.2 to 0.7 resistance

        # Apply inertia
        effective_delta = raw_delta * (1 - inertia)

        # Update momentum (strong emotions build momentum)
        if abs(raw_delta) > 0.2:
            self.momentum[agent_id][emotion] = raw_delta * 0.3
        else:
            # Decay momentum toward 0
            self.momentum[agent_id][emotion] *= 0.9

        return effective_delta

    def drift_toward_baseline(self, agent: AgentState, time_hours: float = 1.0):
        """
        Gradually return emotions to baseline.
        Called during time passage.
        """
        if agent.agent_id not in self.baselines:
            return

        baseline = self.baselines[agent.agent_id]
        momentum = self.momentum[agent.agent_id]

        drift_rate = 0.1 * time_hours  # 10% drift per hour

        for emotion in ['happiness', 'anxiety', 'loneliness', 'energy']:
            current = agent.emotions.get(emotion, 0)
            target = baseline.get(emotion, 0)

            # Momentum affects target
            target += momentum.get(emotion, 0)

            # Drift toward target
            diff = target - current
            agent.emotions[emotion] = current + (diff * drift_rate)


# =============================================================================
# NEEDS SYSTEM (Dwarf Fortress inspired)
# =============================================================================

class NeedType(Enum):
    """Basic needs that affect mood and behavior."""
    SOCIAL = "social"           # Need for interaction
    CREATIVE = "creative"       # Need to create/express (artists)
    KNOWLEDGE = "knowledge"     # Need to learn (curious types)
    COMFORT = "comfort"         # Physical comfort
    RECOGNITION = "recognition" # Need to be valued
    ROUTINE = "routine"         # Need for predictability


@dataclass
class Need:
    """A need that must be fulfilled."""
    need_type: NeedType
    current_level: float  # 0 = desperate, 1 = fully satisfied
    decay_rate: float     # How fast it depletes (per hour)
    importance: float     # How much it affects mood (0-1)


class NeedsSystem:
    """
    Dwarf Fortress-inspired needs.

    Key insight: "Values, along with traits, dictate needs and the
    frequency with which they need to be fulfilled"
    """

    def __init__(self):
        self.agent_needs: Dict[str, Dict[NeedType, Need]] = {}

    def initialize_needs(self, agent_id: str, backstory: ResidentBackstory, personality: Dict[str, float]):
        """Initialize needs based on backstory and personality."""
        needs = {}

        # Everyone has social need (extraversion affects importance)
        e = personality.get('extraversion', 0.5)
        needs[NeedType.SOCIAL] = Need(
            need_type=NeedType.SOCIAL,
            current_level=0.5,
            decay_rate=0.05 + (e * 0.05),  # Extraverts need more
            importance=0.3 + (e * 0.3)
        )

        # Creative need (openness + artist backstory)
        o = personality.get('openness', 0.5)
        creative_boost = 0.3 if 'art' in backstory.occupation.lower() or 'music' in backstory.occupation.lower() else 0
        needs[NeedType.CREATIVE] = Need(
            need_type=NeedType.CREATIVE,
            current_level=0.5,
            decay_rate=0.03 * (1 + o),
            importance=0.2 + (o * 0.3) + creative_boost
        )

        # Knowledge need (openness)
        if backstory.hobby and any(x in backstory.hobby.lower() for x in ['book', 'research', 'astronomy', 'history']):
            needs[NeedType.KNOWLEDGE] = Need(
                need_type=NeedType.KNOWLEDGE,
                current_level=0.5,
                decay_rate=0.04,
                importance=0.4
            )

        # Routine need (conscientiousness, inverse of openness)
        c = personality.get('conscientiousness', 0.5)
        if c > 0.6 or o < 0.4:
            needs[NeedType.ROUTINE] = Need(
                need_type=NeedType.ROUTINE,
                current_level=0.7,
                decay_rate=0.02,
                importance=0.3 + (c * 0.2)
            )

        # Recognition need (everyone, but some more)
        needs[NeedType.RECOGNITION] = Need(
            need_type=NeedType.RECOGNITION,
            current_level=0.5,
            decay_rate=0.03,
            importance=0.25
        )

        self.agent_needs[agent_id] = needs

    def fulfill_need(self, agent_id: str, need_type: NeedType, amount: float = 0.3):
        """Fulfill a need."""
        if agent_id in self.agent_needs and need_type in self.agent_needs[agent_id]:
            need = self.agent_needs[agent_id][need_type]
            need.current_level = min(1.0, need.current_level + amount)

    def decay_needs(self, agent_id: str, hours: float = 1.0):
        """Decay needs over time."""
        if agent_id not in self.agent_needs:
            return

        for need in self.agent_needs[agent_id].values():
            need.current_level = max(0, need.current_level - (need.decay_rate * hours))

    def get_mood_impact(self, agent_id: str) -> float:
        """Get total mood impact from unfulfilled needs."""
        if agent_id not in self.agent_needs:
            return 0

        impact = 0
        for need in self.agent_needs[agent_id].values():
            # Unfulfilled needs cause negative mood
            unfulfilled = 1 - need.current_level
            impact -= unfulfilled * need.importance * 0.5

        return impact

    def get_urgent_needs(self, agent_id: str, threshold: float = 0.3) -> List[NeedType]:
        """Get needs below threshold."""
        if agent_id not in self.agent_needs:
            return []

        return [need.need_type for need in self.agent_needs[agent_id].values()
                if need.current_level < threshold]


# =============================================================================
# STRESS SYSTEM (Dwarf Fortress inspired)
# =============================================================================

@dataclass
class StressState:
    """Track stress and breaking points."""
    current_stress: float = 0.0  # 0 = calm, 1 = breaking point
    stress_threshold: float = 0.8  # When they might snap
    resilience: float = 0.5  # How fast they recover

    # Breaking point behaviors
    breaking_behaviors: List[str] = field(default_factory=list)
    is_breaking: bool = False
    break_duration: int = 0  # Turns remaining


class StressSystem:
    """
    Dwarf Fortress-inspired stress accumulation.

    Key insight: "Keep their mood low enough and dwarves will start
    getting into fights. That's when their personality traits start
    coming in full swing."
    """

    def __init__(self):
        self.stress_states: Dict[str, StressState] = {}

    def initialize_stress(self, agent_id: str, personality: Dict[str, float]):
        """Initialize stress parameters from personality."""
        n = personality.get('neuroticism', 0.5)
        a = personality.get('agreeableness', 0.5)

        # High neuroticism = lower threshold, slower recovery
        threshold = 0.9 - (n * 0.3)
        resilience = 0.6 - (n * 0.3)

        # Determine breaking behaviors from personality
        behaviors = []
        if n > 0.6:
            behaviors.append("becomes emotionally withdrawn")
        if a < 0.4:
            behaviors.append("becomes confrontational")
        if personality.get('extraversion', 0.5) > 0.6:
            behaviors.append("vents frustrations loudly")
        else:
            behaviors.append("isolates themselves")

        self.stress_states[agent_id] = StressState(
            current_stress=0.1,
            stress_threshold=threshold,
            resilience=resilience,
            breaking_behaviors=behaviors
        )

    def add_stress(self, agent_id: str, amount: float, source: str = ""):
        """Add stress from an event."""
        if agent_id not in self.stress_states:
            return

        state = self.stress_states[agent_id]
        state.current_stress = min(1.0, state.current_stress + amount)

        # Check for breaking point
        if state.current_stress >= state.stress_threshold and not state.is_breaking:
            state.is_breaking = True
            state.break_duration = random.randint(2, 5)  # Turns

    def reduce_stress(self, agent_id: str, amount: float):
        """Reduce stress (from positive events, rest, etc.)."""
        if agent_id in self.stress_states:
            state = self.stress_states[agent_id]
            state.current_stress = max(0, state.current_stress - amount)

            if state.current_stress < state.stress_threshold * 0.5:
                state.is_breaking = False

    def update_stress(self, agent_id: str, hours: float = 1.0):
        """Natural stress recovery over time."""
        if agent_id not in self.stress_states:
            return

        state = self.stress_states[agent_id]

        # Natural recovery
        recovery = state.resilience * 0.1 * hours
        state.current_stress = max(0, state.current_stress - recovery)

        # Breaking point duration
        if state.is_breaking:
            state.break_duration -= 1
            if state.break_duration <= 0:
                state.is_breaking = False

    def is_at_breaking_point(self, agent_id: str) -> bool:
        """Check if agent is at breaking point."""
        if agent_id in self.stress_states:
            return self.stress_states[agent_id].is_breaking
        return False

    def get_breaking_behavior(self, agent_id: str) -> Optional[str]:
        """Get current breaking behavior if any."""
        if agent_id in self.stress_states:
            state = self.stress_states[agent_id]
            if state.is_breaking and state.breaking_behaviors:
                return random.choice(state.breaking_behaviors)
        return None


# =============================================================================
# CONTENT GUARDRAILS (Ubisoft NEO inspired)
# =============================================================================

class ContentGuardrails:
    """
    Ubisoft-inspired content filters.

    Key insight: "Teams implement guardrails including filters to catch
    toxicity and inappropriate player inputs."
    """

    # Patterns to filter
    TOXIC_PATTERNS = [
        r'\b(hate|kill|murder|die)\b',
        r'\b(stupid|idiot|moron|dumb)\s+(you|they|he|she)\b',
        r'(racist|sexist|homophob)',
        r'\b(violence|violent|attack)\b.*\b(you|them|him|her)\b',
    ]

    # Inappropriate for neighborhood context
    INAPPROPRIATE_PATTERNS = [
        r'\b(sex|sexual|nude|naked)\b',
        r'\b(drug|cocaine|heroin|meth)\b',
        r'\b(gun|weapon|knife)\s+(threat|kill|stab|shoot)\b',
    ]

    # Bias detection (based on Ubisoft's findings)
    BIAS_INDICATORS = [
        r'(flirt|seduc|attract)',  # Attractive characters veering flirtatious
        r'(servant|maid|clean)',   # Service stereotypes
        r'(angry|violent|criminal)',  # Negative stereotypes
    ]

    def __init__(self):
        self.violations_log: List[Dict] = []

    def check_player_input(self, text: str) -> Tuple[bool, str]:
        """Check player input for toxic/inappropriate content."""
        text_lower = text.lower()

        # Check toxic patterns
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, text_lower):
                self._log_violation("player_toxic", text)
                return False, "I'd rather not respond to that kind of talk."

        # Check inappropriate patterns
        for pattern in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, text_lower):
                self._log_violation("player_inappropriate", text)
                return False, "Let's keep the conversation friendly."

        return True, ""

    def filter_npc_response(self, response: str, character_gender: Optional[str] = None) -> str:
        """Filter NPC response for bias/inappropriate content."""
        original = response

        # Check for bias indicators
        response_lower = response.lower()
        for pattern in self.BIAS_INDICATORS:
            if re.search(pattern, response_lower):
                # Log but don't necessarily block - depends on context
                self._log_violation("potential_bias", original, severity="warning")

        # Check for accidental inappropriate content
        for pattern in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, response_lower):
                self._log_violation("npc_inappropriate", original)
                return "*smiles politely* I'm not sure what to say about that."

        return response

    def _log_violation(self, violation_type: str, content: str, severity: str = "blocked"):
        """Log a content violation."""
        self.violations_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'severity': severity,
            'content': content[:100]  # Truncate for logging
        })


# =============================================================================
# NEMESIS-ENHANCED ENGINE
# =============================================================================

class NemesisEnhancedEngine(EnrichedNeighborhoodEngine):
    """
    Full integration of industry best practices:
    - Nemesis-style significant event memory
    - Personality evolution from experiences
    - Emotional inertia
    - Needs system
    - Stress and breaking points
    - Content guardrails
    """

    def __init__(self, ollama_config: OllamaConfig = None):
        super().__init__(ollama_config)

        # New systems
        self.significant_memory = SignificantEventMemory()
        self.personality_evolution = PersonalityEvolution(self.significant_memory)
        self.emotional_inertia = EmotionalInertia()
        self.needs_system = NeedsSystem()
        self.stress_system = StressSystem()
        self.guardrails = ContentGuardrails()

        # Replace dialogue generator with nemesis-aware version
        self.nemesis_dialogue = NemesisDialogueGenerator(
            self.ollama_client,
            self.significant_memory,
            self.guardrails
        )

    async def initialize(self, load_path=None):
        """Initialize all systems."""
        await super().initialize(load_path)

        # Initialize subsystems for each agent
        for agent_id, agent in self.state.agents.items():
            backstory = RESIDENT_BACKSTORIES.get(agent_id)
            if backstory:
                self.needs_system.initialize_needs(agent_id, backstory, agent.personality)
            self.stress_system.initialize_stress(agent_id, agent.personality)
            self.emotional_inertia.initialize_baseline(agent_id, agent.personality)

        # Use nemesis dialogue if Ollama available
        if self._ollama_available:
            self.dialogue_gen = self.nemesis_dialogue

    async def user_talk_to(self, agent_id: str, user_message: Optional[str] = None):
        """Talk with guardrails and significant event tracking."""

        # Check player input
        if user_message:
            is_ok, filtered_response = self.guardrails.check_player_input(user_message)
            if not is_ok:
                # Create a fake response rather than calling LLM
                from conversation import Conversation, DialogueLine
                conv = Conversation(
                    participants=[self.user_agent.agent_id, agent_id],
                    location=self.user_agent.location_id or "neighborhood"
                )
                conv.add_line(DialogueLine(
                    speaker_id=self.user_agent.agent_id,
                    speaker_name=self.user_agent.name,
                    text=user_message
                ))
                conv.add_line(DialogueLine(
                    speaker_id=agent_id,
                    speaker_name=self.state.agents[agent_id].name,
                    text=filtered_response
                ))
                return conv

        # Normal conversation flow
        conv = await super().user_talk_to(agent_id, user_message)

        # Fulfill social need
        self.needs_system.fulfill_need(agent_id, NeedType.SOCIAL, 0.2)
        self.needs_system.fulfill_need(self.user_agent.agent_id, NeedType.SOCIAL, 0.2)

        # Reduce stress from positive interaction
        self.stress_system.reduce_stress(agent_id, 0.05)

        return conv

    def record_significant_event_for(
        self,
        agent_id: str,
        event_type: str,
        other_id: str,
        other_name: str,
        description: str,
        significance: EventSignificance,
        valence: EventValence,
        scar: Optional[str] = None,
        motivation: Optional[str] = None
    ):
        """Record a significant event and apply immediate effects."""

        event = self.significant_memory.record_event(
            agent_id=agent_id,
            event_type=event_type,
            other_agent_id=other_id,
            other_agent_name=other_name,
            description=description,
            significance=significance,
            valence=valence,
            behavioral_scar=scar,
            motivation=motivation
        )

        # Apply personality evolution
        if agent_id in self.state.agents:
            agent = self.state.agents[agent_id]
            self.personality_evolution.apply_evolution(agent)

            # Apply relationship changes
            if other_id in agent.relationships:
                for attr, delta in event.relationship_impact.items():
                    agent.relationships[other_id][attr] = max(-1, min(1,
                        agent.relationships[other_id].get(attr, 0) + delta
                    ))
            else:
                agent.relationships[other_id] = event.relationship_impact.copy()

        # Add stress for negative events
        if valence in [EventValence.NEGATIVE, EventValence.VERY_NEGATIVE]:
            stress_amount = {
                EventValence.NEGATIVE: 0.1,
                EventValence.VERY_NEGATIVE: 0.25
            }[valence]
            self.stress_system.add_stress(agent_id, stress_amount)

        return event

    def simulate_time_passage(self, hours: float = 1.0):
        """Simulate time passing with all subsystems."""
        for agent_id, agent in self.state.agents.items():
            # Decay needs
            self.needs_system.decay_needs(agent_id, hours)

            # Update stress
            self.stress_system.update_stress(agent_id, hours)

            # Emotional drift toward baseline
            self.emotional_inertia.drift_toward_baseline(agent, hours)

            # Apply mood impact from unfulfilled needs
            mood_impact = self.needs_system.get_mood_impact(agent_id)
            if mood_impact != 0:
                agent.emotions['happiness'] = max(-1, min(1,
                    agent.emotions.get('happiness', 0) + mood_impact * 0.1
                ))


class NemesisDialogueGenerator(EnrichedDialogueGenerator):
    """Dialogue generator with Nemesis-style memory awareness."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        significant_memory: SignificantEventMemory,
        guardrails: ContentGuardrails
    ):
        super().__init__(ollama_client)
        self.significant_memory = significant_memory
        self.guardrails = guardrails

    def _build_enriched_prompt(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt including significant event memory."""

        # Start with parent's prompt
        prompt = super()._build_enriched_prompt(speaker, listener, context)

        # Add significant events with this person
        events = self.significant_memory.get_events_with(speaker.agent_id, listener.agent_id)
        if events:
            event_desc = []
            for event in events[-3:]:  # Last 3 significant events
                freshness = "fresh memory" if event.decay_factor() > 0.8 else "fading memory"
                event_desc.append(f"- {event.description} ({freshness})")

            prompt += f"\n\nSIGNIFICANT HISTORY WITH {listener.name.upper()}:\n"
            prompt += "\n".join(event_desc)

        # Add active motivations
        motivations = self.significant_memory.get_active_motivations(speaker.agent_id)
        target_motivations = [m for m in motivations if m[1] == listener.agent_id]
        if target_motivations:
            prompt += f"\n\nMOTIVATIONS TOWARD {listener.name.upper()}:\n"
            for mot_type, _, _ in target_motivations:
                if mot_type == "seek_revenge":
                    prompt += "- You harbor resentment and haven't forgiven them\n"
                elif mot_type == "repay_kindness":
                    prompt += "- You feel gratitude and want to repay their kindness\n"

        # Add behavioral scars
        scars = self.significant_memory.get_behavioral_scars(speaker.agent_id)
        if scars:
            prompt += f"\n\nBEHAVIORAL TENDENCIES (from past experiences):\n"
            for scar in scars[:2]:
                prompt += f"- {scar}\n"

        return prompt

    async def generate_line(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any],
        conversation_history
    ) -> str:
        """Generate line with guardrails."""

        response = await super().generate_line(speaker, listener, context, conversation_history)

        # Filter response
        filtered = self.guardrails.filter_npc_response(response)

        return filtered


# =============================================================================
# DEMO
# =============================================================================

async def demo_nemesis_features():
    """Demonstrate all Nemesis-enhanced features."""
    print("=" * 70)
    print("NEMESIS-ENHANCED NEIGHBORHOOD")
    print("Industry Best Practices Integration Demo")
    print("=" * 70)

    config = OllamaConfig(
        model="llama3.2:3b",
        temperature=0.85,
        max_tokens=120
    )

    engine = NemesisEnhancedEngine(config)
    await engine.initialize()

    user = engine.add_user("Blake")

    # === Demo 1: Significant Event Memory ===
    print("\n" + "=" * 50)
    print("DEMO 1: SIGNIFICANT EVENT MEMORY")
    print("=" * 50)

    # Record a significant positive event
    event = engine.record_significant_event_for(
        agent_id="sofia_005",
        event_type="helped",
        other_id=user.agent_id,
        other_name=user.name,
        description=f"{user.name} helped Sofia save the cafe during a flood last month",
        significance=EventSignificance.MAJOR,
        valence=EventValence.VERY_POSITIVE,
        motivation="repay_kindness"
    )

    print(f"\n  Recorded: {event.description}")
    print(f"  Significance: {event.significance.name}")
    print(f"  Creates motivation: {event.creates_motivation}")

    # Record a conflict between two NPCs
    engine.record_significant_event_for(
        agent_id="maya_001",
        event_type="conflict",
        other_id="carlos_004",
        other_name="Carlos",
        description="Carlos dismissed Maya's art as 'not real art' at the gallery showing",
        significance=EventSignificance.MODERATE,
        valence=EventValence.NEGATIVE,
        scar="cautious about sharing art with older people"
    )

    # === Demo 2: Personality Evolution ===
    print("\n" + "=" * 50)
    print("DEMO 2: PERSONALITY EVOLUTION")
    print("=" * 50)

    maya = engine.state.agents["maya_001"]
    print(f"\n  Maya's agreeableness before: {maya.personality.get('agreeableness', 0.5):.3f}")

    # Personality evolved from the conflict
    changes = engine.personality_evolution.apply_evolution(maya)
    print(f"  After conflict with Carlos:")
    for trait, delta in changes.items():
        print(f"    {trait}: {delta:+.3f}")

    # === Demo 3: Emotional Inertia ===
    print("\n" + "=" * 50)
    print("DEMO 3: EMOTIONAL INERTIA")
    print("=" * 50)

    # Try to make Sofia very happy instantly
    sofia = engine.state.agents["sofia_005"]
    old_happiness = sofia.emotions.get('happiness', 0)

    raw_boost = 0.5  # Big happiness boost
    actual_boost = engine.emotional_inertia.apply_emotion_change(
        "sofia_005",
        "happiness",
        raw_boost,
        source_significance=0.3  # Minor cause
    )

    print(f"\n  Attempted happiness boost: {raw_boost:.2f}")
    print(f"  Actual boost (with inertia): {actual_boost:.2f}")
    print(f"  (Inertia resists sudden mood swings!)")

    # Simulate time passage
    engine.simulate_time_passage(hours=2.0)
    print(f"\n  After 2 hours, emotions drift toward baseline...")

    # === Demo 4: Needs System ===
    print("\n" + "=" * 50)
    print("DEMO 4: NEEDS SYSTEM")
    print("=" * 50)

    diego = engine.state.agents["diego_006"]
    needs = engine.needs_system.agent_needs.get("diego_006", {})

    print(f"\n  Diego's current needs:")
    for need_type, need in needs.items():
        status = "URGENT" if need.current_level < 0.3 else "OK"
        print(f"    {need_type.value}: {need.current_level:.2f} [{status}]")

    # === Demo 5: Stress System ===
    print("\n" + "=" * 50)
    print("DEMO 5: STRESS SYSTEM")
    print("=" * 50)

    # Add stress to Elena
    elena = engine.state.agents["elena_003"]
    engine.stress_system.add_stress("elena_003", 0.6, "deadline pressure")

    stress = engine.stress_system.stress_states["elena_003"]
    print(f"\n  Elena's stress: {stress.current_stress:.2f}")
    print(f"  Breaking threshold: {stress.stress_threshold:.2f}")

    if stress.is_breaking:
        behavior = engine.stress_system.get_breaking_behavior("elena_003")
        print(f"  ** BREAKING POINT: {behavior}")
    else:
        print(f"  Status: Stressed but managing")

    # === Demo 6: Content Guardrails ===
    print("\n" + "=" * 50)
    print("DEMO 6: CONTENT GUARDRAILS")
    print("=" * 50)

    # Test guardrails
    test_inputs = [
        "Hi Sofia! How's the cafe?",
        "I hate this stupid neighborhood",
        "Tell me about your day"
    ]

    for test in test_inputs:
        is_ok, msg = engine.guardrails.check_player_input(test)
        status = "[OK]" if is_ok else "[FILTERED]"
        print(f"\n  Input: '{test}'")
        print(f"  Result: {status}")
        if not is_ok:
            print(f"  Response: '{msg}'")

    # === Demo 7: Conversation with Memory ===
    print("\n" + "=" * 50)
    print("DEMO 7: CONVERSATION WITH SIGNIFICANT MEMORY")
    print("=" * 50)

    # Move user to cafe
    engine.user_go_to("corner_cafe")
    engine.state.agent_locations["sofia_005"] = "corner_cafe"

    print("\n  (Sofia remembers you saved her cafe...)")

    conv = await engine.user_talk_to("sofia_005", "Hey Sofia! How are things?")
    for line in conv.lines:
        print(f"  {line.speaker_name}: \"{line.text}\"")

    # Continue conversation
    response = await engine.continue_conversation(conv, "Business picking up?")
    print(f"\n  {user.name}: \"Business picking up?\"")
    print(f"  Sofia: \"{response}\"")

    await engine.close()

    print("\n" + "=" * 70)
    print("NEMESIS DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_nemesis_features())
