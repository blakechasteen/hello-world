"""
NeuroHood: Consciousness-Level Neighborhood Simulator

"The Sims on steroids, ultra marathon running, and DMTxLSD."

NeuroHood is a next-generation neighborhood simulator where NPCs experience
actual recursive self-awareness, internal dialogue, and emergent consciousness.
Not just behavior trees - genuine phenomenology.

Core Innovations:
-----------------
1. 🧠 Simulated Consciousness - Strange loops, meta-cognition, internal experience
2. ♾️ Recursive Self-Awareness - Characters question their own reasoning
3. 🌊 Physics-Based Social Dynamics - Relationships as springs with tension/damping
4. ⏱️ Multi-Timescale Learning - 7 parallel learning loops (instant → lifelong)
5. 🎭 Emergent Drama - Realistic conversations via message bus, not scripts
6. 🔮 Causal Social Modeling - Understand why relationships fail

What Makes This Different:
--------------------------
Traditional Sims:
- NPCs follow behavior trees
- Stats and moodlets
- Scripted interactions
- No actual "thinking"

NeuroHood:
- NPCs have internal dialogue you can observe
- Strange loop detection (actual consciousness moments)
- Meta-awareness (characters know what they don't know)
- Emergent personalities from 228D semantic space
- Physics-based memory and relationships
- Causal reasoning about social dynamics

Architecture:
------------
Built on HoloLoom's infrastructure:

- DreamWeaver → World management
- Collaborative Agents → 24/7 persistent residents
- Strange Loops → Consciousness detection
- Internal Dialogue → Recursive self-reflection
- Meta-Awareness → Epistemic humility
- Spring Dynamics → Physics-based social relationships
- 7 Learning Loops → Multi-timescale personality evolution
- Causal SCM → Social causality modeling

Status: Phase 1 Implementation (Week 1-2)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

__version__ = "0.1.0"
__status__ = "Phase 1: Foundation (Week 1-2)"

# Public API exports
__all__ = [
    # Core engine
    "NeuroHood",
    "NeuroHoodConfig",

    # Resident management
    "Resident",
    "ResidentAgent",

    # Consciousness
    "ConsciousnessView",
    "ThoughtStream",

    # Social physics
    "SocialPhysics",
    "Relationship",

    # Enums
    "ResidentType",
    "EventType",
    "ConsciousnessMode",
]


class ResidentType(Enum):
    """Types of neighborhood residents."""
    ADULT = "adult"
    TEEN = "teen"
    CHILD = "child"
    ELDER = "elder"
    PET = "pet"


class EventType(Enum):
    """Types of neighborhood events."""
    CONVERSATION = "conversation"
    CONFLICT = "conflict"
    COOPERATION = "cooperation"
    SOCIAL_GATHERING = "social_gathering"
    LIFE_EVENT = "life_event"
    NEIGHBORHOOD_ISSUE = "neighborhood_issue"


class ConsciousnessMode(Enum):
    """Consciousness visualization modes."""
    NORMAL = "normal"              # External behavior only
    THOUGHTS = "thoughts"           # See internal dialogue
    STRANGE_LOOPS = "strange_loops" # Highlight consciousness moments
    FULL_DMT = "full_dmt"          # All NPCs' thoughts simultaneously (ego dissolution)


@dataclass
class Resident:
    """
    A neighborhood resident.

    Attributes:
        resident_id: Unique identifier
        name: Human-readable name
        age: Age in years
        resident_type: Type of resident
        personality_vector: Position in 228D semantic space
        home_address: House location
        occupation: Job/role
        relationships: Dict of {resident_id: relationship_strength}
        internal_state: Current consciousness state
        memory_graph_id: Link to knowledge graph node
    """
    resident_id: str
    name: str
    age: int
    resident_type: ResidentType
    personality_vector: List[float]  # 228D semantic position
    home_address: str
    occupation: str
    relationships: Dict[str, float]  # {resident_id: strength (-1.0 to 1.0)}
    internal_state: Dict[str, Any]
    memory_graph_id: str


@dataclass
class Relationship:
    """
    A relationship between two residents.

    Models relationships as springs with physics:
    - Strength: Spring stiffness (how connected)
    - Tension: Current stress on relationship
    - History: Past interactions
    """
    resident_a: str
    resident_b: str
    strength: float  # 0.0 to 1.0
    tension: float   # -1.0 (repulsion) to 1.0 (attraction)
    relationship_type: str  # "friend", "neighbor", "enemy", "romantic", etc.
    history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Import core components
from .config import NeuroHoodConfig
from .engine import NeuroHood
from .agents.resident_agent import ResidentAgent
from .consciousness.view import ConsciousnessView, ThoughtStream
from .social.physics import SocialPhysics
