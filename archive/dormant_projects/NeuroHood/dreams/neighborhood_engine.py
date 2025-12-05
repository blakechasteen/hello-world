"""
Neighborhood Engine - Unified Simulation Architecture

Elegant moonshot combining:
- LLM-powered agents (with template fallback)
- User participation as a resident
- Memory → Dream symbol mapping
- Persistent world state

Design Principles:
- Protocol-based: swap implementations freely
- Composable: features work independently, compose beautifully
- Graceful fallback: LLM unavailable? Templates work fine
- Single state: one NeighborhoodState captures everything

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, Callable
import random

# Import existing systems
from resident import Resident, create_neighborhood_residents, Memory, Relationship, RelationshipType
from locations import Location, LocationManager, create_neighborhood_locations
from conversation import ConversationGenerator, ConversationContext, Conversation, DialogueLine
from daytime_simulation import DaytimeSimulation, SimulationState, TimeOfDay
from dream_system import DreamSystem


# =============================================================================
# PROTOCOLS - Clean abstractions for extensibility
# =============================================================================

class DialogueGenerator(Protocol):
    """Protocol for generating dialogue - can be template or LLM."""

    async def generate_line(
        self,
        speaker: "AgentState",
        listener: "AgentState",
        context: Dict[str, Any],
        conversation_history: List[DialogueLine]
    ) -> str:
        """Generate a single line of dialogue."""
        ...


class MemoryToDreamMapper(Protocol):
    """Protocol for mapping memories to dream symbols."""

    def map_memory_to_symbols(self, memory: Memory) -> List[str]:
        """Convert a memory into dream symbol seeds."""
        ...


class StateSerializer(Protocol):
    """Protocol for saving/loading world state."""

    def save(self, state: "NeighborhoodState", path: Path) -> None:
        ...

    def load(self, path: Path) -> "NeighborhoodState":
        ...


# =============================================================================
# AGENT STATE - Unified representation for residents AND user
# =============================================================================

@dataclass
class AgentState:
    """Unified state for any agent (resident or user)."""

    agent_id: str
    name: str
    is_user: bool = False

    # Personality (Big Five)
    personality: Dict[str, float] = field(default_factory=lambda: {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5
    })

    # Emotional state
    emotions: Dict[str, float] = field(default_factory=lambda: {
        "happiness": 0.0,
        "anxiety": 0.0,
        "loneliness": 0.0,
        "energy": 0.5
    })

    # Current location
    location_id: Optional[str] = None

    # Memories
    memories: List[Dict[str, Any]] = field(default_factory=list)

    # Relationships
    relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LLM context (for AI agents)
    system_prompt: Optional[str] = None

    @classmethod
    def from_resident(cls, resident: Resident) -> "AgentState":
        """Create AgentState from existing Resident."""
        return cls(
            agent_id=resident.resident_id,
            name=resident.name,
            is_user=False,
            personality={
                "openness": resident.personality.openness,
                "conscientiousness": resident.personality.conscientiousness,
                "extraversion": resident.personality.extraversion,
                "agreeableness": resident.personality.agreeableness,
                "neuroticism": resident.personality.neuroticism
            },
            emotions={
                "happiness": resident.emotional_state.happiness,
                "anxiety": resident.emotional_state.anxiety,
                "loneliness": resident.emotional_state.loneliness,
                "energy": resident.emotional_state.energy
            },
            memories=[asdict(m) for m in resident.memories],
            relationships={
                k: {"type": v.relationship_type.value, "affection": v.affection, "trust": v.trust}
                for k, v in resident.relationships.items()
            }
        )

    def get_personality_prompt(self) -> str:
        """Generate LLM personality prompt from Big Five."""
        traits = []

        if self.personality["openness"] > 0.6:
            traits.append("creative and curious")
        elif self.personality["openness"] < 0.4:
            traits.append("practical and traditional")

        if self.personality["extraversion"] > 0.6:
            traits.append("outgoing and energetic")
        elif self.personality["extraversion"] < 0.4:
            traits.append("quiet and reflective")

        if self.personality["agreeableness"] > 0.6:
            traits.append("warm and cooperative")
        elif self.personality["agreeableness"] < 0.4:
            traits.append("direct and competitive")

        if self.personality["neuroticism"] > 0.6:
            traits.append("sensitive and emotional")
        elif self.personality["neuroticism"] < 0.4:
            traits.append("calm and resilient")

        mood = []
        if self.emotions["happiness"] > 0.3:
            mood.append("happy")
        elif self.emotions["happiness"] < -0.3:
            mood.append("sad")
        if self.emotions["anxiety"] > 0.3:
            mood.append("anxious")
        if self.emotions["loneliness"] > 0.3:
            mood.append("lonely")

        prompt = f"You are {self.name}, a neighborhood resident who is {', '.join(traits) if traits else 'balanced'}."
        if mood:
            prompt += f" You're currently feeling {', '.join(mood)}."

        # Add recent memories for context
        if self.memories:
            recent = self.memories[-3:]
            prompt += f" Recent experiences: {'; '.join(m['description'] for m in recent)}"

        return prompt


# =============================================================================
# NEIGHBORHOOD STATE - Single source of truth
# =============================================================================

@dataclass
class NeighborhoodState:
    """Complete state of the neighborhood - single serializable object."""

    # Time
    current_time: datetime = field(default_factory=datetime.now)
    time_of_day: str = "morning"
    day_number: int = 1

    # Agents (residents + user)
    agents: Dict[str, AgentState] = field(default_factory=dict)

    # Locations
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Agent locations
    agent_locations: Dict[str, str] = field(default_factory=dict)

    # Conversation history (for context)
    recent_conversations: List[Dict[str, Any]] = field(default_factory=list)

    # Dream history
    dream_journal: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Neighborhood mood (aggregate)
    neighborhood_mood: Dict[str, float] = field(default_factory=lambda: {
        "energy": 0.5,
        "harmony": 0.5,
        "activity": 0.5
    })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_time": self.current_time.isoformat(),
            "time_of_day": self.time_of_day,
            "day_number": self.day_number,
            "agents": {k: asdict(v) for k, v in self.agents.items()},
            "locations": self.locations,
            "agent_locations": self.agent_locations,
            "recent_conversations": self.recent_conversations[-20:],  # Keep last 20
            "dream_journal": self.dream_journal,
            "neighborhood_mood": self.neighborhood_mood
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeighborhoodState":
        """Deserialize from dictionary."""
        state = cls()
        state.current_time = datetime.fromisoformat(data["current_time"])
        state.time_of_day = data["time_of_day"]
        state.day_number = data["day_number"]
        state.agents = {k: AgentState(**v) for k, v in data["agents"].items()}
        state.locations = data["locations"]
        state.agent_locations = data["agent_locations"]
        state.recent_conversations = data["recent_conversations"]
        state.dream_journal = data["dream_journal"]
        state.neighborhood_mood = data["neighborhood_mood"]
        return state


# =============================================================================
# DIALOGUE GENERATORS - Template and LLM implementations
# =============================================================================

class TemplateDialogueGenerator:
    """Template-based dialogue (existing system, always works)."""

    def __init__(self):
        self.conv_generator = ConversationGenerator()

    async def generate_line(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any],
        conversation_history: List[DialogueLine]
    ) -> str:
        """Generate dialogue using templates."""
        # Use existing template system
        templates = [
            f"Hello, {listener.name}!",
            "How are you today?",
            "Beautiful day, isn't it?",
            "I was just thinking about the neighborhood.",
            "Have you been to the cafe lately?",
        ]

        # Personality-based selection
        if speaker.personality["extraversion"] > 0.6:
            templates.extend([
                f"So great to see you, {listener.name}!",
                "Let me tell you what happened!",
            ])
        else:
            templates.extend([
                "*nods quietly*",
                "Mm, yes.",
            ])

        # Emotion-based
        if speaker.emotions["happiness"] > 0.3:
            templates.extend([
                "I'm having such a great day!",
                "*smiles warmly*",
            ])
        elif speaker.emotions["happiness"] < -0.3:
            templates.extend([
                "*sighs*",
                "It's been a tough week.",
            ])

        return random.choice(templates)


class LLMDialogueGenerator:
    """LLM-powered dialogue generation."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.fallback = TemplateDialogueGenerator()

    async def generate_line(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any],
        conversation_history: List[DialogueLine]
    ) -> str:
        """Generate dialogue using LLM with template fallback."""
        if not self.llm_client:
            return await self.fallback.generate_line(speaker, listener, context, conversation_history)

        # Build prompt
        system_prompt = speaker.get_personality_prompt()

        # Format conversation history
        history_text = ""
        for line in conversation_history[-6:]:
            history_text += f"{line.speaker_name}: {line.text}\n"

        user_prompt = f"""You are having a conversation with {listener.name} at {context.get('location', 'the neighborhood')}.

Previous dialogue:
{history_text}

Respond naturally as {speaker.name} in 1-2 sentences. Be conversational, not formal."""

        try:
            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=100
            )
            return response.strip()
        except Exception:
            # Graceful fallback to templates
            return await self.fallback.generate_line(speaker, listener, context, conversation_history)


# =============================================================================
# MEMORY TO DREAM MAPPER
# =============================================================================

class RichMemoryToDreamMapper:
    """Maps memories to dream symbols with semantic understanding."""

    # Location → Symbol mappings
    LOCATION_SYMBOLS = {
        "garden": ["garden", "growth", "nature", "seeds"],
        "cafe": ["hearth", "warmth", "gathering", "sustenance"],
        "park": ["tree", "path", "nature", "freedom"],
        "street": ["journey", "crossroads", "movement"],
        "home": ["shelter", "safety", "nest"],
        "center": ["gathering", "community", "circle"],
        "fountain": ["water", "reflection", "flow"],
        "gazebo": ["music", "harmony", "gathering"],
    }

    # Emotion → Symbol mappings
    EMOTION_SYMBOLS = {
        "happiness": ["sun", "light", "flight", "bloom"],
        "sadness": ["rain", "descent", "shadow", "autumn"],
        "anxiety": ["storm", "maze", "pursuit", "falling"],
        "loneliness": ["desert", "island", "void", "silence"],
        "love": ["heart", "embrace", "union", "fire"],
        "anger": ["fire", "storm", "battle", "volcano"],
    }

    # Relationship → Symbol mappings
    RELATIONSHIP_SYMBOLS = {
        "conversation": ["bridge", "dialogue", "connection"],
        "friendship": ["bond", "circle", "companion"],
        "conflict": ["sword", "division", "storm"],
        "reunion": ["return", "embrace", "homecoming"],
    }

    def map_memory_to_symbols(self, memory: Memory) -> List[str]:
        """Convert a memory into dream symbol seeds."""
        symbols = []

        # Extract location symbols
        desc_lower = memory.description.lower()
        for loc_key, loc_symbols in self.LOCATION_SYMBOLS.items():
            if loc_key in desc_lower:
                symbols.extend(random.sample(loc_symbols, min(2, len(loc_symbols))))

        # Extract emotion symbols based on impact
        for emotion, impact in memory.emotional_impact.items():
            if abs(impact) > 0.05:
                if emotion in self.EMOTION_SYMBOLS:
                    symbols.extend(random.sample(
                        self.EMOTION_SYMBOLS[emotion],
                        min(1, len(self.EMOTION_SYMBOLS[emotion]))
                    ))

        # Extract relationship symbols
        if memory.event_type == "conversation":
            symbols.extend(random.sample(self.RELATIONSHIP_SYMBOLS["conversation"], 1))

        # High importance memories get extra weight
        if memory.importance > 0.6:
            symbols.append("tower")  # Significant/memorable

        return list(set(symbols))  # Deduplicate


# =============================================================================
# STATE SERIALIZER
# =============================================================================

class JSONStateSerializer:
    """Save/load neighborhood state to JSON."""

    def save(self, state: NeighborhoodState, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)

    def load(self, path: Path) -> NeighborhoodState:
        """Load state from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return NeighborhoodState.from_dict(data)


# =============================================================================
# NEIGHBORHOOD ENGINE - The unified orchestrator
# =============================================================================

class NeighborhoodEngine:
    """
    Unified engine for neighborhood simulation.

    Combines:
    - Daytime simulation (movement, conversations)
    - Dream generation (night cycle)
    - User participation
    - LLM-powered or template dialogue
    - Persistent state
    """

    def __init__(
        self,
        dialogue_generator: Optional[DialogueGenerator] = None,
        memory_mapper: Optional[MemoryToDreamMapper] = None,
        state_serializer: Optional[StateSerializer] = None,
        llm_client = None
    ):
        # Core systems
        self.daytime_sim: Optional[DaytimeSimulation] = None
        self.dream_system: Optional[DreamSystem] = None

        # Pluggable components
        self.dialogue_gen = dialogue_generator or (
            LLMDialogueGenerator(llm_client) if llm_client
            else TemplateDialogueGenerator()
        )
        self.memory_mapper = memory_mapper or RichMemoryToDreamMapper()
        self.serializer = state_serializer or JSONStateSerializer()

        # State
        self.state: Optional[NeighborhoodState] = None

        # User agent
        self.user_agent: Optional[AgentState] = None

        # Callbacks
        self.on_conversation: List[Callable] = []
        self.on_dream: List[Callable] = []
        self.on_event: List[Callable] = []

    async def initialize(self, load_path: Optional[Path] = None):
        """Initialize the engine."""
        if load_path and load_path.exists():
            # Load existing state
            self.state = self.serializer.load(load_path)
            print(f"[Engine] Loaded state from {load_path}")
        else:
            # Create new neighborhood
            self.daytime_sim = DaytimeSimulation()
            self.state = NeighborhoodState()

            # Convert residents to agents
            for r_id, resident in self.daytime_sim.residents.items():
                self.state.agents[r_id] = AgentState.from_resident(resident)

            # Store locations
            for loc_id, loc in self.daytime_sim.location_manager.locations.items():
                self.state.locations[loc_id] = {
                    "name": loc.name,
                    "type": loc.location_type.value,
                    "description": loc.description,
                    "atmosphere": loc.atmosphere
                }

            print(f"[Engine] Created new neighborhood with {len(self.state.agents)} residents")

        # Initialize dream system
        self.dream_system = DreamSystem()
        await self.dream_system.initialize()

    def add_user(self, name: str = "You", personality: Optional[Dict[str, float]] = None) -> AgentState:
        """Add user as a participant in the neighborhood."""
        user_id = "user_001"

        self.user_agent = AgentState(
            agent_id=user_id,
            name=name,
            is_user=True,
            personality=personality or {
                "openness": 0.6,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.6,
                "neuroticism": 0.4
            }
        )

        self.state.agents[user_id] = self.user_agent
        print(f"[Engine] Added user '{name}' to neighborhood")
        return self.user_agent

    def user_go_to(self, location_id: str) -> str:
        """Move user to a location."""
        if not self.user_agent:
            return "User not added yet. Call add_user() first."

        if location_id not in self.state.locations:
            return f"Unknown location: {location_id}"

        self.state.agent_locations[self.user_agent.agent_id] = location_id
        self.user_agent.location_id = location_id

        # Add memory
        loc = self.state.locations[location_id]
        self.user_agent.memories.append({
            "event_type": "visit",
            "description": f"Visited {loc['name']}. {loc['atmosphere']}.",
            "participants": [],
            "emotional_impact": {"energy": -0.02},
            "location": location_id,
            "importance": 0.2,
            "timestamp": datetime.now().isoformat()
        })

        # Check who else is there
        others_here = [
            self.state.agents[aid].name
            for aid, loc_id in self.state.agent_locations.items()
            if loc_id == location_id and aid != self.user_agent.agent_id
        ]

        result = f"You arrive at {loc['name']}. {loc['description']}"
        if others_here:
            result += f"\n\nYou see: {', '.join(others_here)}"
        else:
            result += "\n\nThe place is quiet."

        return result

    async def user_talk_to(self, agent_id: str, user_message: Optional[str] = None) -> Conversation:
        """User initiates conversation with an agent."""
        if not self.user_agent:
            raise ValueError("User not added yet")

        if agent_id not in self.state.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        other = self.state.agents[agent_id]

        # Create conversation
        conv = Conversation(
            participants=[self.user_agent.agent_id, agent_id],
            location=self.user_agent.location_id or "neighborhood",
            start_time=datetime.now()
        )

        # User speaks first if they have a message
        if user_message:
            conv.add_line(DialogueLine(
                speaker_id=self.user_agent.agent_id,
                speaker_name=self.user_agent.name,
                text=user_message
            ))

        # Generate response
        context = {
            "location": self.state.locations.get(self.user_agent.location_id, {}).get("name", "neighborhood"),
            "time_of_day": self.state.time_of_day
        }

        response = await self.dialogue_gen.generate_line(
            speaker=other,
            listener=self.user_agent,
            context=context,
            conversation_history=conv.lines
        )

        conv.add_line(DialogueLine(
            speaker_id=agent_id,
            speaker_name=other.name,
            text=response
        ))

        # Update relationships
        self._update_relationship(self.user_agent.agent_id, agent_id)
        self._update_relationship(agent_id, self.user_agent.agent_id)

        # Add memories
        self._add_conversation_memory(self.user_agent, other)
        self._add_conversation_memory(other, self.user_agent)

        # Store conversation
        self.state.recent_conversations.append({
            "participants": [self.user_agent.agent_id, agent_id],
            "location": self.user_agent.location_id,
            "lines": [{"speaker": l.speaker_name, "text": l.text} for l in conv.lines],
            "timestamp": datetime.now().isoformat()
        })

        return conv

    async def continue_conversation(self, conv: Conversation, user_message: str) -> str:
        """Continue an existing conversation."""
        # Add user message
        conv.add_line(DialogueLine(
            speaker_id=self.user_agent.agent_id,
            speaker_name=self.user_agent.name,
            text=user_message
        ))

        # Find other participant
        other_id = [p for p in conv.participants if p != self.user_agent.agent_id][0]
        other = self.state.agents[other_id]

        # Generate response
        context = {
            "location": conv.location,
            "time_of_day": self.state.time_of_day
        }

        response = await self.dialogue_gen.generate_line(
            speaker=other,
            listener=self.user_agent,
            context=context,
            conversation_history=conv.lines
        )

        conv.add_line(DialogueLine(
            speaker_id=other_id,
            speaker_name=other.name,
            text=response
        ))

        return response

    def _update_relationship(self, agent_id: str, other_id: str):
        """Update relationship between agents."""
        agent = self.state.agents[agent_id]
        if other_id not in agent.relationships:
            agent.relationships[other_id] = {
                "type": "acquaintance",
                "affection": 0.1,
                "trust": 0.1,
                "familiarity": 0.1
            }
        else:
            rel = agent.relationships[other_id]
            rel["affection"] = min(1.0, rel["affection"] + 0.02)
            rel["familiarity"] = min(1.0, rel["familiarity"] + 0.03)

    def _add_conversation_memory(self, agent: AgentState, other: AgentState):
        """Add conversation memory to agent."""
        loc_name = self.state.locations.get(agent.location_id, {}).get("name", "the neighborhood")
        agent.memories.append({
            "event_type": "conversation",
            "description": f"Had a conversation with {other.name} at {loc_name}.",
            "participants": [other.agent_id],
            "emotional_impact": {"happiness": 0.05, "loneliness": -0.1},
            "location": agent.location_id,
            "importance": 0.4,
            "timestamp": datetime.now().isoformat()
        })

    def simulate_step(self) -> List[Dict[str, Any]]:
        """Run one simulation step (residents move and interact)."""
        if not self.daytime_sim:
            self.daytime_sim = DaytimeSimulation()

        events = self.daytime_sim.simulate_step()

        # Sync state from daytime sim
        for r_id, resident in self.daytime_sim.residents.items():
            if r_id in self.state.agents:
                agent = self.state.agents[r_id]
                agent.emotions = {
                    "happiness": resident.emotional_state.happiness,
                    "anxiety": resident.emotional_state.anxiety,
                    "loneliness": resident.emotional_state.loneliness,
                    "energy": resident.emotional_state.energy
                }
                agent.memories = [asdict(m) for m in resident.memories]

        # Sync locations
        for r_id in self.daytime_sim.residents:
            loc = self.daytime_sim.location_manager.get_resident_location(r_id)
            if loc:
                self.state.agent_locations[r_id] = loc.location_id

        return [{"type": e.event_type.value, "description": e.description} for e in events]

    async def run_night_cycle(self) -> Dict[str, Any]:
        """Run the night cycle - generate dreams for all agents."""
        dreams = {}

        for agent_id, agent in self.state.agents.items():
            if agent.is_user:
                continue  # User doesn't dream automatically

            # Collect dream seeds from memories
            seed_symbols = []
            for mem_dict in agent.memories[-5:]:
                # Create Memory object for mapper
                ts = mem_dict.get("timestamp", datetime.now())
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                mem = Memory(
                    event_type=mem_dict["event_type"],
                    description=mem_dict["description"],
                    participants=mem_dict.get("participants", []),
                    emotional_impact=mem_dict.get("emotional_impact", {}),
                    location=mem_dict.get("location", ""),
                    importance=mem_dict.get("importance", 0.5),
                    timestamp=ts
                )
                seed_symbols.extend(self.memory_mapper.map_memory_to_symbols(mem))

            # Generate dream
            dream = await self.dream_system.generate_dream(
                resident_id=agent_id,
                emotional_state=agent.emotions,
                consciousness_level=0.5,
                seed_symbols=seed_symbols[:5] if seed_symbols else None
            )

            # Store in journal
            if agent_id not in self.state.dream_journal:
                self.state.dream_journal[agent_id] = []

            dream_record = {
                "day": self.state.day_number,
                "archetype": dream.narrative.archetype.value if dream.narrative.archetype else None,
                "seed_symbols": seed_symbols[:5],
                "description": dream.narrative.acts[0].description if dream.narrative.acts else "",
                "timestamp": datetime.now().isoformat()
            }
            self.state.dream_journal[agent_id].append(dream_record)
            dreams[agent_id] = dream_record

        # Advance day
        self.state.day_number += 1

        return dreams

    def save(self, path: Path):
        """Save current state."""
        self.serializer.save(self.state, path)
        print(f"[Engine] Saved state to {path}")

    def get_agents_at(self, location_id: str) -> List[AgentState]:
        """Get all agents at a location."""
        return [
            self.state.agents[aid]
            for aid, loc_id in self.state.agent_locations.items()
            if loc_id == location_id
        ]

    def get_locations(self) -> Dict[str, Dict[str, Any]]:
        """Get all locations."""
        return self.state.locations

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent by ID."""
        return self.state.agents.get(agent_id)


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demonstrate the unified neighborhood engine."""
    print("=" * 60)
    print("NEIGHBORHOOD ENGINE - Unified Architecture Demo")
    print("=" * 60)

    # Initialize engine
    engine = NeighborhoodEngine()
    await engine.initialize()

    # Add user
    user = engine.add_user("Blake", personality={
        "openness": 0.8,
        "conscientiousness": 0.6,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.3
    })

    print("\n--- DAYTIME ---")

    # Simulate some activity
    print("\nRunning simulation...")
    for _ in range(10):
        events = engine.simulate_step()

    # User participation
    print("\n--- USER PARTICIPATION ---")

    # Go to cafe
    result = engine.user_go_to("corner_cafe")
    print(f"\n{result}")

    # Talk to someone there
    agents_here = engine.get_agents_at("corner_cafe")
    if agents_here:
        other = [a for a in agents_here if not a.is_user][0] if len(agents_here) > 1 else None
        if other:
            print(f"\n[You approach {other.name}]")
            conv = await engine.user_talk_to(other.agent_id, "Hey! Mind if I join you?")
            for line in conv.lines:
                print(f"  {line.speaker_name}: \"{line.text}\"")

            # Continue conversation
            response = await engine.continue_conversation(conv, "How's your day going?")
            print(f"  {other.name}: \"{response}\"")

    print("\n--- NIGHT CYCLE ---")

    # Run dreams
    dreams = await engine.run_night_cycle()
    print(f"\nDay {engine.state.day_number - 1} dreams:")
    for agent_id, dream in dreams.items():
        agent = engine.get_agent(agent_id)
        print(f"\n  {agent.name}:")
        print(f"    Archetype: {dream['archetype']}")
        print(f"    Seeds: {dream['seed_symbols']}")
        print(f"    Dream: \"{dream['description'][:60]}...\"")

    print("\n--- PERSISTENCE ---")

    # Save state
    save_path = Path("./neighborhood_state.json")
    engine.save(save_path)
    print(f"State saved! {len(engine.state.agents)} agents, day {engine.state.day_number}")

    print("\n" + "=" * 60)
    print("ENGINE DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
