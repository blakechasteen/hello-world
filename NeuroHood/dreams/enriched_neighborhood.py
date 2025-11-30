"""
Enriched Neighborhood - Deep AI-Powered Simulation

Expansions:
- Richer personality prompts with backstories
- Relationship-aware dialogue (friends vs strangers vs rivals)
- Group conversations (3+ people)
- Neighborhood events (music, cooking, gardening)
- Emotional contagion (moods spread)
- Memory of past conversations
- Time-of-day effects on dialogue
- Gossip and social dynamics

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from neighborhood_engine import (
    NeighborhoodEngine, AgentState, NeighborhoodState,
    TemplateDialogueGenerator
)
from ollama_client import OllamaClient, OllamaConfig
from conversation import DialogueLine, Conversation


# =============================================================================
# ENRICHED PERSONALITY SYSTEM
# =============================================================================

@dataclass
class ResidentBackstory:
    """Rich backstory for deeper characterization."""
    occupation: str
    hobby: str
    secret: str  # Something they don't share easily
    dream: str   # What they aspire to
    quirk: str   # Unique behavioral trait
    speech_pattern: str  # How they talk
    favorite_topic: str
    pet_peeve: str


# Backstories for each resident
RESIDENT_BACKSTORIES = {
    "maya_001": ResidentBackstory(
        occupation="freelance artist",
        hobby="collecting vintage postcards",
        secret="working on a graphic novel about the neighborhood",
        dream="having her art in a real gallery",
        quirk="sketches people while talking to them",
        speech_pattern="uses colorful metaphors and art references",
        favorite_topic="creativity and self-expression",
        pet_peeve="people who say 'I could do that' about art"
    ),
    "jorge_002": ResidentBackstory(
        occupation="retired chef, now community garden coordinator",
        hobby="growing heirloom tomatoes",
        secret="once cooked for a famous celebrity",
        dream="writing a cookbook with neighborhood recipes",
        quirk="relates everything to food and cooking",
        speech_pattern="warm and nurturing, lots of food metaphors",
        favorite_topic="food, gardening, and family traditions",
        pet_peeve="wasting food"
    ),
    "elena_003": ResidentBackstory(
        occupation="software engineer working remotely",
        hobby="amateur astronomy",
        secret="writes poetry but never shows anyone",
        dream="seeing the northern lights",
        quirk="always has a book with her",
        speech_pattern="thoughtful pauses, precise word choice",
        favorite_topic="science, technology, and the cosmos",
        pet_peeve="small talk without substance"
    ),
    "carlos_004": ResidentBackstory(
        occupation="retired history teacher",
        hobby="genealogy research",
        secret="was once a semi-professional dancer",
        dream="tracing his family tree back to the old country",
        quirk="tells stories that start 'this reminds me of...'",
        speech_pattern="storytelling, historical references",
        favorite_topic="history and neighborhood traditions",
        pet_peeve="people who don't learn from the past"
    ),
    "sofia_005": ResidentBackstory(
        occupation="runs the corner cafe",
        hobby="collecting vinyl records",
        secret="the cafe is barely profitable but she keeps it for the community",
        dream="hosting live music nights",
        quirk="remembers everyone's usual order",
        speech_pattern="warm, welcoming, uses people's names a lot",
        favorite_topic="music, community, and good coffee",
        pet_peeve="chain coffee shops"
    ),
    "diego_006": ResidentBackstory(
        occupation="music teacher at local school",
        hobby="street photography",
        secret="composing a symphony inspired by neighborhood sounds",
        dream="performing his original music at the gazebo",
        quirk="hums or taps rhythms unconsciously",
        speech_pattern="enthusiastic, musical references, expressive",
        favorite_topic="music, art, and capturing moments",
        pet_peeve="people who say they have no musical talent"
    ),
}


class NeighborhoodEvent(Enum):
    """Events that can happen in the neighborhood."""
    MUSIC_AT_GAZEBO = "music_at_gazebo"
    COOKING_AT_CAFE = "cooking_at_cafe"
    GARDENING_TOGETHER = "gardening_together"
    BOOK_CLUB_READING = "book_club_reading"
    SUNSET_WATCHING = "sunset_watching"
    MORNING_COFFEE = "morning_coffee"
    AFTERNOON_GOSSIP = "afternoon_gossip"


@dataclass
class ActiveEvent:
    """An event currently happening."""
    event_type: NeighborhoodEvent
    location_id: str
    participants: List[str]
    description: str
    started: datetime = field(default_factory=datetime.now)


# =============================================================================
# ENRICHED DIALOGUE GENERATOR
# =============================================================================

class EnrichedDialogueGenerator:
    """
    Deep personality-driven dialogue with:
    - Backstory integration
    - Relationship awareness
    - Memory of past conversations
    - Time-of-day effects
    - Event context
    """

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
        self.fallback = TemplateDialogueGenerator()
        self.conversation_memory: Dict[str, List[str]] = {}  # agent_id -> recent topics

    def _build_enriched_prompt(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any]
    ) -> str:
        """Build a rich, nuanced system prompt."""
        backstory = RESIDENT_BACKSTORIES.get(speaker.agent_id)

        # Base personality from Big Five
        traits = self._interpret_big_five(speaker.personality)

        # Build prompt
        prompt = f"""You are {speaker.name}, a resident of a small, close-knit neighborhood.

PERSONALITY: {traits}

"""
        if backstory:
            prompt += f"""BACKGROUND:
- Occupation: {backstory.occupation}
- Hobby: {backstory.hobby}
- Personal dream: {backstory.dream}
- Quirk: {backstory.quirk}
- Speech style: {backstory.speech_pattern}
- Favorite topic: {backstory.favorite_topic}

"""

        # Current emotional state
        mood = self._describe_emotions(speaker.emotions)
        prompt += f"CURRENT MOOD: {mood}\n\n"

        # Relationship with listener
        rel_desc = self._describe_relationship(speaker, listener)
        prompt += f"YOUR RELATIONSHIP WITH {listener.name.upper()}: {rel_desc}\n\n"

        # Time of day context
        time_context = self._get_time_context(context.get('time_of_day', 'afternoon'))
        prompt += f"TIME CONTEXT: {time_context}\n\n"

        # Location context
        location = context.get('location', 'the neighborhood')
        prompt += f"LOCATION: You're at {location}\n\n"

        # Active event
        if context.get('active_event'):
            event = context['active_event']
            prompt += f"WHAT'S HAPPENING: {event.description}\n\n"

        # Recent memories for context
        if speaker.memories:
            recent = speaker.memories[-3:]
            memory_text = "; ".join(m.get('description', '') for m in recent if isinstance(m, dict))
            if memory_text:
                prompt += f"RECENT EXPERIENCES: {memory_text}\n\n"

        # Past conversation topics with this person
        pair_key = f"{speaker.agent_id}_{listener.agent_id}"
        if pair_key in self.conversation_memory:
            topics = self.conversation_memory[pair_key][-3:]
            prompt += f"TOPICS YOU'VE DISCUSSED BEFORE: {', '.join(topics)}\n\n"

        prompt += """DIALOGUE GUIDELINES:
- Keep responses SHORT (1-3 sentences max)
- Stay in character with your speech style
- Reference your background naturally (don't force it)
- React authentically to what's said
- Use *actions* occasionally like *smiles* or *looks thoughtful*
- If you're close friends, be warmer. If rivals, be cooler.

DO NOT:
- Be overly formal or robotic
- Give long explanations
- Break character
- Repeat the same response"""

        return prompt

    def _interpret_big_five(self, personality: Dict[str, float]) -> str:
        """Convert Big Five scores to natural language."""
        traits = []

        o = personality.get('openness', 0.5)
        c = personality.get('conscientiousness', 0.5)
        e = personality.get('extraversion', 0.5)
        a = personality.get('agreeableness', 0.5)
        n = personality.get('neuroticism', 0.5)

        # Openness
        if o > 0.7:
            traits.append("highly creative and imaginative, always exploring new ideas")
        elif o > 0.5:
            traits.append("curious and appreciates art and ideas")
        elif o < 0.3:
            traits.append("practical and prefers routine over novelty")

        # Extraversion
        if e > 0.7:
            traits.append("very outgoing, loves being around people, energized by social interaction")
        elif e > 0.5:
            traits.append("friendly and sociable")
        elif e < 0.3:
            traits.append("quiet and introspective, prefers meaningful one-on-one conversations")

        # Agreeableness
        if a > 0.7:
            traits.append("extremely warm, caring, and always thinking of others")
        elif a > 0.5:
            traits.append("kind and cooperative")
        elif a < 0.3:
            traits.append("direct and doesn't sugarcoat things")

        # Conscientiousness
        if c > 0.7:
            traits.append("highly organized and reliable")
        elif c < 0.3:
            traits.append("spontaneous and flexible")

        # Neuroticism
        if n > 0.7:
            traits.append("emotionally sensitive, feels things deeply")
        elif n < 0.3:
            traits.append("emotionally stable and calm under pressure")

        return ", ".join(traits) if traits else "balanced personality"

    def _describe_emotions(self, emotions: Dict[str, float]) -> str:
        """Describe current emotional state."""
        states = []

        h = emotions.get('happiness', 0)
        a = emotions.get('anxiety', 0)
        l = emotions.get('loneliness', 0)
        e = emotions.get('energy', 0.5)

        if h > 0.5:
            states.append("feeling genuinely happy and content")
        elif h > 0.2:
            states.append("in a good mood")
        elif h < -0.3:
            states.append("feeling down")

        if a > 0.5:
            states.append("quite anxious")
        elif a > 0.3:
            states.append("a bit on edge")

        if l > 0.5:
            states.append("craving connection")
        elif l > 0.3:
            states.append("appreciating the company")

        if e < 0.3:
            states.append("tired")
        elif e > 0.7:
            states.append("full of energy")

        return ", ".join(states) if states else "emotionally neutral"

    def _describe_relationship(self, speaker: AgentState, listener: AgentState) -> str:
        """Describe the relationship between two agents."""
        rel = speaker.relationships.get(listener.agent_id)

        if not rel:
            return f"You don't know {listener.name} very well yet - this is a new acquaintance."

        affection = rel.get('affection', 0)
        trust = rel.get('trust', 0)
        familiarity = rel.get('familiarity', 0)
        rel_type = rel.get('type', 'acquaintance')

        if rel_type == 'close_friend' or affection > 0.7:
            return f"{listener.name} is one of your closest friends. You feel comfortable and can be yourself around them."
        elif rel_type == 'friend' or affection > 0.5:
            return f"You consider {listener.name} a friend. You enjoy their company."
        elif rel_type == 'rival' or affection < -0.3:
            return f"You and {listener.name} don't always see eye to eye. There's some tension."
        elif familiarity > 0.5:
            return f"You know {listener.name} fairly well from the neighborhood."
        else:
            return f"{listener.name} is a neighbor you're still getting to know."

    def _get_time_context(self, time_of_day: str) -> str:
        """Get context based on time of day."""
        contexts = {
            "morning": "It's morning - people are starting their day, there's a fresh energy.",
            "afternoon": "It's afternoon - the day is in full swing, people are out and about.",
            "evening": "It's evening - the day is winding down, there's a more relaxed vibe.",
            "night": "It's night - things are quiet, conversations feel more intimate."
        }
        return contexts.get(time_of_day, "")

    async def generate_line(
        self,
        speaker: AgentState,
        listener: AgentState,
        context: Dict[str, Any],
        conversation_history: List[DialogueLine]
    ) -> str:
        """Generate enriched dialogue."""
        system_prompt = self._build_enriched_prompt(speaker, listener, context)

        # Build conversation context
        history_text = ""
        for line in conversation_history[-6:]:
            history_text += f"{line.speaker_name}: {line.text}\n"

        if not history_text:
            history_text = "(Conversation just started)"

        user_prompt = f"""Conversation so far:
{history_text}

Now respond as {speaker.name}. Stay in character, keep it natural and SHORT (1-3 sentences):"""

        try:
            response = await self.client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=120
            )

            # Clean response
            response = response.strip()
            if response.startswith(f"{speaker.name}:"):
                response = response[len(speaker.name) + 1:].strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            # Track conversation topics (simple extraction)
            self._track_topic(speaker.agent_id, listener.agent_id, response)

            return response if response else "..."

        except Exception as e:
            print(f"[Enriched] Error: {e}")
            return await self.fallback.generate_line(speaker, listener, context, conversation_history)

    def _track_topic(self, speaker_id: str, listener_id: str, response: str):
        """Track conversation topics for memory."""
        # Simple topic extraction (keywords)
        topics = []
        keywords = ['art', 'music', 'food', 'garden', 'book', 'work', 'family',
                    'weather', 'dream', 'hobby', 'cafe', 'neighborhood']
        response_lower = response.lower()
        for kw in keywords:
            if kw in response_lower:
                topics.append(kw)

        if topics:
            pair_key = f"{speaker_id}_{listener_id}"
            if pair_key not in self.conversation_memory:
                self.conversation_memory[pair_key] = []
            self.conversation_memory[pair_key].extend(topics)
            # Keep last 10 topics
            self.conversation_memory[pair_key] = self.conversation_memory[pair_key][-10:]


# =============================================================================
# GROUP CONVERSATION SYSTEM
# =============================================================================

class GroupConversation:
    """Multi-person conversation with turn-taking."""

    def __init__(
        self,
        participants: List[AgentState],
        location: str,
        dialogue_gen: EnrichedDialogueGenerator,
        event: Optional[ActiveEvent] = None
    ):
        self.participants = participants
        self.location = location
        self.dialogue_gen = dialogue_gen
        self.event = event
        self.lines: List[DialogueLine] = []
        self.turn_index = 0

    def _select_next_speaker(self, last_speaker: Optional[AgentState] = None) -> AgentState:
        """Select who speaks next based on personality and context."""
        if not last_speaker:
            # First speaker - most extraverted
            return max(self.participants, key=lambda a: a.personality.get('extraversion', 0.5))

        # Others who haven't spoken recently
        recent_speakers = [l.speaker_id for l in self.lines[-3:]]
        candidates = [p for p in self.participants
                      if p.agent_id != last_speaker.agent_id
                      and p.agent_id not in recent_speakers]

        if not candidates:
            candidates = [p for p in self.participants if p.agent_id != last_speaker.agent_id]

        if not candidates:
            return self.participants[0]

        # Weight by extraversion
        weights = [p.personality.get('extraversion', 0.5) + 0.1 for p in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _select_addressee(self, speaker: AgentState) -> AgentState:
        """Select who the speaker is addressing."""
        # Address someone else
        others = [p for p in self.participants if p.agent_id != speaker.agent_id]

        # Prefer people they have relationship with
        friends = [o for o in others if o.agent_id in speaker.relationships]
        if friends:
            return random.choice(friends)
        return random.choice(others) if others else speaker

    async def generate_round(self, context: Dict[str, Any]) -> List[DialogueLine]:
        """Generate one round of conversation (each person speaks once)."""
        round_lines = []

        for i, _ in enumerate(self.participants):
            last_speaker = None if not self.lines else self.participants[
                next((j for j, p in enumerate(self.participants)
                      if p.agent_id == self.lines[-1].speaker_id), 0)
            ]

            speaker = self._select_next_speaker(last_speaker)
            addressee = self._select_addressee(speaker)

            # Add group context
            group_context = {
                **context,
                'group_conversation': True,
                'group_members': [p.name for p in self.participants],
                'active_event': self.event
            }

            line_text = await self.dialogue_gen.generate_line(
                speaker=speaker,
                listener=addressee,
                context=group_context,
                conversation_history=self.lines
            )

            line = DialogueLine(
                speaker_id=speaker.agent_id,
                speaker_name=speaker.name,
                text=line_text
            )

            self.lines.append(line)
            round_lines.append(line)

        return round_lines


# =============================================================================
# EMOTIONAL CONTAGION SYSTEM
# =============================================================================

class EmotionalContagion:
    """Models how emotions spread between agents."""

    @staticmethod
    def apply_contagion(
        source: AgentState,
        target: AgentState,
        interaction_strength: float = 0.1
    ) -> Dict[str, float]:
        """
        Apply emotional contagion from source to target.
        Returns the emotion deltas for the target.
        """
        deltas = {}

        # Relationship affects contagion strength
        rel = target.relationships.get(source.agent_id, {})
        affection = rel.get('affection', 0.1)

        # Stronger relationships = stronger contagion
        strength = interaction_strength * (0.5 + affection)

        # Happiness is contagious
        h_diff = source.emotions.get('happiness', 0) - target.emotions.get('happiness', 0)
        deltas['happiness'] = h_diff * strength * 0.3

        # Anxiety spreads (but less strongly)
        a_source = source.emotions.get('anxiety', 0)
        if a_source > 0.3:
            deltas['anxiety'] = a_source * strength * 0.2

        # Loneliness decreases when interacting
        deltas['loneliness'] = -0.1 * strength

        return deltas

    @staticmethod
    def apply_group_contagion(agents: List[AgentState]) -> Dict[str, Dict[str, float]]:
        """Apply emotional contagion across a group."""
        all_deltas = {a.agent_id: {} for a in agents}

        # Group mood (average)
        group_happiness = sum(a.emotions.get('happiness', 0) for a in agents) / len(agents)
        group_energy = sum(a.emotions.get('energy', 0.5) for a in agents) / len(agents)

        for agent in agents:
            # Pull toward group mood
            h_diff = group_happiness - agent.emotions.get('happiness', 0)
            e_diff = group_energy - agent.emotions.get('energy', 0.5)

            all_deltas[agent.agent_id]['happiness'] = h_diff * 0.1
            all_deltas[agent.agent_id]['energy'] = e_diff * 0.05
            all_deltas[agent.agent_id]['loneliness'] = -0.15  # Groups reduce loneliness

        return all_deltas


# =============================================================================
# ENRICHED NEIGHBORHOOD ENGINE
# =============================================================================

class EnrichedNeighborhoodEngine(NeighborhoodEngine):
    """
    Extended engine with:
    - Rich personality system
    - Group conversations
    - Emotional contagion
    - Neighborhood events
    """

    def __init__(self, ollama_config: OllamaConfig = None):
        # Create Ollama client
        self.ollama_client = OllamaClient(ollama_config or OllamaConfig())

        # Initialize base engine
        super().__init__()

        # Replace dialogue generator with enriched version
        self.enriched_dialogue = EnrichedDialogueGenerator(self.ollama_client)

        # Active events
        self.active_events: List[ActiveEvent] = []

        # Emotional contagion system
        self.contagion = EmotionalContagion()

        # Track Ollama availability
        self._ollama_available = False

    async def initialize(self, load_path=None):
        """Initialize with Ollama check."""
        await super().initialize(load_path)

        available = await self.ollama_client.is_available()
        if available:
            self._ollama_available = True
            self.dialogue_gen = self.enriched_dialogue
            models = await self.ollama_client.list_models()
            print(f"[Enriched] Ollama connected! Models: {models[:3]}...")
        else:
            print("[Enriched] Ollama not available - using templates")

    def trigger_event(self, event_type: NeighborhoodEvent, location_id: str) -> ActiveEvent:
        """Trigger a neighborhood event."""
        descriptions = {
            NeighborhoodEvent.MUSIC_AT_GAZEBO: "Diego is playing guitar at the gazebo, drawing a small crowd",
            NeighborhoodEvent.COOKING_AT_CAFE: "Sofia is preparing a special dish at the cafe, filling the air with delicious aromas",
            NeighborhoodEvent.GARDENING_TOGETHER: "Jorge is organizing a community gardening session",
            NeighborhoodEvent.BOOK_CLUB_READING: "Elena is hosting an informal book discussion at the reading corner",
            NeighborhoodEvent.SUNSET_WATCHING: "People are gathering at the fountain to watch the sunset",
            NeighborhoodEvent.MORNING_COFFEE: "The morning rush at the cafe - everyone's getting their coffee",
            NeighborhoodEvent.AFTERNOON_GOSSIP: "A few neighbors are chatting and catching up on neighborhood news",
        }

        event = ActiveEvent(
            event_type=event_type,
            location_id=location_id,
            participants=[],
            description=descriptions.get(event_type, "Something is happening")
        )
        self.active_events.append(event)
        return event

    def get_active_event_at(self, location_id: str) -> Optional[ActiveEvent]:
        """Get active event at a location."""
        for event in self.active_events:
            if event.location_id == location_id:
                return event
        return None

    async def user_talk_to(self, agent_id: str, user_message: Optional[str] = None) -> Conversation:
        """User talks to agent with enriched dialogue."""
        if not self.user_agent:
            raise ValueError("User not added")

        if agent_id not in self.state.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        other = self.state.agents[agent_id]

        conv = Conversation(
            participants=[self.user_agent.agent_id, agent_id],
            location=self.user_agent.location_id or "neighborhood",
            start_time=datetime.now()
        )

        if user_message:
            conv.add_line(DialogueLine(
                speaker_id=self.user_agent.agent_id,
                speaker_name=self.user_agent.name,
                text=user_message
            ))

        # Build enriched context
        context = {
            'location': self.state.locations.get(self.user_agent.location_id, {}).get('name', 'neighborhood'),
            'time_of_day': self.state.time_of_day,
            'active_event': self.get_active_event_at(self.user_agent.location_id)
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

        # Apply emotional contagion
        deltas = self.contagion.apply_contagion(other, self.user_agent)
        for emotion, delta in deltas.items():
            self.user_agent.emotions[emotion] = max(-1, min(1,
                self.user_agent.emotions.get(emotion, 0) + delta
            ))

        # Update relationship
        self._update_relationship(self.user_agent.agent_id, agent_id)
        self._update_relationship(agent_id, self.user_agent.agent_id)

        # Add memories
        self._add_conversation_memory(self.user_agent, other)
        self._add_conversation_memory(other, self.user_agent)

        return conv

    async def start_group_conversation(
        self,
        agent_ids: List[str],
        include_user: bool = True
    ) -> GroupConversation:
        """Start a group conversation."""
        participants = []

        if include_user and self.user_agent:
            participants.append(self.user_agent)

        for aid in agent_ids:
            if aid in self.state.agents:
                participants.append(self.state.agents[aid])

        location = self.user_agent.location_id if self.user_agent else "neighborhood"
        event = self.get_active_event_at(location)

        group_conv = GroupConversation(
            participants=participants,
            location=location,
            dialogue_gen=self.enriched_dialogue,
            event=event
        )

        return group_conv

    async def close(self):
        """Cleanup."""
        await self.ollama_client.close()


# =============================================================================
# DEMO
# =============================================================================

async def demo_enriched_neighborhood():
    """Demonstrate enriched neighborhood features."""
    print("=" * 70)
    print("ENRICHED NEIGHBORHOOD - Deep AI-Powered Simulation")
    print("=" * 70)

    config = OllamaConfig(
        model="llama3.2:3b",
        temperature=0.85,
        max_tokens=120,
        timeout=30.0
    )

    engine = EnrichedNeighborhoodEngine(config)
    await engine.initialize()

    # Add user
    user = engine.add_user("Blake")

    # Simulate activity
    print("\n--- MORNING IN THE NEIGHBORHOOD ---\n")
    for _ in range(8):
        events = engine.simulate_step()
        for e in events[:1]:
            print(f"  {e['description']}")

    # Trigger an event
    print("\n--- NEIGHBORHOOD EVENT ---")
    event = engine.trigger_event(NeighborhoodEvent.MORNING_COFFEE, "corner_cafe")
    print(f"  {event.description}")

    # Find a location with people
    best_location = None
    best_count = 0

    for loc_id in ["corner_cafe", "community_garden", "oak_park", "plaza_fountain", "community_center"]:
        agents = engine.get_agents_at(loc_id)
        residents = [a for a in agents if not a.is_user]
        if len(residents) > best_count:
            best_count = len(residents)
            best_location = loc_id

    if best_location:
        print(f"\n--- YOU VISIT {engine.state.locations[best_location]['name'].upper()} ---")
        result = engine.user_go_to(best_location)
        print(result)
    else:
        # Force someone to the cafe for the demo
        print("\n--- YOU VISIT THE CORNER CAFE ---")
        engine.state.agent_locations["sofia_005"] = "corner_cafe"
        engine.state.agent_locations["diego_006"] = "corner_cafe"
        result = engine.user_go_to("corner_cafe")
        print(result)
        print("\n  (Sofia and Diego arrive for their morning coffee)")

    # One-on-one conversation
    agents_here = engine.get_agents_at(engine.user_agent.location_id)
    residents = [a for a in agents_here if not a.is_user]

    if residents:
        other = residents[0]
        backstory = RESIDENT_BACKSTORIES.get(other.agent_id)

        print(f"\n--- CONVERSATION WITH {other.name.upper()} ---")
        if backstory:
            print(f"({backstory.occupation}, speaks with {backstory.speech_pattern})")
        print()

        conv = await engine.user_talk_to(other.agent_id, "Good morning! The coffee smells amazing today.")
        for line in conv.lines:
            print(f"  {line.speaker_name}: \"{line.text}\"")

        # Continue
        exchanges = [
            "So what's been on your mind lately?",
            "That's interesting! Tell me more.",
            "I've been thinking about getting more involved in the neighborhood. Any suggestions?",
        ]

        for msg in exchanges:
            print(f"\n  {user.name}: \"{msg}\"")
            response = await engine.continue_conversation(conv, msg)
            print(f"  {other.name}: \"{response}\"")

    # Group conversation
    if len(residents) >= 2:
        print("\n\n--- GROUP CONVERSATION ---")
        print("(More neighbors join the conversation)\n")

        group_ids = [r.agent_id for r in residents[:3]]
        group_conv = await engine.start_group_conversation(group_ids, include_user=False)

        context = {
            'location': 'Corner Cafe',
            'time_of_day': 'morning',
            'active_event': event
        }

        # Two rounds of group conversation
        for round_num in range(2):
            print(f"\n  [Round {round_num + 1}]")
            round_lines = await group_conv.generate_round(context)
            for line in round_lines:
                print(f"  {line.speaker_name}: \"{line.text}\"")

    # Emotional contagion check
    print("\n--- EMOTIONAL STATE ---")
    print(f"\n  Your mood after socializing:")
    print(f"    Happiness: {user.emotions.get('happiness', 0):.2f}")
    print(f"    Loneliness: {user.emotions.get('loneliness', 0):.2f}")

    await engine.close()

    print("\n" + "=" * 70)
    print("ENRICHED DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_enriched_neighborhood())
