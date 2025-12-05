"""
Daytime Simulation - Life in the Neighborhood

The main simulation engine that:
- Progresses time through the day
- Moves residents between locations
- Generates events and encounters
- Creates conversations
- Integrates with the dream system

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import random
from datetime import datetime, timedelta
import asyncio

from resident import Resident, create_neighborhood_residents
from conversation import ConversationGenerator, Conversation, ConversationContext, DialogueLine
from locations import Location, LocationManager, create_neighborhood_locations, ActivityType


class TimeOfDay(Enum):
    """Time periods in the simulation."""
    DAWN = "dawn"          # 5-7
    MORNING = "morning"    # 7-12
    AFTERNOON = "afternoon"  # 12-17
    EVENING = "evening"    # 17-21
    NIGHT = "night"        # 21-5


class EventType(Enum):
    """Types of simulation events."""
    WAKE_UP = "wake_up"
    MOVEMENT = "movement"
    ENCOUNTER = "encounter"
    CONVERSATION = "conversation"
    ACTIVITY = "activity"
    MEAL = "meal"
    WEATHER_CHANGE = "weather_change"
    COMMUNITY_EVENT = "community_event"
    SLEEP = "sleep"
    USER_ACTION = "user_action"


@dataclass
class SimulationEvent:
    """An event in the simulation."""
    event_type: EventType
    timestamp: datetime
    description: str
    participants: List[str]
    location: str
    data: Dict = field(default_factory=dict)
    conversation: Optional[Conversation] = None

    def __str__(self):
        time_str = self.timestamp.strftime("%H:%M")
        return f"[{time_str}] {self.description}"


@dataclass
class SimulationState:
    """Current state of the simulation."""
    current_time: datetime
    time_of_day: TimeOfDay
    weather: str
    day_number: int
    events_today: List[SimulationEvent] = field(default_factory=list)
    active_conversations: List[Conversation] = field(default_factory=list)


class DaytimeSimulation:
    """Main simulation engine for daytime life."""

    def __init__(self, dream_system=None):
        # Core components
        self.residents = create_neighborhood_residents()
        self.location_manager = LocationManager(create_neighborhood_locations())
        self.conversation_generator = ConversationGenerator()
        self.dream_system = dream_system

        # State
        self.state = SimulationState(
            current_time=datetime.now().replace(hour=7, minute=0),
            time_of_day=TimeOfDay.MORNING,
            weather="pleasant",
            day_number=1,
        )

        # Event callbacks for UI
        self.event_callbacks: List[Callable[[SimulationEvent], None]] = []
        self.conversation_callbacks: List[Callable[[Conversation], None]] = []

        # Simulation control
        self.running = False
        self.paused = False
        self.speed = 1.0  # 1.0 = real-time minutes per second

        # User state
        self.user_location: Optional[str] = None
        self.user_in_conversation: bool = False

        # Initialize residents to their homes
        self._initialize_resident_locations()

    def _initialize_resident_locations(self):
        """Place all residents at their homes."""
        for resident_id in self.residents:
            home_id = f"home_{resident_id}"
            if home_id in self.location_manager.locations:
                self.location_manager.move_resident(resident_id, home_id)

    def add_event_callback(self, callback: Callable[[SimulationEvent], None]):
        """Add callback for when events occur."""
        self.event_callbacks.append(callback)

    def add_conversation_callback(self, callback: Callable[[Conversation], None]):
        """Add callback for conversations."""
        self.conversation_callbacks.append(callback)

    def _emit_event(self, event: SimulationEvent):
        """Emit event to all callbacks."""
        self.state.events_today.append(event)
        for callback in self.event_callbacks:
            callback(event)

    def _emit_conversation(self, conversation: Conversation):
        """Emit conversation to all callbacks."""
        for callback in self.conversation_callbacks:
            callback(conversation)

    def get_time_of_day(self, hour: int) -> TimeOfDay:
        """Get time of day from hour."""
        if 5 <= hour < 7:
            return TimeOfDay.DAWN
        elif 7 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT

    def advance_time(self, minutes: int = 15):
        """Advance simulation time."""
        self.state.current_time += timedelta(minutes=minutes)
        new_tod = self.get_time_of_day(self.state.current_time.hour)

        # Check for time-of-day transition
        if new_tod != self.state.time_of_day:
            self.state.time_of_day = new_tod
            self._handle_time_transition(new_tod)

        # Update resident states
        hours_passed = minutes / 60
        for resident in self.residents.values():
            resident.update_emotional_state(hours_passed)

    def _handle_time_transition(self, new_tod: TimeOfDay):
        """Handle transition to new time of day."""
        if new_tod == TimeOfDay.MORNING:
            self._morning_routine()
        elif new_tod == TimeOfDay.EVENING:
            self._evening_routine()
        elif new_tod == TimeOfDay.NIGHT:
            self._night_routine()

    def _morning_routine(self):
        """Morning: residents wake up and leave homes."""
        for resident_id, resident in self.residents.items():
            event = SimulationEvent(
                event_type=EventType.WAKE_UP,
                timestamp=self.state.current_time,
                description=f"{resident.name} wakes up and starts their day.",
                participants=[resident_id],
                location=f"home_{resident_id}",
            )
            self._emit_event(event)

    def _evening_routine(self):
        """Evening: residents gather at social spots."""
        event = SimulationEvent(
            event_type=EventType.COMMUNITY_EVENT,
            timestamp=self.state.current_time,
            description="The neighborhood comes alive as people finish their day.",
            participants=[],
            location="plaza_fountain",
        )
        self._emit_event(event)

    def _night_routine(self):
        """Night: residents head home."""
        for resident_id, resident in self.residents.items():
            home_id = f"home_{resident_id}"
            self.location_manager.move_resident(resident_id, home_id)

            event = SimulationEvent(
                event_type=EventType.SLEEP,
                timestamp=self.state.current_time,
                description=f"{resident.name} heads home for the night.",
                participants=[resident_id],
                location=home_id,
            )
            self._emit_event(event)

    def simulate_step(self) -> List[SimulationEvent]:
        """Simulate one time step (15 minutes)."""
        events = []

        # Determine what happens this step
        for resident_id, resident in self.residents.items():
            # Skip if resident is asleep
            if self.state.time_of_day == TimeOfDay.NIGHT:
                continue

            # Chance to move locations
            if random.random() < 0.2:
                event = self._maybe_move_resident(resident_id, resident)
                if event:
                    events.append(event)

            # Check for encounters at current location
            nearby = self.location_manager.find_nearby_residents(resident_id)
            if nearby and random.random() < 0.3:
                other_id = random.choice(nearby)
                event = self._trigger_encounter(resident_id, other_id)
                if event:
                    events.append(event)

        # Advance time
        self.advance_time(15)

        return events

    def _maybe_move_resident(self, resident_id: str, resident: Resident) -> Optional[SimulationEvent]:
        """Maybe move a resident to a new location."""
        current_loc = self.location_manager.get_resident_location(resident_id)
        if not current_loc:
            return None

        # Get personality-appropriate destinations
        destinations = self._get_likely_destinations(resident)
        if not destinations:
            return None

        # Choose destination
        new_loc = random.choice(destinations)
        if new_loc.location_id == current_loc.location_id:
            return None

        # Move
        if self.location_manager.move_resident(resident_id, new_loc.location_id):
            event = SimulationEvent(
                event_type=EventType.MOVEMENT,
                timestamp=self.state.current_time,
                description=f"{resident.name} goes to {new_loc.name}.",
                participants=[resident_id],
                location=new_loc.location_id,
            )

            # Add memory of visiting the location
            resident.add_memory(
                event_type="visit",
                description=f"Visited {new_loc.name}. {new_loc.atmosphere}.",
                participants=[],
                emotional_impact={"energy": -0.02},  # Walking uses energy
                location=new_loc.location_id,
                importance=0.2,
            )

            self._emit_event(event)
            return event

        return None

    def _get_likely_destinations(self, resident: Resident) -> List[Location]:
        """Get destinations likely for this resident."""
        tod = self.state.time_of_day.value
        available = self.location_manager.get_public_locations(tod)

        # Weight by personality
        weighted = []
        for loc in available:
            weight = 1.0

            # Extraverts prefer social places
            if loc.conversation_likelihood > 0.6:
                weight *= 1 + resident.personality.extraversion * 0.5

            # Introverts prefer quiet places
            if loc.conversation_likelihood < 0.4:
                weight *= 1 + (1 - resident.personality.extraversion) * 0.5

            # Match activities to hobbies
            for activity in loc.activities:
                if activity.value in [h.lower() for h in resident.schedule.hobbies]:
                    weight *= 1.5

            # Social need drives to social places
            if resident.emotional_state.needs_social():
                weight *= 1 + loc.conversation_likelihood

            weighted.append((loc, weight))

        # Weighted random sample
        if not weighted:
            return []

        total_weight = sum(w for _, w in weighted)
        chosen = []
        for loc, weight in weighted:
            if random.random() < (weight / total_weight) * 3:
                chosen.append(loc)

        return chosen[:3]

    def _trigger_encounter(self, resident1_id: str, resident2_id: str) -> Optional[SimulationEvent]:
        """Trigger an encounter between two residents."""
        resident1 = self.residents[resident1_id]
        resident2 = self.residents[resident2_id]

        # Check if both want to talk
        if not (resident1.wants_to_talk() and resident2.wants_to_talk()):
            return None

        # Get location
        loc = self.location_manager.get_resident_location(resident1_id)
        if not loc:
            return None

        # Generate conversation
        context = ConversationContext(
            location=loc.name,
            time_of_day=self.state.time_of_day.value,
            weather=self.state.weather,
            user_present=(self.user_location == loc.location_id),
        )

        conversation = self.conversation_generator.generate_conversation(
            resident1, resident2, context
        )

        # Create event
        event = SimulationEvent(
            event_type=EventType.CONVERSATION,
            timestamp=self.state.current_time,
            description=f"{resident1.name} and {resident2.name} have a conversation at {loc.name}.",
            participants=[resident1_id, resident2_id],
            location=loc.location_id,
            conversation=conversation,
        )

        # Update relationships and add memories
        for participant_id in [resident1_id, resident2_id]:
            other_id = resident2_id if participant_id == resident1_id else resident1_id
            other_name = self.residents[other_id].name
            self.residents[participant_id].update_relationship(
                other_id,
                affection_delta=0.02,
                familiarity_delta=0.03,
            )
            self.residents[participant_id].satisfy_social_need(0.2)

            # Add memory of the conversation
            self.residents[participant_id].add_memory(
                event_type="conversation",
                description=f"Had a conversation with {other_name} at {loc.name}.",
                participants=[other_id],
                emotional_impact={"happiness": 0.05, "loneliness": -0.1},
                location=loc.location_id,
                importance=0.4,
            )

        self._emit_event(event)
        self._emit_conversation(conversation)

        return event

    def get_location_description(self, location_id: str) -> str:
        """Get rich description of a location."""
        loc = self.location_manager.get_location(location_id)
        if not loc:
            return "Unknown location."

        desc = f"\n{'='*50}\n"
        desc += f"  {loc.name.upper()}\n"
        desc += f"{'='*50}\n\n"
        desc += f"  {loc.description}\n"
        desc += f"  Atmosphere: {loc.atmosphere}\n"

        # Who's here?
        occupants = list(loc.current_occupants)
        if occupants:
            desc += f"\n  People here:\n"
            for occ_id in occupants:
                resident = self.residents.get(occ_id)
                if resident:
                    activity = loc.get_random_activity().value
                    desc += f"    - {resident.name} ({activity})\n"
        else:
            desc += f"\n  The place is quiet - no one else is here.\n"

        return desc

    def get_resident_status(self, resident_id: str) -> str:
        """Get status of a resident."""
        resident = self.residents.get(resident_id)
        if not resident:
            return "Unknown resident."

        loc = self.location_manager.get_resident_location(resident_id)
        loc_name = loc.name if loc else "somewhere"

        status = f"\n  {resident.name.upper()}\n"
        status += f"  {'-'*30}\n"
        status += f"  Location: {loc_name}\n"
        status += f"  Mood: {resident.emotional_state.mood_description()}\n"
        status += f"  Energy: {'*' * int(resident.emotional_state.energy * 10)}\n"
        status += f"  Social need: {'*' * int(resident.emotional_state.social_need * 10)}\n"

        return status

    def user_goes_to(self, location_id: str) -> str:
        """Move user to a location."""
        loc = self.location_manager.get_location(location_id)
        if not loc:
            return f"Can't find location: {location_id}"

        if not loc.is_open(self.state.time_of_day.value):
            return f"{loc.name} is closed right now."

        self.user_location = location_id

        # Create event
        event = SimulationEvent(
            event_type=EventType.USER_ACTION,
            timestamp=self.state.current_time,
            description=f"You arrive at {loc.name}.",
            participants=["user"],
            location=location_id,
        )
        self._emit_event(event)

        return self.get_location_description(location_id)

    def user_talks_to(self, resident_id: str) -> Optional[Tuple[Conversation, List[str]]]:
        """User initiates conversation with a resident."""
        resident = self.residents.get(resident_id)
        if not resident:
            return None

        # Check if at same location
        resident_loc = self.location_manager.resident_locations.get(resident_id)
        if resident_loc != self.user_location:
            return None

        loc = self.location_manager.get_location(self.user_location)

        # Create conversation context
        context = ConversationContext(
            location=loc.name if loc else "unknown",
            time_of_day=self.state.time_of_day.value,
            weather=self.state.weather,
            user_present=True,
        )

        # Get greeting from resident
        rel = resident.get_relationship("user")
        greeting = DialogueLine(
            speaker_id=resident_id,
            speaker_name=resident.name,
            text=resident.get_greeting("friend", rel),
            emotion="happy" if rel.affection > 0.5 else "neutral",
        )

        # Create conversation shell
        conversation = Conversation(
            participants=["user", resident_id],
            location=context.location,
            start_time=self.state.current_time,
        )
        conversation.add_line(greeting)

        # Generate response options for user
        options = self.conversation_generator.generate_user_response_options(
            {"location": context.location},
            resident,
            greeting,
            "greeting",
        )

        self.user_in_conversation = True
        return conversation, options

    def user_says(
        self,
        conversation: Conversation,
        user_text: str,
        other_resident_id: str,
    ) -> Tuple[DialogueLine, List[str]]:
        """User says something in a conversation."""
        resident = self.residents.get(other_resident_id)
        if not resident:
            return None, []

        # Add user's line
        user_line = DialogueLine(
            speaker_id="user",
            speaker_name="You",
            text=user_text,
            emotion="neutral",
        )
        conversation.add_line(user_line)

        # Generate response
        rel = resident.get_relationship("user")
        response = self.conversation_generator.generate_response(
            resident,
            self.residents.get("maya_001"),  # placeholder for user
            user_line,
            rel,
            conversation.topics_covered[-1] if conversation.topics_covered else "small_talk",
        )
        conversation.add_line(response)

        # Generate new options
        options = self.conversation_generator.generate_user_response_options(
            {},
            resident,
            response,
            "general",
        )

        # Add "goodbye" option
        options.append("Well, I should get going. Good talking to you!")

        return response, options

    def end_conversation(self, conversation: Conversation, other_resident_id: str):
        """End a conversation gracefully."""
        resident = self.residents.get(other_resident_id)
        if resident:
            # Update relationship
            resident.update_relationship(
                "user",
                affection_delta=0.03,
                familiarity_delta=0.05,
                trust_delta=0.02,
            )
            resident.satisfy_social_need(0.3)

        self.user_in_conversation = False
        conversation.is_complete = True

    def get_time_string(self) -> str:
        """Get current time as string."""
        return self.state.current_time.strftime("%H:%M")

    def get_status(self) -> str:
        """Get current simulation status."""
        status = f"\n  Day {self.state.day_number} - {self.get_time_string()}\n"
        status += f"  {self.state.time_of_day.value.capitalize()}, {self.state.weather}\n"
        if self.user_location:
            loc = self.location_manager.get_location(self.user_location)
            status += f"  You are at: {loc.name if loc else 'Unknown'}\n"
        return status


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("DAYTIME SIMULATION DEMO")
    print("=" * 70)

    sim = DaytimeSimulation()

    # Track events
    def on_event(event):
        print(f"  {event}")

    sim.add_event_callback(on_event)

    print(sim.get_status())

    print("\n--- Simulating morning ---\n")
    for _ in range(4):  # 1 hour
        events = sim.simulate_step()

    print(sim.get_status())

    print("\n--- Garden Description ---")
    print(sim.get_location_description("community_garden"))

    print("\n--- Maya's Status ---")
    print(sim.get_resident_status("maya_001"))
