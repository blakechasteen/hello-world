"""
Narrative Memory System

Tracks story events, player choices, and causal chains to enable coherent
quest storytelling where NPCs remember past events and react accordingly.

Created: 2025-11-13 (Week 3 - DreamWeaver Phase 1)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from EduVerse.education.curriculum import Subject, LearningObjective
from EduVerse.game.quest_engine import Quest, QuestType


class EventType(Enum):
    """Types of narrative events"""
    QUEST_STARTED = "quest_started"
    QUEST_COMPLETED = "quest_completed"
    QUEST_FAILED = "quest_failed"
    NPC_INTERACTION = "npc_interaction"
    PLAYER_CHOICE = "player_choice"
    WORLD_CHANGE = "world_change"
    LEARNING_MILESTONE = "learning_milestone"
    NARRATIVE_BRANCHING = "narrative_branching"


class EventImportance(Enum):
    """Importance level for narrative events"""
    TRIVIAL = 1      # Minor interactions
    LOW = 2          # Regular quest completions
    MEDIUM = 3       # Significant choices
    HIGH = 4         # Major milestones
    CRITICAL = 5     # Story-changing events


@dataclass
class NarrativeEvent:
    """A single story event with causality tracking"""
    event_id: str
    event_type: EventType
    timestamp: float
    importance: EventImportance

    # Event details
    description: str
    location_id: Optional[str] = None
    npc_id: Optional[str] = None
    quest_id: Optional[str] = None
    subject: Optional[Subject] = None

    # Causality
    caused_by: List[str] = field(default_factory=list)  # Event IDs that caused this
    enables: List[str] = field(default_factory=list)    # Event IDs this enables

    # Player state at time of event
    player_level: Optional[int] = None
    player_performance: Optional[float] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Event {self.event_type.value} @ {self.location_id}: {self.description[:50]}>"


@dataclass
class NarrativeThread:
    """A connected sequence of narrative events forming a story arc"""
    thread_id: str
    name: str
    description: str
    events: List[str] = field(default_factory=list)  # Event IDs in chronological order
    active: bool = True
    subject: Optional[Subject] = None
    start_timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    end_timestamp: Optional[float] = None

    # Thread status
    completion_percentage: float = 0.0  # 0.0 to 1.0
    branching_points: List[str] = field(default_factory=list)  # Event IDs where choices were made

    def __repr__(self) -> str:
        status = "Active" if self.active else "Complete"
        return f"<Thread '{self.name}' ({status}): {len(self.events)} events, {self.completion_percentage:.0%} complete>"


@dataclass
class PlayerChoiceRecord:
    """Records a player choice and its narrative consequences"""
    choice_id: str
    event_id: str
    timestamp: float
    description: str
    options_presented: List[str]
    choice_made: str
    consequences: List[str] = field(default_factory=list)  # Event IDs of consequences
    regret_allowed: bool = True  # Can player revisit this choice?


class NarrativeMemory:
    """Storage and retrieval system for narrative events"""

    def __init__(self):
        self.events: Dict[str, NarrativeEvent] = {}
        self.threads: Dict[str, NarrativeThread] = {}
        self.choices: Dict[str, PlayerChoiceRecord] = {}

        # Indexes for fast retrieval
        self.events_by_quest: Dict[str, List[str]] = {}
        self.events_by_npc: Dict[str, List[str]] = {}
        self.events_by_location: Dict[str, List[str]] = {}
        self.events_by_subject: Dict[Subject, List[str]] = {}
        self.events_by_type: Dict[EventType, List[str]] = {}

        # Chronological index
        self.events_chronological: List[str] = []

    def add_event(self, event: NarrativeEvent) -> None:
        """Add a narrative event to memory"""
        self.events[event.event_id] = event
        self.events_chronological.append(event.event_id)

        # Update indexes
        if event.quest_id:
            if event.quest_id not in self.events_by_quest:
                self.events_by_quest[event.quest_id] = []
            self.events_by_quest[event.quest_id].append(event.event_id)

        if event.npc_id:
            if event.npc_id not in self.events_by_npc:
                self.events_by_npc[event.npc_id] = []
            self.events_by_npc[event.npc_id].append(event.event_id)

        if event.location_id:
            if event.location_id not in self.events_by_location:
                self.events_by_location[event.location_id] = []
            self.events_by_location[event.location_id].append(event.event_id)

        if event.subject:
            if event.subject not in self.events_by_subject:
                self.events_by_subject[event.subject] = []
            self.events_by_subject[event.subject].append(event.event_id)

        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event.event_id)

    def get_event(self, event_id: str) -> Optional[NarrativeEvent]:
        """Retrieve an event by ID"""
        return self.events.get(event_id)

    def get_recent_events(self, limit: int = 10, importance: Optional[EventImportance] = None) -> List[NarrativeEvent]:
        """Get most recent events, optionally filtered by importance"""
        recent_ids = self.events_chronological[-limit:]
        events = [self.events[eid] for eid in reversed(recent_ids)]

        if importance:
            events = [e for e in events if e.importance.value >= importance.value]

        return events

    def get_events_by_quest(self, quest_id: str) -> List[NarrativeEvent]:
        """Get all events related to a specific quest"""
        event_ids = self.events_by_quest.get(quest_id, [])
        return [self.events[eid] for eid in event_ids]

    def get_events_by_npc(self, npc_id: str) -> List[NarrativeEvent]:
        """Get all events involving a specific NPC"""
        event_ids = self.events_by_npc.get(npc_id, [])
        return [self.events[eid] for eid in event_ids]

    def get_events_by_location(self, location_id: str) -> List[NarrativeEvent]:
        """Get all events at a specific location"""
        event_ids = self.events_by_location.get(location_id, [])
        return [self.events[eid] for eid in event_ids]

    def get_causal_chain(self, event_id: str, max_depth: int = 5) -> List[NarrativeEvent]:
        """Get the causal chain leading to an event"""
        chain = []
        seen = set()

        def _trace_backwards(eid: str, depth: int):
            if depth > max_depth or eid in seen:
                return
            seen.add(eid)

            event = self.events.get(eid)
            if not event:
                return

            chain.append(event)

            for cause_id in event.caused_by:
                _trace_backwards(cause_id, depth + 1)

        _trace_backwards(event_id, 0)
        return list(reversed(chain))  # Chronological order

    def create_thread(self, name: str, description: str, subject: Optional[Subject] = None) -> NarrativeThread:
        """Create a new narrative thread"""
        thread = NarrativeThread(
            thread_id=str(uuid.uuid4()),
            name=name,
            description=description,
            subject=subject
        )
        self.threads[thread.thread_id] = thread
        return thread

    def add_event_to_thread(self, thread_id: str, event_id: str, is_branching: bool = False) -> None:
        """Add an event to a narrative thread"""
        thread = self.threads.get(thread_id)
        if not thread:
            return

        thread.events.append(event_id)

        if is_branching:
            thread.branching_points.append(event_id)

    def complete_thread(self, thread_id: str) -> None:
        """Mark a narrative thread as complete"""
        thread = self.threads.get(thread_id)
        if thread:
            thread.active = False
            thread.end_timestamp = datetime.now().timestamp()
            thread.completion_percentage = 1.0

    def record_choice(self, choice: PlayerChoiceRecord) -> None:
        """Record a player choice"""
        self.choices[choice.choice_id] = choice

    def get_narrative_context(
        self,
        npc_id: Optional[str] = None,
        location_id: Optional[str] = None,
        subject: Optional[Subject] = None,
        recent_events_count: int = 5
    ) -> Dict[str, Any]:
        """Get narrative context for generating NPC dialogue or quest descriptions"""
        context = {
            'recent_events': self.get_recent_events(limit=recent_events_count),
            'active_threads': [t for t in self.threads.values() if t.active],
        }

        if npc_id:
            context['npc_history'] = self.get_events_by_npc(npc_id)

        if location_id:
            context['location_history'] = self.get_events_by_location(location_id)

        if subject:
            event_ids = self.events_by_subject.get(subject, [])
            context['subject_events'] = [self.events[eid] for eid in event_ids[-10:]]

        return context

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'total_events': len(self.events),
            'total_threads': len(self.threads),
            'active_threads': sum(1 for t in self.threads.values() if t.active),
            'total_choices': len(self.choices),
            'events_by_type': {
                event_type.value: len(event_ids)
                for event_type, event_ids in self.events_by_type.items()
            },
            'events_by_importance': {
                importance.value: sum(1 for e in self.events.values() if e.importance == importance)
                for importance in EventImportance
            }
        }


class NarrativeMemoryHelper:
    """Helper functions for creating narrative events"""

    @staticmethod
    def create_quest_started_event(
        quest: Quest,
        npc_id: str,
        location_id: str,
        player_level: int
    ) -> NarrativeEvent:
        """Create event for quest started"""
        return NarrativeEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.QUEST_STARTED,
            timestamp=datetime.now().timestamp(),
            importance=EventImportance.LOW,
            description=f"Started quest: {quest.title}",
            location_id=location_id,
            npc_id=npc_id,
            quest_id=quest.quest_id,
            subject=quest.objective.subject if hasattr(quest.objective, 'subject') else None,
            player_level=player_level,
            metadata={'difficulty': quest.difficulty, 'quest_type': quest.quest_type.value}
        )

    @staticmethod
    def create_quest_completed_event(
        quest: Quest,
        npc_id: str,
        location_id: str,
        player_level: int,
        score: float,
        started_event_id: str
    ) -> NarrativeEvent:
        """Create event for quest completed"""
        importance = EventImportance.MEDIUM if score >= 0.8 else EventImportance.LOW

        return NarrativeEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.QUEST_COMPLETED,
            timestamp=datetime.now().timestamp(),
            importance=importance,
            description=f"Completed quest: {quest.title} (Score: {score:.1%})",
            location_id=location_id,
            npc_id=npc_id,
            quest_id=quest.quest_id,
            subject=quest.objective.subject if hasattr(quest.objective, 'subject') else None,
            player_level=player_level,
            player_performance=score,
            caused_by=[started_event_id],
            metadata={'score': score, 'quest_type': quest.quest_type.value}
        )

    @staticmethod
    def create_npc_interaction_event(
        npc_id: str,
        npc_name: str,
        location_id: str,
        dialogue_context: str,
        player_level: int
    ) -> NarrativeEvent:
        """Create event for NPC interaction"""
        return NarrativeEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.NPC_INTERACTION,
            timestamp=datetime.now().timestamp(),
            importance=EventImportance.TRIVIAL,
            description=f"Spoke with {npc_name}",
            location_id=location_id,
            npc_id=npc_id,
            player_level=player_level,
            metadata={'dialogue_context': dialogue_context}
        )

    @staticmethod
    def create_learning_milestone_event(
        objective: LearningObjective,
        location_id: str,
        npc_id: Optional[str],
        player_level: int,
        mastery_level: float
    ) -> NarrativeEvent:
        """Create event for learning milestone"""
        return NarrativeEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.LEARNING_MILESTONE,
            timestamp=datetime.now().timestamp(),
            importance=EventImportance.HIGH,
            description=f"Mastered: {objective.title}",
            location_id=location_id,
            npc_id=npc_id,
            subject=objective.subject,
            player_level=player_level,
            player_performance=mastery_level,
            metadata={'objective_id': objective.objective_id, 'bloom_level': objective.bloom_level.value}
        )

    @staticmethod
    def create_player_choice_event(
        description: str,
        options: List[str],
        choice: str,
        location_id: str,
        player_level: int
    ) -> Tuple[NarrativeEvent, PlayerChoiceRecord]:
        """Create event and record for player choice"""
        choice_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())

        event = NarrativeEvent(
            event_id=event_id,
            event_type=EventType.PLAYER_CHOICE,
            timestamp=datetime.now().timestamp(),
            importance=EventImportance.MEDIUM,
            description=description,
            location_id=location_id,
            player_level=player_level,
            metadata={'choice': choice, 'options': options}
        )

        choice_record = PlayerChoiceRecord(
            choice_id=choice_id,
            event_id=event_id,
            timestamp=event.timestamp,
            description=description,
            options_presented=options,
            choice_made=choice
        )

        return event, choice_record


class ConsistencyChecker:
    """Checks narrative consistency and detects contradictions"""

    def __init__(self, memory: NarrativeMemory):
        self.memory = memory

    def check_quest_consistency(self, quest_id: str) -> Dict[str, Any]:
        """Check if quest events are consistent"""
        events = self.memory.get_events_by_quest(quest_id)

        started = [e for e in events if e.event_type == EventType.QUEST_STARTED]
        completed = [e for e in events if e.event_type == EventType.QUEST_COMPLETED]
        failed = [e for e in events if e.event_type == EventType.QUEST_FAILED]

        issues = []

        # Check: Quest completed without being started
        if completed and not started:
            issues.append("Quest completed without being started")

        # Check: Multiple starts
        if len(started) > 1:
            issues.append(f"Quest started multiple times ({len(started)})")

        # Check: Multiple endings
        if len(completed) + len(failed) > 1:
            issues.append(f"Quest has multiple endings")

        # Check: Quest failed AND completed
        if completed and failed:
            issues.append("Quest both completed and failed")

        return {
            'quest_id': quest_id,
            'consistent': len(issues) == 0,
            'issues': issues,
            'event_count': len(events)
        }

    def check_npc_consistency(self, npc_id: str) -> Dict[str, Any]:
        """Check if NPC interactions are consistent"""
        events = self.memory.get_events_by_npc(npc_id)

        locations = set(e.location_id for e in events if e.location_id)

        issues = []

        # Check: NPC in too many locations (shouldn't move around unless specified)
        if len(locations) > 3:
            issues.append(f"NPC appeared in {len(locations)} different locations")

        return {
            'npc_id': npc_id,
            'consistent': len(issues) == 0,
            'issues': issues,
            'event_count': len(events),
            'locations': list(locations)
        }

    def check_causal_consistency(self, event_id: str) -> Dict[str, Any]:
        """Check if causal chain is consistent"""
        event = self.memory.get_event(event_id)
        if not event:
            return {'consistent': False, 'issues': ['Event not found']}

        issues = []

        # Check: All causes exist
        for cause_id in event.caused_by:
            if cause_id not in self.memory.events:
                issues.append(f"Cause event {cause_id} not found")

        # Check: Causes happened before this event
        for cause_id in event.caused_by:
            cause = self.memory.get_event(cause_id)
            if cause and cause.timestamp > event.timestamp:
                issues.append(f"Cause event {cause_id} happened AFTER this event (time paradox)")

        return {
            'event_id': event_id,
            'consistent': len(issues) == 0,
            'issues': issues
        }


# Demo function
def demo_narrative_memory():
    """Demonstrate the narrative memory system"""
    print("\n" + "="*60)
    print("EduVerse Narrative Memory Demo")
    print("="*60 + "\n")

    # Create narrative memory
    memory = NarrativeMemory()
    helper = NarrativeMemoryHelper()

    # Create a narrative thread for a math learning journey
    thread = memory.create_thread(
        name="Introduction to Algebra",
        description="Student's first journey into algebraic thinking",
        subject=Subject.MATH
    )
    print(f"Created thread: {thread}")
    print()

    # Simulate a quest sequence
    print("Simulating quest sequence...")
    print("-" * 60)

    # Event 1: NPC greeting
    event1 = helper.create_npc_interaction_event(
        npc_id="npc_math_teacher_001",
        npc_name="Ms. Rodriguez",
        location_id="school_math_classroom",
        dialogue_context="greeting",
        player_level=5
    )
    memory.add_event(event1)
    memory.add_event_to_thread(thread.thread_id, event1.event_id)
    print(f"1. {event1.description} at {event1.location_id}")

    # Event 2: Quest started (fake quest object)
    from dataclasses import dataclass as dc

    @dc
    class FakeObjective:
        subject: Subject

    @dc
    class FakeQuest:
        quest_id: str
        title: str
        quest_type: QuestType
        difficulty: float
        objective: FakeObjective

    quest = FakeQuest(
        quest_id="quest_algebra_intro_001",
        title="Introduction to Variables",
        quest_type=QuestType.TUTORIAL,
        difficulty=0.3,
        objective=FakeObjective(subject=Subject.MATH)
    )

    event2 = helper.create_quest_started_event(
        quest=quest,
        npc_id="npc_math_teacher_001",
        location_id="school_math_classroom",
        player_level=5
    )
    memory.add_event(event2)
    memory.add_event_to_thread(thread.thread_id, event2.event_id)
    print(f"2. {event2.description}")

    # Event 3: Player choice during quest
    event3, choice = helper.create_player_choice_event(
        description="Choose learning approach for variables",
        options=["Visual diagrams", "Algebraic notation", "Real-world examples"],
        choice="Visual diagrams",
        location_id="school_math_classroom",
        player_level=5
    )
    event3.caused_by = [event2.event_id]
    memory.add_event(event3)
    memory.record_choice(choice)
    memory.add_event_to_thread(thread.thread_id, event3.event_id, is_branching=True)
    print(f"3. {event3.description} -> Choice: {choice.choice_made}")

    # Event 4: Quest completed
    event4 = helper.create_quest_completed_event(
        quest=quest,
        npc_id="npc_math_teacher_001",
        location_id="school_math_classroom",
        player_level=5,
        score=0.85,
        started_event_id=event2.event_id
    )
    memory.add_event(event4)
    memory.add_event_to_thread(thread.thread_id, event4.event_id)
    print(f"4. {event4.description}")

    # Event 5: Learning milestone
    @dc
    class FakeLO:
        objective_id: str
        title: str
        subject: Subject

        @dc
        class BloomLevel:
            value: str

        bloom_level: 'FakeLO.BloomLevel'

    objective = FakeLO(
        objective_id="math_algebra_variables",
        title="Understanding Variables",
        subject=Subject.MATH,
        bloom_level=FakeLO.BloomLevel(value="understand")
    )

    event5 = helper.create_learning_milestone_event(
        objective=objective,
        location_id="school_math_classroom",
        npc_id="npc_math_teacher_001",
        player_level=5,
        mastery_level=0.85
    )
    event5.caused_by = [event4.event_id]
    memory.add_event(event5)
    memory.add_event_to_thread(thread.thread_id, event5.event_id)
    print(f"5. {event5.description}")

    print()

    # Show narrative context
    print("-" * 60)
    print("Narrative Context for Ms. Rodriguez")
    print("-" * 60)

    context = memory.get_narrative_context(
        npc_id="npc_math_teacher_001",
        location_id="school_math_classroom",
        recent_events_count=3
    )

    print(f"\nRecent Events ({len(context['recent_events'])}):")
    for event in context['recent_events']:
        print(f"  - {event.description} [{event.importance.name}]")

    print(f"\nNPC History ({len(context['npc_history'])}):")
    for event in context['npc_history']:
        print(f"  - {event.description}")

    print(f"\nActive Threads: {len(context['active_threads'])}")
    for t in context['active_threads']:
        print(f"  - {t.name}: {len(t.events)} events")

    # Show causal chain
    print("\n" + "-" * 60)
    print("Causal Chain for Learning Milestone")
    print("-" * 60 + "\n")

    chain = memory.get_causal_chain(event5.event_id)
    for i, event in enumerate(chain, 1):
        print(f"{i}. {event.description}")
        print(f"   -> {event.event_type.value} [{event.importance.name}]")

    # Consistency check
    print("\n" + "-" * 60)
    print("Consistency Checks")
    print("-" * 60 + "\n")

    checker = ConsistencyChecker(memory)

    quest_check = checker.check_quest_consistency(quest.quest_id)
    print(f"Quest Consistency: {quest_check['consistent']}")
    if quest_check['issues']:
        for issue in quest_check['issues']:
            print(f"  - Issue: {issue}")
    else:
        print("  - No issues found")

    npc_check = checker.check_npc_consistency("npc_math_teacher_001")
    print(f"\nNPC Consistency: {npc_check['consistent']}")
    print(f"  - Events: {npc_check['event_count']}")
    print(f"  - Locations: {len(npc_check['locations'])}")

    causal_check = checker.check_causal_consistency(event5.event_id)
    print(f"\nCausal Consistency: {causal_check['consistent']}")
    if causal_check['issues']:
        for issue in causal_check['issues']:
            print(f"  - Issue: {issue}")
    else:
        print("  - No issues found")

    # Summary
    print("\n" + "-" * 60)
    print("Memory Summary")
    print("-" * 60 + "\n")

    summary = memory.get_summary()
    print(f"Total Events: {summary['total_events']}")
    print(f"Total Threads: {summary['total_threads']}")
    print(f"Active Threads: {summary['active_threads']}")
    print(f"Total Choices: {summary['total_choices']}")

    print("\nEvents by Type:")
    for event_type, count in summary['events_by_type'].items():
        print(f"  {event_type}: {count}")

    print("\nEvents by Importance:")
    for importance, count in summary['events_by_importance'].items():
        if count > 0:
            print(f"  {EventImportance(importance).name}: {count}")

    print("\n" + "="*60)
    print("Narrative memory ready! Story events tracked with causality.")
    print("="*60)


if __name__ == "__main__":
    demo_narrative_memory()
