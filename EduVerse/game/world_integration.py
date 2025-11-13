"""
World Integration System

Integrates worlds, NPCs, quests, and narrative memory into a unified
educational game experience with story-aware NPCs and persistent memory.

Created: 2025-11-13 (Week 3 - DreamWeaver Phase 1)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from EduVerse.education.player_model import PlayerModel
from EduVerse.education.curriculum import CurriculumFramework, Subject
from EduVerse.game.world_generator import WorldGenerator, WorldTheme, Location
from EduVerse.game.npc_system import (
    NPCRegistry, NPCProfile, NPCDialogueManager, DialogueContext,
    DialogueResponse, create_default_npc_registry
)
from EduVerse.game.quest_engine import QuestEngine, Quest, QuestType
from EduVerse.game.narrative_memory import (
    NarrativeMemory, NarrativeMemoryHelper, NarrativeThread,
    EventType, EventImportance, ConsistencyChecker
)
from EduVerse.game.minigame_framework import MinigameResult


@dataclass
class GameState:
    """Complete state of the game world"""
    player: PlayerModel
    current_location_id: str
    visited_locations: List[str] = field(default_factory=list)
    current_quest: Optional[Quest] = None
    active_npc: Optional[str] = None  # NPC currently talking to
    session_start_timestamp: float = field(default_factory=lambda: __import__('time').time())

    def __repr__(self) -> str:
        return f"<GameState: {self.player.name} @ {self.current_location_id}, Quest: {self.current_quest.title if self.current_quest else 'None'}>"


class WorldIntegration:
    """
    Unified system integrating worlds, NPCs, quests, and narrative memory.

    This is the main controller for the educational game experience.
    """

    def __init__(
        self,
        player: PlayerModel,
        curriculum: CurriculumFramework,
        starting_location: str = "school_lobby"
    ):
        # Core systems
        self.player = player
        self.curriculum = curriculum
        self.quest_engine = QuestEngine(curriculum, player)

        # World and NPCs
        self.world_generator = WorldGenerator()
        self.npc_registry = create_default_npc_registry()

        # Narrative tracking
        self.narrative_memory = NarrativeMemory()
        self.memory_helper = NarrativeMemoryHelper()
        self.consistency_checker = ConsistencyChecker(self.narrative_memory)

        # Game state
        self.game_state = GameState(
            player=player,
            current_location_id=starting_location
        )

        # Generate all worlds
        self.locations: Dict[str, Location] = {}
        self._generate_all_worlds()

        # Create narrative threads for each subject
        self.subject_threads: Dict[Subject, NarrativeThread] = {}
        self._create_subject_threads()

        # Track active quest event IDs for causal linking
        self.active_quest_events: Dict[str, str] = {}  # quest_id -> quest_started_event_id

    def _get_quest_subject(self, quest: Quest) -> Optional[Subject]:
        """Get the primary subject for a quest"""
        if not quest.learning_objectives:
            return None

        # Get first objective's subject
        obj_id = quest.learning_objectives[0]
        obj = self.curriculum.get_objective(obj_id)
        return obj.subject if obj else None

    def _generate_all_worlds(self):
        """Generate all three themed worlds"""
        # School world
        school_locs = self.world_generator.generate_world(
            WorldTheme.SCHOOL,
            [Subject.MATH, Subject.SCIENCE, Subject.ELA, Subject.SOCIAL_STUDIES, Subject.AI_READINESS]
        )
        self.locations.update(school_locs)

        # Fantasy world
        fantasy_locs = self.world_generator.generate_world(
            WorldTheme.FANTASY,
            [Subject.MATH, Subject.SCIENCE, Subject.ELA]
        )
        self.locations.update(fantasy_locs)

        # Space station
        space_locs = self.world_generator.generate_world(
            WorldTheme.SPACE,
            [Subject.MATH, Subject.SCIENCE, Subject.AI_READINESS]
        )
        self.locations.update(space_locs)

    def _create_subject_threads(self):
        """Create narrative threads for each subject"""
        for subject in Subject:
            thread = self.narrative_memory.create_thread(
                name=f"{subject.value.title()} Learning Journey",
                description=f"The student's learning progression in {subject.value}",
                subject=subject
            )
            self.subject_threads[subject] = thread

    def get_current_location(self) -> Optional[Location]:
        """Get the current location"""
        return self.locations.get(self.game_state.current_location_id)

    def get_npcs_at_current_location(self) -> List[NPCProfile]:
        """Get all NPCs at the current location"""
        return self.npc_registry.get_npcs_at_location(self.game_state.current_location_id)

    def travel_to(self, location_id: str) -> Dict[str, Any]:
        """Travel to a new location"""
        if location_id not in self.locations:
            return {
                'success': False,
                'message': f"Location {location_id} not found"
            }

        old_location = self.game_state.current_location_id
        self.game_state.current_location_id = location_id

        if location_id not in self.game_state.visited_locations:
            self.game_state.visited_locations.append(location_id)

        # Create travel event
        event = self.memory_helper.create_world_change_event(
            description=f"Traveled from {old_location} to {location_id}",
            location_id=location_id,
            player_level=len(self.player.skills)
        )
        self.narrative_memory.add_event(event)

        location = self.locations[location_id]
        npcs = self.get_npcs_at_current_location()

        return {
            'success': True,
            'location': location,
            'npcs': npcs,
            'message': f"Arrived at {location.name}"
        }

    def talk_to_npc(self, npc_id: str, context: DialogueContext = DialogueContext.GREETING) -> Optional[DialogueResponse]:
        """Talk to an NPC and get narrative-aware dialogue"""
        npc = self.npc_registry.get_npc(npc_id)
        if not npc:
            return None

        # Set active NPC
        self.game_state.active_npc = npc_id

        # Get narrative context
        narrative_context = self.narrative_memory.get_narrative_context(
            npc_id=npc_id,
            location_id=self.game_state.current_location_id,
            recent_events_count=5
        )

        # Calculate player performance (average of recent quest scores)
        recent_events = narrative_context['recent_events']
        quest_events = [e for e in recent_events if e.event_type == EventType.QUEST_COMPLETED]

        if quest_events:
            avg_performance = sum(e.player_performance for e in quest_events if e.player_performance) / len(quest_events)
        else:
            avg_performance = 0.5  # Neutral

        # Generate dialogue
        dialogue_manager = self.npc_registry.get_dialogue_manager(npc_id)
        if not dialogue_manager:
            return None

        response = dialogue_manager.generate_dialogue(
            context=context,
            player_performance=avg_performance,
            quest=self.game_state.current_quest
        )

        # Create interaction event
        event = self.memory_helper.create_npc_interaction_event(
            npc_id=npc_id,
            npc_name=npc.name,
            location_id=self.game_state.current_location_id,
            dialogue_context=context.value,
            player_level=len(self.player.skills)
        )
        self.narrative_memory.add_event(event)

        # Add to subject thread if NPC teaches a subject
        if npc.subjects:
            subject = npc.subjects[0]
            thread = self.subject_threads.get(subject)
            if thread:
                self.narrative_memory.add_event_to_thread(thread.thread_id, event.event_id)

        return response

    def get_quest_from_npc(self, npc_id: str) -> Optional[Quest]:
        """Get a quest from an NPC based on player progress"""
        npc = self.npc_registry.get_npc(npc_id)
        if not npc or not npc.can_give_quests:
            return None

        # Generate appropriate quest
        quests = self.quest_engine.generate_recommended_quests(count=3)

        # Filter by NPC's subjects and quest types
        suitable_quests = []
        for q in quests:
            if q.quest_type not in npc.quest_types:
                continue

            # Check if any learning objective matches NPC's subjects
            quest_subjects = set()
            for obj_id in q.learning_objectives:
                obj = self.curriculum.get_objective(obj_id)
                if obj:
                    quest_subjects.add(obj.subject)

            # Quest is suitable if it teaches any of NPC's subjects
            if any(s in npc.subjects for s in quest_subjects):
                suitable_quests.append(q)

        if not suitable_quests:
            return None

        return suitable_quests[0]

    def start_quest(self, quest: Quest, npc_id: str) -> Dict[str, Any]:
        """Start a quest with narrative tracking"""
        if self.game_state.current_quest:
            return {
                'success': False,
                'message': 'Already on a quest. Complete it first.'
            }

        self.game_state.current_quest = quest

        # Create quest started event
        event = self.memory_helper.create_quest_started_event(
            quest=quest,
            npc_id=npc_id,
            location_id=self.game_state.current_location_id,
            player_level=len(self.player.skills)
        )
        self.narrative_memory.add_event(event)

        # Track for later causal linking
        self.active_quest_events[quest.id] = event.event_id

        # Add to subject thread
        quest_subject = self._get_quest_subject(quest)
        if quest_subject:
            thread = self.subject_threads.get(quest_subject)
            if thread:
                self.narrative_memory.add_event_to_thread(thread.thread_id, event.event_id)

        return {
            'success': True,
            'quest': quest,
            'message': f"Quest started: {quest.title}"
        }

    def complete_quest(self, minigame_result: MinigameResult, npc_id: str) -> Dict[str, Any]:
        """Complete a quest with narrative tracking and rewards"""
        if not self.game_state.current_quest:
            return {
                'success': False,
                'message': 'No active quest'
            }

        quest = self.game_state.current_quest
        score = minigame_result.score

        # Get quest started event for causal linking
        started_event_id = self.active_quest_events.get(quest.id, "")

        # Create quest completed event
        event = self.memory_helper.create_quest_completed_event(
            quest=quest,
            npc_id=npc_id,
            location_id=self.game_state.current_location_id,
            player_level=len(self.player.skills),
            score=score,
            started_event_id=started_event_id
        )
        self.narrative_memory.add_event(event)

        # Add to subject thread
        quest_subject = self._get_quest_subject(quest)
        if quest_subject:
            thread = self.subject_threads.get(quest_subject)
            if thread:
                self.narrative_memory.add_event_to_thread(thread.thread_id, event.event_id)

        # Apply rewards
        rewards = quest.calculate_rewards(score)
        self.player.add_experience(rewards['xp'])

        for skill_name, skill_xp in rewards['skill_xp'].items():
            if skill_name in self.player.skills:
                self.player.skills[skill_name] += skill_xp

        # Update quest outcome in player's Thompson Sampling bandit
        self.player.update_quest_outcome(
            difficulty=quest.difficulty,
            success=minigame_result.success,
            score=score
        )

        # Check for learning milestone
        mastery_level = score

        if score >= 0.75 and quest.learning_objectives:  # High performance = mastery
            # Get first learning objective
            obj_id = quest.learning_objectives[0]
            objective = self.curriculum.get_objective(obj_id)

            if objective:
                milestone_event = self.memory_helper.create_learning_milestone_event(
                    objective=objective,
                    location_id=self.game_state.current_location_id,
                    npc_id=npc_id,
                    player_level=len(self.player.skills),
                    mastery_level=mastery_level
                )
                milestone_event.caused_by = [event.event_id]
                self.narrative_memory.add_event(milestone_event)

                # Add milestone to subject thread
                quest_subject = self._get_quest_subject(quest)
                if quest_subject:
                    thread = self.subject_threads.get(quest_subject)
                    if thread:
                        self.narrative_memory.add_event_to_thread(
                            thread.thread_id,
                            milestone_event.event_id
                        )

        # Clear active quest
        self.game_state.current_quest = None
        if quest.id in self.active_quest_events:
            del self.active_quest_events[quest.id]

        return {
            'success': True,
            'score': score,
            'rewards': rewards,
            'message': f"Quest completed! Score: {score:.1%}"
        }

    def get_available_locations(self) -> List[Location]:
        """Get all locations connected to current location"""
        # For now, return all locations (can add travel restrictions later)
        return list(self.locations.values())

    def get_narrative_summary(self) -> Dict[str, Any]:
        """Get summary of player's story so far"""
        summary = self.narrative_memory.get_summary()

        # Add thread progress
        thread_summary = []
        for subject, thread in self.subject_threads.items():
            if thread.events:
                thread_summary.append({
                    'subject': subject.value,
                    'events': len(thread.events),
                    'active': thread.active,
                    'completion': thread.completion_percentage
                })

        summary['threads'] = thread_summary

        # Add recent significant events
        significant_events = self.narrative_memory.get_recent_events(
            limit=10,
            importance=EventImportance.MEDIUM
        )
        summary['recent_significant_events'] = [
            {
                'type': e.event_type.value,
                'description': e.description,
                'importance': e.importance.name
            }
            for e in significant_events
        ]

        return summary

    def check_world_consistency(self) -> Dict[str, Any]:
        """Check narrative consistency across all systems"""
        issues = []

        # Check all completed quests
        completed_quests = self.narrative_memory.events_by_type.get(
            EventType.QUEST_COMPLETED, []
        )

        for event_id in completed_quests:
            event = self.narrative_memory.get_event(event_id)
            if event and event.quest_id:
                check = self.consistency_checker.check_quest_consistency(event.quest_id)
                if not check['consistent']:
                    issues.extend(check['issues'])

        # Check all NPCs
        for npc_id in self.npc_registry.npcs.keys():
            check = self.consistency_checker.check_npc_consistency(npc_id)
            if not check['consistent']:
                issues.extend(check['issues'])

        return {
            'consistent': len(issues) == 0,
            'issues': issues,
            'total_events': len(self.narrative_memory.events),
            'total_npcs': len(self.npc_registry.npcs),
            'total_locations': len(self.locations)
        }


# Add missing helper method to NarrativeMemoryHelper
def create_world_change_event(description: str, location_id: str, player_level: int):
    """Create event for world change (travel)"""
    from EduVerse.game.narrative_memory import NarrativeEvent, EventType, EventImportance
    import uuid
    from datetime import datetime

    return NarrativeEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.WORLD_CHANGE,
        timestamp=datetime.now().timestamp(),
        importance=EventImportance.TRIVIAL,
        description=description,
        location_id=location_id,
        player_level=player_level
    )

# Monkey-patch the helper
NarrativeMemoryHelper.create_world_change_event = staticmethod(create_world_change_event)


# Demo function
def demo_world_integration():
    """Demonstrate the integrated world system"""
    from HoloLoom.memory.graph import KG

    print("\n" + "="*70)
    print("EduVerse World Integration Demo")
    print("="*70 + "\n")

    # Create player and curriculum
    kg = KG()
    player = PlayerModel(
        student_id="student_001",
        name="Alex",
        grade=8,
        knowledge_graph=kg
    )

    curriculum = CurriculumFramework()

    # Create integrated world
    print("Initializing integrated world...")
    world = WorldIntegration(
        player=player,
        curriculum=curriculum,
        starting_location="school_lobby"
    )

    print(f"[OK] Generated {len(world.locations)} locations across 3 worlds")
    print(f"[OK] Registered {len(world.npc_registry.npcs)} NPCs")
    print(f"[OK] Created {len(world.subject_threads)} subject narrative threads")
    print()

    # Show current state
    print("-" * 70)
    print("Current State")
    print("-" * 70)
    location = world.get_current_location()
    print(f"Location: {location.name}")
    print(f"Description: {location.description[:80]}...")

    npcs = world.get_npcs_at_current_location()
    print(f"\nNPCs here ({len(npcs)}):")
    for npc in npcs:
        print(f"  - {npc.name} ({npc.role.value})")
    print()

    # Travel to math classroom
    print("-" * 70)
    print("Action: Travel to Math Classroom")
    print("-" * 70)
    result = world.travel_to("school_math_classroom")
    print(f"[OK] {result['message']}")
    print(f"NPCs: {', '.join(npc.name for npc in result['npcs'])}")
    print()

    # Talk to Ms. Rodriguez
    print("-" * 70)
    print("Action: Greet Ms. Rodriguez")
    print("-" * 70)
    dialogue = world.talk_to_npc("npc_math_teacher_001", DialogueContext.GREETING)
    if dialogue:
        print(f"[{dialogue.npc_name}]: {dialogue.text}")
        print(f"(Tone: {dialogue.emotional_tone})")
    print()

    # Get a quest
    print("-" * 70)
    print("Action: Request Quest")
    print("-" * 70)
    quest = world.get_quest_from_npc("npc_math_teacher_001")
    if quest:
        print(f"Quest offered: {quest.title}")
        print(f"Type: {quest.quest_type.value}")
        print(f"Difficulty: {quest.difficulty:.2f}")
        if quest.learning_objectives:
            obj_id = quest.learning_objectives[0]
            obj = curriculum.get_objective(obj_id)
            if obj:
                print(f"Objective: {obj.title}")

        # Start quest
        start_result = world.start_quest(quest, "npc_math_teacher_001")
        print(f"\n[OK] {start_result['message']}")
    else:
        print("No suitable quest available from this NPC.")
    print()

    # Simulate quest completion (only if quest was started)
    if quest:
        print("-" * 70)
        print("Action: Complete Quest (simulated)")
        print("-" * 70)

        # Create fake minigame result
        @dataclass
        class FakeResult:
            success: bool = True
            score: float = 0.85
            time_taken: float = 120.0

        minigame_result = FakeResult()

        complete_result = world.complete_quest(minigame_result, "npc_math_teacher_001")
        print(f"[OK] {complete_result['message']}")
        if 'rewards' in complete_result:
            print(f"Rewards: {complete_result['rewards']}")
        print()

    # Talk to NPC again (success context)
    print("-" * 70)
    print("Action: Talk to Ms. Rodriguez (after success)")
    print("-" * 70)
    dialogue = world.talk_to_npc("npc_math_teacher_001", DialogueContext.SUCCESS)
    if dialogue:
        print(f"[{dialogue.npc_name}]: {dialogue.text}")
        print(f"(Tone: {dialogue.emotional_tone})")
    print()

    # Show narrative summary
    print("-" * 70)
    print("Narrative Summary")
    print("-" * 70)
    summary = world.get_narrative_summary()
    print(f"Total Events: {summary['total_events']}")
    print(f"Total Threads: {summary['total_threads']}")
    print(f"Active Threads: {summary['active_threads']}")

    print("\nSubject Progress:")
    for thread_info in summary['threads']:
        print(f"  {thread_info['subject']}: {thread_info['events']} events")

    print("\nRecent Significant Events:")
    for event in summary['recent_significant_events'][:5]:
        print(f"  [{event['importance']}] {event['description']}")
    print()

    # Consistency check
    print("-" * 70)
    print("World Consistency Check")
    print("-" * 70)
    consistency = world.check_world_consistency()
    print(f"Status: {'[OK] CONSISTENT' if consistency['consistent'] else '[FAIL] ISSUES FOUND'}")
    print(f"Total Events: {consistency['total_events']}")
    print(f"Total NPCs: {consistency['total_npcs']}")
    print(f"Total Locations: {consistency['total_locations']}")

    if consistency['issues']:
        print("\nIssues:")
        for issue in consistency['issues']:
            print(f"  - {issue}")
    else:
        print("  No issues found - narrative is consistent!")
    print()

    print("="*70)
    print("World integration complete! All systems working together.")
    print("="*70)


if __name__ == "__main__":
    demo_world_integration()
