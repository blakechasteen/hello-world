"""
Game Session Manager

Integrates all EduVerse systems: World Generator, NPC System, Quest Engine,
and Narrative Memory to create a cohesive learning experience.

Created: 2025-11-13 (Week 3 - DreamWeaver Phase 1 Integration)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from EduVerse.education.curriculum import CurriculumFramework, Subject, LearningObjective
from EduVerse.education.player_model import PlayerModel
from EduVerse.game.quest_engine import QuestEngine, Quest
from EduVerse.game.world_generator import WorldGenerator, WorldTheme, Location
from EduVerse.game.npc_system import (
    NPCRegistry, NPCProfile, DialogueContext, DialogueResponse,
    create_default_npc_registry
)
from EduVerse.game.narrative_memory import (
    NarrativeMemory, NarrativeMemoryHelper, ConsistencyChecker,
    EventImportance, NarrativeThread
)


@dataclass
class GameState:
    """Current state of the game session"""
    current_location_id: str
    current_world: WorldTheme
    player_model: PlayerModel
    turn_number: int = 0
    quest_in_progress: Optional[Quest] = None
    quest_start_event_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"<GameState Turn {self.turn_number} @ {self.current_location_id}>"


class GameSession:
    """Main game session manager integrating all systems"""

    def __init__(
        self,
        player_model: PlayerModel,
        curriculum: CurriculumFramework,
        start_world: WorldTheme = WorldTheme.SCHOOL,
        start_location: str = "school_lobby"
    ):
        # Core systems
        self.player = player_model
        self.curriculum = curriculum
        self.quest_engine = QuestEngine(curriculum, player_model)
        self.narrative_memory = NarrativeMemory()
        self.narrative_helper = NarrativeMemoryHelper()
        self.consistency_checker = ConsistencyChecker(self.narrative_memory)

        # World and NPCs
        self.world_generator = WorldGenerator()
        self.npc_registry = create_default_npc_registry()

        # Generate all worlds
        self.worlds: Dict[WorldTheme, Dict[str, Location]] = {
            WorldTheme.SCHOOL: self.world_generator.generate_world(
                WorldTheme.SCHOOL, list(Subject)
            ),
            WorldTheme.FANTASY: self.world_generator.generate_world(
                WorldTheme.FANTASY, list(Subject)
            ),
            WorldTheme.SPACE: self.world_generator.generate_world(
                WorldTheme.SPACE, list(Subject)
            ),
        }

        # Game state
        self.state = GameState(
            current_location_id=start_location,
            current_world=start_world,
            player_model=player_model
        )

        # Create initial narrative thread
        self.main_thread = self.narrative_memory.create_thread(
            name=f"{player_model.name}'s Learning Journey",
            description=f"Following {player_model.name} through their educational adventure"
        )

    def get_player_level(self) -> int:
        """Calculate player level from XP (100 XP per level)"""
        return self.player.stats.total_xp // 100

    def get_current_location(self) -> Location:
        """Get the current location object"""
        world = self.worlds[self.state.current_world]
        return world[self.state.current_location_id]

    def get_npcs_at_current_location(self) -> List[NPCProfile]:
        """Get all NPCs at current location"""
        return self.npc_registry.get_npcs_at_location(self.state.current_location_id)

    def move_to_location(self, location_id: str, world: Optional[WorldTheme] = None) -> Tuple[bool, str]:
        """Move player to a new location"""
        target_world = world or self.state.current_world
        world_locations = self.worlds.get(target_world)

        if not world_locations or location_id not in world_locations:
            return False, f"Location {location_id} not found in {target_world.value} world"

        self.state.current_location_id = location_id
        self.state.current_world = target_world
        return True, f"Moved to {location_id}"

    def greet_npc(self, npc_id: str) -> Optional[DialogueResponse]:
        """Greet an NPC and record the interaction"""
        npc = self.npc_registry.get_npc(npc_id)
        if not npc:
            return None

        # Generate greeting dialogue
        dialogue = self.npc_registry.generate_dialogue(
            npc_id,
            DialogueContext.GREETING,
            player_performance=self.player.calculate_overall_performance()
        )

        # Record interaction in narrative memory
        event = self.narrative_helper.create_npc_interaction_event(
            npc_id=npc_id,
            npc_name=npc.name,
            location_id=self.state.current_location_id,
            dialogue_context="greeting",
            player_level=self.get_player_level()
        )
        self.narrative_memory.add_event(event)
        self.narrative_memory.add_event_to_thread(self.main_thread.thread_id, event.event_id)

        return dialogue

    def request_quest_from_npc(self, npc_id: str) -> Tuple[Optional[Quest], Optional[DialogueResponse]]:
        """Request a quest from an NPC"""
        npc = self.npc_registry.get_npc(npc_id)
        if not npc or not npc.can_give_quests:
            return None, None

        # Check narrative context - has NPC given quest recently?
        npc_events = self.narrative_memory.get_events_by_npc(npc_id)
        recent_quests = [
            e for e in npc_events[-5:]
            if e.event_type.value == "quest_started"
        ]

        if len(recent_quests) > 0 and self.state.quest_in_progress:
            # NPC asks about current quest
            dialogue = self.npc_registry.generate_dialogue(
                npc_id,
                DialogueContext.ENCOURAGEMENT,
                player_performance=self.player.calculate_overall_performance()
            )
            if dialogue:
                dialogue.text = f"How's the quest I gave you going? {dialogue.text}"
            return None, dialogue

        # Generate appropriate quest for NPC's subjects
        quests = self.quest_engine.generate_recommended_quests(count=5)

        # Filter to NPC's subjects
        npc_quests = [
            q for q in quests
            if hasattr(q.objective, 'subject') and q.objective.subject in npc.subjects
        ]

        if not npc_quests:
            # NPC has no quests available
            return None, None

        quest = npc_quests[0]

        # Generate quest offer dialogue
        dialogue = self.npc_registry.generate_dialogue(
            npc_id,
            DialogueContext.QUEST_OFFER,
            quest=quest
        )

        return quest, dialogue

    def accept_quest(self, quest: Quest, npc_id: str) -> bool:
        """Accept a quest from an NPC"""
        if self.state.quest_in_progress:
            return False

        self.state.quest_in_progress = quest

        # Record quest started event
        event = self.narrative_helper.create_quest_started_event(
            quest=quest,
            npc_id=npc_id,
            location_id=self.state.current_location_id,
            player_level=self.get_player_level()
        )
        self.narrative_memory.add_event(event)
        self.narrative_memory.add_event_to_thread(self.main_thread.thread_id, event.event_id)
        self.state.quest_start_event_id = event.event_id

        return True

    def complete_current_quest(self, score: float, npc_id: str) -> Tuple[bool, Optional[DialogueResponse]]:
        """Complete the current quest and get NPC response"""
        if not self.state.quest_in_progress:
            return False, None

        quest = self.state.quest_in_progress

        # Record quest completed event
        event = self.narrative_helper.create_quest_completed_event(
            quest=quest,
            npc_id=npc_id,
            location_id=self.state.current_location_id,
            player_level=self.get_player_level(),
            score=score,
            started_event_id=self.state.quest_start_event_id or "unknown"
        )
        self.narrative_memory.add_event(event)
        self.narrative_memory.add_event_to_thread(self.main_thread.thread_id, event.event_id)

        # Apply rewards to player
        rewards = self.quest_engine.calculate_rewards(quest, success=(score >= 0.6), score=score)
        self.player.stats.total_xp += rewards['xp']

        # Check if learning milestone reached
        if hasattr(quest.objective, 'objective_id'):
            mastery = self.player.skills.get(quest.objective.objective_id, 0.0)
            if mastery >= 0.8:
                milestone_event = self.narrative_helper.create_learning_milestone_event(
                    objective=quest.objective,
                    location_id=self.state.current_location_id,
                    npc_id=npc_id,
                    player_level=self.get_player_level(),
                    mastery_level=mastery
                )
                milestone_event.caused_by = [event.event_id]
                self.narrative_memory.add_event(milestone_event)
                self.narrative_memory.add_event_to_thread(
                    self.main_thread.thread_id,
                    milestone_event.event_id
                )

        # Generate NPC response
        dialogue = self.npc_registry.generate_dialogue(
            npc_id,
            DialogueContext.QUEST_COMPLETE,
            player_performance=score,
            quest=quest
        )

        # Clear quest
        self.state.quest_in_progress = None
        self.state.quest_start_event_id = None

        return True, dialogue

    def get_narrative_context_for_npc(self, npc_id: str) -> Dict[str, Any]:
        """Get narrative context for an NPC (for contextual dialogue)"""
        npc = self.npc_registry.get_npc(npc_id)
        if not npc:
            return {}

        context = self.narrative_memory.get_narrative_context(
            npc_id=npc_id,
            location_id=self.state.current_location_id,
            subject=npc.subjects[0] if npc.subjects else None,
            recent_events_count=5
        )

        # Add player performance
        context['player_performance'] = self.player.calculate_overall_performance()
        context['player_level'] = self.get_player_level()

        return context

    def explore_location(self) -> Dict[str, Any]:
        """Get information about current location"""
        location = self.get_current_location()
        npcs = self.get_npcs_at_current_location()
        location_events = self.narrative_memory.get_events_by_location(
            self.state.current_location_id
        )

        return {
            'location': location,
            'npcs': npcs,
            'past_events_here': location_events[-5:],  # Last 5 events
            'subject_focus': location.subject,
            'atmosphere': location.atmosphere
        }

    def get_available_locations(self) -> Dict[str, Location]:
        """Get all available locations in current world"""
        return self.worlds[self.state.current_world]

    def advance_turn(self) -> None:
        """Advance game turn counter"""
        self.state.turn_number += 1

    def get_session_summary(self) -> Dict[str, Any]:
        """Get complete session summary"""
        memory_summary = self.narrative_memory.get_summary()

        return {
            'player': {
                'name': self.player.name,
                'level': self.get_player_level(),
                'xp': self.player.stats.total_xp,
                'performance': self.player.calculate_overall_performance()
            },
            'progress': {
                'turns': self.state.turn_number,
                'current_location': self.state.current_location_id,
                'current_world': self.state.current_world.value,
                'quest_in_progress': self.state.quest_in_progress is not None
            },
            'narrative': memory_summary,
            'consistency': {
                'quest_checks': sum(
                    1 for quest_id in self.narrative_memory.events_by_quest
                    if self.consistency_checker.check_quest_consistency(quest_id)['consistent']
                ),
                'total_quests': len(self.narrative_memory.events_by_quest)
            }
        }


# Demo function
def demo_game_session():
    """Demonstrate integrated game session"""
    print("\n" + "="*70)
    print("EduVerse Integrated Game Session Demo")
    print("="*70 + "\n")

    # Create curriculum and player
    from EduVerse.education.curriculum import create_sample_curriculum
    from HoloLoom.memory.graph import KG

    curriculum = create_sample_curriculum()
    kg = KG()
    player = PlayerModel(
        student_id="demo_student_001",
        name="Alex",
        grade=6,
        knowledge_graph=kg
    )

    print(f"Created player: {player.name} (Grade {player.grade}, Level {player.stats.total_xp // 100})")
    print()

    # Create game session
    session = GameSession(
        player_model=player,
        curriculum=curriculum,
        start_world=WorldTheme.SCHOOL,
        start_location="school_lobby"
    )

    print("Game session initialized!")
    print(f"Starting world: {session.state.current_world.value}")
    print(f"Starting location: {session.state.current_location_id}")
    print()

    # Turn 1: Explore lobby
    print("-" * 70)
    print(f"TURN {session.state.turn_number + 1}: Explore School Lobby")
    print("-" * 70)

    exploration = session.explore_location()
    location = exploration['location']
    npcs = exploration['npcs']

    print(f"\nLocation: {location.name}")
    print(f"Description: {location.description}")
    print(f"Atmosphere: {location.atmosphere}")
    print(f"\nNPCs here: {len(npcs)}")
    for npc in npcs:
        print(f"  - {npc.name} ({npc.role.value})")

    session.advance_turn()
    print()

    # Turn 2: Move to math classroom
    print("-" * 70)
    print(f"TURN {session.state.turn_number + 1}: Go to Math Classroom")
    print("-" * 70)

    success, message = session.move_to_location("school_math_classroom")
    print(f"\n{message}")

    exploration = session.explore_location()
    location = exploration['location']
    npcs = exploration['npcs']

    print(f"\nLocation: {location.name}")
    print(f"Description: {location.description}")
    print(f"\nNPCs here: {len(npcs)}")
    for npc in npcs:
        print(f"  - {npc.name} ({npc.role.value}, {npc.teaching_style.value})")

    session.advance_turn()
    print()

    # Turn 3: Greet Ms. Rodriguez
    print("-" * 70)
    print(f"TURN {session.state.turn_number + 1}: Greet Ms. Rodriguez")
    print("-" * 70)

    npc_id = "npc_math_teacher_001"
    dialogue = session.greet_npc(npc_id)

    if dialogue:
        print(f"\n[{dialogue.npc_name}]: {dialogue.text}")
        print(f"(Tone: {dialogue.emotional_tone})")

    session.advance_turn()
    print()

    # Turn 4: Request quest
    print("-" * 70)
    print(f"TURN {session.state.turn_number + 1}: Request Quest")
    print("-" * 70)

    quest, dialogue = session.request_quest_from_npc(npc_id)

    if quest and dialogue:
        print(f"\n[{dialogue.npc_name}]: {dialogue.text}")
        print(f"\nQuest Offered: {quest.title}")
        print(f"Description: {quest.description}")
        print(f"Difficulty: {quest.difficulty:.1f}")
        print(f"Type: {quest.quest_type.value}")

        # Accept quest
        accepted = session.accept_quest(quest, npc_id)
        if accepted:
            print(f"\n✓ Quest accepted!")
    else:
        print("No quest available")

    session.advance_turn()
    print()

    # Turn 5: Complete quest (simulated)
    print("-" * 70)
    print(f"TURN {session.state.turn_number + 1}: Complete Quest")
    print("-" * 70)

    if session.state.quest_in_progress:
        print(f"\n[Simulating quest completion...]")
        score = 0.85  # Simulated score

        completed, dialogue = session.complete_current_quest(score, npc_id)

        if completed and dialogue:
            print(f"\n[{dialogue.npc_name}]: {dialogue.text}")
            print(f"(Tone: {dialogue.emotional_tone})")
            print(f"\nQuest completed with score: {score:.1%}")
            print(f"Player level: {session.get_player_level()} (XP: {player.stats.total_xp})")

    session.advance_turn()
    print()

    # Show narrative memory
    print("-" * 70)
    print("Narrative Memory")
    print("-" * 70)

    context = session.get_narrative_context_for_npc(npc_id)

    print(f"\nRecent Events:")
    for event in context['recent_events']:
        print(f"  - {event.description} [{event.importance.name}]")

    print(f"\nNPC History ({len(context['npc_history'])}):")
    for event in context['npc_history']:
        print(f"  - {event.description}")

    # Show session summary
    print("\n" + "-" * 70)
    print("Session Summary")
    print("-" * 70)

    summary = session.get_session_summary()

    print(f"\nPlayer: {summary['player']['name']} (Level {summary['player']['level']})")
    print(f"XP: {summary['player']['xp']}")
    print(f"Performance: {summary['player']['performance']:.1%}")

    print(f"\nProgress:")
    print(f"  Turns: {summary['progress']['turns']}")
    print(f"  Location: {summary['progress']['current_location']}")
    print(f"  World: {summary['progress']['current_world']}")

    print(f"\nNarrative:")
    print(f"  Total Events: {summary['narrative']['total_events']}")
    print(f"  Active Threads: {summary['narrative']['active_threads']}")
    print(f"  Total Choices: {summary['narrative']['total_choices']}")

    print("\n" + "="*70)
    print("All systems integrated! Ready for Week 3 demo.")
    print("="*70)


if __name__ == "__main__":
    demo_game_session()
