"""
Save/Load System - Player Progress Persistence

**Purpose**: Persist player progress, game state, and narrative memory to disk
**Integration**: Works with PlayerModel, WorldIntegration, and NarrativeMemory
**Date**: November 13, 2025

This module implements comprehensive save/load functionality:
- Player state (skills, XP, stats, learning styles)
- Game state (location, quest, visited locations)
- Narrative memory (events, threads, choices)
- Session metadata (timestamps, playtime)

File Format: JSON for human-readable saves
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from EduVerse.education.player_model import PlayerModel, Skill, SkillLevel, LearningStyle
from EduVerse.education.curriculum import Subject
from EduVerse.game.quest_engine import Quest, QuestType, DifficultyLevel
from EduVerse.game.narrative_memory import (
    NarrativeMemory, NarrativeEvent, EventType, EventImportance,
    NarrativeThread, PlayerChoiceRecord
)
from EduVerse.game.world_integration import WorldIntegration, GameState


@dataclass
class SaveMetadata:
    """Metadata about the save file"""
    save_version: str = "1.0.0"
    save_date: str = ""
    total_playtime_minutes: int = 0
    character_name: str = ""
    character_level: int = 1
    current_location: str = ""
    quests_completed: int = 0

    def __post_init__(self):
        if not self.save_date:
            self.save_date = datetime.utcnow().isoformat()


class SaveSystem:
    """
    Handles saving and loading game state to/from JSON files

    Save file structure:
    {
        "metadata": {...},
        "player": {...},
        "game_state": {...},
        "narrative_memory": {...}
    }
    """

    def __init__(self, save_dir: str = "./saves"):
        """
        Initialize save system

        Args:
            save_dir: Directory to store save files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_game(
        self,
        world: WorldIntegration,
        save_name: str = "autosave",
        total_playtime: int = 0
    ) -> Dict[str, Any]:
        """
        Save complete game state to file

        Args:
            world: WorldIntegration instance with all game state
            save_name: Name of save file (without extension)
            total_playtime: Total playtime in minutes

        Returns:
            Result dict with success status and file path
        """
        try:
            # Create save data structure
            save_data = {
                "metadata": self._create_metadata(world, total_playtime),
                "player": self._serialize_player(world.player),
                "game_state": self._serialize_game_state(world.game_state),
                "narrative_memory": self._serialize_narrative_memory(world.narrative_memory),
                "subject_threads": self._serialize_subject_threads(world.subject_threads)
            }

            # Write to file
            save_path = self.save_dir / f"{save_name}.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            return {
                'success': True,
                'path': str(save_path),
                'message': f"Game saved to {save_name}.json"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to save game: {e}"
            }

    def load_game(
        self,
        world: WorldIntegration,
        save_name: str = "autosave"
    ) -> Dict[str, Any]:
        """
        Load game state from file

        Args:
            world: WorldIntegration instance to load into
            save_name: Name of save file (without extension)

        Returns:
            Result dict with success status and metadata
        """
        try:
            save_path = self.save_dir / f"{save_name}.json"

            if not save_path.exists():
                return {
                    'success': False,
                    'message': f"Save file not found: {save_name}.json"
                }

            # Read save file
            with open(save_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)

            # Restore player state
            self._deserialize_player(save_data['player'], world.player)

            # Restore game state
            self._deserialize_game_state(save_data['game_state'], world)

            # Restore narrative memory
            self._deserialize_narrative_memory(save_data['narrative_memory'], world.narrative_memory)

            # Restore subject threads
            if 'subject_threads' in save_data:
                self._deserialize_subject_threads(save_data['subject_threads'], world)

            return {
                'success': True,
                'metadata': save_data['metadata'],
                'message': f"Game loaded from {save_name}.json"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to load game: {e}"
            }

    def list_saves(self) -> List[Dict[str, Any]]:
        """
        List all available save files

        Returns:
            List of save file info dicts
        """
        saves = []

        for save_file in self.save_dir.glob("*.json"):
            try:
                with open(save_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})

                    saves.append({
                        'name': save_file.stem,
                        'date': metadata.get('save_date', 'Unknown'),
                        'character': metadata.get('character_name', 'Unknown'),
                        'level': metadata.get('character_level', 1),
                        'location': metadata.get('current_location', 'Unknown'),
                        'playtime': metadata.get('total_playtime_minutes', 0),
                        'quests': metadata.get('quests_completed', 0)
                    })
            except Exception:
                continue  # Skip corrupted saves

        # Sort by date (newest first)
        saves.sort(key=lambda x: x['date'], reverse=True)
        return saves

    def delete_save(self, save_name: str) -> Dict[str, Any]:
        """Delete a save file"""
        try:
            save_path = self.save_dir / f"{save_name}.json"

            if not save_path.exists():
                return {
                    'success': False,
                    'message': f"Save file not found: {save_name}.json"
                }

            save_path.unlink()

            return {
                'success': True,
                'message': f"Deleted save: {save_name}.json"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to delete save: {e}"
            }

    # ========== Serialization Methods ==========

    def _create_metadata(self, world: WorldIntegration, playtime: int) -> Dict[str, Any]:
        """Create save metadata"""
        return {
            'save_version': '1.0.0',
            'save_date': datetime.utcnow().isoformat(),
            'total_playtime_minutes': playtime,
            'character_name': world.player.name,
            'character_level': world.player.stats.level,
            'current_location': world.game_state.current_location_id,
            'quests_completed': world.player.stats.quests_completed
        }

    def _serialize_player(self, player: PlayerModel) -> Dict[str, Any]:
        """Serialize player state"""
        return {
            'student_id': player.student_id,
            'name': player.name,
            'grade': player.grade,
            'skills': {
                skill_id: {
                    'id': skill.id,
                    'name': skill.name,
                    'subject': skill.subject,
                    'description': skill.description,
                    'level': skill.level.value,
                    'xp': skill.xp,
                    'xp_to_next_level': skill.xp_to_next_level,
                    'prerequisites': skill.prerequisites,
                    'practice_count': skill.practice_count,
                    'success_rate': skill.success_rate
                }
                for skill_id, skill in player.skills.items()
            },
            'learning_styles': {
                style.value: pref
                for style, pref in player.learning_styles.items()
            },
            'stats': {
                'total_xp': player.stats.total_xp,
                'level': player.stats.level,
                'quests_completed': player.stats.quests_completed,
                'quests_attempted': player.stats.quests_attempted,
                'minigames_completed': player.stats.minigames_completed,
                'total_play_time_minutes': player.stats.total_play_time_minutes,
                'streak_days': player.stats.streak_days,
                'achievements': player.stats.achievements
            },
            'difficulty_preference': player.difficulty_preference
        }

    def _serialize_game_state(self, game_state: GameState) -> Dict[str, Any]:
        """Serialize game state"""
        return {
            'current_location_id': game_state.current_location_id,
            'visited_locations': game_state.visited_locations,
            'current_quest': self._serialize_quest(game_state.current_quest) if game_state.current_quest else None,
            'active_npc': game_state.active_npc,
            'session_start_timestamp': game_state.session_start_timestamp
        }

    def _serialize_quest(self, quest: Quest) -> Dict[str, Any]:
        """Serialize a quest"""
        return {
            'id': quest.id,
            'title': quest.title,
            'description': quest.description,
            'quest_type': quest.quest_type.value,
            'difficulty': quest.difficulty,
            'estimated_duration_minutes': quest.estimated_duration_minutes,
            'learning_objectives': quest.learning_objectives,
            'rewards_xp': quest.rewards_xp,
            'rewards_items': quest.rewards_items
        }

    def _serialize_narrative_memory(self, memory: NarrativeMemory) -> Dict[str, Any]:
        """Serialize narrative memory"""
        return {
            'events': {
                event_id: {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp,
                    'importance': event.importance.value,
                    'description': event.description,
                    'location_id': event.location_id,
                    'npc_id': event.npc_id,
                    'quest_id': event.quest_id,
                    'subject': event.subject.value if event.subject else None,
                    'caused_by': event.caused_by,
                    'enables': event.enables
                }
                for event_id, event in memory.events.items()
            },
            'threads': {
                thread_id: {
                    'thread_id': thread.thread_id,
                    'name': thread.name,
                    'description': thread.description,
                    'subject': thread.subject.value if thread.subject else None,
                    'events': thread.events,
                    'start_timestamp': thread.start_timestamp,
                    'active': thread.active
                }
                for thread_id, thread in memory.threads.items()
            },
            'events_chronological': memory.events_chronological
        }

    def _serialize_subject_threads(self, subject_threads: Dict[Subject, NarrativeThread]) -> Dict[str, Any]:
        """Serialize subject threads"""
        return {
            subject.value: {
                'thread_id': thread.thread_id,
                'name': thread.name,
                'description': thread.description,
                'subject': thread.subject.value if thread.subject else None,
                'events': thread.events,
                'start_timestamp': thread.start_timestamp,
                'active': thread.active
            }
            for subject, thread in subject_threads.items()
        }

    # ========== Deserialization Methods ==========

    def _deserialize_player(self, data: Dict[str, Any], player: PlayerModel):
        """Deserialize player state"""
        player.student_id = data['student_id']
        player.name = data['name']
        player.grade = data['grade']

        # Restore skills
        player.skills.clear()
        for skill_id, skill_data in data['skills'].items():
            skill = Skill(
                id=skill_data['id'],
                name=skill_data['name'],
                subject=skill_data['subject'],
                description=skill_data['description'],
                level=SkillLevel(skill_data['level']),
                xp=skill_data['xp'],
                xp_to_next_level=skill_data['xp_to_next_level'],
                prerequisites=skill_data['prerequisites'],
                practice_count=skill_data['practice_count'],
                success_rate=skill_data['success_rate']
            )
            player.skills[skill_id] = skill

        # Restore learning styles
        player.learning_styles = {
            LearningStyle(style_name): pref
            for style_name, pref in data['learning_styles'].items()
        }

        # Restore stats
        stats_data = data['stats']
        player.stats.total_xp = stats_data['total_xp']
        player.stats.level = stats_data['level']
        player.stats.quests_completed = stats_data['quests_completed']
        player.stats.quests_attempted = stats_data['quests_attempted']
        player.stats.minigames_completed = stats_data['minigames_completed']
        player.stats.total_play_time_minutes = stats_data['total_play_time_minutes']
        player.stats.streak_days = stats_data['streak_days']
        player.stats.achievements = stats_data['achievements']

        # Restore difficulty preference
        player.difficulty_preference = data['difficulty_preference']

    def _deserialize_game_state(self, data: Dict[str, Any], world: WorldIntegration):
        """Deserialize game state"""
        world.game_state.current_location_id = data['current_location_id']
        world.game_state.visited_locations = data['visited_locations']
        world.game_state.active_npc = data.get('active_npc')
        world.game_state.session_start_timestamp = data['session_start_timestamp']

        # Restore current quest
        if data['current_quest']:
            quest_data = data['current_quest']
            world.game_state.current_quest = Quest(
                id=quest_data['id'],
                title=quest_data['title'],
                description=quest_data['description'],
                quest_type=QuestType(quest_data['quest_type']),
                difficulty=quest_data['difficulty'],
                estimated_duration_minutes=quest_data['estimated_duration_minutes'],
                learning_objectives=quest_data['learning_objectives'],
                rewards_xp=quest_data['rewards_xp'],
                rewards_items=quest_data['rewards_items']
            )
        else:
            world.game_state.current_quest = None

    def _deserialize_narrative_memory(self, data: Dict[str, Any], memory: NarrativeMemory):
        """Deserialize narrative memory"""
        # Clear existing memory
        memory.events.clear()
        memory.threads.clear()
        memory.events_by_quest.clear()
        memory.events_by_npc.clear()
        memory.events_by_location.clear()
        memory.events_by_subject.clear()
        memory.events_by_type.clear()
        memory.events_chronological.clear()

        # Restore events
        for event_id, event_data in data['events'].items():
            event = NarrativeEvent(
                event_id=event_data['event_id'],
                event_type=EventType(event_data['event_type']),
                timestamp=event_data['timestamp'],
                importance=EventImportance(event_data['importance']),
                description=event_data['description'],
                location_id=event_data.get('location_id'),
                npc_id=event_data.get('npc_id'),
                quest_id=event_data.get('quest_id'),
                subject=Subject(event_data['subject']) if event_data.get('subject') else None,
                caused_by=event_data['caused_by'],
                enables=event_data['enables']
            )
            memory.add_event(event)

        # Restore threads
        for thread_id, thread_data in data['threads'].items():
            thread = NarrativeThread(
                thread_id=thread_data['thread_id'],
                name=thread_data['name'],
                description=thread_data['description'],
                subject=Subject(thread_data['subject']) if thread_data.get('subject') else None,
                events=thread_data.get('events', []),
                start_timestamp=thread_data['start_timestamp'],
                active=thread_data.get('active', True)
            )
            memory.threads[thread_id] = thread

    def _deserialize_subject_threads(self, data: Dict[str, Any], world: WorldIntegration):
        """Deserialize subject threads"""
        world.subject_threads.clear()

        for subject_name, thread_data in data.items():
            subject = Subject(subject_name)
            thread = NarrativeThread(
                thread_id=thread_data['thread_id'],
                name=thread_data['name'],
                description=thread_data['description'],
                subject=Subject(thread_data['subject']) if thread_data.get('subject') else None,
                events=thread_data.get('events', []),
                start_timestamp=thread_data['start_timestamp'],
                active=thread_data.get('active', True)
            )
            world.subject_threads[subject] = thread


def demo_save_load():
    """Demonstrate save/load functionality"""
    print("="*70)
    print("Save/Load System Demo")
    print("="*70)
    print()

    from HoloLoom.memory.graph import KG
    from EduVerse.education.curriculum import CurriculumFramework

    # Create initial world
    print("1. Creating new game...")
    kg = KG()
    player = PlayerModel(
        student_id="demo_player",
        name="Alice",
        grade=7,
        knowledge_graph=kg
    )

    curriculum = CurriculumFramework()
    world = WorldIntegration(player, curriculum, "school_lobby")

    print(f"[OK] Created world with {len(world.locations)} locations")
    print(f"     Player: {player.name}, Grade {player.grade}")
    print(f"     Location: {world.game_state.current_location_id}")
    print()

    # Simulate some gameplay
    print("2. Simulating gameplay...")

    # Travel somewhere
    world.travel_to("school_library")
    print(f"   Traveled to: {world.get_current_location().name}")

    # Add some XP
    player.stats.total_xp = 250.0
    player.stats.quests_completed = 3
    player.stats.quests_attempted = 5
    print(f"   Earned XP: {player.stats.total_xp}")
    print(f"   Quests: {player.stats.quests_completed}/{player.stats.quests_attempted}")
    print()

    # Save game
    print("3. Saving game...")
    save_system = SaveSystem()
    result = save_system.save_game(world, "demo_save", total_playtime=45)

    if result['success']:
        print(f"[OK] {result['message']}")
        print(f"     Path: {result['path']}")
    else:
        print(f"[FAIL] {result['message']}")
    print()

    # List saves
    print("4. Listing saves...")
    saves = save_system.list_saves()
    print(f"Found {len(saves)} save file(s):")
    for save in saves:
        print(f"   - {save['name']}: {save['character']} (Level {save['level']})")
        print(f"     Location: {save['location']}, Quests: {save['quests']}")
    print()

    # Create new world (fresh start)
    print("5. Creating fresh world (simulating game restart)...")
    kg2 = KG()
    player2 = PlayerModel(
        student_id="demo_player_2",
        name="Bob",
        grade=6,
        knowledge_graph=kg2
    )
    world2 = WorldIntegration(player2, curriculum, "school_lobby")

    print(f"   Fresh world player: {player2.name}, XP: {player2.stats.total_xp}")
    print(f"   Fresh world location: {world2.game_state.current_location_id}")
    print()

    # Load game
    print("6. Loading saved game...")
    result = save_system.load_game(world2, "demo_save")

    if result['success']:
        print(f"[OK] {result['message']}")
        print(f"     Restored player: {world2.player.name}")
        print(f"     XP: {world2.player.stats.total_xp}")
        print(f"     Location: {world2.game_state.current_location_id}")
        print(f"     Quests: {world2.player.stats.quests_completed}/{world2.player.stats.quests_attempted}")
    else:
        print(f"[FAIL] {result['message']}")
    print()

    print("="*70)
    print("Save/Load Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_save_load()
