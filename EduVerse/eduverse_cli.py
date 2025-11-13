"""
EduVerse Interactive CLI

A playable command-line interface for exploring the EduVerse educational game.
Showcases all Week 3 systems: worlds, NPCs, quests, and narrative memory.

Created: 2025-11-13 (Week 3 Enhancement)

Commands:
    look              - See current location and NPCs
    travel <location> - Travel to a location
    locations         - List all available locations
    talk <npc>        - Talk to an NPC
    npcs              - List NPCs at current location
    quest             - Get a quest from current NPC
    accept            - Accept the offered quest
    complete          - Complete current quest (simulated)
    status            - Show player stats
    story             - Show narrative history
    save <name>       - Save game to file
    load <name>       - Load game from file
    saves             - List all saved games
    help              - Show this help
    quit              - Exit the game
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Optional, List, Dict, Any
from HoloLoom.memory.graph import KG
from EduVerse.education.player_model import PlayerModel
from EduVerse.education.curriculum import CurriculumFramework
from EduVerse.game.world_integration import WorldIntegration
from EduVerse.game.npc_system import DialogueContext
from EduVerse.game.minigame_framework import MinigameResult
from EduVerse.game.save_system import SaveSystem
from dataclasses import dataclass
import time


class EduVerseCLI:
    """Interactive command-line interface for EduVerse"""

    def __init__(self):
        print("\n" + "="*70)
        print("EduVerse - AI-Powered K-12 Learning Platform")
        print("Week 3 Interactive Demo")
        print("="*70 + "\n")

        print("Initializing world...")

        # Create player
        kg = KG()
        self.player = PlayerModel(
            student_id="demo_player",
            name="Student",
            grade=8,
            knowledge_graph=kg
        )

        # Create curriculum
        self.curriculum = CurriculumFramework()

        # Create integrated world
        self.world = WorldIntegration(
            player=self.player,
            curriculum=self.curriculum,
            starting_location="school_lobby"
        )

        print(f"[OK] World ready with {len(self.world.locations)} locations and {len(self.world.npc_registry.npcs)} NPCs")
        print()

        # Game state
        self.current_npc: Optional[str] = None  # NPC currently talking to
        self.offered_quest = None  # Quest currently offered

        # Command history
        self.command_history: List[str] = []

        # Save/Load system
        self.save_system = SaveSystem()
        self.session_start_time = time.time()

    def run(self):
        """Main game loop"""
        print("Type 'help' for commands or 'quit' to exit.")
        print()
        self.cmd_look()  # Show starting location
        print()

        while True:
            try:
                # Get command
                command = input("> ").strip().lower()

                if not command:
                    continue

                # Add to history
                self.command_history.append(command)

                # Parse and execute
                parts = command.split(maxsplit=1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else ""

                # Route to handler
                if cmd in ['quit', 'exit', 'q']:
                    self.cmd_quit()
                    break
                elif cmd in ['help', 'h', '?']:
                    self.cmd_help()
                elif cmd in ['look', 'l']:
                    self.cmd_look()
                elif cmd in ['travel', 't', 'go']:
                    self.cmd_travel(arg)
                elif cmd in ['locations', 'locs']:
                    self.cmd_locations()
                elif cmd in ['talk', 'speak']:
                    self.cmd_talk(arg)
                elif cmd in ['npcs', 'n']:
                    self.cmd_npcs()
                elif cmd in ['quest', 'q']:
                    self.cmd_quest()
                elif cmd in ['accept', 'a']:
                    self.cmd_accept()
                elif cmd in ['complete', 'c']:
                    self.cmd_complete()
                elif cmd in ['status', 's', 'stats']:
                    self.cmd_status()
                elif cmd in ['story', 'narrative', 'history']:
                    self.cmd_story()
                elif cmd == 'save':
                    self.cmd_save(arg)
                elif cmd == 'load':
                    self.cmd_load(arg)
                elif cmd == 'saves':
                    self.cmd_saves()
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands.")

                print()  # Blank line after command

            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit.")
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()

    def cmd_help(self):
        """Show help"""
        print("\nAvailable Commands:")
        print("-" * 70)
        print("  look, l              - See current location and NPCs")
        print("  travel <loc>, t      - Travel to a location (use location ID)")
        print("  locations, locs      - List all available locations")
        print("  talk <npc>           - Talk to an NPC (use NPC name)")
        print("  npcs, n              - List NPCs at current location")
        print("  quest                - Get a quest from current NPC")
        print("  accept, a            - Accept the offered quest")
        print("  complete, c          - Complete current quest (simulated)")
        print("  status, s            - Show player stats and progress")
        print("  story                - Show narrative history")
        print("  save <name>          - Save game to file")
        print("  load <name>          - Load game from file")
        print("  saves                - List all saved games")
        print("  help, h              - Show this help")
        print("  quit, q              - Exit the game")
        print("-" * 70)

    def cmd_look(self):
        """Look at current location"""
        location = self.world.get_current_location()
        if not location:
            print("Error: Unknown location")
            return

        print(f"\n{location.name}")
        print("-" * 70)
        print(f"{location.description}")
        print()

        # Show visual elements
        if location.visual_elements:
            print("You see:")
            count = 0
            for key, value in location.visual_elements.items():
                if count >= 3:  # Show first 3
                    break
                print(f"  - {value}")
                count += 1
            print()

        # Show NPCs
        npcs = self.world.get_npcs_at_current_location()
        if npcs:
            print(f"NPCs here ({len(npcs)}):")
            for npc in npcs:
                print(f"  - {npc.name} ({npc.role.value})")
        else:
            print("No NPCs here.")

        print()

        # Show exits
        if location.connected_to:
            print("You can travel to:")
            for i, exit_loc in enumerate(location.connected_to):
                if i >= 5:  # Show first 5
                    break
                if exit_loc in self.world.locations:
                    loc = self.world.locations[exit_loc]
                    print(f"  - {loc.name} (ID: {exit_loc})")
        else:
            print("Use 'locations' to see all available locations.")

    def cmd_travel(self, location_id: str):
        """Travel to a location"""
        if not location_id:
            print("Usage: travel <location_id>")
            print("Use 'locations' to see available locations.")
            return

        # Try to find location
        location_id = location_id.strip().replace(" ", "_").lower()

        if location_id not in self.world.locations:
            # Try fuzzy match by name
            for loc_id, loc in self.world.locations.items():
                if location_id in loc.name.lower():
                    location_id = loc_id
                    break
            else:
                print(f"Location not found: {location_id}")
                print("Use 'locations' to see available locations.")
                return

        # Travel
        result = self.world.travel_to(location_id)
        if result['success']:
            print(f"\n[Traveled to {result['location'].name}]")
            print()
            self.cmd_look()
            self.current_npc = None  # Clear current NPC when traveling
        else:
            print(f"Cannot travel: {result['message']}")

    def cmd_locations(self):
        """List all locations"""
        print("\nAvailable Locations:")
        print("-" * 70)

        # Group by world
        worlds = {
            'school': [],
            'fantasy': [],
            'space': []
        }

        for loc_id, loc in self.world.locations.items():
            if 'school' in loc_id:
                worlds['school'].append((loc_id, loc))
            elif 'fantasy' in loc_id:
                worlds['fantasy'].append((loc_id, loc))
            elif 'space' in loc_id:
                worlds['space'].append((loc_id, loc))

        for world_name, locs in worlds.items():
            if locs:
                print(f"\n{world_name.upper()} WORLD:")
                for loc_id, loc in sorted(locs, key=lambda x: x[1].name):
                    print(f"  {loc.name:40s} (ID: {loc_id})")

    def cmd_npcs(self):
        """List NPCs at current location"""
        npcs = self.world.get_npcs_at_current_location()

        if not npcs:
            print("No NPCs at this location.")
            return

        print(f"\nNPCs at {self.world.get_current_location().name}:")
        print("-" * 70)

        for npc in npcs:
            print(f"\n{npc.name} ({npc.role.value})")
            print(f"  Teaching Style: {npc.teaching_style.value}")
            print(f"  Subjects: {', '.join(s.value for s in npc.subjects)}")
            print(f"  {npc.backstory[:80]}...")
            if npc.can_give_quests:
                print(f"  [Can give quests]")

    def cmd_talk(self, npc_name: str):
        """Talk to an NPC"""
        if not npc_name:
            print("Usage: talk <npc_name>")
            print("Use 'npcs' to see NPCs at current location.")
            return

        # Find NPC at current location
        npcs = self.world.get_npcs_at_current_location()
        npc = None

        for n in npcs:
            if npc_name.lower() in n.name.lower():
                npc = n
                break

        if not npc:
            print(f"NPC not found: {npc_name}")
            print("Use 'npcs' to see NPCs at current location.")
            return

        # Set as current NPC
        self.current_npc = npc.npc_id

        # Generate dialogue
        dialogue = self.world.talk_to_npc(npc.npc_id, DialogueContext.GREETING)

        if dialogue:
            print(f"\n[{dialogue.npc_name}]:")
            print(f'"{dialogue.text}"')
            print(f"(Tone: {dialogue.emotional_tone})")

            if npc.can_give_quests:
                print()
                print("Hint: Type 'quest' to get a quest from this NPC.")
        else:
            print(f"Error talking to {npc.name}")

    def cmd_quest(self):
        """Get a quest from current NPC"""
        if not self.current_npc:
            print("No NPC selected. Use 'talk <npc>' first.")
            return

        npc = self.world.npc_registry.get_npc(self.current_npc)
        if not npc or not npc.can_give_quests:
            print(f"{npc.name if npc else 'This NPC'} cannot give quests.")
            return

        # Get quest
        quest = self.world.get_quest_from_npc(self.current_npc)

        if not quest:
            print(f"\n[{npc.name}]:")
            print('"I don\'t have any suitable quests for you right now."')
            print("(Try completing more objectives or talking to other NPCs)")
            return

        # Store offered quest
        self.offered_quest = quest

        # Show quest details
        print(f"\n[{npc.name}]:")
        print(f'"I have a quest for you!"')
        print()
        print(f"Quest: {quest.title}")
        print("-" * 70)
        print(f"{quest.description}")
        print()
        print(f"Type: {quest.quest_type.value}")
        print(f"Difficulty: {quest.difficulty:.1f}/1.0")
        print(f"Estimated Time: {quest.estimated_minutes} minutes")

        if quest.learning_objectives:
            obj_id = quest.learning_objectives[0]
            obj = self.curriculum.get_objective(obj_id)
            if obj:
                print(f"Learning Objective: {obj.title}")

        # Show rewards
        sample_rewards = quest.calculate_rewards(0.8)  # Estimate at 80% score
        print()
        print(f"Rewards (estimated):")
        print(f"  XP: {sample_rewards['xp']}")
        for skill, xp in sample_rewards['skill_xp'].items():
            print(f"  {skill}: +{xp} XP")

        print()
        print("Type 'accept' to accept this quest.")

    def cmd_accept(self):
        """Accept the offered quest"""
        if not self.offered_quest:
            print("No quest offered. Use 'quest' to get a quest first.")
            return

        if not self.current_npc:
            print("No NPC selected.")
            return

        # Start quest
        result = self.world.start_quest(self.offered_quest, self.current_npc)

        if result['success']:
            npc = self.world.npc_registry.get_npc(self.current_npc)
            print(f"\n[{npc.name if npc else 'NPC'}]:")
            print(f'"Quest accepted! Good luck!"')
            print()
            print(f"[Quest Started: {self.offered_quest.title}]")
            print()
            print("Type 'complete' to complete the quest (simulated).")

            self.offered_quest = None  # Clear offered quest
        else:
            print(f"Cannot accept quest: {result['message']}")

    def cmd_complete(self):
        """Complete current quest"""
        if not self.world.game_state.current_quest:
            print("No active quest. Use 'quest' and 'accept' to start one.")
            return

        if not self.current_npc:
            print("No NPC selected.")
            return

        quest = self.world.game_state.current_quest

        # Simulate minigame with random score
        import random
        score = random.uniform(0.6, 1.0)  # Random score between 60% and 100%

        @dataclass
        class SimulatedResult:
            success: bool = True
            score: float = score
            time_taken: float = 120.0

        result_obj = SimulatedResult(score=score)

        # Complete quest
        result = self.world.complete_quest(result_obj, self.current_npc)

        if result['success']:
            npc = self.world.npc_registry.get_npc(self.current_npc)

            print(f"\n[Minigame Simulation]")
            print(f"Your Score: {score:.1%}")
            print()

            # NPC dialogue based on performance
            if score >= 0.85:
                context = DialogueContext.SUCCESS
            elif score < 0.5:
                context = DialogueContext.STRUGGLE
            else:
                context = DialogueContext.QUEST_COMPLETE

            dialogue = self.world.talk_to_npc(self.current_npc, context)
            if dialogue:
                print(f"[{dialogue.npc_name}]:")
                print(f'"{dialogue.text}"')
                print()

            # Show rewards
            print(f"[Quest Completed: {quest.title}]")
            print()
            print("Rewards earned:")
            for key, value in result['rewards'].items():
                if key == 'xp':
                    print(f"  Total XP: +{value}")
                elif key == 'skill_xp':
                    for skill, xp in value.items():
                        print(f"  {skill}: +{xp} XP")

            # Show milestone if achieved
            if score >= 0.75:
                print()
                print("[Learning Milestone Achieved!]")
        else:
            print(f"Cannot complete quest: {result['message']}")

    def cmd_status(self):
        """Show player status"""
        print(f"\nPlayer: {self.player.name}")
        print("-" * 70)
        print(f"Level: {self.player.stats.level}")
        print(f"Total XP: {self.player.stats.total_xp}")
        print(f"Grade: {self.player.grade}")
        print()

        # Show skills
        if self.player.skills:
            print("Skills:")
            for skill_id, skill in sorted(self.player.skills.items()):
                print(f"  {skill.name}: Level {skill.level.name}, {skill.xp:.1f} XP")
        else:
            print("Skills: None yet")

        print()

        # Show learning style
        print(f"Learning Style Preferences:")
        for style, pref in self.player.learning_styles.items():
            print(f"  {style.value}: {pref:.2f}")

        print()

        # Show current quest
        if self.world.game_state.current_quest:
            print(f"Active Quest: {self.world.game_state.current_quest.title}")
        else:
            print("Active Quest: None")

        print()

        # Show stats
        print(f"Quests Completed: {self.player.stats.quests_completed}/{self.player.stats.quests_attempted}")
        print(f"Locations Visited: {len(self.world.game_state.visited_locations)}")

    def cmd_story(self):
        """Show narrative history"""
        summary = self.world.get_narrative_summary()

        print("\nNarrative Summary")
        print("-" * 70)
        print(f"Total Events: {summary['total_events']}")
        print(f"Active Threads: {summary['active_threads']}/{summary['total_threads']}")
        print()

        # Show threads
        if summary['threads']:
            print("Subject Progress:")
            for thread_info in summary['threads']:
                if thread_info['events'] > 0:
                    print(f"  {thread_info['subject']}: {thread_info['events']} events")

        print()

        # Show recent events
        if summary['recent_significant_events']:
            print(f"Recent Significant Events ({len(summary['recent_significant_events'])}):")
            for event in summary['recent_significant_events'][:10]:
                print(f"  [{event['importance']}] {event['description']}")
        else:
            print("No significant events yet. Start exploring and completing quests!")

        print()

        # Show consistency
        consistency = self.world.check_world_consistency()
        if consistency['consistent']:
            print("[Story is consistent - no narrative contradictions]")
        else:
            print("[Warning: Story has inconsistencies]")
            for issue in consistency['issues']:
                print(f"  - {issue}")

    def cmd_save(self, save_name: str):
        """Save game to file"""
        if not save_name:
            print("Usage: save <name>")
            print("Example: save my_game")
            return

        # Calculate playtime
        playtime_minutes = int((time.time() - self.session_start_time) / 60)

        result = self.save_system.save_game(self.world, save_name, playtime_minutes)

        if result['success']:
            print(f"\n[OK] {result['message']}")
        else:
            print(f"\n[FAIL] {result['message']}")

    def cmd_load(self, save_name: str):
        """Load game from file"""
        if not save_name:
            print("Usage: load <name>")
            print("Example: load my_game")
            return

        result = self.save_system.load_game(self.world, save_name)

        if result['success']:
            print(f"\n[OK] {result['message']}")
            print(f"Restored: {self.world.player.name} at {self.world.game_state.current_location_id}")

            # Reset session start time
            self.session_start_time = time.time()

            # Show current location
            self.cmd_look()
        else:
            print(f"\n[FAIL] {result['message']}")

    def cmd_saves(self):
        """List all saved games"""
        saves = self.save_system.list_saves()

        print("\nSaved Games")
        print("-" * 70)

        if not saves:
            print("No saved games found.")
        else:
            for save in saves:
                print(f"\n{save['name']}")
                print(f"  Character: {save['character']} (Level {save['level']})")
                print(f"  Location: {save['location']}")
                print(f"  Quests: {save['quests']} completed")
                print(f"  Playtime: {save['playtime']} minutes")
                print(f"  Saved: {save['date'][:16]}")  # Truncate timestamp

        print()

    def cmd_quit(self):
        """Quit the game"""
        print("\nThanks for playing EduVerse!")
        print()

        # Show final stats
        summary = self.world.get_narrative_summary()
        print("Session Summary:")
        print(f"  Total XP: {self.player.stats.total_xp}")
        print(f"  Events: {summary['total_events']}")
        print(f"  Locations Visited: {len(self.world.game_state.visited_locations)}")
        print()


def main():
    """Main entry point"""
    cli = EduVerseCLI()
    cli.run()


if __name__ == "__main__":
    main()
