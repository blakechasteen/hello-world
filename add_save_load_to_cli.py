"""Script to add save/load functionality to eduverse_cli.py"""

# Read the current CLI file
with open('EduVerse/eduverse_cli.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where to insert SaveSystem initialization (after command_history line)
for i, line in enumerate(lines):
    if 'self.command_history: List[str] = []' in line:
        # Insert SaveSystem and session start time after this line
        lines.insert(i + 1, '\n')
        lines.insert(i + 2, '        # Save/Load system\n')
        lines.insert(i + 3, '        self.save_system = SaveSystem()\n')
        lines.insert(i + 4, '        self.session_start_time = time.time()\n')
        break

# Find where to add the save/load command handlers (before cmd_quit method)
for i, line in enumerate(lines):
    if 'def cmd_quit(self):' in line:
        # Insert save/load commands before quit
        save_load_commands = '''    def cmd_save(self, save_name: str):
        """Save game to file"""
        if not save_name:
            print("Usage: save <name>")
            print("Example: save my_game")
            return

        # Calculate playtime
        playtime_minutes = int((time.time() - self.session_start_time) / 60)

        result = self.save_system.save_game(self.world, save_name, playtime_minutes)

        if result['success']:
            print(f"\\n[OK] {result['message']}")
        else:
            print(f"\\n[FAIL] {result['message']}")

    def cmd_load(self, save_name: str):
        """Load game from file"""
        if not save_name:
            print("Usage: load <name>")
            print("Example: load my_game")
            return

        result = self.save_system.load_game(self.world, save_name)

        if result['success']:
            print(f"\\n[OK] {result['message']}")
            print(f"Restored: {self.world.player.name} at {self.world.game_state.current_location_id}")

            # Reset session start time
            self.session_start_time = time.time()

            # Show current location
            self.cmd_look()
        else:
            print(f"\\n[FAIL] {result['message']}")

    def cmd_saves(self):
        """List all saved games"""
        saves = self.save_system.list_saves()

        print("\\nSaved Games")
        print("-" * 70)

        if not saves:
            print("No saved games found.")
        else:
            for save in saves:
                print(f"\\n{save['name']}")
                print(f"  Character: {save['character']} (Level {save['level']})")
                print(f"  Location: {save['location']}")
                print(f"  Quests: {save['quests']} completed")
                print(f"  Playtime: {save['playtime']} minutes")
                print(f"  Saved: {save['date'][:16]}")  # Truncate timestamp

        print()

'''
        lines.insert(i, save_load_commands)
        break

# Find the command router section and add save/load handlers
for i, line in enumerate(lines):
    if "elif cmd == 'story':" in line:
        # Insert save/load routing after story command
        routing_code = '''                elif cmd == 'save':
                    self.cmd_save(arg)
                elif cmd == 'load':
                    self.cmd_load(arg)
                elif cmd == 'saves':
                    self.cmd_saves()
'''
        lines.insert(i + 2, routing_code)  # Insert after "self.cmd_story()"
        break

# Write the updated file
with open('EduVerse/eduverse_cli.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("[OK] Save/load functionality added to eduverse_cli.py")
