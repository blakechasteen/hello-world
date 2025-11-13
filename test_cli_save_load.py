"""Test script for CLI save/load functionality"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from EduVerse.eduverse_cli import EduVerseCLI
import time

def test_cli_save_load():
    """Test the save/load system integrated with CLI"""
    print("="*70)
    print("CLI Save/Load Integration Test")
    print("="*70)
    print()

    # Create CLI
    print("1. Creating CLI...")
    cli = EduVerseCLI()
    print()

    # Check save system is initialized
    print("2. Checking SaveSystem initialization...")
    assert hasattr(cli, 'save_system'), "SaveSystem not initialized"
    assert hasattr(cli, 'session_start_time'), "Session start time not tracked"
    print(f"[OK] SaveSystem initialized")
    print(f"[OK] Session start time: {cli.session_start_time}")
    print()

    # Test travel and earn XP
    print("3. Simulating gameplay (travel + earn XP)...")
    cli.player.stats.total_xp = 500.0
    cli.player.stats.quests_completed = 5
    cli.player.stats.quests_attempted = 7
    cli.world.travel_to("school_library")
    print(f"[OK] Traveled to: {cli.world.get_current_location().name}")
    print(f"[OK] Player XP: {cli.player.stats.total_xp}")
    print(f"[OK] Quests: {cli.player.stats.quests_completed}/{cli.player.stats.quests_attempted}")
    print()

    # Test save command
    print("4. Testing save command...")
    cli.cmd_save("test_game")
    print()

    # Test saves command
    print("5. Testing saves command...")
    cli.cmd_saves()
    print()

    # Modify player state
    print("6. Modifying player state...")
    cli.player.stats.total_xp = 999.0
    cli.world.travel_to("school_lobby")
    print(f"[Modified] Player XP: {cli.player.stats.total_xp}")
    print(f"[Modified] Location: {cli.world.get_current_location().name}")
    print()

    # Test load command
    print("7. Testing load command...")
    cli.cmd_load("test_game")
    print()

    # Verify restored state
    print("8. Verifying restored state...")
    assert cli.player.stats.total_xp == 500.0, f"XP mismatch: {cli.player.stats.total_xp}"
    assert cli.player.stats.quests_completed == 5, f"Quests mismatch: {cli.player.stats.quests_completed}"
    assert cli.world.game_state.current_location_id == "school_library", f"Location mismatch"
    print(f"[OK] XP restored: {cli.player.stats.total_xp}")
    print(f"[OK] Quests restored: {cli.player.stats.quests_completed}/{cli.player.stats.quests_attempted}")
    print(f"[OK] Location restored: {cli.world.get_current_location().name}")
    print()

    # Test error handling
    print("9. Testing error handling...")
    print("   a. Save with no name:")
    cli.cmd_save("")
    print()

    print("   b. Load non-existent save:")
    cli.cmd_load("nonexistent_save")
    print()

    # Clean up test save
    print("10. Cleaning up test save...")
    result = cli.save_system.delete_save("test_game")
    print(f"[OK] {result['message']}")
    print()

    print("="*70)
    print("All CLI Save/Load tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_cli_save_load()
