"""
Test script for EduVerse CLI
Tests initialization and core commands without interactive input
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from EduVerse.eduverse_cli import EduVerseCLI

def test_cli():
    """Test CLI initialization and commands"""
    print("="*70)
    print("EduVerse CLI Test")
    print("="*70)
    print()

    # Test initialization
    print("1. Testing CLI initialization...")
    try:
        cli = EduVerseCLI()
        print("[OK] CLI initialized successfully")
        print(f"    - Locations: {len(cli.world.locations)}")
        print(f"    - NPCs: {len(cli.world.npc_registry.npcs)}")
        print(f"    - Current location: {cli.world.game_state.current_location_id}")
        print()
    except Exception as e:
        print(f"[FAIL] CLI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cmd_look
    print("2. Testing 'look' command...")
    try:
        cli.cmd_look()
        print("[OK] Look command successful")
        print()
    except Exception as e:
        print(f"[FAIL] Look command failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cmd_status
    print("3. Testing 'status' command...")
    try:
        cli.cmd_status()
        print("[OK] Status command successful")
        print()
    except Exception as e:
        print(f"[FAIL] Status command failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cmd_locations
    print("4. Testing 'locations' command...")
    try:
        cli.cmd_locations()
        print("[OK] Locations command successful")
        print()
    except Exception as e:
        print(f"[FAIL] Locations command failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cmd_npcs
    print("5. Testing 'npcs' command...")
    try:
        cli.cmd_npcs()
        print("[OK] NPCs command successful")
        print()
    except Exception as e:
        print(f"[FAIL] NPCs command failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test cmd_story
    print("6. Testing 'story' command...")
    try:
        cli.cmd_story()
        print("[OK] Story command successful")
        print()
    except Exception as e:
        print(f"[FAIL] Story command failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("="*70)
    print("All tests passed!")
    print("="*70)

if __name__ == "__main__":
    test_cli()
