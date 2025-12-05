"""
Elle Tavern Demo - Integration Layer

Connects all components:
- World data (locations, NPCs, items)
- Quest system (quest progression)
- Game engine (state management)
- Elle Game Engine API (LLM-driven NPCs)

This module provides a complete, production-ready integration for the demo game.

Created: 2025-11-16
"""

import asyncio
import sys
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import httpx

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from games.elle_tavern_demo.world_data import (
    LOCATIONS, NPCS,
    get_location, get_npc, get_npcs_at_location
)
from games.elle_tavern_demo.quests import QUESTS, QuestManager, GameState as QuestGameState
from games.elle_tavern_demo.game_engine import TavernGameEngine


class IntegratedGame:
    """
    Fully integrated Elle Tavern Demo game.

    Combines:
    - World data (10 locations, 7 NPCs, 8 items)
    - Quest system (8 interconnected quests)
    - Game engine (state management + Elle integration)
    - Elle service (LLM-driven narrative intelligence)

    Usage:
        async with IntegratedGame() as game:
            if await game.initialize():
                await game.start_interactive_session()
    """

    def __init__(self, elle_service_url: str = "http://localhost:8000"):
        """
        Initialize integrated game.

        Args:
            elle_service_url: URL of Elle Game Engine service
        """
        self.elle_url = elle_service_url
        self.engine = TavernGameEngine(elle_api_url=elle_service_url)
        self.quest_manager: Optional[QuestManager] = None
        self.world_state: Optional[Dict[str, Any]] = None
        self.initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.engine.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.engine.__aexit__(exc_type, exc_val, exc_tb)

    # ========================================================================
    # Initialization
    # ========================================================================

    async def check_elle_service(self) -> bool:
        """
        Check if Elle service is running and accessible.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.elle_url}/health",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Elle service healthy: {data.get('status', 'ok')}")
                    return True
                else:
                    print(f"❌ Elle service returned status {response.status_code}")
                    return False
        except httpx.ConnectError:
            print(f"❌ Elle service not running at {self.elle_url}")
            print("   Start with: cd apps/elle_game_engine && uvicorn service:app --reload")
            return False
        except Exception as e:
            print(f"❌ Elle service check failed: {e}")
            return False

    async def initialize(self, check_elle: bool = True) -> bool:
        """
        Initialize game with all components.

        Steps:
        1. Check Elle service is running (optional)
        2. Initialize game engine
        3. Load world data (locations, NPCs, items)
        4. Initialize quest system
        5. Set up initial game state

        Args:
            check_elle: Whether to check Elle service before initializing

        Returns:
            True if initialization successful
        """
        if self.initialized:
            print("⚠️ Game already initialized")
            return True

        # 1. Check Elle service
        if check_elle:
            if not await self.check_elle_service():
                print("\n⚠️ Elle service not available. Game will use fallback responses.")
                # Continue anyway - game has fallback responses

        # 2. Initialize game engine
        print("\n🎮 Initializing game engine...")
        result = await self.engine.initialize()
        if result["status"] != "success":
            print(f"❌ Failed to initialize game engine: {result.get('message')}")
            return False

        # 3. Load world data
        print("🗺️ Loading world data...")
        self.world_state = initialize_world_state()

        # Update engine with world data
        if self.engine.game_state:
            # Merge world NPCs with engine NPCs
            for npc_id, npc_data in self.world_state["npcs"].items():
                if npc_id in self.engine.game_state.npcs:
                    # Update existing NPC with world data
                    engine_npc = self.engine.game_state.npcs[npc_id]
                    engine_npc.current_emotion = npc_data["current_emotion"]
                else:
                    # Add NPC from world data
                    from game_engine import NPCData
                    self.engine.game_state.npcs[npc_id] = NPCData(
                        id=npc_data["id"],
                        name=npc_data["name"],
                        role=npc_data["role"],
                        description=npc_data["personality"],
                        location=npc_data["current_location"],
                        current_emotion=npc_data["current_emotion"]
                    )

        # 4. Initialize quest system
        print("📜 Initializing quest system...")
        quest_state = QuestGameState(
            player_level=1,
            npc_emotions={
                npc_id: npc_data["current_emotion"]
                for npc_id, npc_data in self.world_state["npcs"].items()
            }
        )
        self.quest_manager = QuestManager(quest_state)

        # 5. Mark as initialized
        self.initialized = True
        print("\n✅ Game initialized successfully!")
        print(f"   - {len(LOCATIONS)} locations loaded")
        print(f"   - {len(NPCS)} NPCs loaded")
        print(f"   - {len(QUESTS)} quests available")

        return True

    # ========================================================================
    # Game Actions
    # ========================================================================

    async def talk_to_npc(self, npc_id: str, message: str) -> Dict[str, Any]:
        """
        Talk to an NPC.

        Args:
            npc_id: NPC identifier
            message: Player message

        Returns:
            Response including NPC dialogue, emotion, quests, etc.
        """
        if not self.initialized:
            return {"status": "error", "message": "Game not initialized"}

        # Process action through game engine
        result = await self.engine.process_player_action(
            "talk",
            npc_id=npc_id,
            player_message=message
        )

        # Update quest manager with new emotion state
        if result.get("status") == "success" and "npc_emotion" in result:
            if self.quest_manager:
                self.quest_manager.game_state.npc_emotions[npc_id] = result["npc_emotion"]

        return result

    async def accept_quest(self, quest_id: str) -> Dict[str, Any]:
        """
        Accept a quest.

        Args:
            quest_id: Quest identifier

        Returns:
            Result of quest acceptance
        """
        if not self.initialized or not self.quest_manager:
            return {"status": "error", "message": "Game not initialized"}

        quest = self.quest_manager.start_quest(quest_id)
        if quest:
            return {
                "status": "success",
                "message": f"Quest '{quest.title}' accepted!",
                "quest": self.quest_manager.get_quest_progress(quest_id)
            }
        else:
            return {
                "status": "error",
                "message": "Quest not available or already active"
            }

    async def update_quest_objective(
        self,
        quest_id: str,
        objective_id: str,
        progress: Optional[int] = None,
        completed: bool = False
    ) -> Dict[str, Any]:
        """
        Update quest objective progress.

        Args:
            quest_id: Quest identifier
            objective_id: Objective identifier
            progress: New progress value
            completed: Mark as completed

        Returns:
            Update result
        """
        if not self.initialized or not self.quest_manager:
            return {"status": "error", "message": "Game not initialized"}

        success = self.quest_manager.update_quest_objective(
            quest_id, objective_id, progress, completed
        )

        if success:
            # Check if quest can be completed
            if self.quest_manager.check_quest_completion(quest_id):
                return {
                    "status": "success",
                    "message": f"Objective completed! Quest '{quest_id}' ready to complete.",
                    "quest_complete_ready": True,
                    "progress": self.quest_manager.get_quest_progress(quest_id)
                }
            else:
                return {
                    "status": "success",
                    "message": "Objective updated",
                    "progress": self.quest_manager.get_quest_progress(quest_id)
                }
        else:
            return {"status": "error", "message": "Failed to update objective"}

    async def complete_quest(self, quest_id: str) -> Dict[str, Any]:
        """
        Complete a quest.

        Args:
            quest_id: Quest identifier

        Returns:
            Completion result with rewards
        """
        if not self.initialized or not self.quest_manager:
            return {"status": "error", "message": "Game not initialized"}

        # Complete quest through quest manager
        success = self.quest_manager.complete_quest(quest_id)

        if success:
            # Get quest data
            quest_data = QUESTS.get(quest_id)
            if quest_data:
                # Update engine state with rewards
                if self.engine.game_state and quest_data.get("reward"):
                    reward = quest_data["reward"]
                    self.engine.game_state.player.gold += reward.get("gold", 0)
                    self.engine.game_state.player.inventory.extend(reward.get("items", []))

                return {
                    "status": "success",
                    "message": f"Quest '{quest_data['title']}' completed!",
                    "rewards": quest_data.get("reward", {}),
                    "unlocks": quest_data.get("unlocks", [])
                }

        return {"status": "error", "message": "Failed to complete quest"}

    def get_available_quests(self, npc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get quests available to player.

        Args:
            npc_id: Optional NPC filter

        Returns:
            List of available quests
        """
        if not self.initialized or not self.quest_manager:
            return []

        return self.quest_manager.get_available_quests(npc_id)

    def get_active_quests(self) -> List[Dict[str, Any]]:
        """
        Get currently active quests.

        Returns:
            List of active quests with progress
        """
        if not self.initialized or not self.quest_manager:
            return []

        return self.quest_manager.get_active_quests_summary()

    def get_game_state(self) -> Dict[str, Any]:
        """
        Get complete game state.

        Returns:
            Dictionary with all game state data
        """
        if not self.initialized:
            return {}

        return {
            "engine_state": self.engine.get_public_state(),
            "world_state": self.world_state,
            "quest_progress": self.quest_manager.get_quest_chain_progress() if self.quest_manager else {},
            "active_quests": self.get_active_quests(),
            "available_quests": self.get_available_quests()
        }

    # ========================================================================
    # Testing & Validation
    # ========================================================================

    async def run_integration_test(self) -> Dict[str, Any]:
        """
        Run comprehensive integration test.

        Tests:
        1. Talk to innkeeper
        2. View available quests
        3. Accept rat_problem quest
        4. Update quest objectives
        5. Complete quest
        6. Verify emotion changes
        7. Check quest unlocks
        8. Voice synthesis (if available)

        Returns:
            Test results dictionary
        """
        print("\n🧪 Running integration tests...\n")
        test_results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }

        # Test 1: Talk to innkeeper
        print("Test 1: Talk to innkeeper Bob")
        try:
            result = await self.talk_to_npc(
                "innkeeper",
                "Hello, how are you?"
            )
            if result.get("status") == "success" and result.get("npc_response"):
                test_results["passed"].append("Talk to NPC")
                print(f"  ✅ Bob says: {result['npc_response'][:50]}...")
            else:
                test_results["failed"].append("Talk to NPC")
                print(f"  ❌ Failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            test_results["failed"].append(f"Talk to NPC: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 2: View available quests
        print("\nTest 2: View available quests")
        try:
            available = self.get_available_quests(npc_id="innkeeper")
            if any(q["id"] == "rat_problem" for q in available):
                test_results["passed"].append("Quest availability")
                print(f"  ✅ Found {len(available)} available quests")
            else:
                test_results["failed"].append("Quest availability")
                print(f"  ❌ Rat quest not available")
        except Exception as e:
            test_results["failed"].append(f"Quest availability: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 3: Accept rat_problem quest
        print("\nTest 3: Accept rat_problem quest")
        try:
            result = await self.accept_quest("rat_problem")
            if result.get("status") == "success":
                test_results["passed"].append("Quest acceptance")
                print(f"  ✅ Quest accepted")
            else:
                test_results["failed"].append("Quest acceptance")
                print(f"  ❌ Failed: {result.get('message')}")
        except Exception as e:
            test_results["failed"].append(f"Quest acceptance: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 4: Update quest objectives
        print("\nTest 4: Update quest objectives")
        try:
            # Complete "talk_to_bob" objective
            result = await self.update_quest_objective(
                "rat_problem",
                "talk_to_bob",
                completed=True
            )
            if result.get("status") == "success":
                # Complete "clear_rats" objective
                result = await self.update_quest_objective(
                    "rat_problem",
                    "clear_rats",
                    progress=5
                )
                if result.get("status") == "success":
                    # Complete "report_to_bob" objective
                    result = await self.update_quest_objective(
                        "rat_problem",
                        "report_to_bob",
                        completed=True
                    )
                    if result.get("status") == "success":
                        test_results["passed"].append("Quest objectives")
                        print(f"  ✅ All objectives completed")
                    else:
                        test_results["failed"].append("Quest objectives (3/3)")
                        print(f"  ❌ Failed final objective")
                else:
                    test_results["failed"].append("Quest objectives (2/3)")
                    print(f"  ❌ Failed rats objective")
            else:
                test_results["failed"].append("Quest objectives (1/3)")
                print(f"  ❌ Failed talk objective")
        except Exception as e:
            test_results["failed"].append(f"Quest objectives: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 5: Complete quest
        print("\nTest 5: Complete rat_problem quest")
        try:
            result = await self.complete_quest("rat_problem")
            if result.get("status") == "success":
                test_results["passed"].append("Quest completion")
                rewards = result.get("rewards", {})
                print(f"  ✅ Quest completed! Rewards: {rewards.get('gold', 0)} gold, {rewards.get('xp', 0)} XP")
            else:
                test_results["failed"].append("Quest completion")
                print(f"  ❌ Failed: {result.get('message')}")
        except Exception as e:
            test_results["failed"].append(f"Quest completion: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 6: Verify emotion changes
        print("\nTest 6: Verify emotion changes")
        try:
            if self.quest_manager:
                innkeeper_emotion = self.quest_manager.game_state.npc_emotions.get("innkeeper", {})
                if innkeeper_emotion.get("valence", 0) > 0.0:
                    test_results["passed"].append("Emotion changes")
                    print(f"  ✅ Innkeeper valence: {innkeeper_emotion.get('valence', 0):.2f}")
                    print(f"  ✅ Innkeeper trust: {innkeeper_emotion.get('trust', 0.5):.2f}")
                else:
                    test_results["warnings"].append("Emotion changes (no change)")
                    print(f"  ⚠️ No emotion change detected")
        except Exception as e:
            test_results["failed"].append(f"Emotion changes: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 7: Check quest unlocks
        print("\nTest 7: Check quest unlocks")
        try:
            available = self.get_available_quests()
            unlocked_ids = [q["id"] for q in available]
            if "lost_shipment" in unlocked_ids:
                test_results["passed"].append("Quest unlocks")
                print(f"  ✅ Quest 'lost_shipment' unlocked")
            else:
                test_results["warnings"].append("Quest unlocks (lost_shipment not available)")
                print(f"  ⚠️ Expected quest not unlocked yet (may need emotional requirements)")
        except Exception as e:
            test_results["failed"].append(f"Quest unlocks: {e}")
            print(f"  ❌ Exception: {e}")

        # Test 8: Voice synthesis (optional)
        print("\nTest 8: Voice synthesis")
        if await self.test_voice_synthesis():
            test_results["passed"].append("Voice synthesis")
            print("  ✅ Voice synthesis available")
        else:
            test_results["warnings"].append("Voice synthesis not configured")
            print("  ⚠️ Voice synthesis not configured (optional feature)")

        # Summary
        print("\n" + "="*60)
        print("Test Summary:")
        print(f"  ✅ Passed: {len(test_results['passed'])}")
        print(f"  ❌ Failed: {len(test_results['failed'])}")
        print(f"  ⚠️ Warnings: {len(test_results['warnings'])}")
        print("="*60 + "\n")

        return test_results

    async def test_voice_synthesis(self) -> bool:
        """
        Test voice synthesis endpoint.

        Returns:
            True if voice synthesis available
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.elle_url}/elle/game/voice/synthesize",
                    json={
                        "text": "Hello, traveler!",
                        "npc_id": "innkeeper",
                        "format": "wav"
                    },
                    timeout=5.0
                )
                return response.status_code == 200
        except:
            return False

    # ========================================================================
    # Interactive Mode
    # ========================================================================

    async def start_interactive_session(self):
        """
        Start interactive CLI session for testing.

        Commands:
        - talk <npc_id> <message>: Talk to NPC
        - quests: Show available quests
        - accept <quest_id>: Accept quest
        - active: Show active quests
        - complete <quest_id>: Complete quest
        - state: Show game state
        - test: Run integration tests
        - exit: Quit
        """
        print("\n" + "="*60)
        print("🎮 Elle Tavern Demo - Interactive Session")
        print("="*60)
        print("\nCommands:")
        print("  talk <npc_id> <message>  - Talk to NPC")
        print("  quests                   - Show available quests")
        print("  accept <quest_id>        - Accept quest")
        print("  active                   - Show active quests")
        print("  complete <quest_id>      - Complete quest")
        print("  state                    - Show game state")
        print("  test                     - Run integration tests")
        print("  exit                     - Quit")
        print()

        while True:
            try:
                command = input("> ").strip()
                if not command:
                    continue

                parts = command.split(maxsplit=2)
                cmd = parts[0].lower()

                if cmd == "exit":
                    print("Goodbye!")
                    break

                elif cmd == "talk" and len(parts) >= 3:
                    npc_id, message = parts[1], parts[2]
                    result = await self.talk_to_npc(npc_id, message)
                    if result.get("status") == "success":
                        print(f"\n{result['npc_name']}: {result['npc_response']}\n")
                    else:
                        print(f"\nError: {result.get('message')}\n")

                elif cmd == "quests":
                    quests = self.get_available_quests()
                    if quests:
                        print("\nAvailable Quests:")
                        for q in quests:
                            print(f"  - {q['id']}: {q['title']} (from {q['giver']})")
                        print()
                    else:
                        print("\nNo quests available.\n")

                elif cmd == "accept" and len(parts) >= 2:
                    quest_id = parts[1]
                    result = await self.accept_quest(quest_id)
                    print(f"\n{result.get('message')}\n")

                elif cmd == "active":
                    quests = self.get_active_quests()
                    if quests:
                        print("\nActive Quests:")
                        for q in quests:
                            print(f"  - {q['name']}: {len([o for o in q['objectives'] if o['completed']])}/{len(q['objectives'])} objectives")
                        print()
                    else:
                        print("\nNo active quests.\n")

                elif cmd == "complete" and len(parts) >= 2:
                    quest_id = parts[1]
                    result = await self.complete_quest(quest_id)
                    print(f"\n{result.get('message')}\n")

                elif cmd == "state":
                    state = self.get_game_state()
                    print(f"\nGame State: {state.get('quest_progress', {})}\n")

                elif cmd == "test":
                    await self.run_integration_test()

                else:
                    print("\nUnknown command. Type 'exit' to quit.\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for integration testing."""
    async with IntegratedGame() as game:
        if await game.initialize():
            # Run integration tests
            results = await game.run_integration_test()

            # Start interactive session if tests passed
            if len(results["failed"]) == 0:
                print("\n✅ All tests passed! Starting interactive session...\n")
                await game.start_interactive_session()
            else:
                print(f"\n❌ {len(results['failed'])} tests failed. Fix issues before interactive mode.\n")


if __name__ == "__main__":
    asyncio.run(main())
