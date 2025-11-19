# BigPlay Developer Tutorials

**Last Updated**: 2025-11-16

This guide provides hands-on tutorials for building different types of games with BigPlay. Each tutorial includes complete code examples, architecture diagrams, and best practices.

---

## Table of Contents

1. [Tutorial 1: Building a Text Adventure RPG](#tutorial-1-building-a-text-adventure-rpg)
2. [Tutorial 2: Social Simulation Game](#tutorial-2-social-simulation-game)
3. [Tutorial 3: Multiplayer Integration](#tutorial-3-multiplayer-integration)
4. [Tutorial 4: Custom Emotion Systems](#tutorial-4-custom-emotion-systems)
5. [Tutorial 5: Fine-Tuning with Local LLMs](#tutorial-5-fine-tuning-with-local-llms)
6. [Tutorial 6: Voice-First Game Design](#tutorial-6-voice-first-game-design)
7. [Tutorial 7: Quest System Deep Dive](#tutorial-7-quest-system-deep-dive)

---

## Tutorial 1: Building a Text Adventure RPG

**Duration**: 2-3 hours
**Difficulty**: Intermediate
**What You'll Build**: A classic text adventure with combat, inventory, and NPC relationships

### Overview

We'll build "The Lost Kingdom" - a text adventure featuring:
- 5 locations with environmental storytelling
- 8 NPCs with emotional memory
- Turn-based combat system
- Inventory and item system
- Dynamic quest generation based on player actions

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                 GAME LOOP                            │
│                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │ Parse Input │ →  │ BigPlay API │ →  │ Update  │ │
│  │             │    │             │    │ State   │ │
│  └─────────────┘    └─────────────┘    └─────────┘ │
│         ▲                                     │      │
│         └─────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘

State Management:
- Player: Health, inventory, location, relationships
- NPCs: Emotions, memories, quests
- World: Location states, time of day, events
```

### Step 1: Define Game State

```python
# lost_kingdom/models.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class Location(Enum):
    VILLAGE = "village"
    FOREST = "dark_forest"
    CASTLE = "abandoned_castle"
    DUNGEON = "dungeon"
    TEMPLE = "ancient_temple"

@dataclass
class Item:
    id: str
    name: str
    description: str
    item_type: str  # weapon, potion, key, quest_item
    properties: Dict[str, any] = field(default_factory=dict)

@dataclass
class PlayerState:
    name: str
    health: int = 100
    max_health: int = 100
    level: int = 1
    experience: int = 0
    gold: int = 50
    location: Location = Location.VILLAGE
    inventory: List[Item] = field(default_factory=list)
    equipped_weapon: Optional[Item] = None

    # Relationship tracking
    npc_relationships: Dict[str, float] = field(default_factory=dict)
    # -1.0 (hostile) to +1.0 (ally)

@dataclass
class NPCState:
    id: str
    name: str
    role: str
    location: Location
    health: int = 100

    # Emotional state (PAD model)
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0,
        "arousal": 0.5,
        "dominance": 0.5,
        "trust": 0.5
    })

    # Dialogue history for memory
    dialogue_history: List[str] = field(default_factory=list)

    # Quests this NPC can give
    available_quests: List[str] = field(default_factory=list)

@dataclass
class GameState:
    player: PlayerState
    npcs: Dict[str, NPCState]
    world_state: Dict[str, any] = field(default_factory=dict)
    active_quests: List[Dict] = field(default_factory=list)
    completed_quests: List[str] = field(default_factory=list)
    time_of_day: str = "morning"  # morning, afternoon, evening, night
    day_number: int = 1
```

### Step 2: Create Game Manager

```python
# lost_kingdom/game_manager.py
import asyncio
import httpx
from typing import Optional
from .models import GameState, PlayerState, NPCState, Location, Item

class LostKingdomGame:
    """Main game manager for The Lost Kingdom."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient()
        self.game_state = self._initialize_game()

    def _initialize_game(self) -> GameState:
        """Initialize game with starting state."""

        # Create player
        player = PlayerState(
            name="Hero",
            location=Location.VILLAGE
        )

        # Create NPCs
        npcs = {
            "innkeeper": NPCState(
                id="innkeeper",
                name="Bram the Innkeeper",
                role="innkeeper",
                location=Location.VILLAGE,
                emotional_state={
                    "valence": 0.3,
                    "arousal": 0.4,
                    "dominance": 0.6,
                    "trust": 0.5
                }
            ),
            "blacksmith": NPCState(
                id="blacksmith",
                name="Thora Ironforge",
                role="blacksmith",
                location=Location.VILLAGE,
                emotional_state={
                    "valence": 0.0,
                    "arousal": 0.6,
                    "dominance": 0.7,
                    "trust": 0.3
                }
            ),
            "wizard": NPCState(
                id="wizard",
                name="Aldric the Wise",
                role="wizard",
                location=Location.TEMPLE,
                emotional_state={
                    "valence": 0.2,
                    "arousal": 0.3,
                    "dominance": 0.8,
                    "trust": 0.6
                }
            ),
            "guard": NPCState(
                id="guard",
                name="Captain Roderick",
                role="guard_captain",
                location=Location.CASTLE,
                emotional_state={
                    "valence": -0.2,
                    "arousal": 0.7,
                    "dominance": 0.9,
                    "trust": 0.4
                }
            ),
            # Add more NPCs...
        }

        # Initial world state
        world_state = {
            "village_status": "peaceful",
            "castle_explored": False,
            "dungeon_unlocked": False,
            "temple_blessed": False,
            "dragon_defeated": False
        }

        return GameState(
            player=player,
            npcs=npcs,
            world_state=world_state
        )

    async def process_action(self, player_input: str) -> Dict:
        """
        Process player action through BigPlay API.

        Returns dialogue, world updates, quest updates, etc.
        """

        # Detect action type
        action_type = self._classify_input(player_input)

        # Get NPCs in current location
        local_npcs = [
            npc for npc in self.game_state.npcs.values()
            if npc.location == self.game_state.player.location
        ]

        # Build request for BigPlay API
        request = {
            "game_state": {
                "scene_id": self.game_state.player.location.value,
                "npcs": [
                    {
                        "id": npc.id,
                        "name": npc.name,
                        "role": npc.role,
                        "emotional_state": npc.emotional_state,
                        "dialogue_history": npc.dialogue_history[-5:]  # Last 5 exchanges
                    }
                    for npc in local_npcs
                ],
                "player": {
                    "name": self.game_state.player.name,
                    "location": self.game_state.player.location.value,
                    "health": self.game_state.player.health,
                    "level": self.game_state.player.level,
                    "inventory_summary": [item.name for item in self.game_state.player.inventory[:5]]
                },
                "world_state": self.game_state.world_state,
                "time_of_day": self.game_state.time_of_day
            },
            "player_intent": {
                "type": action_type,
                "raw_input": player_input,
                "context": {
                    "active_quests": [q["title"] for q in self.game_state.active_quests],
                    "recent_actions": self._get_recent_actions()
                }
            }
        }

        # Call BigPlay API
        response = await self.client.post(
            f"{self.api_url}/elle/game/action",
            json=request,
            timeout=30.0
        )

        if response.status_code != 200:
            return {
                "action_type": "error",
                "content": {
                    "message": f"Error: {response.text}"
                }
            }

        action = response.json()

        # Update game state based on action
        await self._update_state(action)

        return action

    def _classify_input(self, player_input: str) -> str:
        """Classify player input into action types."""
        lower_input = player_input.lower()

        # Movement
        if any(word in lower_input for word in ["go", "move", "travel", "walk", "north", "south", "east", "west"]):
            return "move"

        # Combat
        if any(word in lower_input for word in ["attack", "fight", "kill", "strike", "defend"]):
            return "combat"

        # Inventory
        if any(word in lower_input for word in ["inventory", "items", "equip", "use"]):
            return "inventory"

        # Talking to NPC
        if any(word in lower_input for word in ["talk", "speak", "ask", "tell", "say"]):
            return "talk_to_npc"

        # Examining
        if any(word in lower_input for word in ["look", "examine", "inspect", "search"]):
            return "examine"

        # Default to dialogue
        return "talk_to_npc"

    async def _update_state(self, action: Dict):
        """Update game state based on BigPlay action."""

        # Update NPC emotions
        for updated_npc in action.get("updated_npcs", []):
            npc_id = updated_npc["id"]
            if npc_id in self.game_state.npcs:
                self.game_state.npcs[npc_id].emotional_state = updated_npc["emotional_state"]

                # Add to dialogue history
                if action.get("action_type") == "dialogue":
                    dialogue = action["content"].get("npc_dialogue", "")
                    self.game_state.npcs[npc_id].dialogue_history.append(dialogue)

        # Update world state
        if "world_updates" in action.get("content", {}):
            self.game_state.world_state.update(action["content"]["world_updates"])

        # Check for new quests
        if action.get("action_type") == "quest_offered":
            quest = action["content"]["quest"]
            self.game_state.active_quests.append(quest)

        # Advance time
        self._advance_time()

    def _advance_time(self):
        """Advance in-game time."""
        time_sequence = ["morning", "afternoon", "evening", "night"]
        current_index = time_sequence.index(self.game_state.time_of_day)

        if current_index == len(time_sequence) - 1:
            # New day
            self.game_state.time_of_day = "morning"
            self.game_state.day_number += 1
        else:
            self.game_state.time_of_day = time_sequence[current_index + 1]

    def _get_recent_actions(self) -> List[str]:
        """Get recent player actions for context."""
        # Implement action logging
        return []

    async def save_game(self, filename: str):
        """Save game state to file."""
        import json
        from dataclasses import asdict

        with open(filename, 'w') as f:
            # Convert dataclasses to dict
            state_dict = {
                "player": asdict(self.game_state.player),
                "npcs": {k: asdict(v) for k, v in self.game_state.npcs.items()},
                "world_state": self.game_state.world_state,
                "active_quests": self.game_state.active_quests,
                "completed_quests": self.game_state.completed_quests,
                "time_of_day": self.game_state.time_of_day,
                "day_number": self.game_state.day_number
            }
            json.dump(state_dict, f, indent=2)

    async def load_game(self, filename: str):
        """Load game state from file."""
        import json

        with open(filename, 'r') as f:
            state_dict = json.load(f)

        # Reconstruct game state
        # (Implementation omitted for brevity)
```

### Step 3: Create Game Loop

```python
# lost_kingdom/main.py
import asyncio
from .game_manager import LostKingdomGame

async def main():
    """Main game loop."""

    # ASCII art title
    print("""
    ╔══════════════════════════════════════╗
    ║      THE LOST KINGDOM               ║
    ║   A Text Adventure Powered by AI     ║
    ╚══════════════════════════════════════╝
    """)

    # Initialize game
    game = LostKingdomGame()

    # Show intro
    print("\nYou awaken in a small village at the edge of a dark forest.")
    print("The kingdom has fallen into darkness, and only you can restore it.")
    print("\nType 'help' for commands, 'quit' to exit.\n")

    # Game loop
    while True:
        # Show current status
        player = game.game_state.player
        print(f"\n[{game.game_state.time_of_day.upper()}] Location: {player.location.value}")
        print(f"Health: {player.health}/{player.max_health} | Gold: {player.gold} | Level: {player.level}")

        # Get player input
        player_input = input("\n> ").strip()

        if not player_input:
            continue

        # Handle special commands
        if player_input.lower() == "quit":
            # Ask to save
            save = input("Save game? (y/n): ").lower()
            if save == 'y':
                await game.save_game("savegame.json")
                print("Game saved!")
            print("Thanks for playing!")
            break

        if player_input.lower() == "help":
            print("""
            Commands:
            - talk to <npc>: Converse with NPCs
            - go <direction>: Move to a new location
            - look: Examine your surroundings
            - inventory: View your items
            - attack <target>: Enter combat
            - use <item>: Use an item
            - quest: View active quests
            - save: Save your progress
            - quit: Exit game

            You can also just type naturally - the AI will understand!
            Example: "Ask the innkeeper about the dark forest"
            """)
            continue

        if player_input.lower() == "inventory":
            if not player.inventory:
                print("Your inventory is empty.")
            else:
                print("\n=== INVENTORY ===")
                for item in player.inventory:
                    print(f"- {item.name}: {item.description}")
            continue

        if player_input.lower() == "quest":
            if not game.game_state.active_quests:
                print("You have no active quests.")
            else:
                print("\n=== ACTIVE QUESTS ===")
                for quest in game.game_state.active_quests:
                    print(f"\n{quest['title']}")
                    print(f"  {quest['description']}")
                    for i, obj in enumerate(quest['objectives'], 1):
                        status = "✓" if obj.get('completed') else "○"
                        print(f"  {status} {obj['description']}")
            continue

        # Process action through BigPlay
        print("\n[Processing...]")
        action = await game.process_action(player_input)

        # Display result
        if action["action_type"] == "dialogue":
            content = action["content"]
            npc_name = content.get("npc_name", "Someone")
            dialogue = content.get("npc_dialogue", "...")

            print(f"\n{npc_name}: \"{dialogue}\"")

            # Play voice if available
            if "voice_audio_url" in content:
                # Optional: Play audio
                pass

        elif action["action_type"] == "world_description":
            description = action["content"]["description"]
            print(f"\n{description}")

        elif action["action_type"] == "combat":
            # Handle combat
            combat_result = action["content"]
            print(f"\n{combat_result['description']}")

            # Update player health
            if "player_damage" in combat_result:
                player.health -= combat_result["player_damage"]
                print(f"You took {combat_result['player_damage']} damage!")

                if player.health <= 0:
                    print("\n=== GAME OVER ===")
                    print("You have been defeated...")
                    break

        elif action["action_type"] == "quest_offered":
            quest = action["content"]["quest"]
            print(f"\n=== NEW QUEST ===")
            print(f"{quest['title']}")
            print(f"{quest['description']}")

            accept = input("\nAccept quest? (y/n): ").lower()
            if accept == 'y':
                print("Quest accepted!")
            else:
                # Remove from active quests
                game.game_state.active_quests = [
                    q for q in game.game_state.active_quests
                    if q['id'] != quest['id']
                ]
                print("Quest declined.")

        elif action["action_type"] == "hint":
            hint = action["content"]["hint_text"]
            print(f"\n💡 Hint: {hint}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Add Combat System

```python
# lost_kingdom/combat.py
import asyncio
import random
from typing import Dict, List
from .models import PlayerState, NPCState, Item

class CombatSystem:
    """Turn-based combat system."""

    def __init__(self, game_manager):
        self.game = game_manager

    async def start_combat(self, enemy_id: str) -> Dict:
        """
        Start combat encounter with enemy.

        Returns combat result with damage, victory/defeat, loot, etc.
        """

        player = self.game.game_state.player
        enemy = self.game.game_state.npcs.get(enemy_id)

        if not enemy:
            return {"error": "Enemy not found"}

        print(f"\n⚔️  COMBAT STARTED ⚔️")
        print(f"You vs {enemy.name}")

        combat_log = []
        turn = 1

        while player.health > 0 and enemy.health > 0:
            print(f"\n--- Turn {turn} ---")
            print(f"Your HP: {player.health}/{player.max_health}")
            print(f"{enemy.name} HP: {enemy.health}/100")

            # Player turn
            print("\nActions: [a]ttack, [d]efend, [i]tem, [r]un")
            action = input("> ").lower()

            if action == 'a':
                damage = self._calculate_damage(player, enemy)
                enemy.health -= damage
                log_entry = f"You dealt {damage} damage to {enemy.name}!"
                print(log_entry)
                combat_log.append(log_entry)

            elif action == 'd':
                print("You brace for impact!")
                # Reduce incoming damage next turn
                # (Implementation omitted)

            elif action == 'i':
                # Use item
                self._use_item_in_combat(player)

            elif action == 'r':
                if random.random() < 0.5:
                    print("You escaped!")
                    return {"result": "fled", "log": combat_log}
                else:
                    print("You failed to escape!")

            # Check enemy defeated
            if enemy.health <= 0:
                print(f"\n🎉 Victory! You defeated {enemy.name}!")

                # Award XP and loot
                xp_gain = 50 * player.level
                gold_gain = random.randint(10, 50)

                player.experience += xp_gain
                player.gold += gold_gain

                print(f"Gained {xp_gain} XP and {gold_gain} gold!")

                # Check level up
                if player.experience >= player.level * 100:
                    player.level += 1
                    player.max_health += 10
                    player.health = player.max_health
                    print(f"\n⭐ LEVEL UP! You are now level {player.level}!")

                return {
                    "result": "victory",
                    "xp_gain": xp_gain,
                    "gold_gain": gold_gain,
                    "log": combat_log
                }

            # Enemy turn
            enemy_damage = self._calculate_damage(enemy, player)
            player.health -= enemy_damage
            log_entry = f"{enemy.name} dealt {enemy_damage} damage to you!"
            print(log_entry)
            combat_log.append(log_entry)

            # Check player defeated
            if player.health <= 0:
                print(f"\n💀 Defeat! You were slain by {enemy.name}...")
                return {
                    "result": "defeat",
                    "log": combat_log
                }

            turn += 1

    def _calculate_damage(self, attacker, defender) -> int:
        """Calculate damage dealt."""
        base_damage = 10

        # Attacker bonuses
        if isinstance(attacker, PlayerState):
            if attacker.equipped_weapon:
                base_damage += attacker.equipped_weapon.properties.get("damage", 0)
            base_damage += attacker.level * 2

        # Random variance
        damage = random.randint(int(base_damage * 0.8), int(base_damage * 1.2))

        return max(1, damage)

    def _use_item_in_combat(self, player: PlayerState):
        """Use item during combat."""
        if not player.inventory:
            print("You have no items!")
            return

        print("\n=== ITEMS ===")
        for i, item in enumerate(player.inventory, 1):
            print(f"{i}. {item.name}")

        try:
            choice = int(input("Use item #: ")) - 1
            item = player.inventory[choice]

            if item.item_type == "potion":
                heal_amount = item.properties.get("heal", 0)
                player.health = min(player.max_health, player.health + heal_amount)
                print(f"You used {item.name} and restored {heal_amount} HP!")
                player.inventory.pop(choice)
            else:
                print("Can't use that in combat!")
        except (ValueError, IndexError):
            print("Invalid item!")
```

### Step 5: Running the Game

```bash
# Install dependencies
pip install httpx

# Start BigPlay server
cd apps/elle_game_engine
PYTHONPATH=. uvicorn app.main:app --reload

# In another terminal, run the game
cd lost_kingdom
python main.py
```

### Example Gameplay

```
╔══════════════════════════════════════╗
║      THE LOST KINGDOM               ║
║   A Text Adventure Powered by AI     ║
╚══════════════════════════════════════╝

You awaken in a small village at the edge of a dark forest.
The kingdom has fallen into darkness, and only you can restore it.

Type 'help' for commands, 'quit' to exit.

[MORNING] Location: village
Health: 100/100 | Gold: 50 | Level: 1

> look around

The village is quiet in the morning light. You see:
- An inn with smoke rising from its chimney
- A blacksmith's forge, hammer ringing on metal
- A path leading north into the dark forest
- Villagers going about their morning routines

[MORNING] Location: village
Health: 100/100 | Gold: 50 | Level: 1

> talk to the innkeeper

[Processing...]

Bram the Innkeeper: "Ah, you're finally awake! Strange times these are.
The old castle to the north has been abandoned since the king disappeared.
Some say there's treasure there, but also great danger. If you're brave
enough to explore, I can offer you some supplies. Interested?"

[MORNING] Location: village
Health: 100/100 | Gold: 50 | Level: 1

> yes, I'm interested

[Processing...]

Bram the Innkeeper: "Excellent! I'll mark the castle on your map. But be
careful - the forest is full of dangers. You might want to visit Thora
at the forge first and get yourself a proper weapon."

=== NEW QUEST ===
Explore the Abandoned Castle
Investigate the old castle and discover what happened to the king.

Accept quest? (y/n): y
Quest accepted!
```

### Best Practices

1. **Session Management**: Use BigPlay's session endpoints to persist NPC memory
2. **Error Handling**: Always handle API failures gracefully
3. **Performance**: Cache responses for repeated interactions
4. **Voice Integration**: Add voice synthesis for immersive experience
5. **Save System**: Regularly save game state to prevent loss

---

## Tutorial 2: Social Simulation Game

**Duration**: 3-4 hours
**Difficulty**: Advanced
**What You'll Build**: A relationship-driven social sim like Stardew Valley

### Overview

We'll build "Valley of Friends" - a social simulation featuring:
- 12 NPCs with unique personalities
- Relationship system (friendship, romance, rivalry)
- Gift-giving mechanics
- Daily schedules and routines
- Special events and festivals
- Emergent storytelling based on NPC interactions

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│               SOCIAL SIMULATION                         │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  NPC Social  │   │ Relationship │   │   Event    │ │
│  │  Network     │ ←→│   Graph      │ →│  Generator │ │
│  └──────────────┘   └──────────────┘   └────────────┘ │
│         │                   │                 │        │
│         └───────────────────┴─────────────────┘        │
│                             │                          │
│                             ▼                          │
│                    ┌──────────────┐                    │
│                    │  BigPlay API │                    │
│                    └──────────────┘                    │
└─────────────────────────────────────────────────────────┘

Key Systems:
- Relationship Network: NPCs have relationships with each other
- Daily Schedules: NPCs move around based on time/day
- Gift System: Items affect relationships
- Event Triggers: Special conversations based on relationship levels
```

### Step 1: Relationship System

```python
# valley_of_friends/relationships.py
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum

class RelationshipStatus(Enum):
    STRANGER = 0
    ACQUAINTANCE = 1  # 100+ points
    FRIEND = 2        # 500+ points
    CLOSE_FRIEND = 3  # 1000+ points
    ROMANTIC = 4      # 2000+ points
    MARRIED = 5       # Special status

@dataclass
class Relationship:
    """Relationship between player and NPC (or NPC and NPC)."""

    target_id: str
    points: int = 0
    status: RelationshipStatus = RelationshipStatus.STRANGER

    # Gift history
    gifts_given: List[Dict] = field(default_factory=list)
    last_gift_date: int = 0  # Day number

    # Conversation history
    conversation_count: int = 0
    last_conversation_date: int = 0

    # Special flags
    has_met: bool = False
    is_dating: bool = False
    is_married: bool = False
    heart_events_seen: List[int] = field(default_factory=list)

    def add_points(self, points: int, reason: str = ""):
        """Add relationship points."""
        self.points += points
        self._update_status()

    def _update_status(self):
        """Update status based on points."""
        if self.points >= 2000:
            self.status = RelationshipStatus.ROMANTIC
        elif self.points >= 1000:
            self.status = RelationshipStatus.CLOSE_FRIEND
        elif self.points >= 500:
            self.status = RelationshipStatus.FRIEND
        elif self.points >= 100:
            self.status = RelationshipStatus.ACQUAINTANCE
        else:
            self.status = RelationshipStatus.STRANGER

class RelationshipManager:
    """Manages all relationships in the game."""

    def __init__(self):
        self.player_relationships: Dict[str, Relationship] = {}
        self.npc_relationships: Dict[str, Dict[str, Relationship]] = {}

    def get_player_relationship(self, npc_id: str) -> Relationship:
        """Get relationship between player and NPC."""
        if npc_id not in self.player_relationships:
            self.player_relationships[npc_id] = Relationship(target_id=npc_id)
        return self.player_relationships[npc_id]

    def give_gift(
        self,
        npc_id: str,
        item_name: str,
        item_preference: str,
        current_day: int
    ) -> Dict:
        """
        Process gift-giving.

        Args:
            npc_id: Target NPC
            item_name: Name of gift
            item_preference: "love", "like", "neutral", "dislike", "hate"
            current_day: Current day number

        Returns:
            Result with points gained and NPC reaction
        """
        relationship = self.get_player_relationship(npc_id)

        # Check if already gave gift today
        if relationship.last_gift_date == current_day:
            return {
                "success": False,
                "reason": "already_gave_gift_today",
                "message": f"You already gave a gift today!"
            }

        # Calculate points based on preference
        points_map = {
            "love": 80,
            "like": 45,
            "neutral": 20,
            "dislike": -20,
            "hate": -40
        }

        points = points_map.get(item_preference, 0)

        # Birthday bonus (2x points)
        # (Would check if today is NPC's birthday)

        # Add points
        relationship.add_points(points, f"gift_{item_name}")
        relationship.last_gift_date = current_day
        relationship.gifts_given.append({
            "item": item_name,
            "day": current_day,
            "preference": item_preference,
            "points": points
        })

        # Generate reaction
        reactions = {
            "love": f"Oh wow, {item_name}! This is my favorite! Thank you so much!",
            "like": f"Oh, {item_name}! I like this, thank you!",
            "neutral": f"Oh, {item_name}. Thanks, I guess.",
            "dislike": f"Uh... {item_name}? I don't really like this...",
            "hate": f"{item_name}?! I hate this! Why would you give me this?"
        }

        return {
            "success": True,
            "points_gained": points,
            "new_total": relationship.points,
            "new_status": relationship.status.name,
            "reaction": reactions.get(item_preference, "Thanks.")
        }

    def check_heart_event(self, npc_id: str) -> Optional[int]:
        """
        Check if relationship level qualifies for heart event.

        Returns event level (2, 4, 6, 8, 10 hearts) or None
        """
        relationship = self.get_player_relationship(npc_id)

        # Heart events at specific relationship thresholds
        heart_events = [
            (2, 500),   # 2 hearts
            (4, 1000),  # 4 hearts
            (6, 1500),  # 6 hearts
            (8, 2000),  # 8 hearts
            (10, 2500)  # 10 hearts (marriage candidate only)
        ]

        for hearts, threshold in reversed(heart_events):
            if (relationship.points >= threshold and
                hearts not in relationship.heart_events_seen):
                return hearts

        return None
```

### Step 2: NPC Schedules

```python
# valley_of_friends/schedules.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class DayOfWeek(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class Weather(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    SNOWY = "snowy"

@dataclass
class ScheduleEntry:
    """Single schedule entry for NPC."""
    time: int  # Hour (0-23)
    location: str
    activity: str
    dialogue_hints: List[str] = None  # Hints for what NPC might say

class NPCSchedule:
    """Manages NPC daily schedule."""

    def __init__(self, npc_id: str):
        self.npc_id = npc_id
        self.schedules: Dict[str, List[ScheduleEntry]] = {}
        self._create_schedules()

    def _create_schedules(self):
        """Create schedules for different conditions."""

        if self.npc_id == "alex":  # Example: Athletic character
            # Regular weekday
            self.schedules["weekday_sunny"] = [
                ScheduleEntry(6, "home", "waking_up"),
                ScheduleEntry(7, "kitchen", "breakfast"),
                ScheduleEntry(9, "beach", "exercising",
                    ["I love working out in the morning!",
                     "The ocean breeze is refreshing."]),
                ScheduleEntry(12, "town_square", "socializing"),
                ScheduleEntry(15, "gym", "training"),
                ScheduleEntry(18, "home", "dinner"),
                ScheduleEntry(20, "living_room", "relaxing"),
                ScheduleEntry(22, "bedroom", "sleeping")
            ]

            # Rainy day (stays inside more)
            self.schedules["weekday_rainy"] = [
                ScheduleEntry(6, "home", "waking_up"),
                ScheduleEntry(7, "kitchen", "breakfast"),
                ScheduleEntry(9, "gym", "indoor_training",
                    ["Can't run outside in this weather.",
                     "Guess I'll hit the weights instead."]),
                ScheduleEntry(14, "library", "reading"),
                ScheduleEntry(18, "home", "dinner"),
                ScheduleEntry(20, "living_room", "watching_tv"),
                ScheduleEntry(22, "bedroom", "sleeping")
            ]

            # Weekend
            self.schedules["weekend_sunny"] = [
                ScheduleEntry(8, "home", "sleeping_in"),
                ScheduleEntry(10, "beach", "playing_gridball",
                    ["Want to join our game?",
                     "Nothing beats weekend beach sports!"]),
                ScheduleEntry(13, "saloon", "lunch_with_friends"),
                ScheduleEntry(16, "town_square", "hanging_out"),
                ScheduleEntry(19, "home", "dinner"),
                ScheduleEntry(21, "saloon", "evening_drinks"),
                ScheduleEntry(23, "home", "sleeping")
            ]

    def get_location(
        self,
        hour: int,
        day_of_week: DayOfWeek,
        weather: Weather,
        relationship_level: int = 0
    ) -> ScheduleEntry:
        """
        Get NPC location and activity for given time.

        Can vary based on:
        - Time of day
        - Day of week
        - Weather
        - Relationship level (high relationship = special schedules)
        - Season
        - Special events
        """

        # Determine schedule key
        is_weekend = day_of_week in [DayOfWeek.SATURDAY, DayOfWeek.SUNDAY]
        day_type = "weekend" if is_weekend else "weekday"
        schedule_key = f"{day_type}_{weather.value}"

        # Get schedule
        schedule = self.schedules.get(
            schedule_key,
            self.schedules.get("weekday_sunny", [])
        )

        # Find entry for current hour
        for entry in reversed(schedule):
            if hour >= entry.time:
                return entry

        # Default to first entry
        return schedule[0] if schedule else ScheduleEntry(
            0, "home", "sleeping"
        )

class ScheduleManager:
    """Manages schedules for all NPCs."""

    def __init__(self):
        self.npc_schedules: Dict[str, NPCSchedule] = {}

    def register_npc(self, npc_id: str):
        """Register NPC and create their schedule."""
        self.npc_schedules[npc_id] = NPCSchedule(npc_id)

    def get_npc_location(
        self,
        npc_id: str,
        hour: int,
        day_of_week: DayOfWeek,
        weather: Weather
    ) -> str:
        """Get NPC's current location."""
        if npc_id not in self.npc_schedules:
            return "home"

        schedule = self.npc_schedules[npc_id]
        entry = schedule.get_location(hour, day_of_week, weather)
        return entry.location

    def get_npcs_at_location(
        self,
        location: str,
        hour: int,
        day_of_week: DayOfWeek,
        weather: Weather
    ) -> List[str]:
        """Get all NPCs at a specific location."""
        npcs = []

        for npc_id, schedule in self.npc_schedules.items():
            entry = schedule.get_location(hour, day_of_week, weather)
            if entry.location == location:
                npcs.append(npc_id)

        return npcs
```

### Step 3: Event System

```python
# valley_of_friends/events.py
import random
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

@dataclass
class GameEvent:
    """Special event that can trigger."""
    id: str
    title: str
    description: str
    event_type: str  # "heart_event", "festival", "cutscene", "random"

    # Trigger conditions
    required_day: Optional[int] = None
    required_season: Optional[str] = None
    required_location: Optional[str] = None
    required_npcs: List[str] = None
    required_relationship: Dict[str, int] = None  # {npc_id: min_points}
    required_quests: List[str] = None

    # Content
    dialogue: List[Dict] = None  # List of dialogue exchanges
    choices: List[Dict] = None  # Player choices
    outcomes: Dict[str, any] = None  # Results (relationship changes, items, etc.)

    # Flags
    can_repeat: bool = False
    priority: int = 0  # Higher = triggers first

class EventManager:
    """Manages game events and triggers."""

    def __init__(self, game_manager):
        self.game = game_manager
        self.events: Dict[str, GameEvent] = {}
        self.triggered_events: List[str] = []
        self._register_events()

    def _register_events(self):
        """Register all game events."""

        # Heart Event Example: Alex 2-heart event
        self.events["alex_2heart"] = GameEvent(
            id="alex_2heart",
            title="Beach Workout",
            description="Alex invites you to work out together",
            event_type="heart_event",
            required_location="beach",
            required_relationship={"alex": 500},
            dialogue=[
                {
                    "speaker": "alex",
                    "text": "Hey! I see you here at the beach a lot. Want to work out together?"
                },
                {
                    "type": "choice",
                    "prompt": "What do you say?",
                    "options": [
                        {
                            "text": "Sure, I'd love to!",
                            "outcome": "accept",
                            "relationship_change": {"alex": 50}
                        },
                        {
                            "text": "Maybe another time.",
                            "outcome": "decline",
                            "relationship_change": {"alex": -10}
                        }
                    ]
                },
                {
                    "speaker": "alex",
                    "text": {
                        "accept": "Awesome! Let's start with some pushups. Follow my lead!",
                        "decline": "Oh, okay... Well, the offer stands if you change your mind."
                    }
                }
            ],
            outcomes={
                "accept": {
                    "relationship_changes": {"alex": 50},
                    "unlocks": ["alex_workout_buddy"],
                    "memories": ["worked_out_with_alex"]
                },
                "decline": {
                    "relationship_changes": {"alex": -10}
                }
            }
        )

        # Festival Event Example
        self.events["spring_festival"] = GameEvent(
            id="spring_festival",
            title="Spring Festival",
            description="Annual spring celebration in town square",
            event_type="festival",
            required_day=13,  # 13th of Spring
            required_season="spring",
            required_location="town_square",
            can_repeat=True,  # Repeats each year
            dialogue=[
                {
                    "speaker": "mayor",
                    "text": "Welcome to the Spring Festival! Enjoy the food, games, and company!"
                }
            ],
            outcomes={
                "attended": {
                    "relationship_changes": {
                        npc_id: 20 for npc_id in ["alex", "emily", "sam", "abigail"]
                    }
                }
            }
        )

        # Random Event Example
        self.events["shooting_star"] = GameEvent(
            id="shooting_star",
            title="Shooting Star",
            description="You see a shooting star!",
            event_type="random",
            dialogue=[
                {
                    "speaker": "narrator",
                    "text": "A shooting star streaks across the night sky! Make a wish..."
                }
            ],
            outcomes={
                "wished": {
                    "luck_boost": 1,  # Extra luck next day
                    "memories": ["saw_shooting_star"]
                }
            }
        )

    async def check_events(self, context: Dict) -> Optional[GameEvent]:
        """
        Check if any events should trigger.

        Args:
            context: Current game context (location, time, relationships, etc.)

        Returns:
            Event to trigger, or None
        """

        eligible_events = []

        for event in self.events.values():
            # Skip if already triggered (unless repeatable)
            if event.id in self.triggered_events and not event.can_repeat:
                continue

            # Check all conditions
            if not self._check_conditions(event, context):
                continue

            eligible_events.append(event)

        if not eligible_events:
            return None

        # Return highest priority event
        return max(eligible_events, key=lambda e: e.priority)

    def _check_conditions(self, event: GameEvent, context: Dict) -> bool:
        """Check if event conditions are met."""

        # Check day
        if event.required_day and context.get("day") != event.required_day:
            return False

        # Check season
        if event.required_season and context.get("season") != event.required_season:
            return False

        # Check location
        if event.required_location and context.get("location") != event.required_location:
            return False

        # Check NPCs present
        if event.required_npcs:
            present_npcs = context.get("npcs_present", [])
            if not all(npc in present_npcs for npc in event.required_npcs):
                return False

        # Check relationships
        if event.required_relationship:
            relationships = context.get("relationships", {})
            for npc_id, min_points in event.required_relationship.items():
                if relationships.get(npc_id, 0) < min_points:
                    return False

        # Check quests
        if event.required_quests:
            completed_quests = context.get("completed_quests", [])
            if not all(q in completed_quests for q in event.required_quests):
                return False

        return True

    async def trigger_event(self, event: GameEvent, player_choices: Dict = None) -> Dict:
        """
        Trigger event and return results.

        Args:
            event: Event to trigger
            player_choices: Dictionary of player choices made during event

        Returns:
            Event outcome with relationship changes, unlocks, etc.
        """

        # Mark as triggered
        if event.id not in self.triggered_events:
            self.triggered_events.append(event.id)

        # Process dialogue with choices
        # (Implementation would present dialogue to player,
        #  collect choices, and determine outcome path)

        # Get final outcome based on choices
        outcome_key = player_choices.get("final_choice", "default")
        outcome = event.outcomes.get(outcome_key, {})

        # Apply relationship changes
        for npc_id, points in outcome.get("relationship_changes", {}).items():
            relationship = self.game.relationship_manager.get_player_relationship(npc_id)
            relationship.add_points(points, f"event_{event.id}")

        return {
            "event_id": event.id,
            "outcome": outcome_key,
            "changes": outcome
        }
```

### Step 4: Integration with BigPlay

```python
# valley_of_friends/bigplay_integration.py
import httpx
from typing import Dict, List

class SocialSimBigPlayClient:
    """BigPlay client customized for social simulation."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient()

    async def get_contextual_dialogue(
        self,
        npc_id: str,
        npc_state: Dict,
        player_action: str,
        context: Dict
    ) -> Dict:
        """
        Get contextually-aware NPC dialogue.

        Context includes:
        - Relationship level
        - Recent gifts
        - Current location and activity
        - Weather, time, day
        - Other NPCs present
        - Recent events
        """

        # Build rich context for BigPlay
        request = {
            "game_state": {
                "scene_id": context["location"],
                "npcs": [{
                    "id": npc_id,
                    "name": npc_state["name"],
                    "role": npc_state["role"],
                    "emotional_state": npc_state["emotional_state"],
                    "current_activity": context.get("npc_activity", "idle"),
                    "relationship_context": {
                        "status": context["relationship_status"],
                        "points": context["relationship_points"],
                        "recent_gifts": context.get("recent_gifts", []),
                        "heart_level": context.get("heart_level", 0)
                    }
                }],
                "player": {
                    "name": context["player_name"],
                    "location": context["location"]
                },
                "world_state": {
                    "time": context["hour"],
                    "day_of_week": context["day_of_week"],
                    "season": context["season"],
                    "weather": context["weather"],
                    "festival_active": context.get("festival", None),
                    "other_npcs_present": context.get("other_npcs", [])
                },
                "memory_context": {
                    "last_conversation": context.get("last_conversation_day", 0),
                    "conversation_count": context.get("conversation_count", 0),
                    "shared_memories": context.get("shared_memories", [])
                }
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": npc_id,
                "raw_input": player_action
            }
        }

        response = await self.client.post(
            f"{self.api_url}/elle/game/action",
            json=request,
            timeout=30.0
        )

        return response.json()

    async def get_gift_reaction(
        self,
        npc_id: str,
        item_name: str,
        item_preference: str,
        relationship_points: int
    ) -> str:
        """Get personalized gift reaction."""

        request = {
            "game_state": {
                "scene_id": "gift_giving",
                "npcs": [{
                    "id": npc_id,
                    "relationship_points": relationship_points
                }]
            },
            "player_intent": {
                "type": "give_gift",
                "target_npc_id": npc_id,
                "raw_input": f"I give you {item_name}",
                "context": {
                    "item_preference": item_preference
                }
            }
        }

        response = await self.client.post(
            f"{self.api_url}/elle/game/action",
            json=request,
            timeout=30.0
        )

        action = response.json()
        return action.get("content", {}).get("npc_dialogue", "Thanks...")
```

### Best Practices for Social Sims

1. **Rich Context**: Pass as much context as possible to BigPlay (relationships, memories, activities)
2. **Emotional Persistence**: Track emotional states over time
3. **Memory System**: Use BigPlay's session management to persist NPC memories
4. **Dialogue Variation**: Same greeting should vary based on relationship/context
5. **Emergent Behavior**: Let NPC relationships with each other create stories

---

## Tutorial 3: Multiplayer Integration

**Duration**: 4-5 hours
**Difficulty**: Expert
**What You'll Build**: Shared NPC interactions in a multiplayer world

### Overview

We'll build multiplayer support for "Valley of Friends" where:
- Multiple players share the same world and NPCs
- NPC relationships are player-specific
- NPCs remember and reference interactions with all players
- Shared events (festivals, cutscenes)
- Player-to-player interactions mediated by NPCs

### Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  MULTIPLAYER ARCHITECTURE                 │
│                                                           │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│   │ Player 1 │  │ Player 2 │  │ Player 3 │              │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│        │             │             │                      │
│        └─────────────┼─────────────┘                      │
│                      │                                    │
│              ┌───────▼────────┐                           │
│              │  Game Server   │                           │
│              │  (FastAPI +    │                           │
│              │   WebSocket)   │                           │
│              └───────┬────────┘                           │
│                      │                                    │
│         ┌────────────┴──────────────┐                     │
│         │                           │                     │
│         ▼                           ▼                     │
│  ┌──────────────┐           ┌──────────────┐             │
│  │ Shared World │           │  BigPlay API │             │
│  │   State      │           │  (NPC Brain) │             │
│  └──────────────┘           └──────────────┘             │
│         │                           │                     │
│         │    ┌─────────────┐        │                     │
│         └───→│ HoloLoom KG │←───────┘                     │
│              │  (Memories) │                              │
│              └─────────────┘                              │
└───────────────────────────────────────────────────────────┘

Key Challenges:
- State synchronization across players
- NPC remembering multiple player relationships
- Conflict resolution (two players talk to same NPC)
- Event coordination (festivals with all players)
```

### Step 1: Server Architecture

```python
# multiplayer/server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import asyncio
import json
import httpx

app = FastAPI()

class ConnectionManager:
    """Manages WebSocket connections for all players."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.player_locations: Dict[str, str] = {}

    async def connect(self, player_id: str, websocket: WebSocket):
        """Register new player connection."""
        await websocket.accept()
        self.active_connections[player_id] = websocket
        print(f"Player {player_id} connected")

    def disconnect(self, player_id: str):
        """Remove player connection."""
        if player_id in self.active_connections:
            del self.active_connections[player_id]
        if player_id in self.player_locations:
            del self.player_locations[player_id]
        print(f"Player {player_id} disconnected")

    async def send_personal_message(self, message: Dict, player_id: str):
        """Send message to specific player."""
        if player_id in self.active_connections:
            await self.active_connections[player_id].send_json(message)

    async def broadcast(self, message: Dict, exclude: Set[str] = None):
        """Broadcast message to all players (optionally excluding some)."""
        exclude = exclude or set()
        for player_id, connection in self.active_connections.items():
            if player_id not in exclude:
                await connection.send_json(message)

    async def broadcast_to_location(self, location: str, message: Dict):
        """Broadcast to all players in a specific location."""
        for player_id, player_loc in self.player_locations.items():
            if player_loc == location:
                await self.send_personal_message(message, player_id)

    def get_players_at_location(self, location: str) -> List[str]:
        """Get all players at a location."""
        return [
            pid for pid, loc in self.player_locations.items()
            if loc == location
        ]

manager = ConnectionManager()

class SharedWorldState:
    """Manages shared game state across all players."""

    def __init__(self):
        self.npcs: Dict[str, Dict] = {}
        self.world_state: Dict = {}
        self.active_conversations: Dict[str, str] = {}  # {npc_id: player_id}
        self.lock = asyncio.Lock()

    async def start_conversation(self, npc_id: str, player_id: str) -> bool:
        """
        Try to start conversation with NPC.

        Returns:
            True if conversation started, False if NPC busy
        """
        async with self.lock:
            if npc_id in self.active_conversations:
                # NPC is busy with another player
                return False

            self.active_conversations[npc_id] = player_id
            return True

    async def end_conversation(self, npc_id: str, player_id: str):
        """End conversation with NPC."""
        async with self.lock:
            if self.active_conversations.get(npc_id) == player_id:
                del self.active_conversations[npc_id]

    def is_npc_busy(self, npc_id: str) -> bool:
        """Check if NPC is in conversation."""
        return npc_id in self.active_conversations

world_state = SharedWorldState()

@app.websocket("/ws/{player_id}")
async def websocket_endpoint(websocket: WebSocket, player_id: str):
    """WebSocket endpoint for player connections."""

    await manager.connect(player_id, websocket)

    try:
        while True:
            # Receive action from player
            data = await websocket.receive_json()

            action_type = data.get("type")

            if action_type == "move":
                # Player moved to new location
                location = data["location"]
                manager.player_locations[player_id] = location

                # Notify other players in location
                await manager.broadcast_to_location(
                    location,
                    {
                        "type": "player_entered",
                        "player_id": player_id,
                        "player_name": data.get("player_name", player_id)
                    }
                )

            elif action_type == "talk_to_npc":
                # Player wants to talk to NPC
                npc_id = data["npc_id"]

                # Check if NPC is available
                if world_state.is_npc_busy(npc_id):
                    await manager.send_personal_message(
                        {
                            "type": "npc_busy",
                            "npc_id": npc_id,
                            "message": f"{npc_id} is talking to someone else right now."
                        },
                        player_id
                    )
                    continue

                # Start conversation
                can_talk = await world_state.start_conversation(npc_id, player_id)

                if not can_talk:
                    await manager.send_personal_message(
                        {
                            "type": "npc_busy",
                            "npc_id": npc_id
                        },
                        player_id
                    )
                    continue

                # Process through BigPlay
                response = await process_npc_interaction(
                    player_id=player_id,
                    npc_id=npc_id,
                    player_input=data["message"],
                    context=data.get("context", {})
                )

                # Send response to player
                await manager.send_personal_message(
                    {
                        "type": "npc_dialogue",
                        "npc_id": npc_id,
                        "dialogue": response["dialogue"],
                        "emotional_state": response.get("emotional_state")
                    },
                    player_id
                )

                # Notify other players in location that NPC is talking
                location = manager.player_locations.get(player_id)
                if location:
                    other_players = [
                        p for p in manager.get_players_at_location(location)
                        if p != player_id
                    ]

                    for other_player in other_players:
                        await manager.send_personal_message(
                            {
                                "type": "npc_talking",
                                "npc_id": npc_id,
                                "with_player": player_id,
                                "message": f"{npc_id} is talking to {player_id}"
                            },
                            other_player
                        )

            elif action_type == "end_conversation":
                npc_id = data["npc_id"]
                await world_state.end_conversation(npc_id, player_id)

    except WebSocketDisconnect:
        manager.disconnect(player_id)

        # Notify other players
        await manager.broadcast({
            "type": "player_disconnected",
            "player_id": player_id
        })

async def process_npc_interaction(
    player_id: str,
    npc_id: str,
    player_input: str,
    context: Dict
) -> Dict:
    """
    Process NPC interaction through BigPlay with multi-player context.
    """

    # Get player-specific relationship
    # (Would query database for this player's relationship with NPC)
    player_relationship = get_player_relationship(player_id, npc_id)

    # Get NPC's memories of this specific player
    # (Stored in HoloLoom with player_id in session)
    session_id = f"{player_id}_{npc_id}"

    # Build request with player-specific context
    request = {
        "game_state": {
            "scene_id": context.get("location", "unknown"),
            "npcs": [{
                "id": npc_id,
                "session_id": session_id,  # Player-specific memory
                "emotional_state": get_npc_emotional_state(npc_id, player_id),
                "relationship_context": {
                    "points": player_relationship["points"],
                    "status": player_relationship["status"]
                }
            }],
            "player": {
                "id": player_id,
                "name": context.get("player_name", player_id)
            },
            "world_state": {
                "other_players_present": context.get("other_players", [])
            }
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": npc_id,
            "raw_input": player_input
        }
    }

    # Call BigPlay
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/elle/game/action",
            json=request,
            timeout=30.0
        )

    action = response.json()

    # Update player-specific NPC state
    if "updated_npcs" in action:
        for updated_npc in action["updated_npcs"]:
            if updated_npc["id"] == npc_id:
                update_npc_state(npc_id, player_id, updated_npc)

    return {
        "dialogue": action.get("content", {}).get("npc_dialogue", "..."),
        "emotional_state": action.get("updated_npcs", [{}])[0].get("emotional_state")
    }

def get_player_relationship(player_id: str, npc_id: str) -> Dict:
    """Get relationship between specific player and NPC."""
    # Would query database
    return {"points": 0, "status": "stranger"}

def get_npc_emotional_state(npc_id: str, player_id: str) -> Dict:
    """Get NPC's emotional state when interacting with this player."""
    # Would query database for player-specific emotional state
    return {
        "valence": 0.0,
        "arousal": 0.5,
        "dominance": 0.5,
        "trust": 0.5
    }

def update_npc_state(npc_id: str, player_id: str, updated_state: Dict):
    """Update NPC state for specific player."""
    # Would save to database
    pass
```

### Step 2: Client Integration

```python
# multiplayer/client.py
import asyncio
import websockets
import json
from typing import Callable, Dict

class MultiplayerClient:
    """Client for connecting to multiplayer game server."""

    def __init__(self, player_id: str, server_url: str = "ws://localhost:8000"):
        self.player_id = player_id
        self.server_url = f"{server_url}/ws/{player_id}"
        self.websocket = None
        self.message_handlers: Dict[str, Callable] = {}

    async def connect(self):
        """Connect to game server."""
        self.websocket = await websockets.connect(self.server_url)
        print(f"Connected as {self.player_id}")

    async def disconnect(self):
        """Disconnect from server."""
        if self.websocket:
            await self.websocket.close()

    def on(self, event_type: str, handler: Callable):
        """Register event handler."""
        self.message_handlers[event_type] = handler

    async def send_action(self, action: Dict):
        """Send action to server."""
        if self.websocket:
            await self.websocket.send(json.dumps(action))

    async def move_to(self, location: str, player_name: str):
        """Move to new location."""
        await self.send_action({
            "type": "move",
            "location": location,
            "player_name": player_name
        })

    async def talk_to_npc(
        self,
        npc_id: str,
        message: str,
        context: Dict = None
    ):
        """Talk to NPC."""
        await self.send_action({
            "type": "talk_to_npc",
            "npc_id": npc_id,
            "message": message,
            "context": context or {}
        })

    async def listen(self):
        """Listen for messages from server."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                event_type = data.get("type")

                # Call registered handler
                if event_type in self.message_handlers:
                    await self.message_handlers[event_type](data)
        except websockets.exceptions.ConnectionClosed:
            print("Disconnected from server")

# Example usage
async def example_multiplayer_game():
    """Example multiplayer game client."""

    client = MultiplayerClient(player_id="player_123")

    # Register event handlers
    @client.on("npc_dialogue")
    async def on_npc_dialogue(data):
        npc_id = data["npc_id"]
        dialogue = data["dialogue"]
        print(f"\n{npc_id}: {dialogue}")

    @client.on("player_entered")
    async def on_player_entered(data):
        player_name = data["player_name"]
        print(f"\n[{player_name} entered the area]")

    @client.on("npc_busy")
    async def on_npc_busy(data):
        npc_id = data["npc_id"]
        print(f"\n{npc_id} is busy talking to someone else.")

    # Connect
    await client.connect()

    # Start listening for messages
    listen_task = asyncio.create_task(client.listen())

    # Game loop
    while True:
        # Move to location
        await client.move_to("town_square", "Hero")

        # Talk to NPC
        print("\n> talk to alex")
        await client.talk_to_npc("alex", "Hello!", {"location": "town_square"})

        await asyncio.sleep(2)

    # Cleanup
    listen_task.cancel()
    await client.disconnect()
```

### Best Practices for Multiplayer

1. **Separate Player Context**: Each player has own relationship/memory with NPCs
2. **Session IDs**: Use `{player_id}_{npc_id}` for HoloLoom sessions
3. **Conflict Resolution**: Lock NPCs during conversations
4. **State Sync**: Broadcast relevant events to nearby players
5. **Scalability**: Consider sharding by location/region for large player counts

---

## Tutorial 4: Custom Emotion Systems

**Duration**: 2-3 hours
**Difficulty**: Intermediate
**What You'll Build**: Custom emotion model beyond PAD (Pleasure-Arousal-Dominance)

### Overview

While BigPlay uses the PAD emotion model by default, you can extend it with custom dimensions for your game's unique needs.

Examples:
- **Honor/Shame** for samurai games
- **Corruption/Purity** for moral choice games
- **Madness** for Lovecraftian horror
- **Loyalty** for political intrigue games

### Step 1: Define Custom Emotion Model

```python
# custom_emotions/model.py
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

class CustomEmotionDimension(Enum):
    """Custom emotion dimensions beyond PAD."""
    HONOR = "honor"              # -1.0 (shameful) to +1.0 (honorable)
    CORRUPTION = "corruption"    # 0.0 (pure) to 1.0 (corrupted)
    MADNESS = "madness"          # 0.0 (sane) to 1.0 (insane)
    LOYALTY = "loyalty"          # -1.0 (traitor) to +1.0 (loyal)
    GREED = "greed"              # 0.0 (generous) to 1.0 (greedy)
    COURAGE = "courage"          # 0.0 (cowardly) to 1.0 (brave)

@dataclass
class ExtendedEmotionalState:
    """
    Extended emotional state with PAD + custom dimensions.
    """

    # Standard PAD model
    valence: float = 0.0       # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.5       # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5     # 0.0 (submissive) to 1.0 (dominant)
    trust: float = 0.5         # 0.0 (distrust) to 1.0 (trust)

    # Custom dimensions
    custom: Dict[str, float] = field(default_factory=dict)

    def set_dimension(self, dimension: str, value: float):
        """Set custom dimension value."""
        # Clamp to valid range
        if dimension in ["valence", "loyalty"]:
            # Bipolar dimensions (-1 to +1)
            value = max(-1.0, min(1.0, value))
        else:
            # Unipolar dimensions (0 to 1)
            value = max(0.0, min(1.0, value))

        if dimension in ["valence", "arousal", "dominance", "trust"]:
            setattr(self, dimension, value)
        else:
            self.custom[dimension] = value

    def get_dimension(self, dimension: str) -> float:
        """Get dimension value."""
        if dimension in ["valence", "arousal", "dominance", "trust"]:
            return getattr(self, dimension)
        return self.custom.get(dimension, 0.0)

    def decay_custom_dimensions(self, decay_rate: float = 0.01):
        """
        Decay custom dimensions toward neutral over time.

        Useful for temporary emotional states like anger or excitement.
        """
        for dimension, value in list(self.custom.items()):
            if dimension in ["corruption", "madness", "greed"]:
                # These typically don't decay (permanent character changes)
                continue

            # Decay toward 0.5 (neutral)
            if value > 0.5:
                self.custom[dimension] = max(0.5, value - decay_rate)
            elif value < 0.5:
                self.custom[dimension] = min(0.5, value + decay_rate)

class EmotionTrigger:
    """Maps game events to emotion changes."""

    def __init__(self):
        # Define emotion change mappings
        self.triggers = {
            # Honor/Shame system (samurai game)
            "honor": {
                "won_duel_honorably": {"honor": +0.3, "valence": +0.2},
                "won_duel_dishonorably": {"honor": -0.4, "valence": +0.1, "trust": -0.2},
                "refused_duel": {"honor": -0.2, "courage": -0.3},
                "protected_innocent": {"honor": +0.2, "valence": +0.3},
                "abandoned_ally": {"honor": -0.5, "loyalty": -0.4, "trust": -0.3}
            },

            # Corruption system (moral choice game)
            "corruption": {
                "accepted_bribe": {"corruption": +0.2, "greed": +0.3},
                "resisted_temptation": {"corruption": -0.1, "valence": +0.1},
                "used_dark_magic": {"corruption": +0.4, "madness": +0.1},
                "performed_sacrifice": {"corruption": +0.6, "valence": -0.3}
            },

            # Madness system (Lovecraftian horror)
            "madness": {
                "witnessed_horror": {"madness": +0.3, "arousal": +0.5, "valence": -0.4},
                "read_forbidden_tome": {"madness": +0.2},
                "solved_mystery": {"madness": +0.1, "valence": +0.3},
                "meditated": {"madness": -0.05, "arousal": -0.2}
            },

            # Loyalty system (political intrigue)
            "loyalty": {
                "supported_leader": {"loyalty": +0.3, "trust": +0.2},
                "betrayed_leader": {"loyalty": -0.8, "honor": -0.5},
                "questioned_orders": {"loyalty": -0.1},
                "defended_leader": {"loyalty": +0.4, "courage": +0.3}
            }
        }

    def apply_trigger(
        self,
        emotion_state: ExtendedEmotionalState,
        category: str,
        event: str
    ) -> Dict[str, float]:
        """
        Apply emotion trigger to state.

        Returns:
            Dictionary of dimension changes
        """
        if category not in self.triggers:
            return {}

        changes = self.triggers[category].get(event, {})

        for dimension, delta in changes.items():
            current = emotion_state.get_dimension(dimension)
            emotion_state.set_dimension(dimension, current + delta)

        return changes

# Example usage
emotion_state = ExtendedEmotionalState()
trigger = EmotionTrigger()

# Player wins duel dishonorably
changes = trigger.apply_trigger(
    emotion_state,
    "honor",
    "won_duel_dishonorably"
)

print(f"Honor: {emotion_state.get_dimension('honor')}")  # -0.4
print(f"Valence: {emotion_state.valence}")                # +0.1
print(f"Trust: {emotion_state.trust}")                    # 0.3
```

### Step 2: Integrate with BigPlay

```python
# custom_emotions/bigplay_integration.py
import httpx
from typing import Dict

async def send_with_custom_emotions(
    npc_id: str,
    npc_state: Dict,
    player_action: str,
    emotion_state: ExtendedEmotionalState
) -> Dict:
    """
    Send request to BigPlay with custom emotion dimensions.

    BigPlay will use custom dimensions to influence dialogue generation.
    """

    # Build enhanced emotional state
    emotional_context = {
        # Standard PAD
        "valence": emotion_state.valence,
        "arousal": emotion_state.arousal,
        "dominance": emotion_state.dominance,
        "trust": emotion_state.trust,

        # Custom dimensions (passed as metadata)
        "custom_dimensions": emotion_state.custom,

        # Derive discrete emotion from dimensions
        "primary_emotion": derive_discrete_emotion(emotion_state)
    }

    request = {
        "game_state": {
            "scene_id": "current_location",
            "npcs": [{
                "id": npc_id,
                "name": npc_state["name"],
                "role": npc_state["role"],
                "emotional_state": emotional_context
            }]
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": npc_id,
            "raw_input": player_action
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/elle/game/action",
            json=request,
            timeout=30.0
        )

    return response.json()

def derive_discrete_emotion(emotion_state: ExtendedEmotionalState) -> str:
    """
    Map dimensional emotions to discrete emotion labels.

    Useful for generating appropriate dialogue hints.
    """

    # Get primary dimensions
    v = emotion_state.valence
    a = emotion_state.arousal
    d = emotion_state.dominance

    # Get custom dimensions
    honor = emotion_state.get_dimension("honor")
    madness = emotion_state.get_dimension("madness")
    corruption = emotion_state.get_dimension("corruption")

    # Custom emotion mappings
    if madness > 0.7:
        return "insane"
    elif madness > 0.4:
        return "unhinged"

    if corruption > 0.7:
        return "corrupted"
    elif corruption > 0.4:
        return "tempted"

    if honor < -0.5:
        return "ashamed"
    elif honor > 0.7:
        return "honorable"

    # Fall back to standard PAD mapping
    if v > 0.5 and a > 0.5:
        return "excited" if d > 0.5 else "happy"
    elif v < -0.5 and a > 0.5:
        return "angry" if d > 0.5 else "fearful"
    elif v < -0.5 and a < 0.5:
        return "sad" if d < 0.5 else "disgusted"
    elif v > 0.5 and a < 0.5:
        return "content" if d > 0.5 else "relaxed"
    else:
        return "neutral"
```

### Step 3: Visualizing Custom Emotions

```python
# custom_emotions/visualization.py
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def visualize_emotion_profile(
    emotion_state: ExtendedEmotionalState,
    title: str = "NPC Emotion Profile"
):
    """
    Create radar chart showing all emotion dimensions.
    """

    # Collect all dimensions
    dimensions = {
        "Valence": (emotion_state.valence + 1) / 2,  # Normalize to 0-1
        "Arousal": emotion_state.arousal,
        "Dominance": emotion_state.dominance,
        "Trust": emotion_state.trust
    }

    # Add custom dimensions
    for dim, value in emotion_state.custom.items():
        # Normalize bipolar dimensions
        if dim in ["honor", "loyalty"]:
            value = (value + 1) / 2
        dimensions[dim.capitalize()] = value

    # Create radar chart
    labels = list(dimensions.keys())
    values = list(dimensions.values())

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]  # Complete the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, values, 'o-', linewidth=2, label=title)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.grid(True)

    plt.title(title, size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def plot_emotion_trajectory(
    history: List[ExtendedEmotionalState],
    dimension: str,
    title: str = "Emotion Over Time"
):
    """
    Plot how a specific emotion dimension changes over time.
    """

    values = [state.get_dimension(dimension) for state in history]

    plt.figure(figsize=(12, 6))
    plt.plot(values, marker='o', linewidth=2)
    plt.xlabel("Time Steps")
    plt.ylabel(dimension.capitalize())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral')
    if dimension in ["honor", "loyalty", "valence"]:
        plt.ylim(-1, 1)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    else:
        plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
```

### Best Practices

1. **Meaningful Dimensions**: Choose dimensions that matter for your game's themes
2. **Decay Rates**: Consider which emotions persist vs. decay over time
3. **Visualization**: Show players their emotional state through UI
4. **Consequences**: Make emotion dimensions affect gameplay (dialogue options, quest availability)
5. **Balance**: Don't overwhelm with too many dimensions (3-5 custom max)

---

## Tutorial 5: Fine-Tuning with Local LLMs

**Duration**: 4-6 hours
**Difficulty**: Expert
**What You'll Build**: Custom fine-tuned NPC personalities using local LLMs

### Overview

While BigPlay works with API-based LLMs (Claude, GPT-4), you can fine-tune local models for:
- Cost savings (no API fees)
- Privacy (data stays local)
- Custom NPC personalities
- Faster iteration

We'll use **Ollama** with **Llama 3.2** for fine-tuning.

### Step 1: Collect Training Data

```python
# fine_tuning/data_collection.py
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class TrainingExample:
    """Single training example for NPC dialogue."""

    npc_id: str
    npc_role: str
    personality_traits: List[str]
    context: str
    player_input: str
    npc_response: str
    emotional_state: Dict[str, float]

class TrainingDataCollector:
    """Collect training data from gameplay sessions."""

    def __init__(self, output_file: str = "training_data.jsonl"):
        self.output_file = output_file
        self.examples: List[TrainingExample] = []

    def add_example(
        self,
        npc_id: str,
        npc_role: str,
        personality: List[str],
        context: str,
        player_input: str,
        npc_response: str,
        emotions: Dict[str, float]
    ):
        """Add training example."""
        example = TrainingExample(
            npc_id=npc_id,
            npc_role=npc_role,
            personality_traits=personality,
            context=context,
            player_input=player_input,
            npc_response=npc_response,
            emotional_state=emotions
        )

        self.examples.append(example)

    def save(self):
        """Save training data to JSONL file."""
        with open(self.output_file, 'w') as f:
            for example in self.examples:
                # Convert to Llama 3.2 instruction format
                prompt = self._format_prompt(example)
                f.write(json.dumps({
                    "instruction": prompt,
                    "output": example.npc_response
                }) + '\n')

    def _format_prompt(self, example: TrainingExample) -> str:
        """Format example as instruction prompt."""

        traits = ", ".join(example.personality_traits)

        prompt = f"""You are {example.npc_id}, a {example.npc_role}.

Your personality: {traits}

Current emotional state:
- Valence: {example.emotional_state.get('valence', 0):.2f}
- Arousal: {example.emotional_state.get('arousal', 0.5):.2f}
- Trust: {example.emotional_state.get('trust', 0.5):.2f}

Context: {example.context}

Player says: "{example.player_input}"

Respond in character:"""

        return prompt

# Example usage
collector = TrainingDataCollector()

# Collect examples from gameplay
collector.add_example(
    npc_id="grumpy_blacksmith",
    npc_role="blacksmith",
    personality=["grumpy", "skilled", "honest", "impatient"],
    context="The blacksmith is working on a sword. It's early morning.",
    player_input="Good morning! Can you fix my sword?",
    npc_response="Hmph. *doesn't look up* Leave it on the counter. Come back this afternoon. And don't bother me while I'm working.",
    emotions={"valence": -0.2, "arousal": 0.6, "trust": 0.3}
)

collector.add_example(
    npc_id="grumpy_blacksmith",
    npc_role="blacksmith",
    personality=["grumpy", "skilled", "honest", "impatient"],
    context="Player has visited multiple times. Blacksmith is warming up to them.",
    player_input="How's the sword coming along?",
    npc_response="*grunts* Better than expected. You've got a good blade here - just needed proper care. Should be ready in an hour. *almost smiles* You actually take care of your equipment, unlike most adventurers.",
    emotions={"valence": 0.1, "arousal": 0.5, "trust": 0.6}
)

# Save training data
collector.save()
```

### Step 2: Fine-Tune Llama Model

```python
# fine_tuning/finetune_llama.py
import subprocess
import json
from pathlib import Path

class LlamaFineTuner:
    """Fine-tune Llama 3.2 model using Ollama."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.base_model = model_name
        self.training_data = "training_data.jsonl"

    def create_modelfile(self, custom_name: str, system_prompt: str) -> str:
        """
        Create Ollama Modelfile for fine-tuned model.

        Modelfile defines model parameters and system prompt.
        """

        modelfile_content = f"""FROM {self.base_model}

# System prompt
SYSTEM \"\"\"
{system_prompt}
\"\"\"

# Parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# Stop sequences
PARAMETER stop "<|endoftext|>"
PARAMETER stop "</s>"
"""

        modelfile_path = f"{custom_name}.modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        return modelfile_path

    def create_custom_model(
        self,
        custom_name: str,
        personality: str,
        traits: List[str]
    ):
        """
        Create custom NPC model.

        Args:
            custom_name: Name for the custom model (e.g., "grumpy_blacksmith")
            personality: Personality description
            traits: List of personality traits
        """

        # Build system prompt
        traits_str = ", ".join(traits)
        system_prompt = f"""You are an NPC in a fantasy RPG game.

Personality: {personality}
Traits: {traits_str}

Guidelines:
- Stay in character at all times
- Respond based on your emotional state
- Remember previous interactions with the player
- Use natural dialogue (contractions, pauses, non-verbal cues)
- Don't break the fourth wall
- Keep responses concise (1-3 sentences)

Example format:
Player: "Hello!"
You: "*looks up from work* What do you want? I'm busy."
"""

        # Create Modelfile
        modelfile = self.create_modelfile(custom_name, system_prompt)

        # Create model using Ollama CLI
        print(f"Creating custom model: {custom_name}")
        subprocess.run([
            "ollama",
            "create",
            custom_name,
            "-f",
            modelfile
        ])

        print(f"✓ Created {custom_name}")

    def test_model(self, model_name: str, test_prompts: List[str]):
        """Test the fine-tuned model."""

        print(f"\n=== Testing {model_name} ===\n")

        for prompt in test_prompts:
            print(f"Player: {prompt}")

            # Run model via Ollama CLI
            result = subprocess.run(
                ["ollama", "run", model_name, prompt],
                capture_output=True,
                text=True
            )

            response = result.stdout.strip()
            print(f"NPC: {response}\n")

# Example usage
finetuner = LlamaFineTuner()

# Create custom grumpy blacksmith model
finetuner.create_custom_model(
    custom_name="grumpy_blacksmith_v1",
    personality="A skilled but impatient blacksmith who warms up to regulars",
    traits=["grumpy", "skilled", "honest", "impatient", "secretly kind"]
)

# Test the model
test_prompts = [
    "Good morning! Can you fix my sword?",
    "Thanks for fixing it! You do great work!",
    "Why are you always so grumpy?"
]

finetuner.test_model("grumpy_blacksmith_v1", test_prompts)
```

### Step 3: Integrate with BigPlay

```python
# fine_tuning/bigplay_custom_llm.py
from app.models import LLMProvider, GamePolicy

class CustomLLMProvider(LLMProvider):
    """Custom LLM provider using fine-tuned Ollama models."""

    def __init__(self):
        self.model_mapping = {
            "grumpy_blacksmith": "grumpy_blacksmith_v1",
            "cheerful_innkeeper": "cheerful_innkeeper_v1",
            "mysterious_wizard": "mysterious_wizard_v1",
            # Map NPC IDs to their custom models
        }
        self.default_model = "llama3.2:3b"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        npc_id: str = None,
        **kwargs
    ) -> str:
        """Generate response using custom model for NPC."""

        # Select model based on NPC
        model = self.model_mapping.get(npc_id, self.default_model)

        # Format messages for Ollama
        prompt = self._format_messages(messages)

        # Call Ollama API
        response = await self._call_ollama(model, prompt)

        return response

    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Call Ollama API."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30.0
            )

        data = response.json()
        return data["response"]

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama prompt."""
        formatted = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"Player: {content}")
            elif role == "assistant":
                formatted.append(f"NPC: {content}")

        return "\n".join(formatted)

# Update BigPlay to use custom provider
# In app/main.py:

from fine_tuning.bigplay_custom_llm import CustomLLMProvider

# During startup
@app.on_event("startup")
async def startup_event():
    # Use custom LLM provider
    game_policy.llm_provider = CustomLLMProvider()
```

### Best Practices

1. **Quality Data**: Collect 50-100 high-quality examples per NPC
2. **Consistency**: Keep personality consistent across examples
3. **Diversity**: Vary contexts and player inputs
4. **Testing**: Test extensively before deploying
5. **Versioning**: Keep model versions (v1, v2, etc.) for rollback

---

## Tutorial 6: Voice-First Game Design

**Duration**: 2-3 hours
**Difficulty**: Intermediate
**What You'll Build**: Voice-controlled text adventure with TTS responses

### Overview

Voice-first games create immersive experiences where players speak naturally instead of typing. We'll build a hands-free adventure game.

### Step 1: Speech Recognition Setup

```python
# voice_game/speech_recognition.py
import speech_recognition as sr
from typing import Optional, Callable
import asyncio

class VoiceInput:
    """Handles speech-to-text for player input."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            print("Calibrating microphone...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Ready for voice input")

    def listen(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for player voice input.

        Args:
            timeout: Seconds to wait for speech

        Returns:
            Transcribed text or None
        """
        try:
            with self.microphone as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)

            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            print(f"You said: \"{text}\"")
            return text

        except sr.WaitTimeoutError:
            print("⏱️  No speech detected")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Error: {e}")
            return None

    async def listen_continuous(
        self,
        callback: Callable[[str], None],
        wake_word: str = "hello game"
    ):
        """
        Continuously listen for wake word, then process command.

        Args:
            callback: Function to call with recognized text
            wake_word: Phrase to activate listening
        """
        print(f"Say '{wake_word}' to start...")

        while True:
            text = self.listen(timeout=10)

            if text and wake_word.lower() in text.lower():
                print("🎮 Activated! Say your command:")

                command = self.listen(timeout=5)
                if command:
                    await callback(command)

            await asyncio.sleep(0.1)
```

### Step 2: Voice-Controlled Game Loop

```python
# voice_game/game.py
import asyncio
from .speech_recognition import VoiceInput
from lost_kingdom.game_manager import LostKingdomGame  # From Tutorial 1
import pyttsx3  # Text-to-speech

class VoiceControlledGame:
    """Voice-controlled version of The Lost Kingdom."""

    def __init__(self):
        self.game = LostKingdomGame()
        self.voice_input = VoiceInput()

        # Text-to-speech engine
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)  # Speed
        self.tts.setProperty('volume', 0.9)  # Volume

    def speak(self, text: str):
        """Speak text aloud."""
        print(f"\n🔊 {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    async def process_voice_command(self, command: str):
        """Process player voice command."""

        # Special commands
        if "quit" in command.lower() or "exit" in command.lower():
            self.speak("Goodbye adventurer!")
            return "quit"

        if "inventory" in command.lower():
            items = [item.name for item in self.game.game_state.player.inventory]
            if items:
                self.speak(f"You have: {', '.join(items)}")
            else:
                self.speak("Your inventory is empty")
            return

        if "health" in command.lower() or "status" in command.lower():
            player = self.game.game_state.player
            self.speak(
                f"Health: {player.health} out of {player.max_health}. "
                f"Level {player.level}. "
                f"You have {player.gold} gold."
            )
            return

        # Process as game action
        action = await self.game.process_action(command)

        # Speak response
        if action["action_type"] == "dialogue":
            npc_name = action["content"].get("npc_name", "Someone")
            dialogue = action["content"]["npc_dialogue"]
            self.speak(f"{npc_name} says: {dialogue}")

        elif action["action_type"] == "world_description":
            description = action["content"]["description"]
            self.speak(description)

        elif action["action_type"] == "hint":
            hint = action["content"]["hint_text"]
            self.speak(f"Hint: {hint}")

    async def run(self):
        """Run voice-controlled game loop."""

        # Intro
        self.speak("Welcome to The Lost Kingdom. A voice-controlled adventure.")
        self.speak("You can say things like: talk to the innkeeper, go north, look around, or check inventory.")
        self.speak("Say 'hello game' followed by your command.")

        # Continuous listening
        await self.voice_input.listen_continuous(
            callback=self.process_voice_command,
            wake_word="hello game"
        )

# Run the game
if __name__ == "__main__":
    game = VoiceControlledGame()
    asyncio.run(game.run())
```

### Step 3: Advanced Voice Features

```python
# voice_game/advanced_features.py
from typing import List, Dict
import re

class VoiceCommandParser:
    """Parse natural language voice commands."""

    def __init__(self):
        self.command_patterns = {
            "move": [
                r"go (to |to the )?(.*)",
                r"(walk|move|travel) (to |to the )?(.*)",
                r"(north|south|east|west)"
            ],
            "talk": [
                r"talk to (the )?(.*)",
                r"speak (to |with )?(the )?(.*)",
                r"ask (the )?(.*) about (.*)"
            ],
            "take": [
                r"(take|grab|get|pick up) (the )?(.*)",
                r"(.*) from (the )?(.*)"
            ],
            "use": [
                r"use (the )?(.*) (on )?(.*)?",
                r"drink (the )?(.*)",
                r"eat (the )?(.*)"
            ],
            "examine": [
                r"(look at|examine|inspect) (the )?(.*)",
                r"what (is|are) (the )?(.*)",
                r"tell me about (the )?(.*)"
            ]
        }

    def parse(self, voice_input: str) -> Dict[str, any]:
        """
        Parse voice input into structured command.

        Returns:
            {
                "intent": "talk",
                "target": "innkeeper",
                "raw": original input
            }
        """

        voice_input = voice_input.lower().strip()

        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, voice_input)
                if match:
                    groups = [g for g in match.groups() if g and g.strip()]

                    return {
                        "intent": intent,
                        "target": groups[-1] if groups else None,
                        "raw": voice_input,
                        "groups": groups
                    }

        # No pattern matched - return as freeform
        return {
            "intent": "freeform",
            "target": None,
            "raw": voice_input
        }

class VoiceContextManager:
    """Maintains conversation context for natural dialogue."""

    def __init__(self):
        self.current_npc = None
        self.conversation_history: List[str] = []

    def set_talking_to(self, npc_id: str):
        """Set current conversation partner."""
        self.current_npc = npc_id
        self.conversation_history = []

    def add_exchange(self, player_input: str, npc_response: str):
        """Add to conversation history."""
        self.conversation_history.append({
            "player": player_input,
            "npc": npc_response
        })

        # Keep last 5 exchanges
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

    def resolve_pronoun(self, text: str) -> str:
        """
        Resolve pronouns using context.

        Example:
        Player: "Talk to the innkeeper"
        Player: "Ask him about the forest"  # "him" → "innkeeper"
        """

        if not self.current_npc:
            return text

        # Replace pronouns with NPC name
        text = re.sub(r'\b(him|her|them)\b', self.current_npc, text)
        text = re.sub(r'\b(his|her|their)\b', f"{self.current_npc}'s", text)

        return text
```

### Best Practices

1. **Wake Word**: Use a consistent activation phrase
2. **Feedback**: Always confirm what was heard
3. **Error Handling**: Gracefully handle misrecognition
4. **Timeout**: Don't wait indefinitely for speech
5. **Ambient Noise**: Calibrate for environment
6. **TTS Voices**: Choose appropriate voice for NPCs

---

## Tutorial 7: Quest System Deep Dive

**Duration**: 3-4 hours
**Difficulty**: Advanced
**What You'll Build**: Dynamic quest generation with branching narratives

### Overview

BigPlay's quest system generates quests dynamically based on:
- Player level and skills
- NPC relationships
- World state
- Previous quest outcomes

We'll build a complete quest system with tracking, branching, and consequences.

### Step 1: Quest Templates

```python
# quest_system/templates.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

class QuestType(Enum):
    FETCH = "fetch"          # Bring item A to person B
    KILL = "kill"            # Defeat X enemies
    ESCORT = "escort"        # Protect NPC to destination
    DISCOVERY = "discovery"  # Find location/secret
    DIALOGUE = "dialogue"    # Talk to NPCs in specific order
    CHOICE = "choice"        # Moral dilemma with consequences

@dataclass
class QuestObjective:
    """Single quest objective."""
    id: str
    description: str
    objective_type: str  # "talk_to", "collect", "kill", "reach", etc.
    target: str
    target_count: int = 1
    current_count: int = 0
    completed: bool = False
    optional: bool = False

@dataclass
class QuestBranch:
    """Quest branch based on player choice."""
    choice_id: str
    choice_text: str
    next_objectives: List[QuestObjective]
    consequences: Dict[str, any] = field(default_factory=dict)

@dataclass
class QuestTemplate:
    """Template for quest generation."""

    quest_type: QuestType
    title_template: str  # "Retrieve the {item} from {location}"
    description_template: str
    difficulty_range: tuple = (1, 5)  # Min/max difficulty

    # Dynamic parameters
    required_params: List[str] = field(default_factory=list)
    # Example: ["item", "location", "npc_name"]

    # Objectives
    objective_templates: List[Dict] = field(default_factory=list)

    # Branching
    has_branches: bool = False
    branch_points: List[Dict] = field(default_factory=list)

    # Rewards
    base_rewards: Dict[str, any] = field(default_factory=dict)

# Example templates
FETCH_QUEST = QuestTemplate(
    quest_type=QuestType.FETCH,
    title_template="Retrieve the {item}",
    description_template="{npc_name} needs you to retrieve {item} from {location}.",
    difficulty_range=(1, 3),
    required_params=["item", "npc_name", "location"],
    objective_templates=[
        {
            "type": "reach",
            "target": "{location}",
            "description": "Travel to {location}"
        },
        {
            "type": "collect",
            "target": "{item}",
            "count": 1,
            "description": "Retrieve the {item}"
        },
        {
            "type": "return",
            "target": "{npc_name}",
            "description": "Return the {item} to {npc_name}"
        }
    ],
    base_rewards={"experience": 100, "gold": 50}
)

MORAL_CHOICE_QUEST = QuestTemplate(
    quest_type=QuestType.CHOICE,
    title_template="The {npc_name} Dilemma",
    description_template="{npc_name} has asked for your help, but doing so may have consequences...",
    difficulty_range=(3, 5),
    required_params=["npc_name", "dilemma"],
    has_branches=True,
    branch_points=[
        {
            "trigger": "talked_to_npc",
            "choices": [
                {
                    "id": "help",
                    "text": "I'll help you",
                    "objectives": ["help_npc_objective"],
                    "consequences": {"reputation": +10, "morality": -5}
                },
                {
                    "id": "refuse",
                    "text": "I can't help with this",
                    "objectives": ["refuse_objective"],
                    "consequences": {"reputation": -5, "morality": +10}
                },
                {
                    "id": "negotiate",
                    "text": "Perhaps there's another way",
                    "objectives": ["find_alternative"],
                    "consequences": {"reputation": +5, "wisdom": +5}
                }
            ]
        }
    ]
)
```

### Step 2: Quest Generator

```python
# quest_system/generator.py
import random
from typing import List, Dict
import httpx

class DynamicQuestGenerator:
    """Generate quests dynamically using BigPlay."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.templates = [FETCH_QUEST, MORAL_CHOICE_QUEST]  # Add more...

    async def generate_quest(
        self,
        player_level: int,
        world_state: Dict,
        available_npcs: List[str],
        relationship_context: Dict
    ) -> Dict:
        """
        Generate quest using BigPlay LLM.

        Args:
            player_level: Player's current level
            world_state: Current world state
            available_npcs: NPCs that can give quests
            relationship_context: Player's relationships with NPCs

        Returns:
            Complete quest specification
        """

        # Select template based on difficulty
        suitable_templates = [
            t for t in self.templates
            if t.difficulty_range[0] <= player_level <= t.difficulty_range[1]
        ]

        if not suitable_templates:
            suitable_templates = self.templates

        template = random.choice(suitable_templates)

        # Build prompt for BigPlay
        prompt = f"""Generate a {template.quest_type.value} quest for a level {player_level} player.

World State:
{json.dumps(world_state, indent=2)}

Available NPCs: {', '.join(available_npcs)}

Quest Template:
Title: {template.title_template}
Description: {template.description_template}

Required Parameters: {', '.join(template.required_params)}

Provide a JSON response with:
{{
    "title": "Quest title with parameters filled",
    "description": "Full quest description",
    "parameters": {{"param_name": "value"}},
    "objectives": [
        {{"type": "...", "target": "...", "description": "..."}}
    ],
    "difficulty": 1-5,
    "estimated_time": "X minutes",
    "rewards": {{"experience": X, "gold": Y, "items": []}}
}}
"""

        # Call BigPlay quest generation endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/elle/quest/generate",
                json={
                    "difficulty": "medium",  # or map from player_level
                    "context": {
                        "player_level": player_level,
                        "world_state": world_state,
                        "available_npcs": available_npcs,
                        "template": template.quest_type.value,
                        "custom_prompt": prompt
                    }
                },
                timeout=30.0
            )

        quest_data = response.json()

        # Add template-specific branching if applicable
        if template.has_branches:
            quest_data["branches"] = template.branch_points

        return quest_data

class QuestTracker:
    """Track active quest progress."""

    def __init__(self):
        self.active_quests: Dict[str, Dict] = {}
        self.completed_quests: List[str] = []

    def start_quest(self, quest: Dict):
        """Start tracking a quest."""
        quest_id = quest.get("id", str(random.randint(1000, 9999)))
        quest["id"] = quest_id

        # Initialize objective tracking
        for obj in quest.get("objectives", []):
            obj["current_count"] = 0
            obj["completed"] = False

        self.active_quests[quest_id] = quest

    def update_objective(
        self,
        quest_id: str,
        objective_index: int,
        progress: int = 1
    ) -> bool:
        """
        Update objective progress.

        Returns:
            True if objective completed
        """
        if quest_id not in self.active_quests:
            return False

        quest = self.active_quests[quest_id]
        objective = quest["objectives"][objective_index]

        objective["current_count"] += progress

        if objective["current_count"] >= objective.get("target_count", 1):
            objective["completed"] = True
            return True

        return False

    def check_quest_completion(self, quest_id: str) -> bool:
        """Check if all objectives completed."""
        if quest_id not in self.active_quests:
            return False

        quest = self.active_quests[quest_id]
        objectives = quest.get("objectives", [])

        # Check if all non-optional objectives complete
        required_objectives = [
            obj for obj in objectives
            if not obj.get("optional", False)
        ]

        return all(obj.get("completed", False) for obj in required_objectives)

    def complete_quest(self, quest_id: str) -> Dict:
        """Mark quest as complete and return rewards."""
        if quest_id not in self.active_quests:
            return {}

        quest = self.active_quests.pop(quest_id)
        self.completed_quests.append(quest_id)

        # Calculate rewards
        rewards = quest.get("rewards", {})

        # Bonus for completing optional objectives
        optional_completed = sum(
            1 for obj in quest["objectives"]
            if obj.get("optional") and obj.get("completed")
        )

        if optional_completed > 0:
            rewards["experience"] = rewards.get("experience", 0) * (1 + 0.2 * optional_completed)
            rewards["bonus_items"] = ["rare_item"] * optional_completed

        return {
            "quest_id": quest_id,
            "title": quest["title"],
            "rewards": rewards
        }
```

### Step 3: Quest Visualization

```python
# quest_system/ui.py
from typing import Dict, List

def render_quest_log(active_quests: Dict[str, Dict]) -> str:
    """Render quest log as formatted text."""

    if not active_quests:
        return "No active quests."

    output = []
    output.append("═" * 60)
    output.append("                     QUEST LOG")
    output.append("═" * 60)

    for quest_id, quest in active_quests.items():
        output.append(f"\n📜 {quest['title']}")
        output.append(f"   {quest.get('description', '')}")
        output.append(f"   Difficulty: {'⭐' * quest.get('difficulty', 1)}")
        output.append("")

        output.append("   Objectives:")
        for i, obj in enumerate(quest.get("objectives", []), 1):
            status = "✓" if obj.get("completed") else "○"
            optional = " (Optional)" if obj.get("optional") else ""

            if "target_count" in obj and obj["target_count"] > 1:
                progress = f" ({obj.get('current_count', 0)}/{obj['target_count']})"
            else:
                progress = ""

            output.append(f"   {status} {i}. {obj['description']}{progress}{optional}")

        output.append("   ─" * 30)

    output.append("\n" + "═" * 60)

    return "\n".join(output)

def render_quest_complete(quest_id: str, title: str, rewards: Dict) -> str:
    """Render quest completion screen."""

    output = []
    output.append("\n" + "═" * 60)
    output.append("                 QUEST COMPLETE!")
    output.append("═" * 60)
    output.append(f"\n🎉 {title}")
    output.append("\nRewards:")

    if "experience" in rewards:
        output.append(f"   💎 {rewards['experience']} Experience")

    if "gold" in rewards:
        output.append(f"   💰 {rewards['gold']} Gold")

    if "items" in rewards:
        for item in rewards["items"]:
            output.append(f"   📦 {item}")

    if "bonus_items" in rewards:
        output.append("\n   ✨ Bonus Rewards:")
        for item in rewards["bonus_items"]:
            output.append(f"      🌟 {item}")

    output.append("\n" + "═" * 60)

    return "\n".join(output)
```

### Best Practices

1. **Clear Objectives**: Make goals explicit and trackable
2. **Meaningful Rewards**: Scale rewards to difficulty
3. **Player Agency**: Offer choices that matter
4. **Consequences**: Make choices affect world/NPCs
5. **Fail States**: Allow quests to be failed, not just completed
6. **Dynamic Content**: Generate quests based on player actions

---

## Conclusion

You've now learned how to build:

1. **Text Adventure RPGs** - Complete game loops with combat and inventory
2. **Social Simulation Games** - Relationship systems and NPC schedules
3. **Multiplayer Games** - Shared world with WebSocket synchronization
4. **Custom Emotion Systems** - Beyond PAD for unique game themes
5. **Fine-Tuned NPCs** - Local LLMs with custom personalities
6. **Voice-First Games** - Speech recognition and TTS integration
7. **Dynamic Quest Systems** - Branching narratives with consequences

### Next Steps

- Combine tutorials to create hybrid games
- Experiment with different LLM providers
- Add persistence (databases, save files)
- Build custom UI/UX
- Deploy to production (see ARCHITECTURE.md)

### Resources

- **BigPlay Documentation**: Complete API reference
- **Example Games**: The Rusty Mug Tavern demo
- **Community**: Discord, forums, GitHub discussions
- **Support**: hello@bigplay.dev

**Happy Building!** 🎮