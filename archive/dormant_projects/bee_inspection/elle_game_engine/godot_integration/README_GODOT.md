# Elle Game Engine - Godot Integration

**LLM-driven narrative intelligence for Godot games**

Date: 2025-11-16
Version: 0.1.0

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Voice Synthesis](#voice-synthesis)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

Elle Game Engine provides intelligent NPC dialogue, hints, and world reactions powered by large language models. This Godot integration allows you to seamlessly add narrative AI to your Godot games without complex setup.

### What Elle Does

- **Dynamic NPC Dialogue**: NPCs respond intelligently based on game state, mood, and player reputation
- **Context-Aware Hints**: Get non-spoilery guidance that adapts to player progress
- **World Reactions**: Environmental responses that reflect narrative tension and player actions
- **Voice Synthesis**: Text-to-speech with emotion-aware voices (optional)

### Architecture

```
Your Godot Game
    ↓ HTTP/JSON
ElleClient (GDScript)
    ↓ REST API
Elle Service (Python/FastAPI)
    ↓ LLM Calls
Anthropic/OpenAI/Ollama
```

---

## Features

✅ **Zero-Config Setup**: Works out of the box with sensible defaults
✅ **Signal-Based**: Godot-native async callbacks
✅ **Autoload Support**: Global access via singleton
✅ **Retry Logic**: Automatic exponential backoff on failures
✅ **Type-Safe**: Full GDScript type hints
✅ **Cache-Friendly**: Response caching for repeated queries
✅ **Voice Integration**: Optional TTS with multiple backends

---

## Installation

### Option 1: Manual Installation

1. **Copy addon files** to your Godot project:
   ```bash
   cp -r addons/elle_game_engine/ YOUR_GODOT_PROJECT/addons/
   ```

2. **Enable plugin** in Godot:
   - Project → Project Settings → Plugins
   - Enable "Elle Game Engine"

3. **Configure Autoload** (if not using plugin):
   - Project → Project Settings → Autoload
   - Add `res://addons/elle_game_engine/ElleClient.gd` as "Elle"

### Option 2: Asset Library (Coming Soon)

Elle will be available on the Godot Asset Library for one-click installation.

### Prerequisites

- **Godot 4.0+** (GDScript 2.0)
- **Elle Service** running (see [Service Setup](#service-setup))

---

## Service Setup

Before using the Godot client, start the Elle service:

```bash
# Clone Elle repository
git clone https://github.com/your-org/elle-game-engine
cd elle-game-engine

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export ELLE_LLM_PROVIDER="openai"  # or "anthropic", "local", "dummy"
export OPENAI_API_KEY="your-key-here"

# Start service
python -m apps.elle_game_engine.service
```

Service will run at `http://localhost:8000`

---

## Quick Start

### 1. Basic NPC Dialogue

```gdscript
extends Node

@onready var elle = Elle  # Autoload singleton

func _ready():
    # Connect signals
    elle.action_received.connect(_on_action_received)
    elle.error_occurred.connect(_on_error_occurred)

    # Get NPC dialogue
    await elle.quick_dialogue("Bob", "Hello! I'm new in town.")


func _on_action_received(action: ElleModels.ElleGameAction):
    if action.has_dialogue():
        for line in action.dialogue:
            print("%s: \"%s\"" % [line.npc_id, line.text])
            # Display in your dialogue UI


func _on_error_occurred(error_message: String):
    print("Elle error: %s" % error_message)
```

### 2. Complete Game State

```gdscript
extends Node

@onready var elle = Elle

func talk_to_innkeeper():
    # Build game state
    var innkeeper = ElleModels.NPCState.new(
        "innkeeper",
        "Bob the Innkeeper",
        "merchant",
        "village_square",
        "neutral"
    )

    var player = ElleModels.PlayerState.new("Adventurer", "village_square")
    player.quest_stage = "intro"
    player.reputation = "outsider"

    var world = ElleModels.WorldState.new("afternoon")
    world.weather = "clear"

    var game_state = ElleModels.GameStateSnapshot.new(
        "village_square",
        player,
        world
    )
    game_state.npcs.append(innkeeper)

    # Create player intent
    var intent = ElleModels.PlayerIntent.new(
        ElleModels.PlayerIntentType.TALK_TO_NPC,
        "innkeeper",
        "Do you have any work for an adventurer?"
    )

    # Get action
    await elle.get_action(game_state, intent)
```

### 3. Request Hint

```gdscript
func get_hint():
    var player = ElleModels.PlayerState.new("Player", "forest")
    player.quest_stage = "mid"

    await elle.get_hint("forest", player)
```

---

## API Reference

### ElleClient

Main client class for communicating with Elle service.

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `base_url` | String | `"http://localhost:8000"` | Elle service URL |
| `timeout_seconds` | float | `10.0` | Request timeout |
| `max_retries` | int | `3` | Number of retry attempts |
| `debug_mode` | bool | `false` | Enable debug logging |

#### Signals

```gdscript
signal action_received(action: ElleModels.ElleGameAction)
signal error_occurred(error_message: String)
signal health_check_completed(is_healthy: bool)
```

#### Methods

##### `get_npc_dialogue(npc_id: String, scene_id: String, player_message: String, game_state: GameStateSnapshot = null)`

Get NPC dialogue based on player message.

**Parameters:**
- `npc_id`: ID of the NPC to talk to
- `scene_id`: Current scene ID
- `player_message`: What the player said/did
- `game_state`: Optional full game state

**Example:**
```gdscript
await elle.get_npc_dialogue(
    "merchant",
    "market_square",
    "Show me your wares"
)
```

##### `get_hint(scene_id: String, player_state: PlayerState)`

Get a hint for the player when they're stuck.

**Example:**
```gdscript
var player = ElleModels.PlayerState.new("Player", "dungeon")
await elle.get_hint("dungeon", player)
```

##### `get_world_reaction(scene_id: String, world_state: WorldState)`

Get world/environmental reaction.

**Example:**
```gdscript
var world = ElleModels.WorldState.new("night")
world.weather = "storm"
world.tension_level = "high"

await elle.get_world_reaction("village", world)
```

##### `get_action(game_state: GameStateSnapshot, player_intent: PlayerIntent)`

Main API method - get action from complete game state.

##### `check_health() -> bool`

Check if Elle service is reachable.

**Example:**
```gdscript
var is_healthy = await elle.check_health()
if !is_healthy:
    print("Elle service is down!")
```

##### `quick_dialogue(npc_name: String, player_message: String, scene: String = "default_scene")`

Quick dialogue with minimal setup.

**Example:**
```gdscript
await elle.quick_dialogue("Bob", "Tell me a joke!")
```

---

### ElleModels

Data models for game state and responses.

#### NPCState

```gdscript
class NPCState:
    var id: String              # Unique ID
    var name: String            # Display name
    var role: String            # Role/type
    var mood: String            # Mood (optional)
    var location: String        # Current location
    var flags: Dictionary       # Custom flags

func _init(id: String, name: String, role: String, location: String = "", mood: String = "")
```

#### PlayerState

```gdscript
class PlayerState:
    var name: String
    var location: String
    var quest_stage: String       # "intro", "mid", "late", "climax"
    var reputation: String        # "hero", "outsider", "villain"
    var traits: Dictionary        # {"brave": 3, "kind": 2}
    var inventory_tags: Array     # ["sword", "healing_herb"]

func _init(name: String, location: String)
```

#### WorldState

```gdscript
class WorldState:
    var time_of_day: String       # "morning", "afternoon", "evening", "night"
    var weather: String           # "clear", "rain", "storm", "fog"
    var tension_level: String     # "calm", "uneasy", "tense", "critical"

func _init(time_of_day: String)
```

#### GameStateSnapshot

```gdscript
class GameStateSnapshot:
    var scene_id: String
    var npcs: Array               # Array of NPCState
    var player: PlayerState
    var world: WorldState
    var tags: Array               # Custom scene tags

func _init(scene_id: String, player: PlayerState, world: WorldState)
```

#### PlayerIntentType (Enum)

```gdscript
enum PlayerIntentType {
    TALK_TO_NPC,      # Player initiating dialogue
    ENTER_SCENE,      # Player entering a location
    REQUEST_HINT,     # Player asking for guidance
    DEBUG_SUMMARY     # Developer requesting summary
}
```

#### ElleGameAction (Response)

```gdscript
class ElleGameAction:
    var mode: ActionMode          # Type of action
    var priority: String          # "low", "medium", "high"
    var dialogue: Array           # Array of DialogueLine
    var hint_text: String
    var world_reaction: WorldChange
    var debug_notes: String
    var audio_url: String         # Voice synthesis URL (optional)
    var audio_data: PackedByteArray  # Voice audio bytes (optional)

# Helper methods
func has_dialogue() -> bool
func has_world_changes() -> bool
func has_audio() -> bool
```

#### DialogueLine

```gdscript
class DialogueLine:
    var npc_id: String
    var text: String
    var tone: String              # "warm", "stern", "excited", etc.

# Get emoji for tone
func get_tone_emoji() -> String  # Returns "😊", "😠", etc.
```

---

## Examples

### Example 1: Multi-Turn Conversation

```gdscript
extends Node

@onready var elle = Elle
var conversation_state = {}

func start_quest_conversation():
    # Turn 1: Greet
    await talk_turn("Hello!")

    await get_tree().create_timer(1.0).timeout

    # Turn 2: Ask about quest
    conversation_state["knows_player"] = true
    await talk_turn("I heard you might have work?")

    await get_tree().create_timer(1.0).timeout

    # Turn 3: Accept quest
    conversation_state["quest_offered"] = true
    await talk_turn("I'll help you with the rats!")


func talk_turn(message: String):
    var innkeeper = create_innkeeper_with_state()

    var game_state = ElleModels.GameStateSnapshot.new(
        "inn",
        create_player(),
        create_world()
    )
    game_state.npcs.append(innkeeper)

    var intent = ElleModels.PlayerIntent.new(
        ElleModels.PlayerIntentType.TALK_TO_NPC,
        "innkeeper",
        message
    )

    await elle.get_action(game_state, intent)


func create_innkeeper_with_state() -> ElleModels.NPCState:
    var innkeeper = ElleModels.NPCState.new(
        "innkeeper",
        "Bob",
        "merchant",
        "inn"
    )
    innkeeper.flags = conversation_state.duplicate()
    return innkeeper
```

### Example 2: Dialogue with Voice

```gdscript
extends Node

@onready var elle = Elle
@onready var audio_player = $AudioStreamPlayer

func _ready():
    elle.action_received.connect(_on_action_received)

    # Get dialogue (Elle will include audio if voice synthesis enabled)
    await elle.quick_dialogue("Guard", "Halt! Who goes there?")


func _on_action_received(action: ElleModels.ElleGameAction):
    # Display dialogue
    if action.has_dialogue():
        var dialogue_text = action.dialogue[0].text
        $DialogueLabel.text = dialogue_text

    # Play voice audio
    if action.has_audio():
        elle.play_action_audio(action, audio_player)
```

### Example 3: Dynamic Mood System

```gdscript
extends Node

var npc_moods = {
    "innkeeper": "neutral",
    "guard": "stern",
    "merchant": "warm"
}

func update_npc_mood(npc_id: String, player_action: String):
    # Simple mood system based on player actions
    if player_action == "help":
        npc_moods[npc_id] = "grateful"
    elif player_action == "refuse":
        npc_moods[npc_id] = "annoyed"
    elif player_action == "threaten":
        npc_moods[npc_id] = "hostile"


func create_npc_with_mood(npc_id: String) -> ElleModels.NPCState:
    var npc = ElleModels.NPCState.new(
        npc_id,
        npc_id.capitalize(),
        "npc",
        "village",
        npc_moods.get(npc_id, "neutral")
    )
    return npc
```

---

## Voice Synthesis

Elle supports text-to-speech for NPC dialogue with emotion-aware voices.

### Enabling Voice Synthesis

1. **Configure Elle service** with TTS backend:
   ```bash
   export ELLE_VOICE_BACKEND="openai"  # or "elevenlabs", "piper"
   export OPENAI_API_KEY="your-key"
   ```

2. **Receive audio in responses**:
   ```gdscript
   func _on_action_received(action: ElleModels.ElleGameAction):
       if action.has_audio():
           # Audio is included automatically
           play_voice(action)
   ```

3. **Play audio**:
   ```gdscript
   func play_voice(action: ElleModels.ElleGameAction):
       if action.audio_data.size() > 0:
           var stream = AudioStreamOggVorbis.new()
           stream.data = action.audio_data
           $AudioStreamPlayer.stream = stream
           $AudioStreamPlayer.play()
   ```

### Voice Backends Comparison

| Backend | Quality | Latency | Cost | Local |
|---------|---------|---------|------|-------|
| **ElevenLabs** | ⭐⭐⭐⭐⭐ | ~2-3s | $$$ | ❌ |
| **OpenAI TTS** | ⭐⭐⭐⭐ | ~1-2s | $$ | ❌ |
| **Google Cloud** | ⭐⭐⭐⭐ | ~1-2s | $$ | ❌ |
| **Piper** | ⭐⭐⭐ | <500ms | FREE | ✅ |
| **Dummy** | ⭐ (silent) | <1ms | FREE | ✅ |

---

## Configuration

### ElleClient Configuration

Configure via Inspector or code:

```gdscript
func _ready():
    # Via code
    Elle.base_url = "http://192.168.1.100:8000"
    Elle.timeout_seconds = 15.0
    Elle.max_retries = 5
    Elle.debug_mode = true
```

### Service Configuration

Configure Elle service via environment variables:

```bash
# LLM Provider
ELLE_LLM_PROVIDER="openai"        # anthropic, openai, local, dummy
ELLE_LLM_MODEL="gpt-4o-mini"      # Model name

# API Keys
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."

# Voice Synthesis
ELLE_VOICE_BACKEND="openai"       # elevenlabs, openai, piper, dummy
ELLE_VOICE_MODEL="tts-1"          # Backend-specific model

# Performance
ELLE_ENABLE_POOL="true"           # Connection pooling
ELLE_POOL_SIZE="10"               # Pool size
ELLE_RATE_LIMIT_PER_MINUTE="60"   # Rate limit
```

---

## Troubleshooting

### Connection Refused

**Problem**: `Failed to connect to Elle service`

**Solutions**:
1. Check Elle service is running: `curl http://localhost:8000/health`
2. Verify `base_url` in ElleClient matches service URL
3. Check firewall/network settings

### Timeout Errors

**Problem**: Requests timing out

**Solutions**:
1. Increase `timeout_seconds` in ElleClient
2. Check LLM provider is responding
3. Use faster LLM model (e.g., `gpt-4o-mini` instead of `gpt-4`)

### Empty/Invalid Responses

**Problem**: Getting empty or malformed responses

**Solutions**:
1. Enable `debug_mode = true` to see request/response logs
2. Check Elle service logs for errors
3. Verify game state has required fields (scene_id, player, world)

### Voice Not Playing

**Problem**: Audio not playing in Godot

**Solutions**:
1. Verify `ELLE_VOICE_BACKEND` is set (not "dummy")
2. Check audio format compatibility (OGG recommended)
3. Ensure AudioStreamPlayer is configured correctly
4. Check action.has_audio() returns true

---

## Best Practices

### 1. Efficient Game State

Send only relevant NPCs:
```gdscript
# ❌ Bad: Send all NPCs
var game_state = create_full_world_state()

# ✅ Good: Send only nearby NPCs
var game_state = create_minimal_state()
game_state.npcs = get_npcs_in_range(player_position, 50.0)
```

### 2. Error Handling

Always handle errors gracefully:
```gdscript
func _on_error_occurred(error_message: String):
    # Log error
    push_error("Elle API error: %s" % error_message)

    # Show fallback dialogue
    show_fallback_dialogue("...")

    # Retry with exponential backoff (ElleClient does this automatically)
```

### 3. Caching

Leverage Elle's response caching:
```gdscript
# Repeated queries with same state are cached automatically
await elle.get_hint("forest", player)  # 150ms (cold)
await elle.get_hint("forest", player)  # 1ms (cached)
```

### 4. Rate Limiting

Avoid spamming requests:
```gdscript
var last_request_time = 0.0
var min_request_interval = 0.5  # 500ms

func request_dialogue():
    var now = Time.get_ticks_msec() / 1000.0

    if now - last_request_time < min_request_interval:
        return  # Too soon

    last_request_time = now
    await elle.get_npc_dialogue(...)
```

### 5. Graceful Degradation

Provide fallback content if Elle fails:
```gdscript
var fallback_dialogues = {
    "innkeeper": "Welcome, traveler!",
    "guard": "Move along.",
}

func get_dialogue_safe(npc_id: String):
    await elle.get_npc_dialogue(npc_id, ...)

    # If error occurred, use fallback
    # (error signal will be emitted)


func _on_error_occurred(_error):
    show_fallback_dialogue(fallback_dialogues.get(current_npc_id, "..."))
```

---

## Performance Tips

### Reduce Latency

1. **Use Local LLMs**: Ollama with llama3.2:3b (~100ms vs ~1-2s cloud)
2. **Enable Connection Pooling**: `ELLE_ENABLE_POOL=true`
3. **Smaller Models**: Use `gpt-4o-mini` instead of `gpt-4`
4. **Cache Aggressively**: Reuse game states when possible

### Reduce Costs

1. **Use Cheaper Models**: `gpt-4o-mini` is 60x cheaper than `gpt-4`
2. **Enable Caching**: Avoid repeat LLM calls
3. **Batch Requests**: Group multiple queries when possible
4. **Local Voice**: Use Piper TTS instead of ElevenLabs

---

## Advanced Topics

### Custom Action Modes

Handle custom action modes:
```gdscript
func _on_action_received(action: ElleModels.ElleGameAction):
    match action.mode:
        ElleModels.ActionMode.NPC_DIALOGUE:
            show_dialogue(action.dialogue)

        ElleModels.ActionMode.HINT:
            show_hint(action.hint_text)

        ElleModels.ActionMode.WORLD_REACTION:
            apply_world_changes(action.world_reaction)

        ElleModels.ActionMode.DEV_DEBUG:
            print("DEBUG: %s" % action.debug_notes)
```

### Streaming Responses (Future)

```gdscript
# Coming soon: Stream dialogue token-by-token
await elle.stream_dialogue("merchant", "Tell me a long story")
```

---

## Support

- **GitHub Issues**: https://github.com/your-org/elle-game-engine/issues
- **Discord**: https://discord.gg/elle-game-engine
- **Documentation**: https://elle-docs.readthedocs.io

---

## License

MIT License - See [LICENSE.txt](LICENSE.txt)

---

**Happy Game Development!** 🎮✨
