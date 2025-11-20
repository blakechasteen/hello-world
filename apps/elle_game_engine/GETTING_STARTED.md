# Getting Started with BigPlay

**Build Your First LLM-Native Game in 30 Minutes**

Version: 1.0.0
Last Updated: 2025-11-16

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First NPC](#your-first-npc)
4. [Adding Emotions](#adding-emotions)
5. [Creating Quests](#creating-quests)
6. [Voice Synthesis](#voice-synthesis)
7. [Game Engine Integration](#game-engine-integration)
8. [Deployment](#deployment)

---

## 🎨 Interactive Visualizations

Before diving into the code, explore BigPlay's architecture and capabilities through our **interactive visualizations**:

📊 **[Open Interactive Visualizations](visualizations/index.html)** - Best viewed in a web browser

**Available Now:**
- **System Architecture** - Click through all 16 components to understand how BigPlay works
- **PAD Emotion Model** - 3D visualization of our emotion system (rotate with your mouse!)
- **Performance Dashboard** - Real-time metrics with live charts

**What You'll Learn:**
- How the 4-layer architecture connects (Client → API → Engine → Data)
- How NPCs model emotions in 3D space (Pleasure, Arousal, Dominance)
- What production performance looks like (latency, throughput, costs)

💡 **Tip:** Open `visualizations/index.html` in your browser and explore the interactive diagrams before reading the docs. It'll make everything clearer!

---

## Prerequisites

**What You Need:**
- Python 3.10+ installed
- Basic understanding of HTTP APIs
- A game engine (Unity, Godot, or web browser)
- Optional: OpenAI or Anthropic API key (for real LLMs)

**Skills Required:**
- ⭐ Beginner: Can follow tutorials
- No LLM experience needed
- No game dev experience needed (we'll teach you!)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/hello-world.git
cd hello-world
```

### Step 2: Install Dependencies

```bash
# Install BigPlay dependencies
pip install -r apps/elle_game_engine/requirements.txt

# Verify installation
python -c "import fastapi, pydantic; print('✅ Installation successful!')"
```

### Step 3: Start BigPlay Engine

```bash
# Start the engine
cd apps/elle_game_engine
uvicorn service:app --reload --port 8000

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete.
```

### Step 4: Verify It's Working

Open a new terminal and run:

```bash
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0"}
```

🎉 **BigPlay is now running!**

---

## Your First NPC

Let's create Bob, a friendly innkeeper who responds to players.

### Create the NPC

```bash
curl -X POST "http://localhost:8000/elle/game/action" \
  -H "Content-Type: application/json" \
  -d '{
    "game_state": {
      "scene_id": "tavern",
      "npcs": [
        {
          "id": "bob",
          "name": "Bob the Innkeeper",
          "role": "innkeeper",
          "mood": "friendly",
          "location": "tavern"
        }
      ],
      "player": {
        "name": "Hero",
        "location": "tavern"
      },
      "world": {
        "time_of_day": "evening"
      }
    },
    "player_intent": {
      "type": "talk_to_npc",
      "target_npc_id": "bob",
      "raw_input": "Hello!"
    }
  }'
```

### Expected Response

```json
{
  "mode": "npc_dialogue",
  "priority": "medium",
  "dialogue": [
    {
      "npc_id": "bob",
      "npc_name": "Bob the Innkeeper",
      "text": "Welcome to the tavern, friend! What brings you here?",
      "tone": "warm"
    }
  ],
  "metadata": {
    "latency_ms": 150,
    "cached": false,
    "provider": "dummy"
  }
}
```

🎊 **Congratulations! You just created your first NPC!**

---

## Adding Emotions

Now let's make Bob emotional - he'll remember how you treat him.

### Step 1: Add Emotional State

```python
# Create a Python script: test_emotions.py
import httpx
import asyncio

async def talk_to_bob(message: str, action: str = None):
    """Talk to Bob and optionally perform an action."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/elle/game/action",
            json={
                "game_state": {
                    "scene_id": "tavern",
                    "npcs": [{
                        "id": "bob",
                        "name": "Bob",
                        "role": "innkeeper",
                        "mood": "neutral",
                        "location": "tavern",
                        "emotional_state": {
                            "valence": 0.0,  # Neutral
                            "arousal": 0.5,
                            "dominance": 0.5,
                            "trust": 0.5
                        }
                    }],
                    "player": {"name": "Hero", "location": "tavern"},
                    "world": {"time_of_day": "evening"}
                },
                "player_intent": {
                    "type": "talk_to_npc" if not action else action,
                    "target_npc_id": "bob",
                    "raw_input": message
                }
            }
        )
        return response.json()

# Test it
async def main():
    # 1. Greet Bob
    print("\\n=== Greeting Bob ===")
    response = await talk_to_bob("Hello!")
    print(f"Bob: {response['dialogue'][0]['text']}")

    # 2. Help Bob
    print("\\n=== Helping Bob ===")
    response = await talk_to_bob("I'll help you clean!", action="help")
    print(f"Bob: {response['dialogue'][0]['text']}")
    print(f"Bob's emotion: {response.get('emotion_change', 'No change')}")

    # 3. Insult Bob
    print("\\n=== Insulting Bob ===")
    response = await talk_to_bob("This tavern is filthy!", action="insult")
    print(f"Bob: {response['dialogue'][0]['text']}")
    print(f"Bob's emotion: {response.get('emotion_change', 'No change')}")

asyncio.run(main())
```

### Run It

```bash
python test_emotions.py
```

### Expected Output

```
=== Greeting Bob ===
Bob: Welcome to the tavern, friend!

=== Helping Bob ===
Bob: Oh, thank you! That's very kind of you!
Bob's emotion: {'valence': +0.3, 'trust': +0.2} (happier)

=== Insulting Bob ===
Bob: How dare you! Get out of my tavern!
Bob's emotion: {'valence': -0.5, 'trust': -0.2} (angry)
```

✨ **Bob now has emotions and remembers how you treat him!**

---

## Creating Quests

Let's make Bob offer a quest when he trusts you.

### Step 1: Check for Quest Availability

```bash
curl -X POST "http://localhost:8000/elle/game/quest/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "npc_id": "bob",
    "npc_name": "Bob",
    "npc_role": "innkeeper",
    "emotional_state_data": {
      "valence": 0.3,
      "arousal": 0.5,
      "dominance": 0.5,
      "trust": 0.7
    },
    "player_level": 1,
    "world_state": {
      "time_of_day": "evening",
      "tension": 0.0
    }
  }'
```

### Expected Response

```json
{
  "id": "bob_12ab34cd",
  "title": "The Cellar Rat Problem",
  "giver": "bob",
  "description": "Bob needs help clearing rats from his tavern cellar.",
  "difficulty": "easy",
  "objectives": [
    {
      "id": "clear_rats",
      "description": "Clear 5 rats from the cellar",
      "target": 5,
      "progress": 0
    }
  ],
  "rewards": {
    "xp": 50,
    "gold": 10,
    "items": []
  },
  "emotional_rationale": "Bob trusts you and needs help with a small problem."
}
```

### Step 2: Complete the Quest

```python
# Update test_emotions.py
async def complete_quest():
    # Accept quest
    print("\\n=== Accepting Quest ===")
    response = await talk_to_bob("I'll help with the rats!")

    # Simulate clearing rats
    print("\\n=== Clearing Rats ===")
    for i in range(1, 6):
        print(f"Cleared rat {i}/5")
        await asyncio.sleep(0.5)

    # Report completion
    print("\\n=== Completing Quest ===")
    response = await talk_to_bob("I cleared all the rats!")
    print(f"Bob: {response['dialogue'][0]['text']}")
    print(f"Rewards: {response.get('rewards', {})}")
    print(f"Bob's new trust: {response.get('emotional_state', {}).get('trust', 'Unknown')}")

asyncio.run(complete_quest())
```

🎯 **You've created dynamic quest generation based on NPC emotions!**

---

## Voice Synthesis

Let's make Bob speak his dialogue.

### Step 1: Enable Voice (OpenAI TTS)

```bash
# Install OpenAI SDK
pip install openai

# Set API key
export OPENAI_API_KEY="sk-..."
export ELLE_VOICE_BACKEND="openai"

# Restart BigPlay engine
uvicorn service:app --reload --port 8000
```

### Step 2: Request Voice

```bash
curl -X POST "http://localhost:8000/elle/game/voice/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to my tavern, friend!",
    "npc_id": "bob",
    "format": "mp3"
  }' \
  --output bob_greeting.mp3

# Play it (macOS)
afplay bob_greeting.mp3

# Play it (Linux)
mpg123 bob_greeting.mp3

# Play it (Windows)
start bob_greeting.mp3
```

### Step 3: Integrate Voice in Dialogue

```python
async def talk_with_voice(message: str):
    # Get dialogue response
    response = await talk_to_bob(message)
    dialogue_text = response['dialogue'][0]['text']

    # Get voice audio
    async with httpx.AsyncClient() as client:
        voice_response = await client.post(
            "http://localhost:8000/elle/game/voice/synthesize",
            json={
                "text": dialogue_text,
                "npc_id": "bob",
                "format": "mp3"
            }
        )

        # Save audio
        with open("bob_response.mp3", "wb") as f:
            f.write(voice_response.content)

        print(f"Bob: {dialogue_text}")
        print("🔊 Audio saved to bob_response.mp3")
```

🔊 **Bob can now speak!**

---

## Game Engine Integration

### Unity Integration

**Step 1: Download Unity Client**

```bash
# Copy Unity client to your Unity project
cp apps/elle_game_engine/unity_integration/*.cs Assets/Scripts/Elle/
```

**Step 2: Create NPC Script**

```csharp
// Assets/Scripts/BobNPC.cs
using UnityEngine;
using ElleGameEngine;

public class BobNPC : MonoBehaviour
{
    private ElleClient elle;
    private GameStateSnapshot gameState;

    async void Start()
    {
        // Initialize Elle client
        elle = new ElleClient("http://localhost:8000");

        // Build game state
        gameState = new GameStateSnapshot
        {
            scene_id = "tavern",
            npcs = new List<NPCState>
            {
                new NPCState
                {
                    id = "bob",
                    name = "Bob the Innkeeper",
                    role = "innkeeper",
                    location = "tavern"
                }
            },
            player = new PlayerState
            {
                name = "Hero",
                location = "tavern"
            }
        };
    }

    public async void OnPlayerTalk(string message)
    {
        // Send to Elle
        var intent = new PlayerIntent
        {
            type = "talk_to_npc",
            target_npc_id = "bob",
            raw_input = message
        };

        var response = await elle.GetGameAction(gameState, intent);

        // Display dialogue
        DialogueUI.ShowText(response.dialogue[0].text);

        // Play voice (if available)
        if (response.dialogue[0].audio_url != null)
        {
            var clip = await elle.GetVoiceClip(response.dialogue[0].audio_url);
            audioSource.PlayOneShot(clip);
        }
    }
}
```

### Godot Integration

**Step 1: Install Godot Plugin**

```bash
# Copy Godot plugin to your project
cp -r apps/elle_game_engine/godot_integration/addons/elle_game_engine godot_project/addons/
```

**Step 2: Enable Plugin**

1. Open Godot
2. Project → Project Settings → Plugins
3. Enable "Elle Game Engine"

**Step 3: Create NPC Script**

```gdscript
# bob_npc.gd
extends Node

@onready var elle = Elle  # Autoload

func _ready():
    # Connect signal
    elle.action_received.connect(_on_action_received)

func talk_to_bob(message: String):
    # Build game state
    var game_state = {
        "scene_id": "tavern",
        "npcs": [{
            "id": "bob",
            "name": "Bob",
            "role": "innkeeper",
            "location": "tavern"
        }],
        "player": {
            "name": "Hero",
            "location": "tavern"
        }
    }

    var player_intent = {
        "type": "talk_to_npc",
        "target_npc_id": "bob",
        "raw_input": message
    }

    # Send to Elle
    await elle.get_game_action(game_state, player_intent)

func _on_action_received(action: ElleModels.ElleGameAction):
    # Show dialogue
    $DialogueLabel.text = action.dialogue[0].text

    # Play voice
    if action.has_audio():
        elle.play_action_audio(action, $AudioStreamPlayer)
```

---

## Deployment

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t bigplay-engine .

# Run container
docker run -d -p 8000:8000 \
  -e ELLE_LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=your_key \
  bigplay-engine
```

### Option 2: Cloud (Railway)

1. Push to GitHub
2. Go to [Railway.app](https://railway.app)
3. "New Project" → "Deploy from GitHub"
4. Select your repository
5. Add environment variables:
   - `ELLE_LLM_PROVIDER=openai`
   - `OPENAI_API_KEY=your_key`
6. Deploy!

### Option 3: Standalone Executable

```bash
# Build standalone
cd games/elle_tavern_demo
python build_standalone.py

# Distribute
# dist/elle-tavern-demo.zip (includes everything)
```

---

## Next Steps

### Tutorials
1. **[Building a Complete RPG](tutorials/rpg.md)** - Town, quests, NPCs
2. **[Social Simulation Game](tutorials/social_sim.md)** - NPC-NPC interactions
3. **[Multiplayer Integration](tutorials/multiplayer.md)** - Shared world state

### Advanced Topics
1. **[Fine-Tuning LLMs](advanced/fine_tuning.md)** - 50-70% cost reduction
2. **[Custom Emotions](advanced/custom_emotions.md)** - Add new emotion types
3. **[Procedural Quests](advanced/procedural_quests.md)** - Infinite quests

### Community
- **Discord**: [Join 1000+ developers](https://discord.gg/bigplay)
- **Examples**: [Browse 50+ games](https://github.com/bigplay/examples)
- **Forum**: [Ask questions](https://forum.bigplay.dev)

---

## Troubleshooting

### "Service not responding"

```bash
# Check if service is running
curl http://localhost:8000/health

# If not, restart
uvicorn service:app --reload --port 8000
```

### "LLM responses are slow"

```bash
# Enable connection pooling
export ELLE_ENABLE_POOL=true
export ELLE_POOL_SIZE=10

# Restart service
```

### "Voice synthesis not working"

```bash
# Check backend configuration
curl http://localhost:8000/metrics | grep voice

# Verify API key
echo $OPENAI_API_KEY

# Try dummy backend for testing
export ELLE_VOICE_BACKEND=dummy
```

---

## Complete Example Game

Here's a complete 100-line game in Python:

```python
# my_first_game.py
import httpx
import asyncio

class TavernGame:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.base_url = "http://localhost:8000"
        self.bob_trust = 0.5

    async def talk_to_bob(self, message: str):
        """Send message to Bob."""
        response = await self.client.post(
            f"{self.base_url}/elle/game/action",
            json={
                "game_state": {
                    "scene_id": "tavern",
                    "npcs": [{
                        "id": "bob",
                        "name": "Bob",
                        "role": "innkeeper",
                        "emotional_state": {
                            "valence": 0.0,
                            "trust": self.bob_trust
                        }
                    }],
                    "player": {"name": "You", "location": "tavern"}
                },
                "player_intent": {
                    "type": "talk_to_npc",
                    "target_npc_id": "bob",
                    "raw_input": message
                }
            }
        )
        data = response.json()
        bob_response = data["dialogue"][0]["text"]

        # Update trust
        if "emotional_state" in data:
            self.bob_trust = data["emotional_state"].get("trust", self.bob_trust)

        return bob_response

    async def play(self):
        """Main game loop."""
        print("=== Welcome to The Tavern ===")
        print("You enter a cozy tavern. Bob the innkeeper greets you.\\n")

        # Intro
        response = await self.talk_to_bob("Hello!")
        print(f"Bob: {response}\\n")

        # Game loop
        while True:
            player_input = input("You: ")
            if player_input.lower() in ["quit", "exit", "bye"]:
                print("Thanks for playing!")
                break

            response = await self.talk_to_bob(player_input)
            print(f"Bob: {response}")
            print(f"(Trust: {self.bob_trust:.0%})\\n")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

# Run the game
async def main():
    async with TavernGame() as game:
        await game.play()

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**

```bash
python my_first_game.py
```

---

## 🎉 Congratulations!

You've successfully:
- ✅ Installed BigPlay
- ✅ Created your first NPC
- ✅ Added emotions
- ✅ Generated quests
- ✅ Integrated voice synthesis
- ✅ Built a complete game

**Ready for more? Check out:**
- [Architecture Guide](ARCHITECTURE.md) - Deep technical dive
- [API Reference](API_REFERENCE.md) - Complete API docs
- [Developer Guide](DEVELOPER_GUIDE.md) - Advanced tutorials

---

*Questions? Join our [Discord](https://discord.gg/bigplay) or email support@bigplay.dev*

*Last Updated: 2025-11-16*
*Version: 1.0.0*
