# Elle Tavern Demo - The Rusty Mug

A fantasy tavern and town demo showcasing the Elle Game Engine's narrative intelligence, emotion modeling, quest generation, voice synthesis, and session persistence.

**Created:** 2025-11-16  
**Engine:** Elle Game Engine v0.1.0  
**Setting:** Fantasy Medieval  
**Locations:** 4 (Tavern, Town Square, Market, Hidden Forest)  
**NPCs:** 5 unique characters with distinct personalities  

---

## Overview

Welcome to **Oakridge**, a peaceful medieval town where you'll experience dynamic NPC interactions powered by LLM-driven narrative intelligence. Every character remembers you, reacts emotionally to your actions, and offers unique quests that adapt to their feelings toward you.

### What Makes This Demo Special

- **Living NPCs**: Characters have real emotions (using PAD psychological model) that change based on your actions
- **Dynamic Quests**: Quest difficulty and rewards adapt to NPC emotional state
- **Voice Synthesis**: NPCs speak with unique voice profiles (OpenAI TTS, ElevenLabs, Piper)
- **Persistent Memory**: The game remembers your conversation history and relationship with each NPC
- **Hidden Secrets**: Earn NPCs' trust to unlock secret areas and knowledge
- **Seamless Elle Integration**: All dialogue powered by Claude, GPT-4, or local LLMs

---

## Installation

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# Install HoloLoom/Elle dependencies
cd /path/to/hello-world
pip install -r apps/elle_game_engine/requirements.txt
```

### 2. Configure LLM Provider

**For Claude (Recommended for best narrative quality):**
```bash
export ELLE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key-here
export ELLE_LLM_MODEL=claude-3-5-sonnet-20241022
```

**For GPT-4 (Fast and cost-effective):**
```bash
export ELLE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
export ELLE_LLM_MODEL=gpt-4o-mini
```

**For Local Models (Free, runs offline):**
```bash
# Install Ollama first: https://ollama.ai
ollama pull llama3.2:3b

export ELLE_LLM_PROVIDER=local
export ELLE_LLM_MODEL=llama3.2:3b
```

### 3. Optional: Voice Synthesis

```bash
# For OpenAI TTS (recommended)
export ELLE_VOICE_BACKEND=openai
export OPENAI_API_KEY=your-key-here

# For ElevenLabs (highest quality)
export ELLE_VOICE_BACKEND=elevenlabs
export ELEVENLABS_API_KEY=your-key-here

# For Piper (free, local)
export ELLE_VOICE_BACKEND=piper
# (No API key needed)
```

---

## How to Play

### Starting the Game

```bash
cd /path/to/hello-world
PYTHONPATH=. python games/elle_tavern_demo/game.py
```

You'll wake up in **The Rusty Mug Tavern** on a peaceful evening in Oakridge.

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `talk <npc>` | Talk to an NPC | `talk Bob` |
| `move <location>` | Move to a location | `move town square` |
| `look` | Examine your surroundings | `look` |
| `inventory` | Check your inventory | `inventory` |
| `quest` | View active quests | `quest` |
| `status` | View character status | `status` |
| `gift <npc>` | Give a gift to NPC (improves mood) | `gift Bob` |
| `help <npc>` | Help an NPC (improves mood) | `help Lily` |
| `insult <npc>` | Insult an NPC (worsens mood, demo only!) | `insult Marcus` |
| `save [file]` | Save your game | `save mygame` |
| `load [file]` | Load a saved game | `load mygame` |
| `quit` | Exit the game | `quit` |

### Pro Tips

1. **Build Relationships**: NPCs remember your actions. Be kind to unlock better quests and rewards!
2. **Explore Thoroughly**: Some locations are hidden until you earn an NPC's trust
3. **Talk to Everyone**: Each NPC has unique knowledge and quests
4. **Watch Emotions**: NPC emotional state affects prices, quest difficulty, and dialogue
5. **Save Often**: Your progress is precious!

---

## The World of Oakridge

### Locations

#### 1. The Rusty Mug Tavern (Starting Location)
*A cozy tavern with worn wooden tables and a crackling fireplace.*

**NPCs Found Here:**
- Bob the Innkeeper (friendly, gossipy)
- The Hooded Stranger (mysterious, cryptic)

**Time Variations:**
- Morning: Sunlight streams through windows
- Afternoon: Warm golden glow
- Evening: Shadows lengthen, dinner crowd arrives
- Night: Lanterns flicker with revelers

**Quest Hooks:**
- Rat problem in the cellar
- Strange noises from below

---

#### 2. Oakridge Town Square
*The heart of the town with a stone fountain and notice board.*

**NPCs Found Here:**
- Captain Sarah Ironhelm (stern guard)
- Lily (heartbroken child)

**Features:**
- Central hub connecting all major areas
- Town notice board with announcements
- Children playing by the fountain

**Quest Hooks:**
- Bandit troubles on the roads
- Missing cat (Lily's quest)

---

#### 3. Market Street
*Colorful merchant stalls with exotic goods and fresh produce.*

**NPCs Found Here:**
- Marcus the Magnificent (enthusiastic merchant)

**Features:**
- Buy supplies (healing potions, rope, torches)
- Prices vary based on Marcus's mood
- Busiest during afternoon hours

**Quest Hooks:**
- Lost shipment to bandits
- Rare ingredient needed
- Supplier negotiations

---

#### 4. Shadowed Forest Path (Hidden)
*A mysterious path into ancient woods. Unlocked by earning the Stranger's trust.*

**NPCs Found Here:**
- None (yet...)

**Features:**
- Glowing mushrooms illuminate the path
- Strange sounds and mysterious lights
- Requires forest_path_discovered flag (trust stranger to 0.75+)

**Quest Hooks:**
- Ancient knowledge
- Prophecy quest
- What lies deeper in the forest?

---

## NPCs - Meet the Characters

### 1. Bob the Innkeeper

**Full Name:** Bob the Innkeeper  
**Role:** Innkeeper/Gossip Hub  
**Location:** The Rusty Mug Tavern  

**Personality:**
- Friendly, helpful, and fatherly
- Loves gossip and knows everyone in town
- Observant—notices things others miss
- Folksy manner of speaking

**Speech Patterns:**
- "Well now..."
- "Bless my stars!"
- "By the light..."
- Calls you "friend" or "traveler"

**Emotional Profile:**
- Valence: 0.0 (neutral, but cheerful baseline)
- Arousal: 0.5 (moderately energized)
- Trust: 0.5 (neutral toward strangers)

**Sample Dialogue:**
> "Welcome, welcome! Pull up a chair and rest your weary bones! Can I get you some stew? Made it myself this morning. Best in Oakridge, if I do say so!"

**Quests:**
- **Rat Problem**: Strange noises from the cellar
- **Cellar Mystery**: Discover what's really down there

**Rumors He Shares:**
- Guard captain stressed about bandits
- Marcus lost a shipment
- Stranger has been here 3 days
- Lily's cat is missing

---

### 2. Captain Sarah Ironhelm

**Full Name:** Captain Sarah Ironhelm  
**Role:** Town Guard Captain  
**Location:** Town Square  

**Personality:**
- Stern, duty-bound, honorable
- Protective of townspeople
- Fair but strict
- Earned her position through merit

**Speech Patterns:**
- Addresses people as "citizen"
- Short, clipped sentences
- Military terminology
- Rarely shows emotion

**Emotional Profile:**
- Valence: -0.1 (stressed about bandits)
- Arousal: 0.6 (alert, energized)
- Dominance: 0.8 (position of authority)
- Trust: 0.4 (naturally suspicious)

**Sample Dialogue:**
> "State your business, traveler. Oakridge is a peaceful town, and I intend to keep it that way. The roads aren't safe. Travel with caution."

**Quests:**
- **Bandit Patrol**: Clear bandits from trade routes
- **Caravan Escort**: Protect merchants
- **Town Defense**: Prepare for potential attack

**Reputation Levels:**
- Suspicious (0.0): Won't trust you
- Neutral (0.5): Professional courtesy
- Trusted (0.7): Offers quests
- Deputy (0.9): Honorary guard status

---

### 3. Marcus the Magnificent

**Full Name:** Marcus the Magnificent  
**Role:** Merchant  
**Location:** Market Street  

**Personality:**
- Shrewd, enthusiastic, optimistic
- Greedy but fair
- Genuinely believes his goods are the finest
- Infectious enthusiasm

**Speech Patterns:**
- Speaks quickly and enthusiastically
- Uses superlatives ("finest," "best," "most amazing")
- Mentions profits and deals
- Exaggerates quality

**Emotional Profile:**
- Valence: 0.2 (loves his work)
- Arousal: 0.7 (high energy)
- Dominance: 0.6 (confident in his domain)
- Trust: 0.6 (trusting enough for business)

**Sample Dialogue:**
> "Welcome, welcome! You've come to the right place for the finest goods in all the realm! Ah, a discerning customer! I can see you have an eye for quality!"

**Quests:**
- **Lost Shipment**: Recover goods from bandits
- **Supplier Negotiation**: Help negotiate better prices
- **Rare Ingredient**: Find exotic spice

**Price Modifiers by Mood:**
- Angry: 1.3x (30% markup)
- Neutral: 1.0x (normal prices)
- Happy: 0.85x (15% discount)
- Grateful: 0.7x (30% discount!)

**Shop Inventory:**
- Healing Potion (50 gold)
- Rope (10 gold)
- Torch (5 gold)
- Rations (15 gold)
- Exotic Spice (100 gold)

---

### 4. The Hooded Stranger

**Full Name:** The Hooded Stranger (true name unknown)  
**Role:** Mysterious Traveler  
**Location:** The Rusty Mug Tavern (corner table)  

**Personality:**
- Cryptic, wise, enigmatic
- Speaks in riddles and metaphors
- Patient and observant
- Guards secret knowledge

**Speech Patterns:**
- Third person or passive voice
- Poetic, flowery language
- "Threads of fate"
- "Paths unseen"
- Rarely gives direct answers

**Emotional Profile:**
- Valence: 0.0 (inscrutable)
- Arousal: 0.3 (very calm, meditative)
- Dominance: 0.7 (quiet confidence)
- Trust: 0.3 (slow to trust)

**Sample Dialogue:**
> "The threads of fate are curious things... they weave and twist in patterns we cannot see. The forest remembers what the town has forgotten. But only the worthy may walk its shadowed ways."

**Quests:**
- **Forest Path Unlock**: Earn trust to learn secret location
- **Ancient Knowledge**: Discover hidden lore
- **Prophecy Quest**: Fulfill cryptic prophecy

**Trust Thresholds:**
- Stranger (0.0): Won't share secrets
- Curious (0.5): Begins to hint
- Worthy (0.75): **Reveals forest path**
- Chosen (0.9): Shares deeper mysteries

**Secret:** The stranger knows the location of the hidden forest path and what ancient secrets lie within. Earn their trust through wisdom, patience, and proving your worth.

---

### 5. Lily

**Full Name:** Lily  
**Role:** Local Child  
**Location:** Town Square  

**Personality:**
- Innocent, curious, playful
- Caring and emotional
- Currently heartbroken (missing cat)
- Brave despite sadness

**Speech Patterns:**
- Simple, direct sentences
- Shows emotions openly (crying, laughing)
- Asks lots of questions
- Easily distracted

**Emotional Profile:**
- Valence: -0.4 (sad—missing cat)
- Arousal: 0.6 (still energetic)
- Dominance: 0.2 (feels powerless)
- Trust: 0.8 (trusting child)

**Sample Dialogue:**
> "*sniffling* Have you seen my cat? She's orange and white and her name is Whiskers! I miss Whiskers so much... Mama says she'll come home, but I'm worried... Whiskers likes to chase butterflies—maybe she chased one too far?"

**Quests:**
- **Find Whiskers**: Locate Lily's missing cat

**Rewards:**
- Gratitude: Lifelong friendship
- Item: Lucky Flower (she picked it for you)

**Emotional Transformation:**
- **Before Quest:** Sad, sniffling, heartbroken
- **After Quest:** Overjoyed, grateful, gives you a flower

---

## Emotion System

NPCs have dynamic emotional states based on the **PAD (Pleasure-Arousal-Dominance) + Trust** model from psychology:

| Dimension | Range | Meaning |
|-----------|-------|---------|
| **Valence** | -1.0 to 1.0 | Negative (sad) to Positive (happy) |
| **Arousal** | 0.0 to 1.0 | Calm (0) to Excited (1) |
| **Dominance** | 0.0 to 1.0 | Submissive (0) to Dominant (1) |
| **Trust** | 0.0 to 1.0 | Distrustful (0) to Trusting (1) |

### How Your Actions Affect Emotions

| Action | Valence | Trust | Arousal | Example |
|--------|---------|-------|---------|---------|
| **help** | +0.3 | +0.2 | +0.1 | "You help Bob clean the tavern" |
| **gift** | +0.4 | +0.3 | +0.2 | "You give Marcus gold" |
| **compliment** | +0.2 | +0.1 | +0.1 | "You praise Sarah's leadership" |
| **insult** | -0.4 | -0.3 | +0.3 | "You mock Marcus's prices" |
| **threaten** | -0.5 | -0.5 | +0.4 | "You intimidate the stranger" |
| **defend** | +0.3 | +0.4 | +0.3 | "You protect Lily from bullies" |

### Emotional Decay

Emotions naturally return to baseline over time:
- **Decay rate**: 5% per hour
- **Baseline**: Each NPC has a natural emotional baseline they return to
- **Example**: If you make Bob angry, he'll gradually calm down over several in-game hours

### Game Mechanics Affected by Emotion

**1. Prices (Merchant)**
- Angry: 1.3x markup
- Happy: 0.85x discount
- Grateful: 0.7x major discount

**2. Quest Difficulty**
- Low trust: Harder quests, lower rewards
- High trust: Easier quests, better rewards, rare items

**3. Hint Generosity**
- Happy NPCs: Give more detailed hints
- Angry NPCs: Refuse to help or give vague answers

**4. Dialogue Tone**
- Emotional state determines NPC tone in LLM responses
- Elle automatically injects emotion context into prompts

---

## Quest System

Quests are dynamically generated by Elle based on:
- NPC emotional state
- Player level and progress
- World tension
- Recent events

### Quest Example: Find Whiskers (Lily's Cat)

**Prerequisites:** Talk to Lily in Town Square  
**Difficulty:** Easy (Lily is a child, quest is simple)  
**Emotional Context:** Lily is sad (valence: -0.4)

**Quest Flow:**
1. Lily tearfully asks for help finding Whiskers
2. Investigate locations where cats might hide
3. Find Whiskers (near forest path)
4. Return Whiskers to Lily
5. Lily's emotion transforms: valence -0.4 → 0.8 (overjoyed!)
6. Receive Lucky Flower as reward

**Completion Impact:**
- Lily's trust: 0.8 → 1.0
- Lily's emotional state: Sad → Overjoyed
- Town reputation: Improved
- Flag set: "cat_quest_complete"

### Quest Example: Lost Shipment (Marcus)

**Prerequisites:** Talk to Marcus, shipment_quest_started flag  
**Difficulty:** Normal (varies by Marcus's mood)  
**Emotional Context:** Marcus is optimistic but worried (valence: 0.2)

**Quest Flow:**
1. Marcus explains bandits stole his shipment
2. Track bandits to their camp
3. Recover shipment (combat or stealth)
4. Return to Marcus
5. Marcus's emotion: valence 0.2 → 0.7 (grateful!)
6. Receive reward: Gold + 30% discount forever

**Emotional Progression:**
- Worried (start) → Grateful (completion) → Loyal Friend (if repeated help)

---

## Voice Synthesis

Each NPC has a unique voice profile for text-to-speech:

| NPC | Voice ID | Pitch | Speed | Emotion | Description |
|-----|----------|-------|-------|---------|-------------|
| **Bob** | alloy | 1.0 | 1.0 | warm | Warm, friendly male voice |
| **Sarah** | nova | 0.9 | 0.95 | firm | Firm, authoritative female voice |
| **Marcus** | onyx | 1.1 | 1.1 | eager | Eager, enthusiastic male voice |
| **Stranger** | echo | 0.85 | 0.9 | calm | Deep, mysterious voice |
| **Lily** | shimmer | 1.3 | 1.1 | earnest | Young, earnest female voice |

### Enabling Voice in Your Game

```python
# In your game loop
from apps.elle_game_engine.voice import create_voice_engine

voice_engine = create_voice_engine(backend="openai")

# After getting Elle's response
result = voice_engine.synthesize(
    text=elle_response.dialogue[0].text,
    npc_id="innkeeper"
)

# Play audio (implementation depends on your game engine)
play_audio(result.audio_data)
```

---

## Technical Details

### File Structure

```
games/elle_tavern_demo/
├── README.md              # This file
├── __init__.py            # Package exports
├── world_data.py          # Locations, NPCs, helper functions
├── initial_state.py       # Game state initialization
├── actions.py             # Player action handlers
├── game.py                # Main game loop (to be created)
└── saves/                 # Save game directory (auto-created)
```

### Integration with Elle Game Engine

The demo uses Elle's full feature set:

**1. Emotion System** (`apps/elle_game_engine/emotion.py`)
- PAD + Trust emotional model
- Action-based emotion updates
- Emotional decay over time
- Emotion history tracking

**2. Quest System** (`apps/elle_game_engine/quest.py`)
- LLM-powered quest generation
- Difficulty scaling based on emotion
- Multi-objective quests
- Contextual rewards

**3. Voice Synthesis** (`apps/elle_game_engine/voice.py`)
- Multi-backend support (OpenAI, ElevenLabs, Piper)
- Per-NPC voice profiles
- Emotion-aware voice modulation
- Smart caching for common phrases

**4. Session Management** (`apps/elle_game_engine/session.py`)
- Conversation history (last 10 exchanges)
- World flags persistence
- NPC relationship tracking
- File-based or in-memory storage

**5. LLM Integration** (`apps/elle_game_engine/llm_client.py`)
- Multi-provider support (Anthropic, OpenAI, Ollama)
- Automatic prompt construction
- Emotion context injection
- Response caching

---

## Development

### Running Tests

```bash
# Validate world data
PYTHONPATH=. python games/elle_tavern_demo/world_data.py

# Test initial state creation
PYTHONPATH=. python games/elle_tavern_demo/initial_state.py

# Test action handlers
PYTHONPATH=. python games/elle_tavern_demo/actions.py
```

### Adding New NPCs

1. Add NPC to `NPCS` dict in `world_data.py`
2. Include: id, name, role, location, personality, initial_emotion, voice_profile
3. Add sample dialogue (4+ lines)
4. Add quest hooks
5. Update location's `npcs` list

### Adding New Locations

1. Add location to `LOCATIONS` dict in `world_data.py`
2. Include: name, description, npcs, exits
3. Add time_descriptions for morning/afternoon/evening/night
4. Update connected locations' exits

### Adding New Quests

1. Add quest hook to NPC's `quest_hooks`
2. Use Elle's quest generation API or create manually
3. Define objectives, rewards, and completion flags
4. Add quest logic to game loop

---

## Troubleshooting

### "No module named 'apps.elle_game_engine'"

**Solution:**
```bash
# Ensure you're running from repository root with PYTHONPATH set
cd /path/to/hello-world
PYTHONPATH=. python games/elle_tavern_demo/game.py
```

### LLM not responding / "DummyClient" responses

**Solution:**
```bash
# Check environment variables
echo $ELLE_LLM_PROVIDER  # Should be: anthropic, openai, or local
echo $ANTHROPIC_API_KEY  # Should be your API key

# Restart Elle service if running
pkill -f "apps.elle_game_engine.service"
python -m apps.elle_game_engine.service
```

### Voice synthesis not working

**Solution:**
```bash
# Check voice backend configuration
echo $ELLE_VOICE_BACKEND  # Should be: openai, elevenlabs, or piper
echo $OPENAI_API_KEY      # If using OpenAI TTS

# Test voice synthesis
PYTHONPATH=. python -c "from apps.elle_game_engine.voice import create_voice_engine; print(create_voice_engine())"
```

### NPCs not remembering conversations

**Solution:**
```bash
# Enable file-based session storage for persistence
export ELLE_SESSION_BACKEND=file
export ELLE_SESSION_PATH=./games/elle_tavern_demo/sessions
```

---

## Credits

**Game Design:** Claude Code  
**Engine:** Elle Game Engine (HoloLoom ecosystem)  
**LLM Providers:** Anthropic (Claude), OpenAI (GPT-4), Ollama (local models)  
**Voice Synthesis:** OpenAI TTS, ElevenLabs, Piper  
**Created:** 2025-11-16

---

## License

See main repository LICENSE file.

---

**Welcome to Oakridge, traveler. Your adventure awaits!**
