# Elle Game Engine - Unity Integration

**Complete Unity integration for Elle's LLM-driven narrative intelligence.**

This package provides ready-to-use C# scripts for integrating Elle Game Engine into your Unity project. Get intelligent NPC dialogue, contextual hints, and dynamic world reactions with just a few lines of code.

## Quick Start (5 Minutes)

### 1. Start Elle Service

```bash
# From repository root
python -m apps.elle_game_engine.service

# Or with a real LLM (recommended for production)
export ELLE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
python -m apps.elle_game_engine.service
```

Service will start on `http://localhost:8000`

### 2. Copy Files to Unity

Copy all `.cs` files to your Unity project:

```
YourUnityProject/
└── Assets/
    └── Scripts/
        └── Elle/
            ├── ElleClient.cs
            ├── ElleModels.cs
            └── ExampleNPCInteraction.cs  (optional - example only)
```

### 3. Add ElleClient to Your Scene

1. Create an empty GameObject: `GameObject` → `Create Empty` → Name it "ElleService"
2. Add component: `Add Component` → `Elle Client`
3. Configure in Inspector:
   - **Base URL**: `http://localhost:8000` (default)
   - **Timeout Seconds**: `10` (default)
   - **Max Retries**: `3` (default)
   - **Debug Mode**: ✓ Check for development

### 4. Write Your First Integration

```csharp
using Elle.GameEngine;
using UnityEngine;

public class MyNPCController : MonoBehaviour
{
    private ElleClient elleClient;

    void Start()
    {
        // Get reference to ElleClient
        elleClient = FindObjectOfType<ElleClient>();
    }

    async void OnNPCClicked()
    {
        // Get dialogue from Elle
        var response = await elleClient.GetNPCDialogue(
            npcId: "innkeeper",
            sceneId: "village_tavern",
            playerMessage: "Tell me about the quest"
        );

        // Display the response
        if (response.HasDialogue)
        {
            var dialogue = response.dialogue[0];
            Debug.Log($"{dialogue.npc_id}: {dialogue.text}");

            // Show in UI
            dialogueText.text = dialogue.text;
            toneEmoji.text = dialogue.GetToneEmoji();
        }
    }
}
```

Done! You now have intelligent NPC dialogue powered by Elle.

## Complete Example

See `ExampleNPCInteraction.cs` for a full working implementation including:
- UI integration (dialogue panel, input field, buttons)
- Loading states and error handling
- Game state construction
- Multiple dialogue types (NPC dialogue, hints, world reactions)

### Setting Up the Example

1. Create UI Canvas with:
   - **Dialogue Panel** (Panel)
     - NPC Name (TextMeshProUGUI)
     - Tone Emoji (TextMeshProUGUI)
     - Dialogue Text (TextMeshProUGUI)
     - Player Input Field (TMP_InputField)
     - Send Button (Button)
     - Loading Indicator (Image/Text, optional)

2. Add `ExampleNPCInteraction` component to your NPC GameObject

3. Assign UI elements in Inspector

4. Click the NPC or call `StartConversation()` to begin

## API Reference

### ElleClient

Main client for calling Elle Game Engine API.

#### Methods

##### GetNPCDialogue
Get NPC dialogue based on player interaction.

```csharp
public async Task<ElleGameAction> GetNPCDialogue(
    string npcId,
    string sceneId,
    string playerMessage,
    GameStateSnapshot gameState = null)
```

**Parameters**:
- `npcId` - Unique NPC identifier (e.g., "innkeeper", "guard_1")
- `sceneId` - Current scene/location (e.g., "village_square")
- `playerMessage` - What the player said/typed
- `gameState` - (Optional) Complete game state. If null, creates minimal state.

**Returns**: `ElleGameAction` with dialogue, priority, and optional world changes

**Example**:
```csharp
var action = await elleClient.GetNPCDialogue(
    "shopkeeper",
    "market_square",
    "What do you have for sale?"
);

Debug.Log(action.dialogue[0].text);
// Output: "Welcome! I've got healing potions, weapons, and rare artifacts."
```

##### GetHint
Get a hint when the player is stuck.

```csharp
public async Task<ElleGameAction> GetHint(
    string sceneId,
    PlayerState playerState)
```

**Parameters**:
- `sceneId` - Current scene
- `playerState` - Player's current state (location, inventory, etc.)

**Returns**: `ElleGameAction` with `hint_text`

**Example**:
```csharp
var player = new PlayerState("Hero", "dark_forest");
var action = await elleClient.GetHint("dark_forest", player);

Debug.Log(action.hint_text);
// Output: "Try exploring to the east, near the old oak tree."
```

##### GetWorldReaction
Get environmental/world response.

```csharp
public async Task<ElleGameAction> GetWorldReaction(
    string sceneId,
    WorldState worldState)
```

**Parameters**:
- `sceneId` - Current scene
- `worldState` - World conditions (time, weather, tension)

**Returns**: `ElleGameAction` with `world_reaction`

**Example**:
```csharp
var world = new WorldState("night") { weather = "storm" };
var action = await elleClient.GetWorldReaction("castle_courtyard", world);

Debug.Log(action.world_reaction.description);
// Output: "Thunder crashes overhead. The guards look nervous."
```

##### GetAction
Generic method for full control over request.

```csharp
public async Task<ElleGameAction> GetAction(
    GameStateSnapshot gameState,
    PlayerIntent playerIntent)
```

All other methods are convenience wrappers around this one.

##### CheckHealth
Verify Elle service is running and reachable.

```csharp
public async Task<bool> CheckHealth()
```

**Returns**: `true` if service is healthy, `false` otherwise

**Example**:
```csharp
if (await elleClient.CheckHealth())
{
    Debug.Log("Elle service is online");
}
else
{
    Debug.LogError("Cannot reach Elle service");
}
```

### Data Models

#### GameStateSnapshot

Complete snapshot of game state for Elle's decision-making.

```csharp
var gameState = new GameStateSnapshot(
    scene_id: "village_square",
    player: playerState,
    world: worldState
);

// Add NPCs
gameState.npcs.Add(new NPCState("guard_1", "Captain Morgan", "guard"));

// Add tags
gameState.tags.Add("post_battle");
gameState.tags.Add("first_visit");
```

#### NPCState

State of a non-player character.

```csharp
var npc = new NPCState(
    id: "innkeeper",
    name: "Bob",
    role: "innkeeper",
    location: "village_inn",
    mood: "nervous"
);

npc.flags.Add("knows_secret", true);
```

**Roles**: "merchant", "guard", "villager", "quest_giver", etc.
**Moods**: "neutral", "annoyed", "grateful", "curious", "hostile"

#### PlayerState

State of the player character.

```csharp
var player = new PlayerState("Hero", "village_square");
player.quest_stage = "mid";
player.reputation = "hero";
player.traits.Add("brave", 3);
player.traits.Add("kind", 2);
player.inventory_tags.Add("ancient_key");
player.inventory_tags.Add("healing_herb");
```

**Quest Stages**: "intro", "mid", "late", "climax"
**Reputation**: "hero", "outsider", "villain", "neutral"

#### WorldState

Ambient world conditions.

```csharp
var world = new WorldState("night");
world.weather = "storm";
world.tension_level = "high";
```

**Time of Day**: "morning", "afternoon", "evening", "night", "dawn", "dusk"
**Weather**: "clear", "rain", "storm", "fog", "snow"
**Tension**: "calm", "uneasy", "tense", "high", "critical"

#### PlayerIntent

What the player is doing.

```csharp
// Talk to NPC
var intent = new PlayerIntent(
    PlayerIntentType.talk_to_npc,
    target_npc_id: "innkeeper",
    raw_input: "Hello!"
);

// Request hint
var hintIntent = new PlayerIntent(PlayerIntentType.request_hint);

// Enter scene
var enterIntent = new PlayerIntent(PlayerIntentType.enter_scene);
```

**Types**: `talk_to_npc`, `enter_scene`, `request_hint`, `debug_summary`

#### ElleGameAction

Response from Elle.

```csharp
var action = await elleClient.GetNPCDialogue(...);

// Check what type of action
if (action.HasDialogue)
{
    foreach (var line in action.dialogue)
    {
        Debug.Log($"{line.npc_id} ({line.tone}): {line.text}");
        dialogueUI.ShowLine(line.text, line.GetToneEmoji());
    }
}

if (action.HasWorldChanges)
{
    foreach (var flag in action.world_reaction.flag_changes)
    {
        gameState.SetFlag(flag.Key, flag.Value);
    }
}
```

**Action Modes**: `npc_dialogue`, `hint`, `world_reaction`, `dev_debug`
**Priority Levels**: `low`, `medium`, `high`

#### DialogueLine

A line of NPC dialogue.

```csharp
var line = action.dialogue[0];

// Display text
dialogueText.text = line.text;

// Show tone emoji
toneEmoji.text = line.GetToneEmoji();  // 😊😠🤔😃😢

// Log
Debug.Log($"{line.npc_id} ({line.tone}): {line.text}");
```

**Tones**: "warm", "stern", "cryptic", "excited", "sad", "neutral", "curious", "grateful", "annoyed", "hostile"

## Advanced Usage

### Building Rich Game State

```csharp
private GameStateSnapshot BuildCompleteGameState()
{
    // Create detailed player
    var player = new PlayerState("Aria the Brave", "throne_room");
    player.quest_stage = "climax";
    player.reputation = "hero";
    player.traits.Add("brave", 5);
    player.traits.Add("diplomatic", 3);
    player.inventory_tags.Add("legendary_sword");
    player.inventory_tags.Add("royal_seal");

    // Create world with tension
    var world = new WorldState("dusk");
    world.weather = "storm";
    world.tension_level = "critical";

    // Create multiple NPCs
    var king = new NPCState("king", "King Aldric", "ruler", "throne_room", "worried");
    king.flags.Add("trusts_player", true);

    var advisor = new NPCState("advisor", "Merlin", "advisor", "throne_room", "cryptic");
    advisor.flags.Add("knows_prophecy", true);

    // Build snapshot
    var gameState = new GameStateSnapshot("throne_room", player, world);
    gameState.npcs.Add(king);
    gameState.npcs.Add(advisor);
    gameState.tags.Add("final_confrontation");
    gameState.tags.Add("all_allies_present");

    return gameState;
}
```

### Error Handling

```csharp
try
{
    var action = await elleClient.GetNPCDialogue(npcId, sceneId, message);
    DisplayDialogue(action);
}
catch (Exception e)
{
    Debug.LogError($"Elle request failed: {e.Message}");

    // Show fallback dialogue
    DisplayFallbackDialogue("The NPC looks at you silently.");

    // Or retry with exponential backoff
    await RetryWithBackoff();
}
```

### Caching Responses

```csharp
private Dictionary<string, ElleGameAction> responseCache = new Dictionary<string, ElleGameAction>();

private async Task<ElleGameAction> GetCachedDialogue(string npcId, string message)
{
    string key = $"{npcId}:{message}";

    if (responseCache.ContainsKey(key))
    {
        Debug.Log("Using cached response");
        return responseCache[key];
    }

    var action = await elleClient.GetNPCDialogue(npcId, sceneId, message);
    responseCache[key] = action;
    return action;
}
```

### Multiple NPCs in Scene

```csharp
var gameState = new GameStateSnapshot(sceneId, player, world);

// Add all NPCs in the scene
gameState.npcs.Add(new NPCState("merchant", "Sarah", "merchant", sceneId, "friendly"));
gameState.npcs.Add(new NPCState("guard", "Marcus", "guard", sceneId, "stern"));
gameState.npcs.Add(new NPCState("child", "Timmy", "villager", sceneId, "curious"));

// Get dialogue - Elle will consider all NPCs for context
var action = await elleClient.GetNPCDialogue("merchant", sceneId, "Hello!", gameState);
```

## Configuration

### ElleClient Inspector Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Base URL** | `http://localhost:8000` | Elle service endpoint |
| **Timeout Seconds** | `10` | Request timeout |
| **Max Retries** | `3` | Retry attempts on failure |
| **Debug Mode** | `false` | Enable verbose logging |

### Environment Variables (Elle Service)

Configure the Elle service with environment variables:

```bash
# Use OpenAI (recommended for production)
export ELLE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
export ELLE_LLM_MODEL=gpt-4o-mini

# Use Anthropic Claude
export ELLE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key-here
export ELLE_LLM_MODEL=claude-3-haiku-20240307

# Use local Ollama (free!)
export ELLE_LLM_PROVIDER=local
export ELLE_LLM_MODEL=llama3.2:3b
```

See main README for complete LLM provider configuration.

## Performance

### Latency

| Provider | Model | Typical Latency | Cost per 1k Interactions |
|----------|-------|-----------------|--------------------------|
| Dummy | - | 10ms | Free |
| Ollama (local) | llama3.2:3b | 200-500ms | Free |
| OpenAI | gpt-4o-mini | 300-800ms | $0.75 |
| Anthropic | claude-3-haiku | 400-900ms | $1.50 |

### Optimization Tips

1. **Use Async/Await**: Don't block the main thread
   ```csharp
   async void OnNPCClicked()
   {
       var action = await elleClient.GetNPCDialogue(...);
       // Main thread remains responsive
   }
   ```

2. **Cache Common Responses**: Store frequently used responses locally

3. **Batch Related Requests**: Send complete game state once, not multiple partial requests

4. **Use Dummy Provider for Development**: Fast iteration without API costs

5. **Enable Retries**: Network issues are transient - retries help

## Troubleshooting

### "Cannot reach Elle service"

**Problem**: ElleClient can't connect to Elle service

**Solutions**:
1. Check Elle service is running: `curl http://localhost:8000/health`
2. Verify Base URL in Inspector matches service address
3. Check firewall isn't blocking port 8000
4. Enable Debug Mode to see detailed error messages

### "Request timeout"

**Problem**: Requests taking too long

**Solutions**:
1. Increase Timeout Seconds in Inspector
2. Switch to faster LLM model (gpt-4o-mini, claude-3-haiku)
3. Use local Ollama for development
4. Check network latency to LLM provider

### "JSON deserialization failed"

**Problem**: Can't parse Elle response

**Solutions**:
1. Update Unity integration to latest version
2. Check Elle service version matches
3. Enable Debug Mode to see raw JSON response
4. Verify custom JSON serialization settings aren't interfering

### "NullReferenceException on dialogue"

**Problem**: `action.dialogue` is null or empty

**Solutions**:
1. Check `action.HasDialogue` before accessing `action.dialogue`
2. Verify player intent is correct (needs `target_npc_id` for talk_to_npc)
3. Check Elle service logs for errors
4. Ensure NPC exists in game state

### Build Errors

**Problem**: Unity build fails with async/await errors

**Solutions**:
1. Unity 2018.3+ required for async/await
2. Set scripting runtime to .NET 4.x: `Edit` → `Project Settings` → `Player` → `Other Settings` → `Scripting Runtime Version` → `.NET 4.x`
3. Ensure `using System.Threading.Tasks;` is imported

## Best Practices

1. **Always check health on startup**:
   ```csharp
   async void Start()
   {
       if (!await elleClient.CheckHealth())
       {
           Debug.LogError("Elle service offline - using fallback dialogue");
           useFallbackDialogue = true;
       }
   }
   ```

2. **Provide rich game state**: More context = better responses
   ```csharp
   // Good - rich context
   var player = new PlayerState("Hero", sceneId);
   player.quest_stage = "mid";
   player.inventory_tags.Add("ancient_key");

   // Bad - minimal context
   var player = new PlayerState("Hero", sceneId);
   ```

3. **Handle errors gracefully**: Always have fallback dialogue
   ```csharp
   try {
       var action = await elleClient.GetNPCDialogue(...);
   } catch {
       ShowFallbackDialogue("...");
   }
   ```

4. **Use async properly**: Don't block the main thread
   ```csharp
   // Good
   async void OnClick() {
       var action = await elleClient.GetNPCDialogue(...);
   }

   // Bad - blocks main thread
   void OnClick() {
       var action = elleClient.GetNPCDialogue(...).Result;
   }
   ```

5. **Enable debug mode during development**: Easier troubleshooting

## Examples

See `ExampleNPCInteraction.cs` for a complete, production-ready example including:
- UI integration
- Loading states
- Error handling
- Game state construction
- Tone emoji display

## Support

For issues, questions, or feature requests:
- See main Elle Game Engine documentation
- Check service logs for errors
- Enable Debug Mode for detailed client logs

## License

See main repository LICENSE file.

---

**Built for Unity game developers who want intelligent NPCs without complexity.**
