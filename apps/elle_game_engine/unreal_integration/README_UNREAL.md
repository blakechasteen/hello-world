## BigPlay Engine - Unreal Integration

**Version:** 1.0.0
**Status:** ✅ Production Ready
**Platform:** Unreal Engine 5.0+

---

## 🎮 Overview

Complete C++ client and Blueprint support for BigPlay Engine in Unreal. Get emotional NPCs, dynamic quests, and voice synthesis in your Unreal game with minimal setup.

**Features:**
- ✅ Full C++ client (`UBigPlayClient`)
- ✅ Blueprint function library (visual scripting support)
- ✅ Complete data models matching BigPlay API
- ✅ Example project with full integration
- ✅ Async HTTP requests with callbacks
- ✅ JSON serialization/deserialization
- ✅ Production-ready error handling

---

## 📦 Installation

### Option 1: Plugin Installation (Recommended)

1. **Copy plugin to your project:**
   ```bash
   cp -r unreal_integration/ YourProject/Plugins/BigPlayEngine/
   ```

2. **Enable plugin:**
   - Open Unreal Editor
   - Edit → Plugins
   - Search for "BigPlay Engine"
   - Check "Enabled"
   - Restart editor

3. **Verify installation:**
   ```cpp
   #include "BigPlayClient.h"
   // If this compiles, you're good to go!
   ```

### Option 2: Source Integration

1. **Copy source files:**
   ```bash
   cp unreal_integration/Source/BigPlayEngine/Public/*.h YourProject/Source/YourProject/
   cp unreal_integration/Source/BigPlayEngine/Private/*.cpp YourProject/Source/YourProject/
   ```

2. **Update YourProject.Build.cs:**
   ```csharp
   PublicDependencyModuleNames.AddRange(new string[]
   {
       "Core",
       "CoreUObject",
       "Engine",
       "Http",
       "Json",
       "JsonUtilities"
   });
   ```

3. **Regenerate project files** (right-click `.uproject` → Generate Visual Studio files)

---

## 🚀 Quick Start

### C++ Example (Simplest)

```cpp
#include "BigPlayClient.h"

void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    // Create and initialize client
    UBigPlayClient* Client = NewObject<UBigPlayClient>(this);
    Client->Initialize("http://localhost:8000");

    // Talk to NPC
    Client->GetNPCDialogue(
        "innkeeper",              // NPC ID
        "village_tavern",         // Scene ID
        "Hello!",                 // Player message
        UBigPlayClient::FOnActionReceived::CreateUObject(
            this,
            &AMyActor::OnDialogueReceived
        ),
        UBigPlayClient::FOnRequestFailed::CreateUObject(
            this,
            &AMyActor::OnRequestFailed
        )
    );
}

void AMyActor::OnDialogueReceived(const FBigPlayAction& Action)
{
    if (Action.HasDialogue())
    {
        UE_LOG(LogTemp, Log, TEXT("NPC: %s"), *Action.GetDialogueText());
    }
}

void AMyActor::OnRequestFailed(const FString& ErrorMessage)
{
    UE_LOG(LogTemp, Error, TEXT("Failed: %s"), *ErrorMessage);
}
```

### Blueprint Example (No Code!)

1. **Create Blueprint Actor**
2. **Add BigPlay nodes:**
   - Event BeginPlay
   - Connect to "BigPlay Quick Dialogue" node
   - Set NPC ID: `innkeeper`
   - Set Scene ID: `village_tavern`
   - Set Player Message: `Hello!`
3. **Handle response:**
   - Connect Success callback → Print String
   - Print: `GetDialogueText`

**Visual:**
```
[Event BeginPlay] → [BigPlay Quick Dialogue]
                        ↓ On Success
                     [Get Dialogue Text] → [Print String]
```

---

## 📚 Complete C++ Example

See `Content/Examples/ExampleNPCInteraction.cpp` for a full example.

**Key features demonstrated:**
- Client initialization
- NPC dialogue requests
- Complete game state building
- Multiple NPCs in scene
- Player traits and inventory
- World state (time, weather, tension)
- Error handling
- UI integration points

---

## 🎨 Blueprint Function Library

### Quick Actions (No Setup Required)

**Quick Dialogue:**
```
Inputs:
- NPC ID (string): "innkeeper"
- Scene ID (string): "village_tavern"
- Player Message (string): "Hello!"

Outputs:
- On Success (FBigPlayAction)
- On Failure (string)
```

**Quick Hint:**
```
Inputs:
- Scene ID (string): "village_tavern"
- Quest Stage (string): "find_herbs" (optional)

Outputs:
- On Success (FBigPlayAction)
- On Failure (string)
```

### Builders (Create Data Structures)

**Make NPC:**
```
Inputs:
- ID: "innkeeper"
- Name: "Bob"
- Role: "innkeeper"
- Mood: "cheerful"
- Location: "behind_counter"

Output: FBigPlayNPC
```

**Make Player:**
```
Inputs:
- Name: "Hero"
- Location: "village_tavern"
- Quest Stage: "seeking_info" (optional)
- Reputation: "stranger" (optional)

Output: FBigPlayPlayer
```

**Make World:**
```
Inputs:
- Time Of Day: "evening"
- Weather: "rainy" (optional)
- Tension Level: "calm" (optional)

Output: FBigPlayWorld
```

**Make Game State:**
```
Inputs:
- Scene ID: "village_tavern"
- NPCs: Array of FBigPlayNPC
- Player: FBigPlayPlayer
- World: FBigPlayWorld

Output: FBigPlayGameState
```

### Helpers (Extract Data)

**Get Dialogue Text:**
```
Input: FBigPlayAction
Output: string (first dialogue line)
```

**Get Tone Emoji:**
```
Input: FBigPlayAction
Output: string (😊, 😠, 😢, etc.)
```

**Is High Priority:**
```
Input: FBigPlayAction
Output: bool
```

---

## 🔧 API Reference

### UBigPlayClient

**Initialization:**
```cpp
void Initialize(const FString& BaseURL);
// Example: client->Initialize("http://localhost:8000");
```

**Simple Actions:**
```cpp
// Get NPC dialogue
void GetNPCDialogue(
    const FString& NPCId,
    const FString& SceneId,
    const FString& PlayerMessage,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure
);

// Get hint for stuck player
void GetHint(
    const FString& SceneId,
    const FString& QuestStage,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure
);

// Get world reaction
void GetWorldReaction(
    const FString& SceneId,
    const FString& TimeOfDay,
    const FString& Weather,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure
);
```

**Advanced Action:**
```cpp
void SendGameAction(
    const FBigPlayGameState& GameState,
    const FBigPlayPlayerIntent& PlayerIntent,
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure
);
```

**System:**
```cpp
void CheckHealth(
    FOnActionReceived OnSuccess,
    FOnRequestFailed OnFailure
);

FString GetBaseURL() const;
void SetTimeout(float TimeoutSeconds); // Default: 30s
```

### Data Models

**FBigPlayAction** (Response):
```cpp
EBigPlayActionMode Mode;           // npc_dialogue | hint | world_reaction | dev_debug
EBigPlayPriority Priority;         // low | medium | high
TArray<FBigPlayDialogueLine> Dialogue;
FString HintText;
FBigPlayWorldReaction WorldReaction;
FString DebugNotes;

// Helpers
bool HasDialogue() const;
FString GetDialogueText() const;   // First dialogue line
FString GetNPCId() const;          // First NPC ID
bool HasHint() const;
bool HasWorldReaction() const;
```

**FBigPlayDialogueLine:**
```cpp
FString NPCId;
FString Text;
FString Tone;

FString GetToneEmoji() const;      // Returns 😊, 😠, 😢, etc.
```

**FBigPlayNPC:**
```cpp
FString Id;
FString Name;
FString Role;
FString Mood;
FString Location;
TMap<FString, bool> Flags;
```

**FBigPlayPlayer:**
```cpp
FString Name;
FString Location;
FString QuestStage;
FString Reputation;
TMap<FString, int32> Traits;      // e.g., "charisma": 7
TArray<FString> InventoryTags;
```

**FBigPlayWorld:**
```cpp
FString TimeOfDay;                 // morning | afternoon | evening | night
FString Weather;                   // clear | rainy | stormy | foggy
FString TensionLevel;              // calm | tense | critical
```

**FBigPlayGameState:**
```cpp
FString SceneId;
TArray<FBigPlayNPC> NPCs;
FBigPlayPlayer Player;
FBigPlayWorld World;
TArray<FString> Tags;
```

**FBigPlayPlayerIntent:**
```cpp
EBigPlayIntentType Type;           // TalkToNPC | EnterScene | RequestHint | DebugSummary
FString TargetNPCId;               // Required for TalkToNPC
FString RawInput;                  // Optional player message
```

---

## 🎯 Common Use Cases

### 1. Simple NPC Conversation

```cpp
void ATownNPC::OnPlayerInteract()
{
    UBigPlayClient* Client = UBigPlayBlueprintLibrary::GetBigPlayClient(this);

    Client->GetNPCDialogue(
        NPCId,
        GetCurrentSceneId(),
        "What can you tell me about this place?",
        UBigPlayClient::FOnActionReceived::CreateUObject(this, &ATownNPC::ShowDialogue),
        UBigPlayClient::FOnRequestFailed::CreateUObject(this, &ATownNPC::ShowError)
    );
}

void ATownNPC::ShowDialogue(const FBigPlayAction& Action)
{
    // Update dialogue widget
    DialogueWidget->SetText(FText::FromString(Action.GetDialogueText()));
    DialogueWidget->SetNPCName(FText::FromString(Action.GetNPCId()));

    // Animate NPC based on tone
    FString Tone = Action.Dialogue[0].Tone;
    if (Tone == "happy")
    {
        PlayAnimation(HappyAnim);
    }
    else if (Tone == "angry")
    {
        PlayAnimation(AngryAnim);
    }
}
```

### 2. Context-Aware Hint System

```cpp
void APlayerController::RequestHint()
{
    UBigPlayClient* Client = UBigPlayBlueprintLibrary::GetBigPlayClient(this);

    Client->GetHint(
        GetCurrentSceneId(),
        GetCurrentQuestStage(),
        UBigPlayClient::FOnActionReceived::CreateUObject(this, &APlayerController::ShowHint),
        UBigPlayClient::FOnRequestFailed::CreateUObject(this, &APlayerController::ShowError)
    );
}

void APlayerController::ShowHint(const FBigPlayAction& Action)
{
    if (Action.HasHint())
    {
        // Show hint UI
        HintWidget->SetText(FText::FromString(Action.HintText));
        HintWidget->SetVisibility(ESlateVisibility::Visible);
    }
}
```

### 3. Dynamic World Reactions

```cpp
void ASceneManager::OnPlayerEnterScene(const FString& SceneId)
{
    UBigPlayClient* Client = UBigPlayBlueprintLibrary::GetBigPlayClient(this);

    FString TimeOfDay = GetCurrentTimeOfDay();  // "morning", "afternoon", etc.
    FString Weather = GetCurrentWeather();      // "rainy", "clear", etc.

    Client->GetWorldReaction(
        SceneId,
        TimeOfDay,
        Weather,
        UBigPlayClient::FOnActionReceived::CreateUObject(this, &ASceneManager::OnWorldReaction),
        UBigPlayClient::FOnRequestFailed::CreateUObject(this, &ASceneManager::ShowError)
    );
}

void ASceneManager::OnWorldReaction(const FBigPlayAction& Action)
{
    if (Action.HasWorldReaction())
    {
        // Show ambient description
        AmbientTextWidget->SetText(FText::FromString(Action.WorldReaction.Description));

        // Apply flag changes
        for (const auto& Pair : Action.WorldReaction.FlagChanges)
        {
            SetGameFlag(Pair.Key, Pair.Value);
        }
    }
}
```

### 4. Complete Game State (Advanced)

```cpp
void AGameMode::ProcessPlayerAction(const FString& Action)
{
    // Build complete game state
    FBigPlayGameState GameState = BuildCurrentGameState();

    FBigPlayPlayerIntent Intent;
    Intent.Type = EBigPlayIntentType::TalkToNPC;
    Intent.TargetNPCId = SelectedNPCId;
    Intent.RawInput = Action;

    UBigPlayClient* Client = UBigPlayBlueprintLibrary::GetBigPlayClient(this);
    Client->SendGameAction(
        GameState,
        Intent,
        UBigPlayClient::FOnActionReceived::CreateUObject(this, &AGameMode::OnActionReceived),
        UBigPlayClient::FOnRequestFailed::CreateUObject(this, &AGameMode::ShowError)
    );
}

FBigPlayGameState AGameMode::BuildCurrentGameState()
{
    FBigPlayGameState State;
    State.SceneId = CurrentSceneId;

    // Add all NPCs in scene
    for (ANPCCharacter* NPC : GetNPCsInScene())
    {
        State.NPCs.Add(NPC->GetBigPlayState());
    }

    // Set player state
    State.Player.Name = PlayerName;
    State.Player.Location = CurrentSceneId;
    State.Player.QuestStage = CurrentQuestStage;
    State.Player.Reputation = GetPlayerReputation();
    State.Player.Traits = GetPlayerTraits();
    State.Player.InventoryTags = GetInventoryTags();

    // Set world state
    State.World.TimeOfDay = GetTimeOfDay();
    State.World.Weather = GetWeather();
    State.World.TensionLevel = GetTensionLevel();

    return State;
}
```

---

## 🐛 Troubleshooting

### Plugin Not Showing Up

**Problem:** BigPlay Engine doesn't appear in Plugins menu

**Solutions:**
1. Verify plugin is in `YourProject/Plugins/BigPlayEngine/`
2. Check `BigPlayEngine.uplugin` exists
3. Restart Unreal Editor
4. Generate Visual Studio project files (right-click `.uproject`)

### Compile Errors

**Problem:** Cannot find `BigPlayClient.h` or similar

**Solutions:**
1. Verify `Http`, `Json`, and `JsonUtilities` are in `YourProject.Build.cs`:
   ```csharp
   PublicDependencyModuleNames.AddRange(new string[] {
       "Http",
       "Json",
       "JsonUtilities"
   });
   ```
2. Regenerate project files
3. Clean and rebuild solution

### Runtime Errors - HTTP Request Failed

**Problem:** `HTTP request failed: http://localhost:8000/elle/game/action`

**Solutions:**
1. Verify BigPlay server is running:
   ```bash
   curl http://localhost:8000/health
   ```
2. Check firewall settings
3. Update base URL if server is on different port/host:
   ```cpp
   Client->Initialize("http://your-server:port");
   ```

### JSON Parsing Errors

**Problem:** `Failed to parse JSON response`

**Solutions:**
1. Check server response format (should match BigPlay API spec)
2. Enable verbose logging:
   ```cpp
   // In BigPlayClient.cpp, change LogTemp to LogVerbosity
   UE_LOG(LogTemp, Verbose, TEXT("Response: %s"), *ResponseBody);
   ```
3. Verify Content-Type header is `application/json`

---

## 📊 Performance Tips

### 1. Cache Client Instance

Don't create new client for every request:
```cpp
// ❌ Bad - creates client every time
void TalkToNPC() {
    UBigPlayClient* Client = NewObject<UBigPlayClient>(this);
    Client->GetNPCDialogue(...);
}

// ✅ Good - reuse client
UPROPERTY()
UBigPlayClient* CachedClient;

void BeginPlay() {
    CachedClient = NewObject<UBigPlayClient>(this);
    CachedClient->Initialize("http://localhost:8000");
}

void TalkToNPC() {
    CachedClient->GetNPCDialogue(...);
}
```

### 2. Use Blueprint Library for Convenience

For quick prototyping, use `UBigPlayBlueprintLibrary::GetBigPlayClient()` which handles caching automatically.

### 3. Adjust Timeout for Slow Connections

```cpp
Client->SetTimeout(60.0f);  // 60 seconds for slow LLM providers
```

### 4. Handle Errors Gracefully

Always provide OnFailure callback:
```cpp
void OnRequestFailed(const FString& ErrorMessage)
{
    // Show fallback dialogue
    ShowGenericNPCResponse();

    // Log for debugging
    UE_LOG(LogTemp, Error, TEXT("BigPlay error: %s"), *ErrorMessage);
}
```

---

## 🚀 Next Steps

1. **Try the example:** Open `Content/Examples/ExampleNPCInteraction.cpp`
2. **Build your first NPC:** Follow the Quick Start guide
3. **Explore advanced features:** See Complete C++ Example
4. **Read full docs:** Check out main [README.md](../README.md)

---

## 📞 Support

- **Documentation:** [Full API Reference](../API_REFERENCE.md)
- **Tutorials:** [Complete Tutorials](../TUTORIALS.md)
- **Issues:** [GitHub Issues](https://github.com/yourusername/bigplay/issues)

---

**Built with ❤️ for Unreal developers who want living NPCs without complexity.**

**Version:** 1.0.0
**Last Updated:** 2025-11-17
**Unreal Engine:** 5.0+
