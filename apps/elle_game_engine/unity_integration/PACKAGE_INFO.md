# Elle Game Engine - Unity Integration Package

**Version**: 1.0.0
**Unity Version**: 2018.3+ (requires .NET 4.x for async/await)
**Elle API Version**: 0.1.0

## Package Contents

### Core Files (Required)

#### ElleModels.cs (~240 lines)
Complete C# data models matching Elle's Python API:

- **Game State Models**: `NPCState`, `PlayerState`, `WorldState`, `GameStateSnapshot`
- **Intent Models**: `PlayerIntent`, `PlayerIntentType` (enum)
- **Response Models**: `ElleGameAction`, `DialogueLine`, `WorldChange`, `ActionMode` (enum)
- **Utility Methods**: `GetToneEmoji()`, `HasDialogue`, `HasWorldChanges`

All models use `[Serializable]` attribute for Unity's JsonUtility.

#### ElleClient.cs (~225 lines)
HTTP client for calling Elle Game Engine API:

**Features**:
- Async/await using UnityWebRequest (Unity native)
- Automatic retry with exponential backoff (3 attempts by default)
- Configurable timeout (10 seconds default)
- Debug logging mode
- Health check endpoint
- Error handling with proper Unity exceptions

**Main Methods**:
- `GetNPCDialogue()` - Get NPC dialogue
- `GetHint()` - Request player hint
- `GetWorldReaction()` - Get environmental response
- `GetAction()` - Generic method for full control
- `CheckHealth()` - Verify service availability

**Configuration** (via Inspector):
- Base URL (default: http://localhost:8000)
- Timeout seconds (default: 10)
- Max retries (default: 3)
- Debug mode (default: false)

### Example Files (Optional)

#### ExampleNPCInteraction.cs (~230 lines)
Complete working example of NPC dialogue integration:

**Features**:
- UI integration (dialogue panel, input field, buttons)
- Loading states and error handling
- Complete game state construction
- Tone emoji display
- Multiple dialogue modes (NPC, hints, world reactions)
- Public API for other scripts

**Demonstrates**:
- How to build rich `GameStateSnapshot` with NPCs, player, world
- Async dialogue requests with loading indicators
- Error handling and fallback dialogue
- UI display with tone emojis
- Reusable patterns for your own implementation

**UI Requirements** (assign in Inspector):
- Dialogue Panel (GameObject)
- NPC Name Text (TextMeshProUGUI)
- Dialogue Text (TextMeshProUGUI)
- Tone Emoji (TextMeshProUGUI)
- Player Input Field (TMP_InputField)
- Send Button (Button)
- Loading Indicator (GameObject, optional)

### Documentation

#### README_UNITY.md (~450 lines)
Comprehensive integration guide:

**Sections**:
1. Quick Start (5-minute setup)
2. Complete Example walkthrough
3. API Reference (all methods + models)
4. Advanced Usage patterns
5. Configuration options
6. Performance optimization
7. Troubleshooting guide
8. Best practices

#### PACKAGE_INFO.md (this file)
Package manifest and installation guide.

## Installation

### Method 1: Direct Copy (Recommended)

1. Copy all `.cs` files to your Unity project:
   ```
   YourUnityProject/Assets/Scripts/Elle/
   ├── ElleClient.cs
   ├── ElleModels.cs
   └── ExampleNPCInteraction.cs  (optional)
   ```

2. Start Elle service:
   ```bash
   python -m apps.elle_game_engine.service
   ```

3. Add `ElleClient` component to a GameObject in your scene

4. Start using the API (see Quick Start in README_UNITY.md)

### Method 2: Unity Package (Future)

Unity Package Manager support coming soon.

## Requirements

### Unity
- **Version**: 2018.3 or higher
- **Scripting Runtime**: .NET 4.x (required for async/await)
- **API Compatibility Level**: .NET 4.x or .NET Standard 2.0

### Set .NET 4.x (if not already set):
1. `Edit` → `Project Settings` → `Player`
2. `Other Settings` → `Scripting Runtime Version` → `.NET 4.x`
3. Restart Unity

### Elle Service
- **Python**: 3.8+
- **Dependencies**: `fastapi`, `uvicorn`, `pydantic`
- **LLM Provider** (optional but recommended):
  - OpenAI API key (for GPT-4o-mini)
  - Anthropic API key (for Claude)
  - Ollama (for free local models)

### Unity Packages (Optional)
- **TextMeshPro**: Recommended for better text rendering (free via Package Manager)
- **Standard UI**: Already included in Unity

## File Sizes

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| ElleModels.cs | ~240 | 8.5 KB | Data models |
| ElleClient.cs | ~225 | 7.8 KB | HTTP client |
| ExampleNPCInteraction.cs | ~230 | 8.2 KB | Example (optional) |
| README_UNITY.md | ~450 | 16 KB | Documentation |
| **Total** | ~1,145 | ~40 KB | Complete package |

Minimal install (core only): **~16 KB** (2 files)

## Version History

### 1.0.0 (November 2025)
- Initial release
- Complete API coverage (dialogue, hints, world reactions)
- Async/await support with UnityWebRequest
- Automatic retry and error handling
- Full working example with UI
- Comprehensive documentation

## What's Next?

### Planned Features
- **Dialogue History**: Track conversation context across multiple exchanges
- **Response Caching**: Built-in cache for common queries
- **Batch Requests**: Send multiple requests in parallel
- **Custom Serialization**: Support for Newtonsoft.Json
- **Unity Events**: UnityEvent callbacks for dialogue received
- **Addressable Integration**: Support for addressable assets

### Requested by Community
- Voice synthesis integration (TTS)
- Localization support (i18n)
- Visual scripting (Bolt/Unity Visual Scripting) nodes
- Timeline integration for cinematic sequences

## Support

### Documentation
- **Quick Start**: See README_UNITY.md section "Quick Start (5 Minutes)"
- **API Reference**: See README_UNITY.md section "API Reference"
- **Examples**: See ExampleNPCInteraction.cs for working code
- **Troubleshooting**: See README_UNITY.md section "Troubleshooting"

### Getting Help
1. Check README_UNITY.md troubleshooting section
2. Enable Debug Mode in ElleClient Inspector
3. Check Elle service logs: `python -m apps.elle_game_engine.service`
4. Verify service health: `curl http://localhost:8000/health`

### Common Issues
- **"Cannot reach Elle service"** → Service not running or wrong URL
- **"Request timeout"** → Increase timeout or switch to faster LLM
- **"JSON deserialization failed"** → Check .NET 4.x is enabled
- **Build errors** → Verify Unity 2018.3+ and .NET 4.x runtime

## License

See main repository LICENSE file.

---

**Ready to add intelligent narrative to your Unity game in 5 minutes.**

Quick Start: Copy files → Add component → Call `GetNPCDialogue()` → Done!
