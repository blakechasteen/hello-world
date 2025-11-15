# Elle Game Engine Integration

**Status**: MVP Complete
**Version**: 0.1.0

A self-contained microservice that provides LLM-driven narrative intelligence for video games. Built on Elle's core philosophy but specialized for game development.

## Philosophy

Elle treats the LLM as a **"plugin brain"** for game systems, not a monolithic overlord:

- **LLM is policy, not control** - Elle suggests actions; the game engine decides
- **Engine-agnostic** - Works with Unity, Godot, Unreal, or any engine via HTTP/JSON
- **Calm and grounded** - Elle enhances gameplay without overwhelming the player
- **One clear action** - No walls of text, just precise narrative responses

## What Elle Does

Elle provides four types of narrative intelligence:

1. **NPC Dialogue** - Contextual, believable character responses
2. **Hints** - Gentle, non-spoilery guidance when players are stuck
3. **World Reactions** - Ambient storytelling and environmental responses
4. **Dev Debug** - Developer-facing narrative analysis and suggestions

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Install for development (from repository root)
pip install -e .
```

### Run the Service

```bash
# From repository root
python -m apps.elle_game_engine.service

# Or with uvicorn directly
uvicorn apps.elle_game_engine.service:app --reload --port 8000
```

Service will start on `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Make Your First Request

```bash
curl -X POST "http://localhost:8000/elle/game/action" \
  -H "Content-Type: application/json" \
  -d '{
    "game_state": {
      "scene_id": "village_square",
      "npcs": [
        {
          "id": "innkeeper",
          "name": "Bob",
          "role": "innkeeper",
          "mood": "nervous",
          "location": "inn"
        }
      ],
      "player": {
        "name": "Hero",
        "location": "village_square"
      },
      "world": {
        "time_of_day": "afternoon"
      }
    },
    "player_intent": {
      "type": "talk_to_npc",
      "target_npc_id": "innkeeper",
      "raw_input": "Hello!"
    }
  }'
```

**Response**:

```json
{
  "mode": "npc_dialogue",
  "priority": "medium",
  "dialogue": [
    {
      "npc_id": "innkeeper",
      "text": "Greetings, traveler. How can I help you?",
      "tone": "neutral"
    }
  ],
  "hint_text": null,
  "world_reaction": null,
  "debug_notes": "Test NPC dialogue from DummyLLMClient"
}
```

## LLM Provider Configuration

By default, Elle uses a `DummyLLMClient` for testing. To use real LLM providers, configure via environment variables:

### Using Anthropic Claude (Recommended)

```bash
# Install Anthropic SDK
pip install anthropic

# Set environment variables
export ELLE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-api-key-here
export ELLE_LLM_MODEL=claude-3-5-sonnet-20241022  # optional, this is default

# Start service
python -m apps.elle_game_engine.service
```

**Models**:
- `claude-3-5-sonnet-20241022` - Best quality, slower, more expensive
- `claude-3-haiku-20240307` - Faster, cheaper, good quality

### Using OpenAI

```bash
# Install OpenAI SDK
pip install openai

# Set environment variables
export ELLE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-api-key-here
export ELLE_LLM_MODEL=gpt-4o-mini  # optional, this is default

# Start service
python -m apps.elle_game_engine.service
```

**Models**:
- `gpt-4o-mini` - Fast, cheap, good for games
- `gpt-4o` - Best quality, more expensive
- `gpt-4-turbo` - Good balance

### Using Local Models (Ollama)

```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull llama3.2:3b

# Set environment variables (no API key needed!)
export ELLE_LLM_PROVIDER=local
export ELLE_LLM_MODEL=llama3.2:3b  # or any model you've pulled
export OLLAMA_BASE_URL=http://localhost:11434  # optional, this is default

# Start service
python -m apps.elle_game_engine.service
```

**Recommended Models**:
- `llama3.2:3b` - Fast, lightweight
- `llama3.1:8b` - Better quality, slower
- `mistral:7b` - Good alternative

### Environment Variables Reference

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ELLE_LLM_PROVIDER` | `dummy`, `anthropic`, `openai`, `local` | `dummy` | Which LLM to use |
| `ELLE_LLM_MODEL` | Model name | (varies) | Specific model to use |
| `ANTHROPIC_API_KEY` | API key | - | Required for Anthropic |
| `OPENAI_API_KEY` | API key | - | Required for OpenAI |
| `OLLAMA_BASE_URL` | URL | `http://localhost:11434` | Ollama API endpoint |

### Cost Comparison

**Per 1,000 Game Interactions** (approx):

| Provider | Model | Input Cost | Output Cost | Total (est) |
|----------|-------|------------|-------------|-------------|
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | **$18.00** |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 | **$1.50** |
| OpenAI | GPT-4o | $2.50 | $10.00 | **$12.50** |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | **$0.75** |
| Ollama | Any | $0.00 | $0.00 | **Free** |

*Assumes ~200 input tokens + ~150 output tokens per interaction*

### Recommendations

**For Production Games**:
- Use `gpt-4o-mini` (OpenAI) or `claude-3-haiku` (Anthropic) for cost efficiency
- ~$1-2 per 1,000 players

**For Development/Testing**:
- Use `dummy` (no cost, predictable responses)
- Or `local` with Ollama (free, runs on your machine)

**For High-Quality Narrative**:
- Use `claude-3-5-sonnet` for best storytelling
- Worth the cost for premium games

## API Reference

### POST /elle/game/action

Get narrative action from Elle based on game state and player intent.

**Request Body**:

```json
{
  "game_state": {
    "scene_id": "string",
    "npcs": [
      {
        "id": "string",
        "name": "string",
        "role": "string",
        "mood": "string (optional)",
        "location": "string",
        "flags": {"key": true}
      }
    ],
    "player": {
      "name": "string",
      "location": "string",
      "quest_stage": "string (optional)",
      "reputation": "string (optional)",
      "traits": {"trait": 3},
      "inventory_tags": ["item1", "item2"]
    },
    "world": {
      "time_of_day": "string",
      "weather": "string (optional)",
      "tension_level": "string (optional)"
    },
    "tags": ["tag1", "tag2"]
  },
  "player_intent": {
    "type": "talk_to_npc | enter_scene | request_hint | debug_summary",
    "target_npc_id": "string (required for talk_to_npc)",
    "raw_input": "string (optional)"
  }
}
```

**Response**:

```json
{
  "mode": "npc_dialogue | hint | world_reaction | dev_debug",
  "priority": "low | medium | high",
  "dialogue": [
    {
      "npc_id": "string",
      "text": "string",
      "tone": "string (optional)"
    }
  ],
  "hint_text": "string (optional)",
  "world_reaction": {
    "description": "string",
    "flag_changes": {"flag": true}
  },
  "debug_notes": "string (optional)"
}
```

### GET /health

Health check endpoint.

**Response**:

```json
{
  "status": "healthy",
  "service": "elle_game_engine"
}
```

## Data Contracts

### Player Intent Types

| Type | Description | Required Fields |
|------|-------------|----------------|
| `talk_to_npc` | Player initiating dialogue | `target_npc_id` |
| `enter_scene` | Player entering new location | - |
| `request_hint` | Player asking for guidance | - |
| `debug_summary` | Developer requesting analysis | - |

### Action Modes

| Mode | Description | Returns |
|------|-------------|---------|
| `npc_dialogue` | NPC speaking | `dialogue` list |
| `hint` | Non-spoilery guidance | `hint_text` |
| `world_reaction` | Environmental response | `world_reaction` |
| `dev_debug` | Developer notes | `debug_notes` |

### Priority Levels

- `low` - Ambient, optional content
- `medium` - Standard narrative response
- `high` - Important, time-sensitive content

## Architecture

```
┌─────────────────────────────────────────┐
│  Game Engine (Unity/Godot/Unreal)      │
│  Sends HTTP/JSON requests               │
└────────────────┬────────────────────────┘
                 │ HTTP
                 ↓
┌─────────────────────────────────────────┐
│  FastAPI Service (service.py)           │
│  - Request validation                   │
│  - Response formatting                  │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│  Game Policy (policy.py)                │
│  - Prompt building                      │
│  - LLM calling                          │
│  - Response parsing                     │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│  LLM Client (llm_client.py)             │
│  - DummyClient (testing)                │
│  - OpenAI (future)                      │
│  - Anthropic (future)                   │
│  - Local models (future)                │
└─────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Run all tests
pytest apps/elle_game_engine/tests/ -v

# Run specific test file
pytest apps/elle_game_engine/tests/test_models.py -v

# Run with coverage
pytest apps/elle_game_engine/tests/ --cov=apps.elle_game_engine
```

### Project Structure

```
apps/elle_game_engine/
├── __init__.py           # Package entry point
├── models.py             # Data contracts (GameStateSnapshot, ElleGameAction, etc.)
├── core_prompt.txt       # Elle's game engine personality
├── llm_client.py         # LLM client abstraction
├── policy.py             # Policy engine (prompt building + action parsing)
├── service.py            # FastAPI application
├── tests/
│   ├── test_models.py    # Model validation tests
│   ├── test_policy.py    # Policy engine tests
│   └── test_service.py   # API endpoint tests
└── README.md             # This file
```

## Integration Examples

### Unity (C#)

```csharp
using UnityEngine;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

public class ElleClient : MonoBehaviour
{
    private readonly HttpClient client = new HttpClient();
    private const string ELLE_URL = "http://localhost:8000/elle/game/action";

    public async Task<ElleGameAction> GetAction(GameStateSnapshot state, PlayerIntent intent)
    {
        var request = new { game_state = state, player_intent = intent };
        var json = JsonUtility.ToJson(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await client.PostAsync(ELLE_URL, content);
        var responseJson = await response.Content.ReadAsStringAsync();

        return JsonUtility.FromJson<ElleGameAction>(responseJson);
    }
}
```

### Godot (GDScript)

```gdscript
extends Node

const ELLE_URL = "http://localhost:8000/elle/game/action"

func get_elle_action(game_state: Dictionary, player_intent: Dictionary):
    var request = HTTPRequest.new()
    add_child(request)
    request.connect("request_completed", self, "_on_request_completed")

    var body = JSON.print({
        "game_state": game_state,
        "player_intent": player_intent
    })

    var headers = ["Content-Type: application/json"]
    request.request(ELLE_URL, headers, true, HTTPClient.METHOD_POST, body)

func _on_request_completed(result, response_code, headers, body):
    var json = JSON.parse(body.get_string_from_utf8())
    var action = json.result
    # Process action...
```

### Python (for testing)

```python
import requests

def get_elle_action(game_state, player_intent):
    response = requests.post(
        "http://localhost:8000/elle/game/action",
        json={
            "game_state": game_state,
            "player_intent": player_intent,
        }
    )
    return response.json()
```

## Configuration

### Using Real LLM Providers (Future)

The current implementation uses `DummyLLMClient` for testing. To use real LLM providers:

1. Install provider SDK:
   ```bash
   pip install openai  # or anthropic
   ```

2. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. Update `service.py` to use real client:
   ```python
   llm_client = create_llm_client("openai", api_key=os.getenv("OPENAI_API_KEY"))
   ```

## Roadmap

### Phase 1: MVP (Complete ✅)
- [x] Core data models
- [x] Dummy LLM client
- [x] Policy engine
- [x] FastAPI service
- [x] Comprehensive tests
- [x] Documentation

### Phase 2: Real LLM Integration (Next)
- [ ] OpenAI client implementation
- [ ] Anthropic Claude client
- [ ] Local model support (Ollama)
- [ ] LLM response caching

### Phase 3: Advanced Features
- [ ] Multi-NPC conversations
- [ ] Dialogue history/context
- [ ] Persistent world state
- [ ] Custom prompt templates
- [ ] Fine-tuning support

### Phase 4: Production Hardening
- [ ] Rate limiting
- [ ] Request queuing
- [ ] Monitoring/metrics
- [ ] Error recovery
- [ ] Load testing

## Design Decisions

### Why HTTP/JSON?

- **Universal**: Works with any game engine
- **Simple**: No complex protocols or SDKs
- **Debuggable**: Easy to inspect with curl/Postman
- **Scalable**: Can run service separately from game

### Why Stateless?

- **Simplicity**: No session management complexity
- **Scalability**: Easy to run multiple instances
- **Reliability**: No state corruption issues
- **Testability**: Pure functions, easy to test

### Why One Action Per Request?

- **Focus**: Prevents overwhelming the player
- **Performance**: Faster responses
- **Clarity**: Easier for game to process
- **Quality**: One strong choice beats five weak ones

## Troubleshooting

### Service won't start

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Use different port
uvicorn apps.elle_game_engine.service:app --port 8001
```

### Tests failing

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run with verbose output
pytest apps/elle_game_engine/tests/ -vv
```

### Import errors

```bash
# Install package in editable mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/hello-world"
```

## Contributing

This is part of the larger HoloLoom/Elle ecosystem. See main repository documentation for contribution guidelines.

## License

See main repository LICENSE file.

---

**Built with care for game developers who want narrative intelligence without complexity.**
