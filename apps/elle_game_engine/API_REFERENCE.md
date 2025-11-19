# BigPlay API Reference

**Last Updated**: 2025-11-16
**Version**: 1.0.0

Complete API reference for the BigPlay game engine. All endpoints return JSON unless otherwise specified.

---

## Table of Contents

1. [Authentication](#authentication)
2. [Game Action Endpoint](#game-action-endpoint)
3. [Quest Generation](#quest-generation)
4. [Voice Synthesis](#voice-synthesis)
5. [Session Management](#session-management)
6. [Performance Monitoring](#performance-monitoring)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Webhooks](#webhooks)

---

## Base URL

```
Production:  https://api.bigplay.dev
Development: http://localhost:8000
```

---

## Authentication

BigPlay uses API keys for authentication. Include your API key in the `Authorization` header:

```bash
Authorization: Bearer YOUR_API_KEY
```

### Get API Key

**Endpoint**: `POST /auth/register`

**Request**:
```json
{
  "email": "you@example.com",
  "game_title": "My Awesome Game"
}
```

**Response**:
```json
{
  "api_key": "bp_live_...",
  "expires_at": "2026-01-01T00:00:00Z",
  "tier": "free",
  "quota": {
    "requests_per_day": 10000,
    "requests_per_minute": 60
  }
}
```

---

## Game Action Endpoint

**The core endpoint for NPC interactions.**

### POST /elle/game/action

Process player action and generate NPC response with emotional updates.

**Request**:

```json
{
  "game_state": {
    "scene_id": "tavern_main_room",
    "npcs": [
      {
        "id": "bob_innkeeper",
        "name": "Bob the Innkeeper",
        "role": "innkeeper",
        "emotional_state": {
          "valence": 0.3,
          "arousal": 0.5,
          "dominance": 0.6,
          "trust": 0.5
        },
        "dialogue_history": [
          "Hello traveler! Welcome to the Rusty Mug!",
          "We have the finest ale in the region."
        ],
        "personality_traits": ["friendly", "talkative", "honest"],
        "current_activity": "cleaning_mugs",
        "inventory": ["health_potion", "room_key"],
        "quests_available": ["find_lost_cat", "deliver_message"]
      }
    ],
    "player": {
      "name": "Hero",
      "location": "tavern_main_room",
      "health": 100,
      "level": 5,
      "inventory": ["sword", "shield", "gold_coins"],
      "active_quests": ["slay_the_dragon"]
    },
    "world_state": {
      "time_of_day": "evening",
      "weather": "raining",
      "day_number": 15,
      "global_flags": {
        "dragon_defeated": false,
        "festival_active": false
      }
    },
    "modifiers": {
      "difficulty": "medium",
      "narrative_style": "tolkien",
      "tone": "friendly"
    }
  },
  "player_intent": {
    "type": "talk_to_npc",
    "target_npc_id": "bob_innkeeper",
    "raw_input": "Tell me about the dragon",
    "context": {
      "previous_action": "entered_tavern",
      "player_emotion": "curious"
    }
  }
}
```

**Required Fields**:
- `game_state.scene_id`: Current scene identifier
- `game_state.npcs`: Array of NPCs present (at least 1)
- `game_state.npcs[].id`: Unique NPC identifier
- `game_state.npcs[].emotional_state`: PAD emotion model
- `player_intent.type`: Action type
- `player_intent.raw_input`: Player's input

**Optional Fields**:
- `game_state.player`: Player state (recommended)
- `game_state.world_state`: World context (improves quality)
- `game_state.npcs[].dialogue_history`: Recent conversation (enables memory)
- `game_state.npcs[].personality_traits`: Character traits
- `game_state.modifiers`: Response modifiers
- `player_intent.target_npc_id`: Specific NPC to address

**Response**:

```json
{
  "action_type": "dialogue",
  "content": {
    "npc_id": "bob_innkeeper",
    "npc_name": "Bob the Innkeeper",
    "npc_dialogue": "Ah, the dragon! *leans in conspiratorially* They say it lives in the mountains to the north. Many brave souls have tried to slay it, but none have returned. Are you planning to face it?",
    "voice_audio_url": "https://cdn.bigplay.dev/audio/12345.mp3",
    "suggested_player_responses": [
      "Yes, I'm going to defeat it",
      "Tell me more about the dragon",
      "Do you know anyone who's fought it?",
      "Maybe I should prepare first"
    ],
    "narration": "*Bob's eyes widen with a mix of concern and admiration*"
  },
  "updated_npcs": [
    {
      "id": "bob_innkeeper",
      "emotional_state": {
        "valence": 0.2,
        "arousal": 0.7,
        "dominance": 0.5,
        "trust": 0.6
      },
      "relationship_change": {
        "respect": +5,
        "friendship": +2
      }
    }
  ],
  "world_updates": {
    "dragon_mentioned": true,
    "bob_knows_player_intent": true
  },
  "triggered_events": [],
  "metadata": {
    "processing_time_ms": 145,
    "llm_provider": "anthropic",
    "llm_model": "claude-3-5-sonnet-20241022",
    "tokens_used": {
      "input": 450,
      "output": 85
    },
    "cost_usd": 0.0025,
    "cache_hit": false
  }
}
```

**Action Types**:

| Type | Description | Content Fields |
|------|-------------|----------------|
| `dialogue` | NPC speaks to player | `npc_dialogue`, `voice_audio_url` |
| `world_description` | Narration/scene description | `description`, `sensory_details` |
| `quest_offered` | NPC offers quest | `quest` (full quest object) |
| `combat` | Combat initiated | `combat_result`, `damage_dealt` |
| `hint` | Gameplay hint | `hint_text`, `hint_type` |
| `error` | Something went wrong | `error_message`, `error_code` |

**Emotional State (PAD Model)**:

```json
{
  "valence": 0.0,     // -1.0 (negative) to +1.0 (positive)
  "arousal": 0.5,     // 0.0 (calm) to 1.0 (excited)
  "dominance": 0.5,   // 0.0 (submissive) to 1.0 (dominant)
  "trust": 0.5        // 0.0 (distrust) to 1.0 (trust)
}
```

**Player Intent Types**:

- `talk_to_npc`: General conversation
- `ask_question`: Specific question
- `give_item`: Transfer item to NPC
- `attack`: Initiate combat
- `examine`: Look at NPC/object
- `trade`: Open trading interface
- `use_item`: Use item on/with NPC

**Example cURL**:

```bash
curl -X POST "https://api.bigplay.dev/elle/game/action" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "game_state": {
      "scene_id": "tavern",
      "npcs": [{
        "id": "bob",
        "name": "Bob",
        "role": "innkeeper",
        "emotional_state": {
          "valence": 0.3,
          "arousal": 0.5,
          "dominance": 0.6,
          "trust": 0.5
        }
      }]
    },
    "player_intent": {
      "type": "talk_to_npc",
      "target_npc_id": "bob",
      "raw_input": "Hello!"
    }
  }'
```

---

## Quest Generation

### POST /elle/quest/generate

Generate dynamic quest based on game context.

**Request**:

```json
{
  "difficulty": "medium",
  "context": {
    "player_level": 5,
    "player_class": "warrior",
    "world_state": {
      "region": "northern_mountains",
      "completed_quests": ["tutorial_quest", "first_hunt"]
    },
    "available_npcs": ["bob_innkeeper", "thora_blacksmith", "aldric_wizard"],
    "player_emotions": {
      "courage": 0.7,
      "greed": 0.3
    },
    "preferred_quest_types": ["fetch", "kill"],
    "max_objectives": 5
  }
}
```

**Response**:

```json
{
  "quest_id": "quest_12345",
  "title": "The Lost Heirloom",
  "description": "Bob's family heirloom was stolen by bandits. He believes they're hiding in the old ruins to the east.",
  "quest_type": "fetch",
  "difficulty": "medium",
  "estimated_time_minutes": 30,
  "objectives": [
    {
      "id": "obj_1",
      "description": "Travel to the old ruins",
      "type": "reach",
      "target": "old_ruins",
      "optional": false
    },
    {
      "id": "obj_2",
      "description": "Defeat the bandit leader",
      "type": "kill",
      "target": "bandit_leader",
      "target_count": 1,
      "optional": false
    },
    {
      "id": "obj_3",
      "description": "Retrieve Bob's heirloom",
      "type": "collect",
      "target": "golden_locket",
      "target_count": 1,
      "optional": false
    },
    {
      "id": "obj_4",
      "description": "Return the heirloom to Bob",
      "type": "return",
      "target": "bob_innkeeper",
      "optional": false
    },
    {
      "id": "obj_5",
      "description": "Spare the bandit leader (moral choice)",
      "type": "choice",
      "optional": true
    }
  ],
  "rewards": {
    "experience": 500,
    "gold": 150,
    "items": ["rare_sword"],
    "reputation": {
      "bob_innkeeper": 50,
      "village": 20
    }
  },
  "branches": [
    {
      "trigger": "obj_5_completed",
      "choice_id": "spare_or_kill",
      "choices": [
        {
          "id": "spare",
          "text": "Spare the bandit leader",
          "consequences": {
            "morality": +10,
            "reputation_village": -5,
            "unlocks": ["bandit_redemption_quest"]
          }
        },
        {
          "id": "kill",
          "text": "Defeat the bandit leader",
          "consequences": {
            "experience": +100,
            "reputation_village": +10
          }
        }
      ]
    }
  ],
  "prerequisite_quests": [],
  "unlocks_quests": ["dragon_investigation"],
  "fail_conditions": [
    "player_death",
    "heirloom_destroyed",
    "time_limit_exceeded"
  ],
  "time_limit_hours": 48,
  "metadata": {
    "generated_at": "2025-11-16T18:30:00Z",
    "generator_version": "1.0.0",
    "processing_time_ms": 280
  }
}
```

**Difficulty Levels**:
- `trivial`: Level 1-2, 1-2 objectives, 10-15 minutes
- `easy`: Level 1-5, 2-3 objectives, 15-20 minutes
- `medium`: Level 5-10, 3-5 objectives, 20-40 minutes
- `hard`: Level 10-20, 5-7 objectives, 40-60 minutes
- `epic`: Level 20+, 7+ objectives, 60+ minutes

---

## Voice Synthesis

### POST /elle/voice/synthesize

Convert NPC dialogue to speech audio.

**Request**:

```json
{
  "text": "Hello traveler! Welcome to the Rusty Mug!",
  "voice_config": {
    "npc_id": "bob_innkeeper",
    "gender": "male",
    "age": "adult",
    "accent": "british",
    "emotion": "friendly",
    "speaking_rate": 1.0,
    "pitch": 0.0
  },
  "output_format": "mp3",
  "quality": "high"
}
```

**Response**:

```json
{
  "audio_url": "https://cdn.bigplay.dev/audio/12345.mp3",
  "duration_seconds": 3.2,
  "file_size_bytes": 51200,
  "format": "mp3",
  "sample_rate": 24000,
  "voice_provider": "elevenlabs",
  "voice_id": "voice_12345",
  "cost_usd": 0.0008,
  "cached": false,
  "metadata": {
    "processing_time_ms": 450,
    "generated_at": "2025-11-16T18:30:00Z"
  }
}
```

**Voice Providers**:
- `elevenlabs`: Premium quality (recommended)
- `openai_tts`: Good quality, fast
- `piper`: Local, free (requires self-hosting)

**Supported Formats**:
- `mp3`: Good quality, widely supported (default)
- `wav`: Uncompressed, high quality
- `ogg`: Open format, good compression
- `flac`: Lossless, larger files

**Quality Levels**:
- `low`: 16kHz, faster generation
- `medium`: 22kHz, balanced (default)
- `high`: 24kHz, premium quality

**Emotion Modifiers**:
- `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `neutral`

---

## Session Management

### POST /elle/session/create

Create persistent NPC memory session.

**Request**:

```json
{
  "session_id": "player123_bob",
  "npc_id": "bob_innkeeper",
  "player_id": "player123",
  "initial_context": {
    "relationship_level": 0,
    "met_before": false,
    "shared_memories": []
  }
}
```

**Response**:

```json
{
  "session_id": "player123_bob",
  "created_at": "2025-11-16T18:30:00Z",
  "expires_at": "2025-12-16T18:30:00Z",
  "status": "active"
}
```

### POST /elle/session/update

Update session with new interaction.

**Request**:

```json
{
  "session_id": "player123_bob",
  "interaction": {
    "player_input": "Tell me about the dragon",
    "npc_response": "The dragon lives in the northern mountains...",
    "emotional_state_before": {
      "valence": 0.3,
      "arousal": 0.5,
      "dominance": 0.6,
      "trust": 0.5
    },
    "emotional_state_after": {
      "valence": 0.2,
      "arousal": 0.7,
      "dominance": 0.5,
      "trust": 0.6
    },
    "relationship_change": 5,
    "timestamp": "2025-11-16T18:30:00Z"
  }
}
```

**Response**:

```json
{
  "session_id": "player123_bob",
  "total_interactions": 15,
  "relationship_level": 75,
  "key_memories": [
    "player_asked_about_dragon",
    "player_helped_find_cat",
    "player_bought_ale"
  ],
  "updated_at": "2025-11-16T18:30:00Z"
}
```

### GET /elle/session/{session_id}

Retrieve session history.

**Response**:

```json
{
  "session_id": "player123_bob",
  "npc_id": "bob_innkeeper",
  "player_id": "player123",
  "created_at": "2025-11-01T10:00:00Z",
  "last_interaction": "2025-11-16T18:30:00Z",
  "total_interactions": 15,
  "relationship_level": 75,
  "emotional_trajectory": [
    {"timestamp": "2025-11-01T10:00:00Z", "valence": 0.0, "trust": 0.3},
    {"timestamp": "2025-11-16T18:30:00Z", "valence": 0.2, "trust": 0.6}
  ],
  "conversation_history": [
    {
      "timestamp": "2025-11-16T18:30:00Z",
      "player_input": "Tell me about the dragon",
      "npc_response": "The dragon lives in the northern mountains..."
    }
  ],
  "key_memories": [
    "player_asked_about_dragon",
    "player_helped_find_cat"
  ]
}
```

---

## Performance Monitoring

### GET /elle/metrics

Get performance metrics.

**Response**:

```json
{
  "uptime_seconds": 86400,
  "total_requests": 150000,
  "requests_per_second": 1.74,
  "average_latency_ms": 145,
  "p50_latency_ms": 120,
  "p95_latency_ms": 280,
  "p99_latency_ms": 450,
  "cache_hit_rate": 0.65,
  "llm_provider_stats": {
    "anthropic": {
      "requests": 50000,
      "avg_latency_ms": 340,
      "tokens_consumed": 25000000,
      "cost_usd": 125.50
    },
    "openai": {
      "requests": 30000,
      "avg_latency_ms": 280,
      "tokens_consumed": 15000000,
      "cost_usd": 150.00
    },
    "ollama": {
      "requests": 70000,
      "avg_latency_ms": 80,
      "tokens_consumed": 0,
      "cost_usd": 0.00
    }
  },
  "error_rate": 0.02,
  "active_sessions": 1200
}
```

### GET /health

Health check endpoint.

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "services": {
    "llm": "operational",
    "database": "operational",
    "cache": "operational",
    "voice": "operational"
  }
}
```

---

## Error Handling

All errors follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: game_state.npcs",
    "details": {
      "missing_fields": ["game_state.npcs"],
      "provided_fields": ["game_state.scene_id", "player_intent"]
    },
    "request_id": "req_12345",
    "timestamp": "2025-11-16T18:30:00Z",
    "documentation_url": "https://docs.bigplay.dev/errors/invalid_request"
  }
}
```

**Error Codes**:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request or missing required fields |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | API key lacks permission for this endpoint |
| `NOT_FOUND` | 404 | Resource not found (session, NPC, etc.) |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `LLM_ERROR` | 500 | LLM provider error |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

**Example Error Response**:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded: 60 requests per minute",
    "details": {
      "limit": 60,
      "period": "minute",
      "current_usage": 62,
      "reset_at": "2025-11-16T18:31:00Z"
    },
    "request_id": "req_12345",
    "timestamp": "2025-11-16T18:30:00Z"
  }
}
```

---

## Rate Limiting

Rate limits vary by tier:

| Tier | Requests/Minute | Requests/Day | Concurrent Sessions |
|------|-----------------|--------------|---------------------|
| **Free** | 60 | 10,000 | 10 |
| **Pro** | 300 | 100,000 | 100 |
| **Enterprise** | 1,000 | 1,000,000 | 1,000 |

**Rate Limit Headers**:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1700000000
```

**429 Response**:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 60,
      "period": "minute",
      "reset_at": "2025-11-16T18:31:00Z"
    }
  }
}
```

**Best Practices**:
- Implement exponential backoff on 429 responses
- Cache responses when possible
- Batch requests where applicable
- Monitor `X-RateLimit-Remaining` header
- Upgrade tier if consistently hitting limits

---

## Webhooks

Configure webhooks to receive real-time events.

### POST /webhooks/register

Register webhook URL.

**Request**:

```json
{
  "url": "https://yourgame.com/webhooks/bigplay",
  "events": [
    "quest.completed",
    "npc.relationship_changed",
    "player.achievement_unlocked"
  ],
  "secret": "webhook_secret_12345"
}
```

**Response**:

```json
{
  "webhook_id": "wh_12345",
  "url": "https://yourgame.com/webhooks/bigplay",
  "events": ["quest.completed", "npc.relationship_changed"],
  "status": "active",
  "created_at": "2025-11-16T18:30:00Z"
}
```

### Webhook Events

**quest.completed**:

```json
{
  "event": "quest.completed",
  "timestamp": "2025-11-16T18:30:00Z",
  "data": {
    "quest_id": "quest_12345",
    "player_id": "player123",
    "title": "The Lost Heirloom",
    "rewards": {
      "experience": 500,
      "gold": 150
    },
    "completion_time_minutes": 28
  }
}
```

**npc.relationship_changed**:

```json
{
  "event": "npc.relationship_changed",
  "timestamp": "2025-11-16T18:30:00Z",
  "data": {
    "npc_id": "bob_innkeeper",
    "player_id": "player123",
    "old_level": 70,
    "new_level": 75,
    "change": 5,
    "trigger": "completed_quest"
  }
}
```

**Webhook Signature**:

Verify webhook authenticity using HMAC-SHA256:

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

**Headers**:
```
X-BigPlay-Signature: sha256=...
X-BigPlay-Event: quest.completed
X-BigPlay-Delivery-ID: delivery_12345
```

---

## SDK Examples

### Python

```python
import httpx
import asyncio

class BigPlayClient:
    def __init__(self, api_key: str, base_url: str = "https://api.bigplay.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def process_action(
        self,
        scene_id: str,
        npcs: list,
        player_intent: dict
    ) -> dict:
        """Process game action."""
        response = await self.client.post(
            f"{self.base_url}/elle/game/action",
            json={
                "game_state": {
                    "scene_id": scene_id,
                    "npcs": npcs
                },
                "player_intent": player_intent
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

# Usage
client = BigPlayClient(api_key="bp_live_...")

action = await client.process_action(
    scene_id="tavern",
    npcs=[{
        "id": "bob",
        "name": "Bob",
        "role": "innkeeper",
        "emotional_state": {
            "valence": 0.3,
            "arousal": 0.5,
            "dominance": 0.6,
            "trust": 0.5
        }
    }],
    player_intent={
        "type": "talk_to_npc",
        "target_npc_id": "bob",
        "raw_input": "Hello!"
    }
)

print(action["content"]["npc_dialogue"])
```

### JavaScript/TypeScript

```typescript
class BigPlayClient {
  constructor(
    private apiKey: string,
    private baseUrl: string = "https://api.bigplay.dev"
  ) {}

  async processAction(
    sceneId: string,
    npcs: NPC[],
    playerIntent: PlayerIntent
  ): Promise<GameAction> {
    const response = await fetch(`${this.baseUrl}/elle/game/action`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        game_state: {
          scene_id: sceneId,
          npcs: npcs
        },
        player_intent: playerIntent
      })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return await response.json();
  }
}

// Usage
const client = new BigPlayClient("bp_live_...");

const action = await client.processAction(
  "tavern",
  [{
    id: "bob",
    name: "Bob",
    role: "innkeeper",
    emotional_state: {
      valence: 0.3,
      arousal: 0.5,
      dominance: 0.6,
      trust: 0.5
    }
  }],
  {
    type: "talk_to_npc",
    target_npc_id: "bob",
    raw_input: "Hello!"
  }
);

console.log(action.content.npc_dialogue);
```

### C# (Unity)

```csharp
using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class BigPlayClient
{
    private readonly string apiKey;
    private readonly string baseUrl;
    private readonly HttpClient client;

    public BigPlayClient(string apiKey, string baseUrl = "https://api.bigplay.dev")
    {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.client = new HttpClient();
        this.client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
    }

    public async Task<GameAction> ProcessAction(
        string sceneId,
        NPC[] npcs,
        PlayerIntent playerIntent
    )
    {
        var request = new
        {
            game_state = new
            {
                scene_id = sceneId,
                npcs = npcs
            },
            player_intent = playerIntent
        };

        var json = JsonConvert.SerializeObject(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await client.PostAsync($"{baseUrl}/elle/game/action", content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonConvert.DeserializeObject<GameAction>(responseJson);
    }
}

// Usage
var client = new BigPlayClient("bp_live_...");

var action = await client.ProcessAction(
    "tavern",
    new NPC[] {
        new NPC {
            id = "bob",
            name = "Bob",
            role = "innkeeper",
            emotional_state = new EmotionalState {
                valence = 0.3f,
                arousal = 0.5f,
                dominance = 0.6f,
                trust = 0.5f
            }
        }
    },
    new PlayerIntent {
        type = "talk_to_npc",
        target_npc_id = "bob",
        raw_input = "Hello!"
    }
);

Debug.Log(action.content.npc_dialogue);
```

---

## Appendix

### Versioning

BigPlay uses semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

**Current Version**: 1.0.0

**Version Header**:
```
X-BigPlay-Version: 1.0.0
```

### Changelog

**v1.0.0** (2025-11-16):
- Initial release
- Core game action endpoint
- Quest generation
- Voice synthesis
- Session management

### Support

- **Documentation**: https://docs.bigplay.dev
- **Discord**: https://discord.gg/bigplay
- **Email**: support@bigplay.dev
- **Status Page**: https://status.bigplay.dev

### Terms of Service

https://bigplay.dev/terms

### Privacy Policy

https://bigplay.dev/privacy

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0