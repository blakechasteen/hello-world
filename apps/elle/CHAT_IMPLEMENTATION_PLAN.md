# Elle Chat Implementation Plan

**Created**: 2025-11-29
**Goal**: Enable chat interaction with Elle, then expand to AR

---

## Current State Assessment

### What Exists ✅

| Component | Location | Status |
|-----------|----------|--------|
| **ElleEngine** | `engine/elle_engine.py` | ✅ Complete - orchestrates request→decision→action |
| **EllePolicy** | `core/policy.py` | ✅ Complete - LLM-based decision making |
| **LLM Clients** | `core/llm_client.py` | ✅ OpenAI, Anthropic, Local stubs |
| **PromptBuilder** | `core/prompt/prompt_builder.py` | ✅ Complete with HoloLoom refinement |
| **Domain Models** | `domain/` | ✅ Scene, Intent, Action, Task, etc. |
| **CLI Adapter** | `adapters/cli_adapter/cli.py` | 🟡 Partial - `interactive()` has TODOs |
| **AR Adapter** | `adapters/ar_adapter/` | ✅ Complete - AR events ↔ Elle requests |
| **Voice Interface** | `voice_interface.py` | ✅ SOP-focused commands |
| **Memory Store** | `memory/in_memory.py` | ✅ In-memory implementation |

### What's Missing ❌

1. **Chat Adapter** - Simple text→ElleRequest translation for conversation
2. **Conversation Context** - Multi-turn memory for chat sessions
3. **Simulated Scene** - Default "virtual space" for text-only interactions
4. **Interactive Chat Loop** - REPL with proper context management
5. **Chat-Optimized Prompt** - Conversational rather than AR-focused

---

## Phase 1: Chat MVP (Today)

**Goal**: Talk to Elle via terminal in natural conversation

### 1.1 Create Chat Adapter (`adapters/chat_adapter/`)

```
elle/adapters/chat_adapter/
├── __init__.py
├── chat_adapter.py      # Text → ElleRequest conversion
├── conversation.py      # Multi-turn conversation state
└── chat_scene.py        # Virtual scene for text-only mode
```

**Key Design**:
- No AR context needed - creates "virtual office" scene
- Maintains conversation history as memory
- Infers intent from text (question, command, exploration)
- Returns text responses (no AR visualizations)

### 1.2 Conversation Memory

Store recent exchanges for multi-turn:
```python
class ConversationMemory:
    def __init__(self, max_turns: int = 20):
        self.turns: List[ConversationTurn] = []
        self.session_id: str = ...

    def add_user_message(self, text: str) -> None: ...
    def add_elle_response(self, action: ElleAction) -> None: ...
    def get_context_for_prompt(self) -> str: ...
```

### 1.3 Interactive Chat Loop

```bash
python -m elle.chat
```

Simple interface:
```
╭──────────────────────────────────────────────────────────────╮
│  Elle - Your calm, grounded companion                        │
│  Type 'quit' to exit, 'help' for commands                    │
╰──────────────────────────────────────────────────────────────╯

You: I'm feeling overwhelmed with my workshop - so much clutter

Elle: Start with one small corner?

You: Which corner should I start with?

Elle: The workbench. It's where you'll need space first.

You: Good idea. What about all these random screws?

Elle: One jar. Sort later. Clear the surface first.
```

### 1.4 Chat-Optimized Prompt

Create `prompts/chat_prompt.txt` that:
- Emphasizes conversational tone
- Maintains Elle's minimal intervention philosophy
- Handles questions naturally
- Tracks conversation context

---

## Phase 2: Enhanced Chat (Week 1)

### 2.1 HoloLoom Memory Integration

Connect Elle's conversation to HoloLoom's knowledge graph:
```python
async with HoloLoom() as loom:
    # Store conversation as experiences
    await loom.experience(f"User said: {user_text}")
    await loom.experience(f"Elle responded: {elle_response}")

    # Recall relevant context
    memories = await loom.recall(user_text)
```

### 2.2 Rich Terminal UI

Using `rich` library for better UX:
- Syntax highlighting for code discussions
- Panels for Elle's "thinking" (optional debug mode)
- Progress indicators for slow LLM calls
- Markdown rendering for structured responses

### 2.3 Voice Input (Optional)

Enable voice-to-text for hands-free:
```bash
python -m elle.chat --voice
```

Uses existing `voice/whisper_stt.py` for transcription.

---

## Phase 3: AR Expansion (Week 2+)

### 3.1 Web-Based AR Client

Leverage existing `ar_web_client/demo.html`:
- Camera feed with object detection
- Overlay rendering for Elle's visual guidance
- WebRTC for real-time communication

### 3.2 Chat → AR Bridge

Seamless transition:
```
[Chat Mode]
You: Show me where to start organizing

[Elle switches to AR Mode]
Elle: Looking at your space now...
*Opens camera, detects objects, highlights workbench*
```

### 3.3 Multi-Modal Input

Support both text and vision simultaneously:
- "What's that thing on the shelf?" + gaze direction
- Voice commands while hands are busy
- Text chat for detailed questions

---

## Implementation Order

### Today (Phase 1.1-1.4)

1. **Create chat adapter module** (30 min)
   - Basic text → ElleRequest conversion
   - Virtual scene creation

2. **Implement conversation memory** (20 min)
   - Store turns in list
   - Format for prompt context

3. **Build interactive chat loop** (30 min)
   - REPL with clean interface
   - Graceful error handling
   - Exit commands

4. **Create chat prompt** (20 min)
   - Conversational version of base_prompt.txt
   - Multi-turn awareness

5. **Wire it all together** (20 min)
   - Create `elle/chat.py` entry point
   - Test with real LLM

### This Week (Phase 2)

- HoloLoom memory integration
- Rich terminal UI
- Voice input option

### Next Week (Phase 3)

- AR client testing
- Chat→AR bridging
- Multi-modal support

---

## File Structure After Phase 1

```
elle/
├── adapters/
│   ├── chat_adapter/          # NEW
│   │   ├── __init__.py
│   │   ├── chat_adapter.py    # Text → Elle translation
│   │   ├── conversation.py    # Multi-turn context
│   │   └── chat_scene.py      # Virtual scene for chat
│   ├── cli_adapter/
│   └── ar_adapter/
├── prompts/
│   ├── base_prompt.txt        # Existing AR-focused
│   └── chat_prompt.txt        # NEW conversational
├── chat.py                    # NEW entry point
└── ...
```

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Entry point | `python -m elle.chat` | Simple, discoverable |
| LLM default | Anthropic (Claude) | Best conversational quality |
| Scene mode | Virtual "thinking space" | No AR needed for pure chat |
| Memory | In-memory + optional HoloLoom | Start simple, scale up |
| UI | Rich terminal | Better than raw print() |

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `python -m elle.chat` starts interactive session
- [ ] Elle responds to questions with her calm, minimal style
- [ ] Multi-turn context is maintained (she remembers previous messages)
- [ ] Clean exit with `quit` or Ctrl+C

### Phase 2 Complete When:
- [ ] Conversations persist to HoloLoom
- [ ] Elle recalls past conversations
- [ ] Voice input works
- [ ] Rich terminal formatting

### Phase 3 Complete When:
- [ ] AR client can display Elle's visual guidance
- [ ] Seamless chat↔AR switching
- [ ] Voice + text + vision all work together

---

## Quick Start (After Implementation)

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Start chat
python -m elle.chat

# With voice input
python -m elle.chat --voice

# Debug mode (show prompts)
python -m elle.chat --debug
```

---

**Ready to implement. Starting with Phase 1.**
