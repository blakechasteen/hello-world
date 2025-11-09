# Promptly Matrix Bot → Full ChatOps: The Big Picture

## 🎯 Vision: Matrix Chat as Your Development Command Center

**Current State**: Simple Matrix bot with conversational AI (Ollama)
**End Goal**: Full ChatOps platform bridging Matrix ↔ Git ↔ Claude Code ↔ HoloLoom

## The Three Bridges

### 1. 🔧 Git Bridge (Phase 1 - In Progress)
**Purpose**: Git operations directly from Matrix chat

```
User: @promptly git status
Bot:  📊 Git Status:
      Branch: cleanUp/11-8-25
      Modified: 5 files
      - HoloLoom/promptly/dspy_bridge.py
      - promptly-matrix-bot/bot/promptly_core.py
      ...

User: @promptly git commit "Add git integration"
Bot:  📝 Commit created:
      Commit: a1b2c3d
      Message: Add git integration

User: @promptly git push
Bot:  ⚠️ This will push to remote. Confirm? (yes/no)
```

**Status**: ✅ Git handler created, commands parsed, ready to integrate

### 2. 🤖 Claude Code Bridge (Phase 2)
**Purpose**: Pass complex tasks to Claude Code, return results to chat

```
User: @promptly code-review src/auth.py
Bot:  🤖 Requesting Claude Code review...
      [2 minutes later]
      ✅ Review complete!

      Issues Found:
      1. Missing null check in auth_handler.py:42
      2. Potential SQL injection in user_query.py:89

      Suggestions:
      1. Add unit tests for edge cases
      2. Consider using prepared statements
```

**How it works**:

**Option A: CLI Bridge** (Simplest)
```python
# Promptly calls Claude Code CLI
import subprocess

result = subprocess.run(
    ['claude', 'code-review', 'src/auth.py'],
    capture_output=True,
    text=True
)

# Format and send to Matrix
await bot.send_message(room, result.stdout)
```

**Option B: HTTP API** (Most Flexible)
```python
# Claude Code exposes API (like HoloLoom's agentic server)
import httpx

response = await httpx.post(
    'http://localhost:8000/code-review',
    json={'file': 'src/auth.py'}
)

await bot.send_message(room, response.json()['result'])
```

**Option C: Shared Memory** (Most Integrated)
```python
# Both Promptly and Claude Code use HoloLoom memory

# Promptly creates task
await hololoom.experience({
    'type': 'code_review_request',
    'file': 'src/auth.py',
    'requester': '@user:matrix.org',
    'status': 'pending'
})

# Claude Code picks up task (polling or webhook)
# Executes review, stores result in HoloLoom

# Promptly detects completion
result = await hololoom.recall('code_review_request for src/auth.py')
await bot.send_message(room, result.content)
```

### 3. 🧠 HoloLoom Memory Bridge (Phase 3)
**Purpose**: All conversations stored in knowledge graph, context-aware responses

```
User: @promptly remember we decided to use PostgreSQL for auth

[3 days later]

User: @promptly what database are we using for auth?
Bot:  🧠 Recalling team memory...
      According to our discussion on Oct 29, we decided to use
      PostgreSQL for the auth service.

User: @promptly what else did we discuss about auth?
Bot:  📚 Related discussions:
      1. Oct 29: PostgreSQL decision
      2. Oct 27: JWT vs session tokens (chose JWT)
      3. Oct 25: OAuth provider selection (Google + GitHub)
```

**How it works**:
```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Store every message as memory
    await loom.experience({
        'content': message_content,
        'user': event.sender,
        'room': room.room_id,
        'timestamp': time.time()
    })

    # Retrieve context when answering
    memories = await loom.recall(query)

    # Use in Ollama response
    context = "\n".join([m.content for m in memories])
    response = ollama.chat(
        messages=[
            {'role': 'system', 'content': context},
            {'role': 'user', 'content': user_query}
        ]
    )
```

## The Complete Flow

```
Matrix Chat
    ↓
Promptly Bot (Python)
    ├─ Git Handler → subprocess → git commands
    ├─ Claude Bridge → CLI/API/Memory → Claude Code
    ├─ HoloLoom Memory → Knowledge Graph → Context
    └─ Ollama Chat → LLM → Conversational responses
    ↓
Formatted Results → Matrix Room
```

## Example Workflow: Complete Feature Development

```
User: @promptly git status

Bot: 📊 Branch: feature/auth
     Modified: 0 files
     Clean working tree

User: @promptly implement user authentication with JWT tokens

Bot: 🤖 Passing to Claude Code...
     [Claude Code creates files, makes commits]
     ✅ Implementation complete!

     Created:
     - src/auth/jwt_handler.py (230 lines)
     - src/auth/middleware.py (85 lines)
     - tests/test_auth.py (142 lines)

     Committed: "feat: Add JWT authentication"

User: @promptly git status

Bot: 📊 Modified: 3 files
     Ready to push

User: @promptly run tests

Bot: 🧪 Running test suite...
     ✅ All 12 tests passed!

User: @promptly git push

Bot: ⚠️ Confirm push to remote? (yes/no)

User: yes

Bot: ✅ Pushed to origin/feature/auth

     Next steps:
     - Create PR for review
     - Tag reviewers

User: @promptly create PR "Add JWT authentication" "Implements secure token-based auth"

Bot: 📝 Pull Request #42 created!
     https://github.com/user/repo/pull/42

     Reviewers notified:
     - @alice (code owner)
     - @bob (security review)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                 Matrix Chat (Element Web/Desktop)            │
│  Users can:                                                  │
│  - Chat with @promptlybot                                    │
│  - Execute git commands                                      │
│  - Request code reviews                                      │
│  - Access team memory                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Promptly Matrix Bot (Python)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Command    │  │     Git      │  │    Claude    │      │
│  │    Parser    │──│   Handler    │  │    Bridge    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Ollama     │  │   HoloLoom   │  │   Response   │      │
│  │   (Chat)     │  │   (Memory)   │  │   Formatter  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────┬──────────────┬──────────────┬───────────────────┘
            │              │              │
            ▼              ▼              ▼
┌────────────────┐  ┌────────────┐  ┌──────────────┐
│  Git Commands  │  │  HoloLoom  │  │ Claude Code  │
│  (subprocess)  │  │   Memory   │  │  CLI/API     │
│                │  │            │  │              │
│  - status      │  │  - Store   │  │  - Review    │
│  - commit      │  │  - Recall  │  │  - Refactor  │
│  - push        │  │  - Context │  │  - Explain   │
│  - create PR   │  │  - Search  │  │  - Implement │
└────────────────┘  └────────────┘  └──────────────┘
```

## What You Can Do RIGHT NOW

With the git handler we just built, you can:

1. **Add git command handling to promptly_bot.py**
2. **Test git commands from Matrix**:
   ```
   @promptly git status
   @promptly git log
   @promptly git diff
   @promptly git commit "your message"
   ```

## Next Steps to Complete the Vision

### Week 1: Finish Git Integration
1. ✅ Git handler created (DONE)
2. ✅ Command parsing added (DONE)
3. ✅ Environment configured (DONE)
4. ⏳ Add git command methods to promptly_bot.py
5. ⏳ Test all git commands from Matrix
6. ⏳ Add safety confirmations for push/merge

### Week 2: Claude Code Bridge
1. Determine Claude Code access method (CLI/API/Memory)
2. Create `claude_bridge.py`
3. Add code-review, refactor, explain commands
4. Test integration

### Week 3: HoloLoom Memory
1. Store all messages in HoloLoom
2. Add context retrieval to Ollama responses
3. Implement memory queries
4. Test team memory features

### Week 4: Polish & Advanced Features
1. PR creation/management
2. Workflow automation
3. Monitoring/alerts
4. Documentation

## How to Complete Git Integration (Next Immediate Step)

I've created:
- ✅ `bot/git_handler.py` - Git command execution
- ✅ `bot/command_parser.py` - Git command parsing (git-status, git-log, etc.)
- ✅ `.env` - GIT_REPO_PATH configured
- ✅ `CHATOPS_ROADMAP.md` - Complete roadmap

What's left:
1. Add git command methods to `promptly_bot.py`:
   - `cmd_git_status()`
   - `cmd_git_log()`
   - `cmd_git_commit()`
   - etc.
2. Initialize git_handler in `__init__`
3. Add git command routing in `handle_command()`
4. Test from Matrix!

Would you like me to:
A. Complete the git integration in promptly_bot.py? ← **Recommended next**
B. Explain how Claude Code integration would work in detail?
C. Show you how to set up the HoloLoom memory integration?

## The Answer to "Can Promptly pass things to Claude Code?"

**YES! Three ways:**

1. **Subprocess/CLI** (easiest): Promptly calls `claude` CLI, captures output
2. **HTTP API** (flexible): If Claude Code has HTTP API, Promptly makes requests
3. **Shared Memory** (advanced): Both use HoloLoom, communicate via memory graph

The key insight: **Promptly is the glue between chat and development tools**. It:
- Takes natural language from Matrix
- Translates to tool actions (git, claude, hololoom)
- Returns results formatted for chat

You're building a **conversational development environment** where everything happens in Matrix chat!
