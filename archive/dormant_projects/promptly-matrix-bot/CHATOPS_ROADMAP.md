# Promptly Matrix Bot - ChatOps Roadmap

## Vision
Transform Promptly from a simple command bot into a full ChatOps platform that bridges Matrix chat with development tools (Git, Claude Code, HoloLoom).

## Current State ✅
- [x] Matrix connection via SSO (access token)
- [x] Command parsing (`@promptly`, `@promptlybot`, `!commands`)
- [x] Conversational AI (Ollama integration)
- [x] Auto-join rooms
- [x] Multi-format command support

## Phase 1: Git Integration 🎯 NEXT

### Goal
Enable Git operations directly from Matrix chat.

### Commands to Add
```
@promptly git status
@promptly git commit "message"
@promptly git push
@promptly git create-pr "title" "description"
@promptly git check-pr #123
```

### Implementation
1. Add `GitHandler` class in `bot/git_handler.py`
2. Use `subprocess` to call `git` commands
3. Parse git output, format for Matrix
4. Add safety checks (confirm before push, etc.)

### Files to Create/Modify
- `bot/git_handler.py` (new) - Git command execution
- `bot/command_parser.py` - Add git command patterns
- `bot/promptly_bot.py` - Add `cmd_git()` method
- `.env` - Add `GIT_REPO_PATH` config

### Safety Features
- Confirm before destructive operations (push, force-push)
- Only allow git operations in configured repo path
- Show diffs before committing
- Require explicit approval for force operations

## Phase 2: Claude Code Bridge 🤖

### Goal
Pass complex coding tasks to Claude Code, return results to chat.

### Commands to Add
```
@promptly code-review [file]
@promptly refactor [file] "description"
@promptly explain [file]
@promptly debug [error message]
@promptly implement "feature description"
```

### Implementation Options

**Option A: CLI Bridge** (Recommended first)
```python
import subprocess

result = subprocess.run(
    ['claude', 'review', 'src/main.py'],
    capture_output=True,
    text=True
)
```

**Option B: HTTP API** (If Claude Code has API)
```python
import httpx

response = await httpx.post(
    'http://localhost:8000/code-review',
    json={'file': 'src/main.py'}
)
```

**Option C: Shared Task Queue** (Most sophisticated)
```python
# Promptly creates task
await hololoom.experience({
    'type': 'code_review_task',
    'file': 'src/main.py',
    'requester': '@user:matrix.org'
})

# Claude Code picks up task, executes, stores result
# Promptly polls for completion, sends to Matrix
```

### Files to Create/Modify
- `bot/claude_bridge.py` (new) - Claude Code integration
- `bot/command_parser.py` - Add code commands
- `bot/promptly_bot.py` - Add `cmd_code_*()` methods
- `.env` - Add `CLAUDE_CODE_PATH` or `CLAUDE_CODE_API_URL`

## Phase 3: HoloLoom Memory Integration 🧠

### Goal
Store all conversations, decisions, and context in HoloLoom knowledge graph.

### Features
- Every message → Memory shard
- Context-aware responses using HoloLoom retrieval
- Team memory (shared knowledge across users)
- Query past conversations

### Commands to Add
```
@promptly remember "fact"
@promptly recall "query"
@promptly what did we discuss about [topic]?
@promptly summarize last hour
```

### Implementation
```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Store message
    await loom.experience(message_content)

    # Retrieve context
    memories = await loom.recall(query)

    # Use in response generation
    context = "\n".join([m.content for m in memories])
    response = ollama.chat(context + user_query)
```

### Files to Create/Modify
- `bot/hololoom_integration.py` (new) - HoloLoom memory
- `bot/promptly_bot.py` - Store all messages
- `.env` - Add `HOLOLOOM_MODE=FUSED`

## Phase 4: Advanced ChatOps Features 🚀

### Team Collaboration
- Room-specific memory contexts
- User permissions (who can git push, etc.)
- Notification preferences

### Workflow Automation
```
@promptly workflow create "name"
  1. git pull
  2. run tests
  3. if tests pass: create PR
  4. notify team
```

### Monitoring & Alerts
```
@promptly watch repo for new PRs
@promptly alert me when CI fails
@promptly daily standup reminder
```

### Advanced Git Features
- Merge conflict resolution suggestions
- Code ownership tracking
- Automated code reviews
- Deployment workflows

## Implementation Timeline

### Week 1: Git Basics
- [ ] Git status, log, diff
- [ ] Git commit, push (with confirmation)
- [ ] Safety checks and repo validation

### Week 2: Git Advanced
- [ ] PR creation/management
- [ ] Branch operations
- [ ] Merge/rebase helpers

### Week 3: Claude Code Bridge
- [ ] CLI integration
- [ ] Code review command
- [ ] Explain/refactor commands

### Week 4: HoloLoom Integration
- [ ] Memory storage
- [ ] Context-aware responses
- [ ] Team memory

### Week 5+: Advanced Features
- [ ] Workflow automation
- [ ] Monitoring/alerts
- [ ] Team collaboration features

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Matrix Chat (Element)                  │
│  Users: @user1, @user2, @promptlybot                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Promptly Matrix Bot                       │
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
└────────────────┘  └────────────┘  └──────────────┘
```

## Example Workflows

### Workflow 1: Review and Merge PR
```
User: @promptly check PR #42

Promptly: 🔍 Checking PR #42...
- Title: Add authentication feature
- Author: @alice
- Status: ✅ CI passing
- Files changed: 5
- +342 -18 lines

User: @promptly review this PR

Promptly: 🤖 Requesting Claude Code review...
[2 minutes later]
Promptly: ✅ Review complete!

**Issues Found:**
1. Missing null check in auth_handler.py:42
2. Potential SQL injection in user_query.py:89

**Suggestions:**
1. Add unit tests for edge cases
2. Consider using prepared statements

[Full report] (link)

User: @promptly git merge #42

Promptly: ⚠️ Are you sure? This will merge PR #42 into main.
Type "yes" to confirm.

User: yes

Promptly: ✅ Merged PR #42 into main
```

### Workflow 2: Quick Bug Fix
```
User: @promptly git status

Promptly: 📊 Git Status:
Branch: feature/auth
Modified: 2 files
- src/auth.py
- tests/test_auth.py

User: @promptly explain src/auth.py

Promptly: 🤖 Asking Claude Code...
[Claude Code analyzes file]
Promptly: 📖 Summary:
This file handles user authentication with JWT tokens.
Key functions:
- authenticate_user(): Validates credentials
- generate_token(): Creates JWT
- verify_token(): Validates JWT

User: @promptly git commit "Fix auth token expiry bug"

Promptly: 📝 Commit created:
Commit: a1b2c3d
Files: 2 changed
Message: Fix auth token expiry bug

User: @promptly git push

Promptly: ⚠️ This will push to remote. Confirm? (yes/no)

User: yes

Promptly: ✅ Pushed to origin/feature/auth
```

### Workflow 3: Team Memory
```
User1: @promptly remember we decided to use PostgreSQL for the auth service

Promptly: 💾 Stored in team memory

[3 days later]

User2: @promptly what database are we using for auth?

Promptly: 🧠 Recalling team memory...
According to our discussion on Oct 29, we decided to use PostgreSQL for the auth service.

User2: @promptly what else did we discuss about auth?

Promptly: 📚 Related discussions:
1. Oct 29: PostgreSQL decision
2. Oct 27: JWT vs session tokens (chose JWT)
3. Oct 25: OAuth provider selection (Google + GitHub)
```

## Security Considerations

### Git Operations
- [ ] Whitelist allowed git commands
- [ ] Require confirmation for destructive operations
- [ ] Only operate in configured repo paths
- [ ] Log all git operations with user attribution

### Claude Code Bridge
- [ ] Rate limiting to prevent abuse
- [ ] Validate file paths (prevent directory traversal)
- [ ] Sandbox code execution
- [ ] Timeout long-running operations

### HoloLoom Memory
- [ ] Room-based access control (private vs shared memory)
- [ ] User permissions (read/write/admin)
- [ ] Audit trail for sensitive operations
- [ ] Data retention policies

### Matrix Bot
- [ ] Validate all user input
- [ ] Escape shell commands
- [ ] Use environment variables for secrets
- [ ] Encrypted room support (pending libolm)

## Configuration

### Required Environment Variables
```bash
# Matrix credentials
MATRIX_ACCESS_TOKEN=mat_xxxxx

# Ollama
LM_MODEL=ollama/llama3.2:3b
OLLAMA_HOST=http://localhost:11434

# Git (Phase 1)
GIT_REPO_PATH=/path/to/repo
GIT_DEFAULT_BRANCH=main

# Claude Code (Phase 2)
CLAUDE_CODE_PATH=/usr/local/bin/claude
# OR
CLAUDE_CODE_API_URL=http://localhost:8000

# HoloLoom (Phase 3)
HOLOLOOM_MODE=FUSED
HOLOLOOM_MEMORY_PATH=./hololoom_memory

# Security
ALLOWED_GIT_COMMANDS=status,log,diff,commit,push,pull
REQUIRE_CONFIRMATION=push,merge,rebase
ADMIN_USERS=@alice:matrix.org,@bob:matrix.org
```

## Success Metrics

### Phase 1 (Git)
- [ ] Can execute 5+ git commands from chat
- [ ] All destructive operations require confirmation
- [ ] Git operations logged to audit trail

### Phase 2 (Claude Code)
- [ ] Can request code reviews from chat
- [ ] Reviews complete in <5 minutes
- [ ] Results formatted nicely in Matrix

### Phase 3 (HoloLoom)
- [ ] 100% of messages stored in memory
- [ ] Context-aware responses (uses past conversations)
- [ ] Team memory queries working

### Phase 4 (Advanced)
- [ ] 3+ automated workflows configured
- [ ] Monitoring/alerts functional
- [ ] Team actively using ChatOps daily

## Resources

### Documentation
- Matrix SDK: https://matrix-nio.readthedocs.io/
- Git Python: https://gitpython.readthedocs.io/ (alternative to subprocess)
- HoloLoom: See CLAUDE.md and HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md

### Inspiration
- GitHub ChatOps: https://github.blog/2013-01-31-using-github-chatops/
- GitLab ChatOps: https://docs.gitlab.com/ee/ci/chatops/
- Hubot: https://hubot.github.com/

## Questions for User

1. **Git Integration Priority**: Which git commands are most important to you first?
   - Status/log/diff (read-only)
   - Commit/push (write operations)
   - PR management
   - Branch operations

2. **Claude Code Access**: How do you currently run Claude Code?
   - CLI commands?
   - API server?
   - VS Code extension only?

3. **HoloLoom Memory**: What should be stored?
   - All messages?
   - Only important decisions?
   - Code-related only?

4. **Use Case**: What's your ideal workflow?
   - Solo developer using Promptly as assistant?
   - Team collaboration tool?
   - Project management + code helper?
