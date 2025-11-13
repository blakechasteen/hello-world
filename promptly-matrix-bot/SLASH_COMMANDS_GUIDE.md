# Making Promptly More Intuitive with Slash Commands

## Current Command Styles

Your Promptly bot currently supports:
- **Mentions**: `@promptly git status` (verbose, good for multi-user rooms)
- **Bang commands**: `!git status` (shorter, IRC-style)

## 🎯 Recommended: Add True Slash Commands

### Option 1: Matrix Native Slash Commands (Recommended)

Matrix supports **true slash commands** that appear in the autocomplete menu. Here's how to add them:

#### Step 1: Register Commands via Bot

```python
# In bot/promptly_bot.py, add after login:

async def register_slash_commands(self):
    """Register slash commands with Matrix"""
    # Matrix supports command hints via m.room.power_levels
    # For now, we'll handle them in message parsing

    commands = {
        '/git': 'Git operations (status, commit, push, etc.)',
        '/review': 'Review code with Claude',
        '/explain': 'Explain code with Claude',
        '/remember': 'Save to HoloLoom memory',
        '/recall': 'Retrieve from HoloLoom memory',
        '/optimize': 'Optimize a prompt',
        '/help': 'Show all commands'
    }

    # Store for autocomplete hints
    self.slash_commands = commands
```

#### Step 2: Update Command Parser

Add to `bot/command_parser.py`:

```python
class CommandParser:
    """Parse natural language commands for Promptly bot"""

    # Add slash command patterns
    SLASH_COMMANDS = {
        # Git shortcuts
        '/git': r'^/git\s+(\w+)(?:\s+(.+))?',
        '/gs': r'^/gs',  # Alias for git status
        '/gl': r'^/gl',  # Alias for git log
        '/gc': r'^/gc\s+"([^"]+)"',  # git commit "message"
        '/gp': r'^/gp',  # git push

        # Claude Code shortcuts
        '/review': r'^/review\s+(.+)',
        '/explain': r'^/explain\s+(.+)',
        '/refactor': r'^/refactor\s+(\S+)\s+"([^"]+)"',

        # HoloLoom shortcuts
        '/remember': r'^/remember\s+(.+)',
        '/recall': r'^/recall\s+(.+)',
        '/forget': r'^/forget\s+(.+)',

        # Prompt optimization
        '/optimize': r'^/optimize',
        '/run': r'^/run\s+(\w+)\s+"([^"]+)"',

        # Help
        '/help': r'^/help(?:\s+(\w+))?',
        '/?': r'^\?',  # Quick help
    }

    def parse(self, message: str) -> Optional[Dict]:
        """Parse message into structured command"""

        # Try slash commands first (highest priority)
        for cmd_name, pattern in self.SLASH_COMMANDS.items():
            match = re.match(pattern, message.strip())
            if match:
                return self.extract_slash_command(cmd_name, match.groups(), message)

        # Fall back to existing @promptly/! commands
        # ... existing code ...

    def extract_slash_command(self, cmd_name: str, groups: tuple, full_message: str) -> Dict:
        """Extract slash command parameters"""

        if cmd_name == '/git':
            operation = groups[0]  # status, log, commit, etc.
            args = groups[1] if len(groups) > 1 else None

            # Map to internal command types
            cmd_map = {
                'status': 'git-status',
                'log': 'git-log',
                'diff': 'git-diff',
                'branch': 'git-branch',
                'commit': 'git-commit',
                'push': 'git-push',
                'pull': 'git-pull',
            }

            cmd_type = cmd_map.get(operation, 'unknown')

            if cmd_type == 'git-commit' and args:
                return {'type': cmd_type, 'message': args.strip('"')}
            else:
                return {'type': cmd_type}

        elif cmd_name == '/gs':
            return {'type': 'git-status'}

        elif cmd_name == '/gl':
            return {'type': 'git-log'}

        elif cmd_name == '/gc':
            return {'type': 'git-commit', 'message': groups[0]}

        elif cmd_name == '/gp':
            return {'type': 'git-push'}

        elif cmd_name == '/review':
            return {'type': 'claude-review', 'file_path': groups[0].strip()}

        elif cmd_name == '/explain':
            return {'type': 'claude-explain', 'file_path': groups[0].strip()}

        elif cmd_name == '/refactor':
            return {
                'type': 'claude-refactor',
                'file_path': groups[0],
                'instruction': groups[1]
            }

        elif cmd_name == '/remember':
            return {'type': 'remember', 'content': groups[0]}

        elif cmd_name == '/recall':
            return {'type': 'recall', 'query': groups[0]}

        elif cmd_name == '/optimize':
            return self.parse_optimize_command(full_message)

        elif cmd_name == '/run':
            return {'type': 'run', 'workflow': groups[0], 'input': groups[1]}

        elif cmd_name == '/help':
            topic = groups[0] if groups and groups[0] else None
            return {'type': 'help', 'topic': topic}

        elif cmd_name == '/?':
            return {'type': 'help', 'quick': True}

        return {'type': 'unknown'}
```

### Usage Examples

**Before (verbose):**
```
@promptly git status
@promptly git commit "Add feature"
@promptly review src/auth.py
@promptly explain src/db.py
```

**After (intuitive):**
```
/gs                          → git status
/gc "Add feature"            → git commit
/review src/auth.py          → Claude review
/explain src/db.py           → Claude explain
```

**Super short aliases:**
```
/gs     → git status
/gl     → git log
/gp     → git push
/gc     → git commit
```

## Option 2: Contextual Command Menus

Add smart context detection:

```python
# In bot/promptly_bot.py

async def handle_command(self, message: str, room: MatrixRoom, event: RoomMessageText) -> Optional[Dict[str, str]]:
    """Parse and execute command"""

    # Smart context detection
    if message.strip() == '?':
        # Quick help based on recent context
        recent_commands = self.state.get_recent_commands(room.room_id, limit=3)
        return self.generate_contextual_help(recent_commands)

    # ... rest of command handling ...

def generate_contextual_help(self, recent_commands: List[str]) -> Dict[str, str]:
    """Generate help based on what user has been doing"""

    if any('git' in cmd for cmd in recent_commands):
        # User is working with git
        help_text = """**Quick Git Commands:**

/gs              → git status
/gl              → git log
/gd              → git diff
/gc "message"    → git commit
/gp              → git push

Type `/help git` for all git commands"""

    elif any('review' in cmd or 'explain' in cmd for cmd in recent_commands):
        # User is doing code review
        help_text = """**Quick Code Review Commands:**

/review <file>              → Review code
/explain <file>             → Explain code
/refactor <file> "task"     → Refactor code

Type `/help claude` for all Claude commands"""

    else:
        # General help
        help_text = """**Promptly Quick Commands:**

/gs              → Git status
/review <file>   → Code review
/?               → This help
/help            → Full help

Type `/help <topic>` for detailed help"""

    return {"body": help_text, "html": f"<pre>{help_text}</pre>"}
```

## Option 3: Natural Language Fallback

Keep it conversational when slash commands aren't used:

```python
async def cmd_chat(self, message: str, room: MatrixRoom) -> Dict[str, str]:
    """Handle conversational chat using Ollama"""

    # Detect intent from natural language
    intents = {
        'git_status': ['status', 'what changed', 'show changes', 'repo state'],
        'git_commit': ['commit', 'save changes', 'create commit'],
        'code_review': ['review', 'check code', 'any issues'],
        'explain': ['explain', 'how does', 'what does this do'],
    }

    message_lower = message.lower()

    for intent, keywords in intents.items():
        if any(keyword in message_lower for keyword in keywords):
            # Route to appropriate handler
            if intent == 'git_status':
                return await self.cmd_git_status({}, room)
            elif intent == 'code_review' and 'review' in message_lower:
                # Try to extract filename
                file_match = re.search(r'review\s+(\S+\.py)', message_lower)
                if file_match:
                    return await self.cmd_claude_review({'file_path': file_match.group(1)}, room)

    # Fall back to LLM chat
    # ... existing chat code ...
```

## Option 4: Integrate with Claude Code Slash Commands

If you want to use Promptly **from within Claude Code** (not just Matrix):

```python
# Create .claude/commands/ directory
# Add files like:

# .claude/commands/git-status.md
"""
Run this command with: /git-status

Shows git repository status via Promptly bot
"""

PYTHONPATH=. python promptly-matrix-bot/cli_bridge.py git status
```

Then create a CLI bridge:

```python
# promptly-matrix-bot/cli_bridge.py

#!/usr/bin/env python3
"""
CLI bridge for using Promptly from command line (or Claude Code slash commands)
"""

import sys
import asyncio
from bot.git_handler import GitHandler
from bot.claude_bridge import ClaudeBridge

async def main():
    args = sys.argv[1:]

    if args[0] == 'git':
        handler = GitHandler(".")
        operation = args[1]

        if operation == 'status':
            print(handler.status())
        elif operation == 'log':
            print(handler.log())
        elif operation == 'commit':
            msg = args[2] if len(args) > 2 else "Update"
            print(handler.commit(msg))
        # ... etc

    elif args[0] == 'review':
        bridge = ClaudeBridge()
        file_path = args[1]
        print(bridge.review(file_path))

if __name__ == '__main__':
    asyncio.run(main())
```

## 🎯 Recommended Implementation Order

### Phase 1: Core Slash Commands (1 hour)
1. Update `command_parser.py` with slash command patterns
2. Add `/gs`, `/gc`, `/gp` for git
3. Add `/review`, `/explain` for Claude
4. Test in Matrix

### Phase 2: Contextual Help (30 min)
1. Add `?` quick help
2. Add context detection
3. Show relevant commands based on history

### Phase 3: Natural Language (1 hour)
1. Improve intent detection in `cmd_chat()`
2. Add file extraction from natural language
3. Smart routing to handlers

### Phase 4: Advanced (Optional)
1. Command aliases (`.g = /git`, `.r = /review`)
2. Multi-step commands (`/workflow`)
3. Autocomplete registration

## Command Comparison

| Style | Example | Pros | Cons |
|-------|---------|------|------|
| **Mentions** | `@promptly git status` | Clear who you're talking to | Verbose |
| **Bang** | `!git status` | Shorter, IRC-style | Less discoverable |
| **Slash** | `/gs` | Intuitive, autocomplete | Need good docs |
| **Natural** | `show me what changed` | Most intuitive | Harder to implement |

## Quick Start

Want the fastest improvement? Add these 3 lines to `command_parser.py`:

```python
# At the top of COMMANDS dict:
'/gs': r'^/gs',
'/gc': r'^/gc\s+"([^"]+)"',
'/review': r'^/review\s+(.+)',
```

Now you have:
- `/gs` → instant git status
- `/gc "message"` → quick commit
- `/review file.py` → instant code review

**That's 80% of the value in 5 minutes!**

## Integration with Departmental Architecture

Looking at your selected text about the departmental agent architecture, you could create **department-specific slash commands**:

```python
# Department routing
DEPARTMENT_COMMANDS = {
    # Infrastructure Department
    '/infra': {
        '/infra/neo4j': 'Query Neo4j status',
        '/infra/qdrant': 'Query Qdrant status',
        '/infra/perf': 'Performance diagnostics',
    },

    # MasterWeaver Department
    '/weave': {
        '/weave/extract': 'Extract entities',
        '/weave/validate': 'Validate consistency',
        '/weave/query': 'Query domain ontology',
    },

    # Verification Department
    '/verify': {
        '/verify/confidence': 'Check confidence',
        '/verify/rerun': 'Request rerun',
        '/verify/cross': 'Cross-check departments',
    },

    # Context (HoloLoom) Department
    '/context': {
        '/context/enrich': 'Enrich context',
        '/context/missing': 'Detect missing context',
        '/context/search': 'Search knowledge graph',
    },
}
```

This gives you a **hierarchical command system** that maps directly to your departmental architecture!

## Next Steps

1. **Quick Win**: Add `/gs`, `/gc`, `/review` to command parser (5 min)
2. **Test**: Try them in Matrix (2 min)
3. **Expand**: Add more slash commands as needed (ongoing)
4. **Document**: Update help command with new shortcuts (10 min)

Want me to implement any of these options for you?
