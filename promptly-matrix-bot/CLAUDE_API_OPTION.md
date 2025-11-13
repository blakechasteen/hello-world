# Claude API Integration Option

Since Claude Code doesn't provide a CLI command, we have alternative options:

## Option 1: Keep Current Setup (Recommended for Now)

The bot gracefully handles the missing CLI and shows users a helpful message.

**Pros:**
- Already working
- No changes needed
- Safe fallback

**Cons:**
- Claude commands don't work from Matrix

## Option 2: Direct Anthropic API Integration

Replace CLI calls with direct API calls using Anthropic's Python SDK.

### Installation

```bash
pip install anthropic
```

### Modified claude_bridge.py

```python
from anthropic import Anthropic
import os

class ClaudeBridge:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def review(self, file_path: str) -> str:
        """Request code review via Anthropic API"""
        if not self.client:
            return "Error: ANTHROPIC_API_KEY not set"

        # Read file
        try:
            with open(file_path, 'r') as f:
                code = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        # Call Claude API
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Please review this code:\n\n```\n{code}\n```"
            }]
        )

        return message.content[0].text
```

### Required Environment Variable

Add to `.env`:
```bash
ANTHROPIC_API_KEY=your-api-key-here
```

**Pros:**
- Claude commands work from Matrix
- Direct API access (no CLI dependency)
- More reliable than subprocess calls

**Cons:**
- Requires API key (paid service)
- API calls have rate limits
- Different from local Claude Code experience

## Option 3: Matrix → Claude Code Workflow

Keep bot simple for git commands, use Claude Code directly for code review via its native interface.

**Workflow:**
1. `@promptly git status` - See what changed
2. Switch to Claude Code → Review files interactively
3. Return to Matrix for git operations

**Pros:**
- Leverages Claude Code's full capabilities
- No API costs
- Best UX for code review

**Cons:**
- Not fully automated from Matrix
- Requires context switching

## Recommendation

**For Phase 2 (Current):** Keep current setup. Git commands work perfectly from Matrix, and that's the main value.

**For Phase 3 (Future):** If you want code review from Matrix, implement Option 2 with API integration.

The ChatOps value is strongest for git operations (check status, make commits, push changes) rather than code review, which benefits from Claude Code's rich interface.
