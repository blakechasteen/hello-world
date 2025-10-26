# MCP Server: Conversational Update 🎯

## What's New

The HoloLoom Memory MCP Server now has **conversational intelligence with auto-spin signal filtering**!

### New MCP Tools

#### 1. `chat` - Conversational Interface

Automatically scores importance and spins signal to memory while filtering noise.

**Usage in Claude Desktop:**
```
"Chat with me: What is HoloLoom?"
"Chat: How does memory recall work?"
```

**What It Does:**
1. Takes your message
2. Searches existing memories for context
3. Generates response based on memories
4. **Scores importance** (0.0-1.0)
5. **Auto-spins to memory if important** (>= 0.4 threshold)
6. Returns response with importance metadata

**Response Format:**
```
Based on your memories, here's what I found:
• Memory 1 text...
• Memory 2 text...
• Memory 3 text...

Found 5 related memories.

─────────────────────────────────
✓ REMEMBERED (Importance: 0.72, Threshold: 0.4)
Turn: 5 | Signal: 3 | Noise: 2
```

#### 2. `conversation_stats` - View Stats

See your signal/noise ratio and conversation quality.

**Usage:**
```
"Show conversation stats"
"What's my signal to noise ratio?"
```

**Response:**
```
Conversation Statistics:
==================================================
Total Turns: 10
Signal (Remembered): 6 (60.0%)
Noise (Filtered): 4 (40.0%)
Avg Importance: 0.58

Recent History (last 5 turns):
  [✓] Turn 6 (Score: 0.72)
      User: What is HoloLoom?...
  [✗] Turn 7 (Score: 0.15)
      User: thanks...
  [✓] Turn 8 (Score: 0.85)
      User: How does Thompson Sampling work?...

Signal vs Noise Filtering:
  • Threshold: 0.4 (default)
  • Signal: Important Q&A, facts, decisions
  • Noise: Greetings, acknowledgments, trivial chat
```

## How Importance Scoring Works

### SIGNAL Indicators (Higher Score)
- Questions (`?`)
- Information keywords (what, how, why, explain, etc.)
- Domain terms (memory, recall, knowledge, graph, etc.)
- Longer messages (substantive content)
- Entity references (capitalized words)
- High confidence tool responses

### NOISE Indicators (Lower Score)
- Greetings (hi, hello, thanks, bye)
- Acknowledgments (ok, sure, got it)
- Very short exchanges
- Error messages
- Low information content

### Threshold: 0.4 (Default)
- **>= 0.4**: REMEMBERED (spun to memory)
- **< 0.4**: FILTERED (discarded as noise)

Adjustable per chat:
```json
{
  "message": "Your question",
  "importance_threshold": 0.6  // More selective
}
```

## Example Conversation

```
Turn 1: "hi"
  Importance: 0.10 → ✗ FILTERED (noise)

Turn 2: "What is memory recall?"
  Importance: 0.75 → ✓ REMEMBERED (signal)
  Auto-spun to memory!

Turn 3: "ok"
  Importance: 0.00 → ✗ FILTERED (noise)

Turn 4: "How does semantic search work?"
  Importance: 0.88 → ✓ REMEMBERED (signal)
  Auto-spun to memory!

Turn 5: "thanks"
  Importance: 0.00 → ✗ FILTERED (noise)

Results: 5 turns, 2 remembered (40% signal, 60% noise)
```

## Technical Implementation

### Location
`HoloLoom/memory/mcp_server.py` (now 697 lines)

### Key Features
1. **Importance Scoring Function** (lines 72-143)
   - Regex-based noise detection
   - Keyword-based signal detection
   - Metadata-aware scoring

2. **Chat Tool Handler** (lines 486-587)
   - Semantic memory recall
   - Response generation
   - Auto-spin if important
   - Conversation history tracking

3. **Stats Tool Handler** (lines 589-626)
   - Aggregated statistics
   - Recent history display
   - Signal/noise breakdown

### Global State
```python
conversation_history: List[Dict]  # Last 100 turns
conversation_stats: Dict  # Aggregated metrics
```

## Integration with Memory System

Auto-spun conversations are stored as regular memories:

```python
{
  "text": "Conversation Turn 5 (2025-10-23T...)
           User: What is HoloLoom?
           System: HoloLoom is a neural...
           Importance: 0.72",
  "context": {"type": "conversation", "importance": 0.72},
  "tags": ["conversation", "auto-spun"],
  "user_id": "blake"
}
```

These become searchable like any other memory and can inform future responses!

## Benefits

### 1. Self-Building Knowledge Base
- Learns from conversations automatically
- No manual curation needed
- Important exchanges become permanent knowledge

### 2. Efficient Memory Usage
- Filters greetings and acknowledgments
- Only stores meaningful exchanges
- Keeps memory lean and signal-focused

### 3. Context-Aware Responses
- Can reference previous conversation
- Builds on prior exchanges
- Maintains conversation continuity

### 4. Observable Quality
- See your signal/noise ratio
- Track conversation quality
- Tune threshold if needed

## Usage in Claude Desktop

1. Ensure MCP server is configured in `claude_desktop_config.json`
2. Restart Claude Desktop
3. Look for 🔌 icon (MCP connected)

### Basic Chat
```
"Chat with me: What is Thompson Sampling?"
```

### Check Stats
```
"Show conversation stats"
"What's my remember rate?"
```

### Adjust Threshold
Via MCP tool call with custom threshold:
```json
{
  "tool": "chat",
  "arguments": {
    "message": "Important question here",
    "importance_threshold": 0.6
  }
}
```

## Architecture Flow

```
User Message
    ↓
MCP chat tool
    ↓
Semantic Recall (find context)
    ↓
Generate Response
    ↓
Score Importance (0.0-1.0)
    ↓
Check Threshold (>= 0.4?)
    ↓
If SIGNAL → Auto-spin to Memory
If NOISE → Discard
    ↓
Return Response + Metadata
    ↓
Update Stats
```

## Configuration

### Default Settings
- Importance threshold: **0.4** (balanced)
- Max history: **100 turns**
- Auto-spin: **Enabled**
- Signal detection: **Keyword + metadata**

### Customization
Edit `score_importance()` in `mcp_server.py` to:
- Add domain-specific keywords
- Adjust noise/signal weights
- Add custom scoring logic

## Signal vs Noise Baby! 🎯

This update brings the ConversationalAutoLoom intelligence to the MCP server:
- ✅ Auto-spins important turns
- ✅ Filters trivial exchanges
- ✅ Tracks conversation quality
- ✅ Builds knowledge from chat

**The MCP server now learns from every conversation, filtering signal from noise automatically!**

---

## Files Modified

- `HoloLoom/memory/mcp_server.py` - Added chat & conversation_stats tools
- `demo_mcp.py` - Updated to show new features
- `MCP_CONVERSATIONAL_UPDATE.md` - This document

## See Also

- `HoloLoom/conversational.py` - Standalone ConversationalAutoLoom
- `HoloLoom/CONVERSATIONAL_README.md` - Full conversational docs
- `example_conversational.py` - Working demo

**The future is conversational, and it filters noise! 🚀**
