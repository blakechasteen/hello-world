# ExpertLoom MCP Server

**Connect Claude Desktop to your ExpertLoom backend!**

This MCP server exposes ExpertLoom's text processing capabilities as tools that Claude Desktop can use.

## What You Get

Claude Desktop can now:
- ✅ **Summarize your notes** while preserving entities and measurements
- ✅ **Extract entities** (Corolla → vehicle-corolla-2015)
- ✅ **Extract measurements** (28 PSI, 3mm, etc.)
- ✅ **Process complete notes** (full pipeline)
- ✅ **Switch between domains** (automotive, beekeeping, etc.)

## Prerequisites

1. **Claude Desktop installed** (from claude.ai/download)
2. **Python 3.12+** with mythRL dependencies
3. **MCP SDK installed**:
   ```bash
   pip install mcp
   ```

## Installation

### Step 1: Install MCP SDK

```bash
pip install mcp
```

### Step 2: Configure Claude Desktop

**On Windows:**

1. Find your Claude Desktop config file:
   ```
   %APPDATA%\Claude\claude_desktop_config.json
   ```

2. Open it and add the ExpertLoom server:
   ```json
   {
     "mcpServers": {
       "expertloom": {
         "command": "python",
         "args": [
           "c:\\Users\\blake\\Documents\\mythRL\\mcp_server\\expertloom_server.py"
         ],
         "env": {
           "PYTHONPATH": "c:\\Users\\blake\\Documents\\mythRL"
         }
       }
     }
   }
   ```

3. **Important:** Adjust the paths to match your installation directory!

4. Restart Claude Desktop

### Step 3: Verify It Works

1. Open Claude Desktop
2. Look for the 🔨 hammer icon (bottom right) - it should show "expertloom"
3. Try asking Claude:
   ```
   "Use the summarize_text tool to summarize this note:
   Checked the Corolla today at 87,450 miles. Oil looks dirty
   and front left tire at 28 PSI. Should change oil soon."
   ```

4. You should see Claude call the tool and get back a structured summary!

## Available Tools

### 1. `summarize_text`

Summarize text while preserving entities and measurements.

**Example:**
```
Use summarize_text on this:
"Checked the Corolla today at 87,450 miles. Oil looks dirty..."
```

**Returns:**
- Concise summary (2-3 sentences)
- Compression ratio
- Preserved entities and measurements

### 2. `extract_entities`

Extract entities and measurements from text.

**Example:**
```
Use extract_entities on this automotive note:
"Front left tire at 28 PSI, Corolla needs oil change"
```

**Returns:**
- Entities: Corolla → vehicle-corolla-2015, Front left tire → component-tire-front-left
- Measurements: tire_pressure_psi: 28.0

### 3. `process_note`

Complete pipeline: extract + summarize.

**Example:**
```
Process this note:
"Brake pads measured at 3mm, squealing sound when braking"
```

**Returns:**
- Extracted entities
- Extracted measurements
- Summary
- Qdrant payload structure

### 4. `list_domains`

Show available domains.

**Example:**
```
What domains are available?
```

**Returns:**
- List of installed domains (automotive, beekeeping, etc.)
- Stats for each domain

### 5. `switch_domain`

Change active domain.

**Example:**
```
Switch to beekeeping domain
```

## Usage Examples

### Example 1: Quick Summarization

**You:**
```
Summarize this automotive note:
"Today I performed a comprehensive inspection of the Corolla at 87,450 miles.
Started by checking all fluid levels. Engine oil was dark and dirty, indicating
it's time for a change. Coolant level was slightly low. Checked the tires - front
left was at 28 PSI which is too low. Brake pads measured at 3mm remaining."
```

**Claude with ExpertLoom:**
```
[Calls summarize_text tool]

Summary (15% of original)

Started by checking all fluid levels. Checked the tires - front left was at 28 PSI which is too low.

Details:
- Original: 342 chars
- Summary: 92 chars
- Entities preserved: 2/3
- Measurements preserved: 1/2

Preserved Measurements:
- tire_pressure_psi: 28.0
```

### Example 2: Entity Extraction

**You:**
```
What entities and measurements can you find in this?
"Front brake pads on the Corolla measured at 3mm, making squealing sound"
```

**Claude with ExpertLoom:**
```
[Calls extract_entities tool]

Extraction Results

Entities Found: 2
- Front Brake Pads → component-brake-pads-front (component)
- Corolla → vehicle-corolla-2015 (vehicle)

Measurements Found: 2
- brake_pad_thickness_mm: 3.0
- sound_type: squealing

Timestamp: 2024-10-24T10:30:00Z
```

### Example 3: Complete Processing

**You:**
```
Process this maintenance note:
"Oil change completed on Corolla - used 4.4 quarts of 5W-30 synthetic.
Old oil was black and thick. Reset maintenance reminder."
```

**Claude with ExpertLoom:**
```
[Calls process_note tool]

Note Processed Successfully

Domain: automotive

Entities: 2
- Corolla → vehicle-corolla-2015
- oil → fluid-engine-oil

Measurements: 2
- oil_quarts: 4.4
- oil_condition: black

Summary: (22% compression)
Oil change completed on Corolla - used 4.4 quarts of 5W-30 synthetic.

Qdrant Payload:
- Primary entity: vehicle-corolla-2015
- Entity IDs: ['vehicle-corolla-2015', 'fluid-engine-oil']
- Has measurements: True

Ready for Qdrant storage!
```

## How It Works

```
┌─────────────────────────────────────────┐
│ Claude Desktop                          │
│ (You type: "Summarize this note...")    │
└────────────┬────────────────────────────┘
             │ JSON-RPC request
             ▼
┌─────────────────────────────────────────┐
│ MCP Server (expertloom_server.py)      │
│ - Receives JSON-RPC call                │
│ - Parses parameters                     │
│ - Calls mythRL_core functions           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ ExpertLoom Backend                      │
│ - Entity resolution                     │
│ - Measurement extraction                │
│ - Summarization                         │
│ - Returns structured data               │
└────────────┬────────────────────────────┘
             │ JSON response
             ▼
┌─────────────────────────────────────────┐
│ Claude Desktop                          │
│ Shows: "Summary: ..." with details     │
└─────────────────────────────────────────┘
```

## Troubleshooting

### "Server not found" in Claude Desktop

1. Check config file path: `%APPDATA%\Claude\claude_desktop_config.json`
2. Verify Python path is correct in `command` field
3. Verify script path in `args` field
4. Restart Claude Desktop

### "Import errors" when server starts

1. Verify PYTHONPATH is set correctly in config
2. Test script manually:
   ```bash
   cd c:\Users\blake\Documents\mythRL
   python mcp_server\expertloom_server.py
   ```
3. Should see: waiting for stdin (server is running)

### "Domain not found"

1. Check that domains exist:
   ```
   mythRL\mythRL_core\domains\automotive\registry.json
   mythRL\mythRL_core\domains\beekeeping\registry.json
   ```
2. Try `list_domains` tool to see what's available

### Claude doesn't see the tools

1. Look for 🔨 icon in Claude Desktop (bottom right)
2. Click it - should show "expertloom" server
3. If red indicator, check Claude Desktop logs:
   ```
   %APPDATA%\Claude\logs\
   ```

## Development

To add new tools:

1. Add tool definition to `list_tools()` in `expertloom_server.py`
2. Implement handler function `async def tool_your_tool(args)`
3. Add case to `call_tool()` dispatcher
4. Restart Claude Desktop

## Next Steps

- [ ] Add Qdrant storage tool (store notes in vector DB)
- [ ] Add search tool (semantic search across notes)
- [ ] Add trend analysis tool (patterns over time)
- [ ] Add micropolicy evaluation tool (check alerts)
- [ ] Add reverse query tool (diagnostic questions)

## Support

Issues? Check:
- MCP documentation: https://modelcontextprotocol.io
- Claude Desktop docs: https://claude.ai/code
- ExpertLoom issues: https://github.com/yourusername/mythRL/issues
