# Live Scratchpad Quick Start

Get started with voice-to-structured-data in 5 minutes.

## Install

```bash
pip install fastapi uvicorn openai-whisper
```

## Run Demo

```bash
cd mythRL
PYTHONPATH=. python demos/demo_live_scratchpad.py
```

**Output:**
- Domain detection: 73-100% accuracy
- Template auto-population: 4-7 fields from natural speech
- Finalization: Creates valid MemoryShard objects

## Start Web UI

### 1. Start API Server

```bash
cd mythRL/HoloLoom
python server/scratchpad_api.py
```

Server runs at `http://localhost:8002`

### 2. Open Browser

```bash
# Open this file in your browser:
HoloLoom/web_dashboard/live_scratchpad.html
```

### 3. Record

1. Click domain button (🍳 Recipe, 🐝 Bee Check, etc.)
2. Click "● Start Recording"
3. Speak naturally
4. Click "■ Stop Recording"
5. Watch template auto-fill
6. Edit any field
7. Click "✓ Finalize & Save"

## Example Session

**You speak:**
> "Inspected Hive 3 today. Temperature was -5 degrees. Colony looks weak, brood pattern is spotty. Saw the queen. Honey stores are low. Noticed varroa mites. Need to feed and treat."

**Template auto-fills:**
```
hive_id: "3"
temperature: -5
queen_seen: "yes"
brood_pattern: "spotty"
honey_stores: "low"
pests_observed: ["varroa"]
actions_needed: ["feed", "treat"]
```

**Save to HoloLoom**

**Query later:**
> "Which hives need feeding?"

**Returns:** Hive 3 inspection with actions_needed: ["feed"]

## Next Steps

- **Add custom domains**: See README section "Adding Custom Templates"
- **Integrate with apps**: Use programmatic API
- **Query your data**: Natural language search in HoloLoom

## Troubleshooting

**"Whisper not available"**
```bash
pip install openai-whisper
```

**"Microphone blocked"**
- Allow mic access in browser settings
- Use Chrome/Edge/Firefox

**"API server not running"**
```bash
# Terminal 1: Start server
python HoloLoom/server/scratchpad_api.py

# Terminal 2: Open browser
# Navigate to live_scratchpad.html
```

## 6 Built-in Domains

| Domain | Auto-extracts | Use For |
|--------|---------------|---------|
| 🍳 Recipe | Ingredients, steps, timing | Cooking notes |
| 🐝 Bee Inspection | Hive, temp, issues, actions | Beekeeping logs |
| 💰 Budget | Amount, category, merchant | Personal expenses |
| 🧾 Expense | Vendor, receipt, amount | Business costs |
| ⏱️ Time | Project, task, hours | Work tracking |
| 📋 SOP | Steps, safety, frequency | Procedures |

## Complete Documentation

See [LIVE_SCRATCHPAD_README.md](LIVE_SCRATCHPAD_README.md) for:
- Full API reference
- Template system details
- Domain detection tuning
- Programmatic usage
- Extension guide
