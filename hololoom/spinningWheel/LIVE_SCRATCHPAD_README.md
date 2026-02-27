# Live Transcription Scratchpad

**Real-time audio → structured templates with live editing**

Zero-friction voice input system that automatically detects domain, populates templates, and saves to HoloLoom memory.

## Features

- 🎙️ **Live Audio Transcription** - Whisper-based local transcription
- 🎯 **Auto Domain Detection** - Detects what you're talking about (recipe, budget, SOP, etc.)
- 📝 **Smart Template Population** - Extracts structured data from natural language
- ✏️ **Live Editing** - Edit any field in real-time web UI
- 💾 **Memory Integration** - Saves directly to HoloLoom for natural language queries

## 6 Domain Templates

| Domain | Icon | Use Case | Fields |
|--------|------|----------|--------|
| **Recipe** | 🍳 | Cooking instructions | Name, ingredients, steps, timing, servings |
| **Budget** | 💰 | Personal expenses | Amount, category, merchant, recurring |
| **Expense** | 🧾 | Business expenses | Vendor, amount, receipt, reimbursable |
| **Time Tracking** | ⏱️ | Work hours | Project, task, duration, billable |
| **Bee Inspection** | 🐝 | Beekeeping notes | Hive ID, temp, queen, brood, actions |
| **SOP** | 📋 | Procedures | Title, purpose, steps, safety, frequency |

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn openai-whisper
```

### 2. Start API Server

```bash
cd HoloLoom
python server/scratchpad_api.py
```

Server starts at `http://localhost:8002`

### 3. Open Web UI

Open `hololoom/web_dashboard/live_scratchpad.html` in your browser.

### 4. Use It!

1. **Select domain** (or let it auto-detect)
2. **Click record** and speak naturally
3. **Watch template populate** in real-time
4. **Edit any field** manually if needed
5. **Finalize** to save to memory

## Usage Examples

### Recipe

**Speak:**
> "Recipe for chocolate chip cookies. Serves 24. Prep time 15 minutes, cook time 12 minutes. Ingredients: 1 cup butter, 3/4 cup sugar, 2 cups chocolate chips. First preheat oven to 375 degrees. Then cream butter and sugar. Finally add chocolate chips."

**Auto-populated:**
```yaml
recipe_name: "chocolate chip cookies"
servings: 24
prep_time: "15 minutes"
cook_time: "12 minutes"
ingredients:
  - "1 cup butter"
  - "3/4 cup sugar"
  - "2 cups chocolate chips"
instructions:
  - "Preheat oven to 375 degrees"
  - "Cream butter and sugar"
  - "Add chocolate chips"
```

### Bee Inspection

**Speak:**
> "Inspected Hive 3 today. Temperature was -5 degrees. Colony looks weak, brood pattern is spotty. I saw the queen. Honey stores are low. Noticed some varroa mites. Will need to feed soon and treat for mites."

**Auto-populated:**
```yaml
hive_id: "3"
temperature: -5
queen_seen: "yes"
brood_pattern: "spotty"
honey_stores: "low"
pests_observed: ["varroa"]
issues: ["weak_colony"]
actions_needed: ["feed", "treat"]
```

### Budget Entry

**Speak:**
> "Spent $45.99 at the grocery store for food shopping. Paid with credit card."

**Auto-populated:**
```yaml
amount: "45.99"
category: "groceries"
description: "food shopping"
merchant: "grocery store"
payment_method: "credit"
```

### SOP

**Speak:**
> "SOP for winter hive inspection. Purpose is to assess colony health during cold weather. First check the weather forecast. Next prepare your smoker and tools. Then gently open the hive. Finally close quickly to preserve heat. This should be done monthly during winter. Safety warning: always wear protective gear."

**Auto-populated:**
```yaml
title: "winter hive inspection"
purpose: "assess colony health during cold weather"
procedure_steps:
  - "Check the weather forecast"
  - "Prepare your smoker and tools"
  - "Gently open the hive"
  - "Close quickly to preserve heat"
frequency: "monthly"
safety_warnings: ["See safety notes in procedure"]
```

## Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (Microphone)   │
└────────┬────────┘
         │ Audio WebM
         ↓
┌─────────────────┐
│  FastAPI Server │ ← http://localhost:8002
│ scratchpad_api  │
└────────┬────────┘
         │
         ├─→ WhisperSpinner (transcribe)
         │
         ├─→ DomainRouter (detect domain)
         │
         ├─→ LiveScratchpad (populate template)
         │
         └─→ HoloLoom (save to memory)
```

## API Endpoints

### POST `/api/scratchpad/start`
Start recording session
```json
{
  "domain": "recipe",
  "auto_detect": false
}
```

### POST `/api/scratchpad/transcribe`
Upload audio for transcription
- Form data: `audio` file
- Returns: transcription + populated template

### GET `/api/scratchpad/template/{domain}`
Get template structure
```json
{
  "name": "recipe",
  "fields": [...],
  "defaults": {...}
}
```

### POST `/api/scratchpad/finalize`
Save to memory
```json
{
  "domain": "recipe",
  "data": {...}
}
```

### GET `/api/scratchpad/state`
Get current state

## Programmatic Usage

```python
from hololoom.spinningWheel.live_scratchpad import LiveScratchpad

scratchpad = LiveScratchpad()

# Start session
await scratchpad.start_recording('recipe')

# Update from transcription
await scratchpad.update_from_transcription(
    "Recipe for cookies. 1 cup butter, 2 cups flour..."
)

# Manual edit
scratchpad.update_field('servings', 24)

# Get current state
state = scratchpad.get_current_state()
print(state['data'])

# Finalize (creates MemoryShard)
shard = await scratchpad.finalize()
```

## Template System

### Adding Custom Templates

```python
from hololoom.spinningWheel.live_scratchpad import (
    DomainTemplate,
    TemplateField
)

WORKOUT_TEMPLATE = DomainTemplate(
    name="workout",
    fields=[
        TemplateField("date", "text", required=True),
        TemplateField("exercise", "text", required=True),
        TemplateField("sets", "number"),
        TemplateField("reps", "number"),
        TemplateField("weight", "text"),
        TemplateField("notes", "text")
    ]
)

# Register in LiveScratchpad
scratchpad.templates['workout'] = WORKOUT_TEMPLATE
```

### Field Types

- `text` - Single-line text
- `number` - Numeric value
- `currency` - Dollar amount (validates XX.XX)
- `duration` - Time duration ("15 min", "2 hours")
- `list` - Array of items

### Validation

```python
template.validate(data)
# Returns: ['field_name is required', ...]
```

## Domain Detection

Auto-detects domain from keyword patterns:

```python
from hololoom.spinningWheel.domain_router import DomainRouter

router = DomainRouter()
scores = router.detect_domain(transcription_text)

# Returns:
# [
#   DomainScore(domain='recipe', confidence=0.85, ...),
#   DomainScore(domain='budget', confidence=0.23, ...),
# ]
```

## Memory Integration

All finalized entries are saved as `MemoryShard` objects and ingested into HoloLoom:

```python
# After finalization
shard = await scratchpad.finalize()

# Ingested into HoloLoom
from hololoom import hololoom

async with HoloLoom() as loom:
    await loom.experience([shard])

    # Query naturally
    results = await loom.recall("Which hives need feeding?")
    # → Returns bee inspection shards with actions_needed: ["feed"]
```

## Demo

Run the demo to see all features:

```bash
cd mythRL
python demos/demo_live_scratchpad.py
```

Demo shows:
1. Domain detection accuracy
2. Template auto-population
3. Manual field editing
4. Finalization and shard creation
5. All 6 domains in action

## Web UI Features

- **Domain Selection** - Click to select or auto-detect
- **Recording Controls** - Start/stop with visual feedback
- **Live Transcription** - Real-time text display
- **Detection Confidence** - Visual confidence bar
- **Field Suggestions** - Quick-fill chips for common values
- **List Management** - Add/remove items dynamically
- **Validation Feedback** - Real-time error display
- **Finalize Button** - Disabled until valid

## Extending

### Add New Domain

1. **Create template** in `live_scratchpad.py`
2. **Add extraction logic** in `LiveScratchpad._extract_from_transcription()`
3. **Add detection patterns** in `domain_router.py` `DOMAIN_PATTERNS`
4. **Update UI** - Add button in `live_scratchpad.html`

### Add New Field Type

1. **Define in TemplateField** - Add to `field_type` options
2. **Add validation** in `DomainTemplate.validate()`
3. **Update UI rendering** in `live_scratchpad.html` `renderField()`

## Performance

- **Transcription**: ~2-5 seconds (depends on Whisper model size)
- **Domain detection**: <1ms (regex-based)
- **Template population**: <5ms (NLP extraction)
- **Total latency**: ~2-5 seconds (transcription dominates)

## Requirements

- **Required**: `fastapi`, `uvicorn`
- **Optional**: `openai-whisper` (for transcription)
- **Browser**: Chrome/Edge/Firefox with microphone access

## Troubleshooting

### "Whisper not available"
```bash
pip install openai-whisper
```

### "Microphone access denied"
- Check browser permissions
- Use HTTPS or localhost

### "API not available"
- Ensure server is running: `python hololoom/server/scratchpad_api.py`
- Check port 8002 is not in use

## Future Enhancements

- [ ] Streaming transcription (real-time word-by-word)
- [ ] Multi-language support
- [ ] Voice commands ("save", "clear", "edit field")
- [ ] Template versioning
- [ ] Export to YAML/JSON/Markdown
- [ ] Mobile app integration
- [ ] Offline mode (IndexedDB)

## License

Part of the HoloLoom project. See repository LICENSE.

## Author

Claude Code - November 2025
