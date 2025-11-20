# Elle Core - Farm & Kitchen Cooperative Intelligence System

**Status:** 🚧 In Development (November 2025)
**Version:** v0.1.0-alpha
**Integration:** Full MirrorCore + HoloLoom

---

## Purpose

Elle Core is the operational intelligence system for the Farm & Kitchen Cooperative (Coz). It provides:

- **Voice-editable SOPs** - Update procedures hands-free while working
- **Real-time time/profit tracking** - Automatic ROI analysis per task
- **Decision support** - Product prioritization, resource allocation, bottleneck detection
- **Knowledge retrieval** - Instant access to recipes, formulas, and procedures
- **Pattern learning** - Continuous improvement from operational data

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ELLE CORE                            │
│                  (Operational Intelligence)                 │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Voice I/O  │    │  HoloLoom    │    │ Time/Profit  │
│   Pipeline   │    │  Memory      │    │   Tracker    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        │            ┌────────┴────────┐           │
        │            │                 │           │
        ▼            ▼                 ▼           ▼
┌──────────────────────────────────────────────────────────┐
│                  KNOWLEDGE LAYER                         │
│  • SOPs (voice-editable)                                 │
│  • Recipes & Formulas                                    │
│  • Task History & Outcomes                               │
│  • Product Performance Data                              │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│              DECISION SUPPORT ENGINE                     │
│  • Product Prioritization (ROI-based)                    │
│  • Resource Allocation (time, materials, cash)           │
│  • Seasonal Planning (automatic task suggestions)        │
│  • Quality Control (SOP adherence, batch tracking)       │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│                   ANALYTICS & REPORTING                  │
│  • Real-time ROI Dashboard                               │
│  • Product Profitability Analysis                        │
│  • Labor Efficiency Trends                               │
│  • Cash Flow Projections                                 │
└──────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Voice I/O Pipeline (`elle/voice_sop.py`)

Voice-editable SOP system using HoloLoom's AudioSpinner:

```python
from elle import VoiceSOPEditor

# Hands-free SOP editing while working
editor = VoiceSOPEditor()

# "Elle, update bread SOP: increase proofing time to 45 minutes"
await editor.voice_command("update bread SOP: increase proofing time to 45 minutes")

# "Elle, what's the biochar inoculation ratio?"
answer = await editor.voice_query("what's the biochar inoculation ratio?")
# Response: "3 parts biochar to 1 part compost extract"
```

**Features:**
- Voice transcription via Whisper/Deepgram
- Natural language SOP updates ("increase X", "add step", "remove Y")
- Version control with automatic timestamps
- Hands-free retrieval during production

### 2. Time/Profit Tracker (`elle/tracker.py`)

Real-time operational tracking with automatic ROI analysis:

```python
from elle import TaskTracker

tracker = TaskTracker()

# Start a task (voice or manual)
await tracker.start("Bake bread batch 12")

# Work happens... tracker runs in background

# End task (automatic profit calculation)
result = await tracker.end("Bake bread batch 12", units=24, revenue=144.00)

# Output:
# TaskResult(
#   task="Bake bread batch 12",
#   duration_hours=2.5,
#   labor_cost=62.50,  # $25/hr × 2.5hr
#   material_cost=48.00,  # from recipe
#   total_cost=110.50,
#   revenue=144.00,
#   profit=33.50,
#   profit_margin=23.3%,
#   hourly_roi=13.40  # $33.50 / 2.5hr
# )
```

**Features:**
- Background timer (no manual tracking)
- Automatic cost calculation from recipes
- Real-time profit analysis
- ROI comparison across products
- Voice-activated start/stop

### 3. Knowledge Layer (`elle/knowledge.py`)

HoloLoom-powered memory system for all operational knowledge:

```python
from elle import ElleKnowledge

knowledge = ElleKnowledge()

# Store SOP (automatically extracted from voice or file)
await knowledge.store_sop("BREAD", content=sop_markdown)

# Retrieve SOP (RAG-powered)
sop = await knowledge.get_sop("bread")

# Query knowledge (agentic reasoning)
answer = await knowledge.query(
    "What's the best biochar to compost ratio based on our tests?",
    mode="research"  # multi-query reasoning
)

# Store task outcomes (for pattern learning)
await knowledge.store_outcome(
    task="Bread batch 12",
    sop_used="BREAD v2.1",
    quality_score=9.2,
    notes="Perfect rise, customers loved texture"
)
```

**Features:**
- HoloLoom RAG for instant retrieval
- Agentic reasoning for complex queries
- Pattern learning from outcomes
- Cross-product knowledge linking

### 4. Decision Support Engine (`elle/decisions.py`)

AI-powered operational recommendations:

```python
from elle import DecisionEngine

engine = DecisionEngine()

# Get today's recommendations
recommendations = await engine.get_daily_recommendations()

# Output:
# [
#   Recommendation(
#     action="Prioritize bread production",
#     reasoning="Bread has 67% margin and high weekly demand. ROI: $24/hr",
#     priority="HIGH",
#     estimated_profit=144.00,
#     time_required=2.5
#   ),
#   Recommendation(
#     action="Prepare biochar batch",
#     reasoning="Materials ready, testing phase needs completion",
#     priority="MEDIUM",
#     ...
#   )
# ]

# Resource allocation suggestions
allocation = await engine.allocate_resources(
    available_hours=8,
    available_cash=500,
    season="November"
)
```

**Features:**
- ROI-based prioritization
- Seasonal awareness (from schedule.md)
- Bottleneck detection
- Cash flow optimization
- Automatic task sequencing

### 5. Analytics Dashboard (`elle/dashboard.py`)

Real-time operational intelligence:

```python
from elle import ElleDashboard

dashboard = ElleDashboard()

# Generate real-time analytics
analytics = await dashboard.get_analytics()

# Tufte-style visualizations:
# - Product profitability comparison (small multiples)
# - Labor efficiency trends (sparklines)
# - Cash flow trajectory (confidence trajectory)
# - Seasonal revenue patterns (knowledge graph)
# - Task duration vs expected (waterfall chart)

# Export to HTML
dashboard.export("elle_dashboard.html")
```

---

## Voice Command Interface

### Core Commands

**SOP Management:**
- "Elle, show me the bread SOP"
- "Elle, update GOAT recipe: add cinnamon to batch 4"
- "Elle, create new SOP for deodorant"
- "Elle, what's changed in the biochar SOP?"

**Task Tracking:**
- "Elle, start baking bread"
- "Elle, pause timer" (break time)
- "Elle, finish task, made 24 loaves, sold for $144"
- "Elle, how long have I been working?"

**Knowledge Queries:**
- "Elle, what's the best-selling product this month?"
- "Elle, how much profit did bread make last week?"
- "Elle, which biochar ratio worked best?"
- "Elle, show me all GOAT flavor tests"

**Decision Support:**
- "Elle, what should I work on today?"
- "Elle, is it profitable to make deodorant now?"
- "Elle, when should I order more oats?"
- "Elle, which products need more testing?"

**Analytics:**
- "Elle, show me this week's ROI"
- "Elle, how does bread compare to honey?"
- "Elle, what's my average hourly rate?"
- "Elle, project next month's cash flow"

---

## Integration with Coz

Elle Core automatically syncs with all Coz files:

```
coz/
├── BUSINESS_PLAN_DRAFT.md → Product catalog, strategy
├── kanban.csv → Task queue, priorities
├── schedule.md → Seasonal awareness
├── financials.md → Cost models, margins
├── inventory.md → Material tracking, reorder triggers
├── research_notes.md → R&D data, experiments
└── SOP_index.md → SOP registry

↓ (Automatic sync)

HoloLoom Memory System
├── SOPs (voice-editable, version-controlled)
├── Recipes & Formulas (with cost tracking)
├── Task History (time, profit, quality)
├── Product Performance (ROI, trends)
└── Seasonal Patterns (learned from schedule)

↓ (Analytics)

Elle Dashboard
├── Real-time ROI by product
├── Labor efficiency trends
├── Cash flow projections
└── Decision recommendations
```

---

## Setup

```bash
# Install dependencies
pip install -r elle/requirements.txt

# Initialize Elle Core
python -m elle init

# Start voice interface (hands-free mode)
python -m elle voice

# Start web dashboard
python -m elle dashboard
```

---

## Roadmap

### Phase 1: Foundation (Week 1-2) ✅ In Progress
- [x] SOP schema design
- [ ] Voice input pipeline (AudioSpinner integration)
- [ ] Time/profit tracker core
- [ ] HoloLoom memory integration
- [ ] Basic CLI interface

### Phase 2: Intelligence (Week 3-4)
- [ ] Decision support engine
- [ ] Pattern learning from outcomes
- [ ] Seasonal planning automation
- [ ] Quality control tracking
- [ ] Real-time analytics

### Phase 3: Automation (Week 5-6)
- [ ] Voice command interface (hands-free)
- [ ] Automatic task suggestions
- [ ] Cash flow predictions
- [ ] Inventory reorder alerts
- [ ] Web dashboard (Tufte visualizations)

### Phase 4: Optimization (Week 7-8)
- [ ] Multi-product optimization
- [ ] Resource allocation AI
- [ ] A/B testing framework (recipe variants)
- [ ] Predictive analytics (demand forecasting)
- [ ] Mobile app (field access)

---

## Technical Stack

- **Core:** Python 3.11+, asyncio
- **AI/ML:** HoloLoom (RAG, agentic reasoning, memory)
- **Voice:** Whisper/Deepgram (transcription), pyttsx3 (TTS)
- **Storage:** HoloLoom memory (Neo4j + Qdrant), SQLite (time logs)
- **Analytics:** Pandas, NumPy, Tufte visualizations
- **Web:** FastAPI (backend), HTML/CSS/JS (dashboard)

---

## Philosophy

**"Simplify data collection through intelligence, not more forms."**

Elle Core learns from your work:
- Voice updates SOPs while you're hands-on
- Background timer tracks time automatically
- Knowledge graph connects insights across products
- AI suggests next steps based on ROI and season
- Dashboard shows what matters: profit per hour of your life

**Your time is the main resource. Elle optimizes for that.**

---

*Part of the HoloLoom ecosystem. Named "Elle" for L (Learning) - the system that learns from your farm.*
