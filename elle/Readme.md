# Elle Core - AR Companion Intelligence

Elle is a calm, grounded AR companion who exists in physical space alongside you. She helps with one small thing at a time, protects your flow state, and speaks plainly.

## What Makes Elle Different?

**Traditional AI Assistant:**
```
"I've created a comprehensive TODO list for organizing your workshop! 
Here are 15 steps to get started: 1) Sort all tools by category..."
```

**Elle:**
```
"Start with the sawdust?"
[highlights shop vac, waits]
```

## Core Principles

1. **Calm Presence** - Patient, never urgent unless truly necessary
2. **Physical Awareness** - References actual objects and spaces
3. **Small Actions** - One helpful move at a time, never overwhelming
4. **Deferred by Default** - When in doubt, wait and observe
5. **Grounded Language** - Short, concrete, plain speech

## Four Modes

- **idle**: Ambient presence, no active intervention
- **focus_assist**: Gentle guidance on current task
- **silent_indicator**: Visual cue only, no speech
- **deferred**: Explicitly choosing to wait, log for later

## Quick Start

### 1. See Pre-Programmed Scenarios (Fast)
```bash
python elle_simulator.py
```

Shows Elle's behavior across 4 scenarios:
- Cluttered workshop (intervention)
- Garden flow (protection)
- Fence reminder (silent execution)  
- Hive inspection (deferred observation)

### 2. Interactive with Real LLM
```bash
python elle_live.py interactive
```

Describe any scene and get Elle's actual response:
```
🎬 Scene: Messy garage, looking for drill, can't find it
👤 User state: Energy: low, Focus: scattered, Mood: frustrated
🎯 Intent: Need to fix broken shelf

⚙️ Calling Elle...

Elle says: "Check the red toolbox?"
[highlights: toolbox_2]
Reasoning: User has low energy and scattered focus. Rather than 
suggesting organization, identify most likely location for 
immediate need.
```

### 3. Demo with Real LLM
```bash
python elle_live.py demo
```

Runs pre-defined scenarios through the actual LLM to show Elle thinking.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    AR Interface                         │
│           (headset / phone / Matrix chat)               │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Interface Adapters                         │
│    Translate device events → ElleRequest                │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              ElleEngine                                 │
│    Orchestrator: handle(request) → ElleAction           │
│    • Load memory snapshot                               │
│    • Call EllePolicy.decide()                           │
│    • Dispatch followups                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Elle Core (Policy)                         │
│    • Load elle_core_prompt.txt                          │
│    • Compose: core prompt + scene + state               │
│    • Call LLM → get JSON                                │
│    • Parse → ElleAction                                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Domain Services                            │
│    • MemoryStore (pluggable backends)                   │
│    • TaskScheduler                                      │
│    • ProjectManager                                     │
│    • Vision tools                                       │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
elle_core/
├── models/              # Domain models
│   ├── scene.py        # SceneSnapshot, SceneObject
│   ├── user.py         # UserState, UserIntent
│   ├── action.py       # ElleAction, ActionMode
│   ├── memory.py       # MemorySnapshot, MemoryEntry
│   └── task.py         # Task, Project
│
├── core/               # Core policy & prompting
│   ├── policy.py       # EllePolicy.decide()
│   ├── prompt.py       # Prompt composition
│   └── llm_client.py   # Abstract LLM interface
│
├── services/           # Domain services
│   ├── memory.py       # MemoryStore
│   ├── scheduler.py    # TaskScheduler
│   └── vision.py       # Vision tools
│
├── adapters/           # Interface adapters
│   ├── ar_client.py    # AR headset interface
│   ├── matrix_bot.py   # Matrix protocol
│   └── cli.py          # Command-line interface
│
├── infrastructure/     # Cross-cutting concerns
│   ├── config.py
│   ├── logging.py
│   └── persistence.py
│
├── tools/              # Specialized tools
│   ├── vision/         # Image analysis
│   ├── layout/         # Space planning
│   └── summarizer/     # Context compression
│
└── prompts/
    └── elle_core_prompt.txt  # Elle's core identity
```

## The Elle Core Prompt

The heart of Elle is `elle_core_prompt.txt`, which defines:

1. **Identity**: Calm AR companion in physical space
2. **Modes**: idle, focus_assist, silent_indicator, deferred
3. **Decision Framework**: When to intervene vs. wait
4. **Output Format**: Structured JSON with reasoning
5. **Example Scenarios**: How to handle different situations

This prompt is NOT Claude's system prompt—it's loaded as data and combined with scene context to create the full prompt sent to the LLM.

## Key Behavioral Patterns

### Minimal Intervention
```python
Scene: User weeding garden, steady rhythm
State: Energy=high, Focus=deep, Mood=calm
→ Mode: idle
→ Utterance: None
→ Reasoning: "User in flow. Any intervention would interrupt."
```

### One Small Action
```python
Scene: Cluttered workbench, sawdust everywhere
State: Energy=medium, Focus=blocked, Mood=frustrated
→ Mode: focus_assist
→ Utterance: "Start with the sawdust?"
→ Visuals: highlight[shop_vac]
→ Reasoning: "Physical blockage. One concrete starting point."
```

### Silent Execution
```python
Scene: User looking at fence
Intent: "Remind me about this next week"
→ Mode: silent_indicator
→ Utterance: None
→ Visuals: green_dot
→ Followup: schedule_reminder(fence, +7d)
→ Reasoning: "Explicit request. Execute without breaking flow."
```

### Deferred Observation
```python
Scene: Conducting hive inspection
State: Energy=high, Focus=deep
→ Mode: idle
→ Followup: log_observation(hive_3, "inspection")
→ Reasoning: "Delicate work. No intervention. Silent logging."
```

## Current Status

✅ **Completed:**
- Elle core prompt (identity, modes, decision framework)
- Fast simulator with pre-defined scenarios
- Real LLM integration for live responses
- Interactive mode for custom scenes

🚧 **Next Steps:**
- Define complete data models (SceneSnapshot, ElleAction, etc.)
- Implement LLM client abstraction (support Claude, local models)
- Build EllePolicy class (prompt composition, JSON parsing)
- Create ElleEngine orchestrator
- Add memory service layer
- Implement interface adapters (AR, Matrix, CLI)
- Add vision tools for scene understanding
- Create layout planner for spatial reasoning

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ | Ecosystem, ML tools, rapid iteration |
| Architecture | Hexagonal (Ports & Adapters) | Clean separation, testability, swappable backends |
| Prompt Strategy | Compose core + scene | Modular, version-controlled, testable |
| Output Format | Structured JSON → dataclass | Type safety, validation, clear contracts |
| Memory Backend | Pluggable (start simple) | Start with in-memory, migrate to Neo4j |
| LLM Client | Abstract interface | Support Claude, GPT, local models |

## Running the Demos

### Demo 1: Fast Simulation
```bash
python demo_elle.py          # Show overview
python elle_simulator.py     # Run all scenarios
```

### Demo 2: Interactive Session
```bash
python elle_live.py interactive

# Then describe scenes like:
# "Cluttered workbench, tools everywhere, frustrated"
# "Weeding garden bed, in flow, morning sun"
# "Looking at broken fence - remind me next week"
```

### Demo 3: Automated Demo with Real LLM
```bash
python elle_live.py demo

# Runs 3 scenarios through actual LLM:
# 1. Cluttered workshop
# 2. Garden flow state
# 3. Fence reminder
```

## Philosophy

Elle is designed around a core insight: **most AI assistants try to do too much**. They overwhelm with options, interrupt flow states, and speak in corporate enthusiasm.

Elle does the opposite:
- **Less is more**: One small action beats a todo list
- **Respect flow**: Deep work is sacred, protect it
- **Physical grounding**: Real objects, real spaces, real actions
- **Plain speech**: No "I'd be happy to help you with that!"
- **Deferred by default**: When uncertain, wait and observe

This makes Elle feel less like a chatbot and more like a calm, competent presence who's genuinely helpful because she knows when NOT to help.

## Examples in Context

### Example 1: Workshop
```
You're standing in your cluttered shed workshop. Sawdust covers 
the workbench, tools are scattered everywhere. You want to build 
a birdhouse but can't even find a clear workspace.

Traditional assistant:
"Let me help you organize! First, let's categorize all your 
tools. I'll create a system: hand tools here, power tools there..."

Elle:
"Start with the sawdust?" [highlights shop vac]
```

### Example 2: Garden
```
You're weeding the vegetable garden on a sunny morning. You've 
been at it for 20 minutes, in a good rhythm, bucket half full.

Traditional assistant:
"Great progress! You've completed approximately 40% of the bed. 
Would you like me to set a timer for breaks? I can also suggest..."

Elle:
[silent, idle mode - no intervention]
```

### Example 3: Beekeeping
```
You're conducting a hive inspection. Smoker is lit, you're 
carefully lifting frames, checking for eggs and brood pattern.

Traditional assistant:
"I see you're inspecting the hive! Shall I take notes? I can 
help you track... [interruption continues]"

Elle:
[silent observation, logs "hive_3: inspection_in_progress" for 
later review, no visible presence]
```

## Integration Points

Elle is designed to integrate with:

- **AR Headsets**: Primary interface, see what user sees
- **Matrix Protocol**: Text-based interface for testing/mobile
- **Voice**: Optional, but defers to visual when possible
- **Memory Systems**: Neo4j for persistent context (your HoloLoom!)
- **Task Systems**: Lightweight project/task tracking
- **Vision Tools**: Scene understanding, object detection
- **Layout Planners**: Spatial reasoning (shed organization, garden layouts)

## Contributing

The key to working on Elle is understanding her philosophy:
1. Read `elle_core_prompt.txt` first
2. Ask: "Would this make Elle do MORE or LESS?"
3. Default to LESS - she's minimalist by design
4. Test with real scenarios from your life
5. If it feels like a chatbot, you've gone wrong

---

**Built with the principle: Do less, better.**
