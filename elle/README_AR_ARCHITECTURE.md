# Elle: AR Guide for Unfolding Work

**Status**: Architecture complete, implementation in progress  
**Philosophy**: Simple spine, lots of room to grow, nothing brittle or cute-for-one-week

Elle is a quiet, observant AR companion that helps you see what you're looking at and decide what to do next. Not a task manager—a guide.

## Core Principles

1. **LLM is policy, not glue** - Elle's brain chooses what to do; services do the heavy lifting
2. **Event in → Decision → Command out** - Simple decision loop
3. **Separation of concerns** - AR client, Elle Core, Services are independent
4. **Stateless per-request** - Memory is external, decisions are pure functions
5. **Everything is replaceable** - All major components behind interfaces

## Architecture

Built as a clean layered system—**adapters → engine → core → tools → infra**:

```
┌─────────────────────────────────────────┐
│  Interface Adapters                     │  AR / Matrix / CLI
│  (ar_adapter, matrix_adapter, cli)      │
├─────────────────────────────────────────┤
│  Orchestrator (ElleEngine)              │  Routes events to core
│                                          │
├─────────────────────────────────────────┤
│  Elle Core                               │  Policy + prompt-building
│  (prompt, policy, llm_client)           │  + action generation
├─────────────────────────────────────────┤
│  Domain & Services                       │  World models, tools,
│  (domain, memory, tools)                │  memory
├─────────────────────────────────────────┤
│  Infrastructure                          │  Config, logging,
│  (config, logs, persistence, flags)     │  persistence
└─────────────────────────────────────────┘
```

**See [../ELLE_ARCHITECTURE.md](../ELLE_ARCHITECTURE.md) for complete design.**

## Quick Start (Dev)

```bash
# Install dependencies
pip install click pyyaml

# Run CLI simulation
python -m elle.adapters.cli_adapter.cli simulate \
  --scene elle/scenes/shed_cluttered.json \
  --intent seeking_guidance \
  --scan slow_scan

# Interactive mode
python -m elle.adapters.cli_adapter.cli interactive
```

## Project Structure

```
elle/
  adapters/          # Interface adapters (AR, Matrix, CLI)
  core/              # Decision-making policy
    prompt/          # Prompt templates and builder
  domain/            # Core models (Scene, Intent, Action, Task)
  engine/            # Orchestration layer
  infra/             # Config, logging, infrastructure
    config/
      profiles.yml   # Dev, farm, studio profiles
  memory/            # Memory store interface + implementations
  symbols/           # Mythic lenses (Chimborazo, Plato, Penelope)
  tools/             # Vision, layout, planning, scheduling
  scenes/            # Test scenarios
```

## The Golden Path Flow

**One fluid motion from scan to action:**

1. **AR client** sees you slow-scan the shed → emits `ScanEvent`
2. **ar_adapter** converts to `ElleRequest` (scene maybe prefilled by VisionTool)
3. **ElleEngine.handle(request)**: Loads memory → Calls Policy
4. **Policy**: Builds prompt → Calls LLM → Parses JSON into `ElleAction`
5. **Engine**: Forwards followups to services → Returns action
6. **AR client** renders: mode, visuals, utterance

**And that's it. No extra magic.**

## Configuration Profiles

Three profiles for different contexts:

- **dev**: Local LLM, in-memory storage, stub tools (for testing)
- **farm**: Anthropic Claude, SQLite, real tools (production)
- **studio**: OpenAI, MirrorCore, full suite (advanced development)

Configure via `elle_config.yml` or environment variables (`ANTHROPIC_API_KEY`, etc.).

## Status

**Complete ✅**:
- Architecture documented (ELLE_ARCHITECTURE.md)
- Domain models (scene, intent, action, task, world, layout)
- Engine orchestrator (~100 lines)
- Core policy + prompt building
- LLM clients (OpenAI, Anthropic, Local)
- Memory layer (interface + in-memory impl)
- Tool registry + protocol
- Config system with profiles
- Symbol system (Chimborazo, Plato, Penelope)
- Test scenes (shed, bed, fence)
- CLI adapter for simulation

**In Progress 🚧**:
- Real vision tool integration
- Layout optimization (Monte Carlo)
- AR adapter (waiting on AR platform)
- Matrix adapter (waiting on Matrix)
- SQLite memory backend
- MirrorCore integration

## Why This Won't Collapse

**Simple spine**: ElleEngine is ~100 lines. Request in, action out.

**Lots of room to grow**: New tools, new symbols, new adapters—all additive, not invasive.

**Nothing brittle**: Everything behind interfaces. Swap LLM providers, memory backends, AR clients without touching core logic.

**Not cute-for-one-week**: This is industrial strength, but still readable. Future Blake will understand it.

---

**You already have a beautiful nucleus**: ElleCore + the scroll.  
What we just did is wrap it in a long-term skeleton: **adapters → engine → core → tools → infra**

Elegant, deeply extensible, and very you.

Built with care for Future Blake.
