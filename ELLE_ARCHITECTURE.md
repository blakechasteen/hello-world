# Elle Architecture: Built to Last

**Philosophy**: Simple spine, lots of room to grow, nothing brittle or cute-for-one-week.

---

## Core Principles

These are the guardrails for all architectural decisions:

### 1. LLM is policy, not glue
Elle's "brain" chooses what to do; other services do the heavy lifting (vision, simulations, scheduling). The LLM doesn't coordinate infrastructure—it makes behavioral decisions.

### 2. Event in → Decision → Command out
Elle is a small agent sitting between a stream of world events and a stream of actions. Pure decision-making core.

### 3. Separation of concerns
- **AR client**: rendering + interaction
- **Elle Core**: reasoning + behavior  
- **Services**: memory, vision, planning, optimization

### 4. Stateless per-request, stateful via memory layer
Each decision is pure given: `scene + intent + user_state + memory_snapshot`.  
Memory is a separate module with its own lifecycle.

### 5. Everything is replaceable
LLM model, memory backend, Matrix bus, AR client—all swappable via interfaces. No vendor lock-in, no framework prison.

---

## Layered Architecture

Think of it as 5 layers, from outside in:

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

---

## Layer 1: Interface Adapters

**Different front-ends, one contract.**

```
adapters/
  ar_adapter/
    # Listens to AR events: gaze, scan, voice
    # Converts them into UserIntent + SceneSnapshot stubs
    # Sends to core via HTTP/Matrix
  
  matrix_adapter/
    # Matrix room(s) → normalized event objects
    # Core responses → Matrix messages / JSON payloads
  
  cli_adapter/
    # For debugging: elle simulate --scene shed.json --intent scan_only
```

**Rule**: Adapters never contain behavior logic. They just translate.

---

## Layer 2: Orchestrator (ElleEngine)

**The thin waist of the system.**

### Responsibilities:

1. Accepts normalized request:
   ```python
   ElleRequest(
       scene: SceneSnapshot,
       intent: UserIntent,
       user: UserState,
       meta: RequestMeta
   )
   ```

2. Pulls memory snapshot
3. Calls Elle Core policy
4. Dispatches follow-ups (e.g. schedule project) to appropriate services
5. Returns an `ElleAction` to the adapter

### Pattern: Pure-ish function with injected deps

```python
class ElleEngine:
    def __init__(
        self,
        policy: EllePolicy,
        memory: MemoryStore,
        tools: ToolRegistry,
        logger: Logger,
    ): ...
    
    def handle(self, request: ElleRequest) -> ElleAction:
        # Load context
        memory_snapshot = self.memory.load_memory(request.user.id)
        
        # Make decision
        action = self.policy.decide(request, memory_snapshot)
        
        # Dispatch follow-ups
        for followup in action.followups:
            self.tools.dispatch(followup)
        
        # Record outcome
        self.memory.record_interaction(request, action)
        
        return action
```

**Benefit**: In tests you can mock everything.

---

## Layer 3: Elle Core (Policy + Prompting)

**This is where `elle_core_prompt.txt` lives.**

```
core/
  prompt/
    base_prompt.txt          # The big scroll we wrote
    prompt_builder.py        # Builds full prompt with current scene + memory
  
  policy.py                  # EllePolicy.decide(request, memory) -> ElleAction
  llm_client.py              # Interface for calling OpenAI/Anthropic/local
```

### Key modules:

#### `core/policy.py`
```python
class EllePolicy:
    def decide(self, request: ElleRequest, memory: MemorySnapshot) -> ElleAction:
        # Build prompt
        prompt = self.prompt_builder.build(request, memory)
        
        # Call LLM
        response = self.llm_client.complete(prompt)
        
        # Parse JSON into ElleAction
        return self.parser.parse(response)
```

#### `core/llm_client.py`
- Interface for calling different LLM providers
- Handles retries, timeouts, temperature, etc.
- Long-term: routing by task type

**Key idea**: Elle Core never touches Matrix, AR, disk, or network beyond the LLM client and tool interfaces. It's pure behavior.

---

## Layer 4: Domain & Services

**This is where real-world complexity lives, but in small, focused pieces.**

### a) Domain Models

```
domain/
  scene.py          # SceneSnapshot
  intent.py         # UserIntent
  action.py         # ElleAction
  task.py           # Task, Project
  world.py          # WorldArea (beds, shed, fence, house)
  layout.py         # LayoutPlan for shed reorg
  estimate.py       # TaskEstimate (time, effort)
```

Keep them small, immutable dataclasses where possible.

### b) Memory Service

```
memory/
  store.py          # MemoryStore interface
  in_memory.py      # InMemoryMemoryStore (for tests)
  sqlite.py         # SqliteMemoryStore
  mirrorcore.py     # MirrorCoreMemoryStore (later)
```

```python
class MemoryStore(Protocol):
    def load_memory(self, user_id: str) -> MemorySnapshot:
        """Load read-only snapshot for decision loop"""
        ...
    
    def record_interaction(self, request: ElleRequest, action: ElleAction):
        """Write happens after decision"""
        ...
    
    def record_task_outcome(self, task_id: str, outcome: TaskOutcome):
        """Track what actually happened"""
        ...
```

**Memory snapshot is read-only in the decision loop; writes happen after.**

### c) Tool / Skill Registry

**For the "agentic work" stuff.**

```
tools/
  registry.py           # ToolRegistry for discovery + dispatch
  vision_tool.py        # Called before Elle, produces SceneSnapshot
  layout_tool.py        # Monte Carlo / heuristics for shed/farm layouts
  task_planner_tool.py  # Converts SceneSnapshot to candidate tasks
  scheduler_tool.py     # Knows about calendars, time windows
  summarizer_tool.py    # Daily/weekly summaries
```

```python
class Tool(Protocol):
    name: str
    
    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ...
```

**Elle doesn't implement these.** She picks how and when to call them, or they're pre-called by upstream processes.

Long term: this looks very agent-framework-like, but still under your control.

---

## Layer 5: Infrastructure

```
infra/
  config/
    loader.py         # Env, files, per-profile
    profiles/
      dev.yml
      farm.yml
      studio.yml
  
  logging/
    structured.py     # Structured logs (maybe OpenTelemetry later)
  
  persistence/
    db.py             # DB clients
    storage.py        # File storage
  
  versioning/
    prompts.py        # Version tracking for prompts
    features.py       # Feature flags
```

### Crucial long-term idea: Version your prompts

- `elle_core_prompt_v1.txt`
- `elle_core_prompt_v2.txt`

Store which version generated which decisions so you can debug/regress later.

---

## Extension Story: How This Doesn't Rot

### 1. New capabilities = new tools, not new gods

If Elle needs to:
- Plan nutrition for lunches
- Help with brewing
- Manage greenhouse climate

**You add:**
- New domain models (`RecipeTask`, `BrewBatch`)
- New tools (`RecipeTool`, `BrewPlannerTool`)
- New prompt snippets ("here is how to think about brewing tasks")

**You don't** jam all of that into the core prompt. You let the core say:

> "Given this is a brew-related scene, prefer calling BrewPlannerTool with these parameters."

Then the tool returns results that get folded into the final `ElleAction`.

### 2. Symbol / Story growth

**Pattern:**
- Keep `elle_core_prompt.txt` as short myth + rules (like you have now)
- Add a separate `symbols/` directory:
  ```
  symbols/
    chimborazo.txt
    platos_cave.txt
    penelope_hands.txt
    # later: many more
  ```

At runtime, you don't need to load all symbol texts every time. Instead:
1. Load the core
2. Load 1–3 symbol snippets relevant to the current theme (e.g., clarity vs clutter, patience vs urgency)

**This keeps the prompt from bloating while still being rich.**

### 3. Testing & Simulation

**To keep this from drifting:**

#### Simulation tests:
```python
# shed_scan_tired_user.json → expected: 1 small action, medium/low priority
# bed_scan_high_pressure.json → expected: weeding task surfaced, maybe high priority
```

#### Snapshot tests for ElleAction:
You don't snapshot raw LLM output; you snapshot post-parsed `ElleAction` shape.

#### Keep a scenes/ folder with canonical scenarios:
```
scenes/
  shed_cluttered.json
  fence_needs_mowing.json
  cucumber_row_weedy.json
```

This lets you regression-test when you change prompts, models, or tools.

### 4. Deployment / Profiles

**Think early about:**

```yaml
profiles:
  dev:
    llm: local
    memory: in_memory
    tools: [vision_stub, layout_stub]
  
  farm:
    llm: anthropic
    memory: sqlite
    tools: [vision_real, layout_mc, task_planner]
  
  studio:
    llm: openai
    memory: mirrorcore
    tools: [full_suite]
```

Each profile can:
- Use different LLM providers
- Enable/disable tools
- Use different memory backends

---

## The Golden Path Flow

**One fluid motion from scan to action:**

1. **AR client** sees you slow-scan the shed → emits `ScanEvent`

2. **ar_adapter** converts to `ElleRequest` (scene maybe prefilled by VisionTool)

3. **ElleEngine.handle(request)**:
   - Loads memory snapshot
   - Calls `EllePolicy.decide(request, memory_snapshot)`

4. **Policy**:
   - Builds prompt from `elle_core_prompt.txt` + context
   - Calls LLM client
   - Parses JSON into `ElleAction`

5. **Engine**:
   - Forwards any followups to services (e.g., schedule shed project)
   - Returns `ElleAction` to ar_adapter

6. **AR client** renders:
   - `mode` → how / where Elle appears
   - `visuals` → highlight, movement
   - `utterance` → spoken line, if any

**And that's it. No extra magic.**

---

## File Structure

```
elle/
  adapters/
    ar_adapter/
      __init__.py
      ar_client.py
      event_translator.py
    matrix_adapter/
      __init__.py
      matrix_client.py
      event_translator.py
    cli_adapter/
      __init__.py
      cli.py
      simulator.py
  
  engine/
    __init__.py
    elle_engine.py
    request.py
  
  core/
    prompt/
      base_prompt.txt
      prompt_builder.py
    policy.py
    llm_client.py
    parser.py
  
  domain/
    __init__.py
    scene.py
    intent.py
    action.py
    task.py
    world.py
    layout.py
    estimate.py
  
  memory/
    __init__.py
    store.py
    in_memory.py
    sqlite.py
    mirrorcore.py
  
  tools/
    __init__.py
    registry.py
    protocol.py
    vision_tool.py
    layout_tool.py
    task_planner_tool.py
    scheduler_tool.py
    summarizer_tool.py
  
  infra/
    config/
      __init__.py
      loader.py
      profiles/
        dev.yml
        farm.yml
        studio.yml
    logging/
      __init__.py
      structured.py
    persistence/
      __init__.py
      db.py
      storage.py
    versioning/
      __init__.py
      prompts.py
      features.py
  
  symbols/
    chimborazo.txt
    platos_cave.txt
    penelope_hands.txt
  
  scenes/
    shed_cluttered.json
    fence_needs_mowing.json
    cucumber_row_weedy.json
  
  tests/
    unit/
      test_policy.py
      test_engine.py
      test_tools.py
    integration/
      test_golden_path.py
    simulation/
      test_scenarios.py
```

---

## Why This Won't Collapse

**Simple spine**: ElleEngine is ~100 lines. Request in, action out.

**Lots of room to grow**: New tools, new symbols, new adapters—all additive, not invasive.

**Nothing brittle**: Everything behind interfaces. Swap LLM providers, memory backends, AR clients without touching core logic.

**Not cute-for-one-week**: This is industrial strength, but still readable. Future Blake will understand it.

---

## Next Steps

1. Scaffold the directory structure
2. Implement domain models as frozen dataclasses
3. Build ElleEngine with injected dependencies
4. Move current prompt into core/prompt/
5. Create in-memory implementations for testing
6. Build CLI adapter for simulation
7. Write first golden path test

Then iterate from there.

---

**Short version:**

You already have a beautiful nucleus: ElleCore + the scroll we wrote.

What we just did is wrap it in a long-term skeleton:

**adapters → engine → core → tools → infra**

Elegant, deeply extensible, and very you.
