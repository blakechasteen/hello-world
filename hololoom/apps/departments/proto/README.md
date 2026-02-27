# Proto - General-Purpose Code Agent for HoloLoom

Proto is a relaxed, context-aware code assistant deeply integrated with HoloLoom's agentic reasoning and memory systems.

**Status**: Package skeleton v1.0.0 (December 2025)
**Architecture**: Thin waist pattern with graceful degradation
**Integration**: 13 HoloLoom skills, Department protocol compliance

## Quick Start

### CLI

```bash
# Ask a question
python proto.py ask "explain recursion"

# Interactive REPL
python proto.py repl

# Review code
python proto.py review path/to/file.py

# Explain code
python proto.py explain path/to/file.py

# Refactor code
python proto.py refactor path/to/file.py

# Run tests
python proto.py test path/to/file.py
```

### Programmatic

```python
from hololoom.apps.departments.proto import ProtoEngine, ProtoConfig

async with ProtoEngine(ProtoConfig.default()) as engine:
    response = await engine.process("explain this code", context)
    print(response.content)
```

## Features

- **13 HoloLoom Skill Integrations**: review, explain, refactor, test, debug, security, performance, architecture, documentation, examples, patterns, edge-cases, optimization
- **Interactive REPL**: Command history, context loading, colored output, syntax highlighting
- **Sideloadable Abilities**: 3-tier extensibility system (skill mapping, plugin protocol, sandbox)
- **HoloLoom Integration**: Agentic reasoning (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE modes), memory system, knowledge graph
- **Graceful Degradation**: Works offline, falls back to neural reasoning if agentic unavailable
- **Relaxed Personality**: Friendly but focused, direct answers, admits uncertainty, patient with beginners

## Personality

Proto is relaxed and context-aware:

- **Friendly but focused** - No unnecessary fluff
- **Direct answers** - Gets to the point
- **Admits uncertainty** - "I'm not sure about X, but here's what I know..."
- **Patient with beginners** - Explains foundational concepts when needed
- **Efficient with experts** - Assumes knowledge, focuses on details

## Architecture

Proto implements the "thin waist" pattern where all requests flow through a single processing pipeline:

```
CLI / Programmatic API
    ↓
ProtoEngine.process()
    ├─ Intent Parser (what does user want?)
    ├─ Context Builder (gather relevant info)
    ├─ Action Selector (which ability to use?)
    ├─ Executor (run ability or agentic reasoning)
    └─ Response Formatter (format with personality)
    ↓
AbilityRegistry
    ├─ Tier 1: Skill Mapping (HoloLoom skills)
    ├─ Tier 2: Plugin Protocol (typed interface)
    └─ Tier 3: Sandbox (container isolation)
    ↓
AgenticBridge
    └─ HoloLoom Agentic Orchestrator
        ├─ Memory (retrieval & persistence)
        ├─ Reasoning Modes (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
        └─ Alignment Framework (safety, deception detection)
```

Benefits of thin waist:
- **Single integration point** - Easy to add new abilities
- **Consistent behavior** - All requests processed same way
- **Testable** - Easy to mock inputs/outputs
- **Extensible** - Abilities added without changing core
- **Observable** - All decisions logged and auditable

## Configuration

```python
from hololoom.apps.departments.proto import ProtoConfig

# Default (balanced)
config = ProtoConfig.default()
config.enable_agentic = True      # Use HoloLoom agentic reasoning
config.enable_memory = True        # Use memory system
config.reasoning_mode = "VERIFY"   # Multi-query verification

# Minimal (fast, no external services)
config = ProtoConfig.minimal()
config.enable_agentic = False
config.enable_memory = False

# Full (all features, best quality)
config = ProtoConfig.full()
config.enable_agentic = True
config.enable_memory = True
config.reasoning_mode = "RESEARCH"
config.max_reasoning_steps = 5
```

## REPL Commands

Interactive REPL for exploratory coding:

```
Proto> /help
Available commands:
  /help              - Show this help
  /quit              - Exit REPL
  /clear             - Clear screen
  /context <file>    - Load context file
  /history           - Show command history
  /save <file>       - Save session to file
  /load <file>       - Load session from file
  /explain <text>    - Explain code/concept
  /refactor <code>   - Suggest refactoring
  /test <code>       - Write tests
  /review <code>     - Code review
```

## The 13 HoloLoom Skills

Proto wraps all 13 core HoloLoom skills:

| Skill | Purpose | Example |
|-------|---------|---------|
| `review` | Code review | `proto review myfile.py` |
| `explain` | Explain code | `proto explain myfunction` |
| `refactor` | Suggest improvements | `proto refactor myfile.py` |
| `test` | Write tests | `proto test myfunction` |
| `debug` | Debug code | `proto debug "error message"` |
| `security` | Security review | `proto security myfile.py` |
| `performance` | Performance analysis | `proto performance myfile.py` |
| `architecture` | Architectural guidance | `proto architecture myproject` |
| `documentation` | Generate docs | `proto documentation myfile.py` |
| `examples` | Show examples | `proto examples "async/await"` |
| `patterns` | Design patterns | `proto patterns "async error handling"` |
| `edge-cases` | Find edge cases | `proto edge-cases myfunction` |
| `optimization` | Code optimization | `proto optimization myfile.py` |

## Abilities System

Proto supports three tiers of extensibility:

### Tier 1: Skill Mapping
Wraps existing HoloLoom skills with consistent interface.

```python
@proto.ability(tier=AbilityTier.SKILL_MAPPING)
async def my_skill(context: AbilityContext) -> AbilityResult:
    return AbilityResult(
        success=True,
        output="result",
        metadata={"type": "skill_mapping"}
    )
```

### Tier 2: Plugin Protocol
Typed interface with permission system, no file access.

```python
@proto.ability(tier=AbilityTier.PLUGIN_PROTOCOL)
async def my_plugin(context: AbilityContext) -> AbilityResult:
    # Can call HoloLoom APIs, limited I/O
    return AbilityResult(...)
```

### Tier 3: Full Sandbox
Container/process isolation for untrusted code.

```python
@proto.ability(tier=AbilityTier.SANDBOX)
async def my_sandbox(context: AbilityContext) -> AbilityResult:
    # Runs in isolated container
    # Full access to OS within sandbox
    return AbilityResult(...)
```

## Integration Points

### With HoloLoom Agentic System

Proto automatically uses HoloLoom's agentic reasoning for complex queries:

```python
# Simple query → single-pass (DIRECT mode)
response = await proto.process("What is this function?")

# Complex query → multi-query reasoning (RESEARCH mode)
response = await proto.process(
    "Analyze all tradeoffs of this algorithm",
    enable_research=True,
    max_steps=5
)
```

### With Memory System

Proto learns from interactions:

```python
# Experience (form memories)
await proto.remember("recursion", "base case terminates recursion")

# Recall (retrieve when relevant)
context = await proto.recall("recursion patterns")

# Reflect (improve from feedback)
await proto.reflect(response, feedback={"helpful": True})
```

### With Knowledge Graph

Proto understands entity relationships:

```python
# Query graph for related concepts
related = await proto.explore("recursion")
# Returns: tail recursion, TCO, stack overflow, etc.
```

## Performance

Typical latencies:

| Operation | Latency | Mode |
|-----------|---------|------|
| **Simple explain** | 50-100ms | DIRECT (neural only) |
| **Verify response** | 200-500ms | VERIFY (2-3 queries) |
| **Research topic** | 1-3s | RESEARCH (4-5 queries) |
| **Refactor code** | 100-200ms | DIRECT + skill |
| **REPL input** | 50-200ms | Depends on skill |

## Error Handling

Proto handles errors gracefully:

```python
response = await proto.process("explain this code", code)

if response.success:
    print(response.content)
else:
    # Fallback behavior
    if response.error_type == "agentic_unavailable":
        # Agentic system unavailable, used neural-only
        print("(Using neural-only mode)")
    elif response.error_type == "memory_unavailable":
        # Memory system unavailable, used neural only
        print("(No persistent memory available)")
```

## Testing

```bash
# Run all Proto tests
pytest hololoom/departments/proto/tests/ -v

# Run specific test suite
pytest hololoom/departments/proto/tests/test_engine.py -v

# With coverage
pytest hololoom/departments/proto/ --cov=hololoom.apps.departments.proto
```

## Documentation

- **Core Architecture**: See `core.py` for engine implementation
- **Domain Types**: See `domain.py` for Intent/Action/Response models
- **Abilities**: See `abilities.py` for ability registration and execution
- **Integration**: See `integration.py` for HoloLoom bridge

## License

Part of HoloLoom. See main LICENSE file.

## Version History

- **v1.0.0** (December 2025) - Package skeleton with graceful degradation
  - Proto module structure
  - Entry point (proto.py)
  - README documentation
  - Core imports with error handling
