# Skills + Zero-G Integration

**Created**: November 22, 2025
**Status**: ✅ Phase 1 Complete
**Location**: `skills/` + `HoloLoom/agentic/`

---

## Overview

The Skills+Zero-G integration unifies three complementary systems into a cohesive meta-system where **skills are apps that dock to Zero-G's orbital platform**:

1. **Meta-Skills** (`skills/meta/`) - Self-improving skills about skills
2. **Domain Skills** (`skills/domain/`) - Task-focused expertise
3. **HoloLoom Agentic Skills** (`HoloLoom/agentic/`) - Production YAML skills

**Key Innovation**: Memory-informed decisioning - all skills receive WarpSpace and YarnGraph in execution context, enabling intelligent decisions based on past experiences.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install mcp  # For MCP server integration
```

### 2. Basic Usage

```python
from skills import SkillRegistry, load_all_skills

# Load all skills (meta + domain + agentic)
registry = await load_all_skills()

# List available skills
skills = registry.list_skills(category="meta")

# Get skill stats
stats = registry.get_stats()
print(f"Total skills: {stats['total_skills']}")
```

### 3. Zero-G Integration

```python
from skills.zero_g_integration import create_zero_g_orchestrator, ZeroGConfig
from skills.protocol import DockableSkill

# Create orchestrator with memory-informed decisioning
config = ZeroGConfig(
    enable_warp_space=True,
    enable_yarn_graph=True,
    enable_continuous_learning=True
)

async with await create_zero_g_orchestrator(config=config) as orchestrator:
    # Create a skill (see examples below)
    skill = MyCustomSkill()

    # Complete launch sequence: Dock → Preflight → Launch → Orbit
    result = await orchestrator.dock_and_launch(
        skill,
        parameters={"query": "What is Thompson Sampling?"}
    )

    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence}")
    print(f"Memory informed: {result.metadata.get('memory_informed')}")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Zero-G Platform                          │
│              (Orbital Infrastructure)                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Loom Core (Shared Infrastructure)                 │   │
│  │  ├─ WarpSpace (semantic retrieval)                 │   │
│  │  ├─ YarnGraph (knowledge graph)                    │   │
│  │  └─ ResonanceShed (multimodal fusion)              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Launch System                                      │   │
│  │  ├─ Preflight ← skill_tester, skill_security_analyzer│ │
│  │  ├─ Countdown                                       │   │
│  │  ├─ Lift-Off                                        │   │
│  │  └─ Orbit ← continuous_learning_capture            │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Mission Control                                    │   │
│  │  ├─ skill_gap_analyzer (identifies gaps)           │   │
│  │  ├─ token_budget_adviser (optimizes tokens)        │   │
│  │  └─ Health monitoring                              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  App Orbit Layer                                    │   │
│  │  ├─ Meta-Skills (5 skills)                         │   │
│  │  ├─ Domain Skills (expandable)                     │   │
│  │  └─ Agentic Skills (13 YAML skills)                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Event Bus (Skill-to-Skill Communication)          │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow**:
1. Skill docks to Zero-G platform
2. Preflight checks run (safety validation)
3. Skill receives execution context with WarpSpace + YarnGraph
4. Memory engine queries past experiences
5. Skill makes memory-informed decisions
6. Results stored in SpacetimeFabric for provenance
7. Continuous learning monitors outcomes
8. Mission Control tracks health and gaps

---

## Key Components

### 1. Skill Registry (`skills/__init__.py`)

Unified interface for all skill types.

```python
from skills import SkillRegistry

registry = SkillRegistry()

# Load all skills
await registry.load_meta_skills()      # 5 meta-skills
await registry.load_domain_skills()    # Domain expertise
# HoloLoom agentic skills loaded automatically (if available)

# List skills
all_skills = registry.list_skills()
meta_only = registry.list_skills(category="meta")
python_skills = registry.list_skills(domain="programming")

# Get specific skill
skill = registry.get_skill("continuous_learning_capture")

# Statistics
stats = registry.get_stats()
# {'total_skills': 18, 'meta_skills': 5, 'domain_skills': 0, 'agentic_skills': 13}
```

### 2. Docking Protocol (`skills/protocol.py`)

Defines how skills dock with Zero-G platform.

**Lifecycle States** (NASA-style):
```python
class SkillLifecycleState(Enum):
    PREFLIGHT = "preflight"      # T-10 to T-0 safety checks
    COUNTDOWN = "countdown"       # Preparation
    LIFT_OFF = "lift_off"        # Execution starting
    ORBIT = "orbit"              # Stable operation
    EVA = "eva"                  # Manual intervention
    REENTRY = "reentry"          # Shutdown
    LANDED = "landed"            # Complete
```

**DockableSkill Protocol**:
```python
from skills.protocol import DockableSkill, SkillManifest, SkillExecutionContext

class MySkill:
    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="my_skill",
            version="1.0.0",
            category="domain",
            description="Does amazing things",
            requires_warp_space=True,
            requires_yarn_graph=True
        )

    async def preflight(self, context: SkillExecutionContext):
        # Validate preconditions
        return PreflightCheckResult(passed=True, ...)

    async def execute(self, parameters: Dict, context: SkillExecutionContext):
        # Access memory systems
        warp_space = context.warp_space
        yarn_graph = context.yarn_graph

        # Make memory-informed decisions
        # ...

        return SkillExecutionResult(success=True, ...)

    async def shutdown(self, context: SkillExecutionContext):
        # Cleanup
        pass
```

### 3. Zero-G Integration (`skills/zero_g_integration.py`)

Main orchestrator integrating skills with Zero-G platform.

**Core Components**:

#### Memory-Informed Decision Engine

```python
from skills.zero_g_integration import MemoryInformedDecisionEngine

engine = MemoryInformedDecisionEngine(
    warp_space=warp_space,
    yarn_graph=yarn_graph
)

# Query memory for relevant knowledge
memories = await engine.query_memory(
    query="How to handle this situation?",
    context={"domain": "code_review"}
)

# Enhance decision context with memory
enhanced = await engine.inform_decision(
    decision_context={"query": "..."},
    skill_name="code_reviewer"
)

print(f"Memory confidence: {enhanced['memory_confidence']}")
print(f"Relevant memories: {len(enhanced['relevant_memories'])}")
```

#### Preflight Coordinator

```python
from skills.zero_g_integration import PreflightCoordinator

coordinator = PreflightCoordinator(
    config=ZeroGConfig(
        enable_skill_tester=True,
        enable_security_analyzer=True,
        preflight_timeout_seconds=30.0
    )
)

# Run complete preflight sequence
result = await coordinator.run_preflight(skill, context)

if result['passed']:
    print("All checks passed!")
else:
    print(f"Failures: {result['failures']}")
```

#### Orbit Monitor

```python
from skills.zero_g_integration import OrbitMonitor

monitor = OrbitMonitor(
    config=ZeroGConfig(
        enable_continuous_learning=True,
        learning_capture_interval=60.0  # 1 minute
    )
)

# Start monitoring skill in orbit
await monitor.start_monitoring("my_skill")

# Monitor automatically:
# - Captures learning from outcomes
# - Detects patterns (≥3 occurrences)
# - Auto-proposes new skills
# - Feeds to ReflectionBuffer
```

#### Mission Control Hub

```python
from skills.zero_g_integration import MissionControlHub

mission_control = MissionControlHub(
    config=ZeroGConfig(
        enable_gap_analyzer=True,
        gap_analysis_interval=3600.0,  # 1 hour
        enable_token_adviser=True,
        token_budget_limit=100000
    )
)

await mission_control.start()

# Register skill
await mission_control.register_skill(skill.manifest)

# Update state
await mission_control.update_state("my_skill", SkillLifecycleState.ORBIT)

# Report metrics
await mission_control.report_metrics("my_skill", {
    'success': True,
    'execution_time_ms': 150.0,
    'tokens_used': 500
})

# Get health status
health = await mission_control.health_check("my_skill")
print(f"Executions: {health['executions']}")
print(f"Success rate: {health['successes'] / health['executions']:.1%}")
```

---

## Complete Launch Sequence

The complete launch sequence integrates all components:

```python
from skills.zero_g_integration import ZeroGOrchestrator, ZeroGConfig

# 1. Create orchestrator
config = ZeroGConfig(
    enable_warp_space=True,
    enable_yarn_graph=True,
    enable_skill_tester=True,
    enable_security_analyzer=True,
    enable_continuous_learning=True,
    enable_mission_control=True
)

orchestrator = ZeroGOrchestrator(
    config=config,
    warp_space=warp_space_instance,  # Your WarpSpace
    yarn_graph=yarn_graph_instance,  # Your YarnGraph
    event_bus=event_bus_instance      # Your EventBus
)

await orchestrator.start()

# 2. Create skill
skill = MyCustomSkill()

# 3. Complete launch: Dock → Preflight → Launch → Orbit
result = await orchestrator.dock_and_launch(
    skill,
    parameters={
        "query": "Review this code",
        "code": "def foo(): pass",
        "language": "python"
    }
)

# 4. Check result
if result.success:
    print(f"✓ Skill executed successfully")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Execution time: {result.execution_time_ms:.1f}ms")
    print(f"  Memory informed: {result.metadata.get('memory_informed')}")
else:
    print(f"✗ Execution failed")
    print(f"  Errors: {result.errors}")

# 5. Get mission status
status = await orchestrator.get_mission_status()
print(f"Docked skills: {status['docked_skills']}")
print(f"Skills in orbit: {status['skills_in_orbit']}")

# 6. Shutdown
await orchestrator.stop()
```

**Lifecycle Breakdown**:

1. **Dock** (0-100ms):
   - Validate manifest
   - Check requirements (WarpSpace, YarnGraph, etc.)
   - Register with Mission Control

2. **Preflight** (100-500ms):
   - Run skill.preflight()
   - skill_tester validates functionality
   - skill_security_analyzer checks vulnerabilities
   - All checks must pass

3. **Countdown** (0-50ms):
   - Enhance parameters with memory
   - Query WarpSpace for relevant knowledge
   - Traverse YarnGraph for context
   - Build enhanced execution context

4. **Lift-Off** (immediate):
   - Update state to LIFT_OFF
   - Emit SKILL_STARTED event
   - Begin execution

5. **Orbit** (variable):
   - Execute skill logic
   - Access memory systems
   - Make informed decisions
   - Generate result

6. **Monitoring** (continuous):
   - continuous_learning_capture monitors outcomes
   - Mission Control tracks health
   - Gap analyzer identifies missing skills
   - Token adviser optimizes usage

7. **Reentry** (on shutdown):
   - Graceful cleanup
   - Emit SKILL_COMPLETED event
   - Update Mission Control

8. **Landed** (final):
   - Resources released
   - Provenance stored in SpacetimeFabric

---

## Meta-Skills

### 1. continuous_learning_capture

**Purpose**: Learns from user interactions and auto-proposes new skills.

**How it works**:
- Monitors all interactions
- Detects recurring patterns (≥3 occurrences)
- Queries memory for similar past successes
- Auto-proposes new skills when patterns emerge

**Integration**: Runs in Orbit phase, feeds to ReflectionBuffer

**Example**:
```python
# Skill detects pattern: user frequently asks about "data visualization"
# After 3+ occurrences, auto-proposes:
{
    "skill_name": "data_visualizer",
    "category": "domain",
    "justification": "5 user requests for data visualization",
    "similar_successes": ["chart_generator", "plot_creator"]
}
```

### 2. skill_gap_analyzer

**Purpose**: Identifies missing capabilities in skill ecosystem.

**How it works**:
- Queries memory for user requests
- Identifies patterns of missing capabilities
- Generates roadmap of needed skills
- Prioritizes by demand

**Integration**: Runs in Mission Control, periodic gap analysis

**Example**:
```python
# Detected gaps:
{
    "gaps": [
        {
            "capability": "pdf_generation",
            "priority": "high",
            "demand": 12,  # 12 user requests
            "suggested_name": "pdf_generator"
        },
        {
            "capability": "api_documentation",
            "priority": "medium",
            "demand": 7
        }
    ]
}
```

### 3. skill_tester

**Purpose**: Validates skill functionality before deployment.

**How it works**:
- Runs automated tests on skill
- Validates error handling
- Checks timeout behavior
- Ensures compliance with protocol

**Integration**: Runs in Preflight phase

**Tests**:
- Functionality test
- Error handling test
- Timeout behavior test
- Protocol compliance test

### 4. skill_security_analyzer

**Purpose**: Checks skills for security vulnerabilities.

**How it works**:
- Input validation checks
- Output sanitization verification
- Resource limit validation
- Privilege escalation detection
- Data leakage prevention

**Integration**: Runs in Preflight phase

**Checks**:
- Input validation
- Output sanitization
- Resource limits
- Privilege escalation
- Data leakage

### 5. token_budget_adviser

**Purpose**: Optimizes token usage across skills.

**How it works**:
- Tracks token usage per skill
- Identifies token-heavy operations
- Suggests optimizations
- Enforces budget limits

**Integration**: Runs in Mission Control

**Features**:
- Per-skill token tracking
- Budget limit enforcement
- Optimization suggestions
- Warning events when budget exceeded

---

## Event System

Skills communicate via event bus with pub/sub pattern.

**Event Types**:
```python
class EventType(Enum):
    SKILL_STARTED = "skill_started"
    SKILL_COMPLETED = "skill_completed"
    SKILL_FAILED = "skill_failed"
    PATTERN_DETECTED = "pattern_detected"
    GAP_IDENTIFIED = "gap_identified"
    SECURITY_ALERT = "security_alert"
    QUALITY_WARNING = "quality_warning"
    CUSTOM = "custom"
```

**Publishing Events**:
```python
from skills.protocol import SkillEvent, EventType

# In your skill
event = SkillEvent(
    event_type=EventType.PATTERN_DETECTED,
    skill_name="my_skill",
    timestamp="2025-11-22T10:30:00Z",
    payload={
        'pattern': 'Users often ask about X',
        'occurrences': 5
    }
)

await context.event_bus.emit(event)
```

**Subscribing to Events**:
```python
async def handle_pattern(event: SkillEvent):
    print(f"Pattern detected: {event.payload['pattern']}")

# Subscribe
subscription_id = await context.event_bus.subscribe(
    EventType.PATTERN_DETECTED,
    handle_pattern
)

# Unsubscribe later
await context.event_bus.unsubscribe(subscription_id)
```

---

## Configuration

### ZeroGConfig

Complete configuration options:

```python
from skills.zero_g_integration import ZeroGConfig

config = ZeroGConfig(
    # Loom Core (shared infrastructure)
    enable_warp_space=True,
    enable_yarn_graph=True,
    enable_resonance_shed=True,
    enable_event_bus=True,
    enable_mission_control=True,

    # Memory-informed decisioning
    memory_query_limit=10,
    memory_confidence_threshold=0.7,
    use_hybrid_retrieval=True,  # BM25 + semantic

    # Preflight configuration
    enable_skill_tester=True,
    enable_security_analyzer=True,
    preflight_timeout_seconds=30.0,

    # Orbit monitoring
    enable_continuous_learning=True,
    learning_capture_interval=60.0,  # seconds

    # Mission Control
    enable_gap_analyzer=True,
    gap_analysis_interval=3600.0,  # 1 hour
    enable_token_adviser=True,
    token_budget_limit=100000,

    # Lifecycle timeouts
    countdown_timeout=10.0,
    liftoff_timeout=120.0,
    orbit_timeout=300.0
)
```

---

## Integration with HoloLoom

### Using Existing HoloLoom Components

```python
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from skills.zero_g_integration import create_zero_g_orchestrator, ZeroGConfig

# 1. Create HoloLoom instance
config = Config.fused()
async with HoloLoom(cfg=config) as loom:

    # 2. Extract WarpSpace and YarnGraph
    warp_space = loom.warp_space
    yarn_graph = loom.yarn_graph

    # 3. Create Zero-G orchestrator with HoloLoom infrastructure
    zero_g_config = ZeroGConfig(
        enable_warp_space=True,
        enable_yarn_graph=True
    )

    orchestrator = await create_zero_g_orchestrator(
        config=zero_g_config,
        warp_space=warp_space,
        yarn_graph=yarn_graph
    )

    # 4. Use orchestrator
    skill = MySkill()
    result = await orchestrator.dock_and_launch(skill, parameters)

    # 5. Store result in HoloLoom memory
    await loom.experience(f"Skill {skill.manifest.name} result: {result.output}")
```

### Using HoloLoom Agentic Skills

```python
from skills import SkillRegistry
from HoloLoom.agentic.skill_agents import execute_skill

# Load all skills (includes HoloLoom agentic skills)
registry = await load_all_skills()

# Get agentic skills
agentic_skills = registry.list_skills(category="agentic")

# Execute HoloLoom agentic skill
result = await execute_skill(
    "code_reviewer",
    parameters={
        "code": "def foo(): pass",
        "language": "python"
    },
    config=config
)
```

---

## Demos

Run the complete integration demo:

```bash
python demos/demo_skills_zero_g_integration.py
```

**5 Demos Included**:

1. **Basic Docking**: Simple skill docking
2. **Preflight Checks**: Safety validation
3. **Memory-Informed Execution**: Using WarpSpace + YarnGraph
4. **Mission Control**: Health monitoring
5. **Complete Lifecycle**: End-to-end (Preflight → Orbit)

---

## Deployment

### Production Deployment

```python
from skills.zero_g_integration import create_zero_g_orchestrator, ZeroGConfig
from HoloLoom import HoloLoom
from HoloLoom.config import Config

# 1. Production configuration
config = Config.fused()
zero_g_config = ZeroGConfig(
    enable_warp_space=True,
    enable_yarn_graph=True,
    enable_skill_tester=True,
    enable_security_analyzer=True,
    enable_continuous_learning=True,
    enable_mission_control=True,
    enable_token_adviser=True,
    token_budget_limit=1000000,  # 1M tokens/day
    gap_analysis_interval=3600.0,  # Hourly gap analysis
    learning_capture_interval=300.0  # 5-minute learning cycles
)

# 2. Create HoloLoom instance with production backend
async with HoloLoom(cfg=config) as loom:

    # 3. Create orchestrator
    orchestrator = await create_zero_g_orchestrator(
        config=zero_g_config,
        warp_space=loom.warp_space,
        yarn_graph=loom.yarn_graph,
        event_bus=loom.event_bus
    )

    # 4. Load all skills
    registry = await load_all_skills()

    # 5. Dock skills
    for skill_name in registry.list_skills():
        skill = registry.get_skill(skill_name)
        # Dock skills implementing DockableSkill protocol
        if hasattr(skill, 'manifest'):
            success, message = await orchestrator.docking_manager.dock_skill(skill)
            if success:
                logger.info(f"Docked: {skill_name}")
            else:
                logger.error(f"Failed to dock {skill_name}: {message}")

    # 6. Run application
    # ...

    # 7. Graceful shutdown
    await orchestrator.stop()
```

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Docking** | 0-100ms | Manifest validation + requirement checks |
| **Preflight** | 100-500ms | skill_tester + skill_security_analyzer |
| **Memory Query** | 50-150ms | WarpSpace semantic search |
| **Graph Traversal** | 20-80ms | YarnGraph multi-hop queries |
| **Execution** | Variable | Depends on skill logic |
| **Orbit Monitoring** | <1ms/cycle | Background async task |
| **Gap Analysis** | 1-5s | Runs hourly in background |

**Total Overhead**: 200-800ms per skill execution (mostly preflight)

**Memory Usage**: ~10-50MB per docked skill

---

## Troubleshooting

### Skill Won't Dock

**Error**: "Missing requirements: WarpSpace, YarnGraph"

**Solution**: Skill requires infrastructure that isn't available. Either:
1. Provide required infrastructure when creating orchestrator
2. Update skill manifest to set `requires_warp_space=False`

```python
# Option 1: Provide infrastructure
orchestrator = await create_zero_g_orchestrator(
    warp_space=warp_space_instance,
    yarn_graph=yarn_graph_instance
)

# Option 2: Update manifest
manifest = SkillManifest(
    ...,
    requires_warp_space=False,  # Disable requirement
    requires_yarn_graph=False
)
```

### Preflight Failures

**Error**: "Preflight failed: [list of failures]"

**Solution**: Check preflight logs for specific failures:
- WarpSpace/YarnGraph not available → Provide infrastructure
- skill_tester failed → Fix skill functionality
- skill_security_analyzer failed → Fix security issues

### Memory Queries Returning Empty

**Issue**: `memory_informed=False` in results

**Solution**: WarpSpace/YarnGraph not integrated or empty. Either:
1. Populate memory with experiences
2. Check WarpSpace/YarnGraph connection

```python
# Populate memory
await loom.experience("Thompson Sampling balances exploration/exploitation")

# Query to verify
memories = await engine.query_memory("What is Thompson Sampling?")
print(f"Found {len(memories)} memories")
```

### Token Budget Exceeded

**Warning**: "Skill X exceeded token budget"

**Solution**: Either increase budget or optimize skill:

```python
# Option 1: Increase budget
config = ZeroGConfig(token_budget_limit=2000000)  # 2M tokens

# Option 2: Optimize skill
# - Reduce context window
# - Use fewer memory queries
# - Implement caching
```

---

## Roadmap

### Phase 1: Foundation (Weeks 1-2) ✅ COMPLETE
- ✅ Fixed broken skills imports
- ✅ Created docking protocol
- ✅ Integrated meta-skills with Zero-G
- ✅ Documentation

### Phase 2: Skill Marketplace (Weeks 3-4)
- Unified skill catalog (browse all skills)
- PackageManager for install/upgrade/remove
- Discovery interface (CLI + Web UI)
- Event bus integration

### Phase 3: Self-Improving Ecosystem (Weeks 5-6)
- Activate continuous learning loop
- Weekly gap analysis reports
- Quality gates (testing, security, optimization)
- Thompson Sampling for skill selection

### Phase 4: Advanced Capabilities (Weeks 7-8)
- Meta-prompt integration
- Cross-skill composition (skill chains)
- Spindles integration (data streaming)
- Enterprise features (B2B, multi-tenancy)

---

## API Reference

### SkillRegistry

```python
class SkillRegistry:
    """Unified registry for all skill types."""

    async def load_all_skills(self) -> None:
        """Load all skills from all sources."""

    async def load_meta_skills(self) -> None:
        """Load meta-skills from skills/meta/."""

    async def load_domain_skills(self) -> None:
        """Load domain skills from skills/domain/."""

    def list_skills(
        self,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List all skills matching filters."""

    def get_skill(self, name: str) -> Optional[Any]:
        """Get a skill by name."""

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
```

### DockableSkill Protocol

```python
class DockableSkill(Protocol):
    """Protocol for skills that can dock with Zero-G."""

    @property
    def manifest(self) -> SkillManifest:
        """Get skill manifest."""

    async def preflight(
        self,
        context: SkillExecutionContext
    ) -> PreflightCheckResult:
        """Run preflight safety checks."""

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: SkillExecutionContext
    ) -> SkillExecutionResult:
        """Execute the skill."""

    async def shutdown(
        self,
        context: SkillExecutionContext
    ) -> None:
        """Graceful shutdown."""
```

### ZeroGOrchestrator

```python
class ZeroGOrchestrator:
    """Main orchestrator integrating skills with Zero-G."""

    async def start(self) -> None:
        """Start orchestrator."""

    async def stop(self) -> None:
        """Stop orchestrator."""

    async def dock_and_launch(
        self,
        skill: DockableSkill,
        parameters: Dict[str, Any]
    ) -> SkillExecutionResult:
        """Complete launch sequence."""

    async def get_mission_status(self) -> Dict[str, Any]:
        """Get complete mission status."""
```

### Convenience Functions

```python
async def load_all_skills() -> SkillRegistry:
    """Load all skills and return registry."""

async def create_zero_g_orchestrator(
    config: Optional[ZeroGConfig] = None,
    warp_space: Optional[Any] = None,
    yarn_graph: Optional[Any] = None,
    event_bus: Optional[EventBus] = None
) -> ZeroGOrchestrator:
    """Create and start Zero-G orchestrator."""
```

---

## Credits

**Integration**: November 22, 2025
**Systems Unified**:
- Meta-Skills (5 skills)
- Domain Skills (expandable)
- HoloLoom Agentic Skills (13 YAML skills)

**Key Innovation**: Memory-informed decisioning enables skills to learn from past experiences and make intelligent decisions.
