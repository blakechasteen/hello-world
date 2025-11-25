# EventBus - Skill-to-Skill Communication

**Status**: ✅ Production Ready (Week 1 Complete - November 22, 2025)
**Performance**: 0.231ms P95 latency (43x better than 10ms target)

---

## Overview

The EventBus enables **real-time pub/sub communication** between skills in the Zero-G platform. Skills can emit events, subscribe to patterns, and build complex workflows.

### Key Features

- ⚡ **Ultra-fast**: <0.5ms event propagation
- 🔀 **Wildcard routing**: `skill.**`, `skill.*.meta`, `pattern.#`
- 🎯 **Priority execution**: CRITICAL → HIGH → NORMAL → LOW
- 🛡️ **Circuit breakers**: Auto-disable failing subscribers
- 📊 **Filtered subscriptions**: Conditional event delivery
- 🔄 **Workflow support**: Event chains with correlation IDs
- 🚀 **Async delivery**: Non-blocking propagation
- 📈 **Statistics tracking**: Built-in metrics

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt  # No external dependencies needed
```

### Basic Usage

```python
from skills.event_bus import EventBroker, SkillEvent, EventType

# Create broker
async with EventBroker() as broker:
    # Subscribe to events
    async def my_handler(event: SkillEvent):
        print(f"Received: {event.skill_name} - {event.payload}")

    await broker.subscribe(topic="skill.started", handler=my_handler)

    # Emit event
    event = SkillEvent(
        event_type=EventType.SKILL_STARTED,
        skill_name="analyzer",
        timestamp="",  # Auto-generated
        payload={"status": "ready"}
    )

    await broker.emit(event)
```

---

## Topic Patterns

### Dot Notation

Topics use hierarchical dot notation:

```
skill.started.meta.continuous_learning_capture
│     │       │    └─ Specific skill name
│     │       └────── Skill category (meta/domain/agentic)
│     └────────────── Event lifecycle
└──────────────────── Domain (skill)
```

### Wildcards

| Pattern | Behavior | Example |
|---------|----------|---------|
| `*` | Single level | `skill.*.meta` matches `skill.started.meta` |
| `**` | Multi-level | `skill.**` matches any skill event |
| `#` | Remainder | `pattern.#` matches `pattern.detected.meta.foo` |

**Examples**:

```python
# All skill events
await broker.subscribe(topic="skill.**", handler=handler)

# All meta skill events
await broker.subscribe(topic="skill.*.meta", handler=handler)

# All pattern detection events
await broker.subscribe(topic="pattern.#", handler=handler)

# Exact match (fastest)
await broker.subscribe(topic="skill.started", handler=handler)
```

---

## Advanced Features

### 1. Priority Subscriptions

Control handler execution order:

```python
from skills.event_bus.subscription import SubscriptionPriority

# Critical handlers execute first
await broker.subscribe(
    topic="security.**",
    handler=critical_handler,
    priority=SubscriptionPriority.CRITICAL.value  # 100
)

# Normal priority (default: 50)
await broker.subscribe(
    topic="skill.completed",
    handler=normal_handler
)

# Low priority handlers execute last
await broker.subscribe(
    topic="metrics.**",
    handler=metrics_handler,
    priority=SubscriptionPriority.LOW.value  # 25
)
```

### 2. Filtered Subscriptions

Conditional event delivery:

```python
# Only high-confidence events
async def high_confidence_filter(event: SkillEvent) -> bool:
    return event.payload.get('confidence', 0.0) >= 0.9

await broker.subscribe(
    topic="skill.completed.**",
    handler=handler,
    filter_fn=high_confidence_filter
)

# Only events from specific skills
async def skill_filter(event: SkillEvent) -> bool:
    return event.skill_name in ['analyzer', 'detector']

await broker.subscribe(
    topic="skill.**",
    handler=handler,
    filter_fn=skill_filter
)
```

### 3. Event Workflows

Build event chains with correlation:

```python
# Parent event
parent_event = SkillEvent(
    event_type=EventType.SKILL_STARTED,
    skill_name="analyzer",
    timestamp="",
    correlation_id="workflow-abc123",  # Link all events
    sequence=1,
    payload={"target": "document.pdf"}
)
parent_id = await broker.emit(parent_event)

# Child event (caused by parent)
child_event = SkillEvent(
    event_type=EventType.PATTERN_DETECTED,
    skill_name="detector",
    timestamp="",
    correlation_id="workflow-abc123",  # Same workflow
    causation_id=parent_id,           # What triggered this
    sequence=2,
    payload={"pattern": "code_smell"}
)
await broker.emit(child_event)

# Query events by correlation
events = await broker.get_workflow_events("workflow-abc123")
```

### 4. Circuit Breakers

Automatic failure isolation:

```python
# Handler fails 5 times consecutively
async def flaky_handler(event: SkillEvent):
    raise Exception("Service unavailable")

await broker.subscribe(topic="test", handler=flaky_handler)

# After 5 consecutive failures, circuit opens
# Handler is automatically disabled
# No more events delivered until manual intervention

# Check circuit state
stats = broker.get_subscription_stats(subscription_id)
if stats['circuit_breaker_open']:
    print("Circuit breaker open - handler disabled")
```

### 5. Async vs Sync Delivery

```python
# Non-blocking async delivery (default)
broker = EventBroker(enable_async_delivery=True)
await broker.emit(event)  # Returns immediately

# Blocking sync delivery (for tests, guaranteed order)
broker = EventBroker(enable_async_delivery=False)
await broker.emit(event)  # Waits for all handlers

# Wait for async deliveries
await broker.wait_for_delivery()
```

---

## Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| **P95 latency** | 0.231ms (43x better than 10ms target) |
| **Mean latency** | 0.132ms |
| **Throughput** | >10,000 events/sec |
| **Exact routing** | <0.5ms |
| **Wildcard routing** | <1ms (cached) |

### Optimizations

1. **Exact match fast path** - O(1) dict lookup
2. **Pattern caching** - LRU cache (1000 entries)
3. **Pre-compiled regex** - Wildcard patterns compiled once
4. **Priority pre-sorting** - Subscriptions sorted at subscribe time
5. **Semaphore concurrency** - Controlled parallel delivery

### Running Benchmarks

```bash
# From repository root
PYTHONPATH=. python -m pytest skills/event_bus/tests/benchmarks/ -v -s -m performance
```

---

## API Reference

### EventBroker

Main pub/sub coordinator.

#### Methods

**`emit(event: SkillEvent, routing_fn: Optional[Callable] = None) -> str`**

Emit event to all matching subscribers.

- Auto-generates `event_id` (UUID) if not provided
- Auto-generates `timestamp` (ISO8601) if not provided
- Returns event ID

```python
event_id = await broker.emit(event)
```

**`subscribe(topic: str, handler: EventHandler, filter_fn: Optional[EventFilter] = None, priority: int = 50, skill_name: Optional[str] = None) -> str`**

Subscribe to topic pattern.

- Returns subscription ID (UUID)
- Handler is async callable: `async def handler(event: SkillEvent)`
- Filter is async predicate: `async def filter(event: SkillEvent) -> bool`

```python
sub_id = await broker.subscribe(
    topic="skill.**",
    handler=my_handler,
    filter_fn=my_filter,
    priority=75
)
```

**`unsubscribe(subscription_id: str) -> bool`**

Unsubscribe from events.

```python
success = await broker.unsubscribe(sub_id)
```

**`unsubscribe_all(skill_name: str) -> int`**

Unsubscribe all subscriptions for a skill.

```python
count = await broker.unsubscribe_all("my_skill")
```

**`get_stats() -> dict`**

Get broker statistics.

```python
stats = broker.get_stats()
print(stats['broker']['events_emitted'])
print(stats['broker']['avg_delivery_time_ms'])
```

**`wait_for_delivery() -> None`**

Wait for all background deliveries to complete.

```python
await broker.wait_for_delivery()
```

**`close() -> None`**

Graceful shutdown.

```python
await broker.close()
```

### SkillEvent

Event data structure (enhanced with workflow support).

#### Fields

```python
@dataclass
class SkillEvent:
    event_type: EventType          # SKILL_STARTED, PATTERN_DETECTED, etc.
    skill_name: str                # Skill that emitted this
    timestamp: str                 # ISO8601 (auto-generated)
    payload: Dict[str, Any]        # Event data
    metadata: Dict[str, Any]       # Additional context

    # Event routing
    event_id: str = ""             # UUID (auto-generated)
    topic: str = ""                # Routing topic (auto-generated if not set)

    # Workflow support
    correlation_id: Optional[str] = None  # Links events in same workflow
    causation_id: Optional[str] = None    # Parent event ID
    sequence: int = 0                     # Order in chain
```

### EventType

Event type enumeration.

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

---

## Architecture

### Components

```
EventBroker (broker.py)
├── TopicRouter (topic_router.py)
│   ├── Exact subscriptions (dict)
│   └── Wildcard patterns (regex)
│
├── SubscriptionManager (subscription.py)
│   ├── Subscriptions (by ID)
│   ├── Priority sorting
│   └── Circuit breakers
│
└── Background tasks (async delivery)
```

### Data Flow

```
1. emit(event)
     ↓
2. TopicRouter.get_subscribers(topic)
     ↓ (returns matching subscription IDs)
3. SubscriptionManager.deliver_event(event, subscriber_ids)
     ↓ (sorts by priority, applies filters)
4. Subscription.execute(event)
     ↓ (circuit breaker check)
5. handler(event)
```

---

## Testing

### Run Tests

```bash
# Functional tests (12 tests)
PYTHONPATH=. python -m pytest skills/event_bus/tests/test_broker.py -v

# Performance benchmarks
PYTHONPATH=. python -m pytest skills/event_bus/tests/benchmarks/ -v -m performance -s

# All tests
PYTHONPATH=. python -m pytest skills/event_bus/tests/ -v
```

### Test Coverage

**Functional**: 12/12 passing ✅
- Basic pub/sub
- Multiple subscribers
- Wildcard patterns (*, **, #)
- Filtered subscriptions
- Priority ordering
- Circuit breaker behavior
- Async delivery
- Statistics collection

**Performance**: All benchmarks passing ✅
- Event emission latency
- Topic routing performance
- Handler execution overhead
- Concurrent throughput
- End-to-end latency

---

## Examples

See `tests/test_broker.py` for complete working examples of:
- Basic pub/sub
- Wildcard subscriptions
- Filtered subscriptions
- Priority-based execution
- Circuit breaker behavior
- Async delivery
- Statistics tracking

---

## Roadmap

### Week 2: Persistence & Resilience (Next)
- Event store (append-only log)
- Event replay (event sourcing)
- Dead letter queue
- Rate limiting
- Health monitoring

### Week 3: Workflows & Integration
- Workflow coordinator (event chains)
- Saga coordinator (distributed transactions)
- HoloLoom memory integration
- Zero-G orchestrator integration

### Week 4: Documentation & Examples
- Complete API documentation
- Usage guides
- 5+ example workflows
- Performance tuning guide

---

## Contributing

Event bus development follows the [Skills Development Guide](../SKILLS_DEVELOPMENT_GUIDE.md).

### Style Guide
- Async/await for all I/O
- Type hints on all public APIs
- Docstrings for all public methods
- Comprehensive tests for new features

### Performance Requirements
- <10ms P95 latency (target)
- >10,000 events/sec throughput
- Graceful degradation on failures

---

## License

Part of HoloLoom project. See root LICENSE file.

---

## Questions?

See [WEEK_1_COMPLETE.md](WEEK_1_COMPLETE.md) for detailed implementation notes.

**Status**: Production ready, Week 1 complete ✅
