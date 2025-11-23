"""
Event Bus for Skill-to-Skill Communication
==========================================

**Created**: November 22, 2025
**Purpose**: Elegant, nimble, and extensible event bus enabling cross-skill workflows

## Overview

The Event Bus provides pub/sub messaging between skills with:
- **Wildcard subscriptions** (*, **, # patterns)
- **Event persistence** (append-only log + replay)
- **Cross-skill workflows** (chains, fan-out/fan-in, sagas)
- **Resilience** (circuit breakers, dead letter queue, rate limiting)
- **Performance**: <10ms event propagation

## Quick Start

```python
from skills.event_bus import EventBroker, SkillEvent, EventType

# Create broker
broker = EventBroker()

# Subscribe to events
async def on_skill_started(event: SkillEvent):
    print(f"Skill {event.skill_name} started!")

subscription_id = await broker.subscribe(
    topic="skill.started.*",
    handler=on_skill_started
)

# Emit events
await broker.emit(SkillEvent(
    event_type=EventType.SKILL_STARTED,
    topic="skill.started.meta.continuous_learning_capture",
    skill_name="continuous_learning_capture",
    payload={"query": "What is Thompson Sampling?"}
))
```

## Architecture

```
EventBroker (Core)
├── TopicRouter (pattern matching)
├── SubscriptionManager (handlers, filters)
├── EventQueue (async dispatch)
└── Middleware (plugins)
```

## Topic Hierarchy

Topics use dot notation with wildcards:
- `skill.started.meta.continuous_learning_capture` - Exact topic
- `skill.*` - Single level wildcard (matches `skill.started`)
- `skill.**` - Multi-level wildcard (matches `skill.started.meta.foo`)
- `pattern.#` - Remainder wildcard (matches `pattern.detected.anything.here`)

## Event Types

- `SKILL_STARTED` - Skill execution began
- `SKILL_COMPLETED` - Skill execution succeeded
- `SKILL_FAILED` - Skill execution failed
- `PATTERN_DETECTED` - Pattern identified
- `GAP_IDENTIFIED` - Missing capability detected
- `SECURITY_ALERT` - Security concern raised
- `QUALITY_WARNING` - Quality issue detected
- `WORKFLOW_STARTED` - Cross-skill workflow initiated
- `WORKFLOW_COMPLETED` - Workflow finished
- `DATA_AVAILABLE` - Skill produced data for others
- `CUSTOM` - Custom event type
"""

from skills.event_bus.broker import EventBroker
from skills.event_bus.subscription import (
    Subscription,
    SubscriptionManager,
    SubscriptionPriority
)
from skills.event_bus.topic_router import TopicRouter, TopicPattern

# Re-export enhanced event types from protocol
from skills.protocol import (
    EventType,
    SkillEvent,
    SkillManifest,
    SkillExecutionContext,
    SkillExecutionResult,
    PreflightCheckResult,
    SkillLifecycleState,
    DockingStatus
)

__all__ = [
    # Core
    'EventBroker',
    'EventType',
    'SkillEvent',

    # Subscriptions
    'Subscription',
    'SubscriptionManager',
    'SubscriptionPriority',

    # Routing
    'TopicRouter',
    'TopicPattern',

    # Protocol types (re-exported for convenience)
    'SkillManifest',
    'SkillExecutionContext',
    'SkillExecutionResult',
    'PreflightCheckResult',
    'SkillLifecycleState',
    'DockingStatus',
]

__version__ = '1.0.0'
