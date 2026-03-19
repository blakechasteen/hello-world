"""
HookEvent — full taxonomy of lifecycle events that can trigger rituals.

These events are fired by the Loom Scheduler and other HoloLoom subsystems.
Rituals bind to these events via HookBinding.
"""

from enum import Enum, auto


class HookEvent(Enum):
    # Lifecycle
    AGENT_BORN = auto()
    AGENT_PRIMED = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    TASK_RECEIVED = auto()
    TASK_PLANNED = auto()
    TASK_EXECUTING = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TOOL_PRE = auto()
    TOOL_POST = auto()
    TOOL_FAILED = auto()

    # Memory
    MEMORY_WRITE = auto()
    MEMORY_READ = auto()
    MEMORY_FORGET = auto()

    # Semantic — fired when content/state cross meaning thresholds
    ENTITY_NEW = auto()
    DOMAIN_CROSSED = auto()
    CONTRADICTION_FOUND = auto()
    CONFIDENCE_DROP = auto()
    EXPERTISE_GAP = auto()

    # Swarm
    AGENT_SPAWNED = auto()
    AGENT_RETURNED = auto()
    CONSENSUS_NEEDED = auto()

    # Temporal
    TEMPORAL_DAILY = auto()
    TEMPORAL_WEEKLY = auto()
