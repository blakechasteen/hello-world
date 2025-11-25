# Voice UX Milestone 2 - Phases 3 & 4: Architectural Design

**Date**: November 22, 2025
**Status**: 🎨 **Strategic Planning Phase**
**Approach**: Metaprompt-driven design with 3x3 refinement passes

---

## Metaprompt: Self-Reflection on Design

**Question**: What is the essence of task delegation and voice feedback?

**Answer**: Task delegation is about **maintaining conversation state across time**, while voice feedback is about **keeping the user informed during execution**. Both require careful state management, clear protocols, and graceful error handling.

**Core Insight**: The best APIs are **protocol-based** and **composable**. Design for:
1. **Elegance**: Simple, clear interfaces that do one thing well
2. **Verification**: Testable state machines with clear invariants
3. **Extensibility**: Plugin architecture for future task types

---

## Phase 3: Task Delegation System

### Vision

Transform placeholder task handlers into a **production-grade task execution engine** with:
- Multi-turn conversations (task spans multiple voice interactions)
- State persistence (tasks survive between sessions)
- Progress tracking (user can ask "how's it going?")
- Context awareness (tasks understand their environment)

### Current State Analysis

**Existing Placeholders** (elle/voice/assistant.py:433-462):
```python
async def _handle_task_run(self, cmd: StructuredCommand) -> str:
    task_name = cmd.parameters.get("task_name", "")
    if not task_name:
        return "Need task name"
    # Placeholder: Task execution system to be implemented in Phase 3
    return f"Running {task_name}"

def _handle_task_stop(self) -> str:
    # Placeholder: Task control to be implemented in Phase 3
    return "Stopped"

def _handle_task_pause(self) -> str:
    return "Paused"

def _handle_task_resume(self) -> str:
    return "Resumed"

def _handle_task_status(self) -> str:
    return "No tasks running"
```

**Problems**:
- ❌ No actual task execution
- ❌ No state persistence
- ❌ No multi-turn support
- ❌ No progress tracking
- ❌ Hardcoded responses

---

## Elegance Pass 1: Task Protocol Design

### Core Abstractions

**1. TaskProtocol** (Protocol-based design):
```python
from typing import Protocol, AsyncIterator, Any
from dataclasses import dataclass
from enum import Enum

class TaskState(Enum):
    """Task lifecycle states"""
    PENDING = "pending"        # Not started
    RUNNING = "running"        # Actively executing
    PAUSED = "paused"          # Temporarily suspended
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Error occurred
    CANCELLED = "cancelled"    # User cancelled

@dataclass
class TaskProgress:
    """Progress snapshot"""
    task_id: str
    state: TaskState
    progress_pct: float        # 0.0-1.0
    current_step: str
    total_steps: int
    completed_steps: int
    status_message: str
    error: Optional[str] = None

class TaskProtocol(Protocol):
    """Protocol for executable tasks"""

    @property
    def task_id(self) -> str:
        """Unique task identifier"""
        ...

    @property
    def name(self) -> str:
        """Human-readable task name"""
        ...

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """
        Execute task with progress updates

        Yields progress updates as task runs.
        Final yield should have state=COMPLETED or FAILED.
        """
        ...

    async def pause(self) -> bool:
        """Pause execution. Returns success."""
        ...

    async def resume(self) -> bool:
        """Resume from pause. Returns success."""
        ...

    async def cancel(self) -> bool:
        """Cancel execution. Returns success."""
        ...
```

**Why Protocol-based?**
- ✅ **Elegance**: Clear contract, no implementation coupling
- ✅ **Verification**: Easy to test (mock implementations)
- ✅ **Extensibility**: New tasks just implement protocol

---

### 2. TaskRegistry (Plugin Architecture)

**Design Philosophy**: Tasks are **plugins** that register themselves.

```python
class TaskRegistry:
    """
    Plugin registry for task types

    Tasks self-register at import time using decorator:
    @register_task("analyze")
    """

    _tasks: Dict[str, Type[TaskProtocol]] = {}

    @classmethod
    def register(cls, name: str, task_class: Type[TaskProtocol]):
        """Register a task type"""
        cls._tasks[name] = task_class

    @classmethod
    def create(cls, name: str, **kwargs) -> TaskProtocol:
        """Create task instance by name"""
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}")
        return cls._tasks[name](**kwargs)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered task names"""
        return list(cls._tasks.keys())

# Decorator for easy registration
def register_task(name: str):
    """Decorator to register a task type"""
    def decorator(task_class: Type[TaskProtocol]):
        TaskRegistry.register(name, task_class)
        return task_class
    return decorator
```

**Example Usage**:
```python
@register_task("analyze")
class AnalyzeTask:
    """Analyze data with progress reporting"""

    def __init__(self, data_source: str):
        self.task_id = f"analyze_{uuid.uuid4().hex[:8]}"
        self.name = f"Analyze {data_source}"
        self.data_source = data_source
        self._state = TaskState.PENDING

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """Execute analysis with progress updates"""
        self._state = TaskState.RUNNING

        steps = ["Load data", "Process", "Generate report"]

        for i, step in enumerate(steps):
            yield TaskProgress(
                task_id=self.task_id,
                state=TaskState.RUNNING,
                progress_pct=(i / len(steps)),
                current_step=step,
                total_steps=len(steps),
                completed_steps=i,
                status_message=f"{step}..."
            )

            # Actual work here
            await asyncio.sleep(2)  # Simulate work

        # Final progress
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Done",
            total_steps=len(steps),
            completed_steps=len(steps),
            status_message="Analysis complete"
        )
```

---

### 3. TaskManager (State Management)

**Design Philosophy**: **Single source of truth** for all running tasks.

```python
class TaskManager:
    """
    Manages task lifecycle and state

    Responsibilities:
    - Track running tasks
    - Persist task state
    - Handle multi-turn conversations
    - Report progress
    """

    def __init__(self, persistence_path: str = "elle/data/tasks"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Active tasks: {task_id: (task, asyncio.Task)}
        self._active_tasks: Dict[str, Tuple[TaskProtocol, asyncio.Task]] = {}

        # Task history: {task_id: TaskProgress}
        self._task_history: Dict[str, TaskProgress] = {}

        # Load persisted tasks
        self._load_tasks()

    async def start_task(
        self,
        task_name: str,
        context: dict,
        **kwargs
    ) -> str:
        """
        Start a new task

        Returns:
            task_id: Unique identifier for task
        """
        # Create task instance
        task = TaskRegistry.create(task_name, **kwargs)

        # Start execution in background
        async_task = asyncio.create_task(
            self._execute_task(task, context)
        )

        # Track active task
        self._active_tasks[task.task_id] = (task, async_task)

        return task.task_id

    async def _execute_task(
        self,
        task: TaskProtocol,
        context: dict
    ):
        """Execute task and track progress"""
        try:
            async for progress in task.execute(context):
                # Update history
                self._task_history[task.task_id] = progress

                # Persist state
                self._persist_task(task.task_id, progress)

                # Voice feedback (Phase 4)
                # await self._announce_progress(progress)

        except Exception as e:
            # Handle failure
            progress = TaskProgress(
                task_id=task.task_id,
                state=TaskState.FAILED,
                progress_pct=0.0,
                current_step="Error",
                total_steps=0,
                completed_steps=0,
                status_message=f"Task failed",
                error=str(e)
            )
            self._task_history[task.task_id] = progress
            self._persist_task(task.task_id, progress)

        finally:
            # Cleanup
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]

    async def pause_task(self, task_id: str) -> bool:
        """Pause running task"""
        if task_id not in self._active_tasks:
            return False

        task, _ = self._active_tasks[task_id]
        return await task.pause()

    async def resume_task(self, task_id: str) -> bool:
        """Resume paused task"""
        if task_id not in self._active_tasks:
            return False

        task, _ = self._active_tasks[task_id]
        return await task.resume()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel running task"""
        if task_id not in self._active_tasks:
            return False

        task, async_task = self._active_tasks[task_id]

        # Cancel task
        success = await task.cancel()

        # Cancel asyncio task
        async_task.cancel()

        return success

    def get_task_status(self, task_id: str) -> Optional[TaskProgress]:
        """Get current task status"""
        return self._task_history.get(task_id)

    def list_active_tasks(self) -> List[TaskProgress]:
        """List all active tasks"""
        return [
            self._task_history[task_id]
            for task_id in self._active_tasks.keys()
            if task_id in self._task_history
        ]

    def _persist_task(self, task_id: str, progress: TaskProgress):
        """Persist task state to disk"""
        import json

        path = self.persistence_path / f"{task_id}.json"

        with open(path, 'w') as f:
            json.dump({
                'task_id': progress.task_id,
                'state': progress.state.value,
                'progress_pct': progress.progress_pct,
                'current_step': progress.current_step,
                'total_steps': progress.total_steps,
                'completed_steps': progress.completed_steps,
                'status_message': progress.status_message,
                'error': progress.error
            }, f, indent=2)

    def _load_tasks(self):
        """Load persisted tasks from disk"""
        # Implementation: Load .json files from persistence_path
        pass
```

---

## Elegance Pass 2: Voice Feedback Design

### Vision

Transform silent task execution into an **interactive, informative experience** with:
- Real-time progress announcements
- Confirmation prompts before destructive actions
- Interrupt handling (user says "stop" mid-task)
- Streaming responses (continuous updates)

### Core Abstractions

**1. FeedbackProtocol**:
```python
class FeedbackType(Enum):
    """Types of voice feedback"""
    PROGRESS = "progress"        # Task progress update
    CONFIRMATION = "confirmation"  # Request user confirmation
    WARNING = "warning"          # Non-critical warning
    ERROR = "error"              # Error occurred
    COMPLETION = "completion"    # Task completed

@dataclass
class VoiceFeedback:
    """Voice feedback message"""
    feedback_type: FeedbackType
    message: str
    task_id: Optional[str] = None
    requires_response: bool = False
    timeout_seconds: Optional[float] = None

class FeedbackProtocol(Protocol):
    """Protocol for voice feedback handlers"""

    async def announce(self, feedback: VoiceFeedback) -> Optional[str]:
        """
        Announce feedback to user

        Returns:
            User's response (if requires_response=True)
        """
        ...
```

**2. VoiceFeedbackManager**:
```python
class VoiceFeedbackManager:
    """
    Manages voice feedback during task execution

    Features:
    - Progress announcements (throttled)
    - Confirmation prompts
    - Interrupt handling
    - Streaming updates
    """

    def __init__(
        self,
        tts_handler: Callable[[str], Awaitable[bool]],
        stt_handler: Callable[[float], Awaitable[str]],
        throttle_seconds: float = 5.0
    ):
        """
        Initialize feedback manager

        Args:
            tts_handler: Function to speak text (async)
            stt_handler: Function to listen for speech (async)
            throttle_seconds: Minimum time between progress announcements
        """
        self.tts = tts_handler
        self.stt = stt_handler
        self.throttle_seconds = throttle_seconds

        # Track last announcement time per task
        self._last_announcement: Dict[str, float] = {}

        # Interrupt handling
        self._interrupt_event = asyncio.Event()
        self._interrupt_command: Optional[str] = None

    async def announce(self, feedback: VoiceFeedback) -> Optional[str]:
        """Announce feedback to user"""

        # Throttle progress updates
        if feedback.feedback_type == FeedbackType.PROGRESS:
            if not self._should_announce(feedback.task_id):
                return None

        # Speak message
        await self.tts(feedback.message)

        # Wait for response if needed
        if feedback.requires_response:
            timeout = feedback.timeout_seconds or 10.0
            response = await self.stt(timeout)
            return response

        return None

    def _should_announce(self, task_id: Optional[str]) -> bool:
        """Check if enough time has passed since last announcement"""
        if task_id is None:
            return True

        now = asyncio.get_event_loop().time()
        last = self._last_announcement.get(task_id, 0.0)

        if now - last >= self.throttle_seconds:
            self._last_announcement[task_id] = now
            return True

        return False

    async def request_confirmation(
        self,
        message: str,
        timeout: float = 10.0
    ) -> bool:
        """
        Request user confirmation

        Returns:
            True if confirmed, False otherwise
        """
        feedback = VoiceFeedback(
            feedback_type=FeedbackType.CONFIRMATION,
            message=f"{message}. Say yes to confirm.",
            requires_response=True,
            timeout_seconds=timeout
        )

        response = await self.announce(feedback)

        if response:
            # Check for affirmative response
            affirmative = ["yes", "yeah", "sure", "ok", "okay", "confirm"]
            return any(word in response.lower() for word in affirmative)

        return False

    async def listen_for_interrupts(self):
        """Listen for interrupt commands in background"""
        # Continuous listening for "stop", "pause", "cancel"
        # Sets _interrupt_event when detected
        pass
```

---

## Elegance Pass 3: Unified Interface

**Goal**: Simplify the APIs exposed to VoiceAssistant.

**Before** (complex):
```python
# User says "run analyze"
task_manager = TaskManager()
task_id = await task_manager.start_task("analyze", context, data_source="logs")
feedback_manager = VoiceFeedbackManager(...)
await feedback_manager.announce(...)
```

**After** (simple):
```python
# User says "run analyze"
await self.task_system.run("analyze", data_source="logs")
# Everything handled automatically:
# - Task creation
# - Background execution
# - Progress announcements
# - Completion notification
```

**Unified Interface**:
```python
class TaskSystem:
    """
    Unified task delegation system

    High-level interface that combines:
    - TaskManager (state management)
    - VoiceFeedbackManager (progress announcements)
    - TaskRegistry (plugin system)
    """

    def __init__(
        self,
        tts_handler: Callable[[str], Awaitable[bool]],
        stt_handler: Callable[[float], Awaitable[str]]
    ):
        self.task_manager = TaskManager()
        self.feedback_manager = VoiceFeedbackManager(tts_handler, stt_handler)

    async def run(self, task_name: str, **kwargs) -> str:
        """
        Run task with automatic feedback

        Returns:
            Brief confirmation message
        """
        # Start task
        task_id = await self.task_manager.start_task(
            task_name,
            context={},
            **kwargs
        )

        # Announce start
        await self.feedback_manager.announce(VoiceFeedback(
            feedback_type=FeedbackType.PROGRESS,
            message=f"Running {task_name}",
            task_id=task_id
        ))

        return f"Running {task_name}"

    async def pause(self) -> str:
        """Pause current task"""
        active = self.task_manager.list_active_tasks()

        if not active:
            return "No tasks running"

        # Pause most recent task
        task = active[-1]
        success = await self.task_manager.pause_task(task.task_id)

        if success:
            await self.feedback_manager.announce(VoiceFeedback(
                feedback_type=FeedbackType.PROGRESS,
                message="Paused",
                task_id=task.task_id
            ))
            return "Paused"
        else:
            return "Cannot pause"

    async def resume(self) -> str:
        """Resume paused task"""
        # Similar to pause()
        pass

    async def stop(self) -> str:
        """Stop current task"""
        # Similar to pause()
        pass

    async def status(self) -> str:
        """Get status of current task"""
        active = self.task_manager.list_active_tasks()

        if not active:
            return "No tasks running"

        task = active[-1]

        # Format brief status message
        pct = int(task.progress_pct * 100)
        return f"{task.current_step}: {pct}% complete"
```

---

## Verification Pass 1: State Machine Validation

**Goal**: Ensure task state transitions are correct and complete.

**State Transition Diagram**:
```
    ┌─────────┐
    │ PENDING │
    └────┬────┘
         │ start()
         ↓
    ┌─────────┐     pause()     ┌────────┐
    │ RUNNING │ ←─────────────→ │ PAUSED │
    └────┬────┘     resume()    └────────┘
         │
         ├─→ COMPLETED (success)
         ├─→ FAILED (error)
         └─→ CANCELLED (user stop)
```

**Invariants**:
1. ✅ Task starts in PENDING state
2. ✅ RUNNING → PAUSED → RUNNING is valid
3. ✅ Terminal states (COMPLETED, FAILED, CANCELLED) cannot transition
4. ✅ Progress percentage is monotonically increasing
5. ✅ completed_steps ≤ total_steps at all times

**Test Cases**:
```python
async def test_task_lifecycle():
    """Test complete task lifecycle"""
    # 1. Create task
    task = AnalyzeTask("data.csv")
    assert task._state == TaskState.PENDING

    # 2. Start execution
    progress_updates = []
    async for progress in task.execute({}):
        progress_updates.append(progress)
        assert progress.progress_pct >= 0.0
        assert progress.progress_pct <= 1.0
        assert progress.completed_steps <= progress.total_steps

    # 3. Verify completion
    final = progress_updates[-1]
    assert final.state == TaskState.COMPLETED
    assert final.progress_pct == 1.0

async def test_pause_resume():
    """Test pause/resume functionality"""
    task = AnalyzeTask("data.csv")

    # Start task
    execution = task.execute({})
    await execution.__anext__()  # Get first progress update

    # Pause
    success = await task.pause()
    assert success == True

    # Resume
    success = await task.resume()
    assert success == True
```

---

## Verification Pass 2: Multi-Turn Conversations

**Goal**: Ensure tasks work across multiple voice interactions.

**Scenario**: User starts task, leaves, comes back later.

**Test Case**:
```python
async def test_multi_turn_task():
    """Test task persists across sessions"""

    # Session 1: Start task
    assistant1 = VoiceAssistant()
    await assistant1.initialize()

    response = await assistant1.process_voice_input("run analyze")
    assert "Running" in response

    # Get task ID
    task_id = assistant1.task_system.task_manager.list_active_tasks()[0].task_id

    # Session 2: Check status (simulating restart)
    assistant2 = VoiceAssistant()
    await assistant2.initialize()

    response = await assistant2.process_voice_input("task status")
    assert "complete" in response.lower() or "running" in response.lower()
```

---

## Verification Pass 3: End-to-End Integration

**Goal**: Verify complete flow from voice command to task completion.

**Test Scenario**:
```
User: "run analyze"
Elle: "Running analyze"
[5 seconds pass]
Elle: "Processing... 30% complete"
[5 seconds pass]
Elle: "Processing... 60% complete"
User: "pause"
Elle: "Paused"
User: "resume"
Elle: "Resumed"
[5 seconds pass]
Elle: "Analysis complete"
```

---

## Extensibility Pass 1: Plugin Architecture

**Goal**: Make it trivial to add new task types.

**Example: Adding a "search" task**:
```python
# File: elle/tasks/search_task.py

from elle.voice.task_system import TaskProtocol, TaskState, TaskProgress
from elle.voice.task_system import register_task

@register_task("search")
class SearchTask:
    """Search knowledge base"""

    def __init__(self, query: str):
        self.task_id = f"search_{uuid.uuid4().hex[:8]}"
        self.name = f"Search: {query}"
        self.query = query

    async def execute(self, context: dict) -> AsyncIterator[TaskProgress]:
        """Execute search"""
        # Step 1: Query knowledge base
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.3,
            current_step="Searching",
            total_steps=2,
            completed_steps=0,
            status_message="Searching knowledge base"
        )

        results = await self._search_kb(self.query)

        # Step 2: Format results
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.RUNNING,
            progress_pct=0.7,
            current_step="Formatting",
            total_steps=2,
            completed_steps=1,
            status_message="Formatting results"
        )

        formatted = self._format_results(results)

        # Completion
        yield TaskProgress(
            task_id=self.task_id,
            state=TaskState.COMPLETED,
            progress_pct=1.0,
            current_step="Done",
            total_steps=2,
            completed_steps=2,
            status_message=formatted
        )
```

**That's it! No changes to VoiceAssistant or TaskManager needed.**

---

## Extensibility Pass 2: Streaming Response Protocol

**Goal**: Support streaming responses for LLM-style generation.

**Design**:
```python
class StreamingTaskProtocol(Protocol):
    """Protocol for tasks that stream responses token-by-token"""

    async def execute_stream(
        self,
        context: dict
    ) -> AsyncIterator[str]:
        """
        Stream response tokens

        Yields individual tokens as they're generated.
        """
        ...

# Example usage
async for token in task.execute_stream(context):
    await tts.speak_partial(token)  # Speak as generated
```

---

## Extensibility Pass 3: Future-Proof APIs

**Design Principles**:
1. ✅ **Protocol-based**: Easy to add implementations
2. ✅ **Async-first**: Non-blocking by default
3. ✅ **Context dictionaries**: Pass arbitrary data without API changes
4. ✅ **Optional parameters**: Backward compatibility via kwargs
5. ✅ **Semantic versioning**: Clear API evolution

**Example: Adding new progress fields without breaking existing code**:
```python
# Before (v1.0)
@dataclass
class TaskProgress:
    task_id: str
    state: TaskState
    progress_pct: float
    current_step: str

# After (v1.1) - backward compatible!
@dataclass
class TaskProgress:
    task_id: str
    state: TaskState
    progress_pct: float
    current_step: str
    # New fields (optional)
    estimated_time_remaining: Optional[float] = None
    resource_usage: Optional[dict] = None
```

---

## Implementation Plan

### Phase 3: Task Delegation (Week 1)

**Day 1: Foundation**
- [ ] Create task_system.py with protocols
- [ ] Implement TaskRegistry
- [ ] Implement TaskManager
- [ ] Write unit tests

**Day 2: Example Tasks**
- [ ] Implement AnalyzeTask (example)
- [ ] Implement SearchTask (example)
- [ ] Test task execution

**Day 3: Integration**
- [ ] Update VoiceAssistant to use TaskSystem
- [ ] Replace placeholder handlers
- [ ] Test end-to-end

### Phase 4: Voice Feedback (Week 2)

**Day 4: Feedback System**
- [ ] Implement VoiceFeedbackManager
- [ ] Add progress throttling
- [ ] Test announcements

**Day 5: Advanced Features**
- [ ] Implement confirmation prompts
- [ ] Add interrupt handling
- [ ] Test multi-turn conversations

**Day 6: Polish**
- [ ] Performance optimization
- [ ] Error handling
- [ ] Documentation

**Day 7: Demo**
- [ ] Create comprehensive demo
- [ ] Record demo video
- [ ] Write completion summary

---

## Success Metrics

**Phase 3**:
- ✅ 3+ task types implemented
- ✅ State persistence working
- ✅ Multi-turn conversations work
- ✅ 20+ test cases passing

**Phase 4**:
- ✅ Progress announcements working (throttled)
- ✅ Confirmation prompts working
- ✅ Interrupt handling working
- ✅ <500ms announcement latency

---

## Files to Create

**Phase 3**:
1. `elle/voice/task_system.py` (500 lines) - Core system
2. `elle/tasks/analyze_task.py` (150 lines) - Example task
3. `elle/tasks/search_task.py` (150 lines) - Example task
4. `elle/voice/test_task_system.py` (300 lines) - Tests

**Phase 4**:
5. `elle/voice/voice_feedback.py` (350 lines) - Feedback manager
6. `elle/voice/test_voice_feedback.py` (200 lines) - Tests
7. `elle/voice/demo_task_delegation.py` (250 lines) - Demo

**Documentation**:
8. `elle/voice/MILESTONE2_PHASE3_COMPLETE.md` - Summary
9. `elle/voice/MILESTONE2_PHASE4_COMPLETE.md` - Summary
10. `elle/voice/MILESTONE2_COMPLETE.md` - Overall summary

---

## Total Estimated Effort

**Phase 3**: 16-20 hours (3 days)
**Phase 4**: 12-16 hours (2-3 days)
**Total**: 28-36 hours (5-6 days)

**With 3x3 refinement passes**: +30% overhead → **36-47 hours total**

---

**Status**: 🎨 **Architecture Complete - Ready for Implementation**
**Next Step**: Begin Phase 3 Day 1 (Foundation)

**Date**: November 22, 2025
