# 12-Factor Agents: Implementation Roadmap

**Date**: 2025-11-18
**Status**: Planning Phase
**Target Completion**: 4-6 weeks
**Priority**: High (addresses gaps in 12-factor compliance)

---

## Executive Summary

This roadmap implements the 3 high-priority recommendations from the 12-Factor Agents analysis:

1. **Pause/Resume State Management** - Enables long-running workflows with human approvals
2. **Explicit Retry System** - Improves reliability when tools fail
3. **Centralized Prompt Management** - Better maintainability and versioning

**Expected Impact**:
- Compliance score: 87% → 93%
- Production-ready long-running workflows
- 20-30% improvement in reliability (retry system)
- Easier prompt iteration and A/B testing

---

## Phase 1: Pause/Resume State Management

**Duration**: 1-2 weeks
**Impact**: High - enables production workflows with human approvals, long-running tasks
**Compliance Gain**: 🟢 Good (75%) → ✅ Excellent (90%)

### Overview

Add explicit state serialization to enable pausing and resuming workflows at any point.

**Use Cases**:
- Human approval workflows (deploy bot waits for approval)
- Long-running API calls (pause while waiting for external service)
- Scheduled workflows (pause overnight, resume in morning)
- Resource-constrained execution (pause when quota exceeded)

### Design

#### 1.1: Create WorkflowState Dataclass

**File**: `HoloLoom/orchestrator/workflow_state.py` (new)

```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import WeavingTrace


@dataclass
class WorkflowState:
    """Complete serializable state for pause/resume."""

    # Execution State
    workflow_id: str
    current_step: int
    complexity_mode: str  # LITE/FAST/FULL/RESEARCH
    steps_completed: List[str]
    steps_remaining: List[str]
    retry_count: int
    start_time: float
    pause_time: Optional[float]

    # Business State
    query: Query
    partial_spacetime: Optional[Dict[str, Any]]  # Serialized Spacetime
    context_window: List[Dict[str, Any]]  # Messages/memories so far
    intermediate_results: Dict[str, Any]  # Tool outputs

    # Metadata
    paused_by: Optional[str]  # "human", "timeout", "quota_exceeded"
    pause_reason: Optional[str]
    resume_after: Optional[float]  # Unix timestamp

    def serialize(self) -> str:
        """Serialize to JSON string."""
        state_dict = asdict(self)
        return json.dumps(state_dict, default=str)

    @classmethod
    def deserialize(cls, json_str: str) -> 'WorkflowState':
        """Deserialize from JSON string."""
        state_dict = json.loads(json_str)
        return cls(**state_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create from dictionary."""
        return cls(**data)
```

**Tests**: `HoloLoom/orchestrator/tests/test_workflow_state.py`
- Serialization/deserialization
- Round-trip (serialize → deserialize → equals)
- Handle None values
- Handle complex nested objects

---

#### 1.2: Add State Persistence Layer

**File**: `HoloLoom/orchestrator/state_store.py` (new)

```python
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import json
import aiofiles
from pathlib import Path

from HoloLoom.orchestrator.workflow_state import WorkflowState


class StateStoreProtocol(ABC):
    """Protocol for workflow state persistence."""

    @abstractmethod
    async def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        """Save workflow state."""
        pass

    @abstractmethod
    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state."""
        pass

    @abstractmethod
    async def delete_state(self, workflow_id: str) -> None:
        """Delete workflow state."""
        pass

    @abstractmethod
    async def list_states(self) -> Dict[str, Any]:
        """List all workflow states."""
        pass


class FileStateStore(StateStoreProtocol):
    """File-based state persistence (development)."""

    def __init__(self, base_path: str = ".hololoom/workflow_states"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        filepath = self.base_path / f"{workflow_id}.json"
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(state.serialize())

    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        filepath = self.base_path / f"{workflow_id}.json"
        if not filepath.exists():
            return None
        async with aiofiles.open(filepath, 'r') as f:
            json_str = await f.read()
        return WorkflowState.deserialize(json_str)

    async def delete_state(self, workflow_id: str) -> None:
        filepath = self.base_path / f"{workflow_id}.json"
        if filepath.exists():
            filepath.unlink()

    async def list_states(self) -> Dict[str, Any]:
        states = {}
        for filepath in self.base_path.glob("*.json"):
            workflow_id = filepath.stem
            state = await self.load_state(workflow_id)
            if state:
                states[workflow_id] = {
                    'current_step': state.current_step,
                    'paused_by': state.paused_by,
                    'pause_time': state.pause_time
                }
        return states


class Neo4jStateStore(StateStoreProtocol):
    """Neo4j-based state persistence (production)."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import AsyncGraphDatabase
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        query = """
        MERGE (w:WorkflowState {workflow_id: $workflow_id})
        SET w.state_json = $state_json,
            w.updated_at = timestamp()
        """
        async with self.driver.session() as session:
            await session.run(query, workflow_id=workflow_id, state_json=state.serialize())

    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        query = """
        MATCH (w:WorkflowState {workflow_id: $workflow_id})
        RETURN w.state_json AS state_json
        """
        async with self.driver.session() as session:
            result = await session.run(query, workflow_id=workflow_id)
            record = await result.single()
            if record:
                return WorkflowState.deserialize(record['state_json'])
        return None

    async def delete_state(self, workflow_id: str) -> None:
        query = """
        MATCH (w:WorkflowState {workflow_id: $workflow_id})
        DELETE w
        """
        async with self.driver.session() as session:
            await session.run(query, workflow_id=workflow_id)

    async def list_states(self) -> Dict[str, Any]:
        query = """
        MATCH (w:WorkflowState)
        RETURN w.workflow_id AS workflow_id, w.state_json AS state_json
        """
        states = {}
        async with self.driver.session() as session:
            result = await session.run(query)
            async for record in result:
                state = WorkflowState.deserialize(record['state_json'])
                states[record['workflow_id']] = {
                    'current_step': state.current_step,
                    'paused_by': state.paused_by,
                    'pause_time': state.pause_time
                }
        return states

    async def close(self):
        await self.driver.close()


def create_state_store(backend: str = "file", **kwargs) -> StateStoreProtocol:
    """Factory for state stores."""
    if backend == "file":
        return FileStateStore(**kwargs)
    elif backend == "neo4j":
        return Neo4jStateStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

**Tests**: `HoloLoom/orchestrator/tests/test_state_store.py`
- File-based save/load/delete
- Neo4j save/load/delete (if Neo4j available)
- List all states
- Handle missing states gracefully

---

#### 1.3: Modify WeavingOrchestrator

**File**: `HoloLoom/weaving_orchestrator.py` (modify)

Add pause/resume methods to `WeavingOrchestrator`:

```python
from HoloLoom.orchestrator.workflow_state import WorkflowState
from HoloLoom.orchestrator.state_store import create_state_store
import uuid


class WeavingOrchestrator:
    """Main orchestrator with pause/resume support."""

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory: Optional[MemoryBackend] = None,
        state_store_backend: str = "file",
        **state_store_kwargs
    ):
        # ... existing init ...
        self.state_store = create_state_store(state_store_backend, **state_store_kwargs)
        self._active_workflow_id: Optional[str] = None

    async def start_workflow(
        self,
        query: Query,
        workflow_id: Optional[str] = None
    ) -> str:
        """
        Start a new workflow (with pause/resume support).

        Returns workflow_id for later resume.
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())

        self._active_workflow_id = workflow_id

        # Create initial state
        state = WorkflowState(
            workflow_id=workflow_id,
            current_step=0,
            complexity_mode=self.config.complexity_mode.value,
            steps_completed=[],
            steps_remaining=self._get_planned_steps(),
            retry_count=0,
            start_time=time.time(),
            pause_time=None,
            query=query,
            partial_spacetime=None,
            context_window=[],
            intermediate_results={},
            paused_by=None,
            pause_reason=None,
            resume_after=None
        )

        # Save initial state
        await self.state_store.save_state(workflow_id, state)

        return workflow_id

    async def pause_workflow(
        self,
        workflow_id: Optional[str] = None,
        paused_by: str = "user",
        pause_reason: Optional[str] = None,
        resume_after: Optional[float] = None
    ) -> None:
        """
        Pause the current workflow.

        Args:
            workflow_id: Workflow to pause (defaults to active)
            paused_by: Who/what paused ("user", "timeout", "quota")
            pause_reason: Human-readable reason
            resume_after: Unix timestamp to auto-resume
        """
        if workflow_id is None:
            workflow_id = self._active_workflow_id

        if workflow_id is None:
            raise ValueError("No active workflow to pause")

        # Load current state
        state = await self.state_store.load_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Update pause metadata
        state.pause_time = time.time()
        state.paused_by = paused_by
        state.pause_reason = pause_reason
        state.resume_after = resume_after

        # Serialize current orchestrator state
        state.partial_spacetime = self._serialize_partial_spacetime()
        state.context_window = self._serialize_context_window()
        state.intermediate_results = self._serialize_intermediate_results()

        # Save updated state
        await self.state_store.save_state(workflow_id, state)

    async def resume_workflow(self, workflow_id: str) -> Spacetime:
        """
        Resume a paused workflow.

        Args:
            workflow_id: Workflow to resume

        Returns:
            Completed Spacetime
        """
        # Load state
        state = await self.state_store.load_state(workflow_id)
        if state is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Restore orchestrator state
        self._restore_from_state(state)
        self._active_workflow_id = workflow_id

        # Continue weaving from where we left off
        spacetime = await self._continue_weaving(state)

        # Cleanup state (workflow complete)
        await self.state_store.delete_state(workflow_id)

        return spacetime

    def _get_planned_steps(self) -> List[str]:
        """Get planned steps based on complexity mode."""
        if self.config.complexity_mode == ComplexityMode.LITE:
            return ['Extract', 'Route', 'Execute']
        elif self.config.complexity_mode == ComplexityMode.FAST:
            return ['Pattern', 'Extract', 'Temporal', 'Route', 'Execute']
        # ... etc.

    def _serialize_partial_spacetime(self) -> Optional[Dict[str, Any]]:
        """Serialize partial spacetime if exists."""
        # Implementation depends on current state tracking
        # May need to add intermediate spacetime to orchestrator
        pass

    def _serialize_context_window(self) -> List[Dict[str, Any]]:
        """Serialize current context window."""
        # Implementation depends on how context is tracked
        pass

    def _serialize_intermediate_results(self) -> Dict[str, Any]:
        """Serialize intermediate tool results."""
        pass

    def _restore_from_state(self, state: WorkflowState) -> None:
        """Restore orchestrator from saved state."""
        # Restore query
        self.current_query = state.query

        # Restore context
        # ... implementation ...

        # Restore intermediate results
        # ... implementation ...

    async def _continue_weaving(self, state: WorkflowState) -> Spacetime:
        """Continue weaving from saved state."""
        # Resume from state.current_step
        # Execute remaining steps in state.steps_remaining
        # ... implementation ...
        pass
```

**Tests**: `HoloLoom/tests/integration/test_pause_resume.py`
- Start workflow → pause → resume → complete
- Save state mid-weaving
- Restore state correctly
- Handle missing workflows
- Handle corrupted state gracefully

---

#### 1.4: Add ChronoTrigger Support

**File**: `HoloLoom/chrono/trigger.py` (modify)

Add state checkpointing to `ChronoTrigger`:

```python
class ChronoTrigger:
    """Temporal control with state checkpointing."""

    def __init__(
        self,
        max_steps: int = 10,
        timeout_ms: float = 5000,
        checkpoint_callback: Optional[Callable] = None
    ):
        self.max_steps = max_steps
        self.timeout_ms = timeout_ms
        self.checkpoint_callback = checkpoint_callback
        self.steps_taken = 0
        self.start_time = time.time()

    def should_continue(self) -> bool:
        """Check if loop should continue (with checkpointing)."""
        # Check max steps
        if self.steps_taken >= self.max_steps:
            return False

        # Check timeout
        elapsed_ms = (time.time() - self.start_time) * 1000
        if elapsed_ms >= self.timeout_ms:
            return False

        # Checkpoint every N steps
        if self.checkpoint_callback and self.steps_taken % 3 == 0:
            self.checkpoint_callback()

        self.steps_taken += 1
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get current trigger state."""
        return {
            'steps_taken': self.steps_taken,
            'start_time': self.start_time,
            'max_steps': self.max_steps,
            'timeout_ms': self.timeout_ms
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore trigger state."""
        self.steps_taken = state['steps_taken']
        self.start_time = state['start_time']
        self.max_steps = state['max_steps']
        self.timeout_ms = state['timeout_ms']
```

---

#### 1.5: Demo Example

**File**: `demos/demo_pause_resume_workflow.py` (new)

```python
"""
Demo: Pause/Resume Workflow

Shows how to pause a long-running workflow and resume later.
Use case: Human approval in deploy bot.
"""

import asyncio
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query


async def main():
    config = Config.fused()

    async with WeavingOrchestrator(
        cfg=config,
        state_store_backend="file",
        base_path=".hololoom/demo_states"
    ) as orch:
        # Start workflow
        query = Query(text="Deploy backend to production")
        workflow_id = await orch.start_workflow(query)
        print(f"Started workflow: {workflow_id}")

        # Simulate some processing...
        # (In real deploy bot, would get to "need approval" step)

        # Pause workflow (waiting for human approval)
        await orch.pause_workflow(
            workflow_id=workflow_id,
            paused_by="human_approval_required",
            pause_reason="Deployment requires approval from DevOps team"
        )
        print(f"Paused workflow: {workflow_id}")
        print("Waiting for approval...")

        # Simulate approval delay (e.g., Slack notification → human clicks approve)
        await asyncio.sleep(2)
        print("Approval received! Resuming...")

        # Resume workflow
        spacetime = await orch.resume_workflow(workflow_id)
        print(f"Workflow completed: {spacetime.response}")
        print(f"Confidence: {spacetime.confidence:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Deliverables

- [ ] `HoloLoom/orchestrator/workflow_state.py` - WorkflowState dataclass
- [ ] `HoloLoom/orchestrator/state_store.py` - State persistence layer
- [ ] `HoloLoom/weaving_orchestrator.py` - Add pause/resume methods
- [ ] `HoloLoom/chrono/trigger.py` - Add checkpointing support
- [ ] `demos/demo_pause_resume_workflow.py` - Demo example
- [ ] Tests (80%+ coverage)
- [ ] Documentation update in CLAUDE.md

---

## Phase 2: Explicit Retry System

**Duration**: 1-2 weeks
**Impact**: Medium-High - 20-30% improvement in reliability
**Compliance Gain**: 🟡 Fair (60%) → ✅ Excellent (90%)

### Overview

Add explicit retry loop with smart error summarization to handle tool failures gracefully.

**Features**:
- Max retry count with exponential backoff
- Error summarization (not full stack traces)
- Clear resolved errors from context
- Learn from retries (Thompson Sampling updates)

### Design

#### 2.1: Create RetryPolicy

**File**: `HoloLoom/orchestrator/retry.py` (new)

```python
from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum
import asyncio
import time


class BackoffStrategy(Enum):
    """Backoff strategies for retries."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_retries: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay_ms: float = 1000  # 1 second
    max_delay_ms: float = 30000  # 30 seconds
    error_summarization: bool = True
    clear_resolved_errors: bool = True
    retry_on_exceptions: tuple = (Exception,)
    summarization_fn: Optional[Callable] = None

    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for given retry attempt."""
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.base_delay_ms
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay_ms * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay_ms * (2 ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            fib = self._fibonacci(attempt)
            delay = self.base_delay_ms * fib
        else:
            delay = self.base_delay_ms

        return min(delay, self.max_delay_ms)

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b


class RetryManager:
    """Manages retry logic with error summarization."""

    def __init__(self, policy: RetryPolicy):
        self.policy = policy
        self.error_history: List[str] = []
        self.resolved_errors: Set[str] = []

    async def execute_with_retry(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            fn: Async function to execute
            *args, **kwargs: Arguments to fn

        Returns:
            Result of fn

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(1, self.policy.max_retries + 1):
            try:
                # Execute function
                result = await fn(*args, **kwargs)

                # Success - clear resolved errors
                if self.policy.clear_resolved_errors:
                    self.error_history.clear()

                return result

            except self.policy.retry_on_exceptions as e:
                last_exception = e

                # Summarize error
                error_summary = self._summarize_error(e)
                self.error_history.append(error_summary)

                # Log retry
                print(f"Retry {attempt}/{self.policy.max_retries}: {error_summary}")

                # Last attempt - don't wait
                if attempt == self.policy.max_retries:
                    break

                # Backoff
                delay_ms = self.policy.get_delay_ms(attempt)
                await asyncio.sleep(delay_ms / 1000)

        # All retries failed
        raise last_exception

    def _summarize_error(self, error: Exception) -> str:
        """Summarize error (not full stack trace)."""
        if self.policy.summarization_fn:
            return self.policy.summarization_fn(error)

        # Default summarization
        return f"{error.__class__.__name__}: {str(error)[:200]}"

    def get_error_context(self) -> str:
        """Get error context for LLM (summarized)."""
        if not self.error_history:
            return ""

        # Only include recent errors (last 3)
        recent_errors = self.error_history[-3:]
        return "\n".join([f"- {err}" for err in recent_errors])
```

**Tests**: `HoloLoom/orchestrator/tests/test_retry.py`
- Retry with exponential backoff
- Retry with different strategies
- Error summarization
- Clear resolved errors
- Max retries exceeded

---

#### 2.2: Integrate with WeavingOrchestrator

**File**: `HoloLoom/weaving_orchestrator.py` (modify)

```python
from HoloLoom.orchestrator.retry import RetryPolicy, RetryManager


class WeavingOrchestrator:
    """Orchestrator with retry support."""

    def __init__(
        self,
        cfg: Config,
        retry_policy: Optional[RetryPolicy] = None,
        **kwargs
    ):
        # ... existing init ...
        self.retry_policy = retry_policy or RetryPolicy()
        self.retry_manager = RetryManager(self.retry_policy)

    async def weave(self, query: Query) -> Spacetime:
        """Weave with automatic retry on tool failures."""

        # Wrap tool execution in retry logic
        async def execute_with_retry():
            # ... existing weaving logic ...

            # Tool execution with retry
            if action_plan.tool == "search":
                result = await self.retry_manager.execute_with_retry(
                    self._execute_search_tool,
                    action_plan.params
                )
            # ... etc.

            return spacetime

        return await execute_with_retry()

    async def _execute_search_tool(self, params: Dict[str, Any]) -> Any:
        """Execute search tool (can raise exceptions)."""
        # ... implementation ...
        pass
```

---

#### 2.3: Add Error Summarization with LLM

**File**: `HoloLoom/orchestrator/error_summarization.py` (new)

```python
"""
LLM-based error summarization for retry context.

Converts verbose stack traces into concise, actionable summaries.
"""

from typing import Optional


class ErrorSummarizer:
    """Summarize errors using LLM or heuristics."""

    def __init__(self, use_llm: bool = False, llm_client: Optional[Any] = None):
        self.use_llm = use_llm
        self.llm_client = llm_client

    def summarize(self, error: Exception) -> str:
        """
        Summarize error for LLM context.

        Args:
            error: Exception to summarize

        Returns:
            Concise error summary (~50-100 chars)
        """
        if self.use_llm and self.llm_client:
            return self._summarize_with_llm(error)
        else:
            return self._summarize_heuristic(error)

    def _summarize_heuristic(self, error: Exception) -> str:
        """Heuristic-based summarization (no LLM)."""
        error_type = error.__class__.__name__
        error_msg = str(error)[:150]  # First 150 chars

        # Common patterns
        if "ConnectionError" in error_type:
            return f"API unreachable: {error_msg}"
        elif "TimeoutError" in error_type:
            return f"Request timed out: {error_msg}"
        elif "AuthenticationError" in error_type:
            return f"Auth failed: {error_msg}"
        elif "ValidationError" in error_type:
            return f"Invalid params: {error_msg}"
        else:
            return f"{error_type}: {error_msg}"

    async def _summarize_with_llm(self, error: Exception) -> str:
        """LLM-based summarization (more intelligent)."""
        import traceback

        full_trace = traceback.format_exc()

        prompt = f"""Summarize this error in one concise sentence (<100 chars):

{full_trace}

Focus on:
1. What failed (API call, validation, etc.)
2. Root cause (timeout, invalid params, auth, etc.)
3. Actionable info (what to fix)

Concise summary:"""

        response = await self.llm_client.generate(prompt, max_tokens=50)
        return response.strip()
```

---

### Deliverables

- [ ] `HoloLoom/orchestrator/retry.py` - RetryPolicy and RetryManager
- [ ] `HoloLoom/orchestrator/error_summarization.py` - Error summarizer
- [ ] `HoloLoom/weaving_orchestrator.py` - Integrate retry logic
- [ ] `demos/demo_retry_system.py` - Demo with failing tool
- [ ] Tests (80%+ coverage)
- [ ] Documentation update

---

## Phase 3: Centralized Prompt Management

**Duration**: 3-5 days
**Impact**: Medium - Easier maintenance and versioning
**Compliance Gain**: ✅ Excellent (85%) → ✅ Excellent (95%)

### Overview

Centralize all prompts in dedicated directory with versioning and hot-reloading.

**Benefits**:
- Single source of truth for all prompts
- Git-based versioning (track changes over time)
- Easier A/B testing (swap prompt files)
- Hot-reload during development
- Audit trail (who changed what, when)

### Design

#### 3.1: Create Prompt Directory Structure

**Structure**:
```bash
HoloLoom/prompts/
├── __init__.py
├── loader.py           # Prompt loading utility
├── version.py          # Version tracking
├── base/
│   ├── system_prompt.txt
│   ├── tool_selection.txt
│   └── refinement.txt
├── agentic/
│   ├── direct_mode.txt
│   ├── verify_mode.txt
│   ├── research_mode.txt
│   └── plan_execute_mode.txt
├── alignment/
│   ├── safety_guidelines.txt
│   └── deception_detection.txt
├── elle/
│   ├── base_prompt.txt
│   ├── context_prompt.txt
│   └── symbols.txt
└── versions/
    ├── v1.0/            # Old versions archived
    ├── v1.1/
    └── v1.2/
```

---

#### 3.2: Create Prompt Loader

**File**: `HoloLoom/prompts/loader.py`

```python
"""
Prompt loading utility with versioning and hot-reload.
"""

from pathlib import Path
from typing import Dict, Optional
import hashlib


class PromptLoader:
    """Load prompts from centralized directory."""

    def __init__(self, base_path: str = "HoloLoom/prompts"):
        self.base_path = Path(base_path)
        self._cache: Dict[str, str] = {}
        self._hashes: Dict[str, str] = {}

    def load(
        self,
        prompt_name: str,
        version: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Load prompt by name.

        Args:
            prompt_name: Prompt path (e.g., "agentic/direct_mode")
            version: Specific version (e.g., "v1.2") or None for latest
            use_cache: Use cached prompt if available

        Returns:
            Prompt text
        """
        # Determine file path
        if version:
            filepath = self.base_path / "versions" / version / f"{prompt_name}.txt"
        else:
            filepath = self.base_path / f"{prompt_name}.txt"

        # Check cache
        cache_key = f"{prompt_name}:{version or 'latest'}"
        if use_cache and cache_key in self._cache:
            # Check if file changed (hot-reload)
            current_hash = self._file_hash(filepath)
            if current_hash == self._hashes.get(cache_key):
                return self._cache[cache_key]

        # Load from file
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt not found: {filepath}")

        with open(filepath, 'r') as f:
            prompt_text = f.read()

        # Update cache
        self._cache[cache_key] = prompt_text
        self._hashes[cache_key] = self._file_hash(filepath)

        return prompt_text

    def _file_hash(self, filepath: Path) -> str:
        """Calculate file hash for change detection."""
        if not filepath.exists():
            return ""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def invalidate_cache(self) -> None:
        """Invalidate all cached prompts (force reload)."""
        self._cache.clear()
        self._hashes.clear()


# Global loader instance
_loader = PromptLoader()


def load_prompt(prompt_name: str, version: Optional[str] = None) -> str:
    """
    Load prompt (convenience function).

    Usage:
        from HoloLoom.prompts import load_prompt
        prompt = load_prompt("agentic/research_mode")
    """
    return _loader.load(prompt_name, version)
```

---

#### 3.3: Create Version Tracking

**File**: `HoloLoom/prompts/version.py`

```python
"""
Prompt version tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path


@dataclass
class PromptVersion:
    """Metadata for a prompt version."""
    prompt_name: str
    version: str
    git_sha: str
    author: str
    timestamp: datetime
    changelog: str


class PromptVersionManager:
    """Track prompt versions."""

    def __init__(self, versions_file: str = "HoloLoom/prompts/versions.json"):
        self.versions_file = Path(versions_file)
        self.versions: Dict[str, List[PromptVersion]] = {}
        self._load()

    def _load(self) -> None:
        """Load versions from JSON."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                # Parse versions
                for prompt_name, versions in data.items():
                    self.versions[prompt_name] = [
                        PromptVersion(**v) for v in versions
                    ]

    def add_version(
        self,
        prompt_name: str,
        version: str,
        git_sha: str,
        author: str,
        changelog: str
    ) -> None:
        """Add new version."""
        new_version = PromptVersion(
            prompt_name=prompt_name,
            version=version,
            git_sha=git_sha,
            author=author,
            timestamp=datetime.now(),
            changelog=changelog
        )

        if prompt_name not in self.versions:
            self.versions[prompt_name] = []

        self.versions[prompt_name].append(new_version)
        self._save()

    def _save(self) -> None:
        """Save versions to JSON."""
        data = {}
        for prompt_name, versions in self.versions.items():
            data[prompt_name] = [
                {
                    'prompt_name': v.prompt_name,
                    'version': v.version,
                    'git_sha': v.git_sha,
                    'author': v.author,
                    'timestamp': v.timestamp.isoformat(),
                    'changelog': v.changelog
                }
                for v in versions
            ]

        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_latest_version(self, prompt_name: str) -> Optional[PromptVersion]:
        """Get latest version of prompt."""
        if prompt_name in self.versions and self.versions[prompt_name]:
            return self.versions[prompt_name][-1]
        return None
```

---

#### 3.4: Migrate Existing Prompts

**Script**: `tools/migrate_prompts.py`

```python
"""
Migrate existing prompts to centralized directory.

Usage:
    python tools/migrate_prompts.py
"""

import re
from pathlib import Path


def extract_prompts_from_file(filepath: Path) -> Dict[str, str]:
    """Extract prompt strings from Python file."""
    prompts = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Find multi-line string literals (f""" or """)
    pattern = r'(?:f)?"""(.*?)"""'
    matches = re.findall(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        # Filter out docstrings (crude heuristic)
        if len(match) > 100 and ("Args:" in match or "Returns:" in match):
            continue

        # Likely a prompt
        prompts[f"extracted_{i}"] = match.strip()

    return prompts


def migrate_prompts():
    """Migrate all prompts to HoloLoom/prompts/."""
    # Create directory structure
    prompts_dir = Path("HoloLoom/prompts")
    prompts_dir.mkdir(exist_ok=True)

    # Subdirectories
    for subdir in ['base', 'agentic', 'alignment', 'elle']:
        (prompts_dir / subdir).mkdir(exist_ok=True)

    # Extract from known files
    files_to_scan = [
        "HoloLoom/agentic/core.py",
        "HoloLoom/alignment/safety_guardrails.py",
        "elle/prompt/builder.py",
    ]

    for filepath in files_to_scan:
        if not Path(filepath).exists():
            continue

        print(f"Scanning {filepath}...")
        prompts = extract_prompts_from_file(Path(filepath))

        for name, text in prompts.items():
            print(f"  Found prompt: {name}")
            # Manual review required - print for inspection
            print(f"  {text[:100]}...")


if __name__ == "__main__":
    migrate_prompts()
```

---

### Deliverables

- [ ] Create `HoloLoom/prompts/` directory structure
- [ ] `HoloLoom/prompts/loader.py` - Prompt loader with caching
- [ ] `HoloLoom/prompts/version.py` - Version tracking
- [ ] Migrate existing prompts from codebase
- [ ] Update all modules to use `load_prompt()`
- [ ] Documentation update
- [ ] Add to CI/CD (validate prompts exist)

---

## Timeline

### Week 1: Pause/Resume
- Days 1-2: WorkflowState + StateStore
- Days 3-4: WeavingOrchestrator integration
- Day 5: ChronoTrigger + Demo + Tests

### Week 2: Retry System
- Days 1-2: RetryPolicy + RetryManager
- Days 3-4: WeavingOrchestrator integration + ErrorSummarizer
- Day 5: Demo + Tests

### Week 3: Centralized Prompts
- Days 1-2: PromptLoader + VersionManager
- Day 3: Migrate existing prompts
- Days 4-5: Update all modules + Tests

### Week 4: Polish + Documentation
- Integration testing (all 3 features together)
- Update CLAUDE.md
- Update 12_FACTOR_COMPLIANCE.md
- Create tutorial docs

---

## Success Metrics

### Compliance Score
- **Before**: 87% (10/12 excellent, 1/12 good, 1/12 partial)
- **After**: 93% (12/12 excellent)

### Specific Improvements
- **Pause/Resume**: 🟢 Good (75%) → ✅ Excellent (90%)
- **Retry System**: 🟡 Fair (60%) → ✅ Excellent (90%)
- **Centralized Prompts**: ✅ Excellent (85%) → ✅ Excellent (95%)

### User-Facing Benefits
- ✅ Production workflows with human approvals
- ✅ 20-30% improvement in reliability (retry system)
- ✅ Easier prompt iteration and A/B testing
- ✅ Better observability (state serialization)

---

## Risk Mitigation

### Risk: State Serialization Performance
**Mitigation**: Use async file I/O, Neo4j for production

### Risk: Retry System Adds Latency
**Mitigation**: Only retry on actual failures, exponential backoff

### Risk: Prompt Migration Breaks Existing Code
**Mitigation**: Migrate incrementally, maintain backward compatibility

---

## Next Steps

1. **Review this roadmap** with team
2. **Assign owners** for each phase
3. **Set sprint dates** (4-6 weeks total)
4. **Create tracking issues** in GitHub
5. **Begin Phase 1** (Pause/Resume)

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-18
**Status**: Ready for Implementation
**Estimated Completion**: 4-6 weeks from start
