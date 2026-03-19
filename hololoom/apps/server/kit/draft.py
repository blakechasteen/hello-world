"""
Draft + Pipeline + Agent.

Three primitives:
    Draft    — the evolving work product
    Pass     — async (Draft) -> Draft
    Pipeline — [Pass] with optional loops

One type:
    Agent    — soul + memory + backend + pipeline → callable

Naming follows the weaving metaphor:
    Draft = cloth before finishing
    Pass  = one pass of the shuttle across the warp
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace  # noqa: F401 — replace re-exported
from typing import Any, Union

from .emit import emit
from .llm import LLMBackend, extract_content, extract_tokens
from .memory import Memory
from .models import ChatRequest, ChatResponse

# Soul: static string or callable returning fresh string per request.
Soul = Union[str, Callable[[], str]]

logger = logging.getLogger(__name__)


# ============================================================================
# Draft
# ============================================================================

@dataclass
class Draft:
    """The work product flowing through a pipeline.

    Passes read and mutate the draft. Metadata is the escape hatch —
    passes stash arbitrary data for downstream passes to read.

    Convention flags (set by passes, read by Agent):
        skip_memory  — if truthy, Agent won't store this turn
        skip_emit    — if truthy, Agent won't emit to the bus

    Trace (populated by compose automatically):
        trace        — list of {"pass": name, "ms": duration} per pass
    """
    query: str
    room_id: str
    messages: list[dict[str, str]]
    text: str = ""
    model: str = ""
    tokens: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pass type + naming
# ============================================================================

# A Pass is an async function: Draft -> Draft
Pass = Callable[[Draft], Awaitable[Draft]]


def _pass_name(p: Pass) -> str:
    """Best-effort name for a pass. Reads __pass_name__, __name__, or class."""
    return getattr(p, '__pass_name__', None) or getattr(p, '__name__', type(p).__name__)


def named(name: str, p: Pass) -> Pass:
    """Attach a name to any pass for tracing."""
    p.__pass_name__ = name
    return p


# ============================================================================
# Core passes
# ============================================================================

def think(backend: LLMBackend) -> Pass:
    """Core pass factory: call the LLM and populate draft.text.

    Returns a named, bound Pass. Each think() can use a different backend:
        pipeline = [think(fast), critique, think(quality)]
    """
    async def _think(draft: Draft) -> Draft:
        start = time.time()
        data = await backend.call(draft.messages)
        draft.text = extract_content(data)
        draft.tokens += extract_tokens(data)
        draft.model = data.get("model", "unknown")
        draft.duration_ms += (time.time() - start) * 1000
        return draft
    _think.__pass_name__ = f"think({getattr(backend, 'model', '?')})"
    return _think


# ============================================================================
# Gates (convergence predicates)
# ============================================================================

def converged(before: str, after: str, threshold: float = 0.8) -> bool:
    """Default convergence check: Jaccard similarity > threshold.

    Empty before means "first pass hasn't run yet" — never converged.
    Empty after or literal "PASS" means "pass declined to produce" — converged.
    """
    if not after or after.strip() == "PASS":
        return True
    if not before:
        return False  # First iteration — can't converge without a baseline
    words_a = set(before.lower().split())
    words_b = set(after.lower().split())
    if not words_b:
        return True
    jaccard = len(words_a & words_b) / len(words_a | words_b)
    return jaccard > threshold


# ============================================================================
# Combinators
# ============================================================================

def loop(
    *passes: Pass,
    until: Callable[[str, str], bool] = converged,
    max: int = 3,
) -> Pass:
    """
    Compose passes into a loop that runs until convergence or max iterations.
    Returns a single Pass (async callable).
    """
    async def _loop(draft: Draft) -> Draft:
        for i in range(max):
            before_text = draft.text
            for p in passes:
                draft = await p(draft)
            if until(before_text, draft.text):
                logger.debug("Loop converged after %d iterations", i + 1)
                break
        return draft
    _loop.__pass_name__ = f"loop({', '.join(_pass_name(p) for p in passes)})"
    return _loop


def compose(*passes: Pass) -> Pass:
    """Chain passes sequentially. Returns a single traced Pass.

    Every pass is timed. Results stored in draft.metadata["trace"]:
        [{"pass": "think(qwen3.5:9b)", "ms": 3200.1}, ...]

    On failure, the trace includes the error and the draft is
    attached to the exception as e.draft for post-mortem.
    """
    async def _chain(draft: Draft) -> Draft:
        trace = draft.metadata.setdefault("trace", [])
        for p in passes:
            pname = _pass_name(p)
            start = time.time()
            try:
                draft = await p(draft)
            except Exception as e:
                ms = round((time.time() - start) * 1000, 1)
                trace.append({"pass": pname, "ms": ms, "error": str(e)})
                e.draft = draft
                e.failed_pass = pname
                raise
            trace.append({"pass": pname, "ms": round((time.time() - start) * 1000, 1)})
        return draft
    return _chain


# ============================================================================
# Agent
# ============================================================================

@dataclass
class Agent:
    """Soul + memory + backend + pipeline → callable.

    Callable:   await agent(ChatRequest) -> ChatResponse
    Clonable:   replace(agent, memory=NullMemory(), name="fresh")
    Printable:  Agent(elle, 10t, qwen3.5:9b, 1p)

    The pipeline controls the Draft transformation. Agent provides the frame:
    1. Resolve soul, load history, build messages → Draft
    2. Run pipeline (each Pass transforms the Draft — traced automatically)
    3. Store turn in memory (unless draft.metadata["skip_memory"])
    4. Emit to bus (unless draft.metadata["skip_emit"])
    5. Return ChatResponse
    """
    soul: Soul
    memory: Memory
    backend: LLMBackend
    pipeline: list[Pass] | None = field(default=None, repr=False)
    name: str = "agent"

    _chain: Pass = field(init=False, repr=False)

    def __post_init__(self):
        if self.pipeline is None:
            self.pipeline = [think(self.backend)]
        self._chain = compose(*self.pipeline)

    async def __call__(self, request: ChatRequest) -> ChatResponse:
        start = time.time()

        soul_text = self.soul() if callable(self.soul) else self.soul
        history = self.memory.get_messages(request.room_id)
        messages = [{"role": "system", "content": soul_text}]
        messages.extend(history)
        messages.append({"role": "user", "content": request.text})

        draft = Draft(
            query=request.text,
            room_id=request.room_id,
            messages=messages,
        )

        draft = await self._chain(draft)

        duration_ms = (time.time() - start) * 1000

        if draft.text and not draft.metadata.get("skip_memory"):
            self.memory.add_turn(request.room_id, request.text, draft.text)

        if not draft.metadata.get("skip_emit"):
            emit(self.name, request.text, draft.text, request.room_id,
                 draft.tokens, duration_ms)

        return ChatResponse(
            response=draft.text,
            room_id=request.room_id,
            model=draft.model,
            tokens=draft.tokens,
            duration_ms=round(duration_ms, 1),
            turns_in_memory=self.memory.turn_count(request.room_id),
            metadata=draft.metadata,
        )

    def __repr__(self):
        model = getattr(self.backend, "model", "?")
        passes = len(self.pipeline) if self.pipeline else 0
        return f"Agent({self.name}, {self.memory.max_turns}t, {model}, {passes}p)"
