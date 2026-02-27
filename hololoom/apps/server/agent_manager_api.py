"""
Agent Manager REST API

FastAPI endpoints for multi-threaded agent execution, priority control,
MRF/MCTS injection, and temporal rollback.

Port: 8002 (avoids conflict with 8000 agentic_api, 8001 workflow)

Status: Phase 1 Implementation (December 2025)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import logging

from hololoom.apps.server.agent_manager_hub import (
    AgentManagerHub,
    ThreadConfig,
    SwarmConfig,
    ReasoningMode,
    ThreadStatus,
    MRFStrategy,
    create_agent_manager_hub
)

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HoloLoom Agent Manager API",
    description="Multi-threaded agent execution with priority control and temporal rollback",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hub instance (initialized on startup)
_hub: Optional[AgentManagerHub] = None


# =============================================================================
# Pydantic Models - Request/Response
# =============================================================================

class CreateThreadRequest(BaseModel):
    """Request to create a new agent thread."""
    name: str = Field(..., description="Human-readable thread name")
    query: str = Field(..., description="Query/task for the agent")
    agent_type: str = Field(default="weaving", description="Agent type: weaving, rag, agentic, custom")
    reasoning_mode: str = Field(default="direct", description="Reasoning mode: direct, verify, research, plan_execute")
    priority: int = Field(default=0, description="Priority level (higher = sooner execution)")
    time_budget_ms: Optional[float] = Field(default=None, description="Time budget in milliseconds")
    token_budget: Optional[int] = Field(default=None, description="Token budget")
    depends_on: List[str] = Field(default_factory=list, description="Thread IDs this thread depends on")
    llm_provider: str = Field(default="ollama", description="LLM provider: ollama, anthropic, openai")
    llm_model: str = Field(default="llama3.2:3b", description="Model name")


class ThreadResponse(BaseModel):
    """Response with thread details."""
    id: str
    name: str
    status: str
    priority: int
    agent_type: str
    reasoning_mode: str
    current_step: int
    total_steps: int
    elapsed_time_ms: float
    time_budget_ms: Optional[float]
    tokens_used: int
    token_budget: Optional[int]
    confidence: float
    estimated_cost_usd: float
    is_local: bool
    depends_on: List[str]
    blocks: List[str]
    swarm_id: Optional[str]
    head_commit_id: Optional[str]
    created_at: str
    updated_at: str
    error: Optional[str]


class ThreadListResponse(BaseModel):
    """Response with list of threads."""
    threads: List[ThreadResponse]
    total: int
    active: int
    completed: int
    failed: int


class StepResponse(BaseModel):
    """Response with step details."""
    id: str
    thread_id: str
    step_type: str
    name: str
    status: str
    progress_pct: float
    elapsed_time_ms: float
    tokens_used: int
    confidence: float
    query: Optional[str]
    response: Optional[str]
    tool_used: Optional[str]
    mrf_eligible: bool
    mcts_eligible: bool
    injection_applied: Optional[str]
    parent_id: Optional[str]
    children_ids: List[str]


class SetPriorityRequest(BaseModel):
    """Request to set thread priority."""
    priority: int = Field(..., description="New priority level")


class InjectMRFRequest(BaseModel):
    """Request to inject MRF strategy."""
    strategy: str = Field(..., description="MRF strategy: auto, verify, elegance, critique, refine, hofstadter")


class InjectMCTSRequest(BaseModel):
    """Request to inject MCTS exploration."""
    budget: int = Field(default=100, description="MCTS iteration budget")
    exploration_c: float = Field(default=1.414, description="UCB1 exploration coefficient")


class ForkThreadRequest(BaseModel):
    """Request to fork a thread from a step."""
    from_step_id: str = Field(..., description="Step ID to fork from")
    new_reasoning_mode: Optional[str] = Field(default=None, description="New reasoning mode for forked thread")
    new_mrf_strategy: Optional[str] = Field(default=None, description="MRF strategy for forked thread")
    new_mcts_budget: Optional[int] = Field(default=None, description="MCTS budget for forked thread")


class CreateSwarmRequest(BaseModel):
    """Request to create a swarm of threads."""
    name: str = Field(..., description="Swarm name")
    query: str = Field(..., description="Query/task for the swarm")
    child_count: int = Field(default=3, description="Number of child threads")
    reasoning_modes: List[str] = Field(default_factory=list, description="Reasoning modes for children (cycles if fewer than child_count)")
    merge_strategy: str = Field(default="best", description="Merge strategy: best, combine, manual")


class SwarmResponse(BaseModel):
    """Response with swarm details."""
    id: str
    name: str
    parent_thread_id: str
    child_thread_ids: List[str]
    merge_strategy: str
    status: str
    created_at: str


class MergeSwarmRequest(BaseModel):
    """Request to merge swarm results."""
    strategy: Optional[str] = Field(default=None, description="Override merge strategy")


class CheckoutRequest(BaseModel):
    """Request to checkout a commit (rollback)."""
    target: str = Field(..., description="Commit ID or branch name")
    create_branch: Optional[str] = Field(default=None, description="Create new branch at target")


class CreateBranchRequest(BaseModel):
    """Request to create a branch."""
    name: str = Field(..., description="Branch name")
    from_commit: Optional[str] = Field(default=None, description="Source commit (default: HEAD)")


class CommitLogResponse(BaseModel):
    """Response with commit log."""
    commits: List[Dict[str, Any]]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    active_threads: int
    active_swarms: int
    uptime_seconds: float


# =============================================================================
# Helper Functions
# =============================================================================

def _thread_to_response(thread) -> ThreadResponse:
    """Convert AgentThread to ThreadResponse."""
    return ThreadResponse(
        id=thread.id,
        name=thread.name,
        status=thread.status.value,
        priority=thread.priority,
        agent_type=thread.agent_type,
        reasoning_mode=thread.reasoning_mode.value,
        current_step=thread.current_step,
        total_steps=thread.total_steps,
        elapsed_time_ms=thread.elapsed_time_ms,
        time_budget_ms=thread.time_budget_ms,
        tokens_used=thread.tokens_used,
        token_budget=thread.token_budget,
        confidence=thread.confidence,
        estimated_cost_usd=thread.cost.estimated_cost_usd,
        is_local=thread.cost.is_local,
        depends_on=thread.depends_on,
        blocks=thread.blocks,
        swarm_id=thread.swarm_id,
        head_commit_id=thread.head_commit_id,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        error=thread.error
    )


def _step_to_response(step) -> StepResponse:
    """Convert TaskNode to StepResponse."""
    return StepResponse(
        id=step.id,
        thread_id=step.thread_id,
        step_type=step.step_type.value,
        name=step.name,
        status=step.status.value,
        progress_pct=step.progress_pct,
        elapsed_time_ms=step.elapsed_time_ms,
        tokens_used=step.tokens_used,
        confidence=step.confidence,
        query=step.query,
        response=step.response,
        tool_used=step.tool_used,
        mrf_eligible=step.mrf_eligible,
        mcts_eligible=step.mcts_eligible,
        injection_applied=step.injection_applied,
        parent_id=step.parent_id,
        children_ids=step.children_ids
    )


def get_hub() -> AgentManagerHub:
    """Get the global hub instance."""
    global _hub
    if _hub is None:
        raise HTTPException(status_code=503, detail="Agent manager not initialized")
    return _hub


# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the agent manager hub on startup."""
    global _hub
    logger.info("Starting Agent Manager API...")
    _hub = create_agent_manager_hub()
    logger.info("Agent Manager API started on port 8002")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global _hub
    if _hub:
        await _hub.shutdown()
        _hub = None
    logger.info("Agent Manager API shut down")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    hub = get_hub()
    stats = hub.get_statistics()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        active_threads=stats.get("active_threads", 0),
        active_swarms=stats.get("active_swarms", 0),
        uptime_seconds=stats.get("uptime_seconds", 0.0)
    )


# =============================================================================
# Thread Endpoints
# =============================================================================

@app.post("/api/threads", response_model=ThreadResponse, status_code=201)
async def create_thread(request: CreateThreadRequest):
    """Create a new agent thread."""
    hub = get_hub()

    try:
        reasoning_mode = ReasoningMode(request.reasoning_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reasoning_mode: {request.reasoning_mode}. Must be one of: direct, verify, research, plan_execute"
        )

    config = ThreadConfig(
        name=request.name,
        query=request.query,
        agent_type=request.agent_type,
        reasoning_mode=reasoning_mode,
        priority=request.priority,
        time_budget_ms=request.time_budget_ms,
        token_budget=request.token_budget,
        depends_on=request.depends_on,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model
    )

    thread_id = await hub.create_thread(config)
    thread = hub.get_thread(thread_id)

    if thread is None:
        raise HTTPException(status_code=500, detail="Failed to create thread")

    return _thread_to_response(thread)


@app.get("/api/threads", response_model=ThreadListResponse)
async def list_threads(
    status: Optional[str] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    """List all threads with optional filtering."""
    hub = get_hub()

    threads = hub.list_threads()

    # Filter by status if provided
    if status:
        try:
            status_enum = ThreadStatus(status)
            threads = [t for t in threads if t.status == status_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    # Pagination
    total = len(threads)
    threads = threads[offset:offset + limit]

    # Count stats
    all_threads = hub.list_threads()
    active = sum(1 for t in all_threads if t.status in [ThreadStatus.RUNNING, ThreadStatus.PAUSED, ThreadStatus.QUEUED])
    completed = sum(1 for t in all_threads if t.status == ThreadStatus.COMPLETED)
    failed = sum(1 for t in all_threads if t.status == ThreadStatus.FAILED)

    return ThreadListResponse(
        threads=[_thread_to_response(t) for t in threads],
        total=total,
        active=active,
        completed=completed,
        failed=failed
    )


@app.get("/api/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str):
    """Get thread details by ID."""
    hub = get_hub()
    thread = hub.get_thread(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return _thread_to_response(thread)


@app.delete("/api/threads/{thread_id}")
async def cancel_thread(thread_id: str):
    """Cancel a thread."""
    hub = get_hub()

    try:
        await hub.cancel_thread(thread_id)
        return {"status": "cancelled", "thread_id": thread_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/threads/{thread_id}/start")
async def start_thread(thread_id: str):
    """Start a queued thread."""
    hub = get_hub()

    try:
        await hub.start_thread(thread_id)
        return {"status": "started", "thread_id": thread_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/pause")
async def pause_thread(thread_id: str):
    """Pause a running thread."""
    hub = get_hub()

    try:
        await hub.pause_thread(thread_id)
        return {"status": "paused", "thread_id": thread_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/resume")
async def resume_thread(thread_id: str):
    """Resume a paused thread."""
    hub = get_hub()

    try:
        await hub.resume_thread(thread_id)
        return {"status": "resumed", "thread_id": thread_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/priority")
async def set_thread_priority(thread_id: str, request: SetPriorityRequest):
    """Set thread priority."""
    hub = get_hub()

    try:
        await hub.set_priority(thread_id, request.priority)
        return {"status": "priority_set", "thread_id": thread_id, "priority": request.priority}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/threads/{thread_id}/upvote")
async def upvote_thread(thread_id: str):
    """Upvote thread (+1 priority)."""
    hub = get_hub()

    try:
        await hub.upvote(thread_id)
        thread = hub.get_thread(thread_id)
        return {"status": "upvoted", "thread_id": thread_id, "priority": thread.priority if thread else 0}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/threads/{thread_id}/downvote")
async def downvote_thread(thread_id: str):
    """Downvote thread (-1 priority)."""
    hub = get_hub()

    try:
        await hub.downvote(thread_id)
        thread = hub.get_thread(thread_id)
        return {"status": "downvoted", "thread_id": thread_id, "priority": thread.priority if thread else 0}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/threads/{thread_id}/fork", response_model=ThreadResponse)
async def fork_thread(thread_id: str, request: ForkThreadRequest):
    """Fork a thread from a specific step."""
    hub = get_hub()

    new_params = {}
    if request.new_reasoning_mode:
        try:
            new_params["reasoning_mode"] = ReasoningMode(request.new_reasoning_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid reasoning_mode: {request.new_reasoning_mode}")
    if request.new_mrf_strategy:
        new_params["mrf_strategy"] = request.new_mrf_strategy
    if request.new_mcts_budget:
        new_params["mcts_budget"] = request.new_mcts_budget

    try:
        new_thread_id = await hub.fork_thread(thread_id, request.from_step_id, new_params)
        new_thread = hub.get_thread(new_thread_id)

        if new_thread is None:
            raise HTTPException(status_code=500, detail="Failed to fork thread")

        return _thread_to_response(new_thread)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Step Endpoints
# =============================================================================

@app.get("/api/threads/{thread_id}/steps", response_model=List[StepResponse])
async def list_thread_steps(thread_id: str):
    """List all steps for a thread."""
    hub = get_hub()
    thread = hub.get_thread(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    return [_step_to_response(step) for step in thread.steps.values()]


@app.get("/api/threads/{thread_id}/steps/{step_id}", response_model=StepResponse)
async def get_step(thread_id: str, step_id: str):
    """Get step details."""
    hub = get_hub()
    thread = hub.get_thread(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    step = thread.steps.get(step_id)
    if step is None:
        raise HTTPException(status_code=404, detail=f"Step not found: {step_id}")

    return _step_to_response(step)


@app.post("/api/threads/{thread_id}/steps/{step_id}/inject/mrf")
async def inject_mrf(thread_id: str, step_id: str, request: InjectMRFRequest):
    """Inject MRF strategy into a step."""
    hub = get_hub()

    try:
        strategy = MRFStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MRF strategy: {request.strategy}. Must be one of: auto, verify, elegance, critique, refine, hofstadter"
        )

    try:
        await hub.inject_mrf(thread_id, step_id, strategy)
        return {"status": "mrf_injected", "thread_id": thread_id, "step_id": step_id, "strategy": request.strategy}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/steps/{step_id}/inject/mcts")
async def inject_mcts(thread_id: str, step_id: str, request: InjectMCTSRequest):
    """Inject MCTS exploration into a step."""
    hub = get_hub()

    try:
        await hub.inject_mcts(thread_id, step_id, request.budget, request.exploration_c)
        return {
            "status": "mcts_injected",
            "thread_id": thread_id,
            "step_id": step_id,
            "budget": request.budget,
            "exploration_c": request.exploration_c
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Swarm Endpoints
# =============================================================================

@app.post("/api/swarms", response_model=SwarmResponse, status_code=201)
async def create_swarm(request: CreateSwarmRequest):
    """Create a new swarm of agent threads."""
    hub = get_hub()

    # Parse reasoning modes
    reasoning_modes = []
    for mode_str in request.reasoning_modes:
        try:
            reasoning_modes.append(ReasoningMode(mode_str))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid reasoning_mode: {mode_str}")

    config = SwarmConfig(
        name=request.name,
        child_count=request.child_count,
        reasoning_modes=reasoning_modes,
        merge_strategy=request.merge_strategy
    )

    try:
        swarm_id = await hub.create_swarm(config, request.query)
        swarm = hub.get_swarm(swarm_id)

        if swarm is None:
            raise HTTPException(status_code=500, detail="Failed to create swarm")

        return SwarmResponse(
            id=swarm.id,
            name=swarm.name,
            parent_thread_id=swarm.parent_thread_id,
            child_thread_ids=swarm.child_thread_ids,
            merge_strategy=swarm.merge_strategy,
            status=swarm.status.value,
            created_at=swarm.created_at
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/swarms/{swarm_id}", response_model=SwarmResponse)
async def get_swarm(swarm_id: str):
    """Get swarm details."""
    hub = get_hub()
    swarm = hub.get_swarm(swarm_id)

    if swarm is None:
        raise HTTPException(status_code=404, detail=f"Swarm not found: {swarm_id}")

    return SwarmResponse(
        id=swarm.id,
        name=swarm.name,
        parent_thread_id=swarm.parent_thread_id,
        child_thread_ids=swarm.child_thread_ids,
        merge_strategy=swarm.merge_strategy,
        status=swarm.status.value,
        created_at=swarm.created_at
    )


@app.post("/api/swarms/{swarm_id}/merge")
async def merge_swarm(swarm_id: str, request: Optional[MergeSwarmRequest] = None):
    """Merge swarm results."""
    hub = get_hub()

    strategy = request.strategy if request else None

    try:
        result = await hub.merge_swarm(swarm_id, strategy)
        return {
            "status": "merged",
            "swarm_id": swarm_id,
            "merged_thread_id": result.get("merged_thread_id"),
            "strategy_used": result.get("strategy_used")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Git/Temporal Endpoints
# =============================================================================

@app.get("/api/threads/{thread_id}/log", response_model=CommitLogResponse)
async def get_commit_log(
    thread_id: str,
    max_count: int = Query(default=50, ge=1, le=200)
):
    """Get commit history for a thread (git log)."""
    hub = get_hub()
    thread = hub.get_thread(thread_id)

    if thread is None:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    try:
        commits = await hub.get_commit_log(thread_id, max_count)
        return CommitLogResponse(commits=commits, total=len(commits))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/threads/{thread_id}/commits/{commit_id}")
async def get_commit(thread_id: str, commit_id: str):
    """Get commit details."""
    hub = get_hub()

    try:
        commit = await hub.get_commit(thread_id, commit_id)
        if commit is None:
            raise HTTPException(status_code=404, detail=f"Commit not found: {commit_id}")
        return commit
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/threads/{thread_id}/diff")
async def get_diff(
    thread_id: str,
    from_commit: str = Query(..., description="Source commit ID"),
    to_commit: str = Query(..., description="Target commit ID")
):
    """Get diff between two commits."""
    hub = get_hub()

    try:
        diff = await hub.get_diff(thread_id, from_commit, to_commit)
        return {"from": from_commit, "to": to_commit, "diff": diff}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/threads/{thread_id}/branches")
async def list_branches(thread_id: str):
    """List all branches for a thread."""
    hub = get_hub()

    try:
        branches = await hub.list_branches(thread_id)
        return {"thread_id": thread_id, "branches": branches}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/threads/{thread_id}/branch")
async def create_branch(thread_id: str, request: CreateBranchRequest):
    """Create a new branch."""
    hub = get_hub()

    try:
        await hub.create_branch(thread_id, request.name, request.from_commit)
        return {"status": "branch_created", "thread_id": thread_id, "branch": request.name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/threads/{thread_id}/branches/{branch_name}")
async def delete_branch(thread_id: str, branch_name: str):
    """Delete a branch."""
    hub = get_hub()

    try:
        await hub.delete_branch(thread_id, branch_name)
        return {"status": "branch_deleted", "thread_id": thread_id, "branch": branch_name}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/checkout")
async def checkout(thread_id: str, request: CheckoutRequest):
    """Checkout a commit or branch (move HEAD)."""
    hub = get_hub()

    try:
        new_head = await hub.checkout_commit(thread_id, request.target)
        if request.create_branch:
            await hub.create_branch(thread_id, request.create_branch, new_head)
        return {
            "status": "checked_out",
            "thread_id": thread_id,
            "target": request.target,
            "new_head": new_head,
            "new_branch": request.create_branch
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/reset")
async def reset_thread(
    thread_id: str,
    target_commit_id: str = Query(..., description="Commit ID to reset to"),
    mode: str = Query(default="soft", description="Reset mode: soft, mixed, hard")
):
    """Reset HEAD to a commit."""
    hub = get_hub()

    if mode not in ["soft", "mixed", "hard"]:
        raise HTTPException(status_code=400, detail=f"Invalid reset mode: {mode}")

    try:
        await hub.reset_thread(thread_id, target_commit_id, mode)
        return {"status": "reset", "thread_id": thread_id, "target": target_commit_id, "mode": mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/cherry-pick")
async def cherry_pick(
    thread_id: str,
    commit_id: str = Query(..., description="Commit ID to cherry-pick")
):
    """Cherry-pick a commit to current branch."""
    hub = get_hub()

    try:
        new_commit = await hub.cherry_pick(thread_id, commit_id)
        return {"status": "cherry_picked", "thread_id": thread_id, "source_commit": commit_id, "new_commit": new_commit}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/threads/{thread_id}/merge")
async def merge_branch(
    thread_id: str,
    source_branch: str = Query(..., description="Branch to merge from"),
    strategy: str = Query(default="best", description="Merge strategy: best, combine, manual")
):
    """Merge a branch into current branch."""
    hub = get_hub()

    try:
        result = await hub.merge_branch(thread_id, source_branch, strategy)
        return {"status": "merged", "thread_id": thread_id, "source_branch": source_branch, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/threads/{thread_id}/merge-base")
async def get_merge_base(
    thread_id: str,
    branch1: str = Query(..., description="First branch"),
    branch2: str = Query(..., description="Second branch")
):
    """Find common ancestor of two branches."""
    hub = get_hub()

    try:
        merge_base = await hub.get_merge_base(thread_id, branch1, branch2)
        return {"thread_id": thread_id, "branch1": branch1, "branch2": branch2, "merge_base": merge_base}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/threads/{thread_id}/reflog")
async def get_reflog(
    thread_id: str,
    limit: int = Query(default=50, ge=1, le=200)
):
    """Get reflog (operation history) for a thread."""
    hub = get_hub()

    try:
        reflog = await hub.get_reflog(thread_id, limit)
        return {"thread_id": thread_id, "entries": reflog}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/agent-manager")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    hub = get_hub()

    # Register connection
    connection_id = id(websocket)
    logger.info(f"WebSocket connected: {connection_id}")

    # Subscription patterns (client can subscribe to specific threads/events)
    subscriptions = set()

    async def send_event(event: Dict[str, Any]):
        """Send event if client is subscribed."""
        event_type = event.get("type", "")
        thread_id = event.get("thread_id", "")

        # Check if client subscribed to this event
        should_send = (
            "*" in subscriptions or
            event_type in subscriptions or
            f"thread:{thread_id}" in subscriptions or
            f"event:{event_type}" in subscriptions
        )

        if should_send:
            try:
                await websocket.send_json(event)
            except Exception as e:
                logger.error(f"Failed to send WebSocket event: {e}")

    # Register event handler with hub
    hub.register_event_handler(send_event)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            msg_type = data.get("type", "")

            if msg_type == "subscribe":
                # Subscribe to patterns
                patterns = data.get("patterns", [])
                if isinstance(patterns, str):
                    patterns = [patterns]
                subscriptions.update(patterns)
                await websocket.send_json({
                    "type": "subscribed",
                    "patterns": list(subscriptions)
                })

            elif msg_type == "unsubscribe":
                # Unsubscribe from patterns
                patterns = data.get("patterns", [])
                if isinstance(patterns, str):
                    patterns = [patterns]
                subscriptions.difference_update(patterns)
                await websocket.send_json({
                    "type": "unsubscribed",
                    "patterns": list(subscriptions)
                })

            elif msg_type == "ping":
                # Heartbeat
                await websocket.send_json({"type": "pong"})

            else:
                logger.warning(f"Unknown WebSocket message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Unregister event handler
        hub.unregister_event_handler(send_event)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
