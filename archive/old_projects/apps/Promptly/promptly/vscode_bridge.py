"""
FastAPI bridge for VS Code extension.

Provides REST API endpoints for prompt management operations with caching.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import promptly modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

# Import Promptly core
try:
    # Try relative import first (when run as module)
    try:
        from .promptly import Promptly
        from .execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend, OllamaExecutor
        from .recursive_loops import RecursiveEngine, LoopType, LoopConfig, Scratchpad
    except ImportError:
        # Fall back to direct import (when run as script)
        from promptly import Promptly
        from execution_engine import ExecutionEngine, ExecutionConfig, ExecutionBackend, OllamaExecutor
        from recursive_loops import RecursiveEngine, LoopType, LoopConfig, Scratchpad
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Promptly core not available: {e}", file=sys.stderr)
    PROMPTLY_AVAILABLE = False


# Simple cache with TTL
class SimpleCache:
    """Simple in-memory cache with time-to-live."""

    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

    def invalidate(self, key: str = None):
        if key:
            self.cache.pop(key, None)
        else:
            self.cache.clear()


# Models
class PromptMetadata(BaseModel):
    name: str
    branch: str
    tags: List[str]
    created: str


class PromptData(BaseModel):
    content: str
    metadata: PromptMetadata


class PromptsListResponse(BaseModel):
    prompts: List[PromptMetadata]


# Execution Models
class SkillExecutionRequest(BaseModel):
    skill_name: str
    user_input: str
    backend: str = "ollama"
    model: str = "llama3.2:3b"


class ChainExecutionRequest(BaseModel):
    skill_names: List[str]
    initial_input: str
    backend: str = "ollama"
    model: str = "llama3.2:3b"


class LoopExecutionRequest(BaseModel):
    skill_name: str
    user_input: str
    loop_type: str = "refine"  # refine, critique, decompose, verify, explore, hofstadter
    max_iterations: int = 5
    quality_threshold: float = 0.9
    backend: str = "ollama"
    model: str = "llama3.2:3b"


class ExecutionResponse(BaseModel):
    execution_id: str
    status: str  # queued, running, completed, failed
    message: str


class ExecutionStatus(BaseModel):
    execution_id: str
    status: str
    progress: float  # 0.0 to 1.0
    current_step: str
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# FastAPI app
app = FastAPI(
    title="Promptly VS Code Bridge",
    description="REST API bridge for Promptly VS Code extension",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize Promptly and cache
if PROMPTLY_AVAILABLE:
    try:
        p = Promptly()
        logger.info("Promptly initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Promptly: {e}")
        p = None
else:
    logger.warning("Promptly core not available")
    p = None

# Initialize cache (30 second TTL for prompt listings, 60s for individual prompts)
list_cache = SimpleCache(ttl_seconds=30)
prompt_cache = SimpleCache(ttl_seconds=60)
logger.info("Cache initialized (list: 30s TTL, prompts: 60s TTL)")

# Execution tracking
import uuid
import asyncio
from typing import Set

executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> status dict
active_websockets: Set[WebSocket] = set()


async def broadcast_execution_event(execution_id: str, event: Dict[str, Any]):
    """Broadcast execution event to all connected WebSocket clients."""
    dead_sockets = set()
    for ws in active_websockets:
        try:
            await ws.send_json({"execution_id": execution_id, **event})
        except Exception:
            dead_sockets.add(ws)

    # Clean up dead connections
    active_websockets.difference_update(dead_sockets)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "promptly_available": PROMPTLY_AVAILABLE and p is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/prompts", response_model=PromptsListResponse)
async def list_prompts():
    """List all prompts with caching."""
    if not p:
        logger.error("Promptly not available for list_prompts")
        raise HTTPException(status_code=503, detail="Promptly not available")

    # Check cache first
    cached = list_cache.get("prompts_list")
    if cached:
        logger.debug(f"Cache hit for prompts_list ({len(cached)} prompts)")
        return PromptsListResponse(prompts=cached)

    try:
        # Get all prompts
        prompts = p.list_prompts()

        # Convert to metadata format
        metadata_list = []
        for prompt in prompts:
            try:
                # Get full prompt data to access metadata
                prompt_data = p.get(prompt['name'])
                if prompt_data:
                    metadata = prompt_data.get('metadata', {})
                    metadata_list.append(PromptMetadata(
                        name=prompt['name'],
                        branch=prompt_data.get('branch', 'main'),
                        tags=metadata.get('tags', []),
                        created=prompt_data.get('created_at', datetime.now().isoformat())
                    ))
            except Exception as e:
                print(f"WARNING: Failed to get metadata for {prompt['name']}: {e}", file=sys.stderr)
                continue

        # Cache the result
        list_cache.set("prompts_list", metadata_list)
        logger.info(f"Listed {len(metadata_list)} prompts (cached)")

        return PromptsListResponse(prompts=metadata_list)

    except Exception as e:
        logger.error(f"Failed to list prompts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {e}")


@app.get("/prompts/{prompt_name}", response_model=PromptData)
async def get_prompt(prompt_name: str):
    """Get a specific prompt by name with caching."""
    if not p:
        logger.error(f"Promptly not available for get_prompt: {prompt_name}")
        raise HTTPException(status_code=503, detail="Promptly not available")

    # Check cache first
    cache_key = f"prompt:{prompt_name}"
    cached = prompt_cache.get(cache_key)
    if cached:
        logger.debug(f"Cache hit for prompt: {prompt_name}")
        return cached

    try:
        prompt_data = p.get(prompt_name)

        if not prompt_data:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")

        metadata = prompt_data.get('metadata', {})
        result = PromptData(
            content=prompt_data.get('content', ''),
            metadata=PromptMetadata(
                name=prompt_name,
                branch=prompt_data.get('branch', 'main'),
                tags=metadata.get('tags', []),
                created=prompt_data.get('created_at', datetime.now().isoformat())
            )
        )

        # Cache the result
        prompt_cache.set(cache_key, result)
        logger.info(f"Retrieved prompt: {prompt_name} (cached)")

        return result

    except KeyError:
        logger.warning(f"Prompt not found: {prompt_name}")
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
    except Exception as e:
        logger.error(f"Failed to get prompt {prompt_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get prompt: {e}")


@app.post("/execute/skill", response_model=ExecutionResponse)
async def execute_skill(request: SkillExecutionRequest):
    """Execute a single skill."""
    if not p:
        raise HTTPException(status_code=503, detail="Promptly not available")

    execution_id = str(uuid.uuid4())
    executions[execution_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_step": "Initializing...",
        "type": "skill",
        "request": request.dict()
    }

    # Start execution in background
    asyncio.create_task(_execute_skill_background(execution_id, request))

    return ExecutionResponse(
        execution_id=execution_id,
        status="queued",
        message=f"Skill execution {execution_id} queued"
    )


async def _execute_skill_background(execution_id: str, request: SkillExecutionRequest):
    """Background task for skill execution."""
    try:
        executions[execution_id]["status"] = "running"
        executions[execution_id]["progress"] = 0.1
        executions[execution_id]["current_step"] = f"Executing skill: {request.skill_name}"
        await broadcast_execution_event(execution_id, {
            "event": "status_update",
            "status": "running",
            "progress": 0.1,
            "step": f"Executing skill: {request.skill_name}"
        })

        # Prepare skill payload
        payload = p.prepare_skill_payload(request.skill_name)
        full_prompt = f"{payload['content']}\n\nUser Input: {request.user_input}"

        executions[execution_id]["progress"] = 0.3
        await broadcast_execution_event(execution_id, {
            "event": "status_update",
            "progress": 0.3,
            "step": "Generating response..."
        })

        # Execute with Ollama
        if request.backend == "ollama":
            output = OllamaExecutor.execute(full_prompt, model=request.model)
        else:
            raise ValueError(f"Unsupported backend: {request.backend}")

        executions[execution_id]["status"] = "completed"
        executions[execution_id]["progress"] = 1.0
        executions[execution_id]["output"] = output
        executions[execution_id]["current_step"] = "Complete"

        await broadcast_execution_event(execution_id, {
            "event": "completed",
            "status": "completed",
            "progress": 1.0,
            "output": output
        })

    except Exception as e:
        logger.error(f"Skill execution {execution_id} failed: {e}", exc_info=True)
        executions[execution_id]["status"] = "failed"
        executions[execution_id]["error"] = str(e)
        await broadcast_execution_event(execution_id, {
            "event": "failed",
            "status": "failed",
            "error": str(e)
        })


@app.post("/execute/chain", response_model=ExecutionResponse)
async def execute_chain(request: ChainExecutionRequest):
    """Execute a chain of skills."""
    if not p:
        raise HTTPException(status_code=503, detail="Promptly not available")

    execution_id = str(uuid.uuid4())
    executions[execution_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_step": "Initializing chain...",
        "type": "chain",
        "request": request.dict()
    }

    asyncio.create_task(_execute_chain_background(execution_id, request))

    return ExecutionResponse(
        execution_id=execution_id,
        status="queued",
        message=f"Chain execution {execution_id} queued"
    )


async def _execute_chain_background(execution_id: str, request: ChainExecutionRequest):
    """Background task for chain execution."""
    try:
        executions[execution_id]["status"] = "running"
        results = []
        current_input = request.initial_input

        for i, skill_name in enumerate(request.skill_names):
            step_progress = (i / len(request.skill_names))
            executions[execution_id]["progress"] = step_progress
            executions[execution_id]["current_step"] = f"Step {i+1}/{len(request.skill_names)}: {skill_name}"

            await broadcast_execution_event(execution_id, {
                "event": "status_update",
                "progress": step_progress,
                "step": f"Step {i+1}/{len(request.skill_names)}: {skill_name}"
            })

            # Execute skill
            payload = p.prepare_skill_payload(skill_name)
            full_prompt = f"{payload['content']}\n\nInput: {current_input}"

            output = OllamaExecutor.execute(full_prompt, model=request.model)
            results.append({"skill": skill_name, "output": output})

            # Output becomes next input
            current_input = output

        executions[execution_id]["status"] = "completed"
        executions[execution_id]["progress"] = 1.0
        executions[execution_id]["output"] = current_input  # Final output
        executions[execution_id]["results"] = results
        executions[execution_id]["current_step"] = "Chain complete"

        await broadcast_execution_event(execution_id, {
            "event": "completed",
            "progress": 1.0,
            "output": current_input,
            "results": results
        })

    except Exception as e:
        logger.error(f"Chain execution {execution_id} failed: {e}", exc_info=True)
        executions[execution_id]["status"] = "failed"
        executions[execution_id]["error"] = str(e)
        await broadcast_execution_event(execution_id, {
            "event": "failed",
            "error": str(e)
        })


@app.post("/execute/loop", response_model=ExecutionResponse)
async def execute_loop(request: LoopExecutionRequest):
    """Execute a recursive loop."""
    if not p:
        raise HTTPException(status_code=503, detail="Promptly not available")

    execution_id = str(uuid.uuid4())
    executions[execution_id] = {
        "status": "queued",
        "progress": 0.0,
        "current_step": "Initializing loop...",
        "type": "loop",
        "request": request.dict()
    }

    asyncio.create_task(_execute_loop_background(execution_id, request))

    return ExecutionResponse(
        execution_id=execution_id,
        status="queued",
        message=f"Loop execution {execution_id} queued"
    )


async def _execute_loop_background(execution_id: str, request: LoopExecutionRequest):
    """Background task for loop execution."""
    try:
        executions[execution_id]["status"] = "running"

        # Get loop type
        loop_type = LoopType(request.loop_type)

        # Configure loop
        config = LoopConfig(
            loop_type=loop_type,
            max_iterations=request.max_iterations,
            quality_threshold=request.quality_threshold,
            enable_scratchpad=True
        )

        # Prepare executor function
        def executor_func(prompt: str) -> str:
            payload = p.prepare_skill_payload(request.skill_name)
            full_prompt = f"{payload['content']}\n\n{prompt}"
            return OllamaExecutor.execute(full_prompt, model=request.model)

        # Create recursive engine
        engine = RecursiveEngine(executor_func)

        # Execute loop with progress callback
        def on_iteration(iteration: int, quality: float):
            progress = iteration / request.max_iterations
            executions[execution_id]["progress"] = progress
            executions[execution_id]["current_step"] = f"Iteration {iteration}/{request.max_iterations} (quality: {quality:.2f})"
            asyncio.create_task(broadcast_execution_event(execution_id, {
                "event": "iteration_update",
                "iteration": iteration,
                "quality": quality,
                "progress": progress
            }))

        # Execute the appropriate loop type
        if loop_type == LoopType.REFINE:
            result = engine.execute_refine_loop(
                task=request.user_input,
                initial_output="",
                config=config
            )
        elif loop_type == LoopType.CRITIQUE:
            result = engine.execute_critique_loop(
                task=request.user_input,
                config=config
            )
        else:
            # For other types, use refine as default
            result = engine.execute_refine_loop(
                task=request.user_input,
                initial_output="",
                config=config
            )

        executions[execution_id]["status"] = "completed"
        executions[execution_id]["progress"] = 1.0
        executions[execution_id]["output"] = result.final_output
        executions[execution_id]["iterations"] = result.iterations
        executions[execution_id]["improvement_history"] = result.improvement_history
        executions[execution_id]["current_step"] = f"Complete ({result.iterations} iterations)"

        await broadcast_execution_event(execution_id, {
            "event": "completed",
            "output": result.final_output,
            "iterations": result.iterations,
            "improvement_history": result.improvement_history,
            "stop_reason": result.stop_reason
        })

    except Exception as e:
        logger.error(f"Loop execution {execution_id} failed: {e}", exc_info=True)
        executions[execution_id]["status"] = "failed"
        executions[execution_id]["error"] = str(e)
        await broadcast_execution_event(execution_id, {
            "event": "failed",
            "error": str(e)
        })


@app.get("/execute/status/{execution_id}", response_model=ExecutionStatus)
async def get_execution_status(execution_id: str):
    """Get execution status."""
    if execution_id not in executions:
        raise HTTPException(status_code=404, detail="Execution not found")

    exec_data = executions[execution_id]
    return ExecutionStatus(
        execution_id=execution_id,
        status=exec_data["status"],
        progress=exec_data["progress"],
        current_step=exec_data["current_step"],
        output=exec_data.get("output"),
        error=exec_data.get("error"),
        metadata={
            "iterations": exec_data.get("iterations"),
            "improvement_history": exec_data.get("improvement_history"),
            "results": exec_data.get("results")
        }
    )


@app.websocket("/ws/execution")
async def execution_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time execution updates."""
    await websocket.accept()
    active_websockets.add(websocket)
    logger.info("WebSocket client connected")

    try:
        while True:
            # Keep connection alive, client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        active_websockets.discard(websocket)


def main():
    """Run the FastAPI server."""
    print("Starting Promptly VS Code Bridge on http://localhost:8765", file=sys.stderr)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info"
    )


if __name__ == "__main__":
    main()