"""
deploy()   — agent spec → FastAPI router with /chat and /status
discover() — scan a package for agent modules, mount all routers

deploy() is the quick path: one function call produces a working agent endpoint.
discover() is the scale path: drop a file in the agents directory, it auto-mounts.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .draft import Agent, Pass, Soul
from .llm import LLMBackend, ollama
from .memory import Memory, conversation
from .models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def deploy(
    name: str,
    soul: Soul,
    backend: LLMBackend | None = None,
    memory: Memory | None = None,
    pipeline: list[Pass] | None = None,
    status_extras: Callable[[], dict[str, Any]] | None = None,
) -> APIRouter:
    """
    Turn an agent spec into a FastAPI router with /chat and /status.

    Minimal:
        router = deploy("cora", soul="You are Cora.")

    Full:
        router = deploy(
            name="elle",
            soul=lambda: load_soul("elle", fallback=FALLBACK),
            memory=conversation(turns=10, db="elle.db", label="elle"),
            backend=ollama("qwen3.5:9b", temperature=0.6),
            pipeline=[with_coz, think(backend)],
            status_extras=lambda: {"vault": True},
        )

    Returns a FastAPI APIRouter with:
        POST /{name}/chat   — ChatRequest → ChatResponse
        GET  /{name}/status — agent config and stats

    The Agent is exposed as router.agent for consult() and introspection.
    """
    if backend is None:
        backend = ollama()
    if memory is None:
        memory = conversation(db=f"{name}.db", label=name)

    handler = Agent(
        soul=soul,
        memory=memory,
        backend=backend,
        pipeline=pipeline,
        name=name,
    )

    router = APIRouter(prefix=f"/{name}", tags=[f"{name}-chat"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        try:
            return await handler(request)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    @router.get("/status")
    async def status():
        info: dict[str, Any] = {
            "name": name,
            "model": getattr(backend, "model", "unknown"),
            "active_rooms": memory.room_count(),
            "total_turns": memory.total_turns(),
            "max_history": memory.max_turns,
        }
        if status_extras:
            try:
                info.update(status_extras())
            except Exception as e:
                logger.warning("status_extras failed for %s: %s", name, e)
        return info

    # Expose for consult() and introspection
    router.agent = handler

    return router


def discover(app: FastAPI, package: str) -> list[str]:
    """
    Scan a package for modules with a `router` attribute. Mount each one.

    Graceful: if one agent fails to import, the rest still mount.
    Returns names of successfully mounted agents.

    Usage:
        mounted = discover(app, "hololoom.apps.server.agents")
        logger.info("Mounted agents: %s", mounted)
    """
    mounted = []

    try:
        pkg = importlib.import_module(package)
    except ImportError as e:
        logger.warning("Could not import agent package %s: %s", package, e)
        return mounted

    for _importer, modname, _ispkg in pkgutil.iter_modules(
        pkg.__path__, prefix=f"{package}."
    ):
        try:
            mod = importlib.import_module(modname)
            router = getattr(mod, "router", None)
            if router is not None:
                app.include_router(router)
                agent_name = getattr(mod, "name", modname.rsplit(".", 1)[-1])
                mounted.append(agent_name)
                logger.info("Mounted agent: %s", agent_name)
        except Exception as e:
            logger.warning("Failed to mount agent %s: %s", modname, e)

    return mounted
