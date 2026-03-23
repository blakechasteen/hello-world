"""
HoloLoom Daemon — background runtime host on ws://127.0.0.1:4242.
==================================================================

Mutable server code. The protocol (daemon_protocol.py) is immutable.

Starts the HoloLoomRuntime with registered layers, opens a WebSocket,
and handles client requests. Streams bus signals to subscribers.

Usage:
    daemon = HoloLoomDaemon()
    await daemon.serve()  # blocks until shutdown

Or as a module:
    python -m hololoom.core.runtime.daemon
"""

import asyncio
import logging
from typing import Any

from .daemon_protocol import DaemonRequest, DaemonResponse, DaemonEvent
from .runtime import HoloLoomRuntime

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4242


def _get_env(key: str, default: str) -> str:
    import os
    return os.environ.get(key, default)

# Optional websockets import — graceful degradation
try:
    import websockets
    import websockets.server
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.debug("websockets not installed — daemon WebSocket disabled")


class HoloLoomDaemon:
    """
    WebSocket daemon that hosts a HoloLoomRuntime.

    Handles:
      - status: return runtime status
      - query: emit a signal and await result (stub for weaving pipeline)
      - fire_ritual: directly fire a ritual (stub)
      - replay: query the ritual journal (stub)
      - subscribe: stream bus signals to client
    """

    def __init__(
        self,
        runtime: HoloLoomRuntime | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self._runtime = runtime or HoloLoomRuntime()
        self._host = host
        self._port = port
        self._running = False
        self._clients: set[Any] = set()
        self._subscribers: dict[Any, set[str]] = {}  # ws → set of signal kinds
        self._stop_event: asyncio.Event | None = None
        self._last_backend: str = "none"

    async def handle_request(self, request: DaemonRequest) -> DaemonResponse:
        """Route a request to the appropriate handler."""
        handlers = {
            "status": self._handle_status,
            "query": self._handle_query,
            "fire_ritual": self._handle_fire_ritual,
            "replay": self._handle_replay,
            "skill_query": self._handle_skill_query,
        }

        handler = handlers.get(request.type)
        if handler is None:
            return DaemonResponse.error(
                f"Unknown request type: {request.type}",
                request_id=request.request_id,
            )

        try:
            return await handler(request)
        except Exception as e:
            logger.exception("Request handler failed: %s", e)
            return DaemonResponse.error(str(e), request_id=request.request_id)

    async def _handle_status(self, request: DaemonRequest) -> DaemonResponse:
        return DaemonResponse(
            type="result",
            request_id=request.request_id,
            payload=self._runtime.status(),
        )

    async def _handle_query(self, request: DaemonRequest) -> DaemonResponse:
        """
        Route a query through the best available intelligence path.

        Tries (in order):
          1. HoloLoom unified API (query/chat — full weaving pipeline)
          2. Promptly model router (Ollama/AirLLM with Thompson Sampling)
          3. Direct Ollama fallback (reflex arc)

        Also fires TASK_RECEIVED ritual for the request.
        """
        text = request.payload.get("text", "")
        if not text:
            return DaemonResponse.error("Missing 'text' in payload", request_id=request.request_id)

        room_id = request.payload.get("room_id", "daemon")

        # Fire rituals for this request (non-blocking)
        try:
            ritual_layer = self._runtime.get_layer("ritual")
            if ritual_layer is not None:
                from ..ritual.hooks import HookEvent
                from ..ritual.types import RitualContext, TokenBudget
                ctx = RitualContext(
                    agent_id=request.payload.get("agent_id", "daemon"),
                    agent_role=request.payload.get("agent_role", "assistant"),
                    domain=request.payload.get("domain", "universal"),
                    task_id=request.request_id,
                    task_type=request.payload.get("task_type", "chat"),
                    task_intent=text[:200],
                    task_complexity="standard",
                    yarn=None, warp=None,
                    confidence=0.8, confidence_history=[0.8],
                    ritual_outputs={}, ritual_log=[],
                    token_budget=TokenBudget(total=5000, used=0),
                    time_budget=60.0, escalation_path=[],
                    session_id=request.payload.get("session_id", room_id),
                )
                await ritual_layer.registry.fire(HookEvent.TASK_RECEIVED, ctx)
        except Exception as e:
            logger.debug("Ritual dispatch failed (non-blocking): %s", e)

        # Try intelligence paths in order
        response_text = await self._try_hololoom_query(text)
        if response_text is None:
            response_text = await self._try_ollama_query(text)
        if response_text is None:
            response_text = "[daemon] No intelligence backend available"

        return DaemonResponse(
            type="result",
            thread_id=request.thread_id or room_id,
            request_id=request.request_id,
            payload={
                "response": response_text,
                "backend": self._last_backend,
            },
        )

    async def _try_hololoom_query(self, text: str) -> str | None:
        """Try the HoloLoom unified API (full weaving pipeline)."""
        try:
            from hololoom import HoloLoom as HoloLoomAPI
            if not hasattr(self, '_loom'):
                self._loom = await HoloLoomAPI.create(pattern="fast")
            result = await self._loom.query(text, return_trace=False)
            self._last_backend = "hololoom"
            return str(result)
        except Exception as e:
            logger.debug("HoloLoom query unavailable: %s", e)
            return None

    async def _try_ollama_query(self, text: str) -> str | None:
        """Try direct Ollama as reflex arc."""
        try:
            import httpx
            ollama_url = _get_env("PROMPTLY_OLLAMA_URL", "http://127.0.0.1:11434")
            ollama_model = _get_env("OLLAMA_MODEL", "qwen3.5:9b")

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": ollama_model, "prompt": text, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
                self._last_backend = f"ollama/{ollama_model}"
                return data.get("response", "")
        except Exception as e:
            logger.debug("Ollama query failed: %s", e)
            return None

    async def _handle_fire_ritual(self, request: DaemonRequest) -> DaemonResponse:
        return DaemonResponse(
            type="result",
            request_id=request.request_id,
            payload={"message": "fire_ritual handler not yet wired"},
        )

    async def _handle_replay(self, request: DaemonRequest) -> DaemonResponse:
        return DaemonResponse(
            type="result",
            request_id=request.request_id,
            payload={"message": "replay handler not yet wired to journal"},
        )

    async def _handle_skill_query(self, request: DaemonRequest) -> DaemonResponse:
        return DaemonResponse(
            type="result",
            request_id=request.request_id,
            payload={"message": "skill_query handler not yet wired"},
        )

    # ========================================================================
    # WebSocket server
    # ========================================================================

    async def serve(self) -> None:
        """Start the runtime and WebSocket server. Blocks until shutdown."""
        if not HAS_WEBSOCKETS:
            logger.error(
                "Cannot start daemon — 'websockets' package not installed. "
                "Install with: pip install hololoom[server]"
            )
            return

        await self.start()
        self._stop_event = asyncio.Event()

        logger.info("HoloLoom daemon listening on ws://%s:%d", self._host, self._port)

        async with websockets.serve(self._ws_handler, self._host, self._port):
            await self._stop_event.wait()

        await self.stop()

    async def _ws_handler(self, ws: Any) -> None:
        """Handle a single WebSocket connection."""
        self._clients.add(ws)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for message in ws:
                valid, err = DaemonRequest.validate(message)
                if not valid:
                    await ws.send(DaemonResponse.error(err).to_json())
                    continue

                req = DaemonRequest.from_json(message)

                if req.type == "subscribe":
                    kinds = req.payload.get("kinds", [])
                    self._subscribers[ws] = set(kinds)
                    await ws.send(DaemonResponse(
                        type="result",
                        request_id=req.request_id,
                        payload={"subscribed": kinds},
                    ).to_json())
                else:
                    resp = await self.handle_request(req)
                    await ws.send(resp.to_json())
        except Exception as e:
            logger.debug("WebSocket connection closed: %s", e)
        finally:
            self._clients.discard(ws)
            self._subscribers.pop(ws, None)
            logger.info("Client disconnected (%d remaining)", len(self._clients))

    async def broadcast_event(self, event: DaemonEvent) -> None:
        """Broadcast a DaemonEvent to all subscribers matching the signal kind."""
        signal_kind = event.signal.get("kind", "")
        message = event.to_json()
        dead: list[Any] = []

        for ws, kinds in self._subscribers.items():
            if not kinds or signal_kind in kinds:
                try:
                    await ws.send(message)
                except Exception:
                    dead.append(ws)

        for ws in dead:
            self._clients.discard(ws)
            self._subscribers.pop(ws, None)

    def request_shutdown(self) -> None:
        """Signal the daemon to shut down gracefully."""
        if self._stop_event is not None:
            self._stop_event.set()

    async def start(self) -> None:
        """Start the runtime (without opening WebSocket)."""
        await self._runtime.start()
        self._running = True
        logger.info("HoloLoom daemon started")

    async def stop(self) -> None:
        """Stop the runtime and close all connections."""
        self._running = False
        await self._runtime.stop()
        logger.info("HoloLoom daemon stopped")

    @property
    def runtime(self) -> HoloLoomRuntime:
        return self._runtime

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================================
# Module entry point
# ============================================================================

if __name__ == "__main__":
    import signal

    logging.basicConfig(level=logging.INFO)
    from .temporal_layer import TemporalLayer
    from .ritual_layer import RitualLayer

    daemon = HoloLoomDaemon()
    daemon.runtime.register_layer(TemporalLayer())
    daemon.runtime.register_layer(RitualLayer(register_defaults=True))

    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        logger.info("Received %s — shutting down", sig)
        daemon.request_shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    loop.run_until_complete(daemon.serve())
