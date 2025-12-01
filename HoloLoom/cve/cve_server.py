"""
CVE WebSocket Server - Real-time Cognitive Event Streaming

Provides WebSocket endpoint for streaming CognitiveFrames to the
consciousness_shell.html client. Integrates with the weaving orchestrator
to extract cognitive events and render them via TufteRenderer.

Created: 2025-11-30

Usage:
    # Standalone server
    python -m HoloLoom.cve.cve_server

    # Or integrate with existing FastAPI app
    from HoloLoom.cve.cve_server import create_cve_router
    app.include_router(create_cve_router())

WebSocket Protocol:
    - Client connects to ws://localhost:8001/cve
    - Server streams CognitiveFrame objects as JSON
    - Each frame contains events with rendered_html (from TufteRenderer)
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
from dataclasses import asdict
import logging

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from HoloLoom.cve.cognitive_protocol import (
    CognitiveEvent,
    CognitiveFrame,
    CognitiveVizType,
    create_event,
    create_frame,
)
from HoloLoom.cve.tufte_renderer import TufteRenderer, render_cognitive_event
from HoloLoom.cve.cognitive_extractors import (
    UnifiedCognitiveExtractor,
    extract_from_awareness,
    extract_from_bandit,
    extract_from_spacetime,
)

logger = logging.getLogger(__name__)


class CVEWebSocketManager:
    """
    Manages WebSocket connections for CVE streaming.

    Handles:
    - Multiple concurrent clients
    - Event broadcasting
    - Connection lifecycle
    - Rate limiting (optional)
    """

    def __init__(self, renderer: Optional[TufteRenderer] = None):
        self.active_connections: Set[WebSocket] = set()
        self.renderer = renderer or TufteRenderer()
        self.extractor = UnifiedCognitiveExtractor()
        self.frame_count = 0
        self.event_count = 0

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"[CVE] Client connected. Total: {len(self.active_connections)}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "CVE WebSocket connected",
            "timestamp": datetime.now().isoformat(),
            "viz_types": [t.value for t in CognitiveVizType],
        })

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        self.active_connections.discard(websocket)
        logger.info(f"[CVE] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast_frame(self, frame: CognitiveFrame):
        """
        Broadcast a CognitiveFrame to all connected clients.

        Each event in the frame is rendered to HTML using TufteRenderer
        before sending to clients.
        """
        self.frame_count += 1

        # Render each event to HTML
        rendered_events = []
        for event in frame.events:
            try:
                rendered_html = self.renderer.render_event(event)
                event_dict = {
                    "event_id": event.event_id,
                    "viz_type": event.viz_type.value,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "data": self._serialize_data(event.data),
                    "rendered_html": rendered_html,
                }
                rendered_events.append(event_dict)
                self.event_count += 1
            except Exception as e:
                logger.warning(f"[CVE] Error rendering event {event.viz_type}: {e}")
                rendered_events.append({
                    "event_id": event.event_id,
                    "viz_type": event.viz_type.value,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "data": self._serialize_data(event.data),
                    "rendered_html": f"<div class='error'>Render error: {e}</div>",
                })

        # Build frame message
        message = {
            "type": "cognitive_frame",
            "frame": {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp.isoformat() if frame.timestamp else None,
                "stage": frame.stage,
                "title": frame.title or f"Frame {self.frame_count}",
                "confidence": frame.confidence,
                "activation": getattr(frame, 'activation', None),
                "coherence": getattr(frame, 'coherence', None),
                "events": rendered_events,
            }
        }

        # Broadcast to all clients
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"[CVE] Failed to send to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_event(self, event: CognitiveEvent):
        """Broadcast a single CognitiveEvent."""
        try:
            rendered_html = self.renderer.render_event(event)
        except Exception as e:
            rendered_html = f"<div class='error'>Render error: {e}</div>"

        message = {
            "type": "cognitive_event",
            "event": {
                "event_id": event.event_id,
                "viz_type": event.viz_type.value,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "data": self._serialize_data(event.data),
                "rendered_html": rendered_html,
            }
        }

        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Broadcast cognitive metrics update."""
        message = {
            "type": "metrics",
            **metrics,
            "timestamp": datetime.now().isoformat(),
        }

        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws)

    def _serialize_data(self, data: Any) -> Any:
        """Serialize dataclass or complex data to JSON-compatible format."""
        if data is None:
            return None
        if hasattr(data, '__dataclass_fields__'):
            return asdict(data)
        if isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        if isinstance(data, dict):
            return {k: self._serialize_data(v) for k, v in data.items()}
        if isinstance(data, datetime):
            return data.isoformat()
        return data

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "active_connections": len(self.active_connections),
            "frames_sent": self.frame_count,
            "events_sent": self.event_count,
            "supported_viz_types": len(CognitiveVizType),
        }


class CVEIntegration:
    """
    Integration layer between HoloLoom weaving cycle and CVE streaming.

    Hooks into the orchestrator to extract cognitive events and stream
    them to connected clients.
    """

    def __init__(self, ws_manager: CVEWebSocketManager):
        self.ws_manager = ws_manager
        self.extractor = UnifiedCognitiveExtractor()

    async def on_weaving_start(self, query: str):
        """Called when weaving cycle starts."""
        event = create_event(
            CognitiveVizType.WEAVING_CYCLE,
            {
                "stage": "start",
                "query": query,
            }
        )
        await self.ws_manager.broadcast_event(event)

    async def on_retrieval_complete(
        self,
        memories: List[Any],
        awareness_graph: Any = None,
    ):
        """Called after memory retrieval."""
        events = []

        # Memory retrieval event
        events.append(create_event(
            CognitiveVizType.MEMORY_RETRIEVAL,
            {
                "retrieved_memories": [
                    {
                        "id": getattr(m, 'id', str(i)),
                        "content": getattr(m, 'content', getattr(m, 'text', str(m))),
                        "relevance": getattr(m, 'relevance', getattr(m, 'score', 0.5)),
                    }
                    for i, m in enumerate(memories[:10])
                ],
                "total_count": len(memories),
            }
        ))

        # Activation spread if awareness graph available
        if awareness_graph:
            try:
                activation_events = extract_from_awareness(awareness_graph)
                events.extend(activation_events)
            except Exception as e:
                logger.warning(f"[CVE] Error extracting awareness: {e}")

        # Create frame
        frame = create_frame(
            events=events,
            stage="retrieval",
            title="Memory Retrieval",
        )
        await self.ws_manager.broadcast_frame(frame)

    async def on_decision_complete(
        self,
        tool_probabilities: Dict[str, float],
        selected_tool: str,
        bandit: Any = None,
    ):
        """Called after decision/convergence."""
        events = []

        # Probability manifold
        events.append(create_event(
            CognitiveVizType.PROBABILITY_MANIFOLD,
            {
                "probabilities": tool_probabilities,
            }
        ))

        # Decision collapse
        events.append(create_event(
            CognitiveVizType.DECISION_COLLAPSE,
            {
                "from_probabilities": tool_probabilities,
                "selected_tool": selected_tool,
                "collapse_strategy": "bayesian_blend",
            }
        ))

        # Tool selection with bandit stats
        if bandit:
            try:
                bandit_events = extract_from_bandit(bandit, selected_tool)
                events.extend(bandit_events)
            except Exception as e:
                logger.warning(f"[CVE] Error extracting bandit: {e}")
        else:
            events.append(create_event(
                CognitiveVizType.TOOL_SELECTION,
                {
                    "selected_tool": selected_tool,
                    "bandit_stats": {
                        "strategy": "bayesian_blend",
                    },
                }
            ))

        frame = create_frame(
            events=events,
            stage="decision",
            title="Decision & Convergence",
        )
        await self.ws_manager.broadcast_frame(frame)

    async def on_generation_start(self):
        """Called when response generation starts."""
        event = create_event(
            CognitiveVizType.TOKEN_STREAM,
            {
                "status": "started",
                "tokens": [],
            }
        )
        await self.ws_manager.broadcast_event(event)

    async def on_token_generated(self, token: str, cumulative: str, index: int):
        """Called for each generated token."""
        event = create_event(
            CognitiveVizType.TOKEN_STREAM,
            {
                "token": token,
                "cumulative_text": cumulative,
                "token_index": index,
            }
        )
        await self.ws_manager.broadcast_event(event)

    async def on_weaving_complete(
        self,
        spacetime: Any,
        confidence: float,
    ):
        """Called when weaving cycle completes."""
        events = []

        # Confidence flow
        events.append(create_event(
            CognitiveVizType.CONFIDENCE_FLOW,
            {
                "confidences": [confidence],
                "final_confidence": confidence,
            }
        ))

        # Extract from spacetime if available
        if spacetime:
            try:
                spacetime_events = extract_from_spacetime(spacetime)
                events.extend(spacetime_events)
            except Exception as e:
                logger.warning(f"[CVE] Error extracting spacetime: {e}")

        frame = create_frame(
            events=events,
            stage="complete",
            title="Weaving Complete",
            confidence=confidence,
        )
        await self.ws_manager.broadcast_frame(frame)

        # Send final metrics
        await self.ws_manager.broadcast_metrics({
            "confidence": confidence,
            "stage": "complete",
        })


def create_cve_app() -> 'FastAPI':
    """
    Create a standalone FastAPI app for CVE streaming.

    Returns a FastAPI application with:
    - WebSocket endpoint at /cve
    - Static file serving for consciousness_shell.html
    - Health check endpoint
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    import os
    app = FastAPI(
        title="CVE - Cognitive Visualization Engine",
        description="Real-time streaming of cognitive events from HoloLoom",
        version="1.0.0",
    )

    # WebSocket manager
    ws_manager = CVEWebSocketManager()

    # Integration layer
    integration = CVEIntegration(ws_manager)

    @app.get("/", response_class=HTMLResponse)
    async def serve_shell():
        """Serve the consciousness_shell.html."""
        shell_path = os.path.join(os.path.dirname(__file__), "consciousness_shell.html")
        if os.path.exists(shell_path):
            with open(shell_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>consciousness_shell.html not found</h1>")

    @app.websocket("/cve")
    async def cve_websocket(websocket: WebSocket):
        """WebSocket endpoint for cognitive event streaming."""
        await ws_manager.connect(websocket)
        try:
            while True:
                # Keep connection alive, handle client messages
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "request_demo":
                    # Client requests demo frames
                    await run_demo_stream(ws_manager)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"[CVE] WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "CVE",
            "stats": ws_manager.get_stats(),
        }

    @app.get("/stats")
    async def stats():
        """Get server statistics."""
        return ws_manager.get_stats()

    # Store references for external access
    app.state.ws_manager = ws_manager
    app.state.integration = integration

    return app


def create_cve_router():
    """
    Create a FastAPI router for CVE endpoints.

    Use this to integrate CVE into an existing FastAPI application:

        from HoloLoom.cve.cve_server import create_cve_router
        app.include_router(create_cve_router(), prefix="/cve")
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required")

    from fastapi import APIRouter
    import os

    router = APIRouter(tags=["CVE"])
    ws_manager = CVEWebSocketManager()
    integration = CVEIntegration(ws_manager)

    @router.get("/", response_class=HTMLResponse)
    async def serve_shell():
        shell_path = os.path.join(os.path.dirname(__file__), "consciousness_shell.html")
        if os.path.exists(shell_path):
            with open(shell_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>consciousness_shell.html not found</h1>")

    @router.websocket("/ws")
    async def cve_websocket(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "request_demo":
                    await run_demo_stream(ws_manager)
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"[CVE] WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    @router.get("/health")
    async def health():
        return {"status": "healthy", "stats": ws_manager.get_stats()}

    # Attach for external access
    router.ws_manager = ws_manager
    router.integration = integration

    return router


async def run_demo_stream(ws_manager: CVEWebSocketManager):
    """
    Run a demo stream of cognitive events.

    Used for testing and demonstration when not connected to a live
    HoloLoom weaving cycle.
    """
    from HoloLoom.cve.cognitive_protocol import (
        ActivationField,
        ProbabilityManifold,
        MemoryRetrieval,
        DecisionCollapse,
        ToolSelection,
        ConfidenceFlow,
    )

    # Demo frame 1: Memory Retrieval
    events = [
        create_event(
            CognitiveVizType.MEMORY_RETRIEVAL,
            MemoryRetrieval(
                retrieved_memories=[
                    {"id": "mem_1", "content": "Thompson Sampling balances exploration and exploitation", "relevance": 0.92},
                    {"id": "mem_2", "content": "Bayesian approach to multi-armed bandits", "relevance": 0.85},
                    {"id": "mem_3", "content": "Beta distribution for binary outcomes", "relevance": 0.78},
                ],
                query="What is Thompson Sampling?",
                retrieval_time_ms=45.2,
                source="hybrid",
            )
        ),
        create_event(
            CognitiveVizType.ACTIVATION_SPREAD,
            ActivationField(
                active_nodes=["thompson_sampling", "bayesian", "exploration", "exploitation", "bandit"],
                activation_levels={"thompson_sampling": 0.95, "bayesian": 0.82, "exploration": 0.75},
                total_activation=0.72,
                coherence=0.68,
            )
        ),
    ]
    frame1 = create_frame(events, stage="retrieval", title="Memory Retrieval")
    await ws_manager.broadcast_frame(frame1)
    await asyncio.sleep(1.5)

    # Demo frame 2: Decision
    events = [
        create_event(
            CognitiveVizType.PROBABILITY_MANIFOLD,
            ProbabilityManifold(
                probabilities={"answer": 0.72, "research": 0.18, "explore": 0.07, "reflect": 0.03},
                entropy=0.45,
                source="unified_policy",
            )
        ),
        create_event(
            CognitiveVizType.DECISION_COLLAPSE,
            DecisionCollapse(
                from_probabilities={"answer": 0.72, "research": 0.18, "explore": 0.07},
                selected_tool="answer",
                collapse_strategy="bayesian_blend",
                collapse_confidence=0.88,
            )
        ),
        create_event(
            CognitiveVizType.TOOL_SELECTION,
            ToolSelection(
                selected_tool="answer",
                available_tools=["answer", "research", "explore", "reflect"],
                bandit_stats={"strategy": "bayesian_blend", "confidence": 0.88},
            )
        ),
    ]
    frame2 = create_frame(events, stage="decision", title="Decision & Convergence", confidence=0.88)
    await ws_manager.broadcast_frame(frame2)
    await asyncio.sleep(1.5)

    # Demo frame 3: Generation
    events = [
        create_event(
            CognitiveVizType.CONFIDENCE_FLOW,
            ConfidenceFlow(
                confidences=[0.72, 0.78, 0.82, 0.85, 0.88, 0.91],
                timestamps=[datetime.now() for _ in range(6)],
                cache_hits=[True, True, False, False, True, True],
            )
        ),
    ]
    frame3 = create_frame(events, stage="complete", title="Weaving Complete", confidence=0.91)
    await ws_manager.broadcast_frame(frame3)


def main():
    """Run the CVE server standalone."""
    import uvicorn

    print("[CVE] Starting Cognitive Visualization Engine server...")
    print("[CVE] Open http://localhost:8001 in your browser")
    print("[CVE] WebSocket endpoint: ws://localhost:8001/cve")

    app = create_cve_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
