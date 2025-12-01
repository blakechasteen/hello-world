"""
ThirdEye WebSocket Server - Bridge between Chat and Visualization
=================================================================

FastAPI server that:
1. Connects to the chat WebSocket (port 8002)
2. Extracts concepts and composes rich scenes from chat messages
3. Streams render commands to the TypeScript client (port 8003)

Architecture:
    Chat (8002) → ThirdEye Server (8003) → TypeScript Client

Visualization Modes:
    - UI_DESIGN: Renders UI mockups and wireframes
    - NARRATIVE: Renders story scenes with characters
    - ARCHITECTURE: Renders system diagrams
    - DATA: Renders charts and visualizations
    - OBJECT: Renders physical objects
    - WORLD: Renders environments
    - ABSTRACT: Fallback concept spheres

Created: 2025-12-01
Author: HoloLoom Team
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from HoloLoom.thirdeye.concept import Concept, ConceptWorld
from HoloLoom.thirdeye.chat_bridge import (
    ChatBridge,
    ConceptExtraction,
    create_chat_bridge,
)
from HoloLoom.thirdeye.thoughtspace import (
    Thoughtspace,
    ThoughtspaceState,
    create_thoughtspace,
)
from HoloLoom.thirdeye.renderers.webgl import WebGLRenderer, create_webgl_renderer
from HoloLoom.thirdeye.renderer_protocol import RenderConfig, RenderQuality

# Import the new scene composer
from HoloLoom.thirdeye.visualizers import (
    VisualizationMode,
    detect_visualization_mode,
)
from HoloLoom.thirdeye.visualizers.scene_composer import (
    SceneComposer,
    get_composer,
    compose_scene,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected ThirdEye client."""
    websocket: WebSocket
    id: str
    subscribed_to_chat: bool = True


class ThirdEyeServer:
    """
    WebSocket server for ThirdEye visualization.

    Manages connections from TypeScript clients and streams
    rich, context-aware visualizations extracted from the chat.

    Supports multiple visualization modes:
    - UI mockups and wireframes
    - Narrative/cinematic scenes
    - Architecture diagrams
    - Data visualizations
    - Physical objects
    - World environments
    - Abstract concept spheres (fallback)
    """

    def __init__(
        self,
        chat_ws_url: str = "ws://localhost:8002",
        embedder: Optional[Callable[[str], List[float]]] = None,
    ):
        self.chat_ws_url = chat_ws_url
        self.embedder = embedder

        # Components
        self.chat_bridge: Optional[ChatBridge] = None
        self.thoughtspace = create_thoughtspace(on_state_change=self._on_state_change)
        self.renderer = create_webgl_renderer(RenderConfig(quality=RenderQuality.MEDIUM))

        # Scene composer for rich visualizations
        self.scene_composer = get_composer()

        # Connected clients
        self.clients: Dict[str, ClientConnection] = {}
        self._client_counter = 0

        # State
        self._running = False
        self._chat_listener_task: Optional[asyncio.Task] = None
        self._current_scene: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        """Start the server and connect to chat."""
        self._running = True

        # Create chat bridge
        self.chat_bridge = await create_chat_bridge(
            chat_ws_url=self.chat_ws_url,
            embedder=self.embedder,
        )

        # Start chat listener in background
        self._chat_listener_task = asyncio.create_task(self._listen_to_chat())
        logger.info("ThirdEye server started")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False

        if self._chat_listener_task:
            self._chat_listener_task.cancel()
            try:
                await self._chat_listener_task
            except asyncio.CancelledError:
                pass

        if self.chat_bridge:
            await self.chat_bridge.stop()

        # Close all client connections
        for client in list(self.clients.values()):
            await client.websocket.close()
        self.clients.clear()

        logger.info("ThirdEye server stopped")

    async def _listen_to_chat(self) -> None:
        """Listen to chat and process concept extractions."""
        if not self.chat_bridge:
            logger.warning("Chat bridge not initialized")
            return

        try:
            async for extraction in self.chat_bridge.listen():
                await self._process_extraction(extraction)
        except asyncio.CancelledError:
            logger.info("Chat listener cancelled")
        except Exception as e:
            logger.error(f"Error in chat listener: {e}")

    async def _process_extraction(self, extraction: ConceptExtraction) -> None:
        """Process a concept extraction and broadcast rich scenes to clients."""
        # Add concepts to thoughtspace (for legacy support)
        for concept in extraction.concepts:
            self.thoughtspace.add_concept(concept)

        # Get the original text from the primary concept
        original_text = ""
        if extraction.primary_concept:
            original_text = extraction.primary_concept.label
            # If the concept has a context attribute, use it
            if hasattr(extraction.primary_concept, 'context'):
                original_text = extraction.primary_concept.context or original_text

        # Compose rich scene from the text
        if original_text:
            scene = self.scene_composer.compose(original_text)
            self._current_scene = scene

            # Broadcast the rich scene
            await self._broadcast({
                "type": "scene_update",
                "data": scene,
            })

            logger.info(f"Composed {scene.get('mode', 'unknown')} scene: {scene.get('title', 'Untitled')}")

        # Generate render commands (for legacy support)
        if extraction.primary_concept and extraction.transition:
            commands = self.renderer.render_transition(
                extraction.transition,
                None,  # Previous concept handled by thoughtspace
                extraction.primary_concept,
            )

            # Broadcast to all clients
            for command in commands:
                await self._broadcast({
                    "type": "render_command",
                    "data": command.to_dict(),
                })

        # Also send the raw extraction for clients that want it
        await self._broadcast({
            "type": "concept_extraction",
            "data": {
                "concepts": [c.to_dict() for c in extraction.concepts],
                "transition": extraction.transition.to_dict() if extraction.transition else None,
                "primary_concept": extraction.primary_concept.to_dict() if extraction.primary_concept else None,
            },
        })

    def _on_state_change(self, state: ThoughtspaceState) -> None:
        """Handle thoughtspace state changes."""
        # Create async task to broadcast state
        asyncio.create_task(self._broadcast({
            "type": "state_update",
            "data": state.to_dict(),
        }))

    async def connect_client(self, websocket: WebSocket) -> str:
        """Register a new client connection."""
        await websocket.accept()

        self._client_counter += 1
        client_id = f"client_{self._client_counter}"

        self.clients[client_id] = ClientConnection(
            websocket=websocket,
            id=client_id,
        )

        logger.info(f"Client {client_id} connected")

        # Send current state
        await self._send_to_client(client_id, {
            "type": "initial_state",
            "data": self.thoughtspace.get_state().to_dict(),
        })

        return client_id

    async def disconnect_client(self, client_id: str) -> None:
        """Unregister a client connection."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def handle_client_message(self, client_id: str, message: dict) -> None:
        """Handle a message from a client."""
        msg_type = message.get("type", "")

        if msg_type == "focus_concept":
            concept_id = message.get("concept_id")
            if concept_id:
                transition = self.thoughtspace.focus_concept(concept_id)
                if transition:
                    await self._broadcast({
                        "type": "render_command",
                        "data": self.renderer.set_camera(
                            self.thoughtspace.camera.x,
                            self.thoughtspace.camera.y,
                            self.thoughtspace.camera.z,
                            self.thoughtspace.camera.target_x,
                            self.thoughtspace.camera.target_y,
                            self.thoughtspace.camera.target_z,
                        ).to_dict(),
                    })

        elif msg_type == "toggle_mode":
            transition = self.thoughtspace.toggle_mode()
            await self._broadcast({
                "type": "mode_changed",
                "data": {
                    "mode": self.thoughtspace.mode.value,
                    "transition": transition.to_dict(),
                },
            })

        elif msg_type == "orbit":
            angle_delta = message.get("angle_delta", 0.1)
            self.thoughtspace.orbit(angle_delta)

        elif msg_type == "zoom":
            factor = message.get("factor", 1.0)
            self.thoughtspace.zoom(factor)

        elif msg_type == "relayout":
            self.thoughtspace.relayout()
            # Send full world render
            frame = self.renderer.render_state(self.thoughtspace.get_state())
            await self._broadcast({
                "type": "render_frame",
                "data": frame.to_dict(),
            })

    async def _send_to_client(self, client_id: str, message: dict) -> None:
        """Send a message to a specific client."""
        client = self.clients.get(client_id)
        if client:
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {client_id}: {e}")
                await self.disconnect_client(client_id)

    async def _broadcast(self, message: dict) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = []
        for client_id, client in self.clients.items():
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect_client(client_id)


# Global server instance
_server: Optional[ThirdEyeServer] = None


def get_server() -> ThirdEyeServer:
    """Get the global server instance."""
    global _server
    if _server is None:
        _server = ThirdEyeServer()
    return _server


# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle."""
    server = get_server()
    await server.start()
    yield
    await server.stop()


app = FastAPI(
    title="ThirdEye Server",
    description="WebSocket server for 3D concept visualization",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "thirdeye",
        "clients": len(get_server().clients),
    }


@app.get("/state")
async def get_state():
    """Get current Thoughtspace state."""
    server = get_server()
    return server.thoughtspace.get_state().to_dict()


class ComposeRequest(BaseModel):
    """Request body for scene composition."""
    text: str
    force_mode: Optional[str] = None


@app.post("/compose")
async def compose_endpoint(request: ComposeRequest):
    """
    Compose a rich visualization scene from text.

    Args:
        text: The content to visualize
        force_mode: Optional mode override (ui_design, narrative, architecture, data, object, world, abstract)

    Returns:
        Scene data ready for rendering
    """
    server = get_server()
    scene = compose_scene(request.text, force_mode=request.force_mode)

    # Broadcast to all connected clients
    await server._broadcast({
        "type": "scene_update",
        "data": scene,
    })

    return {
        "status": "composed",
        "scene": scene,
        "mode": scene.get("mode", "unknown"),
        "element_count": len(scene.get("elements", [])),
    }


@app.get("/compose/modes")
async def get_modes():
    """Get available visualization modes."""
    return {
        "modes": [
            {"value": "ui_design", "description": "UI mockups, wireframes, components"},
            {"value": "narrative", "description": "Stories, scenes, characters, dialogue"},
            {"value": "architecture", "description": "System diagrams, code structure"},
            {"value": "data", "description": "Charts, graphs, metrics"},
            {"value": "object", "description": "Physical things, products"},
            {"value": "world", "description": "Environments, spaces, maps"},
            {"value": "abstract", "description": "Fallback - concept spheres"},
        ]
    }


@app.get("/scene/current")
async def get_current_scene():
    """Get the currently displayed scene."""
    server = get_server()
    if server._current_scene:
        return {
            "status": "active",
            "scene": server._current_scene,
        }
    return {
        "status": "empty",
        "scene": None,
    }


@app.websocket("/ws/thirdeye")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for ThirdEye clients."""
    server = get_server()
    client_id = await server.connect_client(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            await server.handle_client_message(client_id, data)
    except WebSocketDisconnect:
        await server.disconnect_client(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await server.disconnect_client(client_id)


# Demo endpoint with simple HTML
DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>ThirdEye Demo</title>
    <style>
        body { margin: 0; overflow: hidden; font-family: system-ui, sans-serif; }
        #container { width: 100vw; height: 100vh; }
        #status {
            position: fixed;
            top: 10px;
            left: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
        }
        #controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: #3b82f6;
            color: white;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #2563eb; }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="status">Connecting...</div>
    <div id="controls">
        <button onclick="toggleMode()">Toggle Fullscreen</button>
        <button onclick="relayout()">Relayout</button>
        <button onclick="addTestConcept()">Add Test Concept</button>
    </div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/",
            "gsap": "https://cdn.skypack.dev/gsap"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import gsap from 'gsap';

        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#f8fafc');
        scene.fog = new THREE.Fog('#f8fafc', 50, 100);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 5, 15);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lighting
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 10);
        scene.add(dirLight);

        // Floor grid
        const grid = new THREE.GridHelper(100, 20, 0xe2e8f0, 0xe2e8f0);
        grid.position.y = -5;
        grid.material.opacity = 0.3;
        grid.material.transparent = true;
        scene.add(grid);

        // Concept nodes
        const nodes = new Map();

        function createConceptNode(concept) {
            const group = new THREE.Group();
            group.position.set(concept.position.x, concept.position.y, concept.position.z);

            const colors = {
                entity: 0x3b82f6, relationship: 0x10b981, process: 0x8b5cf6,
                comparison: 0xf59e0b, hierarchy: 0x6366f1, temporal: 0x06b6d4,
            };
            const color = colors[concept.type] || 0x64748b;

            const geometry = new THREE.SphereGeometry(0.5 + concept.importance * 1.5, 32, 32);
            const material = new THREE.MeshPhongMaterial({
                color, opacity: 0.7 + concept.confidence * 0.3, transparent: true,
                emissive: color, emissiveIntensity: 0.1,
            });
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);

            scene.add(group);
            nodes.set(concept.id, { group, mesh, concept });

            // Animate in
            group.scale.setScalar(0);
            gsap.to(group.scale, { x: 1, y: 1, z: 1, duration: 0.5, ease: 'back.out(1.7)' });

            return group;
        }

        // WebSocket connection
        let ws;
        const status = document.getElementById('status');

        function connect() {
            ws = new WebSocket('ws://localhost:8003/ws/thirdeye');

            ws.onopen = () => {
                status.textContent = 'Connected';
                status.style.background = 'rgba(16, 185, 129, 0.9)';
                status.style.color = 'white';
            };

            ws.onclose = () => {
                status.textContent = 'Disconnected - Reconnecting...';
                status.style.background = 'rgba(239, 68, 68, 0.9)';
                status.style.color = 'white';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };
        }

        function handleMessage(msg) {
            switch (msg.type) {
                case 'initial_state':
                    // Render existing concepts
                    const concepts = msg.data.world?.concepts || {};
                    for (const concept of Object.values(concepts)) {
                        createConceptNode(concept);
                    }
                    break;

                case 'concept_extraction':
                    const extraction = msg.data;
                    if (extraction.primary_concept) {
                        const node = createConceptNode(extraction.primary_concept);
                        // Focus camera
                        const pos = extraction.primary_concept.position;
                        gsap.to(camera.position, {
                            x: pos.x, y: pos.y + 5, z: pos.z + 15,
                            duration: 0.5, ease: 'power2.inOut'
                        });
                        gsap.to(controls.target, {
                            x: pos.x, y: pos.y, z: pos.z,
                            duration: 0.5, ease: 'power2.inOut'
                        });
                    }
                    break;

                case 'render_command':
                    // Handle render commands
                    console.log('Render command:', msg.data);
                    break;
            }
        }

        connect();

        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        // Resize handler
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // Controls
        window.toggleMode = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'toggle_mode' }));
            }
        };

        window.relayout = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'relayout' }));
            }
        };

        let testCounter = 0;
        window.addTestConcept = () => {
            testCounter++;
            const types = ['entity', 'relationship', 'process', 'comparison'];
            const concept = {
                id: `test_${testCounter}`,
                label: `Test Concept ${testCounter}`,
                type: types[testCounter % types.length],
                position: {
                    x: (Math.random() - 0.5) * 20,
                    y: (Math.random() - 0.5) * 10,
                    z: (Math.random() - 0.5) * 20,
                },
                importance: 0.3 + Math.random() * 0.7,
                confidence: 0.5 + Math.random() * 0.5,
            };
            createConceptNode(concept);
        };
    </script>
</body>
</html>
"""


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve the demo page."""
    return DEMO_HTML


def run_server(host: str = "0.0.0.0", port: int = 8003):
    """Run the ThirdEye server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
