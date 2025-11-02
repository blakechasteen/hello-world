#!/usr/bin/env python3
"""
Agentic Web Dashboard - FastAPI Server
========================================

WebSocket-based server for real-time chat with agentic intelligence.

Features:
- 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- LLM-activated intelligent search
- Persistent memory (Neo4j + Qdrant)
- Complete provenance tracking
- Real-time visualization of reasoning steps

Endpoints:
- GET / - Dashboard HTML
- WS /ws - WebSocket for real-time chat
- GET /api/threads - List all threads
- GET /api/thread/{id} - Get specific thread
- GET /api/status - Server status

Usage:
    python HoloLoom/web_dashboard/agentic_server.py

Then open: http://localhost:8001
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio
from typing import Dict, List
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.alignment.audit_trail import OutcomeType

# Create FastAPI app
app = FastAPI(title="HoloLoom Agentic Dashboard")

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Agentic orchestrator (initialized on startup)
agentic_orchestrator = None
orchestrator_config = None


@app.on_event("startup")
async def startup_event():
    """Initialize agentic orchestrator on startup"""
    global agentic_orchestrator, orchestrator_config

    logger.info("="*60)
    logger.info("🚀 Starting HoloLoom Agentic Dashboard")
    logger.info("="*60)

    # Create configuration
    orchestrator_config = Config.fast()
    orchestrator_config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant with fallback

    # Create initial knowledge shards (can be loaded from files)
    shards = create_demo_shards()
    logger.info(f"✓ Loaded {len(shards)} knowledge shards")

    # Create agentic orchestrator with persistent memory
    try:
        agentic_orchestrator = await create_agentic_orchestrator(
            config=orchestrator_config,
            shards=shards,
            enable_verification=True,
            enable_goal_tracking=True
        )

        # Check LLM status
        llm_status = "available" if agentic_orchestrator.llm else "unavailable"
        logger.info(f"✓ Agentic orchestrator initialized")
        logger.info(f"  - LLM status: {llm_status}")
        logger.info(f"  - Alignment: enabled")
        logger.info(f"  - Memory backend: {orchestrator_config.memory_backend.value}")
        logger.info(f"  - Verification: enabled")
        logger.info(f"  - Goal tracking: enabled")

    except Exception as e:
        logger.error(f"✗ Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        raise

    logger.info("="*60)
    logger.info("✓ Dashboard ready at http://localhost:8001")
    logger.info("="*60)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard HTML"""
    html_file = Path(__file__).parent / "agentic_dashboard.html"

    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Return inline HTML
        return HTMLResponse(content=get_agentic_html())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agentic chat"""
    await websocket.accept()
    active_connections.append(websocket)

    logger.info(f"✓ Client connected. Active connections: {len(active_connections)}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            action = message_data.get('action', 'reason')

            if action == 'reason':
                # Process agentic reasoning request
                user_query = message_data.get('query', '')
                reasoning_mode_str = message_data.get('mode', 'direct')
                max_steps = message_data.get('max_steps', 3)

                # Map string to ReasoningMode enum
                mode_map = {
                    'direct': ReasoningMode.DIRECT,
                    'verify': ReasoningMode.VERIFY,
                    'research': ReasoningMode.RESEARCH,
                    'plan_execute': ReasoningMode.PLAN_EXECUTE
                }
                reasoning_mode = mode_map.get(reasoning_mode_str, ReasoningMode.DIRECT)

                logger.info(f"\n{'='*60}")
                logger.info(f"📥 Query: {user_query}")
                logger.info(f"🧠 Mode: {reasoning_mode.value.upper()}")
                logger.info(f"{'='*60}")

                # Send acknowledgment
                await websocket.send_json({
                    'type': 'reasoning_started',
                    'data': {
                        'query': user_query,
                        'mode': reasoning_mode.value,
                        'max_steps': max_steps
                    }
                })

                try:
                    # Execute agentic reasoning
                    result = await agentic_orchestrator.reason(
                        query=Query(text=user_query),
                        mode=reasoning_mode,
                        max_steps=max_steps
                    )

                    # Extract reasoning steps with LLM markers
                    reasoning_steps = []
                    for step in result.steps_taken:
                        step_info = {
                            'type': step.get('type', 'unknown'),
                            'llm_generated': step.get('llm_generated', False),
                            'confidence': step.get('confidence', 0.0),
                            'query': step.get('query', ''),
                            'findings': step.get('findings', ''),
                            'checks': step.get('checks', []),
                            'goals': step.get('goals', []),
                            'sources': step.get('sources', 0)
                        }
                        reasoning_steps.append(step_info)

                    # Extract audit trail
                    audit_logs = []
                    if agentic_orchestrator.audit_trail:
                        for log in agentic_orchestrator.audit_trail.logs[-5:]:  # Last 5
                            audit_logs.append({
                                'action': log.action_description or log.decision_type.value,
                                'outcome': log.outcome.value,
                                'risk': log.risk_level or 'unknown',
                                'confidence': log.confidence,
                                'timestamp': log.timestamp.isoformat()
                            })

                    # Build response
                    response_data = {
                        'query': user_query,
                        'mode': reasoning_mode.value,
                        'response': result.spacetime.response or "No response generated",
                        'confidence': result.spacetime.confidence,
                        'total_steps': len(result.steps_taken),
                        'total_queries': result.total_queries,
                        'duration_ms': result.total_duration_ms,
                        'reasoning_steps': reasoning_steps,
                        'audit_trail': audit_logs,
                        'llm_status': 'active' if agentic_orchestrator.llm else 'unavailable'
                    }

                    logger.info(f"✓ Reasoning complete:")
                    logger.info(f"  - Steps: {len(result.steps_taken)}")
                    logger.info(f"  - Confidence: {result.spacetime.confidence:.2f}")
                    logger.info(f"  - Duration: {result.total_duration_ms:.1f}ms")

                    # Send response
                    await websocket.send_json({
                        'type': 'reasoning_complete',
                        'data': response_data
                    })

                except Exception as e:
                    logger.error(f"✗ Reasoning failed: {e}")
                    import traceback
                    traceback.print_exc()

                    await websocket.send_json({
                        'type': 'reasoning_error',
                        'data': {
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
                    })

            elif action == 'get_status':
                # Return system status
                status = {
                    'orchestrator_ready': agentic_orchestrator is not None,
                    'llm_available': agentic_orchestrator.llm is not None if agentic_orchestrator else False,
                    'memory_backend': orchestrator_config.memory_backend.value if orchestrator_config else 'unknown',
                    'alignment_enabled': True,
                    'active_connections': len(active_connections)
                }

                await websocket.send_json({
                    'type': 'status_response',
                    'data': status
                })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"✓ Client disconnected. Active connections: {len(active_connections)}")


@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        'status': 'running',
        'orchestrator_ready': agentic_orchestrator is not None,
        'llm_available': agentic_orchestrator.llm is not None if agentic_orchestrator else False,
        'memory_backend': orchestrator_config.memory_backend.value if orchestrator_config else 'unknown',
        'alignment_enabled': True,
        'active_connections': len(active_connections)
    }


def create_demo_shards() -> List[MemoryShard]:
    """Create demo knowledge shards"""
    return [
        MemoryShard(
            id="shard_thompson_1",
            text=(
                "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                "It maintains a probability distribution (typically Beta) for each arm's reward. "
                "At each step, it samples from these distributions and selects the arm with the "
                "highest sample. This naturally balances exploration and exploitation through "
                "the uncertainty in the posterior distributions."
            ),
            metadata={"topic": "reinforcement_learning", "confidence": 0.95}
        ),
        MemoryShard(
            id="shard_thompson_2",
            text=(
                "Compared to epsilon-greedy and UCB, Thompson Sampling often converges faster. "
                "UCB uses deterministic upper confidence bounds while Thompson Sampling uses "
                "probabilistic sampling. Epsilon-greedy has fixed exploration rate, while "
                "Thompson Sampling adapts based on posterior uncertainty."
            ),
            metadata={"topic": "algorithm_comparison", "confidence": 0.90}
        ),
        MemoryShard(
            id="shard_transformers",
            text=(
                "Transformers revolutionized NLP through self-attention mechanisms. Unlike RNNs, "
                "they process sequences in parallel, using attention to weigh the importance of "
                "different positions. Key innovation: attention scores computed as "
                "softmax(QK^T/√d_k)V."
            ),
            metadata={"topic": "deep_learning", "confidence": 0.92}
        )
    ]


def get_agentic_html() -> str:
    """Return agentic dashboard HTML"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom - Agentic Intelligence</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #aaa;
            font-size: 1.1em;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
        }

        .panel-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 15px;
            color: #fff;
        }

        .mode-selector {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .mode-btn {
            padding: 12px;
            background: rgba(102, 126, 234, 0.2);
            border: 2px solid #667eea;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 1em;
        }

        .mode-btn:hover {
            background: rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }

        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: #764ba2;
        }

        .mode-description {
            font-size: 0.85em;
            color: #aaa;
            margin-top: 5px;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .send-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }

        .send-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .response-area {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }

        .reasoning-step {
            background: rgba(0, 0, 0, 0.3);
            border-left: 3px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }

        .reasoning-step.llm-generated {
            border-left-color: #f39c12;
        }

        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .step-type {
            font-weight: 600;
            color: #667eea;
        }

        .step-type.llm-generated::after {
            content: " 🤖";
        }

        .step-confidence {
            color: #aaa;
            font-size: 0.9em;
        }

        .step-content {
            color: #ddd;
            line-height: 1.6;
        }

        .final-response {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 2px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }

        .stat-box {
            background: rgba(0, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.5em;
            font-weight: 600;
            color: #667eea;
        }

        .stat-label {
            font-size: 0.85em;
            color: #aaa;
            margin-top: 5px;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }

        .status-indicator.online {
            background: #2ecc71;
            box-shadow: 0 0 5px #2ecc71;
        }

        .status-indicator.offline {
            background: #e74c3c;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #aaa;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.8);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 HoloLoom Agentic Intelligence</h1>
            <p class="subtitle">
                <span class="status-indicator online" id="statusIndicator"></span>
                <span id="statusText">Connecting...</span>
            </p>
        </header>

        <div class="main-grid">
            <!-- Left Panel: Controls -->
            <div class="panel">
                <div class="panel-title">Reasoning Mode</div>
                <div class="mode-selector">
                    <button class="mode-btn active" data-mode="direct">
                        <div>DIRECT</div>
                        <div class="mode-description">Single-pass answer (~150ms)</div>
                    </button>
                    <button class="mode-btn" data-mode="verify">
                        <div>VERIFY</div>
                        <div class="mode-description">Answer + verification (~340ms)</div>
                    </button>
                    <button class="mode-btn" data-mode="research">
                        <div>RESEARCH ⭐</div>
                        <div class="mode-description">LLM-activated search (~4500ms)</div>
                    </button>
                    <button class="mode-btn" data-mode="plan_execute">
                        <div>PLAN & EXECUTE</div>
                        <div class="mode-description">Goal decomposition (~340ms)</div>
                    </button>
                </div>

                <div style="margin-top: 30px;">
                    <div class="panel-title">Your Query</div>
                    <textarea id="queryInput" placeholder="Enter your question here...
Example: How does Thompson Sampling compare to other exploration strategies?"></textarea>
                    <button class="send-btn" id="sendBtn">🚀 Send Query</button>
                </div>
            </div>

            <!-- Right Panel: Response -->
            <div class="panel">
                <div class="panel-title">Response</div>
                <div class="response-area" id="responseArea">
                    <div class="loading">
                        <p>Ready to process your queries with agentic intelligence.</p>
                        <p style="margin-top: 10px; color: #667eea;">Select a reasoning mode and enter your question.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let selectedMode = 'direct';

        // Connect to WebSocket
        function connect() {
            ws = new WebSocket('ws://localhost:8001/ws');

            ws.onopen = () => {
                console.log('✓ Connected to server');
                document.getElementById('statusIndicator').className = 'status-indicator online';
                document.getElementById('statusText').textContent = 'Connected';

                // Request status
                ws.send(JSON.stringify({ action: 'get_status' }));
            };

            ws.onclose = () => {
                console.log('✗ Disconnected from server');
                document.getElementById('statusIndicator').className = 'status-indicator offline';
                document.getElementById('statusText').textContent = 'Disconnected - Reconnecting...';

                // Reconnect after 3 seconds
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        // Handle incoming messages
        function handleMessage(message) {
            console.log('Received:', message.type);

            if (message.type === 'status_response') {
                const status = message.data;
                const statusText = document.getElementById('statusText');
                statusText.innerHTML = `
                    LLM: ${status.llm_available ? 'Active 🤖' : 'Unavailable'}
                    | Memory: ${status.memory_backend}
                    | Alignment: ${status.alignment_enabled ? 'Enabled ✓' : 'Disabled'}
                `;
            }
            else if (message.type === 'reasoning_started') {
                const responseArea = document.getElementById('responseArea');
                responseArea.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Processing query with ${message.data.mode.toUpperCase()} mode...</p>
                    </div>
                `;
            }
            else if (message.type === 'reasoning_complete') {
                displayResponse(message.data);
            }
            else if (message.type === 'reasoning_error') {
                const responseArea = document.getElementById('responseArea');
                responseArea.innerHTML = `
                    <div style="color: #e74c3c; padding: 20px;">
                        <h3>Error</h3>
                        <p>${message.data.error}</p>
                    </div>
                `;
            }

            // Re-enable send button
            document.getElementById('sendBtn').disabled = false;
        }

        // Display reasoning response
        function displayResponse(data) {
            const responseArea = document.getElementById('responseArea');

            let html = '';

            // Reasoning steps
            if (data.reasoning_steps && data.reasoning_steps.length > 0) {
                html += '<div style="margin-bottom: 20px;">';
                html += '<h3 style="margin-bottom: 15px;">Reasoning Steps:</h3>';

                data.reasoning_steps.forEach((step, i) => {
                    const llmClass = step.llm_generated ? 'llm-generated' : '';
                    const typeClass = step.llm_generated ? 'llm-generated' : '';

                    html += `<div class="reasoning-step ${llmClass}">`;
                    html += `<div class="step-header">`;
                    html += `<div class="step-type ${typeClass}">Step ${i + 1}: ${step.type.toUpperCase()}</div>`;
                    html += `<div class="step-confidence">Confidence: ${(step.confidence * 100).toFixed(0)}%</div>`;
                    html += `</div>`;

                    if (step.query) {
                        html += `<div class="step-content"><strong>Query:</strong> ${step.query}</div>`;
                    }
                    if (step.findings) {
                        const findings = step.findings.substring(0, 200) + (step.findings.length > 200 ? '...' : '');
                        html += `<div class="step-content" style="margin-top: 5px;"><strong>Findings:</strong> ${findings}</div>`;
                    }

                    html += `</div>`;
                });

                html += '</div>';
            }

            // Final response
            html += `<div class="final-response">`;
            html += `<h3 style="margin-bottom: 15px;">Final Response:</h3>`;
            html += `<p style="line-height: 1.6;">${data.response}</p>`;
            html += `</div>`;

            // Stats
            html += `<div class="stats">`;
            html += `<div class="stat-box"><div class="stat-value">${data.total_steps}</div><div class="stat-label">Steps</div></div>`;
            html += `<div class="stat-box"><div class="stat-value">${(data.confidence * 100).toFixed(0)}%</div><div class="stat-label">Confidence</div></div>`;
            html += `<div class="stat-box"><div class="stat-value">${data.duration_ms.toFixed(0)}ms</div><div class="stat-label">Duration</div></div>`;
            html += `<div class="stat-box"><div class="stat-value">${data.llm_status === 'active' ? '🤖' : '⚠️'}</div><div class="stat-label">LLM</div></div>`;
            html += `</div>`;

            responseArea.innerHTML = html;
        }

        // Mode selection
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                selectedMode = btn.getAttribute('data-mode');
            });
        });

        // Send query
        document.getElementById('sendBtn').addEventListener('click', () => {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) {
                alert('Please enter a question');
                return;
            }

            // Disable send button
            document.getElementById('sendBtn').disabled = true;

            // Send to server
            ws.send(JSON.stringify({
                action: 'reason',
                query: query,
                mode: selectedMode,
                max_steps: 3
            }));
        });

        // Allow Enter to send (with Shift+Enter for new line)
        document.getElementById('queryInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('sendBtn').click();
            }
        });

        // Connect on load
        connect();
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("  >> Starting HoloLoom Agentic Dashboard")
    print("="*60)
    print("\n  Open your browser to: http://localhost:8001\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
