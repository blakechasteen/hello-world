"""
HoloLoom Lite - Web Chat UI

A simple web chat interface using FastAPI.

Requirements:
    pip install fastapi uvicorn

Usage:
    python -m HoloLoom.lite web
    # Then open http://localhost:8080

Features:
    - Chat bubble interface
    - Streaming responses via SSE
    - Session management
    - Memory inspection endpoint

Date: December 2025
"""

import asyncio
from typing import Optional
import json


# Global loom instance for the web server
_loom = None


def start_web(host: str = "0.0.0.0", port: int = 8080, **kwargs) -> None:
    """Start the web chat server."""
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        print("Web chat requires 'fastapi' and 'uvicorn' packages.")
        print("Install with: pip install fastapi uvicorn")
        return

    print(f"\nStarting HoloLoom Lite Web Chat on http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def create_app():
    """Create the FastAPI application."""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="HoloLoom Lite",
        description="Simple chat interface for HoloLoom Lite",
        version="1.0.0"
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        """Initialize HoloLoom on startup."""
        global _loom
        from hololoom.lite import HoloLoomLite
        _loom = HoloLoomLite()
        await _loom._initialize()

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup on shutdown."""
        global _loom
        if _loom is not None:
            await _loom.__aexit__(None, None, None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the chat interface."""
        return get_chat_html()

    @app.post("/query")
    async def query(request: Request):
        """Handle chat query."""
        global _loom
        if _loom is None:
            return JSONResponse(
                {"error": "HoloLoom not initialized"},
                status_code=500
            )

        data = await request.json()
        text = data.get("text", "")
        mode = data.get("mode", "direct")

        if not text:
            return JSONResponse(
                {"error": "No text provided"},
                status_code=400
            )

        try:
            # Store experience
            await _loom.experience(text, context={"type": "query", "source": "web"})

            # Reason
            result = await _loom.reason(text, mode=mode)

            return JSONResponse({
                "response": result.text,
                "confidence": result.confidence,
                "sources": result.sources,
                "metadata": result.metadata
            })

        except Exception as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=500
            )

    @app.get("/memories")
    async def get_memories():
        """Get recent memories."""
        global _loom
        if _loom is None:
            return JSONResponse(
                {"error": "HoloLoom not initialized"},
                status_code=500
            )

        try:
            memories = await _loom.recall("", limit=20)
            return JSONResponse({
                "memories": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "timestamp": str(m.timestamp) if hasattr(m, 'timestamp') else None
                    }
                    for m in memories
                ]
            })
        except Exception as e:
            return JSONResponse(
                {"error": str(e)},
                status_code=500
            )

    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        global _loom
        if _loom is None:
            return JSONResponse(
                {"error": "HoloLoom not initialized"},
                status_code=500
            )

        return JSONResponse(_loom.get_metrics())

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        global _loom
        return JSONResponse({
            "status": "healthy" if _loom is not None else "initializing",
            "initialized": _loom._initialized if _loom else False
        })

    return app


def get_chat_html() -> str:
    """Return the chat interface HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HoloLoom Lite</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 20px;
            text-align: center;
            background: rgba(255,255,255,0.05);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #4fc3f7;
        }
        .header p {
            font-size: 14px;
            color: #888;
            margin-top: 5px;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
        }
        .message.user {
            align-self: flex-end;
            background: #4fc3f7;
            color: #1a1a2e;
            border-bottom-right-radius: 4px;
        }
        .message.bot {
            align-self: flex-start;
            background: rgba(255,255,255,0.1);
            border-bottom-left-radius: 4px;
        }
        .message .confidence {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 8px;
        }
        .input-container {
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-top: 1px solid rgba(255,255,255,0.1);
            display: flex;
            gap: 12px;
        }
        .input-container input {
            flex: 1;
            padding: 14px 18px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 24px;
            background: rgba(255,255,255,0.05);
            color: #e0e0e0;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        .input-container input:focus {
            border-color: #4fc3f7;
        }
        .input-container input::placeholder {
            color: #666;
        }
        .input-container button {
            padding: 14px 24px;
            background: #4fc3f7;
            color: #1a1a2e;
            border: none;
            border-radius: 24px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .input-container button:hover {
            background: #81d4fa;
        }
        .input-container button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .mode-selector {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
            justify-content: center;
        }
        .mode-btn {
            padding: 6px 12px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            background: transparent;
            color: #888;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .mode-btn.active {
            background: #4fc3f7;
            color: #1a1a2e;
            border-color: #4fc3f7;
        }
        .mode-btn:hover:not(.active) {
            border-color: #4fc3f7;
            color: #4fc3f7;
        }
        .loading {
            display: flex;
            gap: 4px;
            align-items: center;
        }
        .loading span {
            width: 8px;
            height: 8px;
            background: #4fc3f7;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        .loading span:nth-child(2) { animation-delay: 0.2s; }
        .loading span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HoloLoom Lite</h1>
        <p>Simplified AI with memory</p>
        <div class="mode-selector">
            <button class="mode-btn active" data-mode="direct">Direct</button>
            <button class="mode-btn" data-mode="verify">Verify</button>
            <button class="mode-btn" data-mode="research">Research</button>
        </div>
    </div>

    <div class="chat-container" id="chat"></div>

    <div class="input-container">
        <input type="text" id="input" placeholder="Ask anything..." autocomplete="off">
        <button id="send">Send</button>
    </div>

    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const sendBtn = document.getElementById('send');
        let currentMode = 'direct';

        // Mode selector
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentMode = btn.dataset.mode;
            });
        });

        function addMessage(text, isUser, confidence = null) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'bot'}`;
            div.textContent = text;
            if (confidence !== null && !isUser) {
                const conf = document.createElement('div');
                conf.className = 'confidence';
                conf.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
                div.appendChild(conf);
            }
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        function addLoading() {
            const div = document.createElement('div');
            div.className = 'message bot loading';
            div.innerHTML = '<span></span><span></span><span></span>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function send() {
            const text = input.value.trim();
            if (!text) return;

            input.value = '';
            addMessage(text, true);
            sendBtn.disabled = true;

            const loading = addLoading();

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, mode: currentMode })
                });

                const data = await response.json();
                loading.remove();

                if (data.error) {
                    addMessage('Error: ' + data.error, false);
                } else {
                    addMessage(data.response, false, data.confidence);
                }
            } catch (e) {
                loading.remove();
                addMessage('Error: ' + e.message, false);
            }

            sendBtn.disabled = false;
            input.focus();
        }

        sendBtn.addEventListener('click', send);
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') send();
        });

        // Focus input on load
        input.focus();

        // Welcome message
        addMessage('Hello! I am HoloLoom Lite. Ask me anything, and I will remember our conversation.', false);
    </script>
</body>
</html>'''


if __name__ == "__main__":
    start_web()
