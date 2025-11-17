# Backend Integration Guide

**Last Updated:** 2025-11-17
**For:** Live Demo WebSocket Integration

---

## Overview

The BigPlay visualizations work **completely offline** in demo mode, but can be connected to a real FastAPI backend for:
- Live NPC conversations via LLM
- Real-time emotion updates
- Actual latency metrics
- Production-quality responses

This guide shows how to integrate the visualizations with your BigPlay backend.

---

## Architecture

```
┌─────────────────┐      WebSocket (ws://)      ┌──────────────────┐
│                 │ ◄──────────────────────────► │                  │
│  live-demo.html │                              │  FastAPI Server  │
│  (Frontend)     │      JSON Messages           │  (Backend)       │
│                 │ ◄──────────────────────────► │                  │
└─────────────────┘                              └──────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  LLM Provider    │
                                                  │  (OpenAI, etc.)  │
                                                  └──────────────────┘
```

---

## Message Protocol

### Client → Server Messages

#### 1. NPC Talk Request
```json
{
  "type": "npc_talk",
  "text": "Hello, what's your name?",
  "npc_id": "alice",
  "player_id": "player_001"
}
```

#### 2. Emotion Update Request
```json
{
  "type": "update_emotion",
  "npc_id": "alice",
  "emotion": {
    "pleasure": 0.7,
    "arousal": 0.5,
    "dominance": 0.6,
    "trust": 0.8
  }
}
```

#### 3. Heartbeat (Keep-Alive)
```json
{
  "type": "ping"
}
```

### Server → Client Messages

#### 1. NPC Response
```json
{
  "type": "npc_response",
  "text": "Hello! I'm Alice, the innkeeper. Welcome to my tavern!",
  "emotion": {
    "pleasure": 0.8,
    "arousal": 0.4,
    "dominance": 0.5,
    "trust": 0.7
  },
  "latency": 150,
  "timestamp": 1700000000000,
  "npc_id": "alice"
}
```

#### 2. Welcome Message
```json
{
  "type": "welcome",
  "message": "Connected to BigPlay Engine",
  "server_version": "1.0.0",
  "available_npcs": ["alice", "bob", "merrick"]
}
```

#### 3. Error Message
```json
{
  "type": "error",
  "error": "NPC not found",
  "code": 404
}
```

#### 4. Pong (Heartbeat Response)
```json
{
  "type": "pong",
  "timestamp": 1700000000000
}
```

---

## FastAPI Backend Example

### Minimal WebSocket Server

```python
# main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import time
from typing import Dict

app = FastAPI(title="BigPlay WebSocket Server")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Generate unique connection ID
    connection_id = str(time.time())
    active_connections[connection_id] = websocket

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to BigPlay Engine",
            "server_version": "1.0.0",
            "available_npcs": ["alice", "bob", "merrick"]
        })

        # Message loop
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message["type"] == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": int(time.time() * 1000)
                })

            elif message["type"] == "npc_talk":
                start_time = time.time()

                # Process NPC conversation (integrate with your LLM here)
                response_text = await generate_npc_response(
                    npc_id=message["npc_id"],
                    player_message=message["text"]
                )

                # Calculate latency
                latency = int((time.time() - start_time) * 1000)

                # Send response
                await websocket.send_json({
                    "type": "npc_response",
                    "text": response_text,
                    "emotion": {
                        "pleasure": 0.7,
                        "arousal": 0.5,
                        "dominance": 0.6,
                        "trust": 0.8
                    },
                    "latency": latency,
                    "timestamp": int(time.time() * 1000),
                    "npc_id": message["npc_id"]
                })

    except WebSocketDisconnect:
        # Clean up on disconnect
        del active_connections[connection_id]
        print(f"Client {connection_id} disconnected")

async def generate_npc_response(npc_id: str, player_message: str) -> str:
    """
    Integrate with your LLM provider here.

    Example with OpenAI:
    """
    # TODO: Replace with actual LLM call
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": f"You are {npc_id}, an NPC in a fantasy game."},
    #         {"role": "user", "content": player_message}
    #     ]
    # )
    # return response.choices[0].message.content

    # Demo response
    return f"[NPC {npc_id}] You said: {player_message}"

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_connections": len(active_connections),
        "server_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run the Server

```bash
# Install dependencies
pip install fastapi uvicorn websockets

# Run server
uvicorn main:app --reload --port 8000

# Test health check
curl http://localhost:8000/health
```

---

## Integration with BigPlay Engine

### Full Integration Example

```python
# bigplay_websocket.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from bigplay import BigPlayEngine, NPCConfig
import os
import time

app = FastAPI()
engine = BigPlayEngine(
    api_key=os.getenv("OPENAI_API_KEY"),
    llm_provider="openai"
)

# Create NPCs
npcs = {
    "alice": engine.create_npc(
        name="Alice",
        role="Innkeeper",
        personality="friendly and welcoming"
    ),
    "bob": engine.create_npc(
        name="Bob",
        role="Guard",
        personality="gruff but fair"
    ),
    "merrick": engine.create_npc(
        name="Merrick",
        role="Wizard",
        personality="wise and mysterious"
    )
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to BigPlay Engine",
            "available_npcs": list(npcs.keys())
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "npc_talk":
                start_time = time.time()

                # Get NPC
                npc = npcs.get(message["npc_id"])
                if not npc:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"NPC '{message['npc_id']}' not found",
                        "code": 404
                    })
                    continue

                # Generate response via BigPlay
                response = npc.talk(
                    player_message=message["text"],
                    player_id=message.get("player_id", "player_001")
                )

                latency = int((time.time() - start_time) * 1000)

                # Send response with emotion
                await websocket.send_json({
                    "type": "npc_response",
                    "text": response.text,
                    "emotion": {
                        "pleasure": response.emotion.valence,
                        "arousal": response.emotion.arousal,
                        "dominance": response.emotion.dominance,
                        "trust": response.trust
                    },
                    "latency": latency,
                    "timestamp": int(time.time() * 1000),
                    "npc_id": message["npc_id"]
                })

    except WebSocketDisconnect:
        print("Client disconnected")
```

---

## Frontend Configuration

### Update Connection URL

Edit `live-demo.html` to point to your backend:

```javascript
// Line ~50 in live-demo.html

const connectionModes = {
    demo: 'demo',  // No backend (simulated)
    local: 'ws://localhost:8000/ws',  // Local development
    production: 'wss://api.bigplay.dev/ws'  // Production (update this)
};

// Change default mode from 'demo' to 'local'
let currentMode = 'local';  // Was: 'demo'
```

### Environment-Specific Configuration

For production deployments, use environment variables:

```javascript
// config.js

const WEBSOCKET_URL = process.env.WEBSOCKET_URL || 'ws://localhost:8000/ws';

// In live-demo.html
const wsClient = new BigPlayUI.WebSocketClient({
    url: WEBSOCKET_URL
});
```

---

## Testing

### 1. Test Health Endpoint

```bash
curl http://localhost:8000/health

# Expected:
# {"status":"healthy","active_connections":0,"server_version":"1.0.0"}
```

### 2. Test WebSocket with wscat

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c ws://localhost:8000/ws

# Send message
{"type":"npc_talk","text":"Hello!","npc_id":"alice","player_id":"test"}

# Expected response
{"type":"npc_response","text":"...","emotion":{...},"latency":150}
```

### 3. Test with Browser

1. Open `live-demo.html`
2. Click "Local API" button
3. Send message to NPC
4. Verify response in UI
5. Check browser console (F12) for WebSocket logs

---

## Deployment

### Production Checklist

- [ ] Use WSS (secure WebSocket) in production
- [ ] Configure CORS to allow only your domain
- [ ] Add authentication/authorization
- [ ] Implement rate limiting
- [ ] Add monitoring and logging
- [ ] Use environment variables for secrets
- [ ] Set up auto-reconnect with exponential backoff
- [ ] Handle connection limits (max connections per user)
- [ ] Implement graceful shutdown

### Docker Deployment

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  bigplay-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: unless-stopped
```

### Kubernetes Deployment

```yaml
# deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: bigplay-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bigplay-api
  template:
    metadata:
      labels:
        app: bigplay-api
    spec:
      containers:
      - name: bigplay-api
        image: yourregistry/bigplay-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: bigplay-secrets
              key: openai-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: bigplay-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: bigplay-api
```

---

## Troubleshooting

### Connection Refused

**Problem:** WebSocket connection fails immediately

**Solutions:**
1. Verify server is running: `curl http://localhost:8000/health`
2. Check firewall settings
3. Ensure port 8000 is not in use: `lsof -i :8000`
4. Try different port in both server and frontend

### CORS Errors

**Problem:** Browser blocks WebSocket due to CORS

**Solutions:**
```python
# In FastAPI server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Specify exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### High Latency

**Problem:** Responses take >2 seconds

**Solutions:**
1. Check LLM provider latency
2. Add caching for common queries
3. Use streaming responses
4. Implement connection pooling

### Memory Leaks

**Problem:** Server memory grows over time

**Solutions:**
1. Clean up disconnected WebSockets
2. Implement connection limits
3. Add timeout for idle connections
4. Monitor with `psutil`

---

## Next Steps

1. **Implement authentication:** Add JWT tokens or API keys
2. **Add rate limiting:** Prevent abuse
3. **Implement caching:** Redis for common responses
4. **Add analytics:** Track usage patterns
5. **Scale horizontally:** Multiple server instances with load balancer

---

## Resources

- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [WebSocket Protocol (RFC 6455)](https://tools.ietf.org/html/rfc6455)
- [BigPlay API Reference](../API_REFERENCE.md)

---

**Ready to integrate? Start with the minimal example and expand from there!** 🚀
