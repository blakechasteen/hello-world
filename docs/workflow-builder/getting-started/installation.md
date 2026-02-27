# Installation Guide

This guide covers setting up the HoloLoom Workflow Builder for development and production use.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Backend executor |
| Node.js | 18+ | Optional: for development tools |
| Modern Browser | Chrome 80+, Firefox 78+, Safari 14+, Edge 80+ | UI rendering |

### Python Dependencies

```bash
pip install fastapi uvicorn websockets networkx
```

### Optional Dependencies

```bash
# For full HoloLoom integration
pip install torch numpy sentence-transformers spacy networkx ollama

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/mythRL.git
cd mythRL
```

### 2. Start the Backend

```bash
cd hololoom/web_dashboard
python workflow_executor.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### 3. Open the UI

Open `hololoom/web_dashboard/workflow_builder.html` in your browser.

**Option A**: File URL
```
file:///path/to/mythRL/hololoom/web_dashboard/workflow_builder.html
```

**Option B**: Serve with Python
```bash
cd hololoom/web_dashboard
python -m http.server 8080
# Open http://localhost:8080/workflow_builder.html
```

**Option C**: Live Server (VS Code)
- Install "Live Server" extension
- Right-click `workflow_builder.html` → "Open with Live Server"

## Configuration

### Backend Configuration

The workflow executor uses environment variables for configuration:

```bash
# Server settings
export WORKFLOW_HOST=0.0.0.0
export WORKFLOW_PORT=8001

# HoloLoom integration
export HOLOLOOM_CONFIG=fused  # bare, fast, fused
export MEMORY_BACKEND=hybrid   # inmemory, hybrid, hyperspace

# Performance
export MAX_CONCURRENT_WORKFLOWS=10
export EXECUTION_TIMEOUT=120
```

### Frontend Configuration

Edit `workflow_builder.html` or use the settings panel:

```javascript
// API endpoint
const API_URL = 'http://localhost:8001';

// WebSocket endpoint
const WS_URL = 'ws://localhost:8001/ws';

// Default theme
const DEFAULT_THEME = 'light'; // or 'dark'
```

## Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY HoloLoom /app/HoloLoom

EXPOSE 8001
CMD ["uvicorn", "hololoom.apps.workflow_builder.workflow_executor:app", "--host", "0.0.0.0", "--port", "8001"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  workflow-builder:
    build: .
    ports:
      - "8001:8001"
    environment:
      - HOLOLOOM_CONFIG=fused
      - MEMORY_BACKEND=hybrid
    depends_on:
      - neo4j
      - qdrant

  neo4j:
    image: neo4j:5.12
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/hololoom123

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name workflows.example.com;

    # Static files
    location / {
        root /var/www/workflow-builder;
        index workflow_builder.html;
        try_files $uri $uri/ =404;
    }

    # API proxy
    location /api/ {
        proxy_pass http://localhost:8001/api/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://localhost:8001/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Verification

### Test Backend Connection

```bash
curl http://localhost:8001/health
# Expected: {"status": "healthy", "version": "1.0"}
```

### Test WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');
ws.onopen = () => console.log('Connected!');
ws.onmessage = (e) => console.log('Message:', e.data);
```

### Test Workflow Execution

```bash
curl -X POST http://localhost:8001/api/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "version": "1.0",
      "name": "Test",
      "nodes": [{
        "id": "n1",
        "type": "hololoom_query",
        "config": {"query": "What is Thompson Sampling?"}
      }],
      "connections": []
    },
    "input_data": {}
  }'
```

## Troubleshooting

### Common Issues

**Issue**: Backend won't start
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies
```bash
pip install fastapi uvicorn websockets
```

---

**Issue**: WebSocket connection fails
```
WebSocket connection to 'ws://localhost:8001/ws' failed
```
**Solution**: Ensure backend is running and check CORS settings

---

**Issue**: Workflow execution timeout
```
{"error": "Execution timeout after 120 seconds"}
```
**Solution**: Increase timeout or optimize workflow
```bash
export EXECUTION_TIMEOUT=300
```

---

**Issue**: Memory backend unavailable
```
WARNING: Falling back to INMEMORY backend
```
**Solution**: Start Docker services
```bash
docker-compose up -d neo4j qdrant
```

### Logs

Backend logs:
```bash
tail -f /var/log/workflow-executor.log
```

Browser console:
- Press F12 → Console tab
- Look for WebSocket and API errors

## Next Steps

- [Your First Workflow](first-workflow.md) - Create your first workflow
- [UI Overview](ui-overview.md) - Learn the interface

---

← [Back to Overview](../README.md) | [Your First Workflow](first-workflow.md) →
