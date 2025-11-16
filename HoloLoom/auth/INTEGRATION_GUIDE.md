# HoloLoom Authentication Integration Guide

**Created**: 2025-11-16
**For**: Dashboard developers and system administrators

This guide shows how to integrate the authentication system into the HoloLoom dashboard server.

---

## Quick Integration (5 Minutes)

### Step 1: Install Dependencies

```bash
pip install python-jose[cryptography] bcrypt passlib python-multipart
```

### Step 2: Configure Environment

Create `.env` file (or copy from `.env.example`):

```bash
cp HoloLoom/auth/.env.example .env

# Edit .env:
ENABLE_AUTH=true
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Step 3: Update Dashboard Server

Add to `HoloLoom/dashboard_server.py`:

```python
# At top of file, add import
from HoloLoom.auth.dashboard_integration import (
    register_auth_routes,
    verify_websocket_token,
    is_auth_enabled,
)

# After app initialization (line ~145), add:
# Register authentication routes
register_auth_routes(app, limiter)

# In WebSocket endpoint (around line 217), add token verification:
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """WebSocket endpoint with optional authentication."""
    # Verify token if auth is enabled
    user = await verify_websocket_token(websocket, token)

    # Continue with existing WebSocket logic...
    await manager.connect(websocket)
    # ... rest of WebSocket handler
```

### Step 4: Protect Endpoints (Optional)

Add authentication to protected endpoints:

```python
from HoloLoom.auth.middleware import get_current_user
from HoloLoom.auth.users import User

# Before (public):
@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(request: Request):
    summary = analytics.get_summary()
    return summary

# After (protected):
@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(
    request: Request,
    user: User = Depends(get_current_user)  # Add this
):
    summary = analytics.get_summary()
    return summary
```

### Step 5: Test Authentication

```bash
# Start dashboard
ENABLE_AUTH=true uvicorn HoloLoom.dashboard_server:app --reload --port 8000

# Test login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Test protected endpoint
TOKEN="<access_token_from_above>"
curl -X GET http://localhost:8000/api/v1/analytics/summary \
  -H "Authorization: Bearer $TOKEN"
```

---

## Complete Integration Example

Here's a complete example of integrating auth into the dashboard server:

### Modified `dashboard_server.py`

```python
#!/usr/bin/env python3
"""
HoloLoom Promptly Real-Time Dashboard Server
============================================

Features:
- WebSocket for real-time updates
- REST API (v1) with rate limiting
- **Authentication (opt-in via ENABLE_AUTH env var)**
- FastAPI backend
"""

import os
import asyncio
import json
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ... existing imports ...

# Auth imports
from HoloLoom.auth.dashboard_integration import (
    register_auth_routes,
    verify_websocket_token,
    is_auth_enabled,
    get_auth_status,
)
from HoloLoom.auth.middleware import get_current_user
from HoloLoom.auth.users import User

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HoloLoom Promptly Dashboard",
    version="1.0.0",
    description="Real-time dashboard with authentication"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
# ... existing rate limiter code ...

# Register authentication routes
register_auth_routes(app, limiter)

# Log auth status
auth_status = get_auth_status()
logger.info(f"Authentication: enabled={auth_status['enabled']}, available={auth_status['available']}")

# ============================================================================
# Public Endpoints (no auth required)
# ============================================================================

@app.get("/")
async def get_dashboard():
    """Serve dashboard HTML (public)."""
    dashboard_html = Path(__file__).parent / "dashboard.html"
    if dashboard_html.exists():
        return FileResponse(dashboard_html)
    else:
        return HTMLResponse(content=get_embedded_dashboard_html())


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint (public)."""
    return {
        "status": "healthy",
        "auth_enabled": is_auth_enabled(),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Protected Endpoints (auth required if ENABLE_AUTH=true)
# ============================================================================

# Helper function for optional auth
async def get_user_if_auth_enabled(
    user: Optional[User] = Depends(get_current_user) if is_auth_enabled() else None
) -> Optional[User]:
    """Get user if auth enabled, otherwise None."""
    return user


@app.get("/api/v1/analytics/summary")
@limiter.limit("100/minute")
async def get_analytics_summary(
    request: Request,
    user: User = Depends(get_current_user) if is_auth_enabled() else None
):
    """Get analytics summary (protected if auth enabled)."""
    summary = analytics.get_summary()
    return summary


@app.get("/api/v1/analytics/trends")
@limiter.limit("100/minute")
async def get_analytics_trends(
    request: Request,
    days: int = 7,
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(get_current_user) if is_auth_enabled() else None
):
    """Get quality trends (protected if auth enabled)."""
    # ... existing implementation ...


# ============================================================================
# WebSocket Endpoint with Authentication
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time updates.

    Authentication:
    - If ENABLE_AUTH=true, requires token in query parameter
    - Connect with: ws://localhost:8000/ws?token={jwt_token}
    """
    # Verify token if auth is enabled
    user = await verify_websocket_token(websocket, token)

    # If auth enabled but verification failed, connection already closed
    if is_auth_enabled() and not user:
        return

    # Continue with existing WebSocket logic
    await manager.connect(websocket)

    try:
        # Send initial data
        await send_initial_data(websocket)

        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "request_update":
                await send_analytics_update(websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Protecting Specific Endpoints

### Pattern 1: Require Authentication (if enabled)

```python
from HoloLoom.auth.middleware import get_current_user
from HoloLoom.auth.users import User
from HoloLoom.auth.dashboard_integration import is_auth_enabled

@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(
    user: User = Depends(get_current_user) if is_auth_enabled() else None
):
    """Protected endpoint (if auth enabled)."""
    summary = analytics.get_summary()
    return summary
```

### Pattern 2: Require Admin Role

```python
from HoloLoom.auth.middleware import require_role
from HoloLoom.auth.users import UserRole

@app.delete("/api/v1/admin/reset")
async def admin_reset(
    user: User = Depends(require_role(UserRole.ADMIN))
):
    """Admin-only endpoint."""
    # Only admins can access this
    return {"message": "System reset"}
```

### Pattern 3: Optional Authentication

```python
from HoloLoom.auth.middleware import get_optional_user

@app.get("/api/v1/analytics/public")
async def public_analytics(
    user: Optional[User] = Depends(get_optional_user)
):
    """Public endpoint with optional auth."""
    if user:
        # Return personalized data
        return {"message": f"Hello {user.username}!", "data": {...}}
    else:
        # Return public data
        return {"message": "Hello guest!", "data": {...}}
```

---

## Frontend Integration

### JavaScript (Browser)

```javascript
// Login function
async function login(username, password) {
  const response = await fetch('http://localhost:8000/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ username, password })
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const tokens = await response.json();

  // Store tokens in localStorage
  localStorage.setItem('access_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);

  return tokens;
}

// API request function
async function fetchProtected(endpoint) {
  const token = localStorage.getItem('access_token');

  const response = await fetch(`http://localhost:8000${endpoint}`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });

  if (response.status === 401) {
    // Token expired, try refresh
    await refreshToken();
    return fetchProtected(endpoint);  // Retry
  }

  return response.json();
}

// Refresh token function
async function refreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');

  const response = await fetch('http://localhost:8000/api/v1/auth/refresh', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${refreshToken}`
    }
  });

  if (!response.ok) {
    // Refresh failed, redirect to login
    window.location.href = '/login';
    return;
  }

  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
}

// WebSocket connection
function connectWebSocket() {
  const token = localStorage.getItem('access_token');
  const ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);

  ws.onopen = () => console.log('WebSocket connected');
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
  };

  return ws;
}

// Usage
(async () => {
  // Login
  await login('admin', 'admin');

  // Fetch protected data
  const summary = await fetchProtected('/api/v1/analytics/summary');
  console.log('Summary:', summary);

  // Connect WebSocket
  const ws = connectWebSocket();
})();
```

### Python Client

```python
import requests

class HoloLoomClient:
    """Python client for HoloLoom dashboard API."""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None

    def login(self, username: str, password: str):
        """Login and store tokens."""
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()

        tokens = response.json()
        self.access_token = tokens["access_token"]
        self.refresh_token = tokens["refresh_token"]

    def get(self, endpoint: str):
        """Make authenticated GET request."""
        response = requests.get(
            f"{self.base_url}{endpoint}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )

        if response.status_code == 401:
            # Try refresh
            self._refresh_token()
            return self.get(endpoint)  # Retry

        response.raise_for_status()
        return response.json()

    def _refresh_token(self):
        """Refresh access token."""
        response = requests.post(
            f"{self.base_url}/api/v1/auth/refresh",
            headers={"Authorization": f"Bearer {self.refresh_token}"}
        )
        response.raise_for_status()

        data = response.json()
        self.access_token = data["access_token"]

# Usage
client = HoloLoomClient()
client.login("admin", "admin")

summary = client.get("/api/v1/analytics/summary")
print(f"Total queries: {summary['total_queries']}")
```

---

## Testing

### Test Suite

```python
import pytest
from fastapi.testclient import TestClient
from HoloLoom.dashboard_server import app

client = TestClient(app)

def test_login_success():
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "admin"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data

def test_login_failure():
    """Test failed login with invalid credentials."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "wrong"}
    )

    assert response.status_code == 401

def test_protected_endpoint_without_auth():
    """Test protected endpoint without token."""
    response = client.get("/api/v1/analytics/summary")

    assert response.status_code == 401

def test_protected_endpoint_with_auth():
    """Test protected endpoint with valid token."""
    # Login first
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "admin"}
    )
    token = login_response.json()["access_token"]

    # Access protected endpoint
    response = client.get(
        "/api/v1/analytics/summary",
        headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200

def test_api_key_creation():
    """Test API key generation."""
    # Login first
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "admin"}
    )
    token = login_response.json()["access_token"]

    # Create API key
    response = client.post(
        "/api/v1/auth/api-keys",
        headers={"Authorization": f"Bearer {token}"},
        json={"key_type": "live", "expires_in_days": 30}
    )

    assert response.status_code == 201
    data = response.json()
    assert "key" in data
    assert data["key"].startswith("hololoom_live_")

# Run tests with: pytest HoloLoom/auth/tests/
```

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'jose'`

**Solution**:
```bash
pip install python-jose[cryptography]
```

---

**Issue**: Protected endpoints still accessible without token

**Solution**: Check that `ENABLE_AUTH=true` and restart server

---

**Issue**: WebSocket connection rejected

**Solution**: Include token in query parameter:
```javascript
const token = localStorage.getItem('access_token');
const ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);
```

---

## Migration Path

### From No Auth → Auth Enabled

1. Install dependencies
2. Set `ENABLE_AUTH=false` (keep existing behavior)
3. Test that everything still works
4. Add `register_auth_routes(app, limiter)` to dashboard server
5. Test auth endpoints work (login, API keys)
6. Set `ENABLE_AUTH=true` and restart
7. Update frontend to include tokens
8. Test all flows

### From In-Memory → PostgreSQL

See [README.md#production-deployment](README.md#production-deployment) for complete migration guide.

---

## Support

For questions or issues:
- See complete docs: [README.md](README.md)
- GitHub Issues: Create an issue with `[auth]` prefix

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
