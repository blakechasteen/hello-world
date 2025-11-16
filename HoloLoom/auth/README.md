# HoloLoom Authentication System

**Status**: Production Ready (v1.0.0)
**Created**: 2025-11-16
**Philosophy**: "Safe by default, opt-in by configuration"

Complete dual authentication system for HoloLoom dashboard API:
- **JWT (JSON Web Tokens)** for user sessions
- **API Keys** for programmatic access

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [API Endpoints](#api-endpoints)
5. [Usage Examples](#usage-examples)
6. [Production Deployment](#production-deployment)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install python-jose[cryptography] bcrypt passlib python-multipart
```

### 2. Configure Environment

Create `.env` file:

```bash
# JWT Configuration
JWT_SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60
REFRESH_EXPIRY_DAYS=7

# API Key Configuration
API_KEY_PREFIX=hololoom

# Auth Configuration
ENABLE_AUTH=true  # Set to false to disable authentication
```

### 3. Start Dashboard with Auth

```bash
# Enable auth
export ENABLE_AUTH=true

# Start server
uvicorn HoloLoom.dashboard_server:app --reload --port 8000
```

### 4. Login

```bash
# Get JWT token
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
```

### 5. Access Protected Endpoints

```bash
# Use JWT token
curl -X GET http://localhost:8000/api/v1/analytics/summary \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                 HoloLoom Dashboard API                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │   Public     │    │  Protected   │                 │
│  │  Endpoints   │    │  Endpoints   │                 │
│  │              │    │              │                 │
│  │ - Dashboard  │    │ - Analytics  │                 │
│  │ - Login      │    │ - Skills     │                 │
│  │ - Refresh    │    │ - Executions │                 │
│  └──────────────┘    └───────┬──────┘                 │
│                              │                         │
│                    ┌─────────▼─────────┐               │
│                    │  Auth Middleware  │               │
│                    │  (Depends)        │               │
│                    └─────────┬─────────┘               │
│                              │                         │
│              ┌───────────────┼───────────────┐         │
│              │               │               │         │
│      ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼──────┐  │
│      │ JWT Verify   │ │ API Key   │ │ Role Check   │  │
│      │ (access/     │ │ Verify    │ │ (admin/user) │  │
│      │  refresh)    │ │           │ │              │  │
│      └───────┬──────┘ └─────┬─────┘ └───────┬──────┘  │
│              │              │               │         │
│              └──────────────┼───────────────┘         │
│                             │                         │
│                    ┌────────▼────────┐                │
│                    │   User Store    │                │
│                    │  (in-memory)    │                │
│                    └─────────────────┘                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Authentication Flow

**JWT Login Flow**:
```
1. Client sends username + password → POST /api/v1/auth/login
2. Server verifies credentials → users.authenticate_user()
3. Server generates JWT tokens → authentication.create_access_token()
4. Client receives access_token + refresh_token
5. Client stores tokens (localStorage, secure cookie, etc.)
6. Client includes token in requests → Authorization: Bearer {token}
7. Server verifies token → middleware.get_current_user()
8. Request processed with user context
```

**API Key Flow**:
```
1. User logs in with JWT → POST /api/v1/auth/login
2. User generates API key → POST /api/v1/auth/api-keys
3. Server creates API key → api_keys.generate_api_key()
4. Client stores API key securely
5. Client includes key in requests → Authorization: Bearer {api_key}
6. Server verifies API key → middleware.get_current_user()
7. Request processed with user context
```

### Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `authentication.py` | 220 | JWT generation/verification, token blacklist |
| `users.py` | 250 | User management, password hashing |
| `api_keys.py` | 280 | API key generation, storage, verification |
| `middleware.py` | 150 | FastAPI dependency injection |

**Total**: ~900 lines of production-ready code

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUTH` | `false` | Enable authentication (opt-in) |
| `JWT_SECRET_KEY` | `dev-secret-...` | JWT signing key (CHANGE IN PRODUCTION!) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm (HS256/RS256) |
| `JWT_EXPIRY_MINUTES` | `60` | Access token expiry (minutes) |
| `REFRESH_EXPIRY_DAYS` | `7` | Refresh token expiry (days) |
| `API_KEY_PREFIX` | `hololoom` | API key prefix |

### Example `.env` File

```bash
# Production configuration
ENABLE_AUTH=true
JWT_SECRET_KEY=your-256-bit-secret-key-here-change-this
JWT_ALGORITHM=HS256
JWT_EXPIRY_MINUTES=60
REFRESH_EXPIRY_DAYS=7
API_KEY_PREFIX=hololoom_prod
```

### Generating Secret Key

```python
import secrets
print(secrets.token_urlsafe(32))
# Output: Vx9YF8kL2mN3pQ4rS5tU6vW7xY8zA1bC2dE3fG4hI5jK
```

Or use command line:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## API Endpoints

### Authentication Endpoints

#### POST `/api/v1/auth/login`

Login with username and password.

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Error** (401 Unauthorized):
```json
{
  "detail": "Invalid username or password"
}
```

**Rate Limit**: 10 requests/minute

---

#### POST `/api/v1/auth/refresh`

Refresh access token using refresh token.

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Authorization: Bearer {refresh_token}"
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Rate Limit**: 10 requests/minute

---

#### POST `/api/v1/auth/logout`

Logout (revoke token).

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/auth/logout \
  -H "Authorization: Bearer {access_token}"
```

**Response** (200 OK):
```json
{
  "message": "Logged out successfully"
}
```

---

### API Key Endpoints

#### POST `/api/v1/auth/api-keys`

Generate new API key (requires JWT).

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer {access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "key_type": "live",
    "expires_in_days": 30
  }'
```

**Response** (201 Created):
```json
{
  "key_id": "ak_1a2b3c4d",
  "key": "hololoom_live_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
  "username": "admin",
  "key_type": "live",
  "created_at": "2025-11-16T12:00:00Z",
  "expires_at": "2025-12-16T12:00:00Z",
  "last_used": null,
  "active": true
}
```

**Note**: Full key is only shown once on creation. Store it securely!

---

#### GET `/api/v1/auth/api-keys`

List user's API keys.

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer {access_token}"
```

**Response** (200 OK):
```json
{
  "api_keys": [
    {
      "key_id": "ak_1a2b3c4d",
      "key_prefix": "hololoom_live_a1b2c3...",
      "username": "admin",
      "key_type": "live",
      "created_at": "2025-11-16T12:00:00Z",
      "expires_at": "2025-12-16T12:00:00Z",
      "last_used": "2025-11-16T14:30:00Z",
      "active": true
    }
  ]
}
```

---

#### DELETE `/api/v1/auth/api-keys/{key_id}`

Revoke API key.

**Request**:
```bash
curl -X DELETE http://localhost:8000/api/v1/auth/api-keys/ak_1a2b3c4d \
  -H "Authorization: Bearer {access_token}"
```

**Response** (200 OK):
```json
{
  "message": "API key revoked",
  "key_id": "ak_1a2b3c4d"
}
```

---

## Usage Examples

### Python Client

```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "admin", "password": "admin"}
)
tokens = response.json()

access_token = tokens["access_token"]

# Access protected endpoint
response = requests.get(
    "http://localhost:8000/api/v1/analytics/summary",
    headers={"Authorization": f"Bearer {access_token}"}
)
summary = response.json()

print(f"Total queries: {summary['total_queries']}")
```

### JavaScript (Browser)

```javascript
// Login
const loginResponse = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({ username: 'admin', password: 'admin' })
});

const tokens = await loginResponse.json();
localStorage.setItem('access_token', tokens.access_token);

// Access protected endpoint
const summaryResponse = await fetch('http://localhost:8000/api/v1/analytics/summary', {
  headers: { 'Authorization': `Bearer ${tokens.access_token}` }
});

const summary = await summaryResponse.json();
console.log('Total queries:', summary.total_queries);
```

### cURL

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin" | jq -r '.access_token')

# Access protected endpoint
curl -X GET http://localhost:8000/api/v1/analytics/summary \
  -H "Authorization: Bearer $TOKEN"
```

---

## Production Deployment

### 1. Database Migration

Replace in-memory storage with PostgreSQL/SQLite.

**User Table** (`users`):
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash BYTEA NOT NULL,
    role VARCHAR(50) NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**API Key Table** (`api_keys`):
```sql
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    key_id VARCHAR(50) UNIQUE NOT NULL,
    key_hash BYTEA NOT NULL,  -- Store hashed key
    username VARCHAR(255) REFERENCES users(username),
    key_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    last_used TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_username ON api_keys(username);
```

**Update `users.py`**:
```python
import asyncpg

async def authenticate_user(username: str, password: str) -> Optional[User]:
    async with asyncpg.create_pool(DATABASE_URL) as pool:
        row = await pool.fetchrow(
            "SELECT * FROM users WHERE username = $1 AND active = TRUE",
            username
        )

        if not row:
            return None

        user = User(**dict(row))
        if not user.verify_password(password):
            return None

        return user
```

### 2. Token Blacklist with Redis

Replace in-memory blacklist with Redis (TTL = token expiry).

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def revoke_token(token: str, expires_in: int = 3600) -> None:
    """Add token to Redis blacklist with TTL."""
    redis_client.setex(f"blacklist:{token}", expires_in, "1")

def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted."""
    return redis_client.exists(f"blacklist:{token}") > 0
```

### 3. HTTPS Configuration

**Nginx Reverse Proxy**:
```nginx
server {
    listen 443 ssl http2;
    server_name dashboard.hololoom.ai;

    ssl_certificate /etc/letsencrypt/live/dashboard.hololoom.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/dashboard.hololoom.ai/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://127.0.0.1:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 4. Secret Rotation

**Rotate JWT secret without downtime**:

```python
# Support multiple secrets (old + new)
JWT_SECRETS = [
    os.getenv("JWT_SECRET_KEY_NEW"),  # Try new first
    os.getenv("JWT_SECRET_KEY_OLD"),  # Fallback to old
]

def verify_token_multi_secret(token: str) -> Optional[TokenData]:
    """Try verifying with multiple secrets."""
    for secret in JWT_SECRETS:
        try:
            payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
            return TokenData(**payload)
        except JWTError:
            continue

    return None
```

**Rotation procedure**:
1. Add new secret to `JWT_SECRET_KEY_NEW`
2. Deploy with multi-secret support
3. Wait for old tokens to expire (JWT_EXPIRY_MINUTES)
4. Remove old secret from `JWT_SECRET_KEY_OLD`

### 5. Monitoring

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram

auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['status'])
auth_latency = Histogram('auth_latency_seconds', 'Authentication latency')

@auth_latency.time()
async def authenticate_user(username: str, password: str) -> Optional[User]:
    user = # ... authentication logic

    if user:
        auth_attempts.labels(status='success').inc()
    else:
        auth_attempts.labels(status='failure').inc()

    return user
```

---

## Security Best Practices

### 1. Password Requirements

Enforce strong passwords:

```python
import re

def validate_password(password: str) -> bool:
    """
    Validate password strength.

    Requirements:
    - At least 12 characters
    - Contains uppercase, lowercase, digit, special char
    """
    if len(password) < 12:
        return False

    if not re.search(r'[A-Z]', password):
        return False

    if not re.search(r'[a-z]', password):
        return False

    if not re.search(r'\d', password):
        return False

    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False

    return True
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/auth/login")
@limiter.limit("10/minute")  # Max 10 login attempts per minute
async def login(request: Request, form_data: OAuth2PasswordRequestForm):
    # ... login logic
```

### 3. Token Security

- **Never log tokens** - Redact in logs
- **Short expiry** - 1 hour for access, 7 days for refresh
- **Use HTTPS only** - Never send tokens over HTTP
- **Secure storage** - Use httpOnly cookies or secure storage
- **Token rotation** - Refresh tokens regularly

### 4. API Key Security

- **Hash keys in database** - Store SHA256 hash, not plain text
- **Prefix for identification** - `hololoom_live_`, `hololoom_test_`
- **Last used tracking** - Monitor for suspicious activity
- **Expiry support** - Optional expiry dates
- **Revocation** - Immediate revocation capability

### 5. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dashboard.hololoom.ai"],  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

---

## Troubleshooting

### Issue: "JWT not available" Error

**Cause**: `python-jose` not installed

**Solution**:
```bash
pip install python-jose[cryptography]
```

### Issue: "bcrypt not available" Warning

**Cause**: `bcrypt` not installed (insecure fallback used)

**Solution**:
```bash
pip install bcrypt
```

### Issue: 401 Unauthorized on Protected Endpoints

**Cause**: Token expired or invalid

**Solution**:
1. Check token expiry in JWT payload
2. Refresh token using `/api/v1/auth/refresh`
3. Re-login if refresh token also expired

### Issue: WebSocket Connection Rejected

**Cause**: Token not included in WebSocket connection

**Solution**:
```javascript
// Include token in query parameter
const token = localStorage.getItem('access_token');
const ws = new WebSocket(`ws://localhost:8000/ws?token=${token}`);
```

### Issue: API Key Not Working

**Possible causes**:
1. Key revoked - Check `active` status
2. Key expired - Check `expires_at` timestamp
3. Wrong prefix - Must be `hololoom_live_` or `hololoom_test_`

**Solution**:
```bash
# List keys to check status
curl -X GET http://localhost:8000/api/v1/auth/api-keys \
  -H "Authorization: Bearer {jwt_token}"
```

### Issue: Auth Disabled but Still Getting 401

**Cause**: `ENABLE_AUTH` not set to `false`

**Solution**:
```bash
export ENABLE_AUTH=false
uvicorn HoloLoom.dashboard_server:app --reload
```

---

## Default Users

For development/testing only:

| Username | Password | Role |
|----------|----------|------|
| `admin` | `admin` | admin |
| `demo` | `demo` | user |

**IMPORTANT**: Change these credentials in production!

---

## API Reference

See [API_REFERENCE.md](API_REFERENCE.md) for complete API documentation (auto-generated from FastAPI).

---

## License

Same as HoloLoom main project.

---

## Support

For issues or questions:
- GitHub Issues: [HoloLoom Issues](https://github.com/your-repo/issues)
- Documentation: [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)

---

**Last Updated**: 2025-11-16
**Version**: 1.0.0
**Status**: Production Ready
