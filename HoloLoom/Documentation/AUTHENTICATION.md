# HoloLoom Authentication System

Secure JWT-based authentication for the HoloLoom web interface.

## Features

✅ JWT (JSON Web Token) authentication
✅ Bcrypt password hashing
✅ Session management
✅ Protected WebSocket connections
✅ Demo user accounts for testing
✅ Simple login/logout flow

## Quick Start

### 1. Install Dependencies

```bash
pip install pyjwt passlib[bcrypt] python-multipart
```

### 2. Start the Server

```bash
cd HoloLoom/web
python app.py
```

### 3. Access the Application

Open browser to: http://localhost:8000

**You'll be redirected to the login page** if not authenticated.

### 4. Demo Credentials

```
Username: admin
Password: admin123

Username: demo
Password: demo123
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    HoloLoom Authentication                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Login Page  │───▶│  JWT Tokens  │───▶│  Chat Page   │  │
│  │ (credentials)│    │  (bearer)    │    │  (protected) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              User Database (in-memory)              │  │
│  │  • Password hashing (bcrypt)                        │  │
│  │  • User profiles                                    │  │
│  │  • Session management                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Authentication Flow

1. **User visits `/`** → Redirected to `/login` (if no token)
2. **User submits credentials** → Server validates and returns JWT
3. **Client stores JWT** → Saved in `localStorage`
4. **Client accesses chat** → Sends JWT in Authorization header
5. **WebSocket connection** → JWT passed as query parameter
6. **Server validates JWT** → Every request/WebSocket message

## API Endpoints

### Authentication

#### `POST /api/auth/login_json`

Login with JSON credentials.

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "username": "admin",
  "session_id": "admin_1234567890"
}
```

#### `POST /api/auth/logout`

Logout current user (requires authentication).

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "logged_out",
  "username": "admin"
}
```

#### `GET /api/auth/me`

Get current user information (requires authentication).

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "username": "admin",
  "email": "admin@hololoom.ai",
  "full_name": "Admin User",
  "disabled": false,
  "created_at": "2025-01-15T10:30:00Z"
}
```

### Protected WebSocket

#### `ws://host/ws/chat/{session_id}?token={jwt_token}`

WebSocket endpoint for chat (requires JWT token as query parameter).

**Example:**
```javascript
const token = localStorage.getItem('access_token');
const ws = new WebSocket(`ws://localhost:8000/ws/chat/session_123?token=${token}`);
```

## User Management

### In-Memory User Database

**Default:** Users stored in memory (demo mode).

**Production:** Replace `UserDatabase` with real database (PostgreSQL, MongoDB, etc.).

### Creating New Users

```python
from HoloLoom.web.auth import user_db

# Create user
user = user_db.create_user(
    username="newuser",
    password="securepassword123",
    email="user@example.com",
    full_name="New User"
)
```

### Password Hashing

Passwords are hashed using **bcrypt** with automatic salt generation:

```python
from HoloLoom.web.auth import pwd_context

# Hash password
hashed = pwd_context.hash("mypassword")

# Verify password
is_valid = pwd_context.verify("mypassword", hashed)
```

### JWT Token Configuration

**Environment Variables:**

```bash
# Set secret key for production
export JWT_SECRET_KEY="your-secret-key-min-32-chars"

# Token expiration (default: 24 hours)
export JWT_EXPIRATION_MINUTES=1440
```

**In code:**

```python
from HoloLoom.web.auth import create_access_token
from datetime import timedelta

# Create token with custom expiration
token = create_access_token(
    data={"sub": "username"},
    expires_delta=timedelta(hours=1)
)
```

## Security Considerations

### ⚠️ Production Deployment

**CRITICAL:** Before deploying to production:

1. **Change JWT Secret Key:**
   ```bash
   export JWT_SECRET_KEY="$(openssl rand -hex 32)"
   ```

2. **Use HTTPS:** WebSocket must use `wss://` instead of `ws://`

3. **Replace User Database:** Use PostgreSQL, MongoDB, or other persistent storage

4. **Add Rate Limiting:** Prevent brute-force attacks on login endpoint

5. **Enable CORS properly:** Restrict allowed origins in production

6. **Add Token Revocation:** Implement token blacklist for logout

7. **Add Multi-Factor Authentication (MFA):** For sensitive deployments

### Password Requirements

**Default:** No requirements (demo mode).

**Recommended for production:**

```python
def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True
```

### Session Security

- **Session Timeout:** Default 24 hours
- **Session Cleanup:** Old sessions removed automatically
- **Concurrent Sessions:** Multiple sessions per user allowed

## Testing

### Manual Testing

1. **Login:**
   ```bash
   curl -X POST http://localhost:8000/api/auth/login_json \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin123"}'
   ```

2. **Get User Info:**
   ```bash
   TOKEN="your-jwt-token"
   curl http://localhost:8000/api/auth/me \
     -H "Authorization: Bearer $TOKEN"
   ```

3. **WebSocket (JavaScript):**
   ```javascript
   const token = localStorage.getItem('access_token');
   const ws = new WebSocket(`ws://localhost:8000/ws/chat/test?token=${token}`);

   ws.onopen = () => console.log('Connected!');
   ws.onmessage = (event) => console.log('Message:', event.data);
   ```

### Automated Testing

```python
import pytest
from HoloLoom.web.auth import user_db, create_access_token, decode_access_token

def test_user_authentication():
    # Create user
    user = user_db.create_user("testuser", "testpass123")

    # Authenticate
    auth_user = user_db.authenticate_user("testuser", "testpass123")
    assert auth_user is not None
    assert auth_user.username == "testuser"

    # Wrong password
    auth_user = user_db.authenticate_user("testuser", "wrongpass")
    assert auth_user is None

def test_jwt_tokens():
    # Create token
    token = create_access_token({"sub": "testuser"})
    assert token is not None

    # Decode token
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
```

## Troubleshooting

### "Authentication not available"

**Cause:** Auth module import failed.

**Fix:**
```bash
pip install pyjwt passlib[bcrypt]
```

### WebSocket closes immediately

**Cause:** Invalid or expired JWT token.

**Fix:**
- Check token in localStorage
- Re-login to get new token
- Verify token format: `eyJhbGciOi...`

### "Invalid authentication credentials"

**Cause:** Token expired or invalid.

**Fix:**
- Token expires after 24 hours (default)
- Clear localStorage and re-login
- Check server logs for detailed error

### CORS errors in browser

**Cause:** Browser blocking cross-origin requests.

**Fix:**
```python
# app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Migration to Production Database

### PostgreSQL Example

```python
import asyncpg
from passlib.context import CryptContext

class PostgresUserDatabase:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pwd_context = CryptContext(schemes=["bcrypt"])

    async def connect(self):
        self.pool = await asyncpg.create_pool(self.connection_string)

    async def create_user(self, username: str, password: str, **kwargs):
        hashed_password = self.pwd_context.hash(password)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (username, hashed_password, email, full_name)
                VALUES ($1, $2, $3, $4)
            """, username, hashed_password, kwargs.get('email'), kwargs.get('full_name'))

    async def authenticate_user(self, username: str, password: str):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE username = $1", username
            )

            if not row:
                return None

            if not self.pwd_context.verify(password, row['hashed_password']):
                return None

            return User(
                username=row['username'],
                hashed_password=row['hashed_password'],
                email=row['email'],
                full_name=row['full_name']
            )
```

## Future Enhancements

- [ ] Multi-factor authentication (MFA)
- [ ] OAuth2 integration (Google, GitHub, etc.)
- [ ] Role-based access control (RBAC)
- [ ] Token refresh mechanism
- [ ] Password reset via email
- [ ] Account lockout after failed attempts
- [ ] Audit logging for security events

## References

- **JWT:** https://jwt.io/
- **Passlib:** https://passlib.readthedocs.io/
- **FastAPI Security:** https://fastapi.tiangolo.com/tutorial/security/
- **OWASP Auth Cheatsheet:** https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
