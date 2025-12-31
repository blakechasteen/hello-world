# HoloLoom SaaS Toolkit

**Reusable authentication, API key management, and optional billing for HoloLoom ecosystem apps.**

This toolkit provides production-ready infrastructure for building applications on HoloLoom. Use what you need - authentication only, full billing, or anything in between.

## Philosophy

> **Open Source First**: Self-host your HoloLoom apps with full control. This toolkit makes it easy.

The SaaS toolkit is designed to be:
- **Modular** - Use only what you need (auth-only, or full billing stack)
- **Self-host friendly** - Works with SQLite (dev) or PostgreSQL (prod)
- **Billing optional** - Stripe integration is completely optional
- **Production ready** - Rate limiting, audit logging, secure password hashing

## Quick Start

### Minimal Setup (Auth Only)

```python
from fastapi import FastAPI, Depends
from HoloLoom.saas import SaaSBackend, SaaSConfig, create_saas_backend
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router

app = FastAPI()

# Create backend (SQLite by default - no external dependencies!)
backend = create_saas_backend()

@app.on_event("startup")
async def startup():
    await backend.connect()
    app.state.saas_backend = backend

@app.on_event("shutdown")
async def shutdown():
    await backend.close()

# Mount routes (routers already include /api/v1/... prefixes)
app.include_router(customers_router)
app.include_router(api_keys_router)

# Protect your endpoints
@app.get("/api/v1/my-protected-endpoint")
async def protected(auth: AuthContext = Depends(validate_api_key)):
    return {"customer_id": auth.customer_id, "plan": auth.plan}
```

### What You Get

| Component | Description | Optional? |
|-----------|-------------|-----------|
| **Customer Management** | Signup, login, profile | Required |
| **API Key Management** | Create, revoke, validate | Required |
| **Rate Limiting** | Token bucket per key | Included |
| **Subscriptions** | Plan-based tiers | Optional |
| **Usage Tracking** | Query counting | Optional |
| **Billing (Stripe)** | Payment processing | Optional |
| **Audit Logging** | Event tracking | Included |

## Installation

The SaaS toolkit is included with HoloLoom. No extra installation needed.

**Optional dependencies** for production:
```bash
# PostgreSQL support (recommended for production)
pip install asyncpg

# Secure password hashing (recommended)
pip install bcrypt

# Stripe billing (optional)
pip install stripe
```

## Components

### 1. Backend (`backend.py`)

The `SaaSBackend` handles all database operations.

```python
from HoloLoom.saas import SaaSBackend, SaaSConfig, create_saas_backend

# Development (SQLite)
config = SaaSConfig(
    sqlite_path="./data/my_app.db",
    fallback_to_sqlite=True
)
backend = create_saas_backend(config)

# Production (PostgreSQL)
config = SaaSConfig(
    host="localhost",
    port=5432,
    database="my_app",
    user="app_user",
    password="secure_password",
    fallback_to_sqlite=False  # Fail if PostgreSQL unavailable
)
backend = create_saas_backend(config)

# Use as async context manager
async with backend:
    customer, sub_id = await backend.insert_customer(
        email="user@example.com",
        name="User Name",
        password="secure_password"
    )
```

### 2. Models (`models.py`)

Pydantic models for request/response validation.

```python
from HoloLoom.saas import (
    # Requests
    SignupRequest,
    LoginRequest,
    APIKeyCreateRequest,

    # Responses
    SignupResponse,
    CustomerProfile,
    APIKeyResponse,

    # Errors
    ErrorResponse,
    RateLimitError,
    AuthError,
)

# Example: Create signup endpoint
@app.post("/signup", response_model=SignupResponse)
async def signup(request: SignupRequest):
    customer, _ = await backend.insert_customer(
        email=request.email,
        name=request.name,
        password=request.password
    )
    # ... create API key and return response
```

### 3. Authentication (`auth.py`)

FastAPI dependency for protecting endpoints.

```python
from HoloLoom.saas.auth import validate_api_key, AuthContext

@app.get("/protected")
async def protected(auth: AuthContext = Depends(validate_api_key)):
    """
    AuthContext provides:
    - customer_id: str
    - key_id: str
    - plan: str ("free", "payg", "pro", "enterprise")
    - rate_limit_qps: float
    - subscription_status: str
    - queries_included: int
    - price_per_query_cents: float
    """
    return {
        "customer": auth.customer_id,
        "plan": auth.plan,
        "rate_limit": auth.rate_limit_qps
    }
```

### 4. Routes (`routes/`)

Pre-built FastAPI routers for common operations.

```python
from HoloLoom.saas.routes import customers_router, api_keys_router, health_router

# Mount all routes
app.include_router(customers_router, prefix="/api/v1/customers")
app.include_router(api_keys_router, prefix="/api/v1/api-keys")
app.include_router(health_router)  # No prefix - standard paths
```

**Customer Routes** (`/api/v1/customers`):
- `POST /signup` - Create account
- `POST /login` - Authenticate
- `GET /me` - Get profile
- `PATCH /me` - Update profile

**API Key Routes** (`/api/v1/api-keys`):
- `POST /` - Create new key
- `GET /` - List keys
- `DELETE /{key_id}` - Revoke key

**Health Routes** (root level):
- `GET /health` - Basic health check (for load balancers)
- `GET /health/detailed` - Component status with latency
- `GET /health/features` - Feature flags status
- `GET /metrics` - Prometheus-compatible metrics
- `GET /ready` - Kubernetes readiness probe
- `GET /live` - Kubernetes liveness probe

## Configuration

### SaaSConfig Options

```python
from HoloLoom.saas import SaaSConfig

config = SaaSConfig(
    # PostgreSQL settings
    host="localhost",
    port=5432,
    database="hololoom_saas",
    user="hololoom",
    password="hololoom",

    # Connection pool
    min_pool_size=2,
    max_pool_size=10,
    pool_timeout=30.0,

    # SQLite fallback
    sqlite_path="./data/hololoom_saas.db",
    fallback_to_sqlite=True,  # Auto-fallback if PostgreSQL unavailable

    # API Key settings
    key_prefix="holo",  # Keys will be "holo_..."
    key_length=32,
)
```

### Plan Configuration

Default plans are defined in `models.py`:

```python
from HoloLoom.saas import PLAN_CONFIGS, get_plan_config

# View available plans
for plan_id, config in PLAN_CONFIGS.items():
    print(f"{plan_id}: {config.queries_included} queries, {config.rate_limit_qps} QPS")

# Get specific plan
pro_config = get_plan_config("pro")
```

**Default Plans**:

| Plan | Rate Limit | Queries Included | Price/Query |
|------|------------|------------------|-------------|
| `free` | 1 QPS | 0 (100/day limit) | $0 |
| `payg` | 10 QPS | 0 | $0.001 |
| `pro` | 50 QPS | 50,000 | $0.0005 |
| `enterprise` | 100 QPS | Custom | Custom |

## Usage Patterns

### Pattern 1: Auth Only (No Billing)

For apps that don't need billing - just authentication and rate limiting.

```python
from fastapi import FastAPI, Depends
from HoloLoom.saas import create_saas_backend
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router

app = FastAPI()
backend = create_saas_backend()

@app.on_event("startup")
async def startup():
    await backend.connect()
    app.state.saas_backend = backend

# Mount routes (routers already include /api/v1/... prefixes)
app.include_router(customers_router)
app.include_router(api_keys_router)

# Your protected endpoints
@app.post("/api/v1/query")
async def query(auth: AuthContext = Depends(validate_api_key)):
    # Process query...
    return {"result": "..."}
```

### Pattern 2: Usage Tracking (No Billing)

Track usage for analytics without charging.

```python
@app.post("/api/v1/query")
async def query(
    request: Request,
    auth: AuthContext = Depends(validate_api_key)
):
    # Track usage
    await backend.record_usage(
        customer_id=auth.customer_id,
        queries_delta=1
    )

    # Process query...
    return {"result": "..."}

# Dashboard endpoint
@app.get("/api/v1/usage")
async def get_usage(auth: AuthContext = Depends(validate_api_key)):
    from datetime import date, timedelta

    start = date.today() - timedelta(days=30)
    usage = await backend.get_usage_for_period(auth.customer_id, start)

    total = sum(u.queries_count for u in usage)
    return {"total_queries_30d": total, "daily": usage}
```

### Pattern 3: Full Billing (Stripe)

Complete billing integration (requires Stripe account).

```python
# See HoloLoom/saas/stripe_client.py (coming soon)
# and HoloLoom/saas/routes/billing.py
```

## Ecosystem Integration

The SaaS toolkit integrates with HoloLoom's other major subsystems. See [Integration Strategy](../../docs/INTEGRATION_STRATEGY.md) for complete architecture.

### Pattern 4: HoloLoom Lite + SaaS (Protected Web Service)

Add authentication to HoloLoom Lite for hosted deployments:

```python
from fastapi import FastAPI, Depends
from HoloLoom.lite import HoloLoomLite
from HoloLoom.saas import create_saas_backend
from HoloLoom.saas.auth import validate_api_key, AuthContext
from HoloLoom.saas.routes import customers_router, api_keys_router

app = FastAPI()
backend = create_saas_backend()
loom = HoloLoomLite(persist=True)

@app.on_event("startup")
async def startup():
    await backend.connect()
    await loom.connect()
    app.state.saas_backend = backend

@app.on_event("shutdown")
async def shutdown():
    await backend.close()
    await loom.close()

# Mount SaaS routes
app.include_router(customers_router)
app.include_router(api_keys_router)

# Protected Lite endpoints
@app.post("/api/v1/experience")
async def experience(
    content: str,
    auth: AuthContext = Depends(validate_api_key)
):
    # Track usage
    await backend.record_usage(auth.customer_id, queries_delta=1)
    memory = await loom.experience(content)
    return {"memory_id": memory.id}

@app.post("/api/v1/query")
async def query(
    question: str,
    auth: AuthContext = Depends(validate_api_key)
):
    await backend.record_usage(auth.customer_id, queries_delta=1)
    result = await loom.query(question)
    return {"response": result.response, "confidence": result.confidence}
```

**What you get**:
- Zero-config Lite (5 simple methods)
- API key authentication
- Rate limiting per customer
- Usage tracking
- Safety guardrails included

### Pattern 5: Full HoloLoom + Federation (Decentralized Enterprise)

For multi-node deployments with community verification:

```python
from fastapi import FastAPI
from HoloLoom import HoloLoom
from HoloLoom.federation import FederationNode, FederationConfig
from HoloLoom.saas import create_saas_backend, SaaSConfig
from HoloLoom.saas.routes import customers_router, api_keys_router, health_router

app = FastAPI()

# SaaS for customer management
saas_config = SaaSConfig.with_usage(
    host="db.example.com",
    database="hololoom_prod"
)
saas_backend = create_saas_backend(saas_config)

# Federation for decentralized verification
fed_config = FederationConfig(
    bootstrap_nodes=["bootstrap.hololoom.network:9000"],
    guild_id="enterprise_safety"
)

@app.on_event("startup")
async def startup():
    await saas_backend.connect()
    app.state.saas_backend = saas_backend

    app.state.federation = FederationNode(fed_config)
    await app.state.federation.start()

    app.state.loom = HoloLoom(federation=app.state.federation)
    await app.state.loom.connect()

# Mount routes
app.include_router(health_router)
app.include_router(customers_router)
app.include_router(api_keys_router)
```

**What you get**:
- Full HoloLoom capabilities (50+ methods)
- Decentralized verification (no central authority)
- Byzantine fault tolerance
- Guild-based trust groups
- API key authentication + rate limiting

### Comparison: Lite vs Full vs Federation

| Feature | Lite + SaaS | Full + SaaS | Full + Federation |
|---------|-------------|-------------|-------------------|
| **API Complexity** | 5 methods | 50+ methods | 50+ methods |
| **Dependencies** | Optional | Required | Required + P2P |
| **Default Storage** | In-memory | Configurable | Distributed |
| **Safety** | Built-in | Built-in | Community-verified |
| **Verification** | Local | Local | Byzantine consensus |
| **Use Case** | Personal/embedded | Production | Enterprise/research |

### Related Documentation

- [HoloLoom Lite](../lite/README.md) - Simplified 5-method API
- [Federation](../federation/README.md) - Decentralized P2P network
- [Integration Strategy](../../docs/INTEGRATION_STRATEGY.md) - Complete ecosystem architecture
- [Self-Hosting Guide](../../docs/self-hosting/README.md) - Production deployment

## API Reference

### Backend Methods

#### Customer Management

```python
# Create customer
customer, subscription_id = await backend.insert_customer(
    email="user@example.com",
    name="User Name",
    password="secure_password",
    company="Optional Company"
)

# Get customer
customer = await backend.get_customer(customer_id)
customer = await backend.get_customer_by_email(email)

# Authenticate
customer = await backend.authenticate_customer(email, password)

# Update
customer = await backend.update_customer(customer_id, name="New Name")
```

#### API Key Management

```python
# Create API key (returns key object + secret)
api_key, secret = await backend.create_api_key(
    customer_id=customer.customer_id,
    name="My Key",
    rate_limit_qps=10.0,
    expires_in_days=365  # Optional
)
# IMPORTANT: secret is only returned once!

# Validate key (for authentication)
api_key = await backend.get_api_key_by_secret(secret)

# List customer's keys
keys = await backend.get_api_keys_for_customer(customer_id)

# Revoke key
await backend.revoke_api_key(key_id, customer_id)

# Update usage stats
await backend.update_api_key_usage(key_id)
```

#### Subscription Management

```python
# Get subscription
subscription = await backend.get_subscription(customer_id)

# Update subscription (e.g., upgrade plan)
await backend.update_subscription(
    subscription_id,
    plan="pro",
    status="active"
)
```

#### Usage Tracking

```python
from datetime import date, timedelta

# Record usage
await backend.record_usage(
    customer_id=customer_id,
    queries_delta=1,
    tokens_delta=150  # Optional
)

# Get usage for period
start = date.today() - timedelta(days=30)
usage = await backend.get_usage_for_period(customer_id, start)

# Get today's usage
today_usage = await backend.get_usage_for_date(customer_id, date.today())
```

#### Audit Logging

```python
# Log an event
await backend.log_event(
    event_type="api_call",
    customer_id=customer_id,
    event_data={"endpoint": "/query", "status": 200},
    ip_address=request.client.host
)
```

### AuthContext Fields

When using `validate_api_key` dependency:

```python
@dataclass
class AuthContext:
    customer_id: str       # Customer ID
    key_id: str            # API key ID
    plan: str              # "free", "payg", "pro", "enterprise"
    rate_limit_qps: float  # Queries per second limit
    subscription_status: str  # "active", "trialing", etc.
    queries_included: int  # Monthly included queries
    price_per_query_cents: float  # Overage pricing
    metadata: Dict[str, Any]  # Custom data
```

## Rate Limiting

The toolkit includes a token bucket rate limiter that:
- Tracks per-API-key rate limits
- Returns standard rate limit headers
- Rejects requests with 429 when exceeded

**Headers returned**:
```
X-RateLimit-Limit: 10.0
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1701023400
```

**Customizing rate limits**:
```python
# Per-key rate limit (set when creating key)
api_key, secret = await backend.create_api_key(
    customer_id=customer_id,
    rate_limit_qps=50.0  # 50 requests per second
)

# Or update plan defaults in PLAN_CONFIGS
```

## Security

### Password Hashing

- Uses `bcrypt` if available (recommended)
- Falls back to SHA256 if bcrypt not installed
- Always use bcrypt in production: `pip install bcrypt`

### API Key Secrets

- Secrets are shown **once** on creation
- Only the SHA256 hash is stored
- Keys have an 8-character prefix for identification

### Request Signing (Optional)

For sensitive operations, use HMAC-SHA256 signing:

```python
from HoloLoom.saas.auth import verify_request_signature

@app.post("/sensitive-operation")
async def sensitive(request: Request, auth: AuthContext = Depends(validate_api_key)):
    # Verify signature for extra security
    if not verify_request_signature(request, api_secret):
        raise HTTPException(401, "Invalid signature")
    ...
```

## Database Schema

The schema is defined in `schema.sql`. Key tables:

- `customers` - User accounts
- `subscriptions` - Plan/billing info
- `api_keys` - API key storage (hashed secrets)
- `usage_records` - Daily usage aggregation
- `audit_log` - Event logging

**Auto-initialization**: Schema is created automatically on first `connect()`.

## Self-Hosting Guide

### Development (SQLite)

No setup required - just use defaults:

```python
backend = create_saas_backend()  # Uses SQLite at ./data/hololoom_saas.db
```

### Production (PostgreSQL)

1. **Create database**:
```sql
CREATE DATABASE my_app;
CREATE USER my_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE my_app TO my_user;
```

2. **Configure backend**:
```python
config = SaaSConfig(
    host="db.example.com",
    port=5432,
    database="my_app",
    user="my_user",
    password="secure_password",
    min_pool_size=5,
    max_pool_size=20,
    fallback_to_sqlite=False
)
```

3. **Deploy with Docker** (using included files):
```bash
# From repository root
cd /path/to/mythRL

# Set secure password
export POSTGRES_PASSWORD=your_secure_password_here

# Start PostgreSQL + SaaS API
docker-compose -f HoloLoom/saas/docker-compose.yml up -d

# Check status
docker-compose -f HoloLoom/saas/docker-compose.yml ps

# View logs
docker-compose -f HoloLoom/saas/docker-compose.yml logs -f api

# Access API
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed

# Stop services
docker-compose -f HoloLoom/saas/docker-compose.yml down
```

4. **Build standalone Docker image**:
```bash
# Build image
docker build -t hololoom-saas -f HoloLoom/saas/Dockerfile .

# Run with SQLite (development)
docker run -p 8000:8000 hololoom-saas

# Run with PostgreSQL (production)
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e ENABLE_BILLING=true \
  -e STRIPE_API_KEY=sk_... \
  hololoom-saas
```

5. **Environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite | PostgreSQL connection string |
| `ENABLE_BILLING` | `false` | Enable Stripe billing |
| `ENABLE_USAGE_TRACKING` | `true` | Track query counts |
| `ENABLE_AUDIT_LOG` | `true` | Log all events |
| `STRIPE_API_KEY` | - | Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | - | Stripe webhook secret |
| `PORT` | `8000` | Server port |
| `WORKERS` | `4` | Uvicorn workers |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

## Examples

See `HoloLoom/saas/examples/` for complete examples:

- `auth_only_app.py` - Minimal auth-only setup
- `usage_tracking_app.py` - Auth + usage analytics
- `full_billing_app.py` - Complete Stripe integration

## Roadmap

- [ ] Stripe integration (`stripe_client.py`, `routes/billing.py`)
- [ ] Dashboard routes (`routes/dashboard.py`)
- [ ] Webhook handlers (`routes/webhooks.py`)
- [ ] Redis rate limiting (for distributed deployments)
- [ ] Team/organization support

## License

MIT License - Use freely in your HoloLoom ecosystem apps.

## Support

- **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Documentation**: This file + inline docstrings
- **Examples**: `HoloLoom/saas/examples/`
