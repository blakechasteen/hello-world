#!/usr/bin/env python3
"""
Auth-Only Example App

Demonstrates minimal SaaS toolkit usage - just authentication and API keys,
no billing or usage tracking.

Run with:
    PYTHONPATH=. uvicorn HoloLoom.saas.examples.auth_only_app:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager

from hololoom.saas import (
    SaaSBackend,
    SaaSConfig,
    create_saas_backend,
)
from hololoom.saas.auth import validate_api_key, AuthContext
from hololoom.saas.routes import customers_router, api_keys_router, health_router


# ============================================================================
# App Setup
# ============================================================================

# Create backend with SQLite (no external dependencies)
config = SaaSConfig(
    sqlite_path="./data/auth_only_example.db",
    fallback_to_sqlite=True,
)
backend = create_saas_backend(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend lifecycle"""
    await backend.connect()
    app.state.saas_backend = backend
    print("Backend connected")
    yield
    await backend.close()
    print("Backend closed")


app = FastAPI(
    title="Auth-Only Example",
    description="Minimal HoloLoom SaaS Toolkit example",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Mount SaaS Routes
# ============================================================================

# Customer management (signup, login, profile)
# Note: Router already has /api/v1/customers prefix
app.include_router(
    customers_router,
    tags=["customers"],
)

# API key management (create, list, revoke)
# Note: Router already has /api/v1/api-keys prefix
app.include_router(
    api_keys_router,
    tags=["api-keys"],
)

# Health & monitoring (health checks, metrics, K8s probes)
app.include_router(
    health_router,
    tags=["health"],
)


# ============================================================================
# Your Protected Endpoints
# ============================================================================

@app.get("/api/v1/hello")
async def hello(auth: AuthContext = Depends(validate_api_key)):
    """
    Protected endpoint example.

    Requires valid API key in X-API-Key header.
    Returns customer info from AuthContext.
    """
    return {
        "message": "Hello from your protected endpoint!",
        "customer_id": auth.customer_id,
        "plan": auth.plan,
        "rate_limit_qps": auth.rate_limit_qps,
    }


@app.post("/api/v1/echo")
async def echo(
    data: dict,
    auth: AuthContext = Depends(validate_api_key),
):
    """
    Echo back data with customer info.
    """
    return {
        "customer_id": auth.customer_id,
        "echoed_data": data,
    }


# ============================================================================
# Public Endpoints (No Auth Required)
# ============================================================================

# Note: /health, /health/detailed, /health/features, /metrics, /ready, /live
# are all provided by health_router (see HoloLoom.saas.routes.health)


@app.get("/")
async def root():
    """Root endpoint with usage info"""
    return {
        "name": "Auth-Only Example",
        "docs": "/docs",
        "endpoints": {
            "signup": "POST /api/v1/customers/signup",
            "login": "POST /api/v1/customers/login",
            "create_key": "POST /api/v1/api-keys (requires auth)",
            "hello": "GET /api/v1/hello (requires API key)",
        },
        "health": {
            "basic": "GET /health",
            "detailed": "GET /health/detailed",
            "features": "GET /health/features",
            "metrics": "GET /metrics (Prometheus)",
            "k8s_ready": "GET /ready",
            "k8s_live": "GET /live",
        },
        "usage": "1. Signup -> 2. Login -> 3. Use API key in X-API-Key header",
    }


# ============================================================================
# Usage Example (for testing)
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import httpx

    async def demo():
        """Demo the auth flow"""
        base_url = "http://localhost:8000"

        async with httpx.AsyncClient() as client:
            # 1. Signup
            print("\n1. Signup...")
            signup_resp = await client.post(
                f"{base_url}/api/v1/customers/signup",
                json={
                    "email": "demo@example.com",
                    "name": "Demo User",
                    "password": "secure_password_123",
                },
            )
            print(f"   Status: {signup_resp.status_code}")

            if signup_resp.status_code == 200:
                data = signup_resp.json()
                api_secret = data["api_secret"]
                print(f"   API Secret: {api_secret[:16]}...")

                # 2. Use protected endpoint
                print("\n2. Call protected endpoint...")
                hello_resp = await client.get(
                    f"{base_url}/api/v1/hello",
                    headers={"X-API-Key": api_secret},
                )
                print(f"   Status: {hello_resp.status_code}")
                print(f"   Response: {hello_resp.json()}")

                # 3. Check rate limit headers
                print("\n3. Rate limit headers:")
                for header in ["X-RateLimit-Limit", "X-RateLimit-Remaining"]:
                    if header in hello_resp.headers:
                        print(f"   {header}: {hello_resp.headers[header]}")
            else:
                print(f"   Error: {signup_resp.text}")

    print("Run the server first, then run this file directly to test.")
    print("Server: PYTHONPATH=. uvicorn HoloLoom.saas.examples.auth_only_app:app")
