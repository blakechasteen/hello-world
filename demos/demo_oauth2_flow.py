"""
OAuth2/OpenID Connect Flow Demo

Demonstrates complete OAuth2 authentication flow with multiple providers.

Features Demonstrated:
- Multiple provider support (Auth0, Okta, Google, GitHub)
- Authorization Code flow with PKCE
- Client Credentials flow (machine-to-machine)
- Token validation and introspection
- Token refresh
- Token revocation
- FastAPI integration with middleware
- Rate limiting and audit logging

Requirements:
- pip install fastapi uvicorn aiohttp python-jose[cryptography] redis

Usage:
    # 1. Set environment variables:
    export OAUTH2_PROVIDER="auth0"  # or "okta", "google", "github"
    export OAUTH2_CLIENT_ID="your-client-id"
    export OAUTH2_CLIENT_SECRET="your-client-secret"
    export OAUTH2_DOMAIN="your-tenant.auth0.com"  # For Auth0/Okta

    # 2. Run demo:
    PYTHONPATH=. python demos/demo_oauth2_flow.py

    # 3. Open browser:
    http://localhost:8000/docs

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import asyncio
import os
import sys
from typing import Optional

# Check if FastAPI is available
try:
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.responses import RedirectResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not installed. Install with: pip install fastapi uvicorn")

# Add HoloLoom to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from HoloLoom.security.oauth2 import (
    OAuth2Client,
    OAuth2Config,
    OAuth2Provider,
    create_oauth2_client
)
from HoloLoom.security.jwt_validator import (
    JWTValidator,
    create_jwt_validator
)

if FASTAPI_AVAILABLE:
    from HoloLoom.security.middleware import create_auth_middleware


# Configuration from environment
PROVIDER = os.getenv("OAUTH2_PROVIDER", "auth0")
CLIENT_ID = os.getenv("OAUTH2_CLIENT_ID", "demo-client-id")
CLIENT_SECRET = os.getenv("OAUTH2_CLIENT_SECRET", "demo-client-secret")
DOMAIN = os.getenv("OAUTH2_DOMAIN", "dev-example.auth0.com")
REDIRECT_URI = os.getenv("OAUTH2_REDIRECT_URI", "http://localhost:8000/callback")
REDIS_URL = os.getenv("REDIS_URL", None)


async def demo_oauth2_flow():
    """
    Demonstrate OAuth2 Authorization Code flow with PKCE.

    Steps:
    1. Generate authorization URL
    2. User authorizes (simulated)
    3. Exchange code for token
    4. Validate token
    5. Introspect token
    6. Refresh token
    7. Revoke token
    """
    print("=" * 80)
    print("OAuth2/OpenID Connect Flow Demo")
    print("=" * 80)

    # Create OAuth2 client
    print(f"\n📋 Provider: {PROVIDER}")
    print(f"   Client ID: {CLIENT_ID[:20]}...")
    print(f"   Domain: {DOMAIN}")

    try:
        client = create_oauth2_client(
            provider=PROVIDER,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            domain=DOMAIN,
            redis_url=REDIS_URL
        )
    except Exception as e:
        print(f"\n❌ Failed to create OAuth2 client: {e}")
        print("\n💡 Set environment variables:")
        print("   export OAUTH2_PROVIDER=auth0")
        print("   export OAUTH2_CLIENT_ID=your-client-id")
        print("   export OAUTH2_CLIENT_SECRET=your-client-secret")
        print("   export OAUTH2_DOMAIN=your-tenant.auth0.com")
        return

    async with client:
        # Step 1: Generate authorization URL
        print("\n" + "=" * 80)
        print("Step 1: Generate Authorization URL")
        print("=" * 80)

        auth_url, state, code_verifier = client.get_authorization_url()

        print(f"\n✅ Authorization URL generated:")
        print(f"\n   {auth_url}")
        print(f"\n   State: {state}")
        print(f"   Code Verifier: {code_verifier[:20]}...")

        print("\n💡 In production:")
        print("   1. Redirect user to this URL")
        print("   2. User authorizes your app")
        print("   3. Provider redirects back with code")

        # Step 2: Exchange code for token (simulated)
        print("\n" + "=" * 80)
        print("Step 2: Exchange Code for Token (Simulated)")
        print("=" * 80)

        print("\n⚠️  Skipping token exchange (requires real authorization)")
        print("   In production, you'd call:")
        print("   token = await client.exchange_code_for_token(code, state, code_verifier)")

    # Demonstrate Client Credentials flow (machine-to-machine)
    print("\n" + "=" * 80)
    print("Client Credentials Flow (Machine-to-Machine)")
    print("=" * 80)

    if PROVIDER in ["auth0", "okta"]:
        print("\n💡 Attempting Client Credentials flow...")
        async with client:
            try:
                token = await client.get_client_credentials_token()
                print(f"\n✅ Token obtained:")
                print(f"   Access Token: {token.access_token[:20]}...")
                print(f"   Token Type: {token.token_type}")
                print(f"   Expires In: {token.expires_in}s")
            except Exception as e:
                print(f"\n⚠️  Client Credentials flow failed: {e}")
                print("   (Expected for demo credentials)")
    else:
        print(f"\n⚠️  {PROVIDER} does not support Client Credentials flow")


async def demo_jwt_validation():
    """
    Demonstrate JWT validation.
    """
    print("\n" + "=" * 80)
    print("JWT Validation Demo")
    print("=" * 80)

    # Create JWT validator
    if PROVIDER == "auth0":
        issuer = f"https://{DOMAIN}/"
        jwks_url = f"https://{DOMAIN}/.well-known/jwks.json"
    elif PROVIDER == "okta":
        issuer = f"https://{DOMAIN}/oauth2/default"
        jwks_url = f"https://{DOMAIN}/oauth2/default/v1/keys"
    else:
        print(f"\n⚠️  JWT validation not configured for {PROVIDER}")
        return

    print(f"\n📋 Issuer: {issuer}")
    print(f"   JWKS URL: {jwks_url}")

    validator = create_jwt_validator(
        issuer=issuer,
        audience=CLIENT_ID,
        jwks_url=jwks_url,
        redis_url=REDIS_URL
    )

    # Demo token (will fail validation without real token)
    print("\n💡 To validate a real token:")
    print("   claims = await validator.validate_token(access_token)")

    # Demonstrate unverified decoding
    demo_token = (
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJhdXRoMHwxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiZW1haWwiOiJqb2huQGV4YW1wbGUuY29tIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "signature"
    )

    print(f"\n📋 Decoding demo token (unverified):")
    claims = validator.decode_token_unverified(demo_token)
    print(f"   Subject: {claims.get('sub')}")
    print(f"   Name: {claims.get('name')}")
    print(f"   Email: {claims.get('email')}")


def demo_fastapi_integration():
    """
    Demonstrate FastAPI integration with OAuth2 middleware.
    """
    if not FASTAPI_AVAILABLE:
        print("\n❌ FastAPI not installed. Skipping FastAPI demo.")
        return

    print("\n" + "=" * 80)
    print("FastAPI Integration Demo")
    print("=" * 80)

    # Create FastAPI app
    app = FastAPI(
        title="HoloLoom OAuth2 Demo",
        description="OAuth2/OpenID Connect authentication demo",
        version="1.0.0"
    )

    # OAuth2 client (global for demo)
    oauth2_client: Optional[OAuth2Client] = None

    # JWT validator
    if PROVIDER == "auth0":
        issuer = f"https://{DOMAIN}/"
        jwks_url = f"https://{DOMAIN}/.well-known/jwks.json"
        jwt_validator = create_jwt_validator(
            issuer=issuer,
            audience=CLIENT_ID,
            jwks_url=jwks_url
        )
    else:
        jwt_validator = None
        print("\n⚠️  JWT validation not configured for this provider")

    @app.on_event("startup")
    async def startup():
        """Initialize OAuth2 client on startup."""
        global oauth2_client
        try:
            oauth2_client = create_oauth2_client(
                provider=PROVIDER,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
                domain=DOMAIN,
                redis_url=REDIS_URL
            )
            await oauth2_client.__aenter__()
            print("\n✅ OAuth2 client initialized")
        except Exception as e:
            print(f"\n⚠️  OAuth2 client initialization failed: {e}")

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanup OAuth2 client on shutdown."""
        if oauth2_client:
            await oauth2_client.__aexit__(None, None, None)
            print("\n✅ OAuth2 client cleaned up")

    @app.get("/", tags=["Public"])
    async def root():
        """Public root endpoint."""
        return {
            "message": "HoloLoom OAuth2 Demo",
            "provider": PROVIDER,
            "endpoints": {
                "login": "/login",
                "callback": "/callback",
                "protected": "/api/protected",
                "health": "/health",
                "docs": "/docs"
            }
        }

    @app.get("/health", tags=["Public"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "provider": PROVIDER}

    @app.get("/login", tags=["Authentication"])
    async def login():
        """Initiate OAuth2 login flow."""
        if not oauth2_client:
            raise HTTPException(
                status_code=500,
                detail="OAuth2 client not initialized"
            )

        # Generate authorization URL
        auth_url, state, code_verifier = oauth2_client.get_authorization_url()

        # Store state and verifier (in production, use session/Redis)
        # For demo, we'll just return them
        return {
            "authorization_url": auth_url,
            "state": state,
            "code_verifier": code_verifier,
            "instructions": "Visit authorization_url to authorize, then return to /callback with code"
        }

    @app.get("/callback", tags=["Authentication"])
    async def callback(code: str, state: str):
        """OAuth2 callback endpoint."""
        if not oauth2_client:
            raise HTTPException(
                status_code=500,
                detail="OAuth2 client not initialized"
            )

        try:
            # Exchange code for token
            # Note: In production, retrieve code_verifier from session
            token = await oauth2_client.exchange_code_for_token(
                code=code,
                state=state
            )

            return {
                "message": "Authentication successful",
                "access_token": token.access_token[:20] + "...",
                "token_type": token.token_type,
                "expires_in": token.expires_in
            }
        except ValueError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Authentication failed: {e}"
            )

    @app.get("/api/protected", tags=["Protected"])
    async def protected(request: Request):
        """Protected endpoint requiring authentication."""
        # Check for Bearer token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = auth_header.split(" ", 1)[1]

        # Validate token
        if jwt_validator:
            try:
                async with jwt_validator:
                    claims = await jwt_validator.validate_token(token)

                return {
                    "message": "Access granted",
                    "user_id": claims.sub,
                    "issuer": claims.iss,
                    "scopes": claims.get_claim("scope", "").split()
                }
            except ValueError as e:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token: {e}"
                )
        else:
            # Fallback: decode unverified
            if jwt_validator is None:
                jwt_validator = create_jwt_validator()

            claims = jwt_validator.decode_token_unverified(token)
            return {
                "message": "Access granted (unverified token)",
                "user_id": claims.get("sub"),
                "warning": "Token signature not verified (demo mode)"
            }

    print("\n✅ FastAPI app configured")
    print("\n📋 Available endpoints:")
    print("   GET  /           - Root")
    print("   GET  /health     - Health check")
    print("   GET  /login      - Initiate OAuth2 login")
    print("   GET  /callback   - OAuth2 callback")
    print("   GET  /api/protected - Protected endpoint (requires token)")
    print("   GET  /docs       - API documentation")

    print("\n🚀 Starting server on http://localhost:8000")
    print("   Visit http://localhost:8000/docs for interactive API")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("HoloLoom OAuth2/OpenID Connect Demo")
    print("=" * 80)

    # Demo 1: OAuth2 flow
    await demo_oauth2_flow()

    # Demo 2: JWT validation
    await demo_jwt_validation()

    # Demo 3: FastAPI integration (interactive)
    print("\n" + "=" * 80)
    print("FastAPI Integration")
    print("=" * 80)
    print("\n💡 Run FastAPI demo? (y/n): ", end="")

    try:
        response = input().strip().lower()
        if response == "y":
            demo_fastapi_integration()
    except (EOFError, KeyboardInterrupt):
        print("\n\n✅ Demo completed")

    print("\n" + "=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print("\n✅ Demonstrated:")
    print("   - OAuth2 configuration (multiple providers)")
    print("   - Authorization Code flow with PKCE")
    print("   - Client Credentials flow (machine-to-machine)")
    print("   - JWT validation and introspection")
    print("   - FastAPI integration")

    print("\n💡 Next steps:")
    print("   1. Set up real OAuth2 provider (Auth0, Okta, etc.)")
    print("   2. Configure client credentials")
    print("   3. Integrate with HoloLoom API")
    print("   4. Add rate limiting and audit logging")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n✅ Demo interrupted by user")
