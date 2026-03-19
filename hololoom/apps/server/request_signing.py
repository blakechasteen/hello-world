"""
Request Signing for API Authentication

Implements HMAC-SHA256 request signing for secure API authentication:
- Client signs requests with secret key
- Server validates signatures
- Timestamp validation to prevent replay attacks
- Multiple API key management

Created: 2025-11-26
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """API key status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class APIKey:
    """API Key configuration"""
    key_id: str
    secret: str
    client_name: str
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    allowed_ips: set[str] = field(default_factory=set)
    rate_limit_qps: float = 100.0
    permissions: set[str] = field(default_factory=lambda: {"read", "write"})
    metadata: dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    last_used: datetime | None = None


class SignatureValidator:
    """Validates HMAC-SHA256 request signatures"""

    SIGNATURE_HEADER = "X-API-Signature"
    TIMESTAMP_HEADER = "X-API-Timestamp"
    KEY_ID_HEADER = "X-API-Key-ID"
    NONCE_HEADER = "X-API-Nonce"

    def __init__(
        self,
        max_timestamp_age_seconds: int = 300,  # 5 minutes
        require_nonce: bool = False,
        enforce_replay_protection: bool = True
    ):
        self.max_timestamp_age_seconds = max_timestamp_age_seconds
        self.require_nonce = require_nonce
        self.enforce_replay_protection = enforce_replay_protection

        # Store used nonces to prevent replay attacks
        self.used_nonces: set[str] = set()
        self.nonce_expiry: dict[str, float] = {}

        # API keys storage (in production, use database)
        self.api_keys: dict[str, APIKey] = {}

        logger.info("SignatureValidator initialized")

    def generate_api_key(
        self,
        client_name: str,
        expires_in_days: int | None = None,
        allowed_ips: list[str] | None = None,
        rate_limit_qps: float = 100.0,
        permissions: set[str] | None = None
    ) -> tuple[str, str]:
        """Generate a new API key"""

        # Generate secure random key ID and secret
        key_id = f"holo_{secrets.token_urlsafe(16)}"
        secret = secrets.token_urlsafe(32)

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            secret=secret,
            client_name=client_name,
            expires_at=expires_at,
            allowed_ips=set(allowed_ips) if allowed_ips else set(),
            rate_limit_qps=rate_limit_qps,
            permissions=permissions or {"read", "write"}
        )

        # Store the key
        self.api_keys[key_id] = api_key

        logger.info(f"Generated API key for client: {client_name} (key_id: {key_id})")

        return key_id, secret

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""

        if key_id in self.api_keys:
            self.api_keys[key_id].status = APIKeyStatus.REVOKED
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False

    def calculate_signature(
        self,
        secret: str,
        method: str,
        url: str,
        timestamp: str,
        body: bytes = b"",
        nonce: str | None = None
    ) -> str:
        """Calculate HMAC-SHA256 signature for request"""

        # Create string to sign
        body_hash = hashlib.sha256(body).hexdigest() if body else ""

        parts = [
            method.upper(),
            url,
            timestamp,
            body_hash
        ]

        if nonce:
            parts.append(nonce)

        string_to_sign = "\n".join(parts)

        # Calculate HMAC-SHA256
        signature = hmac.new(
            secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()

        # Return base64 encoded signature
        return base64.b64encode(signature).decode('utf-8')

    async def validate_request(self, request: Request) -> APIKey:
        """Validate request signature and return API key"""

        # Get required headers
        signature = request.headers.get(self.SIGNATURE_HEADER)
        timestamp = request.headers.get(self.TIMESTAMP_HEADER)
        key_id = request.headers.get(self.KEY_ID_HEADER)
        nonce = request.headers.get(self.NONCE_HEADER)

        # Check required headers
        if not signature or not timestamp or not key_id:
            raise HTTPException(
                status_code=401,
                detail="Missing required authentication headers"
            )

        # Check if nonce is required
        if self.require_nonce and not nonce:
            raise HTTPException(
                status_code=401,
                detail="Missing required nonce header"
            )

        # Get API key
        api_key = self.api_keys.get(key_id)
        if not api_key:
            logger.warning(f"Invalid API key ID: {key_id}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

        # Check key status
        if api_key.status != APIKeyStatus.ACTIVE:
            logger.warning(f"Inactive API key used: {key_id} (status: {api_key.status})")
            raise HTTPException(
                status_code=401,
                detail=f"API key is {api_key.status.value}"
            )

        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            api_key.status = APIKeyStatus.EXPIRED
            logger.warning(f"Expired API key used: {key_id}")
            raise HTTPException(
                status_code=401,
                detail="API key expired"
            )

        # Check IP whitelist
        client_ip = request.client.host if request.client else "unknown"
        if api_key.allowed_ips and client_ip not in api_key.allowed_ips:
            logger.warning(f"API key {key_id} used from unauthorized IP: {client_ip}")
            raise HTTPException(
                status_code=403,
                detail="IP address not authorized for this API key"
            )

        # Validate timestamp (prevent replay attacks)
        try:
            request_time = float(timestamp)
            current_time = time.time()

            if abs(current_time - request_time) > self.max_timestamp_age_seconds:
                logger.warning(f"Request timestamp too old: {timestamp}")
                raise HTTPException(
                    status_code=401,
                    detail="Request timestamp too old"
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp format"
            )

        # Check replay protection (nonce)
        if self.enforce_replay_protection and nonce:
            if nonce in self.used_nonces:
                logger.warning(f"Replay attack detected - nonce reused: {nonce}")
                raise HTTPException(
                    status_code=401,
                    detail="Nonce already used"
                )

            # Store nonce
            self.used_nonces.add(nonce)
            self.nonce_expiry[nonce] = current_time + self.max_timestamp_age_seconds

            # Clean up old nonces
            self._cleanup_nonces(current_time)

        # Read request body
        body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            # Create new request with cached body for downstream
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

        # Calculate expected signature
        expected_signature = self.calculate_signature(
            secret=api_key.secret,
            method=request.method,
            url=str(request.url),
            timestamp=timestamp,
            body=body,
            nonce=nonce
        )

        # Compare signatures (constant time comparison)
        if not hmac.compare_digest(signature, expected_signature):
            logger.warning(f"Invalid signature for API key: {key_id}")
            raise HTTPException(
                status_code=401,
                detail="Invalid signature"
            )

        # Update API key usage statistics
        api_key.request_count += 1
        api_key.last_used = datetime.utcnow()

        logger.debug(f"Request validated for API key: {key_id} ({api_key.client_name})")

        return api_key

    def _cleanup_nonces(self, current_time: float):
        """Clean up expired nonces"""

        expired_nonces = [
            nonce for nonce, expiry in self.nonce_expiry.items()
            if expiry < current_time
        ]

        for nonce in expired_nonces:
            self.used_nonces.discard(nonce)
            del self.nonce_expiry[nonce]

    def check_permissions(
        self,
        api_key: APIKey,
        required_permission: str
    ) -> bool:
        """Check if API key has required permission"""

        return required_permission in api_key.permissions

    def get_statistics(self) -> dict[str, Any]:
        """Get API key statistics"""

        active_keys = [k for k in self.api_keys.values() if k.status == APIKeyStatus.ACTIVE]
        expired_keys = [k for k in self.api_keys.values() if k.status == APIKeyStatus.EXPIRED]
        revoked_keys = [k for k in self.api_keys.values() if k.status == APIKeyStatus.REVOKED]

        return {
            "total_keys": len(self.api_keys),
            "active_keys": len(active_keys),
            "expired_keys": len(expired_keys),
            "revoked_keys": len(revoked_keys),
            "top_users": sorted(
                [
                    {
                        "client_name": k.client_name,
                        "key_id": k.key_id,
                        "request_count": k.request_count,
                        "last_used": k.last_used.isoformat() if k.last_used else None
                    }
                    for k in active_keys
                ],
                key=lambda x: x["request_count"],
                reverse=True
            )[:10],
            "cached_nonces": len(self.used_nonces)
        }


class RequestSigningClient:
    """Client-side request signing helper"""

    def __init__(self, key_id: str, secret: str):
        self.key_id = key_id
        self.secret = secret

    def sign_request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        include_nonce: bool = True
    ) -> dict[str, str]:
        """Generate headers for signed request"""

        timestamp = str(int(time.time()))
        nonce = secrets.token_urlsafe(16) if include_nonce else None

        # Calculate signature
        signature = self._calculate_signature(
            method=method,
            url=url,
            timestamp=timestamp,
            body=body or b"",
            nonce=nonce
        )

        # Build headers
        headers = {
            SignatureValidator.SIGNATURE_HEADER: signature,
            SignatureValidator.TIMESTAMP_HEADER: timestamp,
            SignatureValidator.KEY_ID_HEADER: self.key_id
        }

        if nonce:
            headers[SignatureValidator.NONCE_HEADER] = nonce

        return headers

    def _calculate_signature(
        self,
        method: str,
        url: str,
        timestamp: str,
        body: bytes,
        nonce: str | None
    ) -> str:
        """Calculate HMAC-SHA256 signature"""

        body_hash = hashlib.sha256(body).hexdigest() if body else ""

        parts = [
            method.upper(),
            url,
            timestamp,
            body_hash
        ]

        if nonce:
            parts.append(nonce)

        string_to_sign = "\n".join(parts)

        signature = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode('utf-8')


# FastAPI dependency for request validation
security = HTTPBearer()

async def validate_api_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials = None
) -> APIKey:
    """FastAPI dependency for API request validation"""

    # Get validator from app state (set during startup)
    validator: SignatureValidator = request.app.state.signature_validator

    return await validator.validate_request(request)


# Example usage for client
if __name__ == "__main__":
    # Server setup
    validator = SignatureValidator()
    key_id, secret = validator.generate_api_key(
        client_name="Test Client",
        expires_in_days=30,
        allowed_ips=["127.0.0.1"],
        rate_limit_qps=10.0
    )

    print("Generated API Key:")
    print(f"  Key ID: {key_id}")
    print(f"  Secret: {secret}")

    # Client usage
    client = RequestSigningClient(key_id, secret)

    # Sign a request
    headers = client.sign_request(
        method="POST",
        url="https://api.example.com/query",
        body=json.dumps({"query": "test"}).encode('utf-8')
    )

    print("\nRequest Headers:")
    for name, value in headers.items():
        print(f"  {name}: {value}")
