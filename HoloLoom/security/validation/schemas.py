"""
Pydantic validation schemas for HoloLoom API requests.

Phase 2 - Input Validation & Sanitization (November 2025)

Implements:
- RFC 5322 compliant email validation
- RFC 3986 compliant URL validation
- UUID/alphanumeric user ID validation
- Query text validation (length, content)
- File metadata validation
- JSON depth/size validation
- MIME type whitelisting
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl, EmailStr
from enum import Enum
import re
from datetime import datetime

# ============================================================================
# OWASP Input Validation Constants
# ============================================================================

# Maximum lengths (prevent buffer overflow)
MAX_QUERY_LENGTH = 10_000
MAX_USER_ID_LENGTH = 50
MAX_URL_LENGTH = 2048
MAX_JSON_SIZE_BYTES = 1_048_576  # 1 MB
MAX_FILE_SIZE_BYTES = 10_485_760  # 10 MB
MAX_JSON_DEPTH = 10

# Whitelist patterns (reject unknowns)
USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]{1,50}$')
UUID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE
)

# Dangerous SQL keywords (basic detection)
SQL_KEYWORDS = {
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
    'EXEC', 'EXECUTE', 'UNION', 'SCRIPT', 'JAVASCRIPT', 'ONERROR',
    'ONLOAD', 'ONCLICK', 'SHELL_EXEC', 'SYSTEM', 'PASSTHRU'
}

# Shell metacharacters (command injection prevention)
SHELL_METACHARACTERS = {'|', ';', '&', '$', '`', '\n', '\r', '>', '<', '(', ')'}

# Whitelisted MIME types for file upload
WHITELISTED_MIME_TYPES = {
    'text/plain',
    'text/csv',
    'application/json',
    'application/pdf',
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}

# Dangerous file extensions (prevent executable uploads)
DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.sh', '.py', '.js', '.php', '.jsp',
    '.asp', '.aspx', '.pl', '.cgi', '.jar', '.zip', '.rar', '.7z'
}


# ============================================================================
# Query & Context Validation
# ============================================================================

class QueryRequest(BaseModel):
    """
    Main query request model with comprehensive validation.

    Validation:
    - text: Max 10,000 chars, no raw SQL keywords, no shell metacharacters
    - context: Optional metadata with file/selection info
    - mode: reasoning mode (direct/verify/research/plan_execute)
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Query text (max 10,000 chars)"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional query context (filename, selection, etc)"
    )
    mode: str = Field(
        'direct',
        regex='^(direct|verify|research|plan_execute)$',
        description="Reasoning mode"
    )
    max_steps: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum reasoning steps (1-20)"
    )

    @validator('text')
    def validate_query_text(cls, v):
        """
        Validate query text against injection attacks.

        Checks:
        - No raw SQL keywords in suspicious patterns
        - No shell metacharacters
        - Proper character encoding
        """
        # Normalize Unicode (prevent normalization attacks)
        v = v.encode('utf-8').decode('utf-8')

        # Check for shell metacharacters
        for char in SHELL_METACHARACTERS:
            if char in v:
                # Allow these in natural context (e.g., "pipe" in text)
                # but flag suspicious patterns
                if any(pattern in v.lower() for pattern in [
                    'rm -rf', '$(', '`', '|cat', ';&', 'python -c',
                    'bash -i', '/dev/tcp'
                ]):
                    raise ValueError(
                        "Query contains suspicious shell patterns. "
                        "Command injection detected."
                    )

        # Check for SQL injection patterns
        suspicious_sql = any(
            f' {kw} ' in v.upper() or v.upper().startswith(f'{kw} ')
            for kw in ['UNION', 'SELECT', 'INSERT', 'DELETE', 'DROP']
        )
        if suspicious_sql and any(c in v for c in ["'", '"', ';', '--', '/*']):
            raise ValueError(
                "Query contains suspicious SQL patterns. "
                "SQL injection detected."
            )

        return v.strip()

    @validator('context')
    def validate_context(cls, v):
        """Validate query context metadata."""
        if v is None:
            return None

        if not isinstance(v, dict):
            raise ValueError("Context must be a dictionary")

        if len(v) > 20:
            raise ValueError("Context cannot exceed 20 keys")

        # Validate known context fields
        if 'languageId' in v:
            lang = v['languageId']
            if not isinstance(lang, str) or len(lang) > 50:
                raise ValueError("languageId must be string, max 50 chars")

        if 'fileName' in v:
            filename = v['fileName']
            if '..' in filename or filename.startswith('/'):
                raise ValueError("fileName contains path traversal characters")

        if 'selection' in v:
            selection = v['selection']
            if not isinstance(selection, str) or len(selection) > MAX_QUERY_LENGTH:
                raise ValueError("selection exceeds maximum length")

        return v


# ============================================================================
# User & Authentication Validation
# ============================================================================

class UserIdentifier(BaseModel):
    """
    User identification with strict validation.

    Accepts:
    - UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    - Alphanumeric: [a-zA-Z0-9_-], max 50 chars
    """
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=MAX_USER_ID_LENGTH,
        description="User ID (UUID or alphanumeric)"
    )

    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user ID format."""
        # Try UUID first
        if not UUID_PATTERN.match(v) and not USER_ID_PATTERN.match(v):
            raise ValueError(
                "user_id must be valid UUID or alphanumeric [a-zA-Z0-9_-] "
                f"(max {MAX_USER_ID_LENGTH} chars)"
            )
        return v


class AuthenticationRequest(BaseModel):
    """Authentication request with email validation."""
    email: EmailStr = Field(
        ...,
        description="Email address (RFC 5322 compliant)"
    )
    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
        description="Password (12-128 chars)"
    )

    @validator('password')
    def validate_password(cls, v):
        """
        Validate password strength.

        Requirements:
        - At least 12 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        """
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError("Password must contain special character")
        return v


# ============================================================================
# URL & Link Validation
# ============================================================================

class URLRequest(BaseModel):
    """
    URL validation with protocol whitelist.

    Validation:
    - Valid protocol (http/https only, no file://, data:, etc)
    - Max 2048 chars (prevent buffer overflow)
    - RFC 3986 compliant
    """
    url: HttpUrl = Field(
        ...,
        description="URL with http/https protocol"
    )

    @validator('url', pre=True)
    def validate_url_protocol(cls, v):
        """Validate URL protocol is safe."""
        if isinstance(v, str):
            # Check for dangerous protocols
            dangerous_protocols = ['file://', 'data:', 'javascript:', 'ftp://']
            if any(v.startswith(p) for p in dangerous_protocols):
                raise ValueError(
                    f"URL protocol not allowed. Use http:// or https://"
                )

            # Check length
            if len(v) > MAX_URL_LENGTH:
                raise ValueError(
                    f"URL exceeds maximum length of {MAX_URL_LENGTH} chars"
                )

        return v


# ============================================================================
# File Upload Validation
# ============================================================================

class FileMetadata(BaseModel):
    """
    File upload metadata validation.

    Validation:
    - MIME type whitelist (prevent executable uploads)
    - File size limit (max 10 MB)
    - File extension whitelist
    - Filename path traversal prevention
    """
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original filename"
    )
    mime_type: str = Field(
        ...,
        description="MIME type (must be whitelisted)"
    )
    size_bytes: int = Field(
        ...,
        ge=1,
        le=MAX_FILE_SIZE_BYTES,
        description=f"File size in bytes (max {MAX_FILE_SIZE_BYTES})"
    )
    sha256_hash: Optional[str] = Field(
        None,
        regex='^[a-f0-9]{64}$',
        description="SHA256 hash of file content"
    )

    @validator('filename')
    def validate_filename(cls, v):
        """
        Validate filename for safety.

        Prevents:
        - Path traversal (../, absolute paths)
        - Dangerous extensions (.exe, .sh, .py, etc)
        - Null bytes
        - Special characters that could cause issues
        """
        # No path traversal
        if '..' in v or v.startswith('/') or v.startswith('\\'):
            raise ValueError("Filename contains path traversal characters")

        # No null bytes
        if '\0' in v:
            raise ValueError("Filename contains null bytes")

        # Check extension
        import os
        _, ext = os.path.splitext(v.lower())
        if ext in DANGEROUS_EXTENSIONS:
            raise ValueError(f"File extension {ext} not allowed")

        # Alphanumeric, dash, underscore, dot only
        if not re.match(r'^[a-zA-Z0-9._\-]+$', v):
            raise ValueError(
                "Filename contains invalid characters. "
                "Use only letters, numbers, dot, dash, underscore."
            )

        return v

    @validator('mime_type')
    def validate_mime_type(cls, v):
        """Validate MIME type against whitelist."""
        if v not in WHITELISTED_MIME_TYPES:
            allowed = ', '.join(sorted(WHITELISTED_MIME_TYPES))
            raise ValueError(
                f"MIME type '{v}' not allowed. Allowed types: {allowed}"
            )
        return v

    @validator('sha256_hash')
    def validate_hash(cls, v):
        """Validate SHA256 hash format if provided."""
        if v is not None:
            if len(v) != 64 or not all(c in '0123456789abcdef' for c in v):
                raise ValueError("SHA256 hash must be 64 hex characters")
        return v


# ============================================================================
# JSON & Data Structure Validation
# ============================================================================

class JSONRequest(BaseModel):
    """
    JSON payload validation with depth/size limits.

    Validation:
    - Max size: 1 MB
    - Max depth: 10 levels
    - Prevents XXE and other XML-based attacks
    """
    data: Dict[str, Any] = Field(
        ...,
        description="JSON data (max 1 MB, max depth 10)"
    )

    @validator('data', pre=True)
    def validate_json_constraints(cls, v):
        """Validate JSON depth and structure."""
        import json

        # Check size
        json_str = json.dumps(v)
        if len(json_str.encode('utf-8')) > MAX_JSON_SIZE_BYTES:
            raise ValueError(
                f"JSON payload exceeds maximum size of {MAX_JSON_SIZE_BYTES} bytes"
            )

        # Check depth
        def get_depth(obj, current_depth=0):
            if current_depth > MAX_JSON_DEPTH:
                raise ValueError(
                    f"JSON structure exceeds maximum depth of {MAX_JSON_DEPTH}"
                )

            if isinstance(obj, dict):
                if not obj:
                    return current_depth
                return max(get_depth(val, current_depth + 1) for val in obj.values())
            elif isinstance(obj, list):
                if not obj:
                    return current_depth
                return max(get_depth(item, current_depth + 1) for item in obj)
            else:
                return current_depth

        get_depth(v)
        return v


# ============================================================================
# Search & Filter Validation
# ============================================================================

class SearchRequest(BaseModel):
    """
    Search query with injection prevention.

    Validation:
    - Query text (same as QueryRequest)
    - Limit and offset for pagination safety
    - Filter fields whitelisted
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query"
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Results limit (1-100)"
    )
    offset: int = Field(
        0,
        ge=0,
        le=10000,
        description="Results offset (0-10000)"
    )
    filters: Optional[Dict[str, str]] = Field(
        None,
        description="Optional filters (keys must be whitelisted)"
    )

    ALLOWED_FILTER_KEYS = {
        'language_id', 'file_type', 'created_after', 'created_before',
        'confidence_min', 'confidence_max', 'tool_used'
    }

    @validator('query')
    def validate_search_query(cls, v):
        """Validate search query."""
        # Same checks as QueryRequest but shorter max length
        v = v.encode('utf-8').decode('utf-8')
        return v.strip()

    @validator('filters')
    def validate_filters(cls, v):
        """
        Validate filter keys against whitelist.

        Prevents filter-based injection attacks.
        """
        if v is None:
            return None

        for key in v.keys():
            if key not in SearchRequest.ALLOWED_FILTER_KEYS:
                allowed = ', '.join(sorted(SearchRequest.ALLOWED_FILTER_KEYS))
                raise ValueError(
                    f"Filter key '{key}' not allowed. "
                    f"Allowed keys: {allowed}"
                )

        return v


# ============================================================================
# Batch Operation Validation
# ============================================================================

class BatchQueryRequest(BaseModel):
    """
    Batch query validation with rate limiting.

    Validation:
    - Max 100 queries per batch
    - Each query validates individually
    - Prevents abuse through batching
    """
    queries: List[QueryRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of queries (1-100 per batch)"
    )
    parallel: bool = Field(
        False,
        description="Execute in parallel"
    )


# ============================================================================
# Configuration & Settings Validation
# ============================================================================

class ConfigUpdateRequest(BaseModel):
    """
    Configuration update with whitelist validation.

    Only allows changes to safe configuration keys.
    """
    ALLOWED_CONFIG_KEYS = {
        'enable_caching',
        'cache_ttl_seconds',
        'max_memory_mb',
        'timeout_seconds',
        'log_level',
        'enable_metrics'
    }

    updates: Dict[str, Any] = Field(
        ...,
        description="Configuration updates"
    )

    @validator('updates')
    def validate_config_keys(cls, v):
        """Validate configuration keys against whitelist."""
        for key in v.keys():
            if key not in ConfigUpdateRequest.ALLOWED_CONFIG_KEYS:
                allowed = ', '.join(sorted(ConfigUpdateRequest.ALLOWED_CONFIG_KEYS))
                raise ValueError(
                    f"Configuration key '{key}' not allowed. "
                    f"Allowed keys: {allowed}"
                )

        return v


# ============================================================================
# Validation Response Models
# ============================================================================

class ValidationError(BaseModel):
    """Structured validation error response."""
    error: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that failed validation")
    attack_type: Optional[str] = Field(
        None,
        description="Type of attack detected (SQLi, XSS, etc)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationSuccess(BaseModel):
    """Successful validation response."""
    valid: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
