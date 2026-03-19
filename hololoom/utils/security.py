"""
Security Utilities
==================

Common security functions for the HoloLoom codebase.

Functions:
    sanitize_uri: Mask credentials in connection URIs for safe logging
    mask_api_key: Mask API keys in strings for safe logging
"""

import re


def sanitize_uri(uri: str | None) -> str:
    """
    Mask credentials in connection URIs for safe logging.

    Handles common database/service URI formats:
    - bolt://user:password@host:port
    - redis://user:password@host:port
    - mysql://user:password@host:port/db
    - postgres://user:password@host:port/db
    - mongodb://user:password@host:port/db
    - http://user:password@host/path

    Args:
        uri: Connection URI that may contain credentials

    Returns:
        URI with password masked as '****', or empty string if None

    Examples:
        >>> sanitize_uri("bolt://neo4j:secret123@localhost:7687")
        'bolt://neo4j:****@localhost:7687'

        >>> sanitize_uri("redis://:mypassword@redis.example.com:6379")
        'redis://:****@redis.example.com:6379'

        >>> sanitize_uri("bolt://localhost:7687")  # No credentials
        'bolt://localhost:7687'

    SECURITY: Always use this function before logging connection URIs to
    prevent credential leakage in logs.
    """
    if not uri:
        return ""

    # Pattern matches: protocol://[user][:password]@host
    # Captures: user (optional), password, and masks the password
    # Works with:
    #   - bolt://user:pass@host
    #   - redis://:pass@host (no username)
    #   - mysql://user:pass@host/db?params
    return re.sub(
        r'://([^:]*):([^@]+)@',
        r'://\1:****@',
        uri
    )


def mask_api_key(text: str) -> str:
    """
    Mask API keys and secrets in text for safe logging.

    Patterns masked:
    - sk-[40+ chars] (OpenAI API keys)
    - key-[chars] (Generic API keys)
    - Bearer [token] (Auth headers)
    - api_key=value, api-key=value
    - secret=value, password=value

    Args:
        text: Text that may contain API keys or secrets

    Returns:
        Text with API keys and secrets masked

    SECURITY: Use this function before logging error messages or
    response bodies that may contain sensitive information.
    """
    if not text:
        return ""

    patterns = [
        # OpenAI-style API keys (sk-xxx...)
        (r'sk-[a-zA-Z0-9]{20,}', 'sk-****'),
        # Generic API keys
        (r'key-[a-zA-Z0-9]{10,}', 'key-****'),
        # Bearer tokens
        (r'Bearer\s+[a-zA-Z0-9._-]+', 'Bearer ****'),
        # API key parameters
        (r'api[_-]?key[=:]\s*[^\s,&]+', 'api_key=****'),
        # Secret parameters
        (r'secret[=:]\s*[^\s,&]+', 'secret=****'),
        # Password parameters
        (r'password[=:]\s*[^\s,&]+', 'password=****'),
        # Authorization headers
        (r'Authorization:\s*[^\r\n]+', 'Authorization: ****'),
    ]

    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def is_safe_path(user_path: str, allowed_base: str) -> bool:
    """
    Check if a path is safe (no path traversal).

    Args:
        user_path: User-provided path
        allowed_base: Base directory that must contain the path

    Returns:
        True if path is safe, False if path traversal detected

    SECURITY: Use this to validate user-provided file paths before
    file operations.
    """
    from pathlib import Path

    try:
        # Resolve both paths to absolute
        base = Path(allowed_base).resolve()
        target = (base / user_path).resolve()

        # Check if target is within base
        return str(target).startswith(str(base))
    except Exception:
        return False
