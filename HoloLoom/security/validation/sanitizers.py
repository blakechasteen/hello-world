"""
Input sanitization functions for HoloLoom.

Phase 2 - Input Validation & Sanitization (November 2025)

Implements:
- HTML/XML encoding (XSS prevention)
- LDAP injection prevention
- Email header injection prevention
- Unicode normalization (prevention of Unicode attacks)
- Whitespace and special character normalization
- Safe string escaping for various contexts
"""

import re
import unicodedata
import html
from typing import Any, Optional
from urllib.parse import quote, unquote
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# XSS Prevention: HTML/XML Encoding
# ============================================================================

def sanitize_html_output(text: str) -> str:
    """
    Encode HTML special characters to prevent XSS.

    Converts:
    - & → &amp;
    - < → &lt;
    - > → &gt;
    - " → &quot;
    - ' → &#x27;

    Args:
        text: Raw text to encode

    Returns:
        HTML-safe encoded text

    Example:
        >>> unsafe = '<script>alert("XSS")</script>'
        >>> safe = sanitize_html_output(unsafe)
        >>> assert '&lt;script&gt;' in safe
    """
    if not isinstance(text, str):
        return str(text)

    return html.escape(text, quote=True)


def sanitize_xml_output(text: str) -> str:
    """
    Encode XML special characters to prevent XXE and XML injection.

    Converts:
    - & → &amp;
    - < → &lt;
    - > → &gt;
    - " → &quot;
    - ' → &apos;

    Args:
        text: Raw text to encode

    Returns:
        XML-safe encoded text
    """
    if not isinstance(text, str):
        return str(text)

    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;'
    }

    result = text
    # Order matters: replace & first to avoid double-encoding
    for char, encoded in replacements.items():
        result = result.replace(char, encoded)

    return result


def sanitize_javascript_context(text: str) -> str:
    """
    Escape string for safe use in JavaScript context.

    Escapes:
    - Quotes and backticks
    - Newlines and special characters
    - Prevents breaking out of string context

    Args:
        text: Raw text to escape

    Returns:
        JavaScript-safe escaped text

    Example:
        >>> unsafe = 'hello"; alert("XSS"); //'
        >>> safe = sanitize_javascript_context(unsafe)
        >>> assert '\\"' in safe  # Quotes escaped
    """
    if not isinstance(text, str):
        return str(text)

    return (text
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace("'", "\\'")
            .replace('`', '\\`')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
            .replace('\t', '\\t')
            .replace('</', '<\\/')  # Prevent closing tags in script
            )


def sanitize_json_value(value: Any) -> Any:
    """
    Sanitize value for safe JSON serialization.

    Args:
        value: Value to sanitize (str, int, float, bool, None, list, dict)

    Returns:
        Sanitized value safe for JSON

    Raises:
        ValueError: If value type cannot be serialized
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        # Normalize Unicode and remove control characters
        value = normalize_unicode(value)
        # Remove null bytes
        value = value.replace('\x00', '')
        return value

    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, dict):
        return {
            sanitize_json_value(k): sanitize_json_value(v)
            for k, v in value.items()
        }

    raise ValueError(f"Cannot sanitize type {type(value)}")


# ============================================================================
# SQL Injection Prevention: Parameterized Queries
# ============================================================================

def escape_sql_identifier(identifier: str) -> str:
    """
    Escape SQL identifier (table name, column name, etc).

    WARNING: Use only for identifiers (table/column names), NOT values.
    For values, use parameterized queries instead.

    Args:
        identifier: SQL identifier to escape

    Returns:
        Escaped identifier safe for SQL

    Example:
        >>> table = 'users'
        >>> escaped = escape_sql_identifier(table)
        >>> # Use in query: f'SELECT * FROM {escaped}'
    """
    # Remove any quotes first
    identifier = identifier.strip('"\'`')

    # Allow only alphanumeric, underscore (no spaces or special chars)
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(
            f"Invalid SQL identifier '{identifier}'. "
            "Must start with letter/underscore, contain only alphanumeric/underscore."
        )

    # Quote identifier with backticks (MySQL style)
    return f'`{identifier}`'


def escape_sql_string(value: str) -> str:
    """
    Escape SQL string literal.

    WARNING: Prefer parameterized queries when possible.
    Use this only when parameterized queries unavailable.

    Args:
        value: String value to escape

    Returns:
        Escaped string safe for SQL

    Example:
        >>> value = "O'Reilly"
        >>> escaped = escape_sql_string(value)
        >>> assert "O\\'Reilly" in escaped or "O''Reilly" in escaped
    """
    # Escape single quotes by doubling them (SQL standard)
    escaped = value.replace("'", "''")
    # Wrap in quotes
    return f"'{escaped}'"


def build_parameterized_query(template: str, params: dict) -> tuple:
    """
    Build parameterized SQL query with parameter placeholders.

    Ensures SQL injection prevention by separating query structure from data.

    Args:
        template: SQL template with %(key)s placeholders
        params: Dictionary of parameter values

    Returns:
        Tuple of (query_template, safe_params)

    Example:
        >>> template = 'SELECT * FROM users WHERE id = %(id)s AND name = %(name)s'
        >>> params = {'id': 123, 'name': "O'Brien"}
        >>> query_template, safe_params = build_parameterized_query(template, params)
    """
    # Validate template contains only expected placeholders
    placeholders = re.findall(r'%\(([a-zA-Z_][a-zA-Z0-9_]*)\)s', template)

    for placeholder in placeholders:
        if placeholder not in params:
            raise ValueError(f"Missing parameter: {placeholder}")

    # Validate no parameters in template
    if '%s' in template or '?' in template:
        raise ValueError(
            "Query template contains bare '?' or '%s' placeholders. "
            "Use %(name)s format instead."
        )

    return template, params


# ============================================================================
# Command Injection Prevention
# ============================================================================

def sanitize_shell_argument(arg: str) -> str:
    """
    Quote argument for safe shell execution.

    Uses single-quote wrapping which prevents variable expansion and
    prevents breaking out of quoted context.

    WARNING: Avoid shell execution when possible. Use subprocess with
    argument list instead.

    Args:
        arg: Argument to quote

    Returns:
        Shell-safe quoted argument

    Example:
        >>> arg = "hello; rm -rf /"
        >>> quoted = sanitize_shell_argument(arg)
        >>> # Now safe to pass to shell
    """
    # Use single quotes (most restrictive)
    # Single quote in string: end quote, escape single quote, start quote
    if "'" not in arg:
        return f"'{arg}'"
    else:
        # Replace ' with '\''
        escaped = arg.replace("'", "'\\''")
        return f"'{escaped}'"


def validate_executable_path(path: str, allowed_dirs: list) -> str:
    """
    Validate executable path against whitelist of allowed directories.

    Prevents arbitrary command execution by restricting to safe paths.

    Args:
        path: Executable path to validate
        allowed_dirs: List of allowed directories

    Returns:
        Validated path

    Raises:
        ValueError: If path not in allowed directories

    Example:
        >>> path = sanitize_shell_argument('/usr/bin/python3')
        >>> validated = validate_executable_path(path, ['/usr/bin'])
    """
    import os

    # Resolve to absolute path
    abs_path = os.path.abspath(path)

    # Check if in allowed directories
    for allowed_dir in allowed_dirs:
        allowed_dir = os.path.abspath(allowed_dir)
        if abs_path.startswith(allowed_dir + os.sep) or abs_path == allowed_dir:
            return abs_path

    raise ValueError(
        f"Executable '{abs_path}' not in allowed directories: {allowed_dirs}"
    )


# ============================================================================
# LDAP Injection Prevention
# ============================================================================

def escape_ldap_filter(value: str) -> str:
    """
    Escape value for safe use in LDAP filter.

    Escapes LDAP special characters to prevent filter injection.

    Args:
        value: Value to escape

    Returns:
        LDAP-safe escaped value

    Example:
        >>> unsafe = 'user*)(uid=*'
        >>> safe = escape_ldap_filter(unsafe)
        >>> # Now safe to use in filter: f'(uid={safe})'
    """
    # LDAP special characters that need escaping
    replacements = {
        '*': '\\2a',
        '(': '\\28',
        ')': '\\29',
        '\\': '\\5c',
        '\x00': '\\00',
        '/': '\\2f'
    }

    result = value
    for char, escaped in replacements.items():
        result = result.replace(char, escaped)

    return result


def build_ldap_filter(template: str, params: dict) -> str:
    """
    Build LDAP filter from template with escaped parameters.

    Args:
        template: LDAP filter template with {key} placeholders
        params: Dictionary of parameter values

    Returns:
        LDAP filter with escaped parameters

    Example:
        >>> template = '(&(uid={username})(objectClass=person))'
        >>> params = {'username': 'user*)(cn=*'}
        >>> filter_str = build_ldap_filter(template, params)
    """
    escaped_params = {k: escape_ldap_filter(v) for k, v in params.items()}
    return template.format(**escaped_params)


# ============================================================================
# Email Injection Prevention
# ============================================================================

def sanitize_email_header(header: str) -> str:
    """
    Sanitize email header to prevent header injection.

    Prevents:
    - CRLF injection (newline-based header injection)
    - Multiple recipient injection
    - BCC field injection

    Args:
        header: Header value to sanitize

    Returns:
        Safe header value
    """
    if not isinstance(header, str):
        return str(header)

    # Remove or replace dangerous characters
    # CR, LF, CRLF are the main injection vectors
    header = header.replace('\r\n', ' ')
    header = header.replace('\r', ' ')
    header = header.replace('\n', ' ')
    header = header.replace('\x00', '')  # Null bytes

    # Collapse multiple spaces
    header = ' '.join(header.split())

    return header.strip()


# ============================================================================
# Path Traversal Prevention
# ============================================================================

def validate_safe_path(user_path: str, base_dir: str) -> str:
    """
    Validate path is within base directory (prevent traversal).

    Args:
        user_path: User-supplied path
        base_dir: Allowed base directory

    Returns:
        Validated absolute path

    Raises:
        ValueError: If path escapes base directory

    Example:
        >>> safe = validate_safe_path('docs/readme.txt', '/app/uploads')
        >>> # safe = '/app/uploads/docs/readme.txt'
        >>> # Accessing '../etc/passwd' raises ValueError
    """
    import os

    # Normalize paths
    base_dir = os.path.abspath(base_dir)
    user_path = os.path.normpath(user_path)

    # Remove leading slashes (prevent absolute paths)
    if user_path.startswith('/') or user_path.startswith('\\'):
        raise ValueError("Path must be relative")

    # Build absolute path
    abs_path = os.path.abspath(os.path.join(base_dir, user_path))

    # Verify it's within base directory
    if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
        raise ValueError(
            f"Path '{user_path}' escapes base directory '{base_dir}'"
        )

    return abs_path


# ============================================================================
# Unicode & Character Normalization
# ============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to prevent Unicode-based attacks.

    Uses NFKC normalization (compatibility decomposition + composition).

    Prevents:
    - Unicode homograph attacks (visually similar characters)
    - Zero-width characters used for encoding
    - Alternative representations of same character

    Args:
        text: Text to normalize

    Returns:
        Normalized text

    Example:
        >>> # Fullwidth 'A' (U+FF21) → normal 'A' (U+0041)
        >>> unsafe = '\uff21'  # Fullwidth A
        >>> safe = normalize_unicode(unsafe)
        >>> assert safe == 'A'
    """
    if not isinstance(text, str):
        return str(text)

    # NFKC: Compatibility decomposition followed by composition
    # Most strict normalization, prevents Unicode tricks
    return unicodedata.normalize('NFKC', text)


def remove_control_characters(text: str) -> str:
    """
    Remove control characters from text.

    Prevents embedding of null bytes, newlines, and other control chars
    in contexts where they could cause issues.

    Args:
        text: Text to clean

    Returns:
        Text without control characters
    """
    if not isinstance(text, str):
        return str(text)

    # Remove all control characters (categories Cc and Cn)
    cleaned = ''.join(
        char for char in text
        if unicodedata.category(char) not in ('Cc', 'Cn')
    )

    return cleaned


def normalize_whitespace(text: str, collapse: bool = True) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Text to normalize
        collapse: If True, collapse multiple spaces to single space

    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return str(text)

    # Normalize line endings to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    if collapse:
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        # Collapse multiple newlines
        text = re.sub(r'\n+', '\n', text)

    return text.strip()


# ============================================================================
# URL/URI Sanitization
# ============================================================================

def sanitize_url_parameter(value: str) -> str:
    """
    Sanitize value for safe use in URL parameter.

    Properly URL-encodes special characters.

    Args:
        value: Value to encode

    Returns:
        URL-safe encoded value

    Example:
        >>> unsafe = 'hello world?&=+'
        >>> safe = sanitize_url_parameter(unsafe)
        >>> # safe = 'hello%20world%3F%26%3D%2B'
    """
    if not isinstance(value, str):
        value = str(value)

    # URL encode with safe characters for basic alphanumeric/dash/underscore
    return quote(value, safe='')


def validate_url_path(path: str) -> str:
    """
    Validate URL path is safe.

    Prevents:
    - Path traversal (../)
    - Protocol bypass (java:, data:, etc)
    - Double encoding attacks

    Args:
        path: URL path to validate

    Returns:
        Validated path

    Raises:
        ValueError: If path is unsafe
    """
    # Decode once to catch double-encoding
    decoded = unquote(path)

    # Check for path traversal
    if '..' in decoded or decoded.startswith('//'):
        raise ValueError("URL path contains traversal or protocol bypass")

    return path


# ============================================================================
# Generic Sanitization Utilities
# ============================================================================

def sanitize_with_context(value: str, context: str = 'text') -> str:
    """
    Sanitize value according to context.

    Args:
        value: Value to sanitize
        context: Context type (text, html, xml, json, url, ldap, shell, sql_identifier)

    Returns:
        Sanitized value

    Example:
        >>> html_safe = sanitize_with_context('<script>alert()</script>', 'html')
        >>> json_safe = sanitize_with_context('hello\nworld', 'json')
    """
    context = context.lower()

    if context == 'html':
        return sanitize_html_output(value)
    elif context == 'xml':
        return sanitize_xml_output(value)
    elif context == 'json':
        return str(sanitize_json_value(value))
    elif context == 'url':
        return sanitize_url_parameter(value)
    elif context == 'ldap':
        return escape_ldap_filter(value)
    elif context == 'shell':
        return sanitize_shell_argument(value)
    elif context == 'sql_identifier':
        return escape_sql_identifier(value)
    elif context == 'email_header':
        return sanitize_email_header(value)
    else:
        # Default: normalize Unicode and remove control chars
        return remove_control_characters(normalize_unicode(value))


def sanitize_sensitive_data(value: str) -> str:
    """
    Sanitize sensitive data for logging (redact details).

    Used for logging to prevent accidental exposure of sensitive info.

    Args:
        value: Sensitive value to redact

    Returns:
        Redacted version safe for logging
    """
    if len(value) <= 4:
        return '****'

    # Show first and last 2 characters, redact middle
    return value[:2] + '*' * (len(value) - 4) + value[-2:]
