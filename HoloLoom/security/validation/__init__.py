"""
HoloLoom Validation & Sanitization Module

Phase 2 - Input Validation & Sanitization (November 2025)

Comprehensive input validation and output encoding to prevent:
- SQL Injection (parameterized queries)
- XSS (HTML/XML/JavaScript encoding)
- Command Injection (argument sanitization)
- LDAP Injection (special character escaping)
- Path Traversal (safe path validation)
- File Upload attacks (MIME validation, size limits, virus scanning)
- XML Injection (safe XML handling)
- Email Header Injection (CRLF removal)
- Unicode attacks (NFKC normalization)

Usage:
    # Schema validation
    from HoloLoom.security.validation.schemas import QueryRequest
    query = QueryRequest(text="What is Thompson Sampling?")

    # Output encoding
    from HoloLoom.security.validation.sanitizers import sanitize_html_output
    safe = sanitize_html_output('<script>alert("XSS")</script>')

    # File upload validation
    from HoloLoom.security.validation.file_upload import validate_file_upload
    result = validate_file_upload('file.pdf', file_bytes, 'application/pdf')

    # SQL injection prevention
    from HoloLoom.security.validation.sql_safe import build_select_query
    query, params = build_select_query('users', ['id', 'name'], {'active': True})
"""

from HoloLoom.security.validation.schemas import (
    # Validation schemas
    QueryRequest,
    UserIdentifier,
    AuthenticationRequest,
    URLRequest,
    FileMetadata,
    JSONRequest,
    SearchRequest,
    BatchQueryRequest,
    ConfigUpdateRequest,
    # Response models
    ValidationError,
    ValidationSuccess,
)

from HoloLoom.security.validation.sanitizers import (
    # XSS prevention
    sanitize_html_output,
    sanitize_xml_output,
    sanitize_javascript_context,
    sanitize_json_value,
    # SQL injection prevention
    escape_sql_identifier,
    escape_sql_string,
    build_parameterized_query,
    # Command injection prevention
    sanitize_shell_argument,
    validate_executable_path,
    # LDAP injection prevention
    escape_ldap_filter,
    build_ldap_filter,
    # Email header injection prevention
    sanitize_email_header,
    # Path traversal prevention
    validate_safe_path,
    # Unicode & character normalization
    normalize_unicode,
    remove_control_characters,
    normalize_whitespace,
    # URL sanitization
    sanitize_url_parameter,
    validate_url_path,
    # Generic utilities
    sanitize_with_context,
    sanitize_sensitive_data,
)

from HoloLoom.security.validation.file_upload import (
    # MIME type validation
    detect_mime_type_from_magic,
    validate_mime_type,
    is_dangerous_mime_type,
    # File size validation
    validate_file_size,
    # Hash & integrity
    calculate_sha256,
    calculate_sha256_bytes,
    verify_file_hash,
    # Content scanning
    scan_with_clamav,
    scan_for_suspicious_patterns,
    # Filename handling
    generate_safe_filename,
    validate_filename,
    # Main validation
    validate_file_upload,
    FileValidationResult,
    # File storage
    save_uploaded_file,
)

from HoloLoom.security.validation.sql_safe import (
    # Data types & operators
    SQLDataType,
    SQLOperator,
    SQLOrder,
    # Parameter handling
    validate_parameter,
    escape_sql_identifier as escape_sql_id,
    escape_sql_column_list,
    escape_sql_value,
    # Query builder
    SQLQueryBuilder,
    # Common patterns
    build_select_query,
    build_insert_query,
    build_update_query,
    build_delete_query,
    # Detection
    detect_sql_injection_attempt,
)

__all__ = [
    # Schemas
    'QueryRequest',
    'UserIdentifier',
    'AuthenticationRequest',
    'URLRequest',
    'FileMetadata',
    'JSONRequest',
    'SearchRequest',
    'BatchQueryRequest',
    'ConfigUpdateRequest',
    'ValidationError',
    'ValidationSuccess',
    # Sanitizers
    'sanitize_html_output',
    'sanitize_xml_output',
    'sanitize_javascript_context',
    'sanitize_json_value',
    'escape_sql_identifier',
    'escape_sql_string',
    'build_parameterized_query',
    'sanitize_shell_argument',
    'validate_executable_path',
    'escape_ldap_filter',
    'build_ldap_filter',
    'sanitize_email_header',
    'validate_safe_path',
    'normalize_unicode',
    'remove_control_characters',
    'normalize_whitespace',
    'sanitize_url_parameter',
    'validate_url_path',
    'sanitize_with_context',
    'sanitize_sensitive_data',
    # File upload
    'detect_mime_type_from_magic',
    'validate_mime_type',
    'is_dangerous_mime_type',
    'validate_file_size',
    'calculate_sha256',
    'calculate_sha256_bytes',
    'verify_file_hash',
    'scan_with_clamav',
    'scan_for_suspicious_patterns',
    'generate_safe_filename',
    'validate_filename',
    'validate_file_upload',
    'FileValidationResult',
    'save_uploaded_file',
    # SQL safe
    'SQLDataType',
    'SQLOperator',
    'SQLOrder',
    'validate_parameter',
    'escape_sql_id',
    'escape_sql_column_list',
    'escape_sql_value',
    'SQLQueryBuilder',
    'build_select_query',
    'build_insert_query',
    'build_update_query',
    'build_delete_query',
    'detect_sql_injection_attempt',
]
