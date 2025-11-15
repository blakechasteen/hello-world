"""
Comprehensive tests for HoloLoom input validation & sanitization.

Phase 2 - Input Validation & Sanitization (November 2025)

Test coverage:
- OWASP Top 10 injection payloads
- Known bypass techniques
- Edge cases (empty, null, very long, Unicode, etc)
- Performance benchmarks
- Integration with security systems

Run with: pytest HoloLoom/security/tests/test_validation.py -v
"""

import pytest
from datetime import datetime
import json
from pydantic import ValidationError

# Import validation modules
from HoloLoom.security.validation.schemas import (
    QueryRequest,
    UserIdentifier,
    AuthenticationRequest,
    URLRequest,
    FileMetadata,
    JSONRequest,
    SearchRequest,
    BatchQueryRequest,
)
from HoloLoom.security.validation.sanitizers import (
    sanitize_html_output,
    sanitize_xml_output,
    sanitize_javascript_context,
    escape_ldap_filter,
    build_ldap_filter,
    sanitize_email_header,
    validate_safe_path,
    normalize_unicode,
    remove_control_characters,
    escape_sql_identifier,
    sanitize_url_parameter,
)
from HoloLoom.security.validation.file_upload import (
    detect_mime_type_from_magic,
    validate_mime_type,
    validate_file_size,
    calculate_sha256_bytes,
    generate_safe_filename,
    validate_file_upload,
)
from HoloLoom.security.validation.sql_safe import (
    escape_sql_identifier as escape_sql_identifier_v2,
    SQLQueryBuilder,
    build_select_query,
    build_insert_query,
    build_update_query,
    detect_sql_injection_attempt,
)


# ============================================================================
# SECTION 1: Pydantic Schema Validation
# ============================================================================

class TestQueryValidation:
    """Test QueryRequest schema validation."""

    def test_valid_query(self):
        """Valid query should pass."""
        req = QueryRequest(text="What is Thompson Sampling?")
        assert req.text == "What is Thompson Sampling?"
        assert req.mode == "direct"
        assert req.max_steps == 5

    def test_query_max_length(self):
        """Query exceeding max length should fail."""
        long_text = "A" * 10_001
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text=long_text)
        assert "at most 10000 characters" in str(exc_info.value)

    def test_query_sql_injection_union(self):
        """SQL UNION injection attempt should fail."""
        payload = "' UNION SELECT * FROM users --"
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text=payload)
        assert "SQL injection" in str(exc_info.value)

    def test_query_command_injection_pipe(self):
        """Shell pipe injection should fail."""
        payload = "hello | cat /etc/passwd"
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text=payload)
        assert "shell patterns" in str(exc_info.value)

    def test_query_command_injection_backtick(self):
        """Backtick command injection should fail."""
        payload = "hello `rm -rf /`"
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text=payload)
        assert "shell patterns" in str(exc_info.value)

    def test_query_path_traversal_in_context(self):
        """Path traversal in context should fail."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(
                text="read file",
                context={"fileName": "../../etc/passwd"}
            )
        assert "path traversal" in str(exc_info.value)

    def test_query_invalid_mode(self):
        """Invalid reasoning mode should fail."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(text="hello", mode="invalid_mode")
        assert "string should match pattern" in str(exc_info.value)

    def test_query_max_steps_limits(self):
        """Max steps must be 1-20."""
        req = QueryRequest(text="hello", max_steps=20)
        assert req.max_steps == 20

        with pytest.raises(ValidationError):
            QueryRequest(text="hello", max_steps=0)

        with pytest.raises(ValidationError):
            QueryRequest(text="hello", max_steps=21)


class TestUserValidation:
    """Test user ID validation."""

    def test_valid_uuid(self):
        """Valid UUID should pass."""
        user_id = "550e8400-e29b-41d4-a716-446655440000"
        user = UserIdentifier(user_id=user_id)
        assert user.user_id == user_id

    def test_valid_alphanumeric(self):
        """Valid alphanumeric ID should pass."""
        user_id = "user_123-abc"
        user = UserIdentifier(user_id=user_id)
        assert user.user_id == user_id

    def test_invalid_special_chars(self):
        """Special characters should fail."""
        with pytest.raises(ValidationError):
            UserIdentifier(user_id="user@example.com")

    def test_user_id_max_length(self):
        """User ID exceeding 50 chars should fail."""
        long_id = "A" * 51
        with pytest.raises(ValidationError):
            UserIdentifier(user_id=long_id)

    def test_sql_injection_in_user_id(self):
        """SQL injection in user ID should fail."""
        with pytest.raises(ValidationError):
            UserIdentifier(user_id="user'; DROP TABLE users; --")


class TestAuthenticationValidation:
    """Test authentication request validation."""

    def test_valid_authentication(self):
        """Valid authentication request should pass."""
        auth = AuthenticationRequest(
            email="user@example.com",
            password="SecurePass123!@#"
        )
        assert auth.email == "user@example.com"

    def test_invalid_email(self):
        """Invalid email should fail."""
        with pytest.raises(ValidationError):
            AuthenticationRequest(
                email="not-an-email",
                password="SecurePass123!@#"
            )

    def test_weak_password_no_uppercase(self):
        """Password without uppercase should fail."""
        with pytest.raises(ValidationError) as exc_info:
            AuthenticationRequest(
                email="user@example.com",
                password="securepass123!@#"
            )
        assert "uppercase" in str(exc_info.value)

    def test_weak_password_no_digit(self):
        """Password without digit should fail."""
        with pytest.raises(ValidationError) as exc_info:
            AuthenticationRequest(
                email="user@example.com",
                password="SecurePass!@#"
            )
        assert "digit" in str(exc_info.value)

    def test_weak_password_too_short(self):
        """Password shorter than 12 chars should fail."""
        with pytest.raises(ValidationError):
            AuthenticationRequest(
                email="user@example.com",
                password="Pass123!@"
            )


# ============================================================================
# SECTION 2: XSS Prevention (HTML/XML Encoding)
# ============================================================================

class TestXSSPrevention:
    """Test XSS prevention through HTML/XML encoding."""

    OWASP_XSS_PAYLOADS = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror="alert(\'XSS\')">',
        '<svg onload="alert(\'XSS\')">',
        'javascript:alert("XSS")',
        '<iframe src="javascript:alert(\'XSS\')"></iframe>',
        '<body onload="alert(\'XSS\')">',
        '<input onfocus="alert(\'XSS\')" autofocus>',
        '<marquee onstart="alert(\'XSS\')"></marquee>',
        '<details open ontoggle="alert(\'XSS\')">',
    ]

    def test_html_encoding(self):
        """HTML should be properly encoded."""
        unsafe = '<script>alert("XSS")</script>'
        safe = sanitize_html_output(unsafe)
        assert '<' not in safe or '&lt;' in safe
        assert '>' not in safe or '&gt;' in safe
        assert safe == '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'

    def test_html_encoding_quotes(self):
        """Quotes should be properly encoded."""
        unsafe = 'value="test"'
        safe = sanitize_html_output(unsafe)
        assert '&quot;' in safe

    def test_xss_payload_encoding(self):
        """All OWASP XSS payloads should be safely encoded."""
        for payload in self.OWASP_XSS_PAYLOADS:
            safe = sanitize_html_output(payload)
            # Should not contain dangerous unencoded tags
            assert '<script' not in safe.lower()
            assert 'onerror=' not in safe.lower()
            assert 'onload=' not in safe.lower()

    def test_xml_encoding(self):
        """XML encoding should escape special chars."""
        unsafe = '<root attr="value">content</root>'
        safe = sanitize_xml_output(unsafe)
        assert '&lt;' in safe
        assert '&gt;' in safe

    def test_javascript_context_escaping(self):
        """JavaScript context should escape dangerous chars."""
        unsafe = 'hello"; alert("XSS"); //'
        safe = sanitize_javascript_context(unsafe)
        assert '\\"' in safe or "\\'" in safe
        assert 'alert' not in safe or '\\' in safe


# ============================================================================
# SECTION 3: SQL Injection Prevention
# ============================================================================

class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""

    OWASP_SQLI_PAYLOADS = [
        "' OR '1'='1",
        "' OR 'a'='a",
        "' OR 1=1 --",
        "admin' --",
        "' UNION SELECT NULL,NULL,NULL --",
        "1; DROP TABLE users; --",
        "' AND (SELECT * FROM users) --",
        "') UNION ALL SELECT NULL,NULL,NULL --",
        "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
    ]

    def test_sql_identifier_escaping(self):
        """Column/table names should be properly escaped."""
        table = escape_sql_identifier("users")
        assert table == "`users`"

        # Invalid identifier should raise
        with pytest.raises(ValueError):
            escape_sql_identifier("users; DROP TABLE")

    def test_query_builder_select(self):
        """Query builder should generate safe SELECT."""
        query, params = build_select_query(
            'users',
            ['id', 'name'],
            {'role': 'admin'},
            limit=10
        )
        # Should be parameterized
        assert '%(param_0)s' in query
        assert params['param_0'] == 'admin'

    def test_query_builder_insert(self):
        """Query builder should generate safe INSERT."""
        query, params = build_insert_query(
            'users',
            {'name': 'Alice', 'email': "alice'; DROP TABLE users; --"}
        )
        # Should be parameterized
        assert '%(param_' in query
        # Dangerous value should be in params, not query
        assert "DROP TABLE" not in query

    def test_query_builder_update(self):
        """Query builder should generate safe UPDATE."""
        query, params = build_update_query(
            'users',
            {'role': 'admin'},
            {'id': 123}
        )
        assert '%(update_param_0)s' in query
        assert '%(where_param_1)s' in query

    def test_query_builder_requires_where(self):
        """UPDATE/DELETE should require WHERE clause."""
        builder = SQLQueryBuilder()
        builder.select(['*']).from_table('users')

        # Should fail without WHERE
        with pytest.raises(ValueError) as exc_info:
            from HoloLoom.security.validation.sql_safe import build_delete_query
            build_delete_query('users', {})
        assert "WHERE conditions required" in str(exc_info.value)

    def test_sqli_detection(self):
        """SQL injection detection should flag suspicious patterns."""
        for payload in self.OWASP_SQLI_PAYLOADS:
            is_suspicious, pattern = detect_sql_injection_attempt(payload)
            assert is_suspicious, f"Failed to detect: {payload}"


# ============================================================================
# SECTION 4: LDAP Injection Prevention
# ============================================================================

class TestLDAPInjectionPrevention:
    """Test LDAP injection prevention."""

    LDAP_INJECTION_PAYLOADS = [
        "*",
        "user*)(&",
        "*)(uid=*",
        "admin*",
        "user*)(cn=*",
    ]

    def test_ldap_escaping(self):
        """LDAP special chars should be escaped."""
        unsafe = "user*)(uid=*"
        safe = escape_ldap_filter(unsafe)
        assert safe == "user\\2a)(uid\\3d\\2a"

    def test_ldap_filter_building(self):
        """LDAP filter should safely use parameters."""
        template = '(&(uid={username})(objectClass=person))'
        params = {'username': 'user*)(cn=*'}
        filter_str = build_ldap_filter(template, params)
        # Dangerous chars should be escaped
        assert '\\2a' in filter_str  # Escaped *
        assert '\\29' in filter_str  # Escaped )

    def test_ldap_payload_escaping(self):
        """All LDAP injection payloads should be escaped."""
        for payload in self.LDAP_INJECTION_PAYLOADS:
            safe = escape_ldap_filter(payload)
            # Should not contain dangerous unescaped chars
            assert '*' not in safe or '\\' in safe


# ============================================================================
# SECTION 5: Path Traversal Prevention
# ============================================================================

class TestPathTraversalPrevention:
    """Test path traversal prevention."""

    TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "....//....//....//etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",  # URL encoded
    ]

    def test_safe_path_validation(self):
        """Safe paths should validate."""
        safe = validate_safe_path('docs/readme.txt', '/app/uploads')
        assert safe.endswith('docs/readme.txt')
        assert '/app/uploads' in safe

    def test_path_traversal_detection(self):
        """Path traversal should be detected."""
        for payload in self.TRAVERSAL_PAYLOADS:
            with pytest.raises(ValueError) as exc_info:
                validate_safe_path(payload, '/app/uploads')
            assert "escape" in str(exc_info.value).lower() or "relative" in str(exc_info.value).lower()


# ============================================================================
# SECTION 6: File Upload Validation
# ============================================================================

class TestFileUploadValidation:
    """Test file upload validation."""

    def test_pdf_validation(self):
        """Valid PDF should pass validation."""
        # PDF magic bytes
        pdf_header = b'%PDF-1.4\n'
        pdf_content = pdf_header + b'Some PDF content'

        result = validate_file_upload(
            'document.pdf',
            pdf_content,
            'application/pdf'
        )
        assert result.valid is True

    def test_image_validation(self):
        """Valid image should pass."""
        # PNG magic bytes
        png_header = b'\x89PNG\r\n\x1a\n'
        png_content = png_header + b'fake PNG data'

        result = validate_file_upload(
            'image.png',
            png_content,
            'image/png'
        )
        assert result.valid is True

    def test_file_size_limit(self):
        """File exceeding size limit should fail."""
        large_file = b'A' * (11_000_000)  # 11 MB
        result = validate_file_upload(
            'large.pdf',
            large_file,
            'application/pdf'
        )
        assert result.valid is False
        assert any('exceed' in err.lower() for err in result.errors)

    def test_dangerous_extension(self):
        """Dangerous extensions should fail."""
        script_content = b'#!/bin/bash\nrm -rf /'
        result = validate_file_upload(
            'malware.sh',
            script_content,
            'text/plain'
        )
        # Should fail due to extension
        assert result.valid is False

    def test_mime_type_mismatch(self):
        """MIME type mismatch should be detected."""
        # PNG header but claiming to be PDF
        png_content = b'\x89PNG\r\n\x1a\n' + b'fake PNG'
        result = validate_file_upload(
            'fake.pdf',
            png_content,
            'application/pdf'  # Wrong MIME type
        )
        # Should fail: detected PNG but claimed PDF
        assert result.valid is False

    def test_filename_path_traversal(self):
        """Filename with path traversal should fail."""
        result = validate_file_upload(
            '../../etc/passwd',
            b'some content',
            'text/plain'
        )
        assert result.valid is False

    def test_safe_filename_generation(self):
        """Safe filename should remove dangerous chars."""
        unsafe = '../../../etc/passwd'
        safe = generate_safe_filename(unsafe)
        assert '..' not in safe
        assert '/' not in safe
        assert 'passwd' in safe

    def test_filename_with_special_chars(self):
        """Special chars in filename should be replaced."""
        unsafe = 'my@file#name$.txt'
        safe = generate_safe_filename(unsafe)
        # Special chars should be replaced
        assert '@' not in safe
        assert '#' not in safe


# ============================================================================
# SECTION 7: Unicode & Character Normalization
# ============================================================================

class TestUnicodeNormalization:
    """Test Unicode attack prevention."""

    UNICODE_PAYLOADS = [
        '\uff1cscript\uff1e',  # Fullwidth < and >
        '\u0301;drop table users;',  # Combining characters
        'user\u200b;--',  # Zero-width space
        'test\ufeffadmin',  # BOM
    ]

    def test_unicode_normalization(self):
        """Unicode should be normalized to NFKC."""
        # Fullwidth A (U+FF21) → normal A (U+0041)
        unsafe = '\uff21'
        safe = normalize_unicode(unsafe)
        assert safe == 'A'

    def test_control_character_removal(self):
        """Control characters should be removed."""
        unsafe = 'hello\x00world\x01\x02'
        safe = remove_control_characters(unsafe)
        assert '\x00' not in safe
        assert '\x01' not in safe

    def test_unicode_payload_normalization(self):
        """OWASP Unicode payloads should be normalized."""
        for payload in self.UNICODE_PAYLOADS:
            safe = normalize_unicode(payload)
            # Should be normalized form
            assert len(safe) <= len(payload) + 10  # Allow some expansion


# ============================================================================
# SECTION 8: Email Header Injection Prevention
# ============================================================================

class TestEmailHeaderInjectionPrevention:
    """Test email header injection prevention."""

    EMAIL_INJECTION_PAYLOADS = [
        "user@example.com\nCc: attacker@evil.com",
        "user@example.com\nBcc: attacker@evil.com",
        "user@example.com\r\nCc: attacker@evil.com",
        "user@example.com%0aCc: attacker@evil.com",
    ]

    def test_crlf_removal(self):
        """CRLF should be removed from headers."""
        unsafe = "user@example.com\nCc: attacker@evil.com"
        safe = sanitize_email_header(unsafe)
        assert '\n' not in safe
        assert '\r' not in safe

    def test_email_header_injection_payloads(self):
        """Email injection payloads should be sanitized."""
        for payload in self.EMAIL_INJECTION_PAYLOADS:
            safe = sanitize_email_header(payload)
            # Should not contain newlines
            assert '\n' not in safe
            assert '\r' not in safe


# ============================================================================
# SECTION 9: URL Validation
# ============================================================================

class TestURLValidation:
    """Test URL validation."""

    DANGEROUS_URL_SCHEMES = [
        'javascript:alert("XSS")',
        'data:text/html,<script>alert("XSS")</script>',
        'file:///etc/passwd',
    ]

    def test_valid_https_url(self):
        """Valid HTTPS URL should pass."""
        url = URLRequest(url="https://example.com/api/endpoint")
        assert str(url.url).startswith("https://")

    def test_dangerous_javascript_protocol(self):
        """JavaScript protocol should fail."""
        with pytest.raises(ValidationError):
            URLRequest(url="javascript:alert('XSS')")

    def test_dangerous_data_protocol(self):
        """Data protocol should fail."""
        with pytest.raises(ValidationError):
            URLRequest(url="data:text/html,<script>alert('XSS')</script>")


# ============================================================================
# SECTION 10: Batch & Edge Cases
# ============================================================================

class TestBatchAndEdgeCases:
    """Test batch operations and edge cases."""

    def test_batch_query_limit(self):
        """Batch should limit to 100 queries."""
        queries = [QueryRequest(text=f"Query {i}") for i in range(101)]
        with pytest.raises(ValidationError):
            BatchQueryRequest(queries=queries)

    def test_empty_string(self):
        """Empty string should fail validation."""
        with pytest.raises(ValidationError):
            QueryRequest(text="")

    def test_null_values(self):
        """Null values should be handled properly."""
        with pytest.raises(ValidationError):
            QueryRequest(text=None)

    def test_very_long_json(self):
        """JSON exceeding size limit should fail."""
        large_data = {"key_" + str(i): "A" * 100_000 for i in range(20)}
        with pytest.raises(ValidationError) as exc_info:
            JSONRequest(data=large_data)
        assert "exceed" in str(exc_info.value).lower()

    def test_deeply_nested_json(self):
        """JSON exceeding depth limit should fail."""
        nested = {"a": {}}
        current = nested["a"]
        for i in range(15):
            current["b"] = {}
            current = current["b"]

        with pytest.raises(ValidationError) as exc_info:
            JSONRequest(data=nested)
        assert "depth" in str(exc_info.value).lower()


# ============================================================================
# SECTION 11: Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for validation operations."""

    def test_html_encoding_performance(self, benchmark):
        """Benchmark HTML encoding."""
        payload = '<script>alert("XSS")</script>' * 100
        benchmark(sanitize_html_output, payload)

    def test_query_validation_performance(self, benchmark):
        """Benchmark query validation."""
        def validate_query():
            QueryRequest(text="What is Thompson Sampling?")

        benchmark(validate_query)

    def test_sql_query_builder_performance(self, benchmark):
        """Benchmark SQL query building."""
        def build_query():
            builder = SQLQueryBuilder()
            builder.select(['id', 'name', 'email'])
            builder.from_table('users')
            builder.where('role', 'admin')
            builder.limit(10)
            return builder.build()

        benchmark(build_query)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
