"""
Web Application Firewall (WAF) Middleware for HoloLoom

Implements ModSecurity-style rules to protect against common web attacks:
- SQL Injection
- Cross-Site Scripting (XSS)
- Path Traversal
- Command Injection
- Header Validation
- Request Size Limits

Created: 2025-11-26
"""

import hashlib
import json
import logging
import re
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class WAFRule:
    """Base class for WAF rules"""

    def __init__(self, rule_id: str, description: str, severity: str = "HIGH"):
        self.rule_id = rule_id
        self.description = description
        self.severity = severity
        self.violations_count = 0

    def check(self, request: Request, body: bytes = b"") -> str | None:
        """Check if request violates the rule. Return violation details if true."""
        raise NotImplementedError


class SQLInjectionRule(WAFRule):
    """Detect SQL injection attempts"""

    SQL_PATTERNS = [
        r"\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b",
        r"\b(CAST|CONVERT|CHAR|NCHAR|VARCHAR|NVARCHAR)\s*\(",
        r"(--|#|\/\*|\*\/)",  # SQL comments
        r"\b(OR|AND)\s+\d+\s*=\s*\d+",  # OR 1=1
        r"\b(OR|AND)\s+'[^']*'\s*=\s*'[^']*'",  # OR 'a'='a'
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)",  # Stacked queries
        r"\bEXEC\s+(SP_|XP_)",  # Stored procedures
        r"\b(WAITFOR|DELAY|BENCHMARK)\b",  # Time-based attacks
        r"(0x[0-9a-fA-F]+)",  # Hex encoding
    ]

    def __init__(self):
        super().__init__(
            "WAF-001",
            "SQL Injection Detection",
            "CRITICAL"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL parameters
        url_str = str(request.url)
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                return f"SQL injection pattern detected in URL: {pattern.pattern}"

        # Check headers
        for header_name, header_value in request.headers.items():
            for pattern in self.compiled_patterns:
                if pattern.search(header_value):
                    return f"SQL injection pattern detected in header {header_name}: {pattern.pattern}"

        # Check body
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"SQL injection pattern detected in body: {pattern.pattern}"
            except Exception:
                pass

        return None


class XSSRule(WAFRule):
    """Detect Cross-Site Scripting attempts"""

    XSS_PATTERNS = [
        r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",  # Script tags
        r"javascript\s*:",  # JavaScript protocol
        r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
        r"<\s*iframe[^>]*>",  # Iframes
        r"<\s*embed[^>]*>",  # Embed tags
        r"<\s*object[^>]*>",  # Object tags
        r"<\s*applet[^>]*>",  # Applet tags
        r"<\s*meta[^>]*http-equiv",  # Meta refresh
        r"<\s*img[^>]*src[^>]*=",  # Image tags with src
        r"<\s*link[^>]*href",  # Link tags
        r"<\s*base[^>]*href",  # Base tags
        r"<\s*form[^>]*action",  # Form tags
        r"eval\s*\(",  # Eval function
        r"expression\s*\(",  # CSS expressions
        r"vbscript\s*:",  # VBScript protocol
        r"data\s*:\s*text/html",  # Data URLs with HTML
    ]

    def __init__(self):
        super().__init__(
            "WAF-002",
            "Cross-Site Scripting Detection",
            "HIGH"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in self.XSS_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL parameters
        url_str = str(request.url)
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                return f"XSS pattern detected in URL: {pattern.pattern}"

        # Check headers
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ['cookie', 'authorization']:  # Skip sensitive headers
                for pattern in self.compiled_patterns:
                    if pattern.search(header_value):
                        return f"XSS pattern detected in header {header_name}: {pattern.pattern}"

        # Check body
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"XSS pattern detected in body: {pattern.pattern}"
            except Exception:
                pass

        return None


class PathTraversalRule(WAFRule):
    """Detect path traversal attempts"""

    PATH_PATTERNS = [
        r"\.\./",  # Unix path traversal
        r"\.\.\\",  # Windows path traversal
        r"%2e%2e[/\\]",  # URL encoded
        r"\.\.%2f",  # Mixed encoding
        r"\.\.%5c",  # Mixed encoding (backslash)
        r"/etc/passwd",  # Common target
        r"c:\\windows",  # Windows paths
        r"c:/windows",  # Windows paths (forward slash)
        r"\.\./\.\./",  # Multiple traversals
        r"\.\.;",  # Path traversal with semicolon
    ]

    def __init__(self):
        super().__init__(
            "WAF-003",
            "Path Traversal Detection",
            "HIGH"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATH_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL
        url_str = str(request.url)
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                return f"Path traversal pattern detected in URL: {pattern.pattern}"

        # Check body
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"Path traversal pattern detected in body: {pattern.pattern}"
            except Exception:
                pass

        return None


class CommandInjectionRule(WAFRule):
    """Detect command injection attempts"""

    CMD_PATTERNS = [
        r";\s*(?:ls|cat|pwd|whoami|id|uname|ps|netstat)",  # Unix commands
        r"&&\s*(?:ls|cat|pwd|whoami|id|uname|ps|netstat)",  # Command chaining
        r"\|\s*(?:ls|cat|pwd|whoami|id|uname|ps|netstat)",  # Pipe
        r"`[^`]+`",  # Backticks
        r"\$\([^)]+\)",  # Command substitution
        r";\s*(?:dir|type|ver|ipconfig|net)",  # Windows commands
        r"&&\s*(?:dir|type|ver|ipconfig|net)",  # Windows chaining
        r"\|\s*(?:dir|type|ver|ipconfig|net)",  # Windows pipe
        r">\s*/dev/null",  # Redirection
        r"2>&1",  # Error redirection
        r"/bin/(?:sh|bash|dash|ksh|zsh)",  # Shell paths
        r"cmd\.exe",  # Windows shell
        r"powershell",  # PowerShell
    ]

    def __init__(self):
        super().__init__(
            "WAF-004",
            "Command Injection Detection",
            "CRITICAL"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.CMD_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL parameters
        url_str = str(request.url)
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                return f"Command injection pattern detected in URL: {pattern.pattern}"

        # Check headers
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ['user-agent', 'cookie', 'authorization']:
                for pattern in self.compiled_patterns:
                    if pattern.search(header_value):
                        return f"Command injection pattern detected in header {header_name}: {pattern.pattern}"

        # Check body
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"Command injection pattern detected in body: {pattern.pattern}"
            except Exception:
                pass

        return None


class HeaderValidationRule(WAFRule):
    """Validate HTTP headers"""

    MAX_HEADER_SIZE = 8192  # 8KB max per header
    MAX_TOTAL_HEADERS_SIZE = 32768  # 32KB total

    FORBIDDEN_HEADERS = [
        'proxy-authorization',
        'x-forwarded-host',  # Can be used for host header injection
    ]

    def __init__(self):
        super().__init__(
            "WAF-005",
            "Header Validation",
            "MEDIUM"
        )

    def check(self, request: Request, body: bytes = b"") -> str | None:
        total_size = 0

        for header_name, header_value in request.headers.items():
            header_size = len(header_name) + len(header_value)
            total_size += header_size

            # Check individual header size
            if header_size > self.MAX_HEADER_SIZE:
                return f"Header {header_name} exceeds maximum size ({header_size} > {self.MAX_HEADER_SIZE})"

            # Check for forbidden headers
            if header_name.lower() in self.FORBIDDEN_HEADERS:
                return f"Forbidden header detected: {header_name}"

            # Check for null bytes
            if '\x00' in header_value:
                return f"Null byte detected in header {header_name}"

            # Check for line breaks (header injection)
            if '\r' in header_value or '\n' in header_value:
                return f"Line break detected in header {header_name} (possible header injection)"

        # Check total headers size
        if total_size > self.MAX_TOTAL_HEADERS_SIZE:
            return f"Total headers size exceeds maximum ({total_size} > {self.MAX_TOTAL_HEADERS_SIZE})"

        return None


class RequestSizeLimitRule(WAFRule):
    """Enforce request size limits"""

    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB default
    MAX_URL_LENGTH = 2048  # Standard URL length limit

    def __init__(self, max_body_size: int = None):
        super().__init__(
            "WAF-006",
            "Request Size Validation",
            "MEDIUM"
        )
        if max_body_size:
            self.MAX_BODY_SIZE = max_body_size

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL length
        if len(str(request.url)) > self.MAX_URL_LENGTH:
            return f"URL exceeds maximum length ({len(str(request.url))} > {self.MAX_URL_LENGTH})"

        # Check body size
        if body and len(body) > self.MAX_BODY_SIZE:
            return f"Request body exceeds maximum size ({len(body)} > {self.MAX_BODY_SIZE})"

        return None


class XMLExternalEntityRule(WAFRule):
    """Detect XXE (XML External Entity) attacks"""

    XXE_PATTERNS = [
        r"<!DOCTYPE[^>]*\[",  # DOCTYPE with internal subset
        r"<!ENTITY",  # Entity declaration
        r"SYSTEM\s+[\"']",  # SYSTEM identifier
        r"PUBLIC\s+[\"']",  # PUBLIC identifier
        r"file://",  # File protocol
        r"php://",  # PHP protocol
        r"expect://",  # Expect protocol
        r"data:",  # Data protocol in XML
    ]

    def __init__(self):
        super().__init__(
            "WAF-007",
            "XML External Entity Detection",
            "HIGH"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.XXE_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Only check if content type indicates XML
        content_type = request.headers.get('content-type', '').lower()
        if 'xml' not in content_type and 'soap' not in content_type:
            return None

        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"XXE pattern detected: {pattern.pattern}"
            except Exception:
                pass

        return None


class LDAPInjectionRule(WAFRule):
    """Detect LDAP injection attempts"""

    LDAP_PATTERNS = [
        r"\(\s*\|\s*\(",  # LDAP OR
        r"\(\s*&\s*\(",  # LDAP AND
        r"\(\s*!\s*\(",  # LDAP NOT
        r"\*\)\(",  # Wildcard with parenthesis
        r"\)\s*\(\s*\)",  # Empty filter
        r"[^\w]\*[^\w]",  # Wildcard in unusual context
    ]

    def __init__(self):
        super().__init__(
            "WAF-008",
            "LDAP Injection Detection",
            "HIGH"
        )
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.LDAP_PATTERNS]

    def check(self, request: Request, body: bytes = b"") -> str | None:
        # Check URL parameters
        url_str = str(request.url)
        for pattern in self.compiled_patterns:
            if pattern.search(url_str):
                return f"LDAP injection pattern detected in URL: {pattern.pattern}"

        # Check body
        if body:
            try:
                body_str = body.decode('utf-8', errors='ignore')
                for pattern in self.compiled_patterns:
                    if pattern.search(body_str):
                        return f"LDAP injection pattern detected in body: {pattern.pattern}"
            except Exception:
                pass

        return None


class WAFMiddleware(BaseHTTPMiddleware):
    """Web Application Firewall Middleware"""

    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        whitelist_ips: list[str] | None = None,
        blacklist_ips: list[str] | None = None,
        custom_rules: list[WAFRule] | None = None,
        log_violations: bool = True,
        block_on_violation: bool = True,
        security_monitor = None
    ):
        super().__init__(app)
        self.enabled = enabled
        self.whitelist_ips = set(whitelist_ips or [])
        self.blacklist_ips = set(blacklist_ips or [])
        self.log_violations = log_violations
        self.block_on_violation = block_on_violation
        self.security_monitor = security_monitor

        # Initialize default rules
        self.rules = [
            SQLInjectionRule(),
            XSSRule(),
            PathTraversalRule(),
            CommandInjectionRule(),
            HeaderValidationRule(),
            RequestSizeLimitRule(),
            XMLExternalEntityRule(),
            LDAPInjectionRule(),
        ]

        # Add custom rules if provided
        if custom_rules:
            self.rules.extend(custom_rules)

        # Violation statistics
        self.violations_by_ip: dict[str, int] = defaultdict(int)
        self.violations_by_rule: dict[str, int] = defaultdict(int)

        logger.info(f"WAF Middleware initialized with {len(self.rules)} rules")

    async def dispatch(self, request: Request, call_next):
        """Process request through WAF rules"""

        if not self.enabled:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check IP whitelist/blacklist
        if self.whitelist_ips and client_ip not in self.whitelist_ips:
            return self._block_request("IP not in whitelist", client_ip, "IP_WHITELIST")

        if client_ip in self.blacklist_ips:
            return self._block_request("IP in blacklist", client_ip, "IP_BLACKLIST")

        # Read request body (we need to cache it for downstream)
        body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            # Create new request with cached body for downstream
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive

        # Run all WAF rules
        for rule in self.rules:
            try:
                violation = rule.check(request, body)
                if violation:
                    rule.violations_count += 1
                    self.violations_by_ip[client_ip] += 1
                    self.violations_by_rule[rule.rule_id] += 1

                    if self.log_violations:
                        await self._log_violation(
                            rule=rule,
                            violation=violation,
                            request=request,
                            client_ip=client_ip,
                            body=body
                        )

                    if self.block_on_violation:
                        return self._block_request(violation, client_ip, rule.rule_id)
            except Exception as e:
                logger.error(f"Error in WAF rule {rule.rule_id}: {e}")

        # Request passed all checks
        response = await call_next(request)

        # Add security headers to response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

    def _block_request(self, reason: str, client_ip: str, rule_id: str) -> JSONResponse:
        """Block request and return error response"""

        logger.warning(f"WAF blocked request from {client_ip}: {reason} (Rule: {rule_id})")

        # Log to security monitor if available
        if self.security_monitor:
            asyncio.create_task(self.security_monitor.log_waf_violation(
                client_ip=client_ip,
                rule_id=rule_id,
                reason=reason,
                timestamp=datetime.utcnow()
            ))

        return JSONResponse(
            status_code=400,
            content={
                "error": "Bad Request",
                "message": "Request blocked by security policy",
                "rule_id": rule_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _log_violation(
        self,
        rule: WAFRule,
        violation: str,
        request: Request,
        client_ip: str,
        body: bytes
    ):
        """Log WAF violation details"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "rule_id": rule.rule_id,
            "rule_description": rule.description,
            "severity": rule.severity,
            "violation": violation,
            "client_ip": client_ip,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body_hash": hashlib.sha256(body).hexdigest() if body else None,
            "violations_count": rule.violations_count,
            "ip_violations_count": self.violations_by_ip[client_ip]
        }

        logger.warning(f"WAF Violation: {json.dumps(log_entry)}")

        # Send to security monitor
        if self.security_monitor:
            await self.security_monitor.log_detailed_violation(log_entry)

    def get_statistics(self) -> dict[str, Any]:
        """Get WAF statistics"""

        return {
            "enabled": self.enabled,
            "total_rules": len(self.rules),
            "violations_by_rule": dict(self.violations_by_rule),
            "violations_by_ip": dict(self.violations_by_ip),
            "top_violators": sorted(
                self.violations_by_ip.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "rule_details": [
                {
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "severity": rule.severity,
                    "violations_count": rule.violations_count
                }
                for rule in self.rules
            ]
        }
