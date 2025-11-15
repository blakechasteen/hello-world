"""
Security Event Taxonomy for SIEM Integration

Defines standardized categories and subcategories for security events.

Created: 2025-11-15
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re


class SecurityEventCategory(str, Enum):
    """Top-level security event categories"""
    AUTH = "AUTH"  # Authentication events
    AUTHZ = "AUTHZ"  # Authorization events
    ATTACK = "ATTACK"  # Attack attempts
    DATA = "DATA"  # Data access events
    SYSTEM = "SYSTEM"  # System events
    INCIDENT = "INCIDENT"  # Security incidents


class SecurityEventSubcategory(str, Enum):
    """Detailed security event subcategories"""

    # AUTH subcategories
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_AUTH = "failed_auth"
    MFA_CHALLENGE = "mfa_challenge"
    PASSWORD_RESET = "password_reset"
    SESSION_EXPIRED = "session_expired"

    # AUTHZ subcategories
    PERMISSION_DENIED = "permission_denied"
    ROLE_CHANGE = "role_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ACCESS_GRANTED = "access_granted"

    # ATTACK subcategories
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    BRUTE_FORCE = "brute_force"
    DOS = "dos"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    MALICIOUS_PAYLOAD = "malicious_payload"

    # DATA subcategories
    QUERY = "query"
    EXPORT = "export"
    DELETE = "delete"
    MODIFY = "modify"
    BULK_ACCESS = "bulk_access"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"

    # SYSTEM subcategories
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    CONFIG_CHANGE = "config_change"
    SERVICE_ERROR = "service_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

    # INCIDENT subcategories
    BREACH_DETECTED = "breach_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    POLICY_VIOLATION = "policy_violation"
    DATA_LEAK = "data_leak"


class SeverityLevel(str, Enum):
    """Security event severity levels (aligned with logging)"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityEvent:
    """
    Structured security event for SIEM logging

    Follows the taxonomy and provides automatic PII redaction.
    """
    timestamp: datetime
    level: SeverityLevel
    category: SecurityEventCategory
    subcategory: SecurityEventSubcategory
    action: str
    blocked: bool
    risk_score: float  # 0.0-10.0

    # Optional fields
    source_ip: Optional[str] = None
    user_id: Optional[str] = None  # Should be hashed
    payload: Optional[str] = "[REDACTED]"  # Redacted by default
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Auto-generated fields
    event_id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        """Generate event ID if not provided"""
        if not self.event_id:
            # Generate unique event ID
            data = f"{self.timestamp.isoformat()}{self.category}{self.subcategory}{self.action}"
            self.event_id = hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SIEM-compatible JSON dict"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "subcategory": self.subcategory.value,
            "action": self.action,
            "blocked": self.blocked,
            "risk_score": self.risk_score,
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "payload": self.payload,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityEvent":
        """Create SecurityEvent from dict"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=SeverityLevel(data["level"]),
            category=SecurityEventCategory(data["category"]),
            subcategory=SecurityEventSubcategory(data["subcategory"]),
            action=data["action"],
            blocked=data["blocked"],
            risk_score=data["risk_score"],
            source_ip=data.get("source_ip"),
            user_id=data.get("user_id"),
            payload=data.get("payload", "[REDACTED]"),
            target=data.get("target"),
            metadata=data.get("metadata", {}),
            event_id=data.get("event_id", ""),
        )


class PIIRedactor:
    """Automatic PII redaction for security logs"""

    # PII patterns (compiled for performance)
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

    @classmethod
    def redact(cls, text: str, preserve_ips: bool = True) -> str:
        """
        Redact PII from text

        Args:
            text: Text to redact
            preserve_ips: If True, preserve IP addresses (needed for security analysis)

        Returns:
            Redacted text
        """
        if not text:
            return text

        # Redact emails
        text = cls.EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)

        # Redact SSNs
        text = cls.SSN_PATTERN.sub('[SSN_REDACTED]', text)

        # Redact credit cards
        text = cls.CREDIT_CARD_PATTERN.sub('[CC_REDACTED]', text)

        # Redact phone numbers
        text = cls.PHONE_PATTERN.sub('[PHONE_REDACTED]', text)

        # Optionally redact IPs
        if not preserve_ips:
            text = cls.IP_PATTERN.sub('[IP_REDACTED]', text)

        return text

    @classmethod
    def hash_user_id(cls, user_id: str, salt: str = "hololoom_security") -> str:
        """
        Hash user ID for privacy-preserving logging

        Args:
            user_id: User identifier
            salt: Salt for hashing

        Returns:
            Hashed user ID
        """
        return hashlib.sha256(f"{salt}{user_id}".encode()).hexdigest()[:16]


def create_security_event(
    category: SecurityEventCategory,
    subcategory: SecurityEventSubcategory,
    action: str,
    blocked: bool,
    risk_score: float,
    level: Optional[SeverityLevel] = None,
    source_ip: Optional[str] = None,
    user_id: Optional[str] = None,
    payload: Optional[str] = None,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    auto_redact: bool = True,
) -> SecurityEvent:
    """
    Create a security event with automatic PII redaction and severity inference

    Args:
        category: Event category (AUTH, AUTHZ, ATTACK, etc.)
        subcategory: Event subcategory
        action: Action performed
        blocked: Whether action was blocked
        risk_score: Risk score (0.0-10.0)
        level: Severity level (auto-inferred if None)
        source_ip: Source IP address
        user_id: User ID (will be hashed if auto_redact=True)
        payload: Payload (will be redacted if auto_redact=True)
        target: Target of action
        metadata: Additional metadata
        auto_redact: Automatically redact PII

    Returns:
        SecurityEvent
    """
    # Auto-infer severity if not provided
    if level is None:
        level = _infer_severity(category, subcategory, risk_score, blocked)

    # Auto-redact PII
    if auto_redact:
        if user_id:
            user_id = PIIRedactor.hash_user_id(user_id)
        if payload:
            payload = PIIRedactor.redact(payload, preserve_ips=True)

    return SecurityEvent(
        timestamp=datetime.utcnow(),
        level=level,
        category=category,
        subcategory=subcategory,
        action=action,
        blocked=blocked,
        risk_score=risk_score,
        source_ip=source_ip,
        user_id=user_id,
        payload=payload or "[REDACTED]",
        target=target,
        metadata=metadata or {},
    )


def _infer_severity(
    category: SecurityEventCategory,
    subcategory: SecurityEventSubcategory,
    risk_score: float,
    blocked: bool,
) -> SeverityLevel:
    """Infer severity level from event characteristics"""

    # INCIDENT and high-risk attacks are always CRITICAL
    if category == SecurityEventCategory.INCIDENT:
        return SeverityLevel.CRITICAL

    if category == SecurityEventCategory.ATTACK:
        if risk_score >= 8.0:
            return SeverityLevel.CRITICAL
        elif risk_score >= 6.0:
            return SeverityLevel.ERROR
        elif risk_score >= 4.0:
            return SeverityLevel.WARNING
        else:
            return SeverityLevel.INFO

    # Failed auth is WARNING
    if subcategory == SecurityEventSubcategory.FAILED_AUTH:
        return SeverityLevel.WARNING

    # Permission denied is WARNING if not blocked (shouldn't happen)
    if subcategory == SecurityEventSubcategory.PERMISSION_DENIED:
        return SeverityLevel.WARNING if not blocked else SeverityLevel.INFO

    # System errors are ERROR
    if subcategory == SecurityEventSubcategory.SERVICE_ERROR:
        return SeverityLevel.ERROR

    # Resource exhaustion is WARNING
    if subcategory == SecurityEventSubcategory.RESOURCE_EXHAUSTION:
        return SeverityLevel.WARNING

    # Default based on risk score
    if risk_score >= 7.0:
        return SeverityLevel.ERROR
    elif risk_score >= 4.0:
        return SeverityLevel.WARNING
    else:
        return SeverityLevel.INFO


def get_taxonomy_stats() -> Dict[str, int]:
    """Get statistics about the taxonomy"""
    return {
        "categories": len(SecurityEventCategory),
        "subcategories": len(SecurityEventSubcategory),
        "severity_levels": len(SeverityLevel),
        "total_event_types": len(SecurityEventCategory) * len(SecurityEventSubcategory),
    }


# Taxonomy mapping for quick reference
CATEGORY_SUBCATEGORIES: Dict[SecurityEventCategory, List[SecurityEventSubcategory]] = {
    SecurityEventCategory.AUTH: [
        SecurityEventSubcategory.LOGIN,
        SecurityEventSubcategory.LOGOUT,
        SecurityEventSubcategory.FAILED_AUTH,
        SecurityEventSubcategory.MFA_CHALLENGE,
        SecurityEventSubcategory.PASSWORD_RESET,
        SecurityEventSubcategory.SESSION_EXPIRED,
    ],
    SecurityEventCategory.AUTHZ: [
        SecurityEventSubcategory.PERMISSION_DENIED,
        SecurityEventSubcategory.ROLE_CHANGE,
        SecurityEventSubcategory.PRIVILEGE_ESCALATION,
        SecurityEventSubcategory.ACCESS_GRANTED,
    ],
    SecurityEventCategory.ATTACK: [
        SecurityEventSubcategory.SQL_INJECTION,
        SecurityEventSubcategory.XSS,
        SecurityEventSubcategory.CSRF,
        SecurityEventSubcategory.BRUTE_FORCE,
        SecurityEventSubcategory.DOS,
        SecurityEventSubcategory.PATH_TRAVERSAL,
        SecurityEventSubcategory.COMMAND_INJECTION,
        SecurityEventSubcategory.MALICIOUS_PAYLOAD,
    ],
    SecurityEventCategory.DATA: [
        SecurityEventSubcategory.QUERY,
        SecurityEventSubcategory.EXPORT,
        SecurityEventSubcategory.DELETE,
        SecurityEventSubcategory.MODIFY,
        SecurityEventSubcategory.BULK_ACCESS,
        SecurityEventSubcategory.SENSITIVE_DATA_ACCESS,
    ],
    SecurityEventCategory.SYSTEM: [
        SecurityEventSubcategory.STARTUP,
        SecurityEventSubcategory.SHUTDOWN,
        SecurityEventSubcategory.CONFIG_CHANGE,
        SecurityEventSubcategory.SERVICE_ERROR,
        SecurityEventSubcategory.RESOURCE_EXHAUSTION,
    ],
    SecurityEventCategory.INCIDENT: [
        SecurityEventSubcategory.BREACH_DETECTED,
        SecurityEventSubcategory.ANOMALY_DETECTED,
        SecurityEventSubcategory.POLICY_VIOLATION,
        SecurityEventSubcategory.DATA_LEAK,
    ],
}
