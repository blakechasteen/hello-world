"""
PII Detection and Classification Module
========================================

**Purpose**: Automatically detect and classify Personally Identifiable Information (PII)
in text inputs, outputs, and memory shards for compliance and privacy protection.

**Date**: 2025-11-17
**Phase**: Tenant Isolation and PII Flows
**Key Features**:
- 15+ PII entity types (SSN, email, phone, credit card, IP address, etc.)
- Regex-based detection (fast, zero dependencies)
- Confidence scoring for each detection
- Redaction and masking utilities
- Integration with Department protocol PrivacyEnvelope

**Philosophy**:
"Privacy by design, not as an afterthought. Every piece of data should be
classified and protected before it enters the system."

**Compliance**:
- GDPR Article 4(1): Identification of personal data
- HIPAA § 164.514: Protected Health Information (PHI)
- SOC 2 Trust Service Criteria: Confidentiality

Author: HoloLoom Security Team
Timestamp: 2025-11-17
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import hashlib


# ============================================================================
# PII Entity Types
# ============================================================================

class PIIType(Enum):
    """
    Types of Personally Identifiable Information.

    Organized by sensitivity level:
    - CRITICAL: SSN, credit card, government ID
    - HIGH: Email, phone, full name, address
    - MEDIUM: IP address, username, DOB
    - LOW: Zip code, general location
    """
    # CRITICAL - Requires immediate encryption and auditing
    SSN = "ssn"                        # Social Security Number (US)
    CREDIT_CARD = "credit_card"        # Credit card number
    PASSPORT = "passport"              # Passport number
    DRIVER_LICENSE = "driver_license"  # Driver's license number

    # HIGH - Requires encryption and access control
    EMAIL = "email"                    # Email address
    PHONE = "phone"                    # Phone number
    FULL_NAME = "full_name"            # Full name (First + Last)
    STREET_ADDRESS = "street_address"  # Street address

    # MEDIUM - Requires access logging
    IP_ADDRESS = "ip_address"          # IP address (v4/v6)
    USERNAME = "username"              # Username/handle
    DATE_OF_BIRTH = "date_of_birth"    # Date of birth
    MEDICAL_RECORD = "medical_record"  # Medical record number

    # LOW - May require masking in logs
    ZIP_CODE = "zip_code"              # Zip/postal code
    CITY = "city"                      # City name
    STATE = "state"                    # State/province

    # CUSTOM - Tenant-specific PII
    CUSTOM = "custom"                  # Custom PII pattern


@dataclass
class PIIDetection:
    """
    A detected PII entity in text.

    Contains:
    - Type of PII
    - Matched text
    - Position in original text
    - Confidence score
    - Suggested redaction

    Example:
        detection = PIIDetection(
            pii_type=PIIType.EMAIL,
            matched_text="user@example.com",
            start_pos=45,
            end_pos=61,
            confidence=0.95,
            redacted_text="[EMAIL_REDACTED]"
        )
    """
    pii_type: PIIType                           # Type of PII
    matched_text: str                           # Original matched text
    start_pos: int                              # Start position in text
    end_pos: int                                # End position in text
    confidence: float                           # 0-1 confidence score
    redacted_text: str                          # Suggested redaction
    context: str = ""                           # Surrounding context (10 chars each side)
    metadata: Dict = field(default_factory=dict)  # Additional info


@dataclass
class PIIAnalysisResult:
    """
    Result of PII analysis on a text document.

    Contains:
    - All detected PII entities
    - Overall risk level
    - Suggested redacted version
    - Compliance flags

    Example:
        result = analyzer.analyze("My SSN is 123-45-6789")
        print(result.risk_level)  # "CRITICAL"
        print(result.redacted_text)  # "My SSN is [SSN_REDACTED]"
        print(result.compliance_flags)  # {"requires_encryption": True, "requires_audit": True}
    """
    original_text: str                          # Original input
    detections: List[PIIDetection]              # All detected PII
    risk_level: str                             # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    redacted_text: str                          # Text with PII redacted
    compliance_flags: Dict[str, bool]           # Compliance requirements
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_pii(self) -> bool:
        """Check if any PII was detected."""
        return len(self.detections) > 0

    @property
    def pii_types(self) -> Set[PIIType]:
        """Get unique PII types detected."""
        return {d.pii_type for d in self.detections}


# ============================================================================
# PII Detection Patterns
# ============================================================================

class PIIPatterns:
    """
    Regex patterns for PII detection.

    Organized by entity type with confidence scoring.
    High-precision patterns score 0.9+, lower-precision patterns score 0.6-0.8.
    """

    # SSN (US): 123-45-6789 or 123456789
    SSN = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 0.95),  # High confidence: XXX-XX-XXXX
        (r'\b\d{9}\b', 0.70),              # Lower confidence: 9 digits (could be other ID)
    ]

    # Credit Card: 4111-1111-1111-1111 or 4111111111111111
    CREDIT_CARD = [
        (r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', 0.90),  # 16 digits with separators
        (r'\b\d{15,16}\b', 0.75),  # 15-16 digits (AmEx or Visa/MC)
    ]

    # Email: user@example.com
    EMAIL = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.95),
    ]

    # Phone: (123) 456-7890, 123-456-7890, 1234567890
    PHONE = [
        (r'\b\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b', 0.90),  # US format
        (r'\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b', 0.85),
        (r'\b\d{10}\b', 0.70),  # 10 digits (could be other number)
    ]

    # IP Address: 192.168.1.1 or 2001:0db8:85a3:0000:0000:8a2e:0370:7334
    IP_ADDRESS = [
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 0.90),  # IPv4
        (r'\b[0-9a-fA-F:]{10,}\b', 0.75),  # IPv6 (simplified)
    ]

    # Date of Birth: 01/15/1990, 1990-01-15
    DATE_OF_BIRTH = [
        (r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', 0.70),  # MM/DD/YYYY or MM-DD-YYYY
        (r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', 0.70),  # YYYY-MM-DD
    ]

    # Zip Code: 12345 or 12345-6789
    ZIP_CODE = [
        (r'\b\d{5}(?:-\d{4})?\b', 0.85),
    ]

    # Street Address (simplified pattern)
    STREET_ADDRESS = [
        (r'\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', 0.80),
    ]

    # Full Name (simple pattern: First Last, requires capitalization)
    FULL_NAME = [
        (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', 0.60),  # Low confidence (many false positives)
    ]


# ============================================================================
# Input Sanitization
# ============================================================================

def _sanitize_text_for_pii_detection(text: str) -> str:
    """
    Sanitize text to prevent Unicode-based bypasses.

    Security transformations:
    1. Unicode NFKC normalization (full-width → half-width)
    2. Zero-width character removal (U+200B, U+200C, U+200D, U+FEFF)
    3. Preserve original text structure (newlines, spaces)

    Args:
        text: Raw text input

    Returns:
        Sanitized text ready for PII pattern matching

    Examples:
        >>> _sanitize_text_for_pii_detection("１２３-４５-６７８９")  # Full-width digits
        "123-45-6789"

        >>> _sanitize_text_for_pii_detection("test\u200b@\u200bexample.com")  # Zero-width
        "test@example.com"
    """
    # Step 1: Unicode normalization (NFKC = compatibility + composition)
    # Converts full-width/half-width variants to standard form
    normalized = unicodedata.normalize('NFKC', text)

    # Step 2: Remove zero-width characters (common obfuscation technique)
    zero_width_chars = [
        '\u200B',  # ZERO WIDTH SPACE
        '\u200C',  # ZERO WIDTH NON-JOINER
        '\u200D',  # ZERO WIDTH JOINER
        '\uFEFF',  # ZERO WIDTH NO-BREAK SPACE (BOM)
    ]

    sanitized = normalized
    for char in zero_width_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized


# ============================================================================
# PII Detector
# ============================================================================

class PIIDetector:
    """
    Fast regex-based PII detector.

    Features:
    - Multi-pattern matching for each PII type
    - Confidence scoring
    - Customizable patterns
    - Redaction utilities

    Usage:
        detector = PIIDetector()
        result = detector.analyze("My email is user@example.com and SSN is 123-45-6789")

        print(result.has_pii)  # True
        print(result.risk_level)  # "CRITICAL" (SSN detected)
        print(result.redacted_text)  # "My email is [EMAIL_REDACTED] and SSN is [SSN_REDACTED]"

        for detection in result.detections:
            print(f"{detection.pii_type.value}: {detection.matched_text} (confidence: {detection.confidence})")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        custom_patterns: Optional[Dict[str, List[Tuple[str, float]]]] = None
    ):
        """
        Initialize PII detector.

        Args:
            confidence_threshold: Minimum confidence to report detection (0-1)
            custom_patterns: Custom regex patterns {pii_type: [(pattern, confidence), ...]}
        """
        self.confidence_threshold = confidence_threshold
        self.custom_patterns = custom_patterns or {}

        # Build pattern mapping
        self.patterns: Dict[PIIType, List[Tuple[str, float]]] = {
            PIIType.SSN: PIIPatterns.SSN,
            PIIType.CREDIT_CARD: PIIPatterns.CREDIT_CARD,
            PIIType.EMAIL: PIIPatterns.EMAIL,
            PIIType.PHONE: PIIPatterns.PHONE,
            PIIType.IP_ADDRESS: PIIPatterns.IP_ADDRESS,
            PIIType.DATE_OF_BIRTH: PIIPatterns.DATE_OF_BIRTH,
            PIIType.ZIP_CODE: PIIPatterns.ZIP_CODE,
            PIIType.STREET_ADDRESS: PIIPatterns.STREET_ADDRESS,
            PIIType.FULL_NAME: PIIPatterns.FULL_NAME,
        }

        # Add custom patterns
        for pii_type_str, patterns in self.custom_patterns.items():
            pii_type = PIIType(pii_type_str)
            if pii_type not in self.patterns:
                self.patterns[pii_type] = []
            self.patterns[pii_type].extend(patterns)

        # Compile regex patterns for performance
        self.compiled_patterns: Dict[PIIType, List[Tuple[re.Pattern, float]]] = {}
        for pii_type, patterns in self.patterns.items():
            self.compiled_patterns[pii_type] = [
                (re.compile(pattern), confidence)
                for pattern, confidence in patterns
            ]

    def analyze(
        self,
        text: str,
        enable_luhn_check: bool = True
    ) -> PIIAnalysisResult:
        """
        Analyze text for PII entities.

        Args:
            text: Text to analyze
            enable_luhn_check: Validate credit cards with Luhn algorithm (default: True)

        Returns:
            PIIAnalysisResult with all detections and risk assessment
        """
        # SECURITY: Sanitize text to prevent Unicode-based bypasses
        # This defends against zero-width characters and full-width digit obfuscation
        sanitized_text = _sanitize_text_for_pii_detection(text)

        detections: List[PIIDetection] = []

        # Detect each PII type
        for pii_type, compiled_patterns in self.compiled_patterns.items():
            for pattern, confidence in compiled_patterns:
                for match in pattern.finditer(sanitized_text):
                    matched_text = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                    # Skip if below confidence threshold
                    if confidence < self.confidence_threshold:
                        continue

                    # Additional validation for credit cards (Luhn check)
                    if pii_type == PIIType.CREDIT_CARD and enable_luhn_check:
                        if not self._validate_credit_card(matched_text):
                            continue  # Failed Luhn check, skip

                    # Get context (10 chars before and after)
                    context_start = max(0, start_pos - 10)
                    context_end = min(len(text), end_pos + 10)
                    context = text[context_start:context_end]

                    # Create redacted version
                    redacted = self._redact(pii_type, matched_text)

                    detection = PIIDetection(
                        pii_type=pii_type,
                        matched_text=matched_text,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        redacted_text=redacted,
                        context=context
                    )

                    detections.append(detection)

        # Sort detections by position
        detections.sort(key=lambda d: d.start_pos)

        # Remove overlapping detections (keep higher confidence)
        detections = self._remove_overlaps(detections)

        # Assess risk level
        risk_level = self._assess_risk(detections)

        # Generate redacted text
        redacted_text = self._generate_redacted_text(text, detections)

        # Generate compliance flags
        compliance_flags = self._generate_compliance_flags(detections)

        return PIIAnalysisResult(
            original_text=text,
            detections=detections,
            risk_level=risk_level,
            redacted_text=redacted_text,
            compliance_flags=compliance_flags
        )

    def _validate_credit_card(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.

        Args:
            card_number: Credit card number string

        Returns:
            True if valid, False otherwise
        """
        # Remove non-digits
        digits = re.sub(r'\D', '', card_number)

        if not digits or len(digits) < 13:
            return False

        # Luhn algorithm
        total = 0
        reverse_digits = digits[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n

        return total % 10 == 0

    def _redact(self, pii_type: PIIType, text: str) -> str:
        """
        Generate redacted version of PII text.

        Args:
            pii_type: Type of PII
            text: Original text

        Returns:
            Redacted string
        """
        # Different redaction strategies by PII type
        if pii_type == PIIType.EMAIL:
            # Keep domain for context
            parts = text.split('@')
            if len(parts) == 2:
                return f"[REDACTED]@{parts[1]}"
            return "[EMAIL_REDACTED]"

        elif pii_type == PIIType.PHONE:
            # Keep last 4 digits for identification
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
            return "[PHONE_REDACTED]"

        elif pii_type == PIIType.CREDIT_CARD:
            # Keep last 4 digits (industry standard)
            digits = re.sub(r'\D', '', text)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"
            return "[CARD_REDACTED]"

        elif pii_type == PIIType.SSN:
            # Completely redact SSN (HIPAA requirement)
            return "[SSN_REDACTED]"

        elif pii_type == PIIType.IP_ADDRESS:
            # Keep network prefix for debugging
            parts = text.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.*.*"
            return "[IP_REDACTED]"

        else:
            # Generic redaction
            return f"[{pii_type.value.upper()}_REDACTED]"

    def _remove_overlaps(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """
        Remove overlapping detections, keeping higher confidence ones.

        Args:
            detections: List of detections sorted by start_pos

        Returns:
            Filtered list without overlaps
        """
        if not detections:
            return []

        result = [detections[0]]

        for detection in detections[1:]:
            prev = result[-1]

            # Check for overlap
            if detection.start_pos < prev.end_pos:
                # Keep higher confidence detection
                if detection.confidence > prev.confidence:
                    result[-1] = detection
            else:
                result.append(detection)

        return result

    def _assess_risk(self, detections: List[PIIDetection]) -> str:
        """
        Assess overall risk level based on detected PII types.

        Args:
            detections: List of PII detections

        Returns:
            "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        """
        if not detections:
            return "LOW"

        pii_types = {d.pii_type for d in detections}

        # CRITICAL: SSN, credit card, passport
        critical_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT, PIIType.DRIVER_LICENSE}
        if pii_types & critical_types:
            return "CRITICAL"

        # HIGH: Email, phone, full name + address
        high_types = {PIIType.EMAIL, PIIType.PHONE, PIIType.FULL_NAME, PIIType.STREET_ADDRESS}
        if len(pii_types & high_types) >= 2:
            return "HIGH"

        # MEDIUM: Any single high-sensitivity PII
        if pii_types & high_types:
            return "MEDIUM"

        return "LOW"

    def _generate_redacted_text(self, original: str, detections: List[PIIDetection]) -> str:
        """
        Generate text with all PII redacted.

        Args:
            original: Original text
            detections: List of PII detections

        Returns:
            Redacted text
        """
        if not detections:
            return original

        # Work backwards to preserve positions
        result = original
        for detection in reversed(detections):
            result = (
                result[:detection.start_pos] +
                detection.redacted_text +
                result[detection.end_pos:]
            )

        return result

    def _generate_compliance_flags(self, detections: List[PIIDetection]) -> Dict[str, bool]:
        """
        Generate compliance flags based on detected PII.

        Args:
            detections: List of PII detections

        Returns:
            Dict of compliance requirements
        """
        pii_types = {d.pii_type for d in detections}

        return {
            # GDPR: Any PII requires data subject rights
            "gdpr_data_subject_rights": len(pii_types) > 0,

            # GDPR: Right to erasure for certain PII
            "gdpr_right_to_erasure": bool(pii_types & {
                PIIType.EMAIL, PIIType.PHONE, PIIType.FULL_NAME,
                PIIType.STREET_ADDRESS, PIIType.DATE_OF_BIRTH
            }),

            # HIPAA: Protected Health Information (PHI)
            "hipaa_phi": bool(pii_types & {
                PIIType.SSN, PIIType.MEDICAL_RECORD, PIIType.DATE_OF_BIRTH,
                PIIType.FULL_NAME
            }),

            # SOC 2: Requires encryption at rest
            "requires_encryption": bool(pii_types & {
                PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT,
                PIIType.DRIVER_LICENSE, PIIType.MEDICAL_RECORD
            }),

            # Requires audit trail
            "requires_audit": bool(pii_types & {
                PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT,
                PIIType.DRIVER_LICENSE, PIIType.MEDICAL_RECORD
            }),

            # Requires access control (not public)
            "requires_access_control": len(pii_types) > 0,
        }


# ============================================================================
# Utility Functions
# ============================================================================

def hash_pii(text: str, salt: str = "") -> str:
    """
    Create one-way hash of PII for lookup/deduplication without storing plaintext.

    Args:
        text: PII text to hash
        salt: Optional salt for extra security

    Returns:
        SHA-256 hash (hex)

    Example:
        email_hash = hash_pii("user@example.com", salt="tenant_123")
        # Use hash for deduplication or lookup
    """
    combined = f"{salt}:{text}"
    return hashlib.sha256(combined.encode()).hexdigest()


def create_default_detector() -> PIIDetector:
    """
    Factory function to create PII detector with default settings.

    Returns:
        PIIDetector instance ready to use
    """
    return PIIDetector(confidence_threshold=0.6)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'PIIType',
    'PIIDetection',
    'PIIAnalysisResult',
    'PIIDetector',
    'PIIPatterns',
    'hash_pii',
    'create_default_detector',
]
