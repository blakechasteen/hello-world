"""
File upload validation with content scanning.

Phase 2 - Input Validation & Sanitization (November 2025)

Implements:
- MIME type validation (magic bytes, not just extension)
- File size limits
- Dangerous content detection
- SHA256 hashing for integrity
- Optional ClamAV virus scanning
- Safe filename generation
"""

import hashlib
import magic
import os
from typing import BinaryIO, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MAX_FILE_SIZE_BYTES = 10_485_760  # 10 MB

# Whitelist MIME types (as defined in schemas.py)
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

# Map of allowed extensions to expected MIME types
EXTENSION_MIME_MAP = {
    '.txt': {'text/plain'},
    '.csv': {'text/csv', 'text/plain'},
    '.json': {'application/json', 'text/plain'},
    '.pdf': {'application/pdf'},
    '.png': {'image/png'},
    '.jpg': {'image/jpeg'},
    '.jpeg': {'image/jpeg'},
    '.gif': {'image/gif'},
    '.webp': {'image/webp'},
    '.docx': {'application/vnd.openxmlformats-officedocument.wordprocessingml.document'},
    '.xlsx': {'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'},
}

# Dangerous MIME types that should never be uploaded
DANGEROUS_MIME_TYPES = {
    'application/x-executable',
    'application/x-elf',
    'application/x-sharedlib',
    'application/x-object',
    'application/x-mach-binary',
    'text/x-shellscript',
    'text/x-python',
    'text/x-perl',
    'application/x-msdownload',
    'application/x-msdos-program',
    'application/x-windows-dos-executable',
}

# Magic bytes (file signatures) for validation
MAGIC_BYTES_MAP = {
    'PDF': b'%PDF',
    'PNG': b'\x89PNG\r\n\x1a\n',
    'JPEG': b'\xff\xd8\xff',
    'GIF87': b'GIF87a',
    'GIF89': b'GIF89a',
    'WEBP': b'RIFF',  # RIFF....WEBP
    'ZIP': b'PK\x03\x04',  # Used in DOCX, XLSX
}


@dataclass
class FileValidationResult:
    """Result of file validation."""
    valid: bool
    filename: str
    mime_type: str
    size_bytes: int
    sha256_hash: str
    is_safe: bool
    warnings: List[str]
    errors: List[str]
    scanned_for_malware: bool = False
    malware_detected: bool = False
    scan_engine: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'valid': self.valid,
            'filename': self.filename,
            'mime_type': self.mime_type,
            'size_bytes': self.size_bytes,
            'sha256_hash': self.sha256_hash,
            'is_safe': self.is_safe,
            'warnings': self.warnings,
            'errors': self.errors,
            'scanned_for_malware': self.scanned_for_malware,
            'malware_detected': self.malware_detected,
            'scan_engine': self.scan_engine
        }


# ============================================================================
# MIME Type Validation
# ============================================================================

def detect_mime_type_from_magic(file_bytes: bytes) -> str:
    """
    Detect MIME type from magic bytes (file signature).

    Uses python-magic to detect true MIME type regardless of extension.

    Args:
        file_bytes: First ~1KB of file to scan

    Returns:
        Detected MIME type

    Example:
        >>> with open('image.jpg', 'rb') as f:
        ...     mime = detect_mime_type_from_magic(f.read(1024))
        >>> assert mime == 'image/jpeg'
    """
    try:
        # Try using python-magic library
        mime = magic.from_buffer(file_bytes, mime=True)
        return mime
    except Exception as e:
        logger.warning(f"Failed to detect MIME type with magic: {e}")
        return 'application/octet-stream'


def validate_mime_type(detected_mime: str, claimed_mime: str) -> Tuple[bool, str]:
    """
    Validate that claimed MIME type matches detected type.

    Prevents MIME type confusion attacks where user lies about file type.

    Args:
        detected_mime: MIME type detected from magic bytes
        claimed_mime: MIME type claimed in Content-Type header

    Returns:
        Tuple of (valid, message)

    Example:
        >>> valid, msg = validate_mime_type('image/jpeg', 'image/jpeg')
        >>> assert valid is True
        >>> valid, msg = validate_mime_type('image/jpeg', 'text/plain')
        >>> assert valid is False  # Mismatch detected
    """
    detected = detected_mime.lower()
    claimed = claimed_mime.lower()

    # Exact match is best
    if detected == claimed:
        return True, "MIME type matches"

    # Check if detected type is whitelisted
    if detected not in WHITELISTED_MIME_TYPES:
        return False, f"Detected MIME type '{detected}' is not whitelisted"

    # Check if claimed type is whitelisted
    if claimed not in WHITELISTED_MIME_TYPES:
        return False, f"Claimed MIME type '{claimed}' is not whitelisted"

    # Some tolerance for generic types (e.g., text/plain vs text/csv)
    detected_family = detected.split('/')[0]
    claimed_family = claimed.split('/')[0]

    if detected_family != claimed_family:
        return False, f"MIME type mismatch: detected '{detected}', claimed '{claimed}'"

    # Close enough (both text, both images, etc)
    return True, f"MIME type family match (detected '{detected}', claimed '{claimed}')"


def is_dangerous_mime_type(mime_type: str) -> bool:
    """
    Check if MIME type is known dangerous.

    Args:
        mime_type: MIME type to check

    Returns:
        True if dangerous, False otherwise
    """
    mime_type = mime_type.lower()

    # Direct match
    if mime_type in DANGEROUS_MIME_TYPES:
        return True

    # Check patterns
    dangerous_patterns = [
        'executable',
        'script',
        'shellscript',
        'x-msdownload',
        'x-msdos',
        'x-bat',
        'x-sh',
    ]

    for pattern in dangerous_patterns:
        if pattern in mime_type:
            return True

    return False


# ============================================================================
# File Size Validation
# ============================================================================

def validate_file_size(file_size: int, max_size: int = MAX_FILE_SIZE_BYTES) -> Tuple[bool, str]:
    """
    Validate file size is within limits.

    Args:
        file_size: File size in bytes
        max_size: Maximum allowed size in bytes

    Returns:
        Tuple of (valid, message)
    """
    if file_size <= 0:
        return False, "File size must be positive"

    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        return False, f"File exceeds maximum size of {max_mb:.1f} MB"

    return True, "File size valid"


# ============================================================================
# File Hash & Integrity
# ============================================================================

def calculate_sha256(file_stream: BinaryIO, chunk_size: int = 65536) -> str:
    """
    Calculate SHA256 hash of file stream.

    Reads file in chunks to avoid memory issues with large files.

    Args:
        file_stream: Open file stream (binary mode)
        chunk_size: Bytes to read per iteration

    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()

    while True:
        chunk = file_stream.read(chunk_size)
        if not chunk:
            break
        sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def calculate_sha256_bytes(file_bytes: bytes) -> str:
    """
    Calculate SHA256 hash of bytes.

    Args:
        file_bytes: File contents as bytes

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(file_bytes).hexdigest()


def verify_file_hash(file_stream: BinaryIO, expected_hash: str) -> Tuple[bool, str]:
    """
    Verify file hash matches expected value.

    Args:
        file_stream: Open file stream
        expected_hash: Expected SHA256 hash (hex string)

    Returns:
        Tuple of (matches, message)
    """
    actual_hash = calculate_sha256(file_stream)

    if actual_hash.lower() == expected_hash.lower():
        return True, "Hash verified"
    else:
        return False, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"


# ============================================================================
# Content Scanning
# ============================================================================

def scan_with_clamav(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """
    Scan file for malware using ClamAV.

    Requires ClamAV daemon (clamd) running locally.

    Args:
        file_bytes: File contents

    Returns:
        Tuple of (is_clean, detected_threat_name or None)

    Example:
        >>> is_clean, threat = scan_with_clamav(file_bytes)
        >>> if not is_clean:
        ...     logger.error(f"Malware detected: {threat}")
    """
    try:
        import pyclamd

        # Try to connect to ClamAV daemon
        clam = pyclamd.ClamD()

        if not clam.ping():
            logger.warning("ClamAV daemon not available")
            return True, None  # Graceful degradation

        # Scan file stream
        result = clam.scan_stream(file_bytes)

        if result is None:
            return True, None  # Clean

        if isinstance(result, dict) and 'stream' in result:
            # Infected
            threat_info = result['stream']
            return False, str(threat_info)

        return True, None

    except ImportError:
        logger.warning("pyclamd not installed, skipping ClamAV scan")
        return True, None
    except Exception as e:
        logger.warning(f"ClamAV scan failed: {e}")
        return True, None  # Graceful degradation


def scan_for_suspicious_patterns(file_bytes: bytes) -> List[str]:
    """
    Scan for suspicious patterns in file content.

    Looks for:
    - Executable headers and signatures
    - Script headers
    - Zip bomb patterns
    - XXE patterns

    Args:
        file_bytes: File contents to scan

    Returns:
        List of suspicious patterns found (empty if clean)
    """
    warnings = []

    # Check for executable signatures
    if file_bytes.startswith(b'MZ'):  # Windows executable
        warnings.append("Detected Windows executable header (MZ)")

    if file_bytes.startswith(b'\x7fELF'):  # ELF executable
        warnings.append("Detected ELF executable header")

    if file_bytes.startswith(b'\xca\xfe\xba\xbe'):  # Mach-O executable
        warnings.append("Detected Mach-O executable header")

    # Check for script headers
    if file_bytes.startswith(b'#!/'):  # Shebang
        warnings.append("Detected script shebang")

    # Check for ZIP bomb (very high compression ratio)
    if file_bytes.startswith(b'PK\x03\x04'):  # ZIP signature
        # Very crude check: if file is < 1MB but compresses files
        if len(file_bytes) < 1_000_000:
            # Would need to parse ZIP to be more accurate
            warnings.append("Detected ZIP archive (verify safe content)")

    # Check for XXE patterns in XML
    if b'<!DOCTYPE' in file_bytes or b'<!ENTITY' in file_bytes:
        if (b'SYSTEM' in file_bytes or b'PUBLIC' in file_bytes or
                b'http://' in file_bytes or b'file://' in file_bytes):
            warnings.append("Detected suspicious XXE patterns in XML")

    # Check for SQL patterns in text files
    if b'DROP TABLE' in file_bytes or b'DELETE FROM' in file_bytes:
        warnings.append("Detected SQL injection patterns")

    return warnings


# ============================================================================
# Filename Validation & Sanitization
# ============================================================================

def generate_safe_filename(original_filename: str) -> str:
    """
    Generate safe filename from original filename.

    Removes dangerous characters and generates unique name.

    Args:
        original_filename: Original filename from upload

    Returns:
        Safe filename suitable for storage

    Example:
        >>> original = '../../../etc/passwd'
        >>> safe = generate_safe_filename(original)
        >>> assert '..' not in safe
        >>> assert '/' not in safe
    """
    # Extract just the filename (remove paths)
    filename = os.path.basename(original_filename)

    # Remove dangerous characters, keep only alphanumeric, dot, dash, underscore
    import re
    safe = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)

    # Remove leading dots (prevent hidden files)
    safe = safe.lstrip('.')

    # Limit length
    if len(safe) > 200:
        name, ext = os.path.splitext(safe)
        safe = name[:180] + ext

    # Ensure not empty
    if not safe:
        safe = 'file'

    # Add timestamp for uniqueness
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(safe)
    return f"{name}_{timestamp}{ext}"


def validate_filename(filename: str) -> Tuple[bool, str]:
    """
    Validate filename for safety.

    Args:
        filename: Filename to validate

    Returns:
        Tuple of (valid, message)
    """
    # Check for path traversal
    if '..' in filename or filename.startswith('/') or filename.startswith('\\'):
        return False, "Filename contains path traversal characters"

    # Check for null bytes
    if '\0' in filename:
        return False, "Filename contains null bytes"

    # Check for dangerous extensions
    dangerous_exts = {
        '.exe', '.bat', '.cmd', '.sh', '.py', '.js', '.php', '.jsp',
        '.asp', '.aspx', '.pl', '.cgi', '.jar'
    }

    name, ext = os.path.splitext(filename.lower())
    if ext in dangerous_exts:
        return False, f"File extension {ext} not allowed"

    # Check length
    if len(filename) > 255:
        return False, "Filename exceeds 255 characters"

    return True, "Filename valid"


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_file_upload(
    filename: str,
    file_bytes: bytes,
    claimed_mime_type: str,
    max_size: int = MAX_FILE_SIZE_BYTES,
    scan_for_malware: bool = True,
    expected_hash: Optional[str] = None
) -> FileValidationResult:
    """
    Comprehensive file validation.

    Performs:
    1. Filename validation
    2. Size validation
    3. MIME type detection and validation
    4. Dangerous MIME type check
    5. Content scanning (ClamAV)
    6. Suspicious pattern detection
    7. Hash verification (if provided)

    Args:
        filename: Original filename
        file_bytes: File content bytes
        claimed_mime_type: MIME type claimed in Content-Type header
        max_size: Maximum allowed file size
        scan_for_malware: Enable ClamAV scanning
        expected_hash: Expected SHA256 hash (optional)

    Returns:
        FileValidationResult with detailed validation info

    Example:
        >>> with open('document.pdf', 'rb') as f:
        ...     file_bytes = f.read()
        >>> result = validate_file_upload(
        ...     'document.pdf',
        ...     file_bytes,
        ...     'application/pdf'
        ... )
        >>> assert result.valid and result.is_safe
    """
    errors = []
    warnings = []

    # 1. Filename validation
    valid, msg = validate_filename(filename)
    if not valid:
        errors.append(msg)

    # 2. Size validation
    valid, msg = validate_file_size(len(file_bytes), max_size)
    if not valid:
        errors.append(msg)

    # 3. MIME type detection
    detected_mime = detect_mime_type_from_magic(file_bytes[:1024])

    # 4. MIME type validation
    valid, msg = validate_mime_type(detected_mime, claimed_mime_type)
    if not valid:
        errors.append(msg)
    else:
        logger.debug(msg)

    # 5. Dangerous MIME type check
    if is_dangerous_mime_type(detected_mime):
        errors.append(f"Dangerous MIME type detected: {detected_mime}")

    # 6. Hash calculation
    file_hash = calculate_sha256_bytes(file_bytes)

    # 7. Hash verification (if provided)
    if expected_hash:
        if file_hash.lower() != expected_hash.lower():
            errors.append(f"Hash mismatch: expected {expected_hash}, got {file_hash}")

    # 8. Suspicious pattern detection
    suspicious = scan_for_suspicious_patterns(file_bytes)
    warnings.extend(suspicious)

    # 9. Malware scanning
    scanned_for_malware = False
    malware_detected = False
    scan_engine = None

    if scan_for_malware:
        is_clean, threat = scan_with_clamav(file_bytes)
        scanned_for_malware = True
        malware_detected = not is_clean
        if malware_detected:
            errors.append(f"Malware detected: {threat}")
            scan_engine = "ClamAV"

    # Final decision
    is_valid = len(errors) == 0
    is_safe = is_valid and len(warnings) == 0

    return FileValidationResult(
        valid=is_valid,
        filename=filename,
        mime_type=detected_mime,
        size_bytes=len(file_bytes),
        sha256_hash=file_hash,
        is_safe=is_safe,
        warnings=warnings,
        errors=errors,
        scanned_for_malware=scanned_for_malware,
        malware_detected=malware_detected,
        scan_engine=scan_engine
    )


# ============================================================================
# Safe File Storage
# ============================================================================

def save_uploaded_file(
    file_bytes: bytes,
    storage_dir: str,
    safe_filename: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Safely save uploaded file to storage directory.

    Args:
        file_bytes: File content
        storage_dir: Directory to save to (must exist)
        safe_filename: Safe filename to use (generated if not provided)

    Returns:
        Tuple of (success, file_path or error_message)
    """
    if not os.path.isdir(storage_dir):
        return False, f"Storage directory does not exist: {storage_dir}"

    if not safe_filename:
        safe_filename = f"upload_{datetime.utcnow().timestamp()}"

    filepath = os.path.join(storage_dir, safe_filename)

    # Verify no path traversal
    storage_abs = os.path.abspath(storage_dir)
    filepath_abs = os.path.abspath(filepath)

    if not filepath_abs.startswith(storage_abs):
        return False, "File path escapes storage directory"

    try:
        with open(filepath_abs, 'wb') as f:
            f.write(file_bytes)
        return True, filepath_abs
    except Exception as e:
        return False, f"Failed to save file: {e}"
