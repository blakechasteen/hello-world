"""
PKI/mTLS - Certificate-based Node Authentication.

Provides mutual TLS authentication for Portal system components:
- CA certificate management and generation
- Per-node certificate issuance
- SSL context creation for mTLS
- Certificate verification and validation

Directory structure:
    certs/
      ca.crt           # CA certificate (public)
      ca.key           # CA private key (keep secure!)
      portal.crt       # Portal server certificate
      portal.key       # Portal server private key
      nodes/
        {node_id}.crt  # Per-node certificates
        {node_id}.key  # Per-node private keys

Usage:
    from hololoom.portal.shared.pki import PKIManager

    pki = PKIManager(certs_dir=Path("./certs"))

    # Generate CA (one-time setup)
    pki.generate_ca(common_name="Portal CA", validity_days=3650)

    # Issue node certificate
    cert_path, key_path = pki.issue_node_certificate("node-1")

    # Create SSL context for mTLS server
    ssl_ctx = pki.create_server_ssl_context(
        cert_path=Path("certs/portal.crt"),
        key_path=Path("certs/portal.key")
    )

    # Verify client certificate
    info = pki.verify_client_certificate(cert_pem)
    if info and info.is_valid:
        print(f"Authenticated: {info.node_id}")
"""

import ssl
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import hashlib
import base64
import json

# Use cryptography library for certificate operations
try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtensionOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


@dataclass
class CertificateInfo:
    """
    Certificate metadata extracted from X.509 certificate.

    Used for authentication decisions and audit logging.
    """

    common_name: str
    node_id: str
    expires_at: datetime
    issuer: str
    serial_number: int
    is_valid: bool
    not_valid_before: Optional[datetime] = None
    fingerprint: Optional[str] = None  # SHA256 fingerprint
    subject_alt_names: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "common_name": self.common_name,
            "node_id": self.node_id,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "issuer": self.issuer,
            "serial_number": self.serial_number,
            "is_valid": self.is_valid,
            "not_valid_before": self.not_valid_before.isoformat() if self.not_valid_before else None,
            "fingerprint": self.fingerprint,
            "subject_alt_names": self.subject_alt_names,
            "error_message": self.error_message,
        }


@dataclass
class CAInfo:
    """Information about the Certificate Authority."""

    common_name: str
    organization: str
    validity_days: int
    key_size: int
    created_at: datetime
    expires_at: datetime
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "common_name": self.common_name,
            "organization": self.organization,
            "validity_days": self.validity_days,
            "key_size": self.key_size,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "fingerprint": self.fingerprint,
        }


class PKIManager:
    """
    Manages certificates for Portal mTLS authentication.

    Provides:
    - CA certificate generation
    - Server certificate generation
    - Per-node certificate issuance
    - SSL context creation for mTLS
    - Certificate verification

    Args:
        certs_dir: Root directory for certificate storage
        ca_key_size: RSA key size for CA (default: 4096)
        node_key_size: RSA key size for nodes (default: 2048)
        organization: Organization name for certificates
    """

    def __init__(
        self,
        certs_dir: Path,
        ca_key_size: int = 4096,
        node_key_size: int = 2048,
        organization: str = "HoloLoom Portal",
    ):
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError(
                "cryptography library required for PKI. "
                "Install with: pip install cryptography"
            )

        self.certs_dir = Path(certs_dir)
        self.ca_key_size = ca_key_size
        self.node_key_size = node_key_size
        self.organization = organization

        # Standard paths
        self.ca_cert_path = self.certs_dir / "ca.crt"
        self.ca_key_path = self.certs_dir / "ca.key"
        self.nodes_dir = self.certs_dir / "nodes"

        # Ensure directories exist
        self.certs_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)

    def generate_ca(
        self,
        common_name: str = "Portal CA",
        validity_days: int = 3650,
        overwrite: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Generate Certificate Authority certificate and private key.

        The CA is used to sign all other certificates in the system.
        This should be run once during initial setup.

        Args:
            common_name: CA common name (default: "Portal CA")
            validity_days: Certificate validity period (default: 10 years)
            overwrite: If True, overwrite existing CA (dangerous!)

        Returns:
            Tuple of (cert_path, key_path)

        Raises:
            FileExistsError: If CA exists and overwrite=False
        """
        if self.ca_cert_path.exists() and not overwrite:
            raise FileExistsError(
                f"CA already exists at {self.ca_cert_path}. "
                "Use overwrite=True to replace (destroys existing PKI!)"
            )

        # Generate CA private key
        ca_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.ca_key_size,
            backend=default_backend(),
        )

        # Build CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.organization),
        ])

        now = datetime.utcnow()
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256(), default_backend())
        )

        # Save CA certificate (PEM format)
        with open(self.ca_cert_path, "wb") as f:
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

        # Save CA private key (PEM format, unencrypted)
        # NOTE: In production, consider encrypting with a passphrase
        with open(self.ca_key_path, "wb") as f:
            f.write(ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        # Set restrictive permissions on private key
        try:
            os.chmod(self.ca_key_path, 0o600)
        except OSError:
            pass  # Windows may not support chmod

        return self.ca_cert_path, self.ca_key_path

    def _load_ca(self) -> Tuple[Any, Any]:
        """Load CA certificate and key from disk."""
        if not self.ca_cert_path.exists():
            raise FileNotFoundError(
                f"CA certificate not found at {self.ca_cert_path}. "
                "Run generate_ca() first."
            )
        if not self.ca_key_path.exists():
            raise FileNotFoundError(
                f"CA private key not found at {self.ca_key_path}. "
                "Run generate_ca() first."
            )

        with open(self.ca_cert_path, "rb") as f:
            ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        with open(self.ca_key_path, "rb") as f:
            ca_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend(),
            )

        return ca_cert, ca_key

    def issue_node_certificate(
        self,
        node_id: str,
        validity_days: int = 365,
        san_dns_names: Optional[List[str]] = None,
        san_ip_addresses: Optional[List[str]] = None,
    ) -> Tuple[Path, Path]:
        """
        Issue a certificate for a node.

        Args:
            node_id: Unique node identifier (becomes CN)
            validity_days: Certificate validity (default: 1 year)
            san_dns_names: Additional DNS names for SAN extension
            san_ip_addresses: IP addresses for SAN extension

        Returns:
            Tuple of (cert_path, key_path)
        """
        ca_cert, ca_key = self._load_ca()

        # Generate node private key
        node_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.node_key_size,
            backend=default_backend(),
        )

        # Build subject
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, node_id),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, f"{self.organization} Node"),
        ])

        # Build SAN (Subject Alternative Names)
        san_entries = [x509.DNSName(node_id)]
        if san_dns_names:
            for dns in san_dns_names:
                san_entries.append(x509.DNSName(dns))
        if san_ip_addresses:
            from ipaddress import ip_address
            for ip in san_ip_addresses:
                san_entries.append(x509.IPAddress(ip_address(ip)))

        now = datetime.utcnow()
        node_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(node_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )
            .add_extension(
                x509.SubjectAlternativeName(san_entries),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256(), default_backend())
        )

        # Save paths
        cert_path = self.nodes_dir / f"{node_id}.crt"
        key_path = self.nodes_dir / f"{node_id}.key"

        # Save certificate
        with open(cert_path, "wb") as f:
            f.write(node_cert.public_bytes(serialization.Encoding.PEM))

        # Save private key
        with open(key_path, "wb") as f:
            f.write(node_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        # Set restrictive permissions on private key
        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass  # Windows may not support chmod

        return cert_path, key_path

    def issue_server_certificate(
        self,
        server_name: str = "portal",
        validity_days: int = 365,
        san_dns_names: Optional[List[str]] = None,
        san_ip_addresses: Optional[List[str]] = None,
    ) -> Tuple[Path, Path]:
        """
        Issue a certificate for the Portal server.

        Args:
            server_name: Server name (default: "portal")
            validity_days: Certificate validity (default: 1 year)
            san_dns_names: DNS names for SAN (e.g., ["localhost", "portal.example.com"])
            san_ip_addresses: IP addresses for SAN (e.g., ["127.0.0.1"])

        Returns:
            Tuple of (cert_path, key_path)
        """
        # Default SAN entries for development
        if san_dns_names is None:
            san_dns_names = ["localhost"]
        if san_ip_addresses is None:
            san_ip_addresses = ["127.0.0.1"]

        ca_cert, ca_key = self._load_ca()

        # Generate server private key
        server_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.node_key_size,
            backend=default_backend(),
        )

        # Build subject
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, server_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, f"{self.organization} Server"),
        ])

        # Build SAN
        from ipaddress import ip_address
        san_entries = [x509.DNSName(server_name)]
        for dns in san_dns_names:
            if dns != server_name:
                san_entries.append(x509.DNSName(dns))
        for ip in san_ip_addresses:
            san_entries.append(x509.IPAddress(ip_address(ip)))

        now = datetime.utcnow()
        server_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + timedelta(days=validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )
            .add_extension(
                x509.SubjectAlternativeName(san_entries),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256(), default_backend())
        )

        # Save to root certs dir (not nodes/)
        cert_path = self.certs_dir / f"{server_name}.crt"
        key_path = self.certs_dir / f"{server_name}.key"

        with open(cert_path, "wb") as f:
            f.write(server_cert.public_bytes(serialization.Encoding.PEM))

        with open(key_path, "wb") as f:
            f.write(server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        try:
            os.chmod(key_path, 0o600)
        except OSError:
            pass

        return cert_path, key_path

    def create_server_ssl_context(
        self,
        cert_path: Path,
        key_path: Path,
        require_client_cert: bool = True,
    ) -> ssl.SSLContext:
        """
        Create SSL context for mTLS server.

        Args:
            cert_path: Path to server certificate
            key_path: Path to server private key
            require_client_cert: If True, require client certificate (mTLS)

        Returns:
            Configured SSL context for server
        """
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(cert_path), str(key_path))
        context.load_verify_locations(str(self.ca_cert_path))

        if require_client_cert:
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_NONE

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")

        return context

    def create_client_ssl_context(
        self,
        cert_path: Path,
        key_path: Path,
    ) -> ssl.SSLContext:
        """
        Create SSL context for mTLS client (node connecting to server).

        Args:
            cert_path: Path to client certificate
            key_path: Path to client private key

        Returns:
            Configured SSL context for client
        """
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_cert_chain(str(cert_path), str(key_path))
        context.load_verify_locations(str(self.ca_cert_path))
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        return context

    def verify_client_certificate(self, cert_pem: bytes) -> Optional[CertificateInfo]:
        """
        Verify client certificate and extract information.

        Args:
            cert_pem: PEM-encoded certificate bytes

        Returns:
            CertificateInfo if certificate can be parsed, None on error
        """
        try:
            cert = x509.load_pem_x509_certificate(cert_pem, default_backend())

            # Check expiry
            now = datetime.utcnow()
            is_valid = cert.not_valid_before <= now <= cert.not_valid_after

            # Extract CN
            cn_attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            common_name = cn_attrs[0].value if cn_attrs else ""

            # Extract issuer CN
            issuer_attrs = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
            issuer = issuer_attrs[0].value if issuer_attrs else ""

            # Extract SANs
            san_list = []
            try:
                san_ext = cert.extensions.get_extension_for_oid(
                    ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                )
                for name in san_ext.value:
                    if isinstance(name, x509.DNSName):
                        san_list.append(name.value)
            except x509.ExtensionNotFound:
                pass

            # Calculate fingerprint (with colons for standard format)
            fp_hex = cert.fingerprint(hashes.SHA256()).hex().upper()
            fingerprint = ":".join(fp_hex[i:i+2] for i in range(0, len(fp_hex), 2))

            # Verify signature against our CA
            error_message = None
            if is_valid:
                try:
                    ca_cert, _ = self._load_ca()
                    # Use PKCS1v15 padding for RSA signature verification
                    ca_cert.public_key().verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        hashes.SHA256(),
                    )
                except Exception as e:
                    is_valid = False
                    error_message = f"Signature verification failed: {e}"

            return CertificateInfo(
                common_name=common_name,
                node_id=common_name,  # Using CN as node_id
                expires_at=cert.not_valid_after,
                issuer=issuer,
                serial_number=cert.serial_number,
                is_valid=is_valid,
                not_valid_before=cert.not_valid_before,
                fingerprint=fingerprint,
                subject_alt_names=san_list,
                error_message=error_message,
            )

        except Exception as e:
            return CertificateInfo(
                common_name="",
                node_id="",
                expires_at=datetime.min,
                issuer="",
                serial_number=0,
                is_valid=False,
                error_message=f"Certificate parse error: {e}",
            )

    def get_ca_info(self) -> Optional[CAInfo]:
        """
        Get information about the CA certificate.

        Returns:
            CAInfo if CA exists, None otherwise
        """
        if not self.ca_cert_path.exists():
            return None

        try:
            with open(self.ca_cert_path, "rb") as f:
                ca_cert = x509.load_pem_x509_certificate(f.read(), default_backend())

            cn_attrs = ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
            common_name = cn_attrs[0].value if cn_attrs else ""

            org_attrs = ca_cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)
            organization = org_attrs[0].value if org_attrs else ""

            fingerprint = ca_cert.fingerprint(hashes.SHA256()).hex()

            validity_days = (ca_cert.not_valid_after - ca_cert.not_valid_before).days

            return CAInfo(
                common_name=common_name,
                organization=organization,
                validity_days=validity_days,
                key_size=self.ca_key_size,
                created_at=ca_cert.not_valid_before,
                expires_at=ca_cert.not_valid_after,
                fingerprint=fingerprint,
            )
        except Exception:
            return None

    def list_node_certificates(self) -> List[CertificateInfo]:
        """
        List all issued node certificates.

        Returns:
            List of CertificateInfo for each node certificate
        """
        certs = []
        for cert_file in self.nodes_dir.glob("*.crt"):
            try:
                with open(cert_file, "rb") as f:
                    info = self.verify_client_certificate(f.read())
                    if info:
                        certs.append(info)
            except Exception:
                pass
        return certs

    def revoke_certificate(self, node_id: str) -> bool:
        """
        Revoke a node certificate by deleting it.

        Note: This is a simple file-based revocation. For production,
        consider implementing a CRL (Certificate Revocation List).

        Args:
            node_id: Node whose certificate to revoke

        Returns:
            True if certificate was revoked, False if not found
        """
        cert_path = self.nodes_dir / f"{node_id}.crt"
        key_path = self.nodes_dir / f"{node_id}.key"

        revoked = False
        if cert_path.exists():
            cert_path.unlink()
            revoked = True
        if key_path.exists():
            key_path.unlink()
            revoked = True

        return revoked

    def to_dict(self) -> Dict[str, Any]:
        """Export PKI status as dictionary."""
        ca_info = self.get_ca_info()
        node_certs = self.list_node_certificates()

        return {
            "certs_dir": str(self.certs_dir),
            "ca_exists": self.ca_cert_path.exists(),
            "ca_info": ca_info.to_dict() if ca_info else None,
            "node_certificates": [cert.to_dict() for cert in node_certs],
            "total_nodes": len(node_certs),
            "valid_nodes": len([c for c in node_certs if c.is_valid]),
        }
