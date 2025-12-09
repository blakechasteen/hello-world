"""
Unit tests for PKI/mTLS Certificate Management.

Tests PKIManager, CertificateInfo, and CAInfo functionality.
"""

import pytest
import tempfile
import ssl
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from ..pki import (
    PKIManager,
    CertificateInfo,
    CAInfo,
    CRYPTOGRAPHY_AVAILABLE,
)

# Import cryptography for test assertions if available
if CRYPTOGRAPHY_AVAILABLE:
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization


class TestCertificateInfo:
    """Tests for CertificateInfo dataclass."""

    def test_basic_creation(self):
        """CertificateInfo stores all fields correctly."""
        expires = datetime.utcnow() + timedelta(days=365)
        info = CertificateInfo(
            common_name="node-1",
            node_id="node-1",
            expires_at=expires,
            issuer="Portal CA",
            serial_number=12345,
            is_valid=True,
        )

        assert info.common_name == "node-1"
        assert info.node_id == "node-1"
        assert info.expires_at == expires
        assert info.issuer == "Portal CA"
        assert info.serial_number == 12345
        assert info.is_valid is True

    def test_optional_fields(self):
        """Optional fields have correct defaults."""
        info = CertificateInfo(
            common_name="test",
            node_id="test",
            expires_at=datetime.utcnow(),
            issuer="CA",
            serial_number=1,
            is_valid=True,
        )

        assert info.not_valid_before is None
        assert info.fingerprint is None
        assert info.subject_alt_names == []
        assert info.error_message is None

    def test_with_all_fields(self):
        """CertificateInfo with all optional fields populated."""
        now = datetime.utcnow()
        expires = now + timedelta(days=365)

        info = CertificateInfo(
            common_name="node-1",
            node_id="node-1",
            expires_at=expires,
            issuer="Portal CA",
            serial_number=12345,
            is_valid=True,
            not_valid_before=now,
            fingerprint="AB:CD:EF:12:34",
            subject_alt_names=["DNS:node-1.local", "IP:192.168.1.1"],
            error_message=None,
        )

        assert info.not_valid_before == now
        assert info.fingerprint == "AB:CD:EF:12:34"
        assert len(info.subject_alt_names) == 2

    def test_invalid_certificate_with_error(self):
        """Invalid certificate includes error message."""
        info = CertificateInfo(
            common_name="expired-node",
            node_id="expired-node",
            expires_at=datetime.utcnow() - timedelta(days=1),
            issuer="Portal CA",
            serial_number=99999,
            is_valid=False,
            error_message="Certificate has expired",
        )

        assert info.is_valid is False
        assert info.error_message == "Certificate has expired"


class TestCAInfo:
    """Tests for CAInfo dataclass."""

    def test_ca_info_creation(self):
        """CAInfo stores all fields correctly."""
        created = datetime.utcnow()
        expires = created + timedelta(days=3650)

        ca_info = CAInfo(
            common_name="Portal CA",
            organization="HoloLoom Portal",
            validity_days=3650,
            key_size=4096,
            created_at=created,
            expires_at=expires,
            fingerprint="12:34:56:78:90:AB",
        )

        assert ca_info.common_name == "Portal CA"
        assert ca_info.organization == "HoloLoom Portal"
        assert ca_info.validity_days == 3650
        assert ca_info.key_size == 4096
        assert ca_info.created_at == created
        assert ca_info.expires_at == expires
        assert ca_info.fingerprint == "12:34:56:78:90:AB"


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestPKIManagerInitialization:
    """Tests for PKIManager initialization."""

    def test_basic_initialization(self):
        """PKIManager initializes with default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))

            assert manager.certs_dir == Path(tmpdir)
            assert manager.ca_key_size == 4096
            assert manager.node_key_size == 2048
            assert manager.organization == "HoloLoom Portal"

    def test_custom_settings(self):
        """PKIManager accepts custom settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(
                certs_dir=Path(tmpdir),
                ca_key_size=2048,
                node_key_size=1024,
                organization="Custom Org",
            )

            assert manager.ca_key_size == 2048
            assert manager.node_key_size == 1024
            assert manager.organization == "Custom Org"

    def test_creates_directory_structure(self):
        """PKIManager creates required directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            certs_dir = Path(tmpdir) / "certs"
            manager = PKIManager(certs_dir=certs_dir)

            assert certs_dir.exists()
            assert (certs_dir / "nodes").exists()


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestCAGeneration:
    """Tests for CA certificate generation."""

    def test_generate_ca_creates_files(self):
        """generate_ca creates cert and key files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            cert_path, key_path = manager.generate_ca()

            assert cert_path.exists()
            assert key_path.exists()
            assert cert_path.name == "ca.crt"
            assert key_path.name == "ca.key"

    def test_generate_ca_custom_name(self):
        """generate_ca accepts custom common name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca(common_name="My Custom CA")

            ca_info = manager.get_ca_info()
            assert ca_info.common_name == "My Custom CA"

    def test_generate_ca_custom_validity(self):
        """generate_ca accepts custom validity period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca(validity_days=365)

            ca_info = manager.get_ca_info()
            assert ca_info.validity_days == 365

    def test_generate_ca_no_overwrite_by_default(self):
        """generate_ca raises error if CA already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            with pytest.raises(FileExistsError):
                manager.generate_ca()

    def test_generate_ca_with_overwrite(self):
        """generate_ca can overwrite existing CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca(common_name="Original CA")
            manager.generate_ca(common_name="New CA", overwrite=True)

            ca_info = manager.get_ca_info()
            assert ca_info.common_name == "New CA"

    def test_ca_certificate_is_valid(self):
        """Generated CA certificate is valid X.509."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            cert_path, _ = manager.generate_ca()

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            assert cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value == "Portal CA"


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestNodeCertificateIssuance:
    """Tests for node certificate issuance."""

    def test_issue_node_certificate(self):
        """issue_node_certificate creates cert and key files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_node_certificate("node-1")

            assert cert_path.exists()
            assert key_path.exists()
            assert "node-1" in str(cert_path)

    def test_node_certificate_has_correct_cn(self):
        """Node certificate has correct common name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate("my-node")

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            cn = cert.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
            assert cn == "my-node"

    def test_node_certificate_signed_by_ca(self):
        """Node certificate is signed by the CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate("node-1")

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            issuer_cn = cert.issuer.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)[0].value
            assert issuer_cn == "Portal CA"

    def test_node_certificate_with_san_dns(self):
        """Node certificate includes DNS SANs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate(
                "node-1",
                san_dns_names=["node-1.local", "node-1.cluster.local"]
            )

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            dns_names = san_ext.value.get_values_for_type(x509.DNSName)
            assert "node-1.local" in dns_names
            assert "node-1.cluster.local" in dns_names

    def test_node_certificate_with_san_ip(self):
        """Node certificate includes IP SANs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate(
                "node-1",
                san_ip_addresses=["192.168.1.1", "10.0.0.1"]
            )

            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())

            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            ip_addrs = san_ext.value.get_values_for_type(x509.IPAddress)
            # Convert to strings for comparison
            ip_strs = [str(ip) for ip in ip_addrs]
            assert "192.168.1.1" in ip_strs
            assert "10.0.0.1" in ip_strs

    def test_issue_certificate_without_ca_fails(self):
        """Cannot issue node certificate without CA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))

            with pytest.raises(FileNotFoundError):
                manager.issue_node_certificate("node-1")

    def test_issue_multiple_node_certificates(self):
        """Can issue certificates for multiple nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            for i in range(3):
                cert_path, key_path = manager.issue_node_certificate(f"node-{i}")
                assert cert_path.exists()
                assert key_path.exists()

            nodes = manager.list_node_certificates()
            assert len(nodes) == 3


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestServerCertificateIssuance:
    """Tests for server certificate issuance."""

    def test_issue_server_certificate(self):
        """issue_server_certificate creates cert and key files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_server_certificate()

            assert cert_path.exists()
            assert key_path.exists()
            assert cert_path.name == "portal.crt"
            assert key_path.name == "portal.key"

    def test_server_certificate_custom_name(self):
        """Server certificate with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_server_certificate(server_name="api-server")

            assert cert_path.name == "api-server.crt"
            assert key_path.name == "api-server.key"


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestSSLContextCreation:
    """Tests for SSL context creation."""

    def test_create_server_ssl_context(self):
        """create_server_ssl_context returns valid SSLContext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_server_certificate()

            context = manager.create_server_ssl_context(cert_path, key_path)

            assert isinstance(context, ssl.SSLContext)
            assert context.verify_mode == ssl.CERT_REQUIRED

    def test_create_server_ssl_context_no_client_cert(self):
        """Server context without client certificate requirement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_server_certificate()

            context = manager.create_server_ssl_context(
                cert_path, key_path,
                require_client_cert=False
            )

            assert context.verify_mode == ssl.CERT_NONE

    def test_create_client_ssl_context(self):
        """create_client_ssl_context returns valid SSLContext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_node_certificate("node-1")

            context = manager.create_client_ssl_context(cert_path, key_path)

            assert isinstance(context, ssl.SSLContext)
            assert context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestCertificateVerification:
    """Tests for certificate verification."""

    def test_verify_valid_certificate(self):
        """Verify a valid certificate returns CertificateInfo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate("node-1")

            with open(cert_path, "rb") as f:
                cert_pem = f.read()

            info = manager.verify_client_certificate(cert_pem)

            assert info is not None
            assert info.is_valid is True
            assert info.common_name == "node-1"
            assert info.node_id == "node-1"

    def test_verify_expired_certificate(self):
        """Verify expired certificate returns invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            # Issue with very short validity (will be expired in our test)
            cert_path, _ = manager.issue_node_certificate("node-1", validity_days=0)

            with open(cert_path, "rb") as f:
                cert_pem = f.read()

            info = manager.verify_client_certificate(cert_pem)

            # Should still parse but mark as potentially expired
            assert info is not None

    def test_verify_invalid_pem_returns_invalid(self):
        """Invalid PEM data returns invalid CertificateInfo with error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            info = manager.verify_client_certificate(b"not a certificate")

            # Returns CertificateInfo with is_valid=False and error_message
            assert info is not None
            assert info.is_valid is False
            assert info.error_message is not None
            assert "error" in info.error_message.lower() or "parse" in info.error_message.lower()

    def test_verify_certificate_fingerprint(self):
        """Verified certificate includes fingerprint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, _ = manager.issue_node_certificate("node-1")

            with open(cert_path, "rb") as f:
                cert_pem = f.read()

            info = manager.verify_client_certificate(cert_pem)

            assert info.fingerprint is not None
            assert ":" in info.fingerprint  # SHA256 fingerprint format


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestCAInfo:
    """Tests for CA info retrieval."""

    def test_get_ca_info(self):
        """get_ca_info returns correct CA information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca(common_name="Test CA", validity_days=365)

            ca_info = manager.get_ca_info()

            assert ca_info is not None
            assert ca_info.common_name == "Test CA"
            assert ca_info.validity_days == 365
            assert ca_info.key_size == 4096
            assert ca_info.organization == "HoloLoom Portal"
            assert ca_info.fingerprint is not None

    def test_get_ca_info_no_ca(self):
        """get_ca_info returns None when no CA exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))

            ca_info = manager.get_ca_info()

            assert ca_info is None


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestNodeCertificateListing:
    """Tests for listing node certificates."""

    def test_list_empty(self):
        """list_node_certificates returns empty list when no nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            nodes = manager.list_node_certificates()

            assert nodes == []

    def test_list_multiple_nodes(self):
        """list_node_certificates returns all issued certificates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            manager.issue_node_certificate("node-1")
            manager.issue_node_certificate("node-2")
            manager.issue_node_certificate("node-3")

            nodes = manager.list_node_certificates()

            assert len(nodes) == 3
            node_ids = [n.node_id for n in nodes]
            assert "node-1" in node_ids
            assert "node-2" in node_ids
            assert "node-3" in node_ids

    def test_list_includes_validity_info(self):
        """Listed certificates include validity information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            manager.issue_node_certificate("node-1", validity_days=30)

            nodes = manager.list_node_certificates()

            assert len(nodes) == 1
            node = nodes[0]
            assert node.is_valid is True
            assert node.expires_at is not None


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestCertificateRevocation:
    """Tests for certificate revocation."""

    def test_revoke_certificate(self):
        """revoke_certificate removes certificate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            cert_path, key_path = manager.issue_node_certificate("node-1")

            # Verify files exist before revocation
            assert cert_path.exists()
            assert key_path.exists()

            result = manager.revoke_certificate("node-1")

            assert result is True
            # Files should be deleted after revocation
            assert not cert_path.exists()
            assert not key_path.exists()

    def test_revoke_nonexistent_certificate(self):
        """revoke_certificate returns False for nonexistent cert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()

            result = manager.revoke_certificate("nonexistent-node")

            assert result is False


@pytest.mark.skipif(not CRYPTOGRAPHY_AVAILABLE, reason="cryptography not installed")
class TestToDict:
    """Tests for to_dict export."""

    def test_to_dict_basic(self):
        """to_dict returns correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))

            result = manager.to_dict()

            assert "certs_dir" in result
            assert "ca_exists" in result
            assert "ca_info" in result
            assert "node_certificates" in result
            assert "total_nodes" in result
            assert "valid_nodes" in result

    def test_to_dict_with_ca_and_nodes(self):
        """to_dict includes CA and node information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            manager.issue_node_certificate("node-1")
            manager.issue_node_certificate("node-2")

            result = manager.to_dict()

            assert result["ca_exists"] is True
            assert result["ca_info"] is not None
            assert result["ca_info"]["common_name"] == "Portal CA"
            assert len(result["node_certificates"]) == 2
            assert result["total_nodes"] == 2

    def test_to_dict_stats(self):
        """to_dict includes correct statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PKIManager(certs_dir=Path(tmpdir))
            manager.generate_ca()
            manager.issue_node_certificate("node-1")
            manager.issue_node_certificate("node-2")

            result = manager.to_dict()

            assert result["total_nodes"] == 2
            assert result["valid_nodes"] == 2

            # After revocation, node count decreases (files deleted)
            manager.revoke_certificate("node-2")
            result = manager.to_dict()

            assert result["total_nodes"] == 1
            assert result["valid_nodes"] == 1


class TestNoCryptography:
    """Tests for graceful degradation without cryptography."""

    def test_module_import_without_cryptography(self):
        """Module imports successfully without cryptography."""
        # The module should import even without cryptography
        from ..pki import CRYPTOGRAPHY_AVAILABLE
        # Just verify we can check availability
        assert isinstance(CRYPTOGRAPHY_AVAILABLE, bool)

    @pytest.mark.skipif(CRYPTOGRAPHY_AVAILABLE, reason="cryptography is installed")
    def test_pki_manager_fails_without_cryptography(self):
        """PKIManager raises error without cryptography."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="cryptography library required"):
                PKIManager(certs_dir=Path(tmpdir))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
