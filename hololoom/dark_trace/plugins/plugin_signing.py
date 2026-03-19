"""
Dark Trace Plugin Signing

Cryptographic signing and verification for Dark Trace plugins using
RSA and ECDSA algorithms for secure plugin distribution.

Author: HoloLoom Team
Created: December 2025
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

# Try to import cryptography library
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.x509 import load_pem_x509_certificate
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SigningAlgorithm(Enum):
    """Supported signing algorithms."""
    RSA_SHA256 = "RSA-SHA256"
    RSA_SHA384 = "RSA-SHA384"
    RSA_SHA512 = "RSA-SHA512"
    ECDSA_SHA256 = "ECDSA-SHA256"
    ECDSA_SHA384 = "ECDSA-SHA384"


@dataclass
class KeyPair:
    """A public/private key pair for signing."""
    key_id: str
    algorithm: SigningAlgorithm
    private_key_pem: bytes | None = None  # None if only public key available
    public_key_pem: bytes = b""
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def has_private_key(self) -> bool:
        return self.private_key_pem is not None


@dataclass
class PluginManifest:
    """Manifest containing plugin information for signing."""
    name: str
    version: str
    author: str
    files: dict[str, str]  # filename -> SHA256 hash
    metadata_hash: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_bytes(self) -> bytes:
        """Serialize manifest to bytes for signing."""
        data = {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "files": self.files,
            "metadata_hash": self.metadata_hash,
            "created_at": self.created_at.isoformat(),
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> "PluginManifest":
        """Deserialize manifest from bytes."""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            name=obj["name"],
            version=obj["version"],
            author=obj["author"],
            files=obj["files"],
            metadata_hash=obj["metadata_hash"],
            created_at=datetime.fromisoformat(obj["created_at"]),
        )


@dataclass
class SignedManifest:
    """A signed plugin manifest."""
    manifest: PluginManifest
    signature: bytes
    algorithm: SigningAlgorithm
    key_id: str
    signed_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": {
                "name": self.manifest.name,
                "version": self.manifest.version,
                "author": self.manifest.author,
                "files": self.manifest.files,
                "metadata_hash": self.manifest.metadata_hash,
                "created_at": self.manifest.created_at.isoformat(),
            },
            "signature": base64.b64encode(self.signature).decode('ascii'),
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "signed_at": self.signed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SignedManifest":
        manifest_data = data["manifest"]
        manifest = PluginManifest(
            name=manifest_data["name"],
            version=manifest_data["version"],
            author=manifest_data["author"],
            files=manifest_data["files"],
            metadata_hash=manifest_data["metadata_hash"],
            created_at=datetime.fromisoformat(manifest_data["created_at"]),
        )
        return cls(
            manifest=manifest,
            signature=base64.b64decode(data["signature"]),
            algorithm=SigningAlgorithm(data["algorithm"]),
            key_id=data["key_id"],
            signed_at=datetime.fromisoformat(data["signed_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )


class PluginSigner:
    """
    Signs plugins for verification.

    Uses RSA or ECDSA for cryptographic signing of plugin manifests,
    ensuring plugin integrity and authenticity.
    """

    def __init__(
        self,
        key_pair: KeyPair | None = None,
        default_algorithm: SigningAlgorithm = SigningAlgorithm.RSA_SHA256,
    ):
        self.key_pair = key_pair
        self.default_algorithm = default_algorithm

        if not HAS_CRYPTOGRAPHY:
            raise ImportError(
                "cryptography library required for plugin signing. "
                "Install with: pip install cryptography"
            )

    def generate_key_pair(
        self,
        key_id: str,
        algorithm: SigningAlgorithm | None = None,
        key_size: int = 2048,
        expires_in_days: int | None = 365,
    ) -> KeyPair:
        """
        Generate a new key pair for signing.

        Args:
            key_id: Identifier for the key
            algorithm: Signing algorithm
            key_size: Key size in bits (RSA) or curve (ECDSA)
            expires_in_days: Days until key expires (None for no expiration)

        Returns:
            Generated KeyPair
        """
        algorithm = algorithm or self.default_algorithm

        if algorithm.value.startswith("RSA"):
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
        else:  # ECDSA
            curve = ec.SECP256R1() if "256" in algorithm.value else ec.SECP384R1()
            private_key = ec.generate_private_key(curve, default_backend())

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        key_pair = KeyPair(
            key_id=key_id,
            algorithm=algorithm,
            private_key_pem=private_pem,
            public_key_pem=public_pem,
            expires_at=expires_at,
        )

        self.key_pair = key_pair
        return key_pair

    def create_manifest(
        self,
        plugin_dir: str | Path,
        name: str,
        version: str,
        author: str,
    ) -> PluginManifest:
        """
        Create a manifest for a plugin directory.

        Args:
            plugin_dir: Path to plugin directory
            name: Plugin name
            version: Plugin version
            author: Plugin author

        Returns:
            PluginManifest with file hashes
        """
        plugin_path = Path(plugin_dir)
        files = {}

        # Hash all Python files in the plugin
        for py_file in plugin_path.rglob("*.py"):
            rel_path = py_file.relative_to(plugin_path)
            with open(py_file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            files[str(rel_path)] = file_hash

        # Also hash any JSON config files
        for json_file in plugin_path.rglob("*.json"):
            rel_path = json_file.relative_to(plugin_path)
            with open(json_file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            files[str(rel_path)] = file_hash

        # Create metadata hash
        metadata = {
            "name": name,
            "version": version,
            "author": author,
        }
        metadata_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()

        return PluginManifest(
            name=name,
            version=version,
            author=author,
            files=files,
            metadata_hash=metadata_hash,
        )

    def sign_manifest(
        self,
        manifest: PluginManifest,
        expires_in_days: int | None = 365,
    ) -> SignedManifest:
        """
        Sign a plugin manifest.

        Args:
            manifest: The manifest to sign
            expires_in_days: Days until signature expires

        Returns:
            SignedManifest with cryptographic signature
        """
        if not self.key_pair or not self.key_pair.has_private_key():
            raise ValueError("Private key required for signing")

        if self.key_pair.is_expired():
            raise ValueError("Key pair has expired")

        # Load private key
        private_key = serialization.load_pem_private_key(
            self.key_pair.private_key_pem,
            password=None,
            backend=default_backend()
        )

        # Get data to sign
        data = manifest.to_bytes()

        # Sign based on algorithm
        algorithm = self.key_pair.algorithm
        if algorithm.value.startswith("RSA"):
            hash_algo = self._get_hash_algorithm(algorithm)
            signature = private_key.sign(
                data,
                padding.PKCS1v15(),
                hash_algo
            )
        else:  # ECDSA
            hash_algo = self._get_hash_algorithm(algorithm)
            signature = private_key.sign(data, ec.ECDSA(hash_algo))

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        return SignedManifest(
            manifest=manifest,
            signature=signature,
            algorithm=algorithm,
            key_id=self.key_pair.key_id,
            expires_at=expires_at,
        )

    def sign_plugin(
        self,
        plugin_dir: str | Path,
        name: str,
        version: str,
        author: str,
        expires_in_days: int | None = 365,
    ) -> SignedManifest:
        """
        Create and sign a manifest for a plugin.

        Args:
            plugin_dir: Path to plugin directory
            name: Plugin name
            version: Plugin version
            author: Plugin author
            expires_in_days: Days until signature expires

        Returns:
            SignedManifest
        """
        manifest = self.create_manifest(plugin_dir, name, version, author)
        return self.sign_manifest(manifest, expires_in_days)

    def _get_hash_algorithm(self, algorithm: SigningAlgorithm):
        """Get the hash algorithm for a signing algorithm."""
        if "256" in algorithm.value:
            return hashes.SHA256()
        elif "384" in algorithm.value:
            return hashes.SHA384()
        else:
            return hashes.SHA512()


class PluginVerifier:
    """
    Verifies plugin signatures.

    Maintains a trust store of public keys and verifies plugin
    signatures against them.
    """

    def __init__(self):
        self.trusted_keys: dict[str, KeyPair] = {}
        self._verification_cache: dict[str, tuple[bool, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)

    def add_trusted_key(self, key_pair: KeyPair) -> None:
        """Add a public key to the trust store."""
        self.trusted_keys[key_pair.key_id] = key_pair

    def remove_trusted_key(self, key_id: str) -> None:
        """Remove a key from the trust store."""
        if key_id in self.trusted_keys:
            del self.trusted_keys[key_id]
        # Clear cache entries for this key
        self._verification_cache = {
            k: v for k, v in self._verification_cache.items()
            if not k.startswith(f"{key_id}:")
        }

    def load_trusted_key(self, key_id: str, public_key_path: str | Path) -> KeyPair:
        """Load a public key from file and add to trust store."""
        path = Path(public_key_path)
        with open(path, 'rb') as f:
            public_pem = f.read()

        # Detect algorithm from key
        try:
            key = serialization.load_pem_public_key(public_pem, default_backend())
            if isinstance(key, rsa.RSAPublicKey):
                algorithm = SigningAlgorithm.RSA_SHA256
            else:
                algorithm = SigningAlgorithm.ECDSA_SHA256
        except Exception:
            algorithm = SigningAlgorithm.RSA_SHA256

        key_pair = KeyPair(
            key_id=key_id,
            algorithm=algorithm,
            public_key_pem=public_pem,
        )
        self.add_trusted_key(key_pair)
        return key_pair

    def verify_signature(
        self,
        signed_manifest: SignedManifest,
        check_expiry: bool = True,
    ) -> tuple[bool, str]:
        """
        Verify a signed manifest.

        Args:
            signed_manifest: The signed manifest to verify
            check_expiry: Whether to check signature expiry

        Returns:
            Tuple of (is_valid, message)
        """
        if not HAS_CRYPTOGRAPHY:
            return False, "cryptography library not installed"

        # Check cache
        cache_key = f"{signed_manifest.key_id}:{signed_manifest.manifest.name}:{signed_manifest.manifest.version}"
        if cache_key in self._verification_cache:
            valid, cached_at = self._verification_cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return valid, "verified (cached)"

        # Check expiry
        if check_expiry and signed_manifest.is_expired():
            return False, "signature has expired"

        # Get trusted key
        key_id = signed_manifest.key_id
        if key_id not in self.trusted_keys:
            return False, f"unknown key: {key_id}"

        key_pair = self.trusted_keys[key_id]
        if key_pair.is_expired():
            return False, "trusted key has expired"

        # Load public key
        try:
            public_key = serialization.load_pem_public_key(
                key_pair.public_key_pem,
                default_backend()
            )
        except Exception as e:
            return False, f"failed to load public key: {e}"

        # Verify signature
        data = signed_manifest.manifest.to_bytes()
        algorithm = signed_manifest.algorithm

        try:
            if algorithm.value.startswith("RSA"):
                hash_algo = self._get_hash_algorithm(algorithm)
                public_key.verify(
                    signed_manifest.signature,
                    data,
                    padding.PKCS1v15(),
                    hash_algo
                )
            else:  # ECDSA
                hash_algo = self._get_hash_algorithm(algorithm)
                public_key.verify(
                    signed_manifest.signature,
                    data,
                    ec.ECDSA(hash_algo)
                )

            # Cache result
            self._verification_cache[cache_key] = (True, datetime.now())
            return True, "signature verified"

        except InvalidSignature:
            self._verification_cache[cache_key] = (False, datetime.now())
            return False, "invalid signature"
        except Exception as e:
            return False, f"verification error: {e}"

    def verify_plugin_files(
        self,
        plugin_dir: str | Path,
        signed_manifest: SignedManifest,
    ) -> tuple[bool, list[str]]:
        """
        Verify that plugin files match the manifest hashes.

        Args:
            plugin_dir: Path to plugin directory
            signed_manifest: The signed manifest

        Returns:
            Tuple of (all_valid, list of invalid files)
        """
        plugin_path = Path(plugin_dir)
        manifest = signed_manifest.manifest
        invalid_files = []

        for filename, expected_hash in manifest.files.items():
            file_path = plugin_path / filename
            if not file_path.exists():
                invalid_files.append(f"{filename} (missing)")
                continue

            with open(file_path, 'rb') as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

            if actual_hash != expected_hash:
                invalid_files.append(f"{filename} (hash mismatch)")

        return len(invalid_files) == 0, invalid_files

    def full_verify(
        self,
        plugin_dir: str | Path,
        signed_manifest: SignedManifest,
    ) -> tuple[bool, str]:
        """
        Perform full verification: signature + file hashes.

        Args:
            plugin_dir: Path to plugin directory
            signed_manifest: The signed manifest

        Returns:
            Tuple of (is_valid, message)
        """
        # Verify signature
        sig_valid, sig_msg = self.verify_signature(signed_manifest)
        if not sig_valid:
            return False, f"signature verification failed: {sig_msg}"

        # Verify files
        files_valid, invalid_files = self.verify_plugin_files(plugin_dir, signed_manifest)
        if not files_valid:
            return False, f"file verification failed: {', '.join(invalid_files)}"

        return True, "fully verified"

    def _get_hash_algorithm(self, algorithm: SigningAlgorithm):
        """Get the hash algorithm for a signing algorithm."""
        if "256" in algorithm.value:
            return hashes.SHA256()
        elif "384" in algorithm.value:
            return hashes.SHA384()
        else:
            return hashes.SHA512()


# Factory functions
def create_signer(
    key_pair: KeyPair | None = None,
    algorithm: SigningAlgorithm = SigningAlgorithm.RSA_SHA256,
) -> PluginSigner:
    """Create a plugin signer."""
    return PluginSigner(key_pair, algorithm)


def create_verifier() -> PluginVerifier:
    """Create a plugin verifier."""
    return PluginVerifier()


def generate_key_pair(
    key_id: str,
    algorithm: SigningAlgorithm = SigningAlgorithm.RSA_SHA256,
    key_size: int = 2048,
    expires_in_days: int | None = 365,
) -> KeyPair:
    """Generate a new key pair for signing."""
    signer = PluginSigner()
    return signer.generate_key_pair(key_id, algorithm, key_size, expires_in_days)
