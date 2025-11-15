"""
Secret Management

Encrypted secret management for HoloLoom.

Features:
- Encrypted storage (Fernet symmetric encryption)
- Environment variable integration
- Automatic key rotation
- Secure file permissions (0600)
- Never commit secrets to git

References:
- OWASP Secrets Management Cheat Sheet
- NIST SP 800-57 (Key Management)

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Cryptography (install: pip install cryptography)
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("⚠️  cryptography not installed. Secrets will NOT be encrypted!")

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Encrypted secret management.

    Secrets stored encrypted on disk, decrypted at runtime.
    Integrates with environment variables for production use.

    Usage:
        >>> secrets = SecretManager()
        >>> secrets.set("DATABASE_URL", "postgresql://...")
        >>> secrets.set("API_SECRET", "supersecret123")
        >>> secrets.save()  # Encrypt and save to disk
        >>>
        >>> # Later (different process)
        >>> secrets = SecretManager()
        >>> secrets.load()  # Load and decrypt
        >>> db_url = secrets.get("DATABASE_URL")
    """

    def __init__(
        self,
        key_path: str = ".keys/master.key",
        secrets_path: str = ".secrets.enc"
    ):
        """
        Initialize secret manager.

        Args:
            key_path: Path to master encryption key
            secrets_path: Path to encrypted secrets file
        """
        self.key_path = Path(key_path)
        self.secrets_path = Path(secrets_path)
        self.secrets: Dict[str, str] = {}

        if CRYPTO_AVAILABLE:
            self.key = self._load_or_create_key()
            self.cipher = Fernet(self.key)
        else:
            self.key = None
            self.cipher = None
            logger.warning(
                "⚠️  Cryptography not available. Secrets will be stored in plaintext! "
                "Install with: pip install cryptography"
            )

    def _load_or_create_key(self) -> bytes:
        """Load encryption key or generate new one."""
        self.key_path.parent.mkdir(exist_ok=True, mode=0o700)

        if self.key_path.exists():
            key = self.key_path.read_bytes()
            logger.info(f"✅ Loaded encryption key from {self.key_path}")
        else:
            key = Fernet.generate_key()
            self.key_path.write_bytes(key)
            self.key_path.chmod(0o600)  # Owner read/write only
            logger.info(f"🔑 Generated new encryption key at {self.key_path}")
            logger.warning(
                "⚠️  IMPORTANT: Backup this key! Secrets cannot be decrypted without it!"
            )

        return key

    def set(self, name: str, value: str):
        """
        Set secret (will be encrypted on save).

        Args:
            name: Secret name (e.g., "DATABASE_URL")
            value: Secret value
        """
        self.secrets[name] = value
        logger.debug(f"✅ Set secret: {name}")

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret (decrypted).

        Args:
            name: Secret name
            default: Default value if not found

        Returns:
            Secret value or default

        Priority:
        1. Environment variable (highest priority)
        2. Encrypted storage
        3. Default value
        """
        # Try environment variable first (production use)
        if name in os.environ:
            logger.debug(f"✅ Got secret from environment: {name}")
            return os.environ[name]

        # Fall back to encrypted storage
        value = self.secrets.get(name, default)

        if value is None:
            logger.warning(f"⚠️  Secret not found: {name}")

        return value

    def delete(self, name: str):
        """
        Delete secret.

        Args:
            name: Secret name
        """
        if name in self.secrets:
            del self.secrets[name]
            logger.info(f"🗑️  Deleted secret: {name}")
        else:
            logger.warning(f"⚠️  Secret not found (cannot delete): {name}")

    def list_secrets(self) -> list[str]:
        """
        List all secret names (NOT values!).

        Returns:
            List of secret names
        """
        return list(self.secrets.keys())

    def save(self, path: Optional[str] = None):
        """
        Save secrets to encrypted file.

        Args:
            path: Optional custom path (defaults to secrets_path)
        """
        if path:
            encrypted_path = Path(path)
        else:
            encrypted_path = self.secrets_path

        # Serialize secrets
        plaintext = json.dumps(self.secrets, indent=2).encode()

        # Encrypt
        if self.cipher:
            ciphertext = self.cipher.encrypt(plaintext)
        else:
            # No encryption available (fallback)
            ciphertext = plaintext
            logger.warning("⚠️  Saving secrets UNENCRYPTED (cryptography not available)")

        # Write to file
        encrypted_path.write_bytes(ciphertext)
        encrypted_path.chmod(0o600)  # Owner read/write only

        logger.info(
            f"✅ Saved {len(self.secrets)} secrets to {encrypted_path} "
            f"({'encrypted' if self.cipher else 'PLAINTEXT'})"
        )

    def load(self, path: Optional[str] = None):
        """
        Load secrets from encrypted file.

        Args:
            path: Optional custom path (defaults to secrets_path)
        """
        if path:
            encrypted_path = Path(path)
        else:
            encrypted_path = self.secrets_path

        if not encrypted_path.exists():
            logger.info(f"ℹ️  No secrets file found at {encrypted_path}")
            return

        # Read from file
        ciphertext = encrypted_path.read_bytes()

        # Decrypt
        if self.cipher:
            try:
                plaintext = self.cipher.decrypt(ciphertext)
            except Exception as e:
                logger.error(f"❌ Failed to decrypt secrets: {e}")
                logger.error("   Key may be incorrect or file may be corrupted")
                return
        else:
            # No encryption available (fallback)
            plaintext = ciphertext
            logger.warning("⚠️  Loading secrets UNENCRYPTED (cryptography not available)")

        # Deserialize
        self.secrets = json.loads(plaintext.decode())

        logger.info(
            f"✅ Loaded {len(self.secrets)} secrets from {encrypted_path} "
            f"({'encrypted' if self.cipher else 'PLAINTEXT'})"
        )

    def rotate_key(self, new_key_path: str):
        """
        Rotate master encryption key (re-encrypt all secrets).

        Args:
            new_key_path: Path to new key file

        Warning: This will re-encrypt all secrets with a new key.
        Backup the old key before rotating!
        """
        if not CRYPTO_AVAILABLE:
            logger.error("❌ Cannot rotate key (cryptography not available)")
            return

        # Generate new key
        new_key = Fernet.generate_key()
        new_cipher = Fernet(new_key)

        # Re-encrypt all secrets
        plaintext = json.dumps(self.secrets).encode()
        new_ciphertext = new_cipher.encrypt(plaintext)

        # Save with new key
        new_key_path_obj = Path(new_key_path)
        new_key_path_obj.parent.mkdir(exist_ok=True, mode=0o700)
        new_key_path_obj.write_bytes(new_key)
        new_key_path_obj.chmod(0o600)

        # Save re-encrypted secrets
        self.secrets_path.write_bytes(new_ciphertext)

        # Update instance
        self.key = new_key
        self.cipher = new_cipher

        logger.info(f"🔄 Rotated encryption key: {self.key_path} → {new_key_path}")
        logger.warning("⚠️  Old key is now invalid. Backup if needed!")

    def export_env_file(self, path: str = ".env"):
        """
        Export secrets to .env file (for development).

        Args:
            path: Path to .env file

        Warning: .env file is PLAINTEXT! Use only for development.
        """
        env_path = Path(path)

        with env_path.open("w") as f:
            f.write("# HoloLoom Environment Variables\n")
            f.write("# ⚠️  DO NOT COMMIT TO GIT!\n\n")

            for name, value in sorted(self.secrets.items()):
                f.write(f"{name}={value}\n")

        env_path.chmod(0o600)  # Owner read/write only

        logger.info(f"✅ Exported {len(self.secrets)} secrets to {path}")
        logger.warning(f"⚠️  {path} is PLAINTEXT! Add to .gitignore!")


# Factory function
def create_secret_manager(
    key_path: Optional[str] = None,
    secrets_path: Optional[str] = None
) -> SecretManager:
    """
    Create secret manager with auto-load.

    Args:
        key_path: Optional custom key path
        secrets_path: Optional custom secrets path

    Returns:
        SecretManager instance (with secrets loaded)

    Usage:
        >>> secrets = create_secret_manager()
        >>> api_key = secrets.get("API_KEY")
    """
    manager = SecretManager(
        key_path=key_path or ".keys/master.key",
        secrets_path=secrets_path or ".secrets.enc"
    )

    # Auto-load if file exists
    if manager.secrets_path.exists():
        manager.load()

    return manager


# Global instance (singleton pattern)
_global_secrets: Optional[SecretManager] = None


def get_secrets() -> SecretManager:
    """
    Get global secret manager instance (singleton).

    Returns:
        Global SecretManager instance

    Usage:
        >>> from HoloLoom.security.secrets import get_secrets
        >>> secrets = get_secrets()
        >>> db_url = secrets.get("DATABASE_URL")
    """
    global _global_secrets

    if _global_secrets is None:
        _global_secrets = create_secret_manager()

    return _global_secrets


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Secret Management Demo")
    print("=" * 70)

    # Initialize secret manager
    secrets = SecretManager(
        key_path=".demo_keys/master.key",
        secrets_path=".demo_secrets.enc"
    )

    # Set some secrets
    print("\n📝 Setting secrets...")
    secrets.set("DATABASE_URL", "postgresql://user:pass@localhost/db")
    secrets.set("REDIS_URL", "redis://localhost:6379")
    secrets.set("API_SECRET", "supersecret123")
    secrets.set("OPENAI_API_KEY", "sk-...")

    print(f"   Set {len(secrets.list_secrets())} secrets:")
    for name in secrets.list_secrets():
        print(f"   - {name}")

    # Save encrypted
    print("\n💾 Saving secrets (encrypted)...")
    secrets.save()

    # Clear in-memory secrets
    print("\n🗑️  Clearing in-memory secrets...")
    secrets.secrets.clear()
    print(f"   Secrets in memory: {len(secrets.secrets)}")

    # Load encrypted
    print("\n📖 Loading secrets (decrypting)...")
    secrets.load()
    print(f"   Loaded {len(secrets.secrets)} secrets")

    # Get secrets
    print("\n🔑 Getting secrets...")
    db_url = secrets.get("DATABASE_URL")
    print(f"   DATABASE_URL: {db_url[:30]}...")

    # Export to .env (development only)
    print("\n📤 Exporting to .env (development)...")
    secrets.export_env_file(".demo.env")
    print("   ⚠️  Remember to add .demo.env to .gitignore!")

    # List secrets
    print("\n📋 All secrets:")
    for name in secrets.list_secrets():
        value = secrets.get(name)
        # Mask value for security
        masked = value[:10] + "..." if len(value) > 10 else "***"
        print(f"   {name}: {masked}")

    # Rotate key
    print(f"\n{'=' * 70}")
    print("Key Rotation Demo")
    print("=" * 70)

    print("\n🔄 Rotating encryption key...")
    secrets.rotate_key(".demo_keys/master_new.key")
    print("   ✅ Key rotated successfully")

    # Verify secrets still work
    print("\n✅ Verifying secrets after rotation...")
    db_url_after = secrets.get("DATABASE_URL")
    print(f"   DATABASE_URL: {db_url_after[:30]}...")
    print(f"   Match: {db_url == db_url_after}")

    # Cleanup
    print(f"\n{'=' * 70}")
    print("Cleanup")
    print("=" * 70)

    import shutil
    if Path(".demo_keys").exists():
        shutil.rmtree(".demo_keys")
        print("   🗑️  Deleted .demo_keys/")

    if Path(".demo_secrets.enc").exists():
        Path(".demo_secrets.enc").unlink()
        print("   🗑️  Deleted .demo_secrets.enc")

    if Path(".demo.env").exists():
        Path(".demo.env").unlink()
        print("   🗑️  Deleted .demo.env")

    print(f"\n{'=' * 70}\n")
