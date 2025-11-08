"""
GitHub Authentication Module

Handles GitHub App installation, OAuth flow, and token management
for the Promptly Matrix bot.

Features:
- GitHub App OAuth flow
- Token storage and refresh
- Installation verification
- Multi-repo support
"""

import os
import json
import time
import jwt
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta


class GitHubAuth:
    """
    GitHub authentication and token management.

    Uses GitHub App for authentication, which provides:
    - Fine-grained permissions
    - Installation tokens (per-repo)
    - Automatic token refresh
    """

    def __init__(self, config_path: str = "config/github_config.json"):
        """
        Initialize GitHub authentication.

        Args:
            config_path: Path to GitHub App configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Token cache: {installation_id: {token, expires_at}}
        self.token_cache: Dict[int, Dict[str, Any]] = {}

        # GitHub API base URL
        self.api_base = "https://api.github.com"

    def _load_config(self) -> Dict[str, Any]:
        """Load GitHub App configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"GitHub config not found: {self.config_path}\n"
                f"Please create config/github_config.json with your GitHub App credentials"
            )

        with open(self.config_path) as f:
            config = json.load(f)

        # Validate required fields
        required = ["app_id", "private_key", "webhook_secret"]
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

        return config

    def _generate_jwt(self) -> str:
        """
        Generate JWT for GitHub App authentication.

        Returns:
            JWT token for GitHub App
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds in the past (clock skew)
            "exp": now + (10 * 60),  # Expires in 10 minutes
            "iss": self.config["app_id"],
        }

        # Load private key
        private_key = self.config["private_key"]
        if private_key.startswith("-----BEGIN"):
            # Key is embedded in config
            key = private_key.encode()
        else:
            # Key is a file path
            with open(private_key, "rb") as f:
                key = f.read()

        # Generate JWT
        token = jwt.encode(payload, key, algorithm="RS256")
        return token

    def get_installation_token(self, installation_id: int) -> str:
        """
        Get installation access token for a specific GitHub App installation.

        Args:
            installation_id: GitHub App installation ID

        Returns:
            Installation access token
        """
        # Check cache
        if installation_id in self.token_cache:
            cached = self.token_cache[installation_id]
            expires_at = datetime.fromisoformat(cached["expires_at"])

            # Return cached token if still valid (with 5 min buffer)
            if expires_at > datetime.now() + timedelta(minutes=5):
                return cached["token"]

        # Generate new token
        jwt_token = self._generate_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        url = f"{self.api_base}/app/installations/{installation_id}/access_tokens"
        response = requests.post(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        token = data["token"]
        expires_at = data["expires_at"]

        # Cache token
        self.token_cache[installation_id] = {
            "token": token,
            "expires_at": expires_at,
        }

        return token

    def get_installations(self) -> list[Dict[str, Any]]:
        """
        Get all GitHub App installations.

        Returns:
            List of installation objects
        """
        jwt_token = self._generate_jwt()

        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        url = f"{self.api_base}/app/installations"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def get_installation_repos(self, installation_id: int) -> list[Dict[str, Any]]:
        """
        Get repositories accessible by installation.

        Args:
            installation_id: GitHub App installation ID

        Returns:
            List of repository objects
        """
        token = self.get_installation_token(installation_id)

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        url = f"{self.api_base}/installation/repositories"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()["repositories"]

    def find_installation_for_repo(self, owner: str, repo: str) -> Optional[int]:
        """
        Find installation ID for a specific repository.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            Installation ID or None if not found
        """
        installations = self.get_installations()

        for installation in installations:
            installation_id = installation["id"]
            repos = self.get_installation_repos(installation_id)

            for r in repos:
                if r["owner"]["login"] == owner and r["name"] == repo:
                    return installation_id

        return None

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GitHub webhook signature.

        Args:
            payload: Raw webhook payload
            signature: X-Hub-Signature-256 header value

        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib

        secret = self.config["webhook_secret"].encode()
        expected = "sha256=" + hmac.new(secret, payload, hashlib.sha256).hexdigest()

        return hmac.compare_digest(expected, signature)

    def get_authenticated_user(self, token: str) -> Dict[str, Any]:
        """
        Get authenticated user info.

        Args:
            token: GitHub personal access token or installation token

        Returns:
            User object
        """
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        url = f"{self.api_base}/user"
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()


def create_github_config_template():
    """Create template GitHub App configuration file."""
    template = {
        "app_id": "YOUR_GITHUB_APP_ID",
        "private_key": "path/to/private-key.pem",
        "webhook_secret": "YOUR_WEBHOOK_SECRET",
        "oauth_client_id": "YOUR_OAUTH_CLIENT_ID",
        "oauth_client_secret": "YOUR_OAUTH_CLIENT_SECRET",
    }

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "github_config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(template, f, indent=2)

        print(f"Created GitHub config template: {config_path}")
        print("\nPlease update with your GitHub App credentials:")
        print("1. Create a GitHub App: https://github.com/settings/apps/new")
        print("2. Generate a private key and save it")
        print("3. Update config/github_config.json with your credentials")
    else:
        print(f"GitHub config already exists: {config_path}")


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        create_github_config_template()
    else:
        # Test authentication
        try:
            auth = GitHubAuth()
            print("✓ GitHub authentication initialized")

            # List installations
            installations = auth.get_installations()
            print(f"\n✓ Found {len(installations)} installation(s)")

            for installation in installations:
                installation_id = installation["id"]
                account = installation["account"]["login"]
                print(f"  - Installation {installation_id} ({account})")

                # List repos
                repos = auth.get_installation_repos(installation_id)
                print(f"    {len(repos)} repository(ies)")
                for repo in repos[:3]:  # Show first 3
                    print(f"      - {repo['full_name']}")

        except FileNotFoundError as e:
            print(f"\n✗ {e}")
            print("\nRun: python bot/github_auth.py init")
        except Exception as e:
            print(f"\n✗ Error: {e}")
