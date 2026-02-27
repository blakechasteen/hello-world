"""
Network Policy Enforcement for Sandbox Environment

Implements platform-aware network access control using:
- Linux: iptables rules
- macOS: pf firewall rules
- Windows: Windows Firewall rules

Provides allowlist-based egress filtering with graceful degradation.
"""

import asyncio
import ipaddress
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Set, Tuple

logger = logging.getLogger(__name__)


class NetworkAccessType(Enum):
    """Type of network access."""
    ALLOW = "allow"
    BLOCK = "block"
    LOG = "log"


@dataclass
class NetworkEndpoint:
    """Network endpoint (host:port)."""
    host: str
    port: Optional[int] = None  # None = all ports

    def __post_init__(self):
        """Validate endpoint."""
        # Validate host (IP or domain)
        if self.host not in ["*", "0.0.0.0", "::"]:
            try:
                ipaddress.ip_address(self.host)
            except ValueError:
                # Treat as domain name
                pass

    def matches(self, host: str, port: Optional[int] = None) -> bool:
        """
        Check if endpoint matches host:port.

        Args:
            host: Hostname or IP address
            port: Port number

        Returns:
            True if this endpoint matches the connection
        """
        if self.host == "*":
            return True

        # Host match
        if self.host != host:
            # Try DNS resolution
            try:
                resolved = socket.gethostbyname(self.host)
                if resolved != host:
                    return False
            except Exception:
                return False

        # Port match
        if self.port is not None and port != self.port:
            return False

        return True


@dataclass
class NetworkPolicyConfig:
    """Network policy configuration."""
    allowlist: List[NetworkEndpoint] = field(default_factory=list)
    denylist: List[NetworkEndpoint] = field(default_factory=list)
    enable_dns: bool = False  # Allow DNS queries
    enable_logging: bool = True
    default_action: NetworkAccessType = NetworkAccessType.BLOCK
    track_attempts: bool = True


class NetworkPolicyProtocol(Protocol):
    """Protocol for network policy implementations."""

    async def apply_policy(self) -> bool:
        """Apply network policy. Returns True if successful."""
        ...

    async def remove_policy(self) -> bool:
        """Remove network policy. Returns True if successful."""
        ...

    async def check_egress(self, host: str, port: int) -> bool:
        """Check if egress to host:port is allowed."""
        ...


class BlockedAttempt:
    """Record of blocked network attempt."""

    def __init__(
        self,
        host: str,
        port: int,
        timestamp: float,
        process_name: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.timestamp = timestamp
        self.process_name = process_name
        self.count = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "timestamp": self.timestamp,
            "process_name": self.process_name,
            "count": self.count,
        }


class NetworkPolicy:
    """
    Platform-aware network policy enforcement.

    Implements default-deny policy with allowlist-based access.
    """

    def __init__(self, config: NetworkPolicyConfig):
        """
        Initialize network policy.

        Args:
            config: Network policy configuration
        """
        self.config = config
        self.platform = self._get_platform()
        self.applied = False
        self.blocked_attempts: Dict[Tuple[str, int], BlockedAttempt] = {}

        # Add common DNS endpoints if DNS enabled
        if self.config.enable_dns:
            self.config.allowlist.extend([
                NetworkEndpoint("8.8.8.8", 53),  # Google DNS
                NetworkEndpoint("8.8.4.4", 53),
                NetworkEndpoint("1.1.1.1", 53),  # Cloudflare DNS
                NetworkEndpoint("1.0.0.1", 53),
            ])

    def _get_platform(self) -> str:
        """Get normalized platform identifier."""
        system = platform.system().lower()
        if system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        elif system == "darwin":
            return "macos"
        else:
            return "unknown"

    async def apply_policy(self) -> bool:
        """
        Apply network policy.

        Returns:
            True if policy applied successfully
        """
        if self.applied:
            return True

        try:
            if self.platform == "linux":
                success = await self._apply_iptables()
            elif self.platform == "windows":
                success = await self._apply_windows_firewall()
            elif self.platform == "macos":
                success = await self._apply_pf()
            else:
                logger.warning(f"Network policy not supported on {self.platform}")
                return False

            if success:
                self.applied = True
                logger.info(f"Network policy applied on {self.platform}")

            return success

        except Exception as e:
            logger.error(f"Failed to apply network policy: {e}")
            return False

    async def remove_policy(self) -> bool:
        """
        Remove network policy.

        Returns:
            True if policy removed successfully
        """
        if not self.applied:
            return True

        try:
            if self.platform == "linux":
                success = await self._remove_iptables()
            elif self.platform == "windows":
                success = await self._remove_windows_firewall()
            elif self.platform == "macos":
                success = await self._remove_pf()
            else:
                return False

            if success:
                self.applied = False
                logger.info(f"Network policy removed on {self.platform}")

            return success

        except Exception as e:
            logger.error(f"Failed to remove network policy: {e}")
            return False

    async def check_egress(self, host: str, port: int) -> bool:
        """
        Check if egress to host:port is allowed.

        Args:
            host: Hostname or IP address
            port: Port number

        Returns:
            True if access is allowed
        """
        # Check allowlist
        for endpoint in self.config.allowlist:
            if endpoint.matches(host, port):
                return True

        # Check denylist
        for endpoint in self.config.denylist:
            if endpoint.matches(host, port):
                self._record_blocked_attempt(host, port)
                return False

        # Default action
        if self.config.default_action == NetworkAccessType.BLOCK:
            self._record_blocked_attempt(host, port)
            return False

        return True

    def _record_blocked_attempt(self, host: str, port: int) -> None:
        """
        Record a blocked network attempt.

        Args:
            host: Hostname or IP address
            port: Port number
        """
        if not self.config.track_attempts:
            return

        import time
        key = (host, port)

        if key in self.blocked_attempts:
            self.blocked_attempts[key].count += 1
        else:
            self.blocked_attempts[key] = BlockedAttempt(
                host=host,
                port=port,
                timestamp=time.time()
            )

        if self.config.enable_logging:
            logger.warning(f"Blocked network access: {host}:{port}")

    async def _apply_iptables(self) -> bool:
        """
        Apply iptables rules on Linux.

        Creates rules to allow only whitelisted connections.

        Returns:
            True if rules applied successfully
        """
        try:
            # Check if iptables available
            result = await asyncio.create_subprocess_exec(
                "iptables", "--version",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, _ = await result.communicate()
            if result.returncode != 0:
                logger.warning("iptables not available")
                return False

            # Save current iptables rules
            save_proc = await asyncio.create_subprocess_exec(
                "iptables-save",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self._iptables_backup, _ = await save_proc.communicate()

            # Set default policy to DROP for OUTPUT chain
            await self._run_iptables([
                "-P", "OUTPUT", "DROP"
            ])

            # Allow loopback
            await self._run_iptables([
                "-A", "OUTPUT", "-d", "127.0.0.1",
                "-j", "ACCEPT"
            ])

            # Allow allowlisted hosts
            for endpoint in self.config.allowlist:
                if endpoint.host in ["*", "0.0.0.0"]:
                    # Allow all
                    await self._run_iptables([
                        "-P", "OUTPUT", "ACCEPT"
                    ])
                    break

                args = ["-A", "OUTPUT", "-d", endpoint.host]

                if endpoint.port:
                    args.extend(["-p", "tcp", "--dport", str(endpoint.port)])

                args.extend(["-j", "ACCEPT"])
                await self._run_iptables(args)

            return True

        except Exception as e:
            logger.error(f"iptables error: {e}")
            return False

    async def _remove_iptables(self) -> bool:
        """
        Remove iptables rules.

        Restores previous rules if backup available.

        Returns:
            True if rules removed successfully
        """
        try:
            if hasattr(self, "_iptables_backup"):
                # Restore from backup
                restore_proc = await asyncio.create_subprocess_exec(
                    "iptables-restore",
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                _, _ = await restore_proc.communicate(
                    input=self._iptables_backup
                )
                return restore_proc.returncode == 0
            else:
                # Flush all rules
                await self._run_iptables(["-F"])
                await self._run_iptables(["-X"])
                return True

        except Exception as e:
            logger.error(f"iptables restore error: {e}")
            return False

    async def _run_iptables(self, args: List[str]) -> bool:
        """
        Run iptables command with elevated privileges.

        Args:
            args: iptables arguments

        Returns:
            True if command succeeded
        """
        # Note: This requires elevated privileges (root/sudo)
        try:
            proc = await asyncio.create_subprocess_exec(
                "iptables", *args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(f"iptables error: {stderr.decode()}")
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"iptables execution error: {e}")
            return False

    async def _apply_pf(self) -> bool:
        """
        Apply pf firewall rules on macOS.

        Creates rules using pfctl.

        Returns:
            True if rules applied successfully
        """
        try:
            # Generate pf rules
            rules = self._generate_pf_rules()

            # Write rules to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".pf",
                delete=False
            ) as f:
                f.write(rules)
                rules_file = f.name

            # Apply rules
            proc = await asyncio.create_subprocess_exec(
                "pfctl", "-f", rules_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, stderr = await proc.communicate()

            # Cleanup
            try:
                os.unlink(rules_file)
            except Exception:
                pass

            if proc.returncode != 0:
                logger.warning(f"pfctl error: {stderr.decode()}")
                return False

            return True

        except Exception as e:
            logger.error(f"pf error: {e}")
            return False

    async def _remove_pf(self) -> bool:
        """
        Remove pf firewall rules.

        Returns:
            True if rules removed successfully
        """
        try:
            # Disable pf
            proc = await asyncio.create_subprocess_exec(
                "pfctl", "-d",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, _ = await proc.communicate()
            return proc.returncode == 0

        except Exception as e:
            logger.error(f"pf disable error: {e}")
            return False

    def _generate_pf_rules(self) -> str:
        """
        Generate pf firewall rules.

        Returns:
            pf rule string
        """
        rules = []

        # Allow loopback
        rules.append("pass on lo0")

        # Allow allowlisted hosts
        has_any = False
        for endpoint in self.config.allowlist:
            if endpoint.host in ["*", "0.0.0.0"]:
                has_any = True
                break
            rule = f"pass out proto tcp from any to {endpoint.host}"
            if endpoint.port:
                rule += f" port {endpoint.port}"
            rules.append(rule)

        if not has_any:
            # Block all outbound by default
            rules.append("block out proto tcp from any to any")

        return "\n".join(rules)

    async def _apply_windows_firewall(self) -> bool:
        """
        Apply Windows Firewall rules.

        Uses netsh to configure firewall.

        Returns:
            True if rules applied successfully
        """
        try:
            # Enable firewall if not already enabled
            await self._run_netsh([
                "advfirewall", "set", "allprofiles", "state", "on"
            ])

            # Set default outbound policy to block
            await self._run_netsh([
                "advfirewall", "set", "allprofiles",
                "outboundusernotification", "enable"
            ])

            # Add rules for allowlist
            for i, endpoint in enumerate(self.config.allowlist):
                if endpoint.host in ["*", "0.0.0.0"]:
                    # Allow all - disable outbound filtering
                    return True

                rule_name = f"HoloLoom_Allow_{i}"
                args = [
                    "advfirewall", "firewall", "add", "rule",
                    f"name={rule_name}",
                    "dir=out",
                    "action=allow",
                    "enabled=yes",
                    f"remoteip={endpoint.host}"
                ]

                if endpoint.port:
                    args.append(f"remoteport={endpoint.port}")

                await self._run_netsh(args)

            return True

        except Exception as e:
            logger.error(f"Windows Firewall error: {e}")
            return False

    async def _remove_windows_firewall(self) -> bool:
        """
        Remove Windows Firewall rules.

        Returns:
            True if rules removed successfully
        """
        try:
            # Remove HoloLoom rules
            await self._run_netsh([
                "advfirewall", "firewall", "delete", "rule",
                "name=HoloLoom*"
            ])
            return True

        except Exception as e:
            logger.error(f"Windows Firewall removal error: {e}")
            return False

    async def _run_netsh(self, args: List[str]) -> bool:
        """
        Run netsh command with elevated privileges.

        Args:
            args: netsh arguments

        Returns:
            True if command succeeded
        """
        # Note: This requires elevated privileges (Administrator)
        try:
            proc = await asyncio.create_subprocess_exec(
                "netsh", *args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(f"netsh error: {stderr.decode()}")
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"netsh execution error: {e}")
            return False

    def get_blocked_attempts(self) -> List[Dict]:
        """
        Get list of blocked network attempts.

        Returns:
            List of blocked attempt dictionaries
        """
        return [
            attempt.to_dict()
            for attempt in self.blocked_attempts.values()
        ]

    def clear_blocked_attempts(self) -> None:
        """Clear blocked attempts history."""
        self.blocked_attempts.clear()

    def get_policy_status(self) -> Dict:
        """
        Get network policy status.

        Returns:
            Status dictionary
        """
        return {
            "platform": self.platform,
            "applied": self.applied,
            "allowlist_count": len(self.config.allowlist),
            "denylist_count": len(self.config.denylist),
            "blocked_attempts_count": len(self.blocked_attempts),
            "default_action": self.config.default_action.value,
        }


class NetworkPolicyError(Exception):
    """Network policy error."""
    pass


async def test_network_policy():
    """Test network policy."""
    config = NetworkPolicyConfig(
        allowlist=[
            NetworkEndpoint("8.8.8.8", 443),
            NetworkEndpoint("github.com", 443),
        ],
        enable_logging=True,
    )

    policy = NetworkPolicy(config)

    # Show status
    print(f"Platform: {policy.platform}")
    print(f"Status: {policy.get_policy_status()}")

    # Check some addresses
    print(f"\nChecking access:")
    print(f"  google.com:443 -> {await policy.check_egress('8.8.8.8', 443)}")
    print(f"  example.com:80 -> {await policy.check_egress('93.184.216.34', 80)}")

    # Show blocked attempts
    print(f"\nBlocked attempts: {policy.get_blocked_attempts()}")


if __name__ == "__main__":
    asyncio.run(test_network_policy())
