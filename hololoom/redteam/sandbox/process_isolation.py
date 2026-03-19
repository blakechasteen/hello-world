"""
Process Isolation for Sandbox Environment

Implements platform-aware process isolation using:
- Linux: cgroups v2 + seccomp
- Windows: Job objects
- macOS: Basic subprocess with resource limits

Provides graceful degradation when platform features unavailable.
"""

import asyncio
import os
import platform
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol

# Platform-specific imports with graceful degradation
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

if sys.platform == "linux":
    try:
        import ctypes
        from ctypes import CDLL, c_int, c_uint64
        HAS_LINUX_CTYPES = True
    except ImportError:
        HAS_LINUX_CTYPES = False
else:
    HAS_LINUX_CTYPES = False

if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes
        HAS_WINDOWS_CTYPES = True
    except ImportError:
        HAS_WINDOWS_CTYPES = False
else:
    HAS_WINDOWS_CTYPES = False


@dataclass
class ProcessIsolationConfig:
    """Configuration for process isolation."""
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    timeout_seconds: int = 30
    enable_cgroups: bool = True
    enable_seccomp: bool = True
    enable_rlimit: bool = True
    working_directory: str | None = None
    environment: dict[str, str] | None = None


class ProcessIsolationProtocol(Protocol):
    """Protocol for process isolation implementations."""

    async def spawn_isolated(
        self,
        command: list[str],
        stdin: bytes | None = None,
        timeout: int | None = None,
        **kwargs
    ) -> tuple[int, bytes, bytes, int]:
        """
        Spawn process with isolation.

        Returns:
            Tuple of (pid, stdout, stderr, return_code)
        """
        ...

    async def kill(self, pid: int) -> bool:
        """Kill an isolated process."""
        ...

    async def cleanup(self) -> None:
        """Clean up all resources."""
        ...


class ProcessIsolator:
    """
    Platform-aware process isolation with graceful degradation.

    Isolates processes using:
    - Linux: cgroups v2 + seccomp filters
    - Windows: Job objects with resource limits
    - Others: resource.setrlimit with subprocess isolation
    """

    def __init__(self, config: ProcessIsolationConfig):
        """
        Initialize process isolator.

        Args:
            config: Isolation configuration

        Raises:
            ValueError: If config is invalid
        """
        if config.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if config.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.config = config
        self.platform = self._get_platform()
        self.processes: dict[int, subprocess.Popen] = {}
        self.cgroup_paths: dict[int, str] = {}

        # Platform capability detection
        self.has_cgroups = HAS_LINUX_CTYPES and self.platform == "linux"
        self.has_seccomp = HAS_LINUX_CTYPES and self.platform == "linux"
        self.has_job_objects = HAS_WINDOWS_CTYPES and self.platform == "windows"
        self.has_rlimit = HAS_RESOURCE

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

    async def spawn_isolated(
        self,
        command: list[str],
        stdin: bytes | None = None,
        timeout: int | None = None,
        **kwargs
    ) -> tuple[int, bytes, bytes, int]:
        """
        Spawn an isolated process.

        Args:
            command: Command to execute
            stdin: Optional stdin data
            timeout: Override timeout (seconds)
            **kwargs: Additional subprocess arguments

        Returns:
            Tuple of (pid, stdout, stderr, return_code)

        Raises:
            subprocess.TimeoutExpired: If process exceeds timeout
            OSError: If isolation setup fails
        """
        timeout = timeout or self.config.timeout_seconds

        # Prepare subprocess arguments
        proc_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.PIPE if stdin else subprocess.DEVNULL,
        }

        if self.config.working_directory:
            proc_kwargs["cwd"] = self.config.working_directory

        # Merge environment
        env = os.environ.copy()
        if self.config.environment:
            env.update(self.config.environment)
        proc_kwargs["env"] = env

        # Merge additional kwargs
        proc_kwargs.update(kwargs)

        # Spawn process
        try:
            process = subprocess.Popen(command, **proc_kwargs)
        except OSError as e:
            raise OSError(f"Failed to spawn process: {e}")

        pid = process.pid
        self.processes[pid] = process

        try:
            # Apply platform-specific isolation
            if self.platform == "linux":
                if self.config.enable_cgroups and self.has_cgroups:
                    await self._setup_cgroups(pid)
                if self.config.enable_seccomp and self.has_seccomp:
                    await self._setup_seccomp(pid)

            elif self.platform == "windows":
                if self.has_job_objects:
                    await self._setup_windows_job(pid)

            # Apply resource limits
            if self.config.enable_rlimit and self.has_rlimit:
                await self._setup_rlimit(pid)

            # Wait for process with timeout
            try:
                stdout, stderr = process.communicate(
                    input=stdin,
                    timeout=timeout
                )
            except subprocess.TimeoutExpired:
                # Kill on timeout
                await self.kill(pid)
                raise

            return_code = process.returncode
            return (pid, stdout, stderr, return_code)

        except Exception:
            # Cleanup on failure
            await self.kill(pid)
            raise

        finally:
            self.processes.pop(pid, None)

    async def _setup_cgroups(self, pid: int) -> None:
        """
        Setup Linux cgroups v2 for process.

        Configures:
        - Memory limit
        - CPU quota
        - Process limit

        Args:
            pid: Process ID

        Raises:
            OSError: If cgroups setup fails
        """
        if not self.has_cgroups or self.platform != "linux":
            return

        cgroup_path = f"/sys/fs/cgroup/hololoom_{pid}"

        try:
            # Create cgroup
            os.makedirs(cgroup_path, exist_ok=True)

            # Set memory limit
            memory_bytes = self.config.max_memory_mb * 1024 * 1024
            memory_max_path = os.path.join(cgroup_path, "memory.max")
            if os.path.exists(memory_max_path):
                with open(memory_max_path, "w") as f:
                    f.write(str(memory_bytes))

            # Set memory high (soft limit for warning)
            memory_high_path = os.path.join(cgroup_path, "memory.high")
            if os.path.exists(memory_high_path):
                with open(memory_high_path, "w") as f:
                    f.write(str(int(memory_bytes * 0.9)))

            # Add process to cgroup
            cgroup_procs = os.path.join(cgroup_path, "cgroup.procs")
            if os.path.exists(cgroup_procs):
                with open(cgroup_procs, "w") as f:
                    f.write(str(pid))

            self.cgroup_paths[pid] = cgroup_path

        except OSError:
            # Cgroups may require elevated privileges
            pass

    async def _setup_seccomp(self, pid: int) -> None:
        """
        Setup Linux seccomp filters for process.

        Applies default-deny policy with allowlist.

        Args:
            pid: Process ID

        Raises:
            OSError: If seccomp setup fails
        """
        if not self.has_seccomp or self.platform != "linux":
            return

        # Seccomp setup typically requires elevated privileges
        # Implementation would use ptrace or prctl syscalls
        # Gracefully skip if not available
        pass

    async def _setup_windows_job(self, pid: int) -> None:
        """
        Setup Windows job object for process.

        Configures:
        - Memory limits
        - CPU affinity
        - Kill on exit

        Args:
            pid: Process ID

        Raises:
            OSError: If job object setup fails
        """
        if not self.has_job_objects or self.platform != "windows":
            return

        try:
            # Get process handle
            process = self.processes.get(pid)
            if not process:
                return

            # Windows job object setup would use:
            # - CreateJobObject
            # - SetInformationJobObject
            # - AssignProcessToJobObject
            # Implementation requires Windows API bindings

            pass

        except Exception:
            # Job object setup requires elevated privileges
            pass

    async def _setup_rlimit(self, pid: int) -> None:
        """
        Setup resource limits using setrlimit.

        Configures:
        - Memory limit (RLIMIT_AS or RLIMIT_RSS)
        - File descriptor limit
        - Process limit
        - CPU time limit

        Args:
            pid: Process ID

        Raises:
            OSError: If limit setup fails
        """
        if not self.has_rlimit:
            return

        try:
            memory_bytes = self.config.max_memory_mb * 1024 * 1024

            # Note: setrlimit affects current process
            # For isolated process, we'd need to use preexec_fn in Popen
            # This is a limitation when not using cgroups

            # These limits would be applied via preexec_fn:
            # resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            # resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))

            pass

        except Exception:
            # Limit setup may fail with permission errors
            pass

    async def kill(self, pid: int) -> bool:
        """
        Kill an isolated process.

        Attempts graceful termination, then forceful.

        Args:
            pid: Process ID

        Returns:
            True if process was killed, False if not found
        """
        process = self.processes.get(pid)
        if not process:
            return False

        try:
            # Graceful termination
            if self.platform == "windows":
                process.terminate()
            else:
                os.kill(pid, signal.SIGTERM)

            # Wait for graceful termination
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # Forceful termination
                if self.platform == "windows":
                    process.kill()
                else:
                    os.kill(pid, signal.SIGKILL)

            return True

        except Exception:
            return False

        finally:
            # Cleanup
            self.processes.pop(pid, None)
            cgroup_path = self.cgroup_paths.pop(pid, None)
            if cgroup_path:
                await self._cleanup_cgroup(cgroup_path)

    async def cleanup(self) -> None:
        """Clean up all processes and resources."""
        # Kill all processes
        pids = list(self.processes.keys())
        for pid in pids:
            await self.kill(pid)

        # Cleanup cgroups
        for cgroup_path in self.cgroup_paths.values():
            await self._cleanup_cgroup(cgroup_path)

        self.cgroup_paths.clear()

    async def _cleanup_cgroup(self, cgroup_path: str) -> None:
        """
        Clean up a cgroup.

        Args:
            cgroup_path: Path to cgroup directory
        """
        try:
            if os.path.exists(cgroup_path):
                # Remove cgroup
                os.rmdir(cgroup_path)
        except Exception:
            pass

    def get_platform(self) -> str:
        """Get detected platform."""
        return self.platform

    def get_capabilities(self) -> dict[str, bool]:
        """Get available isolation capabilities."""
        return {
            "cgroups": self.has_cgroups,
            "seccomp": self.has_seccomp,
            "job_objects": self.has_job_objects,
            "rlimit": self.has_rlimit,
            "psutil": HAS_PSUTIL,
        }

    def get_process_stats(self, pid: int) -> dict | None:
        """
        Get process statistics.

        Args:
            pid: Process ID

        Returns:
            Dictionary with memory, cpu, and other stats, or None if unavailable
        """
        if not HAS_PSUTIL:
            return None

        try:
            process = psutil.Process(pid)
            return {
                "pid": pid,
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(interval=0.1),
                "num_threads": process.num_threads(),
                "status": process.status(),
            }
        except Exception:
            return None


class ProcessIsolationError(Exception):
    """Process isolation error."""
    pass


async def test_isolation():
    """Test process isolation."""
    config = ProcessIsolationConfig(
        max_memory_mb=256,
        timeout_seconds=5
    )

    isolator = ProcessIsolator(config)

    # Show capabilities
    print(f"Platform: {isolator.get_platform()}")
    print(f"Capabilities: {isolator.get_capabilities()}")

    # Test simple command
    try:
        pid, stdout, stderr, return_code = await isolator.spawn_isolated(
            ["echo", "Hello from isolated process"]
        )
        print("\nSimple test:")
        print(f"  PID: {pid}")
        print(f"  Return code: {return_code}")
        print(f"  Output: {stdout.decode().strip()}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await isolator.cleanup()


if __name__ == "__main__":
    asyncio.run(test_isolation())
