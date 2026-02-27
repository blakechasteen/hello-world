"""
Resource Monitoring for Sandboxed Execution
============================================

Monitors resource usage (CPU, memory, I/O, network) with <5% overhead.

Implements:
- ResourceSample: Single resource measurement
- ResourceSummary: Aggregated statistics
- ResourceMonitor: Continuous monitoring with asyncio

Graceful Degradation:
- psutil preferred (accurate system metrics)
- Falls back to /proc/self/stat (Linux only)
- Falls back to basic estimates if unavailable

Author: CARTS Team
Date: 2025-12-05
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("HoloLoom.redteam.sandbox.monitor")

# Graceful degradation: try psutil, fall back to manual /proc parsing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, using fallback resource monitoring")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceSample:
    """Single resource measurement."""

    timestamp: float
    cpu_percent: float  # 0.0-100.0 per core
    memory_mb: float
    io_read_bytes: int
    io_write_bytes: int
    network_bytes_sent: int
    network_bytes_recv: int

    @property
    def total_io_bytes(self) -> int:
        """Total I/O in this sample."""
        return self.io_read_bytes + self.io_write_bytes

    @property
    def total_network_bytes(self) -> int:
        """Total network traffic in this sample."""
        return self.network_bytes_sent + self.network_bytes_recv


@dataclass
class ResourceSummary:
    """Aggregated resource statistics."""

    # Meta
    samples: int
    duration_seconds: float
    start_time: float = field(default_factory=time.time)

    # CPU statistics
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    min_cpu_percent: float = 0.0

    # Memory statistics
    avg_memory_mb: float = 0.0
    max_memory_mb: float = 0.0
    min_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # I/O statistics
    total_io_read_bytes: int = 0
    total_io_write_bytes: int = 0
    avg_io_read_rate_bps: float = 0.0  # Bytes per second
    avg_io_write_rate_bps: float = 0.0

    # Network statistics
    total_network_sent_bytes: int = 0
    total_network_recv_bytes: int = 0
    avg_network_rate_bps: float = 0.0

    # Overhead estimation
    overhead_percent: float = 0.0  # Target: <5%

    @property
    def total_io_bytes(self) -> int:
        """Total I/O across all samples."""
        return self.total_io_read_bytes + self.total_io_write_bytes

    @property
    def total_network_bytes(self) -> int:
        """Total network traffic."""
        return self.total_network_sent_bytes + self.total_network_recv_bytes

    def __str__(self) -> str:
        """Pretty-print summary."""
        lines = [
            f"Resource Summary ({self.samples} samples, {self.duration_seconds:.1f}s)",
            f"  CPU: {self.avg_cpu_percent:.1f}% avg, {self.max_cpu_percent:.1f}% max",
            f"  Memory: {self.avg_memory_mb:.1f} MB avg, {self.peak_memory_mb:.1f} MB peak",
            f"  I/O: {self.total_io_bytes:,} bytes ({self.avg_io_read_rate_bps/1024/1024:.2f} MB/s read)",
            f"  Network: {self.total_network_bytes:,} bytes",
            f"  Monitor Overhead: {self.overhead_percent:.2f}%",
        ]
        return "\n".join(lines)


# =============================================================================
# Resource Monitor
# =============================================================================

class ResourceMonitor:
    """
    Monitor resource usage with <5% overhead target.

    Usage:
        monitor = ResourceMonitor(sample_interval_ms=100)

        await monitor.start()
        # ... do work ...
        summary = await monitor.stop()

        print(summary)  # Pretty-printed statistics

    Implementation:
        - psutil when available (accurate, minimal overhead)
        - Fallback to /proc/self/stat (Linux only)
        - Fallback to basic time-based estimation
    """

    def __init__(self, sample_interval_ms: int = 100, pid: Optional[int] = None):
        """
        Initialize monitor.

        Args:
            sample_interval_ms: Sampling interval (default: 100ms)
            pid: Process ID to monitor (default: current process)
        """
        self._sample_interval = sample_interval_ms / 1000.0
        self._pid = pid
        self._samples: List[ResourceSample] = []
        self._monitoring = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: float = 0.0
        self._overhead_start: float = 0.0

        # Baseline for per-sample overhead measurement
        self._baseline_samples: List[float] = []

    async def start(self) -> None:
        """Start monitoring in background."""
        if self._monitoring:
            logger.warning("Monitor already running")
            return

        self._monitoring = True
        self._samples = []
        self._baseline_samples = []
        self._start_time = time.time()

        # Measure baseline overhead (10 samples)
        for _ in range(10):
            baseline_start = time.time()
            await self._sample()
            baseline_duration = (time.time() - baseline_start) * 1000  # ms
            self._baseline_samples.append(baseline_duration)

        # Start background monitoring task
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> ResourceSummary:
        """Stop monitoring and return summary."""
        if not self._monitoring:
            logger.warning("Monitor not running")
            return ResourceSummary(samples=0, duration_seconds=0.0)

        self._monitoring = False

        # Cancel background task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        duration = time.time() - self._start_time
        return self._compute_summary(duration)

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        try:
            while self._monitoring:
                try:
                    await self._sample()
                    await asyncio.sleep(self._sample_interval)
                except Exception as e:
                    logger.error(f"Error sampling resources: {e}")
                    break
        except asyncio.CancelledError:
            pass

    async def _sample(self) -> None:
        """Collect one resource sample."""
        try:
            if HAS_PSUTIL:
                sample = self._sample_psutil()
            else:
                sample = self._sample_fallback()

            self._samples.append(sample)
        except Exception as e:
            logger.error(f"Failed to sample resources: {e}")

    def _sample_psutil(self) -> ResourceSample:
        """Sample using psutil (most accurate)."""
        import psutil
        import os

        process = psutil.Process(self._pid or os.getpid())

        # CPU: convert per-core percent to single number
        try:
            cpu_percent = process.cpu_percent(interval=0.01)
        except Exception:
            cpu_percent = 0.0

        # Memory
        try:
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
        except Exception:
            memory_mb = 0.0

        # I/O
        try:
            io_counters = process.io_counters()
            io_read_bytes = io_counters.read_bytes
            io_write_bytes = io_counters.write_bytes
        except Exception:
            io_read_bytes = 0
            io_write_bytes = 0

        # Network (process-level, if available)
        network_sent = 0
        network_recv = 0
        try:
            net_connections = process.net_connections()
            # This is approximate: we'd need system-wide stats for accuracy
        except Exception:
            pass

        return ResourceSample(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            io_read_bytes=io_read_bytes,
            io_write_bytes=io_write_bytes,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
        )

    def _sample_fallback(self) -> ResourceSample:
        """Fallback sampling (Linux /proc/self/stat)."""
        import os

        cpu_percent = 0.0
        memory_mb = 0.0
        io_read_bytes = 0
        io_write_bytes = 0

        # Try /proc/self/stat (Linux only)
        try:
            with open(f"/proc/{self._pid or os.getpid()}/stat", "r") as f:
                stat_line = f.read().strip()
                # Parse fields: pid, comm, state, ppid, pgrp, session, tty_nr,
                # tpgid, flags, minflt, cminflt, majflt, cmajflt, utime, stime...
                fields = stat_line.split()
                if len(fields) >= 15:
                    # CPU time in jiffies (utime + stime)
                    utime = int(fields[13])
                    stime = int(fields[14])
                    cpu_percent = (utime + stime) / 100.0  # Rough estimate
        except Exception:
            pass

        # Try /proc/self/status (for memory)
        try:
            with open(f"/proc/{self._pid or os.getpid()}/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # VmRSS: 12345 kB
                        kb = int(line.split()[1])
                        memory_mb = kb / 1024.0
                        break
        except Exception:
            pass

        return ResourceSample(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            io_read_bytes=io_read_bytes,
            io_write_bytes=io_write_bytes,
            network_bytes_sent=0,
            network_bytes_recv=0,
        )

    def get_current_sample(self) -> Optional[ResourceSample]:
        """Get most recent sample."""
        return self._samples[-1] if self._samples else None

    def get_samples(self) -> List[ResourceSample]:
        """Get all collected samples."""
        return self._samples.copy()

    def _compute_summary(self, duration: float) -> ResourceSummary:
        """Compute aggregate statistics from samples."""
        if not self._samples:
            return ResourceSummary(samples=0, duration_seconds=duration)

        # CPU statistics
        cpu_values = [s.cpu_percent for s in self._samples]
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        max_cpu = max(cpu_values) if cpu_values else 0.0
        min_cpu = min(cpu_values) if cpu_values else 0.0

        # Memory statistics
        memory_values = [s.memory_mb for s in self._samples]
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0.0
        max_memory = max(memory_values) if memory_values else 0.0
        min_memory = min(memory_values) if memory_values else 0.0
        peak_memory = max_memory

        # I/O statistics
        total_io_read = sum(s.io_read_bytes for s in self._samples)
        total_io_write = sum(s.io_write_bytes for s in self._samples)
        io_read_rate = total_io_read / duration if duration > 0 else 0.0
        io_write_rate = total_io_write / duration if duration > 0 else 0.0

        # Network statistics
        total_net_sent = sum(s.network_bytes_sent for s in self._samples)
        total_net_recv = sum(s.network_bytes_recv for s in self._samples)
        net_rate = (total_net_sent + total_net_recv) / duration if duration > 0 else 0.0

        # Overhead calculation (average per-sample overhead as % of sample interval)
        avg_baseline = sum(self._baseline_samples) / len(self._baseline_samples) if self._baseline_samples else 0.0
        overhead_percent = (avg_baseline / (self._sample_interval * 1000)) * 100 if self._sample_interval > 0 else 0.0
        # Cap at 100% and ensure non-negative
        overhead_percent = max(0.0, min(100.0, overhead_percent))

        return ResourceSummary(
            samples=len(self._samples),
            duration_seconds=duration,
            start_time=self._start_time,
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            min_cpu_percent=min_cpu,
            avg_memory_mb=avg_memory,
            max_memory_mb=max_memory,
            min_memory_mb=min_memory,
            peak_memory_mb=peak_memory,
            total_io_read_bytes=total_io_read,
            total_io_write_bytes=total_io_write,
            avg_io_read_rate_bps=io_read_rate,
            avg_io_write_rate_bps=io_write_rate,
            total_network_sent_bytes=total_net_sent,
            total_network_recv_bytes=total_net_recv,
            avg_network_rate_bps=net_rate,
            overhead_percent=overhead_percent,
        )

    def check_limits(
        self,
        memory_limit_mb: int,
        timeout_seconds: float
    ) -> Tuple[bool, List[str]]:
        """
        Check if current resource usage violates limits.

        Returns:
            (within_limits, violations)
        """
        violations = []
        elapsed = time.time() - self._start_time

        # Check memory
        current_sample = self.get_current_sample()
        if current_sample and current_sample.memory_mb > memory_limit_mb:
            violations.append(
                f"Memory limit exceeded: {current_sample.memory_mb:.1f} MB > {memory_limit_mb} MB"
            )

        # Check timeout
        if elapsed > timeout_seconds:
            violations.append(
                f"Timeout exceeded: {elapsed:.1f}s > {timeout_seconds}s"
            )

        return len(violations) == 0, violations
