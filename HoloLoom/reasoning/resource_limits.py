#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource Limits for Reasoning Engine
======================================
Prevents resource exhaustion and DoS attacks.

Phase 2: Production Hardening

Author: Claude Code
Date: 2025-11-17
"""

import asyncio
import logging
import psutil
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ============================================================================
# Resource Limit Configuration
# ============================================================================

@dataclass
class ResourceLimits:
    """
    Resource limits for reasoning operations.

    Prevents:
    - Memory exhaustion (max_memory_mb)
    - Infinite loops (max_chain_steps)
    - DoS via concurrency (max_concurrent_operations)
    - Oversized context (max_context_shards)
    """

    # Memory limits
    max_memory_mb: float = 512.0  # Max memory per operation (MB)
    warn_memory_mb: float = 256.0  # Warning threshold (MB)

    # Chain length limits
    max_chain_steps: int = 50  # Max reasoning steps
    warn_chain_steps: int = 20  # Warning threshold

    # Concurrency limits
    max_concurrent_operations: int = 100  # Max parallel operations
    warn_concurrent_operations: int = 50  # Warning threshold

    # Context size limits
    max_context_shards: int = 1000  # Max context items
    max_context_tokens: int = 100000  # Estimated token limit

    # Enable/disable checks (for testing)
    enable_memory_check: bool = True
    enable_chain_check: bool = True
    enable_concurrency_check: bool = True
    enable_context_check: bool = True

    def validate(self):
        """Validate limits are sensible."""
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_chain_steps <= 0:
            raise ValueError("max_chain_steps must be positive")
        if self.max_concurrent_operations <= 0:
            raise ValueError("max_concurrent_operations must be positive")
        if self.warn_memory_mb >= self.max_memory_mb:
            raise ValueError("warn_memory_mb must be < max_memory_mb")
        if self.warn_chain_steps >= self.max_chain_steps:
            raise ValueError("warn_chain_steps must be < max_chain_steps")


# ============================================================================
# Resource Monitor
# ============================================================================

class ResourceMonitor:
    """
    Monitors resource usage during reasoning operations.

    Tracks:
    - Memory usage (via psutil)
    - Chain length (reasoning steps)
    - Concurrency (active operations)
    - Context size
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.limits.validate()

        # Track active operations (for concurrency limiting)
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrent_operations)
        self._active_operations = 0
        self._peak_operations = 0
        self._total_operations = 0

        # Get process for memory monitoring
        try:
            self._process = psutil.Process(os.getpid())
        except Exception as e:
            logger.warning(f"psutil not available: {e}. Memory monitoring disabled.")
            self._process = None
            self.limits.enable_memory_check = False

        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")

    async def acquire_operation_slot(self) -> bool:
        """
        Acquire slot for new operation (concurrency limiting).

        Returns:
            True if acquired, False if limit exceeded
        """
        if not self.limits.enable_concurrency_check:
            return True

        # Try to acquire semaphore (non-blocking check)
        if self._semaphore.locked() and self._active_operations >= self.limits.max_concurrent_operations:
            self.logger.error(
                f"Concurrency limit exceeded: {self._active_operations}/{self.limits.max_concurrent_operations}"
            )
            return False

        # Acquire semaphore (blocks if at limit)
        await self._semaphore.acquire()
        self._active_operations += 1
        self._total_operations += 1

        if self._active_operations > self._peak_operations:
            self._peak_operations = self._active_operations

        if self._active_operations >= self.limits.warn_concurrent_operations:
            self.logger.warning(
                f"High concurrency: {self._active_operations}/{self.limits.max_concurrent_operations}"
            )

        return True

    def release_operation_slot(self):
        """Release operation slot (concurrency limiting)."""
        if not self.limits.enable_concurrency_check:
            return

        self._semaphore.release()
        self._active_operations -= 1

    @asynccontextmanager
    async def operation_context(self):
        """
        Async context manager for operation lifecycle.

        Usage:
            async with monitor.operation_context():
                # Do reasoning
                pass
        """
        acquired = await self.acquire_operation_slot()
        if not acquired:
            raise RuntimeError(
                f"Concurrency limit exceeded: {self.limits.max_concurrent_operations}"
            )

        try:
            yield
        finally:
            self.release_operation_slot()

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if not self._process or not self.limits.enable_memory_check:
            return 0.0

        try:
            return self._process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def check_memory_limit(self) -> bool:
        """
        Check if memory usage is within limits.

        Returns:
            True if within limits, False if exceeded
        """
        if not self.limits.enable_memory_check:
            return True

        memory_mb = self.get_memory_usage_mb()

        if memory_mb >= self.limits.max_memory_mb:
            self.logger.error(
                f"Memory limit exceeded: {memory_mb:.1f}MB/{self.limits.max_memory_mb:.1f}MB"
            )
            return False

        if memory_mb >= self.limits.warn_memory_mb:
            self.logger.warning(
                f"High memory usage: {memory_mb:.1f}MB/{self.limits.max_memory_mb:.1f}MB"
            )

        return True

    def check_chain_length(self, current_steps: int) -> bool:
        """
        Check if chain length is within limits.

        Args:
            current_steps: Current number of reasoning steps

        Returns:
            True if within limits, False if exceeded
        """
        if not self.limits.enable_chain_check:
            return True

        if current_steps >= self.limits.max_chain_steps:
            self.logger.error(
                f"Chain length limit exceeded: {current_steps}/{self.limits.max_chain_steps}"
            )
            return False

        if current_steps >= self.limits.warn_chain_steps:
            self.logger.warning(
                f"Long chain: {current_steps}/{self.limits.max_chain_steps}"
            )

        return True

    def check_context_size(self, num_shards: int, estimated_tokens: Optional[int] = None) -> bool:
        """
        Check if context size is within limits.

        Args:
            num_shards: Number of context shards
            estimated_tokens: Estimated token count (optional)

        Returns:
            True if within limits, False if exceeded
        """
        if not self.limits.enable_context_check:
            return True

        if num_shards > self.limits.max_context_shards:
            self.logger.error(
                f"Context shard limit exceeded: {num_shards}/{self.limits.max_context_shards}"
            )
            return False

        if estimated_tokens and estimated_tokens > self.limits.max_context_tokens:
            self.logger.error(
                f"Context token limit exceeded: {estimated_tokens}/{self.limits.max_context_tokens}"
            )
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return {
            "active_operations": self._active_operations,
            "peak_operations": self._peak_operations,
            "total_operations": self._total_operations,
            "memory_mb": self.get_memory_usage_mb(),
            "limits": {
                "max_memory_mb": self.limits.max_memory_mb,
                "max_chain_steps": self.limits.max_chain_steps,
                "max_concurrent_operations": self.limits.max_concurrent_operations,
                "max_context_shards": self.limits.max_context_shards,
            }
        }


# ============================================================================
# Resource Limit Exceptions
# ============================================================================

class ResourceLimitError(Exception):
    """Base exception for resource limit violations."""
    pass


class MemoryLimitError(ResourceLimitError):
    """Memory limit exceeded."""
    pass


class ChainLengthLimitError(ResourceLimitError):
    """Chain length limit exceeded."""
    pass


class ConcurrencyLimitError(ResourceLimitError):
    """Concurrency limit exceeded."""
    pass


class ContextSizeLimitError(ResourceLimitError):
    """Context size limit exceeded."""
    pass
