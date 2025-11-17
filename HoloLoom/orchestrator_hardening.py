#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeavingOrchestrator Hardening
===============================
Production hardening for the main orchestration layer.

Applies Phase 2+3+4 hardening to WeavingOrchestrator:
- Resource limits (memory, concurrency, timeout)
- Metrics tracking (operations, latency, errors)
- Distributed tracing (OpenTelemetry)
- Performance profiling

Author: Claude Code
Date: 2025-11-17
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Import hardening infrastructure
from HoloLoom.reasoning.resource_limits import ResourceLimits, ResourceMonitor
from HoloLoom.reasoning.metrics import ReasoningMetrics
from HoloLoom.reasoning.tracing import ReasoningTracer
from HoloLoom.reasoning.profiling import ComponentTimer

logger = logging.getLogger(__name__)


# ============================================================================
# Orchestrator-Specific Resource Limits
# ============================================================================

@dataclass
class OrchestratorLimits:
    """
    Resource limits for orchestrator operations.

    Extended from base ResourceLimits with orchestrator-specific limits.
    """
    # Base limits
    max_memory_mb: float = 1024.0  # Higher for full pipeline
    max_concurrent_operations: int = 200  # Higher concurrency
    max_context_shards: int = 2000  # More shards for full pipeline

    # Orchestrator-specific
    max_weaving_steps: int = 9  # Max steps in weaving cycle
    max_warp_threads: int = 1000  # Max threads in warp space
    max_retrieval_results: int = 100  # Max retrieval results

    # Timeouts
    default_timeout_ms: float = 2000.0  # 2s for full weaving
    extraction_timeout_ms: float = 500.0
    retrieval_timeout_ms: float = 800.0
    reasoning_timeout_ms: float = 500.0
    synthesis_timeout_ms: float = 200.0


# ============================================================================
# Orchestrator Metrics
# ============================================================================

class OrchestratorMetrics(ReasoningMetrics):
    """
    Extended metrics for orchestrator operations.

    Tracks full weaving cycle metrics.
    """

    def __init__(self):
        super().__init__()

        # Additional orchestrator metrics
        self.weaving_operations_total = self.registry.counter(
            "weaving_operations_total",
            "Total weaving operations by pattern"
        )

        self.weaving_duration_ms = self.registry.histogram(
            "weaving_duration_ms",
            "Weaving operation duration in milliseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
        )

        self.extraction_duration_ms = self.registry.histogram(
            "extraction_duration_ms",
            "Feature extraction duration in milliseconds"
        )

        self.retrieval_duration_ms = self.registry.histogram(
            "retrieval_duration_ms",
            "Context retrieval duration in milliseconds"
        )

        self.synthesis_duration_ms = self.registry.histogram(
            "synthesis_duration_ms",
            "Response synthesis duration in milliseconds"
        )

    def track_weaving_operation(
        self,
        pattern: str,
        total_duration_ms: float,
        extraction_ms: float,
        retrieval_ms: float,
        reasoning_ms: float,
        synthesis_ms: float,
        success: bool = True
    ):
        """Track complete weaving operation with component breakdown."""
        # Total operation
        self.weaving_operations_total.inc(pattern=pattern)
        self.weaving_duration_ms.observe(total_duration_ms)

        # Component timings
        self.extraction_duration_ms.observe(extraction_ms)
        self.retrieval_duration_ms.observe(retrieval_ms)
        self.synthesis_duration_ms.observe(synthesis_ms)

        # Also track as reasoning operation
        self.track_operation(
            mode=pattern,
            duration_ms=total_duration_ms,
            confidence=0.0,  # Will be set separately
            chain_length=0,
            success=success
        )


# ============================================================================
# Hardened Orchestrator Wrapper
# ============================================================================

class HardenedOrchestrator:
    """
    Production-hardened wrapper for WeavingOrchestrator.

    Adds:
    - Resource limiting
    - Metrics tracking
    - Distributed tracing
    - Performance profiling
    - Timeout protection

    Usage:
        from HoloLoom.orchestrator_hardening import HardenedOrchestrator

        # Create hardened orchestrator
        async with HardenedOrchestrator(cfg=config, shards=shards) as orch:
            # Weave with full monitoring
            spacetime = await orch.weave(query)

            # Get metrics
            metrics = orch.get_metrics_summary()

            # Get resource stats
            stats = orch.get_resource_stats()
    """

    def __init__(
        self,
        orchestrator: Any,
        limits: Optional[OrchestratorLimits] = None,
        enable_tracing: bool = True,
        enable_profiling: bool = False
    ):
        """
        Initialize hardened orchestrator.

        Args:
            orchestrator: Underlying WeavingOrchestrator instance
            limits: Resource limits (default: OrchestratorLimits())
            enable_tracing: Enable distributed tracing
            enable_profiling: Enable detailed profiling
        """
        self.orchestrator = orchestrator
        self.limits = limits or OrchestratorLimits()
        self.enable_profiling = enable_profiling

        # Setup resource monitoring
        resource_limits = ResourceLimits(
            max_memory_mb=self.limits.max_memory_mb,
            max_concurrent_operations=self.limits.max_concurrent_operations,
            max_context_shards=self.limits.max_context_shards
        )
        self.resource_monitor = ResourceMonitor(resource_limits)

        # Setup metrics
        self.metrics = OrchestratorMetrics()

        # Setup tracing
        self.tracer = ReasoningTracer() if enable_tracing else None

        self.logger = logging.getLogger(f"{__name__}.HardenedOrchestrator")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Resource cleanup handled automatically
        pass

    async def weave(
        self,
        query: Any,
        timeout_ms: Optional[float] = None
    ) -> Any:
        """
        Weave with full monitoring and resource management.

        Args:
            query: Input query
            timeout_ms: Override timeout (default: from limits)

        Returns:
            Spacetime result

        Raises:
            asyncio.TimeoutError: If operation exceeds timeout
            ResourceLimitError: If resource limits exceeded
        """
        timeout = (timeout_ms or self.limits.default_timeout_ms) / 1000
        timer = ComponentTimer("WeavingPipeline") if self.enable_profiling else None

        # Tracing context
        trace_ctx = None
        if self.tracer:
            trace_ctx = self.tracer.trace_reasoning(
                mode="weaving",
                query_text=query.text if hasattr(query, 'text') else str(query)
            )
            trace_ctx.__enter__()

        try:
            # Resource limiting - track operation
            async with self.resource_monitor.operation_context():
                # Check memory
                if not self.resource_monitor.check_memory_limit():
                    raise RuntimeError("Memory limit exceeded before weaving")

                # Timeout protection
                start_time = time.time()

                # Component timing
                extraction_ms = 0.0
                retrieval_ms = 0.0
                reasoning_ms = 0.0
                synthesis_ms = 0.0

                # Execute weaving with timeout
                result = await asyncio.wait_for(
                    self._weave_with_timing(
                        query, timer,
                        extraction_ms, retrieval_ms, reasoning_ms, synthesis_ms
                    ),
                    timeout=timeout
                )

                # Track metrics
                total_duration_ms = (time.time() - start_time) * 1000
                self.metrics.track_weaving_operation(
                    pattern="standard",  # TODO: detect pattern from config
                    total_duration_ms=total_duration_ms,
                    extraction_ms=extraction_ms,
                    retrieval_ms=retrieval_ms,
                    reasoning_ms=reasoning_ms,
                    synthesis_ms=synthesis_ms,
                    success=True
                )

                # Print profiling summary
                if timer:
                    timer.print_summary()

                return result

        except asyncio.TimeoutError:
            self.logger.error(f"Weaving timeout after {timeout*1000:.1f}ms")
            self.metrics.track_weaving_operation(
                pattern="standard",
                total_duration_ms=timeout * 1000,
                extraction_ms=0, retrieval_ms=0,
                reasoning_ms=0, synthesis_ms=0,
                success=False
            )
            raise

        finally:
            if trace_ctx:
                trace_ctx.__exit__(None, None, None)

    async def _weave_with_timing(
        self,
        query: Any,
        timer: Optional[ComponentTimer],
        extraction_ms: float,
        retrieval_ms: float,
        reasoning_ms: float,
        synthesis_ms: float
    ) -> Any:
        """Internal weaving implementation with component timing."""
        # Delegate to underlying orchestrator
        # Note: This is a simplified version - real implementation would
        # need to intercept each component call for timing

        result = await self.orchestrator.weave(query)

        # If timing info available in result metadata, extract it
        if hasattr(result, 'metadata'):
            extraction_ms = result.metadata.get('extraction_duration_ms', 0.0)
            retrieval_ms = result.metadata.get('retrieval_duration_ms', 0.0)
            reasoning_ms = result.metadata.get('reasoning_duration_ms', 0.0)
            synthesis_ms = result.metadata.get('synthesis_duration_ms', 0.0)

        return result

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get orchestrator metrics summary."""
        return self.metrics.get_summary()

    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return self.resource_monitor.get_stats()


# ============================================================================
# Factory Function
# ============================================================================

@asynccontextmanager
async def create_hardened_orchestrator(
    cfg: Any,
    shards: Any,
    limits: Optional[OrchestratorLimits] = None,
    enable_tracing: bool = True,
    enable_profiling: bool = False
):
    """
    Factory function for creating hardened orchestrator.

    Usage:
        async with create_hardened_orchestrator(cfg, shards) as orch:
            spacetime = await orch.weave(query)
    """
    # Import here to avoid circular dependency
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

    # Create underlying orchestrator
    async with WeavingOrchestrator(cfg=cfg, shards=shards) as base_orch:
        # Wrap with hardening
        hardened = HardenedOrchestrator(
            orchestrator=base_orch,
            limits=limits,
            enable_tracing=enable_tracing,
            enable_profiling=enable_profiling
        )

        yield hardened


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("See demos/demo_hardened_orchestrator.py for usage examples")
