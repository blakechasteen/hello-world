#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Hardening Initialization
====================================

Initialize production hardening components for deployment.

Extracted from weaving_orchestrator.py (November 2025 - Elegance Pass)
Original location: lines 824-911 (~88 lines)

This module handles:
- System monitoring (performance, resources, learning metrics)
- Circuit breaker protection for backend failures
- Rate limiting (token bucket + sliding window + concurrent)
- Health check system for load balancers
- Error handler with retry and fallback

Performance: <1ms overhead per query

Author: Claude Code (Elegance Pass Refactoring - Phase 2)
Date: 2025-11-22
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator


logger = logging.getLogger(__name__)


# Module availability flag
PRODUCTION_HARDENING_AVAILABLE = False
try:
    from HoloLoom.context import (
        ProductionConfig,
        create_system_monitor,
        create_circuit_breaker_registry,
        create_rate_limiter,
        create_health_checker,
        create_error_handler
    )
    PRODUCTION_HARDENING_AVAILABLE = True
except ImportError:
    pass


def initialize_production_hardening(
    orchestrator: 'WeavingOrchestrator',
    production_config,
    rate_limit_qps: float,
    rate_limit_concurrent: int,
    enable_circuit_breakers: bool,
    circuit_breaker_threshold: int,
    enable_health_checks: bool
) -> None:
    """
    Initialize production hardening components (Part 5: Days 21-25).

    Creates monitoring, circuit breakers, rate limiting, health checks,
    and error handling components for production deployment.

    Features:
    - System monitoring (performance, resources, learning metrics)
    - Circuit breaker protection for backend failures
    - Rate limiting (token bucket + sliding window + concurrent)
    - Health check system for load balancers
    - Error handler with retry and fallback

    Performance: <1ms overhead per query

    Args:
        orchestrator: The WeavingOrchestrator instance to initialize
        production_config: Production configuration (or None for auto-detect)
        rate_limit_qps: Queries per second limit
        rate_limit_concurrent: Maximum concurrent requests
        enable_circuit_breakers: Enable circuit breaker protection
        circuit_breaker_threshold: Failure threshold for circuit breaker
        enable_health_checks: Enable health check system

    Example:
        >>> initialize_production_hardening(
        ...     orchestrator=shuttle,
        ...     production_config=None,  # Auto-detect from environment
        ...     rate_limit_qps=100.0,
        ...     rate_limit_concurrent=50,
        ...     enable_circuit_breakers=True,
        ...     circuit_breaker_threshold=5,
        ...     enable_health_checks=True
        ... )

    Side Effects:
        Sets the following attributes on orchestrator:
        - monitor: SystemMonitor instance
        - breaker_registry: CircuitBreakerRegistry (or None)
        - rate_limiter: RateLimiter instance
        - health_checker: HealthChecker (or None)
        - error_handler: ErrorHandler instance

    Raises:
        ValueError: If production_config validation fails
    """
    if not PRODUCTION_HARDENING_AVAILABLE:
        logger.warning("Production hardening unavailable (context module not found)")
        return

    logger.info("Initializing production hardening...")

    # Load or create production configuration
    if production_config is None:
        production_config = ProductionConfig.from_environment()
        logger.info(f"  Auto-detected environment: {production_config.environment.value}")
    else:
        logger.info(f"  Using provided config: {production_config.environment.value}")

    # Validate configuration
    errors = production_config.validate()
    if errors:
        logger.error(f"Production config validation failed: {errors}")
        raise ValueError(f"Invalid production configuration: {errors}")

    # Create system monitor
    orchestrator.monitor = create_system_monitor()
    logger.info("  System monitor enabled (performance + resources + learning)")

    # Create circuit breaker registry (if enabled)
    if enable_circuit_breakers:
        orchestrator.breaker_registry = create_circuit_breaker_registry()
        logger.info(
            f"  Circuit breakers enabled "
            f"(threshold: {circuit_breaker_threshold})"
        )
    else:
        orchestrator.breaker_registry = None
        logger.info("  Circuit breakers disabled")

    # Create rate limiter
    orchestrator.rate_limiter = create_rate_limiter(
        rate=rate_limit_qps,
        capacity=int(rate_limit_qps * 0.2),  # 20% burst capacity
        max_concurrent=rate_limit_concurrent
    )
    logger.info(
        f"  Rate limiter enabled "
        f"(qps: {rate_limit_qps}, concurrent: {rate_limit_concurrent})"
    )

    # Create health checker (if enabled)
    if enable_health_checks:
        orchestrator.health_checker = create_health_checker(
            performance_monitor=orchestrator.monitor.performance if orchestrator.monitor else None,
            resource_monitor=orchestrator.monitor.resources if orchestrator.monitor else None,
            learning_monitor=orchestrator.monitor.learning if orchestrator.monitor else None,
            circuit_breaker_registry=orchestrator.breaker_registry
        )
        logger.info("  Health checker enabled (5 component checks)")
    else:
        orchestrator.health_checker = None
        logger.info("  Health checks disabled")

    # Create error handler
    orchestrator.error_handler = create_error_handler()
    logger.info("  Error handler enabled (retry + fallback)")

    logger.info("Production hardening initialization complete (<1ms overhead)")
