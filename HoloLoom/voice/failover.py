"""
HoloLoom VoiceAgent Multi-Region Failover Manager

Provides automatic failover capabilities across multiple regions/deployments
with health monitoring and intelligent routing.

Created: 2025-11-16
Part of: Wave 3 Production Hardening
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from enum import Enum
import asyncio
import aiohttp
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """Health status of a region."""
    HEALTHY = "healthy"       # All systems operational
    DEGRADED = "degraded"     # Operational but slow
    DOWN = "down"             # Not responding
    MAINTENANCE = "maintenance"  # Scheduled downtime


@dataclass
class Region:
    """Represents a deployment region."""
    name: str
    endpoint: str
    priority: int  # 1 = primary, 2 = secondary, etc.
    weight: float = 1.0  # For weighted load balancing
    status: RegionStatus = RegionStatus.HEALTHY
    last_check: float = 0.0
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    def __post_init__(self):
        """Validate region configuration."""
        if self.priority < 1:
            raise ValueError(f"Priority must be >= 1, got {self.priority}")
        if self.weight < 0:
            raise ValueError(f"Weight must be >= 0, got {self.weight}")

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def is_operational(self) -> bool:
        """Check if region can serve traffic."""
        return self.status in (RegionStatus.HEALTHY, RegionStatus.DEGRADED)


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""
    health_check_interval: float = 30.0  # seconds
    health_check_timeout: float = 5.0    # seconds
    degraded_threshold_ms: float = 500.0  # latency for degraded status
    down_threshold_failures: int = 3      # consecutive failures before marking down
    recovery_threshold_successes: int = 2  # consecutive successes to mark healthy
    maintenance_mode_enabled: bool = False
    enable_auto_failback: bool = True     # Automatically failback to primary
    failback_delay: float = 300.0         # Wait time before failback (seconds)


class FailoverStrategy(Enum):
    """Strategy for selecting active region."""
    PRIORITY = "priority"      # Use highest priority healthy region
    ROUND_ROBIN = "round_robin"  # Distribute across healthy regions
    WEIGHTED = "weighted"      # Weighted distribution based on performance
    LEAST_LATENCY = "least_latency"  # Choose region with lowest latency


class FailoverManager:
    """
    Manages multi-region failover for VoiceAgent.

    Features:
    - Continuous health monitoring
    - Automatic failover on failures
    - Multiple failover strategies
    - Graceful degradation
    - Auto-failback to primary
    - Maintenance mode support

    Example:
        regions = [
            Region("us-east-1", "https://voice-us-east.hololoom.ai", priority=1),
            Region("us-west-2", "https://voice-us-west.hololoom.ai", priority=2),
            Region("eu-west-1", "https://voice-eu-west.hololoom.ai", priority=3),
        ]

        manager = FailoverManager(regions)
        await manager.start()

        # Get active endpoint
        endpoint = await manager.get_active_endpoint()

        # Make request with automatic failover
        result = await manager.request("POST", "/query", json=data)
    """

    def __init__(
        self,
        regions: List[Region],
        config: Optional[FailoverConfig] = None,
        strategy: FailoverStrategy = FailoverStrategy.PRIORITY,
        on_failover: Optional[Callable[[Region, Region], None]] = None,
    ):
        """
        Initialize failover manager.

        Args:
            regions: List of regions to manage
            config: Failover configuration
            strategy: Region selection strategy
            on_failover: Callback function(old_region, new_region) on failover
        """
        if not regions:
            raise ValueError("At least one region must be provided")

        self.regions = sorted(regions, key=lambda r: r.priority)
        self.config = config or FailoverConfig()
        self.strategy = strategy
        self.on_failover = on_failover

        self.active_region = self.regions[0]
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_failover: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._round_robin_index = 0

    async def start(self):
        """Start health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring already started")
            return

        self._session = aiohttp.ClientSession()
        self._monitoring_task = asyncio.create_task(self._monitor_health())
        logger.info(f"Failover manager started with {len(self.regions)} regions")

    async def stop(self):
        """Stop health monitoring and cleanup."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Failover manager stopped")

    async def health_check(self, region: Region) -> RegionStatus:
        """
        Check health of a region.

        Args:
            region: Region to check

        Returns:
            Region status (HEALTHY, DEGRADED, or DOWN)
        """
        if region.status == RegionStatus.MAINTENANCE:
            return RegionStatus.MAINTENANCE

        try:
            start = time.time()

            async with self._session.get(
                f"{region.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
            ) as response:
                latency = (time.time() - start) * 1000  # Convert to ms
                region.latency_ms = latency

                if response.status == 200:
                    region.consecutive_successes += 1
                    region.consecutive_failures = 0

                    if latency < self.config.degraded_threshold_ms:
                        return RegionStatus.HEALTHY
                    else:
                        logger.warning(
                            f"Region {region.name} degraded: "
                            f"latency {latency:.1f}ms > {self.config.degraded_threshold_ms}ms"
                        )
                        return RegionStatus.DEGRADED
                else:
                    region.consecutive_failures += 1
                    region.consecutive_successes = 0
                    logger.error(f"Region {region.name} health check failed: HTTP {response.status}")
                    return RegionStatus.DOWN

        except asyncio.TimeoutError:
            region.consecutive_failures += 1
            region.consecutive_successes = 0
            logger.error(f"Region {region.name} health check timed out")
            return RegionStatus.DOWN

        except Exception as e:
            region.consecutive_failures += 1
            region.consecutive_successes = 0
            logger.error(f"Region {region.name} health check error: {e}")
            return RegionStatus.DOWN

    async def _monitor_health(self):
        """Continuously monitor region health."""
        while True:
            try:
                # Check all regions
                for region in self.regions:
                    new_status = await self.health_check(region)
                    old_status = region.status

                    # Update status based on consecutive failures/successes
                    if new_status == RegionStatus.DOWN:
                        if region.consecutive_failures >= self.config.down_threshold_failures:
                            region.status = RegionStatus.DOWN
                    elif new_status == RegionStatus.DEGRADED:
                        if region.status != RegionStatus.DEGRADED:
                            region.status = RegionStatus.DEGRADED
                    else:  # HEALTHY
                        if region.consecutive_successes >= self.config.recovery_threshold_successes:
                            region.status = RegionStatus.HEALTHY

                    region.last_check = time.time()

                    # Log status changes
                    if old_status != region.status:
                        logger.info(f"Region {region.name} status: {old_status.value} → {region.status.value}")

                # Check if failover is needed
                if not self.active_region.is_operational:
                    await self._perform_failover()

                # Check if failback to primary is possible
                if self.config.enable_auto_failback and self.active_region != self.regions[0]:
                    await self._check_failback()

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _perform_failover(self):
        """Perform failover to next healthy region."""
        old_region = self.active_region
        new_region = await self._select_region()

        if new_region is None:
            logger.critical("All regions are down! No healthy region available.")
            # Keep current region even if down (best effort)
            return

        if new_region != old_region:
            self.active_region = new_region
            self._last_failover = datetime.now()

            logger.warning(
                f"Failover: {old_region.name} ({old_region.status.value}) → "
                f"{new_region.name} ({new_region.status.value})"
            )

            # Trigger callback
            if self.on_failover:
                try:
                    self.on_failover(old_region, new_region)
                except Exception as e:
                    logger.error(f"Failover callback error: {e}")

    async def _check_failback(self):
        """Check if we can failback to primary region."""
        primary = self.regions[0]

        # Don't failback if primary is not healthy
        if primary.status != RegionStatus.HEALTHY:
            return

        # Don't failback too soon
        if self._last_failover:
            time_since_failover = (datetime.now() - self._last_failover).total_seconds()
            if time_since_failover < self.config.failback_delay:
                return

        # Failback to primary
        if self.active_region != primary:
            old_region = self.active_region
            self.active_region = primary
            logger.info(f"Failback to primary: {old_region.name} → {primary.name}")

            if self.on_failover:
                try:
                    self.on_failover(old_region, primary)
                except Exception as e:
                    logger.error(f"Failback callback error: {e}")

    async def _select_region(self) -> Optional[Region]:
        """
        Select best region based on strategy.

        Returns:
            Selected region or None if all are down
        """
        healthy_regions = [r for r in self.regions if r.is_operational]

        if not healthy_regions:
            return None

        if self.strategy == FailoverStrategy.PRIORITY:
            # Return highest priority (lowest number) healthy region
            return healthy_regions[0]

        elif self.strategy == FailoverStrategy.LEAST_LATENCY:
            # Return region with lowest latency
            return min(healthy_regions, key=lambda r: r.latency_ms)

        elif self.strategy == FailoverStrategy.ROUND_ROBIN:
            # Round-robin across healthy regions
            region = healthy_regions[self._round_robin_index % len(healthy_regions)]
            self._round_robin_index += 1
            return region

        elif self.strategy == FailoverStrategy.WEIGHTED:
            # Weighted random selection based on performance
            import random
            weights = [r.weight * r.success_rate for r in healthy_regions]
            return random.choices(healthy_regions, weights=weights)[0]

        else:
            return healthy_regions[0]

    async def get_active_endpoint(self) -> str:
        """
        Get currently active endpoint.

        Returns:
            Active region endpoint URL
        """
        return self.active_region.endpoint

    async def request(
        self,
        method: str,
        path: str,
        max_retries: int = 3,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """
        Make HTTP request with automatic failover.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (e.g., "/query")
            max_retries: Maximum retry attempts
            **kwargs: Additional arguments for aiohttp request

        Returns:
            Response from successful request

        Raises:
            Exception if all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                region = await self._select_region()
                if region is None:
                    raise Exception("No healthy regions available")

                url = f"{region.endpoint}{path}"

                async with self._session.request(method, url, **kwargs) as response:
                    region.total_requests += 1

                    if response.status < 400:
                        return response
                    else:
                        region.failed_requests += 1
                        last_error = f"HTTP {response.status}"

            except Exception as e:
                if region:
                    region.failed_requests += 1
                    region.total_requests += 1
                last_error = str(e)
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")

                # Trigger immediate health check
                if region:
                    await self.health_check(region)

                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

        raise Exception(f"All retries failed: {last_error}")

    def get_status(self) -> Dict:
        """
        Get current failover status.

        Returns:
            Dictionary with status information
        """
        return {
            "active_region": {
                "name": self.active_region.name,
                "endpoint": self.active_region.endpoint,
                "status": self.active_region.status.value,
                "latency_ms": self.active_region.latency_ms,
                "success_rate": self.active_region.success_rate,
            },
            "regions": [
                {
                    "name": r.name,
                    "priority": r.priority,
                    "status": r.status.value,
                    "latency_ms": r.latency_ms,
                    "success_rate": r.success_rate,
                    "last_check": r.last_check,
                }
                for r in self.regions
            ],
            "strategy": self.strategy.value,
            "last_failover": self._last_failover.isoformat() if self._last_failover else None,
        }

    def set_maintenance_mode(self, region_name: str, enabled: bool):
        """
        Set maintenance mode for a region.

        Args:
            region_name: Name of region
            enabled: True to enable maintenance mode, False to disable
        """
        for region in self.regions:
            if region.name == region_name:
                if enabled:
                    region.status = RegionStatus.MAINTENANCE
                    logger.info(f"Region {region_name} set to maintenance mode")
                else:
                    region.status = RegionStatus.HEALTHY
                    region.consecutive_failures = 0
                    region.consecutive_successes = 0
                    logger.info(f"Region {region_name} removed from maintenance mode")
                break


# Convenience function for creating standard configurations
def create_failover_manager(
    regions: List[tuple],  # List of (name, endpoint, priority) tuples
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY,
    **config_kwargs
) -> FailoverManager:
    """
    Create failover manager with simple configuration.

    Args:
        regions: List of (name, endpoint, priority) tuples
        strategy: Failover strategy
        **config_kwargs: Additional FailoverConfig parameters

    Returns:
        Configured FailoverManager

    Example:
        manager = create_failover_manager([
            ("us-east-1", "https://voice-us-east.hololoom.ai", 1),
            ("us-west-2", "https://voice-us-west.hololoom.ai", 2),
        ])
    """
    region_objects = [
        Region(name=name, endpoint=endpoint, priority=priority)
        for name, endpoint, priority in regions
    ]

    config = FailoverConfig(**config_kwargs)
    return FailoverManager(region_objects, config=config, strategy=strategy)
