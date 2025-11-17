#!/usr/bin/env python3
"""
Health Checker
=============

Verifies deployment health after deployment.

Checks:
- HTTP 200 response
- Database connection
- Workflow loaded
- Response time < 5s

Created: November 2025
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    passed: bool
    message: str
    duration_ms: float


@dataclass
class HealthStatus:
    """Overall health status."""
    healthy: bool
    checks: List[HealthCheck]
    timestamp: str
    url: str


class HealthChecker:
    """
    Check deployment health.

    Runs multiple checks to verify deployment is working correctly.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    async def check_deployment(self, url: str) -> HealthStatus:
        """
        Run all health checks on deployed application.

        Args:
            url: URL of deployed application

        Returns:
            HealthStatus with all check results
        """
        checks = []

        # Check 1: HTTP 200
        http_check = await self._check_http_200(url)
        checks.append(http_check)

        if http_check.passed:
            # Check 2: Health endpoint
            health_check = await self._check_health_endpoint(url)
            checks.append(health_check)

            # Check 3: Config endpoint
            config_check = await self._check_config_endpoint(url)
            checks.append(config_check)

            # Check 4: Response time
            response_time_check = await self._check_response_time(url)
            checks.append(response_time_check)

        # Determine overall health
        all_passed = all(check.passed for check in checks)

        return HealthStatus(
            healthy=all_passed,
            checks=checks,
            timestamp=datetime.now().isoformat(),
            url=url
        )

    async def _check_http_200(self, url: str) -> HealthCheck:
        """Check if URL returns HTTP 200."""
        start = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    duration = (datetime.now() - start).total_seconds() * 1000

                    if response.status == 200:
                        return HealthCheck(
                            name="HTTP 200",
                            passed=True,
                            message=f"Received HTTP 200 in {duration:.0f}ms",
                            duration_ms=duration
                        )
                    else:
                        return HealthCheck(
                            name="HTTP 200",
                            passed=False,
                            message=f"Received HTTP {response.status}",
                            duration_ms=duration
                        )

        except asyncio.TimeoutError:
            return HealthCheck(
                name="HTTP 200",
                passed=False,
                message=f"Timeout after {self.timeout}s",
                duration_ms=self.timeout * 1000
            )

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            return HealthCheck(
                name="HTTP 200",
                passed=False,
                message=f"Error: {str(e)}",
                duration_ms=duration
            )

    async def _check_health_endpoint(self, url: str) -> HealthCheck:
        """Check /health endpoint."""
        start = datetime.now()
        health_url = f"{url.rstrip('/')}/health"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=self.timeout) as response:
                    duration = (datetime.now() - start).total_seconds() * 1000

                    if response.status == 200:
                        data = await response.json()

                        if data.get('status') == 'ok':
                            return HealthCheck(
                                name="Health Endpoint",
                                passed=True,
                                message="Health endpoint responsive",
                                duration_ms=duration
                            )

                    return HealthCheck(
                        name="Health Endpoint",
                        passed=False,
                        message=f"Unexpected response: {response.status}",
                        duration_ms=duration
                    )

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            return HealthCheck(
                name="Health Endpoint",
                passed=False,
                message=f"Error: {str(e)}",
                duration_ms=duration
            )

    async def _check_config_endpoint(self, url: str) -> HealthCheck:
        """Check /config endpoint."""
        start = datetime.now()
        config_url = f"{url.rstrip('/')}/config"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config_url, timeout=self.timeout) as response:
                    duration = (datetime.now() - start).total_seconds() * 1000

                    if response.status == 200:
                        data = await response.json()

                        if 'workflow_id' in data:
                            return HealthCheck(
                                name="Workflow Loaded",
                                passed=True,
                                message=f"Workflow ID: {data['workflow_id']}",
                                duration_ms=duration
                            )

                    return HealthCheck(
                        name="Workflow Loaded",
                        passed=False,
                        message="Workflow configuration not found",
                        duration_ms=duration
                    )

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            return HealthCheck(
                name="Workflow Loaded",
                passed=False,
                message=f"Config unavailable: {str(e)}",
                duration_ms=duration
            )

    async def _check_response_time(self, url: str) -> HealthCheck:
        """Check if response time is acceptable (<5s)."""
        start = datetime.now()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    duration = (datetime.now() - start).total_seconds() * 1000

                    if duration < 5000:  # <5 seconds
                        return HealthCheck(
                            name="Response Time",
                            passed=True,
                            message=f"Response in {duration:.0f}ms (< 5s)",
                            duration_ms=duration
                        )
                    else:
                        return HealthCheck(
                            name="Response Time",
                            passed=False,
                            message=f"Slow response: {duration:.0f}ms (> 5s)",
                            duration_ms=duration
                        )

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            return HealthCheck(
                name="Response Time",
                passed=False,
                message=f"Error: {str(e)}",
                duration_ms=duration
            )
