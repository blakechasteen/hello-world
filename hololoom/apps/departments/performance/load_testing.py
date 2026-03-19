"""
Load Testing for Departments

Concurrent load testing to stress-test department capacity.

Features:
- Concurrent request simulation
- Ramp-up patterns (gradual, spike, constant)
- Resource saturation testing
- Error rate monitoring

Author: HoloLoom B2B Framework
Date: November 2025
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from statistics import mean


class LoadPattern(Enum):
    """Load testing patterns"""
    CONSTANT = "constant"      # Constant load
    RAMP_UP = "ramp_up"        # Gradual increase
    SPIKE = "spike"            # Sudden spike then drop
    WAVE = "wave"              # Sinusoidal pattern


@dataclass
class LoadTestConfig:
    """Configuration for load test"""
    department_id: str
    pattern: LoadPattern
    max_concurrent: int
    duration_seconds: float
    ramp_up_duration: float = 0.0
    test_queries: list[str] = None


@dataclass
class LoadTestResult:
    """Results from load test"""
    department_id: str
    pattern: LoadPattern
    max_concurrent: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    requests_per_second: float
    error_rate: float
    peak_memory_mb: float
    resource_saturation: bool


class LoadTester:
    """
    Load testing for departments.

    Simulates concurrent load with various patterns.

    Example:
        >>> config = LoadTestConfig(
        ...     department_id="rag",
        ...     pattern=LoadPattern.RAMP_UP,
        ...     max_concurrent=100,
        ...     duration_seconds=60.0,
        ...     test_queries=["Test query 1", "Test query 2"]
        ... )
        >>> tester = LoadTester(config)
        >>> result = await tester.run()
        >>> print(f"Max RPS: {result.requests_per_second:.1f}")
        >>> print(f"Error rate: {result.error_rate:.1%}")
    """

    def __init__(self, config: LoadTestConfig):
        """
        Initialize load tester.

        Args:
            config: Load test configuration
        """
        self.config = config

    async def run(self) -> LoadTestResult:
        """
        Run load test.

        Returns:
            LoadTestResult with all metrics
        """
        from hololoom.departments import get_department

        dept = get_department(self.config.department_id)

        print(f"Load testing {self.config.department_id} with {self.config.pattern.value} pattern...")
        print(f"Max concurrent: {self.config.max_concurrent}, Duration: {self.config.duration_seconds}s")

        latencies = []
        successes = 0
        failures = 0

        start_time = time.time()
        end_time = start_time + self.config.duration_seconds

        # Worker function
        async def worker(query: str):
            nonlocal successes, failures

            query_start = time.perf_counter()
            try:
                response = await dept.process({
                    "task_type": "question_answering",
                    "parameters": {"query": query}
                })
                query_end = time.perf_counter()

                latency_ms = (query_end - query_start) * 1000
                latencies.append(latency_ms)

                if response.get("status") == "success":
                    successes += 1
                else:
                    failures += 1

            except Exception:
                failures += 1
                latencies.append(float('inf'))

        # Execute based on pattern
        if self.config.pattern == LoadPattern.CONSTANT:
            await self._run_constant_load(worker, start_time, end_time)

        elif self.config.pattern == LoadPattern.RAMP_UP:
            await self._run_ramp_up(worker, start_time, end_time)

        elif self.config.pattern == LoadPattern.SPIKE:
            await self._run_spike(worker, start_time, end_time)

        elif self.config.pattern == LoadPattern.WAVE:
            await self._run_wave(worker, start_time, end_time)

        duration = time.time() - start_time
        total_requests = successes + failures

        # Calculate metrics
        valid_latencies = [l for l in latencies if l != float('inf')]
        latencies_sorted = sorted(valid_latencies) if valid_latencies else [0]

        return LoadTestResult(
            department_id=self.config.department_id,
            pattern=self.config.pattern,
            max_concurrent=self.config.max_concurrent,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency_ms=mean(valid_latencies) if valid_latencies else 0.0,
            p95_latency_ms=latencies_sorted[int(len(latencies_sorted) * 0.95)],
            requests_per_second=total_requests / duration,
            error_rate=failures / total_requests if total_requests > 0 else 0.0,
            peak_memory_mb=0.0,  # TODO: Track peak memory
            resource_saturation=failures > (total_requests * 0.05)  # >5% errors = saturation
        )

    async def _run_constant_load(self, worker, start_time, end_time):
        """Run constant load pattern"""
        tasks = []
        query_index = 0

        while time.time() < end_time:
            # Maintain max_concurrent workers
            while len(tasks) < self.config.max_concurrent:
                query = self.config.test_queries[query_index % len(self.config.test_queries)]
                tasks.append(asyncio.create_task(worker(query)))
                query_index += 1

            # Wait for any task to complete
            if tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(tasks)

        # Wait for remaining tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_ramp_up(self, worker, start_time, end_time):
        """Run ramp-up pattern (gradual increase)"""
        ramp_duration = self.config.ramp_up_duration or (self.config.duration_seconds * 0.3)

        tasks = []
        query_index = 0

        while time.time() < end_time:
            elapsed = time.time() - start_time

            # Calculate target concurrency (ramps up to max)
            if elapsed < ramp_duration:
                target = int(self.config.max_concurrent * (elapsed / ramp_duration))
            else:
                target = self.config.max_concurrent

            # Adjust concurrency
            while len(tasks) < target:
                query = self.config.test_queries[query_index % len(self.config.test_queries)]
                tasks.append(asyncio.create_task(worker(query)))
                query_index += 1

            # Wait for tasks
            if tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                tasks = list(tasks)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_spike(self, worker, start_time, end_time):
        """Run spike pattern (sudden spike then drop)"""
        spike_duration = self.config.duration_seconds * 0.2

        tasks = []
        query_index = 0

        while time.time() < end_time:
            elapsed = time.time() - start_time

            # Spike at midpoint
            midpoint = self.config.duration_seconds / 2
            if abs(elapsed - midpoint) < spike_duration:
                target = self.config.max_concurrent
            else:
                target = self.config.max_concurrent // 5  # 20% baseline

            while len(tasks) < target:
                query = self.config.test_queries[query_index % len(self.config.test_queries)]
                tasks.append(asyncio.create_task(worker(query)))
                query_index += 1

            if tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                tasks = list(tasks)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_wave(self, worker, start_time, end_time):
        """Run wave pattern (sinusoidal load)"""
        import math

        tasks = []
        query_index = 0

        while time.time() < end_time:
            elapsed = time.time() - start_time

            # Sinusoidal load
            phase = (elapsed / self.config.duration_seconds) * 2 * math.pi
            target = int(self.config.max_concurrent * (0.5 + 0.5 * math.sin(phase)))
            target = max(1, target)

            while len(tasks) < target:
                query = self.config.test_queries[query_index % len(self.config.test_queries)]
                tasks.append(asyncio.create_task(worker(query)))
                query_index += 1

            if tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.1)
                tasks = list(tasks)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
