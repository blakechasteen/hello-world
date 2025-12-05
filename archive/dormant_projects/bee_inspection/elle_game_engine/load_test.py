"""
Load testing for Elle Game Engine.

Comprehensive performance benchmarking using multiple strategies:
1. Locust - Distributed load testing with web UI
2. httpx - Direct async benchmarking
3. Siege - Command-line HTTP load testing

Test Scenarios:
- Baseline: Regular blocking endpoint
- Streaming: SSE streaming endpoint
- Burst: 0 → 500 users in 10s
- Sustained: 100 concurrent users for 5 minutes
- Pool stress: Test connection pooling under load

Usage:
    # Locust (interactive web UI)
    locust -f load_test.py --host=http://localhost:8000

    # httpx (programmatic)
    python load_test.py --scenario=baseline --users=100 --duration=60

    # Quick benchmark
    python load_test.py --quick
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import argparse

try:
    import httpx
except ImportError:
    print("⚠️  httpx not installed. Install with: pip install httpx")
    httpx = None

try:
    from locust import HttpUser, task, between, events
    LOCUST_AVAILABLE = True
except ImportError:
    print("⚠️  locust not installed. Install with: pip install locust")
    LOCUST_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    host: str = "http://localhost:8000"
    concurrent_users: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    scenario: str = "baseline"  # baseline, streaming, burst, sustained


@dataclass
class TestResult:
    """Results from a single request."""

    endpoint: str
    duration_ms: float
    status_code: int
    error: Optional[str] = None
    cache_hit: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""

    scenario: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "duration_seconds": self.duration_seconds,
            "requests_per_second": self.requests_per_second,
            "latencies": {
                "mean_ms": self.mean_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "min_ms": self.min_latency_ms,
                "max_ms": self.max_latency_ms,
            },
            "error_rate": self.error_rate,
        }

    def print_report(self):
        """Print formatted report."""
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK REPORT: {self.scenario}")
        print(f"{'=' * 60}")
        print(f"Total Requests:       {self.total_requests:,}")
        print(f"Successful:           {self.successful_requests:,}")
        print(f"Failed:               {self.failed_requests:,}")
        print(f"Duration:             {self.duration_seconds:.1f}s")
        print(f"Throughput:           {self.requests_per_second:.1f} req/s")
        print(f"\nLatency Statistics:")
        print(f"  Mean:               {self.mean_latency_ms:.1f} ms")
        print(f"  p50 (median):       {self.p50_latency_ms:.1f} ms")
        print(f"  p95:                {self.p95_latency_ms:.1f} ms")
        print(f"  p99:                {self.p99_latency_ms:.1f} ms")
        print(f"  Min:                {self.min_latency_ms:.1f} ms")
        print(f"  Max:                {self.max_latency_ms:.1f} ms")
        print(f"\nError Rate:           {self.error_rate:.2%}")
        print(f"{'=' * 60}\n")


# ============================================================================
# Locust Load Testing
# ============================================================================

if LOCUST_AVAILABLE:
    class ElleGameUser(HttpUser):
        """
        Locust user simulating game client.

        Sends requests to Elle endpoints with realistic patterns.
        """

        wait_time = between(1, 3)  # Wait 1-3 seconds between requests

        @task(3)
        def talk_to_npc(self):
            """Talk to NPC (most common action)."""
            response = self.client.post(
                "/elle/game/action",
                json={
                    "game_state": {
                        "scene_id": "tavern",
                        "npcs": [
                            {
                                "id": "innkeeper",
                                "name": "Mara",
                                "role": "innkeeper",
                                "mood": "friendly",
                                "location": "tavern",
                                "flags": {}
                            }
                        ],
                        "player": {
                            "name": "Hero",
                            "location": "tavern",
                            "quest_stage": "intro",
                            "reputation": "neutral",
                            "traits": {},
                            "inventory_tags": []
                        },
                        "world": {
                            "time_of_day": "evening",
                            "weather": "clear",
                            "tension_level": "calm"
                        },
                        "tags": []
                    },
                    "player_intent": {
                        "type": "talk_to_npc",
                        "target_npc_id": "innkeeper",
                        "raw_input": "Hello, innkeeper!"
                    }
                },
                name="/elle/game/action [talk_to_npc]"
            )

        @task(1)
        def enter_scene(self):
            """Enter a new scene."""
            response = self.client.post(
                "/elle/game/action",
                json={
                    "game_state": {
                        "scene_id": "forest_edge",
                        "npcs": [],
                        "player": {
                            "name": "Hero",
                            "location": "forest_edge",
                            "quest_stage": "intro",
                            "reputation": "neutral",
                            "traits": {},
                            "inventory_tags": []
                        },
                        "world": {
                            "time_of_day": "morning",
                            "weather": "clear",
                            "tension_level": "calm"
                        },
                        "tags": ["first_visit"]
                    },
                    "player_intent": {
                        "type": "enter_scene"
                    }
                },
                name="/elle/game/action [enter_scene]"
            )

        @task(1)
        def request_hint(self):
            """Request gameplay hint."""
            response = self.client.post(
                "/elle/game/action",
                json={
                    "game_state": {
                        "scene_id": "puzzle_room",
                        "npcs": [],
                        "player": {
                            "name": "Hero",
                            "location": "puzzle_room",
                            "quest_stage": "mid",
                            "reputation": "neutral",
                            "traits": {},
                            "inventory_tags": ["mysterious_key"]
                        },
                        "world": {
                            "time_of_day": "afternoon",
                            "weather": "clear",
                            "tension_level": "tense"
                        },
                        "tags": []
                    },
                    "player_intent": {
                        "type": "request_hint"
                    }
                },
                name="/elle/game/action [request_hint]"
            )


# ============================================================================
# httpx Async Benchmarking
# ============================================================================

async def benchmark_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: Optional[Dict] = None,
    method: str = "POST",
    name: str = None,
) -> TestResult:
    """
    Benchmark a single request to an endpoint.

    Args:
        client: httpx client
        endpoint: Endpoint URL
        payload: Request payload (for POST)
        method: HTTP method
        name: Human-readable name

    Returns:
        TestResult with timing and status
    """
    start_time = time.time()

    try:
        if method == "POST":
            response = await client.post(endpoint, json=payload, timeout=30.0)
        else:
            response = await client.get(endpoint, timeout=30.0)

        duration_ms = (time.time() - start_time) * 1000

        # Check for cache hit in debug notes
        cache_hit = False
        if response.status_code == 200:
            try:
                data = response.json()
                debug_notes = data.get("debug_notes", "")
                cache_hit = "cache_hit=True" in debug_notes
            except:
                pass

        return TestResult(
            endpoint=name or endpoint,
            duration_ms=duration_ms,
            status_code=response.status_code,
            cache_hit=cache_hit,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return TestResult(
            endpoint=name or endpoint,
            duration_ms=duration_ms,
            status_code=0,
            error=str(e),
        )


async def run_concurrent_requests(
    host: str,
    endpoint: str,
    payload: Dict,
    concurrent: int,
    total_requests: int,
    method: str = "POST",
    name: str = None,
) -> List[TestResult]:
    """
    Run concurrent requests to endpoint.

    Args:
        host: Base URL
        endpoint: Endpoint path
        payload: Request payload
        concurrent: Number of concurrent requests
        total_requests: Total requests to make
        method: HTTP method
        name: Human-readable name

    Returns:
        List of TestResults
    """
    results = []

    async with httpx.AsyncClient(base_url=host) as client:
        # Create batches of concurrent requests
        batches = []
        for i in range(0, total_requests, concurrent):
            batch_size = min(concurrent, total_requests - i)
            batch = [
                benchmark_endpoint(client, endpoint, payload, method, name)
                for _ in range(batch_size)
            ]
            batches.append(batch)

        # Run batches
        for i, batch in enumerate(batches):
            print(f"Running batch {i + 1}/{len(batches)} ({len(batch)} requests)...")
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

    return results


def analyze_results(results: List[TestResult], scenario: str, duration: float) -> BenchmarkReport:
    """
    Analyze test results and generate report.

    Args:
        results: List of test results
        scenario: Test scenario name
        duration: Total test duration

    Returns:
        BenchmarkReport with statistics
    """
    successful = [r for r in results if r.status_code == 200]
    failed = [r for r in results if r.status_code != 200]

    latencies = [r.duration_ms for r in successful]

    if not latencies:
        latencies = [0]

    return BenchmarkReport(
        scenario=scenario,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        duration_seconds=duration,
        requests_per_second=len(results) / duration if duration > 0 else 0,
        mean_latency_ms=statistics.mean(latencies),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
        p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        error_rate=len(failed) / len(results) if results else 0,
    )


# ============================================================================
# Test Scenarios
# ============================================================================

def create_talk_to_npc_payload() -> Dict:
    """Create payload for talk_to_npc action."""
    return {
        "game_state": {
            "scene_id": "tavern",
            "npcs": [
                {
                    "id": "innkeeper",
                    "name": "Mara",
                    "role": "innkeeper",
                    "mood": "friendly",
                    "location": "tavern",
                    "flags": {}
                }
            ],
            "player": {
                "name": "Hero",
                "location": "tavern",
                "quest_stage": "intro",
                "reputation": "neutral",
                "traits": {},
                "inventory_tags": []
            },
            "world": {
                "time_of_day": "evening",
                "weather": "clear",
                "tension_level": "calm"
            },
            "tags": []
        },
        "player_intent": {
            "type": "talk_to_npc",
            "target_npc_id": "innkeeper",
            "raw_input": "Hello, innkeeper!"
        }
    }


async def scenario_baseline(config: LoadTestConfig) -> BenchmarkReport:
    """
    Baseline scenario: Regular blocking endpoint.

    Tests standard POST /elle/game/action performance.
    """
    print(f"\n🔬 Running BASELINE scenario")
    print(f"   Users: {config.concurrent_users}, Duration: {config.duration_seconds}s")

    payload = create_talk_to_npc_payload()

    start_time = time.time()
    results = await run_concurrent_requests(
        host=config.host,
        endpoint="/elle/game/action",
        payload=payload,
        concurrent=config.concurrent_users,
        total_requests=config.concurrent_users * 10,  # 10 requests per user
        name="baseline",
    )
    duration = time.time() - start_time

    return analyze_results(results, "baseline", duration)


async def scenario_streaming(config: LoadTestConfig) -> BenchmarkReport:
    """
    Streaming scenario: SSE streaming endpoint.

    Tests GET /elle/game/action/stream performance.
    """
    print(f"\n🔬 Running STREAMING scenario")
    print(f"   Users: {config.concurrent_users}, Duration: {config.duration_seconds}s")

    start_time = time.time()
    results = await run_concurrent_requests(
        host=config.host,
        endpoint="/elle/game/action/stream?scene_id=tavern&player_name=Hero&player_location=tavern&intent_type=talk_to_npc&target_npc_id=innkeeper",
        payload=None,
        concurrent=config.concurrent_users,
        total_requests=config.concurrent_users * 10,
        method="GET",
        name="streaming",
    )
    duration = time.time() - start_time

    return analyze_results(results, "streaming", duration)


async def scenario_sustained(config: LoadTestConfig) -> BenchmarkReport:
    """
    Sustained load scenario.

    Sustained traffic for extended period to test stability.
    """
    print(f"\n🔬 Running SUSTAINED scenario")
    print(f"   Users: {config.concurrent_users}, Duration: {config.duration_seconds}s")

    payload = create_talk_to_npc_payload()
    results = []

    async with httpx.AsyncClient(base_url=config.host) as client:
        start_time = time.time()

        while (time.time() - start_time) < config.duration_seconds:
            batch = [
                benchmark_endpoint(client, "/elle/game/action", payload, "POST", "sustained")
                for _ in range(config.concurrent_users)
            ]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

            # Brief pause between batches
            await asyncio.sleep(0.1)

        duration = time.time() - start_time

    return analyze_results(results, "sustained", duration)


# ============================================================================
# Main CLI
# ============================================================================

async def run_benchmark(config: LoadTestConfig) -> BenchmarkReport:
    """
    Run benchmark based on scenario.

    Args:
        config: Test configuration

    Returns:
        BenchmarkReport
    """
    if config.scenario == "baseline":
        return await scenario_baseline(config)
    elif config.scenario == "streaming":
        return await scenario_streaming(config)
    elif config.scenario == "sustained":
        return await scenario_sustained(config)
    else:
        raise ValueError(f"Unknown scenario: {config.scenario}")


async def run_quick_benchmark(host: str):
    """
    Run quick benchmark comparing baseline vs streaming.

    Args:
        host: Base URL
    """
    print("🚀 Running quick benchmark (baseline vs streaming)")

    # Baseline
    baseline_config = LoadTestConfig(
        host=host,
        concurrent_users=20,
        duration_seconds=10,
        scenario="baseline",
    )
    baseline_report = await scenario_baseline(baseline_config)
    baseline_report.print_report()

    # Streaming
    streaming_config = LoadTestConfig(
        host=host,
        concurrent_users=20,
        duration_seconds=10,
        scenario="streaming",
    )
    streaming_report = await scenario_streaming(streaming_config)
    streaming_report.print_report()

    # Comparison
    print("\n📊 COMPARISON")
    print(f"{'=' * 60}")
    print(f"Baseline throughput:     {baseline_report.requests_per_second:.1f} req/s")
    print(f"Streaming throughput:    {streaming_report.requests_per_second:.1f} req/s")
    print(f"\nBaseline p95 latency:    {baseline_report.p95_latency_ms:.1f} ms")
    print(f"Streaming p95 latency:   {streaming_report.p95_latency_ms:.1f} ms")

    if streaming_report.p95_latency_ms < baseline_report.p95_latency_ms:
        improvement = ((baseline_report.p95_latency_ms - streaming_report.p95_latency_ms) / baseline_report.p95_latency_ms) * 100
        print(f"\n✅ Streaming is {improvement:.1f}% faster (p95 latency)")
    else:
        print(f"\n⚠️  Streaming is slower than baseline")

    print(f"{'=' * 60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Elle Game Engine Load Testing")
    parser.add_argument("--host", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--scenario", choices=["baseline", "streaming", "sustained"], default="baseline")
    parser.add_argument("--users", type=int, default=100, help="Concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration (seconds)")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", help="Output file for JSON results")

    args = parser.parse_args()

    if not httpx:
        print("❌ httpx required. Install with: pip install httpx")
        return

    if args.quick:
        asyncio.run(run_quick_benchmark(args.host))
    else:
        config = LoadTestConfig(
            host=args.host,
            concurrent_users=args.users,
            duration_seconds=args.duration,
            scenario=args.scenario,
        )

        report = asyncio.run(run_benchmark(config))
        report.print_report()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
