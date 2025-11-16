"""
HoloLoom VoiceAgent + Elle AR Integration - Locust Load Testing Suite

This module simulates real-world user behavior for load testing the VoiceAgent
and Elle AR integration system. It includes multiple task types representing
different user interactions in a production environment.

Author: Wave 3 Production Hardening
Date: 2025-11-16
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# Custom Metrics & Event Handlers
# ============================================================================

class MetricsCollector:
    """Collects detailed metrics for analysis."""

    def __init__(self):
        self.request_times = {}
        self.error_counts = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()

    def record_request(self, name: str, response_time: float, success: bool):
        """Record request metrics."""
        if name not in self.request_times:
            self.request_times[name] = []
        self.request_times[name].append(response_time)

        if not success:
            self.error_counts[name] = self.error_counts.get(name, 0) + 1

    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        elapsed = time.time() - self.start_time
        total_requests = sum(len(times) for times in self.request_times.values())
        total_errors = sum(self.error_counts.values())

        summary = {
            "elapsed_seconds": elapsed,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
                             if (self.cache_hits + self.cache_misses) > 0 else 0,
            "endpoints": {}
        }

        for endpoint, times in self.request_times.items():
            if times:
                times_sorted = sorted(times)
                summary["endpoints"][endpoint] = {
                    "count": len(times),
                    "errors": self.error_counts.get(endpoint, 0),
                    "p50": times_sorted[len(times) // 2],
                    "p95": times_sorted[int(len(times) * 0.95)],
                    "p99": times_sorted[int(len(times) * 0.99)] if len(times) > 100 else times_sorted[-1],
                    "max": max(times),
                    "avg": sum(times) / len(times)
                }

        return summary

# Global metrics collector
metrics_collector = MetricsCollector()


# ============================================================================
# VoiceAgent User Simulation
# ============================================================================

class VoiceAgentUser(HttpUser):
    """
    Simulates VoiceAgent user behavior in Elle AR environment.

    Task Weights:
    - Voice command processing: 10 (40% of traffic)
    - TTS synthesis: 5 (20% of traffic)
    - Health checks: 3 (12% of traffic)
    - Stats retrieval: 2 (8% of traffic)
    - Photo annotation: 4 (16% of traffic)
    - Context update: 2 (8% of traffic)
    - Batch operations: 1 (4% of traffic)

    Wait Time: 1-5 seconds between requests (simulating human pauses)
    """

    wait_time = between(1, 5)

    # ========================================================================
    # Test Data
    # ========================================================================

    # Sample voice commands (realistic beekeeping inspection queries)
    voice_commands = [
        "Show me the health status of this hive",
        "What is that hive?",
        "Go to the hive on my left",
        "Start inspection",
        "Tell me about hive 003",
        "Explain why the population is low",
        "What's the frame coverage?",
        "Is there enough honey stored?",
        "Show me the brood pattern",
        "Any signs of disease?",
        "What's the queen status?",
        "How many bees are there?",
        "Is the hive healthy?",
        "What should I do next?",
        "Show me the history of this hive"
    ]

    # TTS voice options (matching API)
    tts_voices = ["nova", "alloy", "shimmer", "onyx", "echo", "fable"]

    # Supported languages
    languages = ["en", "es", "fr", "de", "ja", "zh", "pt", "it"]

    # Hive identifiers
    hive_ids = ["hive_001", "hive_002", "hive_003", "hive_004", "hive_005"]

    # Annotation types
    annotation_types = ["health_assessment", "brood_pattern", "pest_damage",
                       "resource_status", "behavior_observation"]

    # ========================================================================
    # Tasks: Voice Command Processing (40% of traffic)
    # ========================================================================

    @task(10)
    def query_voice_command(self):
        """
        Test voice command processing through main query endpoint.

        This is the primary workload - users asking questions about their hives.
        Simulates single-step queries with optional verification.

        Expected Latency: p50=150ms, p95=300ms
        """
        payload = {
            "text": random.choice(self.voice_commands),
            "context": {
                "user_position": [random.uniform(-5, 5), 1.7, random.uniform(-5, 5)],
                "gaze_target": random.choice(self.hive_ids),
                "visible_objects": [random.choice(self.hive_ids) for _ in range(random.randint(1, 3))],
                "timestamp": datetime.now().isoformat()
            },
            "mode": random.choice(["direct", "verify", "research"]),
            "max_steps": random.choice([1, 3, 5])
        }

        start_time = time.time()
        with self.client.post(
            "/query",
            json=payload,
            catch_response=True,
            name="/query [voice_command]"
        ) as response:
            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code == 200:
                try:
                    data = response.json()
                    processing_time = data.get('processing_time_ms', elapsed)

                    # Check latency SLA
                    if processing_time > 2000:  # >2s
                        response.failure(f"Slow: {processing_time:.0f}ms (SLA: <2000ms)")
                        metrics_collector.record_request("/query", processing_time, False)
                    else:
                        response.success()
                        metrics_collector.record_request("/query", processing_time, True)

                        # Track cache hits for verification queries
                        if data.get('cache_hit'):
                            metrics_collector.record_cache_hit()
                        else:
                            metrics_collector.record_cache_miss()
                except (json.JSONDecodeError, KeyError) as e:
                    response.failure(f"Invalid response format: {e}")
                    metrics_collector.record_request("/query", elapsed, False)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/query", elapsed, False)

    # ========================================================================
    # Tasks: TTS Synthesis (20% of traffic)
    # ========================================================================

    @task(5)
    def synthesize_tts(self):
        """
        Test TTS (Text-To-Speech) synthesis with caching.

        Simulates the system responding to queries with spoken answers.
        Tests cache effectiveness for repeated queries in multiple languages.

        Expected Latency (cached): p50=15ms, p95=30ms
        Expected Latency (uncached): p50=500ms, p95=800ms
        """
        # 60% chance of cache hit (repeated queries)
        force_cache = random.random() < 0.6

        payload = {
            "text": random.choice(self.voice_commands),
            "voice": random.choice(self.tts_voices),
            "language": random.choice(self.languages),
            "force_cache": force_cache
        }

        start_time = time.time()
        with self.client.post(
            "/tts/synthesize",
            json=payload,
            catch_response=True,
            name="/tts/synthesize"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                try:
                    # Check cache header
                    cache_hit = response.headers.get('X-Cache-Hit', 'false').lower() == 'true'

                    # Validate latency based on cache status
                    if cache_hit:
                        if elapsed > 100:  # Cached should be <100ms
                            response.failure(f"Slow cached TTS: {elapsed:.0f}ms")
                            metrics_collector.record_request("/tts/synthesize [cached]", elapsed, False)
                        else:
                            response.success()
                            metrics_collector.record_request("/tts/synthesize [cached]", elapsed, True)
                            metrics_collector.record_cache_hit()
                    else:
                        if elapsed > 1500:  # Uncached should be <1500ms
                            response.failure(f"Slow uncached TTS: {elapsed:.0f}ms")
                            metrics_collector.record_request("/tts/synthesize [uncached]", elapsed, False)
                        else:
                            response.success()
                            metrics_collector.record_request("/tts/synthesize [uncached]", elapsed, True)
                            metrics_collector.record_cache_miss()

                except Exception as e:
                    response.failure(f"TTS error: {e}")
                    metrics_collector.record_request("/tts/synthesize", elapsed, False)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/tts/synthesize", elapsed, False)

    # ========================================================================
    # Tasks: Health Checks (12% of traffic)
    # ========================================================================

    @task(3)
    def check_health(self):
        """
        Test system health endpoint.

        Simulates monitoring/load balancer health checks.
        Should be very fast (<50ms) and have zero errors.

        Expected Latency: p50=5ms, p95=10ms, p99=20ms
        """
        start_time = time.time()
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        response.success()
                        metrics_collector.record_request("/health", elapsed, True)
                    else:
                        response.failure(f"Unhealthy status: {data.get('status')}")
                        metrics_collector.record_request("/health", elapsed, False)
                except Exception as e:
                    response.failure(f"Health check error: {e}")
                    metrics_collector.record_request("/health", elapsed, False)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/health", elapsed, False)

    # ========================================================================
    # Tasks: Statistics Retrieval (8% of traffic)
    # ========================================================================

    @task(2)
    def get_stats(self):
        """
        Test system statistics endpoint.

        Simulates dashboards and monitoring tools retrieving performance metrics.
        Tests ability to serve aggregated data efficiently.

        Expected Latency: p50=50ms, p95=100ms
        """
        params = {
            "interval": random.choice(["1h", "24h", "7d"]),
            "metric": random.choice(["latency", "throughput", "errors", "cache_hit_rate"])
        }

        start_time = time.time()
        with self.client.get(
            "/stats",
            params=params,
            catch_response=True,
            name="/stats"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                response.success()
                metrics_collector.record_request("/stats", elapsed, True)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/stats", elapsed, False)

    # ========================================================================
    # Tasks: Photo Annotation (16% of traffic)
    # ========================================================================

    @task(4)
    def annotate_photo(self):
        """
        Test photo annotation endpoint (Elle AR integration).

        Simulates users annotating inspection photos with assessments.
        Tests ability to handle image metadata and structured data.

        Expected Latency: p50=200ms, p95=400ms
        """
        payload = {
            "hive_id": random.choice(self.hive_ids),
            "annotation_type": random.choice(self.annotation_types),
            "photo_id": f"photo_{random.randint(1000, 9999)}",
            "tags": [random.choice(["healthy", "disease", "pest", "good_resources", "low_resources"])
                    for _ in range(random.randint(1, 3))],
            "confidence": random.uniform(0.7, 1.0),
            "notes": f"Inspection note {random.randint(1, 100)}"
        }

        start_time = time.time()
        with self.client.post(
            "/annotate-photo",
            json=payload,
            catch_response=True,
            name="/annotate-photo"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                if elapsed > 1000:  # >1s
                    response.failure(f"Slow annotation: {elapsed:.0f}ms")
                    metrics_collector.record_request("/annotate-photo", elapsed, False)
                else:
                    response.success()
                    metrics_collector.record_request("/annotate-photo", elapsed, True)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/annotate-photo", elapsed, False)

    # ========================================================================
    # Tasks: Context Update (8% of traffic)
    # ========================================================================

    @task(2)
    def update_context(self):
        """
        Test context update endpoint (AR position/orientation).

        Simulates continuous AR context updates as user moves around apiary.
        Tests ability to handle frequent state updates efficiently.

        Expected Latency: p50=20ms, p95=50ms
        """
        payload = {
            "user_position": [random.uniform(-5, 5), random.uniform(1.5, 2.5),
                            random.uniform(-5, 5)],
            "user_orientation": [random.uniform(0, 360), random.uniform(-90, 90), 0],
            "visible_objects": [random.choice(self.hive_ids) for _ in range(random.randint(1, 5))],
            "timestamp": datetime.now().isoformat()
        }

        start_time = time.time()
        with self.client.post(
            "/context/update",
            json=payload,
            catch_response=True,
            name="/context/update"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                if elapsed > 200:  # >200ms is slow for context
                    response.failure(f"Slow context update: {elapsed:.0f}ms")
                    metrics_collector.record_request("/context/update", elapsed, False)
                else:
                    response.success()
                    metrics_collector.record_request("/context/update", elapsed, True)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/context/update", elapsed, False)

    # ========================================================================
    # Tasks: Batch Operations (4% of traffic)
    # ========================================================================

    @task(1)
    def batch_query(self):
        """
        Test batch query endpoint for advanced workflows.

        Simulates batch processing of multiple hive assessments.
        Tests ability to parallelize and aggregate multiple queries efficiently.

        Expected Latency: p50=500ms, p95=1000ms (linear with query count)
        """
        query_count = random.randint(2, 5)

        payload = {
            "queries": [
                {
                    "text": random.choice(self.voice_commands),
                    "hive_id": random.choice(self.hive_ids)
                }
                for _ in range(query_count)
            ],
            "parallel": random.choice([True, False])
        }

        start_time = time.time()
        with self.client.post(
            "/batch/query",
            json=payload,
            catch_response=True,
            name="/batch/query"
        ) as response:
            elapsed = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Expected time should scale with query count (~150ms each)
                expected_max = query_count * 150 * 2  # 2x for aggregation
                if elapsed > expected_max:
                    response.failure(f"Slow batch (q={query_count}): {elapsed:.0f}ms > {expected_max}ms")
                    metrics_collector.record_request("/batch/query", elapsed, False)
                else:
                    response.success()
                    metrics_collector.record_request("/batch/query", elapsed, True)
            else:
                response.failure(f"HTTP {response.status_code}")
                metrics_collector.record_request("/batch/query", elapsed, False)


# ============================================================================
# Event Handlers for Metrics Collection
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Handle request completion event."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test run."""
    print("\n" + "=" * 80)
    print("HOLOLOOM VOICEAGENT + ELLE AR - LOAD TEST STARTED")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    runner = environment.runner
    if isinstance(runner, MasterRunner):
        target = runner.target_user_count
        spawn_rate = runner.spawn_rate
    else:
        target = getattr(runner, 'target_user_count', 'unknown')
        spawn_rate = getattr(runner, 'spawn_rate', 'unknown')

    print(f"Target Users: {target}")
    print(f"Spawn Rate: {spawn_rate} users/sec")
    print("=" * 80 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Report test summary."""
    print("\n" + "=" * 80)
    print("LOAD TEST SUMMARY")
    print("=" * 80)

    # Get summary from metrics collector
    summary = metrics_collector.get_summary()

    print(f"\nTest Duration: {summary['elapsed_seconds']:.1f}s")
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Total Failures: {summary['total_errors']}")
    print(f"Overall Error Rate: {summary['error_rate']:.2f}%")
    print(f"\nCache Performance:")
    print(f"  Cache Hits: {summary['cache_hits']}")
    print(f"  Cache Misses: {summary['cache_misses']}")
    print(f"  Cache Hit Rate: {summary['cache_hit_rate']:.1f}%")

    print(f"\nEndpoint Performance:")
    print(f"{'Endpoint':<30} {'Count':<8} {'P50':<8} {'P95':<8} {'P99':<8} {'Max':<8} {'Errors':<8}")
    print("-" * 80)

    for endpoint, metrics in summary['endpoints'].items():
        print(f"{endpoint:<30} {metrics['count']:<8} "
              f"{metrics['p50']:<8.1f} {metrics['p95']:<8.1f} "
              f"{metrics['p99']:<8.1f} {metrics['max']:<8.1f} "
              f"{metrics['errors']:<8}")

    print("\n" + "=" * 80)

    # Determine pass/fail
    all_pass = True
    for endpoint, metrics in summary['endpoints'].items():
        error_rate = metrics['errors'] / metrics['count'] * 100 if metrics['count'] > 0 else 0
        if metrics['p95'] > 2000:  # General SLA
            all_pass = False
            print(f"⚠️  {endpoint}: p95={metrics['p95']:.0f}ms exceeds SLA")
        if error_rate > 2:
            all_pass = False
            print(f"⚠️  {endpoint}: error rate {error_rate:.1f}% exceeds SLA")

    if all_pass:
        print("✅ All endpoints within SLA")
    else:
        print("❌ Some endpoints exceeded SLA")

    print("=" * 80 + "\n")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Handle test quit event."""
    print("Load test quitting...")
