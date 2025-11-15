"""
Demo: ML-Based Anomaly Detection for HoloLoom Security

Demonstrates the complete anomaly detection pipeline:
1. Generate synthetic normal and anomalous events
2. Train baseline behavior model
3. Detect anomalies using Isolation Forest
4. Generate explanations for flagged events
5. Show performance metrics

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import asyncio
from datetime import datetime, timedelta
import numpy as np

from HoloLoom.security.anomaly import (
    AnomalyDetector,
    SecurityEvent,
    create_anomaly_detector
)


# ============================================================================
# Generate Synthetic Data
# ============================================================================

def generate_normal_events(num_days: int = 7) -> list:
    """
    Generate synthetic normal user behavior.

    Simulates 3 users with typical work patterns:
    - alice: 9-17, Mon-Fri, from US
    - bob: 20-23, Mon-Sun (night shift), from UK
    - charlie: 10-18, Mon-Fri, from CA
    """
    events = []
    base_time = datetime(2025, 11, 1, 0, 0, 0).timestamp()

    # Alice: Day shift, US, regular hours
    for day in range(num_days):
        if day % 7 < 5:  # Mon-Fri only
            for hour in range(9, 17):
                timestamp = base_time + (day * 86400) + (hour * 3600) + np.random.randint(0, 3600)
                events.append(SecurityEvent(
                    user_id="alice",
                    timestamp=timestamp,
                    event_type="query",
                    ip_address="192.168.1.100",
                    geographic_location="US",
                    endpoint="/api/search",
                    query_complexity=np.random.uniform(0.3, 0.6),
                    request_size=np.random.randint(512, 2048),
                    authentication_method="password"
                ))

    # Bob: Night shift, UK, works weekends
    for day in range(num_days):
        for hour in range(20, 23):
            timestamp = base_time + (day * 86400) + (hour * 3600) + np.random.randint(0, 3600)
            events.append(SecurityEvent(
                user_id="bob",
                timestamp=timestamp,
                event_type="query",
                ip_address="192.168.1.101",
                geographic_location="UK",
                endpoint="/api/data",
                query_complexity=np.random.uniform(0.2, 0.4),
                request_size=np.random.randint(256, 1024),
                authentication_method="mfa"
            ))

    # Charlie: Regular hours, CA
    for day in range(num_days):
        if day % 7 < 5:  # Mon-Fri only
            for hour in range(10, 18):
                timestamp = base_time + (day * 86400) + (hour * 3600) + np.random.randint(0, 3600)
                events.append(SecurityEvent(
                    user_id="charlie",
                    timestamp=timestamp,
                    event_type="api_call",
                    ip_address="192.168.1.102",
                    geographic_location="CA",
                    endpoint="/api/analytics",
                    query_complexity=np.random.uniform(0.4, 0.7),
                    request_size=np.random.randint(1024, 4096),
                    authentication_method="oauth"
                ))

    return events


def generate_anomalous_events() -> list:
    """
    Generate synthetic anomalous events for testing.

    Anomaly types:
    1. Unusual hour (3 AM access)
    2. Unusual location (Russia instead of US)
    3. Unusual endpoint (admin access)
    4. High query complexity
    5. Failed authentication attempts
    """
    base_time = datetime(2025, 11, 8, 3, 0, 0).timestamp()

    events = [
        # 1. Alice at 3 AM (temporal anomaly)
        SecurityEvent(
            user_id="alice",
            timestamp=base_time,
            event_type="query",
            ip_address="192.168.1.100",
            geographic_location="US",
            endpoint="/api/search",
            query_complexity=0.5
        ),

        # 2. Alice from Russia (geographic anomaly)
        SecurityEvent(
            user_id="alice",
            timestamp=base_time + 3600,
            event_type="query",
            ip_address="5.6.7.8",
            geographic_location="RU",
            endpoint="/api/search",
            query_complexity=0.5
        ),

        # 3. Alice accessing admin endpoint (access pattern anomaly)
        SecurityEvent(
            user_id="alice",
            timestamp=base_time + 7200,
            event_type="api_call",
            ip_address="192.168.1.100",
            geographic_location="US",
            endpoint="/api/admin/delete_all",
            query_complexity=0.9
        ),

        # 4. Bob with unusually high query complexity (behavioral anomaly)
        SecurityEvent(
            user_id="bob",
            timestamp=base_time,
            event_type="query",
            ip_address="192.168.1.101",
            geographic_location="UK",
            endpoint="/api/data",
            query_complexity=0.98
        ),

        # 5. Charlie accessing at midnight (temporal anomaly)
        SecurityEvent(
            user_id="charlie",
            timestamp=datetime(2025, 11, 8, 0, 30, 0).timestamp(),
            event_type="api_call",
            ip_address="192.168.1.102",
            geographic_location="CA",
            endpoint="/api/analytics",
            query_complexity=0.6
        ),

        # 6. Bob from China (geographic anomaly)
        SecurityEvent(
            user_id="bob",
            timestamp=base_time + 10800,
            event_type="query",
            ip_address="1.2.3.4",
            geographic_location="CN",
            endpoint="/api/data",
            query_complexity=0.3
        ),
    ]

    return events


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run complete anomaly detection demo."""
    print("=" * 80)
    print("HoloLoom Anomaly Detection Demo")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Generate Data
    # ========================================================================
    print("Step 1: Generating synthetic data...")
    print("-" * 80)

    normal_events = generate_normal_events(num_days=7)
    anomalous_events = generate_anomalous_events()

    print(f"✓ Generated {len(normal_events)} normal events (7 days)")
    print(f"✓ Generated {len(anomalous_events)} anomalous events")
    print()

    # ========================================================================
    # Step 2: Create and Train Detector
    # ========================================================================
    print("Step 2: Training anomaly detector...")
    print("-" * 80)

    # Create detector (Isolation Forest only for demo)
    detector = create_anomaly_detector(enable_all=False)

    # Train on normal behavior
    await detector.fit_baseline(normal_events, per_user=True)

    stats = detector.get_statistics()
    print(f"✓ Detector trained on {len(normal_events)} events")
    print(f"✓ Baseline window: {stats['baseline_window_days']} days")
    print(f"✓ Enabled detectors: {', '.join(k for k, v in stats['enabled_detectors'].items() if v)}")
    print()

    # ========================================================================
    # Step 3: Test on Normal Events
    # ========================================================================
    print("Step 3: Testing on normal events...")
    print("-" * 80)

    normal_results = await detector.detect_batch(normal_events[:20])
    normal_scores = [r.score for r in normal_results]
    avg_normal_score = np.mean(normal_scores)
    max_normal_score = np.max(normal_scores)

    print(f"✓ Tested on {len(normal_results)} normal events")
    print(f"  Avg anomaly score: {avg_normal_score:.3f}")
    print(f"  Max anomaly score: {max_normal_score:.3f}")
    print(f"  False positives: {sum(1 for r in normal_results if r.is_anomalous)}/{len(normal_results)}")
    print()

    # ========================================================================
    # Step 4: Detect Anomalies
    # ========================================================================
    print("Step 4: Detecting anomalies...")
    print("-" * 80)

    results = await detector.detect_batch(anomalous_events)

    print(f"✓ Analyzed {len(results)} potentially anomalous events")
    print()

    # Show detailed results
    for i, result in enumerate(results, 1):
        event = result.event
        print(f"Event {i}: {event.user_id} - {event.event_type}")
        print(f"  Timestamp: {datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Location: {event.geographic_location}")
        print(f"  Endpoint: {event.endpoint}")
        print(f"  Anomaly Score: {result.score:.3f}")
        print(f"  Severity: {result.severity.value.upper()}")
        print(f"  Anomalous: {'✓ YES' if result.is_anomalous else '✗ NO'}")

        if result.is_anomalous:
            print(f"  Type: {result.anomaly_type.value}")
            print(f"  Explanation: {result.explanation}")

            # Recommended actions
            actions = []
            if result.should_alert:
                actions.append("Alert security team")
            if result.should_require_mfa:
                actions.append("Require MFA re-auth")
            if result.should_throttle:
                actions.append("Throttle requests")
            if result.should_block:
                actions.append("Block account temporarily")

            if actions:
                print(f"  Actions: {', '.join(actions)}")

        print()

    # ========================================================================
    # Step 5: Summary Statistics
    # ========================================================================
    print("Step 5: Summary Statistics")
    print("-" * 80)

    anomalies_detected = sum(1 for r in results if r.is_anomalous)
    avg_anomalous_score = np.mean([r.score for r in results if r.is_anomalous]) if anomalies_detected > 0 else 0

    print(f"Anomalies detected: {anomalies_detected}/{len(results)} ({anomalies_detected/len(results)*100:.1f}%)")
    print(f"Avg anomaly score: {avg_anomalous_score:.3f}")
    print()

    # Severity breakdown
    severity_counts = {}
    for result in results:
        if result.is_anomalous:
            severity_counts[result.severity.value] = severity_counts.get(result.severity.value, 0) + 1

    if severity_counts:
        print("Severity breakdown:")
        for severity, count in sorted(severity_counts.items()):
            print(f"  {severity.upper()}: {count}")
        print()

    # Anomaly type breakdown
    type_counts = {}
    for result in results:
        if result.is_anomalous:
            type_counts[result.anomaly_type.value] = type_counts.get(result.anomaly_type.value, 0) + 1

    if type_counts:
        print("Anomaly type breakdown:")
        for atype, count in sorted(type_counts.items()):
            print(f"  {atype}: {count}")
        print()

    # ========================================================================
    # Step 6: Performance Metrics
    # ========================================================================
    print("Step 6: Performance Metrics")
    print("-" * 80)

    import time

    # Measure latency
    start = time.time()
    for _ in range(100):
        await detector.detect(normal_events[0])
    elapsed = time.time() - start

    avg_latency_ms = (elapsed / 100) * 1000

    print(f"Avg detection latency: {avg_latency_ms:.2f}ms")
    print(f"Throughput: {int(1000 / avg_latency_ms)} detections/second")
    print()

    # ========================================================================
    # Done
    # ========================================================================
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
