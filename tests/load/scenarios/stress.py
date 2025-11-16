"""
Stress Test Scenario

Tests system under ramping load:
- 10 → 100 users over 10 minutes (gradual increase)
- Validates performance degradation curves
- Identifies performance breaking points
- Validates auto-scaling triggers

Key Observations:
- Monitor when p95 crosses 500ms threshold
- Monitor when error rate exceeds 1%
- Measure CPU/memory usage curves
- Validate auto-scaling response time

Expected Results:
- p95 latency: 300-800ms (degrades gracefully)
- Error rate: <2% (acceptable degradation)
- Cache hit rate: 40-60% (reduced under load)
- Throughput: ~50-150 requests/second
- HPA should trigger around 70 users (70% CPU)

Run Command:
    PYTHONPATH=/home/user/hello-world locust -f tests/load/locustfile.py \\
        --host=http://localhost:8000 \\
        --users 100 \\
        --spawn-rate 10 \\
        --run-time 10m \\
        --headless \\
        --csv=tests/load/results/stress

Key Metrics to Track:
- When does p95 first exceed 500ms?
- When does error rate reach 1%?
- CPU utilization progression
- Memory growth over time
- Cache hit rate degradation
"""
