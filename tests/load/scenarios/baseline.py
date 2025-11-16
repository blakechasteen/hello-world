"""
Baseline Load Test Scenario

Tests normal production load:
- 10 concurrent users (typical daily traffic)
- 5 minute duration (sufficient for warming and stabilization)
- Validates p95 < 300ms and error rate < 0.5%

This is the minimum recommended test before production deployment.

Expected Results:
- p95 latency: 150-300ms across all endpoints
- Error rate: <0.5%
- Cache hit rate: 50-70%
- Throughput: ~10-20 requests/second

Run Command:
    PYTHONPATH=/home/user/hello-world locust -f tests/load/locustfile.py \\
        --host=http://localhost:8000 \\
        --users 10 \\
        --spawn-rate 2 \\
        --run-time 5m \\
        --headless \\
        --csv=tests/load/results/baseline

Or with web UI:
    PYTHONPATH=/home/user/hello-world locust -f tests/load/locustfile.py \\
        --host=http://localhost:8000

Then in UI:
    Number of users: 10
    Spawn rate: 2 users/sec
    Click "Start swarming"
"""
