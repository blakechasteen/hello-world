"""
Spike Test Scenario

Tests system response to sudden traffic surge:
- 0 → 200 users in 1-2 minutes (sudden spike)
- Tests queue management and backpressure
- Validates auto-scaling response speed
- Simulates unexpected viral events or incidents

Key Observations:
- Measure time to degrade acceptably (>2s p95)
- Measure queue depth and growth rate
- Measure auto-scaling response time (should be <2 min)
- Validate graceful degradation (no crashes)
- Measure recovery time when load subsides

Expected Results:
- Initial latency spike: p95 could reach 1-2 seconds
- Auto-scaling triggered within 30-60 seconds
- Error rate: <5% during spike (some queueing acceptable)
- Recovery time: <5 minutes after spike subsides
- No cascading failures or crashes

Spike Profile:
    Time    Users   Expected p95
    0s      0       5ms
    10s     100     150ms
    20s     200     500-1000ms (peak spike)
    120s    200     500-800ms (sustained if scaling not working)
    140s    150     300ms (ramp down)
    160s    50      150ms (near baseline)

Run Command:
    PYTHONPATH=/home/user/hello-world locust -f tests/load/locustfile.py \\
        --host=http://localhost:8000 \\
        --users 200 \\
        --spawn-rate 50 \\
        --run-time 2m \\
        --headless \\
        --csv=tests/load/results/spike

Key Metrics to Track:
- Time to reach 1000ms p95
- Queue depth during spike peak
- Auto-scaling lag (time from 200 users to new instances ready)
- Memory usage spike
- Error rate during spike
- Recovery trajectory
"""
