"""
Endurance Test Scenario

Tests system stability under sustained load:
- 50 concurrent users for 1 hour (sustained production-like load)
- Validates no memory leaks or resource exhaustion
- Validates long-running stability
- Measures consistency over time

Key Observations:
- Monitor memory growth (should be flat after warmup)
- Monitor latency consistency (should not degrade over time)
- Monitor error rate consistency
- Validate cache hit rate stabilizes
- Check for garbage collection pauses
- Monitor connection pool health
- Validate no cascading slowdowns

Expected Results:
- Stable p95 latency: 200-400ms (no growth over time)
- Stable error rate: <0.5%
- Memory growth: <10% after 10 minute warmup
- Cache hit rate: 60-75% (stable)
- No exceptions or crashes
- Consistent throughput: ~20-30 RPS

1-Hour Timeline:
    0-10 min:   Warmup phase, caches cold
    10-55 min:  Steady state, all metrics should stabilize
    55-60 min:  Cooldown verification

Run Command:
    PYTHONPATH=/home/user/hello-world locust -f tests/load/locustfile.py \\
        --host=http://localhost:8000 \\
        --users 50 \\
        --spawn-rate 5 \\
        --run-time 1h \\
        --headless \\
        --csv=tests/load/results/endurance

Monitor With (in separate terminal):
    # CPU usage
    watch -n 1 'ps aux | grep hololoom'

    # Memory usage
    watch -n 1 'free -m'

    # Network connections
    watch -n 1 'netstat -an | grep ESTABLISHED | wc -l'

    # Application logs
    tail -f logs/app.log

Key Metrics to Track:
- Memory growth trajectory (should flatten)
- Latency percentiles over time (should remain stable)
- Error rate over time (should remain <0.5%)
- GC pause frequency and duration
- Thread pool usage
- Database connection pool health
"""
