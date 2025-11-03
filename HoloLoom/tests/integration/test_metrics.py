"""
Test HoloLoom Prometheus Metrics
=================================
Starts metrics server and simulates some activity.
"""

import asyncio
import time
from HoloLoom.performance.prometheus_metrics import metrics, start_metrics_server

async def simulate_activity():
    """Simulate HoloLoom activity to generate metrics."""
    print("Simulating HoloLoom activity...")
    
    # Simulate queries
    for i in range(10):
        pattern = ['bare', 'fast', 'fused'][i % 3]
        complexity = ['lite', 'fast', 'full', 'research'][i % 4]
        duration = 0.1 + (i * 0.05)
        
        metrics.track_query(pattern, complexity, duration)
        print(f"  Query {i+1}: pattern={pattern}, complexity={complexity}, duration={duration:.2f}s")
        await asyncio.sleep(0.1)
    
    # Simulate cache activity
    print("\nSimulating cache activity...")
    for i in range(20):
        if i % 3 == 0:
            metrics.track_cache_hit()
            print("  Cache hit")
        else:
            metrics.track_cache_miss()
            print("  Cache miss")
    
    # Simulate breathing
    print("\nSimulating breathing cycles...")
    for i in range(5):
        metrics.track_breathing('inhale')
        await asyncio.sleep(0.2)
        metrics.track_breathing('exhale')
        await asyncio.sleep(0.1)
        metrics.track_breathing('rest')
        await asyncio.sleep(0.05)
        print(f"  Breath cycle {i+1} complete")
    
    # Set backend status
    print("\nSetting backend status...")
    metrics.set_backend_status('neo4j', True)
    metrics.set_backend_status('qdrant', True)
    print("  Backends marked healthy")
    
    print("\nActivity simulation complete!")

async def main():
    print("="*60)
    print("HoloLoom Prometheus Metrics Test")
    print("="*60)
    print()
    
    # Start metrics server
    print("Starting metrics server on http://localhost:8001/metrics...")
    start_metrics_server(port=8001)
    print("Metrics server started!")
    print()
    
    # Simulate activity
    await simulate_activity()
    
    print()
    print("="*60)
    print("Metrics endpoint: http://localhost:8001/metrics")
    print("Keep this running and check the endpoint in your browser")
    print("Or run: curl http://localhost:8001/metrics")
    print()
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    asyncio.run(main())
