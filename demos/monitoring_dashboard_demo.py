"""
Monitoring Dashboard Demo
========================

Demonstrates the HoloLoom monitoring system with:
- Real-time metrics collection
- Pattern distribution tracking
- Backend usage statistics
- Live dashboard visualization

Run with:
    $env:PYTHONPATH = "."; python demos/monitoring_dashboard_demo.py
"""

import asyncio
import random
import time
from HoloLoom.monitoring import MonitoringDashboard, MetricsCollector


async def simulate_queries(collector: MetricsCollector, num_queries: int = 50):
    """Simulate various queries with different patterns and outcomes."""
    
    patterns = ['bare', 'fast', 'fused', 'research']
    backends = ['NETWORKX', 'NEO4J_QDRANT', 'HYPERSPACE']
    tools = ['search', 'analyze', 'summarize', 'extract']
    complexity_levels = ['LITE', 'FAST', 'FULL', 'RESEARCH']
    
    print(f"\nSimulating {num_queries} queries...")
    
    for i in range(num_queries):
        # Randomly select pattern with realistic distribution
        # More common: fast/bare, less common: research
        pattern_weights = [0.2, 0.4, 0.3, 0.1]
        pattern = random.choices(patterns, weights=pattern_weights)[0]
        
        # Backend selection based on pattern
        if pattern == 'research':
            backend = 'HYPERSPACE'
        elif pattern == 'fused':
            backend = random.choice(['NEO4J_QDRANT', 'HYPERSPACE'])
        else:
            backend = random.choice(backends)
        
        # Latency varies by pattern
        latency_ranges = {
            'bare': (30, 80),
            'fast': (100, 200),
            'fused': (250, 400),
            'research': (400, 800)
        }
        latency = random.uniform(*latency_ranges[pattern])
        
        # Success rate varies by complexity
        success_rate = 0.95 if pattern != 'research' else 0.85
        success = random.random() < success_rate
        
        tool = random.choice(tools)
        complexity = complexity_levels[patterns.index(pattern)]
        memory_hits = random.randint(5, 50) if backend != 'NETWORKX' else 0
        
        collector.record_query(
            pattern=pattern,
            latency_ms=latency,
            success=success,
            backend=backend,
            tool=tool,
            complexity_level=complexity,
            memory_hits=memory_hits
        )
        
        # Small delay to simulate real-world timing
        await asyncio.sleep(0.05)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_queries} queries...")
    
    print("✅ Query simulation complete!")


async def demo_static_display():
    """Demo: Static dashboard display after query simulation."""
    print("\n" + "="*60)
    print("DEMO 1: Static Dashboard Display")
    print("="*60)
    
    collector = MetricsCollector()
    dashboard = MonitoringDashboard(collector)
    
    # Simulate queries
    await simulate_queries(collector, num_queries=30)
    
    # Display dashboard
    print("\nDisplaying dashboard...")
    time.sleep(1)
    dashboard.display()
    
    # Show summary
    summary = collector.get_summary()
    print(f"\n📊 Summary:")
    print(f"  Total Queries: {summary['total_queries']}")
    print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.1f}ms")


async def demo_live_dashboard():
    """Demo: Live-updating dashboard during query simulation."""
    print("\n" + "="*60)
    print("DEMO 2: Live Dashboard (press Ctrl+C to stop)")
    print("="*60)
    
    collector = MetricsCollector()
    dashboard = MonitoringDashboard(collector)
    
    # Start background task to simulate queries
    async def continuous_queries():
        patterns = ['bare', 'fast', 'fused', 'research']
        backends = ['NETWORKX', 'NEO4J_QDRANT', 'HYPERSPACE']
        tools = ['search', 'analyze', 'summarize', 'extract']
        complexity_levels = ['LITE', 'FAST', 'FULL', 'RESEARCH']
        
        while True:
            pattern = random.choice(patterns)
            backend = random.choice(backends)
            latency = random.uniform(50, 400)
            success = random.random() < 0.92
            tool = random.choice(tools)
            complexity = random.choice(complexity_levels)
            
            collector.record_query(
                pattern=pattern,
                latency_ms=latency,
                success=success,
                backend=backend,
                tool=tool,
                complexity_level=complexity,
                memory_hits=random.randint(0, 30)
            )
            
            await asyncio.sleep(random.uniform(0.2, 0.8))
    
    # Run live dashboard with query simulation
    query_task = asyncio.create_task(continuous_queries())
    
    try:
        dashboard.display_live(refresh_rate=2.0)
    except KeyboardInterrupt:
        print("\n\n✅ Live demo stopped by user")
    finally:
        query_task.cancel()


async def demo_integration_example():
    """Demo: How to integrate monitoring into WeavingOrchestrator."""
    print("\n" + "="*60)
    print("DEMO 3: Integration Example")
    print("="*60)
    
    print("\nIntegration pattern for WeavingOrchestrator:\n")
    
    integration_code = '''
# In weaving_orchestrator.py __init__:
from HoloLoom.monitoring import get_global_collector

class WeavingOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.metrics_collector = get_global_collector() if config.enable_monitoring else None
        ...

# In weaving_orchestrator.py weave() method:
async def weave(self, query: Query) -> Spacetime:
    start_time = time.perf_counter()
    
    try:
        # ... existing weaving logic ...
        result = await self._execute_weaving(query)
        
        # Track successful query
        if self.metrics_collector:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics_collector.record_query(
                pattern=selected_pattern,
                latency_ms=latency_ms,
                success=True,
                backend=self.config.memory_backend.value,
                complexity_level=complexity_level.name
            )
        
        return result
        
    except Exception as e:
        # Track failed query
        if self.metrics_collector:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics_collector.record_query(
                pattern=selected_pattern,
                latency_ms=latency_ms,
                success=False,
                backend=self.config.memory_backend.value
            )
        raise

# View dashboard in separate script or CLI command:
from HoloLoom.monitoring import MonitoringDashboard, get_global_collector

collector = get_global_collector()
dashboard = MonitoringDashboard(collector)
dashboard.display()  # Static display
# or
dashboard.display_live()  # Live updates
'''
    
    print(integration_code)
    print("\n✅ Integration example shown")


async def main():
    """Run all monitoring demos."""
    print("""
╔════════════════════════════════════════════════════════════╗
║         HoloLoom Monitoring Dashboard Demo                 ║
║                                                              ║
║  Features:                                                   ║
║    • Real-time metrics collection                           ║
║    • Pattern distribution tracking                          ║
║    • Backend usage statistics                               ║
║    • Tool usage analytics                                   ║
║    • Complexity level distribution                          ║
║    • Live dashboard with auto-refresh                       ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Demo 1: Static display
    await demo_static_display()
    
    print("\n" + "-"*60)
    input("\nPress Enter to continue to live dashboard demo...")
    
    # Demo 2: Live dashboard
    await demo_live_dashboard()
    
    # Demo 3: Integration example
    await demo_integration_example()
    
    print("\n" + "="*60)
    print("✅ All monitoring demos complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Install rich library: pip install rich")
    print("  2. Add monitoring to WeavingOrchestrator (see Demo 3)")
    print("  3. Enable monitoring in Config: config.enable_monitoring = True")
    print("  4. View metrics: MonitoringDashboard(get_global_collector()).display()")


if __name__ == '__main__':
    asyncio.run(main())
