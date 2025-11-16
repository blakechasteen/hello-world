"""
Distributed Tracing Analysis Demo
==================================

Demonstrates complete distributed tracing workflow with:
- Multiple traced voice commands
- Trace tree visualization
- Latency analysis by span
- Bottleneck identification
- Error trace examples
- Cache performance analysis

Prerequisites:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger-thrift

    # Start Jaeger (in separate terminal)
    docker-compose -f docker-compose.tracing.yml up -d

Usage:
    PYTHONPATH=. python demos/demo_tracing_analysis.py

    Then open Jaeger UI:
    http://localhost:16686

Date: November 16, 2025
"""

import asyncio
import sys
import os
import time
from typing import List, Dict, Any

# Add HoloLoom to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from HoloLoom.voice.tracing import (
    TracingConfig,
    TracingManager,
    SamplingStrategy
)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.tree import Tree
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Install rich for better output: pip install rich")


# ============================================================================
# Mock VoiceAgent with Tracing
# ============================================================================

class MockVoiceAgent:
    """Mock voice agent with distributed tracing."""

    def __init__(self, tracing_manager: TracingManager):
        self.tracing = tracing_manager
        self.cache = {}

    @property
    def trace_voice_command(self):
        return self.tracing.trace_voice_command

    @property
    def trace_tts_synthesis(self):
        return self.tracing.trace_tts_synthesis

    @property
    def trace_cache_operation(self):
        return self.tracing.trace_cache_operation

    @property
    def trace_hololoom_weave(self):
        return self.tracing.trace_hololoom_weave

    @trace_voice_command()
    async def process_voice_input(self, transcript: str) -> Dict[str, Any]:
        """Process voice input with full tracing."""

        # Simulate query classification
        async with self.tracing.span("classify_query") as span:
            await asyncio.sleep(0.005)
            query_type = "question" if "?" in transcript else "command"
            if span:
                span.set_attribute("query.type", query_type)

        # Check cache
        cached_response = await self.get_cached_response(transcript)
        if cached_response:
            return {
                "transcript": transcript,
                "response": cached_response,
                "cache_hit": True,
                "processing_time_ms": 5
            }

        # Process with HoloLoom
        hololoom_result = await self.weave_query(transcript)

        # Generate audio response
        audio = await self.synthesize_response(hololoom_result["response"])

        # Cache result
        await self.cache_response(transcript, hololoom_result["response"])

        return {
            "transcript": transcript,
            "response": hololoom_result["response"],
            "confidence": hololoom_result["confidence"],
            "cache_hit": False,
            "processing_time_ms": hololoom_result["processing_time_ms"]
        }

    @trace_hololoom_weave()
    async def weave_query(self, query: str) -> Dict[str, Any]:
        """Simulate HoloLoom weaving with tracing."""

        class MockQuery:
            text = query

        start = time.time()

        # Simulate retrieval
        async with self.tracing.span("retrieval") as span:
            await asyncio.sleep(0.020)
            if span:
                span.set_attribute("retrieval.num_results", 5)

        # Simulate decision
        async with self.tracing.span("decision") as span:
            await asyncio.sleep(0.015)
            if span:
                span.set_attribute("decision.tool", "answer")

        # Simulate response generation
        async with self.tracing.span("generation") as span:
            await asyncio.sleep(0.025)
            response = f"Answer to: {query}"
            if span:
                span.set_attribute("generation.length", len(response))

        processing_time_ms = (time.time() - start) * 1000

        return {
            "response": response,
            "confidence": 0.92,
            "processing_time_ms": processing_time_ms
        }

    @trace_tts_synthesis()
    async def synthesize_response(self, text: str, voice: str = "nova") -> bytes:
        """Simulate TTS synthesis with tracing."""
        await asyncio.sleep(0.030)  # Simulate OpenAI TTS latency
        return text.encode()

    @trace_cache_operation("lookup")
    async def get_cached_response(self, key: str) -> str:
        """Get cached response."""
        await asyncio.sleep(0.001)
        return self.cache.get(key)

    @trace_cache_operation("store")
    async def cache_response(self, key: str, value: str):
        """Cache response."""
        await asyncio.sleep(0.001)
        self.cache[key] = value

    @trace_voice_command()
    async def process_with_error(self, transcript: str):
        """Simulate error for error trace demonstration."""
        async with self.tracing.span("processing"):
            await asyncio.sleep(0.010)
            raise ValueError(f"Simulated error processing: {transcript}")


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_basic_trace(agent: MockVoiceAgent):
    """Demo 1: Basic voice command trace."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Voice Command Trace")
    print("=" * 60)

    result = await agent.process_voice_input("What is the weather today?")

    if RICH_AVAILABLE:
        table = Table(title="Voice Command Result")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Transcript", result["transcript"])
        table.add_row("Response", result["response"])
        table.add_row("Cache Hit", str(result["cache_hit"]))
        table.add_row("Processing Time", f"{result['processing_time_ms']:.2f} ms")

        console.print(table)
    else:
        print(f"\nTranscript: {result['transcript']}")
        print(f"Response: {result['response']}")
        print(f"Cache Hit: {result['cache_hit']}")
        print(f"Processing Time: {result['processing_time_ms']:.2f} ms")

    print("\n✓ Check Jaeger UI for complete trace tree")


async def demo_cache_hit(agent: MockVoiceAgent):
    """Demo 2: Cache hit trace (much faster)."""
    print("\n" + "=" * 60)
    print("Demo 2: Cache Hit Performance")
    print("=" * 60)

    # First request (cache miss)
    start1 = time.time()
    result1 = await agent.process_voice_input("What is machine learning?")
    time1 = (time.time() - start1) * 1000

    # Second request (cache hit)
    start2 = time.time()
    result2 = await agent.process_voice_input("What is machine learning?")
    time2 = (time.time() - start2) * 1000

    if RICH_AVAILABLE:
        table = Table(title="Cache Performance Comparison")
        table.add_column("Request", style="cyan")
        table.add_column("Cache Hit", style="yellow")
        table.add_column("Time (ms)", style="green")

        table.add_row("First", str(result1["cache_hit"]), f"{time1:.2f}")
        table.add_row("Second", str(result2["cache_hit"]), f"{time2:.2f}")
        table.add_row("Speedup", "-", f"{time1/time2:.1f}x")

        console.print(table)
    else:
        print(f"\nFirst request (miss): {time1:.2f} ms")
        print(f"Second request (hit): {time2:.2f} ms")
        print(f"Speedup: {time1/time2:.1f}x")

    print("\n✓ Compare traces in Jaeger to see cache impact")


async def demo_error_trace(agent: MockVoiceAgent):
    """Demo 3: Error trace with exception recording."""
    print("\n" + "=" * 60)
    print("Demo 3: Error Trace")
    print("=" * 60)

    try:
        await agent.process_with_error("trigger error")
    except ValueError as e:
        if RICH_AVAILABLE:
            console.print(Panel(
                f"[red]Error recorded in trace:[/red]\n{e}",
                title="Error Trace"
            ))
        else:
            print(f"\nError recorded in trace: {e}")

    print("\n✓ Check Jaeger UI to see error span with exception details")


async def demo_concurrent_requests(agent: MockVoiceAgent):
    """Demo 4: Concurrent requests with trace correlation."""
    print("\n" + "=" * 60)
    print("Demo 4: Concurrent Requests")
    print("=" * 60)

    queries = [
        "What is Python?",
        "What is TypeScript?",
        "What is Rust?",
        "What is Go?",
        "What is Java?"
    ]

    start = time.time()

    # Process concurrently
    results = await asyncio.gather(*[
        agent.process_voice_input(q) for q in queries
    ])

    elapsed = time.time() - start

    if RICH_AVAILABLE:
        table = Table(title=f"Concurrent Processing ({len(queries)} requests)")
        table.add_column("Query", style="cyan")
        table.add_column("Processing Time (ms)", style="green")

        for q, r in zip(queries, results):
            table.add_row(q[:30], f"{r['processing_time_ms']:.2f}")

        table.add_row("", "")
        table.add_row("[bold]Total Elapsed[/bold]", f"[bold]{elapsed*1000:.2f}[/bold]")

        console.print(table)
    else:
        for q, r in zip(queries, results):
            print(f"{q[:30]}: {r['processing_time_ms']:.2f} ms")
        print(f"\nTotal elapsed: {elapsed*1000:.2f} ms")

    print("\n✓ Check Jaeger UI to see overlapping traces")


async def demo_latency_breakdown(agent: MockVoiceAgent):
    """Demo 5: Latency breakdown by component."""
    print("\n" + "=" * 60)
    print("Demo 5: Latency Breakdown")
    print("=" * 60)

    result = await agent.process_voice_input("Explain distributed tracing")

    # Simulate latency breakdown (in real traces, this comes from Jaeger)
    breakdown = {
        "Classification": 5,
        "Cache Lookup": 1,
        "HoloLoom Weaving": {
            "Retrieval": 20,
            "Decision": 15,
            "Generation": 25
        },
        "TTS Synthesis": 30,
        "Cache Store": 1
    }

    if RICH_AVAILABLE:
        tree = Tree("🔍 Latency Breakdown")

        for component, time_or_dict in breakdown.items():
            if isinstance(time_or_dict, dict):
                branch = tree.add(f"[cyan]{component}[/cyan]")
                for sub, sub_time in time_or_dict.items():
                    branch.add(f"[yellow]{sub}:[/yellow] [green]{sub_time} ms[/green]")
            else:
                tree.add(f"[cyan]{component}:[/cyan] [green]{time_or_dict} ms[/green]")

        console.print(tree)
    else:
        print("\nLatency Breakdown:")
        for component, time_or_dict in breakdown.items():
            if isinstance(time_or_dict, dict):
                print(f"  {component}:")
                for sub, sub_time in time_or_dict.items():
                    print(f"    {sub}: {sub_time} ms")
            else:
                print(f"  {component}: {time_or_dict} ms")

    print("\n✓ See actual breakdown in Jaeger's Timeline view")


async def demo_bottleneck_identification(agent: MockVoiceAgent):
    """Demo 6: Bottleneck identification."""
    print("\n" + "=" * 60)
    print("Demo 6: Bottleneck Identification")
    print("=" * 60)

    # Process 10 requests and analyze
    results = []
    for i in range(10):
        result = await agent.process_voice_input(f"Query {i}")
        results.append(result)
        await asyncio.sleep(0.1)  # Small delay between requests

    avg_processing = sum(r["processing_time_ms"] for r in results) / len(results)

    if RICH_AVAILABLE:
        panel = Panel(
            f"[cyan]Average Processing Time:[/cyan] [green]{avg_processing:.2f} ms[/green]\n\n"
            f"[yellow]Top Bottlenecks (from traces):[/yellow]\n"
            f"  1. TTS Synthesis: ~30 ms (31%)\n"
            f"  2. HoloLoom Generation: ~25 ms (26%)\n"
            f"  3. HoloLoom Retrieval: ~20 ms (21%)\n\n"
            f"[cyan]Recommendation:[/cyan] Cache TTS audio for 10x speedup on repeated phrases",
            title="Bottleneck Analysis"
        )
        console.print(panel)
    else:
        print(f"\nAverage Processing Time: {avg_processing:.2f} ms")
        print("\nTop Bottlenecks:")
        print("  1. TTS Synthesis: ~30 ms (31%)")
        print("  2. HoloLoom Generation: ~25 ms (26%)")
        print("  3. HoloLoom Retrieval: ~20 ms (21%)")
        print("\nRecommendation: Cache TTS audio for 10x speedup")

    print("\n✓ Use Jaeger's Service Performance view for real-time bottleneck detection")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all tracing demos."""

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold cyan]HoloLoom Distributed Tracing Demo[/bold cyan]\n\n"
            "[yellow]Prerequisites:[/yellow]\n"
            "  1. Start Jaeger: [green]docker-compose -f docker-compose.tracing.yml up -d[/green]\n"
            "  2. Open Jaeger UI: [green]http://localhost:16686[/green]\n\n"
            "[yellow]What to look for in Jaeger:[/yellow]\n"
            "  • Service: [cyan]hololoom-voice-agent[/cyan]\n"
            "  • Operations: [cyan]voice_command.*[/cyan], [cyan]tts.synthesis[/cyan], [cyan]cache.*[/cyan]\n"
            "  • Timeline view for latency breakdown\n"
            "  • Error traces with exception details\n"
            "  • Trace comparison for cache hit vs miss",
            title="Distributed Tracing Demo",
            border_style="cyan"
        ))
    else:
        print("\n" + "=" * 60)
        print("HoloLoom Distributed Tracing Demo")
        print("=" * 60)
        print("\nPrerequisites:")
        print("  1. Start Jaeger: docker-compose -f docker-compose.tracing.yml up -d")
        print("  2. Open Jaeger UI: http://localhost:16686")
        print("\nWhat to look for in Jaeger:")
        print("  • Service: hololoom-voice-agent")
        print("  • Operations: voice_command.*, tts.synthesis, cache.*")
        print("  • Timeline view for latency breakdown")
        print("  • Error traces with exception details")

    # Initialize tracing
    config = TracingConfig(
        enable_tracing=True,
        service_name="hololoom-voice-agent",
        jaeger_host="localhost",
        jaeger_port=6831,
        sampling_strategy=SamplingStrategy.ALWAYS_ON,
        sample_rate=1.0,
        verbose_spans=True
    )

    tracing_manager = TracingManager(config)

    # Create mock agent
    agent = MockVoiceAgent(tracing_manager)

    try:
        # Run demos
        await demo_basic_trace(agent)
        await asyncio.sleep(1)

        await demo_cache_hit(agent)
        await asyncio.sleep(1)

        await demo_error_trace(agent)
        await asyncio.sleep(1)

        await demo_concurrent_requests(agent)
        await asyncio.sleep(1)

        await demo_latency_breakdown(agent)
        await asyncio.sleep(1)

        await demo_bottleneck_identification(agent)

        # Final instructions
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold green]✓ Demo Complete![/bold green]\n\n"
                "[yellow]Next Steps:[/yellow]\n"
                "  1. Open Jaeger UI: [cyan]http://localhost:16686[/cyan]\n"
                "  2. Select service: [cyan]hololoom-voice-agent[/cyan]\n"
                "  3. Click [cyan]Find Traces[/cyan]\n"
                "  4. Explore traces, compare latencies, view errors\n\n"
                "[yellow]Key Features to Try:[/yellow]\n"
                "  • Compare cache hit vs miss traces\n"
                "  • View error trace with exception details\n"
                "  • Analyze latency in Timeline view\n"
                "  • Compare concurrent request traces\n"
                "  • Use Trace Graph for full request flow",
                title="Demo Complete",
                border_style="green"
            ))
        else:
            print("\n" + "=" * 60)
            print("✓ Demo Complete!")
            print("=" * 60)
            print("\nNext Steps:")
            print("  1. Open Jaeger UI: http://localhost:16686")
            print("  2. Select service: hololoom-voice-agent")
            print("  3. Click Find Traces")
            print("  4. Explore traces, compare latencies, view errors")

    finally:
        # Shutdown tracing (flush remaining spans)
        print("\nFlushing traces to Jaeger...")
        tracing_manager.shutdown(timeout_seconds=10)
        print("✓ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
