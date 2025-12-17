#!/usr/bin/env python3
"""
Terminal UI Demo
================
Demonstrates the rich terminal interface for HoloLoom weaving.

Shows:
- Real-time progress display
- Interactive pattern selection
- Conversation history
- Statistics tracking
"""

import asyncio
import sys
import os
import uuid

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.terminal_ui import TerminalUI
from HoloLoom.protocols.types import MemoryShard


def create_test_shards():
    """Create test memory shards for demos."""
    return [
        MemoryShard(
            id=str(uuid.uuid4()),
            text="Thompson Sampling is a Bayesian algorithm for the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions.",
            entities=["Thompson Sampling", "Bayesian", "multi-armed bandit", "exploration", "exploitation"],
            motifs=["algorithm", "probability"],
            timestamp=0.0,
            metadata={"source": "knowledge_base", "confidence": 0.95}
        ),
        MemoryShard(
            id=str(uuid.uuid4()),
            text="HoloLoom uses a 9-step weaving cycle: Loom Command, Chrono Trigger, Yarn Graph, Resonance Shed, Warp Space, Memory Crawl, Convergence, Tool Execution, Spacetime.",
            entities=["HoloLoom", "Loom Command", "Chrono Trigger", "Yarn Graph", "Resonance Shed", "Warp Space"],
            motifs=["weaving", "pipeline"],
            timestamp=0.0,
            metadata={"source": "documentation", "confidence": 0.9}
        ),
        MemoryShard(
            id=str(uuid.uuid4()),
            text="Neural networks are computational models inspired by biological neurons. They learn patterns through backpropagation and gradient descent optimization.",
            entities=["neural networks", "backpropagation", "gradient descent", "machine learning"],
            motifs=["AI", "learning"],
            timestamp=0.0,
            metadata={"source": "knowledge_base", "confidence": 0.9}
        ),
        MemoryShard(
            id=str(uuid.uuid4()),
            text="Machine learning is a subset of AI that enables systems to learn from data. It includes supervised, unsupervised, and reinforcement learning approaches.",
            entities=["machine learning", "AI", "supervised learning", "unsupervised learning", "reinforcement learning"],
            motifs=["AI", "learning"],
            timestamp=0.0,
            metadata={"source": "knowledge_base", "confidence": 0.9}
        ),
    ]


def create_config():
    """Create config with in-memory backend for fast startup."""
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY
    return config


async def demo_basic_queries():
    """Demo: Basic queries with different complexity levels"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Queries with Auto-Complexity")
    print("="*70)

    # Create orchestrator with auto-complexity detection
    config = create_config()
    orchestrator = WeavingOrchestrator(
        cfg=config,
        shards=create_test_shards(),
        enable_complexity_auto_detect=True
    )
    
    # Create UI
    ui = TerminalUI(orchestrator)
    ui.print_banner()
    
    # Test queries with different complexities
    queries = [
        "hi",  # LITE
        "what is machine learning?",  # FAST
        "explain how neural networks process information in detail",  # FULL
        "analyze comprehensive approaches to artificial intelligence systems",  # RESEARCH
    ]
    
    for query in queries:
        print(f"\n[Query]: {query}")
        await ui.weave_with_display(query, show_trace=False)
        await asyncio.sleep(0.5)  # Brief pause between queries
    
    # Show history and stats
    ui.show_history()
    ui.show_stats()


async def demo_pattern_selection():
    """Demo: Interactive pattern selection"""
    print("\n" + "="*70)
    print("DEMO 2: Pattern Selection")
    print("="*70)

    from HoloLoom.loom.command import PatternCard

    config = create_config()
    orchestrator = WeavingOrchestrator(cfg=config, shards=create_test_shards())
    ui = TerminalUI(orchestrator)
    
    ui.print_banner()
    
    # Show pattern menu (non-interactive for demo)
    ui.console.print("\n[bold]Pattern Card Options:[/]\n")
    
    patterns = [
        (PatternCard.BARE, "LITE: Fast greetings and simple commands"),
        (PatternCard.FAST, "FAST: Standard queries and questions"),
        (PatternCard.FUSED, "FULL: Detailed analysis and research"),
    ]
    
    for pattern, desc in patterns:
        ui.console.print(f"  • [cyan]{pattern.value.upper()}[/]: [dim]{desc}[/]")
    
    # Demo with FAST pattern
    await ui.weave_with_display(
        "what is HoloLoom?",
        pattern=PatternCard.FAST,
        show_trace=False
    )


async def demo_trace_display():
    """Demo: Detailed trace visualization"""
    print("\n" + "="*70)
    print("DEMO 3: Provenance Trace Display")
    print("="*70)

    config = create_config()
    orchestrator = WeavingOrchestrator(
        cfg=config,
        shards=create_test_shards(),
        enable_complexity_auto_detect=True
    )
    ui = TerminalUI(orchestrator)
    
    ui.print_banner()
    
    # Execute with trace display
    result = await ui.weave_with_display(
        "analyze neural decision systems",
        show_trace=True  # Show detailed trace
    )


async def demo_conversation_flow():
    """Demo: Multi-turn conversation with history"""
    print("\n" + "="*70)
    print("DEMO 4: Conversation Flow")
    print("="*70)

    config = create_config()
    orchestrator = WeavingOrchestrator(
        cfg=config,
        shards=create_test_shards(),
        enable_complexity_auto_detect=True
    )
    ui = TerminalUI(orchestrator)
    
    ui.print_banner()
    
    # Simulate a conversation
    conversation = [
        "hi",
        "what is a neural network?",
        "how does it learn?",
        "explain backpropagation",
        "compare to human learning",
    ]
    
    for query in conversation:
        ui.console.print(f"\n[bold cyan]User:[/] {query}")
        await ui.weave_with_display(query, show_trace=False)
        await asyncio.sleep(0.3)
    
    # Show full history
    ui.console.print("\n" + "="*70)
    ui.show_history(limit=10)
    ui.show_stats()


async def main():
    """Run all demos"""
    demos = [
        ("Basic Queries", demo_basic_queries),
        ("Pattern Selection", demo_pattern_selection),
        ("Trace Display", demo_trace_display),
        ("Conversation Flow", demo_conversation_flow),
    ]
    
    print("\n" + "="*70)
    print("HOLOLOOM TERMINAL UI DEMOS")
    print("="*70)
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  5. Run all demos")
    print("  6. Interactive session")
    
    try:
        choice = input("\nSelect demo (1-6): ").strip()
        
        if choice == "6":
            # Interactive session
            config = create_config()
            orchestrator = WeavingOrchestrator(
                cfg=config,
                shards=create_test_shards(),
                enable_complexity_auto_detect=True
            )
            ui = TerminalUI(orchestrator)
            await ui.interactive_session()
        elif choice == "5":
            # Run all
            for name, demo_func in demos:
                await demo_func()
                print("\n[Press Enter to continue...]")
                input()
        elif choice in ["1", "2", "3", "4"]:
            # Run specific demo
            idx = int(choice) - 1
            await demos[idx][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
