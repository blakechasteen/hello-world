"""
Learned Routing Demo

Demonstrates intelligent routing with Thompson Sampling and A/B testing.
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.routing.integration import RoutingOrchestrator, classify_query_type
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress


console = Console()


def demo_query_classification():
    """Demo 1: Query classification."""
    console.print("\n[bold cyan]Demo 1: Query Classification[/bold cyan]\n")
    
    test_queries = [
        "What is Python?",
        "Who invented the internet?",
        "Why does gravity work?",
        "How do I build a web app?",
        "Compare Python and JavaScript",
        "Imagine a world without computers",
        "Create a story about AI",
        "Hello, how are you?",
        "Thanks for your help!",
    ]
    
    table = Table(title="Query Classification")
    table.add_column("Query", style="cyan")
    table.add_column("Type", style="green")
    
    for query in test_queries:
        query_type = classify_query_type(query)
        table.add_row(query[:50], query_type)
    
    console.print(table)


def demo_thompson_sampling():
    """Demo 2: Thompson Sampling learning."""
    console.print("\n[bold cyan]Demo 2: Thompson Sampling Learning[/bold cyan]\n")
    
    from HoloLoom.routing import ThompsonBandit
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    bandit = ThompsonBandit(backends=backends)
    
    console.print("Initial state (all backends equal):")
    stats = bandit.get_stats()
    table = Table()
    table.add_column("Backend", style="cyan")
    table.add_column("Success Rate", style="green")
    
    for backend, backend_stats in stats.items():
        table.add_row(backend, f"{backend_stats['expected_success_rate']:.3f}")
    
    console.print(table)
    
    # Simulate learning - HYBRID performs best
    console.print("\n[yellow]Simulating 20 queries...[/yellow]")
    
    outcomes = []
    for i in range(20):
        backend = bandit.select()
        
        # HYBRID: 90% success, NEO4J_QDRANT: 70% success, NETWORKX: 40% success
        import random
        if backend == 'HYBRID':
            success = random.random() < 0.9
        elif backend == 'NEO4J_QDRANT':
            success = random.random() < 0.7
        else:
            success = random.random() < 0.4
        
        bandit.update(backend, success)
        outcomes.append((backend, success))
    
    console.print("\nLearned preferences:")
    stats = bandit.get_stats()
    table = Table()
    table.add_column("Backend", style="cyan")
    table.add_column("Success Rate", style="green")
    table.add_column("Observations", style="yellow")
    
    # Sort by success rate
    sorted_backends = sorted(
        stats.items(),
        key=lambda x: x[1]['expected_success_rate'],
        reverse=True
    )
    
    for backend, backend_stats in sorted_backends:
        table.add_row(
            backend,
            f"{backend_stats['expected_success_rate']:.3f}",
            str(backend_stats['total_observations'])
        )
    
    console.print(table)
    
    # Show selection distribution
    selections = {}
    for _ in range(100):
        backend = bandit.select()
        selections[backend] = selections.get(backend, 0) + 1
    
    console.print("\nSelection distribution (100 samples):")
    table = Table()
    table.add_column("Backend", style="cyan")
    table.add_column("Selections", style="green")
    
    for backend, count in sorted(selections.items(), key=lambda x: -x[1]):
        table.add_row(backend, f"{count}%")
    
    console.print(table)


def demo_per_query_type_learning():
    """Demo 3: Per-query-type specialized learning."""
    console.print("\n[bold cyan]Demo 3: Per-Query-Type Learning[/bold cyan]\n")
    
    from HoloLoom.routing import LearnedRouter
    import random
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    query_types = ['factual', 'analytical', 'creative', 'conversational']
    
    router = LearnedRouter(
        backends=backends,
        query_types=query_types
    )
    
    console.print("[yellow]Simulating query-specific learning...[/yellow]\n")
    
    # Simulate specialized performance:
    # - Factual: NETWORKX best (fast lookups)
    # - Analytical: HYBRID best (deep analysis)
    # - Creative: NEO4J_QDRANT best (graph exploration)
    # - Conversational: Any works fine
    
    for _ in range(30):
        query_type = random.choice(query_types)
        backend = router.select_backend(query_type)
        
        # Simulate success based on optimal matching
        if query_type == 'factual' and backend == 'NETWORKX':
            success = random.random() < 0.9
        elif query_type == 'analytical' and backend == 'HYBRID':
            success = random.random() < 0.9
        elif query_type == 'creative' and backend == 'NEO4J_QDRANT':
            success = random.random() < 0.9
        else:
            success = random.random() < 0.5
        
        router.update(query_type, backend, success)
    
    # Show learned preferences
    stats = router.get_stats()
    
    for query_type in ['factual', 'analytical', 'creative']:
        console.print(f"\n[bold]{query_type.capitalize()} Queries:[/bold]")
        
        table = Table()
        table.add_column("Backend", style="cyan")
        table.add_column("Success Rate", style="green")
        table.add_column("Observations", style="yellow")
        
        backend_stats = stats[query_type]
        sorted_backends = sorted(
            backend_stats.items(),
            key=lambda x: x[1]['expected_success_rate'],
            reverse=True
        )
        
        for backend, data in sorted_backends:
            if data['total_observations'] > 0:
                table.add_row(
                    backend,
                    f"{data['expected_success_rate']:.3f}",
                    str(data['total_observations'])
                )
        
        console.print(table)


def demo_ab_testing():
    """Demo 4: A/B testing learned vs rule-based."""
    console.print("\n[bold cyan]Demo 4: A/B Testing[/bold cyan]\n")
    
    from HoloLoom.routing import ABTestRouter, StrategyVariant
    from HoloLoom.routing.integration import rule_based_routing
    import random
    
    # Create temporary storage
    storage_path = Path(__file__).parent / 'demo_ab_results.json'
    
    # Define strategies
    def learned_strategy(query_type: str, complexity: str) -> str:
        """Simulated learned strategy."""
        # Learned strategy picks better backends
        if query_type == 'analytical':
            return 'HYBRID'
        elif query_type == 'factual':
            return 'NETWORKX'
        else:
            return 'NEO4J_QDRANT'
    
    variants = [
        StrategyVariant(
            name='learned',
            weight=0.5,
            strategy_fn=learned_strategy
        ),
        StrategyVariant(
            name='rule_based',
            weight=0.5,
            strategy_fn=rule_based_routing
        )
    ]
    
    router = ABTestRouter(variants=variants, storage_path=storage_path)
    
    console.print("[yellow]Running A/B test with 50 queries...[/yellow]\n")
    
    # Simulate queries
    query_types = ['factual', 'analytical', 'creative']
    complexities = ['LITE', 'FAST', 'FULL']
    
    for _ in range(50):
        query_type = random.choice(query_types)
        complexity = random.choice(complexities)
        
        backend, variant = router.route(query_type, complexity)
        
        # Learned strategy performs better
        if variant == 'learned':
            latency = random.uniform(80, 120)
            relevance = random.uniform(0.8, 0.95)
            success = random.random() < 0.95
        else:
            latency = random.uniform(120, 180)
            relevance = random.uniform(0.6, 0.8)
            success = random.random() < 0.85
        
        router.record_outcome(variant, latency, relevance, success)
    
    # Show results
    results = router.get_results()
    
    table = Table(title="A/B Test Results")
    table.add_column("Strategy", style="cyan")
    table.add_column("Queries", style="yellow")
    table.add_column("Avg Latency", style="green")
    table.add_column("Avg Relevance", style="green")
    table.add_column("Success Rate", style="green")
    
    for variant_name, metrics in results.items():
        table.add_row(
            variant_name,
            str(metrics['total_queries']),
            f"{metrics['avg_latency_ms']:.1f}ms",
            f"{metrics['avg_relevance']:.2f}",
            f"{metrics['success_rate']:.2%}"
        )
    
    console.print(table)
    
    winner = router.get_winner()
    console.print(f"\n[bold green]Winner: {winner}[/bold green]")
    
    # Cleanup
    if storage_path.exists():
        storage_path.unlink()


def demo_full_orchestrator():
    """Demo 5: Full routing orchestrator."""
    console.print("\n[bold cyan]Demo 5: Full Routing Orchestrator[/bold cyan]\n")
    
    # Create temporary storage
    storage_dir = Path(__file__).parent / 'demo_routing'
    storage_dir.mkdir(exist_ok=True)
    
    backends = ['HYBRID', 'NEO4J_QDRANT', 'NETWORKX']
    query_types = ['factual', 'analytical', 'creative', 'conversational']
    
    orchestrator = RoutingOrchestrator(
        backends=backends,
        query_types=query_types,
        enable_ab_test=True,
        storage_dir=storage_dir
    )
    
    console.print("[yellow]Processing queries with learned routing...[/yellow]\n")
    
    # Sample queries
    queries = [
        ("What is HoloLoom?", "LITE"),
        ("Who created Python?", "LITE"),
        ("Why does Thompson Sampling work?", "FAST"),
        ("How do I optimize performance?", "FULL"),
        ("Compare MCTS and Thompson Sampling", "FULL"),
        ("Imagine a future with AGI", "FAST"),
        ("Create a narrative about AI", "FULL"),
        ("Hello! Tell me about routing", "LITE"),
    ]
    
    table = Table(title="Query Routing")
    table.add_column("Query", style="cyan", width=35)
    table.add_column("Type", style="yellow")
    table.add_column("Backend", style="green")
    table.add_column("Strategy", style="magenta")
    
    import random
    for query, complexity in queries:
        # Select backend
        backend, strategy = orchestrator.select_backend(query, complexity)
        
        query_type = classify_query_type(query)
        table.add_row(query[:35], query_type, backend, strategy)
        
        # Simulate outcome
        latency = random.uniform(50, 150)
        relevance = random.uniform(0.7, 0.95)
        confidence = random.uniform(0.7, 0.9)
        success = random.random() < 0.9
        
        orchestrator.record_outcome(
            query=query,
            complexity=complexity,
            backend_selected=backend,
            strategy_used=strategy,
            latency_ms=latency,
            relevance_score=relevance,
            confidence=confidence,
            success=success,
            memory_size=25
        )
    
    console.print(table)
    
    # Show statistics
    stats = orchestrator.get_routing_stats()
    
    console.print(f"\n[bold]Total Queries:[/bold] {stats['total_queries']}")
    
    if 'ab_test' in stats:
        console.print("\n[bold]A/B Test Status:[/bold]")
        table = Table()
        table.add_column("Strategy", style="cyan")
        table.add_column("Queries", style="yellow")
        
        for variant, metrics in stats['ab_test']['results'].items():
            table.add_row(variant, str(metrics['total_queries']))
        
        console.print(table)
    
    # Cleanup
    import shutil
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold cyan]Learned Routing System Demo[/bold cyan]\n\n"
        "Demonstrates Thompson Sampling, per-query-type learning,\n"
        "A/B testing, and full orchestrator integration.",
        border_style="cyan"
    ))
    
    demos = [
        ("Query Classification", demo_query_classification),
        ("Thompson Sampling", demo_thompson_sampling),
        ("Per-Query-Type Learning", demo_per_query_type_learning),
        ("A/B Testing", demo_ab_testing),
        ("Full Orchestrator", demo_full_orchestrator),
    ]
    
    for i, (name, demo_fn) in enumerate(demos, 1):
        console.print(f"\n[bold white]Running Demo {i}/{len(demos)}...[/bold white]")
        demo_fn()
        
        if i < len(demos):
            input("\n[Press Enter to continue...]")
    
    console.print("\n[bold green]All demos complete![/bold green]")


if __name__ == '__main__':
    main()
