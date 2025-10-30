#!/usr/bin/env python3
"""
Terminal UI Integration for HoloLoom
====================================
Beautiful rich terminal interface for WeavingOrchestrator.

Features:
- Real-time weaving trace visualization
- Interactive pattern card selection
- Conversation history with rich formatting
- Live progress tracking for each pipeline stage
- Syntax-highlighted output

Usage:
    from HoloLoom.terminal_ui import TerminalUI
    
    ui = TerminalUI()
    result = await ui.weave_with_display(query)
    ui.show_history()
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich import box
from rich.text import Text

from HoloLoom.weaving_orchestrator import WeavingOrchestrator, Query
from HoloLoom.loom.command import PatternCard
from HoloLoom.protocols import ComplexityLevel, ProvenceTrace

# Awareness layer imports (graceful degradation if not available)
try:
    from HoloLoom.awareness.compositional_awareness import (
        CompositionalAwarenessLayer,
        UnifiedAwarenessContext,
        format_awareness_for_prompt
    )
    from HoloLoom.awareness.dual_stream import (
        DualStreamGenerator,
        DualStreamResponse
    )
    from HoloLoom.awareness.meta_awareness import (
        MetaAwarenessLayer,
        SelfReflectionResult
    )
    AWARENESS_AVAILABLE = True
except ImportError:
    AWARENESS_AVAILABLE = False
    CompositionalAwarenessLayer = None
    DualStreamGenerator = None
    MetaAwarenessLayer = None


@dataclass
class ConversationEntry:
    """Single conversation turn"""
    timestamp: datetime
    query: str
    complexity: ComplexityLevel
    pattern: PatternCard
    response: str
    confidence: float
    duration_ms: float
    trace: Optional[ProvenceTrace] = None

    # Awareness context (optional)
    awareness_context: Optional[Any] = None
    dual_stream: Optional[Any] = None
    meta_reflection: Optional[Any] = None


class TerminalUI:
    """
    Rich terminal interface for HoloLoom weaving operations.
    
    Provides real-time visualization of the weaving process with:
    - Live progress tracking through 9 pipeline stages
    - Interactive pattern selection
    - Conversation history
    - Detailed trace exploration
    """
    
    def __init__(
        self,
        orchestrator: Optional[WeavingOrchestrator] = None,
        enable_awareness: bool = True
    ):
        """
        Initialize terminal UI.

        Args:
            orchestrator: Existing WeavingOrchestrator, or creates new one
            enable_awareness: Enable compositional awareness features
        """
        self.console = Console()
        self.orchestrator = orchestrator
        self.conversation_history: List[ConversationEntry] = []

        # Awareness layer integration
        self.enable_awareness = enable_awareness and AWARENESS_AVAILABLE
        self.awareness_layer = None
        self.dual_stream_gen = None
        self.meta_awareness = None

        if self.enable_awareness:
            try:
                # Initialize awareness stack (for demo purposes)
                self.awareness_layer = CompositionalAwarenessLayer()
                self.dual_stream_gen = DualStreamGenerator(self.awareness_layer)
                self.meta_awareness = MetaAwarenessLayer(self.awareness_layer)
                self.console.print("[dim]✓ Awareness layer enabled[/]")
            except Exception as e:
                self.console.print(f"[yellow]⚠ Awareness disabled: {e}[/]")
                self.enable_awareness = False

        # Stage names for progress display
        self.stages = [
            ("1", "Loom Command", "Pattern selection"),
            ("2", "Chrono Trigger", "Temporal window"),
            ("3", "Yarn Graph", "Thread selection"),
            ("4", "Resonance Shed", "Feature extraction"),
            ("5", "Warp Space", "Tensor manifold"),
            ("6", "Memory Crawl", "Multipass retrieval"),
            ("7", "Convergence", "Decision engine"),
            ("8", "Tool Execution", "Action execution"),
            ("9", "Spacetime", "Result synthesis"),
        ]
    
    def print_banner(self):
        """Display HoloLoom banner"""
        banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗[/]
[bold cyan]║[/]  [bold magenta]HoloLoom[/] [dim]- Neural Decision-Making System[/]                  [bold cyan]║[/]
[bold cyan]║[/]  [dim]Shuttle-Centric Architecture with Multipass Crawling[/]      [bold cyan]║[/]
[bold cyan]╚═══════════════════════════════════════════════════════════════╝[/]
"""
        self.console.print(banner)
    
    def show_pattern_selection_menu(self) -> PatternCard:
        """
        Interactive pattern card selection.
        
        Returns:
            Selected PatternCard
        """
        self.console.print("\n[bold]Select Pattern Card:[/]\n")
        
        patterns = {
            "1": (PatternCard.BARE, "LITE complexity", "50ms", "Greetings, simple commands"),
            "2": (PatternCard.FAST, "FAST complexity", "150ms", "Standard queries, questions"),
            "3": (PatternCard.FUSED, "FULL complexity", "300ms", "Detailed analysis, research"),
        }
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Pattern", style="bold")
        table.add_column("Complexity", style="yellow")
        table.add_column("Target", style="green")
        table.add_column("Best For", style="dim")
        
        for key, (pattern, complexity, target, use_case) in patterns.items():
            table.add_row(key, pattern.value.upper(), complexity, target, use_case)
        
        self.console.print(table)
        
        choice = Prompt.ask(
            "\n[bold cyan]Choose pattern[/]",
            choices=["1", "2", "3", "auto"],
            default="auto"
        )
        
        if choice == "auto":
            self.console.print("[dim]Using automatic complexity detection[/]")
            return None
        else:
            pattern = patterns[choice][0]
            self.console.print(f"[green]Selected: {pattern.value.upper()}[/]")
            return pattern
    
    async def weave_with_display(
        self,
        query: str,
        pattern: Optional[PatternCard] = None,
        show_trace: bool = True
    ) -> Any:
        """
        Execute weaving with live progress display.
        
        Args:
            query: User query text
            pattern: Optional pattern override (None for auto-detect)
            show_trace: Show detailed trace after completion
        
        Returns:
            Spacetime result from weaving
        """
        start_time = time.perf_counter()
        
        # Create query object
        query_obj = Query(text=query)
        
        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Create task for overall progress
            main_task = progress.add_task(
                "[cyan]Weaving in progress...",
                total=len(self.stages)
            )
            
            # Stage tasks
            stage_tasks = {}
            for stage_num, stage_name, stage_desc in self.stages:
                task_id = progress.add_task(
                    f"[dim]{stage_num}. {stage_name}[/]: {stage_desc}",
                    total=1,
                    visible=False
                )
                stage_tasks[stage_num] = task_id
            
            # Execute weaving (simplified - no stage-by-stage tracking for now)
            # In production, this would hook into orchestrator events
            result = await self.orchestrator.weave(
                query_obj,
                pattern_override=pattern
            )
            
            # Simulate stage completion for display
            for stage_num, stage_name, stage_desc in self.stages:
                task_id = stage_tasks[stage_num]
                progress.update(task_id, visible=True)
                await asyncio.sleep(0.05)  # Brief display
                progress.update(task_id, completed=1)
                progress.update(main_task, advance=1)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Display result
        self._display_result(result, duration_ms)
        
        # Store in history
        entry = ConversationEntry(
            timestamp=datetime.now(),
            query=query,
            complexity=result.complexity if hasattr(result, 'complexity') else ComplexityLevel.FAST,
            pattern=pattern or PatternCard.FAST,
            response=result.answer if hasattr(result, 'answer') else str(result),
            confidence=result.confidence if hasattr(result, 'confidence') else 0.0,
            duration_ms=duration_ms,
            trace=result.trace if hasattr(result, 'trace') else None
        )
        self.conversation_history.append(entry)
        
        # Show trace if requested
        if show_trace and hasattr(result, 'trace'):
            self.show_trace(result.trace)
        
        return result
    
    def _display_result(self, result: Any, duration_ms: float):
        """Display weaving result in a panel"""
        # Extract key information
        answer = result.answer if hasattr(result, 'answer') else str(result)
        confidence = result.confidence if hasattr(result, 'confidence') else 0.0
        complexity = result.complexity.name if hasattr(result, 'complexity') else "UNKNOWN"
        
        # Build result display
        result_text = f"""
[bold]Answer:[/] {answer}

[dim]Metrics:[/]
  • Complexity: [yellow]{complexity}[/]
  • Confidence: [{'green' if confidence > 0.7 else 'yellow' if confidence > 0.5 else 'red'}]{confidence:.2%}[/]
  • Duration: [cyan]{duration_ms:.1f}ms[/]
"""
        
        panel = Panel(
            result_text.strip(),
            title="[bold green]Weaving Complete[/]",
            border_style="green",
            box=box.DOUBLE
        )
        self.console.print("\n", panel)
    
    def show_trace(self, trace: ProvenceTrace):
        """
        Display detailed provenance trace.

        Args:
            trace: ProvenceTrace object with shuttle events
        """
        self.console.print("\n[bold cyan]Provenance Trace:[/]\n")

        # Create tree view of events
        tree = Tree("[bold]Shuttle Events[/]")

        if hasattr(trace, 'shuttle_events'):
            for event in trace.shuttle_events:
                event_type = event.get('event_type', 'unknown')
                description = event.get('description', '')
                data = event.get('data', {})

                # Format event node
                event_node = tree.add(f"[yellow]{event_type}[/]: {description}")

                # Add data as sub-items
                if data:
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            event_node.add(f"[dim]{key}:[/] [cyan]{value}[/]")
                        else:
                            event_node.add(f"[dim]{key}:[/] {value}")

        self.console.print(tree)

    def show_awareness_context(self, context: 'UnifiedAwarenessContext'):
        """
        Display compositional awareness context.

        Args:
            context: UnifiedAwarenessContext from awareness layer
        """
        if not context:
            self.console.print("[yellow]No awareness context available[/]")
            return

        # Build awareness display
        table = Table(title="[bold magenta]Compositional Awareness[/]", box=box.ROUNDED)
        table.add_column("Aspect", style="cyan", width=20)
        table.add_column("Details", style="white", width=50)

        # Structural awareness
        struct = context.structural
        table.add_row(
            "Structure",
            f"Type: [yellow]{struct.phrase_type}[/]\n"
            + (f"Question: [yellow]{struct.question_type}[/]\n" if struct.is_question else "")
            + f"Response: [yellow]{struct.suggested_response_type}[/]"
        )

        # Pattern recognition
        patterns = context.patterns
        table.add_row(
            "Patterns",
            f"Phrase: \"{patterns.phrase}\"\n"
            f"Seen: [cyan]{patterns.seen_count}×[/]\n"
            f"Confidence: [{'green' if patterns.confidence > 0.7 else 'yellow'}]{patterns.confidence:.2f}[/]\n"
            f"Domain: [yellow]{patterns.domain}/{patterns.subdomain}[/]"
        )

        # Confidence signals
        conf = context.confidence
        uncertainty_color = "green" if conf.uncertainty_level < 0.3 else "yellow" if conf.uncertainty_level < 0.7 else "red"
        table.add_row(
            "Confidence",
            f"Cache: [{uncertainty_color}]{conf.query_cache_status}[/]\n"
            f"Uncertainty: [{uncertainty_color}]{conf.uncertainty_level:.2f}[/]\n"
            + ("⚠️ [red]Knowledge gap detected[/]\n" if conf.knowledge_gap_detected else "")
            + (f"⚠️ Suggest: {conf.suggested_clarification}" if conf.should_ask_clarification else "✓ Ready to respond")
        )

        # Internal guidance
        internal = context.internal_guidance
        table.add_row(
            "Internal Strategy",
            f"Structure: [cyan]{internal.reasoning_structure}[/]\n"
            + (f"Shortcuts: {', '.join(internal.high_confidence_shortcuts)}\n" if internal.high_confidence_shortcuts else "")
            + (f"Checks: {', '.join(internal.low_confidence_checks)}" if internal.low_confidence_checks else "")
        )

        # External guidance
        external = context.external_guidance
        table.add_row(
            "External Strategy",
            f"Tone: [cyan]{external.confidence_tone}[/]\n"
            f"Structure: [cyan]{external.response_structure}[/]\n"
            f"Length: [cyan]{external.expected_length}[/]\n"
            + (f"Hedging: {', '.join(external.appropriate_hedging)}" if external.appropriate_hedging else "Direct response")
        )

        self.console.print("\n", table, "\n")

    def show_dual_stream(self, dual_stream: 'DualStreamResponse'):
        """
        Display dual-stream response (internal reasoning + external response).

        Args:
            dual_stream: DualStreamResponse with both streams
        """
        if not dual_stream:
            self.console.print("[yellow]No dual-stream data available[/]")
            return

        # Create two-column layout
        layout = Layout()
        layout.split_row(
            Layout(name="internal"),
            Layout(name="external")
        )

        # Internal reasoning panel
        internal_panel = Panel(
            dual_stream.internal_stream,
            title="[bold cyan]Internal Reasoning[/]",
            border_style="cyan",
            box=box.ROUNDED
        )
        layout["internal"].update(internal_panel)

        # External response panel
        external_panel = Panel(
            dual_stream.external_stream,
            title="[bold green]External Response[/]",
            border_style="green",
            box=box.ROUNDED
        )
        layout["external"].update(external_panel)

        self.console.print("\n", layout, "\n")
        self.console.print(f"[dim]Generated in {dual_stream.generation_time_ms:.1f}ms[/]\n")

    def show_meta_reflection(self, reflection: 'SelfReflectionResult'):
        """
        Display meta-awareness recursive self-reflection.

        Args:
            reflection: SelfReflectionResult from meta-awareness layer
        """
        if not reflection:
            self.console.print("[yellow]No meta-reflection available[/]")
            return

        # Use the built-in formatting
        introspection = reflection.format_introspection()

        # Rich formatting for terminal display
        panel = Panel(
            introspection,
            title="[bold magenta]Meta-Awareness: Recursive Self-Reflection[/]",
            border_style="magenta",
            box=box.DOUBLE
        )

        self.console.print("\n", panel, "\n")

    def _show_last_awareness(self):
        """Show awareness context from last query"""
        if not self.conversation_history:
            self.console.print("[yellow]No queries yet[/]")
            return

        last_entry = self.conversation_history[-1]

        if not last_entry.awareness_context:
            self.console.print("[yellow]No awareness context for last query[/]")
            return

        self.console.print(f"\n[bold]Awareness Context for:[/] \"{last_entry.query}\"\n")

        # Show compositional awareness
        if last_entry.awareness_context:
            self.show_awareness_context(last_entry.awareness_context)

        # Show dual stream
        if last_entry.dual_stream:
            if Confirm.ask("\n[bold]Show dual-stream (internal reasoning)?[/]", default=True):
                self.show_dual_stream(last_entry.dual_stream)

        # Show meta-reflection
        if last_entry.meta_reflection:
            if Confirm.ask("\n[bold]Show meta-awareness (recursive self-reflection)?[/]", default=False):
                self.show_meta_reflection(last_entry.meta_reflection)

    async def weave_with_awareness(
        self,
        query: str,
        pattern: Optional[PatternCard] = None,
        show_awareness_live: bool = False
    ) -> Any:
        """
        Execute weaving with awareness context generation.

        This is the awareness-enhanced version of weave_with_display.

        Args:
            query: User query text
            pattern: Optional pattern override
            show_awareness_live: Show awareness context immediately after generation

        Returns:
            Spacetime result (or dual-stream response if awareness enabled)
        """
        if not self.enable_awareness:
            # Fall back to standard weaving
            return await self.weave_with_display(query, pattern, show_trace=False)

        start_time = time.perf_counter()

        # Generate dual-stream response with awareness
        dual_stream = await self.dual_stream_gen.generate(query, show_internal=True)

        # Generate meta-reflection
        meta_reflection = await self.meta_awareness.recursive_self_reflection(
            query=query,
            response=dual_stream.external_stream,
            awareness_context=dual_stream.awareness_context
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Display external response
        self._display_awareness_result(dual_stream, meta_reflection, duration_ms)

        # Store in history with awareness context
        entry = ConversationEntry(
            timestamp=datetime.now(),
            query=query,
            complexity=ComplexityLevel.FULL,  # Awareness mode always uses FULL
            pattern=pattern or PatternCard.FUSED,
            response=dual_stream.external_stream,
            confidence=1.0 - dual_stream.awareness_context.confidence.uncertainty_level,
            duration_ms=duration_ms,
            trace=None,
            awareness_context=dual_stream.awareness_context,
            dual_stream=dual_stream,
            meta_reflection=meta_reflection
        )
        self.conversation_history.append(entry)

        # Show awareness context if requested
        if show_awareness_live:
            if Confirm.ask("\n[bold]Show compositional awareness?[/]", default=True):
                self.show_awareness_context(dual_stream.awareness_context)
            if Confirm.ask("\n[bold]Show internal reasoning?[/]", default=True):
                self.show_dual_stream(dual_stream)
            if Confirm.ask("\n[bold]Show meta-reflection?[/]", default=False):
                self.show_meta_reflection(meta_reflection)

        return dual_stream

    def _display_awareness_result(
        self,
        dual_stream: 'DualStreamResponse',
        meta_reflection: 'SelfReflectionResult',
        duration_ms: float
    ):
        """Display awareness-enhanced result"""
        ctx = dual_stream.awareness_context

        # Build result display
        confidence = 1.0 - ctx.confidence.uncertainty_level
        confidence_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.5 else "red"

        result_text = f"""
[bold]Response:[/] {dual_stream.external_stream}

[dim]Awareness Metrics:[/]
  • Domain: [yellow]{ctx.patterns.domain}/{ctx.patterns.subdomain}[/]
  • Cache Status: [{confidence_color}]{ctx.confidence.query_cache_status}[/]
  • Confidence: [{confidence_color}]{confidence:.2%}[/]
  • Uncertainty: [{confidence_color}]{ctx.confidence.uncertainty_level:.2f}[/]
  • Epistemic Humility: [cyan]{meta_reflection.epistemic_humility:.2f}[/]
  • Duration: [cyan]{duration_ms:.1f}ms[/]

[dim]Type 'awareness' to see full context[/]
"""

        panel = Panel(
            result_text.strip(),
            title="[bold magenta]Awareness-Guided Response[/]",
            border_style="magenta",
            box=box.DOUBLE
        )
        self.console.print("\n", panel)

    def show_history(self, limit: int = 10):
        """
        Display conversation history.
        
        Args:
            limit: Maximum number of entries to show
        """
        if not self.conversation_history:
            self.console.print("[yellow]No conversation history yet[/]")
            return
        
        self.console.print(f"\n[bold]Conversation History[/] [dim](last {limit})[/]\n")
        
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Query", style="white", width=40)
        table.add_column("Complexity", style="yellow", width=12)
        table.add_column("Confidence", style="green", width=10)
        table.add_column("Duration", style="blue", width=10)
        
        for entry in self.conversation_history[-limit:]:
            time_str = entry.timestamp.strftime("%H:%M:%S")
            query_str = entry.query[:37] + "..." if len(entry.query) > 40 else entry.query
            complexity_str = entry.complexity.name
            confidence_str = f"{entry.confidence:.1%}"
            duration_str = f"{entry.duration_ms:.1f}ms"
            
            table.add_row(
                time_str,
                query_str,
                complexity_str,
                confidence_str,
                duration_str
            )
        
        self.console.print(table)
    
    def show_stats(self):
        """Display session statistics"""
        if not self.conversation_history:
            self.console.print("[yellow]No statistics available yet[/]")
            return
        
        total_queries = len(self.conversation_history)
        avg_duration = sum(e.duration_ms for e in self.conversation_history) / total_queries
        avg_confidence = sum(e.confidence for e in self.conversation_history) / total_queries
        
        # Count by complexity
        complexity_counts = {}
        for entry in self.conversation_history:
            level = entry.complexity.name
            complexity_counts[level] = complexity_counts.get(level, 0) + 1
        
        stats_text = f"""
[bold]Session Statistics:[/]

Total Queries: [cyan]{total_queries}[/]
Average Duration: [cyan]{avg_duration:.1f}ms[/]
Average Confidence: [{'green' if avg_confidence > 0.7 else 'yellow'}]{avg_confidence:.1%}[/]

[bold]Complexity Distribution:[/]
"""
        for level, count in sorted(complexity_counts.items()):
            pct = (count / total_queries) * 100
            stats_text += f"  • {level}: [cyan]{count}[/] [dim]({pct:.0f}%)[/]\n"
        
        panel = Panel(
            stats_text.strip(),
            title="[bold blue]Performance Metrics[/]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print("\n", panel)
    
    async def interactive_session(self):
        """
        Run interactive Q&A session with rich UI.
        """
        self.print_banner()
        
        # Pattern selection
        use_pattern = Confirm.ask(
            "\n[bold]Use specific pattern card?[/]",
            default=False
        )
        
        pattern = None
        if use_pattern:
            pattern = self.show_pattern_selection_menu()
        
        commands_help = "[dim]Commands: 'quit', 'history', 'stats'"
        if self.enable_awareness:
            commands_help += ", 'awareness' (show last awareness context)"
        commands_help += "[/]\n"
        self.console.print("\n", commands_help)
        
        while True:
            try:
                # Get query
                query = Prompt.ask("\n[bold cyan]>[/]")
                
                if not query:
                    continue
                
                query_lower = query.lower().strip()
                
                # Handle special commands
                if query_lower in ('quit', 'exit', 'q'):
                    self.console.print("[yellow]Goodbye![/]")
                    break
                elif query_lower == 'history':
                    self.show_history()
                    continue
                elif query_lower == 'stats':
                    self.show_stats()
                    continue
                elif query_lower == 'clear':
                    self.console.clear()
                    self.print_banner()
                    continue
                elif query_lower == 'awareness' and self.enable_awareness:
                    self._show_last_awareness()
                    continue
                
                # Execute weaving (awareness mode if enabled)
                if self.enable_awareness:
                    await self.weave_with_awareness(query, pattern=pattern, show_awareness_live=False)
                else:
                    await self.weave_with_display(query, pattern=pattern, show_trace=False)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Session interrupted. Type 'quit' to exit.[/]")
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/]")
                import traceback
                traceback.print_exc()


async def main():
    """Demo of terminal UI"""
    from HoloLoom.config import Config
    
    # Create orchestrator
    config = Config.fast()
    orchestrator = WeavingOrchestrator(cfg=config)
    
    # Create UI
    ui = TerminalUI(orchestrator)
    
    # Run interactive session
    await ui.interactive_session()


if __name__ == "__main__":
    asyncio.run(main())
