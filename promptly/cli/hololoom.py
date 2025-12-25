"""
Promptly HoloLoom CLI Commands

Commands for HoloLoom integration including RAG enhancement,
agentic reasoning, and memory synchronization.

Usage:
    promptly hololoom similar <query> [--limit 5] [--min-quality 0.7]
    promptly hololoom context <name> [--k 5] [--task TYPE]
    promptly hololoom enhance <name> [--task TYPE] [--output FILE]
    promptly hololoom run-agentic <name> [--mode verify|research] [--input key=value]
    promptly hololoom recommend <task_type> [--candidates NAME...]
    promptly hololoom insights <task_type>
    promptly hololoom sync
    promptly hololoom status
"""

import click
import sys
import json
import asyncio
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from promptly.core.database import PromptlyDatabase


def get_database() -> PromptlyDatabase:
    """Get database instance."""
    return PromptlyDatabase()


def get_hololoom_bridge(db: PromptlyDatabase):
    """Get HoloLoom bridge instance."""
    from promptly.integrations.hololoom_bridge import HoloLoomBridge
    return HoloLoomBridge(db)


def get_rag_integration(db: PromptlyDatabase, bridge=None):
    """Get RAG integration instance."""
    from promptly.integrations.rag_integration import PromptRAGIntegration
    return PromptRAGIntegration(db=db, hololoom_bridge=bridge)


def get_agentic_executor(db: PromptlyDatabase, bridge=None, rag=None):
    """Get agentic executor instance."""
    from promptly.integrations.agentic_integration import AgenticPromptExecutor
    return AgenticPromptExecutor(db=db, hololoom_bridge=bridge, rag_integration=rag)


def run_async(coro):
    """Run async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ==================== HoloLoom Command Group ====================

@click.group()
def hololoom():
    """HoloLoom integration commands.

    Access RAG-enhanced prompts, agentic reasoning,
    and HoloLoom memory synchronization.

    Examples:
        promptly hololoom similar "explain machine learning"
        promptly hololoom enhance my_prompt --task summarization
        promptly hololoom run-agentic my_prompt --mode verify
    """
    pass


# ==================== Similar Command ====================

@hololoom.command()
@click.argument("query")
@click.option("--limit", "-n", type=int, default=10, help="Maximum results")
@click.option("--min-quality", "-q", type=float, default=0.7, help="Minimum quality score")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def similar(query: str, limit: int, min_quality: float, output_format: str):
    """Find similar prompts based on a query.

    Uses HoloLoom memory to find prompts similar to the query
    with quality filtering.

    Examples:
        promptly hololoom similar "explain neural networks"
        promptly hololoom similar "code review" --min-quality 0.8
        promptly hololoom similar "summarize text" -f json
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)
        results = run_async(bridge.discover_similar_prompts(
            query=query,
            limit=limit,
            min_quality=min_quality
        ))

        if not results:
            click.echo(f"No similar prompts found for '{query}'")
            return

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"Similar Prompts for: {query}")
            click.echo("=" * 60)
            click.echo("")
            click.echo(f"{'Rank':<6} {'Name':<30} {'Quality':<10} {'Relevance':<10} {'Score':<10}")
            click.echo("-" * 60)

            for i, result in enumerate(results, 1):
                name = result.get("name", "unknown")[:28]
                quality = result.get("quality_score", 0)
                relevance = result.get("relevance", 0)
                combined = result.get("combined_score", 0)
                click.echo(f"{i:<6} {name:<30} {quality:<10.3f} {relevance:<10.3f} {combined:<10.3f}")

    finally:
        db.close()


# ==================== Context Command ====================

@hololoom.command()
@click.argument("name")
@click.option("--k", "-k", type=int, default=5, help="Number of context items")
@click.option("--task", "-t", help="Task type for targeted retrieval")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def context(name: str, k: int, task: Optional[str], output_format: str):
    """Retrieve RAG context for a prompt.

    Gets relevant context from HoloLoom memory that can
    enhance the prompt's execution.

    Examples:
        promptly hololoom context my_prompt
        promptly hololoom context my_prompt --task summarization
        promptly hololoom context my_prompt -k 10 -f json
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)

        # Get prompt content first
        cursor = db.execute("SELECT content FROM prompts WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        prompt_content = row[0]

        # Get context
        results = run_async(bridge.get_context_for_prompt(
            prompt_content=prompt_content,
            task_description=task or "general",
            k=k
        ))

        if not results:
            click.echo(f"No context found for prompt '{name}'")
            return

        if output_format == "json":
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"RAG Context for: {name}")
            if task:
                click.echo(f"Task Type: {task}")
            click.echo("=" * 60)
            click.echo("")

            for i, ctx in enumerate(results, 1):
                relevance = ctx.get("relevance", 0)
                source = ctx.get("source", "unknown")
                content = ctx.get("content", "")[:200]

                click.echo(f"[{i}] Relevance: {relevance:.3f} | Source: {source}")
                click.echo(f"    {content}...")
                click.echo("")

    finally:
        db.close()


# ==================== Enhance Command ====================

@hololoom.command()
@click.argument("name")
@click.option("--task", "-t", help="Task type for context retrieval")
@click.option("--context-k", "-k", type=int, default=3, help="Number of context items")
@click.option("--output", "-o", type=click.Path(), help="Output file for enhanced prompt")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def enhance(name: str, task: Optional[str], context_k: int, output: Optional[str], output_format: str):
    """Enhance a prompt with RAG context.

    Retrieves relevant context from HoloLoom memory and
    injects it into the prompt for better results.

    Examples:
        promptly hololoom enhance my_prompt
        promptly hololoom enhance my_prompt --task summarization
        promptly hololoom enhance my_prompt -o enhanced.txt
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)
        rag = get_rag_integration(db, bridge)

        result = run_async(rag.enhance_prompt(
            prompt_name=name,
            variables=None,
            inject_context=True,
            context_k=context_k
        ))

        if output_format == "json":
            output_data = result.to_dict()
            if output:
                Path(output).write_text(json.dumps(output_data, indent=2))
                click.echo(f"Enhanced prompt saved to {output}")
            else:
                click.echo(json.dumps(output_data, indent=2))
        else:
            click.echo(f"Enhanced Prompt: {name}")
            click.echo("=" * 60)
            click.echo("")

            click.echo("Context Injected:")
            click.echo(f"  Items: {len(result.context_items)}")
            click.echo(f"  Tokens: {result.total_context_tokens}")
            click.echo(f"  Avg Relevance: {result.avg_relevance:.3f}")
            click.echo(f"  Method: {result.injection_method}")
            click.echo("")

            if result.context_items:
                click.echo("Context Sources:")
                for ctx in result.context_items:
                    click.echo(f"  - {ctx.source} (relevance: {ctx.relevance:.3f})")
                click.echo("")

            click.echo("Enhanced Prompt:")
            click.echo("-" * 60)
            click.echo(result.enhanced_prompt)

            if output:
                Path(output).write_text(result.enhanced_prompt)
                click.echo("")
                click.echo(f"Saved to {output}")

    finally:
        db.close()


# ==================== Run-Agentic Command ====================

@hololoom.command("run-agentic")
@click.argument("name")
@click.option("--mode", "-m", type=click.Choice(["direct", "verify", "research", "plan_execute"]), default="verify", help="Reasoning mode")
@click.option("--max-steps", "-s", type=int, default=5, help="Maximum reasoning steps")
@click.option("--input", "-i", "inputs", multiple=True, help="Input variables (key=value)")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def run_agentic(name: str, mode: str, max_steps: int, inputs: Tuple[str, ...], output_format: str):
    """Execute a prompt with agentic reasoning.

    Supports multiple reasoning modes:
    - direct: Single-pass execution (~150ms)
    - verify: Execute + verify output (~600ms)
    - research: Multi-query exploration (~900ms)
    - plan_execute: Goal decomposition (~750ms)

    Examples:
        promptly hololoom run-agentic my_prompt --mode verify
        promptly hololoom run-agentic my_prompt --mode research --max-steps 5
        promptly hololoom run-agentic my_prompt -i text="Hello" -i lang="en"
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)
        rag = get_rag_integration(db, bridge)
        executor = get_agentic_executor(db, bridge, rag)

        # Parse input variables
        variables: Dict[str, Any] = {}
        for inp in inputs:
            if "=" in inp:
                key, value = inp.split("=", 1)
                variables[key.strip()] = value.strip()

        # Import ReasoningMode
        from promptly.integrations.agentic_integration import ReasoningMode
        reasoning_mode = ReasoningMode(mode)

        result = run_async(executor.execute(
            prompt_name=name,
            variables=variables if variables else None,
            mode=reasoning_mode,
            max_steps=max_steps
        ))

        if output_format == "json":
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
            click.echo(f"Agentic Execution: {name}")
            click.echo(f"Mode: {mode.upper()}")
            click.echo("=" * 60)
            click.echo("")

            # Summary metrics
            click.echo("Execution Summary:")
            click.echo(f"  Steps Taken: {len(result.steps_taken)}")
            click.echo(f"  Confidence: {result.confidence:.3f}")
            click.echo(f"  Duration: {result.total_duration_ms:.1f}ms")
            click.echo("")

            # Reasoning trace
            if result.steps_taken:
                click.echo("Reasoning Trace:")
                click.echo("-" * 40)
                for step in result.steps_taken:
                    step_icon = {
                        "query": "🔍",
                        "verify": "✓",
                        "plan": "📋",
                        "execute": "▶",
                        "synthesize": "🔗"
                    }.get(step.step_type, "•")
                    click.echo(f"{step_icon} Step {step.step_index + 1}: {step.step_type}")
                    click.echo(f"    Duration: {step.duration_ms:.1f}ms | Confidence: {step.confidence:.3f}")
                    if len(step.output_text) > 100:
                        click.echo(f"    Output: {step.output_text[:100]}...")
                    else:
                        click.echo(f"    Output: {step.output_text}")
                click.echo("")

            # Verification results
            if result.verification:
                v = result.verification
                status_icon = "✅" if v.verified else "❌"
                click.echo(f"Verification: {status_icon}")
                click.echo(f"  Confidence: {v.confidence:.3f}")
                if v.checks_passed:
                    click.echo(f"  Passed: {', '.join(v.checks_passed[:3])}")
                if v.checks_failed:
                    click.echo(f"  Failed: {', '.join(v.checks_failed[:3])}")
                if v.suggestions:
                    click.echo(f"  Suggestions: {v.suggestions[0]}")
                click.echo("")

            # Final response
            click.echo("Response:")
            click.echo("-" * 60)
            click.echo(result.response)

    finally:
        db.close()


# ==================== Recommend Command ====================

@hololoom.command()
@click.argument("task_type")
@click.option("--candidates", "-c", multiple=True, help="Candidate prompts to consider")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def recommend(task_type: str, candidates: Tuple[str, ...], output_format: str):
    """Get Thompson Sampling recommendation for a task type.

    Uses HoloLoom's Thompson Sampling to recommend the best
    prompt for the given task type based on historical performance.

    Examples:
        promptly hololoom recommend summarization
        promptly hololoom recommend code_review -c prompt_a -c prompt_b
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)
        candidate_list = list(candidates) if candidates else None

        recommendation = run_async(bridge.get_thompson_recommendation(
            task_type=task_type,
            candidates=candidate_list
        ))

        if not recommendation:
            click.echo(f"No recommendation available for task type '{task_type}'")
            return

        if output_format == "json":
            click.echo(json.dumps(recommendation, indent=2))
        else:
            click.echo(f"Thompson Sampling Recommendation")
            click.echo(f"Task Type: {task_type}")
            click.echo("=" * 50)
            click.echo("")
            click.echo(f"Recommended: {recommendation.get('recommended_prompt', 'N/A')}")
            click.echo(f"Expected Quality: {recommendation.get('expected_quality', 0):.3f}")
            click.echo(f"Confidence: {recommendation.get('confidence', 0):.1%}")
            click.echo("")

            reasoning = recommendation.get("reasoning", "")
            if reasoning:
                click.echo(f"Reasoning: {reasoning}")
                click.echo("")

            alternatives = recommendation.get("alternatives", [])
            if alternatives:
                click.echo("Alternatives:")
                for alt in alternatives[:5]:
                    name = alt.get("name", "unknown")
                    expected = alt.get("expected_quality", 0)
                    click.echo(f"  - {name}: expected {expected:.3f}")

    finally:
        db.close()


# ==================== Insights Command ====================

@hololoom.command()
@click.argument("task_type")
@click.option("--days", "-d", type=int, default=30, help="Analysis window")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def insights(task_type: str, days: int, output_format: str):
    """Get comprehensive insights for a task type.

    Combines analytics, Thompson Sampling, and HoloLoom memory
    to provide deep insights about prompt performance for a task type.

    Examples:
        promptly hololoom insights summarization
        promptly hololoom insights code_review --days 7
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)

        # Get all prompts for this task type
        cursor = db.execute("""
            SELECT DISTINCT p.name
            FROM prompts p
            JOIN prompt_executions pe ON p.id = pe.prompt_id
            WHERE pe.task_type = ?
        """, (task_type,))
        prompt_names = [row[0] for row in cursor.fetchall()]

        if not prompt_names:
            click.echo(f"No prompts found for task type '{task_type}'")
            return

        # Get comprehensive stats for each prompt
        all_stats = []
        for name in prompt_names[:10]:  # Limit to top 10
            stats = run_async(bridge.get_comprehensive_stats(
                prompt_name=name,
                task_type=task_type,
                window_days=days
            ))
            if stats:
                all_stats.append(stats)

        if output_format == "json":
            click.echo(json.dumps({
                "task_type": task_type,
                "days_analyzed": days,
                "prompts": all_stats
            }, indent=2))
        else:
            click.echo(f"Insights: {task_type}")
            click.echo(f"Analysis Period: {days} days")
            click.echo("=" * 70)
            click.echo("")

            # Summary
            total_executions = sum(s.get("total_executions", 0) for s in all_stats)
            avg_quality = sum(s.get("avg_quality", 0) for s in all_stats) / len(all_stats) if all_stats else 0

            click.echo(f"Summary:")
            click.echo(f"  Prompts: {len(all_stats)}")
            click.echo(f"  Total Executions: {total_executions}")
            click.echo(f"  Average Quality: {avg_quality:.3f}")
            click.echo("")

            # Per-prompt breakdown
            click.echo("Prompt Performance:")
            click.echo("-" * 70)
            click.echo(f"{'Name':<30} {'Executions':<12} {'Quality':<10} {'Success':<10} {'Thompson':<10}")
            click.echo("-" * 70)

            for stats in sorted(all_stats, key=lambda x: x.get("avg_quality", 0), reverse=True):
                name = stats.get("prompt_name", "unknown")[:28]
                executions = stats.get("total_executions", 0)
                quality = stats.get("avg_quality", 0)
                success = stats.get("success_rate", 0)
                thompson = stats.get("thompson_expected", 0)
                click.echo(f"{name:<30} {executions:<12} {quality:<10.3f} {success:<10.1%} {thompson:<10.3f}")

            click.echo("")

            # Recommendations
            recommendations = [s.get("recommendation") for s in all_stats if s.get("recommendation")]
            if recommendations:
                click.echo("Recommendations:")
                for rec in recommendations[:3]:
                    click.echo(f"  • {rec}")

    finally:
        db.close()


# ==================== Sync Command ====================

@hololoom.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def sync(verbose: bool):
    """Synchronize with HoloLoom memory.

    Persists Thompson Sampling priors and execution history
    to HoloLoom's memory system for long-term learning.

    Examples:
        promptly hololoom sync
        promptly hololoom sync --verbose
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)

        click.echo("Synchronizing with HoloLoom...")
        result = run_async(bridge.sync_with_hololoom())

        if result.get("success"):
            click.echo("✅ Synchronization complete!")
            click.echo("")
            click.echo(f"  Priors synced: {result.get('priors_synced', 0)}")
            click.echo(f"  Executions synced: {result.get('executions_synced', 0)}")

            if verbose and result.get("details"):
                click.echo("")
                click.echo("Details:")
                for detail in result.get("details", []):
                    click.echo(f"  - {detail}")
        else:
            click.echo("❌ Synchronization failed")
            errors = result.get("errors", [])
            if errors:
                for error in errors:
                    click.echo(f"  Error: {error}", err=True)

    finally:
        db.close()


# ==================== Status Command ====================

@hololoom.command()
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]), default="text")
def status(output_format: str):
    """Show HoloLoom integration status.

    Displays connection status, memory statistics,
    and integration health.

    Examples:
        promptly hololoom status
        promptly hololoom status -f json
    """
    db = get_database()
    try:
        bridge = get_hololoom_bridge(db)

        # Get status information
        hololoom_available = bridge._hololoom is not None or bridge._ensure_hololoom()

        # Get execution stats
        cursor = db.execute("SELECT COUNT(*) FROM prompt_executions")
        total_executions = cursor.fetchone()[0]

        cursor = db.execute("SELECT COUNT(DISTINCT prompt_id) FROM prompt_executions")
        unique_prompts = cursor.fetchone()[0]

        cursor = db.execute("SELECT COUNT(*) FROM thompson_priors")
        thompson_priors = cursor.fetchone()[0]

        # Get recent activity
        cursor = db.execute("""
            SELECT COUNT(*) FROM prompt_executions
            WHERE created_at > datetime('now', '-24 hours')
        """)
        recent_24h = cursor.fetchone()[0]

        status_data = {
            "hololoom_connected": hololoom_available,
            "total_executions": total_executions,
            "unique_prompts": unique_prompts,
            "thompson_priors": thompson_priors,
            "executions_24h": recent_24h,
            "timestamp": datetime.now().isoformat()
        }

        if output_format == "json":
            click.echo(json.dumps(status_data, indent=2))
        else:
            click.echo("HoloLoom Integration Status")
            click.echo("=" * 50)
            click.echo("")

            # Connection status
            conn_icon = "✅" if hololoom_available else "⚠️"
            conn_status = "Connected" if hololoom_available else "Fallback Mode"
            click.echo(f"{conn_icon} HoloLoom: {conn_status}")
            click.echo("")

            # Statistics
            click.echo("Statistics:")
            click.echo(f"  Total Executions: {total_executions:,}")
            click.echo(f"  Unique Prompts: {unique_prompts}")
            click.echo(f"  Thompson Priors: {thompson_priors}")
            click.echo(f"  Last 24h: {recent_24h} executions")
            click.echo("")

            # Features status
            click.echo("Features:")
            click.echo(f"  ✓ RAG Integration: Available")
            click.echo(f"  ✓ Agentic Reasoning: Available")
            click.echo(f"  ✓ Thompson Sampling: Active ({thompson_priors} priors)")
            if hololoom_available:
                click.echo(f"  ✓ Memory Sync: Enabled")
            else:
                click.echo(f"  ⚠ Memory Sync: Using local fallback")

    finally:
        db.close()
