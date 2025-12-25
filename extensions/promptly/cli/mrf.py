#!/usr/bin/env python3
"""
Promptly MRF CLI - Metaprompt Refinement Framework Commands

Commands:
    promptly mrf status              Show MRF integration status
    promptly mrf enhance <name>      Enhance prompt with 7-component framework
    promptly mrf refine <name>       Refine a response with strategy
    promptly mrf auto-refine <name>  Auto-select strategy with Thompson Sampling
    promptly mrf recommend <type>    Get strategy recommendation
    promptly mrf templates           List available task templates
    promptly mrf add-template        Add custom task template
    promptly mrf stats               Show learning statistics
    promptly mrf save                Save learning state
    promptly mrf load                Load learning state
"""

import click
import asyncio
import sys
import json
from pathlib import Path
from typing import Optional

from promptly.core.promptly import Promptly


# ==================== Helpers ====================

def get_promptly() -> Promptly:
    """Create and connect Promptly instance."""
    p = Promptly()
    p.connect()
    return p


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def get_mrf_bridge():
    """Get MRF bridge instance with HoloLoom availability check."""
    try:
        from promptly.integrations.mrf_integration import MRFBridge, RefinementStrategy, ModelProvider
        return MRFBridge(), RefinementStrategy, ModelProvider
    except ImportError as e:
        click.echo("Error: MRF integration requires HoloLoom.", err=True)
        click.echo(f"  Import error: {e}", err=True)
        click.echo("", err=True)
        click.echo("To enable MRF:", err=True)
        click.echo("  1. Ensure HoloLoom is in your PYTHONPATH", err=True)
        click.echo("  2. Verify HoloLoom/prompting/unified_mrf.py exists", err=True)
        sys.exit(1)


def format_quality_score(score: float) -> str:
    """Format quality score with color indicator."""
    if score >= 0.9:
        indicator = "Excellent"
    elif score >= 0.7:
        indicator = "Good"
    elif score >= 0.5:
        indicator = "Fair"
    else:
        indicator = "Poor"
    return f"{score:.2f} ({indicator})"


# ==================== MRF Command Group ====================

@click.group()
def mrf():
    """Metaprompt Refinement Framework commands.

    MRF provides a 7-component metaprompting structure with Thompson Sampling
    for strategy selection. It enhances prompts through:

    \b
    7-Component Structure:
        ROLE → OBJECTIVE → PROCESS → FORMAT → CONSTRAINTS → UNCERTAINTY → VALIDATION

    \b
    5 Refinement Strategies:
        refine     - Iterative improvement
        critique   - Self-critical analysis
        verify     - Accuracy verification
        elegance   - Clarity optimization
        hofstadter - Recursive self-reference
        auto       - Thompson Sampling selection

    Examples:
        promptly mrf status
        promptly mrf enhance my_prompt --task summarization
        promptly mrf refine my_prompt --strategy verify
        promptly mrf recommend code_review
    """
    pass


# ==================== Status Command ====================

@mrf.command()
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def status(output_format: str):
    """Show MRF integration status and configuration.

    Examples:
        promptly mrf status
        promptly mrf status --format json
    """
    try:
        bridge, RefinementStrategy, ModelProvider = get_mrf_bridge()

        status_info = {
            "available": True,
            "hololoom_connected": bridge._hololoom_available,
            "strategies": [s.value for s in RefinementStrategy],
            "model_providers": [m.value for m in ModelProvider],
            "task_templates": bridge.get_task_templates(),
            "learning_enabled": True,
        }

        # Get learning statistics
        try:
            stats = bridge.get_learning_statistics()
            status_info["learning_stats"] = {
                "total_refinements": stats.get("total_updates", 0),
                "strategy_usage": stats.get("strategy_usage", {}),
            }
        except Exception:
            status_info["learning_stats"] = None

        if output_format == "json":
            click.echo(json.dumps(status_info, indent=2))
        else:
            click.echo("MRF Integration Status")
            click.echo("=" * 50)
            click.echo(f"Available:           {'Yes' if status_info['available'] else 'No'}")
            click.echo(f"HoloLoom Connected:  {'Yes' if status_info['hololoom_connected'] else 'No (Mock Mode)'}")
            click.echo()
            click.echo("Refinement Strategies:")
            for strategy in status_info["strategies"]:
                click.echo(f"  - {strategy}")
            click.echo()
            click.echo("Model Providers:")
            for provider in status_info["model_providers"]:
                click.echo(f"  - {provider}")
            click.echo()
            click.echo(f"Task Templates: {len(status_info['task_templates'])}")
            if status_info["task_templates"]:
                for template in status_info["task_templates"][:5]:
                    click.echo(f"  - {template}")
                if len(status_info["task_templates"]) > 5:
                    click.echo(f"  ... and {len(status_info['task_templates']) - 5} more")
            click.echo()
            if status_info["learning_stats"]:
                click.echo("Learning Statistics:")
                click.echo(f"  Total Refinements: {status_info['learning_stats']['total_refinements']}")
                if status_info["learning_stats"]["strategy_usage"]:
                    click.echo("  Strategy Usage:")
                    for strategy, count in status_info["learning_stats"]["strategy_usage"].items():
                        click.echo(f"    {strategy}: {count}")

    except Exception as e:
        if output_format == "json":
            click.echo(json.dumps({"available": False, "error": str(e)}))
        else:
            click.echo("MRF Integration Status")
            click.echo("=" * 50)
            click.echo(f"Available: No")
            click.echo(f"Error: {e}")
        sys.exit(1)


# ==================== Enhance Command ====================

@mrf.command()
@click.argument("name")
@click.option("--task", "-t", default="general",
              help="Task type (e.g., summarization, code_review, explanation)")
@click.option("--model", "-m",
              type=click.Choice(["claude", "gemini", "gpt", "ollama"]),
              default="claude",
              help="Target model provider for optimization")
@click.option("--output", "-o", type=click.Path(),
              help="Save enhanced prompt to file")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def enhance(name: str, task: str, model: str, output: Optional[str], output_format: str):
    """Enhance a prompt with the 7-component MRF framework.

    The 7-component structure ensures comprehensive prompt engineering:

    \b
    1. ROLE       - AI persona/expertise
    2. OBJECTIVE  - Goal with success criteria
    3. PROCESS    - Step-by-step reasoning
    4. FORMAT     - Expected output structure
    5. CONSTRAINTS - Boundaries and limits
    6. UNCERTAINTY - Confidence handling
    7. VALIDATION - Quality checks

    Examples:
        promptly mrf enhance summarize_prompt --task summarization
        promptly mrf enhance code_helper --task code_review --model ollama
        promptly mrf enhance my_prompt -o enhanced.txt
    """
    p = get_promptly()
    try:
        # Get the prompt
        prompt = p.get(name)
        if not prompt:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        # Get MRF bridge
        bridge, _, ModelProvider = get_mrf_bridge()

        # Map model string to enum
        model_enum = getattr(ModelProvider, model.upper())

        # Enhance the prompt
        result = run_async(bridge.enhance_prompt(
            prompt_name=name,
            prompt_content=prompt.content,
            task_type=task,
            model=model_enum
        ))

        if output_format == "json":
            output_data = {
                "prompt_name": name,
                "task_type": task,
                "model": model,
                "enhanced_prompt": result.enhanced_content,
                "components": result.components,
                "quality_estimate": result.quality_estimate,
                "metadata": result.metadata
            }
            if output:
                Path(output).write_text(json.dumps(output_data, indent=2))
                click.echo(f"Enhanced prompt saved to {output}")
            else:
                click.echo(json.dumps(output_data, indent=2))
        else:
            click.echo(f"Enhanced Prompt: {name}")
            click.echo("=" * 50)
            click.echo(f"Task Type:        {task}")
            click.echo(f"Target Model:     {model}")
            click.echo(f"Quality Estimate: {format_quality_score(result.quality_estimate)}")
            click.echo()
            click.echo("7-Component Breakdown:")
            click.echo("-" * 40)
            for component, content in result.components.items():
                if content:
                    preview = content[:100].replace("\n", " ")
                    if len(content) > 100:
                        preview += "..."
                    click.echo(f"  {component.upper()}: {preview}")
            click.echo()
            click.echo("Enhanced Content:")
            click.echo("-" * 40)
            if output:
                Path(output).write_text(result.enhanced_content)
                click.echo(f"Saved to: {output}")
            else:
                click.echo(result.enhanced_content[:1000])
                if len(result.enhanced_content) > 1000:
                    click.echo("...")
                    click.echo(f"\n(Full content: {len(result.enhanced_content)} chars)")

    finally:
        p.close()


# ==================== Refine Command ====================

@mrf.command()
@click.argument("name")
@click.option("--query", "-q", required=True, help="The query/input used")
@click.option("--response", "-r", help="Response to refine (or read from stdin)")
@click.option("--strategy", "-s",
              type=click.Choice(["refine", "critique", "verify", "elegance", "hofstadter"]),
              default="verify",
              help="Refinement strategy")
@click.option("--query-type", "-t", default="general",
              help="Query type for learning (e.g., factual, analytical, creative)")
@click.option("--max-iterations", "-i", type=int, default=3,
              help="Maximum refinement iterations")
@click.option("--quality-threshold", type=float, default=0.8,
              help="Stop when quality reaches this threshold")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def refine(name: str, query: str, response: Optional[str], strategy: str,
           query_type: str, max_iterations: int, quality_threshold: float,
           output_format: str):
    """Refine a response using a specific strategy.

    \b
    Strategies:
        refine     - Iterative improvement through multiple passes
        critique   - Self-critical analysis and correction
        verify     - Accuracy checking and fact verification
        elegance   - Clarity and simplicity optimization
        hofstadter - Recursive self-reference (meta-reasoning)

    Examples:
        promptly mrf refine my_prompt -q "What is Python?" -r "Python is a language."
        echo "Python is a language." | promptly mrf refine my_prompt -q "What is Python?"
        promptly mrf refine my_prompt -q "Explain ML" -s elegance --max-iterations 5
    """
    p = get_promptly()
    try:
        # Get the prompt
        prompt = p.get(name)
        if not prompt:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        # Get response from argument or stdin
        if not response:
            if not sys.stdin.isatty():
                response = sys.stdin.read().strip()
            else:
                click.echo("Enter response to refine (Ctrl+D to finish):")
                response = sys.stdin.read().strip()

        if not response:
            click.echo("Error: No response provided to refine", err=True)
            sys.exit(1)

        # Get MRF bridge
        bridge, RefinementStrategy, _ = get_mrf_bridge()

        # Map strategy string to enum
        strategy_enum = getattr(RefinementStrategy, strategy.upper())

        # Refine the response
        result = run_async(bridge.refine_response(
            prompt_name=name,
            query=query,
            response=response,
            strategy=strategy_enum,
            query_type=query_type,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold
        ))

        if output_format == "json":
            output_data = {
                "prompt_name": name,
                "query": query,
                "strategy": strategy,
                "original_response": response,
                "refined_response": result.refined_content,
                "quality_before": result.quality_before,
                "quality_after": result.quality_after,
                "iterations": result.iterations,
                "improvements": result.improvements,
                "metadata": result.metadata
            }
            click.echo(json.dumps(output_data, indent=2))
        else:
            improvement = result.quality_after - result.quality_before
            click.echo(f"Refinement Results: {name}")
            click.echo("=" * 50)
            click.echo(f"Strategy:         {strategy}")
            click.echo(f"Query Type:       {query_type}")
            click.echo(f"Iterations:       {result.iterations}")
            click.echo(f"Quality Before:   {format_quality_score(result.quality_before)}")
            click.echo(f"Quality After:    {format_quality_score(result.quality_after)}")
            click.echo(f"Improvement:      {improvement:+.2f}")
            click.echo()
            if result.improvements:
                click.echo("Improvements Made:")
                for i, imp in enumerate(result.improvements, 1):
                    click.echo(f"  {i}. {imp}")
                click.echo()
            click.echo("Refined Response:")
            click.echo("-" * 40)
            click.echo(result.refined_content)

    finally:
        p.close()


# ==================== Auto-Refine Command ====================

@mrf.command("auto-refine")
@click.argument("name")
@click.option("--query", "-q", required=True, help="The query/input used")
@click.option("--response", "-r", help="Response to refine (or read from stdin)")
@click.option("--query-type", "-t", default="general",
              help="Query type for strategy selection")
@click.option("--max-iterations", "-i", type=int, default=3,
              help="Maximum refinement iterations")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def auto_refine(name: str, query: str, response: Optional[str],
                query_type: str, max_iterations: int, output_format: str):
    """Auto-select refinement strategy using Thompson Sampling.

    Uses Bayesian optimization to select the best strategy based on
    historical performance data for the given query type.

    Examples:
        promptly mrf auto-refine my_prompt -q "Explain Python" -r "Python is a language."
        promptly mrf auto-refine my_prompt -q "Compare A and B" -t analytical
    """
    p = get_promptly()
    try:
        # Get the prompt
        prompt = p.get(name)
        if not prompt:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        # Get response from argument or stdin
        if not response:
            if not sys.stdin.isatty():
                response = sys.stdin.read().strip()
            else:
                click.echo("Enter response to refine (Ctrl+D to finish):")
                response = sys.stdin.read().strip()

        if not response:
            click.echo("Error: No response provided to refine", err=True)
            sys.exit(1)

        # Get MRF bridge
        bridge, _, _ = get_mrf_bridge()

        # Auto-refine with Thompson Sampling
        result = run_async(bridge.auto_refine(
            prompt_name=name,
            query=query,
            response=response,
            query_type=query_type,
            max_iterations=max_iterations
        ))

        if output_format == "json":
            output_data = {
                "prompt_name": name,
                "query": query,
                "query_type": query_type,
                "strategy_selected": result.strategy_used,
                "original_response": response,
                "refined_response": result.refined_content,
                "quality_before": result.quality_before,
                "quality_after": result.quality_after,
                "iterations": result.iterations,
                "improvements": result.improvements,
                "metadata": result.metadata
            }
            click.echo(json.dumps(output_data, indent=2))
        else:
            improvement = result.quality_after - result.quality_before
            click.echo(f"Auto-Refinement Results: {name}")
            click.echo("=" * 50)
            click.echo(f"Query Type:       {query_type}")
            click.echo(f"Strategy Selected: {result.strategy_used} (via Thompson Sampling)")
            click.echo(f"Iterations:       {result.iterations}")
            click.echo(f"Quality Before:   {format_quality_score(result.quality_before)}")
            click.echo(f"Quality After:    {format_quality_score(result.quality_after)}")
            click.echo(f"Improvement:      {improvement:+.2f}")
            click.echo()
            if result.improvements:
                click.echo("Improvements Made:")
                for i, imp in enumerate(result.improvements, 1):
                    click.echo(f"  {i}. {imp}")
                click.echo()
            click.echo("Refined Response:")
            click.echo("-" * 40)
            click.echo(result.refined_content)

    finally:
        p.close()


# ==================== Recommend Command ====================

@mrf.command()
@click.argument("query_type")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def recommend(query_type: str, output_format: str):
    """Get strategy recommendation for a query type using Thompson Sampling.

    Returns the recommended refinement strategy based on historical
    performance data, with confidence level and expected reward.

    \b
    Common Query Types:
        factual    - Simple fact lookup
        analytical - Comparison and analysis
        creative   - Creative writing
        code       - Code generation
        explanation - Explaining concepts
        summarization - Summarizing content

    Examples:
        promptly mrf recommend factual
        promptly mrf recommend code --format json
        promptly mrf recommend analytical
    """
    bridge, _, _ = get_mrf_bridge()

    recommendation = bridge.get_strategy_recommendation(query_type)

    if output_format == "json":
        output_data = {
            "query_type": query_type,
            "recommended_strategy": recommendation.strategy,
            "confidence": recommendation.confidence,
            "expected_reward": recommendation.expected_reward,
            "alternatives": [
                {"strategy": s, "expected_reward": r}
                for s, r in recommendation.alternatives
            ],
            "reasoning": recommendation.reasoning
        }
        click.echo(json.dumps(output_data, indent=2))
    else:
        click.echo(f"Strategy Recommendation for '{query_type}'")
        click.echo("=" * 50)
        click.echo(f"Recommended Strategy: {recommendation.strategy}")
        click.echo(f"Confidence:           {recommendation.confidence:.1%}")
        click.echo(f"Expected Reward:      {recommendation.expected_reward:.2f}")
        click.echo()
        click.echo(f"Reasoning: {recommendation.reasoning}")
        click.echo()
        if recommendation.alternatives:
            click.echo("Alternative Strategies:")
            for strategy, reward in recommendation.alternatives[:3]:
                click.echo(f"  - {strategy}: expected reward {reward:.2f}")


# ==================== Templates Command ====================

@mrf.command()
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def templates(output_format: str):
    """List available task templates for MRF enhancement.

    Task templates provide pre-configured settings for common prompt
    engineering patterns.

    Examples:
        promptly mrf templates
        promptly mrf templates --format json
    """
    bridge, _, _ = get_mrf_bridge()

    template_list = bridge.get_task_templates()

    if output_format == "json":
        click.echo(json.dumps({"templates": template_list}, indent=2))
    else:
        click.echo("Available Task Templates")
        click.echo("=" * 50)
        if template_list:
            for i, template in enumerate(template_list, 1):
                click.echo(f"  {i}. {template}")
        else:
            click.echo("  No templates defined")
            click.echo()
            click.echo("Add a template with:")
            click.echo("  promptly mrf add-template <name> --description ...")


# ==================== Add Template Command ====================

@mrf.command("add-template")
@click.argument("name")
@click.option("--description", "-d", required=True, help="Template description")
@click.option("--role", help="Default ROLE component")
@click.option("--format", help="Default FORMAT component")
@click.option("--constraints", help="Default CONSTRAINTS component")
def add_template(name: str, description: str, role: Optional[str],
                 format: Optional[str], constraints: Optional[str]):
    """Add a custom task template for MRF enhancement.

    Templates provide reusable configurations for the 7-component framework.

    Examples:
        promptly mrf add-template code_review -d "Code review template" \\
            --role "expert code reviewer" --format "structured feedback"
    """
    bridge, _, _ = get_mrf_bridge()

    config = {
        "description": description,
    }
    if role:
        config["default_role"] = role
    if format:
        config["default_format"] = format
    if constraints:
        config["default_constraints"] = constraints

    try:
        bridge.add_task_template(name, config)
        click.echo(f"Added template: {name}")
        click.echo(f"  Description: {description}")
        if role:
            click.echo(f"  Default Role: {role}")
        if format:
            click.echo(f"  Default Format: {format}")
        if constraints:
            click.echo(f"  Default Constraints: {constraints}")
    except Exception as e:
        click.echo(f"Error adding template: {e}", err=True)
        sys.exit(1)


# ==================== Stats Command ====================

@mrf.command()
@click.option("--format", "-f", "output_format",
              type=click.Choice(["text", "json"]), default="text",
              help="Output format")
def stats(output_format: str):
    """Show MRF learning statistics and performance metrics.

    Displays Thompson Sampling statistics including:
    - Strategy usage counts
    - Success rates per strategy
    - Beta distribution parameters (alpha, beta)
    - Recent performance trends

    Examples:
        promptly mrf stats
        promptly mrf stats --format json
    """
    bridge, _, _ = get_mrf_bridge()

    statistics = bridge.get_learning_statistics()

    if output_format == "json":
        click.echo(json.dumps(statistics, indent=2))
    else:
        click.echo("MRF Learning Statistics")
        click.echo("=" * 50)

        total = statistics.get("total_updates", 0)
        click.echo(f"Total Refinements: {total}")
        click.echo()

        if statistics.get("by_query_type"):
            click.echo("Statistics by Query Type:")
            click.echo("-" * 40)
            for query_type, type_stats in statistics["by_query_type"].items():
                click.echo(f"\n  {query_type}:")
                click.echo(f"    Total: {type_stats.get('total', 0)}")
                click.echo(f"    Avg Quality: {type_stats.get('avg_quality', 0):.2f}")
                if type_stats.get("strategy_breakdown"):
                    click.echo("    Strategy Breakdown:")
                    for strategy, count in type_stats["strategy_breakdown"].items():
                        click.echo(f"      {strategy}: {count}")

        if statistics.get("by_strategy"):
            click.echo("\nStatistics by Strategy:")
            click.echo("-" * 40)
            for strategy, strat_stats in statistics["by_strategy"].items():
                alpha = strat_stats.get("alpha", 1.0)
                beta = strat_stats.get("beta", 1.0)
                expected = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                click.echo(f"\n  {strategy}:")
                click.echo(f"    Usage Count: {strat_stats.get('count', 0)}")
                click.echo(f"    Success Rate: {strat_stats.get('success_rate', 0):.1%}")
                click.echo(f"    Expected Reward: {expected:.2f}")
                click.echo(f"    Beta Params: alpha={alpha:.2f}, beta={beta:.2f}")

        if statistics.get("recent_refinements"):
            click.echo("\nRecent Refinements:")
            click.echo("-" * 40)
            for ref in statistics["recent_refinements"][:5]:
                click.echo(f"  - {ref.get('query_type', 'unknown')}: "
                          f"{ref.get('strategy', 'unknown')} "
                          f"(quality: {ref.get('quality', 0):.2f})")


# ==================== Save/Load Commands ====================

@mrf.command()
@click.argument("path", type=click.Path())
def save(path: str):
    """Save MRF learning state to file.

    Persists Thompson Sampling parameters and learning statistics
    for later restoration.

    Examples:
        promptly mrf save ./mrf_state.json
        promptly mrf save ~/.promptly/mrf_learning.json
    """
    bridge, _, _ = get_mrf_bridge()

    try:
        bridge.save_learning_state(path)
        click.echo(f"Learning state saved to: {path}")
    except Exception as e:
        click.echo(f"Error saving learning state: {e}", err=True)
        sys.exit(1)


@mrf.command()
@click.argument("path", type=click.Path(exists=True))
def load(path: str):
    """Load MRF learning state from file.

    Restores Thompson Sampling parameters and learning statistics
    from a previously saved state.

    Examples:
        promptly mrf load ./mrf_state.json
        promptly mrf load ~/.promptly/mrf_learning.json
    """
    bridge, _, _ = get_mrf_bridge()

    try:
        bridge.load_learning_state(path)
        click.echo(f"Learning state loaded from: {path}")

        # Show brief stats
        stats = bridge.get_learning_statistics()
        click.echo(f"  Total refinements: {stats.get('total_updates', 0)}")
        if stats.get("by_strategy"):
            click.echo(f"  Strategies tracked: {len(stats['by_strategy'])}")
    except Exception as e:
        click.echo(f"Error loading learning state: {e}", err=True)
        sys.exit(1)
