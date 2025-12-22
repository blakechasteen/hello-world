#!/usr/bin/env python3
"""
Promptly CLI - Git-like Prompt Management

Commands:
    promptly add <name> <content>       Add/update a prompt
    promptly get <name>                 Get a prompt
    promptly list                       List all prompts
    promptly log <name>                 Show prompt history
    promptly branch <name>              Create a branch
    promptly checkout <branch>          Switch branches
    promptly branches                   List all branches
    promptly delete <name>              Delete a prompt
    promptly eval <name>                Evaluate with LLM Judge
    promptly chain create <name>        Create a chain
    promptly chain run <name>           Execute a chain
    promptly skill add <name>           Add a skill
    promptly skill run <name>           Execute a skill
    promptly export <name>              Export to YAML
    promptly import <file>              Import from YAML
    promptly info                       Show configuration
    promptly demo [name]                Run interactive demos
"""

import click
import asyncio
import sys
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
    return asyncio.get_event_loop().run_until_complete(coro)


# ==================== Main CLI Group ====================

@click.group()
@click.version_option(version="1.0.0", prog_name="promptly")
def cli():
    """Promptly - Git-like Prompt Management

    Personal power-user tool for managing prompts, chains, and skills
    with slash-command invocation across all interfaces.

    Storage:
        ~/.promptly/    Global prompts (shared)
        .promptly/      Project prompts (overrides global)
    """
    pass


# ==================== Prompt Commands ====================

@cli.command()
@click.argument("name")
@click.argument("content", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read content from file")
@click.option("--scope", "-s", type=click.Choice(["local", "global"]), default="local", help="Storage scope")
@click.option("--metadata", "-m", multiple=True, help="Metadata as key=value pairs")
def add(name: str, content: Optional[str], file: Optional[str], scope: str, metadata: tuple):
    """Add or update a prompt.

    Examples:
        promptly add greeting "Hello, {{name}}!"
        promptly add system_prompt -f system.txt --scope global
        promptly add task -m "category=coding" -m "priority=high"
    """
    p = get_promptly()
    try:
        # Get content from file if specified
        if file:
            content = Path(file).read_text()
        elif not content:
            # Read from stdin
            click.echo("Enter prompt content (Ctrl+D to finish):")
            content = sys.stdin.read()

        # Parse metadata
        meta = {}
        for m in metadata:
            if "=" in m:
                key, value = m.split("=", 1)
                meta[key.strip()] = value.strip()

        version, commit_hash = p.add(name, content, metadata=meta, scope=scope)
        click.echo(f"[{commit_hash[:12]}] {name} v{version}")
    finally:
        p.close()


@cli.command()
@click.argument("name")
@click.option("--version", "-v", type=int, help="Specific version number")
@click.option("--commit", "-c", help="Commit hash or prefix")
@click.option("--raw", "-r", is_flag=True, help="Output raw content only")
def get(name: str, version: Optional[int], commit: Optional[str], raw: bool):
    """Get a prompt by name.

    Examples:
        promptly get greeting
        promptly get greeting --version 2
        promptly get greeting --commit abc123
        promptly get greeting --raw > greeting.txt
    """
    p = get_promptly()
    try:
        prompt = p.get(name, version=version, commit_hash=commit)
        if not prompt:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        if raw:
            click.echo(prompt.content)
        else:
            click.echo(f"Name: {prompt.name}")
            click.echo(f"Version: {prompt.current_version}")
            click.echo(f"Branch: {prompt.branch}")
            click.echo(f"Commit: {prompt.commit_hash[:12] if prompt.commit_hash else 'N/A'}")
            click.echo(f"Created: {prompt.created_at}")
            click.echo(f"Updated: {prompt.updated_at}")
            if prompt.metadata:
                click.echo(f"Metadata: {prompt.metadata}")
            click.echo()
            click.echo("Content:")
            click.echo("-" * 40)
            click.echo(prompt.content)
    finally:
        p.close()


@cli.command("list")
@click.option("--scope", "-s", type=click.Choice(["all", "local", "global"]), default="all", help="Filter by scope")
@click.option("--branch", "-b", help="Filter by branch")
def list_prompts(scope: str, branch: Optional[str]):
    """List all prompts.

    Examples:
        promptly list
        promptly list --scope local
        promptly list --branch experiments
    """
    p = get_promptly()
    try:
        prompts = p.list_prompts(scope=scope, branch=branch)
        if not prompts:
            click.echo("No prompts found")
            return

        click.echo(f"{'Name':<30} {'Version':<8} {'Branch':<15} {'Updated'}")
        click.echo("-" * 80)
        for prompt in prompts:
            updated = str(prompt.updated_at)[:19]
            click.echo(f"{prompt.name:<30} v{prompt.current_version:<7} {prompt.branch:<15} {updated}")
    finally:
        p.close()


@cli.command()
@click.argument("name")
@click.option("--limit", "-n", type=int, default=10, help="Number of versions to show")
def log(name: str, limit: int):
    """Show version history for a prompt.

    Example:
        promptly log greeting
        promptly log greeting --limit 5
    """
    p = get_promptly()
    try:
        history = p.log(name)
        if not history:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)

        for version in history[:limit]:
            short_hash = version.commit_hash[:12]
            created = str(version.created_at)[:19]
            preview = version.content[:50].replace("\n", " ")
            if len(version.content) > 50:
                preview += "..."
            click.echo(f"[{short_hash}] v{version.version} ({created}): {preview}")
    finally:
        p.close()


@cli.command()
@click.argument("name")
@click.option("--scope", "-s", type=click.Choice(["local", "global"]), default="local")
@click.confirmation_option(prompt="Are you sure you want to delete this prompt?")
def delete(name: str, scope: str):
    """Delete a prompt.

    Example:
        promptly delete old_prompt --scope local
    """
    p = get_promptly()
    try:
        if p.delete(name, scope=scope):
            click.echo(f"Deleted '{name}'")
        else:
            click.echo(f"Prompt '{name}' not found", err=True)
            sys.exit(1)
    finally:
        p.close()


# ==================== Branch Commands ====================

@cli.command()
@click.argument("name")
@click.option("--from", "from_branch", help="Parent branch")
@click.option("--scope", "-s", type=click.Choice(["local", "global"]), default="local")
def branch(name: str, from_branch: Optional[str], scope: str):
    """Create a new branch.

    Examples:
        promptly branch experiments
        promptly branch feature-x --from main
    """
    p = get_promptly()
    try:
        if p.branch(name, from_branch=from_branch, scope=scope):
            click.echo(f"Created branch '{name}'")
        else:
            click.echo(f"Branch '{name}' already exists", err=True)
            sys.exit(1)
    finally:
        p.close()


@cli.command()
@click.argument("name")
@click.option("--scope", "-s", type=click.Choice(["local", "global"]), default="local")
def checkout(name: str, scope: str):
    """Switch to a branch.

    Example:
        promptly checkout experiments
    """
    p = get_promptly()
    try:
        if p.checkout(name, scope=scope):
            click.echo(f"Switched to branch '{name}'")
        else:
            click.echo(f"Branch '{name}' not found", err=True)
            sys.exit(1)
    finally:
        p.close()


@cli.command("branches")
@click.option("--scope", "-s", type=click.Choice(["all", "local", "global"]), default="all")
def list_branches(scope: str):
    """List all branches.

    Example:
        promptly branches
    """
    p = get_promptly()
    try:
        branches = p.branches(scope=scope)
        current = p.current_branch()
        for branch in branches:
            prefix = "* " if branch == current else "  "
            click.echo(f"{prefix}{branch}")
    finally:
        p.close()


# ==================== Evaluation Commands ====================

@cli.command()
@click.argument("name")
@click.option("--criteria", "-c", default="quality", help="Evaluation criteria")
@click.option("--input", "-i", multiple=True, help="Test input as key=value pairs")
def eval(name: str, criteria: str, input: tuple):
    """Evaluate a prompt with LLM Judge.

    Examples:
        promptly eval greeting -i "name=World"
        promptly eval system_prompt --criteria safety
    """
    p = get_promptly()
    try:
        # Parse input variables
        test_input = {}
        for i in input:
            if "=" in i:
                key, value = i.split("=", 1)
                test_input[key.strip()] = value.strip()

        result = run_async(p.eval_prompt(name, criteria=criteria, test_input=test_input))

        click.echo(f"Prompt: {name}")
        click.echo(f"Criteria: {criteria}")
        click.echo(f"Score: {result['score']}/10")
        click.echo(f"Feedback: {result['feedback']}")
        click.echo()
        click.echo("Response Preview:")
        click.echo("-" * 40)
        click.echo(result['response'][:500])
        if len(result['response']) > 500:
            click.echo("...")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        p.close()


# ==================== Chain Commands ====================

@cli.group()
def chain():
    """Manage prompt chains."""
    pass


@chain.command("create")
@click.argument("name")
@click.option("--step", "-s", multiple=True, help="Step as prompt_name:output_key")
def chain_create(name: str, step: tuple):
    """Create a prompt chain.

    Example:
        promptly chain create summarize_flow -s "summarize:summary" -s "translate:result"
    """
    p = get_promptly()
    try:
        steps = []
        for s in step:
            parts = s.split(":")
            prompt_name = parts[0]
            output_key = parts[1] if len(parts) > 1 else f"step_{len(steps)}"
            steps.append({
                "prompt": prompt_name,
                "output_key": output_key,
                "input_mapping": {} if len(steps) == 0 else {output_key: steps[-1]["output_key"]}
            })

        chain_id = p.create_chain(name, steps)
        click.echo(f"Created chain '{name}' with {len(steps)} steps (ID: {chain_id})")
    finally:
        p.close()


@chain.command("run")
@click.argument("name")
@click.option("--input", "-i", multiple=True, help="Input as key=value pairs")
def chain_run(name: str, input: tuple):
    """Execute a prompt chain.

    Example:
        promptly chain run summarize_flow -i "text=Hello world"
    """
    p = get_promptly()
    try:
        # Parse input
        initial_input = {}
        for i in input:
            if "=" in i:
                key, value = i.split("=", 1)
                initial_input[key.strip()] = value.strip()

        result = run_async(p.execute_chain(name, initial_input=initial_input))

        click.echo(f"Chain '{name}' completed")
        click.echo("-" * 40)
        for key, value in result.items():
            click.echo(f"\n{key}:")
            click.echo(str(value)[:500])
            if len(str(value)) > 500:
                click.echo("...")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        p.close()


@chain.command("list")
def chain_list():
    """List all chains."""
    p = get_promptly()
    try:
        chains = p.list_chains()
        if not chains:
            click.echo("No chains found")
            return

        click.echo(f"{'Name':<30} {'Steps':<8} {'Created'}")
        click.echo("-" * 60)
        for c in chains:
            created = str(c.created_at)[:19]
            click.echo(f"{c.name:<30} {len(c.steps):<8} {created}")
    finally:
        p.close()


# ==================== Skill Commands ====================

@cli.group()
def skill():
    """Manage skills."""
    pass


@skill.command("add")
@click.argument("name")
@click.argument("description", required=False)
@click.option("--template", "-t", type=click.Path(exists=True), help="Prompt template file")
@click.option("--file", "-f", multiple=True, type=click.Path(exists=True), help="Attach files")
def skill_add(name: str, description: Optional[str], template: Optional[str], file: tuple):
    """Add a skill with optional attached files.

    Examples:
        promptly skill add data_analysis "Analyze CSV data"
        promptly skill add code_review -t template.txt -f example.py
    """
    p = get_promptly()
    try:
        # Get template
        if template:
            prompt_template = Path(template).read_text()
        else:
            click.echo("Enter prompt template (Ctrl+D to finish):")
            prompt_template = sys.stdin.read()

        # Collect files
        files = []
        for f in file:
            path = Path(f)
            files.append({
                "filename": path.name,
                "content": path.read_text(),
                "file_type": path.suffix.lstrip(".") or "text"
            })

        skill_id = p.create_skill(
            name=name,
            description=description or f"Skill: {name}",
            prompt_template=prompt_template,
            files=files
        )
        click.echo(f"Created skill '{name}' (ID: {skill_id}) with {len(files)} files")
    finally:
        p.close()


@skill.command("run")
@click.argument("name")
@click.option("--input", "-i", multiple=True, help="Input as key=value pairs")
def skill_run(name: str, input: tuple):
    """Execute a skill.

    Example:
        promptly skill run data_analysis -i "file=data.csv"
    """
    p = get_promptly()
    try:
        # Parse input
        input_vars = {}
        for i in input:
            if "=" in i:
                key, value = i.split("=", 1)
                input_vars[key.strip()] = value.strip()

        result = run_async(p.run_skill(name, input_vars=input_vars))

        click.echo(f"Skill '{name}' completed")
        click.echo("-" * 40)
        click.echo(result)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        p.close()


@skill.command("list")
def skill_list():
    """List all skills."""
    p = get_promptly()
    try:
        skills = p.list_skills()
        if not skills:
            click.echo("No skills found")
            return

        click.echo(f"{'Name':<30} {'Description':<40} {'Created'}")
        click.echo("-" * 90)
        for s in skills:
            desc = (s.description[:37] + "...") if len(s.description) > 40 else s.description
            created = str(s.created_at)[:19]
            click.echo(f"{s.name:<30} {desc:<40} {created}")
    finally:
        p.close()


@skill.command("get")
@click.argument("name")
def skill_get(name: str):
    """Get skill details."""
    p = get_promptly()
    try:
        s = p.get_skill(name)
        if not s:
            click.echo(f"Skill '{name}' not found", err=True)
            sys.exit(1)

        click.echo(f"Name: {s.name}")
        click.echo(f"Description: {s.description}")
        click.echo(f"Created: {s.created_at}")
        click.echo(f"Files: {len(s.files)}")
        if s.files:
            for f in s.files:
                click.echo(f"  - {f['filename']} ({f['file_type']})")
        click.echo()
        click.echo("Template:")
        click.echo("-" * 40)
        click.echo(s.prompt_template)
    finally:
        p.close()


# ==================== Import/Export ====================

@cli.command("export")
@click.argument("name")
@click.option("--format", "-f", type=click.Choice(["yaml", "json"]), default="yaml")
@click.option("--output", "-o", type=click.Path(), help="Output file")
def export_prompt(name: str, format: str, output: Optional[str]):
    """Export a prompt to YAML or JSON.

    Examples:
        promptly export greeting
        promptly export greeting -f json -o greeting.json
    """
    p = get_promptly()
    try:
        data = p.export_prompt(name, format=format)
        if output:
            Path(output).write_text(data)
            click.echo(f"Exported to {output}")
        else:
            click.echo(data)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        p.close()


@cli.command("import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--scope", "-s", type=click.Choice(["local", "global"]), default="local")
def import_prompt(file: str, scope: str):
    """Import a prompt from YAML or JSON file.

    Example:
        promptly import greeting.yaml
        promptly import prompt.json --scope global
    """
    p = get_promptly()
    try:
        data = Path(file).read_text()
        version, commit_hash = p.import_prompt(data, scope=scope)
        click.echo(f"Imported [{commit_hash[:12]}] v{version}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        p.close()


# ==================== Info ====================

@cli.command()
def info():
    """Show Promptly configuration and status."""
    p = get_promptly()
    try:
        info = p.info()
        click.echo("Promptly Configuration")
        click.echo("=" * 40)
        click.echo(f"Global Dir:     {info['global_dir']}")
        click.echo(f"Local Dir:      {info['local_dir'] or 'Not set'}")
        click.echo(f"Global Prompts: {info['global_prompts']}")
        click.echo(f"Local Prompts:  {info['local_prompts']}")
        click.echo(f"Current Branch: {info['current_branch']}")
    finally:
        p.close()


# ==================== LLM Commands ====================

@cli.group()
def llm():
    """LLM backend configuration and testing.

    Promptly auto-detects LLM backends in this order:
        1. Ollama (local, free)
        2. Claude API (ANTHROPIC_API_KEY)
        3. Mock (fallback, no real LLM)

    Examples:
        promptly llm status     Check connection
        promptly llm models     List Ollama models
        promptly llm test       Run a quick test
    """
    pass


@llm.command()
def status():
    """Check LLM backend status and connection."""
    click.echo("LLM Backend Status")
    click.echo("=" * 40)

    # Check Ollama
    ollama_status = "Not installed"
    ollama_running = False
    try:
        import ollama
        ollama_status = "Installed"
        try:
            models = ollama.list()
            ollama_running = True
            model_count = len(models.get('models', []))
            ollama_status = f"Running ({model_count} models)"
        except Exception:
            ollama_status = "Installed but not running"
    except ImportError:
        pass

    # Check Claude
    claude_status = "Not configured"
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        claude_status = "API key set"
        try:
            import anthropic
            claude_status = "Ready"
        except ImportError:
            claude_status = "API key set (pip install anthropic)"

    # Determine active backend
    from promptly.judge.llm_judge import LLMJudge
    judge = LLMJudge()
    active = judge.executor.__name__
    active_name = {
        "_ollama_executor": "Ollama",
        "_claude_executor": "Claude",
        "_mock_executor": "Mock (no LLM)"
    }.get(active, active)

    click.echo(f"Ollama:  {ollama_status}")
    click.echo(f"Claude:  {claude_status}")
    click.echo("-" * 40)
    click.echo(f"Active:  {active_name}")

    if active == "_mock_executor":
        click.echo("")
        click.echo("To enable a real LLM backend:")
        click.echo("  Ollama: ollama serve (in another terminal)")
        click.echo("  Claude: export ANTHROPIC_API_KEY=sk-...")


@llm.command()
def models():
    """List available Ollama models."""
    try:
        import ollama
    except ImportError:
        click.echo("Ollama not installed. Run: pip install ollama", err=True)
        sys.exit(1)

    try:
        result = ollama.list()
        # Handle both dict-style (old) and typed object (new v0.13.0+) responses
        if hasattr(result, 'models'):
            models_list = result.models  # New typed response
        else:
            models_list = result.get('models', [])  # Old dict response

        if not models_list:
            click.echo("No models installed.")
            click.echo("")
            click.echo("Install a model with:")
            click.echo("  ollama pull llama3.2:3b    (fast, recommended)")
            click.echo("  ollama pull llama3.2       (larger)")
            click.echo("  ollama pull mistral        (alternative)")
            return

        click.echo("Available Ollama Models")
        click.echo("=" * 50)

        for model in models_list:
            # Handle both typed objects (.model attr) and dicts ('name' key)
            name = getattr(model, 'model', None) or (model.get('name') if hasattr(model, 'get') else 'unknown')
            size = getattr(model, 'size', None) or (model.get('size', 0) if hasattr(model, 'get') else 0)
            size_gb = size / (1024 ** 3) if size else 0

            # Handle modified_at (can be datetime, string, or attribute)
            modified_at = getattr(model, 'modified_at', None) or (model.get('modified_at') if hasattr(model, 'get') else None)
            if modified_at:
                if hasattr(modified_at, 'strftime'):
                    modified = modified_at.strftime('%Y-%m-%d')
                else:
                    modified = str(modified_at)[:10]
            else:
                modified = ''

            if size_gb >= 1:
                size_str = f"{size_gb:.1f}GB"
            else:
                size_str = f"{size / (1024 ** 2):.0f}MB"

            click.echo(f"  {name:<30} {size_str:>8}  {modified}")

    except Exception as e:
        click.echo(f"Error connecting to Ollama: {e}", err=True)
        click.echo("")
        click.echo("Make sure Ollama is running: ollama serve")
        sys.exit(1)


@llm.command()
@click.option("--model", "-m", default="llama3.2:3b", help="Model to use")
@click.option("--prompt", "-p", default=None, help="Custom test prompt")
def test(model: str, prompt: Optional[str]):
    """Run a quick LLM test.

    Examples:
        promptly llm test
        promptly llm test -m mistral
        promptly llm test -p "What is 2+2?"
    """
    from promptly.judge.llm_judge import LLMJudge

    judge = LLMJudge()
    executor_name = judge.executor.__name__

    if executor_name == "_mock_executor":
        click.echo("No LLM backend available (using mock).", err=True)
        click.echo("Start Ollama: ollama serve")
        sys.exit(1)

    test_prompt = prompt or "What is 2+2? Answer in one word."

    click.echo(f"Testing LLM connection...")
    click.echo(f"Model: {model}")
    click.echo(f"Prompt: {test_prompt}")
    click.echo("-" * 40)

    try:
        if executor_name == "_ollama_executor":
            import ollama
            response = ollama.generate(model=model, prompt=test_prompt)
            result = response.get('response', '').strip()
        else:
            result = judge.executor(test_prompt)

        click.echo(f"Response: {result}")
        click.echo("-" * 40)
        click.echo("LLM connection successful!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@llm.command()
@click.argument("model_name")
def pull(model_name: str):
    """Pull/download an Ollama model.

    Examples:
        promptly llm pull llama3.2:3b
        promptly llm pull mistral
    """
    try:
        import ollama
    except ImportError:
        click.echo("Ollama not installed. Run: pip install ollama", err=True)
        sys.exit(1)

    click.echo(f"Pulling model: {model_name}")
    click.echo("This may take a while...")

    try:
        # Stream the pull progress
        for progress in ollama.pull(model_name, stream=True):
            status = progress.get('status', '')
            if 'completed' in progress and 'total' in progress:
                pct = int(progress['completed'] / progress['total'] * 100)
                click.echo(f"\r{status}: {pct}%", nl=False)
            else:
                click.echo(f"\r{status}", nl=False)

        click.echo("")
        click.echo(f"Model {model_name} ready!")

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


# ==================== Demo Commands ====================

@cli.command()
@click.argument("demo_name", required=False)
@click.option("--all", "run_all", is_flag=True, help="Run all demos (automated)")
@click.option("--mock", is_flag=True, help="Run in mock mode (no LLM required)")
def demo(demo_name: Optional[str], run_all: bool, mock: bool):
    """Run interactive demos.

    Available demos:
        basic       Basic prompt management
        versioning  Git-like versioning
        chains      Chains and skills
        judge       LLM Judge evaluation
        hololoom    HoloLoom integration

    Examples:
        promptly demo           Interactive menu
        promptly demo basic     Run specific demo
        promptly demo --all     Run all demos
        promptly demo --mock    Run without LLM
    """
    try:
        from promptly.demos.interactive_demo import run_interactive_demo, run_demo, DemoType
    except ImportError as e:
        click.echo(f"Error: Demo module not available - {e}", err=True)
        sys.exit(1)

    # Map demo names to types
    demo_map = {
        "basic": DemoType.BASIC,
        "versioning": DemoType.VERSIONING,
        "chains": DemoType.CHAINS,
        "judge": DemoType.JUDGE,
        "hololoom": DemoType.HOLOLOOM,
    }

    try:
        if run_all:
            # Run all demos in sequence
            run_async(_run_all_demos(mock))
        elif demo_name:
            # Run specific demo
            if demo_name.lower() not in demo_map:
                click.echo(f"Unknown demo: {demo_name}", err=True)
                click.echo(f"Available: {', '.join(demo_map.keys())}")
                sys.exit(1)
            demo_type = demo_map[demo_name.lower()]
            run_async(run_demo(demo_type, mock_mode=mock))
        else:
            # Run interactive menu
            run_async(run_interactive_demo(mock_mode=mock))
    except KeyboardInterrupt:
        click.echo("\nDemo interrupted.")
    except Exception as e:
        click.echo(f"Error running demo: {e}", err=True)
        sys.exit(1)


async def _run_all_demos(mock: bool):
    """Run all demos in sequence."""
    from promptly.demos.interactive_demo import run_demo, DemoType, DemoRunner

    click.echo("Running all Promptly demos...")
    click.echo("=" * 50)

    runner = DemoRunner(mock_mode=mock)
    await runner.setup_sample_data()

    demos = [
        DemoType.BASIC,
        DemoType.VERSIONING,
        DemoType.CHAINS,
        DemoType.JUDGE,
        DemoType.HOLOLOOM,
    ]

    for demo_type in demos:
        click.echo(f"\n{'='*50}")
        click.echo(f"Demo: {demo_type.value.upper()}")
        click.echo(f"{'='*50}\n")
        await runner.run_demo(demo_type)

    click.echo("\n" + "=" * 50)
    click.echo("All demos completed!")
    click.echo("=" * 50)


# ==================== Entry Point ====================

def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
