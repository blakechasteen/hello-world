"""
Demo 1: Basic Prompt Management

Demonstrates core Promptly functionality:
- Creating prompts with add()
- Variable substitution {{variable}}
- Retrieving prompts with get()
- Listing all prompts
- Viewing prompt history with log()
- Dual storage (global vs local scope)

Interactive flow:
[1] Create a new prompt
[2] List all prompts
[3] Get a prompt by name
[4] Update an existing prompt
[5] View prompt history
[6] Variable substitution demo
[Q] Return to main menu
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactive_demo import DemoRunner


async def run_basic_demo(runner: "DemoRunner"):
    """
    Run the basic prompt management demo.

    Args:
        runner: DemoRunner instance for UI
    """
    runner.print_info("Welcome to Basic Prompt Management!")
    runner.print()

    # Ensure Promptly is initialized
    if not hasattr(runner, 'promptly') or runner.promptly is None:
        await runner.setup_sample_data()

    promptly = runner.promptly

    while True:
        # Show demo menu
        runner.print()
        runner.print_table(
            "📝 Basic Prompts Demo",
            ["Option", "Action"],
            [
                ["1", "Create a new prompt"],
                ["2", "List all prompts"],
                ["3", "Get a prompt by name"],
                ["4", "Update an existing prompt"],
                ["5", "View prompt history"],
                ["6", "Variable substitution demo"],
                ["Q", "Return to main menu"],
            ]
        )

        choice = runner.prompt_choice("Select action", ["1", "2", "3", "4", "5", "6", "q", "Q"], "1")

        if choice.lower() == 'q':
            break

        try:
            if choice == "1":
                await demo_create_prompt(runner, promptly)
            elif choice == "2":
                await demo_list_prompts(runner, promptly)
            elif choice == "3":
                await demo_get_prompt(runner, promptly)
            elif choice == "4":
                await demo_update_prompt(runner, promptly)
            elif choice == "5":
                await demo_view_history(runner, promptly)
            elif choice == "6":
                await demo_variable_substitution(runner, promptly)
        except Exception as e:
            runner.print_error(f"Error: {e}")


async def demo_create_prompt(runner: "DemoRunner", promptly):
    """Demonstrate creating a new prompt."""
    runner.print()
    runner.print_info("Creating a new prompt")
    runner.print()

    # Get prompt details
    name = runner.prompt_input("Prompt name", "my-prompt")
    content = runner.prompt_input(
        "Prompt content (use {{var}} for variables)",
        "Hello {{name}}, welcome to {{place}}!"
    )

    # Create the prompt
    with runner.show_progress("Creating prompt") as progress:
        task = progress.add_task("Creating...", total=1)
        result = promptly.add(name, content)
        progress.update(task, advance=1)

    # Show result
    runner.print_success(f"Created prompt '{name}'")
    runner.print_prompt(name, content, result.get('version'))

    # Show the version hash
    if 'version' in result:
        runner.print_info(f"Version: {result['version'][:12]}...")


async def demo_list_prompts(runner: "DemoRunner", promptly):
    """Demonstrate listing all prompts."""
    runner.print()
    runner.print_info("Listing all prompts")
    runner.print()

    prompts = promptly.list()

    if not prompts:
        runner.print_warning("No prompts found")
        return

    # Build table data
    rows = []
    for p in prompts:
        name = p.get('name', 'unknown')
        version = p.get('version', '')[:8] if p.get('version') else '-'
        content_preview = p.get('content', '')[:40]
        if len(p.get('content', '')) > 40:
            content_preview += "..."

        rows.append([name, version, content_preview])

    runner.print_table(
        f"📋 All Prompts ({len(prompts)})",
        ["Name", "Version", "Content Preview"],
        rows
    )


async def demo_get_prompt(runner: "DemoRunner", promptly):
    """Demonstrate retrieving a prompt."""
    runner.print()
    runner.print_info("Retrieving a prompt")
    runner.print()

    # List available prompts
    prompts = promptly.list()
    if prompts:
        available = [p['name'] for p in prompts]
        runner.print(f"[dim]Available: {', '.join(available)}[/dim]")

    name = runner.prompt_input("Prompt name", "summarize")

    # Get the prompt
    prompt = promptly.get(name)

    if prompt:
        runner.print_prompt(
            name,
            prompt.get('content', ''),
            prompt.get('version')
        )

        # Show metadata if available
        if prompt.get('metadata'):
            runner.print_json(prompt['metadata'], "Metadata")
    else:
        runner.print_warning(f"Prompt '{name}' not found")


async def demo_update_prompt(runner: "DemoRunner", promptly):
    """Demonstrate updating an existing prompt."""
    runner.print()
    runner.print_info("Updating a prompt (creates new version)")
    runner.print()

    # List available prompts
    prompts = promptly.list()
    if prompts:
        available = [p['name'] for p in prompts]
        runner.print(f"[dim]Available: {', '.join(available)}[/dim]")

    name = runner.prompt_input("Prompt name to update", "summarize")

    # Get current content
    current = promptly.get(name)
    if not current:
        runner.print_warning(f"Prompt '{name}' not found")
        return

    runner.print("[dim]Current content:[/dim]")
    runner.print_prompt(name, current.get('content', ''), current.get('version'))

    # Get new content
    new_content = runner.prompt_input(
        "New content",
        current.get('content', '') + "\n\nBe concise."
    )

    # Update
    result = promptly.add(name, new_content)

    runner.print_success(f"Updated prompt '{name}'")
    runner.print_prompt(name, new_content, result.get('version'))

    # Compare versions
    old_version = current.get('version', '')[:8]
    new_version = result.get('version', '')[:8]
    runner.print_info(f"Version changed: {old_version} → {new_version}")


async def demo_view_history(runner: "DemoRunner", promptly):
    """Demonstrate viewing prompt history."""
    runner.print()
    runner.print_info("Viewing prompt version history")
    runner.print()

    # List available prompts
    prompts = promptly.list()
    if prompts:
        available = [p['name'] for p in prompts]
        runner.print(f"[dim]Available: {', '.join(available)}[/dim]")

    name = runner.prompt_input("Prompt name", "summarize")

    # Get history
    history = promptly.log(name)

    if not history:
        runner.print_warning(f"No history found for '{name}'")
        return

    # Build table data
    rows = []
    for i, entry in enumerate(history):
        version = entry.get('version', '')[:12]
        timestamp = entry.get('timestamp', '-')
        if isinstance(timestamp, str) and len(timestamp) > 19:
            timestamp = timestamp[:19]

        message = entry.get('message', '-')
        if len(message) > 30:
            message = message[:27] + "..."

        rows.append([str(i + 1), version, timestamp, message])

    runner.print_table(
        f"📜 History for '{name}' ({len(history)} versions)",
        ["#", "Version", "Timestamp", "Message"],
        rows
    )

    # Offer to view a specific version
    if len(history) > 1:
        runner.print()
        if runner.prompt_confirm("View a specific version?", default=False):
            version_num = runner.prompt_input("Version number", "1")
            try:
                idx = int(version_num) - 1
                if 0 <= idx < len(history):
                    entry = history[idx]
                    version_id = entry.get('version')
                    prompt = promptly.get(name, version=version_id)
                    if prompt:
                        runner.print_prompt(name, prompt.get('content', ''), version_id)
            except (ValueError, IndexError):
                runner.print_warning("Invalid version number")


async def demo_variable_substitution(runner: "DemoRunner", promptly):
    """Demonstrate variable substitution in prompts."""
    runner.print()
    runner.print_info("Variable Substitution Demo")
    runner.print()

    runner.print("[dim]Promptly supports {{variable}} syntax for dynamic content.[/dim]")
    runner.print()

    # Create a demo prompt with variables
    demo_prompt = "Hello {{name}}! Welcome to {{city}}. Today is {{day}}."

    runner.print("[bold]Template:[/bold]")
    runner.print_prompt("greeting", demo_prompt)

    # Show how to substitute
    runner.print()
    runner.print("[bold]Substitution Example:[/bold]")
    runner.print()

    variables = {
        "name": "Alice",
        "city": "San Francisco",
        "day": "Monday"
    }

    runner.print_json(variables, "Variables")

    # Perform substitution (simple string replacement)
    result = demo_prompt
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", value)

    runner.print()
    runner.print("[bold]Result:[/bold]")
    runner.print(f"[green]{result}[/green]")

    # Show more complex example
    runner.print()
    runner.print("[bold]Code Review Template:[/bold]")

    code_review_template = """Review this {{language}} code for:
1. Bugs and errors
2. Best practices
3. Performance issues

Code:
```{{language}}
{{code}}
```

Provide specific, actionable feedback."""

    runner.print_prompt("code-review", code_review_template)

    code_vars = {
        "language": "Python",
        "code": "def add(a, b):\n    return a + b"
    }

    runner.print_json(code_vars, "Variables")

    result = code_review_template
    for key, value in code_vars.items():
        result = result.replace("{{" + key + "}}", value)

    runner.print()
    runner.print("[bold]Rendered Result:[/bold]")
    runner.print(f"[dim]{result}[/dim]")


# Direct execution for testing
if __name__ == "__main__":
    from .interactive_demo import DemoRunner

    async def main():
        runner = DemoRunner()
        await runner.setup_sample_data()
        await run_basic_demo(runner)

    asyncio.run(main())
