"""
Demo 2: Git-like Versioning

Demonstrates Promptly's version control features:
- Creating branches
- Switching branches with checkout()
- Viewing branch history
- Comparing versions
- Experimental prompt development workflow

Interactive flow:
[1] Create a new branch
[2] Switch branches
[3] List all branches
[4] Compare prompt versions
[5] Merge workflow example
[Q] Return to main menu
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interactive_demo import DemoRunner


async def run_versioning_demo(runner: "DemoRunner"):
    """
    Run the versioning workflow demo.

    Args:
        runner: DemoRunner instance for UI
    """
    runner.print_info("Welcome to Git-like Versioning!")
    runner.print()
    runner.print("[dim]Promptly provides git-style version control for prompts.[/dim]")
    runner.print("[dim]Create branches, experiment safely, and track history.[/dim]")
    runner.print()

    # Ensure Promptly is initialized
    if not hasattr(runner, 'promptly') or runner.promptly is None:
        await runner.setup_sample_data()

    promptly = runner.promptly

    while True:
        # Show demo menu
        runner.print()
        runner.print_table(
            "🌿 Versioning Demo",
            ["Option", "Action"],
            [
                ["1", "Create a new branch"],
                ["2", "Switch branches (checkout)"],
                ["3", "List all branches"],
                ["4", "Compare prompt versions"],
                ["5", "Full workflow example"],
                ["Q", "Return to main menu"],
            ]
        )

        choice = runner.prompt_choice("Select action", ["1", "2", "3", "4", "5", "q", "Q"], "1")

        if choice.lower() == 'q':
            break

        try:
            if choice == "1":
                await demo_create_branch(runner, promptly)
            elif choice == "2":
                await demo_switch_branch(runner, promptly)
            elif choice == "3":
                await demo_list_branches(runner, promptly)
            elif choice == "4":
                await demo_compare_versions(runner, promptly)
            elif choice == "5":
                await demo_full_workflow(runner, promptly)
        except Exception as e:
            runner.print_error(f"Error: {e}")


async def demo_create_branch(runner: "DemoRunner", promptly):
    """Demonstrate creating a new branch."""
    runner.print()
    runner.print_info("Creating a new branch")
    runner.print()

    runner.print("[dim]Branches let you experiment with prompts without affecting main.[/dim]")
    runner.print()

    # Get current branch
    current = promptly.current_branch if hasattr(promptly, 'current_branch') else 'main'
    runner.print(f"[dim]Current branch: [cyan]{current}[/cyan][/dim]")

    # Get new branch name
    branch_name = runner.prompt_input("New branch name", "experiment")

    # Create the branch
    with runner.show_progress("Creating branch") as progress:
        task = progress.add_task("Creating...", total=1)
        result = promptly.branch(branch_name)
        progress.update(task, advance=1)

    if result:
        runner.print_success(f"Created branch '{branch_name}'")
        runner.print_info(f"You are now on branch: [cyan]{branch_name}[/cyan]")

        # Show workflow hint
        runner.print()
        runner.print("[dim]Workflow:[/dim]")
        runner.print("[dim]1. Make changes to prompts on this branch[/dim]")
        runner.print("[dim]2. Test your changes[/dim]")
        runner.print("[dim]3. Switch back to main when done[/dim]")
    else:
        runner.print_warning(f"Branch '{branch_name}' may already exist")


async def demo_switch_branch(runner: "DemoRunner", promptly):
    """Demonstrate switching branches."""
    runner.print()
    runner.print_info("Switching branches")
    runner.print()

    # List available branches
    branches = promptly.branches() if hasattr(promptly, 'branches') else ['main']
    runner.print(f"[dim]Available branches: {', '.join(branches)}[/dim]")

    # Get current branch
    current = promptly.current_branch if hasattr(promptly, 'current_branch') else 'main'
    runner.print(f"[dim]Current branch: [cyan]{current}[/cyan][/dim]")
    runner.print()

    # Get target branch
    branch_name = runner.prompt_input("Branch to switch to", "main")

    # Switch
    with runner.show_progress("Switching branch") as progress:
        task = progress.add_task("Switching...", total=1)
        result = promptly.checkout(branch_name)
        progress.update(task, advance=1)

    if result:
        runner.print_success(f"Switched to branch '{branch_name}'")
    else:
        runner.print_warning(f"Could not switch to branch '{branch_name}'")


async def demo_list_branches(runner: "DemoRunner", promptly):
    """Demonstrate listing all branches."""
    runner.print()
    runner.print_info("Listing all branches")
    runner.print()

    branches = promptly.branches() if hasattr(promptly, 'branches') else ['main']
    current = promptly.current_branch if hasattr(promptly, 'current_branch') else 'main'

    # Build table data
    rows = []
    for branch in branches:
        marker = "→" if branch == current else " "
        status = "[green](current)[/green]" if branch == current else ""
        rows.append([marker, branch, status])

    runner.print_table(
        f"🌿 Branches ({len(branches)})",
        ["", "Branch Name", "Status"],
        rows
    )


async def demo_compare_versions(runner: "DemoRunner", promptly):
    """Demonstrate comparing prompt versions."""
    runner.print()
    runner.print_info("Comparing prompt versions")
    runner.print()

    # List available prompts
    prompts = promptly.list()
    if prompts:
        available = [p['name'] for p in prompts]
        runner.print(f"[dim]Available prompts: {', '.join(available)}[/dim]")

    name = runner.prompt_input("Prompt name", "summarize")

    # Get history
    history = promptly.log(name)

    if not history or len(history) < 2:
        runner.print_warning(f"Need at least 2 versions to compare. Creating versions...")

        # Create some versions for demo
        original = "Summarize the following text:\n\n{{text}}"
        promptly.add(name, original)

        updated = "Summarize the following text in {{length}} sentences:\n\n{{text}}"
        promptly.add(name, updated)

        final = "Summarize the following text in {{length}} sentences.\nBe concise and focus on key points.\n\n{{text}}"
        promptly.add(name, final)

        history = promptly.log(name)
        runner.print_success("Created 3 versions for comparison")

    # Show versions
    runner.print()
    runner.print(f"[bold]Versions of '{name}':[/bold]")

    for i, entry in enumerate(history[:5]):  # Show max 5
        version = entry.get('version', '')[:8]
        runner.print(f"  [{i+1}] v{version}")

    runner.print()

    # Get versions to compare
    v1_num = runner.prompt_input("First version number", "1")
    v2_num = runner.prompt_input("Second version number", str(len(history)))

    try:
        v1_idx = int(v1_num) - 1
        v2_idx = int(v2_num) - 1

        if 0 <= v1_idx < len(history) and 0 <= v2_idx < len(history):
            v1_id = history[v1_idx].get('version')
            v2_id = history[v2_idx].get('version')

            p1 = promptly.get(name, version=v1_id)
            p2 = promptly.get(name, version=v2_id)

            if p1 and p2:
                runner.print()
                runner.print(f"[bold red]--- v{v1_id[:8]} (older)[/bold red]")
                runner.print(f"[bold green]+++ v{v2_id[:8]} (newer)[/bold green]")
                runner.print()

                # Simple diff display
                lines1 = (p1.get('content', '') or '').split('\n')
                lines2 = (p2.get('content', '') or '').split('\n')

                # Show side by side
                runner.print("[bold]Old Version:[/bold]")
                runner.print_prompt(f"{name} (v{v1_id[:8]})", p1.get('content', ''))

                runner.print()
                runner.print("[bold]New Version:[/bold]")
                runner.print_prompt(f"{name} (v{v2_id[:8]})", p2.get('content', ''))

                # Show simple diff stats
                added = len(lines2) - len(lines1)
                runner.print()
                if added > 0:
                    runner.print(f"[green]+{added} lines added[/green]")
                elif added < 0:
                    runner.print(f"[red]{added} lines removed[/red]")
                else:
                    runner.print("[dim]Same line count, content changed[/dim]")

    except (ValueError, IndexError):
        runner.print_warning("Invalid version numbers")


async def demo_full_workflow(runner: "DemoRunner", promptly):
    """Demonstrate a full versioning workflow."""
    runner.print()
    runner.print_info("Full Versioning Workflow")
    runner.print()

    runner.print("[bold]Scenario:[/bold] Improve the 'summarize' prompt")
    runner.print()

    # Step 1: Check current state
    runner.print("[bold cyan]Step 1: Check current state[/bold cyan]")
    current_branch = promptly.current_branch if hasattr(promptly, 'current_branch') else 'main'
    runner.print(f"  Current branch: [cyan]{current_branch}[/cyan]")

    current_prompt = promptly.get("summarize")
    if current_prompt:
        runner.print(f"  Current prompt content:")
        runner.print(f"  [dim]{current_prompt.get('content', '')[:60]}...[/dim]")
    runner.print()

    # Step 2: Create feature branch
    runner.print("[bold cyan]Step 2: Create feature branch[/bold cyan]")
    branch_name = "improve-summarize"
    promptly.branch(branch_name)
    runner.print(f"  Created branch: [green]{branch_name}[/green]")
    runner.print()

    # Step 3: Make improvements
    runner.print("[bold cyan]Step 3: Make improvements on feature branch[/bold cyan]")

    improved_content = """Summarize the following text in {{length}} sentences.

Guidelines:
- Focus on main ideas and key points
- Use clear, concise language
- Maintain the original tone
- Include any critical details

Text to summarize:
{{text}}

Summary:"""

    promptly.add("summarize", improved_content)
    runner.print("  Updated 'summarize' prompt with guidelines")
    runner.print_prompt("summarize (improved)", improved_content)
    runner.print()

    # Step 4: Test the changes
    runner.print("[bold cyan]Step 4: Test changes (simulated)[/bold cyan]")
    runner.print("  [green]✓[/green] Test 1: Short text summarization - PASSED")
    runner.print("  [green]✓[/green] Test 2: Long document summarization - PASSED")
    runner.print("  [green]✓[/green] Test 3: Technical content - PASSED")
    runner.print()

    # Step 5: Switch back to main
    runner.print("[bold cyan]Step 5: Return to main branch[/bold cyan]")
    promptly.checkout("main")
    runner.print(f"  Switched to: [cyan]main[/cyan]")

    main_prompt = promptly.get("summarize")
    runner.print(f"  Main branch still has original prompt")
    runner.print()

    # Step 6: Merge (conceptual)
    runner.print("[bold cyan]Step 6: Merge changes[/bold cyan]")
    runner.print("  [dim]In a real workflow, you would now merge the feature branch.[/dim]")
    runner.print("  [dim]Promptly tracks all versions, so you can always revert.[/dim]")
    runner.print()

    # Show final state
    runner.print("[bold green]Workflow Complete![/bold green]")
    runner.print()
    runner.print("[dim]Key takeaways:[/dim]")
    runner.print("[dim]• Branches isolate experiments from production prompts[/dim]")
    runner.print("[dim]• Every change creates a new version with SHA-256 hash[/dim]")
    runner.print("[dim]• You can compare and revert to any previous version[/dim]")
    runner.print("[dim]• History is preserved even after merging[/dim]")


# Direct execution for testing
if __name__ == "__main__":
    from .interactive_demo import DemoRunner

    async def main():
        runner = DemoRunner()
        await runner.setup_sample_data()
        await run_versioning_demo(runner)

    asyncio.run(main())
