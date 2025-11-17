#!/usr/bin/env python3
"""
Enhanced CLI for Promptly with rich formatting
Provides beautiful tables, progress bars, spinners, and interactive prompts
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import click

# Try importing rich (graceful degradation)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich.prompt import Prompt, Confirm
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import Promptly
sys.path.insert(0, str(Path(__file__).parent.parent))
from promptly import Promptly, PromptlyError


class EnhancedCLI:
    """Enhanced CLI with rich formatting"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.promptly = None

    def print(self, *args, **kwargs):
        """Print with rich if available"""
        if RICH_AVAILABLE and self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def print_error(self, message: str):
        """Print error message"""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"Error: {message}")

    def print_success(self, message: str):
        """Print success message"""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"Success: {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold yellow]⚠[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")

    def print_info(self, message: str):
        """Print info message"""
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold cyan]ℹ[/bold cyan] {message}")
        else:
            print(f"Info: {message}")

    def show_table(self, data: List[Dict], title: str = None, headers: List[str] = None):
        """Display data in a formatted table"""
        if not data:
            self.print_warning("No data to display")
            return

        if RICH_AVAILABLE and self.console:
            table = Table(
                title=title,
                show_header=True,
                header_style="bold cyan",
                border_style="blue",
                box=box.ROUNDED
            )

            # Determine headers
            if headers is None:
                headers = list(data[0].keys())

            # Add columns
            for header in headers:
                table.add_column(header.replace('_', ' ').title(), style="cyan")

            # Add rows
            for row in data:
                table.add_row(*[str(row.get(h, '')) for h in headers])

            self.console.print(table)
        else:
            # Fallback to simple text table
            if title:
                print(f"\n{title}\n")
            for row in data:
                print(row)

    def show_tree(self, root_label: str, items: Dict):
        """Display hierarchical data as a tree"""
        if RICH_AVAILABLE and self.console:
            tree = Tree(f"[bold cyan]{root_label}[/bold cyan]")

            def add_items(parent, items_dict):
                for key, value in items_dict.items():
                    if isinstance(value, dict):
                        node = parent.add(f"[bold]{key}[/bold]")
                        add_items(node, value)
                    elif isinstance(value, list):
                        node = parent.add(f"[bold]{key}[/bold]")
                        for item in value:
                            node.add(str(item))
                    else:
                        parent.add(f"{key}: {value}")

            add_items(tree.root, items)
            self.console.print(tree)
        else:
            # Fallback to simple indented text
            def print_items(items_dict, indent=0):
                for key, value in items_dict.items():
                    if isinstance(value, dict):
                        print("  " * indent + f"{key}:")
                        print_items(value, indent + 1)
                    elif isinstance(value, list):
                        print("  " * indent + f"{key}:")
                        for item in value:
                            print("  " * (indent + 1) + str(item))
                    else:
                        print("  " * indent + f"{key}: {value}")

            print(f"\n{root_label}")
            print_items(items)

    def show_progress(self, task_description: str, items: List[Any], process_func):
        """Show progress bar while processing items"""
        if RICH_AVAILABLE and self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(task_description, total=len(items))

                results = []
                for item in items:
                    result = process_func(item)
                    results.append(result)
                    progress.advance(task)

                return results
        else:
            # Fallback to simple processing
            print(f"{task_description}...")
            results = []
            for i, item in enumerate(items, 1):
                result = process_func(item)
                results.append(result)
                print(f"  Processed {i}/{len(items)}")
            return results

    def show_spinner(self, message: str, work_func):
        """Show spinner while performing work"""
        if RICH_AVAILABLE and self.console:
            with self.console.status(f"[bold green]{message}...", spinner="dots"):
                result = work_func()
            return result
        else:
            print(f"{message}...")
            return work_func()

    def prompt_text(self, message: str, default: str = None) -> str:
        """Prompt for text input"""
        if RICH_AVAILABLE:
            return Prompt.ask(message, default=default)
        else:
            return input(f"{message}: ")

    def prompt_confirm(self, message: str, default: bool = False) -> bool:
        """Prompt for yes/no confirmation"""
        if RICH_AVAILABLE:
            return Confirm.ask(message, default=default)
        else:
            response = input(f"{message} (y/n): ").lower()
            return response in ['y', 'yes']

    def prompt_choice(self, message: str, choices: List[str]) -> str:
        """Prompt for a choice from a list"""
        if RICH_AVAILABLE:
            return Prompt.ask(message, choices=choices)
        else:
            print(f"{message}")
            for i, choice in enumerate(choices, 1):
                print(f"  {i}. {choice}")
            idx = int(input("Select: ")) - 1
            return choices[idx]

    def show_panel(self, content: str, title: str = None, subtitle: str = None):
        """Show content in a panel"""
        if RICH_AVAILABLE and self.console:
            panel = Panel(
                content,
                title=title,
                subtitle=subtitle,
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            if title:
                print(f"\n=== {title} ===")
            print(content)
            if subtitle:
                print(f"--- {subtitle} ---")

    def show_syntax(self, code: str, language: str = "python", line_numbers: bool = True):
        """Display code with syntax highlighting"""
        if RICH_AVAILABLE and self.console:
            syntax = Syntax(
                code,
                language,
                theme="monokai",
                line_numbers=line_numbers,
                word_wrap=True
            )
            self.console.print(syntax)
        else:
            print(code)

    def show_markdown(self, markdown_text: str):
        """Display markdown formatted text"""
        if RICH_AVAILABLE and self.console:
            md = Markdown(markdown_text)
            self.console.print(md)
        else:
            print(markdown_text)

    def show_json(self, data: Dict):
        """Display JSON with syntax highlighting"""
        if RICH_AVAILABLE and self.console:
            self.console.print_json(data=data)
        else:
            print(json.dumps(data, indent=2))

    # Command implementations

    def cmd_status(self):
        """Show repository status with rich formatting"""
        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError as e:
            self.print_error(str(e))
            return

        try:
            branch = self.promptly._get_current_branch()
            prompts = self.promptly.list_prompts()

            # Get branch info
            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM branches")
                branch_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM evaluations")
                eval_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM chains")
                chain_count = cursor.fetchone()[0]

            status_content = f"""[bold cyan]Repository Status[/bold cyan]

[bold]Location:[/bold] {self.promptly.promptly_dir}
[bold]Current Branch:[/bold] [green]{branch}[/green]

[bold]Statistics:[/bold]
  • Prompts: {len(prompts)}
  • Branches: {branch_count}
  • Evaluations: {eval_count}
  • Chains: {chain_count}
"""
            self.show_panel(status_content, title="Promptly Status")

        except Exception as e:
            self.print_error(f"Failed to get status: {e}")

    def cmd_list_prompts(self, branch: str = None):
        """List prompts with rich table"""
        try:
            self.promptly = Promptly()
            prompts = self.promptly.list_prompts(branch=branch)

            if not prompts:
                self.print_warning("No prompts found")
                return

            current_branch = branch or self.promptly._get_current_branch()

            # Format data for display
            display_data = []
            for p in prompts:
                display_data.append({
                    'name': p['name'],
                    'version': f"v{p['version']}",
                    'commit': p['commit_hash'][:8],
                    'created': p['created_at']
                })

            self.show_table(
                display_data,
                title=f"Prompts on branch '{current_branch}'",
                headers=['name', 'version', 'commit', 'created']
            )

        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_show_prompt(self, name: str, version: int = None):
        """Show prompt with syntax highlighting"""
        try:
            self.promptly = Promptly()
            prompt = self.promptly.get(name, version=version)

            if not prompt:
                self.print_error(f"Prompt '{name}' not found")
                return

            # Show metadata panel
            metadata_content = f"""[bold]Branch:[/bold] {prompt['branch']}
[bold]Version:[/bold] {prompt['version']}
[bold]Commit:[/bold] {prompt['commit_hash']}
[bold]Created:[/bold] {prompt['created_at']}"""

            self.show_panel(metadata_content, title=f"Prompt: {prompt['name']}")

            # Show content with syntax highlighting
            self.print("\n[bold cyan]Content:[/bold cyan]\n")
            self.show_syntax(prompt['content'], language="text")

            # Show metadata if present
            if prompt.get('metadata'):
                self.print("\n[bold cyan]Metadata:[/bold cyan]\n")
                self.show_json(prompt['metadata'])

        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_branch_tree(self):
        """Show branch tree"""
        try:
            self.promptly = Promptly()

            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT name, head_commit, created_at FROM branches ORDER BY created_at")
                branches = cursor.fetchall()

                current_branch = self.promptly._get_current_branch()

                branch_data = {}
                for branch_row in branches:
                    name = branch_row[0]
                    commit = branch_row[1][:8]
                    is_current = name == current_branch

                    # Get prompts for this branch
                    prompts = self.promptly.list_prompts(branch=name)

                    branch_data[f"{'[*] ' if is_current else ''}{name} ({commit})"] = {
                        'prompts': [f"{p['name']} v{p['version']}" for p in prompts[:5]]
                    }

                self.show_tree("Branches", branch_data)

        except Exception as e:
            self.print_error(f"Failed to show branches: {e}")

    def cmd_log(self, name: str = None, limit: int = 10):
        """Show commit log with rich formatting"""
        try:
            self.promptly = Promptly()
            commits = self.promptly.log(name=name, limit=limit)

            if not commits:
                self.print_warning("No commits found")
                return

            current_branch = self.promptly._get_current_branch()

            # Format for display
            display_data = []
            for c in commits:
                display_data.append({
                    'commit': c['commit_hash'][:8],
                    'prompt': c['name'],
                    'version': f"v{c['version']}",
                    'branch': c['branch'],
                    'date': c['created_at']
                })

            self.show_table(
                display_data,
                title=f"Commit History on '{current_branch}'",
                headers=['commit', 'prompt', 'version', 'branch', 'date']
            )

        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_diff(self, name: str, version1: int, version2: int):
        """Show diff between two versions"""
        try:
            self.promptly = Promptly()

            p1 = self.promptly.get(name, version=version1)
            p2 = self.promptly.get(name, version=version2)

            if not p1 or not p2:
                self.print_error("One or both versions not found")
                return

            self.show_panel(
                f"[bold]Version {version1}:[/bold] {p1['commit_hash']}\n"
                f"[bold]Version {version2}:[/bold] {p2['commit_hash']}",
                title=f"Comparing {name}"
            )

            self.print("\n[bold cyan]Version {version1}:[/bold cyan]\n")
            self.show_syntax(p1['content'], language="text")

            self.print("\n[bold cyan]Version {version2}:[/bold cyan]\n")
            self.show_syntax(p2['content'], language="text")

        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_export(self, output_file: str, format: str = "json"):
        """Export repository"""
        try:
            self.promptly = Promptly()

            def export_work():
                prompts = self.promptly.list_prompts()

                export_data = {
                    'version': '1.0',
                    'branch': self.promptly._get_current_branch(),
                    'prompts': []
                }

                for p in prompts:
                    prompt_data = self.promptly.get(p['name'])
                    export_data['prompts'].append(prompt_data)

                with open(output_file, 'w') as f:
                    if format == 'json':
                        json.dump(export_data, f, indent=2)
                    else:
                        # YAML export
                        import yaml
                        yaml.dump(export_data, f, default_flow_style=False)

                return len(export_data['prompts'])

            count = self.show_spinner(f"Exporting to {output_file}", export_work)
            self.print_success(f"Exported {count} prompts to {output_file}")

        except Exception as e:
            self.print_error(f"Export failed: {e}")


# Click CLI integration

@click.group()
@click.version_option(version='0.2.0')
def cli():
    """Enhanced Promptly CLI with rich formatting"""
    pass


@cli.command()
def status():
    """Show repository status"""
    enhanced = EnhancedCLI()
    enhanced.cmd_status()


@cli.command(name='list')
@click.option('--branch', '-b', help='List prompts from specific branch')
def list_cmd(branch):
    """List all prompts"""
    enhanced = EnhancedCLI()
    enhanced.cmd_list_prompts(branch=branch)


@cli.command()
@click.argument('name')
@click.option('--version', '-v', type=int, help='Specific version')
def show(name, version):
    """Show prompt details with syntax highlighting"""
    enhanced = EnhancedCLI()
    enhanced.cmd_show_prompt(name, version=version)


@cli.command()
def branches():
    """Show branch tree"""
    enhanced = EnhancedCLI()
    enhanced.cmd_branch_tree()


@cli.command()
@click.option('--name', '-n', help='Show log for specific prompt')
@click.option('--limit', '-l', default=10, help='Number of commits')
def log(name, limit):
    """Show commit history"""
    enhanced = EnhancedCLI()
    enhanced.cmd_log(name=name, limit=limit)


@cli.command()
@click.argument('name')
@click.argument('version1', type=int)
@click.argument('version2', type=int)
def diff(name, version1, version2):
    """Compare two versions of a prompt"""
    enhanced = EnhancedCLI()
    enhanced.cmd_diff(name, version1, version2)


@cli.command()
@click.argument('output_file')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'yaml']))
def export(output_file, format):
    """Export repository"""
    enhanced = EnhancedCLI()
    enhanced.cmd_export(output_file, format=format)


if __name__ == '__main__':
    if not RICH_AVAILABLE:
        print("Warning: rich not available. Install with: pip install rich")
        print("Falling back to basic formatting...\n")

    cli()
