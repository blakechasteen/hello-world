#!/usr/bin/env python3
"""
Interactive REPL mode for Promptly
Provides a rich command-line experience with history, completion, and multi-line editing
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Try importing prompt_toolkit (graceful degradation)
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion, WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.shell import BashLexer
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Try importing rich for better output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import Promptly
sys.path.insert(0, str(Path(__file__).parent.parent))
from promptly import Promptly, PromptlyError


class PromptlyCompleter(Completer):
    """Custom completer for Promptly commands"""

    COMMANDS = [
        'help', 'exit', 'quit',
        'init', 'add', 'get', 'list', 'log',
        'branch', 'checkout', 'branches',
        'eval', 'chain', 'diff',
        'show', 'status', 'search',
        'export', 'import', 'config',
        'clear', 'history'
    ]

    def __init__(self, promptly_instance: Optional[Promptly] = None):
        self.promptly = promptly_instance

    def get_completions(self, document, complete_event):
        """Generate completions based on current input"""
        text = document.text_before_cursor
        words = text.split()

        # No input yet - show all commands
        if not words:
            for cmd in self.COMMANDS:
                yield Completion(cmd, start_position=0, display=cmd)
            return

        current_word = words[-1] if words else ''

        # First word - command completion
        if len(words) == 1:
            for cmd in self.COMMANDS:
                if cmd.startswith(current_word.lower()):
                    yield Completion(
                        cmd,
                        start_position=-len(current_word),
                        display=cmd
                    )

        # Context-aware completion for subsequent words
        elif len(words) >= 2:
            command = words[0].lower()

            # Complete prompt names
            if command in ['get', 'eval', 'log', 'show'] and self.promptly:
                try:
                    prompts = self.promptly.list_prompts()
                    for prompt in prompts:
                        name = prompt['name']
                        if name.startswith(current_word):
                            yield Completion(
                                name,
                                start_position=-len(current_word),
                                display=name,
                                display_meta=f"v{prompt['version']}"
                            )
                except:
                    pass

            # Complete branch names
            elif command in ['checkout', 'branch'] and self.promptly:
                try:
                    with self.promptly._get_db() as db:
                        cursor = db.conn.cursor()
                        cursor.execute("SELECT name FROM branches")
                        branches = cursor.fetchall()
                        for branch_row in branches:
                            branch_name = branch_row[0]
                            if branch_name.startswith(current_word):
                                yield Completion(
                                    branch_name,
                                    start_position=-len(current_word),
                                    display=branch_name
                                )
                except:
                    pass


class InteractiveREPL:
    """Interactive REPL for Promptly"""

    def __init__(self):
        self.promptly = None
        self.console = Console() if RICH_AVAILABLE else None
        self.history_file = Path.home() / '.promptly_history'
        self.running = True

        # Custom style for prompt_toolkit
        self.style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'branch': '#0088ff',
            'error': '#ff0000',
            'success': '#00ff00',
            'warning': '#ffaa00',
        }) if PROMPT_TOOLKIT_AVAILABLE else None

        # Initialize session
        if PROMPT_TOOLKIT_AVAILABLE:
            self.session = PromptSession(
                history=FileHistory(str(self.history_file)),
                auto_suggest=AutoSuggestFromHistory(),
                enable_history_search=True,
            )
        else:
            self.session = None

    def _get_prompt_text(self) -> str:
        """Get formatted prompt text"""
        if not PROMPT_TOOLKIT_AVAILABLE:
            return "promptly> "

        if self.promptly:
            try:
                branch = self.promptly._get_current_branch()
                return HTML(f'<prompt>promptly</prompt> <branch>({branch})</branch>> ')
            except:
                return HTML('<prompt>promptly</prompt>> ')
        return HTML('<prompt>promptly</prompt>> ')

    def print(self, message: str, style: str = None):
        """Print with optional styling"""
        if RICH_AVAILABLE and self.console:
            if style:
                self.console.print(message, style=style)
            else:
                self.console.print(message)
        else:
            print(message)

    def print_error(self, message: str):
        """Print error message"""
        self.print(f"[red]Error:[/red] {message}" if RICH_AVAILABLE else f"Error: {message}")

    def print_success(self, message: str):
        """Print success message"""
        self.print(f"[green]{message}[/green]" if RICH_AVAILABLE else message)

    def print_warning(self, message: str):
        """Print warning message"""
        self.print(f"[yellow]{message}[/yellow]" if RICH_AVAILABLE else message)

    def print_table(self, data: List[Dict], title: str = None):
        """Print data as a table"""
        if not data:
            self.print_warning("No data to display")
            return

        if RICH_AVAILABLE and self.console:
            table = Table(title=title, show_header=True, header_style="bold cyan")

            # Add columns from first row
            first_row = data[0]
            for key in first_row.keys():
                table.add_column(key.replace('_', ' ').title())

            # Add rows
            for row in data:
                table.add_row(*[str(v) for v in row.values()])

            self.console.print(table)
        else:
            # Fallback to simple text table
            if title:
                print(f"\n{title}\n")
            for row in data:
                print(row)

    def show_welcome(self):
        """Show welcome message"""
        if RICH_AVAILABLE and self.console:
            welcome = Panel(
                "[bold cyan]Promptly Interactive Mode[/bold cyan]\n\n"
                "Type [bold]help[/bold] for available commands\n"
                "Type [bold]exit[/bold] or press Ctrl+D to quit\n\n"
                "[dim]Features: Command history, auto-completion, multi-line editing[/dim]",
                title="Welcome",
                border_style="cyan"
            )
            self.console.print(welcome)
        else:
            print("\n=== Promptly Interactive Mode ===")
            print("Type 'help' for available commands")
            print("Type 'exit' or press Ctrl+D to quit\n")

    def show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Repository Management:[/bold]
  init                    Initialize a new promptly repository
  status                  Show repository status
  config                  Show configuration

[bold]Prompt Management:[/bold]
  add <name> <content>    Add or update a prompt
  get <name>              Get a prompt
  list                    List all prompts
  show <name>             Show prompt details with syntax highlighting
  search <query>          Search prompts by content
  log [name]              Show commit history

[bold]Branch Management:[/bold]
  branch <name>           Create a new branch
  checkout <name>         Switch to a branch
  branches                List all branches

[bold]Evaluation:[/bold]
  eval <name> <file>      Evaluate a prompt with test cases

[bold]Chains:[/bold]
  chain create <name> <steps...>    Create a chain
  chain run <name> <input>          Run a chain

[bold]Utilities:[/bold]
  diff <name> <v1> <v2>   Compare two versions
  export <format>         Export repository
  import <file>           Import prompts
  clear                   Clear screen
  history                 Show command history
  help                    Show this help
  exit/quit               Exit interactive mode

[bold cyan]Keyboard Shortcuts:[/bold cyan]
  Ctrl+D                  Exit
  Ctrl+R                  Reverse search history
  Ctrl+C                  Cancel current input
  Tab                     Auto-complete
  Up/Down                 Navigate history
"""
        if RICH_AVAILABLE and self.console:
            self.console.print(Markdown(help_text))
        else:
            print(help_text.replace('[bold]', '').replace('[/bold]', '')
                          .replace('[bold cyan]', '').replace('[/bold cyan]', ''))

    def cmd_init(self, args: List[str]):
        """Initialize repository"""
        try:
            self.promptly = Promptly()
            message = self.promptly.init()
            self.print_success(message)
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_status(self, args: List[str]):
        """Show repository status"""
        if not self.promptly:
            try:
                self.promptly = Promptly()
                self.promptly._check_init()
            except PromptlyError as e:
                self.print_error(str(e))
                return

        try:
            branch = self.promptly._get_current_branch()
            prompts = self.promptly.list_prompts()

            if RICH_AVAILABLE and self.console:
                panel = Panel(
                    f"[bold]Current Branch:[/bold] {branch}\n"
                    f"[bold]Total Prompts:[/bold] {len(prompts)}\n"
                    f"[bold]Repository:[/bold] {self.promptly.promptly_dir}",
                    title="Repository Status",
                    border_style="cyan"
                )
                self.console.print(panel)
            else:
                print(f"\nCurrent Branch: {branch}")
                print(f"Total Prompts: {len(prompts)}")
                print(f"Repository: {self.promptly.promptly_dir}\n")
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_add(self, args: List[str]):
        """Add a prompt"""
        if len(args) < 2:
            self.print_error("Usage: add <name> <content>")
            return

        name = args[0]
        content = ' '.join(args[1:])

        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            message = self.promptly.add(name, content)
            self.print_success(message)
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_get(self, args: List[str]):
        """Get a prompt"""
        if len(args) < 1:
            self.print_error("Usage: get <name> [--version V | --commit C]")
            return

        name = args[0]
        version = None
        commit = None

        # Parse options
        i = 1
        while i < len(args):
            if args[i] == '--version' and i + 1 < len(args):
                version = int(args[i + 1])
                i += 2
            elif args[i] == '--commit' and i + 1 < len(args):
                commit = args[i + 1]
                i += 2
            else:
                i += 1

        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            result = self.promptly.get(name, version=version, commit_hash=commit)
            if not result:
                self.print_warning(f"Prompt '{name}' not found")
                return

            if RICH_AVAILABLE and self.console:
                self.console.print(f"\n[bold cyan]Prompt:[/bold cyan] {result['name']}")
                self.console.print(f"[bold]Branch:[/bold] {result['branch']}")
                self.console.print(f"[bold]Version:[/bold] {result['version']}")
                self.console.print(f"[bold]Commit:[/bold] {result['commit_hash']}")
                self.console.print(f"[bold]Created:[/bold] {result['created_at']}\n")

                syntax = Syntax(result['content'], "text", theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="Content", border_style="cyan"))

                if result['metadata']:
                    self.console.print(f"\n[bold]Metadata:[/bold]")
                    self.console.print_json(data=result['metadata'])
            else:
                print(f"\nPrompt: {result['name']}")
                print(f"Branch: {result['branch']}")
                print(f"Version: {result['version']}")
                print(f"Commit: {result['commit_hash']}")
                print(f"Created: {result['created_at']}\n")
                print("Content:")
                print(result['content'])
                if result['metadata']:
                    print("\nMetadata:")
                    print(json.dumps(result['metadata'], indent=2))
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_list(self, args: List[str]):
        """List all prompts"""
        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            prompts = self.promptly.list_prompts()
            if not prompts:
                self.print_warning("No prompts found")
                return

            self.print_table(prompts, title=f"Prompts on branch '{self.promptly._get_current_branch()}'")
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_log(self, args: List[str]):
        """Show commit history"""
        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        name = args[0] if args else None
        limit = 10

        try:
            commits = self.promptly.log(name, limit)
            if not commits:
                self.print_warning("No commits found")
                return

            self.print_table(commits, title=f"Commit History on '{self.promptly._get_current_branch()}'")
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_branches(self, args: List[str]):
        """List all branches"""
        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT name, head_commit, created_at FROM branches ORDER BY name")
                branches = cursor.fetchall()

                current_branch = self.promptly._get_current_branch()

                if RICH_AVAILABLE and self.console:
                    tree = Tree("[bold cyan]Branches[/bold cyan]")
                    for branch_row in branches:
                        name = branch_row[0]
                        marker = "[green]*[/green] " if name == current_branch else "  "
                        tree.add(f"{marker}[bold]{name}[/bold] ({branch_row[1][:8]})")
                    self.console.print(tree)
                else:
                    print("\nBranches:")
                    for branch_row in branches:
                        name = branch_row[0]
                        marker = "* " if name == current_branch else "  "
                        print(f"{marker}{name} ({branch_row[1][:8]})")
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_branch(self, args: List[str]):
        """Create a branch"""
        if len(args) < 1:
            self.print_error("Usage: branch <name>")
            return

        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            message = self.promptly.branch(args[0])
            self.print_success(message)
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_checkout(self, args: List[str]):
        """Checkout a branch"""
        if len(args) < 1:
            self.print_error("Usage: checkout <name>")
            return

        if not self.promptly:
            try:
                self.promptly = Promptly()
            except:
                pass

        try:
            message = self.promptly.checkout(args[0])
            self.print_success(message)
        except PromptlyError as e:
            self.print_error(str(e))

    def cmd_clear(self, args: List[str]):
        """Clear screen"""
        os.system('clear' if os.name != 'nt' else 'cls')

    def cmd_history(self, args: List[str]):
        """Show command history"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                lines = f.readlines()
                recent = lines[-20:] if len(lines) > 20 else lines

                if RICH_AVAILABLE and self.console:
                    self.console.print("[bold cyan]Recent Commands:[/bold cyan]")
                    for i, line in enumerate(recent, 1):
                        self.console.print(f"  {i}. {line.strip()}")
                else:
                    print("\nRecent Commands:")
                    for i, line in enumerate(recent, 1):
                        print(f"  {i}. {line.strip()}")
        else:
            self.print_warning("No command history found")

    def process_command(self, line: str):
        """Process a command line"""
        line = line.strip()
        if not line:
            return

        parts = line.split()
        command = parts[0].lower()
        args = parts[1:]

        # Route to command handlers
        if command in ['exit', 'quit']:
            self.running = False
            self.print_success("Goodbye!")
        elif command == 'help':
            self.show_help()
        elif command == 'init':
            self.cmd_init(args)
        elif command == 'status':
            self.cmd_status(args)
        elif command == 'add':
            self.cmd_add(args)
        elif command == 'get':
            self.cmd_get(args)
        elif command == 'list':
            self.cmd_list(args)
        elif command == 'log':
            self.cmd_log(args)
        elif command == 'branch':
            self.cmd_branch(args)
        elif command == 'branches':
            self.cmd_branches(args)
        elif command == 'checkout':
            self.cmd_checkout(args)
        elif command == 'clear':
            self.cmd_clear(args)
        elif command == 'history':
            self.cmd_history(args)
        else:
            self.print_error(f"Unknown command: {command}. Type 'help' for available commands.")

    def run(self):
        """Run the REPL"""
        self.show_welcome()

        # Try to load existing repository
        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError:
            self.print_warning("No promptly repository found. Use 'init' to create one.")

        # Update completer with promptly instance
        completer = PromptlyCompleter(self.promptly) if PROMPT_TOOLKIT_AVAILABLE else None

        while self.running:
            try:
                if PROMPT_TOOLKIT_AVAILABLE and self.session:
                    line = self.session.prompt(
                        self._get_prompt_text(),
                        completer=completer,
                        style=self.style
                    )
                else:
                    line = input(self._get_prompt_text())

                self.process_command(line)

            except KeyboardInterrupt:
                self.print_warning("\nUse 'exit' or Ctrl+D to quit")
                continue
            except EOFError:
                self.print_success("\nGoodbye!")
                break
            except Exception as e:
                self.print_error(f"Unexpected error: {e}")


def main():
    """Entry point for interactive mode"""
    if not PROMPT_TOOLKIT_AVAILABLE:
        print("Warning: prompt_toolkit not available. Install with: pip install prompt_toolkit")
        print("Falling back to basic mode...\n")

    if not RICH_AVAILABLE:
        print("Warning: rich not available. Install with: pip install rich")
        print("Output will be less colorful...\n")

    repl = InteractiveREPL()
    repl.run()


if __name__ == '__main__':
    main()
