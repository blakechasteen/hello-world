#!/usr/bin/env python3
"""
Setup Wizards for Promptly
Interactive step-by-step guides for common tasks
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import click

# Try importing rich
try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import Promptly
sys.path.insert(0, str(Path(__file__).parent.parent))
from promptly import Promptly, PromptlyError


class BaseWizard:
    """Base class for all wizards"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.promptly = None

    def print(self, *args, **kwargs):
        """Print with rich if available"""
        if RICH_AVAILABLE and self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args)

    def print_success(self, message: str):
        """Print success message"""
        if RICH_AVAILABLE:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"Success: {message}")

    def print_error(self, message: str):
        """Print error message"""
        if RICH_AVAILABLE:
            self.console.print(f"[bold red]✗[/bold red] {message}")
        else:
            print(f"Error: {message}")

    def print_step(self, step: int, total: int, message: str):
        """Print step indicator"""
        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]Step {step}/{total}:[/bold cyan] {message}\n")
        else:
            print(f"\nStep {step}/{total}: {message}\n")

    def show_panel(self, content: str, title: str = None):
        """Show content in panel"""
        if RICH_AVAILABLE and self.console:
            panel = Panel(content, title=title, border_style="cyan")
            self.console.print(panel)
        else:
            if title:
                print(f"\n=== {title} ===")
            print(content)

    def prompt_text(self, message: str, default: str = None) -> str:
        """Prompt for text input"""
        if RICH_AVAILABLE:
            return Prompt.ask(message, default=default)
        else:
            prompt = f"{message}"
            if default:
                prompt += f" [{default}]"
            prompt += ": "
            result = input(prompt)
            return result if result else default

    def prompt_confirm(self, message: str, default: bool = True) -> bool:
        """Prompt for confirmation"""
        if RICH_AVAILABLE:
            return Confirm.ask(message, default=default)
        else:
            default_str = "Y/n" if default else "y/N"
            response = input(f"{message} ({default_str}): ").lower()
            if not response:
                return default
            return response in ['y', 'yes']

    def prompt_int(self, message: str, default: int = None) -> int:
        """Prompt for integer input"""
        if RICH_AVAILABLE:
            return IntPrompt.ask(message, default=default)
        else:
            prompt = f"{message}"
            if default is not None:
                prompt += f" [{default}]"
            prompt += ": "
            result = input(prompt)
            return int(result) if result else default


class ProjectWizard(BaseWizard):
    """Wizard for creating new Promptly projects"""

    def run(self):
        """Run the project setup wizard"""
        self.show_panel(
            "[bold cyan]Promptly Project Setup Wizard[/bold cyan]\n\n"
            "This wizard will guide you through setting up a new Promptly project.",
            title="Welcome"
        )

        # Step 1: Project location
        self.print_step(1, 5, "Choose project location")
        project_dir = self.prompt_text(
            "Project directory",
            default=os.getcwd()
        )
        project_path = Path(project_dir)

        if not project_path.exists():
            if self.prompt_confirm(f"Directory '{project_dir}' does not exist. Create it?"):
                project_path.mkdir(parents=True)
            else:
                self.print_error("Project setup cancelled")
                return

        # Step 2: Initialize repository
        self.print_step(2, 5, "Initialize Promptly repository")

        try:
            self.promptly = Promptly(root_dir=str(project_path))
            message = self.promptly.init()
            self.print_success(message)
        except RepositoryExistsError:
            self.print_error("Promptly repository already exists in this directory")
            if not self.prompt_confirm("Continue with existing repository?"):
                return
            self.promptly = Promptly(root_dir=str(project_path))
        except Exception as e:
            self.print_error(f"Failed to initialize: {e}")
            return

        # Step 3: Create initial prompts
        self.print_step(3, 5, "Create initial prompts")

        if self.prompt_confirm("Would you like to create some starter prompts?"):
            self._create_starter_prompts()

        # Step 4: Set up branches
        self.print_step(4, 5, "Set up branches")

        if self.prompt_confirm("Create development and production branches?"):
            try:
                self.promptly.branch('development')
                self.print_success("Created 'development' branch")

                self.promptly.branch('production')
                self.print_success("Created 'production' branch")
            except Exception as e:
                self.print_error(f"Failed to create branches: {e}")

        # Step 5: Create config file
        self.print_step(5, 5, "Create configuration file")

        if self.prompt_confirm("Create a configuration file?"):
            self._create_config_file(project_path)

        # Summary
        self.show_panel(
            f"[bold green]✓ Project setup complete![/bold green]\n\n"
            f"Location: {project_path}\n\n"
            f"Next steps:\n"
            f"  • cd {project_path}\n"
            f"  • promptly list  # View prompts\n"
            f"  • promptly add <name> <content>  # Add prompts\n"
            f"  • promptly-tui  # Launch TUI interface",
            title="Setup Complete"
        )

    def _create_starter_prompts(self):
        """Create starter prompts"""
        starters = [
            {
                'name': 'summarizer',
                'content': 'Summarize the following text in 2-3 sentences:\n\n{text}',
                'metadata': {'category': 'text-processing', 'tags': ['summarization']}
            },
            {
                'name': 'translator',
                'content': 'Translate the following text to {target_language}:\n\n{text}',
                'metadata': {'category': 'translation', 'tags': ['i18n']}
            },
            {
                'name': 'code_reviewer',
                'content': 'Review the following code and provide feedback:\n\n```{language}\n{code}\n```\n\nFocus on: {focus_areas}',
                'metadata': {'category': 'code', 'tags': ['review', 'quality']}
            }
        ]

        for starter in starters:
            if self.prompt_confirm(f"Create '{starter['name']}' prompt?", default=True):
                try:
                    self.promptly.add(
                        starter['name'],
                        starter['content'],
                        starter['metadata']
                    )
                    self.print_success(f"Created prompt: {starter['name']}")
                except Exception as e:
                    self.print_error(f"Failed to create {starter['name']}: {e}")

    def _create_config_file(self, project_path: Path):
        """Create configuration file"""
        config = {
            'promptly': {
                'version': '1.0',
                'default_branch': 'main',
                'evaluators': {
                    'default': 'promptly_default'
                }
            },
            'integrations': {
                'hololoom': {
                    'enabled': False,
                    'mode': 'fast'
                }
            }
        }

        config_path = project_path / 'promptly_config.yaml'

        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.print_success(f"Created config file: {config_path}")
        except Exception as e:
            self.print_error(f"Failed to create config: {e}")


class PromptWizard(BaseWizard):
    """Wizard for creating prompts"""

    def run(self):
        """Run the prompt creation wizard"""
        self.show_panel(
            "[bold cyan]Prompt Creation Wizard[/bold cyan]\n\n"
            "This wizard will guide you through creating a new prompt.",
            title="Create Prompt"
        )

        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError as e:
            self.print_error(str(e))
            return

        # Step 1: Basic info
        self.print_step(1, 4, "Basic Information")

        name = self.prompt_text("Prompt name")
        if not name:
            self.print_error("Name is required")
            return

        description = self.prompt_text("Description (optional)", default="")

        # Step 2: Content
        self.print_step(2, 4, "Prompt Content")

        self.print("[dim]Enter prompt content. Use {variable} for placeholders.[/dim]")
        self.print("[dim]Examples: {text}, {language}, {query}[/dim]\n")

        if self.prompt_confirm("Use multi-line editor?", default=False):
            content = self._multiline_editor()
        else:
            content = self.prompt_text("Content")

        if not content:
            self.print_error("Content is required")
            return

        # Step 3: Metadata
        self.print_step(3, 4, "Metadata")

        metadata = {}

        if self.prompt_confirm("Add metadata?", default=True):
            category = self.prompt_text("Category (optional)", default="general")
            if category:
                metadata['category'] = category

            tags_str = self.prompt_text("Tags (comma-separated, optional)", default="")
            if tags_str:
                metadata['tags'] = [tag.strip() for tag in tags_str.split(',')]

            if description:
                metadata['description'] = description

        # Step 4: Confirm and save
        self.print_step(4, 4, "Review and Save")

        self.show_panel(
            f"[bold]Name:[/bold] {name}\n"
            f"[bold]Description:[/bold] {description or 'None'}\n\n"
            f"[bold]Content:[/bold]\n{content}\n\n"
            f"[bold]Metadata:[/bold] {metadata or 'None'}",
            title="Review"
        )

        if self.prompt_confirm("Save this prompt?", default=True):
            try:
                message = self.promptly.add(name, content, metadata)
                self.print_success(message)
            except Exception as e:
                self.print_error(f"Failed to save prompt: {e}")

    def _multiline_editor(self) -> str:
        """Simple multi-line editor"""
        self.print("[dim]Enter content (type 'END' on a new line to finish):[/dim]\n")

        lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)

        return '\n'.join(lines)


class ChainWizard(BaseWizard):
    """Wizard for creating prompt chains"""

    def run(self):
        """Run the chain creation wizard"""
        self.show_panel(
            "[bold cyan]Chain Creation Wizard[/bold cyan]\n\n"
            "This wizard will guide you through creating a prompt chain.",
            title="Create Chain"
        )

        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError as e:
            self.print_error(str(e))
            return

        # Step 1: Chain name
        self.print_step(1, 3, "Basic Information")

        chain_name = self.prompt_text("Chain name")
        if not chain_name:
            self.print_error("Name is required")
            return

        description = self.prompt_text("Description (optional)", default="")

        # Step 2: Select prompts
        self.print_step(2, 3, "Select Prompts")

        prompts = self.promptly.list_prompts()

        if not prompts:
            self.print_error("No prompts available. Create prompts first.")
            return

        self.print("\nAvailable prompts:")
        for i, p in enumerate(prompts, 1):
            self.print(f"  {i}. {p['name']} (v{p['version']})")

        steps = []
        self.print("\n[dim]Enter prompt numbers to add to chain (comma-separated):[/dim]")
        indices_str = self.prompt_text("Prompt numbers")

        if indices_str:
            try:
                indices = [int(i.strip()) - 1 for i in indices_str.split(',')]
                steps = [prompts[i]['name'] for i in indices if 0 <= i < len(prompts)]
            except (ValueError, IndexError):
                self.print_error("Invalid prompt numbers")
                return

        if not steps:
            self.print_error("At least one prompt is required")
            return

        # Step 3: Review and save
        self.print_step(3, 3, "Review and Save")

        self.show_panel(
            f"[bold]Name:[/bold] {chain_name}\n"
            f"[bold]Description:[/bold] {description or 'None'}\n\n"
            f"[bold]Steps:[/bold]\n" + ' → '.join(steps),
            title="Review Chain"
        )

        if self.prompt_confirm("Create this chain?", default=True):
            try:
                message = self.promptly.create_chain(chain_name, steps, description)
                self.print_success(message)
            except Exception as e:
                self.print_error(f"Failed to create chain: {e}")


class EvaluationWizard(BaseWizard):
    """Wizard for setting up evaluations"""

    def run(self):
        """Run the evaluation setup wizard"""
        self.show_panel(
            "[bold cyan]Evaluation Setup Wizard[/bold cyan]\n\n"
            "This wizard will guide you through setting up prompt evaluations.",
            title="Setup Evaluation"
        )

        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError as e:
            self.print_error(str(e))
            return

        # Step 1: Select prompt
        self.print_step(1, 3, "Select Prompt")

        prompts = self.promptly.list_prompts()

        if not prompts:
            self.print_error("No prompts available. Create prompts first.")
            return

        self.print("\nAvailable prompts:")
        for i, p in enumerate(prompts, 1):
            self.print(f"  {i}. {p['name']} (v{p['version']})")

        prompt_idx = self.prompt_int("Select prompt number") - 1

        if prompt_idx < 0 or prompt_idx >= len(prompts):
            self.print_error("Invalid selection")
            return

        prompt_name = prompts[prompt_idx]['name']
        prompt_data = self.promptly.get(prompt_name)

        # Step 2: Create test cases
        self.print_step(2, 3, "Create Test Cases")

        self.print(f"\n[bold]Prompt content:[/bold]\n{prompt_data['content']}\n")

        # Extract variables from prompt
        import re
        variables = re.findall(r'\{(\w+)\}', prompt_data['content'])

        if variables:
            self.print(f"[dim]Detected variables: {', '.join(variables)}[/dim]\n")

        test_cases = []
        num_tests = self.prompt_int("Number of test cases to create", default=3)

        for i in range(num_tests):
            self.print(f"\n[cyan]Test case {i + 1}:[/cyan]")

            inputs = {}
            for var in variables:
                value = self.prompt_text(f"  {var}")
                inputs[var] = value

            expected = self.prompt_text("  Expected output (optional)", default="")

            test_case = {
                'id': f"test_{i + 1}",
                'inputs': inputs,
                'expected': expected
            }
            test_cases.append(test_case)

        # Step 3: Save test file
        self.print_step(3, 3, "Save Test Cases")

        test_file = self.prompt_text(
            "Test file name",
            default=f"{prompt_name}_tests.json"
        )

        try:
            with open(test_file, 'w') as f:
                json.dump(test_cases, f, indent=2)

            self.print_success(f"Saved test cases to: {test_file}")

            self.print("\n[bold]To run evaluation:[/bold]")
            self.print(f"  promptly eval run {prompt_name} {test_file}")

        except Exception as e:
            self.print_error(f"Failed to save test cases: {e}")


class TemplateWizard(BaseWizard):
    """Wizard for creating prompt templates"""

    TEMPLATES = {
        'text_processing': {
            'name': 'Text Processing Template',
            'content': 'Process the following text according to {instruction}:\n\n{text}\n\nProvide output in {format} format.',
            'variables': ['instruction', 'text', 'format']
        },
        'code_generation': {
            'name': 'Code Generation Template',
            'content': 'Generate {language} code for the following task:\n\n{task}\n\nRequirements:\n{requirements}\n\nProvide clean, well-commented code.',
            'variables': ['language', 'task', 'requirements']
        },
        'analysis': {
            'name': 'Analysis Template',
            'content': 'Analyze the following {subject}:\n\n{content}\n\nProvide analysis on:\n{analysis_points}',
            'variables': ['subject', 'content', 'analysis_points']
        },
        'qa': {
            'name': 'Q&A Template',
            'content': 'Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:',
            'variables': ['context', 'question']
        }
    }

    def run(self):
        """Run the template wizard"""
        self.show_panel(
            "[bold cyan]Template Wizard[/bold cyan]\n\n"
            "Select a template to create a new prompt based on common patterns.",
            title="Create from Template"
        )

        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError as e:
            self.print_error(str(e))
            return

        # Show template options
        self.print("\nAvailable templates:")
        templates = list(self.TEMPLATES.items())

        for i, (key, template) in enumerate(templates, 1):
            self.print(f"  {i}. {template['name']}")
            self.print(f"     [dim]Variables: {', '.join(template['variables'])}[/dim]")

        # Select template
        choice = self.prompt_int("Select template", default=1) - 1

        if choice < 0 or choice >= len(templates):
            self.print_error("Invalid selection")
            return

        template_key, template = templates[choice]

        # Customize
        self.print(f"\n[bold]Template: {template['name']}[/bold]\n")
        self.print(f"[dim]Content preview:[/dim]\n{template['content']}\n")

        name = self.prompt_text("Prompt name")
        if not name:
            self.print_error("Name is required")
            return

        if self.prompt_confirm("Customize content?", default=False):
            content = self.prompt_text("Modified content", default=template['content'])
        else:
            content = template['content']

        # Save
        try:
            metadata = {
                'template': template_key,
                'category': template_key.replace('_', '-')
            }
            message = self.promptly.add(name, content, metadata)
            self.print_success(message)
        except Exception as e:
            self.print_error(f"Failed to create prompt: {e}")


# CLI Commands

@click.group()
def cli():
    """Promptly wizards - interactive setup guides"""
    pass


@cli.command()
def project():
    """Project setup wizard"""
    wizard = ProjectWizard()
    wizard.run()


@cli.command()
def prompt():
    """Prompt creation wizard"""
    wizard = PromptWizard()
    wizard.run()


@cli.command()
def chain():
    """Chain creation wizard"""
    wizard = ChainWizard()
    wizard.run()


@cli.command()
def evaluation():
    """Evaluation setup wizard"""
    wizard = EvaluationWizard()
    wizard.run()


@cli.command()
def template():
    """Template-based prompt creation wizard"""
    wizard = TemplateWizard()
    wizard.run()


if __name__ == '__main__':
    if not RICH_AVAILABLE:
        print("Warning: rich not available. Install with: pip install rich")
        print("Wizards will work but with limited formatting...\n")

    cli()
