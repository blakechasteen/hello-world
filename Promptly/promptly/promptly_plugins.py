#!/usr/bin/env python3
"""
Promptly with Plugin Support - Refactored version using plugin architecture
"""

import click
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import plugin system
from plugins import get_storage_backend, get_evaluator, list_plugins


class Promptly:
    """Main Promptly class with plugin support"""

    def __init__(self, root_dir: str = None, storage_backend: str = None):
        if root_dir is None:
            root_dir = "."

        self.root_dir = Path(root_dir)
        self.promptly_dir = self.root_dir / ".promptly"
        self.backend_config_file = self.promptly_dir / "backend.json"
        self.storage = None
        self.storage_backend_name = storage_backend

        # Load backend config if exists
        if self.backend_config_file.exists():
            with open(self.backend_config_file, 'r') as f:
                config = json.load(f)
                self.storage_backend_name = config.get('storage_backend', 'sqlite')

    def init(self, storage_backend: str = "sqlite"):
        """Initialize a new promptly repository with specified storage backend"""
        if self.promptly_dir.exists():
            raise Exception("Promptly repository already initialized")

        self.promptly_dir.mkdir(parents=True)

        # Save backend configuration
        with open(self.backend_config_file, 'w') as f:
            json.dump({'storage_backend': storage_backend}, f)

        # Initialize storage backend
        if storage_backend == "sqlite":
            storage_path = str(self.promptly_dir / "promptly.db")
        else:
            storage_path = str(self.promptly_dir)

        storage = get_storage_backend(storage_backend)
        storage.init_storage(storage_path)
        storage.close()

        return f"Initialized empty Promptly repository with {storage_backend} backend"

    def _check_init(self):
        """Check if promptly is initialized"""
        if not self.promptly_dir.exists():
            raise Exception("Not a promptly repository. Run 'promptly init' first.")

    def _get_storage(self):
        """Get storage backend instance"""
        if self.storage is None:
            self._check_init()

            # Load backend config
            backend = self.storage_backend_name or "sqlite"
            if self.backend_config_file.exists():
                with open(self.backend_config_file, 'r') as f:
                    config = json.load(f)
                    backend = config.get('storage_backend', 'sqlite')

            # Initialize storage
            if backend == "sqlite":
                storage_path = str(self.promptly_dir / "promptly.db")
            else:
                storage_path = str(self.promptly_dir)

            self.storage = get_storage_backend(backend)
            self.storage.init_storage(storage_path)

        return self.storage

    def add(self, name: str, content: str, metadata: Dict = None):
        """Add a new prompt or update existing one"""
        storage = self._get_storage()
        current_branch = storage.get_current_branch()

        prompt_data = {
            'name': name,
            'content': content,
            'branch': current_branch,
            'metadata': metadata or {}
        }

        commit_hash = storage.save_prompt(prompt_data)

        # Save to file for easy viewing
        prompts_dir = self.promptly_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        prompt = storage.get_prompt(name, branch=current_branch)
        if prompt:
            prompt_file = prompts_dir / f"{name}.yaml"
            with open(prompt_file, 'w') as f:
                yaml.dump(prompt, f, default_flow_style=False)

        return f"Added prompt '{name}' (v{prompt['version']}) on branch '{current_branch}' [{commit_hash}]"

    def get(self, name: str, version: int = None, commit_hash: str = None):
        """Get a prompt by name, optionally at specific version or commit"""
        storage = self._get_storage()
        current_branch = storage.get_current_branch()

        return storage.get_prompt(name, branch=current_branch, version=version, commit_hash=commit_hash)

    def list_prompts(self, branch: str = None):
        """List all prompts on current or specified branch"""
        storage = self._get_storage()

        if branch is None:
            branch = storage.get_current_branch()

        return storage.list_prompts(branch)

    def branch(self, branch_name: str, from_branch: str = None):
        """Create a new branch"""
        storage = self._get_storage()

        if from_branch is None:
            from_branch = storage.get_current_branch()

        storage.create_branch(branch_name, from_branch)

        return f"Created branch '{branch_name}' from '{from_branch}'"

    def checkout(self, branch_name: str):
        """Switch to a different branch"""
        storage = self._get_storage()
        storage.set_current_branch(branch_name)

        return f"Switched to branch '{branch_name}'"

    def log(self, name: str = None, limit: int = 10):
        """Show commit history"""
        storage = self._get_storage()
        current_branch = storage.get_current_branch()

        return storage.get_commit_history(name, current_branch, limit)

    def eval_prompt(self, name: str, test_cases: List[Dict], model_func=None, evaluator_name: str = "keyword"):
        """Evaluate a prompt against test cases using specified evaluator"""
        storage = self._get_storage()

        prompt = self.get(name)
        if not prompt:
            raise Exception(f"Prompt '{name}' not found")

        # Get evaluator plugin
        evaluator = get_evaluator(evaluator_name)

        results = []

        for test_case in test_cases:
            # Format prompt with test inputs
            formatted_prompt = prompt['content'].format(**test_case.get('inputs', {}))

            # Run through model if provided
            actual = None
            if model_func:
                actual = model_func(formatted_prompt)

            # Calculate score using evaluator
            score = None
            metrics = {}
            if actual:
                expected = test_case.get('expected', '')
                context = test_case.get('context', {})
                score = evaluator.evaluate(actual, expected, context)
                metrics = evaluator.get_metrics(actual, expected, context)

            result = {
                'test_case': test_case,
                'formatted_prompt': formatted_prompt,
                'actual': actual,
                'score': score,
                'metrics': metrics
            }
            results.append(result)

            # Save to storage
            eval_data = {
                'prompt_name': name,
                'commit_hash': prompt['commit_hash'],
                'test_case': test_case,
                'expected': test_case.get('expected'),
                'actual': actual,
                'score': score,
                'metrics': metrics,
                'evaluator_name': evaluator_name
            }
            storage.save_evaluation(eval_data)

        return results

    def create_chain(self, name: str, steps: List[str], description: str = None):
        """Create a prompt chain"""
        storage = self._get_storage()

        # Verify all prompts exist
        for step in steps:
            if not self.get(step):
                raise Exception(f"Prompt '{step}' not found")

        chain_data = {
            'name': name,
            'steps': steps,
            'description': description
        }

        storage.save_chain(chain_data)

        # Save chain definition to file
        chains_dir = self.promptly_dir / "chains"
        chains_dir.mkdir(exist_ok=True)

        chain_file = chains_dir / f"{name}.yaml"
        with open(chain_file, 'w') as f:
            yaml.dump(chain_data, f, default_flow_style=False)

        return f"Created chain '{name}' with {len(steps)} steps"

    def execute_chain(self, name: str, initial_input: Dict, model_func=None):
        """Execute a prompt chain"""
        storage = self._get_storage()

        chain = storage.get_chain(name)
        if not chain:
            raise Exception(f"Chain '{name}' not found")

        steps = chain['steps']

        current_input = initial_input
        results = []

        for step in steps:
            prompt = self.get(step)
            if not prompt:
                raise Exception(f"Prompt '{step}' not found in chain")

            # Format prompt with current input
            formatted_prompt = prompt['content'].format(**current_input)

            # Execute with model if provided
            output = None
            if model_func:
                output = model_func(formatted_prompt)

            step_result = {
                'step': step,
                'prompt': formatted_prompt,
                'output': output
            }
            results.append(step_result)

            # Pass output as input to next step
            current_input = {'output': output, **current_input}

        return results

    def close(self):
        """Close storage backend"""
        if self.storage:
            self.storage.close()


# CLI Commands with Plugin Support

@click.group()
@click.version_option(version='0.2.0')
def cli():
    """Promptly - Manage prompts with versioning, branching, eval, and chaining (Plugin Edition)"""
    pass


@cli.command()
@click.option('--storage', '-s', default='sqlite', help='Storage backend (sqlite, json)')
def init(storage):
    """Initialize a new promptly repository"""
    try:
        promptly = Promptly()
        message = promptly.init(storage_backend=storage)
        click.echo(click.style(message, fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command()
def plugins_cmd():
    """List available plugins"""
    try:
        plugins = list_plugins()

        click.echo(click.style("\nAvailable Plugins:", fg='cyan', bold=True))

        click.echo(click.style("\nEvaluators:", fg='yellow', bold=True))
        for evaluator in plugins['evaluators']:
            click.echo(f"  {click.style(evaluator['name'], fg='green')}: {evaluator['description']}")

        click.echo(click.style("\nStorage Backends:", fg='yellow', bold=True))
        for backend in plugins['storage_backends']:
            click.echo(f"  {click.style(backend['name'], fg='green')}: {backend['description']}")

        click.echo(click.style("\nChain Processors:", fg='yellow', bold=True))
        for processor in plugins['chain_processors']:
            click.echo(f"  {click.style(processor['name'], fg='green')}: {processor['description']}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command()
@click.argument('name')
@click.argument('content')
@click.option('--metadata', '-m', help='JSON metadata')
def add(name, content, metadata):
    """Add a new prompt or update existing one"""
    try:
        promptly = Promptly()
        meta = json.loads(metadata) if metadata else None
        message = promptly.add(name, content, meta)
        click.echo(click.style(message, fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.group()
def eval():
    """Evaluate prompts"""
    pass


@eval.command(name='run')
@click.argument('name')
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--evaluator', '-e', default='keyword', help='Evaluator to use (keyword, semantic, exact_match)')
def eval_run(name, test_file, evaluator):
    """Run evaluation on a prompt with specified evaluator"""
    try:
        promptly = Promptly()

        # Load test cases
        with open(test_file, 'r') as f:
            if test_file.endswith('.json'):
                test_data = json.load(f)
            elif test_file.endswith('.yaml') or test_file.endswith('.yml'):
                test_data = yaml.safe_load(f)
            else:
                raise Exception("Test file must be JSON or YAML")

        test_cases = test_data if isinstance(test_data, list) else test_data.get('tests', [])

        click.echo(click.style(f"\nRunning evaluation for prompt '{name}'...", fg='cyan'))
        click.echo(f"Evaluator: {evaluator}")
        click.echo(f"Test cases: {len(test_cases)}\n")

        results = promptly.eval_prompt(name, test_cases, evaluator_name=evaluator)

        for i, result in enumerate(results, 1):
            click.echo(click.style(f"Test {i}:", fg='cyan', bold=True))
            click.echo(f"  Formatted prompt: {result['formatted_prompt'][:100]}...")
            if result['score'] is not None:
                score_color = 'green' if result['score'] > 0.7 else 'yellow' if result['score'] > 0.4 else 'red'
                click.echo(f"  Score: {click.style(f"{result['score']:.2f}", fg=score_color)}")

            # Show detailed metrics
            if result.get('metrics'):
                click.echo(f"  Metrics: {json.dumps(result['metrics'], indent=4)}")
            click.echo()

        # Calculate average score
        scores = [r['score'] for r in results if r['score'] is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            score_color = 'green' if avg_score > 0.7 else 'yellow' if avg_score > 0.4 else 'red'
            click.echo(click.style(f"Average Score: {avg_score:.2f}", fg=score_color, bold=True))

        click.echo(click.style("\n✓ Evaluation complete", fg='green'))

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


# Add other CLI commands (get, list, branch, checkout, log, chain) from original...
# For brevity, showing key changes related to plugins

if __name__ == '__main__':
    cli()
