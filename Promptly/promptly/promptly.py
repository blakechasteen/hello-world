#!/usr/bin/env python3
"""
Promptly - Promptly manage your prompts with versioning, branching, eval, and chaining
"""

import click
import json
import sqlite3
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import shutil

class PromptlyDB:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def init_db(self):
        """Initialize the database schema"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Prompts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                branch TEXT NOT NULL DEFAULT 'main',
                version INTEGER NOT NULL DEFAULT 1,
                parent_id INTEGER,
                commit_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (parent_id) REFERENCES prompts(id)
            )
        """)
        
        # Branches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                head_commit TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (head_commit) REFERENCES prompts(commit_hash)
            )
        """)
        
        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_name TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                test_case TEXT NOT NULL,
                expected TEXT,
                actual TEXT,
                score REAL,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (commit_hash) REFERENCES prompts(commit_hash)
            )
        """)
        
        # Chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                steps TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Config table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Initialize main branch
        cursor.execute("INSERT OR IGNORE INTO branches (name, head_commit) VALUES ('main', 'init')")
        cursor.execute("INSERT OR IGNORE INTO config (key, value) VALUES ('current_branch', 'main')")
        
        conn.commit()
        self.close()


class Promptly:
    """Main Promptly class handling all operations"""
    
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = os.getcwd()
        
        self.root_dir = Path(root_dir)
        self.promptly_dir = self.root_dir / ".promptly"
        self.db_path = self.promptly_dir / "promptly.db"
        self.prompts_dir = self.promptly_dir / "prompts"
        self.chains_dir = self.promptly_dir / "chains"
        
    def init(self):
        """Initialize a new promptly repository"""
        if self.promptly_dir.exists():
            raise Exception("Promptly repository already initialized")
        
        self.promptly_dir.mkdir(parents=True)
        self.prompts_dir.mkdir()
        self.chains_dir.mkdir()
        
        db = PromptlyDB(str(self.db_path))
        db.init_db()
        
        return "Initialized empty Promptly repository"
    
    def _check_init(self):
        """Check if promptly is initialized"""
        if not self.promptly_dir.exists():
            raise Exception("Not a promptly repository. Run 'promptly init' first.")
    
    def _get_db(self) -> PromptlyDB:
        """Get database connection"""
        return PromptlyDB(str(self.db_path))
    
    def _get_current_branch(self) -> str:
        """Get current branch name"""
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = 'current_branch'")
        result = cursor.fetchone()
        db.close()
        return result[0] if result else 'main'
    
    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    def add(self, name: str, content: str, metadata: Dict = None):
        """Add a new prompt or update existing one"""
        self._check_init()
        
        current_branch = self._get_current_branch()
        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        # Check if prompt exists on this branch
        cursor.execute("""
            SELECT id, version, commit_hash FROM prompts 
            WHERE name = ? AND branch = ?
            ORDER BY version DESC LIMIT 1
        """, (name, current_branch))
        
        existing = cursor.fetchone()
        parent_id = existing[0] if existing else None
        version = existing[1] + 1 if existing else 1
        
        # Insert new version
        cursor.execute("""
            INSERT INTO prompts (name, content, branch, version, parent_id, commit_hash, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, content, current_branch, version, parent_id, commit_hash, 
              json.dumps(metadata) if metadata else None))
        
        # Update branch head
        cursor.execute("""
            UPDATE branches SET head_commit = ? WHERE name = ?
        """, (commit_hash, current_branch))
        
        conn.commit()
        
        # Save prompt to file
        prompt_file = self.prompts_dir / f"{name}.yaml"
        prompt_data = {
            'name': name,
            'content': content,
            'branch': current_branch,
            'version': version,
            'commit_hash': commit_hash,
            'metadata': metadata or {}
        }
        
        with open(prompt_file, 'w') as f:
            yaml.dump(prompt_data, f, default_flow_style=False)
        
        db.close()
        
        return f"Added prompt '{name}' (v{version}) on branch '{current_branch}' [{commit_hash}]"
    
    def get(self, name: str, version: int = None, commit_hash: str = None):
        """Get a prompt by name, optionally at specific version or commit"""
        self._check_init()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        if commit_hash:
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND commit_hash = ?
            """, (name, commit_hash))
        elif version:
            current_branch = self._get_current_branch()
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND branch = ? AND version = ?
            """, (name, current_branch, version))
        else:
            current_branch = self._get_current_branch()
            cursor.execute("""
                SELECT name, content, branch, version, commit_hash, created_at, metadata
                FROM prompts WHERE name = ? AND branch = ?
                ORDER BY version DESC LIMIT 1
            """, (name, current_branch))
        
        result = cursor.fetchone()
        db.close()
        
        if not result:
            return None
        
        return {
            'name': result[0],
            'content': result[1],
            'branch': result[2],
            'version': result[3],
            'commit_hash': result[4],
            'created_at': result[5],
            'metadata': json.loads(result[6]) if result[6] else {}
        }
    
    def list_prompts(self, branch: str = None):
        """List all prompts on current or specified branch"""
        self._check_init()
        
        if branch is None:
            branch = self._get_current_branch()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, MAX(version) as version, commit_hash, created_at
            FROM prompts
            WHERE branch = ?
            GROUP BY name
            ORDER BY name
        """, (branch,))
        
        results = cursor.fetchall()
        db.close()
        
        return [dict(row) for row in results]
    
    def branch(self, branch_name: str, from_branch: str = None):
        """Create a new branch"""
        self._check_init()
        
        if from_branch is None:
            from_branch = self._get_current_branch()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        # Get head commit of source branch
        cursor.execute("SELECT head_commit FROM branches WHERE name = ?", (from_branch,))
        result = cursor.fetchone()
        
        if not result:
            db.close()
            raise Exception(f"Branch '{from_branch}' does not exist")
        
        head_commit = result[0]
        
        # Create new branch
        try:
            cursor.execute("""
                INSERT INTO branches (name, head_commit)
                VALUES (?, ?)
            """, (branch_name, head_commit))
            
            # Copy prompts from source branch
            cursor.execute("""
                INSERT INTO prompts (name, content, branch, version, parent_id, commit_hash, metadata)
                SELECT name, content, ?, version, parent_id, 
                       substr(hex(randomblob(6)), 1, 12), metadata
                FROM prompts
                WHERE branch = ? AND version IN (
                    SELECT MAX(version) FROM prompts WHERE branch = ? GROUP BY name
                )
            """, (branch_name, from_branch, from_branch))
            
            conn.commit()
            db.close()
            
            return f"Created branch '{branch_name}' from '{from_branch}'"
        except sqlite3.IntegrityError:
            db.close()
            raise Exception(f"Branch '{branch_name}' already exists")
    
    def checkout(self, branch_name: str):
        """Switch to a different branch"""
        self._check_init()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        # Check if branch exists
        cursor.execute("SELECT name FROM branches WHERE name = ?", (branch_name,))
        if not cursor.fetchone():
            db.close()
            raise Exception(f"Branch '{branch_name}' does not exist")
        
        # Update current branch
        cursor.execute("UPDATE config SET value = ? WHERE key = 'current_branch'", (branch_name,))
        conn.commit()
        db.close()
        
        return f"Switched to branch '{branch_name}'"
    
    def log(self, name: str = None, limit: int = 10):
        """Show commit history"""
        self._check_init()
        
        current_branch = self._get_current_branch()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        if name:
            cursor.execute("""
                SELECT commit_hash, name, version, branch, created_at
                FROM prompts
                WHERE name = ? AND branch = ?
                ORDER BY version DESC
                LIMIT ?
            """, (name, current_branch, limit))
        else:
            cursor.execute("""
                SELECT commit_hash, name, version, branch, created_at
                FROM prompts
                WHERE branch = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (current_branch, limit))
        
        results = cursor.fetchall()
        db.close()
        
        return [dict(row) for row in results]
    
    def eval_prompt(self, name: str, test_cases: List[Dict], model_func=None):
        """Evaluate a prompt against test cases"""
        self._check_init()
        
        prompt = self.get(name)
        if not prompt:
            raise Exception(f"Prompt '{name}' not found")
        
        results = []
        
        for test_case in test_cases:
            # Format prompt with test inputs
            formatted_prompt = prompt['content'].format(**test_case.get('inputs', {}))
            
            # Run through model if provided
            actual = None
            if model_func:
                actual = model_func(formatted_prompt)
            
            # Calculate score if evaluator provided
            score = None
            if 'evaluator' in test_case and actual:
                score = test_case['evaluator'](actual, test_case.get('expected'))
            
            result = {
                'test_case': test_case,
                'formatted_prompt': formatted_prompt,
                'actual': actual,
                'score': score
            }
            results.append(result)
            
            # Save to database
            db = self._get_db()
            conn = db.connect()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO evaluations (prompt_name, commit_hash, test_case, expected, actual, score, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                prompt['commit_hash'],
                json.dumps(test_case),
                test_case.get('expected'),
                actual,
                score,
                json.dumps({'test_id': test_case.get('id')})
            ))
            
            conn.commit()
            db.close()
        
        return results
    
    def create_chain(self, name: str, steps: List[str], description: str = None):
        """Create a prompt chain"""
        self._check_init()
        
        # Verify all prompts exist
        for step in steps:
            if not self.get(step):
                raise Exception(f"Prompt '{step}' not found")
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO chains (name, steps, description)
                VALUES (?, ?, ?)
            """, (name, json.dumps(steps), description))
            
            conn.commit()
            
            # Save chain definition to file
            chain_file = self.chains_dir / f"{name}.yaml"
            chain_data = {
                'name': name,
                'steps': steps,
                'description': description
            }
            
            with open(chain_file, 'w') as f:
                yaml.dump(chain_data, f, default_flow_style=False)
            
            db.close()
            
            return f"Created chain '{name}' with {len(steps)} steps"
        except sqlite3.IntegrityError:
            db.close()
            raise Exception(f"Chain '{name}' already exists")
    
    def execute_chain(self, name: str, initial_input: Dict, model_func=None):
        """Execute a prompt chain"""
        self._check_init()
        
        db = self._get_db()
        conn = db.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT steps FROM chains WHERE name = ?", (name,))
        result = cursor.fetchone()
        db.close()
        
        if not result:
            raise Exception(f"Chain '{name}' not found")
        
        steps = json.loads(result[0])
        
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


# CLI Commands

@click.group()
@click.version_option(version='0.1.0')
def cli():
    """Promptly - Promptly manage your prompts with versioning, branching, eval, and chaining"""
    pass


@cli.command()
def init():
    """Initialize a new promptly repository"""
    try:
        promptly = Promptly()
        message = promptly.init()
        click.echo(click.style(message, fg='green'))
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


@cli.command()
@click.argument('name')
@click.option('--version', '-v', type=int, help='Specific version')
@click.option('--commit', '-c', help='Specific commit hash')
def get(name, version, commit):
    """Get a prompt by name"""
    try:
        promptly = Promptly()
        result = promptly.get(name, version=version, commit_hash=commit)
        
        if not result:
            click.echo(click.style(f"Prompt '{name}' not found", fg='yellow'))
            return
        
        click.echo(click.style(f"\nPrompt: {result['name']}", fg='cyan', bold=True))
        click.echo(f"Branch: {result['branch']}")
        click.echo(f"Version: {result['version']}")
        click.echo(f"Commit: {result['commit_hash']}")
        click.echo(f"Created: {result['created_at']}")
        click.echo(click.style("\nContent:", fg='cyan'))
        click.echo(result['content'])
        
        if result['metadata']:
            click.echo(click.style("\nMetadata:", fg='cyan'))
            click.echo(json.dumps(result['metadata'], indent=2))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command(name='list')
@click.option('--branch', '-b', help='List prompts from specific branch')
def list_cmd(branch):
    """List all prompts"""
    try:
        promptly = Promptly()
        current_branch = promptly._get_current_branch()
        target_branch = branch or current_branch
        
        prompts = promptly.list_prompts(target_branch)
        
        if not prompts:
            click.echo(click.style(f"No prompts found on branch '{target_branch}'", fg='yellow'))
            return
        
        click.echo(click.style(f"\nPrompts on branch '{target_branch}':", fg='cyan', bold=True))
        click.echo()
        
        for p in prompts:
            click.echo(f"  {click.style(p['name'], fg='green')} (v{p['version']}) [{p['commit_hash']}]")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command()
@click.argument('branch_name')
@click.option('--from', 'from_branch', help='Create branch from this branch')
def branch(branch_name, from_branch):
    """Create a new branch"""
    try:
        promptly = Promptly()
        message = promptly.branch(branch_name, from_branch)
        click.echo(click.style(message, fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command()
@click.argument('branch_name')
def checkout(branch_name):
    """Switch to a different branch"""
    try:
        promptly = Promptly()
        message = promptly.checkout(branch_name)
        click.echo(click.style(message, fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.command()
@click.option('--name', '-n', help='Show log for specific prompt')
@click.option('--limit', '-l', default=10, help='Number of commits to show')
def log(name, limit):
    """Show commit history"""
    try:
        promptly = Promptly()
        current_branch = promptly._get_current_branch()
        commits = promptly.log(name, limit)
        
        if not commits:
            click.echo(click.style("No commits found", fg='yellow'))
            return
        
        click.echo(click.style(f"\nCommit history on branch '{current_branch}':", fg='cyan', bold=True))
        click.echo()
        
        for commit in commits:
            click.echo(click.style(f"commit {commit['commit_hash']}", fg='yellow'))
            click.echo(f"Prompt: {commit['name']} (v{commit['version']})")
            click.echo(f"Date: {commit['created_at']}")
            click.echo()
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.group()
def eval():
    """Evaluate prompts"""
    pass


@eval.command(name='run')
@click.argument('name')
@click.argument('test_file', type=click.Path(exists=True))
def eval_run(name, test_file):
    """Run evaluation on a prompt"""
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
        click.echo(f"Test cases: {len(test_cases)}\n")
        
        results = promptly.eval_prompt(name, test_cases)
        
        for i, result in enumerate(results, 1):
            click.echo(click.style(f"Test {i}:", fg='cyan', bold=True))
            click.echo(f"  Formatted prompt: {result['formatted_prompt'][:100]}...")
            if result['score'] is not None:
                score_color = 'green' if result['score'] > 0.7 else 'yellow' if result['score'] > 0.4 else 'red'
                click.echo(f"  Score: {click.style(str(result['score']), fg=score_color)}")
            click.echo()
        
        click.echo(click.style("✓ Evaluation complete", fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@cli.group()
def chain():
    """Manage prompt chains"""
    pass


@chain.command(name='create')
@click.argument('name')
@click.argument('steps', nargs=-1, required=True)
@click.option('--description', '-d', help='Chain description')
def chain_create(name, steps, description):
    """Create a new prompt chain"""
    try:
        promptly = Promptly()
        message = promptly.create_chain(name, list(steps), description)
        click.echo(click.style(message, fg='green'))
        click.echo(f"Steps: {' -> '.join(steps)}")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


@chain.command(name='run')
@click.argument('name')
@click.argument('input_file', type=click.Path(exists=True))
def chain_run(name, input_file):
    """Execute a prompt chain"""
    try:
        promptly = Promptly()
        
        # Load initial input
        with open(input_file, 'r') as f:
            if input_file.endswith('.json'):
                initial_input = json.load(f)
            elif input_file.endswith('.yaml') or input_file.endswith('.yml'):
                initial_input = yaml.safe_load(f)
            else:
                raise Exception("Input file must be JSON or YAML")
        
        click.echo(click.style(f"\nExecuting chain '{name}'...\n", fg='cyan'))
        
        results = promptly.execute_chain(name, initial_input)
        
        for i, result in enumerate(results, 1):
            click.echo(click.style(f"Step {i}: {result['step']}", fg='cyan', bold=True))
            click.echo(f"Prompt: {result['prompt'][:150]}...")
            if result['output']:
                click.echo(f"Output: {result['output'][:150]}...")
            click.echo()
        
        click.echo(click.style("✓ Chain execution complete", fg='green'))
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'), err=True)


if __name__ == '__main__':
    cli()
