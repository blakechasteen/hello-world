#!/usr/bin/env python3
"""
JSON File Storage Backend Plugin

Alternative storage backend using JSON files instead of SQLite.
Suitable for simple use cases and easy version control integration.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import shutil

from ..base import BaseStorageBackend


class JSONStorage(BaseStorageBackend):
    """JSON file-based storage backend for Promptly"""

    def __init__(self):
        super().__init__(
            name="json",
            description="JSON file storage backend (git-friendly)"
        )
        self.storage_dir = None
        self.prompts_dir = None
        self.branches_dir = None
        self.chains_dir = None
        self.config_file = None

    def init_storage(self, storage_path: str) -> None:
        """Initialize JSON file storage"""
        self.storage_dir = Path(storage_path)
        self.prompts_dir = self.storage_dir / "prompts"
        self.branches_dir = self.storage_dir / "branches"
        self.chains_dir = self.storage_dir / "chains"
        self.config_file = self.storage_dir / "config.json"

        # Create directories
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)
        self.branches_dir.mkdir(exist_ok=True)
        self.chains_dir.mkdir(exist_ok=True)

        # Initialize config
        if not self.config_file.exists():
            config = {
                'current_branch': 'main',
                'created_at': datetime.now().isoformat()
            }
            self._write_json(self.config_file, config)

        # Initialize main branch
        main_branch_file = self.branches_dir / "main.json"
        if not main_branch_file.exists():
            branch_data = {
                'name': 'main',
                'head_commit': 'init',
                'created_at': datetime.now().isoformat(),
                'prompts': {}
            }
            self._write_json(main_branch_file, branch_data)

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Save a prompt to JSON files"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        timestamp = datetime.now().isoformat()
        commit_hash = self._generate_commit_hash(name, content, timestamp)

        # Load branch data
        branch_file = self.branches_dir / f"{branch}.json"
        if not branch_file.exists():
            raise Exception(f"Branch '{branch}' does not exist")

        branch_data = self._read_json(branch_file)

        # Get existing prompt info
        existing = branch_data['prompts'].get(name)
        version = existing['version'] + 1 if existing else 1

        # Create prompt file
        prompt_file = self.prompts_dir / f"{commit_hash}.json"
        prompt_info = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'commit_hash': commit_hash,
            'parent_commit': existing['commit_hash'] if existing else None,
            'created_at': timestamp,
            'metadata': metadata
        }
        self._write_json(prompt_file, prompt_info)

        # Update branch
        branch_data['prompts'][name] = {
            'version': version,
            'commit_hash': commit_hash,
            'updated_at': timestamp
        }
        branch_data['head_commit'] = commit_hash
        self._write_json(branch_file, branch_data)

        return commit_hash

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from JSON files"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        if commit_hash:
            # Load directly by commit hash
            prompt_file = self.prompts_dir / f"{commit_hash}.json"
            if prompt_file.exists():
                return self._read_json(prompt_file)
            return None

        # Load branch data
        branch_file = self.branches_dir / f"{branch}.json"
        if not branch_file.exists():
            return None

        branch_data = self._read_json(branch_file)

        if name not in branch_data['prompts']:
            return None

        if version:
            # Find specific version by traversing history
            current_commit = branch_data['prompts'][name]['commit_hash']
            while current_commit:
                prompt_file = self.prompts_dir / f"{current_commit}.json"
                if prompt_file.exists():
                    prompt = self._read_json(prompt_file)
                    if prompt['version'] == version:
                        return prompt
                    current_commit = prompt.get('parent_commit')
                else:
                    break
            return None

        # Get latest version
        commit = branch_data['prompts'][name]['commit_hash']
        prompt_file = self.prompts_dir / f"{commit}.json"
        if prompt_file.exists():
            return self._read_json(prompt_file)

        return None

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        branch_file = self.branches_dir / f"{branch}.json"
        if not branch_file.exists():
            return []

        branch_data = self._read_json(branch_file)

        prompts = []
        for name, info in sorted(branch_data['prompts'].items()):
            prompts.append({
                'name': name,
                'version': info['version'],
                'commit_hash': info['commit_hash'],
                'created_at': info.get('updated_at', '')
            })

        return prompts

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        branch_file = self.branches_dir / f"{branch_name}.json"
        if branch_file.exists():
            raise Exception(f"Branch '{branch_name}' already exists")

        source_branch_file = self.branches_dir / f"{from_branch}.json"
        if not source_branch_file.exists():
            raise Exception(f"Branch '{from_branch}' does not exist")

        # Copy branch data
        source_data = self._read_json(source_branch_file)
        new_branch_data = {
            'name': branch_name,
            'head_commit': source_data['head_commit'],
            'created_at': datetime.now().isoformat(),
            'prompts': source_data['prompts'].copy()
        }

        self._write_json(branch_file, new_branch_data)

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        config = self._read_json(self.config_file)
        return config.get('current_branch', 'main')

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        branch_file = self.branches_dir / f"{branch_name}.json"
        if not branch_file.exists():
            raise Exception(f"Branch '{branch_name}' does not exist")

        config = self._read_json(self.config_file)
        config['current_branch'] = branch_name
        self._write_json(self.config_file, config)

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        branch_file = self.branches_dir / f"{branch}.json"
        if not branch_file.exists():
            return []

        branch_data = self._read_json(branch_file)

        commits = []

        if name:
            # Get history for specific prompt
            if name in branch_data['prompts']:
                current_commit = branch_data['prompts'][name]['commit_hash']
                count = 0
                while current_commit and count < limit:
                    prompt_file = self.prompts_dir / f"{current_commit}.json"
                    if prompt_file.exists():
                        prompt = self._read_json(prompt_file)
                        commits.append({
                            'commit_hash': prompt['commit_hash'],
                            'name': prompt['name'],
                            'version': prompt['version'],
                            'branch': prompt['branch'],
                            'created_at': prompt['created_at']
                        })
                        current_commit = prompt.get('parent_commit')
                        count += 1
                    else:
                        break
        else:
            # Get all commits on branch
            all_commits = []
            for prompt_name, info in branch_data['prompts'].items():
                prompt_file = self.prompts_dir / f"{info['commit_hash']}.json"
                if prompt_file.exists():
                    prompt = self._read_json(prompt_file)
                    all_commits.append({
                        'commit_hash': prompt['commit_hash'],
                        'name': prompt['name'],
                        'version': prompt['version'],
                        'branch': prompt['branch'],
                        'created_at': prompt['created_at']
                    })

            # Sort by creation time and limit
            commits = sorted(all_commits, key=lambda x: x['created_at'], reverse=True)[:limit]

        return commits

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        # Create evaluations directory if not exists
        evals_dir = self.storage_dir / "evaluations"
        evals_dir.mkdir(exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().isoformat().replace(':', '-')
        eval_file = evals_dir / f"{eval_data['prompt_name']}_{timestamp}.json"

        self._write_json(eval_file, eval_data)

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        chain_file = self.chains_dir / f"{chain_data['name']}.json"
        if chain_file.exists():
            raise Exception(f"Chain '{chain_data['name']}' already exists")

        chain_info = {
            'name': chain_data['name'],
            'steps': chain_data['steps'],
            'description': chain_data.get('description'),
            'created_at': datetime.now().isoformat()
        }

        self._write_json(chain_file, chain_info)

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        if not self.storage_dir:
            raise RuntimeError("Storage not initialized")

        chain_file = self.chains_dir / f"{name}.json"
        if not chain_file.exists():
            return None

        return self._read_json(chain_file)

    def close(self) -> None:
        """Close/cleanup (no-op for JSON storage)"""
        pass
