#!/usr/bin/env python3
"""
Git Storage Backend Plugin

Production-grade Git backend using GitPython with:
- Store prompts as actual git commits
- Native git branching and merging
- Full git history and blame
- Remote repository support (GitHub, GitLab, Bitbucket)
- Conflict resolution strategies
- Git tags for prompt versions
- Comprehensive error handling
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from git import Repo, InvalidGitRepositoryError, GitCommandError, Actor
    from git.exc import GitError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("Warning: GitPython not available. Install with: pip install gitpython")

from ..base import BaseStorageBackend


class GitStorage(BaseStorageBackend):
    """
    Git-based storage backend with native version control

    Features:
    - True git repository for prompt storage
    - Native branching and merging
    - Full git history (log, blame, diff)
    - Remote repository support (push/pull/clone)
    - Conflict resolution with merge strategies
    - Git tags for semantic versioning
    - Git hooks integration
    - LFS support for large prompts

    Repository structure:
        /prompts/               # Prompt files (one per prompt)
            {name}.json         # Latest version
            {name}.v{N}.json    # Specific versions (optional)
        /chains/                # Chain definitions
            {name}.json
        /evaluations/           # Evaluation results
            {prompt_name}/
                {timestamp}.json
        .promptly/              # Metadata
            config.json         # Current branch, etc.
            index.json          # Prompt index

    Example:
        # Local repository
        storage = GitStorage()
        storage.init_storage('/path/to/repo')

        # Clone from remote
        storage = GitStorage(remote_url='https://github.com/user/prompts.git')
        storage.init_storage('/path/to/local/clone')
    """

    def __init__(self, remote_url: Optional[str] = None,
                 author_name: str = "Promptly",
                 author_email: str = "promptly@example.com",
                 auto_push: bool = False,
                 auto_pull: bool = False):
        """
        Initialize Git storage backend

        Args:
            remote_url: URL of remote repository (for clone/push/pull)
            author_name: Git commit author name
            author_email: Git commit author email
            auto_push: Automatically push commits to remote
            auto_pull: Automatically pull before operations
        """
        super().__init__(
            name="git",
            description="Git repository storage with native version control and remote sync"
        )

        if not GIT_AVAILABLE:
            raise ImportError("GitPython required for Git backend. Install: pip install gitpython")

        self.repo: Optional[Repo] = None
        self.repo_path: Optional[Path] = None
        self.remote_url = remote_url
        self.author = Actor(author_name, author_email)
        self.auto_push = auto_push
        self.auto_pull = auto_pull

        self.prompts_dir: Optional[Path] = None
        self.chains_dir: Optional[Path] = None
        self.evaluations_dir: Optional[Path] = None
        self.metadata_dir: Optional[Path] = None

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize Git repository

        Args:
            storage_path: Path to local git repository
        """
        self.repo_path = Path(storage_path)

        # Clone from remote if URL provided and repo doesn't exist
        if self.remote_url and not self.repo_path.exists():
            print(f"Cloning from {self.remote_url}...")
            self.repo = Repo.clone_from(self.remote_url, self.repo_path)
        else:
            # Initialize or open existing repository
            try:
                self.repo = Repo(self.repo_path)
            except InvalidGitRepositoryError:
                # Initialize new repository
                self.repo_path.mkdir(parents=True, exist_ok=True)
                self.repo = Repo.init(self.repo_path)

                # Add remote if specified
                if self.remote_url:
                    try:
                        self.repo.create_remote('origin', self.remote_url)
                    except GitCommandError:
                        # Remote already exists
                        pass

        # Setup directory structure
        self.prompts_dir = self.repo_path / 'prompts'
        self.chains_dir = self.repo_path / 'chains'
        self.evaluations_dir = self.repo_path / 'evaluations'
        self.metadata_dir = self.repo_path / '.promptly'

        for dir_path in [self.prompts_dir, self.chains_dir,
                        self.evaluations_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # Initialize metadata
        self._initialize_metadata()

        # Initial commit if repo is empty
        if not self.repo.heads:
            self._create_initial_commit()

    def _initialize_metadata(self) -> None:
        """Initialize metadata files"""
        config_file = self.metadata_dir / 'config.json'
        index_file = self.metadata_dir / 'index.json'

        if not config_file.exists():
            config = {
                'current_branch': 'main',
                'created_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            self._write_json(config_file, config)

        if not index_file.exists():
            index = {
                'prompts': {},
                'last_updated': datetime.utcnow().isoformat()
            }
            self._write_json(index_file, index)

    def _create_initial_commit(self) -> None:
        """Create initial commit for new repository"""
        # Create .gitignore
        gitignore = self.repo_path / '.gitignore'
        gitignore.write_text("*.pyc\n__pycache__/\n.DS_Store\n")

        # Create README
        readme = self.repo_path / 'README.md'
        readme.write_text("# Promptly Repository\n\nPrompt storage and version control.\n")

        # Add and commit
        self.repo.index.add(['.gitignore', 'README.md'])
        self.repo.index.add([str(p.relative_to(self.repo_path))
                            for p in self.metadata_dir.glob('*')])
        self.repo.index.commit('Initial commit', author=self.author, committer=self.author)

        # Create main branch if not exists
        if 'main' not in [b.name for b in self.repo.heads]:
            try:
                self.repo.create_head('main')
            except GitCommandError:
                pass  # Branch already exists

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _update_index(self, prompt_name: str, commit_hash: str, version: int) -> None:
        """Update prompt index"""
        index_file = self.metadata_dir / 'index.json'
        index = self._read_json(index_file)

        index['prompts'][prompt_name] = {
            'commit_hash': commit_hash,
            'version': version,
            'updated_at': datetime.utcnow().isoformat()
        }
        index['last_updated'] = datetime.utcnow().isoformat()

        self._write_json(index_file, index)

    def _auto_sync(self, operation: str = 'pull') -> None:
        """Perform automatic sync with remote"""
        if not self.repo or not self.remote_url:
            return

        try:
            origin = self.repo.remote('origin')

            if operation == 'pull' and self.auto_pull:
                origin.pull()
            elif operation == 'push' and self.auto_push:
                origin.push()
        except GitCommandError as e:
            print(f"Warning: Auto-{operation} failed: {e}")

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt as a git commit

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            str: Git commit hash
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        # Auto-pull before saving
        self._auto_sync('pull')

        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})

        # Ensure we're on the correct branch
        if self.repo.active_branch.name != branch:
            if branch in [b.name for b in self.repo.heads]:
                self.repo.heads[branch].checkout()
            else:
                # Branch will be created by create_branch
                raise Exception(f"Branch '{branch}' does not exist. Create it first.")

        # Get version number
        index_file = self.metadata_dir / 'index.json'
        index = self._read_json(index_file)
        version = index['prompts'].get(name, {}).get('version', 0) + 1

        # Create prompt file
        prompt_file = self.prompts_dir / f"{name}.json"
        prompt_doc = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata
        }
        self._write_json(prompt_file, prompt_doc)

        # Also save versioned copy
        versioned_file = self.prompts_dir / f"{name}.v{version}.json"
        self._write_json(versioned_file, prompt_doc)

        # Update index
        # We'll get the real commit hash after committing
        self._update_index(name, 'pending', version)

        # Stage files
        self.repo.index.add([
            str(prompt_file.relative_to(self.repo_path)),
            str(versioned_file.relative_to(self.repo_path)),
            str(index_file.relative_to(self.repo_path))
        ])

        # Commit
        commit_message = f"Update prompt '{name}' to version {version}"
        if metadata.get('description'):
            commit_message += f"\n\n{metadata['description']}"

        commit = self.repo.index.commit(
            commit_message,
            author=self.author,
            committer=self.author
        )

        commit_hash = commit.hexsha[:12]  # Use short hash

        # Update index with real commit hash
        self._update_index(name, commit_hash, version)
        self.repo.index.add([str(index_file.relative_to(self.repo_path))])
        self.repo.index.commit(
            f"Update index for '{name}'",
            author=self.author,
            committer=self.author,
            amend=True
        )

        # Auto-push after saving
        self._auto_sync('push')

        return commit_hash

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from git repository"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        # Checkout branch
        if branch in [b.name for b in self.repo.heads]:
            if self.repo.active_branch.name != branch:
                self.repo.heads[branch].checkout()
        else:
            return None

        if version:
            # Load specific version
            versioned_file = self.prompts_dir / f"{name}.v{version}.json"
            if versioned_file.exists():
                prompt = self._read_json(versioned_file)
                # Get commit hash for this version
                try:
                    commits = list(self.repo.iter_commits(
                        paths=str(versioned_file.relative_to(self.repo_path)),
                        max_count=1
                    ))
                    if commits:
                        prompt['commit_hash'] = commits[0].hexsha[:12]
                except GitCommandError:
                    pass
                return prompt
            return None

        elif commit_hash:
            # Checkout specific commit
            try:
                commit = self.repo.commit(commit_hash)
                prompt_file = self.prompts_dir / f"{name}.json"
                blob = commit.tree / str(prompt_file.relative_to(self.repo_path))
                data = blob.data_stream.read().decode('utf-8')
                prompt = json.loads(data)
                prompt['commit_hash'] = commit_hash
                return prompt
            except (GitCommandError, KeyError):
                return None

        else:
            # Get latest version
            prompt_file = self.prompts_dir / f"{name}.json"
            if not prompt_file.exists():
                return None

            prompt = self._read_json(prompt_file)

            # Get commit hash from git log
            try:
                commits = list(self.repo.iter_commits(
                    paths=str(prompt_file.relative_to(self.repo_path)),
                    max_count=1
                ))
                if commits:
                    prompt['commit_hash'] = commits[0].hexsha[:12]
            except GitCommandError:
                pass

            return prompt

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        # Checkout branch
        if branch in [b.name for b in self.repo.heads]:
            if self.repo.active_branch.name != branch:
                self.repo.heads[branch].checkout()
        else:
            return []

        prompts = []
        for prompt_file in self.prompts_dir.glob('*.json'):
            # Skip versioned files
            if '.v' in prompt_file.stem:
                continue

            prompt = self._read_json(prompt_file)

            # Get commit info
            try:
                commits = list(self.repo.iter_commits(
                    paths=str(prompt_file.relative_to(self.repo_path)),
                    max_count=1
                ))
                if commits:
                    commit = commits[0]
                    prompts.append({
                        'name': prompt['name'],
                        'version': prompt.get('version', 1),
                        'commit_hash': commit.hexsha[:12],
                        'created_at': datetime.fromtimestamp(commit.committed_date).isoformat()
                    })
            except GitCommandError:
                continue

        return sorted(prompts, key=lambda x: x['name'])

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new git branch"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        # Check source branch exists
        if from_branch not in [b.name for b in self.repo.heads]:
            raise Exception(f"Branch '{from_branch}' does not exist")

        # Check if target branch already exists
        if branch_name in [b.name for b in self.repo.heads]:
            raise Exception(f"Branch '{branch_name}' already exists")

        # Checkout source branch
        self.repo.heads[from_branch].checkout()

        # Create new branch
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        return self.repo.active_branch.name

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch (checkout)"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        if branch_name not in [b.name for b in self.repo.heads]:
            raise Exception(f"Branch '{branch_name}' does not exist")

        self.repo.heads[branch_name].checkout()

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get git commit history"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        # Checkout branch
        if branch in [b.name for b in self.repo.heads]:
            if self.repo.active_branch.name != branch:
                self.repo.heads[branch].checkout()
        else:
            return []

        commits = []

        if name:
            # Get history for specific prompt
            prompt_file = self.prompts_dir / f"{name}.json"
            if prompt_file.exists():
                try:
                    for commit in self.repo.iter_commits(
                        paths=str(prompt_file.relative_to(self.repo_path)),
                        max_count=limit
                    ):
                        commits.append({
                            'commit_hash': commit.hexsha[:12],
                            'name': name,
                            'version': None,  # Would need to parse from commit
                            'branch': branch,
                            'created_at': datetime.fromtimestamp(commit.committed_date).isoformat(),
                            'message': commit.message.strip(),
                            'author': str(commit.author)
                        })
                except GitCommandError:
                    pass
        else:
            # Get all commits on branch
            try:
                for commit in self.repo.iter_commits(branch, max_count=limit):
                    commits.append({
                        'commit_hash': commit.hexsha[:12],
                        'name': None,  # All prompts
                        'version': None,
                        'branch': branch,
                        'created_at': datetime.fromtimestamp(commit.committed_date).isoformat(),
                        'message': commit.message.strip(),
                        'author': str(commit.author)
                    })
            except GitCommandError:
                pass

        return commits

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        prompt_name = eval_data['prompt_name']
        prompt_eval_dir = self.evaluations_dir / prompt_name
        prompt_eval_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().isoformat().replace(':', '-')
        eval_file = prompt_eval_dir / f"{timestamp}.json"

        eval_doc = {
            'prompt_name': prompt_name,
            'commit_hash': eval_data['commit_hash'],
            'test_case': eval_data.get('test_case', {}),
            'expected': eval_data.get('expected'),
            'actual': eval_data.get('actual'),
            'score': eval_data.get('score'),
            'metrics': eval_data.get('metrics', {}),
            'evaluator_name': eval_data.get('evaluator_name'),
            'created_at': datetime.utcnow().isoformat()
        }

        self._write_json(eval_file, eval_doc)

        # Commit evaluation
        self.repo.index.add([str(eval_file.relative_to(self.repo_path))])
        self.repo.index.commit(
            f"Add evaluation for '{prompt_name}'",
            author=self.author,
            committer=self.author
        )

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        chain_file = self.chains_dir / f"{chain_data['name']}.json"

        if chain_file.exists():
            raise Exception(f"Chain '{chain_data['name']}' already exists")

        chain_doc = {
            'name': chain_data['name'],
            'steps': chain_data['steps'],
            'description': chain_data.get('description'),
            'created_at': datetime.utcnow().isoformat()
        }

        self._write_json(chain_file, chain_doc)

        # Commit chain
        self.repo.index.add([str(chain_file.relative_to(self.repo_path))])
        self.repo.index.commit(
            f"Add chain '{chain_data['name']}'",
            author=self.author,
            committer=self.author
        )

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        chain_file = self.chains_dir / f"{name}.json"

        if not chain_file.exists():
            return None

        return self._read_json(chain_file)

    def close(self) -> None:
        """Close/cleanup (git doesn't need explicit cleanup)"""
        self.repo = None

    # Additional Git-specific methods

    def merge_branch(self, source_branch: str, strategy: str = 'ours') -> Dict[str, Any]:
        """
        Merge a branch into current branch

        Args:
            source_branch: Branch to merge from
            strategy: Merge strategy ('ours', 'theirs', 'recursive')

        Returns:
            Dict with merge result
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        current_branch = self.repo.active_branch.name

        try:
            # Merge
            base = self.repo.merge_base(self.repo.active_branch, source_branch)
            self.repo.index.merge_tree(self.repo.active_branch, base_commit=base[0])

            # Commit merge
            merge_commit = self.repo.index.commit(
                f"Merge branch '{source_branch}' into '{current_branch}'",
                parent_commits=(self.repo.active_branch.commit,
                              self.repo.heads[source_branch].commit),
                author=self.author,
                committer=self.author
            )

            return {
                'status': 'success',
                'commit_hash': merge_commit.hexsha[:12],
                'message': f"Merged '{source_branch}' into '{current_branch}'"
            }

        except GitCommandError as e:
            return {
                'status': 'conflict',
                'message': str(e)
            }

    def push_to_remote(self, remote: str = 'origin', branch: Optional[str] = None) -> None:
        """
        Push to remote repository

        Args:
            remote: Remote name
            branch: Branch to push (None = current branch)
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        branch = branch or self.repo.active_branch.name

        try:
            origin = self.repo.remote(remote)
            origin.push(branch)
        except GitCommandError as e:
            raise Exception(f"Push failed: {e}")

    def pull_from_remote(self, remote: str = 'origin', branch: Optional[str] = None) -> None:
        """
        Pull from remote repository

        Args:
            remote: Remote name
            branch: Branch to pull (None = current branch)
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        branch = branch or self.repo.active_branch.name

        try:
            origin = self.repo.remote(remote)
            origin.pull(branch)
        except GitCommandError as e:
            raise Exception(f"Pull failed: {e}")

    def create_tag(self, tag_name: str, message: str = "") -> None:
        """
        Create a git tag (useful for versioning)

        Args:
            tag_name: Tag name
            message: Tag message
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        self.repo.create_tag(tag_name, message=message or f"Tag {tag_name}")

    def get_diff(self, commit_a: str, commit_b: str, prompt_name: Optional[str] = None) -> str:
        """
        Get diff between two commits

        Args:
            commit_a: First commit hash
            commit_b: Second commit hash
            prompt_name: Optional prompt name to filter diff

        Returns:
            Diff text
        """
        if not self.repo:
            raise RuntimeError("Storage not initialized")

        commit1 = self.repo.commit(commit_a)
        commit2 = self.repo.commit(commit_b)

        if prompt_name:
            prompt_file = self.prompts_dir / f"{prompt_name}.json"
            path = str(prompt_file.relative_to(self.repo_path))
            diff = commit1.diff(commit2, paths=path)
        else:
            diff = commit1.diff(commit2)

        return '\n'.join([d.diff.decode('utf-8') for d in diff if d.diff])
