"""
GitSpinner - Git Repository Ingestion
======================================

Ingests Git repository history into HoloLoom memory.

Features:
- Commit message parsing (conventional commits)
- File change tracking
- Author/committer extraction
- Issue/PR reference detection
- Breaking change detection
- Importance scoring (BREAKING > fix > feat > chore)
- Incremental updates (only new commits)
- Streaming support (large repos)

Author: Claude Code
Date: November 2025
"""

import subprocess
import re
import hashlib
from typing import List, Dict, Optional, AsyncIterator, Set, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import time

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore
)
from HoloLoom.spinningWheel.importance import ImportanceScorer
from HoloLoom.spinningWheel.utils import create_source_id


# ============================================================================
# Git Data Structures
# ============================================================================

@dataclass
class GitCommit:
    """Parsed Git commit."""
    hash: str
    hash_short: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    author_date: str  # ISO format
    commit_date: str  # ISO format
    subject: str
    body: str

    # Extracted data
    files_changed: List[str] = None
    insertions: int = 0
    deletions: int = 0

    # Parsed metadata
    commit_type: Optional[str] = None  # feat, fix, docs, etc.
    scope: Optional[str] = None
    is_breaking: bool = False
    issue_refs: List[str] = None

    def __post_init__(self):
        if self.files_changed is None:
            self.files_changed = []
        if self.issue_refs is None:
            self.issue_refs = []


# ============================================================================
# Git Utilities
# ============================================================================

class GitParser:
    """Parse Git repository data."""

    # Conventional commit pattern: type(scope): subject
    CONVENTIONAL_PATTERN = re.compile(
        r'^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)'
        r'(?:\((?P<scope>[^)]+)\))?'
        r'(?P<breaking>!)?'
        r':\s*(?P<subject>.+)$'
    )

    # Issue/PR references
    ISSUE_PATTERN = re.compile(r'#(\d+)')
    GH_PATTERN = re.compile(r'GH-(\d+)')
    JIRA_PATTERN = re.compile(r'([A-Z]+-\d+)')

    @staticmethod
    def run_git_command(
        repo_path: Path,
        args: List[str],
        check: bool = True
    ) -> Optional[str]:
        """
        Run git command in repository.

        Args:
            repo_path: Path to repository
            args: Git command arguments
            check: Raise on error

        Returns:
            Command output or None on error
        """
        try:
            result = subprocess.run(
                ['git', '-C', str(repo_path)] + args,
                capture_output=True,
                text=True,
                check=check,
                encoding='utf-8',
                errors='replace'
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            if check:
                raise
            return None
        except Exception as e:
            print(f"Warning: Git command failed: {e}")
            return None

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """Check if path is a Git repository."""
        try:
            result = subprocess.run(
                ['git', '-C', str(path), 'rev-parse', '--git-dir'],
                capture_output=True,
                check=True
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def get_commits(
        repo_path: Path,
        since: Optional[str] = None,
        until: Optional[str] = None,
        max_count: Optional[int] = None,
        branch: str = "HEAD"
    ) -> List[GitCommit]:
        """
        Get commits from repository.

        Args:
            repo_path: Repository path
            since: Start commit hash (None = from beginning)
            until: End commit hash (None = to HEAD)
            max_count: Maximum commits to fetch
            branch: Branch to query

        Returns:
            List of GitCommit objects
        """
        # Build git log command
        format_str = '%H%x00%h%x00%an%x00%ae%x00%cn%x00%ce%x00%aI%x00%cI%x00%s%x00%b%x00'

        args = [
            'log',
            branch,
            f'--format={format_str}',
            '--no-merges'  # Skip merge commits by default
        ]

        if since:
            args.append(f'{since}..{until or "HEAD"}')
        elif until:
            args.append(until)

        if max_count:
            args.extend(['--max-count', str(max_count)])

        output = GitParser.run_git_command(repo_path, args)

        if not output:
            return []

        # Parse commits
        commits = []
        commit_strings = output.split('\x00\n')

        for commit_str in commit_strings:
            if not commit_str.strip():
                continue

            parts = commit_str.split('\x00')

            if len(parts) < 10:
                continue

            commit = GitCommit(
                hash=parts[0],
                hash_short=parts[1],
                author_name=parts[2],
                author_email=parts[3],
                committer_name=parts[4],
                committer_email=parts[5],
                author_date=parts[6],
                commit_date=parts[7],
                subject=parts[8],
                body=parts[9] if len(parts) > 9 else ""
            )

            # Parse conventional commit
            GitParser.parse_conventional_commit(commit)

            # Extract issue references
            commit.issue_refs = GitParser.extract_issue_refs(
                commit.subject + ' ' + commit.body
            )

            commits.append(commit)

        return commits

    @staticmethod
    def parse_conventional_commit(commit: GitCommit):
        """
        Parse conventional commit format.

        Updates commit.commit_type, commit.scope, commit.is_breaking.
        """
        match = GitParser.CONVENTIONAL_PATTERN.match(commit.subject)

        if match:
            commit.commit_type = match.group('type')
            commit.scope = match.group('scope')
            commit.is_breaking = match.group('breaking') == '!'

        # Check for BREAKING CHANGE in body
        if 'BREAKING CHANGE' in commit.body or 'BREAKING:' in commit.body:
            commit.is_breaking = True

    @staticmethod
    def extract_issue_refs(text: str) -> List[str]:
        """Extract issue/PR references from text."""
        refs = []

        # GitHub issues (#123)
        refs.extend([f"#{m}" for m in GitParser.ISSUE_PATTERN.findall(text)])

        # GitHub issues (GH-123)
        refs.extend([f"GH-{m}" for m in GitParser.GH_PATTERN.findall(text)])

        # JIRA issues (PROJ-123)
        refs.extend(GitParser.JIRA_PATTERN.findall(text))

        return list(set(refs))

    @staticmethod
    def get_commit_stats(repo_path: Path, commit_hash: str) -> Dict[str, Any]:
        """
        Get file change statistics for commit.

        Returns:
            Dict with files_changed, insertions, deletions
        """
        # Get numstat
        output = GitParser.run_git_command(
            repo_path,
            ['show', '--numstat', '--format=', commit_hash]
        )

        if not output:
            return {'files_changed': [], 'insertions': 0, 'deletions': 0}

        files = []
        total_insertions = 0
        total_deletions = 0

        for line in output.split('\n'):
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                continue

            insertions_str, deletions_str, filename = parts

            # Handle binary files (-)
            insertions = 0 if insertions_str == '-' else int(insertions_str)
            deletions = 0 if deletions_str == '-' else int(deletions_str)

            files.append(filename)
            total_insertions += insertions
            total_deletions += deletions

        return {
            'files_changed': files,
            'insertions': total_insertions,
            'deletions': total_deletions
        }


# ============================================================================
# GitSpinner
# ============================================================================

class GitSpinner(BaseSpinner):
    """
    Spin Git repository commits into MemoryShards.

    Features:
    - Conventional commit parsing (feat, fix, docs, etc.)
    - Breaking change detection
    - File change tracking
    - Author extraction
    - Issue/PR reference detection
    - Importance scoring
    - Incremental updates
    - Streaming support

    Usage:
        >>> spinner = GitSpinner()
        >>> result = await spinner.spin("/path/to/repo")
        >>> print(f"Processed {result.shard_count} commits")
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        checkpoint_dir: Optional[Path] = None,
        include_merge_commits: bool = False,
        fetch_stats: bool = True,
        max_commits: Optional[int] = None
    ):
        """
        Initialize GitSpinner.

        Args:
            importance_threshold: Minimum importance score (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
            include_merge_commits: Include merge commits
            fetch_stats: Fetch file change statistics (slower but richer)
            max_commits: Maximum commits to process (None = all)
        """
        super().__init__(
            name="git",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.include_merge_commits = include_merge_commits
        self.fetch_stats = fetch_stats
        self.max_commits = max_commits

        # Create importance scorer with Git-specific terms
        self.importance_scorer = ImportanceScorer(
            technical_terms={
                # Commit types
                'feat', 'fix', 'refactor', 'perf', 'test', 'docs',
                'breaking', 'deprecated', 'security', 'hotfix',

                # Code terms
                'api', 'bug', 'feature', 'optimize', 'improve',
                'add', 'update', 'remove', 'delete', 'deprecate',
                'implement', 'enhance', 'clean', 'rewrite',

                # Technologies (customize for your domain)
                'python', 'typescript', 'react', 'api', 'database',
                'docker', 'kubernetes', 'ci', 'cd', 'test'
            }
        )

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=True,
            incremental=True,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            max_input_size=None,  # No size limit for repos
            supported_formats=['git']
        )

    def is_available(self) -> bool:
        """Check if git is available."""
        try:
            result = subprocess.run(
                ['git', '--version'],
                capture_output=True,
                check=True
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _spin_impl(
        self,
        source: Any,
        branch: str = "HEAD",
        **kwargs
    ) -> List[MemoryShard]:
        """
        Process Git repository.

        Args:
            source: Repository path
            branch: Branch to process
            **kwargs: Additional options

        Returns:
            List of MemoryShards (one per commit)
        """
        repo_path = Path(source)

        # Validate repository
        if not GitParser.is_git_repo(repo_path):
            raise ValueError(f"Not a Git repository: {repo_path}")

        # Get commits
        commits = GitParser.get_commits(
            repo_path,
            max_count=self.max_commits,
            branch=branch
        )

        # Process commits
        shards = []

        for commit in commits:
            # Fetch stats if enabled
            if self.fetch_stats:
                stats = GitParser.get_commit_stats(repo_path, commit.hash)
                commit.files_changed = stats['files_changed']
                commit.insertions = stats['insertions']
                commit.deletions = stats['deletions']

            # Create shard
            shard = await self._process_commit(commit, repo_path)
            shards.append(shard)

        return shards

    async def spin_stream(
        self,
        source: Any,
        branch: str = "HEAD",
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream commits one by one.

        Yields:
            MemoryShards one at a time
        """
        repo_path = Path(source)

        if not GitParser.is_git_repo(repo_path):
            raise ValueError(f"Not a Git repository: {repo_path}")

        # Get commits
        commits = GitParser.get_commits(
            repo_path,
            max_count=self.max_commits,
            branch=branch
        )

        for commit in commits:
            # Fetch stats
            if self.fetch_stats:
                stats = GitParser.get_commit_stats(repo_path, commit.hash)
                commit.files_changed = stats['files_changed']
                commit.insertions = stats['insertions']
                commit.deletions = stats['deletions']

            # Create shard
            shard = await self._process_commit(commit, repo_path)

            yield shard

    async def spin_incremental(
        self,
        source: Any,
        checkpoint: Optional[SpinnerCheckpoint] = None,
        branch: str = "HEAD",
        **kwargs
    ) -> SpinResult:
        """
        Process only new commits since checkpoint.

        Args:
            source: Repository path
            checkpoint: Previous checkpoint (None = from beginning)
            branch: Branch to process

        Returns:
            SpinResult with only new commits
        """
        repo_path = Path(source)
        source_id = create_source_id(str(repo_path.absolute()))

        # Load checkpoint if not provided
        if checkpoint is None:
            checkpoint = self.load_checkpoint(source_id)

        # Get last processed commit
        since_commit = checkpoint.last_processed_id if checkpoint else None

        # Get new commits
        commits = GitParser.get_commits(
            repo_path,
            since=since_commit,
            branch=branch,
            max_count=self.max_commits
        )

        # Filter out the since_commit if it appears (shouldn't happen, but safety check)
        if since_commit:
            commits = [c for c in commits if c.hash != since_commit]

        shards = []

        for i, commit in enumerate(commits):
            # Fetch stats
            if self.fetch_stats:
                stats = GitParser.get_commit_stats(repo_path, commit.hash)
                commit.files_changed = stats['files_changed']
                commit.insertions = stats['insertions']
                commit.deletions = stats['deletions']

            # Create shard
            shard = await self._process_commit(commit, repo_path)
            shards.append(shard)

            # Checkpoint every 100 commits
            # Save the most recent commit processed so far (commits[0] is newest)
            if (i + 1) % 100 == 0:
                checkpoint = SpinnerCheckpoint(
                    spinner_name=self.get_name(),
                    source_id=source_id,
                    last_processed_id=commits[0].hash,  # Most recent so far
                    items_processed=i + 1,
                    items_total=len(commits)
                )
                self.save_checkpoint(checkpoint)

        # Final checkpoint
        if commits:
            # Save the FIRST commit (most recent, since git log is reverse chronological)
            checkpoint = SpinnerCheckpoint(
                spinner_name=self.get_name(),
                source_id=source_id,
                last_processed_id=commits[0].hash,  # Most recent commit
                items_processed=len(commits),
                items_total=len(commits),
                state={'branch': branch}
            )
            self.save_checkpoint(checkpoint)

        return SpinResult(shards=shards, success=True)

    def score_importance(self, commit: GitCommit) -> ImportanceScore:
        """
        Score commit importance.

        Priority (highest to lowest):
        1. BREAKING CHANGE (1.0)
        2. Security fixes (0.9)
        3. Bug fixes (0.7)
        4. Features (0.6)
        5. Performance (0.5)
        6. Refactoring (0.4)
        7. Documentation (0.3)
        8. Tests (0.3)
        9. Chores (0.2)

        Args:
            commit: GitCommit to score

        Returns:
            ImportanceScore with detailed signals
        """
        signals = ImportanceSignals()

        # === BREAKING CHANGE (Override everything) ===
        if commit.is_breaking:
            return ImportanceScore(
                score=1.0,
                signals=ImportanceSignals(
                    custom_signals={'breaking_change': 1.0}
                ),
                reason="BREAKING CHANGE detected"
            )

        # === LENGTH SIGNAL ===
        text = commit.subject + ' ' + commit.body
        total_len = len(text)
        signals.length_score = min(1.0, total_len / 500)

        # === TECHNICAL SIGNAL ===
        signals.technical_score = self.importance_scorer.technical_scorer.score(text)

        # === STRUCTURAL SIGNAL ===
        # Well-formatted commits (conventional commits, lists, etc.)
        if commit.commit_type:
            signals.structural_score = 0.8
        elif any(line.startswith('-') for line in commit.body.split('\n')):
            signals.structural_score = 0.6
        else:
            signals.structural_score = 0.3

        # === COMMIT TYPE SIGNAL (Custom) ===
        type_scores = {
            'fix': 0.7,
            'feat': 0.6,
            'perf': 0.5,
            'refactor': 0.4,
            'docs': 0.3,
            'test': 0.3,
            'chore': 0.2,
            'ci': 0.2,
            'build': 0.2
        }

        if commit.commit_type:
            signals.custom_signals['commit_type'] = type_scores.get(
                commit.commit_type, 0.4
            )

        # === SECURITY SIGNAL ===
        security_keywords = ['security', 'cve', 'vulnerability', 'exploit', 'xss', 'sql injection']
        if any(kw in text.lower() for kw in security_keywords):
            signals.custom_signals['security'] = 0.9

        # === FILE CHANGE SIGNAL ===
        if commit.files_changed:
            file_count = len(commit.files_changed)
            # More files = potentially more important (but cap at 0.8)
            signals.custom_signals['file_impact'] = min(0.8, file_count / 20)

        # === CODE CHANGE SIGNAL ===
        if commit.insertions or commit.deletions:
            total_changes = commit.insertions + commit.deletions
            # More lines = potentially more important (but cap at 0.7)
            signals.custom_signals['code_impact'] = min(0.7, total_changes / 500)

        # === ISSUE REFERENCE SIGNAL ===
        if commit.issue_refs:
            signals.custom_signals['issue_linked'] = 0.6

        # === NOISE PENALTIES ===

        # Merge commits (even if not filtered)
        if commit.subject.lower().startswith('merge'):
            signals.noise_penalty = -0.4

        # Very short commits (likely trivial)
        if total_len < 20:
            signals.noise_penalty = min(signals.noise_penalty, -0.3)

        # Typo fixes
        if any(word in text.lower() for word in ['typo', 'whitespace', 'formatting']):
            signals.noise_penalty = min(signals.noise_penalty, -0.2)

        # Compute total
        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _process_commit(
        self,
        commit: GitCommit,
        repo_path: Path
    ) -> MemoryShard:
        """
        Convert GitCommit to MemoryShard.

        Args:
            commit: Parsed commit
            repo_path: Repository path

        Returns:
            MemoryShard
        """
        # Build shard text
        text_parts = [
            f"Commit: {commit.hash_short}",
            f"Author: {commit.author_name} <{commit.author_email}>",
            f"Date: {commit.author_date}",
            ""
        ]

        if commit.commit_type:
            text_parts.append(f"Type: {commit.commit_type}")
            if commit.scope:
                text_parts.append(f"Scope: {commit.scope}")

        if commit.is_breaking:
            text_parts.append("⚠️  BREAKING CHANGE")

        text_parts.extend([
            "",
            commit.subject,
            ""
        ])

        if commit.body:
            text_parts.append(commit.body)
            text_parts.append("")

        if commit.files_changed:
            text_parts.append(f"Files changed: {len(commit.files_changed)}")
            text_parts.append(f"Insertions: +{commit.insertions}")
            text_parts.append(f"Deletions: -{commit.deletions}")
            text_parts.append("")

            # List files (limit to 20)
            for file in commit.files_changed[:20]:
                text_parts.append(f"  - {file}")

            if len(commit.files_changed) > 20:
                text_parts.append(f"  ... and {len(commit.files_changed) - 20} more")

        shard_text = '\n'.join(text_parts)

        # Extract entities
        entities = self._extract_entities(commit)

        # Extract motifs
        motifs = self._extract_motifs(commit)

        # Score importance
        importance = self.score_importance(commit)

        # Create shard
        shard = self._create_shard(
            id_suffix=commit.hash_short,
            text=shard_text,
            episode=f"repo_{repo_path.name}",
            entities=entities,
            motifs=motifs,
            metadata={
                'commit_hash': commit.hash,
                'commit_hash_short': commit.hash_short,
                'author_name': commit.author_name,
                'author_email': commit.author_email,
                'author_date': commit.author_date,
                'commit_type': commit.commit_type,
                'scope': commit.scope,
                'is_breaking': commit.is_breaking,
                'files_changed': commit.files_changed,
                'insertions': commit.insertions,
                'deletions': commit.deletions,
                'issue_refs': commit.issue_refs,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0,
                'repository': str(repo_path.absolute())
            }
        )

        return shard

    def _extract_entities(self, commit: GitCommit) -> List[str]:
        """Extract entities from commit."""
        entities = []

        # Author
        entities.append(commit.author_name)
        entities.append(commit.author_email)

        # Committer (if different)
        if commit.committer_name != commit.author_name:
            entities.append(commit.committer_name)
            entities.append(commit.committer_email)

        # File paths (unique directories and files)
        if commit.files_changed:
            for file_path in commit.files_changed[:10]:  # Limit to 10
                # Add file name
                entities.append(Path(file_path).name)

                # Add directory if not root
                directory = str(Path(file_path).parent)
                if directory != '.':
                    entities.append(directory)

        # Issue references
        entities.extend(commit.issue_refs)

        # Deduplicate and return
        return list(set(entities))[:20]

    def _extract_motifs(self, commit: GitCommit) -> List[str]:
        """Extract motifs from commit."""
        motifs = []

        # Commit type
        if commit.commit_type:
            motifs.append(commit.commit_type)

        # Scope
        if commit.scope:
            motifs.append(f"scope_{commit.scope}")

        # Breaking change
        if commit.is_breaking:
            motifs.append("breaking_change")

        # File patterns
        if commit.files_changed:
            # File extensions
            extensions = set()
            for file_path in commit.files_changed:
                ext = Path(file_path).suffix
                if ext:
                    extensions.add(ext[1:])  # Remove leading dot

            motifs.extend([f"lang_{ext}" for ext in extensions])

        # Security
        security_keywords = ['security', 'cve', 'vulnerability']
        if any(kw in (commit.subject + commit.body).lower() for kw in security_keywords):
            motifs.append("security")

        return list(set(motifs))[:10]


# ============================================================================
# Convenience Functions
# ============================================================================

async def spin_repository(
    repo_path: str,
    importance_threshold: float = 0.3,
    branch: str = "HEAD",
    max_commits: Optional[int] = None
) -> SpinResult:
    """
    Quick function to spin a Git repository.

    Args:
        repo_path: Path to Git repository
        importance_threshold: Minimum importance (0.0-1.0)
        branch: Branch to process
        max_commits: Maximum commits (None = all)

    Returns:
        SpinResult with shards

    Example:
        >>> result = await spin_repository("/path/to/repo")
        >>> print(f"Processed {result.shard_count} commits")
    """
    spinner = GitSpinner(
        importance_threshold=importance_threshold,
        max_commits=max_commits
    )

    return await spinner.spin(repo_path, branch=branch)


async def spin_repository_incremental(
    repo_path: str,
    checkpoint_dir: str = "/tmp/git_checkpoints",
    branch: str = "HEAD"
) -> SpinResult:
    """
    Incrementally spin repository (only new commits).

    Args:
        repo_path: Path to Git repository
        checkpoint_dir: Checkpoint directory
        branch: Branch to process

    Returns:
        SpinResult with only new commits

    Example:
        >>> # First run: processes all commits
        >>> result = await spin_repository_incremental("/path/to/repo")
        >>>
        >>> # Second run: only processes new commits
        >>> result = await spin_repository_incremental("/path/to/repo")
    """
    spinner = GitSpinner(checkpoint_dir=Path(checkpoint_dir))

    return await spinner.spin_incremental(repo_path, branch=branch)
