"""
Unit Tests for GitSpinner
==========================

Tests Git repository ingestion including:
- Git command parsing
- Commit extraction
- Conventional commit parsing
- Importance scoring
- Incremental updates
- Streaming support

Author: Claude Code
Date: November 2025
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

from hololoom.spinningWheel.git_spinner import (
    GitSpinner,
    GitParser,
    GitCommit,
    spin_repository,
    spin_repository_incremental
)
from hololoom.spinningWheel.protocol import SpinnerStatus


# ============================================================================
# Test Helpers
# ============================================================================

def create_test_repo(tmp_path: Path) -> Path:
    """
    Create a test Git repository with sample commits.

    Returns:
        Path to repository
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize repo
    subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@example.com'],
        cwd=repo_path,
        check=True,
        capture_output=True
    )
    subprocess.run(
        ['git', 'config', 'user.name', 'Test User'],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    # Create commits
    commits = [
        ('feat: add new feature', 'This is a new feature\n\nImplements #123'),
        ('fix: fix critical bug', 'Fixed a critical bug\n\nCloses GH-456'),
        ('docs: update README', 'Updated documentation'),
        ('feat!: breaking change', 'BREAKING CHANGE: This breaks API'),
        ('chore: update dependencies', 'Minor dependency updates')
    ]

    for i, (subject, body) in enumerate(commits):
        # Create file
        file_path = repo_path / f"file{i}.txt"
        file_path.write_text(f"Content {i}\n")

        # Add and commit
        subprocess.run(['git', 'add', f'file{i}.txt'], cwd=repo_path, check=True, capture_output=True)

        # Create commit with subject and body
        commit_msg = f"{subject}\n\n{body}"
        subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=repo_path,
            check=True,
            capture_output=True
        )

    return repo_path


# ============================================================================
# Test GitParser
# ============================================================================

def test_git_parser_is_git_repo(tmp_path):
    """Test Git repository detection."""
    # Not a repo
    assert not GitParser.is_git_repo(tmp_path)

    # Create repo
    subprocess.run(['git', 'init'], cwd=tmp_path, check=True, capture_output=True)

    # Now it is a repo
    assert GitParser.is_git_repo(tmp_path)


def test_git_parser_get_commits(tmp_path):
    """Test commit extraction."""
    repo_path = create_test_repo(tmp_path)

    commits = GitParser.get_commits(repo_path)

    assert len(commits) == 5
    assert all(isinstance(c, GitCommit) for c in commits)

    # Check first commit (most recent)
    first_commit = commits[0]
    assert first_commit.subject == 'chore: update dependencies'
    assert first_commit.author_name == 'Test User'
    assert first_commit.author_email == 'test@example.com'


def test_git_parser_conventional_commit():
    """Test conventional commit parsing."""
    commit = GitCommit(
        hash='abc123',
        hash_short='abc',
        author_name='Test',
        author_email='test@example.com',
        committer_name='Test',
        committer_email='test@example.com',
        author_date='2025-11-02',
        commit_date='2025-11-02',
        subject='feat(api): add new endpoint',
        body=''
    )

    GitParser.parse_conventional_commit(commit)

    assert commit.commit_type == 'feat'
    assert commit.scope == 'api'
    assert commit.is_breaking is False


def test_git_parser_breaking_change():
    """Test breaking change detection."""
    # Breaking via !
    commit1 = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='feat!: breaking change',
        body=''
    )
    GitParser.parse_conventional_commit(commit1)
    assert commit1.is_breaking is True

    # Breaking via body
    commit2 = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='feat: feature',
        body='BREAKING CHANGE: This breaks things'
    )
    GitParser.parse_conventional_commit(commit2)
    assert commit2.is_breaking is True


def test_git_parser_issue_refs():
    """Test issue reference extraction."""
    text = """
    Fix bug in API endpoint

    This fixes #123 and resolves GH-456.
    Also addresses PROJ-789.
    """

    refs = GitParser.extract_issue_refs(text)

    assert '#123' in refs
    assert 'GH-456' in refs
    assert 'PROJ-789' in refs
    assert len(refs) == 3


def test_git_parser_commit_stats(tmp_path):
    """Test commit statistics extraction."""
    repo_path = create_test_repo(tmp_path)

    commits = GitParser.get_commits(repo_path)
    first_commit_hash = commits[0].hash

    stats = GitParser.get_commit_stats(repo_path, first_commit_hash)

    assert 'files_changed' in stats
    assert 'insertions' in stats
    assert 'deletions' in stats
    assert len(stats['files_changed']) > 0


# ============================================================================
# Test GitSpinner
# ============================================================================

def test_git_spinner_initialization():
    """Test GitSpinner initialization."""
    spinner = GitSpinner(
        importance_threshold=0.5,
        max_commits=100
    )

    assert spinner.get_name() == "git"
    assert spinner.importance_threshold == 0.5
    assert spinner.max_commits == 100


def test_git_spinner_capabilities():
    """Test capabilities reporting."""
    spinner = GitSpinner()
    capabilities = spinner.get_capabilities()

    assert capabilities.basic_processing is True
    assert capabilities.streaming is True
    assert capabilities.incremental is True
    assert capabilities.importance_scoring is True
    assert 'git' in capabilities.supported_formats


def test_git_spinner_availability():
    """Test dependency checking."""
    spinner = GitSpinner()

    # Should be available if git is installed
    if spinner.is_available():
        assert spinner.get_status() == SpinnerStatus.AVAILABLE
    else:
        pytest.skip("Git not installed")


@pytest.mark.asyncio
async def test_git_spinner_spin(tmp_path):
    """Test basic spin operation."""
    repo_path = create_test_repo(tmp_path)

    spinner = GitSpinner(max_commits=10)
    result = await spinner.spin(repo_path)

    assert result.success is True
    assert result.shard_count == 5  # 5 commits created
    assert result.processing_time_ms > 0

    # Check shard structure
    shard = result.shards[0]
    assert len(shard.text) > 0
    assert 'Commit:' in shard.text
    assert len(shard.entities) > 0
    assert len(shard.motifs) > 0
    assert 'commit_hash' in shard.metadata


@pytest.mark.asyncio
async def test_git_spinner_importance_filtering(tmp_path):
    """Test importance threshold filtering."""
    repo_path = create_test_repo(tmp_path)

    # High threshold (only breaking changes)
    spinner = GitSpinner(importance_threshold=0.9)
    result = await spinner.spin(repo_path)

    # Should only have breaking change commit
    assert result.shard_count <= 1  # Breaking change only

    if result.shard_count > 0:
        # Verify it's the breaking change
        assert any('breaking' in s.text.lower() for s in result.shards)


@pytest.mark.asyncio
async def test_git_spinner_streaming(tmp_path):
    """Test streaming support."""
    repo_path = create_test_repo(tmp_path)

    spinner = GitSpinner()
    shards = []

    async for shard in spinner.spin_stream(repo_path):
        shards.append(shard)

    assert len(shards) == 5
    assert all(s.metadata.get('commit_hash') for s in shards)


@pytest.mark.asyncio
async def test_git_spinner_incremental(tmp_path):
    """Test incremental updates."""
    repo_path = create_test_repo(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    spinner = GitSpinner(checkpoint_dir=checkpoint_dir)

    # First run (all commits)
    result1 = await spinner.spin_incremental(repo_path)
    assert result1.shard_count == 5

    # Second run (no new commits)
    result2 = await spinner.spin_incremental(repo_path)
    assert result2.shard_count == 0  # No new commits

    # Add new commit
    file_path = repo_path / "new_file.txt"
    file_path.write_text("New content\n")
    subprocess.run(['git', 'add', 'new_file.txt'], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'feat: add new file'],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    # Third run (1 new commit)
    result3 = await spinner.spin_incremental(repo_path)
    assert result3.shard_count == 1
    assert 'add new file' in result3.shards[0].text


def test_git_spinner_importance_scoring():
    """Test commit importance scoring."""
    spinner = GitSpinner()

    # Breaking change (highest)
    commit_breaking = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='feat!: breaking change',
        body='BREAKING CHANGE: API changed',
        commit_type='feat',
        is_breaking=True
    )
    score_breaking = spinner.score_importance(commit_breaking)
    assert score_breaking.score == 1.0
    assert 'BREAKING' in score_breaking.reason

    # Bug fix (medium-high)
    commit_fix = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='fix: fix critical bug',
        body='Fixed a critical security issue',
        commit_type='fix'
    )
    score_fix = spinner.score_importance(commit_fix)
    assert score_fix.score > 0.4  # Should be at least medium importance

    # Chore (low)
    commit_chore = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='chore: update dependencies',
        body='',
        commit_type='chore'
    )
    score_chore = spinner.score_importance(commit_chore)
    assert score_chore.score < score_fix.score


def test_git_spinner_entity_extraction():
    """Test entity extraction from commits."""
    spinner = GitSpinner()

    commit = GitCommit(
        hash='abc123', hash_short='abc',
        author_name='John Doe', author_email='john@example.com',
        committer_name='John Doe', committer_email='john@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='fix: fix bug in API',
        body='Fixes #123',
        files_changed=['src/api/endpoint.py', 'tests/test_api.py'],
        issue_refs=['#123']
    )

    entities = spinner._extract_entities(commit)

    # Should include author
    assert 'John Doe' in entities
    assert 'john@example.com' in entities

    # Should include file names
    assert 'endpoint.py' in entities or 'src/api' in entities

    # Should include issue refs
    assert '#123' in entities


def test_git_spinner_motif_extraction():
    """Test motif extraction from commits."""
    spinner = GitSpinner()

    commit = GitCommit(
        hash='abc', hash_short='abc',
        author_name='Test', author_email='test@example.com',
        committer_name='Test', committer_email='test@example.com',
        author_date='2025-11-02', commit_date='2025-11-02',
        subject='fix(auth): fix security vulnerability',
        body='CVE-2025-1234',
        commit_type='fix',
        scope='auth',
        files_changed=['src/auth.py', 'tests/test_auth.py']
    )

    motifs = spinner._extract_motifs(commit)

    # Should include commit type
    assert 'fix' in motifs

    # Should include scope
    assert 'scope_auth' in motifs

    # Should include language
    assert 'lang_py' in motifs

    # Should include security
    assert 'security' in motifs


# ============================================================================
# Test Convenience Functions
# ============================================================================

@pytest.mark.asyncio
async def test_spin_repository(tmp_path):
    """Test convenience function."""
    repo_path = create_test_repo(tmp_path)

    result = await spin_repository(str(repo_path), max_commits=10)

    assert result.success
    assert result.shard_count == 5


@pytest.mark.asyncio
async def test_spin_repository_incremental_function(tmp_path):
    """Test incremental convenience function."""
    repo_path = create_test_repo(tmp_path)
    checkpoint_dir = str(tmp_path / "checkpoints")

    # First run
    result1 = await spin_repository_incremental(
        str(repo_path),
        checkpoint_dir=checkpoint_dir
    )
    assert result1.shard_count == 5

    # Second run (no new commits)
    result2 = await spin_repository_incremental(
        str(repo_path),
        checkpoint_dir=checkpoint_dir
    )
    assert result2.shard_count == 0


# ============================================================================
# Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow(tmp_path):
    """Test complete GitSpinner workflow."""
    # Create test repo
    repo_path = create_test_repo(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"

    # Create spinner
    spinner = GitSpinner(
        importance_threshold=0.3,
        checkpoint_dir=checkpoint_dir,
        fetch_stats=True
    )

    # Check capabilities
    assert spinner.is_available()
    capabilities = spinner.get_capabilities()
    assert capabilities.incremental

    # First run (all commits)
    result1 = await spinner.spin_incremental(repo_path)
    assert result1.success
    assert result1.shard_count == 5

    # Verify shards have all metadata
    for shard in result1.shards:
        assert 'commit_hash' in shard.metadata
        assert 'author_name' in shard.metadata
        assert 'importance_score' in shard.metadata
        assert 'files_changed' in shard.metadata

    # Add new commit with breaking change
    file_path = repo_path / "breaking.txt"
    file_path.write_text("Breaking change\n")
    subprocess.run(['git', 'add', 'breaking.txt'], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ['git', 'commit', '-m', 'feat!: breaking API change\n\nBREAKING CHANGE: Removed old API'],
        cwd=repo_path,
        check=True,
        capture_output=True
    )

    # Second run (incremental)
    result2 = await spinner.spin_incremental(repo_path)
    assert result2.shard_count == 1

    # Verify breaking change has max importance
    breaking_shard = result2.shards[0]
    assert breaking_shard.metadata['is_breaking'] is True
    assert breaking_shard.metadata['importance_score'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
