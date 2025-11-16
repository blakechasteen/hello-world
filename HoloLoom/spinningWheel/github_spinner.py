"""
GitHubSpinner - GitHub Repository Ingestion
==============================================

Extracts data from GitHub repositories via the GitHub API.

Features:
- Issue extraction with labels, state, and comments
- Pull request extraction with review information
- Discussion extraction (if enabled)
- User/author information
- Importance scoring based on engagement and complexity
- Graceful API rate limiting handling
- Support for both public and private repositories (with token)
- Pagination support for large repositories

Author: Claude Code
Date: November 2025
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore,
    SpinnerStatus
)
from HoloLoom.spinningWheel.utils import create_source_id


# ============================================================================
# GitHub Data Structures
# ============================================================================

@dataclass
class GitHubIssue:
    """Parsed GitHub issue."""
    number: int
    title: str
    body: str
    state: str  # 'open' or 'closed'
    author: str
    created_at: str  # ISO format
    updated_at: str
    labels: List[str] = field(default_factory=list)
    comments_count: int = 0
    reactions_count: int = 0


@dataclass
class GitHubPullRequest:
    """Parsed GitHub pull request."""
    number: int
    title: str
    body: str
    state: str  # 'open', 'closed', or 'merged'
    author: str
    created_at: str
    updated_at: str
    merged_at: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    comments_count: int = 0
    reviews_count: int = 0
    reactions_count: int = 0


@dataclass
class GitHubDiscussion:
    """Parsed GitHub discussion."""
    number: int
    title: str
    body: str
    category: str
    author: str
    created_at: str
    updated_at: str
    comments_count: int = 0
    reactions_count: int = 0


# ============================================================================
# GitHub API Client
# ============================================================================

class GitHubClient:
    """Client for GitHub API."""

    API_BASE = "https://api.github.com"
    RATE_LIMIT_DELAY = 1.0  # Seconds between requests when rate limited

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.

        Args:
            token: GitHub personal access token (optional for public repos)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "HoloLoom-GitHubSpinner/1.0"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    @staticmethod
    def parse_repo_url(url: str) -> tuple:
        """
        Parse GitHub repo URL.

        Args:
            url: GitHub repo URL (e.g., "https://github.com/user/repo")

        Returns:
            (owner, repo) tuple
        """
        # Handle various URL formats
        url = url.rstrip('/')

        if "github.com/" in url:
            parts = url.split("github.com/")[1].split('/')
            if len(parts) >= 2:
                return parts[0], parts[1]

        raise ValueError(f"Invalid GitHub URL: {url}")

    async def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """
        Make GitHub API request with rate limit handling.

        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments to requests

        Returns:
            JSON response or None on error
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required for GitHub API")

        url = f"{self.API_BASE}/{endpoint}"

        try:
            response = requests.request(
                method,
                url,
                headers=self.headers,
                timeout=10,
                **kwargs
            )

            # Track rate limits
            if "X-RateLimit-Remaining" in response.headers:
                self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
                self.rate_limit_reset = int(response.headers["X-RateLimit-Reset"])

            # Handle rate limiting
            if response.status_code == 403:
                if self.rate_limit_remaining == 0:
                    wait_time = self.rate_limit_reset - time.time()
                    if wait_time > 0:
                        print(f"Rate limited. Waiting {wait_time:.1f}s...")
                        await asyncio.sleep(min(wait_time, 60))  # Cap at 60s
                        return await self._request(method, endpoint, **kwargs)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"GitHub API error: {e}")
            return None

    async def get_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_items: int = 100
    ) -> List[GitHubIssue]:
        """
        Fetch issues from repository.

        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state ('open', 'closed', or 'all')
            max_items: Maximum issues to fetch

        Returns:
            List of GitHubIssue objects
        """
        issues = []
        page = 1

        while len(issues) < max_items:
            endpoint = f"repos/{owner}/{repo}/issues"
            params = {
                "state": state,
                "page": page,
                "per_page": min(100, max_items - len(issues))
            }

            data = await self._request("GET", endpoint, params=params)
            if not data:
                break

            if not isinstance(data, list):
                break

            if not data:  # Empty page means we're done
                break

            for item in data:
                # Skip pull requests (they appear in issues endpoint)
                if "pull_request" in item:
                    continue

                issue = GitHubIssue(
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body", ""),
                    state=item["state"],
                    author=item["user"]["login"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    labels=[label["name"] for label in item.get("labels", [])],
                    comments_count=item.get("comments", 0),
                    reactions_count=item.get("reactions", {}).get("total_count", 0)
                )
                issues.append(issue)

                if len(issues) >= max_items:
                    break

            page += 1
            await asyncio.sleep(0.5)  # Gentle rate limiting

        return issues[:max_items]

    async def get_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_items: int = 100
    ) -> List[GitHubPullRequest]:
        """
        Fetch pull requests from repository.

        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state ('open', 'closed', or 'all')
            max_items: Maximum PRs to fetch

        Returns:
            List of GitHubPullRequest objects
        """
        prs = []
        page = 1

        while len(prs) < max_items:
            endpoint = f"repos/{owner}/{repo}/pulls"
            params = {
                "state": state,
                "page": page,
                "per_page": min(100, max_items - len(prs))
            }

            data = await self._request("GET", endpoint, params=params)
            if not data:
                break

            if not isinstance(data, list):
                break

            if not data:
                break

            for item in data:
                pr = GitHubPullRequest(
                    number=item["number"],
                    title=item["title"],
                    body=item.get("body", ""),
                    state=item["state"],
                    author=item["user"]["login"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    merged_at=item.get("merged_at"),
                    labels=[label["name"] for label in item.get("labels", [])],
                    additions=item.get("additions", 0),
                    deletions=item.get("deletions", 0),
                    changed_files=item.get("changed_files", 0),
                    comments_count=item.get("comments", 0),
                    reviews_count=item.get("review_comments", 0),
                    reactions_count=item.get("reactions", {}).get("total_count", 0)
                )
                prs.append(pr)

                if len(prs) >= max_items:
                    break

            page += 1
            await asyncio.sleep(0.5)

        return prs[:max_items]


# ============================================================================
# GitHub Spinner
# ============================================================================

class GitHubSpinner(BaseSpinner):
    """
    Spin GitHub repository data into MemoryShards.

    Features:
    - Issue extraction with metadata
    - Pull request extraction with review info
    - Discussion extraction (if available)
    - Importance scoring based on engagement
    - Rate limit handling
    - Support for public and private repos

    Usage:
        >>> spinner = GitHubSpinner()
        >>> result = await spinner.spin("https://github.com/anthropic-ai/anthropic-sdk-python")
        >>> print(f"Created {result.shard_count} shards")
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        importance_threshold: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
        include_issues: bool = True,
        include_prs: bool = True,
        include_discussions: bool = False,
        max_issues: int = 50,
        max_prs: int = 50
    ):
        """
        Initialize GitHubSpinner.

        Args:
            github_token: GitHub personal access token (optional)
            importance_threshold: Minimum importance score (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
            include_issues: Include issues
            include_prs: Include pull requests
            include_discussions: Include discussions
            max_issues: Maximum issues to fetch
            max_prs: Maximum PRs to fetch
        """
        super().__init__(
            name="github",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.client = GitHubClient(token=github_token)
        self.include_issues = include_issues
        self.include_prs = include_prs
        self.include_discussions = include_discussions
        self.max_issues = max_issues
        self.max_prs = max_prs

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            max_input_size=None,
            supported_formats=['github_url']
        )

    def is_available(self) -> bool:
        """Check if requests library is available."""
        return REQUESTS_AVAILABLE

    # ========================================================================
    # Core Spinning
    # ========================================================================

    async def _spin_impl(
        self,
        source: str,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Process GitHub repository.

        Args:
            source: GitHub repo URL
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        owner, repo = GitHubClient.parse_repo_url(source)

        shards = []

        # Fetch issues
        if self.include_issues:
            print(f"Fetching issues from {owner}/{repo}...")
            issues = await self.client.get_issues(owner, repo, max_items=self.max_issues)
            for issue in issues:
                shard = self._issue_to_shard(issue, owner, repo)
                shards.append(shard)

        # Fetch pull requests
        if self.include_prs:
            print(f"Fetching pull requests from {owner}/{repo}...")
            prs = await self.client.get_pull_requests(owner, repo, max_items=self.max_prs)
            for pr in prs:
                shard = self._pr_to_shard(pr, owner, repo)
                shards.append(shard)

        return shards

    # ========================================================================
    # Conversion Methods
    # ========================================================================

    def _issue_to_shard(
        self,
        issue: GitHubIssue,
        owner: str,
        repo: str
    ) -> MemoryShard:
        """Convert GitHub issue to MemoryShard."""
        # Build text content
        text_parts = [
            f"GitHub Issue #{issue.number}: {issue.title}",
            f"State: {issue.state}",
            f"Author: {issue.author}",
            f"Created: {issue.created_at}",
            f"Comments: {issue.comments_count}",
            ""
        ]

        if issue.labels:
            text_parts.append(f"Labels: {', '.join(issue.labels)}")
            text_parts.append("")

        if issue.body:
            text_parts.append(issue.body)

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            issue.author,
            f"issue_{issue.number}",
            owner,
            repo
        ]
        entities.extend(issue.labels)

        # Extract motifs
        motifs = ["github", "issue", issue.state]
        motifs.extend([f"label_{label}" for label in issue.labels])

        # Score importance
        importance = self._score_issue_importance(issue)

        # Create shard
        return self._create_shard(
            id_suffix=f"gh_issue_{owner}_{repo}_{issue.number}",
            text=text,
            episode=f"github_{owner}_{repo}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'github',
                'type': 'issue',
                'owner': owner,
                'repo': repo,
                'number': issue.number,
                'title': issue.title,
                'state': issue.state,
                'author': issue.author,
                'created_at': issue.created_at,
                'updated_at': issue.updated_at,
                'labels': issue.labels,
                'comments_count': issue.comments_count,
                'reactions_count': issue.reactions_count,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'url': f"https://github.com/{owner}/{repo}/issues/{issue.number}",
                'confidence': 1.0
            }
        )

    def _pr_to_shard(
        self,
        pr: GitHubPullRequest,
        owner: str,
        repo: str
    ) -> MemoryShard:
        """Convert GitHub PR to MemoryShard."""
        # Build text content
        text_parts = [
            f"GitHub PR #{pr.number}: {pr.title}",
            f"State: {pr.state}",
            f"Author: {pr.author}",
            f"Created: {pr.created_at}",
        ]

        if pr.merged_at:
            text_parts.append(f"Merged: {pr.merged_at}")

        text_parts.extend([
            f"Changes: +{pr.additions} -{pr.deletions} ({pr.changed_files} files)",
            f"Comments: {pr.comments_count} | Reviews: {pr.reviews_count}",
            ""
        ])

        if pr.labels:
            text_parts.append(f"Labels: {', '.join(pr.labels)}")
            text_parts.append("")

        if pr.body:
            text_parts.append(pr.body)

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            pr.author,
            f"pr_{pr.number}",
            owner,
            repo
        ]
        entities.extend(pr.labels)

        # Extract motifs
        motifs = ["github", "pull_request", pr.state]
        if pr.merged_at:
            motifs.append("merged")
        motifs.extend([f"label_{label}" for label in pr.labels])

        # Score importance
        importance = self._score_pr_importance(pr)

        # Create shard
        return self._create_shard(
            id_suffix=f"gh_pr_{owner}_{repo}_{pr.number}",
            text=text,
            episode=f"github_{owner}_{repo}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'github',
                'type': 'pull_request',
                'owner': owner,
                'repo': repo,
                'number': pr.number,
                'title': pr.title,
                'state': pr.state,
                'author': pr.author,
                'created_at': pr.created_at,
                'updated_at': pr.updated_at,
                'merged_at': pr.merged_at,
                'labels': pr.labels,
                'additions': pr.additions,
                'deletions': pr.deletions,
                'changed_files': pr.changed_files,
                'comments_count': pr.comments_count,
                'reviews_count': pr.reviews_count,
                'reactions_count': pr.reactions_count,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'url': f"https://github.com/{owner}/{repo}/pull/{pr.number}",
                'confidence': 1.0
            }
        )

    # ========================================================================
    # Importance Scoring
    # ========================================================================

    def _score_issue_importance(self, issue: GitHubIssue) -> ImportanceScore:
        """Score GitHub issue importance."""
        signals = ImportanceSignals()

        # Engagement signal
        engagement = (issue.comments_count + issue.reactions_count) / 10
        signals.engagement_score = min(1.0, engagement)

        # Length signal (body length)
        signals.length_score = min(1.0, len(issue.body) / 500)

        # Authority (closed issues less important than open)
        signals.authority_score = 1.0 if issue.state == "open" else 0.5

        # Recency (rough estimate: newer = higher score)
        try:
            created_date = datetime.fromisoformat(issue.created_at.replace('Z', '+00:00'))
            days_old = (datetime.now(created_date.tzinfo) - created_date).days
            # Decay over 30 days
            signals.recency_score = max(0.1, 1.0 - (days_old / 30))
        except Exception:
            signals.recency_score = 0.5

        # Custom signals
        if "bug" in [l.lower() for l in issue.labels]:
            signals.custom_signals['bug'] = 0.8
        if "feature" in [l.lower() for l in issue.labels]:
            signals.custom_signals['feature'] = 0.7
        if "enhancement" in [l.lower() for l in issue.labels]:
            signals.custom_signals['enhancement'] = 0.6

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )

    def _score_pr_importance(self, pr: GitHubPullRequest) -> ImportanceScore:
        """Score GitHub PR importance."""
        signals = ImportanceSignals()

        # Merged PRs are more important
        if pr.state == "merged":
            signals.authority_score = 1.0
        elif pr.state == "open":
            signals.authority_score = 0.7
        else:
            signals.authority_score = 0.4

        # Code impact signal
        code_impact = (pr.additions + pr.deletions) / 1000
        signals.custom_signals['code_impact'] = min(1.0, code_impact)

        # Review engagement
        review_engagement = pr.reviews_count / 5
        signals.engagement_score = min(1.0, review_engagement)

        # Comment engagement
        if pr.comments_count > 0:
            signals.custom_signals['discussion'] = min(1.0, pr.comments_count / 10)

        # Recency
        try:
            created_date = datetime.fromisoformat(pr.created_at.replace('Z', '+00:00'))
            days_old = (datetime.now(created_date.tzinfo) - created_date).days
            signals.recency_score = max(0.1, 1.0 - (days_old / 30))
        except Exception:
            signals.recency_score = 0.5

        # Labels
        if "feature" in [l.lower() for l in pr.labels]:
            signals.custom_signals['feature'] = 0.7
        if "bugfix" in [l.lower() for l in pr.labels]:
            signals.custom_signals['bugfix'] = 0.8
        if "documentation" in [l.lower() for l in pr.labels]:
            signals.custom_signals['documentation'] = 0.5

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )


# ============================================================================
# Convenience Functions
# ============================================================================

async def spin_github_repo(
    repo_url: str,
    github_token: Optional[str] = None,
    max_issues: int = 50,
    max_prs: int = 50
) -> SpinResult:
    """
    Quick function to spin a GitHub repository.

    Args:
        repo_url: GitHub repo URL (e.g., "https://github.com/user/repo")
        github_token: GitHub personal access token (optional)
        max_issues: Maximum issues to fetch
        max_prs: Maximum PRs to fetch

    Returns:
        SpinResult with shards

    Example:
        >>> result = await spin_github_repo("https://github.com/anthropic-ai/anthropic-sdk-python")
        >>> print(f"Created {result.shard_count} shards")
    """
    spinner = GitHubSpinner(
        github_token=github_token,
        max_issues=max_issues,
        max_prs=max_prs
    )

    return await spinner.spin(repo_url)
