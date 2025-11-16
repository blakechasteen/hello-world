#!/usr/bin/env python3
"""
Forum Search Module
===================
Search debugging forums (Stack Overflow, GitHub Issues) for solutions.

Capabilities:
- Stack Overflow search via API
- GitHub Issues search
- Reddit search (programming subreddits)
- Answer extraction and ranking
- MemoryShard creation for RAG

Author: Claude Code
Date: November 16, 2025
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from html import unescape
import re

logger = logging.getLogger(__name__)


@dataclass
class ForumPost:
    """Forum post/answer"""
    title: str
    question: str
    answer: str
    url: str
    score: int
    tags: List[str]
    source: str  # stackoverflow, github, reddit
    metadata: Dict[str, Any]


class ForumSearchEngine:
    """Search debugging forums for solutions"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.stats = {
            "searches": 0,
            "results_found": 0,
            "errors": 0
        }

    def search(
        self,
        query: str,
        source: str = "stackoverflow",
        max_results: int = 10
    ) -> List[ForumPost]:
        """
        Search forum for query.

        Args:
            query: Search query
            source: Forum source (stackoverflow, github, reddit)
            max_results: Maximum results to return

        Returns:
            List of forum posts
        """
        logger.info(f"Searching {source} for: {query}")

        if source == "stackoverflow":
            return self._search_stackoverflow(query, max_results)
        elif source == "github":
            return self._search_github_issues(query, max_results)
        elif source == "reddit":
            return self._search_reddit(query, max_results)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _search_stackoverflow(self, query: str, max_results: int) -> List[ForumPost]:
        """Search Stack Overflow via API"""
        try:
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": "stackoverflow",
                "pagesize": min(max_results, 100),
                "filter": "withbody"  # Include question body
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            posts = []

            for item in data.get("items", []):
                # Get accepted answer if available
                answer_text = "No accepted answer yet."
                if item.get("is_answered") and item.get("accepted_answer_id"):
                    answer_text = self._fetch_stackoverflow_answer(item["accepted_answer_id"])

                post = ForumPost(
                    title=self._clean_html(item.get("title", "")),
                    question=self._clean_html(item.get("body", "")[:1000]),
                    answer=answer_text[:1000],
                    url=item.get("link", ""),
                    score=item.get("score", 0),
                    tags=item.get("tags", []),
                    source="stackoverflow",
                    metadata={
                        "question_id": item.get("question_id"),
                        "view_count": item.get("view_count", 0),
                        "answer_count": item.get("answer_count", 0),
                        "is_answered": item.get("is_answered", False)
                    }
                )
                posts.append(post)

            self.stats["searches"] += 1
            self.stats["results_found"] += len(posts)

            logger.info(f"Found {len(posts)} Stack Overflow results")
            return posts

        except Exception as e:
            logger.error(f"Stack Overflow search error: {e}")
            self.stats["errors"] += 1
            return []

    def _fetch_stackoverflow_answer(self, answer_id: int) -> str:
        """Fetch specific Stack Overflow answer"""
        try:
            url = f"https://api.stackexchange.com/2.3/answers/{answer_id}"
            params = {
                "site": "stackoverflow",
                "filter": "withbody"
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data.get("items"):
                return self._clean_html(data["items"][0].get("body", ""))

        except Exception as e:
            logger.warning(f"Failed to fetch answer {answer_id}: {e}")

        return "Answer not available"

    def _search_github_issues(self, query: str, max_results: int) -> List[ForumPost]:
        """Search GitHub Issues"""
        try:
            url = "https://api.github.com/search/issues"
            params = {
                "q": f"{query} is:issue",
                "sort": "reactions",
                "order": "desc",
                "per_page": min(max_results, 100)
            }

            headers = {
                "Accept": "application/vnd.github.v3+json"
            }

            # Add GitHub token if available
            import os
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"

            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            posts = []

            for item in data.get("items", []):
                # Extract labels as tags
                tags = [label["name"] for label in item.get("labels", [])]

                post = ForumPost(
                    title=item.get("title", ""),
                    question=item.get("body", "")[:1000] or "No description",
                    answer=self._extract_github_answer(item),
                    url=item.get("html_url", ""),
                    score=item.get("reactions", {}).get("+1", 0),
                    tags=tags,
                    source="github",
                    metadata={
                        "repository": item.get("repository_url", "").split("/")[-2:],
                        "state": item.get("state"),
                        "comments": item.get("comments", 0)
                    }
                )
                posts.append(post)

            self.stats["searches"] += 1
            self.stats["results_found"] += len(posts)

            logger.info(f"Found {len(posts)} GitHub issues")
            return posts

        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            self.stats["errors"] += 1
            return []

    def _extract_github_answer(self, issue: Dict[str, Any]) -> str:
        """Extract answer from GitHub issue comments"""
        # For closed issues, assume there's a solution
        if issue.get("state") == "closed":
            return f"Issue closed. Check comments at {issue.get('html_url')} for solution."
        else:
            return "Issue still open - no definitive answer yet."

    def _search_reddit(self, query: str, max_results: int) -> List[ForumPost]:
        """Search Reddit programming subreddits"""
        try:
            # Use Reddit JSON API (no auth needed)
            subreddits = ["programming", "learnprogramming", "coding", "python", "javascript"]
            url = f"https://www.reddit.com/r/{'+'.join(subreddits)}/search.json"

            params = {
                "q": query,
                "sort": "relevance",
                "limit": min(max_results, 100)
            }

            headers = {
                "User-Agent": "Squad-Forum-Search/1.0"
            }

            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            posts = []

            for item in data.get("data", {}).get("children", []):
                post_data = item.get("data", {})

                post = ForumPost(
                    title=post_data.get("title", ""),
                    question=post_data.get("selftext", "")[:1000] or "Link post",
                    answer=f"{post_data.get('num_comments', 0)} comments - check link for answers",
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    score=post_data.get("score", 0),
                    tags=[post_data.get("subreddit", "")],
                    source="reddit",
                    metadata={
                        "subreddit": post_data.get("subreddit"),
                        "upvote_ratio": post_data.get("upvote_ratio", 0),
                        "num_comments": post_data.get("num_comments", 0)
                    }
                )
                posts.append(post)

            self.stats["searches"] += 1
            self.stats["results_found"] += len(posts)

            logger.info(f"Found {len(posts)} Reddit posts")
            return posts

        except Exception as e:
            logger.error(f"Reddit search error: {e}")
            self.stats["errors"] += 1
            return []

    def _clean_html(self, html: str) -> str:
        """Clean HTML tags from text"""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', html)
        # Decode HTML entities
        text = unescape(text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def posts_to_memory_shards(self, posts: List[ForumPost]) -> List[Dict[str, Any]]:
        """Convert forum posts to MemoryShards"""
        shards = []

        for post in posts:
            # Create shard for question + answer
            text = f"{post.title}\n\nQuestion: {post.question}\n\nAnswer: {post.answer}"

            shard = {
                "id": f"forum-{post.source}-{post.url.replace('://', '-').replace('/', '-')[:100]}",
                "text": text,
                "entities": post.tags + [post.source],
                "motifs": ["forum", "qa", "debugging", post.source],
                "metadata": {
                    "source": f"forum_{post.source}",
                    "url": post.url,
                    "title": post.title,
                    "score": post.score,
                    "tags": post.tags,
                    "timestamp": datetime.now().isoformat(),
                    **post.metadata
                }
            }
            shards.append(shard)

        return shards

    def get_stats(self) -> Dict[str, int]:
        """Get search statistics"""
        return self.stats.copy()
