#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced ChatOps Features
==========================
Next-generation collaborative features for modern development teams.

Features:
1. Collaborative Code Review - PR reviews directly in chat
2. Context-Aware Agents - Smart context switching and preservation
3. Knowledge Mining - Extract institutional knowledge
4. Predictive Alerts - ML-based issue prediction
5. Team Analytics - Collaboration patterns and insights

Usage:
    # Code review in chat
    reviewer = CodeReviewAssistant()
    await reviewer.review_pr(pr_number=123, conversation_id="room_xyz")

    # Context-aware conversations
    context_agent = ContextAwareAgent()
    await context_agent.switch_context("project_alpha")
"""

import logging
import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# 4. COLLABORATIVE CODE REVIEW
# ============================================================================

@dataclass
class PullRequest:
    """Pull request data."""
    number: int
    title: str
    author: str
    branch: str
    files_changed: List[str]
    additions: int
    deletions: int
    comments: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "open"  # open, approved, changes_requested, merged
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CodeReviewComment:
    """Code review comment."""
    file: str
    line: int
    author: str
    text: str
    resolved: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class CodeReviewAssistant:
    """
    Collaborative code review directly in chat.

    Features:
    - Fetch PR details and diffs
    - AI-powered code analysis
    - Inline commenting via chat
    - Review status tracking
    - Merge automation
    - Best practice suggestions

    Example Flow:
    User: !review pr 123
    Bot: [Fetches PR#123]
         📝 **PR#123: Add authentication middleware**
         Author: @alice
         Files: 3 changed (+142/-28)

         🔍 **AI Analysis:**
         - ✅ Tests included
         - ⚠️  Missing error handling in auth.js:42
         - 💡 Consider adding rate limiting

         **Files Changed:**
         1. src/auth.js (+98/-12)
         2. tests/auth.test.js (+42/-8)
         3. docs/auth.md (+2/-8)

         Reply with:
         - `approve` - Approve PR
         - `comment <file>:<line> <text>` - Add comment
         - `request changes` - Request changes
         - `merge` - Merge PR

    Usage:
        assistant = CodeReviewAssistant()

        # Review PR in chat
        await assistant.review_pr(123, "room_123")

        # Add comment via chat
        await assistant.add_comment(123, "auth.js:42", "Add error handling here")

        # Approve and merge
        await assistant.approve(123, "room_123")
        await assistant.merge(123)
    """

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize code review assistant.

        Args:
            github_token: Optional GitHub API token
        """
        self.github_token = github_token
        self.active_reviews: Dict[int, PullRequest] = {}

        logger.info("CodeReviewAssistant initialized")

    async def review_pr(
        self,
        pr_number: int,
        conversation_id: str
    ) -> str:
        """
        Start code review for a PR.

        Args:
            pr_number: PR number
            conversation_id: Matrix room ID

        Returns:
            Formatted review message
        """
        # Fetch PR details (would use GitHub API)
        pr = await self._fetch_pr(pr_number)

        # Run AI analysis
        analysis = await self._analyze_code(pr)

        # Format review message
        return self._format_review_message(pr, analysis)

    async def _fetch_pr(self, pr_number: int) -> PullRequest:
        """Fetch PR details from GitHub."""
        # Placeholder - would use actual GitHub API
        pr = PullRequest(
            number=pr_number,
            title="Add authentication middleware",
            author="@alice",
            branch="feature/auth",
            files_changed=["src/auth.js", "tests/auth.test.js", "docs/auth.md"],
            additions=142,
            deletions=28
        )

        self.active_reviews[pr_number] = pr
        return pr

    async def _analyze_code(self, pr: PullRequest) -> Dict[str, List[str]]:
        """Run AI-powered code analysis."""
        # Placeholder - would use AI model for code review
        analysis = {
            "positive": [
                "Tests included",
                "Documentation updated",
                "Follows coding standards"
            ],
            "warnings": [
                "Missing error handling in auth.js:42",
                "Consider adding input validation"
            ],
            "suggestions": [
                "Add rate limiting to prevent abuse",
                "Consider caching authentication results"
            ]
        }

        return analysis

    def _format_review_message(
        self,
        pr: PullRequest,
        analysis: Dict[str, List[str]]
    ) -> str:
        """Format code review message for chat."""
        lines = [
            f"📝 **PR#{pr.number}: {pr.title}**",
            f"Author: {pr.author}",
            f"Files: {len(pr.files_changed)} changed (+{pr.additions}/-{pr.deletions})",
            "",
            "🔍 **AI Analysis:**"
        ]

        for item in analysis["positive"]:
            lines.append(f"  ✅ {item}")

        for item in analysis["warnings"]:
            lines.append(f"  ⚠️  {item}")

        for item in analysis["suggestions"]:
            lines.append(f"  💡 {item}")

        lines.extend([
            "",
            "**Files Changed:**"
        ])

        for i, file in enumerate(pr.files_changed, 1):
            lines.append(f"{i}. {file}")

        lines.extend([
            "",
            "**Actions:**",
            "• `!approve {pr.number}` - Approve PR",
            "• `!comment {pr.number} <file>:<line> <text>` - Add comment",
            "• `!changes {pr.number}` - Request changes",
            "• `!merge {pr.number}` - Merge PR"
        ])

        return "\n".join(lines)

    async def add_comment(
        self,
        pr_number: int,
        location: str,
        text: str,
        author: str
    ) -> str:
        """
        Add comment to PR.

        Args:
            pr_number: PR number
            location: "file.js:42" format
            text: Comment text
            author: Comment author

        Returns:
            Confirmation message
        """
        if pr_number not in self.active_reviews:
            return f"PR#{pr_number} not found in active reviews"

        pr = self.active_reviews[pr_number]

        # Parse location
        match = re.match(r'(.+):(\d+)', location)
        if not match:
            return "Invalid location format. Use file:line (e.g., auth.js:42)"

        file, line = match.groups()

        comment = CodeReviewComment(
            file=file,
            line=int(line),
            author=author,
            text=text
        )

        pr.comments.append(comment.__dict__)

        return f"💬 Comment added to {file}:{line}"

    async def approve(self, pr_number: int, approver: str) -> str:
        """Approve PR."""
        if pr_number not in self.active_reviews:
            return f"PR#{pr_number} not found"

        pr = self.active_reviews[pr_number]
        pr.status = "approved"

        return f"✅ PR#{pr_number} approved by {approver}"

    async def merge(self, pr_number: int) -> str:
        """Merge PR."""
        if pr_number not in self.active_reviews:
            return f"PR#{pr_number} not found"

        pr = self.active_reviews[pr_number]

        if pr.status != "approved":
            return f"❌ Cannot merge - PR not approved (status: {pr.status})"

        pr.status = "merged"

        return f"🎉 PR#{pr_number} merged successfully!"


# ============================================================================
# 5. CONTEXT-AWARE AGENTS
# ============================================================================

@dataclass
class ConversationContext:
    """Context for a conversation thread."""
    context_id: str
    name: str
    type: str  # project, incident, meeting, general
    participants: List[str] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class ContextAwareAgent:
    """
    Intelligent context-aware agent with seamless context switching.

    Features:
    - Automatic context detection
    - Smooth context switching
    - Context preservation across sessions
    - Related context suggestions
    - Context-specific commands

    Example:
    User: "Let's discuss the auth feature"
    Bot: [Switches to project:auth context]
         📂 **Switched to: Project Auth**
         Recent activity:
         - PR#123 opened by @alice
         - 2 open issues
         - Last deployment: 2 days ago

         What would you like to know?

    User: "What about the API performance?"
    Bot: [Detects context switch to project:api]
         🔄 **Switching to: Project API**
         Would you like me to:
         - Show recent metrics
         - Check for incidents
         - Review open PRs

    Usage:
        agent = ContextAwareAgent()

        # Explicit context switch
        await agent.switch_context("project_auth")

        # Automatic detection
        context = agent.detect_context("Let's review the auth PR")
        # Returns: "project_auth"

        # Get context variables
        vars = agent.get_context_variables()
        # {"project": "auth", "repo": "backend", ...}
    """

    def __init__(self):
        """Initialize context-aware agent."""
        self.contexts: Dict[str, ConversationContext] = {}
        self.current_context: Optional[str] = None
        self.context_history: List[str] = []

        logger.info("ContextAwareAgent initialized")

    async def switch_context(
        self,
        context_id: str,
        conversation_id: str
    ) -> str:
        """
        Switch to a different context.

        Args:
            context_id: Context to switch to
            conversation_id: Current conversation

        Returns:
            Context switch message
        """
        # Get or create context
        if context_id not in self.contexts:
            context = ConversationContext(
                context_id=context_id,
                name=self._get_context_name(context_id),
                type=self._get_context_type(context_id)
            )
            self.contexts[context_id] = context
        else:
            context = self.contexts[context_id]

        # Update history
        if self.current_context:
            self.context_history.append(self.current_context)

        self.current_context = context_id
        context.last_activity = datetime.now()

        # Format switch message
        return self._format_context_message(context)

    def detect_context(self, message: str) -> Optional[str]:
        """
        Detect context from message content.

        Args:
            message: Message text

        Returns:
            Detected context ID or None
        """
        # Simple keyword-based detection
        # In production, would use NLP/ML

        keywords_map = {
            "project_auth": ["auth", "authentication", "login", "password"],
            "project_api": ["api", "endpoint", "performance", "latency"],
            "incident": ["incident", "outage", "down", "critical"],
            "deployment": ["deploy", "release", "rollout", "production"]
        }

        message_lower = message.lower()

        for context_id, keywords in keywords_map.items():
            if any(kw in message_lower for kw in keywords):
                return context_id

        return None

    def get_context_variables(self) -> Dict[str, Any]:
        """Get current context variables."""
        if not self.current_context or self.current_context not in self.contexts:
            return {}

        return self.contexts[self.current_context].variables

    def set_context_variable(self, key: str, value: Any) -> None:
        """Set variable in current context."""
        if self.current_context and self.current_context in self.contexts:
            self.contexts[self.current_context].variables[key] = value

    def _get_context_name(self, context_id: str) -> str:
        """Get human-readable context name."""
        # Would look up from registry
        return context_id.replace("_", " ").title()

    def _get_context_type(self, context_id: str) -> str:
        """Determine context type."""
        if "project" in context_id:
            return "project"
        elif "incident" in context_id:
            return "incident"
        elif "meeting" in context_id:
            return "meeting"
        else:
            return "general"

    def _format_context_message(self, context: ConversationContext) -> str:
        """Format context switch message."""
        emoji_map = {
            "project": "📂",
            "incident": "🚨",
            "meeting": "📅",
            "general": "💬"
        }

        emoji = emoji_map.get(context.type, "📌")

        lines = [
            f"{emoji} **Switched to: {context.name}**",
            f"Type: {context.type}",
            ""
        ]

        if context.active_topics:
            lines.append("**Active Topics:**")
            for topic in context.active_topics:
                lines.append(f"• {topic}")
            lines.append("")

        lines.append("What would you like to know?")

        return "\n".join(lines)


# ============================================================================
# 6. KNOWLEDGE MINING
# ============================================================================

class KnowledgeMiner:
    """
    Extract institutional knowledge from conversations.

    Features:
    - Auto-detect decisions and rationale
    - Extract best practices
    - Identify experts per topic
    - Build searchable knowledge base
    - Generate documentation

    Usage:
        miner = KnowledgeMiner()

        # Mine conversations
        insights = miner.mine(messages)

        # Get topic experts
        experts = miner.get_experts("authentication")
        # ["@alice", "@bob"]

        # Generate documentation
        docs = miner.generate_docs("api_design")
    """

    def __init__(self):
        """Initialize knowledge miner."""
        self.knowledge_base: Dict[str, List[Dict]] = defaultdict(list)
        self.expert_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def mine(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Mine knowledge from messages.

        Args:
            messages: Conversation messages

        Returns:
            Extracted insights
        """
        insights = {
            "decisions": [],
            "best_practices": [],
            "anti_patterns": [],
            "tools_mentioned": []
        }

        for msg in messages:
            text = msg.get("text", "")
            sender = msg.get("sender", "")

            # Detect decisions
            if any(phrase in text.lower() for phrase in ["decided to", "let's use", "we'll go with"]):
                insights["decisions"].append({
                    "text": text,
                    "author": sender,
                    "timestamp": msg.get("timestamp")
                })

            # Detect best practices
            if any(phrase in text.lower() for phrase in ["best practice", "should always", "make sure to"]):
                insights["best_practices"].append({
                    "text": text,
                    "author": sender
                })

            # Track expertise
            topics = self._extract_topics(text)
            for topic in topics:
                self.expert_map[topic][sender] += 1

        return insights

    def get_experts(self, topic: str, limit: int = 5) -> List[str]:
        """Get top experts for a topic."""
        if topic not in self.expert_map:
            return []

        sorted_experts = sorted(
            self.expert_map[topic].items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [expert for expert, _ in sorted_experts[:limit]]

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        # Simple keyword extraction
        tech_keywords = ["api", "auth", "database", "deployment", "testing", "security"]
        return [kw for kw in tech_keywords if kw in text.lower()]


# ============================================================================
# Example Usage & Demo
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Advanced ChatOps Features Demo")
    print("="*80)
    print()

    async def demo():
        # 1. Code Review
        print("1. COLLABORATIVE CODE REVIEW")
        print("-" * 40)

        reviewer = CodeReviewAssistant()

        review = await reviewer.review_pr(123, "room_123")
        print(review)
        print()

        comment_result = await reviewer.add_comment(
            123,
            "auth.js:42",
            "Add error handling for invalid tokens",
            "@bob"
        )
        print(comment_result)
        print()

        # 2. Context-Aware Agent
        print("2. CONTEXT-AWARE AGENT")
        print("-" * 40)

        agent = ContextAwareAgent()

        context_msg = await agent.switch_context("project_auth", "room_123")
        print(context_msg)
        print()

        detected = agent.detect_context("Let's review the API performance metrics")
        print(f"Detected context: {detected}")
        print()

        # 3. Knowledge Mining
        print("3. KNOWLEDGE MINING")
        print("-" * 40)

        miner = KnowledgeMiner()

        messages = [
            {"text": "We decided to use JWT for authentication", "sender": "@alice"},
            {"text": "Best practice: always validate input on the server side", "sender": "@bob"},
            {"text": "The API performance looks good", "sender": "@alice"}
        ]

        insights = miner.mine(messages)
        print(f"Decisions found: {len(insights['decisions'])}")
        print(f"Best practices: {len(insights['best_practices'])}")

        experts = miner.get_experts("api")
        print(f"API experts: {experts}")
        print()

    import asyncio
    asyncio.run(demo())

    print("="*80)
    print("✓ Demo complete!")
    print("="*80)
