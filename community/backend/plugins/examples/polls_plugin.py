"""
Example Polls Plugin

Adds poll functionality to posts
"""
from typing import Dict, Any
from plugins.base import PluginBase, PluginType, HookType, PluginContext
import logging

logger = logging.getLogger(__name__)


class PollsPlugin(PluginBase):
    """
    Polls plugin - adds poll creation and voting to posts

    Configuration:
        - max_options: Maximum number of poll options (default: 10)
        - allow_multiple: Allow users to vote for multiple options (default: False)
        - show_results_before_vote: Show results before user votes (default: False)
    """

    @property
    def name(self) -> str:
        return "Polls"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Add poll functionality to posts with voting and results"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.CONTENT

    async def initialize(self) -> bool:
        """
        Initialize the polls plugin
        """
        logger.info(f"Initializing {self.name} plugin v{self.version}")

        # Register hooks
        self.register_hook(HookType.BEFORE_POST_CREATE, self._validate_poll)
        self.register_hook(HookType.AFTER_POST_CREATE, self._create_poll)
        self.register_hook(HookType.POST_RENDER, self._render_poll)

        # Set default config
        self.max_options = self.get_config("max_options", 10)
        self.allow_multiple = self.get_config("allow_multiple", False)
        self.show_results_before_vote = self.get_config(
            "show_results_before_vote", False
        )

        logger.info(
            f"{self.name} plugin initialized with max_options={self.max_options}"
        )
        return True

    async def shutdown(self) -> bool:
        """
        Shutdown the polls plugin
        """
        logger.info(f"Shutting down {self.name} plugin")
        # Cleanup resources if needed
        return True

    async def _validate_poll(self, context: PluginContext) -> PluginContext:
        """
        Validate poll data before post creation

        Hook: BEFORE_POST_CREATE
        """
        post_type = context.get("post_type")

        if post_type != "poll":
            return context

        poll_data = context.get("poll_data")

        if not poll_data:
            context.set("error", "Poll data is required for poll posts")
            return context

        # Validate poll structure
        question = poll_data.get("question")
        options = poll_data.get("options", [])

        if not question:
            context.set("error", "Poll question is required")
            return context

        if len(options) < 2:
            context.set("error", "Poll must have at least 2 options")
            return context

        if len(options) > self.max_options:
            context.set(
                "error", f"Poll cannot have more than {self.max_options} options"
            )
            return context

        # Validation passed
        context.set("poll_validated", True)
        logger.debug(f"Poll validated: {question} with {len(options)} options")

        return context

    async def _create_poll(self, context: PluginContext) -> PluginContext:
        """
        Create poll data after post creation

        Hook: AFTER_POST_CREATE
        """
        post_type = context.get("post_type")

        if post_type != "poll":
            return context

        post_id = context.get("post_id")
        poll_data = context.get("poll_data")

        if not poll_data:
            return context

        # In a real implementation, this would:
        # 1. Create poll record in database
        # 2. Create poll options
        # 3. Initialize vote counts

        logger.info(
            f"Created poll for post {post_id}: {poll_data.get('question')}"
        )

        # Store poll metadata in context
        context.set("poll_created", True)
        context.set("poll_id", f"poll_{post_id}")

        return context

    async def _render_poll(self, context: PluginContext) -> PluginContext:
        """
        Render poll UI in post

        Hook: POST_RENDER
        """
        post_type = context.get("post_type")

        if post_type != "poll":
            return context

        poll_data = context.get("poll_data")
        user_id = context.get("user_id")

        if not poll_data:
            return context

        # Check if user has voted
        has_voted = self._check_user_vote(user_id, poll_data.get("poll_id"))

        # Build poll HTML
        html = self._build_poll_html(
            poll_data,
            has_voted=has_voted,
            show_results=has_voted or self.show_results_before_vote,
        )

        context.set("poll_html", html)
        logger.debug(f"Rendered poll for user {user_id}")

        return context

    def _check_user_vote(self, user_id: str, poll_id: str) -> bool:
        """
        Check if user has voted in this poll

        In a real implementation, this would query the database
        """
        # Placeholder
        return False

    def _build_poll_html(
        self, poll_data: Dict[str, Any], has_voted: bool, show_results: bool
    ) -> str:
        """
        Build poll HTML markup

        Args:
            poll_data: Poll data dictionary
            has_voted: Whether user has voted
            show_results: Whether to show results

        Returns:
            HTML string
        """
        question = poll_data.get("question", "")
        options = poll_data.get("options", [])

        html = f'<div class="poll">'
        html += f'<h3 class="poll-question">{question}</h3>'
        html += f'<div class="poll-options">'

        for i, option in enumerate(options):
            if show_results:
                # Show results
                votes = option.get("votes", 0)
                total_votes = sum(opt.get("votes", 0) for opt in options)
                percentage = (votes / total_votes * 100) if total_votes > 0 else 0

                html += f'<div class="poll-option poll-result">'
                html += f'<div class="option-text">{option["text"]}</div>'
                html += f'<div class="option-bar" style="width: {percentage}%"></div>'
                html += f'<div class="option-percentage">{percentage:.1f}% ({votes} votes)</div>'
                html += f'</div>'
            else:
                # Show voting buttons
                html += f'<div class="poll-option poll-vote">'
                html += f'<input type="radio" name="poll_option" value="{i}" id="option_{i}"'
                if not self.allow_multiple:
                    html += f' />'
                else:
                    html += f' type="checkbox" />'
                html += f'<label for="option_{i}">{option["text"]}</label>'
                html += f'</div>'

        html += f'</div>'

        if not has_voted and not show_results:
            html += f'<button class="poll-submit">Vote</button>'

        html += f'</div>'

        return html


# Export plugin class
__all__ = ["PollsPlugin"]
