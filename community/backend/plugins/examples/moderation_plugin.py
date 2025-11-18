"""
Moderation Plugin

Advanced content moderation with AI-powered toxicity detection
"""
from typing import Dict, Any
from plugins.base import PluginBase, PluginType, HookType, PluginContext
import logging

logger = logging.getLogger(__name__)


class ModerationPlugin(PluginBase):
    """
    Moderation plugin - AI-powered content moderation

    Configuration:
        - toxicity_threshold: Toxicity score threshold (0.0-1.0) (default: 0.7)
        - auto_remove_threshold: Auto-remove content above this score (default: 0.9)
        - spam_detection_enabled: Enable spam detection (default: True)
        - profanity_filter_enabled: Enable profanity filtering (default: True)
        - link_blacklist: List of blocked domains (default: [])
        - rate_limit_posts: Max posts per user per hour (default: 10)
    """

    @property
    def name(self) -> str:
        return "AI Moderation"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "AI-powered content moderation with toxicity detection, spam filtering, and automatic actions"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.MODERATION

    async def initialize(self) -> bool:
        """Initialize the moderation plugin"""
        logger.info(f"Initializing {self.name} plugin v{self.version}")

        # Register hooks
        self.register_hook(HookType.BEFORE_POST_CREATE, self._moderate_post)
        self.register_hook(HookType.BEFORE_COMMENT_CREATE, self._moderate_comment)
        self.register_hook(HookType.CONTENT_REPORT, self._handle_report)
        self.register_hook(HookType.USER_REGISTER, self._check_new_user)

        # Configuration
        self.toxicity_threshold = self.get_config("toxicity_threshold", 0.7)
        self.auto_remove_threshold = self.get_config("auto_remove_threshold", 0.9)
        self.spam_detection_enabled = self.get_config("spam_detection_enabled", True)
        self.profanity_filter_enabled = self.get_config("profanity_filter_enabled", True)
        self.link_blacklist = self.get_config("link_blacklist", [])
        self.rate_limit = self.get_config("rate_limit_posts", 10)

        # Initialize AI models (placeholder)
        logger.info(f"{self.name} plugin initialized")
        return True

    async def shutdown(self) -> bool:
        """Shutdown the moderation plugin"""
        logger.info(f"Shutting down {self.name} plugin")
        return True

    async def _moderate_post(self, context: PluginContext) -> PluginContext:
        """
        Moderate post content before creation

        Hook: BEFORE_POST_CREATE
        """
        title = context.get("title", "")
        content = context.get("content", "")
        user_id = context.get("user_id")

        # Combine title and content for analysis
        full_text = f"{title} {content}"

        # Run moderation checks
        moderation_result = await self._run_moderation_checks(full_text, user_id)

        if moderation_result["action"] == "block":
            context.set("error", moderation_result["reason"])
            logger.warning(
                f"Post blocked for user {user_id}: {moderation_result['reason']}"
            )
            return context

        if moderation_result["action"] == "flag":
            context.set("requires_review", True)
            context.set("moderation_flags", moderation_result["flags"])
            logger.info(
                f"Post flagged for review from user {user_id}: {moderation_result['flags']}"
            )

        # Log moderation result
        context.set("moderation_score", moderation_result["toxicity_score"])

        return context

    async def _moderate_comment(self, context: PluginContext) -> PluginContext:
        """
        Moderate comment content before creation

        Hook: BEFORE_COMMENT_CREATE
        """
        content = context.get("content", "")
        user_id = context.get("user_id")

        # Run moderation checks
        moderation_result = await self._run_moderation_checks(content, user_id)

        if moderation_result["action"] == "block":
            context.set("error", moderation_result["reason"])
            logger.warning(
                f"Comment blocked for user {user_id}: {moderation_result['reason']}"
            )
            return context

        if moderation_result["action"] == "flag":
            context.set("requires_review", True)
            context.set("moderation_flags", moderation_result["flags"])

        return context

    async def _handle_report(self, context: PluginContext) -> PluginContext:
        """
        Handle content report

        Hook: CONTENT_REPORT
        """
        content_type = context.get("content_type")
        content_id = context.get("content_id")
        report_reason = context.get("reason")
        reporter_id = context.get("reporter_id")

        logger.info(
            f"Content report received: {content_type}:{content_id} "
            f"by user {reporter_id} - {report_reason}"
        )

        # In a real implementation:
        # 1. Fetch reported content
        # 2. Re-run moderation checks
        # 3. Escalate to human moderators if needed
        # 4. Take automatic action if confidence is high

        # For now, just log
        context.set("report_logged", True)

        return context

    async def _check_new_user(self, context: PluginContext) -> PluginContext:
        """
        Check new user registration for suspicious patterns

        Hook: USER_REGISTER
        """
        username = context.get("username")
        email = context.get("email")

        # Check for suspicious patterns
        # - Email from temporary email services
        # - Username contains spam keywords
        # - IP address flagged

        is_suspicious = False

        if is_suspicious:
            context.set("requires_verification", True)
            context.set("verification_reason", "Suspicious registration pattern detected")
            logger.warning(f"Suspicious registration: {username} ({email})")

        return context

    async def _run_moderation_checks(
        self, text: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Run all moderation checks on content

        Returns:
            {
                "action": "allow"|"flag"|"block",
                "reason": str,
                "toxicity_score": float,
                "flags": List[str]
            }
        """
        flags = []
        toxicity_score = 0.0

        # 1. Toxicity detection (AI model)
        toxicity_score = await self._detect_toxicity(text)

        if toxicity_score >= self.auto_remove_threshold:
            return {
                "action": "block",
                "reason": "Content violates community guidelines (high toxicity)",
                "toxicity_score": toxicity_score,
                "flags": ["high_toxicity"]
            }

        if toxicity_score >= self.toxicity_threshold:
            flags.append("moderate_toxicity")

        # 2. Spam detection
        if self.spam_detection_enabled:
            is_spam = await self._detect_spam(text)
            if is_spam:
                flags.append("spam")

        # 3. Profanity filter
        if self.profanity_filter_enabled:
            has_profanity = self._check_profanity(text)
            if has_profanity:
                flags.append("profanity")

        # 4. Link blacklist
        has_blocked_links = self._check_blacklisted_links(text)
        if has_blocked_links:
            return {
                "action": "block",
                "reason": "Content contains blocked links",
                "toxicity_score": toxicity_score,
                "flags": ["blocked_links"]
            }

        # 5. Rate limiting
        exceeds_rate_limit = self._check_rate_limit(user_id)
        if exceeds_rate_limit:
            return {
                "action": "block",
                "reason": "Rate limit exceeded. Please wait before posting again.",
                "toxicity_score": toxicity_score,
                "flags": ["rate_limit"]
            }

        # Determine action
        if flags:
            action = "flag"
            reason = f"Content flagged for review: {', '.join(flags)}"
        else:
            action = "allow"
            reason = "Content approved"

        return {
            "action": action,
            "reason": reason,
            "toxicity_score": toxicity_score,
            "flags": flags
        }

    async def _detect_toxicity(self, text: str) -> float:
        """
        Detect toxicity using AI model

        In production, this would use:
        - Perspective API
        - Hugging Face transformers (unitary/toxic-bert)
        - Custom fine-tuned model

        Returns:
            Toxicity score from 0.0 (clean) to 1.0 (toxic)
        """
        # Placeholder - simple keyword matching
        toxic_keywords = [
            "hate", "violent", "offensive", "harass", "threat"
        ]

        text_lower = text.lower()
        matches = sum(1 for keyword in toxic_keywords if keyword in text_lower)

        # Simple scoring (replace with actual AI model)
        score = min(matches * 0.2, 1.0)

        logger.debug(f"Toxicity score: {score:.2f}")
        return score

    async def _detect_spam(self, text: str) -> bool:
        """
        Detect spam patterns

        Checks for:
        - Excessive links
        - Repetitive text
        - Known spam keywords
        - All caps
        """
        # Count links
        link_count = text.count("http://") + text.count("https://")
        if link_count > 3:
            return True

        # Check for all caps (spam indicator)
        if len(text) > 20 and text.upper() == text:
            return True

        # Check spam keywords
        spam_keywords = ["buy now", "click here", "limited time", "act now"]
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in spam_keywords):
            return True

        return False

    def _check_profanity(self, text: str) -> bool:
        """
        Check for profanity

        In production, use a library like better-profanity
        """
        # Placeholder
        profanity_list = ["badword1", "badword2"]  # Actual list would be comprehensive
        text_lower = text.lower()
        return any(word in text_lower for word in profanity_list)

    def _check_blacklisted_links(self, text: str) -> bool:
        """Check if text contains blacklisted domains"""
        text_lower = text.lower()
        return any(domain in text_lower for domain in self.link_blacklist)

    def _check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user exceeded rate limit

        In production, use Redis with sliding window
        """
        # Placeholder - would query Redis
        # Key: "rate_limit:posts:{user_id}"
        # Store timestamps of recent posts
        # Check if count in last hour > self.rate_limit

        return False  # Allow for now


# Export plugin class
__all__ = ["ModerationPlugin"]
