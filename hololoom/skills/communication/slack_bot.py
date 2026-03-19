"""Slack Bot Skill - Proactive communication"""

import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)

# Check for slack_sdk availability
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False


class SlackBotSkill(BaseSkill):
    """Slack bot for proactive communication."""

    def __init__(self, token: str | None = None):
        """
        Initialize Slack Bot Skill.

        Args:
            token: Slack Bot OAuth token (xoxb-...). If not provided, looks for SLACK_BOT_TOKEN env var.
        """
        super().__init__()
        self.token = token
        self.client = None

        if SLACK_SDK_AVAILABLE and token:
            self.client = WebClient(token=token)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="slack_bot",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.COMMUNICATION,
            tags=["slack", "messaging", "notifications", "collaboration"],
            description_short="Slack bot integration for proactive communication and notifications",
            description_long="Send messages, post to channels, create threads, upload files, react to messages, and integrate with Slack workflows.",
            requires_code_execution=True,
            requires_network=True,
            external_dependencies=["slack_sdk or slack-bolt"],
            expected_latency_ms="200-2000ms",
            token_usage="50-500 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["send_message", "post_to_channel", "create_thread", "upload_file", "react", "update_status"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        # Check for slack_sdk availability
        if not SLACK_SDK_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="slack_sdk not available. Install with: pip install slack-sdk",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: slack-sdk"]
            )

        # Check for client initialization
        if not self.client:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="Slack client not initialized. Provide token in constructor or set SLACK_BOT_TOKEN env var.",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Client not initialized"]
            )

        try:
            # Dispatch to operation handler
            if skill_input.operation == "send_message":
                result = await self._send_message(skill_input.parameters)
            elif skill_input.operation == "post_to_channel":
                result = await self._post_to_channel(skill_input.parameters)
            elif skill_input.operation == "create_thread":
                result = await self._create_thread(skill_input.parameters)
            elif skill_input.operation == "upload_file":
                result = await self._upload_file(skill_input.parameters)
            elif skill_input.operation == "react":
                result = await self._react(skill_input.parameters)
            elif skill_input.operation == "update_status":
                result = await self._update_status(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Slack '{skill_input.operation}' completed",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except SlackApiError as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Slack API error: {e.response['error']}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Slack operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _send_message(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send direct message to user."""
        user = params["user"]
        text = params["text"]
        blocks = params.get("blocks")

        response = self.client.chat_postMessage(
            channel=user,
            text=text,
            blocks=blocks
        )

        return {
            "operation": "send_message",
            "user": user,
            "message_ts": response["ts"],
            "channel": response["channel"],
            "success": response["ok"]
        }

    async def _post_to_channel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Post message to channel."""
        channel = params["channel"]
        text = params["text"]
        blocks = params.get("blocks")
        thread_ts = params.get("thread_ts")

        response = self.client.chat_postMessage(
            channel=channel,
            text=text,
            blocks=blocks,
            thread_ts=thread_ts
        )

        return {
            "operation": "post_to_channel",
            "channel": channel,
            "message_ts": response["ts"],
            "is_thread_reply": thread_ts is not None,
            "success": response["ok"]
        }

    async def _create_thread(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create threaded conversation."""
        channel = params["channel"]
        parent_text = params["parent_text"]
        replies = params.get("replies", [])

        # Post parent message
        parent_response = self.client.chat_postMessage(
            channel=channel,
            text=parent_text
        )

        parent_ts = parent_response["ts"]
        reply_timestamps = []

        # Post replies to thread
        for reply_text in replies:
            reply_response = self.client.chat_postMessage(
                channel=channel,
                text=reply_text,
                thread_ts=parent_ts
            )
            reply_timestamps.append(reply_response["ts"])

        return {
            "operation": "create_thread",
            "channel": channel,
            "parent_ts": parent_ts,
            "reply_count": len(reply_timestamps),
            "reply_timestamps": reply_timestamps,
            "success": True
        }

    async def _upload_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Upload file to channel."""
        channels = params["channels"]
        file = params["file"]
        title = params.get("title", "")
        comment = params.get("comment", "")

        response = self.client.files_upload(
            channels=channels,
            file=file,
            title=title,
            initial_comment=comment
        )

        return {
            "operation": "upload_file",
            "file_id": response["file"]["id"],
            "file_name": response["file"]["name"],
            "channels": channels,
            "size": response["file"]["size"],
            "success": response["ok"]
        }

    async def _react(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add emoji reaction to message."""
        channel = params["channel"]
        timestamp = params["timestamp"]
        emoji = params["emoji"]

        response = self.client.reactions_add(
            channel=channel,
            timestamp=timestamp,
            name=emoji
        )

        return {
            "operation": "react",
            "channel": channel,
            "timestamp": timestamp,
            "emoji": emoji,
            "success": response["ok"]
        }

    async def _update_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update bot status."""
        status_text = params["status_text"]
        status_emoji = params.get("status_emoji", "")

        response = self.client.users_profile_set(
            profile={
                "status_text": status_text,
                "status_emoji": status_emoji
            }
        )

        return {
            "operation": "update_status",
            "status_text": status_text,
            "status_emoji": status_emoji,
            "success": response["ok"]
        }


from hololoom.skills.base import register_skill

register_skill(SlackBotSkill())
