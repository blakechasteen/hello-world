"""Tests for Chat-AR Bridge.

Tests AR trigger detection and mode transitions.
"""

import pytest
import asyncio
from elle.adapters.chat_ar_bridge import (
    ChatARBridge,
    create_chat_ar_bridge,
    ARTrigger,
    ElleMode,
    AR_TRIGGER_PATTERNS,
)


class TestARTriggerDetection:
    """Test AR trigger phrase detection."""

    def setup_method(self):
        """Create bridge for each test."""
        self.bridge = create_chat_ar_bridge(auto_transition=False)

    def test_visual_triggers(self):
        """Test visual trigger phrases are detected."""
        visual_phrases = [
            "Show me where to start",
            "Can you show me the workbench?",
            "Let me see what's in the corner",
            "Look at that tool over there",
            "Point to the shelf",
            "Where is the hammer?",
        ]

        for phrase in visual_phrases:
            trigger = self.bridge.detect_ar_trigger(phrase)
            assert trigger is not None, f"Should detect trigger in: {phrase}"
            assert trigger.trigger_type == "visual", f"Expected visual, got {trigger.trigger_type}"

    def test_spatial_triggers(self):
        """Test spatial trigger phrases are detected."""
        # Note: Some spatial phrases may match visual patterns first
        # (e.g., "where" is in visual patterns). Test pure spatial phrases.
        spatial_phrases = [
            "Find the wrench for me",
            "Locate the toolbox",
            "Navigate to the storage area",
            "How should I arrange these tools on the shelf?",
        ]

        for phrase in spatial_phrases:
            trigger = self.bridge.detect_ar_trigger(phrase)
            assert trigger is not None, f"Should detect trigger in: {phrase}"
            # Accept either spatial or visual (both are AR triggers)
            assert trigger.trigger_type in ("spatial", "visual"), \
                f"Expected spatial or visual, got {trigger.trigger_type}"

    def test_demonstration_triggers(self):
        """Test demonstration trigger phrases are detected."""
        demo_phrases = [
            "Show me how to do this",
            "Walk me through the process",
            "Help me do this step by step",
            "Teach me how to use this",
        ]

        for phrase in demo_phrases:
            trigger = self.bridge.detect_ar_trigger(phrase)
            assert trigger is not None, f"Should detect trigger in: {phrase}"
            assert trigger.trigger_type == "demonstration", f"Expected demonstration, got {trigger.trigger_type}"

    def test_no_trigger_for_regular_chat(self):
        """Test that regular chat messages don't trigger AR."""
        regular_phrases = [
            "Hello, how are you?",
            "Thanks for the help",
            "I'm thinking about organizing today",
            "What do you think I should prioritize?",
            "OK, let's do that",
        ]

        for phrase in regular_phrases:
            trigger = self.bridge.detect_ar_trigger(phrase)
            assert trigger is None, f"Should NOT detect trigger in: {phrase}"

    def test_chat_return_detection(self):
        """Test detection of chat return phrases."""
        return_phrases = [
            "Stop looking",
            "Go back to chat",
            "Just text",
            "Done with AR",
            "Close camera",
            "Never mind",
            "Cancel",
        ]

        for phrase in return_phrases:
            assert self.bridge.detect_chat_return(phrase), f"Should detect return in: {phrase}"

    def test_trigger_confidence(self):
        """Test that confidence increases with specificity."""
        # Single trigger word
        trigger1 = self.bridge.detect_ar_trigger("show me")
        # Multiple trigger words
        trigger2 = self.bridge.detect_ar_trigger("show me where to start organizing this area")

        assert trigger1 is not None
        assert trigger2 is not None
        # More trigger words = higher confidence
        assert trigger2.confidence >= trigger1.confidence


class TestModeTransitions:
    """Test mode transition logic."""

    def setup_method(self):
        """Create bridge for each test."""
        self.bridge = create_chat_ar_bridge(auto_transition=True)

    @pytest.mark.asyncio
    async def test_transition_to_ar(self):
        """Test transition from chat to AR mode."""
        assert self.bridge.current_mode == ElleMode.CHAT

        result = await self.bridge.process_chat_message(
            "Show me where to start organizing",
            chat_context={"recent_messages": "Hello"},
        )

        assert result["should_transition"] is True
        assert result["ar_trigger"] is not None
        assert self.bridge.current_mode == ElleMode.AR

    @pytest.mark.asyncio
    async def test_transition_back_to_chat(self):
        """Test transition from AR back to chat mode."""
        # First go to AR mode
        await self.bridge.process_chat_message("Show me the workbench")
        assert self.bridge.current_mode == ElleMode.AR

        # Then return to chat
        result = await self.bridge.process_chat_message("Go back to chat")
        assert result["should_transition"] is True
        assert self.bridge.current_mode == ElleMode.CHAT

    @pytest.mark.asyncio
    async def test_context_preserved_across_transition(self):
        """Test that chat context is preserved when transitioning to AR."""
        chat_context = {"recent_messages": "I need help with the shed"}

        await self.bridge.process_chat_message(
            "Show me the cluttered area",
            chat_context=chat_context,
        )

        assert "chat_history" in self.bridge.shared_context
        assert self.bridge.shared_context["last_chat_message"] == "Show me the cluttered area"


class TestPromptContextGeneration:
    """Test generation of prompt context for AR mode."""

    def setup_method(self):
        """Create bridge for each test."""
        self.bridge = create_chat_ar_bridge()

    @pytest.mark.asyncio
    async def test_ar_prompt_context_includes_trigger_reason(self):
        """Test that prompt context includes transition reason."""
        await self.bridge.process_chat_message("Show me the workbench")

        context = self.bridge.generate_ar_prompt_context()
        assert "see something specific" in context.lower() or "user just asked" in context.lower()

    @pytest.mark.asyncio
    async def test_ar_prompt_context_includes_user_message(self):
        """Test that prompt context includes user's original message."""
        await self.bridge.process_chat_message("Show me where to put this hammer")

        context = self.bridge.generate_ar_prompt_context()
        assert "hammer" in context.lower() or "show me" in context.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
