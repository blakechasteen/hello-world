"""
Basic Tests for Voice-First UX Layer

Tests:
- Voice grammar classification
- Voice router intent detection
- Mode state machine transitions
- Basic integration (mock handlers)

Run:
    pytest hololoom/voice_first/tests/test_basic.py -v
"""

import pytest
import asyncio

from hololoom.voice.threads.grammar import VoiceGrammar, CommandType, CommandIntent
from hololoom.voice.threads.core import (
    VoiceMode,
    VoiceModeStateMachine,
    VoiceRouter
)


# ============================================================================
# Grammar Tests
# ============================================================================

class TestVoiceGrammar:
    """Test voice command grammar classification"""

    def setup_method(self):
        """Setup test grammar"""
        self.grammar = VoiceGrammar()

    def test_thread_creation_patterns(self):
        """Test thread creation command detection"""
        test_cases = [
            ("start a new thread for orchard planning", CommandType.THREAD_CREATE, "orchard planning"),
            ("new thread: biochar production", CommandType.THREAD_CREATE, "biochar production"),
            ("let's talk about composting methods", CommandType.THREAD_CREATE, "composting methods"),
        ]

        for text, expected_type, expected_topic in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == expected_type, f"Failed for: {text}"
            assert intent.params.get('topic') == expected_topic
            assert intent.confidence > 0.8

    def test_thread_switching_patterns(self):
        """Test thread switching command detection"""
        test_cases = [
            ("switch to orchard planning thread", CommandType.THREAD_SWITCH, "orchard planning"),
            ("go back to biochar production", CommandType.THREAD_SWITCH, "biochar production"),
            ("continue the composting conversation", CommandType.THREAD_SWITCH, "composting"),
        ]

        for text, expected_type, expected_name in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == expected_type, f"Failed for: {text}"
            assert expected_name in intent.params.get('thread_name', '')
            assert intent.confidence > 0.8

    def test_thread_listing(self):
        """Test thread listing commands"""
        test_cases = [
            "list my threads",
            "show all active threads",
            "what threads do i have?",
            "threads"
        ]

        for text in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == CommandType.THREAD_LIST, f"Failed for: {text}"

    def test_mode_switching(self):
        """Test mode switching commands"""
        test_cases = [
            ("switch to conversational mode", "conversational"),
            ("enter command mode", "command"),
            ("streaming mode", "streaming"),
        ]

        for text, expected_mode in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == CommandType.MODE_SWITCH, f"Failed for: {text}"
            assert intent.params.get('mode') == expected_mode

    def test_help_commands(self):
        """Test help command detection"""
        test_cases = [
            "help",
            "what can you do?",
            "show commands",
        ]

        for text in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == CommandType.HELP, f"Failed for: {text}"

    def test_conversational_fallback(self):
        """Test that unmatched queries are classified as conversational"""
        test_cases = [
            "what is Thompson Sampling?",
            "how do I plant apple trees?",
            "tell me about biochar",
            "random question here"
        ]

        for text in test_cases:
            intent = self.grammar.classify(text)
            assert intent.command_type == CommandType.CONVERSATIONAL, f"Failed for: {text}"
            assert intent.is_conversational
            assert intent.confidence == 1.0

    def test_case_insensitivity(self):
        """Test that commands are case-insensitive"""
        variations = [
            "START A NEW THREAD FOR TESTING",
            "start a new thread for testing",
            "Start A New Thread For Testing",
        ]

        for text in variations:
            intent = self.grammar.classify(text)
            assert intent.command_type == CommandType.THREAD_CREATE
            assert intent.params.get('topic') in ["TESTING", "testing", "Testing"]

    def test_get_command_examples(self):
        """Test command example retrieval"""
        examples = self.grammar.get_command_examples(CommandType.THREAD_CREATE)
        assert len(examples) > 0
        assert all(isinstance(ex, str) for ex in examples)

    def test_get_all_examples(self):
        """Test retrieval of all command examples"""
        all_examples = self.grammar.get_all_examples()
        assert 'THREAD_CREATE' in all_examples
        assert 'THREAD_SWITCH' in all_examples
        assert 'MODE_SWITCH' in all_examples
        assert 'CONVERSATIONAL' not in all_examples  # Not a command


# ============================================================================
# State Machine Tests
# ============================================================================

class TestVoiceModeStateMachine:
    """Test voice mode state machine"""

    def test_initialization(self):
        """Test state machine initialization"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)
        assert sm.current_mode == VoiceMode.CONVERSATIONAL
        assert len(sm.transition_history) == 1  # Initial transition

    def test_valid_transitions(self):
        """Test valid mode transitions"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)

        # CONVERSATIONAL → COMMAND
        transition = sm.transition(VoiceMode.COMMAND)
        assert sm.current_mode == VoiceMode.COMMAND
        assert transition.to_mode == VoiceMode.COMMAND

        # COMMAND → STREAMING
        transition = sm.transition(VoiceMode.STREAMING)
        assert sm.current_mode == VoiceMode.STREAMING

        # STREAMING → CONVERSATIONAL
        transition = sm.transition(VoiceMode.CONVERSATIONAL)
        assert sm.current_mode == VoiceMode.CONVERSATIONAL

    def test_invalid_transitions(self):
        """Test that invalid transitions raise errors"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)

        # All transitions are valid in current design (undirected graph)
        # This test would apply if we had directed constraints

    def test_can_transition_to(self):
        """Test transition validity checking"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)

        assert sm.can_transition_to(VoiceMode.COMMAND)
        assert sm.can_transition_to(VoiceMode.STREAMING)
        assert sm.can_transition_to(VoiceMode.CONVERSATIONAL)  # Same mode

    def test_transition_history(self):
        """Test transition history tracking"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)

        sm.transition(VoiceMode.COMMAND)
        sm.transition(VoiceMode.STREAMING)
        sm.transition(VoiceMode.CONVERSATIONAL)

        # Should have initial + 3 transitions
        assert len(sm.transition_history) == 4

        recent = sm.get_recent_transitions(n=2)
        assert len(recent) == 2
        assert recent[-1].to_mode == VoiceMode.CONVERSATIONAL

    def test_mode_defaults(self):
        """Test mode-specific default settings"""
        sm = VoiceModeStateMachine(initial_mode=VoiceMode.CONVERSATIONAL)

        defaults = sm.get_mode_defaults()
        assert defaults['use_neural_processing'] == True
        assert defaults['response_style'] == 'natural'

        sm.transition(VoiceMode.COMMAND)
        defaults = sm.get_mode_defaults()
        assert defaults['use_neural_processing'] == False
        assert defaults['response_style'] == 'structured'


# ============================================================================
# Router Tests
# ============================================================================

class TestVoiceRouter:
    """Test voice router with mock handlers"""

    @pytest.fixture
    def mock_handlers(self):
        """Create mock handlers for testing"""
        async def hololoom_handler(text, context):
            return f"HoloLoom response to: {text}"

        async def elle_handler(text, command_data):
            command_type = command_data.get('command_type', 'UNKNOWN')
            return f"Elle handled: {command_type}"

        return hololoom_handler, elle_handler

    def test_router_initialization(self, mock_handlers):
        """Test router initialization"""
        hololoom_handler, elle_handler = mock_handlers

        router = VoiceRouter(
            hololoom_handler=hololoom_handler,
            elle_handler=elle_handler,
            initial_mode=VoiceMode.CONVERSATIONAL
        )

        assert router.current_mode == VoiceMode.CONVERSATIONAL
        assert router.hololoom_handler is not None
        assert router.elle_handler is not None

    def test_classify_delegation(self, mock_handlers):
        """Test that router delegates to grammar"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        intent = router.classify("start a new thread for testing")
        assert intent.command_type == CommandType.THREAD_CREATE

    @pytest.mark.asyncio
    async def test_route_thread_command(self, mock_handlers):
        """Test routing of thread commands to Elle"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        response = await router.route("start a new thread for testing")
        assert "Elle handled" in response
        assert "THREAD_CREATE" in response

    @pytest.mark.asyncio
    async def test_route_conversational(self, mock_handlers):
        """Test routing of conversational queries to HoloLoom"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        response = await router.route("what is Thompson Sampling?")
        assert "HoloLoom response" in response

    @pytest.mark.asyncio
    async def test_route_mode_switch(self, mock_handlers):
        """Test mode switching via router"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        response = await router.route("switch to command mode")
        assert "command" in response.lower()
        assert router.current_mode == VoiceMode.COMMAND

    @pytest.mark.asyncio
    async def test_route_help(self, mock_handlers):
        """Test help command routing"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        response = await router.route("help")
        assert "commands" in response.lower() or "thread" in response.lower()

    @pytest.mark.asyncio
    async def test_route_status(self, mock_handlers):
        """Test status command routing"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        context = {
            'active_thread': 'orchard planning',
            'thread_count': 3
        }

        response = await router.route("status", context=context)
        assert "orchard planning" in response
        assert "3" in response

    def test_get_stats(self, mock_handlers):
        """Test router statistics retrieval"""
        hololoom_handler, elle_handler = mock_handlers
        router = VoiceRouter(hololoom_handler, elle_handler)

        stats = router.get_stats()
        assert 'current_mode' in stats
        assert 'transition_count' in stats
        assert stats['current_mode'] == 'CONVERSATIONAL'


# ============================================================================
# Integration Tests (Basic)
# ============================================================================

class TestBasicIntegration:
    """Basic integration tests without full agent initialization"""

    @pytest.mark.asyncio
    async def test_full_routing_pipeline(self):
        """Test complete routing pipeline with mock handlers"""
        responses = []

        async def mock_hololoom(text, context):
            responses.append(('hololoom', text))
            return f"Processed: {text}"

        async def mock_elle(text, command_data):
            responses.append(('elle', command_data['command_type']))
            return f"Command: {command_data['command_type']}"

        router = VoiceRouter(mock_hololoom, mock_elle)

        # Test sequence
        await router.route("start a new thread for testing")
        await router.route("what is biochar?")
        await router.route("switch to testing thread")

        # Verify routing
        assert len(responses) == 3
        assert responses[0][0] == 'elle'  # Thread creation
        assert responses[1][0] == 'hololoom'  # Conversational
        assert responses[2][0] == 'elle'  # Thread switching


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
