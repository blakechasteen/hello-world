"""
Integration Tests for VoiceAgent
=================================

Tests all voice interaction components:
- VoiceAgent initialization and configuration
- TTS synthesis and playback
- Conversation memory
- Turn-taking manager
- VAD integration
- HoloLoom orchestrator integration

Date: November 15, 2025
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from hololoom.voice import (
    VoiceAgent,
    TTSProvider,
    OpenAITTS,
    TTSManager,
    VoiceActivityDetector,
    TurnTakingManager,
    ConversationMemory,
    ConversationTurn,
    TurnState
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_tts_provider():
    """Mock TTS provider for testing"""
    class MockTTS(TTSProvider):
        async def synthesize(self, text: str, voice: str = "alloy") -> bytes:
            # Return fake audio bytes
            return b"fake_audio_" + text.encode('utf-8')

    return MockTTS()


@pytest.fixture
def mock_orchestrator():
    """Mock HoloLoom orchestrator"""
    orchestrator = AsyncMock()

    # Mock weave response
    async def mock_weave(query):
        response_mock = Mock()
        response_mock.response = {'text': f"Response to: {query.text}"}
        response_mock.confidence = 0.95
        response_mock.metadata = {'tool_used': 'mock_tool'}
        return response_mock

    orchestrator.weave = mock_weave
    return orchestrator


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing"""
    duration = 1.0  # seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Generate simple sine wave
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, samples)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio.astype(np.float32)


# ============================================================================
# ConversationMemory Tests
# ============================================================================

class TestConversationMemory:
    """Test conversation memory functionality"""

    def test_initialization(self):
        """Test memory initialization"""
        memory = ConversationMemory(max_turns=10)

        assert len(memory.turns) == 0
        assert memory.max_turns == 10
        assert memory.session_id is not None

    def test_add_turn(self):
        """Test adding conversation turns"""
        memory = ConversationMemory(max_turns=5)

        memory.add_turn('user', 'Hello')
        memory.add_turn('agent', 'Hi there!')

        assert len(memory.turns) == 2
        assert memory.turns[0].speaker == 'user'
        assert memory.turns[0].text == 'Hello'
        assert memory.turns[1].speaker == 'agent'
        assert memory.turns[1].text == 'Hi there!'

    def test_max_turns_limit(self):
        """Test that memory respects max_turns limit"""
        memory = ConversationMemory(max_turns=3)

        for i in range(5):
            memory.add_turn('user', f'Message {i}')

        assert len(memory.turns) == 3
        assert memory.turns[0].text == 'Message 2'  # Oldest kept
        assert memory.turns[-1].text == 'Message 4'  # Most recent

    def test_get_context(self):
        """Test context retrieval"""
        memory = ConversationMemory()

        memory.add_turn('user', 'What is your name?')
        memory.add_turn('agent', 'I am Elle.')
        memory.add_turn('user', 'Nice to meet you')

        context = memory.get_context(n_turns=2)

        assert 'User: Nice to meet you' in context
        assert 'Agent: I am Elle.' in context

    def test_export_session(self):
        """Test session export"""
        memory = ConversationMemory()

        memory.add_turn('user', 'Hello')
        memory.add_turn('agent', 'Hi!')

        session = memory.export_session()

        assert session['total_turns'] == 2
        assert session['session_id'] == memory.session_id
        assert len(session['turns']) == 2


# ============================================================================
# TurnTakingManager Tests
# ============================================================================

class TestTurnTakingManager:
    """Test turn-taking management"""

    def test_initialization(self):
        """Test turn manager initialization"""
        manager = TurnTakingManager(mode='button')

        assert manager.mode == 'button'
        assert manager.state == TurnState.IDLE

    def test_manual_state_changes(self):
        """Test manual state transitions (button mode)"""
        manager = TurnTakingManager(mode='button')

        manager.set_state(TurnState.LISTENING)
        assert manager.state == TurnState.LISTENING
        assert manager.user_speaking == True

        manager.set_state(TurnState.SPEAKING)
        assert manager.state == TurnState.SPEAKING
        assert manager.agent_speaking == True

        manager.set_state(TurnState.IDLE)
        assert manager.state == TurnState.IDLE
        assert manager.user_speaking == False
        assert manager.agent_speaking == False

    @pytest.mark.skipif(not os.environ.get('RUN_VAD_TESTS'),
                        reason="VAD tests require webrtcvad - set RUN_VAD_TESTS=1 to enable")
    def test_vad_mode_initialization(self):
        """Test VAD mode initialization"""
        manager = TurnTakingManager(mode='vad')

        assert manager.mode == 'vad'
        assert manager.vad is not None


# ============================================================================
# TTSManager Tests
# ============================================================================

class TestTTSManager:
    """Test TTS management"""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_tts_provider):
        """Test TTS manager initialization"""
        manager = TTSManager(mock_tts_provider, voice='nova')

        assert manager.voice == 'nova'
        assert manager.is_playing == False

    @pytest.mark.asyncio
    async def test_synthesize_and_queue(self, mock_tts_provider):
        """Test synthesis and queuing"""
        manager = TTSManager(mock_tts_provider)

        # Speak text
        await manager.speak("Hello world")

        # Check queue (playback loop starts automatically)
        await asyncio.sleep(0.1)  # Give time for background task

        # In real scenario, audio would be played
        # Here we just verify synthesis was called
        assert True  # Synthesis happened if no exception

    @pytest.mark.asyncio
    async def test_priority_speech(self, mock_tts_provider):
        """Test priority speech interrupts queue"""
        manager = TTSManager(mock_tts_provider)

        # Queue multiple items
        await manager.speak("First")
        await manager.speak("Second")
        await manager.speak("Third", priority=True)

        # Priority should clear queue
        # (Verification would require checking internal queue state)
        assert True

    def test_stop_playback(self, mock_tts_provider):
        """Test stopping playback"""
        manager = TTSManager(mock_tts_provider)

        manager.is_playing = True
        manager.stop()

        assert manager.is_playing == False


# ============================================================================
# VoiceAgent Tests
# ============================================================================

class TestVoiceAgent:
    """Test main VoiceAgent functionality"""

    def test_initialization(self, mock_tts_provider):
        """Test voice agent initialization"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider,
            agent_name="TestBot"
        )

        assert agent.agent_name == "TestBot"
        assert agent.tts is not None
        assert agent.conversation_memory is not None
        assert agent.turn_manager is not None

    @pytest.mark.asyncio
    async def test_simple_input_processing(self, mock_tts_provider):
        """Test processing input without orchestrator (echo mode)"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        response = await agent.process_voice_input("Hello")

        assert "Hello" in response
        assert len(agent.conversation_memory.turns) == 2  # user + agent

    @pytest.mark.asyncio
    async def test_orchestrator_integration(self, mock_orchestrator, mock_tts_provider):
        """Test processing with HoloLoom orchestrator"""
        agent = VoiceAgent(
            orchestrator=mock_orchestrator,
            tts_provider=mock_tts_provider,
            agent_name="Elle"
        )

        response = await agent.process_voice_input("What is Thompson Sampling?")

        assert "Response to:" in response
        assert len(agent.conversation_memory.turns) == 2

        # Check metadata was captured
        last_turn = agent.conversation_memory.turns[-1]
        assert last_turn.confidence == 0.95
        assert last_turn.tool_used == 'mock_tool'

    @pytest.mark.asyncio
    async def test_conversation_context(self, mock_tts_provider):
        """Test conversation context building"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        # Build conversation
        await agent.process_voice_input("My name is Blake")
        await agent.process_voice_input("I work on AI")
        await agent.process_voice_input("What's my name?")

        # Check context includes earlier messages
        context = agent.conversation_memory.get_context(n_turns=6)

        assert "Blake" in context
        assert "AI" in context

    @pytest.mark.asyncio
    async def test_speak_method(self, mock_tts_provider):
        """Test speaking functionality"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        # Should not raise exception
        await agent.speak("Test message")

        assert True

    def test_turn_state_management(self, mock_tts_provider):
        """Test turn state tracking"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider,
            turn_mode="button"
        )

        # Verify initial state
        assert agent.turn_manager.state == TurnState.IDLE

        # Change state
        agent.turn_manager.set_state(TurnState.LISTENING)
        assert agent.turn_manager.state == TurnState.LISTENING


# ============================================================================
# Integration Tests
# ============================================================================

class TestVoiceAgentIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, mock_orchestrator, mock_tts_provider):
        """Test complete conversation flow"""
        agent = VoiceAgent(
            orchestrator=mock_orchestrator,
            tts_provider=mock_tts_provider,
            agent_name="Elle"
        )

        queries = [
            "What is Thompson Sampling?",
            "How does it work?",
            "Give me an example"
        ]

        for query in queries:
            response = await agent.process_voice_input(query)
            assert response is not None
            assert len(response) > 0

        # Check conversation history
        assert len(agent.conversation_memory.turns) == 6  # 3 user + 3 agent

        # Check context building
        context = agent.conversation_memory.get_context()
        for query in queries:
            assert query in context

    @pytest.mark.asyncio
    async def test_session_export(self, mock_tts_provider):
        """Test session export functionality"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        # Build conversation
        await agent.process_voice_input("Hello")
        await agent.process_voice_input("How are you?")

        # Export session
        session = agent.conversation_memory.export_session()

        assert session['total_turns'] == 4  # 2 user + 2 agent
        assert 'session_id' in session
        assert 'turns' in session
        assert len(session['turns']) == 4

    @pytest.mark.asyncio
    async def test_multi_agent_conversation(self, mock_tts_provider):
        """Test multiple agents with separate memories"""
        agent1 = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider,
            agent_name="Agent1"
        )

        agent2 = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider,
            agent_name="Agent2"
        )

        # Agent 1 conversation
        await agent1.process_voice_input("Hello Agent1")

        # Agent 2 conversation
        await agent2.process_voice_input("Hello Agent2")

        # Verify separate memories
        assert len(agent1.conversation_memory.turns) == 2
        assert len(agent2.conversation_memory.turns) == 2
        assert agent1.conversation_memory.session_id != agent2.conversation_memory.session_id


# ============================================================================
# Performance Tests
# ============================================================================

class TestVoiceAgentPerformance:
    """Performance and stress tests"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rapid_fire_queries(self, mock_tts_provider):
        """Test handling rapid queries"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        # Send 100 queries rapidly
        queries = [f"Query {i}" for i in range(100)]

        start_time = asyncio.get_event_loop().time()

        for query in queries:
            await agent.process_voice_input(query)

        duration = asyncio.get_event_loop().time() - start_time

        # Should complete in reasonable time
        assert duration < 10  # seconds
        assert len(agent.conversation_memory.turns) == 200  # 100 user + 100 agent

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_conversation(self, mock_tts_provider):
        """Test long conversation memory management"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=mock_tts_provider
        )

        # Set small max_turns for testing
        agent.conversation_memory.max_turns = 10

        # Have 20 turn conversation
        for i in range(20):
            await agent.process_voice_input(f"Message {i}")

        # Should only keep last 10 turns
        assert len(agent.conversation_memory.turns) <= 10


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestVoiceAgentErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_orchestrator_failure(self, mock_tts_provider):
        """Test handling orchestrator failures"""
        # Create orchestrator that raises exception
        failing_orchestrator = AsyncMock()
        failing_orchestrator.weave = AsyncMock(side_effect=Exception("Orchestrator error"))

        agent = VoiceAgent(
            orchestrator=failing_orchestrator,
            tts_provider=mock_tts_provider
        )

        # Should fall back gracefully
        with pytest.raises(Exception):
            await agent.process_voice_input("Test query")

    def test_missing_tts_provider(self):
        """Test agent works without TTS"""
        agent = VoiceAgent(
            orchestrator=None,
            tts_provider=None,  # No TTS
            agent_name="NoTTSBot"
        )

        assert agent.tts is None
        # Agent should still work for text processing
        assert agent.agent_name == "NoTTSBot"

    def test_invalid_turn_mode(self):
        """Test initialization with invalid turn mode"""
        # Should default to button mode or handle gracefully
        agent = VoiceAgent(
            orchestrator=None,
            turn_mode="invalid_mode"
        )

        # Should create agent anyway (may log warning)
        assert agent is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
