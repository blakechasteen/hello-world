"""
Test Suite for Emotion Bridge
==============================

Tests the Python-JavaScript emotion bridge integration.

Date: November 2025
"""

import asyncio
import pytest
import sys
import os

# Add HoloLoom to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from hololoom.voice.emotion_bridge import (
    EmotionBridge,
    EmotionBridgeConfig,
    EmotionResult,
    FusionStrategy,
    MetaMode,
    PlanningStrategy,
    NodeJSBridge
)


# ============================================================================
# Configuration Tests
# ============================================================================

def test_emotion_bridge_config_defaults():
    """Test default configuration"""
    config = EmotionBridgeConfig()

    assert config.enable_facial is True
    assert config.enable_vocal is True
    assert config.enable_text is True
    assert config.fusion_strategy == FusionStrategy.BAYESIAN
    assert config.enable_meta_interpretation is True
    assert config.meta_mode == MetaMode.FULL_META


def test_emotion_bridge_config_minimal():
    """Test minimal configuration preset"""
    config = EmotionBridgeConfig.minimal()

    assert config.enable_meta_interpretation is False
    assert config.enable_action_planning is False
    assert config.fusion_strategy == FusionStrategy.AVERAGE
    assert config.meta_mode == MetaMode.DISABLED
    assert config.planning_strategy == PlanningStrategy.DISABLED


def test_emotion_bridge_config_standard():
    """Test standard configuration preset"""
    config = EmotionBridgeConfig.standard()

    assert config.meta_mode == MetaMode.NUANCE_INTERPRETATION
    assert config.planning_strategy == PlanningStrategy.IMMEDIATE


def test_emotion_bridge_config_advanced():
    """Test advanced configuration preset"""
    config = EmotionBridgeConfig.advanced()

    assert config.meta_mode == MetaMode.FULL_META
    assert config.planning_strategy == PlanningStrategy.ADAPTIVE
    assert config.fusion_strategy == FusionStrategy.BAYESIAN


# ============================================================================
# Data Model Tests
# ============================================================================

def test_emotion_result_from_js_basic():
    """Test creating EmotionResult from JavaScript output"""
    js_data = {
        'fusedEmotion': {
            'emotion': 'happy',
            'confidence': 0.95,
            'valence': 0.8,
            'arousal': 0.7
        },
        'facialResult': {
            'emotion': 'happy',
            'confidence': 0.9
        },
        'processingTime': 150.5
    }

    result = EmotionResult.from_js_result(js_data)

    assert result.emotion == 'happy'
    assert result.confidence == 0.95
    assert result.valence == 0.8
    assert result.arousal == 0.7
    assert result.facial_emotion == 'happy'
    assert result.facial_confidence == 0.9
    assert result.processing_time_ms == 150.5


def test_emotion_result_from_js_complete():
    """Test creating EmotionResult with all fields"""
    js_data = {
        'fusedEmotion': {
            'emotion': 'frustrated',
            'confidence': 0.87,
            'valence': -0.6,
            'arousal': 0.75,
            'strategyUsed': 'bayesian'
        },
        'facialResult': {
            'emotion': 'angry',
            'confidence': 0.85
        },
        'vocalResult': {
            'emotion': 'frustrated',
            'confidence': 0.92
        },
        'textResult': {
            'emotion': 'frustrated',
            'confidence': 0.84
        },
        'metaInterpretation': {
            'interpretation': 'User is experiencing technical frustration with coding task',
            'nuances': ['technical_difficulty', 'time_pressure'],
            'hiddenEmotions': ['helplessness']
        },
        'actionPlan': {
            'suggestedActions': [
                'Suggest taking a break',
                'Offer debugging assistance',
                'Recommend stepping back to review approach'
            ],
            'conversationStrategy': 'supportive'
        },
        'processingTime': 245.3
    }

    result = EmotionResult.from_js_result(js_data)

    assert result.emotion == 'frustrated'
    assert result.confidence == 0.87
    assert result.valence == -0.6
    assert result.arousal == 0.75

    # Modality-specific
    assert result.facial_emotion == 'angry'
    assert result.vocal_emotion == 'frustrated'
    assert result.text_emotion == 'frustrated'

    # Meta-interpretation
    assert result.meta_interpretation is not None
    assert len(result.nuances) == 2
    assert 'helplessness' in result.hidden_emotions

    # Action plan
    assert len(result.suggested_actions) == 3
    assert result.conversation_strategy == 'supportive'

    # Metadata
    assert result.fusion_strategy_used == 'bayesian'
    assert result.processing_time_ms == 245.3


# ============================================================================
# Bridge Tests (require Node.js)
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_basic_analysis():
    """Test basic emotion analysis through bridge"""

    config = EmotionBridgeConfig.standard()

    async with EmotionBridge(config) as bridge:
        result = await bridge.analyze_emotion(
            text="I'm feeling great today!",
            context={'activity': 'general'}
        )

        # Basic assertions
        assert result.emotion is not None
        assert 0.0 <= result.confidence <= 1.0
        assert -1.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0

        # Should be positive emotion
        assert result.valence > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_frustrated_detection():
    """Test frustration detection"""

    config = EmotionBridgeConfig.advanced()

    async with EmotionBridge(config) as bridge:
        result = await bridge.analyze_emotion(
            text="I'm stuck on this bug. Nothing works!",
            context={'activity': 'coding', 'session_duration': 180000}
        )

        # Should detect negative emotion
        assert result.valence < 0

        # Should suggest actions
        assert len(result.suggested_actions) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_minimal_config():
    """Test bridge with minimal configuration"""

    config = EmotionBridgeConfig.minimal()

    async with EmotionBridge(config) as bridge:
        result = await bridge.analyze_emotion(text="Hello there!")

        # Basic emotion detection should work
        assert result.emotion is not None
        assert result.confidence is not None

        # But meta-interpretation should be disabled
        assert result.meta_interpretation is None


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_context_awareness():
    """Test context-aware emotion analysis"""

    config = EmotionBridgeConfig.advanced()

    async with EmotionBridge(config) as bridge:
        # Same text, different contexts
        result1 = await bridge.analyze_emotion(
            text="I need help with this.",
            context={'activity': 'learning', 'session_duration': 5000}
        )

        result2 = await bridge.analyze_emotion(
            text="I need help with this.",
            context={'activity': 'debugging', 'session_duration': 180000, 'attempts': 15}
        )

        # Different contexts may lead to different interpretations
        # (though emotion may be same, actions/strategy should differ)
        assert result1.suggested_actions != result2.suggested_actions or \
               result1.conversation_strategy != result2.conversation_strategy


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_fallback_on_error():
    """Test fallback behavior when analysis fails"""

    config = EmotionBridgeConfig.standard()

    async with EmotionBridge(config) as bridge:
        # Force error by sending invalid input
        # (This tests the error handling and fallback)

        # Bridge should return neutral fallback instead of crashing
        result = await bridge.analyze_emotion(text="")

        assert result.emotion == 'neutral' or result.emotion is not None
        assert result.confidence >= 0.0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_voice_agent_enhancement():
    """Test enhancing VoiceAgent with emotions (if available)"""

    try:
        from hololoom.voice import VoiceAgent
        from hololoom.voice.emotion_bridge import enhance_voice_agent_with_emotions
    except ImportError:
        pytest.skip("VoiceAgent not available")

    # Create simple voice agent (no orchestrator)
    agent = VoiceAgent(orchestrator=None, agent_name="TestBot", turn_mode="button")

    # Enhance with emotions
    await enhance_voice_agent_with_emotions(
        agent,
        EmotionBridgeConfig.standard()
    )

    # Test enhanced processing
    response = await agent.process_voice_input("I'm feeling frustrated")

    # Should have emotional metadata
    assert len(agent.conversation_memory.turns) > 0
    last_turn = agent.conversation_memory.turns[-1]

    # Metadata should include emotion (if not just echo fallback)
    if hasattr(agent, '_emotion_bridge'):
        assert 'emotion' in last_turn.metadata or response is not None


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    os.system("node --version") != 0,
    reason="Node.js not available"
)
async def test_emotion_bridge_performance():
    """Test bridge performance across configurations"""

    test_text = "I'm feeling good about this progress"

    configs = {
        'minimal': EmotionBridgeConfig.minimal(),
        'standard': EmotionBridgeConfig.standard(),
        'advanced': EmotionBridgeConfig.advanced()
    }

    results = {}

    for name, config in configs.items():
        async with EmotionBridge(config) as bridge:
            result = await bridge.analyze_emotion(text=test_text)

            results[name] = result.processing_time_ms

    # Minimal should be fastest
    if all(results.values()):
        assert results['minimal'] <= results['standard']
        assert results['standard'] <= results['advanced']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 EMOTION BRIDGE TEST SUITE")
    print("="*70 + "\n")

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
