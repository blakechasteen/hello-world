"""
Test Suite for Shared Dream Synchronization System

Tests all components of the shared dream creation and execution pipeline.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime

from shared_dream_sync import (
    SharedDreamSynchronizer,
    SharedDreamSession,
    SharedDreamType,
    SymbolBlendingStrategy,
    SymbolBlendingEngine,
    PerspectiveGenerator,
    WakingEffectsEngine,
    ParticipantContribution,
)

# Import from symbolic encoder
try:
    from symbolic_encoder import (
        SymbolArchetype, DreamScene, DreamContext, SymbolicEncoder
    )
    HAVE_ENCODER = True
except ImportError:
    HAVE_ENCODER = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_symbols():
    """Create mock symbol archetypes for testing."""
    symbols = {}

    # Caged bird symbol
    symbols['caged_bird'] = SymbolArchetype(
        symbol_id='caged_bird',
        category='trapped',
        embedding=np.random.randn(228),
        emotion_tags=['trapped', 'yearning', 'helpless'],
        context_tags=['work', 'relationship'],
        universality_score=0.9,
        literary_references=['The Nightingale', 'I Know Why the Caged Bird Sings'],
        complementary_symbols=['sky', 'wind', 'freedom'],
        visual_complexity='moderate',
        color_palette=['gold', 'grey', 'white']
    )

    # Storm cloud symbol
    symbols['storm_cloud'] = SymbolArchetype(
        symbol_id='storm_cloud',
        category='anger',
        embedding=np.random.randn(228),
        emotion_tags=['anger', 'chaos', 'power'],
        context_tags=['conflict', 'general'],
        universality_score=0.9,
        literary_references=['King Lear', 'The Tempest'],
        complementary_symbols=['lightning', 'rain', 'wind'],
        visual_complexity='moderate',
        color_palette=['dark_grey', 'black', 'white']
    )

    # Mirror symbol
    symbols['mirror'] = SymbolArchetype(
        symbol_id='mirror',
        category='truth',
        embedding=np.random.randn(228),
        emotion_tags=['reflection', 'truth', 'identity'],
        context_tags=['general'],
        universality_score=0.85,
        literary_references=['Snow White', 'Narcissus'],
        complementary_symbols=['light', 'water'],
        visual_complexity='simple',
        color_palette=['silver', 'grey']
    )

    # Quicksand symbol
    symbols['quicksand'] = SymbolArchetype(
        symbol_id='quicksand',
        category='trapped',
        embedding=np.random.randn(228),
        emotion_tags=['trapped', 'sinking', 'helpless'],
        context_tags=['danger', 'struggle'],
        universality_score=0.8,
        literary_references=['Desert stories'],
        complementary_symbols=['ground', 'water'],
        visual_complexity='moderate',
        color_palette=['brown', 'tan', 'grey']
    )

    return symbols


@pytest.fixture
def mock_participants():
    """Create mock participants with private facts."""
    return [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'Feels trapped at my corporate job'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'Struggling with unresolved anger toward my brother'
        }
    ]


@pytest.fixture
def mock_participants_triad():
    """Create mock participants for triad dream."""
    return [
        {
            'id': 'alice_001',
            'name': 'Alice',
            'private_fact': 'Feels trapped at my corporate job'
        },
        {
            'id': 'bob_001',
            'name': 'Bob',
            'private_fact': 'Struggling with unresolved anger'
        },
        {
            'id': 'charlie_001',
            'name': 'Charlie',
            'private_fact': 'Lost sense of identity after career change'
        }
    ]


# ============================================================================
# Test Symbol Blending Engine
# ============================================================================


class TestSymbolBlendingEngine:
    """Test the symbol blending engine."""

    @pytest.mark.asyncio
    async def test_blend_two_symbols(self, mock_symbols):
        """Test blending two complementary symbols."""
        engine = SymbolBlendingEngine()

        # Create contributions
        contributions = {
            'alice_001': ParticipantContribution(
                resident_id='alice_001',
                resident_name='Alice',
                private_fact='Trapped at work',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob_001': ParticipantContribution(
                resident_id='bob_001',
                resident_name='Bob',
                private_fact='Angry and confused',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            )
        }

        result = await engine.blend_symbols(contributions)

        assert result.blended_scene is not None
        assert result.interaction_description != ""
        assert result.metaphor_explanation != ""
        assert result.emotional_resonance > 0.0
        assert 'alice_001' in result.symbol_roles
        assert 'bob_001' in result.symbol_roles

    @pytest.mark.asyncio
    async def test_blend_three_symbols(self, mock_symbols):
        """Test blending three symbols."""
        engine = SymbolBlendingEngine()

        contributions = {
            'alice': ParticipantContribution(
                resident_id='alice',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob': ParticipantContribution(
                resident_id='bob',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            ),
            'charlie': ParticipantContribution(
                resident_id='charlie',
                resident_name='Charlie',
                private_fact='Lost',
                symbol=mock_symbols['mirror'],
                emotional_essence={'primary_emotion': 'lost', 'intensity': 0.6},
                perspective_prompt=''
            )
        }

        result = await engine.blend_symbols(contributions)

        assert len(result.symbol_roles) == 3
        assert result.emotional_resonance > 0.0

    @pytest.mark.asyncio
    async def test_different_blending_strategies(self, mock_symbols):
        """Test different blending strategies."""
        engine = SymbolBlendingEngine()

        contributions = {
            'alice': ParticipantContribution(
                resident_id='alice',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob': ParticipantContribution(
                resident_id='bob',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            )
        }

        for strategy in SymbolBlendingStrategy:
            result = await engine.blend_symbols(contributions, strategy=strategy)
            assert result.blended_scene is not None

    def test_interaction_strength(self, mock_symbols):
        """Test interaction strength calculation."""
        engine = SymbolBlendingEngine()

        # Test two different symbols
        strength = engine._interaction_strength(
            mock_symbols['caged_bird'],
            mock_symbols['storm_cloud']
        )
        assert 0.0 <= strength <= 1.0

        # Test same-category symbols
        strength = engine._interaction_strength(
            mock_symbols['caged_bird'],
            mock_symbols['quicksand']
        )
        assert strength > 0.0


# ============================================================================
# Test Perspective Generator
# ============================================================================


class TestPerspectiveGenerator:
    """Test perspective generation."""

    @pytest.mark.asyncio
    async def test_generate_perspective_dyadic(self, mock_symbols):
        """Test generating perspective for dyadic dream."""
        gen = PerspectiveGenerator()

        contributions = {
            'alice': ParticipantContribution(
                resident_id='alice',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob': ParticipantContribution(
                resident_id='bob',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            )
        }

        scene = DreamScene(
            description="A bird and a storm face each other",
            symbolic_meaning="Separation and recognition"
        )

        perspectives = await gen.generate_perspectives(contributions, scene)

        assert 'alice' in perspectives
        assert 'bob' in perspectives
        assert len(perspectives['alice']) > 0
        assert len(perspectives['bob']) > 0

    @pytest.mark.asyncio
    async def test_perspective_different_per_resident(self, mock_symbols):
        """Test that each resident gets unique perspective."""
        gen = PerspectiveGenerator()

        contributions = {
            'alice': ParticipantContribution(
                resident_id='alice',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob': ParticipantContribution(
                resident_id='bob',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            )
        }

        scene = DreamScene(
            description="A shared scene",
            symbolic_meaning="Shared meaning"
        )

        perspectives = await gen.generate_perspectives(contributions, scene)

        # Perspectives should be different
        assert perspectives['alice'] != perspectives['bob']


# ============================================================================
# Test Waking Effects
# ============================================================================


class TestWakingEffectsEngine:
    """Test waking effects calculation."""

    def test_calculate_basic_effects(self, mock_symbols, mock_participants):
        """Test basic waking effects calculation."""
        engine = WakingEffectsEngine()

        session = SharedDreamSession(
            session_id='test_001',
            participants=['alice_001', 'bob_001'],
            dream_type=SharedDreamType.DYADIC,
            consciousness_level=0.5
        )

        session.contributions = {
            'alice_001': ParticipantContribution(
                resident_id='alice_001',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.8},
                perspective_prompt=''
            ),
            'bob_001': ParticipantContribution(
                resident_id='bob_001',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.7},
                perspective_prompt=''
            )
        }

        effects = engine.calculate_effects(session)

        assert ('alice_001', 'bob_001') in effects
        assert 'relationship_strength' in effects[('alice_001', 'bob_001')]
        assert 'mutual_understanding' in effects[('alice_001', 'bob_001')]
        assert effects[('alice_001', 'bob_001')]['relationship_strength'] > 0.0

    def test_effects_scale_with_resonance(self, mock_symbols):
        """Test that effects scale with emotional resonance."""
        engine = WakingEffectsEngine()

        session = SharedDreamSession(
            session_id='test_001',
            participants=['alice', 'bob'],
            dream_type=SharedDreamType.DYADIC,
            consciousness_level=0.5,
            intensity=0.9  # High resonance
        )

        session.contributions = {
            'alice': ParticipantContribution(
                resident_id='alice',
                resident_name='Alice',
                private_fact='Trapped',
                symbol=mock_symbols['caged_bird'],
                emotional_essence={'primary_emotion': 'trapped', 'intensity': 0.9},
                perspective_prompt=''
            ),
            'bob': ParticipantContribution(
                resident_id='bob',
                resident_name='Bob',
                private_fact='Angry',
                symbol=mock_symbols['storm_cloud'],
                emotional_essence={'primary_emotion': 'anger', 'intensity': 0.9},
                perspective_prompt=''
            )
        }

        # Import SymbolBlendingResult from shared_dream_sync module
        from shared_dream_sync import SymbolBlendingResult
        blending_result = SymbolBlendingResult(
            blended_scene=DreamScene(description="test", symbolic_meaning="test meaning"),
            interaction_description="test",
            metaphor_explanation="test",
            symbol_roles={'alice': 'seeker', 'bob': 'guide'},
            emotional_resonance=0.85  # High resonance
        )

        effects = engine.calculate_effects(session, blending_result)

        # Should have high effects with high resonance
        assert effects[('alice', 'bob')]['relationship_strength'] >= 0.20

    def test_pairwise_effects_in_triad(self):
        """Test that all pairs get effects in triad dream."""
        engine = WakingEffectsEngine()

        session = SharedDreamSession(
            session_id='test_001',
            participants=['alice', 'bob', 'charlie'],
            dream_type=SharedDreamType.TRIAD,
            consciousness_level=0.5
        )

        effects = engine.calculate_effects(session)

        # Should have effects for all pairs
        assert ('alice', 'bob') in effects
        assert ('alice', 'charlie') in effects
        assert ('bob', 'charlie') in effects
        assert len(effects) == 3


# ============================================================================
# Test Main Synchronizer
# ============================================================================


class TestSharedDreamSynchronizer:
    """Test the main synchronizer."""

    @pytest.mark.asyncio
    async def test_create_dyadic_dream(self, mock_symbols, mock_participants):
        """Test creating a dyadic (2-person) shared dream."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(mock_participants)

        assert session.session_id != ""
        assert session.dream_type == SharedDreamType.DYADIC
        assert len(session.participants) == 2
        assert 0.33 <= session.consciousness_level <= 0.66

    @pytest.mark.asyncio
    async def test_create_triad_dream(self, mock_participants_triad):
        """Test creating a triad (3-person) shared dream."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(mock_participants_triad)

        assert session.dream_type == SharedDreamType.TRIAD
        assert len(session.participants) == 3
        assert len(session.contributions) == 3

    @pytest.mark.asyncio
    async def test_consciousness_level_clamping(self, mock_participants):
        """Test that consciousness level is clamped to shared range."""
        sync = SharedDreamSynchronizer()

        # Try with consciousness level outside shared range
        session = await sync.create_shared_dream(
            mock_participants,
            consciousness_level=0.1  # Too low
        )
        assert session.consciousness_level >= 0.33

        session = await sync.create_shared_dream(
            mock_participants,
            consciousness_level=0.9  # Too high
        )
        assert session.consciousness_level <= 0.66

    @pytest.mark.asyncio
    async def test_too_few_participants_error(self, mock_participants):
        """Test that single participant raises error."""
        sync = SharedDreamSynchronizer()

        with pytest.raises(ValueError):
            await sync.create_shared_dream(mock_participants[:1])

    @pytest.mark.asyncio
    async def test_too_many_participants_error(self, mock_participants):
        """Test that >5 participants raises error."""
        sync = SharedDreamSynchronizer()

        too_many = mock_participants + [
            {
                'id': f'resident_{i}',
                'name': f'Resident {i}',
                'private_fact': f'Fact {i}'
            }
            for i in range(4)
        ]

        with pytest.raises(ValueError):
            await sync.create_shared_dream(too_many)

    @pytest.mark.asyncio
    async def test_blending_strategies(self, mock_participants):
        """Test different blending strategies."""
        sync = SharedDreamSynchronizer()

        for strategy in SymbolBlendingStrategy:
            session = await sync.create_shared_dream(
                mock_participants,
                blending_strategy=strategy
            )
            assert session.blending_strategy == strategy
            assert session.shared_narrative != ""

    @pytest.mark.asyncio
    async def test_session_has_perspectives(self, mock_participants):
        """Test that session has perspectives for all participants."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(mock_participants)

        for res_id in session.participants:
            assert res_id in session.participant_perspectives
            assert len(session.participant_perspectives[res_id]) > 0

    @pytest.mark.asyncio
    async def test_session_has_waking_effects(self, mock_participants):
        """Test that session calculates waking effects."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(mock_participants)

        assert len(session.waking_effects) > 0
        for pair, effects in session.waking_effects.items():
            assert 'relationship_strength' in effects
            assert 'mutual_understanding' in effects

    @pytest.mark.asyncio
    async def test_session_summary(self, mock_participants):
        """Test getting session summary."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(mock_participants)
        summary = sync.get_session_summary(session)

        assert summary['session_id'] == session.session_id
        assert summary['type'] == 'dyadic'
        assert len(summary['participants']) == 2
        assert summary['intensity'] > 0.0
        assert summary['duration_minutes'] > 0.0

    def test_session_serialization(self, mock_participants):
        """Test that session can be serialized to dict."""
        session = SharedDreamSession(
            session_id='test_001',
            participants=['alice', 'bob'],
            dream_type=SharedDreamType.DYADIC,
            consciousness_level=0.5
        )

        data = session.to_dict()

        assert data['session_id'] == 'test_001'
        assert data['dream_type'] == 'dyadic'
        assert len(data['participants']) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_dyadic_dream_pipeline(self, mock_participants):
        """Test complete pipeline for dyadic dream."""
        sync = SharedDreamSynchronizer()

        # Create dream
        session = await sync.create_shared_dream(
            mock_participants,
            consciousness_level=0.5,
            blending_strategy=SymbolBlendingStrategy.INTERACTION
        )

        # Verify all components
        assert session.dream_scene is not None
        assert len(session.participant_perspectives) == 2
        assert len(session.waking_effects) > 0
        assert session.intensity > 0.0
        assert session.duration_minutes > 5.0

        # Verify summary
        summary = sync.get_session_summary(session)
        assert summary['intensity'] > 0.0

    @pytest.mark.asyncio
    async def test_full_triad_dream_pipeline(self, mock_participants_triad):
        """Test complete pipeline for triad dream."""
        sync = SharedDreamSynchronizer()

        session = await sync.create_shared_dream(
            mock_participants_triad,
            consciousness_level=0.5,
            blending_strategy=SymbolBlendingStrategy.NARRATIVE
        )

        assert session.dream_type == SharedDreamType.TRIAD
        assert len(session.participant_perspectives) == 3
        assert len(session.waking_effects) == 3  # 3 pairs
        assert session.duration_minutes > 10.0  # Should be longer


if __name__ == "__main__":
    # Run with: pytest test_shared_dream_sync.py -v
    pytest.main([__file__, "-v"])
