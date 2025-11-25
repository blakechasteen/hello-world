"""
Test Suite for Symbolic Encoder
================================

Tests all 4 layers of the symbolic encoder:
1. Emotional Extraction
2. Symbol Selection
3. Disambiguation
4. Narrative Integration

Plus end-to-end integration tests.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from symbolic_encoder import (
    EmotionalExtractor,
    SymbolSelector,
    SymbolDisambiguator,
    SymbolIntegrator,
    SymbolicEncoder,
    EmotionalEssence,
    SymbolArchetype,
    DreamScene,
    DreamContext,
    create_mock_symbol_database,
    save_symbol_database
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_symbols():
    """Create mock symbol database."""
    return create_mock_symbol_database(num_symbols=20)


@pytest.fixture
def mock_symbol_db_file(tmp_path, mock_symbols):
    """Create temporary symbol database file."""
    db_path = tmp_path / "test_symbols.json"
    save_symbol_database(mock_symbols, str(db_path))
    return str(db_path)


@pytest.fixture
def sample_essence():
    """Sample emotional essence."""
    return EmotionalEssence(
        primary_emotion="trapped",
        intensity=0.8,
        context="work",
        temporal="chronic",
        valence="negative",
        secondary_emotions=["powerless", "exhausted"],
        complexity_score=0.6,
        embedding=np.random.randn(228)
    )


@pytest.fixture
def sample_context():
    """Sample dream context."""
    return DreamContext(
        primary_dreamer="Alice",
        is_shared_dream=False,
        setting_landscape="forest",
        setting_time="twilight",
        setting_atmosphere="mysterious"
    )


# ============================================================================
# Layer 1: Emotional Extraction Tests
# ============================================================================


def test_emotional_extractor_basic():
    """Test basic emotional extraction."""
    extractor = EmotionalExtractor()

    fact = "Bob hates his factory job and feels trapped"
    essence = extractor.extract(fact)

    assert isinstance(essence, EmotionalEssence)
    assert essence.primary_emotion in ["trapped", "anger", "sadness"]
    assert 0.0 <= essence.intensity <= 1.0
    assert essence.context in ["work", "general"]
    assert essence.temporal in ["acute", "chronic", "episodic"]


def test_emotional_extractor_nlp_analysis():
    """Test NLP emotion detection."""
    extractor = EmotionalExtractor()

    # Test trapped emotion
    nlp_result = extractor._nlp_analyze("I feel stuck and confined")
    assert "trapped" in nlp_result["emotions"]

    # Test guilt emotion
    nlp_result = extractor._nlp_analyze("I feel guilty about what I did")
    assert "guilt" in nlp_result["emotions"]


def test_emotional_extractor_blending():
    """Test emotion source blending."""
    extractor = EmotionalExtractor()

    nlp = {"primary": "trapped", "intensity": 0.7, "emotions": {"trapped": 0.7}}
    live = {"emotions": {"trapped": 0.9}}
    llm = {"primary_emotion": "trapped", "intensity": 0.8, "context": "work", "temporal": "chronic"}

    blended = extractor._blend_sources(nlp, live, llm)

    assert blended["primary_emotion"] == "trapped"
    assert 0.0 <= blended["intensity"] <= 1.0
    assert blended["context"] == "work"
    assert blended["temporal"] == "chronic"


def test_emotional_essence_serialization(sample_essence):
    """Test essence to/from dict."""
    # To dict
    data = sample_essence.to_dict()
    assert isinstance(data, dict)
    assert "primary_emotion" in data
    assert "embedding" in data

    # From dict
    restored = EmotionalEssence.from_dict(data)
    assert restored.primary_emotion == sample_essence.primary_emotion
    assert restored.intensity == sample_essence.intensity
    assert np.allclose(restored.embedding, sample_essence.embedding)


# ============================================================================
# Layer 2: Symbol Selection Tests
# ============================================================================


def test_symbol_selector_initialization(mock_symbols):
    """Test symbol selector initialization."""
    selector = SymbolSelector(mock_symbols)

    assert len(selector.symbols) == 20
    assert selector.symbol_embeddings.shape == (20, 228)


def test_symbol_selector_candidates(mock_symbols, sample_essence, sample_context):
    """Test candidate selection."""
    selector = SymbolSelector(mock_symbols)

    candidates = selector.select_candidates(sample_essence, sample_context, k=5)

    assert len(candidates) == 5
    assert all(isinstance(sym, SymbolArchetype) for sym, _ in candidates)
    assert all(isinstance(score, float) for _, score in candidates)
    assert all(0.0 <= score <= 1.0 for _, score in candidates)

    # Should be sorted by score (descending)
    scores = [score for _, score in candidates]
    assert scores == sorted(scores, reverse=True)


def test_symbol_selector_semantic_similarity(mock_symbols):
    """Test semantic similarity computation."""
    selector = SymbolSelector(mock_symbols)

    # Create query embedding
    query = np.random.randn(228)
    query = query / np.linalg.norm(query)

    similarities = selector._semantic_similarity(query)

    assert similarities.shape == (20,)
    assert all(0.0 <= s <= 1.0 for s in similarities)


def test_symbol_selector_emotion_tag_match(mock_symbols, sample_essence):
    """Test emotion tag matching."""
    selector = SymbolSelector(mock_symbols)

    # Find a trapped symbol
    trapped_symbol = next(s for s in mock_symbols if s.category == "trapped")

    # Should match well
    match_score = selector._emotion_tag_match(sample_essence, trapped_symbol)
    assert match_score > 0.0


def test_symbol_selector_fallback(mock_symbols, sample_context):
    """Test fallback selection when no embedding."""
    selector = SymbolSelector(mock_symbols)

    essence_no_embedding = EmotionalEssence(
        primary_emotion="trapped",
        intensity=0.8,
        context="work",
        temporal="chronic",
        valence="negative",
        embedding=None  # No embedding
    )

    candidates = selector._fallback_selection(essence_no_embedding, k=5)

    assert len(candidates) <= 5
    assert all(isinstance(sym, SymbolArchetype) for sym, _ in candidates)


# ============================================================================
# Layer 3: Disambiguation Tests
# ============================================================================


def test_symbol_disambiguator_basic(mock_symbols, sample_context):
    """Test basic disambiguation."""
    disambiguator = SymbolDisambiguator()

    # Create candidates
    candidates = [(mock_symbols[i], 0.9 - i * 0.1) for i in range(5)]

    selected = disambiguator.disambiguate(candidates, sample_context)

    assert isinstance(selected, SymbolArchetype)
    assert selected in [c[0] for c in candidates]


def test_symbol_disambiguator_temporal_consistency(mock_symbols):
    """Test temporal consistency bonus."""
    disambiguator = SymbolDisambiguator()

    symbol = mock_symbols[0]

    # Context without symbol
    context_no = DreamContext(primary_dreamer="Alice")
    assert not disambiguator._is_recurring_symbol(symbol, context_no)

    # Context with symbol
    context_yes = DreamContext(
        primary_dreamer="Alice",
        existing_symbols=[symbol]
    )
    assert disambiguator._is_recurring_symbol(symbol, context_yes)


def test_symbol_disambiguator_narrative_coherence(mock_symbols):
    """Test narrative coherence scoring."""
    disambiguator = SymbolDisambiguator()

    symbol = mock_symbols[0]

    # No existing symbols
    coherence = disambiguator._narrative_coherence(symbol, [])
    assert coherence == 1.0

    # With existing symbols
    existing = mock_symbols[1:3]
    coherence = disambiguator._narrative_coherence(symbol, existing)
    assert 0.0 <= coherence <= 1.0


def test_symbol_disambiguator_visual_clutter(mock_symbols):
    """Test visual clutter detection."""
    disambiguator = SymbolDisambiguator()

    # Create complex symbol
    complex_symbol = mock_symbols[0]
    complex_symbol.visual_complexity = "complex"

    # No existing complex symbols
    assert not disambiguator._creates_visual_clutter(complex_symbol, [])

    # Two existing complex symbols
    existing_complex = [s for s in mock_symbols[1:3]]
    for s in existing_complex:
        s.visual_complexity = "complex"

    assert disambiguator._creates_visual_clutter(complex_symbol, existing_complex)


# ============================================================================
# Layer 4: Narrative Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_symbol_integrator_template_scene(mock_symbols, sample_context):
    """Test template-based scene generation."""
    integrator = SymbolIntegrator(llm_client=None)  # No LLM

    symbol = mock_symbols[0]
    scene = await integrator.generate_scene(symbol, sample_context, privacy_level=0.5)

    assert isinstance(scene, DreamScene)
    assert len(scene.description) > 0
    assert scene.symbol_id == symbol.symbol_id
    assert len(scene.visual_elements) > 0
    assert len(scene.symbolic_actions) > 0


@pytest.mark.asyncio
async def test_symbol_integrator_privacy_levels(mock_symbols, sample_context):
    """Test different privacy levels."""
    integrator = SymbolIntegrator()

    symbol = mock_symbols[0]

    # Low privacy
    scene_low = await integrator.generate_scene(symbol, sample_context, privacy_level=0.2)
    assert scene_low.privacy_level == 0.2

    # High privacy
    scene_high = await integrator.generate_scene(symbol, sample_context, privacy_level=0.9)
    assert scene_high.privacy_level == 0.9


# ============================================================================
# End-to-End Integration Tests
# ============================================================================


def test_symbolic_encoder_initialization():
    """Test encoder initialization."""
    encoder = SymbolicEncoder()

    assert encoder.extractor is not None
    assert encoder.disambiguator is not None
    assert encoder.integrator is not None
    assert encoder.selector is None  # Not initialized until symbols loaded


def test_symbolic_encoder_load_symbols(mock_symbol_db_file):
    """Test symbol database loading."""
    encoder = SymbolicEncoder()

    encoder.load_symbols(mock_symbol_db_file)

    assert len(encoder.symbols) == 20
    assert encoder.selector is not None


def test_symbolic_encoder_load_symbols_from_list(mock_symbols):
    """Test loading symbols from list."""
    encoder = SymbolicEncoder()

    encoder.load_symbols_from_list(mock_symbols)

    assert len(encoder.symbols) == 20
    assert encoder.selector is not None


@pytest.mark.asyncio
async def test_symbolic_encoder_end_to_end(mock_symbols, sample_context):
    """Test full encoding pipeline."""
    encoder = SymbolicEncoder(enable_llm=False)  # Disable LLM for testing
    encoder.load_symbols_from_list(mock_symbols)

    # Encode private fact
    private_fact = "Bob hates his factory job and feels trapped by his bills"

    scene = await encoder.encode(
        private_fact=private_fact,
        dream_context=sample_context,
        privacy_level=0.7
    )

    assert isinstance(scene, DreamScene)
    assert len(scene.description) > 0
    assert scene.symbol_id in [s.symbol_id for s in mock_symbols]
    assert 0.0 <= scene.privacy_level <= 1.0


@pytest.mark.asyncio
async def test_symbolic_encoder_multiple_facts(mock_symbols):
    """Test encoding multiple facts in sequence."""
    encoder = SymbolicEncoder(enable_llm=False)
    encoder.load_symbols_from_list(mock_symbols)

    facts = [
        "Alice fears her husband will leave her",
        "Charlie feels guilty about cheating on his taxes",
        "Diana feels powerless at work"
    ]

    scenes = []

    for fact in facts:
        context = DreamContext(primary_dreamer="Test")
        scene = await encoder.encode(fact, context, privacy_level=0.5)
        scenes.append(scene)

    assert len(scenes) == 3
    assert all(isinstance(s, DreamScene) for s in scenes)


def test_symbolic_encoder_statistics(mock_symbols):
    """Test encoder statistics."""
    encoder = SymbolicEncoder()
    encoder.load_symbols_from_list(mock_symbols)

    stats = encoder.get_statistics()

    assert stats["total_symbols"] == 20
    assert "categories" in stats
    assert "llm_enabled" in stats
    assert "semantic_calculus_available" in stats


def test_symbolic_encoder_no_symbols():
    """Test error handling when symbols not loaded."""
    encoder = SymbolicEncoder()

    context = DreamContext(primary_dreamer="Test")

    with pytest.raises(RuntimeError, match="Symbol database not loaded"):
        asyncio.run(encoder.encode("test fact", context))


# ============================================================================
# Symbol Database Tests
# ============================================================================


def test_create_mock_symbol_database():
    """Test mock symbol database creation."""
    symbols = create_mock_symbol_database(num_symbols=50)

    assert len(symbols) == 50
    assert all(isinstance(s, SymbolArchetype) for s in symbols)
    assert all(s.embedding.shape == (228,) for s in symbols)


def test_save_load_symbol_database(tmp_path, mock_symbols):
    """Test saving and loading symbol database."""
    db_path = tmp_path / "test_db.json"

    # Save
    save_symbol_database(mock_symbols, str(db_path))
    assert db_path.exists()

    # Load
    encoder = SymbolicEncoder()
    encoder.load_symbols(str(db_path))

    assert len(encoder.symbols) == len(mock_symbols)


def test_symbol_archetype_serialization(mock_symbols):
    """Test symbol archetype to/from dict."""
    symbol = mock_symbols[0]

    # To dict
    data = symbol.to_dict()
    assert isinstance(data, dict)
    assert "symbol_id" in data
    assert "embedding" in data

    # From dict
    restored = SymbolArchetype.from_dict(data)
    assert restored.symbol_id == symbol.symbol_id
    assert restored.category == symbol.category
    assert np.allclose(restored.embedding, symbol.embedding)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_empty_candidates_error():
    """Test error handling for empty candidates."""
    disambiguator = SymbolDisambiguator()
    context = DreamContext(primary_dreamer="Test")

    with pytest.raises(ValueError, match="No candidates"):
        disambiguator.disambiguate([], context)


def test_emotional_extraction_empty_text():
    """Test emotion extraction from empty text."""
    extractor = EmotionalExtractor()

    essence = extractor.extract("")

    assert isinstance(essence, EmotionalEssence)
    # Should still return valid essence (may be undefined)


def test_symbol_selection_no_embedding(mock_symbols, sample_context):
    """Test symbol selection without embedding."""
    selector = SymbolSelector(mock_symbols)

    essence_no_embed = EmotionalEssence(
        primary_emotion="trapped",
        intensity=0.8,
        context="work",
        temporal="chronic",
        valence="negative",
        embedding=None
    )

    candidates = selector.select_candidates(essence_no_embed, sample_context, k=5)

    assert len(candidates) > 0  # Should use fallback


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_encoding_performance(mock_symbols):
    """Test encoding performance (should be <100ms)."""
    import time

    encoder = SymbolicEncoder(enable_llm=False)
    encoder.load_symbols_from_list(mock_symbols)

    context = DreamContext(primary_dreamer="Test")

    start = time.time()

    await encoder.encode(
        "Bob hates his factory job",
        context,
        privacy_level=0.5
    )

    duration_ms = (time.time() - start) * 1000

    # Should be fast (<100ms without LLM)
    assert duration_ms < 100, f"Encoding took {duration_ms:.1f}ms (expected <100ms)"


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
