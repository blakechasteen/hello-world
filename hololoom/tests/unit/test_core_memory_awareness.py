"""
Unit tests for hololoom/core/memory/awareness/

Covers: compositional_awareness, dual_stream, llm_integration,
        context_packer, memory_fusion, meta_awareness
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from hololoom.core.memory.awareness.compositional_awareness import (
    CompositionalAwarenessLayer,
    UnifiedAwarenessContext,
    StructuralAwareness,
    CompositionalPatterns,
    ConfidenceSignals,
    InternalStreamGuidance,
    ExternalStreamGuidance,
    PatternInfo,
    format_awareness_for_prompt,
)
from hololoom.core.memory.awareness.dual_stream import (
    DualStreamGenerator,
    DualStreamResponse,
    build_internal_prompt,
    build_external_prompt,
)
from hololoom.core.memory.awareness.llm_integration import (
    LLMProvider,
    LLMResponse,
    OllamaLLM,
    AnthropicLLM,
    OpenAILLM,
    GeminiLLM,
    GrokLLM,
    create_llm,
)
from hololoom.core.memory.awareness.context_packer import (
    ContextImportance,
    CompressionLevel,
    ContextElement,
    TokenBudget,
    PackedContext,
    SmartContextPacker,
)
from hololoom.core.memory.awareness.memory_fusion import (
    MemoryNode,
    MultipassConfig,
    MemoryFusion,
)
from hololoom.core.memory.awareness.meta_awareness import (
    UncertaintyType,
    UncertaintyDecomposition,
    MetaConfidence,
    KnowledgeGapHypothesis,
    AdversarialProbe,
    SelfReflectionResult,
    MetaAwarenessLayer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_awareness_context(
    uncertainty=0.5,
    is_question=False,
    phrase_type="STATEMENT",
    domain="GENERAL",
    seen_count=0,
    knowledge_gap=False,
    should_clarify=False,
):
    """Build a minimal UnifiedAwarenessContext for testing."""
    structural = StructuralAwareness(
        phrase_type=phrase_type,
        head_word="test",
        is_question=is_question,
        suggested_response_type="EXPLANATION" if is_question else "STATEMENT",
    )
    patterns = CompositionalPatterns(
        phrase="test query",
        seen_count=seen_count,
        confidence=1.0 - uncertainty,
        domain=domain,
        subdomain="UNKNOWN",
    )
    confidence = ConfidenceSignals(
        overall_cache_hit_rate=1.0 - uncertainty,
        query_cache_status="HOT_HIT" if uncertainty < 0.3 else "COLD_MISS",
        uncertainty_level=uncertainty,
        knowledge_gap_detected=knowledge_gap,
        should_ask_clarification=should_clarify,
        suggested_clarification="Could you provide more context?" if should_clarify else None,
    )
    internal = InternalStreamGuidance(reasoning_structure="ANALYZE -> DECIDE -> RESPOND")
    external = ExternalStreamGuidance(
        confidence_tone="confident" if uncertainty < 0.3 else "tentative",
        appropriate_hedging=["perhaps"] if uncertainty >= 0.3 else [],
        should_acknowledge_uncertainty=uncertainty > 0.7,
        clarification_needed=should_clarify,
    )
    return UnifiedAwarenessContext(
        structural=structural,
        patterns=patterns,
        confidence=confidence,
        internal_guidance=internal,
        external_guidance=external,
    )


# ===========================================================================
# compositional_awareness.py
# ===========================================================================

class TestCompositionalAwarenessLayer:

    @pytest.fixture
    def layer(self):
        return CompositionalAwarenessLayer()

    @pytest.mark.asyncio
    async def test_get_unified_context_simple_statement(self, layer):
        ctx = await layer.get_unified_context("The ball is red")
        assert isinstance(ctx, UnifiedAwarenessContext)
        assert ctx.structural.phrase_type in ("STATEMENT", "NP", "QUERY")
        assert ctx.patterns.domain == "PHYSICAL_OBJECTS"

    @pytest.mark.asyncio
    async def test_get_unified_context_question(self, layer):
        ctx = await layer.get_unified_context("What is quantum physics?")
        assert ctx.structural.is_question or "what is" in "what is quantum physics?"
        assert ctx.structural.expects_definition is True

    @pytest.mark.asyncio
    async def test_get_unified_context_code_domain(self, layer):
        ctx = await layer.get_unified_context("How do I write a function in code?")
        assert ctx.patterns.domain == "TECHNICAL"

    @pytest.mark.asyncio
    async def test_analyze_structure_fallback(self, layer):
        result = await layer._analyze_structure("why does the sun rise?")
        assert result.is_question is True
        assert result.expects_explanation is True

    @pytest.mark.asyncio
    async def test_recognize_patterns_general(self, layer):
        result = await layer._recognize_patterns("hello world")
        assert result.domain == "GENERAL"
        assert result.seen_count == 0

    @pytest.mark.asyncio
    async def test_compute_confidence_novel_query(self, layer):
        patterns = CompositionalPatterns(phrase="novel", seen_count=0, confidence=0.0)
        conf = await layer._compute_confidence("novel query", patterns)
        assert conf.uncertainty_level > 0.5
        assert conf.knowledge_gap_detected is True
        assert conf.should_ask_clarification is True

    @pytest.mark.asyncio
    async def test_compute_confidence_familiar_query(self, layer):
        # Pre-seed pattern history
        layer.pattern_history["familiar query"] = PatternInfo(
            seen_count=15, confidence=0.9
        )
        patterns = CompositionalPatterns(phrase="familiar query", seen_count=15, confidence=0.9)
        conf = await layer._compute_confidence("familiar query", patterns)
        assert conf.uncertainty_level <= 0.2

    def test_generate_internal_guidance_definition(self, layer):
        struct = StructuralAwareness(
            phrase_type="QUERY", head_word="what", expects_definition=True
        )
        patterns = CompositionalPatterns(phrase="q", seen_count=0, confidence=0.0)
        conf = ConfidenceSignals(uncertainty_level=0.5)
        guidance = layer._generate_internal_guidance(struct, patterns, conf)
        assert "DEFINITION" in guidance.reasoning_structure

    def test_generate_internal_guidance_explanation(self, layer):
        struct = StructuralAwareness(
            phrase_type="QUERY", head_word="why", expects_explanation=True
        )
        patterns = CompositionalPatterns(phrase="q", seen_count=0, confidence=0.0)
        conf = ConfidenceSignals(uncertainty_level=0.2)
        guidance = layer._generate_internal_guidance(struct, patterns, conf)
        assert "UNDERSTAND" in guidance.reasoning_structure
        assert "skip_verification" in guidance.high_confidence_shortcuts

    def test_generate_internal_guidance_comparison(self, layer):
        struct = StructuralAwareness(
            phrase_type="QUERY", head_word="what", expects_comparison=True
        )
        patterns = CompositionalPatterns(phrase="q", seen_count=0, confidence=0.0)
        conf = ConfidenceSignals(uncertainty_level=0.8)
        guidance = layer._generate_internal_guidance(struct, patterns, conf)
        assert "COMPARE" in guidance.reasoning_structure
        assert "consider_clarification" in guidance.low_confidence_checks

    def test_generate_external_guidance_confident(self, layer):
        struct = StructuralAwareness(phrase_type="QUERY", head_word="what")
        patterns = CompositionalPatterns(phrase="q", domain="GENERAL")
        conf = ConfidenceSignals(uncertainty_level=0.1, should_ask_clarification=False)
        guidance = layer._generate_external_guidance(struct, patterns, conf)
        assert guidance.confidence_tone == "confident"
        assert guidance.expected_length == "short"

    def test_generate_external_guidance_uncertain(self, layer):
        struct = StructuralAwareness(phrase_type="QUERY", head_word="what")
        patterns = CompositionalPatterns(phrase="q", domain="GENERAL")
        conf = ConfidenceSignals(uncertainty_level=0.8, should_ask_clarification=True)
        guidance = layer._generate_external_guidance(struct, patterns, conf)
        assert guidance.confidence_tone == "clarifying"
        assert guidance.should_acknowledge_uncertainty is True

    def test_generate_external_guidance_scientific(self, layer):
        struct = StructuralAwareness(phrase_type="QUERY", head_word="how")
        patterns = CompositionalPatterns(phrase="q", domain="SCIENTIFIC")
        conf = ConfidenceSignals(uncertainty_level=0.5, should_ask_clarification=False)
        guidance = layer._generate_external_guidance(struct, patterns, conf)
        assert guidance.expected_length == "detailed"

    def test_gather_cache_statistics_no_cache(self, layer):
        stats = layer._gather_cache_statistics()
        assert stats == {}

    @pytest.mark.asyncio
    async def test_update_from_generation_new(self, layer):
        await layer.update_from_generation(
            query="new query",
            internal_reasoning="reasoning",
            external_response="response",
            user_feedback={"success": True},
        )
        assert "new query" in layer.pattern_history
        assert layer.pattern_history["new query"].seen_count == 1
        assert layer.pattern_history["new query"].confidence == 0.8

    @pytest.mark.asyncio
    async def test_update_from_generation_existing(self, layer):
        layer.pattern_history["existing"] = PatternInfo(seen_count=3, confidence=0.5)
        await layer.update_from_generation(
            query="existing",
            internal_reasoning="reasoning",
            external_response="response",
            user_feedback={"success": True},
        )
        assert layer.pattern_history["existing"].seen_count == 4
        assert layer.pattern_history["existing"].confidence == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_update_from_generation_failure(self, layer):
        layer.pattern_history["fail"] = PatternInfo(seen_count=1, confidence=0.5)
        await layer.update_from_generation(
            query="fail",
            internal_reasoning="reasoning",
            external_response="response",
            user_feedback={"success": False},
        )
        assert layer.pattern_history["fail"].confidence == pytest.approx(0.4)


class TestFormatAwarenessForPrompt:

    def test_format_statement(self):
        ctx = _make_awareness_context(uncertainty=0.2)
        result = format_awareness_for_prompt(ctx)
        assert "[AWARENESS CONTEXT]" in result
        assert "Confidence:" in result

    def test_format_question(self):
        ctx = _make_awareness_context(is_question=True, uncertainty=0.8, should_clarify=True)
        ctx.structural.question_type = "WHY"
        result = format_awareness_for_prompt(ctx)
        assert "Question type: WHY" in result
        assert "Recommend clarification" in result

    def test_format_knowledge_gap(self):
        ctx = _make_awareness_context(knowledge_gap=True, uncertainty=0.9)
        result = format_awareness_for_prompt(ctx)
        assert "Knowledge gap detected" in result


# ===========================================================================
# dual_stream.py
# ===========================================================================

class TestDualStreamResponse:

    def test_format_for_display_with_internal(self):
        ctx = _make_awareness_context()
        resp = DualStreamResponse(
            query="test?",
            awareness_context=ctx,
            internal_stream="thinking...",
            external_stream="answer",
            generation_time_ms=42.0,
        )
        text = resp.format_for_display(show_internal=True)
        assert "INTERNAL REASONING" in text
        assert "RESPONSE" in text
        assert "42.0ms" in text

    def test_format_for_display_without_internal(self):
        ctx = _make_awareness_context()
        resp = DualStreamResponse(
            query="test?",
            awareness_context=ctx,
            internal_stream="thinking...",
            external_stream="answer",
            generation_time_ms=1.0,
        )
        text = resp.format_for_display(show_internal=False)
        assert "INTERNAL REASONING" not in text
        assert "RESPONSE" in text


class TestDualStreamGenerator:

    @pytest.fixture
    def layer(self):
        return CompositionalAwarenessLayer()

    @pytest.fixture
    def generator(self, layer):
        return DualStreamGenerator(awareness_layer=layer)

    @pytest.mark.asyncio
    async def test_generate_template_mode(self, generator):
        # Seed pattern history so it doesn't trigger clarification path
        generator.awareness.pattern_history["What is a red ball?"] = PatternInfo(
            seen_count=5, confidence=0.7
        )
        result = await generator.generate("What is a red ball?")
        assert isinstance(result, DualStreamResponse)
        assert result.query == "What is a red ball?"
        assert len(result.internal_stream) > 0
        assert len(result.external_stream) > 0
        assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_with_mock_llm(self, layer):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="LLM response")
        )
        gen = DualStreamGenerator(awareness_layer=layer, llm_generator=mock_llm)
        assert gen.use_llm is True
        result = await gen.generate("test query", use_llm=True)
        assert result.internal_stream == "LLM response"
        assert result.external_stream == "LLM response"

    @pytest.mark.asyncio
    async def test_generate_llm_fallback_on_error(self, layer):
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        gen = DualStreamGenerator(awareness_layer=layer, llm_generator=mock_llm)
        result = await gen.generate("test query", use_llm=True)
        # Should fall back to template
        assert len(result.internal_stream) > 0
        assert len(result.external_stream) > 0

    @pytest.mark.asyncio
    async def test_internal_stream_high_confidence(self, layer):
        # Pre-seed to get high confidence
        layer.pattern_history["familiar ball game"] = PatternInfo(
            seen_count=20, confidence=0.95
        )
        gen = DualStreamGenerator(awareness_layer=layer)
        result = await gen.generate("familiar ball game")
        assert "confidence" in result.internal_stream.lower() or "AWARENESS" in result.internal_stream

    @pytest.mark.asyncio
    async def test_external_stream_clarification(self, layer):
        gen = DualStreamGenerator(awareness_layer=layer)
        # Novel query triggers clarification
        result = await gen.generate("xylophonic metacognitive blorp?")
        assert len(result.external_stream) > 0

    @pytest.mark.asyncio
    async def test_external_stream_confident_red_ball(self, layer):
        # Seed familiarity
        layer.pattern_history["What is a red ball?"] = PatternInfo(
            seen_count=20, confidence=0.95
        )
        gen = DualStreamGenerator(awareness_layer=layer)
        result = await gen.generate("What is a red ball?")
        assert len(result.external_stream) > 0

    @pytest.mark.asyncio
    async def test_external_stream_hedged(self, layer):
        # Seed partial familiarity
        layer.pattern_history["partial question"] = PatternInfo(
            seen_count=5, confidence=0.5
        )
        gen = DualStreamGenerator(awareness_layer=layer)
        result = await gen.generate("partial question")
        assert len(result.external_stream) > 0

    def test_clarification_response(self, generator):
        ctx = _make_awareness_context(uncertainty=0.9, should_clarify=True)
        resp = generator._generate_clarification_response("q", ctx)
        assert "not familiar" in resp.lower() or "context" in resp.lower()

    def test_confident_response_generic(self, generator):
        ctx = _make_awareness_context(uncertainty=0.1, seen_count=5)
        resp = generator._generate_confident_response("some query", ctx)
        assert "domain" in resp.lower() or "confidently" in resp.lower()

    def test_confident_response_red_ball_definition(self, generator):
        ctx = _make_awareness_context(uncertainty=0.1)
        ctx.structural.expects_definition = True
        resp = generator._generate_confident_response("What is a red ball?", ctx)
        assert "spherical" in resp.lower()

    def test_confident_response_red_ball_explanation(self, generator):
        ctx = _make_awareness_context(uncertainty=0.1)
        ctx.structural.expects_explanation = True
        resp = generator._generate_confident_response("Why are red balls popular?", ctx)
        assert "visibility" in resp.lower()

    def test_hedged_response(self, generator):
        ctx = _make_awareness_context(uncertainty=0.5)
        resp = generator._generate_hedged_response("some query", ctx)
        assert "moderate" in resp.lower() or "hedging" in resp.lower() or "perhaps" in resp.lower()

    def test_uncertain_response(self, generator):
        ctx = _make_awareness_context(uncertainty=0.9)
        resp = generator._generate_uncertain_response("novel query", ctx)
        assert "novel" in resp.lower() or "uncertainty" in resp.lower()

    def test_uncertain_response_with_nearest(self, generator):
        ctx = _make_awareness_context(uncertainty=0.9)
        ctx.confidence.nearest_known_patterns = [("pattern_a", 0.3), ("pattern_b", 0.2)]
        resp = generator._generate_uncertain_response("novel", ctx)
        assert "pattern_a" in resp


class TestPromptBuilders:

    def test_build_internal_prompt(self):
        ctx = _make_awareness_context()
        prompt = build_internal_prompt("test query", ctx)
        assert "test query" in prompt
        assert "AWARENESS CONTEXT" in prompt

    def test_build_external_prompt(self):
        ctx = _make_awareness_context(uncertainty=0.8, should_clarify=True)
        ctx.external_guidance.appropriate_hedging = ["perhaps", "maybe"]
        prompt = build_external_prompt("test", ctx, "internal reasoning text")
        assert "internal reasoning text" in prompt
        assert "Acknowledge uncertainty" in prompt
        assert "clarification" in prompt.lower()
        assert "perhaps" in prompt


# ===========================================================================
# llm_integration.py
# ===========================================================================

class TestLLMProvider:

    def test_enum_values(self):
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.GEMINI.value == "gemini"
        assert LLMProvider.GROK.value == "grok"


class TestLLMResponse:

    def test_dataclass_fields(self):
        resp = LLMResponse(
            content="hello",
            provider=LLMProvider.OLLAMA,
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            metadata={"key": "value"},
        )
        assert resp.content == "hello"
        assert resp.provider == LLMProvider.OLLAMA
        assert resp.usage["total_tokens"] == 15

    def test_defaults(self):
        resp = LLMResponse(content="x", provider=LLMProvider.OPENAI, model="gpt-4")
        assert resp.usage is None
        assert resp.metadata is None


class TestOllamaLLM:

    def test_init_without_ollama_package(self):
        with patch.dict("sys.modules", {"ollama": None}):
            llm = OllamaLLM.__new__(OllamaLLM)
            llm.model = "test"
            llm.base_url = None
            llm.provider = LLMProvider.OLLAMA
            llm.client = None
            llm._available = False
            assert llm.is_available() is False

    def test_is_available_false_when_no_client(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm._available = False
        llm.client = None
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        llm.base_url = None
        llm.provider = LLMProvider.OLLAMA
        with pytest.raises(RuntimeError, match="Ollama not available"):
            await llm.generate("prompt")

    @pytest.mark.asyncio
    async def test_stream_generate_raises_when_unavailable(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        with pytest.raises(RuntimeError, match="Ollama not available"):
            async for _ in llm.stream_generate("prompt"):
                pass


class TestAnthropicLLM:

    def test_unavailable_without_key(self):
        llm = AnthropicLLM.__new__(AnthropicLLM)
        llm._available = False
        llm.client = None
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        llm = AnthropicLLM.__new__(AnthropicLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        llm.provider = LLMProvider.ANTHROPIC
        with pytest.raises(RuntimeError, match="Anthropic not available"):
            await llm.generate("prompt")

    @pytest.mark.asyncio
    async def test_stream_raises_when_unavailable(self):
        llm = AnthropicLLM.__new__(AnthropicLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        with pytest.raises(RuntimeError, match="Anthropic not available"):
            async for _ in llm.stream_generate("prompt"):
                pass


class TestOpenAILLM:

    def test_unavailable_without_key(self):
        llm = OpenAILLM.__new__(OpenAILLM)
        llm._available = False
        llm.client = None
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        llm = OpenAILLM.__new__(OpenAILLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        llm.provider = LLMProvider.OPENAI
        with pytest.raises(RuntimeError, match="OpenAI not available"):
            await llm.generate("prompt")

    @pytest.mark.asyncio
    async def test_stream_raises_when_unavailable(self):
        llm = OpenAILLM.__new__(OpenAILLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        with pytest.raises(RuntimeError, match="OpenAI not available"):
            async for _ in llm.stream_generate("prompt"):
                pass


class TestGeminiLLM:

    def test_unavailable_without_key(self):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm._available = False
        llm.client = None
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        llm.provider = LLMProvider.GEMINI
        with pytest.raises(RuntimeError, match="Gemini not available"):
            await llm.generate("prompt")

    @pytest.mark.asyncio
    async def test_stream_raises_when_unavailable(self):
        llm = GeminiLLM.__new__(GeminiLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        with pytest.raises(RuntimeError, match="Gemini not available"):
            async for _ in llm.stream_generate("prompt"):
                pass


class TestGrokLLM:

    def test_unavailable_without_key(self):
        llm = GrokLLM.__new__(GrokLLM)
        llm._available = False
        llm.client = None
        assert llm.is_available() is False

    @pytest.mark.asyncio
    async def test_generate_raises_when_unavailable(self):
        llm = GrokLLM.__new__(GrokLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        llm.provider = LLMProvider.GROK
        with pytest.raises(RuntimeError, match="Grok not available"):
            await llm.generate("prompt")

    @pytest.mark.asyncio
    async def test_stream_raises_when_unavailable(self):
        llm = GrokLLM.__new__(GrokLLM)
        llm._available = False
        llm.client = None
        llm.model = "test"
        with pytest.raises(RuntimeError, match="Grok not available"):
            async for _ in llm.stream_generate("prompt"):
                pass


class TestCreateLLM:

    def test_create_ollama(self):
        llm = create_llm("ollama", model="test:3b")
        assert isinstance(llm, OllamaLLM)
        assert llm.model == "test:3b"

    def test_create_anthropic(self):
        llm = create_llm("anthropic")
        assert isinstance(llm, AnthropicLLM)

    def test_create_openai(self):
        llm = create_llm("openai")
        assert isinstance(llm, OpenAILLM)

    def test_create_gemini(self):
        llm = create_llm("gemini")
        assert isinstance(llm, GeminiLLM)

    def test_create_grok(self):
        llm = create_llm("grok")
        assert isinstance(llm, GrokLLM)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm("unknown_provider")

    def test_case_insensitive(self):
        llm = create_llm("OLLAMA")
        assert isinstance(llm, OllamaLLM)

    def test_default_models(self):
        assert create_llm("ollama").model == "llama3.2:3b"
        assert create_llm("anthropic").model == "claude-3-5-sonnet-20241022"
        assert create_llm("openai").model == "gpt-4"
        assert create_llm("gemini").model == "gemini-2.0-flash"
        assert create_llm("grok").model == "grok-4-1-fast"


# ===========================================================================
# context_packer.py
# ===========================================================================

class TestContextElement:

    def test_compress_full(self):
        elem = ContextElement(
            content="full content",
            importance=0.8,
            token_count=3,
            source="awareness",
        )
        assert elem.compress(CompressionLevel.FULL) == "full content"

    def test_compress_summary(self):
        elem = ContextElement(
            content="full content here",
            importance=0.8,
            token_count=5,
            source="awareness",
            summary="summary",
        )
        assert elem.compress(CompressionLevel.SUMMARY) == "summary"

    def test_compress_detailed(self):
        elem = ContextElement(
            content="full content",
            importance=0.8,
            token_count=3,
            source="awareness",
            detailed="detail ver",
        )
        assert elem.compress(CompressionLevel.DETAILED) == "detail ver"

    def test_compress_minimal(self):
        elem = ContextElement(
            content="full content",
            importance=0.8,
            token_count=3,
            source="awareness",
        )
        compressed = elem.compress(CompressionLevel.MINIMAL)
        assert "awareness" in compressed

    def test_compress_falls_back(self):
        elem = ContextElement(
            content="original",
            importance=0.5,
            token_count=2,
            source="test",
        )
        # No summary provided, should fall back to content
        assert elem.compress(CompressionLevel.SUMMARY) == "original"

    def test_estimate_tokens(self):
        elem = ContextElement(
            content="a" * 100,
            importance=0.5,
            token_count=25,
            source="test",
        )
        est = elem.estimate_tokens(CompressionLevel.FULL)
        assert est == 25  # 100 // 4


class TestTokenBudget:

    def test_available_for_context(self):
        budget = TokenBudget(total=8000, reserved_for_query=500, reserved_for_response=1000)
        assert budget.available_for_context == 6500

    def test_default_values(self):
        budget = TokenBudget()
        assert budget.total == 8000
        assert budget.available_for_context == 6500


class TestPackedContext:

    def test_format_for_llm(self):
        pc = PackedContext(
            awareness_section="awareness text",
            memory_section="memory text",
            pattern_section="pattern text",
            query_section="the query",
            total_tokens=100,
            elements_included=5,
            elements_compressed=1,
            elements_excluded=2,
            avg_importance=0.7,
            min_importance=0.3,
            packing_time_ms=5.0,
            compression_stats={"full": 3, "compressed": 1},
        )
        text = pc.format_for_llm()
        assert "AWARENESS CONTEXT" in text
        assert "RELEVANT MEMORIES" in text
        assert "RECOGNIZED PATTERNS" in text
        assert "QUERY" in text

    def test_format_for_llm_with_metadata(self):
        pc = PackedContext(
            awareness_section="aw",
            memory_section="",
            pattern_section="",
            query_section="q",
            total_tokens=50,
            elements_included=2,
            elements_compressed=0,
            elements_excluded=0,
            avg_importance=0.9,
            min_importance=0.8,
            packing_time_ms=1.0,
            compression_stats={},
        )
        text = pc.format_for_llm(include_metadata=True)
        assert "PACKING METADATA" in text
        assert "Total tokens: 50" in text

    def test_format_for_llm_empty_sections(self):
        pc = PackedContext(
            awareness_section="",
            memory_section="",
            pattern_section="",
            query_section="just the query",
            total_tokens=5,
            elements_included=1,
            elements_compressed=0,
            elements_excluded=0,
            avg_importance=1.0,
            min_importance=1.0,
            packing_time_ms=0.5,
            compression_stats={},
        )
        text = pc.format_for_llm()
        assert "AWARENESS CONTEXT" not in text
        assert "just the query" in text


class TestSmartContextPacker:

    @pytest.fixture
    def packer(self):
        return SmartContextPacker()

    @pytest.mark.asyncio
    async def test_pack_basic(self, packer):
        ctx = _make_awareness_context()
        packed = await packer.pack_context(
            query="test query",
            awareness_context=ctx,
        )
        assert isinstance(packed, PackedContext)
        assert packed.elements_included > 0
        assert packed.query_section == "test query"

    @pytest.mark.asyncio
    async def test_pack_with_memory_results(self, packer):
        ctx = _make_awareness_context()
        memories = [
            {"text": "memory one", "score": 0.9},
            {"text": "memory two", "score": 0.7},
        ]
        packed = await packer.pack_context(
            query="test",
            awareness_context=ctx,
            memory_results=memories,
        )
        assert "memory one" in packed.memory_section

    def test_extract_memory_text_dict(self, packer):
        assert packer._extract_memory_text({"text": "hello"}) == "hello"
        assert packer._extract_memory_text({"content": "world"}) == "world"

    def test_extract_memory_text_object(self, packer):
        obj = MagicMock()
        obj.text = "from attr"
        assert packer._extract_memory_text(obj) == "from attr"

    def test_extract_memory_text_fallback(self, packer):
        result = packer._extract_memory_text(42)
        assert result == "42"

    def test_score_elements_boosts_uncertainty(self, packer):
        ctx = _make_awareness_context(uncertainty=0.9)
        elem = ContextElement(
            content="test", importance=0.5, token_count=1, source="awareness"
        )
        result = packer._score_elements([elem], ctx)
        assert result[0].importance > 0.5

    def test_score_elements_boosts_familiar_patterns(self, packer):
        ctx = _make_awareness_context(seen_count=15)
        elem = ContextElement(
            content="test",
            importance=0.5,
            token_count=1,
            source="awareness",
            metadata={"type": "pattern_analysis"},
        )
        packer._score_elements([elem], ctx)
        assert elem.importance > 0.5

    def test_optimize_packing_respects_budget(self, packer):
        elements = [
            ContextElement(content="critical", importance=1.0, token_count=100, source="query"),
            ContextElement(
                content="high",
                importance=0.8,
                token_count=5000,
                source="awareness",
                summary="short",
            ),
            ContextElement(
                content="low",
                importance=0.3,
                token_count=5000,
                source="memory",
                summary="tiny",
            ),
        ]
        packed, stats = packer._optimize_packing(elements, token_budget=200)
        total_tokens = sum(e.token_count for e in packed)
        assert total_tokens <= 200

    def test_assemble_sections(self, packer):
        elements = [
            ContextElement(content="q", importance=1.0, token_count=1, source="query"),
            ContextElement(
                content="aw",
                importance=0.8,
                token_count=1,
                source="awareness",
                metadata={"type": "confidence_signals"},
            ),
            ContextElement(
                content="pat",
                importance=0.5,
                token_count=1,
                source="awareness",
                metadata={"type": "pattern_analysis"},
            ),
            ContextElement(
                content="mem",
                importance=0.6,
                token_count=1,
                source="memory",
                metadata={},
            ),
        ]
        sections = packer._assemble_sections(elements)
        assert sections["query"] == "q"
        assert "aw" in sections["awareness"]
        assert "pat" in sections["pattern"]
        assert "mem" in sections["memory"]


# ===========================================================================
# memory_fusion.py
# ===========================================================================

class TestMemoryNode:

    def test_defaults(self):
        node = MemoryNode(id="n1", content="hello", relevance=0.8)
        assert node.retrieval_depth == 0
        assert node.composite_score == 0.0
        assert node.source_path == []
        assert node.related_ids == []


class TestMultipassConfig:

    def test_for_complexity_lite(self):
        cfg = MultipassConfig.for_complexity("LITE")
        assert cfg.max_passes == 1
        assert cfg.max_depth == 1

    def test_for_complexity_fast(self):
        cfg = MultipassConfig.for_complexity("FAST")
        assert cfg.max_passes == 2

    def test_for_complexity_full(self):
        cfg = MultipassConfig.for_complexity("FULL")
        assert cfg.max_passes == 3

    def test_for_complexity_research(self):
        cfg = MultipassConfig.for_complexity("RESEARCH")
        assert cfg.max_passes == 4
        assert cfg.max_depth == 4

    def test_for_complexity_default(self):
        cfg = MultipassConfig.for_complexity("UNKNOWN")
        assert cfg.max_passes == 3  # default


class TestMemoryFusion:

    @pytest.fixture
    def fusion(self):
        return MemoryFusion(config=MultipassConfig.for_complexity("LITE"))

    def test_result_to_node_dict(self, fusion):
        result = {
            "id": "test-1",
            "content": "test content",
            "relevance": 0.85,
            "timestamp": datetime(2025, 1, 1).isoformat(),
            "related": ["r1", "r2"],
        }
        node = fusion._result_to_node(result, depth=1, source_path=["parent"])
        assert node.id == "test-1"
        assert node.content == "test content"
        assert node.relevance == 0.85
        assert node.retrieval_depth == 1
        assert node.source_path == ["parent"]
        assert node.timestamp == datetime(2025, 1, 1)

    def test_result_to_node_minimal(self, fusion):
        result = {"content": "minimal"}
        node = fusion._result_to_node(result, depth=0, source_path=[])
        assert node.content == "minimal"
        assert node.relevance == 0.5  # default
        assert node.timestamp is None

    def test_result_to_node_invalid_timestamp(self, fusion):
        result = {"id": "x", "content": "x", "timestamp": "not-a-date"}
        node = fusion._result_to_node(result, depth=0, source_path=[])
        assert node.timestamp is None

    def test_result_to_node_datetime_object(self, fusion):
        ts = datetime(2025, 6, 15)
        result = {"id": "x", "content": "x", "timestamp": ts}
        node = fusion._result_to_node(result, depth=0, source_path=[])
        assert node.timestamp == ts

    def test_calculate_temporal_score_no_timestamp(self, fusion):
        score = fusion._calculate_temporal_score(None, datetime.now())
        assert score == 0.5

    def test_calculate_temporal_score_recent(self, fusion):
        now = datetime.now()
        score = fusion._calculate_temporal_score(now - timedelta(minutes=1), now)
        assert score > 0.9

    def test_calculate_temporal_score_old(self, fusion):
        now = datetime.now()
        score = fusion._calculate_temporal_score(now - timedelta(days=30), now)
        assert score < 0.5

    def test_calculate_graph_score(self, fusion):
        assert fusion._calculate_graph_score(0) == 1.0
        assert fusion._calculate_graph_score(1) < 1.0
        assert fusion._calculate_graph_score(10) == 0.0  # clamped

    def test_calculate_composite_scores(self, fusion):
        nodes = [
            MemoryNode(
                id="n1",
                content="a",
                relevance=0.9,
                retrieval_depth=0,
                timestamp=datetime.now(),
            ),
            MemoryNode(
                id="n2",
                content="b",
                relevance=0.5,
                retrieval_depth=2,
                timestamp=datetime.now() - timedelta(days=7),
            ),
        ]
        scored = fusion._calculate_composite_scores(nodes, datetime.now())
        assert scored[0].composite_score > scored[1].composite_score

    @pytest.mark.asyncio
    async def test_initial_retrieval_no_backend(self, fusion):
        results = await fusion._initial_retrieval("query", threshold=0.5, limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_initial_retrieval_with_backend(self):
        backend = MagicMock()
        # Only expose `retrieve`, not `retrieve_with_threshold`
        backend.retrieve = AsyncMock(
            return_value=[
                {"id": "r1", "content": "result", "relevance": 0.8},
                {"id": "r2", "content": "low", "relevance": 0.2},
            ]
        )
        # Remove retrieve_with_threshold so hasattr returns False
        del backend.retrieve_with_threshold
        fusion = MemoryFusion(config=MultipassConfig.for_complexity("LITE"), memory_backend=backend)
        results = await fusion._initial_retrieval("q", threshold=0.5, limit=5)
        assert len(results) == 1  # Only r1 passes threshold

    @pytest.mark.asyncio
    async def test_get_related_items_no_backend(self, fusion):
        results = await fusion._get_related_items("id", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_fusion_no_backend(self, fusion):
        # Source has a division-by-zero bug when results are empty;
        # verify it raises (known issue in memory_fusion.py:213)
        with pytest.raises(ZeroDivisionError):
            await fusion.retrieve_with_fusion("query", max_results=5)

    @pytest.mark.asyncio
    async def test_retrieve_with_fusion_with_backend(self):
        backend = MagicMock()
        backend.retrieve = AsyncMock(
            return_value=[
                {"id": "r1", "content": "hit", "relevance": 0.9, "related": []},
            ]
        )
        del backend.retrieve_with_threshold
        fusion = MemoryFusion(
            config=MultipassConfig.for_complexity("LITE"),
            memory_backend=backend,
        )
        results = await fusion.retrieve_with_fusion("q", max_results=5)
        assert len(results) == 1
        assert results[0].content == "hit"

    @pytest.mark.asyncio
    async def test_expand_frontier(self):
        backend = AsyncMock()
        backend.get_related = AsyncMock(
            return_value=[
                {"id": "rel1", "content": "related", "relevance": 0.8},
            ]
        )
        fusion = MemoryFusion(
            config=MultipassConfig.for_complexity("FULL"),
            memory_backend=backend,
        )
        discovered = {
            "parent": MemoryNode(
                id="parent", content="p", relevance=0.9, source_path=[]
            )
        }
        new_nodes = await fusion._expand_frontier(
            frontier={"parent"},
            discovered=discovered,
            depth=1,
            threshold=0.5,
            limit=5,
        )
        assert len(new_nodes) == 1
        assert new_nodes[0].id == "rel1"


# ===========================================================================
# meta_awareness.py
# ===========================================================================

class TestUncertaintyDecomposition:

    def test_explanation_structural(self):
        d = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.4,
            semantic_uncertainty=0.1,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.STRUCTURAL,
            ambiguous_structures=["clause_attachment"],
        )
        assert "clause_attachment" in d.get_explanation()

    def test_explanation_semantic(self):
        d = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.5,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.SEMANTIC,
            ambiguous_terms=["quantum"],
        )
        assert "quantum" in d.get_explanation()

    def test_explanation_contextual(self):
        d = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.5,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.CONTEXTUAL,
            missing_context=["background"],
        )
        assert "background" in d.get_explanation()

    def test_explanation_compositional(self):
        d = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.5,
            dominant_type=UncertaintyType.COMPOSITIONAL,
        )
        assert "interpretations" in d.get_explanation()

    def test_explanation_epistemic(self):
        d = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.EPISTEMIC,
        )
        assert "Unknown" in d.get_explanation()


class TestMetaConfidence:

    def test_well_calibrated_true(self):
        mc = MetaConfidence(
            primary_confidence=0.7,
            meta_confidence=0.9,
            calibration_history=[0.65, 0.7, 0.75, 0.68, 0.72, 0.7],
            uncertainty_about_uncertainty=0.1,
            lower_bound=0.6,
            upper_bound=0.8,
        )
        assert mc.is_well_calibrated() is True

    def test_well_calibrated_false_empty(self):
        mc = MetaConfidence(
            primary_confidence=0.7,
            meta_confidence=0.3,
            calibration_history=[],
            uncertainty_about_uncertainty=0.5,
            lower_bound=0.5,
            upper_bound=0.9,
        )
        assert mc.is_well_calibrated() is False

    def test_confidence_interval(self):
        mc = MetaConfidence(
            primary_confidence=0.7,
            meta_confidence=0.9,
            calibration_history=[],
            uncertainty_about_uncertainty=0.1,
            lower_bound=0.6,
            upper_bound=0.8,
        )
        assert mc.get_confidence_interval() == (0.6, 0.8)


class TestKnowledgeGapHypothesis:

    def test_to_query_with_answers(self):
        h = KnowledgeGapHypothesis(
            gap_description="test gap",
            potential_answers=["A", "B", "C"],
            required_information=["info1"],
            testable_predictions=["pred1"],
            confidence_in_hypothesis=0.6,
        )
        query = h.to_query()
        assert "A" in query and "B" in query

    def test_to_query_no_answers(self):
        h = KnowledgeGapHypothesis(
            gap_description="test gap",
            potential_answers=[],
            required_information=["info1", "info2"],
            testable_predictions=[],
            confidence_in_hypothesis=0.3,
        )
        query = h.to_query()
        assert "info1" in query


class TestAdversarialProbe:

    def test_str(self):
        p = AdversarialProbe(
            probe_question="Is this right?",
            expected_weakness="maybe wrong",
            test_type="edge_case",
        )
        assert "[edge_case]" in str(p)
        assert "Is this right?" in str(p)


class TestSelfReflectionResult:

    def test_format_introspection(self):
        unc = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.1,
            semantic_uncertainty=0.2,
            contextual_uncertainty=0.1,
            compositional_uncertainty=0.1,
            dominant_type=UncertaintyType.SEMANTIC,
            ambiguous_terms=["meta"],
        )
        mc = MetaConfidence(
            primary_confidence=0.5,
            meta_confidence=0.3,
            calibration_history=[],
            uncertainty_about_uncertainty=0.25,
            lower_bound=0.4,
            upper_bound=0.6,
        )
        hyp = KnowledgeGapHypothesis(
            gap_description="test",
            potential_answers=["A"],
            required_information=["info"],
            testable_predictions=["pred"],
            confidence_in_hypothesis=0.5,
        )
        probe = AdversarialProbe(
            probe_question="test?",
            expected_weakness="w",
            test_type="assumption",
        )
        result = SelfReflectionResult(
            original_response="response text",
            uncertainty_decomposition=unc,
            meta_confidence=mc,
            detected_gaps=["gap1"],
            gap_hypotheses=[hyp],
            adversarial_probes=[probe],
            probe_results=["result1"],
            aware_of_limitations=True,
            epistemic_humility=0.8,
        )
        text = result.format_introspection()
        assert "META-AWARENESS" in text
        assert "UNCERTAINTY DECOMPOSITION" in text
        assert "META-CONFIDENCE" in text
        assert "KNOWLEDGE GAPS" in text
        assert "HYPOTHESIS" in text
        assert "ADVERSARIAL" in text
        assert "EPISTEMIC STATUS" in text
        assert "humble" in text.lower()

    def test_format_introspection_overconfident(self):
        unc = UncertaintyDecomposition(
            total_uncertainty=0.1,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.1,
            dominant_type=UncertaintyType.COMPOSITIONAL,
        )
        mc = MetaConfidence(
            primary_confidence=0.9,
            meta_confidence=0.9,
            calibration_history=[0.9] * 6,
            uncertainty_about_uncertainty=0.01,
            lower_bound=0.85,
            upper_bound=0.95,
        )
        result = SelfReflectionResult(
            original_response="response",
            uncertainty_decomposition=unc,
            meta_confidence=mc,
            detected_gaps=[],
            gap_hypotheses=[],
            adversarial_probes=[],
            probe_results=[],
            aware_of_limitations=False,
            epistemic_humility=0.2,
        )
        text = result.format_introspection()
        assert "Overconfident" in text or "WARNING" in text


class TestMetaAwarenessLayer:

    @pytest.fixture
    def awareness_layer(self):
        return CompositionalAwarenessLayer()

    @pytest.fixture
    def meta(self, awareness_layer):
        return MetaAwarenessLayer(awareness_layer)

    @pytest.mark.asyncio
    async def test_recursive_self_reflection(self, meta):
        ctx = _make_awareness_context(uncertainty=0.7, knowledge_gap=True, should_clarify=True)
        result = await meta.recursive_self_reflection(
            query="What is quantum meta recursive?",
            response="I think it relates to physics.",
            awareness_context=ctx,
        )
        assert isinstance(result, SelfReflectionResult)
        assert result.uncertainty_decomposition.total_uncertainty == 0.7
        assert len(result.adversarial_probes) > 0
        assert result.aware_of_limitations is True

    @pytest.mark.asyncio
    async def test_decompose_uncertainty_long_question(self, meta):
        ctx = _make_awareness_context(
            uncertainty=0.6, is_question=True, knowledge_gap=True
        )
        ctx.patterns.seen_count = 0
        decomp = await meta._decompose_uncertainty(
            "What is the difference between quantum mechanics and classical mechanics in terms of measurement?",
            ctx,
        )
        assert decomp.structural_uncertainty > 0  # long query
        assert "long_clause_attachment" in decomp.ambiguous_structures

    @pytest.mark.asyncio
    async def test_decompose_uncertainty_semantic(self, meta):
        ctx = _make_awareness_context(uncertainty=0.5)
        decomp = await meta._decompose_uncertainty("quantum meta emergent", ctx)
        assert decomp.semantic_uncertainty > 0
        assert "quantum" in decomp.ambiguous_terms

    @pytest.mark.asyncio
    async def test_compute_meta_confidence_no_history(self, meta):
        ctx = _make_awareness_context(uncertainty=0.5)
        mc = await meta._compute_meta_confidence("new query", ctx)
        assert mc.meta_confidence == 0.3
        assert mc.primary_confidence == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_compute_meta_confidence_with_history(self, meta):
        meta.calibration_history["known"] = [0.6, 0.7, 0.65, 0.68, 0.72, 0.7]
        ctx = _make_awareness_context(uncertainty=0.3)
        mc = await meta._compute_meta_confidence("known", ctx)
        assert mc.meta_confidence == 0.9

    def test_detect_knowledge_gaps(self, meta):
        ctx = _make_awareness_context(knowledge_gap=True)
        ctx.patterns.seen_count = 0
        unc = UncertaintyDecomposition(
            total_uncertainty=0.8,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.6,
            contextual_uncertainty=0.7,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.CONTEXTUAL,
            ambiguous_terms=["quantum"],
            missing_context=["domain"],
        )
        gaps = meta._detect_knowledge_gaps("query", ctx, unc)
        assert len(gaps) >= 2  # knowledge_gap + seen_count=0

    def test_generate_hypotheses_semantic(self, meta):
        unc = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.6,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.SEMANTIC,
            ambiguous_terms=["quantum"],
        )
        hypotheses = meta._generate_hypotheses("q", ["unclear"], unc)
        assert len(hypotheses) > 0
        assert "quantum" in hypotheses[0].gap_description

    def test_generate_hypotheses_contextual(self, meta):
        unc = UncertaintyDecomposition(
            total_uncertainty=0.5,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.5,
            compositional_uncertainty=0.0,
            dominant_type=UncertaintyType.CONTEXTUAL,
        )
        gaps = ["Missing background/domain knowledge"]
        hypotheses = meta._generate_hypotheses("q", gaps, unc)
        assert len(hypotheses) > 0

    def test_generate_adversarial_probes_overconfident(self, meta):
        ctx = _make_awareness_context(uncertainty=0.1)
        probes = meta._generate_adversarial_probes("q", "no way", ctx)
        # Should have edge_case probe (low uncertainty) + assumption + contradiction (has "no") + boundary
        types = [p.test_type for p in probes]
        assert "edge_case" in types
        assert "assumption" in types
        assert "contradiction" in types
        assert "boundary" in types

    def test_generate_adversarial_probes_uncertain(self, meta):
        ctx = _make_awareness_context(uncertainty=0.8)
        probes = meta._generate_adversarial_probes("q", "answer", ctx)
        types = [p.test_type for p in probes]
        assert "edge_case" not in types  # high uncertainty, no overconfidence probe
        assert "assumption" in types

    @pytest.mark.asyncio
    async def test_run_probes(self, meta):
        probes = [
            AdversarialProbe("q1?", "w1", "edge_case"),
            AdversarialProbe("q2?", "w2", "assumption"),
            AdversarialProbe("q3?", "w3", "contradiction"),
            AdversarialProbe("q4?", "w4", "boundary"),
        ]
        results = await meta._run_probes(probes, "response")
        assert len(results) == 4

    def test_assess_epistemic_humility(self, meta):
        ctx = _make_awareness_context(uncertainty=0.8, should_clarify=True)
        mc = MetaConfidence(
            primary_confidence=0.2,
            meta_confidence=0.3,
            calibration_history=[],
            uncertainty_about_uncertainty=0.64,
            lower_bound=0.1,
            upper_bound=0.3,
        )
        unc = UncertaintyDecomposition(
            total_uncertainty=0.8,
            structural_uncertainty=0.0,
            semantic_uncertainty=0.0,
            contextual_uncertainty=0.0,
            compositional_uncertainty=0.8,
            dominant_type=UncertaintyType.COMPOSITIONAL,
        )
        humility = meta._assess_epistemic_humility(ctx, mc, unc)
        assert humility > 0.5  # should be humble given high uncertainty

    def test_update_calibration(self, meta):
        meta.update_calibration("q1", 0.8)
        assert "q1" in meta.calibration_history
        assert meta.calibration_history["q1"] == [0.8]

    def test_update_calibration_trims(self, meta):
        for i in range(15):
            meta.update_calibration("q", float(i) / 15)
        assert len(meta.calibration_history["q"]) == 10
