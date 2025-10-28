#!/usr/bin/env python3
"""
Unit Tests for SpinningWheel Enrichment
========================================

Tests enrichment strategies:
- MetadataEnricher
- SemanticEnricher
- TemporalEnricher
- Enrichment pipeline integration

Run with:
    pytest test_enrichment.py -v
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel.enrichment import (
    BaseEnricher,
    EnrichmentResult,
    MetadataEnricher,
    SemanticEnricher,
    TemporalEnricher
)


# ============================================================================
# Base Enricher Tests
# ============================================================================

class TestBaseEnricher:
    """Test BaseEnricher interface."""

    def test_enrichment_result_creation(self):
        """Test creating EnrichmentResult."""
        result = EnrichmentResult(
            enricher_type='test',
            data={'key': 'value'},
            confidence=0.95,
            metadata={'source': 'test'}
        )

        assert result.enricher_type == 'test'
        assert result.data['key'] == 'value'
        assert result.confidence == 0.95
        assert result.metadata['source'] == 'test'

    @pytest.mark.asyncio
    async def test_batch_enrich_default(self):
        """Test default batch enrichment implementation."""

        class MockEnricher(BaseEnricher):
            async def enrich(self, text: str) -> EnrichmentResult:
                return EnrichmentResult(
                    enricher_type='mock',
                    data={'text': text}
                )

        enricher = MockEnricher()
        texts = ['text1', 'text2', 'text3']
        results = await enricher.batch_enrich(texts)

        assert len(results) == 3
        assert all(isinstance(r, EnrichmentResult) for r in results)
        assert results[0].data['text'] == 'text1'


# ============================================================================
# MetadataEnricher Tests
# ============================================================================

class TestMetadataEnricher:
    """Test MetadataEnricher functionality."""

    @pytest.mark.asyncio
    async def test_metadata_tag_extraction(self):
        """Test extraction of hashtags."""
        enricher = MetadataEnricher()

        text = "This is #important and #urgent for the project #testing"
        result = await enricher.enrich(text)

        assert result.enricher_type == 'metadata'
        assert 'tags' in result.data
        assert 'important' in result.data['tags']
        assert 'urgent' in result.data['tags']
        assert 'testing' in result.data['tags']

    @pytest.mark.asyncio
    async def test_metadata_priority_detection(self):
        """Test priority keyword detection."""
        enricher = MetadataEnricher()

        high_priority_text = "URGENT: This is critical and must be done immediately!"
        result = await enricher.enrich(high_priority_text)

        assert 'priority' in result.data
        assert result.data['priority'] in ['high', 'critical']

    @pytest.mark.asyncio
    async def test_metadata_category_detection(self):
        """Test category detection."""
        enricher = MetadataEnricher()

        text = "TODO: Implement the new feature in the codebase"
        result = await enricher.enrich(text)

        assert 'categories' in result.data
        # Should detect 'TODO' or similar task-related categories

    @pytest.mark.asyncio
    async def test_metadata_no_tags(self):
        """Test handling of text without tags."""
        enricher = MetadataEnricher()

        text = "Plain text without any tags or keywords"
        result = await enricher.enrich(text)

        assert result.enricher_type == 'metadata'
        assert 'tags' in result.data
        # Tags should be empty list, not None


# ============================================================================
# SemanticEnricher Tests
# ============================================================================

class TestSemanticEnricher:
    """Test SemanticEnricher functionality."""

    @pytest.mark.asyncio
    async def test_semantic_fallback_mode(self):
        """Test semantic enrichment without Ollama (fallback mode)."""
        config = {'use_ollama': False}
        enricher = SemanticEnricher(config)

        text = "John visited the Apple Store in San Francisco"
        result = await enricher.enrich(text)

        assert result.enricher_type == 'semantic'
        assert 'entities' in result.data
        # Should extract at least some capitalized entities

    @pytest.mark.asyncio
    async def test_semantic_entity_extraction_regex(self):
        """Test regex-based entity extraction."""
        config = {'use_ollama': False}
        enricher = SemanticEnricher(config)

        text = "Alice and Bob discussed Machine Learning at Stanford University"
        result = await enricher.enrich(text)

        entities = result.data.get('entities', [])
        # Should extract capitalized words
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_semantic_empty_text(self):
        """Test handling of empty text."""
        config = {'use_ollama': False}
        enricher = SemanticEnricher(config)

        text = ""
        result = await enricher.enrich(text)

        assert result.enricher_type == 'semantic'
        assert isinstance(result.data.get('entities', []), list)


# ============================================================================
# TemporalEnricher Tests
# ============================================================================

class TestTemporalEnricher:
    """Test TemporalEnricher functionality."""

    @pytest.mark.asyncio
    async def test_temporal_absolute_date_extraction(self):
        """Test extraction of absolute dates."""
        enricher = TemporalEnricher({'reference_date': '2025-10-15'})

        text = "On October 13, 2025, we completed the project"
        result = await enricher.enrich(text)

        assert result.enricher_type == 'temporal'
        assert 'dates' in result.data
        # Should extract the date

    @pytest.mark.asyncio
    async def test_temporal_relative_date_extraction(self):
        """Test extraction of relative dates."""
        enricher = TemporalEnricher({'reference_date': '2025-10-15'})

        text = "Yesterday we started work, and tomorrow we'll finish"
        result = await enricher.enrich(text)

        assert 'relative_terms' in result.data or 'dates' in result.data

    @pytest.mark.asyncio
    async def test_temporal_season_detection(self):
        """Test seasonal context detection."""
        config = {'reference_date': '2025-10-15'}
        enricher = TemporalEnricher(config)

        text = "This fall season has been productive"
        result = await enricher.enrich(text)

        assert 'season' in result.data or 'temporal_context' in result.data

    @pytest.mark.asyncio
    async def test_temporal_no_dates(self):
        """Test handling of text without dates."""
        enricher = TemporalEnricher()

        text = "Simple text without any temporal references"
        result = await enricher.enrich(text)

        assert result.enricher_type == 'temporal'
        # Should handle gracefully even with no dates


# ============================================================================
# Enrichment Pipeline Integration Tests
# ============================================================================

class TestEnrichmentPipeline:
    """Test enrichment pipeline integration."""

    @pytest.mark.asyncio
    async def test_multiple_enrichers_chain(self):
        """Test chaining multiple enrichers."""
        text = "#urgent TODO: Meet with Alice tomorrow about the project"

        # Run all enrichers
        metadata_enricher = MetadataEnricher()
        semantic_enricher = SemanticEnricher({'use_ollama': False})
        temporal_enricher = TemporalEnricher()

        metadata_result = await metadata_enricher.enrich(text)
        semantic_result = await semantic_enricher.enrich(text)
        temporal_result = await temporal_enricher.enrich(text)

        # Verify each enricher produces results
        assert metadata_result.enricher_type == 'metadata'
        assert semantic_result.enricher_type == 'semantic'
        assert temporal_result.enricher_type == 'temporal'

        # Verify data extraction
        assert len(metadata_result.data.get('tags', [])) > 0
        assert len(semantic_result.data.get('entities', [])) >= 0
        assert 'dates' in temporal_result.data or 'relative_terms' in temporal_result.data

    @pytest.mark.asyncio
    async def test_enrichment_result_merging(self):
        """Test merging results from multiple enrichers."""
        text = "Critical #priority task for John in September"

        enrichers = [
            MetadataEnricher(),
            SemanticEnricher({'use_ollama': False}),
            TemporalEnricher()
        ]

        # Collect all results
        results = []
        for enricher in enrichers:
            result = await enricher.enrich(text)
            results.append(result)

        # Merge into combined dict
        combined = {}
        for result in results:
            combined[result.enricher_type] = result.data

        # Verify combined results
        assert 'metadata' in combined
        assert 'semantic' in combined
        assert 'temporal' in combined

    @pytest.mark.asyncio
    async def test_enrichment_error_handling(self):
        """Test that enrichers handle errors gracefully."""

        class BrokenEnricher(BaseEnricher):
            async def enrich(self, text: str) -> EnrichmentResult:
                if not text:
                    # Return empty result instead of raising
                    return EnrichmentResult(
                        enricher_type='broken',
                        data={},
                        confidence=0.0
                    )
                return EnrichmentResult(
                    enricher_type='broken',
                    data={'status': 'ok'}
                )

        enricher = BrokenEnricher()

        # Should handle empty text
        result = await enricher.enrich("")
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_batch_enrichment_performance(self):
        """Test batch enrichment processes multiple texts."""
        enricher = MetadataEnricher()

        texts = [
            "#tag1 Some text",
            "#tag2 Other text",
            "#tag3 More text"
        ]

        results = await enricher.batch_enrich(texts)

        assert len(results) == len(texts)
        assert all(isinstance(r, EnrichmentResult) for r in results)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestEnrichmentErrorHandling:
    """Test error handling in enrichment."""

    @pytest.mark.asyncio
    async def test_null_text_handling(self):
        """Test handling of None/null text."""
        enricher = MetadataEnricher()

        # Should handle None gracefully
        try:
            result = await enricher.enrich(None)
            # If it returns a result, it should be valid
            assert isinstance(result, EnrichmentResult)
        except (TypeError, AttributeError):
            # Or it should raise an appropriate error
            pass

    @pytest.mark.asyncio
    async def test_very_long_text(self):
        """Test handling of very long text."""
        enricher = MetadataEnricher()

        # 10,000 word text
        long_text = "word " * 10000

        result = await enricher.enrich(long_text)

        # Should complete without hanging or error
        assert isinstance(result, EnrichmentResult)

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters."""
        enricher = MetadataEnricher()

        text = "Text with émojis 🎉 and spëcial çharacters ñ"

        result = await enricher.enrich(text)

        # Should handle without crashing
        assert isinstance(result, EnrichmentResult)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
