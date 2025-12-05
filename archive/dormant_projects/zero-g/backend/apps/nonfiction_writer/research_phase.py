"""
Research Phase for Nonfiction Writing

Wrapper around HoloLoom's RAG system (RESEARCH mode) for gathering and organizing
research sources. Supports multiple document types via SpinningWheel adapters.

Research Pipeline:
1. Ingest documents (PDFs, web pages, books, etc.)
2. Extract citations and metadata
3. Build knowledge graph of entities and relationships
4. Organize into ResearchCorpus for downstream use

Integrates with:
- SimpleRAG (RESEARCH mode) for multi-query exploration
- PDFSpinner, URLSpinner, etc. for document processing
- CitationManager for bibliographic tracking
- YarnGraph for relationship mapping
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Would integrate with actual HoloLoom in production
# For MVP, we'll create simple interfaces
try:
    from HoloLoom.rag import SimpleRAG, MultimodalRAG, RAGResult
    from HoloLoom.spinningWheel import PDFSpinner, URLSpinner
    from HoloLoom.memory.graph import KG, KGEdge
    from HoloLoom.documentation.types import MemoryShard
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    # Mock types for standalone operation
    class RAGResult:
        def __init__(self):
            self.response = ""
            self.sources = []
            self.confidence = 0.0
            self.metadata = {}

    class MemoryShard:
        def __init__(self, content="", metadata=None):
            self.content = content
            self.metadata = metadata or {}


from .citation_manager import CitationManager, Citation, SourceType


@dataclass
class ResearchSource:
    """
    A research source with content and metadata

    Attributes:
        id: Unique source identifier
        content: Full text content
        summary: AI-generated summary
        key_findings: Extracted key points
        citation: Bibliographic citation
        file_path: Original file path (if from file)
        url: Original URL (if from web)
        ingested_at: When source was added
        relevance_score: Relevance to research topic (0.0-1.0)
    """
    id: str
    content: str
    summary: str
    key_findings: List[str]
    citation: Optional[Citation] = None
    file_path: Optional[str] = None
    url: Optional[str] = None
    ingested_at: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.5


@dataclass
class ResearchCorpus:
    """
    Collection of research sources with organization

    Attributes:
        topic: Research topic
        sources: All research sources
        knowledge_graph: Entity and relationship graph
        themes: Identified themes/topics
        contradictions: Conflicting findings
        gaps: Identified knowledge gaps
        created_at: Corpus creation time
    """
    topic: str
    sources: List[ResearchSource] = field(default_factory=list)
    knowledge_graph: Optional[Dict] = None  # Would be KG in full integration
    themes: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, str]] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def get_sources_by_credibility(self, min_score: float = 0.7) -> List[ResearchSource]:
        """Get high-credibility sources"""
        return [
            s for s in self.sources
            if s.citation and s.citation.credibility_score >= min_score
        ]

    def get_peer_reviewed_sources(self) -> List[ResearchSource]:
        """Get only peer-reviewed sources"""
        return [
            s for s in self.sources
            if s.citation and s.citation.is_peer_reviewed
        ]

    def get_primary_sources(self) -> List[ResearchSource]:
        """Get primary research sources"""
        return [
            s for s in self.sources
            if s.citation and s.citation.is_primary_source
        ]

    def get_total_word_count(self) -> int:
        """Get total words across all sources"""
        return sum(len(s.content.split()) for s in self.sources)

    def get_source_types_breakdown(self) -> Dict[str, int]:
        """Get count of each source type"""
        breakdown = {}
        for source in self.sources:
            if source.citation:
                source_type = source.citation.source_type.value
                breakdown[source_type] = breakdown.get(source_type, 0) + 1
        return breakdown


class ResearchPhase:
    """
    Research phase for nonfiction writing

    Orchestrates research gathering using HoloLoom's RAG system.

    Usage:
        research = ResearchPhase(topic="Climate Change Adaptation")

        # Add sources
        await research.ingest_pdf("paper.pdf")
        await research.ingest_url("https://example.com/article")

        # Perform research queries
        corpus = await research.build_corpus([
            "What are the main adaptation strategies?",
            "What evidence supports these strategies?",
            "What are the limitations?"
        ])

        # Access organized research
        print(f"Found {len(corpus.sources)} sources")
        print(f"Peer-reviewed: {len(corpus.get_peer_reviewed_sources())}")
    """

    def __init__(self, topic: str, citation_manager: Optional[CitationManager] = None):
        """
        Initialize research phase

        Args:
            topic: Research topic/question
            citation_manager: Citation manager (creates new if not provided)
        """
        self.topic = topic
        self.citation_manager = citation_manager or CitationManager()
        self.sources: List[ResearchSource] = []
        self._source_counter = 0

        # Initialize RAG if available
        self.rag = None
        if HOLOLOOM_AVAILABLE:
            # Would initialize with proper config in production
            pass

    async def ingest_pdf(self, file_path: str) -> ResearchSource:
        """
        Ingest a PDF document

        Args:
            file_path: Path to PDF file

        Returns:
            ResearchSource with extracted content and citation
        """
        self._source_counter += 1
        source_id = f"source_{self._source_counter:04d}"

        # Read PDF content (would use PDFSpinner in full integration)
        content = await self._read_pdf(file_path)

        # Extract metadata (would use PDFSpinner's advanced extraction)
        metadata = await self._extract_pdf_metadata(file_path)

        # Create citation
        citation = self.citation_manager.extract_from_pdf_metadata(content, metadata)
        if citation:
            citation_id = self.citation_manager.add_citation(citation)
            citation.id = citation_id

        # Generate summary (would use RAG in full integration)
        summary = await self._generate_summary(content)

        # Extract key findings
        key_findings = await self._extract_key_findings(content)

        # Score relevance to topic
        relevance = await self._score_relevance(content, self.topic)

        source = ResearchSource(
            id=source_id,
            content=content,
            summary=summary,
            key_findings=key_findings,
            citation=citation,
            file_path=file_path,
            relevance_score=relevance,
        )

        self.sources.append(source)
        return source

    async def ingest_url(self, url: str) -> ResearchSource:
        """
        Ingest a web page

        Args:
            url: URL to ingest

        Returns:
            ResearchSource with extracted content and citation
        """
        self._source_counter += 1
        source_id = f"source_{self._source_counter:04d}"

        # Fetch web content (would use URLSpinner in full integration)
        content, metadata = await self._fetch_url(url)

        # Create citation
        citation = self.citation_manager.extract_from_url(url, content, metadata)
        if citation:
            citation_id = self.citation_manager.add_citation(citation)
            citation.id = citation_id

        # Generate summary
        summary = await self._generate_summary(content)

        # Extract key findings
        key_findings = await self._extract_key_findings(content)

        # Score relevance
        relevance = await self._score_relevance(content, self.topic)

        source = ResearchSource(
            id=source_id,
            content=content,
            summary=summary,
            key_findings=key_findings,
            citation=citation,
            url=url,
            relevance_score=relevance,
        )

        self.sources.append(source)
        return source

    async def research_query(self, query: str) -> RAGResult:
        """
        Perform a research query across all ingested sources

        Args:
            query: Research question

        Returns:
            RAG result with answer, sources, and confidence
        """
        if HOLOLOOM_AVAILABLE and self.rag:
            # Use actual RAG RESEARCH mode
            result = await self.rag.query(query, mode="research", max_steps=5)
            return result
        else:
            # Simple fallback: keyword search
            relevant_sources = []
            query_lower = query.lower()

            for source in self.sources:
                if any(word in source.content.lower() for word in query_lower.split()):
                    relevant_sources.append(source.summary)

            response = "\n\n".join(relevant_sources[:3]) if relevant_sources else "No relevant sources found."

            result = RAGResult()
            result.response = response
            result.sources = relevant_sources
            result.confidence = 0.5
            return result

    async def build_corpus(self, research_questions: Optional[List[str]] = None) -> ResearchCorpus:
        """
        Build a complete research corpus

        Args:
            research_questions: Optional research questions to explore

        Returns:
            ResearchCorpus with all sources and analysis
        """
        # Perform research queries if provided
        if research_questions:
            for question in research_questions:
                await self.research_query(question)

        # Identify themes
        themes = await self._identify_themes()

        # Find contradictions
        contradictions = await self._find_contradictions()

        # Identify gaps
        gaps = await self._identify_gaps()

        corpus = ResearchCorpus(
            topic=self.topic,
            sources=self.sources.copy(),
            themes=themes,
            contradictions=contradictions,
            gaps=gaps,
        )

        return corpus

    # ========================================================================
    # Private helper methods
    # ========================================================================

    async def _read_pdf(self, file_path: str) -> str:
        """Read PDF content (simplified)"""
        # In full integration, would use PDFSpinner
        # For MVP, just return placeholder
        return f"[Content of PDF: {file_path}]\n\n" + (
            "This is a placeholder for actual PDF content. "
            "In production, PDFSpinner would extract full text, "
            "tables, citations, and sections."
        )

    async def _extract_pdf_metadata(self, file_path: str) -> Dict:
        """Extract PDF metadata"""
        # Would use PDFSpinner's advanced extraction
        return {
            'title': Path(file_path).stem.replace('_', ' ').title(),
            'authors': ['Unknown Author'],
            'year': datetime.now().year,
        }

    async def _fetch_url(self, url: str) -> tuple[str, Dict]:
        """Fetch URL content"""
        # Would use URLSpinner in full integration
        content = f"[Content from URL: {url}]\n\nPlaceholder web content."
        metadata = {
            'title': 'Web Article',
            'authors': ['Unknown'],
            'year': datetime.now().year,
        }
        return content, metadata

    async def _generate_summary(self, content: str) -> str:
        """Generate summary of content"""
        # Would use LLM via RAG in full integration
        # Simple extraction for MVP
        words = content.split()
        return ' '.join(words[:100]) + "..." if len(words) > 100 else content

    async def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from content"""
        # Would use LLM extraction in full integration
        # Simple heuristic for MVP
        findings = []

        # Look for numbered findings, bullet points, or conclusion sections
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(line.startswith(marker) for marker in ['1.', '2.', '3.', '-', '*', '•']):
                findings.append(line.lstrip('123456789.-*• '))
                if len(findings) >= 5:
                    break

        return findings if findings else ["[Key findings would be extracted by LLM]"]

    async def _score_relevance(self, content: str, topic: str) -> float:
        """Score content relevance to topic"""
        # Would use semantic similarity in full integration
        # Simple keyword matching for MVP
        content_lower = content.lower()
        topic_lower = topic.lower()
        topic_words = topic_lower.split()

        matches = sum(1 for word in topic_words if word in content_lower)
        score = min(1.0, matches / max(1, len(topic_words)))

        return score

    async def _identify_themes(self) -> List[str]:
        """Identify common themes across sources"""
        # Would use topic modeling in full integration
        # Simple extraction for MVP
        if not self.sources:
            return []

        # Count frequent words across summaries
        from collections import Counter
        words = []
        for source in self.sources:
            words.extend(source.summary.lower().split())

        # Filter common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 4]

        # Get top themes
        counter = Counter(filtered_words)
        themes = [word for word, count in counter.most_common(5)]

        return themes

    async def _find_contradictions(self) -> List[Dict[str, str]]:
        """Find contradictory findings across sources"""
        # Would use semantic contradiction detection in full integration
        # Placeholder for MVP
        if len(self.sources) < 2:
            return []

        # Simple heuristic: look for opposing keywords
        contradictions = []
        opposing_pairs = [
            ('positive', 'negative'),
            ('increase', 'decrease'),
            ('support', 'contradict'),
            ('confirm', 'refute'),
        ]

        for i, source1 in enumerate(self.sources):
            for source2 in self.sources[i+1:]:
                for word1, word2 in opposing_pairs:
                    if word1 in source1.summary.lower() and word2 in source2.summary.lower():
                        contradictions.append({
                            'source1': source1.id,
                            'source2': source2.id,
                            'topic': f"{word1}/{word2} disagreement",
                        })

        return contradictions[:3]  # Return up to 3 contradictions

    async def _identify_gaps(self) -> List[str]:
        """Identify knowledge gaps in research"""
        # Would use systematic gap analysis in full integration
        # Placeholder for MVP
        gaps = [
            "Further research needed on long-term effects",
            "Limited studies on specific populations",
            "Methodological limitations in existing research",
        ]
        return gaps[:2]


# Factory function
def create_research_phase(topic: str, citation_manager: Optional[CitationManager] = None) -> ResearchPhase:
    """Create a new research phase"""
    return ResearchPhase(topic, citation_manager)
