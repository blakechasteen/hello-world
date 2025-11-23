"""
Draft Phase for Nonfiction Writing

Generates initial drafts from outlines and research corpus, automatically
integrating citations and maintaining source provenance.

Draft Generation Pipeline:
1. Expand outline sections into paragraphs
2. Insert relevant content from research sources
3. Add inline citations and footnotes
4. Track paragraph-to-source mappings
5. Generate bibliography

Integrates with:
- Outline and ResearchCorpus from previous phases
- CitationManager for automatic citation insertion
- WeavingOrchestrator for context-aware generation
- Writer class (if HoloLoom integration available)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

# Would integrate with actual HoloLoom in production
try:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.documentation.types import Query
    from HoloLoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False


from .outline_phase import Outline, OutlineSection
from .research_phase import ResearchCorpus, ResearchSource
from .citation_manager import CitationManager, CitationStyle


@dataclass
class Paragraph:
    """
    A paragraph in the draft

    Attributes:
        id: Unique paragraph identifier
        section_id: ID of outline section
        content: Paragraph text
        citations: Inline citations in this paragraph
        source_ids: IDs of sources cited
        word_count: Word count
        confidence: Generation confidence (0.0-1.0)
    """
    id: str
    section_id: str
    content: str
    citations: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    word_count: int = 0
    confidence: float = 0.5

    def __post_init__(self):
        """Calculate word count after initialization"""
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass
class Draft:
    """
    Complete draft document

    Attributes:
        topic: Document topic
        thesis: Thesis statement
        paragraphs: All paragraphs in document order
        outline: Source outline
        bibliography: Formatted bibliography
        word_count: Total word count
        citation_style: Citation style used
        created_at: Creation timestamp
        metadata: Additional metadata
    """
    topic: str
    thesis: str
    paragraphs: List[Paragraph] = field(default_factory=list)
    outline: Optional[Outline] = None
    bibliography: str = ""
    word_count: int = 0
    citation_style: CitationStyle = CitationStyle.APA
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get complete draft text"""
        lines = []

        # Title and thesis
        lines.append(f"# {self.topic}\n")
        lines.append(f"{self.thesis}\n")
        lines.append("---\n")

        # Group paragraphs by section
        current_section = None
        for para in self.paragraphs:
            if para.section_id != current_section:
                # Find section in outline
                if self.outline:
                    section = self._find_section(para.section_id)
                    if section:
                        heading_level = "#" * (section.level + 1)
                        lines.append(f"\n{heading_level} {section.number}. {section.title}\n")

                current_section = para.section_id

            lines.append(para.content)
            lines.append("")  # Blank line between paragraphs

        # Bibliography
        if self.bibliography:
            lines.append("\n---\n")
            lines.append(self.bibliography)

        return "\n".join(lines)

    def get_section_text(self, section_id: str) -> str:
        """Get text for a specific section"""
        section_paras = [p for p in self.paragraphs if p.section_id == section_id]
        return "\n\n".join(p.content for p in section_paras)

    def get_citation_count(self) -> int:
        """Get total number of citations"""
        all_citations = set()
        for para in self.paragraphs:
            all_citations.update(para.citations)
        return len(all_citations)

    def get_source_coverage(self) -> float:
        """Get percentage of sources cited"""
        if not self.outline or not hasattr(self, '_total_sources'):
            return 0.0

        cited_sources = set()
        for para in self.paragraphs:
            cited_sources.update(para.source_ids)

        return len(cited_sources) / max(1, self._total_sources)

    def _find_section(self, section_id: str) -> Optional[OutlineSection]:
        """Find section by ID in outline"""
        if not self.outline:
            return None

        def search(sections):
            for section in sections:
                if section.id == section_id:
                    return section
                if section.subsections:
                    result = search(section.subsections)
                    if result:
                        return result
            return None

        return search(self.outline.sections)


class DraftGenerator:
    """
    Generate initial drafts from outlines and research

    Converts structured outlines into flowing prose with proper citations,
    using research corpus for content and HoloLoom for generation.

    Usage:
        generator = DraftGenerator(outline, corpus, citation_manager)

        # Generate draft
        draft = await generator.generate(
            citation_style=CitationStyle.APA,
            target_density='moderate'  # citation frequency
        )

        # Generate specific section
        section_text = await generator.generate_section(
            section_id="section_0001",
            context=previous_sections
        )

        # Export
        print(draft.get_full_text())
    """

    def __init__(self,
                 outline: Outline,
                 corpus: ResearchCorpus,
                 citation_manager: CitationManager):
        """
        Initialize draft generator

        Args:
            outline: Document outline
            corpus: Research corpus
            citation_manager: Citation manager
        """
        self.outline = outline
        self.corpus = corpus
        self.citation_manager = citation_manager
        self._para_counter = 0

        # Initialize orchestrator if available
        self.orchestrator = None
        if HOLOLOOM_AVAILABLE:
            # Would initialize with proper config in production
            pass

    async def generate(self,
                      citation_style: CitationStyle = CitationStyle.APA,
                      target_density: str = 'moderate') -> Draft:
        """
        Generate complete draft from outline

        Args:
            citation_style: Citation format
            target_density: Citation density ('sparse', 'moderate', 'dense')

        Returns:
            Complete draft with citations
        """
        draft = Draft(
            topic=self.corpus.topic,
            thesis=self.outline.thesis,
            outline=self.outline,
            citation_style=citation_style,
        )

        # Store total sources for coverage calculation
        draft._total_sources = len(self.corpus.sources)

        # Generate paragraphs for each section
        paragraphs = []

        for section in self.outline.sections:
            section_paras = await self._generate_section_recursive(
                section,
                citation_style,
                target_density
            )
            paragraphs.extend(section_paras)

        draft.paragraphs = paragraphs
        draft.word_count = sum(p.word_count for p in paragraphs)

        # Generate bibliography
        cited_citation_ids = set()
        for para in paragraphs:
            cited_citation_ids.update(para.citations)

        draft.bibliography = self.citation_manager.generate_bibliography(
            citation_style,
            list(cited_citation_ids)
        )

        return draft

    async def generate_section(self,
                               section: OutlineSection,
                               context: str = "",
                               citation_style: CitationStyle = CitationStyle.APA) -> List[Paragraph]:
        """
        Generate paragraphs for a single section

        Args:
            section: Outline section
            context: Previous sections for context
            citation_style: Citation format

        Returns:
            List of paragraphs
        """
        # Find relevant sources for this section
        relevant_sources = self._find_relevant_sources(section)

        # Determine paragraph count based on word count
        para_count = max(1, section.estimated_word_count // 150)

        paragraphs = []

        for i in range(para_count):
            # Generate paragraph content
            if HOLOLOOM_AVAILABLE and self.orchestrator:
                content = await self._generate_paragraph_with_orchestrator(
                    section, relevant_sources, context, i
                )
            else:
                # Simple fallback
                content = self._generate_paragraph_simple(
                    section, relevant_sources, i
                )

            # Insert citations
            content, citations, source_ids = self._insert_citations(
                content, relevant_sources, citation_style
            )

            self._para_counter += 1
            paragraph = Paragraph(
                id=f"para_{self._para_counter:04d}",
                section_id=section.id,
                content=content,
                citations=citations,
                source_ids=source_ids,
            )

            paragraphs.append(paragraph)

            # Link citations to paragraph
            for citation_id in citations:
                self.citation_manager.link_citation_to_paragraph(
                    citation_id, paragraph.id
                )

        return paragraphs

    async def _generate_section_recursive(self,
                                         section: OutlineSection,
                                         citation_style: CitationStyle,
                                         target_density: str,
                                         context: str = "") -> List[Paragraph]:
        """Recursively generate paragraphs for section and subsections"""
        paragraphs = []

        # Generate paragraphs for this section
        section_paras = await self.generate_section(section, context, citation_style)
        paragraphs.extend(section_paras)

        # Update context with generated content
        new_context = context + "\n\n" + "\n\n".join(p.content for p in section_paras)

        # Generate subsections
        for subsection in section.subsections:
            sub_paras = await self._generate_section_recursive(
                subsection, citation_style, target_density, new_context
            )
            paragraphs.extend(sub_paras)

        return paragraphs

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _find_relevant_sources(self, section: OutlineSection) -> List[ResearchSource]:
        """Find sources relevant to section"""
        # If section specifies sources, use those
        if section.source_ids:
            return [s for s in self.corpus.sources if s.id in section.source_ids]

        # Otherwise, search by keywords
        keywords = section.title.lower().split() + section.description.lower().split()
        keywords = [k for k in keywords if len(k) > 4]  # Filter short words

        relevant = []
        for source in self.corpus.sources:
            # Score by keyword matches
            matches = sum(1 for k in keywords if k in source.summary.lower())
            if matches > 0:
                relevant.append((source, matches))

        # Sort by relevance and return top 5
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in relevant[:5]]

    async def _generate_paragraph_with_orchestrator(self,
                                                   section: OutlineSection,
                                                   sources: List[ResearchSource],
                                                   context: str,
                                                   para_index: int) -> str:
        """Generate paragraph using WeavingOrchestrator"""
        # Create query for generation
        query_text = f"""
        Write a paragraph for the section: {section.title}

        Section description: {section.description}

        Key points to cover:
        {chr(10).join('- ' + p for p in section.key_points)}

        Available research:
        {chr(10).join(s.summary[:200] + '...' for s in sources[:3])}

        Previous context:
        {context[-500:] if context else '(This is the first paragraph)'}

        Write paragraph {para_index + 1} of approximately 150 words.
        """

        # Would call: result = await self.orchestrator.weave(Query(text=query_text))
        # return result.response

        # For MVP, fall back to simple
        return self._generate_paragraph_simple(section, sources, para_index)

    def _generate_paragraph_simple(self,
                                  section: OutlineSection,
                                  sources: List[ResearchSource],
                                  para_index: int) -> str:
        """Simple paragraph generation fallback"""
        # Start with section description
        if para_index == 0:
            text = section.description

            # Add first key point if available
            if section.key_points:
                text += f" {section.key_points[0]}"
        else:
            # Use subsequent key points
            if para_index < len(section.key_points):
                text = section.key_points[para_index]
            else:
                text = f"Additional considerations regarding {section.title.lower()} include various factors."

        # Add content from sources
        if sources:
            source = sources[para_index % len(sources)]
            # Extract relevant sentence
            sentences = source.summary.split('.')
            if sentences:
                text += f" {sentences[0]}."

        # Ensure minimum length
        while len(text.split()) < 100:
            text += " This topic requires further elaboration and detailed analysis based on the available research."

        return text

    def _insert_citations(self,
                         content: str,
                         sources: List[ResearchSource],
                         style: CitationStyle) -> tuple[str, List[str], List[str]]:
        """Insert inline citations into paragraph"""
        citations = []
        source_ids = []

        # Simple heuristic: add citation after each major claim
        sentences = content.split('.')
        cited_content = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            cited_content.append(sentence)

            # Add citation after some sentences (not every one)
            if i > 0 and i % 2 == 0 and sources:
                source = sources[(i // 2) % len(sources)]
                if source.citation:
                    inline_cite = self.citation_manager.get_inline_citation(
                        source.citation.id,
                        style
                    )
                    cited_content.append(inline_cite)
                    citations.append(source.citation.id)
                    source_ids.append(source.id)

            cited_content.append('.')

        return ' '.join(cited_content), citations, source_ids


# Factory function
def create_draft_generator(outline: Outline,
                          corpus: ResearchCorpus,
                          citation_manager: CitationManager) -> DraftGenerator:
    """Create a new draft generator"""
    return DraftGenerator(outline, corpus, citation_manager)
