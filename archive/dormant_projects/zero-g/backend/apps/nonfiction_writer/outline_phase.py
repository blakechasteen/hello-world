"""
Outline Phase for Nonfiction Writing

Uses HoloLoom's agentic reasoning (PLAN_EXECUTE mode) to generate hierarchical
outlines from research corpus. Supports iterative refinement and validation.

Outline Generation Pipeline:
1. Analyze research corpus and identify main themes
2. Generate hierarchical structure (I, II, III → A, B, C → 1, 2, 3)
3. Validate coverage (all research represented)
4. Refine outline based on feedback
5. Export to multiple formats (Markdown, JSON, plain text)

Integrates with:
- AgenticOrchestrator (PLAN_EXECUTE mode) for intelligent decomposition
- ResearchCorpus for source material
- CitationManager for linking sources to sections
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

# Would integrate with actual HoloLoom in production
try:
    from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode
    from HoloLoom.documentation.types import Query
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False


from .research_phase import ResearchCorpus


class OutlineStyle(Enum):
    """Outline numbering styles"""
    ROMAN = "roman"          # I, II, III → A, B, C → 1, 2, 3
    DECIMAL = "decimal"      # 1, 2, 3 → 1.1, 1.2, 1.3 → 1.1.1, 1.1.2
    ALPHANUMERIC = "alpha"   # A, B, C → 1, 2, 3 → a, b, c


@dataclass
class OutlineSection:
    """
    A section in the outline hierarchy

    Attributes:
        id: Unique section identifier
        level: Depth in hierarchy (1 = top level, 2 = subsection, etc.)
        number: Section number (e.g., "II.A.1")
        title: Section title
        description: Section description/summary
        key_points: Bullet points for this section
        source_ids: IDs of sources supporting this section
        subsections: Child sections
        estimated_word_count: Target words for this section
        parent_id: ID of parent section
    """
    id: str
    level: int
    number: str
    title: str
    description: str
    key_points: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    subsections: List['OutlineSection'] = field(default_factory=list)
    estimated_word_count: int = 500
    parent_id: Optional[str] = None

    def add_subsection(self, subsection: 'OutlineSection'):
        """Add a subsection"""
        subsection.parent_id = self.id
        subsection.level = self.level + 1
        self.subsections.append(subsection)

    def get_total_word_count(self) -> int:
        """Get total word count including all subsections"""
        total = self.estimated_word_count
        for sub in self.subsections:
            total += sub.get_total_word_count()
        return total

    def get_depth(self) -> int:
        """Get maximum depth of this section tree"""
        if not self.subsections:
            return self.level
        return max(sub.get_depth() for sub in self.subsections)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'id': self.id,
            'level': self.level,
            'number': self.number,
            'title': self.title,
            'description': self.description,
            'key_points': self.key_points,
            'source_ids': self.source_ids,
            'estimated_word_count': self.estimated_word_count,
            'subsections': [sub.to_dict() for sub in self.subsections],
        }


@dataclass
class Outline:
    """
    Complete document outline

    Attributes:
        topic: Document topic
        thesis: Thesis statement
        sections: Top-level sections
        style: Numbering style
        total_sections: Count of all sections
        created_at: Creation timestamp
        validated: Whether outline has been validated
        validation_notes: Validation feedback
    """
    topic: str
    thesis: str
    sections: List[OutlineSection] = field(default_factory=list)
    style: OutlineStyle = OutlineStyle.ROMAN
    total_sections: int = 0
    created_at: str = ""
    validated: bool = False
    validation_notes: List[str] = field(default_factory=list)

    def add_section(self, section: OutlineSection):
        """Add a top-level section"""
        section.level = 1
        self.sections.append(section)
        self._recalculate_total()

    def get_all_sections(self) -> List[OutlineSection]:
        """Get all sections (flattened)"""
        all_sections = []
        for section in self.sections:
            all_sections.extend(self._flatten_section(section))
        return all_sections

    def _flatten_section(self, section: OutlineSection) -> List[OutlineSection]:
        """Recursively flatten section tree"""
        sections = [section]
        for sub in section.subsections:
            sections.extend(self._flatten_section(sub))
        return sections

    def _recalculate_total(self):
        """Recalculate total section count"""
        self.total_sections = len(self.get_all_sections())

    def get_total_word_count(self) -> int:
        """Get estimated total word count"""
        return sum(section.get_total_word_count() for section in self.sections)

    def get_max_depth(self) -> int:
        """Get maximum depth of outline"""
        if not self.sections:
            return 0
        return max(section.get_depth() for section in self.sections)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'topic': self.topic,
            'thesis': self.thesis,
            'style': self.style.value,
            'total_sections': self.total_sections,
            'estimated_word_count': self.get_total_word_count(),
            'max_depth': self.get_max_depth(),
            'validated': self.validated,
            'validation_notes': self.validation_notes,
            'sections': [section.to_dict() for section in self.sections],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export outline as JSON"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export outline as Markdown"""
        lines = []
        lines.append(f"# {self.topic}\n")
        lines.append(f"**Thesis:** {self.thesis}\n")
        lines.append(f"**Estimated Word Count:** {self.get_total_word_count():,}\n")
        lines.append("---\n")

        for section in self.sections:
            lines.append(self._section_to_markdown(section))

        return "\n".join(lines)

    def _section_to_markdown(self, section: OutlineSection, indent: int = 0) -> str:
        """Convert section to Markdown"""
        lines = []

        # Section heading
        heading_level = section.level + 1  # # is for topic, so start at ##
        heading = "#" * heading_level
        lines.append(f"{heading} {section.number}. {section.title}")

        # Description
        if section.description:
            lines.append(f"\n{section.description}\n")

        # Key points
        if section.key_points:
            lines.append("\nKey points:")
            for point in section.key_points:
                lines.append(f"- {point}")
            lines.append("")

        # Sources
        if section.source_ids:
            lines.append(f"\n*Sources: {', '.join(section.source_ids)}*\n")

        # Word count
        lines.append(f"*Est. {section.estimated_word_count} words*\n")

        # Subsections
        for sub in section.subsections:
            lines.append(self._section_to_markdown(sub, indent + 1))

        return "\n".join(lines)


class OutlineGenerator:
    """
    Generate hierarchical outlines from research corpus

    Uses agentic reasoning (PLAN_EXECUTE mode) to decompose topic into
    structured sections with proper hierarchy and coverage.

    Usage:
        generator = OutlineGenerator(corpus)

        # Generate initial outline
        outline = await generator.generate(
            thesis="Climate adaptation requires multi-scale approaches",
            target_sections=5,
            max_depth=3
        )

        # Validate coverage
        validation = await generator.validate(outline)

        # Refine based on feedback
        refined = await generator.refine(outline, "Add more on urban planning")

        # Export
        print(outline.to_markdown())
    """

    def __init__(self, corpus: ResearchCorpus):
        """
        Initialize outline generator

        Args:
            corpus: Research corpus with sources
        """
        self.corpus = corpus
        self._section_counter = 0

        # Initialize agentic orchestrator if available
        self.orchestrator = None
        if HOLOLOOM_AVAILABLE:
            # Would initialize with proper config in production
            pass

    async def generate(self,
                      thesis: str,
                      target_sections: int = 5,
                      max_depth: int = 3,
                      target_word_count: int = 3000,
                      style: OutlineStyle = OutlineStyle.ROMAN) -> Outline:
        """
        Generate outline from research corpus

        Args:
            thesis: Thesis statement
            target_sections: Target number of main sections
            max_depth: Maximum outline depth
            target_word_count: Target total word count
            style: Numbering style

        Returns:
            Generated outline
        """
        # Use PLAN_EXECUTE mode to decompose topic
        if HOLOLOOM_AVAILABLE and self.orchestrator:
            outline = await self._generate_with_agentic(
                thesis, target_sections, max_depth, target_word_count, style
            )
        else:
            # Fallback: simple rule-based generation
            outline = await self._generate_simple(
                thesis, target_sections, max_depth, target_word_count, style
            )

        return outline

    async def validate(self, outline: Outline) -> Dict[str, any]:
        """
        Validate outline coverage and structure

        Args:
            outline: Outline to validate

        Returns:
            Validation report with issues and suggestions
        """
        issues = []
        suggestions = []

        # Check depth
        if outline.get_max_depth() < 2:
            issues.append("Outline is too shallow - consider adding subsections")

        if outline.get_max_depth() > 4:
            issues.append("Outline is too deep - consider consolidating sections")

        # Check balance
        section_counts = [len(s.subsections) for s in outline.sections]
        if section_counts and max(section_counts) - min(section_counts) > 3:
            issues.append("Sections are unbalanced - some have many more subsections")

        # Check source coverage
        all_sections = outline.get_all_sections()
        sections_with_sources = [s for s in all_sections if s.source_ids]

        if len(sections_with_sources) / len(all_sections) < 0.5:
            issues.append("Many sections lack source citations")

        # Check word count distribution
        words_per_section = [s.estimated_word_count for s in outline.sections]
        if words_per_section and max(words_per_section) / min(words_per_section) > 3:
            suggestions.append("Consider redistributing word counts for better balance")

        # Check themes coverage
        themes_covered = set()
        for theme in self.corpus.themes:
            for section in all_sections:
                if theme.lower() in section.title.lower() or theme.lower() in section.description.lower():
                    themes_covered.add(theme)

        missing_themes = set(self.corpus.themes) - themes_covered
        if missing_themes:
            suggestions.append(f"Consider covering themes: {', '.join(missing_themes)}")

        outline.validation_notes = issues + suggestions
        outline.validated = len(issues) == 0

        return {
            'valid': outline.validated,
            'issues': issues,
            'suggestions': suggestions,
            'coverage_score': len(themes_covered) / max(1, len(self.corpus.themes)),
            'balance_score': 1.0 - (max(section_counts) - min(section_counts)) / max(1, max(section_counts)) if section_counts else 1.0,
        }

    async def refine(self, outline: Outline, feedback: str) -> Outline:
        """
        Refine outline based on feedback

        Args:
            outline: Current outline
            feedback: Refinement feedback

        Returns:
            Refined outline
        """
        # Would use agentic reasoning to interpret feedback in full integration
        # For MVP, simple modifications

        # Parse common feedback patterns
        feedback_lower = feedback.lower()

        if 'more detail' in feedback_lower or 'add subsection' in feedback_lower:
            # Add subsections to sections that lack them
            for section in outline.sections:
                if not section.subsections and section.estimated_word_count > 500:
                    # Split into subsections
                    sub_count = min(3, section.estimated_word_count // 300)
                    sub_words = section.estimated_word_count // sub_count

                    for i in range(sub_count):
                        subsection = OutlineSection(
                            id=self._generate_section_id(),
                            level=section.level + 1,
                            number=f"{section.number}.{chr(65+i)}",
                            title=f"{section.title} - Part {i+1}",
                            description=f"Detailed exploration of aspect {i+1}",
                            estimated_word_count=sub_words,
                        )
                        section.add_subsection(subsection)

        elif 'consolidate' in feedback_lower or 'simpler' in feedback_lower:
            # Remove some subsections
            for section in outline.sections:
                if len(section.subsections) > 3:
                    # Keep only top 3 subsections
                    section.subsections = section.subsections[:3]

        elif 'balance' in feedback_lower:
            # Redistribute word counts
            total_words = outline.get_total_word_count()
            words_per_section = total_words // len(outline.sections)

            for section in outline.sections:
                section.estimated_word_count = words_per_section

        # Recalculate
        outline._recalculate_total()

        return outline

    # ========================================================================
    # Private helper methods
    # ========================================================================

    def _generate_section_id(self) -> str:
        """Generate unique section ID"""
        self._section_counter += 1
        return f"section_{self._section_counter:04d}"

    async def _generate_with_agentic(self,
                                    thesis: str,
                                    target_sections: int,
                                    max_depth: int,
                                    target_word_count: int,
                                    style: OutlineStyle) -> Outline:
        """Generate outline using agentic PLAN_EXECUTE mode"""
        # Would use AgenticOrchestrator in full integration
        # This is a placeholder showing the integration pattern

        # Create query for agentic reasoning
        query_text = f"""
        Create a detailed outline for a nonfiction piece on: {self.corpus.topic}

        Thesis: {thesis}

        Requirements:
        - {target_sections} main sections
        - Maximum depth: {max_depth} levels
        - Total word count: {target_word_count:,} words
        - Use {style.value} numbering

        Research themes available: {', '.join(self.corpus.themes)}

        Provide a hierarchical structure with section titles, descriptions, and key points.
        """

        # Would call: result = await self.orchestrator.reason(Query(text=query_text), mode=ReasoningMode.PLAN_EXECUTE)

        # For MVP, fall back to simple generation
        return await self._generate_simple(thesis, target_sections, max_depth, target_word_count, style)

    async def _generate_simple(self,
                              thesis: str,
                              target_sections: int,
                              max_depth: int,
                              target_word_count: int,
                              style: OutlineStyle) -> Outline:
        """Simple rule-based outline generation"""
        outline = Outline(
            topic=self.corpus.topic,
            thesis=thesis,
            style=style,
        )

        # Distribute word count across sections
        words_per_section = target_word_count // target_sections

        # Generate main sections from themes
        themes = self.corpus.themes[:target_sections]
        if len(themes) < target_sections:
            # Pad with generic sections
            themes.extend([
                "Background and Context",
                "Current State of Research",
                "Future Directions",
                "Conclusion",
            ][:target_sections - len(themes)])

        for i, theme in enumerate(themes[:target_sections]):
            section_num = self._format_number(i + 1, 1, style)

            section = OutlineSection(
                id=self._generate_section_id(),
                level=1,
                number=section_num,
                title=theme.title() if theme.islower() else theme,
                description=f"Exploration of {theme} based on research findings",
                estimated_word_count=words_per_section,
            )

            # Add subsections if max_depth > 1
            if max_depth > 1:
                subsection_count = min(3, len(self.corpus.sources) // target_sections)
                sub_words = words_per_section // max(1, subsection_count)

                for j in range(subsection_count):
                    subsection_num = self._format_number(j + 1, 2, style, parent=section_num)

                    subsection = OutlineSection(
                        id=self._generate_section_id(),
                        level=2,
                        number=subsection_num,
                        title=f"Aspect {j+1} of {theme.title()}",
                        description=f"Detailed analysis of specific aspect",
                        estimated_word_count=sub_words,
                    )

                    # Link to sources
                    if j < len(self.corpus.sources):
                        subsection.source_ids.append(self.corpus.sources[j].id)

                    section.add_subsection(subsection)

            outline.add_section(section)

        return outline

    def _format_number(self, index: int, level: int, style: OutlineStyle, parent: str = "") -> str:
        """Format section number based on style"""
        if style == OutlineStyle.ROMAN:
            if level == 1:
                # I, II, III, ...
                return self._to_roman(index)
            elif level == 2:
                # A, B, C, ...
                return chr(64 + index)
            else:
                # 1, 2, 3, ...
                return str(index)

        elif style == OutlineStyle.DECIMAL:
            if parent:
                return f"{parent}.{index}"
            else:
                return str(index)

        elif style == OutlineStyle.ALPHANUMERIC:
            if level == 1:
                return chr(64 + index)  # A, B, C
            elif level == 2:
                return str(index)       # 1, 2, 3
            else:
                return chr(96 + index)  # a, b, c

        return str(index)

    def _to_roman(self, num: int) -> str:
        """Convert integer to Roman numeral"""
        val_map = [
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]

        result = ""
        for val, numeral in val_map:
            count = num // val
            result += numeral * count
            num -= val * count

        return result


# Factory function
def create_outline_generator(corpus: ResearchCorpus) -> OutlineGenerator:
    """Create a new outline generator"""
    return OutlineGenerator(corpus)
