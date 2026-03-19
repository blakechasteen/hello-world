"""
Technical Mode Writer
=====================

Generates technical documentation from memory context.

Philosophy: Accuracy and completeness over style.
"""

import logging
from typing import Any

from ..core.protocol import MemoryShard, StyleGuide, WritingContext, WritingMode

logger = logging.getLogger(__name__)


class TechnicalWriter:
    """
    Technical mode writer - transforms memory into documentation.

    Approach:
    1. Structured sections (Purpose, Implementation, Usage, Notes)
    2. Code examples when available
    3. Step-by-step instructions
    4. Precise terminology
    5. Complete details
    """

    def __init__(self):
        self.logger = logger

    async def generate(self, context: WritingContext) -> str:
        """
        Generate technical documentation from context.

        Args:
            context: Writing context with query and memories

        Returns:
            Technical documentation draft
        """
        if not context.memories:
            return self._generate_empty_response(context.query)

        # Extract technical elements
        analysis = self._analyze_technical_content(context.memories)

        # Generate structured documentation
        doc = self._generate_technical_structure(
            query=context.query,
            memories=context.memories,
            analysis=analysis,
            style=context.style
        )

        return doc

    def _analyze_technical_content(self, memories: list[MemoryShard]) -> dict[str, Any]:
        """
        Analyze memories for technical content.

        Args:
            memories: Memory shards

        Returns:
            Analysis dict with code samples, parameters, etc.
        """
        code_samples = []
        parameters = []
        procedures = []

        for memory in memories:
            text = memory.text
            metadata = memory.metadata

            # Extract code samples (lines starting with spaces/tabs or in metadata)
            if 'code' in metadata:
                code_samples.append(metadata['code'])
            elif any(text.startswith(' ' * 4) for line in text.split('\n')):
                # Indented code block
                code_samples.append(text)

            # Extract parameters (key-value patterns)
            if 'parameters' in metadata:
                parameters.extend(metadata['parameters'])

            # Detect procedural steps (numbered or bulleted lists)
            if any(line.strip().startswith(('1.', '2.', '-', '*')) for line in text.split('\n')):
                procedures.append(text)

        return {
            'code_samples': code_samples[:3],  # Top 3 code samples
            'parameters': parameters[:10],      # Top 10 parameters
            'procedures': procedures,
            'memory_count': len(memories)
        }

    def _generate_technical_structure(
        self,
        query: str,
        memories: list[MemoryShard],
        analysis: dict[str, Any],
        style: StyleGuide
    ) -> str:
        """
        Generate structured technical documentation.

        Args:
            query: User query
            memories: Ordered memories
            analysis: Technical content analysis
            style: Style guide

        Returns:
            Technical documentation
        """
        sections = []

        # TITLE
        sections.append(f"# {query}\n")

        # PURPOSE/OVERVIEW
        purpose = self._generate_purpose(query, memories)
        sections.append(f"## Overview\n\n{purpose}\n")

        # IMPLEMENTATION/DETAILS
        if analysis['code_samples'] or analysis['procedures']:
            implementation = self._generate_implementation(memories, analysis)
            sections.append(f"## Implementation\n\n{implementation}\n")

        # USAGE/EXAMPLES
        if analysis['code_samples']:
            usage = self._generate_usage(analysis['code_samples'])
            sections.append(f"## Usage\n\n{usage}\n")

        # PARAMETERS (if applicable)
        if analysis['parameters']:
            params = self._generate_parameters(analysis['parameters'])
            sections.append(f"## Parameters\n\n{params}\n")

        # NOTES/ADDITIONAL INFO
        notes = self._generate_notes(memories, analysis)
        if notes:
            sections.append(f"## Notes\n\n{notes}\n")

        return '\n'.join(sections)

    def _generate_purpose(self, query: str, memories: list[MemoryShard]) -> str:
        """Generate purpose/overview section."""
        # Use highest relevance memory for purpose
        if not memories:
            return "No information available."

        top_memory = max(memories, key=lambda m: m.metadata.get('relevance', 0.5))
        purpose = top_memory.text.strip()

        # Make it a complete sentence if it isn't
        if not purpose.endswith('.'):
            purpose += '.'

        return purpose

    def _generate_implementation(
        self,
        memories: list[MemoryShard],
        analysis: dict[str, Any]
    ) -> str:
        """Generate implementation details section."""
        parts = []

        # Add procedural steps if available
        if analysis['procedures']:
            for procedure in analysis['procedures']:
                parts.append(procedure.strip())

        # Add detailed explanations from memories
        for memory in memories[:3]:  # Top 3 memories
            text = memory.text.strip()
            # Skip if already included in procedures
            if text not in analysis['procedures']:
                parts.append(text)

        return '\n\n'.join(parts)

    def _generate_usage(self, code_samples: list[str]) -> str:
        """Generate usage/examples section."""
        parts = []

        for i, code in enumerate(code_samples, 1):
            parts.append(f"**Example {i}:**\n\n```python\n{code.strip()}\n```")

        return '\n\n'.join(parts)

    def _generate_parameters(self, parameters: list[Any]) -> str:
        """Generate parameters section."""
        if not parameters:
            return ""

        parts = []
        for param in parameters:
            if isinstance(param, dict):
                name = param.get('name', 'parameter')
                type_info = param.get('type', 'any')
                desc = param.get('description', '')
                parts.append(f"- **{name}** (`{type_info}`): {desc}")
            else:
                parts.append(f"- {param}")

        return '\n'.join(parts)

    def _generate_notes(
        self,
        memories: list[MemoryShard],
        analysis: dict[str, Any]
    ) -> str:
        """Generate notes/additional info section."""
        # Collect lower-relevance memories as notes
        notes = []

        sorted_memories = sorted(
            memories,
            key=lambda m: m.metadata.get('relevance', 0.5)
        )

        # Take bottom 2 memories as notes (if relevance < 0.7)
        for memory in sorted_memories[:2]:
            if memory.metadata.get('relevance', 0.5) < 0.7:
                notes.append(memory.text.strip())

        if notes:
            return '\n\n'.join(f"- {note}" for note in notes)

        return ""

    def _generate_empty_response(self, query: str) -> str:
        """Generate response when no memories available."""
        return (
            f"# {query}\n\n"
            f"## Overview\n\n"
            f"No documentation available for this topic. "
            f"Additional information or context is needed to generate documentation.\n\n"
            f"## Next Steps\n\n"
            f"- Provide implementation details\n"
            f"- Add code examples\n"
            f"- Include usage instructions\n"
        )

    @property
    def mode(self) -> WritingMode:
        """Writing mode this writer handles."""
        return WritingMode.TECHNICAL

    @property
    def default_style(self) -> StyleGuide:
        """Default style guide for technical mode."""
        return StyleGuide.TECHNICAL
