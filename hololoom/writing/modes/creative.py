"""
Creative Mode Writer
====================

Generates creative content (fiction, poetry, narratives) from memory context.

Philosophy: Imagination grounded in context.
"""

import logging
from typing import Any

from ..core.protocol import MemoryShard, StyleGuide, WritingContext, WritingMode

logger = logging.getLogger(__name__)


class CreativeWriter:
    """
    Creative mode writer - transforms memory into creative content.

    Approach:
    1. Story arc (exposition, rising action, climax, resolution)
    2. Character development from entities
    3. Vivid imagery and metaphor
    4. Emotional resonance
    5. Creative liberties while honoring context
    """

    def __init__(self):
        self.logger = logger

    async def generate(self, context: WritingContext) -> str:
        """
        Generate creative content from context.

        Args:
            context: Writing context with query and memories

        Returns:
            Creative content draft
        """
        if not context.memories:
            return self._generate_empty_response(context.query)

        # Detect creative type
        creative_type = self._detect_creative_type(context.query)

        # Extract creative elements
        elements = self._extract_creative_elements(context.memories)

        # Generate creative content
        content = self._generate_creative_content(
            query=context.query,
            memories=context.memories,
            elements=elements,
            creative_type=creative_type,
            style=context.style
        )

        return content

    def _detect_creative_type(self, query: str) -> str:
        """
        Detect type of creative content from query.

        Args:
            query: User query

        Returns:
            Creative type: story, poem, dialogue, description
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['story', 'tale', 'narrative', 'fiction']):
            return 'story'
        elif any(word in query_lower for word in ['poem', 'poetry', 'verse', 'haiku']):
            return 'poem'
        elif any(word in query_lower for word in ['dialogue', 'conversation', 'exchange']):
            return 'dialogue'
        elif any(word in query_lower for word in ['describe', 'description', 'scene']):
            return 'description'
        else:
            return 'story'  # Default to story

    def _extract_creative_elements(
        self,
        memories: list[MemoryShard]
    ) -> dict[str, Any]:
        """
        Extract creative elements from memories.

        Args:
            memories: Memory shards

        Returns:
            Dict with characters, settings, themes, conflicts
        """
        characters = set()
        settings = set()
        themes = set()
        conflicts = []

        for memory in memories:
            text = memory.text
            metadata = memory.metadata

            # Extract characters (entities that could be characters)
            if 'entities' in metadata:
                characters.update(metadata['entities'])

            # Extract settings (place names, locations)
            if 'location' in metadata:
                settings.add(metadata['location'])

            # Extract themes
            if 'topic' in metadata:
                themes.add(metadata['topic'])

            # Detect conflicts (contrasts, challenges)
            if any(word in text.lower() for word in ['but', 'however', 'challenge', 'struggle', 'conflict']):
                conflicts.append(text)

        return {
            'characters': list(characters)[:5],  # Top 5 characters
            'settings': list(settings)[:3],      # Top 3 settings
            'themes': list(themes)[:3],          # Top 3 themes
            'conflicts': conflicts[:2],          # Top 2 conflicts
            'memory_count': len(memories)
        }

    def _generate_creative_content(
        self,
        query: str,
        memories: list[MemoryShard],
        elements: dict[str, Any],
        creative_type: str,
        style: StyleGuide
    ) -> str:
        """
        Generate creative content based on type.

        Args:
            query: User query
            memories: Ordered memories
            elements: Creative elements
            creative_type: Type of creative content
            style: Style guide

        Returns:
            Creative content
        """
        if creative_type == 'story':
            return self._generate_story(query, memories, elements)
        elif creative_type == 'poem':
            return self._generate_poem(query, memories, elements)
        elif creative_type == 'dialogue':
            return self._generate_dialogue(query, memories, elements)
        elif creative_type == 'description':
            return self._generate_description(query, memories, elements)
        else:
            return self._generate_story(query, memories, elements)

    def _generate_story(
        self,
        query: str,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate short story."""
        parts = []

        # TITLE
        # Extract key words from query for title
        query_words = [w for w in query.split() if len(w) > 3 and w.lower() not in ['story', 'write', 'about']]
        if query_words:
            title = ' '.join(query_words[:3]).title()
        else:
            title = "The Story"
        parts.append(f"# {title}\n")

        # EXPOSITION (introduce characters, setting)
        exposition = self._generate_exposition(memories, elements)
        parts.append(exposition)

        # RISING ACTION (develop conflict)
        if elements['conflicts']:
            rising_action = self._generate_rising_action(memories, elements)
            parts.append(rising_action)

        # CLIMAX/RESOLUTION
        resolution = self._generate_resolution(memories, elements)
        parts.append(resolution)

        return '\n\n'.join(parts)

    def _generate_exposition(
        self,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate story exposition."""
        parts = []

        # Introduce characters
        if elements['characters']:
            char_intro = self._personify_entities(elements['characters'])
            parts.append(char_intro)
        else:
            # Use first memory as introduction
            if memories:
                parts.append(memories[0].text)

        # Set the scene
        if elements['settings']:
            setting = elements['settings'][0]
            parts.append(f"In the realm of {setting}, something remarkable was happening.")
        elif memories:
            # Extract setting from content
            parts.append("The story begins where knowledge meets imagination.")

        return ' '.join(parts)

    def _generate_rising_action(
        self,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate rising action with conflict."""
        parts = []

        # Introduce conflict
        if elements['conflicts']:
            conflict = elements['conflicts'][0]
            # Dramatize the conflict
            parts.append(f"But a challenge emerged: {conflict.lower()}")

        # Build tension from memories
        for memory in memories[1:3]:  # Middle memories
            parts.append(memory.text)

        return ' '.join(parts)

    def _generate_resolution(
        self,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate story resolution."""
        # Use last/highest-relevance memory for resolution
        if memories:
            final_memory = max(memories, key=lambda m: m.metadata.get('relevance', 0.5))
            resolution = final_memory.text

            # Add satisfying conclusion
            if elements['themes']:
                theme = elements['themes'][0].replace('_', ' ')
                resolution += f" And so, the true nature of {theme} was revealed."

            return resolution

        return "The end."

    def _generate_poem(
        self,
        query: str,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate poem."""
        parts = []

        # Title
        title = query.replace('Write a poem about', '').replace('poem', '').strip().title()
        parts.append(f"# {title or 'Untitled'}\n")

        # Generate stanzas from memories (3-4 lines each)
        for i, memory in enumerate(memories[:4]):  # Up to 4 stanzas
            stanza = self._create_poetic_stanza(memory.text, i)
            parts.append(stanza)

        return '\n\n'.join(parts)

    def _create_poetic_stanza(self, text: str, stanza_num: int) -> str:
        """Transform prose into poetic stanza."""
        # Break into lines (simple approach)
        words = text.split()

        # Create 3-4 line stanza
        lines = []
        words_per_line = max(3, len(words) // 3)

        for i in range(0, min(len(words), words_per_line * 3), words_per_line):
            line = ' '.join(words[i:i+words_per_line])
            # Capitalize first word
            if line:
                line = line[0].upper() + line[1:]
            lines.append(line)

        return '\n'.join(lines[:4])  # Max 4 lines

    def _generate_dialogue(
        self,
        query: str,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate dialogue between characters."""
        parts = []

        # Get characters (or create generic ones)
        if elements['characters'] and len(elements['characters']) >= 2:
            char1, char2 = elements['characters'][:2]
        else:
            char1, char2 = "Alice", "Bob"

        parts.append(f"# Dialogue: {char1} and {char2}\n")

        # Alternate dialogue between characters using memories
        for i, memory in enumerate(memories[:6]):  # Up to 6 exchanges
            speaker = char1 if i % 2 == 0 else char2
            text = memory.text.strip()

            # Format as dialogue
            parts.append(f"**{speaker}**: {text}")

        return '\n\n'.join(parts)

    def _generate_description(
        self,
        query: str,
        memories: list[MemoryShard],
        elements: dict[str, Any]
    ) -> str:
        """Generate vivid description."""
        parts = []

        # Extract subject from query
        subject = query.replace('Describe', '').replace('describe', '').strip()
        parts.append(f"# {subject}\n")

        # Create vivid description from memories
        for memory in memories[:3]:
            # Add sensory details
            text = memory.text
            enhanced = self._add_sensory_details(text, elements)
            parts.append(enhanced)

        return '\n\n'.join(parts)

    def _add_sensory_details(self, text: str, elements: dict[str, Any]) -> str:
        """Add sensory language to text."""
        # Simple enhancement: add descriptive phrases
        sensory_words = {
            'is': 'appears as',
            'has': 'possesses',
            'uses': 'employs',
            'works': 'operates'
        }

        enhanced = text
        for plain, vivid in sensory_words.items():
            if f' {plain} ' in enhanced:
                enhanced = enhanced.replace(f' {plain} ', f' {vivid} ', 1)
                break

        return enhanced

    def _personify_entities(self, entities: list[str]) -> str:
        """Create character introductions from entities."""
        if not entities:
            return ""

        if len(entities) == 1:
            return f"Meet {entities[0]}, our protagonist."
        else:
            entity_list = ', '.join(entities[:-1]) + f" and {entities[-1]}"
            return f"In this tale, we encounter {entity_list}."

    def _generate_empty_response(self, query: str) -> str:
        """Generate response when no memories available."""
        return (
            f"# {query}\n\n"
            f"*A blank page awaits, ready for imagination to fill its void.*\n\n"
            f"Without context or inspiration, the story remains unwritten. "
            f"Provide memories or details to breathe life into this creative work."
        )

    @property
    def mode(self) -> WritingMode:
        """Writing mode this writer handles."""
        return WritingMode.CREATIVE

    @property
    def default_style(self) -> StyleGuide:
        """Default style guide for creative mode."""
        return StyleGuide.CREATIVE
