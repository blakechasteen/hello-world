# -*- coding: utf-8 -*-
"""
Org-Mode Spinner
================
Zero-copy integration between Emacs org-mode and HoloLoom.

Parses org-mode files and maps their structure directly to MemoryShards:
- Headings → Entities
- Hierarchy → Graph relationships
- TODO states → Task motifs
- Tags → Entity metadata
- Timestamps → Temporal metadata for ChronoTrigger
- Links → Explicit graph edges

Design Philosophy:
- Preserve org-mode's semantic structure
- Enable "living" knowledge base (watch files for changes)
- Map org outliner to YarnGraph topology
- Zero-copy: minimal transformation, maximal fidelity

Example org-mode structure:
```
* Project HoloLoom                                    :ai:loom:
  :PROPERTIES:
  :ID: project-hololoom
  :END:
** TODO Implement OrgModeSpinner                      :dev:priority:
   DEADLINE: <2025-11-20>

   Need to parse org syntax and extract:
   - Headings as entities
   - Links as relationships
   - Timestamps for temporal context

*** DONE Research org-mode syntax
    CLOSED: [2025-11-17 Sun 14:30]
```

This becomes:
- Entity: "Project HoloLoom" (tags: ai, loom)
- Entity: "Implement OrgModeSpinner" (TODO, deadline, tags: dev, priority)
- Entity: "Research org-mode syntax" (DONE)
- Relationships: hierarchical parent-child links
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseSpinner, SpinnerConfig

try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    # Fallback definition
    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrgHeading:
    """
    Parsed org-mode heading with all metadata.
    """
    level: int                              # 1 for *, 2 for **, etc.
    title: str                              # Heading text
    todo_state: Optional[str] = None        # TODO, DONE, IN-PROGRESS, etc.
    tags: List[str] = field(default_factory=list)
    priority: Optional[str] = None          # [#A], [#B], [#C]
    scheduled: Optional[str] = None         # SCHEDULED timestamp
    deadline: Optional[str] = None          # DEADLINE timestamp
    closed: Optional[str] = None            # CLOSED timestamp
    properties: Dict[str, str] = field(default_factory=dict)
    body: str = ""                          # Content under heading
    line_number: int = 0                    # Source line number
    parent_id: Optional[str] = None         # Parent heading ID


class OrgModeSpinner(BaseSpinner):
    """
    Spinner that parses org-mode files into MemoryShards.

    Zero-copy philosophy:
    - Org headings → Entities
    - Org hierarchy → Graph edges
    - Org timestamps → Temporal metadata
    - Org links → Explicit relationships

    Usage:
    ```python
    spinner = OrgModeSpinner(SpinnerConfig())

    # From file
    with open('notes.org', 'r') as f:
        org_content = f.read()
    shards = await spinner.spin({'content': org_content, 'source': 'notes.org'})

    # From string
    shards = await spinner.spin({
        'content': '* My Note\\n** Subnote',
        'source': 'inline'
    })
    ```
    """

    # Regex patterns for org-mode syntax
    HEADING_PATTERN = re.compile(
        r'^(\*+)\s+'                        # Stars + space
        r'(?:(TODO|DONE|IN-PROGRESS|WAITING|CANCELLED|NEXT|SOMEDAY)\s+)?'  # Optional TODO state
        r'(?:\[#([A-C])\]\s+)?'             # Optional priority
        r'(.+?)'                            # Title
        r'(?:\s+(:[a-zA-Z0-9_@:]+:))?$',    # Optional tags
        re.MULTILINE
    )

    TIMESTAMP_PATTERN = re.compile(
        r'<(\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?)>'
    )

    SCHEDULED_PATTERN = re.compile(
        r'SCHEDULED:\s*<([^>]+)>'
    )

    DEADLINE_PATTERN = re.compile(
        r'DEADLINE:\s*<([^>]+)>'
    )

    CLOSED_PATTERN = re.compile(
        r'CLOSED:\s*\[([^\]]+)\]'
    )

    LINK_PATTERN = re.compile(
        r'\[\[([^\]]+)\](?:\[([^\]]+)\])?\]'
    )

    PROPERTIES_PATTERN = re.compile(
        r':PROPERTIES:\s*\n(.*?):END:',
        re.DOTALL | re.MULTILINE
    )

    PROPERTY_LINE_PATTERN = re.compile(
        r':([A-Za-z0-9_-]+):\s*(.+)'
    )

    def __init__(self, config: SpinnerConfig):
        super().__init__(config)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Parse org-mode content into MemoryShards.

        Args:
            raw_data: Dict with:
                - content: org-mode text
                - source: filename or identifier
                - episode: optional episode/session name

        Returns:
            List of MemoryShards, one per heading
        """
        content = raw_data.get('content', '')
        source = raw_data.get('source', 'unknown')
        episode = raw_data.get('episode', f'org:{source}')

        # Parse all headings
        headings = self._parse_headings(content)

        # Convert to MemoryShards
        shards = []
        for heading in headings:
            shard = self._heading_to_shard(heading, source, episode)
            shards.append(shard)

        # Optionally enrich shards
        if self.config.enable_enrichment:
            shards = await self._enrich_shards(shards)

        return shards

    def _parse_headings(self, content: str) -> List[OrgHeading]:
        """
        Parse all org-mode headings from content.

        Returns hierarchical structure with parent references.
        """
        lines = content.split('\n')
        headings = []
        current_heading: Optional[OrgHeading] = None
        in_properties = False
        properties_lines = []
        body_lines = []
        parent_stack = []  # Track hierarchy: [(level, heading), ...]

        for i, line in enumerate(lines):
            # Check for heading
            match = self.HEADING_PATTERN.match(line)

            if match:
                # Save previous heading if exists
                if current_heading:
                    current_heading.body = '\n'.join(body_lines).strip()
                    headings.append(current_heading)
                    body_lines = []

                # Parse new heading
                stars, todo_state, priority, title, tags_str = match.groups()
                level = len(stars)

                # Parse tags
                tags = []
                if tags_str:
                    tags = [t for t in tags_str.strip(':').split(':') if t]

                # Create heading
                current_heading = OrgHeading(
                    level=level,
                    title=title.strip(),
                    todo_state=todo_state,
                    tags=tags,
                    priority=priority,
                    line_number=i + 1
                )

                # Update parent stack
                # Pop all parents with level >= current level
                while parent_stack and parent_stack[-1][0] >= level:
                    parent_stack.pop()

                # Set parent reference
                if parent_stack:
                    current_heading.parent_id = self._generate_id(parent_stack[-1][1])

                # Push current heading onto stack
                parent_stack.append((level, current_heading))

            elif current_heading:
                # Check for properties drawer
                if line.strip() == ':PROPERTIES:':
                    in_properties = True
                    properties_lines = []
                elif line.strip() == ':END:' and in_properties:
                    in_properties = False
                    # Parse properties
                    props_text = '\n'.join(properties_lines)
                    current_heading.properties = self._parse_properties(props_text)
                elif in_properties:
                    properties_lines.append(line)
                else:
                    # Regular body content
                    body_lines.append(line)

                    # Check for scheduling/deadline/closed
                    if 'SCHEDULED:' in line:
                        match = self.SCHEDULED_PATTERN.search(line)
                        if match:
                            current_heading.scheduled = match.group(1)

                    if 'DEADLINE:' in line:
                        match = self.DEADLINE_PATTERN.search(line)
                        if match:
                            current_heading.deadline = match.group(1)

                    if 'CLOSED:' in line:
                        match = self.CLOSED_PATTERN.search(line)
                        if match:
                            current_heading.closed = match.group(1)

        # Save last heading
        if current_heading:
            current_heading.body = '\n'.join(body_lines).strip()
            headings.append(current_heading)

        return headings

    def _parse_properties(self, props_text: str) -> Dict[str, str]:
        """Parse org-mode properties drawer."""
        properties = {}
        for line in props_text.split('\n'):
            match = self.PROPERTY_LINE_PATTERN.match(line.strip())
            if match:
                key, value = match.groups()
                properties[key] = value.strip()
        return properties

    def _heading_to_shard(
        self,
        heading: OrgHeading,
        source: str,
        episode: str
    ) -> MemoryShard:
        """
        Convert org-mode heading to MemoryShard.

        Mapping:
        - heading.title → entity name
        - heading.body → shard text
        - heading.tags → metadata
        - heading.todo_state → motif
        - timestamps → temporal metadata for ChronoTrigger
        - properties → metadata
        - parent_id → relationship metadata
        """
        # Generate unique ID
        shard_id = self._generate_id(heading)

        # Build text content
        text_parts = [heading.title]
        if heading.body:
            text_parts.append(heading.body)
        text = '\n\n'.join(text_parts)

        # Extract entities (from title and body)
        entities = [heading.title]

        # Extract links as entities
        links = self.LINK_PATTERN.findall(text)
        for link_target, link_desc in links:
            if link_desc:
                entities.append(link_desc)
            else:
                entities.append(link_target)

        # Build motifs
        motifs = []
        if heading.todo_state:
            motifs.append(f'TODO:{heading.todo_state}')
        if heading.priority:
            motifs.append(f'PRIORITY:{heading.priority}')
        if heading.tags:
            motifs.extend([f'TAG:{tag}' for tag in heading.tags])

        # Extract timestamps from body
        timestamps = self.TIMESTAMP_PATTERN.findall(heading.body)
        if timestamps:
            motifs.append('HAS_TIMESTAMP')

        # Build metadata
        metadata = {
            'source': source,
            'line_number': heading.line_number,
            'level': heading.level,
            'org_type': 'heading',
        }

        if heading.tags:
            metadata['tags'] = heading.tags

        if heading.priority:
            metadata['priority'] = heading.priority

        if heading.scheduled:
            metadata['scheduled'] = heading.scheduled
            motifs.append('SCHEDULED')

        if heading.deadline:
            metadata['deadline'] = heading.deadline
            motifs.append('DEADLINE')

        if heading.closed:
            metadata['closed'] = heading.closed

        if heading.properties:
            metadata['properties'] = heading.properties

        if heading.parent_id:
            metadata['parent_id'] = heading.parent_id
            metadata['relationship'] = 'CHILD_OF'

        # Store links as relationships
        if links:
            metadata['links'] = [
                {'target': target, 'description': desc or target}
                for target, desc in links
            ]
            motifs.append('HAS_LINKS')

        return MemoryShard(
            id=shard_id,
            text=text,
            episode=episode,
            entities=entities,
            motifs=motifs,
            metadata=metadata
        )

    def _generate_id(self, heading: OrgHeading) -> str:
        """
        Generate unique ID for heading.

        Priority:
        1. :ID: property (standard org-mode ID)
        2. Hash of title + line number
        """
        if heading.properties.get('ID'):
            return heading.properties['ID']

        # Sanitize title for ID
        sanitized = re.sub(r'[^a-zA-Z0-9-]', '-', heading.title.lower())
        sanitized = re.sub(r'-+', '-', sanitized).strip('-')

        return f"org-{sanitized}-{heading.line_number}"

    async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
        """
        Optionally enrich shards using Ollama/other enrichers.

        For org-mode, we already have rich structure, so enrichment
        is mainly for:
        - Extracting additional entities from body text
        - Adding semantic motifs
        - Sentiment analysis for journal entries
        """
        if not hasattr(self, 'ollama') or not self.ollama:
            return shards

        for shard in shards:
            try:
                enrichment = await self.enrich(shard.text)

                if 'ollama' in enrichment:
                    ollama_data = enrichment['ollama']

                    # Merge entities
                    if ollama_data.get('entities'):
                        shard.entities.extend(ollama_data['entities'])
                        shard.entities = list(set(shard.entities))

                    # Merge motifs
                    if ollama_data.get('motifs'):
                        shard.motifs.extend(ollama_data['motifs'])
                        shard.motifs = list(set(shard.motifs))

                # Store enrichment in metadata
                if enrichment:
                    shard.metadata['enrichment'] = enrichment

            except Exception as e:
                shard.metadata['enrichment_error'] = str(e)

        return shards


# Utility functions for file watching and live updates
class OrgFileWatcher:
    """
    Watch org-mode files for changes and incrementally update MemoryShards.

    Enables "living" knowledge base where org-mode remains the source of truth.

    Future enhancement:
    - Use inotify/watchdog to monitor file changes
    - Parse only modified sections
    - Update YarnGraph incrementally
    - Preserve temporal evolution (keep old versions)
    """

    def __init__(self, spinner: OrgModeSpinner):
        self.spinner = spinner
        self.file_cache: Dict[str, Tuple[float, List[MemoryShard]]] = {}

    async def watch_file(self, filepath: str) -> List[MemoryShard]:
        """
        Watch org file and return updated shards if changed.

        Args:
            filepath: Path to org file

        Returns:
            Updated MemoryShards if file changed, else cached shards
        """
        import os

        mtime = os.path.getmtime(filepath)

        # Check cache
        if filepath in self.file_cache:
            cached_mtime, cached_shards = self.file_cache[filepath]
            if mtime == cached_mtime:
                return cached_shards

        # File changed or not cached - re-parse
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        shards = await self.spinner.spin({
            'content': content,
            'source': filepath,
            'episode': f'org:{os.path.basename(filepath)}'
        })

        # Update cache
        self.file_cache[filepath] = (mtime, shards)

        return shards
