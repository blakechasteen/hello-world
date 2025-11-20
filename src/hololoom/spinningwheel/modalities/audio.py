# -*- coding: utf-8 -*-
"""
Audio Spinner
=============
Input adapter for ingesting audio transcripts and metadata as MemoryShards.

Converts audio data (transcripts, summaries, task lists) into structured
memory shards that can be processed by the HoloLoom orchestrator.
"""

from typing import List, Dict, Any
from dataclasses import dataclass

from .base import BaseSpinner, SpinnerConfig

# Import HoloLoom types
try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    # Fallback if types not available
    from dataclasses import dataclass, field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSpinnerConfig(SpinnerConfig):
    """Configuration for Audio spinner."""
    pass


class AudioSpinner(BaseSpinner):
    """
    Spinner for audio transcripts and metadata.

    Converts audio data into MemoryShards suitable for HoloLoom processing.
    """

    def __init__(self, config: AudioSpinnerConfig = None):
        if config is None:
            config = AudioSpinnerConfig()
        super().__init__(config)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert audio data → MemoryShards.

        Args:
            raw_data: Dict with keys like:
                - 'transcript': Main transcript text
                - 'summary': Summary of content
                - 'tasks': List of tasks/action items
                - 'episode': Episode/session identifier
                - 'metadata': Additional metadata

        Returns:
            List of MemoryShard objects
        """
        shards = []
        episode = raw_data.get('episode', 'audio_session')

        # Create shard from transcript
        if 'transcript' in raw_data:
            shard = MemoryShard(
                id=f"{episode}_transcript",
                text=raw_data['transcript'],
                episode=episode,
                entities=[],
                motifs=[],
                metadata={
                    'type': 'transcript',
                    **raw_data.get('metadata', {})
                }
            )
            shards.append(shard)

        # Create shard from summary
        if 'summary' in raw_data:
            shard = MemoryShard(
                id=f"{episode}_summary",
                text=raw_data['summary'],
                episode=episode,
                entities=[],
                motifs=[],
                metadata={
                    'type': 'summary',
                    **raw_data.get('metadata', {})
                }
            )
            shards.append(shard)

        # Create shard from tasks
        if 'tasks' in raw_data:
            tasks_text = '\n'.join(raw_data['tasks']) if isinstance(raw_data['tasks'], list) else raw_data['tasks']
            shard = MemoryShard(
                id=f"{episode}_tasks",
                text=tasks_text,
                episode=episode,
                entities=[],
                motifs=[],
                metadata={
                    'type': 'tasks',
                    **raw_data.get('metadata', {})
                }
            )
            shards.append(shard)

        # Optional enrichment
        if self.config.enable_enrichment:
            shards = await self._enrich_shards(shards)

        return shards

    async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
        """
        Enrich shards using optional enrichment services.
        Delegates to parent class enrichment infrastructure.
        """
        for shard in shards:
            enrichment = await self.enrich(shard.text)

            # Merge enrichment results into shard
            if enrichment:
                # Update entities from enrichment
                if 'ollama' in enrichment and 'entities' in enrichment['ollama']:
                    shard.entities.extend(enrichment['ollama']['entities'])
                    shard.entities = list(set(shard.entities))

                # Update motifs from enrichment
                if 'ollama' in enrichment and 'motifs' in enrichment['ollama']:
                    shard.motifs.extend(enrichment['ollama']['motifs'])
                    shard.motifs = list(set(shard.motifs))

                # Store enrichment in metadata
                if shard.metadata is None:
                    shard.metadata = {}
                shard.metadata['enrichment'] = enrichment

        return shards
