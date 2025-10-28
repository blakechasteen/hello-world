# -*- coding: utf-8 -*-
"""
Beekeeping Spinners
===================
Domain-specific spinners for beekeeping data ingestion.

Spinners:
- BeeInspectionAudioSpinner: Process voice notes from hive inspections
- HivePhotoSpinner: Process hive photos with frame analysis
- WeatherDataSpinner: Ingest weather data for beekeeping context
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re
from datetime import datetime

# Import base spinners
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinning_wheel.audio import AudioSpinner, AudioSpinnerConfig
from HoloLoom.spinning_wheel.image import ImageSpinner, ImageSpinnerConfig

try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    from dataclasses import field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

logger = logging.getLogger(__name__)


@dataclass
class BeeInspectionConfig(AudioSpinnerConfig):
    """Configuration for bee inspection audio spinner."""
    auto_extract_hive_id: bool = True  # Automatically extract hive ID from text
    auto_extract_queen_status: bool = True  # Extract queen presence/status
    auto_extract_population: bool = True  # Extract population estimates
    auto_extract_health_issues: bool = True  # Extract health concerns
    auto_extract_actions: bool = True  # Extract actions taken/needed
    ollama_enhancement: bool = True  # Use Ollama for structured extraction


class BeeInspectionAudioSpinner(AudioSpinner):
    """
    Spinner for beekeeping hive inspection audio notes.

    Extracts structured data from voice notes recorded during hive inspections:
    - Hive identification
    - Queen status (present, laying, marked)
    - Population estimates
    - Brood pattern (solid, spotty, cells)
    - Health issues (mites, disease, pests)
    - Temperament (calm, defensive, aggressive)
    - Resources (honey stores, pollen, nectar flow)
    - Actions taken (added supers, treated, fed)
    - Weather conditions during inspection
    """

    HEALTH_KEYWORDS = [
        'mites', 'varroa', 'nosema', 'foulbrood', 'chalkbrood', 'disease',
        'beetles', 'wax moths', 'robbing', 'weak', 'dying', 'dead'
    ]

    ACTION_KEYWORDS = [
        'added', 'removed', 'treated', 'fed', 'requeened', 'split', 'combined',
        'moved', 'installed', 'harvested', 'checked', 'reversed'
    ]

    QUEEN_KEYWORDS = [
        'queen', 'laying', 'eggs', 'marked', 'clipped', 'supersedure',
        'emergency cells', 'queen cells'
    ]

    def __init__(self, config: BeeInspectionConfig = None):
        if config is None:
            config = BeeInspectionConfig()
        super().__init__(config)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert bee inspection audio → structured MemoryShards.

        Args:
            raw_data: Dict with keys:
                - 'transcript': Transcribed audio text
                - 'hive_id': Hive identifier (optional, will auto-extract)
                - 'inspection_date': Date of inspection
                - 'inspector': Person conducting inspection
                - 'location': Apiary location
                - 'metadata': Additional metadata

        Returns:
            MemoryShards with structured inspection data
        """
        # Get base audio shard
        shards = await super().spin(raw_data)
        if not shards:
            return shards

        # Extract structured inspection data
        shard = shards[0]
        transcript = raw_data.get('transcript', shard.text)

        inspection_data = await self._parse_inspection(transcript, raw_data)

        # Update shard with structured data
        shard.metadata['inspection'] = inspection_data
        shard.metadata['type'] = 'bee_inspection'
        shard.metadata['hive_id'] = inspection_data.get('hive_id')
        shard.metadata['inspection_date'] = inspection_data.get('date')

        # Extract entities (hives, locations, equipment)
        shard.entities = self._extract_entities(inspection_data)

        # Extract motifs (health issues, actions, concerns)
        shard.motifs = self._extract_motifs(inspection_data, transcript)

        # Create structured summary
        shard.text = self._format_inspection_summary(inspection_data, transcript)

        return shards

    async def _parse_inspection(self, transcript: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse inspection structure from transcript."""
        inspection_data = {
            'hive_id': raw_data.get('hive_id'),
            'date': raw_data.get('inspection_date', datetime.now().isoformat()),
            'inspector': raw_data.get('inspector', 'Unknown'),
            'location': raw_data.get('location'),
            'queen_status': None,
            'population_estimate': None,
            'brood_pattern': None,
            'temperament': None,
            'health_issues': [],
            'actions_taken': [],
            'actions_needed': [],
            'honey_stores': None,
            'pollen_stores': None,
            'weather': raw_data.get('weather'),
            'notes': transcript
        }

        # Auto-extract hive ID if not provided
        if self.config.auto_extract_hive_id and not inspection_data['hive_id']:
            hive_id = self._extract_hive_id(transcript)
            if hive_id:
                inspection_data['hive_id'] = hive_id

        # Extract queen status
        if self.config.auto_extract_queen_status:
            inspection_data['queen_status'] = self._extract_queen_status(transcript)

        # Extract population estimate
        if self.config.auto_extract_population:
            inspection_data['population_estimate'] = self._extract_population(transcript)

        # Extract health issues
        if self.config.auto_extract_health_issues:
            inspection_data['health_issues'] = self._extract_health_issues(transcript)

        # Extract actions
        if self.config.auto_extract_actions:
            actions = self._extract_actions(transcript)
            inspection_data['actions_taken'] = actions.get('taken', [])
            inspection_data['actions_needed'] = actions.get('needed', [])

        # Extract temperament
        inspection_data['temperament'] = self._extract_temperament(transcript)

        # Extract stores
        stores = self._extract_stores(transcript)
        inspection_data['honey_stores'] = stores.get('honey')
        inspection_data['pollen_stores'] = stores.get('pollen')

        # Use Ollama for enhanced extraction if enabled
        if self.config.ollama_enhancement and self.config.enable_enrichment:
            enhanced = await self._enhance_with_ollama(transcript, inspection_data)
            inspection_data.update(enhanced)

        return inspection_data

    def _extract_hive_id(self, transcript: str) -> Optional[str]:
        """Extract hive ID from transcript."""
        # Look for patterns like "hive 1", "hive A", "colony 5", etc.
        patterns = [
            r'hive\s+([A-Za-z0-9]+)',
            r'colony\s+([A-Za-z0-9]+)',
            r'box\s+([A-Za-z0-9]+)',
            r'number\s+([A-Za-z0-9]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, transcript, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _extract_queen_status(self, transcript: str) -> Optional[str]:
        """Extract queen status from transcript."""
        lower_text = transcript.lower()

        if any(word in lower_text for word in ['no queen', 'queenless', "can't find queen"]):
            return 'queenless'
        elif 'queen cells' in lower_text:
            return 'queen_cells_present'
        elif 'eggs' in lower_text or 'laying' in lower_text:
            return 'present_laying'
        elif 'queen' in lower_text and 'saw' in lower_text:
            return 'present_not_confirmed_laying'
        elif 'queen' in lower_text:
            return 'mentioned'

        return None

    def _extract_population(self, transcript: str) -> Optional[str]:
        """Extract population estimate from transcript."""
        lower_text = transcript.lower()

        # Look for frame counts
        frame_match = re.search(r'(\d+)\s+frames?\s+of\s+bees', lower_text)
        if frame_match:
            frames = int(frame_match.group(1))
            if frames >= 8:
                return 'strong'
            elif frames >= 5:
                return 'moderate'
            else:
                return 'weak'

        # Look for qualitative descriptions
        if 'strong' in lower_text or 'booming' in lower_text:
            return 'strong'
        elif 'weak' in lower_text or 'small' in lower_text:
            return 'weak'
        elif 'moderate' in lower_text or 'average' in lower_text:
            return 'moderate'

        return None

    def _extract_health_issues(self, transcript: str) -> List[str]:
        """Extract health issues from transcript."""
        lower_text = transcript.lower()
        issues = []

        for keyword in self.HEALTH_KEYWORDS:
            if keyword in lower_text:
                issues.append(keyword)

        return issues

    def _extract_actions(self, transcript: str) -> Dict[str, List[str]]:
        """Extract actions taken and needed from transcript."""
        lower_text = transcript.lower()
        actions = {'taken': [], 'needed': []}

        for keyword in self.ACTION_KEYWORDS:
            if keyword in lower_text:
                # Determine if past tense (taken) or future (needed)
                # Simple heuristic: look for "will", "need to", "should"
                context_window = 50
                keyword_pos = lower_text.find(keyword)
                if keyword_pos != -1:
                    context = lower_text[max(0, keyword_pos - context_window):keyword_pos + context_window]

                    if any(word in context for word in ['will', 'need', 'should', 'must', 'plan']):
                        actions['needed'].append(keyword)
                    else:
                        actions['taken'].append(keyword)

        return actions

    def _extract_temperament(self, transcript: str) -> Optional[str]:
        """Extract hive temperament from transcript."""
        lower_text = transcript.lower()

        if any(word in lower_text for word in ['aggressive', 'defensive', 'hot', 'angry']):
            return 'defensive'
        elif any(word in lower_text for word in ['calm', 'gentle', 'docile', 'easy']):
            return 'calm'
        elif 'temperament' in lower_text:
            return 'mentioned'

        return None

    def _extract_stores(self, transcript: str) -> Dict[str, Optional[str]]:
        """Extract honey and pollen stores from transcript."""
        lower_text = transcript.lower()
        stores = {'honey': None, 'pollen': None}

        # Honey stores
        if 'lots of honey' in lower_text or 'plenty of honey' in lower_text:
            stores['honey'] = 'abundant'
        elif 'some honey' in lower_text:
            stores['honey'] = 'moderate'
        elif 'no honey' in lower_text or 'low on honey' in lower_text:
            stores['honey'] = 'low'

        # Pollen stores
        if 'lots of pollen' in lower_text or 'plenty of pollen' in lower_text:
            stores['pollen'] = 'abundant'
        elif 'some pollen' in lower_text:
            stores['pollen'] = 'moderate'
        elif 'no pollen' in lower_text or 'low on pollen' in lower_text:
            stores['pollen'] = 'low'

        return stores

    async def _enhance_with_ollama(self, transcript: str, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Ollama to enhance extraction with structured prompting."""
        if not hasattr(self, 'ollama') or not self.ollama:
            return {}

        try:
            prompt = f"""
You are a beekeeping expert analyzing hive inspection notes. Extract structured data from the following transcript:

{transcript}

Please extract:
1. Queen status (present/absent, laying/not laying, marked color if mentioned)
2. Population strength (weak/moderate/strong)
3. Brood pattern quality (excellent/good/spotty/poor)
4. Health concerns (varroa mites, diseases, pests)
5. Actions taken during inspection
6. Actions needed for next time
7. Overall hive assessment

Respond in a structured format.
"""
            enrichment = await self.enrich(prompt)
            # Parse Ollama response and merge with base_data
            # This is a simplified version - you'd parse the actual response
            return enrichment

        except Exception as e:
            logger.warning(f"Ollama enhancement failed: {e}")
            return {}

    def _extract_entities(self, inspection_data: Dict[str, Any]) -> List[str]:
        """Extract entities from structured inspection data."""
        entities = []

        if inspection_data.get('hive_id'):
            entities.append(f"hive_{inspection_data['hive_id']}")

        if inspection_data.get('location'):
            entities.append(inspection_data['location'])

        if inspection_data.get('inspector'):
            entities.append(inspection_data['inspector'])

        # Add health issues as entities
        entities.extend(inspection_data.get('health_issues', []))

        return entities

    def _extract_motifs(self, inspection_data: Dict[str, Any], transcript: str) -> List[str]:
        """Extract motifs from inspection data."""
        motifs = ['bee_inspection', 'beekeeping']

        # Add status-based motifs
        if inspection_data.get('queen_status'):
            motifs.append(f"queen_{inspection_data['queen_status']}")

        if inspection_data.get('population_estimate'):
            motifs.append(f"population_{inspection_data['population_estimate']}")

        if inspection_data.get('health_issues'):
            motifs.append('health_concern')

        if inspection_data.get('temperament') == 'defensive':
            motifs.append('defensive_hive')

        # Add action motifs
        if inspection_data.get('actions_taken'):
            motifs.append('actions_taken')
        if inspection_data.get('actions_needed'):
            motifs.append('actions_needed')

        return motifs

    def _format_inspection_summary(self, inspection_data: Dict[str, Any], transcript: str) -> str:
        """Format structured inspection data as readable summary."""
        lines = []

        lines.append(f"=== Hive Inspection: {inspection_data.get('hive_id', 'Unknown')} ===")
        lines.append(f"Date: {inspection_data.get('date')}")
        lines.append(f"Inspector: {inspection_data.get('inspector')}")

        if inspection_data.get('location'):
            lines.append(f"Location: {inspection_data['location']}")

        lines.append("")
        lines.append("## Status")

        if inspection_data.get('queen_status'):
            lines.append(f"Queen: {inspection_data['queen_status']}")

        if inspection_data.get('population_estimate'):
            lines.append(f"Population: {inspection_data['population_estimate']}")

        if inspection_data.get('temperament'):
            lines.append(f"Temperament: {inspection_data['temperament']}")

        if inspection_data.get('honey_stores'):
            lines.append(f"Honey Stores: {inspection_data['honey_stores']}")

        if inspection_data.get('pollen_stores'):
            lines.append(f"Pollen Stores: {inspection_data['pollen_stores']}")

        if inspection_data.get('health_issues'):
            lines.append("")
            lines.append("## Health Issues")
            for issue in inspection_data['health_issues']:
                lines.append(f"  - {issue}")

        if inspection_data.get('actions_taken'):
            lines.append("")
            lines.append("## Actions Taken")
            for action in inspection_data['actions_taken']:
                lines.append(f"  - {action}")

        if inspection_data.get('actions_needed'):
            lines.append("")
            lines.append("## Actions Needed")
            for action in inspection_data['actions_needed']:
                lines.append(f"  - {action}")

        lines.append("")
        lines.append("## Original Notes")
        lines.append(transcript)

        return '\n'.join(lines)


# Convenience factory
async def process_bee_inspection(transcript: str, hive_id: str = None, **kwargs) -> List[MemoryShard]:
    """Quick helper to process bee inspection audio transcript."""
    # Separate config kwargs from data kwargs
    config_kwargs = {k: v for k, v in kwargs.items() if k in [
        'auto_extract_hive_id', 'auto_extract_queen_status', 'auto_extract_population',
        'auto_extract_health_issues', 'auto_extract_actions', 'ollama_enhancement',
        'enable_enrichment', 'ollama_model'
    ]}
    data_kwargs = {k: v for k, v in kwargs.items() if k not in config_kwargs}

    config = BeeInspectionConfig(**config_kwargs)
    spinner = BeeInspectionAudioSpinner(config)

    raw_data = {
        'transcript': transcript,
        'episode': f"inspection_{hive_id or 'unknown'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }

    if hive_id:
        raw_data['hive_id'] = hive_id

    raw_data.update(data_kwargs)

    return await spinner.spin(raw_data)