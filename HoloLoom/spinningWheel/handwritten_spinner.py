"""
Handwritten Spinner - OCR for Handwritten Notes
================================================

Specialized spinner for extracting text from handwritten notes and documents.

Features:
- Optimized for handwritten text (cursive, print)
- Multiple OCR backends with handwriting support
- Note structure detection (titles, bullet points, sketches)
- Sketch/diagram detection
- Signature detection
- Entity extraction (names, dates, tasks)
- Importance scoring tailored to notes

Supports:
- Handwritten notes (photos, scans)
- Mixed text/drawings
- Forms with handwritten entries
- Signatures and annotations

Usage:
    from HoloLoom.spinningWheel import HandwrittenSpinner

    spinner = HandwrittenSpinner()
    result = await spinner.spin("handwritten_note.jpg")

    # Batch processing
    result = await spinner.spin(["note1.jpg", "note2.png"])

Author: Claude Code
Date: January 2025
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import re

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.ocr_protocol import OCROutputFormat
from HoloLoom.spinningWheel.ocr_backends import get_all_available_backends


@dataclass
class NoteStructure:
    """Structure detected in handwritten note"""
    has_title: bool
    has_bullets: bool
    has_numbering: bool
    has_sketches: bool
    has_signatures: bool
    sections: List[str]
    detected_tasks: List[str]


class HandwrittenSpinner(BaseSpinner):
    """
    Specialized spinner for handwritten notes.

    Optimized for:
    - Handwritten text extraction
    - Note structure detection
    - Task/TODO extraction
    - Sketch/diagram detection
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        detect_tasks: bool = True,
        detect_sketches: bool = True,
        enable_enrichment: bool = False
    ):
        """
        Initialize HandwrittenSpinner.

        Args:
            importance_threshold: Min importance score (0.0-1.0)
            detect_tasks: Detect TODO items and tasks
            detect_sketches: Detect non-text content (sketches, diagrams)
            enable_enrichment: Extract entities/motifs via Ollama
        """
        super().__init__(name="handwritten", importance_threshold=importance_threshold)

        self.detect_tasks = detect_tasks
        self.detect_sketches = detect_sketches
        self.enable_enrichment = enable_enrichment

        # Get OCR backend chain
        self.ocr_chain = get_all_available_backends()

        print(f"[HandwrittenSpinner] Using OCR backend: {self.ocr_chain.get_name()} ({self.ocr_chain.get_quality().value})")
        print(f"[HandwrittenSpinner] Note: For best handwriting recognition, use DeepSeek OCR or Google Cloud Vision")

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,  # Always extract names, dates, etc.
            motif_extraction=True,
            embeddings=False,
            multimodal=False,
            supported_formats=['.jpg', '.jpeg', '.png', '.tiff', '.pdf']
        )

    def is_available(self) -> bool:
        """Check if OCR backend is available."""
        return self.ocr_chain.is_available()

    async def _spin_impl(
        self,
        source: Any,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Extract text from handwritten note(s).

        Args:
            source: Image path(s) - str, Path, or list
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        # Normalize to list of paths
        if isinstance(source, (str, Path)):
            note_paths = [Path(source)]
        elif isinstance(source, list):
            note_paths = [Path(p) for p in source]
        else:
            raise ValueError(f"source must be path or list of paths, got {type(source)}")

        # Validate paths
        for path in note_paths:
            if not path.exists():
                raise FileNotFoundError(f"Note not found: {path}")

        # Extract text via OCR
        all_shards = []

        for note_path in note_paths:
            # OCR extraction (TEXT format for easier parsing)
            ocr_result = await self.ocr_chain.extract_text(
                note_path,
                output_format=OCROutputFormat.TEXT
            )

            # Analyze note structure
            structure = self._analyze_note_structure(ocr_result.text)

            # Create shard
            shard = await self._create_shard_from_note(
                note_path,
                ocr_result.text,
                ocr_result,
                structure
            )

            all_shards.append(shard)

        return all_shards

    async def _create_shard_from_note(
        self,
        note_path: Path,
        text: str,
        ocr_result: Any,
        structure: NoteStructure
    ) -> MemoryShard:
        """
        Create MemoryShard from handwritten note.

        Args:
            note_path: Path to note image
            text: Extracted text
            ocr_result: OCRResult from backend
            structure: Detected note structure

        Returns:
            MemoryShard
        """
        # Extract entities (names, dates, places)
        entities = self._extract_note_entities(text, structure)

        # Extract motifs (topics, categories)
        motifs = self._extract_note_motifs(text, structure)

        # Enrich if enabled
        if self.enable_enrichment and len(text) > 50:
            enriched_entities, enriched_motifs = await self._enrich_content(text)
            entities.extend(enriched_entities)
            motifs.extend(enriched_motifs)

        # Deduplicate
        entities = list(set(entities))[:20]
        motifs = list(set(motifs))[:10]

        # Score importance
        importance = self._score_note_importance(text, structure, ocr_result.confidence)

        # Create shard
        shard = self._create_shard(
            id_suffix=f"{note_path.stem}_{int(time.time())}",
            text=text,
            episode=f"handwritten_{note_path.stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'file_path': str(note_path),
                'file_name': note_path.name,
                'file_size': note_path.stat().st_size,
                'ocr_backend': ocr_result.backend,
                'ocr_confidence': ocr_result.confidence,
                'ocr_quality': ocr_result.quality.value,
                'note_type': 'handwritten',
                'has_title': structure.has_title,
                'has_bullets': structure.has_bullets,
                'has_numbering': structure.has_numbering,
                'has_sketches': structure.has_sketches,
                'has_signatures': structure.has_signatures,
                'section_count': len(structure.sections),
                'task_count': len(structure.detected_tasks),
                'detected_tasks': structure.detected_tasks,
                'importance_score': importance.score,
                'importance_reason': importance.reason
            }
        )

        return shard

    def _analyze_note_structure(self, text: str) -> NoteStructure:
        """
        Analyze structure of handwritten note.

        Detects:
        - Titles (first line, short, capitalized)
        - Bullet points (•, -, *, etc.)
        - Numbered lists (1., 2., etc.)
        - Tasks/TODOs
        - Signatures
        - Sections
        """
        lines = text.split('\n')

        # Detect title (first significant line)
        has_title = False
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and first_line.istitle():
                has_title = True

        # Detect bullets
        has_bullets = any(
            line.strip().startswith(('•', '-', '*', '·'))
            for line in lines
        )

        # Detect numbering
        has_numbering = bool(re.search(r'^\s*\d+[\.)]\s+', text, re.MULTILINE))

        # Detect sketches (non-text indicators)
        sketch_indicators = ['[sketch]', '[diagram]', '[drawing]', '[figure]']
        has_sketches = any(indicator in text.lower() for indicator in sketch_indicators)

        # Detect signatures
        signature_indicators = ['signed', 'signature', '/s/', '___']
        has_signatures = any(indicator in text.lower() for indicator in signature_indicators)

        # Detect sections (blank line separated blocks)
        sections = []
        current_section = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            else:
                current_section.append(stripped)

        if current_section:
            sections.append('\n'.join(current_section))

        # Detect tasks
        detected_tasks = []
        if self.detect_tasks:
            task_patterns = [
                r'(?:TODO|To do|Task):?\s*(.+)',
                r'\[\s*\]\s*(.+)',  # [ ] checkbox
                r'[-*]\s*\[\s*\]\s*(.+)',
                r'(?:Need to|Must|Should):?\s*(.+)'
            ]

            for pattern in task_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    task = match.group(1).strip()
                    if task and len(task) > 3:
                        detected_tasks.append(task)

        return NoteStructure(
            has_title=has_title,
            has_bullets=has_bullets,
            has_numbering=has_numbering,
            has_sketches=has_sketches,
            has_signatures=has_signatures,
            sections=sections,
            detected_tasks=detected_tasks[:10]  # Limit to 10
        )

    def _extract_note_entities(
        self,
        text: str,
        structure: NoteStructure
    ) -> List[str]:
        """
        Extract entities from handwritten note.

        Focuses on:
        - Names (proper nouns, capitalized words)
        - Dates
        - Locations
        - Organizations
        """
        entities = []

        # Extract dates (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
        ]

        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(dates)

        # Extract capitalized words (potential names/places)
        words = text.split()
        for word in words:
            # Capitalized word longer than 2 chars (not all caps)
            if len(word) > 2 and word[0].isupper() and not word.isupper():
                entities.append(word.strip('.,!?'))

        # Extract from title if present
        if structure.has_title and structure.sections:
            title = structure.sections[0]
            entities.append(title)

        return list(set(entities))

    def _extract_note_motifs(
        self,
        text: str,
        structure: NoteStructure
    ) -> List[str]:
        """
        Extract motifs/topics from handwritten note.

        Args:
            text: Note text
            structure: Note structure

        Returns:
            List of motif strings
        """
        motifs = []

        # Structure-based motifs
        if structure.has_title:
            motifs.append('titled_note')
        if structure.has_bullets or structure.has_numbering:
            motifs.append('list_format')
        if structure.has_sketches:
            motifs.append('visual_content')
        if structure.has_signatures:
            motifs.append('signed_document')
        if structure.detected_tasks:
            motifs.append('task_list')
            motifs.append('action_items')

        # Content-based motifs
        text_lower = text.lower()

        # Meeting notes
        if any(word in text_lower for word in ['meeting', 'agenda', 'attendees', 'minutes']):
            motifs.append('meeting_notes')

        # Personal notes
        if any(word in text_lower for word in ['remember', 'note to self', 'reminder']):
            motifs.append('personal_note')

        # Ideas/brainstorming
        if any(word in text_lower for word in ['idea', 'brainstorm', 'concept']):
            motifs.append('ideas')

        # Instructions/procedures
        if structure.has_numbering:
            motifs.append('instructions')

        return list(set(motifs))

    def _score_note_importance(
        self,
        text: str,
        structure: NoteStructure,
        ocr_confidence: float
    ) -> ImportanceScore:
        """
        Score importance of handwritten note.

        Factors:
        - Text length (longer = more substantial)
        - Structure (lists, sections)
        - OCR confidence (readability indicator)
        - Task count (action items)
        - Entity density (names, dates)
        """
        signals = ImportanceSignals()

        # Length score
        text_length = len(text)
        signals.length_score = min(1.0, text_length / 500)  # Handwritten notes shorter

        # Technical score (less relevant for personal notes)
        signals.technical_score = 0.3  # Neutral

        # Structural score (well-organized notes)
        struct_score = 0.0
        if structure.has_title:
            struct_score += 0.3
        if structure.has_bullets or structure.has_numbering:
            struct_score += 0.3
        if len(structure.sections) > 1:
            struct_score += 0.2
        if structure.detected_tasks:
            struct_score += 0.2
        signals.structural_score = min(1.0, struct_score)

        # Authority score (OCR confidence proxy for handwriting quality)
        signals.authority_score = ocr_confidence

        # Engagement score (task count)
        signals.engagement_score = min(1.0, len(structure.detected_tasks) / 5.0)

        # Custom signals
        signals.custom_signals = {
            'ocr_confidence': ocr_confidence,
            'task_count': len(structure.detected_tasks),
            'has_structure': 1.0 if structure.has_bullets or structure.has_numbering else 0.0
        }

        # Noise penalty (very short or illegible)
        if text_length < 20:
            signals.noise_penalty = -0.5
        if ocr_confidence < 0.3:
            signals.noise_penalty = -0.3  # Low readability

        # Compute total
        score = signals.compute_total()

        # Generate reason
        reasons = []
        if structure.detected_tasks:
            reasons.append(f"{len(structure.detected_tasks)} tasks")
        if structure.has_title:
            reasons.append("titled note")
        if structure.has_bullets or structure.has_numbering:
            reasons.append("structured list")
        if ocr_confidence > 0.7:
            reasons.append("readable handwriting")

        reason = " + ".join(reasons) if reasons else "handwritten note"

        return ImportanceScore(
            score=score,
            signals=signals,
            reason=reason
        )

    async def _enrich_content(self, text: str) -> tuple[List[str], List[str]]:
        """
        Extract entities and motifs using Ollama.

        Returns:
            (entities, motifs)
        """
        try:
            import ollama
        except ImportError:
            return [], []

        try:
            # Extract entities
            entity_prompt = f"""Extract key entities (people, organizations, locations, dates) from this handwritten note.
Return only a comma-separated list of entities.

Text: {text[:1000]}

Entities:"""

            entity_response = ollama.generate(
                model="llama3.2:3b",
                prompt=entity_prompt
            )

            entities = [
                e.strip()
                for e in entity_response['response'].split(',')
                if e.strip()
            ][:20]

            # Extract motifs/topics
            motif_prompt = f"""Extract key topics and categories from this handwritten note.
Return only a comma-separated list of topics.

Text: {text[:1000]}

Topics:"""

            motif_response = ollama.generate(
                model="llama3.2:3b",
                prompt=motif_prompt
            )

            motifs = [
                m.strip()
                for m in motif_response['response'].split(',')
                if m.strip()
            ][:10]

            return entities, motifs

        except Exception as e:
            print(f"Warning: Enrichment failed: {e}")
            return [], []


# Convenience function

async def spin_handwritten(
    note_path: str,
    detect_tasks: bool = True,
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to extract text from handwritten note.

    Args:
        note_path: Path to handwritten note image
        detect_tasks: Detect TODO items and tasks
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards

    Example:
        >>> result = await spin_handwritten("meeting_notes.jpg")
        >>> print(result.shards[0].metadata['detected_tasks'])
    """
    spinner = HandwrittenSpinner(
        importance_threshold=importance_threshold,
        detect_tasks=detect_tasks
    )

    result = await spinner.spin(note_path)

    return result