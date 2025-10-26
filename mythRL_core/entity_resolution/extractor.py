"""
Entity Extractor - Auto-extract structured data from text
==========================================================

Combines entity resolution with measurement extraction to create
structured payloads for vector storage.

Supports:
- Numeric extraction (8 frames, 2.5 lbs, etc.)
- Categorical extraction (calm, aggressive, solid brood, etc.)
- Temporal extraction (dates, times)
- Entity linking (automatic canonical ID resolution)
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .resolver import EntityResolver


@dataclass
class ExtractedData:
    """Structured data extracted from text."""

    text: str
    entities: List[Dict[str, Any]]
    measurements: Dict[str, Any]
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert to Qdrant-compatible payload."""
        payload = {
            "text": self.text,
            "entity_ids": [e["canonical_id"] for e in self.entities],
            "entity_types": {e["canonical_id"]: e["entity_type"] for e in self.entities},
        }

        # Add primary entity if exists
        if self.entities:
            primary = self.entities[0]  # First/strongest match
            payload["primary_entity_id"] = primary["canonical_id"]
            payload["primary_entity_name"] = primary["canonical_name"]
            payload["primary_entity_type"] = primary["entity_type"]

        # Add timestamp
        if self.timestamp:
            payload["timestamp"] = self.timestamp.isoformat()
            payload["date"] = self.timestamp.date().isoformat()

        # Add measurements (flattened)
        if self.measurements:
            payload.update(self.measurements)
            payload["has_measurements"] = True
        else:
            payload["has_measurements"] = False

        # Add metadata
        if self.metadata:
            payload["metadata"] = self.metadata

        return payload


class MeasurementExtractor:
    """Extract numeric and categorical measurements from text."""

    # Default patterns (beekeeping - for backwards compatibility)
    DEFAULT_NUMERIC_PATTERNS = {
        "frames_of_bees": r"(\d+(?:\.\d+)?)\s*frames?\s*(?:of\s*)?(?:bees?|brood)",
        "frames_of_brood": r"(\d+(?:\.\d+)?)\s*frames?\s*(?:of\s*)?brood",
        "weight_lbs": r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)",
        "dosage_grams": r"(\d+(?:\.\d+)?)\s*(?:g|grams?)",
        "temperature_f": r"(\d+)\s*(?:degrees?|°)?\s*F",
        "days_interval": r"(\d+)\s*days?",
    }

    DEFAULT_CATEGORICAL_PATTERNS = {
        "temperament": r"\b(calm|defensive|aggressive|gentle|hot)\b",
        "brood_pattern": r"\b(solid|spotty|scattered|patchy|none)\b",
        "strength_level": r"\b(weak|moderate|strong|very\s+strong|excellent)\b",
        "queen_status": r"\b(seen|not\s+seen|missing|laying|not\s+laying)\b",
    }

    def __init__(self, custom_patterns: Optional[Dict[str, Any]] = None):
        """
        Initialize with optional custom patterns from domain registry.

        Args:
            custom_patterns: Dict with 'numeric' and 'categorical' pattern definitions
        """
        self.numeric_patterns = self.DEFAULT_NUMERIC_PATTERNS.copy()
        self.categorical_patterns = self.DEFAULT_CATEGORICAL_PATTERNS.copy()

        if custom_patterns:
            # Load custom numeric patterns
            if "numeric" in custom_patterns:
                for key, spec in custom_patterns["numeric"].items():
                    self.numeric_patterns[key] = spec["pattern"]

            # Load custom categorical patterns
            if "categorical" in custom_patterns:
                for key, spec in custom_patterns["categorical"].items():
                    self.categorical_patterns[key] = spec["pattern"]

    def extract_all(self, text: str) -> Dict[str, Any]:
        """Extract all measurements from text."""
        measurements = {}

        # Extract numeric measurements
        for key, pattern in self.numeric_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Remove commas for large numbers (e.g., "87,450" -> "87450")
                num_str = match.group(1).replace(',', '')
                measurements[key] = float(num_str)

        # Extract categorical measurements
        for key, pattern in self.categorical_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                measurements[key] = match.group(1).lower()

        return measurements


class TemporalExtractor:
    """Extract dates and times from text."""

    # Common date patterns
    DATE_PATTERNS = [
        (r"(\d{4})-(\d{2})-(\d{2})", "%Y-%m-%d"),  # 2024-10-12
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),  # 10/12/2024
        (r"(?:Oct(?:ober)?)\s+(\d{1,2})(?:,\s*(\d{4}))?", None),  # October 12, 2024
    ]

    @classmethod
    def extract_date(cls, text: str, default: Optional[datetime] = None) -> Optional[datetime]:
        """
        Extract date from text.

        Args:
            text: Text to search
            default: Default date if none found

        Returns:
            datetime object or None
        """
        # Try ISO format first
        match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if match:
            try:
                return datetime.fromisoformat(match.group(1))
            except ValueError:
                pass

        # Try other formats
        for pattern, fmt in cls.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and fmt:
                try:
                    return datetime.strptime(match.group(0), fmt)
                except ValueError:
                    continue

        return default


class EntityExtractor:
    """
    Complete entity + measurement extraction pipeline.

    Combines entity resolution with measurement extraction to create
    rich, structured payloads for vector storage.
    """

    def __init__(self, resolver: EntityResolver, custom_patterns: Optional[Dict[str, Any]] = None):
        """
        Initialize extractor with resolver and optional custom measurement patterns.

        Args:
            resolver: EntityResolver instance with loaded domain
            custom_patterns: Optional dict with measurement patterns from domain registry
        """
        self.resolver = resolver
        self.measurement_extractor = MeasurementExtractor(custom_patterns)
        self.temporal_extractor = TemporalExtractor()

    def extract(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extract_measurements: bool = True,
        extract_timestamp: bool = True
    ) -> ExtractedData:
        """
        Extract all structured data from text.

        Args:
            text: Raw text to process
            timestamp: Explicit timestamp (if not provided, tries to extract)
            metadata: Additional metadata to attach
            extract_measurements: Whether to extract numeric/categorical data
            extract_timestamp: Whether to extract temporal information

        Returns:
            ExtractedData with entities, measurements, and metadata
        """
        # Resolve entities
        entity_results = self.resolver.tag_text(text)

        # Extract measurements
        measurements = {}
        if extract_measurements:
            measurements = self.measurement_extractor.extract_all(text)

        # Extract or use provided timestamp
        final_timestamp = timestamp
        if extract_timestamp and not timestamp:
            final_timestamp = self.temporal_extractor.extract_date(text)

        if not final_timestamp:
            final_timestamp = datetime.now()  # Default to now

        return ExtractedData(
            text=text,
            entities=entity_results["entities"],
            measurements=measurements,
            timestamp=final_timestamp,
            metadata=metadata or {}
        )

    def extract_batch(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
        **kwargs
    ) -> List[ExtractedData]:
        """Extract data from multiple texts."""
        if timestamps is None:
            timestamps = [None] * len(texts)

        return [
            self.extract(text, timestamp=ts, **kwargs)
            for text, ts in zip(texts, timestamps)
        ]


# ==============================================================================
# Example Usage
# ==============================================================================

if __name__ == "__main__":
    from .resolver import EntityRegistry
    from pathlib import Path

    print("="*80)
    print("Entity Extractor Demo")
    print("="*80 + "\n")

    # Load beekeeping registry
    registry_path = Path("../domains/beekeeping/registry.json")
    registry = EntityRegistry.load(registry_path)

    # Create extractor
    resolver = EntityResolver(registry)
    extractor = EntityExtractor(resolver)

    # Test extraction
    test_notes = [
        "2024-10-12: Checked jodi - 8 frames of brood, solid pattern, very calm",
        "Applied thymol treatment to dennis and the split. Dosage: 50g each.",
        "Half door hive weak at 2.5 frames. Temperament aggressive today.",
    ]

    print("Extracting structured data from notes...\n")

    for i, note in enumerate(test_notes, 1):
        print(f"Note {i}: {note}")
        print()

        # Extract
        extracted = extractor.extract(note)

        # Show results
        print(f"  Entities ({len(extracted.entities)}):")
        for entity in extracted.entities:
            print(f"    - {entity['matched_text']} -> {entity['canonical_id']}")

        print(f"\n  Measurements ({len(extracted.measurements)}):")
        for key, value in extracted.measurements.items():
            print(f"    - {key}: {value}")

        print(f"\n  Timestamp: {extracted.timestamp}")

        # Show Qdrant payload
        payload = extracted.to_qdrant_payload()
        print(f"\n  Qdrant Payload Preview:")
        print(f"    primary_entity: {payload.get('primary_entity_id')}")
        print(f"    entity_ids: {payload.get('entity_ids')}")
        print(f"    has_measurements: {payload.get('has_measurements')}")

        if payload.get('frames_of_brood'):
            print(f"    frames_of_brood: {payload['frames_of_brood']}")
        if payload.get('temperament'):
            print(f"    temperament: {payload['temperament']}")

        print("\n" + "-"*80 + "\n")

    print("="*80)
    print("Extraction Complete!")
    print("="*80)
    print("\nThis pipeline automatically:")
    print("  1. Resolves entities to canonical IDs")
    print("  2. Extracts numeric measurements (frames, weight, etc.)")
    print("  3. Extracts categorical data (temperament, brood pattern)")
    print("  4. Parses timestamps")
    print("  5. Creates Qdrant-ready payloads")
    print("\nAll from plain text! Ready to plug into storage.")
