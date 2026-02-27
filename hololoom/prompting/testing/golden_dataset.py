"""Golden Dataset Manager for known good prompt-response pairs.

This module provides infrastructure for managing and using golden datasets
of known good prompt-response pairs for testing and validation.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from HoloLoom.prompting.testing.protocol import PromptTestCase


@dataclass
class GoldenPair:
    """A known good prompt-response pair with metadata."""

    prompt: str
    expected_output: str
    quality_score: float
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenPair":
        """Create from dictionary for JSON deserialization."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class GoldenDatasetManager:
    """Manages a collection of golden prompt-response pairs."""

    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the manager, optionally loading from a file.

        Args:
            dataset_path: Optional path to JSON file with golden pairs
        """
        self.pairs: Dict[str, GoldenPair] = {}
        if dataset_path:
            self.load(dataset_path)

    def add_pair(
        self,
        prompt: str,
        expected_output: str,
        quality_score: float,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoldenPair:
        """Add a new golden pair.

        Args:
            prompt: The prompt text
            expected_output: Expected output for this prompt
            quality_score: Quality score (0.0-1.0)
            tags: Optional tags for categorization
            metadata: Optional additional metadata

        Returns:
            The created GoldenPair

        Raises:
            ValueError: If validation fails
        """
        pair = GoldenPair(
            prompt=prompt,
            expected_output=expected_output,
            quality_score=quality_score,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._validate_pair(pair)
        self.pairs[prompt] = pair
        return pair

    def get_pairs(self, tags: Optional[List[str]] = None) -> List[GoldenPair]:
        """Get pairs, optionally filtered by tags.

        Args:
            tags: Optional list of tags to filter by (any match)

        Returns:
            List of matching GoldenPair objects
        """
        if not tags:
            return list(self.pairs.values())

        return [
            pair
            for pair in self.pairs.values()
            if any(tag in pair.tags for tag in tags)
        ]

    def get_test_cases(self, tags: Optional[List[str]] = None) -> List[PromptTestCase]:
        """Convert golden pairs to test cases.

        Args:
            tags: Optional tags to filter by

        Returns:
            List of PromptTestCase objects
        """
        pairs = self.get_pairs(tags)
        return [
            PromptTestCase(
                prompt=pair.prompt,
                expected_output=pair.expected_output,
                metadata={
                    "quality_score": pair.quality_score,
                    "tags": pair.tags,
                    "created_at": pair.created_at.isoformat(),
                    **pair.metadata,
                },
            )
            for pair in pairs
        ]

    def save(self, path: str) -> None:
        """Save golden pairs to JSON file.

        Args:
            path: Path to save to
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            prompt: pair.to_dict() for prompt, pair in self.pairs.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load golden pairs from JSON file.

        Args:
            path: Path to load from
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.pairs = {
            prompt: GoldenPair.from_dict(pair_data)
            for prompt, pair_data in data.items()
        }

    def remove_pair(self, prompt: str) -> bool:
        """Remove a pair by prompt text.

        Args:
            prompt: The prompt to remove

        Returns:
            True if removed, False if not found
        """
        if prompt in self.pairs:
            del self.pairs[prompt]
            return True
        return False

    def update_pair(self, prompt: str, **kwargs) -> Optional[GoldenPair]:
        """Update an existing pair.

        Args:
            prompt: The prompt to update
            **kwargs: Fields to update

        Returns:
            Updated GoldenPair, or None if not found
        """
        if prompt not in self.pairs:
            return None

        pair = self.pairs[prompt]
        for key, value in kwargs.items():
            if hasattr(pair, key):
                setattr(pair, key, value)

        self._validate_pair(pair)
        return pair

    def _validate_pair(self, pair: GoldenPair) -> None:
        """Validate a pair before adding.

        Args:
            pair: The pair to validate

        Raises:
            ValueError: If validation fails
        """
        if not pair.prompt or not pair.prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not pair.expected_output or not pair.expected_output.strip():
            raise ValueError("Expected output cannot be empty")

        if not 0.0 <= pair.quality_score <= 1.0:
            raise ValueError(
                f"Quality score must be between 0.0 and 1.0, got {pair.quality_score}"
            )


def create_golden_dataset(
    path: Optional[str] = None,
) -> GoldenDatasetManager:
    """Factory function to create a golden dataset manager.

    Args:
        path: Optional path to load from

    Returns:
        GoldenDatasetManager instance
    """
    return GoldenDatasetManager(path)
