"""
SpinningWheel Protocol
======================

Standardized interface for all HoloLoom data spinners.

Philosophy:
    "If you need to configure it, we failed."

All spinners implement SpinnerProtocol to ensure:
- Consistent API across all data sources
- Graceful degradation when dependencies unavailable
- Standardized importance scoring
- Streaming and incremental ingestion support
- Checkpointing for resumable operations

Author: Claude Code
Date: November 2025
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from hololoom.protocols.types import MemoryShard

# ============================================================================
# Spinner Status and Metadata
# ============================================================================

class SpinnerStatus(Enum):
    """Spinner operational status."""
    AVAILABLE = "available"           # All dependencies present, ready to spin
    DEGRADED = "degraded"             # Missing optional features, basic functionality works
    UNAVAILABLE = "unavailable"       # Missing required dependencies
    ERROR = "error"                   # Runtime error occurred


@dataclass
class SpinnerCapabilities:
    """
    Capabilities of a spinner.

    Describes what features are available based on installed dependencies.
    """
    # Core capabilities
    basic_processing: bool = True      # Can process basic inputs
    streaming: bool = False            # Supports streaming ingestion
    incremental: bool = False          # Supports incremental updates
    batch_processing: bool = False     # Supports batch operations

    # Advanced features
    importance_scoring: bool = False   # Can score importance
    entity_extraction: bool = False    # Can extract entities
    motif_extraction: bool = False     # Can extract motifs/topics
    embeddings: bool = False           # Can generate embeddings

    # Optional enhancements
    multimodal: bool = False           # Supports multi-modal inputs
    cross_modal_fusion: bool = False   # Can fuse multiple modalities

    # Metadata
    max_input_size: int | None = None  # Max bytes per input (None = unlimited)
    supported_formats: list[str] = field(default_factory=list)  # File extensions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'basic_processing': self.basic_processing,
            'streaming': self.streaming,
            'incremental': self.incremental,
            'batch_processing': self.batch_processing,
            'importance_scoring': self.importance_scoring,
            'entity_extraction': self.entity_extraction,
            'motif_extraction': self.motif_extraction,
            'embeddings': self.embeddings,
            'multimodal': self.multimodal,
            'cross_modal_fusion': self.cross_modal_fusion,
            'max_input_size': self.max_input_size,
            'supported_formats': self.supported_formats
        }


@dataclass
class SpinResult:
    """
    Result of a spin operation.

    Contains shards plus metadata about the operation.
    """
    shards: list[MemoryShard]

    # Operation metadata
    success: bool = True
    error_message: str | None = None

    # Performance metrics
    processing_time_ms: float = 0.0
    input_size_bytes: int = 0

    # Statistics
    shard_count: int = 0
    entity_count: int = 0
    motif_count: int = 0

    # Quality metrics
    avg_importance: float = 0.5
    avg_confidence: float = 1.0

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate statistics from shards."""
        if self.shards:
            self.shard_count = len(self.shards)
            self.entity_count = sum(len(s.entities) for s in self.shards)
            self.motif_count = sum(len(s.motifs) for s in self.shards)

            # Average importance (if available in metadata)
            importances = [
                s.metadata.get('importance_score', 0.5)
                for s in self.shards
            ]
            self.avg_importance = sum(importances) / len(importances) if importances else 0.5

            # Average confidence
            confidences = [
                s.metadata.get('confidence', 1.0)
                for s in self.shards
            ]
            self.avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'shard_count': self.shard_count,
            'entity_count': self.entity_count,
            'motif_count': self.motif_count,
            'avg_importance': self.avg_importance,
            'avg_confidence': self.avg_confidence,
            'processing_time_ms': self.processing_time_ms,
            'input_size_bytes': self.input_size_bytes,
            'warnings': self.warnings
        }


# ============================================================================
# Importance Scoring
# ============================================================================

@dataclass
class ImportanceSignals:
    """
    Standardized importance signals across all spinners.

    Each signal is 0.0-1.0, with recommended weights.
    Final importance = weighted sum of signals.
    """
    # Content quality (0.0-1.0)
    length_score: float = 0.0           # Longer = more substantive (weight: 0.15)
    technical_score: float = 0.0        # Domain-specific terms (weight: 0.20)
    structural_score: float = 0.0       # Well-formatted, organized (weight: 0.10)

    # Source authority (0.0-1.0)
    authority_score: float = 0.5        # Source credibility (weight: 0.20)
    recency_score: float = 0.5          # Time decay (weight: 0.10)

    # Engagement (0.0-1.0)
    engagement_score: float = 0.0       # Reactions, shares, replies (weight: 0.15)
    reference_score: float = 0.0        # Citations, backlinks (weight: 0.10)

    # Penalties (-1.0 to 0.0)
    noise_penalty: float = 0.0          # Greetings, spam, duplicates (weight: 1.0)

    # Custom signals (spinner-specific)
    custom_signals: dict[str, float] = field(default_factory=dict)

    def compute_total(
        self,
        weights: dict[str, float] | None = None
    ) -> float:
        """
        Compute total importance score.

        Args:
            weights: Optional custom weights (None = use defaults)

        Returns:
            Total importance score (0.0-1.0)
        """
        # Default weights
        default_weights = {
            'length': 0.15,
            'technical': 0.20,
            'structural': 0.10,
            'authority': 0.20,
            'recency': 0.10,
            'engagement': 0.15,
            'reference': 0.10
        }

        w = weights or default_weights

        # Weighted sum
        total = (
            self.length_score * w.get('length', 0.15) +
            self.technical_score * w.get('technical', 0.20) +
            self.structural_score * w.get('structural', 0.10) +
            self.authority_score * w.get('authority', 0.20) +
            self.recency_score * w.get('recency', 0.10) +
            self.engagement_score * w.get('engagement', 0.15) +
            self.reference_score * w.get('reference', 0.10) +
            self.noise_penalty  # Penalty is negative
        )

        # Add custom signals (equal weight for each)
        if self.custom_signals:
            custom_weight = 0.1 / len(self.custom_signals)
            total += sum(v * custom_weight for v in self.custom_signals.values())

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, total))

    def explain(self) -> str:
        """
        Generate human-readable explanation of importance.

        Returns:
            Explanation string (e.g., "high technical + good authority")
        """
        factors = []

        if self.length_score > 0.7:
            factors.append("substantive length")
        if self.technical_score > 0.7:
            factors.append("high technical content")
        if self.authority_score > 0.7:
            factors.append("authoritative source")
        if self.engagement_score > 0.7:
            factors.append("high engagement")
        if self.noise_penalty < -0.2:
            factors.append("noise detected")

        if not factors:
            return "moderate importance"

        return " + ".join(factors)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            'length_score': self.length_score,
            'technical_score': self.technical_score,
            'structural_score': self.structural_score,
            'authority_score': self.authority_score,
            'recency_score': self.recency_score,
            'engagement_score': self.engagement_score,
            'reference_score': self.reference_score,
            'noise_penalty': self.noise_penalty,
            **self.custom_signals
        }


@dataclass
class ImportanceScore:
    """
    Complete importance assessment for a data item.
    """
    score: float  # 0.0 to 1.0
    signals: ImportanceSignals
    reason: str  # Human-readable explanation

    def __post_init__(self):
        """Validate score."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Importance score must be in [0.0, 1.0], got {self.score}")


# ============================================================================
# Checkpointing
# ============================================================================

@dataclass
class SpinnerCheckpoint:
    """
    Checkpoint for resumable spinner operations.

    Tracks progress through large data sources (repos, email archives, etc.).
    """
    spinner_name: str
    source_id: str  # Unique identifier for the data source

    # Progress tracking
    last_processed_id: str | None = None  # Last item processed
    last_processed_timestamp: float | None = None  # Unix timestamp
    items_processed: int = 0
    items_total: int | None = None

    # State
    state: dict[str, Any] = field(default_factory=dict)  # Spinner-specific state

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def progress_percentage(self) -> float | None:
        """Calculate progress percentage (None if total unknown)."""
        if self.items_total and self.items_total > 0:
            return (self.items_processed / self.items_total) * 100
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'spinner_name': self.spinner_name,
            'source_id': self.source_id,
            'last_processed_id': self.last_processed_id,
            'last_processed_timestamp': self.last_processed_timestamp,
            'items_processed': self.items_processed,
            'items_total': self.items_total,
            'state': self.state,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SpinnerCheckpoint':
        """Deserialize from dictionary."""
        return cls(
            spinner_name=data['spinner_name'],
            source_id=data['source_id'],
            last_processed_id=data.get('last_processed_id'),
            last_processed_timestamp=data.get('last_processed_timestamp'),
            items_processed=data.get('items_processed', 0),
            items_total=data.get('items_total'),
            state=data.get('state', {}),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


# ============================================================================
# Spinner Protocol
# ============================================================================

class SpinnerProtocol(Protocol):
    """
    Protocol that all HoloLoom spinners must implement.

    Ensures consistent API across all data sources:
    - Text, Image, Audio (existing)
    - Git, Email, Slack, PDF, Calendar, etc. (future)

    Minimal required methods:
        - spin(): Process input → MemoryShards
        - get_name(): Return spinner name
        - get_capabilities(): Return capabilities
        - is_available(): Check if dependencies present

    Optional methods:
        - spin_stream(): Streaming ingestion
        - spin_incremental(): Incremental updates
        - score_importance(): Custom importance scoring
    """

    def get_name(self) -> str:
        """
        Get spinner name.

        Returns:
            Unique name (e.g., 'git', 'email', 'chat_history')
        """
        ...

    def get_capabilities(self) -> SpinnerCapabilities:
        """
        Get spinner capabilities.

        Returns:
            SpinnerCapabilities describing available features
        """
        ...

    def is_available(self) -> bool:
        """
        Check if spinner is available (dependencies installed).

        Returns:
            True if spinner can be used, False otherwise
        """
        ...

    def get_status(self) -> SpinnerStatus:
        """
        Get current operational status.

        Returns:
            SpinnerStatus enum
        """
        ...

    async def spin(
        self,
        source: Any,
        **kwargs
    ) -> SpinResult:
        """
        Process input and return MemoryShards.

        Args:
            source: Input data (varies by spinner type)
                - Git: repository path
                - Email: IMAP credentials or mbox file
                - Chat: ConversationManager instance
                - Text: string or file path
            **kwargs: Spinner-specific options

        Returns:
            SpinResult with shards and metadata

        Example:
            >>> result = await spinner.spin("/path/to/repo")
            >>> print(f"Created {result.shard_count} shards")
        """
        ...

    # Optional: Streaming support
    async def spin_stream(
        self,
        source: Any,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Process input incrementally, yielding shards as available.

        Optional method for large data sources (Git repos, email archives).

        Args:
            source: Input data
            **kwargs: Spinner-specific options

        Yields:
            MemoryShards one at a time

        Example:
            >>> async for shard in spinner.spin_stream("/huge/repo"):
            ...     await memory.add_shard(shard)
        """
        raise NotImplementedError(f"{self.get_name()} doesn't support streaming")

    # Optional: Incremental updates
    async def spin_incremental(
        self,
        source: Any,
        checkpoint: SpinnerCheckpoint | None = None,
        **kwargs
    ) -> SpinResult:
        """
        Process only new data since last checkpoint.

        Optional method for sources that update over time (Git, email, Slack).

        Args:
            source: Input data
            checkpoint: Previous checkpoint (None = start from beginning)
            **kwargs: Spinner-specific options

        Returns:
            SpinResult with only new shards

        Example:
            >>> checkpoint = load_checkpoint("git_repo_xyz")
            >>> result = await spinner.spin_incremental(
            ...     "/path/to/repo",
            ...     checkpoint=checkpoint
            ... )
            >>> print(f"Processed {result.shard_count} new commits")
        """
        raise NotImplementedError(f"{self.get_name()} doesn't support incremental updates")

    # Optional: Custom importance scoring
    def score_importance(self, data: Any) -> ImportanceScore:
        """
        Score importance of a data item.

        Optional method for custom importance heuristics.

        Args:
            data: Data item to score (varies by spinner)

        Returns:
            ImportanceScore with score and explanation

        Example:
            >>> score = spinner.score_importance(commit)
            >>> if score.score > 0.7:
            ...     print(f"Important: {score.reason}")
        """
        raise NotImplementedError(f"{self.get_name()} doesn't support importance scoring")


# ============================================================================
# Base Spinner Implementation
# ============================================================================

class BaseSpinner(ABC):
    """
    Abstract base class for spinners.

    Provides common functionality:
    - Dependency checking
    - Error handling
    - Performance tracking
    - Checkpointing

    Subclasses must implement:
    - _spin_impl(): Core spinning logic
    """

    def __init__(
        self,
        name: str,
        importance_threshold: float = 0.3,
        checkpoint_dir: Path | None = None
    ):
        """
        Initialize base spinner.

        Args:
            name: Spinner name (e.g., 'git', 'email')
            importance_threshold: Min importance to include (0.0-1.0)
            checkpoint_dir: Directory for saving checkpoints (None = no checkpointing)
        """
        self.name = name
        self.importance_threshold = importance_threshold
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Required Protocol Methods
    # ========================================================================

    def get_name(self) -> str:
        """Get spinner name."""
        return self.name

    @abstractmethod
    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities (must be implemented by subclass)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if dependencies available (must be implemented by subclass)."""
        ...

    def get_status(self) -> SpinnerStatus:
        """Get operational status."""
        if not self.is_available():
            return SpinnerStatus.UNAVAILABLE

        capabilities = self.get_capabilities()
        if capabilities.basic_processing:
            if capabilities.entity_extraction and capabilities.importance_scoring:
                return SpinnerStatus.AVAILABLE
            else:
                return SpinnerStatus.DEGRADED
        else:
            return SpinnerStatus.UNAVAILABLE

    async def spin(self, source: Any, **kwargs) -> SpinResult:
        """
        Process input with error handling and performance tracking.

        Delegates to _spin_impl() for actual processing.
        """
        start_time = time.time()

        try:
            # Check availability
            if not self.is_available():
                return SpinResult(
                    shards=[],
                    success=False,
                    error_message=f"{self.name} spinner unavailable (missing dependencies)"
                )

            # Delegate to implementation
            shards = await self._spin_impl(source, **kwargs)

            # Filter by importance if threshold set
            if self.importance_threshold > 0:
                filtered_shards = []
                for shard in shards:
                    importance = shard.metadata.get('importance_score', 0.5)
                    if importance >= self.importance_threshold:
                        filtered_shards.append(shard)

                filtered_count = len(shards) - len(filtered_shards)
                shards = filtered_shards
            else:
                filtered_count = 0

            # Calculate metrics
            processing_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = SpinResult(
                shards=shards,
                success=True,
                processing_time_ms=processing_time_ms
            )

            if filtered_count > 0:
                result.warnings.append(
                    f"Filtered {filtered_count} shards below importance threshold {self.importance_threshold}"
                )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return SpinResult(
                shards=[],
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )

    # ========================================================================
    # Abstract Methods (Subclasses Must Implement)
    # ========================================================================

    @abstractmethod
    async def _spin_impl(self, source: Any, **kwargs) -> list[MemoryShard]:
        """
        Core spinning implementation.

        Subclasses must implement this method.

        Args:
            source: Input data
            **kwargs: Spinner-specific options

        Returns:
            List of MemoryShards
        """
        ...

    # ========================================================================
    # Optional Methods (Subclasses Can Override)
    # ========================================================================

    async def spin_stream(self, source: Any, **kwargs) -> AsyncIterator[MemoryShard]:
        """Default: Not implemented."""
        raise NotImplementedError(f"{self.name} doesn't support streaming")
        yield  # Make this a generator (unreachable)

    async def spin_incremental(
        self,
        source: Any,
        checkpoint: SpinnerCheckpoint | None = None,
        **kwargs
    ) -> SpinResult:
        """Default: Not implemented."""
        raise NotImplementedError(f"{self.name} doesn't support incremental updates")

    def score_importance(self, data: Any) -> ImportanceScore:
        """Default: Return moderate importance."""
        return ImportanceScore(
            score=0.5,
            signals=ImportanceSignals(),
            reason="default importance (no scoring implemented)"
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def load_checkpoint(self, source_id: str) -> SpinnerCheckpoint | None:
        """Load checkpoint for a source."""
        if not self.checkpoint_dir:
            return None

        checkpoint_path = self.checkpoint_dir / f"{self.name}_{source_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            import json
            with open(checkpoint_path) as f:
                data = json.load(f)
            return SpinnerCheckpoint.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def save_checkpoint(self, checkpoint: SpinnerCheckpoint):
        """Save checkpoint for resumption."""
        if not self.checkpoint_dir:
            return

        checkpoint_path = self.checkpoint_dir / f"{self.name}_{checkpoint.source_id}.json"

        try:
            import json
            checkpoint.updated_at = time.time()
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def _create_shard(
        self,
        id_suffix: str,
        text: str,
        episode: str,
        entities: list[str],
        motifs: list[str],
        metadata: dict[str, Any]
    ) -> MemoryShard:
        """
        Helper to create MemoryShard with standard metadata.

        Args:
            id_suffix: Unique suffix for shard ID
            text: Shard text content
            episode: Episode identifier
            entities: Extracted entities
            motifs: Extracted motifs
            metadata: Additional metadata

        Returns:
            MemoryShard with standard fields
        """
        # Add spinner metadata
        metadata['spinner'] = self.name

        return MemoryShard(
            id=f"{self.name}_{id_suffix}",
            text=text,
            episode=episode,
            entities=entities[:20],  # Limit to 20
            motifs=motifs[:10],  # Limit to 10
            metadata=metadata
        )


# ============================================================================
# Utility Functions
# ============================================================================

def create_spinner_registry() -> dict[str, type]:
    """
    Get registry of all available spinners.

    Returns:
        Dict mapping spinner name to class

    Example:
        >>> registry = create_spinner_registry()
        >>> GitSpinner = registry.get('git')
        >>> if GitSpinner:
        ...     spinner = GitSpinner()
    """
    registry = {}

    # Dynamically import all spinners
    try:
        from .chat_history import ChatHistorySpinner
        registry['chat_history'] = ChatHistorySpinner
    except ImportError:
        pass

    try:
        from .multimodal_spinner import MultiModalSpinner
        registry['multimodal'] = MultiModalSpinner
    except ImportError:
        pass

    # Future spinners will be added here automatically
    # try:
    #     from .git_spinner import GitSpinner
    #     registry['git'] = GitSpinner
    # except ImportError:
    #     pass

    return registry


def get_available_spinners() -> list[str]:
    """
    Get list of available spinner names.

    Returns:
        List of spinner names that are currently available

    Example:
        >>> available = get_available_spinners()
        >>> print(f"Available spinners: {', '.join(available)}")
    """
    registry = create_spinner_registry()
    available = []

    for name, spinner_class in registry.items():
        try:
            # Instantiate and check availability
            spinner = spinner_class()
            if spinner.is_available():
                available.append(name)
        except Exception:
            pass

    return available
