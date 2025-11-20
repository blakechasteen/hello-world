"""
SpinningWheel Utilities
=======================

Reusable utilities for all spinners:
- Checkpoint management (resumable operations)
- Streaming helpers (incremental processing)
- Batch processing utilities
- Deduplication
- Error handling

Author: Claude Code
Date: November 2025
"""

import asyncio
import hashlib
import json
from typing import List, Dict, Any, Optional, AsyncIterator, Callable, Set
from pathlib import Path
from dataclasses import dataclass
import time

from HoloLoom.Documentation.types import MemoryShard
from .protocol import SpinnerCheckpoint


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """
    Manage checkpoints for resumable spinner operations.

    Automatically saves/loads checkpoint state for large data sources.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        spinner_name: str,
        source_id: str
    ) -> Optional[SpinnerCheckpoint]:
        """
        Load checkpoint for a spinner + source.

        Args:
            spinner_name: Name of spinner
            source_id: Unique source identifier

        Returns:
            SpinnerCheckpoint if exists, None otherwise
        """
        checkpoint_path = self._get_path(spinner_name, source_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return SpinnerCheckpoint.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: SpinnerCheckpoint):
        """
        Save checkpoint.

        Args:
            checkpoint: Checkpoint to save
        """
        checkpoint_path = self._get_path(
            checkpoint.spinner_name,
            checkpoint.source_id
        )

        try:
            checkpoint.updated_at = time.time()
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def delete(self, spinner_name: str, source_id: str):
        """Delete checkpoint."""
        checkpoint_path = self._get_path(spinner_name, source_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def list_checkpoints(
        self,
        spinner_name: Optional[str] = None
    ) -> List[SpinnerCheckpoint]:
        """
        List all checkpoints.

        Args:
            spinner_name: Filter by spinner (None = all)

        Returns:
            List of SpinnerCheckpoints
        """
        pattern = f"{spinner_name}_*" if spinner_name else "*"
        checkpoint_files = self.checkpoint_dir.glob(f"{pattern}.json")

        checkpoints = []
        for path in checkpoint_files:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                checkpoints.append(SpinnerCheckpoint.from_dict(data))
            except Exception:
                pass

        return checkpoints

    def _get_path(self, spinner_name: str, source_id: str) -> Path:
        """Get checkpoint file path."""
        # Hash source_id to create safe filename
        source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
        filename = f"{spinner_name}_{source_hash}.json"
        return self.checkpoint_dir / filename


# ============================================================================
# Streaming Utilities
# ============================================================================

class StreamBuffer:
    """
    Buffer for streaming shard ingestion.

    Batches shards before ingesting to memory for efficiency.
    """

    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        flush_callback: Optional[Callable] = None
    ):
        """
        Initialize stream buffer.

        Args:
            batch_size: Number of shards to batch before flushing
            flush_interval: Seconds to wait before auto-flush
            flush_callback: Async function to call on flush (receives List[MemoryShard])
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.flush_callback = flush_callback

        self.buffer: List[MemoryShard] = []
        self.last_flush_time = time.time()

    async def add(self, shard: MemoryShard):
        """
        Add shard to buffer.

        Auto-flushes if batch size or interval exceeded.
        """
        self.buffer.append(shard)

        # Check flush conditions
        should_flush = (
            len(self.buffer) >= self.batch_size or
            (time.time() - self.last_flush_time) >= self.flush_interval
        )

        if should_flush:
            await self.flush()

    async def flush(self):
        """Flush buffer to callback."""
        if not self.buffer:
            return

        if self.flush_callback:
            try:
                await self.flush_callback(self.buffer)
            except Exception as e:
                print(f"Warning: Flush callback failed: {e}")

        # Clear buffer
        self.buffer.clear()
        self.last_flush_time = time.time()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (auto-flush)."""
        await self.flush()


async def stream_with_buffer(
    stream: AsyncIterator[MemoryShard],
    batch_size: int = 10,
    callback: Optional[Callable] = None
) -> AsyncIterator[MemoryShard]:
    """
    Wrap streaming shard generation with batched callback.

    Args:
        stream: AsyncIterator yielding MemoryShards
        batch_size: Batch size for callbacks
        callback: Async function to call on each batch

    Yields:
        MemoryShards (same as input stream)

    Example:
        >>> async def ingest_batch(shards):
        ...     await memory.add_shards(shards)
        >>>
        >>> async for shard in stream_with_buffer(spinner.spin_stream(source), callback=ingest_batch):
        ...     print(f"Processed: {shard.id}")
    """
    async with StreamBuffer(batch_size=batch_size, flush_callback=callback) as buffer:
        async for shard in stream:
            await buffer.add(shard)
            yield shard


# ============================================================================
# Batch Processing
# ============================================================================

class BatchProcessor:
    """
    Process multiple items in parallel with concurrency limits.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        retry_on_error: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent tasks
            retry_on_error: Retry failed items
            max_retries: Maximum retry attempts
        """
        self.max_concurrent = max_concurrent
        self.retry_on_error = retry_on_error
        self.max_retries = max_retries

    async def process(
        self,
        items: List[Any],
        processor: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """
        Process items in parallel.

        Args:
            items: Items to process
            processor: Async function to process each item
            progress_callback: Called after each item (item, result, index, total)

        Returns:
            List of results (same order as items)

        Example:
            >>> processor = BatchProcessor(max_concurrent=10)
            >>> results = await processor.process(
            ...     items=file_paths,
            ...     processor=lambda path: spin(path)
            ... )
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = [None] * len(items)

        async def process_one(item: Any, index: int):
            """Process single item with retry logic."""
            retries = 0

            while retries <= self.max_retries:
                async with semaphore:
                    try:
                        result = await processor(item)
                        results[index] = result

                        # Progress callback
                        if progress_callback:
                            await progress_callback(item, result, index, len(items))

                        return

                    except Exception as e:
                        retries += 1

                        if retries > self.max_retries or not self.retry_on_error:
                            print(f"Error processing item {index}: {e}")
                            results[index] = None
                            return

                        # Exponential backoff
                        await asyncio.sleep(2 ** retries)

        # Process all items
        tasks = [process_one(item, i) for i, item in enumerate(items)]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results


# ============================================================================
# Deduplication
# ============================================================================

class ShardDeduplicator:
    """
    Detect and filter duplicate shards.

    Uses content hashing to identify duplicates.
    """

    def __init__(self, hash_method: str = "md5"):
        """
        Initialize deduplicator.

        Args:
            hash_method: Hash algorithm (md5, sha1, sha256)
        """
        self.hash_method = hash_method
        self.seen_hashes: Set[str] = set()

    def hash_shard(self, shard: MemoryShard) -> str:
        """
        Generate hash for shard.

        Args:
            shard: MemoryShard to hash

        Returns:
            Hash string
        """
        # Hash based on text content
        content = shard.text.encode('utf-8')

        if self.hash_method == "md5":
            return hashlib.md5(content).hexdigest()
        elif self.hash_method == "sha1":
            return hashlib.sha1(content).hexdigest()
        elif self.hash_method == "sha256":
            return hashlib.sha256(content).hexdigest()
        else:
            raise ValueError(f"Unknown hash method: {self.hash_method}")

    def is_duplicate(self, shard: MemoryShard) -> bool:
        """
        Check if shard is duplicate.

        Args:
            shard: Shard to check

        Returns:
            True if duplicate, False otherwise
        """
        shard_hash = self.hash_shard(shard)
        return shard_hash in self.seen_hashes

    def mark_seen(self, shard: MemoryShard):
        """Mark shard as seen."""
        shard_hash = self.hash_shard(shard)
        self.seen_hashes.add(shard_hash)

    def filter_duplicates(
        self,
        shards: List[MemoryShard],
        mark_seen: bool = True
    ) -> List[MemoryShard]:
        """
        Filter duplicates from shard list.

        Args:
            shards: List of shards
            mark_seen: Mark filtered shards as seen

        Returns:
            Deduplicated list
        """
        unique_shards = []

        for shard in shards:
            if not self.is_duplicate(shard):
                unique_shards.append(shard)

                if mark_seen:
                    self.mark_seen(shard)

        return unique_shards

    def reset(self):
        """Clear seen hashes."""
        self.seen_hashes.clear()


# ============================================================================
# Error Handling
# ============================================================================

class ErrorHandler:
    """
    Centralized error handling for spinners.
    """

    def __init__(
        self,
        log_errors: bool = True,
        raise_on_critical: bool = False
    ):
        """
        Initialize error handler.

        Args:
            log_errors: Print errors to console
            raise_on_critical: Re-raise critical errors
        """
        self.log_errors = log_errors
        self.raise_on_critical = raise_on_critical
        self.error_log: List[Dict[str, Any]] = []

    def handle(
        self,
        error: Exception,
        context: str,
        critical: bool = False
    ) -> Optional[MemoryShard]:
        """
        Handle error.

        Args:
            error: Exception that occurred
            context: Context string (e.g., "processing file X")
            critical: Whether error is critical

        Returns:
            Error shard (for logging) or None

        Raises:
            Exception: If critical and raise_on_critical=True
        """
        # Log error
        error_info = {
            'error': str(error),
            'type': type(error).__name__,
            'context': context,
            'critical': critical,
            'timestamp': time.time()
        }
        self.error_log.append(error_info)

        # Print if enabled
        if self.log_errors:
            severity = "CRITICAL" if critical else "WARNING"
            print(f"[{severity}] {context}: {error}")

        # Re-raise if critical
        if critical and self.raise_on_critical:
            raise error

        # Create error shard for memory
        error_shard = MemoryShard(
            id=f"error_{hashlib.md5(context.encode()).hexdigest()[:8]}",
            text=f"Error: {context}\n\n{str(error)}",
            episode="errors",
            entities=[],
            motifs=["error", "processing_failure"],
            metadata={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'critical': critical,
                'timestamp': time.time()
            }
        )

        return error_shard

    def get_errors(
        self,
        critical_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get error log.

        Args:
            critical_only: Only return critical errors

        Returns:
            List of error dicts
        """
        if critical_only:
            return [e for e in self.error_log if e['critical']]
        return self.error_log

    def clear(self):
        """Clear error log."""
        self.error_log.clear()


# ============================================================================
# Progress Tracking
# ============================================================================

@dataclass
class ProgressTracker:
    """Track progress for long-running operations."""

    total: int
    current: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        """Initialize start time."""
        if self.start_time == 0.0:
            self.start_time = time.time()

    def increment(self, amount: int = 1):
        """Increment progress."""
        self.current = min(self.total, self.current + amount)

    def percentage(self) -> float:
        """Get progress percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def estimated_remaining(self) -> Optional[float]:
        """
        Estimate remaining time in seconds.

        Returns:
            Estimated seconds or None if can't estimate
        """
        if self.current == 0:
            return None

        elapsed = self.elapsed_time()
        rate = self.current / elapsed  # Items per second
        remaining_items = self.total - self.current

        return remaining_items / rate if rate > 0 else None

    def format_status(self) -> str:
        """
        Format status string.

        Returns:
            Human-readable status (e.g., "450/1000 (45.0%) - 2m remaining")
        """
        percentage = self.percentage()
        status = f"{self.current}/{self.total} ({percentage:.1f}%)"

        remaining = self.estimated_remaining()
        if remaining:
            if remaining < 60:
                time_str = f"{remaining:.0f}s"
            elif remaining < 3600:
                time_str = f"{remaining/60:.0f}m"
            else:
                time_str = f"{remaining/3600:.1f}h"

            status += f" - {time_str} remaining"

        return status

    def print_progress(self, prefix: str = "Progress"):
        """Print progress bar to console."""
        percentage = self.percentage()
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        status = self.format_status()
        print(f"\r{prefix}: {bar} {status}", end='', flush=True)

        # New line when complete
        if self.current >= self.total:
            print()


# ============================================================================
# Convenience Functions
# ============================================================================

async def process_with_progress(
    items: List[Any],
    processor: Callable,
    description: str = "Processing",
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items with progress tracking.

    Args:
        items: Items to process
        processor: Async function to process each item
        description: Description for progress bar
        max_concurrent: Max concurrent tasks

    Returns:
        List of results

    Example:
        >>> results = await process_with_progress(
        ...     file_paths,
        ...     processor=lambda path: spin(path),
        ...     description="Spinning files"
        ... )
    """
    tracker = ProgressTracker(total=len(items))

    async def process_with_tracking(item, result, index, total):
        """Callback to update progress."""
        tracker.increment()
        tracker.print_progress(description)

    batch_processor = BatchProcessor(max_concurrent=max_concurrent)

    results = await batch_processor.process(
        items,
        processor,
        progress_callback=process_with_tracking
    )

    return results


def create_source_id(source: Any) -> str:
    """
    Create unique source ID for checkpointing.

    Args:
        source: Source data (path, URL, dict, etc.)

    Returns:
        Unique source ID string

    Example:
        >>> source_id = create_source_id("/path/to/repo")
        >>> checkpoint = SpinnerCheckpoint(spinner_name="git", source_id=source_id)
    """
    # Convert source to string
    if isinstance(source, Path):
        source_str = str(source.absolute())
    elif isinstance(source, dict):
        # Hash dict
        source_str = json.dumps(source, sort_keys=True)
    else:
        source_str = str(source)

    # Hash to create short ID
    return hashlib.md5(source_str.encode()).hexdigest()[:16]
