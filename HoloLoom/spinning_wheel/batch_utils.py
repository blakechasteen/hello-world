#!/usr/bin/env python3
"""
Batch Ingestion Utilities
==========================

Helper utilities for bulk data ingestion with parallel processing,
progress tracking, and error recovery.

Features:
- Parallel URL ingestion
- Progress tracking with ETA
- Automatic retry on transient failures
- Error recovery and logging
- Resource usage monitoring
- Result aggregation

Usage:
    from HoloLoom.spinning_wheel.batch_utils import (
        batch_ingest_urls,
        batch_ingest_files,
        BatchConfig
    )

    # Ingest multiple URLs
    urls = ['https://example.com/1', 'https://example.com/2', ...]
    results = await batch_ingest_urls(urls, max_workers=5)

    # Ingest multiple files
    files = ['doc1.txt', 'doc2.md', 'code.py']
    results = await batch_ingest_files(files)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch ingestion."""
    max_workers: int = 5  # Parallel workers
    retry_attempts: int = 3  # Retries on failure
    retry_delay: float = 1.0  # Seconds between retries
    timeout_per_item: float = 30.0  # Seconds per item
    show_progress: bool = True  # Print progress updates
    stop_on_error: bool = False  # Stop entire batch on first error
    log_errors: bool = True  # Log errors to file


@dataclass
class BatchResult:
    """Result from batch ingestion."""
    total_items: int
    successful: int
    failed: int
    errors: Dict[str, str] = field(default_factory=dict)  # item -> error message
    shards_created: int = 0
    memories_stored: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    def __str__(self):
        return (
            f"BatchResult(total={self.total_items}, "
            f"success={self.successful}, failed={self.failed}, "
            f"shards={self.shards_created}, duration={self.duration_seconds:.1f}s)"
        )


class ProgressTracker:
    """Track and display batch ingestion progress."""

    def __init__(self, total_items: int, show_progress: bool = True):
        self.total_items = total_items
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.show_progress = show_progress
        self.last_update = 0

    def update(self, success: bool = True):
        """Update progress counter."""
        self.completed += 1
        if not success:
            self.failed += 1

        # Print progress every second
        if self.show_progress and (time.time() - self.last_update) > 1.0:
            self.print_progress()
            self.last_update = time.time()

    def print_progress(self):
        """Print current progress."""
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total_items - self.completed) / rate if rate > 0 else 0

        print(
            f"\r[{self.completed}/{self.total_items}] "
            f"Success: {self.completed - self.failed}, Failed: {self.failed} | "
            f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s",
            end="",
            flush=True
        )

    def finish(self):
        """Print final summary."""
        if self.show_progress:
            print()  # Newline after progress
            elapsed = time.time() - self.start_time
            print(f"\n✓ Completed {self.completed}/{self.total_items} in {elapsed:.1f}s")
            if self.failed > 0:
                print(f"✗ {self.failed} items failed")


async def batch_ingest_urls(
    urls: List[str],
    config: Optional[BatchConfig] = None,
    tags: Optional[List[str]] = None,
    store_in_memory: bool = True,
    user_id: Optional[str] = None
) -> BatchResult:
    """
    Ingest multiple URLs in parallel.

    Args:
        urls: List of URLs to ingest
        config: Batch configuration
        tags: Tags to add to all ingested content
        store_in_memory: Store shards in memory backend
        user_id: User ID for memory storage

    Returns:
        BatchResult with statistics

    Example:
        urls = [
            'https://example.com/page1',
            'https://example.com/page2',
            'https://example.com/page3',
        ]
        result = await batch_ingest_urls(urls, max_workers=5)
        print(f"Ingested {result.successful} pages successfully")
    """
    if config is None:
        config = BatchConfig()

    if tags is None:
        tags = []

    # Import here to avoid circular dependencies
    from .website import spin_webpage

    if store_in_memory:
        from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories
        memory = await create_unified_memory(user_id=user_id or "batch")

    # Initialize result tracking
    result = BatchResult(total_items=len(urls))
    tracker = ProgressTracker(len(urls), config.show_progress)
    errors = {}
    total_shards = 0
    total_memories = 0

    # Semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(config.max_workers)

    async def process_url(url: str) -> Optional[int]:
        """Process a single URL with retry logic."""
        async with semaphore:
            for attempt in range(config.retry_attempts):
                try:
                    # Ingest webpage
                    shards = await asyncio.wait_for(
                        spin_webpage(url=url, tags=tags),
                        timeout=config.timeout_per_item
                    )

                    # Store in memory if requested
                    shard_count = len(shards)
                    memory_count = 0

                    if store_in_memory and shards:
                        memories = shards_to_memories(shards)
                        if user_id:
                            for mem in memories:
                                mem.user_id = user_id
                        memory_ids = await memory.store_many(memories)
                        memory_count = len(memory_ids)

                    tracker.update(success=True)
                    return (shard_count, memory_count)

                except asyncio.TimeoutError:
                    error_msg = f"Timeout after {config.timeout_per_item}s"
                    if attempt < config.retry_attempts - 1:
                        await asyncio.sleep(config.retry_delay * (attempt + 1))
                        continue
                    errors[url] = error_msg
                    break

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    if attempt < config.retry_attempts - 1:
                        await asyncio.sleep(config.retry_delay * (attempt + 1))
                        continue
                    errors[url] = error_msg
                    if config.log_errors:
                        logger.error(f"Failed to ingest {url}: {error_msg}")
                    break

            # If we get here, all attempts failed
            tracker.update(success=False)
            if config.stop_on_error:
                raise RuntimeError(f"Batch ingestion stopped due to error on {url}")
            return None

    # Process all URLs concurrently
    tasks = [process_url(url) for url in urls]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results
    for item_result in results_list:
        if isinstance(item_result, Exception):
            result.failed += 1
        elif item_result is not None:
            shard_count, memory_count = item_result
            result.successful += 1
            total_shards += shard_count
            total_memories += memory_count
        else:
            result.failed += 1

    # Finalize results
    result.errors = errors
    result.shards_created = total_shards
    result.memories_stored = total_memories
    result.end_time = datetime.now()
    result.duration_seconds = (result.end_time - result.start_time).total_seconds()

    tracker.finish()

    return result


async def batch_ingest_files(
    files: List[Path],
    config: Optional[BatchConfig] = None,
    spinner_type: str = 'auto',  # auto, text, code
    tags: Optional[List[str]] = None,
    store_in_memory: bool = True,
    user_id: Optional[str] = None
) -> BatchResult:
    """
    Ingest multiple files in parallel.

    Args:
        files: List of file paths
        config: Batch configuration
        spinner_type: Spinner to use ('auto', 'text', 'code')
        tags: Tags to add to all content
        store_in_memory: Store in memory backend
        user_id: User ID for storage

    Returns:
        BatchResult with statistics

    Example:
        files = [
            Path('notes/doc1.txt'),
            Path('notes/doc2.md'),
            Path('code/main.py'),
        ]
        result = await batch_ingest_files(files, spinner_type='auto')
    """
    if config is None:
        config = BatchConfig()

    if tags is None:
        tags = []

    # Import spinners
    from .text import spin_text
    from .code import spin_code_file

    if store_in_memory:
        from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories
        memory = await create_unified_memory(user_id=user_id or "batch")

    # Initialize tracking
    result = BatchResult(total_items=len(files))
    tracker = ProgressTracker(len(files), config.show_progress)
    errors = {}
    total_shards = 0
    total_memories = 0

    semaphore = asyncio.Semaphore(config.max_workers)

    def detect_spinner(file_path: Path) -> str:
        """Auto-detect appropriate spinner."""
        ext = file_path.suffix.lower()
        code_exts = {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
        if ext in code_exts:
            return 'code'
        return 'text'

    async def process_file(file_path: Path) -> Optional[int]:
        """Process a single file."""
        async with semaphore:
            for attempt in range(config.retry_attempts):
                try:
                    # Read file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Determine spinner
                    actual_spinner = spinner_type
                    if spinner_type == 'auto':
                        actual_spinner = detect_spinner(file_path)

                    # Spin content
                    if actual_spinner == 'code':
                        shards = await spin_code_file(
                            path=str(file_path),
                            content=content
                        )
                    else:
                        shards = await spin_text(
                            text=content,
                            source=str(file_path),
                            tags=tags
                        )

                    # Store if requested
                    shard_count = len(shards)
                    memory_count = 0

                    if store_in_memory and shards:
                        memories = shards_to_memories(shards)
                        if user_id:
                            for mem in memories:
                                mem.user_id = user_id
                        memory_ids = await memory.store_many(memories)
                        memory_count = len(memory_ids)

                    tracker.update(success=True)
                    return (shard_count, memory_count)

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    if attempt < config.retry_attempts - 1:
                        await asyncio.sleep(config.retry_delay * (attempt + 1))
                        continue
                    errors[str(file_path)] = error_msg
                    if config.log_errors:
                        logger.error(f"Failed to ingest {file_path}: {error_msg}")
                    break

            tracker.update(success=False)
            return None

    # Process all files
    tasks = [process_file(f) for f in files]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate
    for item_result in results_list:
        if isinstance(item_result, Exception):
            result.failed += 1
        elif item_result is not None:
            shard_count, memory_count = item_result
            result.successful += 1
            total_shards += shard_count
            total_memories += memory_count
        else:
            result.failed += 1

    result.errors = errors
    result.shards_created = total_shards
    result.memories_stored = total_memories
    result.end_time = datetime.now()
    result.duration_seconds = (result.end_time - result.start_time).total_seconds()

    tracker.finish()

    return result


async def batch_ingest_from_list_file(
    list_file: Path,
    content_type: str = 'url',  # 'url' or 'file'
    config: Optional[BatchConfig] = None,
    **kwargs
) -> BatchResult:
    """
    Ingest items from a text file (one item per line).

    Args:
        list_file: Path to file containing URLs or file paths
        content_type: 'url' or 'file'
        config: Batch configuration
        **kwargs: Additional arguments passed to ingest function

    Returns:
        BatchResult

    Example:
        # urls.txt contains:
        # https://example.com/1
        # https://example.com/2
        # https://example.com/3

        result = await batch_ingest_from_list_file(
            Path('urls.txt'),
            content_type='url',
            tags=['batch-import']
        )
    """
    # Read items from file
    with open(list_file, 'r') as f:
        items = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if content_type == 'url':
        return await batch_ingest_urls(items, config=config, **kwargs)
    elif content_type == 'file':
        file_paths = [Path(item) for item in items]
        return await batch_ingest_files(file_paths, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown content_type: {content_type}")


# Convenience function for CLI usage
async def main_cli():
    """CLI entry point for batch ingestion."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch ingest URLs or files')
    parser.add_argument('input_file', help='File containing URLs or file paths (one per line)')
    parser.add_argument('--type', choices=['url', 'file'], default='url', help='Input type')
    parser.add_argument('--workers', type=int, default=5, help='Parallel workers')
    parser.add_argument('--retry', type=int, default=3, help='Retry attempts')
    parser.add_argument('--tags', nargs='+', help='Tags to add')
    parser.add_argument('--no-store', action='store_true', help='Don\'t store in memory')
    parser.add_argument('--user-id', help='User ID for memory storage')

    args = parser.parse_args()

    config = BatchConfig(
        max_workers=args.workers,
        retry_attempts=args.retry
    )

    result = await batch_ingest_from_list_file(
        Path(args.input_file),
        content_type=args.type,
        config=config,
        tags=args.tags or [],
        store_in_memory=not args.no_store,
        user_id=args.user_id
    )

    print(f"\n{result}")
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for item, error in list(result.errors.items())[:10]:
            print(f"  - {item}: {error}")


if __name__ == '__main__':
    asyncio.run(main_cli())
