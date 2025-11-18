"""
FilesystemSpinner - Ingest files from local filesystem into HoloLoom memory.

**Status**: Production Ready (November 2025)
**Purpose**: Convert local files into MemoryShards for RAG applications
**Features**:
    - Directory scanning with allow/deny patterns
    - Text and Markdown support (v1.0)
    - Smart chunking with overlap
    - Incremental ingestion via mtime tracking
    - Importance scoring based on file properties
    - Graceful handling of binary files

**Usage**:
    ```python
    from HoloLoom.spinningWheel.filesystem_spinner import FilesystemSpinner

    spinner = FilesystemSpinner(
        chunk_size=1000,
        chunk_overlap=200,
        allow_patterns=["*.md", "*.txt"],
        deny_patterns=["*.log", "node_modules/**"]
    )

    # Scan directory
    result = await spinner.spin("/path/to/docs")
    print(f"Created {result.shard_count} shards from {result.input_size_bytes} bytes")

    # Incremental update (only changed files)
    checkpoint = spinner.load_checkpoint("docs_checkpoint")
    result = await spinner.spin_incremental("/path/to/docs", checkpoint)
    ```

Author: Claude Code
Date: November 2025
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
import hashlib
import time
import mimetypes

from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinnerStatus,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FilesystemSpinnerConfig:
    """Configuration for FilesystemSpinner."""

    # Chunking
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks (for context)
    min_chunk_size: int = 100  # Minimum chunk size (discard smaller)

    # File filtering
    allow_patterns: List[str] = field(default_factory=lambda: ["*.txt", "*.md"])
    deny_patterns: List[str] = field(default_factory=lambda: [
        "*.log", ".git/**", "node_modules/**", ".venv/**", "__pycache__/**"
    ])

    # File size limits
    max_file_size: int = 10 * 1024 * 1024  # 10MB default limit
    skip_binary: bool = True  # Skip binary files

    # Importance scoring
    boost_markdown: float = 1.2  # Boost importance for Markdown files
    boost_recent: float = 1.1  # Boost recently modified files

    # Incremental ingestion
    track_mtimes: bool = True  # Track modification times
    checkpoint_enabled: bool = True  # Enable checkpointing


# ============================================================================
# FilesystemSpinner
# ============================================================================

class FilesystemSpinner(BaseSpinner):
    """
    Spinner for ingesting files from local filesystem.

    Converts text files into MemoryShards for HoloLoom's RAG system.

    Features:
        - Directory scanning with glob patterns
        - Text and Markdown support (v1.0)
        - Smart chunking with overlap
        - Incremental ingestion (mtime tracking)
        - Importance scoring
        - Graceful binary file handling
    """

    def __init__(
        self,
        config: Optional[FilesystemSpinnerConfig] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        allow_patterns: Optional[List[str]] = None,
        deny_patterns: Optional[List[str]] = None,
        importance_threshold: float = 0.0,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize FilesystemSpinner.

        Args:
            config: Configuration object (overrides individual params)
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            allow_patterns: Glob patterns to include (e.g., ["*.md", "*.txt"])
            deny_patterns: Glob patterns to exclude (e.g., ["*.log", ".git/**"])
            importance_threshold: Minimum importance score to include
            checkpoint_dir: Directory for saving checkpoints
        """
        super().__init__(
            name="filesystem",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir or Path.home() / ".hololoom" / "checkpoints"
        )

        # Use config or individual params
        if config:
            self.config = config
        else:
            self.config = FilesystemSpinnerConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                allow_patterns=allow_patterns or ["*.txt", "*.md"],
                deny_patterns=deny_patterns or [
                    "*.log", ".git/**", "node_modules/**", ".venv/**", "__pycache__/**"
                ]
            )

        # File tracking for incremental ingestion
        self._file_mtimes: Dict[str, float] = {}

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,  # TODO: Implement streaming in v1.1
            incremental=True,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=False,  # No entity extraction in v1.0
            motif_extraction=False,  # No motif extraction in v1.0
            embeddings=False,
            multimodal=False,
            max_input_size=self.config.max_file_size,
            supported_formats=["txt", "md", "markdown", "text"]
        )

    def is_available(self) -> bool:
        """Check if spinner is available (no external dependencies)."""
        return True  # Filesystem spinner has no external dependencies

    # ========================================================================
    # Core Implementation
    # ========================================================================

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Scan directory and convert files to MemoryShards.

        Args:
            source: Directory path (str or Path)
            **kwargs: Additional options:
                - recursive: Scan subdirectories (default: True)
                - follow_symlinks: Follow symbolic links (default: False)

        Returns:
            List of MemoryShards
        """
        # Validate source
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Directory not found: {source_path}")
        if not source_path.is_dir():
            raise ValueError(f"Source must be a directory: {source_path}")

        # Options
        recursive = kwargs.get("recursive", True)
        follow_symlinks = kwargs.get("follow_symlinks", False)

        # Scan files
        files = self._scan_directory(
            source_path,
            recursive=recursive,
            follow_symlinks=follow_symlinks
        )

        # Process files
        shards = []
        for file_path in files:
            file_shards = self._process_file(file_path)
            shards.extend(file_shards)

        return shards

    async def spin_incremental(
        self,
        source: Any,
        checkpoint: Optional[SpinnerCheckpoint] = None,
        **kwargs
    ) -> SpinResult:
        """
        Process only new/modified files since last checkpoint.

        Args:
            source: Directory path
            checkpoint: Previous checkpoint (None = process all)
            **kwargs: Additional options

        Returns:
            SpinResult with only new/modified file shards
        """
        source_path = Path(source)
        source_id = self._get_source_id(source_path)

        # Load checkpoint if not provided
        if checkpoint is None:
            checkpoint = self.load_checkpoint(source_id)

        # Get previous mtimes
        prev_mtimes = checkpoint.state.get("file_mtimes", {}) if checkpoint else {}

        # Options
        recursive = kwargs.get("recursive", True)
        follow_symlinks = kwargs.get("follow_symlinks", False)

        # Scan files
        all_files = self._scan_directory(
            source_path,
            recursive=recursive,
            follow_symlinks=follow_symlinks
        )

        # Filter to only new/modified files
        new_or_modified = []
        current_mtimes = {}

        for file_path in all_files:
            mtime = file_path.stat().st_mtime
            current_mtimes[str(file_path)] = mtime

            # Include if new or modified
            prev_mtime = prev_mtimes.get(str(file_path))
            if prev_mtime is None or mtime > prev_mtime:
                new_or_modified.append(file_path)

        # Process only new/modified files
        shards = []
        for file_path in new_or_modified:
            file_shards = self._process_file(file_path)
            shards.extend(file_shards)

        # Create/update checkpoint
        new_checkpoint = SpinnerCheckpoint(
            spinner_name=self.name,
            source_id=source_id,
            items_processed=len(all_files),
            items_total=len(all_files),
            state={"file_mtimes": current_mtimes}
        )
        self.save_checkpoint(new_checkpoint)

        # Create result
        result = SpinResult(
            shards=shards,
            success=True,
            processing_time_ms=0.0  # Will be calculated by wrapper
        )

        if checkpoint:
            result.warnings.append(
                f"Incremental update: {len(new_or_modified)} new/modified "
                f"out of {len(all_files)} total files"
            )

        return result

    def score_importance(self, file_path: Path) -> ImportanceScore:
        """
        Score importance of a file.

        Args:
            file_path: Path to file

        Returns:
            ImportanceScore with score and explanation
        """
        signals = ImportanceSignals()

        # Length score (longer files = more substantive)
        file_size = file_path.stat().st_size
        if file_size > 10000:
            signals.length_score = 1.0
        elif file_size > 5000:
            signals.length_score = 0.7
        elif file_size > 1000:
            signals.length_score = 0.5
        else:
            signals.length_score = 0.3

        # Technical score (based on file type)
        suffix = file_path.suffix.lower()
        if suffix in [".md", ".markdown"]:
            signals.technical_score = 0.8  # Markdown is structured
        elif suffix == ".txt":
            signals.technical_score = 0.5
        else:
            signals.technical_score = 0.3

        # Recency score (recently modified = more relevant)
        mtime = file_path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        if age_days < 7:
            signals.recency_score = 1.0
        elif age_days < 30:
            signals.recency_score = 0.7
        elif age_days < 90:
            signals.recency_score = 0.5
        else:
            signals.recency_score = 0.3

        # Structural score (README, docs are important)
        name_lower = file_path.name.lower()
        if "readme" in name_lower:
            signals.structural_score = 1.0
        elif "doc" in name_lower or "guide" in name_lower:
            signals.structural_score = 0.8
        else:
            signals.structural_score = 0.5

        # Compute total
        score = signals.compute_total()
        reason = signals.explain()

        return ImportanceScore(score=score, signals=signals, reason=reason)

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _scan_directory(
        self,
        root: Path,
        recursive: bool = True,
        follow_symlinks: bool = False
    ) -> List[Path]:
        """
        Scan directory for files matching allow/deny patterns.

        Args:
            root: Root directory to scan
            recursive: Scan subdirectories
            follow_symlinks: Follow symbolic links

        Returns:
            List of file paths matching criteria
        """
        matched_files = []

        # Get all files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in root.glob(pattern):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip symlinks if not following
            if file_path.is_symlink() and not follow_symlinks:
                continue

            # Check deny patterns first (exclusion wins)
            if self._matches_patterns(file_path, self.config.deny_patterns, root):
                continue

            # Check allow patterns
            if self._matches_patterns(file_path, self.config.allow_patterns, root):
                matched_files.append(file_path)

        return matched_files

    def _matches_patterns(
        self,
        file_path: Path,
        patterns: List[str],
        root: Path
    ) -> bool:
        """
        Check if file matches any of the glob patterns.

        Args:
            file_path: File to check
            patterns: List of glob patterns
            root: Root directory (for relative matching)

        Returns:
            True if file matches any pattern
        """
        # Get relative path from root
        try:
            rel_path = file_path.relative_to(root)
        except ValueError:
            # File is outside root (shouldn't happen)
            return False

        # Check each pattern
        for pattern in patterns:
            # Handle directory patterns (e.g., "node_modules/**")
            if "**" in pattern:
                # Match as path pattern
                if rel_path.match(pattern):
                    return True
            else:
                # Match as filename pattern
                if file_path.match(pattern):
                    return True

        return False

    def _process_file(self, file_path: Path) -> List[MemoryShard]:
        """
        Process a single file into MemoryShards.

        Args:
            file_path: Path to file

        Returns:
            List of MemoryShards (one per chunk)
        """
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.config.max_file_size:
            # Skip large files
            return []

        # Read file
        try:
            content = self._read_file(file_path)
        except Exception as e:
            # Skip files that can't be read
            return []

        if not content:
            return []

        # Score importance
        importance = self.score_importance(file_path)

        # Chunk content
        chunks = self._chunk_text(content)

        # Create shards
        shards = []
        for i, chunk_text in enumerate(chunks):
            # Skip small chunks
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            # Create shard ID
            shard_id = self._create_shard_id(file_path, i)

            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_size,
                "mtime": file_path.stat().st_mtime,
                "media_type": mimetypes.guess_type(file_path)[0] or "text/plain",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "importance_score": importance.score,
                "importance_reason": importance.reason,
                "importance_signals": importance.signals.to_dict(),
            }

            # Create shard
            shard = MemoryShard(
                id=shard_id,
                text=chunk_text,
                episode=f"file:{file_path.parent.name}",
                entities=[],  # No entity extraction in v1.0
                motifs=[],  # No motif extraction in v1.0
                metadata=metadata
            )

            shards.append(shard)

        return shards

    def _read_file(self, file_path: Path) -> str:
        """
        Read file content as text.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            ValueError: If file is binary and skip_binary is True
            UnicodeDecodeError: If file can't be decoded as UTF-8
        """
        # Try to detect binary files
        if self.config.skip_binary:
            # Read first 8192 bytes to check for null bytes
            with open(file_path, "rb") as f:
                sample = f.read(8192)
            if b"\x00" in sample:
                raise ValueError(f"Binary file: {file_path}")

        # Read as text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Cannot decode file: {file_path}") from e

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            # Get chunk
            end = start + self.config.chunk_size
            chunk = text[start:end]

            # Find good break point (end of sentence/paragraph)
            if end < len(text):
                # Look for paragraph break
                last_newline = chunk.rfind("\n\n")
                if last_newline > self.config.chunk_size // 2:
                    chunk = chunk[:last_newline]
                    end = start + last_newline
                else:
                    # Look for sentence break
                    last_period = max(
                        chunk.rfind(". "),
                        chunk.rfind(".\n"),
                        chunk.rfind("! "),
                        chunk.rfind("? ")
                    )
                    if last_period > self.config.chunk_size // 2:
                        chunk = chunk[:last_period + 1]
                        end = start + last_period + 1

            chunks.append(chunk.strip())

            # Move to next chunk (with overlap)
            start = end - self.config.chunk_overlap

            # Prevent infinite loop on tiny files
            if start <= 0:
                start = end

        return [c for c in chunks if c]  # Filter empty chunks

    def _create_shard_id(self, file_path: Path, chunk_index: int) -> str:
        """
        Create unique shard ID.

        Args:
            file_path: Path to file
            chunk_index: Chunk index

        Returns:
            Unique shard ID
        """
        # Hash the file path for stability
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"filesystem_{path_hash}_chunk{chunk_index}"

    def _get_source_id(self, source_path: Path) -> str:
        """
        Get stable source ID for checkpointing.

        Args:
            source_path: Source directory path

        Returns:
            Source ID (hash of absolute path)
        """
        abs_path = source_path.resolve()
        return hashlib.md5(str(abs_path).encode()).hexdigest()[:16]

    def interactive_select_files(
        self,
        source: Path,
        recursive: bool = True,
        follow_symlinks: bool = False
    ) -> List[Path]:
        """
        Interactively select files from a directory.

        Args:
            source: Directory path to scan
            recursive: Scan subdirectories
            follow_symlinks: Follow symbolic links

        Returns:
            List of selected file paths

        Example:
            >>> spinner = FilesystemSpinner()
            >>> selected = spinner.interactive_select_files(Path("/path/to/docs"))
            >>> result = await spinner.spin_custom_files(selected)
        """
        # Scan directory
        all_files = self._scan_directory(source, recursive, follow_symlinks)

        if not all_files:
            print("No files found matching patterns.")
            return []

        # Calculate importance for each file
        file_info = []
        for file_path in all_files:
            importance = self.score_importance(file_path)
            size = file_path.stat().st_size
            rel_path = file_path.relative_to(source) if file_path.is_relative_to(source) else file_path
            file_info.append({
                'path': file_path,
                'rel_path': rel_path,
                'importance': importance.score,
                'size': size,
                'selected': True  # Default: all selected
            })

        # Sort by importance (descending)
        file_info.sort(key=lambda x: x['importance'], reverse=True)

        # Interactive selection loop
        print(f"\n{'=' * 70}")
        print(f"Interactive File Selection")
        print(f"{'=' * 70}")
        print(f"Found {len(file_info)} files\n")

        while True:
            # Display files
            print(f"\n{'#':<4} {'Sel':<4} {'File':<40} {'Size':<10} {'Importance':<12}")
            print("-" * 70)

            for i, info in enumerate(file_info, 1):
                selected_mark = "[✓]" if info['selected'] else "[ ]"
                size_str = self._format_size(info['size'])
                imp_str = f"{info['importance']:.2f}"

                # Truncate long paths
                path_str = str(info['rel_path'])
                if len(path_str) > 38:
                    path_str = "..." + path_str[-35:]

                print(f"{i:<4} {selected_mark:<4} {path_str:<40} {size_str:<10} {imp_str:<12}")

            # Show summary
            selected_count = sum(1 for f in file_info if f['selected'])
            print("-" * 70)
            print(f"Selected: {selected_count}/{len(file_info)} files")

            # Prompt for action
            print("\nActions:")
            print("  <number>     - Toggle file selection")
            print("  all          - Select all files")
            print("  none         - Deselect all files")
            print("  invert       - Invert selection")
            print("  done         - Proceed with selected files")
            print("  cancel       - Cancel and exit")

            action = input("\nEnter action: ").strip().lower()

            if action == "done":
                # Return selected files
                selected_files = [info['path'] for info in file_info if info['selected']]
                if not selected_files:
                    print("\n⚠️  No files selected. Please select at least one file.")
                    continue
                print(f"\n✅ Proceeding with {len(selected_files)} files")
                return selected_files

            elif action == "cancel":
                print("\n❌ Selection cancelled")
                return []

            elif action == "all":
                for info in file_info:
                    info['selected'] = True
                print("✓ Selected all files")

            elif action == "none":
                for info in file_info:
                    info['selected'] = False
                print("✓ Deselected all files")

            elif action == "invert":
                for info in file_info:
                    info['selected'] = not info['selected']
                print("✓ Inverted selection")

            elif action.isdigit():
                num = int(action)
                if 1 <= num <= len(file_info):
                    info = file_info[num - 1]
                    info['selected'] = not info['selected']
                    status = "selected" if info['selected'] else "deselected"
                    print(f"✓ {status.capitalize()}: {info['rel_path']}")
                else:
                    print(f"❌ Invalid number. Please enter 1-{len(file_info)}")

            else:
                print(f"❌ Unknown action: {action}")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

    async def spin_custom_files(self, file_paths: List[Path]) -> SpinResult:
        """
        Process a custom list of files.

        Args:
            file_paths: List of file paths to process

        Returns:
            SpinResult with shards and metadata

        Example:
            >>> selected = spinner.interactive_select_files(Path("/path/to/docs"))
            >>> result = await spinner.spin_custom_files(selected)
        """
        start_time = time.time()

        try:
            # Process each selected file
            shards = []
            for file_path in file_paths:
                file_shards = self._process_file(file_path)
                shards.extend(file_shards)

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


# ============================================================================
# Convenience Functions
# ============================================================================

async def ingest_directory(
    directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    allow_patterns: Optional[List[str]] = None,
    deny_patterns: Optional[List[str]] = None,
    recursive: bool = True,
    incremental: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> SpinResult:
    """
    Convenience function to ingest a directory.

    Args:
        directory: Directory path to ingest
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        allow_patterns: Glob patterns to include
        deny_patterns: Glob patterns to exclude
        recursive: Scan subdirectories
        incremental: Only process new/modified files
        checkpoint_dir: Directory for checkpoints

    Returns:
        SpinResult with shards and metadata

    Example:
        >>> result = await ingest_directory(
        ...     "/path/to/docs",
        ...     allow_patterns=["*.md"],
        ...     incremental=True
        ... )
        >>> print(f"Ingested {result.shard_count} chunks")
    """
    spinner = FilesystemSpinner(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        allow_patterns=allow_patterns,
        deny_patterns=deny_patterns,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None
    )

    if incremental:
        return await spinner.spin_incremental(directory, recursive=recursive)
    else:
        return await spinner.spin(directory, recursive=recursive)
