"""
File Processor - Multi-Format File Ingestion
=============================================
Auto-detecting file processor with support for PDF, DOCX, text, and more.

Philosophy:
Data comes in many shapes - PDFs, Word docs, audio transcripts, code files.
The FileProcessor abstracts away format complexity, providing a unified
interface that outputs clean, chunked text ready for embedding.
"""

import logging
import asyncio
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .chunker import SmartChunker, ChunkConfig, Chunk

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class ProcessorConfig:
    """Configuration for file processor."""
    chunk_config: ChunkConfig = field(default_factory=ChunkConfig)
    enable_ocr: bool = False  # OCR for image-based PDFs
    max_file_size_mb: int = 100  # Maximum file size
    parallel_processing: bool = True
    progress_callback: Optional[Callable[[str, float], None]] = None


@dataclass
class ProcessResult:
    """Result from file processing."""
    file_path: str
    file_type: FileType
    chunks: List[Chunk]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type.value,
            "num_chunks": len(self.chunks),
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


class FileProcessor:
    """
    Multi-format file processor with smart chunking.

    Automatically detects file types and routes to appropriate parsers.
    Outputs semantically chunked text ready for embedding.

    Usage:
        processor = FileProcessor(ProcessorConfig())

        # Process single file
        result = await processor.process_file("document.pdf")
        for chunk in result.chunks:
            print(f"Chunk {chunk.chunk_id}: {chunk.text[:100]}")

        # Process multiple files
        results = await processor.process_batch(["file1.pdf", "file2.docx"])
    """

    def __init__(self, config: ProcessorConfig):
        """
        Initialize file processor.

        Args:
            config: Processor configuration
        """
        self.config = config
        self.chunker = SmartChunker(config.chunk_config)

        # Import parsers lazily
        self.parsers = self._init_parsers()

        logger.info("FileProcessor initialized")

    def _init_parsers(self) -> Dict[FileType, Any]:
        """Initialize file parsers with graceful fallback."""
        parsers = {}

        # Text parser (always available)
        parsers[FileType.TEXT] = self._parse_text
        parsers[FileType.MARKDOWN] = self._parse_text
        parsers[FileType.CODE] = self._parse_text

        # PDF parser
        try:
            from .parsers.pdf import parse_pdf
            parsers[FileType.PDF] = parse_pdf
            logger.info("PDF parser available")
        except ImportError:
            logger.warning("PDF parser unavailable (install pypdf2)")
            parsers[FileType.PDF] = self._parse_fallback

        # DOCX parser
        try:
            from .parsers.docx import parse_docx
            parsers[FileType.DOCX] = parse_docx
            logger.info("DOCX parser available")
        except ImportError:
            logger.warning("DOCX parser unavailable (install python-docx)")
            parsers[FileType.DOCX] = self._parse_fallback

        return parsers

    def detect_file_type(self, file_path: str) -> FileType:
        """
        Detect file type from extension and MIME type.

        Args:
            file_path: Path to file

        Returns:
            Detected file type
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # Map extensions to types
        ext_map = {
            '.pdf': FileType.PDF,
            '.docx': FileType.DOCX,
            '.doc': FileType.DOCX,
            '.txt': FileType.TEXT,
            '.md': FileType.MARKDOWN,
            '.markdown': FileType.MARKDOWN,
            '.py': FileType.CODE,
            '.js': FileType.CODE,
            '.java': FileType.CODE,
            '.cpp': FileType.CODE,
            '.c': FileType.CODE,
            '.mp3': FileType.AUDIO,
            '.wav': FileType.AUDIO,
            '.m4a': FileType.AUDIO,
        }

        return ext_map.get(ext, FileType.UNKNOWN)

    async def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessResult:
        """
        Process single file.

        Args:
            file_path: Path to file
            metadata: Optional metadata to attach

        Returns:
            ProcessResult with chunks
        """
        import time
        start_time = time.time()

        path = Path(file_path)

        # Check file exists
        if not path.exists():
            return ProcessResult(
                file_path=file_path,
                file_type=FileType.UNKNOWN,
                chunks=[],
                metadata={},
                success=False,
                error=f"File not found: {file_path}"
            )

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return ProcessResult(
                file_path=file_path,
                file_type=FileType.UNKNOWN,
                chunks=[],
                metadata={},
                success=False,
                error=f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB"
            )

        # Detect type
        file_type = self.detect_file_type(file_path)

        # Progress callback
        if self.config.progress_callback:
            self.config.progress_callback(file_path, 0.1)

        try:
            # Parse file
            parser = self.parsers.get(file_type, self._parse_fallback)
            text = await parser(file_path)

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(file_path, 0.5)

            # Chunk text
            file_metadata = metadata or {}
            file_metadata.update({
                "file_name": path.name,
                "file_type": file_type.value,
                "file_size_mb": file_size_mb
            })

            chunks = self.chunker.chunk(text, metadata=file_metadata)

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(file_path, 1.0)

            processing_time = time.time() - start_time

            return ProcessResult(
                file_path=file_path,
                file_type=file_type,
                chunks=chunks,
                metadata=file_metadata,
                success=True,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            processing_time = time.time() - start_time

            return ProcessResult(
                file_path=file_path,
                file_type=file_type,
                chunks=[],
                metadata={},
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    async def process_batch(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessResult]:
        """
        Process multiple files in parallel.

        Args:
            file_paths: List of file paths
            metadata: Optional metadata for all files

        Returns:
            List of ProcessResults
        """
        if self.config.parallel_processing:
            # Process in parallel
            tasks = [
                self.process_file(path, metadata)
                for path in file_paths
            ]
            results = await asyncio.gather(*tasks)
        else:
            # Process sequentially
            results = []
            for path in file_paths:
                result = await self.process_file(path, metadata)
                results.append(result)

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful")

        return results

    async def _parse_text(self, file_path: str) -> str:
        """Parse plain text file."""
        path = Path(file_path)
        return path.read_text(encoding='utf-8', errors='ignore')

    async def _parse_fallback(self, file_path: str) -> str:
        """Fallback parser for unsupported types."""
        logger.warning(f"No parser for {file_path}, using text fallback")
        return await self._parse_text(file_path)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("File Processor Demo")
        print("="*80 + "\n")

        # Progress callback
        def progress(file_path, percent):
            print(f"  Processing {Path(file_path).name}: {percent*100:.0f}%")

        # Create processor
        config = ProcessorConfig(
            chunk_config=ChunkConfig(chunk_size=300, overlap=50),
            progress_callback=progress
        )
        processor = FileProcessor(config)

        # Create sample text file
        sample_file = "/tmp/sample_document.txt"
        Path(sample_file).write_text("""
        Thompson Sampling: A Bayesian Approach to Multi-Armed Bandits

        Introduction
        Thompson Sampling is a classic algorithm for the multi-armed bandit problem.
        It was introduced by William R. Thompson in 1933, making it one of the oldest
        approaches to this fundamental problem in decision theory.

        Algorithm Description
        The algorithm maintains a probability distribution over the reward parameter
        for each arm. At each time step, it samples from these distributions and
        selects the arm with the highest sample value. After observing a reward,
        the posterior distribution is updated using Bayes' rule.

        Applications
        Thompson Sampling is widely used in:
        - Online advertising and recommendation systems
        - Clinical trials and medical decision-making
        - Reinforcement learning and robotics
        - A/B testing and experiment design
        """)

        # Process file
        print("Processing file...\n")
        result = await processor.process_file(
            sample_file,
            metadata={"source": "demo", "topic": "Thompson Sampling"}
        )

        print(f"\nProcessing Result:")
        print(f"  Success: {result.success}")
        print(f"  File type: {result.file_type.value}")
        print(f"  Chunks: {len(result.chunks)}")
        print(f"  Processing time: {result.processing_time:.3f}s\n")

        print("Chunks:")
        for chunk in result.chunks:
            print(f"\n  Chunk {chunk.chunk_id}:")
            print(f"    Length: {len(chunk.text)} chars")
            print(f"    Preview: {chunk.text[:100]}...")

        # Cleanup
        Path(sample_file).unlink()

        print("\n✓ Demo complete!")

    asyncio.run(demo())
