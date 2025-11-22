"""
File Upload Spinner - Universal file ingestion dispatcher

Automatically detects file type and routes to appropriate spinner:
- .pdf → PDFSpinner
- .mbox → EmailSpinner
- .py, .js, .ts → CodebaseSpinner
- .txt, .md → TextSpinner
- .json, .csv, .xlsx → StructuredDataSpinner
- .mp3, .wav → AudioSpinner
- .jpg, .png → ImageSpinner
- .html, .htm → HTMLSpinner

Supports:
- Automatic file type detection (extension + MIME)
- Batch file upload
- Streaming for large files
- Importance filtering
- Unified error handling

Usage:
    from HoloLoom.spinningWheel.file_upload_spinner import FileUploadSpinner

    spinner = FileUploadSpinner()

    # Single file
    result = await spinner.spin("/path/to/file.pdf")

    # Multiple files
    result = await spinner.spin([
        "/path/to/doc.pdf",
        "/path/to/code.py",
        "/path/to/email.mbox"
    ])
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
import mimetypes

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities
)

# Import all specific spinners
try:
    from HoloLoom.spinningWheel.pdf_spinner import PDFSpinner
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.email_spinner import EmailSpinner
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.codebase_spinner import CodebaseSpinner
    CODE_AVAILABLE = True
except ImportError:
    CODE_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.multimodal_spinner import MultiModalSpinner
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False


class FileUploadSpinner(BaseSpinner):
    """
    Universal file upload spinner - automatically routes to appropriate spinner

    Supports:
    - PDF documents
    - Email archives (mbox)
    - Source code (Python, JS, TS, Java, Go, Rust)
    - Text files (txt, md, rst)
    - Images (jpg, png, gif)
    - Audio (mp3, wav, m4a)
    - Structured data (json, csv, xlsx)
    - HTML files
    """

    # File type to spinner mapping
    FILE_TYPE_MAP = {
        # Documents
        '.pdf': 'pdf',
        '.doc': 'document',
        '.docx': 'document',
        '.odt': 'document',

        # Email
        '.mbox': 'email',
        '.eml': 'email',

        # Code
        '.py': 'code',
        '.js': 'code',
        '.jsx': 'code',
        '.ts': 'code',
        '.tsx': 'code',
        '.java': 'code',
        '.go': 'code',
        '.rs': 'code',
        '.c': 'code',
        '.cpp': 'code',
        '.h': 'code',
        '.hpp': 'code',

        # Text
        '.txt': 'text',
        '.md': 'text',
        '.markdown': 'text',
        '.rst': 'text',
        '.log': 'text',

        # Images
        '.jpg': 'image',
        '.jpeg': 'image',
        '.png': 'image',
        '.gif': 'image',
        '.webp': 'image',
        '.svg': 'image',

        # Audio
        '.mp3': 'audio',
        '.wav': 'audio',
        '.m4a': 'audio',
        '.flac': 'audio',
        '.ogg': 'audio',

        # Structured data
        '.json': 'structured',
        '.csv': 'structured',
        '.xlsx': 'structured',
        '.xls': 'structured',
        '.xml': 'structured',
        '.yaml': 'structured',
        '.yml': 'structured',

        # Web
        '.html': 'html',
        '.htm': 'html',
    }

    def __init__(
        self,
        importance_threshold: float = 0.3,
        enable_ocr: bool = False,
        enable_audio_transcription: bool = False
    ):
        """
        Initialize FileUploadSpinner

        Args:
            importance_threshold: Minimum importance score (0.0-1.0)
            enable_ocr: Enable OCR for scanned PDFs/images
            enable_audio_transcription: Enable audio transcription
        """
        super().__init__(name="file_upload")

        self.importance_threshold = importance_threshold
        self.enable_ocr = enable_ocr
        self.enable_audio_transcription = enable_audio_transcription

        # Initialize specific spinners (lazy loading)
        self._spinners: Dict[str, BaseSpinner] = {}

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=False,
            streaming=True,
            supported_formats=list(self.FILE_TYPE_MAP.keys()),
            batch_processing=True,
            multimodal=True
        )

    def is_available(self) -> bool:
        """Check if at least one spinner is available"""
        return any([
            PDF_AVAILABLE,
            EMAIL_AVAILABLE,
            CODE_AVAILABLE,
            MULTIMODAL_AVAILABLE
        ])

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Spin file(s) into MemoryShards

        Args:
            source: File path (str/Path) or list of file paths
            **kwargs: Additional arguments

        Returns:
            List of MemoryShards
        """
        # Handle single file or multiple files
        if isinstance(source, (str, Path)):
            files = [Path(source)]
        elif isinstance(source, list):
            files = [Path(f) for f in source]
        else:
            raise ValueError(f"source must be file path or list of paths, got {type(source)}")

        all_shards = []
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Detect file type
            file_type = self._detect_file_type(file_path)

            # Get appropriate spinner
            spinner = self._get_spinner(file_type)

            if spinner is None:
                # Unsupported file type - skip or use text fallback
                continue

            # Spin file
            try:
                shards = await spinner.spin(str(file_path))
                all_shards.extend(shards)
            except Exception as e:
                # Log error but continue with other files
                print(f"Warning: Failed to process {file_path}: {e}")
                continue

        return all_shards

    async def spin_stream(
        self,
        source: Any,
        batch_size: int = 10,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from file(s)

        Args:
            source: File path or list of file paths
            batch_size: Number of files per batch

        Yields:
            MemoryShard objects
        """
        shards = await self._spin_impl(source, **kwargs)

        for i in range(0, len(shards), batch_size):
            batch = shards[i:i + batch_size]
            for shard in batch:
                yield shard

    def _detect_file_type(self, file_path: Path) -> str:
        """
        Detect file type from extension and MIME type

        Args:
            file_path: Path to file

        Returns:
            File type string ('pdf', 'email', 'code', etc.)
        """
        # Try extension first
        ext = file_path.suffix.lower()
        if ext in self.FILE_TYPE_MAP:
            return self.FILE_TYPE_MAP[ext]

        # Try MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('text/'):
                return 'text'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type == 'application/pdf':
                return 'pdf'
            elif mime_type in ['message/rfc822', 'application/mbox']:
                return 'email'

        # Default to text
        return 'text'

    def _get_spinner(self, file_type: str) -> Optional[BaseSpinner]:
        """
        Get or create spinner for file type

        Args:
            file_type: File type string

        Returns:
            Spinner instance or None if unsupported
        """
        # Check cache
        if file_type in self._spinners:
            return self._spinners[file_type]

        # Create spinner based on type
        spinner = None

        if file_type == 'pdf' and PDF_AVAILABLE:
            spinner = PDFSpinner(
                importance_threshold=self.importance_threshold,
                enable_ocr=self.enable_ocr
            )

        elif file_type == 'email' and EMAIL_AVAILABLE:
            spinner = EmailSpinner(
                importance_threshold=self.importance_threshold
            )

        elif file_type == 'code' and CODE_AVAILABLE:
            spinner = CodebaseSpinner(
                importance_threshold=self.importance_threshold
            )

        elif file_type in ['text', 'html'] and MULTIMODAL_AVAILABLE:
            spinner = MultiModalSpinner(
                importance_threshold=self.importance_threshold
            )

        elif file_type == 'image' and MULTIMODAL_AVAILABLE:
            spinner = MultiModalSpinner(
                importance_threshold=self.importance_threshold
            )

        elif file_type == 'audio' and MULTIMODAL_AVAILABLE:
            if self.enable_audio_transcription:
                spinner = MultiModalSpinner(
                    importance_threshold=self.importance_threshold
                )

        elif file_type == 'structured' and MULTIMODAL_AVAILABLE:
            spinner = MultiModalSpinner(
                importance_threshold=self.importance_threshold
            )

        # Cache spinner
        if spinner:
            self._spinners[file_type] = spinner

        return spinner

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.FILE_TYPE_MAP.keys())

    def get_file_type_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about file type and which spinner will handle it

        Args:
            file_path: Path to file

        Returns:
            Dict with file type info
        """
        file_type = self._detect_file_type(file_path)
        spinner = self._get_spinner(file_type)

        return {
            'file_path': str(file_path),
            'extension': file_path.suffix,
            'detected_type': file_type,
            'spinner': spinner.get_name() if spinner else None,
            'supported': spinner is not None
        }


# Convenience functions

async def spin_uploaded_file(
    file_path: str,
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to spin an uploaded file

    Args:
        file_path: Path to uploaded file
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards
    """
    spinner = FileUploadSpinner(importance_threshold=importance_threshold)
    shards = await spinner.spin(file_path)

    return SpinResult(
        shards=shards,
        success=True,
        items_processed=len(shards),
        items_filtered=0
    )


async def spin_uploaded_files(
    file_paths: List[str],
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to spin multiple uploaded files

    Args:
        file_paths: List of file paths
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards
    """
    spinner = FileUploadSpinner(importance_threshold=importance_threshold)
    shards = await spinner.spin(file_paths)

    return SpinResult(
        shards=shards,
        success=True,
        items_processed=len(shards),
        items_filtered=0
    )
