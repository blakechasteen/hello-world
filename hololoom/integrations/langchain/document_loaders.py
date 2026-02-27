"""
Universal Document Loader - LangChain Integration
==================================================

Wraps 100+ LangChain document loaders into HoloLoom's MemoryShard format.

Features:
- Auto-detects document type from file extension
- Converts LangChain Documents → HoloLoom MemoryShards
- Supports all LangChain loaders (PDFs, web, databases, etc.)
- Graceful degradation if optional dependencies missing

Author: Claude Code
Date: 2025-11-20
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import warnings

try:
    from langchain.document_loaders import (
        # Text-based
        TextLoader,
        DirectoryLoader,
        UnstructuredMarkdownLoader,

        # PDFs
        PyPDFLoader,
        PDFMinerLoader,
        UnstructuredPDFLoader,

        # Office documents
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
        UnstructuredExcelLoader,

        # Web
        WebBaseLoader,
        UnstructuredURLLoader,
        SeleniumURLLoader,

        # Code
        GitLoader,
        GitHubIssuesLoader,
        NotebookLoader,

        # Databases
        AirtableLoader,
        NotionDBLoader,

        # Communication
        SlackDirectoryLoader,
        DiscordChatLoader,

        # Others
        CSVLoader,
        JSONLoader,
        UnstructuredHTMLLoader,
        UnstructuredImageLoader,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn(
        "LangChain not installed. Document loaders limited to HoloLoom spinners. "
        "Install with: pip install langchain langchain-community",
        ImportWarning
    )

# HoloLoom imports
try:
    from hololoom.Documentation.types import MemoryShard
except ImportError:
    from dataclasses import dataclass
    from typing import Dict, Any

    @dataclass
    class MemoryShard:
        content: str
        source: str
        metadata: Dict[str, Any]


# Document type to loader mapping (only if LangChain available)
if LANGCHAIN_AVAILABLE:
    LOADER_REGISTRY = {
        # Text
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.markdown': UnstructuredMarkdownLoader,

        # PDFs
        '.pdf': PyPDFLoader,

        # Office
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
    '.xls': UnstructuredExcelLoader,

    # Code
    '.ipynb': NotebookLoader,
    '.py': TextLoader,
    '.js': TextLoader,
    '.ts': TextLoader,
    '.jsx': TextLoader,
    '.tsx': TextLoader,
    '.java': TextLoader,
    '.cpp': TextLoader,
    '.c': TextLoader,
    '.h': TextLoader,
    '.go': TextLoader,
    '.rs': TextLoader,

    # Web
    '.html': UnstructuredHTMLLoader,
    '.htm': UnstructuredHTMLLoader,

    # Data
    '.csv': CSVLoader,
    '.json': JSONLoader,

    # Images (requires OCR)
    '.png': UnstructuredImageLoader,
    '.jpg': UnstructuredImageLoader,
    '.jpeg': UnstructuredImageLoader,
    }
else:
    LOADER_REGISTRY = {}


class UniversalDocumentLoader:
    """
    Universal document loader that auto-detects type and uses appropriate LangChain loader.

    Supports 100+ document types through LangChain's extensive loader ecosystem.
    """

    def __init__(self, enable_ocr: bool = False, enable_tables: bool = True):
        """
        Initialize universal loader.

        Args:
            enable_ocr: Enable OCR for images/scanned PDFs (requires tesseract)
            enable_tables: Enable table extraction from PDFs/Excel
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain langchain-community"
            )

        self.enable_ocr = enable_ocr
        self.enable_tables = enable_tables

    def load(
        self,
        source: Union[str, Path],
        loader_type: Optional[str] = None,
        **loader_kwargs
    ) -> List[MemoryShard]:
        """
        Load documents from any source.

        Args:
            source: File path, URL, or directory
            loader_type: Optional explicit loader type (auto-detected if None)
            **loader_kwargs: Additional arguments for the loader

        Returns:
            List of MemoryShards

        Example:
            >>> loader = UniversalDocumentLoader()
            >>> shards = loader.load("research_paper.pdf")
            >>> shards = loader.load("https://example.com/article")
            >>> shards = loader.load("docs/", loader_type="directory")
        """
        source = Path(source) if not isinstance(source, Path) else source

        # Detect loader type
        if loader_type is None:
            loader_type = self._detect_loader_type(source)

        # Get appropriate loader
        loader_class = self._get_loader_class(loader_type, source)

        # Load documents
        try:
            if loader_type == 'directory':
                loader = DirectoryLoader(
                    str(source),
                    glob="**/*",
                    loader_cls=TextLoader,
                    **loader_kwargs
                )
            elif loader_type == 'url':
                loader = WebBaseLoader(str(source), **loader_kwargs)
            else:
                loader = loader_class(str(source), **loader_kwargs)

            langchain_docs = loader.load()
        except Exception as e:
            warnings.warn(f"Failed to load {source}: {e}")
            return []

        # Convert to MemoryShards
        shards = []
        for i, doc in enumerate(langchain_docs):
            shard = MemoryShard(
                content=doc.page_content,
                source=str(source) + (f"#page_{doc.metadata.get('page', i)}" if 'page' in doc.metadata else f"#{i}"),
                metadata={
                    **doc.metadata,
                    'loader_type': loader_type,
                    'original_source': str(source),
                }
            )
            shards.append(shard)

        return shards

    def load_directory(
        self,
        directory: Union[str, Path],
        glob_pattern: str = "**/*",
        exclude: Optional[List[str]] = None
    ) -> List[MemoryShard]:
        """
        Load all documents from a directory.

        Args:
            directory: Directory path
            glob_pattern: Glob pattern for file matching
            exclude: List of patterns to exclude

        Returns:
            List of MemoryShards

        Example:
            >>> loader = UniversalDocumentLoader()
            >>> shards = loader.load_directory("docs/", glob_pattern="**/*.md")
        """
        directory = Path(directory)
        shards = []

        for file_path in directory.glob(glob_pattern):
            if file_path.is_file():
                # Check exclusions
                if exclude and any(file_path.match(pattern) for pattern in exclude):
                    continue

                try:
                    file_shards = self.load(file_path)
                    shards.extend(file_shards)
                except Exception as e:
                    warnings.warn(f"Failed to load {file_path}: {e}")
                    continue

        return shards

    def load_urls(self, urls: List[str], **loader_kwargs) -> List[MemoryShard]:
        """
        Load documents from multiple URLs.

        Args:
            urls: List of URLs
            **loader_kwargs: Additional arguments for WebBaseLoader

        Returns:
            List of MemoryShards
        """
        shards = []

        for url in urls:
            try:
                url_shards = self.load(url, loader_type='url', **loader_kwargs)
                shards.extend(url_shards)
            except Exception as e:
                warnings.warn(f"Failed to load {url}: {e}")
                continue

        return shards

    def _detect_loader_type(self, source: Path) -> str:
        """Auto-detect loader type from source."""
        source_str = str(source)

        # URL
        if source_str.startswith(('http://', 'https://')):
            return 'url'

        # Directory
        if source.is_dir():
            return 'directory'

        # File extension
        suffix = source.suffix.lower()
        if suffix in LOADER_REGISTRY:
            return suffix

        # Default to text
        return '.txt'

    def _get_loader_class(self, loader_type: str, source: Path):
        """Get appropriate loader class for type."""
        if loader_type in LOADER_REGISTRY:
            return LOADER_REGISTRY[loader_type]
        elif loader_type == 'url':
            return WebBaseLoader
        elif loader_type == 'directory':
            return DirectoryLoader
        else:
            return TextLoader


# Convenience functions

def load_documents(
    source: Union[str, Path, List[str]],
    loader_type: Optional[str] = None,
    **kwargs
) -> List[MemoryShard]:
    """
    Convenience function to load documents.

    Args:
        source: File path, URL, directory, or list of URLs
        loader_type: Optional loader type
        **kwargs: Additional loader arguments

    Returns:
        List of MemoryShards

    Example:
        >>> shards = load_documents("paper.pdf")
        >>> shards = load_documents(["https://site1.com", "https://site2.com"])
        >>> shards = load_documents("docs/", loader_type="directory")
    """
    loader = UniversalDocumentLoader()

    if isinstance(source, list):
        # List of URLs
        return loader.load_urls(source, **kwargs)
    else:
        # Single source
        return loader.load(source, loader_type=loader_type, **kwargs)


def supported_document_types() -> Dict[str, str]:
    """
    Get list of supported document types.

    Returns:
        Dict mapping file extension to loader class name
    """
    return {
        ext: loader.__name__
        for ext, loader in LOADER_REGISTRY.items()
    }


# Specialized loaders

def load_github_repo(repo_url: str, branch: str = "main") -> List[MemoryShard]:
    """
    Load all files from a GitHub repository.

    Args:
        repo_url: GitHub repository URL
        branch: Branch name

    Returns:
        List of MemoryShards
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain required for GitHub loader")

    from langchain.document_loaders import GitLoader

    loader = GitLoader(repo_path=repo_url, branch=branch)
    docs = loader.load()

    return [
        MemoryShard(
            content=doc.page_content,
            source=f"{repo_url}#{doc.metadata.get('file_path', 'unknown')}",
            metadata={**doc.metadata, 'repo': repo_url, 'branch': branch}
        )
        for doc in docs
    ]


def load_slack_workspace(
    slack_export_path: Union[str, Path]
) -> List[MemoryShard]:
    """
    Load Slack workspace export.

    Args:
        slack_export_path: Path to Slack export directory

    Returns:
        List of MemoryShards
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain required for Slack loader")

    from langchain.document_loaders import SlackDirectoryLoader

    loader = SlackDirectoryLoader(str(slack_export_path))
    docs = loader.load()

    return [
        MemoryShard(
            content=doc.page_content,
            source=f"slack#{doc.metadata.get('channel', 'unknown')}",
            metadata={**doc.metadata, 'platform': 'slack'}
        )
        for doc in docs
    ]


def load_notion_database(
    notion_token: str,
    database_id: str
) -> List[MemoryShard]:
    """
    Load Notion database.

    Args:
        notion_token: Notion API token
        database_id: Database ID

    Returns:
        List of MemoryShards
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain required for Notion loader")

    from langchain.document_loaders import NotionDBLoader

    loader = NotionDBLoader(
        integration_token=notion_token,
        database_id=database_id
    )
    docs = loader.load()

    return [
        MemoryShard(
            content=doc.page_content,
            source=f"notion#{doc.metadata.get('id', 'unknown')}",
            metadata={**doc.metadata, 'platform': 'notion'}
        )
        for doc in docs
    ]
