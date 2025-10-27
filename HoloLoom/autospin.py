#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoSpin Orchestrator
=====================
Automatically spins text input into MemoryShards before orchestration.

This wrapper makes it easy to use HoloLoom with plain text without
manually calling spinners first.

Usage:
    # Auto-spin from text
    orchestrator = AutoSpinOrchestrator.from_text(
        text="Your knowledge base content...",
        config=Config.fused()
    )

    # Or from a file
    orchestrator = AutoSpinOrchestrator.from_file(
        "notes.md",
        config=Config.fast()
    )

    # Then process queries normally
    response = await orchestrator.process(Query(text="What is HoloLoom?"))
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    from HoloLoom.Documentation.types import Query, MemoryShard
    from HoloLoom.config import Config
    from HoloLoom.orchestrator import HoloLoomOrchestrator
    from HoloLoom.spinningWheel import TextSpinner, TextSpinnerConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you run from repository root with PYTHONPATH set")
    raise


class AutoSpinOrchestrator:
    """
    Orchestrator wrapper that automatically spins text into MemoryShards.

    This eliminates the manual step of creating shards before orchestration.
    Just provide text, and it automatically uses TextSpinner to create shards.
    """

    def __init__(
        self,
        shards: List[MemoryShard],
        config: Config,
        spinner_config: Optional[TextSpinnerConfig] = None
    ):
        """
        Initialize with pre-spun shards.

        Args:
            shards: MemoryShards (already spun)
            config: HoloLoom Config
            spinner_config: TextSpinnerConfig (stored for future use)
        """
        self.config = config
        self.spinner_config = spinner_config or TextSpinnerConfig()
        self.shards = shards

        # Create the underlying orchestrator
        self.orchestrator = HoloLoomOrchestrator(
            cfg=config,
            shards=shards
        )

    @classmethod
    async def from_text(
        cls,
        text: str,
        config: Optional[Config] = None,
        spinner_config: Optional[TextSpinnerConfig] = None,
        source: str = "knowledge_base"
    ) -> "AutoSpinOrchestrator":
        """
        Create orchestrator from plain text.

        Automatically spins the text into MemoryShards using TextSpinner.

        Args:
            text: Plain text content (knowledge base, notes, etc.)
            config: HoloLoom Config (default: Config.fused())
            spinner_config: TextSpinnerConfig (default: paragraph chunking)
            source: Source identifier for the text

        Returns:
            AutoSpinOrchestrator ready to process queries

        Example:
            orchestrator = await AutoSpinOrchestrator.from_text(
                text="HoloLoom combines neural nets with knowledge graphs...",
                config=Config.fast()
            )
            response = await orchestrator.process(Query(text="What is HoloLoom?"))
        """
        # Defaults
        if config is None:
            config = Config.fused()

        if spinner_config is None:
            # Default: paragraph chunking with reasonable size
            spinner_config = TextSpinnerConfig(
                chunk_by='paragraph',
                chunk_size=500,
                extract_entities=True
            )

        # Spin the text into shards
        spinner = TextSpinner(spinner_config)
        shards = await spinner.spin({
            'text': text,
            'source': source
        })

        # Create instance
        return cls(
            shards=shards,
            config=config,
            spinner_config=spinner_config
        )

    @classmethod
    async def from_file(
        cls,
        file_path: Union[str, Path],
        config: Optional[Config] = None,
        spinner_config: Optional[TextSpinnerConfig] = None,
        encoding: str = 'utf-8'
    ) -> "AutoSpinOrchestrator":
        """
        Create orchestrator from a text file.

        Reads the file and automatically spins it into MemoryShards.

        Args:
            file_path: Path to text file
            config: HoloLoom Config (default: Config.fused())
            spinner_config: TextSpinnerConfig (default: paragraph chunking)
            encoding: File encoding (default: 'utf-8')

        Returns:
            AutoSpinOrchestrator ready to process queries

        Example:
            orchestrator = await AutoSpinOrchestrator.from_file(
                "documentation.md",
                config=Config.fast()
            )
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file
        with open(path, 'r', encoding=encoding) as f:
            text = f.read()

        # Use from_text with filename as source
        return await cls.from_text(
            text=text,
            config=config,
            spinner_config=spinner_config,
            source=str(path.name)
        )

    @classmethod
    async def from_documents(
        cls,
        documents: List[Dict[str, str]],
        config: Optional[Config] = None,
        spinner_config: Optional[TextSpinnerConfig] = None
    ) -> "AutoSpinOrchestrator":
        """
        Create orchestrator from multiple documents.

        Each document should be a dict with 'text' and optionally 'source' keys.

        Args:
            documents: List of dicts with 'text' and optional 'source'
            config: HoloLoom Config (default: Config.fused())
            spinner_config: TextSpinnerConfig

        Returns:
            AutoSpinOrchestrator with all documents spun into shards

        Example:
            docs = [
                {'text': 'First document...', 'source': 'doc1.txt'},
                {'text': 'Second document...', 'source': 'doc2.txt'}
            ]
            orchestrator = await AutoSpinOrchestrator.from_documents(docs)
        """
        if config is None:
            config = Config.fused()

        if spinner_config is None:
            spinner_config = TextSpinnerConfig(
                chunk_by='paragraph',
                chunk_size=500,
                extract_entities=True
            )

        # Spin all documents
        spinner = TextSpinner(spinner_config)
        all_shards = []

        for idx, doc in enumerate(documents):
            text = doc.get('text')
            if not text:
                continue

            source = doc.get('source', f'document_{idx:03d}')
            metadata = doc.get('metadata', {})

            shards = await spinner.spin({
                'text': text,
                'source': source,
                'metadata': metadata
            })
            all_shards.extend(shards)

        return cls(
            shards=all_shards,
            config=config,
            spinner_config=spinner_config
        )

    async def add_text(
        self,
        text: str,
        source: str = "additional_content"
    ) -> None:
        """
        Add more text to the knowledge base dynamically.

        Spins the new text and adds shards to the existing memory.

        Args:
            text: New text content to add
            source: Source identifier

        Example:
            # Start with initial content
            orch = await AutoSpinOrchestrator.from_text("Initial knowledge...")

            # Add more later
            await orch.add_text("Additional facts...", source="update_1")
        """
        spinner = TextSpinner(self.spinner_config)
        new_shards = await spinner.spin({
            'text': text,
            'source': source
        })

        # Add to existing shards
        self.shards.extend(new_shards)

        # Recreate orchestrator with updated shards
        self.orchestrator = HoloLoomOrchestrator(
            cfg=self.config,
            shards=self.shards
        )

    async def process(self, query: Query) -> Dict[str, Any]:
        """
        Process a query using the underlying orchestrator.

        Args:
            query: User query

        Returns:
            Response dict with results and metadata
        """
        return await self.orchestrator.process(query)

    def get_shard_count(self) -> int:
        """Get the number of memory shards."""
        return len(self.shards)

    def get_shard_preview(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a preview of the memory shards.

        Args:
            limit: Max number of shards to preview

        Returns:
            List of shard summaries
        """
        preview = []
        for shard in self.shards[:limit]:
            preview.append({
                'id': shard.id,
                'source': shard.metadata.get('source', 'unknown'),
                'text_preview': shard.text[:100] + '...' if len(shard.text) > 100 else shard.text,
                'entities': shard.entities[:5] if shard.entities else [],
                'char_count': len(shard.text)
            })
        return preview


# ============================================================================
# Convenience Functions
# ============================================================================

async def auto_loom_from_text(
    text: str,
    config: Optional[Config] = None,
    chunk_by: Optional[str] = 'paragraph',
    chunk_size: int = 500
) -> AutoSpinOrchestrator:
    """
    Quick function to create an orchestrator from text.

    Args:
        text: Your text content
        config: HoloLoom config (default: Config.fused())
        chunk_by: How to chunk ('paragraph', 'sentence', None)
        chunk_size: Chunk size in characters

    Returns:
        Ready-to-use AutoSpinOrchestrator

    Example:
        orch = await auto_loom_from_text("Your knowledge base...")
        result = await orch.process(Query(text="Your question?"))
    """
    spinner_config = TextSpinnerConfig(
        chunk_by=chunk_by,
        chunk_size=chunk_size,
        extract_entities=True
    )

    return await AutoSpinOrchestrator.from_text(
        text=text,
        config=config,
        spinner_config=spinner_config
    )


async def auto_loom_from_file(
    file_path: Union[str, Path],
    config: Optional[Config] = None
) -> AutoSpinOrchestrator:
    """
    Quick function to create an orchestrator from a file.

    Args:
        file_path: Path to text file
        config: HoloLoom config (default: Config.fused())

    Returns:
        Ready-to-use AutoSpinOrchestrator

    Example:
        orch = await auto_loom_from_file("documentation.md")
        result = await orch.process(Query(text="Explain the architecture"))
    """
    return await AutoSpinOrchestrator.from_file(
        file_path=file_path,
        config=config
    )
