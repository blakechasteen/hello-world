"""
Backward Compatibility Shim for Old Spinner Pattern
====================================================

Provides the old BaseSpinner/SpinnerConfig pattern for legacy modality spinners.
New spinners should use the protocol from HoloLoom.spinningWheel.protocol instead.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from HoloLoom.documentation.types import MemoryShard


@dataclass
class SpinnerConfig:
    """
    Base configuration for spinners (legacy pattern).

    Attributes:
        enable_enrichment: Enable Ollama/Neo4j/mem0 pre-enrichment
        ollama_model: Ollama model for context extraction
        neo4j_conn: Neo4j connection (optional)
        mem0_client: mem0ai client (optional)
    """
    enable_enrichment: bool = False
    ollama_model: str = "llama3.2:3b"
    neo4j_conn: Any = None
    mem0_client: Any = None


class BaseSpinner(ABC):
    """
    Lightweight adapter that converts raw input -> MemoryShards (legacy pattern).

    Subclasses must implement:
    - spin(raw_data) -> List[MemoryShard]
    """

    def __init__(self, config: SpinnerConfig = None):
        self.config = config or SpinnerConfig()

    @abstractmethod
    async def spin(self, raw_data: Any) -> List[MemoryShard]:
        """
        Convert raw input to MemoryShards.

        Args:
            raw_data: Input data (format depends on spinner type)

        Returns:
            List of MemoryShards
        """
        ...
