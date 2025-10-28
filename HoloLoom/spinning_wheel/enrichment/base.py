# -*- coding: utf-8 -*-
"""
Base Enricher Interface
=======================

Abstract base class for all enrichment strategies.

All enrichers follow the same pattern:
1. Accept text/data
2. Extract specific type of information
3. Return EnrichmentResult

Created: 2025-10-13
Author: blakechasteen
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class EnrichmentResult:
    """
    Standardized result from an enricher.
    
    Attributes:
        enricher_type: Type of enrichment (e.g., 'semantic', 'temporal')
        data: Extracted data (format varies by enricher)
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata about the enrichment
    """
    enricher_type: str
    data: Dict[str, Any]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseEnricher(ABC):
    """
    Abstract base class for enrichment strategies.
    
    Subclasses must implement:
    - enrich(text) -> EnrichmentResult
    
    Optional:
    - batch_enrich(texts) -> List[EnrichmentResult]
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enricher with optional config.
        
        Args:
            config: Enricher-specific configuration
        """
        self.config = config or {}
    
    @abstractmethod
    async def enrich(self, text: str) -> EnrichmentResult:
        """
        Enrich a single text.
        
        Args:
            text: Input text to enrich
            
        Returns:
            EnrichmentResult with extracted data
        """
        pass
    
    async def batch_enrich(self, texts: List[str]) -> List[EnrichmentResult]:
        """
        Enrich multiple texts (default: sequential).
        
        Subclasses can override for parallel processing.
        
        Args:
            texts: List of texts to enrich
            
        Returns:
            List of EnrichmentResults
        """
        results = []
        for text in texts:
            result = await self.enrich(text)
            results.append(result)
        return results
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"