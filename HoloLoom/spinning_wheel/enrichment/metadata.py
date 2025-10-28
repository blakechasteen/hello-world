# -*- coding: utf-8 -*-
"""
Metadata Enricher
=================

Extracts structured metadata from text:
- Tags and categories
- Priority levels
- Source types
- Document structure

Useful for:
- Task management (priority extraction)
- Content categorization
- Source tracking

Created: 2025-10-13
Author: blakechasteen
"""

import re
from typing import Dict, List, Any
from .base import BaseEnricher, EnrichmentResult


class MetadataEnricher(BaseEnricher):
    """
    Extract structured metadata from text.
    
    Capabilities:
    - Priority detection (high/medium/low, 1-5 scale)
    - Tag extraction (#hashtag, @mentions)
    - Category detection (by keywords)
    - Source type identification
    
    Example:
        >>> enricher = MetadataEnricher()
        >>> result = await enricher.enrich("HIGH PRIORITY: Order supplies #beekeeping @jodi")
        >>> result.data
        {
            'priority': 'high',
            'priority_score': 5,
            'tags': ['beekeeping'],
            'mentions': ['jodi'],
            'category': 'task'
        }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Default categories
        self.categories = self.config.get('categories', {
            'task': ['todo', 'task', 'action', 'order', 'buy', 'schedule'],
            'observation': ['inspected', 'noticed', 'saw', 'found', 'observed'],
            'measurement': ['temperature', 'weight', 'count', 'measured', 'frames'],
            'issue': ['problem', 'concern', 'warning', 'alert', 'issue', 'mites'],
        })
    
    async def enrich(self, text: str) -> EnrichmentResult:
        """Extract metadata from text."""
        
        data = {
            'priority': self._extract_priority(text),
            'priority_score': self._extract_priority_score(text),
            'tags': self._extract_tags(text),
            'mentions': self._extract_mentions(text),
            'category': self._extract_category(text),
            'has_date': self._has_date_reference(text),
        }
        
        # Calculate confidence based on how much we found
        confidence = sum([
            0.2 if data['priority'] != 'medium' else 0.0,
            0.3 if data['tags'] else 0.0,
            0.2 if data['category'] != 'general' else 0.0,
            0.3 if data['has_date'] else 0.0,
        ])
        
        return EnrichmentResult(
            enricher_type='metadata',
            data=data,
            confidence=min(confidence, 1.0),
            metadata={'text_length': len(text)}
        )
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority level."""
        text_lower = text.lower()
        
        # Explicit priority markers
        if re.search(r'\bhigh\s+priority\b|\burgent\b|!{2,}', text_lower):
            return 'high'
        elif re.search(r'\blow\s+priority\b|\boptional\b', text_lower):
            return 'low'
        
        # Numeric priority
        priority_match = re.search(r'priority[:\s]+([1-5])', text_lower)
        if priority_match:
            score = int(priority_match.group(1))
            if score >= 4:
                return 'high'
            elif score <= 2:
                return 'low'
        
        return 'medium'
    
    def _extract_priority_score(self, text: str) -> int:
        """Convert priority to 1-5 scale."""
        priority = self._extract_priority(text)
        return {'low': 2, 'medium': 3, 'high': 5}.get(priority, 3)
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract hashtags."""
        return re.findall(r'#(\w+)', text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions."""
        return re.findall(r'@(\w+)', text)
    
    def _extract_category(self, text: str) -> str:
        """Categorize text by keywords."""
        text_lower = text.lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return 'general'
    
    def _has_date_reference(self, text: str) -> bool:
        """Check if text contains date references."""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 10/13/2025
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',  # October 13
            r'\b(today|tomorrow|yesterday)\b',  # Relative dates
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Days
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False