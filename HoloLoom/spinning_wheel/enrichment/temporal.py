# -*- coding: utf-8 -*-
"""
Temporal Enricher
=================

Extracts temporal information:
- Absolute dates (October 13, 2025)
- Relative dates (today, yesterday, next week)
- Temporal sequences (first X, then Y)
- Seasonal context (spring, fall, winter)

Useful for:
- Timeline construction
- Seasonal pattern detection
- Event ordering

Created: 2025-10-13
Author: blakechasteen
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from .base import BaseEnricher, EnrichmentResult


class TemporalEnricher(BaseEnricher):
    """
    Extract temporal information from text.
    
    Capabilities:
    - Absolute date extraction
    - Relative date interpretation
    - Seasonal context
    - Temporal ordering
    
    Example:
        >>> enricher = TemporalEnricher(reference_date="2025-10-13")
        >>> result = await enricher.enrich("Yesterday I inspected the hives. Next week I'll add supers.")
        >>> result.data
        {
            'dates': ['2025-10-12', '2025-10-20'],
            'relative_terms': ['yesterday', 'next week'],
            'season': 'fall',
            'temporal_sequence': True
        }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Reference date (defaults to today)
        ref_date_str = self.config.get('reference_date')
        if ref_date_str:
            self.reference_date = datetime.strptime(ref_date_str, "%Y-%m-%d")
        else:
            self.reference_date = datetime.utcnow()
        
        # Month name to number mapping
        self.months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        # Relative date mappings
        self.relative_dates = {
            'today': 0,
            'tomorrow': 1,
            'yesterday': -1,
        }
    
    async def enrich(self, text: str) -> EnrichmentResult:
        """Extract temporal information from text."""
        
        dates = []
        relative_terms = []
        
        # Extract absolute dates
        dates.extend(self._extract_absolute_dates(text))
        
        # Extract relative dates
        rel_dates, rel_terms = self._extract_relative_dates(text)
        dates.extend(rel_dates)
        relative_terms.extend(rel_terms)
        
        # Determine season
        season = self._determine_season(dates)
        
        # Check for temporal sequences
        has_sequence = self._detect_temporal_sequence(text)
        
        data = {
            'dates': sorted(set(dates)),  # Deduplicate and sort
            'relative_terms': relative_terms,
            'season': season,
            'temporal_sequence': has_sequence,
            'reference_date': self.reference_date.strftime("%Y-%m-%d"),
        }
        
        confidence = 0.5 + (0.3 if dates else 0.0) + (0.2 if relative_terms else 0.0)
        
        return EnrichmentResult(
            enricher_type='temporal',
            data=data,
            confidence=min(confidence, 1.0),
            metadata={'num_dates': len(dates)}
        )
    
    def _extract_absolute_dates(self, text: str) -> List[str]:
        """Extract dates like 'October 13' or '10/13/2025'."""
        dates = []
        
        # Pattern: Month Day (e.g., "October 13")
        month_day_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})'
        for match in re.finditer(month_day_pattern, text, re.IGNORECASE):
            month_name = match.group(1).lower()
            day = int(match.group(2))
            month = self.months[month_name]
            year = self.reference_date.year
            
            try:
                date_obj = datetime(year, month, day)
                dates.append(date_obj.strftime("%Y-%m-%d"))
            except ValueError:
                pass  # Invalid date
        
        # Pattern: MM/DD/YYYY or M/D/YY
        numeric_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'
        for match in re.finditer(numeric_pattern, text):
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            
            # Handle 2-digit years
            if year < 100:
                year += 2000
            
            try:
                date_obj = datetime(year, month, day)
                dates.append(date_obj.strftime("%Y-%m-%d"))
            except ValueError:
                pass
        
        return dates
    
    def _extract_relative_dates(self, text: str) -> tuple[List[str], List[str]]:
        """Extract relative dates like 'today', 'next week'."""
        dates = []
        terms = []
        
        text_lower = text.lower()
        
        # Simple relative dates (today, tomorrow, yesterday)
        for term, offset in self.relative_dates.items():
            if term in text_lower:
                date_obj = self.reference_date + timedelta(days=offset)
                dates.append(date_obj.strftime("%Y-%m-%d"))
                terms.append(term)
        
        # Week offsets
        if 'next week' in text_lower:
            date_obj = self.reference_date + timedelta(days=7)
            dates.append(date_obj.strftime("%Y-%m-%d"))
            terms.append('next week')
        elif 'last week' in text_lower:
            date_obj = self.reference_date - timedelta(days=7)
            dates.append(date_obj.strftime("%Y-%m-%d"))
            terms.append('last week')
        
        return dates, terms
    
    def _determine_season(self, dates: List[str]) -> Optional[str]:
        """Determine season from extracted dates."""
        if not dates:
            # Use reference date
            month = self.reference_date.month
        else:
            # Use first extracted date
            first_date = datetime.strptime(dates[0], "%Y-%m-%d")
            month = first_date.month
        
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'
    
    def _detect_temporal_sequence(self, text: str) -> bool:
        """Check if text contains temporal ordering words."""
        sequence_markers = [
            'first', 'then', 'next', 'after', 'before', 
            'finally', 'later', 'earlier', 'following'
        ]
        
        text_lower = text.lower()
        return any(marker in text_lower for marker in sequence_markers)