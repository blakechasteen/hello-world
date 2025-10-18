# -*- coding: utf-8 -*-
"""
Semantic Enricher
=================

Extracts semantic information using LLMs (Ollama):
- Named entities (people, places, hives, etc.)
- Domain-specific motifs (INSPECTION, TREATMENT, etc.)
- Sentiment (positive, neutral, negative)
- Key concepts and themes

Uses local Ollama models for privacy and speed.

Created: 2025-10-13
Author: blakechasteen
"""

import json
import re
from typing import Dict, List, Any, Optional
from .base import BaseEnricher, EnrichmentResult


class SemanticEnricher(BaseEnricher):
    """
    Extract semantic information using Ollama.
    
    Capabilities:
    - Entity extraction (hives, people, locations, dates)
    - Motif detection (INSPECTION, TREATMENT, HARVEST, etc.)
    - Sentiment analysis
    - Concept extraction
    
    Example:
        >>> enricher = SemanticEnricher(model="llama3.2:3b")
        >>> result = await enricher.enrich("Hive Jodi looks strong with good brood pattern.")
        >>> result.data
        {
            'entities': ['Hive Jodi', 'brood'],
            'motifs': ['INSPECTION', 'HEALTH_POSITIVE'],
            'sentiment': 'positive',
            'concepts': ['colony strength', 'brood development']
        }
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.model = self.config.get('model', 'llama3.2:3b')
        self.use_ollama = self.config.get('use_ollama', True)
        
        # Initialize Ollama client if enabled
        if self.use_ollama:
            try:
                import ollama
                self.client = ollama
            except ImportError:
                print("Warning: Ollama not installed. Falling back to regex-based extraction.")
                self.use_ollama = False
        
        # Fallback: regex patterns for common beekeeping entities
        self.entity_patterns = {
            'hive': r'\b(hive\s+\w+|\w+\s+hive)\b',
            'treatment': r'\b(thymol|oxalic|formic|apivar|apiguard)\b',
            'pollen': r'\b(goldenrod|clover|dandelion|aster)\b',
            'equipment': r'\b(frames?|super|box|feeder)\b',
        }
    
    async def enrich(self, text: str) -> EnrichmentResult:
        """Extract semantic information from text."""
        
        if self.use_ollama:
            data = await self._enrich_with_ollama(text)
        else:
            data = self._enrich_with_regex(text)
        
        return EnrichmentResult(
            enricher_type='semantic',
            data=data,
            confidence=data.get('confidence', 0.7),
            metadata={'model': self.model if self.use_ollama else 'regex'}
        )
    
    async def _enrich_with_ollama(self, text: str) -> Dict[str, Any]:
        """Use Ollama for semantic extraction."""
        
        prompt = f"""Extract structured information from this text about beekeeping:

Text: "{text}"

Respond in JSON format:
{{
  "entities": ["entity1", "entity2"],
  "motifs": ["MOTIF1", "MOTIF2"],
  "sentiment": "positive/neutral/negative",
  "concepts": ["concept1", "concept2"]
}}

Entities: hive names, people, locations, treatments, equipment
Motifs: INSPECTION, TREATMENT, HARVEST, FEEDING, ISSUE, OBSERVATION, TASK
Sentiment: overall tone (positive/neutral/negative)
Concepts: key themes or ideas

Keep it concise - max 5 entities, 3 motifs, 3 concepts."""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json"
            )
            
            result = json.loads(response['response'])
            result['confidence'] = 0.85
            return result
            
        except Exception as e:
            print(f"Ollama enrichment failed: {e}. Falling back to regex.")
            return self._enrich_with_regex(text)
    
    def _enrich_with_regex(self, text: str) -> Dict[str, Any]:
        """Fallback: regex-based entity extraction."""
        
        entities = []
        motifs = []
        
        # Extract entities
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Detect motifs by keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ['inspect', 'check', 'look']):
            motifs.append('INSPECTION')
        if any(word in text_lower for word in ['treatment', 'treat', 'thymol', 'oxalic']):
            motifs.append('TREATMENT')
        if any(word in text_lower for word in ['harvest', 'extract', 'honey']):
            motifs.append('HARVEST')
        if any(word in text_lower for word in ['feed', 'sugar', 'syrup']):
            motifs.append('FEEDING')
        if any(word in text_lower for word in ['problem', 'issue', 'concern', 'mites']):
            motifs.append('ISSUE')
        if any(word in text_lower for word in ['order', 'buy', 'task', 'todo']):
            motifs.append('TASK')
        
        # Simple sentiment
        positive_words = ['good', 'strong', 'healthy', 'active', 'excellent']
        negative_words = ['weak', 'poor', 'problem', 'mites', 'dead']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
        elif neg_count > pos_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'entities': list(set(entities))[:5],  # Deduplicate and limit
            'motifs': list(set(motifs)),
            'sentiment': sentiment,
            'concepts': [],  # Regex can't extract concepts well
            'confidence': 0.6  # Lower confidence for regex
        }