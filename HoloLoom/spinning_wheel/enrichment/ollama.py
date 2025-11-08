# -*- coding: utf-8 -*-
"""
Ollama Enrichment
=================
Lightweight context enrichment using local Ollama models.

Requires: pip install ollama (or: brew install ollama)
"""

import json
from typing import Dict, Any


class OllamaEnricher:
    """
    Use Ollama for lightweight pre-enrichment.
    
    Extracts:
    - Entities
    - Simple motifs/themes
    - Sentiment
    
    Usage:
        enricher = OllamaEnricher(model="llama3.2:3b")
        result = await enricher.extract_context("The hive is doing well...")
    """
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        
        try:
            import ollama
            self.client = ollama
        except ImportError:
            raise ImportError(
                "Ollama not installed. Install with: pip install ollama\n"
                "Or download from: https://ollama.ai"
            )
    
    async def extract_context(self, text: str) -> Dict[str, Any]:
        """
        Extract context from text using Ollama.
        
        Args:
            text: Input text
            
        Returns:
            Dict with entities, motifs, sentiment
        """
        prompt = f"""Extract structured information from this text:

Text: "{text}"

Respond in JSON format:
{{
  "entities": ["entity1", "entity2"],
  "motifs": ["theme1", "theme2"],
  "sentiment": "positive/neutral/negative"
}}

Keep it concise - max 3 entities and 2 motifs."""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                format="json"
            )
            
            # Parse JSON response
            result = json.loads(response['response'])
            return result
            
        except Exception as e:
            # Fallback if Ollama fails
            return {
                'entities': [],
                'motifs': [],
                'sentiment': 'neutral',
                'error': str(e)
            }