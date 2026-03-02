"""
Enhanced Text Processor

Processes text with NER, sentiment analysis, topic extraction, and keyphrase detection.
"""

import re
import time
from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .protocol import (
    InputProcessorProtocol,
    ProcessedInput,
    ModalityType,
    TextFeatures,
    InputMetadata
)


class TextProcessor:
    """
    Enhanced text processor with NLP features.
    
    Features:
    - Named Entity Recognition (NER)
    - Sentiment analysis
    - Topic extraction
    - Keyphrase extraction
    - Language detection
    """
    
    def __init__(self, embedder=None, use_spacy: bool = True, use_textblob: bool = True):
        """
        Initialize text processor.
        
        Args:
            embedder: Embedding model (MatryoshkaEmbeddings or similar)
            use_spacy: Whether to use spaCy for NER (requires spacy + model)
            use_textblob: Whether to use TextBlob for sentiment
        """
        # If no embedder provided, create simple TF-IDF fallback
        if embedder is None:
            from .simple_embedder import SimpleEmbedder
            self.embedder = SimpleEmbedder()
        else:
            self.embedder = embedder
        
        self.use_spacy = use_spacy
        self.use_textblob = use_textblob
        
        # Try to load optional dependencies
        self.spacy_nlp = None
        if use_spacy:
            try:
                import spacy
                self.spacy_nlp = spacy.load('en_core_web_sm')
            except (ImportError, OSError):
                print("Warning: spaCy not available. NER disabled.")
                self.use_spacy = False
        
        self.textblob = None
        if use_textblob:
            try:
                from textblob import TextBlob
                self.textblob = TextBlob
            except ImportError:
                print("Warning: TextBlob not available. Sentiment disabled.")
                self.use_textblob = False
    
    async def process(
        self,
        input_data: Union[str, Path, Dict],
        extract_entities: bool = True,
        extract_sentiment: bool = True,
        extract_topics: bool = True,
        extract_keyphrases: bool = True,
        **kwargs
    ) -> ProcessedInput:
        """
        Process text input with NLP features.
        
        Args:
            input_data: Text string, file path, or dict with 'text' key
            extract_entities: Whether to extract entities
            extract_sentiment: Whether to analyze sentiment
            extract_topics: Whether to extract topics
            extract_keyphrases: Whether to extract keyphrases
        
        Returns:
            ProcessedInput with text features
        """
        start_time = time.time()
        
        # Extract text from input
        if isinstance(input_data, dict):
            text = input_data.get('text', '')
            source = input_data.get('source')
        elif isinstance(input_data, Path):
            with open(input_data, 'r', encoding='utf-8') as f:
                text = f.read()
            source = str(input_data)
        else:
            text = str(input_data)
            source = None
        
        # Clean text
        text = self._clean_text(text)
        
        # Create features
        features = TextFeatures()
        
        # Extract entities
        if extract_entities and self.use_spacy and self.spacy_nlp:
            features.entities = self._extract_entities(text)
        
        # Analyze sentiment
        if extract_sentiment and self.use_textblob:
            features.sentiment = self._analyze_sentiment(text)
        
        # Extract topics
        if extract_topics:
            features.topics = self._extract_topics(text)
        
        # Extract keyphrases
        if extract_keyphrases:
            features.keyphrases = self._extract_keyphrases(text)
        
        # Generate embedding
        embedding = None
        if self.embedder:
            embedding = self.embedder.encode(text)
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text, features)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedInput(
            modality=ModalityType.TEXT,
            content=text[:500],  # First 500 chars
            embedding=embedding,
            confidence=confidence,
            source=source,
            features={'text': features}
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        return text.strip()
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy."""
        if not self.spacy_nlp:
            return []
        
        doc = self.spacy_nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        if not self.textblob:
            return {'polarity': 0.0, 'subjectivity': 0.5}
        
        try:
            blob = self.textblob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception:
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text.
        
        Simple keyword-based approach. For production, use LDA or BERT.
        """
        # Common topic indicators
        topic_keywords = {
            'technology': ['computer', 'software', 'hardware', 'digital', 'tech', 'AI', 'machine learning'],
            'business': ['company', 'market', 'business', 'corporate', 'finance', 'economy'],
            'science': ['research', 'study', 'experiment', 'theory', 'scientific', 'discovery'],
            'health': ['health', 'medical', 'disease', 'patient', 'treatment', 'doctor'],
            'politics': ['government', 'political', 'election', 'policy', 'law', 'vote'],
            'education': ['school', 'student', 'learning', 'education', 'teacher', 'university'],
            'sports': ['game', 'player', 'team', 'sport', 'match', 'competition'],
            'entertainment': ['movie', 'music', 'show', 'entertainment', 'actor', 'artist']
        }
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]  # Top 3 topics
    
    def _extract_keyphrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text.
        
        Simple noun phrase extraction. For production, use YAKE or KeyBERT.
        """
        if not self.spacy_nlp:
            # Fallback: extract capitalized phrases
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return list(set(capitalized))[:5]
        
        doc = self.spacy_nlp(text)
        
        # Extract noun chunks
        noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Filter by length and frequency
        keyphrases = []
        for chunk in noun_chunks:
            if 2 <= len(chunk.split()) <= 4:  # 2-4 words
                keyphrases.append(chunk)
        
        # Return unique keyphrases
        return list(set(keyphrases))[:10]
    
    def _calculate_confidence(self, text: str, features: TextFeatures) -> float:
        """Calculate processing confidence."""
        confidence = 1.0
        
        # Reduce confidence for very short text
        if len(text) < 10:
            confidence *= 0.5
        
        # Reduce confidence if no entities found in long text
        if len(text) > 100 and not features.entities:
            confidence *= 0.9
        
        # Reduce confidence for low-quality text (lots of special chars)
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        if special_char_ratio > 0.3:
            confidence *= 0.8
        
        return confidence
    
    def get_modality(self) -> ModalityType:
        """Return modality type."""
        return ModalityType.TEXT
    
    def is_available(self) -> bool:
        """Check if processor is available."""
        return True  # Text processing always available
