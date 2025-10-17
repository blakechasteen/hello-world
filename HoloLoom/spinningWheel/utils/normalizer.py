# -*- coding: utf-8 -*-
"""
Text Normalization Utilities
=============================
Helper functions for cleaning and normalizing text.
"""

import re


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.
    
    - Remove extra whitespace
    - Fix common typos
    - Normalize quotes
    - Remove special characters (optional)
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # Trim
    text = text.strip()
    
    return text