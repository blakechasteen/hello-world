"""
HoloLoom Research Assistant
===========================

Personal Research Assistant powered by HoloLoom's memory synthesis.

**Capabilities**:
- Multimodal PDF ingestion (text + images)
- Cross-paper pattern synthesis
- Contradiction detection between papers
- Knowledge gap identification
- Interactive Q&A chatbot

Author: HoloLoom Team
Date: 2025-11-17
"""

__version__ = "1.0.0"

# Graceful imports with fallbacks
try:
    from .pdf_ingestion import PDFIngestionPipeline, ResearchPaper
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from .paper_memory import PaperMemorySystem
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from .synthesis_engine import ResearchSynthesisEngine
    SYNTHESIS_AVAILABLE = True
except ImportError:
    SYNTHESIS_AVAILABLE = False

__all__ = [
    'PDFIngestionPipeline',
    'ResearchPaper',
    'PaperMemorySystem',
    'ResearchSynthesisEngine'
]
