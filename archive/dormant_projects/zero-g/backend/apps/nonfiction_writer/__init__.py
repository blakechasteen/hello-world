"""
Nonfiction Writing Toolkit for Zero-G

A comprehensive writing system leveraging HoloLoom's RAG, recursive refinement,
and agentic reasoning capabilities to support the full nonfiction writing workflow:

Research → Outline → Draft → Revise → Verify → Publish

Built on Zero-G's spaceflight metaphor:
- Preflight: Research gathering, source vetting
- Countdown: Outline planning, structure validation
- Liftoff: Initial draft generation
- Boost: Multi-pass refinement
- Orbit: Publication-ready document
- EVA: Expert manual editing

Phase 3 Advanced Features:
- Source credibility scoring (7-factor quality assessment)
- Zinsser scientific editing (clarity, brevity, strength)
- PDF export (Academic, Magazine, Book, Report templates)
- Advanced bibliography (DOI enrichment, BibTeX/EndNote export)
- Collaboration engine (multi-user editing with conflict resolution)
"""

# Phase 1: Core workflow
from .citation_manager import CitationManager, Citation, CitationStyle
from .research_phase import ResearchPhase, ResearchCorpus
from .outline_phase import OutlineGenerator, Outline, OutlineSection
from .draft_phase import DraftGenerator, Draft
from .revision_phase import RevisionEngine
from .verification_phase import VerificationPhase, VerificationReport

# Phase 3: Advanced features
from .source_credibility import (
    SourceCredibilityScorer,
    CredibilityMetrics,
    CredibilityLevel,
    create_credibility_scorer,
)
from .zinsser_editor import (
    ZinsserEditor,
    ZinsserReport,
    ZinsserPrinciple,
    EditSuggestion,
    create_zinsser_editor,
)
from .pdf_exporter import (
    PDFExporter,
    PDFTemplate,
    PDFMetadata,
    create_pdf_exporter,
)
from .advanced_bibliography import (
    AdvancedBibliographyManager,
    EnrichedCitation,
    create_advanced_bibliography_manager,
)
from .collaboration_engine import (
    CollaborationEngine,
    CollaborationSession,
    User,
    Change,
    Comment,
    Suggestion,
    ChangeType,
    MergeStrategy,
    create_collaboration_engine,
)

__all__ = [
    # Phase 1: Core workflow
    'CitationManager',
    'Citation',
    'CitationStyle',
    'ResearchPhase',
    'ResearchCorpus',
    'OutlineGenerator',
    'Outline',
    'OutlineSection',
    'DraftGenerator',
    'Draft',
    'RevisionEngine',
    'VerificationPhase',
    'VerificationReport',
    # Phase 3: Advanced features
    'SourceCredibilityScorer',
    'CredibilityMetrics',
    'CredibilityLevel',
    'create_credibility_scorer',
    'ZinsserEditor',
    'ZinsserReport',
    'ZinsserPrinciple',
    'EditSuggestion',
    'create_zinsser_editor',
    'PDFExporter',
    'PDFTemplate',
    'PDFMetadata',
    'create_pdf_exporter',
    'AdvancedBibliographyManager',
    'EnrichedCitation',
    'create_advanced_bibliography_manager',
    'CollaborationEngine',
    'CollaborationSession',
    'User',
    'Change',
    'Comment',
    'Suggestion',
    'ChangeType',
    'MergeStrategy',
    'create_collaboration_engine',
]
