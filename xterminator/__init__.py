"""
xTerminator - Automated Code Fixing System
==========================================

🐷 The Complete Barnyard Brigade! 🐷

Phases:
- Phase 1: Classification Engine (Week 1-2) - COMPLETE
  - Context detection (comment/string/code)
  - Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
  - Fix strategy selection (AST/Template/Manual)
  - Confidence scoring (0.0-1.0)

- Phase 2: AST Auto-Fixer (Week 3-4) - COMPLETE
  - Wilbur's structural transformations
  - Extract function, remove dead code, rename variables
  - Safe AST manipulation with rollback

- Phase 3: Template Fixer (Week 5-6) - COMPLETE
  - Charlotte's pattern-based fixes
  - Add try/except, move to env, fix timezones
  - 12 templates for common issues

- Phase 4: Git Integration (Week 7-8) - COMPLETE
  - Templeton's safe git commits
  - Rollback management (last N, by category, by file)
  - Feature branches for high-risk fixes
  - Complete audit trail and metadata

- Phase 5: Validation Pipeline (Week 9-10) - COMPLETE
  - Fern's 5-stage validation
  - Syntax, imports, tests, trough, regression checks
  - Fail-fast with detailed diagnostics

Moonshot Integration:
- Phase 1 (Moonshot Weeks 1-2): Auto-Fix Policy + Feedback Loop - COMPLETE
  - Domain-specific policies (healthcare strict, beekeeping relaxed)
  - Feedback tracking for institutional learning
  - Complete fix pipeline with learning signals
  - Thompson Sampling data collection (prep for Phase 4)
  - Degradation detection (prep for Phase 7)

Usage:
    # Complete pipeline
    from xterminator import (
        ClassificationEngine,  # Phase 1
        ASTFixer,              # Phase 2 (Wilbur)
        TemplateFixer,         # Phase 3 (Charlotte)
        GitApplicator,         # Phase 4 (Templeton)
        ValidationPipeline     # Phase 5 (Fern)
    )

    # 1. Classify
    engine = ClassificationEngine()
    proposal = await engine.classify_and_propose(issue, code, file_path)

    # 2. Fix (AST or Template)
    if proposal.fix_strategy == FixStrategy.AST:
        fixer = ASTFixer()
    else:
        fixer = TemplateFixer()

    result = await fixer.fix_issue(proposal, code)
    if not result:
        return  # Fix failed

    fixed_code, diff = result

    # 3. Validate
    validator = ValidationPipeline()
    report = await validator.validate_fix(code, fixed_code, file_path)

    # 4. Commit (if valid)
    if validator.should_commit(report):
        applicator = GitApplicator()
        await applicator.apply_fix(file_path, fixed_code, proposal)

    # Moonshot integration (Phase 1)
    from xterminator import MoonshotOrchestrator, AutofixPolicy

    # Domain-specific policy
    policy = AutofixPolicy.conservative(domain='healthcare')

    # Complete pipeline with feedback tracking
    orchestrator = MoonshotOrchestrator(policy=policy, enable_feedback=True)
    result = await orchestrator.process_issue(issue, full_code, file_path)

    # View learning statistics
    stats = orchestrator.get_learning_statistics()

Author: The Barnyard Brigade (Wilbur, Charlotte, Templeton, Fern)
Date: November 13, 2025 - 100% Test Coverage Achieved!
Moonshot Phase 1: November 13, 2025 - Feedback Loop Complete!
"""

__version__ = '1.0.0-final'
__author__ = 'The Barnyard Brigade'

# Core type exports
from .xterminator_types import (
    RiskLevel,
    FixStrategy,
    CodeContext,
    ContextType,
    FixProposal,
    ClassificationResult
)

# Phase 1: Classification
from .context_detector import ContextDetector
from .risk_assessor import RiskAssessor
from .strategy_selector import StrategySelector
from .confidence_scorer import ConfidenceScorer
from .classification_engine import ClassificationEngine

# Phase 2: AST Auto-Fixer (Wilbur)
from .ast_fixer import ASTFixer

# Phase 3: Template Fixer (Charlotte)
from .template_fixer import TemplateFixer
from .templates import FixTemplate, get_template, get_templates_for_category

# Phase 4: Git Integration
from .git_applicator import (
    GitApplicator,
    RollbackManager,
    BranchManager,
    CommitMetadata,
    GitOperationResult
)

# Phase 5: Validation Pipeline
from .validator import (
    ValidationPipeline,
    ValidationPipelineReport,
    ValidationResult,
    ValidationStage,
    SyntaxValidator,
    ImportValidator,
    TestValidator,
    TroughValidator,
    RegressionValidator,
    validate_fix,
    quick_validate
)

# Moonshot Phase 1: Auto-Fix Policy & Feedback
from .autofix_policy import (
    AutofixPolicy,
    PolicyProfile,
    FixDecision,
    get_policy_for_domain,
    should_autofix_with_policy
)
from .feedback_tracker import (
    FeedbackTracker,
    FixAttempt,
    FixOutcome
)
from .moonshot_integration import (
    MoonshotOrchestrator,
    MoonshotResult
)

__all__ = [
    # Types
    'RiskLevel',
    'FixStrategy',
    'CodeContext',
    'ContextType',
    'FixProposal',
    'ClassificationResult',

    # Phase 1: Classification Components
    'ContextDetector',
    'RiskAssessor',
    'StrategySelector',
    'ConfidenceScorer',
    'ClassificationEngine',

    # Phase 2: AST Auto-Fixer (Wilbur)
    'ASTFixer',

    # Phase 3: Template Fixer (Charlotte)
    'TemplateFixer',
    'FixTemplate',
    'get_template',
    'get_templates_for_category',

    # Phase 4: Git Integration (Templeton)
    'GitApplicator',
    'RollbackManager',
    'BranchManager',
    'CommitMetadata',
    'GitOperationResult',

    # Phase 5: Validation Pipeline (Fern)
    'ValidationPipeline',
    'ValidationPipelineReport',
    'ValidationResult',
    'ValidationStage',
    'SyntaxValidator',
    'ImportValidator',
    'TestValidator',
    'TroughValidator',
    'RegressionValidator',
    'validate_fix',
    'quick_validate',

    # Moonshot Phase 1: Auto-Fix Policy & Feedback
    'AutofixPolicy',
    'PolicyProfile',
    'FixDecision',
    'get_policy_for_domain',
    'should_autofix_with_policy',
    'FeedbackTracker',
    'FixAttempt',
    'FixOutcome',
    'MoonshotOrchestrator',
    'MoonshotResult',
]
