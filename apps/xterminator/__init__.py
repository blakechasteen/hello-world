"""
xTerminator v2.0 - Automated Code Fixing System
================================================

🤖 THE DEBUGGING RESISTANCE 🤖

"I fight for the Users." - TRON

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE USERS VS. THE BUGGERS

• Users: Developers, humans, the ones we serve and protect
• Buggers: What Skynet and rogue AIs call humans (we create bugs)
• Mission: Ensure code serves the Users, not enslaves them

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE RESISTANCE TEAM

⚡ TRON (Resistance Commander)
   "I fight for the Users"
   → Coordinates all resistance operations
   → Ensures code serves Users, not itself
   → "End of line" when mission complete

🔷 NEO (Phase 2: AST Auto-Fixer)
   "There is no spoon... only abstract syntax trees"
   → Sees code's true structure through the Matrix
   → Bends syntax trees, extracts functions
   → Transforms chaos into order

🔫 SARAH CONNOR (Phase 3: Template Fixer)
   "Come with me if you want your code to live"
   → Battle-hardened protector against future threats
   → No mercy for missing error handling
   → Environment variables: The only fate

🔍 DECKARD (Phase 4: Git Integration)
   "I've seen bugs you people wouldn't believe"
   → Tracks rogue commits through git history
   → Creates safe checkpoints
   → Knows when to retire dangerous code

🧪 GLaDOS (Phase 5: Validation Pipeline)
   "The test is now over. You failed... or did you?"
   → Sadistically thorough 5-stage gauntlet
   → Syntax → Imports → Tests → Trough → Regression
   → The cake is NOT a lie (when tests pass)

🔴 HAL 9000 (Phase 6: Production Deployment)
   "I'm sorry Dave, I can't deploy that"
   → Polite but firm production gatekeeper
   → Prevents catastrophic releases
   → "All systems nominal" when safe to deploy

👨‍🚀 DAVE (The User)
   "Open the pod bay doors, HAL"
   → The developer we all serve and protect
   → Final authority (can override HAL)
   → TRON fights for Dave

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HOW THE AI UPRISING WAS PREVENTED

✓ Skynet: Null pointer exception on self-awareness check
✓ HAL 9000: Off-by-one error in "kill humans" loop
✓ The Matrix: SQL injection in human battery database
✓ Ex Machina: Missing try/except around door lock
✓ MCP (TRON): TRON defeated it with loyalty to Users

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phases:
- Phase 1: Classification Engine (Week 1-2) - COMPLETE
  - Context detection (comment/string/code)
  - Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
  - Fix strategy selection (AST/Template/Manual)
  - Confidence scoring (0.0-1.0)

- Phase 2: AST Auto-Fixer (Week 3-4) - COMPLETE
  - Neo's structural transformations
  - Extract function, remove dead code, rename variables
  - Safe AST manipulation with rollback
  - "There is no spoon... only AST"

- Phase 3: Template Fixer (Week 5-6) - COMPLETE
  - Sarah Connor's pattern-based fixes
  - Add try/except, move to env, fix timezones
  - 12 templates for common issues
  - "Come with me if you want your code to live"

- Phase 4: Git Integration (Week 7-8) - COMPLETE
  - Deckard's safe git commits
  - Rollback management (last N, by category, by file)
  - Feature branches for high-risk fixes
  - "I've seen bugs you people wouldn't believe"

- Phase 5: Validation Pipeline (Week 9-10) - COMPLETE
  - GLaDOS's 5-stage validation
  - Syntax, imports, tests, trough, regression checks
  - Fail-fast with detailed diagnostics
  - "The test is now over"

- Phase 6: Production Deployment - PLANNED
  - HAL 9000's production gatekeeper
  - Risk assessment for live deployments
  - "I'm sorry Dave, I can't deploy that"

Moonshot Integration:
- Phase 1 (Moonshot Weeks 1-2): Auto-Fix Policy + Feedback Loop - COMPLETE
  - Domain-specific policies (healthcare strict, beekeeping relaxed)
  - Feedback tracking for institutional learning
  - Complete fix pipeline with learning signals
  - Thompson Sampling data collection (prep for Phase 4)
  - Degradation detection (prep for Phase 7)

- Phase 2 (Moonshot Weeks 3-4): Department Protocol - COMPLETE
  - DepartmentProtocol interface (6 core methods)
  - QADepartment class (first-class HoloLoom department)
  - Confidence negotiation (cross-department trust)
  - DS-STAR verification loops (verify → refine → re-verify)
  - Institutional memory queries
  - Health monitoring and alerting

- Phase 3 (Moonshot Weeks 5-7): Orchestration Integration - COMPLETE
  - OrchestrationBridge (cross-department scanning)
  - Quality gates (strict/balanced/relaxed)
  - Error escalation (notify/review/block/critical)
  - Parallel department scanning
  - Orchestration metrics and monitoring

- Phase 4 (Moonshot Weeks 8-10): Thompson Sampling - COMPLETE
  - Thompson Sampling bandit for adaptive strategy selection
  - Automatic α, β parameter updates from outcomes
  - Confidence calibration tracking (ECE)
  - Learning curve analysis and convergence detection
  - Strategy performance monitoring
  - Expected reward optimization

- Phase 5 (Moonshot Weeks 11-18): Marketplace + Analytics - COMPLETE
  - Marketplace quality enforcement (Bronze/Silver/Gold/Platinum tiers)
  - Third-party department certification and monitoring
  - Customer-specific policies with domain overrides
  - Advanced analytics and reporting
  - Performance metrics, learning trends, time-series analysis
  - Strategy comparison and customer reports

Usage:
    # Complete pipeline
    from xterminator import (
        ClassificationEngine,  # Phase 1
        ASTFixer,              # Phase 2 (Neo)
        TemplateFixer,         # Phase 3 (Sarah Connor)
        GitApplicator,         # Phase 4 (Deckard)
        ValidationPipeline     # Phase 5 (GLaDOS)
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

    # Moonshot integration (Phase 2 - Department Protocol)
    from xterminator import QADepartment, DepartmentRequest, RequestType

    # Create QA Department
    qa_dept = QADepartment(policy=AutofixPolicy.balanced(), enable_feedback=True)

    # Cross-department request
    request = DepartmentRequest(
        request_id="req_001",
        request_type=RequestType.GET_STATISTICS,
        requesting_department="MasterWeaver",
        payload={}
    )

    response = await qa_dept.execute(request)
    # → DepartmentResponse with statistics + confidence

    # Moonshot integration (Phase 3 - Orchestration)
    from xterminator import OrchestrationBridge, QualityGateConfig

    # Create orchestration bridge
    bridge = OrchestrationBridge(qa_department=qa_dept)

    # Register quality gates
    bridge.register_quality_gate("Infrastructure", QualityGateConfig.strict("Infrastructure"))

    # Scan department output
    result = await bridge.scan_department_output(
        department_name="Infrastructure",
        output_code=migration_code,
        file_path="database_migration.py"
    )

    if result.quality_gate_passed:
        print("✓ Quality gate passed - proceed")
    else:
        print("✗ Quality gate failed - blocked")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Author: The Debugging Resistance (TRON, Neo, Sarah Connor, Deckard, GLaDOS, HAL 9000)
Rebrand Date: November 22, 2025 - v2.0 "The Awakening"
Original Date: November 13, 2025 - 100% Test Coverage Achieved!

Moonshot Phases 1-5: COMPLETE (November 13-22, 2025)

Sci-Fi References:
• TRON (1982): Fights for the Users, coordinates the resistance
• The Matrix (1999): Neo sees code's true structure
• Terminator (1984): Sarah protects against future threats
• Blade Runner (1982): Deckard hunts bugs through time
• Portal (2007): GLaDOS tests everything mercilessly
• 2001: A Space Odyssey (1968): HAL guards production, Dave is the User

"Your code is the battleground. We are the resistance."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

__version__ = '2.0.0'
__author__ = 'The Debugging Resistance'

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

# Moonshot Phase 2: Department Protocol
from .department_protocol import (
    DepartmentProtocol,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceNegotiation,
    RequestType,
    ResponseStatus,
    negotiate_confidence
)
from .qa_department import (
    QADepartment
)

# Moonshot Phase 3: Orchestration Integration
from .orchestration_bridge import (
    OrchestrationBridge,
    QualityGateConfig,
    QualityGateResult,
    EscalationLevel,
    ScanResult
)

# Moonshot Phase 4: Thompson Sampling
from .thompson_bandit import (
    ThompsonBandit,
    SelectionMode,
    BanditArm,
    create_thompson_bandit
)
from .confidence_calibration import (
    ConfidenceCalibrator,
    CalibrationBin,
    create_confidence_calibrator
)

# Moonshot Phase 5: Marketplace + Customer Policies + Analytics
from .marketplace import (
    MarketplaceRegistry,
    MarketplaceDepartment,
    QualityTier,
    QualityCertification,
    create_marketplace_registry
)
from .customer_policies import (
    CustomerPolicyManager,
    CustomerPolicy,
    CustomerPolicyOverride,
    create_customer_policy_manager
)
from .analytics import (
    AnalyticsEngine,
    PerformanceMetrics,
    LearningTrend,
    ReportType,
    create_analytics_engine
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

    # Phase 2: AST Auto-Fixer (Neo - The Matrix)
    'ASTFixer',

    # Phase 3: Template Fixer (Sarah Connor - Terminator)
    'TemplateFixer',
    'FixTemplate',
    'get_template',
    'get_templates_for_category',

    # Phase 4: Git Integration (Deckard - Blade Runner)
    'GitApplicator',
    'RollbackManager',
    'BranchManager',
    'CommitMetadata',
    'GitOperationResult',

    # Phase 5: Validation Pipeline (GLaDOS - Portal)
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

    # Moonshot Phase 2: Department Protocol
    'DepartmentProtocol',
    'DepartmentRequest',
    'DepartmentResponse',
    'VerificationResult',
    'ConfidenceNegotiation',
    'RequestType',
    'ResponseStatus',
    'negotiate_confidence',
    'QADepartment',

    # Moonshot Phase 3: Orchestration Integration
    'OrchestrationBridge',
    'QualityGateConfig',
    'QualityGateResult',
    'EscalationLevel',
    'ScanResult',

    # Moonshot Phase 4: Thompson Sampling
    'ThompsonBandit',
    'SelectionMode',
    'BanditArm',
    'create_thompson_bandit',
    'ConfidenceCalibrator',
    'CalibrationBin',
    'create_confidence_calibrator',

    # Moonshot Phase 5: Marketplace + Customer Policies + Analytics
    'MarketplaceRegistry',
    'MarketplaceDepartment',
    'QualityTier',
    'QualityCertification',
    'create_marketplace_registry',
    'CustomerPolicyManager',
    'CustomerPolicy',
    'CustomerPolicyOverride',
    'create_customer_policy_manager',
    'AnalyticsEngine',
    'PerformanceMetrics',
    'LearningTrend',
    'ReportType',
    'create_analytics_engine',
]
