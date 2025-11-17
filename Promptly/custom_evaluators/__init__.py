"""
Custom Evaluator Examples for Promptly

This package contains real-world custom evaluator implementations demonstrating
how to extend Promptly with specialized evaluation logic:

1. Business Logic Evaluator - Industry-specific validation and scoring
2. Multi-Model Consensus Evaluator - LLM ensemble evaluation
3. Structured Output Evaluator - Schema validation for JSON/XML/YAML/SQL
4. Domain-Specific Evaluator - Medical/Legal/Code/Marketing evaluators
5. Human-in-the-Loop Evaluator - Queue-based human review workflows

Each evaluator follows the BaseEvaluator protocol and includes:
- Complete working implementation
- Comprehensive docstrings and type hints
- Configuration examples
- Test cases
- Real-world use case documentation
"""

from .business_logic import (
    BusinessLogicEvaluator,
    CustomerServiceEvaluator,
    FinancialComplianceEvaluator,
    MedicalAccuracyEvaluator
)

from .consensus import (
    MultiModelConsensusEvaluator,
    ConsensusStrategy,
    VotingStrategy
)

from .structured import (
    StructuredOutputEvaluator,
    JSONValidator,
    XMLValidator,
    YAMLValidator,
    SQLValidator,
    OpenAPIValidator
)

from .domain_specific import (
    DomainSpecificEvaluator,
    MedicalTerminologyEvaluator,
    LegalCitationEvaluator,
    CodeSecurityEvaluator,
    BrandVoiceEvaluator
)

from .hitl import (
    HumanInTheLoopEvaluator,
    ReviewQueue,
    ReviewStatus,
    InterAnnotatorAgreement
)

__all__ = [
    # Business Logic
    'BusinessLogicEvaluator',
    'CustomerServiceEvaluator',
    'FinancialComplianceEvaluator',
    'MedicalAccuracyEvaluator',

    # Consensus
    'MultiModelConsensusEvaluator',
    'ConsensusStrategy',
    'VotingStrategy',

    # Structured
    'StructuredOutputEvaluator',
    'JSONValidator',
    'XMLValidator',
    'YAMLValidator',
    'SQLValidator',
    'OpenAPIValidator',

    # Domain Specific
    'DomainSpecificEvaluator',
    'MedicalTerminologyEvaluator',
    'LegalCitationEvaluator',
    'CodeSecurityEvaluator',
    'BrandVoiceEvaluator',

    # Human-in-the-Loop
    'HumanInTheLoopEvaluator',
    'ReviewQueue',
    'ReviewStatus',
    'InterAnnotatorAgreement'
]

__version__ = '1.0.0'
