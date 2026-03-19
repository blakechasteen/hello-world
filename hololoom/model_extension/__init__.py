"""
HoloLoom Model Extension SDK

Wrap any LLM with HoloLoom's memory and learning infrastructure.
Makes models smarter without retraining through:
- Persistent memory (knowledge graph + vector store)
- Continuous learning (Thompson Sampling adaptation)
- Domain adaptation (instant domain switching)
- Personalization (per-user memory graphs)

Quick Start:
    from hololoom.model_extension import MemoryAugmentedLLM

    async with MemoryAugmentedLLM(provider="anthropic") as llm:
        # Ingest domain knowledge
        await llm.learn("Thompson Sampling balances exploration and exploitation")

        # Query with memory-augmented context
        response = await llm.query("What is Thompson Sampling?")
        print(response.answer)
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Sources: {response.sources}")

Supported Providers:
    - anthropic (Claude)
    - openai (GPT-4, GPT-3.5)
    - ollama (local models)
    - google (Gemini)
    - cohere (Command)
"""

from .benchmark import (
    BenchmarkQuestion,
    BenchmarkResult,
    BenchmarkSummary,
    BenchmarkType,
    DomainBenchmark,
    EfficiencyBenchmark,
    HallucinationBenchmark,
    QualityBenchmark,
    RetentionBenchmark,
    generate_benchmark_report,
    run_benchmark_suite,
)
from .governance import (
    GovernanceEngine,
    PolicyDecision,
    PolicyTier,
)
from .providers import (
    AnthropicProvider,
    CohereProvider,
    GoogleProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)
from .uncertainty import (
    ConfidenceTier,
    UncertaintyEnvelope,
    UncertaintySource,
)
from .verification import (
    VerificationStatus,
    VerificationTier,
    verify_response,
)
from .wrapper import (
    LLMConfig,
    MemoryAugmentedLLM,
    MemoryAugmentedResponse,
)

__all__ = [
    # Core wrapper
    "MemoryAugmentedLLM",
    "MemoryAugmentedResponse",
    "LLMConfig",
    # Providers
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "GoogleProvider",
    "CohereProvider",
    "create_provider",
    # Uncertainty (from addendum A.1)
    "UncertaintyEnvelope",
    "ConfidenceTier",
    "UncertaintySource",
    # Verification (from addendum A.2)
    "VerificationStatus",
    "VerificationTier",
    "verify_response",
    # Governance (from addendum A.3)
    "PolicyTier",
    "PolicyDecision",
    "GovernanceEngine",
    # Benchmarking
    "BenchmarkType",
    "BenchmarkQuestion",
    "BenchmarkResult",
    "BenchmarkSummary",
    "QualityBenchmark",
    "EfficiencyBenchmark",
    "RetentionBenchmark",
    "HallucinationBenchmark",
    "DomainBenchmark",
    "run_benchmark_suite",
    "generate_benchmark_report",
]

__version__ = "1.0.0"
