"""
Promptly LLM Judge - Prompt Evaluation System

Features:
- 12 evaluation criteria
- 6 judging methods (SINGLE_SCORE, CHAIN_OF_THOUGHT, PAIRWISE, etc.)
- Claude + Ollama backends
- Quality scoring and feedback
- Constitutional AI evaluation (Anthropic-style)
- G-Eval methodology (Microsoft Research)
"""

from .llm_judge import (
    # Enums
    JudgeCriteria,
    JudgingMethod,

    # Data classes
    JudgeConfig,
    JudgeScore,
    JudgeResult,

    # Main class
    LLMJudge,

    # Preset configurations
    get_quality_config,
    get_safety_config,
    get_creative_config,
    get_comprehensive_config,
    get_constitutional_config,
    get_geval_config,
)

__all__ = [
    # Enums
    'JudgeCriteria',
    'JudgingMethod',

    # Data classes
    'JudgeConfig',
    'JudgeScore',
    'JudgeResult',

    # Main class
    'LLMJudge',

    # Preset configurations
    'get_quality_config',
    'get_safety_config',
    'get_creative_config',
    'get_comprehensive_config',
    'get_constitutional_config',
    'get_geval_config',
]
