"""
Metaprompt Integration for HoloLoom

Provides the programmatic API for the 7-component metaprompting framework.
Integrates universal CORE_TEMPLATE.md with model-specific adapters.

Architecture:
1. Load CORE_TEMPLATE.md (universal 7-component framework)
2. Auto-detect enhancement strategy based on request
3. Apply model-specific adapter (Claude/Gemini/GPT)
4. Return enhanced prompt ready for LLM

Example:
    >>> from hololoom.config import Config
    >>> from hololoom.prompting import create_metaprompt
    >>>
    >>> config = Config.fused()
    >>> config.llm_provider = "anthropic"
    >>>
    >>> enhanced = create_metaprompt(
    ...     request="Explain Thompson Sampling",
    ...     config=config
    ... )
    >>> # Returns: Core 7-component + Claude thinking tags + artifacts
"""

from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

from .adapters import create_adapter_from_config, ModelAdapter
from .strategy import StrategyContext
from .registry import suggest_strategies

logger = logging.getLogger(__name__)


def load_core_template() -> str:
    """
    Load universal 7-component metaprompt template.

    Returns:
        Core template content from CORE_TEMPLATE.md

    Example:
        >>> template = load_core_template()
        >>> print(template[:100])
        'You are a prompt engineering expert...'
    """
    # Navigate from hololoom/prompting/metaprompt.py -> promptly_skills/meta_prompt/CORE_TEMPLATE.md
    repo_root = Path(__file__).parent.parent.parent
    template_path = repo_root / "promptly_skills" / "meta_prompt" / "CORE_TEMPLATE.md"

    if not template_path.exists():
        logger.error(f"CORE_TEMPLATE.md not found at: {template_path}")
        raise FileNotFoundError(f"CORE_TEMPLATE.md not found at: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    logger.debug(f"Loaded CORE_TEMPLATE.md ({len(content)} chars)")
    return content


def auto_detect_strategy(request: str) -> Tuple[str, float]:
    """
    Auto-detect best prompting strategy for request.

    Uses the strategy registry's auto-detection to suggest the most
    appropriate enhancement strategy based on query characteristics.

    Args:
        request: User's casual request

    Returns:
        Tuple of (strategy_name, confidence)

    Example:
        >>> strategy, confidence = auto_detect_strategy(
        ...     "Analyze this security vulnerability"
        ... )
        >>> print(strategy, confidence)
        ('challenge', 0.92)
    """
    # Create context from request
    context = StrategyContext(query=request)

    # Get top suggestion from registry
    suggestions = suggest_strategies(context, top_k=1)

    if not suggestions:
        logger.warning("No strategy suggestions found, using default")
        return ("none", 0.0)

    strategy_name, confidence = suggestions[0]

    logger.info(f"Auto-detected strategy: {strategy_name} (confidence: {confidence:.2f})")
    return (strategy_name, confidence)


def create_metaprompt(
    request: str,
    config=None,
    features: Optional[Dict[str, bool]] = None,
    apply_adapter: bool = True
) -> str:
    """
    Create enhanced metaprompt from casual request.

    This is the main entry point for the metaprompting system. It:
    1. Loads the universal CORE_TEMPLATE.md
    2. Formats it with the user's request
    3. Optionally applies model-specific adapter enhancements

    Args:
        request: User's casual request (e.g., "help me prepare for meeting")
        config: Optional HoloLoom Config with llm_provider set
        features: Optional feature flags for adapter (e.g., {'thinking_tags': True})
        apply_adapter: Whether to apply model-specific adapter (default: True)

    Returns:
        Enhanced prompt ready for LLM

    Example:
        >>> from hololoom.config import Config
        >>> config = Config.fused()
        >>> config.llm_provider = "anthropic"
        >>>
        >>> enhanced = create_metaprompt(
        ...     "Explain Thompson Sampling",
        ...     config=config
        ... )
        >>> # Returns: Core 7-component framework + Claude optimizations
    """
    # Load core template
    core_template = load_core_template()

    # Format with user request
    core_prompt = core_template.replace(
        "{YOUR_CASUAL_REQUEST}",
        request
    )

    logger.info(f"Created core metaprompt for request: '{request[:50]}...'")

    # Apply model-specific adapter if requested
    if apply_adapter and config is not None:
        adapter = create_adapter_from_config(config)

        if adapter is not None:
            enhanced_prompt = adapter.enhance(core_prompt, features=features)
            logger.info(f"Applied {adapter.name} adapter enhancements")
            return enhanced_prompt
        else:
            logger.debug("No adapter available, returning core prompt")

    return core_prompt


def create_metaprompt_with_strategy(
    request: str,
    strategy_name: str,
    config=None,
    features: Optional[Dict[str, bool]] = None
) -> str:
    """
    Create metaprompt with explicit strategy selection.

    Combines metaprompt framework with a specific prompting strategy
    (e.g., 'verify', 'challenge', 'scaffold').

    Args:
        request: User's casual request
        strategy_name: Name of strategy to apply (e.g., 'verify', 'challenge')
        config: Optional HoloLoom Config with llm_provider set
        features: Optional feature flags for adapter

    Returns:
        Enhanced prompt with both metaprompt framework and strategy

    Example:
        >>> enhanced = create_metaprompt_with_strategy(
        ...     "Analyze this contract",
        ...     strategy_name="verify",
        ...     config=config
        ... )
        >>> # Returns: Metaprompt + verification strategy + adapter
    """
    from .registry import get_strategy

    # Create base metaprompt
    metaprompt = create_metaprompt(
        request=request,
        config=config,
        features=features,
        apply_adapter=True
    )

    # Get strategy
    strategy = get_strategy(strategy_name)

    if strategy is None:
        logger.warning(f"Strategy '{strategy_name}' not found, returning metaprompt only")
        return metaprompt

    # Apply strategy enhancement
    context = StrategyContext(query=request)
    import asyncio
    result = asyncio.run(strategy.enhance(context))

    # Combine metaprompt with strategy
    combined = f"{metaprompt}\n\n---\n\n{result.enhanced_query}"

    logger.info(f"Combined metaprompt with '{strategy_name}' strategy")
    return combined


def create_metaprompt_auto(
    request: str,
    config=None,
    features: Optional[Dict[str, bool]] = None,
    confidence_threshold: float = 0.7
) -> str:
    """
    Create metaprompt with auto-detected strategy.

    Automatically detects the best prompting strategy and applies it
    if confidence is above threshold.

    Args:
        request: User's casual request
        config: Optional HoloLoom Config with llm_provider set
        features: Optional feature flags for adapter
        confidence_threshold: Min confidence to apply strategy (default: 0.7)

    Returns:
        Enhanced prompt with metaprompt + optional strategy

    Example:
        >>> enhanced = create_metaprompt_auto(
        ...     "Analyze this security vulnerability",
        ...     config=config
        ... )
        >>> # Auto-detects 'challenge' strategy (confidence: 0.92) and applies it
    """
    # Auto-detect strategy
    strategy_name, confidence = auto_detect_strategy(request)

    logger.info(
        f"Auto-detected strategy: {strategy_name} "
        f"(confidence: {confidence:.2f}, threshold: {confidence_threshold})"
    )

    # Apply strategy if confidence is high enough
    if confidence >= confidence_threshold:
        return create_metaprompt_with_strategy(
            request=request,
            strategy_name=strategy_name,
            config=config,
            features=features
        )
    else:
        # Just use metaprompt without strategy
        return create_metaprompt(
            request=request,
            config=config,
            features=features,
            apply_adapter=True
        )


# Convenience function for common use case
def enhance_request(request: str, provider: str = "anthropic") -> str:
    """
    One-liner to enhance a casual request with metaprompting.

    This is the simplest API - just provide a request and provider.

    Args:
        request: User's casual request
        provider: LLM provider ('anthropic', 'google', 'openai')

    Returns:
        Enhanced prompt ready for LLM

    Example:
        >>> enhanced = enhance_request(
        ...     "Explain Thompson Sampling",
        ...     provider="anthropic"
        ... )
        >>> # Returns: Core + Claude adapter
    """
    from hololoom.config import Config

    config = Config.fused()
    config.llm_provider = provider

    return create_metaprompt(request, config=config)


# Export convenience functions
__all__ = [
    'create_metaprompt',
    'create_metaprompt_with_strategy',
    'create_metaprompt_auto',
    'auto_detect_strategy',
    'enhance_request'
]
