"""
LLM API Endpoints
=================

FastAPI endpoints for multi-model LLM support.

Features:
- Model selection
- Model comparison (A/B testing)
- Cost tracking
- Available models listing

Created: 2025-01-20
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from hololoom.config import Config
from hololoom.llm import LLMConfig, UnifiedLLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# SECURITY: Allowed Models Whitelist
# =============================================================================

ALLOWED_MODELS = frozenset([
    # Ollama (local)
    "ollama/llama3.2:3b",
    "ollama/llama3.1:8b",
    "ollama/llama3.1:70b",
    "ollama/mistral:7b",
    "ollama/phi3:3.8b",
    "ollama/codellama:7b",
    "ollama/codellama:13b",
    # Anthropic
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022",
    "anthropic/claude-3-opus-20240229",
    # OpenAI
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
    # Google
    "google/gemini-pro",
    "google/gemini-1.5-pro",
])


def validate_model_override(model: str | None) -> str | None:
    """
    SECURITY: Validate model override against whitelist.

    Prevents arbitrary model specification which could:
    - Cause unexpected costs
    - Access unintended models
    - Enable prompt injection via model names

    Args:
        model: Model override string (e.g., "anthropic/claude-3-5-sonnet")

    Returns:
        Validated model string or None

    Raises:
        ValueError: If model not in whitelist
    """
    if model is None:
        return None

    # Normalize
    model_normalized = model.strip().lower()

    # Check whitelist (case-insensitive comparison)
    allowed_lower = {m.lower() for m in ALLOWED_MODELS}
    if model_normalized not in allowed_lower:
        raise ValueError(
            f"Model '{model}' not in allowed list. "
            f"Use /llm/models endpoint to see available models."
        )

    # Return original casing for matching model
    for allowed in ALLOWED_MODELS:
        if allowed.lower() == model_normalized:
            return allowed

    return model

router = APIRouter(prefix="/llm", tags=["llm"])

# Global LLM client instance
llm_client: UnifiedLLMClient | None = None


def get_llm_client(config: Config | None = None) -> UnifiedLLMClient:
    """
    Get or create LLM client instance.

    Args:
        config: Optional HoloLoom config (uses defaults if not provided)

    Returns:
        UnifiedLLMClient instance
    """
    global llm_client

    if llm_client is None:
        if config is None:
            config = Config.fast()

        # Build primary config from hololoom config
        primary = LLMConfig(
            provider=config.llm_primary_provider,
            model=config.llm_primary_model,
            timeout=config.llm_timeout
        )

        # Build fallback configs
        fallbacks = []
        if config.llm_enable_fallback:
            for provider, model in zip(config.llm_fallback_providers, config.llm_fallback_models):
                fallbacks.append(LLMConfig(
                    provider=provider,
                    model=model,
                    timeout=config.llm_timeout
                ))

        llm_client = UnifiedLLMClient(
            primary=primary,
            fallbacks=fallbacks,
            enable_cost_tracking=config.llm_enable_cost_tracking,
            max_cost_per_query=config.llm_max_cost_per_query
        )

        logger.info(f"✓ LLM client initialized: {primary.full_name}")

    return llm_client


# ============== REQUEST/RESPONSE MODELS ==============

class QueryRequest(BaseModel):
    """Request for LLM query"""
    # SECURITY: Limit query length to prevent resource exhaustion
    query: str = Field(..., min_length=1, max_length=100000, description="Query text (max 100KB)")
    max_tokens: int = Field(default=500, ge=1, le=4096, description="Max tokens (1-4096)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature (0.0-2.0)")
    model_override: str | None = Field(default=None, description="Model override (must be whitelisted)")
    system_prompt: str | None = Field(default=None, max_length=10000, description="System prompt (max 10KB)")

    @field_validator('model_override')
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        """SECURITY: Validate model against whitelist"""
        return validate_model_override(v)


class QueryResponse(BaseModel):
    """Response from LLM query"""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: dict[str, Any] | None = None


class CompareRequest(BaseModel):
    """Request for model comparison"""
    # SECURITY: Same limits as QueryRequest
    query: str = Field(..., min_length=1, max_length=100000, description="Query text (max 100KB)")
    models: list[str] = Field(..., min_length=1, max_length=5, description="Models to compare (max 5)")
    max_tokens: int = Field(default=500, ge=1, le=4096, description="Max tokens (1-4096)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature (0.0-2.0)")

    @field_validator('models')
    @classmethod
    def validate_models(cls, v: list[str]) -> list[str]:
        """SECURITY: Validate all models against whitelist"""
        validated = []
        for model in v:
            validated_model = validate_model_override(model)
            if validated_model:
                validated.append(validated_model)
        return validated


class CompareResponse(BaseModel):
    """Response from model comparison"""
    results: dict[str, QueryResponse]
    total_cost_usd: float
    winner: str | None = None  # Model with best response (if determinable)


class ModelInfo(BaseModel):
    """Information about an available model"""
    name: str  # e.g., "ollama/llama3.2:3b"
    provider: str
    model: str
    is_free: bool
    input_cost_per_1m: float  # USD per 1M tokens
    output_cost_per_1m: float
    available: bool


class CostStatsResponse(BaseModel):
    """Cost tracking statistics"""
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_cost_per_call: float
    by_model: dict[str, dict[str, Any]]


# ============== ENDPOINTS ==============

@router.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    """
    Query an LLM model.

    Uses primary model by default, or override with model_override parameter.

    Args:
        request: Query request with prompt and parameters

    Returns:
        Response with generated content and cost information
    """
    try:
        client = get_llm_client()

        # Build kwargs
        kwargs = {}
        if request.system_prompt:
            kwargs["system_prompt"] = request.system_prompt

        # Query LLM
        response = await client.complete(
            prompt=request.query,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            model_override=request.model_override,
            **kwargs
        )

        # Build response
        cost_usd = response.cost_estimate.total_cost_usd if response.cost_estimate else 0.0

        return QueryResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost_usd,
            metadata=response.metadata
        )

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed. Please try again or contact support.")


@router.post("/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Compare multiple models on the same prompt.

    Useful for A/B testing and model selection.

    Args:
        request: Comparison request with prompt and list of models

    Returns:
        Results from all models with cost breakdown
    """
    try:
        client = get_llm_client()

        # Compare models
        results = await client.compare_models(
            prompt=request.query,
            models=request.models,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Build response
        comparison_results = {}
        total_cost = 0.0

        for model_name, response in results.items():
            cost_usd = response.cost_estimate.total_cost_usd if response.cost_estimate else 0.0
            total_cost += cost_usd

            comparison_results[model_name] = QueryResponse(
                content=response.content,
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=cost_usd,
                metadata=response.metadata
            )

        # Determine "winner" (longest response as simple heuristic)
        winner = max(
            comparison_results.items(),
            key=lambda x: len(x[1].content)
        )[0] if comparison_results else None

        return CompareResponse(
            results=comparison_results,
            total_cost_usd=total_cost,
            winner=winner
        )

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model comparison failed. Please try again or contact support.")


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    """
    List all available LLM models.

    Returns:
        List of model information including pricing and availability
    """
    try:
        from hololoom.llm.cost_tracker import ModelPricing

        client = get_llm_client()
        available_models = client.list_available_models()

        # Build model info list
        models = []

        # Add all known models from pricing table
        for model_name, pricing in ModelPricing.PRICING.items():
            # Skip generic "ollama" entry
            if model_name == "ollama":
                continue

            # Determine provider
            if "claude" in model_name:
                provider = "anthropic"
            elif "gpt" in model_name:
                provider = "openai"
            elif "gemini" in model_name:
                provider = "google"
            else:
                provider = "ollama"

            full_name = f"{provider}/{model_name}"
            is_available = full_name in available_models

            models.append(ModelInfo(
                name=full_name,
                provider=provider,
                model=model_name,
                is_free=ModelPricing.is_free(model_name),
                input_cost_per_1m=pricing["input"],
                output_cost_per_1m=pricing["output"],
                available=is_available
            ))

        # Sort: available first, then by cost (free first, then cheapest)
        models.sort(key=lambda m: (
            not m.available,  # Available models first
            not m.is_free,    # Free models first within available
            m.input_cost_per_1m  # Cheapest first within paid
        ))

        return models

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models. Please try again or contact support.")


@router.get("/cost-stats", response_model=CostStatsResponse)
async def get_cost_stats():
    """
    Get cost tracking statistics.

    Returns:
        Statistics about token usage and costs across all queries
    """
    try:
        client = get_llm_client()
        stats = client.get_cost_statistics()

        if stats is None:
            raise HTTPException(
                status_code=503,
                detail="Cost tracking not enabled"
            )

        return CostStatsResponse(**stats)

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Failed to get cost stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get cost statistics. Please try again or contact support.")


@router.post("/cost-stats/reset")
async def reset_cost_stats():
    """
    Reset cost tracking statistics.

    Returns:
        Success message
    """
    try:
        client = get_llm_client()
        client.reset_cost_tracking()
        return {"status": "success", "message": "Cost statistics reset"}

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Failed to reset cost stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset cost statistics. Please try again or contact support.")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and available models count
    """
    try:
        client = get_llm_client()
        available = client.list_available_models()

        return {
            "status": "healthy",
            "available_models": len(available),
            "models": available
        }

    except Exception as e:
        # SECURITY: Log full error internally, return generic message to client
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": "Health check failed. Check server logs for details."
        }
