#!/usr/bin/env python3
"""
Multi-Model Consensus Evaluator

Evaluates prompts by calling multiple LLMs and aggregating their scores.
Supports:
- Multiple LLM providers (OpenAI, Anthropic, local models)
- Various consensus strategies (voting, averaging, weighted)
- Disagreement detection and flagging
- Cost-optimized evaluation with caching
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import hashlib
import json
import time
import statistics
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptly.plugins.base import BaseEvaluator


class ConsensusStrategy(Enum):
    """Strategy for aggregating scores from multiple models"""
    MEAN = "mean"  # Simple average
    MEDIAN = "median"  # Median score
    WEIGHTED = "weighted"  # Weighted average based on model reliability
    MAJORITY_VOTE = "majority_vote"  # Majority voting (binary pass/fail)
    MIN = "min"  # Most conservative (lowest score)
    MAX = "max"  # Most optimistic (highest score)
    UNANIMOUS = "unanimous"  # All models must agree (binary)


class VotingStrategy(Enum):
    """Strategy for binary voting decisions"""
    SIMPLE_MAJORITY = "simple_majority"  # >50%
    SUPERMAJORITY = "supermajority"  # >=2/3
    UNANIMOUS = "unanimous"  # 100%


class ModelProvider:
    """Base class for LLM provider clients"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._cache = {}

    def evaluate(self, prompt: str, actual: str, expected: str) -> Dict[str, Any]:
        """
        Evaluate using this model provider

        Returns:
            Dict with 'score' (float 0-1) and optional 'reasoning' (str)
        """
        raise NotImplementedError


class OpenAIProvider(ModelProvider):
    """OpenAI GPT model provider"""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        super().__init__("openai", {"model": model, "api_key": api_key})
        self.model = model
        self.api_key = api_key
        self._client = None

    def _init_client(self):
        """Initialize OpenAI client"""
        if self._client is None:
            try:
                import openai
                self._client = openai
                if self.api_key:
                    self._client.api_key = self.api_key
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
                return False
        return True

    def evaluate(self, prompt: str, actual: str, expected: str) -> Dict[str, Any]:
        """Evaluate using OpenAI"""
        if not self._init_client():
            return {"score": 0.5, "reasoning": "OpenAI client not available", "error": True}

        try:
            eval_prompt = f"""{prompt}

Expected output:
{expected}

Actual output:
{actual}

Evaluate the actual output on a scale of 0.0 to 1.0.
Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}"""

            response = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.0
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": result.get("reasoning", ""),
                "model": self.model
            }
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "error": True}


class AnthropicProvider(ModelProvider):
    """Anthropic Claude model provider"""

    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        super().__init__("anthropic", {"model": model, "api_key": api_key})
        self.model = model
        self.api_key = api_key
        self._client = None

    def _init_client(self):
        """Initialize Anthropic client"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("Warning: anthropic package not installed. Install with: pip install anthropic")
                return False
        return True

    def evaluate(self, prompt: str, actual: str, expected: str) -> Dict[str, Any]:
        """Evaluate using Anthropic Claude"""
        if not self._init_client():
            return {"score": 0.5, "reasoning": "Anthropic client not available", "error": True}

        try:
            eval_prompt = f"""{prompt}

Expected output:
{expected}

Actual output:
{actual}

Evaluate the actual output on a scale of 0.0 to 1.0.
Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}"""

            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": eval_prompt}]
            )

            content = response.content[0].text
            result = json.loads(content)
            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": result.get("reasoning", ""),
                "model": self.model
            }
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "error": True}


class OllamaProvider(ModelProvider):
    """Local Ollama model provider"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        super().__init__("ollama", {"model": model, "base_url": base_url})
        self.model = model
        self.base_url = base_url

    def evaluate(self, prompt: str, actual: str, expected: str) -> Dict[str, Any]:
        """Evaluate using local Ollama model"""
        try:
            import requests

            eval_prompt = f"""{prompt}

Expected output:
{expected}

Actual output:
{actual}

Evaluate the actual output on a scale of 0.0 to 1.0.
Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}"""

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": eval_prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()["response"]
                # Try to extract JSON from response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                    return {
                        "score": float(result.get("score", 0.5)),
                        "reasoning": result.get("reasoning", ""),
                        "model": self.model
                    }

            return {"score": 0.5, "reasoning": "Failed to parse response", "error": True}

        except Exception as e:
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "error": True}


class MockProvider(ModelProvider):
    """Mock provider for testing (returns random but consistent scores)"""

    def __init__(self, name: str = "mock", bias: float = 0.7):
        super().__init__(name, {"bias": bias})
        self.bias = bias

    def evaluate(self, prompt: str, actual: str, expected: str) -> Dict[str, Any]:
        """Generate mock evaluation score"""
        # Use hash for consistency
        hash_value = hashlib.md5(f"{prompt}{actual}{expected}".encode()).hexdigest()
        score_offset = int(hash_value[:8], 16) / 0xFFFFFFFF
        score = (self.bias + score_offset * 0.3) % 1.0

        return {
            "score": score,
            "reasoning": f"Mock evaluation from {self.name}",
            "model": self.name
        }


class MultiModelConsensusEvaluator(BaseEvaluator):
    """
    Multi-model consensus evaluator.

    Calls multiple LLM providers and aggregates their scores using
    configurable consensus strategies.
    """

    def __init__(self,
                 providers: Optional[List[ModelProvider]] = None,
                 strategy: ConsensusStrategy = ConsensusStrategy.MEAN,
                 weights: Optional[Dict[str, float]] = None,
                 disagreement_threshold: float = 0.3,
                 cache_enabled: bool = True,
                 cache_ttl: int = 3600):
        """
        Initialize multi-model consensus evaluator

        Args:
            providers: List of model providers to use
            strategy: Consensus strategy for aggregation
            weights: Optional weights for each provider (for weighted strategy)
            disagreement_threshold: Threshold for flagging disagreements (std dev)
            cache_enabled: Enable response caching
            cache_ttl: Cache time-to-live in seconds
        """
        super().__init__(
            name="multi_model_consensus",
            description="Multi-model consensus evaluator with LLM aggregation"
        )

        self.providers = providers or [MockProvider("mock1"), MockProvider("mock2")]
        self.strategy = strategy
        self.weights = weights or {}
        self.disagreement_threshold = disagreement_threshold
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self._cache = {}

    def _cache_key(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        key_material = f"{actual}|{expected}|{context_str}"
        return hashlib.sha256(key_material.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached results"""
        if not self.cache_enabled:
            return None

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['results']
            else:
                del self._cache[cache_key]

        return None

    def _set_cached(self, cache_key: str, results: Dict[str, Any]):
        """Cache results"""
        if self.cache_enabled:
            self._cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }

    def _evaluate_with_providers(self, actual: str, expected: str,
                                 context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate with all providers

        Returns:
            List of evaluation results from each provider
        """
        # Get evaluation prompt from context
        prompt = context.get('evaluation_prompt', 'Evaluate the quality of this output.') if context else 'Evaluate the quality of this output.'

        results = []
        for provider in self.providers:
            result = provider.evaluate(prompt, actual, expected)
            result['provider'] = provider.name
            results.append(result)

        return results

    def _aggregate_scores(self, results: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate scores using configured strategy

        Returns:
            Tuple of (final_score, metadata)
        """
        # Filter out errors
        valid_results = [r for r in results if not r.get('error', False)]

        if not valid_results:
            return 0.5, {"error": "No valid results from providers"}

        scores = [r['score'] for r in valid_results]

        if self.strategy == ConsensusStrategy.MEAN:
            final_score = statistics.mean(scores)
        elif self.strategy == ConsensusStrategy.MEDIAN:
            final_score = statistics.median(scores)
        elif self.strategy == ConsensusStrategy.WEIGHTED:
            # Use weighted average
            total_weight = 0.0
            weighted_sum = 0.0
            for r in valid_results:
                weight = self.weights.get(r['provider'], 1.0)
                weighted_sum += r['score'] * weight
                total_weight += weight
            final_score = weighted_sum / max(1.0, total_weight)
        elif self.strategy == ConsensusStrategy.MIN:
            final_score = min(scores)
        elif self.strategy == ConsensusStrategy.MAX:
            final_score = max(scores)
        elif self.strategy == ConsensusStrategy.MAJORITY_VOTE:
            # Binary vote: score >= 0.5 is "pass"
            votes = [1 if s >= 0.5 else 0 for s in scores]
            final_score = 1.0 if sum(votes) > len(votes) / 2 else 0.0
        elif self.strategy == ConsensusStrategy.UNANIMOUS:
            # All must agree (all >= 0.5 or all < 0.5)
            votes = [1 if s >= 0.5 else 0 for s in scores]
            final_score = 1.0 if sum(votes) == len(votes) else 0.0

        # Calculate disagreement
        disagreement = statistics.stdev(scores) if len(scores) > 1 else 0.0

        metadata = {
            'individual_scores': scores,
            'disagreement': disagreement,
            'flagged': disagreement > self.disagreement_threshold,
            'strategy': self.strategy.value,
            'num_providers': len(valid_results)
        }

        return final_score, metadata

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using multi-model consensus

        Args:
            actual: Actual output
            expected: Expected output
            context: Optional context with evaluation_prompt

        Returns:
            float: Consensus score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Check cache
        cache_key = self._cache_key(actual, expected, context)
        cached = self._get_cached(cache_key)
        if cached:
            return cached['score']

        # Evaluate with all providers
        results = self._evaluate_with_providers(actual, expected, context)

        # Aggregate scores
        final_score, metadata = self._aggregate_scores(results)

        # Cache results
        self._set_cached(cache_key, {
            'score': final_score,
            'metadata': metadata,
            'results': results
        })

        return final_score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed consensus metrics"""
        if not actual:
            return {"score": 0.0, "error": "Empty input"}

        # Check cache
        cache_key = self._cache_key(actual, expected, context)
        cached = self._get_cached(cache_key)

        if cached:
            return {
                'score': cached['score'],
                **cached['metadata'],
                'individual_results': cached['results'],
                'cache_hit': True
            }

        # Evaluate with all providers
        results = self._evaluate_with_providers(actual, expected, context)

        # Aggregate scores
        final_score, metadata = self._aggregate_scores(results)

        # Cache results
        full_results = {
            'score': final_score,
            **metadata,
            'individual_results': results,
            'cache_hit': False
        }

        self._set_cached(cache_key, {
            'score': final_score,
            'metadata': metadata,
            'results': results
        })

        return full_results


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Multi-Model Consensus Evaluator Examples")
    print("=" * 80)

    # Example 1: Basic consensus with mock providers
    print("\n1. Basic Consensus (Mean Strategy)")
    print("-" * 80)

    providers = [
        MockProvider("model_a", bias=0.8),
        MockProvider("model_b", bias=0.7),
        MockProvider("model_c", bias=0.6)
    ]

    evaluator = MultiModelConsensusEvaluator(
        providers=providers,
        strategy=ConsensusStrategy.MEAN
    )

    actual = "This is a good response that addresses the question clearly."
    expected = "A clear and comprehensive response."

    metrics = evaluator.get_metrics(actual, expected)
    print(f"Final Score: {metrics['score']:.3f}")
    print(f"Individual Scores: {[f'{s:.3f}' for s in metrics['individual_scores']]}")
    print(f"Disagreement: {metrics['disagreement']:.3f}")
    print(f"Flagged: {metrics['flagged']}")

    # Example 2: Weighted consensus
    print("\n2. Weighted Consensus")
    print("-" * 80)

    evaluator_weighted = MultiModelConsensusEvaluator(
        providers=providers,
        strategy=ConsensusStrategy.WEIGHTED,
        weights={
            "model_a": 2.0,  # Trust this model more
            "model_b": 1.0,
            "model_c": 0.5
        }
    )

    metrics = evaluator_weighted.get_metrics(actual, expected)
    print(f"Final Score (Weighted): {metrics['score']:.3f}")
    print(f"Individual Scores: {[f'{s:.3f}' for s in metrics['individual_scores']]}")

    # Example 3: Disagreement detection
    print("\n3. Disagreement Detection")
    print("-" * 80)

    disagreeing_providers = [
        MockProvider("optimistic", bias=0.9),
        MockProvider("pessimistic", bias=0.3),
        MockProvider("moderate", bias=0.6)
    ]

    evaluator_disagreement = MultiModelConsensusEvaluator(
        providers=disagreeing_providers,
        strategy=ConsensusStrategy.MEDIAN,
        disagreement_threshold=0.2
    )

    metrics = evaluator_disagreement.get_metrics(actual, expected)
    print(f"Final Score (Median): {metrics['score']:.3f}")
    print(f"Individual Scores: {[f'{s:.3f}' for s in metrics['individual_scores']]}")
    print(f"Disagreement: {metrics['disagreement']:.3f}")
    print(f"Flagged for Review: {metrics['flagged']}")

    # Example 4: Majority voting
    print("\n4. Majority Voting Strategy")
    print("-" * 80)

    evaluator_voting = MultiModelConsensusEvaluator(
        providers=providers,
        strategy=ConsensusStrategy.MAJORITY_VOTE
    )

    metrics = evaluator_voting.get_metrics(actual, expected)
    print(f"Final Score (Majority Vote): {metrics['score']:.3f}")
    print(f"Individual Scores: {[f'{s:.3f}' for s in metrics['individual_scores']]}")
    print(f"Passed Majority Vote: {metrics['score'] >= 0.5}")

    # Example 5: Cache performance
    print("\n5. Cache Performance")
    print("-" * 80)

    evaluator_cached = MultiModelConsensusEvaluator(
        providers=providers,
        cache_enabled=True
    )

    # First call (cache miss)
    start = time.time()
    metrics1 = evaluator_cached.get_metrics(actual, expected)
    time1 = time.time() - start

    # Second call (cache hit)
    start = time.time()
    metrics2 = evaluator_cached.get_metrics(actual, expected)
    time2 = time.time() - start

    print(f"First call (cache miss): {time1*1000:.2f}ms - Cache hit: {metrics1['cache_hit']}")
    print(f"Second call (cache hit): {time2*1000:.2f}ms - Cache hit: {metrics2['cache_hit']}")
    print(f"Speedup: {time1/max(time2, 0.0001):.1f}x")

    print("\n" + "=" * 80)
