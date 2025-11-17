#!/usr/bin/env python3
"""
LLM-Based Evaluator Plugins

Uses Large Language Models as judges to evaluate prompt outputs.
Supports OpenAI (GPT-4), Anthropic (Claude), and local LLMs (Ollama).
"""

from typing import Dict, Any, Optional, List
import hashlib
import json
import time
from ..base import BaseEvaluator


class LLMJudgeEvaluator(BaseEvaluator):
    """
    Base class for LLM-as-judge evaluators.

    Uses an LLM to evaluate output quality based on configurable criteria.
    Includes caching to avoid redundant API calls.
    """

    def __init__(self, name: str, description: str, criteria: Optional[List[str]] = None,
                 rubric: Optional[str] = None, cache_size: int = 100):
        """
        Initialize LLM judge evaluator

        Args:
            name: Evaluator name
            description: Evaluator description
            criteria: List of evaluation criteria (e.g., ["accuracy", "clarity", "completeness"])
            rubric: Custom rubric/instructions for the LLM judge
            cache_size: Maximum number of cached evaluations
        """
        super().__init__(name, description)
        self.criteria = criteria or ["accuracy", "relevance", "clarity"]
        self.rubric = rubric or self._default_rubric()
        self._cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _default_rubric(self) -> str:
        """Default evaluation rubric"""
        return """Evaluate the following output on a scale of 0.0 to 1.0 based on these criteria:
{criteria}

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<brief explanation>"}}"""

    def _cache_key(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for an evaluation"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        key_material = f"{actual}|{expected}|{context_str}|{self.rubric}"
        return hashlib.sha256(key_material.encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[float]:
        """Get cached evaluation result"""
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]['score']
        self._cache_misses += 1
        return None

    def _set_cached(self, cache_key: str, score: float, reasoning: str = ""):
        """Cache evaluation result"""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = {
            'score': score,
            'reasoning': reasoning,
            'timestamp': time.time()
        }

    def _format_prompt(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format the evaluation prompt for the LLM"""
        criteria_text = "\n".join(f"- {criterion}" for criterion in self.criteria)
        prompt = self.rubric.format(criteria=criteria_text)

        prompt += f"\n\nExpected output:\n{expected}\n\nActual output:\n{actual}"

        if context and 'instructions' in context:
            prompt += f"\n\nAdditional instructions:\n{context['instructions']}"

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract score and reasoning"""
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                score = float(data.get('score', 0.5))
                reasoning = data.get('reasoning', '')
                return {'score': max(0.0, min(1.0, score)), 'reasoning': reasoning}
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: look for score patterns
            import re
            score_match = re.search(r'score["\s:]+([0-9.]+)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                return {'score': max(0.0, min(1.0, score)), 'reasoning': response}

        # Default fallback
        return {'score': 0.5, 'reasoning': 'Failed to parse LLM response'}

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _call_llm()")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate using LLM as judge

        Args:
            actual: The actual output text
            expected: The expected output or reference
            context: Optional context with additional instructions

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Check cache first
        cache_key = self._cache_key(actual, expected, context)
        cached_score = self._get_cached(cache_key)
        if cached_score is not None:
            return cached_score

        # Format prompt and call LLM
        prompt = self._format_prompt(actual, expected, context)

        try:
            response = self._call_llm(prompt)
            result = self._parse_llm_response(response)
            score = result['score']
            reasoning = result['reasoning']

            # Cache the result
            self._set_cached(cache_key, score, reasoning)

            return score

        except Exception as e:
            print(f"Warning: LLM evaluation failed ({e}), returning 0.5")
            return 0.5

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed metrics including reasoning"""
        cache_key = self._cache_key(actual, expected, context)

        # If cached, return cached metrics
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return {
                'score': cached['score'],
                'reasoning': cached['reasoning'],
                'cached': True,
                'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            }

        # Otherwise evaluate and return
        score = self.evaluate(actual, expected, context)
        cached = self._cache.get(cache_key, {})

        return {
            'score': score,
            'reasoning': cached.get('reasoning', ''),
            'criteria': self.criteria,
            'cached': False,
            'cache_hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        }


class OpenAIEvaluator(LLMJudgeEvaluator):
    """
    OpenAI GPT-4 as judge evaluator.

    Uses GPT-4 to evaluate prompt outputs with sophisticated reasoning.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview",
                 criteria: Optional[List[str]] = None, rubric: Optional[str] = None,
                 temperature: float = 0.1, cache_size: int = 100):
        """
        Initialize OpenAI evaluator

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo)
            criteria: Evaluation criteria
            rubric: Custom rubric
            temperature: Sampling temperature (lower = more consistent)
            cache_size: Maximum cached evaluations
        """
        super().__init__(
            name="openai_judge",
            description=f"OpenAI {model} as evaluation judge",
            criteria=criteria,
            rubric=rubric,
            cache_size=cache_size
        )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        """Initialize OpenAI client"""
        try:
            import openai
            import os

            # Use provided API key or environment variable
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key parameter.")
                return False

            # Initialize client (supports both openai 0.x and 1.x)
            try:
                # OpenAI 1.x
                self._client = openai.OpenAI(api_key=api_key)
            except AttributeError:
                # OpenAI 0.x
                openai.api_key = api_key
                self._client = openai

            return True

        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
            return False

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not self._available:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Try OpenAI 1.x API first
            if hasattr(self._client, 'chat'):
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide objective, consistent assessments."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content
            else:
                # Fallback to OpenAI 0.x API
                response = self._client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide objective, consistent assessments."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return response['choices'][0]['message']['content']

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


class AnthropicEvaluator(LLMJudgeEvaluator):
    """
    Anthropic Claude as judge evaluator.

    Uses Claude to evaluate prompt outputs with nuanced reasoning.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022",
                 criteria: Optional[List[str]] = None, rubric: Optional[str] = None,
                 temperature: float = 0.1, cache_size: int = 100):
        """
        Initialize Anthropic evaluator

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model name (claude-3-5-sonnet-20241022, claude-3-opus-20240229, etc.)
            criteria: Evaluation criteria
            rubric: Custom rubric
            temperature: Sampling temperature
            cache_size: Maximum cached evaluations
        """
        super().__init__(
            name="anthropic_judge",
            description=f"Anthropic {model} as evaluation judge",
            criteria=criteria,
            rubric=rubric,
            cache_size=cache_size
        )
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        """Initialize Anthropic client"""
        try:
            import anthropic
            import os

            # Use provided API key or environment variable
            api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("Warning: Anthropic API key not provided. Set ANTHROPIC_API_KEY env var or pass api_key parameter.")
                return False

            self._client = anthropic.Anthropic(api_key=api_key)
            return True

        except ImportError:
            print("Warning: anthropic package not installed. Install with: pip install anthropic")
            return False

    def _call_llm(self, prompt: str) -> str:
        """Call Anthropic API"""
        if not self._available:
            raise RuntimeError("Anthropic client not initialized")

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=self.temperature,
                system="You are an expert evaluator. Provide objective, consistent assessments.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text

        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")


class OllamaEvaluator(LLMJudgeEvaluator):
    """
    Local LLM evaluator using Ollama.

    Uses locally-hosted LLMs via Ollama for private, offline evaluation.
    """

    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434",
                 criteria: Optional[List[str]] = None, rubric: Optional[str] = None,
                 temperature: float = 0.1, cache_size: int = 100):
        """
        Initialize Ollama evaluator

        Args:
            model: Ollama model name (llama3, mistral, mixtral, etc.)
            host: Ollama server URL
            criteria: Evaluation criteria
            rubric: Custom rubric
            temperature: Sampling temperature
            cache_size: Maximum cached evaluations
        """
        super().__init__(
            name="ollama_judge",
            description=f"Local LLM ({model}) via Ollama as evaluation judge",
            criteria=criteria,
            rubric=rubric,
            cache_size=cache_size
        )
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        """Initialize Ollama client"""
        try:
            import ollama
            self._client = ollama

            # Test connection
            try:
                models = self._client.list()
                available_models = [m['name'] for m in models.get('models', [])]
                if self.model not in available_models and f"{self.model}:latest" not in available_models:
                    print(f"Warning: Model {self.model} not found in Ollama. Available models: {available_models}")
                    print(f"Pull the model with: ollama pull {self.model}")
                return True
            except Exception as e:
                print(f"Warning: Could not connect to Ollama at {self.host}. Is Ollama running? ({e})")
                return False

        except ImportError:
            print("Warning: ollama package not installed. Install with: pip install ollama")
            return False

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama API"""
        if not self._available:
            raise RuntimeError("Ollama client not initialized or server not accessible")

        try:
            # Format system prompt
            full_prompt = f"""You are an expert evaluator. Provide objective, consistent assessments.

{prompt}"""

            response = self._client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 500
                }
            )

            return response['response']

        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")


class LLMPairwiseEvaluator(BaseEvaluator):
    """
    Pairwise comparison evaluator using LLM.

    Compares two outputs and determines which is better.
    Useful for A/B testing and preference learning.
    """

    def __init__(self, llm_evaluator: LLMJudgeEvaluator):
        """
        Initialize pairwise evaluator

        Args:
            llm_evaluator: Base LLM evaluator to use for comparisons
        """
        super().__init__(
            name="llm_pairwise",
            description="LLM-based pairwise comparison evaluator"
        )
        self.llm_evaluator = llm_evaluator

    def compare(self, output_a: str, output_b: str, reference: str,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare two outputs pairwise

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            reference: Reference/expected output
            context: Optional context

        Returns:
            Dict with comparison results: {'winner': 'A'|'B'|'tie', 'confidence': float, 'reasoning': str}
        """
        prompt = f"""Compare these two outputs and determine which is better.

Reference/Expected output:
{reference}

Output A:
{output_a}

Output B:
{output_b}

Respond with ONLY a JSON object in this exact format:
{{"winner": "A" or "B" or "tie", "confidence": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""

        try:
            response = self.llm_evaluator._call_llm(prompt)

            # Parse response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return {
                    'winner': data.get('winner', 'tie'),
                    'confidence': float(data.get('confidence', 0.5)),
                    'reasoning': data.get('reasoning', '')
                }
        except Exception as e:
            print(f"Warning: Pairwise comparison failed ({e})")

        return {'winner': 'tie', 'confidence': 0.5, 'reasoning': 'Comparison failed'}

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate by comparing actual to expected

        Returns 1.0 if actual is better, 0.0 if worse, 0.5 if tie
        """
        result = self.compare(actual, expected, expected, context)

        if result['winner'] == 'A':  # actual is better
            return result['confidence']
        elif result['winner'] == 'B':  # expected is better
            return 1.0 - result['confidence']
        else:  # tie
            return 0.5
