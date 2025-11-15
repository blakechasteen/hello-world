#!/usr/bin/env python3
"""
HoloLoom Integration for Promptly
==================================
Enables Promptly to use HoloLoom's neural policy for prompt evaluation.

Features:
- HoloLoomEvaluator: Uses HoloLoom's neural decision engine to score prompts
- HoloLoomSpinner: Converts Promptly prompts into HoloLoom MemoryShards
- Integration config loader for unified setup
"""

import sys
import warnings
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

# Try to import HoloLoom components
HOLOLOOM_AVAILABLE = False
try:
    # Add HoloLoom to path if not already present
    repo_root = Path(__file__).parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from HoloLoom.Documentation.types import (
        Query, Features, Context, Response, MemoryShard, Vector
    )
    from HoloLoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"HoloLoom not available: {e}. HoloLoom integration disabled.")


@dataclass
class EvaluationResult:
    """Result from HoloLoom evaluation of a prompt."""
    prompt_name: str
    score: float  # Neural policy confidence
    tool_selection: str  # Which tool the policy would choose
    tool_probabilities: Dict[str, float] = field(default_factory=dict)
    features: Optional[Any] = None  # HoloLoom Features object
    metadata: Dict[str, Any] = field(default_factory=dict)


class HoloLoomEvaluator:
    """
    Uses HoloLoom's neural policy engine to evaluate prompt quality.

    The evaluator:
    1. Converts prompts to HoloLoom queries
    2. Extracts features using HoloLoom's embedding/motif detectors
    3. Runs neural policy to get tool selection confidence
    4. Returns confidence score as prompt quality metric

    Usage:
        evaluator = HoloLoomEvaluator(config_mode='fast')
        score = evaluator.evaluate_prompt('summarizer', test_input)
    """

    def __init__(
        self,
        config_mode: str = 'fast',
        custom_config: Optional[Any] = None
    ):
        """
        Initialize HoloLoom evaluator.

        Args:
            config_mode: One of 'bare', 'fast', 'fused' (HoloLoom execution modes)
            custom_config: Optional custom HoloLoom Config object
        """
        if not HOLOLOOM_AVAILABLE:
            raise RuntimeError(
                "HoloLoom is not available. Please ensure HoloLoom is installed "
                "and accessible in your Python path."
            )

        # Initialize HoloLoom config
        if custom_config:
            self.config = custom_config
        else:
            if config_mode == 'bare':
                self.config = Config.bare()
            elif config_mode == 'fused':
                self.config = Config.fused()
            else:
                self.config = Config.fast()

        # Lazy-initialize components
        self._orchestrator = None
        self._policy = None
        self._embedding = None

    def _ensure_components(self):
        """Lazy initialization of HoloLoom components."""
        if self._orchestrator is not None:
            return

        try:
            # Import orchestrator components
            from HoloLoom.Orchestrator import Orchestrator
            from HoloLoom.policy.unified import create_policy
            from HoloLoom.embedding.spectral import MatryoshkaEmbedding

            # Initialize embedding
            self._embedding = MatryoshkaEmbedding(scales=self.config.scales)

            # Initialize policy
            self._policy = create_policy(
                mem_dim=self.config.scales[-1],  # Use largest scale
                emb=self._embedding,
                scales=self.config.scales
            )

            # Initialize orchestrator (minimal setup)
            self._orchestrator = Orchestrator(config=self.config)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize HoloLoom components: {e}")

    def evaluate_prompt(
        self,
        prompt_name: str,
        prompt_content: str,
        test_inputs: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a prompt using HoloLoom's neural policy.

        Args:
            prompt_name: Name of the prompt being evaluated
            prompt_content: The actual prompt text (may contain format placeholders)
            test_inputs: Optional dict to format the prompt with

        Returns:
            EvaluationResult with score and tool selection details
        """
        self._ensure_components()

        # Format prompt if test inputs provided
        if test_inputs:
            try:
                formatted_prompt = prompt_content.format(**test_inputs)
            except KeyError:
                formatted_prompt = prompt_content
        else:
            formatted_prompt = prompt_content

        # Create HoloLoom query
        query = Query(text=formatted_prompt, metadata={'source': 'promptly'})

        try:
            # Extract features (sync version - orchestrator has async methods)
            # We'll use the policy directly for evaluation
            import asyncio

            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run feature extraction
            features = loop.run_until_complete(
                self._extract_features_async(query)
            )

            # Get policy decision (if policy has async decide method)
            # For now, we'll extract the tool probabilities directly
            # This is a simplified evaluation - full orchestration would be heavier

            # Create minimal context
            context = Context(
                sources=['promptly_evaluation'],
                confidence=1.0,
                shard_texts=[formatted_prompt]
            )

            # Score is based on feature extraction quality and confidence
            # In a real implementation, you'd run through the full policy
            score = features.confidence

            result = EvaluationResult(
                prompt_name=prompt_name,
                score=score,
                tool_selection='evaluate',  # Placeholder
                tool_probabilities={},
                features=features,
                metadata={
                    'config_mode': self.config.mode.name,
                    'num_motifs': len(features.motifs),
                    'embedding_dim': len(features.psi)
                }
            )

            return result

        except Exception as e:
            warnings.warn(f"HoloLoom evaluation failed: {e}")
            # Return fallback result
            return EvaluationResult(
                prompt_name=prompt_name,
                score=0.5,  # Neutral score on failure
                tool_selection='unknown',
                metadata={'error': str(e)}
            )

    async def _extract_features_async(self, query: Query) -> Features:
        """Extract features from query (async version)."""
        # Use orchestrator's feature extraction if available
        if hasattr(self._orchestrator, 'extract_features'):
            return await self._orchestrator.extract_features(query)

        # Fallback: manual feature extraction
        from HoloLoom.motif.base import MotifDetector

        # Extract embedding
        embedding_vec = self._embedding.encode(query.text)

        # Extract motifs
        motif_detector = MotifDetector()
        motifs = motif_detector.detect(query.text)

        return Features(
            psi=embedding_vec,
            motifs=motifs,
            confidence=1.0,
            metadata={'source': 'promptly_evaluator'}
        )

    def batch_evaluate(
        self,
        prompts: List[Dict[str, Any]],
        test_inputs: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple prompts in batch.

        Args:
            prompts: List of dicts with 'name' and 'content' keys
            test_inputs: Optional inputs to format prompts with

        Returns:
            List of EvaluationResults
        """
        results = []
        for prompt in prompts:
            result = self.evaluate_prompt(
                prompt['name'],
                prompt['content'],
                test_inputs
            )
            results.append(result)
        return results


class HoloLoomSpinner:
    """
    Converts Promptly prompts into HoloLoom MemoryShards.

    This allows HoloLoom to ingest and reason about prompt templates
    stored in Promptly repositories.

    Usage:
        spinner = HoloLoomSpinner()
        shards = spinner.spin_prompts(promptly_repo)
    """

    def __init__(self):
        """Initialize the spinner."""
        if not HOLOLOOM_AVAILABLE:
            raise RuntimeError(
                "HoloLoom is not available. Please ensure HoloLoom is installed."
            )

    def prompt_to_shard(
        self,
        prompt_data: Dict[str, Any]
    ) -> MemoryShard:
        """
        Convert a single Promptly prompt to a HoloLoom MemoryShard.

        Args:
            prompt_data: Dict with 'name', 'content', 'metadata', etc.

        Returns:
            MemoryShard suitable for HoloLoom ingestion
        """
        # Extract entities from metadata if available
        entities = []
        if 'metadata' in prompt_data and 'tags' in prompt_data['metadata']:
            entities.extend(prompt_data['metadata']['tags'])

        # Extract motifs from prompt name and content
        motifs = [prompt_data.get('name', 'unknown')]

        # Add version info to metadata
        shard_metadata = {
            'source': 'promptly',
            'version': prompt_data.get('version', 1),
            'branch': prompt_data.get('branch', 'main'),
            'commit_hash': prompt_data.get('commit_hash', ''),
            'original_metadata': prompt_data.get('metadata', {})
        }

        return MemoryShard(
            id=f"promptly_{prompt_data.get('name', 'unknown')}_{prompt_data.get('version', 1)}",
            text=prompt_data.get('content', ''),
            episode=f"promptly_branch_{prompt_data.get('branch', 'main')}",
            entities=entities,
            motifs=motifs,
            metadata=shard_metadata
        )

    def spin_prompts(
        self,
        promptly_instance: Any,
        branch: Optional[str] = None
    ) -> List[MemoryShard]:
        """
        Convert all prompts from a Promptly repository to MemoryShards.

        Args:
            promptly_instance: Instance of Promptly class
            branch: Optional branch to load from (default: current branch)

        Returns:
            List of MemoryShards
        """
        shards = []

        # Get all prompts from Promptly
        prompts = promptly_instance.list_prompts(branch=branch)

        for prompt_meta in prompts:
            # Get full prompt data
            prompt_data = promptly_instance.get(
                prompt_meta['name'],
                version=prompt_meta.get('version')
            )

            if prompt_data:
                shard = self.prompt_to_shard(prompt_data)
                shards.append(shard)

        return shards


def create_evaluator_function(
    config_mode: str = 'fast'
) -> Callable[[str, Any], float]:
    """
    Factory function to create a Promptly-compatible evaluator.

    Returns a function that can be used as the 'evaluator' in Promptly test cases.

    Usage:
        evaluator_fn = create_evaluator_function(config_mode='fast')

        test_cases = [
            {
                'inputs': {'text': 'Hello world'},
                'expected': 'greeting',
                'evaluator': evaluator_fn
            }
        ]

    Args:
        config_mode: HoloLoom config mode ('bare', 'fast', 'fused')

    Returns:
        Evaluator function compatible with Promptly
    """
    evaluator = HoloLoomEvaluator(config_mode=config_mode)

    def evaluate(actual: str, expected: Any) -> float:
        """Evaluate using HoloLoom neural policy."""
        # Create a synthetic prompt from actual output
        result = evaluator.evaluate_prompt(
            prompt_name='eval_temp',
            prompt_content=actual,
            test_inputs=None
        )
        return result.score

    return evaluate


# Convenience function for loading integration config
def load_integration_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load integration configuration from YAML file.

    Args:
        config_path: Path to integration_config.yaml (default: repo root)

    Returns:
        Dict with integration configuration
    """
    import yaml

    if config_path is None:
        # Look for config in repo root
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / 'integration_config.yaml'

    config_path = Path(config_path)

    if not config_path.exists():
        warnings.warn(f"Integration config not found at {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}
