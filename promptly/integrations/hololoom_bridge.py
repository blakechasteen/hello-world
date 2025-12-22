"""
HoloLoom Bridge - Integration between Promptly and HoloLoom memory system

Features:
- Store prompt execution results in HoloLoom knowledge graph
- Retrieve past refinement patterns
- Learn which prompts work best for which tasks
- Thompson Sampling for prompt selection optimization

Usage:
    from promptly.integrations.hololoom_bridge import HoloLoomBridge

    # Create bridge (auto-detects HoloLoom availability)
    bridge = HoloLoomBridge()

    # Store prompt execution
    await bridge.store_execution(
        prompt_name="summarize",
        prompt_content="Summarize the following text...",
        input_data={"text": "..."},
        output="Summary: ...",
        quality_score=0.92
    )

    # Retrieve similar prompts
    similar = await bridge.find_similar_prompts("summarization task")

    # Get best prompt for task type
    best = await bridge.recommend_prompt("code_review")
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib
import json


@dataclass
class PromptExecution:
    """Record of a prompt execution."""
    prompt_name: str
    prompt_content: str
    prompt_version: str
    input_data: Dict[str, Any]
    output: str
    quality_score: float  # 0.0-1.0
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_id(self) -> str:
        """Generate unique execution ID."""
        content = f"{self.prompt_name}:{self.timestamp.isoformat()}:{hash(self.output)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class PromptStats:
    """Statistics for a prompt."""
    prompt_name: str
    total_executions: int
    avg_quality: float
    avg_latency_ms: float
    success_rate: float  # quality >= 0.7
    last_used: datetime
    task_types: List[str]


class ThompsonSamplerPrompts:
    """Thompson Sampling for prompt selection optimization."""

    def __init__(self):
        # Beta distribution parameters per (task_type, prompt_name)
        # alpha = successes + 1, beta = failures + 1
        self.priors: Dict[str, Dict[str, tuple]] = {}  # {task: {prompt: (alpha, beta)}}

    def update(self, task_type: str, prompt_name: str, success: bool, quality: float = 0.5):
        """Update priors based on execution outcome."""
        if task_type not in self.priors:
            self.priors[task_type] = {}

        if prompt_name not in self.priors[task_type]:
            self.priors[task_type][prompt_name] = (1.0, 1.0)

        alpha, beta = self.priors[task_type][prompt_name]

        if success:
            alpha += quality
        else:
            beta += (1 - quality)

        self.priors[task_type][prompt_name] = (alpha, beta)

    def sample(self, task_type: str, candidates: List[str]) -> Optional[str]:
        """Sample best prompt using Thompson Sampling."""
        import random

        if task_type not in self.priors:
            return candidates[0] if candidates else None

        best_prompt = None
        best_sample = -1

        for prompt in candidates:
            if prompt in self.priors[task_type]:
                alpha, beta = self.priors[task_type][prompt]
            else:
                alpha, beta = 1.0, 1.0

            # Sample from Beta distribution
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best_prompt = prompt

        return best_prompt or (candidates[0] if candidates else None)

    def get_expected_quality(self, task_type: str, prompt_name: str) -> float:
        """Get expected quality E[X] = alpha / (alpha + beta)."""
        if task_type not in self.priors or prompt_name not in self.priors[task_type]:
            return 0.5  # Uninformed prior

        alpha, beta = self.priors[task_type][prompt_name]
        return alpha / (alpha + beta)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'priors': {
                task: {
                    prompt: {'alpha': a, 'beta': b}
                    for prompt, (a, b) in prompts.items()
                }
                for task, prompts in self.priors.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThompsonSamplerPrompts':
        """Deserialize from persistence."""
        sampler = cls()
        for task, prompts in data.get('priors', {}).items():
            sampler.priors[task] = {
                prompt: (p['alpha'], p['beta'])
                for prompt, p in prompts.items()
            }
        return sampler


class HoloLoomBridge:
    """
    Bridge between Promptly and HoloLoom memory system.

    Provides:
    - Storage of prompt executions in HoloLoom knowledge graph
    - Retrieval of similar prompts based on semantic search
    - Thompson Sampling for prompt selection optimization
    - Learning from execution outcomes
    """

    def __init__(
        self,
        hololoom_instance=None,
        enable_learning: bool = True,
        quality_threshold: float = 0.7
    ):
        """
        Initialize HoloLoom bridge.

        Args:
            hololoom_instance: Optional HoloLoom instance (auto-created if None)
            enable_learning: Enable Thompson Sampling learning
            quality_threshold: Threshold for "success" (quality >= threshold)
        """
        self._hololoom = hololoom_instance
        self._hololoom_available = None
        self.enable_learning = enable_learning
        self.quality_threshold = quality_threshold
        self.sampler = ThompsonSamplerPrompts()
        self._executions: List[PromptExecution] = []  # In-memory fallback

    async def _ensure_hololoom(self):
        """Lazy-load HoloLoom if available."""
        if self._hololoom_available is None:
            try:
                from HoloLoom import HoloLoom
                if self._hololoom is None:
                    self._hololoom = HoloLoom()
                self._hololoom_available = True
            except ImportError:
                self._hololoom_available = False

        return self._hololoom_available

    async def store_execution(
        self,
        prompt_name: str,
        prompt_content: str,
        input_data: Dict[str, Any],
        output: str,
        quality_score: float,
        prompt_version: str = "latest",
        latency_ms: float = 0.0,
        task_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a prompt execution result.

        Args:
            prompt_name: Name of the prompt
            prompt_content: The prompt template
            input_data: Input variables
            output: Generated output
            quality_score: Quality score 0.0-1.0
            prompt_version: Version of the prompt
            latency_ms: Execution time in ms
            task_type: Type of task (for learning)
            metadata: Additional metadata

        Returns:
            Execution ID
        """
        execution = PromptExecution(
            prompt_name=prompt_name,
            prompt_content=prompt_content,
            prompt_version=prompt_version,
            input_data=input_data,
            output=output,
            quality_score=quality_score,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )

        # Store in HoloLoom if available
        if await self._ensure_hololoom():
            memory_content = self._execution_to_memory(execution)
            await self._hololoom.experience(memory_content)
        else:
            # Fallback to in-memory storage
            self._executions.append(execution)

        # Update Thompson Sampling priors
        if self.enable_learning and task_type:
            success = quality_score >= self.quality_threshold
            self.sampler.update(task_type, prompt_name, success, quality_score)

        return execution.execution_id

    def _execution_to_memory(self, execution: PromptExecution) -> str:
        """Convert execution to memory content for HoloLoom."""
        return f"""Prompt Execution: {execution.prompt_name}
Version: {execution.prompt_version}
Quality Score: {execution.quality_score:.2f}
Latency: {execution.latency_ms:.1f}ms
Timestamp: {execution.timestamp.isoformat()}

Prompt Content:
{execution.prompt_content[:500]}...

Output Summary:
{execution.output[:500]}...
"""

    async def find_similar_prompts(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar prompts based on semantic search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of similar prompt records
        """
        if await self._ensure_hololoom():
            memories = await self._hololoom.recall(query)
            results = []
            for mem in memories[:limit]:
                # Parse memory content back to execution data
                results.append({
                    'content': mem.content if hasattr(mem, 'content') else str(mem),
                    'relevance': mem.relevance if hasattr(mem, 'relevance') else 0.5
                })
            return results
        else:
            # Fallback: simple text matching
            results = []
            query_lower = query.lower()
            for exec in self._executions:
                if query_lower in exec.prompt_name.lower() or query_lower in exec.prompt_content.lower():
                    results.append({
                        'prompt_name': exec.prompt_name,
                        'prompt_content': exec.prompt_content,
                        'quality_score': exec.quality_score
                    })
            return results[:limit]

    async def recommend_prompt(
        self,
        task_type: str,
        candidates: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Recommend best prompt for a task type using Thompson Sampling.

        Args:
            task_type: Type of task (e.g., "summarization", "code_review")
            candidates: List of candidate prompt names (uses all known if None)

        Returns:
            Recommended prompt name
        """
        if candidates is None:
            # Get all prompts seen for this task type
            if task_type in self.sampler.priors:
                candidates = list(self.sampler.priors[task_type].keys())
            else:
                return None

        if not candidates:
            return None

        return self.sampler.sample(task_type, candidates)

    async def get_prompt_stats(self, prompt_name: str) -> Optional[PromptStats]:
        """
        Get statistics for a prompt.

        Args:
            prompt_name: Name of the prompt

        Returns:
            PromptStats or None if not found
        """
        # Filter executions for this prompt
        executions = [e for e in self._executions if e.prompt_name == prompt_name]

        if not executions:
            return None

        total = len(executions)
        avg_quality = sum(e.quality_score for e in executions) / total
        avg_latency = sum(e.latency_ms for e in executions) / total
        successes = sum(1 for e in executions if e.quality_score >= self.quality_threshold)
        success_rate = successes / total

        # Extract task types from metadata
        task_types = list(set(
            e.metadata.get('task_type', 'unknown')
            for e in executions
            if 'task_type' in e.metadata
        ))

        return PromptStats(
            prompt_name=prompt_name,
            total_executions=total,
            avg_quality=avg_quality,
            avg_latency_ms=avg_latency,
            success_rate=success_rate,
            last_used=max(e.timestamp for e in executions),
            task_types=task_types
        )

    async def get_learning_insights(self, task_type: str) -> Dict[str, Any]:
        """
        Get learning insights for a task type.

        Args:
            task_type: Type of task

        Returns:
            Dict with insights (best prompts, expected qualities, etc.)
        """
        if task_type not in self.sampler.priors:
            return {
                'task_type': task_type,
                'known_prompts': 0,
                'prompts': []
            }

        prompts = []
        for prompt_name in self.sampler.priors[task_type]:
            alpha, beta = self.sampler.priors[task_type][prompt_name]
            expected_quality = alpha / (alpha + beta)
            prompts.append({
                'name': prompt_name,
                'expected_quality': expected_quality,
                'total_samples': alpha + beta - 2,  # Subtract prior
                'alpha': alpha,
                'beta': beta
            })

        # Sort by expected quality
        prompts.sort(key=lambda p: p['expected_quality'], reverse=True)

        return {
            'task_type': task_type,
            'known_prompts': len(prompts),
            'best_prompt': prompts[0]['name'] if prompts else None,
            'prompts': prompts
        }

    def save_learning_state(self, path: str):
        """Save Thompson Sampling state to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.sampler.to_dict(), f, indent=2)

    def load_learning_state(self, path: str):
        """Load Thompson Sampling state from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.sampler = ThompsonSamplerPrompts.from_dict(data)

    async def close(self):
        """Clean up resources."""
        if self._hololoom is not None and hasattr(self._hololoom, 'close'):
            await self._hololoom.close()


# Convenience functions

async def create_bridge(
    enable_learning: bool = True,
    quality_threshold: float = 0.7
) -> HoloLoomBridge:
    """Create a HoloLoom bridge instance."""
    return HoloLoomBridge(
        enable_learning=enable_learning,
        quality_threshold=quality_threshold
    )


# Global bridge instance (lazy-loaded)
_global_bridge: Optional[HoloLoomBridge] = None


async def get_bridge() -> HoloLoomBridge:
    """Get or create global bridge instance."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = await create_bridge()
    return _global_bridge


async def store_prompt_execution(
    prompt_name: str,
    prompt_content: str,
    input_data: Dict[str, Any],
    output: str,
    quality_score: float,
    task_type: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function to store prompt execution."""
    bridge = await get_bridge()
    return await bridge.store_execution(
        prompt_name=prompt_name,
        prompt_content=prompt_content,
        input_data=input_data,
        output=output,
        quality_score=quality_score,
        task_type=task_type,
        **kwargs
    )


async def recommend_prompt_for_task(
    task_type: str,
    candidates: Optional[List[str]] = None
) -> Optional[str]:
    """Convenience function to recommend prompt."""
    bridge = await get_bridge()
    return await bridge.recommend_prompt(task_type, candidates)
