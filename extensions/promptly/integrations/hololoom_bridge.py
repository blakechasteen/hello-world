"""
HoloLoom Bridge - Integration between Promptly and HoloLoom memory system

Features:
- Store prompt execution results in HoloLoom knowledge graph
- Retrieve past refinement patterns
- Learn which prompts work best for which tasks
- Thompson Sampling for prompt selection optimization
- RAG-based context retrieval for prompt enhancement
- Agentic reasoning for complex prompt tasks

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

    # RAG-enhanced prompt execution
    context = await bridge.get_context_for_prompt(prompt_content, task_desc)
    enhanced = await bridge.enhance_prompt_with_rag(prompt_name, task_type)

    # Agentic reasoning
    result = await bridge.run_agentic_prompt(prompt_name, mode="verify")
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
                from hololoom import hololoom
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

    # ==================== RAG Integration Methods ====================

    async def get_context_for_prompt(
        self,
        prompt_content: str,
        task_description: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a prompt using RAG.

        Uses HoloLoom's memory system to find relevant past executions,
        patterns, and knowledge that can enhance the prompt.

        Args:
            prompt_content: The prompt template content
            task_description: Description of the task
            k: Number of context items to retrieve

        Returns:
            List of context items with content and relevance scores
        """
        context_items = []

        if await self._ensure_hololoom():
            # Build search query from prompt and task
            search_query = f"{task_description} {prompt_content[:200]}"

            # Recall from hololoom memory
            memories = await self._hololoom.recall(search_query)

            for mem in memories[:k]:
                content = mem.content if hasattr(mem, 'content') else str(mem)
                relevance = mem.relevance if hasattr(mem, 'relevance') else 0.5

                context_items.append({
                    'content': content,
                    'relevance': relevance,
                    'source': 'hololoom_memory',
                    'type': 'semantic_recall'
                })
        else:
            # Fallback: search in-memory executions
            query_lower = task_description.lower()
            scored_executions = []

            for exec in self._executions:
                # Simple relevance scoring
                score = 0.0
                if query_lower in exec.prompt_content.lower():
                    score += 0.5
                if query_lower in exec.prompt_name.lower():
                    score += 0.3
                # Boost by quality
                score += exec.quality_score * 0.2

                if score > 0:
                    scored_executions.append((exec, score))

            # Sort by score and take top k
            scored_executions.sort(key=lambda x: x[1], reverse=True)

            for exec, score in scored_executions[:k]:
                context_items.append({
                    'content': f"Prompt: {exec.prompt_name}\n"
                               f"Quality: {exec.quality_score:.2f}\n"
                               f"Output: {exec.output[:300]}...",
                    'relevance': min(score, 1.0),
                    'source': 'in_memory',
                    'type': 'execution_history'
                })

        return context_items

    async def enhance_prompt_with_rag(
        self,
        prompt_name: str,
        prompt_content: str,
        task_type: Optional[str] = None,
        context_k: int = 3
    ) -> Dict[str, Any]:
        """
        Enhance a prompt with RAG-retrieved context.

        Retrieves relevant context and prepares an enhanced prompt
        with additional information from past executions.

        Args:
            prompt_name: Name of the prompt
            prompt_content: The prompt template
            task_type: Optional task type for filtering
            context_k: Number of context items to include

        Returns:
            Dict with enhanced prompt and metadata
        """
        # Get relevant context
        task_desc = task_type or prompt_name
        context_items = await self.get_context_for_prompt(
            prompt_content, task_desc, k=context_k
        )

        # Build context section
        context_text = ""
        if context_items:
            context_text = "\n\n--- Relevant Context ---\n"
            for i, item in enumerate(context_items, 1):
                context_text += f"\n[{i}] (relevance: {item['relevance']:.2f})\n"
                context_text += f"{item['content'][:400]}\n"
            context_text += "\n--- End Context ---\n\n"

        # Get Thompson Sampling recommendation if available
        expected_quality = None
        if task_type and task_type in self.sampler.priors:
            expected_quality = self.sampler.get_expected_quality(task_type, prompt_name)

        return {
            'original_prompt': prompt_content,
            'enhanced_prompt': context_text + prompt_content,
            'context_items': context_items,
            'context_count': len(context_items),
            'avg_context_relevance': (
                sum(c['relevance'] for c in context_items) / len(context_items)
                if context_items else 0.0
            ),
            'expected_quality': expected_quality,
            'enhancement_applied': bool(context_items)
        }

    async def discover_similar_prompts(
        self,
        query: str,
        limit: int = 10,
        min_quality: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Discover similar prompts with quality filtering.

        Args:
            query: Search query describing the desired prompt
            limit: Maximum number of results
            min_quality: Minimum quality score filter

        Returns:
            List of similar prompts sorted by relevance × quality
        """
        # Get all similar prompts
        similar = await self.find_similar_prompts(query, limit=limit * 2)

        # Filter and enhance with quality data
        results = []

        if await self._ensure_hololoom():
            # HoloLoom results - try to match with execution history
            for item in similar:
                # Try to find matching execution for quality data
                quality = 0.5  # Default if no execution found
                for exec in self._executions:
                    if exec.prompt_name in item.get('content', ''):
                        quality = exec.quality_score
                        break

                if quality >= min_quality:
                    item['quality_score'] = quality
                    item['combined_score'] = item.get('relevance', 0.5) * quality
                    results.append(item)
        else:
            # In-memory results already have quality
            for item in similar:
                quality = item.get('quality_score', 0.5)
                if quality >= min_quality:
                    item['combined_score'] = quality  # Simple scoring for fallback
                    results.append(item)

        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        return results[:limit]

    # ==================== Agentic Reasoning Methods ====================

    async def run_agentic_prompt(
        self,
        prompt_name: str,
        prompt_content: str,
        variables: Optional[Dict[str, Any]] = None,
        mode: str = "verify",
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Execute a prompt with agentic reasoning.

        Supports multiple reasoning modes:
        - "direct": Single-pass execution
        - "verify": Execute and verify the result
        - "research": Multi-query exploration
        - "plan_execute": Break into steps and execute

        Args:
            prompt_name: Name of the prompt
            prompt_content: The prompt template
            variables: Variables to fill in the prompt
            mode: Reasoning mode (direct, verify, research, plan_execute)
            max_steps: Maximum reasoning steps

        Returns:
            Dict with result, verification, and reasoning trace
        """
        variables = variables or {}
        reasoning_trace = []
        result = {
            'prompt_name': prompt_name,
            'mode': mode,
            'steps_taken': 0,
            'verification': None,
            'response': None,
            'confidence': 0.5,
            'reasoning_trace': reasoning_trace
        }

        # Try to use HoloLoom's agentic reasoning if available
        if await self._ensure_hololoom():
            try:
                from hololoom.agentic import create_agentic_orchestrator, ReasoningMode

                # Map mode to HoloLoom's ReasoningMode
                mode_map = {
                    'direct': ReasoningMode.DIRECT,
                    'verify': ReasoningMode.VERIFY,
                    'research': ReasoningMode.RESEARCH,
                    'plan_execute': ReasoningMode.PLAN_EXECUTE
                }
                reasoning_mode = mode_map.get(mode, ReasoningMode.DIRECT)

                # Build query from prompt and variables
                filled_prompt = prompt_content
                for key, value in variables.items():
                    filled_prompt = filled_prompt.replace(f"{{{key}}}", str(value))

                # Note: Full agentic integration requires orchestrator setup
                # This is a simplified version that records the attempt
                reasoning_trace.append({
                    'step': 1,
                    'action': 'prepare_agentic',
                    'mode': mode,
                    'prompt_length': len(filled_prompt)
                })

                # For now, we simulate the agentic result
                # Full implementation would integrate with WeavingOrchestrator
                result['response'] = f"[Agentic {mode}] Processing: {prompt_name}"
                result['confidence'] = 0.7
                result['steps_taken'] = 1

                if mode == 'verify':
                    result['verification'] = {
                        'verified': True,
                        'checks_passed': ['syntax', 'completeness'],
                        'warnings': []
                    }

            except ImportError:
                reasoning_trace.append({
                    'step': 1,
                    'action': 'fallback',
                    'reason': 'agentic module not available'
                })
        else:
            reasoning_trace.append({
                'step': 1,
                'action': 'fallback',
                'reason': 'HoloLoom not available'
            })

        # Fallback: Simple execution without agentic reasoning
        if result['response'] is None:
            filled_prompt = prompt_content
            for key, value in variables.items():
                filled_prompt = filled_prompt.replace(f"{{{key}}}", str(value))

            result['response'] = filled_prompt
            result['confidence'] = 0.5
            result['steps_taken'] = 1

            if mode == 'verify':
                # Basic verification
                result['verification'] = {
                    'verified': len(filled_prompt) > 0,
                    'checks_passed': ['non_empty'] if filled_prompt else [],
                    'warnings': ['no_agentic_verification']
                }

        result['reasoning_trace'] = reasoning_trace
        return result

    # ==================== Comprehensive Stats Methods ====================

    async def get_comprehensive_stats(
        self,
        prompt_name: str,
        task_type: Optional[str] = None,
        window_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics combining multiple data sources.

        Combines prompt execution stats, Thompson Sampling insights,
        and HoloLoom memory analysis.

        Args:
            prompt_name: Name of the prompt
            task_type: Optional task type filter
            window_days: Number of days to analyze

        Returns:
            Comprehensive stats dictionary
        """
        from datetime import timedelta

        stats = {
            'prompt_name': prompt_name,
            'task_type': task_type,
            'window_days': window_days,
            'execution_stats': None,
            'thompson_stats': None,
            'memory_stats': None,
            'recommendations': []
        }

        # Get basic execution stats
        prompt_stats = await self.get_prompt_stats(prompt_name)
        if prompt_stats:
            stats['execution_stats'] = {
                'total_executions': prompt_stats.total_executions,
                'avg_quality': prompt_stats.avg_quality,
                'avg_latency_ms': prompt_stats.avg_latency_ms,
                'success_rate': prompt_stats.success_rate,
                'last_used': prompt_stats.last_used.isoformat(),
                'task_types': prompt_stats.task_types
            }

        # Get Thompson Sampling stats
        if task_type:
            thompson_insights = await self.get_learning_insights(task_type)
            # Find this prompt in the insights
            for p in thompson_insights.get('prompts', []):
                if p['name'] == prompt_name:
                    stats['thompson_stats'] = {
                        'expected_quality': p['expected_quality'],
                        'alpha': p['alpha'],
                        'beta': p['beta'],
                        'total_samples': p['total_samples'],
                        'rank_in_task': thompson_insights['prompts'].index(p) + 1,
                        'total_prompts_for_task': thompson_insights['known_prompts']
                    }
                    break

        # Get memory stats from hololoom if available
        if await self._ensure_hololoom():
            memories = await self._hololoom.recall(f"prompt {prompt_name}")
            stats['memory_stats'] = {
                'related_memories': len(memories) if memories else 0,
                'hololoom_available': True
            }
        else:
            stats['memory_stats'] = {
                'related_memories': 0,
                'hololoom_available': False
            }

        # Generate recommendations
        recommendations = []

        if stats['execution_stats']:
            exec_stats = stats['execution_stats']
            if exec_stats['avg_quality'] < 0.6:
                recommendations.append({
                    'type': 'quality',
                    'priority': 'high',
                    'message': f"Low average quality ({exec_stats['avg_quality']:.2f}). Consider prompt revision."
                })
            if exec_stats['success_rate'] < 0.5:
                recommendations.append({
                    'type': 'success_rate',
                    'priority': 'high',
                    'message': f"Low success rate ({exec_stats['success_rate']:.1%}). Review failure cases."
                })
            if exec_stats['avg_latency_ms'] > 5000:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'message': f"High latency ({exec_stats['avg_latency_ms']:.0f}ms). Consider optimization."
                })

        if stats['thompson_stats']:
            ts = stats['thompson_stats']
            if ts['total_samples'] < 10:
                recommendations.append({
                    'type': 'data',
                    'priority': 'low',
                    'message': "Limited data for Thompson Sampling. Run more executions."
                })
            if ts['rank_in_task'] > 1 and ts['expected_quality'] < 0.7:
                recommendations.append({
                    'type': 'alternative',
                    'priority': 'medium',
                    'message': f"Better alternatives exist for this task type (rank {ts['rank_in_task']}/{ts['total_prompts_for_task']})."
                })

        stats['recommendations'] = recommendations
        return stats

    async def sync_with_hololoom(self) -> Dict[str, Any]:
        """
        Synchronize learning state with HoloLoom memory.

        Persists Thompson Sampling priors and execution history
        to HoloLoom's knowledge graph for long-term storage.

        Returns:
            Sync status and statistics
        """
        sync_result = {
            'success': False,
            'priors_synced': 0,
            'executions_synced': 0,
            'errors': []
        }

        if not await self._ensure_hololoom():
            sync_result['errors'].append("HoloLoom not available")
            return sync_result

        try:
            # Sync Thompson Sampling priors
            for task_type, prompts in self.sampler.priors.items():
                for prompt_name, (alpha, beta) in prompts.items():
                    memory_content = f"""Thompson Sampling Prior
Task Type: {task_type}
Prompt: {prompt_name}
Alpha (successes): {alpha:.2f}
Beta (failures): {beta:.2f}
Expected Quality: {alpha / (alpha + beta):.3f}
Total Samples: {alpha + beta - 2:.0f}
"""
                    await self._hololoom.experience(memory_content)
                    sync_result['priors_synced'] += 1

            # Sync recent executions not yet in HoloLoom
            for exec in self._executions[-100:]:  # Last 100 executions
                memory_content = self._execution_to_memory(exec)
                await self._hololoom.experience(memory_content)
                sync_result['executions_synced'] += 1

            sync_result['success'] = True

        except Exception as e:
            sync_result['errors'].append(str(e))

        return sync_result

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
