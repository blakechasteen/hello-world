"""
Production LLM Consolidator (Week 3)
=====================================

High-quality semantic fact extraction using LLM APIs.

Supported Providers:
1. OpenAI (GPT-4, GPT-3.5-turbo) - Best quality, moderate cost
2. Anthropic (Claude 3) - High quality, competitive pricing
3. Ollama (local models) - Free, privacy-focused, slower
4. vLLM (local models) - Free, fast inference, requires GPU

Key Features:
- Multi-provider support with automatic fallback
- Cost tracking and optimization
- Prompt engineering for quality extraction
- Batch processing for efficiency
- Rule-based fallback when LLM unavailable

From LangMem research:
- "LLM extracts semantic facts from episodic memories"
- "Background consolidation reduces 100s of episodes → 10s of facts"
"""

from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import os
import json

from hololoom.protocols.types import MemoryShard

# UnifiedMRF for enhanced prompting (Phase 2.2 - Nov 2025)
from hololoom.prompting.unified_mrf import UnifiedMRF, ModelProvider, MetapromptConfig

logger = logging.getLogger(__name__)

# ============================================================================
# LLM Provider Configuration
# ============================================================================

LLMProvider = Literal["openai", "anthropic", "ollama", "vllm", "none"]


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For Ollama/vLLM
    max_tokens: int = 500
    temperature: float = 0.3  # Low temp for factual extraction
    timeout: float = 30.0


@dataclass
class LLMUsage:
    """LLM usage statistics for cost tracking."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str


@dataclass
class ConsolidationPrompt:
    """Prompt templates for different consolidation strategies."""
    fact_extraction: str = """Extract semantic facts from these episodic memories.

Episodic Memories:
{episodes}

Instructions:
- Extract only factual information (not opinions or speculation)
- Make facts atomic (one claim per fact)
- Remove redundancy
- Keep facts concise (1-2 sentences)
- Format as bullet points

Extracted Facts:"""

    entity_extraction: str = """Extract entity relationships from these memories.

Memories:
{episodes}

Instructions:
- Identify entities (people, concepts, technologies)
- Extract relationships between entities
- Format: "Entity1 → Relation → Entity2"
- Focus on meaningful relationships

Relationships:"""

    deduplication: str = """Identify and merge duplicate/similar memories.

Memories:
{memories}

Instructions:
- Group semantically similar memories
- Keep the most informative version
- List unique memory IDs to keep

Unique Memories:"""


# ============================================================================
# Cost Tracking
# ============================================================================

# Pricing per 1M tokens (as of Nov 2025)
PRICING = {
    "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
    "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
    "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
    "claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    "ollama": {"prompt": 0.0, "completion": 0.0},  # Free (local)
    "vllm": {"prompt": 0.0, "completion": 0.0},  # Free (local)
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for LLM usage."""
    if model not in PRICING:
        logger.warning(f"Unknown model for pricing: {model}")
        return 0.0

    pricing = PRICING[model]
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    return prompt_cost + completion_cost


# ============================================================================
# LLM Client Abstraction
# ============================================================================

class LLMClient:
    """Abstract LLM client with multi-provider support."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.total_cost_usd = 0.0
        self.usage_history: List[LLMUsage] = []

        # Initialize provider-specific client
        if config.provider == "openai":
            self._init_openai()
        elif config.provider == "anthropic":
            self._init_anthropic()
        elif config.provider == "ollama":
            self._init_ollama()
        elif config.provider == "vllm":
            self._init_vllm()
        elif config.provider == "none":
            self.client = None
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key or os.getenv("OPENAI_API_KEY")
            )
            logger.info(f"Initialized OpenAI client with model {self.config.model}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.client = None

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            logger.info(f"Initialized Anthropic client with model {self.config.model}")
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            self.client = None

    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama.AsyncClient(
                host=self.config.base_url or "http://localhost:11434"
            )
            logger.info(f"Initialized Ollama client with model {self.config.model}")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.client = None

    def _init_vllm(self):
        """Initialize vLLM client (OpenAI-compatible API)."""
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                base_url=self.config.base_url or "http://localhost:8000/v1",
                api_key="EMPTY"  # vLLM doesn't require API key
            )
            logger.info(f"Initialized vLLM client with model {self.config.model}")
        except ImportError:
            logger.error("OpenAI package required for vLLM. Run: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            self.client = None

    async def complete(self, prompt: str) -> Optional[str]:
        """
        Generate completion from LLM.

        Args:
            prompt: Input prompt

        Returns:
            Completion text or None if error
        """
        if not self.client:
            logger.warning("LLM client not available, using fallback")
            return None

        try:
            if self.config.provider == "openai" or self.config.provider == "vllm":
                return await self._complete_openai(prompt)
            elif self.config.provider == "anthropic":
                return await self._complete_anthropic(prompt)
            elif self.config.provider == "ollama":
                return await self._complete_ollama(prompt)
        except Exception as e:
            logger.error(f"LLM completion failed: {e}", exc_info=True)
            return None

    async def _complete_openai(self, prompt: str) -> Optional[str]:
        """OpenAI/vLLM completion."""
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout
        )

        # Track usage
        usage = response.usage
        cost = calculate_cost(
            self.config.model,
            usage.prompt_tokens,
            usage.completion_tokens
        )

        self._record_usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost
        )

        return response.choices[0].message.content

    async def _complete_anthropic(self, prompt: str) -> Optional[str]:
        """Anthropic completion."""
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track usage
        cost = calculate_cost(
            self.config.model,
            response.usage.input_tokens,
            response.usage.output_tokens
        )

        self._record_usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cost_usd=cost
        )

        return response.content[0].text

    async def _complete_ollama(self, prompt: str) -> Optional[str]:
        """Ollama completion."""
        response = await self.client.chat(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}]
        )

        # Ollama doesn't provide token counts, estimate
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
        completion_tokens = len(response['message']['content'].split()) * 1.3

        self._record_usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cost_usd=0.0  # Free (local)
        )

        return response['message']['content']

    def _record_usage(self, prompt_tokens: int, completion_tokens: int, cost_usd: float):
        """Record LLM usage for cost tracking."""
        usage = LLMUsage(
            provider=self.config.provider,
            model=self.config.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost_usd,
            timestamp=datetime.now().isoformat()
        )

        self.usage_history.append(usage)
        self.total_cost_usd += cost_usd

        logger.info(
            f"LLM usage: {prompt_tokens}p + {completion_tokens}c = "
            f"{prompt_tokens + completion_tokens}t tokens, ${cost_usd:.4f}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        if not self.usage_history:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0
            }

        total_tokens = sum(u.total_tokens for u in self.usage_history)

        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "total_requests": len(self.usage_history),
            "total_tokens": total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_tokens_per_request": total_tokens / len(self.usage_history),
            "usage_history": [
                {
                    "timestamp": u.timestamp,
                    "tokens": u.total_tokens,
                    "cost_usd": u.cost_usd
                }
                for u in self.usage_history[-10:]  # Last 10 requests
            ]
        }


# ============================================================================
# Production LLM Consolidator
# ============================================================================

class ProductionLLMConsolidator:
    """
    Production LLM consolidator with multi-provider support.

    Key Features:
    - Multi-provider support (OpenAI, Anthropic, Ollama, vLLM)
    - Automatic fallback to rule-based extraction
    - Cost tracking and optimization
    - Batch processing for efficiency
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        enable_fallback: bool = True,
        model_provider: Optional[str] = None  # NEW (Phase 2.2)
    ):
        """
        Initialize production LLM consolidator.

        Args:
            config: LLM configuration (None = rule-based only)
            enable_fallback: Enable rule-based fallback on LLM failure
            model_provider: LLM provider for UnifiedMRF optimization
                ("claude", "gemini", "gpt", "ollama"). NEW (Phase 2.2)
        """
        self.config = config
        self.enable_fallback = enable_fallback
        self.prompts = ConsolidationPrompt()
        self.model_provider = model_provider  # NEW (Phase 2.2)

        # UnifiedMRF for enhanced consolidation prompts (Phase 2.2)
        self.mrf = UnifiedMRF()

        # Initialize LLM client
        if config and config.provider != "none":
            self.llm_client = LLMClient(config)
        else:
            self.llm_client = None

        logger.info(
            f"Initialized ProductionLLMConsolidator: "
            f"provider={config.provider if config else 'none'}, "
            f"fallback={enable_fallback}, "
            f"mrf_model={model_provider}"
        )

    async def extract_facts(self, episodes: List[MemoryShard]) -> List[str]:
        """
        Extract semantic facts from episodic memories.

        Args:
            episodes: List of episodic memories

        Returns:
            List of extracted facts (strings)
        """
        if not episodes:
            return []

        # Try LLM extraction first
        if self.llm_client and self.llm_client.client:
            try:
                facts = await self._extract_facts_llm(episodes)
                if facts:
                    logger.info(f"LLM extracted {len(facts)} facts from {len(episodes)} episodes")
                    return facts
            except Exception as e:
                logger.error(f"LLM fact extraction failed: {e}", exc_info=True)

        # Fallback to rule-based
        if self.enable_fallback:
            logger.info("Using rule-based fact extraction (fallback)")
            return await self._extract_facts_fallback(episodes)

        return []

    async def _extract_facts_llm(self, episodes: List[MemoryShard]) -> List[str]:
        """
        LLM-based fact extraction.

        **UPDATED (Phase 2.2 - Nov 2025):** Uses UnifiedMRF 7-component framework
        for enhanced fact extraction quality.
        """
        # Format episodes for metaprompt context
        episode_texts = "\n".join([
            f"{i+1}. {ep.text}"
            for i, ep in enumerate(episodes[:20])  # Limit to 20 to avoid token overflow
        ])

        # Build metaprompt for fact extraction (Phase 2.2)
        metaprompt = MetapromptConfig(
            role=(
                "You are an expert knowledge curator specializing in:\n"
                "- Semantic fact extraction from episodic memories\n"
                "- Information distillation and deduplication\n"
                "- Atomic fact representation (one claim per fact)\n"
                "- Identifying ground truth vs. opinion/speculation\n"
                "- Concise, precise language for knowledge storage"
            ),
            objective={
                "primary": "Extract atomic semantic facts from episodic memories, removing redundancy while preserving ground truth",
                "secondary": [
                    "Ensure each fact is self-contained and verifiable",
                    "Filter out opinions, speculation, and low-confidence claims",
                    "Maintain factual accuracy and contextual coherence"
                ]
            },
            process=[
                {
                    "step": "1. Memory Review",
                    "actions": [
                        f"Read all episodic memories:\n{episode_texts}",
                        "Identify recurring themes and concepts",
                        "Note any contradictions or inconsistencies"
                    ]
                },
                {
                    "step": "2. Fact Extraction",
                    "actions": [
                        "Extract only factual information (not opinions/speculation)",
                        "Make each fact atomic (one claim per fact)",
                        "Ensure facts are self-contained (understandable without context)",
                        "Remove temporal references unless critical (use timeless form)",
                        "Consolidate duplicate or overlapping information"
                    ]
                },
                {
                    "step": "3. Quality Filtering",
                    "actions": [
                        "Remove facts with low confidence or unclear meaning",
                        "Filter out facts that are too specific or trivial",
                        "Keep only facts with high information value",
                        "Limit to 10 most important facts"
                    ]
                }
            ],
            format_spec=(
                "Return facts as a bullet point list (one fact per line).\n"
                "Format: Clean bullet points with '•' or '-' or numbers.\n"
                "Each fact should be 1-2 sentences max.\n"
                "Example:\n"
                "• Thompson Sampling uses Bayesian probability for exploration\n"
                "• The algorithm balances exploration and exploitation trade-offs"
            ),
            constraints=[
                "MUST extract only factual information (no opinions or speculation)",
                "MUST make facts atomic (one independent claim per fact)",
                "MUST remove redundancy (consolidate overlapping facts)",
                "MUST keep facts concise (1-2 sentences maximum)",
                "MUST return 10 facts or fewer (quality over quantity)",
                "SHOULD use present tense (timeless facts)",
                "SHOULD filter out low-value or trivial information"
            ],
            uncertainty=(
                "If information is ambiguous or contradictory:\n"
                "- Prefer facts with higher confidence/evidence\n"
                "- Note contradictions if they represent different valid perspectives\n"
                "- Omit facts that cannot be verified from the episodes\n"
                "\n"
                "If episodes contain opinions vs. facts:\n"
                "- Extract only verifiable facts, not subjective opinions\n"
                "- Example: 'X is useful' (opinion) vs. 'X is used for Y' (fact)"
            ),
            validation=[
                "Each fact is self-contained (understandable without reading episodes)",
                "Each fact is atomic (single independent claim)",
                "No duplicate or overlapping facts",
                "Facts are concise (1-2 sentences max)",
                "Total facts ≤ 10",
                "No opinions, speculation, or unverifiable claims"
            ]
        )

        # Map model provider string to ModelProvider enum
        provider = None
        if self.model_provider:
            provider_map = {
                "claude": ModelProvider.CLAUDE,
                "gemini": ModelProvider.GEMINI,
                "gpt": ModelProvider.GPT,
                "ollama": ModelProvider.OLLAMA
            }
            provider = provider_map.get(self.model_provider.lower())

        # Generate enhanced prompt using UnifiedMRF
        enhanced_prompt = self.mrf.enhance_prompt(
            metaprompt=metaprompt,
            query="",  # No additional query needed
            model=provider
        )

        # Get LLM completion
        completion = await self.llm_client.complete(enhanced_prompt)
        if not completion:
            return []

        # Parse facts (assume bullet point format)
        facts = []
        for line in completion.split('\n'):
            line = line.strip()
            # Remove bullet points, numbers, dashes
            if line and any(c.isalnum() for c in line):
                fact = line.lstrip('•-*0123456789. ').strip()
                if len(fact) > 20:  # Filter short fragments
                    facts.append(fact)

        return facts[:10]  # Limit to 10 facts

    async def _extract_facts_fallback(self, episodes: List[MemoryShard]) -> List[str]:
        """Rule-based fact extraction (fallback)."""
        facts = set()
        for episode in episodes:
            # Split into sentences (simple approach)
            sentences = episode.text.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter short fragments
                    facts.add(sentence)

        return list(facts)[:10]  # Limit to 10 facts

    async def extract_entities(
        self,
        episodes: List[MemoryShard]
    ) -> List[tuple[str, str, str]]:
        """
        Extract entity relationships from episodes.

        Args:
            episodes: List of episodic memories

        Returns:
            List of (src, dst, relation_type) tuples
        """
        if not episodes:
            return []

        # Try LLM extraction first
        if self.llm_client and self.llm_client.client:
            try:
                edges = await self._extract_entities_llm(episodes)
                if edges:
                    logger.info(f"LLM extracted {len(edges)} entity relationships")
                    return edges
            except Exception as e:
                logger.error(f"LLM entity extraction failed: {e}", exc_info=True)

        # Fallback to rule-based
        if self.enable_fallback:
            logger.info("Using rule-based entity extraction (fallback)")
            return await self._extract_entities_fallback(episodes)

        return []

    async def _extract_entities_llm(
        self,
        episodes: List[MemoryShard]
    ) -> List[tuple[str, str, str]]:
        """
        LLM-based entity extraction.

        **UPDATED (Phase 2.2 - Nov 2025):** Uses UnifiedMRF 7-component framework
        for enhanced entity relationship extraction.
        """
        # Format episodes for metaprompt context
        episode_texts = "\n".join([
            f"{i+1}. {ep.text}"
            for i, ep in enumerate(episodes[:20])
        ])

        # Build metaprompt for entity extraction (Phase 2.2)
        metaprompt = MetapromptConfig(
            role=(
                "You are an expert knowledge graph curator specializing in:\n"
                "- Entity recognition and disambiguation\n"
                "- Relationship extraction from unstructured text\n"
                "- Semantic relationship typing (IS_A, USES, PART_OF, etc.)\n"
                "- Knowledge graph construction best practices\n"
                "- Entity normalization and canonicalization"
            ),
            objective={
                "primary": "Extract meaningful entity relationships from episodic memories for knowledge graph construction",
                "secondary": [
                    "Identify key entities (people, concepts, technologies, organizations)",
                    "Extract semantic relationships between entities",
                    "Use standard relationship types for consistency"
                ]
            },
            process=[
                {
                    "step": "1. Entity Identification",
                    "actions": [
                        f"Review all memories:\n{episode_texts}",
                        "Identify key entities (nouns: people, concepts, technologies, organizations)",
                        "Normalize entity names (use canonical form, e.g., 'Thompson Sampling' not 'TS')",
                        "Focus on important entities (not trivial mentions)"
                    ]
                },
                {
                    "step": "2. Relationship Extraction",
                    "actions": [
                        "Identify relationships between entities",
                        "Use standard relationship types: IS_A, USES, PART_OF, CREATED_BY, SIMILAR_TO, LEADS_TO",
                        "Ensure relationships are meaningful and non-trivial",
                        "Avoid over-connecting entities (only strong relationships)"
                    ]
                },
                {
                    "step": "3. Quality Filtering",
                    "actions": [
                        "Remove trivial or weak relationships",
                        "Consolidate duplicate relationships",
                        "Limit to 20 most important relationships"
                    ]
                }
            ],
            format_spec=(
                "Return relationships in arrow format: Entity1 → Relationship → Entity2\n"
                "Format: One relationship per line with bullet points.\n"
                "Use standard relationship types (IS_A, USES, PART_OF, etc.).\n"
                "Example:\n"
                "• Thompson Sampling → IS_A → Bayesian Algorithm\n"
                "• Thompson Sampling → USES → Beta Distribution\n"
                "• Exploration-Exploitation → PART_OF → Reinforcement Learning"
            ),
            constraints=[
                "MUST use canonical entity names (full names, not abbreviations)",
                "MUST use standard relationship types (IS_A, USES, PART_OF, CREATED_BY, SIMILAR_TO, LEADS_TO)",
                "MUST ensure relationships are meaningful (not trivial)",
                "MUST format as: Entity1 → Relationship → Entity2",
                "MUST return 20 relationships or fewer",
                "SHOULD focus on important entities (not every mention)",
                "SHOULD avoid over-connecting (only strong relationships)"
            ],
            uncertainty=(
                "If entity type is unclear:\n"
                "- Use the most common/canonical form\n"
                "- Prefer full names over abbreviations\n"
                "\n"
                "If relationship type is ambiguous:\n"
                "- Choose the most specific relationship that fits\n"
                "- Example: Prefer 'USES' over generic 'RELATED_TO'\n"
                "- Use IS_A for taxonomy, PART_OF for composition, USES for functional"
            ),
            validation=[
                "Each relationship has 3 parts: Entity1 → Relationship → Entity2",
                "All entities use canonical/normalized names",
                "All relationships use standard types (uppercase)",
                "No duplicate relationships",
                "Total relationships ≤ 20",
                "Relationships are meaningful (not trivial)"
            ]
        )

        # Map model provider string to ModelProvider enum
        provider = None
        if self.model_provider:
            provider_map = {
                "claude": ModelProvider.CLAUDE,
                "gemini": ModelProvider.GEMINI,
                "gpt": ModelProvider.GPT,
                "ollama": ModelProvider.OLLAMA
            }
            provider = provider_map.get(self.model_provider.lower())

        # Generate enhanced prompt using UnifiedMRF
        enhanced_prompt = self.mrf.enhance_prompt(
            metaprompt=metaprompt,
            query="",  # No additional query needed
            model=provider
        )

        # Get LLM completion
        completion = await self.llm_client.complete(enhanced_prompt)
        if not completion:
            return []

        # Parse relationships (assume "Entity1 → Relation → Entity2" format)
        edges = []
        for line in completion.split('\n'):
            line = line.strip()
            if '→' in line:
                parts = [p.strip() for p in line.split('→')]
                if len(parts) == 3:
                    src, rel, dst = parts
                    # Clean entity names (remove bullet points, numbers)
                    src = src.lstrip('•-*0123456789. ').strip()
                    dst = dst.lstrip('•-*0123456789. ').strip()
                    if src and dst and rel:
                        edges.append((src, dst, rel.upper()))

        return edges[:20]  # Limit to 20 edges

    async def _extract_entities_fallback(
        self,
        episodes: List[MemoryShard]
    ) -> List[tuple[str, str, str]]:
        """Rule-based entity extraction (fallback)."""
        edges = []
        for episode in episodes:
            entities = episode.entities
            for i in range(len(entities) - 1):
                edges.append((entities[i], entities[i + 1], "MENTIONS"))

        return edges[:20]  # Limit to 20 edges

    async def deduplicate(
        self,
        memories: List[MemoryShard]
    ) -> List[MemoryShard]:
        """
        Deduplicate similar memories.

        Args:
            memories: List of memories to deduplicate

        Returns:
            Deduplicated list (unique memories)
        """
        if not memories:
            return []

        # Try LLM deduplication first
        if self.llm_client and self.llm_client.client:
            try:
                unique = await self._deduplicate_llm(memories)
                if unique:
                    logger.info(f"LLM deduplicated {len(memories)} → {len(unique)} memories")
                    return unique
            except Exception as e:
                logger.error(f"LLM deduplication failed: {e}", exc_info=True)

        # Fallback to rule-based
        if self.enable_fallback:
            logger.info("Using rule-based deduplication (fallback)")
            return await self._deduplicate_fallback(memories)

        return memories

    async def _deduplicate_llm(
        self,
        memories: List[MemoryShard]
    ) -> List[MemoryShard]:
        """LLM-based deduplication."""
        # Format memories for prompt
        memory_texts = "\n".join([
            f"{i+1}. [ID: {m.id}] {m.text}"
            for i, m in enumerate(memories[:50])  # Limit to 50
        ])

        prompt = self.prompts.deduplication.format(memories=memory_texts)

        # Get LLM completion
        completion = await self.llm_client.complete(prompt)
        if not completion:
            return memories

        # Parse unique memory IDs
        unique_ids = set()
        for line in completion.split('\n'):
            # Extract IDs from lines containing [ID: ...]
            if '[ID:' in line:
                start = line.find('[ID:') + 4
                end = line.find(']', start)
                if end > start:
                    memory_id = line[start:end].strip()
                    unique_ids.add(memory_id)

        # Return memories with unique IDs
        unique_memories = [m for m in memories if m.id in unique_ids]
        return unique_memories if unique_memories else memories

    async def _deduplicate_fallback(
        self,
        memories: List[MemoryShard]
    ) -> List[MemoryShard]:
        """Rule-based deduplication (fallback)."""
        seen_texts = set()
        unique_memories = []

        for mem in memories:
            if mem.text not in seen_texts:
                seen_texts.add(mem.text)
                unique_memories.append(mem)

        return unique_memories

    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        if self.llm_client:
            return self.llm_client.get_statistics()
        return {
            "provider": "none",
            "total_requests": 0,
            "total_cost_usd": 0.0
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_llm_config(
    provider: LLMProvider = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMConfig:
    """
    Create LLM configuration with sensible defaults.

    Args:
        provider: LLM provider
        model: Model name (None = use default for provider)
        api_key: API key (None = read from environment)
        base_url: Base URL for Ollama/vLLM

    Returns:
        LLMConfig
    """
    # Default models for each provider
    default_models = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama2",
        "vllm": "meta-llama/Llama-2-7b-hf"
    }

    model = model or default_models.get(provider, "gpt-3.5-turbo")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url
    )


def create_production_consolidator(
    provider: LLMProvider = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_fallback: bool = True,
    model_provider: Optional[str] = None  # NEW (Phase 2.2)
) -> ProductionLLMConsolidator:
    """
    Create production LLM consolidator with sensible defaults.

    Args:
        provider: LLM provider
        model: Model name (None = use default)
        api_key: API key (None = read from environment)
        enable_fallback: Enable rule-based fallback
        model_provider: LLM provider for UnifiedMRF optimization
            ("claude", "gemini", "gpt", "ollama"). NEW (Phase 2.2)

    Returns:
        ProductionLLMConsolidator
    """
    if provider == "none":
        return ProductionLLMConsolidator(
            config=None,
            enable_fallback=True,
            model_provider=model_provider  # NEW (Phase 2.2)
        )

    config = create_llm_config(provider, model, api_key)
    return ProductionLLMConsolidator(
        config=config,
        enable_fallback=enable_fallback,
        model_provider=model_provider  # NEW (Phase 2.2)
    )
