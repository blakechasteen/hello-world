#!/usr/bin/env python3
"""
Custom Agent Registry
=====================

Extensible framework for registering and managing custom workflow agents.

Features:
- Dynamic agent registration
- Schema validation
- Agent versioning
- Plugin architecture
- Category management

Usage:
    from HoloLoom.workflows.agent_registry import AgentRegistry, CustomAgent

    registry = AgentRegistry()

    # Register custom agent
    @registry.register('my_custom_agent', category='custom')
    async def my_agent(inputs, config):
        return {'result': 'custom output'}

    # Use agent
    result = await registry.execute('my_custom_agent', inputs={}, config={})
"""

import asyncio
import inspect
import logging
from typing import Dict, List, Any, Callable, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AgentCategory(str, Enum):
    """Agent categories"""
    QUERY = "query"
    PROCESS = "process"
    MEMORY = "memory"
    DECISION = "decision"
    OUTPUT = "output"
    CONTROL = "control"
    CUSTOM = "custom"
    LLM = "llm"
    RAG = "rag"
    CODE = "code"
    DATA = "data"
    ML = "ml"
    INTEGRATION = "integration"


@dataclass
class AgentConfig:
    """Configuration schema for agent parameters"""
    name: str
    param_type: str  # 'text', 'number', 'boolean', 'select', 'multiselect'
    default: Any
    required: bool = False
    options: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    description: str = ""


@dataclass
class AgentSchema:
    """Complete schema for a custom agent"""
    agent_id: str
    name: str
    category: AgentCategory
    version: str = "1.0.0"

    # I/O
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Configuration
    config_schema: Dict[str, AgentConfig] = field(default_factory=dict)

    # Visual
    color: str = "#888888"
    icon: str = "⚙️"

    # Metadata
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Execution
    executor: Optional[Callable] = None
    async_executor: bool = True
    timeout: float = 30.0

    # Stats
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0


class AgentRegistry:
    """
    Central registry for all workflow agents (built-in and custom).

    Provides:
    - Agent registration and discovery
    - Schema validation
    - Execution routing
    - Performance tracking
    - Plugin architecture
    """

    def __init__(self):
        self.agents: Dict[str, AgentSchema] = {}
        self.categories: Dict[AgentCategory, Set[str]] = {
            category: set() for category in AgentCategory
        }
        self._load_built_in_agents()

    def _load_built_in_agents(self):
        """Load built-in HoloLoom agents"""

        # Query agents
        self.register_schema(AgentSchema(
            agent_id='hololoom',
            name='HoloLoom Query',
            category=AgentCategory.QUERY,
            inputs=['query'],
            outputs=['spacetime'],
            color='#667eea',
            icon='🧵',
            description='Full HoloLoom weaving cycle with multi-scale embeddings',
            config_schema={
                'pattern': AgentConfig(
                    'pattern', 'select', 'fast',
                    options=['bare', 'fast', 'fused'],
                    description='Execution mode (bare=fastest, fused=highest quality)'
                ),
                'return_trace': AgentConfig(
                    'return_trace', 'boolean', True,
                    description='Include complete execution trace in output'
                )
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='search',
            name='Memory Search',
            category=AgentCategory.QUERY,
            inputs=['query'],
            outputs=['memories'],
            color='#667eea',
            icon='🔍',
            description='Knowledge graph search',
            config_schema={
                'max_results': AgentConfig('max_results', 'number', 10, min_value=1, max_value=100),
                'similarity_threshold': AgentConfig('similarity_threshold', 'number', 0.7, min_value=0.0, max_value=1.0, step=0.1)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='multiquery',
            name='Multi-Query',
            category=AgentCategory.QUERY,
            inputs=['query'],
            outputs=['subqueries'],
            color='#667eea',
            icon='🔀',
            description='Break into sub-questions',
            config_schema={
                'max_subqueries': AgentConfig('max_subqueries', 'number', 5, min_value=2, max_value=10),
                'mode': AgentConfig('mode', 'select', 'research', options=['research', 'verify', 'plan_execute'])
            }
        ))

        # Process agents
        self.register_schema(AgentSchema(
            agent_id='embedder',
            name='Matryoshka Embedder',
            category=AgentCategory.PROCESS,
            inputs=['text'],
            outputs=['embeddings'],
            color='#f093fb',
            icon='📊',
            description='Multi-scale embeddings',
            config_schema={
                'scales': AgentConfig('scales', 'text', '96,192,384'),
                'normalize': AgentConfig('normalize', 'boolean', True)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='synthesizer',
            name='Synthesizer',
            category=AgentCategory.PROCESS,
            inputs=['text'],
            outputs=['synthesis'],
            color='#f093fb',
            icon='⚗️',
            description='Extract entities and motifs',
            config_schema={
                'extract_entities': AgentConfig('extract_entities', 'boolean', True),
                'extract_motifs': AgentConfig('extract_motifs', 'boolean', True)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='refiner',
            name='Recursive Refiner',
            category=AgentCategory.PROCESS,
            inputs=['spacetime'],
            outputs=['refined'],
            color='#f093fb',
            icon='✨',
            description='Quality refinement',
            config_schema={
                'strategy': AgentConfig('strategy', 'select', 'refine', options=['refine', 'critique', 'verify', 'elegance']),
                'max_iterations': AgentConfig('max_iterations', 'number', 3, min_value=1, max_value=10)
            }
        ))

        # Memory agents
        self.register_schema(AgentSchema(
            agent_id='store',
            name='Memory Store',
            category=AgentCategory.MEMORY,
            inputs=['data'],
            outputs=['stored'],
            color='#4facfe',
            icon='💾',
            description='Persist to graph+vector',
            config_schema={
                'backend': AgentConfig('backend', 'select', 'inmemory', options=['inmemory', 'hybrid', 'hyperspace'])
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='retrieve',
            name='Context Retriever',
            category=AgentCategory.MEMORY,
            inputs=['query'],
            outputs=['context'],
            color='#4facfe',
            icon='🔎',
            description='Retrieve context',
            config_schema={
                'k': AgentConfig('k', 'number', 5, min_value=1, max_value=50),
                'use_fusion': AgentConfig('use_fusion', 'boolean', False)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='fusion',
            name='Knowledge Fusion',
            category=AgentCategory.MEMORY,
            inputs=['query'],
            outputs=['expanded'],
            color='#4facfe',
            icon='🌐',
            description='Multi-hop traversal',
            config_schema={
                'max_depth': AgentConfig('max_depth', 'number', 2, min_value=1, max_value=5),
                'min_importance': AgentConfig('min_importance', 'number', 0.5, min_value=0.0, max_value=1.0, step=0.1)
            }
        ))

        # Decision agents
        self.register_schema(AgentSchema(
            agent_id='thompson',
            name='Thompson Sampler',
            category=AgentCategory.DECISION,
            inputs=['options'],
            outputs=['selected'],
            color='#43e97b',
            icon='🎲',
            description='Bayesian exploration',
            config_schema={
                'exploration_rate': AgentConfig('exploration_rate', 'number', 0.1, min_value=0.0, max_value=1.0, step=0.05)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='convergence',
            name='Convergence Engine',
            category=AgentCategory.DECISION,
            inputs=['features'],
            outputs=['decision'],
            color='#43e97b',
            icon='⚡',
            description='Decision collapse',
            config_schema={
                'strategy': AgentConfig('strategy', 'select', 'epsilon_greedy', options=['argmax', 'epsilon_greedy', 'bayesian_blend'])
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='safety',
            name='Safety Guardrails',
            category=AgentCategory.DECISION,
            inputs=['action'],
            outputs=['gated'],
            color='#43e97b',
            icon='🛡️',
            description='Risk gating',
            config_schema={
                'risk_threshold': AgentConfig('risk_threshold', 'select', 'MEDIUM', options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
                'enable_human_in_loop': AgentConfig('enable_human_in_loop', 'boolean', False)
            }
        ))

        # Output agents
        self.register_schema(AgentSchema(
            agent_id='response',
            name='Response Generator',
            category=AgentCategory.OUTPUT,
            inputs=['data'],
            outputs=['response'],
            color='#fa709a',
            icon='📝',
            description='Generate response',
            config_schema={
                'format': AgentConfig('format', 'select', 'text', options=['text', 'json', 'markdown'])
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='format',
            name='Format Converter',
            category=AgentCategory.OUTPUT,
            inputs=['data'],
            outputs=['formatted'],
            color='#fa709a',
            icon='🔄',
            description='Convert formats',
            config_schema={
                'output_format': AgentConfig('output_format', 'select', 'json', options=['json', 'markdown', 'html', 'yaml'])
            }
        ))

        # Control flow
        self.register_schema(AgentSchema(
            agent_id='conditional',
            name='Conditional Branch',
            category=AgentCategory.CONTROL,
            inputs=['condition'],
            outputs=['true', 'false'],
            color='#30cfd0',
            icon='🔀',
            description='If/else logic',
            config_schema={
                'condition_type': AgentConfig('condition_type', 'select', 'confidence', options=['confidence', 'count', 'custom']),
                'threshold': AgentConfig('threshold', 'number', 0.75, min_value=0.0, max_value=1.0, step=0.05)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='loop',
            name='Loop Iterator',
            category=AgentCategory.CONTROL,
            inputs=['data'],
            outputs=['iteration'],
            color='#30cfd0',
            icon='🔁',
            description='Repeat until condition',
            config_schema={
                'max_iterations': AgentConfig('max_iterations', 'number', 10, min_value=1, max_value=100)
            }
        ))

        self.register_schema(AgentSchema(
            agent_id='parallel',
            name='Parallel Executor',
            category=AgentCategory.CONTROL,
            inputs=['tasks'],
            outputs=['results'],
            color='#30cfd0',
            icon='⚡',
            description='Concurrent execution',
            config_schema={
                'max_concurrent': AgentConfig('max_concurrent', 'number', 5, min_value=1, max_value=20)
            }
        ))

        # LLM agents (NEW!)
        self.register_schema(AgentSchema(
            agent_id='llm_prompt',
            name='LLM Prompt',
            category=AgentCategory.LLM,
            inputs=['prompt'],
            outputs=['response'],
            color='#9333ea',
            icon='🤖',
            description='Execute LLM prompt with configurable model',
            config_schema={
                'model': AgentConfig(
                    'model', 'select', 'gpt-4',
                    options=['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet'],
                    description='LLM model to use'
                ),
                'temperature': AgentConfig(
                    'temperature', 'number', 0.7,
                    min_value=0.0, max_value=2.0, step=0.1,
                    description='Sampling temperature (0=deterministic, 2=creative)'
                ),
                'max_tokens': AgentConfig(
                    'max_tokens', 'number', 1000,
                    min_value=1, max_value=8000,
                    description='Maximum response tokens'
                )
            }
        ))

        # Code agents (NEW!)
        self.register_schema(AgentSchema(
            agent_id='code_analyzer',
            name='Code Analyzer',
            category=AgentCategory.CODE,
            inputs=['code'],
            outputs=['analysis'],
            color='#10b981',
            icon='📝',
            description='Analyze code for quality, security, and style issues',
            config_schema={
                'language': AgentConfig(
                    'language', 'select', 'python',
                    options=['python', 'javascript', 'typescript', 'java', 'go', 'rust'],
                    description='Programming language'
                ),
                'checks': AgentConfig(
                    'checks', 'multiselect', ['quality', 'security', 'style'],
                    options=['quality', 'security', 'style', 'performance', 'complexity'],
                    description='Analysis checks to perform'
                )
            }
        ))

        # Data agents (NEW!)
        self.register_schema(AgentSchema(
            agent_id='data_transformer',
            name='Data Transformer',
            category=AgentCategory.DATA,
            inputs=['data'],
            outputs=['transformed'],
            color='#f59e0b',
            icon='🔄',
            description='Transform and clean data',
            config_schema={
                'operations': AgentConfig(
                    'operations', 'multiselect', ['clean', 'normalize'],
                    options=['clean', 'normalize', 'deduplicate', 'validate', 'enrich'],
                    description='Transformation operations'
                ),
                'format': AgentConfig(
                    'format', 'select', 'json',
                    options=['json', 'csv', 'parquet', 'xml'],
                    description='Output format'
                )
            }
        ))

        # RAG agents (NEW!)
        self.register_schema(AgentSchema(
            agent_id='rag_query',
            name='RAG Query',
            category=AgentCategory.RAG,
            inputs=['query'],
            outputs=['response'],
            color='#ec4899',
            icon='🔍',
            description='Retrieval-Augmented Generation query',
            config_schema={
                'mode': AgentConfig(
                    'mode', 'select', 'verify',
                    options=['direct', 'verify', 'research', 'plan_execute'],
                    description='Reasoning mode'
                ),
                'max_sources': AgentConfig(
                    'max_sources', 'number', 5,
                    min_value=1, max_value=20,
                    description='Maximum sources to retrieve'
                ),
                'include_images': AgentConfig(
                    'include_images', 'boolean', False,
                    description='Include image/multimodal sources'
                )
            }
        ))

        logger.info(f"Loaded {len(self.agents)} built-in agents")

    def register_schema(self, schema: AgentSchema):
        """Register an agent schema"""
        if schema.agent_id in self.agents:
            logger.warning(f"Agent {schema.agent_id} already registered, overwriting")

        self.agents[schema.agent_id] = schema
        self.categories[schema.category].add(schema.agent_id)

        logger.info(f"Registered agent: {schema.name} ({schema.agent_id})")

    def register(
        self,
        agent_id: str,
        name: str = None,
        category: AgentCategory = AgentCategory.CUSTOM,
        **kwargs
    ):
        """
        Decorator for registering custom agents.

        Usage:
            @registry.register('my_agent', name='My Agent', category='custom')
            async def my_agent(inputs, config):
                return {'result': inputs['value'] * 2}
        """
        def decorator(func: Callable):
            schema = AgentSchema(
                agent_id=agent_id,
                name=name or agent_id.replace('_', ' ').title(),
                category=category,
                executor=func,
                async_executor=inspect.iscoroutinefunction(func),
                **kwargs
            )

            self.register_schema(schema)
            return func

        return decorator

    def get(self, agent_id: str) -> Optional[AgentSchema]:
        """Get agent schema by ID"""
        return self.agents.get(agent_id)

    def list_by_category(self, category: AgentCategory) -> List[AgentSchema]:
        """List all agents in a category"""
        agent_ids = self.categories.get(category, set())
        return [self.agents[aid] for aid in agent_ids]

    def search(self, query: str, category: Optional[AgentCategory] = None) -> List[AgentSchema]:
        """Search agents by name, description, or tags"""
        query_lower = query.lower()
        results = []

        agents_to_search = (
            self.list_by_category(category) if category
            else self.agents.values()
        )

        for agent in agents_to_search:
            if (query_lower in agent.name.lower() or
                query_lower in agent.description.lower() or
                any(query_lower in tag.lower() for tag in agent.tags)):
                results.append(agent)

        return results

    async def execute(
        self,
        agent_id: str,
        inputs: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an agent.

        Args:
            agent_id: Agent ID
            inputs: Input data
            config: Configuration parameters

        Returns:
            Agent output
        """
        schema = self.get(agent_id)
        if not schema:
            raise ValueError(f"Agent not found: {agent_id}")

        if not schema.executor:
            raise ValueError(f"Agent {agent_id} has no executor registered")

        # Validate config against schema
        validated_config = self.validate_config(schema, config)

        # Execute
        start_time = asyncio.get_event_loop().time()

        try:
            if schema.async_executor:
                result = await asyncio.wait_for(
                    schema.executor(inputs, validated_config),
                    timeout=schema.timeout
                )
            else:
                result = schema.executor(inputs, validated_config)

            # Update stats
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_stats(schema, latency_ms, success=True)

            return result

        except asyncio.TimeoutError:
            logger.error(f"Agent {agent_id} timed out after {schema.timeout}s")
            self._update_stats(schema, schema.timeout * 1000, success=False)
            raise

        except Exception as e:
            logger.error(f"Agent {agent_id} failed: {e}")
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_stats(schema, latency_ms, success=False)
            raise

    def validate_config(self, schema: AgentSchema, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema"""
        validated = {}

        for param_name, param_schema in schema.config_schema.items():
            value = config.get(param_name, param_schema.default)

            # Type validation
            if param_schema.param_type == 'number':
                value = float(value)
                if param_schema.min_value is not None:
                    value = max(value, param_schema.min_value)
                if param_schema.max_value is not None:
                    value = min(value, param_schema.max_value)

            elif param_schema.param_type == 'boolean':
                value = bool(value)

            elif param_schema.param_type == 'select':
                if param_schema.options and value not in param_schema.options:
                    value = param_schema.default

            validated[param_name] = value

        return validated

    def _update_stats(self, schema: AgentSchema, latency_ms: float, success: bool):
        """Update agent performance statistics"""
        schema.usage_count += 1

        # Exponential moving average for latency
        alpha = 0.1
        schema.avg_latency_ms = (
            alpha * latency_ms +
            (1 - alpha) * schema.avg_latency_ms
        )

        # Exponential moving average for success rate
        new_success = 1.0 if success else 0.0
        schema.success_rate = (
            alpha * new_success +
            (1 - alpha) * schema.success_rate
        )

    def export_schema(self) -> Dict[str, Any]:
        """Export all agent schemas as JSON-serializable dict"""
        return {
            'agents': {
                agent_id: {
                    'id': agent.agent_id,
                    'name': agent.name,
                    'category': agent.category.value,
                    'version': agent.version,
                    'inputs': agent.inputs,
                    'outputs': agent.outputs,
                    'color': agent.color,
                    'icon': agent.icon,
                    'description': agent.description,
                    'config': {
                        name: {
                            'name': cfg.name,
                            'type': cfg.param_type,
                            'default': cfg.default,
                            'required': cfg.required,
                            'options': cfg.options,
                            'description': cfg.description
                        }
                        for name, cfg in agent.config_schema.items()
                    },
                    'stats': {
                        'usage_count': agent.usage_count,
                        'avg_latency_ms': agent.avg_latency_ms,
                        'success_rate': agent.success_rate
                    }
                }
                for agent_id, agent in self.agents.items()
            },
            'categories': {
                cat.value: list(agent_ids)
                for cat, agent_ids in self.categories.items()
            }
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_usage = sum(a.usage_count for a in self.agents.values())

        return {
            'total_agents': len(self.agents),
            'categories': {
                cat.value: len(agents)
                for cat, agents in self.categories.items()
            },
            'total_usage': total_usage,
            'most_used': sorted(
                [(a.agent_id, a.usage_count) for a in self.agents.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


# Global registry instance
_registry = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry instance"""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


# Example custom agents
def register_example_agents():
    """Register example custom agents"""
    registry = get_registry()

    @registry.register(
        'sentiment_analyzer',
        name='Sentiment Analyzer',
        category=AgentCategory.ML,
        inputs=['text'],
        outputs=['sentiment'],
        color='#8b5cf6',
        icon='😊',
        description='Analyze sentiment of text (positive/negative/neutral)'
    )
    async def sentiment_analyzer(inputs, config):
        text = inputs.get('text', '')
        # Simple sentiment analysis (replace with actual ML model)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'poor'}

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = 'positive'
            score = 0.7 + (pos_count - neg_count) * 0.1
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = 0.3 - (neg_count - pos_count) * 0.1
        else:
            sentiment = 'neutral'
            score = 0.5

        return {
            'sentiment': sentiment,
            'score': max(0.0, min(1.0, score)),
            'positive_count': pos_count,
            'negative_count': neg_count
        }

    @registry.register(
        'web_scraper',
        name='Web Scraper',
        category=AgentCategory.INTEGRATION,
        inputs=['url'],
        outputs=['content'],
        color='#06b6d4',
        icon='🌐',
        description='Scrape content from web pages'
    )
    async def web_scraper(inputs, config):
        url = inputs.get('url', '')
        # Placeholder - integrate with HoloLoom's SpinningWheel
        return {
            'url': url,
            'title': 'Example Page',
            'content': 'Scraped content here...',
            'links': []
        }


if __name__ == "__main__":
    # Demo usage
    registry = get_registry()
    register_example_agents()

    print("=" * 80)
    print("HoloLoom Agent Registry")
    print("=" * 80)

    print(f"\nTotal agents: {len(registry.agents)}")

    print("\nAgents by category:")
    for category in AgentCategory:
        agents = registry.list_by_category(category)
        if agents:
            print(f"\n{category.value.upper()} ({len(agents)}):")
            for agent in agents:
                print(f"  • {agent.name} ({agent.agent_id})")
                print(f"    {agent.description}")

    print("\n" + "=" * 80)
    print("Export schema to use in frontend:")
    print("=" * 80)
    schema = registry.export_schema()
    print(json.dumps(schema, indent=2)[:500] + "...")
