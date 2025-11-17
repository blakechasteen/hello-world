#!/usr/bin/env python3
"""
Workflow Templates Library
===========================

Pre-built workflow templates for common use cases.

Features:
- RAG research pipeline
- Code review workflow
- Data processing pipeline
- Multi-agent collaboration
- Domain-specific templates

Usage:
    from HoloLoom.workflows.templates import WorkflowTemplates

    templates = WorkflowTemplates()

    # Get RAG research template
    workflow = templates.get('rag_research')

    # List all templates
    all_templates = templates.list_all()

    # Create from natural language
    workflow = templates.from_description("Analyze code for security issues")
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TemplateCategory(str, Enum):
    """Template categories"""
    RAG = "rag"
    CODE = "code"
    DATA = "data"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    INTEGRATION = "integration"
    CUSTOM = "custom"


@dataclass
class WorkflowTemplate:
    """Complete workflow template"""
    template_id: str
    name: str
    category: TemplateCategory
    description: str
    version: str = "1.0.0"

    # Workflow definition
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    author: str = "HoloLoom"
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"  # beginner, intermediate, advanced

    # Configuration
    default_inputs: Dict[str, Any] = field(default_factory=dict)
    configurable_params: List[str] = field(default_factory=list)

    # Stats
    downloads: int = 0
    rating: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class WorkflowTemplates:
    """
    Library of pre-built workflow templates.

    Provides ready-to-use workflows for common use cases.
    """

    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self._load_built_in_templates()

    def _load_built_in_templates(self):
        """Load built-in templates"""

        # 1. RAG Research Pipeline
        self.register(WorkflowTemplate(
            template_id='rag_research',
            name='RAG Research Pipeline',
            category=TemplateCategory.RAG,
            description='Multi-query research with RAG, synthesis, and refinement',
            tags=['rag', 'research', 'multiquery'],
            use_cases=[
                'Research complex topics',
                'Answer questions requiring multiple perspectives',
                'Academic literature review'
            ],
            difficulty='intermediate',
            nodes=[
                {
                    'id': 'node_1',
                    'agentType': 'multiquery',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'max_subqueries': 5,
                        'mode': 'research'
                    }
                },
                {
                    'id': 'node_2',
                    'agentType': 'rag_query',
                    'x': 300,
                    'y': 100,
                    'config': {
                        'mode': 'verify',
                        'max_sources': 10
                    }
                },
                {
                    'id': 'node_3',
                    'agentType': 'rag_query',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'mode': 'verify',
                        'max_sources': 10
                    }
                },
                {
                    'id': 'node_4',
                    'agentType': 'rag_query',
                    'x': 300,
                    'y': 300,
                    'config': {
                        'mode': 'verify',
                        'max_sources': 10
                    }
                },
                {
                    'id': 'node_5',
                    'agentType': 'synthesizer',
                    'x': 500,
                    'y': 200,
                    'config': {
                        'extract_entities': True,
                        'extract_motifs': True
                    }
                },
                {
                    'id': 'node_6',
                    'agentType': 'refiner',
                    'x': 700,
                    'y': 200,
                    'config': {
                        'strategy': 'verify',
                        'max_iterations': 3
                    }
                },
                {
                    'id': 'node_7',
                    'agentType': 'response',
                    'x': 900,
                    'y': 200,
                    'config': {
                        'format': 'markdown'
                    }
                }
            ],
            connections=[
                {'id': 'conn_1', 'from': 'node_1', 'to': 'node_2'},
                {'id': 'conn_2', 'from': 'node_1', 'to': 'node_3'},
                {'id': 'conn_3', 'from': 'node_1', 'to': 'node_4'},
                {'id': 'conn_4', 'from': 'node_2', 'to': 'node_5'},
                {'id': 'conn_5', 'from': 'node_3', 'to': 'node_5'},
                {'id': 'conn_6', 'from': 'node_4', 'to': 'node_5'},
                {'id': 'conn_7', 'from': 'node_5', 'to': 'node_6'},
                {'id': 'conn_8', 'from': 'node_6', 'to': 'node_7'}
            ],
            default_inputs={'query': 'Research Thompson Sampling algorithms'}
        ))

        # 2. Code Review Workflow
        self.register(WorkflowTemplate(
            template_id='code_review',
            name='Automated Code Review',
            category=TemplateCategory.CODE,
            description='Comprehensive code analysis with quality, security, and style checks',
            tags=['code', 'review', 'quality', 'security'],
            use_cases=[
                'Pre-commit code review',
                'Security vulnerability detection',
                'Code quality improvement'
            ],
            difficulty='intermediate',
            nodes=[
                {
                    'id': 'node_1',
                    'agentType': 'code_analyzer',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'language': 'python',
                        'checks': ['quality', 'security', 'style', 'complexity']
                    }
                },
                {
                    'id': 'node_2',
                    'agentType': 'safety',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'risk_threshold': 'MEDIUM',
                        'enable_human_in_loop': False
                    }
                },
                {
                    'id': 'node_3',
                    'agentType': 'conditional',
                    'x': 500,
                    'y': 200,
                    'config': {
                        'condition_type': 'confidence',
                        'threshold': 0.75
                    }
                },
                {
                    'id': 'node_4',
                    'agentType': 'llm_prompt',
                    'x': 700,
                    'y': 100,
                    'config': {
                        'model': 'gpt-4',
                        'temperature': 0.3,
                        'max_tokens': 1000
                    }
                },
                {
                    'id': 'node_5',
                    'agentType': 'response',
                    'x': 900,
                    'y': 100,
                    'config': {
                        'format': 'markdown'
                    }
                },
                {
                    'id': 'node_6',
                    'agentType': 'response',
                    'x': 700,
                    'y': 300,
                    'config': {
                        'format': 'json'
                    }
                }
            ],
            connections=[
                {'id': 'conn_1', 'from': 'node_1', 'to': 'node_2'},
                {'id': 'conn_2', 'from': 'node_2', 'to': 'node_3'},
                {'id': 'conn_3', 'from': 'node_3', 'to': 'node_4'},
                {'id': 'conn_4', 'from': 'node_4', 'to': 'node_5'},
                {'id': 'conn_5', 'from': 'node_3', 'to': 'node_6'}
            ],
            default_inputs={'code': 'def example():\n    pass'}
        ))

        # 3. Data Processing Pipeline
        self.register(WorkflowTemplate(
            template_id='data_pipeline',
            name='Data Processing Pipeline',
            category=TemplateCategory.DATA,
            description='ETL pipeline with validation, transformation, and storage',
            tags=['data', 'etl', 'pipeline', 'validation'],
            use_cases=[
                'Clean and validate incoming data',
                'Transform data for ML/analytics',
                'Data quality monitoring'
            ],
            difficulty='beginner',
            nodes=[
                {
                    'id': 'node_1',
                    'agentType': 'data_transformer',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'operations': ['clean', 'validate', 'normalize'],
                        'format': 'json'
                    }
                },
                {
                    'id': 'node_2',
                    'agentType': 'conditional',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'condition_type': 'custom',
                        'threshold': 0.9
                    }
                },
                {
                    'id': 'node_3',
                    'agentType': 'store',
                    'x': 500,
                    'y': 100,
                    'config': {
                        'backend': 'hybrid'
                    }
                },
                {
                    'id': 'node_4',
                    'agentType': 'response',
                    'x': 700,
                    'y': 100,
                    'config': {
                        'format': 'json'
                    }
                },
                {
                    'id': 'node_5',
                    'agentType': 'response',
                    'x': 500,
                    'y': 300,
                    'config': {
                        'format': 'json'
                    }
                }
            ],
            connections=[
                {'id': 'conn_1', 'from': 'node_1', 'to': 'node_2'},
                {'id': 'conn_2', 'from': 'node_2', 'to': 'node_3'},
                {'id': 'conn_3', 'from': 'node_3', 'to': 'node_4'},
                {'id': 'conn_4', 'from': 'node_2', 'to': 'node_5'}
            ],
            default_inputs={'data': {'records': []}}
        ))

        # 4. Multi-Agent Consensus
        self.register(WorkflowTemplate(
            template_id='multi_agent_consensus',
            name='Multi-Agent Consensus',
            category=TemplateCategory.ANALYSIS,
            description='Multiple agents vote on best answer with Thompson Sampling',
            tags=['multi-agent', 'consensus', 'thompson', 'voting'],
            use_cases=[
                'Aggregate multiple perspectives',
                'Reduce bias in decision making',
                'Ensemble predictions'
            ],
            difficulty='advanced',
            nodes=[
                {
                    'id': 'node_1',
                    'agentType': 'parallel',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'max_concurrent': 5
                    }
                },
                {
                    'id': 'node_2',
                    'agentType': 'hololoom',
                    'x': 300,
                    'y': 100,
                    'config': {
                        'pattern': 'fast',
                        'return_trace': True
                    }
                },
                {
                    'id': 'node_3',
                    'agentType': 'rag_query',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'mode': 'verify',
                        'max_sources': 5
                    }
                },
                {
                    'id': 'node_4',
                    'agentType': 'llm_prompt',
                    'x': 300,
                    'y': 300,
                    'config': {
                        'model': 'gpt-4',
                        'temperature': 0.7
                    }
                },
                {
                    'id': 'node_5',
                    'agentType': 'thompson',
                    'x': 500,
                    'y': 200,
                    'config': {
                        'exploration_rate': 0.15
                    }
                },
                {
                    'id': 'node_6',
                    'agentType': 'synthesizer',
                    'x': 700,
                    'y': 200,
                    'config': {
                        'extract_entities': True
                    }
                },
                {
                    'id': 'node_7',
                    'agentType': 'response',
                    'x': 900,
                    'y': 200,
                    'config': {
                        'format': 'markdown'
                    }
                }
            ],
            connections=[
                {'id': 'conn_1', 'from': 'node_1', 'to': 'node_2'},
                {'id': 'conn_2', 'from': 'node_1', 'to': 'node_3'},
                {'id': 'conn_3', 'from': 'node_1', 'to': 'node_4'},
                {'id': 'conn_4', 'from': 'node_2', 'to': 'node_5'},
                {'id': 'conn_5', 'from': 'node_3', 'to': 'node_5'},
                {'id': 'conn_6', 'from': 'node_4', 'to': 'node_5'},
                {'id': 'conn_7', 'from': 'node_5', 'to': 'node_6'},
                {'id': 'conn_8', 'from': 'node_6', 'to': 'node_7'}
            ],
            default_inputs={'query': 'What is the best approach to this problem?'}
        ))

        # 5. Simple Q&A
        self.register(WorkflowTemplate(
            template_id='simple_qa',
            name='Simple Q&A',
            category=TemplateCategory.RAG,
            description='Basic question answering with single query',
            tags=['qa', 'simple', 'beginner'],
            use_cases=[
                'Quick factual questions',
                'Simple lookups',
                'Getting started with workflows'
            ],
            difficulty='beginner',
            nodes=[
                {
                    'id': 'node_1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {
                        'pattern': 'fast',
                        'return_trace': False
                    }
                },
                {
                    'id': 'node_2',
                    'agentType': 'response',
                    'x': 300,
                    'y': 200,
                    'config': {
                        'format': 'text'
                    }
                }
            ],
            connections=[
                {'id': 'conn_1', 'from': 'node_1', 'to': 'node_2'}
            ],
            default_inputs={'query': 'What is Thompson Sampling?'}
        ))

        logger.info(f"Loaded {len(self.templates)} built-in templates")

    def register(self, template: WorkflowTemplate):
        """Register a workflow template"""
        self.templates[template.template_id] = template
        logger.info(f"Registered template: {template.name}")

    def get(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)

    def list_all(self) -> List[WorkflowTemplate]:
        """List all templates"""
        return list(self.templates.values())

    def list_by_category(self, category: TemplateCategory) -> List[WorkflowTemplate]:
        """List templates by category"""
        return [t for t in self.templates.values() if t.category == category]

    def search(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []

        for template in self.templates.values():
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags) or
                any(query_lower in uc.lower() for uc in template.use_cases)):
                results.append(template)

        return results

    def from_description(self, description: str) -> Optional[WorkflowTemplate]:
        """
        Generate workflow from natural language description.

        This is a simple keyword-based matcher. For more sophisticated
        generation, use the AI workflow generator.
        """
        desc_lower = description.lower()

        # Keyword matching
        if any(word in desc_lower for word in ['rag', 'research', 'multiple', 'perspectives']):
            return self.get('rag_research')

        if any(word in desc_lower for word in ['code', 'review', 'security', 'quality']):
            return self.get('code_review')

        if any(word in desc_lower for word in ['data', 'process', 'etl', 'transform']):
            return self.get('data_pipeline')

        if any(word in desc_lower for word in ['consensus', 'vote', 'ensemble', 'multiple agents']):
            return self.get('multi_agent_consensus')

        # Default to simple Q&A
        return self.get('simple_qa')

    def export_to_json(self, template_id: str) -> str:
        """Export template as JSON workflow definition"""
        template = self.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        workflow = {
            'version': template.version,
            'name': template.name,
            'nodes': template.nodes,
            'connections': template.connections,
            'metadata': {
                'template_id': template.template_id,
                'category': template.category.value,
                'description': template.description,
                'author': template.author,
                'tags': template.tags
            }
        }

        return json.dumps(workflow, indent=2)

    def clone_and_customize(
        self,
        template_id: str,
        name: str,
        customizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Clone a template and apply customizations.

        Args:
            template_id: Template to clone
            name: New workflow name
            customizations: Dict of node_id -> config overrides

        Returns:
            Customized workflow definition
        """
        template = self.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        # Deep copy nodes
        nodes = []
        for node in template.nodes:
            new_node = node.copy()
            if node['id'] in customizations:
                new_node['config'].update(customizations[node['id']])
            nodes.append(new_node)

        return {
            'version': template.version,
            'name': name,
            'nodes': nodes,
            'connections': template.connections.copy()
        }


if __name__ == "__main__":
    # Demo usage
    templates = WorkflowTemplates()

    print("=" * 80)
    print("HoloLoom Workflow Templates")
    print("=" * 80)

    print(f"\nTotal templates: {len(templates.templates)}")

    print("\nAvailable templates:")
    for template in templates.list_all():
        print(f"\n{template.name} ({template.difficulty})")
        print(f"  Category: {template.category.value}")
        print(f"  {template.description}")
        print(f"  Use cases:")
        for uc in template.use_cases:
            print(f"    • {uc}")
        print(f"  Nodes: {len(template.nodes)} | Connections: {len(template.connections)}")

    print("\n" + "=" * 80)
    print("Export RAG Research template:")
    print("=" * 80)
    rag_json = templates.export_to_json('rag_research')
    print(rag_json[:500] + "...")

    print("\n" + "=" * 80)
    print("Search for 'code' templates:")
    print("=" * 80)
    results = templates.search('code')
    for r in results:
        print(f"  • {r.name}")
