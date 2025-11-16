# Constants
CONSTANT_850 = 850  # Extracted magic number
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Auto-Workflow Generator
=================================

Generate workflows from natural language descriptions using pattern matching
and intelligent agent selection.

Usage:
    from workflow_generator import WorkflowGenerator

    generator = WorkflowGenerator()
    workflow = generator.generate("I want to research Thompson Sampling with multiple perspectives")

    # Returns a complete workflow JSON ready for execution
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """Types of workflow intents."""
    SIMPLE_QUERY = "simple_query"
    RESEARCH = "research"
    SAFETY_CHECK = "safety_check"
    MEMORY_OPERATION = "memory_operation"
    PROCESSING = "processing"
    CONDITIONAL_LOGIC = "conditional_logic"
    ITERATIVE = "iterative"
    PARALLEL_TASKS = "parallel_tasks"


@dataclass
class WorkflowIntent:
    """Parsed user intent."""
    intent_type: IntentType
    keywords: List[str]
    confidence: float
    suggested_agents: List[str]
    description: str


class WorkflowGenerator:
    """
    Generate workflows from natural language descriptions.

    Uses pattern matching and rule-based logic to convert text descriptions
    into executable workflow graphs.
    """

    def __init__(self):
        """Initialize the workflow generator."""
        self.patterns = self._initialize_patterns()
        self.agent_library = self._initialize_agent_library()

    def _initialize_patterns(self) -> Dict[IntentType, List[Tuple[str, float]]]:
        """
        Initialize intent detection patterns.

        Returns:
            Dict mapping intent types to (regex pattern, confidence) tuples
        """
        return {
            IntentType.SIMPLE_QUERY: [
                (r'\b(what is|explain|tell me about|define)\b', 0.9),
                (r'\b(simple|basic|quick) (?:question|query)\b', 0.85),
                (r'\b(answer|respond to)\b', 0.8),
            ],
            IntentType.RESEARCH: [
                (r'\b(research|investigate|explore|analyze|study)\b', 0.9),
                (r'\b(multiple (?:perspectives|angles|viewpoints))\b', 0.95),
                (r'\b(deep dive|comprehensive|thorough)\b', 0.85),
                (r'\b(sub-?questions?|break down|decompose)\b', 0.9),
            ],
            IntentType.SAFETY_CHECK: [
                (r'\b(safe|safety|secure|validate|verify|check)\b', 0.9),
                (r'\b(risk|dangerous|harmful)\b', 0.85),
                (r'\b(guardrails?|gate|protect)\b', 0.9),
                (r'\b(confidence|quality|threshold)\b', 0.75),
            ],
            IntentType.MEMORY_OPERATION: [
                (r'\b(store|save|persist|remember)\b', 0.9),
                (r'\b(retrieve|recall|search|find|lookup)\b', 0.9),
                (r'\b(memory|knowledge base|graph)\b', 0.85),
                (r'\b(context|history|past)\b', 0.7),
            ],
            IntentType.PROCESSING: [
                (r'\b(process|transform|convert|extract)\b', 0.85),
                (r'\b(embed|encode|vector)\b', 0.9),
                (r'\b(synthesize|analyze|parse)\b', 0.85),
                (r'\b(refine|improve|enhance|optimize)\b', 0.9),
            ],
            IntentType.CONDITIONAL_LOGIC: [
                (r'\b(if|when|unless|conditional)\b', 0.9),
                (r'\b(branch|split|route|decide)\b', 0.85),
                (r'\b(high|low) (?:confidence|quality)\b', 0.9),
                (r'\b(depends? on|based on)\b', 0.8),
            ],
            IntentType.ITERATIVE: [
                (r'\b(loop|repeat|iterate|until|while)\b', 0.95),
                (r'\b(multiple times|recursively|again)\b', 0.85),
                (r'\b(improve|refine) (?:until|iteratively)\b', 0.9),
            ],
            IntentType.PARALLEL_TASKS: [
                (r'\b(parallel|concurrent|simultaneously|at once)\b', 0.95),
                (r'\b(multiple (?:tasks|agents|processes))\b', 0.9),
                (r'\b(all at once|together)\b', 0.85),
            ],
        }

    def _initialize_agent_library(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize agent library with capabilities.

        Returns:
            Dict mapping agent types to their metadata
        """
        return {
            'hololoom': {
                'category': 'query',
                'capabilities': ['answer', 'query', 'reasoning', 'general'],
                'keywords': ['query', 'question', 'ask', 'answer'],
            },
            'search': {
                'category': 'query',
                'capabilities': ['search', 'find', 'lookup', 'retrieve'],
                'keywords': ['search', 'find', 'lookup', 'memory'],
            },
            'multiquery': {
                'category': 'query',
                'capabilities': ['research', 'explore', 'decompose', 'analyze'],
                'keywords': ['research', 'explore', 'multiple', 'perspectives', 'sub-question'],
            },
            'embedder': {
                'category': 'process',
                'capabilities': ['embed', 'encode', 'vector', 'representation'],
                'keywords': ['embed', 'vector', 'encode', 'representation'],
            },
            'synthesizer': {
                'category': 'process',
                'capabilities': ['extract', 'analyze', 'parse', 'entities'],
                'keywords': ['extract', 'entities', 'motifs', 'synthesize'],
            },
            'refiner': {
                'category': 'process',
                'capabilities': ['refine', 'improve', 'enhance', 'quality'],
                'keywords': ['refine', 'improve', 'enhance', 'quality', 'elegant'],
            },
            'store': {
                'category': 'memory',
                'capabilities': ['store', 'save', 'persist', 'write'],
                'keywords': ['store', 'save', 'persist', 'remember'],
            },
            'retrieve': {
                'category': 'memory',
                'capabilities': ['retrieve', 'recall', 'load', 'fetch'],
                'keywords': ['retrieve', 'recall', 'fetch', 'context'],
            },
            'fusion': {
                'category': 'memory',
                'capabilities': ['expand', 'traverse', 'connect', 'fuse'],
                'keywords': ['fusion', 'expand', 'traverse', 'connect', 'related'],
            },
            'thompson': {
                'category': 'decision',
                'capabilities': ['sample', 'explore', 'exploit', 'bandit'],
                'keywords': ['thompson', 'sampling', 'exploration', 'bandit'],
            },
            'convergence': {
                'category': 'decision',
                'capabilities': ['decide', 'converge', 'select', 'choose'],
                'keywords': ['decide', 'convergence', 'select', 'choose'],
            },
            'safety': {
                'category': 'decision',
                'capabilities': ['gate', 'check', 'validate', 'safe'],
                'keywords': ['safety', 'safe', 'gate', 'validate', 'risk'],
            },
            'response': {
                'category': 'output',
                'capabilities': ['generate', 'format', 'output', 'respond'],
                'keywords': ['response', 'output', 'generate', 'format'],
            },
            'format': {
                'category': 'output',
                'capabilities': ['convert', 'format', 'transform', 'serialize'],
                'keywords': ['format', 'convert', 'json', 'markdown', 'html'],
            },
            'conditional': {
                'category': 'control',
                'capabilities': ['branch', 'route', 'conditional', 'if'],
                'keywords': ['if', 'conditional', 'branch', 'route', 'depends'],
            },
            'loop': {
                'category': 'control',
                'capabilities': ['loop', 'iterate', 'repeat', 'while'],
                'keywords': ['loop', 'repeat', 'iterate', 'until', 'while'],
            },
            'parallel': {
                'category': 'control',
                'capabilities': ['parallel', 'concurrent', 'simultaneous'],
                'keywords': ['parallel', 'concurrent', 'simultaneously', 'together'],
            },
        }

    def generate(self, description: str) -> Dict[str, Any]:
        """
        Generate a workflow from natural language description.

        Args:
            description: Natural language workflow description

        Returns:
            Complete workflow JSON ready for execution

        Example:
            >>> gen = WorkflowGenerator()
            >>> workflow = gen.generate("Research Thompson Sampling with safety checks")
            >>> print(workflow['name'])
            'Research Thompson Sampling with safety checks'
        """
        # Parse intent
        intent = self.parse_intent(description)

        # Generate workflow based on intent
        if intent.intent_type == IntentType.SIMPLE_QUERY:
            workflow = self._generate_simple_query(description, intent)
        elif intent.intent_type == IntentType.RESEARCH:
            workflow = self._generate_research_workflow(description, intent)
        elif intent.intent_type == IntentType.SAFETY_CHECK:
            workflow = self._generate_safety_workflow(description, intent)
        elif intent.intent_type == IntentType.MEMORY_OPERATION:
            workflow = self._generate_memory_workflow(description, intent)
        elif intent.intent_type == IntentType.PROCESSING:
            workflow = self._generate_processing_workflow(description, intent)
        elif intent.intent_type == IntentType.CONDITIONAL_LOGIC:
            workflow = self._generate_conditional_workflow(description, intent)
        elif intent.intent_type == IntentType.ITERATIVE:
            workflow = self._generate_iterative_workflow(description, intent)
        elif intent.intent_type == IntentType.PARALLEL_TASKS:
            workflow = self._generate_parallel_workflow(description, intent)
        else:
            # Default to simple query
            workflow = self._generate_simple_query(description, intent)

        # Add metadata
        workflow['metadata'] = {
            'generated_from': description,
            'intent_type': intent.intent_type.value,
            'confidence': intent.confidence,
            'generator_version': '1.0'
        }

        return workflow

    def parse_intent(self, description: str) -> WorkflowIntent:
        """
        Parse user intent from description.

        Args:
            description: Natural language description

        Returns:
            Parsed workflow intent with confidence score
        """
        description_lower = description.lower()
        intent_scores = {}

        # Calculate scores for each intent type
        for intent_type, patterns in self.patterns.items():
            score = 0.0
            matched_keywords = []

            for pattern, confidence in patterns:
                if re.search(pattern, description_lower):
                    score += confidence
                    matched_keywords.append(pattern)

            intent_scores[intent_type] = (score, matched_keywords)

        # Select best intent
        best_intent = max(intent_scores.items(), key=lambda x: x[1][0])
        intent_type = best_intent[0]
        score, keywords = best_intent[1]

        # Normalize confidence
        confidence = min(score / len(self.patterns[intent_type]), 1.0)

        # Select agents based on keywords
        suggested_agents = self._select_agents(description_lower)

        return WorkflowIntent(
            intent_type=intent_type,
            keywords=keywords,
            confidence=confidence,
            suggested_agents=suggested_agents,
            description=description
        )

    def _select_agents(self, description: str) -> List[str]:
        """
        Select appropriate agents based on keywords in description.

        Args:
            description: Lowercased description

        Returns:
            List of agent types
        """
        selected = []

        for agent_type, metadata in self.agent_library.items():
            for keyword in metadata['keywords']:
                if keyword in description:
                    selected.append(agent_type)
                    break

        return selected

    def _generate_simple_query(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate simple query workflow."""
        return {
            'version': '1.0',
            'name': description[:50] + ('...' if len(description) > 50 else ''),
            'nodes': [
                {
                    'id': 'node-1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {'pattern': 'fast', 'return_trace': True}
                },
                {
                    'id': 'node-2',
                    'agentType': 'response',
                    'x': 400,
                    'y': 200,
                    'config': {'format': 'text'}
                }
            ],
            'connections': [
                {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'}
            ]
        }

    def _generate_research_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate multi-query research workflow."""
        nodes = [
            {
                'id': 'node-1',
                'agentType': 'multiquery',
                'x': 100,
                'y': 200,
                'config': {'max_subqueries': 5, 'mode': 'research'}
            },
            {
                'id': 'node-2',
                'agentType': 'hololoom',
                'x': 350,
                'y': 150,
                'config': {'pattern': 'fused', 'return_trace': True}
            },
            {
                'id': 'node-3',
                'agentType': 'hololoom',
                'x': 350,
                'y': 250,
                'config': {'pattern': 'fused', 'return_trace': True}
            },
            {
                'id': 'node-4',
                'agentType': 'synthesizer',
                'x': 600,
                'y': 200,
                'config': {'extract_entities': True, 'extract_motifs': True}
            }
        ]

        connections = [
            {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
            {'id': 'conn-2', 'from': 'node-1', 'to': 'node-3'},
            {'id': 'conn-3', 'from': 'node-2', 'to': 'node-4'},
            {'id': 'conn-4', 'from': 'node-3', 'to': 'node-4'},
        ]

        # Add refiner if quality/improvement mentioned
        if any(k in description.lower() for k in ['refine', 'quality', 'improve', 'elegant']):
            nodes.append({
                'id': 'node-5',
                'agentType': 'refiner',
                'x': 850,
                'y': 200,
                'config': {'strategy': 'elegance', 'max_iterations': 3}
            })
            nodes.append({
                'id': 'node-6',
                'agentType': 'response',
                'x': 1100,
                'y': 200,
                'config': {'format': 'markdown'}
            })
            connections.extend([
                {'id': 'conn-5', 'from': 'node-4', 'to': 'node-5'},
                {'id': 'conn-6', 'from': 'node-5', 'to': 'node-6'},
            ])
        else:
            nodes.append({
                'id': 'node-5',
                'agentType': 'response',
                'x': 850,
                'y': 200,
                'config': {'format': 'markdown'}
            })
            connections.append({'id': 'conn-5', 'from': 'node-4', 'to': 'node-5'})

        return {
            'version': '1.0',
            'name': f"Research: {description[:40]}",
            'nodes': nodes,
            'connections': connections
        }

    def _generate_safety_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate safety-gated workflow."""
        return {
            'version': '1.0',
            'name': f"Safety-Gated: {description[:40]}",
            'nodes': [
                {
                    'id': 'node-1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {'pattern': 'fast', 'return_trace': True}
                },
                {
                    'id': 'node-2',
                    'agentType': 'safety',
                    'x': 350,
                    'y': 200,
                    'config': {
                        'risk_threshold': 'MEDIUM',
                        'enable_human_in_loop': False
                    }
                },
                {
                    'id': 'node-3',
                    'agentType': 'conditional',
                    'x': 600,
                    'y': 200,
                    'config': {'condition_type': 'confidence', 'threshold': 0.75}
                },
                {
                    'id': 'node-4',
                    'agentType': 'response',
                    'x': 850,
                    'y': 150,
                    'config': {'format': 'text'}
                },
                {
                    'id': 'node-5',
                    'agentType': 'refiner',
                    'x': 850,
                    'y': 250,
                    'config': {'strategy': 'verify', 'max_iterations': 2}
                },
                {
                    'id': 'node-6',
                    'agentType': 'response',
                    'x': 1100,
                    'y': 250,
                    'config': {'format': 'text'}
                }
            ],
            'connections': [
                {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                {'id': 'conn-3', 'from': 'node-3', 'to': 'node-4'},
                {'id': 'conn-4', 'from': 'node-3', 'to': 'node-5'},
                {'id': 'conn-5', 'from': 'node-5', 'to': 'node-6'},
            ]
        }

    def _generate_memory_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate memory operation workflow."""
        if any(k in description.lower() for k in ['store', 'save', 'persist']):
            # Store workflow
            return {
                'version': '1.0',
                'name': f"Store: {description[:40]}",
                'nodes': [
                    {
                        'id': 'node-1',
                        'agentType': 'hololoom',
                        'x': 100,
                        'y': 200,
                        'config': {'pattern': 'fast', 'return_trace': True}
                    },
                    {
                        'id': 'node-2',
                        'agentType': 'store',
                        'x': 350,
                        'y': 200,
                        'config': {'backend': 'inmemory'}
                    },
                    {
                        'id': 'node-3',
                        'agentType': 'response',
                        'x': 600,
                        'y': 200,
                        'config': {'format': 'text'}
                    }
                ],
                'connections': [
                    {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                    {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                ]
            }
        else:
            # Retrieve workflow
            return {
                'version': '1.0',
                'name': f"Retrieve: {description[:40]}",
                'nodes': [
                    {
                        'id': 'node-1',
                        'agentType': 'retrieve',
                        'x': 100,
                        'y': 200,
                        'config': {'k': 10, 'use_fusion': False}
                    },
                    {
                        'id': 'node-2',
                        'agentType': 'hololoom',
                        'x': 350,
                        'y': 200,
                        'config': {'pattern': 'fast', 'return_trace': True}
                    },
                    {
                        'id': 'node-3',
                        'agentType': 'response',
                        'x': 600,
                        'y': 200,
                        'config': {'format': 'text'}
                    }
                ],
                'connections': [
                    {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                    {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                ]
            }

    def _generate_processing_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate processing workflow."""
        nodes = [
            {
                'id': 'node-1',
                'agentType': 'hololoom',
                'x': 100,
                'y': 200,
                'config': {'pattern': 'fast', 'return_trace': True}
            }
        ]
        connections = []
        x_pos = 350
        node_id = 2

        # Add processing agents based on keywords
        if 'embed' in description.lower() or 'vector' in description.lower():
            nodes.append({
                'id': f'node-{node_id}',
                'agentType': 'embedder',
                'x': x_pos,
                'y': 200,
                'config': {'scales': '96,192,384', 'normalize': True}
            })
            connections.append({
                'id': f'conn-{node_id-1}',
                'from': f'node-{node_id-1}',
                'to': f'node-{node_id}'
            })
            node_id += 1
            x_pos += 250

        if 'synthesize' in description.lower() or 'extract' in description.lower():
            nodes.append({
                'id': f'node-{node_id}',
                'agentType': 'synthesizer',
                'x': x_pos,
                'y': 200,
                'config': {'extract_entities': True, 'extract_motifs': True}
            })
            connections.append({
                'id': f'conn-{node_id-1}',
                'from': f'node-{node_id-1}',
                'to': f'node-{node_id}'
            })
            node_id += 1
            x_pos += 250

        if 'refine' in description.lower() or 'improve' in description.lower():
            nodes.append({
                'id': f'node-{node_id}',
                'agentType': 'refiner',
                'x': x_pos,
                'y': 200,
                'config': {'strategy': 'refine', 'max_iterations': 3}
            })
            connections.append({
                'id': f'conn-{node_id-1}',
                'from': f'node-{node_id-1}',
                'to': f'node-{node_id}'
            })
            node_id += 1
            x_pos += 250

        # Add output
        nodes.append({
            'id': f'node-{node_id}',
            'agentType': 'response',
            'x': x_pos,
            'y': 200,
            'config': {'format': 'text'}
        })
        connections.append({
            'id': f'conn-{node_id-1}',
            'from': f'node-{node_id-1}',
            'to': f'node-{node_id}'
        })

        return {
            'version': '1.0',
            'name': f"Process: {description[:40]}",
            'nodes': nodes,
            'connections': connections
        }

    def _generate_conditional_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate conditional branching workflow."""
        return {
            'version': '1.0',
            'name': f"Conditional: {description[:40]}",
            'nodes': [
                {
                    'id': 'node-1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {'pattern': 'fast', 'return_trace': True}
                },
                {
                    'id': 'node-2',
                    'agentType': 'conditional',
                    'x': 350,
                    'y': 200,
                    'config': {'condition_type': 'confidence', 'threshold': 0.75}
                },
                {
                    'id': 'node-3',
                    'agentType': 'response',
                    'x': 600,
                    'y': 150,
                    'config': {'format': 'text'}
                },
                {
                    'id': 'node-4',
                    'agentType': 'refiner',
                    'x': 600,
                    'y': 250,
                    'config': {'strategy': 'verify', 'max_iterations': 2}
                },
                {
                    'id': 'node-5',
                    'agentType': 'response',
                    'x': 850,
                    'y': 250,
                    'config': {'format': 'text'}
                }
            ],
            'connections': [
                {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                {'id': 'conn-3', 'from': 'node-2', 'to': 'node-4'},
                {'id': 'conn-4', 'from': 'node-4', 'to': 'node-5'},
            ]
        }

    def _generate_iterative_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate iterative loop workflow."""
        return {
            'version': '1.0',
            'name': f"Loop: {description[:40]}",
            'nodes': [
                {
                    'id': 'node-1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {'pattern': 'fast', 'return_trace': True}
                },
                {
                    'id': 'node-2',
                    'agentType': 'loop',
                    'x': 350,
                    'y': 200,
                    'config': {
                        'max_iterations': 10,
                        'break_condition': 'confidence > 0.9'
                    }
                },
                {
                    'id': 'node-3',
                    'agentType': 'refiner',
                    'x': 600,
                    'y': 200,
                    'config': {'strategy': 'refine', 'max_iterations': 3}
                },
                {
                    'id': 'node-4',
                    'agentType': 'response',
                    'x': 850,
                    'y': 200,
                    'config': {'format': 'text'}
                }
            ],
            'connections': [
                {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                {'id': 'conn-3', 'from': 'node-3', 'to': 'node-4'},
            ]
        }

    def _generate_parallel_workflow(
        self,
        description: str,
        intent: WorkflowIntent
    ) -> Dict[str, Any]:
        """Generate parallel execution workflow."""
        return {
            'version': '1.0',
            'name': f"Parallel: {description[:40]}",
            'nodes': [
                {
                    'id': 'node-1',
                    'agentType': 'multiquery',
                    'x': 100,
                    'y': 200,
                    'config': {'max_subqueries': 5, 'mode': 'research'}
                },
                {
                    'id': 'node-2',
                    'agentType': 'parallel',
                    'x': 350,
                    'y': 200,
                    'config': {'max_concurrent': 5}
                },
                {
                    'id': 'node-3',
                    'agentType': 'synthesizer',
                    'x': 600,
                    'y': 200,
                    'config': {'extract_entities': True, 'extract_motifs': True}
                },
                {
                    'id': 'node-4',
                    'agentType': 'response',
                    'x': CONSTANT_850,
                    'y': 200,
                    'config': {'format': 'markdown'}
                }
            ],
            'connections': [
                {'id': 'conn-1', 'from': 'node-1', 'to': 'node-2'},
                {'id': 'conn-2', 'from': 'node-2', 'to': 'node-3'},
                {'id': 'conn-3', 'from': 'node-3', 'to': 'node-4'},
            ]
        }


# Command-line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python workflow_generator.py '<description>'")
        print("\nExamples:")
        print("  python workflow_generator.py 'Research Thompson Sampling'")
        print("  python workflow_generator.py 'Query with safety checks'")
        print("  python workflow_generator.py 'Process and refine my data'")
        sys.exit(1)

    description = sys.argv[1]
    generator = WorkflowGenerator()
    workflow = generator.generate(description)

    print(json.dumps(workflow, indent=2))
