#!/usr/bin/env python3
"""
AI Workflow Generator
====================

Automatically generate workflows from natural language descriptions.

Features:
- Natural language → workflow graph
- Intent detection
- Node placement optimization
- Validation and error correction
- Multi-turn workflow refinement

Usage:
    from HoloLoom.workflows.ai_generator import AIWorkflowGenerator

    generator = AIWorkflowGenerator()

    # Generate from description
    workflow = await generator.generate(
        "Create a workflow that analyzes code for security issues, "
        "then uses an LLM to suggest fixes, and saves the results"
    )

    # Refine workflow
    refined = await generator.refine(
        workflow,
        "Make it run in parallel and add error handling"
    )
"""

import asyncio
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from HoloLoom.workflows.agent_registry import AgentRegistry, AgentCategory, get_registry
    from HoloLoom.workflows.templates import WorkflowTemplates
except ImportError:
    # Fallback for standalone use
    AgentRegistry = None
    WorkflowTemplates = None

logger = logging.getLogger(__name__)


@dataclass
class WorkflowIntent:
    """Detected intent from natural language"""
    primary_goal: str  # 'analyze', 'transform', 'query', 'review', etc.
    secondary_goals: List[str] = field(default_factory=list)
    agents_needed: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    input_type: Optional[str] = None
    output_format: Optional[str] = None
    complexity: str = 'medium'  # simple, medium, complex


class AIWorkflowGenerator:
    """
    AI-powered workflow generator that creates workflows from natural language.

    Uses pattern matching, templates, and heuristics to generate workflows.
    Can be enhanced with LLM integration for more sophisticated generation.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        self.registry = registry or get_registry()
        self.templates = WorkflowTemplates() if WorkflowTemplates else None

        # Intent keywords
        self.intent_patterns = {
            'analyze': ['analyze', 'analysis', 'review', 'check', 'inspect', 'examine'],
            'transform': ['transform', 'convert', 'process', 'clean', 'normalize'],
            'query': ['query', 'search', 'find', 'retrieve', 'lookup', 'ask'],
            'generate': ['generate', 'create', 'produce', 'synthesize', 'make'],
            'decide': ['decide', 'choose', 'select', 'vote', 'consensus'],
            'validate': ['validate', 'verify', 'check', 'confirm', 'test'],
            'store': ['store', 'save', 'persist', 'keep', 'cache'],
            'parallel': ['parallel', 'concurrent', 'simultaneous', 'together'],
            'sequential': ['sequential', 'series', 'chain', 'pipeline']
        }

        # Agent type mappings
        self.agent_mappings = {
            'code': ['code_analyzer', 'llm_prompt'],
            'security': ['code_analyzer', 'safety'],
            'data': ['data_transformer', 'store'],
            'llm': ['llm_prompt', 'structured_llm'],
            'rag': ['rag_query', 'multiquery'],
            'research': ['multiquery', 'rag_query', 'synthesizer'],
            'sentiment': ['sentiment_analyzer'],
            'web': ['web_scraper']
        }

    async def generate(
        self,
        description: str,
        use_llm: bool = False
    ) -> Dict[str, Any]:
        """
        Generate workflow from natural language description.

        Args:
            description: Natural language workflow description
            use_llm: Use LLM for generation (if available)

        Returns:
            Workflow definition (nodes + connections)
        """
        logger.info(f"Generating workflow from: {description}")

        # Step 1: Detect intent
        intent = self.detect_intent(description)
        logger.info(f"Detected intent: {intent.primary_goal}")
        logger.info(f"Agents needed: {intent.agents_needed}")

        # Step 2: Check if template matches
        if self.templates:
            template = self.templates.from_description(description)
            if template:
                logger.info(f"Using template: {template.name}")
                return self._workflow_from_template(template)

        # Step 3: Generate from scratch
        if use_llm:
            return await self._generate_with_llm(description, intent)
        else:
            return self._generate_heuristic(description, intent)

    def detect_intent(self, description: str) -> WorkflowIntent:
        """
        Detect user intent from natural language description.

        Uses keyword matching and patterns to identify:
        - Primary goal (analyze, transform, query, etc.)
        - Required agents
        - Constraints (parallel, error handling, etc.)
        - Input/output types
        """
        desc_lower = description.lower()

        # Find primary goal
        primary_goal = 'query'  # default
        for goal, keywords in self.intent_patterns.items():
            if any(kw in desc_lower for kw in keywords):
                primary_goal = goal
                break

        # Find secondary goals
        secondary_goals = []
        for goal, keywords in self.intent_patterns.items():
            if goal != primary_goal and any(kw in desc_lower for kw in keywords):
                secondary_goals.append(goal)

        # Find needed agents
        agents_needed = []
        for domain, agent_types in self.agent_mappings.items():
            if domain in desc_lower:
                agents_needed.extend(agent_types)

        # Remove duplicates
        agents_needed = list(set(agents_needed))

        # Detect constraints
        constraints = []
        if any(kw in desc_lower for kw in self.intent_patterns['parallel']):
            constraints.append('parallel')
        if 'error' in desc_lower or 'safe' in desc_lower:
            constraints.append('error_handling')
        if 'refine' in desc_lower or 'improve' in desc_lower:
            constraints.append('refinement')

        # Detect input/output types
        input_type = None
        if 'code' in desc_lower:
            input_type = 'code'
        elif 'data' in desc_lower or 'json' in desc_lower:
            input_type = 'data'
        elif 'text' in desc_lower or 'query' in desc_lower:
            input_type = 'text'

        output_format = None
        if 'json' in desc_lower:
            output_format = 'json'
        elif 'markdown' in desc_lower:
            output_format = 'markdown'

        # Complexity estimation
        complexity = 'simple'
        if len(agents_needed) > 3 or len(secondary_goals) > 2:
            complexity = 'complex'
        elif len(agents_needed) > 1:
            complexity = 'medium'

        return WorkflowIntent(
            primary_goal=primary_goal,
            secondary_goals=secondary_goals,
            agents_needed=agents_needed,
            constraints=constraints,
            input_type=input_type,
            output_format=output_format,
            complexity=complexity
        )

    def _generate_heuristic(self, description: str, intent: WorkflowIntent) -> Dict[str, Any]:
        """
        Generate workflow using heuristics (no LLM needed).

        Creates a simple linear or branched workflow based on detected intent.
        """
        nodes = []
        connections = []
        node_id = 1
        x_pos = 100
        y_spacing = 100

        # Add starting agent
        start_agent = self._select_start_agent(intent)
        nodes.append({
            'id': f'node_{node_id}',
            'agentType': start_agent,
            'x': x_pos,
            'y': 200,
            'config': self._get_default_config(start_agent)
        })
        prev_node_id = node_id
        node_id += 1
        x_pos += 200

        # Add processing agents
        for agent_type in intent.agents_needed:
            if agent_type == start_agent:
                continue

            nodes.append({
                'id': f'node_{node_id}',
                'agentType': agent_type,
                'x': x_pos,
                'y': 200,
                'config': self._get_default_config(agent_type)
            })

            connections.append({
                'id': f'conn_{len(connections) + 1}',
                'from': f'node_{prev_node_id}',
                'to': f'node_{node_id}'
            })

            prev_node_id = node_id
            node_id += 1
            x_pos += 200

        # Add error handling if requested
        if 'error_handling' in intent.constraints:
            nodes.append({
                'id': f'node_{node_id}',
                'agentType': 'safety',
                'x': x_pos,
                'y': 200,
                'config': {'risk_threshold': 'MEDIUM'}
            })

            connections.append({
                'id': f'conn_{len(connections) + 1}',
                'from': f'node_{prev_node_id}',
                'to': f'node_{node_id}'
            })

            prev_node_id = node_id
            node_id += 1
            x_pos += 200

        # Add refinement if requested
        if 'refinement' in intent.constraints:
            nodes.append({
                'id': f'node_{node_id}',
                'agentType': 'refiner',
                'x': x_pos,
                'y': 200,
                'config': {'strategy': 'verify', 'max_iterations': 3}
            })

            connections.append({
                'id': f'conn_{len(connections) + 1}',
                'from': f'node_{prev_node_id}',
                'to': f'node_{node_id}'
            })

            prev_node_id = node_id
            node_id += 1
            x_pos += 200

        # Add output node
        output_format = intent.output_format or 'text'
        nodes.append({
            'id': f'node_{node_id}',
            'agentType': 'response',
            'x': x_pos,
            'y': 200,
            'config': {'format': output_format}
        })

        connections.append({
            'id': f'conn_{len(connections) + 1}',
            'from': f'node_{prev_node_id}',
            'to': f'node_{node_id}'
        })

        return {
            'version': '1.0',
            'name': self._generate_name(description),
            'nodes': nodes,
            'connections': connections,
            'metadata': {
                'generated': True,
                'generated_at': datetime.now().isoformat(),
                'description': description,
                'intent': {
                    'primary_goal': intent.primary_goal,
                    'complexity': intent.complexity
                }
            }
        }

    async def _generate_with_llm(self, description: str, intent: WorkflowIntent) -> Dict[str, Any]:
        """
        Generate workflow using LLM.

        This requires an LLM integration (OpenAI, Anthropic, etc.)
        For now, falls back to heuristic generation.
        """
        logger.warning("LLM generation not implemented yet, using heuristics")
        return self._generate_heuristic(description, intent)

    def _select_start_agent(self, intent: WorkflowIntent) -> str:
        """Select the starting agent based on intent"""
        if intent.primary_goal == 'query':
            return 'hololoom'
        elif intent.primary_goal == 'analyze':
            if 'code_analyzer' in intent.agents_needed:
                return 'code_analyzer'
            return 'hololoom'
        elif intent.primary_goal == 'transform':
            return 'data_transformer'
        elif intent.primary_goal == 'generate':
            return 'llm_prompt'
        else:
            return 'hololoom'

    def _get_default_config(self, agent_type: str) -> Dict[str, Any]:
        """Get default configuration for agent type"""
        defaults = {
            'hololoom': {'pattern': 'fast', 'return_trace': True},
            'code_analyzer': {'language': 'python', 'checks': ['quality', 'security']},
            'data_transformer': {'operations': ['clean', 'normalize'], 'format': 'json'},
            'llm_prompt': {'model': 'gpt-4', 'temperature': 0.7, 'max_tokens': 1000},
            'rag_query': {'mode': 'verify', 'max_sources': 5},
            'multiquery': {'max_subqueries': 5, 'mode': 'research'},
            'safety': {'risk_threshold': 'MEDIUM'},
            'refiner': {'strategy': 'verify', 'max_iterations': 3},
            'response': {'format': 'text'}
        }

        return defaults.get(agent_type, {})

    def _generate_name(self, description: str) -> str:
        """Generate workflow name from description"""
        # Take first few words, capitalize
        words = description.split()[:5]
        name = ' '.join(words).strip('.,!?')
        if len(name) > 50:
            name = name[:47] + '...'
        return name.title()

    def _workflow_from_template(self, template) -> Dict[str, Any]:
        """Convert template to workflow definition"""
        return {
            'version': template.version,
            'name': template.name,
            'nodes': template.nodes,
            'connections': template.connections,
            'metadata': {
                'template_id': template.template_id,
                'category': template.category.value
            }
        }

    async def refine(
        self,
        workflow: Dict[str, Any],
        refinement: str
    ) -> Dict[str, Any]:
        """
        Refine existing workflow based on natural language feedback.

        Args:
            workflow: Existing workflow definition
            refinement: Natural language refinement request

        Returns:
            Refined workflow
        """
        logger.info(f"Refining workflow: {refinement}")

        # Detect refinement intent
        ref_lower = refinement.lower()

        # Add parallelization
        if 'parallel' in ref_lower:
            workflow = self._add_parallelization(workflow)

        # Add error handling
        if 'error' in ref_lower or 'safe' in ref_lower:
            workflow = self._add_error_handling(workflow)

        # Add refinement step
        if 'refine' in ref_lower or 'improve' in ref_lower:
            workflow = self._add_refinement_step(workflow)

        # Add more agents
        if 'add' in ref_lower:
            # Extract agent type from refinement text
            for agent_type in self.registry.agents.keys():
                if agent_type in ref_lower:
                    workflow = self._add_agent(workflow, agent_type)

        return workflow

    def _add_parallelization(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Add parallel execution to workflow"""
        # Find all nodes without dependencies
        all_nodes = {n['id'] for n in workflow['nodes']}
        nodes_with_deps = {c['to'] for c in workflow['connections']}
        start_nodes = all_nodes - nodes_with_deps

        if len(start_nodes) <= 1:
            return workflow  # Already linear or single start

        # Add parallel node
        parallel_node = {
            'id': f'node_parallel_{len(workflow["nodes"]) + 1}',
            'agentType': 'parallel',
            'x': 50,
            'y': 200,
            'config': {'max_concurrent': len(start_nodes)}
        }

        workflow['nodes'].insert(0, parallel_node)

        # Connect parallel node to all start nodes
        for start_id in start_nodes:
            workflow['connections'].insert(0, {
                'id': f'conn_parallel_{len(workflow["connections"]) + 1}',
                'from': parallel_node['id'],
                'to': start_id
            })

        return workflow

    def _add_error_handling(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Add safety guardrails to workflow"""
        # Find the last node
        connections_from = {c['from'] for c in workflow['connections']}
        connections_to = {c['to'] for c in workflow['connections']}
        end_nodes = connections_from - connections_to

        if not end_nodes:
            return workflow

        end_node_id = list(end_nodes)[0]

        # Insert safety node before end
        safety_node = {
            'id': f'node_safety_{len(workflow["nodes"]) + 1}',
            'agentType': 'safety',
            'x': 500,
            'y': 200,
            'config': {'risk_threshold': 'MEDIUM', 'enable_human_in_loop': False}
        }

        workflow['nodes'].append(safety_node)

        # Update connections
        for conn in workflow['connections']:
            if conn['to'] == end_node_id:
                conn['to'] = safety_node['id']

        workflow['connections'].append({
            'id': f'conn_safety_{len(workflow["connections"]) + 1}',
            'from': safety_node['id'],
            'to': end_node_id
        })

        return workflow

    def _add_refinement_step(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Add refinement step to workflow"""
        # Similar to _add_error_handling, but with refiner
        connections_from = {c['from'] for c in workflow['connections']}
        connections_to = {c['to'] for c in workflow['connections']}
        end_nodes = connections_from - connections_to

        if not end_nodes:
            return workflow

        end_node_id = list(end_nodes)[0]

        refiner_node = {
            'id': f'node_refiner_{len(workflow["nodes"]) + 1}',
            'agentType': 'refiner',
            'x': 600,
            'y': 200,
            'config': {'strategy': 'verify', 'max_iterations': 3}
        }

        workflow['nodes'].append(refiner_node)

        for conn in workflow['connections']:
            if conn['to'] == end_node_id:
                conn['to'] = refiner_node['id']

        workflow['connections'].append({
            'id': f'conn_refiner_{len(workflow["connections"]) + 1}',
            'from': refiner_node['id'],
            'to': end_node_id
        })

        return workflow

    def _add_agent(self, workflow: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """Add a new agent to workflow"""
        # Add at the end of the chain
        new_node = {
            'id': f'node_{len(workflow["nodes"]) + 1}',
            'agentType': agent_type,
            'x': (len(workflow["nodes"]) + 1) * 200,
            'y': 200,
            'config': self._get_default_config(agent_type)
        }

        workflow['nodes'].append(new_node)

        # Connect to previous end node
        connections_from = {c['from'] for c in workflow['connections']}
        connections_to = {c['to'] for c in workflow['connections']}
        end_nodes = connections_from - connections_to

        if end_nodes:
            prev_end = list(end_nodes)[0]
            workflow['connections'].append({
                'id': f'conn_{len(workflow["connections"]) + 1}',
                'from': prev_end,
                'to': new_node['id']
            })

        return workflow

    def validate(self, workflow: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate generated workflow.

        Returns:
            (is_valid, errors)
        """
        errors = []

        # Check required fields
        if 'nodes' not in workflow:
            errors.append("Missing 'nodes' field")
        if 'connections' not in workflow:
            errors.append("Missing 'connections' field")

        if errors:
            return False, errors

        # Check for cycles
        if self._has_cycles(workflow):
            errors.append("Workflow contains cycles")

        # Check for disconnected nodes
        node_ids = {n['id'] for n in workflow['nodes']}
        connected_nodes = set()
        for conn in workflow['connections']:
            connected_nodes.add(conn['from'])
            connected_nodes.add(conn['to'])

        disconnected = node_ids - connected_nodes
        if disconnected and len(workflow['nodes']) > 1:
            errors.append(f"Disconnected nodes: {disconnected}")

        # Check for invalid agent types
        valid_agents = set(self.registry.agents.keys())
        for node in workflow['nodes']:
            if node['agentType'] not in valid_agents:
                errors.append(f"Invalid agent type: {node['agentType']}")

        return len(errors) == 0, errors

    def _has_cycles(self, workflow: Dict[str, Any]) -> bool:
        """Check for cycles in workflow"""
        visited = set()
        rec_stack = set()

        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)

            for conn in workflow['connections']:
                if conn['from'] == node_id:
                    if conn['to'] not in visited:
                        if dfs(conn['to']):
                            return True
                    elif conn['to'] in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for node in workflow['nodes']:
            if node['id'] not in visited:
                if dfs(node['id']):
                    return True

        return False


if __name__ == "__main__":
    # Demo usage
    async def main():
        generator = AIWorkflowGenerator()

        print("=" * 80)
        print("AI Workflow Generator Demo")
        print("=" * 80)

        # Example 1: Code analysis
        print("\nExample 1: Code Analysis")
        print("-" * 80)
        workflow1 = await generator.generate(
            "Create a workflow that analyzes Python code for security issues, "
            "uses an LLM to suggest fixes, and saves the results in JSON format"
        )
        print(json.dumps(workflow1, indent=2))

        # Example 2: Research pipeline
        print("\nExample 2: Research Pipeline")
        print("-" * 80)
        workflow2 = await generator.generate(
            "Build a research workflow that searches multiple sources, "
            "synthesizes the findings, and refines the answer"
        )
        print(f"Generated workflow: {workflow2['name']}")
        print(f"Nodes: {len(workflow2['nodes'])}")
        print(f"Connections: {len(workflow2['connections'])}")

        # Example 3: Refinement
        print("\nExample 3: Workflow Refinement")
        print("-" * 80)
        refined = await generator.refine(workflow2, "Add error handling and make it parallel")
        print(f"Refined workflow nodes: {len(refined['nodes'])} (was {len(workflow2['nodes'])})")

        # Validation
        print("\nValidation:")
        print("-" * 80)
        valid, errors = generator.validate(refined)
        print(f"Valid: {valid}")
        if errors:
            for error in errors:
                print(f"  - {error}")

    asyncio.run(main())
