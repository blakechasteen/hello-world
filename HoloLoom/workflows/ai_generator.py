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
    confidence: float = 0.0  # 0.0-1.0 confidence in detected intent
    explanation: str = ""  # Why this intent was detected


class AIWorkflowGenerator:
    """
    AI-powered workflow generator that creates workflows from natural language.

    Uses pattern matching, templates, and heuristics to generate workflows.
    Can be enhanced with LLM integration for more sophisticated generation.
    """

    def __init__(self, registry: Optional[AgentRegistry] = None):
        self.registry = registry or get_registry()
        self.templates = WorkflowTemplates() if WorkflowTemplates else None

        # Intent keywords (expanded with more patterns)
        self.intent_patterns = {
            'analyze': ['analyze', 'analysis', 'review', 'check', 'inspect', 'examine', 'evaluate', 'assess', 'audit'],
            'transform': ['transform', 'convert', 'process', 'clean', 'normalize', 'format', 'restructure', 'parse'],
            'query': ['query', 'search', 'find', 'retrieve', 'lookup', 'ask', 'research', 'investigate'],
            'generate': ['generate', 'create', 'produce', 'synthesize', 'make', 'compose', 'build'],
            'decide': ['decide', 'choose', 'select', 'vote', 'consensus', 'recommend', 'rank'],
            'validate': ['validate', 'verify', 'check', 'confirm', 'test', 'quality', 'ensure'],
            'store': ['store', 'save', 'persist', 'keep', 'cache', 'database', 'archive'],
            'parallel': ['parallel', 'concurrent', 'simultaneous', 'together', 'async', 'distributed'],
            'sequential': ['sequential', 'series', 'chain', 'pipeline', 'step-by-step', 'ordered'],
            'summarize': ['summarize', 'summary', 'extract', 'abstract', 'condense', 'aggregate'],
            'classify': ['classify', 'categorize', 'label', 'tag', 'bucket', 'sort', 'organize'],
            'detect': ['detect', 'identify', 'find', 'discover', 'locate', 'spot', 'recognize']
        }

        # Enhanced agent type mappings
        self.agent_mappings = {
            'code': ['code_analyzer', 'llm_prompt'],
            'security': ['code_analyzer', 'safety', 'llm_prompt'],
            'data': ['data_transformer', 'store', 'embedder'],
            'llm': ['llm_prompt', 'structured_llm'],
            'rag': ['rag_query', 'multiquery', 'embedder'],
            'research': ['multiquery', 'rag_query', 'synthesizer', 'refiner'],
            'sentiment': ['sentiment_analyzer', 'data_transformer'],
            'web': ['web_scraper', 'data_transformer'],
            'email': ['data_transformer', 'sentiment_analyzer', 'llm_prompt'],
            'sql': ['llm_prompt', 'safety', 'data_transformer'],
            'translation': ['llm_prompt', 'data_transformer', 'conditional'],
            'bug': ['data_transformer', 'llm_prompt', 'synthesizer'],
            'audio': ['data_transformer', 'llm_prompt', 'synthesizer'],
            'recommendation': ['embedder', 'rag_query', 'llm_prompt'],
            'moderation': ['data_transformer', 'llm_prompt', 'safety', 'conditional']
        }

        # Domain keywords
        self.domain_keywords = {
            'code': ['code', 'python', 'javascript', 'java', 'program', 'bug', 'security', 'vulnerab'],
            'data': ['data', 'csv', 'json', 'database', 'sql', 'table', 'transform', 'clean'],
            'email': ['email', 'mail', 'message', 'inbox', 'spam', 'classification'],
            'document': ['document', 'pdf', 'text', 'page', 'paragraph', 'chunk', 'qa'],
            'sentiment': ['sentiment', 'emotion', 'review', 'opinion', 'feedback', 'social'],
            'translation': ['translate', 'language', 'multilingual', 'international'],
            'audio': ['audio', 'meeting', 'transcribe', 'voice', 'recording'],
            'image': ['image', 'photo', 'visual', 'caption', 'ocr', 'detect'],
            'recommendation': ['recommend', 'suggestion', 'product', 'personali', 'collaborat'],
            'moderation': ['moderate', 'filter', 'block', 'offensive', 'spam', 'abuse']
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
        Detect user intent from natural language description with confidence scoring.

        Uses keyword matching and patterns to identify:
        - Primary goal (analyze, transform, query, etc.)
        - Required agents
        - Constraints (parallel, error handling, etc.)
        - Input/output types
        - Confidence score (0.0-1.0)
        """
        desc_lower = description.lower()
        confidence_scores = {}
        matched_keywords = []

        # Score each intent pattern
        for goal, keywords in self.intent_patterns.items():
            matches = sum(1 for kw in keywords if kw in desc_lower)
            if matches > 0:
                confidence_scores[goal] = matches / len(keywords)
                matched_keywords.extend([kw for kw in keywords if kw in desc_lower])

        # Find primary goal with highest confidence
        primary_goal = max(confidence_scores, key=confidence_scores.get) if confidence_scores else 'query'
        primary_confidence = confidence_scores.get(primary_goal, 0.0)

        # Find secondary goals
        secondary_goals = [g for g, conf in confidence_scores.items()
                          if g != primary_goal and conf >= 0.4]

        # Find needed agents by domain
        agents_needed = []
        matched_domains = []
        for domain, keywords in self.domain_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                if domain in self.agent_mappings:
                    agents_needed.extend(self.agent_mappings.get(domain, []))
                    matched_domains.append(domain)

        # Fallback: use primary goal to select agents
        if not agents_needed and primary_goal in self.agent_mappings:
            agents_needed.extend(self.agent_mappings[primary_goal])

        # Remove duplicates and invalid agents
        agents_needed = list(set(agents_needed))
        if self.registry:
            valid_agents = set(self.registry.agents.keys())
            agents_needed = [a for a in agents_needed if a in valid_agents]

        # Detect constraints
        constraints = []
        if any(kw in desc_lower for kw in self.intent_patterns['parallel']):
            constraints.append('parallel')
        if any(word in desc_lower for word in ['error', 'safe', 'protection', 'secure']):
            constraints.append('error_handling')
        if any(word in desc_lower for word in ['refine', 'improve', 'quality', 'enhance']):
            constraints.append('refinement')
        if any(word in desc_lower for word in ['batch', 'multiple', 'many']):
            constraints.append('batching')

        # Detect input/output types
        input_type = None
        if any(domain in matched_domains for domain in ['code', 'program']):
            input_type = 'code'
        elif any(domain in matched_domains for domain in ['data', 'database']):
            input_type = 'data'
        elif any(domain in matched_domains for domain in ['audio', 'meeting']):
            input_type = 'audio'
        elif any(domain in matched_domains for domain in ['image', 'photo']):
            input_type = 'image'
        elif any(domain in matched_domains for domain in ['document', 'text']):
            input_type = 'document'
        else:
            input_type = 'text'

        output_format = None
        if 'json' in desc_lower:
            output_format = 'json'
        elif 'markdown' in desc_lower:
            output_format = 'markdown'
        elif 'html' in desc_lower:
            output_format = 'html'
        else:
            output_format = 'text'

        # Complexity estimation
        complexity = 'simple'
        complexity_score = 0.0

        if len(agents_needed) > 4:
            complexity = 'complex'
            complexity_score = 1.0
        elif len(agents_needed) > 2:
            complexity = 'medium'
            complexity_score = 0.6
        else:
            complexity = 'simple'
            complexity_score = 0.3

        if len(secondary_goals) > 2:
            complexity = 'complex'
            complexity_score = max(complexity_score, 0.9)
        elif len(secondary_goals) > 0:
            complexity_score = max(complexity_score, 0.5)

        # Overall confidence: primary goal confidence + agent matching
        domain_confidence = len(matched_domains) / len(self.domain_keywords) if self.domain_keywords else 0.5
        agent_confidence = min(len(agents_needed) / 3.0, 1.0)  # Good if 3+ agents found
        overall_confidence = (primary_confidence * 0.4 +
                            domain_confidence * 0.3 +
                            agent_confidence * 0.3)

        explanation = f"Detected {primary_goal} intent ({primary_confidence:.0%}) with {len(agents_needed)} agents needed. Matched domains: {', '.join(matched_domains) if matched_domains else 'general'}."

        return WorkflowIntent(
            primary_goal=primary_goal,
            secondary_goals=secondary_goals,
            agents_needed=agents_needed,
            constraints=constraints,
            input_type=input_type,
            output_format=output_format,
            complexity=complexity,
            confidence=overall_confidence,
            explanation=explanation
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
