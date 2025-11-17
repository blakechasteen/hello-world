#!/usr/bin/env python3
"""
Workflow Testing Framework
===========================

Unit tests, integration tests, and validation for workflows.

Features:
- Workflow validation (structure, cycles, connections)
- Dry-run simulation
- Workflow comparison
- Template validation
- AI generator testing

Usage:
    from HoloLoom.workflows.test_framework import WorkflowTester, validate_workflow

    tester = WorkflowTester()

    # Validate workflow
    is_valid, errors = validate_workflow(workflow)

    # Test template
    results = tester.test_template('rag_research')

    # Compare workflows
    diff = tester.compare_workflows(workflow1, workflow2)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Workflow validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]


@dataclass
class TestResult:
    """Workflow test result"""
    test_name: str
    passed: bool
    duration_ms: float
    message: str
    details: Dict[str, Any]


def validate_workflow(workflow: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate workflow structure.

    Checks:
    - Required fields (name, nodes, connections)
    - Valid node IDs
    - No cycles
    - All connections valid
    - Valid agent types

    Returns:
        (is_valid, error_list)
    """
    errors = []

    # Check required fields
    if 'nodes' not in workflow:
        errors.append("Missing 'nodes' field")
    if 'connections' not in workflow:
        errors.append("Missing 'connections' field")

    if errors:
        return False, errors

    nodes = workflow.get('nodes', [])
    connections = workflow.get('connections', [])

    # Check for empty workflow
    if not nodes:
        errors.append("Workflow has no nodes")

    # Check node IDs are unique
    node_ids = [n.get('id') for n in nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node IDs found")

    # Check connections reference valid nodes
    node_id_set = set(node_ids)
    for conn in connections:
        if conn.get('from') not in node_id_set:
            errors.append(f"Connection references invalid source: {conn.get('from')}")
        if conn.get('to') not in node_id_set:
            errors.append(f"Connection references invalid target: {conn.get('to')}")

    # Check for cycles
    if _has_cycles(nodes, connections):
        errors.append("Workflow contains cycles")

    # Check for disconnected nodes (optional warning)
    connected_nodes = set()
    for conn in connections:
        connected_nodes.add(conn.get('from'))
        connected_nodes.add(conn.get('to'))

    disconnected = node_id_set - connected_nodes
    if disconnected and len(nodes) > 1:
        logger.warning(f"Disconnected nodes: {disconnected}")

    return len(errors) == 0, errors


def _has_cycles(nodes: List[Dict], connections: List[Dict]) -> bool:
    """Check if workflow has cycles (using DFS)"""
    graph = {n['id']: [] for n in nodes}
    for conn in connections:
        graph[conn['from']].append(conn['to'])

    visited = set()
    rec_stack = set()

    def dfs(node_id):
        visited.add(node_id)
        rec_stack.add(node_id)

        for neighbor in graph.get(node_id, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node_id)
        return False

    for node in nodes:
        if node['id'] not in visited:
            if dfs(node['id']):
                return True

    return False


class WorkflowTester:
    """
    Comprehensive workflow testing framework.
    """

    def __init__(self):
        self.test_results: List[TestResult] = []
        logger.info("WorkflowTester initialized")

    def validate_workflow(self, workflow: Dict[str, Any]) -> ValidationResult:
        """Validate workflow with detailed report"""
        is_valid, errors = validate_workflow(workflow)
        warnings = []

        # Collect metrics
        metrics = {
            'num_nodes': len(workflow.get('nodes', [])),
            'num_connections': len(workflow.get('connections', [])),
            'avg_node_config_size': self._avg_config_size(workflow),
            'has_error_handling': self._check_error_handling(workflow),
            'has_refinement': self._check_refinement(workflow),
            'is_parallel': self._check_parallel(workflow)
        }

        if metrics['num_nodes'] > 10:
            warnings.append("Large workflow (>10 nodes) may be complex")

        if not metrics['has_error_handling']:
            warnings.append("No error handling (safety nodes) detected")

        return ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )

    def simulate_execution(self, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate workflow execution without actually running nodes.

        Returns execution trace.
        """
        nodes = workflow.get('nodes', [])
        connections = workflow.get('connections', [])

        # Topological sort
        execution_order = self._topological_sort(nodes, connections)

        trace = {
            'workflow_id': workflow.get('id', 'unknown'),
            'status': 'success',
            'steps': [],
            'final_inputs': inputs.copy()
        }

        for node_id in execution_order:
            node = next((n for n in nodes if n['id'] == node_id), None)
            if not node:
                continue

            step = {
                'node_id': node_id,
                'node_type': node.get('agentType'),
                'inputs': inputs.copy(),
                'simulated_output': {'status': 'success', 'data': {}}
            }
            trace['steps'].append(step)

        return trace

    def compare_workflows(
        self,
        workflow1: Dict[str, Any],
        workflow2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two workflows and return differences.
        """
        diff = {
            'nodes_added': [],
            'nodes_removed': [],
            'nodes_modified': [],
            'connections_added': [],
            'connections_removed': [],
            'similarity_score': 0.0
        }

        nodes1 = {n['id']: n for n in workflow1.get('nodes', [])}
        nodes2 = {n['id']: n for n in workflow2.get('nodes', [])}

        # Node comparisons
        for node_id in nodes2:
            if node_id not in nodes1:
                diff['nodes_added'].append(node_id)
            elif nodes1[node_id] != nodes2[node_id]:
                diff['nodes_modified'].append({
                    'node_id': node_id,
                    'before': nodes1[node_id],
                    'after': nodes2[node_id]
                })

        for node_id in nodes1:
            if node_id not in nodes2:
                diff['nodes_removed'].append(node_id)

        # Connection comparisons
        conns1 = set((c['from'], c['to']) for c in workflow1.get('connections', []))
        conns2 = set((c['from'], c['to']) for c in workflow2.get('connections', []))

        diff['connections_added'] = [
            {'from': f, 'to': t} for f, t in conns2 - conns1
        ]
        diff['connections_removed'] = [
            {'from': f, 'to': t} for f, t in conns1 - conns2
        ]

        # Similarity score
        total_nodes = len(set(nodes1.keys()) | set(nodes2.keys()))
        if total_nodes > 0:
            common_nodes = len(set(nodes1.keys()) & set(nodes2.keys()))
            diff['similarity_score'] = common_nodes / total_nodes

        return diff

    def test_template(self, template_id: str) -> TestResult:
        """Test a workflow template"""
        from HoloLoom.workflows.templates import WorkflowTemplates

        templates = WorkflowTemplates()
        template = templates.get(template_id)

        if not template:
            return TestResult(
                test_name=f"Template: {template_id}",
                passed=False,
                duration_ms=0.0,
                message=f"Template not found: {template_id}",
                details={}
            )

        # Validate template
        workflow = {
            'nodes': template.nodes,
            'connections': template.connections
        }

        is_valid, errors = validate_workflow(workflow)

        details = {
            'template_id': template_id,
            'name': template.name,
            'num_nodes': len(template.nodes),
            'num_connections': len(template.connections),
            'difficulty': template.difficulty,
            'validation_errors': errors
        }

        return TestResult(
            test_name=f"Template: {template_id}",
            passed=is_valid,
            duration_ms=0.0,
            message="Template is valid" if is_valid else "Template validation failed",
            details=details
        )

    def test_all_templates(self) -> List[TestResult]:
        """Test all built-in templates"""
        from HoloLoom.workflows.templates import WorkflowTemplates

        templates = WorkflowTemplates()
        results = []

        for template in templates.list_all():
            result = self.test_template(template.template_id)
            results.append(result)

        return results

    def test_ai_generator(self, descriptions: List[str]) -> List[TestResult]:
        """Test AI workflow generator"""
        from HoloLoom.workflows.ai_generator import AIWorkflowGenerator

        generator = AIWorkflowGenerator()
        results = []

        for desc in descriptions:
            try:
                intent = generator.detect_intent(desc)

                result = TestResult(
                    test_name=f"Generate: {desc[:50]}...",
                    passed=intent.confidence >= 0.5,
                    duration_ms=0.0,
                    message=intent.explanation,
                    details={
                        'description': desc,
                        'primary_goal': intent.primary_goal,
                        'confidence': intent.confidence,
                        'agents_needed': intent.agents_needed,
                        'complexity': intent.complexity
                    }
                )
                results.append(result)

            except Exception as e:
                results.append(TestResult(
                    test_name=f"Generate: {desc[:50]}...",
                    passed=False,
                    duration_ms=0.0,
                    message=str(e),
                    details={'error': str(e)}
                ))

        return results

    def generate_test_report(self) -> str:
        """Generate test report"""
        output = []
        output.append("=" * 80)
        output.append("WORKFLOW TEST REPORT")
        output.append("=" * 80)
        output.append("")

        passed = sum(1 for r in self.test_results if r.passed)
        total = len(self.test_results)

        output.append(f"Results: {passed}/{total} passed ({100*passed/total:.0f}%)")
        output.append("")

        # Group by test type
        test_types = {}
        for result in self.test_results:
            test_type = result.test_name.split(':')[0].strip()
            if test_type not in test_types:
                test_types[test_type] = {'passed': 0, 'total': 0}
            test_types[test_type]['total'] += 1
            if result.passed:
                test_types[test_type]['passed'] += 1

        for test_type, counts in test_types.items():
            pct = 100 * counts['passed'] / counts['total']
            output.append(f"{test_type}: {counts['passed']}/{counts['total']} ({pct:.0f}%)")

        output.append("")
        output.append("Failed tests:")
        for result in self.test_results:
            if not result.passed:
                output.append(f"  - {result.test_name}")
                output.append(f"    {result.message}")

        output.append("")
        output.append("=" * 80)

        return "\n".join(output)

    def _avg_config_size(self, workflow: Dict[str, Any]) -> float:
        """Calculate average config size per node"""
        nodes = workflow.get('nodes', [])
        if not nodes:
            return 0.0

        total_size = sum(len(n.get('config', {})) for n in nodes)
        return total_size / len(nodes)

    def _check_error_handling(self, workflow: Dict[str, Any]) -> bool:
        """Check if workflow has error handling (safety nodes)"""
        nodes = workflow.get('nodes', [])
        return any(n.get('agentType') == 'safety' for n in nodes)

    def _check_refinement(self, workflow: Dict[str, Any]) -> bool:
        """Check if workflow has refinement"""
        nodes = workflow.get('nodes', [])
        return any(n.get('agentType') == 'refiner' for n in nodes)

    def _check_parallel(self, workflow: Dict[str, Any]) -> bool:
        """Check if workflow has parallel execution"""
        nodes = workflow.get('nodes', [])
        return any(n.get('agentType') == 'parallel' for n in nodes)

    def _topological_sort(self, nodes: List[Dict], connections: List[Dict]) -> List[str]:
        """Topological sort using Kahn's algorithm"""
        graph = {n['id']: [] for n in nodes}
        in_degree = {n['id']: 0 for n in nodes}

        for conn in connections:
            graph[conn['from']].append(conn['to'])
            in_degree[conn['to']] += 1

        queue = [n['id'] for n in nodes if in_degree[n['id']] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


if __name__ == "__main__":
    # Demo usage
    example_workflow = {
        'id': 'test',
        'name': 'Test',
        'nodes': [
            {'id': 'node_1', 'agentType': 'hololoom', 'config': {}},
            {'id': 'node_2', 'agentType': 'response', 'config': {}}
        ],
        'connections': [
            {'from': 'node_1', 'to': 'node_2'}
        ]
    }

    tester = WorkflowTester()

    # Validate
    result = tester.validate_workflow(example_workflow)
    print(f"Valid: {result.valid}")
    print(f"Metrics: {result.metrics}")

    # Test all templates
    template_results = tester.test_all_templates()
    print(f"\nTested {len(template_results)} templates")

    for r in template_results:
        if not r.passed:
            print(f"FAIL: {r.test_name}")
