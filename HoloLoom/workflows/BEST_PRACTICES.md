# HoloLoom Workflows: Best Practices Guide

**Updated: November 2025**

This guide provides best practices for building, testing, and debugging workflows in HoloLoom.

---

## 1. Workflow Design Principles

### Start Simple

Begin with the simplest workflow that solves your problem:

```python
# GOOD: Simple 2-node workflow
{
  'name': 'Simple Query',
  'nodes': [
    {'id': 'node_1', 'agentType': 'hololoom'},
    {'id': 'node_2', 'agentType': 'response'}
  ],
  'connections': [
    {'from': 'node_1', 'to': 'node_2'}
  ]
}
```

### Single Responsibility

Each node should have a clear, single purpose.

### Use Templates

Start with a template and customize it:

```python
from HoloLoom.workflows import WorkflowTemplates

templates = WorkflowTemplates()
workflow = templates.get('code_review')
```

---

## 2. Workflow Validation

Always validate workflows before execution:

```python
from HoloLoom.workflows.test_framework import validate_workflow

is_valid, errors = validate_workflow(my_workflow)
if not is_valid:
    for error in errors:
        print(f"ERROR: {error}")
```

---

## 3. Error Handling & Safety

Always include safety nodes for high-risk workflows:

```python
{
    'nodes': [
        {'id': 'generate_sql', 'agentType': 'llm_prompt'},
        {'id': 'validate_sql', 'agentType': 'safety', 'config': {
            'risk_threshold': 'HIGH',
            'enable_human_in_loop': True
        }},
        {'id': 'execute', 'agentType': 'store'}
    ]
}
```

---

## 4. Debugging Workflows

Use the debugging tools:

```python
from HoloLoom.workflows.debug_tools import WorkflowDebugger

debugger = WorkflowDebugger(my_workflow)
debugger.set_breakpoint('node_query')
trace = await debugger.step_through()
```

---

## 5. Testing Workflows

### Unit Test Templates

```python
from HoloLoom.workflows.test_framework import WorkflowTester

def test_code_review_template():
    tester = WorkflowTester()
    result = tester.test_template('code_review')
    assert result.passed
```

---

## 6. Performance Optimization

- Reduce node count where possible
- Use parallelization for independent operations
- Cache results in deterministic nodes

---

## Summary

1. Start simple, add complexity incrementally
2. Always validate before execution
3. Use safety nodes for risky operations
4. Test with templates first
5. Debug with provided tools
6. Monitor performance
7. Version your workflows

---

**Created: November 2025**
**Status: Production Ready**
