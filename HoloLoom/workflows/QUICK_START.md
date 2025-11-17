# HoloLoom Workflows: Quick Start Guide

**Updated: November 2025**

Get started with HoloLoom's enhanced workflow system in 5 minutes.

---

## Installation

```bash
# No additional installation needed
# All features included in HoloLoom package
```

---

## 1. Using Templates (Fastest Way)

### Get a Template
```python
from HoloLoom.workflows import WorkflowTemplates

templates = WorkflowTemplates()

# List all templates
for t in templates.list_all():
    print(f"{t.name} ({t.difficulty})")

# Get a specific template
workflow = templates.get('email_analysis')
```

### Available Templates
| Name | Category | Difficulty | Use Case |
|------|----------|-----------|----------|
| RAG Research Pipeline | RAG | Intermediate | Multi-source research |
| Simple Q&A | RAG | Beginner | Quick questions |
| Code Review | CODE | Intermediate | Code analysis |
| Bug Triage | CODE | Intermediate | Bug classification |
| Data Pipeline | DATA | Beginner | ETL workflows |
| SQL Pipeline | DATA | Intermediate | Database queries |
| Translation | DATA | Intermediate | Language translation |
| Email Analysis | ANALYSIS | Intermediate | Email processing |
| Sentiment Analysis | ANALYSIS | Beginner | Emotion detection |
| Document QA | RAG | Intermediate | Document search |
| Meeting Summary | ANALYSIS | Advanced | Meeting notes |
| Product Recommendation | ANALYSIS | Advanced | Personalization |
| Content Moderation | ANALYSIS | Intermediate | Safety filtering |
| Multi-Agent Consensus | ANALYSIS | Advanced | Ensemble voting |

### Customize a Template
```python
# Clone and customize
custom = templates.clone_and_customize(
    'email_analysis',
    'My Email Classifier',
    {
        'node_2': {'model': 'distilbert-base'}  # Override config
    }
)
```

---

## 2. Generating Workflows from Description

### Simple Generation
```python
from HoloLoom.workflows.ai_generator import AIWorkflowGenerator

generator = AIWorkflowGenerator()

# Describe what you want
workflow = await generator.generate(
    "Analyze Python code for security issues, suggest fixes, and save results"
)

print(f"Generated: {workflow['name']}")
print(f"Nodes: {len(workflow['nodes'])}")
print(f"Connections: {len(workflow['connections'])}")
```

### Check Confidence
```python
intent = generator.detect_intent("Analyze code")

print(f"Primary Goal: {intent.primary_goal}")
print(f"Confidence: {intent.confidence:.0%}")  # 0-100%
print(f"Explanation: {intent.explanation}")

if intent.confidence >= 0.5:
    # Good confidence, proceed
    workflow = await generator.generate(description)
```

### Refine Generated Workflows
```python
# Start with basic workflow
workflow = await generator.generate("Analyze data")

# Enhance it
refined = await generator.refine(
    workflow,
    "Add error handling and make it run in parallel"
)
```

---

## 3. Validating Workflows

### Quick Validation
```python
from HoloLoom.workflows.test_framework import validate_workflow

is_valid, errors = validate_workflow(my_workflow)

if not is_valid:
    for error in errors:
        print(f"ERROR: {error}")
```

### Detailed Validation
```python
from HoloLoom.workflows.test_framework import WorkflowTester

tester = WorkflowTester()
result = tester.validate_workflow(my_workflow)

print(f"Valid: {result.valid}")
print(f"Warnings: {result.warnings}")
print(f"Metrics: {result.metrics}")
```

---

## 4. Testing Workflows

### Dry-Run (Simulate)
```python
# Test without executing
trace = tester.simulate_execution(
    my_workflow,
    inputs={'query': 'test query'}
)

print(f"Would execute {len(trace['steps'])} steps")
```

### Test Templates
```python
# Test a single template
result = tester.test_template('rag_research')
print(f"Template valid: {result.passed}")

# Test all templates
all_results = tester.test_all_templates()
passed = sum(1 for r in all_results if r.passed)
print(f"Passed: {passed}/{len(all_results)}")
```

---

## 5. Debugging Workflows

### Step-Through Execution
```python
from HoloLoom.workflows.debug_tools import WorkflowDebugger

debugger = WorkflowDebugger(my_workflow)

# Run step-by-step
trace = await debugger.step_through()

# Print what happened
print(debugger.print_trace())
```

### Set Breakpoints
```python
# Pause at specific node
debugger.set_breakpoint('node_1')

# Run until breakpoint
trace = await debugger.run_to_breakpoint()

# Inspect variables at breakpoint
variables = debugger.inspect_variables()
print(f"Inputs: {variables['inputs']}")
print(f"Outputs: {variables['outputs']}")
```

### Conditional Breakpoints
```python
# Only break if condition met
debugger.set_breakpoint(
    'node_2',
    condition=lambda x: x['confidence'] < 0.5
)
```

---

## 6. Common Tasks

### Build a Code Review Workflow
```python
templates = WorkflowTemplates()
workflow = templates.get('code_review')

# Ready to use!
```

### Build an Email Classifier
```python
# Option 1: Use template
workflow = templates.get('email_analysis')

# Option 2: Generate from description
workflow = await generator.generate(
    "Analyze emails for spam, sentiment, and categories"
)
```

### Build a Research Pipeline
```python
workflow = templates.get('rag_research')

# Customize for your domain
custom = templates.clone_and_customize(
    'rag_research',
    'Finance Research',
    {
        'node_1': {'max_subqueries': 10}
    }
)
```

### Compare Two Workflows
```python
diff = tester.compare_workflows(workflow1, workflow2)

print(f"Similarity: {diff['similarity_score']:.0%}")
print(f"Added nodes: {diff['nodes_added']}")
print(f"Removed nodes: {diff['nodes_removed']}")
```

---

## 7. Complete Example

```python
import asyncio
from HoloLoom.workflows import WorkflowTemplates, AIWorkflowGenerator
from HoloLoom.workflows.test_framework import (
    validate_workflow, WorkflowTester
)
from HoloLoom.workflows.debug_tools import WorkflowDebugger

async def main():
    # Option A: Use a template
    templates = WorkflowTemplates()
    workflow = templates.get('email_analysis')

    # Option B: Generate from description
    # generator = AIWorkflowGenerator()
    # workflow = await generator.generate(
    #     "Parse emails, extract sentiment, and classify"
    # )

    # Validate
    tester = WorkflowTester()
    result = tester.validate_workflow(workflow)
    if not result.valid:
        print(f"Validation failed: {result.errors}")
        return

    # Simulate
    trace = tester.simulate_execution(workflow, inputs={'email': 'test'})
    print(f"Workflow would execute {len(trace['steps'])} steps")

    # Debug (optional)
    debugger = WorkflowDebugger(workflow)
    debugger.set_breakpoint('node_1')
    exec_trace = await debugger.run_to_breakpoint()

    print("Workflow ready to execute!")

asyncio.run(main())
```

---

## 8. Common Patterns

### Safety-Gated Workflow
```python
workflow = {
    'nodes': [
        {'id': 'process', 'agentType': 'llm_prompt'},
        {'id': 'validate', 'agentType': 'safety'},  # Check before proceeding
        {'id': 'execute', 'agentType': 'store'}
    ],
    'connections': [
        {'from': 'process', 'to': 'validate'},
        {'from': 'validate', 'to': 'execute'}
    ]
}
```

### Parallel Processing
```python
workflow = {
    'nodes': [
        {'id': 'start', 'agentType': 'parallel'},
        {'id': 'analyze_1', 'agentType': 'code_analyzer'},
        {'id': 'analyze_2', 'agentType': 'code_analyzer'},
        {'id': 'merge', 'agentType': 'synthesizer'}
    ],
    'connections': [
        {'from': 'start', 'to': 'analyze_1'},
        {'from': 'start', 'to': 'analyze_2'},
        {'from': 'analyze_1', 'to': 'merge'},
        {'from': 'analyze_2', 'to': 'merge'}
    ]
}
```

### Error Handling
```python
workflow = {
    'nodes': [
        {'id': 'query', 'agentType': 'rag_query'},
        {'id': 'check', 'agentType': 'conditional'},
        {'id': 'high_conf', 'agentType': 'response'},
        {'id': 'low_conf', 'agentType': 'refiner'}
    ],
    'connections': [
        {'from': 'query', 'to': 'check'},
        {'from': 'check', 'to': 'high_conf'},
        {'from': 'check', 'to': 'low_conf'}
    ]
}
```

---

## 9. Troubleshooting

**Q: Workflow validation fails**
```python
is_valid, errors = validate_workflow(workflow)
print(errors)  # See what's wrong
```

**Q: Low confidence in generated workflow**
```python
intent = generator.detect_intent(description)
print(f"Confidence: {intent.confidence:.0%}")
# If < 0.5, try: 1) Better description, 2) Use template, 3) Manual creation
```

**Q: Want to step through manually**
```python
debugger = WorkflowDebugger(workflow)
debugger.set_breakpoint('node_important')
trace = await debugger.run_to_breakpoint()
```

**Q: How to compare workflows**
```python
diff = tester.compare_workflows(old_workflow, new_workflow)
print(f"Changed: {diff['nodes_modified']}")
```

---

## 10. Next Steps

1. **Explore Templates** - Browse all 14 available templates
2. **Try Generation** - Generate workflow from description
3. **Validate** - Check workflow before execution
4. **Debug** - Use tools to step through and inspect
5. **Deploy** - Execute validated workflow

---

## Resources

- **Full Guide:** BEST_PRACTICES.md
- **Enhancement Report:** ENHANCEMENT_REPORT.md
- **Demo:** demo_enhanced_workflows.py
- **API Reference:** README.md

---

**Status:** ✅ Production Ready
**Version:** 2.0 (November 2025)
