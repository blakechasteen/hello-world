# HoloLoom Skill Agent System

**Status**: ✅ Complete (Phase 3 - November 2025)
**Location**: `hololoom/agentic/`
**Skills**: 13 professional templates
**Integration**: Promptly → HoloLoom Recursive Reasoning

The Skill Agent System brings Promptly's 13 professional skill templates into HoloLoom's recursive weaving architecture, enabling AI-powered code review, debugging, testing, and more with quality-driven refinement.

## Overview

### What is a Skill?

A **skill** is a specialized AI agent template for professional software engineering tasks. Each skill combines:

1. **Expert system prompt** - Domain-specific knowledge and best practices
2. **Parameterized user prompt** - Template for user input
3. **Recursive reasoning strategy** - REFINE, CRITIQUE, DECOMPOSE, etc.
4. **Quality checks** - Validation criteria for output
5. **Examples** - Reference inputs/outputs

### Architecture

```
YAML Skill Template
    ↓
SkillRegistry (loads all templates)
    ↓
SkillExecutor
    ├─ Validates parameters
    ├─ Builds prompt from template
    └─ Executes via RecursiveWeavingOrchestrator
        ├─ Recursive reasoning strategies
        ├─ Quality-driven refinement
        ├─ Analytics tracking
        └─ Complete provenance (ReasoningJournal)
            ↓
SkillExecutionResult (output + metadata)
```

## The 13 Professional Skills

### Development (7 skills)

1. **code-reviewer** - Review code for best practices and quality
   - Strategy: CRITIQUE (self-critique loop)
   - Focus: SOLID principles, code smells, security

2. **bug-detective** - Systematically debug and find root causes
   - Strategy: DECOMPOSE (break down → analyze → solve)
   - Focus: 5 Whys, binary search debugging

3. **test-generator** - Generate comprehensive test cases
   - Strategy: EXPLORE (explore different test scenarios)
   - Focus: Happy path, edge cases, error handling

4. **documentation-writer** - Generate clear documentation
   - Strategy: REFINE (iterative improvement)
   - Focus: API docs, README files, tutorials

5. **code-explainer** - Explain complex code simply
   - Strategy: REFINE
   - Focus: Educational explanations with analogies

6. **naming-consultant** - Suggest better variable/function names
   - Strategy: CRITIQUE
   - Focus: Naming conventions, clarity

7. **refactoring-expert** - Refactor for maintainability
   - Strategy: CRITIQUE
   - Focus: SOLID, design patterns, code smells

### Architecture (2 skills)

8. **architecture-advisor** - System architecture guidance
   - Strategy: HOFSTADTER (meta-reasoning)
   - Focus: Microservices, scalability, CAP theorem

9. **migration-planner** - Plan technology migrations
   - Strategy: DECOMPOSE
   - Focus: Risk mitigation, step-by-step planning

### Database (1 skill)

10. **sql-optimizer** - Optimize SQL queries
    - Strategy: REFINE
    - Focus: Indexes, N+1 queries, CTEs

### Security (1 skill)

11. **security-auditor** - Audit for vulnerabilities
    - Strategy: VERIFY (verify security claims)
    - Focus: OWASP Top 10, injection, auth

### Optimization (1 skill)

12. **performance-profiler** - Analyze and optimize performance
    - Strategy: DECOMPOSE
    - Focus: Big O complexity, bottlenecks, caching

### API Design (1 skill)

13. **api-designer** - Design RESTful APIs
    - Strategy: REFINE
    - Focus: REST principles, OpenAPI, security

## Quick Start

### Basic Usage

```python
from hololoom.agentic.skill_agents import execute_skill
from hololoom.config import Config

# Execute a skill
result = await execute_skill(
    skill_name="code-reviewer",
    parameters={
        "code": code,
        "language": "python",
        "filename": "utils.py"
    },
    config=Config.fast()
)

print(result.output)       # Review feedback
print(result.confidence)   # 0.0-1.0
print(result.iterations)   # Number of refinement passes
```

### List Available Skills

```python
from hololoom.agentic.skill_agents import list_available_skills

skills = await list_available_skills()

# Output:
# {
#   "development": ["code-reviewer", "bug-detective", ...],
#   "architecture": ["architecture-advisor", "migration-planner"],
#   ...
# }
```

### Override Reasoning Strategy

```python
# Force DECOMPOSE strategy
result = await execute_skill(
    "bug-detective",
    parameters={...},
    override_strategy="decompose",
    override_iterations=5
)
```

## Detailed Examples

### 1. Code Review

```python
code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

result = await execute_skill(
    "code-reviewer",
    parameters={
        "code": code,
        "language": "python",
        "focus_areas": "readability, performance"
    }
)

# Result includes:
# - Overall rating (1-10)
# - Critical issues
# - Improvements
# - Highlights
# - Refactoring suggestions
```

### 2. Bug Detective

```python
buggy_code = """
def get_user_name(user):
    return user.profile.name
"""

result = await execute_skill(
    "bug-detective",
    parameters={
        "code": buggy_code,
        "language": "javascript",
        "bug_description": "Crashes when user has no profile",
        "error_message": "TypeError: Cannot read property 'name' of undefined",
        "expected_behavior": "Return name or 'Unknown'",
        "actual_behavior": "Crashes with TypeError"
    }
)

# Result includes:
# - Root cause analysis
# - Hypothesis chain (5 Whys)
# - Fixed code
# - Test case
# - Edge cases to watch
```

### 3. Test Generation

```python
code = """
def calculate_discount(price, discount_percent):
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
"""

result = await execute_skill(
    "test-generator",
    parameters={
        "code": code,
        "language": "python",
        "framework": "pytest",
        "happy_path": True,
        "edge_cases": True,
        "error_handling": True
    }
)

# Result includes:
# - Complete test file
# - Test count
# - Coverage report
# - Additional test suggestions
```

### 4. API Design

```python
result = await execute_skill(
    "api-designer",
    parameters={
        "requirements": """
        Design a REST API for a task management system.
        Users can create tasks, assign to others, and track progress.
        """,
        "version_strategy": "uri",
        "auth_type": "jwt"
    }
)

# Result includes:
# - Resource models
# - Endpoint specifications
# - OpenAPI schema
# - Security recommendations
# - Pagination strategy
```

### 5. SQL Optimization

```python
slow_query = """
SELECT *
FROM orders o, customers c
WHERE o.customer_id = c.id
AND c.country = 'USA'
"""

result = await execute_skill(
    "sql-optimizer",
    parameters={
        "query": slow_query,
        "database_type": "postgresql",
        "schema": "orders: id, customer_id, total | customers: id, name, country"
    }
)

# Result includes:
# - Optimized query
# - Explanation of changes
# - Index recommendations
# - Estimated speedup
```

## Recursive Reasoning Integration

All skills leverage HoloLoom's recursive reasoning strategies:

| Strategy | When to Use | Example Skills |
|----------|-------------|----------------|
| **REFINE** | Iterative improvement | documentation-writer, sql-optimizer |
| **CRITIQUE** | Self-critique loop | code-reviewer, refactoring-expert |
| **DECOMPOSE** | Break down complex problem | bug-detective, migration-planner |
| **EXPLORE** | Multiple approaches | test-generator |
| **VERIFY** | Verify claims/security | security-auditor |
| **HOFSTADTER** | Meta-reasoning | architecture-advisor |

### Quality-Driven Refinement

Skills automatically refine output when confidence < threshold:

```python
# Skill executes:
# 1. Initial pass (confidence = 0.72)
# 2. Below threshold (0.85) → triggers refinement
# 3. Refinement pass (confidence = 0.89)
# 4. Above threshold → returns result

result = await execute_skill(
    "code-reviewer",
    parameters={...},
    # These come from skill YAML:
    # quality_threshold: 0.85
    # max_iterations: 3
)
```

## Creating Custom Skills

Create a new YAML file in `hololoom/agentic/skills/`:

```yaml
# my_custom_skill.yaml
name: my-custom-skill
version: "1.0.0"
description: "Custom skill description"

metadata:
  category: "development"
  tags: ["custom", "example"]
  created: "2025-11-16"

reasoning:
  default_strategy: "refine"
  max_iterations: 3
  quality_threshold: 0.85
  refinement_passes:
    - name: "initial_analysis"
      focus: "Understand the problem"
    - name: "solution_design"
      focus: "Design comprehensive solution"

system_prompt: |
  You are an expert in...

user_prompt_template: |
  {instruction}

  Input:
  {input}

  Provide:
  1. Analysis
  2. Recommendation

parameters:
  - name: input
    type: string
    required: true
  - name: instruction
    type: string
    default: "Analyze the following"

output:
  format: "structured"
  fields:
    - analysis: string
    - recommendation: string
```

Then use it:

```python
result = await execute_skill(
    "my-custom-skill",
    parameters={"input": "..."}
)
```

## Advanced Usage

### Using SkillRegistry Directly

```python
from hololoom.agentic.skill_agents import SkillRegistry, SkillExecutor
from hololoom.config import Config

# Load registry
registry = SkillRegistry()
await registry.load_all_skills()

# Inspect a skill
skill = registry.get_skill("code-reviewer")
print(skill.description)
print(skill.reasoning.default_strategy)
print(skill.parameters)

# Execute via executor
executor = SkillExecutor(
    registry=registry,
    config=Config.fused(),
    enable_analytics=True
)

result = await executor.execute(
    skill_name="code-reviewer",
    parameters={...}
)
```

### With Memory Context

```python
from hololoom.documentation.types import MemoryShard

# Provide memory shards for context
shards = [
    MemoryShard(
        content="Previous code review flagged missing error handling",
        entities=["error handling"],
        motifs=["best practices"]
    )
]

result = await execute_skill(
    "code-reviewer",
    parameters={...},
    shards=shards  # Context from previous interactions
)
```

### Accessing Reasoning Provenance

```python
result = await execute_skill("bug-detective", parameters={...})

# Complete thought process
if result.metadata.get("reasoning_journal"):
    print(result.metadata["reasoning_journal"])

# Output:
# Iteration 1:
#   Thought: The error indicates a NoneType...
#   Action: Analyze the code path...
#   Confidence: 0.75
#
# Iteration 2:
#   Thought: The root cause is missing null check...
#   Action: Suggest fix with optional chaining...
#   Confidence: 0.92
```

## Running the Demo

```bash
PYTHONPATH=. python demos/demo_skill_agents.py
```

Demonstrates:
1. Listing all available skills
2. Code review with CRITIQUE strategy
3. Bug detective with DECOMPOSE strategy
4. Test generation with EXPLORE strategy
5. Strategy comparison (REFINE vs CRITIQUE vs DECOMPOSE)
6. Reasoning provenance (complete thought process)

## File Structure

```
hololoom/agentic/
├── skill_agents.py           # Main skill system (820 lines)
├── skills/                   # Skill templates (13 files)
│   ├── code_reviewer.yaml
│   ├── bug_detective.yaml
│   ├── test_generator.yaml
│   ├── api_designer.yaml
│   ├── documentation_writer.yaml
│   ├── performance_profiler.yaml
│   ├── architecture_advisor.yaml
│   ├── migration_planner.yaml
│   ├── code_explainer.yaml
│   ├── naming_consultant.yaml
│   ├── sql_optimizer.yaml
│   ├── refactoring_expert.yaml
│   └── security_auditor.yaml
└── SKILL_AGENTS_README.md    # This file

demos/
└── demo_skill_agents.py      # Comprehensive demo (250 lines)
```

## Integration with HoloLoom Components

### Recursive Weaving Orchestrator

Skills execute via `RecursiveWeavingOrchestrator`:
- Automatic quality-driven refinement
- Strategy-based recursive reasoning
- Complete provenance via ReasoningJournal

### Analytics

All skill executions tracked:
- Strategy performance
- Quality improvements
- Token usage and cost
- Execution time

### Alignment Framework

Skills inherit HoloLoom's alignment:
- Safety guardrails
- Deception detection
- Audit trail
- Risk-based gating

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Load skill registry | ~50ms | One-time startup cost |
| Execute skill (FAST config) | ~150-300ms | Depends on strategy/iterations |
| Execute skill (FUSED config) | ~300-600ms | Higher quality, more context |
| Refinement iteration | ~150ms | Per iteration |

## Best Practices

1. **Choose appropriate strategy**: Match strategy to task complexity
   - Simple tasks: REFINE (2 iterations)
   - Code review: CRITIQUE (3 iterations)
   - Debugging: DECOMPOSE (5 iterations)

2. **Set quality thresholds**: Higher threshold = more refinement
   - Development: 0.85 (standard)
   - Security: 0.90 (high confidence needed)
   - Documentation: 0.80 (acceptable)

3. **Use memory context**: Provide relevant shards for better results

4. **Monitor analytics**: Track which skills/strategies work best

5. **Custom skills**: Create domain-specific skills for your use case

## Future Enhancements

Planned for Phase 4-5:

1. **MCP Server Integration**: Expose skills to Claude Desktop
2. **Real-time Dashboard**: Visualize skill performance
3. **Skill Composition**: Chain multiple skills together
4. **Learning from Feedback**: Adapt strategies based on outcomes
5. **Skill Marketplace**: Share custom skills

## Comparison to Promptly

| Feature | Promptly | HoloLoom Skill Agents |
|---------|----------|----------------------|
| **Skill Templates** | ✅ 13 templates | ✅ 13 templates (same) |
| **Recursive Reasoning** | ✅ 6 loop types | ✅ Integrated with HoloLoom |
| **Quality-Driven** | ✅ Yes | ✅ Yes + auto-refinement |
| **Analytics** | ✅ Execution tracking | ✅ Full integration |
| **Provenance** | ✅ Scratchpad | ✅ ReasoningJournal |
| **Memory Integration** | ❌ No | ✅ Knowledge graph + embeddings |
| **Alignment** | ❌ No | ✅ Safety guardrails |
| **Architecture** | Standalone CLI | Protocol-based HoloLoom |

**Key Innovation**: Skill agents leverage HoloLoom's entire infrastructure (memory, alignment, analytics, recursive reasoning) rather than being standalone.

## See Also

- **Phase 1**: [PROMPTLY_HOLOLOOM_INTEGRATION.md](../../PROMPTLY_HOLOLOOM_INTEGRATION.md) - Core recursive reasoning
- **Phase 2**: [analytics/README.md](../analytics/README.md) - Analytics integration
- **Phase 4**: [MCP Server](../../docs/MCP_SERVER.md) - Claude Desktop integration (coming soon)
- **Phase 5**: [Real-time Dashboard](../../docs/DASHBOARD.md) - Visualization (coming soon)

---

**Version**: 1.0.0
**Created**: 2025-11-16
**Integration**: Promptly Professional Skills → HoloLoom Agents
