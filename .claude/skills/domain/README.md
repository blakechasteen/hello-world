# Domain-Specific Skills

**Last Updated**: 2025-01-18

This directory contains domain-specific skills for specialized tasks. Unlike meta-skills (which manage other skills), domain skills perform specific tasks in specialized areas like programming languages, frameworks, data analysis, DevOps, etc.

---

## Overview

**Domain skills** are:
- Task-focused (solve specific problems)
- Reusable across similar use cases
- Often created from recurring user patterns (via `continuous_learning_capture`)
- Integrated with HoloLoom for context-aware operations

**Examples of Domain Skills**:
- `typescript_error_explainer` - Explains TypeScript compiler errors
- `react_performance_optimizer` - Optimizes React components
- `hololoom_rag_helper` - Assists with HoloLoom RAG operations
- `sql_query_optimizer` - Optimizes database queries
- `dockerfile_generator` - Generates Dockerfiles for projects

---

## When to Create a Domain Skill

Create a domain skill when:

1. **Recurring Pattern**: Users ask for the same type of help ≥3 times
2. **Well-Defined Scope**: The task has clear inputs and outputs
3. **Generalizable**: The skill applies to multiple use cases (not one-off)
4. **Specialized Knowledge**: The task requires domain expertise
5. **Performance Gain**: A dedicated skill is faster/better than general queries

**Don't create a domain skill if**:
- The task is too specific (one-off request)
- The task is too broad (needs to be split into multiple skills)
- An existing skill already covers it
- The task changes frequently (unstable requirements)

---

## Creating a New Domain Skill

### 1. Choose a Name

Skill names should be:
- Descriptive and clear (`typescript_error_explainer`, not `ts_helper`)
- Lowercase with underscores
- Action-oriented (what it does)
- Specific enough to avoid confusion

**Good Examples**:
- `json_to_typescript_converter`
- `react_performance_optimizer`
- `sql_query_optimizer`

**Bad Examples**:
- `helper` (too vague)
- `TypeScriptTool` (not lowercase)
- `do_stuff` (not descriptive)

### 2. Create Directory

```bash
mkdir -p skills/domain/your_skill_name
```

### 3. Copy Template

```bash
cp skills/templates/skill.markdown.template \
   skills/domain/your_skill_name/skill.markdown
```

### 4. Fill in Metadata

Required metadata for domain skills:

```markdown
- **Name**: `your_skill_name`
- **Version**: `1.0.0`
- **Author**: `Your Name`
- **Created**: `2025-01-18`
- **Last Updated**: `2025-01-18`
- **Category**: `domain`
- **Tags**: `typescript, errors, debugging` (relevant keywords)
```

### 5. Define Input/Output Schemas

Domain skills should have well-defined schemas:

**Input Schema Example**:
```json
{
  "code_snippet": "TypeScript code to analyze",
  "error_code": "TS2322",
  "context": "Additional context (optional)"
}
```

**Output Schema Example**:
```json
{
  "explanation": "Plain language explanation of error",
  "fix_suggestion": "How to fix the error",
  "example_fix": "Example of corrected code",
  "metadata": {
    "confidence": 0.95,
    "execution_time_ms": 120
  }
}
```

### 6. Write Prompt Template

**Tips for domain skill prompts**:

1. **State Purpose Clearly**:
   ```markdown
   You are a TypeScript error explainer. Your task is to explain
   TypeScript compiler errors in plain language and suggest fixes.
   ```

2. **Define Input Format**:
   ```markdown
   **Input Data**:
   {input_data}

   **Code Snippet**:
   {code_snippet}
   ```

3. **Provide Step-by-Step Instructions**:
   ```markdown
   **Steps**:
   1. Analyze the error code and context
   2. Identify the root cause
   3. Explain in plain language
   4. Suggest specific fix
   ```

4. **Specify Output Format**:
   ```markdown
   Return results as JSON:
   {
     "explanation": "...",
     "fix_suggestion": "...",
     "example_fix": "..."
   }
   ```

### 7. Add Real Examples

Domain skills benefit from **real examples** (from actual usage):

```markdown
### Example 1: Type Mismatch Error

**Input**:
\`\`\`json
{
  "code_snippet": "const x: number = 'hello';",
  "error_code": "TS2322"
}
\`\`\`

**Expected Output**:
\`\`\`json
{
  "explanation": "TS2322 means you're assigning a value of one type to a variable of a different type",
  "fix_suggestion": "Change the type annotation to match the value, or change the value to match the type",
  "example_fix": "const x: string = 'hello'; // OR const x: number = 42;"
}
\`\`\`

**Source**: User interaction 2025-01-18T10:30:00Z
```

**Tip**: If created via `continuous_learning_capture`, real examples are auto-populated!

### 8. Specify Capabilities

Domain skills often need specific capabilities:

**Common Capabilities for Domain Skills**:

- **Code Analysis**: `File system access (read)`, `Code execution (python)`
- **Documentation**: `File system access (read/write)`
- **API/Web**: `Network access (web fetch)`
- **Database**: `External API access`
- **HoloLoom**: Mark which HoloLoom systems used

**Example**:
```markdown
## Required Capabilities

- [x] File system access (read) - to read code files
- [ ] File system access (write)
- [x] Code execution (python) - to parse TypeScript AST
- [ ] Network access (web fetch)
```

### 9. Test Thoroughly

Domain skills should be rigorously tested:

```bash
# Validate structure
python scripts/build_skill.py skills/domain/your_skill_name --validate-only

# Test manually with real inputs
# Run skill in Claude Code and test all examples

# Check security
# Invoke skill_security_analyzer (once implemented)

# Check performance
# Run token_budget_adviser to optimize
```

### 10. Deploy

```bash
# Build package
python scripts/build_skill.py skills/domain/your_skill_name

# Deploy to Claude Code
cp skills/dist/your_skill_name-1.0.0.skill .claude/skills/

# Or upload to Claude Web
# Use web UI to upload .skill package
```

---

## HoloLoom Integration

Many domain skills benefit from hololoom integration:

### Memory System Integration

Use HoloLoom's memory to provide context:

```markdown
**HoloLoom Integration**:
- [x] Uses HoloLoom memory system

**How**: Retrieves similar past errors/solutions from memory graph
to provide more accurate explanations.
```

**Example in Prompt**:
```markdown
Before explaining the error, query HoloLoom memory for similar
TypeScript errors the user has encountered. Use this context to
provide personalized explanations.
```

### RAG Integration

Use HoloLoom RAG for documentation lookup:

```markdown
**HoloLoom Integration**:
- [x] Uses HoloLoom RAG

**How**: Queries official TypeScript documentation via RAG to ensure
accurate explanations.
```

**Example in Prompt**:
```markdown
Use HoloLoom RAG to retrieve relevant sections from TypeScript
documentation before explaining the error.
```

### Alignment Framework Integration

For skills that perform risky operations:

```markdown
**HoloLoom Integration**:
- [x] Uses HoloLoom alignment framework

**How**: All code modifications are gated by safety guardrails to
prevent destructive operations.
```

**Example in Prompt**:
```markdown
Before applying any fix, run through HoloLoom safety guardrails:
1. Risk assessment (LOW/MEDIUM/HIGH)
2. Impact analysis (what will change)
3. Approval gate (auto-approve LOW, ask user for MEDIUM/HIGH)
```

### Learning Systems Integration

Skills can learn from outcomes:

```markdown
**HoloLoom Integration**:
- [x] Uses HoloLoom learning systems

**How**: Tracks success/failure of explanations to improve accuracy
over time via Thompson Sampling.
```

---

## Domain Skill Categories

Organize domain skills by domain:

### Programming Languages
- TypeScript, Python, JavaScript, Rust, Go, etc.
- Error explanation, optimization, conversion

### Frameworks
- React, Vue, Angular, Django, FastAPI, etc.
- Performance optimization, best practices, debugging

### Infrastructure
- Docker, Kubernetes, AWS, GCP, Azure
- Configuration, deployment, troubleshooting

### Data
- SQL, MongoDB, Redis, Elasticsearch
- Query optimization, schema design, migration

### HoloLoom-Specific
- Memory operations, RAG queries, alignment checks
- Skill management, learning analytics

### Tools
- Git, CI/CD, testing, linting
- Workflow automation, integration helpers

---

## Best Practices for Domain Skills

### 1. Single Domain Focus

Each skill should focus on **one domain**:

✅ **Good**: `typescript_error_explainer` (TypeScript only)
❌ **Bad**: `programming_language_error_explainer` (too broad)

### 2. Consistent Input/Output

Within a domain, skills should have consistent formats:

**TypeScript Skills** (consistent schema):
- `typescript_error_explainer` - Input: code + error → Output: explanation
- `typescript_optimizer` - Input: code → Output: optimized code
- `typescript_type_generator` - Input: JSON → Output: TypeScript types

### 3. Leverage HoloLoom Memory

Domain skills should use HoloLoom memory for context:

```markdown
# Before explaining this specific React error, check if user has
# encountered similar React errors before and reference those.
```

### 4. Document Real Examples

Always include **real examples from usage**:

```markdown
**Source**: User interaction 2025-01-15T14:22:00Z
**Success Rate**: 100% (3/3 similar requests)
```

### 5. Optimize for Performance

Domain skills run frequently - optimize token usage:

```bash
# Run token budget adviser
python scripts/run_token_budget_adviser.py skills/domain/your_skill

# Apply recommendations (compress prompt, reduce examples)
```

### 6. Version Incrementally

Track changes via semantic versioning:

```markdown
**Changelog**:
- **v1.2.0** (2025-01-20): Added support for TS5.0 errors
- **v1.1.0** (2025-01-15): Improved explanation clarity
- **v1.0.0** (2025-01-10): Initial release
```

---

## Example Domain Skills

Here are examples to study:

### 1. TypeScript Error Explainer

**Purpose**: Explains TypeScript compiler errors in plain language

**Key Features**:
- Maps error codes (TS2322, TS2345) to explanations
- Provides fix suggestions with examples
- Uses HoloLoom RAG for official docs

**Usage**:
```json
Input: {"error_code": "TS2322", "code": "const x: number = 'hello';"}
Output: "TS2322 is a type mismatch. You're assigning string to number..."
```

### 2. React Performance Optimizer

**Purpose**: Analyzes React components and suggests optimizations

**Key Features**:
- Detects missing memoization (React.memo, useMemo)
- Identifies unnecessary re-renders
- Suggests virtualization for long lists

**Usage**:
```json
Input: {"component_code": "function MyComponent() {...}"}
Output: {"optimizations": ["Add React.memo", "Use useCallback for handlers"]}
```

### 3. HoloLoom RAG Helper

**Purpose**: Assists with HoloLoom RAG operations (ingest, query, multimodal)

**Key Features**:
- Generates RAG queries from natural language
- Suggests optimal reasoning modes (DIRECT/VERIFY/RESEARCH)
- Provides example code snippets

**Usage**:
```json
Input: {"task": "Search for information about Thompson Sampling"}
Output: {"mode": "VERIFY", "code": "await rag.query('What is Thompson Sampling?', mode='verify')"}
```

---

## Skill Discovery

**How users find your skill**:

1. **Tags**: Add relevant tags to metadata
   ```markdown
   **Tags**: `typescript, errors, debugging, compiler`
   ```

2. **Description**: Clear, searchable description
   ```markdown
   **Short Description**: Explains TypeScript compiler errors in plain language with fix suggestions
   ```

3. **Gap Analyzer**: `skill_gap_analyzer` identifies when your skill fills a need

4. **Documentation**: Maintain this README with skill catalog

---

## Contributing

When contributing domain skills:

1. **Check for duplicates**: Run `skill_gap_analyzer` to ensure no overlap
2. **Follow template**: Use `skills/templates/skill.markdown.template`
3. **Test thoroughly**: All examples must work correctly
4. **Security review**: Pass `skill_security_analyzer`
5. **Document well**: Clear descriptions, real examples
6. **Tag appropriately**: Relevant, specific tags

---

## Roadmap

Planned domain skill categories:

**Phase 1** (Current):
- TypeScript/JavaScript skills
- React optimization
- HoloLoom integration helpers

**Phase 2**:
- Python data science skills
- SQL/database optimization
- Docker/Kubernetes helpers

**Phase 3**:
- Multi-language support (Rust, Go, Java)
- Advanced HoloLoom features (multi-agent, custom learning)
- CI/CD automation skills

**Phase 4**:
- Domain-specific AI/ML skills
- Security/compliance helpers
- Performance profiling/optimization

---

## Getting Help

- **Workflow Guide**: See `docs/skills_workflow.md` for complete workflow
- **Template**: Use `skills/templates/skill.markdown.template`
- **Examples**: Study `skills/meta/` for well-structured skills
- **Issues**: Report problems at GitHub issues

---

**Maintained By**: HoloLoom Team
**Last Updated**: 2025-01-18
