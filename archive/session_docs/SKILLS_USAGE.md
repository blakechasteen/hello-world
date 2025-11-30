# How to Create Custom Skills - Complete Guide

**Date**: November 2025
**Audience**: Developers creating custom HoloLoom skills
**Difficulty**: Intermediate
**Time**: 15-30 minutes per skill

---

## Table of Contents

1. [Introduction](#introduction)
2. [YAML Template Structure](#yaml-template-structure)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Parameter Types](#parameter-types)
5. [Reasoning Strategies](#reasoning-strategies)
6. [Prompt Engineering](#prompt-engineering)
7. [Testing Your Skill](#testing-your-skill)
8. [Best Practices](#best-practices)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

HoloLoom Skills are **declarative YAML templates** that define specialized reasoning capabilities. Creating a skill is simple:

1. **Write a YAML file** with prompts and parameters
2. **Save it** to `HoloLoom/agentic/skills/`
3. **Use it** via `execute_skill()`

No Python code required! The Skills System handles:
- Parameter validation
- Prompt rendering
- Orchestrator integration
- Multi-pass refinement
- Quality checking

---

## YAML Template Structure

Every skill YAML file has 7 sections:

```yaml
# 1. Basic Metadata
name: "skill-name"
version: "1.0.0"
description: "One-line description"

# 2. Classification
metadata:
  category: "development"
  author: "Your Name"
  tags: ["tag1", "tag2"]

# 3. System Context
system_prompt: |
  You are an expert in...

# 4. User Task
user_prompt_template: |
  {parameter1}: {parameter1}
  Please do...

# 5. Input Schema
parameters:
  - name: parameter1
    type: string
    required: true

# 6. Quality Control
reasoning:
  default_strategy: "refine"
  max_iterations: 3
  quality_threshold: 0.85
```

---

## Step-by-Step Tutorial

Let's create a **JSON Validator** skill from scratch.

### Step 1: Choose a Name

- Use lowercase with hyphens (not underscores)
- Be descriptive but concise
- Example: `json-validator` (not `JSONValidator` or `json_validator`)

### Step 2: Create YAML File

Create `HoloLoom/agentic/skills/json-validator.yaml`:

```yaml
name: "json-validator"
version: "1.0.0"
description: "Validate JSON syntax and structure"
```

### Step 3: Add Metadata

```yaml
metadata:
  category: "development"  # Choose from: architecture, development, education, optimization, security, database, general
  author: "Your Name"
  tags: ["json", "validation", "data"]
  created: "2025-11-22"
```

### Step 4: Write System Prompt

Define the skill's expertise and perspective:

```yaml
system_prompt: |
  You are a JSON validation expert with deep knowledge of:
  - JSON RFC 8259 specification
  - Common JSON syntax errors
  - Best practices for JSON schema design
  - Error messages that help developers fix issues quickly

  Your role is to:
  1. Validate JSON syntax strictly
  2. Identify structural issues
  3. Suggest corrections with clear explanations
  4. Provide examples of correct format when needed

  Always be precise, helpful, and educational.
```

**Tips**:
- Define expertise clearly (what does the AI know?)
- Specify the role (what should it do?)
- Set the tone (how should it respond?)
- Be specific about output format

### Step 5: Create User Prompt Template

Define the task with parameters:

```yaml
user_prompt_template: |
  JSON to validate:
  ```
  {json_string}
  ```

  {% if schema %}
  Expected schema:
  ```
  {schema}
  ```
  {% endif %}

  Please validate this JSON and provide:
  1. Is it valid? (YES/NO)
  2. If invalid, what are the syntax errors?
  3. If schema provided, does it match?
  4. Suggestions for fixes (if applicable)

  Format your response clearly with numbered sections.
```

**Tips**:
- Use `{parameter_name}` for required parameters
- Use `{% if parameter %}...{% endif %}` for optional parameters
- Structure the request clearly
- Specify output format expectations

### Step 6: Define Parameters

```yaml
parameters:
  - name: json_string
    type: string
    required: true
    description: "JSON string to validate"

  - name: schema
    type: object
    required: false
    description: "Optional JSON schema for structure validation"
```

**Tips**:
- Required parameters must be provided
- Optional parameters have defaults or are conditionally used
- Choose appropriate types (string, number, boolean, array, object, code)
- Write clear descriptions for documentation

### Step 7: Configure Reasoning

```yaml
reasoning:
  default_strategy: "verify"  # Good for validation tasks
  max_iterations: 2           # Validation shouldn't need many passes
  quality_threshold: 0.90     # High confidence for correctness
```

**Strategy Choices**:
- `refine`: General improvement (default)
- `critique`: Quick review
- `verify`: Fact-checking (good for validation!)
- `elegance`: Code quality
- `hofstadter`: Meta-reasoning

### Step 8: Complete File

Your complete `json-validator.yaml`:

```yaml
name: "json-validator"
version: "1.0.0"
description: "Validate JSON syntax and structure"

metadata:
  category: "development"
  author: "Your Name"
  tags: ["json", "validation", "data"]
  created: "2025-11-22"

system_prompt: |
  You are a JSON validation expert with deep knowledge of:
  - JSON RFC 8259 specification
  - Common JSON syntax errors
  - Best practices for JSON schema design
  - Error messages that help developers fix issues quickly

  Your role is to:
  1. Validate JSON syntax strictly
  2. Identify structural issues
  3. Suggest corrections with clear explanations
  4. Provide examples of correct format when needed

  Always be precise, helpful, and educational.

user_prompt_template: |
  JSON to validate:
  ```
  {json_string}
  ```

  {% if schema %}
  Expected schema:
  ```
  {schema}
  ```
  {% endif %}

  Please validate this JSON and provide:
  1. Is it valid? (YES/NO)
  2. If invalid, what are the syntax errors?
  3. If schema provided, does it match?
  4. Suggestions for fixes (if applicable)

  Format your response clearly with numbered sections.

parameters:
  - name: json_string
    type: string
    required: true
    description: "JSON string to validate"

  - name: schema
    type: object
    required: false
    description: "Optional JSON schema for structure validation"

reasoning:
  default_strategy: "verify"
  max_iterations: 2
  quality_threshold: 0.90
```

### Step 9: Test Your Skill

```python
from HoloLoom.agentic import execute_skill
from HoloLoom.config import Config

# Test with valid JSON
result = await execute_skill(
    skill_name="json-validator",
    parameters={
        "json_string": '{"name": "Alice", "age": 30}'
    },
    config=Config.fast()
)

print(result.output)

# Test with invalid JSON
result = await execute_skill(
    skill_name="json-validator",
    parameters={
        "json_string": '{"name": "Alice", "age": 30'  # Missing closing }
    }
)

print(result.output)
```

**Done!** Your skill is now available system-wide.

---

## Parameter Types

### string

General text input:

```yaml
parameters:
  - name: description
    type: string
    required: true
    description: "Free-form text description"
```

**Use for**: Names, descriptions, queries, plain text

### code

Code snippets (syntax-aware):

```yaml
parameters:
  - name: code
    type: code
    required: true
    description: "Code to analyze"
```

**Use for**: Source code, scripts, SQL queries

### number

Integers or floats:

```yaml
parameters:
  - name: timeout
    type: number
    required: false
    default: 30
    description: "Timeout in seconds"
```

**Use for**: Counts, limits, thresholds, durations

### boolean

True/false flags:

```yaml
parameters:
  - name: verbose
    type: boolean
    required: false
    default: false
    description: "Enable verbose output"
```

**Use for**: Feature flags, options, switches

### array

Lists of items:

```yaml
parameters:
  - name: files
    type: array
    required: true
    description: "List of file paths to process"
```

**Use for**: Multiple items, collections, lists

### object

Structured data (dictionaries):

```yaml
parameters:
  - name: config
    type: object
    required: false
    description: "Configuration object with custom settings"
```

**Use for**: Complex nested data, JSON objects, configurations

---

## Reasoning Strategies

### REFINE (Default)

**Best for**: General improvement through context expansion

```yaml
reasoning:
  default_strategy: "refine"
  max_iterations: 3
  quality_threshold: 0.85
```

**How it works**:
- Pass 1: Initial response
- Pass 2: Expand context, improve clarity
- Pass 3: Final refinement

**Use when**: You want iterative improvement without specific focus

### CRITIQUE

**Best for**: Quick self-review

```yaml
reasoning:
  default_strategy: "critique"
  max_iterations: 1
  quality_threshold: 0.80
```

**How it works**:
- Generate response
- Self-critique and improve in single pass

**Use when**: You want one refinement pass for quality

### VERIFY

**Best for**: Fact-checking and accuracy

```yaml
reasoning:
  default_strategy: "verify"
  max_iterations: 3
  quality_threshold: 0.90
```

**How it works**:
- Pass 1: Accuracy check
- Pass 2: Completeness check
- Pass 3: Consistency check

**Use when**: Correctness is critical (validation, auditing, fact-checking)

### ELEGANCE

**Best for**: Code quality and clarity

```yaml
reasoning:
  default_strategy: "elegance"
  max_iterations: 3
  quality_threshold: 0.85
```

**How it works**:
- Pass 1: Clarity improvement
- Pass 2: Simplification
- Pass 3: Beauty (elegance, style)

**Use when**: Code or writing quality matters

### HOFSTADTER

**Best for**: Meta-reasoning and self-reference

```yaml
reasoning:
  default_strategy: "hofstadter"
  max_iterations: 5
  quality_threshold: 0.85
```

**How it works**:
- Recursive self-examination
- Strange loops and meta-awareness
- Iterative deepening

**Use when**: Complex reasoning about reasoning itself

---

## Prompt Engineering

### System Prompt Best Practices

**DO**:
✅ Define expertise clearly
✅ Specify the role and responsibilities
✅ Set tone and style
✅ Provide context about the task domain

**DON'T**:
❌ Include specific instructions (use user_prompt_template instead)
❌ Reference parameters (they don't exist in system prompt)
❌ Make it too long (focus on expertise, not the task)

**Good Example**:
```yaml
system_prompt: |
  You are a security auditing expert specializing in web application vulnerabilities.
  You have deep knowledge of OWASP Top 10, secure coding practices, and common attack vectors.

  Your role is to identify security issues and provide actionable remediation steps.
  Always prioritize critical vulnerabilities and explain risks clearly.
```

**Bad Example**:
```yaml
system_prompt: |
  Review this code: {code}  # ❌ Parameters don't work here
  Find bugs.                # ❌ Too vague, should be in user prompt
  Be helpful.               # ❌ Not expertise-defining
```

### User Prompt Template Best Practices

**DO**:
✅ Use `{parameter_name}` for required parameters
✅ Use `{% if param %}...{% endif %}` for optional parameters
✅ Structure instructions clearly (numbered/bulleted lists)
✅ Specify expected output format

**DON'T**:
❌ Assume parameters exist (check with {% if %})
❌ Be vague about what you want
❌ Mix system context with task instructions

**Good Example**:
```yaml
user_prompt_template: |
  Code to review:
  ```{language}
  {code}
  ```

  {% if focus_areas %}
  Focus on: {focus_areas}
  {% endif %}

  Please provide:
  1. Correctness issues (if any)
  2. Performance concerns
  3. Security vulnerabilities
  4. Readability suggestions

  For each issue, include:
  - Line number (if applicable)
  - Severity (HIGH/MEDIUM/LOW)
  - Recommended fix
```

**Bad Example**:
```yaml
user_prompt_template: |
  Review {code} in {language}.  # ❌ Too vague
  # ❌ No output format specified
  # ❌ Doesn't handle optional parameters
```

### Template Variables

Use Jinja2 syntax for conditionals:

```yaml
user_prompt_template: |
  # Conditional sections
  {% if optional_param %}
  Optional context: {optional_param}
  {% endif %}

  # Loops (if parameter is array)
  {% for item in items %}
  - {item}
  {% endfor %}

  # Filters
  {text|upper}      # Uppercase
  {text|lower}      # Lowercase
  {text|title}      # Title Case
```

---

## Testing Your Skill

### Manual Testing

```python
from HoloLoom.agentic import execute_skill
from HoloLoom.config import Config

# Test 1: Basic functionality
result = await execute_skill(
    skill_name="your-skill",
    parameters={"required_param": "value"},
    config=Config.fast()
)

assert result.success, f"Skill failed: {result.error}"
print(f"Output: {result.output}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Iterations: {result.iterations}")

# Test 2: Edge cases
result = await execute_skill(
    skill_name="your-skill",
    parameters={"required_param": ""},  # Empty string
)

# Test 3: Optional parameters
result = await execute_skill(
    skill_name="your-skill",
    parameters={
        "required_param": "value",
        "optional_param": "optional_value"
    }
)

# Test 4: Performance
import time
start = time.time()
result = await execute_skill("your-skill", params)
duration = time.time() - start
print(f"Execution time: {duration:.2f}s")
```

### Automated Testing

Create integration test in `HoloLoom/agentic/skills/tests/test_integration.py`:

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_your_skill_execution(self, config):
    """Test your-skill end-to-end."""
    result = await execute_skill(
        skill_name="your-skill",
        parameters={"required_param": "test value"},
        config=config
    )

    # Verify result structure
    assert isinstance(result, SkillExecutionResult)
    assert result.skill_name == "your-skill"
    assert result.success is True

    # Verify output quality
    assert result.output, "Output should not be empty"
    assert result.confidence >= 0.7, "Confidence too low"

    # Verify expected content
    assert "expected keyword" in result.output.lower()
```

Run test:
```bash
pytest HoloLoom/agentic/skills/tests/test_integration.py::TestEndToEndExecution::test_your_skill_execution -v
```

### Quality Checklist

Before considering your skill complete:

- [ ] YAML syntax is valid (run YAML validator)
- [ ] All required fields present
- [ ] Parameters have clear descriptions
- [ ] System prompt defines expertise
- [ ] User prompt template is structured
- [ ] Reasoning config is appropriate
- [ ] Manual testing passes
- [ ] Integration test added
- [ ] Documentation updated (README.md)
- [ ] Example usage provided

---

## Best Practices

### 1. Single Responsibility

Each skill should do **one thing well**:

✅ **Good**: `code-reviewer` - Reviews code quality
✅ **Good**: `bug-detective` - Finds potential bugs
❌ **Bad**: `code-helper` - Reviews, finds bugs, generates tests (too broad!)

**Why**: Focused skills are easier to test, maintain, and compose.

### 2. Clear Parameter Names

Use descriptive, unambiguous names:

✅ **Good**: `source_code`, `target_language`, `max_results`
❌ **Bad**: `input`, `output`, `data`, `value`

**Why**: Users should understand what to provide without reading docs.

### 3. Reasonable Defaults

Provide sensible defaults for optional parameters:

```yaml
parameters:
  - name: max_results
    type: number
    required: false
    default: 10  # ✅ Reasonable default

  - name: timeout_seconds
    type: number
    required: false
    default: 30  # ✅ Good default

  - name: include_examples
    type: boolean
    required: false
    default: true  # ✅ Most users want examples
```

### 4. Appropriate Quality Thresholds

Set thresholds based on task criticality:

```yaml
# Security auditing - high threshold
reasoning:
  quality_threshold: 0.95  # Must be very confident

# Code explanation - moderate threshold
reasoning:
  quality_threshold: 0.80  # Explanation doesn't need perfection

# Creative writing - lower threshold
reasoning:
  quality_threshold: 0.70  # Creativity over confidence
```

### 5. Iteration Limits

Prevent infinite loops:

```yaml
reasoning:
  max_iterations: 3  # ✅ Typical - most tasks improve in 2-3 passes
  max_iterations: 1  # ✅ For simple tasks (critique strategy)
  max_iterations: 5  # ✅ For complex reasoning (hofstadter)
  max_iterations: 10 # ❌ Too many - likely won't improve after 5
```

### 6. Structured Output

Request structured formats in user prompt:

```yaml
user_prompt_template: |
  Analyze this code: {code}

  Provide your analysis in this format:

  ## Summary
  [One-line summary]

  ## Issues Found
  1. [Issue 1] - Severity: HIGH/MEDIUM/LOW
  2. [Issue 2] - Severity: HIGH/MEDIUM/LOW

  ## Recommendations
  - [Recommendation 1]
  - [Recommendation 2]
```

**Why**: Structured output is easier to parse and integrate.

### 7. Error Guidance

Help users fix errors:

```yaml
user_prompt_template: |
  {% if not code %}
  ERROR: No code provided. Please provide code to review.
  {% else %}
  # ... actual template ...
  {% endif %}
```

---

## Examples

### Example 1: Simple Skill (Minimal Template)

`greeting-generator.yaml`:
```yaml
name: "greeting-generator"
version: "1.0.0"
description: "Generate personalized greetings"

metadata:
  category: "general"
  author: "Tutorial"
  tags: ["greeting", "text"]

system_prompt: |
  You are a friendly greeting generator.
  Create warm, personalized greetings for any occasion.

user_prompt_template: |
  Generate a greeting for {name} on {occasion}.

parameters:
  - name: name
    type: string
    required: true
    description: "Person's name"

  - name: occasion
    type: string
    required: true
    description: "Occasion (birthday, holiday, etc.)"

reasoning:
  default_strategy: "refine"
  max_iterations: 2
  quality_threshold: 0.80
```

Usage:
```python
result = await execute_skill(
    "greeting-generator",
    {"name": "Alice", "occasion": "birthday"}
)
```

### Example 2: Complex Skill (Full Template)

`api-security-auditor.yaml`:
```yaml
name: "api-security-auditor"
version: "1.0.0"
description: "Comprehensive security audit for REST APIs"

metadata:
  category: "security"
  author: "Security Team"
  tags: ["api", "security", "audit", "owasp"]
  created: "2025-11-22"
  updated: "2025-11-22"

system_prompt: |
  You are an API security expert with deep knowledge of:
  - OWASP API Security Top 10
  - Authentication and authorization best practices
  - Common API vulnerabilities (injection, broken auth, excessive data exposure)
  - Rate limiting and DoS prevention
  - API gateway security

  Your role is to:
  1. Identify security vulnerabilities in API designs
  2. Assess risk levels (CRITICAL, HIGH, MEDIUM, LOW)
  3. Provide actionable remediation steps
  4. Reference OWASP guidelines and industry standards

  Always be thorough, precise, and security-focused.

user_prompt_template: |
  API Endpoint to audit:
  ```
  {endpoint_spec}
  ```

  {% if authentication_method %}
  Authentication Method: {authentication_method}
  {% endif %}

  {% if authorization_model %}
  Authorization Model: {authorization_model}
  {% endif %}

  {% if rate_limiting %}
  Rate Limiting: {rate_limiting}
  {% endif %}

  Please conduct a comprehensive security audit covering:

  1. **Authentication & Authorization**
     - Is authentication properly implemented?
     - Are authorization checks in place?
     - Any broken access control risks?

  2. **Input Validation**
     - Are inputs validated?
     - Any injection vulnerabilities (SQL, NoSQL, command)?
     - Proper encoding/escaping?

  3. **Data Exposure**
     - Any excessive data exposure?
     - Sensitive data properly protected?
     - PII handling appropriate?

  4. **Rate Limiting & DoS**
     - Rate limiting implemented?
     - DoS attack vectors?
     - Resource exhaustion risks?

  5. **Other OWASP API Security Risks**
     - Mass assignment vulnerabilities?
     - Security misconfigurations?
     - Improper asset management?

  For each issue found, provide:
  - **Risk Level**: CRITICAL/HIGH/MEDIUM/LOW
  - **Description**: What's the vulnerability?
  - **Impact**: What's the potential damage?
  - **Remediation**: Specific fix with code example if applicable
  - **OWASP Reference**: Link to relevant OWASP guideline

parameters:
  - name: endpoint_spec
    type: string
    required: true
    description: "API endpoint specification (OpenAPI, description, or code)"

  - name: authentication_method
    type: string
    required: false
    description: "Authentication method used (JWT, OAuth2, API Key, etc.)"

  - name: authorization_model
    type: string
    required: false
    description: "Authorization model (RBAC, ABAC, ACL, etc.)"

  - name: rate_limiting
    type: string
    required: false
    description: "Rate limiting configuration"

reasoning:
  default_strategy: "verify"
  max_iterations: 3
  quality_threshold: 0.95  # High threshold for security
```

Usage:
```python
result = await execute_skill(
    "api-security-auditor",
    {
        "endpoint_spec": """
        POST /api/users/{id}/update
        Body: { "email": "...", "role": "..." }
        """,
        "authentication_method": "JWT Bearer",
        "authorization_model": "RBAC"
    },
    config=Config.fused()  # Use FUSED for security audits
)

print(result.output)  # Comprehensive security audit
```

### Example 3: Code Generation Skill

`function-generator.yaml`:
```yaml
name: "function-generator"
version: "1.0.0"
description: "Generate functions from natural language descriptions"

metadata:
  category: "development"
  author: "Tutorial"
  tags: ["codegen", "functions"]

system_prompt: |
  You are a code generation expert.
  Generate clean, well-documented functions from natural language descriptions.

  Follow these principles:
  - Write idiomatic code for the target language
  - Include docstrings/comments
  - Add type hints (if language supports them)
  - Handle edge cases
  - Follow language-specific naming conventions

user_prompt_template: |
  Generate a function in {language} that:
  {description}

  {% if include_tests %}
  Also generate unit tests for this function.
  {% endif %}

  Requirements:
  - Function should be production-ready
  - Include comprehensive docstring
  - Handle edge cases appropriately
  - Follow {language} best practices

parameters:
  - name: description
    type: string
    required: true
    description: "Natural language description of what function should do"

  - name: language
    type: string
    required: true
    description: "Programming language (python, javascript, go, etc.)"

  - name: include_tests
    type: boolean
    required: false
    default: false
    description: "Whether to generate unit tests"

reasoning:
  default_strategy: "elegance"
  max_iterations: 3
  quality_threshold: 0.85
```

---

## Troubleshooting

### Issue: Skill not loading

**Symptom**: `ValueError: Skill 'my-skill' not found`

**Causes**:
1. YAML file not in `HoloLoom/agentic/skills/`
2. Skill name in YAML doesn't match filename
3. YAML syntax error

**Fix**:
```bash
# Check file exists
ls HoloLoom/agentic/skills/my-skill.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('HoloLoom/agentic/skills/my-skill.yaml'))"

# Check skill name matches
grep "^name:" HoloLoom/agentic/skills/my-skill.yaml
# Should output: name: "my-skill"
```

### Issue: Parameter validation fails

**Symptom**: `ValueError: Missing required parameter: X`

**Causes**:
1. Not providing required parameter
2. Parameter name typo
3. Parameter marked as required but should be optional

**Fix**:
```python
# Check required parameters in YAML
cat HoloLoom/agentic/skills/my-skill.yaml | grep -A 3 "required: true"

# Provide all required parameters
result = await execute_skill(
    "my-skill",
    {
        "required_param1": "value1",
        "required_param2": "value2"
    }
)
```

### Issue: Low confidence scores

**Symptom**: `result.confidence` consistently <0.7

**Causes**:
1. Ambiguous system prompt
2. Vague user prompt template
3. Quality threshold too high
4. Not enough refinement iterations

**Fix**:
```yaml
# Make prompts more specific
system_prompt: |
  You are a [specific expertise] expert.  # ✅ Not just "helpful assistant"

# Request structured output
user_prompt_template: |
  Provide answer in this exact format:
  1. [Section 1]
  2. [Section 2]

# Adjust reasoning config
reasoning:
  max_iterations: 5       # Allow more refinement
  quality_threshold: 0.75 # Lower threshold if appropriate
```

### Issue: Slow execution

**Symptom**: Skill takes >1 second to execute

**Causes**:
1. Too many refinement iterations
2. Using FUSED mode when FAST would suffice
3. Large parameter values

**Fix**:
```python
# Use BARE mode for simple tasks
result = await execute_skill(
    "my-skill",
    params,
    config=Config.bare()  # <50ms typical
)

# Reduce iterations in YAML
reasoning:
  max_iterations: 1  # Single pass
```

```yaml
# Or disable refinement entirely
reasoning:
  default_strategy: "refine"
  max_iterations: 1
  quality_threshold: 1.0  # Never refine (impossible threshold)
```

### Issue: Template rendering errors

**Symptom**: `{parameter}` appears in output literally

**Causes**:
1. Parameter name typo in template
2. Parameter not provided
3. Conditional logic error

**Fix**:
```yaml
# Use correct parameter names
user_prompt_template: |
  Code: {code}  # ✅ Must match parameter name exactly

# Handle missing optional parameters
user_prompt_template: |
  {% if optional_param %}
  Optional: {optional_param}
  {% else %}
  Optional parameter not provided.
  {% endif %}
```

---

## Next Steps

Now that you know how to create custom skills:

1. **Practice**: Create a simple skill following the tutorial
2. **Experiment**: Try different reasoning strategies
3. **Test**: Add integration tests for your skills
4. **Share**: Contribute skills to the HoloLoom community
5. **Read**: Explore existing skills in `HoloLoom/agentic/skills/` for inspiration

**Resources**:
- `skills/README.md` - Complete API reference
- `skills/code-reviewer.yaml` - Comprehensive example
- `demos/demo_skills.py` - Usage examples
- `HoloLoom/recursive/` - Reasoning strategies implementation

---

## Appendix: Complete Skill Template

Copy this template to start creating new skills:

```yaml
name: "your-skill-name"
version: "1.0.0"
description: "One-line description of what this skill does"

metadata:
  category: "general"  # architecture|development|education|optimization|security|database|general
  author: "Your Name"
  tags: ["tag1", "tag2", "tag3"]
  created: "2025-11-22"

system_prompt: |
  You are an expert in [domain].

  Your expertise includes:
  - [Specific knowledge area 1]
  - [Specific knowledge area 2]
  - [Specific knowledge area 3]

  Your role is to:
  1. [Primary responsibility]
  2. [Secondary responsibility]
  3. [Tertiary responsibility]

  Always [tone/style guidance].

user_prompt_template: |
  [Context setup with parameters]

  {required_parameter1}: {required_parameter1}

  {% if optional_parameter %}
  {optional_parameter}: {optional_parameter}
  {% endif %}

  Please provide:
  1. [Expected output section 1]
  2. [Expected output section 2]
  3. [Expected output section 3]

  Format: [Specify output format]

parameters:
  - name: required_parameter1
    type: string
    required: true
    description: "Clear description of what this parameter is"

  - name: optional_parameter
    type: string
    required: false
    default: "default_value"
    description: "Clear description of optional parameter"

reasoning:
  default_strategy: "refine"  # refine|critique|verify|elegance|hofstadter
  max_iterations: 3           # 1-5 typical
  quality_threshold: 0.85     # 0.70-0.95 typical
```

---

**Happy Skill Building!**

*For questions or contributions, see HoloLoom/agentic/skills/README.md*

**Document Version**: 1.0.0 (November 2025)
