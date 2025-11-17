# Promptly Template System - Complete Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Template Authoring](#template-authoring)
5. [Built-in Templates](#built-in-templates)
6. [Composition & Inheritance](#composition--inheritance)
7. [Validation & Testing](#validation--testing)
8. [CLI Reference](#cli-reference)
9. [Best Practices](#best-practices)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

The Promptly Template System is a powerful framework for creating, managing, and reusing prompt templates with:

- **Jinja2-based templating** with custom filters and functions
- **Template inheritance** and composition
- **Versioning** and registry management
- **Validation** with JSON Schema
- **Testing framework** for quality assurance
- **Built-in library** of 20+ professional templates

### Why Use Templates?

- **Consistency**: Standardize prompt patterns across your organization
- **Reusability**: Write once, use everywhere
- **Maintainability**: Update prompts in one place
- **Quality**: Validate and test templates before deployment
- **Collaboration**: Share and version templates with your team

---

## Quick Start

### Initialize Promptly

```bash
# Initialize a Promptly repository
promptly init

# Template system is automatically initialized
# 20+ built-in templates are loaded
```

### List Available Templates

```bash
# List all templates
promptly template list

# Filter by category
promptly template list --category domain

# Search templates
promptly template list --search summarize
```

### Use a Template

```bash
# Render a template with variables
promptly template render summarize-text --vars '{"text": "Your text here", "max_sentences": 5}'

# Use a variables file
promptly template render code-generation --vars-file vars.yaml
```

### Create Your Own Template

```bash
# Interactive creation
promptly template create my-template --interactive

# From a file
promptly template create my-template --file template.txt --category general

# Extend an existing template
promptly template create my-custom --from summarize-text
```

---

## Core Concepts

### Templates

A **template** is a reusable prompt pattern with:
- **Name**: Unique identifier
- **Content**: Jinja2-formatted text
- **Variables**: Placeholders for dynamic content
- **Metadata**: Category, tags, description
- **Versions**: Historical snapshots

### Template Variables

Variables are placeholders in templates:

```jinja2
Summarize the following {{ content_type }}:

{{ content }}
```

Usage:
```python
{
  "content_type": "article",
  "content": "The article text..."
}
```

### Template Categories

Templates are organized by category:

- **base**: Foundational templates
- **role**: System/user/assistant roles
- **domain**: Domain-specific (coding, QA, summarization)
- **pattern**: Advanced patterns (few-shot, CoT, ReAct)
- **general**: Custom templates

### Template Composition

Combine templates using:

- **Mixins**: Reusable components (tone, format)
- **Fragments**: Small template pieces
- **Inheritance**: Extend base templates

---

## Template Authoring

### Basic Template Syntax

Templates use Jinja2 syntax:

```jinja2
{# This is a comment #}

{{ variable }}  {# Variable substitution #}

{% if condition %}  {# Conditional #}
  Text if true
{% endif %}

{% for item in items %}  {# Loop #}
  - {{ item }}
{% endfor %}
```

### Custom Filters

Promptly provides custom filters:

```jinja2
{# String filters #}
{{ text | snippet(100) }}  {# Truncate to 100 chars #}
{{ text | slugify }}  {# Convert to slug #}

{# List filters #}
{{ items | bullet_list }}  {# Convert to bullet points #}
{{ items | numbered_list }}  {# Convert to numbered list #}

{# Code filters #}
{{ code | code_block('python') }}  {# Wrap in code block #}

{# JSON/YAML filters #}
{{ data | to_json }}
{{ data | to_yaml }}
```

### Custom Functions

Built-in global functions:

```jinja2
{# Date/time #}
{{ now() }}  {# Current timestamp #}
{{ today() }}  {# Today's date #}

{# Role messages #}
{{ system_message("You are a helpful assistant") }}
{{ user_message(question) }}
{{ assistant_message(response) }}

{# Patterns #}
{{ cot_prompt() }}  {# "Let's think step by step:" #}
{{ react_prompt(task) }}  {# ReAct pattern #}
```

### Template Example

```jinja2
{# File: templates/my-summarizer.yaml #}
name: my-summarizer
category: domain
description: Custom summarization template
tags:
  - summarization
  - custom

content: |
  Summarize the following {{ content_type | default('text', true) }}:

  {{ content }}

  {% if focus -%}
  Focus on: {{ focus }}
  {% endif -%}

  Provide a {{ length | default('concise', true) }} summary
  {% if audience -%}for {{ audience }}{% endif %}.

defaults:
  content_type: "text"
  length: "concise"
```

### Template Definition Format

Templates are defined in YAML:

```yaml
name: template-name          # Required: unique identifier
category: domain             # Required: category
description: What it does    # Recommended
tags:                        # Optional: for discovery
  - tag1
  - tag2

content: |                   # Required: template content
  Your Jinja2 template here

defaults:                    # Optional: default variable values
  variable1: value1
  variable2: value2

parameters:                  # Optional: parameter documentation
  - name: variable1
    type: string
    description: What it's for
    required: true
  - name: variable2
    type: number
    description: Another parameter
    default: 42

examples:                    # Optional: usage examples
  - description: Example 1
    variables:
      variable1: "test"
      variable2: 100
    expected_output: "Expected result..."
```

---

## Built-in Templates

Promptly includes 20+ professional templates:

### Base Templates

- **simple**: Simple variable substitution
- **instruction**: Basic instruction template
- **conversation**: Multi-turn conversation
- **input-output**: Simple I/O format

### Role Templates

**System Roles:**
- **system-assistant**: General AI assistant
- **system-expert**: Domain expert
- **system-teacher**: Educational persona

**User Roles:**
- **user-question**: User question format
- **user-task**: Task request format

### Domain Templates

**Summarization:**
- **summarize-text**: General text summarization
- **summarize-bullet-points**: Bullet point summaries
- **summarize-meeting**: Meeting notes summarization

**Question-Answering:**
- **qa-basic**: Basic QA
- **qa-with-sources**: QA with citations
- **qa-multiple-choice**: Multiple choice questions

**Coding:**
- **code-generation**: Generate code
- **code-review**: Review code
- **code-explanation**: Explain code
- **debug-code**: Debug errors

### Pattern Templates

**Few-Shot Learning:**
- **few-shot-basic**: Basic few-shot
- **few-shot-classification**: Classification tasks
- **few-shot-extraction**: Information extraction

**Chain-of-Thought:**
- **cot-basic**: Basic CoT
- **cot-with-examples**: CoT with examples
- **cot-math**: Math problem solving
- **cot-analysis**: Analysis tasks

**ReAct:**
- **react-basic**: Basic ReAct pattern
- **react-research**: Research tasks
- **react-problem-solving**: Complex problems

---

## Composition & Inheritance

### Using Mixins

Mixins add reusable functionality:

```python
from Promptly.promptly.templates import TemplateComposer

composer = TemplateComposer()

# Load mixins
composer.load_mixins_from_file('library/mixins.yaml')

# Compose template with mixins
base_template = "{{ question }}"
mixins = ['tone-professional', 'format-markdown', 'think-step-by-step']

composed = composer.compose(
    base_template,
    mixins=mixins,
    context={'domain': 'technical'}
)
```

### Available Mixins

**Tone:**
- `tone-professional`: Professional tone
- `tone-casual`: Casual/friendly tone
- `tone-technical`: Technical language

**Format:**
- `format-markdown`: Markdown output
- `format-json`: JSON output
- `format-bullet-points`: Bullet points

**Length:**
- `length-concise`: Concise responses
- `length-detailed`: Detailed responses

**Behavior:**
- `think-step-by-step`: Encourage reasoning
- `verify-sources`: Source verification
- `examples-required`: Include examples
- `avoid-jargon`: Explain technical terms

### Using Fragments

Fragments are reusable template pieces:

```jinja2
{# Include a fragment #}
{% fragment "disclaimer" %}

{# Or use Jinja2 include #}
{% include 'fragments/thinking-prompt.j2' %}
```

### Template Inheritance

Extend base templates:

```yaml
name: my-advanced-summarizer
base_template: summarize-text

content: |
  {% extends 'summarize-text' %}

  {% block additional_instructions %}
  Also identify key themes and sentiment.
  {% endblock %}
```

---

## Validation & Testing

### Validate Templates

```bash
# Validate a template
promptly template validate my-template

# Strict validation
promptly template validate my-template --strict
```

### Validation Checks

The validator performs:

1. **JSON Schema validation**: Structure compliance
2. **Syntax validation**: Jinja2 syntax errors
3. **Variable validation**: Missing defaults, unused variables
4. **Security checks**: Dangerous patterns (eval, exec, imports)
5. **Best practices**: Naming, length, documentation

### Template Testing

Create test suites for templates:

```yaml
# test-summarizer.yaml
name: Summarization Tests
template: summarize-text
description: Test suite for summarization template

tests:
  - name: Basic summarization
    variables:
      text: "This is a long article about AI..."
      max_sentences: 3
    expected_contains:
      - "AI"
    expected_length_max: 500

  - name: With focus
    variables:
      text: "Article text..."
      focus: "applications"
    expected_contains:
      - "applications"
```

Run tests programmatically:

```python
from Promptly.promptly.templates import TemplateEngine, TemplateTestRunner
from Promptly.promptly.templates.testing import load_test_suite_from_yaml

# Load template and tests
engine = TemplateEngine()
suite = load_test_suite_from_yaml('test-summarizer.yaml')

# Get template content
registry = TemplateRegistry()
template = registry.get('summarize-text')

# Run tests
runner = TemplateTestRunner(engine)
result = runner.run_suite(template.content, suite)

print(result)
```

---

## CLI Reference

### Template Management

```bash
# Create template
promptly template create NAME [OPTIONS]
  --from BASE              # Extend from base template
  --category CATEGORY      # Set category
  --description DESC       # Set description
  --content CONTENT        # Template content
  --file FILE              # Load from file
  --interactive            # Interactive mode

# List templates
promptly template list [OPTIONS]
  --category CATEGORY      # Filter by category
  --tag TAG                # Filter by tag(s)
  --search QUERY           # Search query

# Show template
promptly template show NAME [OPTIONS]
  --version VERSION        # Specific version

# Render template
promptly template render NAME [OPTIONS]
  --vars JSON              # Variables as JSON
  --vars-file FILE         # Variables from file
  --output FILE            # Output file

# Validate template
promptly template validate NAME [OPTIONS]
  --strict                 # Strict validation

# Delete template
promptly template delete NAME [OPTIONS]
  --yes                    # Skip confirmation

# Export template
promptly template export NAME OUTPUT_FILE [OPTIONS]
  --version VERSION        # Specific version

# Import template
promptly template import INPUT_FILE

# List categories
promptly template categories

# Discover templates
promptly template discover DIRECTORY [OPTIONS]
  --category CATEGORY      # Set category
```

---

## Best Practices

### Template Design

1. **Use Clear Names**: `summarize-meeting` not `sm1`
2. **Provide Descriptions**: Help users understand purpose
3. **Set Defaults**: Make templates usable without all variables
4. **Add Examples**: Include usage examples in metadata
5. **Use Categories**: Organize templates logically
6. **Tag Appropriately**: Enable discovery

### Variable Naming

```jinja2
{# Good #}
{{ user_question }}
{{ max_length }}
{{ output_format }}

{# Avoid #}
{{ q }}
{{ ml }}
{{ fmt }}
```

### Template Structure

```jinja2
{# 1. System instructions #}
You are an expert in {{ domain }}.

{# 2. Task description #}
{{ task_description }}

{# 3. Context/input #}
{{ context }}

{# 4. Constraints/requirements #}
{% if constraints -%}
Requirements:
{{ constraints | bullet_list }}
{% endif -%}

{# 5. Output format #}
Provide your response as {{ output_format }}.
```

### Composition Over Duplication

Instead of duplicating:

```jinja2
{# BAD: Duplicated content #}
Template 1: "Be professional. {{ task }}"
Template 2: "Be professional. {{ other_task }}"
```

Use mixins:

```jinja2
{# GOOD: Reusable mixin #}
Mixin: tone-professional
Template 1: {{ task }}
Template 2: {{ other_task }}
```

### Testing

1. **Test Variable Combinations**: Test with/without optional vars
2. **Test Edge Cases**: Empty strings, long text, special characters
3. **Test Output Quality**: Verify expected content appears
4. **Regression Testing**: Test after changes

---

## Advanced Topics

### Custom Filters

Add custom filters to the engine:

```python
from Promptly.promptly.templates import TemplateEngine

engine = TemplateEngine()

# Add custom filter
def reverse_text(text):
    return text[::-1]

engine.add_filter('reverse', reverse_text)

# Use in template
template = "{{ text | reverse }}"
output = engine.render_string(template, text="hello")
# Output: "olleh"
```

### Custom Functions

Add custom global functions:

```python
def custom_greeting(name, time_of_day):
    return f"Good {time_of_day}, {name}!"

engine.add_function('greet', custom_greeting)

# Use in template
template = "{{ greet(user_name, 'morning') }}"
```

### Custom Validators

Add custom validation rules:

```python
from Promptly.promptly.templates import TemplateValidator

validator = TemplateValidator()

def check_word_count(template_data, result):
    content = template_data.get('content', '')
    word_count = len(content.split())

    if word_count > 1000:
        result.add_warning(
            f"Template is very long ({word_count} words)",
            suggestion="Consider breaking into smaller templates"
        )

validator.add_custom_validator(check_word_count)
```

### Programmatic Template Creation

```python
from Promptly.promptly.templates import TemplateRegistry

registry = TemplateRegistry('registry.yaml')

# Register template
registry.register(
    name='my-template',
    content='{{ question }}',
    category='general',
    description='A simple template',
    tags=['simple', 'qa']
)

# Save registry
registry.save()
```

### Template Discovery

```python
from Promptly.promptly.templates import TemplateRegistry, TemplateDiscovery

registry = TemplateRegistry('registry.yaml')
discovery = TemplateDiscovery(registry)

# Discover from directory
count = discovery.discover_from_directory(
    'my-templates',
    category='custom',
    recursive=True
)

print(f"Discovered {count} templates")
```

### Batch Operations

```python
# Validate multiple templates
validator = TemplateValidator()
templates = [template1_data, template2_data, template3_data]

results = validator.batch_validate(templates, strict=True)

for name, result in results.items():
    if not result.valid:
        print(f"{name}: FAILED")
        for error in result.get_errors():
            print(f"  - {error}")
```

---

## Examples

### Example 1: Code Review Template

```yaml
name: code-review-detailed
category: domain
description: Comprehensive code review template

content: |
  Review the following {{ language }} code:

  {{ code | code_block(language) }}

  {% if context -%}
  Context: {{ context }}
  {% endif -%}

  Provide a comprehensive review covering:

  1. **Code Quality**
     - Readability and clarity
     - Naming conventions
     - Code organization

  2. **Functionality**
     - Logic correctness
     - Edge case handling
     - Error handling

  3. **Performance**
     - Time complexity
     - Space complexity
     - Optimization opportunities

  4. **Security**
     - Potential vulnerabilities
     - Input validation
     - Security best practices

  5. **Best Practices**
     - Design patterns
     - {{ language }}-specific idioms
     - Testing considerations

  {% if severity_filter -%}
  Focus on {{ severity_filter }} severity issues.
  {% endif -%}

defaults:
  language: "Python"
  severity_filter: null

tags:
  - coding
  - review
  - quality
```

### Example 2: Research Summary Template

```yaml
name: research-summary
category: domain
description: Academic research summarization

content: |
  Summarize the following research paper:

  **Title**: {{ title }}
  {% if authors -%}
  **Authors**: {{ authors | join_with }}
  {% endif -%}
  {% if year -%}
  **Year**: {{ year }}
  {% endif -%}

  **Abstract/Content**:
  {{ content }}

  Provide a structured summary:

  ## Research Question
  [Main research question or hypothesis]

  ## Methodology
  [Research methods and approach]

  ## Key Findings
  {{ findings_count | default(3, true) }} most important findings

  ## Implications
  [Theoretical and practical implications]

  ## Limitations
  [Study limitations and future work]

  {% if target_audience -%}
  Write for: {{ target_audience }}
  {% endif -%}

defaults:
  findings_count: 3
  target_audience: "general academic audience"

tags:
  - research
  - academic
  - summarization
```

---

## Troubleshooting

### Common Issues

**Issue**: Template not found
```bash
# Check template exists
promptly template list --search template-name

# Check category
promptly template list --category domain
```

**Issue**: Variables not substituting
```jinja2
{# Wrong #}
{ variable }

{# Correct #}
{{ variable }}
```

**Issue**: Validation errors
```bash
# Run validation with details
promptly template validate my-template

# Check syntax in isolation
echo "{{ test }}" | promptly template render simple --vars '{"text":"test"}'
```

**Issue**: Template too slow
```python
# Pre-compile templates
engine = TemplateEngine(cache_size=1000)
compiled = engine.compile_template(template_content)

# Reuse compiled template
output = compiled.render(variables)
```

---

## Resources

- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [JSON Schema](https://json-schema.org/)
- [Promptly Repository](https://github.com/your-org/promptly)

---

## Contributing

To contribute templates to the built-in library:

1. Create template following best practices
2. Add comprehensive tests
3. Validate with strict mode
4. Submit PR with:
   - Template definition
   - Test suite
   - Usage examples
   - Documentation

---

**Last Updated**: 2024-01-17
**Version**: 1.0.0
