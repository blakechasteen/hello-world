# Promptly Template System - Quick Reference

## CLI Commands

```bash
# Template Management
promptly template create NAME              # Create template
promptly template list                     # List all templates
promptly template show NAME                # Show template details
promptly template render NAME              # Render template
promptly template validate NAME            # Validate template
promptly template delete NAME              # Delete template
promptly template categories               # List categories
promptly template discover DIR             # Discover templates

# Options
--from BASE                   # Extend base template
--category CAT                # Set/filter category
--tag TAG                     # Filter by tag
--vars JSON                   # Variables as JSON
--vars-file FILE              # Variables from file
--output FILE                 # Output to file
--strict                      # Strict validation
--interactive                 # Interactive mode
```

## Template Syntax

```jinja2
{# Comments #}

{{ variable }}                           # Variable substitution
{{ var | default('fallback') }}          # Default value
{{ var | default('fallback', true) }}    # Always use default if falsy

{% if condition %}...{% endif %}         # Conditional
{% for item in items %}...{% endfor %}   # Loop
{% set var = value %}                    # Set variable

{# Whitespace control #}
{%- if x -%}...{%- endif -%}            # Strip whitespace
```

## Custom Filters

```jinja2
{{ text | snippet(100) }}                # Truncate to N chars
{{ text | titlecase }}                   # Title case
{{ text | slugify }}                     # Convert to slug

{{ items | bullet_list }}                # Bullet points
{{ items | numbered_list }}              # Numbered list
{{ items | join_with(', ', ' and ') }}  # Smart join

{{ code | code_block('python') }}        # Code block
{{ text | xml_tag('div') }}             # Wrap in XML tag

{{ examples | few_shot }}                # Format few-shot examples
{{ data | to_json }}                     # Convert to JSON
{{ data | to_yaml }}                     # Convert to YAML
{{ num | percentage }}                   # Format as percentage
```

## Custom Functions

```jinja2
{{ now() }}                              # Current timestamp
{{ today() }}                            # Today's date

{{ system_message('content') }}          # System role message
{{ user_message('content') }}            # User role message
{{ assistant_message('content') }}       # Assistant role message

{{ cot_prompt() }}                       # Chain-of-thought prompt
{{ react_prompt(task) }}                 # ReAct pattern
```

## Python API

### Quick Start

```python
from Promptly.promptly.templates import TemplateEngine

engine = TemplateEngine()
output = engine.render_string("{{ greeting }}, {{ name }}!",
                              greeting="Hello", name="World")
```

### Template Engine

```python
# Create engine
engine = TemplateEngine(template_dirs=['/path/to/templates'])

# Render
output = engine.render_string(template, **vars)
output = engine.render_file('template.j2', **vars)

# Custom filters/functions
engine.add_filter('name', function)
engine.add_function('name', function)

# List templates
templates = engine.list_templates()
```

### Template Registry

```python
from Promptly.promptly.templates import TemplateRegistry

registry = TemplateRegistry('registry.yaml')

# Register
registry.register(name='x', content='...', category='general')

# Get
template = registry.get('name', version=1)
entry = registry.get_entry('name')

# List
templates = registry.list_templates(category='domain', tags=['coding'])

# Save/Load
registry.save()
registry.load()
```

### Template Composer

```python
from Promptly.promptly.templates import TemplateComposer

composer = TemplateComposer()

# Load components
composer.load_mixins_from_file('mixins.yaml')
composer.load_fragments_from_file('fragments.yaml')

# Compose
composed = composer.compose(
    base_template='{{ task }}',
    mixins=['tone-professional', 'format-markdown']
)
```

### Template Validator

```python
from Promptly.promptly.templates import TemplateValidator

validator = TemplateValidator()

result = validator.validate(template_data, strict=True)

if result.valid:
    print("Valid")
else:
    for error in result.get_errors():
        print(error)
```

### Template Testing

```python
from Promptly.promptly.templates.testing import (
    TestCase, TestSuite, TemplateTestRunner
)

# Create suite
suite = TestSuite(name='Tests', template_name='my-template')

# Add tests
suite.add_test(TestCase(
    name='Test 1',
    variables={'var': 'value'},
    expected_contains=['text'],
    expected_length_max=100
))

# Run tests
runner = TemplateTestRunner(engine)
result = runner.run_suite(template_content, suite)
```

## Template Definition

```yaml
name: template-name               # Required
category: general                 # Required
description: What it does         # Recommended
tags: [tag1, tag2]               # Optional

content: |                        # Required
  Template content with {{ variables }}

defaults:                         # Optional
  variable: "default value"

parameters:                       # Optional
  - name: variable
    type: string
    description: What it does
    required: true
    default: value

examples:                         # Optional
  - description: Example usage
    variables:
      variable: "test"
    expected_output: "result"
```

## Built-in Templates

### Base (4)
- `simple`, `instruction`, `conversation`, `input-output`

### Roles (5)
- `system-assistant`, `system-expert`, `system-teacher`
- `user-question`, `user-task`

### Summarization (3)
- `summarize-text`, `summarize-bullet-points`, `summarize-meeting`

### QA (3)
- `qa-basic`, `qa-with-sources`, `qa-multiple-choice`

### Coding (4)
- `code-generation`, `code-review`, `code-explanation`, `debug-code`

### Few-Shot (3)
- `few-shot-basic`, `few-shot-classification`, `few-shot-extraction`

### Chain-of-Thought (4)
- `cot-basic`, `cot-with-examples`, `cot-math`, `cot-analysis`

### ReAct (3)
- `react-basic`, `react-research`, `react-problem-solving`

## Mixins

### Tone
`tone-professional`, `tone-casual`, `tone-technical`

### Format
`format-markdown`, `format-json`, `format-bullet-points`

### Length
`length-concise`, `length-detailed`

### Behavior
`think-step-by-step`, `verify-sources`, `examples-required`,
`avoid-jargon`, `creative-thinking`, `safety-check`

## Fragments

`disclaimer`, `citation-format`, `code-best-practices`,
`thinking-prompt`, `output-structure`, `confidence-level`,
`alternatives`, `pros-cons`, `step-by-step`, `qa-format`,
`context-reminder`, `clarification-request`

## Common Patterns

### Template with Defaults

```jinja2
{{ task }}

{% if requirements -%}
Requirements:
{{ requirements | bullet_list }}
{% endif -%}

Output format: {{ format | default('text', true) }}
```

### Few-Shot Pattern

```jinja2
Examples:
{{ examples | few_shot }}

Now complete:
Input: {{ input }}
Output:
```

### Chain-of-Thought

```jinja2
{{ question }}

{{ cot_prompt() }}

1.
2.
3.

Answer:
```

### Role-Based

```jinja2
{% for message in messages -%}
{{ message.role | upper }}: {{ message.content }}

{% endfor -%}
```

## Test Assertions

```python
TestCase(
    name='test',
    variables={'var': 'val'},

    # Assertions
    expected_output='exact match',
    expected_contains=['text1', 'text2'],
    expected_not_contains=['bad'],
    expected_pattern=r'regex.*pattern',
    expected_length_min=10,
    expected_length_max=100,
    custom_assertion=lambda x: len(x) > 0
)
```

## Validation Checks

- ✅ JSON Schema structure
- ✅ Jinja2 syntax errors
- ✅ Variable usage (missing defaults, unused vars)
- ✅ Security (eval, exec, imports, XSS)
- ✅ Best practices (naming, length, documentation)
- ✅ Custom validators

## Tips & Tricks

### Whitespace Control

```jinja2
{%- if x -%}      # Strip before and after
  text
{%- endif %}

{{ x -}}          # Strip after
{{- x }}          # Strip before
```

### Variable Chaining

```jinja2
{{ user.name | default('Anonymous') | titlecase }}
```

### Complex Loops

```jinja2
{% for item in items -%}
{{ loop.index }}. {{ item }}
{% if loop.last %}
(last item)
{% endif -%}
{% endfor %}
```

### Macros

```jinja2
{% macro format_code(code, lang) -%}
```{{ lang }}
{{ code }}
```
{%- endmacro %}

{{ format_code(my_code, 'python') }}
```

### Template Inheritance

```jinja2
{# base.j2 #}
Header
{% block content %}Default content{% endblock %}
Footer

{# child.j2 #}
{% extends 'base.j2' %}
{% block content %}Custom content{% endblock %}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Template not found | Check `promptly template list` |
| Variable not substituting | Use `{{ var }}` not `{ var }` |
| Validation fails | Run `promptly template validate NAME` |
| Syntax error | Check Jinja2 syntax, use `{# comments #}` |
| Slow rendering | Pre-compile with `engine.compile_template()` |

## Resources

- Full Guide: [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md)
- Jinja2 Docs: https://jinja.palletsprojects.com/
- JSON Schema: https://json-schema.org/

---

**Version**: 1.0.0
**Last Updated**: 2024-01-17
