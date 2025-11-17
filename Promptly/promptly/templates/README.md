# Promptly Template System

A powerful, production-ready template system for managing and reusing prompts with inheritance, composition, validation, and testing.

## Features

### Core Features
- ✅ **Jinja2-based templating** - Full Jinja2 support with custom filters and functions
- ✅ **Template inheritance** - Extend base templates and reuse patterns
- ✅ **Composition system** - Mix-ins and fragments for modular design
- ✅ **Versioning** - Track template history with full version control
- ✅ **Registry management** - Centralized template discovery and organization
- ✅ **Validation** - JSON Schema validation with security checks
- ✅ **Testing framework** - Comprehensive testing with assertions
- ✅ **Built-in library** - 20+ professional templates ready to use

### Custom Extensions
- 🎨 **Custom Filters**: `snippet`, `bullet_list`, `code_block`, `few_shot`, and more
- 🛠️ **Custom Functions**: `now()`, `system_message()`, `cot_prompt()`, `react_prompt()`
- 🔍 **Custom Validators**: Extensible validation with custom rules
- 🧪 **Test Runners**: Automated testing with multiple assertion types

## Quick Start

```python
from Promptly.promptly.templates import TemplateEngine, TemplateContext

# Create engine
engine = TemplateEngine()

# Render a simple template
template = "Summarize: {{ text }}"
output = engine.render_string(template, text="Your text here")

# Use context with defaults
context = TemplateContext(
    variables={'text': 'Article...'},
    defaults={'max_words': 100}
)
output = engine.render_string(template, context)
```

## Built-in Templates

### Categories

#### Base Templates (4)
- `simple` - Simple variable substitution
- `instruction` - Basic instruction format
- `conversation` - Multi-turn conversations
- `input-output` - I/O format

#### Role Templates (5)
- `system-assistant` - AI assistant persona
- `system-expert` - Domain expert persona
- `system-teacher` - Teaching persona
- `user-question` - User question format
- `user-task` - Task request format

#### Domain Templates (10)
- **Summarization**: `summarize-text`, `summarize-bullet-points`, `summarize-meeting`
- **QA**: `qa-basic`, `qa-with-sources`, `qa-multiple-choice`
- **Coding**: `code-generation`, `code-review`, `code-explanation`, `debug-code`

#### Pattern Templates (9)
- **Few-Shot**: `few-shot-basic`, `few-shot-classification`, `few-shot-extraction`
- **Chain-of-Thought**: `cot-basic`, `cot-with-examples`, `cot-math`, `cot-analysis`
- **ReAct**: `react-basic`, `react-research`, `react-problem-solving`

#### Composition (26 mixins + 12 fragments)
- **Tone Mixins**: professional, casual, technical
- **Format Mixins**: markdown, json, bullet-points
- **Behavior Mixins**: think-step-by-step, verify-sources, examples-required

## CLI Usage

```bash
# List templates
promptly template list

# Create a template
promptly template create my-template --interactive

# Render a template
promptly template render summarize-text --vars '{"text": "..."}'

# Validate a template
promptly template validate my-template

# Show template details
promptly template show code-review
```

## Python API

### Template Engine

```python
from Promptly.promptly.templates import TemplateEngine

engine = TemplateEngine(template_dirs=['/path/to/templates'])

# Render from string
output = engine.render_string("{{ greeting }}, {{ name }}!",
                              greeting="Hello", name="World")

# Render from file
output = engine.render_file('my-template.j2', context={'var': 'value'})

# Add custom filter
engine.add_filter('custom', lambda x: x.upper())

# Add custom function
engine.add_function('timestamp', lambda: datetime.now().isoformat())
```

### Template Registry

```python
from Promptly.promptly.templates import TemplateRegistry, TemplateDiscovery

# Create registry
registry = TemplateRegistry('registry.yaml')

# Register template
registry.register(
    name='my-template',
    content='{{ question }}',
    category='general',
    description='A simple template',
    tags=['simple']
)

# Get template
template = registry.get('my-template')
print(template.content)

# List templates
templates = registry.list_templates(category='domain', tags=['coding'])

# Discover templates
discovery = TemplateDiscovery(registry)
count = discovery.discover_from_directory('templates/')
```

### Template Composition

```python
from Promptly.promptly.templates import TemplateComposer, TemplateMixin

composer = TemplateComposer()

# Register mixin
mixin = TemplateMixin(
    name='tone-professional',
    content='Use a professional tone.',
    inject_position='start'
)
composer.register_mixin(mixin)

# Compose template
composed = composer.compose(
    base_template='{{ task }}',
    mixins=['tone-professional', 'format-markdown']
)
```

### Template Validation

```python
from Promptly.promptly.templates import TemplateValidator

validator = TemplateValidator()

template_data = {
    'name': 'my-template',
    'content': '{{ question }}',
    'category': 'general'
}

result = validator.validate(template_data, strict=True)

if result.valid:
    print("✓ Validation passed")
else:
    for error in result.get_errors():
        print(f"✗ {error}")
```

### Template Testing

```python
from Promptly.promptly.templates import TemplateTestRunner
from Promptly.promptly.templates.testing import TestCase, TestSuite

# Create test suite
suite = TestSuite(name='My Tests', template_name='my-template')

# Add test cases
suite.add_test(TestCase(
    name='Basic test',
    variables={'question': 'What is AI?'},
    expected_contains=['AI', 'artificial intelligence'],
    expected_length_max=500
))

# Run tests
runner = TemplateTestRunner(engine)
result = runner.run_suite(template_content, suite)

print(f"Passed: {result.passed_tests}/{result.total_tests}")
```

## Template Syntax

### Variables

```jinja2
{{ variable }}
{{ variable | default('fallback') }}
{{ variable | default('fallback', true) }}  {# true = always use default if falsy #}
```

### Conditionals

```jinja2
{% if condition %}
  Text if true
{% elif other_condition %}
  Text if other
{% else %}
  Text if false
{% endif %}
```

### Loops

```jinja2
{% for item in items %}
  {{ loop.index }}. {{ item }}
{% endfor %}
```

### Custom Filters

```jinja2
{{ text | snippet(100) }}           {# Truncate to 100 chars #}
{{ items | bullet_list }}           {# Convert to bullets #}
{{ items | numbered_list }}         {# Convert to numbered #}
{{ code | code_block('python') }}   {# Wrap in code block #}
{{ examples | few_shot }}           {# Format few-shot examples #}
{{ data | to_json }}                {# Convert to JSON #}
```

### Custom Functions

```jinja2
{{ now() }}                                    {# Current timestamp #}
{{ today() }}                                  {# Today's date #}
{{ system_message('You are helpful') }}        {# System role message #}
{{ user_message(question) }}                   {# User role message #}
{{ cot_prompt() }}                             {# "Let's think step by step" #}
{{ react_prompt(task) }}                       {# ReAct pattern #}
```

## Template Definition Format

```yaml
name: template-name
category: domain
description: What this template does
tags:
  - tag1
  - tag2

content: |
  Your Jinja2 template content here
  {{ variable }}

defaults:
  variable: "default value"

parameters:
  - name: variable
    type: string
    description: What this parameter does
    required: true

examples:
  - description: Example usage
    variables:
      variable: "test value"
    expected_output: "Expected result"
```

## Best Practices

### 1. Use Meaningful Names
```yaml
# Good
name: summarize-meeting-notes

# Bad
name: smn1
```

### 2. Provide Defaults
```jinja2
{{ max_length | default(100, true) }}
{{ output_format | default('text', true) }}
```

### 3. Document Parameters
```yaml
parameters:
  - name: text
    type: string
    description: The text to summarize
    required: true
  - name: max_sentences
    type: number
    description: Maximum sentences in summary
    default: 3
```

### 4. Add Examples
```yaml
examples:
  - description: Summarize an article
    variables:
      text: "Article content..."
      max_sentences: 3
    expected_output: "Summary..."
```

### 5. Validate Templates
```bash
promptly template validate my-template --strict
```

### 6. Test Templates
```python
# Create comprehensive test suites
suite = TestSuite(name='Tests', template_name='my-template')
suite.add_test(TestCase(...))
runner.run_suite(content, suite)
```

## Architecture

```
templates/
├── __init__.py           # Package exports
├── engine.py             # Jinja2 template engine
├── composition.py        # Mixins and fragments
├── registry.py           # Template versioning and storage
├── validator.py          # Validation with JSON Schema
├── testing.py            # Testing framework
├── library/              # Built-in templates
│   ├── __init__.py
│   ├── base.yaml
│   ├── roles/
│   ├── domains/
│   ├── patterns/
│   ├── mixins.yaml
│   └── fragments.yaml
└── schemas/
    └── template.schema.json
```

## Dependencies

Required:
- `jinja2` - Template engine
- `pyyaml` - YAML parsing

Optional:
- `jsonschema` - JSON Schema validation (recommended)

Install:
```bash
pip install jinja2 pyyaml jsonschema
```

## Examples

See [TEMPLATE_GUIDE.md](../../TEMPLATE_GUIDE.md) for comprehensive examples and tutorials.

## Contributing

Contributions welcome! Please:
1. Follow existing template patterns
2. Add tests for new templates
3. Validate with strict mode
4. Update documentation

## License

MIT License - see LICENSE file

## Support

- Documentation: [TEMPLATE_GUIDE.md](../../TEMPLATE_GUIDE.md)
- Issues: [GitHub Issues](https://github.com/your-org/promptly/issues)
- Discussions: [GitHub Discussions](https://github.com/your-org/promptly/discussions)
