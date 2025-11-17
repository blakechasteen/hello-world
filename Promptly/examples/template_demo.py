#!/usr/bin/env python3
"""
Template System Demo - Comprehensive demonstration of template features
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from promptly.templates import (
    TemplateEngine,
    TemplateContext,
    TemplateRegistry,
    TemplateDiscovery,
    TemplateValidator,
    TemplateComposer,
    TemplateMixin,
    TemplateFragment,
)
from promptly.templates.testing import (
    TestCase,
    TestSuite,
    TemplateTestRunner,
    TestSuiteBuilder
)
from promptly.templates.library import get_library_path


def demo_1_basic_rendering():
    """Demo 1: Basic template rendering"""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Template Rendering")
    print("=" * 60)

    engine = TemplateEngine()

    # Simple variable substitution
    template = "Hello, {{ name }}!"
    output = engine.render_string(template, name="World")
    print(f"\nSimple: {output}")

    # With conditionals
    template = """
    {% if greeting -%}
    {{ greeting }}, {{ name }}!
    {% else -%}
    Hello, {{ name }}!
    {% endif -%}
    """
    output = engine.render_string(template, greeting="Hi", name="Alice")
    print(f"Conditional: {output.strip()}")

    # With loops
    template = """
    Items:
    {% for item in items -%}
    - {{ item }}
    {% endfor -%}
    """
    output = engine.render_string(template, items=["Apple", "Banana", "Cherry"])
    print(f"Loop:\n{output}")


def demo_2_custom_filters():
    """Demo 2: Custom filters and functions"""
    print("\n" + "=" * 60)
    print("DEMO 2: Custom Filters and Functions")
    print("=" * 60)

    engine = TemplateEngine()

    # Built-in filters
    template = "{{ text | snippet(50) }}"
    text = "This is a very long text that should be truncated to show only the first part."
    output = engine.render_string(template, text=text)
    print(f"\nSnippet filter: {output}")

    # Bullet list filter
    template = "{{ items | bullet_list }}"
    items = ["First item", "Second item", "Third item"]
    output = engine.render_string(template, items=items)
    print(f"\nBullet list:\n{output}")

    # Code block filter
    template = "{{ code | code_block('python') }}"
    code = "def hello():\n    print('Hello')"
    output = engine.render_string(template, code=code)
    print(f"\nCode block:\n{output}")

    # Custom functions
    template = "Generated at: {{ now() }}"
    output = engine.render_string(template)
    print(f"\n{output}")

    template = "{{ cot_prompt() }}"
    output = engine.render_string(template)
    print(f"\nCoT prompt: {output}")


def demo_3_template_context():
    """Demo 3: Template context with defaults"""
    print("\n" + "=" * 60)
    print("DEMO 3: Template Context")
    print("=" * 60)

    engine = TemplateEngine()

    template = """
    Task: {{ task }}
    Priority: {{ priority | default('medium', true) }}
    Due: {{ due_date | default('TBD', true) }}
    """

    # Create context with defaults
    context = TemplateContext(
        variables={'task': 'Complete project'},
        defaults={'priority': 'high', 'due_date': '2024-01-31'}
    )

    output = engine.render_string(template, context)
    print(f"\nWith defaults:\n{output}")

    # Override defaults
    context2 = context.with_vars(priority='urgent', due_date='2024-01-20')
    output = engine.render_string(template, context2)
    print(f"With overrides:\n{output}")


def demo_4_registry():
    """Demo 4: Template registry"""
    print("\n" + "=" * 60)
    print("DEMO 4: Template Registry")
    print("=" * 60)

    registry = TemplateRegistry()

    # Register templates
    registry.register(
        name='greeting',
        content='Hello, {{ name }}!',
        category='general',
        description='Simple greeting template',
        tags=['greeting', 'simple']
    )

    registry.register(
        name='farewell',
        content='Goodbye, {{ name }}!',
        category='general',
        description='Simple farewell template',
        tags=['farewell', 'simple']
    )

    # List templates
    print("\nRegistered templates:")
    templates = registry.list_templates()
    for tmpl in templates:
        print(f"  - {tmpl.name} ({tmpl.category}) - {tmpl.description}")

    # Get template
    template = registry.get('greeting')
    print(f"\nGreeting template content: {template.content}")

    # Render
    engine = TemplateEngine()
    output = engine.render_string(template.content, name="Developer")
    print(f"Rendered: {output}")


def demo_5_composition():
    """Demo 5: Template composition with mixins"""
    print("\n" + "=" * 60)
    print("DEMO 5: Template Composition")
    print("=" * 60)

    composer = TemplateComposer()

    # Register mixins
    composer.register_mixin(TemplateMixin(
        name='tone-professional',
        content='Use a professional, formal tone.',
        inject_position='start',
        priority=10
    ))

    composer.register_mixin(TemplateMixin(
        name='format-markdown',
        content='Format your response using Markdown.',
        inject_position='end',
        priority=5
    ))

    # Base template
    base_template = "{{ task }}"

    # Compose with mixins
    composed = composer.compose(
        base_template,
        mixins=['tone-professional', 'format-markdown'],
        mixin_separator='\n\n'
    )

    print(f"\nComposed template:\n{composed}")

    # Render composed template
    engine = TemplateEngine()
    output = engine.render_string(
        composed,
        task="Explain the benefits of template systems."
    )
    print(f"\nRendered:\n{output}")


def demo_6_validation():
    """Demo 6: Template validation"""
    print("\n" + "=" * 60)
    print("DEMO 6: Template Validation")
    print("=" * 60)

    validator = TemplateValidator()

    # Valid template
    valid_template = {
        'name': 'test-template',
        'content': 'Hello {{ name }}!',
        'category': 'general',
        'description': 'A test template',
        'tags': ['test']
    }

    result = validator.validate(valid_template)
    print(f"\nValid template: {'PASS' if result.valid else 'FAIL'}")
    if not result.valid:
        for error in result.get_errors():
            print(f"  Error: {error}")

    # Invalid template (missing required field)
    invalid_template = {
        'name': 'bad-template',
        # Missing 'content'
        'category': 'general'
    }

    result = validator.validate(invalid_template, strict=True)
    print(f"\nInvalid template: {'PASS' if result.valid else 'FAIL'}")
    if not result.valid:
        for error in result.get_errors():
            print(f"  Error: {error}")

    # Template with warnings
    warning_template = {
        'name': 'warning-template',
        'content': 'Very short',
        'category': 'general',
        'defaults': {'unused_var': 'value'}
    }

    result = validator.validate(warning_template)
    print(f"\nTemplate with warnings: {'PASS' if result.valid else 'FAIL'}")
    warnings = result.get_warnings()
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"    - {warning}")


def demo_7_testing():
    """Demo 7: Template testing"""
    print("\n" + "=" * 60)
    print("DEMO 7: Template Testing")
    print("=" * 60)

    engine = TemplateEngine()

    # Create test suite
    suite = TestSuite(
        name='Greeting Tests',
        template_name='greeting',
        description='Test suite for greeting template'
    )

    # Add test cases
    suite.add_test(TestCase(
        name='Basic greeting',
        variables={'name': 'Alice'},
        expected_contains=['Hello', 'Alice'],
        expected_pattern=r'Hello, \w+!'
    ))

    suite.add_test(TestCase(
        name='Long name',
        variables={'name': 'Dr. Elizabeth Montgomery III'},
        expected_contains=['Dr. Elizabeth Montgomery III']
    ))

    suite.add_test(TestCase(
        name='Short output',
        variables={'name': 'Bob'},
        expected_length_max=50
    ))

    # Run tests
    template_content = 'Hello, {{ name }}!'
    runner = TemplateTestRunner(engine)
    result = runner.run_suite(template_content, suite)

    print(f"\n{result}")


def demo_8_built_in_templates():
    """Demo 8: Using built-in templates"""
    print("\n" + "=" * 60)
    print("DEMO 8: Built-in Templates")
    print("=" * 60)

    # Create registry and discover built-in templates
    registry = TemplateRegistry()
    discovery = TemplateDiscovery(registry)

    library_path = get_library_path()
    count = discovery.discover_builtin_templates(library_path)

    print(f"\nDiscovered {count} built-in templates")

    # List by category
    categories = registry.list_categories()
    print(f"\nCategories: {', '.join(categories)}")

    # Show some templates
    print("\nSample templates:")
    for category in ['base', 'domain', 'pattern']:
        templates = registry.list_templates(category=category)
        if templates:
            print(f"\n  {category.upper()}:")
            for tmpl in templates[:3]:  # Show first 3
                print(f"    - {tmpl.name}: {tmpl.description}")

    # Use a built-in template
    print("\n" + "-" * 60)
    print("Using 'summarize-text' template:")
    print("-" * 60)

    template = registry.get('summarize-text')
    if template:
        engine = TemplateEngine()
        output = engine.render_string(
            template.content,
            text="Promptly is a powerful prompt management system with versioning, branching, and template support.",
            max_sentences=2
        )
        print(f"\n{output}")


def demo_9_advanced_composition():
    """Demo 9: Advanced composition with fragments"""
    print("\n" + "=" * 60)
    print("DEMO 9: Advanced Composition")
    print("=" * 60)

    composer = TemplateComposer()

    # Register fragments
    composer.register_fragment(TemplateFragment(
        name='disclaimer',
        content='Note: This is AI-generated content.',
        description='Standard disclaimer',
        tags=['disclaimer']
    ))

    composer.register_fragment(TemplateFragment(
        name='thinking-prompt',
        content="Let's approach this step by step:",
        description='Thinking encouragement',
        tags=['reasoning']
    ))

    # Load library mixins and fragments
    library_path = get_library_path()
    mixins_file = Path(library_path) / 'mixins.yaml'
    fragments_file = Path(library_path) / 'fragments.yaml'

    if mixins_file.exists():
        composer.load_mixins_from_file(str(mixins_file))
        print(f"\nLoaded {len(composer.mixins)} mixins")

    if fragments_file.exists():
        composer.load_fragments_from_file(str(fragments_file))
        print(f"Loaded {len(composer.fragments)} fragments")

    # List available mixins
    print("\nAvailable mixins:")
    for mixin in list(composer.mixins.values())[:5]:
        print(f"  - {mixin.name}: {mixin.description}")

    # List available fragments
    print("\nAvailable fragments:")
    for fragment in list(composer.fragments.values())[:5]:
        print(f"  - {fragment.name}: {fragment.description}")


def demo_10_test_builder():
    """Demo 10: Test suite builder"""
    print("\n" + "=" * 60)
    print("DEMO 10: Test Suite Builder")
    print("=" * 60)

    engine = TemplateEngine()

    # Build test suite fluently
    suite = (TestSuiteBuilder('Math Tests', 'math-solver')
        .description('Test mathematical problem solving')
        .test(
            'Addition',
            variables={'problem': '2 + 2'},
            expected_contains=['4']
        )
        .test(
            'Multiplication',
            variables={'problem': '5 * 6'},
            expected_contains=['30']
        )
        .build())

    print(f"\nBuilt test suite: {suite.name}")
    print(f"Template: {suite.template_name}")
    print(f"Tests: {len(suite.test_cases)}")

    for test in suite.test_cases:
        print(f"  - {test.name}")


def run_all_demos():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("PROMPTLY TEMPLATE SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)

    demos = [
        demo_1_basic_rendering,
        demo_2_custom_filters,
        demo_3_template_context,
        demo_4_registry,
        demo_5_composition,
        demo_6_validation,
        demo_7_testing,
        demo_8_built_in_templates,
        demo_9_advanced_composition,
        demo_10_test_builder,
    ]

    for i, demo in enumerate(demos, 1):
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    run_all_demos()
