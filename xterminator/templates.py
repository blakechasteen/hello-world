"""
Template Definitions for xTerminator
====================================

Fix templates with pattern matching and context extraction.

Charlotte's Template Library - "Some Spider spun templates!"
"""

from typing import Dict, List, Callable, Optional
import re
from dataclasses import dataclass


@dataclass
class FixTemplate:
    """A template for fixing code issues"""
    name: str
    category: str
    pattern: str  # Regex pattern to match
    template: str  # Replacement template
    context_extractors: List[str]  # Names of context variables to extract
    required_imports: List[str] = None  # Imports to add
    description: str = ""
    example_before: str = ""
    example_after: str = ""

    def __post_init__(self):
        if self.required_imports is None:
            self.required_imports = []


# Template catalog - Charlotte's web of fixes!
TEMPLATES: Dict[str, FixTemplate] = {
    # ========================================
    # ERROR HANDLING TEMPLATES
    # ========================================

    'add_try_except_file_io': FixTemplate(
        name='add_try_except_file_io',
        category='error_handling',
        pattern=r'(?P<indent>[ \t]*)(?P<statement>(?:with\s+)?open\([^)]+\)(?:\s+as\s+\w+)?:?)(?P<body>.*?)(?=\n(?P=indent)\w|\Z)',
        template='''{indent}try:
{indent}    {statement}
{indent}{body}
{indent}except (IOError, OSError) as e:
{indent}    logger.error(f"Failed to {operation}: {{e}}")
{indent}    {error_handling}''',
        context_extractors=['indent', 'statement', 'body', 'operation', 'error_handling'],
        required_imports=['import logging', 'logger = logging.getLogger(__name__)'],
        description='Wrap file I/O in try/except with logging',
        example_before='''f = open("config.txt")
data = f.read()''',
        example_after='''try:
    with open("config.txt") as f:
        data = f.read()
except (IOError, OSError) as e:
    logger.error(f"Failed to read file: {e}")
    data = None'''
    ),

    'add_try_except_json': FixTemplate(
        name='add_try_except_json',
        category='error_handling',
        pattern=r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*json\.(?P<method>load|loads)\((?P<arg>[^)]+)\)',
        template='''{indent}try:
{indent}    {var} = json.{method}({arg})
{indent}except json.JSONDecodeError as e:
{indent}    logger.error(f"Failed to parse JSON: {{e}}")
{indent}    {var} = {{}}  # or None, or raise''',
        context_extractors=['indent', 'var', 'method', 'arg'],
        required_imports=['import json', 'import logging', 'logger = logging.getLogger(__name__)'],
        description='Wrap JSON parsing in try/except',
        example_before='''data = json.load(f)''',
        example_after='''try:
    data = json.load(f)
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse JSON: {e}")
    data = {}'''
    ),

    'add_try_except_requests': FixTemplate(
        name='add_try_except_requests',
        category='error_handling',
        pattern=r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*requests\.(?P<method>get|post|put|delete)\((?P<args>[^)]+)\)',
        template='''{indent}try:
{indent}    {var} = requests.{method}({args}, timeout=30)
{indent}    {var}.raise_for_status()
{indent}except requests.RequestException as e:
{indent}    logger.error(f"HTTP request failed: {{e}}")
{indent}    {error_handling}''',
        context_extractors=['indent', 'var', 'method', 'args', 'error_handling'],
        required_imports=['import requests', 'import logging', 'logger = logging.getLogger(__name__)'],
        description='Wrap HTTP requests in try/except with timeout',
        example_before='''response = requests.get(url)''',
        example_after='''try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.RequestException as e:
    logger.error(f"HTTP request failed: {e}")
    response = None'''
    ),

    # ========================================
    # RESOURCE MANAGEMENT TEMPLATES
    # ========================================

    'add_context_manager': FixTemplate(
        name='add_context_manager',
        category='resource_management',
        pattern=r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*open\((?P<args>[^)]+)\)',
        template='''{indent}with open({args}) as {var}:
{indent}    {body}''',
        context_extractors=['indent', 'var', 'args', 'body'],
        required_imports=[],
        description='Replace open() with context manager',
        example_before='''f = open("data.txt")
content = f.read()
f.close()''',
        example_after='''with open("data.txt") as f:
    content = f.read()'''
    ),

    'add_db_context_manager': FixTemplate(
        name='add_db_context_manager',
        category='resource_management',
        pattern=r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*(?P<db_type>sqlite3|psycopg2|pymongo)\.connect\((?P<args>[^)]+)\)',
        template='''{indent}with {db_type}.connect({args}) as {var}:
{indent}    {body}''',
        context_extractors=['indent', 'var', 'db_type', 'args', 'body'],
        required_imports=[],
        description='Add context manager for database connections',
        example_before='''conn = sqlite3.connect("db.sqlite")
cursor = conn.cursor()''',
        example_after='''with sqlite3.connect("db.sqlite") as conn:
    cursor = conn.cursor()'''
    ),

    # ========================================
    # HARDCODED VALUES TEMPLATES
    # ========================================

    'move_to_env_var': FixTemplate(
        name='move_to_env_var',
        category='hardcoded_values',
        pattern=r'(?P<indent>[ \t]*)(?P<var>[A-Z_]+)\s*=\s*["\'](?P<value>(?:sk-|api[_-]?key|token|secret|password)[^"\']*)["\']',
        template='''{indent}{var} = os.getenv("{var}", "")
{indent}if not {var}:
{indent}    raise ValueError("{var} environment variable not set")''',
        context_extractors=['indent', 'var', 'value'],
        required_imports=['import os'],
        description='Move hardcoded secrets to environment variables',
        example_before='''API_KEY = "sk-abc123..."''',
        example_after='''API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")'''
    ),

    'move_to_constant': FixTemplate(
        name='move_to_constant',
        category='hardcoded_values',
        pattern=r'(?P<indent>[ \t]*)(?P<statement>.+?["\'](?P<value>http[s]?://[^"\']+)["\'])',
        template='''{indent}# TODO: Move to config
{indent}{constant_name} = "{value}"
{indent}{statement_updated}''',
        context_extractors=['indent', 'constant_name', 'value', 'statement_updated'],
        required_imports=[],
        description='Extract hardcoded URL to constant',
        example_before='''response = requests.get("https://api.example.com/data")''',
        example_after='''# TODO: Move to config
API_ENDPOINT = "https://api.example.com/data"
response = requests.get(API_ENDPOINT)'''
    ),

    # ========================================
    # CODE QUALITY TEMPLATES
    # ========================================

    'add_null_check': FixTemplate(
        name='add_null_check',
        category='defensive_coding',
        pattern=r'(?P<indent>[ \t]*)(?P<statement>(?P<var>\w+)\.(?P<attr>\w+))',
        template='''{indent}if {var} is None:
{indent}    raise ValueError("{var} cannot be None")
{indent}{statement}''',
        context_extractors=['indent', 'var', 'statement', 'attr'],
        required_imports=[],
        description='Add None check before attribute access',
        example_before='''result = obj.process()''',
        example_after='''if obj is None:
    raise ValueError("obj cannot be None")
result = obj.process()'''
    ),

    'add_docstring': FixTemplate(
        name='add_docstring',
        category='documentation',
        pattern=r'(?P<indent>[ \t]*)def\s+(?P<func_name>\w+)\((?P<params>[^)]*)\)(?:\s*->\s*(?P<return_type>\w+))?\s*:',
        template='''{indent}def {func_name}({params}){return_annotation}:
{indent}    """
{indent}    {summary}
{indent}
{indent}    Args:
{indent}        {param_docs}
{indent}
{indent}    Returns:
{indent}        {return_doc}
{indent}    """''',
        context_extractors=['indent', 'func_name', 'params', 'return_annotation', 'summary', 'param_docs', 'return_doc'],
        required_imports=[],
        description='Add basic docstring template',
        example_before='''def process_data(user_id, config):
    pass''',
        example_after='''def process_data(user_id, config):
    """
    Process user data with configuration.

    Args:
        user_id: User identifier
        config: Configuration dictionary

    Returns:
        Processed data
    """
    pass'''
    ),

    'fix_timezone_naive': FixTemplate(
        name='fix_timezone_naive',
        category='datetime',
        pattern=r'(?P<indent>[ \t]*)(?P<var>\w+)\s*=\s*datetime\.(?:now|utcnow)\(\)',
        template='''{indent}{var} = datetime.now(timezone.utc)''',
        context_extractors=['indent', 'var'],
        required_imports=['from datetime import timezone'],
        description='Replace naive datetime with timezone-aware',
        example_before='''now = datetime.now()''',
        example_after='''now = datetime.now(timezone.utc)'''
    ),

    'add_type_hints': FixTemplate(
        name='add_type_hints',
        category='type_safety',
        pattern=r'(?P<indent>[ \t]*)def\s+(?P<func_name>\w+)\((?P<params>[^)]*)\)\s*:',
        template='''{indent}def {func_name}({typed_params}) -> {return_type}:''',
        context_extractors=['indent', 'func_name', 'typed_params', 'return_type'],
        required_imports=['from typing import Any, Optional, Dict, List'],
        description='Add basic type hints to function signature',
        example_before='''def process(data, config):
    pass''',
        example_after='''def process(data: Dict, config: Optional[Dict] = None) -> Any:
    pass'''
    ),

    'add_logging': FixTemplate(
        name='add_logging',
        category='observability',
        pattern=r'(?P<indent>[ \t]*)(?P<statement>(?:raise|return)\s+.+)',
        template='''{indent}logger.{level}("{message}")
{indent}{statement}''',
        context_extractors=['indent', 'statement', 'level', 'message'],
        required_imports=['import logging', 'logger = logging.getLogger(__name__)'],
        description='Add logging before raise/return',
        example_before='''raise ValueError("Invalid data")''',
        example_after='''logger.error("Invalid data - raising ValueError")
raise ValueError("Invalid data")'''
    ),
}


def get_template(name: str) -> Optional[FixTemplate]:
    """Get template by name"""
    return TEMPLATES.get(name)


def get_templates_for_category(category: str) -> List[FixTemplate]:
    """Get all templates for a category"""
    return [t for t in TEMPLATES.values() if t.category == category]


def list_templates() -> List[str]:
    """List all template names"""
    return list(TEMPLATES.keys())


def get_template_catalog() -> str:
    """
    Get human-readable template catalog.

    Charlotte's web of templates!
    """
    catalog = []
    catalog.append("=" * 70)
    catalog.append("CHARLOTTE'S TEMPLATE CATALOG")
    catalog.append("Some Spider spun templates!")
    catalog.append("=" * 70)
    catalog.append("")

    # Group by category
    by_category: Dict[str, List[FixTemplate]] = {}
    for template in TEMPLATES.values():
        if template.category not in by_category:
            by_category[template.category] = []
        by_category[template.category].append(template)

    for category, templates in sorted(by_category.items()):
        catalog.append(f"\n{category.upper().replace('_', ' ')}")
        catalog.append("-" * 70)

        for template in templates:
            catalog.append(f"\n  {template.name}")
            catalog.append(f"  {template.description}")

            if template.required_imports:
                catalog.append(f"  Requires: {', '.join(template.required_imports)}")

            if template.example_before:
                catalog.append("\n  Before:")
                for line in template.example_before.strip().split('\n'):
                    catalog.append(f"    {line}")

            if template.example_after:
                catalog.append("\n  After:")
                for line in template.example_after.strip().split('\n'):
                    catalog.append(f"    {line}")

            catalog.append("")

    catalog.append("=" * 70)
    catalog.append(f"Total: {len(TEMPLATES)} templates")
    catalog.append("=" * 70)

    return '\n'.join(catalog)


if __name__ == '__main__':
    # Print template catalog
    print(get_template_catalog())
