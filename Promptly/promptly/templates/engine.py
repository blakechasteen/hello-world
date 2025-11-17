#!/usr/bin/env python3
"""
Template Engine - Jinja2-based templating with custom filters and functions
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import yaml
import json

try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        select_autoescape,
        TemplateNotFound,
        TemplateSyntaxError,
        UndefinedError
    )
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available. Install with: pip install jinja2")


class TemplateEngineError(Exception):
    """Base exception for template engine errors"""
    pass


class TemplateNotFoundError(TemplateEngineError):
    """Raised when a template is not found"""
    pass


class TemplateRenderError(TemplateEngineError):
    """Raised when template rendering fails"""
    pass


@dataclass
class TemplateContext:
    """Context for template rendering with defaults and validation"""

    variables: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: 'TemplateContext') -> 'TemplateContext':
        """Merge two contexts, with other taking precedence"""
        return TemplateContext(
            variables={**self.variables, **other.variables},
            defaults={**self.defaults, **other.defaults},
            metadata={**self.metadata, **other.metadata}
        )

    def get_all_vars(self) -> Dict[str, Any]:
        """Get all variables with defaults applied"""
        return {**self.defaults, **self.variables}

    def with_vars(self, **kwargs) -> 'TemplateContext':
        """Create a new context with additional variables"""
        new_vars = {**self.variables, **kwargs}
        return TemplateContext(
            variables=new_vars,
            defaults=self.defaults,
            metadata=self.metadata
        )


class TemplateEngine:
    """
    Jinja2-based template engine with custom filters and functions

    Features:
    - Variable substitution with defaults
    - Conditional blocks
    - Loops and iterations
    - Macro definitions
    - Template inheritance (extends, includes)
    - Custom filters and functions
    """

    def __init__(
        self,
        template_dirs: Optional[List[str]] = None,
        autoescape: bool = False,
        cache_size: int = 400
    ):
        if not JINJA2_AVAILABLE:
            raise TemplateEngineError("Jinja2 is required for template engine")

        self.template_dirs = template_dirs or []
        self.autoescape = autoescape
        self.cache_size = cache_size

        # Initialize Jinja2 environment
        self.env = self._create_environment()

        # Register custom filters and functions
        self._register_filters()
        self._register_functions()
        self._register_tests()

    def _create_environment(self) -> Environment:
        """Create Jinja2 environment with loaders"""
        if self.template_dirs:
            loader = FileSystemLoader(self.template_dirs)
        else:
            loader = None

        env = Environment(
            loader=loader,
            autoescape=select_autoescape() if self.autoescape else False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            cache_size=self.cache_size
        )

        return env

    def _register_filters(self):
        """Register custom Jinja2 filters"""

        # String manipulation filters
        self.env.filters['snippet'] = lambda s, length=100: s[:length] + '...' if len(s) > length else s
        self.env.filters['titlecase'] = lambda s: s.title()
        self.env.filters['slugify'] = lambda s: re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

        # List filters
        self.env.filters['join_with'] = lambda lst, sep=', ', last_sep=' and ': \
            sep.join(lst[:-1]) + last_sep + lst[-1] if len(lst) > 1 else lst[0] if lst else ''

        # JSON/YAML filters
        self.env.filters['to_json'] = lambda obj, indent=2: json.dumps(obj, indent=indent)
        self.env.filters['from_json'] = lambda s: json.loads(s)
        self.env.filters['to_yaml'] = lambda obj: yaml.dump(obj, default_flow_style=False)
        self.env.filters['from_yaml'] = lambda s: yaml.safe_load(s)

        # Number filters
        self.env.filters['percentage'] = lambda n, decimals=1: f"{n*100:.{decimals}f}%"
        self.env.filters['round_to'] = lambda n, decimals=2: round(n, decimals)

        # Prompt-specific filters
        self.env.filters['few_shot'] = self._few_shot_filter
        self.env.filters['bullet_list'] = lambda items: '\n'.join(f'- {item}' for item in items)
        self.env.filters['numbered_list'] = lambda items: '\n'.join(f'{i+1}. {item}' for i, item in enumerate(items))
        self.env.filters['xml_tag'] = lambda content, tag: f'<{tag}>{content}</{tag}>'
        self.env.filters['code_block'] = lambda code, lang='': f'```{lang}\n{code}\n```'

    def _few_shot_filter(self, examples: List[Dict[str, str]], format_str: str = None) -> str:
        """Format few-shot examples"""
        if not format_str:
            format_str = "Input: {input}\nOutput: {output}"

        formatted_examples = []
        for example in examples:
            formatted = format_str.format(**example)
            formatted_examples.append(formatted)

        return '\n\n'.join(formatted_examples)

    def _register_functions(self):
        """Register custom Jinja2 global functions"""

        # Utility functions
        self.env.globals['now'] = lambda: datetime.now().isoformat()
        self.env.globals['today'] = lambda: datetime.now().strftime('%Y-%m-%d')
        self.env.globals['enumerate'] = enumerate
        self.env.globals['zip'] = zip
        self.env.globals['len'] = len
        self.env.globals['range'] = range

        # Prompt construction helpers
        self.env.globals['role_message'] = self._role_message
        self.env.globals['system_message'] = lambda content: self._role_message('system', content)
        self.env.globals['user_message'] = lambda content: self._role_message('user', content)
        self.env.globals['assistant_message'] = lambda content: self._role_message('assistant', content)

        # Chain-of-thought helpers
        self.env.globals['cot_prompt'] = lambda: "Let's think step by step:"
        self.env.globals['react_prompt'] = self._react_prompt

    def _role_message(self, role: str, content: str) -> Dict[str, str]:
        """Create a role-based message structure"""
        return {'role': role, 'content': content}

    def _react_prompt(self, task: str) -> str:
        """Generate ReAct pattern prompt"""
        return f"""Solve this task using the ReAct (Reasoning + Acting) pattern:

Task: {task}

Follow this format:
Thought: [Your reasoning about what to do next]
Action: [The action to take]
Observation: [What you observe from the action]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Answer: [Final answer]
"""

    def _register_tests(self):
        """Register custom Jinja2 tests"""
        self.env.tests['empty_string'] = lambda s: s == ''
        self.env.tests['non_empty'] = lambda s: bool(s)

    def add_filter(self, name: str, func: Callable):
        """Add a custom filter"""
        self.env.filters[name] = func

    def add_function(self, name: str, func: Callable):
        """Add a custom global function"""
        self.env.globals[name] = func

    def add_template_dir(self, path: str):
        """Add a template directory to the search path"""
        if path not in self.template_dirs:
            self.template_dirs.append(path)
            # Recreate environment with new loader
            self.env = self._create_environment()
            self._register_filters()
            self._register_functions()
            self._register_tests()

    def render_string(
        self,
        template_string: str,
        context: Optional[TemplateContext] = None,
        **kwargs
    ) -> str:
        """Render a template from a string"""
        try:
            template = self.env.from_string(template_string)

            # Prepare variables
            if context:
                variables = context.get_all_vars()
            else:
                variables = {}

            # Merge with kwargs
            variables.update(kwargs)

            return template.render(**variables)

        except TemplateSyntaxError as e:
            raise TemplateRenderError(f"Template syntax error: {e}")
        except UndefinedError as e:
            raise TemplateRenderError(f"Undefined variable: {e}")
        except Exception as e:
            raise TemplateRenderError(f"Rendering failed: {e}")

    def render_file(
        self,
        template_name: str,
        context: Optional[TemplateContext] = None,
        **kwargs
    ) -> str:
        """Render a template from a file"""
        try:
            template = self.env.get_template(template_name)

            # Prepare variables
            if context:
                variables = context.get_all_vars()
            else:
                variables = {}

            # Merge with kwargs
            variables.update(kwargs)

            return template.render(**variables)

        except TemplateNotFound:
            raise TemplateNotFoundError(f"Template '{template_name}' not found")
        except TemplateSyntaxError as e:
            raise TemplateRenderError(f"Template syntax error in '{template_name}': {e}")
        except UndefinedError as e:
            raise TemplateRenderError(f"Undefined variable in '{template_name}': {e}")
        except Exception as e:
            raise TemplateRenderError(f"Rendering '{template_name}' failed: {e}")

    def render_template(
        self,
        template_data: Dict[str, Any],
        context: Optional[TemplateContext] = None,
        **kwargs
    ) -> str:
        """
        Render a template from a dictionary definition

        Template data format:
        {
            'name': 'template_name',
            'content': 'template content or path',
            'type': 'string' or 'file',
            'defaults': {...},
            'metadata': {...}
        }
        """
        template_type = template_data.get('type', 'string')
        content = template_data.get('content', '')
        defaults = template_data.get('defaults', {})

        # Create or merge context
        if context is None:
            context = TemplateContext(defaults=defaults)
        else:
            # Merge defaults
            context = TemplateContext(
                variables=context.variables,
                defaults={**defaults, **context.defaults},
                metadata=context.metadata
            )

        # Render based on type
        if template_type == 'file':
            return self.render_file(content, context, **kwargs)
        else:
            return self.render_string(content, context, **kwargs)

    def load_template_definition(self, path: str) -> Dict[str, Any]:
        """Load template definition from YAML/JSON file"""
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                return yaml.safe_load(f)
            elif path.endswith('.json'):
                return json.load(f)
            else:
                raise TemplateEngineError(f"Unsupported template file format: {path}")

    def render_from_file(
        self,
        definition_path: str,
        context: Optional[TemplateContext] = None,
        **kwargs
    ) -> str:
        """Load and render a template from a definition file"""
        template_data = self.load_template_definition(definition_path)
        return self.render_template(template_data, context, **kwargs)

    def list_templates(self) -> List[str]:
        """List all available templates in template directories"""
        templates = []
        for template_dir in self.template_dirs:
            path = Path(template_dir)
            if path.exists():
                for file in path.rglob('*'):
                    if file.is_file() and file.suffix in ['.j2', '.jinja', '.jinja2', '.txt', '.md']:
                        # Get relative path from template dir
                        rel_path = file.relative_to(path)
                        templates.append(str(rel_path))
        return sorted(templates)

    def compile_template(self, template_string: str) -> Any:
        """Compile a template for reuse (returns Jinja2 template object)"""
        try:
            return self.env.from_string(template_string)
        except TemplateSyntaxError as e:
            raise TemplateRenderError(f"Template compilation failed: {e}")


# Convenience function for quick rendering
def render(template_string: str, **kwargs) -> str:
    """Quick template rendering without creating an engine instance"""
    engine = TemplateEngine()
    return engine.render_string(template_string, **kwargs)
