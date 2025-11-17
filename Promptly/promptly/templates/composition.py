#!/usr/bin/env python3
"""
Template Composition System - Mix-ins, fragments, and component composition
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import json


class CompositionError(Exception):
    """Base exception for composition errors"""
    pass


@dataclass
class TemplateFragment:
    """
    A reusable template fragment (partial template)

    Fragments are small, reusable pieces of templates that can be
    included or composed into larger templates.
    """

    name: str
    content: str
    description: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, engine, **kwargs) -> str:
        """Render this fragment with the given engine"""
        return engine.render_string(self.content, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateFragment':
        """Create a fragment from a dictionary"""
        return cls(
            name=data.get('name', ''),
            content=data.get('content', ''),
            description=data.get('description'),
            parameters=data.get('parameters', []),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert fragment to dictionary"""
        return {
            'name': self.name,
            'content': self.content,
            'description': self.description,
            'parameters': self.parameters,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class TemplateMixin:
    """
    A template mixin that can be mixed into other templates

    Mixins provide reusable functionality that can be composed
    into multiple templates (e.g., tone modifiers, formatting rules).
    """

    name: str
    content: str
    description: Optional[str] = None
    inject_position: str = 'end'  # 'start', 'end', 'before_output', 'after_input'
    priority: int = 0  # Higher priority mixins are applied later
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditions for applying mixin
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemplateMixin':
        """Create a mixin from a dictionary"""
        return cls(
            name=data.get('name', ''),
            content=data.get('content', ''),
            description=data.get('description'),
            inject_position=data.get('inject_position', 'end'),
            priority=data.get('priority', 0),
            parameters=data.get('parameters', {}),
            conditions=data.get('conditions', {}),
            metadata=data.get('metadata', {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert mixin to dictionary"""
        return {
            'name': self.name,
            'content': self.content,
            'description': self.description,
            'inject_position': self.inject_position,
            'priority': self.priority,
            'parameters': self.parameters,
            'conditions': self.conditions,
            'metadata': self.metadata
        }

    def should_apply(self, context: Dict[str, Any]) -> bool:
        """Check if this mixin should be applied based on conditions"""
        if not self.conditions:
            return True

        for key, expected_value in self.conditions.items():
            if key not in context:
                return False
            if context[key] != expected_value:
                return False

        return True


class TemplateComposer:
    """
    Composes templates from fragments and mixins

    Supports:
    - Mixing in reusable components
    - Template fragments (includes)
    - Parameter passing
    - Composition validation
    - Dependency resolution
    """

    def __init__(self):
        self.fragments: Dict[str, TemplateFragment] = {}
        self.mixins: Dict[str, TemplateMixin] = {}

    def register_fragment(self, fragment: TemplateFragment):
        """Register a template fragment"""
        self.fragments[fragment.name] = fragment

    def register_mixin(self, mixin: TemplateMixin):
        """Register a template mixin"""
        self.mixins[mixin.name] = mixin

    def get_fragment(self, name: str) -> Optional[TemplateFragment]:
        """Get a registered fragment by name"""
        return self.fragments.get(name)

    def get_mixin(self, name: str) -> Optional[TemplateMixin]:
        """Get a registered mixin by name"""
        return self.mixins.get(name)

    def compose(
        self,
        base_template: str,
        mixins: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        mixin_separator: str = '\n\n'
    ) -> str:
        """
        Compose a template from base template and mixins

        Args:
            base_template: The base template content
            mixins: List of mixin names to apply
            context: Context for conditional mixin application
            mixin_separator: Separator between mixins and base template

        Returns:
            Composed template string
        """
        if context is None:
            context = {}

        if mixins is None:
            return base_template

        # Get mixin objects and filter by conditions
        mixin_objects = []
        for mixin_name in mixins:
            mixin = self.get_mixin(mixin_name)
            if mixin is None:
                raise CompositionError(f"Mixin '{mixin_name}' not found")

            if mixin.should_apply(context):
                mixin_objects.append(mixin)

        # Sort by priority
        mixin_objects.sort(key=lambda m: m.priority)

        # Group by injection position
        start_mixins = [m for m in mixin_objects if m.inject_position == 'start']
        end_mixins = [m for m in mixin_objects if m.inject_position == 'end']

        # Compose template
        parts = []

        # Add start mixins
        for mixin in start_mixins:
            parts.append(mixin.content)

        # Add base template
        parts.append(base_template)

        # Add end mixins
        for mixin in end_mixins:
            parts.append(mixin.content)

        return mixin_separator.join(parts)

    def expand_fragments(self, template: str, engine=None, **kwargs) -> str:
        """
        Expand fragment references in a template

        Fragment syntax: {% fragment "fragment_name" %}
        or using Jinja2 include syntax if engine is provided
        """
        if engine:
            # Use Jinja2's native include mechanism
            return engine.render_string(template, **kwargs)

        # Simple fragment expansion
        import re
        pattern = r'\{%\s*fragment\s+"([^"]+)"\s*%\}'

        def replace_fragment(match):
            fragment_name = match.group(1)
            fragment = self.get_fragment(fragment_name)
            if fragment is None:
                raise CompositionError(f"Fragment '{fragment_name}' not found")
            return fragment.content

        return re.sub(pattern, replace_fragment, template)

    def load_fragments_from_file(self, path: str):
        """Load fragments from a YAML/JSON file"""
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif path.endswith('.json'):
                data = json.load(f)
            else:
                raise CompositionError(f"Unsupported file format: {path}")

        fragments_data = data.get('fragments', [])
        for fragment_data in fragments_data:
            fragment = TemplateFragment.from_dict(fragment_data)
            self.register_fragment(fragment)

    def load_mixins_from_file(self, path: str):
        """Load mixins from a YAML/JSON file"""
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif path.endswith('.json'):
                data = json.load(f)
            else:
                raise CompositionError(f"Unsupported file format: {path}")

        mixins_data = data.get('mixins', [])
        for mixin_data in mixins_data:
            mixin = TemplateMixin.from_dict(mixin_data)
            self.register_mixin(mixin)

    def load_from_directory(self, directory: str):
        """
        Load all fragments and mixins from a directory

        Expected structure:
        - fragments/*.yaml
        - mixins/*.yaml
        """
        dir_path = Path(directory)

        # Load fragments
        fragments_dir = dir_path / 'fragments'
        if fragments_dir.exists():
            for file in fragments_dir.glob('*.yaml'):
                self.load_fragments_from_file(str(file))
            for file in fragments_dir.glob('*.yml'):
                self.load_fragments_from_file(str(file))
            for file in fragments_dir.glob('*.json'):
                self.load_fragments_from_file(str(file))

        # Load mixins
        mixins_dir = dir_path / 'mixins'
        if mixins_dir.exists():
            for file in mixins_dir.glob('*.yaml'):
                self.load_mixins_from_file(str(file))
            for file in mixins_dir.glob('*.yml'):
                self.load_mixins_from_file(str(file))
            for file in mixins_dir.glob('*.json'):
                self.load_mixins_from_file(str(file))

    def list_fragments(self, tags: Optional[List[str]] = None) -> List[TemplateFragment]:
        """List all registered fragments, optionally filtered by tags"""
        fragments = list(self.fragments.values())

        if tags:
            fragments = [
                f for f in fragments
                if any(tag in f.tags for tag in tags)
            ]

        return sorted(fragments, key=lambda f: f.name)

    def list_mixins(self) -> List[TemplateMixin]:
        """List all registered mixins"""
        return sorted(self.mixins.values(), key=lambda m: m.name)

    def validate_composition(
        self,
        base_template: str,
        mixins: List[str],
        required_params: Optional[List[str]] = None
    ) -> List[str]:
        """
        Validate a template composition

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check that all mixins exist
        for mixin_name in mixins:
            if mixin_name not in self.mixins:
                warnings.append(f"Mixin '{mixin_name}' not found")

        # Check for conflicting mixins (same injection position)
        position_counts = {}
        for mixin_name in mixins:
            mixin = self.get_mixin(mixin_name)
            if mixin:
                pos = mixin.inject_position
                position_counts[pos] = position_counts.get(pos, 0) + 1

        for pos, count in position_counts.items():
            if count > 3:
                warnings.append(f"Many mixins ({count}) at position '{pos}' - may cause conflicts")

        # Check required parameters
        if required_params:
            import re
            # Find all {{ variable }} references
            var_pattern = r'\{\{\s*(\w+)'
            found_vars = set(re.findall(var_pattern, base_template))

            for mixin_name in mixins:
                mixin = self.get_mixin(mixin_name)
                if mixin:
                    found_vars.update(re.findall(var_pattern, mixin.content))

            for param in required_params:
                if param not in found_vars:
                    warnings.append(f"Required parameter '{param}' not used in template")

        return warnings

    def export_composition(
        self,
        name: str,
        base_template: str,
        mixins: List[str],
        description: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export a composition as a reusable template definition

        Returns:
            Template definition dictionary
        """
        return {
            'name': name,
            'description': description,
            'type': 'composition',
            'base_template': base_template,
            'mixins': mixins,
            'defaults': defaults or {},
            'metadata': {
                'composition': {
                    'fragment_count': len([m for m in mixins if m in self.fragments]),
                    'mixin_count': len([m for m in mixins if m in self.mixins])
                }
            }
        }
