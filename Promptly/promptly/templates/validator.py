#!/usr/bin/env python3
"""
Template Validator - Validation with JSON Schema support
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re

try:
    import jsonschema
    from jsonschema import validate, ValidationError as JsonSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Warning: jsonschema not available. Install with: pip install jsonschema")


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = 'error'
    WARNING = 'warning'
    INFO = 'info'


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""

    level: ValidationLevel
    message: str
    location: Optional[str] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.level.value.upper()}]"]
        if self.location:
            parts.append(f"at {self.location}:")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"(Suggestion: {self.suggestion})")
        return ' '.join(parts)


@dataclass
class ValidationResult:
    """Results from template validation"""

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str, **kwargs):
        """Add an error issue"""
        self.issues.append(ValidationIssue(level=ValidationLevel.ERROR, message=message, **kwargs))
        self.valid = False

    def add_warning(self, message: str, **kwargs):
        """Add a warning issue"""
        self.issues.append(ValidationIssue(level=ValidationLevel.WARNING, message=message, **kwargs))

    def add_info(self, message: str, **kwargs):
        """Add an info issue"""
        self.issues.append(ValidationIssue(level=ValidationLevel.INFO, message=message, **kwargs))

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues"""
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues"""
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]

    def get_info(self) -> List[ValidationIssue]:
        """Get all info-level issues"""
        return [i for i in self.issues if i.level == ValidationLevel.INFO]

    def __str__(self) -> str:
        lines = [f"Validation {'PASSED' if self.valid else 'FAILED'}"]
        lines.append(f"Issues: {len(self.issues)}")

        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                lines.append(f"  {issue}")

        return '\n'.join(lines)


class TemplateValidator:
    """
    Template validator with multiple validation strategies

    Validations:
    - JSON Schema validation
    - Jinja2 syntax validation
    - Variable completeness checks
    - Best practices validation
    - Security checks
    """

    # Default template schema
    DEFAULT_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name", "content"],
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[a-zA-Z0-9_-]+$"
            },
            "content": {
                "type": "string",
                "minLength": 1
            },
            "description": {
                "type": "string"
            },
            "category": {
                "type": "string"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "defaults": {
                "type": "object"
            },
            "metadata": {
                "type": "object"
            }
        }
    }

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or self.DEFAULT_SCHEMA
        self.custom_validators: List[Any] = []

    def validate(
        self,
        template_data: Dict[str, Any],
        strict: bool = False,
        check_syntax: bool = True,
        check_variables: bool = True,
        check_security: bool = True,
        check_best_practices: bool = True
    ) -> ValidationResult:
        """
        Validate a template

        Args:
            template_data: Template data dictionary
            strict: Enable strict validation
            check_syntax: Check Jinja2 syntax
            check_variables: Check variable usage
            check_security: Check for security issues
            check_best_practices: Check best practices

        Returns:
            ValidationResult with all issues
        """
        result = ValidationResult(valid=True)

        # Schema validation
        if JSONSCHEMA_AVAILABLE:
            self._validate_schema(template_data, result, strict)
        else:
            result.add_warning("JSON Schema validation skipped (jsonschema not installed)")

        # Syntax validation
        if check_syntax:
            self._validate_syntax(template_data, result)

        # Variable validation
        if check_variables:
            self._validate_variables(template_data, result)

        # Security validation
        if check_security:
            self._validate_security(template_data, result)

        # Best practices validation
        if check_best_practices:
            self._validate_best_practices(template_data, result)

        # Custom validators
        for validator in self.custom_validators:
            validator(template_data, result)

        return result

    def _validate_schema(
        self,
        template_data: Dict[str, Any],
        result: ValidationResult,
        strict: bool
    ):
        """Validate against JSON schema"""
        try:
            validate(instance=template_data, schema=self.schema)
        except JsonSchemaValidationError as e:
            if strict:
                result.add_error(
                    f"Schema validation failed: {e.message}",
                    location=f"$.{'.'.join(str(p) for p in e.path)}",
                    code='SCHEMA_ERROR'
                )
            else:
                result.add_warning(
                    f"Schema validation warning: {e.message}",
                    code='SCHEMA_WARNING'
                )

    def _validate_syntax(
        self,
        template_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate Jinja2 template syntax"""
        content = template_data.get('content', '')

        try:
            # Try to parse as Jinja2 template
            from jinja2 import Environment, TemplateSyntaxError

            env = Environment()
            env.parse(content)

        except TemplateSyntaxError as e:
            result.add_error(
                f"Template syntax error: {e.message}",
                location=f"line {e.lineno}",
                code='SYNTAX_ERROR'
            )
        except ImportError:
            result.add_warning("Syntax validation skipped (jinja2 not installed)")
        except Exception as e:
            result.add_error(
                f"Unexpected syntax validation error: {e}",
                code='SYNTAX_ERROR'
            )

    def _validate_variables(
        self,
        template_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Validate variable usage"""
        content = template_data.get('content', '')
        defaults = template_data.get('defaults', {})

        # Find all {{ variable }} references
        var_pattern = r'\{\{\s*(\w+)'
        variables = set(re.findall(var_pattern, content))

        # Find all {% for var in ... %} and {% set var = ... %}
        for_pattern = r'\{%\s*for\s+(\w+)\s+in'
        set_pattern = r'\{%\s*set\s+(\w+)\s*='
        loop_vars = set(re.findall(for_pattern, content))
        set_vars = set(re.findall(set_pattern, content))

        # Variables that need to be provided
        required_vars = variables - loop_vars - set_vars

        # Check for variables without defaults
        missing_defaults = required_vars - set(defaults.keys())

        if missing_defaults:
            result.add_info(
                f"Variables without defaults: {', '.join(sorted(missing_defaults))}",
                code='MISSING_DEFAULTS',
                suggestion="Consider adding defaults for these variables"
            )

        # Check for unused defaults
        unused_defaults = set(defaults.keys()) - variables

        if unused_defaults:
            result.add_warning(
                f"Unused default values: {', '.join(sorted(unused_defaults))}",
                code='UNUSED_DEFAULTS',
                suggestion="Remove unused defaults or check variable names"
            )

        # Store variable info in metadata
        result.metadata['variables'] = {
            'total': len(variables),
            'required': sorted(required_vars),
            'loop_vars': sorted(loop_vars),
            'set_vars': sorted(set_vars),
            'with_defaults': sorted(set(defaults.keys()) & variables)
        }

    def _validate_security(
        self,
        template_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Check for potential security issues"""
        content = template_data.get('content', '')

        # Check for dangerous patterns
        dangerous_patterns = [
            (r'import\s+os', 'OS module import detected', 'Avoid importing os module in templates'),
            (r'import\s+subprocess', 'Subprocess import detected', 'Avoid importing subprocess in templates'),
            (r'eval\s*\(', 'eval() usage detected', 'Never use eval() in templates'),
            (r'exec\s*\(', 'exec() usage detected', 'Never use exec() in templates'),
            (r'__import__', '__import__ usage detected', 'Avoid dynamic imports in templates'),
        ]

        for pattern, message, suggestion in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result.add_error(
                    message,
                    code='SECURITY_RISK',
                    suggestion=suggestion
                )

        # Check for unescaped variables in contexts where XSS might be a concern
        if '{{ ' in content and '|safe' in content:
            result.add_warning(
                'Manual |safe filter usage detected',
                code='POTENTIAL_XSS',
                suggestion='Ensure all |safe filters are necessary and safe'
            )

    def _validate_best_practices(
        self,
        template_data: Dict[str, Any],
        result: ValidationResult
    ):
        """Check for best practices"""
        name = template_data.get('name', '')
        content = template_data.get('content', '')
        description = template_data.get('description', '')

        # Check for description
        if not description:
            result.add_info(
                'Template has no description',
                code='MISSING_DESCRIPTION',
                suggestion='Add a description to help users understand the template'
            )

        # Check template length
        if len(content) > 5000:
            result.add_warning(
                f'Template is very long ({len(content)} characters)',
                code='LONG_TEMPLATE',
                suggestion='Consider breaking into smaller templates or using includes'
            )

        # Check for hardcoded values that might be variables
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        url_pattern = r'https?://[^\s<>"\']+'

        if re.search(email_pattern, content) and '{{' not in content:
            result.add_info(
                'Email address found in template',
                code='HARDCODED_EMAIL',
                suggestion='Consider using a variable instead of hardcoding'
            )

        # Check for TODO/FIXME comments
        if 'TODO' in content or 'FIXME' in content:
            result.add_info(
                'TODO/FIXME comments found',
                code='TODO_COMMENT',
                suggestion='Complete TODOs before using template in production'
            )

        # Check naming convention
        if not re.match(r'^[a-z0-9_-]+$', name):
            result.add_warning(
                'Template name should use lowercase letters, numbers, hyphens, and underscores only',
                code='NAMING_CONVENTION',
                suggestion='Rename template to follow naming convention'
            )

        # Check for proper whitespace control
        if '{%' in content and not ('{%-' in content or '-%}' in content):
            result.add_info(
                'Template may benefit from whitespace control',
                code='WHITESPACE_CONTROL',
                suggestion='Use {%- and -%} to control whitespace in output'
            )

    def add_custom_validator(self, validator: callable):
        """
        Add a custom validator function

        Function signature: validator(template_data: Dict, result: ValidationResult) -> None
        """
        self.custom_validators.append(validator)

    def validate_file(self, filepath: str, **kwargs) -> ValidationResult:
        """Validate a template file"""
        import yaml

        with open(filepath, 'r') as f:
            template_data = yaml.safe_load(f)

        return self.validate(template_data, **kwargs)

    def batch_validate(
        self,
        templates: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, ValidationResult]:
        """Validate multiple templates and return results"""
        results = {}

        for template in templates:
            name = template.get('name', 'unknown')
            results[name] = self.validate(template, **kwargs)

        return results


# Convenience function for quick validation
def validate_template(template_data: Dict[str, Any], **kwargs) -> ValidationResult:
    """Quick template validation without creating a validator instance"""
    validator = TemplateValidator()
    return validator.validate(template_data, **kwargs)
