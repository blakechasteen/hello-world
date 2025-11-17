#!/usr/bin/env python3
"""
Structured Output Evaluator

Validates structured output formats including:
- JSON with schema validation
- XML with XSD/DTD validation
- YAML validation
- SQL query validation
- OpenAPI/AsyncAPI schema compliance
"""

from typing import Dict, Any, Optional, List, Set
import json
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptly.plugins.base import BaseEvaluator


class StructuredOutputEvaluator(BaseEvaluator):
    """
    Base class for structured output evaluators.

    Provides common functionality for validating structured formats.
    """

    def __init__(self, name: str, description: str, strict: bool = True):
        """
        Initialize structured output evaluator

        Args:
            name: Evaluator name
            description: Evaluator description
            strict: Whether to use strict validation
        """
        super().__init__(name, description)
        self.strict = strict

    def validate_structure(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate structure of text

        Returns:
            Dict with validation results
        """
        raise NotImplementedError


class JSONValidator(StructuredOutputEvaluator):
    """
    JSON validation evaluator.

    Validates JSON structure, schema compliance, and required fields.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None,
                 required_fields: Optional[List[str]] = None,
                 strict: bool = True):
        """
        Initialize JSON validator

        Args:
            schema: JSON schema for validation
            required_fields: List of required field paths (e.g., ["user.name", "user.email"])
            strict: Whether to enforce strict schema validation
        """
        super().__init__(
            name="json_validator",
            description="JSON structure and schema validator",
            strict=strict
        )
        self.schema = schema
        self.required_fields = required_fields or []
        self._validator = None
        self._init_validator()

    def _init_validator(self):
        """Initialize JSON schema validator"""
        try:
            import jsonschema
            self._validator = jsonschema
        except ImportError:
            print("Warning: jsonschema not available. Install with: pip install jsonschema")

    def _get_nested_field(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested field from data using dot notation"""
        parts = path.split('.')
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def validate_structure(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate JSON structure"""
        result = {
            'is_valid_json': False,
            'is_valid_schema': False,
            'missing_fields': [],
            'errors': []
        }

        # Try to parse JSON
        try:
            data = json.loads(text)
            result['is_valid_json'] = True
            result['data'] = data
        except json.JSONDecodeError as e:
            result['errors'].append(f"Invalid JSON: {str(e)}")
            return result

        # Check required fields
        for field_path in self.required_fields:
            value = self._get_nested_field(data, field_path)
            if value is None:
                result['missing_fields'].append(field_path)

        # Validate against schema
        schema = schema or self.schema
        if schema and self._validator:
            try:
                self._validator.validate(data, schema)
                result['is_valid_schema'] = True
            except self._validator.ValidationError as e:
                result['errors'].append(f"Schema validation failed: {e.message}")
            except Exception as e:
                result['errors'].append(f"Validation error: {str(e)}")
        elif schema:
            # Basic validation without jsonschema library
            result['is_valid_schema'] = self._basic_schema_validation(data, schema)
        else:
            # No schema, just check if JSON is valid
            result['is_valid_schema'] = True

        return result

    def _basic_schema_validation(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Basic schema validation without jsonschema library"""
        # Simple type checking
        schema_type = schema.get('type')

        if schema_type == 'object':
            if not isinstance(data, dict):
                return False

            # Check required properties
            required = schema.get('required', [])
            for field in required:
                if field not in data:
                    return False

            # Recursively validate properties
            properties = schema.get('properties', {})
            for key, value in data.items():
                if key in properties:
                    if not self._basic_schema_validation(value, properties[key]):
                        return False

        elif schema_type == 'array':
            if not isinstance(data, list):
                return False

            # Validate items
            items_schema = schema.get('items')
            if items_schema:
                for item in data:
                    if not self._basic_schema_validation(item, items_schema):
                        return False

        elif schema_type == 'string':
            return isinstance(data, str)
        elif schema_type == 'number':
            return isinstance(data, (int, float))
        elif schema_type == 'integer':
            return isinstance(data, int)
        elif schema_type == 'boolean':
            return isinstance(data, bool)

        return True

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate JSON validity and schema compliance

        Args:
            actual: JSON string to validate
            expected: Expected JSON or schema
            context: Optional context with schema

        Returns:
            float: Score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Get schema from context or use provided schema
        schema = context.get('schema', self.schema) if context else self.schema

        # Validate structure
        result = self.validate_structure(actual, schema)

        # Calculate score
        score = 0.0

        # Valid JSON: 40%
        if result['is_valid_json']:
            score += 0.4

        # Schema compliance: 40%
        if result['is_valid_schema']:
            score += 0.4

        # All required fields present: 20%
        if not result['missing_fields']:
            score += 0.2

        return score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed JSON validation metrics"""
        schema = context.get('schema', self.schema) if context else self.schema
        result = self.validate_structure(actual, schema)

        return {
            'score': self.evaluate(actual, expected, context),
            **result,
            'has_schema': schema is not None,
            'required_fields': self.required_fields
        }


class XMLValidator(StructuredOutputEvaluator):
    """
    XML validation evaluator.

    Validates XML structure and well-formedness.
    """

    def __init__(self, xsd_schema: Optional[str] = None,
                 required_elements: Optional[List[str]] = None):
        """
        Initialize XML validator

        Args:
            xsd_schema: XSD schema string for validation
            required_elements: List of required XML elements
        """
        super().__init__(
            name="xml_validator",
            description="XML structure validator"
        )
        self.xsd_schema = xsd_schema
        self.required_elements = required_elements or []
        self._parser = None
        self._init_parser()

    def _init_parser(self):
        """Initialize XML parser"""
        try:
            from lxml import etree
            self._parser = etree
        except ImportError:
            try:
                import xml.etree.ElementTree as ET
                self._parser = ET
            except ImportError:
                print("Warning: No XML parser available")

    def validate_structure(self, text: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Validate XML structure"""
        result = {
            'is_well_formed': False,
            'is_valid_schema': False,
            'missing_elements': [],
            'errors': []
        }

        if not self._parser:
            result['errors'].append("No XML parser available")
            return result

        # Try to parse XML
        try:
            if hasattr(self._parser, 'fromstring'):
                root = self._parser.fromstring(text.encode('utf-8'))
            else:
                root = self._parser.XML(text)
            result['is_well_formed'] = True
        except Exception as e:
            result['errors'].append(f"Invalid XML: {str(e)}")
            return result

        # Check required elements
        for element in self.required_elements:
            found = root.findall(f".//{element}")
            if not found:
                result['missing_elements'].append(element)

        # Validate against XSD schema (if lxml available)
        schema = schema or self.xsd_schema
        if schema and hasattr(self._parser, 'XMLSchema'):
            try:
                schema_root = self._parser.fromstring(schema.encode('utf-8'))
                xsd = self._parser.XMLSchema(schema_root)
                if xsd.validate(root):
                    result['is_valid_schema'] = True
                else:
                    result['errors'].append(f"Schema validation failed: {xsd.error_log}")
            except Exception as e:
                result['errors'].append(f"Schema error: {str(e)}")
        else:
            # No schema validation
            result['is_valid_schema'] = True

        return result

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate XML validity"""
        if not actual:
            return 0.0

        result = self.validate_structure(actual)

        score = 0.0

        # Well-formed XML: 50%
        if result['is_well_formed']:
            score += 0.5

        # Schema compliance: 30%
        if result['is_valid_schema']:
            score += 0.3

        # Required elements: 20%
        if not result['missing_elements']:
            score += 0.2

        return score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed XML validation metrics"""
        result = self.validate_structure(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            **result,
            'has_schema': self.xsd_schema is not None,
            'required_elements': self.required_elements
        }


class YAMLValidator(StructuredOutputEvaluator):
    """
    YAML validation evaluator.

    Validates YAML structure and schema.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None,
                 required_keys: Optional[List[str]] = None):
        """
        Initialize YAML validator

        Args:
            schema: YAML schema for validation
            required_keys: List of required top-level keys
        """
        super().__init__(
            name="yaml_validator",
            description="YAML structure validator"
        )
        self.schema = schema
        self.required_keys = required_keys or []
        self._yaml = None
        self._init_yaml()

    def _init_yaml(self):
        """Initialize YAML parser"""
        try:
            import yaml
            self._yaml = yaml
        except ImportError:
            print("Warning: PyYAML not available. Install with: pip install pyyaml")

    def validate_structure(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate YAML structure"""
        result = {
            'is_valid_yaml': False,
            'missing_keys': [],
            'errors': []
        }

        if not self._yaml:
            result['errors'].append("YAML parser not available")
            return result

        # Try to parse YAML
        try:
            data = self._yaml.safe_load(text)
            result['is_valid_yaml'] = True
            result['data'] = data
        except Exception as e:
            result['errors'].append(f"Invalid YAML: {str(e)}")
            return result

        # Check required keys
        if isinstance(data, dict):
            for key in self.required_keys:
                if key not in data:
                    result['missing_keys'].append(key)

        return result

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate YAML validity"""
        if not actual:
            return 0.0

        result = self.validate_structure(actual)

        score = 0.0

        # Valid YAML: 70%
        if result['is_valid_yaml']:
            score += 0.7

        # Required keys: 30%
        if not result['missing_keys']:
            score += 0.3

        return score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed YAML validation metrics"""
        result = self.validate_structure(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            **result,
            'required_keys': self.required_keys
        }


class SQLValidator(StructuredOutputEvaluator):
    """
    SQL query validation evaluator.

    Validates SQL syntax and structure.
    """

    def __init__(self, allowed_operations: Optional[Set[str]] = None,
                 forbidden_operations: Optional[Set[str]] = None):
        """
        Initialize SQL validator

        Args:
            allowed_operations: Set of allowed SQL operations (SELECT, INSERT, etc.)
            forbidden_operations: Set of forbidden operations (DROP, DELETE, etc.)
        """
        super().__init__(
            name="sql_validator",
            description="SQL query validator"
        )
        self.allowed_operations = allowed_operations
        self.forbidden_operations = forbidden_operations or {'DROP', 'TRUNCATE', 'DELETE'}
        self._parser = None
        self._init_parser()

    def _init_parser(self):
        """Initialize SQL parser"""
        try:
            import sqlparse
            self._parser = sqlparse
        except ImportError:
            print("Warning: sqlparse not available. Install with: pip install sqlparse")

    def validate_structure(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate SQL structure"""
        result = {
            'is_valid_sql': False,
            'operations': [],
            'has_forbidden_ops': False,
            'follows_allowed_ops': True,
            'errors': []
        }

        if not text:
            result['errors'].append("Empty SQL query")
            return result

        # Basic SQL validation
        if self._parser:
            try:
                parsed = self._parser.parse(text)
                if parsed:
                    result['is_valid_sql'] = True

                    # Extract statement types
                    for statement in parsed:
                        stmt_type = statement.get_type()
                        result['operations'].append(stmt_type)
            except Exception as e:
                result['errors'].append(f"SQL parse error: {str(e)}")
                return result
        else:
            # Fallback: simple regex-based validation
            sql_keywords = r'\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\b'
            if re.search(sql_keywords, text, re.IGNORECASE):
                result['is_valid_sql'] = True

                # Extract operations
                matches = re.findall(sql_keywords, text, re.IGNORECASE)
                result['operations'] = [m.upper() for m in matches]

        # Check forbidden operations
        for op in result['operations']:
            if op.upper() in self.forbidden_operations:
                result['has_forbidden_ops'] = True
                result['errors'].append(f"Forbidden operation: {op}")

        # Check allowed operations
        if self.allowed_operations:
            for op in result['operations']:
                if op.upper() not in self.allowed_operations:
                    result['follows_allowed_ops'] = False
                    result['errors'].append(f"Operation not allowed: {op}")

        return result

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate SQL validity"""
        if not actual:
            return 0.0

        result = self.validate_structure(actual)

        score = 0.0

        # Valid SQL: 40%
        if result['is_valid_sql']:
            score += 0.4

        # No forbidden operations: 40%
        if not result['has_forbidden_ops']:
            score += 0.4

        # Follows allowed operations: 20%
        if result['follows_allowed_ops']:
            score += 0.2

        return score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed SQL validation metrics"""
        result = self.validate_structure(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            **result,
            'allowed_operations': list(self.allowed_operations) if self.allowed_operations else None,
            'forbidden_operations': list(self.forbidden_operations)
        }


class OpenAPIValidator(StructuredOutputEvaluator):
    """
    OpenAPI/Swagger schema validator.

    Validates API responses against OpenAPI specifications.
    """

    def __init__(self, openapi_spec: Optional[Dict[str, Any]] = None):
        """
        Initialize OpenAPI validator

        Args:
            openapi_spec: OpenAPI specification dictionary
        """
        super().__init__(
            name="openapi_validator",
            description="OpenAPI/Swagger schema validator"
        )
        self.openapi_spec = openapi_spec

    def validate_structure(self, text: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate against OpenAPI schema"""
        result = {
            'is_valid_json': False,
            'matches_schema': False,
            'errors': []
        }

        # Parse JSON response
        try:
            data = json.loads(text)
            result['is_valid_json'] = True
        except json.JSONDecodeError as e:
            result['errors'].append(f"Invalid JSON: {str(e)}")
            return result

        # Get schema from spec or context
        spec = schema or self.openapi_spec
        if not spec:
            result['errors'].append("No OpenAPI spec provided")
            return result

        # Extract response schema from OpenAPI spec
        # This is a simplified version - full validation would use openapi-spec-validator
        if 'components' in spec and 'schemas' in spec['components']:
            # Validate against schema
            try:
                # Use JSONValidator for schema validation
                validator = JSONValidator(schema=spec)
                validation = validator.validate_structure(text, spec)
                result['matches_schema'] = validation['is_valid_schema']
                if not result['matches_schema']:
                    result['errors'].extend(validation['errors'])
            except Exception as e:
                result['errors'].append(f"Schema validation error: {str(e)}")

        return result

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate OpenAPI compliance"""
        if not actual:
            return 0.0

        result = self.validate_structure(actual)

        score = 0.0

        # Valid JSON: 30%
        if result['is_valid_json']:
            score += 0.3

        # Matches schema: 70%
        if result['matches_schema']:
            score += 0.7

        return score

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed OpenAPI validation metrics"""
        result = self.validate_structure(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            **result,
            'has_spec': self.openapi_spec is not None
        }


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Structured Output Evaluator Examples")
    print("=" * 80)

    # Example 1: JSON Validation
    print("\n1. JSON Validation")
    print("-" * 80)

    json_schema = {
        "type": "object",
        "required": ["name", "email"],
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "age": {"type": "integer"}
        }
    }

    json_validator = JSONValidator(
        schema=json_schema,
        required_fields=["name", "email"]
    )

    valid_json = '{"name": "John Doe", "email": "john@example.com", "age": 30}'
    invalid_json = '{"name": "John Doe"}'  # Missing email
    malformed_json = '{"name": "John Doe", email: invalid}'

    print("\nValid JSON:")
    metrics = json_validator.get_metrics(valid_json, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Valid JSON: {metrics['is_valid_json']}")
    print(f"Valid Schema: {metrics['is_valid_schema']}")

    print("\nInvalid JSON (missing field):")
    metrics = json_validator.get_metrics(invalid_json, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Missing fields: {metrics['missing_fields']}")

    print("\nMalformed JSON:")
    metrics = json_validator.get_metrics(malformed_json, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Errors: {metrics['errors']}")

    # Example 2: SQL Validation
    print("\n\n2. SQL Validation")
    print("-" * 80)

    sql_validator = SQLValidator(
        allowed_operations={'SELECT', 'INSERT', 'UPDATE'},
        forbidden_operations={'DROP', 'DELETE', 'TRUNCATE'}
    )

    safe_query = "SELECT * FROM users WHERE id = 1"
    unsafe_query = "DROP TABLE users"
    invalid_query = "SELCT * FORM users"

    print("\nSafe SQL Query:")
    metrics = sql_validator.get_metrics(safe_query, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Operations: {metrics['operations']}")
    print(f"Has forbidden ops: {metrics['has_forbidden_ops']}")

    print("\nUnsafe SQL Query:")
    metrics = sql_validator.get_metrics(unsafe_query, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Operations: {metrics['operations']}")
    print(f"Errors: {metrics['errors']}")

    # Example 3: YAML Validation
    print("\n\n3. YAML Validation")
    print("-" * 80)

    yaml_validator = YAMLValidator(
        required_keys=['name', 'version', 'dependencies']
    )

    valid_yaml = """
name: my-app
version: 1.0.0
dependencies:
  - package1
  - package2
"""

    invalid_yaml = """
name: my-app
version: 1.0.0
# Missing dependencies
"""

    print("\nValid YAML:")
    metrics = yaml_validator.get_metrics(valid_yaml, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Valid YAML: {metrics['is_valid_yaml']}")

    print("\nInvalid YAML (missing keys):")
    metrics = yaml_validator.get_metrics(invalid_yaml, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Missing keys: {metrics['missing_keys']}")

    print("\n" + "=" * 80)
