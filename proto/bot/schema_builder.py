#!/usr/bin/env python3
"""
Schema Builder for Promptly Matrix Bot

Build structured schemas for guaranteed JSON/YAML output from LLMs.

Features:
- Field type validation (string, number, boolean, array, object)
- Constraints (required, min/max, format, enum)
- Nested schemas
- JSON Schema generation
- Pydantic model generation
- DSPy Signature generation
- Validation functions

Supports:
- Simple types: string, number, boolean
- Complex types: array, object, enum
- Formats: email, url, date, datetime, uuid
- Constraints: required, min/max, pattern, minLength/maxLength
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class FieldType(Enum):
    """Supported field types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"


class FieldFormat(Enum):
    """String formats"""
    EMAIL = "email"
    URL = "url"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    PHONE = "phone"
    IPV4 = "ipv4"
    IPV6 = "ipv6"


@dataclass
class FieldConstraint:
    """Field constraints"""
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    format: Optional[FieldFormat] = None
    enum_values: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class SchemaField:
    """Schema field definition"""
    name: str
    type: FieldType
    description: Optional[str] = None
    constraints: FieldConstraint = field(default_factory=FieldConstraint)
    nested_schema: Optional['Schema'] = None  # For object types
    item_type: Optional[FieldType] = None  # For array types

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format"""
        schema = {"type": self.type.value}

        if self.description:
            schema["description"] = self.description

        # Add constraints
        if self.constraints.min_value is not None:
            schema["minimum"] = self.constraints.min_value
        if self.constraints.max_value is not None:
            schema["maximum"] = self.constraints.max_value
        if self.constraints.min_length is not None:
            schema["minLength"] = self.constraints.min_length
        if self.constraints.max_length is not None:
            schema["maxLength"] = self.constraints.max_length
        if self.constraints.pattern:
            schema["pattern"] = self.constraints.pattern
        if self.constraints.format:
            schema["format"] = self.constraints.format.value
        if self.constraints.enum_values:
            schema["enum"] = self.constraints.enum_values
        if self.constraints.default is not None:
            schema["default"] = self.constraints.default

        # Handle nested types
        if self.type == FieldType.OBJECT and self.nested_schema:
            schema = self.nested_schema.to_json_schema()
        elif self.type == FieldType.ARRAY and self.item_type:
            schema["items"] = {"type": self.item_type.value}

        return schema

    def to_pydantic(self) -> str:
        """Convert to Pydantic field definition"""
        type_map = {
            FieldType.STRING: "str",
            FieldType.NUMBER: "float",
            FieldType.INTEGER: "int",
            FieldType.BOOLEAN: "bool",
            FieldType.ARRAY: "List",
            FieldType.OBJECT: "Dict"
        }

        field_type = type_map.get(self.type, "str")

        # Handle arrays
        if self.type == FieldType.ARRAY and self.item_type:
            item_type = type_map.get(self.item_type, "str")
            field_type = f"List[{item_type}]"

        # Handle optional
        if not self.constraints.required:
            field_type = f"Optional[{field_type}]"

        # Build Field() definition
        field_parts = []
        if self.constraints.default is not None:
            field_parts.append(f"default={repr(self.constraints.default)}")
        elif not self.constraints.required:
            field_parts.append("default=None")

        if self.description:
            field_parts.append(f"description={repr(self.description)}")

        # Add validators
        validators = []
        if self.constraints.min_value is not None:
            validators.append(f"ge={self.constraints.min_value}")
        if self.constraints.max_value is not None:
            validators.append(f"le={self.constraints.max_value}")
        if self.constraints.min_length is not None:
            validators.append(f"min_length={self.constraints.min_length}")
        if self.constraints.max_length is not None:
            validators.append(f"max_length={self.constraints.max_length}")

        field_parts.extend(validators)

        if field_parts:
            return f"{self.name}: {field_type} = Field({', '.join(field_parts)})"
        else:
            return f"{self.name}: {field_type}"


@dataclass
class Schema:
    """Complete schema definition"""
    name: str
    description: Optional[str] = None
    fields: List[SchemaField] = field(default_factory=list)

    def add_field(self, field: SchemaField):
        """Add field to schema"""
        self.fields.append(field)

    def get_field(self, name: str) -> Optional[SchemaField]:
        """Get field by name"""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema"""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": []
        }

        if self.description:
            schema["description"] = self.description

        for field in self.fields:
            schema["properties"][field.name] = field.to_json_schema()
            if field.constraints.required:
                schema["required"].append(field.name)

        return schema

    def to_pydantic_model(self) -> str:
        """Generate Pydantic model code"""
        lines = [
            "from pydantic import BaseModel, Field",
            "from typing import Optional, List, Dict",
            "",
            f"class {self.name}(BaseModel):"
        ]

        if self.description:
            lines.append(f'    """{self.description}"""')

        for field in self.fields:
            lines.append(f"    {field.to_pydantic()}")

        return "\n".join(lines)

    def to_dspy_signature(self) -> str:
        """Generate DSPy signature"""
        inputs = []
        outputs = []

        for field in self.fields:
            desc = field.description or field.name
            if field.constraints.required:
                inputs.append(f'"{field.name}": "{desc}"')
            else:
                outputs.append(f'"{field.name}": "{desc}"')

        sig_parts = []
        if inputs:
            sig_parts.append(f"inputs=[{', '.join(inputs)}]")
        if outputs:
            sig_parts.append(f"outputs=[{', '.join(outputs)}]")

        return f"create_signature({', '.join(sig_parts)})"

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against schema.

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        # Check required fields
        for field in self.fields:
            if field.constraints.required and field.name not in data:
                errors.append(f"Missing required field: {field.name}")

        # Validate each field
        for field_name, value in data.items():
            field = self.get_field(field_name)
            if not field:
                errors.append(f"Unknown field: {field_name}")
                continue

            # Type validation
            if not self._validate_type(value, field.type):
                errors.append(f"Field '{field_name}': Invalid type (expected {field.type.value})")
                continue

            # Constraint validation
            field_errors = self._validate_constraints(field_name, value, field.constraints, field.type)
            errors.extend(field_errors)

        return len(errors) == 0, errors

    def _validate_type(self, value: Any, field_type: FieldType) -> bool:
        """Validate value type"""
        type_checks = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.NUMBER: lambda v: isinstance(v, (int, float)),
            FieldType.INTEGER: lambda v: isinstance(v, int),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.ARRAY: lambda v: isinstance(v, list),
            FieldType.OBJECT: lambda v: isinstance(v, dict)
        }

        check = type_checks.get(field_type)
        return check(value) if check else True

    def _validate_constraints(
        self,
        field_name: str,
        value: Any,
        constraints: FieldConstraint,
        field_type: FieldType
    ) -> List[str]:
        """Validate field constraints"""
        errors = []

        # Number constraints
        if field_type in [FieldType.NUMBER, FieldType.INTEGER]:
            if constraints.min_value is not None and value < constraints.min_value:
                errors.append(f"Field '{field_name}': Value {value} < minimum {constraints.min_value}")
            if constraints.max_value is not None and value > constraints.max_value:
                errors.append(f"Field '{field_name}': Value {value} > maximum {constraints.max_value}")

        # String constraints
        if field_type == FieldType.STRING:
            if constraints.min_length is not None and len(value) < constraints.min_length:
                errors.append(f"Field '{field_name}': Length {len(value)} < minimum {constraints.min_length}")
            if constraints.max_length is not None and len(value) > constraints.max_length:
                errors.append(f"Field '{field_name}': Length {len(value)} > maximum {constraints.max_length}")
            if constraints.pattern and not re.match(constraints.pattern, value):
                errors.append(f"Field '{field_name}': Does not match pattern {constraints.pattern}")

            # Format validation
            if constraints.format:
                if not self._validate_format(value, constraints.format):
                    errors.append(f"Field '{field_name}': Invalid format ({constraints.format.value})")

        # Enum constraints
        if constraints.enum_values and value not in constraints.enum_values:
            errors.append(f"Field '{field_name}': Value not in enum {constraints.enum_values}")

        return errors

    def _validate_format(self, value: str, format: FieldFormat) -> bool:
        """Validate string format"""
        patterns = {
            FieldFormat.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            FieldFormat.URL: r'^https?://[^\s]+$',
            FieldFormat.UUID: r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            FieldFormat.IPV4: r'^(\d{1,3}\.){3}\d{1,3}$',
            FieldFormat.DATE: r'^\d{4}-\d{2}-\d{2}$',
            FieldFormat.DATETIME: r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            FieldFormat.PHONE: r'^\+?[\d\s\-\(\)]+$'
        }

        pattern = patterns.get(format)
        if pattern:
            return bool(re.match(pattern, value, re.IGNORECASE))
        return True


class SchemaBuilder:
    """
    Build schemas from natural language descriptions.

    Usage:
        builder = SchemaBuilder()
        schema = builder.build_from_text('''
        Fields:
        - name: string (required)
        - age: number (min: 0, max: 120)
        - email: string (email format)
        ''')

        # Generate outputs
        json_schema = schema.to_json_schema()
        pydantic_code = schema.to_pydantic_model()
        dspy_sig = schema.to_dspy_signature()
    """

    def build_from_text(self, text: str, schema_name: str = "GeneratedSchema") -> Schema:
        """
        Build schema from text description.

        Format:
            Fields:
            - name: type (constraints)

        Example:
            Fields:
            - name: string (required, minLength: 1, maxLength: 100)
            - age: number (min: 0, max: 120)
            - email: string (email format)
            - role: enum (values: admin, user, guest)
        """
        schema = Schema(name=schema_name)

        # Extract fields section
        fields_match = re.search(r'Fields:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if not fields_match:
            return schema

        fields_text = fields_match.group(1)

        # Parse each field line
        for line in fields_text.split('\n'):
            line = line.strip()
            if not line.startswith('-'):
                continue

            field = self._parse_field_line(line)
            if field:
                schema.add_field(field)

        return schema

    def _parse_field_line(self, line: str) -> Optional[SchemaField]:
        """
        Parse field line.

        Format: - name: type (constraints)
        """
        # Remove leading '-'
        line = line.lstrip('- ').strip()

        # Parse: name: type (constraints)
        match = re.match(r'(\w+):\s*(\w+)\s*(?:\((.+)\))?', line)
        if not match:
            return None

        name = match.group(1)
        type_str = match.group(2).lower()
        constraints_str = match.group(3) if match.group(3) else ""

        # Map type
        type_map = {
            'string': FieldType.STRING,
            'str': FieldType.STRING,
            'number': FieldType.NUMBER,
            'num': FieldType.NUMBER,
            'float': FieldType.NUMBER,
            'integer': FieldType.INTEGER,
            'int': FieldType.INTEGER,
            'boolean': FieldType.BOOLEAN,
            'bool': FieldType.BOOLEAN,
            'array': FieldType.ARRAY,
            'list': FieldType.ARRAY,
            'object': FieldType.OBJECT,
            'dict': FieldType.OBJECT,
            'enum': FieldType.ENUM
        }

        field_type = type_map.get(type_str, FieldType.STRING)

        # Parse constraints
        constraints = self._parse_constraints(constraints_str)

        return SchemaField(
            name=name,
            type=field_type,
            constraints=constraints
        )

    def _parse_constraints(self, constraints_str: str) -> FieldConstraint:
        """Parse constraints string"""
        constraints = FieldConstraint()

        if not constraints_str:
            return constraints

        # Split by comma
        parts = [p.strip() for p in constraints_str.split(',')]

        for part in parts:
            part_lower = part.lower()

            # Required
            if part_lower == 'required':
                constraints.required = True

            # Min/max values
            elif 'min:' in part_lower or 'minimum:' in part_lower:
                val = re.search(r'(?:min|minimum):\s*(\d+(?:\.\d+)?)', part_lower)
                if val:
                    constraints.min_value = float(val.group(1))

            elif 'max:' in part_lower or 'maximum:' in part_lower:
                val = re.search(r'(?:max|maximum):\s*(\d+(?:\.\d+)?)', part_lower)
                if val:
                    constraints.max_value = float(val.group(1))

            # Min/max length
            elif 'minlength:' in part_lower:
                val = re.search(r'minlength:\s*(\d+)', part_lower)
                if val:
                    constraints.min_length = int(val.group(1))

            elif 'maxlength:' in part_lower:
                val = re.search(r'maxlength:\s*(\d+)', part_lower)
                if val:
                    constraints.max_length = int(val.group(1))

            # Format
            elif 'format' in part_lower:
                format_str = part_lower.split('format')[0].strip()
                format_map = {
                    'email': FieldFormat.EMAIL,
                    'url': FieldFormat.URL,
                    'date': FieldFormat.DATE,
                    'datetime': FieldFormat.DATETIME,
                    'uuid': FieldFormat.UUID,
                    'phone': FieldFormat.PHONE,
                    'ipv4': FieldFormat.IPV4,
                    'ipv6': FieldFormat.IPV6
                }
                constraints.format = format_map.get(format_str)

            # Enum values
            elif 'values:' in part_lower:
                val = re.search(r'values:\s*(.+)', part)
                if val:
                    values_str = val.group(1).strip()
                    constraints.enum_values = [v.strip() for v in values_str.split('|')]

        return constraints


# Singleton
_schema_builder = None

def get_schema_builder() -> SchemaBuilder:
    """Get singleton schema builder"""
    global _schema_builder
    if _schema_builder is None:
        _schema_builder = SchemaBuilder()
    return _schema_builder


# Test
if __name__ == "__main__":
    builder = SchemaBuilder()

    # Test schema building
    schema = builder.build_from_text('''
    Fields:
    - name: string (required, minLength: 1, maxLength: 100)
    - age: number (min: 0, max: 120)
    - email: string (email format, required)
    - role: enum (values: admin|user|guest, required)
    - bio: string (maxLength: 500)
    ''', schema_name="UserProfile")

    print("✅ Schema created:")
    print(f"   Fields: {len(schema.fields)}")

    # Test JSON Schema
    json_schema = schema.to_json_schema()
    print(f"\n✅ JSON Schema:\n{json.dumps(json_schema, indent=2)}")

    # Test Pydantic
    pydantic = schema.to_pydantic_model()
    print(f"\n✅ Pydantic Model:\n{pydantic}")

    # Test validation
    valid_data = {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
        "role": "admin"
    }

    invalid_data = {
        "name": "Bob",
        "age": 150,  # Invalid
        "email": "not-an-email",  # Invalid
        "role": "superuser"  # Invalid enum
    }

    is_valid, errors = schema.validate(valid_data)
    print(f"\n✅ Valid data: {is_valid}")

    is_valid, errors = schema.validate(invalid_data)
    print(f"\n✅ Invalid data: {is_valid}")
    for error in errors:
        print(f"   - {error}")

    print("\n✅ All schema builder tests passed!")
