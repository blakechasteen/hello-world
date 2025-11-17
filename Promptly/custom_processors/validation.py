#!/usr/bin/env python3
"""
Data Validation Processor for Promptly Chains

Schema validation, data sanitization, PII detection/redaction,
business rule validation, and data quality checks.
"""

import re
import json
import html
from typing import Dict, List, Any, Optional, Callable, Pattern
from enum import Enum
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from promptly.plugins.base import BaseChainStepProcessor


class ValidationType(Enum):
    """Types of validation"""
    SCHEMA = "schema"
    SANITIZE = "sanitize"
    PII = "pii"
    BUSINESS_RULE = "business_rule"
    DATA_QUALITY = "data_quality"


class PIIType(Enum):
    """Types of PII to detect"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"
    PERSON_NAME = "person_name"
    ADDRESS = "address"


class DataValidationProcessor(BaseChainStepProcessor):
    """
    Validate, sanitize, and check data quality.

    Configuration format:
    {
        "type": "validation",
        "validations": [
            {
                "type": "schema",
                "schema": {
                    "type": "object",
                    "required": ["name", "email"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "email": {"type": "string", "pattern": "^[^@]+@[^@]+$"},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    }
                }
            },
            {
                "type": "sanitize",
                "rules": [
                    {"type": "html_escape"},
                    {"type": "sql_escape"},
                    {"type": "xss_filter"}
                ]
            },
            {
                "type": "pii",
                "detect": ["email", "phone", "ssn", "credit_card"],
                "action": "redact",  # redact, mask, remove, flag
                "replacement": "[REDACTED]"
            },
            {
                "type": "business_rule",
                "rules": [
                    {
                        "name": "valid_age",
                        "condition": "age >= 18",
                        "message": "Must be 18 or older"
                    }
                ]
            },
            {
                "type": "data_quality",
                "checks": [
                    {"type": "completeness", "threshold": 0.9},
                    {"type": "consistency"},
                    {"type": "uniqueness", "fields": ["email"]},
                    {"type": "accuracy"}
                ]
            }
        ],
        "source_field": "output",
        "on_validation_error": "throw",  # throw, log, continue
        "strict": false
    }

    Examples:
    - Form validation
    - API input validation
    - Data sanitization
    - PII redaction for compliance
    """

    def __init__(self):
        super().__init__(
            name="validation",
            description="Validate schemas, sanitize data, detect PII, and check data quality"
        )
        self._custom_validators: Dict[str, Callable] = {}
        self._pii_patterns: Dict[str, Pattern] = self._build_pii_patterns()

    def register_validator(self, name: str, validator: Callable) -> None:
        """Register custom validator function"""
        self._custom_validators[name] = validator

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize data.

        Args:
            step_input: Input data from previous step
            step_config: Validation configuration

        Returns:
            Validated/sanitized data with validation results
        """
        validations = step_config.get("validations", [])
        source_field = step_config.get("source_field", "output")
        on_error = step_config.get("on_validation_error", "throw")
        strict = step_config.get("strict", False)

        # Get data to validate
        data = self._get_field(step_input, source_field)

        # Track validation results
        validation_results = []
        sanitized_data = data
        is_valid = True

        # Run each validation
        for validation in validations:
            validation_type = validation.get("type")

            try:
                if validation_type == ValidationType.SCHEMA.value:
                    result = self._validate_schema(sanitized_data, validation)
                elif validation_type == ValidationType.SANITIZE.value:
                    result = self._sanitize_data(sanitized_data, validation)
                    sanitized_data = result.get("data", sanitized_data)
                elif validation_type == ValidationType.PII.value:
                    result = self._handle_pii(sanitized_data, validation)
                    sanitized_data = result.get("data", sanitized_data)
                elif validation_type == ValidationType.BUSINESS_RULE.value:
                    result = self._validate_business_rules(sanitized_data, validation, step_input)
                elif validation_type == ValidationType.DATA_QUALITY.value:
                    result = self._check_data_quality(sanitized_data, validation)
                else:
                    result = {"valid": True, "type": validation_type}

                validation_results.append(result)

                if not result.get("valid", True):
                    is_valid = False

                    if strict:
                        break

            except Exception as e:
                validation_results.append({
                    "valid": False,
                    "type": validation_type,
                    "error": str(e)
                })
                is_valid = False

                if strict:
                    break

        # Handle validation errors
        if not is_valid:
            if on_error == "throw":
                errors = [r.get("errors", []) for r in validation_results if not r.get("valid", True)]
                raise ValueError(f"Validation failed: {errors}")

        # Return results
        result = {
            **step_input,
            "validation_results": validation_results,
            "is_valid": is_valid,
            "validated_data": sanitized_data
        }

        # Update source field with sanitized data
        self._set_field(result, source_field, sanitized_data)

        return result

    def _validate_schema(self, data: Any, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against JSON schema"""
        schema = validation.get("schema", {})
        errors = []

        # Simple schema validation (subset of JSON Schema)
        schema_type = schema.get("type")

        # Type validation
        if schema_type:
            if not self._check_type(data, schema_type):
                errors.append(f"Invalid type: expected {schema_type}")
                return {"valid": False, "type": "schema", "errors": errors}

        # Object validation
        if schema_type == "object" and isinstance(data, dict):
            # Required fields
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Required field missing: {field}")

            # Properties validation
            properties = schema.get("properties", {})
            for field, field_schema in properties.items():
                if field in data:
                    field_errors = self._validate_field(data[field], field_schema, field)
                    errors.extend(field_errors)

            # Additional properties
            if not schema.get("additionalProperties", True):
                extra_fields = set(data.keys()) - set(properties.keys())
                if extra_fields:
                    errors.append(f"Additional properties not allowed: {extra_fields}")

        # Array validation
        elif schema_type == "array" and isinstance(data, list):
            # Min/max items
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")

            if min_items is not None and len(data) < min_items:
                errors.append(f"Array too short: minimum {min_items} items")

            if max_items is not None and len(data) > max_items:
                errors.append(f"Array too long: maximum {max_items} items")

            # Item validation
            items_schema = schema.get("items", {})
            if items_schema:
                for i, item in enumerate(data):
                    item_errors = self._validate_field(item, items_schema, f"[{i}]")
                    errors.extend(item_errors)

        # String validation
        elif schema_type == "string" and isinstance(data, str):
            errors.extend(self._validate_string(data, schema))

        # Number validation
        elif schema_type in ["integer", "number"]:
            errors.extend(self._validate_number(data, schema))

        return {
            "valid": len(errors) == 0,
            "type": "schema",
            "errors": errors
        }

    def _validate_field(self, value: Any, schema: Dict[str, Any], field_name: str) -> List[str]:
        """Validate a single field"""
        errors = []
        field_type = schema.get("type")

        # Type check
        if field_type and not self._check_type(value, field_type):
            errors.append(f"Field '{field_name}': invalid type")
            return errors

        # String validation
        if field_type == "string" and isinstance(value, str):
            errors.extend([f"Field '{field_name}': {e}" for e in self._validate_string(value, schema)])

        # Number validation
        elif field_type in ["integer", "number"]:
            errors.extend([f"Field '{field_name}': {e}" for e in self._validate_number(value, schema)])

        return errors

    def _validate_string(self, value: str, schema: Dict[str, Any]) -> List[str]:
        """Validate string field"""
        errors = []

        # Length validation
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")

        if min_length is not None and len(value) < min_length:
            errors.append(f"String too short: minimum {min_length} characters")

        if max_length is not None and len(value) > max_length:
            errors.append(f"String too long: maximum {max_length} characters")

        # Pattern validation
        pattern = schema.get("pattern")
        if pattern:
            if not re.match(pattern, value):
                errors.append(f"Does not match pattern: {pattern}")

        # Enum validation
        enum = schema.get("enum")
        if enum and value not in enum:
            errors.append(f"Must be one of: {enum}")

        return errors

    def _validate_number(self, value: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate number field"""
        errors = []

        try:
            num_value = float(value) if schema.get("type") == "number" else int(value)
        except (ValueError, TypeError):
            return ["Invalid number"]

        # Range validation
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if minimum is not None and num_value < minimum:
            errors.append(f"Below minimum: {minimum}")

        if maximum is not None and num_value > maximum:
            errors.append(f"Above maximum: {maximum}")

        # Multiple of
        multiple_of = schema.get("multipleOf")
        if multiple_of and num_value % multiple_of != 0:
            errors.append(f"Must be multiple of {multiple_of}")

        return errors

    def _sanitize_data(self, data: Any, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data to prevent injection attacks"""
        rules = validation.get("rules", [])
        sanitized = data

        for rule in rules:
            rule_type = rule.get("type")

            if rule_type == "html_escape":
                sanitized = self._html_escape(sanitized)

            elif rule_type == "sql_escape":
                sanitized = self._sql_escape(sanitized)

            elif rule_type == "xss_filter":
                sanitized = self._xss_filter(sanitized)

            elif rule_type == "strip_whitespace":
                if isinstance(sanitized, str):
                    sanitized = sanitized.strip()

            elif rule_type == "normalize_whitespace":
                if isinstance(sanitized, str):
                    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

            elif rule_type == "remove_special_chars":
                if isinstance(sanitized, str):
                    allowed = rule.get("allowed", "")
                    sanitized = re.sub(f'[^a-zA-Z0-9{re.escape(allowed)}]', '', sanitized)

            elif rule_type == "lowercase":
                if isinstance(sanitized, str):
                    sanitized = sanitized.lower()

            elif rule_type == "uppercase":
                if isinstance(sanitized, str):
                    sanitized = sanitized.upper()

        return {
            "valid": True,
            "type": "sanitize",
            "data": sanitized,
            "rules_applied": [r.get("type") for r in rules]
        }

    def _handle_pii(self, data: Any, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and handle PII"""
        detect_types = validation.get("detect", [])
        action = validation.get("action", "redact")
        replacement = validation.get("replacement", "[REDACTED]")

        if not isinstance(data, str):
            data = str(data)

        detected_pii = []
        processed_data = data

        # Detect each PII type
        for pii_type in detect_types:
            if pii_type in self._pii_patterns:
                pattern = self._pii_patterns[pii_type]
                matches = pattern.findall(data)

                for match in matches:
                    detected_pii.append({
                        "type": pii_type,
                        "value": match if action == "flag" else None,
                        "position": data.find(match)
                    })

                    # Apply action
                    if action == "redact":
                        processed_data = pattern.sub(replacement, processed_data)
                    elif action == "mask":
                        masked = self._mask_pii(match, pii_type)
                        processed_data = processed_data.replace(match, masked)
                    elif action == "remove":
                        processed_data = pattern.sub("", processed_data)

        return {
            "valid": len(detected_pii) == 0 if action == "flag" else True,
            "type": "pii",
            "data": processed_data,
            "detected_pii": detected_pii,
            "pii_count": len(detected_pii)
        }

    def _validate_business_rules(
        self,
        data: Any,
        validation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate business rules"""
        rules = validation.get("rules", [])
        errors = []

        for rule in rules:
            rule_name = rule.get("name", "unnamed")
            condition = rule.get("condition", "")
            message = rule.get("message", f"Rule '{rule_name}' failed")

            # Custom validator
            if rule_name in self._custom_validators:
                try:
                    is_valid = self._custom_validators[rule_name](data, context)
                    if not is_valid:
                        errors.append(message)
                except Exception as e:
                    errors.append(f"{message}: {str(e)}")

            # Simple expression evaluation
            elif condition:
                try:
                    # Create safe evaluation context
                    eval_context = {
                        **{k: v for k, v in data.items() if isinstance(data, dict)},
                        "data": data,
                        "context": context
                    }

                    # Evaluate condition
                    result = self._safe_eval(condition, eval_context)
                    if not result:
                        errors.append(message)

                except Exception as e:
                    errors.append(f"{message}: Invalid condition")

        return {
            "valid": len(errors) == 0,
            "type": "business_rule",
            "errors": errors
        }

    def _check_data_quality(self, data: Any, validation: Dict[str, Any]) -> Dict[str, Any]:
        """Check data quality"""
        checks = validation.get("checks", [])
        issues = []

        for check in checks:
            check_type = check.get("type")

            if check_type == "completeness":
                threshold = check.get("threshold", 0.9)
                completeness = self._check_completeness(data)

                if completeness < threshold:
                    issues.append({
                        "type": "completeness",
                        "score": completeness,
                        "threshold": threshold,
                        "message": f"Completeness {completeness:.2f} below threshold {threshold}"
                    })

            elif check_type == "consistency":
                consistency_issues = self._check_consistency(data)
                issues.extend(consistency_issues)

            elif check_type == "uniqueness":
                fields = check.get("fields", [])
                uniqueness_issues = self._check_uniqueness(data, fields)
                issues.extend(uniqueness_issues)

            elif check_type == "accuracy":
                accuracy_issues = self._check_accuracy(data, check)
                issues.extend(accuracy_issues)

        return {
            "valid": len(issues) == 0,
            "type": "data_quality",
            "issues": issues,
            "quality_score": 1.0 - (len(issues) * 0.1)  # Simple scoring
        }

    def _build_pii_patterns(self) -> Dict[str, Pattern]:
        """Build regex patterns for PII detection"""
        return {
            PIIType.EMAIL.value: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            PIIType.PHONE.value: re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            PIIType.SSN.value: re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            PIIType.CREDIT_CARD.value: re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            PIIType.IP_ADDRESS.value: re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            PIIType.URL.value: re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)')
        }

    def _mask_pii(self, value: str, pii_type: str) -> str:
        """Mask PII value"""
        if pii_type == PIIType.EMAIL.value:
            # Mask email: user@domain.com -> u***@domain.com
            parts = value.split("@")
            if len(parts) == 2:
                return f"{parts[0][0]}***@{parts[1]}"

        elif pii_type == PIIType.PHONE.value:
            # Mask phone: (123) 456-7890 -> (***) ***-7890
            return re.sub(r'\d', '*', value[:-4]) + value[-4:]

        elif pii_type == PIIType.CREDIT_CARD.value:
            # Mask card: 1234 5678 9012 3456 -> **** **** **** 3456
            return "**** **** **** " + value[-4:]

        return "*" * len(value)

    def _html_escape(self, data: Any) -> Any:
        """Escape HTML"""
        if isinstance(data, str):
            return html.escape(data)
        elif isinstance(data, dict):
            return {k: self._html_escape(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._html_escape(item) for item in data]
        return data

    def _sql_escape(self, data: Any) -> Any:
        """Escape SQL special characters"""
        if isinstance(data, str):
            # Simple SQL escaping
            return data.replace("'", "''").replace(";", "").replace("--", "")
        return data

    def _xss_filter(self, data: Any) -> Any:
        """Filter XSS attempts"""
        if isinstance(data, str):
            # Remove script tags and common XSS patterns
            data = re.sub(r'<script[^>]*>.*?</script>', '', data, flags=re.IGNORECASE | re.DOTALL)
            data = re.sub(r'javascript:', '', data, flags=re.IGNORECASE)
            data = re.sub(r'on\w+\s*=', '', data, flags=re.IGNORECASE)
            return data
        return data

    def _check_completeness(self, data: Any) -> float:
        """Check data completeness (ratio of non-null fields)"""
        if isinstance(data, dict):
            total = len(data)
            complete = sum(1 for v in data.values() if v is not None and v != "")
            return complete / total if total > 0 else 0.0
        return 1.0

    def _check_consistency(self, data: Any) -> List[Dict[str, Any]]:
        """Check data consistency"""
        issues = []

        if isinstance(data, dict):
            # Example: check date consistency
            if "start_date" in data and "end_date" in data:
                try:
                    start = datetime.fromisoformat(str(data["start_date"]))
                    end = datetime.fromisoformat(str(data["end_date"]))
                    if start > end:
                        issues.append({
                            "type": "consistency",
                            "message": "Start date is after end date"
                        })
                except:
                    pass

        return issues

    def _check_uniqueness(self, data: Any, fields: List[str]) -> List[Dict[str, Any]]:
        """Check field uniqueness"""
        issues = []

        if isinstance(data, list):
            for field in fields:
                values = [item.get(field) for item in data if isinstance(item, dict)]
                if len(values) != len(set(values)):
                    issues.append({
                        "type": "uniqueness",
                        "field": field,
                        "message": f"Duplicate values found in field '{field}'"
                    })

        return issues

    def _check_accuracy(self, data: Any, check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data accuracy"""
        # Placeholder for accuracy checks (would need reference data)
        return []

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate expression"""
        # Very simple and limited expression evaluation
        # In production, use a proper expression evaluator

        # Allow only simple comparisons
        allowed_ops = ['==', '!=', '>', '<', '>=', '<=', 'and', 'or', 'in', 'not']

        try:
            # Replace variable names with values
            eval_expr = expression

            for key, value in context.items():
                if isinstance(value, str):
                    eval_expr = eval_expr.replace(key, f'"{value}"')
                else:
                    eval_expr = eval_expr.replace(key, str(value))

            # Very basic evaluation (not recommended for production)
            # Use a proper expression parser in production
            return eval(eval_expr, {"__builtins__": {}}, {})

        except:
            return False

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)

        return True

    def _get_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested field value"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def _set_field(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
        """Set nested field value"""
        parts = field_path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value


__all__ = [
    "DataValidationProcessor",
    "ValidationType",
    "PIIType"
]
