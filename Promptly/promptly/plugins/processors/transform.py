#!/usr/bin/env python3
"""
Transform Processor for Promptly Chains

Implements data extraction, format conversion, template rendering, validation, and sanitization.
"""

import re
import json
import html
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from ..base import BaseChainStepProcessor


class TransformType(Enum):
    """Types of transformations"""
    EXTRACT = "extract"         # Extract data from text
    CONVERT = "convert"         # Convert between formats
    TEMPLATE = "template"       # Render template
    VALIDATE = "validate"       # Validate data
    SANITIZE = "sanitize"       # Clean/sanitize data
    NORMALIZE = "normalize"     # Normalize data
    AGGREGATE = "aggregate"     # Aggregate fields


class ExtractionMethod(Enum):
    """Methods for extracting data"""
    JSON = "json"
    REGEX = "regex"
    SPLIT = "split"
    XPATH = "xpath"
    CSV = "csv"
    KEY_VALUE = "key_value"


class TransformProcessor(BaseChainStepProcessor):
    """
    Transform and extract data.

    Configuration format:
    {
        "type": "transform",
        "transform_type": "extract|convert|template|validate|sanitize",
        "method": "json|regex|split|xpath|csv",  # for extract
        "source_field": "output",                 # field to transform
        "target_field": "result",                 # where to store result
        "pattern": "...",                         # for regex
        "template": "...",                        # for template
        "format": "json|xml|yaml|csv",           # for convert
        "validation_rules": [...],                # for validate
        "sanitize_rules": [...]                   # for sanitize
    }
    """

    def __init__(self):
        super().__init__(
            name="transform",
            description="Transform and extract data"
        )
        self._custom_validators: Dict[str, Callable] = {}
        self._custom_sanitizers: Dict[str, Callable] = {}
        self._custom_converters: Dict[str, Callable] = {}

    def register_validator(self, name: str, validator: Callable[[Any], bool]) -> None:
        """Register custom validator function"""
        self._custom_validators[name] = validator

    def register_sanitizer(self, name: str, sanitizer: Callable[[Any], Any]) -> None:
        """Register custom sanitizer function"""
        self._custom_sanitizers[name] = sanitizer

    def register_converter(self, name: str, converter: Callable[[Any, str], Any]) -> None:
        """Register custom converter function"""
        self._custom_converters[name] = converter

    def process(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute transformation.

        Args:
            step_input: Input data
            step_config: Transformation configuration

        Returns:
            Transformed data
        """
        transform_type = step_config.get("transform_type", "extract")

        if transform_type == TransformType.EXTRACT.value:
            return self._extract(step_input, step_config)
        elif transform_type == TransformType.CONVERT.value:
            return self._convert(step_input, step_config)
        elif transform_type == TransformType.TEMPLATE.value:
            return self._template(step_input, step_config)
        elif transform_type == TransformType.VALIDATE.value:
            return self._validate(step_input, step_config)
        elif transform_type == TransformType.SANITIZE.value:
            return self._sanitize(step_input, step_config)
        elif transform_type == TransformType.NORMALIZE.value:
            return self._normalize(step_input, step_config)
        elif transform_type == TransformType.AGGREGATE.value:
            return self._aggregate(step_input, step_config)
        else:
            return step_input

    def _extract(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from text"""
        method = step_config.get("method", "regex")
        source_field = step_config.get("source_field", "output")
        target_field = step_config.get("target_field", "extracted")

        source_value = self._get_field(step_input, source_field)

        if method == ExtractionMethod.JSON.value:
            extracted = self._extract_json(source_value, step_config)
        elif method == ExtractionMethod.REGEX.value:
            extracted = self._extract_regex(source_value, step_config)
        elif method == ExtractionMethod.SPLIT.value:
            extracted = self._extract_split(source_value, step_config)
        elif method == ExtractionMethod.KEY_VALUE.value:
            extracted = self._extract_key_value(source_value, step_config)
        elif method == ExtractionMethod.CSV.value:
            extracted = self._extract_csv(source_value, step_config)
        else:
            extracted = source_value

        result = step_input.copy()
        self._set_field(result, target_field, extracted)

        return result

    def _extract_json(self, text: str, config: Dict[str, Any]) -> Any:
        """Extract JSON from text"""
        try:
            # Try to parse entire text as JSON
            data = json.loads(text)

            # Extract specific path if provided
            path = config.get("path", "")
            if path:
                return self._get_field({"data": data}, f"data.{path}")

            return data

        except json.JSONDecodeError:
            # Try to find JSON in text
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text)

            if matches:
                try:
                    return json.loads(matches[0])
                except:
                    pass

            return None

    def _extract_regex(self, text: str, config: Dict[str, Any]) -> Any:
        """Extract using regex pattern"""
        pattern = config.get("pattern", "")
        if not pattern:
            return text

        flags = 0
        if config.get("case_insensitive", False):
            flags |= re.IGNORECASE
        if config.get("multiline", False):
            flags |= re.MULTILINE
        if config.get("dotall", False):
            flags |= re.DOTALL

        # Check if we want all matches or just first
        find_all = config.get("find_all", False)

        if find_all:
            matches = re.findall(pattern, str(text), flags)
            return matches if matches else []
        else:
            match = re.search(pattern, str(text), flags)
            if match:
                # Return named groups if any, otherwise group 0
                if match.groupdict():
                    return match.groupdict()
                elif match.groups():
                    return match.groups()[0] if len(match.groups()) == 1 else match.groups()
                else:
                    return match.group(0)
            return None

    def _extract_split(self, text: str, config: Dict[str, Any]) -> List[str]:
        """Extract by splitting text"""
        delimiter = config.get("delimiter", "\n")
        max_splits = config.get("max_splits", -1)
        strip = config.get("strip", True)

        parts = str(text).split(delimiter, max_splits)

        if strip:
            parts = [p.strip() for p in parts]

        # Filter empty if requested
        if config.get("filter_empty", True):
            parts = [p for p in parts if p]

        return parts

    def _extract_key_value(self, text: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Extract key-value pairs from text"""
        separator = config.get("separator", ":")
        line_delimiter = config.get("line_delimiter", "\n")

        lines = str(text).split(line_delimiter)
        result = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if separator in line:
                parts = line.split(separator, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    result[key] = value

        return result

    def _extract_csv(self, text: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract CSV data"""
        import csv
        from io import StringIO

        delimiter = config.get("delimiter", ",")
        has_header = config.get("has_header", True)

        reader = csv.reader(StringIO(str(text)), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return []

        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
            return [dict(zip(headers, row)) for row in data_rows]
        else:
            return [{"col_" + str(i): val for i, val in enumerate(row)} for row in rows]

    def _convert(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert between formats"""
        source_field = step_config.get("source_field", "output")
        target_field = step_config.get("target_field", "converted")
        source_format = step_config.get("source_format", "auto")
        target_format = step_config.get("target_format", "json")

        source_value = self._get_field(step_input, source_field)

        # Custom converter
        converter_name = step_config.get("converter")
        if converter_name and converter_name in self._custom_converters:
            converted = self._custom_converters[converter_name](source_value, target_format)
        else:
            # Built-in conversion
            converted = self._convert_format(source_value, source_format, target_format)

        result = step_input.copy()
        self._set_field(result, target_field, converted)

        return result

    def _convert_format(self, value: Any, source_format: str, target_format: str) -> Any:
        """Convert between formats"""
        # Parse source format
        if source_format == "json" or (source_format == "auto" and isinstance(value, str)):
            try:
                data = json.loads(value)
            except:
                data = value
        else:
            data = value

        # Convert to target format
        if target_format == "json":
            return json.dumps(data, indent=2)
        elif target_format == "yaml":
            try:
                import yaml
                return yaml.dump(data, default_flow_style=False)
            except ImportError:
                return str(data)
        elif target_format == "csv":
            # Convert dict/list to CSV
            if isinstance(data, list) and data and isinstance(data[0], dict):
                import csv
                from io import StringIO
                output = StringIO()
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                return output.getvalue()
            return str(data)
        elif target_format == "string":
            return str(data)
        elif target_format == "lines":
            if isinstance(data, list):
                return "\n".join(str(item) for item in data)
            return str(data)
        else:
            return data

    def _template(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render template"""
        template = step_config.get("template", "")
        target_field = step_config.get("target_field", "rendered")
        template_engine = step_config.get("engine", "format")

        if template_engine == "format":
            # Python string formatting
            rendered = template.format(**step_input)
        elif template_engine == "fstring":
            # F-string style (unsafe - use carefully)
            rendered = template.format_map(step_input)
        else:
            # Simple replacement
            rendered = template
            for key, value in step_input.items():
                rendered = rendered.replace(f"{{{key}}}", str(value))

        result = step_input.copy()
        self._set_field(result, target_field, rendered)

        return result

    def _validate(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data"""
        rules = step_config.get("validation_rules", [])
        source_field = step_config.get("source_field", "output")

        source_value = self._get_field(step_input, source_field)

        validation_errors = []
        is_valid = True

        for rule in rules:
            rule_type = rule.get("type", "")

            if rule_type == "required":
                if source_value is None or source_value == "":
                    validation_errors.append("Value is required")
                    is_valid = False

            elif rule_type == "type":
                expected_type = rule.get("expected_type", "string")
                if not self._check_type(source_value, expected_type):
                    validation_errors.append(f"Expected type {expected_type}")
                    is_valid = False

            elif rule_type == "regex":
                pattern = rule.get("pattern", "")
                if not re.match(pattern, str(source_value)):
                    validation_errors.append(f"Does not match pattern {pattern}")
                    is_valid = False

            elif rule_type == "range":
                min_val = rule.get("min")
                max_val = rule.get("max")
                try:
                    val = float(source_value)
                    if min_val is not None and val < min_val:
                        validation_errors.append(f"Value below minimum {min_val}")
                        is_valid = False
                    if max_val is not None and val > max_val:
                        validation_errors.append(f"Value above maximum {max_val}")
                        is_valid = False
                except (ValueError, TypeError):
                    validation_errors.append("Value is not numeric")
                    is_valid = False

            elif rule_type == "custom":
                validator_name = rule.get("validator")
                if validator_name and validator_name in self._custom_validators:
                    if not self._custom_validators[validator_name](source_value):
                        validation_errors.append(f"Custom validation '{validator_name}' failed")
                        is_valid = False

        return {
            **step_input,
            "is_valid": is_valid,
            "validation_errors": validation_errors
        }

    def _sanitize(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data"""
        rules = step_config.get("sanitize_rules", [])
        source_field = step_config.get("source_field", "output")
        target_field = step_config.get("target_field", "sanitized")

        value = self._get_field(step_input, source_field)

        for rule in rules:
            rule_type = rule.get("type", "")

            if rule_type == "html_escape":
                value = html.escape(str(value))

            elif rule_type == "strip":
                value = str(value).strip()

            elif rule_type == "lowercase":
                value = str(value).lower()

            elif rule_type == "uppercase":
                value = str(value).upper()

            elif rule_type == "remove_whitespace":
                value = re.sub(r'\s+', '', str(value))

            elif rule_type == "normalize_whitespace":
                value = re.sub(r'\s+', ' ', str(value)).strip()

            elif rule_type == "remove_special_chars":
                allowed = rule.get("allowed", "")
                value = re.sub(f'[^a-zA-Z0-9{re.escape(allowed)}]', '', str(value))

            elif rule_type == "truncate":
                max_length = rule.get("max_length", 100)
                value = str(value)[:max_length]

            elif rule_type == "custom":
                sanitizer_name = rule.get("sanitizer")
                if sanitizer_name and sanitizer_name in self._custom_sanitizers:
                    value = self._custom_sanitizers[sanitizer_name](value)

        result = step_input.copy()
        self._set_field(result, target_field, value)

        return result

    def _normalize(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data"""
        source_field = step_config.get("source_field", "output")
        target_field = step_config.get("target_field", "normalized")
        method = step_config.get("method", "minmax")

        value = self._get_field(step_input, source_field)

        if method == "minmax":
            # Min-max normalization (requires numeric value)
            min_val = step_config.get("min", 0.0)
            max_val = step_config.get("max", 1.0)
            try:
                num_val = float(value)
                normalized = (num_val - min_val) / (max_val - min_val) if max_val != min_val else 0.0
            except (ValueError, TypeError):
                normalized = value

        elif method == "zscore":
            # Z-score normalization
            mean = step_config.get("mean", 0.0)
            std = step_config.get("std", 1.0)
            try:
                num_val = float(value)
                normalized = (num_val - mean) / std if std != 0 else 0.0
            except (ValueError, TypeError):
                normalized = value

        else:
            normalized = value

        result = step_input.copy()
        self._set_field(result, target_field, normalized)

        return result

    def _aggregate(self, step_input: Dict[str, Any], step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate multiple fields"""
        fields = step_config.get("fields", [])
        target_field = step_config.get("target_field", "aggregated")
        method = step_config.get("method", "concat")

        values = [self._get_field(step_input, field) for field in fields]

        if method == "concat":
            separator = step_config.get("separator", " ")
            aggregated = separator.join(str(v) for v in values if v is not None)

        elif method == "sum":
            try:
                aggregated = sum(float(v) for v in values if v is not None)
            except (ValueError, TypeError):
                aggregated = 0

        elif method == "average":
            try:
                numeric_values = [float(v) for v in values if v is not None]
                aggregated = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            except (ValueError, TypeError):
                aggregated = 0

        elif method == "list":
            aggregated = [v for v in values if v is not None]

        elif method == "dict":
            aggregated = dict(zip(fields, values))

        else:
            aggregated = values

        result = step_input.copy()
        self._set_field(result, target_field, aggregated)

        return result

    def _get_field(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get field value, supporting nested paths"""
        parts = field_path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                value = value[idx] if 0 <= idx < len(value) else None
            else:
                return None

        return value

    def _set_field(self, data: Dict[str, Any], field_path: str, value: Any) -> None:
        """Set field value, supporting nested paths"""
        parts = field_path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            "string": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }

        if expected_type in type_map:
            return isinstance(value, type_map[expected_type])

        return True


# Export processors
__all__ = ["TransformProcessor", "TransformType", "ExtractionMethod"]
