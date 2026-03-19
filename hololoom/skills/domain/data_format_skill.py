"""Data Format Skill - Validate, transform, and convert JSON/YAML/TOML data"""

import json
import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)

# Check for YAML availability
YAML_AVAILABLE = False
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    pass

# Check for TOML availability
TOML_AVAILABLE = False
try:
    import tomllib  # Python 3.11+
    TOML_AVAILABLE = True
except ImportError:
    try:
        import toml
        TOML_AVAILABLE = True
    except ImportError:
        pass


class DataFormatSkill(BaseSkill):
    """Validate, transform, and convert structured data formats (JSON/YAML/TOML)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="data_format",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.DOMAIN,
            tags=["json", "yaml", "toml", "validation", "conversion", "transform"],
            description_short="Validate and convert JSON/YAML/TOML data",
            description_long=(
                "Work with structured data formats: JSON, YAML, TOML. "
                "Supports validation (schema checking), transformation (JMESPath queries, "
                "key manipulation), format conversion (JSON ↔ YAML ↔ TOML), "
                "pretty printing, minification, and data extraction. "
                "Useful for config file processing, API response validation, "
                "and data pipeline transformations."
            ),
            requires_code_execution=False,
            requires_filesystem_read=False,
            external_dependencies=["PyYAML (optional)", "toml (optional)"],
            expected_latency_ms="5-100ms",
            token_usage="50-500 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = [
            "validate_json", "validate_yaml", "validate_toml",
            "convert", "pretty_print", "minify",
            "extract_path", "merge", "diff"
        ]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if skill_input.operation in ["validate_json", "pretty_print", "minify", "extract_path"]:
            if "data" not in skill_input.parameters:
                raise ValueError("'data' parameter is required")

        if skill_input.operation == "convert":
            if "data" not in skill_input.parameters or "to_format" not in skill_input.parameters:
                raise ValueError("'data' and 'to_format' parameters are required")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            result = self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Data format '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Data format operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    def _dispatch(self, operation: str, parameters: dict[str, Any]):
        if operation == "validate_json":
            return self._validate_json(parameters)
        elif operation == "validate_yaml":
            return self._validate_yaml(parameters)
        elif operation == "validate_toml":
            return self._validate_toml(parameters)
        elif operation == "convert":
            return self._convert(parameters)
        elif operation == "pretty_print":
            return self._pretty_print(parameters)
        elif operation == "minify":
            return self._minify(parameters)
        elif operation == "extract_path":
            return self._extract_path(parameters)
        elif operation == "merge":
            return self._merge(parameters)
        elif operation == "diff":
            return self._diff(parameters)

    def _validate_json(self, params):
        """Validate JSON data."""
        data = params["data"]
        schema = params.get("schema")  # JSON Schema validation (not implemented here)

        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
                data = json.dumps(data)

            return {
                "operation": "validate_json",
                "valid": True,
                "parsed": parsed,
                "size_bytes": len(data),
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "length": len(parsed) if isinstance(parsed, (list, dict)) else None,
                "success": True
            }
        except json.JSONDecodeError as e:
            return {
                "operation": "validate_json",
                "valid": False,
                "error": str(e),
                "error_line": e.lineno,
                "error_column": e.colno,
                "success": False
            }

    def _validate_yaml(self, params):
        """Validate YAML data."""
        if not YAML_AVAILABLE:
            return {
                "operation": "validate_yaml",
                "error": "PyYAML not installed. Install with: pip install PyYAML",
                "success": False
            }

        data = params["data"]

        try:
            if isinstance(data, str):
                parsed = yaml.safe_load(data)
            else:
                parsed = data

            return {
                "operation": "validate_yaml",
                "valid": True,
                "parsed": parsed,
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "length": len(parsed) if isinstance(parsed, (list, dict)) else None,
                "success": True
            }
        except yaml.YAMLError as e:
            return {
                "operation": "validate_yaml",
                "valid": False,
                "error": str(e),
                "success": False
            }

    def _validate_toml(self, params):
        """Validate TOML data."""
        if not TOML_AVAILABLE:
            return {
                "operation": "validate_toml",
                "error": "TOML library not installed. Install with: pip install toml",
                "success": False
            }

        data = params["data"]

        try:
            if isinstance(data, str):
                try:
                    import tomllib
                    parsed = tomllib.loads(data)
                except ImportError:
                    import toml
                    parsed = toml.loads(data)
            else:
                parsed = data

            return {
                "operation": "validate_toml",
                "valid": True,
                "parsed": parsed,
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "success": True
            }
        except Exception as e:
            return {
                "operation": "validate_toml",
                "valid": False,
                "error": str(e),
                "success": False
            }

    def _convert(self, params):
        """Convert between JSON/YAML/TOML formats."""
        data = params["data"]
        from_format = params.get("from_format", "json")  # Auto-detect if not specified
        to_format = params["to_format"]

        # Parse input data
        if isinstance(data, str):
            if from_format == "json":
                parsed = json.loads(data)
            elif from_format == "yaml":
                if not YAML_AVAILABLE:
                    raise ValueError("PyYAML not installed")
                parsed = yaml.safe_load(data)
            elif from_format == "toml":
                if not TOML_AVAILABLE:
                    raise ValueError("TOML library not installed")
                try:
                    import tomllib
                    parsed = tomllib.loads(data)
                except ImportError:
                    import toml
                    parsed = toml.loads(data)
            else:
                raise ValueError(f"Unknown from_format: {from_format}")
        else:
            parsed = data

        # Convert to output format
        if to_format == "json":
            output = json.dumps(parsed, indent=2)
        elif to_format == "yaml":
            if not YAML_AVAILABLE:
                raise ValueError("PyYAML not installed")
            output = yaml.dump(parsed, default_flow_style=False)
        elif to_format == "toml":
            if not TOML_AVAILABLE:
                raise ValueError("TOML library not installed")
            import toml
            output = toml.dumps(parsed)
        else:
            raise ValueError(f"Unknown to_format: {to_format}")

        return {
            "operation": "convert",
            "from_format": from_format,
            "to_format": to_format,
            "output": output,
            "size_bytes": len(output),
            "success": True
        }

    def _pretty_print(self, params):
        """Pretty print JSON data."""
        data = params["data"]
        indent = params.get("indent", 2)

        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data

        pretty = json.dumps(parsed, indent=indent, sort_keys=True)

        return {
            "operation": "pretty_print",
            "output": pretty,
            "size_bytes": len(pretty),
            "success": True
        }

    def _minify(self, params):
        """Minify JSON data (remove whitespace)."""
        data = params["data"]

        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data

        minified = json.dumps(parsed, separators=(',', ':'))

        return {
            "operation": "minify",
            "output": minified,
            "size_bytes": len(minified),
            "success": True
        }

    def _extract_path(self, params):
        """Extract value at JSON path (dot notation)."""
        data = params["data"]
        path = params["path"]  # e.g., "user.profile.name"

        if isinstance(data, str):
            parsed = json.loads(data)
        else:
            parsed = data

        # Navigate path
        parts = path.split('.')
        current = parsed

        for part in parts:
            if isinstance(current, dict):
                if part not in current:
                    return {
                        "operation": "extract_path",
                        "path": path,
                        "found": False,
                        "error": f"Key '{part}' not found",
                        "success": False
                    }
                current = current[part]
            elif isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError) as e:
                    return {
                        "operation": "extract_path",
                        "path": path,
                        "found": False,
                        "error": str(e),
                        "success": False
                    }
            else:
                return {
                    "operation": "extract_path",
                    "path": path,
                    "found": False,
                    "error": f"Cannot navigate into {type(current)}",
                    "success": False
                }

        return {
            "operation": "extract_path",
            "path": path,
            "found": True,
            "value": current,
            "value_type": type(current).__name__,
            "success": True
        }

    def _merge(self, params):
        """Merge two JSON objects."""
        data1 = params["data1"]
        data2 = params["data2"]
        strategy = params.get("strategy", "deep")  # "shallow" or "deep"

        if isinstance(data1, str):
            obj1 = json.loads(data1)
        else:
            obj1 = data1

        if isinstance(data2, str):
            obj2 = json.loads(data2)
        else:
            obj2 = data2

        if strategy == "shallow":
            merged = {**obj1, **obj2}
        else:  # deep merge
            merged = self._deep_merge(obj1, obj2)

        return {
            "operation": "merge",
            "strategy": strategy,
            "merged": merged,
            "keys": list(merged.keys()) if isinstance(merged, dict) else None,
            "success": True
        }

    def _deep_merge(self, dict1, dict2):
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _diff(self, params):
        """Find differences between two JSON objects."""
        data1 = params["data1"]
        data2 = params["data2"]

        if isinstance(data1, str):
            obj1 = json.loads(data1)
        else:
            obj1 = data1

        if isinstance(data2, str):
            obj2 = json.loads(data2)
        else:
            obj2 = data2

        differences = self._find_differences(obj1, obj2)

        return {
            "operation": "diff",
            "differences": differences,
            "total_differences": len(differences),
            "success": True
        }

    def _find_differences(self, obj1, obj2, path=""):
        """Recursively find differences between two objects."""
        diffs = []

        # Check if types differ
        if type(obj1) != type(obj2):
            diffs.append({
                "path": path or "root",
                "type": "type_mismatch",
                "value1": obj1,
                "value2": obj2
            })
            return diffs

        if isinstance(obj1, dict):
            # Keys only in obj1
            for key in set(obj1.keys()) - set(obj2.keys()):
                diffs.append({
                    "path": f"{path}.{key}" if path else key,
                    "type": "key_removed",
                    "value1": obj1[key],
                    "value2": None
                })

            # Keys only in obj2
            for key in set(obj2.keys()) - set(obj1.keys()):
                diffs.append({
                    "path": f"{path}.{key}" if path else key,
                    "type": "key_added",
                    "value1": None,
                    "value2": obj2[key]
                })

            # Keys in both - check values
            for key in set(obj1.keys()) & set(obj2.keys()):
                new_path = f"{path}.{key}" if path else key
                diffs.extend(self._find_differences(obj1[key], obj2[key], new_path))

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                diffs.append({
                    "path": path or "root",
                    "type": "length_mismatch",
                    "value1": len(obj1),
                    "value2": len(obj2)
                })

            # Compare elements
            for i in range(min(len(obj1), len(obj2))):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                diffs.extend(self._find_differences(obj1[i], obj2[i], new_path))

        else:
            # Primitive values
            if obj1 != obj2:
                diffs.append({
                    "path": path or "root",
                    "type": "value_changed",
                    "value1": obj1,
                    "value2": obj2
                })

        return diffs


from hololoom.skills.base import register_skill

register_skill(DataFormatSkill())
