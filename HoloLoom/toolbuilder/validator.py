"""
Tool Definition Validator

Validates tool definitions for correctness, security, and best practices.
"""

from typing import List, Set, Dict, Any
import re
from .definition import (
    ToolDefinition,
    LogicBlock,
    LogicBlockType,
    ActionType,
    ParameterType,
    HTTPRequestAction,
    FileOperationAction,
)


class ValidationError(Exception):
    """Validation error exception"""
    pass


class ToolValidator:
    """Validates tool definitions"""

    # Dangerous imports that should not be allowed
    DANGEROUS_IMPORTS = {
        'os', 'sys', 'subprocess', 'eval', 'exec', 'compile',
        '__import__', 'open', 'input', 'execfile', 'reload',
        'shutil', 'pickle', 'shelve', 'marshal', 'imp', 'importlib',
    }

    # Dangerous patterns in expressions
    DANGEROUS_PATTERNS = [
        r'__\w+__',  # Dunder methods
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'__import__',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
    ]

    # Maximum complexity limits
    MAX_LOGIC_BLOCKS = 100
    MAX_NESTING_DEPTH = 10
    MAX_LOOP_ITERATIONS = 1000
    MAX_PARAMETERS = 50

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, tool_def: ToolDefinition) -> bool:
        """
        Validate a tool definition

        Args:
            tool_def: The tool definition to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If critical validation errors found
        """
        self.errors = []
        self.warnings = []

        # Basic validation
        self._validate_basic_fields(tool_def)

        # Parameter validation
        self._validate_parameters(tool_def)

        # Logic flow validation
        self._validate_logic_flow(tool_def)

        # Security validation
        self._validate_security(tool_def)

        # Complexity validation
        self._validate_complexity(tool_def)

        if self.errors:
            raise ValidationError(
                f"Tool validation failed with {len(self.errors)} error(s):\n" +
                "\n".join(f"  - {e}" for e in self.errors)
            )

        return True

    def _validate_basic_fields(self, tool_def: ToolDefinition):
        """Validate basic required fields"""
        if not tool_def.id:
            self.errors.append("Tool ID is required")

        if not tool_def.name:
            self.errors.append("Tool name is required")
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', tool_def.name):
            self.errors.append(
                "Tool name must start with letter and contain only "
                "letters, numbers, hyphens, and underscores"
            )

        if not tool_def.description:
            self.errors.append("Tool description is required")
        elif len(tool_def.description) < 10:
            self.warnings.append("Tool description should be more descriptive")

        if not tool_def.version:
            self.errors.append("Tool version is required")
        elif not re.match(r'^\d+\.\d+\.\d+$', tool_def.version):
            self.errors.append("Tool version must be in format X.Y.Z")

    def _validate_parameters(self, tool_def: ToolDefinition):
        """Validate input parameters"""
        if len(tool_def.parameters) > self.MAX_PARAMETERS:
            self.errors.append(
                f"Too many parameters ({len(tool_def.parameters)}). "
                f"Maximum allowed: {self.MAX_PARAMETERS}"
            )

        param_names: Set[str] = set()

        for param in tool_def.parameters:
            # Check for duplicate names
            if param.name in param_names:
                self.errors.append(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)

            # Validate parameter name
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', param.name):
                self.errors.append(
                    f"Invalid parameter name '{param.name}'. "
                    "Must start with letter and contain only letters, numbers, underscores"
                )

            # Validate dropdown options
            if param.type == ParameterType.DROPDOWN:
                if not param.options or len(param.options) == 0:
                    self.errors.append(
                        f"Dropdown parameter '{param.name}' must have options"
                    )

            # Validate default value type
            if param.default is not None:
                if param.type == ParameterType.NUMBER:
                    if not isinstance(param.default, (int, float)):
                        self.errors.append(
                            f"Parameter '{param.name}' default value must be a number"
                        )
                elif param.type == ParameterType.CHECKBOX:
                    if not isinstance(param.default, bool):
                        self.errors.append(
                            f"Parameter '{param.name}' default value must be boolean"
                        )

            # Validate validation rules
            if param.validation:
                if param.type == ParameterType.NUMBER:
                    if 'min' in param.validation and 'max' in param.validation:
                        if param.validation['min'] > param.validation['max']:
                            self.errors.append(
                                f"Parameter '{param.name}' min value cannot exceed max"
                            )

    def _validate_logic_flow(self, tool_def: ToolDefinition):
        """Validate logic flow"""
        if not tool_def.logic_blocks:
            self.errors.append("Tool must have at least one logic block")
            return

        # Check for cycles and infinite loops
        visited_blocks: Set[str] = set()

        for block in tool_def.logic_blocks:
            self._validate_logic_block(block, visited_blocks, depth=0)

        # Check for return statement
        has_return = self._has_return_statement(tool_def.logic_blocks)
        if not has_return:
            self.warnings.append("Tool should have at least one return statement")

    def _validate_logic_block(
        self,
        block: LogicBlock,
        visited: Set[str],
        depth: int,
    ):
        """Validate a single logic block"""
        if depth > self.MAX_NESTING_DEPTH:
            self.errors.append(
                f"Logic nesting too deep (>{self.MAX_NESTING_DEPTH}). "
                "Consider simplifying the logic."
            )
            return

        if block.id in visited:
            self.errors.append(f"Duplicate block ID: {block.id}")
        visited.add(block.id)

        # Validate based on block type
        if block.type == LogicBlockType.IF_ELSE:
            if not block.condition:
                self.errors.append(f"If/else block {block.id} missing condition")
            if not block.then_blocks:
                self.warnings.append(f"If/else block {block.id} has no then blocks")

            # Recursively validate nested blocks
            if block.then_blocks:
                for nested in block.then_blocks:
                    self._validate_logic_block(nested, visited, depth + 1)
            if block.else_blocks:
                for nested in block.else_blocks:
                    self._validate_logic_block(nested, visited, depth + 1)

        elif block.type == LogicBlockType.FOR_LOOP:
            if not block.iterator:
                self.errors.append(f"For loop block {block.id} missing iterator")
            if not block.iterable:
                self.errors.append(f"For loop block {block.id} missing iterable")
            if not block.loop_blocks:
                self.warnings.append(f"For loop block {block.id} has no body")

            if block.loop_blocks:
                for nested in block.loop_blocks:
                    self._validate_logic_block(nested, visited, depth + 1)

        elif block.type == LogicBlockType.WHILE_LOOP:
            if not block.condition:
                self.errors.append(f"While loop block {block.id} missing condition")
            if not block.loop_blocks:
                self.warnings.append(f"While loop block {block.id} has no body")

            # Check for potential infinite loops
            self.warnings.append(
                f"While loop block {block.id}: Ensure loop has exit condition"
            )

            if block.loop_blocks:
                for nested in block.loop_blocks:
                    self._validate_logic_block(nested, visited, depth + 1)

        elif block.type == LogicBlockType.TRANSFORM:
            if not block.input_var:
                self.errors.append(f"Transform block {block.id} missing input_var")
            if not block.output_var:
                self.errors.append(f"Transform block {block.id} missing output_var")
            if not block.transform_expr:
                self.errors.append(f"Transform block {block.id} missing transform_expr")

            # Validate transform expression
            if block.transform_expr:
                self._validate_expression(block.transform_expr, block.id)

        elif block.type == LogicBlockType.VARIABLE:
            if not block.variable_name:
                self.errors.append(f"Variable block {block.id} missing variable_name")
            if block.variable_value is None:
                self.errors.append(f"Variable block {block.id} missing variable_value")

        # Validate action configurations
        if block.action_type:
            self._validate_action(block, block.id)

    def _validate_action(self, block: LogicBlock, block_id: str):
        """Validate action configurations"""
        if not block.action_config:
            self.errors.append(f"Block {block_id} has action_type but no action_config")
            return

        if block.action_type == ActionType.HTTP_REQUEST:
            config: HTTPRequestAction = block.action_config
            if config.method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                self.errors.append(f"Invalid HTTP method in block {block_id}")

            # Validate URL
            if not config.url:
                self.errors.append(f"HTTP request in block {block_id} missing URL")
            elif not config.url.startswith(('http://', 'https://', '${')):
                self.errors.append(
                    f"HTTP request in block {block_id} must use http:// or https://"
                )

            # Check timeout
            if config.timeout and config.timeout > 300:
                self.warnings.append(
                    f"HTTP timeout in block {block_id} is very long ({config.timeout}s)"
                )

        elif block.action_type == ActionType.FILE_OPERATION:
            config: FileOperationAction = block.action_config
            if config.operation not in ['read', 'write', 'append', 'delete']:
                self.errors.append(f"Invalid file operation in block {block_id}")

            # Security check for file operations
            if config.path and not config.path.startswith('${'):
                # Check for path traversal attempts
                if '..' in config.path or config.path.startswith('/'):
                    self.errors.append(
                        f"File path in block {block_id} may be unsafe. "
                        "Use relative paths within allowed directories."
                    )

        elif block.action_type == ActionType.COMPUTE:
            config = block.action_config
            if config.expression:
                self._validate_expression(config.expression, block_id)

    def _validate_expression(self, expr: str, block_id: str):
        """Validate an expression for security"""
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, expr, re.IGNORECASE):
                self.errors.append(
                    f"Expression in block {block_id} contains dangerous pattern: {pattern}"
                )

        # Check for dangerous keywords
        for keyword in self.DANGEROUS_IMPORTS:
            if keyword in expr:
                self.errors.append(
                    f"Expression in block {block_id} contains dangerous keyword: {keyword}"
                )

    def _validate_security(self, tool_def: ToolDefinition):
        """Perform security validation"""
        # Check for SQL injection patterns
        for block in self._get_all_blocks(tool_def.logic_blocks):
            if block.action_type == ActionType.DATABASE_QUERY:
                config = block.action_config
                if config.query:
                    # Simple check for string concatenation in queries
                    if '+' in config.query or 'f"' in config.query or "f'" in config.query:
                        self.warnings.append(
                            f"Block {block.id} may be vulnerable to SQL injection. "
                            "Use parameterized queries."
                        )

    def _validate_complexity(self, tool_def: ToolDefinition):
        """Validate complexity limits"""
        total_blocks = len(self._get_all_blocks(tool_def.logic_blocks))

        if total_blocks > self.MAX_LOGIC_BLOCKS:
            self.errors.append(
                f"Too many logic blocks ({total_blocks}). "
                f"Maximum allowed: {self.MAX_LOGIC_BLOCKS}"
            )

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(tool_def.logic_blocks)
        if complexity > 20:
            self.warnings.append(
                f"High complexity ({complexity}). Consider breaking into smaller tools."
            )

    def _get_all_blocks(self, blocks: List[LogicBlock]) -> List[LogicBlock]:
        """Get all blocks including nested ones"""
        all_blocks = []
        for block in blocks:
            all_blocks.append(block)
            if block.then_blocks:
                all_blocks.extend(self._get_all_blocks(block.then_blocks))
            if block.else_blocks:
                all_blocks.extend(self._get_all_blocks(block.else_blocks))
            if block.loop_blocks:
                all_blocks.extend(self._get_all_blocks(block.loop_blocks))
        return all_blocks

    def _has_return_statement(self, blocks: List[LogicBlock]) -> bool:
        """Check if logic flow has a return statement"""
        for block in blocks:
            if block.type == LogicBlockType.RETURN:
                return True
            if block.then_blocks and self._has_return_statement(block.then_blocks):
                return True
            if block.else_blocks and self._has_return_statement(block.else_blocks):
                return True
            if block.loop_blocks and self._has_return_statement(block.loop_blocks):
                return True
        return False

    def _calculate_complexity(self, blocks: List[LogicBlock]) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for block in blocks:
            if block.type in [LogicBlockType.IF_ELSE, LogicBlockType.WHILE_LOOP]:
                complexity += 1
            if block.type == LogicBlockType.FOR_LOOP:
                complexity += 1

            # Recursively calculate for nested blocks
            if block.then_blocks:
                complexity += self._calculate_complexity(block.then_blocks) - 1
            if block.else_blocks:
                complexity += self._calculate_complexity(block.else_blocks) - 1
            if block.loop_blocks:
                complexity += self._calculate_complexity(block.loop_blocks) - 1

        return complexity

    def get_warnings(self) -> List[str]:
        """Get validation warnings"""
        return self.warnings

    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors
