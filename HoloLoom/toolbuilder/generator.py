"""
Code Generator

Generates Python code from tool definitions using Jinja2 templates.
"""

from typing import Dict, Any, List
import re
from jinja2 import Template
from .definition import (
    ToolDefinition,
    LogicBlock,
    LogicBlockType,
    ActionType,
    ParameterType,
    TransformType,
)


class CodeGenerator:
    """Generates Python code from tool definitions"""

    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "

    def generate(self, tool_def: ToolDefinition) -> str:
        """
        Generate Python code from tool definition

        Args:
            tool_def: The tool definition

        Returns:
            Generated Python code as string
        """
        # Generate function signature
        func_signature = self._generate_signature(tool_def)

        # Generate docstring
        docstring = self._generate_docstring(tool_def)

        # Generate function body
        func_body = self._generate_body(tool_def)

        # Combine into complete function
        code = f"{func_signature}\n{docstring}\n{func_body}\n"

        # Format code (basic formatting)
        code = self._format_code(code)

        return code

    def _generate_signature(self, tool_def: ToolDefinition) -> str:
        """Generate function signature"""
        params = []
        for param in tool_def.parameters:
            param_str = param.name

            # Add type hint
            type_hint = self._get_type_hint(param.type)
            param_str += f": {type_hint}"

            # Add default value if not required
            if not param.required and param.default is not None:
                if isinstance(param.default, str):
                    param_str += f' = "{param.default}"'
                else:
                    param_str += f" = {param.default}"
            elif not param.required:
                param_str += " = None"

            params.append(param_str)

        params_str = ", ".join(params)
        return f"async def {tool_def.name}({params_str}) -> Dict[str, Any]:"

    def _get_type_hint(self, param_type: ParameterType) -> str:
        """Get Python type hint for parameter type"""
        type_map = {
            ParameterType.TEXT: "str",
            ParameterType.NUMBER: "float",
            ParameterType.FILE: "str",
            ParameterType.DROPDOWN: "str",
            ParameterType.CHECKBOX: "bool",
            ParameterType.DATE: "str",
            ParameterType.JSON: "Dict[str, Any]",
            ParameterType.ARRAY: "List[Any]",
        }
        return type_map.get(param_type, "Any")

    def _generate_docstring(self, tool_def: ToolDefinition) -> str:
        """Generate function docstring"""
        lines = [
            '    """',
            f"    {tool_def.description}",
            "",
        ]

        if tool_def.parameters:
            lines.append("    Args:")
            for param in tool_def.parameters:
                req = "" if param.required else " (optional)"
                lines.append(f"        {param.name}: {param.description}{req}")
            lines.append("")

        lines.append("    Returns:")
        lines.append("        Dict containing the tool execution result")
        lines.append('    """')

        return "\n".join(lines)

    def _generate_body(self, tool_def: ToolDefinition) -> str:
        """Generate function body"""
        lines = []

        # Import statements
        imports = self._get_required_imports(tool_def)
        if imports:
            for imp in imports:
                lines.append(f"    {imp}")
            lines.append("")

        # Initialize context dictionary
        lines.append("    # Initialize execution context")
        lines.append("    context = {")
        for param in tool_def.parameters:
            lines.append(f'        "{param.name}": {param.name},')
        lines.append("    }")
        lines.append("")

        # Generate logic blocks
        for block in tool_def.logic_blocks:
            block_code = self._generate_block(block, indent=1)
            lines.append(block_code)

        # Default return if no explicit return
        lines.append("")
        lines.append("    # Default return")
        lines.append('    return {"status": "success", "result": context}')

        return "\n".join(lines)

    def _get_required_imports(self, tool_def: ToolDefinition) -> List[str]:
        """Determine required imports based on actions"""
        imports = set()
        imports.add("from typing import Dict, Any, List")

        # Check all blocks for required imports
        all_blocks = self._get_all_blocks(tool_def.logic_blocks)

        for block in all_blocks:
            if block.action_type == ActionType.HTTP_REQUEST:
                imports.add("import aiohttp")
            elif block.action_type == ActionType.FILE_OPERATION:
                imports.add("import aiofiles")
            elif block.action_type == ActionType.DATABASE_QUERY:
                imports.add("# Database imports would go here")
            elif block.action_type == ActionType.COMPUTE:
                imports.add("import json")
                imports.add("import math")

        return sorted(list(imports))

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

    def _generate_block(self, block: LogicBlock, indent: int = 0) -> str:
        """Generate code for a logic block"""
        indent_str = self.indent_str * indent

        if block.type == LogicBlockType.IF_ELSE:
            return self._generate_if_else(block, indent)
        elif block.type == LogicBlockType.FOR_LOOP:
            return self._generate_for_loop(block, indent)
        elif block.type == LogicBlockType.WHILE_LOOP:
            return self._generate_while_loop(block, indent)
        elif block.type == LogicBlockType.TRANSFORM:
            return self._generate_transform(block, indent)
        elif block.type == LogicBlockType.VARIABLE:
            return self._generate_variable(block, indent)
        elif block.type == LogicBlockType.RETURN:
            return self._generate_return(block, indent)
        elif block.action_type:
            return self._generate_action(block, indent)

        return f"{indent_str}# Unknown block type: {block.type}"

    def _generate_if_else(self, block: LogicBlock, indent: int) -> str:
        """Generate if/else block"""
        indent_str = self.indent_str * indent
        lines = []

        # Generate condition
        condition = self._generate_condition(block.condition)
        lines.append(f"{indent_str}if {condition}:")

        # Generate then blocks
        if block.then_blocks:
            for nested in block.then_blocks:
                lines.append(self._generate_block(nested, indent + 1))
        else:
            lines.append(f"{indent_str}    pass")

        # Generate else blocks
        if block.else_blocks:
            lines.append(f"{indent_str}else:")
            for nested in block.else_blocks:
                lines.append(self._generate_block(nested, indent + 1))

        return "\n".join(lines)

    def _generate_for_loop(self, block: LogicBlock, indent: int) -> str:
        """Generate for loop"""
        indent_str = self.indent_str * indent
        lines = []

        # Resolve iterable (could be variable or parameter)
        iterable = self._resolve_variable(block.iterable)

        lines.append(f"{indent_str}for {block.iterator} in {iterable}:")

        # Generate loop body
        if block.loop_blocks:
            for nested in block.loop_blocks:
                lines.append(self._generate_block(nested, indent + 1))
        else:
            lines.append(f"{indent_str}    pass")

        return "\n".join(lines)

    def _generate_while_loop(self, block: LogicBlock, indent: int) -> str:
        """Generate while loop"""
        indent_str = self.indent_str * indent
        lines = []

        condition = self._generate_condition(block.condition)
        lines.append(f"{indent_str}# WARNING: Ensure this loop has an exit condition")
        lines.append(f"{indent_str}iteration_count = 0")
        lines.append(f"{indent_str}while {condition} and iteration_count < 1000:")

        # Generate loop body
        if block.loop_blocks:
            for nested in block.loop_blocks:
                lines.append(self._generate_block(nested, indent + 1))
            lines.append(f"{indent_str}    iteration_count += 1")
        else:
            lines.append(f"{indent_str}    pass")

        return "\n".join(lines)

    def _generate_transform(self, block: LogicBlock, indent: int) -> str:
        """Generate transform operation"""
        indent_str = self.indent_str * indent
        lines = []

        input_var = self._resolve_variable(block.input_var)
        output_var = block.output_var
        expr = block.transform_expr

        if block.transform_type == TransformType.MAP:
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"[{expr} for item in {input_var}]"
            )
        elif block.transform_type == TransformType.FILTER:
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"[item for item in {input_var} if {expr}]"
            )
        elif block.transform_type == TransformType.REDUCE:
            lines.append(f"{indent_str}from functools import reduce")
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"reduce(lambda acc, item: {expr}, {input_var})"
            )
        elif block.transform_type == TransformType.SORT:
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"sorted({input_var}, key=lambda item: {expr})"
            )
        elif block.transform_type == TransformType.JOIN:
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"{expr}.join({input_var})"
            )
        elif block.transform_type == TransformType.SPLIT:
            lines.append(
                f"{indent_str}context['{output_var}'] = "
                f"{input_var}.split({expr})"
            )

        return "\n".join(lines)

    def _generate_variable(self, block: LogicBlock, indent: int) -> str:
        """Generate variable assignment"""
        indent_str = self.indent_str * indent
        value = self._resolve_variable(block.variable_value)
        return f"{indent_str}context['{block.variable_name}'] = {value}"

    def _generate_return(self, block: LogicBlock, indent: int) -> str:
        """Generate return statement"""
        indent_str = self.indent_str * indent
        return_val = self._resolve_variable(block.return_value)
        return f'{indent_str}return {{"status": "success", "result": {return_val}}}'

    def _generate_action(self, block: LogicBlock, indent: int) -> str:
        """Generate action code"""
        indent_str = self.indent_str * indent

        if block.action_type == ActionType.HTTP_REQUEST:
            return self._generate_http_request(block, indent)
        elif block.action_type == ActionType.DATABASE_QUERY:
            return self._generate_database_query(block, indent)
        elif block.action_type == ActionType.FILE_OPERATION:
            return self._generate_file_operation(block, indent)
        elif block.action_type == ActionType.COMPUTE:
            return self._generate_compute(block, indent)
        elif block.action_type == ActionType.CALL_TOOL:
            return self._generate_call_tool(block, indent)
        elif block.action_type == ActionType.MEMORY_OPERATION:
            return self._generate_memory_operation(block, indent)

        return f"{indent_str}# Unknown action type"

    def _generate_http_request(self, block: LogicBlock, indent: int) -> str:
        """Generate HTTP request code"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        lines.append(f"{indent_str}# HTTP Request")
        lines.append(f"{indent_str}async with aiohttp.ClientSession() as session:")

        method = config.method.lower()
        url = self._resolve_variable(config.url)

        lines.append(f"{indent_str}    async with session.{method}(")
        lines.append(f"{indent_str}        {url},")

        if config.headers:
            lines.append(f"{indent_str}        headers={config.headers},")
        if config.body and method in ['post', 'put', 'patch']:
            lines.append(f"{indent_str}        json={config.body},")
        lines.append(f"{indent_str}        timeout={config.timeout}")
        lines.append(f"{indent_str}    ) as response:")
        lines.append(
            f"{indent_str}        context['http_response'] = await response.json()"
        )

        return "\n".join(lines)

    def _generate_database_query(self, block: LogicBlock, indent: int) -> str:
        """Generate database query code"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        lines.append(f"{indent_str}# Database Query")
        lines.append(f"{indent_str}# TODO: Implement {config.database} query")
        lines.append(f"{indent_str}# Query: {config.query}")
        if config.parameters:
            lines.append(f"{indent_str}# Parameters: {config.parameters}")

        return "\n".join(lines)

    def _generate_file_operation(self, block: LogicBlock, indent: int) -> str:
        """Generate file operation code"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        path = self._resolve_variable(config.path)

        if config.operation == "read":
            lines.append(f"{indent_str}async with aiofiles.open({path}, 'r') as f:")
            lines.append(f"{indent_str}    context['file_content'] = await f.read()")
        elif config.operation == "write":
            content = self._resolve_variable(config.content)
            lines.append(f"{indent_str}async with aiofiles.open({path}, 'w') as f:")
            lines.append(f"{indent_str}    await f.write({content})")
        elif config.operation == "append":
            content = self._resolve_variable(config.content)
            lines.append(f"{indent_str}async with aiofiles.open({path}, 'a') as f:")
            lines.append(f"{indent_str}    await f.write({content})")

        return "\n".join(lines)

    def _generate_compute(self, block: LogicBlock, indent: int) -> str:
        """Generate compute operation code"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        lines.append(f"{indent_str}# Compute operation")

        # Replace variable references in expression
        expr = config.expression
        for var_name in config.variables or {}:
            var_ref = config.variables[var_name]
            resolved = self._resolve_variable(var_ref)
            expr = expr.replace(f"${{{var_name}}}", resolved)

        lines.append(f"{indent_str}context['compute_result'] = {expr}")

        return "\n".join(lines)

    def _generate_call_tool(self, block: LogicBlock, indent: int) -> str:
        """Generate call to another tool"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        lines.append(f"{indent_str}# Call tool: {config.tool_name}")
        lines.append(f"{indent_str}tool_params = {{")
        for param_name, param_value in config.parameters.items():
            resolved = self._resolve_variable(param_value)
            lines.append(f"{indent_str}    '{param_name}': {resolved},")
        lines.append(f"{indent_str}}}")
        lines.append(
            f"{indent_str}context['tool_result'] = "
            f"await {config.tool_name}(**tool_params)"
        )

        return "\n".join(lines)

    def _generate_memory_operation(self, block: LogicBlock, indent: int) -> str:
        """Generate memory operation code"""
        indent_str = self.indent_str * indent
        config = block.action_config
        lines = []

        lines.append(f"{indent_str}# Memory operation: {config.operation}")
        lines.append(f"{indent_str}# Collection: {config.collection}")
        if config.operation == "store":
            lines.append(f"{indent_str}# TODO: Store data to memory")
        elif config.operation == "retrieve":
            lines.append(f"{indent_str}# TODO: Retrieve data from memory")

        return "\n".join(lines)

    def _generate_condition(self, condition) -> str:
        """Generate condition expression"""
        left = self._resolve_variable(condition.left)
        right = condition.right

        # Quote string literals
        if isinstance(right, str) and not right.startswith(('context[', 'item', 'acc')):
            right = f'"{right}"'

        operator_map = {
            '==': '==',
            '!=': '!=',
            '<': '<',
            '>': '>',
            '<=': '<=',
            '>=': '>=',
            'in': 'in',
            'not in': 'not in',
            'contains': 'in',  # Reversed
        }

        op = operator_map.get(condition.operator, condition.operator)

        if condition.operator == 'contains':
            return f"{right} in {left}"
        else:
            return f"{left} {op} {right}"

    def _resolve_variable(self, var: str) -> str:
        """Resolve variable reference to context access"""
        if not var:
            return "None"

        # Already a context reference
        if var.startswith('context['):
            return var

        # Template variable ${var_name}
        if var.startswith('${') and var.endswith('}'):
            var_name = var[2:-1]
            return f"context['{var_name}']"

        # Literal value
        if isinstance(var, str) and (var.startswith('"') or var.startswith("'")):
            return var

        # Try to parse as number
        try:
            float(var)
            return var
        except (ValueError, TypeError):
            pass

        # Assume it's a variable name
        if var in ['True', 'False', 'None']:
            return var

        return f"context['{var}']"

    def _format_code(self, code: str) -> str:
        """Basic code formatting"""
        lines = code.split('\n')
        formatted = []

        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            formatted.append(line)

        # Remove multiple consecutive blank lines
        result = []
        prev_blank = False
        for line in formatted:
            is_blank = len(line.strip()) == 0
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank

        return '\n'.join(result)
