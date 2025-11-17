"""
Built-in Tool Templates

Pre-built templates for common tool patterns.
"""

from typing import Dict, Any


def get_calculator_template() -> Dict[str, Any]:
    """Calculator tool template"""
    return {
        "id": "template_calculator",
        "name": "calculator",
        "description": "Basic calculator that performs arithmetic operations",
        "version": "1.0.0",
        "author": "HoloLoom",
        "tags": ["math", "calculator", "arithmetic"],
        "category": "utility",
        "parameters": [
            {
                "name": "operation",
                "type": "dropdown",
                "description": "The operation to perform",
                "required": True,
                "options": ["add", "subtract", "multiply", "divide", "power", "sqrt"]
            },
            {
                "name": "x",
                "type": "number",
                "description": "First number",
                "required": True
            },
            {
                "name": "y",
                "type": "number",
                "description": "Second number (not required for sqrt)",
                "required": False
            }
        ],
        "logic_blocks": [
            {
                "id": "block_1",
                "type": "if_else",
                "condition": {
                    "left": "${operation}",
                    "operator": "==",
                    "right": "add"
                },
                "then_blocks": [
                    {
                        "id": "block_1_1",
                        "type": "variable",
                        "variable_name": "result",
                        "variable_value": "${x} + ${y}"
                    }
                ],
                "else_blocks": [
                    {
                        "id": "block_1_2",
                        "type": "if_else",
                        "condition": {
                            "left": "${operation}",
                            "operator": "==",
                            "right": "subtract"
                        },
                        "then_blocks": [
                            {
                                "id": "block_1_2_1",
                                "type": "variable",
                                "variable_name": "result",
                                "variable_value": "${x} - ${y}"
                            }
                        ],
                        "else_blocks": [
                            {
                                "id": "block_1_2_2",
                                "type": "variable",
                                "variable_name": "result",
                                "variable_value": "0"
                            }
                        ]
                    }
                ]
            },
            {
                "id": "block_2",
                "type": "return",
                "return_value": "${result}"
            }
        ],
        "output_format": "json",
        "enabled": True
    }


def get_web_scraper_template() -> Dict[str, Any]:
    """Web scraper tool template"""
    return {
        "id": "template_web_scraper",
        "name": "web_scraper",
        "description": "Fetch and parse content from a web URL",
        "version": "1.0.0",
        "author": "HoloLoom",
        "tags": ["web", "http", "scraper"],
        "category": "integration",
        "parameters": [
            {
                "name": "url",
                "type": "text",
                "description": "The URL to fetch",
                "required": True,
                "validation": {
                    "pattern": "^https?://.+"
                }
            },
            {
                "name": "timeout",
                "type": "number",
                "description": "Request timeout in seconds",
                "required": False,
                "default": 30,
                "validation": {
                    "min": 1,
                    "max": 300
                }
            }
        ],
        "logic_blocks": [
            {
                "id": "block_1",
                "type": "variable",
                "action_type": "http_request",
                "action_config": {
                    "method": "GET",
                    "url": "${url}",
                    "timeout": 30
                }
            },
            {
                "id": "block_2",
                "type": "return",
                "return_value": "${http_response}"
            }
        ],
        "output_format": "json",
        "enabled": True
    }


def get_data_transformer_template() -> Dict[str, Any]:
    """Data transformer tool template"""
    return {
        "id": "template_data_transformer",
        "name": "data_transformer",
        "description": "Transform and filter data arrays",
        "version": "1.0.0",
        "author": "HoloLoom",
        "tags": ["data", "transform", "filter"],
        "category": "utility",
        "parameters": [
            {
                "name": "data",
                "type": "array",
                "description": "Input data array",
                "required": True
            },
            {
                "name": "operation",
                "type": "dropdown",
                "description": "Transform operation",
                "required": True,
                "options": ["map", "filter", "sort"]
            },
            {
                "name": "field",
                "type": "text",
                "description": "Field name to operate on",
                "required": True
            }
        ],
        "logic_blocks": [
            {
                "id": "block_1",
                "type": "if_else",
                "condition": {
                    "left": "${operation}",
                    "operator": "==",
                    "right": "filter"
                },
                "then_blocks": [
                    {
                        "id": "block_1_1",
                        "type": "transform",
                        "transform_type": "filter",
                        "input_var": "${data}",
                        "output_var": "result",
                        "transform_expr": "item.get('${field}') is not None"
                    }
                ],
                "else_blocks": [
                    {
                        "id": "block_1_2",
                        "type": "transform",
                        "transform_type": "sort",
                        "input_var": "${data}",
                        "output_var": "result",
                        "transform_expr": "item.get('${field}', 0)"
                    }
                ]
            },
            {
                "id": "block_2",
                "type": "return",
                "return_value": "${result}"
            }
        ],
        "output_format": "json",
        "enabled": True
    }


def get_api_client_template() -> Dict[str, Any]:
    """API client tool template"""
    return {
        "id": "template_api_client",
        "name": "api_client",
        "description": "Generic API client for making HTTP requests",
        "version": "1.0.0",
        "author": "HoloLoom",
        "tags": ["api", "http", "rest"],
        "category": "integration",
        "parameters": [
            {
                "name": "method",
                "type": "dropdown",
                "description": "HTTP method",
                "required": True,
                "options": ["GET", "POST", "PUT", "DELETE", "PATCH"]
            },
            {
                "name": "url",
                "type": "text",
                "description": "API endpoint URL",
                "required": True
            },
            {
                "name": "headers",
                "type": "json",
                "description": "Request headers",
                "required": False
            },
            {
                "name": "body",
                "type": "json",
                "description": "Request body (for POST/PUT/PATCH)",
                "required": False
            }
        ],
        "logic_blocks": [
            {
                "id": "block_1",
                "type": "variable",
                "action_type": "http_request",
                "action_config": {
                    "method": "${method}",
                    "url": "${url}",
                    "headers": "${headers}",
                    "body": "${body}",
                    "timeout": 60
                }
            },
            {
                "id": "block_2",
                "type": "return",
                "return_value": "${http_response}"
            }
        ],
        "output_format": "json",
        "enabled": True
    }


def get_file_converter_template() -> Dict[str, Any]:
    """File converter tool template"""
    return {
        "id": "template_file_converter",
        "name": "file_converter",
        "description": "Convert between different file formats",
        "version": "1.0.0",
        "author": "HoloLoom",
        "tags": ["file", "convert", "format"],
        "category": "utility",
        "parameters": [
            {
                "name": "input_path",
                "type": "file",
                "description": "Input file path",
                "required": True
            },
            {
                "name": "output_path",
                "type": "text",
                "description": "Output file path",
                "required": True
            },
            {
                "name": "format",
                "type": "dropdown",
                "description": "Output format",
                "required": True,
                "options": ["json", "csv", "txt", "xml"]
            }
        ],
        "logic_blocks": [
            {
                "id": "block_1",
                "type": "variable",
                "action_type": "file_operation",
                "action_config": {
                    "operation": "read",
                    "path": "${input_path}",
                    "encoding": "utf-8"
                }
            },
            {
                "id": "block_2",
                "type": "compute",
                "action_type": "compute",
                "action_config": {
                    "operation": "json",
                    "expression": "json.loads(${file_content})",
                    "variables": {
                        "content": "file_content"
                    }
                }
            },
            {
                "id": "block_3",
                "type": "variable",
                "action_type": "file_operation",
                "action_config": {
                    "operation": "write",
                    "path": "${output_path}",
                    "content": "${compute_result}",
                    "encoding": "utf-8"
                }
            },
            {
                "id": "block_4",
                "type": "return",
                "return_value": "{'status': 'success', 'output_path': '${output_path}'}"
            }
        ],
        "output_format": "json",
        "enabled": True
    }
