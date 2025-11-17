"""
Tool Definition Schema

Defines the data structures for custom tool definitions that can be
created through the visual no-code interface.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json


class ParameterType(str, Enum):
    """Types of input parameters"""
    TEXT = "text"
    NUMBER = "number"
    FILE = "file"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    DATE = "date"
    JSON = "json"
    ARRAY = "array"


class LogicBlockType(str, Enum):
    """Types of logic blocks"""
    IF_ELSE = "if_else"
    FOR_LOOP = "for_loop"
    WHILE_LOOP = "while_loop"
    TRANSFORM = "transform"
    VARIABLE = "variable"
    RETURN = "return"


class ActionType(str, Enum):
    """Types of actions"""
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    FILE_OPERATION = "file_operation"
    COMPUTE = "compute"
    CALL_TOOL = "call_tool"
    MEMORY_OPERATION = "memory_operation"


class OutputFormat(str, Enum):
    """Output format types"""
    TEXT = "text"
    JSON = "json"
    FILE = "file"
    MEMORY = "memory"


class TransformType(str, Enum):
    """Transform operation types"""
    MAP = "map"
    FILTER = "filter"
    REDUCE = "reduce"
    SORT = "sort"
    JOIN = "join"
    SPLIT = "split"


@dataclass
class Parameter:
    """Input parameter definition"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    options: Optional[List[str]] = None  # For dropdown
    validation: Optional[Dict[str, Any]] = None  # Min/max, pattern, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Condition:
    """Conditional expression"""
    left: str
    operator: str  # ==, !=, <, >, <=, >=, in, not in, contains
    right: Union[str, int, float, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HTTPRequestAction:
    """HTTP request action configuration"""
    method: str  # GET, POST, PUT, DELETE
    url: str
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict[str, Any]] = None
    timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DatabaseQueryAction:
    """Database query action configuration"""
    database: str  # neo4j, qdrant, etc.
    query: str
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FileOperationAction:
    """File operation action configuration"""
    operation: str  # read, write, append, delete
    path: str
    content: Optional[str] = None
    encoding: str = "utf-8"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ComputeAction:
    """Compute action configuration"""
    operation: str  # math, string, json, custom
    expression: str
    variables: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CallToolAction:
    """Call another tool action configuration"""
    tool_name: str
    parameters: Dict[str, str]  # Map to current context variables

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryOperationAction:
    """Memory operation action configuration"""
    operation: str  # store, retrieve, search, delete
    collection: str
    data: Optional[Dict[str, Any]] = None
    query: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogicBlock:
    """Logic block in the tool flow"""
    id: str
    type: LogicBlockType

    # Conditional blocks
    condition: Optional[Condition] = None
    then_blocks: Optional[List['LogicBlock']] = None
    else_blocks: Optional[List['LogicBlock']] = None

    # Loop blocks
    iterator: Optional[str] = None  # Variable name for iteration
    iterable: Optional[str] = None  # What to iterate over
    loop_blocks: Optional[List['LogicBlock']] = None

    # Transform blocks
    transform_type: Optional[TransformType] = None
    input_var: Optional[str] = None
    output_var: Optional[str] = None
    transform_expr: Optional[str] = None

    # Variable assignment
    variable_name: Optional[str] = None
    variable_value: Optional[str] = None

    # Action blocks
    action_type: Optional[ActionType] = None
    action_config: Optional[Union[
        HTTPRequestAction,
        DatabaseQueryAction,
        FileOperationAction,
        ComputeAction,
        CallToolAction,
        MemoryOperationAction
    ]] = None

    # Return block
    return_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "type": self.type.value}

        if self.condition:
            result["condition"] = self.condition.to_dict()
        if self.then_blocks:
            result["then_blocks"] = [b.to_dict() for b in self.then_blocks]
        if self.else_blocks:
            result["else_blocks"] = [b.to_dict() for b in self.else_blocks]
        if self.iterator:
            result["iterator"] = self.iterator
        if self.iterable:
            result["iterable"] = self.iterable
        if self.loop_blocks:
            result["loop_blocks"] = [b.to_dict() for b in self.loop_blocks]
        if self.transform_type:
            result["transform_type"] = self.transform_type.value
        if self.input_var:
            result["input_var"] = self.input_var
        if self.output_var:
            result["output_var"] = self.output_var
        if self.transform_expr:
            result["transform_expr"] = self.transform_expr
        if self.variable_name:
            result["variable_name"] = self.variable_name
        if self.variable_value:
            result["variable_value"] = self.variable_value
        if self.action_type:
            result["action_type"] = self.action_type.value
        if self.action_config:
            result["action_config"] = self.action_config.to_dict()
        if self.return_value:
            result["return_value"] = self.return_value

        return result


@dataclass
class ToolDefinition:
    """Complete tool definition"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: str = "custom"

    # Input parameters
    parameters: List[Parameter] = field(default_factory=list)

    # Logic flow
    logic_blocks: List[LogicBlock] = field(default_factory=list)

    # Output configuration
    output_format: OutputFormat = OutputFormat.JSON
    output_schema: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters],
            "logic_blocks": [b.to_dict() for b in self.logic_blocks],
            "output_format": self.output_format.value,
            "output_schema": self.output_schema,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "enabled": self.enabled,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        """Create from dictionary"""
        # Convert parameters
        parameters = []
        for p_data in data.get("parameters", []):
            parameters.append(Parameter(
                name=p_data["name"],
                type=ParameterType(p_data["type"]),
                description=p_data["description"],
                required=p_data.get("required", True),
                default=p_data.get("default"),
                options=p_data.get("options"),
                validation=p_data.get("validation"),
            ))

        # Convert logic blocks (simplified - full conversion would be recursive)
        logic_blocks = []
        for b_data in data.get("logic_blocks", []):
            logic_blocks.append(_parse_logic_block(b_data))

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
            category=data.get("category", "custom"),
            parameters=parameters,
            logic_blocks=logic_blocks,
            output_format=OutputFormat(data.get("output_format", "json")),
            output_schema=data.get("output_schema"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            enabled=data.get("enabled", True),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ToolDefinition':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


def _parse_logic_block(data: Dict[str, Any]) -> LogicBlock:
    """Parse a logic block from dictionary data"""
    block_type = LogicBlockType(data["type"])

    # Parse condition if present
    condition = None
    if "condition" in data:
        c_data = data["condition"]
        condition = Condition(
            left=c_data["left"],
            operator=c_data["operator"],
            right=c_data["right"]
        )

    # Parse action config if present
    action_config = None
    if "action_config" in data:
        action_type = ActionType(data["action_type"])
        config_data = data["action_config"]

        if action_type == ActionType.HTTP_REQUEST:
            action_config = HTTPRequestAction(**config_data)
        elif action_type == ActionType.DATABASE_QUERY:
            action_config = DatabaseQueryAction(**config_data)
        elif action_type == ActionType.FILE_OPERATION:
            action_config = FileOperationAction(**config_data)
        elif action_type == ActionType.COMPUTE:
            action_config = ComputeAction(**config_data)
        elif action_type == ActionType.CALL_TOOL:
            action_config = CallToolAction(**config_data)
        elif action_type == ActionType.MEMORY_OPERATION:
            action_config = MemoryOperationAction(**config_data)

    # Recursively parse nested blocks
    then_blocks = None
    if "then_blocks" in data:
        then_blocks = [_parse_logic_block(b) for b in data["then_blocks"]]

    else_blocks = None
    if "else_blocks" in data:
        else_blocks = [_parse_logic_block(b) for b in data["else_blocks"]]

    loop_blocks = None
    if "loop_blocks" in data:
        loop_blocks = [_parse_logic_block(b) for b in data["loop_blocks"]]

    transform_type = None
    if "transform_type" in data:
        transform_type = TransformType(data["transform_type"])

    action_type = None
    if "action_type" in data:
        action_type = ActionType(data["action_type"])

    return LogicBlock(
        id=data["id"],
        type=block_type,
        condition=condition,
        then_blocks=then_blocks,
        else_blocks=else_blocks,
        iterator=data.get("iterator"),
        iterable=data.get("iterable"),
        loop_blocks=loop_blocks,
        transform_type=transform_type,
        input_var=data.get("input_var"),
        output_var=data.get("output_var"),
        transform_expr=data.get("transform_expr"),
        variable_name=data.get("variable_name"),
        variable_value=data.get("variable_value"),
        action_type=action_type,
        action_config=action_config,
        return_value=data.get("return_value"),
    )
