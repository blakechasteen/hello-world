"""
SQL injection prevention utilities.

Phase 2 - Input Validation & Sanitization (November 2025)

Implements:
- Parameterized query building
- Identifier escaping
- Safe query templates
- Query builder API
- Common SQL patterns with protection
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SQLDataType(Enum):
    """SQL parameter data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    NULL = "null"


class SQLOperator(Enum):
    """Safe SQL comparison operators."""
    EQUALS = "="
    NOT_EQUALS = "!="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LIKE = "LIKE"
    IN = "IN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


class SQLOrder(Enum):
    """Safe SQL ordering."""
    ASC = "ASC"
    DESC = "DESC"


# ============================================================================
# Parameter Validation & Escaping
# ============================================================================

def validate_parameter(value: Any, data_type: SQLDataType) -> Tuple[bool, str]:
    """
    Validate parameter matches expected data type.

    Args:
        value: Parameter value to validate
        data_type: Expected SQL data type

    Returns:
        Tuple of (valid, error_message or empty string)

    Example:
        >>> valid, msg = validate_parameter(123, SQLDataType.INTEGER)
        >>> assert valid is True
        >>> valid, msg = validate_parameter("not_a_number", SQLDataType.INTEGER)
        >>> assert valid is False
    """
    if data_type == SQLDataType.NULL:
        if value is not None:
            return False, "Expected NULL value"
        return True, ""

    if value is None:
        return False, "NULL not allowed for this parameter"

    if data_type == SQLDataType.INTEGER:
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"Expected integer, got {type(value).__name__}"
        return True, ""

    elif data_type == SQLDataType.FLOAT:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"Expected float, got {type(value).__name__}"
        return True, ""

    elif data_type == SQLDataType.STRING:
        if not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        return True, ""

    elif data_type == SQLDataType.BOOLEAN:
        if not isinstance(value, bool):
            return False, f"Expected boolean, got {type(value).__name__}"
        return True, ""

    elif data_type == SQLDataType.DATE:
        # Should be ISO 8601 string YYYY-MM-DD
        if not isinstance(value, str):
            return False, f"Expected date string, got {type(value).__name__}"
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return False, "Date must be ISO 8601 format (YYYY-MM-DD)"
        return True, ""

    elif data_type == SQLDataType.DATETIME:
        # Should be ISO 8601 string YYYY-MM-DDTHH:MM:SS
        if not isinstance(value, str):
            return False, f"Expected datetime string, got {type(value).__name__}"
        if not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
            return False, "DateTime must be ISO 8601 format (YYYY-MM-DDTHH:MM:SS)"
        return True, ""

    return False, f"Unknown data type: {data_type}"


def escape_sql_identifier(identifier: str) -> str:
    """
    Escape SQL identifier (table, column, database name).

    WARNING: For values, use parameterized queries instead!
    This is only for structural identifiers.

    Uses backtick quoting (MySQL style), which is also safe in PostgreSQL.

    Args:
        identifier: Identifier to escape

    Returns:
        Escaped identifier safe for SQL

    Raises:
        ValueError: If identifier is invalid

    Example:
        >>> table = escape_sql_identifier('users')
        >>> assert table == '`users`'
        >>> # Use: f'SELECT * FROM {table}'
    """
    # Remove any existing quotes
    identifier = identifier.strip('`"\'')

    # Must be alphanumeric or underscore, starting with letter/underscore
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(
            f"Invalid SQL identifier '{identifier}'. "
            "Must start with letter/underscore, contain only alphanumeric/underscore."
        )

    # Limit length
    if len(identifier) > 255:
        raise ValueError("Identifier exceeds 255 characters")

    # Quote with backticks
    return f'`{identifier}`'


def escape_sql_column_list(columns: List[str]) -> str:
    """
    Escape list of column names.

    Args:
        columns: List of column names

    Returns:
        Escaped comma-separated columns

    Example:
        >>> cols = escape_sql_column_list(['id', 'name', 'email'])
        >>> # Use: f'SELECT {cols} FROM users'
    """
    if not columns:
        raise ValueError("Column list cannot be empty")

    escaped = [escape_sql_identifier(col) for col in columns]
    return ', '.join(escaped)


def escape_sql_value(value: Any) -> str:
    """
    Escape SQL value literal.

    WARNING: Prefer parameterized queries when possible!
    Use this only when absolutely necessary.

    Args:
        value: Python value to escape

    Returns:
        SQL-safe literal

    Example:
        >>> # DO NOT DO THIS (vulnerable):
        >>> query = f"SELECT * FROM users WHERE name = '{value}'"
        >>>
        >>> # DO THIS INSTEAD (safe):
        >>> value_param = escape_sql_value(value)
        >>> query = f"SELECT * FROM users WHERE name = {value_param}"
        >>>
        >>> # OR BETTER (parameterized):
        >>> query = "SELECT * FROM users WHERE name = %(name)s"
        >>> cursor.execute(query, {'name': value})
    """
    if value is None:
        return 'NULL'

    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    raise ValueError(f"Cannot escape value of type {type(value)}")


# ============================================================================
# Query Builder
# ============================================================================

class SQLQueryBuilder:
    """
    Safe SQL query builder that prevents injection through parameterization.

    Usage:
        >>> query = (SQLQueryBuilder()
        ...     .select(['id', 'name', 'email'])
        ...     .from_table('users')
        ...     .where('role', SQLOperator.EQUALS, 'admin')
        ...     .where('active', SQLOperator.EQUALS, True)
        ...     .order_by('created_at', SQLOrder.DESC)
        ...     .limit(10)
        ...     .build())
        >>> # query.template = parameterized SQL template
        >>> # query.params = parameter dictionary
    """

    def __init__(self):
        """Initialize query builder."""
        self.select_columns: List[str] = []
        self.from_table_name: Optional[str] = None
        self.where_conditions: List[Tuple[str, str, str]] = []  # (column, op, param_name)
        self.order_clauses: List[Tuple[str, str]] = []  # (column, direction)
        self.limit_value: Optional[int] = None
        self.offset_value: Optional[int] = None
        self.parameters: Dict[str, Tuple[Any, SQLDataType]] = {}
        self.param_counter = 0

    def select(self, columns: List[str]) -> 'SQLQueryBuilder':
        """Add SELECT clause."""
        if not columns:
            raise ValueError("Column list cannot be empty")
        self.select_columns = columns
        return self

    def from_table(self, table: str) -> 'SQLQueryBuilder':
        """Add FROM clause."""
        self.from_table_name = escape_sql_identifier(table)
        return self

    def where(self, column: str, operator: SQLOperator, value: Any) -> 'SQLQueryBuilder':
        """
        Add WHERE condition (AND).

        Args:
            column: Column name
            operator: Comparison operator
            value: Parameter value (will be parameterized)

        Returns:
            Self for chaining
        """
        column = escape_sql_identifier(column)

        # Generate parameter name
        param_name = f'param_{self.param_counter}'
        self.param_counter += 1

        # Determine parameter type from value
        if value is None:
            data_type = SQLDataType.NULL
        elif isinstance(value, bool):
            data_type = SQLDataType.BOOLEAN
        elif isinstance(value, int):
            data_type = SQLDataType.INTEGER
        elif isinstance(value, float):
            data_type = SQLDataType.FLOAT
        else:
            data_type = SQLDataType.STRING

        # Validate parameter
        valid, error_msg = validate_parameter(value, data_type)
        if not valid:
            raise ValueError(f"Invalid parameter for column '{column}': {error_msg}")

        # Store parameter
        self.parameters[param_name] = (value, data_type)

        # Add condition
        if operator in (SQLOperator.IS_NULL, SQLOperator.IS_NOT_NULL):
            # These operators don't need parameters
            self.where_conditions.append((column, operator.value, ''))
        else:
            self.where_conditions.append((column, operator.value, param_name))

        return self

    def order_by(self, column: str, direction: SQLOrder = SQLOrder.ASC) -> 'SQLQueryBuilder':
        """Add ORDER BY clause."""
        column = escape_sql_identifier(column)
        self.order_clauses.append((column, direction.value))
        return self

    def limit(self, count: int) -> 'SQLQueryBuilder':
        """Add LIMIT clause."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("LIMIT must be non-negative integer")
        if count > 1_000_000:  # Sanity limit
            raise ValueError("LIMIT value too large")
        self.limit_value = count
        return self

    def offset(self, count: int) -> 'SQLQueryBuilder':
        """Add OFFSET clause."""
        if not isinstance(count, int) or count < 0:
            raise ValueError("OFFSET must be non-negative integer")
        self.offset_value = count
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build the SQL query.

        Returns:
            Tuple of (query_template, parameters)

        The query_template should be used with:
        - PyMySQL: cursor.execute(template, parameters)
        - psycopg2: cursor.execute(template, parameters.values())
        - SQLAlchemy: text(template).bindparams(**parameters)
        """
        if not self.select_columns:
            raise ValueError("SELECT columns not specified")

        if not self.from_table_name:
            raise ValueError("FROM table not specified")

        # Build SELECT
        columns = ', '.join(escape_sql_identifier(col) for col in self.select_columns)
        query = f'SELECT {columns} FROM {self.from_table_name}'

        # Build WHERE
        if self.where_conditions:
            where_clauses = []
            for column, operator, param_name in self.where_conditions:
                if param_name:  # Has parameter
                    where_clauses.append(f'{column} {operator} %({param_name})s')
                else:  # IS NULL / IS NOT NULL
                    where_clauses.append(f'{column} {operator}')

            query += ' WHERE ' + ' AND '.join(where_clauses)

        # Build ORDER BY
        if self.order_clauses:
            order = ', '.join(f'{col} {dir}' for col, dir in self.order_clauses)
            query += f' ORDER BY {order}'

        # Build LIMIT/OFFSET
        if self.limit_value is not None:
            query += f' LIMIT {self.limit_value}'

        if self.offset_value is not None:
            query += f' OFFSET {self.offset_value}'

        # Extract just values from parameters (remove type info)
        params = {k: v[0] for k, v in self.parameters.items()}

        return query, params


# ============================================================================
# Common SQL Pattern Builders
# ============================================================================

def build_select_query(
    table: str,
    columns: Optional[List[str]] = None,
    where_conditions: Optional[Dict[str, Any]] = None,
    limit: int = 100,
    offset: int = 0
) -> Tuple[str, Dict[str, Any]]:
    """
    Build safe SELECT query.

    Args:
        table: Table name
        columns: Columns to select (None = *)
        where_conditions: WHERE clause conditions {column: value}
        limit: LIMIT clause
        offset: OFFSET clause

    Returns:
        Tuple of (query, parameters)

    Example:
        >>> query, params = build_select_query(
        ...     'users',
        ...     ['id', 'name'],
        ...     {'role': 'admin', 'active': True},
        ...     limit=10
        ... )
    """
    builder = SQLQueryBuilder()

    # SELECT
    if columns is None or columns == ['*']:
        columns = ['*']
    builder.select(columns)

    # FROM
    builder.from_table(table)

    # WHERE
    if where_conditions:
        for column, value in where_conditions.items():
            builder.where(column, SQLOperator.EQUALS, value)

    # LIMIT/OFFSET
    builder.limit(limit)
    if offset > 0:
        builder.offset(offset)

    return builder.build()


def build_insert_query(
    table: str,
    values: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Build safe INSERT query.

    Args:
        table: Table name
        values: {column: value} dictionary

    Returns:
        Tuple of (query, parameters)

    Example:
        >>> query, params = build_insert_query(
        ...     'users',
        ...     {'name': 'Alice', 'email': 'alice@example.com', 'role': 'admin'}
        ... )
    """
    if not values:
        raise ValueError("INSERT values cannot be empty")

    table = escape_sql_identifier(table)
    columns = [escape_sql_identifier(col) for col in values.keys()]
    placeholders = [f'%(param_{i})s' for i in range(len(values))]

    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

    # Build parameters with names
    params = {f'param_{i}': value for i, value in enumerate(values.values())}

    return query, params


def build_update_query(
    table: str,
    updates: Dict[str, Any],
    where_conditions: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Build safe UPDATE query.

    Args:
        table: Table name
        updates: {column: new_value} dictionary
        where_conditions: {column: value} WHERE clause

    Returns:
        Tuple of (query, parameters)

    Example:
        >>> query, params = build_update_query(
        ...     'users',
        ...     {'role': 'moderator', 'active': False},
        ...     {'id': 123}
        ... )
    """
    if not updates:
        raise ValueError("UPDATE values cannot be empty")

    if not where_conditions:
        raise ValueError("UPDATE WHERE conditions required (safety check)")

    table = escape_sql_identifier(table)

    # Build SET clause
    set_clauses = []
    params = {}
    param_counter = 0

    for column, value in updates.items():
        column = escape_sql_identifier(column)
        param_name = f'update_param_{param_counter}'
        set_clauses.append(f'{column} = %({param_name})s')
        params[param_name] = value
        param_counter += 1

    # Build WHERE clause
    where_clauses = []
    for column, value in where_conditions.items():
        column = escape_sql_identifier(column)
        param_name = f'where_param_{param_counter}'
        where_clauses.append(f'{column} = %({param_name})s')
        params[param_name] = value
        param_counter += 1

    query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"

    return query, params


def build_delete_query(
    table: str,
    where_conditions: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """
    Build safe DELETE query.

    Args:
        table: Table name
        where_conditions: {column: value} WHERE clause (required safety check)

    Returns:
        Tuple of (query, parameters)

    Example:
        >>> query, params = build_delete_query(
        ...     'sessions',
        ...     {'user_id': 123}
        ... )
    """
    if not where_conditions:
        raise ValueError("DELETE WHERE conditions required (safety check)")

    table = escape_sql_identifier(table)

    # Build WHERE clause
    where_clauses = []
    params = {}
    param_counter = 0

    for column, value in where_conditions.items():
        column = escape_sql_identifier(column)
        param_name = f'where_param_{param_counter}'
        where_clauses.append(f'{column} = %({param_name})s')
        params[param_name] = value
        param_counter += 1

    query = f"DELETE FROM {table} WHERE {' AND '.join(where_clauses)}"

    return query, params


# ============================================================================
# Query Validation
# ============================================================================

def detect_sql_injection_attempt(query_string: str) -> Tuple[bool, Optional[str]]:
    """
    Detect obvious SQL injection attempts in query string.

    WARNING: This is heuristic-based detection. It's not foolproof.
    Parameterized queries are the real protection.

    Args:
        query_string: Raw query string to check

    Returns:
        Tuple of (is_suspicious, detected_pattern or None)

    Example:
        >>> is_sus, pattern = detect_sql_injection_attempt("' OR '1'='1")
        >>> assert is_sus is True
    """
    query_lower = query_string.lower()

    # Suspicious patterns
    patterns = [
        (r"'\s*or\s*'", "OR-based logic manipulation"),
        (r"'\s*union\s+select", "UNION-based injection"),
        (r";\s*(drop|delete|insert|update|create)", "Stacked query"),
        (r"--\s", "SQL comment injection"),
        (r"/\*.*?\*/", "Multi-line comment injection"),
        (r"xp_\w+", "Extended stored procedure (MSSQL)"),
        (r"sp_\w+", "System stored procedure"),
        (r"exec\s*\(", "EXEC with dynamic query"),
        (r"execute\s*\(", "EXECUTE with dynamic query"),
    ]

    for pattern, description in patterns:
        if re.search(pattern, query_lower):
            return True, description

    return False, None
