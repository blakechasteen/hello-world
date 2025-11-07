"""
Schema Registry - RAG-Powered Schema Management
================================================

Registry of database schemas with RAG-powered semantic matching.

The "wool → yarn" transformation layer:
- Stores schemas (SQL, Graph, JSON)
- Uses RAG to find best matching schema
- Suggests field mappings
- Validates transformations

Philosophy:
    "The spinner shouldn't care about your schema -
     the schema should find the spinner."

Usage:
    from HoloLoom.spinningWheel import SchemaRegistry

    # Create registry
    registry = SchemaRegistry(memory_backend)

    # Register schemas
    await registry.register_schema("expenses", expense_schema)

    # Auto-detect schema for data
    schema_name = await registry.find_best_schema(sample_data)

    # Get field mappings
    mappings = await registry.suggest_field_mapping("expenses", extracted_data)

Author: Claude Code
Date: January 2025
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from HoloLoom.documentation.types import MemoryShard


# ============================================================================
# Schema Types
# ============================================================================

class SchemaType(Enum):
    """Schema format types."""
    SQL = "sql"                    # Relational database schema
    GRAPH = "graph"                # Property graph schema
    JSON_SCHEMA = "json_schema"    # JSON Schema format
    RDF = "rdf"                    # RDF/OWL ontology
    CUSTOM = "custom"              # Custom format


@dataclass
class SchemaDefinition:
    """
    Schema definition with metadata.

    Represents a complete schema that can be used to structure data.
    """
    name: str
    schema_type: SchemaType
    version: str = "1.0"

    # Schema structure (format depends on schema_type)
    nodes: Dict[str, Any] = field(default_factory=dict)  # For graph schemas
    tables: Dict[str, Any] = field(default_factory=dict)  # For SQL schemas
    properties: Dict[str, Any] = field(default_factory=dict)  # For JSON schemas

    # Relationships/edges
    edges: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = ""
    domain: str = "general"  # Domain: finance, healthcare, etc.
    tags: List[str] = field(default_factory=list)

    # Validation rules
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'schema_type': self.schema_type.value,
            'version': self.version,
            'nodes': self.nodes,
            'tables': self.tables,
            'properties': self.properties,
            'edges': self.edges,
            'description': self.description,
            'domain': self.domain,
            'tags': self.tags,
            'constraints': self.constraints
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaDefinition':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            schema_type=SchemaType(data['schema_type']),
            version=data.get('version', '1.0'),
            nodes=data.get('nodes', {}),
            tables=data.get('tables', {}),
            properties=data.get('properties', {}),
            edges=data.get('edges', {}),
            description=data.get('description', ''),
            domain=data.get('domain', 'general'),
            tags=data.get('tags', []),
            constraints=data.get('constraints', {})
        )


@dataclass
class FieldMapping:
    """Mapping from extracted field to schema field."""
    source_field: str
    target_field: str
    confidence: float  # 0.0-1.0
    transform: Optional[str] = None  # Optional transformation function


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ============================================================================
# Schema Registry
# ============================================================================

class SchemaRegistry:
    """
    Registry of database schemas with RAG-powered matching.

    Central repository for all schemas in the system.
    Uses RAG (Retrieval-Augmented Generation) to:
    - Find best matching schema for unknown data
    - Suggest field mappings
    - Learn from successful transformations
    """

    def __init__(
        self,
        memory_backend: Optional[Any] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize schema registry.

        Args:
            memory_backend: HoloLoom memory backend for RAG
            storage_path: Path to persist schemas (None = in-memory only)
        """
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.memory = memory_backend
        self.storage_path = Path(storage_path) if storage_path else None

        # Create storage directory
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Schema usage statistics (for learning)
        self.usage_stats: Dict[str, int] = {}

        # Field mapping cache (for performance)
        self._mapping_cache: Dict[str, Dict[str, str]] = {}

    # ========================================================================
    # Schema Registration
    # ========================================================================

    async def register_schema(
        self,
        name: str,
        schema: Dict[str, Any],
        schema_type: SchemaType = SchemaType.GRAPH
    ) -> None:
        """
        Register a schema and store in RAG memory.

        Args:
            name: Schema name (unique identifier)
            schema: Schema definition (dict format)
            schema_type: Type of schema

        Example:
            >>> await registry.register_schema("expenses", {
            ...     'nodes': {
            ...         'Transaction': {'properties': {'date': 'date', 'amount': 'float'}},
            ...         'Merchant': {'properties': {'name': 'string'}}
            ...     },
            ...     'edges': {
            ...         'PURCHASED_FROM': {'from': 'Transaction', 'to': 'Merchant'}
            ...     }
            ... })
        """
        # Create schema definition
        schema_def = SchemaDefinition(
            name=name,
            schema_type=schema_type,
            nodes=schema.get('nodes', {}),
            tables=schema.get('tables', {}),
            properties=schema.get('properties', {}),
            edges=schema.get('edges', {}),
            description=schema.get('description', ''),
            domain=schema.get('domain', 'general'),
            tags=schema.get('tags', []),
            constraints=schema.get('constraints', {})
        )

        # Store locally
        self.schemas[name] = schema_def

        # Initialize usage stats
        self.usage_stats[name] = 0

        # Store in RAG memory for semantic matching
        if self.memory:
            await self._store_in_rag(schema_def)

        # Persist to disk
        if self.storage_path:
            await self._persist_schema(schema_def)

        print(f"[SchemaRegistry] Registered schema: {name} ({schema_type.value})")

    async def _store_in_rag(self, schema: SchemaDefinition) -> None:
        """Store schema in RAG memory for semantic matching."""
        # Convert schema to searchable text
        schema_text = self._schema_to_text(schema)

        # Create memory shard
        shard = MemoryShard(
            id=f"schema_{schema.name}",
            text=schema_text,
            episode="schema_registry",
            entities=[schema.name, schema.domain],
            motifs=['schema', 'database', schema.schema_type.value] + schema.tags,
            metadata={
                'type': 'schema_definition',
                'schema_name': schema.name,
                'schema_type': schema.schema_type.value,
                'domain': schema.domain,
                'version': schema.version,
                'node_types': list(schema.nodes.keys()),
                'table_names': list(schema.tables.keys())
            }
        )

        # Store in memory (assuming memory has add_shard or similar method)
        # Note: This is a placeholder - actual implementation depends on memory backend
        try:
            if hasattr(self.memory, 'add_shard'):
                await self.memory.add_shard(shard)
            elif hasattr(self.memory, 'experience'):
                await self.memory.experience(schema_text)
        except Exception as e:
            print(f"Warning: Failed to store schema in RAG: {e}")

    def _schema_to_text(self, schema: SchemaDefinition) -> str:
        """
        Convert schema to searchable text description.

        Creates a natural language description of the schema for RAG matching.
        """
        parts = [f"Schema: {schema.name}"]

        if schema.description:
            parts.append(f"Description: {schema.description}")

        parts.append(f"Domain: {schema.domain}")

        # Describe nodes/tables
        if schema.nodes:
            parts.append("\nNode Types:")
            for node_name, node_def in schema.nodes.items():
                props = node_def.get('properties', {})
                prop_str = ', '.join(f"{k}:{v}" for k, v in props.items())
                parts.append(f"  - {node_name}: {prop_str}")

        if schema.tables:
            parts.append("\nTables:")
            for table_name, table_def in schema.tables.items():
                cols = table_def.get('columns', [])
                col_str = ', '.join(c.get('name', '') for c in cols)
                parts.append(f"  - {table_name}: {col_str}")

        # Describe edges
        if schema.edges:
            parts.append("\nRelationships:")
            for edge_name, edge_def in schema.edges.items():
                from_node = edge_def.get('from', '?')
                to_node = edge_def.get('to', '?')
                parts.append(f"  - {edge_name}: {from_node} → {to_node}")

        # Add tags
        if schema.tags:
            parts.append(f"\nTags: {', '.join(schema.tags)}")

        return '\n'.join(parts)

    async def _persist_schema(self, schema: SchemaDefinition) -> None:
        """Persist schema to disk."""
        if not self.storage_path:
            return

        schema_file = self.storage_path / f"{schema.name}.json"

        with open(schema_file, 'w') as f:
            json.dump(schema.to_dict(), f, indent=2)

    # ========================================================================
    # Schema Discovery (RAG-Powered)
    # ========================================================================

    async def find_best_schema(
        self,
        sample_data: str,
        k: int = 3
    ) -> Optional[str]:
        """
        Use RAG to find best matching schema for data.

        Args:
            sample_data: Sample of the data (text representation)
            k: Number of candidates to consider

        Returns:
            Schema name (None if no good match)

        Example:
            >>> schema = await registry.find_best_schema('''
            ... WHOLE FOODS MARKET
            ... Organic Bananas $3.99
            ... Total: $45.44
            ... ''')
            >>> print(schema)  # "expenses"
        """
        if not self.memory:
            # Fallback: simple keyword matching
            return self._fallback_schema_match(sample_data)

        # Query RAG for similar schemas
        query = f"""What database schema best fits this data?

Data sample:
{sample_data[:1000]}

Look for schemas that match:
- The data structure and fields
- The domain (finance, tasks, notes, etc.)
- The relationships present"""

        try:
            # Query memory
            if hasattr(self.memory, 'recall'):
                results = await self.memory.recall(query, k=k)
            else:
                # Fallback
                return self._fallback_schema_match(sample_data)

            # Extract schema names from results
            for result in results:
                schema_name = result.metadata.get('schema_name')
                if schema_name and schema_name in self.schemas:
                    # Update usage stats
                    self.usage_stats[schema_name] = self.usage_stats.get(schema_name, 0) + 1
                    return schema_name

        except Exception as e:
            print(f"Warning: RAG schema matching failed: {e}")

        # Fallback
        return self._fallback_schema_match(sample_data)

    def _fallback_schema_match(self, sample_data: str) -> Optional[str]:
        """Fallback schema matching using keywords."""
        sample_lower = sample_data.lower()

        # Receipt/expense indicators
        if any(kw in sample_lower for kw in ['total', 'tax', 'receipt', 'merchant', 'purchased']):
            return 'expenses' if 'expenses' in self.schemas else None

        # Task/note indicators
        if any(kw in sample_lower for kw in ['todo', 'task', '[ ]', 'meeting', 'notes']):
            return 'tasks' if 'tasks' in self.schemas else None

        # Return most-used schema as last resort
        if self.usage_stats:
            most_used = max(self.usage_stats.items(), key=lambda x: x[1])
            return most_used[0]

        return None

    # ========================================================================
    # Field Mapping (RAG-Powered)
    # ========================================================================

    async def suggest_field_mapping(
        self,
        schema_name: str,
        extracted_data: Dict[str, Any],
        confidence_threshold: float = 0.5
    ) -> Dict[str, FieldMapping]:
        """
        Suggest how to map extracted fields to schema fields.

        Args:
            schema_name: Target schema name
            extracted_data: Extracted data (field → value)
            confidence_threshold: Min confidence to suggest

        Returns:
            Dict of field_name → FieldMapping

        Example:
            >>> mappings = await registry.suggest_field_mapping("expenses", {
            ...     'merchant_name': 'Whole Foods',
            ...     'purchase_date': '01/15/2025',
            ...     'total_amount': 45.44
            ... })
            >>> print(mappings['merchant_name'].target_field)  # 'merchant.name'
        """
        # Check cache
        cache_key = f"{schema_name}:{','.join(sorted(extracted_data.keys()))}"
        if cache_key in self._mapping_cache:
            return self._mapping_cache[cache_key]

        schema = self.schemas.get(schema_name)
        if not schema:
            return {}

        mappings = {}

        # Get all schema fields
        schema_fields = self._get_schema_fields(schema)

        # Map each extracted field
        for field_name, field_value in extracted_data.items():
            # Try RAG-based mapping first
            if self.memory:
                target_field = await self._rag_field_mapping(
                    schema_name,
                    field_name,
                    field_value,
                    schema_fields
                )
            else:
                # Fallback: heuristic matching
                target_field = self._heuristic_field_mapping(
                    field_name,
                    schema_fields
                )

            if target_field:
                mappings[field_name] = FieldMapping(
                    source_field=field_name,
                    target_field=target_field,
                    confidence=0.8 if self.memory else 0.6
                )

        # Cache mappings
        self._mapping_cache[cache_key] = mappings

        return mappings

    async def _rag_field_mapping(
        self,
        schema_name: str,
        field_name: str,
        field_value: Any,
        schema_fields: List[str]
    ) -> Optional[str]:
        """Use RAG to find best field mapping."""
        query = f"""In schema '{schema_name}', which field best matches:

Field name: {field_name}
Sample value: {field_value}

Available schema fields: {', '.join(schema_fields)}

Return only the field name."""

        try:
            if hasattr(self.memory, 'recall'):
                results = await self.memory.recall(query, k=1)
                if results:
                    suggested = results[0].metadata.get('mapped_field')
                    if suggested in schema_fields:
                        return suggested
        except Exception as e:
            print(f"Warning: RAG field mapping failed: {e}")

        return None

    def _heuristic_field_mapping(
        self,
        field_name: str,
        schema_fields: List[str]
    ) -> Optional[str]:
        """Heuristic field mapping based on name similarity."""
        field_lower = field_name.lower().replace('_', ' ')

        # Direct match
        if field_name in schema_fields:
            return field_name

        # Fuzzy match
        for schema_field in schema_fields:
            schema_lower = schema_field.lower().replace('_', ' ').replace('.', ' ')

            # Check if names overlap significantly
            if field_lower in schema_lower or schema_lower in field_lower:
                return schema_field

        return None

    def _get_schema_fields(self, schema: SchemaDefinition) -> List[str]:
        """Get all fields from schema."""
        fields = []

        # Extract from nodes
        for node_name, node_def in schema.nodes.items():
            props = node_def.get('properties', {})
            for prop_name in props.keys():
                fields.append(f"{node_name}.{prop_name}")

        # Extract from tables
        for table_name, table_def in schema.tables.items():
            cols = table_def.get('columns', [])
            for col in cols:
                col_name = col.get('name', '')
                if col_name:
                    fields.append(f"{table_name}.{col_name}")

        return fields

    # ========================================================================
    # Schema Validation
    # ========================================================================

    def validate_data(
        self,
        schema_name: str,
        data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate data against schema.

        Args:
            schema_name: Schema to validate against
            data: Data to validate

        Returns:
            ValidationResult with errors/warnings
        """
        schema = self.schemas.get(schema_name)
        if not schema:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema '{schema_name}' not found"]
            )

        errors = []
        warnings = []
        suggestions = []

        # Validate based on schema type
        if schema.schema_type == SchemaType.GRAPH:
            # Validate graph structure
            if 'nodes' in data:
                for node in data['nodes']:
                    node_type = node.get('type')
                    if node_type not in schema.nodes:
                        errors.append(f"Unknown node type: {node_type}")

                    # Validate properties
                    expected_props = schema.nodes.get(node_type, {}).get('properties', {})
                    for prop_name, prop_value in node.get('properties', {}).items():
                        if prop_name not in expected_props:
                            warnings.append(f"Unexpected property: {node_type}.{prop_name}")

                        # Type check
                        expected_type = expected_props.get(prop_name)
                        if expected_type and not self._check_type(prop_value, expected_type):
                            errors.append(
                                f"Type mismatch: {node_type}.{prop_name} "
                                f"expected {expected_type}, got {type(prop_value).__name__}"
                            )

            if 'edges' in data:
                for edge in data['edges']:
                    edge_type = edge.get('type')
                    if edge_type not in schema.edges:
                        errors.append(f"Unknown edge type: {edge_type}")

        is_valid = len(errors) == 0

        if not is_valid:
            suggestions.append("Check field names and types against schema")
            suggestions.append("Use suggest_field_mapping() for automatic mapping")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'date': str,  # Simplified
            'datetime': str,
            'text': str
        }

        expected_python_type = type_map.get(expected_type.lower())
        if not expected_python_type:
            return True  # Unknown type, assume valid

        return isinstance(value, expected_python_type)

    # ========================================================================
    # Schema Management
    # ========================================================================

    def get_schema(self, name: str) -> Optional[SchemaDefinition]:
        """Get schema by name."""
        return self.schemas.get(name)

    def list_schemas(self) -> List[str]:
        """List all registered schema names."""
        return list(self.schemas.keys())

    def get_usage_stats(self) -> Dict[str, int]:
        """Get schema usage statistics."""
        return self.usage_stats.copy()

    async def load_schemas_from_directory(self, directory: Path) -> int:
        """Load all schemas from directory."""
        directory = Path(directory)
        if not directory.exists():
            return 0

        loaded = 0
        for schema_file in directory.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_dict = json.load(f)

                schema = SchemaDefinition.from_dict(schema_dict)
                self.schemas[schema.name] = schema
                loaded += 1

            except Exception as e:
                print(f"Warning: Failed to load schema {schema_file}: {e}")

        return loaded


# ============================================================================
# Convenience Functions
# ============================================================================

def create_expense_schema() -> Dict[str, Any]:
    """Create standard expense tracking schema."""
    return {
        'description': 'Expense tracking schema for receipts and transactions',
        'domain': 'finance',
        'tags': ['expense', 'receipt', 'finance'],
        'nodes': {
            'Transaction': {
                'properties': {
                    'date': 'date',
                    'time': 'time',
                    'total': 'float',
                    'subtotal': 'float',
                    'tax': 'float',
                    'category': 'string'
                }
            },
            'Merchant': {
                'properties': {
                    'name': 'string',
                    'address': 'string',
                    'category': 'string'
                }
            },
            'Item': {
                'properties': {
                    'name': 'string',
                    'quantity': 'float',
                    'price': 'float'
                }
            }
        },
        'edges': {
            'PURCHASED_FROM': {'from': 'Transaction', 'to': 'Merchant'},
            'INCLUDES': {'from': 'Transaction', 'to': 'Item'}
        }
    }


def create_task_schema() -> Dict[str, Any]:
    """Create standard task management schema."""
    return {
        'description': 'Task management schema for notes and action items',
        'domain': 'productivity',
        'tags': ['task', 'note', 'todo'],
        'nodes': {
            'Note': {
                'properties': {
                    'title': 'string',
                    'date': 'date',
                    'author': 'string',
                    'content': 'text'
                }
            },
            'Task': {
                'properties': {
                    'description': 'string',
                    'status': 'string',
                    'priority': 'string',
                    'due_date': 'date'
                }
            },
            'Person': {
                'properties': {
                    'name': 'string',
                    'email': 'string'
                }
            }
        },
        'edges': {
            'HAS_TASK': {'from': 'Note', 'to': 'Task'},
            'ASSIGNED_TO': {'from': 'Task', 'to': 'Person'},
            'MENTIONS': {'from': 'Note', 'to': 'Person'}
        }
    }