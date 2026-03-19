# Schema-Aware Spinners: RAG-Powered Wool → Yarn Graph Transformation

**Concept**: January 2025
**Status**: Design Phase
**Innovation**: Spinners that understand database schemas and use RAG to intelligently map raw data ("wool") into structured Yarn Graph ("yarn")

---

## Vision

Imagine spinners that don't just extract text - they **understand the semantic structure** you want and intelligently map raw data into it:

```
RAW DATA (Wool)                SPINNER WITH SCHEMA AWARENESS              YARN GRAPH
┌─────────────┐               ┌───────────────────────────┐            ┌──────────────┐
│  Receipt    │               │  1. Extract text (OCR)    │            │   Entities   │
│  Image      │──────────────>│  2. RAG: "What schema     │───────────>│   + Edges    │
│             │               │      fits this data?"     │            │   + Props    │
└─────────────┘               │  3. Validate & transform  │            └──────────────┘
                              │  4. Generate KG nodes     │
┌─────────────┐               └───────────────────────────┘            ┌──────────────┐
│ Handwritten │                                                        │  Task nodes  │
│  Note       │──────────────> Schema-aware transformation ───────────>│  Person nodes│
│             │                                                        │  Date nodes  │
└─────────────┘                                                        └──────────────┘
```

### The Big Idea

**Current State**: Spinners extract text and basic entities
```python
result = await spinner.spin("receipt.jpg")
# Returns: text, entities=['merchant', 'date'], motifs=['receipt']
```

**Future State**: Spinners produce structured graph data
```python
result = await spinner.spin("receipt.jpg", schema="receipts_db")
# Returns: KG nodes + edges + properties that match your schema
# Automatic validation, type coercion, relationship inference
```

---

## Architecture

### 1. Schema Registry (RAG-Powered)

```python
class SchemaRegistry:
    """
    Registry of database schemas with RAG-powered matching.

    Stores:
    - Database schemas (tables, columns, types, relationships)
    - Document schemas (JSON, XML structures)
    - Graph schemas (node types, edge types, properties)

    Uses RAG to:
    - Find best schema for unknown data
    - Suggest field mappings
    - Validate transformations
    """

    def __init__(self, memory_backend):
        self.schemas = {}
        self.memory = memory_backend  # For RAG

    async def register_schema(self, name: str, schema: Dict):
        """Register a schema and store in RAG memory."""
        self.schemas[name] = schema

        # Store in RAG for semantic matching
        await self.memory.add_shard(MemoryShard(
            id=f"schema_{name}",
            text=self._schema_to_text(schema),
            entities=[name],
            motifs=['schema', 'database'],
            metadata={'schema': schema, 'type': 'schema_definition'}
        ))

    async def find_best_schema(self, sample_data: str) -> Optional[str]:
        """Use RAG to find best matching schema."""
        # Query RAG memory with sample data
        query = f"What database schema best fits this data?\n{sample_data[:500]}"
        results = await self.memory.recall(query, k=3)

        # Return schema with highest similarity
        if results:
            return results[0].metadata.get('schema')
        return None

    async def suggest_field_mapping(
        self,
        schema_name: str,
        extracted_data: Dict
    ) -> Dict[str, str]:
        """Suggest how to map extracted fields to schema."""
        schema = self.schemas.get(schema_name)
        if not schema:
            return {}

        # Use RAG to find similar field mappings
        mappings = {}
        for field_name, field_value in extracted_data.items():
            query = f"In schema {schema_name}, which field matches: {field_name} = {field_value}"
            results = await self.memory.recall(query, k=1)

            if results:
                suggested_field = results[0].metadata.get('mapped_field')
                mappings[field_name] = suggested_field

        return mappings
```

### 2. Schema-Aware Spinners

Enhance spinners with schema awareness:

```python
class SchemaAwareSpinner(BaseSpinner):
    """Base class for schema-aware spinners."""

    def __init__(
        self,
        schema_registry: SchemaRegistry,
        auto_detect_schema: bool = True,
        validate_output: bool = True
    ):
        self.schema_registry = schema_registry
        self.auto_detect_schema = auto_detect_schema
        self.validate_output = validate_output

    async def spin_with_schema(
        self,
        source: Any,
        schema_name: Optional[str] = None,
        **kwargs
    ) -> GraphOutput:
        """
        Spin with schema awareness.

        Returns GraphOutput instead of MemoryShard:
        - nodes: List of graph nodes
        - edges: List of graph edges
        - validation_results: Schema validation info
        """
        # 1. Extract raw data (existing OCR/parsing logic)
        raw_shards = await self._spin_impl(source, **kwargs)

        # 2. Auto-detect schema if needed
        if schema_name is None and self.auto_detect_schema:
            sample_text = raw_shards[0].text if raw_shards else ""
            schema_name = await self.schema_registry.find_best_schema(sample_text)

        # 3. Transform to graph structure
        graph_output = await self._transform_to_graph(
            raw_shards,
            schema_name,
            **kwargs
        )

        # 4. Validate against schema
        if self.validate_output and schema_name:
            validation = await self._validate_graph(graph_output, schema_name)
            graph_output.validation_results = validation

        return graph_output

    async def _transform_to_graph(
        self,
        shards: List[MemoryShard],
        schema_name: Optional[str],
        **kwargs
    ) -> GraphOutput:
        """Transform raw shards into graph structure."""
        # Extract structured data from shards
        structured_data = self._extract_structured_data(shards)

        # Get schema
        schema = self.schema_registry.schemas.get(schema_name) if schema_name else None

        if schema:
            # Map to schema using RAG
            mappings = await self.schema_registry.suggest_field_mapping(
                schema_name,
                structured_data
            )

            # Transform with schema
            nodes, edges = await self._apply_schema_transform(
                structured_data,
                schema,
                mappings
            )
        else:
            # Generic transform (no schema)
            nodes, edges = await self._generic_transform(structured_data)

        return GraphOutput(
            nodes=nodes,
            edges=edges,
            schema_name=schema_name,
            raw_shards=shards
        )
```

### 3. Specific Schema-Aware Implementations

**Receipt Spinner with Schema**:

```python
class SchemaAwareReceiptSpinner(SchemaAwareSpinner, ReceiptSpinner):
    """Receipt spinner that produces structured graph data."""

    async def _extract_structured_data(
        self,
        shards: List[MemoryShard]
    ) -> Dict:
        """Extract structured receipt data."""
        receipt_data = shards[0].metadata.get('receipt_data', {})

        return {
            'merchant': {
                'name': receipt_data.get('merchant'),
                'address': receipt_data.get('merchant_address')
            },
            'transaction': {
                'date': receipt_data.get('date'),
                'time': receipt_data.get('time'),
                'total': receipt_data.get('total'),
                'subtotal': receipt_data.get('subtotal'),
                'tax': receipt_data.get('tax'),
                'payment_method': receipt_data.get('payment_method'),
                'category': receipt_data.get('category')
            },
            'items': [
                {
                    'name': item['name'],
                    'quantity': item.get('quantity', 1),
                    'price': item.get('total_price')
                }
                for item in receipt_data.get('items', [])
            ]
        }

    async def _apply_schema_transform(
        self,
        data: Dict,
        schema: Dict,
        mappings: Dict
    ) -> Tuple[List[Node], List[Edge]]:
        """Transform receipt data to graph nodes/edges."""
        nodes = []
        edges = []

        # Create merchant node
        merchant_node = Node(
            id=f"merchant_{hash(data['merchant']['name'])}",
            type='Merchant',
            properties={
                'name': data['merchant']['name'],
                'address': data['merchant'].get('address')
            }
        )
        nodes.append(merchant_node)

        # Create transaction node
        transaction_node = Node(
            id=f"transaction_{data['transaction']['date']}_{data['transaction']['time']}",
            type='Transaction',
            properties={
                'date': data['transaction']['date'],
                'time': data['transaction']['time'],
                'total': float(data['transaction']['total']) if data['transaction']['total'] else 0,
                'subtotal': float(data['transaction']['subtotal']) if data['transaction']['subtotal'] else 0,
                'tax': float(data['transaction']['tax']) if data['transaction']['tax'] else 0,
                'category': data['transaction']['category']
            }
        )
        nodes.append(transaction_node)

        # Create edge: Transaction -> Merchant
        edges.append(Edge(
            source=transaction_node.id,
            target=merchant_node.id,
            type='PURCHASED_FROM',
            properties={}
        ))

        # Create item nodes
        for i, item in enumerate(data['items']):
            item_node = Node(
                id=f"item_{transaction_node.id}_{i}",
                type='Item',
                properties={
                    'name': item['name'],
                    'quantity': item['quantity'],
                    'price': float(item['price']) if item['price'] else 0
                }
            )
            nodes.append(item_node)

            # Edge: Transaction -> Item
            edges.append(Edge(
                source=transaction_node.id,
                target=item_node.id,
                type='INCLUDES',
                properties={'quantity': item['quantity']}
            ))

        return nodes, edges
```

**Handwritten Spinner with Schema**:

```python
class SchemaAwareHandwrittenSpinner(SchemaAwareSpinner, HandwrittenSpinner):
    """Handwritten spinner that produces task graph."""

    async def _extract_structured_data(
        self,
        shards: List[MemoryShard]
    ) -> Dict:
        """Extract structured note data."""
        metadata = shards[0].metadata

        return {
            'note': {
                'title': self._extract_title(shards[0].text),
                'date': self._extract_date(shards[0].text),
                'author': self._extract_author(shards[0].text)
            },
            'tasks': [
                {'description': task, 'status': 'pending'}
                for task in metadata.get('detected_tasks', [])
            ],
            'entities': shards[0].entities,
            'sections': metadata.get('sections', [])
        }

    async def _apply_schema_transform(
        self,
        data: Dict,
        schema: Dict,
        mappings: Dict
    ) -> Tuple[List[Node], List[Edge]]:
        """Transform note data to graph nodes/edges."""
        nodes = []
        edges = []

        # Create note node
        note_node = Node(
            id=f"note_{hash(data['note']['title'])}",
            type='Note',
            properties={
                'title': data['note']['title'],
                'date': data['note']['date'],
                'author': data['note']['author']
            }
        )
        nodes.append(note_node)

        # Create task nodes
        for i, task in enumerate(data['tasks']):
            task_node = Node(
                id=f"task_{note_node.id}_{i}",
                type='Task',
                properties={
                    'description': task['description'],
                    'status': task['status'],
                    'priority': 'normal'
                }
            )
            nodes.append(task_node)

            # Edge: Note -> Task
            edges.append(Edge(
                source=note_node.id,
                target=task_node.id,
                type='HAS_TASK',
                properties={}
            ))

        # Create person nodes from entities
        for entity in data['entities']:
            if self._is_person_name(entity):
                person_node = Node(
                    id=f"person_{hash(entity)}",
                    type='Person',
                    properties={'name': entity}
                )
                nodes.append(person_node)

                # Edge: Note -> Person (mentions)
                edges.append(Edge(
                    source=note_node.id,
                    target=person_node.id,
                    type='MENTIONS',
                    properties={}
                ))

        return nodes, edges
```

### 4. RAG-Powered Schema Suggestion

Use RAG to suggest schemas based on data patterns:

```python
class SchemaSuggestionEngine:
    """RAG-powered schema suggestion."""

    def __init__(self, memory_backend, schema_registry):
        self.memory = memory_backend
        self.registry = schema_registry

    async def suggest_schema_from_data(
        self,
        sample_data: str,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Suggest schemas that match the data.

        Args:
            sample_data: Sample of the data
            k: Number of suggestions

        Returns:
            List of (schema_name, confidence) tuples
        """
        # Query RAG for similar data patterns
        query = f"""Analyze this data and suggest appropriate database schema:

Data sample:
{sample_data[:1000]}

What schema would best represent this data?"""

        results = await self.memory.recall(query, k=k)

        suggestions = []
        for result in results:
            schema_name = result.metadata.get('schema_name')
            confidence = result.metadata.get('similarity', 0.0)

            if schema_name:
                suggestions.append((schema_name, confidence))

        return suggestions

    async def suggest_field_types(
        self,
        field_name: str,
        sample_values: List[Any]
    ) -> Dict[str, Any]:
        """
        Suggest field type and constraints.

        Returns:
            {
                'type': 'string' | 'integer' | 'float' | 'date' | 'boolean',
                'constraints': {'max_length': 100, 'nullable': False, ...},
                'confidence': 0.95
            }
        """
        # Analyze sample values
        query = f"""What data type should be used for field '{field_name}' with these values?

Sample values: {sample_values[:10]}

Consider:
- Data type (string, integer, float, date, boolean)
- Constraints (max length, nullable, unique)
- Format patterns"""

        results = await self.memory.recall(query, k=1)

        if results:
            return results[0].metadata.get('suggested_type', {})

        # Fallback: infer from values
        return self._infer_type_from_values(sample_values)
```

### 5. Yarn Graph Integration

Direct integration with Yarn Graph:

```python
class WoolToYarnPipeline:
    """
    Complete pipeline: Raw Data (Wool) → Structured Graph (Yarn)
    """

    def __init__(
        self,
        schema_registry: SchemaRegistry,
        yarn_graph: YarnGraph  # HoloLoom.memory.graph.KG
    ):
        self.registry = schema_registry
        self.yarn = yarn_graph

    async def process(
        self,
        source: Any,
        spinner: SchemaAwareSpinner,
        schema_name: Optional[str] = None,
        merge_strategy: str = "upsert"
    ) -> ProcessingResult:
        """
        Complete wool → yarn transformation.

        Steps:
        1. Spin: Extract data from source (wool)
        2. Transform: Apply schema and create graph structure
        3. Validate: Check against schema constraints
        4. Weave: Insert into Yarn Graph

        Args:
            source: Raw data source (image, PDF, etc.)
            spinner: Schema-aware spinner
            schema_name: Target schema (None = auto-detect)
            merge_strategy: How to handle existing data

        Returns:
            ProcessingResult with nodes added, edges created, validation info
        """
        # Step 1: Spin (extract)
        graph_output = await spinner.spin_with_schema(
            source,
            schema_name=schema_name
        )

        # Step 2: Validate
        if graph_output.validation_results:
            if not graph_output.validation_results.is_valid:
                return ProcessingResult(
                    success=False,
                    error="Validation failed",
                    validation_results=graph_output.validation_results
                )

        # Step 3: Weave into Yarn Graph
        nodes_added = 0
        edges_added = 0

        for node in graph_output.nodes:
            # Check if node exists
            existing = self.yarn.get_node(node.id)

            if existing and merge_strategy == "skip":
                continue
            elif existing and merge_strategy == "upsert":
                # Update properties
                self.yarn.update_node(node.id, node.properties)
            else:
                # Add new node
                self.yarn.add_node(
                    node.id,
                    node_type=node.type,
                    **node.properties
                )
                nodes_added += 1

        for edge in graph_output.edges:
            # Add edge with type and properties
            self.yarn.add_edge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.type,
                weight=edge.properties.get('weight', 1.0),
                **edge.properties
            )
            edges_added += 1

        return ProcessingResult(
            success=True,
            nodes_added=nodes_added,
            edges_added=edges_added,
            schema_name=graph_output.schema_name,
            validation_results=graph_output.validation_results
        )
```

---

## Use Cases

### Use Case 1: Expense Tracking

```python
# Register expense schema
await registry.register_schema("expenses", {
    'nodes': {
        'Transaction': {
            'properties': {
                'date': 'date',
                'amount': 'float',
                'category': 'string'
            }
        },
        'Merchant': {
            'properties': {
                'name': 'string',
                'category': 'string'
            }
        },
        'Item': {
            'properties': {
                'name': 'string',
                'price': 'float'
            }
        }
    },
    'edges': {
        'PURCHASED_FROM': {'from': 'Transaction', 'to': 'Merchant'},
        'INCLUDES': {'from': 'Transaction', 'to': 'Item'}
    }
})

# Process receipts with schema
spinner = SchemaAwareReceiptSpinner(registry)
pipeline = WoolToYarnPipeline(registry, yarn_graph)

# Process all receipts
for receipt_path in receipt_paths:
    result = await pipeline.process(
        receipt_path,
        spinner,
        schema_name="expenses"
    )

    print(f"Added {result.nodes_added} nodes, {result.edges_added} edges")

# Query the graph
total_spent = yarn_graph.query("""
    MATCH (t:Transaction)
    RETURN SUM(t.amount) as total
""")

by_category = yarn_graph.query("""
    MATCH (t:Transaction)-[:PURCHASED_FROM]->(m:Merchant)
    RETURN m.category, SUM(t.amount) as total
    GROUP BY m.category
""")
```

### Use Case 2: Project Management

```python
# Register project schema
await registry.register_schema("projects", {
    'nodes': {
        'Note': {
            'properties': {
                'title': 'string',
                'date': 'date',
                'author': 'string'
            }
        },
        'Task': {
            'properties': {
                'description': 'string',
                'status': 'enum[pending,in_progress,done]',
                'priority': 'enum[low,normal,high]',
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
})

# Process handwritten notes
spinner = SchemaAwareHandwrittenSpinner(registry)
pipeline = WoolToYarnPipeline(registry, yarn_graph)

# Process meeting notes
result = await pipeline.process(
    "meeting_notes.jpg",
    spinner,
    schema_name="projects"
)

# Query all pending tasks
pending_tasks = yarn_graph.query("""
    MATCH (t:Task)
    WHERE t.status = 'pending'
    RETURN t.description, t.priority
    ORDER BY t.priority DESC
""")
```

---

## Benefits

### 1. Semantic Structure
✅ Data isn't just text - it's **structured knowledge**
✅ Automatic mapping to your database schema
✅ Type validation and coercion
✅ Relationship inference

### 2. RAG-Powered Intelligence
✅ Auto-detect best schema for unknown data
✅ Suggest field mappings based on similarity
✅ Learn from past transformations
✅ Improve over time

### 3. Direct Graph Integration
✅ Spin data directly into Yarn Graph
✅ No manual transformation needed
✅ Query with Cypher/SPARQL
✅ Temporal and provenance tracking

### 4. Validation & Quality
✅ Schema validation before insertion
✅ Type checking and constraints
✅ Duplicate detection
✅ Data quality metrics

### 5. Extensibility
✅ Register any schema (SQL, JSON, RDF, etc.)
✅ Custom transformation rules
✅ Plugin-based validators
✅ Multi-schema support

---

## Implementation Phases

### Phase 1: Foundation (1 week)
- [ ] Create SchemaRegistry with basic registration
- [ ] Implement GraphOutput data types (Node, Edge)
- [ ] Create SchemaAwareSpinner base class
- [ ] Add validation framework

### Phase 2: RAG Integration (1 week)
- [ ] Integrate RAG memory for schema matching
- [ ] Implement suggestion engine
- [ ] Add field mapping suggestions
- [ ] Create learning pipeline

### Phase 3: Specialized Spinners (1 week)
- [ ] Enhance ReceiptSpinner with schema awareness
- [ ] Enhance HandwrittenSpinner with schema awareness
- [ ] Add generic DocumentSpinner with schema
- [ ] Create FormSpinner with field detection

### Phase 4: Yarn Graph Integration (1 week)
- [ ] Create WoolToYarnPipeline
- [ ] Implement merge strategies (upsert, skip, replace)
- [ ] Add transaction support
- [ ] Create query helpers

### Phase 5: Advanced Features (2 weeks)
- [ ] Add schema versioning
- [ ] Implement schema migration
- [ ] Create schema inference from data
- [ ] Add multi-schema support
- [ ] Build visual schema editor

---

## Example Schemas

### Receipt Schema (SQL-like)

```yaml
name: receipts_db
type: relational
tables:
  merchants:
    columns:
      - {name: id, type: integer, primary_key: true}
      - {name: name, type: string, max_length: 200}
      - {name: address, type: string}
      - {name: category, type: enum, values: [grocery, restaurant, retail]}

  transactions:
    columns:
      - {name: id, type: integer, primary_key: true}
      - {name: merchant_id, type: integer, foreign_key: merchants.id}
      - {name: date, type: date}
      - {name: time, type: time}
      - {name: total, type: decimal, precision: 10, scale: 2}
      - {name: tax, type: decimal}
      - {name: payment_method, type: string}

  line_items:
    columns:
      - {name: id, type: integer, primary_key: true}
      - {name: transaction_id, type: integer, foreign_key: transactions.id}
      - {name: name, type: string}
      - {name: quantity, type: float}
      - {name: price, type: decimal}
```

### Task Management Schema (Graph-like)

```yaml
name: task_graph
type: graph
nodes:
  Note:
    properties:
      title: {type: string, required: true}
      date: {type: date}
      author: {type: string}
      content: {type: text}

  Task:
    properties:
      description: {type: string, required: true}
      status: {type: enum, values: [pending, in_progress, done]}
      priority: {type: enum, values: [low, normal, high]}
      due_date: {type: date}
      estimated_hours: {type: float}

  Person:
    properties:
      name: {type: string, required: true}
      email: {type: string, format: email}
      role: {type: string}

edges:
  HAS_TASK:
    from: Note
    to: Task
    properties: {}

  ASSIGNED_TO:
    from: Task
    to: Person
    properties:
      assigned_date: {type: date}

  MENTIONS:
    from: Note
    to: Person
    properties: {}

  DEPENDS_ON:
    from: Task
    to: Task
    properties:
      dependency_type: {type: enum, values: [blocks, requires]}
```

---

## Technical Considerations

### 1. Schema Flexibility
- Support multiple schema formats (SQL, JSON Schema, RDF, Property Graph)
- Allow schema composition and inheritance
- Version schemas for evolution

### 2. Performance
- Cache schema lookups
- Batch graph insertions
- Lazy validation for large datasets
- Parallel processing for multiple sources

### 3. Error Handling
- Graceful degradation (schema validation fails → store as generic)
- Detailed validation reports
- Suggestion for fixes
- Manual override capability

### 4. Privacy & Security
- Schema-aware data masking
- PII detection and handling
- Audit trail for transformations
- Access control per schema

---

## Next Steps

1. **Prototype SchemaRegistry** with basic RAG integration
2. **Enhance one spinner** (Receipt) with schema awareness
3. **Demo wool→yarn pipeline** with real data
4. **Measure quality improvement** vs. current approach
5. **Iterate based on results**

---

**This is the future of HoloLoom**: Not just extracting text, but **intelligently structuring knowledge** according to your needs.