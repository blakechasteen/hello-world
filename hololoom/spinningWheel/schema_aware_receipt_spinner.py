"""
Schema-Aware Receipt Spinner - Wool → Yarn Transformation
=========================================================

Combines receipt OCR extraction with schema-aware graph transformation.

The KILLER FEATURE of HoloLoom:
Raw receipt image (wool) → Structured knowledge graph (yarn)

Pipeline:
1. OCR extracts text from receipt (ReceiptSpinner)
2. RAG finds matching schema (SchemaRegistry)
3. Validates and maps fields (SchemaRegistry)
4. Transforms to graph nodes/edges (this module)
5. Inserts into Yarn Graph (NetworkX)

Result: Queryable knowledge graph instead of raw text!

Usage:
    from hololoom.spinningWheel import SchemaAwareReceiptSpinner
    from hololoom.memory.graph import KG

    # Create spinner with Yarn Graph
    yarn_graph = KG()
    spinner = SchemaAwareReceiptSpinner(
        yarn_graph=yarn_graph,
        schema_registry=registry
    )

    # Process receipt - automatically creates graph nodes
    result = await spinner.spin("receipt.jpg")

    # Query structured data
    transactions = yarn_graph.subgraph_for_entities(['transaction'])

Author: Claude Code
Date: January 2025
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import hashlib

from hololoom.spinningWheel.receipt_spinner import ReceiptSpinner, ReceiptData
from hololoom.spinningWheel.schema_registry import SchemaRegistry, SchemaDefinition, FieldMapping
from hololoom.spinningWheel.protocol import SpinResult
from hololoom.protocols.types import MemoryShard
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Graph Transformation Results
# ============================================================================

@dataclass
class GraphTransformation:
    """Result of transforming data to graph structure."""
    success: bool

    # Nodes created
    nodes_created: List[Dict[str, Any]] = field(default_factory=list)

    # Edges created
    edges_created: List[KGEdge] = field(default_factory=list)

    # Schema used
    schema_name: Optional[str] = None

    # Field mappings applied
    field_mappings: Dict[str, FieldMapping] = field(default_factory=dict)

    # Validation results
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0

    @property
    def node_count(self) -> int:
        return len(self.nodes_created)

    @property
    def edge_count(self) -> int:
        return len(self.edges_created)

    def summary(self) -> str:
        """Get human-readable summary."""
        parts = [
            f"✓ Transformation {'succeeded' if self.success else 'failed'}",
            f"  Schema: {self.schema_name}",
            f"  Nodes: {self.node_count}",
            f"  Edges: {self.edge_count}",
            f"  Mappings: {len(self.field_mappings)}",
            f"  Time: {self.processing_time_ms:.1f}ms"
        ]

        if self.validation_errors:
            parts.append(f"  Errors: {len(self.validation_errors)}")
        if self.validation_warnings:
            parts.append(f"  Warnings: {len(self.validation_warnings)}")

        return '\n'.join(parts)


# ============================================================================
# Schema-Aware Receipt Spinner
# ============================================================================

class SchemaAwareReceiptSpinner(ReceiptSpinner):
    """
    Receipt spinner with automatic graph transformation.

    Extends ReceiptSpinner to automatically transform extracted receipt
    data into structured Yarn Graph nodes and edges.

    Philosophy:
        "The spinner extracts facts.
         The schema organizes facts.
         The graph remembers facts."

    This is the complete wool → yarn pipeline.
    """

    def __init__(
        self,
        yarn_graph: Optional[KG] = None,
        schema_registry: Optional[SchemaRegistry] = None,
        importance_threshold: float = 0.3,
        verify_calculations: bool = True,
        categorize: bool = True,
        enable_enrichment: bool = False,
        auto_create_graph: bool = True,
        default_schema: str = "expenses"
    ):
        """
        Initialize schema-aware receipt spinner.

        Args:
            yarn_graph: Yarn Graph instance to populate (None = create new)
            schema_registry: Schema registry for schema matching
            importance_threshold: Min importance for shard creation
            verify_calculations: Verify receipt math
            categorize: Categorize merchant/items
            enable_enrichment: Use Ollama enrichment
            auto_create_graph: Automatically create graph nodes/edges
            default_schema: Default schema name if auto-detection fails
        """
        super().__init__(
            importance_threshold=importance_threshold,
            verify_calculations=verify_calculations,
            categorize=categorize,
            enable_enrichment=enable_enrichment
        )

        self.yarn_graph = yarn_graph or KG()
        self.schema_registry = schema_registry
        self.auto_create_graph = auto_create_graph
        self.default_schema = default_schema

        # Track transformations
        self.transformations: List[GraphTransformation] = []

    # ========================================================================
    # Main Processing (Override)
    # ========================================================================

    async def spin(
        self,
        source: Union[str, Path, List],
        **kwargs
    ) -> SpinResult:
        """
        Process receipt and create graph nodes/edges.

        Extends base ReceiptSpinner.spin() to automatically transform
        extracted data into Yarn Graph structure.

        Args:
            source: Receipt image path(s)
            **kwargs: Additional arguments

        Returns:
            SpinResult with shards + graph transformation metadata
        """
        # First, run base receipt extraction
        result = await super().spin(source, **kwargs)

        if not result.success:
            return result

        # Transform each shard's receipt data to graph
        if self.auto_create_graph:
            for shard in result.shards:
                receipt_data = shard.metadata.get('receipt_data')

                if receipt_data:
                    transformation = await self._transform_to_graph(
                        receipt_data,
                        shard
                    )

                    # Store transformation result in shard metadata
                    shard.metadata['graph_transformation'] = {
                        'success': transformation.success,
                        'nodes_created': transformation.node_count,
                        'edges_created': transformation.edge_count,
                        'schema_name': transformation.schema_name,
                        'errors': transformation.validation_errors,
                        'warnings': transformation.validation_warnings
                    }

                    self.transformations.append(transformation)

        return result

    # ========================================================================
    # Graph Transformation
    # ========================================================================

    async def _transform_to_graph(
        self,
        receipt_data: ReceiptData,
        shard: MemoryShard
    ) -> GraphTransformation:
        """
        Transform receipt data to graph nodes and edges.

        This is the core wool → yarn transformation:
        1. Find/select schema
        2. Map fields to schema
        3. Create graph nodes (Transaction, Merchant, Items)
        4. Create graph edges (PURCHASED_FROM, INCLUDES)
        5. Insert into Yarn Graph

        Args:
            receipt_data: Extracted receipt data
            shard: Memory shard with context

        Returns:
            GraphTransformation result
        """
        import time
        start_time = time.time()

        transformation = GraphTransformation(success=False)

        try:
            # Step 1: Find matching schema
            schema_name = await self._find_schema(receipt_data)
            transformation.schema_name = schema_name

            if not schema_name:
                transformation.validation_errors.append(
                    "No matching schema found"
                )
                return transformation

            # Step 2: Get field mappings
            extracted_data = self._receipt_to_dict(receipt_data)

            if self.schema_registry:
                mappings = await self.schema_registry.suggest_field_mapping(
                    schema_name,
                    extracted_data
                )
                transformation.field_mappings = mappings

            # Step 3: Create graph nodes
            nodes, edges = self._create_graph_structure(
                receipt_data,
                schema_name,
                shard
            )

            transformation.nodes_created = nodes
            transformation.edges_created = edges

            # Step 4: Insert into Yarn Graph
            self._insert_into_graph(nodes, edges)

            transformation.success = True

        except Exception as e:
            transformation.validation_errors.append(f"Transformation failed: {e}")

        finally:
            transformation.processing_time_ms = (time.time() - start_time) * 1000

        return transformation

    async def _find_schema(self, receipt_data: ReceiptData) -> Optional[str]:
        """Find best matching schema for receipt data."""
        if not self.schema_registry:
            # No registry - use default
            return self.default_schema

        # Create sample text for RAG matching
        sample_text = f"""
        Merchant: {receipt_data.merchant}
        Total: ${receipt_data.total}
        Items: {receipt_data.item_count}
        Category: {receipt_data.category}
        """

        # Use RAG to find schema
        schema_name = await self.schema_registry.find_best_schema(sample_text)

        return schema_name or self.default_schema

    def _receipt_to_dict(self, receipt_data: ReceiptData) -> Dict[str, Any]:
        """Convert ReceiptData to flat dict for field mapping."""
        return {
            'merchant_name': receipt_data.merchant,
            'merchant_address': receipt_data.merchant_address,
            'transaction_date': receipt_data.date,
            'transaction_time': receipt_data.time,
            'transaction_total': receipt_data.total,
            'transaction_subtotal': receipt_data.subtotal,
            'transaction_tax': receipt_data.tax,
            'transaction_category': receipt_data.category,
            'payment_method': receipt_data.payment_method,
            'items': receipt_data.items
        }

    def _create_graph_structure(
        self,
        receipt_data: ReceiptData,
        schema_name: str,
        shard: MemoryShard
    ) -> tuple[List[Dict], List[KGEdge]]:
        """
        Create graph nodes and edges from receipt data.

        Graph structure:
        - Transaction node (main entity)
        - Merchant node (where purchased)
        - Item nodes (what was purchased)
        - PURCHASED_FROM edge (Transaction → Merchant)
        - INCLUDES edges (Transaction → Items)

        Args:
            receipt_data: Extracted receipt data
            schema_name: Schema being used
            shard: Source memory shard

        Returns:
            (nodes, edges) tuple
        """
        nodes = []
        edges = []

        # Create stable IDs using hash
        def create_id(entity_type: str, *parts) -> str:
            """Create stable ID from entity type and components."""
            content = f"{entity_type}:{'|'.join(str(p) for p in parts)}"
            hash_digest = hashlib.sha256(content.encode()).hexdigest()[:16]
            return f"{entity_type}_{hash_digest}"

        # ====================================================================
        # Create Merchant Node
        # ====================================================================

        merchant_id = create_id(
            'merchant',
            receipt_data.merchant,
            receipt_data.merchant_address
        )

        merchant_node = {
            'id': merchant_id,
            'type': 'Merchant',
            'properties': {
                'name': receipt_data.merchant,
                'address': receipt_data.merchant_address,
                'category': receipt_data.category
            },
            'metadata': {
                'schema': schema_name,
                'source_shard': shard.id
            }
        }

        nodes.append(merchant_node)

        # ====================================================================
        # Create Transaction Node
        # ====================================================================

        transaction_id = create_id(
            'transaction',
            receipt_data.date,
            receipt_data.time,
            receipt_data.merchant,
            receipt_data.total
        )

        transaction_node = {
            'id': transaction_id,
            'type': 'Transaction',
            'properties': {
                'date': receipt_data.date,
                'time': receipt_data.time,
                'total': float(receipt_data.total) if receipt_data.total else 0.0,
                'subtotal': float(receipt_data.subtotal) if receipt_data.subtotal else 0.0,
                'tax': float(receipt_data.tax) if receipt_data.tax else 0.0,
                'category': receipt_data.category,
                'payment_method': receipt_data.payment_method
            },
            'metadata': {
                'schema': schema_name,
                'source_shard': shard.id,
                'calculation_verified': shard.metadata.get('calculation_verified', False)
            }
        }

        nodes.append(transaction_node)

        # ====================================================================
        # Create Edge: Transaction → Merchant
        # ====================================================================

        purchased_from_edge = KGEdge(
            src=transaction_id,
            dst=merchant_id,
            type='PURCHASED_FROM',
            weight=1.0,
            span_id=shard.id,
            metadata={
                'schema': schema_name,
                'date': receipt_data.date
            }
        )

        edges.append(purchased_from_edge)

        # ====================================================================
        # Create Item Nodes and Edges
        # ====================================================================

        for i, item in enumerate(receipt_data.items):
            item_id = create_id(
                'item',
                transaction_id,
                item.get('name', f'item_{i}')
            )

            item_node = {
                'id': item_id,
                'type': 'Item',
                'properties': {
                    'name': item.get('name', ''),
                    'quantity': item.get('quantity', 1),
                    'unit_price': float(item.get('unit_price', 0)) if item.get('unit_price') else 0.0,
                    'total_price': float(item.get('total_price', 0)) if item.get('total_price') else 0.0
                },
                'metadata': {
                    'schema': schema_name,
                    'transaction_id': transaction_id,
                    'source_shard': shard.id
                }
            }

            nodes.append(item_node)

            # Edge: Transaction → Item
            includes_edge = KGEdge(
                src=transaction_id,
                dst=item_id,
                type='INCLUDES',
                weight=1.0,
                span_id=shard.id,
                metadata={
                    'quantity': item.get('quantity', 1),
                    'schema': schema_name
                }
            )

            edges.append(includes_edge)

        return nodes, edges

    def _insert_into_graph(
        self,
        nodes: List[Dict],
        edges: List[KGEdge]
    ) -> None:
        """
        Insert nodes and edges into Yarn Graph.

        Args:
            nodes: Node definitions
            edges: Edge definitions
        """
        # Add nodes to graph
        for node in nodes:
            node_id = node['id']
            node_type = node['type']

            # Add node with properties as attributes
            self.yarn_graph.G.add_node(
                node_id,
                type=node_type,
                **node['properties'],
                **node.get('metadata', {})
            )

        # Add edges to graph
        for edge in edges:
            self.yarn_graph.add_edge(edge)

    # ========================================================================
    # Query Interface
    # ========================================================================

    def get_transformations(self) -> List[GraphTransformation]:
        """Get all graph transformations created by this spinner."""
        return self.transformations.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get transformation statistics.

        Returns:
            Dict with aggregated stats
        """
        total_nodes = sum(t.node_count for t in self.transformations)
        total_edges = sum(t.edge_count for t in self.transformations)
        successes = sum(1 for t in self.transformations if t.success)
        failures = len(self.transformations) - successes

        avg_time = (
            sum(t.processing_time_ms for t in self.transformations)
            / len(self.transformations)
        ) if self.transformations else 0.0

        return {
            'total_transformations': len(self.transformations),
            'successful': successes,
            'failed': failures,
            'total_nodes_created': total_nodes,
            'total_edges_created': total_edges,
            'avg_processing_time_ms': avg_time,
            'schemas_used': list(set(
                t.schema_name for t in self.transformations
                if t.schema_name
            ))
        }

    def query_transactions(
        self,
        merchant: Optional[str] = None,
        category: Optional[str] = None,
        min_total: Optional[float] = None,
        max_total: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query transactions from Yarn Graph.

        Args:
            merchant: Filter by merchant name
            category: Filter by category
            min_total: Minimum transaction total
            max_total: Maximum transaction total

        Returns:
            List of matching transaction nodes
        """
        results = []

        for node_id, node_data in self.yarn_graph.G.nodes(data=True):
            if node_data.get('type') != 'Transaction':
                continue

            # Apply filters
            if merchant:
                # Find merchant via PURCHASED_FROM edge
                neighbors = list(self.yarn_graph.G.successors(node_id))
                merchant_nodes = [
                    n for n in neighbors
                    if self.yarn_graph.G.nodes[n].get('type') == 'Merchant'
                ]

                if not any(
                    self.yarn_graph.G.nodes[m].get('name', '').lower() == merchant.lower()
                    for m in merchant_nodes
                ):
                    continue

            if category and node_data.get('category') != category:
                continue

            total = node_data.get('total', 0)
            if min_total is not None and total < min_total:
                continue
            if max_total is not None and total > max_total:
                continue

            # Include transaction
            results.append({
                'id': node_id,
                'date': node_data.get('date'),
                'time': node_data.get('time'),
                'total': total,
                'category': node_data.get('category'),
                'payment_method': node_data.get('payment_method')
            })

        return results

    def query_merchants(self) -> List[Dict[str, Any]]:
        """Get all merchants from Yarn Graph."""
        results = []

        for node_id, node_data in self.yarn_graph.G.nodes(data=True):
            if node_data.get('type') == 'Merchant':
                results.append({
                    'id': node_id,
                    'name': node_data.get('name'),
                    'address': node_data.get('address'),
                    'category': node_data.get('category')
                })

        return results

    def get_transaction_details(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full details for a transaction including merchant and items.

        Args:
            transaction_id: Transaction node ID

        Returns:
            Dict with transaction, merchant, and items
        """
        if transaction_id not in self.yarn_graph.G:
            return None

        transaction_data = self.yarn_graph.G.nodes[transaction_id]

        # Find merchant
        merchant = None
        items = []

        for successor in self.yarn_graph.G.successors(transaction_id):
            node_data = self.yarn_graph.G.nodes[successor]
            node_type = node_data.get('type')

            if node_type == 'Merchant':
                merchant = {
                    'name': node_data.get('name'),
                    'address': node_data.get('address'),
                    'category': node_data.get('category')
                }
            elif node_type == 'Item':
                items.append({
                    'name': node_data.get('name'),
                    'quantity': node_data.get('quantity'),
                    'unit_price': node_data.get('unit_price'),
                    'total_price': node_data.get('total_price')
                })

        return {
            'transaction': {
                'id': transaction_id,
                'date': transaction_data.get('date'),
                'time': transaction_data.get('time'),
                'total': transaction_data.get('total'),
                'subtotal': transaction_data.get('subtotal'),
                'tax': transaction_data.get('tax'),
                'category': transaction_data.get('category'),
                'payment_method': transaction_data.get('payment_method')
            },
            'merchant': merchant,
            'items': items
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def process_receipt_to_graph(
    receipt_path: Union[str, Path],
    yarn_graph: Optional[KG] = None,
    schema_registry: Optional[SchemaRegistry] = None,
    **spinner_kwargs
) -> tuple[SpinResult, GraphTransformation]:
    """
    Convenience function: process receipt and create graph in one call.

    Args:
        receipt_path: Path to receipt image
        yarn_graph: Yarn Graph instance (None = create new)
        schema_registry: Schema registry (None = use default schema)
        **spinner_kwargs: Additional spinner arguments

    Returns:
        (SpinResult, GraphTransformation) tuple
    """
    spinner = SchemaAwareReceiptSpinner(
        yarn_graph=yarn_graph,
        schema_registry=schema_registry,
        **spinner_kwargs
    )

    result = await spinner.spin(receipt_path)

    # Get last transformation
    transformation = spinner.transformations[-1] if spinner.transformations else None

    return result, transformation
