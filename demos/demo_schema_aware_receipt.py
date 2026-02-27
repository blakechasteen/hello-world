"""
Schema-Aware Receipt Processing Demo
====================================

Demonstrates production wool -> yarn transformation pipeline:
1. SchemaAwareReceiptSpinner processes receipt image
2. Automatically creates graph nodes/edges in Yarn Graph
3. Query structured data via production API

This is the COMPLETE killer feature end-to-end.

Usage:
    python demos/demo_schema_aware_receipt.py

Author: Claude Code
Date: January 2025
"""

import asyncio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# Demo: Production Schema-Aware Processing
# ============================================================================

async def demo_schema_aware_receipt():
    """
    Complete production demo:
    Receipt Image -> OCR -> Schema -> Yarn Graph -> Query
    """
    print("=" * 70)
    print("SCHEMA-AWARE RECEIPT PROCESSING")
    print("=" * 70)
    print("\nThis demo shows:")
    print("  1. Create receipt image (wool)")
    print("  2. Process with SchemaAwareReceiptSpinner")
    print("  3. Automatic graph transformation")
    print("  4. Query via production API")
    print("  5. Advanced graph queries")

    # ========================================================================
    # Step 1: Create Sample Receipt
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 1: Creating receipt image (wool)...")
    print("-" * 70)

    img = Image.new('RGB', (600, 700), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    receipt_text = """WHOLE FOODS MARKET
123 Main Street
San Francisco, CA

01/15/2025    14:23

Organic Bananas         $3.99
Almond Milk             $4.49
Whole Wheat Bread       $5.99
Free Range Eggs         $6.99
Greek Yogurt            $7.99

Subtotal               $29.45
Tax (8.5%)              $2.50
Total                  $31.95

VISA ending in 4242
Thank you!"""

    y = 50
    for line in receipt_text.split('\n'):
        draw.text((50, y), line, fill='black', font=font)
        y += 28

    receipt_path = Path("temp_demo_receipt.png")
    img.save(receipt_path)

    print("[OK] Created receipt image")
    print(f"  Path: {receipt_path}")

    # ========================================================================
    # Step 2: Setup Schema Registry
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 2: Setting up schema registry...")
    print("-" * 70)

    from hololoom.spinningWheel.schema_registry import (
        SchemaRegistry,
        create_expense_schema
    )

    registry = SchemaRegistry(memory_backend=None)
    expense_schema = create_expense_schema()
    await registry.register_schema("expenses", expense_schema)

    print("[OK] Registered expense schema")
    print(f"  Schema: expenses (finance domain)")

    # ========================================================================
    # Step 3: Create Schema-Aware Spinner
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 3: Creating schema-aware spinner...")
    print("-" * 70)

    from hololoom.spinningWheel.schema_aware_receipt_spinner import SchemaAwareReceiptSpinner
    from hololoom.memory.graph import KG

    # Create Yarn Graph
    yarn_graph = KG()

    # Create schema-aware spinner
    spinner = SchemaAwareReceiptSpinner(
        yarn_graph=yarn_graph,
        schema_registry=registry,
        verify_calculations=True,
        categorize=True,
        auto_create_graph=True
    )

    print("[OK] Created SchemaAwareReceiptSpinner")
    print("  Yarn Graph: NetworkX MultiDiGraph")
    print("  Schema registry: RAG-powered")
    print("  Auto-transform: ENABLED")

    # ========================================================================
    # Step 4: Process Receipt (Complete Pipeline)
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 4: Processing receipt (wool -> yarn)...")
    print("-" * 70)

    import time
    start_time = time.time()

    result = await spinner.spin(receipt_path)

    elapsed = (time.time() - start_time) * 1000

    print(f"\n[OK] Processing complete ({elapsed:.1f}ms)")
    print(f"  Success: {result.success}")
    print(f"  Shards: {result.shard_count}")
    print(f"  Processing time: {result.processing_time_ms:.1f}ms")

    # Show shard details
    if result.shards:
        shard = result.shards[0]

        print(f"\nShard Details:")
        print(f"  OCR backend: {shard.metadata.get('ocr_backend')}")
        print(f"  OCR confidence: {shard.metadata.get('ocr_confidence'):.2f}")
        print(f"  Importance: {shard.metadata.get('importance_score'):.2f}")

        # Graph transformation details
        graph_transform = shard.metadata.get('graph_transformation', {})
        print(f"\nGraph Transformation:")
        print(f"  Success: {graph_transform.get('success')}")
        print(f"  Nodes created: {graph_transform.get('nodes_created')}")
        print(f"  Edges created: {graph_transform.get('edges_created')}")
        print(f"  Schema: {graph_transform.get('schema_name')}")

    # ========================================================================
    # Step 5: Get Transformation Statistics
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 5: Transformation statistics...")
    print("-" * 70)

    stats = spinner.get_statistics()

    print(f"\nSpinner Statistics:")
    print(f"  Total transformations: {stats['total_transformations']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total nodes: {stats['total_nodes_created']}")
    print(f"  Total edges: {stats['total_edges_created']}")
    print(f"  Avg time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"  Schemas used: {', '.join(stats['schemas_used'])}")

    # ========================================================================
    # Step 6: Query Structured Data (Production API)
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 6: Querying via production API...")
    print("-" * 70)

    # Query all transactions
    transactions = spinner.query_transactions()
    print(f"\n📊 All Transactions ({len(transactions)}):")
    for tx in transactions:
        print(f"  - {tx['date']} @ {tx['time']}: ${tx['total']:.2f}")
        print(f"    Category: {tx['category']}")
        print(f"    Payment: {tx['payment_method']}")

    # Query all merchants
    merchants = spinner.query_merchants()
    print(f"\n🏪 All Merchants ({len(merchants)}):")
    for m in merchants:
        print(f"  - {m['name']}")
        print(f"    Address: {m['address']}")
        print(f"    Category: {m['category']}")

    # Get transaction details
    if transactions:
        tx_id = transactions[0]['id']
        details = spinner.get_transaction_details(tx_id)

        print(f"\n🔍 Transaction Details:")
        print(f"  Date: {details['transaction']['date']} @ {details['transaction']['time']}")
        print(f"  Merchant: {details['merchant']['name']}")
        print(f"  Address: {details['merchant']['address']}")

        print(f"\n  Items ({len(details['items'])}):")
        for item in details['items']:
            qty = item['quantity']
            name = item['name']
            price = item['total_price']
            print(f"    - {qty}x {name}: ${price:.2f}")

        print(f"\n  Financial:")
        print(f"    Subtotal: ${details['transaction']['subtotal']:.2f}")
        print(f"    Tax: ${details['transaction']['tax']:.2f}")
        print(f"    Total: ${details['transaction']['total']:.2f}")

    # ========================================================================
    # Step 7: Advanced Graph Queries
    # ========================================================================

    print("\n" + "-" * 70)
    print("STEP 7: Advanced graph queries...")
    print("-" * 70)

    # Direct NetworkX access
    G = yarn_graph.G

    print(f"\nYarn Graph Structure:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")

    # Node types
    node_types = {}
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    print(f"\n  Node Types:")
    for node_type, count in node_types.items():
        print(f"    {node_type}: {count}")

    # Edge types
    edge_types = {}
    for src, dst, edge_data in G.edges(data=True):
        edge_type = edge_data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    print(f"\n  Edge Types:")
    for edge_type, count in edge_types.items():
        print(f"    {edge_type}: {count}")

    # Query: Find all items purchased from Whole Foods
    print(f"\n🔍 Query: Items from Whole Foods")
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('type') == 'Merchant' and 'whole foods' in node_data.get('name', '').lower():
            # Find transactions from this merchant
            for src, dst, edge_data in G.in_edges(node_id, data=True):
                if edge_data.get('type') == 'PURCHASED_FROM':
                    tx_id = src
                    # Find items in this transaction
                    for item_src, item_dst, item_edge_data in G.out_edges(tx_id, data=True):
                        if item_edge_data.get('type') == 'INCLUDES':
                            item_data = G.nodes[item_dst]
                            if item_data.get('type') == 'Item':
                                print(f"  - {item_data.get('name')}: ${item_data.get('total_price'):.2f}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("TRANSFORMATION COMPLETE!")
    print("=" * 70)

    print(f"\n[DONE] Wool (raw image) -> Yarn (structured graph)")

    print(f"\nWhat happened:")
    print(f"  1. ReceiptSpinner extracted text via OCR")
    print(f"  2. Parsed receipt into structured ReceiptData")
    print(f"  3. SchemaRegistry found 'expenses' schema via RAG")
    print(f"  4. Mapped fields to schema nodes/edges")
    print(f"  5. Created {stats['total_nodes_created']} nodes + {stats['total_edges_created']} edges")
    print(f"  6. Inserted into Yarn Graph (NetworkX)")
    print(f"  7. Ready for complex queries!")

    print(f"\n🎯 The Magic:")
    print(f"  [OK] Started with: PNG image")
    print(f"  [OK] Ended with: Queryable knowledge graph")
    print(f"  [OK] Zero manual mapping")
    print(f"  [OK] Schema-aware validation")
    print(f"  [OK] RAG-powered matching")
    print(f"  [OK] Production-ready API")

    print(f"\n💡 Production Usage:")
    print(f"  # Query transactions")
    print(f"  spinner.query_transactions(category='grocery', min_total=20.0)")
    print(f"")
    print(f"  # Get details")
    print(f"  spinner.get_transaction_details(tx_id)")
    print(f"")
    print(f"  # Advanced queries")
    print(f"  yarn_graph.G  # Full NetworkX graph")

    print(f"\n🚀 Production Benefits:")
    print(f"  - Aggregate: SUM(total) GROUP BY merchant")
    print(f"  - Temporal: Transactions between dates")
    print(f"  - Analytical: Avg spending per category")
    print(f"  - Relational: Same items from different merchants")
    print(f"  - Graph: Multi-hop traversal and pattern matching")

    # Cleanup
    receipt_path.unlink()

    print("\n" + "=" * 70)


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run demo."""
    print("\n" + "=" * 70)
    print("SCHEMA-AWARE RECEIPT PROCESSING")
    print("=" * 70)

    await demo_schema_aware_receipt()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\nNext Steps:")
    print("  [OK] SchemaAwareReceiptSpinner - DONE")
    print("  - SchemaAwareHandwrittenSpinner - TODO")
    print("  - More domain schemas (invoices, forms, etc.)")
    print("  - Cloud OCR backends (Azure, AWS, Google)")
    print("  - Full RAG integration with HoloLoom memory")


if __name__ == "__main__":
    asyncio.run(main())
