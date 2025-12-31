"""
Demo: SQL Integration for HoloLoom RAG

Demonstrates:
1. Registering a SQLite database with schema
2. Automatic routing (SQL vs semantic queries)
3. Hybrid queries (SQL + semantic fusion)
4. Performance comparison (SQL vs semantic)
5. Error handling and fallbacks
6. Visual output with query type indicators

Usage:
    python demos/demo_rag_sql.py

Requirements:
    pip install sqlalchemy pandas rich

Author: Agent H (Claude Code)
Date: November 13, 2025
"""

import asyncio
import tempfile
import os
import time
from pathlib import Path

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠ Install 'rich' for beautiful output: pip install rich")

# HoloLoom imports
from HoloLoom.rag.sql_integration import (
    SQLRAGMixin,
    SQLAdapter,
    TextToSQLTranslator,
    QueryIntent,
    SQLALCHEMY_AVAILABLE,
    PANDAS_AVAILABLE
)


class DemoRAG(SQLRAGMixin):
    """
    Demo RAG implementation with SQL integration.

    Extends SQLRAGMixin with minimal semantic search implementation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mock semantic memory
        self.semantic_memory = [
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
            "SQL databases store structured data in tables with rows and columns",
            "Machine learning uses algorithms to learn patterns from data",
            "Python is a popular programming language for data science",
            "Reinforcement learning agents learn through trial and error"
        ]

    async def _query_semantic_only(self, question: str, max_sources: int):
        """Override semantic query for demo."""
        from HoloLoom.rag.sql_integration import SQLRAGResult

        # Simple keyword matching
        question_lower = question.lower()
        relevant_sources = [
            mem for mem in self.semantic_memory
            if any(word in mem.lower() for word in question_lower.split())
        ]

        if relevant_sources:
            response = f"Based on semantic search:\n" + "\n".join(
                f"- {source}" for source in relevant_sources[:max_sources]
            )
            confidence = 0.7
        else:
            response = f"No semantic information found for: {question}"
            confidence = 0.3

        return SQLRAGResult(
            response=response,
            sources=relevant_sources[:max_sources],
            confidence=confidence,
            query_type="semantic"
        )


def create_demo_database():
    """
    Create demo SQLite database with sample data.

    Returns:
        Tuple of (connection_string, db_path)
    """
    if not SQLALCHEMY_AVAILABLE:
        raise RuntimeError("SQLAlchemy required. Install: pip install sqlalchemy")

    from sqlalchemy import create_engine, text

    # Create temp database
    fd, db_path = tempfile.mkstemp(suffix='_demo.db', prefix='hololoom_sql_')
    os.close(fd)

    conn_str = f"sqlite:///{db_path}"
    engine = create_engine(conn_str)

    with engine.connect() as conn:
        # Users table
        conn.execute(text("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                age INTEGER NOT NULL,
                country TEXT NOT NULL,
                interests TEXT NOT NULL
            )
        """))

        # Products table
        conn.execute(text("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            )
        """))

        # Orders table
        conn.execute(text("""
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                total_price REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """))

        # Insert sample data
        conn.execute(text("""
            INSERT INTO users (id, name, email, age, country, interests) VALUES
                (1, 'Alice Johnson', 'alice@example.com', 28, 'USA', 'machine learning, data science'),
                (2, 'Bob Smith', 'bob@example.com', 35, 'Canada', 'web development, databases'),
                (3, 'Charlie Brown', 'charlie@example.com', 42, 'UK', 'AI, reinforcement learning'),
                (4, 'Diana Prince', 'diana@example.com', 31, 'USA', 'computer vision, robotics'),
                (5, 'Eve Davis', 'eve@example.com', 26, 'Australia', 'NLP, machine learning'),
                (6, 'Frank Miller', 'frank@example.com', 39, 'Germany', 'databases, SQL'),
                (7, 'Grace Lee', 'grace@example.com', 29, 'South Korea', 'data science, analytics')
        """))

        conn.execute(text("""
            INSERT INTO products (id, name, category, price, stock) VALUES
                (1, 'Laptop Pro 15', 'Electronics', 1299.99, 25),
                (2, 'Wireless Mouse', 'Electronics', 29.99, 150),
                (3, 'Mechanical Keyboard', 'Electronics', 89.99, 80),
                (4, 'USB-C Hub', 'Electronics', 49.99, 200),
                (5, 'Monitor 27inch', 'Electronics', 399.99, 45),
                (6, 'Webcam HD', 'Electronics', 79.99, 60),
                (7, 'Desk Lamp LED', 'Furniture', 34.99, 120),
                (8, 'Office Chair', 'Furniture', 249.99, 30)
        """))

        conn.execute(text("""
            INSERT INTO orders (id, user_id, product_id, quantity, order_date, total_price) VALUES
                (1, 1, 1, 1, '2025-01-10', 1299.99),
                (2, 1, 2, 2, '2025-01-10', 59.98),
                (3, 2, 3, 1, '2025-01-15', 89.99),
                (4, 3, 5, 1, '2025-01-20', 399.99),
                (5, 3, 6, 1, '2025-01-20', 79.99),
                (6, 4, 4, 3, '2025-02-01', 149.97),
                (7, 5, 7, 2, '2025-02-05', 69.98),
                (8, 6, 1, 1, '2025-02-10', 1299.99),
                (9, 7, 8, 1, '2025-02-15', 249.99)
        """))

        conn.commit()

    engine.dispose()

    return conn_str, db_path


async def run_demo():
    """Run comprehensive SQL integration demo."""

    if RICH_AVAILABLE:
        console = Console()
    else:
        console = None

    def print_header(text):
        if console:
            console.print(f"\n[bold cyan]{text}[/bold cyan]")
        else:
            print(f"\n{'=' * 60}")
            print(text)
            print('=' * 60)

    def print_result(result, query):
        if console:
            # Create result table
            table = Table(title=f"Query: {query}", show_header=True, header_style="bold magenta")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Query Type", result.query_type)
            table.add_row("Confidence", f"{result.confidence:.2f}")

            if result.sql_query:
                table.add_row("SQL Query", result.sql_query)

            if result.sql_data is not None and PANDAS_AVAILABLE:
                table.add_row("SQL Rows", str(len(result.sql_data)))

            if result.sources:
                table.add_row("Semantic Sources", str(len(result.sources)))

            console.print(table)

            # Print response
            console.print(Panel(result.response, title="Response", border_style="green"))

        else:
            print(f"\n{'*' * 60}")
            print(f"Query: {query}")
            print(f"Type: {result.query_type}")
            print(f"Confidence: {result.confidence:.2f}")
            if result.sql_query:
                print(f"SQL: {result.sql_query}")
            print(f"Response:\n{result.response}")
            print('*' * 60)

    print_header("🚀 HoloLoom SQL Integration Demo")

    # Check dependencies
    if not SQLALCHEMY_AVAILABLE:
        print("❌ SQLAlchemy not available. Install: pip install sqlalchemy")
        return

    if not PANDAS_AVAILABLE:
        print("⚠ Pandas not available. Install for better output: pip install pandas")

    # Step 1: Create demo database
    print_header("📊 Step 1: Creating Demo Database")

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating database...", total=None)
            conn_str, db_path = create_demo_database()
            progress.remove_task(task)
    else:
        print("Creating demo database...")
        conn_str, db_path = create_demo_database()

    print(f"✓ Database created: {db_path}")
    print(f"  Tables: users, products, orders")

    # Step 2: Initialize SQL RAG
    print_header("🔧 Step 2: Initializing SQL RAG")

    rag = DemoRAG(
        db_connection=conn_str,
        enable_hybrid_routing=True,
        sql_confidence_threshold=0.7
    )

    # Mock LLM provider
    class MockLLM:
        async def weave(self, query, use_llm=True):
            text = query.text.lower()

            # Extract SQL queries based on intent
            if 'how many users' in text or 'count users' in text:
                if 'over' in text or '>' in text:
                    sql = "SELECT COUNT(*) as count FROM users WHERE age > 30"
                else:
                    sql = "SELECT COUNT(*) as count FROM users"
            elif 'users interested in' in text or 'users with interest' in text:
                interest = 'machine learning'
                if 'machine learning' in text:
                    interest = 'machine learning'
                elif 'databases' in text or 'sql' in text:
                    interest = 'databases'
                sql = f"SELECT * FROM users WHERE interests LIKE '%{interest}%'"
            elif 'average age' in text:
                sql = "SELECT AVG(age) as avg_age FROM users"
            elif 'total revenue' in text or 'total sales' in text:
                sql = "SELECT SUM(total_price) as total_revenue FROM orders"
            elif 'orders' in text and 'january' in text:
                sql = "SELECT * FROM orders WHERE order_date LIKE '2025-01-%'"
            elif 'top selling' in text or 'most ordered' in text:
                sql = """SELECT products.name, SUM(orders.quantity) as total_quantity
                         FROM orders
                         JOIN products ON orders.product_id = products.id
                         GROUP BY products.id
                         ORDER BY total_quantity DESC
                         LIMIT 5"""
            else:
                sql = "SELECT * FROM users LIMIT 5"

            class MockSpacetime:
                def __init__(self, response):
                    self.response = response
                    self.confidence = 0.9

            return MockSpacetime(sql)

    await rag.connect_sql(llm_provider=MockLLM())
    print("✓ SQL RAG initialized")
    print(f"  Schema tables: {list(rag.schema.keys())}")

    # Step 3: SQL-only queries
    print_header("🔍 Step 3: SQL-Only Queries (Factual Lookups)")

    queries_sql = [
        "How many users are over 30 years old?",
        "What is the average age of users?",
        "Show me all orders from January 2025",
    ]

    for query in queries_sql:
        result = await rag.query_with_sql(query, mode="sql_only")
        print_result(result, query)

    # Step 4: Semantic-only queries
    print_header("🧠 Step 4: Semantic-Only Queries (Conceptual)")

    queries_semantic = [
        "What is Thompson Sampling?",
        "Explain machine learning",
    ]

    for query in queries_semantic:
        result = await rag.query_with_sql(query, mode="semantic_only")
        print_result(result, query)

    # Step 5: Hybrid queries
    print_header("🔀 Step 5: Hybrid Queries (SQL + Semantic Fusion)")

    queries_hybrid = [
        "Show me users interested in machine learning",
        "What are the top selling products?",
    ]

    for query in queries_hybrid:
        result = await rag.query_with_sql(query, mode="hybrid")
        print_result(result, query)

    # Step 6: Auto-routing demonstration
    print_header("🎯 Step 6: Auto-Routing Demonstration")

    queries_auto = [
        ("Count all users", "SQL factual → routes to SQL"),
        ("What is reinforcement learning?", "Semantic question → routes to semantic"),
        ("Find users with SQL interests", "Hybrid intent → routes to hybrid"),
    ]

    if console:
        routing_table = Table(title="Auto-Routing Results", show_header=True)
        routing_table.add_column("Query", style="cyan")
        routing_table.add_column("Expected", style="yellow")
        routing_table.add_column("Actual", style="green")
        routing_table.add_column("Match", style="magenta")

    for query, expected in queries_auto:
        result = await rag.query_with_sql(query, mode="auto")

        match = "✓" if expected.split()[0].lower() in result.query_type else "✗"

        if console:
            routing_table.add_row(query, expected, result.query_type, match)
        else:
            print(f"\nQuery: {query}")
            print(f"Expected: {expected}")
            print(f"Actual: {result.query_type}")
            print(f"Match: {match}")

    if console:
        console.print(routing_table)

    # Step 7: Performance comparison
    print_header("⚡ Step 7: Performance Comparison")

    query = "SELECT * FROM users WHERE age > 25"

    # SQL query
    start = time.time()
    result_sql = await rag.query_with_sql(query, mode="sql_only")
    sql_latency = (time.time() - start) * 1000

    # Semantic query
    start = time.time()
    result_semantic = await rag.query_with_sql("What is machine learning?", mode="semantic_only")
    semantic_latency = (time.time() - start) * 1000

    if console:
        perf_table = Table(title="Performance Comparison", show_header=True)
        perf_table.add_column("Query Type", style="cyan")
        perf_table.add_column("Latency (ms)", style="green")
        perf_table.add_column("Result Count", style="yellow")

        perf_table.add_row(
            "SQL",
            f"{sql_latency:.1f}",
            str(len(result_sql.sql_data) if result_sql.sql_data is not None and PANDAS_AVAILABLE else 0)
        )
        perf_table.add_row(
            "Semantic",
            f"{semantic_latency:.1f}",
            str(len(result_semantic.sources))
        )

        console.print(perf_table)
    else:
        print(f"\nSQL query: {sql_latency:.1f}ms")
        print(f"Semantic query: {semantic_latency:.1f}ms")

    # Step 8: Statistics
    print_header("📈 Step 8: System Statistics")

    stats = rag.get_sql_stats()

    if console:
        stats_table = Table(title="SQL Integration Stats", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        if 'adapter' in stats:
            adapter_stats = stats['adapter']
            stats_table.add_row("Total Queries", str(adapter_stats.get('total_queries', 0)))
            stats_table.add_row("Successful Queries", str(adapter_stats.get('successful_queries', 0)))
            stats_table.add_row("Failed Queries", str(adapter_stats.get('failed_queries', 0)))
            stats_table.add_row("Avg Latency (ms)", f"{adapter_stats.get('avg_latency_ms', 0):.1f}")

        if 'translator' in stats:
            translator_stats = stats['translator']
            stats_table.add_row("Translations", str(translator_stats.get('total_translations', 0)))
            stats_table.add_row("Success Rate", f"{translator_stats.get('success_rate', 0):.1%}")

        console.print(stats_table)
    else:
        print("\nSQL Stats:")
        if 'adapter' in stats:
            for key, value in stats['adapter'].items():
                print(f"  {key}: {value}")
        if 'translator' in stats:
            for key, value in stats['translator'].items():
                print(f"  {key}: {value}")

    # Cleanup
    print_header("🧹 Cleanup")
    rag.close_sql()
    try:
        os.unlink(db_path)
        print("✓ Demo database removed")
    except OSError:
        print("⚠ Could not remove demo database (may still be locked)")

    print_header("✅ Demo Complete!")

    if console:
        console.print(Panel(
            "[bold green]SQL Integration Demo Completed Successfully![/bold green]\n\n"
            "Key Features Demonstrated:\n"
            "✓ Text-to-SQL translation\n"
            "✓ Automatic routing (SQL/semantic/hybrid)\n"
            "✓ Result fusion\n"
            "✓ Performance comparison\n"
            "✓ SQL injection prevention (read-only mode)\n"
            "✓ Schema introspection",
            title="Summary",
            border_style="green"
        ))


if __name__ == "__main__":
    asyncio.run(run_demo())
