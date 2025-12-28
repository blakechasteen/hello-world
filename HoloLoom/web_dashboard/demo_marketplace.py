"""
Demo: Workflow Templates Marketplace

Demonstrates the complete marketplace functionality:
1. Seeds the database with categories and templates
2. Starts the marketplace API server
3. Shows example API calls for browsing, searching, using templates

Usage:
    # Seed database and start server
    python -m HoloLoom.web_dashboard.demo_marketplace

    # Just seed (no server)
    python -m HoloLoom.web_dashboard.demo_marketplace --seed-only

    # Just show API examples
    python -m HoloLoom.web_dashboard.demo_marketplace --examples-only
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Enable UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


async def seed_database(db_path: str = "./workflows.db"):
    """Seed the marketplace with categories and templates."""
    print("=" * 60)
    print("Step 1: Seeding Marketplace Database")
    print("=" * 60)

    from .marketplace_seed import run_seed
    await run_seed(db_path)


async def show_api_examples(db_path: str = "./workflows.db"):
    """Demonstrate marketplace API usage."""
    print("\n" + "=" * 60)
    print("Step 2: Marketplace API Examples")
    print("=" * 60)

    from .workflow_marketplace import create_marketplace_router
    from .workflow_persistence import WorkflowPersistence

    # Initialize persistence
    persistence = WorkflowPersistence(db_path)
    await persistence.initialize()

    # --- Categories ---
    print("\n📂 Categories:")
    print("-" * 40)
    categories = await persistence.get_all_categories()
    for cat in categories:
        print(f"  {cat.icon} {cat.name}: {cat.description}")

    # --- Featured Templates ---
    print("\n⭐ Featured Templates:")
    print("-" * 40)
    featured = await persistence.list_marketplace_templates(featured_only=True)
    for t in featured:
        workflow = await persistence.get_workflow(t.workflow_id)
        if workflow:
            rating = f"★{t.avg_rating:.1f}" if t.rating_count > 0 else "No ratings"
            print(f"  {workflow.name}")
            print(f"    Category: {t.category} | Difficulty: {t.difficulty}")
            print(f"    Rating: {rating} | Used: {t.usage_count}x")
            print()

    # --- All Templates by Category ---
    print("\n📋 All Templates by Category:")
    print("-" * 40)
    all_templates = await persistence.list_marketplace_templates()
    by_category = {}
    for t in all_templates:
        if t.category not in by_category:
            by_category[t.category] = []
        workflow = await persistence.get_workflow(t.workflow_id)
        if workflow:
            by_category[t.category].append((workflow.name, t.difficulty))

    for category, templates in sorted(by_category.items()):
        print(f"\n  {category.upper()}:")
        for name, difficulty in templates:
            badge = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(difficulty, "⚪")
            print(f"    {badge} {name}")

    # --- Search Example ---
    print("\n\n🔍 Search Example: 'research'")
    print("-" * 40)
    search_results = await persistence.search_marketplace_templates(
        query="research",
        limit=5
    )
    for t in search_results:
        workflow = await persistence.get_workflow(t.workflow_id)
        if workflow:
            print(f"  • {workflow.name} [{t.category}]")

    # --- Statistics ---
    print("\n\n📊 Marketplace Statistics:")
    print("-" * 40)
    total = len(all_templates)
    featured_count = len(featured)
    total_usage = sum(t.usage_count for t in all_templates)

    by_difficulty = {"beginner": 0, "intermediate": 0, "advanced": 0}
    for t in all_templates:
        if t.difficulty in by_difficulty:
            by_difficulty[t.difficulty] += 1

    print(f"  Total Templates: {total}")
    print(f"  Featured: {featured_count}")
    print(f"  Total Usage: {total_usage}")
    print(f"  By Difficulty:")
    print(f"    🟢 Beginner: {by_difficulty['beginner']}")
    print(f"    🟡 Intermediate: {by_difficulty['intermediate']}")
    print(f"    🔴 Advanced: {by_difficulty['advanced']}")


async def run_server(db_path: str = "./workflows.db", port: int = 8002):
    """Start the marketplace API server."""
    print("\n" + "=" * 60)
    print("Step 3: Starting Marketplace API Server")
    print("=" * 60)

    from .workflow_marketplace import run_standalone
    await run_standalone(db_path, port)


async def main():
    """Main demo entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Workflow Templates Marketplace Demo")
    parser.add_argument("--seed-only", action="store_true", help="Only seed database")
    parser.add_argument("--examples-only", action="store_true", help="Only show API examples")
    parser.add_argument("--no-server", action="store_true", help="Don't start server")
    parser.add_argument("--db-path", default="./workflows.db", help="Database path")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  HoloLoom Workflow Templates Marketplace Demo")
    print("=" * 60)
    print()

    db_path = args.db_path

    if args.examples_only:
        await show_api_examples(db_path)
        return

    if args.seed_only:
        await seed_database(db_path)
        return

    # Full demo: seed + examples + server
    await seed_database(db_path)
    await show_api_examples(db_path)

    if not args.no_server:
        print("\n" + "=" * 60)
        print("Starting Marketplace API Server")
        print("=" * 60)
        print(f"\nAPI Docs: http://localhost:{args.port}/docs")
        print(f"Templates: http://localhost:{args.port}/api/marketplace/templates")
        print(f"Featured: http://localhost:{args.port}/api/marketplace/templates/featured")
        print(f"Search: http://localhost:{args.port}/api/marketplace/templates/search?q=research")
        print("\nPress Ctrl+C to stop the server\n")
        await run_server(db_path, args.port)


def run():
    """CLI entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")


if __name__ == "__main__":
    run()
