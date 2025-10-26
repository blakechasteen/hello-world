"""
HoloLoom Knowledge Graph Migration Utility
==========================================
Migrate knowledge graph data between NetworkX and Neo4j backends.

This utility provides bidirectional migration:
- NetworkX → Neo4j: Persist in-memory graph to Neo4j
- Neo4j → NetworkX: Export Neo4j graph for analysis
- JSONL → Neo4j: Bulk load from saved NetworkX format

Usage:
    python -m holoLoom.memory.migrate_to_neo4j --from-networkx kg.jsonl
    python -m holoLoom.memory.migrate_to_neo4j --from-neo4j --output kg_export.jsonl
    python -m holoLoom.memory.migrate_to_neo4j --merge kg1.jsonl kg2.jsonl
"""

import argparse
from pathlib import Path
from typing import Optional
import sys

try:
    from holoLoom.memory.graph import KG
    from holoLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig
except ImportError:
    from HoloLoom.memory.graph import KG
    from HoloLoom.memory.neo4j_graph import Neo4jKG, Neo4jConfig


def migrate_networkx_to_neo4j(
    jsonl_path: str,
    neo4j_config: Optional[Neo4jConfig] = None,
    clear_existing: bool = False
) -> None:
    """
    Migrate NetworkX graph (JSONL format) to Neo4j.

    Args:
        jsonl_path: Path to NetworkX JSONL export
        neo4j_config: Neo4j connection config
        clear_existing: If True, clear Neo4j database before migration
    """
    print(f"Loading NetworkX graph from {jsonl_path}...")
    kg = KG.load(jsonl_path)
    stats = kg.stats()
    print(f"  Loaded: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    print("\nConnecting to Neo4j...")
    neo4j_kg = Neo4jKG(neo4j_config or Neo4jConfig())

    if clear_existing:
        print("Clearing existing Neo4j data...")
        neo4j_kg.clear()

    print("Migrating to Neo4j...")
    neo4j_kg.import_from_networkx(kg.G)

    neo4j_stats = neo4j_kg.stats()
    print(f"  Migrated: {neo4j_stats['num_nodes']} nodes, {neo4j_stats['num_edges']} edges")

    neo4j_kg.close()
    print("\n✓ Migration complete!")


def migrate_neo4j_to_networkx(
    output_path: str,
    neo4j_config: Optional[Neo4jConfig] = None
) -> None:
    """
    Export Neo4j graph to NetworkX JSONL format.

    Args:
        output_path: Path to save NetworkX JSONL export
        neo4j_config: Neo4j connection config
    """
    print("Connecting to Neo4j...")
    neo4j_kg = Neo4jKG(neo4j_config or Neo4jConfig())

    neo4j_stats = neo4j_kg.stats()
    print(f"  Neo4j graph: {neo4j_stats['num_nodes']} nodes, {neo4j_stats['num_edges']} edges")

    print("\nExporting to NetworkX...")
    nx_graph = neo4j_kg.export_to_networkx()

    print(f"Saving to {output_path}...")
    kg = KG()
    kg.G = nx_graph
    kg.save(output_path)

    stats = kg.stats()
    print(f"  Saved: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    neo4j_kg.close()
    print("\n✓ Export complete!")


def merge_graphs(
    input_paths: list[str],
    output_path: str,
    to_neo4j: bool = False,
    neo4j_config: Optional[Neo4jConfig] = None
) -> None:
    """
    Merge multiple NetworkX graphs.

    Args:
        input_paths: List of JSONL files to merge
        output_path: Path to save merged graph (or "neo4j" for Neo4j backend)
        to_neo4j: If True, merge into Neo4j instead of file
        neo4j_config: Neo4j connection config
    """
    print(f"Merging {len(input_paths)} graphs...")

    kg_merged = KG()

    for path in input_paths:
        print(f"  Loading {path}...")
        kg = KG.load(path)
        stats = kg.stats()
        print(f"    {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        kg_merged.merge(kg)

    merged_stats = kg_merged.stats()
    print(f"\nMerged graph: {merged_stats['num_nodes']} nodes, {merged_stats['num_edges']} edges")

    if to_neo4j:
        print("\nMigrating merged graph to Neo4j...")
        neo4j_kg = Neo4jKG(neo4j_config or Neo4jConfig())
        neo4j_kg.import_from_networkx(kg_merged.G)
        neo4j_kg.close()
        print("✓ Merged into Neo4j!")
    else:
        print(f"\nSaving merged graph to {output_path}...")
        kg_merged.save(output_path)
        print("✓ Merge complete!")


def verify_migration(
    jsonl_path: str,
    neo4j_config: Optional[Neo4jConfig] = None
) -> bool:
    """
    Verify that NetworkX and Neo4j graphs match.

    Args:
        jsonl_path: Path to NetworkX JSONL export
        neo4j_config: Neo4j connection config

    Returns:
        True if graphs match, False otherwise
    """
    print("Loading NetworkX graph...")
    kg = KG.load(jsonl_path)
    nx_stats = kg.stats()

    print("Loading Neo4j graph...")
    neo4j_kg = Neo4jKG(neo4j_config or Neo4jConfig())
    neo4j_stats = neo4j_kg.stats()

    print("\nComparison:")
    print(f"  NetworkX: {nx_stats['num_nodes']} nodes, {nx_stats['num_edges']} edges")
    print(f"  Neo4j:    {neo4j_stats['num_nodes']} nodes, {neo4j_stats['num_edges']} edges")

    match = (
        nx_stats['num_nodes'] == neo4j_stats['num_nodes'] and
        nx_stats['num_edges'] == neo4j_stats['num_edges']
    )

    if match:
        print("\n✓ Graphs match!")
    else:
        print("\n✗ Graphs do not match!")

    neo4j_kg.close()
    return match


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate HoloLoom knowledge graphs between NetworkX and Neo4j"
    )

    # Migration mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--from-networkx",
        metavar="JSONL_FILE",
        help="Migrate from NetworkX JSONL to Neo4j"
    )
    mode_group.add_argument(
        "--from-neo4j",
        action="store_true",
        help="Export from Neo4j to NetworkX JSONL"
    )
    mode_group.add_argument(
        "--merge",
        nargs="+",
        metavar="JSONL_FILE",
        help="Merge multiple NetworkX graphs"
    )
    mode_group.add_argument(
        "--verify",
        metavar="JSONL_FILE",
        help="Verify NetworkX and Neo4j graphs match"
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        default="kg_export.jsonl",
        help="Output path for exported/merged graph (default: kg_export.jsonl)"
    )

    # Neo4j connection
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        default="hololoom123",
        help="Neo4j password (default: hololoom123)"
    )
    parser.add_argument(
        "--neo4j-database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )

    # Additional options
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear Neo4j database before migration"
    )
    parser.add_argument(
        "--to-neo4j",
        action="store_true",
        help="Merge directly into Neo4j (use with --merge)"
    )

    args = parser.parse_args()

    # Build Neo4j config
    neo4j_config = Neo4jConfig(
        uri=args.neo4j_uri,
        username=args.neo4j_user,
        password=args.neo4j_password,
        database=args.neo4j_database
    )

    try:
        # Execute migration
        if args.from_networkx:
            migrate_networkx_to_neo4j(
                args.from_networkx,
                neo4j_config,
                clear_existing=args.clear
            )

        elif args.from_neo4j:
            migrate_neo4j_to_networkx(
                args.output,
                neo4j_config
            )

        elif args.merge:
            merge_graphs(
                args.merge,
                args.output,
                to_neo4j=args.to_neo4j,
                neo4j_config=neo4j_config
            )

        elif args.verify:
            success = verify_migration(args.verify, neo4j_config)
            sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
