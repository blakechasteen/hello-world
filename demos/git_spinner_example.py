"""
GitSpinner Usage Examples
==========================

Demonstrates Git repository ingestion with HoloLoom.

Examples:
1. Basic repository ingestion
2. Importance filtering
3. Incremental updates
4. Streaming large repositories
5. Integration with HoloLoom memory

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path

# Adjust import path as needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.spinningWheel.git_spinner import (
    GitSpinner,
    spin_repository,
    spin_repository_incremental
)
from hololoom.spinningWheel.utils import process_with_progress


# ============================================================================
# Example 1: Basic Repository Ingestion
# ============================================================================

async def example_basic():
    """Basic Git repository ingestion."""
    print("\n" + "="*70)
    print("Example 1: Basic Repository Ingestion")
    print("="*70)

    # Create spinner
    spinner = GitSpinner(max_commits=20)  # Limit to 20 most recent commits

    # Process repository (current directory or specify path)
    repo_path = "."  # Current directory
    # repo_path = "/path/to/your/repo"

    print(f"\nProcessing repository: {repo_path}")
    print("Fetching commits...")

    result = await spinner.spin(repo_path)

    # Display results
    print(f"\n✓ Success: {result.success}")
    print(f"✓ Processed {result.shard_count} commits")
    print(f"✓ Time: {result.processing_time_ms:.1f}ms")
    print(f"✓ Avg importance: {result.avg_importance:.2f}")
    print(f"✓ Total entities: {result.entity_count}")
    print(f"✓ Total motifs: {result.motif_count}")

    # Show sample commits
    print("\nSample commits:")
    for i, shard in enumerate(result.shards[:3]):  # First 3
        print(f"\n{i+1}. {shard.metadata['commit_hash_short']} - {shard.metadata.get('commit_type', 'unknown')}")
        print(f"   Author: {shard.metadata['author_name']}")
        print(f"   Importance: {shard.metadata['importance_score']:.2f}")
        print(f"   Files: {len(shard.metadata.get('files_changed', []))}")
        print(f"   Entities: {', '.join(shard.entities[:5])}")
        if shard.metadata['is_breaking']:
            print(f"   ⚠️  BREAKING CHANGE")

    return result


# ============================================================================
# Example 2: Importance Filtering
# ============================================================================

async def example_importance_filtering():
    """Filter commits by importance."""
    print("\n" + "="*70)
    print("Example 2: Importance Filtering")
    print("="*70)

    repo_path = "."

    # Low threshold (keep most commits)
    print("\nLow threshold (0.2) - Keep most commits:")
    spinner_low = GitSpinner(importance_threshold=0.2, max_commits=20)
    result_low = await spinner_low.spin(repo_path)
    print(f"  Kept: {result_low.shard_count} commits")

    # Medium threshold (balanced)
    print("\nMedium threshold (0.5) - Balanced:")
    spinner_med = GitSpinner(importance_threshold=0.5, max_commits=20)
    result_med = await spinner_med.spin(repo_path)
    print(f"  Kept: {result_med.shard_count} commits")

    # High threshold (only important)
    print("\nHigh threshold (0.8) - Only important:")
    spinner_high = GitSpinner(importance_threshold=0.8, max_commits=20)
    result_high = await spinner_high.spin(repo_path)
    print(f"  Kept: {result_high.shard_count} commits")

    if result_high.shard_count > 0:
        print("\nImportant commits:")
        for shard in result_high.shards[:5]:
            print(f"  - {shard.metadata['commit_hash_short']}: {shard.metadata.get('commit_type', '?')}")
            print(f"    Score: {shard.metadata['importance_score']:.2f}")
            print(f"    Reason: {shard.metadata['importance_reason']}")


# ============================================================================
# Example 3: Incremental Updates
# ============================================================================

async def example_incremental():
    """Incremental repository processing."""
    print("\n" + "="*70)
    print("Example 3: Incremental Updates")
    print("="*70)

    repo_path = "."
    checkpoint_dir = Path("/tmp/git_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # First run (processes all commits)
    print("\nFirst run (all commits):")
    result1 = await spin_repository_incremental(
        str(repo_path),
        checkpoint_dir=str(checkpoint_dir)
    )
    print(f"  Processed: {result1.shard_count} commits")

    # Second run (no new commits)
    print("\nSecond run (no new commits):")
    result2 = await spin_repository_incremental(
        str(repo_path),
        checkpoint_dir=str(checkpoint_dir)
    )
    print(f"  Processed: {result2.shard_count} new commits")

    print("\n✓ Checkpoint saved! Future runs will only process new commits.")
    print(f"✓ Checkpoint location: {checkpoint_dir}")


# ============================================================================
# Example 4: Streaming Large Repositories
# ============================================================================

async def example_streaming():
    """Stream commits from large repository."""
    print("\n" + "="*70)
    print("Example 4: Streaming Large Repository")
    print("="*70)

    repo_path = "."

    print("\nStreaming commits (processing one at a time):")
    print("This is memory-efficient for large repositories.\n")

    spinner = GitSpinner(max_commits=10)  # Limit for demo
    commit_count = 0

    async for shard in spinner.spin_stream(repo_path):
        commit_count += 1
        commit_type = shard.metadata.get('commit_type', 'unknown')
        importance = shard.metadata['importance_score']
        hash_short = shard.metadata['commit_hash_short']

        print(f"{commit_count}. [{hash_short}] {commit_type:10s} importance={importance:.2f}")

        # Could ingest immediately
        # await memory.add_shard(shard)

    print(f"\n✓ Streamed {commit_count} commits")


# ============================================================================
# Example 5: Integration with HoloLoom Memory
# ============================================================================

async def example_memory_integration():
    """Integrate with HoloLoom memory system."""
    print("\n" + "="*70)
    print("Example 5: Integration with HoloLoom Memory")
    print("="*70)

    try:
        from hololoom.memory.backend_factory import create_memory_backend
        from hololoom.config import Config, MemoryBackend
    except ImportError:
        print("\n⚠ HoloLoom memory backend not available")
        print("  Install required dependencies or use MinimalMemory fallback")
        return

    repo_path = "."

    # Create memory backend
    print("\nCreating memory backend...")
    config = Config.bare()
    config.memory_backend = MemoryBackend.INMEMORY

    memory = await create_memory_backend(config)
    print(f"✓ Created {config.memory_backend.value} backend")

    # Process repository
    print("\nProcessing Git repository...")
    spinner = GitSpinner(max_commits=10)
    result = await spinner.spin(repo_path)

    # Ingest shards into memory
    print(f"\nIngesting {result.shard_count} commits into memory...")

    if hasattr(memory, 'add_shards'):
        await memory.add_shards(result.shards)
    else:
        for shard in result.shards:
            if hasattr(memory, 'add_shard'):
                await memory.add_shard(shard)

    print(f"✓ Ingested {result.shard_count} commits")

    # Now you can query the memory
    print("\nMemory now contains Git commit history!")
    print("You can query commits like:")
    print("  - 'What commits fixed bugs?'")
    print("  - 'Show me breaking changes'")
    print("  - 'Who authored the most commits?'")


# ============================================================================
# Example 6: Custom Importance Scoring
# ============================================================================

async def example_custom_importance():
    """Custom importance scoring for specific needs."""
    print("\n" + "="*70)
    print("Example 6: Custom Importance Scoring")
    print("="*70)

    from hololoom.spinningWheel.protocol import ImportanceSignals, ImportanceScore
    from hololoom.spinningWheel.git_spinner import GitCommit

    # Create custom spinner with modified scoring
    class CustomGitSpinner(GitSpinner):
        def score_importance(self, commit: GitCommit) -> ImportanceScore:
            """Custom scoring: prioritize documentation updates."""
            # Use parent scoring first
            base_score = super().score_importance(commit)

            # Boost documentation commits
            if commit.commit_type == 'docs':
                signals = base_score.signals
                signals.custom_signals['documentation_boost'] = 0.8
                total = signals.compute_total()

                return ImportanceScore(
                    score=total,
                    signals=signals,
                    reason="documentation update (boosted)"
                )

            return base_score

    repo_path = "."

    print("\nProcessing with custom importance scoring...")
    print("(Documentation commits receive +0.8 boost)\n")

    spinner = CustomGitSpinner(importance_threshold=0.4, max_commits=20)
    result = await spinner.spin(repo_path)

    # Show docs commits
    docs_commits = [
        s for s in result.shards
        if s.metadata.get('commit_type') == 'docs'
    ]

    print(f"✓ Processed {result.shard_count} commits")
    print(f"✓ Documentation commits: {len(docs_commits)}")

    if docs_commits:
        print("\nDocumentation commits (boosted importance):")
        for shard in docs_commits[:3]:
            print(f"  - {shard.metadata['commit_hash_short']}")
            print(f"    Score: {shard.metadata['importance_score']:.2f}")


# ============================================================================
# Example 7: Analyze Repository Statistics
# ============================================================================

async def example_repository_stats():
    """Analyze repository using GitSpinner."""
    print("\n" + "="*70)
    print("Example 7: Repository Statistics")
    print("="*70)

    repo_path = "."

    print("\nAnalyzing repository...")

    spinner = GitSpinner(fetch_stats=True, max_commits=100)
    result = await spinner.spin(repo_path)

    # Compute statistics
    commit_types = {}
    total_files = 0
    total_insertions = 0
    total_deletions = 0
    authors = {}
    breaking_changes = 0

    for shard in result.shards:
        # Commit types
        commit_type = shard.metadata.get('commit_type', 'unknown')
        commit_types[commit_type] = commit_types.get(commit_type, 0) + 1

        # File stats
        files_changed = shard.metadata.get('files_changed', [])
        total_files += len(files_changed)
        total_insertions += shard.metadata.get('insertions', 0)
        total_deletions += shard.metadata.get('deletions', 0)

        # Authors
        author = shard.metadata['author_name']
        authors[author] = authors.get(author, 0) + 1

        # Breaking changes
        if shard.metadata['is_breaking']:
            breaking_changes += 1

    # Display stats
    print(f"\n📊 Repository Statistics (last {result.shard_count} commits)")
    print(f"\n{'='*50}")

    print(f"\n📝 Commit Types:")
    for commit_type, count in sorted(commit_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {commit_type:12s}: {count:3d} ({count/result.shard_count*100:5.1f}%)")

    print(f"\n👥 Top Authors:")
    for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {author:30s}: {count:3d} commits")

    print(f"\n📈 Code Changes:")
    print(f"  Files changed: {total_files}")
    print(f"  Insertions:    +{total_insertions:,}")
    print(f"  Deletions:     -{total_deletions:,}")
    print(f"  Net change:    {total_insertions - total_deletions:+,}")

    print(f"\n⚠️  Breaking Changes: {breaking_changes}")

    print(f"\n💡 Average Importance: {result.avg_importance:.2f}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GitSpinner Usage Examples")
    print("="*70)

    # Run examples
    await example_basic()
    await example_importance_filtering()
    await example_incremental()
    await example_streaming()
    await example_memory_integration()
    await example_custom_importance()
    await example_repository_stats()

    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)


if __name__ == "__main__":
    # Check if git is available
    import subprocess
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
    except Exception:
        print("Error: Git not found. Please install Git to run these examples.")
        sys.exit(1)

    asyncio.run(main())
