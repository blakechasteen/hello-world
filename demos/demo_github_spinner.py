"""
GitHub Spinner Demo
===================

Demonstrates the GitHubSpinner for ingesting GitHub repository data.

Features shown:
- Issue extraction
- Pull request extraction
- Importance scoring
- Entity/motif extraction

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.github_spinner import GitHubSpinner, spin_github_repo


async def demo_github_spinner():
    """Run GitHub spinner demo."""
    print("=" * 70)
    print("GitHub Spinner Demo")
    print("=" * 70)

    # Create spinner
    spinner = GitHubSpinner(
        github_token=None,  # Uses GITHUB_TOKEN env var if set
        max_issues=10,
        max_prs=10
    )

    # Check availability
    if not spinner.is_available():
        print("Error: requests library not available")
        print("Install with: pip install requests")
        return

    print("\nSpinner capabilities:")
    caps = spinner.get_capabilities()
    print(f"  - Basic processing: {caps.basic_processing}")
    print(f"  - Importance scoring: {caps.importance_scoring}")
    print(f"  - Entity extraction: {caps.entity_extraction}")
    print(f"  - Motif extraction: {caps.motif_extraction}")

    # Demo with a public repository
    repo_url = "https://github.com/anthropic-ai/anthropic-sdk-python"

    print(f"\nProcessing repository: {repo_url}")
    print("-" * 70)

    try:
        result = await spinner.spin(repo_url)

        if result.success:
            print(f"\nSuccessfully processed repository!")
            print(f"  - Shards created: {result.shard_count}")
            print(f"  - Total entities: {result.entity_count}")
            print(f"  - Total motifs: {result.motif_count}")
            print(f"  - Average importance: {result.avg_importance:.2f}")
            print(f"  - Average confidence: {result.avg_confidence:.2f}")
            print(f"  - Processing time: {result.processing_time_ms:.1f}ms")

            # Display sample shards
            if result.shards:
                print("\nSample shards:")
                for i, shard in enumerate(result.shards[:3]):
                    print(f"\n  [{i+1}] {shard.metadata.get('title', 'Untitled')}")
                    print(f"      Type: {shard.metadata.get('type')}")
                    print(f"      State: {shard.metadata.get('state')}")
                    print(f"      Importance: {shard.metadata.get('importance_score', 0):.2f}")
                    print(f"      Entities: {len(shard.entities)}")
                    print(f"      Motifs: {shard.motifs[:3]}")

                if len(result.shards) > 3:
                    print(f"\n  ... and {len(result.shards) - 3} more shards")

        else:
            print(f"Error: {result.error_message}")
            if result.warnings:
                print(f"Warnings: {result.warnings}")

    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


async def demo_convenience_function():
    """Demonstrate using the convenience function."""
    print("\n" + "=" * 70)
    print("Convenience Function Demo")
    print("=" * 70)

    repo_url = "https://github.com/anthropic-ai/anthropic-sdk-python"

    print(f"\nUsing convenience function: spin_github_repo()")
    print(f"Repository: {repo_url}")

    try:
        result = await spin_github_repo(repo_url, max_issues=5, max_prs=5)

        if result.success:
            print(f"Result: {result.shard_count} shards created")
            print(f"Time: {result.processing_time_ms:.1f}ms")
        else:
            print(f"Error: {result.error_message}")

    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all demos."""
    await demo_github_spinner()
    await demo_convenience_function()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
