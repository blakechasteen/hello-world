"""
Debug script for GitSpinner checkpoint logic.
"""
import asyncio
import subprocess
from pathlib import Path
import tempfile
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.git_spinner import GitSpinner

async def main():
    # Create test repo
    tmp_path = Path(tempfile.mkdtemp())
    repo_path = tmp_path / "test_repo"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    repo_path.mkdir()

    # Initialize repo
    subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True, capture_output=True)

    # Create 5 commits (same as test)
    commits_data = [
        ('feat: add new feature', 'This is a new feature\n\nImplements #123'),
        ('fix: fix critical bug', 'Fixed a critical bug\n\nCloses GH-456'),
        ('docs: update README', 'Updated documentation'),
        ('feat!: breaking change', 'BREAKING CHANGE: This breaks API'),
        ('chore: update dependencies', 'Minor dependency updates')
    ]

    for i, (subject, body) in enumerate(commits_data):
        file_path = repo_path / f"file{i}.txt"
        file_path.write_text(f"Content {i}\n")
        subprocess.run(['git', 'add', f'file{i}.txt'], cwd=repo_path, check=True, capture_output=True)
        commit_msg = f"{subject}\n\n{body}"
        subprocess.run(['git', 'commit', '-m', commit_msg], cwd=repo_path, check=True, capture_output=True)

    # Create spinner
    spinner = GitSpinner(checkpoint_dir=checkpoint_dir)

    print("=" * 80)
    print("FIRST RUN (should get 5 commits)")
    print("=" * 80)
    result1 = await spinner.spin_incremental(repo_path)
    print(f"Shard count: {result1.shard_count}")
    print(f"Expected: 5")

    # Check checkpoint file
    from HoloLoom.spinningWheel.utils import create_source_id
    source_id = create_source_id(str(repo_path.absolute()))
    checkpoint_file = checkpoint_dir / f"git_{source_id}.json"

    print(f"\nCheckpoint file: {checkpoint_file}")
    print(f"Exists: {checkpoint_file.exists()}")

    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint data:")
        print(json.dumps(checkpoint_data, indent=2))
        print(f"\nLast processed commit: {checkpoint_data.get('last_processed_id')}")

    print("\n" + "=" * 80)
    print("SECOND RUN (should get 0 commits - no new commits)")
    print("=" * 80)
    result2 = await spinner.spin_incremental(repo_path)
    print(f"Shard count: {result2.shard_count}")
    print(f"Expected: 0")

    if result2.shard_count > 0:
        print(f"\n❌ FAILURE: Got {result2.shard_count} commits, expected 0")
        print(f"Commits returned:")
        for shard in result2.shards:
            commit_hash = shard.metadata.get('commit_hash_short', 'unknown')
            commit_msg = shard.text.split('\n')[0]
            print(f"  - {commit_hash}: {commit_msg}")
    else:
        print(f"\n✅ SUCCESS: Got 0 commits as expected")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmp_path)
        print(f"\n✅ Test repo cleaned up")
    except Exception as e:
        print(f"\n⚠ Warning: Could not clean up test repo: {e}")

if __name__ == '__main__':
    asyncio.run(main())
