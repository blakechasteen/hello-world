"""
Debug script for GitSpinner incremental logic.
"""
import subprocess
from pathlib import Path
import tempfile

# Create test repo
tmp_path = Path(tempfile.mkdtemp())
repo_path = tmp_path / "test_repo"
repo_path.mkdir()

# Initialize repo
subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo_path, check=True, capture_output=True)
subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=repo_path, check=True, capture_output=True)

# Create 5 commits
commits_data = [
    ('feat: add new feature', 'This is a new feature'),
    ('fix: fix critical bug', 'Fixed a critical bug'),
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

# Get all commits
print("=" * 80)
print("ALL COMMITS (git log HEAD)")
print("=" * 80)
result = subprocess.run(
    ['git', 'log', 'HEAD', '--oneline'],
    cwd=repo_path,
    capture_output=True,
    text=True
)
print(result.stdout)
commits_list = result.stdout.strip().split('\n')
print(f"Total: {len(commits_list)} commits")

# Get the most recent commit hash
most_recent_hash = commits_list[0].split()[0]
print(f"\nMost recent commit: {most_recent_hash}")

# Now try the range syntax with branch specified (WRONG)
print("\n" + "=" * 80)
print(f"WRONG: git log HEAD {most_recent_hash}..HEAD")
print("=" * 80)
result = subprocess.run(
    ['git', 'log', 'HEAD', f'{most_recent_hash}..HEAD', '--oneline'],
    cwd=repo_path,
    capture_output=True,
    text=True
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"Count: {len([l for l in result.stdout.strip().split('\\n') if l])}")

# Now try the range syntax WITHOUT branch (CORRECT)
print("\n" + "=" * 80)
print(f"CORRECT: git log {most_recent_hash}..HEAD")
print("=" * 80)
result = subprocess.run(
    ['git', 'log', f'{most_recent_hash}..HEAD', '--oneline'],
    cwd=repo_path,
    capture_output=True,
    text=True
)
print(f"stdout: {result.stdout}")
print(f"stderr: {result.stderr}")
print(f"Count: {len([l for l in result.stdout.strip().split('\\n') if l])}")

# Cleanup
import shutil
shutil.rmtree(tmp_path)
print("\n✅ Test repo cleaned up")
