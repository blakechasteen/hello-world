"""
Agent Git Store - Actual Git Storage for Thread History

Uses dulwich (pure Python git implementation) to provide real git storage
for agent thread history. Each thread gets its own bare git repo, enabling:
- Standard git tooling works (`git log`, `gitk`, `git diff`)
- Delta compression via packfiles
- Battle-tested DAG and object store
- Could push/pull agent histories to remote repos

Status: Production Ready (December 2025)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
import logging
import time

logger = logging.getLogger(__name__)

# Lazy dulwich imports - graceful degradation if not installed
try:
    from dulwich.repo import Repo
    from dulwich.objects import Blob, Tree, Commit
    from dulwich import porcelain
    DULWICH_AVAILABLE = True
except ImportError:
    DULWICH_AVAILABLE = False
    logger.warning("dulwich not installed - git storage will use in-memory fallback")


@dataclass(frozen=True)
class StepCommit:
    """
    Atomic, immutable snapshot of thread state after a step.

    Content-addressable: commit_id = SHA256(parent_id + state_hash)
    """
    # Identity
    commit_id: str                          # SHA256 hash
    parent_id: Optional[str]                # Previous commit (None for root)

    # Metadata
    thread_id: str
    step_index: int
    timestamp: float                        # Unix timestamp
    author: str                             # "agent" or "user"
    message: str                            # Auto-generated description

    # Execution state
    query: str
    reasoning_mode: str
    confidence: float
    epistemic_confidence: float
    tokens_used: int
    cost_usd: float

    # Context (stored as tuple for immutability)
    context_window: Tuple[str, ...] = field(default_factory=tuple)

    # Reasoning state
    mrf_strategy: Optional[str] = None
    mcts_state: Optional[bytes] = None      # Serialized MCTSState

    # Results
    response: Optional[str] = None
    tool_selected: Optional[str] = None

    # Integrity
    state_hash: str = ""

    def __hash__(self) -> int:
        return hash(self.commit_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "commit_id": self.commit_id,
            "parent_id": self.parent_id,
            "thread_id": self.thread_id,
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "author": self.author,
            "message": self.message,
            "query": self.query,
            "reasoning_mode": self.reasoning_mode,
            "confidence": self.confidence,
            "epistemic_confidence": self.epistemic_confidence,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "context_window": list(self.context_window),
            "mrf_strategy": self.mrf_strategy,
            "response": self.response,
            "tool_selected": self.tool_selected,
            "state_hash": self.state_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepCommit":
        """Deserialize from dict."""
        return cls(
            commit_id=data["commit_id"],
            parent_id=data.get("parent_id"),
            thread_id=data["thread_id"],
            step_index=data["step_index"],
            timestamp=data["timestamp"],
            author=data["author"],
            message=data["message"],
            query=data["query"],
            reasoning_mode=data["reasoning_mode"],
            confidence=data["confidence"],
            epistemic_confidence=data.get("epistemic_confidence", 0.0),
            tokens_used=data["tokens_used"],
            cost_usd=data["cost_usd"],
            context_window=tuple(data.get("context_window", [])),
            mrf_strategy=data.get("mrf_strategy"),
            response=data.get("response"),
            tool_selected=data.get("tool_selected"),
            state_hash=data.get("state_hash", "")
        )


@dataclass
class ReflogEntry:
    """Git-style reflog for operation history."""
    timestamp: float
    old_commit_id: str
    new_commit_id: str
    operation: str                          # "commit", "checkout", "branch", "merge", "reset"
    message: str                            # e.g., "checkout: moving from main to feature/mcts"


class InMemoryGitStore:
    """
    In-memory fallback when dulwich is not available.

    Provides same API but stores everything in memory (no persistence).
    """

    def __init__(self):
        # thread_id -> {commits: {sha: data}, branches: {name: sha}, HEAD: sha, reflog: []}
        self._repos: Dict[str, Dict[str, Any]] = {}

    def init_thread_repo(self, thread_id: str) -> None:
        """Initialize in-memory repo for a thread."""
        if thread_id not in self._repos:
            self._repos[thread_id] = {
                "commits": {},
                "branches": {"main": None},
                "HEAD": None,
                "head_ref": "main",
                "reflog": []
            }
            logger.info(f"Initialized in-memory repo for thread: {thread_id}")

    def commit_step(
        self,
        thread_id: str,
        step_commit: StepCommit,
        parent_sha: Optional[str] = None
    ) -> str:
        """Create a commit from StepCommit. Returns commit SHA."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        # Generate SHA from content
        content = json.dumps(step_commit.to_dict(), sort_keys=True)
        sha = hashlib.sha256(content.encode()).hexdigest()[:40]

        # Store commit
        repo["commits"][sha] = {
            "data": step_commit.to_dict(),
            "parent": parent_sha,
            "message": step_commit.message,
            "timestamp": step_commit.timestamp,
            "author": step_commit.author
        }

        # Update branch and HEAD
        current_branch = repo["head_ref"]
        old_head = repo["HEAD"]
        repo["branches"][current_branch] = sha
        repo["HEAD"] = sha

        # Add reflog entry
        repo["reflog"].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_head or "",
            new_commit_id=sha,
            operation="commit",
            message=f"commit: {step_commit.message}"
        ))

        return sha

    def create_branch(self, thread_id: str, branch_name: str, commit_sha: Optional[str] = None) -> None:
        """Create a branch pointing to commit (default: HEAD)."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        target = commit_sha or repo["HEAD"]
        if target is None:
            raise ValueError("Cannot create branch: no commits yet")

        repo["branches"][branch_name] = target
        logger.info(f"Created branch {branch_name} at {target[:8]}")

    def checkout(self, thread_id: str, target: str) -> str:
        """Move HEAD to branch or commit. Returns new HEAD SHA."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        old_head = repo["HEAD"]

        # Try as branch first
        if target in repo["branches"]:
            repo["head_ref"] = target
            repo["HEAD"] = repo["branches"][target]
        # Try as commit SHA
        elif target in repo["commits"]:
            repo["HEAD"] = target
            repo["head_ref"] = None  # Detached HEAD
        else:
            raise ValueError(f"Unknown target: {target}")

        # Add reflog entry
        repo["reflog"].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_head or "",
            new_commit_id=repo["HEAD"],
            operation="checkout",
            message=f"checkout: moving to {target}"
        ))

        return repo["HEAD"]

    def get_log(self, thread_id: str, max_count: int = 50) -> List[Dict]:
        """Get commit history (git log)."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        commits = []
        current = repo["HEAD"]

        while current and len(commits) < max_count:
            commit_data = repo["commits"].get(current)
            if not commit_data:
                break

            commits.append({
                "sha": current,
                "message": commit_data["message"],
                "timestamp": commit_data["timestamp"],
                "author": commit_data["author"],
                "parents": [commit_data["parent"]] if commit_data["parent"] else [],
                "state": commit_data["data"]
            })

            current = commit_data["parent"]

        return commits

    def diff(self, thread_id: str, sha1: str, sha2: str) -> Dict[str, Any]:
        """Compare two commits."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        commit1 = repo["commits"].get(sha1)
        commit2 = repo["commits"].get(sha2)

        if not commit1 or not commit2:
            raise ValueError("Commit not found")

        state1 = commit1["data"]
        state2 = commit2["data"]

        # Compute diff
        diff = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        for key in all_keys:
            v1, v2 = state1.get(key), state2.get(key)
            if v1 != v2:
                diff[key] = {"old": v1, "new": v2}

        return {
            "from_sha": sha1,
            "to_sha": sha2,
            "changes": diff
        }

    def list_branches(self, thread_id: str) -> List[str]:
        """List all branches."""
        self._ensure_repo(thread_id)
        return list(self._repos[thread_id]["branches"].keys())

    def delete_branch(self, thread_id: str, branch_name: str) -> None:
        """Delete a branch."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        if branch_name == "main":
            raise ValueError("Cannot delete main branch")
        if branch_name == repo["head_ref"]:
            raise ValueError("Cannot delete current branch")
        if branch_name not in repo["branches"]:
            raise ValueError(f"Branch not found: {branch_name}")

        del repo["branches"][branch_name]

    def reset(self, thread_id: str, target_sha: str, mode: str = "soft") -> None:
        """Reset HEAD to target commit."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        if target_sha not in repo["commits"]:
            raise ValueError(f"Commit not found: {target_sha}")

        old_head = repo["HEAD"]

        # Update HEAD and current branch
        repo["HEAD"] = target_sha
        if repo["head_ref"]:
            repo["branches"][repo["head_ref"]] = target_sha

        # Add reflog entry
        repo["reflog"].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_head or "",
            new_commit_id=target_sha,
            operation="reset",
            message=f"reset ({mode}): moving to {target_sha[:8]}"
        ))

    def cherry_pick(self, thread_id: str, commit_sha: str) -> str:
        """Cherry-pick a commit onto current branch. Returns new commit SHA."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        source_commit = repo["commits"].get(commit_sha)
        if not source_commit:
            raise ValueError(f"Commit not found: {commit_sha}")

        # Create new commit with same content but different parent
        new_data = source_commit["data"].copy()
        new_data["timestamp"] = time.time()

        content = json.dumps(new_data, sort_keys=True)
        new_sha = hashlib.sha256(content.encode()).hexdigest()[:40]

        repo["commits"][new_sha] = {
            "data": new_data,
            "parent": repo["HEAD"],
            "message": f"Cherry-pick: {source_commit['message']}",
            "timestamp": time.time(),
            "author": source_commit["author"]
        }

        # Update branch and HEAD
        current_branch = repo["head_ref"]
        if current_branch:
            repo["branches"][current_branch] = new_sha
        repo["HEAD"] = new_sha

        return new_sha

    def get_reflog(self, thread_id: str, limit: int = 50) -> List[Dict]:
        """Get reflog entries."""
        self._ensure_repo(thread_id)
        entries = self._repos[thread_id]["reflog"][-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "old_commit_id": e.old_commit_id,
                "new_commit_id": e.new_commit_id,
                "operation": e.operation,
                "message": e.message
            }
            for e in reversed(entries)
        ]

    def get_merge_base(self, thread_id: str, branch1: str, branch2: str) -> Optional[str]:
        """Find common ancestor of two branches."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        # Get commit SHAs for branches
        sha1 = repo["branches"].get(branch1)
        sha2 = repo["branches"].get(branch2)

        if not sha1 or not sha2:
            return None

        # Build ancestor set for sha1
        ancestors1 = set()
        current = sha1
        while current:
            ancestors1.add(current)
            commit = repo["commits"].get(current)
            if not commit:
                break
            current = commit["parent"]

        # Walk sha2 ancestors until we find common one
        current = sha2
        while current:
            if current in ancestors1:
                return current
            commit = repo["commits"].get(current)
            if not commit:
                break
            current = commit["parent"]

        return None

    def merge(
        self,
        thread_id: str,
        source_branch: str,
        strategy: str = "best"
    ) -> str:
        """Merge source branch into current branch. Returns merge commit SHA."""
        self._ensure_repo(thread_id)
        repo = self._repos[thread_id]

        source_sha = repo["branches"].get(source_branch)
        if not source_sha:
            raise ValueError(f"Branch not found: {source_branch}")

        source_commit = repo["commits"].get(source_sha)
        if not source_commit:
            raise ValueError(f"Commit not found: {source_sha}")

        # Create merge commit
        merge_data = source_commit["data"].copy()
        merge_data["timestamp"] = time.time()
        merge_data["message"] = f"Merge {source_branch} into {repo['head_ref']} ({strategy})"

        content = json.dumps(merge_data, sort_keys=True)
        merge_sha = hashlib.sha256(content.encode()).hexdigest()[:40]

        repo["commits"][merge_sha] = {
            "data": merge_data,
            "parent": repo["HEAD"],  # Simplified: only tracks first parent
            "message": merge_data["message"],
            "timestamp": time.time(),
            "author": "agent"
        }

        # Update branch and HEAD
        current_branch = repo["head_ref"]
        if current_branch:
            repo["branches"][current_branch] = merge_sha
        repo["HEAD"] = merge_sha

        # Add reflog entry
        repo["reflog"].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=source_sha,
            new_commit_id=merge_sha,
            operation="merge",
            message=f"merge {source_branch}: {strategy}"
        ))

        return merge_sha

    def _ensure_repo(self, thread_id: str) -> None:
        """Ensure repo exists, create if not."""
        if thread_id not in self._repos:
            self.init_thread_repo(thread_id)


class DulwichGitStore:
    """
    Actual git storage using dulwich (pure Python git implementation).

    Each thread gets its own bare git repo. Standard git tooling works:
    - git log --graph --all
    - git show <commit>:state.json
    - git diff <sha1>..<sha2>
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # In-memory tracking for HEAD ref (dulwich doesn't track this well)
        self._head_refs: Dict[str, str] = {}  # thread_id -> branch name
        self._reflogs: Dict[str, List[ReflogEntry]] = {}

    def _repo_path(self, thread_id: str) -> Path:
        """Get repo path for thread."""
        return self.base_path / f"{thread_id}.git"

    def _get_repo(self, thread_id: str) -> "Repo":
        """Get or create repo for thread."""
        repo_path = self._repo_path(thread_id)
        if not repo_path.exists():
            self.init_thread_repo(thread_id)
        return Repo(str(repo_path))

    def init_thread_repo(self, thread_id: str) -> "Repo":
        """Create a new bare git repo for a thread."""
        repo_path = self._repo_path(thread_id)
        if repo_path.exists():
            return Repo(str(repo_path))

        repo = Repo.init_bare(str(repo_path))
        self._head_refs[thread_id] = "main"
        self._reflogs[thread_id] = []
        logger.info(f"Initialized git repo for thread: {thread_id} at {repo_path}")
        return repo

    def commit_step(
        self,
        thread_id: str,
        step_commit: StepCommit,
        parent_sha: Optional[bytes] = None
    ) -> str:
        """Create a real git commit from StepCommit. Returns commit SHA hex."""
        repo = self._get_repo(thread_id)

        # 1. Create blob for step state (JSON serialized)
        state_json = json.dumps(step_commit.to_dict(), indent=2).encode('utf-8')
        blob = Blob.from_string(state_json)
        repo.object_store.add_object(blob)

        # 2. Create tree with single "state.json" entry
        tree = Tree()
        tree.add(b"state.json", 0o100644, blob.id)
        repo.object_store.add_object(tree)

        # 3. Create commit
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = (
            f"{step_commit.author} <agent@hololoom.local>".encode()
        )
        commit.author_time = commit.commit_time = int(step_commit.timestamp)
        commit.author_timezone = commit.commit_timezone = 0
        commit.message = step_commit.message.encode('utf-8')

        if parent_sha:
            commit.parents = [parent_sha]
        else:
            commit.parents = []

        repo.object_store.add_object(commit)

        # 4. Update branch ref
        branch_name = self._head_refs.get(thread_id, "main")
        ref_name = f"refs/heads/{branch_name}".encode()
        old_sha = repo.refs.get(ref_name)
        repo.refs[ref_name] = commit.id

        # 5. Update HEAD
        repo.refs[b"HEAD"] = commit.id

        # 6. Add reflog entry
        if thread_id not in self._reflogs:
            self._reflogs[thread_id] = []
        self._reflogs[thread_id].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_sha.decode() if old_sha else "",
            new_commit_id=commit.id.hex(),
            operation="commit",
            message=f"commit: {step_commit.message}"
        ))

        return commit.id.hex()

    def create_branch(self, thread_id: str, branch_name: str, commit_sha: Optional[str] = None) -> None:
        """Create a git branch (ref) pointing to commit."""
        repo = self._get_repo(thread_id)

        if commit_sha:
            target = bytes.fromhex(commit_sha)
        else:
            target = repo.refs.get(b"HEAD")
            if not target:
                raise ValueError("No commits to branch from")

        repo.refs[f"refs/heads/{branch_name}".encode()] = target
        logger.info(f"Created branch {branch_name} at {target.hex()[:8]}")

    def checkout(self, thread_id: str, target: str) -> str:
        """Move HEAD to branch or commit. Returns new HEAD SHA hex."""
        repo = self._get_repo(thread_id)
        old_head = repo.refs.get(b"HEAD")

        # Try as branch first
        ref_name = f"refs/heads/{target}".encode()
        if ref_name in repo.refs:
            repo.refs[b"HEAD"] = repo.refs[ref_name]
            self._head_refs[thread_id] = target
            new_sha = repo.refs[ref_name].hex()
        else:
            # Try as commit SHA
            try:
                commit_sha = bytes.fromhex(target)
                if commit_sha in repo.object_store:
                    repo.refs[b"HEAD"] = commit_sha
                    self._head_refs[thread_id] = None  # Detached HEAD
                    new_sha = target
                else:
                    raise ValueError(f"Unknown target: {target}")
            except ValueError:
                raise ValueError(f"Unknown target: {target}")

        # Add reflog entry
        if thread_id not in self._reflogs:
            self._reflogs[thread_id] = []
        self._reflogs[thread_id].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_head.hex() if old_head else "",
            new_commit_id=new_sha,
            operation="checkout",
            message=f"checkout: moving to {target}"
        ))

        return new_sha

    def get_log(self, thread_id: str, max_count: int = 50) -> List[Dict]:
        """Get commit history (git log)."""
        repo = self._get_repo(thread_id)
        head = repo.refs.get(b"HEAD")

        if not head:
            return []

        commits = []
        from dulwich.walk import Walker

        walker = Walker(repo.object_store, include=[head], max_entries=max_count)

        for entry in walker:
            commit = entry.commit

            # Read state.json blob
            tree = repo.object_store[commit.tree]
            state_entry = None
            for item in tree.items():
                if item.path == b"state.json":
                    state_entry = item
                    break

            state = {}
            if state_entry:
                state_blob = repo.object_store[state_entry.sha]
                state = json.loads(state_blob.data.decode())

            commits.append({
                "sha": commit.id.hex(),
                "message": commit.message.decode(),
                "timestamp": commit.commit_time,
                "author": commit.author.decode(),
                "parents": [p.hex() for p in commit.parents],
                "state": state
            })

        return commits

    def diff(self, thread_id: str, sha1: str, sha2: str) -> Dict[str, Any]:
        """Compare two commits."""
        repo = self._get_repo(thread_id)

        # Load both states
        state1 = self._load_state(repo, bytes.fromhex(sha1))
        state2 = self._load_state(repo, bytes.fromhex(sha2))

        # Compute diff
        changes = {}
        all_keys = set(state1.keys()) | set(state2.keys())
        for key in all_keys:
            v1, v2 = state1.get(key), state2.get(key)
            if v1 != v2:
                changes[key] = {"old": v1, "new": v2}

        return {
            "from_sha": sha1,
            "to_sha": sha2,
            "changes": changes
        }

    def _load_state(self, repo: "Repo", commit_sha: bytes) -> Dict:
        """Load state.json from a commit."""
        commit = repo.object_store[commit_sha]
        tree = repo.object_store[commit.tree]

        for item in tree.items():
            if item.path == b"state.json":
                blob = repo.object_store[item.sha]
                return json.loads(blob.data.decode())

        return {}

    def list_branches(self, thread_id: str) -> List[str]:
        """List all branches."""
        repo = self._get_repo(thread_id)
        branches = []

        for ref in repo.refs.allkeys():
            if ref.startswith(b"refs/heads/"):
                branch_name = ref[len(b"refs/heads/"):].decode()
                branches.append(branch_name)

        return branches

    def delete_branch(self, thread_id: str, branch_name: str) -> None:
        """Delete a branch."""
        if branch_name == "main":
            raise ValueError("Cannot delete main branch")

        current_branch = self._head_refs.get(thread_id)
        if branch_name == current_branch:
            raise ValueError("Cannot delete current branch")

        repo = self._get_repo(thread_id)
        ref_name = f"refs/heads/{branch_name}".encode()

        if ref_name not in repo.refs:
            raise ValueError(f"Branch not found: {branch_name}")

        del repo.refs[ref_name]

    def reset(self, thread_id: str, target_sha: str, mode: str = "soft") -> None:
        """Reset HEAD to target commit."""
        repo = self._get_repo(thread_id)
        old_head = repo.refs.get(b"HEAD")

        target = bytes.fromhex(target_sha)
        if target not in repo.object_store:
            raise ValueError(f"Commit not found: {target_sha}")

        # Update HEAD
        repo.refs[b"HEAD"] = target

        # Update current branch
        branch_name = self._head_refs.get(thread_id)
        if branch_name:
            repo.refs[f"refs/heads/{branch_name}".encode()] = target

        # Add reflog entry
        if thread_id not in self._reflogs:
            self._reflogs[thread_id] = []
        self._reflogs[thread_id].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=old_head.hex() if old_head else "",
            new_commit_id=target_sha,
            operation="reset",
            message=f"reset ({mode}): moving to {target_sha[:8]}"
        ))

    def cherry_pick(self, thread_id: str, commit_sha: str) -> str:
        """Cherry-pick a commit onto current branch. Returns new commit SHA."""
        repo = self._get_repo(thread_id)

        source = bytes.fromhex(commit_sha)
        source_commit = repo.object_store[source]

        # Create new commit with same tree but different parent
        commit = Commit()
        commit.tree = source_commit.tree
        commit.author = commit.committer = source_commit.author
        commit.author_time = commit.commit_time = int(time.time())
        commit.author_timezone = commit.commit_timezone = 0
        commit.message = f"Cherry-pick: {source_commit.message.decode()}".encode()

        current_head = repo.refs.get(b"HEAD")
        if current_head:
            commit.parents = [current_head]
        else:
            commit.parents = []

        repo.object_store.add_object(commit)

        # Update refs
        branch_name = self._head_refs.get(thread_id)
        if branch_name:
            repo.refs[f"refs/heads/{branch_name}".encode()] = commit.id
        repo.refs[b"HEAD"] = commit.id

        return commit.id.hex()

    def get_reflog(self, thread_id: str, limit: int = 50) -> List[Dict]:
        """Get reflog entries."""
        entries = self._reflogs.get(thread_id, [])[-limit:]
        return [
            {
                "timestamp": e.timestamp,
                "old_commit_id": e.old_commit_id,
                "new_commit_id": e.new_commit_id,
                "operation": e.operation,
                "message": e.message
            }
            for e in reversed(entries)
        ]

    def get_merge_base(self, thread_id: str, branch1: str, branch2: str) -> Optional[str]:
        """Find common ancestor of two branches."""
        repo = self._get_repo(thread_id)

        sha1 = repo.refs.get(f"refs/heads/{branch1}".encode())
        sha2 = repo.refs.get(f"refs/heads/{branch2}".encode())

        if not sha1 or not sha2:
            return None

        # Build ancestor set for sha1
        ancestors1 = set()
        to_visit = [sha1]
        while to_visit:
            current = to_visit.pop()
            if current in ancestors1:
                continue
            ancestors1.add(current)
            commit = repo.object_store[current]
            to_visit.extend(commit.parents)

        # Walk sha2 ancestors until we find common one
        to_visit = [sha2]
        visited = set()
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in ancestors1:
                return current.hex()
            commit = repo.object_store[current]
            to_visit.extend(commit.parents)

        return None

    def merge(
        self,
        thread_id: str,
        source_branch: str,
        strategy: str = "best"
    ) -> str:
        """Merge source branch into current branch. Returns merge commit SHA."""
        repo = self._get_repo(thread_id)

        source_ref = f"refs/heads/{source_branch}".encode()
        source_sha = repo.refs.get(source_ref)
        if not source_sha:
            raise ValueError(f"Branch not found: {source_branch}")

        current_head = repo.refs.get(b"HEAD")
        if not current_head:
            raise ValueError("No commits on current branch")

        source_commit = repo.object_store[source_sha]

        # Create merge commit
        commit = Commit()
        commit.tree = source_commit.tree  # Use source tree (simplified merge)
        commit.author = commit.committer = b"agent <agent@hololoom.local>"
        commit.author_time = commit.commit_time = int(time.time())
        commit.author_timezone = commit.commit_timezone = 0

        branch_name = self._head_refs.get(thread_id, "main")
        commit.message = f"Merge {source_branch} into {branch_name} ({strategy})".encode()
        commit.parents = [current_head, source_sha]  # Merge commit has 2 parents

        repo.object_store.add_object(commit)

        # Update refs
        if branch_name:
            repo.refs[f"refs/heads/{branch_name}".encode()] = commit.id
        repo.refs[b"HEAD"] = commit.id

        # Add reflog entry
        if thread_id not in self._reflogs:
            self._reflogs[thread_id] = []
        self._reflogs[thread_id].append(ReflogEntry(
            timestamp=time.time(),
            old_commit_id=current_head.hex(),
            new_commit_id=commit.id.hex(),
            operation="merge",
            message=f"merge {source_branch}: {strategy}"
        ))

        return commit.id.hex()


class AgentGitStore:
    """
    Factory class that provides either dulwich-based git storage
    or in-memory fallback depending on availability.

    Usage:
        store = AgentGitStore("./data/agent-repos")
        store.init_thread_repo("thread-123")
        sha = store.commit_step("thread-123", step_commit)
    """

    def __init__(self, base_path: str = "./data/agent-repos"):
        if DULWICH_AVAILABLE:
            self._store = DulwichGitStore(base_path)
            logger.info(f"Using dulwich git storage at {base_path}")
        else:
            self._store = InMemoryGitStore()
            logger.warning("dulwich not available - using in-memory git storage")

    @property
    def is_persistent(self) -> bool:
        """Returns True if using persistent git storage."""
        return DULWICH_AVAILABLE

    def init_thread_repo(self, thread_id: str) -> None:
        """Initialize repo for a thread."""
        self._store.init_thread_repo(thread_id)

    def commit_step(
        self,
        thread_id: str,
        step_commit: StepCommit,
        parent_sha: Optional[str] = None
    ) -> str:
        """Create a commit. Returns SHA hex string."""
        if DULWICH_AVAILABLE and parent_sha:
            parent_bytes = bytes.fromhex(parent_sha)
            return self._store.commit_step(thread_id, step_commit, parent_bytes)
        return self._store.commit_step(thread_id, step_commit, parent_sha)

    def create_branch(self, thread_id: str, branch_name: str, commit_sha: Optional[str] = None) -> None:
        """Create a branch."""
        self._store.create_branch(thread_id, branch_name, commit_sha)

    def checkout(self, thread_id: str, target: str) -> str:
        """Checkout branch or commit."""
        return self._store.checkout(thread_id, target)

    def get_log(self, thread_id: str, max_count: int = 50) -> List[Dict]:
        """Get commit history."""
        return self._store.get_log(thread_id, max_count)

    def diff(self, thread_id: str, sha1: str, sha2: str) -> Dict[str, Any]:
        """Diff two commits."""
        return self._store.diff(thread_id, sha1, sha2)

    def list_branches(self, thread_id: str) -> List[str]:
        """List all branches."""
        return self._store.list_branches(thread_id)

    def delete_branch(self, thread_id: str, branch_name: str) -> None:
        """Delete a branch."""
        self._store.delete_branch(thread_id, branch_name)

    def reset(self, thread_id: str, target_sha: str, mode: str = "soft") -> None:
        """Reset to commit."""
        self._store.reset(thread_id, target_sha, mode)

    def cherry_pick(self, thread_id: str, commit_sha: str) -> str:
        """Cherry-pick a commit."""
        return self._store.cherry_pick(thread_id, commit_sha)

    def get_reflog(self, thread_id: str, limit: int = 50) -> List[Dict]:
        """Get reflog entries."""
        return self._store.get_reflog(thread_id, limit)

    def get_merge_base(self, thread_id: str, branch1: str, branch2: str) -> Optional[str]:
        """Find common ancestor."""
        return self._store.get_merge_base(thread_id, branch1, branch2)

    def merge(self, thread_id: str, source_branch: str, strategy: str = "best") -> str:
        """Merge branch."""
        return self._store.merge(thread_id, source_branch, strategy)


def create_step_commit(
    thread_id: str,
    step_index: int,
    query: str,
    reasoning_mode: str,
    confidence: float,
    response: Optional[str] = None,
    tool_selected: Optional[str] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    parent_id: Optional[str] = None,
    epistemic_confidence: float = 0.0,
    mrf_strategy: Optional[str] = None,
    context_window: Optional[List[str]] = None
) -> StepCommit:
    """
    Factory function to create a StepCommit with automatic ID generation.
    """
    timestamp = time.time()

    # Build state for hashing
    state = {
        "step_index": step_index,
        "query": query,
        "reasoning_mode": reasoning_mode,
        "confidence": confidence,
        "response": response,
        "tool_selected": tool_selected,
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
        "timestamp": timestamp
    }
    state_json = json.dumps(state, sort_keys=True)
    state_hash = hashlib.sha256(state_json.encode()).hexdigest()

    # Generate commit ID
    parent_str = parent_id or ""
    commit_input = f"{parent_str}{state_hash}"
    commit_id = hashlib.sha256(commit_input.encode()).hexdigest()[:40]

    # Generate message
    if response:
        response_preview = response[:50] + "..." if len(response) > 50 else response
        message = f"Step {step_index}: {reasoning_mode} - {response_preview}"
    else:
        message = f"Step {step_index}: {reasoning_mode}"

    return StepCommit(
        commit_id=commit_id,
        parent_id=parent_id,
        thread_id=thread_id,
        step_index=step_index,
        timestamp=timestamp,
        author="agent",
        message=message,
        query=query,
        reasoning_mode=reasoning_mode,
        confidence=confidence,
        epistemic_confidence=epistemic_confidence,
        tokens_used=tokens_used,
        cost_usd=cost_usd,
        context_window=tuple(context_window or []),
        mrf_strategy=mrf_strategy,
        response=response,
        tool_selected=tool_selected,
        state_hash=state_hash
    )
