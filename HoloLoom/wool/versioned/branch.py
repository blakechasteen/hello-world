"""
Branch Management for Versioned Wool Storage
==============================================

Git-like branching and merging for versioned storage.

This module provides:
- Branch creation from any version
- Branch listing and HEAD tracking
- Merge operations (fast-forward and 3-way)
- Conflict detection and resolution strategies
- Branch lifecycle management

Branching Model:
- Each branch has a HEAD (latest version)
- Branches can diverge from common ancestors
- Merges create new versions with multiple parents
- Fast-forward merges when possible (no conflicts)

Author: Claude Code
Date: November 17, 2025 (Phase 8 Branch/Merge)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from enum import Enum

from HoloLoom.wool.versioned.reference import VersionedWoolReference


logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Merge strategies for conflict resolution."""
    OURS = "ours"           # Take our branch's version
    THEIRS = "theirs"       # Take their branch's version
    MANUAL = "manual"       # Require manual resolution
    LATEST = "latest"       # Take most recent version


@dataclass
class Branch:
    """
    Branch in version history.

    Attributes:
        name: Branch name (e.g., 'main', 'feature-x')
        head_version: Latest version ID on this branch
        created_at: Branch creation timestamp
        created_from: Version ID branch was created from
        author: Who created the branch
        description: Branch description/purpose
    """
    name: str
    head_version: int
    created_at: float = field(default_factory=time.time)
    created_from: Optional[int] = None
    author: str = "system"
    description: str = ""

    def __repr__(self) -> str:
        return f"Branch({self.name}, HEAD=v{self.head_version})"


@dataclass
class MergeConflict:
    """
    Conflict detected during merge.

    Attributes:
        file_id: File with conflict
        our_version: Our branch's version
        their_version: Their branch's version
        base_version: Common ancestor version
        description: Human-readable conflict description
    """
    file_id: str
    our_version: int
    their_version: int
    base_version: Optional[int]
    description: str

    def __repr__(self) -> str:
        return (
            f"MergeConflict({self.file_id[:12]}..., "
            f"ours=v{self.our_version}, theirs=v{self.their_version})"
        )


@dataclass
class MergeResult:
    """
    Result of merge operation.

    Attributes:
        success: Whether merge succeeded
        merged_version: New version created (if successful)
        conflicts: List of conflicts (if failed)
        strategy_used: Strategy used for resolution
        fast_forward: Whether merge was fast-forward
    """
    success: bool
    merged_version: Optional[int] = None
    conflicts: List[MergeConflict] = field(default_factory=list)
    strategy_used: Optional[MergeStrategy] = None
    fast_forward: bool = False

    def __repr__(self) -> str:
        if self.success:
            ff = " (fast-forward)" if self.fast_forward else ""
            return f"MergeResult(success=True, v{self.merged_version}{ff})"
        else:
            return f"MergeResult(success=False, conflicts={len(self.conflicts)})"


class BranchManager:
    """
    Manages branches for versioned storage.

    Features:
    - Create branches from any version
    - Track branch HEADs
    - Detect merge conflicts
    - Fast-forward and 3-way merges
    - Multiple merge strategies

    Usage:
        manager = BranchManager()

        # Create branch
        manager.create_branch("feature-x", from_version=10)

        # Update HEAD
        manager.update_head("feature-x", new_version=15)

        # Merge
        result = manager.merge(
            source="feature-x",
            target="main",
            strategy=MergeStrategy.OURS
        )
    """

    def __init__(self):
        """Initialize branch manager."""
        # Branch registry
        self.branches: Dict[str, Branch] = {}

        # Default branch
        self.default_branch = "main"
        self.branches[self.default_branch] = Branch(
            name=self.default_branch,
            head_version=0,
            created_at=time.time(),
            author="system",
            description="Default branch"
        )

        # Statistics
        self.stats = {
            'branches_created': 1,  # main branch
            'merges_attempted': 0,
            'merges_succeeded': 0,
            'fast_forward_merges': 0,
            'conflicts_detected': 0
        }

        logger.info("Initialized branch manager with 'main' branch")

    def create_branch(
        self,
        name: str,
        from_version: int,
        author: str = "system",
        description: str = ""
    ) -> Branch:
        """
        Create new branch from version.

        Args:
            name: Branch name
            from_version: Version to branch from
            author: Who created the branch
            description: Branch description

        Returns:
            Created Branch

        Raises:
            ValueError: If branch already exists
        """
        if name in self.branches:
            raise ValueError(f"Branch '{name}' already exists")

        branch = Branch(
            name=name,
            head_version=from_version,
            created_at=time.time(),
            created_from=from_version,
            author=author,
            description=description
        )

        self.branches[name] = branch
        self.stats['branches_created'] += 1

        logger.info(
            f"Created branch '{name}' from v{from_version} "
            f"(author: {author})"
        )

        return branch

    def delete_branch(self, name: str):
        """
        Delete branch.

        Args:
            name: Branch name

        Raises:
            ValueError: If deleting default branch or branch doesn't exist
        """
        if name == self.default_branch:
            raise ValueError(f"Cannot delete default branch '{self.default_branch}'")

        if name not in self.branches:
            raise ValueError(f"Branch '{name}' does not exist")

        del self.branches[name]
        logger.info(f"Deleted branch '{name}'")

    def get_branch(self, name: str) -> Branch:
        """
        Get branch by name.

        Args:
            name: Branch name

        Returns:
            Branch

        Raises:
            ValueError: If branch doesn't exist
        """
        if name not in self.branches:
            raise ValueError(f"Branch '{name}' does not exist")
        return self.branches[name]

    def list_branches(self) -> List[Branch]:
        """
        List all branches.

        Returns:
            List of branches (sorted by name)
        """
        return sorted(self.branches.values(), key=lambda b: b.name)

    def update_head(self, branch_name: str, new_version: int):
        """
        Update branch HEAD to new version.

        Args:
            branch_name: Branch to update
            new_version: New HEAD version

        Raises:
            ValueError: If branch doesn't exist
        """
        branch = self.get_branch(branch_name)
        old_version = branch.head_version
        branch.head_version = new_version

        logger.info(
            f"Updated '{branch_name}' HEAD: v{old_version} → v{new_version}"
        )

    def find_common_ancestor(
        self,
        version_a: int,
        version_b: int,
        get_lineage_func
    ) -> Optional[int]:
        """
        Find common ancestor of two versions.

        Args:
            version_a: First version
            version_b: Second version
            get_lineage_func: Function to get version lineage

        Returns:
            Common ancestor version ID, or None
        """
        # Get lineages (root → version)
        lineage_a = get_lineage_func(version_a)
        lineage_b = get_lineage_func(version_b)

        # Find last common version
        common_ancestor = None
        for va, vb in zip(lineage_a, lineage_b):
            if va == vb:
                common_ancestor = va
            else:
                break

        return common_ancestor

    def can_fast_forward(
        self,
        source_version: int,
        target_version: int,
        get_lineage_func
    ) -> bool:
        """
        Check if merge can be fast-forward.

        Fast-forward is possible when source is direct descendant of target.

        Args:
            source_version: Source branch HEAD
            target_version: Target branch HEAD
            get_lineage_func: Function to get version lineage

        Returns:
            True if fast-forward possible
        """
        # Get source lineage
        source_lineage = get_lineage_func(source_version)

        # Check if target is in source's lineage
        return target_version in source_lineage

    def detect_conflicts(
        self,
        source_version: int,
        target_version: int,
        base_version: Optional[int],
        get_version_func
    ) -> List[MergeConflict]:
        """
        Detect conflicts between two versions.

        Args:
            source_version: Source branch version
            target_version: Target branch version
            base_version: Common ancestor version
            get_version_func: Function to get version reference

        Returns:
            List of conflicts
        """
        conflicts = []

        # Get version references
        source_ref = get_version_func(source_version)
        target_ref = get_version_func(target_version)

        # If file IDs differ, we have a conflict
        if source_ref.file_id != target_ref.file_id:
            conflict = MergeConflict(
                file_id=source_ref.file_id,
                our_version=target_version,
                their_version=source_version,
                base_version=base_version,
                description=(
                    f"Content diverged: source (v{source_version}) and "
                    f"target (v{target_version}) have different content"
                )
            )
            conflicts.append(conflict)

        return conflicts

    def merge(
        self,
        source: str,
        target: str,
        strategy: MergeStrategy,
        get_lineage_func,
        get_version_func,
        create_merge_func
    ) -> MergeResult:
        """
        Merge source branch into target branch.

        Args:
            source: Source branch name
            target: Target branch name
            strategy: Merge strategy for conflicts
            get_lineage_func: Function to get version lineage
            get_version_func: Function to get version reference
            create_merge_func: Function to create merge version

        Returns:
            MergeResult
        """
        self.stats['merges_attempted'] += 1

        # Get branches
        source_branch = self.get_branch(source)
        target_branch = self.get_branch(target)

        source_version = source_branch.head_version
        target_version = target_branch.head_version

        logger.info(
            f"Merging '{source}' (v{source_version}) into "
            f"'{target}' (v{target_version})"
        )

        # Check for fast-forward
        if self.can_fast_forward(source_version, target_version, get_lineage_func):
            # Fast-forward merge: just move target HEAD to source HEAD
            self.update_head(target, source_version)
            self.stats['merges_succeeded'] += 1
            self.stats['fast_forward_merges'] += 1

            logger.info(f"Fast-forward merge: '{target}' → v{source_version}")

            return MergeResult(
                success=True,
                merged_version=source_version,
                strategy_used=strategy,
                fast_forward=True
            )

        # Find common ancestor
        base_version = self.find_common_ancestor(
            source_version,
            target_version,
            get_lineage_func
        )

        # Detect conflicts
        conflicts = self.detect_conflicts(
            source_version,
            target_version,
            base_version,
            get_version_func
        )

        if conflicts:
            self.stats['conflicts_detected'] += len(conflicts)

            # Check if strategy can auto-resolve
            if strategy == MergeStrategy.MANUAL:
                logger.warning(
                    f"Merge conflicts detected ({len(conflicts)}), "
                    f"manual resolution required"
                )
                return MergeResult(
                    success=False,
                    conflicts=conflicts,
                    strategy_used=strategy
                )

            # Auto-resolve based on strategy
            if strategy == MergeStrategy.OURS:
                resolved_version = target_version
            elif strategy == MergeStrategy.THEIRS:
                resolved_version = source_version
            elif strategy == MergeStrategy.LATEST:
                # Get timestamps and pick latest
                source_ref = get_version_func(source_version)
                target_ref = get_version_func(target_version)
                resolved_version = (
                    source_version if source_ref.timestamp > target_ref.timestamp
                    else target_version
                )
            else:
                logger.error(f"Unknown strategy: {strategy}")
                return MergeResult(success=False, conflicts=conflicts)

            logger.info(
                f"Auto-resolved {len(conflicts)} conflicts using "
                f"strategy '{strategy.value}' → v{resolved_version}"
            )

        else:
            # No conflicts: create merge version
            resolved_version = None

        # Create merge version
        merged_version = create_merge_func(
            source_version=source_version,
            target_version=target_version,
            resolved_version=resolved_version,
            strategy=strategy
        )

        # Update target branch HEAD
        self.update_head(target, merged_version)

        self.stats['merges_succeeded'] += 1

        logger.info(
            f"Merge succeeded: '{source}' → '{target}' "
            f"(v{merged_version})"
        )

        return MergeResult(
            success=True,
            merged_version=merged_version,
            conflicts=conflicts if conflicts else [],
            strategy_used=strategy,
            fast_forward=False
        )

    def get_stats(self) -> Dict:
        """Get branch statistics."""
        return {
            **self.stats,
            'total_branches': len(self.branches),
            'merge_success_rate': (
                self.stats['merges_succeeded'] / self.stats['merges_attempted']
                if self.stats['merges_attempted'] > 0 else 0.0
            ),
            'fast_forward_percentage': (
                self.stats['fast_forward_merges'] / self.stats['merges_succeeded']
                if self.stats['merges_succeeded'] > 0 else 0.0
            )
        }

    def __repr__(self) -> str:
        return (
            f"BranchManager("
            f"branches={len(self.branches)}, "
            f"merges={self.stats['merges_succeeded']})"
        )
