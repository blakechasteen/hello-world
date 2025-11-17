#!/usr/bin/env python3
"""
Merge Tool - Three-way merge with automatic and interactive conflict resolution
"""

import difflib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MergeStrategy(Enum):
    """Strategy for resolving conflicts"""
    OURS = "ours"           # Keep our changes, discard theirs
    THEIRS = "theirs"       # Keep their changes, discard ours
    UNION = "union"         # Keep both (concatenate)
    MANUAL = "manual"       # Require manual resolution
    AUTO = "auto"           # Attempt automatic resolution


class ConflictResolution(Enum):
    """How a conflict was resolved"""
    AUTO = "auto"
    MANUAL = "manual"
    OURS = "ours"
    THEIRS = "theirs"
    UNION = "union"
    UNRESOLVED = "unresolved"


@dataclass
class MergeConflict:
    """Represents a merge conflict"""
    start_line: int
    end_line: int
    base_content: str
    ours_content: str
    theirs_content: str
    resolved: bool = False
    resolution: Optional[ConflictResolution] = None
    merged_content: Optional[str] = None

    def __repr__(self):
        status = "RESOLVED" if self.resolved else "UNRESOLVED"
        return f"MergeConflict(lines {self.start_line}-{self.end_line}, {status})"

    def to_marker_format(self) -> str:
        """Format conflict with git-style markers"""
        lines = [
            "<<<<<<< OURS",
            self.ours_content.rstrip(),
            "=======",
            self.theirs_content.rstrip(),
            ">>>>>>> THEIRS"
        ]

        if self.base_content:
            lines.insert(1, "||||||| BASE")
            lines.insert(2, self.base_content.rstrip())

        return '\n'.join(lines)


@dataclass
class MergeResult:
    """Result of a merge operation"""
    merged_content: str
    conflicts: List[MergeConflict]
    strategy: MergeStrategy
    success: bool
    stats: Dict[str, int]

    def has_conflicts(self) -> bool:
        """Check if merge has unresolved conflicts"""
        return any(not c.resolved for c in self.conflicts)

    def __repr__(self):
        status = "SUCCESS" if self.success else "CONFLICTS"
        conflict_count = len([c for c in self.conflicts if not c.resolved])
        return f"MergeResult({status}, {conflict_count} unresolved conflicts)"


class ThreeWayMerge:
    """
    Three-way merge implementation
    Merges 'ours' and 'theirs' using 'base' as common ancestor
    """

    @staticmethod
    def merge(
        base: str,
        ours: str,
        theirs: str,
        strategy: MergeStrategy = MergeStrategy.AUTO
    ) -> MergeResult:
        """
        Perform three-way merge

        Args:
            base: Common ancestor content
            ours: Our version content
            theirs: Their version content
            strategy: Conflict resolution strategy

        Returns:
            MergeResult with merged content and conflicts
        """
        base_lines = base.splitlines(keepends=True)
        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        # Find changes from base
        ours_changes = ThreeWayMerge._get_changes(base_lines, ours_lines)
        theirs_changes = ThreeWayMerge._get_changes(base_lines, theirs_lines)

        # Merge changes
        merged, conflicts = ThreeWayMerge._merge_changes(
            base_lines,
            ours_changes,
            theirs_changes,
            strategy
        )

        # Apply strategy to conflicts
        for conflict in conflicts:
            if not conflict.resolved:
                ThreeWayMerge._apply_strategy(conflict, strategy)

        # Generate merged content
        merged_content = ''.join(merged)

        # Calculate stats
        stats = {
            'total_conflicts': len(conflicts),
            'resolved_conflicts': len([c for c in conflicts if c.resolved]),
            'unresolved_conflicts': len([c for c in conflicts if not c.resolved]),
            'lines_merged': len(merged)
        }

        success = not any(not c.resolved for c in conflicts)

        return MergeResult(
            merged_content=merged_content,
            conflicts=conflicts,
            strategy=strategy,
            success=success,
            stats=stats
        )

    @staticmethod
    def _get_changes(base: List[str], modified: List[str]) -> List[Tuple[str, int, int, int, int]]:
        """Get changes between base and modified version"""
        matcher = difflib.SequenceMatcher(None, base, modified)
        return list(matcher.get_opcodes())

    @staticmethod
    def _merge_changes(
        base: List[str],
        ours_changes: List[Tuple],
        theirs_changes: List[Tuple],
        strategy: MergeStrategy
    ) -> Tuple[List[str], List[MergeConflict]]:
        """Merge two sets of changes"""
        merged = []
        conflicts = []
        base_pos = 0

        # Create change maps for easier lookup
        ours_map = {(i1, i2): (tag, j1, j2) for tag, i1, i2, j1, j2 in ours_changes}
        theirs_map = {(i1, i2): (tag, j1, j2) for tag, i1, i2, j1, j2 in theirs_changes}

        # Process all base positions
        while base_pos < len(base):
            # Find changes at current position
            ours_change = ThreeWayMerge._find_change_at(base_pos, ours_changes)
            theirs_change = ThreeWayMerge._find_change_at(base_pos, theirs_changes)

            if ours_change is None and theirs_change is None:
                # No changes - keep base
                merged.append(base[base_pos])
                base_pos += 1

            elif ours_change is not None and theirs_change is None:
                # Only ours changed - accept ours
                tag, i1, i2, j1, j2 = ours_change
                if tag == 'delete':
                    # Skip deleted lines
                    base_pos = i2
                elif tag == 'replace' or tag == 'insert':
                    # Add replacement/insertion
                    ours_lines = ThreeWayMerge._get_lines_from_change(ours_change, base)
                    merged.extend(ours_lines)
                    base_pos = i2

            elif ours_change is None and theirs_change is not None:
                # Only theirs changed - accept theirs
                tag, i1, i2, j1, j2 = theirs_change
                if tag == 'delete':
                    # Skip deleted lines
                    base_pos = i2
                elif tag == 'replace' or tag == 'insert':
                    # Add replacement/insertion
                    theirs_lines = ThreeWayMerge._get_lines_from_change(theirs_change, base)
                    merged.extend(theirs_lines)
                    base_pos = i2

            else:
                # Both changed - potential conflict
                if ThreeWayMerge._changes_equal(ours_change, theirs_change):
                    # Same change - no conflict
                    ours_lines = ThreeWayMerge._get_lines_from_change(ours_change, base)
                    merged.extend(ours_lines)
                    base_pos = ours_change[2]
                else:
                    # Conflict!
                    conflict = ThreeWayMerge._create_conflict(
                        base_pos,
                        ours_change,
                        theirs_change,
                        base
                    )
                    conflicts.append(conflict)

                    # Add conflict marker to merged content
                    merged.append(conflict.to_marker_format() + '\n')

                    # Advance past conflict
                    base_pos = max(ours_change[2], theirs_change[2])

        return merged, conflicts

    @staticmethod
    def _find_change_at(pos: int, changes: List[Tuple]) -> Optional[Tuple]:
        """Find change that affects position"""
        for change in changes:
            tag, i1, i2, j1, j2 = change
            if i1 <= pos < i2 or (tag == 'insert' and i1 == pos):
                return change
        return None

    @staticmethod
    def _get_lines_from_change(change: Tuple, base: List[str]) -> List[str]:
        """Extract lines from a change operation"""
        # This is a simplified version - in practice, would need access to modified text
        tag, i1, i2, j1, j2 = change
        if tag == 'delete':
            return []
        elif tag == 'equal':
            return base[i1:i2]
        else:
            # For replace/insert, would need the modified lines
            # Placeholder - real implementation would get from modified text
            return base[i1:i2]

    @staticmethod
    def _changes_equal(change1: Tuple, change2: Tuple) -> bool:
        """Check if two changes are identical"""
        # Simplified - real implementation would compare actual content
        return change1[0] == change2[0] and change1[1:3] == change2[1:3]

    @staticmethod
    def _create_conflict(
        base_pos: int,
        ours_change: Tuple,
        theirs_change: Tuple,
        base: List[str]
    ) -> MergeConflict:
        """Create conflict from two differing changes"""
        ours_tag, ours_i1, ours_i2, ours_j1, ours_j2 = ours_change
        theirs_tag, theirs_i1, theirs_i2, theirs_j1, theirs_j2 = theirs_change

        base_content = ''.join(base[ours_i1:ours_i2])

        # Get content (simplified - would need actual modified texts)
        ours_content = f"[{ours_tag} operation]"
        theirs_content = f"[{theirs_tag} operation]"

        return MergeConflict(
            start_line=base_pos,
            end_line=max(ours_i2, theirs_i2),
            base_content=base_content,
            ours_content=ours_content,
            theirs_content=theirs_content
        )

    @staticmethod
    def _apply_strategy(conflict: MergeConflict, strategy: MergeStrategy):
        """Apply resolution strategy to conflict"""
        if strategy == MergeStrategy.OURS:
            conflict.merged_content = conflict.ours_content
            conflict.resolution = ConflictResolution.OURS
            conflict.resolved = True

        elif strategy == MergeStrategy.THEIRS:
            conflict.merged_content = conflict.theirs_content
            conflict.resolution = ConflictResolution.THEIRS
            conflict.resolved = True

        elif strategy == MergeStrategy.UNION:
            conflict.merged_content = conflict.ours_content + "\n" + conflict.theirs_content
            conflict.resolution = ConflictResolution.UNION
            conflict.resolved = True

        elif strategy == MergeStrategy.AUTO:
            # Try automatic resolution heuristics
            if conflict.ours_content.strip() == "":
                # Ours deleted, theirs modified - keep theirs
                conflict.merged_content = conflict.theirs_content
                conflict.resolution = ConflictResolution.AUTO
                conflict.resolved = True
            elif conflict.theirs_content.strip() == "":
                # Theirs deleted, ours modified - keep ours
                conflict.merged_content = conflict.ours_content
                conflict.resolution = ConflictResolution.AUTO
                conflict.resolved = True
            # Otherwise leave unresolved

        # MANUAL strategy leaves conflict unresolved


class MergeTool:
    """High-level merge tool for Promptly prompts"""

    def __init__(self, promptly_instance=None):
        """
        Initialize merge tool

        Args:
            promptly_instance: Promptly instance for accessing prompts
        """
        self.promptly = promptly_instance

    def merge_branches(
        self,
        source_branch: str,
        target_branch: str,
        strategy: MergeStrategy = MergeStrategy.AUTO
    ) -> Dict[str, MergeResult]:
        """
        Merge source branch into target branch

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Conflict resolution strategy

        Returns:
            Dict mapping prompt names to merge results
        """
        if not self.promptly:
            raise ValueError("Promptly instance required for branch merge")

        # Get prompts from both branches
        source_prompts = {p['name']: p for p in self.promptly.list_prompts(source_branch)}
        target_prompts = {p['name']: p for p in self.promptly.list_prompts(target_branch)}

        merge_results = {}

        # Find common ancestor (simplified - would need proper base tracking)
        # For now, assume main branch is common ancestor
        base_branch = 'main'

        # Merge each prompt
        for name in set(source_prompts.keys()) | set(target_prompts.keys()):
            # Get versions
            base = self.promptly.get(name, version=1)  # Simplified
            ours = target_prompts.get(name)
            theirs = source_prompts.get(name)

            if ours and theirs:
                # Both exist - three-way merge
                base_content = base['content'] if base else ""
                ours_full = self.promptly.get(name, version=ours['version'])
                theirs_full = self.promptly.get(name, version=theirs['version'])

                result = ThreeWayMerge.merge(
                    base_content,
                    ours_full['content'],
                    theirs_full['content'],
                    strategy
                )
                merge_results[name] = result

            elif theirs and not ours:
                # Only in source - add to target
                merge_results[name] = MergeResult(
                    merged_content=theirs['content'],
                    conflicts=[],
                    strategy=strategy,
                    success=True,
                    stats={'added': True}
                )

            # If only in target, no action needed

        return merge_results

    def merge_prompts(
        self,
        name: str,
        base_version: int,
        ours_version: int,
        theirs_version: int,
        strategy: MergeStrategy = MergeStrategy.AUTO
    ) -> MergeResult:
        """
        Merge three versions of a prompt

        Args:
            name: Prompt name
            base_version: Common ancestor version
            ours_version: Our version
            theirs_version: Their version
            strategy: Conflict resolution strategy

        Returns:
            MergeResult
        """
        if not self.promptly:
            raise ValueError("Promptly instance required for prompt merge")

        base = self.promptly.get(name, version=base_version)
        ours = self.promptly.get(name, version=ours_version)
        theirs = self.promptly.get(name, version=theirs_version)

        if not all([base, ours, theirs]):
            raise ValueError(f"Could not find all versions for prompt '{name}'")

        return ThreeWayMerge.merge(
            base['content'],
            ours['content'],
            theirs['content'],
            strategy
        )

    def resolve_conflict(
        self,
        conflict: MergeConflict,
        resolution: str
    ) -> MergeConflict:
        """
        Manually resolve a conflict

        Args:
            conflict: Unresolved conflict
            resolution: Resolution content

        Returns:
            Resolved conflict
        """
        conflict.merged_content = resolution
        conflict.resolution = ConflictResolution.MANUAL
        conflict.resolved = True
        return conflict

    def list_conflicts(self, merge_result: MergeResult) -> List[MergeConflict]:
        """Get list of unresolved conflicts"""
        return [c for c in merge_result.conflicts if not c.resolved]

    def abort_merge(self) -> str:
        """Abort ongoing merge operation"""
        # In a real implementation, would restore state
        return "Merge aborted"


class InteractiveMerge:
    """Interactive conflict resolution"""

    def __init__(self, merge_result: MergeResult):
        self.merge_result = merge_result
        self.current_conflict_index = 0

    def next_conflict(self) -> Optional[MergeConflict]:
        """Get next unresolved conflict"""
        conflicts = [c for c in self.merge_result.conflicts if not c.resolved]
        if self.current_conflict_index < len(conflicts):
            conflict = conflicts[self.current_conflict_index]
            self.current_conflict_index += 1
            return conflict
        return None

    def resolve_current(self, choice: str) -> bool:
        """
        Resolve current conflict

        Args:
            choice: 'ours', 'theirs', 'union', or custom text

        Returns:
            True if resolved, False otherwise
        """
        conflict = self.merge_result.conflicts[self.current_conflict_index - 1]

        if choice == 'ours':
            conflict.merged_content = conflict.ours_content
            conflict.resolution = ConflictResolution.OURS
        elif choice == 'theirs':
            conflict.merged_content = conflict.theirs_content
            conflict.resolution = ConflictResolution.THEIRS
        elif choice == 'union':
            conflict.merged_content = conflict.ours_content + "\n" + conflict.theirs_content
            conflict.resolution = ConflictResolution.UNION
        else:
            # Custom resolution
            conflict.merged_content = choice
            conflict.resolution = ConflictResolution.MANUAL

        conflict.resolved = True
        return True

    def has_more_conflicts(self) -> bool:
        """Check if there are more unresolved conflicts"""
        unresolved = [c for c in self.merge_result.conflicts if not c.resolved]
        return self.current_conflict_index < len(unresolved)

    def get_progress(self) -> Tuple[int, int]:
        """Get (resolved, total) conflict counts"""
        resolved = len([c for c in self.merge_result.conflicts if c.resolved])
        total = len(self.merge_result.conflicts)
        return (resolved, total)


def create_merge_tool(promptly_instance=None) -> MergeTool:
    """Factory function to create merge tool"""
    return MergeTool(promptly_instance)
