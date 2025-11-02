#!/usr/bin/env python3
"""
Promptly Diff & Merge Tools
============================
Compare prompt versions and merge branches with conflict resolution.
"""

import difflib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ChangeType(Enum):
    """Type of change in diff"""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class DiffLine:
    """Single line in a diff"""
    line_num_a: Optional[int]
    line_num_b: Optional[int]
    change_type: ChangeType
    content: str


@dataclass
class PromptDiff:
    """Diff between two prompts"""
    prompt_name: str
    version_a: int
    version_b: int
    changes: List[DiffLine]
    stats: Dict[str, int]

    def to_unified_diff(self) -> str:
        """Generate unified diff format"""
        lines = [
            f"--- {self.prompt_name} (v{self.version_a})",
            f"+++ {self.prompt_name} (v{self.version_b})",
            ""
        ]

        for change in self.changes:
            if change.change_type == ChangeType.REMOVED:
                lines.append(f"- {change.content}")
            elif change.change_type == ChangeType.ADDED:
                lines.append(f"+ {change.content}")
            elif change.change_type == ChangeType.MODIFIED:
                lines.append(f"! {change.content}")
            else:  # UNCHANGED
                lines.append(f"  {change.content}")

        return "\n".join(lines)

    def to_stats_summary(self) -> str:
        """Generate statistics summary"""
        return (
            f"Changes: {self.stats['modified']} modified, "
            f"{self.stats['added']} added, "
            f"{self.stats['removed']} removed, "
            f"{self.stats['unchanged']} unchanged"
        )


class ConflictType(Enum):
    """Type of merge conflict"""
    CONTENT = "content"
    METADATA = "metadata"
    BOTH = "both"


@dataclass
class MergeConflict:
    """A merge conflict"""
    prompt_name: str
    conflict_type: ConflictType
    base_content: str
    branch_a_content: str
    branch_b_content: str
    resolution: Optional[str] = None


@dataclass
class MergeResult:
    """Result of a merge operation"""
    success: bool
    merged_prompts: List[str]
    conflicts: List[MergeConflict]
    stats: Dict[str, int]

    def has_conflicts(self) -> bool:
        """Check if merge has unresolved conflicts"""
        return len(self.conflicts) > 0

    def to_report(self) -> str:
        """Generate merge report"""
        lines = [
            f"Merge Result: {'✓ Success' if self.success else '✗ Conflicts'}",
            f"Merged: {self.stats['merged']} prompts",
            ""
        ]

        if self.conflicts:
            lines.append(f"Conflicts: {len(self.conflicts)}")
            for conflict in self.conflicts:
                lines.append(f"  - {conflict.prompt_name} ({conflict.conflict_type.value})")

        return "\n".join(lines)


class DiffTool:
    """Diff comparison tool"""

    @staticmethod
    def diff_prompts(
        prompt_a: Dict[str, Any],
        prompt_b: Dict[str, Any]
    ) -> PromptDiff:
        """
        Compare two prompt versions.

        Args:
            prompt_a: First prompt data
            prompt_b: Second prompt data

        Returns:
            PromptDiff with detailed changes
        """
        content_a = prompt_a["content"].splitlines()
        content_b = prompt_b["content"].splitlines()

        # Use difflib for comparison
        diff = list(difflib.unified_diff(
            content_a,
            content_b,
            lineterm='',
            n=0  # No context lines
        ))

        changes = []
        stats = {
            "added": 0,
            "removed": 0,
            "modified": 0,
            "unchanged": 0
        }

        # Parse diff
        line_a = 0
        line_b = 0

        for line in diff[2:]:  # Skip headers
            if line.startswith('@@'):
                continue
            elif line.startswith('-'):
                changes.append(DiffLine(
                    line_num_a=line_a,
                    line_num_b=None,
                    change_type=ChangeType.REMOVED,
                    content=line[1:]
                ))
                stats["removed"] += 1
                line_a += 1
            elif line.startswith('+'):
                changes.append(DiffLine(
                    line_num_a=None,
                    line_num_b=line_b,
                    change_type=ChangeType.ADDED,
                    content=line[1:]
                ))
                stats["added"] += 1
                line_b += 1
            else:
                changes.append(DiffLine(
                    line_num_a=line_a,
                    line_num_b=line_b,
                    change_type=ChangeType.UNCHANGED,
                    content=line
                ))
                stats["unchanged"] += 1
                line_a += 1
                line_b += 1

        return PromptDiff(
            prompt_name=prompt_a["name"],
            version_a=prompt_a["version"],
            version_b=prompt_b["version"],
            changes=changes,
            stats=stats
        )

    @staticmethod
    def similarity_score(text_a: str, text_b: str) -> float:
        """
        Calculate similarity between two texts.

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()


class MergeTool:
    """Merge tool for branches"""

    def __init__(self, promptly_instance):
        """
        Initialize merge tool.

        Args:
            promptly_instance: Instance of Promptly
        """
        self.promptly = promptly_instance

    def merge_branches(
        self,
        source_branch: str,
        target_branch: str,
        auto_resolve: bool = False
    ) -> MergeResult:
        """
        Merge source branch into target branch.

        Args:
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            auto_resolve: Automatically resolve simple conflicts

        Returns:
            MergeResult with conflicts and stats
        """
        conflicts = []
        merged = []
        stats = {
            "merged": 0,
            "conflicts": 0,
            "auto_resolved": 0
        }

        # Get prompts from both branches
        source_prompts = self.promptly.list(branch=source_branch)
        target_prompts = self.promptly.list(branch=target_branch)

        # Create lookup
        target_lookup = {p["name"]: p for p in target_prompts}

        for source_prompt in source_prompts:
            name = source_prompt["name"]

            if name not in target_lookup:
                # New prompt in source - no conflict
                self._apply_merge(source_prompt, target_branch)
                merged.append(name)
                stats["merged"] += 1
                continue

            target_prompt = target_lookup[name]

            # Check for conflicts
            if source_prompt["content"] != target_prompt["content"]:
                conflict = MergeConflict(
                    prompt_name=name,
                    conflict_type=ConflictType.CONTENT,
                    base_content="",  # Would need base commit to determine
                    branch_a_content=target_prompt["content"],
                    branch_b_content=source_prompt["content"]
                )

                if auto_resolve:
                    # Simple auto-resolve: use source if newer
                    if source_prompt["version"] > target_prompt["version"]:
                        conflict.resolution = source_prompt["content"]
                        self._apply_merge(source_prompt, target_branch)
                        stats["auto_resolved"] += 1
                    else:
                        conflicts.append(conflict)
                        stats["conflicts"] += 1
                else:
                    conflicts.append(conflict)
                    stats["conflicts"] += 1
            else:
                # No conflict
                merged.append(name)
                stats["merged"] += 1

        success = len(conflicts) == 0

        return MergeResult(
            success=success,
            merged_prompts=merged,
            conflicts=conflicts,
            stats=stats
        )

    def _apply_merge(self, prompt: Dict[str, Any], target_branch: str):
        """Apply a merge (add prompt to target branch)"""
        # Switch to target branch
        current_branch = self.promptly._get_current_branch()
        self.promptly.checkout(target_branch)

        # Add prompt
        self.promptly.add(
            name=prompt["name"],
            content=prompt["content"],
            metadata=prompt.get("metadata", {})
        )

        # Switch back
        self.promptly.checkout(current_branch)

    def resolve_conflict(
        self,
        conflict: MergeConflict,
        resolution: str
    ) -> MergeConflict:
        """
        Resolve a merge conflict manually.

        Args:
            conflict: MergeConflict to resolve
            resolution: Resolved content

        Returns:
            Updated conflict with resolution
        """
        conflict.resolution = resolution
        return conflict

    def three_way_merge(
        self,
        base_content: str,
        branch_a_content: str,
        branch_b_content: str
    ) -> Tuple[str, bool]:
        """
        Perform three-way merge.

        Args:
            base_content: Common ancestor content
            branch_a_content: Content from branch A
            branch_b_content: Content from branch B

        Returns:
            Tuple of (merged_content, has_conflicts)
        """
        # Simple three-way merge
        if branch_a_content == branch_b_content:
            # Both branches identical
            return branch_a_content, False

        if branch_a_content == base_content:
            # Only branch B changed
            return branch_b_content, False

        if branch_b_content == base_content:
            # Only branch A changed
            return branch_a_content, False

        # Both changed - conflict
        merged = f"""<<<<<<< Branch A
{branch_a_content}
=======
{branch_b_content}
>>>>>>> Branch B
"""
        return merged, True


# ============================================================================
# Convenience Functions
# ============================================================================

def diff_versions(
    promptly_instance,
    prompt_name: str,
    version_a: int,
    version_b: int
) -> str:
    """Quick diff between two versions"""
    prompt_a = promptly_instance.get(prompt_name, version=version_a)
    prompt_b = promptly_instance.get(prompt_name, version=version_b)

    diff = DiffTool.diff_prompts(prompt_a, prompt_b)
    return diff.to_unified_diff()


def quick_merge(
    promptly_instance,
    source: str,
    target: str,
    auto_resolve: bool = True
) -> MergeResult:
    """Quick branch merge"""
    tool = MergeTool(promptly_instance)
    return tool.merge_branches(source, target, auto_resolve=auto_resolve)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Diff & Merge Tools")
    print("\nExample - Diff:")
    print("""
from promptly import Promptly
from diff_merge import diff_versions

p = Promptly()

# Compare two versions
diff = diff_versions(p, "summarizer", version_a=1, version_b=2)
print(diff)
""")

    print("\nExample - Merge:")
    print("""
from promptly import Promptly
from diff_merge import quick_merge

p = Promptly()

# Merge experimental into main
result = quick_merge(p, source="experimental", target="main")

if result.has_conflicts():
    print(f"Conflicts: {len(result.conflicts)}")
    for conflict in result.conflicts:
        print(f"  - {conflict.prompt_name}")
else:
    print(f"✓ Merged {result.stats['merged']} prompts")
""")
