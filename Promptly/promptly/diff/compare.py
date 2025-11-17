#!/usr/bin/env python3
"""
Comparison Tools - Compare prompt versions, branches, and evaluation results
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from .engine import DiffEngine, DiffResult, DiffStats, DiffLevel


@dataclass
class VersionComparison:
    """Comparison between two prompt versions"""
    name: str
    old_version: int
    new_version: int
    old_content: str
    new_content: str
    diff_result: DiffResult
    metadata_diff: Dict[str, Any]
    old_commit: str
    new_commit: str
    time_delta: Optional[str] = None

    def summary(self) -> str:
        """Generate summary of comparison"""
        return (
            f"Prompt: {self.name}\n"
            f"Versions: v{self.old_version} -> v{self.new_version}\n"
            f"Commits: {self.old_commit} -> {self.new_commit}\n"
            f"{self.diff_result.stats}\n"
            f"Time delta: {self.time_delta or 'unknown'}"
        )


@dataclass
class BranchComparison:
    """Comparison between branches"""
    branch_old: str
    branch_new: str
    prompts_only_in_old: List[str]
    prompts_only_in_new: List[str]
    prompts_modified: List[str]
    prompts_identical: List[str]
    diff_results: Dict[str, DiffResult]

    def summary(self) -> str:
        """Generate summary of branch comparison"""
        lines = [
            f"Branch comparison: {self.branch_old} <-> {self.branch_new}",
            f"",
            f"Only in {self.branch_old}: {len(self.prompts_only_in_old)}",
            f"Only in {self.branch_new}: {len(self.prompts_only_in_new)}",
            f"Modified: {len(self.prompts_modified)}",
            f"Identical: {len(self.prompts_identical)}",
        ]

        if self.prompts_only_in_old:
            lines.append(f"\nPrompts only in {self.branch_old}:")
            for name in self.prompts_only_in_old[:5]:
                lines.append(f"  - {name}")
            if len(self.prompts_only_in_old) > 5:
                lines.append(f"  ... and {len(self.prompts_only_in_old) - 5} more")

        if self.prompts_only_in_new:
            lines.append(f"\nPrompts only in {self.branch_new}:")
            for name in self.prompts_only_in_new[:5]:
                lines.append(f"  + {name}")
            if len(self.prompts_only_in_new) > 5:
                lines.append(f"  ... and {len(self.prompts_only_in_new) - 5} more")

        if self.prompts_modified:
            lines.append(f"\nModified prompts:")
            for name in self.prompts_modified[:10]:
                if name in self.diff_results:
                    stats = self.diff_results[name].stats
                    lines.append(f"  ~ {name} ({stats.similarity:.0%} similar)")
            if len(self.prompts_modified) > 10:
                lines.append(f"  ... and {len(self.prompts_modified) - 10} more")

        return '\n'.join(lines)


@dataclass
class EvaluationComparison:
    """Comparison of evaluation results"""
    prompt_name: str
    old_score: float
    new_score: float
    score_delta: float
    old_results: List[Dict]
    new_results: List[Dict]
    improved_tests: List[int]
    regressed_tests: List[int]
    unchanged_tests: List[int]

    def summary(self) -> str:
        """Generate summary of evaluation comparison"""
        delta_sign = "+" if self.score_delta >= 0 else ""
        improvement = "improvement" if self.score_delta >= 0 else "regression"

        return (
            f"Evaluation comparison for: {self.prompt_name}\n"
            f"Score: {self.old_score:.3f} -> {self.new_score:.3f} "
            f"({delta_sign}{self.score_delta:.3f}, {improvement})\n"
            f"Improved tests: {len(self.improved_tests)}\n"
            f"Regressed tests: {len(self.regressed_tests)}\n"
            f"Unchanged tests: {len(self.unchanged_tests)}"
        )


class ComparisonEngine:
    """Engine for comparing prompts, versions, and branches"""

    def __init__(self, promptly_instance=None):
        """
        Initialize comparison engine

        Args:
            promptly_instance: Instance of Promptly class for accessing prompts
        """
        self.promptly = promptly_instance
        self.diff_engine = DiffEngine()

    def compare_versions(
        self,
        name: str,
        version1: int,
        version2: int,
        level: DiffLevel = DiffLevel.LINE
    ) -> VersionComparison:
        """
        Compare two versions of a prompt

        Args:
            name: Prompt name
            version1: First version number
            version2: Second version number
            level: Diff granularity level

        Returns:
            VersionComparison with diff results
        """
        if not self.promptly:
            raise ValueError("Promptly instance required for version comparison")

        # Get both versions
        prompt1 = self.promptly.get(name, version=version1)
        prompt2 = self.promptly.get(name, version=version2)

        if not prompt1:
            raise ValueError(f"Version {version1} not found for prompt '{name}'")
        if not prompt2:
            raise ValueError(f"Version {version2} not found for prompt '{name}'")

        # Compute diff
        diff_result = self.diff_engine.diff(
            prompt1['content'],
            prompt2['content'],
            level=level
        )

        # Compare metadata
        metadata_diff = self._diff_metadata(
            prompt1.get('metadata', {}),
            prompt2.get('metadata', {})
        )

        # Calculate time delta
        try:
            time1 = datetime.fromisoformat(prompt1['created_at'])
            time2 = datetime.fromisoformat(prompt2['created_at'])
            delta = time2 - time1
            time_delta = self._format_timedelta(delta)
        except:
            time_delta = None

        return VersionComparison(
            name=name,
            old_version=version1,
            new_version=version2,
            old_content=prompt1['content'],
            new_content=prompt2['content'],
            diff_result=diff_result,
            metadata_diff=metadata_diff,
            old_commit=prompt1['commit_hash'],
            new_commit=prompt2['commit_hash'],
            time_delta=time_delta
        )

    def compare_commits(
        self,
        name: str,
        commit1: str,
        commit2: str,
        level: DiffLevel = DiffLevel.LINE
    ) -> VersionComparison:
        """
        Compare two commits of a prompt

        Args:
            name: Prompt name
            commit1: First commit hash
            commit2: Second commit hash
            level: Diff granularity level

        Returns:
            VersionComparison with diff results
        """
        if not self.promptly:
            raise ValueError("Promptly instance required for commit comparison")

        # Get both commits
        prompt1 = self.promptly.get(name, commit_hash=commit1)
        prompt2 = self.promptly.get(name, commit_hash=commit2)

        if not prompt1:
            raise ValueError(f"Commit {commit1} not found for prompt '{name}'")
        if not prompt2:
            raise ValueError(f"Commit {commit2} not found for prompt '{name}'")

        # Compute diff
        diff_result = self.diff_engine.diff(
            prompt1['content'],
            prompt2['content'],
            level=level
        )

        # Compare metadata
        metadata_diff = self._diff_metadata(
            prompt1.get('metadata', {}),
            prompt2.get('metadata', {})
        )

        return VersionComparison(
            name=name,
            old_version=prompt1['version'],
            new_version=prompt2['version'],
            old_content=prompt1['content'],
            new_content=prompt2['content'],
            diff_result=diff_result,
            metadata_diff=metadata_diff,
            old_commit=commit1,
            new_commit=commit2
        )

    def compare_branches(
        self,
        branch1: str,
        branch2: str,
        level: DiffLevel = DiffLevel.LINE
    ) -> BranchComparison:
        """
        Compare two branches

        Args:
            branch1: First branch name
            branch2: Second branch name
            level: Diff granularity level

        Returns:
            BranchComparison with detailed comparison
        """
        if not self.promptly:
            raise ValueError("Promptly instance required for branch comparison")

        # Get prompts from both branches
        prompts1 = {p['name']: p for p in self.promptly.list_prompts(branch1)}
        prompts2 = {p['name']: p for p in self.promptly.list_prompts(branch2)}

        names1 = set(prompts1.keys())
        names2 = set(prompts2.keys())

        # Categorize prompts
        only_in_1 = sorted(names1 - names2)
        only_in_2 = sorted(names2 - names1)
        common = sorted(names1 & names2)

        # Compare common prompts
        modified = []
        identical = []
        diff_results = {}

        for name in common:
            # Fetch full prompt content
            p1 = self.promptly.get(name, version=prompts1[name]['version'])
            p2 = self.promptly.get(name, version=prompts2[name]['version'])

            if p1['content'] != p2['content']:
                modified.append(name)
                # Compute diff for modified prompts
                diff_results[name] = self.diff_engine.diff(
                    p1['content'],
                    p2['content'],
                    level=level
                )
            else:
                identical.append(name)

        return BranchComparison(
            branch_old=branch1,
            branch_new=branch2,
            prompts_only_in_old=only_in_1,
            prompts_only_in_new=only_in_2,
            prompts_modified=modified,
            prompts_identical=identical,
            diff_results=diff_results
        )

    def compare_evaluations(
        self,
        prompt_name: str,
        old_results: List[Dict],
        new_results: List[Dict]
    ) -> EvaluationComparison:
        """
        Compare evaluation results between two prompt versions

        Args:
            prompt_name: Name of the prompt
            old_results: Evaluation results from old version
            new_results: Evaluation results from new version

        Returns:
            EvaluationComparison with score analysis
        """
        # Calculate average scores
        old_scores = [r.get('score', 0) for r in old_results if r.get('score') is not None]
        new_scores = [r.get('score', 0) for r in new_results if r.get('score') is not None]

        old_avg = sum(old_scores) / len(old_scores) if old_scores else 0.0
        new_avg = sum(new_scores) / len(new_scores) if new_scores else 0.0
        delta = new_avg - old_avg

        # Compare individual test results
        improved = []
        regressed = []
        unchanged = []

        min_len = min(len(old_results), len(new_results))
        for i in range(min_len):
            old_score = old_results[i].get('score', 0)
            new_score = new_results[i].get('score', 0)

            if old_score is not None and new_score is not None:
                if new_score > old_score + 0.01:  # Improvement threshold
                    improved.append(i)
                elif new_score < old_score - 0.01:  # Regression threshold
                    regressed.append(i)
                else:
                    unchanged.append(i)

        return EvaluationComparison(
            prompt_name=prompt_name,
            old_score=old_avg,
            new_score=new_avg,
            score_delta=delta,
            old_results=old_results,
            new_results=new_results,
            improved_tests=improved,
            regressed_tests=regressed,
            unchanged_tests=unchanged
        )

    def compare_prompts(
        self,
        prompt1: Dict,
        prompt2: Dict,
        level: DiffLevel = DiffLevel.LINE
    ) -> DiffResult:
        """
        Compare two prompt dictionaries directly

        Args:
            prompt1: First prompt dict with 'content' key
            prompt2: Second prompt dict with 'content' key
            level: Diff granularity level

        Returns:
            DiffResult
        """
        return self.diff_engine.diff(
            prompt1.get('content', ''),
            prompt2.get('content', ''),
            level=level
        )

    def _diff_metadata(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """Compare metadata dictionaries"""
        diff = {
            'added': {},
            'removed': {},
            'changed': {},
            'unchanged': {}
        }

        keys1 = set(meta1.keys())
        keys2 = set(meta2.keys())

        # Added keys
        for key in keys2 - keys1:
            diff['added'][key] = meta2[key]

        # Removed keys
        for key in keys1 - keys2:
            diff['removed'][key] = meta1[key]

        # Common keys
        for key in keys1 & keys2:
            if meta1[key] != meta2[key]:
                diff['changed'][key] = {
                    'old': meta1[key],
                    'new': meta2[key]
                }
            else:
                diff['unchanged'][key] = meta1[key]

        return diff

    def _format_timedelta(self, delta) -> str:
        """Format timedelta for display"""
        seconds = delta.total_seconds()
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds / 60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds / 3600)} hours"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''}"


def create_comparison_engine(promptly_instance=None) -> ComparisonEngine:
    """Factory function to create comparison engine"""
    return ComparisonEngine(promptly_instance)
