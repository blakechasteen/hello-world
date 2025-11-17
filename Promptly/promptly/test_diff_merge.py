#!/usr/bin/env python3
"""
Test suite for diff and merge functionality
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from promptly import Promptly
from diff import (
    DiffEngine,
    DiffLevel,
    DiffType,
    ComparisonEngine,
    VersionComparison,
    BranchComparison,
    TerminalDiff,
    HTMLDiff,
    quick_diff
)
from merge import (
    MergeTool,
    ThreeWayMerge,
    MergeStrategy,
    InteractiveMerge,
    MergeConflict
)


class TestDiffEngine(unittest.TestCase):
    """Test diff engine functionality"""

    def setUp(self):
        self.engine = DiffEngine()

    def test_character_diff(self):
        """Test character-level diff"""
        old_text = "Hello World"
        new_text = "Hello Beautiful World"

        result = self.engine.diff(old_text, new_text, DiffLevel.CHARACTER)

        self.assertIsNotNone(result)
        self.assertEqual(result.level, DiffLevel.CHARACTER)
        self.assertTrue(len(result.chunks) > 0)

        # Should have an insert chunk
        insert_chunks = [c for c in result.chunks if c.type == DiffType.INSERT]
        self.assertTrue(len(insert_chunks) > 0)

    def test_word_diff(self):
        """Test word-level diff"""
        old_text = "The quick brown fox"
        new_text = "The fast brown fox"

        result = self.engine.diff(old_text, new_text, DiffLevel.WORD)

        self.assertEqual(result.level, DiffLevel.WORD)
        self.assertTrue(result.stats.changes > 0 or result.stats.additions > 0)

    def test_line_diff(self):
        """Test line-level diff"""
        old_text = "Line 1\nLine 2\nLine 3"
        new_text = "Line 1\nLine 2 modified\nLine 3\nLine 4"

        result = self.engine.diff(old_text, new_text, DiffLevel.LINE)

        self.assertEqual(result.level, DiffLevel.LINE)
        self.assertIsNotNone(result.unified_format)
        self.assertTrue(len(result.unified_format) > 0)

    def test_semantic_diff(self):
        """Test semantic diff with context"""
        old_text = "You must complete the task"
        new_text = "You should complete the task. For example, finish all steps."

        result = self.engine.diff(old_text, new_text, DiffLevel.SEMANTIC)

        self.assertEqual(result.level, DiffLevel.SEMANTIC)

        # Check for semantic annotations
        modified_chunks = [c for c in result.chunks if c.type != DiffType.EQUAL]
        self.assertTrue(any(c.context for c in modified_chunks))

    def test_diff_stats(self):
        """Test diff statistics calculation"""
        old_text = "Line 1\nLine 2"
        new_text = "Line 1\nLine 2 modified\nLine 3"

        result = self.engine.diff(old_text, new_text, DiffLevel.LINE)

        self.assertGreater(result.stats.additions, 0)
        self.assertGreaterEqual(result.stats.similarity, 0)
        self.assertLessEqual(result.stats.similarity, 1)

    def test_side_by_side(self):
        """Test side-by-side diff generation"""
        old_text = "Old line 1\nOld line 2"
        new_text = "New line 1\nNew line 2"

        output = self.engine.side_by_side(old_text, new_text, width=80)

        self.assertIn("OLD", output)
        self.assertIn("NEW", output)
        self.assertIn("Old line 1", output)
        self.assertIn("New line 1", output)

    def test_identical_texts(self):
        """Test diff of identical texts"""
        text = "Same content"

        result = self.engine.diff(text, text, DiffLevel.LINE)

        self.assertEqual(result.stats.similarity, 1.0)
        self.assertEqual(result.stats.additions, 0)
        self.assertEqual(result.stats.deletions, 0)

    def test_quick_diff(self):
        """Test quick diff convenience function"""
        old = "A"
        new = "B"

        result = quick_diff(old, new, level="char")

        self.assertIsNotNone(result)
        self.assertEqual(result.level, DiffLevel.CHARACTER)


class TestComparisonEngine(unittest.TestCase):
    """Test comparison engine for prompts and branches"""

    def setUp(self):
        # Create temporary directory for test repository
        self.test_dir = tempfile.mkdtemp()
        self.promptly = Promptly(self.test_dir)
        self.promptly.init()

        # Add test prompts
        self.promptly.add("test_prompt", "Version 1 content", {"tag": "v1"})
        self.promptly.add("test_prompt", "Version 2 content modified", {"tag": "v2"})
        self.promptly.add("test_prompt", "Version 3 content completely different", {"tag": "v3"})

        self.engine = ComparisonEngine(self.promptly)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_compare_versions(self):
        """Test comparing two versions of a prompt"""
        comparison = self.engine.compare_versions("test_prompt", 1, 2, DiffLevel.WORD)

        self.assertIsInstance(comparison, VersionComparison)
        self.assertEqual(comparison.name, "test_prompt")
        self.assertEqual(comparison.old_version, 1)
        self.assertEqual(comparison.new_version, 2)
        self.assertIsNotNone(comparison.diff_result)

    def test_compare_metadata(self):
        """Test metadata comparison"""
        comparison = self.engine.compare_versions("test_prompt", 1, 2)

        self.assertIn('changed', comparison.metadata_diff)
        # Tag changed from v1 to v2
        if 'tag' in comparison.metadata_diff['changed']:
            self.assertEqual(comparison.metadata_diff['changed']['tag']['old'], 'v1')
            self.assertEqual(comparison.metadata_diff['changed']['tag']['new'], 'v2')

    def test_compare_branches(self):
        """Test branch comparison"""
        # Create a new branch
        self.promptly.branch("feature", "main")
        self.promptly.checkout("feature")
        self.promptly.add("feature_prompt", "Feature content")

        comparison = self.engine.compare_branches("main", "feature")

        self.assertIsInstance(comparison, BranchComparison)
        self.assertEqual(comparison.branch_old, "main")
        self.assertEqual(comparison.branch_new, "feature")
        self.assertIn("feature_prompt", comparison.prompts_only_in_new)

    def test_compare_prompts_directly(self):
        """Test direct prompt comparison"""
        p1 = self.promptly.get("test_prompt", version=1)
        p2 = self.promptly.get("test_prompt", version=2)

        diff_result = self.engine.compare_prompts(p1, p2, DiffLevel.LINE)

        self.assertIsNotNone(diff_result)
        self.assertTrue(diff_result.stats.similarity < 1.0)

    def test_version_not_found(self):
        """Test error handling for non-existent version"""
        with self.assertRaises(ValueError):
            self.engine.compare_versions("test_prompt", 1, 99)


class TestVisualDiff(unittest.TestCase):
    """Test visual diff rendering"""

    def setUp(self):
        self.engine = DiffEngine()

    def test_terminal_render(self):
        """Test terminal color rendering"""
        old = "Line 1\nLine 2"
        new = "Line 1\nLine 2 modified"

        result = self.engine.diff(old, new, DiffLevel.LINE)
        output = TerminalDiff.render(result)

        self.assertIsNotNone(output)
        self.assertTrue(len(output) > 0)
        # Should contain ANSI color codes
        self.assertIn('\033[', output)

    def test_html_render(self):
        """Test HTML rendering"""
        old = "Old content"
        new = "New content"

        result = self.engine.diff(old, new, DiffLevel.LINE)
        html = HTMLDiff.render(result, title="Test Diff")

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Test Diff", html)
        self.assertIn("diff-container", html)

    def test_side_by_side_html(self):
        """Test side-by-side HTML rendering"""
        old = "Old"
        new = "New"

        html = HTMLDiff.render_side_by_side(old, new, "Version 1", "Version 2")

        self.assertIn("side-by-side", html)
        self.assertIn("Version 1", html)
        self.assertIn("Version 2", html)


class TestThreeWayMerge(unittest.TestCase):
    """Test three-way merge functionality"""

    def test_clean_merge(self):
        """Test merge with no conflicts"""
        base = "Line 1\nLine 2\nLine 3"
        ours = "Line 1 modified\nLine 2\nLine 3"
        theirs = "Line 1\nLine 2\nLine 3 modified"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.AUTO)

        self.assertTrue(result.success)
        self.assertEqual(len(result.conflicts), 0)

    def test_conflicting_merge(self):
        """Test merge with conflicts"""
        base = "Line 1\nLine 2\nLine 3"
        ours = "Line 1 our change\nLine 2\nLine 3"
        theirs = "Line 1 their change\nLine 2\nLine 3"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.MANUAL)

        # Should have conflicts
        self.assertFalse(result.success)
        self.assertGreater(len(result.conflicts), 0)

    def test_merge_strategy_ours(self):
        """Test OURS merge strategy"""
        base = "Base"
        ours = "Ours"
        theirs = "Theirs"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.OURS)

        # All conflicts should be resolved with OURS
        for conflict in result.conflicts:
            if conflict.resolved:
                self.assertIn(conflict.merged_content, [conflict.ours_content, ""])

    def test_merge_strategy_theirs(self):
        """Test THEIRS merge strategy"""
        base = "Base"
        ours = "Ours"
        theirs = "Theirs"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.THEIRS)

        # All conflicts should be resolved with THEIRS
        for conflict in result.conflicts:
            if conflict.resolved:
                self.assertIn(conflict.merged_content, [conflict.theirs_content, ""])

    def test_merge_strategy_union(self):
        """Test UNION merge strategy"""
        base = "Base"
        ours = "Ours"
        theirs = "Theirs"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.UNION)

        # Conflicts should be resolved with concatenation
        for conflict in result.conflicts:
            if conflict.resolved:
                # Union combines both
                self.assertIsNotNone(conflict.merged_content)

    def test_conflict_markers(self):
        """Test conflict marker generation"""
        conflict = MergeConflict(
            start_line=1,
            end_line=3,
            base_content="base content",
            ours_content="our content",
            theirs_content="their content"
        )

        markers = conflict.to_marker_format()

        self.assertIn("<<<<<<< OURS", markers)
        self.assertIn("=======", markers)
        self.assertIn(">>>>>>> THEIRS", markers)
        self.assertIn("our content", markers)
        self.assertIn("their content", markers)


class TestMergeTool(unittest.TestCase):
    """Test high-level merge tool"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.promptly = Promptly(self.test_dir)
        self.promptly.init()

        # Setup test data
        self.promptly.add("shared_prompt", "Base content")

        # Create feature branch
        self.promptly.branch("feature", "main")

        # Modify on main
        self.promptly.add("shared_prompt", "Main branch modification")
        self.promptly.add("main_only", "Main only content")

        # Modify on feature
        self.promptly.checkout("feature")
        self.promptly.add("shared_prompt", "Feature branch modification")
        self.promptly.add("feature_only", "Feature only content")

        self.merge_tool = MergeTool(self.promptly)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_merge_branches(self):
        """Test merging two branches"""
        self.promptly.checkout("main")

        results = self.merge_tool.merge_branches("feature", "main", MergeStrategy.AUTO)

        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)

    def test_merge_prompts(self):
        """Test merging specific prompt versions"""
        # Get version numbers
        p1 = self.promptly.get("shared_prompt", version=1)

        self.promptly.checkout("main")
        p2 = self.promptly.get("shared_prompt")

        self.promptly.checkout("feature")
        p3 = self.promptly.get("shared_prompt")

        result = self.merge_tool.merge_prompts(
            "shared_prompt",
            p1['version'],
            p2['version'],
            p3['version'],
            MergeStrategy.AUTO
        )

        self.assertIsNotNone(result)


class TestInteractiveMerge(unittest.TestCase):
    """Test interactive merge resolution"""

    def setUp(self):
        base = "Line 1\nLine 2"
        ours = "Line 1 ours\nLine 2"
        theirs = "Line 1 theirs\nLine 2"

        result = ThreeWayMerge.merge(base, ours, theirs, MergeStrategy.MANUAL)
        self.interactive = InteractiveMerge(result)

    def test_next_conflict(self):
        """Test getting next conflict"""
        conflict = self.interactive.next_conflict()
        # Should have at least one conflict
        self.assertIsNotNone(conflict)

    def test_resolve_conflict(self):
        """Test resolving conflicts interactively"""
        conflict = self.interactive.next_conflict()
        if conflict:
            success = self.interactive.resolve_current('ours')
            self.assertTrue(success)
            self.assertTrue(conflict.resolved)

    def test_progress_tracking(self):
        """Test conflict resolution progress"""
        resolved, total = self.interactive.get_progress()

        self.assertGreaterEqual(resolved, 0)
        self.assertGreaterEqual(total, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.promptly = Promptly(self.test_dir)
        self.promptly.init()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_diff_merge_workflow(self):
        """Test complete diff and merge workflow"""
        # Create initial prompt
        self.promptly.add("workflow_test", "Initial content")

        # Create branch and modify
        self.promptly.branch("experiment", "main")
        self.promptly.checkout("experiment")
        self.promptly.add("workflow_test", "Experimental changes")

        # Compare branches
        comparison_engine = ComparisonEngine(self.promptly)
        comparison = comparison_engine.compare_branches("main", "experiment")

        self.assertIsNotNone(comparison)
        self.assertIn("workflow_test", comparison.prompts_modified)

        # Diff specific prompt
        diff_result = comparison_engine.compare_versions("workflow_test", 1, 2)
        self.assertIsNotNone(diff_result)
        self.assertGreater(diff_result.diff_result.stats.additions, 0)

        # Merge back to main
        self.promptly.checkout("main")
        merge_tool = MergeTool(self.promptly)
        results = merge_tool.merge_branches("experiment", "main", MergeStrategy.AUTO)

        self.assertIsNotNone(results)

    def test_html_export_workflow(self):
        """Test exporting diffs to HTML"""
        self.promptly.add("html_test", "Version 1")
        self.promptly.add("html_test", "Version 2 with changes")

        # Generate comparison
        engine = ComparisonEngine(self.promptly)
        comparison = engine.compare_versions("html_test", 1, 2)

        # Export to HTML
        html = HTMLDiff.render_comparison(comparison)

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("html_test", html)

        # Save to file
        html_path = Path(self.test_dir) / "diff_report.html"
        with open(html_path, 'w') as f:
            f.write(html)

        self.assertTrue(html_path.exists())
        self.assertGreater(html_path.stat().st_size, 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDiffEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestComparisonEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualDiff))
    suite.addTests(loader.loadTestsFromTestCase(TestThreeWayMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestMergeTool))
    suite.addTests(loader.loadTestsFromTestCase(TestInteractiveMerge))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)
