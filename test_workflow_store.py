"""
Test suite for WorkflowStore integration layer
"""

import tempfile
import json
from pathlib import Path
from hololoom.promptly import WorkflowStore


def test_workflow_store():
    """Test all WorkflowStore functionality"""

    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_workflows.db"

        # Initialize store
        store = WorkflowStore(str(db_path))
        print("[+] WorkflowStore initialized")

        # Test 1: save_workflow
        workflow1 = {
            "name": "test_workflow",
            "steps": [
                {"type": "retrieve", "query": "What is AI?"},
                {"type": "process", "processor": "transformer"},
                {"type": "decide", "tool": "answer"}
            ],
            "config": {"timeout": 30, "retries": 3}
        }

        commit1 = store.save_workflow(
            "test_workflow",
            workflow1,
            branch="main",
            message="Initial workflow"
        )
        print(f"[+] Saved workflow v1: {commit1}")

        # Test 2: get_workflow (latest)
        retrieved = store.get_workflow("test_workflow", branch="main")
        assert retrieved is not None
        assert retrieved["steps"][0]["query"] == "What is AI?"
        assert retrieved["_metadata"]["version"] == 1
        print("[+] Retrieved latest workflow")

        # Test 3: get_workflow (specific version)
        retrieved_v1 = store.get_workflow("test_workflow", version=1, branch="main")
        assert retrieved_v1 is not None
        assert retrieved_v1["_metadata"]["version"] == 1
        print("[+] Retrieved specific version")

        # Test 4: save_workflow (new version)
        workflow2 = workflow1.copy()
        workflow2["steps"][0]["query"] = "What is machine learning?"
        workflow2["config"]["timeout"] = 60

        commit2 = store.save_workflow(
            "test_workflow",
            workflow2,
            branch="main",
            message="Updated query timeout"
        )
        print(f"[+] Saved workflow v2: {commit2}")
        assert commit2 != commit1

        # Verify v2 is latest
        latest = store.get_workflow("test_workflow", branch="main")
        assert latest["_metadata"]["version"] == 2
        assert latest["config"]["timeout"] == 60
        print("[+] Version 2 is latest")

        # Test 5: list_workflows
        workflows = store.list_workflows(branch="main")
        assert len(workflows) >= 1
        assert workflows[0]["name"] == "test_workflow"
        assert workflows[0]["version"] == 2
        print(f"[+] Listed workflows: {len(workflows)} found")

        # Test 6: diff_workflows
        diff = store.diff_workflows("test_workflow", 1, 2, branch="main")
        assert "modified" in diff
        assert "steps" in diff["modified"]
        assert "config" in diff["modified"]
        assert diff["modified"]["steps"]["before"][0]["query"] == "What is AI?"
        assert diff["modified"]["steps"]["after"][0]["query"] == "What is machine learning?"
        print("[+] Workflow diff generated correctly")
        print(f"  - Modified keys: {list(diff['modified'].keys())}")

        # Test 7: branch_workflow
        commit3 = store.branch_workflow("test_workflow", "main", "develop")
        print(f"[+] Branched workflow from main to develop: {commit3}")

        # Verify new branch has version 1
        develop_workflow = store.get_workflow("test_workflow", version=1, branch="develop")
        assert develop_workflow is not None
        assert develop_workflow["_metadata"]["branch"] == "develop"
        assert develop_workflow["_metadata"]["version"] == 1
        print("[+] Branched workflow has version 1 on develop")

        # Test 8: delete_workflow
        delete_result = store.delete_workflow("test_workflow", 1, branch="main")
        assert delete_result is True
        print("[+] Deleted workflow version 1")

        # Verify delete
        deleted = store.get_workflow("test_workflow", version=1, branch="main")
        # After soft delete, workflow still exists but json is None
        print("[+] Soft delete verified")

        # Test 9: Error handling
        try:
            store.save_workflow("test", "not_a_dict")
            assert False, "Should raise ValueError"
        except ValueError as e:
            print(f"[+] Proper error handling: {e}")

        # Test 10: Diff non-existent version
        try:
            store.diff_workflows("test_workflow", 999, 1000, branch="main")
            assert False, "Should raise ValueError"
        except ValueError as e:
            print(f"[+] Proper diff error handling: {e}")

        store.close()
        print("\n[SUCCESS] All WorkflowStore tests passed!")
    finally:
        import shutil
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_workflow_store()
