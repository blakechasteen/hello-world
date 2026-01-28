"""Tests for Promptly CLI commands using Click's test runner."""
import os
import sys
import pytest
from click.testing import CliRunner

# Ensure package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promptly.cli.main import cli


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """CLI runner with isolated temp directories."""
    global_dir = tmp_path / "global"
    local_dir = tmp_path / "local"
    global_dir.mkdir()
    local_dir.mkdir()

    monkeypatch.setenv("PROMPTLY_GLOBAL_DIR", str(global_dir))
    monkeypatch.setenv("PROMPTLY_LOCAL_DIR", str(local_dir))

    # Import the actual module (not the function shadowed by __init__)
    import importlib
    cli_module = importlib.import_module("promptly.cli.main")

    def patched_get():
        from promptly.core.promptly import Promptly
        p = Promptly(global_dir=global_dir, local_dir=local_dir, auto_detect_local=False)
        p.connect()
        return p

    monkeypatch.setattr(cli_module, "get_promptly", patched_get)

    return CliRunner()


class TestAddCommand:
    """Test 'promptly add' command."""

    def test_add_inline(self, runner):
        result = runner.invoke(cli, ["add", "greeting", "Hello, {{name}}!"])
        assert result.exit_code == 0
        assert "greeting" in result.output
        assert "v1" in result.output

    def test_add_from_file(self, runner, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Content from file")
        result = runner.invoke(cli, ["add", "from-file", "-f", str(prompt_file)])
        assert result.exit_code == 0
        assert "from-file" in result.output

    def test_add_with_metadata(self, runner):
        result = runner.invoke(cli, [
            "add", "tagged", "Some content",
            "-m", "category=test", "-m", "priority=high"
        ])
        assert result.exit_code == 0
        assert "tagged" in result.output

    def test_add_global_scope(self, runner):
        result = runner.invoke(cli, ["add", "global-p", "Content", "--scope", "global"])
        assert result.exit_code == 0


class TestGetCommand:
    """Test 'promptly get' command."""

    def test_get_existing(self, runner):
        runner.invoke(cli, ["add", "greeting", "Hello!"])
        result = runner.invoke(cli, ["get", "greeting"])
        assert result.exit_code == 0
        assert "Hello!" in result.output

    def test_get_raw(self, runner):
        runner.invoke(cli, ["add", "greeting", "Hello!"])
        result = runner.invoke(cli, ["get", "greeting", "--raw"])
        assert result.exit_code == 0
        assert result.output.strip() == "Hello!"

    def test_get_nonexistent(self, runner):
        result = runner.invoke(cli, ["get", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestListCommand:
    """Test 'promptly list' command."""

    def test_list_empty(self, runner):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_list_with_prompts(self, runner):
        runner.invoke(cli, ["add", "alpha", "Content A"])
        runner.invoke(cli, ["add", "beta", "Content B"])
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output


class TestLogCommand:
    """Test 'promptly log' command."""

    def test_log_shows_versions(self, runner):
        runner.invoke(cli, ["add", "greeting", "v1"])
        runner.invoke(cli, ["add", "greeting", "v2"])
        result = runner.invoke(cli, ["log", "greeting"])
        assert result.exit_code == 0


class TestDeleteCommand:
    """Test 'promptly delete' command."""

    def test_delete_existing(self, runner):
        runner.invoke(cli, ["add", "greeting", "Hello!"])
        result = runner.invoke(cli, ["delete", "greeting", "--yes"])
        assert result.exit_code == 0

    def test_delete_nonexistent(self, runner):
        result = runner.invoke(cli, ["delete", "nonexistent", "--yes"])
        assert result.exit_code != 0 or "not found" in result.output.lower()


class TestBranchCommands:
    """Test branching CLI commands."""

    def test_create_branch(self, runner):
        result = runner.invoke(cli, ["branch", "experiments"])
        assert result.exit_code == 0

    def test_checkout(self, runner):
        runner.invoke(cli, ["branch", "dev"])
        result = runner.invoke(cli, ["checkout", "dev"])
        assert result.exit_code == 0

    def test_list_branches(self, runner):
        runner.invoke(cli, ["branch", "dev"])
        result = runner.invoke(cli, ["branches"])
        assert result.exit_code == 0
        assert "main" in result.output


class TestExportImportCommands:
    """Test export/import CLI commands."""

    def test_export(self, runner):
        runner.invoke(cli, ["add", "greeting", "Hello, {{name}}!"])
        result = runner.invoke(cli, ["export", "greeting"])
        assert result.exit_code == 0
        assert "greeting" in result.output
        assert "Hello" in result.output

    def test_import_from_file(self, runner, tmp_path):
        # First export
        runner.invoke(cli, ["add", "original", "Original content"])
        export_result = runner.invoke(cli, ["export", "original"])

        # Write to file
        import_file = tmp_path / "prompt.yaml"
        import_file.write_text(export_result.output)

        # Import
        result = runner.invoke(cli, ["import", str(import_file)])
        assert result.exit_code == 0


class TestInfoCommand:
    """Test 'promptly info' command."""

    def test_info(self, runner):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "global" in result.output.lower() or "Global" in result.output


class TestVersionFlag:
    """Test version output."""

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output
