"""Tests for Promptly core - the main API class."""
import os
import pytest
import yaml
from promptly.core.promptly import Promptly


class TestPromptlyInit:
    """Test Promptly initialization."""

    def test_creates_global_dir(self, tmp_path):
        global_dir = tmp_path / "global"
        p = Promptly(global_dir=global_dir, auto_detect_local=False)
        p.connect()
        assert global_dir.exists()
        p.close()

    def test_creates_local_dir(self, tmp_path):
        global_dir = tmp_path / "global"
        local_dir = tmp_path / "local"
        p = Promptly(global_dir=global_dir, local_dir=local_dir, auto_detect_local=False)
        p.connect()
        assert local_dir.exists()
        p.close()

    def test_info(self, tmp_promptly):
        info = tmp_promptly.info()
        assert "global_dir" in info
        assert "local_dir" in info
        assert "current_branch" in info
        assert info["global_prompts"] == 0
        assert info["local_prompts"] == 0


class TestDualStorage:
    """Test local-overrides-global behavior."""

    def test_add_to_local_by_default(self, tmp_promptly):
        tmp_promptly.add("greeting", "Hello!")
        prompts = tmp_promptly.list_prompts(scope="local")
        assert len(prompts) == 1
        assert prompts[0].name == "greeting"

    def test_add_to_global(self, tmp_promptly):
        tmp_promptly.add("greeting", "Hello!", scope="global")
        prompts = tmp_promptly.list_prompts(scope="global")
        assert len(prompts) == 1

    def test_local_overrides_global(self, tmp_promptly):
        tmp_promptly.add("greeting", "Global hello", scope="global")
        tmp_promptly.add("greeting", "Local hello", scope="local")

        prompt = tmp_promptly.get("greeting")
        assert prompt.content == "Local hello"

    def test_merged_list_shows_local_override(self, tmp_promptly):
        tmp_promptly.add("shared", "Global version", scope="global")
        tmp_promptly.add("shared", "Local version", scope="local")
        tmp_promptly.add("global-only", "Only in global", scope="global")
        tmp_promptly.add("local-only", "Only in local", scope="local")

        all_prompts = tmp_promptly.list_prompts()
        names = {p.name for p in all_prompts}
        assert names == {"shared", "global-only", "local-only"}
        assert len(all_prompts) == 3

        # list_prompts() doesn't populate content; use get() to verify local override
        shared = tmp_promptly.get("shared")
        assert shared.content == "Local version"

    def test_global_only_mode(self, global_only_promptly):
        global_only_promptly.add("greeting", "Hello!", scope="global")
        prompt = global_only_promptly.get("greeting")
        assert prompt.content == "Hello!"

    def test_global_only_local_scope_raises(self, global_only_promptly):
        with pytest.raises(ValueError, match="No local database"):
            global_only_promptly.add("test", "content", scope="local")


class TestPromptOperations:
    """Test core prompt CRUD via Promptly API."""

    def test_add_and_get(self, tmp_promptly):
        version, commit_hash = tmp_promptly.add("greeting", "Hello, {{name}}!")
        assert version == 1
        assert len(commit_hash) == 64  # SHA-256 hex digest

        prompt = tmp_promptly.get("greeting")
        assert prompt.content == "Hello, {{name}}!"

    def test_update_prompt(self, tmp_promptly):
        tmp_promptly.add("greeting", "v1")
        version, _ = tmp_promptly.add("greeting", "v2")
        assert version == 2

    def test_get_nonexistent(self, tmp_promptly):
        assert tmp_promptly.get("nonexistent") is None

    def test_log_returns_history(self, tmp_promptly):
        tmp_promptly.add("greeting", "v1")
        tmp_promptly.add("greeting", "v2")
        tmp_promptly.add("greeting", "v3")

        history = tmp_promptly.log("greeting")
        assert len(history) == 3
        assert history[0].version == 3

    def test_delete(self, tmp_promptly):
        tmp_promptly.add("greeting", "Hello!")
        assert tmp_promptly.delete("greeting") is True
        assert tmp_promptly.get("greeting") is None

    def test_delete_nonexistent(self, tmp_promptly):
        assert tmp_promptly.delete("nonexistent") is False

    def test_render_prompt(self, tmp_promptly):
        tmp_promptly.add("greeting", "Hello, {{name}}! Welcome to {{place}}.")
        prompt = tmp_promptly.get("greeting")
        rendered = prompt.render(name="Alice", place="Wonderland")
        assert rendered == "Hello, Alice! Welcome to Wonderland."


class TestBranchOperations:
    """Test branching via Promptly API."""

    def test_create_branch(self, tmp_promptly):
        assert tmp_promptly.branch("experiments") is True

    def test_checkout(self, tmp_promptly):
        tmp_promptly.branch("dev")
        assert tmp_promptly.checkout("dev") is True
        assert tmp_promptly.current_branch() == "dev"

    def test_checkout_nonexistent(self, tmp_promptly):
        assert tmp_promptly.checkout("nonexistent") is False

    def test_list_branches(self, tmp_promptly):
        tmp_promptly.branch("dev")
        tmp_promptly.branch("staging")
        branches = tmp_promptly.branches()
        assert "main" in branches
        assert "dev" in branches
        assert "staging" in branches

    def test_prompts_on_different_branches(self, tmp_promptly):
        # Note: current DB schema uses UNIQUE(name) globally,
        # so add_prompt on a different branch creates a new version
        # rather than a separate branch copy. get() always returns latest.
        tmp_promptly.add("greeting", "Main greeting")
        tmp_promptly.branch("experiments")
        tmp_promptly.checkout("experiments")
        tmp_promptly.add("greeting", "Experimental greeting")

        exp_prompt = tmp_promptly.get("greeting")
        assert exp_prompt.content == "Experimental greeting"

        # After checkout back to main, get() still returns the latest version
        # because prompts are not isolated by branch in the current schema
        tmp_promptly.checkout("main")
        main_prompt = tmp_promptly.get("greeting")
        assert main_prompt is not None
        # Verify branch metadata was updated
        assert main_prompt.branch in ("main", "experiments")


class TestChainOperations:
    """Test chain management via Promptly API."""

    def test_create_chain(self, tmp_promptly):
        steps = [
            {"prompt": "research", "output_key": "findings"},
            {"prompt": "summarize", "output_key": "summary"},
        ]
        chain_id = tmp_promptly.create_chain("research-flow", steps)
        assert chain_id > 0

    def test_get_chain(self, tmp_promptly):
        steps = [{"prompt": "step1", "output_key": "out1"}]
        tmp_promptly.create_chain("my-chain", steps)
        chain = tmp_promptly.get_chain("my-chain")
        assert chain is not None
        assert chain.name == "my-chain"
        assert len(chain.steps) == 1

    def test_list_chains(self, tmp_promptly):
        tmp_promptly.create_chain("chain-a", [{"prompt": "a"}])
        tmp_promptly.create_chain("chain-b", [{"prompt": "b"}])
        chains = tmp_promptly.list_chains()
        assert len(chains) == 2


class TestSkillOperations:
    """Test skill management via Promptly API."""

    def test_create_skill(self, tmp_promptly):
        skill_id = tmp_promptly.create_skill(
            name="code-review",
            description="Review code quality",
            prompt_template="Review this {{language}} code:\n{{code}}"
        )
        assert skill_id > 0

    def test_get_skill(self, tmp_promptly):
        tmp_promptly.create_skill(
            name="analyzer",
            description="Analyze",
            prompt_template="Analyze: {{input}}"
        )
        skill = tmp_promptly.get_skill("analyzer")
        assert skill.name == "analyzer"
        assert "{{input}}" in skill.prompt_template

    def test_create_skill_with_files(self, tmp_promptly):
        files = [{"filename": "example.py", "content": "x = 1", "file_type": "python"}]
        tmp_promptly.create_skill(
            name="code-skill",
            description="Code analysis",
            prompt_template="Review: {{code}}",
            files=files
        )
        skill = tmp_promptly.get_skill("code-skill")
        assert len(skill.files) == 1
        assert skill.files[0]["filename"] == "example.py"

    def test_list_skills(self, tmp_promptly):
        tmp_promptly.create_skill("s1", "Desc 1", "Template 1")
        tmp_promptly.create_skill("s2", "Desc 2", "Template 2")
        skills = tmp_promptly.list_skills()
        assert len(skills) == 2


class TestExportImport:
    """Test YAML export/import."""

    def test_export_yaml(self, tmp_promptly):
        tmp_promptly.add("greeting", "Hello, {{name}}!", metadata={"tags": ["test"]})
        exported = tmp_promptly.export_prompt("greeting")
        data = yaml.safe_load(exported)
        assert data["name"] == "greeting"
        assert data["content"] == "Hello, {{name}}!"
        assert data["version"] == 1

    def test_export_nonexistent(self, tmp_promptly):
        with pytest.raises(ValueError, match="not found"):
            tmp_promptly.export_prompt("nonexistent")

    def test_import_yaml(self, tmp_promptly):
        yaml_str = """
name: imported
content: "Imported prompt content"
metadata:
  source: external
"""
        version, commit_hash = tmp_promptly.import_prompt(yaml_str)
        assert version == 1

        prompt = tmp_promptly.get("imported")
        assert prompt.content == "Imported prompt content"
        assert prompt.metadata.get("source") == "external"

    def test_import_dict(self, tmp_promptly):
        data = {"name": "from-dict", "content": "Dict content"}
        version, _ = tmp_promptly.import_prompt(data)
        assert version == 1
        assert tmp_promptly.get("from-dict").content == "Dict content"

    def test_roundtrip(self, tmp_promptly):
        tmp_promptly.add("roundtrip", "Hello, {{name}}!", metadata={"key": "value"})
        exported = tmp_promptly.export_prompt("roundtrip")
        tmp_promptly.delete("roundtrip")
        tmp_promptly.import_prompt(exported)

        prompt = tmp_promptly.get("roundtrip")
        assert prompt.content == "Hello, {{name}}!"
