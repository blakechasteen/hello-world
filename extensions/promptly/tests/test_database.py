"""Tests for PromptlyDB - SQLite storage layer."""
import json
import pytest
from promptly.core.database import PromptlyDB, Prompt, PromptVersion, Chain, Skill


class TestDatabaseConnection:
    """Test database initialization and schema."""

    def test_connect_creates_tables(self, tmp_db):
        cursor = tmp_db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        expected = {"prompts", "prompt_versions", "branches", "evaluations",
                    "chains", "skills", "skill_files", "config"}
        assert expected.issubset(tables)

    def test_main_branch_created_on_connect(self, tmp_db):
        branches = tmp_db.list_branches()
        assert "main" in branches

    def test_default_branch_is_main(self, tmp_db):
        assert tmp_db.get_current_branch() == "main"


class TestPromptCRUD:
    """Test prompt create, read, update, delete."""

    def test_add_prompt(self, tmp_db):
        version, commit_hash = tmp_db.add_prompt("greeting", "Hello, {{name}}!")
        assert version == 1
        assert len(commit_hash) == 64  # SHA-256 hex digest

    def test_get_prompt(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello, {{name}}!")
        prompt = tmp_db.get_prompt("greeting")
        assert prompt is not None
        assert prompt.name == "greeting"
        assert prompt.content == "Hello, {{name}}!"
        assert prompt.current_version == 1
        assert prompt.branch == "main"

    def test_get_nonexistent_prompt(self, tmp_db):
        assert tmp_db.get_prompt("nonexistent") is None

    def test_update_prompt_increments_version(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello, {{name}}!")
        version, _ = tmp_db.add_prompt("greeting", "Hey {{name}}!")
        assert version == 2

        prompt = tmp_db.get_prompt("greeting")
        assert prompt.content == "Hey {{name}}!"
        assert prompt.current_version == 2

    def test_add_prompt_with_metadata(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello!", metadata={"tags": ["test"]})
        prompt = tmp_db.get_prompt("greeting")
        assert prompt.metadata == {"tags": ["test"]}

    def test_list_prompts(self, tmp_db):
        tmp_db.add_prompt("alpha", "Content A")
        tmp_db.add_prompt("beta", "Content B")
        tmp_db.add_prompt("gamma", "Content C")
        prompts = tmp_db.list_prompts()
        names = [p.name for p in prompts]
        assert len(names) == 3
        assert "alpha" in names
        assert "beta" in names
        assert "gamma" in names

    def test_list_prompts_empty(self, tmp_db):
        assert tmp_db.list_prompts() == []

    def test_delete_prompt(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello!")
        assert tmp_db.delete_prompt("greeting") is True
        assert tmp_db.get_prompt("greeting") is None

    def test_delete_nonexistent_prompt(self, tmp_db):
        assert tmp_db.delete_prompt("nonexistent") is False

    def test_delete_removes_versions(self, tmp_db):
        tmp_db.add_prompt("greeting", "v1")
        tmp_db.add_prompt("greeting", "v2")
        tmp_db.delete_prompt("greeting")
        history = tmp_db.get_prompt_history("greeting")
        assert history == []


class TestPromptVersioning:
    """Test version history and commit hashes."""

    def test_version_history(self, tmp_db):
        tmp_db.add_prompt("greeting", "v1")
        tmp_db.add_prompt("greeting", "v2")
        tmp_db.add_prompt("greeting", "v3")

        history = tmp_db.get_prompt_history("greeting")
        assert len(history) == 3
        assert history[0].version == 3  # newest first
        assert history[0].content == "v3"
        assert history[2].version == 1
        assert history[2].content == "v1"

    def test_each_version_has_unique_hash(self, tmp_db):
        _, hash1 = tmp_db.add_prompt("greeting", "v1")
        _, hash2 = tmp_db.add_prompt("greeting", "v2")
        assert hash1 != hash2

    def test_commit_hash_is_deterministic(self, tmp_db):
        _, hash1 = tmp_db.add_prompt("test", "content")
        tmp_db.delete_prompt("test")
        _, hash2 = tmp_db.add_prompt("test", "content")
        assert hash1 == hash2

    def test_get_prompt_by_version(self, tmp_db):
        tmp_db.add_prompt("greeting", "v1")
        tmp_db.add_prompt("greeting", "v2")
        prompt = tmp_db.get_prompt("greeting", version=1)
        assert prompt.content == "v1"
        # current_version reflects the prompt's latest version, not the requested one
        assert prompt.current_version == 2

    def test_get_prompt_by_commit_hash(self, tmp_db):
        _, hash1 = tmp_db.add_prompt("greeting", "v1")
        tmp_db.add_prompt("greeting", "v2")
        prompt = tmp_db.get_prompt("greeting", commit_hash=hash1)
        assert prompt.content == "v1"

    def test_history_of_nonexistent_prompt(self, tmp_db):
        assert tmp_db.get_prompt_history("nonexistent") == []


class TestPromptRendering:
    """Test variable substitution."""

    def test_render_simple(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello, {{name}}!")
        prompt = tmp_db.get_prompt("greeting")
        rendered = prompt.render(name="World")
        assert rendered == "Hello, World!"

    def test_render_multiple_variables(self, tmp_db):
        tmp_db.add_prompt("intro", "Hi {{name}}, welcome to {{project}}!")
        prompt = tmp_db.get_prompt("intro")
        rendered = prompt.render(name="Alice", project="Promptly")
        assert rendered == "Hi Alice, welcome to Promptly!"

    def test_render_missing_variable_leaves_placeholder(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello, {{name}}!")
        prompt = tmp_db.get_prompt("greeting")
        rendered = prompt.render()
        assert rendered == "Hello, {{name}}!"

    def test_render_no_variables(self, tmp_db):
        tmp_db.add_prompt("static", "No variables here.")
        prompt = tmp_db.get_prompt("static")
        rendered = prompt.render()
        assert rendered == "No variables here."


class TestBranching:
    """Test branch operations."""

    def test_create_branch(self, tmp_db):
        assert tmp_db.create_branch("experiments") is True
        assert "experiments" in tmp_db.list_branches()

    def test_create_duplicate_branch(self, tmp_db):
        tmp_db.create_branch("experiments")
        assert tmp_db.create_branch("experiments") is False

    def test_checkout_branch(self, tmp_db):
        tmp_db.create_branch("experiments")
        assert tmp_db.checkout_branch("experiments") is True
        assert tmp_db.get_current_branch() == "experiments"

    def test_checkout_nonexistent_branch(self, tmp_db):
        assert tmp_db.checkout_branch("nonexistent") is False

    def test_checkout_main(self, tmp_db):
        tmp_db.create_branch("experiments")
        tmp_db.checkout_branch("experiments")
        tmp_db.checkout_branch("main")
        assert tmp_db.get_current_branch() == "main"

    def test_list_branches(self, tmp_db):
        tmp_db.create_branch("dev")
        tmp_db.create_branch("staging")
        branches = tmp_db.list_branches()
        assert "main" in branches
        assert "dev" in branches
        assert "staging" in branches


class TestChains:
    """Test chain operations."""

    def test_create_chain(self, tmp_db):
        steps = [
            {"prompt": "research", "output_key": "findings"},
            {"prompt": "summarize", "output_key": "summary"},
        ]
        chain_id = tmp_db.create_chain("research-flow", steps)
        assert chain_id > 0

    def test_get_chain(self, tmp_db):
        steps = [{"prompt": "step1", "output_key": "out1"}]
        tmp_db.create_chain("my-chain", steps, metadata={"desc": "test"})

        chain = tmp_db.get_chain("my-chain")
        assert chain is not None
        assert chain.name == "my-chain"
        assert len(chain.steps) == 1
        assert chain.steps[0]["prompt"] == "step1"
        assert chain.metadata == {"desc": "test"}

    def test_get_nonexistent_chain(self, tmp_db):
        assert tmp_db.get_chain("nonexistent") is None

    def test_list_chains(self, tmp_db):
        tmp_db.create_chain("chain-a", [{"prompt": "a"}])
        tmp_db.create_chain("chain-b", [{"prompt": "b"}])
        chains = tmp_db.list_chains()
        names = [c.name for c in chains]
        assert "chain-a" in names
        assert "chain-b" in names


class TestSkills:
    """Test skill operations."""

    def test_create_skill(self, tmp_db):
        skill_id = tmp_db.create_skill(
            name="analyzer",
            description="Analyze code",
            prompt_template="Review this: {{code}}"
        )
        assert skill_id > 0

    def test_create_skill_with_files(self, tmp_db):
        files = [
            {"filename": "main.py", "content": "print('hello')", "file_type": "python"},
            {"filename": "config.yaml", "content": "key: value", "file_type": "yaml"},
        ]
        tmp_db.create_skill(
            name="analyzer",
            description="Analyze code",
            prompt_template="Review: {{code}}",
            files=files
        )

        skill = tmp_db.get_skill("analyzer")
        assert skill is not None
        assert len(skill.files) == 2
        assert skill.files[0]["filename"] == "main.py"
        assert skill.files[1]["content"] == "key: value"

    def test_get_skill(self, tmp_db):
        tmp_db.create_skill("test-skill", "A test", "Template: {{input}}")
        skill = tmp_db.get_skill("test-skill")
        assert skill.name == "test-skill"
        assert skill.description == "A test"
        assert skill.prompt_template == "Template: {{input}}"

    def test_get_nonexistent_skill(self, tmp_db):
        assert tmp_db.get_skill("nonexistent") is None

    def test_list_skills(self, tmp_db):
        tmp_db.create_skill("skill-a", "Desc A", "Template A")
        tmp_db.create_skill("skill-b", "Desc B", "Template B")
        skills = tmp_db.list_skills()
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names


class TestEvaluations:
    """Test evaluation operations."""

    def test_add_evaluation(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello!")
        eval_id = tmp_db.add_evaluation(
            prompt_name="greeting",
            version=1,
            score=8.5,
            criteria="quality",
            feedback="Good prompt",
            model="test-model"
        )
        assert eval_id > 0

    def test_add_evaluation_nonexistent_prompt(self, tmp_db):
        with pytest.raises(ValueError, match="not found"):
            tmp_db.add_evaluation("nonexistent", 1, 5.0, "quality", "test", "model")

    def test_get_evaluations(self, tmp_db):
        tmp_db.add_prompt("greeting", "Hello!")
        tmp_db.add_evaluation("greeting", 1, 8.0, "quality", "Good", "model-a")
        tmp_db.add_evaluation("greeting", 1, 7.0, "clarity", "Clear", "model-b")

        evals = tmp_db.get_evaluations("greeting")
        assert len(evals) == 2

    def test_get_evaluations_nonexistent_prompt(self, tmp_db):
        assert tmp_db.get_evaluations("nonexistent") == []


class TestConfig:
    """Test config operations."""

    def test_set_and_get_config(self, tmp_db):
        tmp_db.set_config("theme", "dark")
        assert tmp_db.get_config("theme") == "dark"

    def test_get_config_default(self, tmp_db):
        assert tmp_db.get_config("missing", "fallback") == "fallback"

    def test_config_json_values(self, tmp_db):
        tmp_db.set_config("settings", {"color": "blue", "size": 12})
        result = tmp_db.get_config("settings")
        assert result == {"color": "blue", "size": 12}

    def test_config_overwrite(self, tmp_db):
        tmp_db.set_config("key", "old")
        tmp_db.set_config("key", "new")
        assert tmp_db.get_config("key") == "new"
