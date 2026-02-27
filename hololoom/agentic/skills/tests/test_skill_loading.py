#!/usr/bin/env python3
"""
Test Skill Loading and YAML Validation
======================================
Tests for SkillRegistry YAML loading, validation, and template parsing.

Test Coverage:
- YAML syntax validation
- Required field validation
- Parameter schema validation
- Template metadata parsing
- Registry initialization
- Skill discovery
"""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any

from hololoom.agentic.skills import (
    SkillRegistry,
    SkillTemplate,
    SkillParameter,
    get_registry,
    list_available_skills
)


class TestYAMLValidation:
    """Test YAML file syntax and structure validation."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Get path to skills directory."""
        # Skills are in same directory as __init__.py, not in yaml/ subdirectory
        return Path(__file__).parent.parent

    def test_all_yamls_valid_syntax(self, skills_dir: Path):
        """Test that all YAML files have valid syntax."""
        yaml_files = list(skills_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found in skills directory"

        errors = []
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                errors.append(f"{yaml_file.name}: {e}")

        assert not errors, f"YAML syntax errors found:\n" + "\n".join(errors)

    def test_all_yamls_have_required_fields(self, skills_dir: Path):
        """Test that all YAML files have required top-level fields."""
        required_fields = {"name", "version", "description", "metadata", "system_prompt", "user_prompt_template", "parameters"}
        yaml_files = list(skills_dir.glob("*.yaml"))

        errors = []
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            missing = required_fields - set(data.keys())
            if missing:
                errors.append(f"{yaml_file.name}: Missing fields {missing}")

        assert not errors, f"Missing required fields:\n" + "\n".join(errors)

    def test_all_yamls_have_valid_categories(self, skills_dir: Path):
        """Test that all skills have valid categories."""
        valid_categories = {
            "architecture", "development", "education",
            "optimization", "security", "database", "general"
        }
        yaml_files = list(skills_dir.glob("*.yaml"))

        errors = []
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Category is in metadata, not top-level
            metadata = data.get("metadata", {})
            category = metadata.get("category", "").lower()
            if category not in valid_categories:
                errors.append(
                    f"{yaml_file.name}: Invalid category '{category}'. "
                    f"Valid: {valid_categories}"
                )

        assert not errors, f"Invalid categories:\n" + "\n".join(errors)

    def test_all_templates_have_prompt(self, skills_dir: Path):
        """Test that all skill templates have a user_prompt_template field."""
        yaml_files = list(skills_dir.glob("*.yaml"))

        errors = []
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            # Check for user_prompt_template, not template.prompt
            if "user_prompt_template" not in data or not data["user_prompt_template"]:
                errors.append(f"{yaml_file.name}: Missing 'user_prompt_template' field")

        assert not errors, f"Missing prompts:\n" + "\n".join(errors)


class TestParameterValidation:
    """Test skill parameter schema validation."""

    @pytest.fixture
    def skills_dir(self) -> Path:
        """Get path to skills directory."""
        # Skills are in same directory as __init__.py, not in yaml/ subdirectory
        return Path(__file__).parent.parent

    def test_parameters_have_required_fields(self, skills_dir: Path):
        """Test that all parameters have name and type (description is optional)."""
        yaml_files = list(skills_dir.glob("*.yaml"))

        errors = []
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            parameters = data.get("parameters", [])
            for i, param in enumerate(parameters):
                missing = []
                if "name" not in param:
                    missing.append("name")
                if "type" not in param:
                    missing.append("type")
                # description is optional

                if missing:
                    errors.append(
                        f"{yaml_file.name}: Parameter {i} missing {missing}"
                    )

        assert not errors, f"Invalid parameters:\n" + "\n".join(errors)

    def test_parameters_have_valid_types(self, skills_dir: Path):
        """Test that all parameters have valid type annotations."""
        valid_types = {"string", "code", "number", "boolean", "array", "object"}
        yaml_files = list(skills_dir.glob("*.yaml"))

        errors = []
        for yaml_file in yaml_files:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            parameters = data.get("parameters", [])
            for param in parameters:
                param_type = param.get("type", "").lower()
                if param_type and param_type not in valid_types:
                    errors.append(
                        f"{yaml_file.name}: Invalid type '{param_type}' in "
                        f"parameter '{param.get('name')}'. Valid: {valid_types}"
                    )

        assert not errors, f"Invalid parameter types:\n" + "\n".join(errors)


class TestSkillRegistry:
    """Test SkillRegistry initialization and loading."""

    @pytest.mark.asyncio
    async def test_registry_initializes(self):
        """Test that registry can be created and loads skills."""
        registry = SkillRegistry()
        assert registry is not None

        await registry.load_all_skills()
        assert len(registry.skills) > 0, "No skills loaded"

    @pytest.mark.asyncio
    async def test_registry_loads_skills(self):
        """Test that registry loads all YAML skills."""
        registry = SkillRegistry()
        await registry.load_all_skills()

        skills = registry.list_skills()

        assert len(skills) > 0, "No skills loaded"
        assert "code-reviewer" in skills, "Missing code-reviewer skill"

    @pytest.mark.asyncio
    async def test_registry_get_skill_template(self):
        """Test getting a skill template by name."""
        registry = SkillRegistry()
        await registry.load_all_skills()

        template = registry.get_skill("code-reviewer")

        assert template is not None, "Could not get code-reviewer template"
        assert template.name == "code-reviewer"
        assert template.metadata.category == "development"
        assert template.user_prompt_template, "Template missing user_prompt_template"

    @pytest.mark.asyncio
    async def test_registry_get_nonexistent_skill(self):
        """Test that getting a nonexistent skill returns None."""
        registry = SkillRegistry()
        await registry.load_all_skills()

        template = registry.get_skill("nonexistent_skill_12345")

        assert template is None, "Should return None for nonexistent skill"

    @pytest.mark.asyncio
    async def test_get_global_registry(self):
        """Test that get_registry() returns global singleton."""
        registry1 = await get_registry()
        registry2 = await get_registry()

        # Should be same instance (singleton)
        assert registry1 is registry2




class TestSkillDiscovery:
    """Test skill discovery and listing."""

    @pytest.mark.asyncio
    async def test_list_all_skills(self):
        """Test listing all available skills."""
        skills_by_category = await list_available_skills()

        # Check structure
        assert isinstance(skills_by_category, dict)

        # Check categories exist
        expected_categories = {"architecture", "development", "education", "optimization", "security", "database"}
        actual_categories = set(skills_by_category.keys())

        # At least some expected categories should exist
        assert len(actual_categories & expected_categories) > 0, \
            f"No expected categories found. Got: {actual_categories}"

    @pytest.mark.asyncio
    async def test_list_skills_by_category(self):
        """Test listing skills in a specific category."""
        registry = SkillRegistry()
        await registry.load_all_skills()

        skills_by_category = registry.list_skills_by_category()

        assert isinstance(skills_by_category, dict)
        assert "development" in skills_by_category, "No development category found"
        assert len(skills_by_category["development"]) > 0, "No development skills found"
        assert "code-reviewer" in skills_by_category["development"]

    @pytest.mark.asyncio
    async def test_count_total_skills(self):
        """Test counting total number of skills."""
        skills_by_category = await list_available_skills()

        total = sum(len(skill_list) for skill_list in skills_by_category.values())

        # We know there are 13 skills from the demo
        assert total >= 13, f"Expected at least 13 skills, got {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
