"""
Promptly - Main API Class

Provides git-like prompt versioning with dual storage:
- Global: ~/.promptly/ (user home, shared across projects)
- Local: .promptly/ (project-specific, overrides global)

Local prompts with the same name override global prompts.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field

from promptly.core.database import PromptlyDB, Prompt, Chain, Skill, PromptVersion


@dataclass
class PromptlyConfig:
    """Configuration for Promptly instance."""
    global_dir: Path = field(default_factory=lambda: Path.home() / ".promptly")
    local_dir: Optional[Path] = None
    auto_detect_local: bool = True
    llm_provider: Optional[str] = None  # None = auto-detect
    llm_model: Optional[str] = None


class Promptly:
    """
    Main Promptly API with dual storage support.

    Storage hierarchy:
    1. Local (.promptly/) - Project-specific prompts
    2. Global (~/.promptly/) - User-wide prompts

    Local prompts override global prompts with the same name.

    Usage:
        # Auto-detect project root
        p = Promptly()

        # Explicit paths
        p = Promptly(local_dir=Path("./my_project/.promptly"))

        # Add a prompt (stored locally by default)
        p.add("greeting", "Hello, {{name}}!")

        # Get a prompt (checks local first, then global)
        prompt = p.get("greeting")

        # List all prompts (merged view)
        prompts = p.list_prompts()
    """

    def __init__(
        self,
        config: PromptlyConfig = None,
        global_dir: Path = None,
        local_dir: Path = None,
        auto_detect_local: bool = True
    ):
        """
        Initialize Promptly.

        Args:
            config: PromptlyConfig object (overrides other args)
            global_dir: Path to global ~/.promptly/ directory
            local_dir: Path to project .promptly/ directory
            auto_detect_local: Auto-detect project root and create .promptly/
        """
        if config:
            self.config = config
        else:
            self.config = PromptlyConfig(
                global_dir=global_dir or Path.home() / ".promptly",
                local_dir=local_dir,
                auto_detect_local=auto_detect_local
            )

        # Initialize global database
        self.global_dir = self.config.global_dir
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.global_db = PromptlyDB(self.global_dir / "prompts.db", is_global=True)

        # Initialize local database (if applicable)
        self.local_dir = self._resolve_local_dir()
        self.local_db: Optional[PromptlyDB] = None
        if self.local_dir:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            self.local_db = PromptlyDB(self.local_dir / "prompts.db", is_global=False)

        # LLM backend (lazy initialization)
        self._llm_backend = None

    def _resolve_local_dir(self) -> Optional[Path]:
        """Resolve the local .promptly/ directory."""
        if self.config.local_dir:
            return self.config.local_dir

        if not self.config.auto_detect_local:
            return None

        # Auto-detect: walk up from cwd looking for .promptly/ or .git/
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            local_promptly = parent / ".promptly"
            if local_promptly.exists():
                return local_promptly

            # Also check for .git as project root indicator
            if (parent / ".git").exists():
                return parent / ".promptly"

        # Default to cwd/.promptly
        return cwd / ".promptly"

    def __enter__(self) -> "Promptly":
        """Context manager entry."""
        self.global_db.connect()
        if self.local_db:
            self.local_db.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.global_db.close()
        if self.local_db:
            self.local_db.close()

    def connect(self) -> None:
        """Open database connections."""
        self.global_db.connect()
        if self.local_db:
            self.local_db.connect()

    def close(self) -> None:
        """Close database connections."""
        self.global_db.close()
        if self.local_db:
            self.local_db.close()

    # ==================== Prompt Operations ====================

    def add(
        self,
        name: str,
        content: str,
        metadata: Dict = None,
        scope: str = "local",
        branch: str = None
    ) -> Tuple[int, str]:
        """
        Add or update a prompt.

        Args:
            name: Prompt name (unique identifier)
            content: Prompt content (supports {{variable}} syntax)
            metadata: Optional metadata dict
            scope: "local" (default) or "global"
            branch: Branch name (default: current branch)

        Returns:
            Tuple of (version, commit_hash)

        Example:
            p.add("greeting", "Hello, {{name}}! Welcome to {{project}}.")
            p.add("system_prompt", "You are a helpful assistant.", scope="global")
        """
        db = self._get_db_for_scope(scope)
        current_branch = branch or db.get_current_branch()
        return db.add_prompt(name, content, metadata, current_branch)

    def get(
        self,
        name: str,
        version: int = None,
        commit_hash: str = None
    ) -> Optional[Prompt]:
        """
        Get a prompt by name.

        Checks local database first, then global. Local prompts override global.

        Args:
            name: Prompt name
            version: Specific version number (optional)
            commit_hash: Specific commit hash or prefix (optional)

        Returns:
            Prompt object or None if not found

        Example:
            prompt = p.get("greeting")
            print(prompt.content)
            print(prompt.render(name="World", project="Promptly"))
        """
        # Check local first
        if self.local_db:
            prompt = self.local_db.get_prompt(name, version, commit_hash)
            if prompt:
                return prompt

        # Fall back to global
        return self.global_db.get_prompt(name, version, commit_hash)

    def list_prompts(
        self,
        scope: str = "all",
        branch: str = None
    ) -> List[Prompt]:
        """
        List all prompts.

        Args:
            scope: "all" (merged), "local", or "global"
            branch: Filter by branch name (optional)

        Returns:
            List of Prompt objects (local overrides global for same name)
        """
        if scope == "local" and self.local_db:
            return self.local_db.list_prompts(branch)
        elif scope == "global":
            return self.global_db.list_prompts(branch)
        else:
            # Merged view: local overrides global
            prompts_by_name: Dict[str, Prompt] = {}

            # Add global prompts first
            for prompt in self.global_db.list_prompts(branch):
                prompts_by_name[prompt.name] = prompt

            # Override with local prompts
            if self.local_db:
                for prompt in self.local_db.list_prompts(branch):
                    prompts_by_name[prompt.name] = prompt

            return list(prompts_by_name.values())

    def log(self, name: str) -> List[PromptVersion]:
        """
        Get version history for a prompt.

        Checks local first, then global.

        Args:
            name: Prompt name

        Returns:
            List of PromptVersion objects (newest first)
        """
        # Check local first
        if self.local_db:
            history = self.local_db.get_prompt_history(name)
            if history:
                return history

        return self.global_db.get_prompt_history(name)

    def delete(self, name: str, scope: str = "local") -> bool:
        """
        Delete a prompt.

        Args:
            name: Prompt name
            scope: "local" or "global"

        Returns:
            True if deleted, False if not found
        """
        db = self._get_db_for_scope(scope)
        return db.delete_prompt(name)

    # ==================== Branch Operations ====================

    def branch(self, name: str, from_branch: str = None, scope: str = "local") -> bool:
        """
        Create a new branch.

        Args:
            name: New branch name
            from_branch: Parent branch (default: current branch)
            scope: "local" or "global"

        Returns:
            True if created, False if already exists
        """
        db = self._get_db_for_scope(scope)
        parent = from_branch or db.get_current_branch()
        return db.create_branch(name, parent)

    def checkout(self, name: str, scope: str = "local") -> bool:
        """
        Switch to a branch.

        Args:
            name: Branch name
            scope: "local" or "global"

        Returns:
            True if switched, False if branch doesn't exist
        """
        db = self._get_db_for_scope(scope)
        return db.checkout_branch(name)

    def branches(self, scope: str = "all") -> List[str]:
        """
        List all branches.

        Args:
            scope: "all", "local", or "global"

        Returns:
            List of branch names
        """
        branches = set()

        if scope in ("all", "global"):
            branches.update(self.global_db.list_branches())

        if scope in ("all", "local") and self.local_db:
            branches.update(self.local_db.list_branches())

        return sorted(branches)

    def current_branch(self, scope: str = "local") -> str:
        """Get current branch for specified scope."""
        db = self._get_db_for_scope(scope)
        return db.get_current_branch()

    # ==================== Chain Operations ====================

    def create_chain(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        metadata: Dict = None,
        scope: str = "local"
    ) -> int:
        """
        Create a prompt chain.

        Args:
            name: Chain name
            steps: List of step definitions, each with:
                - prompt: Prompt name to execute
                - input_mapping: Dict mapping variables to previous outputs
                - output_key: Key to store this step's output
            metadata: Optional metadata
            scope: "local" or "global"

        Returns:
            Chain ID

        Example:
            p.create_chain("summarize_and_translate", [
                {"prompt": "summarize", "output_key": "summary"},
                {"prompt": "translate", "input_mapping": {"text": "summary"}}
            ])
        """
        db = self._get_db_for_scope(scope)
        return db.create_chain(name, steps, metadata)

    def get_chain(self, name: str) -> Optional[Chain]:
        """Get a chain by name (checks local first)."""
        if self.local_db:
            chain = self.local_db.get_chain(name)
            if chain:
                return chain
        return self.global_db.get_chain(name)

    def list_chains(self, scope: str = "all") -> List[Chain]:
        """List all chains."""
        if scope == "local" and self.local_db:
            return self.local_db.list_chains()
        elif scope == "global":
            return self.global_db.list_chains()
        else:
            chains_by_name: Dict[str, Chain] = {}
            for chain in self.global_db.list_chains():
                chains_by_name[chain.name] = chain
            if self.local_db:
                for chain in self.local_db.list_chains():
                    chains_by_name[chain.name] = chain
            return list(chains_by_name.values())

    async def execute_chain(
        self,
        name: str,
        initial_input: Dict[str, Any] = None,
        llm_backend: Any = None
    ) -> Dict[str, Any]:
        """
        Execute a prompt chain.

        Args:
            name: Chain name
            initial_input: Initial variables for first step
            llm_backend: LLM backend to use (auto-detected if None)

        Returns:
            Dict with all outputs keyed by output_key from each step
        """
        chain = self.get_chain(name)
        if not chain:
            raise ValueError(f"Chain '{name}' not found")

        backend = llm_backend or await self._get_llm_backend()
        outputs = dict(initial_input or {})

        for step in chain.steps:
            prompt_name = step.get("prompt")
            prompt = self.get(prompt_name)
            if not prompt:
                raise ValueError(f"Prompt '{prompt_name}' not found in chain '{name}'")

            # Apply input mapping
            input_vars = {}
            for var_name, source in step.get("input_mapping", {}).items():
                if source in outputs:
                    input_vars[var_name] = outputs[source]

            # Merge with any direct inputs
            input_vars.update(step.get("inputs", {}))

            # Render and execute
            rendered = prompt.render(**input_vars)
            result = await backend.complete(rendered)

            # Store output
            output_key = step.get("output_key", f"step_{chain.steps.index(step)}")
            outputs[output_key] = result

        return outputs

    # ==================== Skill Operations ====================

    def create_skill(
        self,
        name: str,
        description: str,
        prompt_template: str,
        files: List[Dict] = None,
        metadata: Dict = None,
        scope: str = "local"
    ) -> int:
        """
        Create a skill with optional attached files.

        Args:
            name: Skill name
            description: What the skill does
            prompt_template: The prompt template
            files: List of {filename, content, file_type} dicts
            metadata: Optional metadata
            scope: "local" or "global"

        Returns:
            Skill ID
        """
        db = self._get_db_for_scope(scope)
        return db.create_skill(name, description, prompt_template, files, metadata)

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name (checks local first)."""
        if self.local_db:
            skill = self.local_db.get_skill(name)
            if skill:
                return skill
        return self.global_db.get_skill(name)

    def list_skills(self, scope: str = "all") -> List[Skill]:
        """List all skills."""
        if scope == "local" and self.local_db:
            return self.local_db.list_skills()
        elif scope == "global":
            return self.global_db.list_skills()
        else:
            skills_by_name: Dict[str, Skill] = {}
            for skill in self.global_db.list_skills():
                skills_by_name[skill.name] = skill
            if self.local_db:
                for skill in self.local_db.list_skills():
                    skills_by_name[skill.name] = skill
            return list(skills_by_name.values())

    async def run_skill(
        self,
        name: str,
        input_vars: Dict[str, Any] = None,
        llm_backend: Any = None
    ) -> str:
        """
        Execute a skill.

        Args:
            name: Skill name
            input_vars: Variables to substitute in prompt
            llm_backend: LLM backend (auto-detected if None)

        Returns:
            LLM response
        """
        skill = self.get_skill(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        backend = llm_backend or await self._get_llm_backend()

        # Build context from files
        file_context = ""
        if skill.files:
            file_context = "\n\n---\nAttached Files:\n"
            for f in skill.files:
                file_context += f"\n### {f['filename']}\n```{f.get('file_type', 'text')}\n{f['content']}\n```\n"

        # Render prompt
        template = skill.prompt_template
        for key, value in (input_vars or {}).items():
            template = template.replace(f"{{{{{key}}}}}", str(value))

        full_prompt = template + file_context
        return await backend.complete(full_prompt)

    # ==================== Evaluation Operations ====================

    async def eval_prompt(
        self,
        name: str,
        criteria: str = "quality",
        test_input: Dict[str, Any] = None,
        llm_backend: Any = None
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt using LLM Judge.

        Args:
            name: Prompt name
            criteria: Evaluation criteria (e.g., "quality", "clarity", "safety")
            test_input: Variables to use for testing
            llm_backend: LLM backend (auto-detected if None)

        Returns:
            Evaluation result with score and feedback
        """
        prompt = self.get(name)
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found")

        backend = llm_backend or await self._get_llm_backend()

        # Render prompt with test input
        rendered = prompt.render(**(test_input or {}))

        # Generate response
        response = await backend.complete(rendered)

        # Evaluate with LLM Judge
        judge_prompt = f"""Evaluate the following prompt and response on {criteria}.

PROMPT:
{rendered}

RESPONSE:
{response}

Provide a score from 0-10 and brief feedback. Format:
SCORE: [number]
FEEDBACK: [your feedback]
"""
        judge_response = await backend.complete(judge_prompt)

        # Parse response
        score = 5.0  # default
        feedback = judge_response

        for line in judge_response.split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()

        # Store evaluation
        db = self.local_db or self.global_db
        db.add_evaluation(
            name,
            prompt.current_version,
            score,
            criteria,
            feedback,
            backend.model_name if hasattr(backend, 'model_name') else "unknown"
        )

        return {
            "score": score,
            "feedback": feedback,
            "criteria": criteria,
            "prompt_version": prompt.current_version,
            "response": response
        }

    # ==================== Export/Import ====================

    def export_prompt(self, name: str, format: str = "yaml") -> str:
        """
        Export a prompt to YAML or JSON.

        Args:
            name: Prompt name
            format: "yaml" or "json"

        Returns:
            Formatted string
        """
        prompt = self.get(name)
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found")

        data = {
            "name": prompt.name,
            "content": prompt.content,
            "version": prompt.current_version,
            "branch": prompt.branch,
            "commit_hash": prompt.commit_hash,
            "metadata": prompt.metadata
        }

        if format == "yaml":
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        else:
            import json
            return json.dumps(data, indent=2)

    def import_prompt(
        self,
        data: Union[str, Dict],
        scope: str = "local"
    ) -> Tuple[int, str]:
        """
        Import a prompt from YAML/JSON string or dict.

        Args:
            data: YAML/JSON string or dict
            scope: "local" or "global"

        Returns:
            Tuple of (version, commit_hash)
        """
        if isinstance(data, str):
            try:
                parsed = yaml.safe_load(data)
            except yaml.YAMLError:
                import json
                parsed = json.loads(data)
        else:
            parsed = data

        return self.add(
            name=parsed["name"],
            content=parsed["content"],
            metadata=parsed.get("metadata", {}),
            scope=scope
        )

    # ==================== Helpers ====================

    def _get_db_for_scope(self, scope: str) -> PromptlyDB:
        """Get the appropriate database for the given scope."""
        if scope == "global":
            return self.global_db
        elif scope == "local":
            if self.local_db:
                return self.local_db
            raise ValueError("No local database available. Use scope='global' or initialize with local_dir.")
        else:
            raise ValueError(f"Invalid scope: {scope}. Use 'local' or 'global'.")

    async def _get_llm_backend(self):
        """Get or create LLM backend (auto-detect)."""
        if self._llm_backend:
            return self._llm_backend

        # Try Ollama first (local, free)
        try:
            from promptly.integrations.llm_backends import OllamaBackend
            backend = OllamaBackend(model="llama3.2:3b")
            if await backend.is_available():
                self._llm_backend = backend
                return backend
        except ImportError:
            pass

        # Fall back to Claude API
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from promptly.integrations.llm_backends import ClaudeBackend
                self._llm_backend = ClaudeBackend(model="claude-3-5-sonnet-20241022")
                return self._llm_backend
            except ImportError:
                pass

        raise RuntimeError(
            "No LLM backend available. "
            "Install Ollama or set ANTHROPIC_API_KEY environment variable."
        )

    def info(self) -> Dict[str, Any]:
        """Get information about current Promptly configuration."""
        return {
            "global_dir": str(self.global_dir),
            "local_dir": str(self.local_dir) if self.local_dir else None,
            "global_prompts": len(self.global_db.list_prompts()) if self.global_db.conn else 0,
            "local_prompts": len(self.local_db.list_prompts()) if self.local_db and self.local_db.conn else 0,
            "current_branch": self.current_branch() if self.local_db else self.global_db.get_current_branch(),
        }

    def __repr__(self) -> str:
        return (
            f"Promptly(global='{self.global_dir}', "
            f"local='{self.local_dir}')"
        )
