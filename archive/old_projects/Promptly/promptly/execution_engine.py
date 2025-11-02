#!/usr/bin/env python3
"""
Promptly Execution Engine
==========================
Execute skills and prompts with various LLM backends (Claude API, Ollama, custom).
Supports parallel execution, error handling, retries, and execution tracking.
"""

import asyncio
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import sys


class ExecutionBackend(Enum):
    """Supported execution backends"""
    CLAUDE_API = "claude_api"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class ExecutionConfig:
    """Configuration for skill execution"""
    backend: ExecutionBackend = ExecutionBackend.OLLAMA
    model: str = "llama3.2:3b"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120  # seconds
    retries: int = 3
    retry_delay: int = 2  # seconds
    api_key: Optional[str] = None  # For Claude API
    custom_executor: Optional[Callable] = None  # For custom backend


@dataclass
class ExecutionResult:
    """Result from skill execution"""
    skill_name: str
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: Optional[int] = None
    model: str = ""
    backend: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OllamaExecutor:
    """Execute prompts using Ollama"""

    @staticmethod
    def get_ollama_path():
        """Find ollama executable"""
        import os

        # Try in PATH
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return "ollama"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try Windows default
        default_path = os.path.join(
            os.environ.get('LOCALAPPDATA', ''),
            'Programs', 'Ollama', 'ollama.exe'
        )
        if os.path.exists(default_path):
            return default_path

        return None

    @staticmethod
    def execute(prompt: str, model: str = "llama3.2:3b", timeout: int = 120) -> str:
        """Execute a prompt with Ollama"""
        ollama_path = OllamaExecutor.get_ollama_path()

        if not ollama_path:
            raise RuntimeError("Ollama not found. Install from https://ollama.ai")

        try:
            result = subprocess.run(
                [ollama_path, "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Execution timeout after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")


class ClaudeAPIExecutor:
    """Execute prompts using Claude API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy load Anthropic client"""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def execute(
        self,
        prompt: str,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> tuple[str, int]:
        """
        Execute a prompt with Claude API.

        Returns:
            tuple of (response_text, tokens_used)
        """
        client = self._get_client()

        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            tokens_used = message.usage.input_tokens + message.usage.output_tokens

            return response_text, tokens_used

        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")


class ExecutionEngine:
    """Main execution engine for skills and prompts"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._claude_executor = None

    def _get_executor(self) -> Callable:
        """Get the appropriate executor based on backend"""
        if self.config.backend == ExecutionBackend.OLLAMA:
            return lambda prompt: OllamaExecutor.execute(
                prompt,
                model=self.config.model,
                timeout=self.config.timeout
            )

        elif self.config.backend == ExecutionBackend.CLAUDE_API:
            if not self.config.api_key:
                raise ValueError("API key required for Claude API backend")

            if self._claude_executor is None:
                self._claude_executor = ClaudeAPIExecutor(self.config.api_key)

            return lambda prompt: self._claude_executor.execute(
                prompt,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

        elif self.config.backend == ExecutionBackend.CUSTOM:
            if not self.config.custom_executor:
                raise ValueError("Custom executor function required")
            return self.config.custom_executor

        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def execute_prompt(
        self,
        prompt: str,
        skill_name: str = "unnamed"
    ) -> ExecutionResult:
        """Execute a single prompt with retry logic"""
        executor = self._get_executor()

        for attempt in range(self.config.retries):
            try:
                start_time = time.time()

                # Execute
                result = executor(prompt)

                # Handle different return types
                output = result
                tokens = None
                if isinstance(result, tuple):
                    output, tokens = result

                execution_time = time.time() - start_time

                return ExecutionResult(
                    skill_name=skill_name,
                    success=True,
                    output=output,
                    execution_time=execution_time,
                    tokens_used=tokens,
                    model=self.config.model,
                    backend=self.config.backend.value
                )

            except Exception as e:
                if attempt == self.config.retries - 1:
                    # Last attempt failed
                    return ExecutionResult(
                        skill_name=skill_name,
                        success=False,
                        output="",
                        error=str(e),
                        model=self.config.model,
                        backend=self.config.backend.value
                    )

                # Wait before retry
                time.sleep(self.config.retry_delay)

        # Should never reach here
        return ExecutionResult(
            skill_name=skill_name,
            success=False,
            output="",
            error="Max retries exceeded",
            model=self.config.model,
            backend=self.config.backend.value
        )

    def execute_skill(
        self,
        skill_payload: Dict[str, Any],
        user_input: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a skill from its payload.

        Args:
            skill_payload: Skill payload from prepare_skill_payload()
            user_input: Optional user input/request

        Returns:
            ExecutionResult with output
        """
        # Build prompt from skill
        prompt_parts = []

        # Description
        if skill_payload.get('description'):
            prompt_parts.append(f"# Skill: {skill_payload['skill_name']}")
            prompt_parts.append(f"\n{skill_payload['description']}\n")

        # Files
        for file_info in skill_payload.get('files', []):
            prompt_parts.append(
                f"\n## Reference: {file_info['filename']} ({file_info['filetype']})"
            )
            prompt_parts.append(f"```{file_info['filetype']}\n{file_info['content']}\n```\n")

        # User input
        if user_input:
            prompt_parts.append(f"\n## Task:\n{user_input}\n")

        prompt = "\n".join(prompt_parts)

        return self.execute_prompt(prompt, skill_name=skill_payload['skill_name'])

    async def execute_parallel(
        self,
        prompts: List[tuple[str, str]]  # (skill_name, prompt)
    ) -> List[ExecutionResult]:
        """
        Execute multiple prompts in parallel.

        Args:
            prompts: List of (skill_name, prompt) tuples

        Returns:
            List of ExecutionResults
        """
        tasks = []

        for skill_name, prompt in prompts:
            # Wrap sync execution in async
            task = asyncio.to_thread(self.execute_prompt, prompt, skill_name)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ExecutionResult(
                    skill_name=prompts[i][0],
                    success=False,
                    output="",
                    error=str(result),
                    model=self.config.model,
                    backend=self.config.backend.value
                ))
            else:
                final_results.append(result)

        return final_results


class ChainExecutor:
    """Execute chains of skills with data flow"""

    def __init__(self, engine: ExecutionEngine):
        self.engine = engine

    def execute_chain(
        self,
        skills: List[Dict[str, Any]],
        initial_input: str
    ) -> List[ExecutionResult]:
        """
        Execute a chain of skills, passing output to next input.

        Args:
            skills: List of skill payloads
            initial_input: Input for first skill

        Returns:
            List of ExecutionResults (one per skill)
        """
        results = []
        current_input = initial_input

        for i, skill in enumerate(skills):
            print(f"Executing step {i+1}/{len(skills)}: {skill['skill_name']}...")

            result = self.engine.execute_skill(skill, user_input=current_input)
            results.append(result)

            if not result.success:
                print(f"✗ Step {i+1} failed: {result.error}")
                break

            print(f"✓ Step {i+1} completed ({result.execution_time:.2f}s)")

            # Pass output to next skill
            current_input = result.output

        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def execute_with_ollama(prompt: str, model: str = "llama3.2:3b") -> ExecutionResult:
    """Quick execution with Ollama"""
    config = ExecutionConfig(
        backend=ExecutionBackend.OLLAMA,
        model=model
    )
    engine = ExecutionEngine(config)
    return engine.execute_prompt(prompt)


def execute_with_claude(
    prompt: str,
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022"
) -> ExecutionResult:
    """Quick execution with Claude API"""
    config = ExecutionConfig(
        backend=ExecutionBackend.CLAUDE_API,
        model=model,
        api_key=api_key
    )
    engine = ExecutionEngine(config)
    return engine.execute_prompt(prompt)


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Promptly Execution Engine")
    parser.add_argument("prompt", help="Prompt to execute")
    parser.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--api-key", help="API key (for Claude)")

    args = parser.parse_args()

    if args.backend == "ollama":
        model = args.model or "llama3.2:3b"
        result = execute_with_ollama(args.prompt, model=model)
    else:
        if not args.api_key:
            print("Error: --api-key required for Claude backend")
            sys.exit(1)
        model = args.model or "claude-3-5-sonnet-20241022"
        result = execute_with_claude(args.prompt, api_key=args.api_key, model=model)

    print(f"\nBackend: {result.backend}")
    print(f"Model: {result.model}")
    print(f"Success: {result.success}")
    print(f"Time: {result.execution_time:.2f}s")
    if result.tokens_used:
        print(f"Tokens: {result.tokens_used}")
    print(f"\nOutput:\n{result.output}")
    if result.error:
        print(f"\nError: {result.error}")
