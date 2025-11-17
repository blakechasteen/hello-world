#!/usr/bin/env python3
"""
Code Generation Engine
======================
Modular, extensible code generation using LLMs.

Capabilities:
- Code generation from natural language
- Code refactoring with diffs
- Bug fixing
- Test generation
- Code review and suggestions
- Code explanation

Author: Claude Code
Date: November 16, 2025
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from llm_providers import LLMClient
from promptly_integration import PromptRegistry, PromptType

logger = logging.getLogger(__name__)


class CodeTask(Enum):
    """Types of code generation tasks"""
    GENERATE = "generate"
    REFACTOR = "refactor"
    FIX = "fix"
    TEST = "test"
    REVIEW = "review"
    EXPLAIN = "explain"


@dataclass
class CodeContext:
    """Context for code generation"""
    language: Optional[str] = None
    file_name: Optional[str] = None
    selection: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None
    workspace: Optional[str] = None
    surrounding_code: Optional[str] = None


@dataclass
class CodeGenerationResult:
    """Result of code generation"""
    code: str
    explanation: str
    confidence: float
    task_type: CodeTask
    language: Optional[str] = None
    diff: Optional[str] = None  # For refactoring tasks


class CodeGenerationEngine:
    """
    Modular code generation engine.

    Uses LLM to generate, refactor, fix, test, and review code.
    Designed to be elegant, extensible, and production-ready.
    """

    def __init__(self, llm_client: LLMClient, prompt_registry: Optional[PromptRegistry] = None):
        """
        Initialize code generation engine.

        Args:
            llm_client: Initialized LLM client
            prompt_registry: Optional Promptly registry (uses defaults if not provided)
        """
        self.llm = llm_client
        self.prompt_registry = prompt_registry

        # Use Promptly if available, otherwise fallback to hardcoded prompts
        if self.prompt_registry:
            logger.info("Using Promptly prompt registry for code generation")
            self._system_prompts = None  # Will fetch from registry on demand
        else:
            logger.warning("No prompt registry provided, using hardcoded prompts")
            self._system_prompts = self._initialize_system_prompts()

    def _initialize_system_prompts(self) -> Dict[CodeTask, str]:
        """Initialize task-specific system prompts"""
        return {
            CodeTask.GENERATE: """You are an expert software engineer specializing in code generation.

When asked to generate code:
1. Write clean, idiomatic, production-ready code
2. Include type hints/annotations where appropriate
3. Add concise comments for complex logic
4. Follow language best practices
5. Consider edge cases and error handling

Output format:
```<language>
<code>
```

<explanation>""",

            CodeTask.REFACTOR: """You are an expert code reviewer specializing in refactoring.

When asked to refactor code:
1. Improve code structure and readability
2. Eliminate code smells (duplication, complexity)
3. Preserve functionality (behavior unchanged)
4. Suggest modern patterns where applicable
5. Explain each significant change

Output format:
```<language>
<refactored_code>
```

DIFF:
```diff
<unified_diff>
```

<explanation_of_changes>""",

            CodeTask.FIX: """You are an expert debugger and bug fixer.

When asked to fix code:
1. Identify the root cause of the issue
2. Provide a minimal, targeted fix
3. Preserve existing functionality
4. Add defensive checks if needed
5. Explain what was wrong and how it's fixed

Output format:
```<language>
<fixed_code>
```

BUG EXPLANATION:
<what_was_wrong>

FIX EXPLANATION:
<how_its_fixed>""",

            CodeTask.TEST: """You are an expert test engineer specializing in comprehensive testing.

When asked to generate tests:
1. Write thorough unit tests covering edge cases
2. Use appropriate testing framework for language
3. Include setup/teardown if needed
4. Add descriptive test names
5. Test both success and failure paths

Output format:
```<language>
<test_code>
```

TEST COVERAGE:
<what_is_tested>""",

            CodeTask.REVIEW: """You are a senior code reviewer with years of experience.

When asked to review code:
1. Identify potential bugs and issues
2. Suggest improvements for readability
3. Check for security vulnerabilities
4. Assess performance considerations
5. Recommend best practices

Output format:
## Issues Found
- <issue_1>
- <issue_2>

## Suggestions
1. <suggestion_1>
2. <suggestion_2>

## Security Considerations
<security_notes>

## Recommended Changes
```<language>
<improved_code>
```""",

            CodeTask.EXPLAIN: """You are an expert educator specializing in code explanation.

When asked to explain code:
1. Explain what the code does (high-level)
2. Break down complex parts step-by-step
3. Identify key algorithms or patterns
4. Highlight potential issues or edge cases
5. Use analogies when helpful

Output format:
## Overview
<high_level_explanation>

## Step-by-Step Breakdown
1. <step_1>
2. <step_2>

## Key Concepts
- <concept_1>
- <concept_2>

## Potential Issues
<issues_or_edge_cases>"""
        }

    def _get_system_prompt(self, task_type: CodeTask) -> str:
        """
        Get system prompt for task type.

        Uses Promptly registry if available, otherwise fallback to hardcoded prompts.

        Args:
            task_type: Type of code generation task

        Returns:
            System prompt string
        """
        if self.prompt_registry:
            # Map CodeTask to prompt name
            prompt_name_map = {
                CodeTask.GENERATE: "code_generation_system",
                CodeTask.REFACTOR: "code_refactoring",
                CodeTask.FIX: "bug_fixing",
                CodeTask.TEST: "test_generation",
                CodeTask.REVIEW: "code_review_system",
                CodeTask.EXPLAIN: "code_explanation"
            }

            prompt_name = prompt_name_map.get(task_type, "code_generation_system")
            prompt_template = self.prompt_registry.get(prompt_name)

            if prompt_template:
                return prompt_template.template
            else:
                logger.warning(f"Prompt '{prompt_name}' not found in registry, using fallback")
                return self._system_prompts.get(task_type, self._system_prompts[CodeTask.GENERATE])
        else:
            # Fallback to hardcoded prompts
            return self._system_prompts.get(task_type, self._system_prompts[CodeTask.GENERATE])

    async def generate(
        self,
        task: str,
        context: Optional[CodeContext] = None,
        task_type: CodeTask = CodeTask.GENERATE
    ) -> CodeGenerationResult:
        """
        Generate code based on task and context.

        Args:
            task: Natural language description of task
            context: Code context (language, file, selection, etc.)
            task_type: Type of code generation task

        Returns:
            CodeGenerationResult with generated code and explanation
        """
        # Build prompt from task and context
        prompt = self._build_prompt(task, context, task_type)

        # Get system prompt for task type (from Promptly or fallback)
        system_prompt = self._get_system_prompt(task_type)

        # Generate with LLM
        logger.info(f"Generating code: {task_type.value} - {task[:50]}...")
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3 if task_type in [CodeTask.FIX, CodeTask.TEST] else 0.7
        )

        # Parse response into structured result
        result = self._parse_response(response, task_type, context)

        logger.info(f"Generated {len(result.code)} chars of code (confidence: {result.confidence:.2f})")
        return result

    def _build_prompt(
        self,
        task: str,
        context: Optional[CodeContext],
        task_type: CodeTask
    ) -> str:
        """Build comprehensive prompt from task and context"""
        prompt_parts = []

        # Add task description
        if task_type == CodeTask.GENERATE:
            prompt_parts.append(f"Generate code for: {task}")
        elif task_type == CodeTask.REFACTOR:
            prompt_parts.append(f"Refactor this code: {task}")
        elif task_type == CodeTask.FIX:
            prompt_parts.append(f"Fix this issue: {task}")
        elif task_type == CodeTask.TEST:
            prompt_parts.append(f"Generate tests for: {task}")
        elif task_type == CodeTask.REVIEW:
            prompt_parts.append(f"Review this code: {task}")
        elif task_type == CodeTask.EXPLAIN:
            prompt_parts.append(f"Explain: {task}")

        # Add context if available
        if context:
            if context.language:
                prompt_parts.append(f"\nLanguage: {context.language}")

            if context.file_name:
                prompt_parts.append(f"File: {context.file_name}")

            if context.selection:
                prompt_parts.append(f"\nCurrent code:\n```{context.language or ''}\n{context.selection}\n```")

            if context.diagnostics:
                errors = "\n".join(f"- {d.get('message', '')}" for d in context.diagnostics)
                prompt_parts.append(f"\nDiagnostics:\n{errors}")

            if context.surrounding_code:
                prompt_parts.append(f"\nSurrounding context:\n```{context.language or ''}\n{context.surrounding_code}\n```")

        return "\n".join(prompt_parts)

    def _parse_response(
        self,
        response: str,
        task_type: CodeTask,
        context: Optional[CodeContext]
    ) -> CodeGenerationResult:
        """Parse LLM response into structured result"""
        # Extract code blocks
        code = self._extract_code_block(response)
        if not code:
            code = response  # Fallback to full response

        # Extract diff if present (for refactoring)
        diff = None
        if task_type == CodeTask.REFACTOR:
            diff = self._extract_diff(response)

        # Extract explanation
        explanation = self._extract_explanation(response, code)

        # Estimate confidence based on response quality
        confidence = self._estimate_confidence(response, code, task_type)

        return CodeGenerationResult(
            code=code,
            explanation=explanation,
            confidence=confidence,
            task_type=task_type,
            language=context.language if context else None,
            diff=diff
        )

    def _extract_code_block(self, response: str) -> Optional[str]:
        """Extract code from markdown code blocks"""
        import re

        # Try to find code block with language
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            # Return first code block
            return matches[0].strip()

        return None

    def _extract_diff(self, response: str) -> Optional[str]:
        """Extract diff from response"""
        import re

        # Look for DIFF: section or diff code block
        diff_pattern = r"```diff\n(.*?)```"
        matches = re.findall(diff_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    def _extract_explanation(self, response: str, code: str) -> str:
        """Extract explanation text (non-code parts)"""
        # Remove code blocks
        import re
        cleaned = re.sub(r"```.*?```", "", response, flags=re.DOTALL)

        # Remove common markers
        cleaned = re.sub(r"DIFF:.*", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"BUG EXPLANATION:.*", "", cleaned)
        cleaned = re.sub(r"FIX EXPLANATION:.*", "", cleaned)
        cleaned = re.sub(r"TEST COVERAGE:.*", "", cleaned)

        return cleaned.strip() or "Code generated successfully."

    def _estimate_confidence(self, response: str, code: str, task_type: CodeTask) -> float:
        """Estimate confidence in generated code"""
        confidence = 0.7  # Base confidence

        # Boost for code blocks (proper formatting)
        if "```" in response:
            confidence += 0.1

        # Boost for explanation
        if len(response) > len(code) + 50:
            confidence += 0.1

        # Boost for diff (refactoring)
        if task_type == CodeTask.REFACTOR and "```diff" in response:
            confidence += 0.1

        return min(confidence, 1.0)

    async def generate_code(
        self,
        description: str,
        language: Optional[str] = None,
        context: Optional[CodeContext] = None
    ) -> CodeGenerationResult:
        """
        Generate new code from natural language description.

        Args:
            description: What the code should do
            language: Target programming language
            context: Additional context

        Returns:
            Generated code with explanation
        """
        if not context:
            context = CodeContext(language=language)
        elif language:
            context.language = language

        return await self.generate(description, context, CodeTask.GENERATE)

    async def refactor_code(
        self,
        code: str,
        instructions: str,
        language: Optional[str] = None
    ) -> CodeGenerationResult:
        """
        Refactor existing code with specific instructions.

        Args:
            code: Code to refactor
            instructions: How to refactor
            language: Programming language

        Returns:
            Refactored code with diff and explanation
        """
        context = CodeContext(language=language, selection=code)
        return await self.generate(instructions, context, CodeTask.REFACTOR)

    async def fix_code(
        self,
        code: str,
        error_message: Optional[str] = None,
        diagnostics: Optional[List[Dict]] = None,
        language: Optional[str] = None
    ) -> CodeGenerationResult:
        """
        Fix buggy code based on error messages.

        Args:
            code: Buggy code
            error_message: Error/issue description
            diagnostics: VS Code diagnostics
            language: Programming language

        Returns:
            Fixed code with explanation
        """
        context = CodeContext(
            language=language,
            selection=code,
            diagnostics=diagnostics
        )

        task = error_message or "Fix errors in this code"
        return await self.generate(task, context, CodeTask.FIX)

    async def generate_tests(
        self,
        code: str,
        language: Optional[str] = None,
        test_framework: Optional[str] = None
    ) -> CodeGenerationResult:
        """
        Generate comprehensive tests for code.

        Args:
            code: Code to test
            language: Programming language
            test_framework: Specific test framework (pytest, jest, etc.)

        Returns:
            Test code with coverage explanation
        """
        context = CodeContext(language=language, selection=code)

        task = f"Generate comprehensive tests"
        if test_framework:
            task += f" using {test_framework}"

        return await self.generate(task, context, CodeTask.TEST)

    async def review_code(
        self,
        code: str,
        language: Optional[str] = None
    ) -> CodeGenerationResult:
        """
        Review code for issues, improvements, security.

        Args:
            code: Code to review
            language: Programming language

        Returns:
            Review with issues, suggestions, and improvements
        """
        context = CodeContext(language=language, selection=code)
        return await self.generate("Review this code", context, CodeTask.REVIEW)

    async def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        question: Optional[str] = None
    ) -> CodeGenerationResult:
        """
        Explain what code does.

        Args:
            code: Code to explain
            language: Programming language
            question: Specific question about code

        Returns:
            Detailed explanation
        """
        context = CodeContext(language=language, selection=code)

        task = question or "Explain what this code does"
        return await self.generate(task, context, CodeTask.EXPLAIN)
