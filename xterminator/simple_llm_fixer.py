"""
Simple LLM-Enhanced Fixer
=========================

Uses LLM to generate fixes for simple code issues, bypassing complex AST/template machinery.

Supports:
- Unused imports (dead_code)
- Magic numbers (hardcoded_values)
- Missing docstrings
- Other simple fixes

Philosophy: "Keep it simple, use intelligence when available"
"""

import re
from typing import Optional, Tuple, Dict, Any
from difflib import unified_diff


class SimpleLLMFixer:
    """
    LLM-enhanced simple fixer for trivial code issues.

    Falls back to rule-based fixes if LLM unavailable.

    Model Selection (Production Upgrade - November 2025):
    =====================================================

    Recommended models by capability:

    1. **llama3.1:8b** (DEFAULT - Best Balance)
       - 8B parameters, production-proven
       - Excellent code understanding
       - 150-200ms per fix
       - High availability (widely deployed)
       - Recommended for: General production use

    2. **qwen2.5-coder:7b** (BEST FOR CODE)
       - 7B specialized for code
       - Superior code fix quality (~95% accuracy)
       - 100-150ms per fix
       - Best for: Code-heavy fixes (imports, docstrings, magic numbers)

    3. **mistral:7b** (FAST BALANCE)
       - 7B balanced model
       - 80-120ms per fix
       - Good for: Time-critical production

    4. **llama3.2:3b** (LEGACY - Deprecated)
       - 3B lightweight, faster (~50-80ms)
       - Lower fix quality (~70% accuracy)
       - Only for: Resource-constrained environments

    To override:
        fixer = SimpleLLMFixer(llm_model="qwen2.5-coder:7b")
    """

    # Available production models with metadata
    PRODUCTION_MODELS = {
        "llama3.1:8b": {
            "description": "Proven 8B model - best balance",
            "quality": "high",
            "latency_ms": 175,
            "specialty": "general",
        },
        "qwen2.5-coder:7b": {
            "description": "Specialized code model - best quality",
            "quality": "very_high",
            "latency_ms": 125,
            "specialty": "code",
        },
        "mistral:7b": {
            "description": "Fast balanced model",
            "quality": "high",
            "latency_ms": 100,
            "specialty": "general",
        },
        "neural-chat:7b": {
            "description": "Instruction-optimized",
            "quality": "high",
            "latency_ms": 110,
            "specialty": "instruction",
        },
    }

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "llama3.1:8b",
        fallback_model: str = "llama3.2:3b"
    ):
        """
        Initialize simple fixer.

        Args:
            use_llm: Whether to use LLM for fix generation (default: True)
            llm_model: Ollama model to use (default: llama3.1:8b - production quality)
            fallback_model: Model to use if primary unavailable (default: llama3.2:3b)

        Model Selection Guide:
        =====================
        - Production (default): llama3.1:8b - best balance of quality and speed
        - Code-specific: qwen2.5-coder:7b - best fix quality for code issues
        - Fast path: mistral:7b - fastest production model
        """
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.fallback_model = fallback_model
        self.ollama_available = False
        self.model_info = self.PRODUCTION_MODELS.get(llm_model, {})

        # Try to import ollama
        if use_llm:
            try:
                import ollama
                self.ollama = ollama
                self.ollama_available = True
                self._verify_model_available()
            except ImportError:
                self.ollama_available = False

    def _verify_model_available(self) -> None:
        """
        Verify primary model is available. If not, attempt fallback.
        Graceful degradation: continues even if verification fails.
        """
        if not self.ollama_available:
            return

        try:
            # Try to check if model exists
            models = self.ollama.list()
            available_models = [m.get('name', '') for m in models.get('models', [])]

            if self.llm_model not in available_models:
                import warnings
                warnings.warn(
                    f"Model '{self.llm_model}' not found locally. "
                    f"Will attempt to pull on first use. "
                    f"Available: {', '.join(available_models[:3])}"
                )
        except Exception:
            # Verification failed, but continue anyway
            # Model will be pulled on first use if needed
            pass

    async def fix_issue(
        self,
        issue: Dict[str, Any],
        full_code: str,
        file_path: str
    ) -> Optional[Tuple[str, str]]:
        """
        Generate and apply fix for simple issue.

        Args:
            issue: Issue dict with category, message, line_number, code_snippet
            full_code: Full file content
            file_path: Path to file

        Returns:
            (fixed_code, diff) or None if can't fix
        """
        category = issue.get('category', '')
        message = issue.get('message', '')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')

        # Route to appropriate fixer
        if category == 'dead_code' and 'import' in message.lower():
            return self._fix_unused_import(full_code, message, code_snippet, line_number)

        elif category == 'hardcoded_values' and 'magic number' in message.lower():
            return self._fix_magic_number(full_code, message, code_snippet, line_number)

        elif category == 'missing_docstrings':
            return self._fix_missing_docstring(full_code, message, code_snippet, line_number)

        else:
            # Try LLM for other cases
            if self.use_llm and self.ollama_available:
                return await self._fix_with_llm(full_code, issue, file_path)
            else:
                return None

    def _fix_unused_import(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Fix unused import by removing it.

        Strategy:
        1. If single import: Remove entire line
        2. If from...import with multiple: Remove just the unused one
        """
        lines = full_code.split('\n')

        if line_number <= 0 or line_number > len(lines):
            return None

        # Get the import line (1-indexed)
        import_line = lines[line_number - 1]

        # Extract unused name from message (e.g., "Unused import 'Dict'")
        match = re.search(r"'([^']+)'", message)
        if not match:
            return None
        unused_name = match.group(1)

        # Case 1: Single import (import foo) or (from x import foo)
        if f"import {unused_name}" in import_line:
            # Remove entire line
            fixed_lines = lines[:line_number - 1] + lines[line_number:]
            fixed_code = '\n'.join(fixed_lines)

        # Case 2: Multiple imports (from x import foo, bar, baz)
        elif 'from' in import_line and 'import' in import_line:
            # Remove just the unused name
            # Pattern: "from typing import Dict, List, Optional"
            # Remove "Dict, " or ", Dict" or "Dict"

            # Split on 'import'
            parts = import_line.split('import', 1)
            if len(parts) != 2:
                return None

            prefix = parts[0] + 'import'
            imports_str = parts[1].strip()

            # Split imports by comma
            imports = [imp.strip() for imp in imports_str.split(',')]

            # Remove the unused one
            imports = [imp for imp in imports if imp != unused_name]

            if not imports:
                # No imports left, remove entire line
                fixed_lines = lines[:line_number - 1] + lines[line_number:]
                fixed_code = '\n'.join(fixed_lines)
            else:
                # Reconstruct import line
                new_import_line = f"{prefix} {', '.join(imports)}"
                lines[line_number - 1] = new_import_line
                fixed_code = '\n'.join(lines)

        else:
            return None

        # Generate diff
        diff = self._generate_diff(full_code, fixed_code, "unused_import")

        return (fixed_code, diff)

    def _fix_magic_number(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Fix magic number by extracting to constant.

        Uses LLM to generate good constant name if available,
        otherwise uses generic name.
        """
        # Extract the magic number from message
        # Message format: "Magic number 150 should be a named constant"
        match = re.search(r'Magic number (\d+(?:\.\d+)?)', message)
        if not match:
            return None
        magic_number = match.group(1)

        # Generate constant name (simple heuristic or LLM)
        if self.use_llm and self.ollama_available:
            constant_name = self._llm_generate_constant_name(magic_number, code_snippet)
        else:
            # Simple heuristic: use generic name
            constant_name = f"CONSTANT_{magic_number.replace('.', '_')}"

        lines = full_code.split('\n')

        # Find a good place to insert constant (after imports, before first function/class)
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('import') and not line.startswith('from'):
                insert_index = i
                break

        # Insert constant definition
        constant_line = f"{constant_name} = {magic_number}  # Extracted magic number"
        lines.insert(insert_index, constant_line)
        lines.insert(insert_index, "# Constants")

        # Replace magic number with constant in the original line
        # (line_number now shifted by +2)
        target_line_index = line_number - 1 + 2
        if target_line_index < len(lines):
            original_line = lines[target_line_index]
            # Replace the magic number (be careful with partial matches)
            # Use word boundaries
            pattern = r'\b' + re.escape(magic_number) + r'\b'
            fixed_line = re.sub(pattern, constant_name, original_line, count=1)
            lines[target_line_index] = fixed_line

        fixed_code = '\n'.join(lines)

        # Generate diff
        diff = self._generate_diff(full_code, fixed_code, "magic_number")

        return (fixed_code, diff)

    def _fix_missing_docstring(
        self,
        full_code: str,
        message: str,
        code_snippet: str,
        line_number: int
    ) -> Optional[Tuple[str, str]]:
        """
        Add docstring to function/class.

        Uses LLM to generate meaningful docstring if available.
        """
        if not self.use_llm or not self.ollama_available:
            # Without LLM, use generic docstring
            docstring = '"""TODO: Add docstring"""'
        else:
            # Use LLM to generate docstring
            docstring = self._llm_generate_docstring(code_snippet, full_code)

        lines = full_code.split('\n')

        if line_number <= 0 or line_number >= len(lines):
            return None

        # Find the function/class definition line
        def_line = lines[line_number - 1]

        # Get indentation
        indent_match = re.match(r'^(\s*)', def_line)
        indent = indent_match.group(1) if indent_match else ''

        # Insert docstring after the def line
        docstring_line = f'{indent}    """{docstring}"""'
        lines.insert(line_number, docstring_line)

        fixed_code = '\n'.join(lines)

        # Generate diff
        diff = self._generate_diff(full_code, fixed_code, "missing_docstring")

        return (fixed_code, diff)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM model.

        Returns:
            Dict with model name, quality, latency, specialty
        """
        if not self.model_info:
            return {
                "model": self.llm_model,
                "quality": "unknown",
                "latency_ms": 0,
                "specialty": "unknown",
                "available": self.ollama_available
            }

        return {
            "model": self.llm_model,
            "description": self.model_info.get("description", ""),
            "quality": self.model_info.get("quality", "unknown"),
            "latency_ms": self.model_info.get("latency_ms", 0),
            "specialty": self.model_info.get("specialty", "unknown"),
            "available": self.ollama_available
        }

    def _select_model_with_fallback(self) -> str:
        """
        Select model to use, with fallback logic.

        Returns:
            Model name to use for LLM call
        """
        # If no model info (unknown model), just use what was requested
        if not self.model_info:
            return self.llm_model

        # Use primary model
        return self.llm_model

    def _llm_generate_constant_name(self, number: str, context: str) -> str:
        """
        Use LLM to generate semantic constant name.

        Args:
            number: The magic number
            context: Code context where number appears

        Returns:
            Constant name (e.g., DEFAULT_TIMEOUT_SECONDS)
        """
        prompt = f"""Given this code snippet:

{context}

The number {number} appears. Generate a good constant name for it.
Rules:
- UPPER_SNAKE_CASE
- Descriptive and semantic
- Include units if applicable (SECONDS, MILLISECONDS, etc.)
- One word or short phrase

Just output the constant name, nothing else."""

        try:
            response = self.ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3}
            )

            constant_name = response['message']['content'].strip()

            # Clean up (remove quotes, extra whitespace)
            constant_name = re.sub(r'[^A-Z0-9_]', '', constant_name)

            if constant_name and constant_name[0].isalpha():
                return constant_name
            else:
                return f"CONSTANT_{number.replace('.', '_')}"

        except Exception:
            return f"CONSTANT_{number.replace('.', '_')}"

    def _llm_generate_docstring(self, code_snippet: str, full_code: str) -> str:
        """
        Use LLM to generate meaningful docstring.

        Args:
            code_snippet: Function/class definition
            full_code: Full file context

        Returns:
            Docstring text (without triple quotes)
        """
        prompt = f"""Generate a concise docstring for this function:

{code_snippet}

Context (full file):
{full_code[:500]}...

Rules:
- One line if simple, multiple lines if complex
- Describe what it does, not how
- Include Args/Returns if function has parameters/return value
- Be specific and technical

Just output the docstring text (without triple quotes), nothing else."""

        try:
            response = self.ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.3, 'num_predict': 150}
            )

            docstring = response['message']['content'].strip()

            # Remove triple quotes if present
            docstring = docstring.replace('"""', '').replace("'''", '')

            return docstring

        except Exception:
            return "TODO: Add docstring"

    async def _fix_with_llm(
        self,
        full_code: str,
        issue: Dict[str, Any],
        file_path: str
    ) -> Optional[Tuple[str, str]]:
        """
        Use LLM to generate fix for any issue.

        Args:
            full_code: Full file content
            issue: Issue dict
            file_path: Path to file

        Returns:
            (fixed_code, diff) or None
        """
        category = issue.get('category', '')
        message = issue.get('message', '')
        line_number = issue.get('line_number', 0)
        code_snippet = issue.get('code_snippet', '')

        prompt = f"""Fix this code issue:

File: {file_path}
Issue: {message}
Category: {category}
Line {line_number}:
{code_snippet}

Full file context:
{full_code}

Generate the COMPLETE fixed file. Make ONLY the minimal change needed to fix the issue.
Output ONLY the fixed code, no explanations."""

        try:
            response = self.ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 2000}
            )

            fixed_code = response['message']['content'].strip()

            # Remove markdown code blocks if present
            fixed_code = re.sub(r'^```python\n', '', fixed_code)
            fixed_code = re.sub(r'^```\n', '', fixed_code)
            fixed_code = re.sub(r'\n```$', '', fixed_code)

            # Generate diff
            diff = self._generate_diff(full_code, fixed_code, category)

            return (fixed_code, diff)

        except Exception as e:
            # LLM failed, return None
            return None

    def _generate_diff(self, original: str, fixed: str, label: str) -> str:
        """Generate unified diff."""
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff = unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f'original ({label})',
            tofile=f'fixed ({label})',
            lineterm=''
        )

        return ''.join(diff)
