"""
Template Fixer - Pattern-Based Code Fixes
==========================================

Charlotte's Template Fixer - "Some Spider spun templates!"

Applies pattern-based fixes using templates for common issues:
- Add try/except blocks
- Add context managers
- Move secrets to environment variables
- Add null checks
- Add docstrings
- Fix timezone-naive datetime
- Add type hints
- Add logging

Key Difference from AST Fixer:
- AST Fixer: Structural transformations (extract function, remove dead code)
- Template Fixer: Pattern replacement (add try/except, move to env var)
"""

import re
import ast
from typing import Optional, Tuple, Dict, List
from difflib import unified_diff

from .xterminator_types import FixProposal, FixStrategy
from .templates import (
    FixTemplate,
    get_template,
    get_templates_for_category,
    TEMPLATES
)


class TemplateFixer:
    """
    Template-based code fixer.

    "With great templates comes great responsibility!" - Charlotte
    """

    def __init__(self):
        self.templates = TEMPLATES
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        self.compiled_patterns = {}
        for name, template in self.templates.items():
            try:
                self.compiled_patterns[name] = re.compile(
                    template.pattern,
                    re.MULTILINE | re.DOTALL
                )
            except re.error as e:
                print(f"Warning: Failed to compile pattern for {name}: {e}")

    async def fix_issue(
        self,
        proposal: FixProposal,
        full_code: str
    ) -> Optional[Tuple[str, str]]:
        """
        Apply template-based fix.

        Args:
            proposal: Fix proposal from classifier
            full_code: Full source code (not just snippet)

        Returns:
            (fixed_code, diff) or None if can't fix
        """
        if proposal.fix_strategy != FixStrategy.TEMPLATE:
            return None

        # Select template based on issue category
        template = self._select_template(proposal)
        if not template:
            return None

        # Extract context from code
        context = self._extract_context(full_code, proposal, template)
        if not context:
            return None

        # Apply template
        fixed_code = self._apply_template(template, context, full_code, proposal)
        if not fixed_code:
            return None

        # Generate diff
        diff = self._generate_diff(full_code, fixed_code, proposal)

        return (fixed_code, diff)

    def _select_template(self, proposal: FixProposal) -> Optional[FixTemplate]:
        """
        Select appropriate template for the issue.

        Strategy:
        1. Get category-specific templates
        2. Try each template's pattern against the code
        3. Return first matching template
        4. If no match, return None
        """
        category = proposal.issue_category

        # Get all templates for category
        candidates = get_templates_for_category(category)

        if not candidates:
            # No templates for this category
            return None

        # Try each candidate's pattern
        for template in candidates:
            pattern = self.compiled_patterns.get(template.name)
            if not pattern:
                continue

            # Try to match against original code
            if pattern.search(proposal.original_code):
                return template

            # Also try context if available
            if proposal.context and proposal.context.code_snippet:
                if pattern.search(proposal.context.code_snippet):
                    return template

        # No matching template found
        return None

    def _extract_context(
        self,
        full_code: str,
        proposal: FixProposal,
        template: FixTemplate
    ) -> Optional[Dict]:
        """
        Extract context variables from code for template.

        Uses regex pattern matching to extract named groups.
        """
        pattern = self.compiled_patterns.get(template.name)
        if not pattern:
            return None

        # Try to match pattern in original code
        match = pattern.search(proposal.original_code)
        if not match:
            # Try in full context
            if proposal.context:
                context_code = '\n'.join(
                    proposal.context.lines_before +
                    [proposal.original_code] +
                    proposal.context.lines_after
                )
                match = pattern.search(context_code)

        if not match:
            return None

        # Extract named groups as context
        context = match.groupdict()

        # Add intelligent defaults for missing extractors
        context = self._enrich_context(context, proposal, template)

        return context

    def _enrich_context(
        self,
        context: Dict,
        proposal: FixProposal,
        template: FixTemplate
    ) -> Dict:
        """
        Enrich context with intelligent defaults.

        Example: If template needs 'error_handling' but it's not in context,
        provide a sensible default based on context.
        """
        enriched = context.copy()

        # Error handling defaults
        if 'error_handling' not in enriched:
            # Provide default error handling based on context
            if proposal.context and proposal.context.in_try_block:
                enriched['error_handling'] = 'raise'
            else:
                # Guess based on what's being operated on
                if 'var' in enriched:
                    enriched['error_handling'] = f"{enriched['var']} = None"
                else:
                    enriched['error_handling'] = 'pass  # TODO: Handle error'

        # Operation description
        if 'operation' not in enriched:
            # Derive from template category
            if template.category == 'error_handling':
                if 'open(' in proposal.original_code:
                    enriched['operation'] = 'open file'
                elif 'json.' in proposal.original_code:
                    enriched['operation'] = 'parse JSON'
                elif 'requests.' in proposal.original_code:
                    enriched['operation'] = 'make HTTP request'
                else:
                    enriched['operation'] = 'perform operation'

        # Body extraction for context managers
        if 'body' not in enriched:
            # Try to extract from original code (multi-line)
            original_lines = proposal.original_code.split('\n')
            if len(original_lines) > 1:
                # Has multiple lines - use them as body
                enriched['body'] = '\n'.join(original_lines[1:])
            elif proposal.context and proposal.context.lines_after:
                # Extract lines after the problematic line
                body_lines = []
                base_indent = len(proposal.original_code) - len(proposal.original_code.lstrip())

                for line in proposal.context.lines_after:
                    # Stop at dedent or blank line
                    if line.strip():
                        line_indent = len(line) - len(line.lstrip())
                        if line_indent <= base_indent:
                            break
                    body_lines.append(line)

                if body_lines:
                    enriched['body'] = '\n'.join(body_lines)
            else:
                # Default empty body
                enriched['body'] = 'pass  # TODO: Add body'

        # Method extraction (for json.load vs json.loads)
        if 'method' not in enriched and 'json.' in proposal.original_code:
            if 'json.load(' in proposal.original_code:
                enriched['method'] = 'load'
            elif 'json.loads(' in proposal.original_code:
                enriched['method'] = 'loads'

        # Return type for docstrings
        if 'return_doc' not in enriched:
            enriched['return_doc'] = 'Result of operation'

        # Summary for docstrings
        if 'summary' not in enriched and 'func_name' in enriched:
            func_name = enriched['func_name']
            # Convert snake_case to human readable
            summary = func_name.replace('_', ' ').capitalize()
            enriched['summary'] = summary + '.'

        # Param docs for docstrings
        if 'param_docs' not in enriched and 'params' in enriched:
            params = enriched['params'].split(',')
            param_docs = []
            for param in params:
                param = param.strip()
                if param and param != 'self':
                    # Extract param name (before : or =)
                    param_name = param.split(':')[0].split('=')[0].strip()
                    param_docs.append(f"{param_name}: TODO: Document parameter")
            enriched['param_docs'] = '\n{indent}        '.join(param_docs)

        # Return annotation
        if 'return_annotation' not in enriched and 'return_type' in enriched:
            enriched['return_annotation'] = f" -> {enriched['return_type']}"
        elif 'return_annotation' not in enriched:
            enriched['return_annotation'] = ''

        return enriched

    def _apply_template(
        self,
        template: FixTemplate,
        context: Dict,
        full_code: str,
        proposal: FixProposal
    ) -> Optional[str]:
        """
        Apply template with context, preserving indentation.

        Steps:
        1. Format template with context variables
        2. Preserve original indentation
        3. Add required imports
        4. Replace in full code
        """
        try:
            # Format template
            formatted = template.template.format(**context)

            # Find and replace in full code
            original = proposal.original_code

            # If we have context, replace in context
            if proposal.context and proposal.context.code_snippet:
                original_snippet = proposal.context.code_snippet
                fixed_snippet = original_snippet.replace(original, formatted)
                fixed_code = full_code.replace(original_snippet, fixed_snippet)
            else:
                # Direct replacement
                fixed_code = full_code.replace(original, formatted)

            # Add required imports
            if template.required_imports:
                fixed_code = self._add_imports(fixed_code, template.required_imports)

            return fixed_code

        except KeyError as e:
            # Missing context variable
            print(f"Warning: Missing context variable {e} for template {template.name}")
            return None
        except Exception as e:
            print(f"Warning: Failed to apply template {template.name}: {e}")
            return None

    def _add_imports(self, code: str, imports: List[str]) -> str:
        """
        Add imports to code if not already present.

        Smart import insertion:
        - After existing imports
        - Before first non-import statement
        - Preserves blank lines
        - Merges into existing from imports when possible
        """
        lines = code.split('\n')
        existing_imports = set()
        existing_from_imports = {}  # module -> list of names

        # Find where imports end
        import_end_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip docstrings and comments at top
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith('#'):
                continue

            # Track imports
            if stripped.startswith('import ') or stripped.startswith('from '):
                existing_imports.add(stripped)
                import_end_idx = i + 1

                # Parse from imports
                if stripped.startswith('from '):
                    # "from datetime import datetime" -> module="datetime", names=["datetime"]
                    parts = stripped.split()
                    if len(parts) >= 4 and parts[2] == 'import':
                        module = parts[1]
                        names = [n.strip(',') for n in parts[3:]]
                        if module not in existing_from_imports:
                            existing_from_imports[module] = set()
                        existing_from_imports[module].update(names)

            # Stop at first non-import
            elif stripped and not stripped.startswith('#'):
                break

        # Check which imports are needed
        imports_to_add = []
        modified_lines = False

        for imp in imports:
            imp_stripped = imp.strip()

            # Handle single name imports (like "timezone")
            # These should be merged into existing from imports
            if not imp_stripped.startswith('import ') and not imp_stripped.startswith('from '):
                # This is just a name, need to infer the module
                # For now, skip - templates should provide full import
                continue

            # Check if already present
            already_exists = False
            for existing in existing_imports:
                if imp_stripped in existing or existing in imp_stripped:
                    already_exists = True
                    break

            if not already_exists:
                # Check if we can merge into existing from import
                if imp_stripped.startswith('from '):
                    parts = imp_stripped.split()
                    if len(parts) >= 4 and parts[2] == 'import':
                        module = parts[1]
                        names = [n.strip(',') for n in parts[3:]]

                        if module in existing_from_imports:
                            # Merge into existing
                            for i, line in enumerate(lines):
                                if line.strip().startswith(f'from {module} import '):
                                    # Update this line
                                    existing_names = existing_from_imports[module]
                                    all_names = sorted(existing_names | set(names))
                                    lines[i] = f'from {module} import {", ".join(all_names)}'
                                    modified_lines = True
                                    break
                        else:
                            imports_to_add.append(imp_stripped)
                else:
                    imports_to_add.append(imp_stripped)

        if not imports_to_add and not modified_lines:
            return code

        if modified_lines and not imports_to_add:
            return '\n'.join(lines)

        # Insert imports after existing imports
        insert_idx = import_end_idx if import_end_idx > 0 else 0

        # Add blank line before if needed
        if insert_idx > 0 and lines[insert_idx - 1].strip():
            imports_to_add.insert(0, '')

        # Add blank line after if needed
        if insert_idx < len(lines) and lines[insert_idx].strip():
            imports_to_add.append('')

        # Insert
        lines = lines[:insert_idx] + imports_to_add + lines[insert_idx:]

        return '\n'.join(lines)

    def _generate_diff(
        self,
        original: str,
        fixed: str,
        proposal: FixProposal
    ) -> str:
        """
        Generate unified diff.

        Shows what changed in human-readable format.
        """
        original_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)

        diff = unified_diff(
            original_lines,
            fixed_lines,
            fromfile=f"{proposal.context.file_path} (original)" if proposal.context else "original",
            tofile=f"{proposal.context.file_path} (fixed)" if proposal.context else "fixed",
            lineterm=''
        )

        return ''.join(diff)

    def can_fix(self, proposal: FixProposal) -> bool:
        """
        Check if this fixer can handle the proposal.

        Returns:
            True if proposal uses TEMPLATE strategy and we have a template
        """
        if proposal.fix_strategy != FixStrategy.TEMPLATE:
            return False

        # Check if we have templates for this category
        templates = get_templates_for_category(proposal.issue_category)
        return len(templates) > 0

    def get_available_fixes(self, category: str) -> List[str]:
        """Get list of available template fixes for category"""
        templates = get_templates_for_category(category)
        return [t.name for t in templates]


# Convenience functions

async def fix_with_template(
    proposal: FixProposal,
    full_code: str
) -> Optional[Tuple[str, str]]:
    """
    Convenience function to apply template fix.

    Args:
        proposal: Fix proposal
        full_code: Full source code

    Returns:
        (fixed_code, diff) or None
    """
    fixer = TemplateFixer()
    return await fixer.fix_issue(proposal, full_code)


def list_available_templates() -> Dict[str, List[str]]:
    """
    List all available templates grouped by category.

    Returns:
        {category: [template_names]}
    """
    result = {}
    for template in TEMPLATES.values():
        if template.category not in result:
            result[template.category] = []
        result[template.category].append(template.name)
    return result
