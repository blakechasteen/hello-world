"""
LSP Integration Skill - Deep code intelligence via Language Server Protocol
"""

import asyncio
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from hololoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)

# Check for jedi availability (Python code intelligence)
try:
    import jedi
    JEDI_AVAILABLE = True
except ImportError:
    JEDI_AVAILABLE = False

# Check for mypy availability (type checking)
try:
    import mypy
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False


@dataclass
class CodeSymbol:
    """Code symbol (function, class, variable)."""
    name: str
    kind: str  # function, class, method, variable, etc.
    location: str  # file:line:col
    detail: Optional[str] = None


class LSPIntegrationSkill(BaseSkill):
    """LSP integration for deep code intelligence."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="lsp_integration",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CODE,
            tags=["lsp", "code", "intelligence", "analysis", "navigation"],
            description_short="Language Server Protocol integration for deep code intelligence",
            description_long=(
                "LSP integration provides deep code understanding: goto definition, "
                "find references, hover documentation, symbol search, code completion, "
                "diagnostics, and refactoring support. Works with any LSP-compatible "
                "language server (Python: pyright/pylsp, TypeScript: tsserver, etc.)."
            ),
            requires_code_execution=True,
            requires_filesystem_read=True,
            external_dependencies=["Language servers (pyright, rust-analyzer, etc.)"],
            expected_latency_ms="50-500ms",
            token_usage="100-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = [
            "goto_definition", "find_references", "hover", "symbols",
            "completion", "diagnostics", "rename", "format"
        ]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if skill_input.operation in ["goto_definition", "find_references", "hover", "completion"]:
            if "file" not in skill_input.parameters or "position" not in skill_input.parameters:
                raise ValueError(f"{skill_input.operation} requires 'file' and 'position' parameters")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        # Check for jedi availability for most operations
        needs_jedi = skill_input.operation in [
            "goto_definition", "find_references", "hover", "symbols", "completion"
        ]

        if needs_jedi and not JEDI_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="jedi not available. Install with: pip install jedi",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependency: jedi"]
            )

        try:
            result = await self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"LSP operation '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"LSP operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _dispatch(self, operation: str, parameters: Dict[str, Any]):
        if operation == "goto_definition":
            return await self._goto_definition(parameters)
        elif operation == "find_references":
            return await self._find_references(parameters)
        elif operation == "hover":
            return await self._hover(parameters)
        elif operation == "symbols":
            return await self._symbols(parameters)
        elif operation == "completion":
            return await self._completion(parameters)
        elif operation == "diagnostics":
            return await self._diagnostics(parameters)
        elif operation == "rename":
            return await self._rename(parameters)
        elif operation == "format":
            return await self._format(parameters)

    async def _goto_definition(self, params):
        """Find definition of symbol at position using jedi."""
        file_path = params["file"]
        line, col = params["position"]["line"], params["position"]["character"]

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to find definition
        script = jedi.Script(source, path=file_path)
        definitions = script.goto(line, col)

        if not definitions:
            return {
                "file": file_path,
                "found": False,
                "message": "No definition found"
            }

        # Return first definition
        defn = definitions[0]
        return {
            "file": str(defn.module_path) if defn.module_path else file_path,
            "position": {"line": defn.line, "character": defn.column},
            "symbol": defn.name,
            "kind": defn.type,
            "found": True
        }

    async def _find_references(self, params):
        """Find all references to symbol at position using jedi."""
        file_path = params["file"]
        line, col = params["position"]["line"], params["position"]["character"]

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to find references
        script = jedi.Script(source, path=file_path)
        references = script.get_references(line, col)

        ref_list = []
        for ref in references:
            ref_list.append({
                "file": str(ref.module_path) if ref.module_path else file_path,
                "line": ref.line,
                "character": ref.column
            })

        return {
            "symbol": references[0].name if references else "unknown",
            "references": ref_list,
            "count": len(ref_list)
        }

    async def _hover(self, params):
        """Get hover documentation for symbol at position using jedi."""
        file_path = params["file"]
        line, col = params["position"]["line"], params["position"]["character"]

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to get help
        script = jedi.Script(source, path=file_path)
        completions = script.help(line, col)

        if not completions:
            return {
                "symbol": "unknown",
                "found": False
            }

        comp = completions[0]
        return {
            "symbol": comp.name,
            "type": comp.type,
            "documentation": comp.docstring() if hasattr(comp, 'docstring') else "",
            "signature": comp.full_name if hasattr(comp, 'full_name') else comp.name,
            "found": True
        }

    async def _symbols(self, params):
        """Get all symbols in file using jedi."""
        file_path = params.get("file")
        workspace = params.get("workspace", False)

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to get symbols
        script = jedi.Script(source, path=file_path)
        names = script.get_names(all_scopes=True)

        symbols = []
        for name in names:
            symbols.append({
                "name": name.name,
                "kind": name.type,
                "location": f"{file_path}:{name.line}:{name.column}",
                "detail": name.full_name if hasattr(name, 'full_name') else name.name
            })

        return {
            "file": file_path if not workspace else "workspace",
            "symbols": symbols,
            "count": len(symbols)
        }

    async def _completion(self, params):
        """Get code completion suggestions at position using jedi."""
        file_path = params["file"]
        line, col = params["position"]["line"], params["position"]["character"]

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to get completions
        script = jedi.Script(source, path=file_path)
        completions = script.complete(line, col)

        completion_list = []
        for comp in completions[:20]:  # Limit to 20 completions
            completion_list.append({
                "label": comp.name,
                "kind": comp.type,
                "detail": comp.complete if hasattr(comp, 'complete') else comp.name,
                "documentation": comp.docstring(raw=True)[:200] if hasattr(comp, 'docstring') else ""
            })

        return {
            "completions": completion_list,
            "count": len(completion_list)
        }

    async def _diagnostics(self, params):
        """Get diagnostics using mypy or pylint subprocess."""
        file_path = params["file"]
        tool = params.get("tool", "mypy")  # mypy or pylint

        diagnostics = []
        error_count = 0
        warning_count = 0

        try:
            if tool == "mypy":
                # Run mypy
                proc = await asyncio.create_subprocess_exec(
                    "mypy", file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                output = stdout.decode('utf-8', errors='ignore')

                # Parse mypy output: file.py:line:col: error: message
                for line in output.split('\n'):
                    if ':' in line and file_path in line:
                        parts = line.split(':', 4)
                        if len(parts) >= 4:
                            line_num = int(parts[1]) if parts[1].isdigit() else 0
                            col_num = int(parts[2]) if parts[2].isdigit() else 0
                            severity = "error" if "error" in parts[3] else "warning"
                            message = parts[4].strip() if len(parts) > 4 else parts[3].strip()

                            diagnostics.append({
                                "line": line_num,
                                "character": col_num,
                                "severity": severity,
                                "message": message,
                                "source": "mypy"
                            })

                            if severity == "error":
                                error_count += 1
                            else:
                                warning_count += 1

            elif tool == "pylint":
                # Run pylint
                proc = await asyncio.create_subprocess_exec(
                    "pylint", file_path, "--output-format=json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                output = stdout.decode('utf-8', errors='ignore')

                try:
                    messages = json.loads(output)
                    for msg in messages:
                        severity = "error" if msg["type"] == "error" else "warning"
                        diagnostics.append({
                            "line": msg.get("line", 0),
                            "character": msg.get("column", 0),
                            "severity": severity,
                            "message": msg.get("message", ""),
                            "source": "pylint"
                        })

                        if severity == "error":
                            error_count += 1
                        else:
                            warning_count += 1
                except json.JSONDecodeError:
                    pass

        except FileNotFoundError:
            return {
                "file": file_path,
                "diagnostics": [],
                "error_count": 0,
                "warning_count": 0,
                "message": f"{tool} not found. Install with: pip install {tool}"
            }

        return {
            "file": file_path,
            "tool": tool,
            "diagnostics": diagnostics,
            "error_count": error_count,
            "warning_count": warning_count
        }

    async def _rename(self, params):
        """Rename symbol using text search/replace."""
        file_path = params["file"]
        line, col = params["position"]["line"], params["position"]["character"]
        new_name = params["new_name"]

        # Read source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use jedi to find symbol name at position
        script = jedi.Script(source, path=file_path)
        definitions = script.goto(line, col)

        if not definitions:
            return {
                "old_name": "unknown",
                "new_name": new_name,
                "changes": [],
                "change_count": 0,
                "message": "Symbol not found at position"
            }

        old_name = definitions[0].name

        # Simple text replacement (production would use AST)
        lines = source.split('\n')
        changes = []

        for line_idx, line_content in enumerate(lines):
            if old_name in line_content:
                lines[line_idx] = line_content.replace(old_name, new_name)
                changes.append({
                    "file": file_path,
                    "line": line_idx + 1,
                    "old": old_name,
                    "new": new_name
                })

        # Write renamed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return {
            "old_name": old_name,
            "new_name": new_name,
            "changes": changes,
            "change_count": len(changes)
        }

    async def _format(self, params):
        """Format code in file using black or autopep8."""
        file_path = params["file"]
        formatter = params.get("formatter", "black")  # black, autopep8, etc.

        try:
            if formatter == "black":
                # Run black formatter
                proc = await asyncio.create_subprocess_exec(
                    "black", file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                success = proc.returncode == 0
                message = stdout.decode('utf-8', errors='ignore') if success else stderr.decode('utf-8', errors='ignore')

            elif formatter == "autopep8":
                # Run autopep8 formatter (in-place)
                proc = await asyncio.create_subprocess_exec(
                    "autopep8", "--in-place", file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                success = proc.returncode == 0
                message = f"File formatted with autopep8" if success else stderr.decode('utf-8', errors='ignore')

            else:
                return {
                    "file": file_path,
                    "formatter": formatter,
                    "changes_made": False,
                    "message": f"Unknown formatter: {formatter}. Use 'black' or 'autopep8'"
                }

            return {
                "file": file_path,
                "formatter": formatter,
                "changes_made": success,
                "message": message.strip()
            }

        except FileNotFoundError:
            return {
                "file": file_path,
                "formatter": formatter,
                "changes_made": False,
                "message": f"{formatter} not found. Install with: pip install {formatter}"
            }


from hololoom.skills.base import register_skill
register_skill(LSPIntegrationSkill())
