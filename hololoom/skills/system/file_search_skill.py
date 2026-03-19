"""File Search Skill - Fast file content search using ripgrep"""

import asyncio
import json
import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)

# Check for ripgrep availability
RIPGREP_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(['rg', '--version'], capture_output=True, timeout=1)
    RIPGREP_AVAILABLE = result.returncode == 0
except Exception:
    pass


class FileSearchSkill(BaseSkill):
    """Fast file content search using ripgrep (or fallback to grep)."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="file_search",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.SYSTEM,
            tags=["search", "grep", "ripgrep", "files", "text"],
            description_short="Fast file content search",
            description_long=(
                "Search file contents using ripgrep (rg) or grep. "
                "Supports regex patterns, file type filtering, context lines, "
                "and JSON output. Ripgrep is significantly faster than grep "
                "for large codebases. Falls back to standard grep if ripgrep "
                "is not available."
            ),
            requires_code_execution=True,
            requires_filesystem_read=True,
            external_dependencies=["ripgrep (optional)", "grep"],
            expected_latency_ms="50-500ms",
            token_usage="100-1000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["search", "search_files", "count_matches", "list_files"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if "pattern" not in skill_input.parameters:
            raise ValueError("'pattern' parameter is required")

        if "path" not in skill_input.parameters:
            raise ValueError("'path' parameter is required")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            result = await self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"File search '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"File search failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _dispatch(self, operation: str, parameters: dict[str, Any]):
        if operation == "search":
            return await self._search(parameters)
        elif operation == "search_files":
            return await self._search_files(parameters)
        elif operation == "count_matches":
            return await self._count_matches(parameters)
        elif operation == "list_files":
            return await self._list_files(parameters)

    async def _run_ripgrep(self, pattern: str, path: str, args: list[str]) -> tuple:
        """Run ripgrep command."""
        cmd = ['rg', '--json'] + args + [pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return proc.returncode, stdout, stderr

    async def _run_grep(self, pattern: str, path: str, args: list[str]) -> tuple:
        """Fallback to grep command."""
        cmd = ['grep', '-r'] + args + [pattern, path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        return proc.returncode, stdout, stderr

    async def _search(self, params):
        """Search for pattern in files."""
        pattern = params["pattern"]
        path = params.get("path", ".")
        case_sensitive = params.get("case_sensitive", True)
        context_lines = params.get("context_lines", 0)
        file_type = params.get("file_type")  # e.g., "py", "js", "md"
        max_results = params.get("max_results", 100)

        # Build command arguments
        args = []
        if not case_sensitive:
            args.append('-i')
        if context_lines > 0:
            args.extend(['-C', str(context_lines)])
        if file_type:
            args.extend(['-t', file_type])
        args.extend(['--max-count', str(max_results)])

        # Use ripgrep if available, otherwise grep
        if RIPGREP_AVAILABLE:
            returncode, stdout, stderr = await self._run_ripgrep(pattern, path, args)

            # Parse JSON output
            matches = []
            for line in stdout.decode('utf-8', errors='ignore').split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('type') == 'match':
                        match_data = data.get('data', {})
                        matches.append({
                            'file': match_data.get('path', {}).get('text', ''),
                            'line_number': match_data.get('line_number'),
                            'text': match_data.get('lines', {}).get('text', '').strip(),
                            'match_start': match_data.get('submatches', [{}])[0].get('start'),
                            'match_end': match_data.get('submatches', [{}])[0].get('end')
                        })
                except json.JSONDecodeError:
                    pass
        else:
            # Fallback to grep
            returncode, stdout, stderr = await self._run_grep(pattern, path, ['-n'] + args)
            matches = []
            for line in stdout.decode('utf-8', errors='ignore').split('\n'):
                if not line:
                    continue
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    matches.append({
                        'file': parts[0],
                        'line_number': int(parts[1]) if parts[1].isdigit() else 0,
                        'text': parts[2].strip(),
                        'match_start': None,
                        'match_end': None
                    })

        return {
            "operation": "search",
            "pattern": pattern,
            "path": path,
            "matches": matches[:max_results],
            "total_matches": len(matches),
            "truncated": len(matches) > max_results,
            "tool_used": "ripgrep" if RIPGREP_AVAILABLE else "grep",
            "success": True
        }

    async def _search_files(self, params):
        """Search for files matching pattern in name."""
        pattern = params["pattern"]
        path = params.get("path", ".")
        max_results = params.get("max_results", 100)

        if RIPGREP_AVAILABLE:
            cmd = ['rg', '--files', '--glob', f"*{pattern}*", path]
        else:
            cmd = ['find', path, '-name', f"*{pattern}*", '-type', 'f']

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        files = [f for f in stdout.decode('utf-8', errors='ignore').split('\n') if f]

        return {
            "operation": "search_files",
            "pattern": pattern,
            "path": path,
            "files": files[:max_results],
            "total_files": len(files),
            "truncated": len(files) > max_results,
            "success": True
        }

    async def _count_matches(self, params):
        """Count matches per file."""
        pattern = params["pattern"]
        path = params.get("path", ".")
        file_type = params.get("file_type")

        args = ['--count']
        if file_type:
            args.extend(['-t', file_type])

        if RIPGREP_AVAILABLE:
            returncode, stdout, stderr = await self._run_ripgrep(pattern, path, args)
        else:
            returncode, stdout, stderr = await self._run_grep(pattern, path, ['-c'] + args)

        # Parse counts
        counts = {}
        for line in stdout.decode('utf-8', errors='ignore').split('\n'):
            if not line or ':' not in line:
                continue
            parts = line.rsplit(':', 1)
            if len(parts) == 2:
                file_path, count_str = parts
                try:
                    counts[file_path] = int(count_str)
                except ValueError:
                    pass

        return {
            "operation": "count_matches",
            "pattern": pattern,
            "path": path,
            "counts": counts,
            "total_files": len(counts),
            "total_matches": sum(counts.values()),
            "success": True
        }

    async def _list_files(self, params):
        """List all files in path."""
        path = params.get("path", ".")
        file_type = params.get("file_type")
        max_results = params.get("max_results", 1000)

        if RIPGREP_AVAILABLE:
            cmd = ['rg', '--files']
            if file_type:
                cmd.extend(['-t', file_type])
            cmd.append(path)
        else:
            cmd = ['find', path, '-type', 'f']

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        files = [f for f in stdout.decode('utf-8', errors='ignore').split('\n') if f]

        return {
            "operation": "list_files",
            "path": path,
            "files": files[:max_results],
            "total_files": len(files),
            "truncated": len(files) > max_results,
            "success": True
        }


from hololoom.skills.base import register_skill

register_skill(FileSearchSkill())
