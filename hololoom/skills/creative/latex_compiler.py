"""LaTeX Compiler Skill - Professional document generation"""

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from hololoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)


class LaTeXCompilerSkill(BaseSkill):
    """Professional document generation with LaTeX."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="latex_compiler",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CREATIVE,
            tags=["latex", "document", "pdf", "typesetting", "academic"],
            description_short="Professional document generation with LaTeX",
            description_long=(
                "Compile LaTeX documents to PDF, DVI, or other formats. "
                "Supports custom templates, bibliography management (BibTeX), "
                "multi-pass compilation, package installation, and error diagnostics. "
                "Ideal for academic papers, reports, presentations (Beamer), and "
                "professional documentation with precise typographical control."
            ),
            requires_code_execution=True,
            requires_filesystem_read=True,
            requires_filesystem_write=True,
            external_dependencies=["texlive", "pdflatex", "bibtex"],
            expected_latency_ms="2000-30000ms",
            token_usage="100-500 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = [
            "compile_pdf", "compile_dvi", "compile_with_bib",
            "from_template", "diagnostics", "install_package"
        ]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if skill_input.operation in ["compile_pdf", "compile_dvi", "compile_with_bib"]:
            if "source" not in skill_input.parameters:
                raise ValueError(f"{skill_input.operation} requires 'source' parameter")

        if skill_input.operation == "from_template":
            if "template" not in skill_input.parameters:
                raise ValueError("from_template requires 'template' parameter")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            result = await self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"LaTeX '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"LaTeX compilation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _dispatch(self, operation: str, parameters: Dict[str, Any]):
        if operation == "compile_pdf":
            return await self._compile_pdf(parameters)
        elif operation == "compile_dvi":
            return await self._compile_dvi(parameters)
        elif operation == "compile_with_bib":
            return await self._compile_with_bib(parameters)
        elif operation == "from_template":
            return await self._from_template(parameters)
        elif operation == "diagnostics":
            return await self._diagnostics(parameters)
        elif operation == "install_package":
            return await self._install_package(parameters)

    async def _run_latex(self, command: List[str], cwd: Optional[str] = None) -> tuple[str, str, int]:
        """Run LaTeX command and return stdout, stderr, returncode."""
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        stdout, stderr = await proc.communicate()
        return (
            stdout.decode('utf-8', errors='ignore'),
            stderr.decode('utf-8', errors='ignore'),
            proc.returncode
        )

    def _parse_latex_log(self, log_content: str) -> Dict[str, Any]:
        """Parse LaTeX .log file for errors, warnings, and page count."""
        errors = []
        warnings = []
        pages = 0

        for line in log_content.split('\n'):
            # Extract error messages
            if line.startswith('!'):
                error_match = re.search(r'! (.+)', line)
                if error_match:
                    errors.append(error_match.group(1))

            # Extract warnings
            if 'Warning:' in line or 'LaTeX Warning:' in line:
                warning_match = re.search(r'Warning: (.+)', line)
                if warning_match:
                    warnings.append(warning_match.group(1))

            # Extract page count
            if 'Output written on' in line:
                page_match = re.search(r'\((\d+) page', line)
                if page_match:
                    pages = int(page_match.group(1))

        return {
            "errors": errors,
            "warnings": warnings,
            "pages": pages,
            "error_count": len(errors),
            "warning_count": len(warnings)
        }

    async def _compile_pdf(self, params):
        """Compile LaTeX to PDF."""
        source = params["source"]
        output = params.get("output", "output.pdf")
        passes = params.get("passes", 2)  # Multi-pass for references

        # Get working directory from source file path
        source_path = Path(source)
        cwd = str(source_path.parent) if source_path.parent != Path('.') else None

        # Base name without extension
        base_name = source_path.stem

        # Run pdflatex multiple passes for cross-references
        log_data = None
        for pass_num in range(passes):
            cmd = [
                "pdflatex",
                "-interaction=nonstopmode",  # Don't stop on errors
                "-file-line-error",          # Better error messages
                source_path.name
            ]

            stdout, stderr, returncode = await self._run_latex(cmd, cwd=cwd)

            # Read log file after final pass
            if pass_num == passes - 1:
                log_file = source_path.with_suffix('.log')
                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        log_data = self._parse_latex_log(f.read())

        # Get output PDF size
        pdf_path = source_path.with_suffix('.pdf')
        size_kb = 0
        if pdf_path.exists():
            size_kb = os.path.getsize(pdf_path) / 1024

        return {
            "operation": "compile_pdf",
            "source": source,
            "output": str(pdf_path),
            "passes": passes,
            "pages": log_data["pages"] if log_data else 0,
            "warnings": log_data["warning_count"] if log_data else 0,
            "errors": log_data["error_count"] if log_data else 0,
            "size_kb": round(size_kb, 2),
            "success": returncode == 0 and (log_data["error_count"] == 0 if log_data else True)
        }

    async def _compile_dvi(self, params):
        """Compile LaTeX to DVI."""
        source = params["source"]
        output = params.get("output", "output.dvi")

        source_path = Path(source)
        cwd = str(source_path.parent) if source_path.parent != Path('.') else None

        # Run latex command (not pdflatex)
        cmd = [
            "latex",
            "-interaction=nonstopmode",
            "-file-line-error",
            source_path.name
        ]

        stdout, stderr, returncode = await self._run_latex(cmd, cwd=cwd)

        # Parse log file
        log_file = source_path.with_suffix('.log')
        log_data = None
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_data = self._parse_latex_log(f.read())

        return {
            "operation": "compile_dvi",
            "source": source,
            "output": str(source_path.with_suffix('.dvi')),
            "pages": log_data["pages"] if log_data else 0,
            "warnings": log_data["warning_count"] if log_data else 0,
            "errors": log_data["error_count"] if log_data else 0,
            "success": returncode == 0 and (log_data["error_count"] == 0 if log_data else True)
        }

    async def _compile_with_bib(self, params):
        """Compile with bibliography (LaTeX → BibTeX → LaTeX × 2)."""
        source = params["source"]
        bib_file = params.get("bib_file", "references.bib")
        output = params.get("output", "output.pdf")

        source_path = Path(source)
        cwd = str(source_path.parent) if source_path.parent != Path('.') else None
        base_name = source_path.stem

        # 4-pass compilation for bibliography
        # Pass 1: pdflatex (generates .aux file with citations)
        cmd1 = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", source_path.name]
        await self._run_latex(cmd1, cwd=cwd)

        # Pass 2: bibtex (processes .aux and .bib files)
        cmd2 = ["bibtex", base_name]
        await self._run_latex(cmd2, cwd=cwd)

        # Pass 3: pdflatex (incorporates bibliography)
        cmd3 = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", source_path.name]
        await self._run_latex(cmd3, cwd=cwd)

        # Pass 4: pdflatex (finalizes cross-references)
        cmd4 = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", source_path.name]
        stdout, stderr, returncode = await self._run_latex(cmd4, cwd=cwd)

        # Parse final log
        log_file = source_path.with_suffix('.log')
        log_data = None
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
                log_data = self._parse_latex_log(log_content)

                # Count citations from log
                citations = len(re.findall(r'\\citation{', log_content))
                log_data["citations"] = citations

        return {
            "operation": "compile_with_bib",
            "source": source,
            "bib_file": bib_file,
            "output": str(source_path.with_suffix('.pdf')),
            "passes": 4,
            "citations": log_data.get("citations", 0) if log_data else 0,
            "pages": log_data["pages"] if log_data else 0,
            "warnings": log_data["warning_count"] if log_data else 0,
            "errors": log_data["error_count"] if log_data else 0,
            "success": returncode == 0 and (log_data["error_count"] == 0 if log_data else True)
        }

    async def _from_template(self, params):
        """Generate document from template."""
        template = params["template"]  # e.g., "article", "beamer", "report"
        title = params.get("title", "Untitled Document")
        author = params.get("author", "Anonymous")
        variables = params.get("variables", {})
        output_dir = params.get("output_dir", ".")

        # Template definitions
        templates = {
            "article": r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\maketitle
{{content}}
\end{{document}}""",
            "beamer": r"""\documentclass{{beamer}}
\usepackage[utf8]{{inputenc}}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\frame{{\titlepage}}
{{content}}
\end{{document}}""",
            "report": r"""\documentclass{{report}}
\usepackage[utf8]{{inputenc}}
\title{{{title}}}
\author{{{author}}}
\date{{\today}}
\begin{{document}}
\maketitle
\tableofcontents
{{content}}
\end{{document}}"""
        }

        if template not in templates:
            raise ValueError(f"Unknown template: {template}. Available: {', '.join(templates.keys())}")

        # Generate LaTeX source from template
        latex_source = templates[template].format(
            title=title,
            author=author,
            content=variables.get("content", r"\section{Introduction}\nYour content here...")
        )

        # Write generated source file
        generated_file = Path(output_dir) / f"{template}_generated.tex"
        with open(generated_file, 'w', encoding='utf-8') as f:
            f.write(latex_source)

        # Compile to PDF
        compile_result = await self._compile_pdf({
            "source": str(generated_file),
            "passes": 2
        })

        return {
            "operation": "from_template",
            "template": template,
            "title": title,
            "author": author,
            "variables": variables,
            "generated_source": str(generated_file),
            "output": compile_result["output"],
            "success": compile_result["success"]
        }

    async def _diagnostics(self, params):
        """Analyze LaTeX errors and warnings."""
        log_file = params.get("log_file", "document.log")

        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Log file not found: {log_file}")

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        errors = []
        warnings = []

        # Parse errors with line numbers
        for match in re.finditer(r'l\.(\d+)\s+(.+?)(?:\n|$)', log_content):
            line_num = int(match.group(1))
            message = match.group(2).strip()
            errors.append({
                "line": line_num,
                "message": message,
                "severity": "error"
            })

        # Parse LaTeX warnings
        for match in re.finditer(r'LaTeX Warning: (.+?) on input line (\d+)', log_content):
            message = match.group(1).strip()
            line_num = int(match.group(2))
            warnings.append({
                "line": line_num,
                "message": message,
                "severity": "warning"
            })

        # Parse overfull/underfull boxes
        for match in re.finditer(r'(Overfull|Underfull) \\hbox \((.+?)\) (?:in paragraph at lines (\d+)--(\d+)|detected at line (\d+))', log_content):
            box_type = match.group(1)
            amount = match.group(2)
            line_num = int(match.group(3) or match.group(5) or 0)
            warnings.append({
                "line": line_num,
                "message": f"{box_type} hbox ({amount})",
                "severity": "warning"
            })

        return {
            "operation": "diagnostics",
            "log_file": log_file,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "success": True
        }

    async def _install_package(self, params):
        """Install LaTeX package (tlmgr)."""
        package = params["package"]

        # Run tlmgr install command
        cmd = ["tlmgr", "install", package]

        stdout, stderr, returncode = await self._run_latex(cmd)

        # Check if already installed
        already_installed = "already installed" in stdout.lower()

        return {
            "operation": "install_package",
            "package": package,
            "installed": returncode == 0 or already_installed,
            "already_installed": already_installed,
            "message": f"Package '{package}' installed successfully" if returncode == 0 else f"Failed to install '{package}': {stderr}"
        }


from hololoom.skills.base import register_skill
register_skill(LaTeXCompilerSkill())
