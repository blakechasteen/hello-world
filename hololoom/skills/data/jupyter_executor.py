"""Jupyter Executor Skill - Run notebooks programmatically"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from HoloLoom.skills.base import (BaseSkill, SkillInput, SkillOutput, SkillMetadata,
                                   SkillCategory, SkillStatus)

# Check for nbformat availability
try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False

# Check for nbconvert availability
try:
    from nbconvert import HTMLExporter, PDFExporter, PythonExporter
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    NBCONVERT_AVAILABLE = True
except ImportError:
    NBCONVERT_AVAILABLE = False


class JupyterExecutorSkill(BaseSkill):
    """Execute Jupyter notebooks programmatically."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="jupyter_executor",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.DATA,
            tags=["jupyter", "notebooks", "data", "analysis", "ml"],
            description_short="Run Jupyter notebooks programmatically and extract results",
            description_long="Execute Jupyter notebooks, extract outputs, capture variables, run specific cells, and parameterize execution. Useful for automated data analysis and ML pipelines.",
            requires_code_execution=True,
            requires_filesystem_read=True,
            external_dependencies=["nbformat", "nbconvert", "jupyter"],
            expected_latency_ms="1000-60000ms",
            token_usage="100-5000 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = ["run_notebook", "run_cell", "extract_outputs", "parameterize", "convert"]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")
        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()

        # Check availability
        if not NBFORMAT_AVAILABLE or not NBCONVERT_AVAILABLE:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message="nbformat or nbconvert not available. Install with: pip install nbformat nbconvert jupyter",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=["Missing dependencies: nbformat, nbconvert"]
            )

        try:
            # Dispatch to operation handler
            if skill_input.operation == "run_notebook":
                result = await self._run_notebook(skill_input.parameters)
            elif skill_input.operation == "run_cell":
                result = await self._run_cell(skill_input.parameters)
            elif skill_input.operation == "extract_outputs":
                result = await self._extract_outputs(skill_input.parameters)
            elif skill_input.operation == "parameterize":
                result = await self._parameterize(skill_input.parameters)
            elif skill_input.operation == "convert":
                result = await self._convert(skill_input.parameters)
            else:
                raise ValueError(f"Unknown operation: {skill_input.operation}")

            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Jupyter '{skill_input.operation}' completed",
                execution_time_ms=(time.time() - start_time) * 1000,
                details=result
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Jupyter operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _run_notebook(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete notebook."""
        notebook_path = params["notebook_path"]
        output_path = params.get("output_path")
        timeout = params.get("timeout", 600)
        kernel_name = params.get("kernel_name", "python3")

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Create executor
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

        # Execute notebook
        start_time = time.time()
        try:
            ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
            cells_errored = 0
        except CellExecutionError as e:
            cells_errored = 1

        execution_time = time.time() - start_time

        # Save executed notebook if output path specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

        return {
            "operation": "run_notebook",
            "notebook_path": notebook_path,
            "output_path": output_path,
            "cells_executed": len(nb.cells),
            "cells_errored": cells_errored,
            "total_execution_time_seconds": round(execution_time, 2),
            "kernel_used": kernel_name
        }

    async def _run_cell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single cell."""
        notebook_path = params["notebook_path"]
        cell_index = params["cell_index"]
        timeout = params.get("timeout", 60)

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        if cell_index >= len(nb.cells):
            raise IndexError(f"Cell index {cell_index} out of range (notebook has {len(nb.cells)} cells)")

        # Get cell
        cell = nb.cells[cell_index]

        # Execute single cell
        ep = ExecutePreprocessor(timeout=timeout)
        start_time = time.time()

        try:
            ep.preprocess_cell(cell, {'metadata': {'path': str(Path(notebook_path).parent)}}, cell_index)
            success = True
            outputs = cell.get('outputs', [])
        except Exception as e:
            success = False
            outputs = []

        execution_time = time.time() - start_time

        return {
            "operation": "run_cell",
            "notebook_path": notebook_path,
            "cell_index": cell_index,
            "cell_type": cell.cell_type,
            "execution_time_seconds": round(execution_time, 2),
            "outputs": [{"output_type": o.output_type, "text": o.get('text', '')} for o in outputs],
            "success": success
        }

    async def _extract_outputs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract outputs from executed notebook."""
        notebook_path = params["notebook_path"]
        variables_to_extract = params.get("variables_to_extract", [])

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Extract cell outputs
        outputs = {}
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and cell.get('outputs'):
                cell_outputs = []
                for output in cell.outputs:
                    if output.output_type == 'stream':
                        cell_outputs.append({
                            "type": "stream",
                            "content": output.get('text', '')
                        })
                    elif output.output_type == 'execute_result':
                        cell_outputs.append({
                            "type": "execute_result",
                            "content": output.get('data', {}).get('text/plain', '')
                        })
                    elif output.output_type == 'display_data':
                        cell_outputs.append({
                            "type": "display_data",
                            "content": str(output.get('data', {}))
                        })

                if cell_outputs:
                    outputs[f"cell_{idx}"] = cell_outputs

        # Note: Variable extraction requires kernel execution state,
        # which isn't available from static notebook files
        variables = {}
        if variables_to_extract:
            variables = {var: "Variable extraction requires live kernel execution" for var in variables_to_extract}

        return {
            "operation": "extract_outputs",
            "notebook_path": notebook_path,
            "outputs": outputs,
            "variables": variables
        }

    async def _parameterize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute notebook with parameters (like papermill)."""
        notebook_path = params["notebook_path"]
        output_path = params.get("output_path", "parameterized_output.ipynb")
        parameters = params.get("parameters", {})
        timeout = params.get("timeout", 600)

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Insert parameters cell at the beginning
        params_code = "\n".join([f"{key} = {json.dumps(value)}" for key, value in parameters.items()])
        params_cell = new_code_cell(source=params_code)
        params_cell.metadata['tags'] = ['parameters']

        # Insert after first cell (or at beginning if no cells)
        if len(nb.cells) > 0:
            nb.cells.insert(1, params_cell)
        else:
            nb.cells.insert(0, params_cell)

        # Execute notebook
        ep = ExecutePreprocessor(timeout=timeout)
        start_time = time.time()

        try:
            ep.preprocess(nb, {'metadata': {'path': str(Path(notebook_path).parent)}})
            cells_errored = 0
        except CellExecutionError as e:
            cells_errored = 1

        execution_time = time.time() - start_time

        # Save executed notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        return {
            "operation": "parameterize",
            "notebook_path": notebook_path,
            "output_path": output_path,
            "parameters": parameters,
            "cells_executed": len(nb.cells),
            "cells_errored": cells_errored,
            "total_execution_time_seconds": round(execution_time, 2)
        }

    async def _convert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert notebook to different format."""
        notebook_path = params["notebook_path"]
        output_format = params["output_format"]  # html, pdf, python, markdown
        output_path = params.get("output_path")

        # Read notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # Select exporter
        if output_format == "html":
            exporter = HTMLExporter()
            default_ext = ".html"
        elif output_format == "pdf":
            exporter = PDFExporter()
            default_ext = ".pdf"
        elif output_format == "python":
            exporter = PythonExporter()
            default_ext = ".py"
        else:
            raise ValueError(f"Unsupported format: {output_format}. Use html, pdf, or python")

        # Convert
        (body, resources) = exporter.from_notebook_node(nb)

        # Determine output path
        if not output_path:
            output_path = str(Path(notebook_path).with_suffix(default_ext))

        # Write output
        with open(output_path, 'w' if output_format != 'pdf' else 'wb', encoding='utf-8' if output_format != 'pdf' else None) as f:
            f.write(body if output_format != 'pdf' else body.encode() if isinstance(body, str) else body)

        # Get file size
        file_size_kb = os.path.getsize(output_path) / 1024

        return {
            "operation": "convert",
            "notebook_path": notebook_path,
            "output_format": output_format,
            "converted_file": output_path,
            "file_size_kb": round(file_size_kb, 2)
        }


from HoloLoom.skills.base import register_skill
register_skill(JupyterExecutorSkill())
