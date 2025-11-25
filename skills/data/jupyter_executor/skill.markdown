# Skill: Jupyter Executor

## Metadata

- **Name**: `jupyter_executor`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `data`
- **Tags**: `jupyter, notebooks, data, analysis, ml, execution`

## Description

**Short Description**:
Run Jupyter notebooks programmatically and extract results for automated data analysis pipelines.

**Detailed Description**:
The Jupyter Executor skill provides comprehensive notebook execution capabilities for automated data analysis and ML pipelines. Execute complete notebooks or specific cells, extract outputs (text, images, data), capture variables, parameterize execution with custom values, and convert notebooks to different formats (HTML, PDF, Python scripts). Supports timeout handling, error capture, and execution state management. Ideal for automated reporting, ML model training, data processing workflows, and reproducible analysis.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [x] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `nbformat` (notebook file format library)
- `nbconvert` (notebook conversion library)
- `jupyter_client` or `jupyter_core` (Jupyter execution engine)
- Optional: `papermill` (parameterized notebook execution)
- Optional: `nbval` (notebook validation)

**HoloLoom Integration**: Integrates with data pipelines, ML training workflows, automated reporting, and analysis orchestration.

## Input Schema

```json
{
  "operation": "string - run_notebook|run_cell|extract_outputs|parameterize|convert",
  "parameters": {
    "notebook_path": "string (required) - Path to .ipynb notebook file",
    "output_path": "string (optional) - Path to save executed notebook",
    "cell_index": "number (required for run_cell) - Cell index to execute (0-based)",
    "timeout": "number (optional) - Execution timeout in seconds (default: 600)",
    "parameters": "object (optional for parameterize) - Parameter values {param_name: value}",
    "kernel_name": "string (optional) - Jupyter kernel to use (default: python3)",
    "output_format": "string (required for convert) - html|pdf|python|markdown|latex",
    "capture_outputs": "boolean (optional) - Capture cell outputs (default: true)",
    "stop_on_error": "boolean (optional) - Stop execution on error (default: false)",
    "variables_to_extract": "array (optional for extract_outputs) - Variable names to extract"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "notebook_path": "string - Executed notebook path",
    "cells_executed": "number - Number of cells executed",
    "cells_errored": "number - Number of cells with errors",
    "total_execution_time_seconds": "number - Notebook execution time",
    "outputs": "array - Cell outputs (text, images, data)",
    "variables": "object - Extracted variable values",
    "errors": "array - Execution errors by cell",
    "converted_file": "string - Path to converted file (for convert)",
    "kernel_used": "string - Kernel name used for execution"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Run Complete Notebook

**Input**:
```json
{
  "operation": "run_notebook",
  "parameters": {
    "notebook_path": "analysis/weekly_report.ipynb",
    "output_path": "results/weekly_report_executed.ipynb",
    "kernel_name": "python3",
    "timeout": 1800,
    "stop_on_error": false
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "run_notebook",
    "notebook_path": "analysis/weekly_report.ipynb",
    "output_path": "results/weekly_report_executed.ipynb",
    "cells_executed": 25,
    "cells_errored": 0,
    "total_execution_time_seconds": 42.5,
    "kernel_used": "python3"
  },
  "message": "Notebook executed successfully (25 cells, 42.5s)",
  "execution_time_ms": 43200
}
```

**Explanation**: Executes entire notebook with 25 cells, saves executed version with all outputs. Useful for automated weekly reporting.

### Example 2: Parameterized Execution

**Input**:
```json
{
  "operation": "parameterize",
  "parameters": {
    "notebook_path": "ml_training/train_model.ipynb",
    "output_path": "results/train_model_lr0.001.ipynb",
    "parameters": {
      "learning_rate": 0.001,
      "epochs": 50,
      "batch_size": 32,
      "model_type": "resnet50"
    },
    "timeout": 3600
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "parameterize",
    "notebook_path": "ml_training/train_model.ipynb",
    "output_path": "results/train_model_lr0.001.ipynb",
    "parameters": {
      "learning_rate": 0.001,
      "epochs": 50,
      "batch_size": 32,
      "model_type": "resnet50"
    },
    "cells_executed": 18,
    "cells_errored": 0,
    "total_execution_time_seconds": 3420.8
  },
  "message": "Parameterized notebook executed (18 cells, 57m)",
  "execution_time_ms": 3421500
}
```

**Explanation**: Executes ML training notebook with custom hyperparameters. Enables hyperparameter sweeps and experiment tracking.

### Example 3: Extract Specific Outputs

**Input**:
```json
{
  "operation": "extract_outputs",
  "parameters": {
    "notebook_path": "results/analysis_executed.ipynb",
    "variables_to_extract": ["final_accuracy", "confusion_matrix", "feature_importance"],
    "output_types": ["text", "image", "data"]
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "extract_outputs",
    "notebook_path": "results/analysis_executed.ipynb",
    "outputs": {
      "cell_5": {
        "type": "text",
        "content": "Final accuracy: 0.945"
      },
      "cell_8": {
        "type": "image/png",
        "content_base64": "iVBORw0KGgoAAAANSUhEUg..."
      }
    },
    "variables": {
      "final_accuracy": 0.945,
      "confusion_matrix": [[95, 5], [3, 97]],
      "feature_importance": [0.35, 0.28, 0.18, 0.12, 0.07]
    }
  },
  "message": "Extracted 3 variables and 2 cell outputs",
  "execution_time_ms": 850
}
```

**Explanation**: Extracts specific variables and outputs from executed notebook for downstream processing or reporting.

### Example 4: Run Single Cell

**Input**:
```json
{
  "operation": "run_cell",
  "parameters": {
    "notebook_path": "data_processing/etl_pipeline.ipynb",
    "cell_index": 12,
    "timeout": 300,
    "capture_outputs": true
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "run_cell",
    "notebook_path": "data_processing/etl_pipeline.ipynb",
    "cell_index": 12,
    "cell_type": "code",
    "execution_time_seconds": 45.2,
    "outputs": [
      {
        "output_type": "stream",
        "text": "Processing 15,000 records...\nETL complete. 14,987 records processed successfully."
      }
    ],
    "success": true
  },
  "message": "Cell 12 executed successfully (45.2s)",
  "execution_time_ms": 45800
}
```

**Explanation**: Executes a single cell for debugging or partial execution. Useful for iterative development and testing specific steps.

### Example 5: Convert to HTML Report

**Input**:
```json
{
  "operation": "convert",
  "parameters": {
    "notebook_path": "results/weekly_report_executed.ipynb",
    "output_format": "html",
    "output_path": "reports/weekly_report.html",
    "template": "classic",
    "embed_images": true
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "convert",
    "notebook_path": "results/weekly_report_executed.ipynb",
    "output_format": "html",
    "converted_file": "reports/weekly_report.html",
    "file_size_kb": 245,
    "embedded_images": 8
  },
  "message": "Notebook converted to HTML: reports/weekly_report.html",
  "execution_time_ms": 1250
}
```

**Explanation**: Converts executed notebook to standalone HTML report with embedded images. Perfect for sharing analysis results with stakeholders.

## Testing Checklist

- [x] **Functionality**: All 5 operations execute correctly
- [x] **Error Handling**: Graceful handling of execution errors, timeouts, missing notebooks
- [x] **Security**: No arbitrary code execution without validation
- [x] **Performance**: Operations complete within expected time (<60s for typical notebooks)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Jupyter ecosystem documented
- [x] **Edge Cases**: Handles kernel crashes, infinite loops, large outputs
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom data processing and ML pipelines

## Security Considerations

**Potential Risks**:
- **Arbitrary Code Execution**: Notebooks can execute any Python code -> Validate notebook source, run in sandboxed environment
- **Resource Exhaustion**: Infinite loops or memory leaks -> Implement timeouts and memory limits
- **Sensitive Data**: Notebooks may contain credentials -> Never log notebook outputs, sanitize extracted data

**Data Privacy**:
- [x] Does not log notebook cell contents
- [x] Does not expose extracted variables outside designated scope
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Kernel execution isolated from host system
- [x] Timeout enforcement prevents runaway execution

## Performance Characteristics

- **Expected Latency**: 1000-60000ms (1-60 seconds depending on notebook complexity)
- **Token Usage**: 100-5000 tokens per execution
- **Resource Requirements**: Jupyter kernel, sufficient memory for data processing
- **Scalability**: Limited by kernel resources and notebook complexity

**Operation-Specific Latencies**:
- `run_notebook`: 1000-60000ms (depends on number of cells and computation)
- `run_cell`: 100-10000ms (depends on cell computation)
- `extract_outputs`: 200-2000ms (depends on output size)
- `parameterize`: Similar to run_notebook + parameter injection overhead
- `convert`: 500-5000ms (depends on output format and size)

## License

MIT License

## Related Documentation

- **Jupyter Docs**: [jupyter.org/documentation](https://jupyter.org/documentation)
- **nbformat**: [nbformat.readthedocs.io](https://nbformat.readthedocs.io)
- **Papermill**: [papermill.readthedocs.io](https://papermill.readthedocs.io)
- **HoloLoom Data Skills**: [../README.md](../README.md)
