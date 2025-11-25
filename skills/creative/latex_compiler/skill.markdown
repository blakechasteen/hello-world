# Skill: LaTeX Compiler

## Metadata

- **Name**: `latex_compiler`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `creative`
- **Tags**: `latex, document, pdf, typesetting, academic, publishing`

## Description

**Short Description**:
Professional document generation with LaTeX for academic papers, reports, and presentations.

**Detailed Description**:
The LaTeX Compiler skill provides comprehensive LaTeX document compilation capabilities for professional typesetting. Compile LaTeX sources to PDF or DVI, manage bibliographies with BibTeX, use custom templates (article, report, beamer), perform multi-pass compilation for cross-references, install missing packages, and diagnose compilation errors. Supports complex documents with figures, tables, equations, citations, and custom styling. Ideal for academic publishing, technical documentation, presentations, and high-quality reports.

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
- `texlive` or `miktex` (LaTeX distribution)
- `pdflatex` (PDF compiler)
- `bibtex` or `biber` (bibliography processor)
- Optional: `latexmk` (automated build tool)

**HoloLoom Integration**: Integrates with documentation pipelines, academic publishing workflows, report generation, and presentation creation.

## Input Schema

```json
{
  "operation": "string - compile_pdf|compile_dvi|compile_with_bib|from_template|diagnostics|install_package",
  "parameters": {
    "source": "string (required for compile operations) - LaTeX source file path (.tex)",
    "output": "string (optional) - Output file path",
    "passes": "number (optional) - Number of compilation passes (default: 2)",
    "bib_file": "string (required for compile_with_bib) - BibTeX file (.bib)",
    "template": "string (required for from_template) - Template name: article|report|beamer|book",
    "title": "string (optional for from_template) - Document title",
    "author": "string (optional for from_template) - Document author",
    "variables": "object (optional for from_template) - Template variables",
    "log_file": "string (required for diagnostics) - LaTeX log file (.log)",
    "package": "string (required for install_package) - LaTeX package name",
    "keep_aux": "boolean (optional) - Keep auxiliary files (default: false)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Compilation details",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "source": "string - Source file compiled",
    "output": "string - Output file path",
    "passes": "number - Compilation passes executed",
    "pages": "number - Number of pages in output",
    "warnings": "number - LaTeX warnings count",
    "errors": "number - LaTeX errors count",
    "size_kb": "number - Output file size",
    "citations": "number - Bibliography citations (for compile_with_bib)",
    "diagnostics": "array - Parsed errors/warnings (for diagnostics)"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Compile LaTeX to PDF

**Input**:
```json
{
  "operation": "compile_pdf",
  "parameters": {
    "source": "papers/research_paper.tex",
    "output": "output/research_paper.pdf",
    "passes": 2
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "compile_pdf",
    "source": "papers/research_paper.tex",
    "output": "output/research_paper.pdf",
    "passes": 2,
    "pages": 15,
    "warnings": 2,
    "errors": 0,
    "size_kb": 245.6
  },
  "message": "PDF compiled successfully: 15 pages, 0 errors",
  "execution_time_ms": 3500
}
```

**Explanation**: Compiles LaTeX source to PDF with 2 passes for cross-references. Standard academic paper workflow.

### Example 2: Compile with Bibliography

**Input**:
```json
{
  "operation": "compile_with_bib",
  "parameters": {
    "source": "papers/thesis.tex",
    "bib_file": "references/thesis_refs.bib",
    "output": "output/thesis.pdf"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "compile_with_bib",
    "source": "papers/thesis.tex",
    "bib_file": "references/thesis_refs.bib",
    "output": "output/thesis.pdf",
    "passes": 4,
    "citations": 87,
    "pages": 156,
    "warnings": 5,
    "errors": 0
  },
  "message": "Thesis compiled with 87 citations (156 pages)",
  "execution_time_ms": 12000
}
```

**Explanation**: Multi-pass compilation (pdflatex -> bibtex -> pdflatex -> pdflatex) for proper bibliography generation. Essential for academic documents.

### Example 3: Generate from Template

**Input**:
```json
{
  "operation": "from_template",
  "parameters": {
    "template": "beamer",
    "title": "Machine Learning Pipeline Architecture",
    "author": "HoloLoom Research Team",
    "variables": {
      "theme": "Madrid",
      "colortheme": "whale",
      "sections": ["Introduction", "Methods", "Results", "Conclusion"]
    },
    "output": "presentations/ml_pipeline.pdf"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "from_template",
    "template": "beamer",
    "title": "Machine Learning Pipeline Architecture",
    "author": "HoloLoom Research Team",
    "generated_source": "presentations/ml_pipeline.tex",
    "output": "presentations/ml_pipeline.pdf",
    "pages": 12,
    "slides": 12
  },
  "message": "Beamer presentation generated: 12 slides",
  "execution_time_ms": 4200
}
```

**Explanation**: Creates professional presentation from template with custom theme and sections. Automated slide deck generation.

### Example 4: Error Diagnostics

**Input**:
```json
{
  "operation": "diagnostics",
  "parameters": {
    "log_file": "logs/document.log"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "diagnostics",
    "log_file": "logs/document.log",
    "errors": [
      {
        "line": 42,
        "message": "Undefined control sequence \\foo",
        "severity": "error"
      },
      {
        "line": 58,
        "message": "Missing $ inserted",
        "severity": "error"
      }
    ],
    "warnings": [
      {
        "line": 15,
        "message": "Overfull hbox (2.34pt too wide)",
        "severity": "warning"
      },
      {
        "line": 23,
        "message": "Citation 'Smith2020' undefined",
        "severity": "warning"
      }
    ],
    "error_count": 2,
    "warning_count": 2
  },
  "message": "Found 2 errors and 2 warnings",
  "execution_time_ms": 150
}
```

**Explanation**: Parses LaTeX log file to extract and categorize errors and warnings. Essential for debugging compilation issues.

### Example 5: Install Missing Package

**Input**:
```json
{
  "operation": "install_package",
  "parameters": {
    "package": "algorithmicx"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "install_package",
    "package": "algorithmicx",
    "installed": true,
    "message": "Package 'algorithmicx' installed successfully"
  },
  "message": "LaTeX package installed: algorithmicx",
  "execution_time_ms": 2500
}
```

**Explanation**: Installs missing LaTeX package via tlmgr (TeX Live Manager). Resolves compilation errors due to missing dependencies.

## Testing Checklist

- [x] **Functionality**: All 6 operations execute correctly
- [x] **Error Handling**: Graceful handling of LaTeX errors, missing files
- [x] **Security**: No command injection, safe file handling
- [x] **Performance**: Operations complete within expected time (<30s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: TeX Live documented
- [x] **Edge Cases**: Handles syntax errors, missing packages, long documents
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom documentation pipelines

## Security Considerations

**Potential Risks**:
- **Command Injection**: LaTeX sources can execute shell commands -> Disable shell-escape, validate inputs
- **File System Access**: LaTeX can read arbitrary files -> Restrict input directory access
- **Resource Exhaustion**: Infinite loops in macros -> Implement timeouts

**Data Privacy**:
- [x] Does not log document content
- [x] Does not upload documents to external servers
- [x] Does not access files outside project directory

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Shell-escape disabled by default
- [x] File operations restricted to designated directories

## Performance Characteristics

- **Expected Latency**: 2000-30000ms (2-30 seconds depending on document complexity)
- **Token Usage**: 100-500 tokens per execution
- **Resource Requirements**: TeX Live installation, sufficient disk space
- **Scalability**: Limited by document size and CPU performance

**Operation-Specific Latencies**:
- `compile_pdf`: 2000-10000ms (depends on document size)
- `compile_dvi`: 1500-8000ms (slightly faster than PDF)
- `compile_with_bib`: 5000-20000ms (4-pass compilation)
- `from_template`: 3000-12000ms (template processing + compilation)
- `diagnostics`: 100-500ms (log parsing only)
- `install_package`: 2000-10000ms (depends on package size)

## License

MIT License

## Related Documentation

- **LaTeX Project**: [latex-project.org](https://latex-project.org)
- **TeX Live**: [tug.org/texlive](https://tug.org/texlive)
- **Beamer Class**: [ctan.org/pkg/beamer](https://ctan.org/pkg/beamer)
- **HoloLoom Creative Skills**: [../README.md](../README.md)
