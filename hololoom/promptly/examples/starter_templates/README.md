# Starter Templates

Welcome to Promptly's starter templates! These examples help you get started quickly with common use cases.

## Quick Start

Each template includes:
- **Python script**: Complete working example
- **YAML workflow**: Declarative workflow definition
- **README**: Explanation and customization guide

## Available Templates

### 1. Question Answering (qa_basic/)

**Use Case**: Answer questions using a knowledge base

**What it does**:
- Takes a question as input
- Retrieves relevant context from HoloLoom memory
- Generates accurate, context-grounded answer

**Best for**:
- Documentation Q&A
- Customer support bots
- Technical knowledge bases

**Files**:
- `qa_basic.py` - Python implementation
- `qa_workflow.yaml` - YAML workflow
- `README.md` - Setup guide

**Run it**:
```bash
cd qa_basic
python qa_basic.py
```

---

### 2. Content Summarization (summarization/)

**Use Case**: Summarize long documents into key points

**What it does**:
- Takes long-form content
- Extracts key entities and concepts
- Generates concise summary with confidence scores

**Best for**:
- Meeting notes summarization
- Article summaries
- Report generation

**Files**:
- `summarization.py` - Python implementation
- `summarization_workflow.yaml` - YAML workflow
- `README.md` - Setup guide

**Run it**:
```bash
cd summarization
python summarization.py
```

---

### 3. Multi-Step Research (research/)

**Use Case**: Research a topic by generating and answering sub-questions

**What it does**:
- Breaks complex topic into sub-questions
- Answers each sub-question using retrieval
- Synthesizes findings into comprehensive report

**Best for**:
- Competitive analysis
- Technical research
- Market research

**Files**:
- `research.py` - Python implementation
- `research_workflow.yaml` - YAML workflow (3 steps)
- `README.md` - Setup guide

**Run it**:
```bash
cd research
python research.py
```

---

### 4. Code Review (code_review/)

**Use Case**: Automated code review with best practices

**What it does**:
- Analyzes code for bugs, style issues, security
- Suggests improvements with explanations
- Scores code quality across dimensions

**Best for**:
- PR review automation
- Code quality checks
- Learning best practices

**Files**:
- `code_review.py` - Python implementation
- `code_review_workflow.yaml` - YAML workflow (5 steps)
- `README.md` - Setup guide

**Run it**:
```bash
cd code_review
python code_review.py
```

---

### 5. Structured Data Extraction (extraction/)

**Use Case**: Extract structured data from unstructured text

**What it does**:
- Defines schema (fields, types, validation)
- Extracts data matching schema
- Validates and scores extraction quality

**Best for**:
- Resume parsing
- Invoice data extraction
- Form processing

**Files**:
- `extraction.py` - Python implementation
- `extraction_workflow.yaml` - YAML workflow with schema
- `README.md` - Setup guide

**Run it**:
```bash
cd extraction
python extraction.py
```

---

### 6. Workflow Composition (advanced_workflow/)

**Use Case**: Compose complex multi-agent workflows

**What it does**:
- Chains multiple steps with input/output mapping
- Uses `{step.output}` references
- Handles errors and retries

**Best for**:
- Complex pipelines
- Multi-stage processing
- Learning workflow composition

**Files**:
- `advanced_workflow.py` - Python implementation
- `advanced_workflow.yaml` - 7-step workflow
- `README.md` - Setup guide

**Run it**:
```bash
cd advanced_workflow
python advanced_workflow.py
```

---

## Customizing Templates

### 1. Modify Python Script

```python
# Original
signature = create_signature(
    "Answer questions using context",
    inputs=["question", "context"],
    outputs=["answer"]
)

# Customized
signature = create_signature(
    "Your custom instruction",
    inputs=["your_input_1", "your_input_2"],
    outputs=["your_output_1", "your_output_2"]
)
```

### 2. Modify YAML Workflow

```yaml
# Original
steps:
  - name: retrieve
    module: memory_retrieval
    inputs:
      query: "{input.question}"

# Customized
steps:
  - name: my_custom_step
    module: my_custom_module
    inputs:
      my_param: "{input.my_value}"
```

### 3. Add Your Own Examples

Each template loads examples from `examples.json`:

```json
[
  {
    "input": "Your example input",
    "output": "Expected output"
  }
]
```

Add 5-10 examples to improve optimization quality.

---

## Template Matrix

| Template | Complexity | Steps | Time | Best For |
|----------|-----------|-------|------|----------|
| QA Basic | Simple | 1 | <1s | Getting started |
| Summarization | Simple | 2 | <2s | Content processing |
| Research | Medium | 3 | ~3s | Multi-query research |
| Code Review | Medium | 5 | ~5s | Code analysis |
| Extraction | Medium | 2 | ~2s | Data extraction |
| Advanced Workflow | Complex | 7 | ~7s | Learning workflows |

---

## Learning Path

**Beginners**: Start here
1. QA Basic - Understand signatures
2. Summarization - Learn multi-step workflows
3. Extraction - Explore schema-first approach

**Intermediate**: Build on basics
1. Research - Multi-query decomposition
2. Code Review - Complex quality scoring
3. Advanced Workflow - Composition patterns

**Advanced**: Customize and extend
1. Create your own templates
2. Add custom metrics
3. Integrate with your systems

---

## Contributing Templates

Want to share a template? We'd love to include it!

**Requirements**:
- Working Python script
- YAML workflow definition
- README with setup instructions
- 3-5 example inputs/outputs
- Test showing it works

**Submit**:
1. Fork the repository
2. Add your template to `examples/starter_templates/your_template/`
3. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

---

## Getting Help

**Issues with templates?**
- Check template README
- Review [Master Index](../../MASTER_INDEX.md)
- Ask in [GitHub Discussions](https://github.com/promptly/promptly/discussions)
- Join [Discord](https://discord.gg/promptly) (coming soon)

---

## Next Steps

After exploring templates:
1. Read [QUICK_START_GUIDE.md](../../QUICK_START_GUIDE.md)
2. Review [ARCHITECTURE_6_PROBLEMS.md](../../ARCHITECTURE_6_PROBLEMS.md)
3. Build your own workflows
4. Share with the community

---

**Happy building! 🚀**
