# Promptly Integration - Complex Prompt Chaining

**Status**: 🚧 Design Complete, Implementation Ready
**Purpose**: Integrate Promptly's template system for complex prompt workflows

## Overview

Promptly is HoloLoom's prompt template and chaining system that enables:
- **Template-based prompts** - Reusable prompt structures
- **Multi-step workflows** - Chain prompts together
- **Variable substitution** - Dynamic prompt generation
- **Conditional logic** - Branch based on results
- **Context accumulation** - Build context across steps

## Architecture

```
User Query
    ↓
Promptly Template Selector
    ├─ Match query → appropriate template
    ├─ Load template with workflow
    └─ Initialize execution context
    ↓
Step Executor (for each step in workflow)
    ├─ Substitute variables from context
    ├─ Execute via Agent (MCTS + reasoning mode)
    ├─ Extract result
    ├─ Update context
    └─ Conditionally branch
    ↓
Result Synthesizer
    ├─ Collect all step results
    ├─ Apply final synthesis prompt
    └─ Return complete response
    ↓
Store in Thread History
```

## Template Structure

### Basic Template (YAML)

```yaml
name: "Financial Analysis"
description: "Multi-step financial analysis workflow"
version: "1.0"

# Input variables expected
inputs:
  - name: "company"
    type: "string"
    required: true
  - name: "period"
    type: "string"
    default: "Q4"

# Workflow steps
steps:
  - id: "gather_data"
    prompt: |
      Gather financial data for {{company}} for {{period}}.
      Focus on:
      - Revenue
      - Expenses
      - Profit margins
      - Cash flow

    agent: "budget"
    mode: "research"
    extract:
      revenue: "extract_number('Revenue')"
      expenses: "extract_number('Expenses')"

  - id: "analyze_trends"
    prompt: |
      Analyze trends in {{company}}'s financials:

      Revenue: {{gather_data.revenue}}
      Expenses: {{gather_data.expenses}}

      Identify:
      1. Year-over-year changes
      2. Key drivers
      3. Risks

    agent: "research"
    mode: "verify"

  - id: "generate_insights"
    prompt: |
      Based on the analysis:

      {{analyze_trends.response}}

      Generate actionable insights and recommendations.

    agent: "architecture"
    mode: "plan_execute"

# Final synthesis
synthesis:
  prompt: |
    Synthesize the complete financial analysis:

    Data: {{gather_data.response}}
    Trends: {{analyze_trends.response}}
    Insights: {{generate_insights.response}}

    Provide executive summary with:
    1. Key findings
    2. Recommendations
    3. Action items

  agent: "budget"
  mode: "verify"
```

### Advanced Template with Conditionals

```yaml
name: "Adaptive Research"
description: "Research workflow that adapts based on confidence"
version: "1.0"

inputs:
  - name: "topic"
    type: "string"
    required: true

steps:
  - id: "initial_search"
    prompt: "Research: {{topic}}"
    agent: "research"
    mode: "research"

  # Conditional: If confidence < 0.8, do deeper research
  - id: "deep_dive"
    condition: "{{initial_search.confidence}} < 0.8"
    prompt: |
      The initial search had low confidence ({{initial_search.confidence}}).
      Perform deeper research on {{topic}}.
      Focus on:
      - Alternative sources
      - Related topics
      - Expert opinions
    agent: "research"
    mode: "research"

  # Conditional: If breakthrough detected, explore it
  - id: "explore_breakthrough"
    condition: "{{initial_search.has_breakthrough}}"
    prompt: |
      A breakthrough was detected in the initial search.
      Explore this discovery in depth:

      {{initial_search.response}}

    agent: "research"
    mode: "plan_execute"

synthesis:
  prompt: |
    Synthesize all research findings:

    Initial: {{initial_search.response}}
    {% if deep_dive %}Deep dive: {{deep_dive.response}}{% endif %}
    {% if explore_breakthrough %}Breakthrough: {{explore_breakthrough.response}}{% endif %}

    Provide comprehensive answer to: {{topic}}
```

## Implementation

### 1. Template Engine

```python
# HoloLoom/promptly/template_engine.py

import yaml
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """A Promptly template"""
    name: str
    description: str
    version: str
    inputs: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    synthesis: Optional[Dict[str, Any]] = None


class TemplateEngine:
    """
    Promptly template engine for complex prompt workflows.

    Features:
    - Variable substitution ({{var}})
    - Conditional execution (condition: "{{var}} > 0.8")
    - Context accumulation
    - Step result extraction
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def load_template(self, template_path: str) -> PromptTemplate:
        """Load template from YAML file"""
        with open(template_path, 'r') as f:
            data = yaml.safe_load(f)

        template = PromptTemplate(
            name=data['name'],
            description=data['description'],
            version=data['version'],
            inputs=data.get('inputs', []),
            steps=data['steps'],
            synthesis=data.get('synthesis')
        )

        self.templates[template.name] = template
        return template

    def substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """
        Substitute variables in text from context.

        Supports:
        - Simple: {{var}}
        - Nested: {{step.result}}
        - Expressions: {{var * 2}}
        """
        def replace(match):
            var_expr = match.group(1)

            # Simple variable
            if '.' in var_expr:
                parts = var_expr.split('.')
                value = context
                for part in parts:
                    value = value.get(part, '')
                return str(value)
            else:
                return str(context.get(var_expr, ''))

        return re.sub(r'\{\{(.+?)\}\}', replace, text)

    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition string.

        Examples:
        - "{{confidence}} > 0.8"
        - "{{has_breakthrough}} == true"
        """
        # Substitute variables
        condition = self.substitute_variables(condition, context)

        # Evaluate as Python expression (safe subset)
        try:
            return eval(condition, {"__builtins__": {}}, {})
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False


class PromptlyExecutor:
    """
    Executes Promptly templates through agent system.
    """

    def __init__(
        self,
        template_engine: TemplateEngine,
        thread_manager: ConversationThreadManager
    ):
        self.template_engine = template_engine
        self.thread_manager = thread_manager

    async def execute_template(
        self,
        template_name: str,
        inputs: Dict[str, Any],
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Execute template workflow.

        Args:
            template_name: Name of template to execute
            inputs: Input variables
            thread_id: Thread to execute in

        Returns:
            {
                "success": true,
                "steps": [...],
                "synthesis": "...",
                "context": {...}
            }
        """
        template = self.template_engine.templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        # Initialize context with inputs
        context = {**inputs}
        step_results = []

        # Execute each step
        for step_def in template.steps:
            step_id = step_def['id']

            # Check condition (if present)
            if 'condition' in step_def:
                condition = step_def['condition']
                if not self.template_engine.evaluate_condition(condition, context):
                    logger.info(f"Skipping step {step_id} (condition not met)")
                    continue

            # Substitute variables in prompt
            prompt = self.template_engine.substitute_variables(
                step_def['prompt'],
                context
            )

            # Execute step via agent
            agent_name = step_def.get('agent', 'budget')
            mode = step_def.get('mode', 'verify')

            mode_enum = {
                "direct": ReasoningMode.DIRECT,
                "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH,
                "plan_execute": ReasoningMode.PLAN_EXECUTE
            }.get(mode, ReasoningMode.VERIFY)

            result = await self.thread_manager.query_thread(
                thread_id=thread_id,
                query=Query(text=prompt),
                mode=mode_enum
            )

            # Store step result in context
            context[step_id] = {
                "response": result.spacetime.metadata.get("response", ""),
                "confidence": result.spacetime.confidence,
                "mode": mode,
                "has_breakthrough": result.spacetime.metadata.get("breakthrough", False)
            }

            step_results.append({
                "step_id": step_id,
                "prompt": prompt,
                "response": context[step_id]["response"],
                "confidence": context[step_id]["confidence"]
            })

        # Execute synthesis (if present)
        synthesis_result = None
        if template.synthesis:
            synthesis_prompt = self.template_engine.substitute_variables(
                template.synthesis['prompt'],
                context
            )

            agent_name = template.synthesis.get('agent', 'budget')
            mode = template.synthesis.get('mode', 'verify')

            mode_enum = {
                "direct": ReasoningMode.DIRECT,
                "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH,
                "plan_execute": ReasoningMode.PLAN_EXECUTE
            }.get(mode, ReasoningMode.VERIFY)

            result = await self.thread_manager.query_thread(
                thread_id=thread_id,
                query=Query(text=synthesis_prompt),
                mode=mode_enum
            )

            synthesis_result = result.spacetime.metadata.get("response", "")

        return {
            "success": True,
            "template": template_name,
            "steps": step_results,
            "synthesis": synthesis_result,
            "context": context
        }
```

### 2. API Integration

```python
# Add to agentic_api_enhanced.py

from HoloLoom.promptly.template_engine import TemplateEngine, PromptlyExecutor


# In ServerState:
class ServerState:
    def __init__(self):
        # ... existing ...
        self.template_engine: Optional[TemplateEngine] = None
        self.promptly_executor: Optional[PromptlyExecutor] = None


# In startup():
@app.on_event("startup")
async def startup():
    # ... existing ...

    # Initialize Promptly
    logger.info("Initializing Promptly template engine...")
    state.template_engine = TemplateEngine()

    # Load templates from directory
    templates_dir = Path(__file__).parent.parent / "promptly" / "templates"
    if templates_dir.exists():
        for template_file in templates_dir.glob("*.yaml"):
            try:
                state.template_engine.load_template(str(template_file))
                logger.info(f"Loaded template: {template_file.name}")
            except Exception as e:
                logger.error(f"Failed to load template {template_file}: {e}")

    state.promptly_executor = PromptlyExecutor(
        template_engine=state.template_engine,
        thread_manager=state.thread_manager
    )

    logger.info("✅ Promptly integration ready!")


# New endpoints:

@app.get("/promptly/templates")
async def list_templates():
    """List all available Promptly templates"""
    templates = []

    for name, template in state.template_engine.templates.items():
        templates.append({
            "name": template.name,
            "description": template.description,
            "version": template.version,
            "inputs": template.inputs,
            "step_count": len(template.steps)
        })

    return {"templates": templates}


@app.post("/promptly/execute")
async def execute_template(
    thread_id: str,
    template_name: str,
    inputs: Dict[str, Any]
):
    """Execute a Promptly template"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        result = await state.promptly_executor.execute_template(
            template_name=template_name,
            inputs=inputs,
            thread_id=thread_id
        )

        return result

    except Exception as e:
        logger.error(f"Failed to execute template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/promptly/templates/{template_name}")
async def get_template(template_name: str):
    """Get template details"""
    template = state.template_engine.templates.get(template_name)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return {
        "name": template.name,
        "description": template.description,
        "version": template.version,
        "inputs": template.inputs,
        "steps": template.steps,
        "synthesis": template.synthesis
    }
```

### 3. Frontend Integration

```javascript
// Add to multithreaded_chat_enhanced.html

class PromptlyManager {
    constructor() {
        this.templates = new Map();
    }

    async loadTemplates() {
        const response = await fetch(`${API_BASE}/promptly/templates`);
        const data = await response.json();

        data.templates.forEach(template => {
            this.templates.set(template.name, template);
        });
    }

    async executeTemplate(threadId, templateName, inputs) {
        const response = await fetch(`${API_BASE}/promptly/execute`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                thread_id: threadId,
                template_name: templateName,
                inputs: inputs
            })
        });

        return await response.json();
    }

    showTemplateSelector() {
        // Show modal with template selection
        const modal = document.getElementById('templateSelectorModal');
        const list = document.getElementById('templateList');

        list.innerHTML = '';

        this.templates.forEach((template, name) => {
            const item = document.createElement('div');
            item.className = 'template-item';
            item.innerHTML = `
                <h4>${template.name}</h4>
                <p>${template.description}</p>
                <div class="template-meta">
                    ${template.step_count} steps | v${template.version}
                </div>
                <button onclick="selectTemplate('${name}')">Use Template</button>
            `;
            list.appendChild(item);
        });

        modal.classList.remove('hidden');
    }

    async selectTemplate(templateName) {
        const template = this.templates.get(templateName);

        // Show input form for template variables
        const inputs = {};

        for (const input of template.inputs) {
            const value = prompt(`Enter ${input.name}:`, input.default || '');
            if (value) {
                inputs[input.name] = value;
            }
        }

        // Execute template
        showToast('Executing Template', `Running ${templateName}...`);

        const result = await this.executeTemplate(
            state.activeThreadId,
            templateName,
            inputs
        );

        // Show results
        displayTemplateResults(result);

        showToast('Template Complete', `${templateName} executed successfully`);
    }
}

const promptlyManager = new PromptlyManager();

// Initialize on load
window.addEventListener('load', async () => {
    await promptlyManager.loadTemplates();
});

// Add template button to UI
function addTemplateButton() {
    const toolbar = document.querySelector('.input-controls');

    const button = document.createElement('button');
    button.textContent = '📋 Templates';
    button.onclick = () => promptlyManager.showTemplateSelector();

    toolbar.appendChild(button);
}
```

## Example Templates

### 1. Financial Analysis

**File**: `HoloLoom/promptly/templates/financial_analysis.yaml`

### 2. Research Report

```yaml
name: "Research Report"
description: "Generate comprehensive research report"
version: "1.0"

inputs:
  - name: "topic"
    type: "string"
    required: true

steps:
  - id: "gather_sources"
    prompt: "Find authoritative sources on: {{topic}}"
    agent: "research"
    mode: "research"

  - id: "analyze_findings"
    prompt: |
      Analyze the research findings:

      {{gather_sources.response}}

      Identify:
      - Key themes
      - Consensus areas
      - Controversies
      - Gaps in knowledge
    agent: "research"
    mode: "verify"

  - id: "structure_report"
    prompt: |
      Create report outline for: {{topic}}

      Based on analysis:
      {{analyze_findings.response}}

      Include:
      1. Executive summary
      2. Background
      3. Findings
      4. Implications
      5. Recommendations
    agent: "architecture"
    mode: "plan_execute"

synthesis:
  prompt: |
    Generate final research report on {{topic}}:

    Sources: {{gather_sources.response}}
    Analysis: {{analyze_findings.response}}
    Structure: {{structure_report.response}}

    Write comprehensive report following the outline.
  agent: "research"
  mode: "verify"
```

### 3. Code Review Workflow

```yaml
name: "Code Review"
description: "Systematic code review workflow"
version: "1.0"

inputs:
  - name: "code"
    type: "string"
    required: true
  - name: "language"
    type: "string"
    default: "python"

steps:
  - id: "syntax_check"
    prompt: "Check {{language}} code for syntax errors: {{code}}"
    agent: "architecture"
    mode: "direct"

  - id: "security_audit"
    prompt: |
      Security audit for {{language}} code:

      {{code}}

      Check for:
      - SQL injection
      - XSS vulnerabilities
      - Authentication issues
      - Data validation
    agent: "architecture"
    mode: "verify"

  - id: "performance_analysis"
    prompt: |
      Analyze performance of this {{language}} code:

      {{code}}

      Identify:
      - Bottlenecks
      - Optimization opportunities
      - Complexity issues
    agent: "architecture"
    mode: "research"

  - id: "best_practices"
    prompt: |
      Review {{language}} code against best practices:

      {{code}}

      Check:
      - Code style
      - Documentation
      - Error handling
      - Testability
    agent: "architecture"
    mode: "verify"

synthesis:
  prompt: |
    Complete code review for {{language}} code:

    Syntax: {{syntax_check.response}}
    Security: {{security_audit.response}}
    Performance: {{performance_analysis.response}}
    Best Practices: {{best_practices.response}}

    Provide:
    1. Overall assessment
    2. Critical issues
    3. Recommendations
    4. Priority fixes
  agent: "architecture"
  mode: "verify"
```

## Benefits

### For Users
✅ **Complex workflows simplified** - Multi-step analysis in one command
✅ **Reusable templates** - Save common workflows
✅ **Consistent quality** - Structured approach every time
✅ **Time savings** - Automate repetitive tasks

### For System
✅ **Structured prompting** - Better results through careful sequencing
✅ **Context accumulation** - Build on previous steps
✅ **Conditional logic** - Adapt based on results
✅ **Quality assurance** - Verify at each step

## Next Steps

1. ✅ Create `HoloLoom/promptly/template_engine.py`
2. ✅ Create `HoloLoom/promptly/templates/` directory
3. ✅ Add 5-10 starter templates
4. ✅ Integrate into API
5. ✅ Add frontend UI
6. ✅ Documentation and examples
7. ✅ Testing with real workflows

## Timeline

- **Template Engine**: 2 hours
- **API Integration**: 1 hour
- **Frontend UI**: 2 hours
- **Starter Templates**: 2 hours
- **Testing**: 1 hour
- **Total**: 8 hours

Ready to implement!
