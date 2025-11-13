# Promptly Strategy Framework: Extensible & Elegant Design

**Date:** November 13, 2025
**Philosophy:** "One framework, infinite strategies"

---

## Core Insight

All 10 advanced prompting techniques follow **the same pattern**:
```
Input Query → Strategy Enhancement → Enhanced Query → LLM → Result
```

Instead of 10 separate commands, we build **one extensible framework** where:
- Each technique is a **strategy** (implements common interface)
- Strategies are **composable** (chain multiple together)
- Strategies are **discoverable** (auto-loaded from `promptly_skills/strategies/`)
- Strategies are **configurable** (YAML-defined behavior)

---

## Architecture: The Strategy Pattern

### Core Interface

```python
# HoloLoom/prompting/strategy.py (NEW)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class StrategyContext:
    """Context passed to all strategies"""
    query: str
    user_context: Optional[Dict[str, Any]] = None
    selection: Optional[str] = None
    file_path: Optional[str] = None
    previous_result: Optional[Any] = None

@dataclass
class StrategyResult:
    """Result from strategy execution"""
    enhanced_query: str
    metadata: Dict[str, Any]
    requires_followup: bool = False
    followup_strategies: List[str] = None

class PromptingStrategy(ABC):
    """Base class for all prompting strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'verify', 'challenge')"""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: self-correction, meta-prompting, scaffolding, perspective"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description"""
        pass

    @abstractmethod
    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """Apply strategy to enhance the query"""
        pass

    @abstractmethod
    def can_apply(self, context: StrategyContext) -> float:
        """
        Return confidence (0-1) that this strategy is appropriate.
        Used for auto-suggestion.
        """
        pass

    def compose_with(self, other: 'PromptingStrategy') -> 'CompositeStrategy':
        """Compose this strategy with another"""
        return CompositeStrategy([self, other])
```

### Strategy Registry (Auto-Discovery)

```python
# HoloLoom/prompting/registry.py (NEW)

from typing import Dict, List
import importlib
import pkgutil
from pathlib import Path

class StrategyRegistry:
    """Auto-discovers and registers all strategies"""

    def __init__(self):
        self.strategies: Dict[str, PromptingStrategy] = {}
        self._discover_strategies()

    def _discover_strategies(self):
        """Auto-discover strategies from promptly_skills/strategies/"""
        strategies_dir = Path(__file__).parent.parent.parent / "promptly_skills" / "strategies"

        if not strategies_dir.exists():
            return

        for item in strategies_dir.iterdir():
            if item.is_dir() and (item / "strategy.py").exists():
                # Load strategy module
                module = importlib.import_module(f"promptly_skills.strategies.{item.name}.strategy")

                # Find strategy class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, PromptingStrategy) and
                        attr != PromptingStrategy):

                        strategy = attr()
                        self.strategies[strategy.name] = strategy

    def get(self, name: str) -> Optional[PromptingStrategy]:
        """Get strategy by name"""
        return self.strategies.get(name)

    def list_by_category(self, category: str) -> List[PromptingStrategy]:
        """List all strategies in a category"""
        return [s for s in self.strategies.values() if s.category == category]

    def suggest(self, context: StrategyContext, top_k: int = 3) -> List[tuple[str, float]]:
        """Suggest strategies for a context (auto-detection)"""
        scores = [(s.name, s.can_apply(context)) for s in self.strategies.values()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Global registry
_registry = StrategyRegistry()

def get_strategy(name: str) -> Optional[PromptingStrategy]:
    return _registry.get(name)

def suggest_strategies(context: StrategyContext, top_k: int = 3) -> List[tuple[str, float]]:
    return _registry.suggest(context, top_k)
```

### Composite Strategy (Chaining)

```python
# HoloLoom/prompting/composite.py (NEW)

class CompositeStrategy(PromptingStrategy):
    """Composes multiple strategies into a pipeline"""

    def __init__(self, strategies: List[PromptingStrategy]):
        self.strategies = strategies

    @property
    def name(self) -> str:
        return "+".join(s.name for s in self.strategies)

    @property
    def category(self) -> str:
        return "composite"

    @property
    def description(self) -> str:
        return f"Composite: {', '.join(s.name for s in self.strategies)}"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """Apply strategies sequentially"""
        current_context = context
        all_metadata = {}

        for strategy in self.strategies:
            result = await strategy.enhance(current_context)

            # Update context for next strategy
            current_context = StrategyContext(
                query=result.enhanced_query,
                user_context=context.user_context,
                selection=context.selection,
                file_path=context.file_path,
                previous_result=result
            )

            # Accumulate metadata
            all_metadata[strategy.name] = result.metadata

        return StrategyResult(
            enhanced_query=current_context.query,
            metadata={"composite": all_metadata},
            requires_followup=False
        )

    def can_apply(self, context: StrategyContext) -> float:
        # Composite is as applicable as its weakest link
        return min(s.can_apply(context) for s in self.strategies)
```

---

## Strategy Implementations

### Strategy Directory Structure

```
promptly_skills/strategies/
├── verify/                          # Chain of Verification
│   ├── strategy.py                  # VerifyStrategy class
│   ├── config.yaml                  # Configuration
│   ├── template.md                  # Prompt template
│   └── README.md
├── challenge/                       # Adversarial Prompting
│   ├── strategy.py
│   ├── config.yaml
│   └── template.md
├── edge_cases/                      # Few-shot Edge Cases
│   ├── strategy.py
│   ├── config.yaml
│   ├── library/                     # Edge case library
│   │   ├── sql_injection.yaml
│   │   ├── xss.yaml
│   │   └── race_conditions.yaml
│   └── README.md
├── reverse/                         # Reverse Prompting
├── optimize/                        # Recursive Optimization
├── deep/                           # Deliberate Over-Instruction
├── scaffold/                       # Zero-shot CoT Structure
│   ├── strategy.py
│   ├── templates/
│   │   ├── root_cause_analysis.yaml
│   │   ├── code_review.yaml
│   │   └── security_audit.yaml
│   └── README.md
├── prime/                          # Reference Class Priming
├── debate/                         # Multi-Persona Debate
│   ├── strategy.py
│   ├── personas/
│   │   ├── vendor_selection.yaml
│   │   └── architecture_review.yaml
│   └── README.md
└── temp_sim/                       # Temperature Simulation
```

### Example: Verify Strategy

```python
# promptly_skills/strategies/verify/strategy.py

from HoloLoom.prompting.strategy import PromptingStrategy, StrategyContext, StrategyResult
from HoloLoom.recursive import AdvancedRefiner, RefinementStrategy
from HoloLoom.documentation.types import Query
import yaml
from pathlib import Path

class VerifyStrategy(PromptingStrategy):
    """Chain of Verification strategy"""

    def __init__(self):
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load prompt template
        template_path = Path(__file__).parent / "template.md"
        with open(template_path) as f:
            self.template = f.read()

    @property
    def name(self) -> str:
        return "verify"

    @property
    def category(self) -> str:
        return "self-correction"

    @property
    def description(self) -> str:
        return "Chain of Verification: Force model to critique its own output"

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """Apply chain of verification"""

        # Format template with query
        enhanced = self.template.format(
            original_query=context.query,
            passes=self.config.get('passes', 3),
            verification_depth=self.config.get('depth', 'standard')
        )

        metadata = {
            "strategy": "verify",
            "passes": self.config['passes'],
            "original_query": context.query,
            "template_version": self.config.get('version', '1.0')
        }

        return StrategyResult(
            enhanced_query=enhanced,
            metadata=metadata,
            requires_followup=False
        )

    def can_apply(self, context: StrategyContext) -> float:
        """
        Verify is good for:
        - Claims that need checking
        - Analysis that might be incomplete
        - Security-critical outputs
        """
        query_lower = context.query.lower()

        # High confidence triggers
        high_confidence_words = ['analyze', 'review', 'check', 'verify', 'validate']
        if any(word in query_lower for word in high_confidence_words):
            return 0.9

        # Medium confidence triggers
        medium_confidence_words = ['explain', 'describe', 'what is']
        if any(word in query_lower for word in medium_confidence_words):
            return 0.6

        # Security context
        if context.file_path and any(word in context.file_path for word in ['security', 'auth', 'crypto']):
            return 0.95

        return 0.3  # Default: can apply to most queries
```

```yaml
# promptly_skills/strategies/verify/config.yaml

name: verify
version: 1.0.0
category: self-correction
description: Chain of Verification forces model to critique its own output

config:
  passes: 3
  depth: standard  # standard, deep, exhaustive

  # Quality thresholds
  min_confidence: 0.75
  target_quality: 0.85

  # Performance
  max_iterations: 5
  timeout_seconds: 30

triggers:
  high_confidence:
    - analyze
    - review
    - check
    - verify
    - validate
  medium_confidence:
    - explain
    - describe
    - what is
  context_triggers:
    - security
    - auth
    - crypto
    - contract
    - legal
```

```markdown
# promptly_skills/strategies/verify/template.md

You are performing a Chain of Verification analysis.

**Original Query:** {original_query}

## Instructions

You will perform {passes} verification passes:

### Pass 1: Initial Analysis
Provide your best answer to the query.

### Pass 2: Identify Incompleteness
List 3 specific ways your analysis might be incomplete:
1. [What might you have missed?]
2. [What assumptions did you make?]
3. [What edge cases weren't considered?]

### Pass 3: Cite Evidence
For each concern from Pass 2:
- Cite specific evidence that confirms OR refutes the concern
- If information is insufficient, state what's missing

### Pass 4: Revised Analysis
Provide a revised answer that addresses all identified gaps.

## Output Format

**Pass 1 - Initial Analysis:**
[Your initial answer]

**Pass 2 - Potential Incompleteness:**
1. [Concern 1]
2. [Concern 2]
3. [Concern 3]

**Pass 3 - Evidence Review:**
- Concern 1: [Evidence + assessment]
- Concern 2: [Evidence + assessment]
- Concern 3: [Evidence + assessment]

**Pass 4 - Revised Analysis:**
[Your improved, complete answer]

**Quality Check:**
- ✓ [Criterion 1]
- ✓ [Criterion 2]
- ✓ [Criterion 3]
```

---

## Unified Command Interface

### Single Command: `/strategy`

Instead of 10 commands, we have **one unified command**:

```bash
# Use a single strategy
/strategy verify "analyze this contract"

# Chain multiple strategies
/strategy verify+challenge "review security architecture"

# Auto-detect best strategy
/strategy auto "optimize this SQL query"

# List available strategies
/strategy list

# Get strategy info
/strategy info verify
```

### VS Code Implementation

```typescript
// promptly-vscode/src/commands/strategyCommand.ts (NEW)

import * as vscode from 'vscode';
import { HoloLoomClient } from '../hololoomClient';

export async function strategyCommand(context: vscode.ExtensionContext) {
    // Get strategy name
    const strategyInput = await vscode.window.showInputBox({
        prompt: 'Strategy name (or "auto" for auto-detection)',
        placeHolder: 'verify, challenge, verify+challenge, auto',
        value: 'auto'
    });

    if (!strategyInput) return;

    // Get query
    const query = await vscode.window.showInputBox({
        prompt: 'What do you want to do?',
        placeHolder: 'e.g., "analyze this contract"'
    });

    if (!query) return;

    // Get selection if available
    const editor = vscode.window.activeTextEditor;
    const selection = editor?.document.getText(editor.selection);

    const client = new HoloLoomClient();

    // Handle special commands
    if (strategyInput === 'list') {
        const strategies = await client.listStrategies();
        showStrategyList(strategies);
        return;
    }

    if (strategyInput.startsWith('info ')) {
        const strategyName = strategyInput.substring(5);
        const info = await client.getStrategyInfo(strategyName);
        showStrategyInfo(info);
        return;
    }

    // Execute strategy
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Running strategy: ${strategyInput}`,
        cancellable: false
    }, async (progress) => {
        const result = await client.executeStrategy({
            strategy: strategyInput,
            query: query,
            selection: selection,
            filePath: editor?.document.fileName
        });

        displayStrategyResult(result);
    });
}

function showStrategyList(strategies: any[]) {
    const panel = vscode.window.createWebviewPanel(
        'strategies',
        'Available Strategies',
        vscode.ViewColumn.Beside,
        {}
    );

    const byCategory = groupBy(strategies, 'category');

    panel.webview.html = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: sans-serif; padding: 20px; }
                .category { margin-bottom: 30px; }
                .category h2 {
                    color: #007ACC;
                    border-bottom: 2px solid #007ACC;
                    padding-bottom: 5px;
                }
                .strategy {
                    margin: 10px 0;
                    padding: 10px;
                    background: #f5f5f5;
                    border-left: 3px solid #4CAF50;
                }
                .strategy-name {
                    font-weight: bold;
                    font-size: 18px;
                    color: #333;
                }
                .strategy-desc {
                    margin-top: 5px;
                    color: #666;
                }
                .usage {
                    margin-top: 5px;
                    font-family: monospace;
                    color: #007ACC;
                }
            </style>
        </head>
        <body>
            <h1>Promptly Strategy Library</h1>
            ${Object.entries(byCategory).map(([category, strategies]) => `
                <div class="category">
                    <h2>${category}</h2>
                    ${strategies.map(s => `
                        <div class="strategy">
                            <div class="strategy-name">${s.name}</div>
                            <div class="strategy-desc">${s.description}</div>
                            <div class="usage">Usage: /strategy ${s.name} "your query"</div>
                        </div>
                    `).join('')}
                </div>
            `).join('')}
        </body>
        </html>
    `;
}
```

### Matrix Bot Integration

```python
# promptly-matrix-bot/bot/strategy_handler.py (NEW)

from HoloLoom.prompting.registry import get_strategy, suggest_strategies
from HoloLoom.prompting.strategy import StrategyContext

class StrategyHandler:
    """Handle strategy commands in Matrix bot"""

    async def handle_strategy_command(self, message: str, room: MatrixRoom, event: RoomMessageText):
        """
        Parse: /strategy <name> <query>
        Examples:
        - /strategy verify analyze this contract
        - /strategy verify+challenge review security
        - /strategy auto optimize SQL query
        """
        parts = message.split(maxsplit=2)

        if len(parts) < 3:
            return await self.send_strategy_help(room)

        strategy_name = parts[1]
        query = parts[2]

        # Handle special commands
        if strategy_name == 'list':
            return await self.list_strategies(room)

        if strategy_name == 'auto':
            return await self.auto_detect_strategy(query, room)

        # Execute strategy
        return await self.execute_strategy(strategy_name, query, room)

    async def auto_detect_strategy(self, query: str, room: MatrixRoom):
        """Auto-detect best strategy for query"""
        context = StrategyContext(query=query)
        suggestions = suggest_strategies(context, top_k=3)

        # Show suggestions
        message = f"**Auto-Detection for:** {query}\n\n"
        message += "**Suggested Strategies:**\n"
        for i, (name, confidence) in enumerate(suggestions, 1):
            message += f"{i}. `{name}` (confidence: {confidence:.0%})\n"

        message += "\nUsing top suggestion...\n"

        await self.send_message(room, message)

        # Execute top suggestion
        top_strategy = suggestions[0][0]
        return await self.execute_strategy(top_strategy, query, room)

    async def execute_strategy(self, strategy_name: str, query: str, room: MatrixRoom):
        """Execute a strategy"""

        # Handle composite strategies (verify+challenge)
        if '+' in strategy_name:
            strategy_names = strategy_name.split('+')
            strategies = [get_strategy(name) for name in strategy_names]

            if None in strategies:
                return await self.send_message(room, "❌ Unknown strategy in chain")

            from HoloLoom.prompting.composite import CompositeStrategy
            strategy = CompositeStrategy(strategies)
        else:
            strategy = get_strategy(strategy_name)

            if not strategy:
                return await self.send_message(room, f"❌ Unknown strategy: {strategy_name}")

        # Execute
        context = StrategyContext(query=query)
        result = await strategy.enhance(context)

        # Send enhanced query to LLM
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        orchestrator = WeavingOrchestrator(cfg=Config.fast())

        spacetime = await orchestrator.weave(Query(text=result.enhanced_query))

        # Format response
        response = f"**Strategy:** {strategy.name}\n\n"
        response += f"**Result:**\n{spacetime.response}\n\n"
        response += f"**Metadata:**\n"
        response += f"- Quality: {spacetime.confidence:.0%}\n"
        response += f"- Strategy Metadata: {result.metadata}\n"

        return await self.send_message(room, response)
```

---

## Auto-Detection Engine

```python
# HoloLoom/prompting/auto_detect.py (NEW)

from typing import List, Tuple
from HoloLoom.prompting.strategy import StrategyContext
from HoloLoom.prompting.registry import suggest_strategies

class AutoDetector:
    """
    Analyzes queries and suggests appropriate strategies.
    Learns from feedback to improve suggestions over time.
    """

    def __init__(self):
        self.history = []  # (context, suggested_strategy, was_helpful)

    async def detect(self, context: StrategyContext, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Detect best strategies for context.

        Returns:
            List of (strategy_name, confidence) tuples
        """
        # Get base suggestions from registry
        suggestions = suggest_strategies(context, top_k=top_k * 2)

        # Apply learning adjustments
        adjusted = self._apply_learning(context, suggestions)

        return adjusted[:top_k]

    def _apply_learning(self, context: StrategyContext, suggestions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Adjust suggestions based on historical feedback"""

        # Find similar historical contexts
        similar = [h for h in self.history if self._is_similar(context, h[0])]

        if not similar:
            return suggestions

        # Calculate success rates for each strategy
        strategy_success = {}
        for _, strategy_name, was_helpful in similar:
            if strategy_name not in strategy_success:
                strategy_success[strategy_name] = []
            strategy_success[strategy_name].append(1.0 if was_helpful else 0.0)

        # Adjust confidence scores
        adjusted = []
        for name, confidence in suggestions:
            if name in strategy_success:
                success_rate = sum(strategy_success[name]) / len(strategy_success[name])
                # Blend base confidence with learned success rate
                adjusted_confidence = 0.7 * confidence + 0.3 * success_rate
            else:
                adjusted_confidence = confidence

            adjusted.append((name, adjusted_confidence))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _is_similar(self, ctx1: StrategyContext, ctx2: StrategyContext) -> bool:
        """Check if two contexts are similar"""
        # Simple similarity based on keyword overlap
        words1 = set(ctx1.query.lower().split())
        words2 = set(ctx2.query.lower().split())

        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5

    def record_feedback(self, context: StrategyContext, strategy_name: str, was_helpful: bool):
        """Record feedback for learning"""
        self.history.append((context, strategy_name, was_helpful))

        # Trim history to last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
```

---

## Configuration-Driven Strategy Creation

### Add New Strategy Without Code

```yaml
# promptly_skills/strategies/custom_security_audit/config.yaml

name: security_audit
version: 1.0.0
category: scaffolding
description: Comprehensive security audit with OWASP Top 10 coverage

# Strategy type (determines which base class to use)
type: template  # template, composite, llm-powered

# Template configuration
template:
  file: template.md
  variables:
    audit_depth: comprehensive  # quick, standard, comprehensive
    owasp_version: 2021
    focus_areas:
      - Authentication & Session Management
      - Input Validation
      - Cryptography
      - Access Control
      - Error Handling

# Auto-detection rules
detection:
  high_confidence_keywords:
    - security audit
    - vulnerability scan
    - penetration test
  medium_confidence_keywords:
    - review security
    - check for vulnerabilities
  context_triggers:
    file_patterns:
      - "**/auth/**"
      - "**/security/**"
      - "**/crypto/**"

  # Confidence calculation
  base_confidence: 0.3
  keyword_boost: 0.4
  context_boost: 0.3

# Composability
composable_with:
  - verify  # Often want to verify security findings
  - challenge  # Adversarial testing complements audit

recommended_after:
  - code_review  # Security audit after code review

# Performance
performance:
  timeout_seconds: 60
  max_iterations: 1
  cache_results: true
```

This YAML automatically generates a working strategy!

---

## Benefits of This Design

### 1. Extensibility ⭐⭐⭐⭐⭐
- Add new strategy: Drop YAML + template in `strategies/` directory
- No code changes needed
- Auto-discovered by registry

### 2. Elegance ⭐⭐⭐⭐⭐
- Single command: `/strategy <name> <query>`
- Composable: `verify+challenge+deep`
- Auto-detection: `/strategy auto <query>`

### 3. Maintainability ⭐⭐⭐⭐⭐
- Each strategy is isolated
- Common interface via base class
- Easy to test individually

### 4. Discoverability ⭐⭐⭐⭐⭐
- `/strategy list` shows all strategies
- Auto-detection suggests best strategy
- Learning from feedback improves suggestions

### 5. Composability ⭐⭐⭐⭐⭐
- Chain strategies: `verify+challenge`
- Strategies can recommend follow-ups
- Composite strategies are first-class

---

## Implementation Phases (Revised)

### Phase 1: Core Framework (Week 1) 🔥
**Goal:** Build the strategy pattern infrastructure

**Deliverables:**
1. `HoloLoom/prompting/strategy.py` - Base interface
2. `HoloLoom/prompting/registry.py` - Auto-discovery
3. `HoloLoom/prompting/composite.py` - Chaining
4. `HoloLoom/prompting/auto_detect.py` - Auto-detection

**Success:** Can load and execute a single strategy

---

### Phase 2: First 3 Strategies (Week 2) 🔥
**Goal:** Implement verify, challenge, optimize

**Deliverables:**
1. `promptly_skills/strategies/verify/` - Chain of Verification
2. `promptly_skills/strategies/challenge/` - Adversarial Prompting
3. `promptly_skills/strategies/optimize/` - Recursive Optimization

**Success:** `/strategy verify "query"` works end-to-end

---

### Phase 3: Remaining 7 Strategies (Week 3-4) 🟡
**Goal:** Complete the 10-strategy library

**Deliverables:**
- `edge_cases/`, `reverse/`, `deep/`, `scaffold/`, `prime/`, `debate/`, `temp_sim/`

**Success:** All 10 strategies from video implemented

---

### Phase 4: UI Integration (Week 5) 🟡
**Goal:** VS Code + Matrix bot integration

**Deliverables:**
1. VS Code `/strategy` command with autocomplete
2. Matrix bot strategy handler
3. Webview for strategy results

**Success:** Natural usage from VS Code and Matrix

---

### Phase 5: Auto-Detection & Learning (Week 6) 🟢
**Goal:** Smart strategy suggestions

**Deliverables:**
1. Auto-detector with confidence scoring
2. Feedback collection
3. Learning from usage

**Success:** Auto-detection accuracy >80%

---

### Phase 6: Analytics & Polish (Week 7-8) 🔵
**Goal:** Production-ready with analytics

**Deliverables:**
1. Strategy effectiveness dashboard
2. A/B testing framework
3. Complete documentation

**Success:** Production deployment

---

## Migration Path

### For Users

**Before (10 commands):**
```
/verify
/challenge
/reverse
/optimize
/deep
/scaffold
/prime
/teach
/debate
/temp-sim
```

**After (1 command):**
```
/strategy verify
/strategy challenge
/strategy reverse
...

# Or just:
/strategy auto
```

### Backward Compatibility

```python
# Keep legacy commands as aliases
LEGACY_ALIASES = {
    'verify': 'strategy verify',
    'challenge': 'strategy challenge',
    'optimize': 'strategy optimize',
    # ... etc
}
```

---

## Code Volume Comparison

### Approach 1: 10 Separate Commands
```
10 commands × 500 lines each = 5,000 lines
10 VS Code integrations = 2,000 lines
10 Matrix bot handlers = 2,000 lines
---
TOTAL: ~9,000 lines
```

### Approach 2: Strategy Framework
```
Core framework: 800 lines
10 strategy configs: 100 lines each = 1,000 lines
10 strategy templates: 50 lines each = 500 lines
1 VS Code integration: 300 lines
1 Matrix bot handler: 200 lines
---
TOTAL: ~2,800 lines (70% reduction!)
```

---

## Next Steps

1. **Review this design** - Does the strategy pattern make sense?
2. **Start Phase 1** - Build core framework (Week 1)
3. **Implement verify strategy** - First concrete example
4. **Test composability** - Chain verify+challenge

**Ready to build the elegant, extensible system?** 🚀
