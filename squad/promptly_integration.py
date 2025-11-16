#!/usr/bin/env python3
"""
Promptly Integration for Squad
================================
Comprehensive prompt management system for Squad's LLM interactions.

Features:
- Centralized prompt management
- Version control for prompts
- A/B testing support
- Performance tracking
- User customization
- Template engine

Author: Claude Code
Date: November 16, 2025
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class PromptType(Enum):
    """Types of prompts in Squad"""
    SYSTEM = "system"          # LLM system prompts
    TASK = "task"              # Task-specific prompts
    CONTEXT = "context"        # RAG context formatting
    REFLECTION = "reflection"  # Reflection/learning prompts


class PromptProvider(Enum):
    """LLM providers"""
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    UNIVERSAL = "universal"    # Works with all providers


@dataclass
class PromptTemplate:
    """A prompt template with metadata"""
    id: str
    name: str
    type: PromptType
    provider: PromptProvider
    template: str
    variables: List[str]
    version: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0

    def render(self, **kwargs) -> str:
        """Render template with variables"""
        rendered = self.template
        for var in self.variables:
            if var in kwargs:
                placeholder = f"{{{var}}}"
                rendered = rendered.replace(placeholder, str(kwargs[var]))
        return rendered

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['provider'] = self.provider.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary"""
        data['type'] = PromptType(data['type'])
        data['provider'] = PromptProvider(data['provider'])
        return cls(**data)


class PromptRegistry:
    """
    Central registry for all prompts in Squad.

    Features:
    - Store and retrieve prompts
    - Version control
    - Performance tracking
    - A/B testing support
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize prompt registry"""
        self.storage_path = storage_path or Path("./prompts")
        self.storage_path.mkdir(exist_ok=True)

        self.prompts: Dict[str, PromptTemplate] = {}
        self.versions: Dict[str, List[PromptTemplate]] = {}
        self.ab_tests: Dict[str, Dict[str, Any]] = {}

        # Load existing prompts
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from storage"""
        prompts_file = self.storage_path / "prompts.json"
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                data = json.load(f)
                for prompt_data in data.get('prompts', []):
                    prompt = PromptTemplate.from_dict(prompt_data)
                    self.prompts[prompt.id] = prompt

                    # Track versions
                    if prompt.name not in self.versions:
                        self.versions[prompt.name] = []
                    self.versions[prompt.name].append(prompt)

                self.ab_tests = data.get('ab_tests', {})

    def _save_prompts(self):
        """Save prompts to storage"""
        prompts_file = self.storage_path / "prompts.json"
        data = {
            'prompts': [p.to_dict() for p in self.prompts.values()],
            'ab_tests': self.ab_tests,
            'updated_at': datetime.now().isoformat()
        }
        with open(prompts_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        template: str,
        type: PromptType,
        provider: PromptProvider = PromptProvider.UNIVERSAL,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptTemplate:
        """
        Register a new prompt template.

        Args:
            name: Prompt name (e.g., "code_generation_system")
            template: Template string with {variables}
            type: PromptType
            provider: LLM provider
            variables: List of variable names
            metadata: Additional metadata

        Returns:
            Created PromptTemplate
        """
        # Generate unique ID
        prompt_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        # Auto-detect variables if not provided
        if variables is None:
            import re
            variables = list(set(re.findall(r'\{(\w+)\}', template)))

        # Create prompt
        prompt = PromptTemplate(
            id=prompt_id,
            name=name,
            type=type,
            provider=provider,
            template=template,
            variables=variables,
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata=metadata or {},
            performance_score=0.0,
            usage_count=0
        )

        # Store
        self.prompts[prompt_id] = prompt

        # Track version
        if name not in self.versions:
            self.versions[name] = []
        self.versions[name].append(prompt)

        # Save
        self._save_prompts()

        return prompt

    def get(
        self,
        name: str,
        provider: Optional[PromptProvider] = None,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get prompt by name.

        Args:
            name: Prompt name
            provider: Filter by provider (optional)
            version: Specific version (optional, defaults to latest)

        Returns:
            PromptTemplate or None
        """
        candidates = self.versions.get(name, [])

        # Filter by provider
        if provider:
            candidates = [p for p in candidates if p.provider == provider or p.provider == PromptProvider.UNIVERSAL]

        # Filter by version
        if version:
            candidates = [p for p in candidates if p.version == version]

        # Return latest
        if candidates:
            return sorted(candidates, key=lambda p: p.updated_at, reverse=True)[0]

        return None

    def update_performance(
        self,
        prompt_id: str,
        score: float,
        feedback: Optional[Dict[str, Any]] = None
    ):
        """
        Update prompt performance based on usage.

        Args:
            prompt_id: Prompt ID
            score: Performance score (0.0-1.0)
            feedback: Optional feedback data
        """
        if prompt_id not in self.prompts:
            return

        prompt = self.prompts[prompt_id]

        # Update running average
        total_score = prompt.performance_score * prompt.usage_count
        prompt.usage_count += 1
        prompt.performance_score = (total_score + score) / prompt.usage_count
        prompt.updated_at = datetime.now().isoformat()

        # Store feedback
        if feedback:
            if 'feedback' not in prompt.metadata:
                prompt.metadata['feedback'] = []
            prompt.metadata['feedback'].append({
                'timestamp': datetime.now().isoformat(),
                'score': score,
                'data': feedback
            })

        self._save_prompts()

    def create_ab_test(
        self,
        name: str,
        prompt_a_id: str,
        prompt_b_id: str,
        traffic_split: float = 0.5
    ) -> str:
        """
        Create an A/B test between two prompts.

        Args:
            name: Test name
            prompt_a_id: First prompt ID
            prompt_b_id: Second prompt ID
            traffic_split: Percentage to route to A (0.0-1.0)

        Returns:
            Test ID
        """
        import random

        test_id = hashlib.md5(f"{name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

        self.ab_tests[test_id] = {
            'name': name,
            'prompt_a': prompt_a_id,
            'prompt_b': prompt_b_id,
            'traffic_split': traffic_split,
            'created_at': datetime.now().isoformat(),
            'results': {
                'a': {'count': 0, 'total_score': 0.0},
                'b': {'count': 0, 'total_score': 0.0}
            }
        }

        self._save_prompts()
        return test_id

    def get_ab_test_prompt(self, test_id: str) -> Optional[PromptTemplate]:
        """
        Get prompt for A/B test (randomly selected).

        Args:
            test_id: Test ID

        Returns:
            Selected prompt or None
        """
        import random

        if test_id not in self.ab_tests:
            return None

        test = self.ab_tests[test_id]
        use_a = random.random() < test['traffic_split']

        prompt_id = test['prompt_a'] if use_a else test['prompt_b']
        return self.prompts.get(prompt_id)

    def record_ab_test_result(
        self,
        test_id: str,
        prompt_id: str,
        score: float
    ):
        """Record A/B test result"""
        if test_id not in self.ab_tests:
            return

        test = self.ab_tests[test_id]

        # Determine which variant
        variant = 'a' if prompt_id == test['prompt_a'] else 'b'

        # Update results
        test['results'][variant]['count'] += 1
        test['results'][variant]['total_score'] += score

        self._save_prompts()

    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results with statistical analysis"""
        if test_id not in self.ab_tests:
            return None

        test = self.ab_tests[test_id]
        results = test['results']

        # Calculate averages
        avg_a = results['a']['total_score'] / max(results['a']['count'], 1)
        avg_b = results['b']['total_score'] / max(results['b']['count'], 1)

        # Determine winner
        winner = 'a' if avg_a > avg_b else 'b'
        confidence = abs(avg_a - avg_b) / max(avg_a, avg_b) if max(avg_a, avg_b) > 0 else 0

        return {
            'test_id': test_id,
            'name': test['name'],
            'prompt_a': test['prompt_a'],
            'prompt_b': test['prompt_b'],
            'results': {
                'a': {
                    'count': results['a']['count'],
                    'avg_score': avg_a
                },
                'b': {
                    'count': results['b']['count'],
                    'avg_score': avg_b
                }
            },
            'winner': winner,
            'confidence': confidence,
            'improvement': abs(avg_a - avg_b)
        }

    def get_leaderboard(self, type: Optional[PromptType] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get prompt leaderboard by performance.

        Args:
            type: Filter by type (optional)
            limit: Max results

        Returns:
            List of prompts sorted by performance
        """
        prompts = list(self.prompts.values())

        # Filter by type
        if type:
            prompts = [p for p in prompts if p.type == type]

        # Sort by performance
        prompts.sort(key=lambda p: p.performance_score, reverse=True)

        return [
            {
                'id': p.id,
                'name': p.name,
                'type': p.type.value,
                'provider': p.provider.value,
                'performance_score': p.performance_score,
                'usage_count': p.usage_count,
                'version': p.version
            }
            for p in prompts[:limit]
        ]


# ============================================================================
# Default Squad Prompts
# ============================================================================

def initialize_squad_prompts(registry: PromptRegistry):
    """Initialize default prompts for Squad"""

    # Code Generation System Prompt
    registry.register(
        name="code_generation_system",
        template="""You are an expert code generation assistant. Your task is to generate high-quality, production-ready code based on user descriptions.

Follow these principles:
1. Write clean, readable, well-documented code
2. Follow language-specific best practices
3. Include type hints where applicable
4. Add error handling
5. Write idiomatic code for the target language
6. Include usage examples when helpful

Language: {language}
Task Type: {task_type}

Generate code that is:
- Correct and functional
- Well-structured and maintainable
- Properly documented
- Following modern best practices""",
        type=PromptType.SYSTEM,
        provider=PromptProvider.UNIVERSAL,
        variables=["language", "task_type"]
    )

    # RAG Context Formatting
    registry.register(
        name="rag_context_formatting",
        template="""# Relevant Context from Project

The following context has been retrieved from your codebase, APIs, documentation, and forums. Use this to inform your code generation.

{context_items}

# User Request

{user_request}

Please generate code that follows the patterns and conventions shown in the context above.""",
        type=PromptType.CONTEXT,
        provider=PromptProvider.UNIVERSAL,
        variables=["context_items", "user_request"]
    )

    # Code Review System Prompt
    registry.register(
        name="code_review_system",
        template="""You are an expert code reviewer. Analyze the provided code and identify:

1. **Bugs and Errors**: Logical errors, edge cases, potential crashes
2. **Security Issues**: SQL injection, XSS, auth problems, etc.
3. **Performance**: Inefficient algorithms, memory leaks, bottlenecks
4. **Best Practices**: Code style, naming, structure
5. **Suggestions**: How to improve the code

Be specific and constructive. For each issue:
- Explain what's wrong
- Why it's a problem
- How to fix it

Language: {language}""",
        type=PromptType.SYSTEM,
        provider=PromptProvider.UNIVERSAL,
        variables=["language"]
    )

    # Refactoring Prompt
    registry.register(
        name="code_refactoring",
        template="""Refactor the following code according to these instructions:

**Instructions**: {instructions}

**Original Code**:
```{language}
{code}
```

Provide:
1. Refactored code
2. Explanation of changes
3. Unified diff showing modifications

Focus on:
- Maintaining functionality
- Improving readability
- Following {language} best practices
- Applying the requested changes""",
        type=PromptType.TASK,
        provider=PromptProvider.UNIVERSAL,
        variables=["instructions", "code", "language"]
    )

    # Bug Fixing Prompt
    registry.register(
        name="bug_fixing",
        template="""Fix the bug in the following code:

**Error Message**: {error_message}

**Code**:
```{language}
{code}
```

Provide:
1. Fixed code
2. Explanation of the bug
3. How the fix works
4. How to prevent similar bugs in the future

The fix should:
- Resolve the error
- Maintain existing functionality
- Add appropriate error handling
- Follow {language} best practices""",
        type=PromptType.TASK,
        provider=PromptProvider.UNIVERSAL,
        variables=["error_message", "code", "language"]
    )

    # Test Generation Prompt
    registry.register(
        name="test_generation",
        template="""Generate comprehensive tests for the following code:

**Code**:
```{language}
{code}
```

**Test Framework**: {test_framework}

Generate tests that cover:
1. Normal cases (happy path)
2. Edge cases
3. Error cases
4. Boundary conditions

Include:
- Setup/teardown if needed
- Clear test names
- Assertions for expected behavior
- Comments explaining test purpose""",
        type=PromptType.TASK,
        provider=PromptProvider.UNIVERSAL,
        variables=["code", "language", "test_framework"]
    )

    # Code Explanation Prompt
    registry.register(
        name="code_explanation",
        template="""Explain the following code in detail:

```{language}
{code}
```

{question}

Provide:
1. **High-level overview**: What the code does
2. **Step-by-step breakdown**: How it works
3. **Key concepts**: Important algorithms/patterns used
4. **Complexity analysis**: Time/space complexity if applicable

Make the explanation:
- Clear and accessible
- Technically accurate
- With concrete examples where helpful""",
        type=PromptType.TASK,
        provider=PromptProvider.UNIVERSAL,
        variables=["code", "language", "question"]
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of using Promptly integration"""

    # Create registry
    registry = PromptRegistry(storage_path=Path("./squad_prompts"))

    # Initialize default prompts
    initialize_squad_prompts(registry)

    # Get a prompt
    prompt = registry.get("code_generation_system")
    if prompt:
        rendered = prompt.render(
            language="python",
            task_type="generate"
        )
        print("System Prompt:")
        print(rendered)

    # Update performance
    registry.update_performance(
        prompt_id=prompt.id,
        score=0.92,
        feedback={"quality": "excellent", "helpful": True}
    )

    # Create A/B test
    # First, create variant prompt
    variant = registry.register(
        name="code_generation_system_v2",
        template="Enhanced version of system prompt...",
        type=PromptType.SYSTEM,
        provider=PromptProvider.UNIVERSAL
    )

    test_id = registry.create_ab_test(
        name="System Prompt Optimization",
        prompt_a_id=prompt.id,
        prompt_b_id=variant.id,
        traffic_split=0.5
    )

    # Get leaderboard
    leaderboard = registry.get_leaderboard(type=PromptType.SYSTEM, limit=5)
    print("\nTop System Prompts:")
    for idx, p in enumerate(leaderboard, 1):
        print(f"{idx}. {p['name']}: {p['performance_score']:.2f} ({p['usage_count']} uses)")


if __name__ == "__main__":
    print("Promptly Integration for Squad")
    print("=" * 80)
    example_usage()
