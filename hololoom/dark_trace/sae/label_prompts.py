"""
Dark Trace SAE: Label Prompt Templates
======================================

Structured prompt templates for LLM-powered feature labeling.
Based on Anthropic's feature interpretation methodology.

Templates:
- SINGLE_FEATURE_PROMPT: Label one feature at a time
- BATCH_PROMPT: Efficient batch labeling
- VERIFICATION_PROMPT: Verify/refine existing labels
- COMPARISON_PROMPT: Compare feature behaviors

Created: 2025-12-28
"""

from dataclasses import dataclass
from typing import Any

# =============================================================================
# Single Feature Labeling
# =============================================================================

SINGLE_FEATURE_PROMPT_TEMPLATE = """You are an expert in neural network interpretability, specializing in sparse autoencoder (SAE) feature analysis.

Your task is to describe what concept or pattern an SAE feature detects based on its activation behavior.

## Feature Information

**Feature Index**: {feature_idx}

**Top Activating Examples** (highest activation first):
{examples}

**Activation Statistics**:
{activation_stats}

**Semantic Alignment** (correlation with interpretable dimensions):
{semantic_alignments}

## Instructions

1. Analyze the common theme across the top-activating examples
2. Consider the semantic alignments as supporting evidence
3. Provide a concise, specific description (1-3 sentences)
4. Rate your confidence (0.0-1.0) in the label

## Output Format

Description: [Your description of what this feature detects]
Confidence: [0.0-1.0]
Reasoning: [Brief reasoning for your label]
"""


# =============================================================================
# Batch Feature Labeling
# =============================================================================

BATCH_PROMPT_TEMPLATE = """You are an expert in neural network interpretability, specializing in sparse autoencoder (SAE) feature analysis.

Your task is to label multiple SAE features based on their activation patterns.

## Features to Label

{feature_sections}

## Instructions

For each feature:
1. Identify the common theme across top-activating examples
2. Provide a concise description (1-2 sentences)
3. Rate confidence (0.0-1.0)

## Output Format (JSON)

```json
[
  {{
    "feature_idx": <idx>,
    "description": "<description>",
    "confidence": <0.0-1.0>,
    "primary_concept": "<main concept in 1-3 words>"
  }},
  ...
]
```
"""


FEATURE_SECTION_TEMPLATE = """### Feature {feature_idx}

**Top Examples**:
{examples}

**Pattern**: {activation_pattern}

**Semantic Alignment**: {primary_semantic_axis} (r={alignment_score:.3f})
---
"""


# =============================================================================
# Label Verification
# =============================================================================

VERIFICATION_PROMPT_TEMPLATE = """You are an expert in neural network interpretability.

Your task is to verify and potentially refine an existing feature label.

## Current Label

**Feature {feature_idx}**
- Description: {current_description}
- Confidence: {current_confidence}

## Evidence

**Top Activating Examples**:
{examples}

**Activation Statistics**:
{activation_stats}

**Semantic Alignments**:
{semantic_alignments}

## Instructions

1. Assess whether the current label accurately describes this feature
2. If accurate, confirm with "VERIFIED"
3. If inaccurate or incomplete, provide a refined label

## Output Format

Verdict: [VERIFIED | REFINED | UNCLEAR]
Refined Description: [New description if REFINED, otherwise repeat current]
Confidence: [0.0-1.0]
Reasoning: [Why you verified/refined/found unclear]
"""


# =============================================================================
# Feature Comparison
# =============================================================================

COMPARISON_PROMPT_TEMPLATE = """You are an expert in neural network interpretability.

Your task is to compare two SAE features and describe their relationship.

## Feature A (Index {feature_a_idx})

**Description**: {feature_a_description}
**Top Examples**:
{feature_a_examples}

## Feature B (Index {feature_b_idx})

**Description**: {feature_b_description}
**Top Examples**:
{feature_b_examples}

## Instructions

1. Identify similarities and differences between the features
2. Determine if they are: SYNONYMOUS, RELATED, DISTINCT, or COMPOSITIONAL
3. Explain the relationship

## Output Format

Relationship: [SYNONYMOUS | RELATED | DISTINCT | COMPOSITIONAL]
Similarity Score: [0.0-1.0]
Explanation: [Your analysis of the relationship]
"""


# =============================================================================
# Prompt Builder
# =============================================================================

@dataclass
class LabelPromptContext:
    """Context for building label prompts."""
    feature_idx: int
    examples: list[str]
    activation_values: list[float]
    activation_pattern: str = ""
    semantic_alignments: dict[str, float] = None
    primary_semantic_axis: str | None = None
    alignment_score: float = 0.0

    # For verification
    current_description: str | None = None
    current_confidence: float | None = None

    # Metadata
    max_example_length: int = 200
    max_examples: int = 10

    def __post_init__(self):
        if self.semantic_alignments is None:
            self.semantic_alignments = {}


class LabelPromptBuilder:
    """
    Build prompts for feature labeling.

    Usage:
        builder = LabelPromptBuilder()

        # Single feature
        prompt = builder.build_single_feature_prompt(context)

        # Batch
        prompt = builder.build_batch_prompt([context1, context2, ...])

        # Verification
        prompt = builder.build_verification_prompt(context)
    """

    def __init__(
        self,
        single_template: str = SINGLE_FEATURE_PROMPT_TEMPLATE,
        batch_template: str = BATCH_PROMPT_TEMPLATE,
        verification_template: str = VERIFICATION_PROMPT_TEMPLATE,
        comparison_template: str = COMPARISON_PROMPT_TEMPLATE,
    ):
        self.single_template = single_template
        self.batch_template = batch_template
        self.verification_template = verification_template
        self.comparison_template = comparison_template

    def build_single_feature_prompt(
        self,
        context: LabelPromptContext,
    ) -> str:
        """
        Build prompt for labeling a single feature.

        Args:
            context: Feature context with examples and statistics

        Returns:
            Formatted prompt string
        """
        # Format examples
        examples_text = self._format_examples(
            context.examples,
            context.activation_values,
            max_length=context.max_example_length,
            max_count=context.max_examples,
        )

        # Format activation stats
        activation_stats = self._format_activation_stats(context)

        # Format semantic alignments
        semantic_text = self._format_semantic_alignments(context)

        return self.single_template.format(
            feature_idx=context.feature_idx,
            examples=examples_text,
            activation_stats=activation_stats,
            semantic_alignments=semantic_text,
        )

    def build_batch_prompt(
        self,
        contexts: list[LabelPromptContext],
        max_features: int = 10,
    ) -> str:
        """
        Build prompt for batch labeling multiple features.

        Args:
            contexts: List of feature contexts
            max_features: Maximum features per batch

        Returns:
            Formatted batch prompt
        """
        contexts = contexts[:max_features]

        feature_sections = []
        for ctx in contexts:
            examples_text = self._format_examples(
                ctx.examples,
                ctx.activation_values,
                max_length=150,  # Shorter for batch
                max_count=5,
            )

            section = FEATURE_SECTION_TEMPLATE.format(
                feature_idx=ctx.feature_idx,
                examples=examples_text,
                activation_pattern=ctx.activation_pattern or "Not available",
                primary_semantic_axis=ctx.primary_semantic_axis or "None",
                alignment_score=ctx.alignment_score,
            )
            feature_sections.append(section)

        return self.batch_template.format(
            feature_sections="\n".join(feature_sections)
        )

    def build_verification_prompt(
        self,
        context: LabelPromptContext,
    ) -> str:
        """
        Build prompt for verifying/refining an existing label.

        Args:
            context: Feature context with current label

        Returns:
            Formatted verification prompt
        """
        if context.current_description is None:
            raise ValueError("Verification requires current_description")

        examples_text = self._format_examples(
            context.examples,
            context.activation_values,
            max_length=context.max_example_length,
            max_count=context.max_examples,
        )

        activation_stats = self._format_activation_stats(context)
        semantic_text = self._format_semantic_alignments(context)

        return self.verification_template.format(
            feature_idx=context.feature_idx,
            current_description=context.current_description,
            current_confidence=context.current_confidence or 0.5,
            examples=examples_text,
            activation_stats=activation_stats,
            semantic_alignments=semantic_text,
        )

    def build_comparison_prompt(
        self,
        context_a: LabelPromptContext,
        context_b: LabelPromptContext,
        description_a: str,
        description_b: str,
    ) -> str:
        """
        Build prompt for comparing two features.

        Args:
            context_a: First feature context
            context_b: Second feature context
            description_a: Description of first feature
            description_b: Description of second feature

        Returns:
            Formatted comparison prompt
        """
        examples_a = self._format_examples(
            context_a.examples,
            context_a.activation_values,
            max_length=150,
            max_count=5,
        )

        examples_b = self._format_examples(
            context_b.examples,
            context_b.activation_values,
            max_length=150,
            max_count=5,
        )

        return self.comparison_template.format(
            feature_a_idx=context_a.feature_idx,
            feature_a_description=description_a,
            feature_a_examples=examples_a,
            feature_b_idx=context_b.feature_idx,
            feature_b_description=description_b,
            feature_b_examples=examples_b,
        )

    def _format_examples(
        self,
        examples: list[str],
        activation_values: list[float],
        max_length: int = 200,
        max_count: int = 10,
    ) -> str:
        """Format examples with activation values."""
        if not examples:
            return "No strongly activating examples found."

        lines = []
        for i, (text, act) in enumerate(zip(examples[:max_count], activation_values[:max_count])):
            # Truncate long examples
            if len(text) > max_length:
                text = text[:max_length] + "..."

            lines.append(f"{i+1}. (activation: {act:.3f}) {text}")

        return "\n".join(lines)

    def _format_activation_stats(
        self,
        context: LabelPromptContext,
    ) -> str:
        """Format activation statistics."""
        if not context.activation_values:
            return "No activation data available."

        values = context.activation_values

        max_act = max(values)
        min_act = min(values) if len(values) > 1 else max_act
        mean_act = sum(values) / len(values)

        return (
            f"- Max activation: {max_act:.3f}\n"
            f"- Mean activation (top-k): {mean_act:.3f}\n"
            f"- Min activation (top-k): {min_act:.3f}\n"
            f"- Pattern: {context.activation_pattern or 'Not analyzed'}"
        )

    def _format_semantic_alignments(
        self,
        context: LabelPromptContext,
    ) -> str:
        """Format semantic alignment scores."""
        if not context.semantic_alignments:
            return "No semantic alignment data available."

        # Sort by absolute correlation
        sorted_alignments = sorted(
            context.semantic_alignments.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        lines = []
        for axis, corr in sorted_alignments[:5]:  # Top 5
            direction = "+" if corr > 0 else "-"
            lines.append(f"- {axis}: {direction}{abs(corr):.3f}")

        if context.primary_semantic_axis:
            lines.append(f"\nPrimary axis: {context.primary_semantic_axis}")

        return "\n".join(lines)


# =============================================================================
# Response Parsing
# =============================================================================

@dataclass
class ParsedLabelResponse:
    """Parsed response from labeling prompt."""
    description: str
    confidence: float
    reasoning: str | None = None
    primary_concept: str | None = None

    # For verification
    verdict: str | None = None  # VERIFIED, REFINED, UNCLEAR

    # For comparison
    relationship: str | None = None  # SYNONYMOUS, RELATED, DISTINCT, COMPOSITIONAL
    similarity_score: float | None = None


class LabelResponseParser:
    """
    Parse LLM responses from labeling prompts.

    Usage:
        parser = LabelResponseParser()
        result = parser.parse_single_response(llm_output)
        results = parser.parse_batch_response(llm_output)
    """

    def parse_single_response(
        self,
        response: str,
    ) -> ParsedLabelResponse:
        """
        Parse response from single feature labeling.

        Args:
            response: Raw LLM output

        Returns:
            ParsedLabelResponse
        """
        lines = response.strip().split("\n")

        description = ""
        confidence = 0.5
        reasoning = ""

        for line in lines:
            line = line.strip()

            if line.lower().startswith("description:"):
                description = line[len("description:"):].strip()
            elif line.lower().startswith("confidence:"):
                conf_str = line[len("confidence:"):].strip()
                try:
                    confidence = float(conf_str.replace(",", "."))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    confidence = 0.5
            elif line.lower().startswith("reasoning:"):
                reasoning = line[len("reasoning:"):].strip()

        # Fallback: use whole response as description if not parsed
        if not description:
            description = response.strip()[:500]

        return ParsedLabelResponse(
            description=description,
            confidence=confidence,
            reasoning=reasoning if reasoning else None,
        )

    def parse_batch_response(
        self,
        response: str,
    ) -> list[dict[str, Any]]:
        """
        Parse JSON response from batch labeling.

        Args:
            response: Raw LLM output (should contain JSON array)

        Returns:
            List of parsed label dicts
        """
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return []

        try:
            results = json.loads(json_match.group())

            # Validate and normalize
            normalized = []
            for item in results:
                if "feature_idx" not in item or "description" not in item:
                    continue

                normalized.append({
                    "feature_idx": int(item["feature_idx"]),
                    "description": str(item["description"]),
                    "confidence": float(item.get("confidence", 0.5)),
                    "primary_concept": item.get("primary_concept"),
                })

            return normalized

        except json.JSONDecodeError:
            return []

    def parse_verification_response(
        self,
        response: str,
    ) -> ParsedLabelResponse:
        """
        Parse response from verification prompt.

        Args:
            response: Raw LLM output

        Returns:
            ParsedLabelResponse with verdict
        """
        lines = response.strip().split("\n")

        verdict = "UNCLEAR"
        description = ""
        confidence = 0.5
        reasoning = ""

        for line in lines:
            line = line.strip()

            if line.lower().startswith("verdict:"):
                v = line[len("verdict:"):].strip().upper()
                if v in ("VERIFIED", "REFINED", "UNCLEAR"):
                    verdict = v
            elif line.lower().startswith("refined description:"):
                description = line[len("refined description:"):].strip()
            elif line.lower().startswith("confidence:"):
                try:
                    confidence = float(line[len("confidence:"):].strip())
                except ValueError:
                    pass
            elif line.lower().startswith("reasoning:"):
                reasoning = line[len("reasoning:"):].strip()

        return ParsedLabelResponse(
            description=description,
            confidence=confidence,
            reasoning=reasoning if reasoning else None,
            verdict=verdict,
        )

    def parse_comparison_response(
        self,
        response: str,
    ) -> ParsedLabelResponse:
        """
        Parse response from comparison prompt.

        Args:
            response: Raw LLM output

        Returns:
            ParsedLabelResponse with relationship
        """
        lines = response.strip().split("\n")

        relationship = "DISTINCT"
        similarity = 0.0
        explanation = ""

        for line in lines:
            line = line.strip()

            if line.lower().startswith("relationship:"):
                r = line[len("relationship:"):].strip().upper()
                if r in ("SYNONYMOUS", "RELATED", "DISTINCT", "COMPOSITIONAL"):
                    relationship = r
            elif line.lower().startswith("similarity score:"):
                try:
                    similarity = float(line[len("similarity score:"):].strip())
                except ValueError:
                    pass
            elif line.lower().startswith("explanation:"):
                explanation = line[len("explanation:"):].strip()

        return ParsedLabelResponse(
            description=explanation,
            confidence=similarity,
            relationship=relationship,
            similarity_score=similarity,
        )
