#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Planner Component
========================
Analyzes query intent and creates execution plans.

Author: Claude Code
Date: 2025-11-15
Phase: 1 (Foundation)
"""

import logging
from typing import List, Optional
import re

from HoloLoom.documentation.types import Query, Features, Context
from HoloLoom.reasoning.types import (
    QueryIntent,
    QueryType,
    QueryPlan,
    PlanStep,
    QUERY_TYPE_COMPLEXITY,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Query Intent Analysis
# ============================================================================

class QueryPlanner:
    """
    Analyzes query intent and creates execution plans.

    Phase 1: Rule-based intent classification
    Future: ML-based intent prediction
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QueryPlanner")

        # Intent classification patterns
        self.patterns = {
            QueryType.FACTUAL: [
                r'\b(what is|what are|define|definition of)\b',
                r'\b(who is|who was|who are)\b',
                r'\b(when did|when was|when were)\b',
                r'\b(where is|where are|where was)\b',
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|how does|how can)\b',
                r'\b(steps to|process for|method for)\b',
                r'\b(explain how|show me how)\b',
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|contrast|difference between)\b',
                r'\b(versus|vs\.?|compared to)\b',
                r'\b(better than|worse than|similar to)\b',
            ],
            QueryType.ANALYTICAL: [
                r'\b(why does|why is|why are|why did)\b',
                r'\b(analyze|analysis|reason for)\b',
                r'\b(cause of|consequence of|impact of)\b',
            ],
            QueryType.CREATIVE: [
                r'\b(design|create|build|develop)\b',
                r'\b(generate|produce|make)\b',
                r'\b(propose|suggest|recommend)\b',
            ],
            QueryType.VERIFICATION: [
                r'\b(is it true|verify|check|confirm)\b',
                r'\b(true or false|correct or incorrect)\b',
                r'\b(validate|proof|prove)\b',
            ],
        }

    def analyze_intent(self, query: Query, features: Features) -> QueryIntent:
        """
        Analyze query intent.

        Args:
            query: Input query
            features: Extracted features

        Returns:
            QueryIntent with classification and complexity
        """
        query_text = query.text.lower()

        # Classify query type
        query_type = self._classify_type(query_text)

        # Determine requirements
        requirements = self._determine_requirements(query_type, features)

        # Estimate complexity
        complexity = self._estimate_complexity(query_type, query_text, features)

        # Extract key concepts from motifs
        key_concepts = features.motifs[:5] if features.motifs else []

        # Confidence in classification
        confidence = self._classification_confidence(query_type, query_text)

        return QueryIntent(
            type=query_type,
            requirements=requirements,
            complexity=complexity,
            confidence=confidence,
            key_concepts=key_concepts,
        )

    def _classify_type(self, query_text: str) -> QueryType:
        """
        Classify query type using pattern matching.

        Args:
            query_text: Lowercased query text

        Returns:
            Most likely QueryType
        """
        scores = {qtype: 0 for qtype in QueryType}

        for qtype, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_text, re.IGNORECASE):
                    scores[qtype] += 1

        # Return type with highest score, default to FACTUAL
        max_type = max(scores.items(), key=lambda x: x[1])
        return max_type[0] if max_type[1] > 0 else QueryType.FACTUAL

    def _determine_requirements(
        self,
        query_type: QueryType,
        features: Features
    ) -> List[str]:
        """
        Determine what's needed to answer this query.

        Args:
            query_type: Classified query type
            features: Extracted features

        Returns:
            List of requirements
        """
        requirements = []

        # Base requirements by type
        type_requirements = {
            QueryType.FACTUAL: ["definition", "basic facts"],
            QueryType.PROCEDURAL: ["step-by-step instructions", "examples"],
            QueryType.COMPARATIVE: ["multi-source evidence", "contrasting points"],
            QueryType.ANALYTICAL: ["causal reasoning", "supporting evidence"],
            QueryType.CREATIVE: ["domain knowledge", "synthesis"],
            QueryType.VERIFICATION: ["multiple sources", "fact-checking"],
        }

        requirements.extend(type_requirements.get(query_type, []))

        # Add context-specific requirements
        if features.motifs:
            requirements.append("motif-guided retrieval")

        if len(features.embeddings) > 1:
            requirements.append("multi-scale retrieval")

        return requirements

    def _estimate_complexity(
        self,
        query_type: QueryType,
        query_text: str,
        features: Features
    ) -> float:
        """
        Estimate query complexity [0.0, 1.0].

        Args:
            query_type: Classified query type
            query_text: Query text
            features: Extracted features

        Returns:
            Complexity score [0.0, 1.0]
        """
        # Base complexity from query type
        base_complexity = QUERY_TYPE_COMPLEXITY.get(query_type, 0.5)

        # Adjust for query length (longer = more complex)
        length_factor = min(1.0, len(query_text.split()) / 20.0)

        # Adjust for motif count (more motifs = more complex)
        motif_factor = min(1.0, len(features.motifs) / 5.0) if features.motifs else 0.0

        # Weighted combination
        complexity = (
            0.5 * base_complexity +
            0.3 * length_factor +
            0.2 * motif_factor
        )

        return min(1.0, max(0.0, complexity))

    def _classification_confidence(
        self,
        query_type: QueryType,
        query_text: str
    ) -> float:
        """
        Estimate confidence in type classification.

        Args:
            query_type: Classified type
            query_text: Query text

        Returns:
            Confidence [0.0, 1.0]
        """
        # Count pattern matches
        match_count = 0
        patterns = self.patterns.get(query_type, [])

        for pattern in patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                match_count += 1

        # Confidence based on number of matches
        if match_count >= 2:
            return 0.9
        elif match_count == 1:
            return 0.7
        else:
            return 0.5  # Default classification, low confidence

    def create_plan(
        self,
        query: Query,
        features: Features,
        context: Context
    ) -> QueryPlan:
        """
        Create execution plan for complex query (DEEP mode).

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context

        Returns:
            QueryPlan with sub-steps
        """
        intent = self.analyze_intent(query, features)

        # For Phase 1, create simple linear plan
        # Future: More sophisticated plan generation
        steps = []

        # Step 1: Understand the question
        steps.append(PlanStep(
            question=f"What is the core question about {intent.key_concepts[0] if intent.key_concepts else 'the topic'}?",
            required_for="Understanding scope",
            complexity=0.3
        ))

        # Step 2: Gather evidence
        steps.append(PlanStep(
            question="What evidence is available in the context?",
            required_for="Building answer foundation",
            complexity=0.4
        ))

        # Step 3: Type-specific step
        if intent.type == QueryType.COMPARATIVE:
            steps.append(PlanStep(
                question="What are the key differences and similarities?",
                required_for="Comparative analysis",
                complexity=0.6
            ))
        elif intent.type == QueryType.ANALYTICAL:
            steps.append(PlanStep(
                question="What are the causal relationships?",
                required_for="Analytical reasoning",
                complexity=0.7
            ))
        else:
            steps.append(PlanStep(
                question="How do the pieces fit together?",
                required_for="Synthesis",
                complexity=0.5
            ))

        # Step 4: Verify and synthesize
        steps.append(PlanStep(
            question="Is the answer complete and consistent?",
            required_for="Final verification",
            complexity=0.3
        ))

        return QueryPlan(
            steps=steps,
            estimated_complexity=intent.complexity,
            dependencies={
                1: [0],  # Step 1 depends on step 0
                2: [1],  # Step 2 depends on step 1
                3: [2],  # Step 3 depends on step 2
            }
        )
