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
from typing import List, Optional, Dict
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
        if features.motifs:
            # Motifs are Motif objects with .pattern attribute
            key_concepts = [m.pattern if hasattr(m, 'pattern') else str(m)
                          for m in features.motifs[:5]]
        else:
            key_concepts = []

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
        motif_count = len(features.motifs) if features.motifs else 0
        motif_factor = min(1.0, motif_count / 5.0)

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

        Decomposes query into sub-questions with dependency tracking.

        Args:
            query: Input query
            features: Extracted features
            context: Retrieved context

        Returns:
            QueryPlan with sub-steps and dependencies
        """
        intent = self.analyze_intent(query, features)

        # Phase 3: More sophisticated plan generation
        steps = []
        dependencies = {}

        # Decompose based on query type
        if intent.type == QueryType.COMPARATIVE:
            steps = self._create_comparative_plan(intent, features, context)
        elif intent.type == QueryType.ANALYTICAL:
            steps = self._create_analytical_plan(intent, features, context)
        elif intent.type == QueryType.PROCEDURAL:
            steps = self._create_procedural_plan(intent, features, context)
        elif intent.type == QueryType.VERIFICATION:
            steps = self._create_verification_plan(intent, features, context)
        else:
            # Default: factual or creative
            steps = self._create_default_plan(intent, features, context)

        # Build dependency graph (sequential by default, can be customized)
        dependencies = self._build_dependencies(steps, intent)

        return QueryPlan(
            steps=steps,
            estimated_complexity=intent.complexity,
            dependencies=dependencies
        )

    def _create_comparative_plan(
        self,
        intent: QueryIntent,
        features: Features,
        context: Context
    ) -> List[PlanStep]:
        """Create plan for comparative queries."""
        steps = []

        # Step 1: Identify entities to compare
        entities = intent.key_concepts[:2] if len(intent.key_concepts) >= 2 else ["A", "B"]
        steps.append(PlanStep(
            question=f"What are we comparing: {entities[0]} vs {entities[1]}?",
            required_for="Identifying comparison subjects",
            complexity=0.3
        ))

        # Step 2: Gather evidence for first entity
        steps.append(PlanStep(
            question=f"What are the key characteristics of {entities[0]}?",
            required_for="Understanding first subject",
            complexity=0.5
        ))

        # Step 3: Gather evidence for second entity
        steps.append(PlanStep(
            question=f"What are the key characteristics of {entities[1]}?",
            required_for="Understanding second subject",
            complexity=0.5
        ))

        # Step 4: Identify differences
        steps.append(PlanStep(
            question=f"What are the key differences between {entities[0]} and {entities[1]}?",
            required_for="Comparative analysis",
            complexity=0.6
        ))

        # Step 5: Identify similarities
        steps.append(PlanStep(
            question=f"What are the similarities between {entities[0]} and {entities[1]}?",
            required_for="Complete comparison",
            complexity=0.5
        ))

        # Step 6: Synthesize comparison
        steps.append(PlanStep(
            question="Which is better suited for different use cases?",
            required_for="Final synthesis",
            complexity=0.4
        ))

        return steps

    def _create_analytical_plan(
        self,
        intent: QueryIntent,
        features: Features,
        context: Context
    ) -> List[PlanStep]:
        """Create plan for analytical queries."""
        steps = []

        concept = intent.key_concepts[0] if intent.key_concepts else "the phenomenon"

        # Step 1: Understand the phenomenon
        steps.append(PlanStep(
            question=f"What is {concept} and why is it important?",
            required_for="Understanding context",
            complexity=0.4
        ))

        # Step 2: Identify potential causes
        steps.append(PlanStep(
            question=f"What are the potential causes or factors behind {concept}?",
            required_for="Causal analysis",
            complexity=0.7
        ))

        # Step 3: Gather supporting evidence
        steps.append(PlanStep(
            question="What evidence supports each potential cause?",
            required_for="Evidence gathering",
            complexity=0.6
        ))

        # Step 4: Evaluate explanations
        steps.append(PlanStep(
            question="Which explanations are most strongly supported?",
            required_for="Critical evaluation",
            complexity=0.7
        ))

        # Step 5: Consider alternative explanations
        steps.append(PlanStep(
            question="Are there alternative explanations we should consider?",
            required_for="Comprehensive analysis",
            complexity=0.6
        ))

        # Step 6: Synthesize conclusion
        steps.append(PlanStep(
            question=f"What is the most likely explanation for {concept}?",
            required_for="Final conclusion",
            complexity=0.5
        ))

        return steps

    def _create_procedural_plan(
        self,
        intent: QueryIntent,
        features: Features,
        context: Context
    ) -> List[PlanStep]:
        """Create plan for procedural queries."""
        steps = []

        task = intent.key_concepts[0] if intent.key_concepts else "the task"

        # Step 1: Understand goal
        steps.append(PlanStep(
            question=f"What is the desired outcome of {task}?",
            required_for="Goal identification",
            complexity=0.3
        ))

        # Step 2: Prerequisites
        steps.append(PlanStep(
            question=f"What prerequisites or materials are needed for {task}?",
            required_for="Preparation",
            complexity=0.4
        ))

        # Step 3: Main steps
        steps.append(PlanStep(
            question=f"What are the main steps to accomplish {task}?",
            required_for="Procedure identification",
            complexity=0.6
        ))

        # Step 4: Common pitfalls
        steps.append(PlanStep(
            question=f"What are common mistakes or pitfalls when doing {task}?",
            required_for="Risk mitigation",
            complexity=0.5
        ))

        # Step 5: Verification
        steps.append(PlanStep(
            question=f"How can we verify that {task} was completed successfully?",
            required_for="Quality assurance",
            complexity=0.4
        ))

        return steps

    def _create_verification_plan(
        self,
        intent: QueryIntent,
        features: Features,
        context: Context
    ) -> List[PlanStep]:
        """Create plan for verification queries."""
        steps = []

        claim = intent.key_concepts[0] if intent.key_concepts else "the claim"

        # Step 1: Understand the claim
        steps.append(PlanStep(
            question=f"What exactly is being claimed about {claim}?",
            required_for="Claim identification",
            complexity=0.3
        ))

        # Step 2: Find supporting evidence
        steps.append(PlanStep(
            question=f"What evidence supports the claim about {claim}?",
            required_for="Support assessment",
            complexity=0.6
        ))

        # Step 3: Find contradicting evidence
        steps.append(PlanStep(
            question=f"What evidence contradicts the claim about {claim}?",
            required_for="Contradiction assessment",
            complexity=0.6
        ))

        # Step 4: Evaluate sources
        steps.append(PlanStep(
            question="How reliable are the sources of evidence?",
            required_for="Source credibility",
            complexity=0.5
        ))

        # Step 5: Reach verdict
        steps.append(PlanStep(
            question=f"Is the claim about {claim} true, false, or uncertain?",
            required_for="Final verdict",
            complexity=0.4
        ))

        return steps

    def _create_default_plan(
        self,
        intent: QueryIntent,
        features: Features,
        context: Context
    ) -> List[PlanStep]:
        """Create default plan for factual/creative queries."""
        steps = []

        concept = intent.key_concepts[0] if intent.key_concepts else "the topic"

        # Step 1: Understand the question
        steps.append(PlanStep(
            question=f"What is the core question about {concept}?",
            required_for="Understanding scope",
            complexity=0.3
        ))

        # Step 2: Gather relevant information
        steps.append(PlanStep(
            question=f"What information is available about {concept}?",
            required_for="Knowledge gathering",
            complexity=0.5
        ))

        # Step 3: Organize information
        steps.append(PlanStep(
            question=f"How can we organize this information about {concept}?",
            required_for="Structuring knowledge",
            complexity=0.4
        ))

        # Step 4: Synthesize answer
        steps.append(PlanStep(
            question=f"What is the complete answer about {concept}?",
            required_for="Final synthesis",
            complexity=0.5
        ))

        return steps

    def _build_dependencies(
        self,
        steps: List[PlanStep],
        intent: QueryIntent
    ) -> Dict[int, List[int]]:
        """
        Build dependency graph for plan steps.

        Args:
            steps: Plan steps
            intent: Query intent

        Returns:
            Dependency mapping {step_idx: [prerequisite_indices]}
        """
        dependencies = {}

        if intent.type == QueryType.COMPARATIVE:
            # Steps 1,2,3 can be parallel
            # Step 4 depends on 2,3
            # Step 5 depends on 2,3
            # Step 6 depends on 4,5
            dependencies = {
                1: [0],
                2: [0],
                3: [0],
                4: [2, 3],
                5: [2, 3],
            }
            if len(steps) > 5:
                dependencies[5] = [4, 5]

        elif intent.type == QueryType.VERIFICATION:
            # Steps 2 and 3 can be parallel
            # Step 4 depends on both
            dependencies = {
                1: [0],
                2: [1],
                3: [1],
                4: [2, 3],
            }
            if len(steps) > 4:
                dependencies[4] = [3]

        else:
            # Default: sequential dependencies
            for i in range(1, len(steps)):
                dependencies[i] = [i - 1]

        return dependencies
