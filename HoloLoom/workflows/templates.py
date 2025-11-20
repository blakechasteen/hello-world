"""
Workflow Templates - Pre-built workflow definitions
====================================================

Provides 8+ ready-to-use workflow templates for common patterns:
1. Simple Q&A
2. Verified Q&A with DS-STAR
3. Auto-refining Q&A
4. Recursive research
5. Multi-strategy parallel
6. Human-in-loop approval
7. Complex decomposition
8. Iterative refinement loop

Author: HoloLoom Team
Date: November 2025
"""

from HoloLoom.workflows.schema import (
    WorkflowDefinition,
    WorkflowNode,
    NodeType,
    RetryPolicy,
)


class WorkflowTemplates:
    """Pre-built workflow templates."""

    @staticmethod
    def simple_qa() -> WorkflowDefinition:
        """
        Simple Q&A: Query → Done.

        Simplest workflow: single RAG query, return response.
        """
        nodes = {
            "query": WorkflowNode(
                id="query",
                type=NodeType.QUERY,
                name="RAG Query",
                params={"mode": "direct", "max_sources": 5},
                description="Execute RAG query",
            )
        }

        return WorkflowDefinition(
            name="Simple Q&A",
            version="1.0.0",
            nodes=nodes,
            entry_point="query",
            metadata={"description": "Basic question answering workflow"},
        )

    @staticmethod
    def verified_qa() -> WorkflowDefinition:
        """
        Verified Q&A: Query → Verify → Done.

        Add DS-STAR verification for quality assurance.
        """
        nodes = {
            "query": WorkflowNode(
                id="query",
                type=NodeType.QUERY,
                name="RAG Query",
                params={"mode": "verify", "max_sources": 5},
                next=["verify"],
                description="Execute RAG query with verification mode",
            ),
            "verify": WorkflowNode(
                id="verify",
                type=NodeType.VERIFY,
                name="DS-STAR Verification",
                params={},
                description="Verify response using DS-STAR framework",
            ),
        }

        return WorkflowDefinition(
            name="Verified Q&A",
            version="1.0.0",
            nodes=nodes,
            entry_point="query",
            metadata={"description": "Q&A with DS-STAR verification"},
        )

    @staticmethod
    def auto_refining_qa() -> WorkflowDefinition:
        """
        Auto-refining Q&A: Query → Verify → Condition → Refine (if low confidence).

        Automatically refine low-confidence responses.
        """
        nodes = {
            "query": WorkflowNode(
                id="query",
                type=NodeType.QUERY,
                name="RAG Query",
                params={"mode": "verify", "max_sources": 5},
                next=["verify"],
            ),
            "verify": WorkflowNode(
                id="verify",
                type=NodeType.VERIFY,
                name="DS-STAR Verification",
                params={},
                next=["condition"],
            ),
            "condition": WorkflowNode(
                id="condition",
                type=NodeType.CONDITION,
                name="Check Confidence",
                params={
                    "condition_type": "confidence",
                    "threshold": 0.75,
                    "branch_false": "refine",
                    "branch_true": "response",
                },
            ),
            "refine": WorkflowNode(
                id="refine",
                type=NodeType.REFINE,
                name="Refine Response",
                params={"strategy": "refine", "max_iterations": 3},
                next=["response"],
            ),
            "response": WorkflowNode(
                id="response",
                type=NodeType.RESPONSE,
                name="Generate Response",
                params={"format": "text"},
            ),
        }

        return WorkflowDefinition(
            name="Auto-Refining Q&A",
            version="1.0.0",
            nodes=nodes,
            entry_point="query",
            metadata={"description": "Q&A with automatic refinement for low confidence"},
        )

    @staticmethod
    def recursive_research() -> WorkflowDefinition:
        """
        Recursive Research: Recursive(Query) → Verify → Learn.

        Deep research using recursive reasoning.
        """
        nodes = {
            "recursive": WorkflowNode(
                id="recursive",
                type=NodeType.RECURSIVE,
                name="Recursive Reasoning",
                params={"max_depth": 5},
                next=["verify"],
                description="Multi-hop recursive reasoning",
            ),
            "verify": WorkflowNode(
                id="verify",
                type=NodeType.VERIFY,
                name="Verify Results",
                params={},
                next=["store"],
            ),
            "store": WorkflowNode(
                id="store",
                type=NodeType.STORE,
                name="Store Learnings",
                params={"backend": "memory"},
                description="Store research findings in memory",
            ),
        }

        return WorkflowDefinition(
            name="Recursive Research",
            version="1.0.0",
            nodes=nodes,
            entry_point="recursive",
            metadata={"description": "Deep research with recursive reasoning"},
        )

    @staticmethod
    def multi_strategy() -> WorkflowDefinition:
        """
        Multi-Strategy: Parallel(Direct, Research, Plan-Execute) → Merge → Pick Best.

        Try multiple strategies in parallel, pick best result.
        """
        nodes = {
            "parallel": WorkflowNode(
                id="parallel",
                type=NodeType.PARALLEL,
                name="Parallel Strategies",
                params={
                    "task_nodes": ["direct", "research", "plan_execute"],
                    "max_concurrent": 3,
                },
                next=["merge"],
                description="Execute multiple strategies in parallel",
            ),
            "direct": WorkflowNode(
                id="direct",
                type=NodeType.QUERY,
                name="Direct Query",
                params={"mode": "direct", "max_sources": 5},
            ),
            "research": WorkflowNode(
                id="research",
                type=NodeType.QUERY,
                name="Research Query",
                params={"mode": "research", "max_sources": 10},
            ),
            "plan_execute": WorkflowNode(
                id="plan_execute",
                type=NodeType.QUERY,
                name="Plan-Execute Query",
                params={"mode": "plan_execute", "max_sources": 8},
            ),
            "merge": WorkflowNode(
                id="merge",
                type=NodeType.MERGE,
                name="Merge Results",
                params={"merge_strategy": "best_confidence"},
                next=["response"],
            ),
            "response": WorkflowNode(
                id="response",
                type=NodeType.RESPONSE,
                name="Final Response",
                params={"format": "text"},
            ),
        }

        return WorkflowDefinition(
            name="Multi-Strategy Parallel",
            version="1.0.0",
            nodes=nodes,
            entry_point="parallel",
            metadata={"description": "Parallel strategy execution with best-result selection"},
        )

    @staticmethod
    def human_in_loop() -> WorkflowDefinition:
        """
        Human-in-Loop: Query → Human Review → Execute (if approved) → Learn.

        Require human approval before execution.
        """
        nodes = {
            "query": WorkflowNode(
                id="query",
                type=NodeType.QUERY,
                name="RAG Query",
                params={"mode": "verify", "max_sources": 5},
                next=["human_review"],
            ),
            "human_review": WorkflowNode(
                id="human_review",
                type=NodeType.HUMAN_IN_LOOP,
                name="Human Approval",
                params={"timeout_seconds": 300},
                next=["condition"],
                timeout_seconds=300,
            ),
            "condition": WorkflowNode(
                id="condition",
                type=NodeType.CONDITION,
                name="Check Approval",
                params={
                    "condition_type": "approved",
                    "branch_true": "execute",
                    "branch_false": "reject",
                },
            ),
            "execute": WorkflowNode(
                id="execute",
                type=NodeType.RESPONSE,
                name="Execute Approved Action",
                params={"format": "text"},
                next=["store"],
            ),
            "store": WorkflowNode(
                id="store",
                type=NodeType.STORE,
                name="Store Result",
                params={"backend": "memory"},
            ),
            "reject": WorkflowNode(
                id="reject",
                type=NodeType.RESPONSE,
                name="Rejected Response",
                params={"format": "text", "template": "Action rejected by human reviewer"},
            ),
        }

        return WorkflowDefinition(
            name="Human-in-Loop Approval",
            version="1.0.0",
            nodes=nodes,
            entry_point="query",
            metadata={"description": "Workflow with human approval gate"},
        )

    @staticmethod
    def complex_decomposition() -> WorkflowDefinition:
        """
        Complex Decomposition: Multi-Query → Parallel(Sub-queries) → Synthesize → Verify.

        Break complex queries into sub-queries, execute in parallel, synthesize results.
        """
        nodes = {
            "multiquery": WorkflowNode(
                id="multiquery",
                type=NodeType.MULTIQUERY,
                name="Decompose Query",
                params={"max_subqueries": 5},
                next=["parallel"],
                description="Break complex query into sub-queries",
            ),
            "parallel": WorkflowNode(
                id="parallel",
                type=NodeType.PARALLEL,
                name="Execute Sub-Queries",
                params={"task_nodes": ["subquery"], "max_concurrent": 5},
                next=["synthesize"],
            ),
            "subquery": WorkflowNode(
                id="subquery",
                type=NodeType.QUERY,
                name="Execute Sub-Query",
                params={"mode": "direct", "max_sources": 3},
            ),
            "synthesize": WorkflowNode(
                id="synthesize",
                type=NodeType.SYNTHESIZE,
                name="Synthesize Results",
                params={},
                next=["verify"],
                description="Combine sub-query results",
            ),
            "verify": WorkflowNode(
                id="verify",
                type=NodeType.VERIFY,
                name="Verify Synthesis",
                params={},
                next=["response"],
            ),
            "response": WorkflowNode(
                id="response",
                type=NodeType.RESPONSE,
                name="Final Response",
                params={"format": "text"},
            ),
        }

        return WorkflowDefinition(
            name="Complex Decomposition",
            version="1.0.0",
            nodes=nodes,
            entry_point="multiquery",
            metadata={"description": "Decompose complex queries into parallel sub-queries"},
        )

    @staticmethod
    def iterative_refinement() -> WorkflowDefinition:
        """
        Iterative Refinement: Loop(Query → Refine) until convergence.

        Iteratively refine until quality threshold met or max iterations reached.
        """
        nodes = {
            "loop": WorkflowNode(
                id="loop",
                type=NodeType.LOOP,
                name="Refinement Loop",
                params={
                    "max_iterations": 5,
                    "condition_node": "check_quality",
                    "body_node": "refine_step",
                },
                next=["response"],
                description="Iteratively refine until convergence",
            ),
            "refine_step": WorkflowNode(
                id="refine_step",
                type=NodeType.REFINE,
                name="Refinement Step",
                params={"strategy": "elegance", "max_iterations": 1},
                next=["check_quality"],
            ),
            "check_quality": WorkflowNode(
                id="check_quality",
                type=NodeType.CONDITION,
                name="Check Quality",
                params={
                    "condition_type": "confidence",
                    "threshold": 0.95,
                },
                description="Check if quality threshold reached",
            ),
            "response": WorkflowNode(
                id="response",
                type=NodeType.RESPONSE,
                name="Final Response",
                params={"format": "text"},
            ),
        }

        return WorkflowDefinition(
            name="Iterative Refinement Loop",
            version="1.0.0",
            nodes=nodes,
            entry_point="loop",
            metadata={"description": "Iterative refinement until quality convergence"},
        )

    @staticmethod
    def safety_gated() -> WorkflowDefinition:
        """
        Safety-Gated: Query → Safety Check → Conditional Execute.

        Execute action only if safety check passes.
        """
        nodes = {
            "query": WorkflowNode(
                id="query",
                type=NodeType.QUERY,
                name="RAG Query",
                params={"mode": "verify", "max_sources": 5},
                next=["safety"],
            ),
            "safety": WorkflowNode(
                id="safety",
                type=NodeType.SAFETY,
                name="Safety Guardrails",
                params={"risk_threshold": "MEDIUM"},
                next=["condition"],
                description="Check action safety",
            ),
            "condition": WorkflowNode(
                id="condition",
                type=NodeType.CONDITION,
                name="Check Safety",
                params={
                    "condition_type": "safety_allowed",
                    "branch_true": "execute",
                    "branch_false": "block",
                },
            ),
            "execute": WorkflowNode(
                id="execute",
                type=NodeType.RESPONSE,
                name="Execute Safe Action",
                params={"format": "text"},
            ),
            "block": WorkflowNode(
                id="block",
                type=NodeType.RESPONSE,
                name="Blocked Response",
                params={"format": "text", "template": "Action blocked by safety guardrails: {reason}"},
            ),
        }

        return WorkflowDefinition(
            name="Safety-Gated Workflow",
            version="1.0.0",
            nodes=nodes,
            entry_point="query",
            metadata={"description": "Workflow with safety guardrails"},
        )

    @classmethod
    def get_all_templates(cls) -> dict:
        """Get all available templates."""
        return {
            "simple_qa": cls.simple_qa(),
            "verified_qa": cls.verified_qa(),
            "auto_refining_qa": cls.auto_refining_qa(),
            "recursive_research": cls.recursive_research(),
            "multi_strategy": cls.multi_strategy(),
            "human_in_loop": cls.human_in_loop(),
            "complex_decomposition": cls.complex_decomposition(),
            "iterative_refinement": cls.iterative_refinement(),
            "safety_gated": cls.safety_gated(),
        }

    @classmethod
    def list_templates(cls) -> list:
        """List available template names."""
        return list(cls.get_all_templates().keys())
