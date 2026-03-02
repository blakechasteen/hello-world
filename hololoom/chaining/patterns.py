"""
Pre-built Chain Patterns - Common workflow templates

Provides ready-to-use chains for common operations:
- simple_query: Execute only
- verified_query: Execute + verify
- auto_refine: Execute + verify + refine if low confidence
- iterative_improve: Loop execute/refine until high confidence
- multi_strategy: Try multiple strategies in sequence
- research_pipeline: Full cycle with learning

Author: HoloLoom Architecture Team
Date: November 2025
"""

from .chain import Chain, ChainStep, StepType
from .conditions import Conditions, CommonConditions


class ChainPatterns:
    """Pre-built chain patterns for common workflows."""

    @staticmethod
    def simple_query() -> Chain:
        """
        Simple query: Execute only.

        Flow:
            [execute] → [output]

        Use when:
        - Speed is critical
        - Confidence scoring is sufficient
        - No verification needed
        """
        chain = Chain(name="simple_query", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 5,
            },
        ))
        return chain

    @staticmethod
    def verified_query() -> Chain:
        """
        Verified query: Execute + verify.

        Flow:
            [execute] → [verify] → [output]

        Use when:
        - Verification is important
        - Standard quality checks needed
        - Can tolerate ~10-15ms extra latency
        """
        chain = Chain(name="verified_query", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 5,
            },
            next_step="verify",
        ))
        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
        ))
        return chain

    @staticmethod
    def auto_refine() -> Chain:
        """
        Auto-refine: Execute → verify → refine if low confidence.

        Flow:
            [execute]
                ↓
            [verify]
                ↓
            [confidence < 0.75?]
                ├─ Yes → [refine] → [output]
                └─ No → [output]

        Use when:
        - Quality is important
        - Can handle variable latency
        - Want automatic improvement on low confidence
        """
        chain = Chain(name="auto_refine", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 5,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_confidence",
        ))

        # Conditional: Check if confidence is low
        check_step = ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_below(0.75),
            on_success="refine",  # Refine if low confidence
            on_failure=None,  # Skip refine if high confidence
        )
        chain.add_step("check_confidence", check_step)

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
        ))

        return chain

    @staticmethod
    def iterative_improve() -> Chain:
        """
        Iterative improvement: Loop execute/refine until high confidence.

        Flow:
            [execute]
                ↓
            [verify]
                ↓
            [confidence >= 0.85?]
                ├─ Yes → [output]
                └─ No → [refine] → [back to verify]

        Use when:
        - Quality is critical
        - Can accept higher latency (up to 5-10s)
        - Want guaranteed high confidence
        """
        chain = Chain(name="iterative_improve", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 10,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_quality",
        ))

        # Conditional: Check quality
        chain.add_step("check_quality", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.85),
            on_success=None,  # Success: output
            on_failure="refine",  # Failure: refine
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify",  # Loop back to verify
            max_iterations=3,  # Safety limit
        ))

        return chain

    @staticmethod
    def multi_strategy() -> Chain:
        """
        Multi-strategy: Try multiple retrieval strategies in sequence.

        Flow:
            [execute (direct)]
                ↓
            [verify]
                ↓
            [confidence >= 0.75?]
                ├─ Yes → [output]
                └─ No → [execute (research)] → [verify] → [output]

        Use when:
        - First approach might fail
        - Want fallback strategies
        - Can't accept complete failure
        """
        chain = Chain(name="multi_strategy", entry_point="execute_direct")

        # Strategy 1: Direct reasoning
        chain.add_step("execute_direct", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 5,
            },
            next_step="verify_direct",
        ))

        chain.add_step("verify_direct", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_direct",
        ))

        # Decision point
        chain.add_step("check_direct", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success=None,  # Good enough
            on_failure="execute_research",  # Try research
        ))

        # Strategy 2: Research mode
        chain.add_step("execute_research", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 15,
            },
            next_step="verify_research",
        ))

        chain.add_step("verify_research", ChainStep(
            step_type=StepType.VERIFY,
            params={},
        ))

        return chain

    @staticmethod
    def research_pipeline() -> Chain:
        """
        Research pipeline: Full cycle with learning.

        Flow:
            [execute (research)]
                ↓
            [verify]
                ↓
            [confidence >= 0.75?]
                ├─ Yes → [learn] → [output]
                └─ No → [refine] → [verify] → [learn] → [output]

        Use when:
        - Deep research is needed
        - System should learn from queries
        - Quality is paramount
        """
        chain = Chain(name="research_pipeline", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 20,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_quality",
        ))

        # Quality check
        chain.add_step("check_quality", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success="learn",
            on_failure="refine",
        ))

        # Refinement path
        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify_refined",
        ))

        chain.add_step("verify_refined", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="learn",
        ))

        # Learning: update strategy
        chain.add_step("learn", ChainStep(
            step_type=StepType.UPDATE_STRATEGY,
            params={
                "feedback": {
                    "helpful": True,
                    "refinement_needed": False,
                }
            },
        ))

        return chain

    @staticmethod
    def quality_first() -> Chain:
        """
        Quality-first: Prioritize quality over speed.

        Flow:
            [execute (research)]
                ↓
            [verify (strict)]
                ↓
            [all checks passed?]
                ├─ Yes → [output]
                └─ No → [loop refine 3x] → [output]

        Use when:
        - Accuracy is critical (medical, legal, financial)
        - Can spend 5-20s per query
        - Must pass all verification checks
        """
        chain = Chain(name="quality_first", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "plan_execute",  # Most thorough
                "max_sources": 20,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_all_passed",
        ))

        # Strict check: all must pass
        chain.add_step("check_all_passed", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success=None,
            on_failure="refine_loop",
        ))

        # Refinement loop
        chain.add_step("refine_loop", ChainStep(
            step_type=StepType.LOOP,
            params={},
            next_step="refine",
            max_iterations=3,
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify",  # Loop back
        ))

        return chain

    @staticmethod
    def quick_answer() -> Chain:
        """
        Quick answer: Minimum latency, acceptable quality.

        Flow:
            [execute (direct)]
                ↓
            [output]

        Use when:
        - Speed is critical (real-time chat, typing suggestions)
        - Users accept lower confidence
        - Latency < 100ms is important
        """
        chain = Chain(name="quick_answer", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 3,
            },
            timeout_seconds=0.1,  # 100ms timeout
        ))
        return chain

    @staticmethod
    def balanced() -> Chain:
        """
        Balanced: Trade-off between speed and quality.

        Flow:
            [execute (verify mode)]
                ↓
            [verify]
                ↓
            [confidence >= 0.7?]
                ├─ Yes → [output]
                └─ No → [refine once] → [output]

        Use when:
        - Balance is important
        - Most common production use case
        - Latency 150-300ms acceptable
        """
        chain = Chain(name="balanced", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 7,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_confidence",
        ))

        chain.add_step("check_confidence", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_below(0.7),
            on_success="refine",
            on_failure=None,
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
        ))

        return chain

    # =========================================================================
    # NEW TEMPLATES (December 2025) - Domain-Specific Chain Patterns
    # =========================================================================

    @staticmethod
    def fact_check() -> Chain:
        """
        Fact-checking: Extract claims → Verify sources → Generate verdict.

        Flow:
            [extract_claims]
                ↓
            [verify_sources]
                ↓
            [all claims verified?]
                ├─ Yes → [generate_verdict]
                └─ No → [flag_unverified] → [generate_verdict_with_warnings]

        Use when:
        - Verifying factual claims in content
        - Detecting misinformation
        - Building trust with source citations

        Latency: ~300-500ms (depends on claim count)
        """
        chain = Chain(name="fact_check", entry_point="extract_claims")

        # Step 1: Extract discrete claims from input
        chain.add_step("extract_claims", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "extract_claims",
                "max_claims": 10,
            },
            next_step="verify_sources",
        ))

        # Step 2: Verify each claim against sources
        chain.add_step("verify_sources", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "source_check",
                "min_sources_per_claim": 2,
            },
            next_step="check_verification",
        ))

        # Step 3: Check if all claims are verified
        chain.add_step("check_verification", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success="generate_verdict",
            on_failure="flag_unverified",
        ))

        # Path A: All verified - clean verdict
        chain.add_step("generate_verdict", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "generate_verdict",
                "include_citations": True,
            },
        ))

        # Path B: Some unverified - flag and warn
        chain.add_step("flag_unverified", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "flag_unverified_claims",
            },
            next_step="generate_verdict_with_warnings",
        ))

        chain.add_step("generate_verdict_with_warnings", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "generate_verdict",
                "include_warnings": True,
                "include_citations": True,
            },
        ))

        return chain

    @staticmethod
    def code_review() -> Chain:
        """
        Code review: Security scan → Style check → Suggest fixes.

        Flow:
            [security_scan]
                ↓
            [style_check]
                ↓
            [any issues?]
                ├─ Yes → [suggest_fixes] → [output]
                └─ No → [output (clean)]

        Use when:
        - Automated code review in CI/CD
        - Security vulnerability detection
        - Style consistency enforcement

        Latency: ~200-400ms per file
        """
        chain = Chain(name="code_review", entry_point="security_scan")

        # Step 1: Security vulnerability scan
        chain.add_step("security_scan", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "security_scan",
                "check_categories": [
                    "sql_injection", "xss", "command_injection",
                    "path_traversal", "hardcoded_secrets"
                ],
            },
            next_step="style_check",
        ))

        # Step 2: Style and convention check
        chain.add_step("style_check", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "style_check",
                "check_naming": True,
                "check_complexity": True,
                "check_documentation": True,
            },
            next_step="check_issues",
        ))

        # Step 3: Check if any issues found
        chain.add_step("check_issues", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.combine_not(Conditions.all_checks_passed()),
            on_success="suggest_fixes",  # Issues found
            on_failure=None,  # No issues - clean output
        ))

        # Step 4: Generate fix suggestions
        chain.add_step("suggest_fixes", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "task": "suggest_code_fixes",
                "include_examples": True,
                "prioritize_security": True,
            },
        ))

        return chain

    @staticmethod
    def summarize() -> Chain:
        """
        Document summarization: Extract key points → Draft → Verify accuracy.

        Flow:
            [extract_key_points]
                ↓
            [draft_summary]
                ↓
            [verify_accuracy]
                ↓
            [accuracy >= 0.8?]
                ├─ Yes → [output]
                └─ No → [refine_summary] → [output]

        Use when:
        - Summarizing long documents
        - Creating executive summaries
        - Content compression with accuracy checks

        Latency: ~400-800ms (depends on document length)
        """
        chain = Chain(name="summarize", entry_point="extract_key_points")

        # Step 1: Extract key points from document
        chain.add_step("extract_key_points", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "task": "extract_key_points",
                "max_points": 10,
                "preserve_structure": True,
            },
            next_step="draft_summary",
        ))

        # Step 2: Draft the summary
        chain.add_step("draft_summary", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "draft_summary",
                "target_length": "concise",
            },
            next_step="verify_accuracy",
        ))

        # Step 3: Verify summary accuracy against source
        chain.add_step("verify_accuracy", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "factual_accuracy",
                "compare_to_source": True,
            },
            next_step="check_accuracy",
        ))

        # Step 4: Check accuracy threshold
        chain.add_step("check_accuracy", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.8),
            on_success=None,  # Good accuracy
            on_failure="refine_summary",  # Needs refinement
        ))

        # Step 5: Refine if accuracy low
        chain.add_step("refine_summary", ChainStep(
            step_type=StepType.REFINE,
            params={
                "focus": "accuracy",
                "preserve_key_points": True,
            },
        ))

        return chain

    @staticmethod
    def safety_gated() -> Chain:
        """
        Safety-gated execution: Safety check → Execute → Audit.

        Flow:
            [safety_check]
                ↓
            [safe to proceed?]
                ├─ Yes → [execute] → [audit] → [output]
                └─ No → [refuse_with_reason]

        Use when:
        - High-risk operations (code execution, system commands)
        - Compliance-critical workflows
        - Human-in-the-loop systems

        Latency: ~150-300ms (+ execution time)
        """
        chain = Chain(name="safety_gated", entry_point="safety_check")

        # Step 1: Pre-execution safety check
        chain.add_step("safety_check", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "safety_assessment",
                "check_categories": [
                    "harmful_content", "pii_exposure",
                    "system_access", "data_modification"
                ],
            },
            next_step="check_safe",
        ))

        # Step 2: Gate based on safety score
        chain.add_step("check_safe", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.9),  # High safety threshold
            on_success="execute",
            on_failure="refuse_with_reason",
        ))

        # Path A: Safe - proceed with execution
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "sandboxed": True,
            },
            next_step="audit",
        ))

        # Step 4: Audit the execution
        chain.add_step("audit", ChainStep(
            step_type=StepType.UPDATE_STRATEGY,
            params={
                "audit": True,
                "log_execution": True,
                "track_outcomes": True,
            },
        ))

        # Path B: Unsafe - refuse with explanation
        chain.add_step("refuse_with_reason", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "generate_refusal",
                "include_reason": True,
                "suggest_alternatives": True,
            },
        ))

        return chain

    @staticmethod
    def memory_augmented() -> Chain:
        """
        Memory-augmented: Recall → Execute → Store learnings.

        Flow:
            [recall_context]
                ↓
            [execute_with_memory]
                ↓
            [high quality?]
                ├─ Yes → [store_learning] → [output]
                └─ No → [output (don't store)]

        Use when:
        - Building adaptive systems that learn
        - Long-term knowledge accumulation
        - Personalized assistants

        Latency: ~200-400ms (+ memory operations)
        """
        chain = Chain(name="memory_augmented", entry_point="recall_context")

        # Step 1: Recall relevant context from memory
        chain.add_step("recall_context", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "task": "recall",
                "max_memories": 10,
                "recency_weight": 0.3,
                "relevance_weight": 0.7,
            },
            next_step="execute_with_memory",
        ))

        # Step 2: Execute with retrieved context
        chain.add_step("execute_with_memory", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "use_context": True,
                "max_sources": 10,
            },
            next_step="check_quality",
        ))

        # Step 3: Check quality for learning
        chain.add_step("check_quality", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success="store_learning",
            on_failure=None,  # Don't store low-quality results
        ))

        # Step 4: Store successful interaction for future recall
        chain.add_step("store_learning", ChainStep(
            step_type=StepType.UPDATE_STRATEGY,
            params={
                "store_memory": True,
                "memory_type": "episodic",
                "extract_patterns": True,
            },
        ))

        return chain

    @staticmethod
    def hallucination_guard() -> Chain:
        """
        Hallucination prevention: Generate → Verify sources → Filter unsupported.

        Flow:
            [generate_response]
                ↓
            [extract_claims]
                ↓
            [verify_against_sources]
                ↓
            [all claims sourced?]
                ├─ Yes → [output]
                └─ No → [filter_unsupported] → [output_with_caveats]

        Use when:
        - High-stakes factual accuracy required
        - Building trustworthy AI systems
        - Reducing confident but wrong outputs

        Latency: ~400-600ms (verification overhead)
        """
        chain = Chain(name="hallucination_guard", entry_point="generate_response")

        # Step 1: Generate initial response
        chain.add_step("generate_response", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 15,
            },
            next_step="extract_claims",
        ))

        # Step 2: Extract factual claims from response
        chain.add_step("extract_claims", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "extract_factual_claims",
            },
            next_step="verify_against_sources",
        ))

        # Step 3: Verify each claim against retrieved sources
        chain.add_step("verify_against_sources", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "source_attribution",
                "require_exact_match": False,
                "semantic_similarity_threshold": 0.8,
            },
            next_step="check_all_sourced",
        ))

        # Step 4: Check if all claims are properly sourced
        chain.add_step("check_all_sourced", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success=None,  # All good
            on_failure="filter_unsupported",
        ))

        # Step 5: Filter out unsupported claims
        chain.add_step("filter_unsupported", ChainStep(
            step_type=StepType.REFINE,
            params={
                "task": "remove_unsourced_claims",
                "add_uncertainty_markers": True,
            },
            next_step="output_with_caveats",
        ))

        chain.add_step("output_with_caveats", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "format_with_caveats",
                "include_confidence_indicators": True,
            },
        ))

        return chain

    @staticmethod
    def rag_optimized() -> Chain:
        """
        Optimized RAG: Retrieve → Rerank → Generate → Cite.

        Flow:
            [retrieve]
                ↓
            [rerank]
                ↓
            [generate_with_citations]
                ↓
            [verify_citations]
                ↓
            [citations valid?]
                ├─ Yes → [output]
                └─ No → [fix_citations] → [output]

        Use when:
        - Production RAG systems
        - High-quality retrieval-augmented generation
        - Citation accuracy is critical

        Latency: ~300-500ms
        """
        chain = Chain(name="rag_optimized", entry_point="retrieve")

        # Step 1: Initial retrieval (BM25 + semantic)
        chain.add_step("retrieve", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "task": "hybrid_retrieve",
                "bm25_weight": 0.3,
                "semantic_weight": 0.7,
                "max_candidates": 50,
            },
            next_step="rerank",
        ))

        # Step 2: Rerank for relevance
        chain.add_step("rerank", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "cross_encoder_rerank",
                "top_k": 10,
            },
            next_step="generate_with_citations",
        ))

        # Step 3: Generate response with inline citations
        chain.add_step("generate_with_citations", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "task": "generate",
                "citation_style": "inline",
                "require_citations": True,
            },
            next_step="verify_citations",
        ))

        # Step 4: Verify citation accuracy
        chain.add_step("verify_citations", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "citation_check",
                "check_quote_accuracy": True,
                "check_source_relevance": True,
            },
            next_step="check_citations_valid",
        ))

        # Step 5: Check citation validity
        chain.add_step("check_citations_valid", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success=None,  # Good citations
            on_failure="fix_citations",
        ))

        # Step 6: Fix invalid citations
        chain.add_step("fix_citations", ChainStep(
            step_type=StepType.REFINE,
            params={
                "task": "fix_citations",
                "remove_invalid": True,
                "add_missing_sources": True,
            },
        ))

        return chain

    @staticmethod
    def agent_planning() -> Chain:
        """
        Agent planning: Decompose → Plan → Execute → Reflect.

        Flow:
            [decompose_goal]
                ↓
            [generate_plan]
                ↓
            [execute_steps] (loop)
                ↓
            [reflect_on_outcome]
                ↓
            [goal achieved?]
                ├─ Yes → [output success]
                └─ No → [replan] → [back to execute]

        Use when:
        - Complex multi-step tasks
        - Goal-oriented agents
        - Tasks requiring planning and adaptation

        Latency: 1-10s (depends on task complexity)
        """
        chain = Chain(name="agent_planning", entry_point="decompose_goal")

        # Step 1: Decompose high-level goal into subgoals
        chain.add_step("decompose_goal", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "plan_execute",
                "task": "goal_decomposition",
                "max_subgoals": 5,
            },
            next_step="generate_plan",
        ))

        # Step 2: Generate execution plan
        chain.add_step("generate_plan", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "task": "plan_generation",
                "include_dependencies": True,
                "estimate_complexity": True,
            },
            next_step="execute_steps",
        ))

        # Step 3: Execute plan steps (loop)
        chain.add_step("execute_steps", ChainStep(
            step_type=StepType.LOOP,
            params={
                "task": "execute_plan_step",
                "track_progress": True,
            },
            next_step="execute_single_step",
            max_iterations=10,
        ))

        chain.add_step("execute_single_step", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "task": "execute_step",
                "validate_output": True,
            },
            next_step="reflect_on_outcome",
        ))

        # Step 4: Reflect on execution outcome
        chain.add_step("reflect_on_outcome", ChainStep(
            step_type=StepType.VERIFY,
            params={
                "verification_type": "goal_progress",
                "check_subgoal_completion": True,
            },
            next_step="check_goal_achieved",
        ))

        # Step 5: Check if goal achieved
        chain.add_step("check_goal_achieved", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success="output_success",
            on_failure="replan",
        ))

        # Path A: Goal achieved
        chain.add_step("output_success", ChainStep(
            step_type=StepType.UPDATE_STRATEGY,
            params={
                "log_success": True,
                "extract_learnings": True,
            },
        ))

        # Path B: Need to replan
        chain.add_step("replan", ChainStep(
            step_type=StepType.REFINE,
            params={
                "task": "replan",
                "incorporate_feedback": True,
            },
            next_step="execute_steps",  # Loop back
            max_iterations=2,  # Limit replanning
        ))

        return chain
