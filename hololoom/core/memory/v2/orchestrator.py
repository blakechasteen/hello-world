"""
Orchestrator — Matryoshka shell loop with cognitive architecture.

The core control loop where memory drives the model:

  PRIME → (production rules) → VERIFY → (production rules) → FLAG

Classical AI patterns composed:
  - Blackboard architecture: shared workspace, all components post signals
  - SOAR production rules: if-then conditions on blackboard state gate shells
  - Global Workspace Theory: specialists read/write blackboard, attention filters
  - Conditional execution (Self-RAG): skip shells when confidence is high

Each shell:
  1. Navigator constructs context via PPR traversal → posts to blackboard
  2. Confidence estimator scores the result → posts to blackboard
  3. Formatter packs context into a fixed token budget → posts to blackboard
  4. LLM generates a response within that context
  5. Production rules on blackboard state decide: stop, verify, or flag

Key design principle: The reasoning LLM never calls tools or fetches memory.
It receives a pre-constructed context block and just reasons. The memory
model (navigator + formatter) does all the retrieval work.

Shell budgets (default):
  PRIME:   2048 tokens — fast retrieval, best PPR hits
  VERIFY:  1024 tokens — targeted verification of PRIME's weak spots
  FLAG:     512 tokens — minimal context for contradiction resolution
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .navigator import Navigator, NavigatorResult, PPRConfig
from .formatter import Formatter, FormatConfig, ContextBlock
from .confidence import ConfidenceEstimator, ConfidenceConfig, DualConfidence
from .blackboard import (
    Blackboard,
    WorkingMemory,
    ProductionRule,
    default_skip_verify_rules,
    default_skip_flag_rules,
    compute_ppr_entropy,
    extract_seed_sources,
)
from .activation import ActivationAdapter
from .knowledge_source import SourceRegistry, Phase

logger = logging.getLogger(__name__)


# =============================================================================
# Shell Definitions
# =============================================================================

class ShellType(Enum):
    PRIME = "prime"
    VERIFY = "verify"
    FLAG = "flag"


@dataclass
class ShellConfig:
    """Configuration for a single shell in the Matryoshka loop."""
    shell_type: ShellType
    token_budget: int
    ppr_alpha: float = 0.15    # PPR reset probability (higher = more local)
    ppr_max_results: int = 50
    max_seeds: int = 10
    prompt_template: str = ""


DEFAULT_SHELLS = {
    ShellType.PRIME: ShellConfig(
        shell_type=ShellType.PRIME,
        token_budget=2048,
        ppr_alpha=0.15,
        ppr_max_results=50,
        max_seeds=10,
        prompt_template=(
            "Using the following context, answer the question.\n\n"
            "{context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    ),
    ShellType.VERIFY: ShellConfig(
        shell_type=ShellType.VERIFY,
        token_budget=1024,
        ppr_alpha=0.25,  # Higher alpha = more focused on weak spots
        ppr_max_results=20,
        max_seeds=5,
        prompt_template=(
            "Review the following answer for accuracy. The context below "
            "contains additional information that may confirm or contradict "
            "parts of the answer.\n\n"
            "Previous answer:\n{previous_response}\n\n"
            "{context}\n\n"
            "Question: {query}\n\n"
            "Provide a corrected and verified answer:"
        ),
    ),
    ShellType.FLAG: ShellConfig(
        shell_type=ShellType.FLAG,
        token_budget=512,
        ppr_alpha=0.4,  # Very focused
        ppr_max_results=10,
        max_seeds=3,
        prompt_template=(
            "The following answer has low confidence. There may be "
            "contradictions or missing information. Identify what is "
            "uncertain and provide the best possible answer with caveats.\n\n"
            "Previous answer:\n{previous_response}\n\n"
            "{context}\n\n"
            "Question: {query}\n\n"
            "Answer with explicit uncertainty markers:"
        ),
    ),
}


# =============================================================================
# Constrained Generation Schemas
# =============================================================================

# JSON schemas for structured shell outputs. When the LLM supports
# structured generation (Ollama `format`, vLLM grammar, etc.),
# these make outputs machine-parseable for richer bandit signal.

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "enum": ["confirmed", "corrected", "uncertain"],
        },
        "corrections": {
            "type": "array",
            "items": {"type": "string"},
        },
        "answer": {"type": "string"},
        "confidence_self": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["assessment", "answer"],
}

FLAG_SCHEMA = {
    "type": "object",
    "properties": {
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
        },
        "answer": {"type": "string"},
        "caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence_self": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["answer"],
}

SHELL_SCHEMAS = {
    ShellType.VERIFY: VERIFY_SCHEMA,
    ShellType.FLAG: FLAG_SCHEMA,
}


def parse_structured_response(response: str, shell_type: ShellType) -> Optional[Dict]:
    """Try to parse a structured JSON response from VERIFY/FLAG shells.

    Returns parsed dict on success, None on failure (free-text response).
    """
    import json as _json
    try:
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = _json.loads(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed
    except (ValueError, _json.JSONDecodeError):
        pass
    return None


# =============================================================================
# Shell Result
# =============================================================================

@dataclass
class ShellResult:
    """Result from a single shell execution."""
    shell_type: ShellType
    response: str
    context_block: ContextBlock
    confidence: DualConfidence
    nav_result: NavigatorResult
    elapsed_seconds: float
    prompt: str

    @property
    def decision(self) -> str:
        return self.confidence.decision


# =============================================================================
# Loop Result
# =============================================================================

@dataclass
class LoopResult:
    """Result from the full Matryoshka shell loop."""
    response: str                       # Final response text
    shells_executed: List[ShellResult]   # Results from each shell
    final_confidence: DualConfidence     # Confidence of the final response
    total_tokens: int                    # Total context tokens across all shells
    total_elapsed: float                 # Total wall time

    @property
    def shell_count(self) -> int:
        return len(self.shells_executed)

    @property
    def final_shell(self) -> ShellType:
        return self.shells_executed[-1].shell_type if self.shells_executed else ShellType.PRIME

    def summary(self) -> str:
        shells = " → ".join(s.shell_type.value for s in self.shells_executed)
        return (
            f"Loop: {shells} | "
            f"confidence={self.final_confidence.combined:.2f} ({self.final_confidence.decision}) | "
            f"tokens={self.total_tokens} | "
            f"{self.total_elapsed:.1f}s"
        )

    def audit_entry(self) -> Dict[str, Any]:
        """Generate an audit log entry for training pipeline consumption."""
        return {
            "query": self.shells_executed[0].prompt if self.shells_executed else "",
            "response": self.response,
            "shells": [
                {
                    "type": s.shell_type.value,
                    "confidence": s.confidence.combined,
                    "decision": s.confidence.decision,
                    "context_tokens": s.context_block.token_count,
                    "context_items": s.context_block.items_packed,
                    "elapsed": s.elapsed_seconds,
                }
                for s in self.shells_executed
            ],
            "final_confidence": self.final_confidence.combined,
            "final_decision": self.final_confidence.decision,
            "total_tokens": self.total_tokens,
        }


# =============================================================================
# LLM Protocol
# =============================================================================

class LLMProtocol:
    """Protocol for the reasoning LLM.

    The orchestrator doesn't care what LLM is used — it just needs
    an async function that takes a prompt and returns text.
    """
    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        raise NotImplementedError


# =============================================================================
# Matryoshka Orchestrator
# =============================================================================

class MatryoshkaOrchestrator:
    """The Matryoshka shell loop.

    Memory drives the model:
      1. Navigator retrieves context via PPR
      2. Formatter packs it into a budget
      3. LLM reasons over the packed context
      4. Confidence decides: stop, verify, or flag
      5. Repeat with tighter focus if needed

    Usage:
        orch = MatryoshkaOrchestrator(
            navigator=Navigator.from_lite_bus(bus),
            llm_fn=my_llm_generate,
        )
        result = await orch.run("Compare Thompson Sampling vs epsilon-greedy")
        print(result.response)
        print(result.summary())
    """

    def __init__(
        self,
        navigator: Navigator,
        llm_fn: Callable[[str, int], Coroutine[Any, Any, str]],
        confidence_config: Optional[ConfidenceConfig] = None,
        shell_configs: Optional[Dict[ShellType, ShellConfig]] = None,
        audit_fn: Optional[Callable[[Dict], None]] = None,
        working_memory: Optional[WorkingMemory] = None,
        skip_verify_rules: Optional[List[ProductionRule]] = None,
        skip_flag_rules: Optional[List[ProductionRule]] = None,
        activation_adapter: Optional[ActivationAdapter] = None,
        enable_structured_output: bool = False,
        sources: Optional[SourceRegistry] = None,
    ):
        """
        Args:
            navigator: PPR navigator instance
            llm_fn: async fn(prompt, max_tokens) → response text
            confidence_config: Thresholds for confidence decisions
            shell_configs: Override default shell configs
            audit_fn: Optional callback for audit log entries
            working_memory: Persistent cross-turn state (SOAR working memory)
            skip_verify_rules: Production rules for skipping VERIFY (all must fire)
            skip_flag_rules: Production rules for skipping FLAG (any must fire)
            activation_adapter: ACT-R activation for seed weight boosting
            enable_structured_output: Use JSON schemas for VERIFY/FLAG outputs
            sources: KnowledgeSource registry for classical AI composition
        """
        self.navigator = navigator
        self.llm_fn = llm_fn
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.shells = shell_configs or DEFAULT_SHELLS
        self.audit_fn = audit_fn

        # Cognitive architecture: persistent working memory across queries
        self.working_memory = working_memory or WorkingMemory()

        # SOAR production rules for conditional shell execution
        self.skip_verify_rules = skip_verify_rules if skip_verify_rules is not None else default_skip_verify_rules()
        self.skip_flag_rules = skip_flag_rules if skip_flag_rules is not None else default_skip_flag_rules()

        # ACT-R activation adapter (optional)
        self.activation_adapter = activation_adapter

        # Constrained generation for structured VERIFY/FLAG outputs
        self.enable_structured_output = enable_structured_output

        # KnowledgeSource registry for classical AI composition
        self.sources = sources or SourceRegistry()

        self.confidence_estimator = ConfidenceEstimator(
            graph=navigator.graph,
            config=self.confidence_config,
        )

    async def run(
        self,
        query: str,
        max_shells: int = 3,
        force_verify: bool = False,
        blackboard: Optional[Blackboard] = None,
    ) -> LoopResult:
        """Execute the Matryoshka shell loop with blackboard-driven control.

        The blackboard is the shared cognitive workspace. All components post
        signals to it, and production rules on blackboard state decide whether
        to execute each shell (conditional execution from Self-RAG).

        Args:
            query: User's question
            max_shells: Maximum number of shells to execute (1-3)
            force_verify: Always run VERIFY even if PRIME is sufficient
            blackboard: Optional pre-configured blackboard (creates fresh if None)

        Returns:
            LoopResult with final response, all shell results, and blackboard
        """
        t0 = time.perf_counter()
        shell_results: List[ShellResult] = []
        total_tokens = 0

        # Create or reuse blackboard (working memory persists across queries)
        if blackboard is None:
            blackboard = Blackboard(
                query=query,
                working_memory=self.working_memory,
            )
        else:
            blackboard.query = query

        # Update working memory for new turn
        self.working_memory.update_turn()

        # ACT-R: decay activation between turns (forgetting)
        if self.activation_adapter is not None:
            self.activation_adapter.decay_step()

        # KS hook: PRE_SHELL (PRIME)
        self.sources.invoke(Phase.PRE_SHELL, blackboard,
                            self.navigator.graph,
                            {"shell_type": ShellType.PRIME})

        # ── PRIME ───────────────────────────────────────────────────────
        # PRIME always executes — it's the base retrieval
        prime_result = await self._execute_shell(
            ShellType.PRIME, query, previous_response=None,
            blackboard=blackboard,
        )
        shell_results.append(prime_result)
        total_tokens += prime_result.context_block.token_count

        logger.info(
            "PRIME: confidence=%.2f decision=%s (%d tokens, %d seeds, entropy=%.2f)",
            prime_result.confidence.combined,
            prime_result.decision,
            prime_result.context_block.token_count,
            blackboard.seed_count,
            blackboard.ppr_entropy,
        )

        # ── CONDITIONAL: Should we skip VERIFY? ─────────────────────────
        # Production rules (SOAR): ALL rules must fire to skip VERIFY
        if not force_verify and self._should_skip_verify(blackboard):
            logger.info(
                "SKIP VERIFY: all production rules fired (confidence=%s, seeds=%d, ratio=%.2f)",
                blackboard.latest_decision,
                blackboard.seed_count,
                blackboard.entity_match_ratio,
            )
            loop_result = self._build_loop_result(
                shell_results, total_tokens, t0, blackboard
            )
            self.sources.invoke(Phase.POST_LOOP, blackboard,
                                self.navigator.graph,
                                {"loop_result": loop_result})
            return loop_result

        # ── VERIFY ──────────────────────────────────────────────────────
        if max_shells >= 2:
            # KS hook: PRE_SHELL (VERIFY)
            self.sources.invoke(Phase.PRE_SHELL, blackboard,
                                self.navigator.graph,
                                {"shell_type": ShellType.VERIFY})

            verify_result = await self._execute_shell(
                ShellType.VERIFY, query,
                previous_response=prime_result.response,
                blackboard=blackboard,
            )
            shell_results.append(verify_result)
            total_tokens += verify_result.context_block.token_count

            logger.info(
                "VERIFY: confidence=%.2f decision=%s trend=%+.3f (%d tokens)",
                verify_result.confidence.combined,
                verify_result.decision,
                blackboard.confidence_trend,
                verify_result.context_block.token_count,
            )

            # ── CONDITIONAL: Should we skip FLAG? ───────────────────────
            # Production rules: ANY rule firing skips FLAG
            if self._should_skip_flag(blackboard):
                logger.info(
                    "SKIP FLAG: production rule fired (decision=%s, trend=%+.3f)",
                    blackboard.latest_decision,
                    blackboard.confidence_trend,
                )
                loop_result = self._build_loop_result(
                    shell_results, total_tokens, t0, blackboard
                )
                self.sources.invoke(Phase.POST_LOOP, blackboard,
                                    self.navigator.graph,
                                    {"loop_result": loop_result})
                return loop_result

        # ── FLAG ────────────────────────────────────────────────────────
        if max_shells >= 3:
            # KS hook: PRE_SHELL (FLAG)
            self.sources.invoke(Phase.PRE_SHELL, blackboard,
                                self.navigator.graph,
                                {"shell_type": ShellType.FLAG})

            last_response = shell_results[-1].response
            flag_result = await self._execute_shell(
                ShellType.FLAG, query,
                previous_response=last_response,
                blackboard=blackboard,
            )
            shell_results.append(flag_result)
            total_tokens += flag_result.context_block.token_count

            logger.info(
                "FLAG: confidence=%.2f trend=%+.3f (%d tokens)",
                flag_result.confidence.combined,
                blackboard.confidence_trend,
                flag_result.context_block.token_count,
            )

            # Record FLAG corrections in working memory
            if flag_result.decision == "flag":
                self.working_memory.add_correction(
                    f"Q: {query[:80]} → flagged low confidence"
                )

        loop_result = self._build_loop_result(shell_results, total_tokens, t0, blackboard)

        # KS hook: POST_LOOP
        self.sources.invoke(Phase.POST_LOOP, blackboard,
                            self.navigator.graph,
                            {"loop_result": loop_result})

        return loop_result

    # =========================================================================
    # Production Rules: Conditional Shell Execution
    # =========================================================================

    def _should_skip_verify(self, bb: Blackboard) -> bool:
        """Check if ALL skip-verify production rules fire.

        SOAR pattern: conjunctive condition. All rules must be satisfied
        to skip VERIFY, ensuring we only skip when truly confident.
        """
        return all(rule.fires(bb) for rule in self.skip_verify_rules)

    def _should_skip_flag(self, bb: Blackboard) -> bool:
        """Check if ANY skip-flag production rule fires.

        Disjunctive condition. Any single rule is enough to stop at VERIFY,
        since FLAG is an escalation that should only happen when things
        are clearly going wrong.
        """
        return any(rule.fires(bb) for rule in self.skip_flag_rules)

    # =========================================================================
    # Internal: Execute a single shell
    # =========================================================================

    async def _execute_shell(
        self,
        shell_type: ShellType,
        query: str,
        previous_response: Optional[str] = None,
        blackboard: Optional[Blackboard] = None,
    ) -> ShellResult:
        """Execute a single shell: navigate → confidence → format → generate.

        Each step posts signals to the blackboard, building a complete picture
        of the query state that production rules can inspect.
        """
        t0 = time.perf_counter()
        shell_cfg = self.shells[shell_type]

        # 1. Navigate: PPR retrieval with shell-specific config
        ppr_config = PPRConfig(
            alpha=shell_cfg.ppr_alpha,
            max_results=shell_cfg.ppr_max_results,
        )
        nav_result = await self.navigator.navigate(
            query=query,
            max_seeds=shell_cfg.max_seeds,
            override_config=ppr_config,
            activation_adapter=self.activation_adapter,
        )

        # Post navigation signals to blackboard
        if blackboard is not None:
            ppr_scores = [score for _, score in nav_result.ranked_nodes]
            seed_sources = extract_seed_sources(nav_result.seed_nodes)
            entity_ids = {nid for nid, _ in nav_result.ranked_nodes[:20]}
            blackboard.post_navigation(
                seed_count=len(nav_result.seed_nodes),
                seed_sources=seed_sources,
                ppr_entropy=compute_ppr_entropy(ppr_scores),
                ppr_converged=nav_result.converged,
                entity_ids=entity_ids,
            )

        # KS hook: POST_NAVIGATE
        self.sources.invoke(Phase.POST_NAVIGATE, blackboard,
                            self.navigator.graph, {"nav_result": nav_result})

        # 2. Confidence: score the retrieved context
        confidence = self.confidence_estimator.estimate(nav_result, query=query)

        # Post confidence signals to blackboard
        if blackboard is not None:
            blackboard.post_confidence(
                combined=confidence.combined,
                structural=confidence.structural.score,
                narrative=confidence.narrative.score,
                decision=confidence.decision,
            )

        # KS hook: POST_CONFIDENCE
        self.sources.invoke(Phase.POST_CONFIDENCE, blackboard,
                            self.navigator.graph,
                            {"confidence": confidence, "nav_result": nav_result})

        # 3. Format: pack into budget
        format_config = FormatConfig(token_budget=shell_cfg.token_budget)
        formatter = Formatter(config=format_config)
        context_block = formatter.pack(
            ranked_nodes=nav_result.ranked_nodes,
            node_data=nav_result.node_data,
            confidence_scores=confidence.per_node,
        )

        # Post formatter signals to blackboard
        if blackboard is not None:
            blackboard.post_format(
                tokens=context_block.token_count,
                items_packed=context_block.items_packed,
                items_offered=len(nav_result.ranked_nodes),
            )

        # KS hook: POST_FORMAT
        self.sources.invoke(Phase.POST_FORMAT, blackboard,
                            self.navigator.graph,
                            {"context_block": context_block})

        # 4. Build prompt
        prompt = shell_cfg.prompt_template.format(
            context=context_block.text,
            query=query,
            previous_response=previous_response or "",
        )

        # 4.5. Constrained generation: add JSON schema instruction
        if self.enable_structured_output and shell_type in SHELL_SCHEMAS:
            import json as _json
            schema = SHELL_SCHEMAS[shell_type]
            prompt += (
                "\n\nRespond with a JSON object matching this schema:\n"
                f"{_json.dumps(schema, indent=2)}\n"
            )

        # 5. Generate
        max_gen_tokens = max(256, shell_cfg.token_budget)
        response = await self.llm_fn(prompt, max_gen_tokens)

        # 5.5. Parse structured output if available
        structured = None
        if self.enable_structured_output and shell_type in SHELL_SCHEMAS:
            structured = parse_structured_response(response, shell_type)
            if structured:
                # Use the answer field as the response text
                response = structured.get("answer", response)
                # Post self-assessed confidence to blackboard
                if blackboard is not None and "confidence_self" in structured:
                    blackboard.post_flag("self_confidence", structured["confidence_self"])
                if structured.get("corrections"):
                    if blackboard is not None:
                        blackboard.post_flag("corrections", structured["corrections"])

        elapsed = time.perf_counter() - t0

        # Post shell execution to blackboard
        if blackboard is not None:
            blackboard.post_shell(shell_type.value)

        shell_result = ShellResult(
            shell_type=shell_type,
            response=response,
            context_block=context_block,
            confidence=confidence,
            nav_result=nav_result,
            elapsed_seconds=elapsed,
            prompt=prompt,
        )

        # KS hook: POST_GENERATE
        self.sources.invoke(Phase.POST_GENERATE, blackboard,
                            self.navigator.graph,
                            {"response": response, "shell_result": shell_result})

        return shell_result

    # =========================================================================
    # Internal: Build final result
    # =========================================================================

    def _build_loop_result(
        self,
        shell_results: List[ShellResult],
        total_tokens: int,
        start_time: float,
        blackboard: Optional[Blackboard] = None,
    ) -> LoopResult:
        """Build the final loop result and emit audit entry."""
        final = shell_results[-1]
        result = LoopResult(
            response=final.response,
            shells_executed=shell_results,
            final_confidence=final.confidence,
            total_tokens=total_tokens,
            total_elapsed=time.perf_counter() - start_time,
        )

        # Attach blackboard for downstream access (bandits, training)
        result.blackboard = blackboard

        # Emit audit entry for training pipeline
        if self.audit_fn:
            try:
                entry = result.audit_entry()
                # Enrich audit with blackboard context features
                if blackboard is not None:
                    entry["context_features"] = blackboard.context_features()
                    entry["ppr_entropy"] = blackboard.ppr_entropy
                    entry["seed_count"] = blackboard.seed_count
                    entry["entity_match_ratio"] = blackboard.entity_match_ratio
                    entry["wm_overlap"] = blackboard.wm_overlap
                self.audit_fn(entry)
            except Exception as e:
                logger.warning("Audit callback failed: %s", e)

        return result
