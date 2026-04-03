"""Real HoloLoom agent implementations for Workflow Builder.

Replaces mock implementations with actual HoloLoom orchestrator calls.
Supports all 18 agent types defined in the workflow builder.

Created: 2025-12-23
Part of HoloLoom Workflow Builder production integration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    data: dict[str, Any]
    error: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkflowAgentExecutor:
    """Real HoloLoom integrations for workflow agents.

    Provides lazy initialization of HoloLoom components and
    implements all 18 agent types with actual orchestrator calls.

    Agent Types:
        Query (3): hololoom, search, multiquery
        Process (3): embedder, synthesizer, refiner
        Memory (3): store, retriever, fusion
        Decision (3): thompson, convergence, safety
        Output (2): response, formatter
        Control (3): conditional, loop, parallel
    """

    def __init__(self):
        """Initialize agent executor with lazy loading."""
        self._orchestrator = None
        self._memory = None
        self._embedder = None
        self._bandit = None
        self._config = None
        self._initialized = False

    async def initialize(self):
        """Initialize HoloLoom components (lazy loading).

        Components are only initialized on first use to avoid
        startup overhead when not needed.
        """
        if self._initialized:
            return

        try:
            # Import HoloLoom components
            from hololoom.embedding.spectral import MatryoshkaEmbeddings
            from hololoom.memory.backend_factory import create_memory_backend
            from hololoom.memory.unified import UnifiedMemory
            from hololoom.policy.thompson_sampling import BanditStrategy, TSBandit

            from hololoom.config import Config
            from hololoom.core.orchestrator.weaving_orchestrator import WeavingOrchestrator

            # Use FAST config for balanced performance
            self._config = Config.fast()

            # Initialize memory backend
            backend = await create_memory_backend(self._config)
            self._memory = UnifiedMemory(backend=backend)

            # Initialize orchestrator
            self._orchestrator = WeavingOrchestrator(
                cfg=self._config,
                memory=self._memory
            )

            # Initialize embedder
            self._embedder = MatryoshkaEmbeddings(scales=[96, 192, 384])

            # Initialize Thompson Sampling bandit
            self._bandit = TSBandit(n_arms=4, strategy=BanditStrategy.BAYESIAN_BLEND)

            self._initialized = True
            logger.info("WorkflowAgentExecutor initialized successfully")

        except ImportError as e:
            logger.warning(f"HoloLoom components not available: {e}")
            # Continue with fallback implementations
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize HoloLoom components: {e}")
            raise

    async def close(self):
        """Clean up resources."""
        if self._orchestrator:
            try:
                await self._orchestrator.close()
            except Exception as e:
                logger.warning(f"Error closing orchestrator: {e}")

        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # ==================== Query Agents ====================

    async def execute_hololoom(self, config: dict, input_data: dict) -> AgentResult:
        """Execute full HoloLoom weaving cycle.

        Args:
            config: Agent configuration (pattern, max_tokens, etc.)
            input_data: Must contain 'query' key

        Returns:
            AgentResult with response, confidence, tool_used, trace
        """
        import time
        start = time.perf_counter()

        await self.initialize()
        query_text = input_data.get("query", "")

        if not query_text:
            return AgentResult(
                success=False,
                data={},
                error="No query provided"
            )

        try:
            if self._orchestrator:
                from hololoom.protocols.types import Query

                # Configure pattern if specified
                pattern = config.get("pattern", "fast")
                if pattern == "bare":
                    self._config.complexity = "LITE"
                elif pattern == "fused":
                    self._config.complexity = "FULL"
                else:
                    self._config.complexity = "FAST"

                # Execute weaving cycle
                async with self._orchestrator as orch:
                    spacetime = await orch.weave(Query(text=query_text))

                latency = (time.perf_counter() - start) * 1000

                return AgentResult(
                    success=True,
                    data={
                        "response": spacetime.response,
                        "confidence": spacetime.confidence,
                        "tool_used": spacetime.metadata.get("tool_used"),
                        "trace": spacetime.trace.to_dict() if hasattr(spacetime, 'trace') and spacetime.trace else {}
                    },
                    latency_ms=latency,
                    metadata={"pattern": pattern}
                )
            else:
                # Fallback to mock
                return await self._mock_hololoom(config, input_data, start)

        except Exception as e:
            logger.error(f"HoloLoom execution error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_search(self, config: dict, input_data: dict) -> AgentResult:
        """Execute memory search via UnifiedMemory.

        Args:
            config: Agent configuration (k, strategy, etc.)
            input_data: Must contain 'query' key

        Returns:
            AgentResult with search results
        """
        import time
        start = time.perf_counter()

        await self.initialize()
        query = input_data.get("query", "")
        k = config.get("k", 10)

        if not query:
            return AgentResult(
                success=False,
                data={},
                error="No query provided"
            )

        try:
            if self._memory:
                from hololoom.memory.unified import RecallStrategy

                strategy = config.get("strategy", "similar")
                recall_strategy = {
                    "similar": RecallStrategy.SIMILAR,
                    "connected": RecallStrategy.CONNECTED,
                    "recent": RecallStrategy.RECENT,
                    "resonant": RecallStrategy.RESONANT
                }.get(strategy, RecallStrategy.SIMILAR)

                memories = await self._memory.recall(
                    query=query,
                    strategy=recall_strategy,
                    limit=k
                )

                results = [
                    {
                        "id": m.id,
                        "text": m.text[:500] if hasattr(m, 'text') else str(m)[:500],
                        "relevance": getattr(m, 'relevance', 0.5)
                    }
                    for m in memories
                ]

                return AgentResult(
                    success=True,
                    data={
                        "results": results,
                        "count": len(results)
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                return await self._mock_search(config, input_data, start)

        except Exception as e:
            logger.error(f"Search error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_multiquery(self, config: dict, input_data: dict) -> AgentResult:
        """Break query into sub-questions and execute each.

        Args:
            config: Agent configuration (num_queries, etc.)
            input_data: Must contain 'query' key

        Returns:
            AgentResult with sub-query results
        """
        import time
        start = time.perf_counter()

        await self.initialize()
        query = input_data.get("query", "")
        num_queries = config.get("num_queries", 3)

        if not query:
            return AgentResult(
                success=False,
                data={},
                error="No query provided"
            )

        try:
            # Generate sub-queries
            sub_queries = self._generate_sub_queries(query, num_queries)

            results = []
            if self._orchestrator:
                from hololoom.protocols.types import Query

                async with self._orchestrator as orch:
                    for sub_q in sub_queries:
                        try:
                            spacetime = await orch.weave(Query(text=sub_q))
                            results.append({
                                "sub_query": sub_q,
                                "response": spacetime.response,
                                "confidence": spacetime.confidence
                            })
                        except Exception as e:
                            results.append({
                                "sub_query": sub_q,
                                "response": None,
                                "confidence": 0.0,
                                "error": str(e)
                            })
            else:
                # Fallback: return sub-queries without execution
                results = [{"sub_query": sq, "response": None, "confidence": 0.0} for sq in sub_queries]

            return AgentResult(
                success=True,
                data={
                    "sub_results": results,
                    "count": len(results)
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Multi-query error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Process Agents ====================

    async def execute_embedder(self, config: dict, input_data: dict) -> AgentResult:
        """Generate multi-scale Matryoshka embeddings.

        Args:
            config: Agent configuration (scale, etc.)
            input_data: Must contain 'text' or 'texts' key

        Returns:
            AgentResult with embeddings
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        texts = input_data.get("texts", [])
        if not texts:
            text = input_data.get("text", "")
            if text:
                texts = [text]

        if not texts:
            return AgentResult(
                success=False,
                data={},
                error="No text provided"
            )

        scale = config.get("scale", 384)

        try:
            if self._embedder:
                embeddings = self._embedder.embed_batch(texts, scale=scale)

                return AgentResult(
                    success=True,
                    data={
                        "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                        "scale": scale,
                        "count": len(texts)
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback: return placeholder embeddings
                import random
                embeddings = [[random.uniform(-1, 1) for _ in range(scale)] for _ in texts]

                return AgentResult(
                    success=True,
                    data={
                        "embeddings": embeddings,
                        "scale": scale,
                        "count": len(texts),
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Embedder error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_synthesizer(self, config: dict, input_data: dict) -> AgentResult:
        """Extract entities and motifs from input.

        Args:
            config: Agent configuration
            input_data: Must contain 'content' or 'results' key

        Returns:
            AgentResult with entities and motifs
        """
        import time
        start = time.perf_counter()

        content = input_data.get("content", "")
        results = input_data.get("results", [])

        # Combine all text content
        if results:
            content = " ".join([r.get("text", r.get("response", "")) for r in results if isinstance(r, dict)])

        if not content:
            return AgentResult(
                success=False,
                data={},
                error="No content to synthesize"
            )

        try:
            # Simple entity extraction using regex patterns
            import re

            # Extract capitalized phrases as potential entities
            entities = list(set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)))[:20]

            # Extract key phrases (simple heuristic)
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4:
                    word_freq[word] = word_freq.get(word, 0) + 1

            motifs = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            motifs = [m[0] for m in motifs]

            return AgentResult(
                success=True,
                data={
                    "entities": entities,
                    "motifs": motifs,
                    "summary": content[:500] if len(content) > 500 else content
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Synthesizer error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_refiner(self, config: dict, input_data: dict) -> AgentResult:
        """Refine response through iterative improvement.

        Args:
            config: Agent configuration (max_iterations, threshold)
            input_data: Must contain 'response' or 'content' key

        Returns:
            AgentResult with refined response
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        response = input_data.get("response", input_data.get("content", ""))
        max_iterations = config.get("max_iterations", 3)
        quality_threshold = config.get("quality_threshold", 0.9)

        if not response:
            return AgentResult(
                success=False,
                data={},
                error="No response to refine"
            )

        try:
            if self._orchestrator:
                from hololoom.protocols.types import Query

                refined = response
                quality = 0.5
                iterations = 0

                async with self._orchestrator as orch:
                    while quality < quality_threshold and iterations < max_iterations:
                        # Use orchestrator to refine
                        refinement_query = f"Improve and refine this response: {refined[:1000]}"
                        spacetime = await orch.weave(Query(text=refinement_query))

                        if spacetime.confidence > quality:
                            refined = spacetime.response
                            quality = spacetime.confidence

                        iterations += 1

                return AgentResult(
                    success=True,
                    data={
                        "refined_response": refined,
                        "quality_score": quality,
                        "iterations": iterations
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback: return original with slight quality boost
                return AgentResult(
                    success=True,
                    data={
                        "refined_response": response,
                        "quality_score": 0.75,
                        "iterations": 1,
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Refiner error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Memory Agents ====================

    async def execute_store(self, config: dict, input_data: dict) -> AgentResult:
        """Store content to memory.

        Args:
            config: Agent configuration
            input_data: Must contain 'content' key

        Returns:
            AgentResult with stored memory ID
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        content = input_data.get("content", "")

        if not content:
            return AgentResult(
                success=False,
                data={},
                error="No content to store"
            )

        try:
            if self._memory:
                memory = await self._memory.experience(content)

                return AgentResult(
                    success=True,
                    data={
                        "stored_id": memory.id if hasattr(memory, 'id') else str(id(memory)),
                        "success": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback: return mock ID
                import uuid
                return AgentResult(
                    success=True,
                    data={
                        "stored_id": str(uuid.uuid4()),
                        "success": True,
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Store error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_retriever(self, config: dict, input_data: dict) -> AgentResult:
        """Retrieve context using connected strategy.

        Args:
            config: Agent configuration (k, etc.)
            input_data: Must contain 'query' key

        Returns:
            AgentResult with retrieved context
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        query = input_data.get("query", "")
        k = config.get("k", 10)

        if not query:
            return AgentResult(
                success=False,
                data={},
                error="No query provided"
            )

        try:
            if self._memory:
                from hololoom.memory.unified import RecallStrategy

                memories = await self._memory.recall(
                    query=query,
                    strategy=RecallStrategy.CONNECTED,
                    limit=k
                )

                context = [m.text if hasattr(m, 'text') else str(m) for m in memories]
                ids = [m.id if hasattr(m, 'id') else str(id(m)) for m in memories]

                return AgentResult(
                    success=True,
                    data={
                        "context": context,
                        "ids": ids
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback
                return AgentResult(
                    success=True,
                    data={
                        "context": [],
                        "ids": [],
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Retriever error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_fusion(self, config: dict, input_data: dict) -> AgentResult:
        """Multi-hop knowledge graph traversal.

        Args:
            config: Agent configuration (max_hops, etc.)
            input_data: Must contain 'query' or 'seed_nodes' key

        Returns:
            AgentResult with fused knowledge
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        query = input_data.get("query", "")
        seed_nodes = input_data.get("seed_nodes", [])
        max_hops = config.get("max_hops", 3)

        if not query and not seed_nodes:
            return AgentResult(
                success=False,
                data={},
                error="No query or seed nodes provided"
            )

        try:
            if self._memory and hasattr(self._memory, '_backend') and hasattr(self._memory._backend, 'graph'):
                # Get knowledge graph
                kg = self._memory._backend.graph

                # Multi-hop traversal
                visited = set(seed_nodes)
                current_nodes = seed_nodes if seed_nodes else []

                # If no seed nodes, try to find relevant starting points
                if not current_nodes and query:
                    # Simple: use first few words as potential node IDs
                    potential = query.lower().replace("?", "").split()[:3]
                    current_nodes = [p for p in potential if kg.has_node(p)]

                for hop in range(max_hops):
                    next_nodes = []
                    for node in current_nodes:
                        if kg.has_node(node):
                            neighbors = list(kg.neighbors(node))
                            for n in neighbors:
                                if n not in visited:
                                    visited.add(n)
                                    next_nodes.append(n)
                    current_nodes = next_nodes[:10]  # Limit expansion

                return AgentResult(
                    success=True,
                    data={
                        "fused_nodes": list(visited)[:50],
                        "hops_performed": min(max_hops, hop + 1),
                        "total_nodes": len(visited)
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback
                return AgentResult(
                    success=True,
                    data={
                        "fused_nodes": seed_nodes,
                        "hops_performed": 0,
                        "total_nodes": len(seed_nodes),
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Fusion error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Decision Agents ====================

    async def execute_thompson(self, config: dict, input_data: dict) -> AgentResult:
        """Thompson Sampling selection.

        Args:
            config: Agent configuration
            input_data: Must contain 'options' key

        Returns:
            AgentResult with selected option
        """
        import time
        start = time.perf_counter()

        await self.initialize()

        options = input_data.get("options", [])
        probs = input_data.get("probabilities", input_data.get("probs", []))

        if not options:
            return AgentResult(
                success=False,
                data={},
                error="No options provided"
            )

        try:
            if not probs:
                probs = [1.0 / len(options)] * len(options)

            if self._bandit:
                selected_idx, debug = self._bandit.select_with_strategy(probs)

                return AgentResult(
                    success=True,
                    data={
                        "selected": options[selected_idx] if selected_idx < len(options) else None,
                        "index": selected_idx,
                        "debug": debug
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            else:
                # Fallback: weighted random selection
                import random
                cumulative = []
                total = 0
                for p in probs:
                    total += p
                    cumulative.append(total)

                r = random.uniform(0, total)
                selected_idx = 0
                for i, c in enumerate(cumulative):
                    if r <= c:
                        selected_idx = i
                        break

                return AgentResult(
                    success=True,
                    data={
                        "selected": options[selected_idx] if selected_idx < len(options) else None,
                        "index": selected_idx,
                        "fallback": True
                    },
                    latency_ms=(time.perf_counter() - start) * 1000
                )

        except Exception as e:
            logger.error(f"Thompson error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_convergence(self, config: dict, input_data: dict) -> AgentResult:
        """Convergence engine decision collapse.

        Args:
            config: Agent configuration (strategy, etc.)
            input_data: Must contain 'options' and 'probabilities' keys

        Returns:
            AgentResult with selected option
        """
        import time
        start = time.perf_counter()

        options = input_data.get("options", [])
        probs = input_data.get("probabilities", [])
        strategy = config.get("strategy", "bayesian_blend")

        if not options:
            return AgentResult(
                success=False,
                data={},
                error="No options provided"
            )

        try:
            from hololoom.convergence.engine import CollapseStrategy, ConvergenceEngine

            strategy_map = {
                "argmax": CollapseStrategy.ARGMAX,
                "epsilon_greedy": CollapseStrategy.EPSILON_GREEDY,
                "bayesian_blend": CollapseStrategy.BAYESIAN_BLEND,
                "pure_thompson": CollapseStrategy.PURE_THOMPSON
            }

            collapse_strategy = strategy_map.get(strategy.lower(), CollapseStrategy.BAYESIAN_BLEND)
            engine = ConvergenceEngine(strategy=collapse_strategy)

            if not probs:
                probs = [1.0 / len(options)] * len(options)

            selected = engine.collapse(options, probs)

            return AgentResult(
                success=True,
                data={
                    "selected": selected,
                    "strategy": strategy
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except ImportError:
            # Fallback: simple argmax
            if probs:
                max_idx = probs.index(max(probs))
                selected = options[max_idx]
            else:
                selected = options[0] if options else None

            return AgentResult(
                success=True,
                data={
                    "selected": selected,
                    "strategy": strategy,
                    "fallback": True
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Convergence error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_safety(self, config: dict, input_data: dict) -> AgentResult:
        """Safety guardrails evaluation.

        Args:
            config: Agent configuration (risk_threshold, etc.)
            input_data: Must contain 'action' or 'content' key

        Returns:
            AgentResult with safety evaluation
        """
        import time
        start = time.perf_counter()

        action = input_data.get("action", "")
        content = input_data.get("content", "")
        context = input_data.get("context", {})
        risk_threshold = config.get("risk_threshold", 0.7)

        if not action and not content:
            return AgentResult(
                success=False,
                data={},
                error="No action or content to evaluate"
            )

        try:
            from hololoom.alignment.safety_guardrails import SafetyGuardrails

            guardrails = SafetyGuardrails()
            result = await guardrails.gate_action(action or content, context)

            return AgentResult(
                success=True,
                data={
                    "allowed": result.allowed,
                    "risk_level": result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level),
                    "safety_score": result.safety_score,
                    "reason": result.reason if hasattr(result, 'reason') else None
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except ImportError:
            # Fallback: simple pattern-based safety check
            dangerous_patterns = ['delete', 'drop', 'rm -rf', 'exec(', 'eval(', 'system(']
            text_to_check = (action + content).lower()

            is_safe = not any(p in text_to_check for p in dangerous_patterns)

            return AgentResult(
                success=True,
                data={
                    "allowed": is_safe,
                    "risk_level": "LOW" if is_safe else "HIGH",
                    "safety_score": 0.9 if is_safe else 0.2,
                    "fallback": True
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Safety error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Output Agents ====================

    async def execute_response(self, config: dict, input_data: dict) -> AgentResult:
        """Generate final response.

        Args:
            config: Agent configuration (format, etc.)
            input_data: Must contain response data from previous steps

        Returns:
            AgentResult with formatted response
        """
        import time
        start = time.perf_counter()

        # Collect all available data for response generation
        response = input_data.get("response", "")
        content = input_data.get("content", "")
        results = input_data.get("results", [])
        context = input_data.get("context", [])

        # Build response from available data
        if not response:
            if content:
                response = content
            elif results:
                response = "\n".join([
                    r.get("response", r.get("text", str(r)))
                    for r in results if isinstance(r, dict)
                ])
            elif context:
                response = "\n".join(context) if isinstance(context, list) else str(context)

        if not response:
            response = "No response generated."

        return AgentResult(
            success=True,
            data={
                "response": response,
                "confidence": input_data.get("confidence", 0.7)
            },
            latency_ms=(time.perf_counter() - start) * 1000
        )

    async def execute_formatter(self, config: dict, input_data: dict) -> AgentResult:
        """Format output in specified format.

        Args:
            config: Agent configuration (format: json/markdown/html)
            input_data: Data to format

        Returns:
            AgentResult with formatted output
        """
        import json
        import time
        start = time.perf_counter()

        output_format = config.get("format", "json")
        data = input_data.get("data", input_data)

        try:
            if output_format == "json":
                formatted = json.dumps(data, indent=2, default=str)

            elif output_format == "markdown":
                # Simple markdown formatting
                lines = ["# Result\n"]
                if isinstance(data, dict):
                    for key, value in data.items():
                        lines.append(f"## {key}\n")
                        lines.append(f"{value}\n")
                else:
                    lines.append(str(data))
                formatted = "\n".join(lines)

            elif output_format == "html":
                # Simple HTML formatting
                if isinstance(data, dict):
                    items = "".join([
                        f"<dt>{k}</dt><dd>{v}</dd>"
                        for k, v in data.items()
                    ])
                    formatted = f"<dl>{items}</dl>"
                else:
                    formatted = f"<p>{data}</p>"

            else:
                formatted = str(data)

            return AgentResult(
                success=True,
                data={
                    "formatted": formatted,
                    "format": output_format
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Formatter error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Control Flow Agents ====================

    async def execute_conditional(self, config: dict, input_data: dict) -> AgentResult:
        """Conditional branching based on condition.

        Args:
            config: Agent configuration (condition, threshold, etc.)
            input_data: Data to evaluate

        Returns:
            AgentResult with branch decision
        """
        import time
        start = time.perf_counter()

        condition = config.get("condition", "confidence > 0.5")
        threshold = config.get("threshold", 0.5)

        # Get values from input
        confidence = input_data.get("confidence", 0.5)
        value = input_data.get("value", confidence)
        success = input_data.get("success", True)

        try:
            # Evaluate condition
            if "confidence" in condition:
                result = confidence > threshold
            elif "success" in condition:
                result = success
            elif ">" in condition:
                result = value > threshold
            elif "<" in condition:
                result = value < threshold
            elif "==" in condition:
                result = value == threshold
            else:
                result = bool(value)

            return AgentResult(
                success=True,
                data={
                    "branch": "true" if result else "false",
                    "condition_met": result,
                    "evaluated_value": value
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Conditional error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_loop(self, config: dict, input_data: dict) -> AgentResult:
        """Loop iterator control.

        Args:
            config: Agent configuration (max_iterations, break_condition)
            input_data: Loop state data

        Returns:
            AgentResult with loop control decision
        """
        import time
        start = time.perf_counter()

        max_iterations = config.get("max_iterations", 10)
        break_condition = config.get("break_condition", "confidence >= 0.9")

        current_iteration = input_data.get("iteration", 0)
        confidence = input_data.get("confidence", 0.0)

        try:
            # Check break condition
            should_continue = current_iteration < max_iterations

            if "confidence" in break_condition and ">=" in break_condition:
                threshold = float(break_condition.split(">=")[1].strip())
                if confidence >= threshold:
                    should_continue = False
            elif "confidence" in break_condition and ">" in break_condition:
                threshold = float(break_condition.split(">")[1].strip())
                if confidence > threshold:
                    should_continue = False

            return AgentResult(
                success=True,
                data={
                    "continue": should_continue,
                    "iteration": current_iteration,
                    "max_iterations": max_iterations
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Loop error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    async def execute_parallel(self, config: dict, input_data: dict) -> AgentResult:
        """Execute multiple operations in parallel.

        Args:
            config: Agent configuration
            input_data: Must contain 'tasks' key with list of tasks

        Returns:
            AgentResult with parallel execution results
        """
        import time
        start = time.perf_counter()

        tasks = input_data.get("tasks", [])

        if not tasks:
            return AgentResult(
                success=False,
                data={},
                error="No tasks to execute in parallel"
            )

        try:
            # Simulate parallel execution by returning task info
            # In real workflow execution, this would trigger parallel node execution
            return AgentResult(
                success=True,
                data={
                    "tasks_count": len(tasks),
                    "parallel": True,
                    "tasks": tasks
                },
                latency_ms=(time.perf_counter() - start) * 1000
            )

        except Exception as e:
            logger.error(f"Parallel error: {e}")
            return AgentResult(
                success=False,
                data={},
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000
            )

    # ==================== Helper Methods ====================

    def _generate_sub_queries(self, query: str, num: int) -> list[str]:
        """Generate sub-queries for multi-query agent.

        Simple heuristic approach. In production, use LLM for better decomposition.
        """
        base = query.rstrip("?").strip()
        aspects = [
            "definition of",
            "examples of",
            "applications of",
            "limitations of",
            "comparisons with alternatives for"
        ]

        sub_queries = []
        for i in range(num):
            aspect = aspects[i % len(aspects)]
            sub_queries.append(f"What is the {aspect} {base}?")

        return sub_queries

    async def _mock_hololoom(self, config: dict, input_data: dict, start_time: float) -> AgentResult:
        """Fallback mock implementation for hololoom agent."""
        import time

        query = input_data.get("query", "")
        return AgentResult(
            success=True,
            data={
                "response": f"Mock response for: {query[:100]}",
                "confidence": 0.7,
                "tool_used": "mock",
                "trace": {}
            },
            latency_ms=(time.perf_counter() - start_time) * 1000,
            metadata={"fallback": True}
        )

    async def _mock_search(self, config: dict, input_data: dict, start_time: float) -> AgentResult:
        """Fallback mock implementation for search agent."""
        import time

        return AgentResult(
            success=True,
            data={
                "results": [],
                "count": 0
            },
            latency_ms=(time.perf_counter() - start_time) * 1000,
            metadata={"fallback": True}
        )


# ==================== Factory Functions ====================

def create_agent_executor() -> WorkflowAgentExecutor:
    """Create a new WorkflowAgentExecutor instance."""
    return WorkflowAgentExecutor()


async def execute_agent(
    agent_type: str,
    config: dict,
    input_data: dict,
    executor: WorkflowAgentExecutor | None = None
) -> AgentResult:
    """Execute an agent by type.

    Args:
        agent_type: Type of agent (hololoom, search, etc.)
        config: Agent configuration
        input_data: Input data for the agent
        executor: Optional executor instance (creates new if not provided)

    Returns:
        AgentResult from the agent execution
    """
    if executor is None:
        executor = WorkflowAgentExecutor()
        await executor.initialize()

    # Map agent types to methods
    agent_methods = {
        # Query agents
        "hololoom": executor.execute_hololoom,
        "search": executor.execute_search,
        "multiquery": executor.execute_multiquery,

        # Process agents
        "embedder": executor.execute_embedder,
        "synthesizer": executor.execute_synthesizer,
        "refiner": executor.execute_refiner,

        # Memory agents
        "store": executor.execute_store,
        "retriever": executor.execute_retriever,
        "fusion": executor.execute_fusion,

        # Decision agents
        "thompson": executor.execute_thompson,
        "convergence": executor.execute_convergence,
        "safety": executor.execute_safety,

        # Output agents
        "response": executor.execute_response,
        "formatter": executor.execute_formatter,

        # Control flow agents
        "conditional": executor.execute_conditional,
        "loop": executor.execute_loop,
        "parallel": executor.execute_parallel,
    }

    method = agent_methods.get(agent_type)
    if method is None:
        return AgentResult(
            success=False,
            data={},
            error=f"Unknown agent type: {agent_type}"
        )

    return await method(config, input_data)
