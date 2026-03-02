#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reasoning Department - Advanced multi-hop reasoning and causal inference.

This department provides sophisticated reasoning capabilities:
- Multi-hop graph traversal (find connections across multiple steps)
- Causal inference (identify cause-effect relationships)
- Counterfactual analysis ("what if" scenarios)

Design Philosophy:
"Reason beyond the obvious. Connect the dots that others miss."

Supported Tasks:
- "multi_hop_query": Multi-step graph traversal to answer complex queries
- "causal_inference": Identify causal relationships between entities
- "counterfactual": Analyze "what if" scenarios
- "explain_reasoning": Provide explanations for reasoning chains
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import defaultdict

from ..base import BaseDepartment
from ..protocol import (
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
    ConfidenceMetadata,
    DepartmentConfig
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

class CausalDirection(Enum):
    """Direction of causal influence."""
    FORWARD = "forward"  # A → B
    BACKWARD = "backward"  # B ← A
    BIDIRECTIONAL = "bidirectional"  # A ↔ B


@dataclass
class ReasoningChain:
    """A chain of reasoning steps."""
    query: str
    steps: List[Dict[str, Any]]
    final_answer: Optional[str] = None
    confidence: float = 0.0
    total_hops: int = 0

    def add_step(self, entity: str, relation: str, target: str, hop_num: int):
        """Add a reasoning step."""
        self.steps.append({
            "hop": hop_num,
            "entity": entity,
            "relation": relation,
            "target": target
        })
        self.total_hops = hop_num


@dataclass
class CausalRelation:
    """A causal relationship between two entities."""
    cause: str
    effect: str
    direction: CausalDirection
    strength: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CounterfactualScenario:
    """A counterfactual 'what if' scenario."""
    original_condition: str
    modified_condition: str
    predicted_outcome: str
    confidence: float = 0.0
    affected_entities: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)


# ============================================================================
# Multi-Hop Reasoning Engine
# ============================================================================

class MultiHopReasoner:
    """
    Multi-hop graph traversal for complex queries.

    Traverses knowledge graphs to find connections across multiple steps,
    enabling answers to questions like:
    - "What connects A to B through intermediate entities?"
    - "Find all entities related to X within N hops"
    - "What's the shortest path from A to B?"

    Example:
        Query: "What affects both varroa mites and hive temperature?"

        Hop 1: varroa_mites → [attacks] → honey_bees
        Hop 2: honey_bees → [requires] → hive_ventilation
        Hop 3: hive_ventilation → [regulates] → hive_temperature

        Answer: "Hive ventilation connects varroa mites and temperature"
    """

    def __init__(self, max_hops: int = 5, min_confidence: float = 0.6):
        """
        Initialize multi-hop reasoner.

        Args:
            max_hops: Maximum number of hops to traverse
            min_confidence: Minimum confidence threshold for path selection
        """
        self.max_hops = max_hops
        self.min_confidence = min_confidence

        # Path cache for performance
        self._path_cache: Dict[Tuple[str, str], List[ReasoningChain]] = {}

        logger.info(f"MultiHopReasoner initialized (max_hops={max_hops})")

    def find_paths(
        self,
        knowledge_graph: Any,  # KG instance
        start_entity: str,
        end_entity: str,
        max_hops: Optional[int] = None
    ) -> List[ReasoningChain]:
        """
        Find all paths between two entities.

        Args:
            knowledge_graph: Knowledge graph to traverse
            start_entity: Starting entity
            end_entity: Target entity
            max_hops: Maximum hops (defaults to self.max_hops)

        Returns:
            List of reasoning chains connecting start to end
        """
        max_hops = max_hops or self.max_hops

        # Check cache
        cache_key = (start_entity, end_entity)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # BFS traversal
        paths = []
        queue = [(start_entity, [start_entity], 0)]  # (current, path, hop_count)
        visited = set()

        while queue:
            current, path, hops = queue.pop(0)

            # Skip if exceeded max hops
            if hops >= max_hops:
                continue

            # Skip if visited
            if current in visited:
                continue
            visited.add(current)

            # Found target!
            if current == end_entity and hops > 0:
                chain = self._path_to_chain(knowledge_graph, path, start_entity, end_entity)
                paths.append(chain)
                continue

            # Get neighbors
            neighbors = self._get_neighbors(knowledge_graph, current)

            for neighbor, relation in neighbors:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, hops + 1))

        # Sort by confidence (prefer shorter, higher-confidence paths)
        paths.sort(key=lambda c: (c.total_hops, -c.confidence))

        # Cache result
        self._path_cache[cache_key] = paths

        return paths

    def find_common_connections(
        self,
        knowledge_graph: Any,
        entities: List[str],
        max_hops: int = 3
    ) -> List[ReasoningChain]:
        """
        Find entities that connect all given entities.

        Example:
            entities = ["varroa_mites", "hive_temperature"]
            Returns chains showing how "hive_ventilation" connects both

        Args:
            knowledge_graph: Knowledge graph to traverse
            entities: List of entities to connect
            max_hops: Maximum hops from each entity

        Returns:
            List of reasoning chains showing common connections
        """
        if len(entities) < 2:
            return []

        # Find entities reachable from each starting entity
        reachable_sets = []
        for entity in entities:
            reachable = self._find_reachable(knowledge_graph, entity, max_hops)
            reachable_sets.append(set(reachable.keys()))

        # Find intersection (common connections)
        common = reachable_sets[0]
        for reachable_set in reachable_sets[1:]:
            common = common.intersection(reachable_set)

        # Build chains to common entities
        chains = []
        for common_entity in common:
            # Build multi-source chain
            chain = ReasoningChain(
                query=f"Common connections for {entities}",
                steps=[],
                final_answer=common_entity,
                confidence=0.8,
                total_hops=0
            )

            for source_entity in entities:
                paths = self.find_paths(knowledge_graph, source_entity, common_entity, max_hops)
                if paths:
                    shortest = paths[0]
                    chain.steps.extend(shortest.steps)
                    chain.total_hops += shortest.total_hops

            chains.append(chain)

        return chains

    def _get_neighbors(self, knowledge_graph: Any, entity: str) -> List[Tuple[str, str]]:
        """Get neighboring entities and their relations."""
        neighbors = []

        # Get outgoing edges
        if hasattr(knowledge_graph, 'graph'):
            graph = knowledge_graph.graph

            if entity in graph:
                for target in graph.neighbors(entity):
                    # Get edge data
                    edges = graph.get_edge_data(entity, target)
                    if edges:
                        # Handle MultiDiGraph (multiple edges)
                        for edge_key, edge_data in edges.items():
                            relation = edge_data.get('type', 'RELATED_TO')
                            neighbors.append((target, relation))

        return neighbors

    def _find_reachable(
        self,
        knowledge_graph: Any,
        start_entity: str,
        max_hops: int
    ) -> Dict[str, int]:
        """Find all entities reachable within max_hops."""
        reachable = {start_entity: 0}
        queue = [(start_entity, 0)]
        visited = set()

        while queue:
            current, hops = queue.pop(0)

            if hops >= max_hops:
                continue

            if current in visited:
                continue
            visited.add(current)

            neighbors = self._get_neighbors(knowledge_graph, current)

            for neighbor, _ in neighbors:
                if neighbor not in reachable:
                    reachable[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1))

        return reachable

    def _path_to_chain(
        self,
        knowledge_graph: Any,
        path: List[str],
        start: str,
        end: str
    ) -> ReasoningChain:
        """Convert path to reasoning chain."""
        chain = ReasoningChain(
            query=f"{start} → {end}",
            steps=[],
            final_answer=end,
            confidence=0.85,
            total_hops=len(path) - 1
        )

        for i in range(len(path) - 1):
            entity = path[i]
            target = path[i + 1]

            # Get relation
            neighbors = self._get_neighbors(knowledge_graph, entity)
            relation = "RELATED_TO"
            for neighbor, rel in neighbors:
                if neighbor == target:
                    relation = rel
                    break

            chain.add_step(entity, relation, target, i + 1)

        return chain


# ============================================================================
# Causal Inference Engine
# ============================================================================

class CausalInferenceEngine:
    """
    Causal relationship inference from knowledge graphs.

    Identifies cause-effect relationships using:
    - Graph structure (directed edges, temporal ordering)
    - Relation types (CAUSES, LEADS_TO, AFFECTS)
    - Statistical patterns (co-occurrence, correlation)

    Example:
        Input: "varroa_mites" → "colony_collapse"
        Output: CausalRelation(
            cause="varroa_mites",
            effect="colony_collapse",
            direction=FORWARD,
            strength=0.9,
            evidence=["attacks honey_bees", "weakens immunity", ...]
        )
    """

    # Causal relation types
    CAUSAL_RELATIONS = {
        "CAUSES", "LEADS_TO", "RESULTS_IN", "TRIGGERS",
        "AFFECTS", "INFLUENCES", "IMPACTS"
    }

    def __init__(self, min_strength: float = 0.5):
        """
        Initialize causal inference engine.

        Args:
            min_strength: Minimum causal strength threshold
        """
        self.min_strength = min_strength
        self._causal_cache: Dict[Tuple[str, str], CausalRelation] = {}

        logger.info(f"CausalInferenceEngine initialized (min_strength={min_strength})")

    def infer_causality(
        self,
        knowledge_graph: Any,
        entity_a: str,
        entity_b: str
    ) -> Optional[CausalRelation]:
        """
        Infer causal relationship between two entities.

        Args:
            knowledge_graph: Knowledge graph
            entity_a: First entity
            entity_b: Second entity

        Returns:
            CausalRelation if causal link found, None otherwise
        """
        # Check cache
        cache_key = (entity_a, entity_b)
        if cache_key in self._causal_cache:
            return self._causal_cache[cache_key]

        # Check direct causal relations
        direct_relation = self._check_direct_causality(knowledge_graph, entity_a, entity_b)
        if direct_relation:
            self._causal_cache[cache_key] = direct_relation
            return direct_relation

        # Check indirect causality (via intermediaries)
        indirect_relation = self._check_indirect_causality(knowledge_graph, entity_a, entity_b)
        if indirect_relation:
            self._causal_cache[cache_key] = indirect_relation
            return indirect_relation

        return None

    def find_all_causal_relations(
        self,
        knowledge_graph: Any,
        entity: str,
        max_depth: int = 2
    ) -> Dict[str, List[CausalRelation]]:
        """
        Find all causal relations involving an entity.

        Args:
            knowledge_graph: Knowledge graph
            entity: Entity to analyze
            max_depth: Maximum graph depth to search

        Returns:
            Dict with 'causes' and 'effects' lists
        """
        relations = {
            "causes": [],  # What this entity causes
            "effects": [],  # What causes this entity
            "correlates": []  # What correlates with this entity
        }

        # BFS search for causal relations
        visited = set()
        queue = [(entity, 0, None)]  # (entity, depth, relation_type)

        while queue:
            current, depth, rel_type = queue.pop(0)

            if depth > max_depth:
                continue

            if current in visited:
                continue
            visited.add(current)

            # Get neighbors
            if hasattr(knowledge_graph, 'graph'):
                graph = knowledge_graph.graph

                if current in graph:
                    for neighbor in graph.neighbors(current):
                        edges = graph.get_edge_data(current, neighbor)

                        if edges:
                            for edge_data in edges.values():
                                relation = edge_data.get('type', '')

                                # Check if causal relation
                                if relation in self.CAUSAL_RELATIONS:
                                    causal_rel = CausalRelation(
                                        cause=current,
                                        effect=neighbor,
                                        direction=CausalDirection.FORWARD,
                                        strength=0.8,
                                        evidence=[f"{current} {relation} {neighbor}"],
                                        confidence=0.85
                                    )

                                    if current == entity:
                                        relations["causes"].append(causal_rel)
                                    else:
                                        relations["effects"].append(causal_rel)

                                queue.append((neighbor, depth + 1, relation))

        return relations

    def _check_direct_causality(
        self,
        knowledge_graph: Any,
        entity_a: str,
        entity_b: str
    ) -> Optional[CausalRelation]:
        """Check for direct causal edge between entities."""
        if hasattr(knowledge_graph, 'graph'):
            graph = knowledge_graph.graph

            if entity_a in graph and entity_b in graph.neighbors(entity_a):
                edges = graph.get_edge_data(entity_a, entity_b)

                if edges:
                    for edge_data in edges.values():
                        relation = edge_data.get('type', '')

                        if relation in self.CAUSAL_RELATIONS:
                            return CausalRelation(
                                cause=entity_a,
                                effect=entity_b,
                                direction=CausalDirection.FORWARD,
                                strength=0.9,  # Direct causal link = high strength
                                evidence=[f"Direct relation: {entity_a} {relation} {entity_b}"],
                                confidence=0.95
                            )

        return None

    def _check_indirect_causality(
        self,
        knowledge_graph: Any,
        entity_a: str,
        entity_b: str,
        max_hops: int = 3
    ) -> Optional[CausalRelation]:
        """Check for indirect causal chain between entities."""
        # Use multi-hop reasoner to find paths
        reasoner = MultiHopReasoner(max_hops=max_hops)
        paths = reasoner.find_paths(knowledge_graph, entity_a, entity_b, max_hops)

        if not paths:
            return None

        # Check if path contains causal relations
        for path in paths:
            causal_steps = []

            for step in path.steps:
                relation = step.get('relation', '')
                if relation in self.CAUSAL_RELATIONS:
                    causal_steps.append(step)

            # If path contains causal relations, infer indirect causality
            if causal_steps:
                strength = 0.7 * (0.9 ** (path.total_hops - 1))  # Decay with distance

                return CausalRelation(
                    cause=entity_a,
                    effect=entity_b,
                    direction=CausalDirection.FORWARD,
                    strength=max(strength, self.min_strength),
                    evidence=[f"Indirect via {len(causal_steps)} causal steps"],
                    confidence=0.75
                )

        return None


# ============================================================================
# Counterfactual Analyzer
# ============================================================================

class CounterfactualAnalyzer:
    """
    Counterfactual 'what if' scenario analysis.

    Analyzes hypothetical scenarios by:
    - Modifying conditions in the knowledge graph
    - Tracing downstream effects
    - Predicting outcomes

    Example:
        Query: "What if varroa mites were eliminated?"

        Analysis:
        1. Remove varroa_mites node
        2. Trace effects: honey_bee_health improves
        3. Downstream: colony_collapse_rate decreases
        4. Prediction: "Bee population would increase by ~30%"
    """

    def __init__(self):
        """Initialize counterfactual analyzer."""
        logger.info("CounterfactualAnalyzer initialized")

    def analyze_counterfactual(
        self,
        knowledge_graph: Any,
        modification: str,
        entity: str,
        causal_engine: CausalInferenceEngine
    ) -> CounterfactualScenario:
        """
        Analyze a counterfactual scenario.

        Args:
            knowledge_graph: Knowledge graph
            modification: Type of modification ("remove", "add", "increase", "decrease")
            entity: Entity to modify
            causal_engine: Causal inference engine for tracing effects

        Returns:
            CounterfactualScenario with predicted outcomes
        """
        scenario = CounterfactualScenario(
            original_condition=f"{entity} exists in current state",
            modified_condition=f"{modification} {entity}",
            predicted_outcome="",
            confidence=0.7,
            affected_entities=[],
            reasoning=[]
        )

        # Find causal relations
        causal_relations = causal_engine.find_all_causal_relations(
            knowledge_graph, entity, max_depth=3
        )

        # Trace effects based on modification type
        if modification == "remove":
            scenario.reasoning.append(f"Removing {entity}")

            # Effects this entity causes would not occur
            for causal_rel in causal_relations["causes"]:
                effect_entity = causal_rel.effect
                scenario.affected_entities.append(effect_entity)
                scenario.reasoning.append(
                    f"→ {effect_entity} would not be affected by {entity}"
                )

            scenario.predicted_outcome = f"Removing {entity} would affect {len(scenario.affected_entities)} downstream entities"

        elif modification == "increase":
            scenario.reasoning.append(f"Increasing {entity}")

            # Amplify causal effects
            for causal_rel in causal_relations["causes"]:
                effect_entity = causal_rel.effect
                scenario.affected_entities.append(effect_entity)
                scenario.reasoning.append(
                    f"→ {effect_entity} would increase (strength: {causal_rel.strength:.2f})"
                )

            scenario.predicted_outcome = f"Increasing {entity} would amplify {len(scenario.affected_entities)} effects"

        elif modification == "decrease":
            scenario.reasoning.append(f"Decreasing {entity}")

            # Weaken causal effects
            for causal_rel in causal_relations["causes"]:
                effect_entity = causal_rel.effect
                scenario.affected_entities.append(effect_entity)
                scenario.reasoning.append(
                    f"→ {effect_entity} would decrease (strength: {causal_rel.strength:.2f})"
                )

            scenario.predicted_outcome = f"Decreasing {entity} would weaken {len(scenario.affected_entities)} effects"

        # Calculate confidence based on number of affected entities and causal strengths
        if scenario.affected_entities:
            avg_strength = sum(r.strength for r in causal_relations["causes"][:5]) / min(5, len(causal_relations["causes"]) + 1)
            scenario.confidence = min(0.95, avg_strength * 0.9)

        return scenario

    def compare_scenarios(
        self,
        scenarios: List[CounterfactualScenario]
    ) -> Dict[str, Any]:
        """
        Compare multiple counterfactual scenarios.

        Args:
            scenarios: List of scenarios to compare

        Returns:
            Comparison analysis
        """
        comparison = {
            "best_scenario": None,
            "worst_scenario": None,
            "most_affected_entities": [],
            "summary": []
        }

        if not scenarios:
            return comparison

        # Sort by number of affected entities (proxy for impact)
        sorted_scenarios = sorted(
            scenarios,
            key=lambda s: len(s.affected_entities),
            reverse=True
        )

        comparison["best_scenario"] = sorted_scenarios[-1]  # Least impact = most stable
        comparison["worst_scenario"] = sorted_scenarios[0]  # Most impact = most disruptive

        # Find most commonly affected entities
        entity_counts = defaultdict(int)
        for scenario in scenarios:
            for entity in scenario.affected_entities:
                entity_counts[entity] += 1

        comparison["most_affected_entities"] = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Generate summary
        for scenario in scenarios:
            comparison["summary"].append({
                "modification": scenario.modified_condition,
                "outcome": scenario.predicted_outcome,
                "affected_count": len(scenario.affected_entities),
                "confidence": scenario.confidence
            })

        return comparison


# ============================================================================
# Reasoning Department
# ============================================================================

class ReasoningDepartment(BaseDepartment):
    """
    Reasoning Department - Advanced multi-hop reasoning and causal inference.

    Provides sophisticated reasoning capabilities:
    1. Multi-hop queries: Find connections across multiple graph hops
    2. Causal inference: Identify cause-effect relationships
    3. Counterfactual analysis: "What if" scenario prediction
    4. Reasoning explanation: Provide justifications for conclusions

    Example:

        async with ReasoningDepartment(registry=registry) as dept:
            # Multi-hop query
            request = DepartmentRequest(
                task_id="reason_001",
                task_type="multi_hop_query",
                parameters={
                    "query": "What connects varroa mites to hive temperature?",
                    "start_entity": "varroa_mites",
                    "end_entity": "hive_temperature",
                    "max_hops": 5
                }
            )

            response = await dept.execute(request)
            # Returns reasoning chains showing connections
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        max_hops: int = 5,
        min_causal_strength: float = 0.5,
        dept_config: Optional[DepartmentConfig] = None
    ):
        """
        Initialize Reasoning Department.

        Args:
            registry: Department registry for accessing other departments
            max_hops: Maximum hops for multi-hop reasoning
            min_causal_strength: Minimum causal strength threshold
            dept_config: Department configuration
        """
        super().__init__(
            name="reasoning",
            domain="advanced",
            version="1.0.0",
            supported_tasks=[
                "multi_hop_query",
                "causal_inference",
                "counterfactual",
                "explain_reasoning"
            ],
            confidence_range=(0.70, 0.95),
            config=dept_config or DepartmentConfig()
        )

        self.registry = registry

        # Initialize reasoning engines
        self.multi_hop_reasoner = MultiHopReasoner(
            max_hops=max_hops,
            min_confidence=0.6
        )

        self.causal_engine = CausalInferenceEngine(
            min_strength=min_causal_strength
        )

        self.counterfactual_analyzer = CounterfactualAnalyzer()

        # Cache for knowledge graph reference
        self._knowledge_graph = None

        logger.info(f"Reasoning Department initialized (max_hops={max_hops})")

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize reasoning department."""
        if self._initialized:
            return

        await super().initialize()

    async def close(self) -> None:
        """Cleanup resources."""
        await super().close()

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """Execute a reasoning task."""
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        try:
            # Get knowledge graph (from infrastructure department or registry)
            kg = await self._get_knowledge_graph()

            if request.task_type == "multi_hop_query":
                result, confidence = await self._multi_hop_query(request, kg)
            elif request.task_type == "causal_inference":
                result, confidence = await self._causal_inference(request, kg)
            elif request.task_type == "counterfactual":
                result, confidence = await self._counterfactual(request, kg)
            elif request.task_type == "explain_reasoning":
                result, confidence = await self._explain_reasoning(request, kg)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score
            }

            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            self._record_request(
                success=True,
                latency_ms=response.execution_time_ms,
                confidence=confidence.score
            )

            return response

        except Exception as e:
            self._record_error(e)

            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """Verify reasoning quality."""
        confidence_valid = response.confidence.score >= 0.70
        has_error = "error" in response.result
        has_reasoning = "reasoning_chains" in response.result or "causal_relations" in response.result

        reasoning_sound = not has_error and has_reasoning
        sufficient = confidence_valid and reasoning_sound

        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {
                "retry": True,
                "reason": "Low confidence or missing reasoning chains"
            }

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=reasoning_sound,
            alternative_paths=["expand_search_depth", "alternative_reasoning_strategy"] if not sufficient else [],
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.5,
            escalation_reason="Very low confidence" if response.confidence.score < 0.5 else None,
            verifier_confidence=0.85
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """Refine reasoning operation."""
        # Increase search depth and retry
        if "max_hops" in request.parameters:
            request.parameters["max_hops"] += 2

        return await self.execute(request)

    # ========================================================================
    # Task Handlers
    # ========================================================================

    async def _multi_hop_query(
        self,
        request: DepartmentRequest,
        kg: Any
    ) -> Tuple[Dict[str, Any], ConfidenceMetadata]:
        """Execute multi-hop reasoning query."""
        start_entity = request.parameters.get("start_entity")
        end_entity = request.parameters.get("end_entity")
        max_hops = request.parameters.get("max_hops", 5)

        if not start_entity or not end_entity:
            raise ValueError("Missing required parameters: 'start_entity', 'end_entity'")

        # Find paths
        paths = self.multi_hop_reasoner.find_paths(kg, start_entity, end_entity, max_hops)

        # Build result
        result = {
            "start_entity": start_entity,
            "end_entity": end_entity,
            "paths_found": len(paths),
            "reasoning_chains": [
                {
                    "steps": chain.steps,
                    "total_hops": chain.total_hops,
                    "confidence": chain.confidence
                }
                for chain in paths[:5]  # Top 5 paths
            ]
        }

        # Calculate confidence
        if paths:
            best_chain = paths[0]
            confidence_score = best_chain.confidence
        else:
            confidence_score = 0.3  # Low confidence if no paths found

        confidence = ConfidenceMetadata.from_score(
            confidence_score,
            justification=[f"Found {len(paths)} paths connecting entities"],
            uncertainty_sources=["Graph incompleteness"] if len(paths) == 0 else []
        )

        return result, confidence

    async def _causal_inference(
        self,
        request: DepartmentRequest,
        kg: Any
    ) -> Tuple[Dict[str, Any], ConfidenceMetadata]:
        """Infer causal relationships."""
        entity_a = request.parameters.get("entity_a")
        entity_b = request.parameters.get("entity_b")

        if not entity_a:
            raise ValueError("Missing required parameter: 'entity_a'")

        if entity_b:
            # Infer causality between two entities
            causal_rel = self.causal_engine.infer_causality(kg, entity_a, entity_b)

            if causal_rel:
                result = {
                    "causal_relation": {
                        "cause": causal_rel.cause,
                        "effect": causal_rel.effect,
                        "direction": causal_rel.direction.value,
                        "strength": causal_rel.strength,
                        "evidence": causal_rel.evidence
                    }
                }
                confidence_score = causal_rel.confidence
            else:
                result = {
                    "causal_relation": None,
                    "message": f"No causal relationship found between {entity_a} and {entity_b}"
                }
                confidence_score = 0.4
        else:
            # Find all causal relations for entity_a
            causal_relations = self.causal_engine.find_all_causal_relations(kg, entity_a)

            result = {
                "entity": entity_a,
                "causes_count": len(causal_relations["causes"]),
                "effects_count": len(causal_relations["effects"]),
                "causes": [
                    {
                        "effect": rel.effect,
                        "strength": rel.strength,
                        "confidence": rel.confidence
                    }
                    for rel in causal_relations["causes"][:10]
                ],
                "effects": [
                    {
                        "cause": rel.cause,
                        "strength": rel.strength,
                        "confidence": rel.confidence
                    }
                    for rel in causal_relations["effects"][:10]
                ]
            }

            confidence_score = 0.85 if causal_relations["causes"] or causal_relations["effects"] else 0.5

        confidence = ConfidenceMetadata.from_score(
            confidence_score,
            justification=["Causal inference based on graph structure"],
            uncertainty_sources=[]
        )

        return result, confidence

    async def _counterfactual(
        self,
        request: DepartmentRequest,
        kg: Any
    ) -> Tuple[Dict[str, Any], ConfidenceMetadata]:
        """Analyze counterfactual scenario."""
        modification = request.parameters.get("modification", "remove")
        entity = request.parameters.get("entity")

        if not entity:
            raise ValueError("Missing required parameter: 'entity'")

        # Analyze counterfactual
        scenario = self.counterfactual_analyzer.analyze_counterfactual(
            kg, modification, entity, self.causal_engine
        )

        result = {
            "original_condition": scenario.original_condition,
            "modified_condition": scenario.modified_condition,
            "predicted_outcome": scenario.predicted_outcome,
            "affected_entities": scenario.affected_entities,
            "reasoning": scenario.reasoning,
            "confidence": scenario.confidence
        }

        confidence = ConfidenceMetadata.from_score(
            scenario.confidence,
            justification=["Counterfactual analysis via causal tracing"],
            uncertainty_sources=["Indirect effects not fully modeled"]
        )

        return result, confidence

    async def _explain_reasoning(
        self,
        request: DepartmentRequest,
        kg: Any
    ) -> Tuple[Dict[str, Any], ConfidenceMetadata]:
        """Explain reasoning for a conclusion."""
        conclusion = request.parameters.get("conclusion")
        entity = request.parameters.get("entity")

        if not conclusion:
            raise ValueError("Missing required parameter: 'conclusion'")

        # Build explanation using causal relations
        explanations = []

        if entity:
            causal_relations = self.causal_engine.find_all_causal_relations(kg, entity)

            for causal_rel in causal_relations["causes"][:5]:
                explanations.append(
                    f"{entity} causes {causal_rel.effect} (strength: {causal_rel.strength:.2f})"
                )

            for causal_rel in causal_relations["effects"][:5]:
                explanations.append(
                    f"{causal_rel.cause} causes {entity} (strength: {causal_rel.strength:.2f})"
                )

        result = {
            "conclusion": conclusion,
            "explanations": explanations,
            "explanation_count": len(explanations)
        }

        confidence = ConfidenceMetadata.from_score(
            0.80,
            justification=["Explanation based on causal relations"],
            uncertainty_sources=[]
        )

        return result, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _get_knowledge_graph(self) -> Any:
        """Get knowledge graph from infrastructure department."""
        if self._knowledge_graph is not None:
            return self._knowledge_graph

        # For now, return a mock KG (in practice, fetch from infrastructure department)
        # This will be implemented properly when integrating with infrastructure
        from HoloLoom.memory.graph import KG

        self._knowledge_graph = KG()
        return self._knowledge_graph

    def _build_reasoning(
        self,
        result: Dict[str, Any],
        request: DepartmentRequest
    ) -> Dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "task_type": request.task_type,
            "reasoning_depth": result.get("paths_found", 0) or result.get("causes_count", 0)
        }

    async def _build_session_state(
        self,
        session_id: Optional[str],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_reasoning_task": result.get("start_entity") or result.get("entity")
        }
