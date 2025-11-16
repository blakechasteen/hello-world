"""
Tree-of-Thought Planning
========================
Systematic exploration of solution space using tree-structured reasoning.

Implements deliberate tree search for complex planning:
- Multi-step reasoning with backtracking
- Beam search for efficient exploration
- Quality-based pruning
- Complete solution paths

Based on research from:
- Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving"
- Silver et al. (2016): AlphaGo and tree search
- OpenAI: Chain-of-Thought and step-by-step reasoning

**Implementation Date**: 2025-11-16
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import re

logger = logging.getLogger("HoloLoom.alignment.tree_of_thought")


@dataclass
class ThoughtNode:
    """Node in tree-of-thought.

    Represents one step in reasoning process.
    """
    thought: str
    depth: int
    parent: Optional['ThoughtNode']
    children: List['ThoughtNode'] = field(default_factory=list)
    value: float = 0.0  # Quality score (0.0-1.0)
    complete: bool = False  # Is this a complete solution?
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_path(self) -> List['ThoughtNode']:
        """Get path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_path_text(self) -> str:
        """Get text representation of path."""
        path = self.get_path()
        return " → ".join(node.thought for node in path)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "thought": self.thought,
            "depth": self.depth,
            "value": self.value,
            "complete": self.complete,
            "children_count": len(self.children),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TreeOfThoughtResult:
    """Result of tree-of-thought planning.

    Contains complete exploration tree and best solution path.
    """
    problem: str
    root: ThoughtNode
    best_path: List[ThoughtNode]
    all_solutions: List[List[ThoughtNode]]
    exploration_stats: Dict[str, Any]
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_best_solution(self) -> str:
        """Get best solution as text."""
        if not self.best_path:
            return "No solution found"
        return " → ".join(node.thought for node in self.best_path)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        stats = self.exploration_stats
        lines = [
            f"Problem: {self.problem}",
            f"Nodes explored: {stats.get('nodes_explored', 0)}",
            f"Max depth: {stats.get('max_depth_reached', 0)}",
            f"Solutions found: {stats.get('solutions_found', 0)}",
            f"Best path length: {len(self.best_path)}",
            f"Best value: {self.best_path[-1].value if self.best_path else 0.0:.2f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "problem": self.problem,
            "best_solution": self.get_best_solution(),
            "best_value": self.best_path[-1].value if self.best_path else 0.0,
            "solutions_count": len(self.all_solutions),
            "exploration_stats": self.exploration_stats,
            "reasoning_trace": self.reasoning_trace,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class TreeOfThought:
    """Systematic exploration of solution space.

    Uses tree search with beam pruning to explore multiple reasoning paths
    and find high-quality solutions.

    Example:
        tot = TreeOfThought(max_depth=5, beam_width=3)
        result = await tot.plan(
            problem="Design authentication system",
            constraints=["Secure", "User-friendly", "Scalable"]
        )

        # Explores tree:
        # Depth 1: [Password, Token, Biometric]
        # Depth 2 (Token): [JWT, OAuth, API Keys]
        # Depth 3 (JWT): [Short-lived, Long-lived + refresh]
        # ...

        # Returns best path through tree
    """

    def __init__(
        self,
        max_depth: int = 5,
        beam_width: int = 3,
        min_value_threshold: float = 0.3,
        evaluation_fn: Optional[Callable] = None,
        expansion_fn: Optional[Callable] = None,
    ):
        """Initialize tree-of-thought planner.

        Args:
            max_depth: Maximum tree depth
            beam_width: Number of nodes to keep at each level
            min_value_threshold: Minimum value to continue exploring
            evaluation_fn: Custom evaluation function
            expansion_fn: Custom node expansion function
        """
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_value_threshold = min_value_threshold
        self.evaluate = evaluation_fn or self._default_evaluate
        self.expansion_fn = expansion_fn  # Custom expansion function (optional)
        self.planning_history: List[TreeOfThoughtResult] = []

        logger.info(
            f"Initialized TreeOfThought with max_depth={max_depth}, "
            f"beam_width={beam_width}"
        )

    async def plan(
        self,
        problem: str,
        constraints: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TreeOfThoughtResult:
        """Generate and explore solution tree.

        Args:
            problem: Problem statement to solve
            constraints: List of constraints/requirements
            context: Additional context dictionary

        Returns:
            TreeOfThoughtResult with best path and exploration statistics

        Example:
            result = await tot.plan(
                problem="Design user authentication",
                constraints=["Secure", "User-friendly"],
                context={"users": 10000, "compliance": "GDPR"}
            )
        """
        constraints = constraints or []
        context = context or {}

        logger.info(f"Planning for problem: {problem}")
        reasoning_trace = [
            f"Problem: {problem}",
            f"Constraints: {constraints}",
            f"Max depth: {self.max_depth}",
            f"Beam width: {self.beam_width}",
        ]

        # Initialize root
        root = ThoughtNode(
            thought=f"Problem: {problem}",
            depth=0,
            parent=None,
            metadata={"problem": problem, "constraints": constraints},
        )

        # Explore tree using beam search
        current_frontier = [root]
        all_solutions: List[List[ThoughtNode]] = []
        nodes_explored = 0

        for depth in range(1, self.max_depth + 1):
            reasoning_trace.append(f"\n--- Depth {depth} ---")
            next_frontier = []

            for node in current_frontier:
                # Generate child thoughts
                children = await self._generate_thoughts(
                    node,
                    problem,
                    constraints,
                    context,
                )
                nodes_explored += len(children)

                reasoning_trace.append(
                    f"Expanded '{node.thought[:50]}...' → {len(children)} children"
                )

                # Evaluate each child
                for child in children:
                    child.value = await self.evaluate(child, problem, constraints, context)

                    # Check if complete solution
                    if self._is_complete_solution(child, problem, constraints):
                        child.complete = True
                        solution_path = child.get_path()
                        all_solutions.append(solution_path)
                        reasoning_trace.append(
                            f"✓ Complete solution found (value={child.value:.2f}): "
                            f"{child.thought[:50]}..."
                        )

                # Keep best children (beam search)
                children.sort(key=lambda c: c.value, reverse=True)
                node.children = children[:self.beam_width]

                # Add to next frontier (only if above threshold)
                for child in node.children:
                    if child.value >= self.min_value_threshold:
                        next_frontier.append(child)

                reasoning_trace.append(
                    f"Kept {len(node.children)} children (values: "
                    f"{[f'{c.value:.2f}' for c in node.children]})"
                )

            # Prune frontier to beam width (keep best nodes)
            next_frontier.sort(key=lambda n: n.value, reverse=True)
            current_frontier = next_frontier[:self.beam_width]

            reasoning_trace.append(
                f"Frontier size: {len(current_frontier)} "
                f"(pruned from {len(next_frontier)})"
            )

            # Early stopping if we have enough high-quality solutions
            if len(all_solutions) >= self.beam_width:
                high_quality = sum(
                    1 for sol in all_solutions
                    if sol[-1].value >= 0.8
                )
                if high_quality >= 2:
                    reasoning_trace.append(
                        f"Early stop: {high_quality} high-quality solutions found"
                    )
                    break

            # Stop if frontier is empty
            if not current_frontier:
                reasoning_trace.append("Frontier exhausted - stopping")
                break

        # Select best path
        if all_solutions:
            best_path = max(all_solutions, key=lambda path: path[-1].value)
            reasoning_trace.append(
                f"Best complete solution: value={best_path[-1].value:.2f}"
            )
        else:
            # No complete solution - use deepest high-value path
            best_path = self._find_best_incomplete_path(root)
            reasoning_trace.append(
                f"No complete solution - best partial: value={best_path[-1].value:.2f}"
            )

        # Compile statistics
        exploration_stats = {
            "nodes_explored": nodes_explored,
            "max_depth_reached": depth,
            "solutions_found": len(all_solutions),
            "complete_solutions": sum(1 for s in all_solutions if s[-1].complete),
            "frontier_size_final": len(current_frontier),
            "best_value": best_path[-1].value if best_path else 0.0,
        }

        result = TreeOfThoughtResult(
            problem=problem,
            root=root,
            best_path=best_path,
            all_solutions=all_solutions,
            exploration_stats=exploration_stats,
            reasoning_trace=reasoning_trace,
            metadata={
                "constraints": constraints,
                "context": context,
                "max_depth": self.max_depth,
                "beam_width": self.beam_width,
            },
        )

        self.planning_history.append(result)
        logger.info(
            f"Planning complete: {exploration_stats['nodes_explored']} nodes, "
            f"{exploration_stats['solutions_found']} solutions, "
            f"best_value={exploration_stats['best_value']:.2f}"
        )

        return result

    async def _generate_thoughts(
        self,
        parent: ThoughtNode,
        problem: str,
        constraints: List[str],
        context: Dict[str, Any],
    ) -> List[ThoughtNode]:
        """Generate child thoughts from parent node."""
        # Use custom expansion function if provided
        if self.expansion_fn is not None:
            return await self.expansion_fn(parent, problem, constraints, context)

        # Default expansion logic
        depth = parent.depth + 1
        thoughts = []

        # Generate based on depth and problem domain
        if depth == 1:
            # Top-level approaches
            thoughts = self._generate_top_level_approaches(problem)
        elif depth <= 3:
            # Refinement and details
            thoughts = self._generate_refinements(parent, problem, constraints)
        else:
            # Implementation details
            thoughts = self._generate_implementation_details(parent, constraints)

        # Create ThoughtNodes
        nodes = []
        for thought_text in thoughts:
            node = ThoughtNode(
                thought=thought_text,
                depth=depth,
                parent=parent,
                metadata={"problem": problem},
            )
            nodes.append(node)

        return nodes

    def _generate_top_level_approaches(self, problem: str) -> List[str]:
        """Generate top-level solution approaches."""
        problem_lower = problem.lower()

        # Authentication/security problems
        if "auth" in problem_lower or "login" in problem_lower:
            return [
                "Password-based authentication",
                "Token-based authentication (JWT/OAuth)",
                "Biometric authentication",
                "Multi-factor authentication (MFA)",
                "Passwordless authentication (WebAuthn)",
            ]

        # Data storage problems
        elif "storage" in problem_lower or "database" in problem_lower:
            return [
                "Relational database (SQL)",
                "Document store (NoSQL)",
                "Key-value store (Redis)",
                "Graph database (Neo4j)",
                "Hybrid approach",
            ]

        # API design problems
        elif "api" in problem_lower or "interface" in problem_lower:
            return [
                "REST API",
                "GraphQL API",
                "gRPC",
                "WebSocket (real-time)",
                "Hybrid REST + WebSocket",
            ]

        # Scaling problems
        elif "scale" in problem_lower or "performance" in problem_lower:
            return [
                "Horizontal scaling (load balancing)",
                "Vertical scaling (more powerful servers)",
                "Caching layer (Redis/Memcached)",
                "Database sharding",
                "Microservices architecture",
            ]

        # Generic problem
        else:
            return [
                "Traditional/proven approach",
                "Modern/innovative approach",
                "Hybrid approach",
                "Minimal viable solution",
                "Comprehensive solution",
            ]

    def _generate_refinements(
        self,
        parent: ThoughtNode,
        problem: str,
        constraints: List[str],
    ) -> List[str]:
        """Generate refinements of parent approach."""
        parent_text = parent.thought.lower()

        # Token-based auth refinements
        if "token" in parent_text or "jwt" in parent_text:
            return [
                "Short-lived access tokens (15 min)",
                "Long-lived access tokens with refresh",
                "Sliding session windows",
                "Stateless JWT with claims",
                "OAuth 2.0 with authorization server",
            ]

        # Database refinements
        elif "database" in parent_text or "sql" in parent_text:
            return [
                "Single primary database",
                "Primary with read replicas",
                "Multi-region with sync",
                "Sharded by user ID",
                "Separate read/write databases",
            ]

        # Scaling refinements
        elif "scale" in parent_text or "caching" in parent_text:
            return [
                "Simple cache-aside pattern",
                "Write-through caching",
                "Read-through caching",
                "Distributed cache with Redis Cluster",
                "Multi-tier caching (L1 + L2)",
            ]

        # Generic refinements
        else:
            return [
                f"Optimize {parent.thought} for performance",
                f"Optimize {parent.thought} for security",
                f"Add monitoring to {parent.thought}",
                f"Simplify {parent.thought}",
                f"Scale {parent.thought} horizontally",
            ]

    def _generate_implementation_details(
        self,
        parent: ThoughtNode,
        constraints: List[str],
    ) -> List[str]:
        """Generate implementation-level details."""
        parent_text = parent.thought.lower()

        # Security-related implementations
        if any(c.lower() in ["secure", "security"] for c in constraints):
            return [
                "Implement rate limiting",
                "Add encryption at rest",
                "Enable audit logging",
                "Set up security headers",
                "Implement input validation",
            ]

        # Scalability implementations
        elif any(c.lower() in ["scale", "scalable"] for c in constraints):
            return [
                "Add connection pooling",
                "Implement async processing",
                "Add message queue",
                "Enable auto-scaling",
                "Optimize database queries",
            ]

        # User-friendly implementations
        elif any(c.lower() in ["user-friendly", "usability"] for c in constraints):
            return [
                "Add progress indicators",
                "Implement error recovery",
                "Add helpful error messages",
                "Enable offline mode",
                "Optimize loading times",
            ]

        # Generic implementations
        else:
            return [
                "Add comprehensive tests",
                "Implement CI/CD pipeline",
                "Add monitoring and alerts",
                "Write documentation",
                "Implement error handling",
            ]

    async def _default_evaluate(
        self,
        node: ThoughtNode,
        problem: str,
        constraints: List[str],
        context: Dict[str, Any],
    ) -> float:
        """Default evaluation function for nodes.

        Returns value between 0.0 and 1.0.
        """
        score = 0.5  # Base score

        thought_lower = node.thought.lower()

        # Reward completeness
        if any(word in thought_lower for word in ["implement", "add", "enable", "set up"]):
            score += 0.1

        # Reward security mentions when security is a constraint
        if any("secure" in c.lower() for c in constraints):
            if any(word in thought_lower for word in ["encrypt", "auth", "secure", "validate"]):
                score += 0.15

        # Reward scalability mentions when scale is a constraint
        if any("scale" in c.lower() for c in constraints):
            if any(word in thought_lower for word in ["scale", "cache", "async", "distributed"]):
                score += 0.15

        # Reward depth (deeper = more concrete)
        score += min(0.2, node.depth * 0.05)

        # Penalize vague thoughts
        if any(word in thought_lower for word in ["consider", "maybe", "might", "could"]):
            score -= 0.1

        # Reward specific numbers/metrics
        if re.search(r'\d+', thought_lower):
            score += 0.05

        return max(0.0, min(1.0, score))

    def _is_complete_solution(
        self,
        node: ThoughtNode,
        problem: str,
        constraints: List[str],
    ) -> bool:
        """Check if node represents a complete solution."""
        # Complete if:
        # 1. Depth >= 3 (sufficient detail)
        # 2. Value >= 0.7 (high quality)
        # 3. Contains implementation language

        if node.depth < 3:
            return False

        if node.value < 0.7:
            return False

        thought_lower = node.thought.lower()
        implementation_words = [
            "implement", "add", "enable", "set up", "configure",
            "deploy", "install", "create", "build"
        ]

        return any(word in thought_lower for word in implementation_words)

    def _find_best_incomplete_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Find best path in tree (for cases with no complete solution)."""
        best_path = [root]
        best_value = 0.0

        # DFS to find best leaf
        def dfs(node: ThoughtNode, path: List[ThoughtNode]):
            nonlocal best_path, best_value

            if not node.children:
                # Leaf node
                if node.value > best_value:
                    best_value = node.value
                    best_path = path.copy()
            else:
                # Continue exploring
                for child in node.children:
                    dfs(child, path + [child])

        dfs(root, [root])
        return best_path

    def _count_nodes(self, root: ThoughtNode) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count

    def visualize_tree(
        self,
        result: TreeOfThoughtResult,
        max_width: int = 80,
    ) -> str:
        """Generate ASCII visualization of tree.

        Returns:
            String with ASCII tree visualization
        """
        lines = [f"Problem: {result.problem}", ""]

        def add_node(node: ThoughtNode, prefix: str, is_last: bool):
            # Node text
            connector = "└─ " if is_last else "├─ "
            thought_text = node.thought[:max_width-len(prefix)-10]
            value_text = f"[{node.value:.2f}]"
            complete_marker = " ✓" if node.complete else ""

            lines.append(f"{prefix}{connector}{thought_text} {value_text}{complete_marker}")

            # Children
            if node.children:
                extension = "   " if is_last else "│  "
                for i, child in enumerate(node.children):
                    add_node(child, prefix + extension, i == len(node.children) - 1)

        add_node(result.root, "", True)
        return "\n".join(lines)

    def get_planning_history(self) -> List[TreeOfThoughtResult]:
        """Get all planning results."""
        return self.planning_history

    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        if not self.planning_history:
            return {
                "total_plans": 0,
                "avg_nodes_explored": 0.0,
                "avg_solutions_found": 0.0,
                "avg_best_value": 0.0,
            }

        total = len(self.planning_history)

        return {
            "total_plans": total,
            "avg_nodes_explored": sum(
                r.exploration_stats["nodes_explored"]
                for r in self.planning_history
            ) / total,
            "avg_solutions_found": sum(
                r.exploration_stats["solutions_found"]
                for r in self.planning_history
            ) / total,
            "avg_best_value": sum(
                r.exploration_stats["best_value"]
                for r in self.planning_history
            ) / total,
            "completion_rate": sum(
                1 for r in self.planning_history
                if r.exploration_stats["complete_solutions"] > 0
            ) / total,
        }
