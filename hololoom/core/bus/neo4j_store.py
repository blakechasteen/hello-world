"""
Neo4j MCTS Store — Persistent MCTS tree and pattern-domain graphs.
====================================================================

Extends the existing neo4j_schema.py with:
- MCTSNode: Tree nodes with visit counts, values, actions
- MCTSTree: Named trees (strategic, tactical) with domain tags
- PatternDomain: Strategy-domain performance associations

The store provides:
- save_tree(): Persist an MCTS tree after search
- load_tree(): Warm-start from a previously stored tree
- backprop_update(): Update node statistics from execution results
- query_best_strategy(): Look up best-performing strategy for a domain
- record_strategy_outcome(): Update pattern-domain association

Graceful degradation: All methods return sensible defaults when Neo4j
is unavailable. The system works without Neo4j (in-memory only),
and gets smarter with it (cross-session learning).

Node labels added:
- MCTSNode: id, tree_id, state_sig, action, visits, total_value, depth
- MCTSTree: id, domain, level, created_at, updated_at
- PatternDomain: strategy, domain, successes, failures, total_confidence, uses

Relationships added:
- CHILD_OF: MCTSNode -> MCTSNode
- BELONGS_TO: MCTSNode -> MCTSTree
- PERFORMS_IN: PatternDomain (links strategy to domain performance)
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Schema Extensions
# ============================================================================

MCTS_SCHEMA_STATEMENTS = [
    # MCTSNode constraints and indexes
    "CREATE CONSTRAINT mcts_node_id_unique IF NOT EXISTS FOR (n:MCTSNode) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX mcts_node_tree IF NOT EXISTS FOR (n:MCTSNode) ON (n.tree_id)",
    "CREATE INDEX mcts_node_visits IF NOT EXISTS FOR (n:MCTSNode) ON (n.visits)",

    # MCTSTree constraints and indexes
    "CREATE CONSTRAINT mcts_tree_id_unique IF NOT EXISTS FOR (t:MCTSTree) REQUIRE t.id IS UNIQUE",
    "CREATE INDEX mcts_tree_domain IF NOT EXISTS FOR (t:MCTSTree) ON (t.domain)",
    "CREATE INDEX mcts_tree_level IF NOT EXISTS FOR (t:MCTSTree) ON (t.level)",

    # PatternDomain indexes
    "CREATE INDEX pattern_domain_strategy IF NOT EXISTS FOR (p:PatternDomain) ON (p.strategy)",
    "CREATE INDEX pattern_domain_domain IF NOT EXISTS FOR (p:PatternDomain) ON (p.domain)",
]


def ensure_mcts_schema(driver, database: str = "neo4j") -> dict:
    """
    Create MCTS-specific constraints and indexes.
    Extends the base schema from neo4j_schema.py.
    Idempotent — safe to call on every startup.
    """
    results = {"executed": 0, "errors": [], "skipped": 0}

    with driver.session(database=database) as session:
        for stmt in MCTS_SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
                results["executed"] += 1
            except Exception as e:
                err_msg = str(e)
                if "already exists" in err_msg.lower() or "equivalent" in err_msg.lower():
                    results["skipped"] += 1
                else:
                    results["errors"].append({"statement": stmt, "error": err_msg})
                    logger.warning("MCTS schema statement failed: %s — %s", stmt[:60], err_msg)

    logger.info(
        "MCTS schema: %d executed, %d skipped, %d errors",
        results["executed"], results["skipped"], len(results["errors"]),
    )
    return results


# ============================================================================
# MCTS Store
# ============================================================================

class MCTSStore:
    """
    Neo4j-backed store for MCTS trees and pattern-domain associations.

    Gracefully degrades when Neo4j is unavailable — returns empty results
    and logs warnings instead of raising exceptions.
    """

    def __init__(self, driver=None, database: str = "neo4j"):
        """
        Args:
            driver: Neo4j driver instance (None = degraded mode)
            database: Neo4j database name
        """
        self.driver = driver
        self.database = database
        self._available = driver is not None

        if self._available:
            try:
                ensure_mcts_schema(driver, database)
            except Exception as e:
                logger.warning(f"MCTS schema setup failed: {e}")
                self._available = False

    # ========================================================================
    # Tree Persistence
    # ========================================================================

    def save_tree(
        self,
        tree_id: str,
        domain: str,
        level: str,
        nodes: list[dict[str, Any]],
    ) -> bool:
        """
        Save an MCTS tree to Neo4j.

        Args:
            tree_id: Unique tree identifier
            domain: Domain (code, factual, reasoning, etc.)
            level: Tree level (strategic, tactical)
            nodes: List of node dicts with keys:
                   id, state_sig, action, visits, total_value, depth, parent_id

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._available:
            logger.debug("Neo4j unavailable — tree not persisted")
            return False

        try:
            with self.driver.session(database=self.database) as session:
                # Create/update tree node
                session.run(
                    """MERGE (t:MCTSTree {id: $tree_id})
                       SET t.domain = $domain, t.level = $level,
                           t.updated_at = $now""",
                    tree_id=tree_id, domain=domain, level=level,
                    now=datetime.now().isoformat(),
                )

                # Batch upsert nodes
                session.run(
                    """UNWIND $nodes AS n
                       MERGE (node:MCTSNode {id: n.id})
                       SET node.tree_id = $tree_id,
                           node.state_sig = n.state_sig,
                           node.action = n.action,
                           node.visits = n.visits,
                           node.total_value = n.total_value,
                           node.depth = n.depth
                       WITH node, n
                       MATCH (tree:MCTSTree {id: $tree_id})
                       MERGE (node)-[:BELONGS_TO]->(tree)""",
                    nodes=nodes, tree_id=tree_id,
                )

                # Create parent-child relationships
                edges = [n for n in nodes if n.get("parent_id")]
                if edges:
                    session.run(
                        """UNWIND $edges AS e
                           MATCH (child:MCTSNode {id: e.id})
                           MATCH (parent:MCTSNode {id: e.parent_id})
                           MERGE (child)-[:CHILD_OF]->(parent)""",
                        edges=edges,
                    )

            logger.info(f"Saved MCTS tree '{tree_id}' ({len(nodes)} nodes)")
            return True

        except Exception as e:
            logger.error(f"Failed to save MCTS tree: {e}")
            return False

    def load_tree(
        self,
        domain: str,
        level: str = "strategic",
    ) -> dict[str, Any] | None:
        """
        Load the most recent MCTS tree for a domain.

        Returns:
            Dict with 'tree_id', 'nodes', 'edges' or None if not found
        """
        if not self._available:
            return None

        try:
            with self.driver.session(database=self.database) as session:
                # Find most recently updated tree for this domain
                result = session.run(
                    """MATCH (t:MCTSTree {domain: $domain, level: $level})
                       RETURN t ORDER BY t.updated_at DESC LIMIT 1""",
                    domain=domain, level=level,
                )
                tree_record = result.single()
                if not tree_record:
                    return None

                tree_id = tree_record["t"]["id"]

                # Load all nodes
                nodes_result = session.run(
                    """MATCH (n:MCTSNode)-[:BELONGS_TO]->(t:MCTSTree {id: $tree_id})
                       RETURN n ORDER BY n.depth""",
                    tree_id=tree_id,
                )
                nodes = [dict(record["n"]) for record in nodes_result]

                # Load edges
                edges_result = session.run(
                    """MATCH (child:MCTSNode)-[:CHILD_OF]->(parent:MCTSNode)
                       WHERE child.tree_id = $tree_id
                       RETURN child.id AS child_id, parent.id AS parent_id""",
                    tree_id=tree_id,
                )
                edges = [
                    {"child_id": r["child_id"], "parent_id": r["parent_id"]}
                    for r in edges_result
                ]

                return {
                    "tree_id": tree_id,
                    "domain": domain,
                    "level": level,
                    "nodes": nodes,
                    "edges": edges,
                }

        except Exception as e:
            logger.error(f"Failed to load MCTS tree: {e}")
            return None

    def backprop_update(
        self,
        node_id: str,
        value: float,
    ) -> bool:
        """
        Update visit count and value for a node and all ancestors.

        Mirrors MCTSEngine._backpropagate but in Neo4j.
        """
        if not self._available:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    """MATCH (n:MCTSNode {id: $node_id})
                       SET n.visits = n.visits + 1,
                           n.total_value = n.total_value + $value
                       WITH n
                       MATCH (n)-[:CHILD_OF*]->(ancestor:MCTSNode)
                       SET ancestor.visits = ancestor.visits + 1,
                           ancestor.total_value = ancestor.total_value + $value""",
                    node_id=node_id, value=value,
                )
            return True

        except Exception as e:
            logger.error(f"Failed backprop update: {e}")
            return False

    # ========================================================================
    # Pattern-Domain Associations
    # ========================================================================

    def record_strategy_outcome(
        self,
        strategy: str,
        domain: str,
        success: bool,
        confidence: float = 0.0,
    ) -> bool:
        """
        Record a strategy execution outcome for a domain.

        Updates the PatternDomain node with success/failure counts.
        """
        if not self._available:
            return False

        try:
            with self.driver.session(database=self.database) as session:
                success_inc = 1 if success else 0
                failure_inc = 0 if success else 1

                session.run(
                    """MERGE (p:PatternDomain {strategy: $strategy, domain: $domain})
                       ON CREATE SET p.successes = $s_inc, p.failures = $f_inc,
                                     p.total_confidence = $confidence, p.uses = 1,
                                     p.created_at = $now
                       ON MATCH SET p.successes = p.successes + $s_inc,
                                    p.failures = p.failures + $f_inc,
                                    p.total_confidence = p.total_confidence + $confidence,
                                    p.uses = p.uses + 1,
                                    p.updated_at = $now""",
                    strategy=strategy, domain=domain,
                    s_inc=success_inc, f_inc=failure_inc,
                    confidence=confidence,
                    now=datetime.now().isoformat(),
                )
            return True

        except Exception as e:
            logger.error(f"Failed to record strategy outcome: {e}")
            return False

    def query_best_strategy(
        self,
        domain: str,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query best-performing strategies for a domain.

        Returns strategies sorted by success rate (descending).
        """
        if not self._available:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """MATCH (p:PatternDomain {domain: $domain})
                       WHERE p.uses >= 3
                       RETURN p.strategy AS strategy,
                              p.successes AS successes,
                              p.failures AS failures,
                              p.total_confidence AS total_confidence,
                              p.uses AS uses,
                              toFloat(p.successes) / (p.successes + p.failures) AS success_rate
                       ORDER BY success_rate DESC
                       LIMIT $top_n""",
                    domain=domain, top_n=top_n,
                )
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to query best strategy: {e}")
            return []

    def get_all_pattern_domains(self) -> list[dict[str, Any]]:
        """Get all pattern-domain associations."""
        if not self._available:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """MATCH (p:PatternDomain)
                       RETURN p.strategy AS strategy, p.domain AS domain,
                              p.successes AS successes, p.failures AS failures,
                              p.uses AS uses,
                              toFloat(p.successes) / (p.successes + p.failures) AS success_rate
                       ORDER BY p.domain, success_rate DESC"""
                )
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to get pattern domains: {e}")
            return []

    # ========================================================================
    # Utility
    # ========================================================================

    @property
    def available(self) -> bool:
        """Whether Neo4j is available."""
        return self._available

    def prune_low_visit_nodes(self, tree_id: str, min_visits: int = 2) -> int:
        """
        Prune nodes with very few visits from a tree.

        Useful for periodic cleanup during heartbeat signals.

        Returns:
            Number of nodes pruned
        """
        if not self._available:
            return 0

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """MATCH (n:MCTSNode {tree_id: $tree_id})
                       WHERE n.visits < $min_visits AND n.depth > 0
                       DETACH DELETE n
                       RETURN count(n) AS pruned""",
                    tree_id=tree_id, min_visits=min_visits,
                )
                record = result.single()
                return record["pruned"] if record else 0

        except Exception as e:
            logger.error(f"Failed to prune nodes: {e}")
            return 0


__all__ = [
    "MCTSStore",
    "ensure_mcts_schema",
    "MCTS_SCHEMA_STATEMENTS",
]
