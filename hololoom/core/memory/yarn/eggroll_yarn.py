from typing import Any

import networkx as nx
import numpy as np


class Yarn:
    """
    Yarn (Lineage + Memory)
    
    Responsibilities:
    * Logs ancestry, perturbation seeds, rewards, embeddings.
    * Generates lineage graphs.
    * Stores reversible perturbation traces.
    * Analyzes lineage centrality.
    """

    def __init__(self):
        self.history = []
        self.graph = nx.DiGraph()

        # Connect to Unified Memory
        try:
            from hololoom.memory.unified import UnifiedMemory
            self.memory = UnifiedMemory()
            self.has_memory = True
        except ImportError:
            self.has_memory = False
            print("Warning: UnifiedMemory not available, running in local mode.")

    def log(self, perturb_spec: Any, reward: float, embedding: np.ndarray, metadata: dict[str, Any]):
        """
        Log a step in the lineage.
        """
        entry = {
            "perturb_spec": perturb_spec,
            "reward": reward,
            "embedding": embedding,
            "metadata": metadata
        }
        self.history.append(entry)

        # Update Graph
        model_id = metadata.get("model_id")
        parent_id = metadata.get("parent_model_id")

        if model_id:
            self.graph.add_node(model_id, reward=reward)
            if parent_id and self.graph.has_node(parent_id):
                self.graph.add_edge(parent_id, model_id)

            # Persist to Unified Memory
            if self.has_memory:
                try:
                    self.memory.store(
                        text=f"Lineage: {model_id} <- {parent_id} | Reward: {reward}",
                        metadata=metadata
                    )
                except Exception as e:
                    print(f"Failed to store lineage: {e}")

    def analyze_lineage(self) -> str:
        """
        Analyze the lineage graph using Graph Theory.
        Returns the ID of the most central/influential model.
        """
        try:
            from hololoom.eggroll.math_crusher import GraphTheory

            if len(self.graph) < 2:
                return "N/A"

            # Convert to adjacency matrix
            nodes = list(self.graph.nodes())
            adj = nx.to_numpy_array(self.graph, nodelist=nodes)

            # Compute Eigenvector Centrality
            centrality = GraphTheory.eigenvector_centrality(adj)

            # Find max centrality
            if len(centrality) > 0:
                idx = np.argmax(centrality)
                return nodes[idx]
            return "N/A"
        except Exception as e:
            print(f"Lineage analysis failed: {e}")
            return "Error"

    def lineage(self, model_id: str) -> nx.DiGraph:
        """
        Return the lineage graph for a given model.
        """
        return self.graph
