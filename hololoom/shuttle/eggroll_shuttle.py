import numpy as np
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class PerturbSpec:
    seed_A: int
    seed_B: int
    rank: int
    scale: float

class ThompsonSampler:
    """
    Thompson Sampling for selecting continuous hyperparameters (Rank, Scale).
    Discretized into 'arms' for simplicity.
    """
    def __init__(self):
        # Arms: (rank, scale) tuples
        self.arms = [
            (8, 0.01), (8, 0.05), (8, 0.1),
            (16, 0.01), (16, 0.05), (16, 0.1),
            (32, 0.01), (32, 0.05), (32, 0.1)
        ]
        # Beta distribution parameters for each arm (alpha, beta)
        # We assume rewards are normalized to [0, 1]
        self.stats = {arm: {'alpha': 1.0, 'beta': 1.0} for arm in self.arms}

    def select_arm(self) -> Tuple[int, float]:
        """Sample from Beta distributions and pick the best arm."""
        samples = {}
        for arm, params in self.stats.items():
            samples[arm] = np.random.beta(params['alpha'], params['beta'])
        return max(samples, key=samples.get)

    def update(self, arm: Tuple[int, float], reward: float):
        """Update Beta distribution for the arm."""
        # Clip reward to [0, 1] just in case
        r = max(0.0, min(1.0, reward))
        self.stats[arm]['alpha'] += r
        self.stats[arm]['beta'] += (1.0 - r)

class MCTSNode:
    """Node for Monte Carlo Tree Search over Pattern Cards."""
    def __init__(self, pattern: str, parent=None):
        self.pattern = pattern
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_patterns = [
            "balanced", "quality_first", "quick_answer", "research_pipeline"
        ]

    def uct_select_child(self, exploration_weight=1.41):
        """Select child with highest UCB1 score."""
        log_n = math.log(self.visits)
        
        def ucb(node):
            return (node.value / node.visits) + exploration_weight * math.sqrt(log_n / node.visits)
            
        return max(self.children.values(), key=ucb)

    def expand(self):
        """Expand a random untried pattern."""
        pattern = self.untried_patterns.pop()
        child = MCTSNode(pattern, parent=self)
        self.children[pattern] = child
        return child

    def update(self, reward: float):
        """Backpropagate reward."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)

class MCTS:
    """MCTS Orchestrator for Pattern Selection."""
    def __init__(self):
        self.root = MCTSNode("root")
        self.current_node = self.root

    def select_next_pattern(self) -> str:
        """
        Select the next pattern to try using MCTS.
        Performs 100 simulations to decide.
        """
        # 1. Selection & Expansion (Simulation)
        for _ in range(50):
            node = self.root
            
            # Select
            while not node.untried_patterns and node.children:
                node = node.uct_select_child()
            
            # Expand
            if node.untried_patterns:
                node = node.expand()
            
            # Rollout (Simulation) - Random reward for simulation
            # In a real system, we might use a heuristic or value network
            sim_reward = random.random() 
            
            # Backpropagate
            node.update(sim_reward)
            
        # 2. Real Selection (Exploitation)
        # Choose child with most visits
        if not self.root.children:
             # Should not happen if we ran simulations, but fallback
             return random.choice(["balanced", "quality_first"])
             
        best_child = max(self.root.children.values(), key=lambda n: n.visits)
        
        # For the actual run, we don't move the root down yet (unless we want to commit to a path).
        # We just want to know which pattern looks best NOW.
        # But to support "Curriculum", we might want to track history.
        # For now, let's just return the pattern name.
        return best_child.pattern

    def update(self, pattern: str, reward: float):
        """Update the tree with real feedback."""
        if pattern in self.root.children:
            self.root.children[pattern].update(reward)
        else:
            # If we somehow selected a pattern not in children (e.g. random fallback)
            pass

class Shuttle:
    """
    Shuttle (Exploration Orchestrator)
    
    Responsibilities:
    * Uses Thompson Sampling to select perturbation hyperparameters (Rank, Scale).
    * Uses MCTS to select high-level strategies (Pattern Cards).
    * Allocates per-worker seeds.
    """
    
    def __init__(self):
        self.ts_sampler = ThompsonSampler()
        self.mcts = MCTS()
        from HoloLoom.eggroll.math_crusher import DifferentialEvolution
        self.de = DifferentialEvolution()
        self.last_selected_arm = None
        self.last_selected_pattern = None

    def select_pattern(self) -> str:
        """Select a pattern card using MCTS."""
        self.last_selected_pattern = self.mcts.select_next_pattern()
        return self.last_selected_pattern

    def propose_perturbations(self, model_id: str, num_workers: int, yarn: Any = None) -> List[PerturbSpec]:
        """
        Propose a batch of perturbations, potentially guided by Lineage Analysis.
        """
        specs = []
        
        # Select hyperparameters for this batch using Thompson Sampling
        rank, scale = self.ts_sampler.select_arm()
        self.last_selected_arm = (rank, scale)
        
        # Intelligent Targeting: If Yarn is provided, analyze the graph
        target_model_id = model_id
        
        if yarn:
            try:
                # 1. Find most central node (Exploitation)
                central_node = yarn.analyze_lineage()
                
                # 2. Find "Stagnant" nodes (Exploration)
                # Nodes with low reward but high centrality? Or just leaf nodes?
                # For now, we mix strategies: 80% on current model, 20% on central model
                if central_node != "N/A" and central_node != model_id:
                    # Occasional "Backtracking" to a strong ancestor
                    if np.random.random() < 0.2:
                        target_model_id = central_node
                        # Increase scale for backtracking to break out of local optima
                        scale = 0.05
                        print(f"[Shuttle] Backtracking to central ancestor: {central_node}")
            except Exception as e:
                print(f"[Shuttle] Lineage analysis failed: {e}")

        for i in range(num_workers):
            # Differential Evolution: Mutate the scale
            # We want diversity in step sizes
            worker_scale = self.de.mutate(scale)
            
            # Create Spec
            spec = PerturbSpec(
                seed_A=np.random.randint(0, 1000000),
                seed_B=np.random.randint(0, 1000000),
                rank=rank,
                scale=worker_scale
            )
            specs.append(spec)
            
        return specs

    def update_bandit(self, worker_results: List[Dict[str, Any]]):
        """
        Update policies based on worker results.
        """
        if not worker_results:
            return

        # Average reward for this batch
        avg_reward = np.mean([r['reward'] for r in worker_results])
        
        # Update Thompson Sampler
        if self.last_selected_arm:
            self.ts_sampler.update(self.last_selected_arm, avg_reward)
            
        # Update MCTS
        if self.last_selected_pattern:
            self.mcts.update(self.last_selected_pattern, avg_reward)
