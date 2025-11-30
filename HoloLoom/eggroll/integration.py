import asyncio
import numpy as np
from typing import List, Dict, Any

from HoloLoom.shuttle.eggroll_shuttle import Shuttle, PerturbSpec
from HoloLoom.warp.eggroll_warp import Warp
from HoloLoom.yarn.eggroll_yarn import Yarn
from HoloLoom.weaving.eggroll_weave import Weave
from HoloLoom.eggroll.nodes import LoomNode
from HoloLoom.eggroll.mirror_core import MirrorCoreAgent

class EggrollIntegration:
    """
    EGGROLL-HoloLoom Integration
    
    Orchestrates the distributed evolutionary training loop.
    """
    
    def __init__(self, num_workers: int = 10):
        self.num_workers = num_workers
        self.shuttle = Shuttle()
        self.warp = Warp()
        self.yarn = Yarn()
        self.weave = Weave()
        self.mirror_core = MirrorCoreAgent("model_v1")
        
        # Simulate a cluster of workers
        self.workers = [LoomNode() for _ in range(num_workers)]
        
        # Control
        self.forced_pattern = None

    async def generate(self, prompt: str) -> str:
        """
        Generate a response using the MirrorCore agent.
        """
        return await self.mirror_core.generate(prompt)

    def set_forced_pattern(self, pattern: str):
        """Force a specific pattern card."""
        self.forced_pattern = pattern
        
    def get_lineage_data(self) -> Dict[str, Any]:
        """
        Get lineage graph data for visualization.
        """
        import networkx as nx
        # Convert to node-link format compatible with D3/Vis.js
        data = nx.node_link_data(self.yarn.graph)
        return data

    async def run_evolution_loop(self, num_epochs: int = 1):
        """
        Run the high-level evolution loop.
        """
        print("Initializing EGGROLL Cluster...")
        
        for epoch in range(num_epochs):
            print(f"--- Epoch {epoch+1} ---")
            
            # 1. Weave generates tasks
            # Select pattern card using MCTS from Shuttle OR use forced pattern
            if self.forced_pattern:
                selected_pattern = self.forced_pattern
                print(f"Selected Pattern Card (Forced): {selected_pattern}")
            else:
                selected_pattern = self.shuttle.select_pattern()
                print(f"Selected Pattern Card (MCTS): {selected_pattern}")
            
            tasks = self.weave.generate_tasks(self.mirror_core.model_id)
            
            # 2. Shuttle proposes perturbations
            # Pass Yarn for intelligent graph-based targeting
            perturbations = self.shuttle.propose_perturbations(
                self.mirror_core.model_id, 
                self.num_workers,
                yarn=self.yarn
            )
            
            # 3. Federated Nodes apply perturbations and evaluate (Simulated Async)
            # Use Spinner for UX
            with Spinner(f"Running {self.num_workers} workers on '{selected_pattern}'..."):
                worker_results = []
                for i, worker in enumerate(self.workers):
                    spec = perturbations[i]
                    
                    # Apply perturbation (conceptually)
                    perturbed_model = worker.apply_perturbation(self.mirror_core, spec)
                    
                    # Evaluate
                    outputs = await worker.evaluate(perturbed_model, tasks)
                    
                    # 4. Warp evaluates outputs
                    # Pass the selected pattern to Warp for dynamic weighting
                    reward = self.warp.score("target", outputs, pattern_name=selected_pattern)
                    
                    # 5. Yarn logs lineage
                    embedding = self.warp.embed(outputs)
                    self.yarn.log(spec, reward, embedding, {
                        "worker_id": i,
                        "model_id": f"{self.mirror_core.model_id}_epoch{epoch}_worker{i}",
                        "parent_model_id": self.mirror_core.model_id,
                        "pattern_card": selected_pattern
                    })
                    
                    worker_results.append({
                        "worker_id": i,
                        "reward": reward,
                        "perturb_spec": spec,
                        "outputs": outputs
                    })
            
            # 6. Shuttle updates bandit
            self.shuttle.update_bandit(worker_results)
            
            # 7. Parameter Update
            update_matrix = self.aggregate_updates(worker_results, pattern=selected_pattern)
            self.mirror_core.update_weights(update_matrix)
            
            print(f"Epoch {epoch+1} complete. Average Reward: {np.mean([r['reward'] for r in worker_results]):.4f}")

    def aggregate_updates(self, results: List[Dict[str, Any]], pattern: str = "balanced") -> np.ndarray:
        """
        Aggregate updates from worker results using Evolution Strategies (ES).
        """
        if not results:
            return np.zeros((100, 100))
            
        # 1. Normalize rewards (advantage estimation)
        rewards = np.array([r['reward'] for r in results])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        normalized_rewards = (rewards - mean_reward) / std_reward
        
        # 2. Reconstruct and aggregate perturbations
        dim = self.mirror_core.dim
        total_update = np.zeros((dim, dim))
        
        for i, res in enumerate(results):
            spec = res['perturb_spec']
            advantage = normalized_rewards[i]
            
            rng_A = np.random.default_rng(spec.seed_A)
            A = rng_A.standard_normal((dim, spec.rank)) 
            
            rng_B = np.random.default_rng(spec.seed_B)
            B = rng_B.standard_normal((dim, spec.rank))
            
            perturbation = A @ B.T
            total_update += advantage * perturbation
            
        # 3. Scale by learning rate and population size
        # --- Advanced Math: Calculus (PID Control & Signal Processing) ---
        from HoloLoom.eggroll.math_crusher import PIDController, CalculusTools, CyberneticRegulator, AdvancedRegression
        
        if not hasattr(self, 'pid'):
            # Kp=0.5, Ki=0.1, Kd=0.05
            self.pid = PIDController(kp=0.5, ki=0.1, kd=0.05, setpoint=0.1)
            self.regulator = CyberneticRegulator() # Homeostasis
            self.predictor = AdvancedRegression(alpha=1.0, beta=10.0) # Bayesian Regression
            self.prev_avg_reward = 0.0
            self.smoothed_improvement = 0.0
            self.epoch_history = []
            self.reward_history = []
            
        # Dynamic PID Tuning based on Pattern
        if pattern == "quick_answer":
            self.pid.kp = 0.8 # Aggressive
            self.pid.ki = 0.2
        elif pattern == "research_pipeline":
            self.pid.kp = 0.3 # Conservative
            self.pid.ki = 0.05
        else:
            self.pid.kp = 0.5 # Balanced
            self.pid.ki = 0.1
            
        current_avg_reward = mean_reward
        raw_improvement = current_avg_reward - self.prev_avg_reward
        self.prev_avg_reward = current_avg_reward
        
        # Update History for Regression
        self.epoch_history.append(len(self.epoch_history))
        self.reward_history.append(current_avg_reward)
        
        # --- Advanced Math: Bayesian Regression (Predictive Control) ---
        # Fit model to predict reward trend
        if len(self.epoch_history) > 2:
            # Design matrix: [1, epoch, epoch^2] (Quadratic trend)
            epochs = np.array(self.epoch_history)
            Phi = np.column_stack([np.ones_like(epochs), epochs, epochs**2])
            targets = np.array(self.reward_history)
            
            self.predictor.fit_bayesian_linear(Phi, targets)
            
            # Predict next epoch reward
            next_epoch = len(self.epoch_history)
            phi_next = np.array([1.0, next_epoch, next_epoch**2])
            pred_mean, pred_var = self.predictor.predict(phi_next)
            
            # Adaptive Setpoint: We aim for the predicted improvement + a small push
            # If prediction is high, we set a high bar. If low, we relax.
            predicted_improvement = pred_mean - current_avg_reward
            self.pid.setpoint = max(0.05, predicted_improvement + 0.05)
            
            print(f"[Bayesian] Next Reward: {pred_mean:.4f} +/- {np.sqrt(pred_var):.4f} | Setpoint: {self.pid.setpoint:.4f}")
        
        # Smooth the improvement signal (Low-pass filter)
        self.smoothed_improvement = CalculusTools.exponential_moving_average(
            raw_improvement, self.smoothed_improvement, alpha=0.7
        )
        
        # PID Output adjusts the base learning rate
        pid_adjustment = self.pid.update(self.smoothed_improvement)
        
        # --- Advanced Math: Cybernetics (Ashby's Law) ---
        # Regulate system energy based on "Disturbance" (Variance of rewards)
        # High variance = High disturbance -> Need more energy/capacity
        disturbance = std_reward # Use reward std dev as proxy for environmental variety
        system_energy = self.regulator.regulate(disturbance)
        
        # Base LR = 0.01, modulated by PID and System Energy
        # Energy acts as a gain multiplier (Homeostatic regulation)
        learning_rate = 0.01 * (1.0 + np.tanh(pid_adjustment)) * system_energy
        
        print(f"[Cybernetics] Energy: {system_energy:.4f} | PID Adj: {pid_adjustment:.4f} | Smoothed Imp: {self.smoothed_improvement:.4f}")
        
        sigma = results[0]['perturb_spec'].scale
        
        final_update = (learning_rate / (len(results) * sigma)) * total_update
        return final_update

class Spinner:
    """Simple CLI Spinner"""
    def __init__(self, message="Processing..."):
        self.message = message
        self.running = False
        self.task = None
        self._chars = "|/-\\"
        
    def __enter__(self):
        print(self.message, end=" ", flush=True)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        print("Done!")

async def main():
    integration = EggrollIntegration(num_workers=5)
    await integration.run_evolution_loop(num_epochs=3)

if __name__ == "__main__":
    asyncio.run(main())
