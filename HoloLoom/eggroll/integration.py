"""
EGGROLL-HoloLoom Integration: The Orchestration Layer

This module bridges the gap between the chaotic evolutionary processes of Eggroll
and the structured weaving of HoloLoom. It employs advanced cybernetic regulation
and Bayesian optimization to steer the distributed learning process.
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto

# Core HoloLoom Components
from HoloLoom.shuttle.eggroll_shuttle import Shuttle, PerturbSpec
from HoloLoom.warp.eggroll_warp import Warp
from HoloLoom.yarn.eggroll_yarn import Yarn
from HoloLoom.weaving.eggroll_weave import Weave
import torch
from HoloLoom.eggroll.loom_node import LoomNode
from HoloLoom.eggroll.mirror_core import MirrorCoreAgent

# Advanced Math Subsystems
from HoloLoom.eggroll.math_crusher import (
    PIDController, 
    CalculusTools, 
    CyberneticRegulator, 
    AdvancedRegression
)
from HoloLoom.eggroll.distributed_backend import LocalBackend, RayBackend

# --- Configuration & Types ---

class OptimizationMode(Enum):
    """Defines the strategic intent of the optimization loop."""
    BALANCED = auto()
    EXPLORATION = auto()
    EXPLOITATION = auto()
    RESEARCH = auto()

@dataclass
class OptimizationConfig:
    """Hyperparameters for the cybernetic control system."""
    learning_rate: float = 0.01
    
    # PID settings
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.05
    
    # Bayesian settings
    regression_alpha: float = 1.0
    regression_beta: float = 10.0
    
    # Homeostasis
    target_variety: float = 1.0
    
    @classmethod
    def from_mode(cls, mode: OptimizationMode) -> 'OptimizationConfig':
        """Factory method for preset configurations."""
        match mode:
            case OptimizationMode.EXPLORATION:
                return cls(kp=0.8, ki=0.0, kd=0.1, target_variety=1.5)
            case OptimizationMode.EXPLOITATION:
                return cls(kp=0.2, ki=0.3, kd=0.01, target_variety=0.5)
            case OptimizationMode.RESEARCH:
                return cls(kp=0.3, ki=0.05, kd=0.05, learning_rate=0.005)
            case _:
                return cls()

@dataclass
class WorkerResult:
    """Standardized result container from a worker node."""
    worker_id: int
    reward: float
    perturb_spec: PerturbSpec
    outputs: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

# --- Strategies ---

class OptimizerStrategy(Protocol):
    """Protocol for optimization strategies to allow hot-swapping."""
    def compute_update(self, results: List[WorkerResult], current_model_dim: int) -> np.ndarray:
        ...

class CyberneticOptimizer:
    """
    An advanced optimizer that uses PID control and Cybernetic regulation
    to dynamically adjust learning rates based on system entropy and reward history.
    """
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Subsystems
        self.pid = PIDController(config.kp, config.ki, config.kd, setpoint=0.1)
        self.regulator = CyberneticRegulator(target_variety=config.target_variety)
        self.predictor = AdvancedRegression(alpha=config.regression_alpha, beta=config.regression_beta)
        
        # State
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "reward": [],
            "energy": [],
            "pid_out": []
        }
        self.smoothed_improvement = 0.0
        self.prev_avg_reward = 0.0

    def compute_update(self, results: List[WorkerResult], current_model_dim: int) -> np.ndarray:
        if not results:
            return np.zeros((current_model_dim, current_model_dim))
            
        # 1. Advantage Estimation (Z-score normalization)
        rewards = np.array([r.reward for r in results])
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8
        normalized_rewards = (rewards - mean_reward) / std_reward
        
        # 2. Predictive Control (Bayesian Lookahead)
        self._update_bayesian_priors(mean_reward)
        
        # 3. PID Tuning
        raw_improvement = mean_reward - self.prev_avg_reward
        self.prev_avg_reward = mean_reward
        
        self.smoothed_improvement = CalculusTools.exponential_moving_average(
            raw_improvement, self.smoothed_improvement, alpha=0.7
        )
        pid_adjustment = self.pid.update(self.smoothed_improvement)
        
        # 4. Cybernetic Regulation (Ashby's Law)
        # Disturbance is the variance of rewards (instability of landscape)
        system_energy = self.regulator.regulate(std_reward)
        
        # 5. Compute Adaptive Learning Rate
        # Energy acts as a gain multiplier (Homeostatic regulation)
        adaptive_lr = self.config.learning_rate * (1.0 + np.tanh(pid_adjustment)) * system_energy
        
        # Log state
        self._log_state(mean_reward, system_energy, pid_adjustment)
        
        # 6. Aggregate Gradients (Evolution Strategies)
        total_update = np.zeros((current_model_dim, current_model_dim))
        
        # Mock aggregation for simulation
        # In real world, we use ES: sum(reward * noise)
        simulated_gradient = np.random.randn(current_model_dim, current_model_dim) * adaptive_lr
        
        return simulated_gradient

    def _update_bayesian_priors(self, current_reward: float):
        """Fits the Bayesian regressor to anticipate future rewards."""
        hist_len = len(self.history["epoch"])
        if hist_len > 2:
            epochs = np.array(self.history["epoch"])
            # Feature map: [1, epoch, epoch^2]
            Phi = np.column_stack([np.ones_like(epochs), epochs, epochs**2])
            targets = np.array(self.history["reward"])
            
            self.predictor.fit_bayesian_linear(Phi, targets)
            
            # Predict next step
            phi_next = np.array([1.0, hist_len, hist_len**2])
            pred_mean, _ = self.predictor.predict(phi_next)
            
            # Adjust PID setpoint based on predicted potential
            predicted_improvement = pred_mean - current_reward
            self.pid.setpoint = max(0.05, predicted_improvement + 0.05)

    def _log_state(self, reward: float, energy: float, pid_out: float):
        idx = len(self.history["epoch"])
        self.history["epoch"].append(idx)
        self.history["reward"].append(reward)
        self.history["energy"].append(energy)
        self.history["pid_out"].append(pid_out)


# --- Main Integration Class ---

class EggrollIntegration:
    """
    EGGROLL-HoloLoom Integration Node.
    
    This class orchestrates a population of 'LoomNodes' (workers) to evolve
    a 'MirrorCoreAgent' (central model) using evolutionary strategies.
    It leverages a distributed architecture (Local or Ray).
    """
    
    def __init__(self, num_workers: int = 10, config: Optional[OptimizationConfig] = None, 
                 model_type: str = "standard", backend_type: str = "local", **model_kwargs):
        self.num_workers = num_workers
        self.config = config or OptimizationConfig()
        
        # Subsystems
        self.shuttle = Shuttle()
        self.warp = Warp()
        self.yarn = Yarn()
        self.weave = Weave() 
        
        self.mirror_core = MirrorCoreAgent(model_id="model_v1", model_type=model_type, **model_kwargs)
        self.optimizer = CyberneticOptimizer(self.config)
        
        # Distributed Backend Initialization
        if backend_type == "local":
            self.backend = LocalBackend()
        elif backend_type == "ray":
            self.backend = RayBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
            
        print(f"🔗 Initializing Distributed Backend ({backend_type}) with {num_workers} workers...")
        # Initialize workers (LoomNodes)
        self.backend.initialize(
            num_workers, 
            LoomNode, 
            model_type=model_type, 
            model_kwargs=model_kwargs,
            use_dark_trace=True # Enable Dark Trace on workers
        )
        
        # Control State
        self.forced_pattern: Optional[str] = None
        self._lock = asyncio.Lock()

    async def generate(self, prompt: str) -> str:
        """Proxies generation to the central MirrorCore agent."""
        return await self.mirror_core.generate(prompt)

    def set_forced_pattern(self, pattern: str) -> None:
        """Forces all retrieval/task generation to bias towards a specific pattern."""
        self.forced_pattern = pattern
        
    def set_optimization_mode(self, mode: OptimizationMode) -> None:
        """Swaps the optimizer configuration on the fly."""
        print(f"🔄 Switching optimization mode to: {mode.name}")
        self.config = OptimizationConfig.from_mode(mode)
        self.optimizer = CyberneticOptimizer(self.config)

    def export_dashboard_data(self, filepath: str = "HoloLoom/eggroll/dashboard/data.json") -> None:
        """Exports the Yarn lineage graph to a JSON file for the D3.js dashboard."""
        import json
        import networkx as nx
        import os
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = nx.node_link_data(self.yarn.graph)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"📊 Dashboard data exported to {filepath}")
        except Exception as e:
            print(f"Failed to export dashboard data: {e}")

    async def run_evolution_loop(self, num_epochs: int = 1) -> None:
        print(f"🚀 Starting EGGROLL Loop ({self.num_workers} nodes)...")
        
        for epoch in range(num_epochs):
            async with self._lock: 
                await self._run_single_epoch(epoch)
    
    async def _run_single_epoch(self, epoch: int) -> None:
        banner = f"Epoch {epoch+1}"
        
        # 1. Sync Central Weights to Workers
        if hasattr(self.mirror_core.custom_model, "state_dict"):
            central_weights = self.mirror_core.custom_model.state_dict()
            self.backend.scatter_broadcast(central_weights)
        
        # 2. Trigger Steps on Workers (Simulated Inputs)
        dummy_input = torch.randint(0, 1000, (1, 16))
        
        # Map Async: step(task_input, target) -> Returns (fitness, metrics, output_sample)
        self.backend.map_async("step", dummy_input, dummy_input)
        
        # 3. Gather Results
        with RichSpinner(f"{banner} | Evolution in progress...") as spinner:
            raw_results = self.backend.gather_results()
            
            worker_results = []
            fitness_scores = []
            
            for item in raw_results:
                if isinstance(item, Exception):
                    # print(f"⚠️ Worker Error: {item}")
                    continue
                    
                # Item structure: (worker_id, (fitness, metrics, output))
                worker_id, step_res = item
                
                if isinstance(step_res, Exception):
                     # print(f"⚠️ Step Error on Worker {worker_id}: {step_res}")
                     continue
                     
                fitness, metrics, output = step_res
                fitness_scores.append(fitness)
                
                result = WorkerResult(
                    worker_id=worker_id,
                    reward=fitness,
                    perturb_spec=PerturbSpec(rank=4, scale=1.0, seed_A=worker_id, seed_B=worker_id), # Mock spec
                    outputs=[output],
                    metrics=metrics
                )
                worker_results.append(result)
                
                # 4. Yarn Logging (Historian)
                node_id = f"epoch_{epoch}_worker_{worker_id}"
                
                # Warp Score (Refined using metrics!)
                refined_reward = self.warp.score("target", output, metrics=metrics)
                
                self.yarn.graph.add_node(
                    node_id, 
                    reward=refined_reward, 
                    architecture=self.mirror_core.model_type,
                    metrics=metrics,
                    parent=f"epoch_{epoch-1}_best" if epoch > 0 else "root"
                )
                if epoch > 0:
                    self.yarn.graph.add_edge(f"epoch_{epoch-1}_best", node_id)
            
            avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
            
            # Find Winner
            if worker_results:
                best_worker = max(worker_results, key=lambda x: x.reward)
                self.yarn.graph.add_node(f"epoch_{epoch}_best", reward=best_worker.reward, architecture=self.mirror_core.model_type)
                
            spinner.update(f"{banner} | Mean Fitness: {avg_fitness:.4f}")
            
            # 5. Export for Visualization
            self.export_dashboard_data()

# --- Utilities ---

class RichSpinner:
    """A context-manager for CLI feedback, simulating a loading spinner."""
    def __init__(self, message="Processing..."):
        self.message = message
        self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._running = False
        self._task = None
        
    def update(self, message: str):
        self.message = message
        
    async def _spin(self):
        idx = 0
        while self._running:
            print(f"\r{self._frames[idx % len(self._frames)]} {self.message}", end="", flush=True)
            idx += 1
            await asyncio.sleep(0.1)
            
    def __enter__(self):
        self._running = True
        print(f"\n⚡ STARTER: {self.message}")
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._running = False
        print(f"\n✅ DONE: {self.message}\n")

async def main():
    # Demo utilization
    config = OptimizationConfig.from_mode(OptimizationMode.RESEARCH)
    # Testing Spiking Network for Efficiency Rewards
    integration = EggrollIntegration(num_workers=4, config=config, model_type="spiking")
    
    # Run a few loops
    await integration.run_evolution_loop(num_epochs=3)
    
    # Switch to MoE
    print("\n🔀 Switching architecture simulation to MoE...")
    # (In real life we'd spin up new integration, here we just simulate the change in future loop)
    # integration.backend.shutdown()
    # integration = EggrollIntegration(num_workers=4, config=config, model_type="moe")
    # await integration.run_evolution_loop(num_epochs=2)

if __name__ == "__main__":
    asyncio.run(main())
