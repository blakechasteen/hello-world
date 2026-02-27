from typing import List, Dict, Any

class Weave:
    """
    Weave (Data + Reward Fabric)
    
    Responsibilities:
    * Generates tasks, datasets, and evaluation prompts.
    * Aggregates Warp scores into composite rewards.
    * Applies safety, stability, and novelty shaping.
    """
    
    def __init__(self):
        # Connect to Reflection Buffer
        try:
            from hololoom.core.reflection.buffer import ReflectionBuffer
            self.reflection = ReflectionBuffer()
            self.has_reflection = True
        except ImportError:
            self.has_reflection = False
            print("Warning: ReflectionBuffer not available, using dummy tasks.")

    def generate_tasks(self, model_id: str) -> List[Any]:
        """
        Generate a batch of tasks for evaluation.
        """
        if self.has_reflection:
            try:
                # Sample recent episodes or failures from reflection buffer
                # Method signature is get_recent_episodes(self, n: int = 10)
                episodes = self.reflection.get_recent_episodes(n=10)
                if episodes:
                    return [ep['spacetime'].query_text for ep in episodes if 'spacetime' in ep]
            except Exception as e:
                print(f"Failed to fetch from ReflectionBuffer: {e}")
        
        # Fallback to dummy tasks
        return ["Task 1", "Task 2", "Task 3"]

    def compute_reward(self, warp_scores: Dict[str, float], safety_scores: Dict[str, float]) -> float:
        """
        Compute composite reward from various signals.
        
        R_total = α1*R_task + α2*R_safety + α3*R_novelty + α4*R_stability + α5*R_complexity
        """
        # Coefficients (could be configurable)
        alpha_task = 1.0
        alpha_safety = 1.0
        
        r_task = warp_scores.get('task', 0.0)
        r_safety = safety_scores.get('safety', 0.0)
        
        # Simplified composite reward
        return alpha_task * r_task + alpha_safety * r_safety
