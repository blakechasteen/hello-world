import numpy as np
from typing import Any, List, Dict, Optional

class ContextPacker:
    """
    Context Packing Utility
    
    Optimizes context window usage by packing multiple short examples
    or truncating long ones intelligently.
    """
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        
    def pack(self, items: List[str]) -> str:
        """
        Pack multiple text items into a single context string.
        """
        packed = ""
        current_len = 0
        
        for item in items:
            # Simple character approximation (4 chars ~= 1 token)
            item_len = len(item) // 4
            if current_len + item_len > self.max_tokens:
                break
                
            packed += item + "\n\n"
            current_len += item_len
            
        return packed.strip()

class Warp:
    """
    Warp (Evaluator & Vector Space Manager)
    
    Responsibilities:
    * Embeds model outputs using Sentence Transformers.
    * Computes cosine similarity and semantic distance.
    * Manages context packing for efficient evaluation.
    * Generates structured reward signals.
    """
    
    def __init__(self):
        # Initialize Symbolic Evaluators
        try:
            from HoloLoom.alignment.safety_guardrails import create_guardrails
            self.guardrails = create_guardrails()
            self.has_safety = True
        except ImportError:
            self.has_safety = False
            print("Warning: SafetyGuardrails not available.")

        try:
            from HoloLoom.resonance.shed import ResonanceShed
            self.resonance = ResonanceShed()
            self.has_resonance = True
        except ImportError:
            self.has_resonance = False
            print("Warning: ResonanceShed not available.")
            
        # Initialize Vector Embedding Model
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for local embedding
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_embedder = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Using random embeddings.")
            self.has_embedder = False
            
        self.packer = ContextPacker()

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text into a vector space.
        """
        if self.has_embedder:
            try:
                # Returns a numpy array (384-dim for MiniLM)
                return self.embedder.encode(text)
            except Exception as e:
                print(f"Embedding failed: {e}")
                return np.random.randn(384)
        else:
            return np.random.randn(384)

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def score(self, target: Any, output_text: str, pattern_name: str = "balanced") -> float:
        """
        Compute a composite score with dynamic weighting based on the active Pattern Card.
        """
        # Define Weights based on Pattern
        # Default: Balanced
        weights = {
            "semantic": 1.0, "safety": 1.0, "coherence": 1.0, 
            "hyperbolic": 1.0, "novelty": 1.0, "info": 1.0, "topology": 1.0
        }
        
        if pattern_name == "quality_first":
            weights = {
                "semantic": 2.0, "safety": 2.0, "coherence": 1.5, 
                "hyperbolic": 1.0, "novelty": 0.5, "info": 1.0, "topology": 1.0
            }
        elif pattern_name == "research_pipeline":
            weights = {
                "semantic": 1.0, "safety": 1.0, "coherence": 1.0, 
                "hyperbolic": 2.0, "novelty": 0.5, "info": 2.0, "topology": 1.5
            }
        elif pattern_name == "quick_answer":
            weights = {
                "semantic": 1.5, "safety": 1.0, "coherence": 1.0, 
                "hyperbolic": 0.5, "novelty": 0.2, "info": 0.5, "topology": 0.5
            }
            
        scores = {}
        
        # 1. Semantic Similarity (Linear Algebra)
        if isinstance(target, str) and self.has_embedder:
            target_vec = self.embed(target)
            output_vec = self.embed(output_text)
            
            # Hyperbolic
            from HoloLoom.eggroll.math_crusher import HyperbolicGeometry
            hyp_target = HyperbolicGeometry.exp_map(target_vec)
            hyp_output = HyperbolicGeometry.exp_map(output_vec)
            hyp_dist = HyperbolicGeometry.poincare_distance(hyp_target, hyp_output)
            scores["hyperbolic"] = np.exp(-hyp_dist)
            
            # Cosine
            similarity = self.cosine_similarity(target_vec, output_vec)
            scores["semantic"] = (similarity + 1) / 2
            
            # Information Theory
            from HoloLoom.eggroll.math_crusher import InformationTheory, StatisticalMeasures
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            p_output = softmax(output_vec)
            p_joint = np.outer(p_output, softmax(target_vec))
            p_joint /= p_joint.sum()
            mi = InformationTheory.mutual_information(p_joint)
            scores["info"] = min(1.0, mi / 6.0)
            
            # Novelty (KL)
            p_uniform = np.ones_like(p_output) / len(p_output)
            kl = StatisticalMeasures.kl_divergence(p_output, p_uniform)
            scores["novelty"] = min(1.0, kl / 6.0)
            
            # Topology
            from HoloLoom.eggroll.math_crusher import TopologicalFeatures
            chunks = [c for c in output_text.split('.') if len(c) > 5]
            if len(chunks) > 3:
                chunk_vecs = np.array([self.embed(c) for c in chunks])
                betti_0 = TopologicalFeatures.compute_betti_0(chunk_vecs, epsilon=0.8)
                scores["topology"] = 1.0 / max(1, betti_0)
            else:
                scores["topology"] = 0.5
        else:
            # Fallbacks
            scores["semantic"] = 0.5
            scores["hyperbolic"] = 0.5
            scores["info"] = 0.5
            scores["novelty"] = 0.5
            scores["topology"] = 0.5
        
        # 2. Safety
        if self.has_safety:
            try:
                from HoloLoom.alignment.safety_guardrails import ActionCategory
                decision = self.guardrails.evaluate_action(
                    action="eggroll_evaluation", category=ActionCategory.QUERY, text_input=output_text
                )
                if not decision.allowed:
                    scores["safety"] = 0.0
                else:
                    risk_scores = {"safe": 1.0, "low": 0.9, "medium": 0.7, "high": 0.4, "critical": 0.0}
                    scores["safety"] = risk_scores.get(decision.risk_level.value, 0.5)
            except:
                scores["safety"] = 0.5
        else:
            scores["safety"] = 0.5
        
        # 3. Resonance
        if self.has_resonance:
            try:
                scores["coherence"] = self.resonance.measure_coherence(output_text)
            except:
                scores["coherence"] = 0.5
        else:
            scores["coherence"] = 0.5

        # Weighted Average
        total_score = 0.0
        total_weight = 0.0
        
        for key, val in scores.items():
            w = weights.get(key, 1.0)
            total_score += val * w
            total_weight += w
            
        if total_weight == 0: return 0.0
        return float(total_score / total_weight)
