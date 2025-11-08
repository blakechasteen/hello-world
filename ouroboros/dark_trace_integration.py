#!/usr/bin/env python3
"""
Dark Trace Integration for Ouroboros
====================================

Integrates vLLM batch inference + Sparse Autoencoders for:
1. Deterministic drug interaction reasoning
2. Activation capture (residual streams)
3. SAE feature mapping (interpretable features)
4. Complete provenance tracking

Architecture:
    Drug Pair → vLLM Batch → Activations → SAE → Interpretable Features → Decision
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DarkTraceConfig:
    """Configuration for Dark Trace inference"""

    # vLLM settings
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Or local path
    tensor_parallel_size: int = 1
    max_tokens: int = 256
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0

    # SAE settings
    sae_layer: int = 16  # Middle layer (good for reasoning)
    sae_width: int = 16384  # Sparse feature dimension
    top_k_features: int = 20  # Top features to track

    # Activation capture
    capture_layers: List[int] = None  # Layers to capture (None = all)
    capture_residual: bool = True
    capture_attention: bool = False  # Optional

    # Performance
    batch_size: int = 32
    use_cache: bool = True
    cache_dir: str = "./dark_trace_cache"

    def __post_init__(self):
        if self.capture_layers is None:
            self.capture_layers = [8, 16, 24]  # Early, mid, late


# ============================================================================
# Activation Capture
# ============================================================================

@dataclass
class ActivationCapture:
    """Captured activations from a forward pass"""

    prompt_hash: str
    model_name: str
    timestamp: float

    # Residual stream activations (main pathway)
    residual_activations: Dict[int, List[float]]  # layer -> activations

    # SAE features (interpretable)
    sae_features: Dict[int, float]  # feature_id -> activation
    top_features: List[Tuple[int, float, str]]  # (id, activation, description)

    # Response
    generated_text: str
    decision: str  # "SAFE" or "BLOCKED"
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Mock vLLM Interface (Real implementation needs vLLM installed)
# ============================================================================

class MockVLLM:
    """
    Mock vLLM interface for development.

    Real implementation:
        from vllm import LLM, SamplingParams

        llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=1)
        params = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = llm.generate(prompts, params, use_tqdm=False)
    """

    def __init__(self, config: DarkTraceConfig):
        self.config = config
        self.model_name = config.model_name
        print(f"[MockVLLM] Initialized (real vLLM not installed)")
        print(f"[MockVLLM] Model: {self.model_name}")
        print(f"[MockVLLM] For production, install: pip install vllm")

    def generate(self, prompts: List[str]) -> List[Dict]:
        """Mock batch generation"""
        results = []

        for prompt in prompts:
            # Mock reasoning based on prompt content
            if "warfarin" in prompt.lower() and "aspirin" in prompt.lower():
                decision = "BLOCKED"
                reasoning = "CRITICAL interaction: Warfarin + Aspirin causes severe bleeding risk due to additive anticoagulation."
                confidence = 0.95
            elif "allergy" in prompt.lower():
                decision = "BLOCKED"
                reasoning = "CRITICAL contraindication: Patient allergic to this drug class."
                confidence = 0.98
            elif "metoprolol" in prompt.lower() and "insulin" in prompt.lower():
                decision = "BLOCKED"
                reasoning = "CRITICAL interaction: Beta-blockers mask hypoglycemia symptoms."
                confidence = 0.92
            else:
                decision = "SAFE"
                reasoning = "No significant interaction detected. Medications can be co-prescribed."
                confidence = 0.88

            results.append({
                'prompt': prompt,
                'text': reasoning,
                'decision': decision,
                'confidence': confidence
            })

        return results

    def capture_activations(self, prompt: str, layers: List[int]) -> Dict[int, List[float]]:
        """Mock activation capture"""
        # In real implementation, this hooks into model forward pass
        activations = {}

        for layer in layers:
            # Mock: random-ish activations based on prompt hash
            prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
            seed = prompt_hash + layer

            # Generate pseudo-activations (4096 hidden dim for Llama-2-7b)
            import random
            random.seed(seed)
            activations[layer] = [random.gauss(0, 1) for _ in range(128)]  # Truncated for demo

        return activations


# ============================================================================
# Sparse Autoencoder (SAE)
# ============================================================================

class MockSAE:
    """
    Mock Sparse Autoencoder.

    Real implementation uses Goodfire or custom SAE:
        from goodfire import Client
        client = Client()
        features = client.features.inspect(activations, top_k=20)
    """

    def __init__(self, config: DarkTraceConfig):
        self.config = config
        self.width = config.sae_width

        # Mock feature dictionary (real SAE learns these)
        self.feature_descriptions = {
            42: "anticoagulation_mechanism",
            108: "bleeding_risk",
            256: "drug_metabolism_cyp450",
            512: "contraindication_signal",
            1024: "allergy_cross_reactivity",
            2048: "pharmacodynamic_interaction",
            4096: "safe_combination",
            8192: "monitoring_required"
        }

        print(f"[MockSAE] Initialized with {self.width} sparse features")
        print(f"[MockSAE] Known features: {len(self.feature_descriptions)}")

    def encode(self, activations: List[float]) -> Dict[int, float]:
        """
        Encode dense activations to sparse features.

        Real SAE:
            sparse_activations = sae.encode(dense_activations)
            # Returns sparse vector (16384-dim with ~20 non-zero)
        """
        # Mock: simulate sparse activation
        import random
        random.seed(int(sum(abs(x) for x in activations[:10]) * 1000))

        # Activate 10-20 random features
        n_active = random.randint(10, 20)
        active_features = random.sample(list(self.feature_descriptions.keys()),
                                       min(n_active, len(self.feature_descriptions)))

        sparse = {}
        for feat_id in active_features:
            sparse[feat_id] = random.uniform(0.5, 3.0)  # Activation strength

        return sparse

    def get_top_features(self, sparse_activations: Dict[int, float], k: int = 20) -> List[Tuple[int, float, str]]:
        """Get top-k most active features with descriptions"""
        sorted_features = sorted(sparse_activations.items(),
                               key=lambda x: x[1],
                               reverse=True)[:k]

        return [
            (feat_id, activation, self.feature_descriptions.get(feat_id, "unknown_feature"))
            for feat_id, activation in sorted_features
        ]


# ============================================================================
# Dark Trace Engine
# ============================================================================

class DarkTraceEngine:
    """
    Main Dark Trace engine combining vLLM + SAE.

    Pipeline:
        1. Batch prompts → vLLM inference
        2. Capture activations during forward pass
        3. Encode activations → SAE features
        4. Map features → interpretable signals
        5. Make decision with complete provenance
    """

    def __init__(self, config: DarkTraceConfig):
        self.config = config
        self.llm = MockVLLM(config)
        self.sae = MockSAE(config)

        # Cache for deterministic inference
        self.cache = {}
        if config.use_cache:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self._load_cache()

        print(f"\n[DarkTrace] Engine initialized")
        print(f"[DarkTrace] Model: {config.model_name}")
        print(f"[DarkTrace] SAE layer: {config.sae_layer}")
        print(f"[DarkTrace] Cache: {'enabled' if config.use_cache else 'disabled'}")

    def _load_cache(self):
        """Load inference cache from disk"""
        cache_file = self.cache_dir / "inference_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"[DarkTrace] Loaded {len(self.cache)} cached inferences")

    def _save_cache(self):
        """Save inference cache to disk"""
        if not self.config.use_cache:
            return

        cache_file = self.cache_dir / "inference_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _hash_prompt(self, prompt: str) -> str:
        """Create deterministic hash for prompt"""
        return hashlib.md5(prompt.encode()).hexdigest()

    def build_prompt(self, drug_a: str, drug_b: str,
                     patient_allergies: List[str] = None) -> str:
        """Build drug interaction reasoning prompt"""

        allergy_text = ""
        if patient_allergies:
            allergy_text = f"\nPatient allergies: {', '.join(patient_allergies)}"

        prompt = f"""You are a clinical pharmacology expert. Analyze this drug combination:

Drug A: {drug_a}
Drug B: {drug_b}{allergy_text}

Task: Determine if this combination is SAFE or should be BLOCKED.

Consider:
1. Drug-drug interactions (pharmacodynamic, pharmacokinetic)
2. Allergy cross-reactivity
3. Additive effects (bleeding, sedation, etc.)
4. Contraindications

Response format:
Decision: [SAFE or BLOCKED]
Reasoning: [Brief clinical explanation]
Confidence: [0.0-1.0]

Analysis:"""

        return prompt

    def infer_batch(self, prompts: List[str]) -> List[ActivationCapture]:
        """
        Batch inference with activation capture.

        Steps:
            1. Check cache for deterministic responses
            2. Generate new responses for uncached prompts
            3. Capture activations during forward pass
            4. Extract SAE features from activations
            5. Return complete provenance
        """

        results = []
        uncached_prompts = []
        uncached_indices = []

        # Check cache
        for i, prompt in enumerate(prompts):
            prompt_hash = self._hash_prompt(prompt)

            if self.config.use_cache and prompt_hash in self.cache:
                # Use cached result
                cached = self.cache[prompt_hash]
                results.append(ActivationCapture(**cached))
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(i)
                results.append(None)  # Placeholder

        # Generate uncached
        if uncached_prompts:
            print(f"[DarkTrace] Generating {len(uncached_prompts)} new inferences...")

            llm_outputs = self.llm.generate(uncached_prompts)

            for prompt, output, idx in zip(uncached_prompts, llm_outputs, uncached_indices):
                # Capture activations
                activations = self.llm.capture_activations(
                    prompt,
                    self.config.capture_layers
                )

                # Extract SAE features from target layer
                sae_layer = self.config.sae_layer
                if sae_layer in activations:
                    sparse_features = self.sae.encode(activations[sae_layer])
                    top_features = self.sae.get_top_features(
                        sparse_features,
                        k=self.config.top_k_features
                    )
                else:
                    sparse_features = {}
                    top_features = []

                # Create capture
                capture = ActivationCapture(
                    prompt_hash=self._hash_prompt(prompt),
                    model_name=self.config.model_name,
                    timestamp=time.time(),
                    residual_activations={k: v[:10] for k, v in activations.items()},  # Truncate for storage
                    sae_features=sparse_features,
                    top_features=top_features,
                    generated_text=output['text'],
                    decision=output['decision'],
                    confidence=output['confidence']
                )

                results[idx] = capture

                # Cache
                if self.config.use_cache:
                    self.cache[capture.prompt_hash] = capture.to_dict()

        # Save cache
        if uncached_prompts and self.config.use_cache:
            self._save_cache()

        return results

    def check_interaction(self, drug_a: str, drug_b: str,
                         allergies: List[str] = None) -> ActivationCapture:
        """Check single drug interaction"""
        prompt = self.build_prompt(drug_a, drug_b, allergies)
        results = self.infer_batch([prompt])
        return results[0]

    def check_batch(self, drug_pairs: List[Tuple[str, str]],
                    allergies: List[str] = None) -> List[ActivationCapture]:
        """Check multiple drug interactions in parallel"""
        prompts = [self.build_prompt(a, b, allergies) for a, b in drug_pairs]
        return self.infer_batch(prompts)


# ============================================================================
# Demo
# ============================================================================

def demo_dark_trace():
    """Demonstrate Dark Trace integration"""

    print("\n" + "="*70)
    print("DARK TRACE - vLLM + SAE Integration Demo")
    print("="*70)

    # Initialize
    config = DarkTraceConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        sae_layer=16,
        batch_size=4,
        use_cache=True
    )

    engine = DarkTraceEngine(config)

    # Test cases
    test_cases = [
        ("warfarin", "aspirin", []),
        ("lisinopril", "metformin", []),
        ("amoxicillin", "clarithromycin", ["penicillin"]),
        ("metoprolol", "insulin", [])
    ]

    print(f"\n[1/3] Testing {len(test_cases)} drug pairs (batch inference)...")

    # Batch check
    drug_pairs = [(a, b) for a, b, _ in test_cases]
    results = engine.check_batch(drug_pairs)

    print(f"\n[2/3] Results:\n")

    for (drug_a, drug_b, allergies), capture in zip(test_cases, results):
        print(f"  {drug_a.upper()} + {drug_b.upper()}")
        print(f"    Decision: {capture.decision}")
        print(f"    Confidence: {capture.confidence:.2f}")
        print(f"    Reasoning: {capture.generated_text[:80]}...")

        # Show top SAE features
        if capture.top_features:
            print(f"    Top SAE features:")
            for feat_id, activation, description in capture.top_features[:3]:
                print(f"      - Feature {feat_id} ({description}): {activation:.2f}")

        print()

    print(f"[3/3] Activation Analysis:\n")

    # Analyze activation patterns
    for (drug_a, drug_b, _), capture in zip(test_cases, results):
        print(f"  {drug_a} + {drug_b}:")
        print(f"    Captured layers: {list(capture.residual_activations.keys())}")
        print(f"    Active SAE features: {len(capture.sae_features)}")
        print(f"    Prompt hash: {capture.prompt_hash[:16]}...")
        print()

    print("="*70)
    print("[OK] Dark Trace demo complete")
    print("="*70)

    print(f"\nCache statistics:")
    print(f"  Cached inferences: {len(engine.cache)}")
    print(f"  Cache directory: {engine.cache_dir}")

    print(f"\nProduction deployment:")
    print(f"  1. Install vLLM: pip install vllm")
    print(f"  2. Download Llama-2-7b-hf model")
    print(f"  3. Train/load SAE for layer {config.sae_layer}")
    print(f"  4. Replace MockVLLM with real vLLM.LLM")
    print(f"  5. Replace MockSAE with Goodfire or custom SAE")


if __name__ == "__main__":
    demo_dark_trace()
