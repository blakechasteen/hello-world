"""
vLLM Integration for EGGROLL
============================

Provides high-performance Multi-LoRA inference using vLLM.
This enables the "Evolutionary" part of EGGROLL to scale by evaluating
dozens of perturbed models (adapters) in parallel batches.

Requirements:
    pip install vllm

Usage:
    engine = VLLMInferenceEngine(model_id="meta-llama/Llama-2-7b-hf")
    
    # Evaluate multiple perturbations in one batch
    results = await engine.generate_batch([
        ("Task 1", "adapters/epoch1/worker1", "worker1"),
        ("Task 1", "adapters/epoch1/worker2", "worker2"),
    ])
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. High-performance inference unavailable.")
    # Mock classes for type hinting if vLLM is missing
    class AsyncLLMEngine: pass
    class AsyncEngineArgs: pass
    class SamplingParams: pass
    class LoRARequest: pass


@dataclass
class InferenceRequest:
    prompt: str
    adapter_path: str | None = None
    adapter_name: str | None = None
    request_id: str | None = None


class VLLMInferenceEngine:
    """
    Manages a vLLM engine for high-throughput Multi-LoRA inference.
    """

    def __init__(self, model_id: str, max_loras: int = 16, gpu_memory_utilization: float = 0.8):
        self.model_id = model_id
        self.engine: AsyncLLMEngine | None = None
        self.max_loras = max_loras
        self.gpu_memory_utilization = gpu_memory_utilization

        if VLLM_AVAILABLE:
            self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the vLLM Async Engine with LoRA support enabled."""
        logger.info(f"Initializing vLLM Engine with model: {self.model_id}")

        engine_args = AsyncEngineArgs(
            model=self.model_id,
            enable_lora=True,
            max_loras=self.max_loras,
            max_lora_rank=64, # Allow sufficient rank for perturbations
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True
        )

        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM Engine: {e}")
            self.engine = None

    async def generate_batch(
        self,
        requests: list[InferenceRequest],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Generate responses for a batch of requests, each potentially using a different LoRA adapter.
        """
        if not self.engine:
            if not VLLM_AVAILABLE:
                return [f"[Simulated vLLM] Response to: {r.prompt} (Adapter: {r.adapter_name})" for r in requests]
            raise RuntimeError("vLLM Engine not initialized")

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens
        )

        tasks = []
        for i, req in enumerate(requests):
            # Unique request ID is required by vLLM
            request_id = req.request_id or f"req_{i}_{hash(req.prompt)}"

            # Construct LoRA request if adapter is specified
            lora_request = None
            if req.adapter_path and req.adapter_name:
                lora_request = LoRARequest(
                    lora_name=req.adapter_name,
                    lora_int_id=i + 1, # Unique integer ID for this adapter in the batch
                    lora_path=req.adapter_path
                )

            # Create generation task
            tasks.append(self._generate_single(
                request_id,
                req.prompt,
                sampling_params,
                lora_request
            ))

        # Run all generations in parallel (vLLM handles the batching internally)
        results = await asyncio.gather(*tasks)
        return results

    async def _generate_single(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None
    ) -> str:
        """Helper to await a single generation request."""
        try:
            results_generator = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                lora_request=lora_request
            )

            # Stream the results, we only care about the final output
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            if final_output:
                return final_output.outputs[0].text
            return ""

        except Exception as e:
            logger.error(f"Generation failed for {request_id}: {e}")
            return f"Error: {e}"

# Example Integration Logic for LoomNode
class VLLMLoomNode:
    """
    A LoomNode that delegates to a shared vLLM engine.
    """
    def __init__(self, engine: VLLMInferenceEngine, worker_id: int):
        self.engine = engine
        self.worker_id = worker_id

    async def evaluate_perturbation(self, prompt: str, adapter_path: str) -> str:
        """
        Evaluate a specific perturbation (saved as an adapter) on a task.
        """
        req = InferenceRequest(
            prompt=prompt,
            adapter_path=adapter_path,
            adapter_name=f"worker_{self.worker_id}_adapter"
        )

        # In a real scenario, we would batch these at the Integration level,
        # but here we show the node-level API.
        results = await self.engine.generate_batch([req])
        return results[0]

async def main():
    """Demonstration of vLLM Worker Logic"""
    print("Initializing vLLM Integration Prototype...")

    # Initialize Engine (Simulated if vLLM not installed)
    engine = VLLMInferenceEngine(model_id="gpt2")

    # Create a batch of requests simulating different workers/perturbations
    requests = [
        InferenceRequest(
            prompt="Explain quantum mechanics.",
            adapter_path="/path/to/adapter_v1",
            adapter_name="worker_1"
        ),
        InferenceRequest(
            prompt="Explain quantum mechanics.",
            adapter_path="/path/to/adapter_v2",
            adapter_name="worker_2"
        ),
        InferenceRequest(
            prompt="Explain quantum mechanics.",
            adapter_path="/path/to/adapter_v3",
            adapter_name="worker_3"
        )
    ]

    print(f"\nSubmitting batch of {len(requests)} requests...")
    results = await engine.generate_batch(requests)

    print("\nResults:")
    for i, res in enumerate(results):
        print(f"[{requests[i].adapter_name}]: {res}")

if __name__ == "__main__":
    asyncio.run(main())
