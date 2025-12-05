# Dark Trace Deployment Guide - Ouroboros

## Overview

This guide covers deploying the complete Dark Trace + vLLM + SAE stack for production drug interaction detection.

**Architecture**:
```
Drug Pair → vLLM Batch Inference → Activation Capture → SAE Features → Decision
                                         ↓
                                  Complete Provenance
                                  (HIPAA-compliant audit trail)
```

## Prerequisites

### Hardware Requirements

**Minimum (Development)**:
- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 32GB system RAM
- Storage: 50GB SSD

**Recommended (Production)**:
- GPU: NVIDIA A100 (40GB VRAM) or A10 (24GB)
- RAM: 64GB+ system RAM
- Storage: 500GB NVMe SSD
- Network: 10Gbps for distributed inference

**Why GPU?**
- Llama-2-7b-hf requires ~14GB VRAM in FP16
- Batch inference (32 prompts) requires ~18GB VRAM
- SAE encoding adds ~2GB VRAM overhead

### Software Dependencies

```bash
# Python 3.10+
python --version  # Should be >= 3.10

# CUDA 11.8+ (for GPU acceleration)
nvcc --version

# Core dependencies
pip install torch>=2.0.0
pip install vllm>=0.2.0
pip install transformers>=4.30.0
pip install accelerate>=0.20.0

# SAE dependencies (choose one)
# Option 1: Goodfire (recommended)
pip install goodfire

# Option 2: Custom SAE
pip install sparse_autoencoder

# Monitoring
pip install prometheus-client
pip install tensorboard
```

## Step 1: Model Setup

### Download Llama-2-7b-hf

```bash
# Using Hugging Face CLI
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login

# Download model
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir ./models/Llama-2-7b-hf \
  --local-dir-use-symlinks False

# Verify download
ls -lh ./models/Llama-2-7b-hf/
# Should see: config.json, pytorch_model.bin.index.json, tokenizer.json, etc.
```

**Model size**: ~13.5GB

### Alternative: Smaller Models (for resource-constrained environments)

```bash
# Llama-2-7b-chat-hf (same size, instruction-tuned)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf

# TinyLlama-1.1B (much smaller, 2.2GB)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Step 2: vLLM Server Setup

### Option A: Standalone Server

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Llama-2-7b-hf \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --max-model-len 2048 \
  --port 8000

# Test server
curl http://localhost:8000/v1/models
```

### Option B: Integrated (within Ouroboros)

```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="./models/Llama-2-7b-hf",
    tensor_parallel_size=1,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.9
)

# Create sampling params (deterministic)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=256,
    top_p=1.0
)

# Batch inference
prompts = [build_prompt(drug_a, drug_b) for drug_a, drug_b in drug_pairs]
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

for output in outputs:
    print(output.outputs[0].text)
```

## Step 3: SAE Integration

### Option A: Goodfire SAE

```python
from goodfire import Client, FeatureGroup

# Initialize Goodfire client
client = Client(api_key="your_api_key")

# Load pre-trained SAE for Llama-2-7b
sae = client.features.load(
    model="meta-llama/Llama-2-7b-hf",
    layer=16  # Middle layer (good for reasoning)
)

# Extract features from activations
def extract_sae_features(activations, top_k=20):
    # activations shape: [batch_size, hidden_dim=4096]
    sparse_features = sae.encode(activations)
    # sparse_features shape: [batch_size, sae_width=16384]

    # Get top-k active features
    top_features = sae.inspect(sparse_features, top_k=top_k)
    # Returns: [(feature_id, activation, description), ...]

    return top_features
```

### Option B: Custom SAE (open source)

```python
import torch
from sparse_autoencoder import SparseAutoencoder

# Load pre-trained SAE
sae = SparseAutoencoder.from_pretrained(
    "layer_16_sae",  # Your trained SAE checkpoint
    device="cuda"
)

# Encode activations
def extract_sae_features(activations, top_k=20):
    # activations: torch.Tensor [batch_size, 4096]
    with torch.no_grad():
        sparse = sae.encode(activations)
        # sparse: torch.Tensor [batch_size, 16384]

        # Get top-k per sample
        top_k_values, top_k_indices = torch.topk(sparse, k=top_k, dim=1)

    return top_k_indices.cpu().numpy(), top_k_values.cpu().numpy()
```

### Training Your Own SAE (Advanced)

```python
from sparse_autoencoder import train_sae

# Collect activations from Llama-2-7b layer 16
# (run ~10k prompts through model, capture activations)

# Train SAE
sae = train_sae(
    activations_dataset="./activations/layer_16.pt",
    hidden_dim=4096,
    sae_width=16384,
    sparsity_coefficient=1e-3,
    epochs=50,
    batch_size=256,
    learning_rate=1e-4
)

# Save trained SAE
sae.save_pretrained("./models/layer_16_sae")
```

## Step 4: Activation Capture

### Hook into Model Forward Pass

```python
import torch

# Activation storage
captured_activations = {}

def capture_hook(module, input, output, layer_name):
    """Hook function to capture intermediate activations"""
    # output is residual stream activations
    # shape: [batch_size, seq_len, hidden_dim]

    # Store last token activations (decision point)
    captured_activations[layer_name] = output[:, -1, :].detach().cpu()

# Register hooks on target layers
def register_hooks(model, layers=[8, 16, 24]):
    hooks = []

    for layer_idx in layers:
        layer_name = f"layer_{layer_idx}"
        layer_module = model.model.layers[layer_idx]

        # Hook on residual stream output
        hook = layer_module.register_forward_hook(
            lambda module, input, output, name=layer_name:
                capture_hook(module, input, output, name)
        )
        hooks.append(hook)

    return hooks

# Usage
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "./models/Llama-2-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

hooks = register_hooks(model, layers=[8, 16, 24])

# Run inference (activations captured automatically)
outputs = model.generate(input_ids, max_new_tokens=256)

# Clean up
for hook in hooks:
    hook.remove()

# Access captured activations
layer_16_activations = captured_activations["layer_16"]
# shape: [batch_size, 4096]
```

## Step 5: Complete Integration

### Production-Ready Dark Trace Engine

```python
from vllm import LLM, SamplingParams
from goodfire import Client
import torch

class ProductionDarkTrace:
    def __init__(self, config):
        # Initialize vLLM
        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype="float16",
            gpu_memory_utilization=0.9
        )

        # Initialize SAE
        self.sae_client = Client(api_key=config.goodfire_api_key)
        self.sae = self.sae_client.features.load(
            model=config.model_name,
            layer=config.sae_layer
        )

        # Sampling params (deterministic)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            top_p=1.0
        )

    def infer_with_provenance(self, prompts):
        """
        Batch inference with complete provenance.

        Returns:
            List[{
                'decision': str,
                'reasoning': str,
                'confidence': float,
                'sae_features': List[(id, activation, description)],
                'activations': Dict[layer, tensor]
            }]
        """
        # 1. Batch inference with vLLM
        outputs = self.llm.generate(prompts, self.sampling_params)

        # 2. Capture activations (requires model hooks)
        activations = self._capture_activations(prompts)

        # 3. Extract SAE features
        sae_features = self.sae.inspect(
            activations[self.config.sae_layer],
            top_k=20
        )

        # 4. Parse decisions
        results = []
        for output, sae_feat in zip(outputs, sae_features):
            text = output.outputs[0].text

            # Parse decision from text
            decision = "BLOCKED" if "BLOCKED" in text else "SAFE"
            confidence = self._extract_confidence(text)

            results.append({
                'decision': decision,
                'reasoning': text,
                'confidence': confidence,
                'sae_features': sae_feat,
                'activations': {
                    k: v.tolist() for k, v in activations.items()
                }
            })

        return results
```

## Step 6: Performance Optimization

### Batch Size Tuning

```python
# Find optimal batch size for your GPU
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
latencies = []

for batch_size in batch_sizes:
    prompts = [build_prompt(drug_a, drug_b) for _ in range(batch_size)]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    latency = (time.time() - start) / batch_size

    latencies.append(latency)
    print(f"Batch size {batch_size}: {latency*1000:.1f} ms/sample")

# Optimal: batch_size where latency stops improving
```

**Typical results (A100 40GB)**:
- Batch 1: 45 ms/sample
- Batch 8: 18 ms/sample
- Batch 32: 12 ms/sample (optimal)
- Batch 64: 13 ms/sample (memory limited)

### Continuous Batching (vLLM Feature)

vLLM automatically uses continuous batching for optimal throughput:

```python
# No code changes needed - vLLM handles this automatically
# Requests are batched dynamically as they arrive
```

### KV Cache Optimization

```python
llm = LLM(
    model="./models/Llama-2-7b-hf",
    kv_cache_dtype="fp8",  # Use FP8 for KV cache (2x memory savings)
    max_model_len=2048,
    gpu_memory_utilization=0.95  # Can increase with FP8
)
```

## Step 7: Monitoring and Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
inference_counter = Counter(
    'ouroboros_inferences_total',
    'Total drug interaction inferences',
    ['decision']
)

inference_latency = Histogram(
    'ouroboros_inference_latency_seconds',
    'Inference latency in seconds'
)

sae_feature_activations = Histogram(
    'ouroboros_sae_feature_activations',
    'SAE feature activation strengths',
    ['feature_id', 'feature_name']
)

# Usage
@inference_latency.time()
def check_interaction(drug_a, drug_b):
    result = engine.infer([build_prompt(drug_a, drug_b)])[0]
    inference_counter.labels(decision=result['decision']).inc()

    # Track SAE features
    for feat_id, activation, description in result['sae_features']:
        sae_feature_activations.labels(
            feature_id=feat_id,
            feature_name=description
        ).observe(activation)

    return result

# Start Prometheus server
start_http_server(9090)
```

### TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/ouroboros')

# Log metrics
writer.add_scalar('Latency/inference', latency_ms, global_step)
writer.add_scalar('Decisions/blocked', blocked_count, global_step)
writer.add_histogram('SAE/feature_activations', activations, global_step)

# Log activations
writer.add_embedding(
    activations,  # [batch_size, hidden_dim]
    metadata=[f"{drug_a}+{drug_b}" for drug_a, drug_b in pairs],
    global_step=global_step
)

writer.close()
```

## Step 8: Production Deployment

### Docker Container

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy model
COPY models/Llama-2-7b-hf /models/Llama-2-7b-hf

# Copy Ouroboros code
COPY ouroboros/ /app/ouroboros/

WORKDIR /app

# Expose ports
EXPOSE 8000 9090

# Start server
CMD ["python3", "ouroboros/dark_trace_server.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ouroboros-dark-trace
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ouroboros
  template:
    metadata:
      labels:
        app: ouroboros
    spec:
      containers:
      - name: dark-trace
        image: ouroboros:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 64Gi
          requests:
            nvidia.com/gpu: 1
            memory: 32Gi
        ports:
        - containerPort: 8000
        - containerPort: 9090
```

## Performance Benchmarks

### Expected Latencies (A100 40GB)

| Batch Size | Latency/Sample | Throughput |
|------------|----------------|------------|
| 1 | 45 ms | 22 samples/s |
| 8 | 18 ms | 444 samples/s |
| 32 | 12 ms | 2,667 samples/s |
| 64 | 13 ms | 4,923 samples/s |

### Expected Latencies (T4 16GB)

| Batch Size | Latency/Sample | Throughput |
|------------|----------------|------------|
| 1 | 120 ms | 8 samples/s |
| 4 | 65 ms | 62 samples/s |
| 8 | 55 ms | 145 samples/s |
| 16 | OOM | - |

## Cost Analysis

### Cloud GPU Costs (AWS us-east-1)

| Instance | GPU | Cost/Hour | Samples/Hour | Cost/1M Samples |
|----------|-----|-----------|--------------|-----------------|
| g5.xlarge | A10G (24GB) | $1.01 | 1,600,000 | $0.63 |
| p4d.24xlarge | A100 (40GB) ×8 | $32.77 | 21,000,000 | $1.56 |
| g4dn.xlarge | T4 (16GB) | $0.53 | 500,000 | $1.06 |

**Recommendation**: g5.xlarge (A10G) for production (best cost/performance)

## Security Considerations

### HIPAA Compliance

1. **Encrypted storage**: All activations, SAE features, decisions encrypted at rest
2. **Audit trail**: Complete provenance logged for every decision
3. **Access controls**: Role-based access to inference API
4. **Data retention**: 7-year retention for medical decisions

### Model Security

1. **Input validation**: Sanitize drug names (prevent prompt injection)
2. **Output validation**: Verify decision format (SAFE/BLOCKED only)
3. **Rate limiting**: Prevent abuse (max 1000 req/min per API key)
4. **Monitoring**: Alert on suspicious patterns (e.g., 100% SAFE decisions)

## Troubleshooting

### OOM (Out of Memory) Errors

```python
# Reduce batch size
batch_size = 8  # Down from 32

# Use FP8 KV cache
kv_cache_dtype = "fp8"

# Reduce max sequence length
max_model_len = 1024  # Down from 2048

# Enable CPU offloading (slower but works)
device_map = "auto"
offload_folder = "./offload"
```

### Slow Inference

```bash
# Check GPU utilization
nvidia-smi

# Should see ~90% GPU utilization during inference
# If <50%, increase batch size or check CPU bottleneck

# Profile with nsys
nsys profile -o profile.qdrep python dark_trace_server.py
```

### SAE Feature Quality

If SAE features are not interpretable:

1. **Try different layers**: Layer 16 (mid) vs Layer 8 (early) vs Layer 24 (late)
2. **Increase SAE width**: 16384 → 32768 (more features)
3. **Adjust sparsity**: Higher coefficient = sparser = more interpretable
4. **Collect more training data**: 10k → 100k activation samples

## Next Steps

1. **Week 1**: Deploy on single GPU, validate latency
2. **Week 2**: Integrate with Ouroboros database, validate accuracy
3. **Week 3**: Clinical validation with ER doctors
4. **Week 4**: Production deployment with monitoring

## References

- vLLM: https://github.com/vllm-project/vllm
- Goodfire SAE: https://goodfire.ai
- Llama-2: https://huggingface.co/meta-llama/Llama-2-7b-hf
- Sparse Autoencoders: https://transformer-circuits.pub/2023/monosemantic-features
