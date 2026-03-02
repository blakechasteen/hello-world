# Matrix Multiplication Optimization

Where matmul operations occur in HoloLoom and how to optimize them.

## Where Matmuls Happen

| Component | Operation | Size | Frequency |
|-----------|-----------|------|-----------|
| Embedding | `embed @ projection` | [B, 384] x [384, D] | Every query |
| Policy | `features @ weights` | [B, F] x [F, A] | Every decision |
| WarpSpace | `tensor @ manifold` | [B, D] x [D, D] | Per weave cycle |
| Convergence | `probs @ values` | [B, A] x [A, 1] | Per tool selection |

## Zero-Copy Optimization

The key optimization: avoid creating intermediate tensors.

```python
# Before (allocates intermediate)
result = torch.matmul(a, b)
result = torch.add(result, bias)

# After (zero-copy, 37x speedup for small matrices)
result = torch.addmm(bias, a, b)
```

## Performance Characteristics

| Matrix Size | CPU (ms) | GPU (ms) | Speedup |
|-------------|----------|----------|---------|
| [1, 96] x [96, 96] | 0.05 | 0.01 | 5x |
| [32, 384] x [384, 384] | 0.8 | 0.05 | 16x |
| [128, 384] x [384, 768] | 3.2 | 0.1 | 32x |

## Optimization Strategies

### Batch Operations

```python
# Instead of per-item matmul
for item in batch:
    result = item @ weights

# Batch all at once
results = batch @ weights  # Single kernel launch
```

### Mixed Precision

```python
with torch.cuda.amp.autocast():
    result = features @ weights  # FP16 matmul, ~2x throughput
```

### Fused Kernels

```python
# torch.compile for automatic kernel fusion (PyTorch 2.0+)
@torch.compile
def fused_forward(x, w1, w2, bias):
    return torch.addmm(bias, torch.relu(x @ w1), w2)
```

### Sparse Operations

For the Thompson Sampling policy where most tool probabilities are near-zero:

```python
# Only compute for top-k tools
topk_indices = torch.topk(probs, k=5).indices
result = features[:, topk_indices] @ weights[topk_indices]
```

## Profiling

```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    result = model(input)
print(prof.key_averages().table(sort_by="cpu_time_total"))
```

## GPU Recommendations

- Batch size >= 32 to amortize kernel launch overhead
- Use `torch.compile` for automatic optimization on PyTorch 2.0+
- For 11GB VRAM (GTX 1080 Ti), max batch ~512 at 384d embeddings
