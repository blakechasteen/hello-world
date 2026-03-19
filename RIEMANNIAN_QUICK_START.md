# Riemannian Embeddings Quick Start

**Priority 2 - Mathematical Moonshot**

---

## 🚀 Quick Start (30 seconds)

```python
from HoloLoom.embedding.riemannian_matryoshka import create_riemannian_embedder

# Enable Riemannian geometry
embedder = create_riemannian_embedder(use_riemannian=True)

# Encode hierarchical concepts
texts = ["mammal", "dog", "beagle"]
embs = embedder.encode(texts)

# Compute geodesic distance (not Euclidean!)
dist = embedder.distance(embs[0], embs[1])
print(f"Geodesic distance (mammal → dog): {dist:.4f}")
```

---

## 📖 What is This?

**Riemannian embeddings** use curved manifold geometry instead of flat Euclidean space.

### Why it matters:
- **Hierarchies live in hyperbolic space** (trees, taxonomies)
- **Clusters live on spheres** (normalized embeddings)
- **Euclidean distance is wrong** for semantic similarity

### Product Manifold: H×S×E

```
768D Embedding
    ↓
Split:
  ├─ Hyperbolic (256D) → Hierarchies (K = -1)
  ├─ Spherical (256D)  → Clusters (K = +1)
  └─ Euclidean (256D)  → Linear (K = 0)
    ↓
Geodesic Distance: d² = d_H² + d_S² + d_E²
```

---

## 🔧 Configuration

### Enable in Config

```python
from HoloLoom.config import Config

config = Config.fused()
config.use_riemannian = True
config.riemannian_hyperbolic_dim = 256
config.riemannian_spherical_dim = 256
config.riemannian_euclidean_dim = 256
```

### Custom Curvatures

```python
config.riemannian_hyperbolic_curvature = -1.0  # Negative (hierarchies)
config.riemannian_spherical_curvature = 1.0    # Positive (clusters)
```

---

## 📊 Usage Examples

### 1. Hierarchical Distance

```python
from HoloLoom.embedding.riemannian_matryoshka import RiemannianMatryoshka

# Create embedder
embedder = RiemannianMatryoshka(
    use_riemannian=True,
    hyperbolic_dim=128,
    spherical_dim=128,
    euclidean_dim=128
)

# Test hierarchy
concepts = ["animal", "mammal", "dog", "beagle"]
embs = embedder.encode(concepts)

# Distances
for i in range(len(concepts) - 1):
    dist = embedder.distance(embs[i], embs[i+1])
    print(f"{concepts[i]} → {concepts[i+1]}: {dist:.4f}")
```

### 2. Pairwise Distance Matrix

```python
# Compute all pairwise distances
concepts = ["mammal", "dog", "cat", "bird"]
embs = embedder.encode(concepts)
dist_matrix = embedder.pairwise_distances(embs)

# Print matrix
for i, c1 in enumerate(concepts):
    for j, c2 in enumerate(concepts):
        print(f"{c1} ↔ {c2}: {dist_matrix[i,j]:.4f}")
```

### 3. Geodesic Interpolation

```python
# Get tangent vector from A to B
start = embs[0]
end = embs[1]
tangent = embedder.log_map(start, end)

# Move halfway along geodesic
midpoint = embedder.exp_map(start, 0.5 * tangent)
```

---

## 🎯 Run the Demo

```bash
PYTHONPATH=. python demos/demo_riemannian_embeddings.py
```

**Demos included**:
1. Hierarchical concepts (Euclidean vs Riemannian)
2. Pairwise distance matrix
3. Geodesic interpolation
4. Performance comparison

---

## ⚡ Performance

| Operation | Euclidean | Riemannian | Overhead |
|-----------|-----------|------------|----------|
| Single distance | ~0.1 µs | ~2-5 µs | **20-50x** |
| Pairwise (n=100) | ~1 ms | ~20-50 ms | **20-50x** |

### When to use:

**✅ Use Riemannian**:
- Hierarchical concepts (taxonomies)
- Semantic precision required
- Research and analysis

**✅ Use Euclidean**:
- Large-scale retrieval (>10k items)
- Real-time (<100ms latency)
- Production systems

---

## 🔄 Backward Compatibility

```python
# Disable Riemannian → same as MatryoshkaEmbeddings
embedder = RiemannianMatryoshka(use_riemannian=False)

# Or explicitly:
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
embedder = MatryoshkaEmbeddings()  # Standard Euclidean
```

---

## 🧪 Testing

```python
# Test basic functionality
from HoloLoom.embedding.riemannian_matryoshka import RiemannianMatryoshka

embedder = RiemannianMatryoshka(use_riemannian=True)
embs = embedder.encode(["test1", "test2"])
dist = embedder.distance(embs[0], embs[1])
assert dist > 0  # Should be positive
```

---

## 📚 Files

| File | Purpose | Lines |
|------|---------|-------|
| `HoloLoom/embedding/riemannian_matryoshka.py` | Main implementation | 320 |
| `HoloLoom/semantic_calculus/dimensions.py` | Manifold projection | +35 |
| `HoloLoom/config.py` | Configuration | +7 |
| `demos/demo_riemannian_embeddings.py` | Demo script | 440 |
| `PRIORITY_2_RIEMANNIAN_COMPLETE.md` | Full documentation | - |

---

## 🎓 Theory

### Hyperbolic Space (Poincaré Ball)
- **Curvature K < 0**: Negative curvature
- **Geometry**: Distances grow exponentially
- **Use case**: Trees, hierarchies, taxonomies

### Spherical Space
- **Curvature K > 0**: Positive curvature
- **Geometry**: Great circles, angular distance
- **Use case**: Clusters, normalized embeddings

### Euclidean Space
- **Curvature K = 0**: Flat space
- **Geometry**: Standard L2 distance
- **Use case**: Linear features, residuals

---

## 🔗 Next Steps

1. **Run demo**: See Euclidean vs Riemannian comparison
2. **Enable in config**: Set `use_riemannian=True`
3. **Test on your data**: Try hierarchical concepts
4. **Tune dimensions**: Adjust H/S/E split for your domain
5. **Benchmark**: Measure performance impact

---

## 📖 References

- **Implementation**: `HoloLoom/warp/riemannian_geometry.py`
- **Theory**: Poincaré Embeddings (Nickel & Kiela, 2017)
- **Mixed Curvature**: Product Spaces (Gu et al., 2019)
- **Full Docs**: `PRIORITY_2_RIEMANNIAN_COMPLETE.md`

---

## ❓ FAQ

**Q: When should I use Riemannian embeddings?**
A: For hierarchical concepts (taxonomies, ontologies) where Euclidean distance underestimates similarity.

**Q: What's the performance cost?**
A: 20-50x slower than Euclidean. Use for offline analysis, not real-time retrieval.

**Q: Can I disable it?**
A: Yes! Set `use_riemannian=False` for standard Euclidean embeddings.

**Q: What if riemannian_geometry is unavailable?**
A: Graceful fallback to Euclidean with a warning.

**Q: How do I tune the dimension splits?**
A: Start with equal splits (256/256/256). Increase hyperbolic_dim for more hierarchy, spherical_dim for more clustering.

---

**Ready to start?** Run the demo and see the difference! 🚀

```bash
PYTHONPATH=. python demos/demo_riemannian_embeddings.py
```
